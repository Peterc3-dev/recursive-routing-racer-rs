//! Dynamic transformer forward pass — supports Phi-3/4, Qwen2, LLaMA architectures.
//! Native K-quant shaders (Q4_K/Q5_K/Q6_K) with GPU-native repack.

use crate::gpu::{ComputeEngine, GpuBuffer, DescSetHandle};
use crate::model::gguf::GGUFModel;
use std::time::Instant;

/// Dynamic model configuration — read from GGUF metadata at load time.
#[derive(Clone)]
pub struct ModelConfig {
    pub arch: String,          // "phi3", "qwen2", "llama", etc.
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub rope_dim: usize,       // partial rotation dims (= head_dim for most models)
    pub rope_theta: f32,
    pub rope_attn_factor: f32, // 1.0 for standard RoPE, >1 for LongRoPE
    pub has_qkv_bias: bool,
    pub has_rope_factors: bool,
    pub separate_qkv: bool,    // true for Qwen2 (separate Q/K/V tensors)
    pub separate_gate_up: bool, // true for Qwen2 (separate gate + up tensors)
    pub has_output_weight: bool, // true if output.weight exists (not tied to embedding)
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GGUFModel) -> Self {
        // Detect architecture
        let arch = gguf.metadata.get("general.architecture")
            .and_then(|v| v.as_str()).unwrap_or("llama").to_string();
        let prefix = arch.clone();

        let get_u32 = |key: &str| -> u32 {
            gguf.metadata.get(&format!("{}.{}", prefix, key))
                .and_then(|v| v.as_u64()).unwrap_or(0) as u32
        };
        let get_f32 = |key: &str| -> f32 {
            gguf.metadata.get(&format!("{}.{}", prefix, key))
                .and_then(|v| v.as_f32()).unwrap_or(0.0)
        };

        let d_model = get_u32("embedding_length") as usize;
        let n_heads = get_u32("attention.head_count") as usize;
        let n_kv_heads = get_u32("attention.head_count_kv") as usize;
        let head_dim = if n_heads > 0 { d_model / n_heads } else { 128 };
        let ffn_dim = get_u32("feed_forward_length") as usize;
        let n_layers = get_u32("block_count") as usize;
        let rms_eps = get_f32("attention.layer_norm_rms_epsilon");
        let rms_eps = if rms_eps > 0.0 { rms_eps } else { 1e-5 };
        let rope_dim = {
            let rd = get_u32("rope.dimension_count") as usize;
            if rd > 0 { rd } else { head_dim }  // default: full rotation
        };
        let rope_theta = {
            let rt = get_f32("rope.freq_base");
            if rt > 0.0 { rt } else { 10000.0 }
        };
        let rope_attn_factor = {
            let af = get_f32("rope.scaling.attn_factor");
            if af > 0.0 { af } else { 1.0 }  // 1.0 = standard RoPE, no scaling
        };

        // Detect tensor patterns
        let has_qkv_combined = gguf.tensor_infos.iter().any(|t| t.name == "blk.0.attn_qkv.weight");
        let has_separate_qkv = gguf.tensor_infos.iter().any(|t| t.name == "blk.0.attn_q.weight");
        let has_gate = gguf.tensor_infos.iter().any(|t| t.name == "blk.0.ffn_gate.weight");
        let has_qkv_bias = gguf.tensor_infos.iter().any(|t| t.name == "blk.0.attn_q.bias");
        let has_rope_factors = gguf.tensor_infos.iter().any(|t| t.name.contains("rope_factors"));
        let has_output_weight = gguf.tensor_infos.iter().any(|t| t.name == "output.weight");

        let vocab_size = gguf.tensor_infos.iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.shape[1] as usize).unwrap_or(0);

        let qkv_dim = n_heads * head_dim + 2 * n_kv_heads * head_dim;
        let gate_up_dim = 2 * ffn_dim;

        eprintln!("[config] arch={} d_model={} n_heads={} n_kv_heads={} head_dim={} ffn_dim={} \
            n_layers={} vocab={} rope_dim={} rope_theta={:.0} attn_factor={:.3}",
            arch, d_model, n_heads, n_kv_heads, head_dim, ffn_dim,
            n_layers, vocab_size, rope_dim, rope_theta, rope_attn_factor);
        eprintln!("[config] separate_qkv={} separate_gate_up={} qkv_bias={} rope_factors={} output_weight={}",
            has_separate_qkv, has_gate, has_qkv_bias, has_rope_factors, has_output_weight);

        ModelConfig {
            arch, d_model, n_heads, n_kv_heads, head_dim, ffn_dim, n_layers, vocab_size,
            rms_eps, rope_dim, rope_theta, rope_attn_factor,
            has_qkv_bias, has_rope_factors,
            separate_qkv: has_separate_qkv && !has_qkv_combined,
            separate_gate_up: has_gate,
            has_output_weight,
        }
    }

    pub fn qkv_dim(&self) -> usize { self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim }
    pub fn gate_up_dim(&self) -> usize { 2 * self.ffn_dim }
}

// Legacy constants — used by code not yet migrated to ModelConfig
const D_MODEL: usize = 3072;
const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FFN_DIM: usize = 8192;
const N_LAYERS: usize = 32;
const VOCAB_SIZE: usize = 200064;
const RMS_EPS: f32 = 1e-5;
const QKV_DIM: usize = N_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM;
const GATE_UP_DIM: usize = 2 * FFN_DIM;
const ROPE_DIM: usize = 96;
const ROPE_THETA: f32 = 10000.0;
const ROPE_ATTN_FACTOR: f32 = 1.190238;

/// Quant type determines which shader + block size to use
#[derive(Clone, Copy)]
enum QType { Q4K, Q5K, Q6K, GPUQ4, GPUQ5, GPUQ6, F32 }

impl QType {
    fn shader_name(&self) -> &str {
        match self {
            QType::Q4K => "matmul_q4k", QType::Q5K => "matmul_q5k", QType::Q6K => "matmul_q6k",
            QType::GPUQ4 => "matmul_gpuq4", QType::GPUQ5 => "matmul_gpuq5", QType::GPUQ6 => "matmul_gpuq6",
            QType::F32 => "matmul_tiled",
        }
    }
    fn batch_shader_name(&self) -> &str {
        match self {
            QType::Q4K | QType::GPUQ4 => "matmul_gpuq4_batch",
            QType::Q5K | QType::GPUQ5 => "matmul_gpuq5_batch",
            QType::Q6K | QType::GPUQ6 => "matmul_gpuq6_batch",
            QType::F32 => "matmul_tiled",  // TODO: F32 batch shader
        }
    }
    fn fused_shader_name(&self) -> &str {
        match self {
            QType::Q4K | QType::GPUQ4 => "fused_norm_matmul_q4k",
            QType::Q5K | QType::GPUQ5 => "fused_norm_matmul_q5k",
            QType::Q6K | QType::GPUQ6 => "fused_norm_matmul_q6k",
            QType::F32 => "matmul_tiled",
        }
    }
    fn block_bytes(&self) -> usize {
        match self {
            QType::Q4K => 144, QType::Q5K => 176, QType::Q6K => 210,
            QType::GPUQ4 => 192, QType::GPUQ5 | QType::GPUQ6 => 320,
            QType::F32 => 1024, // 256 elements * 4 bytes (not block-based)
        }
    }
    fn workgroups(&self, n: u32) -> u32 { n }
    fn from_gguf(typ: u32) -> Self {
        match typ { 12 => QType::Q4K, 13 => QType::Q5K, 14 => QType::Q6K, _ => QType::F32 }
    }
}

/// Per-layer weights — raw GGUF K-quant bytes on GPU (GPU-native repacked)
pub struct LayerWeights {
    pub attn_norm_buf: GpuBuffer,
    pub ffn_norm_buf: GpuBuffer,
    // QKV: either combined (Phi-4) or separate Q+K+V (Qwen2)
    // For separate: attn_qkv_buf holds Q, attn_k_buf/attn_v_buf hold K/V
    pub attn_qkv_buf: GpuBuffer,     pub attn_qkv_type: QType,     pub attn_qkv_k: u32,   pub attn_qkv_n: u32,
    pub attn_k_buf: Option<(GpuBuffer, QType, u32, u32)>,  // K weight (separate QKV only)
    pub attn_v_buf: Option<(GpuBuffer, QType, u32, u32)>,  // V weight (separate QKV only)
    pub attn_output_buf: GpuBuffer,   pub attn_output_type: QType,  pub attn_output_k: u32, pub attn_output_n: u32,
    // FFN: either combined gate+up (Phi-4) or separate gate + up (Qwen2)
    pub ffn_up_buf: GpuBuffer,        pub ffn_up_type: QType,       pub ffn_up_k: u32,      pub ffn_up_n: u32,
    pub ffn_gate_buf: Option<(GpuBuffer, QType, u32, u32)>,  // gate weight (separate only)
    pub ffn_down_buf: GpuBuffer,      pub ffn_down_type: QType,     pub ffn_down_k: u32,    pub ffn_down_n: u32,
    pub qkv_bias: Option<Vec<f32>>,
    // Persistent descriptor set handles
    pub ds_norm_qkv: (DescSetHandle, DescSetHandle),
    pub ds_out_proj: DescSetHandle,
    pub ds_norm_ffnup: (DescSetHandle, DescSetHandle),
    pub ds_ffn_down: DescSetHandle,
}

const MAX_SEQ: usize = 2048;

/// Pre-built dispatch entry for the mega-batch (zero allocation in hot path).
pub struct PrebuiltDispatch {
    pub pipeline_name: String,
    pub desc_set: DescSetHandle,
    pub workgroups: [u32; 3],
    pub needs_barrier: bool,  // false for parallel logit chunks
}

pub struct Phi4Model {
    pub config: ModelConfig,
    pub embedding: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub buf_a: GpuBuffer,
    pub buf_c: GpuBuffer,
    pub logits_buf: GpuBuffer,
    pub emb_bufs: Vec<(GpuBuffer, usize)>,
    pub rope_factors: Vec<f32>,
    pub attn_q_buf: GpuBuffer,
    pub attn_out_buf: GpuBuffer,
    pub kv_bufs: Vec<(GpuBuffer, GpuBuffer)>,
    pub logit_out_bufs: Vec<GpuBuffer>,
    // GPU-resident pipeline buffers
    pub gpu_hidden: GpuBuffer,
    pub gpu_normed: GpuBuffer,
    pub gpu_qkv: GpuBuffer,
    pub gpu_gate_up: GpuBuffer,
    pub gpu_ffn_mid: GpuBuffer,
    pub gpu_tmp: GpuBuffer,
    pub gpu_rope_factors: GpuBuffer,
    // Pre-built dispatch table (12 per layer × 32 layers + logit chunks + argmax)
    pub mega_dispatches: Vec<PrebuiltDispatch>,
    // GPU-side logit projection + argmax buffers
    pub gpu_output_norm_w: GpuBuffer,  // output norm weights on GPU
    pub gpu_logits: GpuBuffer,         // [VOCAB_SIZE] full logit buffer on GPU
    pub gpu_argmax_scratch: GpuBuffer, // scratch for argmax workgroup results
    pub gpu_argmax_result: GpuBuffer,  // [1 uint32] — the winning token ID
}

pub struct KVCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            k: (0..N_LAYERS).map(|_| Vec::with_capacity(512 * N_KV_HEADS * HEAD_DIM)).collect(),
            v: (0..N_LAYERS).map(|_| Vec::with_capacity(512 * N_KV_HEADS * HEAD_DIM)).collect(),
            len: 0,
        }
    }
}

fn apply_rope_cfg(x: &mut [f32], pos: usize, n_heads: usize, cfg: &ModelConfig, rope_factors: &[f32]) {
    for h in 0..n_heads {
        let base = h * cfg.head_dim;
        for i in 0..(cfg.rope_dim / 2) {
            let base_freq = 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.rope_dim as f32);
            let factor = if i < rope_factors.len() { rope_factors[i] } else { 1.0 };
            let freq = base_freq / factor;
            let angle = pos as f32 * freq;
            let (sin_t, cos_t) = angle.sin_cos();
            let cos_scaled = cos_t * cfg.rope_attn_factor;
            let sin_scaled = sin_t * cfg.rope_attn_factor;
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0 * cos_scaled - x1 * sin_scaled;
            x[base + 2 * i + 1] = x0 * sin_scaled + x1 * cos_scaled;
        }
    }
}

// Compat wrapper using legacy constants
fn apply_rope(x: &mut [f32], pos: usize, n_heads: usize, rope_factors: &[f32]) {
    let cfg = ModelConfig {
        arch: "phi3".into(), d_model: D_MODEL, n_heads: N_HEADS, n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM, ffn_dim: FFN_DIM, n_layers: N_LAYERS, vocab_size: VOCAB_SIZE,
        rms_eps: RMS_EPS, rope_dim: ROPE_DIM, rope_theta: ROPE_THETA,
        rope_attn_factor: ROPE_ATTN_FACTOR, has_qkv_bias: false, has_rope_factors: true,
        separate_qkv: false, separate_gate_up: false, has_output_weight: false,
    };
    apply_rope_cfg(x, pos, n_heads, &cfg, rope_factors);
}

// ============================================================================
// GPU-native Q4_K repacker
// ============================================================================

/// Repack Q4_K blocks into GPU-friendly layout.
/// Input: raw Q4_K bytes (144 bytes per 256 elements, N rows × K/256 blocks per row)
/// Output: repacked bytes (192 bytes per 256 elements)
///   Per block: f32[8] scales + f32[8] mins + u32[32] nibbles
///   Element e → nibble at word[e/8] bits [(e%8)*4 +: 4]
fn repack_q4k(raw: &[u8], n_elements: usize) -> Vec<u8> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0u8; n_blocks * 192];

    for bi in 0..n_blocks {
        let b = &raw[bi * 144..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let sc_bytes = &b[4..16];
        let qs = &b[16..144];

        let ob = &mut out[bi * 192..];

        // Pre-decode scales and mins to f32
        for sub in 0..8usize {
            let (sc_val, mn_val) = if sub < 4 {
                (sc_bytes[sub] as u32 & 0x3F, sc_bytes[sub + 4] as u32 & 0x3F)
            } else {
                let i = sub - 4;
                (
                    (sc_bytes[8 + i] as u32 & 0x0F) | ((sc_bytes[i] as u32 >> 6) << 4),
                    (sc_bytes[8 + i] as u32 >> 4) | ((sc_bytes[4 + i] as u32 >> 6) << 4),
                )
            };
            let sc_f32 = d * sc_val as f32;
            let mn_f32 = dmin * mn_val as f32;
            ob[sub * 4..sub * 4 + 4].copy_from_slice(&sc_f32.to_le_bytes());
            ob[32 + sub * 4..32 + sub * 4 + 4].copy_from_slice(&mn_f32.to_le_bytes());
        }

        // Repack nibbles: element e → word[e/8] bits [(e%8)*4 +: 4]
        // Q4_K layout: sub-block s (0..7), position j (0..31)
        // qs byte at pair*32+j contains low nibble (sub even) and high nibble (sub odd)
        let nibble_base = 64; // offset in output block for nibble data
        for e in 0..256usize {
            let sub = e / 32;
            let j = e % 32;
            let pair = sub / 2;
            let q_byte = qs[pair * 32 + j];
            let nibble = if sub % 2 == 0 { q_byte & 0x0F } else { (q_byte >> 4) & 0x0F };

            let word_idx = e / 8;
            let bit_pos = (e % 8) * 4;
            let off = nibble_base + word_idx * 4;
            let mut word = u32::from_le_bytes([ob[off], ob[off + 1], ob[off + 2], ob[off + 3]]);
            word |= (nibble as u32) << bit_pos;
            ob[off..off + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    out
}

// ============================================================================
// GPU-native Q5_K repacker
// ============================================================================

/// Repack Q5_K blocks into GPU-friendly layout.
/// Input: raw Q5_K bytes (176 bytes per 256 elements)
/// Output: 320 bytes per block: f32[8] scales + f32[8] mins + u8[256] quants (as u32[64])
///   Quant value = 5-bit combined (low nibble | qh_bit << 4), range 0-31
///   Element e at word[e/4] byte [e%4]
fn repack_q5k(raw: &[u8], n_elements: usize) -> Vec<u8> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0u8; n_blocks * 320];

    for bi in 0..n_blocks {
        let b = &raw[bi * 176..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let sc_bytes = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];

        let ob = &mut out[bi * 320..];

        // Pre-decode scales and mins (same 6-bit extraction as Q4_K)
        for sub in 0..8usize {
            let (sc_val, mn_val) = if sub < 4 {
                (sc_bytes[sub] as u32 & 0x3F, sc_bytes[sub + 4] as u32 & 0x3F)
            } else {
                let i = sub - 4;
                (
                    (sc_bytes[8 + i] as u32 & 0x0F) | ((sc_bytes[i] as u32 >> 6) << 4),
                    (sc_bytes[8 + i] as u32 >> 4) | ((sc_bytes[4 + i] as u32 >> 6) << 4),
                )
            };
            let sc_f32 = d * sc_val as f32;
            let mn_f32 = dmin * mn_val as f32;
            ob[sub * 4..sub * 4 + 4].copy_from_slice(&sc_f32.to_le_bytes());
            ob[32 + sub * 4..32 + sub * 4 + 4].copy_from_slice(&mn_f32.to_le_bytes());
        }

        // Pack combined 5-bit quants as u8 per element
        let q_base = 64; // offset for quant data
        for e in 0..256usize {
            let sub = e / 32;
            let j = e % 32;
            let pair = sub / 2;
            let q_byte = qs[pair * 32 + j];
            let qi_lo = if sub % 2 == 0 { q_byte & 0x0F } else { (q_byte >> 4) & 0x0F };
            let qh_bit = (qh[j] >> sub) & 1;
            let qi = qi_lo | (qh_bit << 4); // 5-bit value, 0-31
            ob[q_base + e] = qi;
        }
    }
    out
}

// ============================================================================
// GPU-native Q6_K repacker
// ============================================================================

/// Repack Q6_K blocks into GPU-friendly layout.
/// Input: raw Q6_K bytes (210 bytes per 256 elements)
/// Output: 320 bytes per block: f32[16] scales + u8[256] quants (as u32[64])
///   Scale: pre-decoded d * int8_scale for each of 16 sub-blocks
///   Quant value = 6-bit combined (ql_nibble | qh_2bits << 4), stored as u8 (0-63, add 32 at use)
///   Element e at byte offset [64 + e]
fn repack_q6k(raw: &[u8], n_elements: usize) -> Vec<u8> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0u8; n_blocks * 320];

    for bi in 0..n_blocks {
        let b = &raw[bi * 210..];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let scales_raw = &b[192..208];
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));

        let ob = &mut out[bi * 320..];

        // Pre-decode 16 scales to f32: d * int8(scale)
        for s in 0..16usize {
            let sc_i8 = scales_raw[s] as i8;
            let sc_f32 = d * sc_i8 as f32;
            ob[s * 4..s * 4 + 4].copy_from_slice(&sc_f32.to_le_bytes());
        }

        // Pack combined 6-bit quants as u8 per element
        let q_base = 64; // 16 * 4 = 64 bytes for scales
        for e in 0..256usize {
            let g = e / 128;
            let pos_in_group = e % 128;
            let quad = pos_in_group / 32;
            let l = pos_in_group % 32;

            let ql_idx = g * 64 + l + ((quad & 1) * 32);
            let ql_byte = ql[ql_idx];
            let ql_val = if quad < 2 { ql_byte & 0x0F } else { (ql_byte >> 4) & 0x0F };

            let qh_byte = qh[g * 32 + l];
            let qh_val = (qh_byte >> (quad * 2)) & 0x03;

            let qi = ql_val | (qh_val << 4); // 6-bit unsigned, 0-63
            ob[q_base + e] = qi;
        }
    }
    out
}

// ============================================================================
// GGUF dequant (CPU, for embedding + norms only)
// ============================================================================

fn f16_to_f32(bits: u16) -> f32 { half::f16::from_bits(bits).to_f32() }

fn dequant_q6_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    // Q6_K: llama.cpp interleaved layout.
    // Per block (256 elems): [128B ql][64B qh][16B scales][2B d]
    // Two groups of 128. Each group has 4 quads of 32 elements.
    let n_blocks = n_elements / 256;
    let mut out = vec![0.0f32; n_elements];
    for bi in 0..n_blocks {
        let b = &data[bi * 210..];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let scales = &b[192..208];
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        let base = bi * 256;

        for g in 0..2usize {
            for l in 0..32usize {
                let ql_lo = ql[g * 64 + l];
                let ql_hi = ql[g * 64 + l + 32];
                let qh_byte = qh[g * 32 + l];
                let is = g * 8 + l / 16;  // FIX: l/16 selects between two scale halves per quad

                let q1 = ((ql_lo & 0x0F) | (((qh_byte >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql_hi & 0x0F) | (((qh_byte >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql_lo >> 4) | (((qh_byte >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql_hi >> 4) | (((qh_byte >> 6) & 3) << 4)) as i32 - 32;

                let pos = base + g * 128;
                out[pos + l]      = d * (scales[is]     as i8 as f32) * q1 as f32;
                out[pos + l + 32] = d * (scales[is + 2] as i8 as f32) * q2 as f32;
                out[pos + l + 64] = d * (scales[is + 4] as i8 as f32) * q3 as f32;
                out[pos + l + 96] = d * (scales[is + 6] as i8 as f32) * q4 as f32;
            }
        }
    }
    out
}

fn dequant_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0.0f32; n_elements];
    for bi in 0..n_blocks {
        let b = &data[bi * 144..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let sc = &b[4..16];
        let qs = &b[16..144];
        let base = bi * 256;
        for sub in 0..8usize {
            let (sv, mv) = if sub < 4 {
                (sc[sub] as u32 & 0x3F, sc[sub+4] as u32 & 0x3F)
            } else {
                let i = sub - 4;
                ((sc[8+i] as u32 & 0x0F)|((sc[i] as u32>>6)<<4),
                 (sc[8+i] as u32>>4)|((sc[4+i] as u32>>6)<<4))
            };
            let s = d * sv as f32;
            let m = dmin * mv as f32;
            let pair = sub / 2;
            for j in 0..32usize {
                let qb = qs[pair*32+j];
                let qi = if sub%2==0 { qb&0xF } else { (qb>>4)&0xF };
                out[base+sub*32+j] = qi as f32 * s - m;
            }
        }
    }
    out
}

fn dequant_q5_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0.0f32; n_elements];
    for bi in 0..n_blocks {
        let b = &data[bi * 176..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let sc = &b[4..16]; let qh = &b[16..48]; let qs = &b[48..176];
        let base = bi * 256;
        for sub in 0..8usize {
            let (sv, mv) = if sub < 4 {
                (sc[sub] as u32 & 0x3F, sc[sub+4] as u32 & 0x3F)
            } else {
                let i = sub - 4;
                ((sc[8+i] as u32 & 0x0F)|((sc[i] as u32>>6)<<4),
                 (sc[8+i] as u32>>4)|((sc[4+i] as u32>>6)<<4))
            };
            let s = d * sv as f32;
            let m = dmin * mv as f32;
            let pair = sub / 2;
            for j in 0..32usize {
                let qb = qs[pair*32+j];
                let qi_lo = if sub%2==0 { qb&0xF } else { (qb>>4)&0xF };
                let qh_bit = (qh[j] >> sub) & 1;
                let qi = qi_lo as u32 | ((qh_bit as u32) << 4);
                out[base+sub*32+j] = qi as f32 * s - m;
            }
        }
    }
    out
}

fn dequant_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = vec![0.0f32; n_elements];
    for bi in 0..n_blocks {
        let b = &data[bi * 34..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for j in 0..32usize { out[bi*32+j] = d * (b[2+j] as i8 as f32); }
    }
    out
}

fn dequant_q5_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = vec![0.0f32; n_elements];
    for bi in 0..n_blocks {
        let b = &data[bi * 22..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        for j in 0..32usize {
            let qi_lo = if j < 16 { b[6+j] & 0xF } else { (b[6+j-16] >> 4) & 0xF };
            let qh_bit = ((qh >> j) & 1) as u8;
            let qi = (qi_lo | (qh_bit << 4)) as i32 - 16;
            out[bi*32+j] = d * qi as f32;
        }
    }
    out
}

fn dequant_tensor_f32(gguf: &GGUFModel, name: &str) -> Vec<f32> {
    let info = gguf.tensor_infos.iter().find(|t| t.name == name)
        .unwrap_or_else(|| panic!("Tensor not found: {}", name));
    let n_elements: usize = info.shape.iter().product::<u64>() as usize;
    let bytes = gguf.tensor_bytes(name);
    match info.typ {
        0 => unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_elements).to_vec() },
        1 => unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, n_elements).iter().map(|&b| f16_to_f32(b)).collect() },
        6 => dequant_q5_0(bytes, n_elements),
        8 => dequant_q8_0(bytes, n_elements),
        12 => dequant_q4_k(bytes, n_elements),
        13 => dequant_q5_k(bytes, n_elements),
        14 => dequant_q6_k(bytes, n_elements),
        t => panic!("dequant_tensor_f32: unsupported type {} for {}", t, name),
    }
}

// ============================================================================
// CPU ops
// ============================================================================

fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq: f32 = 0.0;
    for &v in x { sum_sq += v * v; }
    let inv_rms = 1.0 / (sum_sq / n as f32 + RMS_EPS).sqrt();
    for i in 0..n { out[i] = weight[i] * x[i] * inv_rms; }
}

fn cpu_silu(x: &mut [f32]) {
    // Phi-4 uses SiLU (Swish), NOT GELU
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn elemwise_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() { out[i] = a[i] + b[i]; }
}

// ============================================================================
// GPU helpers
// ============================================================================

unsafe fn upload(buf: &GpuBuffer, data: &[f32]) {
    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.mapped as *mut u8, data.len() * 4);
}

unsafe fn upload_bytes(buf: &GpuBuffer, data: &[u8]) {
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.mapped as *mut u8, data.len());
}

unsafe fn download(buf: &GpuBuffer, out: &mut [f32]) {
    std::ptr::copy_nonoverlapping(buf.mapped as *const u8, out.as_mut_ptr() as *mut u8, out.len() * 4);
}

/// GPU native K-quant matmul: C[N] = A[K] @ Weight_gguf[N, K]
/// The shader reads raw K-quant blocks and dequantizes on the fly.
/// No transpose needed — shader accesses GGUF row layout directly.
unsafe fn gpu_matmul_kquant(
    engine: &ComputeEngine,
    buf_a: &GpuBuffer, weight_buf: &GpuBuffer, buf_c: &GpuBuffer,
    qtype: QType,
    a: &[f32], c: &mut [f32],
    k: u32, n: u32,
) {
    upload(buf_a, a);
    if matches!(qtype, QType::F32) {
        // F32 path: use matmul_tiled with M=1
        let push: [u32; 3] = [1, k, n];
        let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
        engine.dispatch_by_name("matmul_tiled",
            &[buf_a, weight_buf, buf_c],
            &[k as u64 * 4, k as u64 * n as u64 * 4, n as u64 * 4],
            &pb, [(n + 15) / 16, 1, 1]);
    } else {
        let push: [u32; 2] = [k, n];
        let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
        let blocks_per_row = k as u64 / 256;
        let total_weight_bytes = n as u64 * blocks_per_row * qtype.block_bytes() as u64;
        engine.dispatch_by_name(qtype.shader_name(),
            &[buf_a, weight_buf, buf_c],
            &[k as u64 * 4, total_weight_bytes, n as u64 * 4],
            &pb, [qtype.workgroups(n), 1, 1]);
    }
    download(buf_c, c);
}

/// GPU F32 matmul for logits (embedding is F32)
unsafe fn gpu_matmul_f32(
    engine: &ComputeEngine,
    buf_a: &GpuBuffer, buf_b: &GpuBuffer, buf_c: &GpuBuffer,
    a: &[f32], c: &mut [f32],
    m: u32, k: u32, n: u32,
) {
    upload(buf_a, a);
    let push: [u32; 3] = [m, k, n];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name(
        "matmul_tiled",
        &[buf_a, buf_b, buf_c],
        &[(m as u64) * (k as u64) * 4, (k as u64) * (n as u64) * 4, (m as u64) * (n as u64) * 4],
        &pb,
        [(n + 15) / 16, (m + 15) / 16, 1],
    );
    download(buf_c, c);
}

unsafe fn gpu_rmsnorm(
    engine: &ComputeEngine,
    buf_in: &GpuBuffer, weight_buf: &GpuBuffer, buf_out: &GpuBuffer,
    x: &[f32], out: &mut [f32],
) {
    let d = D_MODEL as u32;
    upload(buf_in, x);
    let push: [u32; 2] = [1, d];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name(
        "rmsnorm",
        &[buf_in, weight_buf, buf_out],
        &[d as u64 * 4, d as u64 * 4, d as u64 * 4],
        &pb,
        [1, 1, 1],
    );
    download(buf_out, out);
}

// ============================================================================
// Model
// ============================================================================

impl Phi4Model {
    pub unsafe fn load_from_gguf(gguf: &GGUFModel, engine: &mut ComputeEngine) -> Self {
        let t0 = Instant::now();
        let config = ModelConfig::from_gguf(gguf);

        eprintln!("[load] Loading embedding...");
        let embedding = dequant_tensor_f32(gguf, "token_embd.weight");
        let output_norm = dequant_tensor_f32(gguf, "output_norm.weight");
        let rope_factors = if config.has_rope_factors {
            dequant_tensor_f32(gguf, "rope_factors_short.weight")
        } else {
            vec![]  // standard RoPE, no factors
        };
        eprintln!("[phi4] Embedding + norms: {:.1}s", t0.elapsed().as_secs_f32());

        // Register pipelines
        engine.get_pipeline("matmul_q4k", 3, 8);
        engine.get_pipeline("matmul_q5k", 3, 8);
        engine.get_pipeline("matmul_q6k", 3, 8);
        engine.get_pipeline("matmul_tiled", 3, 12);
        engine.get_pipeline("rmsnorm", 3, 8);
        engine.get_pipeline("attention", 4, 16);  // 4 push u32s: seq_len, n_kv_heads, heads_per_kv, head_dim
        engine.get_pipeline("fused_norm_matmul_q4k", 4, 8);
        engine.get_pipeline("fused_norm_matmul_q5k", 4, 8);
        engine.get_pipeline("fused_norm_matmul_q6k", 4, 8);
        engine.get_pipeline("matmul_gpuq4", 3, 8);
        engine.get_pipeline("matmul_gpuq5", 3, 8);
        engine.get_pipeline("matmul_gpuq6", 3, 8);
        engine.get_pipeline("matmul_gpuq4_batch", 3, 12);  // K, N, M
        engine.get_pipeline("matmul_gpuq5_batch", 3, 12);
        engine.get_pipeline("matmul_gpuq6_batch", 3, 12);
        engine.get_pipeline("rope", 2, 20);
        engine.get_pipeline("rope_kv_store", 4, 28);  // 7 push u32s          // 5 push u32s = 20 bytes
        engine.get_pipeline("silu_gate", 2, 4);      // 1 push u32
        engine.get_pipeline("residual_add", 3, 4);   // 1 push u32
        engine.get_pipeline("kv_store", 3, 12);      // 3 push u32s

        let mut layers = Vec::with_capacity(config.n_layers);
        let mut total_gpu: u64 = 0;

        // Helper: load a weight tensor, repack to GPU-native format
        fn load_weight_gpu(gguf: &GGUFModel, engine: &ComputeEngine, name: &str, total: &mut u64) -> (GpuBuffer, QType, u32, u32) {
            let info = gguf.tensor_infos.iter().find(|t| t.name == name)
                .unwrap_or_else(|| panic!("Tensor not found: {}", name));
            let gguf_qtype = QType::from_gguf(info.typ);
            let ne0 = info.shape[0] as u32;
            let ne1 = info.shape[1] as u32;
            let raw_bytes = gguf.tensor_bytes(name);
            let n_elements = ne0 as usize * ne1 as usize;
            let (data, qtype) = match gguf_qtype {
                QType::Q4K => (repack_q4k(raw_bytes, n_elements), QType::GPUQ4),
                QType::Q5K => (repack_q5k(raw_bytes, n_elements), QType::GPUQ5),
                QType::Q6K => (repack_q6k(raw_bytes, n_elements), QType::GPUQ6),
                QType::F32 => {
                    // Non-K-quant type — dequant to F32 for matmul_tiled shader
                    // Transpose to [K, N] row-major for matmul_tiled (which expects B[K*N])
                    let f32_data = dequant_tensor_f32(gguf, name);
                    // GGUF stores [ne0=K, ne1=N] row-major. matmul_tiled expects column-major B[K,N].
                    // The data is already in the right layout: N rows of K elements each.
                    let bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (bytes, QType::F32)
                }
                _ => (raw_bytes.to_vec(), gguf_qtype),
            };
            let buf = unsafe { engine.alloc_buffer(data.len() as u64) };
            unsafe { upload_bytes(&buf, &data) };
            *total += data.len() as u64;
            (buf, qtype, ne0, ne1)
        }

        // Helper: try to load a tensor, return None if it doesn't exist
        fn try_load_weight(gguf: &GGUFModel, engine: &ComputeEngine, name: &str, total: &mut u64) -> Option<(GpuBuffer, QType, u32, u32)> {
            if gguf.tensor_infos.iter().any(|t| t.name == name) {
                Some(load_weight_gpu(gguf, engine, name, total))
            } else { None }
        }

        for i in 0..config.n_layers {
            let prefix = format!("blk.{}", i);

            // QKV weights: combined (Phi-4) or separate Q+K+V (Qwen2)
            let (qkv_buf, qkv_t, qkv_k, qkv_n, qkv_bias);
            if config.separate_qkv {
                // Separate Q, K, V — load each, keep Q as the "QKV" for norm→Q matmul
                // K and V stored separately for individual matmul dispatches
                let (q_buf, q_t, q_k, q_n) = load_weight_gpu(gguf, engine, &format!("{}.attn_q.weight", prefix), &mut total_gpu);
                qkv_buf = q_buf; qkv_t = q_t; qkv_k = q_k; qkv_n = q_n;

                // Load biases if present
                qkv_bias = if config.has_qkv_bias {
                    let qb = dequant_tensor_f32(gguf, &format!("{}.attn_q.bias", prefix));
                    let kb = dequant_tensor_f32(gguf, &format!("{}.attn_k.bias", prefix));
                    let vb = dequant_tensor_f32(gguf, &format!("{}.attn_v.bias", prefix));
                    let mut combined = qb; combined.extend_from_slice(&kb); combined.extend_from_slice(&vb);
                    Some(combined)
                } else { None };
            } else {
                // Combined QKV (Phi-4 style)
                let loaded = load_weight_gpu(gguf, engine, &format!("{}.attn_qkv.weight", prefix), &mut total_gpu);
                qkv_buf = loaded.0; qkv_t = loaded.1; qkv_k = loaded.2; qkv_n = loaded.3;
                qkv_bias = None;
            }

            // K and V separate buffers (Qwen2 only)
            let attn_k = if config.separate_qkv {
                Some(load_weight_gpu(gguf, engine, &format!("{}.attn_k.weight", prefix), &mut total_gpu))
            } else { None };
            let attn_v = if config.separate_qkv {
                Some(load_weight_gpu(gguf, engine, &format!("{}.attn_v.weight", prefix), &mut total_gpu))
            } else { None };

            let (out_buf, out_t, out_k, out_n) = load_weight_gpu(gguf, engine, &format!("{}.attn_output.weight", prefix), &mut total_gpu);

            // FFN weights: combined gate+up (Phi-4) or separate (Qwen2)
            let (up_buf, up_t, up_k, up_n);
            let ffn_gate;
            if config.separate_gate_up {
                let gate = load_weight_gpu(gguf, engine, &format!("{}.ffn_gate.weight", prefix), &mut total_gpu);
                let up = load_weight_gpu(gguf, engine, &format!("{}.ffn_up.weight", prefix), &mut total_gpu);
                // Store gate as the "up" (first half of SwiGLU), up as separate
                // The forward pass will dispatch gate and up separately then combine
                up_buf = gate.0; up_t = gate.1; up_k = gate.2; up_n = gate.3;
                ffn_gate = Some((up.0, up.1, up.2, up.3));
            } else {
                let loaded = load_weight_gpu(gguf, engine, &format!("{}.ffn_up.weight", prefix), &mut total_gpu);
                up_buf = loaded.0; up_t = loaded.1; up_k = loaded.2; up_n = loaded.3;
                ffn_gate = None;
            }

            let (down_buf, down_t, down_k, down_n) = load_weight_gpu(gguf, engine, &format!("{}.ffn_down.weight", prefix), &mut total_gpu);

            let attn_norm = dequant_tensor_f32(gguf, &format!("{}.attn_norm.weight", prefix));
            let attn_norm_buf = engine.alloc_buffer(config.d_model as u64 * 4);
            upload(&attn_norm_buf, &attn_norm);
            let ffn_norm = dequant_tensor_f32(gguf, &format!("{}.ffn_norm.weight", prefix));
            let ffn_norm_buf = engine.alloc_buffer(config.d_model as u64 * 4);
            upload(&ffn_norm_buf, &ffn_norm);

            layers.push(LayerWeights {
                attn_norm_buf, ffn_norm_buf,
                attn_qkv_buf: qkv_buf, attn_qkv_type: qkv_t, attn_qkv_k: qkv_k, attn_qkv_n: qkv_n,
                attn_k_buf: attn_k, attn_v_buf: attn_v,
                attn_output_buf: out_buf, attn_output_type: out_t, attn_output_k: out_k, attn_output_n: out_n,
                ffn_up_buf: up_buf, ffn_up_type: up_t, ffn_up_k: up_k, ffn_up_n: up_n,
                ffn_gate_buf: ffn_gate,
                ffn_down_buf: down_buf, ffn_down_type: down_t, ffn_down_k: down_k, ffn_down_n: down_n,
                qkv_bias: qkv_bias,
                ds_norm_qkv: (DescSetHandle(0), DescSetHandle(0)),
                ds_out_proj: DescSetHandle(0),
                ds_norm_ffnup: (DescSetHandle(0), DescSetHandle(0)),
                ds_ffn_down: DescSetHandle(0),
            });

            if i % 8 == 7 || i == config.n_layers - 1 {
                eprintln!("[load] Layer {}: {:.1}s ({:.1}MB GPU)", i, t0.elapsed().as_secs_f32(), total_gpu as f64 / 1e6);
            }
        }

        let gate_up_dim = config.gate_up_dim();
        let buf_a = engine.alloc_buffer((gate_up_dim as u64) * 4);
        let buf_c = engine.alloc_buffer((gate_up_dim as u64) * 4);
        let logits_buf = engine.alloc_buffer(8192 * 4);

        // For logit projection: use output.weight if available, else use embedding (tied weights)
        let output_weight = if config.has_output_weight {
            eprintln!("[load] Loading separate output.weight...");
            dequant_tensor_f32(gguf, "output.weight")
        } else {
            embedding.clone()  // tied weights
        };

        // F32 embedding chunks for logits (need transpose for matmul_tiled)
        let d_model = config.d_model;
        let vocab_size = config.vocab_size;
        let chunk_size: usize = 8192;
        let mut emb_bufs = Vec::new();
        let mut chunk_start = 0usize;
        while chunk_start < vocab_size {
            let chunk_end = (chunk_start + chunk_size).min(vocab_size);
            let n = chunk_end - chunk_start;
            let mut transposed = vec![0.0f32; d_model * n];
            for d in 0..d_model {
                for v in 0..n {
                    transposed[d * n + v] = output_weight[(chunk_start + v) * d_model + d];
                }
            }
            let buf = engine.alloc_buffer((d_model * n) as u64 * 4);
            upload(&buf, &transposed);
            emb_bufs.push((buf, n));
            chunk_start = chunk_end;
        }
        // GPU attention buffers
        let attn_q_buf = engine.alloc_buffer((d_model as u64) * 4);
        let attn_out_buf = engine.alloc_buffer((d_model as u64) * 4);
        let kv_stride = config.n_kv_heads * config.head_dim;
        let kv_size = (MAX_SEQ * kv_stride) as u64 * 2;  // f16
        let mut kv_bufs = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let kb = engine.alloc_buffer(kv_size);
            let vb = engine.alloc_buffer(kv_size);
            kv_bufs.push((kb, vb));
        }
        total_gpu += config.n_layers as u64 * kv_size * 2 + d_model as u64 * 4 * 2;

        // Per-chunk logit output buffers for batched logits
        let mut logit_out_bufs = Vec::with_capacity(emb_bufs.len());
        for (_, n) in &emb_bufs {
            let buf = engine.alloc_buffer((*n as u64) * 4);
            logit_out_bufs.push(buf);
        }

        // Pre-allocate persistent descriptor sets for each layer
        let d4 = d_model as u64 * 4;
        for i in 0..config.n_layers {
            let layer = &layers[i];
            let bpr_qkv = layer.attn_qkv_k as u64 / 256;
            let wb_qkv = layer.attn_qkv_n as u64 * bpr_qkv * layer.attn_qkv_type.block_bytes() as u64;
            let bpr_out = layer.attn_output_k as u64 / 256;
            let wb_out = layer.attn_output_n as u64 * bpr_out * layer.attn_output_type.block_bytes() as u64;
            let bpr_up = layer.ffn_up_k as u64 / 256;
            let wb_up = layer.ffn_up_n as u64 * bpr_up * layer.ffn_up_type.block_bytes() as u64;
            let bpr_dn = layer.ffn_down_k as u64 / 256;
            let wb_dn = layer.ffn_down_n as u64 * bpr_dn * layer.ffn_down_type.block_bytes() as u64;

            // Batch 1: rmsnorm(buf_a→buf_c) then QKV matmul(buf_c→buf_a)
            let ds_norm1 = engine.pre_allocate_desc_set("rmsnorm",
                &[&buf_a, &layer.attn_norm_buf, &buf_c], &[d4, d4, d4]);
            let ds_qkv = engine.pre_allocate_desc_set(layer.attn_qkv_type.shader_name(),
                &[&buf_c, &layer.attn_qkv_buf, &buf_a],
                &[layer.attn_qkv_k as u64 * 4, wb_qkv, layer.attn_qkv_n as u64 * 4]);

            // Batch 2 (output proj): attn_out_buf → buf_c
            let ds_out = engine.pre_allocate_desc_set(layer.attn_output_type.shader_name(),
                &[&attn_out_buf, &layer.attn_output_buf, &buf_c],
                &[layer.attn_output_k as u64 * 4, wb_out, layer.attn_output_n as u64 * 4]);

            // Batch 3: rmsnorm(buf_a→buf_c) then FFN up(buf_c→buf_a)
            let ds_norm2 = engine.pre_allocate_desc_set("rmsnorm",
                &[&buf_a, &layer.ffn_norm_buf, &buf_c], &[d4, d4, d4]);
            let ds_up = engine.pre_allocate_desc_set(layer.ffn_up_type.shader_name(),
                &[&buf_c, &layer.ffn_up_buf, &buf_a],
                &[layer.ffn_up_k as u64 * 4, wb_up, layer.ffn_up_n as u64 * 4]);

            // FFN down: buf_a → buf_c
            let ds_dn = engine.pre_allocate_desc_set(layer.ffn_down_type.shader_name(),
                &[&buf_a, &layer.ffn_down_buf, &buf_c],
                &[layer.ffn_down_k as u64 * 4, wb_dn, layer.ffn_down_n as u64 * 4]);

            layers[i].ds_norm_qkv = (ds_norm1, ds_qkv);
            layers[i].ds_out_proj = ds_out;
            layers[i].ds_norm_ffnup = (ds_norm2, ds_up);
            layers[i].ds_ffn_down = ds_dn;
        }

        // GPU-resident pipeline buffers
        let qkv_dim = config.qkv_dim();
        let gpu_hidden = engine.alloc_buffer(d_model as u64 * 4);
        let gpu_normed = engine.alloc_buffer(d_model as u64 * 4);
        let gpu_qkv = engine.alloc_buffer(qkv_dim as u64 * 4);
        let gpu_gate_up = engine.alloc_buffer(gate_up_dim as u64 * 4);
        let gpu_ffn_mid = engine.alloc_buffer(config.ffn_dim as u64 * 4);
        let gpu_tmp = engine.alloc_buffer(d_model as u64 * 4);
        let rope_factor_len = if rope_factors.is_empty() { 1 } else { rope_factors.len() };
        let gpu_rope_factors = engine.alloc_buffer(rope_factor_len as u64 * 4);
        if !rope_factors.is_empty() { upload(&gpu_rope_factors, &rope_factors); }

        // Pre-allocate descriptor sets for the mega-batch (combined QKV models only)
        let d4 = d_model as u64 * 4;
        let kv_max_bytes = (MAX_SEQ * kv_stride) as u64 * 2;
        let mut mega_dispatches: Vec<PrebuiltDispatch> = Vec::new();

        if !config.separate_qkv && !config.separate_gate_up {
        for i in 0..config.n_layers {
            let layer = &layers[i];
            let (ref kb, ref vb) = kv_bufs[i];
            let bpr_qkv = layer.attn_qkv_k as u64 / 256;
            let wb_qkv = layer.attn_qkv_n as u64 * bpr_qkv * layer.attn_qkv_type.block_bytes() as u64;
            let bpr_out = layer.attn_output_k as u64 / 256;
            let wb_out = layer.attn_output_n as u64 * bpr_out * layer.attn_output_type.block_bytes() as u64;
            let bpr_up = layer.ffn_up_k as u64 / 256;
            let wb_up = layer.ffn_up_n as u64 * bpr_up * layer.ffn_up_type.block_bytes() as u64;
            let bpr_dn = layer.ffn_down_k as u64 / 256;
            let wb_dn = layer.ffn_down_n as u64 * bpr_dn * layer.ffn_down_type.block_bytes() as u64;
            let rope_pairs = (config.n_heads + config.n_kv_heads) * (config.rope_dim / 2);
            let rope_wg = ((rope_pairs + 63) / 64) as u32;
            let kv_store_wg = ((kv_stride / 2 + 255) / 256) as u32;
            let silu_wg = ((config.ffn_dim + 255) / 256) as u32;
            let resid_wg = ((d_model + 255) / 256) as u32;

            macro_rules! pre {
                ($name:expr, $bufs:expr, $sizes:expr, $wg:expr) => {
                    pre!($name, $bufs, $sizes, $wg, true)
                };
                ($name:expr, $bufs:expr, $sizes:expr, $wg:expr, $barrier:expr) => {{
                    let ds = engine.pre_allocate_desc_set($name, $bufs, $sizes);
                    mega_dispatches.push(PrebuiltDispatch {
                        pipeline_name: $name.to_string(), desc_set: ds, workgroups: $wg, needs_barrier: $barrier,
                    });
                }}
            }

            // All barriers required for correctness on RADV (L2 not implicitly coherent
            // between dispatches). Tested: removing any barrier corrupts output.
            pre!("rmsnorm", &[&gpu_hidden, &layer.attn_norm_buf, &gpu_normed], &[d4, d4, d4], [1,1,1]);
            pre!(layer.attn_qkv_type.shader_name(), &[&gpu_normed, &layer.attn_qkv_buf, &gpu_qkv],
                &[layer.attn_qkv_k as u64*4, wb_qkv, layer.attn_qkv_n as u64*4],
                [layer.attn_qkv_type.workgroups(layer.attn_qkv_n), 1, 1]);
            // Fused RoPE + KV store
            let fused_wg = {
                let rp = (config.n_heads + config.n_kv_heads) * (config.rope_dim / 2);
                let hkv = kv_stride / 2;
                ((std::cmp::max(rp, hkv) + 63) / 64) as u32
            };
            pre!("rope_kv_store", &[&gpu_qkv, &gpu_rope_factors, kb, vb],
                &[qkv_dim as u64*4, rope_factor_len as u64*4, kv_max_bytes, kv_max_bytes], [fused_wg, 1, 1]);
            pre!("attention", &[&gpu_qkv, kb, vb, &attn_out_buf],
                &[d4, kv_max_bytes, kv_max_bytes, d4], [config.n_heads as u32, 1, 1]);
            pre!(layer.attn_output_type.shader_name(), &[&attn_out_buf, &layer.attn_output_buf, &gpu_tmp],
                &[layer.attn_output_k as u64*4, wb_out, layer.attn_output_n as u64*4],
                [layer.attn_output_type.workgroups(layer.attn_output_n), 1, 1]);
            pre!("residual_add", &[&gpu_hidden, &gpu_tmp, &gpu_hidden], &[d4, d4, d4], [resid_wg, 1, 1]);
            pre!("rmsnorm", &[&gpu_hidden, &layer.ffn_norm_buf, &gpu_normed], &[d4, d4, d4], [1,1,1]);
            pre!(layer.ffn_up_type.shader_name(), &[&gpu_normed, &layer.ffn_up_buf, &gpu_gate_up],
                &[layer.ffn_up_k as u64*4, wb_up, layer.ffn_up_n as u64*4],
                [layer.ffn_up_type.workgroups(layer.ffn_up_n), 1, 1]);
            pre!("silu_gate", &[&gpu_gate_up, &gpu_ffn_mid],
                &[gate_up_dim as u64*4, config.ffn_dim as u64*4], [silu_wg, 1, 1]);
            pre!(layer.ffn_down_type.shader_name(), &[&gpu_ffn_mid, &layer.ffn_down_buf, &gpu_tmp],
                &[layer.ffn_down_k as u64*4, wb_dn, layer.ffn_down_n as u64*4],
                [layer.ffn_down_type.workgroups(layer.ffn_down_n), 1, 1]);
            pre!("residual_add", &[&gpu_hidden, &gpu_tmp, &gpu_hidden], &[d4, d4, d4], [resid_wg, 1, 1]);
        }

        } // end if !separate_qkv

        // Append output norm + logit chunk dispatches to mega_dispatches (if mega-batch enabled)
        let gpu_output_norm_w = engine.alloc_buffer(d_model as u64 * 4);
        upload(&gpu_output_norm_w, &output_norm);
        // Output RMS norm: gpu_hidden → gpu_normed
        {
            let ds = engine.pre_allocate_desc_set("rmsnorm",
                &[&gpu_hidden, &gpu_output_norm_w, &gpu_normed], &[d4, d4, d4]);
            mega_dispatches.push(PrebuiltDispatch {
                pipeline_name: "rmsnorm".to_string(), desc_set: ds, workgroups: [1,1,1], needs_barrier: true,
            });
        }
        // Logit matmul chunks: gpu_normed × emb_chunk → logit_out_bufs
        if !mega_dispatches.is_empty() {
        let mk = 1u32;
        let kk = d_model as u32;
        for (i, (emb_buf, chunk_n)) in emb_bufs.iter().enumerate() {
            let n = *chunk_n as u32;
            let ds = engine.pre_allocate_desc_set("matmul_tiled",
                &[&gpu_normed, emb_buf, &logit_out_bufs[i]],
                &[(mk as u64)*(kk as u64)*4, (kk as u64)*(n as u64)*4, (mk as u64)*(n as u64)*4]);
            mega_dispatches.push(PrebuiltDispatch {
                pipeline_name: "matmul_tiled".to_string(), desc_set: ds,
                workgroups: [(n + 15) / 16, (mk + 15) / 16, 1], needs_barrier: false,
            });
        }

        } // end if mega_dispatches logit chunks

        // Scratch buffers for argmax (not used yet — keep CPU argmax for now)
        let gpu_logits = engine.alloc_buffer(4);  // placeholder
        let gpu_argmax_scratch = engine.alloc_buffer(4);
        let gpu_argmax_result = engine.alloc_buffer(4);

        eprintln!("[phi4] Ready: {:.1}s, {:.1}MB total GPU ({} pre-built dispatches)",
            t0.elapsed().as_secs_f32(), total_gpu as f64 / 1e6, mega_dispatches.len());

        Phi4Model { config, embedding, output_norm, layers, buf_a, buf_c, logits_buf, emb_bufs, rope_factors,
                    attn_q_buf, attn_out_buf, kv_bufs, logit_out_bufs,
                    gpu_hidden, gpu_normed, gpu_qkv, gpu_gate_up, gpu_ffn_mid, gpu_tmp, gpu_rope_factors,
                    mega_dispatches, gpu_output_norm_w, gpu_logits, gpu_argmax_scratch, gpu_argmax_result }
    }

    /// Helper: build a K-quant matmul dispatch tuple for use with dispatch_batch.
    fn kquant_dispatch<'a>(
        buf_in: &'a GpuBuffer, weight_buf: &'a GpuBuffer, buf_out: &'a GpuBuffer,
        qtype: QType, k: u32, n: u32,
    ) -> (String, Vec<&'a GpuBuffer>, Vec<u64>, Vec<u8>, [u32; 3]) {
        let blocks_per_row = k as u64 / 256;
        let total_weight_bytes = n as u64 * blocks_per_row * qtype.block_bytes() as u64;
        let push: [u32; 2] = [k, n];
        let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
        (
            qtype.shader_name().to_string(),
            vec![buf_in, weight_buf, buf_out],
            vec![k as u64 * 4, total_weight_bytes, n as u64 * 4],
            pb,
            [qtype.workgroups(n), 1, 1],
        )
    }

    /// Forward pass — mega-batch if available, else per-dispatch fallback.
    pub unsafe fn forward_gpu(&self, engine: &ComputeEngine, hidden: &mut Vec<f32>, cache: &mut KVCache) -> Vec<f32> {
        // Mega-batch only works for combined QKV models (Phi-4). Others use dynamic path.
        if self.config.separate_qkv || self.config.separate_gate_up {
            return self.forward_dynamic(engine, hidden, cache);
        }
        assert_eq!(hidden.len(), D_MODEL);
        let pos = cache.len;
        let d = D_MODEL as u32;
        let heads_per_kv = N_HEADS / N_KV_HEADS;
        let kv_stride = (N_KV_HEADS * HEAD_DIM) as u32;
        let seq_len = pos as u32 + 1;

        upload(&self.gpu_hidden, hidden);

        // Build push constants (only thing that changes per token)
        let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();
        let rope_kv_push: Vec<u8> = [pos as u32, N_HEADS as u32, N_KV_HEADS as u32, HEAD_DIM as u32, ROPE_DIM as u32,
            (N_HEADS * HEAD_DIM) as u32, (N_KV_HEADS * HEAD_DIM) as u32]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        let silu_push: Vec<u8> = [FFN_DIM as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let resid_push: Vec<u8> = [d].iter().flat_map(|v| v.to_le_bytes()).collect();
        let attn_push: Vec<u8> = [seq_len, N_KV_HEADS as u32, heads_per_kv as u32, HEAD_DIM as u32]
            .iter().flat_map(|v| v.to_le_bytes()).collect();

        // 11 dispatches per layer: rmsnorm, qkv, rope_kv_store, attn, out_proj, resid, rmsnorm, ffn_up, silu, ffn_down, resid
        let dispatches_per_layer = 11usize;
        let push_map: [&[u8]; 11] = [
            &norm_push, &[],  // qkv
            &rope_kv_push, &attn_push,
            &[],  // out_proj
            &resid_push, &norm_push,
            &[],  // ffn_up
            &silu_push,
            &[],  // ffn_down
            &resid_push,
        ];

        // Build push constants for ALL dispatches
        let n_layer_dispatches = dispatches_per_layer * N_LAYERS;
        let n_total = self.mega_dispatches.len();
        let mut push_data: Vec<Vec<u8>> = Vec::with_capacity(n_total);

        for i in 0..n_layer_dispatches {
            let slot = i % dispatches_per_layer;
            let li = i / dispatches_per_layer;
            let layer = &self.layers[li];
            let push = match slot {
                1 => [layer.attn_qkv_k, layer.attn_qkv_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                4 => [layer.attn_output_k, layer.attn_output_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                7 => [layer.ffn_up_k, layer.ffn_up_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                9 => [layer.ffn_down_k, layer.ffn_down_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                _ => push_map[slot].to_vec(),
            };
            push_data.push(push);
        }
        // Output norm dispatch (index n_layer_dispatches)
        push_data.push(norm_push.clone());
        // Logit chunk dispatches (indices n_layer_dispatches+1 ..)
        for (_, chunk_n) in &self.emb_bufs {
            let n = *chunk_n as u32;
            let push: Vec<u8> = [1u32, D_MODEL as u32, n].iter().flat_map(|v| v.to_le_bytes()).collect();
            push_data.push(push);
        }

        let dispatches: Vec<(&str, DescSetHandle, &[u8], [u32; 3], bool)> = self.mega_dispatches.iter()
            .enumerate()
            .map(|(i, d)| (d.pipeline_name.as_str(), d.desc_set, push_data[i].as_slice(), d.workgroups, d.needs_barrier))
            .collect();

        engine.dispatch_batch_persistent_v2(&dispatches);

        cache.len += 1;

        // CPU argmax directly from mapped logit output buffers (zero-copy)
        let mut best_val = f32::NEG_INFINITY;
        let mut best_id = 0u32;
        let mut vocab_offset = 0u32;
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            for j in 0..n {
                if src[j] > best_val { best_val = src[j]; best_id = vocab_offset + j as u32; }
            }
            vocab_offset += n as u32;
        }

        // Still return logits for compatibility, but the hot path just needs best_id
        // (caller can access self.last_best_id if needed)
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        vocab_offset = 0;
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            logits[vocab_offset as usize..(vocab_offset as usize + n)].copy_from_slice(src);
            vocab_offset += n as u32;
        }
        logits
    }

    /// Batched prefill: process M tokens per layer, batching matmuls to share weight reads.
    /// Weight data loaded once per matmul instead of M times. ~80% bandwidth savings for M=5.
    pub unsafe fn forward_prefill_batched(
        &self, engine: &ComputeEngine, token_ids: &[u32], cache: &mut KVCache,
    ) -> Vec<f32> {
        let base_pos = cache.len;
        let m = token_ids.len();
        let heads_per_kv = N_HEADS / N_KV_HEADS;
        let d = D_MODEL as u32;
        let kv_stride = N_KV_HEADS * HEAD_DIM;
        let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();
        let kv_max_bytes = (MAX_SEQ * kv_stride) as u64 * 2;

        // Embed all tokens
        let mut hiddens: Vec<Vec<f32>> = token_ids.iter().map(|&t| self.embed(t)).collect();

        // Allocate batch GPU buffers (M copies)
        let buf_hiddens = engine.alloc_buffer((m * D_MODEL) as u64 * 4);
        let buf_normed = engine.alloc_buffer((m * D_MODEL) as u64 * 4);
        let buf_qkv = engine.alloc_buffer((m * QKV_DIM) as u64 * 4);
        let buf_gate_up = engine.alloc_buffer((m * GATE_UP_DIM) as u64 * 4);
        let buf_ffn_mid = engine.alloc_buffer((m * FFN_DIM) as u64 * 4);
        let buf_tmp = engine.alloc_buffer((m * D_MODEL) as u64 * 4);

        for (_li, layer) in self.layers.iter().enumerate() {
            let (ref kbuf, ref vbuf) = self.kv_bufs[_li];
            let bpr_qkv = layer.attn_qkv_k as u64 / 256;
            let wb_qkv = layer.attn_qkv_n as u64 * bpr_qkv * layer.attn_qkv_type.block_bytes() as u64;
            let bpr_out = layer.attn_output_k as u64 / 256;
            let wb_out = layer.attn_output_n as u64 * bpr_out * layer.attn_output_type.block_bytes() as u64;
            let bpr_up = layer.ffn_up_k as u64 / 256;
            let wb_up = layer.ffn_up_n as u64 * bpr_up * layer.ffn_up_type.block_bytes() as u64;
            let bpr_dn = layer.ffn_down_k as u64 / 256;
            let wb_dn = layer.ffn_down_n as u64 * bpr_dn * layer.ffn_down_type.block_bytes() as u64;

            // Upload all M hidden states
            for t in 0..m {
                let off = t * D_MODEL * 4;
                std::ptr::copy_nonoverlapping(
                    hiddens[t].as_ptr() as *const u8,
                    (buf_hiddens.mapped as *mut u8).add(off), D_MODEL * 4);
            }

            // RMS norm: M separate dispatches (different hidden states)
            for t in 0..m {
                let t_off = (t * D_MODEL) as u64 * 4;
                // Use buf_a/buf_c as temp, norm each token separately
                let h_slice = &hiddens[t];
                upload(&self.buf_a, h_slice);
                let pb = norm_push.clone();
                engine.dispatch_by_name("rmsnorm",
                    &[&self.buf_a, &layer.attn_norm_buf, &self.buf_c],
                    &[d as u64*4, d as u64*4, d as u64*4], &pb, [1,1,1]);
                // Copy normed result into batch buffer at token t's slot
                std::ptr::copy_nonoverlapping(
                    self.buf_c.mapped as *const u8,
                    (buf_normed.mapped as *mut u8).add(t * D_MODEL * 4), D_MODEL * 4);
            }

            // Batched QKV matmul: [M, K] × W → [M, N], ONE dispatch for all tokens
            {
                let push: Vec<u8> = [layer.attn_qkv_k, layer.attn_qkv_n, m as u32]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name(
                    layer.attn_qkv_type.batch_shader_name(),
                    &[&buf_normed, &layer.attn_qkv_buf, &buf_qkv],
                    &[(m * D_MODEL) as u64 * 4, wb_qkv, (m * QKV_DIM) as u64 * 4],
                    &push,
                    [layer.attn_qkv_type.workgroups(layer.attn_qkv_n), m as u32, 1],
                );
            }

            // Per-token: RoPE + KV store + attention
            for t in 0..m {
                let pos = base_pos + t;
                let seq_len = pos + 1;

                // Download this token's QKV, apply RoPE, store KV, run attention
                let mut qkv = vec![0.0f32; QKV_DIM];
                std::ptr::copy_nonoverlapping(
                    (buf_qkv.mapped as *const u8).add(t * QKV_DIM * 4),
                    qkv.as_mut_ptr() as *mut u8, QKV_DIM * 4);

                // Upload QKV back to GPU for RoPE + KV store (f32→f16 packing)
                upload(&self.gpu_qkv, &qkv);

                // GPU: RoPE + KV store (f16 packing on GPU)
                let rope_kv_push: Vec<u8> = [pos as u32, N_HEADS as u32, N_KV_HEADS as u32,
                    HEAD_DIM as u32, ROPE_DIM as u32, (N_HEADS * HEAD_DIM) as u32, kv_stride as u32]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                let fused_wg = {
                    let rope_pairs = (N_HEADS + N_KV_HEADS) * (ROPE_DIM / 2);
                    let half_kv = kv_stride / 2;
                    ((std::cmp::max(rope_pairs, half_kv) + 63) / 64) as u32
                };
                engine.dispatch_by_name("rope_kv_store",
                    &[&self.gpu_qkv, &self.gpu_rope_factors, kbuf, vbuf],
                    &[QKV_DIM as u64*4, (ROPE_DIM as u64/2)*4, kv_max_bytes, kv_max_bytes],
                    &rope_kv_push, [fused_wg, 1, 1]);

                // Read back Q for attention (post-RoPE)
                let mut q = vec![0.0f32; N_HEADS * HEAD_DIM];
                std::ptr::copy_nonoverlapping(
                    self.gpu_qkv.mapped as *const u8,
                    q.as_mut_ptr() as *mut u8, (N_HEADS * HEAD_DIM) * 4);

                // GPU attention + output projection
                upload(&self.attn_q_buf, &q);
                let attn_push: Vec<u8> = [seq_len as u32, N_KV_HEADS as u32, heads_per_kv as u32, HEAD_DIM as u32]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                let kv_bytes = (seq_len * kv_stride) as u64 * 2; // f16
                engine.dispatch_by_name("attention",
                    &[&self.attn_q_buf, kbuf, vbuf, &self.attn_out_buf],
                    &[D_MODEL as u64*4, kv_bytes, kv_bytes, D_MODEL as u64*4],
                    &attn_push, [N_HEADS as u32, 1, 1]);

                // Output projection → tmp
                let out_push: Vec<u8> = [layer.attn_output_k, layer.attn_output_n]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name(layer.attn_output_type.shader_name(),
                    &[&self.attn_out_buf, &layer.attn_output_buf, &self.buf_c],
                    &[layer.attn_output_k as u64*4, wb_out, layer.attn_output_n as u64*4],
                    &out_push, [layer.attn_output_type.workgroups(layer.attn_output_n), 1, 1]);

                // Download tmp, add residual
                let mut tmp = vec![0.0f32; D_MODEL];
                download(&self.buf_c, &mut tmp);
                for i in 0..D_MODEL { hiddens[t][i] += tmp[i]; }
            }

            // Upload all M updated hidden states for FFN norm
            for t in 0..m {
                std::ptr::copy_nonoverlapping(
                    hiddens[t].as_ptr() as *const u8,
                    (buf_hiddens.mapped as *mut u8).add(t * D_MODEL * 4), D_MODEL * 4);
            }

            // RMS norm (FFN): M separate dispatches
            for t in 0..m {
                upload(&self.buf_a, &hiddens[t]);
                engine.dispatch_by_name("rmsnorm",
                    &[&self.buf_a, &layer.ffn_norm_buf, &self.buf_c],
                    &[d as u64*4, d as u64*4, d as u64*4], &norm_push, [1,1,1]);
                std::ptr::copy_nonoverlapping(
                    self.buf_c.mapped as *const u8,
                    (buf_normed.mapped as *mut u8).add(t * D_MODEL * 4), D_MODEL * 4);
            }

            // Batched FFN up matmul
            {
                let push: Vec<u8> = [layer.ffn_up_k, layer.ffn_up_n, m as u32]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name(
                    layer.ffn_up_type.batch_shader_name(),
                    &[&buf_normed, &layer.ffn_up_buf, &buf_gate_up],
                    &[(m * D_MODEL) as u64*4, wb_up, (m * GATE_UP_DIM) as u64*4],
                    &push,
                    [layer.ffn_up_type.workgroups(layer.ffn_up_n), m as u32, 1],
                );
            }

            // Per-token SiLU + gate*up
            let mut all_ffn_mid = vec![0.0f32; m * FFN_DIM];
            {
                let mut all_gate_up = vec![0.0f32; m * GATE_UP_DIM];
                std::ptr::copy_nonoverlapping(
                    buf_gate_up.mapped as *const u8,
                    all_gate_up.as_mut_ptr() as *mut u8, m * GATE_UP_DIM * 4);
                for t in 0..m {
                    let go = t * GATE_UP_DIM;
                    let fo = t * FFN_DIM;
                    for i in 0..FFN_DIM {
                        let gate = all_gate_up[go + i];
                        let silu = gate / (1.0 + (-gate).exp());
                        all_ffn_mid[fo + i] = silu * all_gate_up[go + FFN_DIM + i];
                    }
                }
                std::ptr::copy_nonoverlapping(
                    all_ffn_mid.as_ptr() as *const u8,
                    buf_ffn_mid.mapped as *mut u8, m * FFN_DIM * 4);
            }

            // Batched FFN down matmul
            {
                let push: Vec<u8> = [layer.ffn_down_k, layer.ffn_down_n, m as u32]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name(
                    layer.ffn_down_type.batch_shader_name(),
                    &[&buf_ffn_mid, &layer.ffn_down_buf, &buf_tmp],
                    &[(m * FFN_DIM) as u64*4, wb_dn, (m * D_MODEL) as u64*4],
                    &push,
                    [layer.ffn_down_type.workgroups(layer.ffn_down_n), m as u32, 1],
                );
            }

            // Download FFN results, add residual
            {
                let mut all_tmp = vec![0.0f32; m * D_MODEL];
                std::ptr::copy_nonoverlapping(
                    buf_tmp.mapped as *const u8,
                    all_tmp.as_mut_ptr() as *mut u8, m * D_MODEL * 4);
                for t in 0..m {
                    let to = t * D_MODEL;
                    for i in 0..D_MODEL { hiddens[t][i] += all_tmp[to + i]; }
                }
            }
        }

        cache.len += m;

        // Logits for last token
        let hidden = &hiddens[m - 1];
        let mut norm = vec![0.0f32; D_MODEL];
        rms_norm(hidden, &self.output_norm, &mut norm);
        upload(&self.buf_a, &norm);

        let mk = 1u32;
        let kk = D_MODEL as u32;
        let mut dispatches_data: Vec<(Vec<u8>, [u32; 3], usize)> = Vec::new();
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n as u32;
            let push: [u32; 3] = [mk, kk, n];
            let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
            dispatches_data.push((pb, [(n + 15) / 16, (mk + 15) / 16, 1], i));
        }
        let dispatches: Vec<(&str, Vec<&GpuBuffer>, Vec<u64>, &[u8], [u32; 3])> = dispatches_data.iter()
            .map(|(pb, groups, i)| {
                let n = self.emb_bufs[*i].1;
                ("matmul_tiled",
                 vec![&self.buf_a, &self.emb_bufs[*i].0, &self.logit_out_bufs[*i]],
                 vec![(mk as u64)*(kk as u64)*4, (kk as u64)*(n as u64)*4, (mk as u64)*(n as u64)*4],
                 pb.as_slice(), *groups)
            }).collect();
        let dispatch_refs: Vec<(&str, &[&GpuBuffer], &[u64], &[u8], [u32; 3])> = dispatches.iter()
            .map(|(name, bufs, sizes, push, groups)| (*name, bufs.as_slice(), sizes.as_slice(), *push, *groups))
            .collect();
        engine.dispatch_batch_parallel(&dispatch_refs);

        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let mut vocab_offset = 0usize;
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            logits[vocab_offset..vocab_offset + n].copy_from_slice(src);
            vocab_offset += n;
        }
        logits
    }

    /// Draft forward: run only the first `n_layers` layers, then output norm + logits.
    /// Uses the same mega_dispatches table but only dispatches entries 0..n_layers*11.
    /// Returns logits (for draft token prediction).
    pub unsafe fn forward_draft(
        &self, engine: &ComputeEngine, hidden: &mut Vec<f32>, cache: &mut KVCache, n_layers: usize,
    ) -> Vec<f32> {
        assert_eq!(hidden.len(), D_MODEL);
        let pos = cache.len;
        let d = D_MODEL as u32;
        let heads_per_kv = N_HEADS / N_KV_HEADS;
        let kv_stride = (N_KV_HEADS * HEAD_DIM) as u32;
        let seq_len = pos as u32 + 1;
        let dispatches_per_layer = 11usize;

        upload(&self.gpu_hidden, hidden);

        // Build push constants for draft layers only
        let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();
        let rope_kv_push: Vec<u8> = [pos as u32, N_HEADS as u32, N_KV_HEADS as u32, HEAD_DIM as u32, ROPE_DIM as u32,
            (N_HEADS * HEAD_DIM) as u32, (N_KV_HEADS * HEAD_DIM) as u32]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        let attn_push: Vec<u8> = [seq_len, N_KV_HEADS as u32, heads_per_kv as u32, HEAD_DIM as u32]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        let silu_push: Vec<u8> = [FFN_DIM as u32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let resid_push: Vec<u8> = [d].iter().flat_map(|v| v.to_le_bytes()).collect();

        let push_map: [&[u8]; 11] = [
            &norm_push, &[], &rope_kv_push, &attn_push,
            &[], &resid_push, &norm_push, &[], &silu_push, &[], &resid_push,
        ];

        let n_draft_dispatches = dispatches_per_layer * n_layers;
        let mut push_data: Vec<Vec<u8>> = Vec::with_capacity(n_draft_dispatches);
        for i in 0..n_draft_dispatches {
            let slot = i % dispatches_per_layer;
            let li = i / dispatches_per_layer;
            let layer = &self.layers[li];
            let push = match slot {
                1 => [layer.attn_qkv_k, layer.attn_qkv_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                4 => [layer.attn_output_k, layer.attn_output_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                7 => [layer.ffn_up_k, layer.ffn_up_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                9 => [layer.ffn_down_k, layer.ffn_down_n].iter().flat_map(|v| v.to_le_bytes()).collect(),
                _ => push_map[slot].to_vec(),
            };
            push_data.push(push);
        }

        // Dispatch only the first n_layers worth of the pre-built table
        let dispatches: Vec<(&str, DescSetHandle, &[u8], [u32; 3], bool)> = self.mega_dispatches[..n_draft_dispatches]
            .iter().enumerate()
            .map(|(i, d)| (d.pipeline_name.as_str(), d.desc_set, push_data[i].as_slice(), d.workgroups, d.needs_barrier))
            .collect();
        engine.dispatch_batch_persistent_v2(&dispatches);

        cache.len += 1;

        // Download hidden, output norm, logits (same as forward_gpu tail)
        download(&self.gpu_hidden, hidden);
        let mut norm = vec![0.0f32; D_MODEL];
        rms_norm(hidden, &self.output_norm, &mut norm);

        // Quick CPU logits (argmax only, skip full vocab projection for speed)
        // Actually we need full logits for speculative verify. Use GPU path.
        upload(&self.buf_a, &norm);
        let mk = 1u32; let kk = D_MODEL as u32;
        let mut dispatches_data: Vec<(Vec<u8>, [u32; 3], usize)> = Vec::new();
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n as u32;
            let pb: Vec<u8> = [mk, kk, n].iter().flat_map(|v| v.to_le_bytes()).collect();
            dispatches_data.push((pb, [(n+15)/16, (mk+15)/16, 1], i));
        }
        let dsp: Vec<(&str, Vec<&GpuBuffer>, Vec<u64>, &[u8], [u32; 3])> = dispatches_data.iter()
            .map(|(pb, groups, i)| {
                let n = self.emb_bufs[*i].1;
                ("matmul_tiled",
                 vec![&self.buf_a, &self.emb_bufs[*i].0, &self.logit_out_bufs[*i]],
                 vec![(mk as u64)*(kk as u64)*4, (kk as u64)*(n as u64)*4, (mk as u64)*(n as u64)*4],
                 pb.as_slice(), *groups)
            }).collect();
        let refs: Vec<(&str, &[&GpuBuffer], &[u64], &[u8], [u32; 3])> = dsp.iter()
            .map(|(n, b, s, p, g)| (*n, b.as_slice(), s.as_slice(), *p, *g)).collect();
        engine.dispatch_batch_parallel(&refs);

        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let mut vo = 0usize;
        for (i, (_, cn)) in self.emb_bufs.iter().enumerate() {
            let n = *cn;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            logits[vo..vo+n].copy_from_slice(src);
            vo += n;
        }
        logits
    }

    /// Speculative decode: draft K tokens with first `draft_layers` layers,
    /// then verify with full model. Returns accepted tokens.
    pub unsafe fn speculative_decode(
        &self, engine: &ComputeEngine, last_logits: &[f32], cache: &mut KVCache,
        draft_layers: usize, draft_k: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        // Step 1: Draft K tokens using shallow model
        let mut draft_tokens = Vec::with_capacity(draft_k);
        let mut draft_logits = last_logits.to_vec();
        let save_pos = cache.len;

        for _ in 0..draft_k {
            let (best, _) = draft_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            let tok = best as u32;
            draft_tokens.push(tok);

            let mut hidden = self.embed(tok);
            draft_logits = self.forward_draft(engine, &mut hidden, cache, draft_layers);
        }

        // Step 2: Rewind cache to before draft
        cache.len = save_pos;

        // Step 3: Verify all draft tokens with full model
        // Run forward_gpu for each draft token (using full 32 layers)
        let mut verified_logits = Vec::new();
        let mut accepted = Vec::new();

        for (i, &tok) in draft_tokens.iter().enumerate() {
            let mut hidden = self.embed(tok);
            let verify_logits = self.forward_gpu(engine, &mut hidden, cache);

            let (verify_top, _) = verify_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

            if verify_top as u32 == draft_tokens.get(i + 1).copied().unwrap_or(verify_top as u32) || i == draft_k - 1 {
                // Draft's NEXT token matches verify's prediction at this position, accept
                accepted.push(tok);
                verified_logits = verify_logits;
            } else {
                // Mismatch — accept this token but use verify's prediction going forward
                accepted.push(tok);
                verified_logits = verify_logits;
                break;
            }
        }

        (accepted, verified_logits)
    }

    /// Config-aware forward pass for any architecture (separate Q/K/V, biases, etc.)
    pub unsafe fn forward_dynamic(&self, engine: &ComputeEngine, hidden: &mut Vec<f32>, cache: &mut KVCache) -> Vec<f32> {
        let cfg = &self.config;
        let pos = cache.len;
        let d = cfg.d_model as u32;
        let heads_per_kv = cfg.n_heads / cfg.n_kv_heads;
        let kv_stride = cfg.n_kv_heads * cfg.head_dim;
        let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();

        for (li, layer) in self.layers.iter().enumerate() {
            // --- RMS norm → Q matmul (or combined QKV matmul) ---
            upload(&self.buf_a, hidden);
            let qkv_d = Self::kquant_dispatch(
                &self.buf_c, &layer.attn_qkv_buf, &self.buf_a,
                layer.attn_qkv_type, layer.attn_qkv_k, layer.attn_qkv_n);
            {
                let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.attn_norm_buf, &self.buf_c];
                let sizes1: Vec<u64> = vec![d as u64 * 4; 3];
                engine.dispatch_batch(&[
                    ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                    (qkv_d.0.as_str(), &qkv_d.1, &qkv_d.2, &qkv_d.3, qkv_d.4),
                ]);
            }
            let q_dim = cfg.n_heads * cfg.head_dim;
            let k_dim = cfg.n_kv_heads * cfg.head_dim;

            let (mut q, mut k, v);
            if cfg.separate_qkv {
                // Separate Q/K/V: buf_a has Q result. Dispatch K and V separately.
                let mut q_raw = vec![0.0f32; q_dim];
                download(&self.buf_a, &mut q_raw);

                // K matmul: normed (in buf_c) → K
                let (ref k_buf, ref k_type, k_k, k_n) = layer.attn_k_buf.as_ref().unwrap();
                let k_d = Self::kquant_dispatch(&self.buf_c, k_buf, &self.buf_a, *k_type, *k_k, *k_n);
                engine.dispatch_by_name(k_d.0.as_str(), &k_d.1, &k_d.2, &k_d.3, k_d.4);
                let mut k_raw = vec![0.0f32; k_dim];
                download(&self.buf_a, &mut k_raw);

                // V matmul
                let (ref v_buf, ref v_type, v_k, v_n) = layer.attn_v_buf.as_ref().unwrap();
                let v_d = Self::kquant_dispatch(&self.buf_c, v_buf, &self.buf_a, *v_type, *v_k, *v_n);
                engine.dispatch_by_name(v_d.0.as_str(), &v_d.1, &v_d.2, &v_d.3, v_d.4);
                let mut v_raw = vec![0.0f32; k_dim];
                download(&self.buf_a, &mut v_raw);

                // Apply bias if present
                if let Some(ref bias) = layer.qkv_bias {
                    for i in 0..q_dim { q_raw[i] += bias[i]; }
                    for i in 0..k_dim { k_raw[i] += bias[q_dim + i]; }
                    for i in 0..k_dim { v_raw[i] += bias[q_dim + k_dim + i]; }
                }
                q = q_raw; k = k_raw; v = v_raw;
            } else {
                // Combined QKV
                let qkv_dim = cfg.qkv_dim();
                let mut qkv = vec![0.0f32; qkv_dim];
                download(&self.buf_a, &mut qkv[..qkv_dim]);
                let k_end = q_dim + k_dim;
                q = qkv[..q_dim].to_vec();
                k = qkv[q_dim..k_end].to_vec();
                v = qkv[k_end..].to_vec();
            }

            apply_rope_cfg(&mut q, pos, cfg.n_heads, cfg, &self.rope_factors);
            apply_rope_cfg(&mut k, pos, cfg.n_kv_heads, cfg, &self.rope_factors);

            // KV cache update (f16 via GPU shader)
            let mut qkv_concat = Vec::with_capacity(q.len() + k.len() + v.len());
            qkv_concat.extend_from_slice(&q);
            qkv_concat.extend_from_slice(&k);
            qkv_concat.extend_from_slice(&v);
            upload(&self.gpu_qkv, &qkv_concat);
            let (ref kbuf, ref vbuf) = self.kv_bufs[li];
            let rope_kv_push: Vec<u8> = [pos as u32, cfg.n_heads as u32, cfg.n_kv_heads as u32,
                cfg.head_dim as u32, 0u32, // rope_dim=0 means skip rope (already applied)
                q_dim as u32, kv_stride as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            // Use kv_store directly instead of rope_kv_store (RoPE already done)
            let half_kv = kv_stride / 2;
            let kv_store_push: Vec<u8> = [pos as u32, q_dim as u32, kv_stride as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            let kv_store_wg = ((half_kv + 255) / 256) as u32;
            engine.dispatch_by_name("kv_store",
                &[&self.gpu_qkv, kbuf, vbuf],
                &[(q_dim + 2*kv_stride) as u64 * 4, (MAX_SEQ * kv_stride) as u64 * 2, (MAX_SEQ * kv_stride) as u64 * 2],
                &kv_store_push, [kv_store_wg, 1, 1]);

            // Attention
            let seq_len = pos + 1;
            upload(&self.attn_q_buf, &q);
            let attn_push: Vec<u8> = [seq_len as u32, cfg.n_kv_heads as u32, heads_per_kv as u32, cfg.head_dim as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            let kv_bytes = (seq_len * kv_stride) as u64 * 2;
            engine.dispatch_by_name("attention",
                &[&self.attn_q_buf, kbuf, vbuf, &self.attn_out_buf],
                &[cfg.d_model as u64 * 4, kv_bytes, kv_bytes, cfg.d_model as u64 * 4],
                &attn_push, [cfg.n_heads as u32, 1, 1]);

            // Output projection + residual
            let out_push: Vec<u8> = [layer.attn_output_k, layer.attn_output_n]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu_matmul_kquant(engine, &self.buf_a, &layer.attn_output_buf, &self.buf_c,
                layer.attn_output_type, &{let mut ao = vec![0.0f32; cfg.d_model]; download(&self.attn_out_buf, &mut ao); ao},
                &mut vec![0.0f32; cfg.d_model], layer.attn_output_k, layer.attn_output_n);
            let mut tmp = vec![0.0f32; cfg.d_model];
            download(&self.buf_c, &mut tmp);
            for i in 0..cfg.d_model { hidden[i] += tmp[i]; }

            // FFN: norm → gate + up → SiLU → down → residual
            upload(&self.buf_a, hidden);
            if cfg.separate_gate_up {
                // Separate gate and up matmuls
                let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.ffn_norm_buf, &self.buf_c];
                let sizes1: Vec<u64> = vec![d as u64 * 4; 3];
                engine.dispatch_batch(&[("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1])]);
                // normed is in buf_c. Dispatch gate matmul (stored in ffn_up_buf)
                let gate_d = Self::kquant_dispatch(&self.buf_c, &layer.ffn_up_buf, &self.buf_a,
                    layer.ffn_up_type, layer.ffn_up_k, layer.ffn_up_n);
                engine.dispatch_by_name(gate_d.0.as_str(), &gate_d.1, &gate_d.2, &gate_d.3, gate_d.4);
                let mut gate = vec![0.0f32; cfg.ffn_dim];
                download(&self.buf_a, &mut gate);
                // up matmul
                let (ref up_buf, ref up_type, up_k, up_n) = layer.ffn_gate_buf.as_ref().unwrap();
                let up_d = Self::kquant_dispatch(&self.buf_c, up_buf, &self.buf_a, *up_type, *up_k, *up_n);
                engine.dispatch_by_name(up_d.0.as_str(), &up_d.1, &up_d.2, &up_d.3, up_d.4);
                let mut up = vec![0.0f32; cfg.ffn_dim];
                download(&self.buf_a, &mut up);
                // SiLU(gate) * up
                let mut ffn_mid = vec![0.0f32; cfg.ffn_dim];
                for i in 0..cfg.ffn_dim {
                    let silu = gate[i] / (1.0 + (-gate[i]).exp());
                    ffn_mid[i] = silu * up[i];
                }
                gpu_matmul_kquant(engine, &self.buf_a, &layer.ffn_down_buf, &self.buf_c,
                    layer.ffn_down_type, &ffn_mid, &mut tmp, layer.ffn_down_k, layer.ffn_down_n);
            } else {
                // Combined gate+up (Phi-4 style)
                let up_d = Self::kquant_dispatch(&self.buf_c, &layer.ffn_up_buf, &self.buf_a,
                    layer.ffn_up_type, layer.ffn_up_k, layer.ffn_up_n);
                {
                    let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.ffn_norm_buf, &self.buf_c];
                    let sizes1: Vec<u64> = vec![d as u64 * 4; 3];
                    engine.dispatch_batch(&[
                        ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                        (up_d.0.as_str(), &up_d.1, &up_d.2, &up_d.3, up_d.4),
                    ]);
                }
                let mut gate_up = vec![0.0f32; cfg.gate_up_dim()];
                download(&self.buf_a, &mut gate_up);
                let (gate, up) = gate_up.split_at_mut(cfg.ffn_dim);
                cpu_silu(gate);
                let mut ffn_mid = vec![0.0f32; cfg.ffn_dim];
                for i in 0..cfg.ffn_dim { ffn_mid[i] = gate[i] * up[i]; }
                gpu_matmul_kquant(engine, &self.buf_a, &layer.ffn_down_buf, &self.buf_c,
                    layer.ffn_down_type, &ffn_mid, &mut tmp, layer.ffn_down_k, layer.ffn_down_n);
            }
            for i in 0..cfg.d_model { hidden[i] += tmp[i]; }
        }

        cache.len += 1;

        // Output norm + logits
        let mut norm = vec![0.0f32; cfg.d_model];
        rms_norm(hidden, &self.output_norm, &mut norm);
        upload(&self.buf_a, &norm);
        let mk = 1u32; let kk = cfg.d_model as u32;
        let mut dispatches_data: Vec<(Vec<u8>, [u32; 3], usize)> = Vec::new();
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n as u32;
            let pb: Vec<u8> = [mk, kk, n].iter().flat_map(|v| v.to_le_bytes()).collect();
            dispatches_data.push((pb, [(n+15)/16, (mk+15)/16, 1], i));
        }
        let dsp: Vec<(&str, Vec<&GpuBuffer>, Vec<u64>, &[u8], [u32; 3])> = dispatches_data.iter()
            .map(|(pb, groups, i)| {
                let n = self.emb_bufs[*i].1;
                ("matmul_tiled",
                 vec![&self.buf_a, &self.emb_bufs[*i].0, &self.logit_out_bufs[*i]],
                 vec![(mk as u64)*(kk as u64)*4, (kk as u64)*(n as u64)*4, (mk as u64)*(n as u64)*4],
                 pb.as_slice(), *groups)
            }).collect();
        let refs: Vec<(&str, &[&GpuBuffer], &[u64], &[u8], [u32; 3])> = dsp.iter()
            .map(|(n, b, s, p, g)| (*n, b.as_slice(), s.as_slice(), *p, *g)).collect();
        engine.dispatch_batch_parallel(&refs);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        let mut vo = 0usize;
        for (i, (_, cn)) in self.emb_bufs.iter().enumerate() {
            let n = *cn;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            logits[vo..vo+n].copy_from_slice(src);
            vo += n;
        }
        logits
    }

    pub unsafe fn forward(&self, engine: &ComputeEngine, hidden: &mut Vec<f32>, cache: &mut KVCache) -> Vec<f32> {
        assert_eq!(hidden.len(), D_MODEL);
        let pos = cache.len;
        let mut qkv = vec![0.0f32; QKV_DIM];
        let mut gate_up = vec![0.0f32; GATE_UP_DIM];
        let mut ffn_mid = vec![0.0f32; FFN_DIM];
        let mut tmp = vec![0.0f32; D_MODEL];
        let heads_per_kv = N_HEADS / N_KV_HEADS;
        let d = D_MODEL as u32;
        let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();
        let kv_stride = N_KV_HEADS * HEAD_DIM;

        for (li, layer) in self.layers.iter().enumerate() {
            // ---- Batch 1: rmsnorm → QKV matmul (buf_a → buf_c → buf_a) ----
            upload(&self.buf_a, hidden);
            let qkv_d = Self::kquant_dispatch(
                &self.buf_c, &layer.attn_qkv_buf, &self.buf_a,
                layer.attn_qkv_type, layer.attn_qkv_k, layer.attn_qkv_n);
            {
                let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.attn_norm_buf, &self.buf_c];
                let sizes1: Vec<u64> = vec![d as u64 * 4, d as u64 * 4, d as u64 * 4];
                engine.dispatch_batch(&[
                    ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                    (qkv_d.0.as_str(), &qkv_d.1, &qkv_d.2, &qkv_d.3, qkv_d.4),
                ]);
            }
            download(&self.buf_a, &mut qkv[..QKV_DIM]);

            // CPU: split QKV, RoPE, update KV cache
            let q_end = N_HEADS * HEAD_DIM;
            let k_end = q_end + N_KV_HEADS * HEAD_DIM;
            let mut q = qkv[..q_end].to_vec();
            let mut k = qkv[q_end..k_end].to_vec();
            let v = qkv[k_end..].to_vec();
            apply_rope(&mut q, pos, N_HEADS, &self.rope_factors);
            apply_rope(&mut k, pos, N_KV_HEADS, &self.rope_factors);

            let kv_off = pos * kv_stride;
            let (ref kbuf, ref vbuf) = self.kv_bufs[li];
            std::ptr::copy_nonoverlapping(k.as_ptr() as *const u8,
                (kbuf.mapped as *mut u8).add(kv_off * 4), kv_stride * 4);
            std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8,
                (vbuf.mapped as *mut u8).add(kv_off * 4), kv_stride * 4);

            // ---- Batch 2: attention → output projection (dynamic desc, single submit) ----
            upload(&self.attn_q_buf, &q);
            let seq_len = pos + 1;
            let attn_push: Vec<u8> = [seq_len as u32, N_KV_HEADS as u32, heads_per_kv as u32, HEAD_DIM as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            let kv_bytes = (seq_len * kv_stride) as u64 * 4;
            let out_push: Vec<u8> = [layer.attn_output_k, layer.attn_output_n]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            {
                let abufs: Vec<&GpuBuffer> = vec![&self.attn_q_buf, kbuf, vbuf, &self.attn_out_buf];
                let asizes: Vec<u64> = vec![D_MODEL as u64 * 4, kv_bytes, kv_bytes, D_MODEL as u64 * 4];
                let obufs: Vec<&GpuBuffer> = vec![&self.attn_out_buf, &layer.attn_output_buf, &self.buf_c];
                let bpr = layer.attn_output_k as u64 / 256;
                let wb = layer.attn_output_n as u64 * bpr * layer.attn_output_type.block_bytes() as u64;
                let osizes: Vec<u64> = vec![layer.attn_output_k as u64 * 4, wb, layer.attn_output_n as u64 * 4];
                engine.dispatch_batch(&[
                    ("attention", &abufs, &asizes, &attn_push, [N_HEADS as u32, 1, 1]),
                    (layer.attn_output_type.shader_name(), &obufs, &osizes, &out_push, [layer.attn_output_n, 1, 1]),
                ]);
            }
            download(&self.buf_c, &mut tmp[..D_MODEL]);

            // CPU: attention residual
            for i in 0..D_MODEL { hidden[i] += tmp[i]; }

            // ---- Batch 3: rmsnorm → FFN up (buf_a → buf_c → buf_a) ----
            upload(&self.buf_a, hidden);
            let up_d = Self::kquant_dispatch(
                &self.buf_c, &layer.ffn_up_buf, &self.buf_a,
                layer.ffn_up_type, layer.ffn_up_k, layer.ffn_up_n);
            {
                let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.ffn_norm_buf, &self.buf_c];
                let sizes1: Vec<u64> = vec![d as u64 * 4, d as u64 * 4, d as u64 * 4];
                engine.dispatch_batch(&[
                    ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                    (up_d.0.as_str(), &up_d.1, &up_d.2, &up_d.3, up_d.4),
                ]);
            }
            download(&self.buf_a, &mut gate_up);

            // CPU: SiLU + gate * up
            let (gate, up) = gate_up.split_at_mut(FFN_DIM);
            cpu_silu(gate);
            for i in 0..FFN_DIM { ffn_mid[i] = gate[i] * up[i]; }

            // ---- Dispatch 4: FFN down (standalone) ----
            gpu_matmul_kquant(engine, &self.buf_a, &layer.ffn_down_buf, &self.buf_c,
                layer.ffn_down_type, &ffn_mid, &mut tmp, layer.ffn_down_k, layer.ffn_down_n);

            // CPU: FFN residual
            for i in 0..D_MODEL { hidden[i] += tmp[i]; }
        }

        cache.len += 1;

        // Output norm (CPU) + logits (GPU, single batched submit)
        let mut norm = vec![0.0f32; D_MODEL];
        rms_norm(hidden, &self.output_norm, &mut norm);
        upload(&self.buf_a, &norm);

        // Build batch of all logit chunk dispatches
        let m = 1u32;
        let k = D_MODEL as u32;
        let mut dispatches_data: Vec<(Vec<u8>, [u32; 3], usize)> = Vec::new();
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n as u32;
            let push: [u32; 3] = [m, k, n];
            let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
            dispatches_data.push((pb, [(n + 15) / 16, (m + 15) / 16, 1], i));
        }
        let dispatches: Vec<(&str, Vec<&GpuBuffer>, Vec<u64>, &[u8], [u32; 3])> = dispatches_data.iter()
            .map(|(pb, groups, i)| {
                let n = self.emb_bufs[*i].1;
                let emb_buf = &self.emb_bufs[*i].0;
                let out_buf = &self.logit_out_bufs[*i];
                (
                    "matmul_tiled",
                    vec![&self.buf_a, emb_buf, out_buf],
                    vec![(m as u64) * (k as u64) * 4, (k as u64) * (n as u64) * 4, (m as u64) * (n as u64) * 4],
                    pb.as_slice(),
                    *groups,
                )
            }).collect();
        // Convert to the slice-of-tuples format dispatch_batch expects
        let dispatch_refs: Vec<(&str, &[&GpuBuffer], &[u64], &[u8], [u32; 3])> = dispatches.iter()
            .map(|(name, bufs, sizes, push, groups)| (*name, bufs.as_slice(), sizes.as_slice(), *push, *groups))
            .collect();
        engine.dispatch_batch_parallel(&dispatch_refs);

        // Download all chunks
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let mut vocab_offset = 0usize;
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            logits[vocab_offset..vocab_offset + n].copy_from_slice(src);
            vocab_offset += n;
        }
        logits
    }

    /// Prefill multiple tokens: process all tokens through each layer together.
    /// Returns logits for the last token only.
    pub unsafe fn forward_prefill(
        &self, engine: &ComputeEngine, token_ids: &[u32], cache: &mut KVCache,
    ) -> Vec<f32> {
        let base_pos = cache.len;
        let m = token_ids.len();
        let heads_per_kv = N_HEADS / N_KV_HEADS;
        let d = D_MODEL as u32;

        // Embed all tokens
        let mut hiddens: Vec<Vec<f32>> = token_ids.iter()
            .map(|&tok| self.embed(tok)).collect();

        for (_li, layer) in self.layers.iter().enumerate() {
            let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();

            // For each token: rmsnorm+QKV (batched), RoPE, KV cache update
            let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut qs: Vec<Vec<f32>> = Vec::with_capacity(m);
            for t in 0..m {
                upload(&self.buf_a, &hiddens[t]);
                let qkv_d = Self::kquant_dispatch(
                    &self.buf_c, &layer.attn_qkv_buf, &self.buf_a,
                    layer.attn_qkv_type, layer.attn_qkv_k, layer.attn_qkv_n);
                {
                    let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.attn_norm_buf, &self.buf_c];
                    let sizes1: Vec<u64> = vec![d as u64 * 4, d as u64 * 4, d as u64 * 4];
                    engine.dispatch_batch(&[
                        ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                        (qkv_d.0.as_str(), &qkv_d.1, &qkv_d.2, &qkv_d.3, qkv_d.4),
                    ]);
                }
                let mut qkv = vec![0.0f32; QKV_DIM];
                download(&self.buf_a, &mut qkv);

                let pos = base_pos + t;
                let q_end = N_HEADS * HEAD_DIM;
                let k_end = q_end + N_KV_HEADS * HEAD_DIM;
                let mut q = qkv[..q_end].to_vec();
                let mut k = qkv[q_end..k_end].to_vec();
                let v = qkv[k_end..].to_vec();

                apply_rope(&mut q, pos, N_HEADS, &self.rope_factors);
                apply_rope(&mut k, pos, N_KV_HEADS, &self.rope_factors);

                let kv_stride = N_KV_HEADS * HEAD_DIM;
                let kv_off = pos * kv_stride;
                let (ref kbuf, ref vbuf) = self.kv_bufs[_li];
                std::ptr::copy_nonoverlapping(k.as_ptr() as *const u8,
                    (kbuf.mapped as *mut u8).add(kv_off * 4), kv_stride * 4);
                std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8,
                    (vbuf.mapped as *mut u8).add(kv_off * 4), kv_stride * 4);
                qs.push(q);
            }

            // Attention + output proj for each token
            for t in 0..m {
                let pos = base_pos + t;
                let seq_len = pos + 1;
                let kv_stride = N_KV_HEADS * HEAD_DIM;

                upload(&self.attn_q_buf, &qs[t]);
                let attn_push: Vec<u8> = [seq_len as u32, N_KV_HEADS as u32, heads_per_kv as u32, HEAD_DIM as u32]
                    .iter().flat_map(|v| v.to_le_bytes()).collect();
                let kv_bytes = (seq_len * kv_stride) as u64 * 4;
                let (ref kbuf, ref vbuf) = self.kv_bufs[_li];
                let out_d = Self::kquant_dispatch(
                    &self.attn_out_buf, &layer.attn_output_buf, &self.buf_c,
                    layer.attn_output_type, layer.attn_output_k, layer.attn_output_n);
                {
                    let abufs: Vec<&GpuBuffer> = vec![&self.attn_q_buf, kbuf, vbuf, &self.attn_out_buf];
                    let asizes: Vec<u64> = vec![D_MODEL as u64 * 4, kv_bytes, kv_bytes, D_MODEL as u64 * 4];
                    engine.dispatch_batch(&[
                        ("attention", &abufs, &asizes, &attn_push, [N_HEADS as u32, 1, 1]),
                        (out_d.0.as_str(), &out_d.1, &out_d.2, &out_d.3, out_d.4),
                    ]);
                }
                let mut tmp = vec![0.0f32; D_MODEL];
                download(&self.buf_c, &mut tmp);
                for i in 0..D_MODEL { hiddens[t][i] += tmp[i]; }
            }

            // FFN for each token: rmsnorm+up (batched) → SiLU → down
            for t in 0..m {
                upload(&self.buf_a, &hiddens[t]);
                let up_d = Self::kquant_dispatch(
                    &self.buf_c, &layer.ffn_up_buf, &self.buf_a,
                    layer.ffn_up_type, layer.ffn_up_k, layer.ffn_up_n);
                {
                    let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.ffn_norm_buf, &self.buf_c];
                    let sizes1: Vec<u64> = vec![d as u64 * 4, d as u64 * 4, d as u64 * 4];
                    engine.dispatch_batch(&[
                        ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                        (up_d.0.as_str(), &up_d.1, &up_d.2, &up_d.3, up_d.4),
                    ]);
                }
                let mut gate_up = vec![0.0f32; GATE_UP_DIM];
                download(&self.buf_a, &mut gate_up);

                let (gate, up) = gate_up.split_at_mut(FFN_DIM);
                cpu_silu(gate);
                let mut ffn_mid = vec![0.0f32; FFN_DIM];
                for i in 0..FFN_DIM { ffn_mid[i] = gate[i] * up[i]; }

                let mut tmp = vec![0.0f32; D_MODEL];
                gpu_matmul_kquant(engine, &self.buf_a, &layer.ffn_down_buf, &self.buf_c,
                    layer.ffn_down_type, &ffn_mid, &mut tmp, layer.ffn_down_k, layer.ffn_down_n);
                for i in 0..D_MODEL { hiddens[t][i] += tmp[i]; }
            }
        }

        cache.len += m;

        // Logits for last token only
        let mut norm = vec![0.0f32; D_MODEL];
        rms_norm(&hiddens[m - 1], &self.output_norm, &mut norm);
        upload(&self.buf_a, &norm);

        let mk = 1u32;
        let kk = D_MODEL as u32;
        let mut dispatches_data: Vec<(Vec<u8>, [u32; 3], usize)> = Vec::new();
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n as u32;
            let push: [u32; 3] = [mk, kk, n];
            let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
            dispatches_data.push((pb, [(n + 15) / 16, (mk + 15) / 16, 1], i));
        }
        let dispatches: Vec<(&str, Vec<&GpuBuffer>, Vec<u64>, &[u8], [u32; 3])> = dispatches_data.iter()
            .map(|(pb, groups, i)| {
                let n = self.emb_bufs[*i].1;
                ("matmul_tiled",
                 vec![&self.buf_a, &self.emb_bufs[*i].0, &self.logit_out_bufs[*i]],
                 vec![(mk as u64)*(kk as u64)*4, (kk as u64)*(n as u64)*4, (mk as u64)*(n as u64)*4],
                 pb.as_slice(), *groups)
            }).collect();
        let dispatch_refs: Vec<(&str, &[&GpuBuffer], &[u64], &[u8], [u32; 3])> = dispatches.iter()
            .map(|(name, bufs, sizes, push, groups)| (*name, bufs.as_slice(), sizes.as_slice(), *push, *groups))
            .collect();
        engine.dispatch_batch_parallel(&dispatch_refs);

        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let mut vocab_offset = 0usize;
        for (i, (_, chunk_n)) in self.emb_bufs.iter().enumerate() {
            let n = *chunk_n;
            let src = std::slice::from_raw_parts(self.logit_out_bufs[i].mapped as *const f32, n);
            logits[vocab_offset..vocab_offset + n].copy_from_slice(src);
            vocab_offset += n;
        }
        logits
    }

    pub fn embed(&self, token_id: u32) -> Vec<f32> {
        let d = self.config.d_model;
        let s = token_id as usize * d;
        self.embedding[s..s + d].to_vec()
    }

    pub unsafe fn generate(
        &self, engine: &ComputeEngine, prompt: &[u32], n_tokens: usize, vocab: Option<&[String]>,
    ) {
        println!("\n=== Phi-4 Mini (Native K-Quant, RoPE + KV Cache) ===");
        let mut cache = KVCache::new();
        let mut tokens = prompt.to_vec();

        let pt = Instant::now();
        for (i, &tok) in prompt.iter().enumerate() {
            let mut hidden = self.embed(tok);
            if i < prompt.len() - 1 {
                let _ = self.forward(engine, &mut hidden, &mut cache);
                eprintln!("[prefill] pos {}", i);
            } else {
                let logits = self.forward(engine, &mut hidden, &mut cache);
                let mut idx: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i,&s)|(i,s)).collect();
                idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
                println!("[top-5]");
                for r in 0..5 {
                    let s = vocab.map(|v| v.get(idx[r].0).map(|s| s.as_str()).unwrap_or("?")).unwrap_or("?");
                    println!("  #{}: {} \"{}\" = {:.2}", r+1, idx[r].0, s, idx[r].1);
                }
                tokens.push(idx[0].0 as u32);
                println!("[prefill {:.2}s]", pt.elapsed().as_secs_f32());
            }
        }

        for step in 0..n_tokens - 1 {
            let t0 = Instant::now();
            let mut hidden = self.embed(*tokens.last().unwrap());
            let logits = self.forward(engine, &mut hidden, &mut cache);
            let (bt, _) = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            tokens.push(bt as u32);
            let s = vocab.map(|v| v.get(bt).map(|s| s.as_str()).unwrap_or("?")).unwrap_or("?");
            println!("  [{}] \"{}\" ({:.1} tok/s)", step+1, s, 1.0/t0.elapsed().as_secs_f32());
        }

        if let Some(v) = vocab {
            let out: String = tokens[prompt.len()..].iter().filter_map(|&id| v.get(id as usize)).cloned().collect();
            println!("\n--- Output ---\n{}", out);
        }
        println!("=== Done ({} tokens) ===", tokens.len());
    }
}
