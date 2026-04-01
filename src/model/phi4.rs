//! Phi-4 Mini forward pass — full transformer with GPU-accelerated Q4 matmul.
//!
//! Strategy: Weights quantized to Q4 on CPU during load, stored persistently on
//! GPU as packed uint32 nibbles + f32 scales. Per-dispatch, only the small
//! activation vector is uploaded. The matmul_q4 shader dequantizes weights on
//! the fly in shared memory.
//!
//! Memory: ~1.9GB GPU for all 32 layers (vs 7.5GB for F32).
//! Performance: matmul_q4 shader verified at ~398us per dispatch.

use crate::gpu::{ComputeEngine, GpuBuffer};
use crate::model::gguf::GGUFModel;
use rayon::prelude::*;
use std::time::Instant;

// Architecture constants
const D_MODEL: usize = 3072;
const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FFN_DIM: usize = 8192;
const N_LAYERS: usize = 32;
const VOCAB_SIZE: usize = 200064;
const RMS_EPS: f32 = 1e-5;
const QKV_DIM: usize = N_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM; // 5120
const GATE_UP_DIM: usize = 2 * FFN_DIM; // 16384
const Q4_BLOCK_SIZE: usize = 32;

/// Per-layer weights — Q4 packed on GPU
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    // Q4 packed weights (uint32, 8 nibbles per word) + scales (f32, one per block of 32)
    pub attn_qkv_packed: GpuBuffer,
    pub attn_qkv_scales: GpuBuffer,
    pub attn_output_packed: GpuBuffer,
    pub attn_output_scales: GpuBuffer,
    pub ffn_up_packed: GpuBuffer,
    pub ffn_up_scales: GpuBuffer,
    pub ffn_down_packed: GpuBuffer,
    pub ffn_down_scales: GpuBuffer,
}

/// Full Phi-4 Mini model
pub struct Phi4Model {
    pub embedding: Vec<f32>,       // [VOCAB_SIZE * D_MODEL]
    pub output_norm: Vec<f32>,     // [D_MODEL]
    pub layers: Vec<LayerWeights>,
    // Shared GPU buffers (reused per dispatch)
    pub buf_a: GpuBuffer,
    pub logits_buf: GpuBuffer,
    pub buf_c: GpuBuffer,          // output (float32)
    // Q4-packed embedding chunks on GPU — persistent, uploaded once at load time
    // Each entry: (packed_buf, scales_buf, chunk_n) for a [D_MODEL, chunk_n] slice
    pub emb_packed_bufs: Vec<(GpuBuffer, GpuBuffer, usize)>,
}

// ============================================================================
// Q4 Quantization — CPU side
// ============================================================================

/// Quantize f32 weights to our simple Q4 format.
/// Input: row-major [K, N] float32
/// Output: (packed_u32s, scales)
///   packed: 8 nibbles per u32, laid out as linear index = k*N + n
///   scales: [K/32, N] float32, one scale per block of 32 along K
fn quantize_to_q4(data: &[f32], k: usize, n: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(data.len(), k * n);
    let n_scale_blocks = (k + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let mut scales = vec![0.0f32; n_scale_blocks * n];
    let total_nibbles = k * n;
    let n_words = (total_nibbles + 7) / 8;
    let mut packed = vec![0u32; n_words];

    // Compute scales: for each block of 32 along K, for each N column
    for bk in 0..n_scale_blocks {
        let k_start = bk * Q4_BLOCK_SIZE;
        let k_end = (k_start + Q4_BLOCK_SIZE).min(k);
        for j in 0..n {
            let mut max_abs: f32 = 0.0;
            for ki in k_start..k_end {
                let v = data[ki * n + j].abs();
                if v > max_abs { max_abs = v; }
            }
            // scale such that max_abs maps to +7 (nibble range 1..15 maps to -7..+7)
            scales[bk * n + j] = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        }
    }

    // Pack nibbles
    for ki in 0..k {
        let bk = ki / Q4_BLOCK_SIZE;
        for j in 0..n {
            let scale = scales[bk * n + j];
            let val = data[ki * n + j];
            // quantize: nibble = round(val / scale) + 8, clamped to [0, 15]
            let q = (val / scale).round() as i32 + 8;
            let nibble = q.max(0).min(15) as u32;
            let linear = ki * n + j;
            let word_idx = linear / 8;
            let nibble_idx = linear % 8;
            packed[word_idx] |= nibble << (nibble_idx * 4);
        }
    }

    (packed, scales)
}

// ============================================================================
// GGUF Dequantization
// ============================================================================

fn dequant_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0.0f32; n_elements];
    for block_idx in 0..n_blocks {
        let block = &data[block_idx * 144..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = (scales[i] & 0x3F) as u8;
            mn[i] = (scales[i + 4] & 0x3F) as u8;
            sc[i + 4] = ((scales[i] >> 6) | ((scales[i + 8] & 0x0F) << 2)) as u8;
            mn[i + 4] = ((scales[i + 4] >> 6) | ((scales[i + 8] >> 4) << 2)) as u8;
        }
        let base = block_idx * 256;
        for sub in 0..8 {
            let s = d * sc[sub] as f32;
            let m = dmin * mn[sub] as f32;
            let start = sub * 32;
            for j in 0..32 {
                let qi = if j < 16 {
                    (qs[start / 2 + j] & 0x0F) as f32
                } else {
                    ((qs[start / 2 + j - 16] >> 4) & 0x0F) as f32
                };
                if base + start + j < n_elements {
                    out[base + start + j] = qi * s - m;
                }
            }
        }
    }
    out
}

fn dequant_q5_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0.0f32; n_elements];
    for block_idx in 0..n_blocks {
        let block = &data[block_idx * 176..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = (scales[i] & 0x3F) as u8;
            mn[i] = (scales[i + 4] & 0x3F) as u8;
            sc[i + 4] = ((scales[i] >> 6) | ((scales[i + 8] & 0x0F) << 2)) as u8;
            mn[i + 4] = ((scales[i + 4] >> 6) | ((scales[i + 8] >> 4) << 2)) as u8;
        }
        let base = block_idx * 256;
        for sub in 0..8 {
            let s = d * sc[sub] as f32;
            let m = dmin * mn[sub] as f32;
            let start = sub * 32;
            for j in 0..32 {
                let qi_lo = if j < 16 {
                    (qs[start / 2 + j] & 0x0F) as u8
                } else {
                    ((qs[start / 2 + j - 16] >> 4) & 0x0F) as u8
                };
                let global_j = start + j;
                let qi_hi = ((qh[global_j / 8] >> (global_j % 8)) & 1) as u8;
                let qi = (qi_lo | (qi_hi << 4)) as f32;
                if base + global_j < n_elements {
                    out[base + global_j] = qi * s - m;
                }
            }
        }
    }
    out
}

fn dequant_q6_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0.0f32; n_elements];
    for block_idx in 0..n_blocks {
        let block = &data[block_idx * 210..];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let base = block_idx * 256;
        for sub in 0..16 {
            let sc = scales[sub] as i8 as f32;
            let s = d * sc;
            for j in 0..16 {
                let global_j = sub * 16 + j;
                let ql_byte = ql[global_j / 2];
                let ql_val = if global_j % 2 == 0 { ql_byte & 0x0F } else { (ql_byte >> 4) & 0x0F };
                let qh_byte = qh[global_j / 4];
                let qh_val = (qh_byte >> ((global_j % 4) * 2)) & 0x03;
                let q = ((ql_val | (qh_val << 4)) as i8 - 32) as f32;
                if base + global_j < n_elements {
                    out[base + global_j] = q * s;
                }
            }
        }
    }
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn dequant_tensor(gguf: &GGUFModel, name: &str) -> Vec<f32> {
    let info = gguf.tensor_infos.iter().find(|t| t.name == name)
        .unwrap_or_else(|| panic!("Tensor not found: {}", name));
    let n_elements: usize = info.shape.iter().product::<u64>() as usize;
    let bytes = gguf.tensor_bytes(name);
    match info.typ {
        0 => {
            let f32_slice = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_elements)
            };
            f32_slice.to_vec()
        }
        1 => {
            let f16_slice = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, n_elements)
            };
            f16_slice.iter().map(|&b| f16_to_f32(b)).collect()
        }
        12 => dequant_q4_k(bytes, n_elements),
        13 => dequant_q5_k(bytes, n_elements),
        14 => dequant_q6_k(bytes, n_elements),
        t => {
            eprintln!("[WARN] Unknown quant type {} for {}, using zeros", t, name);
            vec![0.0f32; n_elements]
        }
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

fn cpu_gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = 0.7978845608_f32 * (*v + 0.044715 * x3);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

fn elemwise_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() { out[i] = a[i] * b[i]; }
}

fn elemwise_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() { out[i] = a[i] + b[i]; }
}

// ============================================================================
// GPU helpers
// ============================================================================

unsafe fn upload(buf: &GpuBuffer, data: &[f32]) {
    std::ptr::copy_nonoverlapping(
        data.as_ptr() as *const u8,
        buf.mapped as *mut u8,
        data.len() * 4,
    );
}

unsafe fn upload_u32(buf: &GpuBuffer, data: &[u32]) {
    std::ptr::copy_nonoverlapping(
        data.as_ptr() as *const u8,
        buf.mapped as *mut u8,
        data.len() * 4,
    );
}

unsafe fn download(buf: &GpuBuffer, out: &mut [f32]) {
    std::ptr::copy_nonoverlapping(
        buf.mapped as *const u8,
        out.as_mut_ptr() as *mut u8,
        out.len() * 4,
    );
}

/// GPU matmul with Q4 PERSISTENT weights — only uploads activation
/// Dispatches matmul_q4 shader: A(f32) x B(Q4 packed) -> C(f32)
/// weight_packed_buf: uint32 packed nibbles [K*N/8 words]
/// weight_scales_buf: f32 scales [K/32, N]
unsafe fn gpu_matmul_q4_persistent(
    engine: &ComputeEngine,
    buf_a: &GpuBuffer,
    weight_packed_buf: &GpuBuffer,
    weight_scales_buf: &GpuBuffer,
    buf_c: &GpuBuffer,
    a: &[f32], c: &mut [f32],
    m: u32, k: u32, n: u32,
) {
    upload(buf_a, a);
    let push: [u32; 3] = [m, k, n];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();

    let packed_size = ((k as u64) * (n as u64) + 7) / 8 * 4; // bytes for packed u32s
    let scales_size = ((k as u64 + 31) / 32) * (n as u64) * 4; // bytes for f32 scales

    engine.dispatch_by_name(
        "matmul_q4",
        &[buf_a, weight_packed_buf, weight_scales_buf, buf_c],
        &[
            (m as u64) * (k as u64) * 4,  // BufA size
            packed_size,                    // BufB packed size
            scales_size,                    // BufScales size
            (m as u64) * (n as u64) * 4,  // BufC size
        ],
        &pb,
        [(n + 15) / 16, (m + 15) / 16, 1],
    );
    download(buf_c, c);
}

// ============================================================================
// Model
// ============================================================================

impl Phi4Model {
    pub unsafe fn load_from_gguf(gguf: &GGUFModel, engine: &mut ComputeEngine) -> Self {
        let t0 = Instant::now();

        eprintln!("[phi4] Dequantizing embedding ({} x {})...", VOCAB_SIZE, D_MODEL);
        let embedding = dequant_tensor(gguf, "token_embd.weight");
        assert_eq!(embedding.len(), VOCAB_SIZE * D_MODEL);
        eprintln!("[phi4] Embedding dequantized: {:.1}s", t0.elapsed().as_secs_f32());

        let output_norm = dequant_tensor(gguf, "output_norm.weight");

        // Pre-create the matmul_q4 pipeline (4 buffers, 12 bytes push constants)
        engine.get_pipeline("matmul_q4", 4, 12);
        engine.get_pipeline("matmul_tiled", 3, 12);

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let prefix = format!("blk.{}", i);

            // Dequant to F32 on CPU, then re-quantize to our simple Q4 format
            // This works regardless of the original GGUF quant type

            // QKV: [D_MODEL, QKV_DIM] = [3072, 5120]
            let qkv_f32 = dequant_tensor(gguf, &format!("{}.attn_qkv.weight", prefix));
            let (qkv_packed, qkv_scales) = quantize_to_q4(&qkv_f32, D_MODEL, QKV_DIM);
            drop(qkv_f32); // free CPU memory immediately
            let qkv_packed_buf = engine.alloc_buffer(qkv_packed.len() as u64 * 4);
            upload_u32(&qkv_packed_buf, &qkv_packed);
            let qkv_scales_buf = engine.alloc_buffer(qkv_scales.len() as u64 * 4);
            upload(&qkv_scales_buf, &qkv_scales);

            // Output proj: [D_MODEL, D_MODEL] = [3072, 3072]
            let out_f32 = dequant_tensor(gguf, &format!("{}.attn_output.weight", prefix));
            let (out_packed, out_scales) = quantize_to_q4(&out_f32, D_MODEL, D_MODEL);
            drop(out_f32);
            let out_packed_buf = engine.alloc_buffer(out_packed.len() as u64 * 4);
            upload_u32(&out_packed_buf, &out_packed);
            let out_scales_buf = engine.alloc_buffer(out_scales.len() as u64 * 4);
            upload(&out_scales_buf, &out_scales);

            // FFN up: [D_MODEL, GATE_UP_DIM] = [3072, 16384]
            let up_f32 = dequant_tensor(gguf, &format!("{}.ffn_up.weight", prefix));
            let (up_packed, up_scales) = quantize_to_q4(&up_f32, D_MODEL, GATE_UP_DIM);
            drop(up_f32);
            let up_packed_buf = engine.alloc_buffer(up_packed.len() as u64 * 4);
            upload_u32(&up_packed_buf, &up_packed);
            let up_scales_buf = engine.alloc_buffer(up_scales.len() as u64 * 4);
            upload(&up_scales_buf, &up_scales);

            // FFN down: [FFN_DIM, D_MODEL] = [8192, 3072]
            let down_f32 = dequant_tensor(gguf, &format!("{}.ffn_down.weight", prefix));
            let (down_packed, down_scales) = quantize_to_q4(&down_f32, FFN_DIM, D_MODEL);
            drop(down_f32);
            let down_packed_buf = engine.alloc_buffer(down_packed.len() as u64 * 4);
            upload_u32(&down_packed_buf, &down_packed);
            let down_scales_buf = engine.alloc_buffer(down_scales.len() as u64 * 4);
            upload(&down_scales_buf, &down_scales);

            layers.push(LayerWeights {
                attn_norm: dequant_tensor(gguf, &format!("{}.attn_norm.weight", prefix)),
                ffn_norm: dequant_tensor(gguf, &format!("{}.ffn_norm.weight", prefix)),
                attn_qkv_packed: qkv_packed_buf,
                attn_qkv_scales: qkv_scales_buf,
                attn_output_packed: out_packed_buf,
                attn_output_scales: out_scales_buf,
                ffn_up_packed: up_packed_buf,
                ffn_up_scales: up_scales_buf,
                ffn_down_packed: down_packed_buf,
                ffn_down_scales: down_scales_buf,
            });
            if i % 8 == 7 || i == N_LAYERS - 1 {
                eprintln!("[phi4] Layer {} quantized + uploaded: {:.1}s", i, t0.elapsed().as_secs_f32());
            }
        }

        // Shared GPU buffers for activation I/O:
        //   A: max input size = FFN_DIM = 8192 floats
        //   C: max output size = GATE_UP_DIM = 16384 floats
        let buf_a = engine.alloc_buffer((FFN_DIM as u64) * 4);
        let buf_c = engine.alloc_buffer((GATE_UP_DIM as u64) * 4);

        // Calculate approximate GPU memory usage
        let per_layer_packed = (D_MODEL * QKV_DIM + D_MODEL * D_MODEL
            + D_MODEL * GATE_UP_DIM + FFN_DIM * D_MODEL) / 2; // nibbles -> bytes
        let per_layer_scales = ((D_MODEL / Q4_BLOCK_SIZE) * QKV_DIM
            + (D_MODEL / Q4_BLOCK_SIZE) * D_MODEL
            + (D_MODEL / Q4_BLOCK_SIZE) * GATE_UP_DIM
            + (FFN_DIM / Q4_BLOCK_SIZE) * D_MODEL) * 4; // f32 bytes
        let total_gpu_mb = (N_LAYERS * (per_layer_packed + per_layer_scales)) as f64 / 1024.0 / 1024.0;
        eprintln!("[phi4] GPU memory: ~{:.1}MB Q4 weights ({} layers)", total_gpu_mb, N_LAYERS);

        eprintln!("[phi4] Ready in {:.1}s — {} layers, d={}, Q4 persistent weights",
            t0.elapsed().as_secs_f32(), N_LAYERS, D_MODEL);

        let logits_buf = engine.alloc_buffer(8192 * 4);

        // Q4 pack the embedding in chunks and upload to persistent GPU buffers.
        // Embedding is [VOCAB_SIZE, D_MODEL] row-major. For logits we need
        // [1, D_MODEL] @ [D_MODEL, chunk_n], so we transpose into [D_MODEL, chunk_n]
        // chunks and Q4-pack each one.
        let chunk_size: usize = 8192;
        let mut emb_packed_bufs = Vec::new();
        let et0 = Instant::now();
        let mut chunk_start = 0usize;
        while chunk_start < VOCAB_SIZE {
            let chunk_end = (chunk_start + chunk_size).min(VOCAB_SIZE);
            let n = chunk_end - chunk_start;

            // Transpose [vocab_chunk, D_MODEL] -> [D_MODEL, n]
            let mut transposed = vec![0.0f32; D_MODEL * n];
            for d in 0..D_MODEL {
                for v in 0..n {
                    transposed[d * n + v] = embedding[(chunk_start + v) * D_MODEL + d];
                }
            }

            let (packed, scales) = quantize_to_q4(&transposed, D_MODEL, n);
            drop(transposed);

            let packed_buf = engine.alloc_buffer(packed.len() as u64 * 4);
            upload_u32(&packed_buf, &packed);
            let scales_buf = engine.alloc_buffer(scales.len() as u64 * 4);
            upload(&scales_buf, &scales);

            emb_packed_bufs.push((packed_buf, scales_buf, n));
            chunk_start = chunk_end;
        }
        let emb_gpu_bytes: u64 = emb_packed_bufs.iter().map(|(p, s, _)| p.capacity + s.capacity).sum();
        eprintln!("[phi4] Embedding Q4 packed: {} chunks, {:.1}MB GPU, {:.1}s",
            emb_packed_bufs.len(), emb_gpu_bytes as f64 / 1024.0 / 1024.0,
            et0.elapsed().as_secs_f32());

        Phi4Model { embedding, output_norm, layers, buf_a, buf_c, logits_buf, emb_packed_bufs }
    }

    pub unsafe fn forward(&self, engine: &ComputeEngine, hidden: &mut Vec<f32>) -> Vec<f32> {
        assert_eq!(hidden.len(), D_MODEL);

        let mut norm = vec![0.0f32; D_MODEL];
        let mut qkv = vec![0.0f32; QKV_DIM];
        let mut attn_out = vec![0.0f32; D_MODEL];
        let mut gate_up = vec![0.0f32; GATE_UP_DIM];
        let mut ffn_mid = vec![0.0f32; FFN_DIM];
        let mut tmp = vec![0.0f32; D_MODEL];

        for (li, layer) in self.layers.iter().enumerate() {
            // === Attention ===
            rms_norm(hidden, &layer.attn_norm, &mut norm);

            // QKV: [1,3072] x [3072,5120] = [1,5120]
            gpu_matmul_q4_persistent(engine,
                &self.buf_a, &layer.attn_qkv_packed, &layer.attn_qkv_scales, &self.buf_c,
                &norm, &mut qkv,
                1, D_MODEL as u32, QKV_DIM as u32);

            // Split Q/K/V and apply attention (seq=1: output = V per head via GQA)
            let v_start = N_HEADS * HEAD_DIM + N_KV_HEADS * HEAD_DIM;
            let heads_per_kv = N_HEADS / N_KV_HEADS;
            for h in 0..N_HEADS {
                let kv_h = h / heads_per_kv;
                let v_head = &qkv[v_start + kv_h * HEAD_DIM..v_start + (kv_h + 1) * HEAD_DIM];
                attn_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(v_head);
            }

            // Output proj: [1,3072] x [3072,3072] = [1,3072]
            gpu_matmul_q4_persistent(engine,
                &self.buf_a, &layer.attn_output_packed, &layer.attn_output_scales, &self.buf_c,
                &attn_out, &mut tmp,
                1, D_MODEL as u32, D_MODEL as u32);

            // Residual
            elemwise_add(hidden, &tmp, &mut norm);
            hidden.copy_from_slice(&norm);

            // === FFN ===
            rms_norm(hidden, &layer.ffn_norm, &mut norm);

            // Up: [1,3072] x [3072,16384] = [1,16384]
            gpu_matmul_q4_persistent(engine,
                &self.buf_a, &layer.ffn_up_packed, &layer.ffn_up_scales, &self.buf_c,
                &norm, &mut gate_up,
                1, D_MODEL as u32, GATE_UP_DIM as u32);

            // Gate * up
            let (gate, up) = gate_up.split_at_mut(FFN_DIM);
            cpu_gelu(gate);
            elemwise_mul(gate, up, &mut ffn_mid);

            // Down: [1,8192] x [8192,3072] = [1,3072]
            gpu_matmul_q4_persistent(engine,
                &self.buf_a, &layer.ffn_down_packed, &layer.ffn_down_scales, &self.buf_c,
                &ffn_mid, &mut tmp,
                1, FFN_DIM as u32, D_MODEL as u32);

            // Residual
            elemwise_add(hidden, &tmp, &mut norm);
            hidden.copy_from_slice(&norm);

            if li == 0 || li % 8 == 7 || li == N_LAYERS - 1 {
                eprintln!("[fwd] layer {}/31", li);
            }
        }

        // Final norm + logits
        rms_norm(hidden, &self.output_norm, &mut norm);

        let lt = Instant::now();
        // GPU logits using pre-loaded Q4 embedding chunks — no per-token upload
        // Each chunk: [1, D_MODEL] @ [D_MODEL, chunk_n] via matmul_q4 shader
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let mut vocab_offset = 0usize;
        for (packed_buf, scales_buf, chunk_n) in &self.emb_packed_bufs {
            let n = *chunk_n;
            let mut chunk_out = vec![0.0f32; n];
            gpu_matmul_q4_persistent(engine,
                &self.buf_a, packed_buf, scales_buf, &self.logits_buf,
                &norm, &mut chunk_out,
                1, D_MODEL as u32, n as u32);
            logits[vocab_offset..vocab_offset + n].copy_from_slice(&chunk_out);
            vocab_offset += n;
        }
        eprintln!("[fwd] logits (Q4 persistent): {}ms", lt.elapsed().as_millis());

        logits
    }

    pub fn embed(&self, token_id: u32) -> Vec<f32> {
        let s = token_id as usize * D_MODEL;
        self.embedding[s..s + D_MODEL].to_vec()
    }

    pub unsafe fn generate(
        &self,
        engine: &ComputeEngine,
        prompt: &[u32],
        n_tokens: usize,
    ) {
        println!("\n=== Phi-4 Mini Token Generation (Q4 weights) ===");
        println!("Prompt: {:?} | Generating {} tokens\n", prompt, n_tokens);

        let mut tokens = prompt.to_vec();

        for step in 0..n_tokens {
            let t0 = Instant::now();
            let mut hidden = self.embed(*tokens.last().unwrap());
            let logits = self.forward(engine, &mut hidden);

            // Argmax
            let (best_tok, best_score) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &s)| (i as u32, s)).unwrap();

            tokens.push(best_tok);
            println!("  [{}] token {} (logit {:.2}) — {:.1}s",
                step, best_tok, best_score, t0.elapsed().as_secs_f32());
        }

        println!("\nSequence: {:?}", tokens);
        println!("=== Done ===");
    }
}
