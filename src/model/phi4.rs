//! Phi-4 Mini forward pass — native K-quant shaders (Q4_K/Q5_K/Q6_K).
//! Route 4: Read GGUF blocks directly in shader. Zero requantization loss.
//! ~1.4GB GPU for 32 layers.

use crate::gpu::{ComputeEngine, GpuBuffer};
use crate::model::gguf::GGUFModel;
use std::time::Instant;

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

/// Quant type determines which shader + block size to use
#[derive(Clone, Copy)]
enum QType { Q4K, Q5K, Q6K }

impl QType {
    fn shader_name(&self) -> &str {
        match self { QType::Q4K => "matmul_q4k", QType::Q5K => "matmul_q5k", QType::Q6K => "matmul_q6k" }
    }
    fn block_bytes(&self) -> usize {
        match self { QType::Q4K => 144, QType::Q5K => 176, QType::Q6K => 210 }
    }
    fn from_gguf(typ: u32) -> Self {
        match typ { 12 => QType::Q4K, 13 => QType::Q5K, 14 => QType::Q6K, _ => panic!("Unsupported quant type {}", typ) }
    }
}

/// Per-layer weights — raw GGUF K-quant bytes on GPU
pub struct LayerWeights {
    pub attn_norm_buf: GpuBuffer,
    pub ffn_norm_buf: GpuBuffer,
    pub attn_qkv_buf: GpuBuffer,     pub attn_qkv_type: QType,     pub attn_qkv_k: u32,   pub attn_qkv_n: u32,
    pub attn_output_buf: GpuBuffer,   pub attn_output_type: QType,  pub attn_output_k: u32, pub attn_output_n: u32,
    pub ffn_up_buf: GpuBuffer,        pub ffn_up_type: QType,       pub ffn_up_k: u32,      pub ffn_up_n: u32,
    pub ffn_down_buf: GpuBuffer,      pub ffn_down_type: QType,     pub ffn_down_k: u32,    pub ffn_down_n: u32,
}

const MAX_SEQ: usize = 2048;

pub struct Phi4Model {
    pub embedding: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub buf_a: GpuBuffer,
    pub buf_c: GpuBuffer,
    pub logits_buf: GpuBuffer,
    pub emb_bufs: Vec<(GpuBuffer, usize)>,
    pub rope_factors: Vec<f32>,
    // GPU attention buffers
    pub attn_q_buf: GpuBuffer,     // [N_HEADS * HEAD_DIM]
    pub attn_out_buf: GpuBuffer,   // [N_HEADS * HEAD_DIM]
    pub kv_bufs: Vec<(GpuBuffer, GpuBuffer)>,  // per-layer (K_cache, V_cache) on GPU
    pub logit_out_bufs: Vec<GpuBuffer>,  // per-chunk logit output buffers for batching
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

const ROPE_ATTN_FACTOR: f32 = 1.190238;  // phi3.rope.scaling.attn_factor

fn apply_rope(x: &mut [f32], pos: usize, n_heads: usize, rope_factors: &[f32]) {
    for h in 0..n_heads {
        let base = h * HEAD_DIM;
        for i in 0..(ROPE_DIM / 2) {
            let base_freq = 1.0 / ROPE_THETA.powf(2.0 * i as f32 / ROPE_DIM as f32);
            let factor = if i < rope_factors.len() { rope_factors[i] } else { 1.0 };
            let freq = base_freq / factor;
            let angle = pos as f32 * freq;
            let (sin_t, cos_t) = angle.sin_cos();
            // LongRoPE: scale rotated dims by attn_factor (magnitude scaling)
            let cos_scaled = cos_t * ROPE_ATTN_FACTOR;
            let sin_scaled = sin_t * ROPE_ATTN_FACTOR;
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0 * cos_scaled - x1 * sin_scaled;
            x[base + 2 * i + 1] = x0 * sin_scaled + x1 * cos_scaled;
        }
    }
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

fn dequant_tensor_f32(gguf: &GGUFModel, name: &str) -> Vec<f32> {
    let info = gguf.tensor_infos.iter().find(|t| t.name == name).unwrap();
    let n_elements: usize = info.shape.iter().product::<u64>() as usize;
    let bytes = gguf.tensor_bytes(name);
    match info.typ {
        0 => unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_elements).to_vec() },
        1 => unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, n_elements).iter().map(|&b| f16_to_f32(b)).collect() },
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
    let push: [u32; 2] = [k, n];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    let weight_bytes = (n as u64 / 256) * (k as u64) / 256 * (qtype.block_bytes() as u64) * 256;
    // Actually: blocks_per_row = K/256, rows = N, total blocks = N * K/256
    let blocks_per_row = k as u64 / 256;
    let total_weight_bytes = n as u64 * blocks_per_row * qtype.block_bytes() as u64;
    engine.dispatch_by_name(
        qtype.shader_name(),
        &[buf_a, weight_buf, buf_c],
        &[k as u64 * 4, total_weight_bytes, n as u64 * 4],
        &pb,
        [n, 1, 1],
    );
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

        eprintln!("[phi4] Loading embedding...");
        let embedding = dequant_tensor_f32(gguf, "token_embd.weight");
        let output_norm = dequant_tensor_f32(gguf, "output_norm.weight");
        let rope_factors = dequant_tensor_f32(gguf, "rope_factors_short.weight");
        eprintln!("[phi4] Embedding + norms: {:.1}s", t0.elapsed().as_secs_f32());

        // Register pipelines
        engine.get_pipeline("matmul_q4k", 3, 8);
        engine.get_pipeline("matmul_q5k", 3, 8);
        engine.get_pipeline("matmul_q6k", 3, 8);
        engine.get_pipeline("matmul_tiled", 3, 12);
        engine.get_pipeline("rmsnorm", 3, 8);
        engine.get_pipeline("attention", 4, 12);  // 4 buffers, 3 push u32s

        let mut layers = Vec::with_capacity(N_LAYERS);
        let mut total_gpu: u64 = 0;

        for i in 0..N_LAYERS {
            let prefix = format!("blk.{}", i);

            // Upload raw GGUF bytes to GPU — NO dequant, NO transpose, NO requant
            let mut load_weight = |name: &str| -> (GpuBuffer, QType, u32, u32) {
                let info = gguf.tensor_infos.iter().find(|t| t.name == name).unwrap();
                let qtype = QType::from_gguf(info.typ);
                let ne0 = info.shape[0] as u32;  // K (input dim, fast)
                let ne1 = info.shape[1] as u32;  // N (output dim)
                let raw_bytes = gguf.tensor_bytes(name);
                let buf = engine.alloc_buffer(raw_bytes.len() as u64);
                upload_bytes(&buf, raw_bytes);
                total_gpu += raw_bytes.len() as u64;
                (buf, qtype, ne0, ne1)
            };

            let (qkv_buf, qkv_t, qkv_k, qkv_n) = load_weight(&format!("{}.attn_qkv.weight", prefix));
            let (out_buf, out_t, out_k, out_n) = load_weight(&format!("{}.attn_output.weight", prefix));
            let (up_buf, up_t, up_k, up_n) = load_weight(&format!("{}.ffn_up.weight", prefix));
            let (down_buf, down_t, down_k, down_n) = load_weight(&format!("{}.ffn_down.weight", prefix));

            let attn_norm = dequant_tensor_f32(gguf, &format!("{}.attn_norm.weight", prefix));
            let attn_norm_buf = engine.alloc_buffer(D_MODEL as u64 * 4);
            upload(&attn_norm_buf, &attn_norm);
            let ffn_norm = dequant_tensor_f32(gguf, &format!("{}.ffn_norm.weight", prefix));
            let ffn_norm_buf = engine.alloc_buffer(D_MODEL as u64 * 4);
            upload(&ffn_norm_buf, &ffn_norm);

            layers.push(LayerWeights {
                attn_norm_buf, ffn_norm_buf,
                attn_qkv_buf: qkv_buf, attn_qkv_type: qkv_t, attn_qkv_k: qkv_k, attn_qkv_n: qkv_n,
                attn_output_buf: out_buf, attn_output_type: out_t, attn_output_k: out_k, attn_output_n: out_n,
                ffn_up_buf: up_buf, ffn_up_type: up_t, ffn_up_k: up_k, ffn_up_n: up_n,
                ffn_down_buf: down_buf, ffn_down_type: down_t, ffn_down_k: down_k, ffn_down_n: down_n,
            });

            if i % 8 == 7 || i == N_LAYERS - 1 {
                eprintln!("[phi4] Layer {}: {:.1}s ({:.1}MB GPU)", i, t0.elapsed().as_secs_f32(), total_gpu as f64 / 1e6);
            }
        }

        let buf_a = engine.alloc_buffer((GATE_UP_DIM as u64) * 4);
        let buf_c = engine.alloc_buffer((GATE_UP_DIM as u64) * 4);
        let logits_buf = engine.alloc_buffer(8192 * 4);

        // F32 embedding chunks for logits (need transpose for matmul_tiled)
        let chunk_size: usize = 8192;
        let mut emb_bufs = Vec::new();
        let mut chunk_start = 0usize;
        while chunk_start < VOCAB_SIZE {
            let chunk_end = (chunk_start + chunk_size).min(VOCAB_SIZE);
            let n = chunk_end - chunk_start;
            let mut transposed = vec![0.0f32; D_MODEL * n];
            for d in 0..D_MODEL {
                for v in 0..n {
                    transposed[d * n + v] = embedding[(chunk_start + v) * D_MODEL + d];
                }
            }
            let buf = engine.alloc_buffer((D_MODEL * n) as u64 * 4);
            upload(&buf, &transposed);
            emb_bufs.push((buf, n));
            chunk_start = chunk_end;
        }
        // GPU attention buffers
        let attn_q_buf = engine.alloc_buffer((D_MODEL as u64) * 4);
        let attn_out_buf = engine.alloc_buffer((D_MODEL as u64) * 4);
        let kv_size = (MAX_SEQ * N_KV_HEADS * HEAD_DIM) as u64 * 4;
        let mut kv_bufs = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            let kb = engine.alloc_buffer(kv_size);
            let vb = engine.alloc_buffer(kv_size);
            kv_bufs.push((kb, vb));
        }
        total_gpu += N_LAYERS as u64 * kv_size * 2 + D_MODEL as u64 * 4 * 2;

        // Per-chunk logit output buffers for batched logits
        let mut logit_out_bufs = Vec::with_capacity(emb_bufs.len());
        for (_, n) in &emb_bufs {
            let buf = engine.alloc_buffer((*n as u64) * 4);
            logit_out_bufs.push(buf);
        }

        eprintln!("[phi4] Ready: {:.1}s, {:.1}MB total GPU ({} emb chunks)",
            t0.elapsed().as_secs_f32(), total_gpu as f64 / 1e6, emb_bufs.len());

        Phi4Model { embedding, output_norm, layers, buf_a, buf_c, logits_buf, emb_bufs, rope_factors,
                    attn_q_buf, attn_out_buf, kv_bufs, logit_out_bufs }
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
            [n, 1, 1],
        )
    }

    pub unsafe fn forward(&self, engine: &ComputeEngine, hidden: &mut Vec<f32>, cache: &mut KVCache) -> Vec<f32> {
        assert_eq!(hidden.len(), D_MODEL);
        let pos = cache.len;
        let mut qkv = vec![0.0f32; QKV_DIM];
        let mut attn_out = vec![0.0f32; D_MODEL];
        let mut gate_up = vec![0.0f32; GATE_UP_DIM];
        let mut ffn_mid = vec![0.0f32; FFN_DIM];
        let mut tmp = vec![0.0f32; D_MODEL];
        let heads_per_kv = N_HEADS / N_KV_HEADS;
        let d = D_MODEL as u32;

        for (li, layer) in self.layers.iter().enumerate() {
            // ---- Batch 1: rmsnorm → QKV matmul (buf_a → buf_c → buf_a) ----
            upload(&self.buf_a, hidden);
            let norm_push: Vec<u8> = [1u32, d].iter().flat_map(|v| v.to_le_bytes()).collect();
            let qkv_d = Self::kquant_dispatch(
                &self.buf_c, &layer.attn_qkv_buf, &self.buf_a,
                layer.attn_qkv_type, layer.attn_qkv_k, layer.attn_qkv_n);
            {
                let bufs1: Vec<&GpuBuffer> = vec![&self.buf_a, &layer.attn_norm_buf, &self.buf_c];
                let sizes1: Vec<u64> = vec![d as u64 * 4, d as u64 * 4, d as u64 * 4];
                let bufs2: Vec<&GpuBuffer> = qkv_d.1;
                let sizes2: Vec<u64> = qkv_d.2;
                engine.dispatch_batch(&[
                    ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                    (qkv_d.0.as_str(), &bufs2, &sizes2, &qkv_d.3, qkv_d.4),
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

            let kv_stride = N_KV_HEADS * HEAD_DIM;
            let kv_off = pos * kv_stride;
            let (ref kbuf, ref vbuf) = self.kv_bufs[li];
            std::ptr::copy_nonoverlapping(k.as_ptr() as *const u8,
                (kbuf.mapped as *mut u8).add(kv_off * 4), kv_stride * 4);
            std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8,
                (vbuf.mapped as *mut u8).add(kv_off * 4), kv_stride * 4);

            // ---- Batch 2: attention → output projection (attn_q→attn_out→buf_c) ----
            upload(&self.attn_q_buf, &q);
            let seq_len = pos + 1;
            let attn_push: Vec<u8> = [seq_len as u32, N_KV_HEADS as u32, heads_per_kv as u32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            let kv_bytes = (seq_len * kv_stride) as u64 * 4;
            let out_d = Self::kquant_dispatch(
                &self.attn_out_buf, &layer.attn_output_buf, &self.buf_c,
                layer.attn_output_type, layer.attn_output_k, layer.attn_output_n);
            {
                let abufs: Vec<&GpuBuffer> = vec![&self.attn_q_buf, kbuf, vbuf, &self.attn_out_buf];
                let asizes: Vec<u64> = vec![D_MODEL as u64 * 4, kv_bytes, kv_bytes, D_MODEL as u64 * 4];
                let obufs: Vec<&GpuBuffer> = out_d.1;
                let osizes: Vec<u64> = out_d.2;
                engine.dispatch_batch(&[
                    ("attention", &abufs, &asizes, &attn_push, [N_HEADS as u32, 1, 1]),
                    (out_d.0.as_str(), &obufs, &osizes, &out_d.3, out_d.4),
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
                let bufs2: Vec<&GpuBuffer> = up_d.1;
                let sizes2: Vec<u64> = up_d.2;
                engine.dispatch_batch(&[
                    ("rmsnorm", &bufs1, &sizes1, &norm_push, [1, 1, 1]),
                    (up_d.0.as_str(), &bufs2, &sizes2, &up_d.3, up_d.4),
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

            // For each token: rmsnorm+QKV (batched pair), RoPE, KV cache update
            let mut qs: Vec<Vec<f32>> = Vec::with_capacity(m);
            for t in 0..m {
                // Batch 1: rmsnorm → QKV
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
                let attn_push: Vec<u8> = [seq_len as u32, N_KV_HEADS as u32, heads_per_kv as u32]
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

            // FFN for each token: rmsnorm+up → SiLU → down
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
        let s = token_id as usize * D_MODEL;
        self.embedding[s..s + D_MODEL].to_vec()
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
