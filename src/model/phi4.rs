//! Phi-4 Mini forward pass — full transformer with GPU-accelerated matmul.
//!
//! Strategy: Weights dequantized to CPU Vec<f32>. Per-dispatch, upload the small
//! activation vector + the weight matrix to shared GPU buffers, dispatch matmul,
//! download result. CPU handles everything else (fast at seq=1).
//!
//! The weight upload IS the bottleneck (~200MB memcpy for ffn_up), but it proves
//! the complete pipeline. Next step: device-local buffers or keep weights on GPU
//! with double-buffering.

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

/// Per-layer weights (CPU, dequantized to f32)
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,       // [D_MODEL]
    pub attn_qkv: Vec<f32>,       // [D_MODEL * QKV_DIM]
    pub attn_output: Vec<f32>,    // [D_MODEL * D_MODEL]
    pub ffn_norm: Vec<f32>,        // [D_MODEL]
    pub ffn_up: Vec<f32>,         // [D_MODEL * GATE_UP_DIM]
    pub ffn_down: Vec<f32>,       // [FFN_DIM * D_MODEL]
}

/// Full Phi-4 Mini model
pub struct Phi4Model {
    pub embedding: Vec<f32>,       // [VOCAB_SIZE * D_MODEL]
    pub output_norm: Vec<f32>,     // [D_MODEL]
    pub layers: Vec<LayerWeights>,
    // Shared GPU buffers (reused per dispatch)
    pub buf_a: GpuBuffer,          // activation input
    pub buf_b: GpuBuffer,          // weight matrix
    pub buf_c: GpuBuffer,          // output
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

unsafe fn download(buf: &GpuBuffer, out: &mut [f32]) {
    std::ptr::copy_nonoverlapping(
        buf.mapped as *const u8,
        out.as_mut_ptr() as *mut u8,
        out.len() * 4,
    );
}

/// GPU matmul: C[m,n] = A[m,k] * B[k,n]
unsafe fn gpu_matmul(
    engine: &ComputeEngine,
    buf_a: &GpuBuffer, buf_b: &GpuBuffer, buf_c: &GpuBuffer,
    a: &[f32], b: &[f32], c: &mut [f32],
    m: u32, k: u32, n: u32,
) {
    upload(buf_a, a);
    upload(buf_b, b);
    let push: [u32; 3] = [m, k, n];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name(
        "matmul_tiled",
        &[buf_a, buf_b, buf_c],
        &[(m * k) as u64 * 4, (k * n) as u64 * 4, (m * n) as u64 * 4],
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

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let prefix = format!("blk.{}", i);
            layers.push(LayerWeights {
                attn_norm: dequant_tensor(gguf, &format!("{}.attn_norm.weight", prefix)),
                attn_qkv: dequant_tensor(gguf, &format!("{}.attn_qkv.weight", prefix)),
                attn_output: dequant_tensor(gguf, &format!("{}.attn_output.weight", prefix)),
                ffn_norm: dequant_tensor(gguf, &format!("{}.ffn_norm.weight", prefix)),
                ffn_up: dequant_tensor(gguf, &format!("{}.ffn_up.weight", prefix)),
                ffn_down: dequant_tensor(gguf, &format!("{}.ffn_down.weight", prefix)),
            });
            if i % 8 == 7 || i == N_LAYERS - 1 {
                eprintln!("[phi4] Layer {} dequantized: {:.1}s", i, t0.elapsed().as_secs_f32());
            }
        }

        engine.get_pipeline("matmul_tiled", 3, 12);

        // Shared GPU buffers — largest needed per role:
        //   A (activation input): max FFN_DIM = 8192 floats
        //   B (weight): max D_MODEL * GATE_UP_DIM = 50,331,648 floats (~192MB)
        //   C (output): max GATE_UP_DIM = 16384 floats
        let buf_a = engine.alloc_buffer((FFN_DIM as u64) * 4);
        let buf_b = engine.alloc_buffer((D_MODEL as u64) * (GATE_UP_DIM as u64) * 4);
        let buf_c = engine.alloc_buffer((GATE_UP_DIM as u64) * 4);

        eprintln!("[phi4] Ready in {:.1}s — {} layers, d={}",
            t0.elapsed().as_secs_f32(), N_LAYERS, D_MODEL);

        Phi4Model { embedding, output_norm, layers, buf_a, buf_b, buf_c }
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
            gpu_matmul(engine, &self.buf_a, &self.buf_b, &self.buf_c,
                &norm, &layer.attn_qkv, &mut qkv,
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
            gpu_matmul(engine, &self.buf_a, &self.buf_b, &self.buf_c,
                &attn_out, &layer.attn_output, &mut tmp,
                1, D_MODEL as u32, D_MODEL as u32);

            // Residual
            elemwise_add(hidden, &tmp, &mut norm);
            hidden.copy_from_slice(&norm);

            // === FFN ===
            rms_norm(hidden, &layer.ffn_norm, &mut norm);

            // Up: [1,3072] x [3072,16384] = [1,16384]
            gpu_matmul(engine, &self.buf_a, &self.buf_b, &self.buf_c,
                &norm, &layer.ffn_up, &mut gate_up,
                1, D_MODEL as u32, GATE_UP_DIM as u32);

            // Gate * up
            let (gate, up) = gate_up.split_at_mut(FFN_DIM);
            cpu_gelu(gate);
            elemwise_mul(gate, up, &mut ffn_mid);

            // Down: [1,8192] x [8192,3072] = [1,3072]
            gpu_matmul(engine, &self.buf_a, &self.buf_b, &self.buf_c,
                &ffn_mid, &layer.ffn_down, &mut tmp,
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
        let norm_ref = &norm;
        let emb_ref = &self.embedding;
        let logits: Vec<f32> = (0..VOCAB_SIZE).into_par_iter().map(|v| {
            let row = &emb_ref[v * D_MODEL..(v + 1) * D_MODEL];
            let mut s = 0.0f32;
            for c in 0..D_MODEL / 8 {
                let b = c * 8;
                s += norm_ref[b]*row[b] + norm_ref[b+1]*row[b+1]
                   + norm_ref[b+2]*row[b+2] + norm_ref[b+3]*row[b+3]
                   + norm_ref[b+4]*row[b+4] + norm_ref[b+5]*row[b+5]
                   + norm_ref[b+6]*row[b+6] + norm_ref[b+7]*row[b+7];
            }
            s
        }).collect();
        eprintln!("[fwd] logits: {}ms", lt.elapsed().as_millis());

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
        println!("\n=== Phi-4 Mini Token Generation ===");
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
