//! High-level op dispatch — wraps the Vulkan engine for transformer ops.

use crate::gpu::{ComputeEngine, GpuBuffer};

/// Layer norm on GPU via SPIR-V shader
pub unsafe fn layer_norm(
    engine: &ComputeEngine, 
    input: &GpuBuffer, weight: &GpuBuffer, bias: &GpuBuffer, output: &GpuBuffer,
    num_rows: u32, row_size: u32,
    in_size: u64, w_size: u64, b_size: u64, out_size: u64,
) {
    let push: [u32; 2] = [num_rows, row_size];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name("layer_norm",
        &[input, weight, bias, output],
        &[in_size, w_size, b_size, out_size],
        &pb, [num_rows, 1, 1]);
}

/// GELU activation on GPU
pub unsafe fn gelu(
    engine: &ComputeEngine,
    input: &GpuBuffer, output: &GpuBuffer,
    n: u32, size: u64,
) {
    let push: [u32; 1] = [n];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name("gelu",
        &[input, output],
        &[size, size],
        &pb, [(n + 255) / 256, 1, 1]);
}

/// Fused scaled dot-product attention on GPU
pub unsafe fn attention(
    engine: &ComputeEngine,
    q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer, output: &GpuBuffer,
    seq_len: u32, head_dim: u32, size: u64,
) {
    let push: [u32; 2] = [seq_len, head_dim];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name("attention",
        &[q, k, v, output],
        &[size, size, size, size],
        &pb, [seq_len, 1, 1]);
}

/// Elementwise add: out = a + b
pub unsafe fn add(
    engine: &ComputeEngine,
    a: &GpuBuffer, b: &GpuBuffer, output: &GpuBuffer,
    n: u32, size: u64,
) {
    // alpha = 1.0 as bits
    let alpha_bits: u32 = 1.0f32.to_bits();
    let push: [u32; 2] = [n, alpha_bits];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    engine.dispatch_by_name("add",
        &[a, b, output],
        &[size, size, size],
        &pb, [(n + 255) / 256, 1, 1]);
}

/// Matmul with automatic shader selection based on seq length
pub unsafe fn matmul(
    engine: &ComputeEngine,
    a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer,
    m: u32, k: u32, n: u32,
    a_size: u64, b_size: u64, c_size: u64,
) {
    let push: [u32; 3] = [m, k, n];
    let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
    
    // HDC-informed shader selection: crossover at seq=32
    let (shader, groups) = if m <= 16 {
        ("matmul_tiled", [(n + 15) / 16, (m + 15) / 16, 1])
    } else {
        ("matmul_reg_tiled", [(n + 63) / 64, (m + 63) / 64, 1])
    };
    
    engine.dispatch_by_name(shader,
        &[a, b, c], &[a_size, b_size, c_size],
        &pb, groups);
}
