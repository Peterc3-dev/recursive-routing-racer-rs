mod gpu;
mod model;

use std::time::Instant;

/// Simple Q4 quantization in Rust (vectorized, no per-element loops)
fn quantize_q4(weight: &[f32], k: usize, n: usize) -> (Vec<u32>, Vec<f32>) {
    let block_size = 32;
    let n_blocks = k / block_size;
    
    // Compute scales per block
    let mut scales = vec![0.0f32; n_blocks * n];
    for bk in 0..n_blocks {
        for j in 0..n {
            let mut max_abs: f32 = 0.0;
            for i in 0..block_size {
                let v = weight[(bk * block_size + i) * n + j].abs();
                if v > max_abs { max_abs = v; }
            }
            scales[bk * n + j] = (max_abs / 7.0).max(1e-8);
        }
    }
    
    // Quantize and pack nibbles into u32
    let total = k * n;
    let n_words = (total + 7) / 8;
    let mut packed = vec![0u32; n_words];
    
    for idx in 0..total {
        let row = idx / n;
        let col = idx % n;
        let bk = row / block_size;
        let scale = scales[bk * n + col];
        let q = ((weight[idx] / scale + 8.0).round().clamp(0.0, 15.0)) as u32;
        packed[idx / 8] |= q << ((idx % 8) * 4);
    }
    
    (packed, scales)
}

fn main() {
    println!("=== RRR — Rust Q4 vs F32 Matmul Benchmark ===\n");

    unsafe {
        let mut engine = gpu::ComputeEngine::new(
            "/home/raz/projects/torch-vulkan/csrc/shaders");
        
        let m: u32 = 4;
        let k: u32 = 3072;
        let n: u32 = 3072;
        
        // === F32 PATH ===
        let a_size = (m * k) as u64 * 4;
        let b_size = (k * n) as u64 * 4;
        let c_size = (m * n) as u64 * 4;
        
        let a_buf = engine.acquire_buffer(a_size);
        let b_buf = engine.acquire_buffer(b_size);
        let c_buf = engine.acquire_buffer(c_size);
        
        let a_f32 = std::slice::from_raw_parts_mut(a_buf.mapped as *mut f32, (m * k) as usize);
        let b_f32 = std::slice::from_raw_parts_mut(b_buf.mapped as *mut f32, (k * n) as usize);
        for (i, v) in a_f32.iter_mut().enumerate() { *v = 0.01 * ((i % 100) as f32 - 50.0); }
        for (i, v) in b_f32.iter_mut().enumerate() { *v = 0.001 * ((i % 100) as f32 - 50.0); }
        
        engine.get_pipeline("matmul_tiled", 3, 12);
        let push_f32: [u32; 3] = [m, k, n];
        let push_bytes_f32: &[u8] = std::slice::from_raw_parts(push_f32.as_ptr() as *const u8, 12);
        let groups = [(n + 15) / 16, (m + 15) / 16, 1];
        
        for _ in 0..20 {
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes_f32, groups);
        }
        
        let mut f32_times = Vec::new();
        for _ in 0..100 {
            let t0 = Instant::now();
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes_f32, groups);
            f32_times.push(t0.elapsed().as_micros() as f64);
        }
        let f32_avg = f32_times.iter().sum::<f64>() / f32_times.len() as f64;
        let f32_min = f32_times.iter().cloned().fold(f64::MAX, f64::min);
        
        let c_f32_out = std::slice::from_raw_parts(c_buf.mapped as *const f32, 4);
        println!("F32 matmul [4x3072 @ 3072x3072]:");
        println!("  {:.0}us avg / {:.0}us min", f32_avg, f32_min);
        println!("  Output: [{:.6}, {:.6}]", c_f32_out[0], c_f32_out[1]);
        
        // === Q4 PATH ===
        // Quantize B weights
        print!("\nQuantizing weights to Q4...");
        let t0 = Instant::now();
        let b_data: Vec<f32> = b_f32.to_vec();
        let (packed, scales) = quantize_q4(&b_data, k as usize, n as usize);
        println!(" {:.0}ms", t0.elapsed().as_millis());
        println!("  Packed: {} u32s ({:.1} KB)", packed.len(), packed.len() as f64 * 4.0 / 1024.0);
        println!("  Scales: {} floats ({:.1} KB)", scales.len(), scales.len() as f64 * 4.0 / 1024.0);
        println!("  Compression: {:.1}x", b_size as f64 / (packed.len() * 4 + scales.len() * 4) as f64);
        
        let bp_size = packed.len() as u64 * 4;
        let sc_size = scales.len() as u64 * 4;
        
        let bp_buf = engine.acquire_buffer(bp_size);
        let sc_buf = engine.acquire_buffer(sc_size);
        let c2_buf = engine.acquire_buffer(c_size);
        
        // Copy packed data (reinterpret u32 as bytes)
        std::ptr::copy_nonoverlapping(packed.as_ptr() as *const u8, bp_buf.mapped as *mut u8, bp_size as usize);
        std::ptr::copy_nonoverlapping(scales.as_ptr() as *const u8, sc_buf.mapped as *mut u8, sc_size as usize);
        
        engine.get_pipeline("matmul_q4", 4, 12);
        let push_q4: [u32; 3] = [m, k, n];
        let push_bytes_q4: &[u8] = std::slice::from_raw_parts(push_q4.as_ptr() as *const u8, 12);
        
        for _ in 0..20 {
            engine.dispatch_by_name("matmul_q4", &[&a_buf, &bp_buf, &sc_buf, &c2_buf],
                &[a_size, bp_size, sc_size, c_size], push_bytes_q4, groups);
        }
        
        let mut q4_times = Vec::new();
        for _ in 0..100 {
            let t0 = Instant::now();
            engine.dispatch_by_name("matmul_q4", &[&a_buf, &bp_buf, &sc_buf, &c2_buf],
                &[a_size, bp_size, sc_size, c_size], push_bytes_q4, groups);
            q4_times.push(t0.elapsed().as_micros() as f64);
        }
        let q4_avg = q4_times.iter().sum::<f64>() / q4_times.len() as f64;
        let q4_min = q4_times.iter().cloned().fold(f64::MAX, f64::min);
        
        let c_q4_out = std::slice::from_raw_parts(c2_buf.mapped as *const f32, 4);
        
        println!("\nQ4 matmul [4x3072 @ 3072x3072]:");
        println!("  {:.0}us avg / {:.0}us min", q4_avg, q4_min);
        println!("  Output: [{:.6}, {:.6}]", c_q4_out[0], c_q4_out[1]);
        
        // Projections
        let q4_layer = q4_avg * 4.0 / 1000.0;  // 4 matmuls per layer
        let f32_layer = f32_avg * 4.0 / 1000.0;
        
        println!("\n=== COMPARISON ===");
        println!("  F32: {:.0}us/mm  -> {:.1}ms/layer -> {:.0}ms/32L -> {:.1} tok/s",
            f32_avg, f32_layer, f32_layer * 32.0, 1000.0 / (f32_layer * 32.0));
        println!("  Q4:  {:.0}us/mm  -> {:.1}ms/layer -> {:.0}ms/32L -> {:.1} tok/s",
            q4_avg, q4_layer, q4_layer * 32.0, 1000.0 / (q4_layer * 32.0));
        println!("  Q4 speedup: {:.2}x per matmul", f32_avg / q4_avg);
        println!("  Memory:     {:.1}x less", b_size as f64 / (bp_size + sc_size) as f64);
        
        engine.release_buffer(a_buf);
        engine.release_buffer(b_buf);
        engine.release_buffer(c_buf);
        engine.release_buffer(bp_buf);
        engine.release_buffer(sc_buf);
        engine.release_buffer(c2_buf);
    }
}
