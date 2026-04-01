mod gpu;
mod model;

use std::time::Instant;

fn main() {
    println!("=== Recursive Routing Racer — Rust Phi-4 Forward Pass ===\n");

    // Load model
    let t0 = Instant::now();
    let gguf = model::GGUFModel::load(
        "/home/raz/.lmstudio/models/lmstudio-community/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf"
    );
    println!("[GGUF] Loaded in {:.1}s", t0.elapsed().as_secs_f64());
    println!("[GGUF] Arch: {}", gguf.meta_str("general.architecture"));
    println!("[GGUF] Name: {}", gguf.meta_str("general.name"));
    println!("[GGUF] d_model={}, layers={}, heads={}, kv_heads={}",
        gguf.meta_u64("phi3.embedding_length"),
        gguf.meta_u64("phi3.block_count"),
        gguf.meta_u64("phi3.attention.head_count"),
        gguf.meta_u64("phi3.attention.head_count_kv"));
    println!("[GGUF] {} tensors", gguf.n_tensors);
    
    // Test: load a weight tensor
    let norm_bytes = gguf.tensor_bytes("blk.0.attn_norm.weight");
    let norm_f32 = gguf.tensor_f32("blk.0.attn_norm.weight");
    println!("[GGUF] blk.0.attn_norm: {} bytes, {} floats, first=[{:.4}, {:.4}]",
        norm_bytes.len(), norm_f32.len(), norm_f32[0], norm_f32[1]);

    // Init Vulkan engine  
    unsafe {
        let mut engine = gpu::ComputeEngine::new(
            "/home/raz/projects/torch-vulkan/csrc/shaders");
        println!("\n[VK] Engine ready");

        // Benchmark: just matmul dispatch speed with real-sized buffers
        // This proves the Rust dispatch overhead is minimal
        let m: u32 = 4;
        let k: u32 = 3072;
        let n: u32 = 3072;
        let a_size = (m * k) as u64 * 4;
        let b_size = (k * n) as u64 * 4;
        let c_size = (m * n) as u64 * 4;

        let a_buf = engine.acquire_buffer(a_size);
        let b_buf = engine.acquire_buffer(b_size);
        let c_buf = engine.acquire_buffer(c_size);

        // Copy real norm weights into A buffer as test data
        let a_slice = std::slice::from_raw_parts_mut(a_buf.mapped as *mut u8, a_size as usize);
        // Fill with small values
        let a_f32 = std::slice::from_raw_parts_mut(a_buf.mapped as *mut f32, (m * k) as usize);
        for (i, v) in a_f32.iter_mut().enumerate() { *v = 0.01 * ((i % 100) as f32 - 50.0); }
        let b_f32 = std::slice::from_raw_parts_mut(b_buf.mapped as *mut f32, (k * n) as usize);
        for (i, v) in b_f32.iter_mut().enumerate() { *v = 0.001 * ((i % 100) as f32 - 50.0); }

        engine.get_pipeline("matmul_tiled", 3, 12);

        let push: [u32; 3] = [m, k, n];
        let push_bytes: &[u8] = std::slice::from_raw_parts(push.as_ptr() as *const u8, 12);
        let groups = [(n + 15) / 16, (m + 15) / 16, 1];

        // Warmup
        for _ in 0..20 {
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes, groups);
        }

        // Simulate 32-layer forward pass timing (4 matmuls per layer)
        // This measures pure dispatch overhead for 128 matmul dispatches
        println!("\n[BENCH] Simulating 32-layer dispatch (128 matmuls)...");
        let t0 = Instant::now();
        for _ in 0..128 {
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes, groups);
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let per_mm = total_ms / 128.0;
        let per_layer = per_mm * 4.0;  // ~4 matmuls per layer
        let tok_s = 1000.0 / (per_layer * 32.0);

        let c_f32 = std::slice::from_raw_parts(c_buf.mapped as *const f32, 4);

        println!("\n=== RUST PHI-4 PROJECTION ===");
        println!("  128 matmuls: {:.1}ms total", total_ms);
        println!("  Per matmul:  {:.1}ms ({:.0}us)", per_mm, per_mm * 1000.0);
        println!("  Per layer:   {:.1}ms (4 matmuls)", per_layer);
        println!("  32 layers:   {:.0}ms", per_layer * 32.0);
        println!("  Tokens/sec:  {:.1}", tok_s);
        println!("  Output:      [{:.6}, {:.6}]", c_f32[0], c_f32[1]);
        println!();
        println!("  Kompute was: 553ms / 1.53 tok/s");
        println!("  Speedup:     {:.1}x", 553.0 / (per_layer * 32.0));
        
        engine.release_buffer(a_buf);
        engine.release_buffer(b_buf);
        engine.release_buffer(c_buf);
    }
}
