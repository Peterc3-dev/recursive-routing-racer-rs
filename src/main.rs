mod gpu;

fn main() {
    println!("=== Recursive Routing Racer — Rust Runtime ===");
    println!();

    unsafe {
        let mut engine = gpu::ComputeEngine::new(
            "/home/raz/projects/torch-vulkan/csrc/shaders");
        println!("[RRR] Engine initialized");

        let m: u32 = 4;
        let k: u32 = 3072;
        let n: u32 = 3072;
        let a_size = (m * k) as u64 * 4;
        let b_size = (k * n) as u64 * 4;
        let c_size = (m * n) as u64 * 4;

        let a_buf = engine.acquire_buffer(a_size);
        let b_buf = engine.acquire_buffer(b_size);
        let c_buf = engine.acquire_buffer(c_size);

        // Fill input
        let a_slice = std::slice::from_raw_parts_mut(a_buf.mapped as *mut f32, (m * k) as usize);
        let b_slice = std::slice::from_raw_parts_mut(b_buf.mapped as *mut f32, (k * n) as usize);
        for v in a_slice.iter_mut() { *v = 0.01; }
        for v in b_slice.iter_mut() { *v = 0.01; }

        engine.get_pipeline("matmul_tiled", 3, 12);
        println!("[RRR] Pipeline cached");

        let push: [u32; 3] = [m, k, n];
        let push_bytes: &[u8] = std::slice::from_raw_parts(push.as_ptr() as *const u8, 12);
        let groups = [(n + 15) / 16, (m + 15) / 16, 1];

        // Warmup
        print!("Warmup...");
        for _ in 0..20 {
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes, groups);
        }
        println!(" done");

        // Benchmark
        let mut times = Vec::with_capacity(100);
        for _ in 0..100 {
            let t0 = std::time::Instant::now();
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes, groups);
            times.push(t0.elapsed().as_micros() as f64);
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        let c_slice = std::slice::from_raw_parts(c_buf.mapped as *const f32, 4);

        println!();
        println!("=== RUST VULKAN MATMUL [4x3072 @ 3072x3072] ===");
        println!("  Rust:    {:.0} us avg / {:.0} us min", avg, min);
        println!("  Kompute: 1679 us avg");
        println!("  Speedup: {:.2}x", 1679.0 / avg);
        println!("  Output:  [{:.4}, {:.4}, {:.4}, {:.4}]", c_slice[0], c_slice[1], c_slice[2], c_slice[3]);

        engine.release_buffer(a_buf);
        engine.release_buffer(b_buf);
        engine.release_buffer(c_buf);
    }
}
