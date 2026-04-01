mod gpu;
mod model;
use std::time::Instant;

fn main() {
    println!("=== RRR — Op Profiling ===\n");
    unsafe {
        let mut engine = gpu::ComputeEngine::new("/home/raz/projects/torch-vulkan/csrc/shaders");
        let d: u32 = 3072; let seq: u32 = 4;

        engine.get_pipeline("matmul_tiled", 3, 12);
        engine.get_pipeline("layer_norm", 4, 8);
        engine.get_pipeline("gelu", 2, 4);
        engine.get_pipeline("attention", 4, 8);
        engine.get_pipeline("add", 3, 8);

        let x = engine.acquire_buffer((seq * d) as u64 * 4);
        let w = engine.acquire_buffer((d * d) as u64 * 4);
        let b = engine.acquire_buffer(d as u64 * 4);
        let o = engine.acquire_buffer((seq * d) as u64 * 4);

        let a_f32 = std::slice::from_raw_parts_mut(x.mapped as *mut f32, (seq * d) as usize);
        for (i, v) in a_f32.iter_mut().enumerate() { *v = 0.01; }
        let w_f32 = std::slice::from_raw_parts_mut(w.mapped as *mut f32, (d * d) as usize);
        for v in w_f32.iter_mut() { *v = 0.001; }
        std::ptr::write_bytes(b.mapped as *mut f32, 0, d as usize);

        // Profile each op 20 times
        for (name, op_fn) in [
            ("matmul", Box::new(|| {
                let push: [u32; 3] = [seq, d, d];
                let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name("matmul_tiled", &[&x, &w, &o],
                    &[(seq*d) as u64*4, (d*d) as u64*4, (seq*d) as u64*4], &pb, [(d+15)/16, (seq+15)/16, 1]);
            }) as Box<dyn Fn()>),
            ("layer_norm", Box::new(|| {
                let push: [u32; 2] = [seq, 128]; // per-head dim, not full d_model
                let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name("layer_norm", &[&x, &b, &b, &o],
                    &[(seq*d) as u64*4, d as u64*4, d as u64*4, (seq*d) as u64*4], &pb, [seq, 1, 1]);
            }) as Box<dyn Fn()>),
            ("gelu", Box::new(|| {
                let n = seq * d;
                let push: [u32; 1] = [n];
                let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name("gelu", &[&x, &o],
                    &[(n as u64)*4, (n as u64)*4], &pb, [(n+255)/256, 1, 1]);
            }) as Box<dyn Fn()>),
            ("add", Box::new(|| {
                let n = seq * d;
                let alpha_bits: u32 = 1.0f32.to_bits();
                let push: [u32; 2] = [n, alpha_bits];
                let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name("add", &[&x, &o, &o],
                    &[(n as u64)*4, (n as u64)*4, (n as u64)*4], &pb, [(n+255)/256, 1, 1]);
            }) as Box<dyn Fn()>),
            ("attention", Box::new(|| {
                let push: [u32; 2] = [seq, 128]; // per-head dim, not full d_model
                let pb: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
                engine.dispatch_by_name("attention", &[&x, &x, &x, &o],
                    &[(seq*d) as u64*4, (seq*d) as u64*4, (seq*d) as u64*4, (seq*d) as u64*4],
                    &pb, [seq, 1, 1]);
            }) as Box<dyn Fn()>),
        ] {
            // warmup
            for _ in 0..5 { op_fn(); }
            let mut times = Vec::new();
            for _ in 0..20 {
                let t0 = Instant::now();
                op_fn();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::MAX, f64::min);
            println!("{:>12}: {:.0}us avg / {:.0}us min", name, avg, min);
        }

        engine.release_buffer(x);
        engine.release_buffer(w);
        engine.release_buffer(b);
        engine.release_buffer(o);
    }
}
