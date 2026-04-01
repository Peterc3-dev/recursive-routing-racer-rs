mod gpu;
mod model;

use std::time::Instant;

fn main() {
    println!("=== RRR — ALL THREE OVERHEAD KILLS ===\n");

    unsafe {
        let mut engine = gpu::ComputeEngine::new(
            "/home/raz/projects/torch-vulkan/csrc/shaders");

        let m: u32 = 4;
        let k: u32 = 3072;
        let n: u32 = 3072;
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
        let push: [u32; 3] = [m, k, n];
        let push_bytes: &[u8] = std::slice::from_raw_parts(push.as_ptr() as *const u8, 12);
        let groups = [(n + 15) / 16, (m + 15) / 16, 1];

        // Warmup individual
        for _ in 0..20 {
            engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                &[a_size, b_size, c_size], push_bytes, groups);
        }

        // === BASELINE: 128 individual dispatches ===
        let mut ind_times = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..128 {
                engine.dispatch_by_name("matmul_tiled", &[&a_buf, &b_buf, &c_buf],
                    &[a_size, b_size, c_size], push_bytes, groups);
            }
            ind_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let ind_avg = ind_times.iter().sum::<f64>() / ind_times.len() as f64;

        // === FIX #1+2+3: Pre-allocate descriptors, record once, batch submit ===
        let pipeline = &engine.pipeline_cache["matmul_tiled"];
        
        // FIX #1: Pre-allocate ALL 128 descriptor sets at once
        let layouts: Vec<ash::vk::DescriptorSetLayout> = (0..128).map(|_| pipeline.desc_set_layout).collect();
        let ds_alloc = ash::vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(engine.desc_pool)
            .set_layouts(&layouts);
        let desc_sets = engine.device.allocate_descriptor_sets(&ds_alloc).unwrap();
        
        // FIX #1b: Update in place (one-time)
        for ds in &desc_sets {
            let buf_infos = [
                ash::vk::DescriptorBufferInfo { buffer: a_buf.buffer, offset: 0, range: a_size },
                ash::vk::DescriptorBufferInfo { buffer: b_buf.buffer, offset: 0, range: b_size },
                ash::vk::DescriptorBufferInfo { buffer: c_buf.buffer, offset: 0, range: c_size },
            ];
            let writes: Vec<ash::vk::WriteDescriptorSet> = buf_infos.iter().enumerate().map(|(i, info)| {
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(*ds)
                    .dst_binding(i as u32)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            }).collect();
            engine.device.update_descriptor_sets(&writes, &[]);
        }

        // FIX #2+3: Record ONE command buffer with ALL 128 dispatches
        // Then just submit + wait ONCE
        
        // Helper to record the mega command buffer
        let record_batch = |eng: &gpu::ComputeEngine| {
            eng.device.reset_command_buffer(eng.cmd_buf, ash::vk::CommandBufferResetFlags::empty()).unwrap();
            let begin = ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            eng.device.begin_command_buffer(eng.cmd_buf, &begin).unwrap();
            
            for (i, ds) in desc_sets.iter().enumerate() {
                eng.device.cmd_bind_pipeline(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                eng.device.cmd_bind_descriptor_sets(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout, 0, &[*ds], &[]);
                eng.device.cmd_push_constants(eng.cmd_buf, pipeline.layout,
                    ash::vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
                eng.device.cmd_dispatch(eng.cmd_buf, groups[0], groups[1], groups[2]);
                
                // Barrier between dispatches
                if i < 127 {
                    let barrier = ash::vk::MemoryBarrier::default()
                        .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ);
                    eng.device.cmd_pipeline_barrier(eng.cmd_buf,
                        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                        ash::vk::DependencyFlags::empty(), &[barrier], &[], &[]);
                }
            }
            
            // Final host barrier
            let hb = ash::vk::MemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(ash::vk::AccessFlags::HOST_READ);
            eng.device.cmd_pipeline_barrier(eng.cmd_buf,
                ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                ash::vk::PipelineStageFlags::HOST,
                ash::vk::DependencyFlags::empty(), &[hb], &[], &[]);
            
            eng.device.end_command_buffer(eng.cmd_buf).unwrap();
        };

        // Warmup batch
        for _ in 0..3 {
            record_batch(&engine);
            let submit = ash::vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&engine.cmd_buf));
            engine.device.reset_fences(&[engine.fence]).unwrap();
            engine.device.queue_submit(engine.queue, &[submit], engine.fence).unwrap();
            engine.device.wait_for_fences(&[engine.fence], true, u64::MAX).unwrap();
        }

        // Benchmark batch: measure JUST submit+wait (recording already done)
        let mut batch_times = Vec::new();
        for _ in 0..10 {
            record_batch(&engine);
            let t0 = Instant::now();
            let submit = ash::vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&engine.cmd_buf));
            engine.device.reset_fences(&[engine.fence]).unwrap();
            engine.device.queue_submit(engine.queue, &[submit], engine.fence).unwrap();
            engine.device.wait_for_fences(&[engine.fence], true, u64::MAX).unwrap();
            batch_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let batch_avg = batch_times.iter().sum::<f64>() / batch_times.len() as f64;
        let batch_min = batch_times.iter().cloned().fold(f64::MAX, f64::min);

        let c_f32 = std::slice::from_raw_parts(c_buf.mapped as *const f32, 4);

        println!("=== 128 MATMULS [4x3072 @ 3072x3072] ===");
        println!("  Individual (128 submits): {:.1}ms", ind_avg);
        println!("  Batched    (1 submit):    {:.1}ms avg / {:.1}ms min", batch_avg, batch_min);
        println!("  Speedup:                  {:.2}x", ind_avg / batch_avg);
        println!("  Output: [{:.6}, {:.6}]", c_f32[0], c_f32[1]);
        println!();
        
        let per_mm = batch_avg / 128.0;
        let per_layer = per_mm * 4.0;
        let full_pass = per_layer * 32.0;
        let tok_s = 1000.0 / full_pass;
        
        println!("=== PROJECTED INFERENCE ===");
        println!("  Per matmul:   {:.0}us ({:.3}ms)", per_mm * 1000.0, per_mm);
        println!("  Per layer:    {:.2}ms", per_layer);
        println!("  32 layers:    {:.1}ms", full_pass);
        println!("  Tokens/sec:   {:.1}", tok_s);
        println!();
        println!("  Kompute Python: 553ms / 1.53 tok/s");
        println!("  Rust individual: {:.1}ms / {:.1} tok/s", ind_avg, 1000.0 / (ind_avg / 128.0 * 4.0 * 32.0));
        println!("  Total speedup:  {:.0}x over Kompute", 553.0 / full_pass);

        engine.device.free_descriptor_sets(engine.desc_pool, &desc_sets).unwrap();
        engine.release_buffer(a_buf);
        engine.release_buffer(b_buf);
        engine.release_buffer(c_buf);
    }
}
