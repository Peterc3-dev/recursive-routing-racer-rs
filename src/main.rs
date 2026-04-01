mod gpu;
mod model;

use std::time::Instant;

fn main() {
    println!("=== RRR — Register-Tiled vs Standard Tiled ===\n");

    unsafe {
        let mut engine = gpu::ComputeEngine::new(
            "/home/raz/projects/torch-vulkan/csrc/shaders");

        let m: u32 = 4; let k: u32 = 3072; let n: u32 = 3072;
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

        let push: [u32; 3] = [m, k, n];
        let push_bytes: &[u8] = std::slice::from_raw_parts(push.as_ptr() as *const u8, 12);

        // Helper: batch 128 dispatches of a given shader
        unsafe fn bench_batch(eng: &gpu::ComputeEngine, shader: &str, 
                       a: &gpu::GpuBuffer, b: &gpu::GpuBuffer, c: &gpu::GpuBuffer,
                       a_s: u64, b_s: u64, c_s: u64,
                       push: &[u8], groups: [u32; 3]) -> (f64, f64) {
            let pipeline = &eng.pipeline_cache[shader];
            let layouts: Vec<_> = (0..128).map(|_| pipeline.desc_set_layout).collect();
            let ds_alloc = ash::vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(eng.desc_pool).set_layouts(&layouts);
            let dsets = eng.device.allocate_descriptor_sets(&ds_alloc).unwrap();
            for ds in &dsets {
                let bi = [
                    ash::vk::DescriptorBufferInfo { buffer: a.buffer, offset: 0, range: a_s },
                    ash::vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: b_s },
                    ash::vk::DescriptorBufferInfo { buffer: c.buffer, offset: 0, range: c_s },
                ];
                let w: Vec<_> = bi.iter().enumerate().map(|(i, info)| {
                    ash::vk::WriteDescriptorSet::default().dst_set(*ds).dst_binding(i as u32)
                        .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                }).collect();
                eng.device.update_descriptor_sets(&w, &[]);
            }
            
            // Warmup
            for _ in 0..3 {
                eng.device.reset_command_buffer(eng.cmd_buf, ash::vk::CommandBufferResetFlags::empty()).unwrap();
                let begin = ash::vk::CommandBufferBeginInfo::default().flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                eng.device.begin_command_buffer(eng.cmd_buf, &begin).unwrap();
                for (i, ds) in dsets.iter().enumerate() {
                    eng.device.cmd_bind_pipeline(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                    eng.device.cmd_bind_descriptor_sets(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.layout, 0, &[*ds], &[]);
                    eng.device.cmd_push_constants(eng.cmd_buf, pipeline.layout, ash::vk::ShaderStageFlags::COMPUTE, 0, push);
                    eng.device.cmd_dispatch(eng.cmd_buf, groups[0], groups[1], groups[2]);
                    if i < 127 {
                        let bar = ash::vk::MemoryBarrier::default()
                            .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE).dst_access_mask(ash::vk::AccessFlags::SHADER_READ);
                        eng.device.cmd_pipeline_barrier(eng.cmd_buf, ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                            ash::vk::PipelineStageFlags::COMPUTE_SHADER, ash::vk::DependencyFlags::empty(), &[bar], &[], &[]);
                    }
                }
                let hb = ash::vk::MemoryBarrier::default().src_access_mask(ash::vk::AccessFlags::SHADER_WRITE).dst_access_mask(ash::vk::AccessFlags::HOST_READ);
                eng.device.cmd_pipeline_barrier(eng.cmd_buf, ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                    ash::vk::PipelineStageFlags::HOST, ash::vk::DependencyFlags::empty(), &[hb], &[], &[]);
                eng.device.end_command_buffer(eng.cmd_buf).unwrap();
                let sub = ash::vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&eng.cmd_buf));
                eng.device.reset_fences(&[eng.fence]).unwrap();
                eng.device.queue_submit(eng.queue, &[sub], eng.fence).unwrap();
                eng.device.wait_for_fences(&[eng.fence], true, u64::MAX).unwrap();
            }
            
            let mut times = Vec::new();
            for _ in 0..10 {
                eng.device.reset_command_buffer(eng.cmd_buf, ash::vk::CommandBufferResetFlags::empty()).unwrap();
                let begin = ash::vk::CommandBufferBeginInfo::default().flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                eng.device.begin_command_buffer(eng.cmd_buf, &begin).unwrap();
                for (i, ds) in dsets.iter().enumerate() {
                    eng.device.cmd_bind_pipeline(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                    eng.device.cmd_bind_descriptor_sets(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.layout, 0, &[*ds], &[]);
                    eng.device.cmd_push_constants(eng.cmd_buf, pipeline.layout, ash::vk::ShaderStageFlags::COMPUTE, 0, push);
                    eng.device.cmd_dispatch(eng.cmd_buf, groups[0], groups[1], groups[2]);
                    if i < 127 {
                        let bar = ash::vk::MemoryBarrier::default()
                            .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE).dst_access_mask(ash::vk::AccessFlags::SHADER_READ);
                        eng.device.cmd_pipeline_barrier(eng.cmd_buf, ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                            ash::vk::PipelineStageFlags::COMPUTE_SHADER, ash::vk::DependencyFlags::empty(), &[bar], &[], &[]);
                    }
                }
                let hb = ash::vk::MemoryBarrier::default().src_access_mask(ash::vk::AccessFlags::SHADER_WRITE).dst_access_mask(ash::vk::AccessFlags::HOST_READ);
                eng.device.cmd_pipeline_barrier(eng.cmd_buf, ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                    ash::vk::PipelineStageFlags::HOST, ash::vk::DependencyFlags::empty(), &[hb], &[], &[]);
                eng.device.end_command_buffer(eng.cmd_buf).unwrap();
                let t0 = Instant::now();
                let sub = ash::vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&eng.cmd_buf));
                eng.device.reset_fences(&[eng.fence]).unwrap();
                eng.device.queue_submit(eng.queue, &[sub], eng.fence).unwrap();
                eng.device.wait_for_fences(&[eng.fence], true, u64::MAX).unwrap();
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            eng.device.free_descriptor_sets(eng.desc_pool, &dsets).unwrap();
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().cloned().fold(f64::MAX, f64::min);
            (avg, min)
        }

        // Standard tiled (16x16 threads, 1 output per thread)
        engine.get_pipeline("matmul_tiled", 3, 12);
        let std_groups = [(n + 15) / 16, (m + 15) / 16, 1];
        let (std_avg, std_min) = bench_batch(&engine, "matmul_tiled",
            &a_buf, &b_buf, &c_buf, a_size, b_size, c_size, push_bytes, std_groups);
        let c_std = std::slice::from_raw_parts(c_buf.mapped as *const f32, 4);
        
        println!("Standard tiled (16x16, 1 output/thread):");
        println!("  128 batched: {:.1}ms avg / {:.1}ms min  ({:.0}us/mm)", std_avg, std_min, std_avg * 1000.0 / 128.0);
        println!("  32 layers:   {:.1}ms -> {:.1} tok/s", std_avg / 128.0 * 4.0 * 32.0, 1000.0 / (std_avg / 128.0 * 4.0 * 32.0));
        println!("  Output: [{:.6}, {:.6}]", c_std[0], c_std[1]);

        // Register tiled (16x16 threads, 4x4 outputs per thread = 64x64 per WG)
        engine.get_pipeline("matmul_reg_tiled", 3, 12);
        let reg_groups = [(n + 63) / 64, (m + 63) / 64, 1];
        let (reg_avg, reg_min) = bench_batch(&engine, "matmul_reg_tiled",
            &a_buf, &b_buf, &c_buf, a_size, b_size, c_size, push_bytes, reg_groups);
        let c_reg = std::slice::from_raw_parts(c_buf.mapped as *const f32, 4);
        
        println!("\nRegister-tiled (16x16, 4x4 outputs/thread):");
        println!("  128 batched: {:.1}ms avg / {:.1}ms min  ({:.0}us/mm)", reg_avg, reg_min, reg_avg * 1000.0 / 128.0);
        println!("  32 layers:   {:.1}ms -> {:.1} tok/s", reg_avg / 128.0 * 4.0 * 32.0, 1000.0 / (reg_avg / 128.0 * 4.0 * 32.0));
        println!("  Speedup:     {:.2}x", std_avg / reg_avg);
        println!("  Output: [{:.6}, {:.6}]", c_reg[0], c_reg[1]);

        engine.release_buffer(a_buf);
        engine.release_buffer(b_buf);
        engine.release_buffer(c_buf);
    }
}
