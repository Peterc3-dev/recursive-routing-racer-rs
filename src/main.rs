mod gpu;
mod model;

use std::time::Instant;

fn main() {
    println!("=== RRR — Multi-Sequence + Q4 Batched ===\n");

    unsafe {
        let mut engine = gpu::ComputeEngine::new(
            "/home/raz/projects/torch-vulkan/csrc/shaders");

        let k: u32 = 3072;
        let n: u32 = 3072;
        let push_bytes_fn = |m: u32| -> Vec<u8> {
            let push: [u32; 3] = [m, k, n];
            push.iter().flat_map(|v| v.to_le_bytes()).collect()
        };

        engine.get_pipeline("matmul_tiled", 3, 12);
        engine.get_pipeline("matmul_reg_tiled", 3, 12);

        // Test multiple sequence lengths
        for &seq in &[4u32, 16, 64, 128, 256] {
            let a_size = (seq * k) as u64 * 4;
            let b_size = (k * n) as u64 * 4;
            let c_size = (seq * n) as u64 * 4;

            let a_buf = engine.acquire_buffer(a_size);
            let b_buf = engine.acquire_buffer(b_size);
            let c_buf = engine.acquire_buffer(c_size);

            let a_f32 = std::slice::from_raw_parts_mut(a_buf.mapped as *mut f32, (seq * k) as usize);
            let b_f32 = std::slice::from_raw_parts_mut(b_buf.mapped as *mut f32, (k * n) as usize);
            for (i, v) in a_f32.iter_mut().enumerate() { *v = 0.01; }
            for (i, v) in b_f32.iter_mut().enumerate() { *v = 0.001; }

            let pb = push_bytes_fn(seq);

            // Bench standard tiled
            let std_groups = [(n + 15) / 16, (seq + 15) / 16, 1];
            let std_ms = bench_single(&engine, "matmul_tiled", &a_buf, &b_buf, &c_buf,
                a_size, b_size, c_size, &pb, std_groups);

            // Bench register tiled  
            let reg_groups = [(n + 63) / 64, (seq + 63) / 64, 1];
            let reg_ms = bench_single(&engine, "matmul_reg_tiled", &a_buf, &b_buf, &c_buf,
                a_size, b_size, c_size, &pb, reg_groups);

            let winner = if std_ms < reg_ms { "STD" } else { "REG" };
            println!("seq={:>4}: std={:.0}us  reg={:.0}us  winner={} ({:.2}x)",
                seq, std_ms * 1000.0, reg_ms * 1000.0, winner,
                if std_ms < reg_ms { reg_ms / std_ms } else { std_ms / reg_ms });

            engine.release_buffer(a_buf);
            engine.release_buffer(b_buf);
            engine.release_buffer(c_buf);
        }
    }
}

unsafe fn bench_single(eng: &gpu::ComputeEngine, shader: &str,
    a: &gpu::GpuBuffer, b: &gpu::GpuBuffer, c: &gpu::GpuBuffer,
    a_s: u64, b_s: u64, c_s: u64, push: &[u8], groups: [u32; 3]) -> f64 {
    
    let pipeline = &eng.pipeline_cache[shader];
    let layouts = vec![pipeline.desc_set_layout];
    let ds_alloc = ash::vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(eng.desc_pool).set_layouts(&layouts);
    let dsets = eng.device.allocate_descriptor_sets(&ds_alloc).unwrap();
    let ds = dsets[0];
    
    let bi = [
        ash::vk::DescriptorBufferInfo { buffer: a.buffer, offset: 0, range: a_s },
        ash::vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: b_s },
        ash::vk::DescriptorBufferInfo { buffer: c.buffer, offset: 0, range: c_s },
    ];
    let w: Vec<_> = bi.iter().enumerate().map(|(i, info)| {
        ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(i as u32)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(info))
    }).collect();
    eng.device.update_descriptor_sets(&w, &[]);

    // Warmup
    for _ in 0..10 {
        eng.device.reset_command_buffer(eng.cmd_buf, ash::vk::CommandBufferResetFlags::empty()).unwrap();
        let begin = ash::vk::CommandBufferBeginInfo::default().flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        eng.device.begin_command_buffer(eng.cmd_buf, &begin).unwrap();
        eng.device.cmd_bind_pipeline(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        eng.device.cmd_bind_descriptor_sets(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.layout, 0, &[ds], &[]);
        eng.device.cmd_push_constants(eng.cmd_buf, pipeline.layout, ash::vk::ShaderStageFlags::COMPUTE, 0, push);
        eng.device.cmd_dispatch(eng.cmd_buf, groups[0], groups[1], groups[2]);
        let hb = ash::vk::MemoryBarrier::default().src_access_mask(ash::vk::AccessFlags::SHADER_WRITE).dst_access_mask(ash::vk::AccessFlags::HOST_READ);
        eng.device.cmd_pipeline_barrier(eng.cmd_buf, ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            ash::vk::PipelineStageFlags::HOST, ash::vk::DependencyFlags::empty(), &[hb], &[], &[]);
        eng.device.end_command_buffer(eng.cmd_buf).unwrap();
        let sub = ash::vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&eng.cmd_buf));
        eng.device.reset_fences(&[eng.fence]).unwrap();
        eng.device.queue_submit(eng.queue, &[sub], eng.fence).unwrap();
        eng.device.wait_for_fences(&[eng.fence], true, u64::MAX).unwrap();
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..20 {
        eng.device.reset_command_buffer(eng.cmd_buf, ash::vk::CommandBufferResetFlags::empty()).unwrap();
        let begin = ash::vk::CommandBufferBeginInfo::default().flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        eng.device.begin_command_buffer(eng.cmd_buf, &begin).unwrap();
        eng.device.cmd_bind_pipeline(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        eng.device.cmd_bind_descriptor_sets(eng.cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline.layout, 0, &[ds], &[]);
        eng.device.cmd_push_constants(eng.cmd_buf, pipeline.layout, ash::vk::ShaderStageFlags::COMPUTE, 0, push);
        eng.device.cmd_dispatch(eng.cmd_buf, groups[0], groups[1], groups[2]);
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
    times.iter().sum::<f64>() / times.len() as f64
}
