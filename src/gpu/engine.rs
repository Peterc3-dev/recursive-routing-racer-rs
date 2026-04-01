//! Raw Vulkan compute engine — zero ceremony dispatch.
//! 
//! Creates VkDevice once. Caches pipelines. Reuses command buffers.
//! Dispatch = memcpy + submit + fence. ~50us target.

use ash::vk;
use ash::Device;
use std::collections::HashMap;
use std::ffi::CString;

pub struct ComputeEngine {
    pub instance: ash::Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buf: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub desc_pool: vk::DescriptorPool,
    pub pipeline_cache: HashMap<String, CachedPipeline>,
    pub buffer_pool: HashMap<u64, Vec<GpuBuffer>>,  // size-bucketed pool
    pub shader_dir: String,
}

pub struct CachedPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub num_buffers: u32,
}

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut std::ffi::c_void,
    pub capacity: u64,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl ComputeEngine {
    pub unsafe fn new(shader_dir: &str) -> Self {
        let entry = ash::Entry::linked();
        
        // Instance
        let app_name = CString::new("rrr").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::make_api_version(0, 1, 2, 0));
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);
        let instance = entry.create_instance(&instance_info, None).unwrap();

        // Physical device
        let pdevices = instance.enumerate_physical_devices().unwrap();
        let physical_device = pdevices[0];
        let props = instance.get_physical_device_properties(physical_device);
        let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr());
        eprintln!("[RRR] GPU: {}", name.to_string_lossy());

        // Find compute queue
        let queue_props = instance.get_physical_device_queue_family_properties(physical_device);
        let queue_family = queue_props.iter().position(|p| {
            p.queue_flags.contains(vk::QueueFlags::COMPUTE)
        }).unwrap() as u32;

        // Device + queue
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priority);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info));
        let device = instance.create_device(physical_device, &device_info, None).unwrap();
        let queue = device.get_device_queue(queue_family, 0);

        // Command pool + buffer (reusable)
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = device.create_command_pool(&pool_info, None).unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = device.allocate_command_buffers(&alloc_info).unwrap()[0];

        // Fence
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();

        // Descriptor pool
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2048,
        };
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(512)
            .pool_sizes(std::slice::from_ref(&pool_size))
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let desc_pool = device.create_descriptor_pool(&dp_info, None).unwrap();

        ComputeEngine {
            instance,
            device,
            physical_device,
            queue,
            queue_family,
            cmd_pool,
            cmd_buf,
            fence,
            desc_pool,
            pipeline_cache: HashMap::new(),
            buffer_pool: HashMap::new(),
            shader_dir: shader_dir.to_string(),
        }
    }

    /// Load SPIR-V shader from disk
    pub fn load_spirv(&self, name: &str) -> Vec<u32> {
        let path = format!("{}/{}.spv", self.shader_dir, name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("Failed to load shader {}: {}", path, e));
        assert!(bytes.len() % 4 == 0);
        bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Get or create a cached compute pipeline
    pub unsafe fn get_pipeline(&mut self, name: &str, num_buffers: u32, push_size: u32) -> &CachedPipeline {
        if self.pipeline_cache.contains_key(name) {
            return &self.pipeline_cache[name];
        }

        let spirv = self.load_spirv(name);
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader = self.device.create_shader_module(&shader_info, None).unwrap();

        // Descriptor set layout
        let bindings: Vec<_> = (0..num_buffers).map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        }).collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = self.device.create_descriptor_set_layout(&dsl_info, None).unwrap();

        // Pipeline layout
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_size,
        };
        let pl_info = if push_size > 0 {
            vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&desc_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range))
        } else {
            vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&desc_set_layout))
        };
        let layout = self.device.create_pipeline_layout(&pl_info, None).unwrap();

        // Compute pipeline
        let entry_name = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        let cp_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = self.device.create_compute_pipelines(
            vk::PipelineCache::null(), &[cp_info], None
        ).unwrap()[0];

        self.device.destroy_shader_module(shader, None);

        self.pipeline_cache.insert(name.to_string(), CachedPipeline {
            pipeline, layout, desc_set_layout, num_buffers,
        });
        &self.pipeline_cache[name]
    }

    /// Allocate a host-visible, coherent GPU buffer
    pub unsafe fn alloc_buffer(&self, size: u64) -> GpuBuffer {
        eprintln!("[alloc] size={}", size);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.device.create_buffer(&buf_info, None).unwrap();

        let mem_req = self.device.get_buffer_memory_requirements(buffer);
        let mem_props = self.instance.get_physical_device_memory_properties(self.physical_device);
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                let bits_match = (mem_req.memory_type_bits & (1u32 << i)) != 0;
                let flags = mem_props.memory_types[i as usize].property_flags;
                bits_match && flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                )
            })
            .expect("No suitable memory type found");

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let memory = self.device.allocate_memory(&alloc, None).unwrap();
        self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
        eprintln!("[alloc] bound ok");
        let mapped = self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();

        GpuBuffer { buffer, memory, mapped, capacity: size }
    }

    /// Acquire a pooled buffer (reuse if available)
    pub unsafe fn acquire_buffer(&mut self, size: u64) -> GpuBuffer {
        let bucket = size.next_power_of_two().max(256);
        if let Some(pool) = self.buffer_pool.get_mut(&bucket) {
            if let Some(buf) = pool.pop() {
                return buf;
            }
        }
        self.alloc_buffer(bucket)
    }

    /// Return buffer to pool
    pub fn release_buffer(&mut self, buf: GpuBuffer) {
        self.buffer_pool.entry(buf.capacity).or_default().push(buf);
    }


    /// Batch multiple dispatches into ONE command buffer, ONE submit, ONE fence wait.
    pub unsafe fn dispatch_batch(
        &self,
        dispatches: &[(&str, &[&GpuBuffer], &[u64], &[u8], [u32; 3])],
    ) {
        let mut desc_sets = Vec::with_capacity(dispatches.len());
        for (name, bufs, sizes, _, _) in dispatches {
            let pipeline = &self.pipeline_cache[*name];
            let ds_alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.desc_pool)
                .set_layouts(std::slice::from_ref(&pipeline.desc_set_layout));
            let ds = self.device.allocate_descriptor_sets(&ds_alloc).unwrap()[0];
            
            let buf_infos: Vec<vk::DescriptorBufferInfo> = bufs.iter().zip(sizes.iter()).map(|(b, &s)| {
                vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: s }
            }).collect();
            let writes: Vec<vk::WriteDescriptorSet> = buf_infos.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            }).collect();
            self.device.update_descriptor_sets(&writes, &[]);
            desc_sets.push(ds);
        }

        self.device.reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(self.cmd_buf, &begin).unwrap();

        for (i, (name, _, _, push, groups)) in dispatches.iter().enumerate() {
            let pipeline = &self.pipeline_cache[*name];
            self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buf, vk::PipelineBindPoint::COMPUTE,
                pipeline.layout, 0, &[desc_sets[i]], &[]);
            if !push.is_empty() {
                self.device.cmd_push_constants(
                    self.cmd_buf, pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            self.device.cmd_dispatch(self.cmd_buf, groups[0], groups[1], groups[2]);

            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                self.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[]);
        }

        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        self.device.cmd_pipeline_barrier(
            self.cmd_buf,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[host_barrier], &[], &[]);

        self.device.end_command_buffer(self.cmd_buf).unwrap();

        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.cmd_buf));
        self.device.reset_fences(&[self.fence]).unwrap();
        self.device.queue_submit(self.queue, &[submit], self.fence).unwrap();
        self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();

        self.device.free_descriptor_sets(self.desc_pool, &desc_sets).unwrap();
    }

    /// Dispatch by pipeline name (avoids borrow issues)
    pub unsafe fn dispatch_by_name(
        &self,
        name: &str,
        buffers: &[&GpuBuffer],
        buffer_sizes: &[u64],
        push_constants: &[u8],
        groups: [u32; 3],
    ) {
        let pipeline = &self.pipeline_cache[name];
        self.dispatch(pipeline, buffers, buffer_sizes, push_constants, groups);
    }

    /// Dispatch compute shader — the hot path, must be fast
    pub unsafe fn dispatch(
        &self,
        pipeline: &CachedPipeline,
        buffers: &[&GpuBuffer],
        buffer_sizes: &[u64],
        push_constants: &[u8],
        groups: [u32; 3],
    ) {
        // Allocate descriptor set
        let ds_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(std::slice::from_ref(&pipeline.desc_set_layout));
        let desc_set = self.device.allocate_descriptor_sets(&ds_alloc).unwrap()[0];

        // Update descriptors
        let buf_infos: Vec<_> = buffers.iter().zip(buffer_sizes).map(|(b, &s)| {
            vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: s }
        }).collect();
        let writes: Vec<_> = buf_infos.iter().enumerate().map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }).collect();
        self.device.update_descriptor_sets(&writes, &[]);

        // Record
        self.device.reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(self.cmd_buf, &begin).unwrap();
        self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        self.device.cmd_bind_descriptor_sets(
            self.cmd_buf, vk::PipelineBindPoint::COMPUTE,
            pipeline.layout, 0, &[desc_set], &[]);
        if !push_constants.is_empty() {
            self.device.cmd_push_constants(
                self.cmd_buf, pipeline.layout,
                vk::ShaderStageFlags::COMPUTE, 0, push_constants);
        }
        self.device.cmd_dispatch(self.cmd_buf, groups[0], groups[1], groups[2]);

        // Barrier
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        self.device.cmd_pipeline_barrier(
            self.cmd_buf,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier], &[], &[]);

        self.device.end_command_buffer(self.cmd_buf).unwrap();

        // Submit + wait
        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.cmd_buf));
        self.device.reset_fences(&[self.fence]).unwrap();
        self.device.queue_submit(self.queue, &[submit], self.fence).unwrap();
        self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();

        // Free descriptor set
        self.device.free_descriptor_sets(self.desc_pool, &[desc_set]).unwrap();
    }
}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            for (_, pool) in &self.buffer_pool {
                for buf in pool {
                    self.device.destroy_buffer(buf.buffer, None);
                    self.device.free_memory(buf.memory, None);
                }
            }
            for (_, cp) in &self.pipeline_cache {
                self.device.destroy_pipeline(cp.pipeline, None);
                self.device.destroy_pipeline_layout(cp.layout, None);
                self.device.destroy_descriptor_set_layout(cp.desc_set_layout, None);
            }
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
