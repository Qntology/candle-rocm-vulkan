//! Vulkan compute kernels for the [`super`] backend.
//!
//! Kernels are written in GLSL, compiled to SPIR-V at runtime with `naga`
//! (pure Rust, no `glslc` needed) and cached per device together with their
//! pipeline, so a kernel is only compiled the first time it is used. Every
//! dispatch reuses one command buffer / descriptor pool / fence, records a
//! compute -> host barrier and waits for completion, so results can be read
//! through the mapped memory right after the call returns.
//!
//! The kernels cover f32 tensors: unary and affine maps and binary ops on
//! arbitrary strided (broadcast) layouts, sum/max/min reductions, and a tiled
//! batched matmul. Anything else returns `None` and the caller runs the CPU
//! implementation on the mapped memory.
//!
//! Whether the kernels run is decided by [`native_enabled`]:
//! `CANDLE_VULKAN_NATIVE=1|0` (or the original `JOSHUA_VULKAN_NATIVE`) forces
//! them on or off; by default they run on integrated GPUs and whenever the
//! tensor memory is device local (unified memory), and stay off on discrete
//! GPUs whose tensors live in system memory behind a PCIe link, where the CPU
//! path is usually faster, and on software (CPU type) Vulkan devices. Small ops
//! always use the CPU path because a dispatch costs more than the work itself
//! (`CANDLE_VULKAN_NATIVE_MIN_ELEMS`, `CANDLE_VULKAN_NATIVE_MIN_FLOPS`).

use super::{VulkanDevice, VulkanStorage};
use crate::op::{CmpOp, ReduceOp};
use crate::{DType, Error, Layout, Result};
use ash::vk;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

const MAX_BINDINGS: u32 = 8;
/// Descriptor sets (dispatches) one recorded command buffer may use.
pub(crate) const MAX_SETS: u32 = 32;
const MAX_RANK: usize = 6;
const WG: u32 = 256;

static NATIVE_EXEC: AtomicUsize = AtomicUsize::new(0);
static NATIVE_OVERRIDE: AtomicU8 = AtomicU8::new(0);
static MIN_ELEMS_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);
static MIN_FLOPS_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn env_flag(names: &[&str]) -> Option<bool> {
    for n in names {
        if let Ok(v) = std::env::var(n) {
            let v = v.trim().to_ascii_lowercase();
            return Some(!(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no"));
        }
    }
    None
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn env_native() -> Option<bool> {
    static FLAG: OnceLock<Option<bool>> = OnceLock::new();
    *FLAG.get_or_init(|| env_flag(&["CANDLE_VULKAN_NATIVE", "JOSHUA_VULKAN_NATIVE"]))
}

fn debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| env_flag(&["CANDLE_VULKAN_DEBUG", "JOSHUA_VULKAN_DEBUG"]).unwrap_or(false))
}

fn min_elems() -> usize {
    let o = MIN_ELEMS_OVERRIDE.load(Ordering::Relaxed);
    if o != usize::MAX {
        return o;
    }
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_usize("CANDLE_VULKAN_NATIVE_MIN_ELEMS").unwrap_or(1 << 15))
}

fn min_flops() -> usize {
    let o = MIN_FLOPS_OVERRIDE.load(Ordering::Relaxed);
    if o != usize::MAX {
        return o;
    }
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_usize("CANDLE_VULKAN_NATIVE_MIN_FLOPS").unwrap_or(1 << 22))
}

/// Forces the GPU kernels on (`Some(true)`) or off (`Some(false)`) for the
/// whole process, `None` restores the environment / device default.
pub fn set_native_override(v: Option<bool>) {
    NATIVE_OVERRIDE.store(
        match v {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        },
        Ordering::Relaxed,
    );
}

/// Overrides the minimum sizes below which ops stay on the CPU path
/// (`None` restores the environment defaults).
pub fn set_native_thresholds(min_elems: Option<usize>, min_flops: Option<usize>) {
    MIN_ELEMS_OVERRIDE.store(min_elems.unwrap_or(usize::MAX), Ordering::Relaxed);
    MIN_FLOPS_OVERRIDE.store(min_flops.unwrap_or(usize::MAX), Ordering::Relaxed);
}

/// Whether GPU kernels are used on `dev`.
pub fn native_enabled(dev: &VulkanDevice) -> bool {
    if !dev.ctx().native_ok.load(Ordering::Relaxed) {
        return false;
    }
    match NATIVE_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    match env_native() {
        Some(v) => v,
        None => dev.ctx().default_native,
    }
}

/// Number of GPU kernel dispatches so far (diagnostic counter).
pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(Ordering::Relaxed)
}

fn log_fallback(op: &str, err: &Error) {
    if debug_enabled() {
        eprintln!("vulkan: {op} fell back to CPU: {err}");
    }
}

fn vk_err(dev: &VulkanDevice, what: &str, e: vk::Result) -> Error {
    if e == vk::Result::ERROR_DEVICE_LOST {
        dev.ctx().native_ok.store(false, Ordering::Relaxed);
        eprintln!("vulkan: device lost during {what}, GPU kernels disabled for this device");
    }
    Error::Msg(format!("vulkan {what} failed: {e:?}"))
}

// ---------------------------------------------------------------------------
// GLSL -> SPIR-V
// ---------------------------------------------------------------------------

fn glsl_to_spirv(source: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let options = Options::from(naga::ShaderStage::Compute);
    let mut frontend = Frontend::default();
    let module = frontend
        .parse(&options, source)
        .map_err(|e| Error::Msg(format!("vulkan shader parse failed: {e:?}")))?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(&module)
        .map_err(|e| Error::Msg(format!("vulkan shader validation failed: {e:?}")))?;
    let mut writer = spv::Writer::new(&spv::Options::default())
        .map_err(|e| Error::Msg(format!("vulkan spv writer init failed: {e:?}")))?;
    let pipeline_options = spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    let mut words = Vec::new();
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|e| Error::Msg(format!("vulkan spv write failed: {e:?}")))?;
    Ok(words)
}

// ---------------------------------------------------------------------------
// Pipelines and dispatch
// ---------------------------------------------------------------------------

/// A compiled compute pipeline, cached per device.
pub(crate) struct Kernel {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bindings: u32,
    push_size: u32,
}

impl Kernel {
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.module, None);
        }
    }

    fn new(dev: &VulkanDevice, spirv: &[u32], bindings: u32, push_size: u32) -> Result<Self> {
        let d = dev.device();
        let module = unsafe {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
        }
        .map_err(|e| vk_err(dev, "create_shader_module", e))?;
        let binding_list: Vec<_> = (0..bindings)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = match unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding_list),
                None,
            )
        } {
            Ok(l) => l,
            Err(e) => {
                unsafe { d.destroy_shader_module(module, None) };
                return Err(vk_err(dev, "create_descriptor_set_layout", e));
            }
        };
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_size,
        }];
        let set_layouts = [set_layout];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if push_size > 0 {
            layout_info = layout_info.push_constant_ranges(&ranges);
        }
        let layout = match unsafe { d.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                unsafe {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    d.destroy_shader_module(module, None);
                }
                return Err(vk_err(dev, "create_pipeline_layout", e));
            }
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline =
            match unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &info, None) } {
                Ok(p) => p[0],
                Err((_, e)) => {
                    unsafe {
                        d.destroy_pipeline_layout(layout, None);
                        d.destroy_descriptor_set_layout(set_layout, None);
                        d.destroy_shader_module(module, None);
                    }
                    return Err(vk_err(dev, "create_compute_pipelines", e));
                }
            };
        Ok(Kernel {
            module,
            set_layout,
            layout,
            pipeline,
            bindings,
            push_size,
        })
    }
}

/// Returns the cached kernel `key`, compiling `source()` on first use.
pub(crate) fn kernel(
    dev: &VulkanDevice,
    key: &str,
    bindings: u32,
    push_size: u32,
    source: impl FnOnce() -> String,
) -> Result<Arc<Kernel>> {
    let ctx = dev.ctx();
    let mut cache = ctx
        .kernels
        .lock()
        .map_err(|_| Error::Msg("vulkan: kernel cache poisoned".into()))?;
    if let Some(k) = cache.get(key) {
        return Ok(k.clone());
    }
    let spirv = glsl_to_spirv(&source())?;
    let k = Arc::new(Kernel::new(dev, &spirv, bindings, push_size)?);
    cache.insert(key.to_string(), k.clone());
    Ok(k)
}

/// Reusable command buffer, descriptor pool and fence. Guarded by the
/// context's `exec` mutex, which also serializes access to the queue.
pub(crate) struct ExecState {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    desc_pool: vk::DescriptorPool,
    fence: vk::Fence,
}

impl ExecState {
    pub(crate) fn new(device: &ash::Device, queue_family: u32) -> Result<Self> {
        let err = |what: &str, e: vk::Result| Error::Msg(format!("vulkan {what} failed: {e:?}"));
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| err("create_command_pool", e))?;
        let cmd = match unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(c) => c[0],
            Err(e) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(err("allocate_command_buffers", e));
            }
        };
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_BINDINGS * MAX_SETS,
        }];
        let desc_pool = match unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_SETS)
                    .pool_sizes(&sizes),
                None,
            )
        } {
            Ok(p) => p,
            Err(e) => {
                unsafe { device.destroy_command_pool(pool, None) };
                return Err(err("create_descriptor_pool", e));
            }
        };
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                unsafe {
                    device.destroy_descriptor_pool(desc_pool, None);
                    device.destroy_command_pool(pool, None);
                }
                return Err(err("create_fence", e));
            }
        };
        Ok(Self {
            pool,
            cmd,
            desc_pool,
            fence,
        })
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Records and runs one dispatch of `k` over `buffers` and waits for it.
fn run(
    dev: &VulkanDevice,
    k: &Kernel,
    buffers: &[&VulkanStorage],
    push: &[u8],
    groups: [u32; 3],
) -> Result<()> {
    if buffers.len() as u32 != k.bindings || push.len() as u32 != k.push_size {
        crate::bail!("vulkan dispatch: binding/push constant mismatch")
    }
    let limits = dev.limits();
    if push.len() as u32 > limits.max_push_constants_size {
        crate::bail!(
            "vulkan dispatch: {} bytes of push constants exceed the device limit",
            push.len()
        )
    }
    for (i, g) in groups.iter().enumerate() {
        if *g == 0 {
            return Ok(());
        }
        if *g > limits.max_workgroup_count[i] {
            crate::bail!("vulkan dispatch: {g} workgroups exceed the device limit on axis {i}")
        }
    }
    for b in buffers {
        if b.capacity_bytes as u64 > limits.max_storage_buffer_range {
            crate::bail!(
                "vulkan dispatch: buffer of {} bytes exceeds maxStorageBufferRange",
                b.capacity_bytes
            )
        }
    }
    let ctx = dev.ctx();
    let d = dev.device();
    let exec = ctx
        .exec
        .lock()
        .map_err(|_| Error::Msg("vulkan: exec state poisoned".into()))?;
    unsafe {
        d.reset_descriptor_pool(exec.desc_pool, vk::DescriptorPoolResetFlags::empty())
            .map_err(|e| vk_err(dev, "reset_descriptor_pool", e))?;
        let set_layouts = [k.set_layout];
        let set = d
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(exec.desc_pool)
                    .set_layouts(&set_layouts),
            )
            .map_err(|e| vk_err(dev, "allocate_descriptor_sets", e))?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo {
                    buffer: b.buffer,
                    offset: 0,
                    range: (b.capacity_bytes.max(4).div_ceil(4) * 4) as vk::DeviceSize,
                }]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        d.update_descriptor_sets(&writes, &[]);

        let cmd = exec.cmd;
        d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| vk_err(dev, "reset_command_buffer", e))?;
        d.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| vk_err(dev, "begin_command_buffer", e))?;
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[set],
            &[],
        );
        if !push.is_empty() {
            d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        }
        d.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::HOST_WRITE)];
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
        d.end_command_buffer(cmd)
            .map_err(|e| vk_err(dev, "end_command_buffer", e))?;
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        d.queue_submit(ctx.queue, &submit, exec.fence)
            .map_err(|e| vk_err(dev, "queue_submit", e))?;
        let wait = d.wait_for_fences(&[exec.fence], true, u64::MAX);
        let reset = d.reset_fences(&[exec.fence]);
        wait.map_err(|e| vk_err(dev, "wait_for_fences", e))?;
        reset.map_err(|e| vk_err(dev, "reset_fences", e))?;
    }
    NATIVE_EXEC.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// A storage buffer binding: `range` bytes of `buffer` from `offset`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bind {
    pub(crate) buffer: vk::Buffer,
    pub(crate) offset: u64,
    pub(crate) range: u64,
}

/// One step of a command sequence recorded by [`run_ops`].
pub(crate) enum Op<'a> {
    Copy {
        src: vk::Buffer,
        src_off: u64,
        dst: vk::Buffer,
        dst_off: u64,
        size: u64,
    },
    Dispatch {
        kernel: &'a Kernel,
        binds: Vec<Bind>,
        push: Vec<u8>,
        groups: [u32; 3],
    },
    /// Makes every earlier transfer / shader write visible to later transfers and shaders.
    Barrier,
}

/// Records `ops` into one command buffer, submits it and waits for completion. A host
/// barrier is appended so results are readable through mapped memory on return.
pub(crate) fn run_ops(dev: &VulkanDevice, ops: &[Op<'_>]) -> Result<()> {
    let limits = dev.limits();
    let mut dispatches = 0u32;
    for op in ops {
        if let Op::Dispatch {
            kernel,
            binds,
            push,
            groups,
        } = op
        {
            dispatches += 1;
            if binds.len() as u32 != kernel.bindings || push.len() as u32 != kernel.push_size {
                crate::bail!("vulkan dispatch: binding/push constant mismatch")
            }
            if push.len() as u32 > limits.max_push_constants_size {
                crate::bail!("vulkan dispatch: push constants exceed the device limit")
            }
            for (i, g) in groups.iter().enumerate() {
                if *g == 0 || *g > limits.max_workgroup_count[i] {
                    crate::bail!("vulkan dispatch: {g} workgroups on axis {i} out of range")
                }
            }
            for b in binds {
                if b.range == 0 || b.range > limits.max_storage_buffer_range {
                    crate::bail!(
                        "vulkan dispatch: binding of {} bytes outside maxStorageBufferRange",
                        b.range
                    )
                }
            }
        }
    }
    if dispatches > MAX_SETS {
        crate::bail!("vulkan run_ops: {dispatches} dispatches exceed {MAX_SETS} per submission")
    }
    if ops.is_empty() {
        return Ok(());
    }
    let ctx = dev.ctx();
    let d = dev.device();
    let exec = ctx
        .exec
        .lock()
        .map_err(|_| Error::Msg("vulkan: exec state poisoned".into()))?;
    let all_writes = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE;
    let all_stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    unsafe {
        d.reset_descriptor_pool(exec.desc_pool, vk::DescriptorPoolResetFlags::empty())
            .map_err(|e| vk_err(dev, "reset_descriptor_pool", e))?;
        let cmd = exec.cmd;
        d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| vk_err(dev, "reset_command_buffer", e))?;
        d.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| vk_err(dev, "begin_command_buffer", e))?;
        for op in ops {
            match op {
                Op::Copy {
                    src,
                    src_off,
                    dst,
                    dst_off,
                    size,
                } => {
                    if *size > 0 {
                        d.cmd_copy_buffer(
                            cmd,
                            *src,
                            *dst,
                            &[vk::BufferCopy {
                                src_offset: *src_off,
                                dst_offset: *dst_off,
                                size: *size,
                            }],
                        );
                    }
                }
                Op::Barrier => {
                    let b = [vk::MemoryBarrier::default()
                        .src_access_mask(all_writes)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::TRANSFER_READ
                                | vk::AccessFlags::TRANSFER_WRITE,
                        )];
                    d.cmd_pipeline_barrier(
                        cmd,
                        all_stages,
                        all_stages,
                        vk::DependencyFlags::empty(),
                        &b,
                        &[],
                        &[],
                    );
                }
                Op::Dispatch {
                    kernel,
                    binds,
                    push,
                    groups,
                } => {
                    let set_layouts = [kernel.set_layout];
                    let set = d
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(exec.desc_pool)
                                .set_layouts(&set_layouts),
                        )
                        .map_err(|e| vk_err(dev, "allocate_descriptor_sets", e))?[0];
                    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = binds
                        .iter()
                        .map(|b| {
                            [vk::DescriptorBufferInfo {
                                buffer: b.buffer,
                                offset: b.offset,
                                range: b.range,
                            }]
                        })
                        .collect();
                    let writes: Vec<_> = infos
                        .iter()
                        .enumerate()
                        .map(|(i, info)| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(i as u32)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(info)
                        })
                        .collect();
                    d.update_descriptor_sets(&writes, &[]);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                    d.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.layout,
                        0,
                        &[set],
                        &[],
                    );
                    if !push.is_empty() {
                        d.cmd_push_constants(
                            cmd,
                            kernel.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            push,
                        );
                    }
                    d.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                }
            }
        }
        let host = [vk::MemoryBarrier::default()
            .src_access_mask(all_writes)
            .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::HOST_WRITE)];
        d.cmd_pipeline_barrier(
            cmd,
            all_stages,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &host,
            &[],
            &[],
        );
        d.end_command_buffer(cmd)
            .map_err(|e| vk_err(dev, "end_command_buffer", e))?;
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        d.queue_submit(ctx.queue, &submit, exec.fence)
            .map_err(|e| vk_err(dev, "queue_submit", e))?;
        let wait = d.wait_for_fences(&[exec.fence], true, u64::MAX);
        let reset = d.reset_fences(&[exec.fence]);
        wait.map_err(|e| vk_err(dev, "wait_for_fences", e))?;
        reset.map_err(|e| vk_err(dev, "reset_fences", e))?;
    }
    NATIVE_EXEC.fetch_add(dispatches as usize, Ordering::Relaxed);
    Ok(())
}

pub(crate) fn wait_idle(dev: &VulkanDevice) -> Result<()> {
    let ctx = dev.ctx();
    let _exec = ctx
        .exec
        .lock()
        .map_err(|_| Error::Msg("vulkan: exec state poisoned".into()))?;
    unsafe { dev.device().queue_wait_idle(ctx.queue) }
        .map_err(|e| vk_err(dev, "queue_wait_idle", e))
}

struct Push(Vec<u8>);

impl Push {
    fn new() -> Self {
        Push(Vec::with_capacity(128))
    }
    fn u(mut self, v: usize) -> Self {
        self.0.extend_from_slice(&(v as u32).to_le_bytes());
        self
    }
    fn f(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn arr(mut self, vs: &[usize], len: usize) -> Self {
        for i in 0..len {
            let v = vs.get(i).copied().unwrap_or(0);
            self.0.extend_from_slice(&(v as u32).to_le_bytes());
        }
        self
    }
}

fn fits_u32(v: usize) -> bool {
    v <= u32::MAX as usize
}

fn groups_1d(dev: &VulkanDevice, n: usize) -> [u32; 3] {
    let max = dev.limits().max_workgroup_count[0].min(65535) as usize;
    let g = n.div_ceil(WG as usize).clamp(1, max);
    [g as u32, 1, 1]
}

/// Removes size one dims and merges dims that are contiguous with respect to
/// each other for every operand. Returns `None` when more than `MAX_RANK`
/// dims remain or an offset does not fit the kernels' 32 bit indexing.
fn collapse(dims: &[usize], strides: &[&[usize]]) -> Option<(Vec<usize>, Vec<Vec<usize>>)> {
    let mut out_dims: Vec<usize> = Vec::new();
    let mut out_strides: Vec<Vec<usize>> = vec![Vec::new(); strides.len()];
    for (i, &d) in dims.iter().enumerate() {
        if d == 1 {
            continue;
        }
        if let Some(last) = out_dims.last_mut() {
            let mergeable = strides
                .iter()
                .enumerate()
                .all(|(j, s)| *out_strides[j].last().unwrap() == s[i] * d);
            if mergeable {
                *last *= d;
                for (j, s) in strides.iter().enumerate() {
                    *out_strides[j].last_mut().unwrap() = s[i];
                }
                continue;
            }
        }
        out_dims.push(d);
        for (j, s) in strides.iter().enumerate() {
            out_strides[j].push(s[i]);
        }
    }
    if out_dims.is_empty() {
        out_dims.push(1);
        for s in out_strides.iter_mut() {
            s.push(0);
        }
    }
    if out_dims.len() > MAX_RANK {
        return None;
    }
    Some((out_dims, out_strides))
}

/// Largest element index reachable through `layout` (plus one).
pub(crate) fn span(layout: &Layout) -> usize {
    let mut max = layout.start_offset();
    for (d, s) in layout.dims().iter().zip(layout.stride().iter()) {
        if *d == 0 {
            return 0;
        }
        max += (d - 1) * s;
    }
    max + 1
}

fn ready(s: &VulkanStorage, n: usize) -> bool {
    s.dtype == DType::F32 && native_enabled(&s.device) && n >= min_elems() && fits_u32(n)
}

/// Runs `f` and turns an error into a CPU fallback.
fn attempt<T>(op: &str, f: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
    match f() {
        Ok(v) => Ok(Some(v)),
        Err(e) => {
            log_fallback(op, &e);
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------

const STRIDED_HEADER: &str = r#"
uint strided_index(uint i, uint rank, uint off) {
    uint idx = off;
    uint r = i;
    for (int d = int(rank) - 1; d >= 0; d--) {
        uint c = r % pc.dims[d];
        r = r / pc.dims[d];
        idx += c * pc.s0[d];
    }
    return idx;
}
"#;

const FUNCS: &str = r#"
float erf_approx(float x) {
    float s = sign(x);
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
float tanh_approx(float x) {
    float a = abs(x);
    if (a > 15.0) {
        return sign(x);
    }
    float e = exp(-2.0 * a);
    return sign(x) * (1.0 - e) / (1.0 + e);
}
float round_away(float x) {
    float t = trunc(x);
    return abs(x - t) >= 0.5 ? t + sign(x) : t;
}
float sign_of(float x) {
    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}
"#;

fn unary_expr(name: &str) -> Option<&'static str> {
    Some(match name {
        "exp" => "exp(v)",
        "log" => "log(v)",
        "tanh" => "tanh_approx(v)",
        "neg" => "-v",
        "recip" => "1.0 / v",
        "sqr" => "v * v",
        "sqrt" => "sqrt(v)",
        "abs" => "abs(v)",
        "relu" => "max(v, 0.0)",
        "silu" => "v / (1.0 + exp(-v))",
        "gelu" => {
            "0.5 * v * (1.0 + tanh_approx(0.7978845608028654 * v * (1.0 + 0.044715 * v * v)))"
        }
        "gelu_erf" => "0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476))",
        "erf" => "erf_approx(v)",
        "ceil" => "ceil(v)",
        "floor" => "floor(v)",
        "round" => "round_away(v)",
        "sign" => "sign_of(v)",
        "sigmoid" => "1.0 / (1.0 + exp(-v))",
        _ => return None,
    })
}

fn map_source(expr: &str) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer I {{ float x[]; }};
layout(set = 0, binding = 1) buffer O {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint n; uint rank; uint off; float p0; float p1;
    uint dims[6]; uint s0[6];
}} pc;
{STRIDED_HEADER}
{FUNCS}
void main() {{
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < pc.n; i += stride) {{
        float v = x[strided_index(i, pc.rank, pc.off)];
        y[i] = {expr};
    }}
}}
"#
    )
}

fn run_map(
    src: &VulkanStorage,
    layout: &Layout,
    key: &str,
    expr: &str,
    p0: f32,
    p1: f32,
) -> Result<Option<VulkanStorage>> {
    let n = layout.shape().elem_count();
    if !ready(src, n) || !fits_u32(span(layout)) {
        return Ok(None);
    }
    let Some((dims, strides)) = collapse(layout.dims(), &[layout.stride()]) else {
        return Ok(None);
    };
    attempt(key, || {
        let dev = &src.device;
        let k = kernel(dev, key, 2, 20 + 48, || map_source(expr))?;
        let out = dev.alloc_buffer(n, DType::F32)?;
        let push = Push::new()
            .u(n)
            .u(dims.len())
            .u(layout.start_offset())
            .f(p0)
            .f(p1)
            .arr(&dims, MAX_RANK)
            .arr(&strides[0], MAX_RANK);
        run(dev, &k, &[src, &out], &push.0, groups_1d(dev, n))?;
        Ok(out)
    })
}

pub(crate) fn unary(
    src: &VulkanStorage,
    name: &str,
    layout: &Layout,
) -> Result<Option<VulkanStorage>> {
    let Some(expr) = unary_expr(name) else {
        return Ok(None);
    };
    run_map(src, layout, &format!("unary_{name}"), expr, 0.0, 0.0)
}

pub(crate) fn affine(
    src: &VulkanStorage,
    layout: &Layout,
    mul: f64,
    add: f64,
) -> Result<Option<VulkanStorage>> {
    run_map(
        src,
        layout,
        "affine",
        "v * pc.p0 + pc.p1",
        mul as f32,
        add as f32,
    )
}

pub(crate) fn powf(_: &VulkanStorage, _: &Layout, _: f64) -> Result<Option<VulkanStorage>> {
    Ok(None)
}

pub(crate) fn elu(_: &VulkanStorage, _: &Layout, _: f64) -> Result<Option<VulkanStorage>> {
    Ok(None)
}

fn binary_expr(name: &str) -> Option<&'static str> {
    Some(match name {
        "add" => "a + b",
        "sub" => "a - b",
        "mul" => "a * b",
        "div" => "a / b",
        "minimum" => "a > b ? b : a",
        "maximum" => "a < b ? b : a",
        _ => return None,
    })
}

fn binary_source(expr: &str) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer L {{ float lhs[]; }};
layout(set = 0, binding = 1) readonly buffer R {{ float rhs[]; }};
layout(set = 0, binding = 2) buffer O {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint n; uint rank; uint off_l; uint off_r;
    uint dims[6]; uint s0[6]; uint s1[6];
}} pc;
void main() {{
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < pc.n; i += stride) {{
        uint il = pc.off_l;
        uint ir = pc.off_r;
        uint r = i;
        for (int d = int(pc.rank) - 1; d >= 0; d--) {{
            uint c = r % pc.dims[d];
            r = r / pc.dims[d];
            il += c * pc.s0[d];
            ir += c * pc.s1[d];
        }}
        float a = lhs[il];
        float b = rhs[ir];
        y[i] = {expr};
    }}
}}
"#
    )
}

pub(crate) fn binary(
    lhs: &VulkanStorage,
    name: &str,
    rhs: &VulkanStorage,
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Option<VulkanStorage>> {
    let n = lhs_l.shape().elem_count();
    if !ready(lhs, n)
        || rhs.dtype != DType::F32
        || lhs_l.dims() != rhs_l.dims()
        || !fits_u32(span(lhs_l))
        || !fits_u32(span(rhs_l))
    {
        return Ok(None);
    }
    let Some(expr) = binary_expr(name) else {
        return Ok(None);
    };
    let Some((dims, strides)) = collapse(lhs_l.dims(), &[lhs_l.stride(), rhs_l.stride()]) else {
        return Ok(None);
    };
    let key = format!("binary_{name}");
    attempt(&key, || {
        let dev = &lhs.device;
        let k = kernel(dev, &key, 3, 16 + 72, || binary_source(expr))?;
        let out = dev.alloc_buffer(n, DType::F32)?;
        let push = Push::new()
            .u(n)
            .u(dims.len())
            .u(lhs_l.start_offset())
            .u(rhs_l.start_offset())
            .arr(&dims, MAX_RANK)
            .arr(&strides[0], MAX_RANK)
            .arr(&strides[1], MAX_RANK);
        run(dev, &k, &[lhs, rhs, &out], &push.0, groups_1d(dev, n))?;
        Ok(out)
    })
}

pub(crate) fn cmp(
    _: &VulkanStorage,
    _: CmpOp,
    _: &VulkanStorage,
    _: &Layout,
    _: &Layout,
) -> Result<Option<VulkanStorage>> {
    Ok(None)
}

pub(crate) fn to_dtype(_: &VulkanStorage, _: &Layout, _: DType) -> Result<Option<VulkanStorage>> {
    Ok(None)
}

pub(crate) fn where_cond(
    _: &VulkanStorage,
    _: &Layout,
    _: &VulkanStorage,
    _: &Layout,
    _: &VulkanStorage,
    _: &Layout,
) -> Result<Option<VulkanStorage>> {
    Ok(None)
}

pub(crate) fn index_select(
    _: &VulkanStorage,
    _: &VulkanStorage,
    _: &Layout,
    _: &Layout,
    _: usize,
) -> Result<Option<VulkanStorage>> {
    Ok(None)
}

pub(crate) fn const_set(
    _: &mut VulkanStorage,
    _: crate::scalar::Scalar,
    _: &Layout,
) -> Result<bool> {
    Ok(false)
}

pub(crate) fn copy_strided(
    _: &VulkanStorage,
    _: &mut VulkanStorage,
    _: usize,
    _: &Layout,
) -> Result<bool> {
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn copy2d(
    _: &VulkanStorage,
    _: &mut VulkanStorage,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
) -> Result<bool> {
    Ok(false)
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

fn reduce_source(op: &str) -> String {
    let (init, combine) = match op {
        "sum" => ("0.0", "acc + v"),
        "max" => ("-1.0 / 0.0", "v > acc ? v : acc"),
        _ => ("1.0 / 0.0", "v < acc ? v : acc"),
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer I {{ float x[]; }};
layout(set = 0, binding = 1) buffer O {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint rows; uint cols; uint off; uint row_stride;
}} pc;
shared float part[256];
float combine(float acc, float v) {{
    return {combine};
}}
void main() {{
    uint t = gl_LocalInvocationID.x;
    for (uint row = gl_WorkGroupID.x; row < pc.rows; row += gl_NumWorkGroups.x) {{
        uint base = pc.off + row * pc.row_stride;
        float acc = {init};
        for (uint c = t; c < pc.cols; c += 256u) {{
            acc = combine(acc, x[base + c]);
        }}
        part[t] = acc;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {{
            if (t < s) {{
                part[t] = combine(part[t], part[t + s]);
            }}
            barrier();
        }}
        if (t == 0u) {{
            y[row] = part[0];
        }}
        barrier();
    }}
}}
"#
    )
}

/// Sum / max / min over the last dimension of a row major (possibly offset)
/// f32 tensor.
pub(crate) fn reduce(
    src: &VulkanStorage,
    op: ReduceOp,
    layout: &Layout,
    reduce_dims: &[usize],
) -> Result<Option<VulkanStorage>> {
    let n = layout.shape().elem_count();
    let rank = layout.dims().len();
    if !ready(src, n) || n == 0 || rank == 0 || !layout.is_contiguous() {
        return Ok(None);
    }
    if reduce_dims != [rank - 1] {
        return Ok(None);
    }
    let name = match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        _ => return Ok(None),
    };
    let cols = layout.dims()[rank - 1];
    if cols == 0 || !fits_u32(span(layout)) {
        return Ok(None);
    }
    let rows = n / cols;
    let key = format!("reduce_{name}");
    attempt(&key, || {
        let dev = &src.device;
        let k = kernel(dev, &key, 2, 16, || reduce_source(name))?;
        let out = dev.alloc_buffer(rows, DType::F32)?;
        let push = Push::new().u(rows).u(cols).u(layout.start_offset()).u(cols);
        let g = rows.clamp(1, dev.limits().max_workgroup_count[0].min(65535) as usize);
        run(dev, &k, &[src, &out], &push.0, [g as u32, 1, 1])?;
        Ok(out)
    })
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

/// Tiled f32 matmul: each 16x16 workgroup computes a 64x64 tile of
/// `C[b] = A[b] @ B[b]`, every thread a 4x4 block, with 16 wide K slices of
/// A and B staged in shared memory. A and B are read through arbitrary
/// row/column strides (transposed operands need no copy) and up to two batch
/// dims with their own strides (0 for broadcast batches). C is row major.
pub(crate) const GLSL_MATMUL: &str = r#"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float b[]; };
layout(set = 0, binding = 2) buffer C { float c[]; };
layout(push_constant, std430) uniform PC {
    uint m; uint n; uint k;
    uint sam; uint sak; uint sbk; uint sbn;
    uint off_a; uint off_b;
    uint b_inner;
    uint sa_outer; uint sa_inner; uint sb_outer; uint sb_inner;
    uint row0; uint z0;
} pc;
shared float As[1040];
shared float Bs[1040];
void store_row(uint cbase, uint gr, uint col, vec4 v) {
    if (gr >= pc.m) {
        return;
    }
    uint o = cbase + gr * pc.n;
    if (col < pc.n) { c[o + col] = v.x; }
    if (col + 1u < pc.n) { c[o + col + 1u] = v.y; }
    if (col + 2u < pc.n) { c[o + col + 2u] = v.z; }
    if (col + 3u < pc.n) { c[o + col + 3u] = v.w; }
}
void main() {
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tid = ty * 16u + tx;
    uint z = pc.z0 + gl_WorkGroupID.z;
    uint zo = z / pc.b_inner;
    uint zi = z - zo * pc.b_inner;
    uint abase = pc.off_a + zo * pc.sa_outer + zi * pc.sa_inner;
    uint bbase = pc.off_b + zo * pc.sb_outer + zi * pc.sb_inner;
    uint cbase = z * pc.m * pc.n;
    uint row_base = pc.row0 + gl_WorkGroupID.y * 64u;
    uint col_base = gl_WorkGroupID.x * 64u;
    vec4 c0 = vec4(0.0);
    vec4 c1 = vec4(0.0);
    vec4 c2 = vec4(0.0);
    vec4 c3 = vec4(0.0);
    for (uint k0 = 0u; k0 < pc.k; k0 += 16u) {
        for (uint l = 0u; l < 4u; l++) {
            uint e = tid + l * 256u;
            uint kk = e % 16u;
            uint rr = e / 16u;
            uint gr = row_base + rr;
            uint gk = k0 + kk;
            float av = 0.0;
            if (gr < pc.m && gk < pc.k) {
                av = a[abase + gr * pc.sam + gk * pc.sak];
            }
            As[kk * 65u + rr] = av;
            uint cc = e % 64u;
            uint kb = e / 64u;
            uint gc = col_base + cc;
            uint gkb = k0 + kb;
            float bv = 0.0;
            if (gc < pc.n && gkb < pc.k) {
                bv = b[bbase + gkb * pc.sbk + gc * pc.sbn];
            }
            Bs[kb * 65u + cc] = bv;
        }
        barrier();
        for (uint kk = 0u; kk < 16u; kk++) {
            uint ao = kk * 65u + ty * 4u;
            uint bo = kk * 65u + tx * 4u;
            vec4 bvec = vec4(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
            c0 += As[ao] * bvec;
            c1 += As[ao + 1u] * bvec;
            c2 += As[ao + 2u] * bvec;
            c3 += As[ao + 3u] * bvec;
        }
        barrier();
    }
    uint gr0 = row_base + ty * 4u;
    uint col = col_base + tx * 4u;
    store_row(cbase, gr0, col, c0);
    store_row(cbase, gr0 + 1u, col, c1);
    store_row(cbase, gr0 + 2u, col, c2);
    store_row(cbase, gr0 + 3u, col, c3);
}
"#;

/// Batch dims (in front of the last two) of a matmul operand as
/// `(outer, inner, outer_stride, inner_stride)`; a batch that collapses to a
/// single linear dim is reported as `(1, batch, 0, stride)`.
pub(crate) fn batch_strides(layout: &Layout, batch: usize) -> Option<(usize, usize, usize, usize)> {
    let dims = layout.dims();
    let stride = layout.stride();
    let r = dims.len();
    let bdims = &dims[..r - 2];
    let bstr = &stride[..r - 2];
    if bdims.iter().product::<usize>() != batch {
        return None;
    }
    if batch == 1 {
        return Some((1, 1, 0, 0));
    }
    let (cd, cs) = collapse(bdims, &[bstr])?;
    match cd.len() {
        1 => Some((1, cd[0], 0, cs[0][0])),
        2 => Some((cd[0], cd[1], cs[0][0], cs[0][1])),
        _ => None,
    }
}

/// Expresses both operands' batch dims with one `(outer, inner)` split.
/// Returns `(inner, a_outer, a_inner, b_outer, b_inner)` strides.
pub(crate) fn unify_batches(
    a: (usize, usize, usize, usize),
    b: (usize, usize, usize, usize),
) -> Option<(usize, usize, usize, usize, usize)> {
    match (a.0 == 1, b.0 == 1) {
        (true, true) => Some((a.1, 0, a.3, 0, b.3)),
        (true, false) => Some((b.1, b.1 * a.3, a.3, b.2, b.3)),
        (false, true) => Some((a.1, a.2, a.3, a.1 * b.3, b.3)),
        (false, false) if (a.0, a.1) == (b.0, b.1) => Some((a.1, a.2, a.3, b.2, b.3)),
        _ => None,
    }
}

pub(crate) fn matmul(
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    (batch, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Option<VulkanStorage>> {
    let flops = 2usize
        .saturating_mul(batch)
        .saturating_mul(m)
        .saturating_mul(n)
        .saturating_mul(k);
    if lhs.dtype != DType::F32
        || rhs.dtype != DType::F32
        || !native_enabled(&lhs.device)
        || flops < min_flops()
        || lhs_l.dims().len() < 2
        || rhs_l.dims().len() < 2
        || !fits_u32(span(lhs_l))
        || !fits_u32(span(rhs_l))
        || !fits_u32(batch * m * n)
    {
        return Ok(None);
    }
    let (Some(a), Some(b)) = (batch_strides(lhs_l, batch), batch_strides(rhs_l, batch)) else {
        return Ok(None);
    };
    let Some((inner, sao, sai, sbo, sbi)) = unify_batches(a, b) else {
        return Ok(None);
    };
    let ls = lhs_l.stride();
    let rs = rhs_l.stride();
    let lr = ls.len();
    let rr = rs.len();
    let (sam, sak) = (ls[lr - 2], ls[lr - 1]);
    let (sbk, sbn) = (rs[rr - 2], rs[rr - 1]);
    attempt("matmul", || {
        let dev = &lhs.device;
        let k_ = kernel(dev, "matmul_tiled", 3, 64, || GLSL_MATMUL.to_string())?;
        let out = dev.alloc_buffer(batch * m * n, DType::F32)?;
        if m == 0 || n == 0 || batch == 0 {
            return Ok(out);
        }
        if k == 0 {
            let mut out = out;
            out.as_bytes_mut().fill(0);
            return Ok(out);
        }
        let limits = dev.limits();
        let gx = n.div_ceil(64);
        if gx > limits.max_workgroup_count[0] as usize {
            crate::bail!("matmul: n too large for one dispatch")
        }
        let max_band_flops: usize = 1 << 31;
        let row_flops = 2 * n * k;
        let rows_per_band = ((max_band_flops / row_flops.max(1)) / 64 * 64)
            .clamp(64, (limits.max_workgroup_count[1] as usize) * 64);
        let batch_per_band = (max_band_flops / (row_flops * m.min(rows_per_band)).max(1))
            .clamp(1, (limits.max_workgroup_count[2] as usize).min(65535));
        let mut z0 = 0;
        while z0 < batch {
            let zb = batch_per_band.min(batch - z0);
            let mut r0 = 0;
            while r0 < m {
                let rb = rows_per_band.min(m - r0);
                let push = Push::new()
                    .u(m)
                    .u(n)
                    .u(k)
                    .u(sam)
                    .u(sak)
                    .u(sbk)
                    .u(sbn)
                    .u(lhs_l.start_offset())
                    .u(rhs_l.start_offset())
                    .u(inner)
                    .u(sao)
                    .u(sai)
                    .u(sbo)
                    .u(sbi)
                    .u(r0)
                    .u(z0);
                run(
                    dev,
                    &k_,
                    &[lhs, rhs, &out],
                    &push.0,
                    [gx as u32, rb.div_ceil(64) as u32, zb as u32],
                )?;
                r0 += rb;
            }
            z0 += zb;
        }
        Ok(out)
    })
}

// ---------------------------------------------------------------------------
// Row kernels used by candle-nn (softmax, rms-norm, layer-norm)
// ---------------------------------------------------------------------------

const ROW_REDUCE: &str = r#"
shared float part[256];
float row_sum(float v) {
    uint t = gl_LocalInvocationID.x;
    part[t] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            part[t] = part[t] + part[t + s];
        }
        barrier();
    }
    float r = part[0];
    barrier();
    return r;
}
float row_max(float v) {
    uint t = gl_LocalInvocationID.x;
    part[t] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            part[t] = max(part[t], part[t + s]);
        }
        barrier();
    }
    float r = part[0];
    barrier();
    return r;
}
"#;

fn softmax_source() -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer I {{ float x[]; }};
layout(set = 0, binding = 1) buffer O {{ float y[]; }};
layout(push_constant, std430) uniform PC {{ uint rows; uint cols; uint off; }} pc;
{ROW_REDUCE}
void main() {{
    uint t = gl_LocalInvocationID.x;
    for (uint row = gl_WorkGroupID.x; row < pc.rows; row += gl_NumWorkGroups.x) {{
        uint base = pc.off + row * pc.cols;
        uint obase = row * pc.cols;
        float m = -1.0 / 0.0;
        for (uint c = t; c < pc.cols; c += 256u) {{
            m = max(m, x[base + c]);
        }}
        m = row_max(m);
        float s = 0.0;
        for (uint c = t; c < pc.cols; c += 256u) {{
            float e = exp(x[base + c] - m);
            y[obase + c] = e;
            s += e;
        }}
        s = row_sum(s);
        for (uint c = t; c < pc.cols; c += 256u) {{
            y[obase + c] = y[obase + c] / s;
        }}
    }}
}}
"#
    )
}

fn rms_norm_source() -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer I {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer W {{ float w[]; }};
layout(set = 0, binding = 2) buffer O {{ float y[]; }};
layout(push_constant, std430) uniform PC {{ uint rows; uint cols; uint off; uint off_w; float eps; }} pc;
{ROW_REDUCE}
void main() {{
    uint t = gl_LocalInvocationID.x;
    for (uint row = gl_WorkGroupID.x; row < pc.rows; row += gl_NumWorkGroups.x) {{
        uint base = pc.off + row * pc.cols;
        uint obase = row * pc.cols;
        float s = 0.0;
        for (uint c = t; c < pc.cols; c += 256u) {{
            float v = x[base + c];
            s += v * v;
        }}
        s = row_sum(s);
        float m = sqrt(s / float(pc.cols) + pc.eps);
        for (uint c = t; c < pc.cols; c += 256u) {{
            y[obase + c] = x[base + c] / m * w[pc.off_w + c];
        }}
    }}
}}
"#
    )
}

fn layer_norm_source() -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer I {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer W {{ float w[]; }};
layout(set = 0, binding = 2) readonly buffer Bi {{ float bias[]; }};
layout(set = 0, binding = 3) buffer O {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint rows; uint cols; uint off; uint off_w; uint off_b; float eps;
}} pc;
{ROW_REDUCE}
void main() {{
    uint t = gl_LocalInvocationID.x;
    for (uint row = gl_WorkGroupID.x; row < pc.rows; row += gl_NumWorkGroups.x) {{
        uint base = pc.off + row * pc.cols;
        uint obase = row * pc.cols;
        float s = 0.0;
        float s2 = 0.0;
        for (uint c = t; c < pc.cols; c += 256u) {{
            float v = x[base + c];
            s += v;
            s2 += v * v;
        }}
        s = row_sum(s);
        s2 = row_sum(s2);
        float mean = s / float(pc.cols);
        float var = s2 / float(pc.cols) - mean * mean;
        float inv_std = 1.0 / sqrt(var + pc.eps);
        for (uint c = t; c < pc.cols; c += 256u) {{
            y[obase + c] = (x[base + c] - mean) * inv_std * w[pc.off_w + c] + bias[pc.off_b + c];
        }}
    }}
}}
"#
    )
}

fn row_groups(dev: &VulkanDevice, rows: usize) -> [u32; 3] {
    let g = rows.clamp(1, dev.limits().max_workgroup_count[0].min(65535) as usize);
    [g as u32, 1, 1]
}

/// `(rows, cols, start_offset)` of a contiguous f32 layout, or `None`.
fn rows_of(s: &VulkanStorage, layout: &Layout) -> Option<(usize, usize, usize)> {
    let n = layout.shape().elem_count();
    let rank = layout.dims().len();
    if !ready(s, n) || n == 0 || rank == 0 || !layout.is_contiguous() || !fits_u32(span(layout)) {
        return None;
    }
    let cols = layout.dims()[rank - 1];
    if cols == 0 {
        return None;
    }
    Some((n / cols, cols, layout.start_offset()))
}

fn vector_of(s: &VulkanStorage, layout: &Layout, cols: usize) -> Option<usize> {
    if s.dtype != DType::F32 || !layout.is_contiguous() || layout.shape().elem_count() != cols {
        return None;
    }
    Some(layout.start_offset())
}

pub(crate) fn softmax_last_dim(
    src: &VulkanStorage,
    layout: &Layout,
) -> Result<Option<VulkanStorage>> {
    let Some((rows, cols, off)) = rows_of(src, layout) else {
        return Ok(None);
    };
    attempt("softmax_last_dim", || {
        let dev = &src.device;
        let k = kernel(dev, "softmax_last_dim", 2, 12, softmax_source)?;
        let out = dev.alloc_buffer(rows * cols, DType::F32)?;
        let push = Push::new().u(rows).u(cols).u(off);
        run(dev, &k, &[src, &out], &push.0, row_groups(dev, rows))?;
        Ok(out)
    })
}

pub(crate) fn rms_norm(
    src: &VulkanStorage,
    layout: &Layout,
    alpha: &VulkanStorage,
    alpha_l: &Layout,
    eps: f32,
) -> Result<Option<VulkanStorage>> {
    let Some((rows, cols, off)) = rows_of(src, layout) else {
        return Ok(None);
    };
    let Some(off_w) = vector_of(alpha, alpha_l, cols) else {
        return Ok(None);
    };
    attempt("rms_norm", || {
        let dev = &src.device;
        let k = kernel(dev, "rms_norm", 3, 20, rms_norm_source)?;
        let out = dev.alloc_buffer(rows * cols, DType::F32)?;
        let push = Push::new().u(rows).u(cols).u(off).u(off_w).f(eps);
        run(dev, &k, &[src, alpha, &out], &push.0, row_groups(dev, rows))?;
        Ok(out)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm(
    src: &VulkanStorage,
    layout: &Layout,
    alpha: &VulkanStorage,
    alpha_l: &Layout,
    beta: &VulkanStorage,
    beta_l: &Layout,
    eps: f32,
) -> Result<Option<VulkanStorage>> {
    let Some((rows, cols, off)) = rows_of(src, layout) else {
        return Ok(None);
    };
    let (Some(off_w), Some(off_b)) = (
        vector_of(alpha, alpha_l, cols),
        vector_of(beta, beta_l, cols),
    ) else {
        return Ok(None);
    };
    attempt("layer_norm", || {
        let dev = &src.device;
        let k = kernel(dev, "layer_norm", 4, 24, layer_norm_source)?;
        let out = dev.alloc_buffer(rows * cols, DType::F32)?;
        let push = Push::new().u(rows).u(cols).u(off).u(off_w).u(off_b).f(eps);
        run(
            dev,
            &k,
            &[src, alpha, beta, &out],
            &push.0,
            row_groups(dev, rows),
        )?;
        Ok(out)
    })
}
