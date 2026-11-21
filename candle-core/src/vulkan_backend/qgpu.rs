use super::shaders::{self, Bind, Op, GLSL_PACK16_HELPERS};
use super::{VulkanContext, VulkanDevice, VulkanStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::quantized::GgmlDType;
use crate::{DType, Error, Layout, Result};
use ash::vk;
use half::f16;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

fn env_on(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no" || v == "cpu")
        }
        Err(_) => default,
    }
}

fn env_mb(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn weights_enabled_env() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_on("CANDLE_VULKAN_GPU_WEIGHTS", true))
}

fn mirrors_enabled_env() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_on("CANDLE_VULKAN_DENSE_MIRROR", true))
}

fn act_enabled_env() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_on("CANDLE_VULKAN_ACT_MATMUL", true))
}

fn act_min_flops() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_mb("CANDLE_VULKAN_ACT_MIN_MFLOP", 256) * 1_000_000)
}

fn scratch_cap() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_mb("CANDLE_VULKAN_SCRATCH_MB", 128).max(8) << 20)
}

fn vram_reserve() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| (env_mb("CANDLE_VULKAN_VRAM_RESERVE_MB", 512) as u64) << 20)
}

pub(crate) fn enabled(dev: &VulkanDevice) -> bool {
    shaders::exec_mode() != shaders::ExecMode::Host
        && weights_enabled_env()
        && dev.ctx().device_type != vk::PhysicalDeviceType::CPU
        && dev.ctx().native_ok.load(Ordering::Relaxed)
}

fn pretend_integrated() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_on("CANDLE_VULKAN_PRETEND_INTEGRATED", false))
}

fn unified(ctx: &VulkanContext) -> bool {
    ctx.default_native || pretend_integrated()
}

#[derive(Default)]
pub(crate) struct GpuState {
    pub(crate) local_bytes: AtomicUsize,
    pub(crate) shared_bytes: AtomicUsize,
    pub(crate) buffers: AtomicUsize,
    scratch: Mutex<Option<RawBuf>>,
}

pub(crate) struct RawBuf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    local: bool,
}

impl GpuState {
    pub(crate) fn release_scratch(&self, device: &ash::Device) {
        if let Ok(mut s) = self.scratch.lock() {
            if let Some(r) = s.take() {
                self.free_raw(device, r);
            }
        }
    }

    fn free_raw(&self, device: &ash::Device, r: RawBuf) {
        unsafe {
            device.destroy_buffer(r.buffer, None);
            device.free_memory(r.memory, None);
        }
        self.sub(r.size, r.local);
    }

    fn add(&self, size: usize, local: bool) {
        if local {
            self.local_bytes.fetch_add(size, Ordering::Relaxed);
        } else {
            self.shared_bytes.fetch_add(size, Ordering::Relaxed);
        }
        self.buffers.fetch_add(1, Ordering::Relaxed);
    }

    fn sub(&self, size: usize, local: bool) {
        if local {
            self.local_bytes.fetch_sub(size, Ordering::Relaxed);
        } else {
            self.shared_bytes.fetch_sub(size, Ordering::Relaxed);
        }
        self.buffers.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Use {
    Weights,
    Scratch,
}

pub(crate) fn heap_budgets(
    ctx: &VulkanContext,
) -> Option<([u64; vk::MAX_MEMORY_HEAPS], [u64; vk::MAX_MEMORY_HEAPS])> {
    if !ctx.memory_budget_ext {
        return None;
    }
    let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
    unsafe {
        ctx.instance
            .get_physical_device_memory_properties2(ctx.physical, &mut props2)
    };
    let budget = b.heap_budget;
    let usage = b.heap_usage;
    if budget.iter().all(|&x| x == 0) {
        return None;
    }
    Some((budget, usage))
}

fn is_discrete(ctx: &VulkanContext) -> bool {
    ctx.device_type == vk::PhysicalDeviceType::DISCRETE_GPU && !pretend_integrated()
}

fn candidates(ctx: &VulkanContext, use_: Use, type_bits: u32) -> Vec<u32> {
    let mp = &ctx.memory_properties;
    let discrete = is_discrete(ctx);
    let mut v: Vec<(u32, u64, u32)> = Vec::new();
    for t in 0..mp.memory_type_count {
        if type_bits & (1 << t) == 0 {
            continue;
        }
        let f = mp.memory_types[t as usize].property_flags;
        if f.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
            || f.contains(vk::MemoryPropertyFlags::PROTECTED)
        {
            continue;
        }
        let dl = f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let hv = f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        if hv && !f.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            continue;
        }
        let heap = mp.memory_types[t as usize].heap_index as usize;
        let hsize = mp.memory_heaps[heap].size;
        let score = match (use_, discrete, dl, hv) {
            (Use::Scratch, _, true, false) => 0,
            (Use::Scratch, _, true, true) => 1,
            (Use::Scratch, _, false, _) => continue,
            (Use::Weights, true, true, false) => 0,
            (Use::Weights, true, true, true) if hsize >= (1 << 30) => 1,
            (Use::Weights, true, _, _) => continue,
            (Use::Weights, false, true, true) => 0,
            (Use::Weights, false, false, true) => 1,
            (Use::Weights, false, true, false) => 2,
            (Use::Weights, false, false, false) => continue,
        };
        v.push((score, hsize, t));
    }
    v.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    v.into_iter().map(|(_, _, t)| t).collect()
}

fn heap_fits(
    ctx: &VulkanContext,
    heap: usize,
    size: u64,
    budgets: &Option<([u64; vk::MAX_MEMORY_HEAPS], [u64; vk::MAX_MEMORY_HEAPS])>,
) -> bool {
    let h = ctx.memory_properties.memory_heaps[heap];
    let local = h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
    let reserve = if local {
        vram_reserve().min(h.size / 4)
    } else {
        (1u64 << 30).min(h.size / 4)
    };
    match budgets {
        Some((b, u)) => u[heap].saturating_add(size).saturating_add(reserve) <= b[heap],
        None => {
            let own = if local {
                ctx.gpu.local_bytes.load(Ordering::Relaxed)
            } else {
                ctx.gpu.shared_bytes.load(Ordering::Relaxed)
            } as u64;
            own.saturating_add(size).saturating_add(reserve) <= h.size
        }
    }
}

pub(crate) fn working_heaps(ctx: &VulkanContext) -> Vec<usize> {
    let mut heaps: Vec<usize> = Vec::new();
    for t in candidates(ctx, Use::Weights, u32::MAX) {
        let h = ctx.memory_properties.memory_types[t as usize].heap_index as usize;
        if !heaps.contains(&h) {
            heaps.push(h);
        }
    }
    heaps
}

fn create_buffer(ctx: &VulkanContext, size: usize) -> Result<vk::Buffer> {
    unsafe {
        ctx.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size as vk::DeviceSize)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|e| Error::Msg(format!("vulkan create_buffer ({size} bytes) failed: {e:?}")))
}

fn alloc_raw(
    ctx: &VulkanContext,
    size: usize,
    use_: Use,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, bool, usize)> {
    let size = size.max(256).div_ceil(256) * 256;
    let buffer = create_buffer(ctx, size)?;
    let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
    let budgets = heap_budgets(ctx);
    let mut last_err = None;
    for t in candidates(ctx, use_, req.memory_type_bits) {
        let mt = ctx.memory_properties.memory_types[t as usize];
        let heap = mt.heap_index as usize;
        if !heap_fits(ctx, heap, req.size, &budgets) {
            continue;
        }
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(t);
        let memory = match unsafe { ctx.device.allocate_memory(&info, None) } {
            Ok(m) => m,
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        };
        if let Err(e) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe { ctx.device.free_memory(memory, None) };
            last_err = Some(e);
            continue;
        }
        let hv = mt
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let mapped = if hv {
            match unsafe {
                ctx.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(p) => p as *mut u8,
                Err(e) => {
                    unsafe { ctx.device.free_memory(memory, None) };
                    last_err = Some(e);
                    continue;
                }
            }
        } else {
            std::ptr::null_mut()
        };
        let local = ctx.memory_properties.memory_heaps[heap]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        ctx.gpu.add(req.size as usize, local);
        return Ok((buffer, memory, mapped, local, req.size as usize));
    }
    unsafe { ctx.device.destroy_buffer(buffer, None) };
    Err(Error::Msg(format!(
        "vulkan: no GPU memory with room for {size} bytes ({use_:?}, last error {last_err:?})"
    )))
}

pub(crate) struct GpuBuf {
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) size: usize,
    pub(crate) mapped: *mut u8,
    pub(crate) local: bool,
    dev: VulkanDevice,
}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

impl GpuBuf {
    pub(crate) fn new(dev: &VulkanDevice, size: usize, use_: Use) -> Result<Self> {
        let (buffer, memory, mapped, local, size) = alloc_raw(dev.ctx(), size, use_)?;
        Ok(Self {
            buffer,
            memory,
            size,
            mapped,
            local,
            dev: dev.clone(),
        })
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        let ctx = self.dev.ctx();
        if ctx.has_pending_gpu_work.load(Ordering::Acquire) {
            let _ = shaders::flush(&self.dev);
        }
        unsafe {
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
        ctx.gpu.sub(self.size, self.local);
    }
}

impl std::fmt::Debug for GpuBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuf")
            .field("size", &self.size)
            .field("local", &self.local)
            .finish()
    }
}

const STAGING_CHUNK: usize = 32 << 20;

pub(crate) fn write(dev: &VulkanDevice, dst: &GpuBuf, offset: usize, data: &[u8]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    if offset + data.len() > dst.size {
        crate::bail!("vulkan gpu write out of bounds")
    }
    if !dst.mapped.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst.mapped.add(offset), data.len()) };
        return Ok(());
    }
    let chunk = STAGING_CHUNK.min(data.len());
    let mut staging = dev.alloc_buffer(chunk, DType::U8)?;
    let mut done = 0;
    while done < data.len() {
        let n = chunk.min(data.len() - done);
        staging.as_bytes_mut()[..n].copy_from_slice(&data[done..done + n]);
        shaders::run_ops_sync(
            dev,
            &[Op::Copy {
                src: staging.buffer,
                src_off: 0,
                dst: dst.buffer,
                dst_off: (offset + done) as u64,
                size: n as u64,
            }],
        )?;
        done += n;
    }
    Ok(())
}

pub(crate) fn read(dev: &VulkanDevice, src: &GpuBuf, offset: usize, out: &mut [u8]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    if offset + out.len() > src.size {
        crate::bail!("vulkan gpu read out of bounds")
    }
    if !src.mapped.is_null() {
        if dev.ctx().has_pending_gpu_work.load(Ordering::Acquire) {
            shaders::flush(dev)?;
        }
        unsafe { std::ptr::copy_nonoverlapping(src.mapped.add(offset), out.as_mut_ptr(), out.len()) };
        return Ok(());
    }
    let chunk = STAGING_CHUNK.min(out.len());
    let staging = dev.alloc_buffer(chunk, DType::U8)?;
    let mut done = 0;
    while done < out.len() {
        let n = chunk.min(out.len() - done);
        shaders::run_ops_sync(
            dev,
            &[Op::Copy {
                src: src.buffer,
                src_off: (offset + done) as u64,
                dst: staging.buffer,
                dst_off: 0,
                size: n as u64,
            }],
        )?;
        out[done..done + n].copy_from_slice(&staging.as_bytes()[..n]);
        done += n;
    }
    Ok(())
}

struct Scratch<'a> {
    guard: MutexGuard<'a, Option<RawBuf>>,
}

impl Scratch<'_> {
    fn buffer(&self) -> vk::Buffer {
        self.guard.as_ref().map(|r| r.buffer).unwrap_or(vk::Buffer::null())
    }
}

fn scratch(ctx: &VulkanContext, bytes: usize) -> Result<Scratch<'_>> {
    let mut guard = ctx
        .gpu
        .scratch
        .lock()
        .map_err(|_| Error::Msg("vulkan: scratch state poisoned".into()))?;
    let need = bytes.max(1 << 20).next_power_of_two();
    let ok = guard.as_ref().map(|r| r.size >= bytes).unwrap_or(false);
    if !ok {
        if let Some(r) = guard.take() {
            ctx.gpu.free_raw(&ctx.device, r);
        }
        let (buffer, memory, _mapped, local, size) = alloc_raw(ctx, need, Use::Scratch)?;
        *guard = Some(RawBuf {
            buffer,
            memory,
            size,
            local,
        });
    }
    Ok(Scratch { guard })
}

fn push(vals: &[usize]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        v.extend_from_slice(&(x as u32).to_le_bytes());
    }
    v
}

fn storage_bind(s: &VulkanStorage) -> Bind {
    Bind {
        buffer: s.buffer,
        offset: 0,
        range: (s.capacity_bytes.max(16).div_ceil(16) * 16) as u64,
    }
}

fn fits_u32(v: usize) -> bool {
    v <= u32::MAX as usize
}

fn submit(
    dev: &VulkanDevice,
    prefix: Vec<Op<'_>>,
    dispatches: Vec<Op<'_>>,
    uses_scratch: bool,
) -> Result<()> {
    let mut pending: Vec<Op<'_>> = prefix;
    let mut count = 0u32;
    let total_dispatches = dispatches.len();
    for (idx, d) in dispatches.into_iter().enumerate() {
        let is_last = idx + 1 == total_dispatches;
        if count == shaders::MAX_SETS {
            shaders::run_ops(dev, &pending)?;
            pending = Vec::new();
            count = 0;
        }
        pending.push(d);
        count += 1;
        if is_last && !pending.is_empty() {
            if uses_scratch {
                shaders::run_ops_sync(dev, &pending)?;
            } else {
                shaders::run_ops(dev, &pending)?;
            }
            pending = Vec::new();
            count = 0;
        }
    }
    if !pending.is_empty() {
        if uses_scratch {
            shaders::run_ops_sync(dev, &pending)?;
        } else {
            shaders::run_ops(dev, &pending)?;
        }
    }
    Ok(())
}

fn unpack_p16_to_f32_src(is_bf16: bool) -> String {
    let ld = if is_bf16 { "ld_bf16" } else { "ld_f16" };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Sb {{ uint src[]; }};
layout(set = 0, binding = 1) buffer Db {{ float dst[]; }};
layout(push_constant, std430) uniform PC {{ uint n; uint src_off; }} pc;
{GLSL_PACK16_HELPERS}
void main() {{
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < pc.n; i += stride) {{
        uint si = pc.src_off + i;
        dst[i] = {ld}(src[si >> 1u], si);
    }}
}}
"#
    )
}

fn pack_f32_to_p16_src(is_bf16: bool) -> String {
    let pack = if is_bf16 {
        "pack_bf16x2(a, b)"
    } else {
        "packHalf2x16(vec2(a, b))"
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Sb {{ float src[]; }};
layout(set = 0, binding = 1) buffer Db {{ uint dst[]; }};
layout(push_constant, std430) uniform PC {{ uint n; }} pc;
{GLSL_PACK16_HELPERS}
void main() {{
    uint pairs = (pc.n + 1u) >> 1u;
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint p = gl_GlobalInvocationID.x; p < pairs; p += stride) {{
        uint i0 = 2u * p;
        float a = src[i0];
        float b = (i0 + 1u < pc.n) ? src[i0 + 1u] : 0.0;
        dst[p] = {pack};
    }}
}}
"#
    )
}

pub(crate) fn f16_bf16_to_f32_gpu(
    dev: &VulkanDevice,
    src: &VulkanStorage,
    off: usize,
    n: usize,
) -> Result<VulkanStorage> {
    let dst = dev.alloc_buffer(n, DType::F32)?;
    if n == 0 {
        return Ok(dst);
    }
    let is_bf16 = src.dtype == DType::BF16;
    let key = if is_bf16 { "vk_unpack_bf16_f32" } else { "vk_unpack_f16_f32" };
    let kern = shaders::kernel(dev, key, 2, 8, || unpack_p16_to_f32_src(is_bf16))?;
    let g = n.div_ceil(256).clamp(1, 65535) as u32;
    shaders::run(dev, &kern, &[src, &dst], &push(&[n, off]), [g, 1, 1])?;
    Ok(dst)
}

pub(crate) fn f32_to_f16_bf16_gpu(
    dev: &VulkanDevice,
    src: &VulkanStorage,
    dst_dtype: DType,
) -> Result<VulkanStorage> {
    let n = src.numel;
    let dst = dev.alloc_buffer(n, dst_dtype)?;
    if n == 0 {
        return Ok(dst);
    }
    let is_bf16 = dst_dtype == DType::BF16;
    let key = if is_bf16 { "vk_pack_f32_bf16" } else { "vk_pack_f32_f16" };
    let kern = shaders::kernel(dev, key, 2, 4, || pack_f32_to_p16_src(is_bf16))?;
    let g = n.div_ceil(2).div_ceil(256).clamp(1, 65535) as u32;
    shaders::run(dev, &kern, &[src, &dst], &push(&[n]), [g, 1, 1])?;
    Ok(dst)
}

const GLSL_Q8_GEMV: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) readonly buffer WQ { uint wq[]; };
layout(set = 0, binding = 2) readonly buffer WD { float wd[]; };
layout(set = 0, binding = 3) buffer Y { float y[]; };
layout(push_constant, std430) uniform PC {
    uint m; uint n; uint k; uint nb;
    uint x_off; uint ldx; uint y_off; uint ldy; uint c0;
} pc;
shared float red[256];
void main() {
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u;
    uint col = pc.c0 + gl_WorkGroupID.x * 8u + (t >> 5u);
    uint mi = gl_WorkGroupID.y;
    float acc = 0.0;
    if (col < pc.n) {
        uint xrow = pc.x_off + mi * pc.ldx;
        uint brow = col * pc.nb;
        for (uint b = lane; b < pc.nb; b += 32u) {
            uint qb = (brow + b) * 8u;
            uint xb = xrow + b * 32u;
            float sum = 0.0;
            for (uint j = 0u; j < 8u; j++) {
                uint w = wq[qb + j];
                uint xo = xb + j * 4u;
                sum += float(int(w << 24u) >> 24u) * x[xo]
                     + float(int(w << 16u) >> 24u) * x[xo + 1u]
                     + float(int(w << 8u) >> 24u) * x[xo + 2u]
                     + float(int(w) >> 24u) * x[xo + 3u];
            }
            acc += wd[brow + b] * sum;
        }
    }
    red[t] = acc;
    barrier();
    for (uint st = 16u; st > 0u; st >>= 1u) {
        if (lane < st) {
            red[t] = red[t] + red[t + st];
        }
        barrier();
    }
    if (lane == 0u && col < pc.n) {
        y[pc.y_off + mi * pc.ldy + col] = red[t];
    }
}
"#;

const GLSL_Q8_GEMV_V: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer X { vec4 x[]; };
layout(set = 0, binding = 1) readonly buffer WQ { uvec4 wq[]; };
layout(set = 0, binding = 2) readonly buffer WD { float wd[]; };
layout(set = 0, binding = 3) buffer Y { float y[]; };
layout(push_constant, std430) uniform PC {
    uint m; uint n; uint k; uint nb;
    uint x_off; uint ldx; uint y_off; uint ldy; uint c0;
} pc;
shared float red[256];
vec4 q4(uint w) {
    return vec4(float(int(w << 24u) >> 24u), float(int(w << 16u) >> 24u),
                float(int(w << 8u) >> 24u), float(int(w) >> 24u));
}
float dot16(uvec4 q, uint xv) {
    return dot(q4(q.x), x[xv]) + dot(q4(q.y), x[xv + 1u])
         + dot(q4(q.z), x[xv + 2u]) + dot(q4(q.w), x[xv + 3u]);
}
void main() {
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u;
    uint col = pc.c0 + gl_WorkGroupID.x * 8u + (t >> 5u);
    uint mi = gl_WorkGroupID.y;
    float acc = 0.0;
    if (col < pc.n) {
        uint xrow = (pc.x_off + mi * pc.ldx) >> 2u;
        uint brow = col * pc.nb;
        uint b = lane;
        while (b + 32u < pc.nb) {
            uvec4 a0 = wq[(brow + b) * 2u];
            uvec4 a1 = wq[(brow + b) * 2u + 1u];
            uvec4 c0 = wq[(brow + b + 32u) * 2u];
            uvec4 c1 = wq[(brow + b + 32u) * 2u + 1u];
            float d0 = wd[brow + b];
            float d1 = wd[brow + b + 32u];
            uint xa = xrow + b * 8u;
            uint xc = xrow + (b + 32u) * 8u;
            acc += d0 * (dot16(a0, xa) + dot16(a1, xa + 4u))
                 + d1 * (dot16(c0, xc) + dot16(c1, xc + 4u));
            b += 64u;
        }
        while (b < pc.nb) {
            uvec4 a0 = wq[(brow + b) * 2u];
            uvec4 a1 = wq[(brow + b) * 2u + 1u];
            uint xa = xrow + b * 8u;
            acc += wd[brow + b] * (dot16(a0, xa) + dot16(a1, xa + 4u));
            b += 32u;
        }
    }
    red[t] = acc;
    barrier();
    for (uint st = 16u; st > 0u; st >>= 1u) {
        if (lane < st) {
            red[t] = red[t] + red[t + st];
        }
        barrier();
    }
    if (lane == 0u && col < pc.n) {
        y[pc.y_off + mi * pc.ldy + col] = red[t];
    }
}
"#;

const GLSL_Q8_GEMM: &str = r#"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer X { float x[]; };
layout(set = 0, binding = 1) readonly buffer WQ { uint wq[]; };
layout(set = 0, binding = 2) readonly buffer WD { float wd[]; };
layout(set = 0, binding = 3) buffer Y { float y[]; };
layout(push_constant, std430) uniform PC {
    uint m; uint n; uint k; uint nb;
    uint x_off; uint ldx; uint y_off; uint ldy; uint c0;
} pc;
shared float As[2080];
shared float Bs[2080];
void store_row(uint gr, uint col, vec4 v) {
    if (gr >= pc.m) {
        return;
    }
    uint o = pc.y_off + gr * pc.ldy;
    if (col < pc.n) { y[o + col] = v.x; }
    if (col + 1u < pc.n) { y[o + col + 1u] = v.y; }
    if (col + 2u < pc.n) { y[o + col + 2u] = v.z; }
    if (col + 3u < pc.n) { y[o + col + 3u] = v.w; }
}
void main() {
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tid = ty * 16u + tx;
    uint row_base = gl_WorkGroupID.y * 64u;
    uint col_base = pc.c0 + gl_WorkGroupID.x * 64u;
    vec4 c0 = vec4(0.0);
    vec4 c1 = vec4(0.0);
    vec4 c2 = vec4(0.0);
    vec4 c3 = vec4(0.0);
    uint wr = tid >> 2u;
    uint part = tid & 3u;
    uint gc = col_base + wr;
    for (uint kb = 0u; kb < pc.nb; kb++) {
        for (uint l = 0u; l < 8u; l++) {
            uint e = tid + l * 256u;
            uint kk = e & 31u;
            uint rr = e >> 5u;
            uint gr = row_base + rr;
            float av = 0.0;
            if (gr < pc.m) {
                av = x[pc.x_off + gr * pc.ldx + kb * 32u + kk];
            }
            As[kk * 65u + rr] = av;
        }
        uint kq = part * 8u;
        if (gc < pc.n) {
            uint blk = gc * pc.nb + kb;
            float d = wd[blk];
            uint q0 = wq[blk * 8u + part * 2u];
            uint q1 = wq[blk * 8u + part * 2u + 1u];
            Bs[(kq + 0u) * 65u + wr] = d * float(int(q0 << 24u) >> 24u);
            Bs[(kq + 1u) * 65u + wr] = d * float(int(q0 << 16u) >> 24u);
            Bs[(kq + 2u) * 65u + wr] = d * float(int(q0 << 8u) >> 24u);
            Bs[(kq + 3u) * 65u + wr] = d * float(int(q0) >> 24u);
            Bs[(kq + 4u) * 65u + wr] = d * float(int(q1 << 24u) >> 24u);
            Bs[(kq + 5u) * 65u + wr] = d * float(int(q1 << 16u) >> 24u);
            Bs[(kq + 6u) * 65u + wr] = d * float(int(q1 << 8u) >> 24u);
            Bs[(kq + 7u) * 65u + wr] = d * float(int(q1) >> 24u);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                Bs[(kq + j) * 65u + wr] = 0.0;
            }
        }
        barrier();
        for (uint kk = 0u; kk < 32u; kk++) {
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
    store_row(gr0, col, c0);
    store_row(gr0 + 1u, col, c1);
    store_row(gr0 + 2u, col, c2);
    store_row(gr0 + 3u, col, c3);
}
"#;

const GLSL_Q8_GATHER: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IDS { uint ids[]; };
layout(set = 0, binding = 1) readonly buffer WQ { uint wq[]; };
layout(set = 0, binding = 2) readonly buffer WD { float wd[]; };
layout(set = 0, binding = 3) buffer Y { float y[]; };
layout(push_constant, std430) uniform PC {
    uint total; uint k; uint nb; uint rows; uint stride;
} pc;
void main() {
    for (uint idx = gl_GlobalInvocationID.x; idx < pc.total; idx += pc.stride) {
        uint i = idx / pc.k;
        uint c = idx - i * pc.k;
        uint id = ids[i];
        float v = 0.0;
        if (id < pc.rows) {
            uint blk = id * pc.nb + (c >> 5u);
            uint w = wq[blk * 8u + ((c & 31u) >> 2u)];
            uint sh = (c & 3u) * 8u;
            v = wd[blk] * float(int(w << (24u - sh)) >> 24u);
        }
        y[idx] = v;
    }
}
"#;

const GLSL_Q8_DEQUANT: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer WQ { uint wq[]; };
layout(set = 0, binding = 1) readonly buffer WD { float wd[]; };
layout(set = 0, binding = 2) buffer Y { uint y[]; };
layout(push_constant, std430) uniform PC {
    uint total; uint half_out; uint stride;
} pc;
float q8_at(uint e) {
    uint w = wq[e >> 2u];
    uint sh = (e & 3u) * 8u;
    return wd[e >> 5u] * float(int(w << (24u - sh)) >> 24u);
}
void main() {
    if (pc.half_out == 0u) {
        for (uint e = gl_GlobalInvocationID.x; e < pc.total; e += pc.stride) {
            y[e] = floatBitsToUint(q8_at(e));
        }
    } else {
        uint words = pc.total >> 1u;
        for (uint p = gl_GlobalInvocationID.x; p < words; p += pc.stride) {
            y[p] = packHalf2x16(vec2(q8_at(2u * p), q8_at(2u * p + 1u)));
        }
    }
}
"#;

const GEMV_MAX_M: usize = 8;
const MAX_DISPATCH_FLOPS: usize = 1 << 34;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantFmt {
    Q8_0,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
}

impl QuantFmt {
    pub(crate) fn from_ggml(dtype: GgmlDType) -> Option<Self> {
        match dtype {
            GgmlDType::Q8_0 => Some(Self::Q8_0),
            GgmlDType::Q4_0 => Some(Self::Q4_0),
            GgmlDType::Q4K => Some(Self::Q4K),
            GgmlDType::Q5K => Some(Self::Q5K),
            GgmlDType::Q6K => Some(Self::Q6K),
            _ => None,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
        }
    }

    fn block_elems(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    fn ggml_block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    fn qs_bytes_per_block(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_0 => 16,
            Self::Q4K => 128,
            Self::Q5K => 160,
            Self::Q6K => 192,
        }
    }

    fn aux_bytes_per_block(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q4_0 => 4,
            Self::Q4K | Self::Q5K => 16,
            Self::Q6K => 20,
        }
    }

    fn min_gpu_bytes(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q4_0 => 64 * self.ggml_block_bytes(),
            Self::Q4K | Self::Q5K | Self::Q6K => 8 * self.ggml_block_bytes(),
        }
    }
}

const QUANT_BYTE_HELPERS: &str = r#"
uint byte_wq(uint bi) { return (wq[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xffu; }
int sbyte_wq(uint bi) { return int(wq[bi >> 2u] << (24u - (bi & 3u) * 8u)) >> 24u; }
uint byte_wa(uint bi) { return (wa[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xffu; }
int sbyte_wa(uint bi) { return int(wa[bi >> 2u] << (24u - (bi & 3u) * 8u)) >> 24u; }
uvec2 k4_sc_min(uint blk, uint is) {
    uint sb = blk * 16u + 4u;
    if (is < 4u) {
        return uvec2(byte_wa(sb + is) & 63u, byte_wa(sb + is + 4u) & 63u);
    }
    uint b = byte_wa(sb + is + 4u);
    uint sc = (b & 15u) | ((byte_wa(sb + is - 4u) >> 6u) << 4u);
    uint mn = (b >> 4u) | ((byte_wa(sb + is) >> 6u) << 4u);
    return uvec2(sc, mn);
}
"#;

fn quant_elem_fn(fmt: QuantFmt) -> &'static str {
    match fmt {
        QuantFmt::Q8_0 => {
            r#"
float q_elem(uint blk, uint e) {
    return uintBitsToFloat(wa[blk]) * float(sbyte_wq(blk * 32u + e));
}
"#
        }
        QuantFmt::Q4_0 => {
            r#"
float q_elem(uint blk, uint e) {
    uint bv = byte_wq(blk * 16u + (e & 15u));
    int q = int((e < 16u) ? (bv & 15u) : (bv >> 4u)) - 8;
    return uintBitsToFloat(wa[blk]) * float(q);
}
"#
        }
        QuantFmt::Q4K => {
            r#"
float q_elem(uint blk, uint e) {
    vec2 dm = unpackHalf2x16(wa[blk * 4u]);
    uint chunk = e >> 6u;
    uint sub = (e >> 5u) & 1u;
    uint l = e & 31u;
    uvec2 sm = k4_sc_min(blk, chunk * 2u + sub);
    uint qb = byte_wq(blk * 128u + chunk * 32u + l);
    uint q4 = (sub == 0u) ? (qb & 15u) : (qb >> 4u);
    return (dm.x * float(sm.x)) * float(q4) - (dm.y * float(sm.y));
}
"#
        }
        QuantFmt::Q5K => {
            r#"
float q_elem(uint blk, uint e) {
    vec2 dm = unpackHalf2x16(wa[blk * 4u]);
    uint chunk = e >> 6u;
    uint sub = (e >> 5u) & 1u;
    uint l = e & 31u;
    uint is = chunk * 2u + sub;
    uvec2 sm = k4_sc_min(blk, is);
    uint qb = byte_wq(blk * 160u + chunk * 32u + l);
    uint hb = byte_wq(blk * 160u + 128u + l);
    uint q5 = ((sub == 0u) ? (qb & 15u) : (qb >> 4u)) | (((hb >> is) & 1u) << 4u);
    return (dm.x * float(sm.x)) * float(q5) - (dm.y * float(sm.y));
}
"#
        }
        QuantFmt::Q6K => {
            r#"
float q_elem(uint blk, uint e) {
    float d = unpackHalf2x16(wa[blk * 5u + 4u]).x;
    uint half_i = e >> 7u;
    uint quad = (e >> 5u) & 3u;
    uint l = e & 31u;
    uint is = half_i * 8u + quad * 2u + (l >> 4u);
    int sc = sbyte_wa(blk * 20u + is);
    uint qlb = byte_wq(blk * 192u + half_i * 64u + (quad & 1u) * 32u + l);
    uint lo4 = (quad < 2u) ? (qlb & 15u) : (qlb >> 4u);
    uint qhb = byte_wq(blk * 192u + 128u + half_i * 32u + l);
    uint hi2 = (qhb >> (quad * 2u)) & 3u;
    int q6 = int(lo4 | (hi2 << 4u)) - 32;
    return d * float(sc) * float(q6);
}
"#
        }
    }
}

fn quant_gemv_body(fmt: QuantFmt) -> &'static str {
    match fmt {
        QuantFmt::Q4_0 => {
            r#"
        for (uint b = lane; b < pc.nb; b += 32u) {
            uint blk = brow + b;
            uint xb = xrow + b * 32u;
            float sum = 0.0;
            for (uint j = 0u; j < 16u; j++) {
                uint bv = byte_wq(blk * 16u + j);
                sum += float(int(bv & 15u) - 8) * x[xb + j]
                     + float(int(bv >> 4u) - 8) * x[xb + j + 16u];
            }
            acc += uintBitsToFloat(wa[blk]) * sum;
        }
"#
        }
        QuantFmt::Q4K => {
            r#"
        for (uint sb = 0u; sb < pc.nb; sb++) {
            uint blk = brow + sb;
            vec2 dm = unpackHalf2x16(wa[blk * 4u]);
            uint xb = xrow + sb * 256u;
            for (uint c = 0u; c < 4u; c++) {
                uvec2 sm0 = k4_sc_min(blk, c * 2u);
                uvec2 sm1 = k4_sc_min(blk, c * 2u + 1u);
                uint qb = byte_wq(blk * 128u + c * 32u + lane);
                float w0 = (dm.x * float(sm0.x)) * float(qb & 15u) - (dm.y * float(sm0.y));
                float w1 = (dm.x * float(sm1.x)) * float(qb >> 4u) - (dm.y * float(sm1.y));
                acc += w0 * x[xb + c * 64u + lane] + w1 * x[xb + c * 64u + 32u + lane];
            }
        }
"#
        }
        QuantFmt::Q5K => {
            r#"
        for (uint sb = 0u; sb < pc.nb; sb++) {
            uint blk = brow + sb;
            vec2 dm = unpackHalf2x16(wa[blk * 4u]);
            uint hb = byte_wq(blk * 160u + 128u + lane);
            uint xb = xrow + sb * 256u;
            for (uint c = 0u; c < 4u; c++) {
                uint is0 = c * 2u;
                uint is1 = is0 + 1u;
                uvec2 sm0 = k4_sc_min(blk, is0);
                uvec2 sm1 = k4_sc_min(blk, is1);
                uint qb = byte_wq(blk * 160u + c * 32u + lane);
                uint q0 = (qb & 15u) | (((hb >> is0) & 1u) << 4u);
                uint q1 = (qb >> 4u) | (((hb >> is1) & 1u) << 4u);
                float w0 = (dm.x * float(sm0.x)) * float(q0) - (dm.y * float(sm0.y));
                float w1 = (dm.x * float(sm1.x)) * float(q1) - (dm.y * float(sm1.y));
                acc += w0 * x[xb + c * 64u + lane] + w1 * x[xb + c * 64u + 32u + lane];
            }
        }
"#
        }
        QuantFmt::Q6K => {
            r#"
        for (uint sb = 0u; sb < pc.nb; sb++) {
            uint blk = brow + sb;
            float d = unpackHalf2x16(wa[blk * 5u + 4u]).x;
            uint xb = xrow + sb * 256u;
            uint is_l = lane >> 4u;
            for (uint h = 0u; h < 2u; h++) {
                uint ql0 = byte_wq(blk * 192u + h * 64u + lane);
                uint ql1 = byte_wq(blk * 192u + h * 64u + 32u + lane);
                uint qhb = byte_wq(blk * 192u + 128u + h * 32u + lane);
                int s0 = sbyte_wa(blk * 20u + h * 8u + is_l);
                int s1 = sbyte_wa(blk * 20u + h * 8u + 2u + is_l);
                int s2 = sbyte_wa(blk * 20u + h * 8u + 4u + is_l);
                int s3 = sbyte_wa(blk * 20u + h * 8u + 6u + is_l);
                int q0 = int((ql0 & 15u) | ((qhb & 3u) << 4u)) - 32;
                int q1 = int((ql1 & 15u) | (((qhb >> 2u) & 3u) << 4u)) - 32;
                int q2 = int((ql0 >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32;
                int q3 = int((ql1 >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32;
                uint base = xb + h * 128u + lane;
                acc += d * (float(s0 * q0) * x[base]
                          + float(s1 * q1) * x[base + 32u]
                          + float(s2 * q2) * x[base + 64u]
                          + float(s3 * q3) * x[base + 96u]);
            }
        }
"#
        }
        QuantFmt::Q8_0 => {
            r#"
        for (uint b = lane; b < pc.nb; b += 32u) {
            uint blk = brow + b;
            uint xb = xrow + b * 32u;
            float sum = 0.0;
            for (uint j = 0u; j < 8u; j++) {
                uint w = wq[blk * 8u + j];
                uint xo = xb + j * 4u;
                sum += float(int(w << 24u) >> 24u) * x[xo]
                     + float(int(w << 16u) >> 24u) * x[xo + 1u]
                     + float(int(w << 8u) >> 24u) * x[xo + 2u]
                     + float(int(w) >> 24u) * x[xo + 3u];
            }
            acc += uintBitsToFloat(wa[blk]) * sum;
        }
"#
        }
    }
}

fn quant_gemv_src(fmt: QuantFmt) -> String {
    let body = quant_gemv_body(fmt);
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer X {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer WQ {{ uint wq[]; }};
layout(set = 0, binding = 2) readonly buffer WA {{ uint wa[]; }};
layout(set = 0, binding = 3) buffer Y {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint m; uint n; uint k; uint nb;
    uint x_off; uint ldx; uint y_off; uint ldy; uint c0;
}} pc;
{QUANT_BYTE_HELPERS}
shared float red[256];
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u;
    uint col = pc.c0 + gl_WorkGroupID.x * 8u + (t >> 5u);
    uint mi = gl_WorkGroupID.y;
    float acc = 0.0;
    if (col < pc.n) {{
        uint xrow = pc.x_off + mi * pc.ldx;
        uint brow = col * pc.nb;
        {body}
    }}
    red[t] = acc;
    barrier();
    for (uint st = 16u; st > 0u; st >>= 1u) {{
        if (lane < st) {{ red[t] = red[t] + red[t + st]; }}
        barrier();
    }}
    if (lane == 0u && col < pc.n) {{
        y[pc.y_off + mi * pc.ldy + col] = red[t];
    }}
}}
"#
    )
}

fn quant_gemm_src(fmt: QuantFmt) -> String {
    let elem = quant_elem_fn(fmt);
    let blk_shift = if fmt.block_elems() == 256 { 3u32 } else { 0u32 };
    let sub_mask = if fmt.block_elems() == 256 { 7u32 } else { 0u32 };
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer X {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer WQ {{ uint wq[]; }};
layout(set = 0, binding = 2) readonly buffer WA {{ uint wa[]; }};
layout(set = 0, binding = 3) buffer Y {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint m; uint n; uint k; uint nb;
    uint x_off; uint ldx; uint y_off; uint ldy; uint c0;
}} pc;
{QUANT_BYTE_HELPERS}
{elem}
shared float As[2080];
shared float Bs[2080];
void store_row(uint gr, uint col, vec4 v) {{
    if (gr >= pc.m) {{ return; }}
    uint o = pc.y_off + gr * pc.ldy;
    if (col < pc.n) {{ y[o + col] = v.x; }}
    if (col + 1u < pc.n) {{ y[o + col + 1u] = v.y; }}
    if (col + 2u < pc.n) {{ y[o + col + 2u] = v.z; }}
    if (col + 3u < pc.n) {{ y[o + col + 3u] = v.w; }}
}}
void main() {{
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tid = ty * 16u + tx;
    uint row_base = gl_WorkGroupID.y * 64u;
    uint col_base = pc.c0 + gl_WorkGroupID.x * 64u;
    vec4 c0 = vec4(0.0); vec4 c1 = vec4(0.0); vec4 c2 = vec4(0.0); vec4 c3 = vec4(0.0);
    uint wr = tid >> 2u;
    uint part = tid & 3u;
    uint gc = col_base + wr;
    uint nk32 = pc.k >> 5u;
    for (uint kb = 0u; kb < nk32; kb++) {{
        for (uint l = 0u; l < 8u; l++) {{
            uint e = tid + l * 256u;
            uint kk = e & 31u;
            uint rr = e >> 5u;
            uint gr = row_base + rr;
            float av = (gr < pc.m) ? x[pc.x_off + gr * pc.ldx + kb * 32u + kk] : 0.0;
            As[kk * 65u + rr] = av;
        }}
        uint kq = part * 8u;
        if (gc < pc.n) {{
            uint blk = gc * pc.nb + (kb >> {blk_shift}u);
            uint e0 = ((kb & {sub_mask}u) * 32u) + kq;
            for (uint j = 0u; j < 8u; j++) {{
                Bs[(kq + j) * 65u + wr] = q_elem(blk, e0 + j);
            }}
        }} else {{
            for (uint j = 0u; j < 8u; j++) {{
                Bs[(kq + j) * 65u + wr] = 0.0;
            }}
        }}
        barrier();
        for (uint kk = 0u; kk < 32u; kk++) {{
            uint ao = kk * 65u + ty * 4u;
            uint bo = kk * 65u + tx * 4u;
            vec4 bvec = vec4(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
            c0 += As[ao] * bvec;
            c1 += As[ao + 1u] * bvec;
            c2 += As[ao + 2u] * bvec;
            c3 += As[ao + 3u] * bvec;
        }}
        barrier();
    }}
    uint gr0 = row_base + ty * 4u;
    uint col = col_base + tx * 4u;
    store_row(gr0, col, c0);
    store_row(gr0 + 1u, col, c1);
    store_row(gr0 + 2u, col, c2);
    store_row(gr0 + 3u, col, c3);
}}
"#
    )
}

fn quant_dequant_src(fmt: QuantFmt) -> String {
    let elem = quant_elem_fn(fmt);
    let blk_shift = if fmt.block_elems() == 256 { 8u32 } else { 5u32 };
    let blk_mask = (fmt.block_elems() - 1) as u32;
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer WQ {{ uint wq[]; }};
layout(set = 0, binding = 1) readonly buffer WA {{ uint wa[]; }};
layout(set = 0, binding = 2) buffer Y {{ uint y[]; }};
layout(push_constant, std430) uniform PC {{ uint total; uint half_out; uint stride; }} pc;
{QUANT_BYTE_HELPERS}
{elem}
float at(uint e) {{ return q_elem(e >> {blk_shift}u, e & {blk_mask}u); }}
void main() {{
    if (pc.half_out == 0u) {{
        for (uint e = gl_GlobalInvocationID.x; e < pc.total; e += pc.stride) {{
            y[e] = floatBitsToUint(at(e));
        }}
    }} else {{
        uint words = pc.total >> 1u;
        for (uint p = gl_GlobalInvocationID.x; p < words; p += pc.stride) {{
            y[p] = packHalf2x16(vec2(at(2u * p), at(2u * p + 1u)));
        }}
    }}
}}
"#
    )
}

fn quant_gather_src(fmt: QuantFmt) -> String {
    let elem = quant_elem_fn(fmt);
    let blk_shift = if fmt.block_elems() == 256 { 8u32 } else { 5u32 };
    let blk_mask = (fmt.block_elems() - 1) as u32;
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IDS {{ uint ids[]; }};
layout(set = 0, binding = 1) readonly buffer WQ {{ uint wq[]; }};
layout(set = 0, binding = 2) readonly buffer WA {{ uint wa[]; }};
layout(set = 0, binding = 3) buffer Y {{ float y[]; }};
layout(push_constant, std430) uniform PC {{ uint total; uint k; uint nb; uint rows; uint stride; }} pc;
{QUANT_BYTE_HELPERS}
{elem}
void main() {{
    for (uint idx = gl_GlobalInvocationID.x; idx < pc.total; idx += pc.stride) {{
        uint i = idx / pc.k;
        uint c = idx - i * pc.k;
        uint id = ids[i];
        float v = 0.0;
        if (id < pc.rows) {{
            uint blk = id * pc.nb + (c >> {blk_shift}u);
            v = q_elem(blk, c & {blk_mask}u);
        }}
        y[idx] = v;
    }}
}}
"#
    )
}

fn quant_moe_src(fmt: QuantFmt) -> String {
    let body = quant_gemv_body(fmt);
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer X {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer IDS {{ uint ids[]; }};
layout(set = 0, binding = 2) readonly buffer WQ {{ uint wq[]; }};
layout(set = 0, binding = 3) readonly buffer WA {{ uint wa[]; }};
layout(set = 0, binding = 4) buffer Y {{ float y[]; }};
layout(push_constant, std430) uniform PC {{
    uint bsz; uint topk; uint x_stride_t; uint n_exp;
    uint n; uint k; uint nb; uint x_off; uint ids_off; uint c0;
}} pc;
{QUANT_BYTE_HELPERS}
shared float red[256];
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u;
    uint col = pc.c0 + gl_WorkGroupID.x * 8u + (t >> 5u);
    uint bt = gl_WorkGroupID.y;
    uint bi = bt / pc.topk;
    uint ti = bt - bi * pc.topk;
    float acc = 0.0;
    if (bi < pc.bsz && col < pc.n) {{
        uint eid = ids[pc.ids_off + bt];
        if (eid < pc.n_exp) {{
            uint xrow = pc.x_off + (bi * pc.x_stride_t + ((pc.x_stride_t > 1u) ? ti : 0u)) * pc.k;
            uint brow = (eid * pc.n + col) * pc.nb;
            {body}
        }}
    }}
    red[t] = acc;
    barrier();
    for (uint st = 16u; st > 0u; st >>= 1u) {{
        if (lane < st) {{ red[t] = red[t] + red[t + st]; }}
        barrier();
    }}
    if (lane == 0u && bi < pc.bsz && col < pc.n) {{
        y[bt * pc.n + col] = red[t];
    }}
}}
"#
    )
}

pub(crate) struct GpuQuant {
    pub(crate) buf: GpuBuf,
    pub(crate) fmt: QuantFmt,
    pub(crate) nblocks: usize,
    aux_off: usize,
}

impl std::fmt::Debug for GpuQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuQuant")
            .field("fmt", &self.fmt)
            .field("nblocks", &self.nblocks)
            .field("buf", &self.buf)
            .finish()
    }
}

impl GpuQuant {
    pub(crate) fn from_ggml(dev: &VulkanDevice, dtype: GgmlDType, bytes: &[u8]) -> Result<Self> {
        let Some(fmt) = QuantFmt::from_ggml(dtype) else {
            crate::bail!("vulkan quant: unsupported GPU format {dtype:?}")
        };
        let gb = fmt.ggml_block_bytes();
        if !bytes.len().is_multiple_of(gb) {
            crate::bail!("vulkan quant {dtype:?}: {} bytes not a multiple of {gb}", bytes.len())
        }
        let nb = bytes.len() / gb;
        let qs_pb = fmt.qs_bytes_per_block();
        let aux_pb = fmt.aux_bytes_per_block();
        let qs_bytes = nb * qs_pb;
        let aux_bytes = nb * aux_pb;
        let aux_off = qs_bytes.div_ceil(256) * 256;
        let total = aux_off + aux_bytes;
        let max_range = dev.limits().max_storage_buffer_range as usize;
        if qs_bytes > max_range || aux_bytes > max_range || !fits_u32(nb * fmt.block_elems()) {
            crate::bail!("vulkan quant {dtype:?}: tensor exceeds storage buffer range")
        }
        let buf = GpuBuf::new(dev, total, Use::Weights)?;
        // Stream-repack in chunks bounded by STAGING_CHUNK to avoid 2x host RAM spikes.
        let chunk_blocks = (STAGING_CHUNK / (qs_pb + aux_pb).max(1)).max(1).min(nb.max(1));
        let mut qs_chunk = vec![0u8; chunk_blocks * qs_pb];
        let mut aux_chunk = vec![0u8; chunk_blocks * aux_pb];
        let mut b0 = 0;
        while b0 < nb {
            let cnt = chunk_blocks.min(nb - b0);
            for i in 0..cnt {
                let blk = &bytes[(b0 + i) * gb..(b0 + i + 1) * gb];
                let qdst = &mut qs_chunk[i * qs_pb..(i + 1) * qs_pb];
                let adst = &mut aux_chunk[i * aux_pb..(i + 1) * aux_pb];
                match fmt {
                    QuantFmt::Q8_0 => {
                        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                        qdst.copy_from_slice(&blk[2..34]);
                        adst.copy_from_slice(&d.to_le_bytes());
                    }
                    QuantFmt::Q4_0 => {
                        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                        qdst.copy_from_slice(&blk[2..18]);
                        adst.copy_from_slice(&d.to_le_bytes());
                    }
                    QuantFmt::Q4K => {
                        // GGML: [0..2] d, [2..4] dmin, [4..16] scales, [16..144] qs
                        qdst.copy_from_slice(&blk[16..144]);
                        adst[0..16].copy_from_slice(&blk[0..16]);
                    }
                    QuantFmt::Q5K => {
                        // GGML: [0..2] d, [2..4] dmin, [4..16] scales, [16..48] qh, [48..176] qs
                        qdst[0..128].copy_from_slice(&blk[48..176]);
                        qdst[128..160].copy_from_slice(&blk[16..48]);
                        adst[0..16].copy_from_slice(&blk[0..16]);
                    }
                    QuantFmt::Q6K => {
                        // GGML: [0..128] ql, [128..192] qh, [192..208] scales, [208..210] d
                        qdst[0..192].copy_from_slice(&blk[0..192]);
                        adst[0..16].copy_from_slice(&blk[192..208]);
                        adst[16..18].copy_from_slice(&blk[208..210]);
                        adst[18..20].fill(0);
                    }
                }
            }
            write(dev, &buf, b0 * qs_pb, &qs_chunk[..cnt * qs_pb])?;
            write(dev, &buf, aux_off + b0 * aux_pb, &aux_chunk[..cnt * aux_pb])?;
            b0 += cnt;
        }
        Ok(Self {
            buf,
            fmt,
            nblocks: nb,
            aux_off,
        })
    }

    pub(crate) fn to_ggml(&self) -> Result<Vec<u8>> {
        let nb = self.nblocks;
        let gb = self.fmt.ggml_block_bytes();
        let qs_pb = self.fmt.qs_bytes_per_block();
        let aux_pb = self.fmt.aux_bytes_per_block();
        let mut qs_all = vec![0u8; nb * qs_pb];
        let mut aux_all = vec![0u8; nb * aux_pb];
        read(&self.buf.dev, &self.buf, 0, &mut qs_all)?;
        read(&self.buf.dev, &self.buf, self.aux_off, &mut aux_all)?;
        let mut out = vec![0u8; nb * gb];
        for i in 0..nb {
            let o = &mut out[i * gb..(i + 1) * gb];
            let qsrc = &qs_all[i * qs_pb..(i + 1) * qs_pb];
            let asrc = &aux_all[i * aux_pb..(i + 1) * aux_pb];
            match self.fmt {
                QuantFmt::Q8_0 => {
                    let d = f32::from_le_bytes([asrc[0], asrc[1], asrc[2], asrc[3]]);
                    o[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                    o[2..34].copy_from_slice(qsrc);
                }
                QuantFmt::Q4_0 => {
                    let d = f32::from_le_bytes([asrc[0], asrc[1], asrc[2], asrc[3]]);
                    o[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                    o[2..18].copy_from_slice(qsrc);
                }
                QuantFmt::Q4K => {
                    o[0..16].copy_from_slice(&asrc[0..16]);
                    o[16..144].copy_from_slice(qsrc);
                }
                QuantFmt::Q5K => {
                    o[0..16].copy_from_slice(&asrc[0..16]);
                    o[16..48].copy_from_slice(&qsrc[128..160]);
                    o[48..176].copy_from_slice(&qsrc[0..128]);
                }
                QuantFmt::Q6K => {
                    o[0..192].copy_from_slice(&qsrc[0..192]);
                    o[192..208].copy_from_slice(&asrc[0..16]);
                    o[208..210].copy_from_slice(&asrc[16..18]);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn ggml_bytes(&self) -> usize {
        self.nblocks * self.fmt.ggml_block_bytes()
    }

    fn qs_bind(&self) -> Bind {
        Bind {
            buffer: self.buf.buffer,
            offset: 0,
            range: (self.nblocks * self.fmt.qs_bytes_per_block()).max(4) as u64,
        }
    }

    fn aux_bind(&self) -> Bind {
        Bind {
            buffer: self.buf.buffer,
            offset: self.aux_off as u64,
            range: (self.nblocks * self.fmt.aux_bytes_per_block()).max(4) as u64,
        }
    }

    fn prep_f32_input<'a>(
        dev: &VulkanDevice,
        x: &'a VulkanStorage,
        layout: &Layout,
        converted: &'a mut Option<VulkanStorage>,
    ) -> Result<(&'a VulkanStorage, usize)> {
        match (x.dtype, layout.contiguous_offsets()) {
            (DType::F32, Some((a, _))) => Ok((x, a)),
            (DType::F16 | DType::BF16, Some((a, b))) => {
                *converted = Some(f16_bf16_to_f32_gpu(dev, x, a, b - a)?);
                Ok((converted.as_ref().unwrap(), 0))
            }
            _ => {
                let view = x.host_view();
                let cpu = view.to_dtype(layout, DType::F32)?;
                *converted = Some(dev.storage_from_cpu_storage(&cpu)?);
                Ok((converted.as_ref().unwrap(), 0))
            }
        }
    }

    pub(crate) fn matmul(
        &self,
        (m, k, n): (usize, usize, usize),
        x: &VulkanStorage,
        layout: &Layout,
    ) -> Result<VulkanStorage> {
        let dev = x.device.clone();
        let be = self.fmt.block_elems();
        if !k.is_multiple_of(be) || self.nblocks * be != n * k {
            crate::bail!("vulkan {:?} matmul: weights do not hold {n}x{k} values", self.fmt)
        }
        let mut converted = None;
        let (xs, x_off) = Self::prep_f32_input(&dev, x, layout, &mut converted)?;
        let out = dev.alloc_buffer(m * n, DType::F32)?;
        if m == 0 || n == 0 {
            return Ok(out);
        }
        if !fits_u32(m * n) || !fits_u32(x_off + m * k) {
            crate::bail!("vulkan quant matmul: problem too large for 32 bit indexing")
        }
        let nb = k / be;
        let gemv = m <= GEMV_MAX_M;
        let x_aligned = !unified(dev.ctx()) || x_off % 4 == 0;
        let tag = self.fmt.tag();
        let kern = match (self.fmt, gemv, x_aligned) {
            (QuantFmt::Q8_0, true, true) => {
                shaders::kernel(&dev, "q8_gemv_v", 4, 36, || GLSL_Q8_GEMV_V.to_string())?
            }
            (QuantFmt::Q8_0, true, false) => {
                shaders::kernel(&dev, "q8_gemv", 4, 36, || GLSL_Q8_GEMV.to_string())?
            }
            (QuantFmt::Q8_0, false, _) => {
                shaders::kernel(&dev, "q8_gemm", 4, 36, || GLSL_Q8_GEMM.to_string())?
            }
            (fmt, true, _) => {
                shaders::kernel(&dev, &format!("{tag}_gemv"), 4, 36, || quant_gemv_src(fmt))?
            }
            (fmt, false, _) => {
                shaders::kernel(&dev, &format!("{tag}_gemm"), 4, 36, || quant_gemm_src(fmt))?
            }
        };
        let max_groups = dev.limits().max_workgroup_count;
        let ctx = dev.ctx();
        let row_bytes = k * 4;
        let band = if unified(ctx) || m * row_bytes <= scratch_cap() {
            m
        } else {
            ((scratch_cap() / row_bytes) / 64 * 64).max(64).min(m)
        };
        let cols_per = if gemv {
            (max_groups[0] as usize).min(65535) * 8
        } else {
            let by_flops = (MAX_DISPATCH_FLOPS / (2 * band.min(m) * k).max(1)) / 64 * 64;
            by_flops.max(64).min((max_groups[0] as usize).min(65535) * 64)
        };
        let scratch_guard = if unified(ctx) {
            None
        } else {
            Some(scratch(ctx, band * row_bytes)?)
        };
        let uses_scratch = scratch_guard.is_some();
        let mut r0 = 0;
        while r0 < m {
            let rows = band.min(m - r0);
            let (x_bind, xo, prefix) = match &scratch_guard {
                None => (storage_bind(xs), x_off + r0 * k, Vec::new()),
                Some(s) => {
                    let sb = s.buffer();
                    let prefix = vec![
                        Op::Copy {
                            src: xs.buffer,
                            src_off: ((x_off + r0 * k) * 4) as u64,
                            dst: sb,
                            dst_off: 0,
                            size: (rows * row_bytes) as u64,
                        },
                        Op::Barrier,
                    ];
                    (
                        Bind {
                            buffer: sb,
                            offset: 0,
                            range: (rows * row_bytes) as u64,
                        },
                        0,
                        prefix,
                    )
                }
            };
            let mut dispatches = Vec::new();
            let mut c0 = 0;
            while c0 < n {
                let cols = cols_per.min(n - c0);
                let groups = if gemv {
                    [cols.div_ceil(8) as u32, rows as u32, 1]
                } else {
                    [cols.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1]
                };
                let p = push(&[rows, (c0 + cols), k, nb, xo, k, r0 * n, n, c0]);
                dispatches.push(Op::Dispatch {
                    kernel: kern.as_ref(),
                    binds: vec![x_bind, self.qs_bind(), self.aux_bind(), storage_bind(&out)],
                    push: p,
                    groups,
                });
                c0 += cols;
            }
            submit(&dev, prefix, dispatches, uses_scratch)?;
            r0 += rows;
        }
        drop(scratch_guard);
        Ok(out)
    }

    pub(crate) fn indexed_moe_forward(
        &self,
        (n_exp, n, k): (usize, usize, usize),
        x: &VulkanStorage,
        x_l: &Layout,
        ids: &VulkanStorage,
        ids_l: &Layout,
    ) -> Result<VulkanStorage> {
        let dev = x.device.clone();
        let be = self.fmt.block_elems();
        if !k.is_multiple_of(be) || self.nblocks * be != n_exp * n * k {
            crate::bail!("vulkan {:?} moe: weight size mismatch", self.fmt)
        }
        let (bsz, topk) = ids_l.shape().dims2()?;
        let x_dims = x_l.dims();
        let x_stride_t = if x_dims.len() == 3 { x_dims[1] } else { 1 };
        let mut converted = None;
        let (xs, x_off) = Self::prep_f32_input(&dev, x, x_l, &mut converted)?;
        let ids_owned;
        let (ids_s, ids_off) = match (ids.dtype, ids_l.contiguous_offsets()) {
            (DType::U32, Some((a, _))) => (ids, a),
            _ => {
                let v = ids.host_view().to_dtype(ids_l, DType::U32)?;
                ids_owned = dev.storage_from_cpu_storage(&v)?;
                (&ids_owned, 0)
            }
        };
        let out = dev.alloc_buffer(bsz * topk * n, DType::F32)?;
        if bsz == 0 || topk == 0 || n == 0 {
            return Ok(out);
        }
        let bt = bsz * topk;
        if !fits_u32(bt * n) || !fits_u32(x_off + bsz * x_stride_t * k) || bt > 65535 {
            crate::bail!("vulkan quant moe: dimensions exceed kernel limits")
        }
        let nb = k / be;
        let tag = self.fmt.tag();
        let kern = shaders::kernel(&dev, &format!("{tag}_moe"), 5, 40, || {
            quant_moe_src(self.fmt)
        })?;
        let cols_per = (dev.limits().max_workgroup_count[0] as usize).min(65535) * 8;
        let mut dispatches = Vec::new();
        let mut c0 = 0;
        while c0 < n {
            let cols = cols_per.min(n - c0);
            let p = push(&[bsz, topk, x_stride_t, n_exp, c0 + cols, k, nb, x_off, ids_off, c0]);
            dispatches.push(Op::Dispatch {
                kernel: kern.as_ref(),
                binds: vec![
                    storage_bind(xs),
                    storage_bind(ids_s),
                    self.qs_bind(),
                    self.aux_bind(),
                    storage_bind(&out),
                ],
                push: p,
                groups: [cols.div_ceil(8) as u32, bt as u32, 1],
            });
            c0 += cols;
        }
        submit(&dev, Vec::new(), dispatches, false)?;
        Ok(out)
    }

    pub(crate) fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &[u32],
        dev: &VulkanDevice,
    ) -> Result<VulkanStorage> {
        let be = self.fmt.block_elems();
        if !hidden.is_multiple_of(be) || self.nblocks * be != rows * hidden {
            crate::bail!("vulkan {:?} embedding: table does not hold {rows}x{hidden}", self.fmt)
        }
        let total = ids.len() * hidden;
        let out = dev.alloc_buffer(total, DType::F32)?;
        if total == 0 {
            return Ok(out);
        }
        if !fits_u32(total) {
            crate::bail!("vulkan quant embedding: output too large")
        }
        let ids_s = dev.upload_u32_slice(ids)?;
        let tag = self.fmt.tag();
        let kern = if self.fmt == QuantFmt::Q8_0 {
            shaders::kernel(dev, "q8_gather", 4, 20, || GLSL_Q8_GATHER.to_string())?
        } else {
            shaders::kernel(dev, &format!("{tag}_gather"), 4, 20, || {
                quant_gather_src(self.fmt)
            })?
        };
        let groups = total.div_ceil(256).min(65535);
        let p = push(&[total, hidden, hidden / be, rows, groups * 256]);
        shaders::run_ops(
            dev,
            &[Op::Dispatch {
                kernel: kern.as_ref(),
                binds: vec![
                    storage_bind(&ids_s),
                    self.qs_bind(),
                    self.aux_bind(),
                    storage_bind(&out),
                ],
                push: p,
                groups: [groups as u32, 1, 1],
            }],
        )?;
        Ok(out)
    }

    pub(crate) fn dequantize(
        &self,
        elem_count: usize,
        dtype: DType,
        dev: &VulkanDevice,
    ) -> Result<VulkanStorage> {
        let elem_count = elem_count.min(self.nblocks * self.fmt.block_elems());
        let half = match dtype {
            DType::F32 => false,
            DType::F16 => true,
            dt => crate::bail!("vulkan quant dequantize to {dt:?} is not supported"),
        };
        if half && elem_count % 2 != 0 {
            crate::bail!("vulkan quant dequantize: odd element count for f16 output")
        }
        let out = dev.alloc_buffer(elem_count, dtype)?;
        if elem_count == 0 {
            return Ok(out);
        }
        if !fits_u32(elem_count) {
            crate::bail!("vulkan quant dequantize: tensor too large")
        }
        let tag = self.fmt.tag();
        let kern = if self.fmt == QuantFmt::Q8_0 {
            shaders::kernel(dev, "q8_dequant", 3, 12, || GLSL_Q8_DEQUANT.to_string())?
        } else {
            shaders::kernel(dev, &format!("{tag}_dequant"), 3, 12, || {
                quant_dequant_src(self.fmt)
            })?
        };
        let work = if half { elem_count / 2 } else { elem_count };
        let groups = work.div_ceil(256).min(65535);
        let p = push(&[elem_count, half as usize, groups * 256]);
        shaders::run_ops(
            dev,
            &[Op::Dispatch {
                kernel: kern.as_ref(),
                binds: vec![self.qs_bind(), self.aux_bind(), storage_bind(&out)],
                push: p,
                groups: [groups as u32, 1, 1],
            }],
        )?;
        Ok(out)
    }
}

pub(crate) fn want_quant_on_gpu(dev: &VulkanDevice, dtype: GgmlDType, ggml_bytes: usize) -> bool {
    if !enabled(dev) {
        return false;
    }
    let Some(fmt) = QuantFmt::from_ggml(dtype) else {
        return false;
    };
    if ggml_bytes < fmt.min_gpu_bytes() {
        return false;
    }
    let nb = ggml_bytes / fmt.ggml_block_bytes();
    let vram_need = (nb * fmt.qs_bytes_per_block()).div_ceil(256) * 256
        + nb * fmt.aux_bytes_per_block();
    let ctx = dev.ctx();
    let budgets = heap_budgets(ctx);
    working_heaps(ctx)
        .into_iter()
        .any(|h| heap_fits(ctx, h, vram_need as u64, &budgets))
}

fn rhs_variant(dtype: DType) -> Option<(&'static str, &'static str, &'static str)> {
    match dtype {
        DType::F32 => Some(("f32", "float w[];", "return w[i];")),
        DType::F16 => Some((
            "f16",
            "uint w[];",
            "uint v = w[i >> 1u]; vec2 h = unpackHalf2x16(v); return ((i & 1u) == 0u) ? h.x : h.y;",
        )),
        DType::BF16 => Some((
            "bf16",
            "uint w[];",
            "uint v = w[i >> 1u]; uint b = ((i & 1u) == 0u) ? (v << 16u) : (v & 4294901760u); return uintBitsToFloat(b);",
        )),
        _ => None,
    }
}

const DENSE_PC: &str = r#"
layout(push_constant, std430) uniform PC {
    uint m; uint n; uint k; uint x_off; uint ldx; uint y_off; uint ldy;
    uint w_off; uint sbk; uint sbn; uint c0;
} pc;
"#;

fn dense_gemv_src(decl: &str, load: &str) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer X {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer W {{ {decl} }};
layout(set = 0, binding = 2) buffer Y {{ float y[]; }};
{DENSE_PC}
shared float red[256];
float load_w(uint i) {{ {load} }}
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u;
    uint col = pc.c0 + gl_WorkGroupID.x * 8u + (t >> 5u);
    uint mi = gl_WorkGroupID.y;
    float acc = 0.0;
    if (col < pc.n) {{
        uint xb = pc.x_off + mi * pc.ldx;
        uint wb = pc.w_off + col * pc.sbn;
        for (uint kk = lane; kk < pc.k; kk += 32u) {{
            acc += x[xb + kk] * load_w(wb + kk * pc.sbk);
        }}
    }}
    red[t] = acc;
    barrier();
    for (uint st = 16u; st > 0u; st >>= 1u) {{
        if (lane < st) {{
            red[t] = red[t] + red[t + st];
        }}
        barrier();
    }}
    if (lane == 0u && col < pc.n) {{
        y[pc.y_off + mi * pc.ldy + col] = red[t];
    }}
}}
"#
    )
}

fn dense_vec_variant(dtype: DType) -> Option<(&'static str, usize, &'static str)> {
    match dtype {
        DType::F32 => Some((
            "vec4 w[];",
            4,
            "float dotv(vec4 q, uint xo) { return dot(q, x[xo >> 2u]); }",
        )),
        DType::F16 => Some((
            "uvec4 w[];",
            8,
            "float dotv(uvec4 q, uint xo) { vec4 x0 = x[xo >> 2u]; vec4 x1 = x[(xo >> 2u) + 1u]; return dot(vec4(unpackHalf2x16(q.x), unpackHalf2x16(q.y)), x0) + dot(vec4(unpackHalf2x16(q.z), unpackHalf2x16(q.w)), x1); }",
        )),
        DType::BF16 => Some((
            "uvec4 w[];",
            8,
            "float dotv(uvec4 q, uint xo) { vec4 x0 = x[xo >> 2u]; vec4 x1 = x[(xo >> 2u) + 1u]; vec4 a = vec4(uintBitsToFloat(q.x << 16u), uintBitsToFloat(q.x & 4294901760u), uintBitsToFloat(q.y << 16u), uintBitsToFloat(q.y & 4294901760u)); vec4 b = vec4(uintBitsToFloat(q.z << 16u), uintBitsToFloat(q.z & 4294901760u), uintBitsToFloat(q.w << 16u), uintBitsToFloat(q.w & 4294901760u)); return dot(a, x0) + dot(b, x1); }",
        )),
        _ => None,
    }
}

fn dense_gemv_vec_src(decl: &str, width: usize, dotv: &str) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer X {{ vec4 x[]; }};
layout(set = 0, binding = 1) readonly buffer W {{ {decl} }};
layout(set = 0, binding = 2) buffer Y {{ float y[]; }};
{DENSE_PC}
shared float red[256];
{dotv}
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint lane = t & 31u;
    uint col = pc.c0 + gl_WorkGroupID.x * 8u + (t >> 5u);
    uint mi = gl_WorkGroupID.y;
    float acc = 0.0;
    if (col < pc.n) {{
        uint xb = pc.x_off + mi * pc.ldx;
        uint base = (pc.w_off + col * pc.sbn) / {width}u;
        uint nv = pc.k / {width}u;
        uint v = lane;
        while (v + 96u < nv) {{
            {decl_q} q0 = w[base + v];
            {decl_q} q1 = w[base + v + 32u];
            {decl_q} q2 = w[base + v + 64u];
            {decl_q} q3 = w[base + v + 96u];
            acc += dotv(q0, xb + v * {width}u)
                 + dotv(q1, xb + (v + 32u) * {width}u)
                 + dotv(q2, xb + (v + 64u) * {width}u)
                 + dotv(q3, xb + (v + 96u) * {width}u);
            v += 128u;
        }}
        while (v < nv) {{
            acc += dotv(w[base + v], xb + v * {width}u);
            v += 32u;
        }}
    }}
    red[t] = acc;
    barrier();
    for (uint st = 16u; st > 0u; st >>= 1u) {{
        if (lane < st) {{
            red[t] = red[t] + red[t + st];
        }}
        barrier();
    }}
    if (lane == 0u && col < pc.n) {{
        y[pc.y_off + mi * pc.ldy + col] = red[t];
    }}
}}
"#,
        decl_q = if decl.starts_with("vec4") { "vec4" } else { "uvec4" }
    )
}

fn dense_gemm_src(decl: &str, load: &str) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer X {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer W {{ {decl} }};
layout(set = 0, binding = 2) buffer Y {{ float y[]; }};
{DENSE_PC}
shared float As[1040];
shared float Bs[1040];
float load_w(uint i) {{ {load} }}
void store_row(uint gr, uint col, vec4 v) {{
    if (gr >= pc.m) {{
        return;
    }}
    uint o = pc.y_off + gr * pc.ldy;
    if (col < pc.n) {{ y[o + col] = v.x; }}
    if (col + 1u < pc.n) {{ y[o + col + 1u] = v.y; }}
    if (col + 2u < pc.n) {{ y[o + col + 2u] = v.z; }}
    if (col + 3u < pc.n) {{ y[o + col + 3u] = v.w; }}
}}
void main() {{
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tid = ty * 16u + tx;
    uint row_base = gl_WorkGroupID.y * 64u;
    uint col_base = pc.c0 + gl_WorkGroupID.x * 64u;
    vec4 c0 = vec4(0.0);
    vec4 c1 = vec4(0.0);
    vec4 c2 = vec4(0.0);
    vec4 c3 = vec4(0.0);
    for (uint k0 = 0u; k0 < pc.k; k0 += 16u) {{
        for (uint l = 0u; l < 4u; l++) {{
            uint e = tid + l * 256u;
            uint kk = e % 16u;
            uint rr = e / 16u;
            uint gr = row_base + rr;
            uint gk = k0 + kk;
            float av = 0.0;
            if (gr < pc.m && gk < pc.k) {{
                av = x[pc.x_off + gr * pc.ldx + gk];
            }}
            As[kk * 65u + rr] = av;
            uint cc = e % 64u;
            uint kb = e / 64u;
            uint gc = col_base + cc;
            uint gkb = k0 + kb;
            float bv = 0.0;
            if (gc < pc.n && gkb < pc.k) {{
                bv = load_w(pc.w_off + gkb * pc.sbk + gc * pc.sbn);
            }}
            Bs[kb * 65u + cc] = bv;
        }}
        barrier();
        for (uint kk = 0u; kk < 16u; kk++) {{
            uint ao = kk * 65u + ty * 4u;
            uint bo = kk * 65u + tx * 4u;
            vec4 bvec = vec4(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
            c0 += As[ao] * bvec;
            c1 += As[ao + 1u] * bvec;
            c2 += As[ao + 2u] * bvec;
            c3 += As[ao + 3u] * bvec;
        }}
        barrier();
    }}
    uint gr0 = row_base + ty * 4u;
    uint col = col_base + tx * 4u;
    store_row(gr0, col, c0);
    store_row(gr0 + 1u, col, c1);
    store_row(gr0 + 2u, col, c2);
    store_row(gr0 + 3u, col, c3);
}}
"#
    )
}

const MIRROR_MIN_BYTES: usize = 4 << 20;

fn mirror_after() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| env_mb("CANDLE_VULKAN_MIRROR_AFTER", 2).max(1) as u32)
}

/// VRAM copy of a weight. `force` mirrors on first use (fixed [`shaders::ExecMode::Gpu`]),
/// otherwise only from the `CANDLE_VULKAN_MIRROR_AFTER`-th use.
fn mirror_of(rhs: &VulkanStorage, force: bool) -> Option<Arc<GpuBuf>> {
    let mut slot = rhs.mirror.lock().ok()?;
    if let Some(m) = slot.as_ref() {
        return Some(m.clone());
    }
    let uses = rhs.rhs_uses.fetch_add(1, Ordering::Relaxed) + 1;
    if !force && uses < mirror_after() {
        return None;
    }
    let bytes = rhs.as_bytes();
    let size = bytes.len().max(4).div_ceil(4) * 4;
    let buf = match GpuBuf::new(&rhs.device, size, Use::Weights) {
        Ok(b) => b,
        Err(_) => {
            rhs.rhs_uses.store(0, Ordering::Relaxed);
            return None;
        }
    };
    if write(&rhs.device, &buf, 0, bytes).is_err() {
        return None;
    }
    let buf = Arc::new(buf);
    *slot = Some(buf.clone());
    Some(buf)
}

/// Fixed GPU mode on host-shared memory: weight matmuls with at most this many rows
/// (`CANDLE_VULKAN_HOST_SMALL_M`, 0 = off) run on the host instead of one GPU round trip each,
/// for the weight shapes whose host results are bit-identical to the GPU kernel's
/// ([`host_route_ok`]), so the rows still do not depend on the batch they are in.
fn host_small_m() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CANDLE_VULKAN_HOST_SMALL_M")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(32)
    })
}

thread_local! {
    static HOST_ROUTE_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the host matmul of the fixed modes (`host_matmul_row_invariant`) gives the bits of
/// the GPU GEMM kernel for this F32 weight: both run one k-ordered multiply-add chain per output
/// element (measured on Apple Silicon: CPU, MoltenVK and Metal give the same digest). Checked
/// once per (device, K, N, weight strides) on 16 rows of the real weight; a difference keeps
/// that shape on the GPU.
fn host_route_ok(
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    (n, k): (usize, usize),
    (sbk, sbn, w_off): (usize, usize, usize),
) -> bool {
    type Key = (usize, usize, usize, usize, usize);
    static CACHE: OnceLock<Mutex<std::collections::HashMap<Key, bool>>> = OnceLock::new();
    let dev = &lhs.device;
    let key = (dev.ctx() as *const VulkanContext as usize, k, n, sbk, sbn);
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(ok) = cache.lock().ok().and_then(|g| g.get(&key).copied()) {
        return ok;
    }
    let rows = 16usize;
    let check = || -> Result<bool> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let data: Vec<f32> = (0..rows * k)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 * 4.0) - 2.0
            })
            .collect();
        let cpu = crate::CpuStorage::F32(data);
        let lhs_g = dev.storage_from_cpu_storage(&cpu)?;
        let lhs_t = Layout::contiguous((1, rows, k));
        let rhs_t = Layout::new(crate::Shape::from((1, k, n)), vec![0, sbk, sbn], w_off);
        HOST_ROUTE_CHECK.with(|f| f.set(true));
        let gpu = dense_matmul(&lhs_g, rhs, (1, rows, n, k), &lhs_t, &rhs_t);
        HOST_ROUTE_CHECK.with(|f| f.set(false));
        let Some(gpu) = gpu? else { return Ok(false) };
        let host = super::host_matmul_row_invariant(&cpu, &rhs.host_view(), (1, rows, n, k), &lhs_t, &rhs_t)?;
        let g = gpu.host_view();
        Ok(match (&*g, &host) {
            (crate::CpuStorage::F32(a), crate::CpuStorage::F32(b)) => {
                a.len() >= b.len() && a[..b.len()].iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        })
    };
    let ok = check().unwrap_or(false);
    eprintln!(
        "[VK-HOST-SMALL-M] weight {k}x{n}: host bits {} the GPU kernel's -> rows <= {} {}",
        if ok { "==" } else { "!=" },
        host_small_m(),
        if ok { "run on the host" } else { "stay on the GPU" }
    );
    if let Ok(mut g) = cache.lock() {
        g.insert(key, ok);
    }
    ok
}

pub(crate) fn dense_matmul(
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Option<VulkanStorage>> {
    let dev = &lhs.device;
    if !enabled(dev) || !mirrors_enabled_env() {
        return Ok(None);
    }
    let Some((tag, decl, load)) = rhs_variant(rhs.dtype) else {
        return Ok(None);
    };
    // Fixed GPU mode: no size thresholds, no GEMV, mirror on first use, errors are returned
    // (the path never depends on the batch size or on earlier calls).
    let fixed = shaders::exec_mode() == shaders::ExecMode::Gpu;
    let worthwhile = fixed
        || rhs.capacity_bytes >= MIRROR_MIN_BYTES
        || (rhs.capacity_bytes >= (256 << 10) && b * m >= 32);
    if lhs.dtype != rhs.dtype
        || !worthwhile
        || (!fixed && 2 * b * m * n * k < (1 << 22))
        || rhs_l.dims().len() < 2
        || lhs_l.dims().len() < 2
    {
        return Ok(None);
    }
    let rd = rhs_l.dims();
    let rs = rhs_l.stride();
    let rr = rd.len();
    if rd[..rr - 2].iter().zip(rs[..rr - 2].iter()).any(|(&d, &s)| d > 1 && s != 0) {
        return Ok(None);
    }
    let (sbk, sbn) = (rs[rr - 2], rs[rr - 1]);
    let w_off = rhs_l.start_offset();
    let big_m = b * m;
    let max_w = w_off + (k.max(1) - 1) * sbk + (n.max(1) - 1) * sbn + 1;
    if !fits_u32(max_w) || !fits_u32(big_m * n) || !fits_u32(big_m * k) {
        return Ok(None);
    }
    let ctx = dev.ctx();
    if fixed
        && unified(ctx)
        && lhs.dtype == DType::F32
        && (2..=host_small_m()).contains(&big_m)
        && !HOST_ROUTE_CHECK.with(|f| f.get())
        && host_route_ok(lhs, rhs, (n, k), (sbk, sbn, w_off))
    {
        return Ok(None);
    }
    let mirror = if unified(ctx) {
        None
    } else {
        match mirror_of(rhs, fixed) {
            Some(mb) => Some(mb),
            None if fixed => {
                return Err(Error::Msg(
                    "vulkan dense matmul (fixed GPU mode): cannot mirror the weight to VRAM".into(),
                ))
            }
            None => return Ok(None),
        }
    };
    let w_range = (rhs.capacity_bytes.max(16).div_ceil(16) * 16) as u64;
    let w_bind = match &mirror {
        None => Bind {
            buffer: rhs.buffer,
            offset: 0,
            range: w_range,
        },
        Some(mb) => Bind {
            buffer: mb.buffer,
            offset: 0,
            range: w_range,
        },
    };
    let res = (|| -> Result<VulkanStorage> {
        let mut converted = None;
        let (xs, x_off) = GpuQuant::prep_f32_input(dev, lhs, lhs_l, &mut converted)?;
        let out32 = dev.alloc_buffer(big_m * n, DType::F32)?;
        let gemv = !fixed && big_m <= GEMV_MAX_M;
        let x_aligned = unified(ctx) && x_off % 4 == 0 || !unified(ctx);
        let vec = dense_vec_variant(rhs.dtype).filter(|&(_, width, _)| {
            gemv && x_aligned && sbk == 1 && w_off % width == 0 && sbn % width == 0 && k % width == 0
        });
        let kern = match (gemv, vec) {
            (true, Some((vdecl, width, dotv))) => shaders::kernel(
                dev,
                &format!("dense_gemv_vx_{tag}"),
                3,
                44,
                || dense_gemv_vec_src(vdecl, width, dotv),
            )?,
            (true, None) => {
                shaders::kernel(dev, &format!("dense_gemv_{tag}"), 3, 44, || dense_gemv_src(decl, load))?
            }
            (false, _) => {
                shaders::kernel(dev, &format!("dense_gemm_{tag}"), 3, 44, || dense_gemm_src(decl, load))?
            }
        };
        let max_groups = dev.limits().max_workgroup_count;
        let row_bytes = k * 4;
        let band = if unified(ctx) || big_m * row_bytes <= scratch_cap() {
            big_m
        } else {
            ((scratch_cap() / row_bytes) / 64 * 64).max(64).min(big_m)
        };
        let cols_per = if gemv {
            (max_groups[0] as usize).min(65535) * 8
        } else {
            let by_flops = (MAX_DISPATCH_FLOPS / (2 * band * k).max(1)) / 64 * 64;
            by_flops.max(64).min((max_groups[0] as usize).min(65535) * 64)
        };
        let guard = if unified(ctx) {
            None
        } else {
            Some(scratch(ctx, band * row_bytes)?)
        };
        let uses_scratch = guard.is_some();
        let mut r0 = 0;
        while r0 < big_m {
            let rows = band.min(big_m - r0);
            let (x_bind, xo, prefix) = match &guard {
                None => (storage_bind(xs), x_off + r0 * k, Vec::new()),
                Some(s) => {
                    let sb = s.buffer();
                    (
                        Bind {
                            buffer: sb,
                            offset: 0,
                            range: (rows * row_bytes).max(4) as u64,
                        },
                        0,
                        vec![
                            Op::Copy {
                                src: xs.buffer,
                                src_off: ((x_off + r0 * k) * 4) as u64,
                                dst: sb,
                                dst_off: 0,
                                size: (rows * row_bytes) as u64,
                            },
                            Op::Barrier,
                        ],
                    )
                }
            };
            let mut dispatches = Vec::new();
            let mut c0 = 0;
            while c0 < n {
                let cols = cols_per.min(n - c0);
                let groups = if gemv {
                    [cols.div_ceil(8) as u32, rows as u32, 1]
                } else {
                    [cols.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1]
                };
                let p = push(&[rows, c0 + cols, k, xo, k, r0 * n, n, w_off, sbk, sbn, c0]);
                dispatches.push(Op::Dispatch {
                    kernel: kern.as_ref(),
                    binds: vec![x_bind, w_bind, storage_bind(&out32)],
                    push: p,
                    groups,
                });
                c0 += cols;
            }
            submit(dev, prefix, dispatches, uses_scratch)?;
            r0 += rows;
        }
        drop(guard);
        match lhs.dtype {
            DType::F32 => Ok(out32),
            DType::F16 | DType::BF16 => f32_to_f16_bf16_gpu(dev, &out32, lhs.dtype),
            dt => out32.to_dtype(&Layout::contiguous(big_m * n), dt),
        }
    })();
    match res {
        Ok(out) => Ok(Some(out)),
        Err(e) if fixed => Err(e),
        Err(_) => Ok(None)
    }
}

#[allow(dead_code)]
pub(crate) fn has_mirror(s: &VulkanStorage) -> bool {
    s.mirror.lock().map(|m| m.is_some()).unwrap_or(false)
}

pub(crate) fn act_matmul(
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    (batch, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Option<VulkanStorage>> {
    let dev = &lhs.device;
    let ctx = dev.ctx();
    // Fixed GPU mode keeps activation x activation matmuls on the CPU path (its flops
    // threshold would make the path depend on the batch size).
    if !enabled(dev) || unified(ctx) || !act_enabled_env() || shaders::exec_mode() == shaders::ExecMode::Gpu {
        return Ok(None);
    }
    if lhs.dtype != DType::F32 || rhs.dtype != DType::F32 {
        return Ok(None);
    }
    let flops = 2usize
        .saturating_mul(batch)
        .saturating_mul(m)
        .saturating_mul(n)
        .saturating_mul(k);
    if flops < act_min_flops() || lhs_l.dims().len() < 2 || rhs_l.dims().len() < 2 {
        return Ok(None);
    }
    let a0 = lhs_l.start_offset();
    let b0 = rhs_l.start_offset();
    let a_len = shaders::span(lhs_l).saturating_sub(a0);
    let b_len = shaders::span(rhs_l).saturating_sub(b0);
    if !fits_u32(a_len) || !fits_u32(b_len) || !fits_u32(batch * m * n) {
        return Ok(None);
    }
    let (Some(a), Some(b)) = (
        shaders::batch_strides(lhs_l, batch),
        shaders::batch_strides(rhs_l, batch),
    ) else {
        return Ok(None);
    };
    let Some((inner, sao, sai, sbo, sbi)) = shaders::unify_batches(a, b) else {
        return Ok(None);
    };
    let ls = lhs_l.stride();
    let rs = rhs_l.stride();
    let (lr, rr) = (ls.len(), rs.len());
    let (sam, sak) = (ls[lr - 2], ls[lr - 1]);
    let (sbk, sbn) = (rs[rr - 2], rs[rr - 1]);
    let a_bytes = a_len * 4;
    let b_bytes = b_len * 4;
    let b_off = a_bytes.div_ceil(256) * 256;
    let total = b_off + b_bytes.div_ceil(256) * 256 + 256;
    if total > 2 * scratch_cap() {
        return Ok(None);
    }
    let res = (|| -> Result<VulkanStorage> {
        let mut out = dev.alloc_buffer(batch * m * n, DType::F32)?;
        if m == 0 || n == 0 || batch == 0 {
            return Ok(out);
        }
        if k == 0 {
            out.as_bytes_mut().fill(0);
            return Ok(out);
        }
        let kern = shaders::kernel(dev, "matmul_tiled", 3, 64, || {
            shaders::GLSL_MATMUL.to_string()
        })?;
        let limits = dev.limits();
        let gx = n.div_ceil(64);
        if gx > limits.max_workgroup_count[0] as usize {
            crate::bail!("act matmul: n too large for one dispatch")
        }
        let guard = scratch(ctx, total)?;
        let sb = guard.buffer();
        let prefix = vec![
            Op::Copy {
                src: lhs.buffer,
                src_off: (a0 * 4) as u64,
                dst: sb,
                dst_off: 0,
                size: a_bytes as u64,
            },
            Op::Copy {
                src: rhs.buffer,
                src_off: (b0 * 4) as u64,
                dst: sb,
                dst_off: b_off as u64,
                size: b_bytes as u64,
            },
            Op::Barrier,
        ];
        let a_bind = Bind {
            buffer: sb,
            offset: 0,
            range: (a_bytes.max(16).div_ceil(16) * 16) as u64,
        };
        let b_bind = Bind {
            buffer: sb,
            offset: b_off as u64,
            range: (b_bytes.max(16).div_ceil(16) * 16) as u64,
        };
        let max_band_flops: usize = 1 << 31;
        let row_flops = 2 * n * k;
        let rows_per_band = ((max_band_flops / row_flops.max(1)) / 64 * 64)
            .clamp(64, (limits.max_workgroup_count[1] as usize) * 64);
        let batch_per_band = (max_band_flops / (row_flops * m.min(rows_per_band)).max(1))
            .clamp(1, (limits.max_workgroup_count[2] as usize).min(65535));
        let mut dispatches = Vec::new();
        let mut z0 = 0;
        while z0 < batch {
            let zb = batch_per_band.min(batch - z0);
            let mut r0 = 0;
            while r0 < m {
                let rb = rows_per_band.min(m - r0);
                let p = push(&[
                    m, n, k, sam, sak, sbk, sbn, 0, 0, inner, sao, sai, sbo, sbi, r0, z0,
                ]);
                dispatches.push(Op::Dispatch {
                    kernel: kern.as_ref(),
                    binds: vec![a_bind, b_bind, storage_bind(&out)],
                    push: p,
                    groups: [gx as u32, rb.div_ceil(64) as u32, zb as u32],
                });
                r0 += rb;
            }
            z0 += zb;
        }
        submit(dev, prefix, dispatches, true)?;
        drop(guard);
        Ok(out)
    })();
    match res {
        Ok(out) => Ok(Some(out)),
        Err(_) => Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        use naga::front::glsl::{Frontend, Options};
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = Frontend::default()
            .parse(&Options::from(naga::ShaderStage::Compute), src)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}\n{src}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("validation failed: {e:?}"));
    }

    #[test]
    fn kquant_shaders_validate() {
        for fmt in [
            QuantFmt::Q8_0,
            QuantFmt::Q4_0,
            QuantFmt::Q4K,
            QuantFmt::Q5K,
            QuantFmt::Q6K,
        ] {
            validate(&quant_gemv_src(fmt));
            validate(&quant_gemm_src(fmt));
            validate(&quant_dequant_src(fmt));
            validate(&quant_gather_src(fmt));
            validate(&quant_moe_src(fmt));
        }
        validate(&unpack_p16_to_f32_src(false));
        validate(&unpack_p16_to_f32_src(true));
        validate(&pack_f32_to_p16_src(false));
        validate(&pack_f32_to_p16_src(true));
    }
}
