//! GPU memory for the large read-only matmul operands and the kernels that read it.
//!
//! The rest of this backend keeps every tensor in host visible memory so that the CPU
//! fallbacks can work on it in place. That is the right layout on unified memory (iGPU),
//! but on a discrete GPU it means every weight would be read over PCIe, so the GPU
//! kernels stay off there and the work runs on the CPU. This module moves the operands
//! that dominate an LLM forward pass into memory the GPU reads at full speed:
//!
//! * quantized weights (`Q8_0`) are repacked into a GPU friendly layout and run through
//!   dedicated GEMV/GEMM, embedding and dequantization kernels;
//! * dense weights (`F32`/`F16`/`BF16`) used as the right hand side of a matmul get a GPU
//!   copy (a "mirror") the second time they are used that way.
//!
//! Placement follows the device kind:
//! * discrete GPU: device local VRAM, uploaded through a staging buffer; activations are
//!   copied into a VRAM scratch buffer per call and results are written straight into
//!   the host visible output tensor;
//! * integrated GPU: the device local carve-out first and then the shared system memory
//!   heap (the "shared GPU memory" of the OS), both read directly by the GPU, no copies.
//!
//! A heap only takes an allocation while its `VK_EXT_memory_budget` budget keeps a
//! reserve free, otherwise the next memory kind is tried and finally the CPU path.
//!
//! Environment: `CANDLE_VULKAN_GPU_WEIGHTS=0` disables all of it,
//! `CANDLE_VULKAN_DENSE_MIRROR=0` only the dense mirrors, `CANDLE_VULKAN_SCRATCH_MB`
//! (default 128) bounds the activation scratch, `CANDLE_VULKAN_VRAM_RESERVE_MB`
//! (default 512) is the free VRAM kept for the desktop and other processes.

use super::shaders::{self, Bind, Op};
use super::{VulkanContext, VulkanDevice, VulkanStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{DType, Error, Layout, Result};
use ash::vk;
use half::f16;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

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

/// Activation matmuls below this many FLOPs stay on the CPU (copies cost more).
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

fn debug() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_on("CANDLE_VULKAN_DEBUG", false))
}

pub(crate) fn note_fallback(what: &str, e: &Error) {
    if debug() {
        eprintln!("vulkan: {what} fell back to the CPU path: {e}");
    }
}

/// Whether weights may be placed in GPU memory on `dev`.
pub(crate) fn enabled(dev: &VulkanDevice) -> bool {
    weights_enabled_env()
        && dev.ctx().device_type != vk::PhysicalDeviceType::CPU
        && dev.ctx().native_ok.load(Ordering::Relaxed)
}

/// Testing aid: treat a discrete GPU like an integrated one (weights go to the small
/// device local + host visible heap first, then spill to shared system memory, and
/// activations are bound in place), to exercise the integrated GPU code paths.
fn pretend_integrated() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_on("CANDLE_VULKAN_PRETEND_INTEGRATED", false))
}

/// Unified memory: the GPU reads the host visible tensor memory at full speed, so
/// activations and dense weights are bound directly instead of being copied.
fn unified(ctx: &VulkanContext) -> bool {
    ctx.default_native || pretend_integrated()
}

// ---------------------------------------------------------------------------
// Per-device state
// ---------------------------------------------------------------------------

/// Bookkeeping of this module, stored in the device context.
#[derive(Default)]
pub(crate) struct GpuState {
    /// Bytes of device local memory held by GPU buffers of this module.
    pub(crate) local_bytes: AtomicUsize,
    /// Bytes of host (shared) memory held by GPU buffers of this module.
    pub(crate) shared_bytes: AtomicUsize,
    pub(crate) buffers: AtomicUsize,
    scratch: Mutex<Option<RawBuf>>,
}

/// A buffer owned by the context itself (no back reference to the device).
pub(crate) struct RawBuf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    local: bool,
}

impl GpuState {
    /// Frees the activation scratch buffer (it is re-created on demand).
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

// ---------------------------------------------------------------------------
// Memory placement
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Use {
    Weights,
    Scratch,
}

/// `(budget, usage)` per heap from `VK_EXT_memory_budget`.
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

/// Memory types for `use_`, best first.
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
            // discrete: VRAM, then a resizable BAR window; never system RAM (the CPU
            // path reads that faster than the GPU does over PCIe)
            (Use::Weights, true, true, false) => 0,
            (Use::Weights, true, true, true) if hsize >= (1 << 30) => 1,
            (Use::Weights, true, _, _) => continue,
            // integrated: the carve-out, then the shared system memory heap
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

/// Whether `size` more bytes fit in `heap` while keeping its reserve free.
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

/// Heaps that hold GPU working memory, for `mem_info`: the VRAM heap on a discrete GPU,
/// the carve-out plus the shared heap on an integrated one.
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

/// Allocates and binds memory for a new buffer of `size` bytes. Returns
/// `(buffer, memory, mapped, device_local, alloc_size)`.
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

/// A buffer in GPU memory owned by a tensor.
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

/// Copies `data` into `dst` at `offset` (directly when mapped, else through staging).
pub(crate) fn write(dev: &VulkanDevice, dst: &GpuBuf, offset: usize, data: &[u8]) -> Result<()> {
    let mut _p = super::prof::scope("gpu", "write(staging)");
    _p.work(data.len());
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
        shaders::run_ops(
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

/// Copies `out.len()` bytes of `src` starting at `offset` into `out`.
pub(crate) fn read(dev: &VulkanDevice, src: &GpuBuf, offset: usize, out: &mut [u8]) -> Result<()> {
    let mut _p = super::prof::scope("gpu", "read(staging)");
    _p.work(out.len());
    if out.is_empty() {
        return Ok(());
    }
    if offset + out.len() > src.size {
        crate::bail!("vulkan gpu read out of bounds")
    }
    if !src.mapped.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(src.mapped.add(offset), out.as_mut_ptr(), out.len()) };
        return Ok(());
    }
    let chunk = STAGING_CHUNK.min(out.len());
    let staging = dev.alloc_buffer(chunk, DType::U8)?;
    let mut done = 0;
    while done < out.len() {
        let n = chunk.min(out.len() - done);
        shaders::run_ops(
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

/// The activation scratch buffer (VRAM), grown on demand, held while the guard lives.
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

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

fn push(vals: &[usize]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        v.extend_from_slice(&(x as u32).to_le_bytes());
    }
    v
}

/// The whole storage, rounded to 16 bytes so vec4 views cover the tail (storages are
/// bucketed: the buffer behind them is always at least that large).
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

/// Splits `ops` dispatch-wise into submissions of at most `shaders::MAX_SETS`
/// dispatches; `prefix` (copies) runs first in the first submission.
fn submit(dev: &VulkanDevice, prefix: Vec<Op<'_>>, dispatches: Vec<Op<'_>>) -> Result<()> {
    let mut pending: Vec<Op<'_>> = prefix;
    let mut count = 0u32;
    for d in dispatches {
        if count == shaders::MAX_SETS {
            shaders::run_ops(dev, &pending)?;
            pending = Vec::new();
            count = 0;
        }
        pending.push(d);
        count += 1;
    }
    if !pending.is_empty() {
        shaders::run_ops(dev, &pending)?;
    }
    Ok(())
}

// Q8_0, repacked: the 32 int8 quants of block `i` at bytes `[32 i, 32 i + 32)` of `wq`,
// its scale as an f32 at `wd[i]`. Block `i` of row `r` of an `(n, k)` matrix is
// `r * k / 32 + c`.

/// Small m: 8 output columns per workgroup, 32 lanes per column striding over blocks.
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

/// Small m with 16 byte loads: the 32 quants of a block as two uvec4, x as vec4
/// (needs x rows aligned to 4 floats).
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

/// Larger m: 64x64 output tiles, 4x4 per thread, one Q8_0 block (32 k) per step.
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

/// Embedding rows: `y[i, c] = wd[blk] * q(blk, c)` with `blk = ids[i] * nb + c / 32`.
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

/// Whole tensor to f32 (`HALF == 0`) or to packed f16 pairs (`HALF == 1`).
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
/// Upper bound of the work of one dispatch (2 * rows * cols * k), keeps every
/// dispatch far below the OS GPU watchdog (2 s on Windows).
const MAX_DISPATCH_FLOPS: usize = 1 << 34;

/// Repacked `Q8_0` weights in GPU memory.
pub(crate) struct GpuQ8 {
    pub(crate) buf: GpuBuf,
    pub(crate) nblocks: usize,
    d_off: usize,
}

impl std::fmt::Debug for GpuQ8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuQ8")
            .field("nblocks", &self.nblocks)
            .field("buf", &self.buf)
            .finish()
    }
}

pub(crate) const Q8_BLOCK_BYTES: usize = 34;

impl GpuQ8 {
    /// Moves GGML `Q8_0` blocks into GPU memory, or fails when no GPU memory kind has
    /// room (the caller keeps the CPU layout then).
    pub(crate) fn from_ggml(dev: &VulkanDevice, bytes: &[u8]) -> Result<Self> {
        let mut _p = super::prof::scope("q8", "upload(repack+copy)");
        _p.work(bytes.len());
        if !bytes.len().is_multiple_of(Q8_BLOCK_BYTES) {
            crate::bail!("vulkan q8: {} bytes are not whole Q8_0 blocks", bytes.len())
        }
        let nb = bytes.len() / Q8_BLOCK_BYTES;
        let qs_bytes = nb * 32;
        let d_off = qs_bytes.div_ceil(256) * 256;
        let total = d_off + nb * 4;
        let max_range = dev.limits().max_storage_buffer_range as usize;
        if qs_bytes > max_range || !fits_u32(nb * 32) {
            crate::bail!("vulkan q8: tensor of {qs_bytes} bytes exceeds the storage buffer range")
        }
        let buf = GpuBuf::new(dev, total, Use::Weights)?;
        let mut packed = vec![0u8; total];
        for i in 0..nb {
            let blk = &bytes[i * Q8_BLOCK_BYTES..(i + 1) * Q8_BLOCK_BYTES];
            let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            packed[i * 32..i * 32 + 32].copy_from_slice(&blk[2..34]);
            packed[d_off + i * 4..d_off + i * 4 + 4].copy_from_slice(&d.to_le_bytes());
        }
        write(dev, &buf, 0, &packed)?;
        Ok(Self {
            buf,
            nblocks: nb,
            d_off,
        })
    }

    /// The GGML `Q8_0` bytes of this tensor (downloads them from the GPU).
    pub(crate) fn to_ggml(&self) -> Result<Vec<u8>> {
        let _p = super::prof::scope("q8", "download");
        let nb = self.nblocks;
        let mut packed = vec![0u8; self.d_off + nb * 4];
        read(&self.buf.dev, &self.buf, 0, &mut packed)?;
        let mut out = vec![0u8; nb * Q8_BLOCK_BYTES];
        for i in 0..nb {
            let o = &mut out[i * Q8_BLOCK_BYTES..(i + 1) * Q8_BLOCK_BYTES];
            let d = f32::from_le_bytes([
                packed[self.d_off + i * 4],
                packed[self.d_off + i * 4 + 1],
                packed[self.d_off + i * 4 + 2],
                packed[self.d_off + i * 4 + 3],
            ]);
            o[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            o[2..34].copy_from_slice(&packed[i * 32..i * 32 + 32]);
        }
        Ok(out)
    }

    pub(crate) fn ggml_bytes(&self) -> usize {
        self.nblocks * Q8_BLOCK_BYTES
    }

    fn qs_bind(&self) -> Bind {
        Bind {
            buffer: self.buf.buffer,
            offset: 0,
            range: (self.nblocks * 32) as u64,
        }
    }

    fn d_bind(&self) -> Bind {
        Bind {
            buffer: self.buf.buffer,
            offset: self.d_off as u64,
            range: (self.nblocks * 4) as u64,
        }
    }

    /// `x @ W^T` for `W` of shape `(n, k)`, `x` of `m` rows. Output is f32.
    pub(crate) fn matmul(
        &self,
        (m, k, n): (usize, usize, usize),
        x: &VulkanStorage,
        layout: &Layout,
    ) -> Result<VulkanStorage> {
        let dev = x.device.clone();
        let mut _p = super::prof::scope("q8", if m <= GEMV_MAX_M { "gemv" } else { "gemm" });
        _p.work(2 * m * n * k);
        if !k.is_multiple_of(32) || self.nblocks * 32 != n * k {
            crate::bail!("vulkan q8 matmul: weights do not hold {n}x{k} values")
        }
        let converted;
        let (xs, x_off) = match (x.dtype, layout.contiguous_offsets()) {
            (DType::F32, Some((a, _))) => (x, a),
            _ => {
                let view = x.host_view();
                let cpu = view.to_dtype(layout, DType::F32)?;
                converted = dev.storage_from_cpu_storage(&cpu)?;
                (&converted, 0)
            }
        };
        let out = dev.alloc_buffer(m * n, DType::F32)?;
        if m == 0 || n == 0 {
            return Ok(out);
        }
        if !fits_u32(m * n) || !fits_u32(x_off + m * k) {
            crate::bail!("vulkan q8 matmul: problem too large for 32 bit indexing")
        }
        let nb = k / 32;
        let gemv = m <= GEMV_MAX_M;
        // the vector kernel reads x as vec4: rows must start 16 byte aligned
        let x_aligned = !unified(dev.ctx()) || x_off % 4 == 0;
        let kern = match (gemv, x_aligned) {
            (true, true) => shaders::kernel(&dev, "q8_gemv_v", 4, 36, || GLSL_Q8_GEMV_V.to_string())?,
            (true, false) => shaders::kernel(&dev, "q8_gemv", 4, 36, || GLSL_Q8_GEMV.to_string())?,
            (false, _) => shaders::kernel(&dev, "q8_gemm", 4, 36, || GLSL_Q8_GEMM.to_string())?,
        };
        let max_groups = dev.limits().max_workgroup_count;
        let ctx = dev.ctx();
        let row_bytes = k * 4;
        // rows processed per pass (all of them unless the scratch would get too big)
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
                // c0 is folded into the column index; n stays the full width so the
                // stores land in the right place.
                let p = push(&[rows, (c0 + cols), k, nb, xo, k, r0 * n, n, c0]);
                dispatches.push(Op::Dispatch {
                    kernel: kern.as_ref(),
                    binds: vec![x_bind, self.qs_bind(), self.d_bind(), storage_bind(&out)],
                    push: p,
                    groups,
                });
                c0 += cols;
            }
            submit(&dev, prefix, dispatches)?;
            r0 += rows;
        }
        drop(scratch_guard);
        Ok(out)
    }

    /// Rows `ids` of the `(rows, hidden)` table, as f32.
    pub(crate) fn embedding(&self, rows: usize, hidden: usize, ids: &[u32], dev: &VulkanDevice) -> Result<VulkanStorage> {
        let _p = super::prof::scope("q8", "gather");
        if !hidden.is_multiple_of(32) || self.nblocks * 32 != rows * hidden {
            crate::bail!("vulkan q8 embedding: table does not hold {rows}x{hidden} values")
        }
        let total = ids.len() * hidden;
        let out = dev.alloc_buffer(total, DType::F32)?;
        if total == 0 {
            return Ok(out);
        }
        if !fits_u32(total) {
            crate::bail!("vulkan q8 embedding: output too large")
        }
        let id_bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
        let ids_s = dev.upload_bytes(id_bytes, ids.len(), DType::U32)?;
        let kern = shaders::kernel(dev, "q8_gather", 4, 20, || GLSL_Q8_GATHER.to_string())?;
        let groups = total.div_ceil(256).min(65535);
        let p = push(&[total, hidden, hidden / 32, rows, groups * 256]);
        shaders::run_ops(
            dev,
            &[Op::Dispatch {
                kernel: kern.as_ref(),
                binds: vec![storage_bind(&ids_s), self.qs_bind(), self.d_bind(), storage_bind(&out)],
                push: p,
                groups: [groups as u32, 1, 1],
            }],
        )?;
        Ok(out)
    }

    /// The first `elem_count` values as f32 or f16.
    pub(crate) fn dequantize(&self, elem_count: usize, dtype: DType, dev: &VulkanDevice) -> Result<VulkanStorage> {
        let _p = super::prof::scope("q8", "dequant");
        let elem_count = elem_count.min(self.nblocks * 32);
        let half = match dtype {
            DType::F32 => false,
            DType::F16 => true,
            dt => crate::bail!("vulkan q8 dequantize to {dt:?} is not supported"),
        };
        if half && elem_count % 2 != 0 {
            crate::bail!("vulkan q8 dequantize: odd element count for f16 output")
        }
        let out = dev.alloc_buffer(elem_count, dtype)?;
        if elem_count == 0 {
            return Ok(out);
        }
        if !fits_u32(elem_count) {
            crate::bail!("vulkan q8 dequantize: tensor too large")
        }
        let kern = shaders::kernel(dev, "q8_dequant", 3, 12, || GLSL_Q8_DEQUANT.to_string())?;
        let work = if half { elem_count / 2 } else { elem_count };
        let groups = work.div_ceil(256).min(65535);
        let p = push(&[elem_count, half as usize, groups * 256]);
        shaders::run_ops(
            dev,
            &[Op::Dispatch {
                kernel: kern.as_ref(),
                binds: vec![self.qs_bind(), self.d_bind(), storage_bind(&out)],
                push: p,
                groups: [groups as u32, 1, 1],
            }],
        )?;
        Ok(out)
    }
}

/// Whether a `Q8_0` tensor of `bytes` should live in GPU memory.
pub(crate) fn want_q8_on_gpu(dev: &VulkanDevice, bytes: usize) -> bool {
    enabled(dev) && bytes >= 64 * Q8_BLOCK_BYTES
}

// ---------------------------------------------------------------------------
// Dense matmul with GPU resident right hand side
// ---------------------------------------------------------------------------

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

/// GEMV for a right hand side that is contiguous along k (a weight matrix used as
/// `w.t()`), reading 16 bytes per lane per load with four loads in flight:
/// `(binding declaration, elements per vector, dot function)`.
fn dense_vec_variant(dtype: DType) -> Option<(&'static str, usize, &'static str)> {
    // x is read as vec4 at element offset `xo` (a multiple of 4)
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

/// Matmuls whose right hand side has fewer bytes stay on the regular path.
const MIRROR_MIN_BYTES: usize = 4 << 20;

fn mirror_after() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| env_mb("CANDLE_VULKAN_MIRROR_AFTER", 2).max(1) as u32)
}

/// The GPU copy of `rhs` (made on its `mirror_after`-th use as a matmul rhs).
fn mirror_of(rhs: &VulkanStorage) -> Option<Arc<GpuBuf>> {
    let mut slot = rhs.mirror.lock().ok()?;
    if let Some(m) = slot.as_ref() {
        return Some(m.clone());
    }
    let uses = rhs.rhs_uses.fetch_add(1, Ordering::Relaxed) + 1;
    if uses < mirror_after() {
        return None;
    }
    let bytes = rhs.as_bytes();
    let mut _p = super::prof::scope("gpu", "mirror_create");
    _p.work(bytes.len());
    let size = bytes.len().max(4).div_ceil(4) * 4;
    let buf = match GpuBuf::new(&rhs.device, size, Use::Weights) {
        Ok(b) => b,
        Err(e) => {
            note_fallback("dense mirror allocation", &e);
            // do not retry on every call
            rhs.rhs_uses.store(0, Ordering::Relaxed);
            return None;
        }
    };
    if let Err(e) = write(&rhs.device, &buf, 0, bytes) {
        note_fallback("dense mirror upload", &e);
        return None;
    }
    let buf = Arc::new(buf);
    *slot = Some(buf.clone());
    Some(buf)
}

/// `lhs @ rhs` on the GPU when `rhs` looks like a weight matrix (one matrix shared by
/// all batches, large enough) and lives, or can be mirrored, in GPU memory.
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
    // worth a GPU copy: a big weight, or a smaller one that multiplies many rows
    let worthwhile = rhs.capacity_bytes >= MIRROR_MIN_BYTES
        || (rhs.capacity_bytes >= (256 << 10) && b * m >= 32);
    if lhs.dtype != rhs.dtype
        || !worthwhile
        || 2 * b * m * n * k < (1 << 22)
        || rhs_l.dims().len() < 2
        || lhs_l.dims().len() < 2
    {
        return Ok(None);
    }
    // the rhs must be one matrix shared by every batch
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
    // the mirror stays alive (Arc) for the duration of the call
    let mirror = if unified(ctx) {
        None
    } else {
        match mirror_of(rhs) {
            Some(mb) => Some(mb),
            None => return Ok(None),
        }
    };
    // ranges rounded to 16 bytes so vec4/uvec4 views cover the tail (the buffers are
    // at least that large: storages are bucketed, mirrors rounded to 256 bytes)
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
        // contiguous f32 lhs of (b*m, k)
        let view = lhs.host_view();
        let converted;
        let (xs, x_off) = match (lhs.dtype, lhs_l.contiguous_offsets()) {
            (DType::F32, Some((a, _))) => (lhs, a),
            _ => {
                let cpu = view.to_dtype(lhs_l, DType::F32)?;
                converted = dev.storage_from_cpu_storage(&cpu)?;
                (&converted, 0)
            }
        };
        let out32 = dev.alloc_buffer(big_m * n, DType::F32)?;
        let gemv = big_m <= GEMV_MAX_M;
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
            submit(dev, prefix, dispatches)?;
            r0 += rows;
        }
        drop(guard);
        if lhs.dtype == DType::F32 {
            return Ok(out32);
        }
        out32.to_dtype(&Layout::contiguous(big_m * n), lhs.dtype)
    })();
    match res {
        Ok(out) => Ok(Some(out)),
        Err(e) => {
            note_fallback("dense matmul", &e);
            Ok(None)
        }
    }
}

#[allow(dead_code)]
/// Whether `s` has a GPU mirror (diagnostics).
pub(crate) fn has_mirror(s: &VulkanStorage) -> bool {
    s.mirror.lock().map(|m| m.is_some()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Activation x activation matmul on a discrete GPU
// ---------------------------------------------------------------------------

/// `lhs @ rhs` for two activations (attention scores, attention x values, ...) on a
/// discrete GPU: the spans of both operands are copied into the VRAM scratch, the tiled
/// matmul kernel runs there and writes the result straight into the host visible
/// output. Only for f32 and large enough problems; everything else stays on the CPU.
pub(crate) fn act_matmul(
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    (batch, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Option<VulkanStorage>> {
    let dev = &lhs.device;
    let ctx = dev.ctx();
    if !enabled(dev) || unified(ctx) || !act_enabled_env() {
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
        submit(dev, prefix, dispatches)?;
        drop(guard);
        Ok(out)
    })();
    match res {
        Ok(out) => Ok(Some(out)),
        Err(e) => {
            note_fallback("activation matmul", &e);
            Ok(None)
        }
    }
}
