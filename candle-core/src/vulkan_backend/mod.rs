//! Vulkan backend for candle-core.
//!
//! Enabled with the `vulkan` feature. Uses `ash` with its `loaded` feature, so
//! the Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`) is opened at runtime
//! rather than linked: a machine without a Vulkan driver gets a clear runtime
//! error instead of a link failure, and no Vulkan SDK is needed to build.
//!
//! Memory model: every tensor lives in one `HOST_VISIBLE | HOST_COHERENT`
//! buffer that stays mapped for its whole lifetime (`HOST_CACHED` memory is
//! preferred so CPU reads are fast). Compute kernels compiled from GLSL with
//! `naga` (see [`shaders`]) run on the GPU for the shapes/dtypes they cover;
//! every other operation runs the regular CPU implementation directly on the
//! mapped memory (no staging copies for the inputs) and writes its result into
//! a new device buffer.
//!
//! Devices are cached per ordinal: `VulkanDevice::new(i)` returns the same
//! logical device (same `VkDevice`, queue, kernel cache) for as long as any
//! clone or storage of it is alive, which keeps `Device::same_device` true for
//! independently created handles of the same GPU.
#![allow(clippy::missing_safety_doc)]

pub(crate) mod qgpu;
pub mod shaders;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use ash::vk;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// The `DeviceLocation::Vulkan` variant carries a `gpu_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

const DEFAULT_SEED: u64 = 299792458;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub max_storage_buffer_range: u64,
    pub max_workgroup_count: [u32; 3],
    pub max_push_constants_size: u32,
}

/// One Vulkan logical device with its queue, kernel cache and allocator state.
pub(crate) struct VulkanContext {
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    /// Command pool / buffer / descriptor pool / fence used for every
    /// dispatch. Host access to a `VkQueue` is externally synchronized, so the
    /// mutex also serializes every submit + wait.
    exec: Mutex<shaders::ExecState>,
    kernels: Mutex<HashMap<String, Arc<shaders::Kernel>>>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// HOST_VISIBLE | HOST_COHERENT memory types, best first.
    memory_types: Vec<u32>,
    memory_budget_ext: bool,
    allocated_bytes: AtomicUsize,
    live_allocations: AtomicUsize,
    /// Released buffers kept for reuse, keyed by their size bucket.
    pool: Mutex<Pool>,
    pool_limit: usize,
    /// Shared allocations carved into equal slots, keyed by size bucket. Used
    /// once the number of live allocations approaches the driver limit.
    slabs: Mutex<HashMap<usize, Vec<SlabChunk>>>,
    /// `maxMemoryAllocationCount` reported by the driver.
    max_allocations: usize,
    /// 0: automatic, 1: always use slabs, 2: never (`CANDLE_VULKAN_SLAB`).
    slab_mode: u8,
    limits: Limits,
    name: String,
    device_type: vk::PhysicalDeviceType,
    rng: Mutex<(u64, rand::rngs::StdRng)>,
    /// Cleared after an unrecoverable dispatch error (e.g. device lost) so the
    /// remaining work keeps running on the CPU path.
    native_ok: AtomicBool,
    /// GPU kernels are used by default on integrated GPUs and whenever the
    /// tensor memory is device local.
    default_native: bool,
    pub(crate) gpu: qgpu::GpuState,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

struct PooledBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    alloc_size: usize,
    bucket: usize,
    /// Slot index when the buffer lives in a slab allocation.
    slot: Option<u32>,
}

#[derive(Default)]
pub(crate) struct Pool {
    free: HashMap<usize, Vec<PooledBuffer>>,
    bytes: usize,
}

/// One device memory allocation shared by `slots` buffers of the same bucket.
struct SlabChunk {
    memory: vk::DeviceMemory,
    memory_type: u32,
    mapped: *mut u8,
    alloc_size: usize,
    stride: usize,
    free: Vec<u32>,
    in_use: usize,
}

impl VulkanContext {
    unsafe fn release(&self, b: PooledBuffer) {
        if let Some(slot) = b.slot {
            unsafe { self.device.destroy_buffer(b.buffer, None) };
            self.release_slot(b.bucket, b.memory, slot);
            return;
        }
        unsafe {
            self.device.unmap_memory(b.memory);
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
        self.allocated_bytes
            .fetch_sub(b.alloc_size, Ordering::Relaxed);
        self.live_allocations.fetch_sub(1, Ordering::Relaxed);
    }

    /// Returns a slab slot; the allocation is freed with its last slot.
    fn release_slot(&self, bucket: usize, memory: vk::DeviceMemory, slot: u32) {
        let emptied = {
            let mut slabs = match self.slabs.lock() {
                Ok(s) => s,
                Err(_) => return,
            };
            let Some(list) = slabs.get_mut(&bucket) else {
                return;
            };
            let Some(i) = list.iter().position(|c| c.memory == memory) else {
                return;
            };
            let chunk = &mut list[i];
            chunk.free.push(slot);
            chunk.in_use = chunk.in_use.saturating_sub(1);
            if chunk.in_use == 0 {
                Some(list.swap_remove(i))
            } else {
                None
            }
        };
        if let Some(chunk) = emptied {
            unsafe { self.free_chunk(chunk) };
        }
    }

    unsafe fn free_chunk(&self, chunk: SlabChunk) {
        unsafe {
            self.device.unmap_memory(chunk.memory);
            self.device.free_memory(chunk.memory, None);
        }
        self.allocated_bytes
            .fetch_sub(chunk.alloc_size, Ordering::Relaxed);
        self.live_allocations.fetch_sub(1, Ordering::Relaxed);
    }

    fn use_slabs(&self) -> bool {
        match self.slab_mode {
            1 => true,
            2 => false,
            _ => self.live_allocations.load(Ordering::Relaxed) >= self.max_allocations / 2,
        }
    }

    fn trim_pool(&self) {
        let drained: Vec<PooledBuffer> = match self.pool.lock() {
            Ok(mut pool) => {
                pool.bytes = 0;
                pool.free.drain().flat_map(|(_, v)| v).collect()
            }
            Err(_) => return,
        };
        for b in drained {
            unsafe { self.release(b) };
        }
    }
}

/// Allocation size class: powers of two up to 1 MiB, then 1 MiB steps.
fn bucket_size(bytes: usize) -> usize {
    let b = bytes.max(256);
    if b <= (1 << 20) {
        b.next_power_of_two()
    } else {
        b.div_ceil(1 << 20) << 20
    }
}

const POOL_MAX_BUFFER: usize = 256 << 20;

/// Buckets up to this size can be carved out of shared slab allocations.
const SLAB_MAX_BUCKET: usize = 16 << 20;
/// Target size of one slab allocation.
const SLAB_CHUNK_BYTES: usize = 64 << 20;
/// Upper bound on the number of slots in one slab.
const SLAB_MAX_SLOTS: usize = 1024;

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.trim_pool();
        self.gpu.release_scratch(&self.device);
        let chunks: Vec<SlabChunk> = match self.slabs.lock() {
            Ok(mut slabs) => slabs.drain().flat_map(|(_, v)| v).collect(),
            Err(_) => Vec::new(),
        };
        for chunk in chunks {
            unsafe { self.free_chunk(chunk) };
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(mut kernels) = self.kernels.lock() {
                for (_, k) in kernels.drain() {
                    k.destroy(&self.device);
                }
            }
            if let Ok(mut exec) = self.exec.lock() {
                exec.destroy(&self.device);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A Vulkan device: an immutable `gpu_id` + the shared device/queue handles.
pub struct VulkanDevice {
    gpu_id: usize,
    inner: Arc<VulkanContext>,
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        VulkanDevice {
            gpu_id: self.gpu_id,
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("gpu_id", &self.gpu_id)
            .field("name", &self.inner.name)
            .finish()
    }
}

fn registry() -> &'static Mutex<HashMap<usize, Weak<VulkanContext>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<VulkanContext>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_entry() -> Result<ash::Entry> {
    match unsafe { ash::Entry::load() } {
        Ok(e) => Ok(e),
        Err(e) => Err(Error::Msg(format!(
            "vulkan: failed to load the Vulkan loader ({e}); is a Vulkan driver installed?"
        ))),
    }
}

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app_name = c"candle-vulkan";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(app_name)
        .api_version(vk::API_VERSION_1_2);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| Error::Msg(format!("vulkan: create_instance failed: {e:?}")))
}

fn device_type_rank(t: vk::PhysicalDeviceType) -> u32 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::OTHER => 3,
        vk::PhysicalDeviceType::CPU => 4,
        _ => 5,
    }
}

fn sorted_physical_devices(instance: &ash::Instance) -> Result<Vec<vk::PhysicalDevice>> {
    let list = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| Error::Msg(format!("vulkan: enumerate_physical_devices failed: {e:?}")))?;
    let allow_cpu = std::env::var("CANDLE_VULKAN_ALLOW_CPU")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
        })
        .unwrap_or(false);
    let mut ranked = Vec::with_capacity(list.len());
    for (i, p) in list.into_iter().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(p) };
        if props.api_version < vk::API_VERSION_1_1 {
            continue;
        }
        if props.device_type == vk::PhysicalDeviceType::CPU && !allow_cpu {
            continue;
        }
        let queues = unsafe { instance.get_physical_device_queue_family_properties(p) };
        if !queues
            .iter()
            .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            continue;
        }
        let mem = unsafe { instance.get_physical_device_memory_properties(p) };
        let vram = (0..mem.memory_heap_count as usize)
            .filter(|&h| mem.memory_heaps[h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| mem.memory_heaps[h].size)
            .max()
            .unwrap_or(0);
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        ranked.push((
            device_type_rank(props.device_type),
            std::cmp::Reverse(vram),
            i,
            p,
            name,
        ));
    }
    ranked.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    if let Ok(want) = std::env::var("CANDLE_VULKAN_DEVICE") {
        let want = want.trim().to_lowercase();
        if !want.is_empty() {
            if let Some(pos) = ranked.iter().position(|r| r.4.contains(&want)) {
                let r = ranked.remove(pos);
                ranked.insert(0, r);
            }
        }
    }
    Ok(ranked.into_iter().map(|r| r.3).collect())
}

/// Number of Vulkan devices usable as candle devices.
pub fn device_count() -> Result<usize> {
    let entry = load_entry()?;
    let instance = create_instance(&entry)?;
    let res = sorted_physical_devices(&instance).map(|v| v.len());
    unsafe { instance.destroy_instance(None) };
    res
}

/// Orders the host visible + coherent memory types: cached memory first (fast
/// CPU reads for the CPU fallbacks), device local cached memory (unified memory
/// APUs) before plain cached memory, and uncached memory last, preferring
/// system memory over a BAR window. `CANDLE_VULKAN_MEMORY=device` puts device
/// local types (resizable BAR) first instead.
fn memory_type_order(props: &vk::PhysicalDeviceMemoryProperties) -> Vec<u32> {
    let prefer_device = std::env::var("CANDLE_VULKAN_MEMORY")
        .map(|v| v.eq_ignore_ascii_case("device"))
        .unwrap_or(false);
    let mut ranked = Vec::new();
    for i in 0..props.memory_type_count as usize {
        let t = props.memory_types[i];
        let f = t.property_flags;
        if !f.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            continue;
        }
        let cached = f.contains(vk::MemoryPropertyFlags::HOST_CACHED);
        let local = f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let heap = props.memory_heaps[t.heap_index as usize].size;
        let small_heap = heap < (512u64 << 20);
        let score: i32 = if prefer_device {
            (local as i32) * 8 + (cached as i32) * 2
        } else {
            (cached as i32) * 8 + ((cached && local) as i32) * 2 - ((!cached && local) as i32) * 2
        } - (small_heap as i32) * 16;
        ranked.push((score, heap, i as u32));
    }
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    ranked.into_iter().map(|(_, _, i)| i).collect()
}

fn init_vulkan(gpu_id: usize) -> Result<VulkanContext> {
    let entry = load_entry()?;
    let instance = create_instance(&entry)?;
    let destroy_instance = |instance: &ash::Instance| unsafe { instance.destroy_instance(None) };

    let physical = match sorted_physical_devices(&instance) {
        Ok(list) if gpu_id < list.len() => list[gpu_id],
        Ok(list) => {
            destroy_instance(&instance);
            return Err(Error::Msg(format!(
                "vulkan: gpu_id {gpu_id} out of range ({} usable device(s))",
                list.len()
            )));
        }
        Err(e) => {
            destroy_instance(&instance);
            return Err(e);
        }
    };

    let props = unsafe { instance.get_physical_device_properties(physical) };
    let name = props
        .device_name_as_c_str()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let compute_only = queues.iter().position(|q| {
        q.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    });
    let any_compute = queues
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
    let graphics = queues.iter().position(|q| {
        q.queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    });
    let want_graphics = std::env::var("CANDLE_VULKAN_QUEUE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "graphics" || v == "3d" || v == "universal"
        })
        .unwrap_or(false);
    let picked = if want_graphics {
        graphics.or(compute_only).or(any_compute)
    } else {
        compute_only.or(any_compute)
    };
    let queue_family = match picked {
        Some(i) => i as u32,
        None => {
            destroy_instance(&instance);
            return Err(Error::Msg(format!(
                "vulkan: device {name} has no compute queue"
            )));
        }
    };

    let memory_budget_ext = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map(|exts| {
            exts.iter().any(|e| {
                e.extension_name_as_c_str()
                    .map(|n| n == ash::ext::memory_budget::NAME)
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    let queue_priorities = [1.0f32];
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities)];
    let mut ext_names = Vec::new();
    if memory_budget_ext {
        ext_names.push(ash::ext::memory_budget::NAME.as_ptr());
    }
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&ext_names);
    let device = match unsafe { instance.create_device(physical, &device_create_info, None) } {
        Ok(d) => d,
        Err(e) => {
            destroy_instance(&instance);
            return Err(Error::Msg(format!(
                "vulkan: create_device on {name} failed: {e:?}"
            )));
        }
    };
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let memory_types = memory_type_order(&memory_properties);
    if memory_types.is_empty() {
        unsafe {
            device.destroy_device(None);
        }
        destroy_instance(&instance);
        return Err(Error::Msg(format!(
            "vulkan: {name} exposes no HOST_VISIBLE|HOST_COHERENT memory type"
        )));
    }

    let exec = match shaders::ExecState::new(&device, queue_family) {
        Ok(e) => e,
        Err(e) => {
            unsafe {
                device.destroy_device(None);
            }
            destroy_instance(&instance);
            return Err(e);
        }
    };

    let limits = Limits {
        max_storage_buffer_range: props.limits.max_storage_buffer_range as u64,
        max_workgroup_count: props.limits.max_compute_work_group_count,
        max_push_constants_size: props.limits.max_push_constants_size,
    };

    let unified = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        || memory_properties.memory_types[memory_types[0] as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let default_native = unified && props.device_type != vk::PhysicalDeviceType::CPU;
    let heap = memory_properties.memory_types[memory_types[0] as usize].heap_index as usize;
    let heap_size = memory_properties.memory_heaps[heap].size as usize;
    let pool_limit = std::env::var("CANDLE_VULKAN_POOL_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|mb| mb << 20)
        .unwrap_or_else(|| (heap_size / 8).min(1 << 30));
    let slab_mode = match std::env::var("CANDLE_VULKAN_SLAB") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => 1,
            "0" | "false" | "off" | "no" => 2,
            _ => 0,
        },
        Err(_) => 0,
    };
    let max_allocations = (props.limits.max_memory_allocation_count as usize).max(64);

    use rand::SeedableRng;
    Ok(VulkanContext {
        entry,
        instance,
        physical,
        device,
        queue,
        exec: Mutex::new(exec),
        kernels: Mutex::new(HashMap::new()),
        memory_properties,
        memory_types,
        memory_budget_ext,
        allocated_bytes: AtomicUsize::new(0),
        live_allocations: AtomicUsize::new(0),
        pool: Mutex::new(Pool::default()),
        pool_limit,
        slabs: Mutex::new(HashMap::new()),
        max_allocations,
        slab_mode,
        limits,
        name,
        device_type: props.device_type,
        rng: Mutex::new((
            DEFAULT_SEED,
            rand::rngs::StdRng::seed_from_u64(DEFAULT_SEED),
        )),
        native_ok: AtomicBool::new(true),
        default_native,
        gpu: Default::default(),
    })
}

impl VulkanDevice {
    /// Returns the logical device for the `gpu_id`-th usable physical device,
    /// creating it on first use.
    pub fn new(gpu_id: usize) -> Result<Self> {
        let mut reg = registry()
            .lock()
            .map_err(|_| Error::Msg("vulkan: device registry poisoned".into()))?;
        if let Some(inner) = reg.get(&gpu_id).and_then(|w| w.upgrade()) {
            return Ok(Self { gpu_id, inner });
        }
        let inner = Arc::new(init_vulkan(gpu_id)?);
        reg.insert(gpu_id, Arc::downgrade(&inner));
        Ok(Self { gpu_id, inner })
    }

    pub fn new_with_stream(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.gpu_id)
    }

    pub fn ordinal(&self) -> usize {
        self.gpu_id
    }

    /// The physical device name reported by the driver.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// `"discrete"`, `"integrated"`, `"virtual"`, `"cpu"` or `"other"`.
    pub fn device_type(&self) -> &'static str {
        match self.inner.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => "cpu",
            _ => "other",
        }
    }

    /// `(free, total)` bytes of the memory heap that backs this device's
    /// tensors. Uses `VK_EXT_memory_budget` when the driver exposes it,
    /// otherwise the heap size minus what this process allocated.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let ctx = &self.inner;
        if qgpu::enabled(self) {
            let heaps = qgpu::working_heaps(ctx);
            if !heaps.is_empty() {
                let budgets = qgpu::heap_budgets(ctx);
                let (mut free, mut total) = (0usize, 0usize);
                for h in heaps {
                    let heap = ctx.memory_properties.memory_heaps[h];
                    let size = heap.size as usize;
                    total += size;
                    free += match &budgets {
                        Some((b, u)) => b[h].saturating_sub(u[h]) as usize,
                        None => {
                            let own = if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                                ctx.gpu.local_bytes.load(Ordering::Relaxed)
                            } else {
                                ctx.gpu.shared_bytes.load(Ordering::Relaxed)
                                    + ctx.allocated_bytes.load(Ordering::Relaxed)
                            };
                            size.saturating_sub(own)
                        }
                    };
                }
                return Ok((free, total));
            }
        }
        let mt = ctx.memory_types[0] as usize;
        let heap = ctx.memory_properties.memory_types[mt].heap_index as usize;
        let total = ctx.memory_properties.memory_heaps[heap].size as usize;
        if ctx.memory_budget_ext {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                ctx.instance
                    .get_physical_device_memory_properties2(ctx.physical, &mut props2)
            };
            let b = budget.heap_budget[heap] as usize;
            let u = budget.heap_usage[heap] as usize;
            if b > 0 {
                return Ok((b.saturating_sub(u), total));
            }
        }
        let used = ctx.allocated_bytes.load(Ordering::Relaxed);
        Ok((total.saturating_sub(used), total))
    }

    /// Bytes of device memory held by this device, including released
    /// buffers kept in the reuse pool.
    pub fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Bytes held by released buffers waiting for reuse.
    pub fn pooled_bytes(&self) -> usize {
        self.inner.pool.lock().map(|p| p.bytes).unwrap_or(0)
    }

    /// Returns the buffers kept for reuse to the driver.
    pub fn trim_memory_pool(&self) -> Result<()> {
        self.inner.trim_pool();
        self.inner.gpu.release_scratch(&self.inner.device);
        Ok(())
    }

    /// Whether the GPU kernels are in use (see [`shaders::native_enabled`]).
    pub fn native_kernels_enabled(&self) -> bool {
        shaders::native_enabled(self)
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    pub(crate) fn ctx(&self) -> &VulkanContext {
        &self.inner
    }

    pub(crate) fn limits(&self) -> Limits {
        self.inner.limits
    }

    /// Allocates a mapped host visible buffer of `numel` elements of `dtype`.
    pub(crate) fn alloc_buffer(&self, numel: usize, dtype: DType) -> Result<VulkanStorage> {
        let bytes = numel
            .checked_mul(elem_size(dtype))
            .ok_or_else(|| Error::Msg("vulkan: overflow in storage size".into()))?;
        let ctx = &self.inner;
        let backing = bucket_size(bytes);
        let reused = if backing <= POOL_MAX_BUFFER {
            ctx.pool.lock().ok().and_then(|mut pool| {
                let b = pool.free.get_mut(&backing).and_then(|v| v.pop())?;
                pool.bytes -= b.alloc_size;
                Some(b)
            })
        } else {
            None
        };
        if let Some(b) = reused {
            return Ok(VulkanStorage {
                buffer: b.buffer,
                memory: b.memory,
                mapped: b.mapped,
                capacity_bytes: bytes,
                alloc_size: b.alloc_size,
                bucket: backing,
                slot: b.slot,
                dtype,
                numel,
                device: self.clone(),
                mirror: Default::default(),
                rhs_uses: AtomicU32::new(0),
            });
        }
        let slab_ok = backing <= SLAB_MAX_BUCKET;
        if slab_ok && ctx.use_slabs() {
            if let Ok(s) = self.alloc_slab(backing, bytes, numel, dtype) {
                return Ok(s);
            }
        }
        match self.alloc_dedicated(backing, bytes, numel, dtype) {
            Ok(s) => Ok(s),
            Err(first) => {
                // Out of memory or out of allocation handles: give the pooled
                // buffers back to the driver and try again, then fall back to
                // a slot of an existing slab.
                ctx.trim_pool();
                match self.alloc_dedicated(backing, bytes, numel, dtype) {
                    Ok(s) => Ok(s),
                    Err(_) if slab_ok => self.alloc_slab(backing, bytes, numel, dtype).map_err(|_| first),
                    Err(_) => Err(first),
                }
            }
        }
    }

    fn create_raw_buffer(&self, size: usize) -> Result<vk::Buffer> {
        unsafe {
            self.inner.device.create_buffer(
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

    /// Allocates host visible memory of one of the preferred types allowed by
    /// `type_bits`, returning the memory and its type index.
    fn allocate_host_memory(
        &self,
        size: u64,
        type_bits: u32,
    ) -> std::result::Result<(vk::DeviceMemory, u32), Option<vk::Result>> {
        let ctx = &self.inner;
        let mut last_err = None;
        for &mt in ctx.memory_types.iter() {
            if type_bits & (1 << mt) == 0 {
                continue;
            }
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(mt);
            match unsafe { ctx.device.allocate_memory(&info, None) } {
                Ok(m) => return Ok((m, mt)),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err)
    }

    /// One buffer backed by its own device memory allocation.
    fn alloc_dedicated(
        &self,
        backing: usize,
        bytes: usize,
        numel: usize,
        dtype: DType,
    ) -> Result<VulkanStorage> {
        let ctx = &self.inner;
        let buffer = self.create_raw_buffer(backing)?;
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let memory = match self.allocate_host_memory(req.size, req.memory_type_bits) {
            Ok((m, _)) => m,
            Err(last_err) => {
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                return Err(Error::Msg(format!(
                    "vulkan: failed to allocate {bytes} bytes of host visible memory ({:?}, {} live allocations)",
                    last_err,
                    ctx.live_allocations.load(Ordering::Relaxed)
                )));
            }
        };
        if let Err(e) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                ctx.device.destroy_buffer(buffer, None);
                ctx.device.free_memory(memory, None);
            }
            return Err(Error::Msg(format!(
                "vulkan bind_buffer_memory failed: {e:?}"
            )));
        }
        let mapped = match unsafe {
            ctx.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                unsafe {
                    ctx.device.destroy_buffer(buffer, None);
                    ctx.device.free_memory(memory, None);
                }
                return Err(Error::Msg(format!("vulkan map_memory failed: {e:?}")));
            }
        };
        ctx.allocated_bytes
            .fetch_add(req.size as usize, Ordering::Relaxed);
        ctx.live_allocations.fetch_add(1, Ordering::Relaxed);
        Ok(VulkanStorage {
            buffer,
            memory,
            mapped,
            capacity_bytes: bytes,
            alloc_size: req.size as usize,
            bucket: backing,
            slot: None,
            dtype,
            numel,
            device: self.clone(),
            mirror: Default::default(),
            rhs_uses: AtomicU32::new(0),
        })
    }

    /// One buffer bound to a slot of a shared slab allocation. Slabs keep the
    /// number of device memory allocations far below `maxMemoryAllocationCount`
    /// (4096 on most Windows drivers) when many tensors are alive.
    fn alloc_slab(
        &self,
        backing: usize,
        bytes: usize,
        numel: usize,
        dtype: DType,
    ) -> Result<VulkanStorage> {
        let ctx = &self.inner;
        let buffer = self.create_raw_buffer(backing)?;
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let align = (req.alignment as usize).max(1);
        let need = req.size as usize;
        let fits = |c: &SlabChunk| {
            !c.free.is_empty()
                && c.stride >= need
                && c.stride % align == 0
                && req.memory_type_bits & (1 << c.memory_type) != 0
        };
        let mut slabs = match ctx.slabs.lock() {
            Ok(s) => s,
            Err(_) => {
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                return Err(Error::Msg("vulkan: slab state poisoned".into()));
            }
        };
        let list = slabs.entry(backing).or_default();
        let idx = match list.iter().position(|c| fits(c)) {
            Some(i) => i,
            None => {
                let stride = need.div_ceil(align) * align;
                let slots = (SLAB_CHUNK_BYTES / stride).clamp(1, SLAB_MAX_SLOTS);
                let size = stride * slots;
                let (memory, memory_type) =
                    match self.allocate_host_memory(size as u64, req.memory_type_bits) {
                        Ok(m) => m,
                        Err(last_err) => {
                            drop(slabs);
                            unsafe { ctx.device.destroy_buffer(buffer, None) };
                            return Err(Error::Msg(format!(
                                "vulkan: failed to allocate a {size} byte slab ({:?}, {} live allocations)",
                                last_err,
                                ctx.live_allocations.load(Ordering::Relaxed)
                            )));
                        }
                    };
                let mapped = match unsafe {
                    ctx.device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                } {
                    Ok(p) => p as *mut u8,
                    Err(e) => {
                        unsafe {
                            ctx.device.free_memory(memory, None);
                        }
                        drop(slabs);
                        unsafe { ctx.device.destroy_buffer(buffer, None) };
                        return Err(Error::Msg(format!("vulkan map_memory failed: {e:?}")));
                    }
                };
                ctx.allocated_bytes.fetch_add(size, Ordering::Relaxed);
                ctx.live_allocations.fetch_add(1, Ordering::Relaxed);
                list.push(SlabChunk {
                    memory,
                    memory_type,
                    mapped,
                    alloc_size: size,
                    stride,
                    free: (0..slots as u32).rev().collect(),
                    in_use: 0,
                });
                list.len() - 1
            }
        };
        let chunk = &mut list[idx];
        let slot = match chunk.free.pop() {
            Some(s) => s,
            None => {
                drop(slabs);
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                return Err(Error::Msg("vulkan: slab has no free slot".into()));
            }
        };
        let offset = slot as usize * chunk.stride;
        if let Err(e) = unsafe {
            ctx.device
                .bind_buffer_memory(buffer, chunk.memory, offset as vk::DeviceSize)
        } {
            chunk.free.push(slot);
            drop(slabs);
            unsafe { ctx.device.destroy_buffer(buffer, None) };
            return Err(Error::Msg(format!(
                "vulkan bind_buffer_memory failed: {e:?}"
            )));
        }
        chunk.in_use += 1;
        let memory = chunk.memory;
        let alloc_size = chunk.stride;
        let mapped = unsafe { chunk.mapped.add(offset) };
        drop(slabs);
        Ok(VulkanStorage {
            buffer,
            memory,
            mapped,
            capacity_bytes: bytes,
            alloc_size,
            bucket: backing,
            slot: Some(slot),
            dtype,
            numel,
            device: self.clone(),
            mirror: Default::default(),
            rhs_uses: AtomicU32::new(0),
        })
    }

    /// Number of live device memory allocations (buffers and slabs).
    pub fn live_allocations(&self) -> usize {
        self.inner.live_allocations.load(Ordering::Relaxed)
    }

    /// Uploads `data` (raw bytes of `numel` elements of `dtype`) to a new buffer.
    pub(crate) fn upload_bytes(
        &self,
        data: &[u8],
        numel: usize,
        dtype: DType,
    ) -> Result<VulkanStorage> {
        let storage = self.alloc_buffer(numel, dtype)?;
        if data.len() != storage.capacity_bytes {
            crate::bail!(
                "vulkan upload: {} bytes for {numel} elements of {dtype:?}",
                data.len()
            )
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), storage.mapped, data.len()) };
        Ok(storage)
    }
}

/// Bytes used to store one element: dummy sub-byte dtypes are stored the way
/// `CpuStorage` keeps them, one raw byte per stored value.
pub(crate) fn elem_size(dtype: DType) -> usize {
    match dtype.size_in_bytes() {
        0 => 1,
        s => s,
    }
}

fn cpu_bytes(cpu: &CpuStorage) -> (&[u8], usize) {
    fn b<T>(v: &[T]) -> (&[u8], usize) {
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        (bytes, v.len())
    }
    match cpu {
        CpuStorage::U8(v) => b(v),
        CpuStorage::U32(v) => b(v),
        CpuStorage::I16(v) => b(v),
        CpuStorage::I32(v) => b(v),
        CpuStorage::I64(v) => b(v),
        CpuStorage::BF16(v) => b(v),
        CpuStorage::F16(v) => b(v),
        CpuStorage::F32(v) => b(v),
        CpuStorage::F64(v) => b(v),
        CpuStorage::F8E4M3(v) => b(v),
        CpuStorage::F6E2M3(v) => b(v),
        CpuStorage::F6E3M2(v) => b(v),
        CpuStorage::F4(v) => b(v),
        CpuStorage::F8E8M0(v) => b(v),
    }
}

/// A Vulkan storage: a `VkBuffer` backed by host visible memory that stays
/// mapped at `mapped` for the lifetime of the storage.
pub struct VulkanStorage {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut u8,
    /// Logical size in bytes (`numel` elements of `dtype`).
    pub capacity_bytes: usize,
    alloc_size: usize,
    bucket: usize,
    slot: Option<u32>,
    pub dtype: DType,
    pub numel: usize,
    pub device: VulkanDevice,
    pub(crate) mirror: Mutex<Option<Arc<qgpu::GpuBuf>>>,
    pub(crate) rhs_uses: AtomicU32,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage")
            .field("capacity_bytes", &self.capacity_bytes)
            .field("dtype", &self.dtype)
            .field("numel", &self.numel)
            .field("device", &self.device)
            .finish()
    }
}

unsafe impl Send for VulkanStorage {}
unsafe impl Sync for VulkanStorage {}

impl Drop for VulkanStorage {
    fn drop(&mut self) {
        let ctx = &self.device.inner;
        let b = PooledBuffer {
            buffer: self.buffer,
            memory: self.memory,
            mapped: self.mapped,
            alloc_size: self.alloc_size,
            bucket: self.bucket,
            slot: self.slot,
        };
        if self.bucket <= POOL_MAX_BUFFER {
            if let Ok(mut pool) = ctx.pool.lock() {
                if pool.bytes + b.alloc_size <= ctx.pool_limit {
                    pool.bytes += b.alloc_size;
                    pool.free.entry(self.bucket).or_default().push(b);
                    return;
                }
            }
        }
        unsafe { ctx.release(b) };
    }
}

/// Builds a `Vec` header over memory that is not owned by the Rust allocator.
/// The result must never be dropped, grown or replaced; callers wrap it in
/// `ManuallyDrop` and only hand out `&`/`&mut` borrows of the `CpuStorage`
/// built from it to operations that read or write elements in place.
unsafe fn view_vec<T>(ptr: *mut u8, len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    debug_assert_eq!(ptr as usize % std::mem::align_of::<T>(), 0);
    unsafe { Vec::from_raw_parts(ptr as *mut T, len, len) }
}

impl VulkanStorage {
    pub fn transfer_to_device(&self, dst: &VulkanDevice) -> Result<Self> {
        dst.upload_bytes(self.as_bytes(), self.numel, self.dtype)
    }

    pub fn from_vec<T: crate::WithDType>(data: Vec<T>, device: &VulkanDevice) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(&data[..]))
        };
        device.upload_bytes(bytes, data.len(), T::DTYPE)
    }

    /// The raw bytes of the buffer.
    pub fn as_bytes(&self) -> &[u8] {
        if self.capacity_bytes == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.mapped, self.capacity_bytes) }
    }

    pub(crate) fn invalidate_mirror(&mut self) {
        if let Ok(m) = self.mirror.get_mut() {
            *m = None;
        }
        *self.rhs_uses.get_mut() = 0;
    }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.invalidate_mirror();
        if self.capacity_bytes == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.mapped, self.capacity_bytes) }
    }

    /// The elements of the buffer as a slice of `T`.
    pub fn as_slice<T: crate::WithDType>(&self) -> Result<&[T]> {
        if T::DTYPE != self.dtype {
            crate::bail!(
                "vulkan as_slice: expected {:?}, got {:?}",
                T::DTYPE,
                self.dtype
            )
        }
        if self.numel == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { std::slice::from_raw_parts(self.mapped as *const T, self.numel) })
    }

    pub(crate) fn as_mut_slice<T: crate::WithDType>(&mut self) -> Result<&mut [T]> {
        self.invalidate_mirror();
        if T::DTYPE != self.dtype {
            crate::bail!(
                "vulkan as_mut_slice: expected {:?}, got {:?}",
                T::DTYPE,
                self.dtype
            )
        }
        if self.numel == 0 {
            return Ok(&mut []);
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(self.mapped as *mut T, self.numel) })
    }

    /// A `CpuStorage` aliasing the mapped memory, for read-only CPU fallbacks.
    pub(crate) fn host_view(&self) -> ManuallyDrop<CpuStorage> {
        let (p, n) = (self.mapped, self.numel);
        let s = unsafe {
            match self.dtype {
                DType::U8 => CpuStorage::U8(view_vec::<u8>(p, n)),
                DType::U32 => CpuStorage::U32(view_vec::<u32>(p, n)),
                DType::I16 => CpuStorage::I16(view_vec::<i16>(p, n)),
                DType::I32 => CpuStorage::I32(view_vec::<i32>(p, n)),
                DType::I64 => CpuStorage::I64(view_vec::<i64>(p, n)),
                DType::BF16 => CpuStorage::BF16(view_vec::<bf16>(p, n)),
                DType::F16 => CpuStorage::F16(view_vec::<f16>(p, n)),
                DType::F32 => CpuStorage::F32(view_vec::<f32>(p, n)),
                DType::F64 => CpuStorage::F64(view_vec::<f64>(p, n)),
                DType::F8E4M3 => CpuStorage::F8E4M3(view_vec::<F8E4M3>(p, n)),
                DType::F6E2M3 => CpuStorage::F6E2M3(view_vec::<u8>(p, n)),
                DType::F6E3M2 => CpuStorage::F6E3M2(view_vec::<u8>(p, n)),
                DType::F4 => CpuStorage::F4(view_vec::<u8>(p, n)),
                DType::F8E8M0 => CpuStorage::F8E8M0(view_vec::<u8>(p, n)),
            }
        };
        ManuallyDrop::new(s)
    }

    /// Same as [`Self::host_view`] for the CPU operations that update the
    /// elements in place (copies, scatter, const_set).
    fn host_view_mut(&mut self) -> ManuallyDrop<CpuStorage> {
        self.invalidate_mirror();
        self.host_view()
    }

    /// Overwrites the buffer with `cpu`, reallocating when the element count or
    /// the dtype changed.
    pub fn overwrite_from_cpu(&mut self, cpu: &CpuStorage) -> Result<()> {
        let (bytes, numel) = cpu_bytes(cpu);
        if cpu.dtype() == self.dtype && numel == self.numel {
            self.as_bytes_mut().copy_from_slice(bytes);
            Ok(())
        } else {
            *self = self.device.upload_bytes(bytes, numel, cpu.dtype())?;
            Ok(())
        }
    }

    fn upload(&self, cpu: CpuStorage) -> Result<Self> {
        self.device.storage_from_cpu_storage(&cpu)
    }

    /// Runs `f` on a `CpuStorage` that reads this buffer's memory in place.
    pub fn with_cpu_view<R>(&self, f: impl FnOnce(&CpuStorage) -> Result<R>) -> Result<R> {
        let view = self.host_view();
        f(&view)
    }

    /// A contiguous copy of `layout` when it is not contiguous already.
    pub fn contiguous(&self, layout: &Layout) -> Result<Option<(Self, Layout)>> {
        if layout.is_contiguous() {
            return Ok(None);
        }
        let mut dst = self
            .device
            .alloc_buffer(layout.shape().elem_count(), self.dtype)?;
        self.copy_strided_src(&mut dst, 0, layout)?;
        Ok(Some((dst, Layout::contiguous(layout.shape()))))
    }

    /// `1 / (1 + exp(-x))` with the GPU kernel, `None` when the kernel does
    /// not apply.
    pub fn sigmoid(&self, layout: &Layout) -> Result<Option<Self>> {
        shaders::unary(self, "sigmoid", layout)
    }

    /// Softmax over the last dimension with the GPU kernel, `None` when the
    /// kernel does not apply (dtype, layout, size, kernels disabled).
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<Option<Self>> {
        shaders::softmax_last_dim(self, layout)
    }

    /// RMS norm over the last dimension with the GPU kernel, `None` when the
    /// kernel does not apply.
    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_l: &Layout,
        eps: f32,
    ) -> Result<Option<Self>> {
        shaders::rms_norm(self, layout, alpha, alpha_l, eps)
    }

    /// Layer norm over the last dimension with the GPU kernel, `None` when the
    /// kernel does not apply.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_l: &Layout,
        beta: &Self,
        beta_l: &Layout,
        eps: f32,
    ) -> Result<Option<Self>> {
        shaders::layer_norm(self, layout, alpha, alpha_l, beta, beta_l, eps)
    }

    /// Runs a two-input CPU op on f32 copies of bf16 inputs (the CPU backend
    /// has no bf16 gemm) and converts the result back.
    fn bf16_via_f32(
        &self,
        l: &Layout,
        other: &Self,
        other_l: &Layout,
        f: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<CpuStorage>,
    ) -> Result<Self> {
        let a = self.host_view().to_dtype(l, DType::F32)?;
        let b = other.host_view().to_dtype(other_l, DType::F32)?;
        let out = f(
            &a,
            &Layout::contiguous(l.shape()),
            &b,
            &Layout::contiguous(other_l.shape()),
        )?;
        let n = out.as_slice::<f32>()?.len();
        self.upload(out.to_dtype(&Layout::contiguous(n), self.dtype)?)
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        self.device
            .upload_bytes(self.as_bytes(), self.numel, self.dtype)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        self.invalidate_mirror();
        if shaders::const_set(self, s, layout)? {
            return Ok(());
        }
        let mut view = self.host_view_mut();
        view.const_set(s, layout)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let view = self.host_view();
        Ok((*view).clone())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if let Some(out) = shaders::affine(self, layout, mul, add)? {
            return Ok(out);
        }
        let view = self.host_view();
        self.upload(view.affine(layout, mul, add)?)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if let Some(out) = shaders::powf(self, layout, e)? {
            return Ok(out);
        }
        let view = self.host_view();
        self.upload(view.powf(layout, e)?)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if let Some(out) = shaders::elu(self, layout, alpha)? {
            return Ok(out);
        }
        let view = self.host_view();
        self.upload(view.elu(layout, alpha)?)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        if let Some(out) = shaders::reduce(self, op, layout, dims)? {
            return Ok(out);
        }
        let view = self.host_view();
        self.upload(view.reduce_op(op, layout, dims)?)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if let Some(out) = shaders::cmp(self, op, rhs, lhs_l, rhs_l)? {
            return Ok(out);
        }
        let lhs = self.host_view();
        let rhs = rhs.host_view();
        self.upload(lhs.cmp(op, &rhs, lhs_l, rhs_l)?)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if let Some(out) = shaders::to_dtype(self, layout, dtype)? {
            return Ok(out);
        }
        let view = self.host_view();
        self.upload(view.to_dtype(layout, dtype)?)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if let Some(out) = shaders::unary(self, B::NAME, layout)? {
            return Ok(out);
        }
        let view = self.host_view();
        self.upload(view.unary_impl::<B>(layout)?)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if let Some(out) = shaders::binary(self, B::NAME, rhs, lhs_l, rhs_l)? {
            return Ok(out);
        }
        let lhs = self.host_view();
        let rhs = rhs.host_view();
        self.upload(lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)?)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if let Some(out) = shaders::where_cond(self, layout, t, t_l, f, f_l)? {
            return Ok(out);
        }
        let cond = self.host_view();
        let t = t.host_view();
        let f = f.host_view();
        self.upload(cond.where_cond(layout, &t, t_l, &f, f_l)?)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if self.dtype == DType::BF16 && kernel.dtype == DType::BF16 {
            return self.bf16_via_f32(l, kernel, kernel_l, |a, al, b, bl| {
                a.conv1d(al, b, bl, params)
            });
        }
        let inp = self.host_view();
        let kernel = kernel.host_view();
        self.upload(inp.conv1d(l, &kernel, kernel_l, params)?)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        if self.dtype == DType::BF16 && kernel.dtype == DType::BF16 {
            return self.bf16_via_f32(l, kernel, kernel_l, |a, al, b, bl| {
                a.conv_transpose1d(al, b, bl, params)
            });
        }
        let inp = self.host_view();
        let kernel = kernel.host_view();
        self.upload(inp.conv_transpose1d(l, &kernel, kernel_l, params)?)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if self.dtype == DType::BF16 && kernel.dtype == DType::BF16 {
            return self.bf16_via_f32(l, kernel, kernel_l, |a, al, b, bl| {
                a.conv2d(al, b, bl, params)
            });
        }
        let inp = self.host_view();
        let kernel = kernel.host_view();
        self.upload(inp.conv2d(l, &kernel, kernel_l, params)?)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        if self.dtype == DType::BF16 && kernel.dtype == DType::BF16 {
            return self.bf16_via_f32(l, kernel, kernel_l, |a, al, b, bl| {
                a.conv_transpose2d(al, b, bl, params)
            });
        }
        let inp = self.host_view();
        let kernel = kernel.host_view();
        self.upload(inp.conv_transpose2d(l, &kernel, kernel_l, params)?)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if let Some(out) = shaders::index_select(self, ids, l, ids_l, dim)? {
            return Ok(out);
        }
        let src = self.host_view();
        let ids = ids.host_view();
        self.upload(src.index_select(&ids, l, ids_l, dim)?)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src = self.host_view();
        let ids = ids.host_view();
        self.upload(src.gather(l, &ids, ids_l, dim)?)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let ids = ids.host_view();
        let src = src.host_view();
        let mut dst = self.host_view_mut();
        dst.scatter_set(l, &ids, ids_l, &src, src_l, dim)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let ids = ids.host_view();
        let src = src.host_view();
        let mut dst = self.host_view_mut();
        dst.scatter_add_set(l, &ids, ids_l, &src, src_l, dim)
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let tgt = self.host_view();
        let ids = ids.host_view();
        let src = src.host_view();
        self.upload(tgt.index_add(l, &ids, ids_l, &src, src_l, dim)?)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if let Some(out) = shaders::matmul(self, rhs, bmnk, lhs_l, rhs_l)? {
            return Ok(out);
        }
        if let Some(out) = qgpu::dense_matmul(self, rhs, bmnk, lhs_l, rhs_l)? {
            return Ok(out);
        }
        if let Some(out) = qgpu::act_matmul(self, rhs, bmnk, lhs_l, rhs_l)? {
            return Ok(out);
        }
        if self.dtype == DType::BF16 && rhs.dtype == DType::BF16 {
            let (b, m, n, _) = bmnk;
            let l32 = self.to_dtype(lhs_l, DType::F32)?;
            let r32 = rhs.to_dtype(rhs_l, DType::F32)?;
            let out = l32.matmul(
                &r32,
                bmnk,
                &Layout::contiguous(lhs_l.shape()),
                &Layout::contiguous(rhs_l.shape()),
            )?;
            return out.to_dtype(&Layout::contiguous(b * m * n), self.dtype);
        }
        let lhs = self.host_view();
        let rhs = rhs.host_view();
        self.upload(lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)?)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        dst.invalidate_mirror();
        if shaders::copy_strided(self, dst, dst_offset, src_l)? {
            return Ok(());
        }
        let src = self.host_view();
        let mut dst = dst.host_view_mut();
        src.copy_strided_src(&mut dst, dst_offset, src_l)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        dst.invalidate_mirror();
        if shaders::copy2d(self, dst, d1, d2, src_s, dst_s, src_o, dst_o)? {
            return Ok(());
        }
        let src = self.host_view();
        let mut dst = dst.host_view_mut();
        src.copy2d(&mut dst, d1, d2, src_s, dst_s, src_o, dst_o)
    }

    fn avg_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let view = self.host_view();
        self.upload(view.avg_pool2d(layout, k, s)?)
    }

    fn max_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let view = self.host_view();
        self.upload(view.max_pool2d(layout, k, s)?)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        let view = self.host_view();
        self.upload(view.upsample_nearest1d(layout, sz)?)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        let view = self.host_view();
        self.upload(view.upsample_nearest2d(layout, h, w)?)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let view = self.host_view();
        self.upload(view.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)?)
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        use rand::SeedableRng;
        let mut rng = self
            .inner
            .rng
            .lock()
            .map_err(|_| Error::Msg("vulkan: rng lock poisoned".into()))?;
        *rng = (seed, rand::rngs::StdRng::seed_from_u64(seed));
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let rng = self
            .inner
            .rng
            .lock()
            .map_err(|_| Error::Msg("vulkan: rng lock poisoned".into()))?;
        Ok(rng.0)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.gpu_id,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.gpu_id == other.gpu_id && Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype.size_in_bytes() == 0 || dtype == DType::F8E8M0 {
            return Err(Error::UnsupportedDTypeForOp(dtype, "zeros").bt());
        }
        let mut storage = self.alloc_buffer(shape.elem_count(), dtype)?;
        storage.as_bytes_mut().fill(0);
        Ok(storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype.size_in_bytes() == 0 || dtype == DType::F8E8M0 {
            return Err(Error::UnsupportedDTypeForOp(dtype, "alloc_uninit").bt());
        }
        self.alloc_buffer(shape.elem_count(), dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let bytes = unsafe {
            std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
        };
        self.upload_bytes(bytes, s.len(), T::DTYPE)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        let (bytes, numel) = cpu_bytes(cpu);
        self.upload_bytes(bytes, numel, cpu.dtype())
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self::Storage> {
        use rand::Rng;
        let elem_count = shape.elem_count();
        let mut guard = self
            .inner
            .rng
            .lock()
            .map_err(|_| Error::Msg("vulkan: rng lock poisoned".into()))?;
        let rng = &mut guard.1;
        let cpu = match dtype {
            DType::BF16 => {
                let u = rand::distr::Uniform::new(bf16::from_f64(min), bf16::from_f64(max))
                    .map_err(Error::wrap)?;
                CpuStorage::BF16((0..elem_count).map(|_| rng.sample::<bf16, _>(u)).collect())
            }
            DType::F16 => {
                let u = rand::distr::Uniform::new(f16::from_f64(min), f16::from_f64(max))
                    .map_err(Error::wrap)?;
                CpuStorage::F16((0..elem_count).map(|_| rng.sample::<f16, _>(u)).collect())
            }
            DType::F8E4M3 => {
                let u = rand::distr::Uniform::new(F8E4M3::from_f64(min), F8E4M3::from_f64(max))
                    .map_err(Error::wrap)?;
                CpuStorage::F8E4M3(
                    (0..elem_count)
                        .map(|_| rng.sample::<F8E4M3, _>(u))
                        .collect(),
                )
            }
            DType::F32 => {
                let u = rand::distr::Uniform::new(min as f32, max as f32).map_err(Error::wrap)?;
                CpuStorage::F32((0..elem_count).map(|_| rng.sample::<f32, _>(u)).collect())
            }
            DType::F64 => {
                let u = rand::distr::Uniform::new(min, max).map_err(Error::wrap)?;
                CpuStorage::F64((0..elem_count).map(|_| rng.sample::<f64, _>(u)).collect())
            }
            _ => return Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt()),
        };
        drop(guard);
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        use rand::distr::Distribution;
        let elem_count = shape.elem_count();
        let mut guard = self
            .inner
            .rng
            .lock()
            .map_err(|_| Error::Msg("vulkan: rng lock poisoned".into()))?;
        let rng = &mut guard.1;
        let cpu = match dtype {
            DType::BF16 => {
                let n = rand_distr::Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                    .map_err(Error::wrap)?;
                CpuStorage::BF16((0..elem_count).map(|_| n.sample(rng)).collect())
            }
            DType::F16 => {
                let n = rand_distr::Normal::new(f16::from_f64(mean), f16::from_f64(std))
                    .map_err(Error::wrap)?;
                CpuStorage::F16((0..elem_count).map(|_| n.sample(rng)).collect())
            }
            DType::F8E4M3 => {
                let n = rand_distr::Normal::new(F8E4M3::from_f64(mean), F8E4M3::from_f64(std))
                    .map_err(Error::wrap)?;
                CpuStorage::F8E4M3((0..elem_count).map(|_| n.sample(rng)).collect())
            }
            DType::F32 => {
                let n = rand_distr::Normal::new(mean as f32, std as f32).map_err(Error::wrap)?;
                CpuStorage::F32((0..elem_count).map(|_| n.sample(rng)).collect())
            }
            DType::F64 => {
                let n = rand_distr::Normal::new(mean, std).map_err(Error::wrap)?;
                CpuStorage::F64((0..elem_count).map(|_| n.sample(rng)).collect())
            }
            _ => return Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt()),
        };
        drop(guard);
        self.storage_from_cpu_storage(&cpu)
    }

    fn synchronize(&self) -> Result<()> {
        shaders::wait_idle(self)
    }
}
