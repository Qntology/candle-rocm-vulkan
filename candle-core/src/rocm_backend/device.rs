use super::kernels::Module;
use super::utils::{elem_bytes, LaunchArgs};
use super::{RocmError, RocmStorage, WrapErr};
use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, DeviceLocation, Result, Shape};
use hip_runtime::blas::RocBlas;
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::module::HipModule;
use hip_sys::hip_runtime::hipFunction_t;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Snapshot of the GPU memory state, split by who owns the bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryReport {
    /// Free / total device memory as reported by the driver.
    pub free: usize,
    pub total: usize,
    /// Bytes held by live tensors (application side).
    pub live_bytes: usize,
    pub live_allocs: usize,
    pub peak_bytes: usize,
    /// Default stream-ordered pool: bytes reserved from the driver / handed out to tensors.
    pub pool_reserved: u64,
    pub pool_used: u64,
    pub pool_reserved_high: u64,
}

impl MemoryReport {
    /// Driver-side bytes in use that no live tensor accounts for (context, kernel code objects,
    /// rocBLAS workspace, pool cache, other processes ...).
    pub fn unattributed_used(&self) -> usize {
        self.total
            .saturating_sub(self.free)
            .saturating_sub(self.live_bytes)
    }
}

struct RngState {
    seed: Option<u64>,
    rng: Option<rand::rngs::StdRng>,
}

struct DeviceInner {
    ordinal: usize,
    hip: HipDevice,
    blas: Mutex<RocBlas>,
    modules: Vec<OnceLock<std::result::Result<HipModule, String>>>,
    stream_ordered: bool,
    trim_on_sync: bool,
    rng: Mutex<RngState>,
}

/// A ROCm (HIP) GPU device. Clones share the same underlying context, rocBLAS handle and
/// loaded kernels; `RocmDevice::new` returns the cached instance for an ordinal.
#[derive(Clone)]
pub struct RocmDevice {
    inner: Arc<DeviceInner>,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({})", self.inner.ordinal)
    }
}

static DEVICES: OnceLock<Mutex<HashMap<usize, RocmDevice>>> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => default,
    }
}

impl RocmDevice {
    fn create(ordinal: usize) -> Result<Self> {
        let hip = HipDevice::new(ordinal).w()?;
        hip.set_current().w()?;
        let blas = RocBlas::new().w()?;
        let stream_ordered = env_flag("CANDLE_ROCM_ASYNC_ALLOC", true)
            && hip_runtime::memory::stream_ordered_alloc_supported();
        let trim_on_sync = env_flag("CANDLE_ROCM_TRIM_ON_SYNC", true);
        if stream_ordered && env_flag("CANDLE_ROCM_POOL_RELEASE_ZERO", true) {
            // Make the pool give memory back at every synchronization instead of caching it.
            let _ = hip_runtime::memory::set_pool_release_threshold(ordinal as i32, 0);
        }
        let modules = Module::ALL.iter().map(|_| OnceLock::new()).collect();
        Ok(Self {
            inner: Arc::new(DeviceInner {
                ordinal,
                hip,
                blas: Mutex::new(blas),
                modules,
                stream_ordered,
                trim_on_sync,
                rng: Mutex::new(RngState {
                    seed: None,
                    rng: None,
                }),
            }),
        })
    }

    pub fn ordinal(&self) -> usize {
        self.inner.ordinal
    }

    pub fn hip_device(&self) -> &HipDevice {
        &self.inner.hip
    }

    pub fn set_current(&self) -> Result<()> {
        self.inner.hip.set_current().w()
    }

    /// Marketing name of the GPU, e.g. "AMD Radeon RX 7900 XTX".
    pub fn name(&self) -> Result<String> {
        self.inner.hip.name().w()
    }

    /// `(free, total)` device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.inner.hip.mem_info().w()
    }

    pub fn total_memory(&self) -> Result<usize> {
        self.inner.hip.total_memory().w()
    }

    /// GPU architectures the embedded kernels were compiled for.
    pub fn compiled_archs() -> &'static str {
        super::kernels::COMPILED_ARCHS
    }

    pub fn uses_stream_ordered_alloc(&self) -> bool {
        self.inner.stream_ordered
    }

    /// Returns the memory cached by the stream-ordered allocator back to the driver.
    pub fn trim_memory_pool(&self) -> Result<()> {
        if self.inner.stream_ordered {
            self.set_current()?;
            hip_runtime::memory::trim_default_pool(self.inner.ordinal as i32).w()?;
        }
        Ok(())
    }

    /// Memory snapshot split into application-owned and driver-side bytes.
    pub fn memory_report(&self) -> Result<MemoryReport> {
        let (free, total) = self.mem_info()?;
        let a = hip_runtime::memory::alloc_stats();
        let pool = if self.inner.stream_ordered {
            hip_runtime::memory::pool_stats(self.inner.ordinal as i32).unwrap_or_default()
        } else {
            Default::default()
        };
        Ok(MemoryReport {
            free,
            total,
            live_bytes: a.live_bytes,
            live_allocs: a.live_allocs,
            peak_bytes: a.peak_bytes,
            pool_reserved: pool.reserved_current,
            pool_used: pool.used_current,
            pool_reserved_high: pool.reserved_high,
        })
    }

    /// Largest live allocation groups `(bytes_each, count)`, biggest total first.
    pub fn live_size_histogram(&self, top: usize) -> Vec<(usize, usize)> {
        hip_runtime::memory::live_size_histogram(top)
    }

    /// Hands back everything that can be handed back without destroying the device context:
    /// synchronizes, trims the stream-ordered pool and re-creates the rocBLAS handle (its
    /// internal workspace only grows and is otherwise kept for the lifetime of the handle).
    /// Must not be called while another thread is running GEMMs it still needs the result of.
    pub fn release_cached_resources(&self) -> Result<()> {
        self.inner.hip.synchronize().w()?;
        self.trim_memory_pool()?;
        {
            let mut blas = self.inner.blas.lock().unwrap();
            let fresh = RocBlas::new().w()?;
            *blas = fresh; // drops (destroys) the old handle and its workspace
        }
        self.trim_memory_pool()?;
        self.inner.hip.synchronize().w()
    }

    fn module(&self, module: Module) -> Result<&HipModule> {
        let cell = &self.inner.modules[module.index()];
        let loaded = cell.get_or_init(|| {
            if module.image().is_empty() {
                return Err(format!(
                    "the ROCm kernels were not compiled into this build (module {})",
                    module.name()
                ));
            }
            if let Err(e) = self.set_current() {
                return Err(e.to_string());
            }
            HipModule::load_data(module.image()).map_err(|e| {
                format!(
                    "cannot load ROCm kernel module {} ({e}); it was compiled for [{}], rebuild with HIP_ARCH set to this GPU's gfx target",
                    module.name(),
                    Self::compiled_archs()
                )
            })
        });
        match loaded {
            Ok(m) => Ok(m),
            Err(msg) => Err(RocmError::Message(msg.clone()).into()),
        }
    }

    pub(crate) fn get_func(&self, module: Module, name: &str) -> Result<hipFunction_t> {
        let m = self.module(module)?;
        m.get_function(name).w()
    }

    pub(crate) fn launch(
        &self,
        func: hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        args: &mut LaunchArgs,
    ) -> Result<()> {
        self.set_current()?;
        let mut ptrs = args.pointers();
        unsafe { HipModule::launch(func, grid, block, shared_mem, &mut ptrs) }.w()
    }

    pub(crate) fn launch_1d(
        &self,
        module: Module,
        name: &str,
        n: usize,
        args: &mut LaunchArgs,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let func = self.get_func(module, name)?;
        let block = 256u32;
        let grid = super::utils::grid_1d(n, block);
        self.launch(func, (grid, 1, 1), (block, 1, 1), 0, args)
    }

    pub(crate) fn alloc(&self, bytes: usize) -> Result<DeviceBuffer<u8>> {
        self.set_current()?;
        if self.inner.stream_ordered {
            DeviceBuffer::<u8>::alloc_async(bytes).w()
        } else {
            DeviceBuffer::<u8>::alloc(bytes).w()
        }
    }

    pub(crate) fn alloc_zeros(&self, bytes: usize) -> Result<DeviceBuffer<u8>> {
        let buf = self.alloc(bytes)?;
        unsafe { hip_runtime::memory::memset(buf.as_void_ptr(), 0, bytes) }.w()?;
        Ok(buf)
    }

    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer<u8>> {
        let mut buf = self.alloc(bytes.len())?;
        buf.copy_from_host(bytes).w()?;
        Ok(buf)
    }

    pub(crate) fn with_blas<R>(&self, f: impl FnOnce(&RocBlas) -> Result<R>) -> Result<R> {
        self.set_current()?;
        let blas = self.inner.blas.lock().unwrap();
        f(&blas)
    }

    fn seeded_rand(
        &self,
        shape: &Shape,
        dtype: DType,
        a: f64,
        b: f64,
        normal: bool,
    ) -> Result<Option<CpuStorage>> {
        use rand::Rng;
        let mut state = self.inner.rng.lock().unwrap();
        let rng = match state.rng.as_mut() {
            None => return Ok(None),
            Some(rng) => rng,
        };
        let n = shape.elem_count();
        let mut values = Vec::with_capacity(n);
        if normal {
            let dist = rand_distr::Normal::new(a, b).map_err(crate::Error::wrap)?;
            for _ in 0..n {
                values.push(rng.sample(dist));
            }
        } else {
            let dist = rand::distr::Uniform::new(a, b).map_err(crate::Error::wrap)?;
            for _ in 0..n {
                values.push(rng.sample(dist));
            }
        }
        let storage = match dtype {
            DType::F64 => CpuStorage::F64(values),
            DType::F32 => CpuStorage::F32(values.into_iter().map(|v| v as f32).collect()),
            DType::F16 => CpuStorage::F16(values.into_iter().map(half::f16::from_f64).collect()),
            DType::BF16 => CpuStorage::BF16(values.into_iter().map(half::bf16::from_f64).collect()),
            DType::F8E4M3 => {
                CpuStorage::F8E4M3(values.into_iter().map(float8::F8E4M3::from_f64).collect())
            }
            _ => {
                return Err(crate::Error::UnsupportedDTypeForOp(
                    dtype,
                    if normal { "rand_normal" } else { "rand_uniform" },
                )
                .bt())
            }
        };
        Ok(Some(storage))
    }
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let cache = DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().unwrap();
        if let Some(dev) = cache.get(&ordinal) {
            return Ok(dev.clone());
        }
        let dev = Self::create(ordinal)?;
        cache.insert(ordinal, dev.clone());
        Ok(dev)
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Rocm {
            gpu_id: self.inner.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.inner.ordinal == rhs.inner.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if matches!(dtype, DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0) {
            return Err(crate::Error::UnsupportedDTypeForOp(dtype, "zeros").bt());
        }
        let bytes = shape.elem_count() * elem_bytes(dtype);
        let buf = self.alloc_zeros(bytes)?;
        Ok(RocmStorage::new(buf, dtype, self.clone()))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let bytes = shape.elem_count() * elem_bytes(dtype);
        let buf = self.alloc(bytes)?;
        Ok(RocmStorage::new(buf, dtype, self.clone()))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        let buf = self.upload(bytes)?;
        Ok(RocmStorage::new(buf, T::DTYPE, self.clone()))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        RocmStorage::from_cpu(self, storage)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        RocmStorage::from_cpu(self, &storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        let cpu = match self.seeded_rand(shape, dtype, lo, up, false)? {
            Some(cpu) => cpu,
            None => CpuDevice.rand_uniform(shape, dtype, lo, up)?,
        };
        RocmStorage::from_cpu(self, &cpu)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let cpu = match self.seeded_rand(shape, dtype, mean, std, true)? {
            Some(cpu) => cpu,
            None => CpuDevice.rand_normal(shape, dtype, mean, std)?,
        };
        RocmStorage::from_cpu(self, &cpu)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        use rand::SeedableRng;
        let mut state = self.inner.rng.lock().unwrap();
        state.seed = Some(seed);
        state.rng = Some(rand::rngs::StdRng::seed_from_u64(seed));
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        match self.inner.rng.lock().unwrap().seed {
            Some(seed) => Ok(seed),
            None => crate::bail!("the ROCm rng has not been seeded, call set_seed first"),
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.hip.synchronize().w()?;
        if self.inner.trim_on_sync {
            self.trim_memory_pool()?;
        }
        Ok(())
    }
}
