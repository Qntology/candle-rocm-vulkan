use super::kernels::Module;
use super::utils::{elem_bytes, LaunchArgs};
use super::{RocmError, RocmStorage, WrapErr};
use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, DeviceLocation, Result, Shape};
use hip_runtime::blas::{GemmType, RocBlas};
use hip_runtime::device::{GpuArch, HipDevice};
use hip_runtime::error::HipError;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::module::HipModule;
use hip_sys::hip_runtime::hipFunction_t;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

struct RngState {
    seed: Option<u64>,
    rng: Option<rand::rngs::StdRng>,
}

/// Which implementation runs the GEMMs (matmul, quantized matmul) of a device.
///
/// `CANDLE_ROCM_GEMM=auto` (default) uses rocBLAS, unless the rocBLAS of the loaded ROCm has no
/// kernels for this GPU: rocBLAS then aborts the process on its first GEMM, so the HIP kernel of
/// `kernels/gemm.hip` is used instead. `rocblas` / `hip` force one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmBackend {
    RocBlas,
    Hip,
}

impl GemmBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::RocBlas => "rocblas",
            Self::Hip => "hip",
        }
    }

    fn from_u8(v: u8) -> Self {
        if v == 1 {
            Self::Hip
        } else {
            Self::RocBlas
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::RocBlas => 0,
            Self::Hip => 1,
        }
    }
}

/// One GEMM in BLAS (column-major) terms: `C = op(A) * op(B)` (alpha = 1, beta = 0), with
/// `C` of size `m x n` and `k` the inner dimension.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmCall {
    pub ty: GemmType,
    pub transa: bool,
    pub transb: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: *const c_void,
    pub lda: usize,
    pub stride_a: i64,
    pub b: *const c_void,
    pub ldb: usize,
    pub stride_b: i64,
    pub c: *mut c_void,
    pub ldc: usize,
    pub stride_c: i64,
    pub batch: usize,
    /// `false` uses the non batched rocBLAS entry point (`rocblas_sgemm`, F32 only).
    pub batched: bool,
}

/// Result of [`RocmDevice::gemm`].
#[derive(Debug)]
pub(crate) enum GemmOutcome {
    Done,
    /// rocBLAS returned this status; callers keep their own fallbacks.
    Rocblas(HipError),
    /// The HIP GEMM kernel cannot be loaded on this GPU (kernels built for other targets).
    NoKernel(String),
}

/// Runtime and build information of a ROCm device, see [`RocmDevice::runtime_info`].
#[derive(Debug, Clone)]
pub struct RocmRuntimeInfo {
    /// Version of the loaded HIP runtime, e.g. `7.2.x` (ROCm 7.2) or `7.16.0` (ROCm 10.1).
    pub hip_version: Option<hip_runtime::track::HipVersion>,
    /// ROCm track of the loaded runtime (`legacy` or `core`).
    pub runtime_track: Option<hip_runtime::track::RocmTrack>,
    /// GPU target of the device, e.g. `gfx1100`.
    pub arch: Option<String>,
    /// GPU targets the embedded kernels were compiled for.
    pub compiled_archs: &'static str,
    /// ROCm track and HIP version of the toolchain the kernels were compiled with.
    pub build_track: &'static str,
    pub build_hip_version: &'static str,
    /// Whether the embedded kernels contain code for this GPU (`None` when the GPU is unknown).
    pub kernels_match: Option<bool>,
    /// Whether the ROCm Core SDK 10.1 supports this GPU.
    pub core_track_supported: Option<bool>,
    /// `Some(false)` when the loaded rocBLAS has no kernels for this GPU.
    pub rocblas_has_kernels: Option<bool>,
    pub gemm_backend: GemmBackend,
}

impl std::fmt::Display for RocmRuntimeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opt = |v: Option<String>| v.unwrap_or_else(|| "unknown".to_string());
        write!(
            f,
            "HIP {} ({} track), GPU {}, kernels [{}] built with HIP {} ({} track), GEMM: {}",
            opt(self.hip_version.map(|v| v.to_string())),
            opt(self.runtime_track.map(|t| t.name().to_string())),
            opt(self.arch.clone()),
            self.compiled_archs,
            self.build_hip_version,
            self.build_track,
            self.gemm_backend.name()
        )
    }
}

struct DeviceInner {
    ordinal: usize,
    hip: HipDevice,
    /// Created on first use, so that a GPU that the loaded rocBLAS cannot drive never touches it.
    blas: Mutex<Option<RocBlas>>,
    modules: Vec<OnceLock<std::result::Result<HipModule, String>>>,
    stream_ordered: bool,
    trim_on_sync: bool,
    rng: Mutex<RngState>,
    arch: Option<GpuArch>,
    gemm_backend: AtomicU8,
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

fn warn_once(flag: &'static std::sync::atomic::AtomicBool, msg: impl FnOnce() -> String) {
    if !flag.swap(true, Ordering::Relaxed) {
        eprintln!("candle-rocm: {}", msg());
    }
}

fn select_gemm_backend(arch: Option<&GpuArch>) -> GemmBackend {
    use std::sync::atomic::AtomicBool;
    static WARNED: AtomicBool = AtomicBool::new(false);
    match std::env::var("CANDLE_ROCM_GEMM")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "rocblas" => return GemmBackend::RocBlas,
        "hip" | "kernel" => return GemmBackend::Hip,
        _ => {}
    }
    if let Some(arch) = arch {
        if hip_runtime::blas::has_kernels_for(&arch.name) == Some(false) {
            warn_once(&WARNED, || {
                format!(
                    "the loaded rocBLAS has no kernels for {} (looked in {}), GEMMs use the HIP kernel instead",
                    arch.name,
                    hip_runtime::blas::tensile_library_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                )
            });
            return GemmBackend::Hip;
        }
    }
    GemmBackend::RocBlas
}

impl RocmDevice {
    fn create(ordinal: usize) -> Result<Self> {
        use std::sync::atomic::AtomicBool;
        static ARCH_WARNED: AtomicBool = AtomicBool::new(false);
        let hip = HipDevice::new(ordinal).w()?;
        hip.set_current().w()?;
        let arch = hip.arch().ok();
        if let Some(arch) = &arch {
            let compiled = super::kernels::COMPILED_ARCHS;
            if compiled != "none" && !super::kernels::compiled_for(&arch.name) {
                warn_once(&ARCH_WARNED, || {
                    format!(
                        "GPU {ordinal} is {} but the ROCm kernels were compiled for [{compiled}] ({} track, HIP {}); rebuild with HIP_ARCH={}",
                        arch.full,
                        super::kernels::BUILD_TRACK,
                        super::kernels::BUILD_HIP_VERSION,
                        arch.name
                    )
                });
            }
        }
        let gemm_backend = select_gemm_backend(arch.as_ref());
        let blas = match gemm_backend {
            GemmBackend::RocBlas => Some(RocBlas::new().w()?),
            GemmBackend::Hip => None,
        };
        let stream_ordered = env_flag("CANDLE_ROCM_ASYNC_ALLOC", true)
            && hip_runtime::memory::stream_ordered_alloc_supported();
        let trim_on_sync = env_flag("CANDLE_ROCM_TRIM_ON_SYNC", true);
        if stream_ordered && env_flag("CANDLE_ROCM_POOL_RELEASE_ZERO", true) {
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
                arch,
                gemm_backend: AtomicU8::new(gemm_backend.as_u8()),
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

    /// GPU target of this device (e.g. `gfx1100`), when the runtime reports it.
    pub fn arch(&self) -> Option<&GpuArch> {
        self.inner.arch.as_ref()
    }

    /// Implementation currently used for GEMMs, see [`GemmBackend`].
    pub fn gemm_backend(&self) -> GemmBackend {
        GemmBackend::from_u8(self.inner.gemm_backend.load(Ordering::Relaxed))
    }

    /// Overrides the GEMM implementation of this device (and of all its clones).
    pub fn set_gemm_backend(&self, backend: GemmBackend) {
        self.inner.gemm_backend.store(backend.as_u8(), Ordering::Relaxed);
    }

    /// HIP runtime version and track, GPU target, kernel targets and GEMM implementation.
    pub fn runtime_info(&self) -> RocmRuntimeInfo {
        let hip_version = hip_runtime::device::runtime_hip_version().ok();
        let arch = self.inner.arch.as_ref().map(|a| a.name.clone());
        let compiled_archs = super::kernels::COMPILED_ARCHS;
        RocmRuntimeInfo {
            hip_version,
            runtime_track: hip_version.map(|v| v.track()),
            kernels_match: arch
                .as_deref()
                .filter(|_| compiled_archs != "none")
                .map(super::kernels::compiled_for),
            core_track_supported: arch.as_deref().map(hip_runtime::track::core_track_supports),
            rocblas_has_kernels: arch.as_deref().and_then(hip_runtime::blas::has_kernels_for),
            arch,
            compiled_archs,
            build_track: super::kernels::BUILD_TRACK,
            build_hip_version: super::kernels::BUILD_HIP_VERSION,
            gemm_backend: self.gemm_backend(),
        }
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

    pub fn release_cached_resources(&self) -> Result<()> {
        self.inner.hip.synchronize().w()?;
        self.trim_memory_pool()?;
        {
            let mut blas = self.inner.blas.lock().unwrap();
            if blas.is_some() {
                let fresh = RocBlas::new().w()?;
                *blas = Some(fresh);
            }
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
                let (gpu, hint) = match &self.inner.arch {
                    Some(a) => (a.full.clone(), a.name.clone()),
                    None => ("of an unknown target".to_string(), "<gfx target>".to_string()),
                };
                format!(
                    "cannot load ROCm kernel module {} ({e}); this GPU is {gpu}, the kernels were compiled for [{}] with the {} ROCm toolchain (HIP {}); rebuild with HIP_ARCH={hint}",
                    module.name(),
                    Self::compiled_archs(),
                    super::kernels::BUILD_TRACK,
                    super::kernels::BUILD_HIP_VERSION,
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
        let mut blas = self.inner.blas.lock().unwrap();
        if blas.is_none() {
            *blas = Some(RocBlas::new().w()?);
        }
        f(blas.as_ref().expect("rocBLAS handle"))
    }

    /// Runs one GEMM with the current [`GemmBackend`].
    ///
    /// The rocBLAS path is unchanged; a failing rocBLAS status is handed back so that callers keep
    /// their own fallbacks (F16 / BF16 without `gemm_ex` kernels go through F32). F32 / F64 GEMMs
    /// that rocBLAS cannot run on this GPU (`not implemented`, `excluded from build`,
    /// `arch mismatch`) switch the device to the HIP kernel.
    pub(crate) fn gemm(&self, call: &GemmCall) -> Result<GemmOutcome> {
        use std::sync::atomic::AtomicBool;
        static SWITCHED: AtomicBool = AtomicBool::new(false);
        if self.gemm_backend() == GemmBackend::Hip {
            return self.hip_gemm(call);
        }
        let status = self.with_blas(|blas| Ok(unsafe { rocblas_gemm(blas, call) }))?;
        match status {
            Ok(()) => Ok(GemmOutcome::Done),
            Err(HipError::RocblasError { code: code @ (2 | 14 | 15) })
                if matches!(call.ty, GemmType::F32 | GemmType::F64) =>
            {
                warn_once(&SWITCHED, || {
                    format!(
                        "rocBLAS cannot run {:?} GEMMs on this GPU (status {code}), using the HIP kernel instead",
                        call.ty
                    )
                });
                self.set_gemm_backend(GemmBackend::Hip);
                self.hip_gemm(call)
            }
            Err(e) => Ok(GemmOutcome::Rocblas(e)),
        }
    }

    /// GEMM with the tiled kernel of `kernels/gemm.hip` (same contract as rocBLAS).
    pub(crate) fn hip_gemm(&self, c: &GemmCall) -> Result<GemmOutcome> {
        if c.m == 0 || c.n == 0 || c.batch == 0 {
            return Ok(GemmOutcome::Done);
        }
        let name = match c.ty {
            GemmType::F32 => "gemm_f32",
            GemmType::F64 => "gemm_f64",
            GemmType::F16 => "gemm_f16",
            GemmType::BF16 => "gemm_bf16",
        };
        let func = match self.get_func(Module::Gemm, name) {
            Ok(f) => f,
            Err(e) => return Ok(GemmOutcome::NoKernel(e.to_string())),
        };
        const TILE: usize = 64;
        const MAX_GRID: usize = 65535;
        let grid = (
            c.m.div_ceil(TILE).min(MAX_GRID) as u32,
            c.n.div_ceil(TILE).min(MAX_GRID) as u32,
            c.batch.min(MAX_GRID) as u32,
        );
        let mut args = LaunchArgs::new();
        args.ptr(c.a)
            .ptr(c.b)
            .ptr(c.c as *const c_void)
            .u32(c.transa as u32)
            .u32(c.transb as u32)
            .usize(c.m)
            .usize(c.n)
            .usize(c.k)
            .usize(c.lda)
            .usize(c.ldb)
            .usize(c.ldc)
            .bits(c.stride_a as u64)
            .bits(c.stride_b as u64)
            .bits(c.stride_c as u64)
            .usize(c.batch);
        self.launch(func, grid, (256, 1, 1), 0, &mut args)?;
        Ok(GemmOutcome::Done)
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

/// The rocBLAS calls of the GEMM paths, unchanged from the single-backend version.
unsafe fn rocblas_gemm(blas: &RocBlas, c: &GemmCall) -> std::result::Result<(), HipError> {
    match c.ty {
        GemmType::F32 if !c.batched => blas.sgemm_raw(
            c.transa, c.transb, c.m, c.n, c.k, 1.0, c.a, c.lda, c.b, c.ldb, 0.0, c.c, c.ldc,
        ),
        GemmType::F32 => blas.sgemm_strided_batched_raw(
            c.transa, c.transb, c.m, c.n, c.k, 1.0, c.a, c.lda, c.stride_a, c.b, c.ldb, c.stride_b, 0.0, c.c,
            c.ldc, c.stride_c, c.batch,
        ),
        GemmType::F64 => blas.dgemm_strided_batched_raw(
            c.transa, c.transb, c.m, c.n, c.k, 1.0, c.a, c.lda, c.stride_a, c.b, c.ldb, c.stride_b, 0.0, c.c,
            c.ldc, c.stride_c, c.batch,
        ),
        GemmType::F16 | GemmType::BF16 => blas.gemm_strided_batched_ex_raw(
            c.ty, c.transa, c.transb, c.m, c.n, c.k, c.a, c.lda, c.stride_a, c.b, c.ldb, c.stride_b, c.c, c.ldc,
            c.stride_c, c.batch,
        ),
    }
}
