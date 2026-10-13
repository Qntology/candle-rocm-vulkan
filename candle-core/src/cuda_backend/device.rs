use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::{result as drv, sys, CudaContext, CudaFunction, CudaSlice, DevicePtr};
use float8::F8E4M3;
use half::{bf16, f16};
use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// Makes `ctx` current on the calling thread with a raw `cuCtxSetCurrent`.
///
/// Unlike `CudaContext::bind_to_thread`, this never looks at cudarc's deferred error state, so a
/// stale error recorded by an earlier `Drop` can neither make it fail nor be swallowed by it.
pub(crate) fn bind_context_raw(ctx: &CudaContext) -> std::result::Result<(), drv::DriverError> {
    let want = ctx.cu_ctx();
    match drv::ctx::get_current()? {
        Some(current) if current == want => Ok(()),
        _ => unsafe { drv::ctx::set_current(want) },
    }
}

/// Binds the context on the thread that drops the last handle of a device, so that the cuBLAS and
/// cuRAND handles, modules and scratch buffers owned by the device are released with a current
/// context. Must stay the first field of [`CudaDevice`] (fields drop in declaration order).
struct ContextBinder(Arc<CudaContext>);

impl Drop for ContextBinder {
    fn drop(&mut self) {
        let _ = bind_context_raw(&self.0);
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

/// Scratch buffers owned by a device and reused across calls (quantized matmul workspaces). They
/// are reallocated when too small and freed by [`CudaDevice::release_workspaces`].
#[derive(Default)]
pub(crate) struct Workspaces {
    slots: HashMap<&'static str, CudaSlice<u8>>,
}

impl Workspaces {
    /// Device address of the workspace `name`, (re)allocated when smaller than `bytes`.
    ///
    /// The previous buffer is freed in stream order before the new one is allocated, so kernels
    /// queued earlier still see valid memory and the pool can hand the bytes straight back.
    pub(crate) fn ensure(
        &mut self,
        dev: &CudaDevice,
        name: &'static str,
        bytes: usize,
    ) -> Result<u64> {
        let bytes = bytes.max(1);
        if self.slots.get(name).map_or(true, |s| s.len() < bytes) {
            // The old buffer is freed right below: the context must be current.
            dev.bind_to_thread()?;
            self.slots.remove(name);
            let slice = unsafe { dev.alloc::<u8>(bytes)? };
            self.slots.insert(name, slice);
        }
        let slice = &self.slots[name];
        Ok(slice.device_ptr(slice.stream()).0)
    }

    fn bytes(&self) -> usize {
        self.slots.values().map(|s| s.len()).sum()
    }
}

/// Usage of the stream-ordered memory pool of a device (`cuMemPoolGetAttribute`), in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPoolStats {
    /// Memory reserved by the pool from the driver (`CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT`).
    pub reserved: u64,
    /// Memory handed out to live allocations (`CU_MEMPOOL_ATTR_USED_MEM_CURRENT`).
    pub used: u64,
    pub reserved_high: u64,
    pub used_high: u64,
    /// Reserved memory the pool keeps when it releases at a synchronization point.
    pub release_threshold: u64,
}

impl CudaPoolStats {
    /// Reserved bytes that no allocation uses: the pool reuses them without asking the driver.
    pub fn idle(&self) -> u64 {
        self.reserved.saturating_sub(self.used)
    }
}

/// Default release threshold of the memory pool (`CANDLE_CUDA_POOL_KEEP_MB`, or the shared
/// `CANDLE_GPU_POOL_KEEP_MB`; default 256 MiB, `max` never releases on synchronization).
fn pool_keep_bytes_from_env() -> u64 {
    const DEFAULT_MB: u64 = 256;
    let v = std::env::var("CANDLE_CUDA_POOL_KEEP_MB")
        .or_else(|_| std::env::var("CANDLE_GPU_POOL_KEEP_MB"));
    match v {
        Ok(v) if v.trim().eq_ignore_ascii_case("max") => u64::MAX,
        Ok(v) => v.trim().parse::<u64>().unwrap_or(DEFAULT_MB).saturating_mul(1 << 20),
        Err(_) => DEFAULT_MB << 20,
    }
}

/// `CANDLE_CUDA_CHECK_LAUNCH=1`: check the launches of the statically compiled (cudart) kernels,
/// see [`CudaDevice::check_ffi_launch`].
fn check_launch_from_env() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("CANDLE_CUDA_CHECK_LAUNCH") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    })
}

extern "C" {
    /// CUDA runtime (linked through `candle-kernels`): returns and clears the last error of the
    /// calling thread's runtime calls, launch configuration errors included.
    fn cudaGetLastError() -> std::ffi::c_int;
}

fn event_tracking_from_env() -> bool {
    match std::env::var("CANDLE_CUDA_EVENT_TRACKING") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

/// Process-wide devices returned by [`BackendDevice::new`], one per ordinal. Held strongly so the
/// primary context, its loaded modules and the FFI kernels' static attribute caches stay valid.
static SHARED_DEVICES: OnceLock<Mutex<HashMap<usize, CudaDevice>>> = OnceLock::new();

/// NVRTC output shared by every device: (module name, GPU architecture) -> PTX.
static PTX_CACHE: OnceLock<Mutex<HashMap<(String, String), cudarc::nvrtc::Ptx>>> = OnceLock::new();

const CUDA_GRAPH_HTOD_CACHE_MAX_BYTES: usize = 4096;
/// Entries kept by the per-thread graph upload cache before it is reset.
const CUDA_GRAPH_HTOD_CACHE_MAX_ENTRIES: usize = 4096;

/// A cached upload of the graph cache. Binds the slice's context before it is freed: the
/// thread-local cache is dropped when its thread exits, typically with no context current.
struct CachedHtod<T>(CudaSlice<T>);

impl<T> Drop for CachedHtod<T> {
    fn drop(&mut self) {
        let _ = bind_context_raw(self.0.context());
    }
}

type CudaGraphHtodCacheKey = (DeviceId, TypeId, Vec<u8>);
type CudaGraphHtodCache = HashMap<CudaGraphHtodCacheKey, Box<dyn Any>>;

thread_local! {
    static CUDA_GRAPH_HTOD_CACHE: RefCell<CudaGraphHtodCache> =
        RefCell::new(HashMap::new());
    static CUDA_GRAPH_HTOD_CACHE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[must_use]
pub struct CudaGraphHtodCacheGuard;

impl Drop for CudaGraphHtodCacheGuard {
    fn drop(&mut self) {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| {
            debug_assert!(depth.get() > 0);
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

/// A CUDA device: context, stream, cuBLAS/cuRAND handles, loaded kernels and scratch buffers.
///
/// [`BackendDevice::new`] (i.e. `Device::new_cuda`) returns one process-wide device per ordinal
/// whose work, from every thread, goes to the context's legacy default stream (cudarc stays in
/// single-stream mode, CUDA events are not tracked unless `CANDLE_CUDA_EVENT_TRACKING=1`).
/// [`CudaDevice::new_with_stream`] creates an independent device with its own stream and event
/// tracking (needed e.g. for CUDA graph capture, which the legacy stream does not support).
#[derive(Clone)]
pub struct CudaDevice {
    // Must stay the first field, see `ContextBinder`.
    _binder: Arc<ContextBinder>,
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    /// Created on the first random op or `set_seed`.
    curand: Arc<Mutex<Option<CudaRng>>>,
    seed_value: Arc<RwLock<u64>>,
    workspaces: Arc<Mutex<Workspaces>>,
    /// Kernel parameter uploads cached while the CUDA graph upload cache is enabled.
    param_cache: Arc<Mutex<HashMap<Vec<usize>, Arc<CudaSlice<usize>>>>>,
    /// Raw non-blocking stream used by [`super::offload::CopyStream`], created on first use.
    copy_stream: Arc<Mutex<Option<Arc<super::offload::RawStream>>>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn enable_cuda_graph_htod_cache(&self) -> CudaGraphHtodCacheGuard {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        CudaGraphHtodCacheGuard
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        if cuda_graph_htod_cache_enabled() && !src.is_empty() {
            return self.memcpy_htod_capture_cached(src, dst);
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.check_capture_copy("clone_dtoh")?;
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.check_capture_copy("memcpy_dtoh")?;
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if cuda_graph_htod_cache_enabled() && !src.is_empty() {
            return self.clone_htod_capture_cached(src);
        }
        self.stream.clone_htod(src).w()
    }

    fn check_capture_copy(&self, op: &str) -> Result<()> {
        if cuda_graph_htod_cache_enabled() {
            let is_capturing = self.cuda_graph_capture_active();
            if is_capturing {
                crate::bail!(
                    "{op} during CUDA graph capture: host/device copies are not \
                     permitted while capturing (use param cache for kernel parameters)"
                );
            }
        }
        Ok(())
    }

    fn memcpy_htod_capture_cached<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        assert!(dst.len() >= src.len());
        if let Some(cached) = self.cached_htod_slice(src)? {
            return self.stream.memcpy_dtod(&cached, dst).w();
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    fn clone_htod_capture_cached<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if let Some(cached) = self.cached_htod_slice(src)? {
            return Ok(cached);
        }
        self.stream.clone_htod(src).w()
    }

    fn cached_htod_slice<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<Option<cudarc::driver::CudaSlice<T>>> {
        let (src_slice, _sync) = unsafe { src.stream_synced_slice(&self.stream) };
        let byte_len = std::mem::size_of_val(src_slice);
        if byte_len > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES {
            if self.cuda_graph_capture_active() {
                crate::bail!("CUDA graph capture cannot upload uncached host data");
            }
            return Ok(None);
        }

        let bytes =
            unsafe { std::slice::from_raw_parts(src_slice.as_ptr().cast::<u8>(), byte_len) }
                .to_vec();
        let key = (self.id, TypeId::of::<T>(), bytes);
        if let Some(cached) = CUDA_GRAPH_HTOD_CACHE.with(|cache| {
            cache.borrow().get(&key).and_then(|cached| {
                cached
                    .downcast_ref::<CachedHtod<T>>()
                    .map(|c| c.0.clone())
            })
        }) {
            return Ok(Some(cached));
        }
        if self.cuda_graph_capture_active() {
            crate::bail!("CUDA graph capture missing cached host data");
        }

        let cached = self.stream.clone_htod(src).w()?;
        CUDA_GRAPH_HTOD_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= CUDA_GRAPH_HTOD_CACHE_MAX_ENTRIES {
                cache.clear();
            }
            cache.insert(key, Box::new(CachedHtod(cached.clone())));
        });
        Ok(Some(cached))
    }

    fn cuda_graph_capture_active(&self) -> bool {
        matches!(
            self.stream.capture_status(),
            Ok(status) if status != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        )
    }
}

pub(crate) fn cuda_graph_htod_cache_enabled() -> bool {
    CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| depth.get() > 0)
}

/// Frees the uploads cached by the CUDA graph upload cache of the calling thread.
pub fn clear_cuda_graph_htod_cache() {
    let _ = CUDA_GRAPH_HTOD_CACHE.try_with(|cache| {
        let drained: Vec<_> = cache.borrow_mut().drain().collect();
        drop(drained);
    });
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }

    /// Allows `bytes` of dynamic shared memory (needed above 48 KiB).
    pub fn set_max_dynamic_shared_memory(&self, bytes: usize) -> Result<()> {
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        self.func
            .set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32)
            .w()
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        candle_ug::cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    /// Function `fn_name` of the run-time compiled module `module_name`; `ptx` produces the
    /// module the first time it is needed in this process for this GPU architecture (the PTX is
    /// cached process-wide by `(module_name, arch)`, so other devices never recompile it).
    pub fn get_or_compile_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: impl FnOnce() -> Result<cudarc::nvrtc::Ptx>,
    ) -> Result<CudaFunc> {
        {
            let ms = self.custom_modules.read().unwrap();
            if let Some(mdl) = ms.get(module_name) {
                let func = mdl.load_function(fn_name).w()?;
                return Ok(CudaFunc {
                    func,
                    stream: self.stream.clone(),
                });
            }
        }
        let key = (module_name.to_string(), self.arch_key());
        let cache = PTX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let cached = cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .cloned();
        let ptx = match cached {
            Some(ptx) => ptx,
            None => {
                // Compiled outside the lock: two threads may race on a cold module, both results
                // are identical.
                let ptx = ptx()?;
                cache
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .entry(key)
                    .or_insert_with(|| ptx.clone());
                ptx
            }
        };
        let mut ms = self.custom_modules.write().unwrap();
        let mdl = match ms.get(module_name) {
            Some(m) => m.clone(),
            None => {
                let m = self.context.load_module(ptx).w()?;
                ms.insert(module_name.to_string(), m.clone());
                m
            }
        };
        let func = mdl.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    /// A device attribute (compute capability, SM count, shared memory, ...).
    pub fn device_attribute(&self, attr: cudarc::driver::sys::CUdevice_attribute) -> Result<i32> {
        // Raw query: `CudaContext::attribute` would consume a pending deferred error.
        unsafe { drv::device::get_attribute(self.context.cu_device(), attr) }.w()
    }

    /// `sm_<major><minor>` of this GPU, the key of the process-wide PTX cache.
    fn arch_key(&self) -> String {
        use sys::CUdevice_attribute as A;
        let major = self
            .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let minor = self
            .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        format!("sm_{major}{minor}")
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }

    /// Ordinal of the GPU.
    pub fn ordinal(&self) -> usize {
        self.context.ordinal()
    }

    /// The cudarc context of this device.
    pub fn cuda_context(&self) -> &Arc<cudarc::driver::CudaContext> {
        &self.context
    }

    /// Makes this device's context current on the calling thread (raw `cuCtxSetCurrent`, which
    /// neither fails on nor clears a pending deferred error, see [`Self::take_deferred_error`]).
    pub fn bind_to_thread(&self) -> Result<()> {
        bind_context_raw(&self.context).w()
    }

    /// Takes the error that cudarc recorded in a `Drop` impl (which cannot return it) since the
    /// last call. cudarc returns such an error from the next safe call that binds the context,
    /// which then skips its work, so it is better drained explicitly.
    pub fn take_deferred_error(&self) -> Option<String> {
        self.context.check_err().err().map(|e| format!("{e:?}"))
    }

    /// With `CANDLE_CUDA_CHECK_LAUNCH=1`, turns a failed launch of the kernels that `candle-kernels`
    /// compiles statically (MMQ/MMVQ, launched through the CUDA runtime, which reports nothing to
    /// the caller) into an error: checks `cudaGetLastError`, then waits for the stream so that
    /// execution faults surface here. A no-op otherwise.
    pub(crate) fn check_ffi_launch(&self, what: &str) -> Result<()> {
        if !check_launch_from_env() {
            return Ok(());
        }
        let code = unsafe { cudaGetLastError() };
        if code != 0 {
            crate::bail!("{what}: kernel launch failed (cudaError_t {code})")
        }
        self.synchronize()
            .map_err(|e| crate::Error::Msg(format!("{what}: kernel execution failed: {e}")))
    }

    /// Drains and logs a stale deferred error before a synchronization.
    fn drain_deferred_error(&self, at: &str) {
        if let Some(e) = self.take_deferred_error() {
            eprintln!(
                "[CUDA-DEFERRED] GPU {} {at}: cleared a deferred error recorded earlier: {e}",
                self.ordinal()
            );
        }
    }

    /// Waits for all the work queued in this device's context, by any thread and on any stream
    /// (`cuCtxSynchronize`). Pending stream-ordered frees complete, and the memory pool releases
    /// what it holds above its release threshold.
    pub fn synchronize_all(&self) -> Result<()> {
        self.bind_to_thread()?;
        self.drain_deferred_error("synchronize_all");
        drv::ctx::synchronize().w()
    }

    /// `(free, total)` device memory in bytes, as reported by the driver (`cuMemGetInfo`). Memory
    /// reserved by the stream-ordered pool counts as used, see [`Self::effective_free_memory`].
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.bind_to_thread()?;
        drv::mem_get_info().w()
    }

    /// The default stream-ordered memory pool of the device, `None` when the device does not
    /// support stream-ordered allocation (cudarc then uses `cuMemAlloc`/`cuMemFree`).
    fn mem_pool(&self) -> Result<Option<sys::CUmemoryPool>> {
        if !self.context.has_async_alloc() {
            return Ok(None);
        }
        let pool = unsafe { drv::device::get_default_mem_pool(self.context.cu_device()) }.w()?;
        Ok(Some(pool))
    }

    /// Usage of the stream-ordered memory pool, `None` without stream-ordered allocation.
    pub fn pool_stats(&self) -> Result<Option<CudaPoolStats>> {
        use sys::CUmemPool_attribute as A;
        let Some(pool) = self.mem_pool()? else {
            return Ok(None);
        };
        let get = |attr: A| -> Result<u64> {
            let mut v = 0u64;
            unsafe {
                drv::mem_pool::get_attribute(pool, attr, &mut v as *mut u64 as *mut std::ffi::c_void)
            }
            .w()?;
            Ok(v)
        };
        Ok(Some(CudaPoolStats {
            reserved: get(A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            used: get(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            reserved_high: get(A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
            used_high: get(A::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
            release_threshold: get(A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)?,
        }))
    }

    /// Bytes reserved by the stream-ordered pool that no allocation uses (0 without a pool).
    pub fn pool_idle_bytes(&self) -> Result<usize> {
        Ok(self
            .pool_stats()?
            .map_or(0, |s| s.idle().min(usize::MAX as u64) as usize))
    }

    /// Memory this process can allocate without waiting: driver free memory plus the idle bytes
    /// of the stream-ordered pool.
    pub fn effective_free_memory(&self) -> Result<usize> {
        let (free, _) = self.mem_info()?;
        Ok(free.saturating_add(self.pool_idle_bytes()?))
    }

    /// Waits for the device ([`Self::synchronize_all`]) and returns the pool memory above
    /// `keep_bytes` to the driver (`cuMemPoolTrimTo`). `trim_memory_pool(0)` at the end of a task
    /// gives everything back.
    pub fn trim_memory_pool(&self, keep_bytes: usize) -> Result<()> {
        self.synchronize_all()?;
        if let Some(pool) = self.mem_pool()? {
            unsafe { drv::mem_pool::trim_to(pool, keep_bytes) }.w()?;
        }
        Ok(())
    }

    /// Sets how much reserved memory the pool keeps when it releases memory at a synchronization
    /// point (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`; `u64::MAX` never releases implicitly). The
    /// shared devices start with `CANDLE_CUDA_POOL_KEEP_MB` (default 256 MiB).
    pub fn set_memory_pool_release_threshold(&self, bytes: u64) -> Result<()> {
        use sys::CUmemPool_attribute as A;
        if let Some(pool) = self.mem_pool()? {
            let mut v = bytes;
            unsafe {
                drv::mem_pool::set_attribute(
                    pool,
                    A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut v as *mut u64 as *mut std::ffi::c_void,
                )
            }
            .w()?;
        }
        Ok(())
    }

    /// Scratch buffers of this device, see [`Workspaces`].
    pub(crate) fn workspaces(&self) -> MutexGuard<'_, Workspaces> {
        self.workspaces.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Bytes currently held by the quantized matmul workspaces of this device.
    pub fn workspace_bytes(&self) -> usize {
        self.workspaces().bytes()
    }

    /// Frees the quantized matmul workspaces and cached kernel parameters of this device (they
    /// are reallocated on demand). The frees are stream ordered, so work in flight is not
    /// affected; the memory goes back to the pool (see [`Self::trim_memory_pool`]).
    pub fn release_workspaces(&self) -> Result<()> {
        self.bind_to_thread()?;
        let slots = std::mem::take(&mut self.workspaces().slots);
        drop(slots);
        let params = std::mem::take(&mut *self.param_cache.lock().unwrap_or_else(|p| p.into_inner()));
        drop(params);
        clear_cuda_graph_htod_cache();
        Ok(())
    }

    /// Cached device copy of the kernel parameters `params` (used in CUDA graph upload cache
    /// mode, where parameters must outlive the capture).
    pub(crate) fn cached_params(&self, params: &[usize]) -> Option<Arc<CudaSlice<usize>>> {
        self.param_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(params)
            .cloned()
    }

    pub(crate) fn insert_cached_params(
        &self,
        params: Vec<usize>,
        slice: Arc<CudaSlice<usize>>,
    ) -> Arc<CudaSlice<usize>> {
        self.param_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(params)
            .or_insert(slice)
            .clone()
    }

    /// The raw copy stream, created on first use.
    pub(crate) fn raw_copy_stream(&self) -> Result<Arc<super::offload::RawStream>> {
        let mut slot = self.copy_stream.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(s) = slot.as_ref() {
            return Ok(s.clone());
        }
        let s = Arc::new(super::offload::RawStream::new(self.context.clone())?);
        *slot = Some(s.clone());
        Ok(s)
    }

    /// The cuRAND generator, created with the current seed on first use.
    fn rng(&self) -> Result<MutexGuard<'_, Option<CudaRng>>> {
        let mut guard = self.curand.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            let seed = *self.seed_value.read().unwrap();
            let rng = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
            *guard = Some(CudaRng(rng));
        }
        Ok(guard)
    }
}

impl CudaDevice {
    /// Creates an independent device with its own stream (cudarc multi-stream mode with event
    /// tracking), which does not share kernels, handles or scratch buffers with the device that
    /// `Device::new_cuda` returns.
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        Self::from_context_and_stream(context, stream)
    }

    /// The process-wide device of `ordinal`: legacy default stream, event tracking off (unless
    /// `CANDLE_CUDA_EVENT_TRACKING=1`), pool release threshold from `CANDLE_CUDA_POOL_KEEP_MB`.
    fn create_shared(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        // One stream for every thread: events would protect nothing, and before any allocation
        // so that no slice carries them. Every drop of device memory binds the context instead
        // (see `impl Drop for CudaStorage`).
        if !event_tracking_from_env() {
            unsafe { context.disable_event_tracking() };
        }
        // The legacy NULL stream keeps cudarc in single-stream mode (`new_stream` would count a
        // second stream) and orders the work of all threads; stream-ordered allocation works on it.
        let stream = context.default_stream();
        let dev = Self::from_context_and_stream(context, stream)?;
        if let Err(e) = dev.set_memory_pool_release_threshold(pool_keep_bytes_from_env()) {
            eprintln!("candle-cuda: cannot set the memory pool release threshold: {e}");
        }
        Ok(dev)
    }

    fn from_context_and_stream(
        context: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            _binder: Arc::new(ContextBinder(context.clone())),
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(None)),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            workspaces: Arc::new(Mutex::new(Workspaces::default())),
            param_cache: Arc::new(Mutex::new(HashMap::new())),
            copy_stream: Arc::new(Mutex::new(None)),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    /// Returns the process-wide device of `ordinal` (created on first use), see [`CudaDevice`].
    fn new(ordinal: usize) -> Result<Self> {
        let registry = SHARED_DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(dev) = registry.get(&ordinal) {
            dev.bind_to_thread()?;
            return Ok(dev.clone());
        }
        let dev = Self::create_shared(ordinal)?;
        registry.insert(ordinal, dev.clone());
        Ok(dev)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap_or_else(|p| p.into_inner());
        *curand = Some(CudaRng(
            cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?,
        ));
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let mut rng = self.rng()?;
        let curand = rng.as_mut().expect("curand initialized");
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let mut rng = self.rng()?;
        let curand = rng.as_mut().expect("curand initialized");
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    /// Waits for the work queued on this device's stream. For the shared devices that is the
    /// work of every thread. A stale deferred error is logged and cleared first, so the wait
    /// always happens (cudarc would otherwise return the stale error and skip it).
    fn synchronize(&self) -> Result<()> {
        self.bind_to_thread()?;
        self.drain_deferred_error("synchronize");
        unsafe { drv::stream::synchronize(self.stream.cu_stream()) }.w()
    }
}
