//! Typed GPU memory buffers with RAII.

use crate::error::{check_hip, Result};
use hip_sys::hip_runtime::{self, hipMemcpyKind};
use std::ffi::c_void;
use std::marker::PhantomData;

/// Typed GPU memory buffer with RAII.
pub struct DeviceBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    stream_ordered: bool,
    /// Bytes actually allocated (the cache size class, `>= len * size_of::<T>()`).
    cap: usize,
    /// Device the buffer was allocated on when it may go back to the buffer cache, else -1.
    cache_device: i32,
    _phantom: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate uninitialized GPU memory for `len` elements.
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        if bytes > 0 {
            check_hip(unsafe { hip_runtime::hipMalloc(&mut ptr, bytes) })?;
        }
        Ok(Self {
            ptr,
            len,
            stream_ordered: false,
            cap: bytes,
            cache_device: -1,
            _phantom: PhantomData,
        })
    }

    /// Allocate uninitialized memory from the default stream-ordered memory pool.
    /// Freeing such a buffer does not synchronize the device.
    ///
    /// When the buffer cache is on ([`set_cache_limit`]), a buffer of the same size class freed
    /// earlier on the current device is handed out again instead of calling `hipMallocAsync`.
    pub fn alloc_async(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        let mut cap = bytes;
        let mut cache_device = -1;
        if bytes > 0 {
            if cache_limit() > 0 {
                let mut device = 0;
                check_hip(unsafe { hip_runtime::hipGetDevice(&mut device) })?;
                cache_device = device;
                cap = size_class(bytes);
                if let Some(p) = cache_take(device, cap) {
                    ptr = p;
                }
            }
            if ptr.is_null() {
                let mut code =
                    unsafe { hip_runtime::hipMallocAsync(&mut ptr, cap, std::ptr::null_mut()) };
                if code == hip_runtime::HIP_ERROR_OUT_OF_MEMORY
                    && cache_device >= 0
                    && cached_bytes(cache_device) > 0
                {
                    // The cache holds memory the pool could hand out: give it back and retry.
                    unsafe { hip_runtime::hipGetLastError() };
                    release_cached(cache_device)?;
                    code = unsafe { hip_runtime::hipMallocAsync(&mut ptr, cap, std::ptr::null_mut()) };
                }
                check_hip(code)?;
            }
        }
        Ok(Self {
            ptr,
            len,
            stream_ordered: true,
            cap,
            cache_device,
            _phantom: PhantomData,
        })
    }

    /// Allocate zero-initialized GPU memory.
    pub fn alloc_zeros(len: usize) -> Result<Self> {
        let buf = Self::alloc(len)?;
        let bytes = buf.byte_size();
        if bytes > 0 {
            check_hip(unsafe { hip_runtime::hipMemset(buf.ptr, 0, bytes) })?;
        }
        Ok(buf)
    }

    /// Copy from host slice to device.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Overwrite the start of the buffer with `data`.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        assert!(data.len() <= self.len, "copy_from_host: source larger than buffer");
        let bytes = std::mem::size_of_val(data);
        unsafe { memcpy_htod(self.ptr, data.as_ptr() as *const c_void, bytes) }
    }

    /// Copy the start of the buffer into `out`.
    pub fn copy_to_host(&self, out: &mut [T]) -> Result<()> {
        assert!(out.len() <= self.len, "copy_to_host: destination larger than buffer");
        let bytes = std::mem::size_of_val(out);
        unsafe { memcpy_dtoh(out.as_mut_ptr() as *mut c_void, self.ptr, bytes) }
    }

    /// Copy device buffer back to host.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut result: Vec<T> = Vec::with_capacity(self.len);
        let bytes = self.byte_size();
        unsafe {
            memcpy_dtoh(result.as_mut_ptr() as *mut c_void, self.ptr, bytes)?;
            result.set_len(self.len);
        }
        Ok(result)
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Raw void pointer (for kernel params and rocBLAS).
    pub fn as_void_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// Number of `hipFree` / `hipFreeAsync` calls that failed in [`DeviceBuffer`]'s `Drop` (the memory
/// of those buffers is leaked). Exposed for diagnostics, see [`free_failures`].
static FREE_FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of device buffer frees that failed so far in this process.
pub fn free_failures() -> usize {
    FREE_FAILURES.load(std::sync::atomic::Ordering::Relaxed)
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if self.stream_ordered
                && self.cache_device >= 0
                && cache_put(self.cache_device, self.cap, self.ptr)
            {
                return;
            }
            let code = unsafe {
                if self.stream_ordered {
                    hip_runtime::hipFreeAsync(self.ptr, std::ptr::null_mut())
                } else {
                    hip_runtime::hipFree(self.ptr)
                }
            };
            if code != hip_runtime::HIP_SUCCESS {
                FREE_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Buffer cache of the stream-ordered allocator
// ---------------------------------------------------------------------------------------------
// On Windows (HIP SDK 7.2, RX 6600) a hipMallocAsync / hipFreeAsync pair costs ~45 µs of host
// time even when the pool holds the memory, and with a busy GPU the allocation waits until the
// queue reaches the earlier free. A fused GDN decode step (3 outputs, 24 µs of GPU time) then
// took ~166 µs. Stream-ordered buffers dropped by `DeviceBuffer` therefore go to this cache and
// `alloc_async` hands them out again on the same device.
//
// This is ordered like hipFreeAsync + hipMallocAsync: every launch, copy, allocation and free of
// this crate's users runs on the legacy NULL stream of the device, so the new owner's work is
// queued after all the work of the previous owner.
//
// The cache is off (limit 0) until a caller sets a limit with `set_cache_limit`. It holds at most
// that many bytes per device; buffers beyond it are freed as before. `release_cached` returns
// the cached buffers to the pool (call it before trimming the pool).

struct BufferCache {
    /// (device, size class in bytes) -> free device pointers.
    free: std::collections::BTreeMap<(i32, usize), Vec<usize>>,
    /// Cached bytes per device.
    bytes: std::collections::BTreeMap<i32, usize>,
}

static CACHE: std::sync::Mutex<BufferCache> = std::sync::Mutex::new(BufferCache {
    free: std::collections::BTreeMap::new(),
    bytes: std::collections::BTreeMap::new(),
});
static CACHE_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn cache_lock() -> std::sync::MutexGuard<'static, BufferCache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Allocation size of a request: 8 classes per power of two (at most 12.5 % slack), 512 B
/// granularity below 4 KiB, so that buffers of nearby sizes can be reused.
fn size_class(bytes: usize) -> usize {
    if bytes <= 4096 {
        return bytes.next_multiple_of(512);
    }
    let log2 = usize::BITS - 1 - bytes.leading_zeros();
    bytes.next_multiple_of(1usize << (log2 - 3))
}

fn cache_take(device: i32, class: usize) -> Option<*mut c_void> {
    let mut c = cache_lock();
    let ptr = c.free.get_mut(&(device, class))?.pop()?;
    if let Some(b) = c.bytes.get_mut(&device) {
        *b -= class;
    }
    Some(ptr as *mut c_void)
}

fn cache_put(device: i32, class: usize, ptr: *mut c_void) -> bool {
    let limit = cache_limit();
    let mut c = cache_lock();
    let held = c.bytes.get(&device).copied().unwrap_or(0);
    if limit == 0 || held + class > limit {
        return false;
    }
    c.free.entry((device, class)).or_default().push(ptr as usize);
    c.bytes.insert(device, held + class);
    true
}

/// Sets the most bytes the buffer cache keeps per device (0 turns it off; already cached
/// buffers stay until [`release_cached`]).
pub fn set_cache_limit(bytes: usize) {
    CACHE_LIMIT.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Most bytes the buffer cache keeps per device (0 = off).
pub fn cache_limit() -> usize {
    CACHE_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Bytes of freed buffers the cache holds for `device`.
pub fn cached_bytes(device: i32) -> usize {
    cache_lock().bytes.get(&device).copied().unwrap_or(0)
}

/// Returns the cached buffers of `device` to the stream-ordered pool (`hipFreeAsync`).
/// `device` must be the current device of the calling thread.
pub fn release_cached(device: i32) -> Result<()> {
    let ptrs: Vec<usize> = {
        let mut c = cache_lock();
        let keys: Vec<(i32, usize)> = c.free.range((device, 0)..=(device, usize::MAX)).map(|(k, _)| *k).collect();
        let mut ptrs = Vec::new();
        for k in keys {
            if let Some(v) = c.free.remove(&k) {
                ptrs.extend(v);
            }
        }
        c.bytes.remove(&device);
        ptrs
    };
    let mut first_err = hip_runtime::HIP_SUCCESS;
    for p in ptrs {
        let code = unsafe { hip_runtime::hipFreeAsync(p as *mut c_void, std::ptr::null_mut()) };
        if code != hip_runtime::HIP_SUCCESS {
            FREE_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if first_err == hip_runtime::HIP_SUCCESS {
                first_err = code;
            }
        }
    }
    check_hip(first_err)
}

/// Returns `true` when the stream-ordered allocator (`hipMallocAsync`) works on the current device.
pub fn stream_ordered_alloc_supported() -> bool {
    let mut ptr = std::ptr::null_mut();
    let code = unsafe { hip_runtime::hipMallocAsync(&mut ptr, 256, std::ptr::null_mut()) };
    if code != hip_runtime::HIP_SUCCESS || ptr.is_null() {
        unsafe { hip_runtime::hipGetLastError() };
        return false;
    }
    let code = unsafe { hip_runtime::hipFreeAsync(ptr, std::ptr::null_mut()) };
    code == hip_runtime::HIP_SUCCESS
}

/// Releases the unused memory cached by the default memory pool of `device` back to the system.
pub fn trim_default_pool(device: i32) -> Result<()> {
    trim_default_pool_to(device, 0)
}

/// Releases the unused memory cached by the default memory pool of `device` back to the system,
/// keeping at least `keep_bytes` reserved.
pub fn trim_default_pool_to(device: i32, keep_bytes: usize) -> Result<()> {
    let mut pool: hip_runtime::hipMemPool_t = std::ptr::null_mut();
    check_hip(unsafe { hip_runtime::hipDeviceGetDefaultMemPool(&mut pool, device) })?;
    check_hip(unsafe { hip_runtime::hipMemPoolTrimTo(pool, keep_bytes) })
}

/// `(reserved, used)` bytes of the default memory pool of `device`: memory the pool holds from
/// the system, and the part of it handed out to live allocations.
pub fn default_pool_usage(device: i32) -> Result<(u64, u64)> {
    let mut pool: hip_runtime::hipMemPool_t = std::ptr::null_mut();
    check_hip(unsafe { hip_runtime::hipDeviceGetDefaultMemPool(&mut pool, device) })?;
    let mut reserved = 0u64;
    let mut used = 0u64;
    check_hip(unsafe {
        hip_runtime::hipMemPoolGetAttribute(
            pool,
            hip_runtime::HIP_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut c_void,
        )
    })?;
    check_hip(unsafe {
        hip_runtime::hipMemPoolGetAttribute(
            pool,
            hip_runtime::HIP_MEMPOOL_ATTR_USED_MEM_CURRENT,
            &mut used as *mut u64 as *mut c_void,
        )
    })?;
    Ok((reserved, used))
}

pub fn set_pool_release_threshold(device: i32, bytes: u64) -> Result<()> {
    let mut pool: hip_runtime::hipMemPool_t = std::ptr::null_mut();
    check_hip(unsafe { hip_runtime::hipDeviceGetDefaultMemPool(&mut pool, device) })?;
    let mut v = bytes;
    check_hip(unsafe {
        hip_runtime::hipMemPoolSetAttribute(
            pool,
            hip_runtime::HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &mut v as *mut u64 as *mut c_void,
        )
    })
}

// Device pointers are plain addresses; the HIP runtime is thread safe.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

/// # Safety
/// `dst` must be a device allocation of at least `bytes` bytes, `src` a readable host region.
pub unsafe fn memcpy_htod(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check_hip(hip_runtime::hipMemcpy(dst, src, bytes, hipMemcpyKind::hipMemcpyHostToDevice))
}

/// # Safety
/// `src` must be a device allocation of at least `bytes` bytes, `dst` a writable host region.
pub unsafe fn memcpy_dtoh(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check_hip(hip_runtime::hipMemcpy(dst, src, bytes, hipMemcpyKind::hipMemcpyDeviceToHost))
}

/// # Safety
/// Both pointers must be device allocations of at least `bytes` bytes.
pub unsafe fn memcpy_dtod(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check_hip(hip_runtime::hipMemcpy(dst, src, bytes, hipMemcpyKind::hipMemcpyDeviceToDevice))
}

/// # Safety
/// `dst` must be a device allocation of at least `bytes` bytes.
pub unsafe fn memset(dst: *mut c_void, value: u8, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    check_hip(hip_runtime::hipMemset(dst, value as i32, bytes))
}
