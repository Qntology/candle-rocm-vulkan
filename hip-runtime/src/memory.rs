//! Typed GPU memory buffers with RAII.

use crate::error::{check_hip, Result};
use hip_sys::hip_runtime::{self, hipMemcpyKind};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn account_alloc(bytes: usize) {
    if bytes == 0 {
        return;
    }
    let now = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    LIVE_ALLOCS.fetch_add(1, Ordering::Relaxed);
    PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
}

fn account_free(bytes: usize) {
    if bytes == 0 {
        return;
    }
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
    LIVE_ALLOCS.fetch_sub(1, Ordering::Relaxed);
}

/// Bytes / allocations currently owned by live `DeviceBuffer`s of this process
/// (i.e. what the application still references; excludes driver-side caches).
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocStats {
    pub live_bytes: usize,
    pub live_allocs: usize,
    pub peak_bytes: usize,
}

pub fn alloc_stats() -> AllocStats {
    AllocStats {
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        live_allocs: LIVE_ALLOCS.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

/// Resets the peak counter to the current live size.
pub fn reset_peak_alloc() {
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// State of the default stream-ordered memory pool of a device.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    pub reserved_current: u64,
    pub reserved_high: u64,
    pub used_current: u64,
    pub used_high: u64,
    pub release_threshold: u64,
}

/// Typed GPU memory buffer with RAII.
pub struct DeviceBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    stream_ordered: bool,
    _phantom: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate uninitialized GPU memory for `len` elements.
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        if bytes > 0 {
            check_hip(unsafe { hip_runtime::hipMalloc(&mut ptr, bytes) })?;
            account_alloc(bytes);
        }
        Ok(Self {
            ptr,
            len,
            stream_ordered: false,
            _phantom: PhantomData,
        })
    }

    /// Allocate uninitialized memory from the default stream-ordered memory pool.
    /// Freeing such a buffer does not synchronize the device.
    pub fn alloc_async(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        if bytes > 0 {
            check_hip(unsafe { hip_runtime::hipMallocAsync(&mut ptr, bytes, std::ptr::null_mut()) })?;
            account_alloc(bytes);
        }
        Ok(Self {
            ptr,
            len,
            stream_ordered: true,
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

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            account_free(self.len * std::mem::size_of::<T>());
            unsafe {
                if self.stream_ordered {
                    hip_runtime::hipFreeAsync(self.ptr, std::ptr::null_mut());
                } else {
                    hip_runtime::hipFree(self.ptr);
                }
            }
        }
    }
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
    let mut pool: hip_runtime::hipMemPool_t = std::ptr::null_mut();
    check_hip(unsafe { hip_runtime::hipDeviceGetDefaultMemPool(&mut pool, device) })?;
    check_hip(unsafe { hip_runtime::hipMemPoolTrimTo(pool, 0) })
}

/// Reads the usage counters of the default memory pool of `device`.
pub fn pool_stats(device: i32) -> Result<PoolStats> {
    let mut pool: hip_runtime::hipMemPool_t = std::ptr::null_mut();
    check_hip(unsafe { hip_runtime::hipDeviceGetDefaultMemPool(&mut pool, device) })?;
    let get = |attr: i32| -> Result<u64> {
        let mut v: u64 = 0;
        check_hip(unsafe {
            hip_runtime::hipMemPoolGetAttribute(pool, attr, &mut v as *mut u64 as *mut c_void)
        })?;
        Ok(v)
    };
    Ok(PoolStats {
        reserved_current: get(hip_runtime::HIP_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        reserved_high: get(hip_runtime::HIP_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
        used_current: get(hip_runtime::HIP_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
        used_high: get(hip_runtime::HIP_MEMPOOL_ATTR_USED_MEM_HIGH)?,
        release_threshold: get(hip_runtime::HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD).unwrap_or(0),
    })
}

/// Sets how many bytes the default pool may keep cached after a synchronization (0 = give everything back).
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
