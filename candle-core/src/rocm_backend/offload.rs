//! Pinned host buffers and an asynchronous copy stream, to overlap host <-> device transfers of
//! offloaded data (RAM/SSD-resident KV cache blocks, layer weights) with compute. Same API as
//! `cuda_backend::offload`.
//!
//! Every kernel, rocBLAS call, allocation and free of the ROCm backend runs on the device's
//! legacy NULL stream (the "compute stream" below). The copy stream is a separate non-blocking
//! HIP stream (`hipStreamCreateWithFlags(hipStreamNonBlocking)`); the ordering between the two is
//! done here with HIP events:
//!
//! - [`CopyStream::upload`] / [`CopyStream::download`] first make the copy stream wait for the
//!   work queued so far on the compute stream. This covers the stream-ordered allocation of the
//!   device buffer (`hipMallocAsync` on the NULL stream, or a buffer handed out again by the
//!   backend's buffer cache) and earlier kernels that still read or write the buffer.
//! - They return a [`CopyEvent`]. Call [`CopyEvent::wait_on_compute`] before queuing compute
//!   work that reads the uploaded bytes (or that overwrites / frees the source of a download), and
//!   [`CopyEvent::synchronize`] before reading downloaded bytes on the host.
//!
//! Typical prefetch with a staging slot (tensors from `Tensor::zeros(.., &device)`, whose storage
//! is reached through `Tensor::storage_and_layout`):
//!
//! ```ignore
//! let copy = rocm_dev.copy_stream()?;
//! let mut host = rocm_dev.alloc_pinned(bytes)?;          // once
//! host.as_mut_slice()[..n].copy_from_slice(&kv_bytes);    // waits for earlier copies of `host`
//! let ev = {
//!     let (storage, _) = slot.storage_and_layout();
//!     let candle_core::Storage::Rocm(dst) = &*storage else { unreachable!() };
//!     copy.upload(&host, 0, dst, 0, n)?
//! };
//! // ... queue other layers: they run while the DMA engine copies ...
//! ev.wait_on_compute(&rocm_dev)?;                        // before the kernels reading `slot`
//! ```
//!
//! The destination of an upload (source of a download) must stay alive until
//! [`CopyEvent::wait_on_compute`] was called for it: a dropped buffer goes back to the backend's
//! buffer cache (or the stream-ordered pool) and may be handed to another allocation while the
//! copy still runs. Dropping a [`CopyEvent`] that was never waited on queues that wait on the
//! compute stream, which orders everything the compute stream does afterwards (including the
//! reuse of a freed buffer) after the copy.
//!
//! Performance notes (RX 6600, HIP SDK 7.2, Windows, `rocm_memory_tests` bench): a pinned upload
//! reaches ~6.65 GB/s against ~5.4 GB/s for a pageable `Tensor::from_slice` (64 MiB; 5.75 vs
//! 5.0 GB/s at 8 MiB), and a 32 MiB upload queued before ~110 ms of unrelated kernels completed
//! while they still ran. For small copies (1 MiB) the pinned path measured slower (2.4 vs 3.1
//! GB/s): keep it for large transfers, issued a few layers ahead.

use super::{RocmDevice, RocmStorage, WrapErr};
use crate::{bail, Result};
use hip_runtime::error::check_hip;
use hip_sys::hip_runtime as sys;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The legacy NULL stream: the compute stream of every ROCm device.
const COMPUTE: sys::hipStream_t = std::ptr::null_mut();

fn set_device(ordinal: usize) -> Result<()> {
    check_hip(unsafe { sys::hipSetDevice(ordinal as i32) }).w()
}

/// A HIP event created with `hipEventDisableTiming` on a given device.
pub(crate) struct RawEvent {
    event: sys::hipEvent_t,
    ordinal: usize,
}

// HIP events can be used from any thread.
unsafe impl Send for RawEvent {}
unsafe impl Sync for RawEvent {}

impl RawEvent {
    fn new(ordinal: usize) -> Result<Self> {
        set_device(ordinal)?;
        let mut event = std::ptr::null_mut();
        check_hip(unsafe { sys::hipEventCreateWithFlags(&mut event, sys::hipEventDisableTiming) })
            .w()?;
        Ok(Self { event, ordinal })
    }

    fn record(&self, stream: sys::hipStream_t) -> Result<()> {
        check_hip(unsafe { sys::hipEventRecord(self.event, stream) }).w()
    }

    fn is_complete(&self) -> bool {
        set_device(self.ordinal).is_ok()
            && unsafe { sys::hipEventQuery(self.event) } == sys::HIP_SUCCESS
    }

    fn synchronize(&self) -> Result<()> {
        set_device(self.ordinal)?;
        check_hip(unsafe { sys::hipEventSynchronize(self.event) }).w()
    }
}

impl Drop for RawEvent {
    fn drop(&mut self) {
        // Does not block: the runtime releases the event once the recorded work completed.
        if set_device(self.ordinal).is_ok() {
            let _ = unsafe { sys::hipEventDestroy(self.event) };
        }
    }
}

/// A non-blocking HIP stream (`hipStreamCreateWithFlags(hipStreamNonBlocking)`).
pub(crate) struct RawStream {
    stream: sys::hipStream_t,
    ordinal: usize,
}

unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

impl RawStream {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        set_device(ordinal)?;
        let mut stream = std::ptr::null_mut();
        check_hip(unsafe { sys::hipStreamCreateWithFlags(&mut stream, sys::hipStreamNonBlocking) })
            .w()?;
        Ok(Self { stream, ordinal })
    }
}

impl Drop for RawStream {
    fn drop(&mut self) {
        // Does not block: queued work completes before the runtime releases the stream.
        if set_device(self.ordinal).is_ok() {
            let _ = unsafe { sys::hipStreamDestroy(self.stream) };
        }
    }
}

/// Page-locked host memory (`hipHostMalloc(hipHostMallocPortable)`), usable with the copy
/// streams of every device. Accessing the bytes ([`Self::as_slice`], [`Self::as_mut_slice`])
/// first waits for the copies queued on this buffer, so a reused staging buffer is never
/// overwritten while the DMA engine still reads it, and downloaded bytes are complete.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
    ordinal: usize,
    /// Completion events of the copies queued on this buffer that may still be running.
    pending: Mutex<Vec<Arc<RawEvent>>>,
}

// The buffer is plain host memory; accesses are synchronized through `pending`.
unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl std::fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PinnedHostBuffer({} bytes)", self.len)
    }
}

impl PinnedHostBuffer {
    fn new(ordinal: usize, len: usize) -> Result<Self> {
        let ptr = if len == 0 {
            std::ptr::null_mut()
        } else {
            set_device(ordinal)?;
            let mut p: *mut c_void = std::ptr::null_mut();
            check_hip(unsafe { sys::hipHostMalloc(&mut p, len, sys::hipHostMallocPortable) }).w()?;
            p as *mut u8
        };
        Ok(Self {
            ptr,
            len,
            ordinal,
            pending: Mutex::new(Vec::new()),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Blocks until every copy queued on this buffer has completed.
    pub fn wait_idle(&self) -> Result<()> {
        let pending = std::mem::take(&mut *self.pending.lock().unwrap_or_else(|p| p.into_inner()));
        for ev in pending.iter() {
            ev.synchronize()?;
        }
        Ok(())
    }

    /// The bytes, after the copies queued on this buffer completed.
    pub fn as_slice(&self) -> &[u8] {
        if let Err(e) = self.wait_idle() {
            eprintln!("candle-rocm: waiting for the copies of a pinned buffer failed: {e}");
        }
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The bytes, after the copies queued on this buffer completed.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if let Err(e) = self.wait_idle() {
            eprintln!("candle-rocm: waiting for the copies of a pinned buffer failed: {e}");
        }
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn track(&self, ev: &Arc<RawEvent>) {
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        pending.retain(|e| !e.is_complete());
        pending.push(ev.clone());
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        if !self.ptr.is_null() && set_device(self.ordinal).is_ok() {
            let _ = unsafe { sys::hipHostFree(self.ptr as *mut c_void) };
        }
    }
}

/// Completion of a copy queued on a [`CopyStream`].
pub struct CopyEvent {
    event: Arc<RawEvent>,
    device: RocmDevice,
    waited: AtomicBool,
}

impl std::fmt::Debug for CopyEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CopyEvent(complete: {})", self.event.is_complete())
    }
}

impl CopyEvent {
    /// Makes the compute stream of `dev` wait (on the GPU, the host does not block) for the copy:
    /// work queued on `dev` afterwards sees the uploaded bytes, and may overwrite or free the
    /// source of a download.
    pub fn wait_on_compute(&self, dev: &RocmDevice) -> Result<()> {
        dev.set_current()?;
        check_hip(unsafe { sys::hipStreamWaitEvent(COMPUTE, self.event.event, 0) }).w()?;
        self.waited.store(true, Ordering::Release);
        Ok(())
    }

    /// Blocks the calling thread until the copy completed.
    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()?;
        self.waited.store(true, Ordering::Release);
        Ok(())
    }

    /// Whether the copy has completed (non-blocking).
    pub fn is_complete(&self) -> bool {
        self.event.is_complete()
    }
}

impl Drop for CopyEvent {
    fn drop(&mut self) {
        if !self.waited.load(Ordering::Acquire) {
            // Orders what the compute stream does from now on (frees, buffer reuse) after the copy.
            let _ = self.wait_on_compute(&self.device);
        }
    }
}

/// The copy stream of a device, see the module documentation. Cheap to clone; all clones (and
/// all calls to [`RocmDevice::copy_stream`]) share one stream, so copies run in issue order.
#[derive(Clone)]
pub struct CopyStream {
    stream: Arc<RawStream>,
    device: RocmDevice,
}

impl std::fmt::Debug for CopyStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CopyStream({:?})", self.device)
    }
}

impl CopyStream {
    fn check(&self, storage: &RocmStorage, what: &str) -> Result<()> {
        if storage.device.ordinal() != self.device.ordinal() {
            bail!(
                "copy stream: the {what} lives on ROCm device {} but the copy stream on device {}",
                storage.device.ordinal(),
                self.device.ordinal()
            )
        }
        Ok(())
    }

    /// Makes the copy stream wait for the work queued so far on the compute (NULL) stream.
    fn after_compute(&self) -> Result<()> {
        let ev = RawEvent::new(self.stream.ordinal)?;
        ev.record(COMPUTE)?;
        check_hip(unsafe { sys::hipStreamWaitEvent(self.stream.stream, ev.event, 0) }).w()
    }

    fn finish(&self, host: &PinnedHostBuffer) -> Result<CopyEvent> {
        let ev = Arc::new(RawEvent::new(self.stream.ordinal)?);
        ev.record(self.stream.stream)?;
        host.track(&ev);
        Ok(CopyEvent {
            event: ev,
            device: self.device.clone(),
            waited: AtomicBool::new(false),
        })
    }

    /// Queues the copy of `src[src_off..src_off + bytes]` to `dst` at byte offset `dst_off`. The
    /// copy starts after the work queued so far on the compute stream; the returned event must be
    /// waited on by the compute stream ([`CopyEvent::wait_on_compute`]) before reading `dst`.
    pub fn upload(
        &self,
        src: &PinnedHostBuffer,
        src_off: usize,
        dst: &RocmStorage,
        dst_off: usize,
        bytes: usize,
    ) -> Result<CopyEvent> {
        self.check(dst, "destination")?;
        let dst_len = dst.buf.byte_size();
        if src_off.checked_add(bytes).map_or(true, |end| end > src.len) {
            bail!(
                "copy stream upload: source range {src_off}+{bytes} exceeds the {} byte pinned buffer",
                src.len
            )
        }
        if dst_off.checked_add(bytes).map_or(true, |end| end > dst_len) {
            bail!("copy stream upload: destination range {dst_off}+{bytes} exceeds {dst_len} bytes")
        }
        set_device(self.stream.ordinal)?;
        self.after_compute()?;
        if bytes > 0 {
            check_hip(unsafe {
                sys::hipMemcpyAsync(
                    (dst.buf.as_void_ptr() as *mut u8).add(dst_off) as *mut c_void,
                    src.ptr.add(src_off) as *const c_void,
                    bytes,
                    sys::hipMemcpyKind::hipMemcpyHostToDevice,
                    self.stream.stream,
                )
            })
            .w()?;
        }
        self.finish(src)
    }

    /// Queues the copy of `bytes` bytes of `src` from byte offset `src_off` into
    /// `dst[dst_off..dst_off + bytes]`. The copy starts after the work queued so far on the
    /// compute stream (so it sees the results of earlier kernels). Read `dst` after
    /// [`CopyEvent::synchronize`] (or through [`PinnedHostBuffer::as_slice`], which waits).
    pub fn download(
        &self,
        src: &RocmStorage,
        src_off: usize,
        dst: &mut PinnedHostBuffer,
        dst_off: usize,
        bytes: usize,
    ) -> Result<CopyEvent> {
        self.check(src, "source")?;
        let src_len = src.buf.byte_size();
        if src_off.checked_add(bytes).map_or(true, |end| end > src_len) {
            bail!("copy stream download: source range {src_off}+{bytes} exceeds {src_len} bytes")
        }
        if dst_off.checked_add(bytes).map_or(true, |end| end > dst.len) {
            bail!(
                "copy stream download: destination range {dst_off}+{bytes} exceeds the {} byte pinned buffer",
                dst.len
            )
        }
        set_device(self.stream.ordinal)?;
        self.after_compute()?;
        if bytes > 0 {
            check_hip(unsafe {
                sys::hipMemcpyAsync(
                    dst.ptr.add(dst_off) as *mut c_void,
                    (src.buf.as_void_ptr() as *const u8).add(src_off) as *const c_void,
                    bytes,
                    sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                    self.stream.stream,
                )
            })
            .w()?;
        }
        self.finish(dst)
    }

    /// Blocks until every copy queued so far on this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        set_device(self.stream.ordinal)?;
        check_hip(unsafe { sys::hipStreamSynchronize(self.stream.stream) }).w()
    }
}

impl RocmDevice {
    /// Allocates `bytes` of page-locked host memory for asynchronous copies.
    pub fn alloc_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
        PinnedHostBuffer::new(self.ordinal(), bytes)
    }

    /// The copy stream of this device (created on first use and shared by all callers).
    pub fn copy_stream(&self) -> Result<CopyStream> {
        Ok(CopyStream {
            stream: self.raw_copy_stream()?,
            device: self.clone(),
        })
    }
}
