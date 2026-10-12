//! Pinned host buffers and an asynchronous copy stream, to overlap host <-> device transfers of
//! offloaded data (RAM/SSD-resident KV cache blocks, layer weights) with compute.
//!
//! The copy stream is a raw non-blocking CUDA stream created with driver calls, so cudarc stays in
//! single-stream mode (no per-slice event tracking); the ordering between the copy stream and
//! the device's compute stream is done here with CUDA events:
//!
//! - [`CopyStream::upload`] / [`CopyStream::download`] first make the copy stream wait for the
//!   work queued so far on the compute stream. This covers the stream-ordered allocation of the
//!   device buffer (`cuMemAllocAsync` memory is only valid on other streams after such a wait) and
//!   earlier kernels that still read or write the buffer.
//! - They return a [`CopyEvent`]. Call [`CopyEvent::wait_on_compute`] before queuing compute
//!   work that reads the uploaded bytes (or that overwrites / frees the source of a download), and
//!   [`CopyEvent::synchronize`] before reading downloaded bytes on the host.
//!
//! Typical prefetch with a staging slot (tensors from `Tensor::zeros(.., &device)`, whose storage
//! is reached through `Tensor::storage_and_layout`):
//!
//! ```ignore
//! let copy = cuda_dev.copy_stream()?;
//! let mut host = cuda_dev.alloc_pinned(bytes)?;          // once
//! host.as_mut_slice()[..n].copy_from_slice(&kv_bytes);    // waits for earlier copies of `host`
//! let ev = {
//!     let (storage, _) = slot.storage_and_layout();
//!     let candle_core::Storage::Cuda(dst) = &*storage else { unreachable!() };
//!     copy.upload(&host, 0, dst, 0, n)?
//! };
//! // ... queue other layers: they run while the DMA engine copies ...
//! ev.wait_on_compute(&cuda_dev)?;                        // before the kernels reading `slot`
//! ```
//!
//! The destination of an upload (source of a download) must stay alive until
//! [`CopyEvent::wait_on_compute`] was called for it: a buffer freed before that may be handed to
//! another allocation while the copy still runs. Dropping a [`CopyEvent`] that was never waited
//! on queues that wait on the compute stream, which orders frees issued afterwards.

use super::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::{bail, Result};
use cudarc::driver::{result as drv, sys, CudaContext, DevicePtr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::device::bind_context_raw;

/// A CUDA event created and destroyed with raw driver calls.
pub(crate) struct RawEvent {
    event: sys::CUevent,
    ctx: Arc<CudaContext>,
}

// CUDA events can be used from any thread.
unsafe impl Send for RawEvent {}
unsafe impl Sync for RawEvent {}

impl RawEvent {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        bind_context_raw(ctx).w()?;
        let event = drv::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).w()?;
        Ok(Self {
            event,
            ctx: ctx.clone(),
        })
    }

    fn record(&self, stream: sys::CUstream) -> Result<()> {
        unsafe { drv::event::record(self.event, stream) }.w()
    }

    fn is_complete(&self) -> bool {
        unsafe { drv::event::query(self.event) }.is_ok()
    }

    fn synchronize(&self) -> Result<()> {
        bind_context_raw(&self.ctx).w()?;
        unsafe { drv::event::synchronize(self.event) }.w()
    }
}

impl Drop for RawEvent {
    fn drop(&mut self) {
        // Does not block: the driver releases the event once the recorded work completed.
        if bind_context_raw(&self.ctx).is_ok() {
            let _ = unsafe { drv::event::destroy(self.event) };
        }
    }
}

/// A raw non-blocking stream (`cuStreamCreate(CU_STREAM_NON_BLOCKING)`), invisible to cudarc.
pub(crate) struct RawStream {
    stream: sys::CUstream,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

impl RawStream {
    pub(crate) fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        bind_context_raw(&ctx).w()?;
        let stream = drv::stream::create(drv::stream::StreamKind::NonBlocking).w()?;
        Ok(Self { stream, ctx })
    }
}

impl Drop for RawStream {
    fn drop(&mut self) {
        // Does not block: queued work completes before the driver releases the stream.
        if bind_context_raw(&self.ctx).is_ok() {
            let _ = unsafe { drv::stream::destroy(self.stream) };
        }
    }
}

/// Page-locked host memory (`cuMemHostAlloc(CU_MEMHOSTALLOC_PORTABLE)`), usable with the copy
/// streams of every device. Accessing the bytes ([`Self::as_slice`], [`Self::as_mut_slice`])
/// first waits for the copies queued on this buffer, so a reused staging buffer is never
/// overwritten while the DMA engine still reads it, and downloaded bytes are complete.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
    ctx: Arc<CudaContext>,
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
    fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        let ptr = if len == 0 {
            std::ptr::null_mut()
        } else {
            bind_context_raw(ctx).w()?;
            unsafe { drv::malloc_host(len, sys::CU_MEMHOSTALLOC_PORTABLE) }.w()? as *mut u8
        };
        Ok(Self {
            ptr,
            len,
            ctx: ctx.clone(),
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
            eprintln!("candle-cuda: waiting for the copies of a pinned buffer failed: {e}");
        }
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The bytes, after the copies queued on this buffer completed.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if let Err(e) = self.wait_idle() {
            eprintln!("candle-cuda: waiting for the copies of a pinned buffer failed: {e}");
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
        if !self.ptr.is_null() && bind_context_raw(&self.ctx).is_ok() {
            let _ = unsafe { drv::free_host(self.ptr as *mut std::ffi::c_void) };
        }
    }
}

/// Completion of a copy queued on a [`CopyStream`].
pub struct CopyEvent {
    event: Arc<RawEvent>,
    device: CudaDevice,
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
    pub fn wait_on_compute(&self, dev: &CudaDevice) -> Result<()> {
        dev.bind_to_thread()?;
        unsafe {
            drv::stream::wait_event(
                dev.cuda_stream().cu_stream(),
                self.event.event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .w()?;
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
            // Orders the frees the compute stream issues from now on after the copy.
            let _ = self.wait_on_compute(&self.device.clone());
        }
    }
}

/// The copy stream of a device, see the module documentation. Cheap to clone; all clones (and
/// all calls to [`CudaDevice::copy_stream`]) share one stream, so copies run in issue order.
#[derive(Clone)]
pub struct CopyStream {
    stream: Arc<RawStream>,
    device: CudaDevice,
}

impl std::fmt::Debug for CopyStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CopyStream({:?})", self.device)
    }
}

/// Raw device address and size in bytes of the buffer of `s`.
fn storage_ptr(s: &CudaStorage) -> (sys::CUdeviceptr, usize) {
    macro_rules! p {
        ($x:expr) => {{
            let (ptr, _guard) = $x.device_ptr($x.stream());
            (ptr, $x.num_bytes())
        }};
    }
    match &s.slice {
        CudaStorageSlice::U8(x) => p!(x),
        CudaStorageSlice::U32(x) => p!(x),
        CudaStorageSlice::I16(x) => p!(x),
        CudaStorageSlice::I32(x) => p!(x),
        CudaStorageSlice::I64(x) => p!(x),
        CudaStorageSlice::BF16(x) => p!(x),
        CudaStorageSlice::F16(x) => p!(x),
        CudaStorageSlice::F32(x) => p!(x),
        CudaStorageSlice::F64(x) => p!(x),
        CudaStorageSlice::F8E4M3(x) => p!(x),
        CudaStorageSlice::F6E2M3(x)
        | CudaStorageSlice::F6E3M2(x)
        | CudaStorageSlice::F4(x)
        | CudaStorageSlice::F8E8M0(x) => p!(x),
    }
}

impl CopyStream {
    fn check(&self, storage: &CudaStorage, what: &str) -> Result<()> {
        if storage.device.cuda_context().cu_ctx() != self.device.cuda_context().cu_ctx() {
            bail!("copy stream: the {what} lives on another CUDA context than the copy stream")
        }
        Ok(())
    }

    /// Makes the copy stream wait for the work queued so far on the compute stream.
    fn after_compute(&self) -> Result<()> {
        let ev = RawEvent::new(&self.stream.ctx)?;
        ev.record(self.device.cuda_stream().cu_stream())?;
        unsafe {
            drv::stream::wait_event(
                self.stream.stream,
                ev.event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .w()
    }

    fn finish(&self, host: &PinnedHostBuffer) -> Result<CopyEvent> {
        let ev = Arc::new(RawEvent::new(&self.stream.ctx)?);
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
        dst: &CudaStorage,
        dst_off: usize,
        bytes: usize,
    ) -> Result<CopyEvent> {
        self.check(dst, "destination")?;
        let (dst_ptr, dst_len) = storage_ptr(dst);
        if src_off.checked_add(bytes).map_or(true, |end| end > src.len) {
            bail!(
                "copy stream upload: source range {src_off}+{bytes} exceeds the {} byte pinned buffer",
                src.len
            )
        }
        if dst_off.checked_add(bytes).map_or(true, |end| end > dst_len) {
            bail!("copy stream upload: destination range {dst_off}+{bytes} exceeds {dst_len} bytes")
        }
        bind_context_raw(&self.stream.ctx).w()?;
        self.after_compute()?;
        if bytes > 0 {
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst_ptr + dst_off as u64,
                    src.ptr.add(src_off) as *const std::ffi::c_void,
                    bytes,
                    self.stream.stream,
                )
                .result()
            }
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
        src: &CudaStorage,
        src_off: usize,
        dst: &mut PinnedHostBuffer,
        dst_off: usize,
        bytes: usize,
    ) -> Result<CopyEvent> {
        self.check(src, "source")?;
        let (src_ptr, src_len) = storage_ptr(src);
        if src_off.checked_add(bytes).map_or(true, |end| end > src_len) {
            bail!("copy stream download: source range {src_off}+{bytes} exceeds {src_len} bytes")
        }
        if dst_off.checked_add(bytes).map_or(true, |end| end > dst.len) {
            bail!(
                "copy stream download: destination range {dst_off}+{bytes} exceeds the {} byte pinned buffer",
                dst.len
            )
        }
        bind_context_raw(&self.stream.ctx).w()?;
        self.after_compute()?;
        if bytes > 0 {
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    dst.ptr.add(dst_off) as *mut std::ffi::c_void,
                    src_ptr + src_off as u64,
                    bytes,
                    self.stream.stream,
                )
                .result()
            }
            .w()?;
        }
        self.finish(dst)
    }

    /// Blocks until every copy queued so far on this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        bind_context_raw(&self.stream.ctx).w()?;
        unsafe { drv::stream::synchronize(self.stream.stream) }.w()
    }
}

impl CudaDevice {
    /// Allocates `bytes` of page-locked host memory for asynchronous copies.
    pub fn alloc_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
        PinnedHostBuffer::new(self.cuda_context(), bytes)
    }

    /// The copy stream of this device (created on first use and shared by all callers).
    pub fn copy_stream(&self) -> Result<CopyStream> {
        Ok(CopyStream {
            stream: self.raw_copy_stream()?,
            device: self.clone(),
        })
    }
}
