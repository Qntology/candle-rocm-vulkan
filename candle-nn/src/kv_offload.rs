//! Double-buffered asynchronous host-to-device offloading engine for compressed TurboQuant KV caches.
//!
//! Uses 2 persistent GPU staging slots (`slot 0` and `slot 1`) plus backend `CopyStream` and
//! page-locked [`PinnedKvSlab`] buffers on ROCm and CUDA so layer `L + 1` (or `L + 2`) compressed
//! KV slabs transfer over PCIe DMA while the compute stream executes layer `L`.
//!
//! Small transfers (`< ASYNC_DMA_MIN_BYTES`, 4 MiB) bypass async pinned DMA to avoid the measured
//! small-transfer pinned regression on Windows/Linux HIP/CUDA drivers.

use crate::flash_attn::FlashAttnConfig;
use crate::turbo_quant::{attention_packed, TurboQuantizer};
use candle::{bail, DType, Device, Result, Tensor};

/// Minimum combined K+V slab byte size required to route transfers through the non-blocking
/// pinned `CopyStream`. Smaller transfers use direct host-to-device staging to avoid small-copy
/// pinned overhead.
pub const ASYNC_DMA_MIN_BYTES: usize = 4 * 1024 * 1024;

/// Host-side compressed K/V slab backed by page-locked pinned memory on ROCm/CUDA when allocated
/// for a GPU device, or aligned host `Vec<u8>` on CPU/Vulkan/Metal.
enum HostBuf {
    Vec(Vec<u8>),
    #[cfg(feature = "rocm")]
    RocmPinned(candle::rocm_backend::offload::PinnedHostBuffer),
    #[cfg(feature = "cuda")]
    CudaPinned(candle::cuda_backend::offload::PinnedHostBuffer),
}

impl HostBuf {
    #[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(unused_variables))]
    fn alloc(device: &Device, bytes: usize, prefer_pinned: bool) -> Result<Self> {
        if prefer_pinned && bytes > 0 {
            #[cfg(feature = "rocm")]
            if let Device::Rocm(r) = device {
                if let Ok(p) = r.alloc_pinned(bytes) {
                    return Ok(Self::RocmPinned(p));
                }
            }
            #[cfg(feature = "cuda")]
            if let Device::Cuda(c) = device {
                if let Ok(p) = c.alloc_pinned(bytes) {
                    return Ok(Self::CudaPinned(p));
                }
            }
        }
        Ok(Self::Vec(vec![0u8; bytes]))
    }

    fn len(&self) -> usize {
        match self {
            Self::Vec(v) => v.len(),
            #[cfg(feature = "rocm")]
            Self::RocmPinned(p) => p.len(),
            #[cfg(feature = "cuda")]
            Self::CudaPinned(p) => p.len(),
        }
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Vec(v) => v.as_slice(),
            #[cfg(feature = "rocm")]
            Self::RocmPinned(p) => p.as_slice(),
            #[cfg(feature = "cuda")]
            Self::CudaPinned(p) => p.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Vec(v) => v.as_mut_slice(),
            #[cfg(feature = "rocm")]
            Self::RocmPinned(p) => p.as_mut_slice(),
            #[cfg(feature = "cuda")]
            Self::CudaPinned(p) => p.as_mut_slice(),
        }
    }
}

/// A compressed KV slab (`u32` TurboQuant packed rows) stored on the host (page-locked when
/// `bytes >= ASYNC_DMA_MIN_BYTES` on ROCm/CUDA).
pub struct PinnedKvSlab {
    k_buf: HostBuf,
    v_buf: HostBuf,
    pub batch: usize,
    pub hkv: usize,
    pub kv_len: usize,
    pub k_words: usize,
    pub v_words: usize,
}

impl std::fmt::Debug for PinnedKvSlab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedKvSlab")
            .field("batch", &self.batch)
            .field("hkv", &self.hkv)
            .field("kv_len", &self.kv_len)
            .field("k_words", &self.k_words)
            .field("v_words", &self.v_words)
            .field("bytes", &self.total_bytes())
            .finish()
    }
}

impl PinnedKvSlab {
    /// Packs host/device `u32` tensors `k_rows: (b, hkv, kv_len, k_words)` and
    /// `v_rows: (b, hkv, kv_len, v_words)` into a host slab (page-locked when large enough on
    /// ROCm/CUDA for zero-bounce async DMA).
    pub fn from_packed_tensors(
        k_rows: &Tensor,
        v_rows: &Tensor,
        target_device: &Device,
    ) -> Result<Self> {
        let (b, hkv, kv_len, k_words) = k_rows.dims4()?;
        let (bv, hv, lv, v_words) = v_rows.dims4()?;
        if k_rows.dtype() != DType::U32 || v_rows.dtype() != DType::U32 {
            bail!("PinnedKvSlab: packed K/V rows must have DType::U32");
        }
        if (bv, hv, lv) != (b, hkv, kv_len) {
            bail!(
                "PinnedKvSlab: K {:?} and V {:?} shape mismatch",
                k_rows.dims(),
                v_rows.dims()
            );
        }
        let k_u32 = k_rows
            .contiguous()?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let v_u32 = v_rows
            .contiguous()?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        Self::from_u32_slices(&k_u32, &v_u32, b, hkv, kv_len, k_words, v_words, target_device)
    }

    /// Creates a host slab directly from contiguous `u32` packed slices.
    #[allow(clippy::too_many_arguments)]
    pub fn from_u32_slices(
        k_u32: &[u32],
        v_u32: &[u32],
        batch: usize,
        hkv: usize,
        kv_len: usize,
        k_words: usize,
        v_words: usize,
        target_device: &Device,
    ) -> Result<Self> {
        let k_elems = batch * hkv * kv_len * k_words;
        let v_elems = batch * hkv * kv_len * v_words;
        if k_u32.len() != k_elems || v_u32.len() != v_elems {
            bail!(
                "PinnedKvSlab: slice lengths ({}, {}) != expected ({k_elems}, {v_elems})",
                k_u32.len(),
                v_u32.len()
            );
        }
        let k_bytes = k_elems * 4;
        let v_bytes = v_elems * 4;
        let prefer_pinned = k_bytes + v_bytes >= ASYNC_DMA_MIN_BYTES;
        let mut k_buf = HostBuf::alloc(target_device, k_bytes, prefer_pinned)?;
        let mut v_buf = HostBuf::alloc(target_device, v_bytes, prefer_pinned)?;
        if k_bytes > 0 {
            let src = unsafe { std::slice::from_raw_parts(k_u32.as_ptr() as *const u8, k_bytes) };
            k_buf.as_mut_slice().copy_from_slice(src);
        }
        if v_bytes > 0 {
            let src = unsafe { std::slice::from_raw_parts(v_u32.as_ptr() as *const u8, v_bytes) };
            v_buf.as_mut_slice().copy_from_slice(src);
        }
        Ok(Self {
            k_buf,
            v_buf,
            batch,
            hkv,
            kv_len,
            k_words,
            v_words,
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.k_buf.len() + self.v_buf.len()
    }

    pub fn is_pinned(&self) -> bool {
        !matches!(self.k_buf, HostBuf::Vec(_))
    }

    fn k_u32_slice(&self) -> &[u32] {
        let bytes = self.k_buf.as_slice();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
    }

    fn v_u32_slice(&self) -> &[u32] {
        let bytes = self.v_buf.as_slice();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
    }
}

enum InflightEvent {
    #[cfg(feature = "rocm")]
    Rocm(
        candle::rocm_backend::offload::CopyEvent,
        candle::rocm_backend::offload::CopyEvent,
    ),
    #[cfg(feature = "cuda")]
    Cuda(
        candle::cuda_backend::offload::CopyEvent,
        candle::cuda_backend::offload::CopyEvent,
    ),
    None,
}

struct StagingSlot {
    k_tensor: Tensor,
    v_tensor: Tensor,
    active_shape: Option<(usize, usize, usize, usize, usize)>, // (b, hkv, kv_len, kw, vw)
    compact_async: bool,
    inflight: InflightEvent,
}

impl StagingSlot {
    fn new(
        device: &Device,
        batch: usize,
        hkv: usize,
        max_kv_len: usize,
        k_words: usize,
        v_words: usize,
    ) -> Result<Self> {
        let k_tensor = Tensor::zeros((batch, hkv, max_kv_len, k_words), DType::U32, device)?;
        let v_tensor = Tensor::zeros((batch, hkv, max_kv_len, v_words), DType::U32, device)?;
        Ok(Self {
            k_tensor,
            v_tensor,
            active_shape: None,
            compact_async: false,
            inflight: InflightEvent::None,
        })
    }

    #[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(unused_variables))]
    fn wait_on_compute(&mut self, device: &Device) -> Result<()> {
        let ev = std::mem::replace(&mut self.inflight, InflightEvent::None);
        match ev {
            InflightEvent::None => Ok(()),
            #[cfg(feature = "rocm")]
            InflightEvent::Rocm(ek, ev) => {
                if let Device::Rocm(r) = device {
                    ek.wait_on_compute(r)?;
                    ev.wait_on_compute(r)?;
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            InflightEvent::Cuda(ek, ev) => {
                if let Device::Cuda(c) = device {
                    ek.wait_on_compute(c)?;
                    ev.wait_on_compute(c)?;
                }
                Ok(())
            }
        }
    }

    fn synchronize_host(&mut self) -> Result<()> {
        let ev = std::mem::replace(&mut self.inflight, InflightEvent::None);
        match ev {
            InflightEvent::None => Ok(()),
            #[cfg(feature = "rocm")]
            InflightEvent::Rocm(ek, ev) => {
                ek.synchronize()?;
                ev.synchronize()?;
                Ok(())
            }
            #[cfg(feature = "cuda")]
            InflightEvent::Cuda(ek, ev) => {
                ek.synchronize()?;
                ev.synchronize()?;
                Ok(())
            }
        }
    }
}

/// 2-slot ping-pong staging engine for overlapping PCIe transfers of offloaded compressed KV slabs
/// with layer attention / MLP compute.
pub struct OffloadedTurboKvEngine {
    device: Device,
    batch: usize,
    hkv: usize,
    max_kv_len: usize,
    k_words: usize,
    v_words: usize,
    slots: [StagingSlot; 2],
    #[cfg(feature = "rocm")]
    rocm_copy: Option<candle::rocm_backend::offload::CopyStream>,
    #[cfg(feature = "cuda")]
    cuda_copy: Option<candle::cuda_backend::offload::CopyStream>,
    async_min_bytes: usize,
}

impl OffloadedTurboKvEngine {
    /// Allocates 2 persistent GPU staging slots of shape `(batch, hkv, max_kv_len, {k,v}_words)`.
    pub fn new(
        device: &Device,
        batch: usize,
        hkv: usize,
        max_kv_len: usize,
        k_words: usize,
        v_words: usize,
    ) -> Result<Self> {
        let s0 = StagingSlot::new(device, batch, hkv, max_kv_len, k_words, v_words)?;
        let s1 = StagingSlot::new(device, batch, hkv, max_kv_len, k_words, v_words)?;
        #[cfg(feature = "rocm")]
        let rocm_copy = match device {
            Device::Rocm(r) => Some(r.copy_stream()?),
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let cuda_copy = match device {
            Device::Cuda(c) => Some(c.copy_stream()?),
            _ => None,
        };
        Ok(Self {
            device: device.clone(),
            batch,
            hkv,
            max_kv_len,
            k_words,
            v_words,
            slots: [s0, s1],
            #[cfg(feature = "rocm")]
            rocm_copy,
            #[cfg(feature = "cuda")]
            cuda_copy,
            async_min_bytes: ASYNC_DMA_MIN_BYTES,
        })
    }

    /// Overrides the async DMA threshold (primarily for testing).
    pub fn with_async_min_bytes(mut self, min_bytes: usize) -> Self {
        self.async_min_bytes = min_bytes;
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Queues an upload of `slab` into staging slot `slot_idx % 2`.
    ///
    /// When `slab` is backed by pinned memory and `slab.total_bytes() >= async_min_bytes` on
    /// ROCm/CUDA, the transfer is launched asynchronously on the device `CopyStream` and overlaps
    /// with kernels running on the compute stream. Otherwise it stages directly onto `device`.
    pub fn prefetch_slab(&mut self, slot_idx: usize, slab: &PinnedKvSlab) -> Result<()> {
        if slab.batch != self.batch
            || slab.hkv != self.hkv
            || slab.kv_len > self.max_kv_len
            || slab.k_words != self.k_words
            || slab.v_words != self.v_words
        {
            bail!(
                "OffloadedTurboKvEngine: slab ({}, {}, {}, {}, {}) exceeds slot ({}, {}, {}, {}, {})",
                slab.batch,
                slab.hkv,
                slab.kv_len,
                slab.k_words,
                slab.v_words,
                self.batch,
                self.hkv,
                self.max_kv_len,
                self.k_words,
                self.v_words
            );
        }
        let slot = &mut self.slots[slot_idx & 1];
        slot.synchronize_host()?;
        slot.active_shape = Some((
            slab.batch,
            slab.hkv,
            slab.kv_len,
            slab.k_words,
            slab.v_words,
        ));

        let use_async = slab.total_bytes() >= self.async_min_bytes;

        #[cfg(feature = "rocm")]
        if use_async {
            if let (
                Some(copy),
                HostBuf::RocmPinned(kp),
                HostBuf::RocmPinned(vp),
            ) = (&self.rocm_copy, &slab.k_buf, &slab.v_buf)
            {
                let (ks, _) = slot.k_tensor.storage_and_layout();
                let (vs, _) = slot.v_tensor.storage_and_layout();
                if let (candle::Storage::Rocm(kd), candle::Storage::Rocm(vd)) = (&*ks, &*vs) {
                    let ek = copy.upload(kp, 0, kd, 0, kp.len())?;
                    let ev = copy.upload(vp, 0, vd, 0, vp.len())?;
                    slot.compact_async = true;
                    slot.inflight = InflightEvent::Rocm(ek, ev);
                    return Ok(());
                }
            }
        }

        #[cfg(feature = "cuda")]
        if use_async {
            if let (
                Some(copy),
                HostBuf::CudaPinned(kp),
                HostBuf::CudaPinned(vp),
            ) = (&self.cuda_copy, &slab.k_buf, &slab.v_buf)
            {
                let (ks, _) = slot.k_tensor.storage_and_layout();
                let (vs, _) = slot.v_tensor.storage_and_layout();
                if let (candle::Storage::Cuda(kd), candle::Storage::Cuda(vd)) = (&*ks, &*vs) {
                    let ek = copy.upload(kp, 0, kd, 0, kp.len())?;
                    let ev = copy.upload(vp, 0, vd, 0, vp.len())?;
                    slot.compact_async = true;
                    slot.inflight = InflightEvent::Cuda(ek, ev);
                    return Ok(());
                }
            }
        }

        let _ = use_async;
        slot.compact_async = false;
        // Direct staging fallback (CPU / Vulkan / Metal / small transfers below ASYNC_DMA_MIN_BYTES).
        let k_src = Tensor::from_slice(
            slab.k_u32_slice(),
            (slab.batch, slab.hkv, slab.kv_len, slab.k_words),
            &self.device,
        )?;
        let v_src = Tensor::from_slice(
            slab.v_u32_slice(),
            (slab.batch, slab.hkv, slab.kv_len, slab.v_words),
            &self.device,
        )?;
        if slab.kv_len == self.max_kv_len {
            slot.k_tensor = k_src;
            slot.v_tensor = v_src;
        } else {
            slot.k_tensor.slice_set(&k_src, 2, 0)?;
            slot.v_tensor.slice_set(&v_src, 2, 0)?;
        }
        Ok(())
    }

    /// Waits on the GPU compute stream (non-blocking on host) for `slot_idx % 2` to finish
    /// uploading and runs [`attention_packed`] directly over the staged compressed KV rows.
    pub fn attend_slot(
        &mut self,
        slot_idx: usize,
        q: &Tensor,
        kq: &TurboQuantizer,
        vq: &TurboQuantizer,
        cfg: &FlashAttnConfig,
    ) -> Result<Tensor> {
        let slot = &mut self.slots[slot_idx & 1];
        let Some((b, hkv, kv_len, kw, vw)) = slot.active_shape else {
            bail!("OffloadedTurboKvEngine: slot {} has no staged slab", slot_idx & 1);
        };
        slot.wait_on_compute(&self.device)?;
        let (k_view, v_view) = if kv_len == self.max_kv_len {
            (slot.k_tensor.clone(), slot.v_tensor.clone())
        } else if slot.compact_async {
            // CopyStream::upload wrote compact `(b, hkv, kv_len, words)` bytes at offset 0.
            let k_elems = b * hkv * kv_len * kw;
            let v_elems = b * hkv * kv_len * vw;
            (
                slot.k_tensor
                    .flatten_all()?
                    .narrow(0, 0, k_elems)?
                    .reshape((b, hkv, kv_len, kw))?,
                slot.v_tensor
                    .flatten_all()?
                    .narrow(0, 0, v_elems)?
                    .reshape((b, hkv, kv_len, vw))?,
            )
        } else {
            (
                slot.k_tensor.narrow(2, 0, kv_len)?,
                slot.v_tensor.narrow(2, 0, kv_len)?,
            )
        };
        attention_packed(q, &k_view, &v_view, kv_len, kq, vq, cfg)
    }

    /// Convenience helper for multi-layer models: runs `attend` across `slabs` with 1-layer
    /// lookahead ping-pong prefetching (`slot 0` / `slot 1`), calling `q_for_layer(i)` per layer.
    pub fn run_layers<F>(
        &mut self,
        slabs: &[PinnedKvSlab],
        kq: &TurboQuantizer,
        vq: &TurboQuantizer,
        cfg: &FlashAttnConfig,
        mut q_for_layer: F,
    ) -> Result<Vec<Tensor>>
    where
        F: FnMut(usize) -> Result<Tensor>,
    {
        if slabs.is_empty() {
            return Ok(Vec::new());
        }
        let mut outs = Vec::with_capacity(slabs.len());
        self.prefetch_slab(0, &slabs[0])?;
        for i in 0..slabs.len() {
            if i + 1 < slabs.len() {
                self.prefetch_slab(i + 1, &slabs[i + 1])?;
            }
            let q = q_for_layer(i)?;
            let out = self.attend_slot(i, &q, kq, vq, cfg)?;
            outs.push(out);
        }
        Ok(outs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::accel::{QuantFormat, RotationKind};

    #[test]
    fn offloaded_turbo_kv_engine_matches_direct_cpu() -> Result<()> {
        let dev = Device::Cpu;
        let (b, hq, hkv, lq, lk, d) = (1, 4, 2, 2, 16, 64);
        let kq = TurboQuantizer::new(QuantFormat::Nvfp4, d, RotationKind::Hadamard, 11, &dev)?;
        let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::Hadamard, 13, &dev)?;

        let q = Tensor::randn(0f32, 0.5, (b, hq, lq, d), &dev)?;
        let k0 = Tensor::randn(0f32, 0.5, (b, hkv, lk, d), &dev)?;
        let v0 = Tensor::randn(0f32, 0.5, (b, hkv, lk, d), &dev)?;
        let k1 = Tensor::randn(0f32, 0.5, (b, hkv, lk, d), &dev)?;
        let v1 = Tensor::randn(0f32, 0.5, (b, hkv, lk, d), &dev)?;

        let pack = |qz: &TurboQuantizer, x: &Tensor| -> Result<Tensor> {
            let flat = x.reshape((b * hkv * lk, d))?;
            qz.quantize_rows(&flat)?.reshape((b, hkv, lk, qz.row_words()))
        };
        let kp0 = pack(&kq, &k0)?;
        let vp0 = pack(&vq, &v0)?;
        let kp1 = pack(&kq, &k1)?;
        let vp1 = pack(&vq, &v1)?;

        let slab0 = PinnedKvSlab::from_packed_tensors(&kp0, &vp0, &dev)?;
        let slab1 = PinnedKvSlab::from_packed_tensors(&kp1, &vp1, &dev)?;

        let cfg = FlashAttnConfig::new(1.0 / (d as f32).sqrt()).causal();
        let ref0 = attention_packed(&q, &kp0, &vp0, lk, &kq, &vq, &cfg)?;
        let ref1 = attention_packed(&q, &kp1, &vp1, lk, &kq, &vq, &cfg)?;

        let mut engine = OffloadedTurboKvEngine::new(&dev, b, hkv, lk, kq.row_words(), vq.row_words())?;
        let outs = engine.run_layers(&[slab0, slab1], &kq, &vq, &cfg, |_| Ok(q.clone()))?;
        assert_eq!(outs.len(), 2);
        let d0 = (&outs[0] - &ref0)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let d1 = (&outs[1] - &ref1)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(d0 <= 1e-6, "layer 0 diff {d0}");
        assert!(d1 <= 1e-6, "layer 1 diff {d1}");
        Ok(())
    }
}
