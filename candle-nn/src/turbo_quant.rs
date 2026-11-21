//! TurboQuant and microscaling quantization of vectors and KV caches on CUDA, ROCm, Vulkan and
//! the CPU.
//!
//! [`TurboQuantizer`] packs vectors of `dim` values into `u32` rows in one of the
//! [`QuantFormat`]s:
//!
//! * TurboQuant (Zandieh et al. 2025, reference: github.com/abdelstark/turboquant): a random
//!   rotation, then a Lloyd-Max codebook (`TurboMse`, 1-4 bits) or a codebook of one bit less
//!   plus a 1-bit QJL sketch of the residual (`TurboProd`, 2-5 bits, unbiased inner products);
//! * NVFP4, MXFP4, MXFP8 and per-vector FP8, optionally after the same rotation (incoherent
//!   processing, which tames outliers before FP4 / FP8 rounding).
//!
//! [`TurboKvCache`] keeps K/V compressed and computes attention directly on the packed rows
//! (`tq_attn` kernels, flash-decoding style), or dequantizes and runs
//! [`crate::flash_attn`] when the fused path is off.
//!
//! The formats, bits, rotation, QJL stage, fused attention and backends follow the global
//! switches of `candle::accel::config()` (`CANDLE_TURBO_QUANT*` environment variables) unless
//! given explicitly. When nothing is set the formats are picked per GPU: the native 4-bit / 8-bit
//! format of the GPU when it has one (NVFP4 on Blackwell, MXFP4 on MI350, ...), TurboQuant
//! codebooks otherwise.

use crate::flash_attn::{cat_blocks, check_blocks, flash_attn_bhsd, FlashAttnConfig};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::flash_attn::paged_block_layout;
pub use crate::flash_attn::KvBlockRef;
use crate::kv_cache::Cache;
use candle::accel::attention as att;
use candle::accel::formats::{row_layout, RowLayout};
use candle::accel::turbo::{
    fwht_f32_inplace, hadamard_sign_bits, qjl_matrix, rotation_matrix, RowCodec, TqAttnArgs,
    TqRowArgs, TqSide,
};
use candle::accel::{self, RotationKind, TurboQuantOptions};
use candle::backend::BackendStorage;
use candle::{bail, CpuStorage, DType, Device, Layout, Result, Shape, Tensor};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub use candle::accel::QuantFormat;

/// A tensor quantized row by row along its last dimension.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// `(rows, row_words)` packed rows.
    pub data: Tensor,
    /// Shape and dtype of the original tensor.
    pub shape: Shape,
    pub dtype: DType,
}

impl QuantizedTensor {
    pub fn size_in_bytes(&self) -> usize {
        self.data.elem_count() * 4
    }
}

/// Quantizes vectors of a fixed length in one format.
#[derive(Debug, Clone)]
pub struct TurboQuantizer {
    format: QuantFormat,
    rotation: RotationKind,
    seed: u64,
    dim: usize,
    layout: RowLayout,
    codec: Arc<RowCodec>,
    /// `Π` and `Πᵀ` (`dim x dim`), lazily populated when Hadamard uses O(d log d) FWHT so
    /// CPU/CUDA/ROCm never allocate unused `d x d` rotation matrices (`Q3-F06`).
    rot: Arc<Mutex<Option<(Tensor, Tensor)>>>,
    /// Packed sign bits (`ceil(dim / 32)` u32 words) for O(d log d) FWHT when
    /// `rotation == Hadamard && dim.is_power_of_two() && dim <= 256`.
    hadamard_signs: Option<(Tensor, Arc<Vec<u32>>)>,
    /// `S` and `Sᵀ` of the QJL stage (prod only).
    qjl: Option<(Tensor, Tensor)>,
    /// Codebook table (32 f32).
    tab: Tensor,
    /// Table of the attention kernels (this table, then the value quantizer's), cached per value
    /// quantizer table.
    pair_tab: Arc<Mutex<Option<(candle::TensorId, Tensor)>>>,
    device: Device,
}

fn warn_cpu_fallback(what: &str, err: &candle::Error) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let msg = format!("{what}: {err}");
    let first = SEEN
        .lock()
        .map(|mut g| g.get_or_insert_with(HashSet::new).insert(msg.clone()))
        .unwrap_or(false);
    if first || accel::config().turbo_quant.debug {
        eprintln!("candle turbo-quant: GPU kernel unavailable ({msg}); using the CPU codec");
    }
}

impl TurboQuantizer {
    /// A quantizer for vectors of length `dim`. TurboQuant formats without a bit width get 4
    /// bits; `rotation` is resolved for `dim` (`Auto`: Hadamard for powers of two, dense
    /// otherwise).
    pub fn new(
        format: QuantFormat,
        dim: usize,
        rotation: RotationKind,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        let format = format.with_default_bits(4);
        let layout = row_layout(format, dim)?;
        let codec = Arc::new(RowCodec::new(format, dim, seed)?);
        let rotation = rotation.resolve(dim);
        let hadamard_signs = if rotation == RotationKind::Hadamard
            && dim.is_power_of_two()
            && (dim <= 256 || device.is_cpu())
        {
            let bits = hadamard_sign_bits(dim, seed)?;
            let t = Tensor::from_vec(bits.clone(), bits.len(), device)?;
            Some((t, Arc::new(bits)))
        } else {
            None
        };
        // Skip eager d x d Π/Πᵀ allocation when FWHT handles rotation directly on this device
        // (CPU, CUDA, ROCm); materialize lazily only if dense rotation is actually needed.
        let eager_rot = rotation != RotationKind::None
            && !(hadamard_signs.is_some()
                && matches!(device, Device::Cpu | Device::Cuda(_) | Device::Rocm(_)));
        let rot_init = if eager_rot {
            let m = Tensor::from_vec(rotation_matrix(dim, rotation, seed)?, (dim, dim), device)?;
            let mt = m.t()?.contiguous()?;
            Some((m, mt))
        } else {
            None
        };
        let qjl = match &codec.qjl {
            Some(_) => {
                let s = Tensor::from_vec(qjl_matrix(dim, seed), (dim, dim), device)?;
                let st = s.t()?.contiguous()?;
                Some((s, st))
            }
            None => None,
        };
        let table = codec
            .codebook
            .as_ref()
            .map(|c| c.table().to_vec())
            .unwrap_or_else(|| vec![0f32; 32]);
        let tab = Tensor::from_vec(table, 32, device)?;
        Ok(Self {
            format,
            rotation,
            seed,
            dim,
            layout,
            codec,
            rot: Arc::new(Mutex::new(rot_init)),
            hadamard_signs,
            qjl,
            tab,
            pair_tab: Arc::new(Mutex::new(None)),
            device: device.clone(),
        })
    }

    /// Key (`keys = true`) or value quantizer with the global switches: format (per GPU when not
    /// set), bits, rotation, QJL and seed (values use `seed + 2`, like the reference).
    pub fn from_options(
        o: &TurboQuantOptions,
        dim: usize,
        keys: bool,
        device: &Device,
    ) -> Result<Self> {
        let (kf, vf) = accel::profile(device).turbo_formats(o);
        let (format, seed) = if keys {
            (kf, o.seed)
        } else {
            (vf, o.seed.wrapping_add(2))
        };
        Self::new(format, dim, o.rotation, seed, device)
    }

    pub fn format(&self) -> QuantFormat {
        self.format
    }

    pub fn rotation(&self) -> RotationKind {
        self.rotation
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Words per packed row.
    pub fn row_words(&self) -> usize {
        self.layout.row_words
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn kernels(&self) -> bool {
        !self.device.is_cpu() && accel::config().turbo_quant_kernels_allowed(&self.device)
    }

    fn fwht_transform(&self, x: &Tensor, forward: bool) -> Result<Option<Tensor>> {
        let Some((signs_t, signs_cpu)) = &self.hadamard_signs else {
            return Ok(None);
        };
        let shape = x.shape().clone();
        let d = self.dim;
        let rows = x.elem_count() / d.max(1);
        if rows == 0 {
            return Ok(Some(Tensor::zeros(shape, DType::F32, x.device())?));
        }
        if x.device().is_cpu() {
            let mut v = x
                .to_dtype(DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let sb = signs_cpu.as_slice();
            v.par_chunks_mut(d)
                .for_each(|row| fwht_f32_inplace(row, sb, forward));
            return Ok(Some(Tensor::from_vec(v, shape, x.device())?));
        }
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if self.kernels() && matches!(x.device(), Device::Cuda(_) | Device::Rocm(_)) && d <= 256 {
            let x2 = x.contiguous()?.reshape((rows, d))?;
            match x2.apply_op2_no_bwd(
                signs_t,
                &TqFwht {
                    rows,
                    dim: d,
                    forward,
                },
            ) {
                Ok(out) => return Ok(Some(out.reshape(shape)?)),
                Err(err) => {
                    // Fall back cleanly to dense rotation instead of aborting append/attention (`Q3-F10`).
                    warn_cpu_fallback("tq_fwht", &err);
                }
            }
        }
        let _ = signs_t;
        Ok(None)
    }

    fn dense_rot(&self) -> Result<Option<(Tensor, Tensor)>> {
        if self.rotation == RotationKind::None {
            return Ok(None);
        }
        let mut g = self.rot.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pair) = g.as_ref() {
            return Ok(Some(pair.clone()));
        }
        let m = Tensor::from_vec(
            rotation_matrix(self.dim, self.rotation, self.seed)?,
            (self.dim, self.dim),
            &self.device,
        )?;
        let mt = m.t()?.contiguous()?;
        let pair = (m, mt);
        *g = Some(pair.clone());
        Ok(Some(pair))
    }

    /// `x Πᵀ` in f32 (rows of the rotated domain). Uses O(d log d) FWHT when `Hadamard`.
    fn rotate(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(y) = self.fwht_transform(x, true)? {
            return Ok(y);
        }
        let x = x.to_dtype(DType::F32)?;
        match self.dense_rot()? {
            Some((_, mt)) => x.broadcast_matmul(&mt),
            None => x.contiguous(),
        }
    }

    /// `y Π`: back from the rotated domain. Uses O(d log d) FWHT when `Hadamard`.
    fn unrotate(&self, y: &Tensor) -> Result<Tensor> {
        if let Some(z) = self.fwht_transform(y, false)? {
            return Ok(z);
        }
        match self.dense_rot()? {
            Some((m, _)) => y.broadcast_matmul(&m),
            None => Ok(y.clone()),
        }
    }

    /// Packs `x: (rows, dim)` into `(rows, row_words)` u32 rows.
    pub fn quantize_rows(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, d) = x.dims2()?;
        if d != self.dim {
            bail!("turbo-quant: expected rows of {} values, got {d}", self.dim)
        }
        let y = self.rotate(x)?.contiguous()?;
        if self.kernels() {
            match self.quantize_gpu(&y, rows) {
                Ok(t) => return Ok(t),
                Err(e) => warn_cpu_fallback("quantize", &e),
            }
        }
        self.quantize_cpu(&y, rows)
    }

    fn quantize_gpu(&self, y: &Tensor, rows: usize) -> Result<Tensor> {
        let args = TqRowArgs::new(self.format, rows, self.dim)?;
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if let Some((_, st)) = &self.qjl {
            if matches!(y.device(), Device::Cuda(_) | Device::Rocm(_)) && self.dim <= 256 {
                return y.apply_op3_no_bwd(
                    &self.tab,
                    st,
                    &TqQuantProdFused { args },
                );
            }
        }
        let packed = y.apply_op1_no_bwd(&TqQuant {
            format: self.format,
            args,
            tab: self.tab.clone(),
        })?;
        match &self.qjl {
            None => Ok(packed),
            Some((_, st)) => {
                let resid = y.apply_op2_no_bwd(
                    &packed,
                    &TqResid {
                        args,
                        tab: self.tab.clone(),
                    },
                )?;
                let proj = resid.matmul(st)?;
                packed.apply_op3_no_bwd(&proj, &resid, &TqSigns { args })
            }
        }
    }

    fn quantize_cpu(&self, y: &Tensor, rows: usize) -> Result<Tensor> {
        let y = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let rw = self.layout.row_words;
        let d = self.dim;
        let mut words = vec![0u32; rows * rw];
        words
            .par_chunks_mut(rw)
            .zip(y.par_chunks(d))
            .for_each(|(w, x)| self.codec.quantize_row(x, w));
        Tensor::from_vec(words, (rows, rw), &self.device)
    }

    /// Reconstructs `(rows, dim)` f32 values from `(rows, row_words)` rows.
    pub fn dequantize_rows(&self, data: &Tensor) -> Result<Tensor> {
        let (rows, rw) = data.dims2()?;
        if rw != self.layout.row_words {
            bail!(
                "turbo-quant: expected rows of {} words, got {rw}",
                self.layout.row_words
            )
        }
        let y = if self.kernels() {
            match self.dequantize_gpu(data, rows) {
                Ok(t) => t,
                Err(e) => {
                    warn_cpu_fallback("dequantize", &e);
                    self.dequantize_cpu(data, rows)?
                }
            }
        } else {
            self.dequantize_cpu(data, rows)?
        };
        self.unrotate(&y)
    }

    fn dequantize_gpu(&self, data: &Tensor, rows: usize) -> Result<Tensor> {
        let args = TqRowArgs::new(self.format, rows, self.dim)?;
        let with_qjl = self.qjl.is_some();
        let out = data.contiguous()?.apply_op1_no_bwd(&TqDequant {
            format: self.format,
            args,
            tab: self.tab.clone(),
            with_qjl,
        })?;
        let mse = out.narrow(0, 0, 1)?.squeeze(0)?;
        match &self.qjl {
            Some((s, _)) => mse + out.narrow(0, 1, 1)?.squeeze(0)?.matmul(s)?,
            None => Ok(mse),
        }
    }

    fn dequantize_cpu(&self, data: &Tensor, rows: usize) -> Result<Tensor> {
        let words = data
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let rw = self.layout.row_words;
        let d = self.dim;
        let mut out = vec![0f32; rows * d];
        out.par_chunks_mut(d)
            .zip(words.par_chunks(rw))
            .for_each(|(o, w)| self.codec.dequantize_row(w, o));
        Tensor::from_vec(out, (rows, d), &self.device)
    }

    /// Quantizes a tensor along its last dimension.
    pub fn quantize(&self, x: &Tensor) -> Result<QuantizedTensor> {
        let d = x.dim(candle::D::Minus1)?;
        if d != self.dim {
            bail!("turbo-quant: expected a last dim of {}, got {d}", self.dim)
        }
        let rows = x.elem_count() / d.max(1);
        Ok(QuantizedTensor {
            data: self.quantize_rows(&x.reshape((rows, d))?)?,
            shape: x.shape().clone(),
            dtype: x.dtype(),
        })
    }

    pub fn dequantize(&self, q: &QuantizedTensor) -> Result<Tensor> {
        self.dequantize_rows(&q.data)?
            .reshape(q.shape.clone())?
            .to_dtype(q.dtype)
    }
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// `with_tab!(self.tab, Variant, |tab_storage, tab_offset| ...)`
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
macro_rules! with_tab {
    ($tab:expr, $variant:ident, |$s:ident, $o:ident| $body:expr) => {{
        let (g, l) = $tab.storage_and_layout();
        let candle::Storage::$variant($s) = &*g else {
            bail!("turbo-quant: the codebook table is on another device")
        };
        let $o = l.start_offset();
        $body
    }};
}

fn contiguous_offset(l: &Layout, what: &str) -> Result<usize> {
    match l.contiguous_offsets() {
        Some((o, _)) => Ok(o),
        None => bail!("turbo-quant: {what} must be contiguous"),
    }
}

/// On-chip `O(d log d)` FWHT rotation/unrotation op on CUDA/ROCm.
#[cfg(any(feature = "cuda", feature = "rocm"))]
struct TqFwht {
    rows: usize,
    dim: usize,
    forward: bool,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl candle::CustomOp2 for TqFwht {
    fn name(&self) -> &'static str {
        "turbo-quant-fwht"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        bail!("turbo-quant-fwht: CPU path handled directly in fwht_transform")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        xl: &Layout,
        signs: &candle::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let xo = contiguous_offset(xl, "fwht input")?;
        let so = contiguous_offset(sl, "fwht signs")?;
        let out = x.tq_fwht(xo, signs, so, self.rows, self.dim, self.forward)?;
        Ok((out, Shape::from((self.rows, self.dim))))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        x: &candle::RocmStorage,
        xl: &Layout,
        signs: &candle::RocmStorage,
        sl: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        let xo = contiguous_offset(xl, "fwht input")?;
        let so = contiguous_offset(sl, "fwht signs")?;
        let out = x.tq_fwht(xo, signs, so, self.rows, self.dim, self.forward)?;
        Ok((out, Shape::from((self.rows, self.dim))))
    }
}

/// Single-pass TurboProd quantization op on CUDA/ROCm (codes + residual + QJL signs in 1 launch).
#[cfg(any(feature = "cuda", feature = "rocm"))]
struct TqQuantProdFused {
    args: TqRowArgs,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl candle::CustomOp3 for TqQuantProdFused {
    fn name(&self) -> &'static str {
        "turbo-quant-prod-fused"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        bail!("turbo-quant-prod-fused: GPU-only op")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        y: &candle::CudaStorage,
        yl: &Layout,
        tab: &candle::CudaStorage,
        tl: &Layout,
        s_mat: &candle::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let yo = contiguous_offset(yl, "prod input")?;
        let to = contiguous_offset(tl, "prod tab")?;
        let so = contiguous_offset(sl, "prod s_mat")?;
        let out = y.tq_quantize_prod_fused(yo, tab, to, s_mat, so, &self.args)?;
        Ok((out, Shape::from((self.args.rows, self.args.row_words))))
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        y: &candle::RocmStorage,
        yl: &Layout,
        tab: &candle::RocmStorage,
        tl: &Layout,
        s_mat: &candle::RocmStorage,
        sl: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        let yo = contiguous_offset(yl, "prod input")?;
        let to = contiguous_offset(tl, "prod tab")?;
        let so = contiguous_offset(sl, "prod s_mat")?;
        let out = y.tq_quantize_prod_fused(yo, tab, to, s_mat, so, &self.args)?;
        Ok((out, Shape::from((self.args.rows, self.args.row_words))))
    }
}

/// Stage one of the quantization: codes, block scales and `rsc0` (prod rows get their QJL
/// sketch from [`TqSigns`]).
#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "vulkan")),
    allow(dead_code)
)]
struct TqQuant {
    format: QuantFormat,
    args: TqRowArgs,
    tab: Tensor,
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
macro_rules! quant_fwd {
    ($self:ident, $variant:ident, $s:ident, $l:ident) => {{
        let o = contiguous_offset($l, "the input")?;
        let out = with_tab!($self.tab, $variant, |t, to| $s.tq_quantize(
            o,
            t,
            to,
            &$self.args
        ))?;
        Ok((out, Shape::from((($self.args.rows, $self.args.row_words)))))
    }};
}

impl candle::CustomOp1 for TqQuant {
    fn name(&self) -> &'static str {
        "turbo-quant-quantize"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let y = s.to_dtype(l, DType::F32)?;
        let y = y.as_slice::<f32>()?;
        let a = &self.args;
        let codec = RowCodec::without_qjl(self.format, a.dim)?;
        let mut words = vec![0u32; a.rows * a.row_words];
        words
            .par_chunks_mut(a.row_words)
            .zip(y.par_chunks(a.dim))
            .for_each(|(w, x)| codec.quantize_row(x, w));
        Ok((CpuStorage::U32(words), Shape::from((a.rows, a.row_words))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        quant_fwd!(self, Cuda, s, l)
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s: &candle::RocmStorage,
        l: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        quant_fwd!(self, Rocm, s, l)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s: &candle::VulkanStorage,
        l: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        quant_fwd!(self, Vulkan, s, l)
    }
}

/// Residual of the unit vectors to their codebook values (prod rows).
struct TqResid {
    args: TqRowArgs,
    tab: Tensor,
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
macro_rules! resid_fwd {
    ($self:ident, $variant:ident, $s1:ident, $l1:ident, $s2:ident, $l2:ident) => {{
        let o1 = contiguous_offset($l1, "the input")?;
        let o2 = contiguous_offset($l2, "the packed rows")?;
        let out = with_tab!($self.tab, $variant, |t, to| $s1.tq_residual(
            o1,
            $s2,
            o2,
            t,
            to,
            &$self.args
        ))?;
        Ok((out, Shape::from(($self.args.rows, $self.args.dim))))
    }};
}

impl candle::CustomOp2 for TqResid {
    fn name(&self) -> &'static str {
        "turbo-quant-residual"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let a = &self.args;
        let y = s1.to_dtype(l1, DType::F32)?;
        let y = y.as_slice::<f32>()?;
        let o2 = contiguous_offset(l2, "the packed rows")?;
        let words = &s2.as_slice::<u32>()?[o2..];
        let tab = self.tab.to_vec1::<f32>()?;
        let ro = a.rsc_offset();
        let mut out = vec![0f32; a.rows * a.dim];
        out.par_chunks_mut(a.dim).enumerate().for_each(|(r, o)| {
            let w = &words[r * a.row_words..(r + 1) * a.row_words];
            let norm = f32::from_bits(w[ro]);
            let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for (j, oj) in o.iter_mut().enumerate() {
                let c = candle::accel::formats::get_bits(w, j * a.code_bits as usize, a.code_bits);
                *oj = y[r * a.dim + j] * inv - tab[c as usize];
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((a.rows, a.dim))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        resid_fwd!(self, Cuda, s1, l1, s2, l2)
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s1: &candle::RocmStorage,
        l1: &Layout,
        s2: &candle::RocmStorage,
        l2: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        resid_fwd!(self, Rocm, s1, l1, s2, l2)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s1: &candle::VulkanStorage,
        l1: &Layout,
        s2: &candle::VulkanStorage,
        l2: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        resid_fwd!(self, Vulkan, s1, l1, s2, l2)
    }
}

/// QJL sign bits and residual norms of prod rows.
struct TqSigns {
    args: TqRowArgs,
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
macro_rules! signs_fwd {
    ($self:ident, $s1:ident, $l1:ident, $s2:ident, $l2:ident, $s3:ident, $l3:ident) => {{
        let o1 = contiguous_offset($l1, "the packed rows")?;
        let o2 = contiguous_offset($l2, "the projections")?;
        let o3 = contiguous_offset($l3, "the residuals")?;
        let out = $s1.tq_signs(o1, $s2, o2, $s3, o3, &$self.args)?;
        Ok((out, Shape::from(($self.args.rows, $self.args.row_words))))
    }};
}

impl candle::CustomOp3 for TqSigns {
    fn name(&self) -> &'static str {
        "turbo-quant-signs"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let a = &self.args;
        let words = &s1.as_slice::<u32>()?[contiguous_offset(l1, "the packed rows")?..];
        let proj = s2.to_dtype(l2, DType::F32)?;
        let proj = proj.as_slice::<f32>()?;
        let resid = s3.to_dtype(l3, DType::F32)?;
        let resid = resid.as_slice::<f32>()?;
        let mut out = words[..a.rows * a.row_words].to_vec();
        let sign_words = a.dim.div_ceil(32);
        out.par_chunks_mut(a.row_words)
            .enumerate()
            .for_each(|(r, w)| {
                for k in 0..sign_words {
                    w[a.code_words + k] = 0;
                }
                for i in 0..a.dim {
                    if proj[r * a.dim + i] >= 0.0 {
                        w[a.code_words + i / 32] |= 1 << (i % 32);
                    }
                }
                let g: f32 = resid[r * a.dim..(r + 1) * a.dim]
                    .iter()
                    .map(|v| v * v)
                    .sum();
                w[a.code_words + sign_words + 1] = g.sqrt().to_bits();
            });
        Ok((CpuStorage::U32(out), Shape::from((a.rows, a.row_words))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        signs_fwd!(self, s1, l1, s2, l2, s3, l3)
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s1: &candle::RocmStorage,
        l1: &Layout,
        s2: &candle::RocmStorage,
        l2: &Layout,
        s3: &candle::RocmStorage,
        l3: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        signs_fwd!(self, s1, l1, s2, l2, s3, l3)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s1: &candle::VulkanStorage,
        l1: &Layout,
        s2: &candle::VulkanStorage,
        l2: &Layout,
        s3: &candle::VulkanStorage,
        l3: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        signs_fwd!(self, s1, l1, s2, l2, s3, l3)
    }
}

/// `rsc0 * element` of every element (rotated domain), followed by the QJL coefficients of prod
/// rows: output `(1 or 2, rows, dim)`.
#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "vulkan")),
    allow(dead_code)
)]
struct TqDequant {
    format: QuantFormat,
    args: TqRowArgs,
    tab: Tensor,
    with_qjl: bool,
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
macro_rules! dequant_fwd {
    ($self:ident, $variant:ident, $s:ident, $l:ident) => {{
        let o = contiguous_offset($l, "the packed rows")?;
        let out = with_tab!($self.tab, $variant, |t, to| $s.tq_dequantize(
            o,
            t,
            to,
            &$self.args,
            $self.with_qjl
        ))?;
        let parts = if $self.with_qjl { 2 } else { 1 };
        Ok((out, Shape::from((parts, $self.args.rows, $self.args.dim))))
    }};
}

impl candle::CustomOp1 for TqDequant {
    fn name(&self) -> &'static str {
        "turbo-quant-dequantize"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let a = &self.args;
        let words = &s.as_slice::<u32>()?[contiguous_offset(l, "the packed rows")?..];
        let codec = RowCodec::without_qjl(self.format, a.dim)?;
        let parts = if self.with_qjl { 2 } else { 1 };
        let n = a.rows * a.dim;
        let mut out = vec![0f32; parts * n];
        let (mse, zs) = out.split_at_mut(n);
        mse.par_chunks_mut(a.dim).enumerate().for_each(|(r, o)| {
            let w = &words[r * a.row_words..(r + 1) * a.row_words];
            let (r0, _) = codec.rsc(w);
            for (j, oj) in o.iter_mut().enumerate() {
                *oj = r0 * codec.elem(w, j);
            }
        });
        if self.with_qjl {
            zs.par_chunks_mut(a.dim).enumerate().for_each(|(r, o)| {
                codec.qjl_signs(&words[r * a.row_words..(r + 1) * a.row_words], o);
            });
        }
        Ok((CpuStorage::F32(out), Shape::from((parts, a.rows, a.dim))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        dequant_fwd!(self, Cuda, s, l)
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s: &candle::RocmStorage,
        l: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        dequant_fwd!(self, Rocm, s, l)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s: &candle::VulkanStorage,
        l: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        dequant_fwd!(self, Vulkan, s, l)
    }
}

/// Attention of rotated queries over packed K/V rows (`(b, hkv, capacity, words)` caches).
#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "vulkan")),
    allow(dead_code)
)]
struct TqAttn {
    args: TqAttnArgs,
    tab: Tensor,
    alibi: Option<Tensor>,
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
macro_rules! attn_fwd {
    ($self:ident, $variant:ident, $q:ident, $ql:ident, $k:ident, $kl:ident, $v:ident, $vl:ident) => {{
        let mut a = $self.args;
        a.q_offset = contiguous_offset($ql, "the queries")?;
        a.k_offset = contiguous_offset($kl, "the key cache")?;
        a.v_offset = contiguous_offset($vl, "the value cache")?;
        let alibi_g = $self.alibi.as_ref().map(|t| t.storage_and_layout());
        let alibi = match alibi_g.as_ref() {
            Some((g, l)) => {
                let candle::Storage::$variant(s) = &**g else {
                    bail!("turbo-quant: the alibi slopes are on another device")
                };
                Some((s, l.start_offset()))
            }
            None => None,
        };
        let out = with_tab!($self.tab, $variant, |t, to| $q
            .tq_attention($k, $v, t, to, alibi, &a))?;
        Ok((out, Shape::from((a.b, a.hq, a.lq, a.d))))
    }};
}

impl candle::CustomOp3 for TqAttn {
    fn name(&self) -> &'static str {
        "turbo-quant-attention"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        bail!("turbo-quant attention: no fused CPU kernel (the cache dequantizes on the CPU)")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        ql: &Layout,
        k: &candle::CudaStorage,
        kl: &Layout,
        v: &candle::CudaStorage,
        vl: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        attn_fwd!(self, Cuda, q, ql, k, kl, v, vl)
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        q: &candle::RocmStorage,
        ql: &Layout,
        k: &candle::RocmStorage,
        kl: &Layout,
        v: &candle::RocmStorage,
        vl: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        attn_fwd!(self, Rocm, q, ql, k, kl, v, vl)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        q: &candle::VulkanStorage,
        ql: &Layout,
        k: &candle::VulkanStorage,
        kl: &Layout,
        v: &candle::VulkanStorage,
        vl: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        attn_fwd!(self, Vulkan, q, ql, k, kl, v, vl)
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// A KV cache that stores keys and values compressed (TurboQuant / NVFP4 / MXFP4 / MXFP8 /
/// FP8) and attends over them without decompressing them (fused path), or dequantizes and runs
/// flash attention. With TurboQuant switched off it is a plain cache.
///
/// Layouts: `append(k, v)` with `(batch, kv_heads, seq, head_dim)`, `attention(q)` with
/// `(batch, q_heads, seq_q, head_dim)`, output in the same layout as `q`.
#[derive(Debug, Clone)]
pub struct TurboKvCache {
    quant: Option<(TurboQuantizer, TurboQuantizer)>,
    k: Cache,
    v: Cache,
    fused: bool,
    dim: usize,
    dtype: Option<DType>,
}

impl TurboKvCache {
    /// A cache with the global switches (`candle::accel::config().turbo_quant`).
    pub fn new(dim: usize, grow_by: usize, device: &Device) -> Result<Self> {
        Self::with_options(&accel::config().turbo_quant, dim, grow_by, device)
    }

    /// A cache with explicit options (formats left at `None` are picked for the GPU).
    pub fn with_options(
        o: &TurboQuantOptions,
        dim: usize,
        grow_by: usize,
        device: &Device,
    ) -> Result<Self> {
        let quant = if o.enabled {
            Some((
                TurboQuantizer::from_options(o, dim, true, device)?,
                TurboQuantizer::from_options(o, dim, false, device)?,
            ))
        } else {
            None
        };
        if o.debug {
            match &quant {
                Some((k, v)) => eprintln!(
                    "candle turbo-quant: kv cache k {} v {} (rotation {:?}) on {}",
                    k.format(),
                    v.format(),
                    k.rotation(),
                    accel::profile(device).arch
                ),
                None => eprintln!("candle turbo-quant: off, plain kv cache"),
            }
        }
        Ok(Self {
            quant,
            k: Cache::new(2, grow_by.max(1)),
            v: Cache::new(2, grow_by.max(1)),
            fused: o.fused_attention,
            dim,
            dtype: None,
        })
    }

    /// [`Self::with_options`] sized for `capacity` positions: the K/V buffers are allocated once
    /// with that many rows at the first append (e.g. the planned prompt + generation length), so
    /// the cache does not grow while it stays within it (past it, it grows geometrically).
    pub fn with_capacity(
        o: &TurboQuantOptions,
        dim: usize,
        capacity: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::with_options(o, dim, capacity, device)
    }

    /// A compressed cache with explicit formats.
    pub fn with_formats(
        key: QuantFormat,
        value: QuantFormat,
        dim: usize,
        rotation: RotationKind,
        seed: u64,
        grow_by: usize,
        device: &Device,
    ) -> Result<Self> {
        let k = TurboQuantizer::new(key, dim, rotation, seed, device)?;
        let v = TurboQuantizer::new(value, dim, rotation, seed.wrapping_add(2), device)?;
        if v.format().mode_id() == candle::accel::formats::mode::TURBO_PROD {
            bail!("turbo-quant: values use the MSE codebooks, not the prod (QJL) format")
        }
        Ok(Self {
            quant: Some((k, v)),
            k: Cache::new(2, grow_by.max(1)),
            v: Cache::new(2, grow_by.max(1)),
            fused: accel::config().turbo_quant.fused_attention,
            dim,
            dtype: None,
        })
    }

    /// Run attention directly on the compressed rows (default from the configuration).
    pub fn set_fused(&mut self, fused: bool) {
        self.fused = fused;
    }

    pub fn is_quantized(&self) -> bool {
        self.quant.is_some()
    }

    pub fn formats(&self) -> Option<(QuantFormat, QuantFormat)> {
        self.quant.as_ref().map(|(k, v)| (k.format(), v.format()))
    }

    pub fn seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }

    /// Bytes held by the K/V buffers (capacity included).
    pub fn storage_bytes(&self) -> usize {
        let bytes = |c: &Cache| {
            c.all_data()
                .as_ref()
                .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                .unwrap_or(0)
        };
        bytes(&self.k) + bytes(&self.v)
    }

    /// Appends `k`, `v`: `(batch, kv_heads, seq, head_dim)`.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (b, h, t, d) = k.dims4()?;
        if d != self.dim || v.dims4()? != (b, h, t, d) {
            bail!(
                "turbo-quant cache: k {:?} / v {:?} do not match head dim {}",
                k.dims(),
                v.dims(),
                self.dim
            )
        }
        self.dtype.get_or_insert(k.dtype());
        match &self.quant {
            None => {
                self.k.append(&k.contiguous()?)?;
                self.v.append(&v.contiguous()?)?;
            }
            Some((kq, vq)) => {
                let rows = b * h * t;
                let kd =
                    kq.quantize_rows(&k.reshape((rows, d))?)?
                        .reshape((b, h, t, kq.row_words()))?;
                let vd =
                    vq.quantize_rows(&v.reshape((rows, d))?)?
                        .reshape((b, h, t, vq.row_words()))?;
                self.k.append(&kd)?;
                self.v.append(&vd)?;
            }
        }
        Ok(())
    }

    /// Keys and values `(batch, kv_heads, seq, head_dim)`, dequantized when compressed.
    pub fn keys_values(&self) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(k), Some(v)) = (self.k.current_data()?, self.v.current_data()?) else {
            return Ok(None);
        };
        match &self.quant {
            None => Ok(Some((k, v))),
            Some((kq, vq)) => {
                let (b, h, t, _) = k.dims4()?;
                let dt = self.dtype.unwrap_or(DType::F32);
                let deq = |q: &TurboQuantizer, x: &Tensor| -> Result<Tensor> {
                    let x = x.contiguous()?.reshape((b * h * t, q.row_words()))?;
                    q.dequantize_rows(&x)?
                        .reshape((b, h, t, self.dim))?
                        .to_dtype(dt)
                };
                Ok(Some((deq(kq, &k)?, deq(vq, &v)?)))
            }
        }
    }

    /// Attention of `q: (batch, q_heads, seq_q, head_dim)` over the cache with the masks,
    /// soft-capping and ALiBi of `cfg` (bottom-right aligned: the queries are the last tokens).
    pub fn attention(&self, q: &Tensor, cfg: &FlashAttnConfig) -> Result<Tensor> {
        if let Some((kq, vq)) = &self.quant {
            let fused = cfg.fused.unwrap_or(self.fused);
            if fused {
                if let (Some(kd), Some(vd)) =
                    (self.k.all_data().as_ref(), self.v.all_data().as_ref())
                {
                    match fused_attention_rows(q, kd, vd, self.seq_len(), kq, vq, cfg) {
                        Ok(Some(o)) => return Ok(o),
                        Ok(None) => {}
                        Err(e) if cfg.strict => return Err(e),
                        Err(e) => warn_cpu_fallback("attention", &e),
                    }
                }
            }
        }
        let Some((k, v)) = self.keys_values()? else {
            bail!("turbo-quant cache: attention on an empty cache")
        };
        flash_attn_bhsd(q, &k.to_dtype(q.dtype())?, &v.to_dtype(q.dtype())?, cfg)
    }
}

/// Attention of `q: (batch, q_heads, seq_q, head_dim)` over keys and values that are already
/// packed by `kq` / `vq`: `k_rows: (batch, kv_heads, seq, k_words)` and
/// `v_rows: (batch, kv_heads, seq, v_words)` u32, of which the first `kv_len` rows of the third
/// axis are attended (the rest is spare capacity). Masks, soft-capping and ALiBi come from `cfg`
/// (bottom-right aligned: the queries are the last `seq_q` of the `kv_len` positions).
///
/// This is the attention of [`TurboKvCache`] for caches that live outside it, e.g. paged KV
/// blocks that are offloaded to host memory and concatenated / uploaded per step: the rows stay
/// compressed through the transfer and the fused kernel reads them directly. Without the fused
/// kernel (CPU, kernels switched off, unsupported shape or failure) the rows are dequantized and
/// [`crate::flash_attn::flash_attn_bhsd`] runs, so the result is the same up to rounding.
/// The output is in `q`'s dtype and layout.
pub fn attention_packed(
    q: &Tensor,
    k_rows: &Tensor,
    v_rows: &Tensor,
    kv_len: usize,
    kq: &TurboQuantizer,
    vq: &TurboQuantizer,
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    let (b, _, _, d) = q.dims4()?;
    let (bk, hkv, cap, kw) = k_rows.dims4()?;
    let (bv, hv, capv, vw) = v_rows.dims4()?;
    if k_rows.dtype() != DType::U32 || v_rows.dtype() != DType::U32 {
        bail!("turbo-quant attention: packed rows must be u32")
    }
    // rows offloaded to host memory are uploaded compressed (a no-op when already on q's device)
    let k_rows = &k_rows.to_device(q.device())?;
    let v_rows = &v_rows.to_device(q.device())?;
    if bk != b || bv != b || hv != hkv || capv < kv_len || cap < kv_len || kv_len == 0 {
        bail!(
            "turbo-quant attention: q {:?} / k {:?} / v {:?} / kv_len {kv_len} do not match",
            q.dims(),
            k_rows.dims(),
            v_rows.dims()
        )
    }
    if kw != kq.row_words() || vw != vq.row_words() || d != kq.dim() || d != vq.dim() {
        bail!(
            "turbo-quant attention: rows of {kw} / {vw} words and head dim {d} do not match the quantizers ({} / {} words, dim {})",
            kq.row_words(),
            vq.row_words(),
            kq.dim()
        )
    }
    if cfg
        .fused
        .unwrap_or(accel::config().turbo_quant.fused_attention)
    {
        // the kernel reads K and V with the same capacity (row stride of a head)
        let (kd, vd) = if cap == capv {
            (k_rows.contiguous()?, v_rows.contiguous()?)
        } else {
            (
                k_rows.narrow(2, 0, kv_len)?.contiguous()?,
                v_rows.narrow(2, 0, kv_len)?.contiguous()?,
            )
        };
        match fused_attention_rows(q, &kd, &vd, kv_len, kq, vq, cfg) {
            Ok(Some(o)) => return Ok(o),
            Ok(None) => {}
            Err(e) if cfg.strict => return Err(e),
            Err(e) => warn_cpu_fallback("attention", &e),
        }
    }
    let deq = |qz: &TurboQuantizer, rows: &Tensor| -> Result<Tensor> {
        let w = rows.dim(3)?;
        let flat = rows
            .narrow(2, 0, kv_len)?
            .contiguous()?
            .reshape((b * hkv * kv_len, w))?;
        qz.dequantize_rows(&flat)?
            .reshape((b, hkv, kv_len, d))?
            .to_dtype(q.dtype())
    };
    let k = deq(kq, k_rows)?;
    let v = deq(vq, v_rows)?;
    flash_attn_bhsd(q, &k, &v, cfg)
}

/// Attention of `q: (batch, q_heads, seq_q, head_dim)` over packed rows that live in paged
/// cache blocks (`(batch, kv_heads, capacity, words)` u32 tensors packed by `kq` / `vq`, the
/// first `rows` positions of each block valid), without concatenating them: on CUDA / ROCm one
/// `tq_attnp*` launch reads the blocks through a block table (up to 128 blocks, any query
/// length). Other devices, blocks on another device, more blocks or other layouts concatenate
/// the valid rows and run [`attention_packed`], which gives the same result up to the
/// summation order. Masks, soft-capping and ALiBi of `cfg` are bottom-right aligned over the
/// concatenated keys. Output in `q`'s dtype, `(batch, q_heads, seq_q, head_dim)`.
pub fn attention_packed_blocks(
    q: &Tensor,
    blocks: &[KvBlockRef<'_>],
    kq: &TurboQuantizer,
    vq: &TurboQuantizer,
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    let (b, _, _, d) = q.dims4()?;
    let (hkv, lk) = check_blocks(blocks, b, "turbo-quant blocks")?;
    let (kw, vw) = (blocks[0].k.dim(3)?, blocks[0].v.dim(3)?);
    if blocks[0].k.dtype() != DType::U32 || blocks[0].v.dtype() != DType::U32 {
        bail!("turbo-quant blocks: packed rows must be u32")
    }
    if kw != kq.row_words() || vw != vq.row_words() || d != kq.dim() || d != vq.dim() {
        bail!(
            "turbo-quant blocks: rows of {kw} / {vw} words and head dim {d} do not match the quantizers ({} / {} words, dim {})",
            kq.row_words(),
            vq.row_words(),
            kq.dim()
        )
    }
    let conf = accel::config();
    if cfg.fused.unwrap_or(conf.turbo_quant.fused_attention) {
        match paged_attention_rows(q, blocks, hkv, lk, kq, vq, cfg, &conf) {
            Ok(Some(o)) => return Ok(o),
            Ok(None) => {}
            Err(e) if cfg.strict => return Err(e),
            Err(e) => warn_cpu_fallback("paged attention", &e),
        }
    }
    let (k_rows, v_rows) = cat_blocks(blocks, q.device())?;
    attention_packed(q, &k_rows, &v_rows, lk, kq, vq, cfg)
}

/// Kernel inputs of a `tq_attn*` call shared by the dense and the paged paths.
struct TqPrep {
    /// Rotated queries (followed by `S·q` for prod keys), contiguous `(b, hq, lq, w)` f32.
    qx: Tensor,
    args: TqAttnArgs,
    tab: Tensor,
    alibi: Option<Tensor>,
}

/// The combined 64-float table (key centroids, value centroids) of a quantizer pair, built
/// once per value quantizer and kept by the key quantizer.
fn pair_tab(kq: &TurboQuantizer, vq: &TurboQuantizer) -> Result<Tensor> {
    let mut g = kq.pair_tab.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((id, t)) = g.as_ref() {
        if *id == vq.tab.id() {
            return Ok(t.clone());
        }
    }
    let t = Tensor::cat(&[&kq.tab, &vq.tab], 0)?;
    *g = Some((vq.tab.id(), t.clone()));
    Ok(t)
}

/// Queries, arguments and tables of a fused call over `lk` keys whose K / V rows have the
/// `(batch, head, row)` word strides `k_strides` / `v_strides`; `Ok(None)` when the fused
/// kernels do not apply (CPU, kernels off, shape out of range).
#[allow(clippy::too_many_arguments)]
fn tq_prepare(
    q: &Tensor,
    hkv: usize,
    lk: usize,
    k_strides: [usize; 3],
    v_strides: [usize; 3],
    kq: &TurboQuantizer,
    vq: &TurboQuantizer,
    cfg: &FlashAttnConfig,
    conf: &accel::AccelConfig,
) -> Result<Option<TqPrep>> {
    if q.device().is_cpu() || !conf.turbo_quant_kernels_allowed(q.device()) {
        return Ok(None);
    }
    let (b, hq, lq, d) = q.dims4()?;
    if d != kq.dim() || hkv == 0 || hq % hkv != 0 || d > att::FA_MAX_HEAD_DIM || lk == 0 {
        return Ok(None);
    }
    // rotated queries, plus S·q for prod keys
    let qr = kq.rotate(q)?;
    let qx = match &kq.qjl {
        Some((_, st)) => Tensor::cat(&[&qr, &qr.broadcast_matmul(st)?], 3)?,
        None => qr,
    }
    .contiguous()?;
    let w = qx.dim(3)?;
    let win = |v: Option<usize>| v.map(|v| v.min(i32::MAX as usize) as i32).unwrap_or(-1);
    let alibi = match &cfg.alibi_slopes {
        Some(t) => Some(
            t.to_device(q.device())?
                .to_dtype(DType::F32)?
                .contiguous()?,
        ),
        None => None,
    };
    let alibi_bstride = match &alibi {
        Some(t) => att::alibi_batch_stride(t.shape(), b, hq)?,
        None => 0,
    };
    let args = TqAttnArgs {
        b,
        lq,
        lk,
        hq,
        hkv,
        d,
        q_strides: [hq * lq * w, lq * w, w],
        q_offset: 0,
        k_strides,
        k_offset: 0,
        v_strides,
        v_offset: 0,
        k: TqSide::new(kq.format(), d)?,
        v: TqSide::new(vq.format(), d)?,
        scale: cfg.softmax_scale,
        softcap: cfg.softcap.unwrap_or(0.0),
        window_left: win(cfg.window_size_left),
        window_right: win(cfg.window_size_right),
        // GQA-packed kernels run several heads per workgroup (CUDA / ROCm)
        splits: att::split_count_grouped(
            b * hq * lq,
            match q.device() {
                Device::Cuda(_) | Device::Rocm(_) => accel::turbo::tq_attn_group(hq, hkv),
                _ => 1,
            },
            lk,
            cfg.split_kv.unwrap_or(conf.flash_attn.split_kv),
            accel::profile(q.device()).compute_units,
        ),
        alibi_bstride,
    };
    Ok(Some(TqPrep {
        qx,
        args,
        tab: pair_tab(kq, vq)?,
        alibi,
    }))
}

/// Zero-dequant fused CPU attention directly over packed `u32` K/V rows.
/// Uses fast 16-entry FP4 LUT + 256-entry E4M3 LUT and rayon-parallel online softmax without
/// materializing `[B, Hkv, S, D]` dequantized K/V buffers.
fn cpu_attention_packed_fused(
    q: &Tensor,
    kd: &Tensor,
    vd: &Tensor,
    lk: usize,
    kq: &TurboQuantizer,
    vq: &TurboQuantizer,
    cfg: &FlashAttnConfig,
) -> Result<Option<Tensor>> {
    use candle::accel::formats::{e4m3_to_f32, e8m0_to_f32, fp4_to_f32, mode};
    use std::sync::OnceLock;

    let (b, hq, lq, d) = q.dims4()?;
    let (bk, hkv, cap, kw) = kd.dims4()?;
    let vw = vd.dim(3)?;
    if bk != b || vd.dim(2)? != cap || lk > cap || lk == 0 || hkv == 0 || hq % hkv != 0 {
        return Ok(None);
    }

    static FP4_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    static E4M3_LUT: OnceLock<[f32; 256]> = OnceLock::new();
    let e4m3_lut = E4M3_LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_to_f32(i as u8);
        }
        t
    });
    let _ = fp4_to_f32;

    let qr = kq.rotate(q)?;
    let q_sk = match &kq.qjl {
        Some((_, st)) => Some(qr.broadcast_matmul(st)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    let qr_v = qr.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let kd_c = kd.contiguous()?;
    let vd_c = vd.contiguous()?;
    let (kg, kl) = kd_c.storage_and_layout();
    let (vg, vl) = vd_c.storage_and_layout();
    let (CpuStorage::U32(k_all), CpuStorage::U32(v_all)) = (
        match &*kg {
            candle::Storage::Cpu(c) => c,
            _ => return Ok(None),
        },
        match &*vg {
            candle::Storage::Cpu(c) => c,
            _ => return Ok(None),
        },
    ) else {
        return Ok(None);
    };
    let k_words = &k_all[kl.start_offset()..];
    let v_words = &v_all[vl.start_offset()..];

    let alibi_v = match &cfg.alibi_slopes {
        Some(t) => Some((
            t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            att::alibi_batch_stride(t.shape(), b, hq)?,
        )),
        None => None,
    };

    let k_side = TqSide::new(kq.format(), d)?;
    let v_side = TqSide::new(vq.format(), d)?;
    let k_codec = &kq.codec;
    let v_codec = &vq.codec;
    let group = hq / hkv;
    let wl = cfg.window_size_left.map(|v| v.min(i32::MAX as usize) as i32).unwrap_or(-1);
    let wr = cfg.window_size_right.map(|v| v.min(i32::MAX as usize) as i32).unwrap_or(-1);
    let scale = cfg.softmax_scale;
    let softcap = cfg.softcap.unwrap_or(0.0);
    const QJL_FACTOR: f32 = 1.2533141;

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot16_fp4_avx2(q_blk8: *const u8, bsc: f32, qr: *const f32) -> f32 {
        use std::arch::x86_64::*;
        let lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        let raw = _mm_loadl_epi64(q_blk8 as *const __m128i);
        let mask = _mm_set1_epi8(0x0f);
        let lo = _mm_and_si128(raw, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
        let codes16 = _mm_unpacklo_epi8(lo, hi);
        let vals_x2 = _mm_shuffle_epi8(lut, codes16);
        let v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_x2));
        let v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_x2, 8)));
        let q0 = _mm256_loadu_ps(qr);
        let q1 = _mm256_loadu_ps(qr.add(8));
        let prod = _mm256_fmadd_ps(v0, q0, _mm256_mul_ps(v1, q1));
        let lo128 = _mm256_castps256_ps128(prod);
        let hi128 = _mm256_extractf128_ps(prod, 1);
        let s4 = _mm_add_ps(lo128, hi128);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        _mm_cvtss_f32(s1) * (bsc * 0.5)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fmadd16_fp4_avx2(v_blk8: *const u8, pb: f32, acc: *mut f32) {
        use std::arch::x86_64::*;
        let lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        let raw = _mm_loadl_epi64(v_blk8 as *const __m128i);
        let mask = _mm_set1_epi8(0x0f);
        let lo = _mm_and_si128(raw, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
        let codes16 = _mm_unpacklo_epi8(lo, hi);
        let vals_x2 = _mm_shuffle_epi8(lut, codes16);
        let v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_x2));
        let v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_x2, 8)));
        let wv = _mm256_set1_ps(pb * 0.5);
        let a0 = _mm256_fmadd_ps(v0, wv, _mm256_loadu_ps(acc));
        let a1 = _mm256_fmadd_ps(v1, wv, _mm256_loadu_ps(acc.add(8)));
        _mm256_storeu_ps(acc, a0);
        _mm256_storeu_ps(acc.add(8), a1);
    }

    #[cfg(target_arch = "x86_64")]
    let has_avx2_fma = std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && cfg!(target_endian = "little");

    let decode_key_dot = |kr: &[u32], qrow: &[f32]| -> f32 {
        match k_side.mode {
            mode::NVFP4 => {
                let mut dot = 0.0f32;
                let aux = &kr[k_side.code_words..];
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma {
                    for bk in 0..(d / 16) {
                        let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as usize;
                        let bs = e4m3_lut[sbyte];
                        unsafe {
                            let p = kr.as_ptr().add(bk * 2) as *const u8;
                            dot += dot16_fp4_avx2(p, bs, qrow.as_ptr().add(bk * 16));
                        }
                    }
                    return dot;
                }
                for bk in 0..(d / 16) {
                    let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as usize;
                    let bs = e4m3_lut[sbyte];
                    let w0 = kr[bk * 2];
                    let w1 = kr[bk * 2 + 1];
                    let d0 = bk * 16;
                    let mut sub = 0.0f32;
                    for e in 0..8 {
                        sub = FP4_LUT[((w0 >> (e * 4)) & 0xf) as usize].mul_add(qrow[d0 + e], sub);
                    }
                    for e in 0..8 {
                        sub = FP4_LUT[((w1 >> (e * 4)) & 0xf) as usize]
                            .mul_add(qrow[d0 + 8 + e], sub);
                    }
                    dot = sub.mul_add(bs, dot);
                }
                dot
            }
            mode::MXFP4 => {
                let mut dot = 0.0f32;
                let aux = &kr[k_side.code_words..];
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma {
                    for bk in 0..(d / 32) {
                        let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as u8;
                        let bs = e8m0_to_f32(sbyte);
                        unsafe {
                            let p0 = kr.as_ptr().add(bk * 4) as *const u8;
                            let p1 = kr.as_ptr().add(bk * 4 + 2) as *const u8;
                            dot += dot16_fp4_avx2(p0, bs, qrow.as_ptr().add(bk * 32));
                            dot += dot16_fp4_avx2(p1, bs, qrow.as_ptr().add(bk * 32 + 16));
                        }
                    }
                    return dot;
                }
                for bk in 0..(d / 32) {
                    let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as u8;
                    let bs = e8m0_to_f32(sbyte);
                    let d0 = bk * 32;
                    let mut sub = 0.0f32;
                    for wi in 0..4 {
                        let w = kr[bk * 4 + wi];
                        for e in 0..8 {
                            sub = FP4_LUT[((w >> (e * 4)) & 0xf) as usize]
                                .mul_add(qrow[d0 + wi * 8 + e], sub);
                        }
                    }
                    dot = sub.mul_add(bs, dot);
                }
                dot
            }
            mode::FP8 => {
                let mut dot = 0.0f32;
                let full = d / 4;
                for wi in 0..full {
                    let w = kr[wi];
                    let d0 = wi * 4;
                    dot = e4m3_lut[(w & 0xff) as usize].mul_add(qrow[d0], dot);
                    dot = e4m3_lut[((w >> 8) & 0xff) as usize].mul_add(qrow[d0 + 1], dot);
                    dot = e4m3_lut[((w >> 16) & 0xff) as usize].mul_add(qrow[d0 + 2], dot);
                    dot = e4m3_lut[((w >> 24) & 0xff) as usize].mul_add(qrow[d0 + 3], dot);
                }
                let rem = d & 3;
                if rem > 0 {
                    let w = kr[full];
                    let d0 = full * 4;
                    for r in 0..rem {
                        dot = e4m3_lut[((w >> (r * 8)) & 0xff) as usize].mul_add(qrow[d0 + r], dot);
                    }
                }
                dot
            }
            _ => {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot = k_codec.elem(kr, j).mul_add(qrow[j], dot);
                }
                dot
            }
        }
    };

    let accum_val = |vr: &[u32], p: f32, acc: &mut [f32]| {
        match v_side.mode {
            mode::NVFP4 => {
                let aux = &vr[v_side.code_words..];
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma {
                    for bk in 0..(d / 16) {
                        let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as usize;
                        let pb = p * e4m3_lut[sbyte];
                        unsafe {
                            let vp = vr.as_ptr().add(bk * 2) as *const u8;
                            fmadd16_fp4_avx2(vp, pb, acc.as_mut_ptr().add(bk * 16));
                        }
                    }
                    return;
                }
                for bk in 0..(d / 16) {
                    let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as usize;
                    let pb = p * e4m3_lut[sbyte];
                    let w0 = vr[bk * 2];
                    let w1 = vr[bk * 2 + 1];
                    let d0 = bk * 16;
                    for e in 0..8 {
                        acc[d0 + e] = FP4_LUT[((w0 >> (e * 4)) & 0xf) as usize].mul_add(pb, acc[d0 + e]);
                    }
                    for e in 0..8 {
                        acc[d0 + 8 + e] =
                            FP4_LUT[((w1 >> (e * 4)) & 0xf) as usize].mul_add(pb, acc[d0 + 8 + e]);
                    }
                }
            }
            mode::MXFP4 => {
                let aux = &vr[v_side.code_words..];
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma {
                    for bk in 0..(d / 32) {
                        let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as u8;
                        let pb = p * e8m0_to_f32(sbyte);
                        unsafe {
                            let vp0 = vr.as_ptr().add(bk * 4) as *const u8;
                            let vp1 = vr.as_ptr().add(bk * 4 + 2) as *const u8;
                            fmadd16_fp4_avx2(vp0, pb, acc.as_mut_ptr().add(bk * 32));
                            fmadd16_fp4_avx2(vp1, pb, acc.as_mut_ptr().add(bk * 32 + 16));
                        }
                    }
                    return;
                }
                for bk in 0..(d / 32) {
                    let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as u8;
                    let pb = p * e8m0_to_f32(sbyte);
                    let d0 = bk * 32;
                    for wi in 0..4 {
                        let w = vr[bk * 4 + wi];
                        for e in 0..8 {
                            acc[d0 + wi * 8 + e] = FP4_LUT[((w >> (e * 4)) & 0xf) as usize]
                                .mul_add(pb, acc[d0 + wi * 8 + e]);
                        }
                    }
                }
            }
            mode::FP8 => {
                let full = d / 4;
                for wi in 0..full {
                    let w = vr[wi];
                    let d0 = wi * 4;
                    acc[d0] = e4m3_lut[(w & 0xff) as usize].mul_add(p, acc[d0]);
                    acc[d0 + 1] = e4m3_lut[((w >> 8) & 0xff) as usize].mul_add(p, acc[d0 + 1]);
                    acc[d0 + 2] = e4m3_lut[((w >> 16) & 0xff) as usize].mul_add(p, acc[d0 + 2]);
                    acc[d0 + 3] = e4m3_lut[((w >> 24) & 0xff) as usize].mul_add(p, acc[d0 + 3]);
                }
                let rem = d & 3;
                if rem > 0 {
                    let w = vr[full];
                    let d0 = full * 4;
                    for r in 0..rem {
                        acc[d0 + r] =
                            e4m3_lut[((w >> (r * 8)) & 0xff) as usize].mul_add(p, acc[d0 + r]);
                    }
                }
            }
            _ => {
                for j in 0..d {
                    acc[j] = v_codec.elem(vr, j).mul_add(p, acc[j]);
                }
            }
        }
    };

    let total_rows = b * hq * lq;
    let mut out = vec![0.0f32; total_rows * d];

    // Helper to decode a compressed K/V row once into a dense f32 buffer for GQA-shared reuse.
    let decode_row_to_f32 = |words: &[u32], side_mode: u32, code_words: usize, codec: &RowCodec, dst: &mut [f32]| {
        match side_mode {
            mode::NVFP4 => {
                let aux = &words[code_words..];
                for bk in 0..(d / 16) {
                    let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as usize;
                    let bs = e4m3_lut[sbyte];
                    let w0 = words[bk * 2];
                    let w1 = words[bk * 2 + 1];
                    let d0 = bk * 16;
                    for e in 0..8 {
                        dst[d0 + e] = FP4_LUT[((w0 >> (e * 4)) & 0xf) as usize] * bs;
                    }
                    for e in 0..8 {
                        dst[d0 + 8 + e] = FP4_LUT[((w1 >> (e * 4)) & 0xf) as usize] * bs;
                    }
                }
            }
            mode::MXFP4 => {
                let aux = &words[code_words..];
                for bk in 0..(d / 32) {
                    let sbyte = ((aux[bk >> 2] >> ((bk & 3) * 8)) & 0xff) as u8;
                    let bs = e8m0_to_f32(sbyte);
                    let d0 = bk * 32;
                    for wi in 0..4 {
                        let w = words[bk * 4 + wi];
                        for e in 0..8 {
                            dst[d0 + wi * 8 + e] = FP4_LUT[((w >> (e * 4)) & 0xf) as usize] * bs;
                        }
                    }
                }
            }
            mode::FP8 => {
                let full = d / 4;
                for wi in 0..full {
                    let w = words[wi];
                    let d0 = wi * 4;
                    dst[d0] = e4m3_lut[(w & 0xff) as usize];
                    dst[d0 + 1] = e4m3_lut[((w >> 8) & 0xff) as usize];
                    dst[d0 + 2] = e4m3_lut[((w >> 16) & 0xff) as usize];
                    dst[d0 + 3] = e4m3_lut[((w >> 24) & 0xff) as usize];
                }
                let rem = d & 3;
                if rem > 0 {
                    let w = words[full];
                    let d0 = full * 4;
                    for r in 0..rem {
                        dst[d0 + r] = e4m3_lut[((w >> (r * 8)) & 0xff) as usize];
                    }
                }
            }
            _ => {
                for j in 0..d {
                    dst[j] = codec.elem(words, j);
                }
            }
        }
    };

    // When GQA shares a KV head across `group >= 2` query heads (and `d <= 256`), decode each
    // K/V row once per `(b, hk, qi)` group and reuse it across all `group` query heads via
    // runtime-dispatched SIMD `VecOps` (`Q6-F05` / `Q6-O05`).
    if group >= 2 && d <= 256 {
        use candle::cpu::kernels::VecOps;
        out.par_chunks_mut(group * lq * d)
            .enumerate()
            .for_each(|(bg_idx, out_group)| {
                let bi = bg_idx / hkv;
                let hk = bg_idx % hkv;
                let h0 = hk * group;
                let k_head = (bi * hkv + hk) * cap * kw;
                let v_head = (bi * hkv + hk) * cap * vw;
                let mut k_buf = [0.0f32; 256];
                let mut v_buf = [0.0f32; 256];
                let mut m = vec![f32::NEG_INFINITY; group];
                let mut lsum = vec![0.0f32; group];
                for qi in 0..lq {
                    let pos = qi as i32 + lk as i32 - lq as i32;
                    let mut lo = if wl >= 0 { Ord::max(pos - wl, 0) } else { 0 };
                    let mut hi = if wr >= 0 {
                        Ord::min(pos + wr + 1, lk as i32)
                    } else {
                        lk as i32
                    };
                    if lo > lk as i32 {
                        lo = lk as i32;
                    }
                    if hi < lo {
                        hi = lo;
                    }
                    m.fill(f32::NEG_INFINITY);
                    lsum.fill(0.0);
                    for g in 0..group {
                        let o_off = (g * lq + qi) * d;
                        out_group[o_off..o_off + d].fill(0.0);
                    }
                    for j in (lo as usize)..(hi as usize) {
                        let kr = &k_words[k_head + j * kw..k_head + (j + 1) * kw];
                        let vr = &v_words[v_head + j * vw..v_head + (j + 1) * vw];
                        let r0 = f32::from_bits(kr[k_side.rsc_offset]);
                        let vscale = f32::from_bits(vr[v_side.rsc_offset]);
                        decode_row_to_f32(kr, k_side.mode, k_side.code_words, &k_codec, &mut k_buf[..d]);
                        decode_row_to_f32(vr, v_side.mode, v_side.code_words, &v_codec, &mut v_buf[..d]);
                        for g in 0..group {
                            let h = h0 + g;
                            let row_idx = (bi * hq + h) * lq + qi;
                            let qrow = &qr_v[row_idx * d..(row_idx + 1) * d];
                            let mut sc = r0 * unsafe { f32::vec_dot_f32(k_buf.as_ptr(), qrow.as_ptr(), d) };
                            if let Some(sk_all) = &q_sk {
                                let sk = &sk_all[row_idx * d..(row_idx + 1) * d];
                                let mut sdot = 0.0f32;
                                for dd in 0..d {
                                    let bit = (kr[k_side.code_words + (dd >> 5)] >> (dd & 31)) & 1;
                                    sdot += if bit != 0 { sk[dd] } else { -sk[dd] };
                                }
                                let r1 = f32::from_bits(kr[k_side.rsc_offset + 1]);
                                sc += r0 * r1 * QJL_FACTOR / (d as f32) * sdot;
                            }
                            let mut s = sc * scale;
                            if softcap > 0.0 {
                                s = softcap * (s / softcap).tanh();
                            }
                            if let Some((a, bs)) = &alibi_v {
                                let slope = a[bi * *bs + h];
                                if slope != 0.0 {
                                    s -= slope * (pos - j as i32).abs() as f32;
                                }
                            }
                            if s == f32::NEG_INFINITY && m[g] == f32::NEG_INFINITY {
                                continue;
                            }
                            let o_off = (g * lq + qi) * d;
                            let out_row = &mut out_group[o_off..o_off + d];
                            if s > m[g] {
                                let alpha = if m[g].is_finite() { (m[g] - s).exp() } else { 0.0 };
                                for v in out_row.iter_mut() {
                                    *v *= alpha;
                                }
                                lsum[g] = lsum[g] * alpha + 1.0;
                                m[g] = s;
                                f32::vec_fmadd_to_f32(out_row, &v_buf[..d], vscale);
                            } else {
                                let p = (s - m[g]).exp();
                                lsum[g] += p;
                                f32::vec_fmadd_to_f32(out_row, &v_buf[..d], p * vscale);
                            }
                        }
                    }
                    for g in 0..group {
                        if lsum[g] > 0.0 {
                            let inv = 1.0 / lsum[g];
                            let o_off = (g * lq + qi) * d;
                            for v in &mut out_group[o_off..o_off + d] {
                                *v *= inv;
                            }
                        }
                    }
                }
            });
    } else {
        out.par_chunks_mut(d).enumerate().for_each(|(row_idx, out_row)| {
            let bh = row_idx / lq;
            let qi = row_idx % lq;
            let bi = bh / hq;
            let h = bh % hq;
            let hk = h / group;
            let qrow = &qr_v[row_idx * d..(row_idx + 1) * d];
            let qsk = q_sk.as_ref().map(|v| &v[row_idx * d..(row_idx + 1) * d]);
            let slope = alibi_v
                .as_ref()
                .map(|(a, bs)| a[bi * bs + h])
                .unwrap_or(0.0);

            let pos = qi as i32 + lk as i32 - lq as i32;
            let mut lo = if wl >= 0 { (pos - wl).max(0) } else { 0 };
            let mut hi = if wr >= 0 { (pos + wr + 1).min(lk as i32) } else { lk as i32 };
            if lo > lk as i32 {
                lo = lk as i32;
            }
            if hi < lo {
                hi = lo;
            }
            let k_head = (bi * hkv + hk) * cap * kw;
            let v_head = (bi * hkv + hk) * cap * vw;
            let mut m = f32::NEG_INFINITY;
            let mut lsum = 0.0f32;
            out_row.fill(0.0);

            for j in (lo as usize)..(hi as usize) {
                let kr = &k_words[k_head + j * kw..k_head + (j + 1) * kw];
                let r0 = f32::from_bits(kr[k_side.rsc_offset]);
                let mut sc = r0 * decode_key_dot(kr, qrow);
                if let Some(sk) = qsk {
                    let mut sdot = 0.0f32;
                    for dd in 0..d {
                        let bit = (kr[k_side.code_words + (dd >> 5)] >> (dd & 31)) & 1;
                        sdot += if bit != 0 { sk[dd] } else { -sk[dd] };
                    }
                    let r1 = f32::from_bits(kr[k_side.rsc_offset + 1]);
                    sc += r0 * r1 * QJL_FACTOR / (d as f32) * sdot;
                }
                let mut s = sc * scale;
                if softcap > 0.0 {
                    s = softcap * (s / softcap).tanh();
                }
                if slope != 0.0 {
                    s -= slope * (pos - j as i32).abs() as f32;
                }
                if s == f32::NEG_INFINITY && m == f32::NEG_INFINITY {
                    continue;
                }
                let vr = &v_words[v_head + j * vw..v_head + (j + 1) * vw];
                let vscale = f32::from_bits(vr[v_side.rsc_offset]);
                if s > m {
                    let alpha = if m.is_finite() { (m - s).exp() } else { 0.0 };
                    for v in out_row.iter_mut() {
                        *v *= alpha;
                    }
                    lsum = lsum * alpha + 1.0;
                    m = s;
                    accum_val(vr, vscale, out_row);
                } else {
                    let p = (s - m).exp();
                    lsum += p;
                    accum_val(vr, p * vscale, out_row);
                }
            }
            if lsum > 0.0 {
                let inv = 1.0 / lsum;
                for v in out_row.iter_mut() {
                    *v *= inv;
                }
            }
        });
    }

    let o = Tensor::from_vec(out, (b, hq, lq, d), q.device())?;
    Ok(Some(vq.unrotate(&o)?.to_dtype(q.dtype())?))
}

/// The fused kernel over `kd` / `vd` (`(b, hkv, capacity, words)`, contiguous, same capacity),
/// `Ok(None)` when it does not apply (kernels off, prefill on CPU, shape out of range).
fn fused_attention_rows(
    q: &Tensor,
    kd: &Tensor,
    vd: &Tensor,
    lk: usize,
    kq: &TurboQuantizer,
    vq: &TurboQuantizer,
    cfg: &FlashAttnConfig,
) -> Result<Option<Tensor>> {
    if q.device().is_cpu() {
        let allow_fused = cfg
            .fused
            .unwrap_or_else(|| accel::config().flash_attn_allowed(q.device()));
        let lq = q.dim(2)?;
        if !allow_fused || lq > 8 {
            return Ok(None);
        }
        return cpu_attention_packed_fused(q, kd, vd, lk, kq, vq, cfg);
    }
    let (b, _, _, _) = q.dims4()?;
    let (bk, hkv, cap, kw) = kd.dims4()?;
    let vw = vd.dim(3)?;
    if bk != b || vd.dim(2)? != cap || lk > cap {
        return Ok(None);
    }
    let conf = accel::config();
    let Some(p) = tq_prepare(
        q,
        hkv,
        lk,
        [hkv * cap * kw, cap * kw, kw],
        [hkv * cap * vw, cap * vw, vw],
        kq,
        vq,
        cfg,
        &conf,
    )?
    else {
        return Ok(None);
    };
    let o = p.qx.apply_op3_no_bwd(
        kd,
        vd,
        &TqAttn {
            args: p.args,
            tab: p.tab,
            alibi: p.alibi,
        },
    )?;
    Ok(Some(vq.unrotate(&o)?.to_dtype(q.dtype())?))
}

/// The paged kernel over the blocks (`tq_attnp*`), `Ok(None)` when it does not apply.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm")),
    allow(unused_variables, clippy::needless_return)
)]
fn paged_attention_rows(
    q: &Tensor,
    blocks: &[KvBlockRef<'_>],
    hkv: usize,
    lk: usize,
    kq: &TurboQuantizer,
    vq: &TurboQuantizer,
    cfg: &FlashAttnConfig,
    conf: &accel::AccelConfig,
) -> Result<Option<Tensor>> {
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    return Ok(None);
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        use att::PagedKv;
        if !matches!(q.device(), Device::Cuda(_) | Device::Rocm(_))
            || blocks.len() > att::FA_MAX_PAGES
            || blocks.iter().any(|x| {
                !x.k.device().same_device(q.device()) || !x.v.device().same_device(q.device())
            })
        {
            return Ok(None);
        }
        // (rows, k offset, k capacity, v offset, v capacity) of the non-empty blocks
        let mut place = Vec::with_capacity(blocks.len());
        let (mut k_row, mut v_row) = (None, None);
        for blk in blocks.iter().filter(|x| x.rows > 0) {
            let (Some((ko, kr, kc)), Some((vo, vr, vc))) = (
                paged_block_layout(blk.k, blk.rows)?,
                paged_block_layout(blk.v, blk.rows)?,
            ) else {
                return Ok(None);
            };
            if *k_row.get_or_insert(kr) != kr || *v_row.get_or_insert(vr) != vr {
                return Ok(None);
            }
            place.push((blk.rows, ko, kc, vo, vc));
        }
        let (Some(k_row), Some(v_row)) = (k_row, v_row) else {
            return Ok(None);
        };
        let Some(p) = tq_prepare(
            q,
            hkv,
            lk,
            [0, 0, k_row],
            [0, 0, v_row],
            kq,
            vq,
            cfg,
            conf,
        )?
        else {
            return Ok(None);
        };
        let (qg, ql) = p.qx.storage_and_layout();
        let mut a = p.args;
        a.q_offset = contiguous_offset(ql, "the queries")?;
        let guards: Vec<_> = blocks
            .iter()
            .filter(|x| x.rows > 0)
            .map(|blk| (blk.k.storage_and_layout().0, blk.v.storage_and_layout().0))
            .collect();
        let alibi_g = p.alibi.as_ref().map(|t| t.storage_and_layout());
        macro_rules! launch {
            ($variant:ident, $qs:ident) => {{
                let mut pk = Vec::with_capacity(place.len());
                for (&(rows, ko, kc, vo, vc), (kg, vg)) in place.iter().zip(guards.iter()) {
                    let (candle::Storage::$variant(ks), candle::Storage::$variant(vs)) =
                        (&**kg, &**vg)
                    else {
                        bail!("turbo-quant blocks: a block is on another device")
                    };
                    pk.push(PagedKv {
                        k: ks,
                        k_offset: ko,
                        v: vs,
                        v_offset: vo,
                        rows,
                        k_cap: kc,
                        v_cap: vc,
                    });
                }
                let alibi = match alibi_g.as_ref() {
                    Some((g, l)) => {
                        let candle::Storage::$variant(s) = &**g else {
                            bail!("turbo-quant: the alibi slopes are on another device")
                        };
                        Some((s, l.start_offset()))
                    }
                    None => None,
                };
                let out = with_tab!(p.tab, $variant, |t, to| $qs
                    .tq_attention_paged(&pk, t, to, alibi, &a))?;
                candle::Storage::$variant(out)
            }};
        }
        let out = match &*qg {
            #[cfg(feature = "cuda")]
            candle::Storage::Cuda(qs) => launch!(Cuda, qs),
            #[cfg(feature = "rocm")]
            candle::Storage::Rocm(qs) => launch!(Rocm, qs),
            _ => return Ok(None),
        };
        let o = Tensor::from_storage(
            out,
            (a.b, a.hq, a.lq, a.d),
            candle::op::BackpropOp::none(),
            false,
        );
        Ok(Some(vq.unrotate(&o)?.to_dtype(q.dtype())?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::accel::turbo::SplitMix64;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let mut rng = SplitMix64::new(seed);
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.next_normal() as f32).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
        let e = (a - b)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let n = b
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        (e / n).sqrt()
    }

    #[test]
    fn quantizer_roundtrip_all_formats() {
        let x = rand(&[3, 5, 64], 1);
        for (f, rot, tol) in [
            (
                QuantFormat::TurboMse { bits: 4 },
                RotationKind::Hadamard,
                0.12,
            ),
            (QuantFormat::TurboMse { bits: 2 }, RotationKind::Dense, 0.4),
            (QuantFormat::TurboProd { bits: 4 }, RotationKind::Auto, 0.3),
            (QuantFormat::Nvfp4, RotationKind::Hadamard, 0.12),
            (QuantFormat::Mxfp4, RotationKind::None, 0.15),
            (QuantFormat::Mxfp8, RotationKind::Hadamard, 0.04),
            (QuantFormat::Fp8, RotationKind::None, 0.04),
        ] {
            let q = TurboQuantizer::new(f, 64, rot, 3, &Device::Cpu).unwrap();
            let qt = q.quantize(&x).unwrap();
            let back = q.dequantize(&qt).unwrap();
            assert_eq!(back.dims(), x.dims());
            let e = rel_err(&back, &x);
            assert!(e < tol, "{f} {rot:?}: relative error {e}");
        }
    }

    #[test]
    fn cpu_ops_match_codec() {
        // the op graph (stage one, residual, projection, signs) gives the codec's rows
        let x = rand(&[7, 32], 2);
        let q = TurboQuantizer::new(
            QuantFormat::TurboProd { bits: 3 },
            32,
            RotationKind::None,
            5,
            &Device::Cpu,
        )
        .unwrap();
        let via_ops = q
            .quantize_gpu(&x, 7)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let via_codec = q
            .quantize_cpu(&x, 7)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(via_ops, via_codec);
        let data = q.quantize_cpu(&x, 7).unwrap();
        let a = q.dequantize_gpu(&data, 7).unwrap();
        let b = q.dequantize_cpu(&data, 7).unwrap();
        assert!(rel_err(&a, &b) < 1e-5);
    }

    #[test]
    fn kv_cache_attention_close_to_exact() {
        for (kf, tol) in [
            (QuantFormat::TurboMse { bits: 4 }, 0.25),
            // the QJL sketch is unbiased but noisier: larger attention error at equal bits
            (QuantFormat::TurboProd { bits: 4 }, 0.4),
        ] {
            kv_cache_case(kf, tol);
        }
    }

    fn kv_cache_case(kf: QuantFormat, tol: f32) {
        let (b, hq, hkv, d) = (1, 4, 2, 64);
        let mut cache = TurboKvCache::with_formats(
            kf,
            QuantFormat::TurboMse { bits: 4 },
            d,
            RotationKind::Auto,
            7,
            16,
            &Device::Cpu,
        )
        .unwrap();
        let k1 = rand(&[b, hkv, 20, d], 3);
        let v1 = rand(&[b, hkv, 20, d], 4);
        let k2 = rand(&[b, hkv, 1, d], 5);
        let v2 = rand(&[b, hkv, 1, d], 6);
        cache.append(&k1, &v1).unwrap();
        cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len(), 21);
        let q = rand(&[b, hq, 1, d], 8);
        let cfg = FlashAttnConfig::new(0.125).causal();
        let got = cache.attention(&q, &cfg).unwrap();
        let k = Tensor::cat(&[&k1, &k2], 2).unwrap();
        let v = Tensor::cat(&[&v1, &v2], 2).unwrap();
        let exact = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
        assert!(
            rel_err(&got, &exact) < tol,
            "{kf}: {}",
            rel_err(&got, &exact)
        );
        // 4-bit rows: about 4.5 bits per value instead of 32
        assert!(cache.storage_bytes() < 2 * 32 * hkv * d * 4 / 6);
    }

    /// The paged entry points concatenate the valid rows on the CPU (and on GPUs when the paged
    /// kernels do not apply): same results as the dense functions; bad blocks are errors.
    #[test]
    fn attention_blocks_cpu_fallback() {
        use crate::flash_attn::flash_attn_blocks;
        let (b, hq, hkv, d) = (1, 4, 2, 64);
        let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &Device::Cpu)
            .unwrap();
        let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 5, &Device::Cpu)
            .unwrap();
        let pack = |q: &TurboQuantizer, x: &Tensor| {
            let (b, h, t, d) = x.dims4().unwrap();
            q.quantize_rows(&x.reshape((b * h * t, d)).unwrap())
                .unwrap()
                .reshape((b, h, t, q.row_words()))
                .unwrap()
        };
        let (k1, v1) = (rand(&[b, hkv, 16, d], 1), rand(&[b, hkv, 16, d], 2));
        let (k2, v2) = (rand(&[b, hkv, 12, d], 3), rand(&[b, hkv, 12, d], 4));
        let q = rand(&[b, hq, 1, d], 5);
        let cfg = FlashAttnConfig::new(0.125).causal();
        // dense blocks: 16 rows, then 9 of 12
        let blocks = [
            KvBlockRef { k: &k1, v: &v1, rows: 16 },
            KvBlockRef { k: &k2, v: &v2, rows: 9 },
        ];
        let k = Tensor::cat(&[&k1, &k2.narrow(2, 0, 9).unwrap()], 2).unwrap();
        let v = Tensor::cat(&[&v1, &v2.narrow(2, 0, 9).unwrap()], 2).unwrap();
        let got = flash_attn_blocks(&q, &blocks, &cfg).unwrap();
        let want = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
        assert!(rel_err(&got, &want) < 1e-6);
        // packed blocks
        let (pk1, pv1, pk2, pv2) = (pack(&kq, &k1), pack(&vq, &v1), pack(&kq, &k2), pack(&vq, &v2));
        let pblocks = [
            KvBlockRef { k: &pk1, v: &pv1, rows: 16 },
            KvBlockRef { k: &pk2, v: &pv2, rows: 9 },
        ];
        let kr = Tensor::cat(&[&pk1, &pk2.narrow(2, 0, 9).unwrap()], 2).unwrap();
        let vr = Tensor::cat(&[&pv1, &pv2.narrow(2, 0, 9).unwrap()], 2).unwrap();
        let got = attention_packed_blocks(&q, &pblocks, &kq, &vq, &cfg).unwrap();
        let want = attention_packed(&q, &kr, &vr, 25, &kq, &vq, &cfg).unwrap();
        assert!(rel_err(&got, &want) < 1e-6);
        // errors: no blocks, more rows than capacity, shape mismatch, no keys
        assert!(attention_packed_blocks(&q, &[], &kq, &vq, &cfg).is_err());
        let bad = [KvBlockRef { k: &pk2, v: &pv2, rows: 13 }];
        assert!(attention_packed_blocks(&q, &bad, &kq, &vq, &cfg).is_err());
        let mixed = [KvBlockRef { k: &pk1, v: &pv1, rows: 3 }, KvBlockRef { k: &k1, v: &v1, rows: 3 }];
        assert!(attention_packed_blocks(&q, &mixed, &kq, &vq, &cfg).is_err());
        let empty = [KvBlockRef { k: &k1, v: &v1, rows: 0 }];
        assert!(flash_attn_blocks(&q, &empty, &cfg).is_err());
    }

    /// `attention_packed` over rows packed outside a cache (concatenated blocks, spare capacity)
    /// computes the same attention as the cache holding the same rows.
    #[test]
    fn attention_packed_matches_cache() {
        let (b, hq, hkv, d) = (1, 4, 2, 64);
        for (kf, vf, rot) in [
            (QuantFormat::Fp8, QuantFormat::Fp8, RotationKind::None),
            (
                QuantFormat::TurboMse { bits: 4 },
                QuantFormat::TurboMse { bits: 4 },
                RotationKind::Hadamard,
            ),
        ] {
            let seed = 11;
            let mut cache =
                TurboKvCache::with_formats(kf, vf, d, rot, seed, 8, &Device::Cpu).unwrap();
            let kq = TurboQuantizer::new(kf, d, rot, seed, &Device::Cpu).unwrap();
            let vq = TurboQuantizer::new(vf, d, rot, seed.wrapping_add(2), &Device::Cpu).unwrap();
            let pack = |q: &TurboQuantizer, x: &Tensor| {
                let (b, h, t, d) = x.dims4().unwrap();
                q.quantize_rows(&x.reshape((b * h * t, d)).unwrap())
                    .unwrap()
                    .reshape((b, h, t, q.row_words()))
                    .unwrap()
            };
            // two "blocks" of 13 and 7 tokens
            let (k1, v1) = (rand(&[b, hkv, 13, d], 21), rand(&[b, hkv, 13, d], 22));
            let (k2, v2) = (rand(&[b, hkv, 7, d], 23), rand(&[b, hkv, 7, d], 24));
            cache.append(&k1, &v1).unwrap();
            cache.append(&k2, &v2).unwrap();
            let k_rows = Tensor::cat(&[&pack(&kq, &k1), &pack(&kq, &k2)], 2).unwrap();
            let v_rows = Tensor::cat(&[&pack(&vq, &v1), &pack(&vq, &v2)], 2).unwrap();
            for lq in [1usize, 5] {
                let q = rand(&[b, hq, lq, d], 30 + lq as u64);
                let cfg = FlashAttnConfig::new(0.125).causal();
                let want = cache.attention(&q, &cfg).unwrap();
                let got = attention_packed(&q, &k_rows, &v_rows, 20, &kq, &vq, &cfg).unwrap();
                assert!(
                    rel_err(&got, &want) < 1e-5,
                    "{kf} lq {lq}: {}",
                    rel_err(&got, &want)
                );
                // spare capacity after the valid rows is ignored
                let junk_k = pack(&kq, &rand(&[b, hkv, 3, d], 99));
                let junk_v = pack(&vq, &rand(&[b, hkv, 3, d], 98));
                let k_cap = Tensor::cat(&[&k_rows, &junk_k], 2).unwrap();
                let v_cap = Tensor::cat(&[&v_rows, &junk_v], 2).unwrap();
                let got = attention_packed(&q, &k_cap, &v_cap, 20, &kq, &vq, &cfg).unwrap();
                assert!(
                    rel_err(&got, &want) < 1e-5,
                    "{kf} lq {lq} capacity: {}",
                    rel_err(&got, &want)
                );
            }
            // shape / format mismatches are errors, not silent garbage
            let q = rand(&[b, hq, 1, d], 40);
            let cfg = FlashAttnConfig::new(0.125);
            assert!(attention_packed(&q, &k_rows, &v_rows, 21, &kq, &vq, &cfg).is_err());
            assert!(attention_packed(
                &q,
                &v_rows,
                &k_rows.narrow(3, 0, 1).unwrap(),
                20,
                &kq,
                &vq,
                &cfg
            )
            .is_err());
        }
    }
}

#[cfg(test)]
mod accuracy {
    use super::*;
    use candle::accel::turbo::SplitMix64;

    /// Attention error per KV format (the table of the README):
    /// `cargo test -p candle-nn --lib turbo_quant::accuracy -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attention_error_by_format() {
        let d = 128;
        for (kf, vf) in [
            (QuantFormat::Fp8, QuantFormat::Fp8),
            (QuantFormat::Nvfp4, QuantFormat::Nvfp4),
            (
                QuantFormat::TurboMse { bits: 4 },
                QuantFormat::TurboMse { bits: 4 },
            ),
            (QuantFormat::Mxfp4, QuantFormat::Mxfp4),
            (
                QuantFormat::TurboProd { bits: 4 },
                QuantFormat::TurboMse { bits: 4 },
            ),
            (
                QuantFormat::TurboMse { bits: 3 },
                QuantFormat::TurboMse { bits: 3 },
            ),
        ] {
            let mut total = 0f32;
            for seed in 0..6u64 {
                let mut rng = SplitMix64::new(seed * 77 + d as u64);
                let mut r = |s: &[usize]| {
                    let n: usize = s.iter().product();
                    let v: Vec<f32> = (0..n).map(|_| rng.next_normal() as f32).collect();
                    Tensor::from_vec(v, s, &Device::Cpu).unwrap()
                };
                let (k, v, q) = (r(&[1, 2, 256, d]), r(&[1, 2, 256, d]), r(&[1, 8, 1, d]));
                let mut c = TurboKvCache::with_formats(
                    kf,
                    vf,
                    d,
                    RotationKind::Auto,
                    seed,
                    64,
                    &Device::Cpu,
                )
                .unwrap();
                c.append(&k, &v).unwrap();
                let cfg = FlashAttnConfig::new(1.0 / (d as f32).sqrt());
                let got = c.attention(&q, &cfg).unwrap();
                let exact = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                let e = (&got - &exact)
                    .unwrap()
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                let n = exact
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                total += (e / n).sqrt();
            }
            let bits = kf.bits_per_element(d);
            println!(
                "{kf} / {vf}: {bits} bits per key value, relative error {:.3}",
                total / 6.0
            );
        }
    }

    /// Odd head dimension (`d = 6`, not divisible by 4) FP8 fused CPU TurboQuant must not drop
    /// tail elements (`Q3-F01`), and `fused=true` must match `fused=false` (dequant + reference)
    /// across all formats including NVFP4/MXFP4 (`Q6-F04`).
    #[test]
    fn cpu_fused_tq_odd_fp8_and_all_formats_match_unfused() {
        let rand = |shape: &[usize], seed: u64| -> Tensor {
            let mut rng = SplitMix64::new(seed);
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| rng.next_normal() as f32).collect();
            Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
        };
        let rel_err = |a: &Tensor, b: &Tensor| -> f32 {
            let e = (a - b).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let n = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            (e / n).sqrt()
        };
        // 1) Odd dim d = 6 FP8 tail check (Q3-F01)
        let (b, hq, hkv, lk, d_odd) = (1usize, 4, 2, 17, 6);
        let kq_odd = TurboQuantizer::new(QuantFormat::Fp8, d_odd, RotationKind::None, 11, &Device::Cpu).unwrap();
        let vq_odd = TurboQuantizer::new(QuantFormat::Fp8, d_odd, RotationKind::None, 12, &Device::Cpu).unwrap();
        let k_odd = rand(&[b, hkv, lk, d_odd], 101);
        let v_odd = rand(&[b, hkv, lk, d_odd], 102);
        let q_odd = rand(&[b, hq, 3, d_odd], 103);
        let kr_odd = kq_odd.quantize_rows(&k_odd.reshape((b * hkv * lk, d_odd)).unwrap()).unwrap().reshape((b, hkv, lk, kq_odd.row_words())).unwrap();
        let vr_odd = vq_odd.quantize_rows(&v_odd.reshape((b * hkv * lk, d_odd)).unwrap()).unwrap().reshape((b, hkv, lk, vq_odd.row_words())).unwrap();
        let cfg_fused = FlashAttnConfig::new(1.0 / (d_odd as f32).sqrt()).causal().fused(true);
        let cfg_ref = FlashAttnConfig::new(1.0 / (d_odd as f32).sqrt()).causal().fused(false);
        let out_f = attention_packed(&q_odd, &kr_odd, &vr_odd, lk, &kq_odd, &vq_odd, &cfg_fused).unwrap();
        let out_r = attention_packed(&q_odd, &kr_odd, &vr_odd, lk, &kq_odd, &vq_odd, &cfg_ref).unwrap();
        let err = rel_err(&out_f, &out_r);
        assert!(err <= 1e-5, "odd-dim d=6 FP8 fused vs unfused rel_err={err}");

        // 2) Full format matrix: Fp8 / Nvfp4 / Mxfp4 / Mxfp8 / TurboMse / TurboProd x lq {1, 5}
        let d = 64usize;
        let k = rand(&[1, 2, 33, d], 201);
        let v = rand(&[1, 2, 33, d], 202);
        for (kf, vf) in [
            (QuantFormat::Fp8, QuantFormat::Fp8),
            (QuantFormat::Nvfp4, QuantFormat::Nvfp4),
            (QuantFormat::Nvfp4, QuantFormat::Fp8),
            (QuantFormat::Mxfp4, QuantFormat::Mxfp8),
            (QuantFormat::TurboMse { bits: 4 }, QuantFormat::TurboMse { bits: 4 }),
            (QuantFormat::TurboProd { bits: 4 }, QuantFormat::TurboMse { bits: 4 }),
        ] {
            let kq = TurboQuantizer::new(kf, d, RotationKind::Hadamard, 31, &Device::Cpu).unwrap();
            let vq = TurboQuantizer::new(vf, d, RotationKind::Hadamard, 37, &Device::Cpu).unwrap();
            let kr = kq.quantize_rows(&k.reshape((2 * 33, d)).unwrap()).unwrap().reshape((1, 2, 33, kq.row_words())).unwrap();
            let vr = vq.quantize_rows(&v.reshape((2 * 33, d)).unwrap()).unwrap().reshape((1, 2, 33, vq.row_words())).unwrap();
            for lq in [1usize, 5] {
                let q = rand(&[1, 6, lq, d], 300 + lq as u64);
                let base = FlashAttnConfig::new(1.0 / (d as f32).sqrt()).causal().window(Some(17), Some(0)).softcap(30.0);
                let out_f = attention_packed(&q, &kr, &vr, 33, &kq, &vq, &base.clone().fused(true)).unwrap();
                let out_r = attention_packed(&q, &kr, &vr, 33, &kq, &vq, &base.fused(false)).unwrap();
                let e = rel_err(&out_f, &out_r);
                assert!(e <= 1e-5, "CPU TQ fused vs unfused {kf}/{vf} lq={lq}: rel_err={e}");
            }
        }
    }
}
