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

use crate::flash_attn::{flash_attn_bhsd, FlashAttnConfig};
use crate::kv_cache::Cache;
use candle::accel::attention as att;
use candle::accel::formats::{row_layout, RowLayout};
use candle::accel::turbo::{qjl_matrix, rotation_matrix, RowCodec, TqAttnArgs, TqRowArgs, TqSide};
use candle::accel::{self, RotationKind, TurboQuantOptions};
use candle::backend::BackendStorage;
use candle::{bail, CpuStorage, DType, Device, Layout, Result, Shape, Tensor};
use rayon::prelude::*;
use std::sync::Arc;

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
    dim: usize,
    layout: RowLayout,
    codec: Arc<RowCodec>,
    /// `Π` and `Πᵀ` (`dim x dim`), `None` without rotation.
    rot: Option<(Tensor, Tensor)>,
    /// `S` and `Sᵀ` of the QJL stage (prod only).
    qjl: Option<(Tensor, Tensor)>,
    /// Codebook table (32 f32).
    tab: Tensor,
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
        let rot = if rotation == RotationKind::None {
            None
        } else {
            let m = Tensor::from_vec(rotation_matrix(dim, rotation, seed)?, (dim, dim), device)?;
            let mt = m.t()?.contiguous()?;
            Some((m, mt))
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
            dim,
            layout,
            codec,
            rot,
            qjl,
            tab,
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

    /// `x Πᵀ` in f32 (rows of the rotated domain).
    fn rotate(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        match &self.rot {
            Some((_, mt)) => x.broadcast_matmul(mt),
            None => x.contiguous(),
        }
    }

    /// `y Π`: back from the rotated domain.
    fn unrotate(&self, y: &Tensor) -> Result<Tensor> {
        match &self.rot {
            Some((m, _)) => y.broadcast_matmul(m),
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
            if fused
                && !q.device().is_cpu()
                && accel::config().turbo_quant_kernels_allowed(q.device())
            {
                match self.fused_attention(q, cfg, kq, vq) {
                    Ok(Some(o)) => return Ok(o),
                    Ok(None) => {}
                    Err(e) => warn_cpu_fallback("attention", &e),
                }
            }
        }
        let Some((k, v)) = self.keys_values()? else {
            bail!("turbo-quant cache: attention on an empty cache")
        };
        flash_attn_bhsd(q, &k.to_dtype(q.dtype())?, &v.to_dtype(q.dtype())?, cfg)
    }

    fn fused_attention(
        &self,
        q: &Tensor,
        cfg: &FlashAttnConfig,
        kq: &TurboQuantizer,
        vq: &TurboQuantizer,
    ) -> Result<Option<Tensor>> {
        let (Some(kd), Some(vd)) = (self.k.all_data().as_ref(), self.v.all_data().as_ref()) else {
            return Ok(None);
        };
        let (b, hq, lq, d) = q.dims4()?;
        let (bk, hkv, cap, kw) = kd.dims4()?;
        let vw = vd.dim(3)?;
        if bk != b || d != self.dim || hkv == 0 || hq % hkv != 0 || d > att::FA_MAX_HEAD_DIM {
            return Ok(None);
        }
        let lk = self.seq_len();
        // rotated queries, plus S·q for prod keys
        let qr = kq.rotate(q)?;
        let qx = match &kq.qjl {
            Some((_, st)) => Tensor::cat(&[&qr, &qr.broadcast_matmul(st)?], 3)?,
            None => qr,
        }
        .contiguous()?;
        let w = qx.dim(3)?;
        let conf = accel::config();
        let cus = accel::profile(q.device()).compute_units;
        let (wl, wr) = (
            cfg.window_size_left
                .map(|v| v.min(i32::MAX as usize) as i32)
                .unwrap_or(-1),
            cfg.window_size_right
                .map(|v| v.min(i32::MAX as usize) as i32)
                .unwrap_or(-1),
        );
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
            k_strides: [hkv * cap * kw, cap * kw, kw],
            k_offset: 0,
            v_strides: [hkv * cap * vw, cap * vw, vw],
            v_offset: 0,
            k: TqSide::new(kq.format(), d)?,
            v: TqSide::new(vq.format(), d)?,
            scale: cfg.softmax_scale,
            softcap: cfg.softcap.unwrap_or(0.0),
            window_left: wl,
            window_right: wr,
            splits: att::split_count_for(
                b * hq * lq,
                lk,
                cfg.split_kv.unwrap_or(conf.flash_attn.split_kv),
                cus,
            ),
            alibi_bstride,
        };
        let tab = Tensor::cat(&[&kq.tab, &vq.tab], 0)?;
        let o = qx.apply_op3_no_bwd(kd, vd, &TqAttn { args, tab, alibi })?;
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
}
