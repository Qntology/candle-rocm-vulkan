//! Gated delta rule (Gated DeltaNet linear attention of Qwen3.5 / Qwen3-Next / Qwen4-Exp).
//!
//! [`fused_gated_delta_rule`] runs a whole linear-attention mixer step after the input
//! projections in one kernel launch on CUDA (NVRTC) and ROCm: causal depthwise conv1d over the
//! previous conv state and the new tokens, SiLU, q / k L2 normalization, the gates
//! `beta = sigmoid(b)` and `g = neg_a * softplus(a + dt_bias)`, and the sequential gated delta
//! rule scan with the recurrent state kept on chip. The CPU implementation (rayon over
//! (batch, value head)) follows the transformers reference
//! (`torch_recurrent_gated_delta_rule`) operation by operation and is the test oracle. Vulkan
//! and Metal return an error ([`fused_gdn_supported`] is false) so that callers keep their
//! tensor-op path.

#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::accel::delta::{self as gd, GdnScanArgs};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::{op::BackpropOp, Storage};
use crate::{bail, DType, Device, Result, Tensor};
use rayon::prelude::*;

pub use crate::accel::delta::GDN_MAX_HEAD_K_DIM;

/// Which key head a value head reads when there are more value heads than key heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GdnHeadOrder {
    /// Value head `h` uses key head `h / (Hv / Hk)` (torch `repeat_interleave`, transformers).
    #[default]
    Grouped,
    /// Value head `h` uses key head `h % Hk` (llama.cpp re-ordered GGUFs).
    Tiled,
}

/// Geometry of a Gated DeltaNet layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    /// Causal depthwise conv width (4 for Qwen3.5).
    pub conv_kernel: usize,
    /// L2 normalization of q and k (true for Qwen3.5 / Qwen4-Exp).
    pub use_qk_l2norm: bool,
    /// Epsilon inside the square root of the L2 norm (1e-6).
    pub l2norm_eps: f32,
    pub v_head_order: GdnHeadOrder,
}

impl GdnConfig {
    /// Qwen3.5 defaults: conv width 4, L2-normalized q / k with eps 1e-6, grouped heads.
    pub fn new(num_k_heads: usize, num_v_heads: usize, head_k_dim: usize, head_v_dim: usize) -> Self {
        Self {
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel: 4,
            use_qk_l2norm: true,
            l2norm_eps: 1e-6,
            v_head_order: GdnHeadOrder::Grouped,
        }
    }

    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_k_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }

    /// Channels of `mixed_qkv`: `2 * Hk * Dk + Hv * Dv` (q, then k, then v).
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Key head read by value head `v_head`.
    pub fn k_head(&self, v_head: usize) -> usize {
        match self.v_head_order {
            GdnHeadOrder::Grouped => v_head / (self.num_v_heads / self.num_k_heads),
            GdnHeadOrder::Tiled => v_head % self.num_k_heads,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.num_k_heads == 0
            || self.num_v_heads == 0
            || self.head_k_dim == 0
            || self.head_v_dim == 0
            || self.conv_kernel == 0
        {
            bail!("gated delta rule: empty geometry {self:?}")
        }
        if self.num_v_heads % self.num_k_heads != 0 {
            bail!(
                "gated delta rule: num_v_heads {} is not a multiple of num_k_heads {}",
                self.num_v_heads,
                self.num_k_heads
            )
        }
        Ok(())
    }
}

/// Result of [`fused_gated_delta_rule`].
#[derive(Debug, Clone)]
pub struct GdnOutput {
    /// `[B, T, Hv, Dv]` in the dtype of `mixed_qkv` (before the gated norm).
    pub out: Tensor,
    /// `[B, C, K-1]` in the dtype of `mixed_qkv`: the last `K-1` conv inputs.
    pub conv_state: Tensor,
    /// `[B, Hv, Dk, Dv]` f32.
    pub recurrent_state: Tensor,
}

/// Whether [`fused_gated_delta_rule`] / [`fused_gated_rms_norm`] have a native implementation
/// on `device` (CPU, and CUDA / ROCm builds that carry the kernels). Callers keep their tensor
/// path when this is false.
pub fn fused_gdn_supported(device: &Device) -> bool {
    match device {
        Device::Cpu => true,
        Device::Cuda(_) => cfg!(feature = "cuda"),
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => !crate::rocm_backend::kernels::Module::DeltaRule
            .image()
            .is_empty(),
        _ => false,
    }
}

/// [`fused_gdn_supported`] for a given geometry (the GPU kernels take `head_k_dim <=`
/// [`GDN_MAX_HEAD_K_DIM`]).
pub fn fused_gdn_supported_for(device: &Device, cfg: &GdnConfig) -> bool {
    fused_gdn_supported(device)
        && cfg.validate().is_ok()
        && (device.is_cpu() || cfg.head_k_dim <= GDN_MAX_HEAD_K_DIM)
}

fn float_dtype(dtype: DType, what: &str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        d => bail!("gated delta rule: {what} must be f32 / f16 / bf16, got {d:?}"),
    }
}

fn expect_dims(t: &Tensor, want: &[usize], what: &str) -> Result<()> {
    if t.dims() != want {
        bail!(
            "gated delta rule: {what} has shape {:?}, expected {want:?}",
            t.dims()
        )
    }
    Ok(())
}

fn expect_device(t: &Tensor, dev: &Device, what: &str) -> Result<()> {
    if !t.device().same_device(dev) {
        bail!("gated delta rule: {what} is on {:?}, expected {dev:?}", t.device().location())
    }
    Ok(())
}

/// One Gated DeltaNet mixer step over `T` new tokens (one launch on CUDA / ROCm):
///
/// 1. causal depthwise conv over `[conv_state ; mixed_qkv]` (+ `conv_bias`), then SiLU;
/// 2. split into q `[Hk, Dk]`, k `[Hk, Dk]`, v `[Hv, Dv]`; L2-normalize q and k
///    (`x / sqrt(sum x^2 + eps)`) when `cfg.use_qk_l2norm`, q *= `1 / sqrt(Dk)`;
/// 3. `beta = sigmoid(b)`, `g = neg_a * softplus(a + dt_bias)` (torch softplus, threshold 20);
/// 4. per token: `S = exp(g) S`, `delta = beta (v - S^T k)`, `S += k delta^T`, `o = S^T q`.
///
/// All math is f32. Shapes: `mixed_qkv` `[B, T, C]` (f32 / f16 / bf16, channel stride 1, any
/// batch / token strides), `conv_weight` `[C, K]` or `[C, 1, K]`, `conv_bias` `[C]`,
/// `conv_state` `[B, C, K-1]` (None = zeros), `a` / `b` `[B, T, Hv]`, `neg_a` (`-exp(A_log)`)
/// and `dt_bias` with `Hv` elements, `recurrent_state` `[B, Hv, Dk, Dv]` (None = zeros). New
/// states are returned out of place; the inputs are not modified.
#[allow(clippy::too_many_arguments)]
pub fn fused_gated_delta_rule(
    mixed_qkv: &Tensor,
    conv_weight: &Tensor,
    conv_bias: Option<&Tensor>,
    conv_state: Option<&Tensor>,
    a: &Tensor,
    b: &Tensor,
    neg_a: &Tensor,
    dt_bias: &Tensor,
    recurrent_state: Option<&Tensor>,
    cfg: &GdnConfig,
) -> Result<GdnOutput> {
    cfg.validate()?;
    let (bsz, t, c) = mixed_qkv.dims3()?;
    if c != cfg.conv_dim() {
        bail!(
            "gated delta rule: mixed_qkv has {c} channels, expected 2 * {} * {} + {} * {} = {}",
            cfg.num_k_heads,
            cfg.head_k_dim,
            cfg.num_v_heads,
            cfg.head_v_dim,
            cfg.conv_dim()
        )
    }
    let dtype = mixed_qkv.dtype();
    float_dtype(dtype, "mixed_qkv")?;
    let dev = mixed_qkv.device().clone();
    let (hv, dk, dv, k) = (cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.conv_kernel);
    let conv_weight = match conv_weight.rank() {
        2 => conv_weight.clone(),
        3 if conv_weight.dim(1)? == 1 => conv_weight.squeeze(1)?,
        _ => bail!(
            "gated delta rule: conv_weight must be [C, K] or [C, 1, K], got {:?}",
            conv_weight.dims()
        ),
    };
    expect_dims(&conv_weight, &[c, k], "conv_weight")?;
    if let Some(cb) = conv_bias {
        expect_dims(cb, &[c], "conv_bias")?;
    }
    if let Some(cs) = conv_state {
        expect_dims(cs, &[bsz, c, k - 1], "conv_state")?;
    }
    expect_dims(a, &[bsz, t, hv], "a")?;
    expect_dims(b, &[bsz, t, hv], "b")?;
    if neg_a.elem_count() != hv || dt_bias.elem_count() != hv {
        bail!(
            "gated delta rule: neg_a {:?} / dt_bias {:?} must have num_v_heads = {hv} elements",
            neg_a.dims(),
            dt_bias.dims()
        )
    }
    if let Some(s) = recurrent_state {
        expect_dims(s, &[bsz, hv, dk, dv], "recurrent_state")?;
    }
    for (x, what) in [
        (&conv_weight, "conv_weight"),
        (a, "a"),
        (b, "b"),
        (neg_a, "neg_a"),
        (dt_bias, "dt_bias"),
    ] {
        expect_device(x, &dev, what)?;
        float_dtype(x.dtype(), what)?;
    }
    for (x, what) in [
        (conv_bias, "conv_bias"),
        (conv_state, "conv_state"),
        (recurrent_state, "recurrent_state"),
    ] {
        if let Some(x) = x {
            expect_device(x, &dev, what)?;
            float_dtype(x.dtype(), what)?;
        }
    }
    if t == 0 {
        let conv_state = match conv_state {
            Some(cs) if cs.elem_count() > 0 => cs.to_dtype(dtype)?.contiguous()?,
            _ => Tensor::zeros((bsz, c, k - 1), dtype, &dev)?,
        };
        let recurrent_state = match recurrent_state {
            Some(s) => s.to_dtype(DType::F32)?.contiguous()?,
            None => Tensor::zeros((bsz, hv, dk, dv), DType::F32, &dev)?,
        };
        return Ok(GdnOutput {
            out: Tensor::zeros((bsz, 0, hv, dv), dtype, &dev)?,
            conv_state,
            recurrent_state,
        });
    }
    let inputs = GdnInputs {
        mixed_qkv,
        conv_weight: &conv_weight,
        conv_bias,
        conv_state,
        a,
        b,
        neg_a,
        dt_bias,
        recurrent_state,
    };
    match &dev {
        Device::Cpu => cpu::gated_delta_rule(&inputs, cfg),
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        Device::Cuda(_) | Device::Rocm(_) => gpu_gated_delta_rule(&inputs, cfg),
        d => bail!(
            "gated delta rule: no fused implementation on {:?}, use the tensor path",
            d.location()
        ),
    }
}

struct GdnInputs<'a> {
    mixed_qkv: &'a Tensor,
    conv_weight: &'a Tensor,
    conv_bias: Option<&'a Tensor>,
    conv_state: Option<&'a Tensor>,
    a: &'a Tensor,
    b: &'a Tensor,
    neg_a: &'a Tensor,
    dt_bias: &'a Tensor,
    recurrent_state: Option<&'a Tensor>,
}

/// `t` with a unit stride on its last dim; returns it with (offset, stride 0, stride 1).
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn rows_view(t: &Tensor) -> Result<(Tensor, [usize; 3])> {
    let t = if t.stride()[2] == 1 { t.clone() } else { t.contiguous()? };
    let off = t.layout().start_offset();
    let st = t.stride();
    let (s0, s1) = (st[0], st[1]);
    Ok((t, [off, s0, s1]))
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn f32_contiguous(t: &Tensor) -> Result<Tensor> {
    t.to_dtype(DType::F32)?.contiguous()
}

#[cfg(feature = "cuda")]
fn as_cuda(s: &Storage) -> Result<&crate::CudaStorage> {
    match s {
        Storage::Cuda(s) => Ok(s),
        _ => bail!("gated delta rule: inputs on different backends"),
    }
}

#[cfg(feature = "rocm")]
fn as_rocm(s: &Storage) -> Result<&crate::RocmStorage> {
    match s {
        Storage::Rocm(s) => Ok(s),
        _ => bail!("gated delta rule: inputs on different backends"),
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn gpu_gated_delta_rule(inp: &GdnInputs, cfg: &GdnConfig) -> Result<GdnOutput> {
    let x = inp.mixed_qkv;
    let dtype = x.dtype();
    let dev = x.device().clone();
    let (bsz, t, c) = x.dims3()?;
    let (hk, hv, dk, dv, k) = (
        cfg.num_k_heads,
        cfg.num_v_heads,
        cfg.head_k_dim,
        cfg.head_v_dim,
        cfg.conv_kernel,
    );
    if dk > GDN_MAX_HEAD_K_DIM {
        bail!("gated delta rule: head_k_dim {dk} > {GDN_MAX_HEAD_K_DIM} has no fused GPU kernel")
    }
    // Gate inputs stay in the activation dtype, else go to f32 (lossless).
    let ab_f32 = inp.a.dtype() != dtype || inp.b.dtype() != dtype;
    let gate = |g: &Tensor| -> Result<Tensor> {
        if ab_f32 {
            g.to_dtype(DType::F32)
        } else {
            Ok(g.clone())
        }
    };
    let (x, xv) = rows_view(x)?;
    let (a, av) = rows_view(&gate(inp.a)?)?;
    let (b, bv) = rows_view(&gate(inp.b)?)?;
    let cw = f32_contiguous(inp.conv_weight)?;
    let cb = inp.conv_bias.map(f32_contiguous).transpose()?;
    // (an empty conv state, conv width 1, is passed as null)
    let cs = inp
        .conv_state
        .filter(|s| s.elem_count() > 0)
        .map(|s| s.to_dtype(dtype)?.contiguous())
        .transpose()?;
    let na = f32_contiguous(&inp.neg_a.flatten_all()?)?;
    let dtb = f32_contiguous(&inp.dt_bias.flatten_all()?)?;
    let st = inp.recurrent_state.map(f32_contiguous).transpose()?;
    let off = |t: &Tensor| t.layout().start_offset();
    let mut flags = 0u32;
    if cfg.use_qk_l2norm {
        flags |= gd::GDN_FLAG_L2NORM;
    }
    if cfg.v_head_order == GdnHeadOrder::Tiled {
        flags |= gd::GDN_FLAG_TILED;
    }
    if ab_f32 {
        flags |= gd::GDN_FLAG_AB_F32;
    }
    let args = GdnScanArgs {
        b: bsz,
        t,
        hk,
        hv,
        dk,
        dv,
        kc: k,
        flags,
        eps: cfg.l2norm_eps,
        scale: (1.0 / (dk as f64).sqrt()) as f32,
        x: xv,
        a: av,
        bb: bv,
        cw_off: off(&cw),
        cb_off: cb.as_ref().map(off).unwrap_or(0),
        cs_off: cs.as_ref().map(off).unwrap_or(0),
        na_off: off(&na),
        dtb_off: off(&dtb),
        s_off: st.as_ref().map(off).unwrap_or(0),
    };
    let gx = x.storage_and_layout().0;
    let ga = a.storage_and_layout().0;
    let gb = b.storage_and_layout().0;
    let gcw = cw.storage_and_layout().0;
    let gcb = cb.as_ref().map(|t| t.storage_and_layout().0);
    let gcs = cs.as_ref().map(|t| t.storage_and_layout().0);
    let gna = na.storage_and_layout().0;
    let gdt = dtb.storage_and_layout().0;
    let gst = st.as_ref().map(|t| t.storage_and_layout().0);
    let (out, cs_out, s_out) = match &*gx {
        #[cfg(feature = "cuda")]
        Storage::Cuda(xs) => {
            let (o, c, s) = xs.gdn_scan(
                as_cuda(&ga)?,
                as_cuda(&gb)?,
                as_cuda(&gcw)?,
                gcb.as_deref().map(as_cuda).transpose()?,
                gcs.as_deref().map(as_cuda).transpose()?,
                as_cuda(&gna)?,
                as_cuda(&gdt)?,
                gst.as_deref().map(as_cuda).transpose()?,
                &args,
            )?;
            (Storage::Cuda(o), Storage::Cuda(c), Storage::Cuda(s))
        }
        #[cfg(feature = "rocm")]
        Storage::Rocm(xs) => {
            let (o, c, s) = xs.gdn_scan(
                as_rocm(&ga)?,
                as_rocm(&gb)?,
                as_rocm(&gcw)?,
                gcb.as_deref().map(as_rocm).transpose()?,
                gcs.as_deref().map(as_rocm).transpose()?,
                as_rocm(&gna)?,
                as_rocm(&gdt)?,
                gst.as_deref().map(as_rocm).transpose()?,
                &args,
            )?;
            (Storage::Rocm(o), Storage::Rocm(c), Storage::Rocm(s))
        }
        _ => bail!(
            "gated delta rule: this build has no fused kernel for {:?}",
            dev.location()
        ),
    };
    let mk = |s, shape: (usize, usize, usize, usize)| {
        Tensor::from_storage(s, shape, BackpropOp::none(), false)
    };
    Ok(GdnOutput {
        out: mk(out, (bsz, t, hv, dv)),
        conv_state: Tensor::from_storage(cs_out, (bsz, c, k - 1), BackpropOp::none(), false),
        recurrent_state: mk(s_out, (bsz, hv, dk, dv)),
    })
}

/// Fused gated RMS norm of Qwen3.5 linear attention:
/// `y = x / sqrt(mean(x^2) + eps) * weight * silu(z)` over the last dim, math in f32, output in
/// the dtype of `x`. `x` and `z` are `[..., D]` (same shape), `weight` is `[D]`. Devices
/// without a native kernel run the same formula with tensor ops.
pub fn fused_gated_rms_norm(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    if x.dims() != z.dims() {
        bail!(
            "gated rms norm: x {:?} and z {:?} shapes differ",
            x.dims(),
            z.dims()
        )
    }
    let d = x.dim(crate::D::Minus1)?;
    if weight.elem_count() != d {
        bail!(
            "gated rms norm: weight {:?} must have {d} elements",
            weight.dims()
        )
    }
    let dtype = x.dtype();
    float_dtype(dtype, "x")?;
    float_dtype(z.dtype(), "z")?;
    float_dtype(weight.dtype(), "weight")?;
    let dev = x.device().clone();
    expect_device(z, &dev, "z")?;
    expect_device(weight, &dev, "weight")?;
    match &dev {
        Device::Cpu => cpu::gated_rms_norm(x, z, weight, eps),
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        Device::Cuda(_) | Device::Rocm(_) => {
            let rows = if d == 0 { 0 } else { x.elem_count() / d };
            let x = x.contiguous()?;
            let z = z.to_dtype(dtype)?.contiguous()?;
            let w = f32_contiguous(&weight.flatten_all()?)?;
            let (xo, zo, wo) = (
                x.layout().start_offset(),
                z.layout().start_offset(),
                w.layout().start_offset(),
            );
            let gx = x.storage_and_layout().0;
            let gz = z.storage_and_layout().0;
            let gw = w.storage_and_layout().0;
            let y = match &*gx {
                #[cfg(feature = "cuda")]
                Storage::Cuda(xs) => Storage::Cuda(xs.gdn_gated_rms_norm(
                    xo,
                    as_cuda(&gz)?,
                    zo,
                    as_cuda(&gw)?,
                    wo,
                    rows,
                    d,
                    eps as f32,
                )?),
                #[cfg(feature = "rocm")]
                Storage::Rocm(xs) => Storage::Rocm(xs.gdn_gated_rms_norm(
                    xo,
                    as_rocm(&gz)?,
                    zo,
                    as_rocm(&gw)?,
                    wo,
                    rows,
                    d,
                    eps as f32,
                )?),
                _ => return gated_rms_norm_tensor_ops(&x, &z, weight, eps),
            };
            Ok(Tensor::from_storage(
                y,
                x.shape().clone(),
                BackpropOp::none(),
                false,
            ))
        }
        _ => gated_rms_norm_tensor_ops(x, z, weight, eps),
    }
}

fn gated_rms_norm_tensor_ops(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let z32 = z.to_dtype(DType::F32)?;
    let w = weight.to_dtype(DType::F32)?.flatten_all()?;
    let ms = x32.sqr()?.mean_keepdim(crate::D::Minus1)?;
    let inv = (ms + eps)?.sqrt()?.recip()?;
    x32.broadcast_mul(&inv)?
        .broadcast_mul(&w)?
        .mul(&z32.silu()?)?
        .to_dtype(x.dtype())
}

mod cpu {
    use super::{GdnConfig, GdnInputs, GdnOutput};
    use crate::{DType, Result, Tensor};
    use rayon::prelude::*;

    fn to_f32(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            x.exp().ln_1p()
        }
    }

    pub(super) fn gated_delta_rule(inp: &GdnInputs, cfg: &GdnConfig) -> Result<GdnOutput> {
        let dtype = inp.mixed_qkv.dtype();
        let dev = inp.mixed_qkv.device().clone();
        let (bsz, t, c) = inp.mixed_qkv.dims3()?;
        let (hv, dk, dv, k) = (
            cfg.num_v_heads,
            cfg.head_k_dim,
            cfg.head_v_dim,
            cfg.conv_kernel,
        );
        let km1 = k - 1;
        let key_dim = cfg.key_dim();
        let x = to_f32(inp.mixed_qkv)?;
        let w = to_f32(inp.conv_weight)?;
        let bias = match inp.conv_bias {
            Some(b) => to_f32(b)?,
            None => vec![0f32; c],
        };
        let cs = match inp.conv_state {
            Some(s) => to_f32(s)?,
            None => vec![0f32; bsz * c * km1],
        };
        let av = to_f32(inp.a)?;
        let bv = to_f32(inp.b)?;
        let na = to_f32(inp.neg_a)?;
        let dtb = to_f32(inp.dt_bias)?;
        let s0 = match inp.recurrent_state {
            Some(s) => to_f32(s)?,
            None => vec![0f32; bsz * hv * dk * dv],
        };
        // Element i of [conv_state ; tokens] for channel ch of batch bi.
        let xe = |bi: usize, ch: usize, i: usize| -> f32 {
            if i < km1 {
                cs[(bi * c + ch) * km1 + i]
            } else {
                x[(bi * t + (i - km1)) * c + ch]
            }
        };
        let mut cs_out = vec![0f32; bsz * c * km1];
        for bi in 0..bsz {
            for ch in 0..c {
                for m in 0..km1 {
                    cs_out[(bi * c + ch) * km1 + m] = xe(bi, ch, t + m);
                }
            }
        }
        let scale = (1.0 / (dk as f64).sqrt()) as f32;
        let eps = cfg.l2norm_eps;
        let heads: Vec<(Vec<f32>, Vec<f32>)> = (0..bsz * hv)
            .into_par_iter()
            .map(|bh| {
                let (bi, h) = (bh / hv, bh % hv);
                let kh = cfg.k_head(h);
                let conv = |ch: usize, tt: usize| -> f32 {
                    let mut acc = bias[ch];
                    for j in 0..k {
                        acc += w[ch * k + j] * xe(bi, ch, tt + j);
                    }
                    silu(acc)
                };
                let mut s = s0[bh * dk * dv..(bh + 1) * dk * dv].to_vec();
                let mut out = vec![0f32; t * dv];
                let mut q = vec![0f32; dk];
                let mut kk = vec![0f32; dk];
                let mut v = vec![0f32; dv];
                let mut kv_mem = vec![0f32; dv];
                for tt in 0..t {
                    for i in 0..dk {
                        q[i] = conv(kh * dk + i, tt);
                        kk[i] = conv(key_dim + kh * dk + i, tt);
                    }
                    for (j, vj) in v.iter_mut().enumerate() {
                        *vj = conv(2 * key_dim + h * dv + j, tt);
                    }
                    if cfg.use_qk_l2norm {
                        for u in [&mut q, &mut kk] {
                            let ss: f32 = u.iter().map(|e| e * e).sum();
                            let inv = 1.0 / (ss + eps).sqrt();
                            u.iter_mut().for_each(|e| *e *= inv);
                        }
                    }
                    q.iter_mut().for_each(|e| *e *= scale);
                    let ab = (bi * t + tt) * hv + h;
                    let g = na[h] * softplus(av[ab] + dtb[h]);
                    let eg = g.exp();
                    let beta = sigmoid(bv[ab]);
                    s.iter_mut().for_each(|e| *e *= eg);
                    kv_mem.iter_mut().for_each(|e| *e = 0.0);
                    for (i, ki) in kk.iter().enumerate() {
                        let row = &s[i * dv..(i + 1) * dv];
                        for (m, r) in kv_mem.iter_mut().zip(row) {
                            *m += r * ki;
                        }
                    }
                    for j in 0..dv {
                        kv_mem[j] = (v[j] - kv_mem[j]) * beta;
                    }
                    let o = &mut out[tt * dv..(tt + 1) * dv];
                    for i in 0..dk {
                        let row = &mut s[i * dv..(i + 1) * dv];
                        for (r, d) in row.iter_mut().zip(kv_mem.iter()) {
                            *r += kk[i] * d;
                        }
                        for (oj, r) in o.iter_mut().zip(row.iter()) {
                            *oj += r * q[i];
                        }
                    }
                }
                (out, s)
            })
            .collect();
        let mut out = vec![0f32; bsz * t * hv * dv];
        let mut state = Vec::with_capacity(bsz * hv * dk * dv);
        for (bh, (o, s)) in heads.into_iter().enumerate() {
            let (bi, h) = (bh / hv, bh % hv);
            for tt in 0..t {
                let dst = ((bi * t + tt) * hv + h) * dv;
                out[dst..dst + dv].copy_from_slice(&o[tt * dv..(tt + 1) * dv]);
            }
            state.extend_from_slice(&s);
        }
        Ok(GdnOutput {
            out: Tensor::from_vec(out, (bsz, t, hv, dv), &dev)?.to_dtype(dtype)?,
            conv_state: Tensor::from_vec(cs_out, (bsz, c, km1), &dev)?.to_dtype(dtype)?,
            recurrent_state: Tensor::from_vec(state, (bsz, hv, dk, dv), &dev)?,
        })
    }

    pub(super) fn gated_rms_norm(x: &Tensor, z: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
        let d = x.dim(crate::D::Minus1)?;
        let xv = to_f32(x)?;
        let zv = to_f32(z)?;
        let wv = to_f32(w)?;
        let mut y = vec![0f32; xv.len()];
        if d > 0 {
            y.par_chunks_mut(d)
                .zip(xv.par_chunks(d).zip(zv.par_chunks(d)))
                .for_each(|(yr, (xr, zr))| {
                    let ms = xr.iter().map(|e| e * e).sum::<f32>() / d as f32;
                    let inv = 1.0 / (ms + eps as f32).sqrt();
                    for j in 0..d {
                        yr[j] = xr[j] * inv * wv[j] * silu(zr[j]);
                    }
                });
        }
        Tensor::from_vec(y, x.shape().clone(), x.device())?.to_dtype(x.dtype())
    }
}

/// Forward substitution of the chunked delta rule: for each `(n, n)` matrix of `attn`
/// (strictly lower triangular input), row `i` becomes `a_i + a_i[..i] * T[..i]`. Runs on the
/// host (one rayon task per matrix). Kept for the tensor-op path; callers that use
/// [`fused_gated_delta_rule`] do not need it.
pub fn chunk_tril_recurrence(attn: &Tensor) -> Result<Tensor> {
    let dims = attn.dims().to_vec();
    let r = dims.len();
    if r < 2 {
        bail!("chunk_tril_recurrence: expected (..., n, n), got {dims:?}")
    }
    let n = dims[r - 1];
    if dims[r - 2] != n {
        bail!("chunk_tril_recurrence: last two dims must be square, got {dims:?}")
    }
    let dtype = attn.dtype();
    let mut v: Vec<f32> = attn.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mat = n * n;
    if mat > 0 {
        v.par_chunks_mut(mat).for_each(|a| {
            let mut row = vec![0f32; n];
            for i in 1..n {
                row[..i].copy_from_slice(&a[i * n..i * n + i]);
                for c in 0..i {
                    let mut acc = 0f32;
                    for (j, rj) in row[..i].iter().enumerate() {
                        acc += rj * a[j * n + c];
                    }
                    a[i * n + c] = acc + row[c];
                }
            }
        });
    }
    let out = Tensor::from_vec(v, dims, attn.device())?;
    if dtype == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(dtype)
    }
}
