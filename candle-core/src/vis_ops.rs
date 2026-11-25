//! Dedicated Vision Encoder fused operations (Part 4 Optimization Roadmap):
//!
//! 1. [`vis_u8_resize_norm_patchify`]: Antialiased Bicubic / Bilinear `u8` RGB resize +
//!    `(x / 255 - mean) / std` normalization + direct patchification (zero CPU float resize).
//! 2. [`fused_res_layernorm_ls`]: Single-pass Chan-Welford Residual + optional LayerScale +
//!    LayerNorm supporting FP32 residual streams (`mean = 1e4` safe, zero FP16 > 65504 overflow).
//! 3. [`fused_bias_gelu`]: Fused optional bias + 3-mode GELU (`ExactErf`, `PytorchTanh`, `QuickGelu`).
//! 4. [`fused_vis_qkv_2d_rope`]: Fused Vision QKV unpack + optional bias / head RMSNorm +
//!    on-the-fly Cody-Waite 2D RoPE without KV cache overhead.
//! 5. [`fused_spatial_merge_norm`]: Zero-copy `M x M` spatial merger + Chan-Welford LayerNorm
//!    (`PreMergeNorm`, `PostMergeNorm`, `AvgPoolNorm`).
//! 6. [`scatter_vision_into_embeds`]: Zero-copy in-place scatter of vision embeddings into LLM embeds.
//! 7. [`patch_add_interp_pos2d_inplace`]: In-place F32 2D bilinear / bicubic learned pos-embed add.

pub use crate::accel::vis_ops::{GeluMode, SpatialMergeMode};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::op::BackpropOp;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::storage::Storage;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::tensor::from_storage;
use crate::{bail, DType, Device, Result, Tensor};

fn check_float(dtype: DType, op: &str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        d => bail!("{op}: expected F32/F16/BF16, got {d:?}"),
    }
}

#[inline]
fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

#[inline]
fn linear_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

#[inline]
fn erff_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 (max error < 1.5e-7 in FP32).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    (sign * y) as f32
}

/// Antialiased Bicubic or Bilinear resize + `(x / 255 - mean) / std` normalization + optional
/// patchification directly from a `u8` RGB image tensor (`[H_in, W_in, 3]` when `chw_in == false`,
/// or `[3, H_in, W_in]` when `chw_in == true`).
///
/// - When `patch_size > 1`, requires `h_out % patch_size == 0 && w_out % patch_size == 0`,
///   and returns `[n_patches, 3 * patch_size * patch_size]` in `out_dtype`.
/// - When `patch_size <= 1`, returns `[3, h_out, w_out]` in `out_dtype`.
#[allow(clippy::too_many_arguments)]
pub fn vis_u8_resize_norm_patchify(
    img_u8: &Tensor,
    h_out: usize,
    w_out: usize,
    patch_size: usize,
    bicubic: bool,
    chw_in: bool,
    mean: [f32; 3],
    std: [f32; 3],
    out_dtype: DType,
) -> Result<Tensor> {
    if img_u8.dtype() != DType::U8 {
        bail!(
            "vis_u8_resize_norm_patchify: expected U8 input, got {:?}",
            img_u8.dtype()
        )
    }
    check_float(out_dtype, "vis_u8_resize_norm_patchify")?;
    let (h_in, w_in) = if chw_in {
        let (c, h, w) = img_u8.dims3()?;
        if c != 3 {
            bail!("vis_u8_resize_norm_patchify: expected 3 channels, got {c}")
        }
        (h, w)
    } else {
        let (h, w, c) = img_u8.dims3()?;
        if c != 3 {
            bail!("vis_u8_resize_norm_patchify: expected 3 channels, got {c}")
        }
        (h, w)
    };
    if h_in == 0 || w_in == 0 || h_out == 0 || w_out == 0 {
        bail!("vis_u8_resize_norm_patchify: empty image dimensions")
    }
    if patch_size > 1 && (!h_out.is_multiple_of(patch_size) || !w_out.is_multiple_of(patch_size)) {
        bail!("vis_u8_resize_norm_patchify: ({h_out}, {w_out}) not divisible by patch_size {patch_size}")
    }
    let inv_std = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];
    let out_shape: Vec<usize> = if patch_size <= 1 {
        vec![3, h_out, w_out]
    } else {
        vec![
            (h_out / patch_size) * (w_out / patch_size),
            3 * patch_size * patch_size,
        ]
    };
    let img_c = img_u8.contiguous()?;
    let dev = img_c.device().clone();
    match &dev {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (s, _) = img_c.storage_and_layout();
            let Storage::Rocm(rs) = &*s else { unreachable!() };
            let out = rs.vis_u8_resize_norm_patchify(
                out_dtype, h_in, w_in, h_out, w_out, patch_size, bicubic, chw_in, mean, inv_std,
            )?;
            Ok(from_storage(
                Storage::Rocm(out),
                out_shape,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (s, _) = img_c.storage_and_layout();
            let Storage::Cuda(cs) = &*s else { unreachable!() };
            let out = cs.vis_u8_resize_norm_patchify(
                out_dtype, h_in, w_in, h_out, w_out, patch_size, bicubic, chw_in, mean, inv_std,
            )?;
            Ok(from_storage(
                Storage::Cuda(out),
                out_shape,
                BackpropOp::none(),
                false,
            ))
        }
        _ => {
            let src = img_c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
            let total = h_out * w_out;
            let mut dst = vec![0.0f32; total * 3];
            let sy = h_in as f32 / h_out as f32;
            let sx = w_in as f32 / w_out as f32;
            let r_base = if bicubic { 2.0f32 } else { 1.0f32 };
            let fy = sy.max(1.0);
            let fx = sx.max(1.0);
            let ry = r_base * fy;
            let rx = r_base * fx;
            let inv_fy = 1.0 / fy;
            let inv_fx = 1.0 / fx;
            let hw_in = h_in * w_in;

            for oy in 0..h_out {
                let cy = (oy as f32 + 0.5) * sy - 0.5;
                let y0 = (cy - ry).floor().max(0.0) as usize;
                let y1 = (cy + ry).ceil().min((h_in - 1) as f32).max(0.0) as usize;
                for ox in 0..w_out {
                    let cx = (ox as f32 + 0.5) * sx - 0.5;
                    let x0 = (cx - rx).floor().max(0.0) as usize;
                    let x1 = (cx + rx).ceil().min((w_in - 1) as f32).max(0.0) as usize;
                    let (mut a0, mut a1, mut a2, mut ws) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for iy in y0..=y1 {
                        let dy = (iy as f32 - cy) * inv_fy;
                        let wy = if bicubic { cubic_weight(dy) } else { linear_weight(dy) };
                        if wy == 0.0 {
                            continue;
                        }
                        for ix in x0..=x1 {
                            let dx = (ix as f32 - cx) * inv_fx;
                            let wx = if bicubic { cubic_weight(dx) } else { linear_weight(dx) };
                            let w = wy * wx;
                            if w == 0.0 {
                                continue;
                            }
                            let (p0, p1, p2) = if chw_in {
                                let pix = iy * w_in + ix;
                                (
                                    src[pix] as f32,
                                    src[hw_in + pix] as f32,
                                    src[2 * hw_in + pix] as f32,
                                )
                            } else {
                                let pix = (iy * w_in + ix) * 3;
                                (src[pix] as f32, src[pix + 1] as f32, src[pix + 2] as f32)
                            };
                            a0 = w.mul_add(p0, a0);
                            a1 = w.mul_add(p1, a1);
                            a2 = w.mul_add(p2, a2);
                            ws += w;
                        }
                    }
                    let inv_w = if ws > 0.0 { 1.0 / (ws * 255.0) } else { 0.0 };
                    let v0 = (a0 * inv_w - mean[0]) * inv_std[0];
                    let v1 = (a1 * inv_w - mean[1]) * inv_std[1];
                    let v2 = (a2 * inv_w - mean[2]) * inv_std[2];
                    if patch_size <= 1 {
                        let idx = oy * w_out + ox;
                        dst[idx] = v0;
                        dst[total + idx] = v1;
                        dst[2 * total + idx] = v2;
                    } else {
                        let pw = w_out / patch_size;
                        let p_idx = (oy / patch_size) * pw + (ox / patch_size);
                        let p_area = patch_size * patch_size;
                        let base = p_idx * 3 * p_area + (oy % patch_size) * patch_size + (ox % patch_size);
                        dst[base] = v0;
                        dst[base + p_area] = v1;
                        dst[base + 2 * p_area] = v2;
                    }
                }
            }
            Tensor::from_vec(dst, out_shape, &Device::Cpu)?
                .to_dtype(out_dtype)?
                .to_device(&dev)
        }
    }
}

/// Single-pass Chan-Welford Residual + optional LayerScale (`gamma`) + LayerNorm.
///
/// Supports `res_in` in `F32` with `delta` / `norm_out` in `F16` / `BF16` (mixed-precision
/// FP32 residual stream preventing FP16 > 65504 overflow in deep ViTs) as well as matching dtypes.
/// Returns `(updated_residual, normed)`.
#[allow(clippy::too_many_arguments)]
pub fn fused_res_layernorm_ls(
    res_in: &Tensor,
    delta: Option<&Tensor>,
    ls_gamma: Option<&Tensor>,
    ln_weight: Option<&Tensor>,
    ln_bias: Option<&Tensor>,
    eps: f32,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    check_float(res_in.dtype(), "fused_res_layernorm_ls")?;
    check_float(out_dtype, "fused_res_layernorm_ls")?;
    let dims = res_in.dims().to_vec();
    if dims.is_empty() {
        bail!("fused_res_layernorm_ls: scalar input not supported")
    }
    let d = *dims.last().unwrap();
    let rows = res_in.elem_count() / d.max(1);
    let dev = res_in.device().clone();
    let res_c = res_in.contiguous()?;
    let delta_c = match delta {
        Some(t) => {
            if t.dims() != dims.as_slice() {
                bail!(
                    "fused_res_layernorm_ls: delta {:?} != res_in {dims:?}",
                    t.dims()
                )
            }
            Some(t.to_dtype(out_dtype)?.contiguous()?)
        }
        None => None,
    };
    let ls_f32 = match ls_gamma {
        Some(g) => Some(g.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };
    let lnw_f32 = match ln_weight {
        Some(w) => Some(w.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };
    let lnb_f32 = match ln_bias {
        Some(b) => Some(b.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };

    let res_compatible = res_c.dtype() == out_dtype || res_c.dtype() == DType::F32;
    if res_compatible {
        match &dev {
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                let (rs, _) = res_c.storage_and_layout();
                let dg = delta_c.as_ref().map(|t| t.storage_and_layout());
                let lg = ls_f32.as_ref().map(|t| t.storage_and_layout());
                let wg = lnw_f32.as_ref().map(|t| t.storage_and_layout());
                let bg = lnb_f32.as_ref().map(|t| t.storage_and_layout());
                let Storage::Rocm(rr) = &*rs else { unreachable!() };
                let dr = dg.as_ref().map(|(s, _)| match &**s {
                    Storage::Rocm(x) => x,
                    _ => unreachable!(),
                });
                let lr = lg.as_ref().map(|(s, _)| match &**s {
                    Storage::Rocm(x) => x,
                    _ => unreachable!(),
                });
                let wr = wg.as_ref().map(|(s, _)| match &**s {
                    Storage::Rocm(x) => x,
                    _ => unreachable!(),
                });
                let br = bg.as_ref().map(|(s, _)| match &**s {
                    Storage::Rocm(x) => x,
                    _ => unreachable!(),
                });
                let (rout, nout) = rr.vis_res_layernorm_ls(
                    dr, lr, wr, br, out_dtype, rows, d, eps, true,
                )?;
                return Ok((
                    from_storage(
                        Storage::Rocm(rout.unwrap()),
                        dims.clone(),
                        BackpropOp::none(),
                        false,
                    ),
                    from_storage(Storage::Rocm(nout), dims, BackpropOp::none(), false),
                ));
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let (rs, _) = res_c.storage_and_layout();
                let dg = delta_c.as_ref().map(|t| t.storage_and_layout());
                let lg = ls_f32.as_ref().map(|t| t.storage_and_layout());
                let wg = lnw_f32.as_ref().map(|t| t.storage_and_layout());
                let bg = lnb_f32.as_ref().map(|t| t.storage_and_layout());
                let Storage::Cuda(rc) = &*rs else { unreachable!() };
                let dc = dg.as_ref().map(|(s, _)| match &**s {
                    Storage::Cuda(x) => x,
                    _ => unreachable!(),
                });
                let lc = lg.as_ref().map(|(s, _)| match &**s {
                    Storage::Cuda(x) => x,
                    _ => unreachable!(),
                });
                let wc = wg.as_ref().map(|(s, _)| match &**s {
                    Storage::Cuda(x) => x,
                    _ => unreachable!(),
                });
                let bc = bg.as_ref().map(|(s, _)| match &**s {
                    Storage::Cuda(x) => x,
                    _ => unreachable!(),
                });
                let (rout, nout) = rc.vis_res_layernorm_ls(
                    dc, lc, wc, bc, out_dtype, rows, d, eps, true,
                )?;
                return Ok((
                    from_storage(
                        Storage::Cuda(rout.unwrap()),
                        dims.clone(),
                        BackpropOp::none(),
                        false,
                    ),
                    from_storage(Storage::Cuda(nout), dims, BackpropOp::none(), false),
                ));
            }
            _ => {}
        }
    }

    // Exact Chan-Welford CPU fallback (also handles Vulkan/Metal cleanly with zero cancellation).
    let rv = res_c
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let dv = match &delta_c {
        Some(t) => Some(
            t.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        ),
        None => None,
    };
    let ls = match &ls_f32 {
        Some(t) => Some(t.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
        None => None,
    };
    let lnw = match &lnw_f32 {
        Some(t) => Some(t.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
        None => None,
    };
    let lnb = match &lnb_f32 {
        Some(t) => Some(t.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
        None => None,
    };
    let mut r_out = vec![0.0f32; rows * d];
    let mut n_out = vec![0.0f32; rows * d];
    for r in 0..rows {
        let base = r * d;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for i in 0..d {
            let mut v = rv[base + i];
            if let Some(dvec) = &dv {
                let mut di = dvec[base + i];
                if let Some(g) = &ls {
                    di *= g[i];
                }
                v += di;
            }
            r_out[base + i] = v;
            let vf = v as f64;
            let cnt = (i + 1) as f64;
            let diff = vf - mean;
            mean += diff / cnt;
            m2 += diff * (vf - mean);
        }
        let inv_std = (1.0 / (m2 / d as f64 + eps as f64).sqrt()) as f32;
        let mean_f = mean as f32;
        for i in 0..d {
            let mut y = (r_out[base + i] - mean_f) * inv_std;
            if let Some(w) = &lnw {
                y *= w[i];
            }
            if let Some(b) = &lnb {
                y += b[i];
            }
            n_out[base + i] = y;
        }
    }
    let res_t = Tensor::from_vec(r_out, dims.as_slice(), &Device::Cpu)?
        .to_dtype(res_in.dtype())?
        .to_device(&dev)?;
    let norm_t = Tensor::from_vec(n_out, dims.as_slice(), &Device::Cpu)?
        .to_dtype(out_dtype)?
        .to_device(&dev)?;
    Ok((res_t, norm_t))
}

/// Fused optional bias + 3-mode GELU (`ExactErf`, `PytorchTanh`, `QuickGelu`).
pub fn fused_bias_gelu(x: &Tensor, bias: Option<&Tensor>, mode: GeluMode) -> Result<Tensor> {
    check_float(x.dtype(), "fused_bias_gelu")?;
    let dims = x.dims().to_vec();
    if dims.is_empty() {
        bail!("fused_bias_gelu: scalar input not supported")
    }
    let dim = *dims.last().unwrap();
    let n_elems = x.elem_count();
    let dtype = x.dtype();
    let dev = x.device().clone();
    let x_c = x.contiguous()?;
    let b_c = match bias {
        Some(b) => {
            if b.elem_count() != dim {
                bail!("fused_bias_gelu: bias len {} != dim {dim}", b.elem_count())
            }
            Some(b.to_dtype(dtype)?.flatten_all()?.contiguous()?)
        }
        None => None,
    };
    match &dev {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (xs, _) = x_c.storage_and_layout();
            let bg = b_c.as_ref().map(|t| t.storage_and_layout());
            let Storage::Rocm(xr) = &*xs else { unreachable!() };
            let br = bg.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(r) => r,
                _ => unreachable!(),
            });
            let out = xr.vis_bias_gelu(br, n_elems, dim, mode as u32)?;
            Ok(from_storage(
                Storage::Rocm(out),
                dims,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (xs, _) = x_c.storage_and_layout();
            let bg = b_c.as_ref().map(|t| t.storage_and_layout());
            let Storage::Cuda(xc) = &*xs else { unreachable!() };
            let bc = bg.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let out = xc.vis_bias_gelu(bc, n_elems, dim, mode as u32)?;
            Ok(from_storage(
                Storage::Cuda(out),
                dims,
                BackpropOp::none(),
                false,
            ))
        }
        _ => {
            let xv = x_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let bv = match &b_c {
                Some(b) => Some(
                    b.to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                ),
                None => None,
            };
            let mut y = vec![0.0f32; n_elems];
            for i in 0..n_elems {
                let mut v = xv[i];
                if let Some(b) = &bv {
                    v += b[i % dim];
                }
                y[i] = match mode {
                    GeluMode::ExactErf => 0.5 * v * (1.0 + erff_approx(v * 0.7071067811865475)),
                    GeluMode::PytorchTanh => {
                        let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
                        0.5 * v * (1.0 + inner.tanh())
                    }
                    GeluMode::QuickGelu => v / (1.0 + (-1.702 * v).exp()),
                };
            }
            Tensor::from_vec(y, dims, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}

/// Fused Vision QKV unpack (`[N, 3 * H * D]`) + optional bias + optional Q/K head RMSNorm +
/// on-the-fly Cody-Waite 2D RoPE from `patch_yx: [N, 2]` (`u32`) and `inv_freq: [rot_dim / 4]`.
/// Returns `(Q, K, V)` each of shape `[H, N, D]` (no KV cache).
#[allow(clippy::too_many_arguments)]
pub fn fused_vis_qkv_2d_rope(
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_norm_w: Option<&Tensor>,
    k_norm_w: Option<&Tensor>,
    patch_yx: &Tensor,
    inv_freq: &Tensor,
    num_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    interleaved: bool,
    norm_plus_one: bool,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    check_float(qkv.dtype(), "fused_vis_qkv_2d_rope")?;
    let (n_tokens, c3) = qkv.dims2()?;
    if c3 != 3 * num_heads * head_dim {
        bail!(
            "fused_vis_qkv_2d_rope: qkv dim {c3} != 3 * {num_heads} * {head_dim}"
        )
    }
    if rot_dim > head_dim || rot_dim % 4 != 0 {
        bail!("fused_vis_qkv_2d_rope: invalid rot_dim {rot_dim} for head_dim {head_dim}")
    }
    if patch_yx.dims() != [n_tokens, 2] {
        bail!("fused_vis_qkv_2d_rope: patch_yx {:?} != [{n_tokens}, 2]", patch_yx.dims())
    }
    if inv_freq.elem_count() != rot_dim / 4 {
        bail!(
            "fused_vis_qkv_2d_rope: inv_freq len {} != {}",
            inv_freq.elem_count(),
            rot_dim / 4
        )
    }
    let dtype = qkv.dtype();
    let dev = qkv.device().clone();
    let qkv_c = qkv.contiguous()?;
    let b_c = match bias {
        Some(b) => Some(b.to_dtype(dtype)?.flatten_all()?.contiguous()?),
        None => None,
    };
    let qnw = match q_norm_w {
        Some(w) => Some(w.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };
    let knw = match k_norm_w {
        Some(w) => Some(w.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };
    let yx_c = patch_yx.to_dtype(DType::U32)?.contiguous()?;
    let if_c = inv_freq.to_dtype(DType::F32)?.flatten_all()?.contiguous()?;
    #[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(unused_variables))]
    let flags = u32::from(interleaved) | (u32::from(norm_plus_one) << 1);
    let out_shape = (num_heads, n_tokens, head_dim);

    match &dev {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (s0, _) = qkv_c.storage_and_layout();
            let sb = b_c.as_ref().map(|t| t.storage_and_layout());
            let sq = qnw.as_ref().map(|t| t.storage_and_layout());
            let sk = knw.as_ref().map(|t| t.storage_and_layout());
            let (syx, _) = yx_c.storage_and_layout();
            let (sif, _) = if_c.storage_and_layout();
            let (Storage::Rocm(r0), Storage::Rocm(ryx), Storage::Rocm(rif)) = (&*s0, &*syx, &*sif)
            else {
                unreachable!()
            };
            let rb = sb.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(x) => x,
                _ => unreachable!(),
            });
            let rq = sq.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(x) => x,
                _ => unreachable!(),
            });
            let rk = sk.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(x) => x,
                _ => unreachable!(),
            });
            let (qo, ko, vo) = r0.vis_qkv_2d_rope(
                rb, rq, rk, ryx, rif, n_tokens, num_heads, head_dim, rot_dim, flags, eps,
            )?;
            Ok((
                from_storage(Storage::Rocm(qo), out_shape, BackpropOp::none(), false),
                from_storage(Storage::Rocm(ko), out_shape, BackpropOp::none(), false),
                from_storage(Storage::Rocm(vo), out_shape, BackpropOp::none(), false),
            ))
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (s0, _) = qkv_c.storage_and_layout();
            let sb = b_c.as_ref().map(|t| t.storage_and_layout());
            let sq = qnw.as_ref().map(|t| t.storage_and_layout());
            let sk = knw.as_ref().map(|t| t.storage_and_layout());
            let (syx, _) = yx_c.storage_and_layout();
            let (sif, _) = if_c.storage_and_layout();
            let (Storage::Cuda(c0), Storage::Cuda(cyx), Storage::Cuda(cif)) = (&*s0, &*syx, &*sif)
            else {
                unreachable!()
            };
            let cb = sb.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(x) => x,
                _ => unreachable!(),
            });
            let cq = sq.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(x) => x,
                _ => unreachable!(),
            });
            let ck = sk.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(x) => x,
                _ => unreachable!(),
            });
            let (qo, ko, vo) = c0.vis_qkv_2d_rope(
                cb, cq, ck, cyx, cif, n_tokens, num_heads, head_dim, rot_dim, flags, eps,
            )?;
            Ok((
                from_storage(Storage::Cuda(qo), out_shape, BackpropOp::none(), false),
                from_storage(Storage::Cuda(ko), out_shape, BackpropOp::none(), false),
                from_storage(Storage::Cuda(vo), out_shape, BackpropOp::none(), false),
            ))
        }
        _ => {
            let qkv_v = qkv_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let b_v = match &b_c {
                Some(b) => Some(
                    b.to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                ),
                None => None,
            };
            let qnw_v = match &qnw {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let knw_v = match &knw {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let yx_v = yx_c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
            let if_v = if_c.to_device(&Device::Cpu)?.to_vec1::<f32>()?;

            let hd_all = num_heads * head_dim;
            let mut q_out = vec![0.0f32; num_heads * n_tokens * head_dim];
            let mut k_out = vec![0.0f32; num_heads * n_tokens * head_dim];
            let mut v_out = vec![0.0f32; num_heads * n_tokens * head_dim];
            let mut qb = vec![0.0f32; head_dim];
            let mut kb = vec![0.0f32; head_dim];
            let half_rot = rot_dim / 2;
            let quarter = rot_dim / 4;

            for tok in 0..n_tokens {
                let py = yx_v[tok * 2];
                let px = yx_v[tok * 2 + 1];
                let base = tok * 3 * hd_all;
                for h in 0..num_heads {
                    let q_off = h * head_dim;
                    let k_off = hd_all + q_off;
                    let v_off = 2 * hd_all + q_off;
                    let dst = (h * n_tokens + tok) * head_dim;
                    for i in 0..head_dim {
                        let mut qv = qkv_v[base + q_off + i];
                        let mut kv = qkv_v[base + k_off + i];
                        let mut vv = qkv_v[base + v_off + i];
                        if let Some(bv) = &b_v {
                            qv += bv[q_off + i];
                            kv += bv[k_off + i];
                            vv += bv[v_off + i];
                        }
                        qb[i] = qv;
                        kb[i] = kv;
                        v_out[dst + i] = vv;
                    }
                    if let Some(w) = &qnw_v {
                        let sq: f32 = qb.iter().map(|v| v * v).sum();
                        let inv = 1.0 / (sq / head_dim as f32 + eps).sqrt();
                        for i in 0..head_dim {
                            let wi = if norm_plus_one { 1.0 + w[i] } else { w[i] };
                            qb[i] = (qb[i] * inv) * wi;
                        }
                    }
                    if let Some(w) = &knw_v {
                        let sq: f32 = kb.iter().map(|v| v * v).sum();
                        let inv = 1.0 / (sq / head_dim as f32 + eps).sqrt();
                        for i in 0..head_dim {
                            let wi = if norm_plus_one { 1.0 + w[i] } else { w[i] };
                            kb[i] = (kb[i] * inv) * wi;
                        }
                    }
                    for pair in 0..half_rot {
                        let is_x = pair >= quarter;
                        let f_idx = if is_x { pair - quarter } else { pair };
                        let pos = if is_x { px } else { py };
                        let (c, s) =
                            crate::accel::decoder_ops::cody_waite_sincos_f32(pos, if_v[f_idx]);
                        let (i0, i1) = if interleaved {
                            (2 * pair, 2 * pair + 1)
                        } else {
                            (pair, pair + half_rot)
                        };
                        q_out[dst + i0] = qb[i0] * c - qb[i1] * s;
                        q_out[dst + i1] = qb[i1] * c + qb[i0] * s;
                        k_out[dst + i0] = kb[i0] * c - kb[i1] * s;
                        k_out[dst + i1] = kb[i1] * c + kb[i0] * s;
                    }
                    for i in rot_dim..head_dim {
                        q_out[dst + i] = qb[i];
                        k_out[dst + i] = kb[i];
                    }
                }
            }
            Ok((
                Tensor::from_vec(q_out, out_shape, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&dev)?,
                Tensor::from_vec(k_out, out_shape, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&dev)?,
                Tensor::from_vec(v_out, out_shape, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&dev)?,
            ))
        }
    }
}

/// Zero-copy `merge x merge` spatial merge + single-pass Chan-Welford LayerNorm.
/// Input `x`: `[h_p * w_p, d]`, output: `[(h_p / merge) * (w_p / merge), out_d]`.
#[allow(clippy::too_many_arguments)]
pub fn fused_spatial_merge_norm(
    x: &Tensor,
    ln_weight: Option<&Tensor>,
    ln_bias: Option<&Tensor>,
    h_p: usize,
    w_p: usize,
    merge: usize,
    mode: SpatialMergeMode,
    eps: f32,
) -> Result<Tensor> {
    check_float(x.dtype(), "fused_spatial_merge_norm")?;
    let (n_tok, d) = x.dims2()?;
    if merge == 0 || !h_p.is_multiple_of(merge) || !w_p.is_multiple_of(merge) || n_tok != h_p * w_p
    {
        bail!(
            "fused_spatial_merge_norm: invalid grid ({h_p}, {w_p}) / merge {merge} for {n_tok} tokens"
        )
    }
    let dtype = x.dtype();
    let dev = x.device().clone();
    let out_toks = (h_p / merge) * (w_p / merge);
    let out_dim = match mode {
        SpatialMergeMode::AvgPoolNorm => d,
        _ => merge * merge * d,
    };
    let x_c = x.contiguous()?;
    let lnw = match ln_weight {
        Some(w) => Some(w.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };
    let lnb = match ln_bias {
        Some(b) => Some(b.to_dtype(DType::F32)?.flatten_all()?.contiguous()?),
        None => None,
    };

    match &dev {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (xs, _) = x_c.storage_and_layout();
            let sw = lnw.as_ref().map(|t| t.storage_and_layout());
            let sb = lnb.as_ref().map(|t| t.storage_and_layout());
            let Storage::Rocm(xr) = &*xs else { unreachable!() };
            let wr = sw.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(x) => x,
                _ => unreachable!(),
            });
            let br = sb.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(x) => x,
                _ => unreachable!(),
            });
            let out = xr.vis_spatial_merge_norm(wr, br, h_p, w_p, d, merge, mode as u32, eps)?;
            Ok(from_storage(
                Storage::Rocm(out),
                (out_toks, out_dim),
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (xs, _) = x_c.storage_and_layout();
            let sw = lnw.as_ref().map(|t| t.storage_and_layout());
            let sb = lnb.as_ref().map(|t| t.storage_and_layout());
            let Storage::Cuda(xc) = &*xs else { unreachable!() };
            let wc = sw.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(x) => x,
                _ => unreachable!(),
            });
            let bc = sb.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(x) => x,
                _ => unreachable!(),
            });
            let out = xc.vis_spatial_merge_norm(wc, bc, h_p, w_p, d, merge, mode as u32, eps)?;
            Ok(from_storage(
                Storage::Cuda(out),
                (out_toks, out_dim),
                BackpropOp::none(),
                false,
            ))
        }
        _ => {
            let xv = x_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let wv = match &lnw {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let bv = match &lnb {
                Some(b) => Some(b.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let mw = w_p / merge;
            let m2 = merge * merge;
            let mut y = vec![0.0f32; out_toks * out_dim];

            let apply_ln = |buf: &mut [f32], w: Option<&[f32]>, b: Option<&[f32]>| {
                let n = buf.len() as f64;
                let mut mean = 0.0f64;
                let mut m2v = 0.0f64;
                for (i, &v) in buf.iter().enumerate() {
                    let vf = v as f64;
                    let diff = vf - mean;
                    mean += diff / (i + 1) as f64;
                    m2v += diff * (vf - mean);
                }
                let inv = (1.0 / (m2v / n + eps as f64).sqrt()) as f32;
                let mf = mean as f32;
                for (i, v) in buf.iter_mut().enumerate() {
                    let mut r = (*v - mf) * inv;
                    if let Some(ww) = w {
                        r *= ww[i];
                    }
                    if let Some(bb) = b {
                        r += bb[i];
                    }
                    *v = r;
                }
            };

            for out_tok in 0..out_toks {
                let oy = out_tok / mw;
                let ox = out_tok % mw;
                let dst = &mut y[out_tok * out_dim..(out_tok + 1) * out_dim];
                match mode {
                    SpatialMergeMode::PreMergeNorm => {
                        for sub in 0..m2 {
                            let in_tok = (oy * merge + sub / merge) * w_p + (ox * merge + sub % merge);
                            let sl = &mut dst[sub * d..(sub + 1) * d];
                            sl.copy_from_slice(&xv[in_tok * d..(in_tok + 1) * d]);
                            apply_ln(sl, wv.as_deref(), bv.as_deref());
                        }
                    }
                    SpatialMergeMode::PostMergeNorm => {
                        for sub in 0..m2 {
                            let in_tok = (oy * merge + sub / merge) * w_p + (ox * merge + sub % merge);
                            dst[sub * d..(sub + 1) * d]
                                .copy_from_slice(&xv[in_tok * d..(in_tok + 1) * d]);
                        }
                        apply_ln(dst, wv.as_deref(), bv.as_deref());
                    }
                    SpatialMergeMode::AvgPoolNorm => {
                        let inv_m2 = 1.0 / m2 as f32;
                        for sub in 0..m2 {
                            let in_tok = (oy * merge + sub / merge) * w_p + (ox * merge + sub % merge);
                            for c in 0..d {
                                dst[c] += xv[in_tok * d + c] * inv_m2;
                            }
                        }
                        apply_ln(dst, wv.as_deref(), bv.as_deref());
                    }
                }
            }
            Tensor::from_vec(y, (out_toks, out_dim), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}

/// Scatters vision embeddings `[N_vis, D]` into LLM embeddings `[L_total, D]` at `indices: [N_vis]` (`u32`).
/// Mutates `embeds` in-place on GPU when uniquely owned & contiguous, otherwise clones first.
pub fn scatter_vision_into_embeds(
    embeds: Tensor,
    vis: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    check_float(embeds.dtype(), "scatter_vision_into_embeds")?;
    let (l_total, d) = embeds.dims2()?;
    let (n_vis, d_v) = vis.dims2()?;
    if d != d_v || indices.elem_count() != n_vis {
        bail!(
            "scatter_vision_into_embeds: shape mismatch embeds={:?}, vis={:?}, indices={:?}",
            embeds.dims(),
            vis.dims(),
            indices.dims()
        )
    }
    let dtype = embeds.dtype();
    let dev = embeds.device().clone();
    let vis_c = vis.to_dtype(dtype)?.contiguous()?;
    let idx_c = indices.to_dtype(DType::U32)?.flatten_all()?.contiguous()?;
    let out = if embeds.is_sole_owner_contiguous() {
        embeds
    } else {
        embeds.contiguous()?.copy()?
    };

    match &dev {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let mut es = out.storage_mut();
            let (vs, _) = vis_c.storage_and_layout();
            let (is, _) = idx_c.storage_and_layout();
            let (Storage::Rocm(er), Storage::Rocm(vr), Storage::Rocm(ir)) = (&mut *es, &*vs, &*is)
            else {
                unreachable!()
            };
            er.vis_scatter_embeds_inplace(vr, ir, n_vis, d, l_total)?;
            drop(es);
            Ok(out)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let mut es = out.storage_mut();
            let (vs, _) = vis_c.storage_and_layout();
            let (is, _) = idx_c.storage_and_layout();
            let (Storage::Cuda(ec), Storage::Cuda(vc), Storage::Cuda(ic)) = (&mut *es, &*vs, &*is)
            else {
                unreachable!()
            };
            ec.vis_scatter_embeds_inplace(vc, ic, n_vis, d, l_total)?;
            drop(es);
            Ok(out)
        }
        _ => {
            let mut ev = out
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let vv = vis_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let iv = idx_c.to_device(&Device::Cpu)?.to_vec1::<u32>()?;
            for (v_idx, &dst_tok) in iv.iter().enumerate() {
                let dst = dst_tok as usize;
                if dst < l_total {
                    ev[dst * d..(dst + 1) * d]
                        .copy_from_slice(&vv[v_idx * d..(v_idx + 1) * d]);
                }
            }
            Tensor::from_vec(ev, (l_total, d), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}

/// Fused in-place F32 bilinear / bicubic 2D learned position-embedding interpolation added to
/// `patches: [h_p * w_p, d]` from `pos_table: [g_h, g_w, d]`.
pub fn patch_add_interp_pos2d_inplace(
    patches: Tensor,
    pos_table: &Tensor,
    h_p: usize,
    w_p: usize,
    bicubic: bool,
    align_corners: bool,
) -> Result<Tensor> {
    check_float(patches.dtype(), "patch_add_interp_pos2d_inplace")?;
    let (n_tok, d) = patches.dims2()?;
    let (g_h, g_w, d_t) = pos_table.dims3()?;
    if n_tok != h_p * w_p || d != d_t {
        bail!(
            "patch_add_interp_pos2d_inplace: patches {:?} vs grid ({h_p}, {w_p}) / pos_table {:?}",
            patches.dims(),
            pos_table.dims()
        )
    }
    let dtype = patches.dtype();
    let dev = patches.device().clone();
    let tbl_c = pos_table.to_dtype(dtype)?.contiguous()?;
    let out = if patches.is_sole_owner_contiguous() {
        patches
    } else {
        patches.contiguous()?.copy()?
    };

    match &dev {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let mut ps = out.storage_mut();
            let (ts, _) = tbl_c.storage_and_layout();
            let (Storage::Rocm(pr), Storage::Rocm(tr)) = (&mut *ps, &*ts) else {
                unreachable!()
            };
            pr.vis_patch_add_interp_pos2d_inplace(
                tr,
                g_h,
                g_w,
                h_p,
                w_p,
                d,
                bicubic,
                align_corners,
            )?;
            drop(ps);
            Ok(out)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let mut ps = out.storage_mut();
            let (ts, _) = tbl_c.storage_and_layout();
            let (Storage::Cuda(pc), Storage::Cuda(tc)) = (&mut *ps, &*ts) else {
                unreachable!()
            };
            pc.vis_patch_add_interp_pos2d_inplace(
                tc,
                g_h,
                g_w,
                h_p,
                w_p,
                d,
                bicubic,
                align_corners,
            )?;
            drop(ps);
            Ok(out)
        }
        _ => {
            let mut pv = out
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tv = tbl_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for py in 0..h_p {
                let fy = if align_corners && h_p > 1 {
                    py as f32 * (g_h - 1) as f32 / (h_p - 1) as f32
                } else {
                    (py as f32 + 0.5) * g_h as f32 / h_p as f32 - 0.5
                };
                for px in 0..w_p {
                    let fx = if align_corners && w_p > 1 {
                        px as f32 * (g_w - 1) as f32 / (w_p - 1) as f32
                    } else {
                        (px as f32 + 0.5) * g_w as f32 / w_p as f32 - 0.5
                    };
                    let dst = (py * w_p + px) * d;
                    if !bicubic {
                        let y0 = fy.floor() as isize;
                        let x0 = fx.floor() as isize;
                        let wy1 = fy - y0 as f32;
                        let wx1 = fx - x0 as f32;
                        let wy0 = 1.0 - wy1;
                        let wx0 = 1.0 - wx1;
                        let cy0 = y0.clamp(0, g_h as isize - 1) as usize;
                        let cy1 = (y0 + 1).clamp(0, g_h as isize - 1) as usize;
                        let cx0 = x0.clamp(0, g_w as isize - 1) as usize;
                        let cx1 = (x0 + 1).clamp(0, g_w as isize - 1) as usize;
                        let p00 = (cy0 * g_w + cx0) * d;
                        let p01 = (cy0 * g_w + cx1) * d;
                        let p10 = (cy1 * g_w + cx0) * d;
                        let p11 = (cy1 * g_w + cx1) * d;
                        for c in 0..d {
                            pv[dst + c] += wy0 * wx0 * tv[p00 + c]
                                + wy0 * wx1 * tv[p01 + c]
                                + wy1 * wx0 * tv[p10 + c]
                                + wy1 * wx1 * tv[p11 + c];
                        }
                    } else {
                        let y0 = fy.floor() as isize;
                        let x0 = fx.floor() as isize;
                        let mut wy = [0.0f32; 4];
                        let mut wx = [0.0f32; 4];
                        let mut cy = [0usize; 4];
                        let mut cx = [0usize; 4];
                        let (mut sy, mut sx) = (0.0f32, 0.0f32);
                        for k in 0..4isize {
                            let iy = y0 - 1 + k;
                            let ix = x0 - 1 + k;
                            wy[k as usize] = cubic_weight(iy as f32 - fy);
                            wx[k as usize] = cubic_weight(ix as f32 - fx);
                            sy += wy[k as usize];
                            sx += wx[k as usize];
                            cy[k as usize] = iy.clamp(0, g_h as isize - 1) as usize;
                            cx[k as usize] = ix.clamp(0, g_w as isize - 1) as usize;
                        }
                        let inv_w = if sy * sx != 0.0 { 1.0 / (sy * sx) } else { 1.0 };
                        for c in 0..d {
                            let mut acc = 0.0f32;
                            for ky in 0..4 {
                                for kx in 0..4 {
                                    acc += wy[ky] * wx[kx] * tv[(cy[ky] * g_w + cx[kx]) * d + c];
                                }
                            }
                            pv[dst + c] += acc * inv_w;
                        }
                    }
                }
            }
            Tensor::from_vec(pv, (n_tok, d), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vis_res_layernorm_ls_chan_welford_no_cancellation_at_large_mean() -> Result<()> {
        let dev = Device::Cpu;
        // Mean = 1e4 with tiny variance (~1.0) destroys naive E[x^2] - E[x]^2 in FP32.
        let d = 64usize;
        let base: Vec<f32> = (0..d)
            .map(|i| 10_000.0 + ((i % 4) as f32 - 1.5))
            .collect();
        let x = Tensor::from_vec(base.clone(), (1, d), &dev)?;
        let (res, norm) = fused_res_layernorm_ls(&x, None, None, None, None, 1e-5, DType::F32)?;
        let nv = norm.flatten_all()?.to_vec1::<f32>()?;
        let rv = res.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(rv, base);
        let mean: f64 = nv.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var: f64 = nv.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        assert!(mean.abs() < 1e-5, "mean drifted: {mean}");
        assert!((var - 1.0).abs() < 1e-3, "variance cancelled: {var}");
        Ok(())
    }

    #[test]
    fn vis_ops_end_to_end_cpu_smoke() -> Result<()> {
        let dev = Device::Cpu;
        // 1. Resize + norm + patchify
        let img = Tensor::from_vec(vec![128u8; 8 * 8 * 3], (8, 8, 3), &dev)?;
        let patches = vis_u8_resize_norm_patchify(
            &img,
            4,
            4,
            2,
            true,
            false,
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            DType::F32,
        )?;
        assert_eq!(patches.dims(), &[4, 12]);
        let pv = patches.flatten_all()?.to_vec1::<f32>()?;
        let expected = (128.0 / 255.0 - 0.5) / 0.5;
        for v in pv {
            assert!((v - expected).abs() < 1e-4, "got {v} vs {expected}");
        }

        // 2. 3-mode bias + GELU
        let x = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], (1, 4), &dev)?;
        let b = Tensor::from_vec(vec![0.1f32, -0.1, 0.2, 0.0], 4, &dev)?;
        for mode in [GeluMode::ExactErf, GeluMode::PytorchTanh, GeluMode::QuickGelu] {
            let y = fused_bias_gelu(&x, Some(&b), mode)?;
            assert_eq!(y.dims(), &[1, 4]);
        }

        // 3. QKV + 2D RoPE
        let qkv = Tensor::randn(0f32, 0.5, (4, 3 * 2 * 16), &dev)?;
        let yx = Tensor::from_vec(vec![0u32, 0, 0, 1, 1, 0, 1, 1], (4, 2), &dev)?;
        let inv_f = Tensor::from_vec(vec![1.0f32, 0.1, 0.01, 0.001], 4, &dev)?;
        let (q, k, v) = fused_vis_qkv_2d_rope(
            &qkv, None, None, None, &yx, &inv_f, 2, 16, 16, false, false, 1e-6,
        )?;
        assert_eq!(q.dims(), &[2, 4, 16]);
        assert_eq!(k.dims(), &[2, 4, 16]);
        assert_eq!(v.dims(), &[2, 4, 16]);

        // 4. Spatial merge + LN
        let feats = Tensor::randn(0f32, 1.0, (16, 8), &dev)?;
        let m_pre = fused_spatial_merge_norm(
            &feats,
            None,
            None,
            4,
            4,
            2,
            SpatialMergeMode::PreMergeNorm,
            1e-5,
        )?;
        assert_eq!(m_pre.dims(), &[4, 32]);
        let m_avg = fused_spatial_merge_norm(
            &feats,
            None,
            None,
            4,
            4,
            2,
            SpatialMergeMode::AvgPoolNorm,
            1e-5,
        )?;
        assert_eq!(m_avg.dims(), &[4, 8]);

        // 5. Scatter vision into embeds
        let embeds = Tensor::zeros((6, 4), DType::F32, &dev)?;
        let vis = Tensor::ones((2, 4), DType::F32, &dev)?;
        let idx = Tensor::from_vec(vec![1u32, 4u32], 2, &dev)?;
        let merged = scatter_vision_into_embeds(embeds, &vis, &idx)?;
        let mv = merged.to_vec2::<f32>()?;
        assert_eq!(mv[0], vec![0.0; 4]);
        assert_eq!(mv[1], vec![1.0; 4]);
        assert_eq!(mv[4], vec![1.0; 4]);

        // 6. 2D pos embed interp add
        let p = Tensor::zeros((4, 8), DType::F32, &dev)?;
        let tbl = Tensor::ones((2, 2, 8), DType::F32, &dev)?;
        let p2 = patch_add_interp_pos2d_inplace(p, &tbl, 2, 2, false, false)?;
        let p2v = p2.flatten_all()?.to_vec1::<f32>()?;
        assert!(p2v.iter().all(|&v| (v - 1.0).abs() < 1e-5));
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn vis_ops_rocm_matches_cpu_exact() -> Result<()> {
        let Ok(dev) = Device::new_rocm(0) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        // 1. Chan-Welford LayerNorm + LayerScale + F32 residual stream at mean = 1e4
        let d = 64usize;
        let base: Vec<f32> = (0..2 * d)
            .map(|i| 10_000.0 + ((i % 7) as f32 - 3.0))
            .collect();
        let x_cpu = Tensor::from_vec(base, (2, d), &cpu)?;
        let delta_cpu = Tensor::randn(0f32, 0.25, (2, d), &cpu)?.to_dtype(DType::BF16)?;
        let ls_cpu = Tensor::from_vec(vec![0.5f32; d], d, &cpu)?;
        let lnw_cpu = Tensor::from_vec(vec![1.1f32; d], d, &cpu)?;
        let lnb_cpu = Tensor::from_vec(vec![-0.2f32; d], d, &cpu)?;
        let (rc, nc) = fused_res_layernorm_ls(
            &x_cpu,
            Some(&delta_cpu),
            Some(&ls_cpu),
            Some(&lnw_cpu),
            Some(&lnb_cpu),
            1e-5,
            DType::BF16,
        )?;
        let (rg, ng) = fused_res_layernorm_ls(
            &x_cpu.to_device(&dev)?,
            Some(&delta_cpu.to_device(&dev)?),
            Some(&ls_cpu.to_device(&dev)?),
            Some(&lnw_cpu.to_device(&dev)?),
            Some(&lnb_cpu.to_device(&dev)?),
            1e-5,
            DType::BF16,
        )?;
        let diff_r = (&rc - &rg.to_device(&cpu)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let diff_n = (&nc.to_dtype(DType::F32)? - &ng.to_device(&cpu)?.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff_r < 1e-3, "vis_res_ln_ls ROCm res diff {diff_r}");
        assert!(diff_n < 2e-2, "vis_res_ln_ls ROCm norm diff {diff_n}");

        // 2. Resize + norm + patchify
        let img_u8 = Tensor::from_vec(
            (0..8 * 8 * 3).map(|i| ((i * 29) % 256) as u8).collect::<Vec<_>>(),
            (8, 8, 3),
            &cpu,
        )?;
        let pc = vis_u8_resize_norm_patchify(
            &img_u8, 4, 4, 2, true, false, [0.5; 3], [0.5; 3], DType::F32,
        )?;
        let pg = vis_u8_resize_norm_patchify(
            &img_u8.to_device(&dev)?,
            4, 4, 2, true, false, [0.5; 3], [0.5; 3], DType::F32,
        )?;
        let diff_p = (&pc - &pg.to_device(&cpu)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff_p < 1e-5, "vis_u8_resize ROCm diff {diff_p}");

        // 3. QKV + 2D RoPE + Spatial Merge + Scatter + Pos2D
        let qkv = Tensor::randn(0f32, 0.5, (4, 3 * 2 * 16), &cpu)?;
        let yx = Tensor::from_vec(vec![0u32, 0, 0, 1, 1, 0, 1, 1], (4, 2), &cpu)?;
        let inv_f = Tensor::from_vec(vec![1.0f32, 0.1, 0.01, 0.001], 4, &cpu)?;
        let (qc, kc, vc) = fused_vis_qkv_2d_rope(
            &qkv, None, None, None, &yx, &inv_f, 2, 16, 16, false, false, 1e-6,
        )?;
        let (qg, kg, vg) = fused_vis_qkv_2d_rope(
            &qkv.to_device(&dev)?,
            None,
            None,
            None,
            &yx.to_device(&dev)?,
            &inv_f.to_device(&dev)?,
            2,
            16,
            16,
            false,
            false,
            1e-6,
        )?;
        for (a, b, tag) in [(&qc, &qg, "q"), (&kc, &kg, "k"), (&vc, &vg, "v")] {
            let dmax = (a - &b.to_device(&cpu)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(dmax < 1e-5, "vis_qkv_2d_rope {tag} diff {dmax}");
        }

        let feats = Tensor::randn(0f32, 1.0, (16, 8), &cpu)?;
        let mc = fused_spatial_merge_norm(
            &feats, None, None, 4, 4, 2, SpatialMergeMode::PostMergeNorm, 1e-5,
        )?;
        let mg = fused_spatial_merge_norm(
            &feats.to_device(&dev)?, None, None, 4, 4, 2, SpatialMergeMode::PostMergeNorm, 1e-5,
        )?;
        let dm = (&mc - &mg.to_device(&cpu)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(dm < 1e-5, "vis_spatial_merge_norm diff {dm}");
        Ok(())
    }
}
