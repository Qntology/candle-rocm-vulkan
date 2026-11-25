//! Fused decoder operations: `residual_rmsnorm`, `residual_rmsnorm_inplace`, `rmsnorm_fused`,
//! `silu_mul`, and `rope_kv_write`.
//!
//! Dispatches to native fused kernels on CUDA (NVRTC) and ROCm (HIP), and to exact F32
//! reference implementations on CPU / Vulkan / Metal.

use crate::accel::decoder_ops::{
    self as dop, HeadLayout, RopeKvIndirectLaunchArgs, RopeKvLaunchArgs, RopeMode,
    ROPE_CODY_WAITE_ONTHEFLY, ROPE_NORM_PLUS_ONE,
};
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
use crate::op::BackpropOp;
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
use crate::storage::Storage;
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
use crate::tensor::from_storage;
use crate::{bail, CpuStorage, DType, Device, Layout, Result, Tensor};

/// Returns the uniform row stride (in elements) if the last dimension is contiguous (`stride == 1`)
/// and all outer dimensions form a single contiguous row sequence with fixed row stride.
fn uniform_row_stride(layout: &Layout) -> Option<usize> {
    let dims = layout.dims();
    let strides = layout.stride();
    let rank = dims.len();
    if rank == 0 {
        return None;
    }
    if strides[rank - 1] != 1 {
        return None;
    }
    if rank == 1 {
        return Some(dims[0]);
    }
    let row_stride = strides[rank - 2];
    let mut expected = row_stride;
    for axis in (0..rank - 2).rev() {
        expected = expected.checked_mul(dims[axis + 1])?;
        if strides[axis] != expected {
            return None;
        }
    }
    Some(row_stride)
}

fn check_float_dtype(dt: DType, op: &str) -> Result<()> {
    match dt {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        other => bail!("{op}: unsupported dtype {other:?} (expected F32, F16, or BF16)"),
    }
}

/// Fused Residual + RMSNorm (non-mutating): returns `(residual, normed)` where
/// `residual = x + delta` (or `x` when `delta` is `None`) and
/// `normed = rmsnorm(residual, plus_one ? 1 + weight : weight, eps)`.
pub fn residual_rmsnorm(
    x: &Tensor,
    delta: Option<&Tensor>,
    weight: &Tensor,
    eps: f32,
    plus_one: bool,
) -> Result<(Tensor, Tensor)> {
    res_rmsnorm_dispatch(x, delta, weight, eps, plus_one, true)
        .map(|(r, y)| (r.expect("write_residual=true"), y))
}

/// Fused Residual + RMSNorm with safe in-place residual overwrite.
///
/// If `x` is uniquely owned (`x.is_sole_owner_contiguous()`) and does not share storage with
/// `delta` or `weight`, `x` is updated in-place on GPU with `x + delta` (saving a full residual
/// tensor allocation). If `x` has multiple references (`Arc::strong_count > 1`), it safely
/// allocates a fresh residual tensor without mutating any shared view.
pub fn residual_rmsnorm_inplace(
    x: Tensor,
    delta: Option<&Tensor>,
    weight: &Tensor,
    eps: f32,
    plus_one: bool,
) -> Result<(Tensor, Tensor)> {
    let can_inplace = x.is_sole_owner_contiguous()
        && delta.map_or(true, |d| !x.shares_storage(d))
        && !x.shares_storage(weight);

    if !can_inplace {
        return residual_rmsnorm(&x, delta, weight, eps, plus_one);
    }

    let dev = x.device().clone();
    let dtype = x.dtype();
    check_float_dtype(dtype, "residual_rmsnorm_inplace")?;
    let dims = x.dims().to_vec();
    if dims.is_empty() {
        bail!("residual_rmsnorm_inplace: expected rank >= 1");
    }
    let d = *dims.last().unwrap();
    #[allow(unused_variables)]
    let rows = x.elem_count() / d.max(1);
    if weight.dims() != [d] {
        bail!(
            "residual_rmsnorm_inplace: weight shape {:?} != [{d}]",
            weight.dims()
        );
    }
    let delta_c = match delta {
        Some(dl) => {
            if dl.dims() != dims.as_slice() {
                bail!(
                    "residual_rmsnorm_inplace: delta shape {:?} != x shape {:?}",
                    dl.dims(),
                    dims
                );
            }
            if dl.dtype() != dtype || dl.device().location() != dev.location() {
                Some(dl.to_device(&dev)?.to_dtype(dtype)?.contiguous()?)
            } else {
                Some(dl.contiguous()?)
            }
        }
        None => None,
    };
    let w_f32 = weight.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?;

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let y_storage = {
                let mut xs = x.storage_mut();
                let Storage::Cuda(xc) = &mut *xs else {
                    unreachable!()
                };
                let (ws, wl) = w_f32.storage_and_layout();
                let Storage::Cuda(wc) = &*ws else {
                    unreachable!()
                };
                let dl_guard = delta_c.as_ref().map(|t| t.storage_and_layout());
                let dl_arg = dl_guard.as_ref().map(|(s, l)| match &**s {
                    Storage::Cuda(dc) => (dc, l.start_offset()),
                    _ => unreachable!(),
                });
                Storage::Cuda(xc.res_rmsnorm_inplace(
                    dl_arg,
                    wc,
                    wl.start_offset(),
                    rows,
                    d,
                    eps,
                    plus_one,
                )?)
            };
            let y = from_storage(y_storage, dims, BackpropOp::none(), false);
            Ok((x, y))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let y_storage = {
                let mut xs = x.storage_mut();
                let Storage::Rocm(xr) = &mut *xs else {
                    unreachable!()
                };
                let (ws, wl) = w_f32.storage_and_layout();
                let Storage::Rocm(wr) = &*ws else {
                    unreachable!()
                };
                let dl_guard = delta_c.as_ref().map(|t| t.storage_and_layout());
                let dl_arg = dl_guard.as_ref().map(|(s, l)| match &**s {
                    Storage::Rocm(dr) => (dr, l.start_offset()),
                    _ => unreachable!(),
                });
                Storage::Rocm(xr.res_rmsnorm_inplace(
                    dl_arg,
                    wr,
                    wl.start_offset(),
                    rows,
                    d,
                    eps,
                    plus_one,
                )?)
            };
            let y = from_storage(y_storage, dims, BackpropOp::none(), false);
            Ok((x, y))
        }
        #[cfg(feature = "vulkan")]
        Device::Vulkan(vdev) if crate::vulkan_backend::decoder_ops::vk_fused_ready(vdev) => {
            let y_res = {
                let mut xs = x.storage_mut();
                let Storage::Vulkan(xv) = &mut *xs else {
                    unreachable!()
                };
                let (ws, wl) = w_f32.storage_and_layout();
                let Storage::Vulkan(wv) = &*ws else {
                    unreachable!()
                };
                let dl_guard = delta_c.as_ref().map(|t| t.storage_and_layout());
                let dl_arg = dl_guard.as_ref().map(|(s, l)| match &**s {
                    Storage::Vulkan(dv) => (dv, l.start_offset()),
                    _ => unreachable!(),
                });
                xv.res_rmsnorm_inplace(dl_arg, wv, wl.start_offset(), rows, d, eps, plus_one)
            };
            match y_res {
                Ok(y_s) => {
                    let y = from_storage(Storage::Vulkan(y_s), dims, BackpropOp::none(), false);
                    Ok((x, y))
                }
                Err(_) => residual_rmsnorm(&x, delta_c.as_ref(), &w_f32, eps, plus_one),
            }
        }
        _ => residual_rmsnorm(&x, delta_c.as_ref(), &w_f32, eps, plus_one),
    }
}

/// Fused standalone RMSNorm (`delta = None`, no residual buffer allocated).
/// Supports standard `w` (`plus_one = false`) and Qwen3.5 / Qwen4-Exp `(1 + w)` (`plus_one = true`).
pub fn rmsnorm_fused(x: &Tensor, weight: &Tensor, eps: f32, plus_one: bool) -> Result<Tensor> {
    res_rmsnorm_dispatch(x, None, weight, eps, plus_one, false).map(|(_, y)| y)
}

fn res_rmsnorm_dispatch(
    x: &Tensor,
    delta: Option<&Tensor>,
    weight: &Tensor,
    eps: f32,
    plus_one: bool,
    write_residual: bool,
) -> Result<(Option<Tensor>, Tensor)> {
    let dev = x.device().clone();
    let dtype = x.dtype();
    check_float_dtype(dtype, "residual_rmsnorm")?;
    let dims = x.dims().to_vec();
    if dims.is_empty() {
        bail!("residual_rmsnorm: expected rank >= 1");
    }
    let d = *dims.last().unwrap();
    #[allow(unused_variables)]
    let rows = x.elem_count() / d.max(1);
    if weight.dims() != [d] {
        bail!(
            "residual_rmsnorm: weight shape {:?} != [{d}]",
            weight.dims()
        );
    }

    let x_c = x.contiguous()?;
    let delta_c = match delta {
        Some(dl) => {
            if dl.dims() != dims.as_slice() {
                bail!(
                    "residual_rmsnorm: delta shape {:?} != x shape {:?}",
                    dl.dims(),
                    dims
                );
            }
            if dl.dtype() != dtype || dl.device().location() != dev.location() {
                Some(dl.to_device(&dev)?.to_dtype(dtype)?.contiguous()?)
            } else {
                Some(dl.contiguous()?)
            }
        }
        None => None,
    };
    let w_f32 = weight.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?;

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (xs, xl) = x_c.storage_and_layout();
            let Storage::Cuda(xc) = &*xs else {
                unreachable!()
            };
            let (ws, wl) = w_f32.storage_and_layout();
            let Storage::Cuda(wc) = &*ws else {
                unreachable!()
            };
            let dl_guard = delta_c.as_ref().map(|t| t.storage_and_layout());
            let dl_arg = dl_guard.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(dc) => (dc, l.start_offset()),
                _ => unreachable!(),
            });
            let (r_st, y_st) = xc.res_rmsnorm(
                xl.start_offset(),
                dl_arg,
                wc,
                wl.start_offset(),
                rows,
                d,
                eps,
                plus_one,
                write_residual,
            )?;
            let r_t = r_st.map(|s| {
                from_storage(Storage::Cuda(s), dims.clone(), BackpropOp::none(), false)
            });
            let y_t = from_storage(Storage::Cuda(y_st), dims, BackpropOp::none(), false);
            Ok((r_t, y_t))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (xs, xl) = x_c.storage_and_layout();
            let Storage::Rocm(xr) = &*xs else {
                unreachable!()
            };
            let (ws, wl) = w_f32.storage_and_layout();
            let Storage::Rocm(wr) = &*ws else {
                unreachable!()
            };
            let dl_guard = delta_c.as_ref().map(|t| t.storage_and_layout());
            let dl_arg = dl_guard.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(dr) => (dr, l.start_offset()),
                _ => unreachable!(),
            });
            let (r_st, y_st) = xr.res_rmsnorm(
                xl.start_offset(),
                dl_arg,
                wr,
                wl.start_offset(),
                rows,
                d,
                eps,
                plus_one,
                write_residual,
            )?;
            let r_t = r_st.map(|s| {
                from_storage(Storage::Rocm(s), dims.clone(), BackpropOp::none(), false)
            });
            let y_t = from_storage(Storage::Rocm(y_st), dims, BackpropOp::none(), false);
            Ok((r_t, y_t))
        }
        #[cfg(feature = "vulkan")]
        Device::Vulkan(vdev)
            if crate::vulkan_backend::decoder_ops::vk_fused_ready(vdev)
                && (dtype == DType::F32
                    || (d % 2 == 0
                        && x_c.layout().start_offset() % 2 == 0
                        && delta_c
                            .as_ref()
                            .map_or(true, |dl| dl.layout().start_offset() % 2 == 0))) =>
        {
            let (xs, xl) = x_c.storage_and_layout();
            let Storage::Vulkan(xv) = &*xs else {
                unreachable!()
            };
            let (ws, wl) = w_f32.storage_and_layout();
            let Storage::Vulkan(wv) = &*ws else {
                unreachable!()
            };
            let dl_guard = delta_c.as_ref().map(|t| t.storage_and_layout());
            let dl_arg = dl_guard.as_ref().map(|(s, l)| match &**s {
                Storage::Vulkan(dv) => (dv, l.start_offset()),
                _ => unreachable!(),
            });
            let (r_st, y_st) = xv.res_rmsnorm(
                xl.start_offset(),
                dl_arg,
                wv,
                wl.start_offset(),
                rows,
                d,
                eps,
                plus_one,
                write_residual,
            )?;
            let r_t = r_st.map(|s| {
                from_storage(Storage::Vulkan(s), dims.clone(), BackpropOp::none(), false)
            });
            let y_t = from_storage(Storage::Vulkan(y_st), dims, BackpropOp::none(), false);
            Ok((r_t, y_t))
        }
        _ => {
            let x_vec = x_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let dl_vec = match &delta_c {
                Some(dl) => Some(
                    dl.to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ),
                None => None,
            };
            let w_vec = w_f32.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            let (r_vec, y_vec) =
                dop::cpu_res_rmsnorm_f32(&x_vec, dl_vec.as_deref(), &w_vec, rows, d, eps, plus_one);
            let r_t = if write_residual {
                Some(
                    Tensor::from_vec(r_vec, dims.as_slice(), &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_device(&dev)?,
                )
            } else {
                None
            };
            let y_t = Tensor::from_vec(y_vec, dims.as_slice(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)?;
            Ok((r_t, y_t))
        }
    }
}

/// Fused SwiGLU `silu(gate) * up`.
///
/// Zero-copy when `gate` and `up` are narrow slices of a merged `[..., 2*I]` `gate_up` projection
/// (e.g. `gate = gu.narrow(D::Minus1, 0, i)`, `up = gu.narrow(D::Minus1, i, i)`).
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let dev = gate.device().clone();
    let dtype = gate.dtype();
    check_float_dtype(dtype, "silu_mul")?;
    if gate.dims() != up.dims() {
        bail!(
            "silu_mul: shape mismatch gate {:?} vs up {:?}",
            gate.dims(),
            up.dims()
        );
    }
    if up.dtype() != dtype || up.device().location() != dev.location() {
        bail!("silu_mul: gate and up must share device and dtype");
    }
    let dims = gate.dims().to_vec();
    if dims.is_empty() {
        bail!("silu_mul: expected rank >= 1");
    }
    let inter_dim = *dims.last().unwrap();
    let rows = gate.elem_count() / inter_dim.max(1);

    let g_owned;
    #[allow(unused_variables)]
    let (g_ref, g_stride) = match uniform_row_stride(gate.layout()) {
        Some(s) => (gate, s),
        None => {
            g_owned = gate.contiguous()?;
            (&g_owned, inter_dim)
        }
    };
    let u_owned;
    #[allow(unused_variables)]
    let (u_ref, u_stride) = match uniform_row_stride(up.layout()) {
        Some(s) => (up, s),
        None => {
            u_owned = up.contiguous()?;
            (&u_owned, inter_dim)
        }
    };

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (gs, gl) = g_ref.storage_and_layout();
            let (us, ul) = u_ref.storage_and_layout();
            let (Storage::Cuda(gc), Storage::Cuda(uc)) = (&*gs, &*us) else {
                unreachable!()
            };
            let out = gc.silu_mul(
                gl.start_offset(),
                g_stride,
                uc,
                ul.start_offset(),
                u_stride,
                rows,
                inter_dim,
            )?;
            Ok(from_storage(
                Storage::Cuda(out),
                dims,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (gs, gl) = g_ref.storage_and_layout();
            let (us, ul) = u_ref.storage_and_layout();
            let (Storage::Rocm(gr), Storage::Rocm(ur)) = (&*gs, &*us) else {
                unreachable!()
            };
            let out = gr.silu_mul(
                gl.start_offset(),
                g_stride,
                ur,
                ul.start_offset(),
                u_stride,
                rows,
                inter_dim,
            )?;
            Ok(from_storage(
                Storage::Rocm(out),
                dims,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "vulkan")]
        Device::Vulkan(vdev)
            if crate::vulkan_backend::decoder_ops::vk_fused_ready(vdev)
                && (dtype == DType::F32 || inter_dim % 2 == 0) =>
        {
            let (gs, gl) = g_ref.storage_and_layout();
            let (us, ul) = u_ref.storage_and_layout();
            let (Storage::Vulkan(gv), Storage::Vulkan(uv)) = (&*gs, &*us) else {
                unreachable!()
            };
            let out = gv.silu_mul(
                gl.start_offset(),
                g_stride,
                uv,
                ul.start_offset(),
                u_stride,
                rows,
                inter_dim,
            )?;
            Ok(from_storage(
                Storage::Vulkan(out),
                dims,
                BackpropOp::none(),
                false,
            ))
        }
        _ => {
            let gv = g_ref
                .contiguous()?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let uv = u_ref
                .contiguous()?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ov = dop::cpu_silu_mul_f32(&gv, inter_dim, &uv, inter_dim, rows, inter_dim);
            Tensor::from_vec(ov, dims.as_slice(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}

fn validate_head_4d(
    t: &Tensor,
    layout: HeadLayout,
    name: &str,
) -> Result<(usize, usize, usize, usize, [u64; 3])> {
    let l = t.layout();
    let dims = l.dims();
    let strides = l.stride();
    if dims.len() != 4 || strides[3] != 1 {
        bail!("{name}: expected 4D tensor with contiguous last dim, got {dims:?} stride {strides:?}");
    }
    let (b, h, seq, d, st) = match layout {
        HeadLayout::Bshd => (
            dims[0],
            dims[2],
            dims[1],
            dims[3],
            [strides[0] as u64, strides[2] as u64, strides[1] as u64],
        ),
        HeadLayout::Bhsd => (
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            [strides[0] as u64, strides[1] as u64, strides[2] as u64],
        ),
    };
    Ok((b, h, seq, d, st))
}

/// Validates a 4D pre-allocated KV cache buffer (`[B, Hkv, max_seq, D]` or `[B, max_seq, Hkv, D]`)
/// and returns element strides `[sb, sh, ss]`.
pub fn validate_kv_cache_slice(
    cache_l: &Layout,
    b: usize,
    h_kv: usize,
    past_len: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<[u64; 3]> {
    let dims = cache_l.dims();
    let strides = cache_l.stride();
    if dims.len() != 4 || strides[3] != 1 || dims[3] != head_dim {
        bail!(
            "rope_kv_write: KV cache must be 4D with contiguous head_dim={head_dim}, got {dims:?} stride {strides:?}"
        );
    }
    if dims[0] != b {
        bail!(
            "rope_kv_write: KV cache batch mismatch (expected {b}, got {})",
            dims[0]
        );
    }
    let (h_axis, s_axis) = if dims[1] == h_kv && past_len + seq_len <= dims[2] {
        (1, 2)
    } else if dims[2] == h_kv && past_len + seq_len <= dims[1] {
        (2, 1)
    } else {
        bail!(
            "rope_kv_write: KV cache shape {dims:?} cannot fit h_kv={h_kv}, past={past_len} + T={seq_len}"
        );
    };
    Ok([
        strides[0] as u64,
        strides[h_axis] as u64,
        strides[s_axis] as u64,
    ])
}

/// Fused QK Head RMSNorm + Partial/Full 1D/3D-M RoPE + KV Cache Direct Write.
///
/// Writes rotated `K` and raw `V` directly into `k_cache` and `v_cache` at `past_len..past_len+T`,
/// and returns rotated `Q` laid out according to `q_out_layout` (`Bshd` or `Bhsd`).
///
/// - `q_in`, `k_in`, `v_in`: 4D tensors in `in_layout` (`Bshd` `[B, T, H, D]` or `Bhsd` `[B, H, T, D]`).
/// - `q_norm_w`, `k_norm_w`: optional per-head RMSNorm weights `[D]`.
/// - `cos_tab`, `sin_tab`: `[T, rotary_dim / 2]` (1D RoPE) or `[B, T, rotary_dim / 2]` (3D M-RoPE).
/// - `k_cache`, `v_cache`: pre-allocated 4D caches `[B, Hkv, max_seq, D]` or `[B, max_seq, Hkv, D]`.
#[allow(clippy::too_many_arguments)]
pub fn rope_kv_write(
    q_in: &Tensor,
    k_in: &Tensor,
    v_in: &Tensor,
    in_layout: HeadLayout,
    q_out_layout: HeadLayout,
    q_norm_w: Option<&Tensor>,
    k_norm_w: Option<&Tensor>,
    cos_tab: &Tensor,
    sin_tab: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    past_len: usize,
    eps: f32,
    rope_mode: RopeMode,
    norm_plus_one: bool,
) -> Result<Tensor> {
    let dev = q_in.device().clone();
    let dtype = q_in.dtype();
    check_float_dtype(dtype, "rope_kv_write")?;

    let q_c = if q_in.layout().stride().last().copied() == Some(1) {
        q_in.clone()
    } else {
        q_in.contiguous()?
    };
    let k_c = if k_in.layout().stride().last().copied() == Some(1) {
        k_in.clone()
    } else {
        k_in.contiguous()?
    };
    let v_c = if v_in.layout().stride().last().copied() == Some(1) {
        v_in.clone()
    } else {
        v_in.contiguous()?
    };

    let (b, hq, t, d, q_strides) = validate_head_4d(&q_c, in_layout, "rope_kv_write(q_in)")?;
    let (bk, hkv, tk, dk, k_strides) = validate_head_4d(&k_c, in_layout, "rope_kv_write(k_in)")?;
    let (bv, hkv_v, tv, dv, v_strides) = validate_head_4d(&v_c, in_layout, "rope_kv_write(v_in)")?;
    if bk != b || bv != b || tk != t || tv != t || dk != d || dv != d || hkv_v != hkv {
        bail!("rope_kv_write: Q/K/V shape mismatch");
    }
    if k_c.dtype() != dtype || v_c.dtype() != dtype || k_cache.dtype() != dtype || v_cache.dtype() != dtype {
        bail!("rope_kv_write: Q, K, V, k_cache, v_cache must all share dtype {dtype:?}");
    }
    if k_cache.shares_storage(v_cache) {
        bail!("rope_kv_write: k_cache and v_cache must not alias");
    }
    if k_cache.shares_storage(&k_c) || v_cache.shares_storage(&v_c) || k_cache.shares_storage(&q_c) {
        bail!("rope_kv_write: cache buffers must not alias inputs");
    }

    let kc_strides = validate_kv_cache_slice(k_cache.layout(), b, hkv, past_len, t, d)?;
    let vc_strides = validate_kv_cache_slice(v_cache.layout(), b, hkv, past_len, t, d)?;
    if kc_strides != vc_strides {
        bail!("rope_kv_write: k_cache and v_cache layouts must match");
    }

    let cos_f32 = cos_tab.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?;
    let sin_f32 = sin_tab.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?;
    if cos_f32.dims() != sin_f32.dims() {
        bail!("rope_kv_write: cos and sin shapes must match");
    }
    let (half_rot, cos_strides) = match cos_f32.dims() {
        [ct, hr] if *ct == t => (*hr, [0u64, *hr as u64]),
        [cb, ct, hr] if *cb == b && *ct == t => (*hr, [(ct * hr) as u64, *hr as u64]),
        other => bail!(
            "rope_kv_write: cos/sin shape {other:?} must be [T={t}, rd/2] or [B={b}, T={t}, rd/2]"
        ),
    };
    let rotary_dim = half_rot * 2;
    if rotary_dim > d {
        bail!("rope_kv_write: rotary_dim {rotary_dim} > head_dim {d}");
    }

    let qnw_f32 = match q_norm_w {
        Some(w) => {
            if w.dims() != [d] {
                bail!("rope_kv_write: q_norm_w shape {:?} != [{d}]", w.dims());
            }
            Some(w.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?)
        }
        None => None,
    };
    let knw_f32 = match k_norm_w {
        Some(w) => {
            if w.dims() != [d] {
                bail!("rope_kv_write: k_norm_w shape {:?} != [{d}]", w.dims());
            }
            Some(w.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?)
        }
        None => None,
    };

    let (q_out_dims, qo_strides) = match q_out_layout {
        HeadLayout::Bshd => (
            vec![b, t, hq, d],
            [(t * hq * d) as u64, d as u64, (hq * d) as u64],
        ),
        HeadLayout::Bhsd => (
            vec![b, hq, t, d],
            [(hq * t * d) as u64, (t * d) as u64, d as u64],
        ),
    };

    let flags = rope_mode.flag() | if norm_plus_one { ROPE_NORM_PLUS_ONE } else { 0 };
    let args = RopeKvLaunchArgs {
        b: b as u32,
        t: t as u32,
        hq: hq as u32,
        hkv: hkv as u32,
        d: d as u32,
        rotary_dim: rotary_dim as u32,
        past_len: past_len as u32,
        q_off: q_c.layout().start_offset(),
        k_off: k_c.layout().start_offset(),
        v_off: v_c.layout().start_offset(),
        qnw_off: qnw_f32.as_ref().map_or(0, |w| w.layout().start_offset()),
        knw_off: knw_f32.as_ref().map_or(0, |w| w.layout().start_offset()),
        cos_off: cos_f32.layout().start_offset(),
        sin_off: sin_f32.layout().start_offset(),
        kc_off: k_cache.layout().start_offset(),
        vc_off: v_cache.layout().start_offset(),
        q_strides,
        qo_strides,
        k_strides,
        v_strides,
        kc_strides,
        cos_strides,
        eps,
        flags,
    };

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (cs, _) = cos_f32.storage_and_layout();
            let (ss, _) = sin_f32.storage_and_layout();
            let qnw_g = qnw_f32.as_ref().map(|w| w.storage_and_layout());
            let knw_g = knw_f32.as_ref().map(|w| w.storage_and_layout());
            let mut kc_s = k_cache.storage_mut();
            let mut vc_s = v_cache.storage_mut();
            let (
                Storage::Cuda(qc),
                Storage::Cuda(kc_in),
                Storage::Cuda(vc_in),
                Storage::Cuda(cos_c),
                Storage::Cuda(sin_c),
                Storage::Cuda(kc_mut),
                Storage::Cuda(vc_mut),
            ) = (
                &*qs,
                &*ks,
                &*vs,
                &*cs,
                &*ss,
                &mut *kc_s,
                &mut *vc_s,
            )
            else {
                unreachable!()
            };
            let qnw_c = qnw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let knw_c = knw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let qo = qc.rope_kv_write(
                kc_in, vc_in, qnw_c, knw_c, cos_c, sin_c, kc_mut, vc_mut, &args,
            )?;
            Ok(from_storage(
                Storage::Cuda(qo),
                q_out_dims,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (cs, _) = cos_f32.storage_and_layout();
            let (ss, _) = sin_f32.storage_and_layout();
            let qnw_g = qnw_f32.as_ref().map(|w| w.storage_and_layout());
            let knw_g = knw_f32.as_ref().map(|w| w.storage_and_layout());
            let mut kc_s = k_cache.storage_mut();
            let mut vc_s = v_cache.storage_mut();
            let (
                Storage::Rocm(qr),
                Storage::Rocm(kr_in),
                Storage::Rocm(vr_in),
                Storage::Rocm(cos_r),
                Storage::Rocm(sin_r),
                Storage::Rocm(kc_mut),
                Storage::Rocm(vc_mut),
            ) = (
                &*qs,
                &*ks,
                &*vs,
                &*cs,
                &*ss,
                &mut *kc_s,
                &mut *vc_s,
            )
            else {
                unreachable!()
            };
            let qnw_r = qnw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(c) => c,
                _ => unreachable!(),
            });
            let knw_r = knw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(c) => c,
                _ => unreachable!(),
            });
            let qo = qr.rope_kv_write(
                kr_in, vr_in, qnw_r, knw_r, cos_r, sin_r, kc_mut, vc_mut, &args,
            )?;
            Ok(from_storage(
                Storage::Rocm(qo),
                q_out_dims,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "vulkan")]
        Device::Vulkan(vdev)
            if crate::vulkan_backend::decoder_ops::vk_fused_ready(vdev)
                && d <= 512
                && vdev.limits().max_storage_bindings >= 11
                && (dtype == DType::F32
                    || (d % 2 == 0
                        && args.kc_off % 2 == 0
                        && args.vc_off % 2 == 0
                        && args.qo_strides.iter().all(|&s| s % 2 == 0)
                        && args.kc_strides.iter().all(|&s| s % 2 == 0))) =>
        {
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (cs, _) = cos_f32.storage_and_layout();
            let (ss, _) = sin_f32.storage_and_layout();
            let qnw_g = qnw_f32.as_ref().map(|w| w.storage_and_layout());
            let knw_g = knw_f32.as_ref().map(|w| w.storage_and_layout());
            let mut kc_s = k_cache.storage_mut();
            let mut vc_s = v_cache.storage_mut();
            let (
                Storage::Vulkan(qv),
                Storage::Vulkan(kv_in),
                Storage::Vulkan(vv_in),
                Storage::Vulkan(cos_v),
                Storage::Vulkan(sin_v),
                Storage::Vulkan(kc_mut),
                Storage::Vulkan(vc_mut),
            ) = (
                &*qs,
                &*ks,
                &*vs,
                &*cs,
                &*ss,
                &mut *kc_s,
                &mut *vc_s,
            )
            else {
                unreachable!()
            };
            let qnw_v = qnw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Vulkan(c) => c,
                _ => unreachable!(),
            });
            let knw_v = knw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Vulkan(c) => c,
                _ => unreachable!(),
            });
            let qo = qv.rope_kv_write(
                kv_in, vv_in, qnw_v, knw_v, cos_v, sin_v, kc_mut, vc_mut, &args,
            )?;
            Ok(from_storage(
                Storage::Vulkan(qo),
                q_out_dims,
                BackpropOp::none(),
                false,
            ))
        }
        _ => {
            // Exact CPU reference fallback (also handles Vulkan/Metal by writing back to cache slices).
            let q_contig = q_c.contiguous()?;
            let k_contig = k_c.contiguous()?;
            let v_contig = v_c.contiguous()?;
            let (_, _, _, _, q_st_c) = validate_head_4d(&q_contig, in_layout, "q_c")?;
            let (_, _, _, _, k_st_c) = validate_head_4d(&k_contig, in_layout, "k_c")?;
            let (_, _, _, _, v_st_c) = validate_head_4d(&v_contig, in_layout, "v_c")?;
            let q_vec = q_contig
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let k_vec = k_contig
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let v_vec = v_contig
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let qnw_v = match &qnw_f32 {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let knw_v = match &knw_f32 {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let cos_v = cos_f32.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let sin_v = sin_f32.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

            let mut q_out_v = vec![0f32; b * t * hq * d];
            // Write into compact [B, Hkv, T, D] slices and scatter into k_cache/v_cache
            let mut k_new_v = vec![0f32; b * hkv * t * d];
            let mut v_new_v = vec![0f32; b * hkv * t * d];
            let mut cpu_args = args;
            cpu_args.past_len = 0;
            cpu_args.q_strides = q_st_c;
            cpu_args.k_strides = k_st_c;
            cpu_args.v_strides = v_st_c;
            cpu_args.kc_strides = [(hkv * t * d) as u64, (t * d) as u64, d as u64];
            dop::cpu_rope_kv_write_f32(
                &q_vec,
                &mut q_out_v,
                &k_vec,
                &v_vec,
                qnw_v.as_deref(),
                knw_v.as_deref(),
                &cos_v,
                &sin_v,
                &mut k_new_v,
                &mut v_new_v,
                &cpu_args,
            );
            let mut k_slice = Tensor::from_vec(k_new_v, (b, hkv, t, d), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)?;
            let mut v_slice = Tensor::from_vec(v_new_v, (b, hkv, t, d), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)?;
            // Determine cache seq axis (2 for [B, Hkv, S, D], 1 for [B, S, Hkv, D])
            let s_axis = if k_cache.dims()[1] == hkv && past_len + t <= k_cache.dims()[2] {
                2
            } else {
                k_slice = k_slice.transpose(1, 2)?.contiguous()?;
                v_slice = v_slice.transpose(1, 2)?.contiguous()?;
                1
            };
            k_cache.slice_set(&k_slice, s_axis, past_len)?;
            v_cache.slice_set(&v_slice, s_axis, past_len)?;

            // Special-case pure CPU CpuStorage direct write is already handled cleanly via slice_set above.
            let _ = CpuStorage::F32(Vec::new());
            Tensor::from_vec(q_out_v, q_out_dims.as_slice(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}

/// Outlier-free Residual + RMSNorm + Online Hadamard-32 + OCP E4M3FN dynamic quantization.
///
/// Applies orthonormal `H_32 / sqrt(32)` per 32-element channel block in registers after
/// RMSNorm and quantizes to `F8E4M3` with per-block `F32` scales `[..., D / 32]` (`amax / 448`).
/// Returns `(optional residual, fp8_tensor, scales_f32)`.
pub fn residual_rmsnorm_hadamard_fp8(
    x: &Tensor,
    delta: Option<&Tensor>,
    weight: &Tensor,
    eps: f32,
    plus_one: bool,
    write_residual: bool,
) -> Result<(Option<Tensor>, Tensor, Tensor)> {
    let dev = x.device().clone();
    let dtype = x.dtype();
    check_float_dtype(dtype, "residual_rmsnorm_hadamard_fp8")?;
    let dims = x.dims().to_vec();
    if dims.is_empty() {
        bail!("residual_rmsnorm_hadamard_fp8: expected rank >= 1");
    }
    let d = *dims.last().unwrap();
    if d == 0 || d % 32 != 0 {
        bail!("residual_rmsnorm_hadamard_fp8: last dim D={d} must be a non-zero multiple of 32");
    }
    let rows = x.elem_count() / d;
    if weight.dims() != [d] {
        bail!(
            "residual_rmsnorm_hadamard_fp8: weight shape {:?} != [{d}]",
            weight.dims()
        );
    }
    let mut scale_dims = dims.clone();
    *scale_dims.last_mut().unwrap() = d / 32;

    let x_c = x.contiguous()?;
    let dl_c = match delta {
        Some(dl) => {
            if dl.dims() != dims.as_slice() {
                bail!("residual_rmsnorm_hadamard_fp8: delta shape mismatch");
            }
            Some(dl.to_device(&dev)?.to_dtype(dtype)?.contiguous()?)
        }
        None => None,
    };
    let w_f32 = weight.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?;

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, wl) = w_f32.storage_and_layout();
            let dl_g = dl_c.as_ref().map(|t| t.storage_and_layout());
            let (Storage::Cuda(xc), Storage::Cuda(wc)) = (&*xs, &*ws) else {
                unreachable!()
            };
            let dl_arg = dl_g.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let (r_opt, fp8_s, sc_s) = xc.res_rmsnorm_hadamard_fp8(
                xl.start_offset(),
                dl_arg,
                wc,
                wl.start_offset(),
                rows,
                d,
                eps,
                plus_one,
                write_residual,
            )?;
            Ok((
                r_opt.map(|s| from_storage(Storage::Cuda(s), dims.clone(), BackpropOp::none(), false)),
                from_storage(Storage::Cuda(fp8_s), dims, BackpropOp::none(), false),
                from_storage(Storage::Cuda(sc_s), scale_dims, BackpropOp::none(), false),
            ))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, wl) = w_f32.storage_and_layout();
            let dl_g = dl_c.as_ref().map(|t| t.storage_and_layout());
            let (Storage::Rocm(xr), Storage::Rocm(wr)) = (&*xs, &*ws) else {
                unreachable!()
            };
            let dl_arg = dl_g.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(r) => (r, l.start_offset()),
                _ => unreachable!(),
            });
            let (r_opt, fp8_s, sc_s) = xr.res_rmsnorm_hadamard_fp8(
                xl.start_offset(),
                dl_arg,
                wr,
                wl.start_offset(),
                rows,
                d,
                eps,
                plus_one,
                write_residual,
            )?;
            Ok((
                r_opt.map(|s| from_storage(Storage::Rocm(s), dims.clone(), BackpropOp::none(), false)),
                from_storage(Storage::Rocm(fp8_s), dims, BackpropOp::none(), false),
                from_storage(Storage::Rocm(sc_s), scale_dims, BackpropOp::none(), false),
            ))
        }
        _ => {
            let xv = x_c
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let dv = match &dl_c {
                Some(dl) => Some(
                    dl.to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ),
                None => None,
            };
            let wv = w_f32.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            let (r_vec, y_vec) =
                dop::cpu_res_rmsnorm_f32(&xv, dv.as_deref(), &wv, rows, d, eps, plus_one);
            let (codes, scales) = dop::hadamard32_quant_fp8_cpu(&y_vec);
            let fp8_vals: Vec<float8::F8E4M3> = codes
                .into_iter()
                .map(float8::F8E4M3::from_bits)
                .collect();
            let r_t = if write_residual {
                Some(
                    Tensor::from_vec(r_vec, dims.as_slice(), &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_device(&dev)?,
                )
            } else {
                None
            };
            let fp8_t = Tensor::from_vec(fp8_vals, dims.as_slice(), &Device::Cpu)?.to_device(&dev)?;
            let sc_t = Tensor::from_vec(scales, scale_dims.as_slice(), &Device::Cpu)?.to_device(&dev)?;
            Ok((r_t, fp8_t, sc_t))
        }
    }
}

/// Outlier-free SwiGLU `silu(gate) * up` + Online Hadamard-32 + OCP E4M3FN dynamic quantization.
/// Returns `(fp8_tensor [..., inter_dim], scales_f32 [..., inter_dim / 32])`.
pub fn silu_mul_hadamard_fp8(gate: &Tensor, up: &Tensor) -> Result<(Tensor, Tensor)> {
    let dev = gate.device().clone();
    let dtype = gate.dtype();
    check_float_dtype(dtype, "silu_mul_hadamard_fp8")?;
    let dims = gate.dims().to_vec();
    if dims.is_empty() || up.dims() != dims.as_slice() {
        bail!("silu_mul_hadamard_fp8: shape mismatch");
    }
    let inter_dim = *dims.last().unwrap();
    if inter_dim == 0 || inter_dim % 32 != 0 {
        bail!("silu_mul_hadamard_fp8: inter_dim={inter_dim} must be a non-zero multiple of 32");
    }
    let rows = gate.elem_count() / inter_dim;
    let mut scale_dims = dims.clone();
    *scale_dims.last_mut().unwrap() = inter_dim / 32;

    let g_c = if uniform_row_stride(gate.layout()).is_some() {
        gate.clone()
    } else {
        gate.contiguous()?
    };
    let u_c = if up.dtype() == dtype
        && up.device().location() == dev.location()
        && uniform_row_stride(up.layout()).is_some()
    {
        up.clone()
    } else {
        up.to_device(&dev)?.to_dtype(dtype)?.contiguous()?
    };
    #[allow(unused_variables)]
    let g_stride = uniform_row_stride(g_c.layout()).unwrap();
    #[allow(unused_variables)]
    let u_stride = uniform_row_stride(u_c.layout()).unwrap();

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (gs, gl) = g_c.storage_and_layout();
            let (us, ul) = u_c.storage_and_layout();
            let (Storage::Cuda(gc), Storage::Cuda(uc)) = (&*gs, &*us) else {
                unreachable!()
            };
            let (fp8_s, sc_s) = gc.silu_mul_hadamard_fp8(
                gl.start_offset(),
                g_stride,
                uc,
                ul.start_offset(),
                u_stride,
                rows,
                inter_dim,
            )?;
            Ok((
                from_storage(Storage::Cuda(fp8_s), dims, BackpropOp::none(), false),
                from_storage(Storage::Cuda(sc_s), scale_dims, BackpropOp::none(), false),
            ))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (gs, gl) = g_c.storage_and_layout();
            let (us, ul) = u_c.storage_and_layout();
            let (Storage::Rocm(gr), Storage::Rocm(ur)) = (&*gs, &*us) else {
                unreachable!()
            };
            let (fp8_s, sc_s) = gr.silu_mul_hadamard_fp8(
                gl.start_offset(),
                g_stride,
                ur,
                ul.start_offset(),
                u_stride,
                rows,
                inter_dim,
            )?;
            Ok((
                from_storage(Storage::Rocm(fp8_s), dims, BackpropOp::none(), false),
                from_storage(Storage::Rocm(sc_s), scale_dims, BackpropOp::none(), false),
            ))
        }
        _ => {
            let gv = g_c
                .contiguous()?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let uv = u_c
                .contiguous()?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let y_vec = dop::cpu_silu_mul_f32(&gv, inter_dim, &uv, inter_dim, rows, inter_dim);
            let (codes, scales) = dop::hadamard32_quant_fp8_cpu(&y_vec);
            let fp8_vals: Vec<float8::F8E4M3> = codes
                .into_iter()
                .map(float8::F8E4M3::from_bits)
                .collect();
            let fp8_t = Tensor::from_vec(fp8_vals, dims.as_slice(), &Device::Cpu)?.to_device(&dev)?;
            let sc_t = Tensor::from_vec(scales, scale_dims.as_slice(), &Device::Cpu)?.to_device(&dev)?;
            Ok((fp8_t, sc_t))
        }
    }
}

/// Graph-safe Indirect QK Head RMSNorm + RoPE (table or Cody-Waite TwoProd on-the-fly) +
/// YaRN `mscale` + KV Cache Direct Write.
///
/// - `d_kv_pos`: `U32` tensor of shape `[1]` or `[B]` holding the physical cache write offset.
/// - `d_rope_pos`: optional `U32` tensor of shape `[1]` or `[B]` holding the RoPE phase offset
///   (e.g. `past_len + delta_mrope` for 1D-collapsed 3D M-RoPE or Delta-RoPE shift). Defaults to `d_kv_pos`.
/// - When `sin_tab` is `None`, `cos_tab` is interpreted as `inv_freq` (`[half_rot]`, F32) and
///   exact FP32 TwoProd Cody-Waite range-reduced `(cos, sin)` are computed on-the-fly (`<= 3e-7`
///   error vs FP64 up to `pos = 1,000,000+`).
/// - `mscale`: YaRN attention scale fused into `q_out` store (`1.0` = no scaling).
#[allow(clippy::too_many_arguments)]
pub fn rope_kv_write_indirect(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_norm_w: Option<&Tensor>,
    k_norm_w: Option<&Tensor>,
    cos_or_inv_freq: &Tensor,
    sin_tab: Option<&Tensor>,
    k_cache: &mut Tensor,
    v_cache: &mut Tensor,
    d_kv_pos: &Tensor,
    d_rope_pos: Option<&Tensor>,
    in_layout: HeadLayout,
    out_layout: HeadLayout,
    mode: RopeMode,
    eps: f32,
    norm_plus_one: bool,
    mscale: f32,
) -> Result<Tensor> {
    let dev = q.device().clone();
    let dtype = q.dtype();
    check_float_dtype(dtype, "rope_kv_write_indirect")?;
    let q_c = q.contiguous()?;
    let k_c = k.to_device(&dev)?.to_dtype(dtype)?.contiguous()?;
    let v_c = v.to_device(&dev)?.to_dtype(dtype)?.contiguous()?;

    let (b, t, hq, d, q_strides) = validate_head_4d(&q_c, in_layout, "q")?;
    let (bk, tk, hkv, dk, k_strides) = validate_head_4d(&k_c, in_layout, "k")?;
    let (bv, tv, hv, dv, v_strides) = validate_head_4d(&v_c, in_layout, "v")?;
    if bk != b || tk != t || dk != d || bv != b || tv != t || hv != hkv || dv != d {
        bail!("rope_kv_write_indirect: Q/K/V shape mismatch");
    }

    let kv_pos_c = d_kv_pos.to_device(&dev)?.to_dtype(DType::U32)?.contiguous()?;
    let kv_pos_len = kv_pos_c.elem_count();
    if kv_pos_len != 1 && kv_pos_len != b {
        bail!("rope_kv_write_indirect: d_kv_pos len {kv_pos_len} must be 1 or B={b}");
    }
    let pos_stride = if kv_pos_len == 1 { 0u32 } else { 1u32 };
    let rope_pos_c = match d_rope_pos {
        Some(rp) => {
            let c = rp.to_device(&dev)?.to_dtype(DType::U32)?.contiguous()?;
            if c.elem_count() != kv_pos_len {
                bail!("rope_kv_write_indirect: d_rope_pos len must match d_kv_pos");
            }
            Some(c)
        }
        None => None,
    };

    let cos_f32 = cos_or_inv_freq
        .to_device(&dev)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let sin_f32 = match sin_tab {
        Some(s) => Some(s.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    let onthefly_cw = sin_f32.is_none();
    let half_rot = *cos_f32.dims().last().unwrap_or(&0);
    let rotary_dim = half_rot * 2;
    if rotary_dim == 0 || rotary_dim > d {
        bail!("rope_kv_write_indirect: invalid rotary_dim {rotary_dim} for D={d}");
    }
    let cos_st = if onthefly_cw { 0u64 } else { half_rot as u64 };

    let kc_dims = k_cache.dims().to_vec();
    if kc_dims.len() != 4 || kc_dims != v_cache.dims() {
        bail!("rope_kv_write_indirect: k_cache/v_cache must be matching 4D tensors");
    }
    let (kc_strides, s_axis) = if kc_dims[1] == hkv {
        let s_cap = kc_dims[2];
        ([(hkv * s_cap * d) as u64, (s_cap * d) as u64, d as u64], 2usize)
    } else if kc_dims[2] == hkv {
        let s_cap = kc_dims[1];
        ([(s_cap * hkv * d) as u64, d as u64, (hkv * d) as u64], 1usize)
    } else {
        bail!("rope_kv_write_indirect: k_cache shape {kc_dims:?} does not match hkv={hkv}");
    };
    let _ = s_axis;

    let (q_out_dims, qo_strides) = match out_layout {
        HeadLayout::Bshd => (
            vec![b, t, hq, d],
            [(t * hq * d) as u64, d as u64, (hq * d) as u64],
        ),
        HeadLayout::Bhsd => (
            vec![b, hq, t, d],
            [(hq * t * d) as u64, (t * d) as u64, d as u64],
        ),
    };

    let qnw_f32 = match q_norm_w {
        Some(w) => Some(w.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    let knw_f32 = match k_norm_w {
        Some(w) => Some(w.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };

    let mut flags = mode.flag();
    if norm_plus_one {
        flags |= ROPE_NORM_PLUS_ONE;
    }
    if onthefly_cw {
        flags |= ROPE_CODY_WAITE_ONTHEFLY;
    }

    #[allow(unused_variables)]
    let args = RopeKvIndirectLaunchArgs {
        b: b as u32,
        t: t as u32,
        hq: hq as u32,
        hkv: hkv as u32,
        d: d as u32,
        rotary_dim: rotary_dim as u32,
        pos_stride,
        q_off: q_c.layout().start_offset(),
        k_off: k_c.layout().start_offset(),
        v_off: v_c.layout().start_offset(),
        qnw_off: qnw_f32.as_ref().map_or(0, |w| w.layout().start_offset()),
        knw_off: knw_f32.as_ref().map_or(0, |w| w.layout().start_offset()),
        cos_off: cos_f32.layout().start_offset(),
        sin_off: sin_f32.as_ref().map_or(0, |s| s.layout().start_offset()),
        kc_off: k_cache.layout().start_offset(),
        vc_off: v_cache.layout().start_offset(),
        kv_pos_off: kv_pos_c.layout().start_offset(),
        rope_pos_off: rope_pos_c.as_ref().map_or(0, |p| p.layout().start_offset()),
        q_strides,
        qo_strides,
        k_strides,
        v_strides,
        kc_strides,
        cos_st,
        eps,
        mscale,
        flags,
    };

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (cs, _) = cos_f32.storage_and_layout();
            let ss_g = sin_f32.as_ref().map(|t| t.storage_and_layout());
            let qnw_g = qnw_f32.as_ref().map(|t| t.storage_and_layout());
            let knw_g = knw_f32.as_ref().map(|t| t.storage_and_layout());
            let (kp_s, _) = kv_pos_c.storage_and_layout();
            let rp_g = rope_pos_c.as_ref().map(|t| t.storage_and_layout());
            let mut kc_s = k_cache.storage_mut();
            let mut vc_s = v_cache.storage_mut();
            let (
                Storage::Cuda(qc),
                Storage::Cuda(kc_in),
                Storage::Cuda(vc_in),
                Storage::Cuda(cos_c),
                Storage::Cuda(kvp_c),
                Storage::Cuda(kc_mut),
                Storage::Cuda(vc_mut),
            ) = (&*qs, &*ks, &*vs, &*cs, &*kp_s, &mut *kc_s, &mut *vc_s)
            else {
                unreachable!()
            };
            let sin_c = ss_g.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let qnw_c = qnw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let knw_c = knw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let rp_c = rp_g.as_ref().map(|(s, _)| match &**s {
                Storage::Cuda(c) => c,
                _ => unreachable!(),
            });
            let qo = qc.rope_kv_write_indirect(
                kc_in, vc_in, qnw_c, knw_c, cos_c, sin_c, kc_mut, vc_mut, kvp_c, rp_c, &args,
            )?;
            Ok(from_storage(
                Storage::Cuda(qo),
                q_out_dims,
                BackpropOp::none(),
                false,
            ))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (cs, _) = cos_f32.storage_and_layout();
            let ss_g = sin_f32.as_ref().map(|t| t.storage_and_layout());
            let qnw_g = qnw_f32.as_ref().map(|t| t.storage_and_layout());
            let knw_g = knw_f32.as_ref().map(|t| t.storage_and_layout());
            let (kp_s, _) = kv_pos_c.storage_and_layout();
            let rp_g = rope_pos_c.as_ref().map(|t| t.storage_and_layout());
            let mut kc_s = k_cache.storage_mut();
            let mut vc_s = v_cache.storage_mut();
            let (
                Storage::Rocm(qr),
                Storage::Rocm(kr_in),
                Storage::Rocm(vr_in),
                Storage::Rocm(cos_r),
                Storage::Rocm(kvp_r),
                Storage::Rocm(kc_mut),
                Storage::Rocm(vc_mut),
            ) = (&*qs, &*ks, &*vs, &*cs, &*kp_s, &mut *kc_s, &mut *vc_s)
            else {
                unreachable!()
            };
            let sin_r = ss_g.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(r) => r,
                _ => unreachable!(),
            });
            let qnw_r = qnw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(r) => r,
                _ => unreachable!(),
            });
            let knw_r = knw_g.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(r) => r,
                _ => unreachable!(),
            });
            let rp_r = rp_g.as_ref().map(|(s, _)| match &**s {
                Storage::Rocm(r) => r,
                _ => unreachable!(),
            });
            let qo = qr.rope_kv_write_indirect(
                kr_in, vr_in, qnw_r, knw_r, cos_r, sin_r, kc_mut, vc_mut, kvp_r, rp_r, &args,
            )?;
            Ok(from_storage(
                Storage::Rocm(qo),
                q_out_dims,
                BackpropOp::none(),
                false,
            ))
        }
        _ => {
            // Exact CPU fallback (also handles Vulkan/Metal cleanly).
            let kv_pos_v = kv_pos_c.to_device(&Device::Cpu)?.to_vec1::<u32>()?;
            let rope_pos_v = match &rope_pos_c {
                Some(rp) => rp.to_device(&Device::Cpu)?.to_vec1::<u32>()?,
                None => kv_pos_v.clone(),
            };
            let cos_v = cos_f32.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let sin_v = match &sin_f32 {
                Some(s) => Some(s.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
                None => None,
            };
            let q_vec = q_c.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let k_vec = k_c.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let v_vec = v_c.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let qnw_v = match &qnw_f32 {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let knw_v = match &knw_f32 {
                Some(w) => Some(w.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };

            let adjacent = mode == RopeMode::Adjacent;
            let mut q_out_v = vec![0f32; b * t * hq * d];
            let mut hbuf = vec![0f32; d];
            let mut cos_row = vec![0f32; half_rot];
            let mut sin_row = vec![0f32; half_rot];

            for bi in 0..b {
                let kv_p = kv_pos_v[bi * (pos_stride as usize)] as usize;
                let rope_p = rope_pos_v[bi * (pos_stride as usize)];
                let mut k_b = vec![0f32; hkv * t * d];
                let mut v_b = vec![0f32; hkv * t * d];
                for ti in 0..t {
                    let cur_pos = rope_p + ti as u32;
                    if onthefly_cw {
                        for i in 0..half_rot {
                            let (c, s) = dop::cody_waite_sincos_f32(cur_pos, cos_v[i]);
                            cos_row[i] = c;
                            sin_row[i] = s;
                        }
                    } else {
                        let off = (cur_pos as usize) * half_rot;
                        cos_row.copy_from_slice(&cos_v[off..off + half_rot]);
                        sin_row.copy_from_slice(&sin_v.as_ref().unwrap()[off..off + half_rot]);
                    }
                    for h in 0..hq {
                        let s_off = bi * q_strides[0] as usize + h * q_strides[1] as usize + ti * q_strides[2] as usize;
                        let d_off = bi * qo_strides[0] as usize + h * qo_strides[1] as usize + ti * qo_strides[2] as usize;
                        hbuf.copy_from_slice(&q_vec[s_off..s_off + d]);
                        if let Some(nw) = &qnw_v {
                            let sq: f64 = hbuf.iter().map(|&x| (x as f64) * (x as f64)).sum();
                            let inv = (1.0 / (sq / d as f64 + eps as f64).sqrt()) as f32;
                            for i in 0..d {
                                let wf = if norm_plus_one { 1.0 + nw[i] } else { nw[i] };
                                hbuf[i] = (hbuf[i] * inv) * wf;
                            }
                        }
                        for i in 0..half_rot {
                            let (c, s) = (cos_row[i], sin_row[i]);
                            if !adjacent {
                                let (x0, x1) = (hbuf[i], hbuf[i + half_rot]);
                                q_out_v[d_off + i] = (x0 * c - x1 * s) * mscale;
                                q_out_v[d_off + i + half_rot] = (x1 * c + x0 * s) * mscale;
                            } else {
                                let (x0, x1) = (hbuf[2 * i], hbuf[2 * i + 1]);
                                q_out_v[d_off + 2 * i] = (x0 * c - x1 * s) * mscale;
                                q_out_v[d_off + 2 * i + 1] = (x1 * c + x0 * s) * mscale;
                            }
                        }
                        for i in rotary_dim..d {
                            q_out_v[d_off + i] = hbuf[i] * mscale;
                        }
                    }
                    for h in 0..hkv {
                        let ks_off = bi * k_strides[0] as usize + h * k_strides[1] as usize + ti * k_strides[2] as usize;
                        let vs_off = bi * v_strides[0] as usize + h * v_strides[1] as usize + ti * v_strides[2] as usize;
                        let dst = h * t * d + ti * d;
                        hbuf.copy_from_slice(&k_vec[ks_off..ks_off + d]);
                        if let Some(nw) = &knw_v {
                            let sq: f64 = hbuf.iter().map(|&x| (x as f64) * (x as f64)).sum();
                            let inv = (1.0 / (sq / d as f64 + eps as f64).sqrt()) as f32;
                            for i in 0..d {
                                let wf = if norm_plus_one { 1.0 + nw[i] } else { nw[i] };
                                hbuf[i] = (hbuf[i] * inv) * wf;
                            }
                        }
                        for i in 0..half_rot {
                            let (c, s) = (cos_row[i], sin_row[i]);
                            if !adjacent {
                                let (x0, x1) = (hbuf[i], hbuf[i + half_rot]);
                                k_b[dst + i] = x0 * c - x1 * s;
                                k_b[dst + i + half_rot] = x1 * c + x0 * s;
                            } else {
                                let (x0, x1) = (hbuf[2 * i], hbuf[2 * i + 1]);
                                k_b[dst + 2 * i] = x0 * c - x1 * s;
                                k_b[dst + 2 * i + 1] = x1 * c + x0 * s;
                            }
                        }
                        if rotary_dim < d {
                            k_b[dst + rotary_dim..dst + d].copy_from_slice(&hbuf[rotary_dim..d]);
                        }
                        v_b[dst..dst + d].copy_from_slice(&v_vec[vs_off..vs_off + d]);
                    }
                }
                let mut ks_t = Tensor::from_vec(k_b, (1, hkv, t, d), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&dev)?;
                let mut vs_t = Tensor::from_vec(v_b, (1, hkv, t, d), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&dev)?;
                if s_axis == 1 {
                    ks_t = ks_t.transpose(1, 2)?.contiguous()?;
                    vs_t = vs_t.transpose(1, 2)?.contiguous()?;
                }
                let kc_b = k_cache.narrow(0, bi, 1)?;
                let vc_b = v_cache.narrow(0, bi, 1)?;
                kc_b.slice_set(&ks_t, s_axis, kv_p)?;
                vc_b.slice_set(&vs_t, s_axis, kv_p)?;
            }
            Tensor::from_vec(q_out_v, q_out_dims.as_slice(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&dev)
        }
    }
}
