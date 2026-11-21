//! Lossless `lm_head` Fused Tile Argmax + Mean-Centered Cauchy-Schwarz (CS) Guard.
//!
//! - Mode A (`fused_linear_argmax_cs`): 2-Stage dense tile matmul + in-register argmax + CS guard
//!   with **0 bytes** of `[M, V]` logits materialization on CUDA and ROCm.
//! - Mode B (`logits_argmax_cs`): 1-Stage argmax + `core_ids` remap + CS guard directly over
//!   quantized `QMatMul` sub-logits `[M, V_core]` without expanding to `[M, V_full]`.

use crate::accel::lm_head::{self as lmh, better_argmax};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::op::BackpropOp;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::storage::Storage;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::tensor::from_storage;
use crate::{bail, DType, Device, Layout, Result, Tensor};

/// Precomputed Mean-Centered Cauchy-Schwarz (CS) upper bound over the pruned REST vocabulary.
///
/// For any hidden state `x`:
///   `max_{r in R} (x · w_r + b_r) <= x · mu_rest + ||x||_2 * w_rest_bound + b_rest_max`
#[derive(Clone, Debug)]
pub struct CsGuardParams {
    /// Mean weight vector of REST vocab rows: shape `[K]`, `DType::F32`.
    pub mu_rest: Tensor,
    /// `(1 + safety_margin) * max_{r in R} ||w_r - mu_rest||_2` (rounding safety margin >= 1e-3).
    pub w_rest_bound: f32,
    /// Maximum bias across REST vocab rows (`0.0` when there is no bias).
    pub b_rest_max: f32,
}

impl CsGuardParams {
    /// Precomputes `mu_rest`, centered max norm bound (with `safety_margin.max(1e-3)`),
    /// and `b_rest_max` once from dense REST weight `[V_rest, K]`.
    pub fn from_rest_weight(
        w_rest: &Tensor,
        b_rest: Option<&Tensor>,
        safety_margin: f32,
    ) -> Result<Self> {
        let (v_rest, _k) = w_rest.dims2()?;
        if v_rest == 0 {
            bail!("CsGuardParams::from_rest_weight: w_rest must have at least 1 row");
        }
        let w_f32 = w_rest.to_dtype(DType::F32)?;
        let mu_rest = w_f32.mean(0)?.contiguous()?;
        let centered = w_f32.broadcast_sub(&mu_rest)?;
        let max_norm = centered
            .sqr()?
            .sum(1)?
            .sqrt()?
            .max(0)?
            .to_scalar::<f32>()?;
        let b_rest_max = match b_rest {
            Some(b) => b.to_dtype(DType::F32)?.max(0)?.to_scalar::<f32>()?,
            None => 0.0,
        };
        Ok(Self {
            mu_rest,
            w_rest_bound: max_norm * (1.0 + safety_margin.max(1e-3)),
            b_rest_max,
        })
    }
}

/// Output of `fused_linear_argmax_cs` and `logits_argmax_cs`.
#[derive(Clone, Debug)]
pub struct ArgmaxCsOutput {
    /// Selected (or `core_ids`-remapped) token IDs: shape `[M]`, `DType::U32`.
    pub ids: Tensor,
    /// Winning CORE logit value per row: shape `[M]`, `DType::F32`.
    pub max_vals: Option<Tensor>,
    /// Per-row REST rescue flag (`0` = guaranteed exact, `1` = must evaluate REST): `[M]`, `DType::U8`.
    pub rescue_mask: Option<Tensor>,
}

fn check_float_dtype(dt: DType, op: &str) -> Result<()> {
    match dt {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        other => bail!("{op}: unsupported dtype {other:?}"),
    }
}

fn uniform_row_stride(layout: &Layout) -> Option<usize> {
    let dims = layout.dims();
    let strides = layout.stride();
    let rank = dims.len();
    if rank == 0 || strides[rank - 1] != 1 {
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

/// Mode A: Dense `lm_head` fused tile matmul + in-register argmax + optional CS guard.
///
/// - `hidden`: `[..., K]` (`F32` / `F16` / `BF16`).
/// - `weight`: `[V_core, K]` (`F32` / `F16` / `BF16`).
/// - `bias`: optional `[V_core]`.
/// - `core_ids`: optional `[V_core]` (`U32`) mapping local core indices to full vocabulary IDs.
/// - `cs_guard`: optional [`CsGuardParams`] for lossless REST rescue detection.
pub fn fused_linear_argmax_cs(
    hidden: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    core_ids: Option<&Tensor>,
    cs_guard: Option<&CsGuardParams>,
) -> Result<ArgmaxCsOutput> {
    let dev = hidden.device().clone();
    let dtype = hidden.dtype();
    check_float_dtype(dtype, "fused_linear_argmax_cs")?;
    let (v_valid, k) = weight.dims2()?;
    if hidden.dims().is_empty() || *hidden.dims().last().unwrap() != k {
        bail!(
            "fused_linear_argmax_cs: hidden shape {:?} incompatible with weight [{v_valid}, {k}]",
            hidden.dims()
        );
    }
    let m = hidden.elem_count() / k.max(1);

    let x_c = hidden.contiguous()?;
    let w_c = if weight.dtype() == dtype && weight.device().location() == dev.location() {
        weight.contiguous()?
    } else {
        weight.to_device(&dev)?.to_dtype(dtype)?.contiguous()?
    };
    let bias_f32 = match bias {
        Some(b) => {
            if b.dims() != [v_valid] {
                bail!(
                    "fused_linear_argmax_cs: bias shape {:?} != [{v_valid}]",
                    b.dims()
                );
            }
            Some(b.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?)
        }
        None => None,
    };
    let cids_u32 = match core_ids {
        Some(c) => {
            if c.dims() != [v_valid] {
                bail!(
                    "fused_linear_argmax_cs: core_ids shape {:?} != [{v_valid}]",
                    c.dims()
                );
            }
            Some(c.to_device(&dev)?.to_dtype(DType::U32)?.contiguous()?)
        }
        None => None,
    };
    let (mu_f32, w_bound, b_max, check_cs) = match cs_guard {
        Some(g) => {
            if g.mu_rest.dims() != [k] {
                bail!(
                    "fused_linear_argmax_cs: mu_rest shape {:?} != [{k}]",
                    g.mu_rest.dims()
                );
            }
            (
                Some(g.mu_rest.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?),
                g.w_rest_bound,
                g.b_rest_max,
                true,
            )
        }
        None => (None, 0.0f32, 0.0f32, false),
    };

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, wl) = w_c.storage_and_layout();
            let (Storage::Cuda(xc), Storage::Cuda(wc)) = (&*xs, &*ws) else {
                unreachable!()
            };
            let bg = bias_f32.as_ref().map(|t| t.storage_and_layout());
            let cg = cids_u32.as_ref().map(|t| t.storage_and_layout());
            let mg = mu_f32.as_ref().map(|t| t.storage_and_layout());
            let b_arg = bg.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let c_arg = cg.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let m_arg = mg.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let (id_s, val_s, res_s) = xc.fused_linear_argmax_cs(
                xl.start_offset(),
                wc,
                wl.start_offset(),
                b_arg,
                c_arg,
                m_arg,
                w_bound,
                b_max,
                check_cs,
                m,
                v_valid,
                k,
            )?;
            Ok(ArgmaxCsOutput {
                ids: from_storage(Storage::Cuda(id_s), (m,), BackpropOp::none(), false),
                max_vals: Some(from_storage(
                    Storage::Cuda(val_s),
                    (m,),
                    BackpropOp::none(),
                    false,
                )),
                rescue_mask: if check_cs {
                    Some(from_storage(
                        Storage::Cuda(res_s),
                        (m,),
                        BackpropOp::none(),
                        false,
                    ))
                } else {
                    None
                },
            })
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, wl) = w_c.storage_and_layout();
            let (Storage::Rocm(xr), Storage::Rocm(wr)) = (&*xs, &*ws) else {
                unreachable!()
            };
            let bg = bias_f32.as_ref().map(|t| t.storage_and_layout());
            let cg = cids_u32.as_ref().map(|t| t.storage_and_layout());
            let mg = mu_f32.as_ref().map(|t| t.storage_and_layout());
            let b_arg = bg.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let c_arg = cg.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let m_arg = mg.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let (id_s, val_s, res_s) = xr.fused_linear_argmax_cs(
                xl.start_offset(),
                wr,
                wl.start_offset(),
                b_arg,
                c_arg,
                m_arg,
                w_bound,
                b_max,
                check_cs,
                m,
                v_valid,
                k,
            )?;
            Ok(ArgmaxCsOutput {
                ids: from_storage(Storage::Rocm(id_s), (m,), BackpropOp::none(), false),
                max_vals: Some(from_storage(
                    Storage::Rocm(val_s),
                    (m,),
                    BackpropOp::none(),
                    false,
                )),
                rescue_mask: if check_cs {
                    Some(from_storage(
                        Storage::Rocm(res_s),
                        (m,),
                        BackpropOp::none(),
                        false,
                    ))
                } else {
                    None
                },
            })
        }
        _ => {
            // Chunked matmul + deterministic tie-break fallback (keeps logits memory tiny).
            let x_2d = x_c.reshape((m, k))?.to_dtype(DType::F32)?;
            let mut best_v = vec![f32::NEG_INFINITY; m];
            let mut best_i = vec![u32::MAX; m];
            const CHUNK_V: usize = 4096;
            let mut v0 = 0usize;
            while v0 < v_valid {
                let vn = (v_valid - v0).min(CHUNK_V);
                let wt = w_c.narrow(0, v0, vn)?.to_dtype(DType::F32)?.t()?;
                let mut chunk = x_2d.matmul(&wt)?;
                if let Some(ref b) = bias_f32 {
                    chunk = chunk.broadcast_add(&b.narrow(0, v0, vn)?.unsqueeze(0)?)?;
                }
                let cvec = chunk
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                for r in 0..m {
                    let row = &cvec[r * vn..(r + 1) * vn];
                    for (j, &val) in row.iter().enumerate() {
                        let gi = (v0 + j) as u32;
                        if better_argmax(val, gi, best_v[r], best_i[r]) {
                            best_v[r] = val;
                            best_i[r] = gi;
                        }
                    }
                }
                v0 += vn;
            }
            let cids_vec = match &cids_u32 {
                Some(c) => Some(c.to_device(&Device::Cpu)?.to_vec1::<u32>()?),
                None => None,
            };
            let out_ids: Vec<u32> = best_i
                .iter()
                .map(|&idx| match &cids_vec {
                    Some(c) if (idx as usize) < c.len() => c[idx as usize],
                    _ => idx,
                })
                .collect();
            let rescue = if check_cs {
                let x_cpu = x_2d.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                let mu_cpu = match &mu_f32 {
                    Some(mu) => Some(mu.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                    None => None,
                };
                let mut mask = vec![0u8; m];
                for r in 0..m {
                    let xr = &x_cpu[r * k..(r + 1) * k];
                    let mut xx = 0.0f32;
                    let mut xm = 0.0f32;
                    for kk in 0..k {
                        let hv = xr[kk];
                        xx = hv.mul_add(hv, xx);
                        if let Some(ref mu) = mu_cpu {
                            xm = hv.mul_add(mu[kk], xm);
                        }
                    }
                    let upper = xx.sqrt().mul_add(w_bound, xm) + b_max;
                    mask[r] = u8::from(best_v[r] < upper);
                }
                Some(Tensor::from_vec(mask, (m,), &dev)?)
            } else {
                None
            };
            Ok(ArgmaxCsOutput {
                ids: Tensor::from_vec(out_ids, (m,), &dev)?,
                max_vals: Some(Tensor::from_vec(best_v, (m,), &dev)?),
                rescue_mask: rescue,
            })
        }
    }
}

/// Mode B: 1-Stage Argmax + `core_ids` Remap + Mean-Centered CS Guard directly over
/// precomputed `sub_logits [..., V_core]` (e.g. GGUF `QMatMul` output).
pub fn logits_argmax_cs(
    sub_logits: &Tensor,
    hidden: Option<&Tensor>,
    core_ids: Option<&Tensor>,
    cs_guard: Option<&CsGuardParams>,
) -> Result<ArgmaxCsOutput> {
    let dev = sub_logits.device().clone();
    let l_dtype = sub_logits.dtype();
    check_float_dtype(l_dtype, "logits_argmax_cs")?;
    if sub_logits.dims().is_empty() {
        bail!("logits_argmax_cs: sub_logits must have rank >= 1");
    }
    let v_valid = *sub_logits.dims().last().unwrap();
    let m = sub_logits.elem_count() / v_valid.max(1);

    let l_owned;
    #[allow(unused_variables)]
    let (l_ref, l_stride) = match uniform_row_stride(sub_logits.layout()) {
        Some(s) => (sub_logits, s),
        None => {
            l_owned = sub_logits.contiguous()?;
            (&l_owned, v_valid)
        }
    };

    let cids_u32 = match core_ids {
        Some(c) => {
            if c.dims() != [v_valid] {
                bail!(
                    "logits_argmax_cs: core_ids shape {:?} != [{v_valid}]",
                    c.dims()
                );
            }
            Some(c.to_device(&dev)?.to_dtype(DType::U32)?.contiguous()?)
        }
        None => None,
    };

    let check_cs = cs_guard.is_some();
    let (h_c, k) = match hidden {
        Some(h) => {
            check_float_dtype(h.dtype(), "logits_argmax_cs(hidden)")?;
            if h.dims().is_empty() {
                bail!("logits_argmax_cs: hidden must have rank >= 1");
            }
            let k = *h.dims().last().unwrap();
            let hm = h.elem_count() / k.max(1);
            if hm != m {
                bail!("logits_argmax_cs: hidden rows {hm} != logits rows {m}");
            }
            // Match supported (TL, TH) kernel combinations: (f32, f32|f16|bf16) or same dtype.
            let target_h_dt = if l_dtype == DType::F32 || h.dtype() == l_dtype {
                h.dtype()
            } else {
                l_dtype
            };
            let hc = if h.device().location() == dev.location() && h.dtype() == target_h_dt {
                h.contiguous()?
            } else {
                h.to_device(&dev)?.to_dtype(target_h_dt)?.contiguous()?
            };
            (Some(hc), k)
        }
        None => {
            if check_cs {
                bail!("logits_argmax_cs: hidden is required when cs_guard is Some");
            }
            (None, 0)
        }
    };

    let (mu_f32, w_bound, b_max) = match cs_guard {
        Some(g) => {
            if g.mu_rest.dims() != [k] {
                bail!(
                    "logits_argmax_cs: mu_rest shape {:?} != [{k}]",
                    g.mu_rest.dims()
                );
            }
            (
                Some(g.mu_rest.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?),
                g.w_rest_bound,
                g.b_rest_max,
            )
        }
        None => (None, 0.0f32, 0.0f32),
    };

    match &dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            let (ls, ll) = l_ref.storage_and_layout();
            let Storage::Cuda(lc) = &*ls else {
                unreachable!()
            };
            let cg = cids_u32.as_ref().map(|t| t.storage_and_layout());
            let hg = h_c.as_ref().map(|t| t.storage_and_layout());
            let mg = mu_f32.as_ref().map(|t| t.storage_and_layout());
            let c_arg = cg.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let h_arg = hg.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let m_arg = mg.as_ref().map(|(s, l)| match &**s {
                Storage::Cuda(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let (id_s, val_s, res_s) = lc.logits_argmax_cs(
                ll.start_offset(),
                l_stride,
                c_arg,
                h_arg,
                m_arg,
                w_bound,
                b_max,
                check_cs,
                m,
                v_valid,
                k,
            )?;
            Ok(ArgmaxCsOutput {
                ids: from_storage(Storage::Cuda(id_s), (m,), BackpropOp::none(), false),
                max_vals: Some(from_storage(
                    Storage::Cuda(val_s),
                    (m,),
                    BackpropOp::none(),
                    false,
                )),
                rescue_mask: if check_cs {
                    Some(from_storage(
                        Storage::Cuda(res_s),
                        (m,),
                        BackpropOp::none(),
                        false,
                    ))
                } else {
                    None
                },
            })
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            let (ls, ll) = l_ref.storage_and_layout();
            let Storage::Rocm(lr) = &*ls else {
                unreachable!()
            };
            let cg = cids_u32.as_ref().map(|t| t.storage_and_layout());
            let hg = h_c.as_ref().map(|t| t.storage_and_layout());
            let mg = mu_f32.as_ref().map(|t| t.storage_and_layout());
            let c_arg = cg.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let h_arg = hg.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let m_arg = mg.as_ref().map(|(s, l)| match &**s {
                Storage::Rocm(c) => (c, l.start_offset()),
                _ => unreachable!(),
            });
            let (id_s, val_s, res_s) = lr.logits_argmax_cs(
                ll.start_offset(),
                l_stride,
                c_arg,
                h_arg,
                m_arg,
                w_bound,
                b_max,
                check_cs,
                m,
                v_valid,
                k,
            )?;
            Ok(ArgmaxCsOutput {
                ids: from_storage(Storage::Rocm(id_s), (m,), BackpropOp::none(), false),
                max_vals: Some(from_storage(
                    Storage::Rocm(val_s),
                    (m,),
                    BackpropOp::none(),
                    false,
                )),
                rescue_mask: if check_cs {
                    Some(from_storage(
                        Storage::Rocm(res_s),
                        (m,),
                        BackpropOp::none(),
                        false,
                    ))
                } else {
                    None
                },
            })
        }
        _ => {
            let l_cpu = l_ref
                .contiguous()?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let c_cpu = match &cids_u32 {
                Some(c) => Some(c.to_device(&Device::Cpu)?.to_vec1::<u32>()?),
                None => None,
            };
            let h_cpu = match &h_c {
                Some(h) => Some(
                    h.to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ),
                None => None,
            };
            let mu_cpu = match &mu_f32 {
                Some(mu) => Some(mu.to_device(&Device::Cpu)?.to_vec1::<f32>()?),
                None => None,
            };
            let (ids, vals, mask) = lmh::cpu_logits_argmax_cs_f32(
                &l_cpu,
                v_valid,
                c_cpu.as_deref(),
                h_cpu.as_deref(),
                mu_cpu.as_deref(),
                w_bound,
                b_max,
                check_cs,
                m,
                v_valid,
                k,
            );
            Ok(ArgmaxCsOutput {
                ids: Tensor::from_vec(ids, (m,), &dev)?,
                max_vals: Some(Tensor::from_vec(vals, (m,), &dev)?),
                rescue_mask: if check_cs {
                    Some(Tensor::from_vec(mask, (m,), &dev)?)
                } else {
                    None
                },
            })
        }
    }
}
