//! Lossless `lm_head` Fused Tile Argmax + Mean-Centered Cauchy-Schwarz (CS) Guard.
//!
//! Exposes both low-level functional operators ([`fused_linear_argmax_cs`], [`logits_argmax_cs`])
//! and a high-level VRAM-resident [`GuardedLmHead`] module that guarantees 100% greedy token
//! equivalence with full-vocabulary `argmax(hidden @ W^T + b)`.

use candle::{DType, Result, Tensor};
use std::sync::Arc;

pub use candle::lm_head::{
    fused_linear_argmax_cs, fused_quant_argmax_cs, logits_argmax_cs, ArgmaxCsOutput, CsGuardParams,
};

/// Weight representation for a guarded `lm_head` (dense float or block-quantized [`QTensor`]).
#[derive(Clone)]
pub enum LmWeight {
    Dense(Tensor),
    Quantized(Arc<candle::quantized::QTensor>),
}

impl std::fmt::Debug for LmWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(t) => f.debug_tuple("Dense").field(t).finish(),
            Self::Quantized(q) => f
                .debug_tuple("Quantized")
                .field(q.shape())
                .field(&q.dtype())
                .finish(),
        }
    }
}

impl LmWeight {
    pub fn forward_argmax_cs(
        &self,
        hidden: &Tensor,
        bias: Option<&Tensor>,
        core_ids: Option<&Tensor>,
        cs_guard: Option<&CsGuardParams>,
    ) -> Result<ArgmaxCsOutput> {
        match self {
            Self::Dense(w) => fused_linear_argmax_cs(hidden, w, bias, core_ids, cs_guard),
            Self::Quantized(q) => fused_quant_argmax_cs(hidden, q, bias, core_ids, cs_guard),
        }
    }
}

/// Two-tier lossless greedy `lm_head` accepting dense or quantized CORE/REST weights.
///
/// Uses Mode A (`fused_linear_argmax_cs` / `fused_quant_argmax_cs`) for zero-materialization
/// decoding where supported and guarantees 100% greedy token agreement with full-vocabulary argmax.
#[derive(Clone, Debug)]
pub struct GuardedQuantLmHead {
    pub w_core: LmWeight,
    pub b_core: Option<Tensor>,
    pub core_ids: Tensor,
    pub w_rest: Option<LmWeight>,
    pub b_rest: Option<Tensor>,
    pub rest_ids: Option<Tensor>,
    pub cs_guard: Option<CsGuardParams>,
}

impl GuardedQuantLmHead {
    /// Builds a two-tier head from pre-split quantized or dense CORE/REST weights and computes
    /// [`CsGuardParams`] over the exact dequantized REST weights.
    pub fn from_split_weights(
        w_core: LmWeight,
        b_core: Option<Tensor>,
        core_ids: Tensor,
        w_rest: Option<(LmWeight, Option<Tensor>, Tensor)>,
        safety_margin: f32,
        device: &candle::Device,
    ) -> Result<Self> {
        let (w_rest_out, b_rest_out, rest_ids_out, cs_guard) = match w_rest {
            None => (None, None, None, None),
            Some((wr, br, rids)) => {
                let guard = match &wr {
                    LmWeight::Dense(t) => {
                        CsGuardParams::from_rest_weight(t, br.as_ref(), safety_margin)?
                    }
                    LmWeight::Quantized(q) => CsGuardParams::from_quant_rest_weight(
                        q,
                        br.as_ref(),
                        safety_margin,
                        device,
                    )?,
                };
                (Some(wr), br, Some(rids), Some(guard))
            }
        };
        Ok(Self {
            w_core,
            b_core,
            core_ids,
            w_rest: w_rest_out,
            b_rest: b_rest_out,
            rest_ids: rest_ids_out,
            cs_guard,
        })
    }

    /// Runs lossless greedy token selection for `hidden [..., K]`.
    /// Returns `(token_ids [M] u32, rescued_row_count)`.
    pub fn forward_greedy(&self, hidden: &Tensor) -> Result<(Tensor, usize)> {
        let dev = hidden.device();
        let k = *hidden.dims().last().unwrap_or(&0);
        let m = hidden.elem_count() / k.max(1);
        let h_2d = hidden.reshape((m, k))?;

        let core_out = self.w_core.forward_argmax_cs(
            &h_2d,
            self.b_core.as_ref(),
            Some(&self.core_ids),
            self.cs_guard.as_ref(),
        )?;

        let (Some(w_rest), Some(rest_ids), Some(mask_t), Some(core_vals_t)) = (
            self.w_rest.as_ref(),
            self.rest_ids.as_ref(),
            core_out.rescue_mask.as_ref(),
            core_out.max_vals.as_ref(),
        ) else {
            return Ok((core_out.ids, 0));
        };

        let masks = mask_t.to_vec1::<u8>()?;
        let rescue_rows: Vec<u32> = masks
            .iter()
            .enumerate()
            .filter_map(|(i, &flag)| (flag != 0).then_some(i as u32))
            .collect();
        if rescue_rows.is_empty() {
            return Ok((core_out.ids, 0));
        }

        let r_count = rescue_rows.len();
        let row_idx = Tensor::from_vec(rescue_rows.clone(), (r_count,), dev)?;
        let h_sub = h_2d.index_select(&row_idx, 0)?.contiguous()?;

        let rest_out = w_rest.forward_argmax_cs(
            &h_sub,
            self.b_rest.as_ref(),
            Some(rest_ids),
            None,
        )?;

        let mut ids = core_out.ids.to_vec1::<u32>()?;
        let core_vals = core_vals_t.to_vec1::<f32>()?;
        let rest_best_ids = rest_out.ids.to_vec1::<u32>()?;
        let rest_best_vals = rest_out
            .max_vals
            .expect("forward_argmax_cs always returns max_vals")
            .to_vec1::<f32>()?;

        for (k_i, &orig_row) in rescue_rows.iter().enumerate() {
            let r = orig_row as usize;
            let rv = rest_best_vals[k_i];
            let ri = rest_best_ids[k_i];
            let cv = core_vals[r];
            let ci = ids[r];
            if rv > cv || (rv == cv && ri < ci) {
                ids[r] = ri;
            }
        }

        let merged = Tensor::from_vec(ids, (m,), dev)?.to_dtype(DType::U32)?;
        Ok((merged, r_count))
    }
}

/// Two-tier lossless greedy `lm_head` holding a small CORE vocabulary weight (`w_core`) plus
/// optional VRAM-resident REST weight (`w_rest`) guarded by [`CsGuardParams`].
///
/// On >= 95% of decode steps only `w_core` is evaluated with zero `[M, V]` logits materialization;
/// on the rare rows where `rescue_mask[m] == 1`, `w_rest` is evaluated via pure fused tile argmax
/// (only on those rows) and replaces the winning token iff `rest_val > core_val` (or tie with
/// smaller original token ID).
#[derive(Clone, Debug)]
pub struct GuardedLmHead {
    pub w_core: Tensor,
    pub b_core: Option<Tensor>,
    pub core_ids: Tensor,
    pub w_rest: Option<Tensor>,
    pub b_rest: Option<Tensor>,
    pub rest_ids: Option<Tensor>,
    pub cs_guard: Option<CsGuardParams>,
}

impl GuardedLmHead {
    /// Splits a full dense weight `[V_full, K]` (and optional bias `[V_full]`) into CORE rows
    /// (`core_ids`) and REST rows, precomputing [`CsGuardParams`] once.
    pub fn from_full_weight(
        w_full: &Tensor,
        b_full: Option<&Tensor>,
        core_ids: &[u32],
        safety_margin: f32,
    ) -> Result<Self> {
        let (v_full, _k) = w_full.dims2()?;
        let dev = w_full.device();

        let mut is_core = vec![false; v_full];
        for &id in core_ids {
            if (id as usize) < v_full {
                is_core[id as usize] = true;
            }
        }
        let rest_vec: Vec<u32> = (0..v_full as u32)
            .filter(|&id| !is_core[id as usize])
            .collect();

        let core_ids_t = Tensor::from_vec(core_ids.to_vec(), (core_ids.len(),), dev)?;
        let w_core = w_full.index_select(&core_ids_t, 0)?.contiguous()?;
        let b_core = match b_full {
            Some(b) => Some(b.index_select(&core_ids_t, 0)?.contiguous()?),
            None => None,
        };

        if rest_vec.is_empty() {
            return Ok(Self {
                w_core,
                b_core,
                core_ids: core_ids_t,
                w_rest: None,
                b_rest: None,
                rest_ids: None,
                cs_guard: None,
            });
        }

        let rest_ids_t = Tensor::from_vec(rest_vec, (v_full - core_ids.len(),), dev)?;
        let w_rest = w_full.index_select(&rest_ids_t, 0)?.contiguous()?;
        let b_rest = match b_full {
            Some(b) => Some(b.index_select(&rest_ids_t, 0)?.contiguous()?),
            None => None,
        };
        let cs_guard = CsGuardParams::from_rest_weight(&w_rest, b_rest.as_ref(), safety_margin)?;

        Ok(Self {
            w_core,
            b_core,
            core_ids: core_ids_t,
            w_rest: Some(w_rest),
            b_rest,
            rest_ids: Some(rest_ids_t),
            cs_guard: Some(cs_guard),
        })
    }

    /// Runs lossless greedy token selection for `hidden [..., K]`.
    /// Returns `(token_ids [M] u32, rescued_row_count)`.
    pub fn forward_greedy(&self, hidden: &Tensor) -> Result<(Tensor, usize)> {
        let dev = hidden.device();
        let k = *hidden.dims().last().unwrap_or(&0);
        let m = hidden.elem_count() / k.max(1);
        let h_2d = hidden.reshape((m, k))?;

        let core_out = fused_linear_argmax_cs(
            &h_2d,
            &self.w_core,
            self.b_core.as_ref(),
            Some(&self.core_ids),
            self.cs_guard.as_ref(),
        )?;

        let (Some(w_rest), Some(rest_ids), Some(mask_t), Some(core_vals_t)) = (
            self.w_rest.as_ref(),
            self.rest_ids.as_ref(),
            core_out.rescue_mask.as_ref(),
            core_out.max_vals.as_ref(),
        ) else {
            return Ok((core_out.ids, 0));
        };

        let masks = mask_t.to_vec1::<u8>()?;
        let rescue_rows: Vec<u32> = masks
            .iter()
            .enumerate()
            .filter_map(|(i, &flag)| (flag != 0).then_some(i as u32))
            .collect();
        if rescue_rows.is_empty() {
            return Ok((core_out.ids, 0));
        }

        let r_count = rescue_rows.len();
        let row_idx = Tensor::from_vec(rescue_rows.clone(), (r_count,), dev)?;
        let h_sub = h_2d.index_select(&row_idx, 0)?.contiguous()?;

        let rest_out = fused_linear_argmax_cs(
            &h_sub,
            w_rest,
            self.b_rest.as_ref(),
            Some(rest_ids),
            None,
        )?;

        let mut ids = core_out.ids.to_vec1::<u32>()?;
        let core_vals = core_vals_t.to_vec1::<f32>()?;
        let rest_best_ids = rest_out.ids.to_vec1::<u32>()?;
        let rest_best_vals = rest_out
            .max_vals
            .expect("fused_linear_argmax_cs always returns max_vals")
            .to_vec1::<f32>()?;

        for (k_i, &orig_row) in rescue_rows.iter().enumerate() {
            let r = orig_row as usize;
            let rv = rest_best_vals[k_i];
            let ri = rest_best_ids[k_i];
            let cv = core_vals[r];
            let ci = ids[r];
            if rv > cv || (rv == cv && ri < ci) {
                ids[r] = ri;
            }
        }

        let merged = Tensor::from_vec(ids, (m,), dev)?.to_dtype(DType::U32)?;
        Ok((merged, r_count))
    }
}
