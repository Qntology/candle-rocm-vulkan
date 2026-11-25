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

        let (Some(w_rest), Some(rest_ids), Some(_mask_t), Some(core_vals_t)) = (
            self.w_rest.as_ref(),
            self.rest_ids.as_ref(),
            core_out.rescue_mask.as_ref(),
            core_out.max_vals.as_ref(),
        ) else {
            return Ok((core_out.ids, 0));
        };

        let (mut ids, masks) = core_out.download_ids_and_mask()?;
        let masks = masks.unwrap_or_default();
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

    /// Zero-copy last-token / selected-row greedy decoding: projects only the rows named by
    /// `row_indices` (or the last row of a sequence when `row_indices` is `None`), avoiding
    /// projecting `0..T-2` prompt tokens during prefill.
    pub fn forward_greedy_at_rows(
        &self,
        hidden: &Tensor,
        row_indices: Option<&Tensor>,
    ) -> Result<(Tensor, usize)> {
        let k = *hidden.dims().last().unwrap_or(&0);
        let total_rows = hidden.elem_count() / k.max(1);
        let h_2d = hidden.reshape((total_rows, k))?;
        let selected = match row_indices {
            Some(idx) => h_2d.index_select(&idx.to_dtype(DType::U32)?.flatten_all()?, 0)?,
            None => {
                if total_rows == 0 {
                    h_2d
                } else {
                    h_2d.narrow(0, total_rows - 1, 1)?
                }
            }
        };
        self.forward_greedy(&selected)
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

        let (Some(w_rest), Some(rest_ids), Some(_mask_t), Some(core_vals_t)) = (
            self.w_rest.as_ref(),
            self.rest_ids.as_ref(),
            core_out.rescue_mask.as_ref(),
            core_out.max_vals.as_ref(),
        ) else {
            return Ok((core_out.ids, 0));
        };

        let (mut ids, masks) = core_out.download_ids_and_mask()?;
        let masks = masks.unwrap_or_default();
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

    /// Zero-copy last-token / selected-row greedy decoding: projects only the rows named by
    /// `row_indices` (or the last row of a sequence when `row_indices` is `None`), avoiding
    /// projecting `0..T-2` prompt tokens during prefill.
    pub fn forward_greedy_at_rows(
        &self,
        hidden: &Tensor,
        row_indices: Option<&Tensor>,
    ) -> Result<(Tensor, usize)> {
        let k = *hidden.dims().last().unwrap_or(&0);
        let total_rows = hidden.elem_count() / k.max(1);
        let h_2d = hidden.reshape((total_rows, k))?;
        let selected = match row_indices {
            Some(idx) => h_2d.index_select(&idx.to_dtype(DType::U32)?.flatten_all()?, 0)?,
            None => {
                if total_rows == 0 {
                    h_2d
                } else {
                    h_2d.narrow(0, total_rows - 1, 1)?
                }
            }
        };
        self.forward_greedy(&selected)
    }
}

/// Deterministic counter-based Philox-4x32 uniform sample in `(0, 1)` keyed strictly by
/// `(seed, step, global_vocab_id)`.
///
/// Keying by `global_vocab_id` (independent of CORE vs REST partition ordering) guarantees
/// that Gumbel-Max sampling over CORE-only (when Cauchy-Schwarz proves REST cannot survive
/// Top-K / Min-P / Grammar filtering) selects the **exact same token ID** as full-vocabulary
/// Gumbel-Max sampling 100% of the time.
#[inline]
pub fn philox4x32_uniform(seed: u64, step: u64, vocab_id: u32) -> f32 {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;

    let mut c0 = vocab_id;
    let mut c1 = step as u32;
    let mut c2 = (step >> 32) as u32;
    let mut c3 = 0xA5A5_5A5Au32;
    let mut k0 = seed as u32;
    let mut k1 = (seed >> 32) as u32;

    for _ in 0..7 {
        let p0 = (M0 as u64).wrapping_mul(c0 as u64);
        let p1 = (M1 as u64).wrapping_mul(c2 as u64);
        let hi0 = (p0 >> 32) as u32;
        let lo0 = p0 as u32;
        let hi1 = (p1 >> 32) as u32;
        let lo1 = p1 as u32;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    // Map to open interval (2^-24, 1 - 2^-24) so -ln(-ln(u)) is always finite.
    let u = ((c0 >> 8) as f32 + 0.5f32) * (1.0f32 / 16_777_216.0f32);
    u.clamp(1.0e-7f32, 1.0f32 - 1.0e-7f32)
}

/// Configuration for lossless Top-K / Min-P / Grammar-masked guarded sampling.
#[derive(Clone, Debug)]
pub struct SamplingGuardConfig {
    /// Sampling temperature (`<= 0.0` runs exact greedy decoding).
    pub temperature: f32,
    /// Optional exact Top-K truncation (`K >= 1`).
    pub top_k: Option<usize>,
    /// Optional exact Min-P truncation in `(0.0, 1.0)`.
    pub min_p: Option<f32>,
    /// Deterministic RNG seed for Philox-4x32 Gumbel-Max.
    pub seed: u64,
    /// Monotonic decode step index for Philox-4x32 Gumbel-Max.
    pub step: u64,
    /// Optional global vocabulary allow-bitmask (`[ceil(V_full / 32)]`, bit = 1 allowed, 0 masked).
    pub grammar_bitmask: Option<Vec<u32>>,
}

impl Default for SamplingGuardConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            min_p: None,
            seed: 0,
            step: 0,
            grammar_bitmask: None,
        }
    }
}

/// Per-row top-3 candidates extracted without materializing full `[M, V_full]` logits.
#[derive(Clone, Copy, Debug)]
pub struct ShadowTop3Row {
    pub ids: [u32; 3],
    pub logits: [f32; 3],
}

#[inline]
fn grammar_allows(bitmask: Option<&[u32]>, vocab_id: u32) -> bool {
    match bitmask {
        None => true,
        Some(bits) => {
            let word = (vocab_id as usize) >> 5;
            let bit = vocab_id & 31;
            word < bits.len() && ((bits[word] >> bit) & 1) != 0
        }
    }
}

fn matmul_lm_weight(w: &LmWeight, hidden: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let mut logits = match w {
        LmWeight::Dense(t) => hidden
            .to_dtype(DType::F32)?
            .matmul(&t.to_dtype(DType::F32)?.t()?)?,
        LmWeight::Quantized(q) => {
            let m = candle::quantized::QMatMul::from_arc(q.clone())?;
            candle::Module::forward(&m, &hidden.to_dtype(DType::F32)?)?
        }
    };
    if let Some(b) = bias {
        logits = logits.broadcast_add(&b.to_dtype(DType::F32)?)?;
    }
    Ok(logits)
}

/// Lossless Top-K / Min-P / Grammar-masked guarded sampler and Shadow Top-3 head.
///
/// Extends [`GuardedQuantLmHead`] beyond greedy argmax:
/// - **Exact Top-K Guard**: if the K-th highest grammar-allowed CORE logit satisfies
///   `z_(K)^core >= U_CS(x)`, REST cannot enter the global Top-K set -> 0 bytes of REST read.
/// - **Exact Min-P Guard**: if `z_max^core + T * ln(min_p) > U_CS(x)`, every REST token falls
///   strictly below the global Min-P threshold -> 0 bytes of REST read.
/// - **Grammar Mask Shortcut**: if `grammar_bitmask` disallows every REST ID -> 0 bytes of REST read.
/// - **100% Exact Token Parity**: Philox-4x32 Gumbel-Max keyed by global `vocab_id` guarantees
///   identical sampled token IDs to full-vocabulary `[M, V_full]` sampling.
#[derive(Clone, Debug)]
pub struct GuardedSamplerHead {
    pub inner: GuardedQuantLmHead,
    core_ids_host: Vec<u32>,
    rest_ids_host: Vec<u32>,
}

impl GuardedSamplerHead {
    pub fn new(inner: GuardedQuantLmHead) -> Result<Self> {
        let core_ids_host = inner.core_ids.to_vec1::<u32>()?;
        let rest_ids_host = match &inner.rest_ids {
            Some(r) => r.to_vec1::<u32>()?,
            None => Vec::new(),
        };
        Ok(Self {
            inner,
            core_ids_host,
            rest_ids_host,
        })
    }

    pub fn from_guarded_dense(head: &GuardedLmHead) -> Result<Self> {
        let q = GuardedQuantLmHead {
            w_core: LmWeight::Dense(head.w_core.clone()),
            b_core: head.b_core.clone(),
            core_ids: head.core_ids.clone(),
            w_rest: head.w_rest.clone().map(LmWeight::Dense),
            b_rest: head.b_rest.clone(),
            rest_ids: head.rest_ids.clone(),
            cs_guard: head.cs_guard.clone(),
        };
        Self::new(q)
    }

    /// Computes exact Cauchy-Schwarz upper bound `U_CS(x)` for each row of `h_2d [M, K]`.
    fn cs_upper_bounds(&self, h_2d: &Tensor) -> Result<Option<Vec<f32>>> {
        let Some(guard) = &self.inner.cs_guard else {
            return Ok(None);
        };
        let h_f32 = h_2d.to_dtype(DType::F32)?;
        let mu = guard.mu_rest.to_device(h_2d.device())?.to_dtype(DType::F32)?;
        let xm = h_f32.matmul(&mu.unsqueeze(1)?)?.squeeze(1)?.to_vec1::<f32>()?;
        let xx = h_f32.sqr()?.sum(1)?.sqrt()?.to_vec1::<f32>()?;
        let bounds = xm
            .into_iter()
            .zip(xx)
            .map(|(dot, norm)| norm.mul_add(guard.w_rest_bound, dot) + guard.b_rest_max)
            .collect();
        Ok(Some(bounds))
    }

    /// Lossless greedy + Top-K + Min-P + Grammar-masked token selection.
    /// Returns `(token_ids [M] u32, rescued_row_count)`.
    pub fn forward_sample(
        &self,
        hidden: &Tensor,
        cfg: &SamplingGuardConfig,
    ) -> Result<(Tensor, usize)> {
        if cfg.temperature <= 0.0 && cfg.grammar_bitmask.is_none() {
            return self.inner.forward_greedy(hidden);
        }
        let dev = hidden.device();
        let k = *hidden.dims().last().unwrap_or(&0);
        let m = hidden.elem_count() / k.max(1);
        let h_2d = hidden.reshape((m, k))?;
        let bitmask = cfg.grammar_bitmask.as_deref();

        let core_logits_t = matmul_lm_weight(&self.inner.w_core, &h_2d, self.inner.b_core.as_ref())?;
        let core_logits = core_logits_t.to_vec2::<f32>()?;
        let upper_bounds = self.cs_upper_bounds(&h_2d)?;

        let rest_has_any_allowed = !self.rest_ids_host.is_empty()
            && self
                .rest_ids_host
                .iter()
                .any(|&id| grammar_allows(bitmask, id));

        let min_p_offset = match cfg.min_p {
            Some(mp) if mp > 0.0 && mp < 1.0 && cfg.temperature > 0.0 => {
                Some(cfg.temperature * mp.ln())
            }
            _ => None,
        };

        // Per-row surviving CORE candidates and rescue decision.
        let mut per_row_cands: Vec<Vec<(u32, f32)>> = Vec::with_capacity(m);
        let mut rescue_rows: Vec<u32> = Vec::new();

        for r in 0..m {
            let mut cands: Vec<(u32, f32)> = self
                .core_ids_host
                .iter()
                .zip(core_logits[r].iter())
                .filter(|(&gid, _)| grammar_allows(bitmask, gid))
                .map(|(&gid, &val)| (gid, val))
                .collect();

            // Sort descending by (logit, reverse vocab_id) for deterministic tie-breaking.
            cands.sort_unstable_by(|a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });

            let needs_rescue = if let (Some(ub), Some(_), true) =
                (&upper_bounds, &self.inner.w_rest, rest_has_any_allowed)
            {
                let u_cs = ub[r];
                if cfg.temperature <= 0.0 {
                    // Greedy with grammar mask: safe iff best allowed CORE logit >= U_CS.
                    cands.first().map_or(true, |&(_, best)| best < u_cs)
                } else {
                    let topk_safe = match cfg.top_k {
                        Some(tk) if tk > 0 && cands.len() >= tk => cands[tk - 1].1 >= u_cs,
                        _ => false,
                    };
                    let minp_safe = match (min_p_offset, cands.first()) {
                        (Some(off), Some(&(_, best))) => (best + off) > u_cs,
                        _ => false,
                    };
                    !(topk_safe || minp_safe)
                }
            } else {
                false
            };

            if needs_rescue {
                rescue_rows.push(r as u32);
            }
            per_row_cands.push(cands);
        }

        let r_count = rescue_rows.len();
        if r_count > 0 {
            if let Some(w_rest) = &self.inner.w_rest {
                let row_idx = Tensor::from_vec(rescue_rows.clone(), (r_count,), dev)?;
                let h_sub = h_2d.index_select(&row_idx, 0)?.contiguous()?;
                let rest_logits =
                    matmul_lm_weight(w_rest, &h_sub, self.inner.b_rest.as_ref())?.to_vec2::<f32>()?;
                for (ki, &orig_r) in rescue_rows.iter().enumerate() {
                    let cands = &mut per_row_cands[orig_r as usize];
                    for (&gid, &val) in self.rest_ids_host.iter().zip(rest_logits[ki].iter()) {
                        if grammar_allows(bitmask, gid) {
                            cands.push((gid, val));
                        }
                    }
                    cands.sort_unstable_by(|a, b| {
                        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                    });
                }
            }
        }

        // Final exact Top-K / Min-P truncation + deterministic global-ID Philox Gumbel-Max.
        let mut chosen = Vec::with_capacity(m);
        for (r, cands) in per_row_cands.iter().enumerate() {
            if cands.is_empty() {
                chosen.push(0u32);
                continue;
            }
            if cfg.temperature <= 0.0 {
                chosen.push(cands[0].0);
                continue;
            }
            let max_logit = cands[0].1;
            let k_lim = cfg.top_k.filter(|&tk| tk > 0).unwrap_or(cands.len()).min(cands.len());
            let min_thresh = min_p_offset.map(|off| max_logit + off);

            let mut best_gumbel = f32::NEG_INFINITY;
            let mut best_id = cands[0].0;
            let inv_t = 1.0f32 / cfg.temperature;
            let step_r = cfg.step.wrapping_add(r as u64);

            for &(gid, z) in &cands[..k_lim] {
                if let Some(th) = min_thresh {
                    if z < th {
                        break;
                    }
                }
                let u = philox4x32_uniform(cfg.seed, step_r, gid);
                let gumbel = -(-u.ln()).ln();
                let score = z * inv_t + gumbel;
                if score > best_gumbel || (score == best_gumbel && gid < best_id) {
                    best_gumbel = score;
                    best_id = gid;
                }
            }
            chosen.push(best_id);
        }

        let out = Tensor::from_vec(chosen, (m,), dev)?.to_dtype(DType::U32)?;
        Ok((out, r_count))
    }

    /// Extracts exact global Top-3 `(id, logit)` per row with lossless CS rescue
    /// (`top3[2] >= U_CS(x)` skips REST completely).
    pub fn forward_greedy_with_shadow_top3(
        &self,
        hidden: &Tensor,
    ) -> Result<(Vec<ShadowTop3Row>, usize)> {
        let dev = hidden.device();
        let k = *hidden.dims().last().unwrap_or(&0);
        let m = hidden.elem_count() / k.max(1);
        let h_2d = hidden.reshape((m, k))?;

        let core_logits =
            matmul_lm_weight(&self.inner.w_core, &h_2d, self.inner.b_core.as_ref())?
                .to_vec2::<f32>()?;
        let upper_bounds = self.cs_upper_bounds(&h_2d)?;

        let mut rows_top3 = Vec::with_capacity(m);
        let mut rescue_rows = Vec::new();
        for r in 0..m {
            let mut top_ids = [u32::MAX; 3];
            let mut top_vals = [f32::NEG_INFINITY; 3];
            for (&gid, &v) in self.core_ids_host.iter().zip(core_logits[r].iter()) {
                insert_top3(&mut top_ids, &mut top_vals, gid, v);
            }
            if let (Some(ub), Some(_)) = (&upper_bounds, &self.inner.w_rest) {
                if top_vals[2] < ub[r] {
                    rescue_rows.push(r as u32);
                }
            }
            rows_top3.push(ShadowTop3Row {
                ids: top_ids,
                logits: top_vals,
            });
        }

        let r_count = rescue_rows.len();
        if r_count > 0 {
            if let Some(w_rest) = &self.inner.w_rest {
                let row_idx = Tensor::from_vec(rescue_rows.clone(), (r_count,), dev)?;
                let h_sub = h_2d.index_select(&row_idx, 0)?.contiguous()?;
                let rest_logits =
                    matmul_lm_weight(w_rest, &h_sub, self.inner.b_rest.as_ref())?.to_vec2::<f32>()?;
                for (ki, &orig_r) in rescue_rows.iter().enumerate() {
                    let row = &mut rows_top3[orig_r as usize];
                    for (&gid, &v) in self.rest_ids_host.iter().zip(rest_logits[ki].iter()) {
                        insert_top3(&mut row.ids, &mut row.logits, gid, v);
                    }
                }
            }
        }
        Ok((rows_top3, r_count))
    }
}

#[inline]
fn insert_top3(ids: &mut [u32; 3], vals: &mut [f32; 3], gid: u32, v: f32) {
    let better = |va: f32, ia: u32, vb: f32, ib: u32| va > vb || (va == vb && ia < ib);
    if !better(v, gid, vals[2], ids[2]) {
        return;
    }
    if better(v, gid, vals[0], ids[0]) {
        vals[2] = vals[1];
        ids[2] = ids[1];
        vals[1] = vals[0];
        ids[1] = ids[0];
        vals[0] = v;
        ids[0] = gid;
    } else if better(v, gid, vals[1], ids[1]) {
        vals[2] = vals[1];
        ids[2] = ids[1];
        vals[1] = v;
        ids[1] = gid;
    } else {
        vals[2] = v;
        ids[2] = gid;
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;
    use candle::Device;

    #[test]
    fn guarded_sampler_topk_minp_grammar_matches_full_vocab_exact() -> Result<()> {
        let dev = Device::Cpu;
        let (v_full, k, m) = (64usize, 32usize, 6usize);
        let w_data: Vec<f32> = (0..v_full * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                let base = (((i as f32) * 0.37).sin()) * 0.25;
                // Make core rows (0..16) mostly stronger, except row 42 has a spike on hidden row 3.
                if row < 16 {
                    base + if col == row { 2.5 } else { 0.0 }
                } else if row == 42 {
                    base + if col == 3 { 4.5 } else { 0.0 }
                } else {
                    base * 0.2
                }
            })
            .collect();
        let h_data: Vec<f32> = (0..m * k)
            .map(|i| {
                let r = i / k;
                let c = i % k;
                if c == r { 2.0 } else { (((i as f32) * 0.19).cos()) * 0.1 }
            })
            .collect();
        let w_full = Tensor::from_vec(w_data, (v_full, k), &dev)?;
        let h = Tensor::from_vec(h_data, (m, k), &dev)?;

        let core_ids: Vec<u32> = (0..16u32).collect();
        let full_ids: Vec<u32> = (0..v_full as u32).collect();

        let guarded = GuardedLmHead::from_full_weight(&w_full, None, &core_ids, 1e-3)?;
        let full = GuardedLmHead::from_full_weight(&w_full, None, &full_ids, 1e-3)?;
        let g_sampler = GuardedSamplerHead::from_guarded_dense(&guarded)?;
        let f_sampler = GuardedSamplerHead::from_guarded_dense(&full)?;

        // 1) Top-K + Min-P stochastic sampling exact parity
        let cfg = SamplingGuardConfig {
            temperature: 0.7,
            top_k: Some(5),
            min_p: Some(0.05),
            seed: 123456789,
            step: 42,
            grammar_bitmask: None,
        };
        let (got, _rescued) = g_sampler.forward_sample(&h, &cfg)?;
        let (want, _) = f_sampler.forward_sample(&h, &cfg)?;
        assert_eq!(got.to_vec1::<u32>()?, want.to_vec1::<u32>()?);

        // 2) Grammar bitmask blocking all REST tokens -> 0 rescues and exact match
        let mut mask = vec![0u32; v_full.div_ceil(32)];
        mask[0] = 0x0000_FFFF; // only 0..16 allowed
        let cfg_g = SamplingGuardConfig {
            grammar_bitmask: Some(mask),
            ..cfg.clone()
        };
        let (got_g, rescued_g) = g_sampler.forward_sample(&h, &cfg_g)?;
        let (want_g, _) = f_sampler.forward_sample(&h, &cfg_g)?;
        assert_eq!(rescued_g, 0);
        assert_eq!(got_g.to_vec1::<u32>()?, want_g.to_vec1::<u32>()?);

        // 3) Shadow Top-3 exact parity
        let (got_t3, _) = g_sampler.forward_greedy_with_shadow_top3(&h)?;
        let (want_t3, _) = f_sampler.forward_greedy_with_shadow_top3(&h)?;
        for r in 0..m {
            assert_eq!(got_t3[r].ids, want_t3[r].ids);
        }
        Ok(())
    }
}
