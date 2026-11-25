//! Part 5 Vision-Decode optimizations for multimodal LLM/VLM serving:
//!
//! 1. [`VisMicroPageIndex`]: `P_v = 16` micro-page bounding-box `(k_min, k_max)` index with exact
//!    lossless upper-bound pruning `U_p(q) = sum_d max(q_d * k_min[p,d], q_d * k_max[p,d]) * scale`.
//!    Skipping pages where `U_p(q) < m_anchor - tau` (`tau >= 16.0 => exp(-16) < 1.2e-7`) preserves
//!    exact attention output within FP16/BF16 ULP while skipping background visual tiles.
//! 2. [`HeteroMultimodalKvCache`] + [`WindowedZeroRetention`]: separates unquantized attention-sink
//!    tokens, vision-prefix KV segments, and text-decode KV segments, and reclaims expired vision
//!    KV pages immediately once sliding-window generation passes `vis_end + window_left`.
//! 3. [`DeltaRope`]: exact Cody-Waite TwoProd phase un-rotation (`pos_cur - pos_cached`) enabling
//!    cross-turn Vision KV prefix reuse without re-encoding or re-rotating cached keys.
//! 4. [`VisOcrNgramDraft`]: bounded (<= 256 KiB) zero-weight suffix / N-gram speculative drafter
//!    tailored for structured OCR, CAD/BOM tables, and JSON schema continuations.

use candle::{bail, DType, Device, Result, Tensor};
use std::collections::{HashMap, VecDeque};

/// Default visual micro-page size (`P_v = 16` tokens per bounding box).
pub const VIS_MICRO_PAGE_SIZE: usize = 16;

/// Default lossless pruning threshold (`exp(-16) = 1.125e-7`, strictly below FP16/BF16 ULP).
pub const DEFAULT_LOSSLESS_TAU: f32 = 16.0;

/// Per-head micro-page bounding-box index for lossless dynamic sparse vision attention.
///
/// Stores `(k_min, k_max)` of shape `[num_heads, num_pages, head_dim]` in FP32. For any query
/// `q[h, d]`, the dot product with every key `k` in page `p` satisfies:
/// `q . k <= sum_d max(q_d * k_min[h, p, d], q_d * k_max[h, p, d])`.
#[derive(Debug, Clone)]
pub struct VisMicroPageIndex {
    pub num_heads: usize,
    pub num_pages: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub num_tokens: usize,
    k_min: Vec<f32>,
    k_max: Vec<f32>,
}

impl VisMicroPageIndex {
    /// Builds a micro-page bounding-box index from visual keys `k` of shape `[Hkv, L_vis, D]`
    /// (or `[1, Hkv, L_vis, D]`).
    pub fn build(k: &Tensor, page_size: usize) -> Result<Self> {
        let page_size = page_size.max(1);
        let k3 = match k.dims() {
            [h, l, d] => k.reshape((*h, *l, *d))?,
            [1, h, l, d] => k.reshape((*h, *l, *d))?,
            dims => bail!("VisMicroPageIndex::build: expected [H, L, D] or [1, H, L, D], got {dims:?}"),
        };
        let (hkv, l_vis, d) = k3.dims3()?;
        let num_pages = l_vis.div_ceil(page_size);
        let data = k3
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut k_min = vec![f32::INFINITY; hkv * num_pages * d];
        let mut k_max = vec![f32::NEG_INFINITY; hkv * num_pages * d];
        for h in 0..hkv {
            for p in 0..num_pages {
                let t0 = p * page_size;
                let t1 = (t0 + page_size).min(l_vis);
                let dst = (h * num_pages + p) * d;
                for t in t0..t1 {
                    let src = (h * l_vis + t) * d;
                    for j in 0..d {
                        let v = data[src + j];
                        if v < k_min[dst + j] {
                            k_min[dst + j] = v;
                        }
                        if v > k_max[dst + j] {
                            k_max[dst + j] = v;
                        }
                    }
                }
            }
        }
        Ok(Self {
            num_heads: hkv,
            num_pages,
            head_dim: d,
            page_size,
            num_tokens: l_vis,
            k_min,
            k_max,
        })
    }

    /// Computes exact upper-bound logits `U_p(q)` for all pages `0..num_pages` of key head `kv_head`.
    pub fn upper_bounds(&self, kv_head: usize, q: &[f32], scale: f32) -> Vec<f32> {
        debug_assert_eq!(q.len(), self.head_dim);
        let mut out = vec![0.0f32; self.num_pages];
        let d = self.head_dim;
        for p in 0..self.num_pages {
            let base = (kv_head * self.num_pages + p) * d;
            let mut acc = 0.0f32;
            for j in 0..d {
                let qj = q[j];
                let lo = qj * self.k_min[base + j];
                let hi = qj * self.k_max[base + j];
                acc += lo.max(hi);
            }
            out[p] = acc * scale;
        }
        out
    }

    /// Returns the sorted indices of visual micro-pages that cannot be provably pruned under
    /// anchor max logit `m_anchor` and lossless safety margin `tau` (`U_p(q) >= m_anchor - tau`).
    ///
    /// When `m_anchor` is `None`, uses the maximum upper bound across pages minus `0.0` only as
    /// a conservative lower bound? No — an upper bound cannot substitute for a true lower bound!
    /// Therefore `m_anchor` MUST be a true achieved logit (e.g. from sink/text tokens or the
    /// exact evaluation of the highest-upper-bound page). Passing `None` keeps all pages.
    pub fn active_pages(
        &self,
        kv_head: usize,
        q: &[f32],
        scale: f32,
        m_anchor: Option<f32>,
        tau: f32,
    ) -> Vec<usize> {
        let Some(anchor) = m_anchor else {
            return (0..self.num_pages).collect();
        };
        let threshold = anchor - tau.max(0.0);
        let ubs = self.upper_bounds(kv_head, q, scale);
        ubs.into_iter()
            .enumerate()
            .filter_map(|(p, u)| (u >= threshold).then_some(p))
            .collect()
    }

    /// Converts active micro-page indices into contiguous token `[start, end)` spans.
    pub fn page_token_range(&self, page: usize) -> (usize, usize) {
        let start = page * self.page_size;
        let end = (start + self.page_size).min(self.num_tokens);
        (start, end)
    }

    /// Returns device/CPU `(k_min, k_max)` tensors `[num_heads, num_pages, head_dim]` for GPU
    /// upper-bound kernels (`tq_vis_page_upper_bound`).
    pub fn bounds_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let shape = (self.num_heads, self.num_pages, self.head_dim);
        Ok((
            Tensor::from_slice(&self.k_min, shape, device)?,
            Tensor::from_slice(&self.k_max, shape, device)?,
        ))
    }
}

/// Heterogeneous Multimodal KV Cache policy & container:
/// - **Attention Sinks** (`0..sink_len`): always preserved in unquantized full/half precision.
/// - **Vision Prefix Segment** (`[vis_start, vis_end)`): independently stored and optionally
///   indexed with [`VisMicroPageIndex`] for lossless sparse vision decode and reclaimable via
///   [`Self::evict_expired_vision`] on sliding-window layers (`WindowedZeroRetention`).
/// - **Text Segment**: appended during prompt/decode steps.
#[derive(Debug, Clone)]
pub struct HeteroMultimodalKvCache {
    pub sink_len: usize,
    pub window_left: Option<usize>,
    sink_kv: Option<(Tensor, Tensor)>,
    vision_kv: Option<(Tensor, Tensor)>,
    vision_range: Option<(usize, usize)>,
    vision_index: Option<VisMicroPageIndex>,
    text_kv: Option<(Tensor, Tensor)>,
    total_seq_len: usize,
    evicted_vision_tokens: usize,
}

impl HeteroMultimodalKvCache {
    pub fn new(sink_len: usize, window_left: Option<usize>) -> Self {
        Self {
            sink_len,
            window_left,
            sink_kv: None,
            vision_kv: None,
            vision_range: None,
            vision_index: None,
            text_kv: None,
            total_seq_len: 0,
            evicted_vision_tokens: 0,
        }
    }

    pub fn total_seq_len(&self) -> usize {
        self.total_seq_len
    }

    pub fn has_vision_kv(&self) -> bool {
        self.vision_kv.is_some()
    }

    pub fn evicted_vision_tokens(&self) -> usize {
        self.evicted_vision_tokens
    }

    pub fn vision_index(&self) -> Option<&VisMicroPageIndex> {
        self.vision_index.as_ref()
    }

    /// Registers the unquantized attention-sink KV tensors `[B, Hkv, L_sink, D]`.
    pub fn set_sink(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        let l = k.dim(2)?;
        self.sink_kv = Some((k, v));
        self.total_seq_len = self.total_seq_len.max(l);
        Ok(())
    }

    /// Registers the vision-prefix KV tensors `[B, Hkv, L_vis, D]` covering global positions
    /// `[vis_start, vis_start + L_vis)` and optionally builds a [`VisMicroPageIndex`].
    pub fn set_vision(
        &mut self,
        vis_start: usize,
        k: Tensor,
        v: Tensor,
        build_micro_index: bool,
    ) -> Result<()> {
        let l = k.dim(2)?;
        let idx = if build_micro_index && k.dim(0)? == 1 && l > 0 {
            Some(VisMicroPageIndex::build(&k, VIS_MICRO_PAGE_SIZE)?)
        } else {
            None
        };
        self.vision_kv = Some((k, v));
        self.vision_range = Some((vis_start, vis_start + l));
        self.vision_index = idx;
        self.total_seq_len = self.total_seq_len.max(vis_start + l);
        Ok(())
    }

    /// Appends text KV tensors `[B, Hkv, L_new, D]` along the sequence dimension (`dim = 2`).
    pub fn append_text(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let l_new = k.dim(2)?;
        self.text_kv = match self.text_kv.take() {
            None => Some((k.clone(), v.clone())),
            Some((pk, pv)) => Some((Tensor::cat(&[&pk, k], 2)?, Tensor::cat(&[&pv, v], 2)?)),
        };
        self.total_seq_len += l_new;
        Ok(())
    }

    /// **WindowedZeroRetention**: drops the vision KV buffers and micro-page index immediately
    /// once `current_pos >= vis_end + window_left` on a sliding-window layer, reclaiming VRAM
    /// without waiting for generation end. Returns the number of reclaimed visual tokens.
    pub fn evict_expired_vision(&mut self, current_pos: usize) -> usize {
        let (Some(wl), Some((vis_start, vis_end))) = (self.window_left, self.vision_range) else {
            return 0;
        };
        if self.vision_kv.is_some() && current_pos >= vis_end.saturating_add(wl) {
            let freed = vis_end.saturating_sub(vis_start);
            self.vision_kv = None;
            self.vision_index = None;
            self.evicted_vision_tokens += freed;
            return freed;
        }
        0
    }

    /// Materializes concatenated `(K, V)` across active segments `[sink ; vision ; text]` for
    /// standard dense attention calls.
    pub fn materialized_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        let mut ks = Vec::with_capacity(3);
        let mut vs = Vec::with_capacity(3);
        if let Some((k, v)) = &self.sink_kv {
            ks.push(k);
            vs.push(v);
        }
        if let Some((k, v)) = &self.vision_kv {
            ks.push(k);
            vs.push(v);
        }
        if let Some((k, v)) = &self.text_kv {
            ks.push(k);
            vs.push(v);
        }
        if ks.is_empty() {
            return Ok(None);
        }
        if ks.len() == 1 {
            return Ok(Some((ks[0].clone(), vs[0].clone())));
        }
        Ok(Some((Tensor::cat(&ks, 2)?, Tensor::cat(&vs, 2)?)))
    }
}

/// Relative phase un-rotation (`Delta-RoPE`) using Cody-Waite TwoProd FP32 range reduction.
///
/// Enables cross-turn reuse of cached Vision KV tensors without re-encoding or mutating cached
/// keys: shifting a cached block by `delta_pos = pos_cur - pos_cached` is mathematically
/// equivalent to rotating queries by `-delta_pos` (`R(-delta_pos) q . k_cached == q . R(delta_pos) k_cached`).
pub struct DeltaRope;

impl DeltaRope {
    /// Applies NeoX-style (split-half) Delta-RoPE phase rotation by `delta_pos` (`pos_cur - pos_cached`)
    /// to query tensor `q` of shape `[..., D]` (last dim even, `rope_dim <= D`).
    pub fn apply_query_delta(
        q: &Tensor,
        delta_pos: i64,
        rope_dim: usize,
        rope_theta: f64,
        mscale: f32,
    ) -> Result<Tensor> {
        if delta_pos == 0 || rope_dim == 0 {
            return Ok(q.clone());
        }
        if rope_dim % 2 != 0 {
            bail!("DeltaRope: rope_dim must be even, got {rope_dim}");
        }
        let d = q.dim(candle::D::Minus1)?;
        if rope_dim > d {
            bail!("DeltaRope: rope_dim {rope_dim} exceeds last dim {d}");
        }
        let dtype = q.dtype();
        let dev = q.device().clone();
        let shape = q.shape().clone();
        let rows = q.elem_count() / d;
        let mut data = q
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let half = rope_dim / 2;
        let mut cos_v = vec![0.0f32; half];
        let mut sin_v = vec![0.0f32; half];
        // Rotate Q by -delta_pos so `Q' . K_cached == Q . (R(+delta_pos) K_cached)`.
        let abs_delta = delta_pos.unsigned_abs();
        let neg_angle = delta_pos > 0;
        for j in 0..half {
            let inv_freq = 1.0f64 / rope_theta.powf((2 * j) as f64 / rope_dim as f64);
            let (c, s_abs) = if abs_delta <= u32::MAX as u64 {
                candle::accel::decoder_ops::cody_waite_sincos_f32(abs_delta as u32, inv_freq as f32)
            } else {
                let theta = (abs_delta as f64) * inv_freq;
                (theta.cos() as f32, theta.sin() as f32)
            };
            let s = if neg_angle { -s_abs } else { s_abs };
            cos_v[j] = c * mscale;
            sin_v[j] = s * mscale;
        }
        for r in 0..rows {
            let row = &mut data[r * d..(r + 1) * d];
            for j in 0..half {
                let x0 = row[j];
                let x1 = row[j + half];
                let c = cos_v[j];
                let s = sin_v[j];
                row[j] = x0 * c - x1 * s;
                row[j + half] = x1 * c + x0 * s;
            }
        }
        Tensor::from_vec(data, shape, &dev)?.to_dtype(dtype)
    }
}

/// Zero-weight bounded (~256 KiB) suffix / N-gram speculative drafter for structured Vision-OCR,
/// BOM tables, CAD specs, and JSON outputs.
#[derive(Debug, Clone)]
pub struct VisOcrNgramDraft {
    max_n: usize,
    max_entries: usize,
    history: VecDeque<u32>,
    max_history: usize,
    /// Maps n-gram key (encoded as up to 4 u32 tokens -> 64-bit hash + length) to most recent
    /// continuation slice of up to `draft_len` tokens.
    table: HashMap<u64, Vec<u32>>,
    order: VecDeque<u64>,
    max_draft_len: usize,
}

impl VisOcrNgramDraft {
    /// Creates a bounded N-gram drafter (`max_n` clamped to `2..=6`, default budget ~256 KiB).
    pub fn new(max_n: usize, max_draft_len: usize) -> Self {
        Self {
            max_n: max_n.clamp(2, 6),
            max_entries: 4096,
            history: VecDeque::with_capacity(4096),
            max_history: 4096,
            table: HashMap::with_capacity(1024),
            order: VecDeque::with_capacity(4096),
            max_draft_len: max_draft_len.clamp(1, 16),
        }
    }

    fn hash_ngram(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (tokens.len() as u64);
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Ingests prompt or generated tokens into the N-gram suffix index in O(T * max_n).
    pub fn ingest(&mut self, tokens: &[u32]) {
        for &tok in tokens {
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history.push_back(tok);
            let hlen = self.history.len();
            // Update continuation entries for suffix keys ending `1..=max_draft_len` tokens back.
            let max_back = self.max_draft_len.min(hlen.saturating_sub(2));
            for back in 1..=max_back {
                let cont_start = hlen - back;
                for n in 2..=self.max_n {
                    if cont_start < n {
                        break;
                    }
                    let key_slice: Vec<u32> = self
                        .history
                        .range((cont_start - n)..cont_start)
                        .copied()
                        .collect();
                    let key = Self::hash_ngram(&key_slice);
                    let cont: Vec<u32> = self.history.range(cont_start..hlen).copied().collect();
                    if !self.table.contains_key(&key) {
                        if self.order.len() >= self.max_entries {
                            if let Some(evict) = self.order.pop_front() {
                                self.table.remove(&evict);
                            }
                        }
                        self.order.push_back(key);
                    }
                    self.table.insert(key, cont);
                }
            }
        }
    }

    /// Proposes up to `k` draft continuation tokens from the longest matching suffix in `context`
    /// (falling back to internal history when `context` is empty).
    pub fn draft(&self, context: &[u32], k: usize) -> Vec<u32> {
        let k = k.min(self.max_draft_len);
        if k == 0 {
            return Vec::new();
        }
        let ctx_len = context.len();
        for n in (2..=self.max_n).rev() {
            if ctx_len >= n {
                let key = Self::hash_ngram(&context[ctx_len - n..]);
                if let Some(cont) = self.table.get(&key) {
                    return cont.iter().copied().take(k).collect();
                }
            }
        }
        Vec::new()
    }

    pub fn len_entries(&self) -> usize {
        self.table.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vis_micro_page_lossless_pruning_matches_dense_attention() -> Result<()> {
        let dev = Device::Cpu;
        let (hkv, l_vis, d) = (2usize, 64usize, 32usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        // Page 0 (tokens 0..16) aligns strongly with query; pages 1..4 are strongly anti-aligned
        // so their exact upper bound falls far below `m_anchor - 16.0`.
        let mut k_data = vec![0.0f32; hkv * l_vis * d];
        let mut v_data = vec![0.0f32; hkv * l_vis * d];
        let mut q_data = vec![0.0f32; hkv * d];
        for h in 0..hkv {
            for j in 0..d {
                q_data[h * d + j] = 2.0;
            }
            for t in 0..l_vis {
                let val = if t < 16 { 1.5 } else { -1.5 };
                for j in 0..d {
                    k_data[(h * l_vis + t) * d + j] = val + ((t % 5) as f32) * 0.01;
                    v_data[(h * l_vis + t) * d + j] = ((t + j + h) % 13) as f32 * 0.1;
                }
            }
        }
        let k = Tensor::from_vec(k_data.clone(), (hkv, l_vis, d), &dev)?;
        let idx = VisMicroPageIndex::build(&k, VIS_MICRO_PAGE_SIZE)?;
        assert_eq!(idx.num_pages, 4);

        for h in 0..hkv {
            let q = &q_data[h * d..(h + 1) * d];
            // Dense exact attention for head h.
            let mut logits = vec![0.0f32; l_vis];
            let mut m_all = f32::NEG_INFINITY;
            for t in 0..l_vis {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += q[j] * k_data[(h * l_vis + t) * d + j];
                }
                logits[t] = dot * scale;
                m_all = m_all.max(logits[t]);
            }
            let mut denom = 0.0f32;
            let mut dense_out = vec![0.0f32; d];
            for t in 0..l_vis {
                let w = (logits[t] - m_all).exp();
                denom += w;
                for j in 0..d {
                    dense_out[j] += w * v_data[(h * l_vis + t) * d + j];
                }
            }
            for j in 0..d {
                dense_out[j] /= denom;
            }

            // Evaluate highest-UB page first to get a valid lower bound `m_anchor`.
            let ubs = idx.upper_bounds(h, q, scale);
            let best_page = ubs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let (s0, e0) = idx.page_token_range(best_page);
            let m_anchor = logits[s0..e0].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let active = idx.active_pages(h, q, scale, Some(m_anchor), DEFAULT_LOSSLESS_TAU);
            assert_eq!(active, vec![0], "background pages should be pruned losslessly");

            // Sparse attention over active pages only.
            let mut sp_denom = 0.0f32;
            let mut sp_out = vec![0.0f32; d];
            for &p in &active {
                let (ps, pe) = idx.page_token_range(p);
                for t in ps..pe {
                    let w = (logits[t] - m_anchor).exp();
                    sp_denom += w;
                    for j in 0..d {
                        sp_out[j] += w * v_data[(h * l_vis + t) * d + j];
                    }
                }
            }
            for j in 0..d {
                sp_out[j] /= sp_denom;
                assert!(
                    (sp_out[j] - dense_out[j]).abs() < 1e-6,
                    "lossless sparse diff at j={j}: {} vs {}",
                    sp_out[j],
                    dense_out[j]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn hetero_kv_windowed_zero_retention_evicts_expired_vision() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = HeteroMultimodalKvCache::new(4, Some(32));
        let sink_k = Tensor::zeros((1, 2, 4, 16), DType::F32, &dev)?;
        let sink_v = Tensor::zeros((1, 2, 4, 16), DType::F32, &dev)?;
        cache.set_sink(sink_k, sink_v)?;

        let vis_k = Tensor::ones((1, 2, 24, 16), DType::F32, &dev)?;
        let vis_v = Tensor::ones((1, 2, 24, 16), DType::F32, &dev)?;
        cache.set_vision(4, vis_k, vis_v, true)?;
        assert!(cache.has_vision_kv());
        assert_eq!(cache.evict_expired_vision(59), 0); // vis_end=28, 28+32=60
        assert!(cache.has_vision_kv());
        assert_eq!(cache.evict_expired_vision(60), 24);
        assert!(!cache.has_vision_kv());
        assert_eq!(cache.evicted_vision_tokens(), 24);
        Ok(())
    }

    #[test]
    fn delta_rope_matches_direct_key_shift() -> Result<()> {
        let dev = Device::Cpu;
        let d = 16usize;
        let theta = 10000.0;
        let q = Tensor::from_vec(
            (0..d).map(|i| (i as f32 + 1.0) * 0.1).collect::<Vec<_>>(),
            (1, d),
            &dev,
        )?;
        let k = Tensor::from_vec(
            (0..d).map(|i| (i as f32 - 5.0) * 0.15).collect::<Vec<_>>(),
            (1, d),
            &dev,
        )?;
        // Check Q_delta(-delta) . K == Q . K_shifted(+delta)
        let delta = 1234i64;
        let q_rot = DeltaRope::apply_query_delta(&q, delta, d, theta, 1.0)?;
        let k_rot = DeltaRope::apply_query_delta(&k, -delta, d, theta, 1.0)?;
        let dot_lhs = (&q_rot * &k)?.sum_all()?.to_scalar::<f32>()?;
        let dot_rhs = (&q * &k_rot)?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            (dot_lhs - dot_rhs).abs() < 1e-4,
            "DeltaRope orthogonality mismatch: {dot_lhs} vs {dot_rhs}"
        );
        Ok(())
    }

    #[test]
    fn vis_ocr_ngram_drafter_proposes_repeated_schema_tokens() {
        let mut drafter = VisOcrNgramDraft::new(4, 6);
        let prompt = vec![10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60];
        drafter.ingest(&prompt);
        let proposal = drafter.draft(&[10, 20, 30], 4);
        assert_eq!(proposal, vec![40, 50, 60]);
    }
}
