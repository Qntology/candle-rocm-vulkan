//! Fused attention for CUDA, ROCm, Vulkan and the CPU.
//!
//! The FlashAttention tier (v1 / v2 / v3) is picked per GPU architecture
//! (`candle::accel::profile`), can be forced globally (`CANDLE_FLASH_ATTN_VERSION`,
//! `candle::accel::update_config`) or per call ([`FlashAttnConfig::version`]). Every entry point
//! falls back to an unfused `matmul` + `softmax` implementation when the fused path is switched
//! off (`CANDLE_FLASH_ATTN=off`, `CANDLE_FLASH_ATTN_BACKENDS`, the `flash-attn` Cargo feature)
//! or cannot run (unsupported shape, kernels not built for the GPU, Metal / OpenCL devices).
//!
//! Layouts follow `candle-flash-attn`: `q: (batch, seq_q, heads_q, head_dim)`,
//! `k`/`v: (batch, seq_kv, heads_kv, head_dim)` with `heads_q % heads_kv == 0` (GQA / MQA),
//! output `(batch, seq_q, heads_q, head_dim)`. [`flash_attn_bhsd`] takes the
//! `(batch, heads, seq, head_dim)` layout of most candle models without copies. Masks are
//! aligned to the bottom right (the query `i` sits at key position `i + seq_kv - seq_q`), as in
//! FlashAttention. K/V may be F8E4M3 (FlashAttention-3 style FP8 with descale factors).

#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
use candle::accel::attention::FaExtra;
use candle::accel::attention::{self as att, AttnGeometry, FlashAttnParams};
use candle::accel::{self, FlashAttnVersion, SplitKv};
use candle::backend::BackendStorage;
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
use candle::Storage;
use candle::{bail, CpuStorage, DType, Device, Layout, Result, Shape, Tensor, D};

/// Options of one attention call. Fields left at `None` use the global configuration.
#[derive(Debug, Clone)]
pub struct FlashAttnConfig {
    pub softmax_scale: f32,
    /// Keys before the diagonal that stay visible (`None`: all).
    pub window_size_left: Option<usize>,
    /// Keys after the diagonal that stay visible (`None`: all, `Some(0)`: causal).
    pub window_size_right: Option<usize>,
    /// Gemma style logit soft-capping.
    pub softcap: Option<f32>,
    /// ALiBi slopes, `(heads_q)` or `(batch, heads_q)`.
    pub alibi_slopes: Option<Tensor>,
    /// Factors applied to FP8 keys / values.
    pub k_descale: f32,
    pub v_descale: f32,
    /// Force a FlashAttention tier.
    pub version: Option<FlashAttnVersion>,
    /// Split-KV (flash-decoding) policy.
    pub split_kv: Option<SplitKv>,
    /// Force the fused (`true`) or unfused (`false`) path for this call.
    pub fused: Option<bool>,
    /// Return the error of the fused kernels instead of falling back to the unfused path (also
    /// when `fused == Some(true)` but the device has no fused kernels). Self-checks use it so
    /// that they test the kernels, not the fallback.
    pub strict: bool,
}

impl FlashAttnConfig {
    pub fn new(softmax_scale: f32) -> Self {
        Self {
            softmax_scale,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
            alibi_slopes: None,
            k_descale: 1.0,
            v_descale: 1.0,
            version: None,
            split_kv: None,
            fused: None,
            strict: false,
        }
    }

    /// Causal mask (`window_size_right = Some(0)`).
    pub fn causal(mut self) -> Self {
        self.window_size_right = Some(0);
        self
    }

    pub fn window(mut self, left: Option<usize>, right: Option<usize>) -> Self {
        self.window_size_left = left;
        self.window_size_right = right;
        self
    }

    pub fn softcap(mut self, softcap: f32) -> Self {
        self.softcap = (softcap > 0.0).then_some(softcap);
        self
    }

    pub fn alibi(mut self, slopes: &Tensor) -> Self {
        self.alibi_slopes = Some(slopes.clone());
        self
    }

    pub fn version(mut self, version: FlashAttnVersion) -> Self {
        self.version = Some(version);
        self
    }

    pub fn split_kv(mut self, split_kv: SplitKv) -> Self {
        self.split_kv = Some(split_kv);
        self
    }

    pub fn fused(mut self, fused: bool) -> Self {
        self.fused = Some(fused);
        self
    }

    /// Fused kernel errors propagate instead of falling back (the `strict` field).
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn descale(mut self, k: f32, v: f32) -> Self {
        self.k_descale = k;
        self.v_descale = v;
        self
    }

    fn window_i32(&self) -> (i32, i32) {
        let cv = |w: Option<usize>| w.map(|v| v.min(i32::MAX as usize) as i32).unwrap_or(-1);
        (cv(self.window_size_left), cv(self.window_size_right))
    }
}

#[derive(Debug, Clone)]
struct VarlenSeqs {
    cu_q: Tensor,
    cu_k: Tensor,
    max_q: usize,
    max_k: usize,
    batch: usize,
}

/// The fused attention operation (a [`candle::CustomOp3`] over `q`, `k`, `v`).
#[derive(Debug, Clone)]
pub struct FlashAttnOp {
    cfg: FlashAttnConfig,
    alibi: Option<Tensor>,
    varlen: Option<VarlenSeqs>,
}

fn cu_tensor(t: &Tensor, device: &Device) -> Result<Tensor> {
    t.to_device(device)?.to_dtype(DType::U32)?.contiguous()
}

impl FlashAttnOp {
    pub fn new(cfg: &FlashAttnConfig, device: &Device) -> Result<Self> {
        let alibi = match &cfg.alibi_slopes {
            Some(t) => Some(t.to_device(device)?.to_dtype(DType::F32)?.contiguous()?),
            None => None,
        };
        Ok(Self {
            cfg: cfg.clone(),
            alibi,
            varlen: None,
        })
    }

    /// Variable length batches: `seqlens_q` / `seqlens_k` are the cumulative `u32` offsets
    /// (`batch + 1` values) of the packed `(total, heads, head_dim)` tensors.
    pub fn new_varlen(
        cfg: &FlashAttnConfig,
        device: &Device,
        seqlens_q: &Tensor,
        seqlens_k: &Tensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
    ) -> Result<Self> {
        let mut op = Self::new(cfg, device)?;
        let n = seqlens_q.dims1()?;
        if n < 1 || seqlens_k.dims1()? != n {
            bail!("flash-attn varlen: seqlens_q / seqlens_k must both have batch + 1 entries")
        }
        op.varlen = Some(VarlenSeqs {
            cu_q: cu_tensor(seqlens_q, device)?,
            cu_k: cu_tensor(seqlens_k, device)?,
            max_q: max_seqlen_q,
            max_k: max_seqlen_k,
            batch: n - 1,
        });
        Ok(op)
    }

    fn geometry(&self, q_l: &Layout, k_l: &Layout, v_l: &Layout) -> Result<AttnGeometry> {
        match &self.varlen {
            Some(s) => AttnGeometry::varlen(q_l, k_l, v_l, s.batch, s.max_q, s.max_k),
            None => AttnGeometry::dense(q_l, k_l, v_l),
        }
    }

    fn params(&self, auto_version: FlashAttnVersion) -> FlashAttnParams {
        let conf = accel::config();
        let (wl, wr) = self.cfg.window_i32();
        let mut p = FlashAttnParams::new(self.cfg.softmax_scale);
        p.window_left = wl;
        p.window_right = wr;
        p.softcap = self.cfg.softcap.unwrap_or(0.0);
        p.k_descale = self.cfg.k_descale;
        p.v_descale = self.cfg.v_descale;
        p.version = self
            .cfg
            .version
            .or(conf.flash_attn.version)
            .unwrap_or(auto_version);
        p.split_kv = self.cfg.split_kv.unwrap_or(conf.flash_attn.split_kv);
        p.gqa_pack = conf.flash_attn.gqa_decode;
        p
    }

    fn alibi_bs(&self, geom: &AttnGeometry) -> Result<usize> {
        match &self.alibi {
            Some(t) => att::alibi_batch_stride(t.shape(), geom.batch, geom.hq),
            None => Ok(0),
        }
    }

    /// Whether K/V go to the kernels as they are: same dtype as the queries, FP8 when FP8 K/V
    /// are enabled, or exact F16 / BF16 K/V with F32 queries (the kernels widen them to f32,
    /// which is exact, so the results equal those of converted K/V).
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn kv_as_is(q: DType, k: DType, v: DType) -> bool {
        k == v
            && (k == q
                || (k == DType::F8E4M3 && accel::config().flash_attn.fp8_kv)
                || (q == DType::F32 && matches!(k, DType::F16 | DType::BF16)))
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fwd {
    ($self:ident, $variant:ident, $q:ident, $q_l:ident, $k:ident, $k_l:ident, $v:ident, $v_l:ident) => {{
        let qd = $q.dtype();
        let (kc, vc);
        let (k_s, k_l, v_s, v_l) = if FlashAttnOp::kv_as_is(qd, $k.dtype(), $v.dtype()) {
            ($k, $k_l.clone(), $v, $v_l.clone())
        } else {
            kc = $k.to_dtype($k_l, qd)?;
            vc = $v.to_dtype($v_l, qd)?;
            (
                &kc,
                Layout::contiguous($k_l.shape()),
                &vc,
                Layout::contiguous($v_l.shape()),
            )
        };
        let geom = $self.geometry($q_l, &k_l, &v_l)?;
        let prof = $q.device().accel_profile();
        let params = $self.params(prof.flash_attn_version);
        let alibi_g = $self.alibi.as_ref().map(|t| t.storage_and_layout());
        let cu_g = $self
            .varlen
            .as_ref()
            .map(|s| (s.cu_q.storage_and_layout(), s.cu_k.storage_and_layout()));
        let mut extra = FaExtra::default();
        if let Some((g, l)) = alibi_g.as_ref() {
            let Storage::$variant(s) = &**g else {
                bail!("flash-attn: the alibi slopes are on another device")
            };
            extra.alibi = Some((s, l.start_offset(), $self.alibi_bs(&geom)?));
        }
        if let Some(((gq, lq), (gk, lk))) = cu_g.as_ref() {
            let (Storage::$variant(sq), Storage::$variant(sk)) = (&**gq, &**gk) else {
                bail!("flash-attn: the sequence offsets are on another device")
            };
            extra.cu_seqlens = Some((sq, lq.start_offset(), sk, lk.start_offset()));
        }
        if !geom.kernel_supported() {
            bail!("flash-attn: shape not supported by the fused kernels")
        }
        let out = $q.flash_attn(k_s, v_s, &geom, &params, &extra)?;
        Ok((out, geom.out_shape()))
    }};
}

impl candle::CustomOp3 for FlashAttnOp {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        q: &CpuStorage,
        q_l: &Layout,
        k: &CpuStorage,
        k_l: &Layout,
        v: &CpuStorage,
        v_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let geom = self.geometry(q_l, k_l, v_l)?;
        let q32 = q.to_dtype(q_l, DType::F32)?;
        let k32 = k.to_dtype(k_l, DType::F32)?;
        let v32 = v.to_dtype(v_l, DType::F32)?;
        let alibi = match &self.alibi {
            Some(t) => Some(t.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };
        let alibi_bs = self.alibi_bs(&geom)?;
        let cu = match &self.varlen {
            Some(s) => Some((s.cu_q.to_vec1::<u32>()?, s.cu_k.to_vec1::<u32>()?)),
            None => None,
        };
        let inp = att::ReferenceInputs {
            q: q32.as_slice::<f32>()?,
            k: k32.as_slice::<f32>()?,
            v: v32.as_slice::<f32>()?,
            batch: geom.batch,
            lq: geom.lq,
            lk: geom.lk,
            hq: geom.hq,
            hkv: geom.hkv,
            d: geom.d,
            cu_q: cu.as_ref().map(|c| c.0.as_slice()),
            cu_k: cu.as_ref().map(|c| c.1.as_slice()),
            alibi: alibi.as_deref().map(|a| (a, alibi_bs)),
        };
        let out = att::reference_attention(&inp, &self.params(FlashAttnVersion::V2))?;
        let shape = geom.out_shape();
        let out = CpuStorage::F32(out);
        let out = match q.dtype() {
            DType::F32 => out,
            dt => out.to_dtype(&Layout::contiguous(&shape), dt)?,
        };
        Ok((out, shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        gpu_fwd!(self, Cuda, q, q_l, k, k_l, v, v_l)
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        q: &candle::RocmStorage,
        q_l: &Layout,
        k: &candle::RocmStorage,
        k_l: &Layout,
        v: &candle::RocmStorage,
        v_l: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        gpu_fwd!(self, Rocm, q, q_l, k, k_l, v, v_l)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        q: &candle::VulkanStorage,
        q_l: &Layout,
        k: &candle::VulkanStorage,
        k_l: &Layout,
        v: &candle::VulkanStorage,
        v_l: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        // The shaders compute in f32: f16 / bf16 inputs are converted, FP8 K/V stay bytes.
        let qd = q.dtype();
        let fp8 = k.dtype() == DType::F8E4M3
            && v.dtype() == DType::F8E4M3
            && accel::config().flash_attn.fp8_kv;
        let qc;
        let (q_s, q_l2) = if qd == DType::F32 {
            (q, q_l.clone())
        } else {
            qc = q.to_dtype(q_l, DType::F32)?;
            (&qc, Layout::contiguous(q_l.shape()))
        };
        let (kc, vc);
        let (k_s, k_l, v_s, v_l) = if fp8 || (k.dtype() == DType::F32 && v.dtype() == DType::F32) {
            (k, k_l.clone(), v, v_l.clone())
        } else {
            kc = k.to_dtype(k_l, DType::F32)?;
            vc = v.to_dtype(v_l, DType::F32)?;
            (
                &kc,
                Layout::contiguous(k_l.shape()),
                &vc,
                Layout::contiguous(v_l.shape()),
            )
        };
        let geom = self.geometry(&q_l2, &k_l, &v_l)?;
        let prof = q.device().accel_profile();
        let params = self.params(prof.flash_attn_version);
        let alibi_g = self.alibi.as_ref().map(|t| t.storage_and_layout());
        let cu_g = self
            .varlen
            .as_ref()
            .map(|s| (s.cu_q.storage_and_layout(), s.cu_k.storage_and_layout()));
        let mut extra = FaExtra::default();
        if let Some((g, l)) = alibi_g.as_ref() {
            let Storage::Vulkan(s) = &**g else {
                bail!("flash-attn: the alibi slopes are on another device")
            };
            extra.alibi = Some((s, l.start_offset(), self.alibi_bs(&geom)?));
        }
        if let Some(((gq, lq), (gk, lk))) = cu_g.as_ref() {
            let (Storage::Vulkan(sq), Storage::Vulkan(sk)) = (&**gq, &**gk) else {
                bail!("flash-attn: the sequence offsets are on another device")
            };
            extra.cu_seqlens = Some((sq, lq.start_offset(), sk, lk.start_offset()));
        }
        if !geom.kernel_supported() {
            bail!("flash-attn: shape not supported by the fused kernels")
        }
        let out = q_s.flash_attn(k_s, v_s, &geom, &params, &extra)?;
        let shape = geom.out_shape();
        let out = if qd == DType::F32 {
            out
        } else {
            out.to_dtype(&Layout::contiguous(&shape), qd)?
        };
        Ok((out, shape))
    }
}

static LAST_FALLBACK: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Why the last call that wanted the fused kernels ran the unfused path (or a concatenating
/// fallback of [`flash_attn_blocks`]) instead; `None` when no call fell back yet.
pub fn last_fallback_reason() -> Option<String> {
    LAST_FALLBACK
        .lock()
        .map(|g| g.clone())
        .unwrap_or_else(|p| p.into_inner().clone())
}

fn warn_fallback(err: &candle::Error) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let msg = err.to_string();
    if let Ok(mut g) = LAST_FALLBACK.lock() {
        *g = Some(msg.clone());
    }
    let first = SEEN
        .lock()
        .map(|mut g| g.get_or_insert_with(HashSet::new).insert(msg.clone()))
        .unwrap_or(false);
    if first || accel::config().flash_attn.debug {
        eprintln!("candle flash-attn: fused kernel unavailable ({msg}); using the unfused path");
    }
}

fn debug_choice(q: &Tensor, fused: bool) {
    if !accel::config().flash_attn.debug {
        return;
    }
    let p = accel::profile(q.device());
    eprintln!(
        "candle flash-attn: {} on {} (tier {}), q {:?} {:?}",
        if fused { "fused" } else { "unfused" },
        p.arch,
        accel::config()
            .flash_attn
            .version
            .unwrap_or(p.flash_attn_version),
        q.dims(),
        q.dtype()
    );
}

fn run(q: &Tensor, k: &Tensor, v: &Tensor, op: &FlashAttnOp) -> Result<Tensor> {
    let fused = op
        .cfg
        .fused
        .unwrap_or_else(|| accel::config().flash_attn_allowed(q.device()));
    let supported = matches!(
        q.device(),
        Device::Cpu | Device::Cuda(_) | Device::Rocm(_) | Device::Vulkan(_)
    );
    debug_choice(q, fused && supported);
    if fused && supported {
        match q.apply_op3_no_bwd(k, v, op) {
            Ok(t) => return Ok(t),
            Err(e) if op.cfg.strict => return Err(e),
            Err(e) => warn_fallback(&e),
        }
    } else if op.cfg.strict && op.cfg.fused == Some(true) {
        bail!(
            "flash-attn: no fused kernels for {:?} (strict mode)",
            q.device().location()
        )
    }
    unfused(q, k, v, op)
}

/// Attention with the options of `cfg`. `q: (b, sq, hq, d)`, `k`/`v: (b, skv, hkv, d)`.
pub fn flash_attn_with_config(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    run(q, k, v, &FlashAttnOp::new(cfg, q.device())?)
}

/// `softmax(Q Kᵀ · softmax_scale) V`, causal when `causal` (the `candle-flash-attn` API).
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let mut cfg = FlashAttnConfig::new(softmax_scale);
    if causal {
        cfg = cfg.causal();
    }
    flash_attn_with_config(q, k, v, &cfg)
}

/// Sliding window attention (`window_size_left=None`, `window_size_right=Some(0)` is causal).
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let cfg = FlashAttnConfig::new(softmax_scale).window(window_size_left, window_size_right);
    flash_attn_with_config(q, k, v, &cfg)
}

/// Attention in the `(batch, heads, seq, head_dim)` layout, without copies. Output in the same
/// layout.
pub fn flash_attn_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    let o = flash_attn_with_config(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        cfg,
    )?;
    o.transpose(1, 2)
}

/// One block of a paged KV cache: `k` / `v` are `(batch, kv_heads, capacity, row)` tensors of
/// which the first `rows` positions of the third axis are valid. `row` is the head dim for dense
/// blocks ([`flash_attn_blocks`]) and the packed row words for compressed blocks
/// ([`crate::turbo_quant::attention_packed_blocks`]). The keys of a call are the valid rows of
/// its blocks in order.
///
/// The paged kernels read the blocks in place when every block has a contiguous last dim, the
/// same row stride as the other blocks (separately for K and V), a head stride that is a whole
/// number of rows and a batch stride of `kv_heads` head strides, e.g. full capacity buffers
/// or their `narrow(2, 0, n)` views. Other layouts take the concatenating fallback.
#[derive(Debug, Clone, Copy)]
pub struct KvBlockRef<'a> {
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub rows: usize,
}

/// Placement of one block for the paged kernels: start offset (elements), row stride
/// (elements) and capacity (head stride in rows); `None` when the layout does not fit.
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
pub(crate) fn paged_block_layout(
    t: &Tensor,
    rows: usize,
) -> Result<Option<(usize, usize, usize)>> {
    let (b, h, cap, w) = t.dims4()?;
    let s = t.stride();
    if rows > cap || (w > 1 && s[3] != 1) {
        return Ok(None);
    }
    let row = s[2];
    if row < w || row == 0 {
        return Ok(None);
    }
    let head = if h > 1 {
        s[1]
    } else if b > 1 {
        s[0]
    } else {
        cap.max(rows) * row
    };
    if head % row != 0 || (b > 1 && h > 1 && s[0] != h * s[1]) {
        return Ok(None);
    }
    let cap_rows = head / row;
    if cap_rows < rows.max(1) || cap_rows > u32::MAX as usize {
        return Ok(None);
    }
    Ok(Some((t.layout().start_offset(), row, cap_rows)))
}

/// Checks the shapes of paged blocks against `(batch, kv_heads, _, row)` of the first one and
/// returns `(kv_heads, total rows)`.
pub(crate) fn check_blocks(
    blocks: &[KvBlockRef<'_>],
    batch: usize,
    what: &str,
) -> Result<(usize, usize)> {
    let Some(first) = blocks.first() else {
        bail!("{what}: no blocks")
    };
    let (_, hkv, _, kw) = first.k.dims4()?;
    let (_, _, _, vw) = first.v.dims4()?;
    let mut lk = 0;
    for (i, blk) in blocks.iter().enumerate() {
        let (bk, hk, ck, wk) = blk.k.dims4()?;
        let (bv, hv, cv, wv) = blk.v.dims4()?;
        if bk != batch || bv != batch || hk != hkv || hv != hkv || wk != kw || wv != vw {
            bail!(
                "{what}: block {i} k {:?} / v {:?} do not match batch {batch}, {hkv} kv heads, rows of {kw} / {vw}",
                blk.k.dims(),
                blk.v.dims()
            )
        }
        if blk.rows > ck || blk.rows > cv {
            bail!(
                "{what}: block {i} has {} rows but capacities {ck} / {cv}",
                blk.rows
            )
        }
        if blk.k.dtype() != first.k.dtype() || blk.v.dtype() != first.v.dtype() {
            bail!("{what}: block {i} has another dtype than block 0")
        }
        lk += blk.rows;
    }
    if lk == 0 {
        bail!("{what}: the blocks hold no keys")
    }
    Ok((hkv, lk))
}

/// The valid rows of the blocks concatenated on `device` (the fallback of the paged paths).
pub(crate) fn cat_blocks(blocks: &[KvBlockRef<'_>], device: &Device) -> Result<(Tensor, Tensor)> {
    let mut ks = Vec::with_capacity(blocks.len());
    let mut vs = Vec::with_capacity(blocks.len());
    for blk in blocks.iter().filter(|b| b.rows > 0) {
        ks.push(blk.k.narrow(2, 0, blk.rows)?.to_device(device)?);
        vs.push(blk.v.narrow(2, 0, blk.rows)?.to_device(device)?);
    }
    if ks.len() == 1 {
        return Ok((ks.remove(0), vs.remove(0)));
    }
    Ok((Tensor::cat(&ks, 2)?, Tensor::cat(&vs, 2)?))
}

/// Attention of `q: (batch, q_heads, seq_q, head_dim)` over the dense K/V of paged cache
/// blocks (any dtype the kernels read: the query dtype, FP8, or F16 / BF16 with F32 queries),
/// without concatenating them: decoding (`seq_q <= 4`) on CUDA / ROCm runs a row kernel over a
/// block table of up to 128 blocks. Longer queries, other devices, other layouts and more
/// blocks concatenate the valid rows and run [`flash_attn_bhsd`] (same results up to the
/// summation order). Masks, soft-capping and ALiBi of `cfg` are bottom-right aligned over the
/// concatenated keys. Output `(batch, q_heads, seq_q, head_dim)` in the query dtype.
pub fn flash_attn_blocks(
    q: &Tensor,
    blocks: &[KvBlockRef<'_>],
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    let (b, _, _, d) = q.dims4()?;
    let (_hkv, _lk) = check_blocks(blocks, b, "flash-attn blocks")?;
    if blocks[0].k.dim(3)? != d || blocks[0].v.dim(3)? != d {
        bail!(
            "flash-attn blocks: k {:?} / v {:?} do not match the head dim {d} of q",
            blocks[0].k.dims(),
            blocks[0].v.dims()
        )
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    match paged_flash_attn(q, blocks, cfg, _hkv, _lk) {
        Ok(Some(t)) => return Ok(t),
        Ok(None) => {}
        Err(e) if cfg.strict => return Err(e),
        Err(e) => warn_fallback(&e),
    }
    let (k, v) = cat_blocks(blocks, q.device())?;
    flash_attn_bhsd(q, &k, &v, cfg)
}

/// The paged row kernel, `Ok(None)` when it does not apply.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn paged_flash_attn(
    q: &Tensor,
    blocks: &[KvBlockRef<'_>],
    cfg: &FlashAttnConfig,
    hkv: usize,
    lk: usize,
) -> Result<Option<Tensor>> {
    use att::PagedKv;
    let fused = cfg
        .fused
        .unwrap_or_else(|| accel::config().flash_attn_allowed(q.device()));
    if !fused || !matches!(q.device(), Device::Cuda(_) | Device::Rocm(_)) {
        return Ok(None);
    }
    let (_, _, lq, _) = q.dims4()?;
    let kv_dtype = blocks[0].k.dtype();
    if lq > att::FA_DECODE_MAX_Q
        || blocks.len() > att::FA_MAX_PAGES
        || blocks[0].v.dtype() != kv_dtype
        || !FlashAttnOp::kv_as_is(q.dtype(), kv_dtype, kv_dtype)
        || blocks.iter().any(|x| {
            !x.k.device().same_device(q.device()) || !x.v.device().same_device(q.device())
        })
    {
        return Ok(None);
    }
    // (block, k offset, k capacity, v offset, v capacity) of the non-empty blocks
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
    let op = FlashAttnOp::new(cfg, q.device())?;
    let qt = q.transpose(1, 2)?;
    let geom = AttnGeometry::paged(qt.layout(), hkv, lk, k_row, v_row)?;
    if !geom.kernel_supported() {
        return Ok(None);
    }
    let params = op.params(accel::profile(q.device()).flash_attn_version);
    let alibi_bs = op.alibi_bs(&geom)?;
    let (qg, _) = qt.storage_and_layout();
    let guards: Vec<_> = blocks
        .iter()
        .filter(|x| x.rows > 0)
        .map(|blk| (blk.k.storage_and_layout().0, blk.v.storage_and_layout().0))
        .collect();
    let alibi_g = op.alibi.as_ref().map(|t| t.storage_and_layout());
    macro_rules! launch {
        ($variant:ident, $qs:ident) => {{
            let mut pk = Vec::with_capacity(place.len());
            for (&(rows, ko, kc, vo, vc), (kg, vg)) in place.iter().zip(guards.iter()) {
                let (Storage::$variant(ks), Storage::$variant(vs)) = (&**kg, &**vg) else {
                    bail!("flash-attn blocks: a block is on another device")
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
            let mut extra = FaExtra::default();
            if let Some((g, l)) = alibi_g.as_ref() {
                let Storage::$variant(s) = &**g else {
                    bail!("flash-attn: the alibi slopes are on another device")
                };
                extra.alibi = Some((s, l.start_offset(), alibi_bs));
            }
            Storage::$variant($qs.flash_attn_paged(&pk, &geom, &params, &extra)?)
        }};
    }
    let out = match &*qg {
        #[cfg(feature = "cuda")]
        Storage::Cuda(qs) => launch!(Cuda, qs),
        #[cfg(feature = "rocm")]
        Storage::Rocm(qs) => launch!(Rocm, qs),
        _ => return Ok(None),
    };
    let o = Tensor::from_storage(
        out,
        geom.out_shape(),
        candle::op::BackpropOp::none(),
        false,
    );
    Ok(Some(o.transpose(1, 2)?))
}

/// Variable length attention over packed sequences: `q: (total_q, hq, d)`,
/// `k`/`v: (total_k, hkv, d)`, cumulative `u32` offsets `seqlens_q` / `seqlens_k`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_with_config(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    let op = FlashAttnOp::new_varlen(
        cfg,
        q.device(),
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )?;
    run(q, k, v, &op)
}

/// FlashAttention-3 style FP8 attention: K and V are quantized to FP8 (e4m3) with per-tensor
/// scales (descale factors); with Hadamard incoherent processing (switch
/// `CANDLE_FLASH_ATTN_HADAMARD`, power of two head dims) Q and K are first rotated by the same
/// random Hadamard transform, which leaves `Q Kᵀ` unchanged and spreads outliers before the FP8
/// rounding. Returns the attention output and keeps queries in their dtype.
pub fn flash_attn_fp8(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &FlashAttnConfig) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let conf = accel::config();
    let (q, k) = if conf.flash_attn.hadamard && d.is_power_of_two() && d > 1 {
        let pi = accel::turbo::rotation_matrix(d, accel::RotationKind::Hadamard, 0x4641_3346)?;
        let pi_t = Tensor::from_vec(pi, (d, d), q.device())?.t()?;
        let rot = |x: &Tensor| -> Result<Tensor> {
            x.to_dtype(DType::F32)?
                .broadcast_matmul(&pi_t)?
                .to_dtype(x.dtype())
        };
        (rot(q)?, rot(k)?)
    } else {
        (q.clone(), k.clone())
    };
    let quant = |x: &Tensor| -> Result<(Tensor, f32)> {
        let x32 = x.to_dtype(DType::F32)?;
        let amax = x32.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
        Ok(((x32 / scale as f64)?.to_dtype(DType::F8E4M3)?, scale))
    };
    let (k8, ks) = quant(&k)?;
    let (v8, vs) = quant(v)?;
    let cfg = cfg.clone().descale(cfg.k_descale * ks, cfg.v_descale * vs);
    flash_attn_with_config(&q, &k8, &v8, &cfg)
}

// ---------------------------------------------------------------------------
// Unfused path
// ---------------------------------------------------------------------------

/// The unfused attention (`matmul`, mask, `softmax`, `matmul`) with the semantics of the fused
/// kernels; runs on every device and is the fallback of the fused path.
pub fn flash_attn_unfused(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &FlashAttnConfig,
) -> Result<Tensor> {
    unfused(q, k, v, &FlashAttnOp::new(cfg, q.device())?)
}

fn unfused(q: &Tensor, k: &Tensor, v: &Tensor, op: &FlashAttnOp) -> Result<Tensor> {
    let alibi = op.alibi.as_ref();
    match &op.varlen {
        None => unfused_dense(q, k, v, op, alibi, None),
        Some(s) => {
            let cq = s.cu_q.to_vec1::<u32>()?;
            let ck = s.cu_k.to_vec1::<u32>()?;
            let mut outs = Vec::with_capacity(s.batch);
            for b in 0..s.batch {
                let (q0, q1) = (cq[b] as usize, cq[b + 1] as usize);
                let (k0, k1) = (ck[b] as usize, ck[b + 1] as usize);
                let qb = q.narrow(0, q0, q1 - q0)?.unsqueeze(0)?;
                let kb = k.narrow(0, k0, k1 - k0)?.unsqueeze(0)?;
                let vb = v.narrow(0, k0, k1 - k0)?.unsqueeze(0)?;
                outs.push(unfused_dense(&qb, &kb, &vb, op, alibi, Some(b))?.squeeze(0)?);
            }
            Tensor::cat(&outs, 0)
        }
    }
}

fn unfused_dense(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    op: &FlashAttnOp,
    alibi: Option<&Tensor>,
    batch_index: Option<usize>,
) -> Result<Tensor> {
    let (b, lq, hq, d) = q.dims4()?;
    let (_, lk, hkv, _) = k.dims4()?;
    if hkv == 0 || hq % hkv != 0 {
        bail!("flash-attn: {hq} query heads are not a multiple of {hkv} kv heads")
    }
    let dev = q.device();
    let cfg = &op.cfg;
    let (wl, wr) = cfg.window_i32();
    let q32 = q.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
    let expand = |x: &Tensor, descale: f32| -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?.transpose(1, 2)?;
        let x = if descale != 1.0 {
            (x * descale as f64)?
        } else {
            x
        };
        let g = hq / hkv;
        if g == 1 {
            x.contiguous()
        } else {
            x.unsqueeze(2)?
                .expand((b, hkv, g, lk, d))?
                .reshape((b, hq, lk, d))
        }
    };
    let k32 = expand(k, cfg.k_descale)?;
    let v32 = expand(v, cfg.v_descale)?;
    let mut s = (q32.matmul(&k32.t()?)? * cfg.softmax_scale as f64)?;
    if let Some(cap) = cfg.softcap.filter(|c| *c > 0.0) {
        s = ((s / cap as f64)?.tanh()? * cap as f64)?;
    }
    // additive mask, ALiBi distances and row validity, built on the host
    let mut mask = vec![0f32; lq * lk];
    let mut dist = vec![0f32; lq * lk];
    let mut valid = vec![0f32; lq];
    for i in 0..lq {
        let (lo, hi) = att::key_range(i, lq, lk, wl, wr);
        valid[i] = if lo < hi { 1.0 } else { 0.0 };
        let pos = i as i64 + lk as i64 - lq as i64;
        for j in 0..lk {
            if j < lo || j >= hi {
                mask[i * lk + j] = -1e30;
            }
            dist[i * lk + j] = (pos - j as i64).abs() as f32;
        }
    }
    s = s.broadcast_add(&Tensor::from_vec(mask, (1, 1, lq, lk), dev)?)?;
    if let Some(slopes) = alibi {
        let slopes = slopes.to_dtype(DType::F32)?;
        let slopes = match (slopes.rank(), batch_index) {
            (1, _) => slopes.reshape((1, hq, 1, 1))?,
            (2, None) => slopes.reshape((b, hq, 1, 1))?,
            (2, Some(bi)) => slopes.narrow(0, bi, 1)?.reshape((1, hq, 1, 1))?,
            _ => bail!("flash-attn: alibi slopes must be (hq) or (batch, hq)"),
        };
        let dist = Tensor::from_vec(dist, (1, 1, lq, lk), dev)?;
        s = s.broadcast_sub(&slopes.broadcast_mul(&dist)?)?;
    }
    let p = crate::ops::softmax_last_dim(&s)?;
    let o = p.matmul(&v32)?;
    let o = o.broadcast_mul(&Tensor::from_vec(valid, (1, 1, lq, 1), dev)?)?;
    o.transpose(1, 2)?.contiguous()?.to_dtype(q.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let mut rng = accel::turbo::SplitMix64::new(seed);
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.next_normal() as f32).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn fused_cpu_matches_unfused() {
        let q = rand(&[2, 5, 4, 32], 1);
        let k = rand(&[2, 9, 2, 32], 2);
        let v = rand(&[2, 9, 2, 32], 3);
        let slopes = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &Device::Cpu).unwrap();
        for cfg in [
            FlashAttnConfig::new(0.2).causal(),
            FlashAttnConfig::new(0.2),
            FlashAttnConfig::new(0.2)
                .window(Some(3), Some(0))
                .softcap(4.0),
            FlashAttnConfig::new(0.2).causal().alibi(&slopes),
        ] {
            let a = flash_attn_with_config(&q, &k, &v, &cfg.clone().fused(true)).unwrap();
            let b = flash_attn_unfused(&q, &k, &v, &cfg).unwrap();
            assert!(max_diff(&a, &b) < 1e-5, "{cfg:?}");
        }
    }

    #[test]
    fn varlen_matches_dense() {
        let q = rand(&[7, 2, 16], 4);
        let k = rand(&[9, 2, 16], 5);
        let v = rand(&[9, 2, 16], 6);
        let cq = Tensor::new(&[0u32, 3, 7], &Device::Cpu).unwrap();
        let ck = Tensor::new(&[0u32, 4, 9], &Device::Cpu).unwrap();
        let cfg = FlashAttnConfig::new(0.25).causal();
        let fused =
            flash_attn_varlen_with_config(&q, &k, &v, &cq, &ck, 4, 5, &cfg.clone().fused(true))
                .unwrap();
        let unf =
            flash_attn_varlen_with_config(&q, &k, &v, &cq, &ck, 4, 5, &cfg.clone().fused(false))
                .unwrap();
        assert!(max_diff(&fused, &unf) < 1e-5);
        // the second sequence alone
        let q1 = q.narrow(0, 3, 4).unwrap().unsqueeze(0).unwrap();
        let k1 = k.narrow(0, 4, 5).unwrap().unsqueeze(0).unwrap();
        let v1 = v.narrow(0, 4, 5).unwrap().unsqueeze(0).unwrap();
        let dense = flash_attn_with_config(&q1, &k1, &v1, &cfg)
            .unwrap()
            .squeeze(0)
            .unwrap();
        assert!(max_diff(&fused.narrow(0, 3, 4).unwrap(), &dense) < 1e-5);
    }

    #[test]
    fn fp8_attention() {
        let q = rand(&[1, 24, 2, 64], 7);
        let k = rand(&[1, 24, 2, 64], 8);
        let v = rand(&[1, 24, 2, 64], 9);
        let cfg = FlashAttnConfig::new(0.125).causal();
        // fused and unfused agree on the same FP8 inputs
        let fused = flash_attn_fp8(&q, &k, &v, &cfg.clone().fused(true)).unwrap();
        let unfused = flash_attn_fp8(&q, &k, &v, &cfg.clone().fused(false)).unwrap();
        assert!(max_diff(&fused, &unfused) < 1e-5);
        // e4m3 keeps 3 mantissa bits: a few percent of relative error
        let exact = flash_attn_with_config(&q, &k, &v, &cfg).unwrap();
        let err = (&fused - &exact)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let nrm = exact
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (err / nrm).sqrt() < 0.06,
            "relative rms error {}",
            (err / nrm).sqrt()
        );
    }
}
