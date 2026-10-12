//! Drop-in replacement of the crates.io `candle-flash-attn` 0.11 for the CUDA, ROCm, Vulkan and
//! CPU backends of `candle-rocm-vulkan`.
//!
//! Same functions, arguments and `FlashAttn` op as the crates.io crate (which only runs the
//! Dao-AILab FlashAttention-2 CUDA kernels). Here the kernels come from
//! `candle_nn::flash_attn`: FlashAttention v1 / v2 / v3 tiers chosen per GPU architecture, FP8
//! K/V, ALiBi, soft-capping, sliding windows, GQA, varlen and paged KV, with an unfused fallback
//! for devices or shapes the fused kernels do not cover.
//!
//! Use it from an application through `[patch.crates-io]`:
//!
//! ```toml
//! [dependencies]
//! candle-flash-attn = { version = "0.11.0", optional = true }
//!
//! [features]
//! flash-attn = ["dep:candle-flash-attn"]
//!
//! [patch.crates-io]
//! candle-core = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
//! candle-nn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
//! candle-flash-attn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
//! ```
//!
//! The backend comes from the `cuda` / `rocm` / `vulkan` features of `candle-nn` (or of this
//! crate). Switches: `candle::accel::config()` / the `CANDLE_FLASH_ATTN*` variables.

use candle::{bail, CpuStorage, DType, Device, Layout, Result, Shape, Tensor};
use candle_nn::flash_attn::{flash_attn_varlen_with_config, flash_attn_with_config};

pub use candle::accel::{
    self as accel, config, describe, set_config, update_config, AccelConfig, FlashAttnVersion,
    SplitKv,
};
pub use candle_nn::flash_attn::{
    flash_attn_bhsd, flash_attn_fp8, flash_attn_unfused, FlashAttnConfig, FlashAttnOp,
};

/// The fused attention op with the fields of the crates.io crate. Prefer the functions below:
/// they fall back to the unfused path when the fused kernels cannot run.
pub struct FlashAttn {
    pub softmax_scale: f32,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

impl FlashAttn {
    fn config(&self) -> FlashAttnConfig {
        let mut c = FlashAttnConfig::new(self.softmax_scale)
            .window(self.window_size_left, self.window_size_right);
        c.softcap = self.softcap.filter(|v| *v > 0.0);
        c.alibi_slopes = self.alibi_slopes.clone();
        c
    }

    fn op(&self, device: &Device) -> Result<FlashAttnOp> {
        FlashAttnOp::new(&self.config(), device)
    }
}

impl candle::CustomOp3 for FlashAttn {
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
        self.op(&Device::Cpu)?.cpu_fwd(q, q_l, k, k_l, v, v_l)
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        let dev = Device::Cuda(q.device().clone());
        self.op(&dev)?.cuda_fwd(q, q_l, k, k_l, v, v_l)
    }

    fn rocm_fwd(
        &self,
        q: &candle::RocmStorage,
        q_l: &Layout,
        k: &candle::RocmStorage,
        k_l: &Layout,
        v: &candle::RocmStorage,
        v_l: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        use candle::backend::BackendStorage;
        let dev = Device::Rocm(q.device().clone());
        self.op(&dev)?.rocm_fwd(q, q_l, k, k_l, v, v_l)
    }

    fn vulkan_fwd(
        &self,
        q: &candle::VulkanStorage,
        q_l: &Layout,
        k: &candle::VulkanStorage,
        k_l: &Layout,
        v: &candle::VulkanStorage,
        v_l: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        use candle::backend::BackendStorage;
        let dev = Device::Vulkan(q.device().clone());
        self.op(&dev)?.vulkan_fwd(q, q_l, k, k_l, v, v_l)
    }
}

fn causal_window(causal: bool) -> (Option<usize>, Option<usize>) {
    (None, if causal { Some(0) } else { None })
}

fn run(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    alibi_slopes: Option<&Tensor>,
    window: (Option<usize>, Option<usize>),
    softcap: Option<f32>,
) -> Result<Tensor> {
    let mut cfg = FlashAttnConfig::new(softmax_scale).window(window.0, window.1);
    cfg.alibi_slopes = alibi_slopes.cloned();
    cfg.softcap = softcap.filter(|v| *v > 0.0);
    flash_attn_with_config(q, k, v, &cfg)
}

/// Flash attention, `softmax(Q @ K^T . softmax_scale) @ V`.
///
/// * `q` - `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k`, `v` - `(batch, seq_len_kv, num_heads_kv, head_size)`, `num_heads_q` a multiple of
///   `num_heads_kv` (MQA / GQA).
///
/// The result has the shape of `q`.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    run(q, k, v, softmax_scale, None, causal_window(causal), None)
}

/// Flash attention with a sliding window (`None` = unbounded; `window_size_left = None`,
/// `window_size_right = Some(0)` is causal).
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    run(
        q,
        k,
        v,
        softmax_scale,
        None,
        (window_size_left, window_size_right),
        None,
    )
}

/// Flash attention with ALiBi slopes `(num_heads_q)` or `(batch, num_heads_q)`.
pub fn flash_attn_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    run(
        q,
        k,
        v,
        softmax_scale,
        Some(alibi_slopes),
        causal_window(causal),
        None,
    )
}

/// Flash attention with ALiBi slopes and a sliding window.
pub fn flash_attn_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    run(
        q,
        k,
        v,
        softmax_scale,
        Some(alibi_slopes),
        (window_size_left, window_size_right),
        None,
    )
}

/// Flash attention with optional ALiBi slopes, a sliding window and Gemma style soft-capping.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    run(
        q,
        k,
        v,
        softmax_scale,
        alibi_slopes,
        (window_size_left, window_size_right),
        Some(softcap),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    alibi_slopes: Option<&Tensor>,
    window: (Option<usize>, Option<usize>),
    softcap: Option<f32>,
) -> Result<Tensor> {
    let mut cfg = FlashAttnConfig::new(softmax_scale).window(window.0, window.1);
    cfg.alibi_slopes = alibi_slopes.cloned();
    cfg.softcap = softcap.filter(|v| *v > 0.0);
    flash_attn_varlen_with_config(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        &cfg,
    )
}

/// Variable length flash attention.
///
/// * `q` - `(total_q, num_heads_q, head_size)`; `k`, `v` - `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q`, `seqlens_k` - cumulative `u32` sequence offsets, `batch + 1` entries.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    run_varlen(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        None,
        causal_window(causal),
        None,
    )
}

/// Variable length flash attention with a sliding window.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    run_varlen(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        None,
        (window_size_left, window_size_right),
        None,
    )
}

fn int_rows(t: &Tensor) -> Result<Vec<Vec<usize>>> {
    let t = t.to_device(&Device::Cpu)?.to_dtype(DType::I64)?;
    Ok(t.to_vec2::<i64>()?
        .into_iter()
        .map(|r| r.into_iter().map(|v| v.max(0) as usize).collect())
        .collect())
}

/// Variable length flash attention over a paged KV cache: `k`, `v` are
/// `(num_blocks, page_block_size, num_heads_kv, head_size)` and `block_table`
/// `(batch, max_blocks_per_seq)` lists the blocks of each sequence. The pages of each sequence
/// are gathered, then the varlen kernels run. `mm_prefix_ranges` is not supported.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_paged_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Tensor,
    mm_prefix_ranges: Option<&Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    page_block_size: usize,
    softcap: Option<f32>,
) -> Result<Tensor> {
    if mm_prefix_ranges.is_some() {
        bail!("flash-attn: mm_prefix_ranges is not supported by this implementation")
    }
    let (_, page, hkv, d) = k.dims4()?;
    if page != page_block_size {
        bail!("flash-attn: k has pages of {page} tokens, page_block_size is {page_block_size}")
    }
    let cu_k: Vec<usize> = seqlens_k
        .to_device(&Device::Cpu)?
        .to_dtype(DType::I64)?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| v.max(0) as usize)
        .collect();
    let table = int_rows(block_table)?;
    let batch = cu_k.len().saturating_sub(1);
    if table.len() < batch {
        bail!(
            "flash-attn: block_table has {} rows for {batch} sequences",
            table.len()
        )
    }
    let mut ks = Vec::with_capacity(batch);
    let mut vs = Vec::with_capacity(batch);
    for b in 0..batch {
        let len = cu_k[b + 1] - cu_k[b];
        let nb = len.div_ceil(page);
        if nb > table[b].len() {
            bail!(
                "flash-attn: sequence {b} needs {nb} blocks, block_table has {}",
                table[b].len()
            )
        }
        let idx: Vec<u32> = table[b][..nb].iter().map(|&x| x as u32).collect();
        let idx = Tensor::new(idx.as_slice(), k.device())?;
        let gather = |x: &Tensor| -> Result<Tensor> {
            x.index_select(&idx, 0)?
                .reshape((nb * page, hkv, d))?
                .narrow(0, 0, len)
        };
        ks.push(gather(k)?);
        vs.push(gather(v)?);
    }
    let k = Tensor::cat(&ks, 0)?;
    let v = Tensor::cat(&vs, 0)?;
    run_varlen(
        q,
        &k,
        &v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        None,
        (window_size_left, window_size_right),
        softcap,
    )
}

/// Variable length flash attention with ALiBi slopes.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    run_varlen(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        Some(alibi_slopes),
        causal_window(causal),
        None,
    )
}

/// Variable length flash attention with ALiBi slopes and a sliding window.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    run_varlen(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        Some(alibi_slopes),
        (window_size_left, window_size_right),
        None,
    )
}

/// Variable length flash attention with optional ALiBi slopes, a sliding window and
/// soft-capping.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    run_varlen(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        alibi_slopes,
        (window_size_left, window_size_right),
        Some(softcap),
    )
}
