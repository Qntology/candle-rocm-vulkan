//! Parameters, kernel planning and the CPU reference of the fused attention.
//!
//! Semantics follow FlashAttention (Dao-AILab): `softmax(scale · Q Kᵀ + bias) V` per head with
//!
//! * layouts `(batch, seq, heads, head_dim)` (dense) or `(total_tokens, heads, head_dim)` with
//!   cumulative `u32` sequence offsets (varlen); any strides, the last dim contiguous;
//! * grouped / multi-query attention when K/V have fewer heads (`hq % hkv == 0`);
//! * masks aligned to the bottom right: query `i` sits at key position `i + lk - lq`; a key `j`
//!   is visible when `pos - window_left <= j <= pos + window_right` (negative = unbounded,
//!   `window_right = 0` is causal);
//! * optional soft-capping `cap · tanh(s / cap)` and ALiBi `-slope · |pos - j|`;
//! * FP8 (e4m3) K/V multiplied by `k_descale` / `v_descale`.
//!
//! Rows whose keys are all masked produce zeros.

use super::config::{FlashAttnVersion, SplitKv};
use crate::{bail, DType, Layout, Result, Shape};
use rayon::prelude::*;

/// Threads per workgroup of every attention kernel.
pub const FA_THREADS: usize = 128;
/// Largest head dim of the fused kernels.
pub const FA_MAX_HEAD_DIM: usize = 256;
/// Queries up to this length run the row (flash-decoding) kernel.
pub const FA_DECODE_MAX_Q: usize = 4;
/// Upper bound of the split-KV factor.
pub const FA_MAX_SPLITS: usize = 64;
/// Most blocks of one paged attention call ([`PageTable`]).
pub const FA_MAX_PAGES: usize = 128;

/// Run time parameters of one attention call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashAttnParams {
    pub softmax_scale: f32,
    /// Soft-capping value, 0 disables it.
    pub softcap: f32,
    /// Left window (keys before the diagonal), negative = unbounded.
    pub window_left: i32,
    /// Right window (keys after the diagonal), negative = unbounded, 0 = causal.
    pub window_right: i32,
    /// Factor applied to FP8 keys (1 otherwise).
    pub k_descale: f32,
    /// Factor applied to FP8 values (1 otherwise).
    pub v_descale: f32,
    pub version: FlashAttnVersion,
    pub split_kv: SplitKv,
    /// Let the row kernels run the query heads that share a kv head in one workgroup
    /// (`fa_rowg*`, bit-identical results, fewer K/V loads).
    pub gqa_pack: bool,
}

impl FlashAttnParams {
    pub fn new(softmax_scale: f32) -> Self {
        Self {
            softmax_scale,
            softcap: 0.0,
            window_left: -1,
            window_right: -1,
            k_descale: 1.0,
            v_descale: 1.0,
            version: FlashAttnVersion::V2,
            split_kv: SplitKv::Auto,
            gqa_pack: true,
        }
    }

    /// Window from the `candle-flash-attn` convention (`None` = unbounded).
    pub fn with_window(mut self, left: Option<usize>, right: Option<usize>) -> Self {
        let cv = |w: Option<usize>| w.map(|v| v.min(i32::MAX as usize) as i32).unwrap_or(-1);
        self.window_left = cv(left);
        self.window_right = cv(right);
        self
    }

    pub fn causal(&self) -> bool {
        self.window_left < 0 && self.window_right == 0
    }
}

/// Visible key range `[lo, hi)` of query `i` (bottom-right aligned masks).
#[inline]
pub fn key_range(i: usize, lq: usize, lk: usize, wl: i32, wr: i32) -> (usize, usize) {
    let pos = i as i64 + lk as i64 - lq as i64;
    let lo = if wl >= 0 { (pos - wl as i64).max(0) } else { 0 };
    let hi = if wr >= 0 {
        (pos + wr as i64 + 1).min(lk as i64)
    } else {
        lk as i64
    };
    let lo = lo.min(lk as i64);
    (lo as usize, hi.max(lo) as usize)
}

/// Shapes and strides of one attention call, from the candle layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnGeometry {
    pub batch: usize,
    /// Query length (the maximum one for varlen).
    pub lq: usize,
    /// Key length (the maximum one for varlen).
    pub lk: usize,
    pub hq: usize,
    pub hkv: usize,
    pub d: usize,
    /// `(batch, seq, head)` strides in elements; the batch stride is unused for varlen.
    pub q_strides: [usize; 3],
    pub k_strides: [usize; 3],
    pub v_strides: [usize; 3],
    pub q_offset: usize,
    pub k_offset: usize,
    pub v_offset: usize,
    pub varlen: bool,
    /// Output tokens: `batch * lq` (dense) or the total number of query tokens (varlen).
    pub total_q: usize,
}

fn check_last_contiguous(name: &str, l: &Layout) -> Result<()> {
    let s = l.stride();
    let d = l.dims();
    if *s.last().unwrap_or(&1) != 1 && *d.last().unwrap_or(&1) > 1 {
        bail!("flash-attn: the last dim of {name} must be contiguous (strides {s:?})")
    }
    Ok(())
}

impl AttnGeometry {
    /// `q: (b, lq, hq, d)`, `k`/`v: (b, lk, hkv, d)`.
    pub fn dense(q_l: &Layout, k_l: &Layout, v_l: &Layout) -> Result<Self> {
        let (b, lq, hq, d) = q_l.shape().dims4()?;
        let (bk, lk, hkv, dk) = k_l.shape().dims4()?;
        let (bv, lv, hv, dv) = v_l.shape().dims4()?;
        if bk != b || bv != b {
            bail!("flash-attn: batch mismatch q {b}, k {bk}, v {bv}")
        }
        if lv != lk || hv != hkv {
            bail!(
                "flash-attn: k {:?} and v {:?} shapes differ",
                k_l.dims(),
                v_l.dims()
            )
        }
        Self::finish(
            b,
            lq,
            lk,
            hq,
            hkv,
            [d, dk, dv],
            [q_l, k_l, v_l],
            false,
            b * lq,
        )
    }

    /// `q: (total_q, hq, d)`, `k`/`v: (total_k, hkv, d)` with `batch` sequences.
    pub fn varlen(
        q_l: &Layout,
        k_l: &Layout,
        v_l: &Layout,
        batch: usize,
        max_q: usize,
        max_k: usize,
    ) -> Result<Self> {
        let (tq, hq, d) = q_l.shape().dims3()?;
        let (tk, hkv, dk) = k_l.shape().dims3()?;
        let (tv, hv, dv) = v_l.shape().dims3()?;
        if tv != tk || hv != hkv {
            bail!(
                "flash-attn varlen: k {:?} and v {:?} shapes differ",
                k_l.dims(),
                v_l.dims()
            )
        }
        Self::finish(
            batch,
            max_q,
            max_k,
            hq,
            hkv,
            [d, dk, dv],
            [q_l, k_l, v_l],
            true,
            tq,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        batch: usize,
        lq: usize,
        lk: usize,
        hq: usize,
        hkv: usize,
        ds: [usize; 3],
        ls: [&Layout; 3],
        varlen: bool,
        total_q: usize,
    ) -> Result<Self> {
        let d = ds[0];
        if ds[1] != d || ds[2] != d {
            bail!(
                "flash-attn: head dims differ q {}, k {}, v {}",
                ds[0],
                ds[1],
                ds[2]
            )
        }
        if hkv == 0 || hq % hkv != 0 {
            bail!("flash-attn: {hq} query heads are not a multiple of {hkv} kv heads")
        }
        for (n, l) in ["q", "k", "v"].iter().zip(ls.iter()) {
            check_last_contiguous(n, l)?;
        }
        let st = |l: &Layout| -> [usize; 3] {
            let s = l.stride();
            if varlen {
                [0, s[0], s[1]]
            } else {
                [s[0], s[1], s[2]]
            }
        };
        Ok(Self {
            batch,
            lq,
            lk,
            hq,
            hkv,
            d,
            q_strides: st(ls[0]),
            k_strides: st(ls[1]),
            v_strides: st(ls[2]),
            q_offset: ls[0].start_offset(),
            k_offset: ls[1].start_offset(),
            v_offset: ls[2].start_offset(),
            varlen,
            total_q,
        })
    }

    /// Geometry of a paged call: `q: (b, lq, hq, d)` (any strides, last dim contiguous) over
    /// `lk` keys of `hkv` heads spread over the blocks of a [`PageTable`]; `k_row` / `v_row` are
    /// the row strides (elements) shared by all blocks. The K/V offsets and the batch / head
    /// strides come from the table.
    pub fn paged(q_l: &Layout, hkv: usize, lk: usize, k_row: usize, v_row: usize) -> Result<Self> {
        let (b, lq, hq, d) = q_l.shape().dims4()?;
        if hkv == 0 || hq % hkv != 0 {
            bail!("flash-attn: {hq} query heads are not a multiple of {hkv} kv heads")
        }
        check_last_contiguous("q", q_l)?;
        let s = q_l.stride();
        Ok(Self {
            batch: b,
            lq,
            lk,
            hq,
            hkv,
            d,
            q_strides: [s[0], s[1], s[2]],
            k_strides: [0, k_row, 0],
            v_strides: [0, v_row, 0],
            q_offset: q_l.start_offset(),
            k_offset: 0,
            v_offset: 0,
            varlen: false,
            total_q: b * lq,
        })
    }

    pub fn out_shape(&self) -> Shape {
        if self.varlen {
            Shape::from((self.total_q, self.hq, self.d))
        } else {
            Shape::from((self.batch, self.lq, self.hq, self.d))
        }
    }

    /// Output rows (`total_q * hq`), each of `d` elements.
    pub fn out_rows(&self) -> usize {
        self.total_q * self.hq
    }

    /// Whether the fused kernels support this shape.
    pub fn kernel_supported(&self) -> bool {
        self.d > 0
            && self.d <= FA_MAX_HEAD_DIM
            && self.lq <= u32::MAX as usize / 2
            && self.lk <= u32::MAX as usize / 2
            && self.batch <= 65535
            && (self.total_q * self.hq) < (1usize << 31)
    }
}

/// One block of a paged K/V cache on a backend storage type `S`: K (V) element / word offset of
/// its (batch 0, kv head 0, row 0), `rows` valid keys, and the head strides `k_cap` / `v_cap` in
/// rows (layout `(batch, kv head, capacity, row)`, batch stride `hkv` head strides).
pub struct PagedKv<'a, S> {
    pub k: &'a S,
    pub k_offset: usize,
    pub v: &'a S,
    pub v_offset: usize,
    pub rows: usize,
    pub k_cap: usize,
    pub v_cap: usize,
}

/// Block table of the paged kernels, passed to them by value (`ax_pages` in
/// `kernels/accel_common.h`, same layout): device pointers of the blocks, their valid rows and
/// head strides (in rows).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageTable {
    pub k: [u64; FA_MAX_PAGES],
    pub v: [u64; FA_MAX_PAGES],
    pub rows: [u32; FA_MAX_PAGES],
    pub k_cap: [u32; FA_MAX_PAGES],
    pub v_cap: [u32; FA_MAX_PAGES],
    pub n: u32,
    pub pad: u32,
}

impl Default for PageTable {
    fn default() -> Self {
        Self {
            k: [0; FA_MAX_PAGES],
            v: [0; FA_MAX_PAGES],
            rows: [0; FA_MAX_PAGES],
            k_cap: [0; FA_MAX_PAGES],
            v_cap: [0; FA_MAX_PAGES],
            n: 0,
            pad: 0,
        }
    }
}

impl PageTable {
    /// Appends a block: device addresses of its first K / V row, valid rows, head strides in
    /// rows.
    pub fn push(&mut self, k: u64, v: u64, rows: usize, k_cap: usize, v_cap: usize) -> Result<()> {
        let i = self.n as usize;
        if i >= FA_MAX_PAGES {
            bail!("paged attention: more than {FA_MAX_PAGES} blocks")
        }
        if rows > k_cap.max(1) || rows > v_cap.max(1) || k_cap > u32::MAX as usize || v_cap > u32::MAX as usize {
            bail!("paged attention: block of {rows} rows with capacities {k_cap} / {v_cap}")
        }
        self.k[i] = k;
        self.v[i] = v;
        self.rows[i] = rows as u32;
        self.k_cap[i] = k_cap as u32;
        self.v_cap[i] = v_cap as u32;
        self.n += 1;
        Ok(())
    }

    /// Total number of keys.
    pub fn total_rows(&self) -> usize {
        self.rows[..self.n as usize].iter().map(|&r| r as usize).sum()
    }

    /// The table as 64-bit words in memory order (kernel argument buffers of the ROCm backend).
    pub fn to_words(&self) -> Vec<u64> {
        const N: usize = std::mem::size_of::<PageTable>() / 8;
        // SAFETY: repr(C) of u64 / u32 arrays and two u32: 8-byte aligned, no padding, N words.
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u64, N) }.to_vec()
    }
}

/// Optional device inputs of a fused attention call on a backend storage type `S`.
pub struct FaExtra<'a, S> {
    /// f32 slopes: storage, start offset (elements) and batch stride (0 or `hq`).
    pub alibi: Option<(&'a S, usize, usize)>,
    /// Varlen `u32` cumulative sequence offsets of the queries and keys: storages and start
    /// offsets.
    pub cu_seqlens: Option<(&'a S, usize, &'a S, usize)>,
}

impl<S> Default for FaExtra<'_, S> {
    fn default() -> Self {
        Self {
            alibi: None,
            cu_seqlens: None,
        }
    }
}

/// Largest grid size along x of the GPU launches.
pub const MAX_GRID_X: usize = (1 << 31) - 1;

/// Grid and shared memory of a plan: `((x, y, z), shared bytes, splits)`.
pub fn launch_dims(geom: &AttnGeometry, plan: &FaPlan) -> ((u32, u32, u32), usize, usize) {
    match *plan {
        FaPlan::Row { splits } => (
            (
                (geom.lq * geom.hq).clamp(1, MAX_GRID_X) as u32,
                splits as u32,
                geom.batch as u32,
            ),
            row_smem_bytes(geom.d),
            splits,
        ),
        FaPlan::Tile { br, bc, .. } => (
            (
                geom.lq.div_ceil(br).clamp(1, MAX_GRID_X) as u32,
                geom.hq as u32,
                geom.batch as u32,
            ),
            tile_smem_bytes(geom.d, br, bc),
            1,
        ),
    }
}

/// Kernel launch plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaPlan {
    /// One workgroup per query row (and key split); FA1 tier and decoding.
    Row { splits: usize },
    /// Q tiles of `br` rows, K/V tiles of `bc` keys, head dim instantiated up to `dmax`.
    Tile {
        version: FlashAttnVersion,
        dmax: usize,
        br: usize,
        bc: usize,
    },
}

/// `(dmax, br, bc)` of the tile kernel of a version for head dim `d`.
pub fn tile_config(version: FlashAttnVersion, d: usize) -> (usize, usize, usize) {
    let dmax = if d <= 64 {
        64
    } else if d <= 128 {
        128
    } else {
        256
    };
    // V2 tiles fit in 32 KiB of shared memory (the Vulkan minimum of most desktop GPUs), V3
    // tiles in 64 KiB (AMD LDS, NVIDIA with the dynamic shared memory opt-in).
    let (br, bc) = match (version, dmax) {
        (FlashAttnVersion::V3, 64) => (64, 64),
        (FlashAttnVersion::V3, 128) => (32, 64),
        (FlashAttnVersion::V3, _) => (16, 32),
        (_, 64) => (32, 32),
        (_, 128) => (16, 32),
        (_, _) => (8, 16),
    };
    (dmax, br, bc)
}

/// Dynamic shared memory of the tile kernel for head dim `d` (Q and K/V rows padded by one
/// float against bank conflicts).
pub fn tile_smem_bytes(d: usize, br: usize, bc: usize) -> usize {
    ((br + bc) * (d + 1) + br * bc) * 4
}

/// Dynamic shared memory of the row kernel for head dim `d` (per query head of a GQA pack).
pub fn row_smem_bytes(d: usize) -> usize {
    (d + 2 * FA_THREADS) * 4
}

/// Number of key splits for the row kernel.
pub fn split_count(geom: &AttnGeometry, policy: SplitKv, compute_units: usize) -> usize {
    split_count_for(geom.out_rows(), geom.lk, policy, compute_units)
}

/// Workgroups per compute unit that [`split_count_for`] aims for.
pub const SPLIT_WORKGROUPS_PER_CU: usize = 64;

/// Number of key splits for `rows` query rows over `lk` keys (row kernels `fa_row*` / `tq_attn*`):
/// one 128-key window per split, as long as the GPU gets fewer than
/// [`SPLIT_WORKGROUPS_PER_CU`] workgroups per compute unit, at most [`FA_MAX_SPLITS`].
///
/// The row kernels are latency bound (one serial dot product per lane and key), so more,
/// shorter splits keep winning until a split is a single window. Measured on an RTX 3050 laptop
/// GPU (16 SMs) for decoding (16 or 64 query rows, 1K-8K keys, f32 / bf16 queries, f32 / f16 /
/// bf16 / FP8 / TurboQuant FP8 K/V): within 3% of the best fixed split count on average (worst
/// case 10%), 1.4x-3x faster than the previous rule (2 workgroups per CU, 256+ keys per split).
pub fn split_count_for(rows: usize, lk: usize, policy: SplitKv, compute_units: usize) -> usize {
    match policy {
        SplitKv::Off => 1,
        SplitKv::Fixed(n) => n.clamp(1, FA_MAX_SPLITS),
        SplitKv::Auto => {
            let rows = rows.max(1);
            let by_occupancy = (SPLIT_WORKGROUPS_PER_CU * compute_units.max(1)).div_ceil(rows);
            let by_keys = lk.div_ceil(FA_THREADS).max(1);
            by_occupancy.min(by_keys).clamp(1, FA_MAX_SPLITS)
        }
    }
}

/// Picks the kernel for a call. `max_smem` is the shared memory a workgroup may use.
pub fn plan(
    geom: &AttnGeometry,
    params: &FlashAttnParams,
    max_smem: usize,
    compute_units: usize,
) -> FaPlan {
    let row = FaPlan::Row {
        splits: split_count(geom, params.split_kv, compute_units),
    };
    if params.version == FlashAttnVersion::V1 || geom.lq <= FA_DECODE_MAX_Q {
        return row;
    }
    for version in [params.version, FlashAttnVersion::V2] {
        let (dmax, br, bc) = tile_config(version, geom.d);
        if tile_smem_bytes(geom.d, br, bc) <= max_smem {
            return FaPlan::Tile {
                version,
                dmax,
                br,
                bc,
            };
        }
    }
    row
}

/// Suffix of a kernel element type: `f32`, `f16`, `bf16`, `fp8`.
pub fn dtype_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        DType::F8E4M3 => Some("fp8"),
        _ => None,
    }
}

/// Kernel name of a plan for query / KV dtypes.
pub fn kernel_name(plan: &FaPlan, q: DType, kv: DType) -> Result<String> {
    let (Some(qs), Some(ks)) = (dtype_suffix(q), dtype_suffix(kv)) else {
        bail!("flash-attn: unsupported dtypes q {q:?}, kv {kv:?}")
    };
    if q == DType::F8E4M3 {
        bail!("flash-attn: FP8 queries are not supported, use f32/f16/bf16 queries")
    }
    if !kv_dtype_supported(q, kv) {
        bail!("flash-attn: k/v dtype {kv:?} must match q {q:?}, be F8E4M3, or F16 / BF16 with F32 queries")
    }
    Ok(match plan {
        FaPlan::Row { .. } => format!("fa_row_{qs}_{ks}"),
        FaPlan::Tile { version, dmax, .. } => {
            let v = match version {
                FlashAttnVersion::V3 => 3,
                _ => 2,
            };
            format!("fa{v}_d{dmax}_{qs}_{ks}")
        }
    })
}

/// Whether the CUDA / ROCm kernels read K/V of dtype `kv` with queries of dtype `q` directly: the
/// same dtype, FP8 (e4m3), or exact F16 / BF16 K/V with F32 queries (converted to f32 in the
/// kernels, f32 math).
pub fn kv_dtype_supported(q: DType, kv: DType) -> bool {
    kv == q
        || kv == DType::F8E4M3
        || (q == DType::F32 && matches!(kv, DType::F16 | DType::BF16))
}

/// Kernel name of the paged row kernel for query / KV dtypes.
pub fn paged_kernel_name(q: DType, kv: DType) -> Result<String> {
    let name = kernel_name(&FaPlan::Row { splits: 1 }, q, kv)?;
    Ok(name.replacen("fa_row_", "fa_rowp_", 1))
}

/// Query heads per workgroup of the GQA-packed row kernels for `hq` query and `hkv` kv heads:
/// the largest of 8, 4, 2 that divides `hq / hkv`, 1 (plain row kernel) otherwise.
pub fn row_group(hq: usize, hkv: usize) -> usize {
    let group = hq / hkv.max(1);
    [8, 4, 2]
        .into_iter()
        .find(|g| group >= *g && group % g == 0)
        .unwrap_or(1)
}

/// Launch of a row kernel (`fa_row*`, `fa_rowp*`, GQA-packed `fa_rowg{G}*` / `fa_rowpg{G}*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLaunch {
    pub name: String,
    pub grid: (u32, u32, u32),
    pub shared_bytes: usize,
    pub splits: usize,
    /// Query heads per workgroup.
    pub group: usize,
}

/// Row kernel launch for a call: GQA packing when `p.gqa_pack` allows it (dense batches,
/// `hq / hkv` a multiple of 2), key splits from [`split_count_for`] over the workgroups.
pub fn row_launch(
    geom: &AttnGeometry,
    p: &FlashAttnParams,
    compute_units: usize,
    q: DType,
    kv: DType,
    paged: bool,
) -> Result<RowLaunch> {
    let base = kernel_name(&FaPlan::Row { splits: 1 }, q, kv)?;
    let g = if p.gqa_pack && !geom.varlen {
        row_group(geom.hq, geom.hkv)
    } else {
        1
    };
    let splits = split_count_for(geom.out_rows() / g, geom.lk, p.split_kv, compute_units);
    let prefix = match (paged, g) {
        (false, 1) => "fa_row_".to_string(),
        (true, 1) => "fa_rowp_".to_string(),
        (false, g) => format!("fa_rowg{g}_"),
        (true, g) => format!("fa_rowpg{g}_"),
    };
    Ok(RowLaunch {
        name: base.replacen("fa_row_", &prefix, 1),
        grid: (
            (geom.lq * geom.hq / g).clamp(1, MAX_GRID_X) as u32,
            splits as u32,
            geom.batch as u32,
        ),
        shared_bytes: g * row_smem_bytes(geom.d),
        splits,
        group: g,
    })
}

/// ALiBi slopes `(hq)` or `(batch, hq)`: returns the batch stride (0 or `hq`).
pub fn alibi_batch_stride(shape: &Shape, batch: usize, hq: usize) -> Result<usize> {
    match shape.dims() {
        [h] if *h == hq => Ok(0),
        [b, h] if *b == batch && *h == hq => Ok(hq),
        d => bail!("flash-attn: alibi slopes must be ({hq}) or ({batch}, {hq}), got {d:?}"),
    }
}

/// Inputs of [`reference_attention`]: contiguous f32 data, already decoded from FP8.
pub struct ReferenceInputs<'a> {
    /// `(b, lq, hq, d)` or `(total_q, hq, d)`.
    pub q: &'a [f32],
    /// `(b, lk, hkv, d)` or `(total_k, hkv, d)`.
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub batch: usize,
    pub lq: usize,
    pub lk: usize,
    pub hq: usize,
    pub hkv: usize,
    pub d: usize,
    pub cu_q: Option<&'a [u32]>,
    pub cu_k: Option<&'a [u32]>,
    /// Slopes and their batch stride.
    pub alibi: Option<(&'a [f32], usize)>,
}

/// Multi-threaded CPU attention with the kernel semantics. Returns `(rows, d)` with
/// `row = q_token * hq + head`.
pub fn reference_attention(inp: &ReferenceInputs<'_>, p: &FlashAttnParams) -> Result<Vec<f32>> {
    let (hq, hkv, d) = (inp.hq, inp.hkv, inp.d);
    if hkv == 0 || hq % hkv != 0 {
        bail!("flash-attn: {hq} query heads are not a multiple of {hkv} kv heads")
    }
    let total_q = match inp.cu_q {
        Some(c) => *c.last().unwrap_or(&0) as usize,
        None => inp.batch * inp.lq,
    };
    let mut out = vec![0f32; total_q * hq * d];
    let group = hq / hkv;
    out.par_chunks_mut(d).enumerate().for_each(|(row, o)| {
        let tok = row / hq;
        let h = row % hq;
        let (b, i, lq_b, lk_b, k0) = match (inp.cu_q, inp.cu_k) {
            (Some(cq), Some(ck)) => {
                let b = cq.partition_point(|&c| c as usize <= tok).saturating_sub(1);
                let lq_b = (cq[b + 1] - cq[b]) as usize;
                let lk_b = (ck[b + 1] - ck[b]) as usize;
                (b, tok - cq[b] as usize, lq_b, lk_b, ck[b] as usize)
            }
            _ => {
                let b = tok / inp.lq;
                (b, tok % inp.lq, inp.lq, inp.lk, b * inp.lk)
            }
        };
        let hk = h / group;
        let qv = &inp.q[(tok * hq + h) * d..(tok * hq + h + 1) * d];
        let (lo, hi) = key_range(i, lq_b, lk_b, p.window_left, p.window_right);
        if lo >= hi {
            return;
        }
        let slope = inp.alibi.map(|(s, bs)| s[b * bs + h]).unwrap_or(0.0);
        let pos = i as i64 + lk_b as i64 - lq_b as i64;
        let mut s = Vec::with_capacity(hi - lo);
        let mut m = f32::NEG_INFINITY;
        for j in lo..hi {
            let kv = &inp.k[((k0 + j) * hkv + hk) * d..((k0 + j) * hkv + hk + 1) * d];
            let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * p.k_descale;
            let mut l = dot * p.softmax_scale;
            if p.softcap > 0.0 {
                l = p.softcap * (l / p.softcap).tanh();
            }
            if slope != 0.0 {
                l -= slope * (pos - j as i64).abs() as f32;
            }
            m = m.max(l);
            s.push(l);
        }
        let mut sum = 0f32;
        for x in s.iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        for (t, j) in (lo..hi).enumerate() {
            let w = s[t] / sum * p.v_descale;
            let vv = &inp.v[((k0 + j) * hkv + hk) * d..((k0 + j) * hkv + hk + 1) * d];
            for (oe, ve) in o.iter_mut().zip(vv) {
                *oe += w * ve;
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        // causal, square
        assert_eq!(key_range(0, 4, 4, -1, 0), (0, 1));
        assert_eq!(key_range(3, 4, 4, -1, 0), (0, 4));
        // causal, decoding one token against 10 keys
        assert_eq!(key_range(0, 1, 10, -1, 0), (0, 10));
        // sliding window of 2 to the left
        assert_eq!(key_range(5, 8, 8, 2, 0), (3, 6));
        // no mask
        assert_eq!(key_range(1, 3, 7, -1, -1), (0, 7));
        // more queries than keys: the first rows see nothing
        assert_eq!(key_range(0, 4, 2, -1, 0), (0, 0));
    }

    #[test]
    fn plans() {
        let geom = AttnGeometry {
            batch: 1,
            lq: 512,
            lk: 512,
            hq: 8,
            hkv: 8,
            d: 128,
            q_strides: [0; 3],
            k_strides: [0; 3],
            v_strides: [0; 3],
            q_offset: 0,
            k_offset: 0,
            v_offset: 0,
            varlen: false,
            total_q: 512,
        };
        let mut p = FlashAttnParams::new(1.0);
        p.version = FlashAttnVersion::V3;
        assert_eq!(
            plan(&geom, &p, 64 * 1024, 60),
            FaPlan::Tile {
                version: FlashAttnVersion::V3,
                dmax: 128,
                br: 32,
                bc: 64
            }
        );
        // not enough shared memory for the V3 tiles: V2 tiles
        assert!(matches!(
            plan(&geom, &p, 32 * 1024, 60),
            FaPlan::Tile {
                version: FlashAttnVersion::V2,
                ..
            }
        ));
        let dec = AttnGeometry {
            lq: 1,
            total_q: 1,
            lk: 8192,
            ..geom
        };
        // one 128-key window per split, at most 64 splits
        assert_eq!(plan(&dec, &p, 64 * 1024, 60), FaPlan::Row { splits: 64 });
        let short = AttnGeometry { lk: 1000, ..dec };
        assert_eq!(plan(&short, &p, 64 * 1024, 60), FaPlan::Row { splits: 8 });
        // many rows: fewer splits, none past 64 workgroups per compute unit
        assert_eq!(split_count_for(64, 2048, SplitKv::Auto, 16), 16);
        assert_eq!(split_count_for(4096, 2048, SplitKv::Auto, 16), 1);
        p.split_kv = SplitKv::Off;
        assert_eq!(plan(&dec, &p, 64 * 1024, 60), FaPlan::Row { splits: 1 });
    }

    #[test]
    fn kernel_names() {
        let row = FaPlan::Row { splits: 1 };
        assert_eq!(kernel_name(&row, DType::F32, DType::F16).unwrap(), "fa_row_f32_f16");
        assert_eq!(kernel_name(&row, DType::F32, DType::BF16).unwrap(), "fa_row_f32_bf16");
        assert!(kernel_name(&row, DType::BF16, DType::F16).is_err());
        assert!(kernel_name(&row, DType::F16, DType::F32).is_err());
        assert_eq!(
            paged_kernel_name(DType::BF16, DType::F8E4M3).unwrap(),
            "fa_rowp_bf16_fp8"
        );
    }

    #[test]
    fn row_launches() {
        let geom = AttnGeometry {
            batch: 1,
            lq: 1,
            lk: 2048,
            hq: 16,
            hkv: 4,
            d: 256,
            q_strides: [0; 3],
            k_strides: [0; 3],
            v_strides: [0; 3],
            q_offset: 0,
            k_offset: 0,
            v_offset: 0,
            varlen: false,
            total_q: 1,
        };
        let mut p = FlashAttnParams::new(1.0);
        let l = row_launch(&geom, &p, 16, DType::F32, DType::F16, false).unwrap();
        assert_eq!(l.name, "fa_rowg4_f32_f16");
        assert_eq!((l.grid, l.splits, l.group), ((4, 16, 1), 16, 4));
        assert_eq!(l.shared_bytes, 4 * row_smem_bytes(256));
        let l = row_launch(&geom, &p, 16, DType::BF16, DType::F8E4M3, true).unwrap();
        assert_eq!(l.name, "fa_rowpg4_bf16_fp8");
        p.gqa_pack = false;
        let l = row_launch(&geom, &p, 16, DType::F32, DType::F32, true).unwrap();
        assert_eq!((l.name.as_str(), l.grid.0, l.group), ("fa_rowp_f32_f32", 16, 1));
        assert_eq!(row_group(16, 8), 2);
        assert_eq!(row_group(32, 2), 8);
        assert_eq!(row_group(12, 4), 1);
        assert_eq!(row_group(8, 8), 1);
    }

    #[test]
    fn page_table_layout() {
        // must match `struct ax_pages` of kernels/accel_common.h
        assert_eq!(std::mem::size_of::<PageTable>(), 3592);
        assert_eq!(std::mem::align_of::<PageTable>(), 8);
        let mut t = PageTable::default();
        t.push(0x1000, 0x2000, 7, 16, 8).unwrap();
        t.push(0x3000, 0x4000, 3, 3, 3).unwrap();
        assert!(t.push(0, 0, 9, 8, 8).is_err());
        assert_eq!(t.total_rows(), 10);
        let w = t.to_words();
        assert_eq!(w.len(), 449);
        assert_eq!(w[0], 0x1000);
        assert_eq!(w[FA_MAX_PAGES + 1], 0x4000);
        // rows[0], rows[1] share word 256 (little endian)
        assert_eq!(w[2 * FA_MAX_PAGES], 7 | (3 << 32));
        assert_eq!(w[448], 2);
    }
}
