//! Fused attention and TurboQuant kernels on the CUDA backend.
//!
//! The kernels are the ones of the ROCm backend (`kernels/flash_attn_impl.h`,
//! `kernels/turbo_quant_impl.h`), compiled with NVRTC for the GPU's compute capability the first
//! time they are used (one module per attention kernel, one for all TurboQuant kernels).

use super::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::accel::attention::{
    self as att, AttnGeometry, FaExtra, FaPlan, FlashAttnParams, PageTable, PagedKv, FA_THREADS,
    MAX_GRID_X,
};
use crate::accel::config::FlashAttnVersion;
use crate::accel::profile::AccelProfile;
use crate::accel::sources;
use crate::accel::turbo::{
    tq_attn_group, tq_attn_kernel, tq_tile_launch, TqAttnArgs, TqRowArgs, TQ_THREADS,
};
use crate::{bail, DType, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg, SyncOnDrop};
use std::sync::MutexGuard;

impl CudaDevice {
    /// Cached accelerator profile reference (zero lock, zero clone on hot path).
    pub(crate) fn accel_profile_ref(&self) -> &AccelProfile {
        self.accel_profile_slot().get_or_init(|| {
            use cudarc::driver::sys::CUdevice_attribute as A;
            let (major, minor) = self.compute_cap();
            let attr = |a| self.device_attribute(a).unwrap_or(0);
            let sms = attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).max(1) as usize;
            let smem = attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
                .max(48 * 1024) as usize;
            AccelProfile::cuda(major.max(0) as u32, minor.max(0) as u32, smem, sms)
        })
    }

    /// Fused attention / TurboQuant profile of this GPU (see `candle_core::accel::profile`).
    pub fn accel_profile(&self) -> AccelProfile {
        self.accel_profile_ref().clone()
    }

    fn accel_func(&self, name: &str) -> Result<super::device::CudaFunc> {
        if name.starts_with("tq_") {
            return self.get_or_compile_internal_func(
                name,
                || "candle_accel_turbo_quant".to_string(),
                || Ok(sources::turbo_quant_cuda_source()),
            );
        }
        self.get_or_compile_internal_func(
            name,
            || format!("candle_accel_{name}"),
            || {
                let Some(src) = sources::flash_attn_cuda_source(name) else {
                    bail!("flash-attn: unknown kernel {name}")
                };
                Ok(src)
            },
        )
    }
}

/// Raw device pointer at an element offset, with the guard that orders the access.
fn dptr<'a>(
    s: &'a CudaStorage,
    elem_offset: usize,
    stream: &'a cudarc::driver::CudaStream,
) -> (u64, SyncOnDrop<'a>) {
    macro_rules! p {
        ($x:expr, $t:ty) => {{
            let (ptr, g) = $x.device_ptr(stream);
            (ptr + (elem_offset * std::mem::size_of::<$t>()) as u64, g)
        }};
    }
    match &s.slice {
        CudaStorageSlice::U8(x) => p!(x, u8),
        CudaStorageSlice::U32(x) => p!(x, u32),
        CudaStorageSlice::I16(x) => p!(x, i16),
        CudaStorageSlice::I32(x) => p!(x, i32),
        CudaStorageSlice::I64(x) => p!(x, i64),
        CudaStorageSlice::BF16(x) => p!(x, half::bf16),
        CudaStorageSlice::F16(x) => p!(x, half::f16),
        CudaStorageSlice::F32(x) => p!(x, f32),
        CudaStorageSlice::F64(x) => p!(x, f64),
        CudaStorageSlice::F8E4M3(x) => p!(x, float8::F8E4M3),
        CudaStorageSlice::F6E2M3(x)
        | CudaStorageSlice::F6E3M2(x)
        | CudaStorageSlice::F4(x)
        | CudaStorageSlice::F8E8M0(x) => p!(x, u8),
    }
}

/// Uninitialized output of `n` elements.
fn alloc(dev: &CudaDevice, dtype: DType, n: usize) -> Result<CudaStorageSlice> {
    // SAFETY: every element is written by the kernels before it is read.
    unsafe {
        Ok(match dtype {
            DType::F32 => CudaStorageSlice::F32(dev.alloc::<f32>(n)?),
            DType::F16 => CudaStorageSlice::F16(dev.alloc::<half::f16>(n)?),
            DType::BF16 => CudaStorageSlice::BF16(dev.alloc::<half::bf16>(n)?),
            DType::U32 => CudaStorageSlice::U32(dev.alloc::<u32>(n)?),
            d => bail!("accel: unsupported output dtype {d:?}"),
        })
    }
}

fn dptr_mut<'a>(
    s: &'a mut CudaStorageSlice,
    stream: &'a cudarc::driver::CudaStream,
) -> (u64, SyncOnDrop<'a>) {
    match s {
        CudaStorageSlice::F32(x) => x.device_ptr_mut(stream),
        CudaStorageSlice::F16(x) => x.device_ptr_mut(stream),
        CudaStorageSlice::BF16(x) => x.device_ptr_mut(stream),
        CudaStorageSlice::U32(x) => x.device_ptr_mut(stream),
        _ => unreachable!("accel outputs are f32 / f16 / bf16 / u32"),
    }
}

fn half_sfx(dtype: DType, op: &str) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("{op}: unsupported dtype {d:?} on the CUDA backend"),
    }
}

#[allow(dead_code)]
fn fa_combine_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("fa_combine_f32"),
        DType::F16 => Ok("fa_combine_f16"),
        DType::BF16 => Ok("fa_combine_bf16"),
        d => bail!("flash-attn: unsupported dtype {d:?} on the CUDA backend"),
    }
}

fn cfg(grid: (u32, u32, u32), smem: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: (FA_THREADS as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    }
}

fn grid_rows(n: usize) -> u32 {
    n.clamp(1, MAX_GRID_X) as u32
}

fn grid_elems(n: usize) -> u32 {
    n.div_ceil(TQ_THREADS).clamp(1, 1 << 20) as u32
}

// SAFETY: plain repr(C) data with the layout of `struct ax_pages` (kernels/accel_common.h).
unsafe impl cudarc::driver::DeviceRepr for PageTable {}

/// Partial outputs and log-sum-exps of a split-KV launch, backed by the per-device `"fa_split_kv"`
/// workspace (zero per-call stream allocations). Holds the workspace lock until both the main
/// kernel and the combine kernel have been queued on the device stream.
type Parts<'a> = Option<(MutexGuard<'a, super::device::Workspaces>, u64, u64)>;

fn alloc_parts<'a>(
    dev: &'a CudaDevice,
    splits: usize,
    rows: usize,
    d: usize,
) -> Result<Parts<'a>> {
    if splits <= 1 || rows == 0 || d == 0 {
        return Ok(None);
    }
    let po_bytes = (splits * rows * d * 4).div_ceil(256) * 256;
    let pl_bytes = splits * rows * 4;
    let mut ws = dev.workspaces();
    let base = ws.ensure(dev, "fa_split_kv", po_bytes + pl_bytes)?;
    Ok(Some((ws, base, base + po_bytes as u64)))
}

/// Merges the key splits in `part` into `out` with the combine kernel `name`.
fn launch_combine(
    dev: &CudaDevice,
    name: &str,
    part: Parts<'_>,
    out: &mut CudaStorageSlice,
    rows: usize,
    d: usize,
    splits: usize,
) -> Result<()> {
    let Some((_ws, po_p, pl_p)) = part else {
        return Ok(());
    };
    let stream = dev.cuda_stream();
    let comb = dev.accel_func(name)?;
    let (o_p, _go) = dptr_mut(out, &stream);
    let ptrs: [u64; 3] = [po_p, pl_p, o_p];
    let rows64 = rows as u64;
    let dd: [u32; 2] = [d as u32, splits as u32];
    let mut b = comb.builder();
    for x in ptrs.iter() {
        b.arg(x);
    }
    b.arg(&rows64);
    for x in dd.iter() {
        b.arg(x);
    }
    let smem = if name.starts_with("fa_combine_tree_") {
        128 * 4
    } else {
        splits * 4
    };
    // SAFETY: arguments match fa_combine[_tree] / tq_combine.
    unsafe { b.launch(cfg((grid_rows(rows), 1, 1), smem)) }.w()?;
    Ok(())
}

/// Device pointers of the blocks of a paged call, with the guards that order the accesses.
fn page_table<'a>(
    blocks: &[PagedKv<'a, CudaStorage>],
    stream: &'a cudarc::driver::CudaStream,
) -> Result<(PageTable, Vec<SyncOnDrop<'a>>)> {
    let mut table = PageTable::default();
    let mut guards = Vec::with_capacity(2 * blocks.len());
    for bl in blocks {
        let (kp, gk) = dptr(bl.k, bl.k_offset, stream);
        let (vp, gv) = dptr(bl.v, bl.v_offset, stream);
        guards.push(gk);
        guards.push(gv);
        table.push(kp, vp, bl.rows, bl.k_cap, bl.v_cap)?;
    }
    Ok((table, guards))
}

impl CudaStorage {
    /// Fused attention of `self` (queries) over `k` / `v`, see `RocmStorage::flash_attn`.
    pub fn flash_attn(
        &self,
        k: &Self,
        v: &Self,
        geom: &AttnGeometry,
        p: &FlashAttnParams,
        extra: &FaExtra<'_, Self>,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        if k.dtype() != v.dtype() {
            bail!(
                "flash-attn: k {:?} and v {:?} dtypes differ",
                k.dtype(),
                v.dtype()
            )
        }
        let dev = &self.device;
        let prof = dev.accel_profile_ref();
        let plan = att::plan(geom, p, prof.max_shared_mem, prof.compute_units);
        // row plans may pack the query heads of a kv head (fa_rowg*, same results)
        let (name, grid, smem, splits) = match plan {
            FaPlan::Row { .. } => {
                let l = att::row_launch(
                    geom,
                    p,
                    prof.compute_units,
                    self.dtype(),
                    k.dtype(),
                    false,
                )?;
                (l.name, l.grid, l.shared_bytes, l.splits)
            }
            FaPlan::Tile { .. } => {
                let name = att::kernel_name(&plan, self.dtype(), k.dtype())?;
                let (grid, smem, splits) = att::launch_dims(geom, &plan);
                (name, grid, smem, splits)
            }
        };
        let rows = geom.out_rows();
        let d = geom.d;
        let mut out = alloc(dev, self.dtype(), rows * d)?;
        if rows > 0 && d > 0 {
            let func = dev.accel_func(&name)?;
            dev.ensure_func_max_smem(&name, &func, smem)?;
            let stream = dev.cuda_stream();
            let part = alloc_parts(dev, splits, rows, d)?;
            {
                let (q_p, _gq) = dptr(self, geom.q_offset, &stream);
                let (k_p, _gk) = dptr(k, geom.k_offset, &stream);
                let (v_p, _gv) = dptr(v, geom.v_offset, &stream);
                let (o_p, _go) = dptr_mut(&mut out, &stream);
                let (po_p, pl_p) = part.as_ref().map(|(_, a, b)| (*a, *b)).unwrap_or((0, 0));
                let (al_p, alibi_bs, _ga) = match extra.alibi {
                    Some((s, off, bs)) => {
                        let (p, g) = dptr(s, off, &stream);
                        (p, bs as u32, Some(g))
                    }
                    None => (0, 0, None),
                };
                let (cq_p, ck_p, _gc) = match extra.cu_seqlens {
                    Some((sq, oq, sk, ok)) => {
                        let (a, ga) = dptr(sq, oq, &stream);
                        let (b, gb) = dptr(sk, ok, &stream);
                        (a, b, Some((ga, gb)))
                    }
                    None => (0, 0, None),
                };
                let ptrs: [u64; 9] = [q_p, k_p, v_p, o_p, po_p, pl_p, al_p, cq_p, ck_p];
                let dims: [u32; 5] = [
                    geom.lq as u32,
                    geom.lk as u32,
                    geom.hq as u32,
                    geom.hkv as u32,
                    d as u32,
                ];
                let mut strides = [0u64; 9];
                for (i, s) in [geom.q_strides, geom.k_strides, geom.v_strides]
                    .iter()
                    .enumerate()
                {
                    for j in 0..3 {
                        strides[i * 3 + j] = s[j] as u64;
                    }
                }
                let scale_cap: [f32; 2] = [p.softmax_scale, p.softcap];
                let windows: [i32; 2] = [p.window_left, p.window_right];
                let descale: [f32; 2] = [p.k_descale, p.v_descale];
                let tail: [u32; 3] = [splits as u32, alibi_bs, geom.varlen as u32];
                let rows64 = rows as u64;
                let mut b = func.builder();
                for x in ptrs.iter() {
                    b.arg(x);
                }
                for x in dims.iter() {
                    b.arg(x);
                }
                for x in strides.iter() {
                    b.arg(x);
                }
                for x in scale_cap.iter() {
                    b.arg(x);
                }
                for x in windows.iter() {
                    b.arg(x);
                }
                for x in descale.iter() {
                    b.arg(x);
                }
                for x in tail.iter() {
                    b.arg(x);
                }
                b.arg(&rows64);
                // SAFETY: arguments match the kernel signature (FA_PARAMS).
                unsafe { b.launch(cfg(grid, smem)) }.w()?;
            }
            let sfx = half_sfx(self.dtype(), "flash-attn")?;
            let comb = if p.deterministic {
                format!("fa_combine_tree_{sfx}")
            } else {
                format!("fa_combine_{sfx}")
            };
            launch_combine(dev, &comb, part, &mut out, rows, d, splits)?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// Attention of `self` (queries, geometry from `AttnGeometry::paged`) over the K/V blocks of
    /// a paged cache, without concatenating them (`fa_rowp_*` for decode, `fa2p_*`/`fa3p_*` for
    /// chunked prefill). Output contiguous in the query dtype.
    pub fn flash_attn_paged(
        &self,
        blocks: &[PagedKv<'_, Self>],
        geom: &AttnGeometry,
        p: &FlashAttnParams,
        extra: &FaExtra<'_, Self>,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let Some(first) = blocks.first() else {
            bail!("paged flash-attn: no blocks")
        };
        let kv_dtype = first.k.dtype();
        if blocks
            .iter()
            .any(|b| b.k.dtype() != kv_dtype || b.v.dtype() != kv_dtype)
        {
            bail!("paged flash-attn: every block needs the same k / v dtype")
        }
        if geom.varlen || extra.cu_seqlens.is_some() {
            bail!("paged flash-attn: varlen batches are not supported")
        }
        let dev = &self.device;
        let prof = dev.accel_profile_ref();
        let plan = att::plan(geom, p, prof.max_shared_mem, prof.compute_units);
        let (name, grid, smem, splits) = match plan {
            FaPlan::Tile { .. } => {
                let name = att::paged_tile_kernel_name(&plan, self.dtype(), kv_dtype)?;
                let (grid, smem, splits) = att::launch_dims(geom, &plan);
                (name, grid, smem, splits)
            }
            FaPlan::Row { .. } => {
                let l =
                    att::row_launch(geom, p, prof.compute_units, self.dtype(), kv_dtype, true)?;
                (l.name, l.grid, l.shared_bytes, l.splits)
            }
        };
        let rows = geom.out_rows();
        let d = geom.d;
        let mut out = alloc(dev, self.dtype(), rows * d)?;
        if rows > 0 && d > 0 {
            let func = dev.accel_func(&name)?;
            dev.ensure_func_max_smem(&name, &func, smem)?;
            let stream = dev.cuda_stream();
            let part = alloc_parts(dev, splits, rows, d)?;
            {
                let (table, _gt) = page_table(blocks, &stream)?;
                if table.total_rows() != geom.lk {
                    bail!(
                        "paged flash-attn: blocks hold {} keys, geometry {}",
                        table.total_rows(),
                        geom.lk
                    )
                }
                let (q_p, _gq) = dptr(self, geom.q_offset, &stream);
                let (o_p, _go) = dptr_mut(&mut out, &stream);
                let (po_p, pl_p) = part.as_ref().map(|(_, a, b)| (*a, *b)).unwrap_or((0, 0));
                let (al_p, alibi_bs, _ga) = match extra.alibi {
                    Some((s, off, bs)) => {
                        let (p, g) = dptr(s, off, &stream);
                        (p, bs as u32, Some(g))
                    }
                    None => (0, 0, None),
                };
                let ptrs: [u64; 9] = [q_p, 0, 0, o_p, po_p, pl_p, al_p, 0, 0];
                let dims: [u32; 5] = [
                    geom.lq as u32,
                    geom.lk as u32,
                    geom.hq as u32,
                    geom.hkv as u32,
                    d as u32,
                ];
                let mut strides = [0u64; 9];
                for (i, s) in [geom.q_strides, geom.k_strides, geom.v_strides]
                    .iter()
                    .enumerate()
                {
                    for j in 0..3 {
                        strides[i * 3 + j] = s[j] as u64;
                    }
                }
                let scale_cap: [f32; 2] = [p.softmax_scale, p.softcap];
                let windows: [i32; 2] = [p.window_left, p.window_right];
                let descale: [f32; 2] = [p.k_descale, p.v_descale];
                let tail: [u32; 3] = [splits as u32, alibi_bs, 0];
                let rows64 = rows as u64;
                let mut b = func.builder();
                for x in ptrs.iter() {
                    b.arg(x);
                }
                for x in dims.iter() {
                    b.arg(x);
                }
                for x in strides.iter() {
                    b.arg(x);
                }
                for x in scale_cap.iter() {
                    b.arg(x);
                }
                for x in windows.iter() {
                    b.arg(x);
                }
                for x in descale.iter() {
                    b.arg(x);
                }
                for x in tail.iter() {
                    b.arg(x);
                }
                b.arg(&rows64);
                b.arg(&table);
                // SAFETY: arguments match FA_ROWP_KERNEL (FA_PARAMS + the block table).
                unsafe { b.launch(cfg(grid, smem)) }.w()?;
            }
            let sfx = half_sfx(self.dtype(), "flash-attn")?;
            let comb = if p.deterministic {
                format!("fa_combine_tree_{sfx}")
            } else {
                format!("fa_combine_{sfx}")
            };
            launch_combine(dev, &comb, part, &mut out, rows, d, splits)?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// O(1) device block-table paged varlen FlashAttention on CUDA (zero D2H sync, zero Tensor::cat).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_paged_block_table(
        &self,
        q_off: usize,
        k_pool: &Self,
        k_off: usize,
        v_pool: &Self,
        v_off: usize,
        cu_q: &Self,
        cu_q_off: usize,
        cu_k: &Self,
        cu_k_off: usize,
        block_table: &Self,
        bt_off: usize,
        mm_prefix_ranges: Option<(&Self, usize)>,
        sink_logits: Option<(&Self, usize)>,
        batch: usize,
        total_q: usize,
        max_q: usize,
        hq: usize,
        hkv: usize,
        d: usize,
        page_size: usize,
        bt_stride: usize,
        p: &FlashAttnParams,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "flash-attn-bt")?;
        if k_pool.dtype() != self.dtype() || v_pool.dtype() != self.dtype() {
            bail!(
                "flash-attn-bt: q/k/v dtypes must match ({:?}, {:?}, {:?})",
                self.dtype(),
                k_pool.dtype(),
                v_pool.dtype()
            );
        }
        if d > att::FA_MAX_HEAD_DIM {
            bail!("flash-attn-bt: head dim {d} > {}", att::FA_MAX_HEAD_DIM);
        }
        let dev = &self.device;
        let (dmax, br, bc) = att::tile_config(FlashAttnVersion::V2, d);
        let smem = att::tile_smem_bytes(d, br, bc);
        let mut out = alloc(dev, self.dtype(), (total_q * hq * d).max(1))?;
        if batch > 0 && total_q > 0 && max_q > 0 && hq > 0 && d > 0 {
            let kname = format!("fa_bt_d{dmax}_{sfx}");
            let func = dev.accel_func(&kname)?;
            dev.ensure_func_max_smem(&kname, &func, smem)?;
            let stream = dev.cuda_stream();
            let (q_p, _gq) = dptr(self, q_off, &stream);
            let (k_p, _gk) = dptr(k_pool, k_off, &stream);
            let (v_p, _gv) = dptr(v_pool, v_off, &stream);
            let (o_p, _go) = dptr_mut(&mut out, &stream);
            let (cq_p, _gcq) = dptr(cu_q, cu_q_off, &stream);
            let (ck_p, _gck) = dptr(cu_k, cu_k_off, &stream);
            let (bt_p, _gbt) = dptr(block_table, bt_off, &stream);
            let (pfx_p, _gpfx) = match mm_prefix_ranges {
                Some((s, off)) => {
                    let (ptr, g) = dptr(s, off, &stream);
                    (ptr, Some(g))
                }
                None => (0u64, None),
            };
            let (sink_p, _gsink) = match sink_logits {
                Some((s, off)) => {
                    let (ptr, g) = dptr(s, off, &stream);
                    (ptr, Some(g))
                }
                None => (0u64, None),
            };
            let hq32 = hq as u32;
            let hkv32 = hkv as u32;
            let d32 = d as u32;
            let ps32 = page_size as u32;
            let bts32 = bt_stride as u32;
            let scale = p.softmax_scale;
            let softcap = p.softcap;
            let wl = p.window_left;
            let wr = p.window_right;
            let mut b = func.builder();
            b.arg(&q_p)
                .arg(&k_p)
                .arg(&v_p)
                .arg(&o_p)
                .arg(&cq_p)
                .arg(&ck_p)
                .arg(&bt_p)
                .arg(&pfx_p)
                .arg(&sink_p)
                .arg(&hq32)
                .arg(&hkv32)
                .arg(&d32)
                .arg(&ps32)
                .arg(&bts32)
                .arg(&scale)
                .arg(&softcap)
                .arg(&wl)
                .arg(&wr);
            let grid = (
                max_q.div_ceil(br).max(1) as u32,
                hq as u32,
                batch as u32,
            );
            // SAFETY: arguments match FA_TILE_BT_KERNEL.
            unsafe { b.launch(cfg(grid, smem)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// Single-pass TurboProd quantization (codes + residual + QJL signs + rsc0/rsc1 in 1 launch).
    pub fn tq_quantize_prod_fused(
        &self,
        offset: usize,
        tab: &Self,
        tab_off: usize,
        s_mat: &Self,
        s_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "turbo-quant")?;
        let dev = &self.device;
        let mut out = alloc(dev, DType::U32, a.rows * a.row_words)?;
        if a.rows > 0 {
            let func = dev.accel_func(&format!("tq_quant_prod_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (s_p, _g1) = dptr(self, offset, &stream);
            let (t_p, _g2) = dptr(tab, tab_off, &stream);
            let (sm_p, _g3) = dptr(s_mat, s_off, &stream);
            let (o_p, _g4) = dptr_mut(&mut out, &stream);
            let rows64 = a.rows as u64;
            let dim32 = a.dim as u32;
            let stride64 = a.dim as u64;
            let cb32 = a.code_bits;
            let cw32 = a.code_words as u32;
            let rw32 = a.row_words as u32;
            let mut b = func.builder();
            b.arg(&s_p)
                .arg(&t_p)
                .arg(&sm_p)
                .arg(&o_p)
                .arg(&rows64)
                .arg(&dim32)
                .arg(&stride64)
                .arg(&cb32)
                .arg(&cw32)
                .arg(&rw32);
            let smem = (2 * a.dim + TQ_THREADS) * 4;
            unsafe { b.launch(cfg((grid_rows(a.rows), 1, 1), smem)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// On-chip O(d log d) Fast Walsh-Hadamard Transform (`rows x dim` -> F32 `rows x dim`).
    pub fn tq_fwht(
        &self,
        offset: usize,
        sign_bits: &Self,
        sign_off: usize,
        rows: usize,
        dim: usize,
        forward: bool,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "tq_fwht")?;
        let dev = &self.device;
        let mut out = alloc(dev, DType::F32, rows * dim)?;
        if rows > 0 && dim > 0 {
            let func = dev.accel_func(&format!("tq_fwht_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (s_p, _g1) = dptr(self, offset, &stream);
            let (o_p, _g2) = dptr_mut(&mut out, &stream);
            let (sb_p, _g3) = dptr(sign_bits, sign_off, &stream);
            let rows64 = rows as u64;
            let dim32 = dim as u32;
            let stride64 = dim as u64;
            let inv_sqrt_d = 1.0f32 / (dim as f32).sqrt();
            let fwd32 = u32::from(forward);
            let mut b = func.builder();
            b.arg(&s_p)
                .arg(&o_p)
                .arg(&sb_p)
                .arg(&rows64)
                .arg(&dim32)
                .arg(&stride64)
                .arg(&inv_sqrt_d)
                .arg(&fwd32);
            unsafe { b.launch(cfg((grid_rows(rows), 1, 1), dim * 4)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// Single-pass zero-HBM-roundtrip FWHT + Quantize on CUDA.
    pub fn tq_fwht_quant_fused(
        &self,
        offset: usize,
        tab: &Self,
        tab_off: usize,
        sign_bits: &Self,
        sign_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "tq_fwht_quant")?;
        let dev = &self.device;
        let mut out = alloc(dev, DType::U32, a.rows * a.row_words)?;
        if a.rows > 0 && a.dim > 0 {
            let func = dev.accel_func(&format!("tq_fwht_quant_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (s_p, _g1) = dptr(self, offset, &stream);
            let (o_p, _g2) = dptr_mut(&mut out, &stream);
            let (t_p, _g3) = dptr(tab, tab_off, &stream);
            let (sb_p, _g4) = dptr(sign_bits, sign_off, &stream);
            let rows64 = a.rows as u64;
            let dim32 = a.dim as u32;
            let stride64 = a.dim as u64;
            let inv_sqrt_d = 1.0f32 / (a.dim as f32).sqrt();
            let mode32 = a.mode;
            let cb32 = a.code_bits;
            let cw32 = a.code_words as u32;
            let aw32 = a.aux_words as u32;
            let rw32 = a.row_words as u32;
            let mut b = func.builder();
            b.arg(&s_p)
                .arg(&o_p)
                .arg(&t_p)
                .arg(&sb_p)
                .arg(&rows64)
                .arg(&dim32)
                .arg(&stride64)
                .arg(&inv_sqrt_d)
                .arg(&mode32)
                .arg(&cb32)
                .arg(&cw32)
                .arg(&aw32)
                .arg(&rw32);
            unsafe { b.launch(cfg((grid_rows(a.rows), 1, 1), a.quant_smem_bytes())) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// Single-pass zero-HBM-roundtrip FWHT + TurboProd (codebook + QJL signs) on CUDA.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_fwht_quant_prod_fused(
        &self,
        offset: usize,
        tab: &Self,
        tab_off: usize,
        s_mat: &Self,
        s_off: usize,
        sign_bits: &Self,
        sign_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "tq_fwht_quant_prod")?;
        let dev = &self.device;
        let mut out = alloc(dev, DType::U32, a.rows * a.row_words)?;
        if a.rows > 0 && a.dim > 0 {
            let func = dev.accel_func(&format!("tq_fwht_quant_prod_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (s_p, _g1) = dptr(self, offset, &stream);
            let (t_p, _g2) = dptr(tab, tab_off, &stream);
            let (sm_p, _g3) = dptr(s_mat, s_off, &stream);
            let (sb_p, _g4) = dptr(sign_bits, sign_off, &stream);
            let (o_p, _g5) = dptr_mut(&mut out, &stream);
            let rows64 = a.rows as u64;
            let dim32 = a.dim as u32;
            let stride64 = a.dim as u64;
            let inv_sqrt_d = 1.0f32 / (a.dim as f32).sqrt();
            let cb32 = a.code_bits;
            let cw32 = a.code_words as u32;
            let rw32 = a.row_words as u32;
            let mut b = func.builder();
            b.arg(&s_p)
                .arg(&t_p)
                .arg(&sm_p)
                .arg(&sb_p)
                .arg(&o_p)
                .arg(&rows64)
                .arg(&dim32)
                .arg(&stride64)
                .arg(&inv_sqrt_d)
                .arg(&cb32)
                .arg(&cw32)
                .arg(&rw32);
            let smem = (2 * a.dim + TQ_THREADS) * 4;
            unsafe { b.launch(cfg((grid_rows(a.rows), 1, 1), smem)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// See `RocmStorage::tq_quantize`.
    pub fn tq_quantize(
        &self,
        offset: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "turbo-quant")?;
        let dev = &self.device;
        let mut out = alloc(dev, DType::U32, a.rows * a.row_words)?;
        if a.rows > 0 {
            let func = dev.accel_func(&format!("tq_quant_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g1) = dptr(self, offset, &stream);
            let (t_p, _g2) = dptr(tab, tab_off, &stream);
            let (o_p, _g3) = dptr_mut(&mut out, &stream);
            let ptrs: [u64; 3] = [x_p, o_p, t_p];
            let rows = a.rows as u64;
            let d = a.dim as u32;
            let stride = a.dim as u64;
            let rest: [u32; 5] = [
                a.mode,
                a.code_bits,
                a.code_words as u32,
                a.aux_words as u32,
                a.row_words as u32,
            ];
            let mut b = func.builder();
            for x in ptrs.iter() {
                b.arg(x);
            }
            b.arg(&rows).arg(&d).arg(&stride);
            for x in rest.iter() {
                b.arg(x);
            }
            // SAFETY: arguments match tq_quant.
            unsafe { b.launch(cfg((grid_rows(a.rows), 1, 1), a.quant_smem_bytes())) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// See `RocmStorage::tq_residual`.
    pub fn tq_residual(
        &self,
        offset: usize,
        data: &Self,
        data_off: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let sfx = half_sfx(self.dtype(), "turbo-quant")?;
        let dev = &self.device;
        let n = a.rows * a.dim;
        let mut out = alloc(dev, DType::F32, n)?;
        if n > 0 {
            let func = dev.accel_func(&format!("tq_resid_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g1) = dptr(self, offset, &stream);
            let (d_p, _g2) = dptr(data, data_off, &stream);
            let (t_p, _g3) = dptr(tab, tab_off, &stream);
            let (o_p, _g4) = dptr_mut(&mut out, &stream);
            let ptrs: [u64; 4] = [x_p, d_p, t_p, o_p];
            let rows = a.rows as u64;
            let d = a.dim as u32;
            let stride = a.dim as u64;
            let rest: [u32; 3] = [a.code_bits, a.row_words as u32, a.rsc_offset() as u32];
            let mut b = func.builder();
            for x in ptrs.iter() {
                b.arg(x);
            }
            b.arg(&rows).arg(&d).arg(&stride);
            for x in rest.iter() {
                b.arg(x);
            }
            // SAFETY: arguments match tq_resid.
            unsafe { b.launch(cfg((grid_elems(n), 1, 1), 0)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// See `RocmStorage::tq_signs`.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_signs(
        &self,
        data_off: usize,
        proj: &Self,
        proj_off: usize,
        resid: &Self,
        resid_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        let dev = &self.device;
        let mut out = alloc(dev, DType::U32, a.rows * a.row_words)?;
        if a.rows > 0 {
            let func = dev.accel_func("tq_signs")?;
            let stream = dev.cuda_stream();
            let (i_p, _g1) = dptr(self, data_off, &stream);
            let (p_p, _g2) = dptr(proj, proj_off, &stream);
            let (r_p, _g3) = dptr(resid, resid_off, &stream);
            let (o_p, _g4) = dptr_mut(&mut out, &stream);
            let ptrs: [u64; 4] = [i_p, p_p, r_p, o_p];
            let rows = a.rows as u64;
            let rest: [u32; 3] = [a.dim as u32, a.code_words as u32, a.row_words as u32];
            let mut b = func.builder();
            for x in ptrs.iter() {
                b.arg(x);
            }
            b.arg(&rows);
            for x in rest.iter() {
                b.arg(x);
            }
            // SAFETY: arguments match tq_signs.
            unsafe { b.launch(cfg((grid_rows(a.rows), 1, 1), TQ_THREADS * 4)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// See `RocmStorage::tq_dequantize`.
    pub fn tq_dequantize(
        &self,
        data_off: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
        with_qjl: bool,
    ) -> Result<Self> {
        let dev = &self.device;
        let n = a.rows * a.dim;
        let parts = if with_qjl { 2 } else { 1 };
        let mut out = alloc(dev, DType::F32, parts * n)?;
        if n > 0 {
            let func = dev.accel_func("tq_dequant")?;
            let stream = dev.cuda_stream();
            let (d_p, _g1) = dptr(self, data_off, &stream);
            let (t_p, _g2) = dptr(tab, tab_off, &stream);
            let (o_p, _g3) = dptr_mut(&mut out, &stream);
            let zs = if with_qjl { o_p + (n * 4) as u64 } else { 0 };
            let ptrs: [u64; 4] = [d_p, t_p, o_p, zs];
            let rows = a.rows as u64;
            let rest: [u32; 6] = [
                a.dim as u32,
                a.mode,
                a.code_bits,
                a.code_words as u32,
                a.row_words as u32,
                a.rsc_offset() as u32,
            ];
            let mut b = func.builder();
            for x in ptrs.iter() {
                b.arg(x);
            }
            b.arg(&rows);
            for x in rest.iter() {
                b.arg(x);
            }
            // SAFETY: arguments match tq_dequant.
            unsafe { b.launch(cfg((grid_elems(n), 1, 1), 0)) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// See `RocmStorage::tq_attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_attention(
        &self,
        kd: &Self,
        vd: &Self,
        tab: &Self,
        tab_off: usize,
        alibi: Option<(&Self, usize)>,
        a: &TqAttnArgs,
    ) -> Result<Self> {
        self.tq_attention_impl(Some((kd, vd)), &[], tab, tab_off, alibi, a)
    }

    /// See `RocmStorage::tq_attention_paged`.
    pub fn tq_attention_paged(
        &self,
        blocks: &[PagedKv<'_, Self>],
        tab: &Self,
        tab_off: usize,
        alibi: Option<(&Self, usize)>,
        a: &TqAttnArgs,
    ) -> Result<Self> {
        if blocks.is_empty() {
            bail!("paged turbo-quant attention: no blocks")
        }
        self.tq_attention_impl(None, blocks, tab, tab_off, alibi, a)
    }

    /// `tq_attn*` over dense rows (`dense = Some((kd, vd))`) or the paged `blocks`.
    #[allow(clippy::too_many_arguments)]
    fn tq_attention_impl(
        &self,
        dense: Option<(&Self, &Self)>,
        blocks: &[PagedKv<'_, Self>],
        tab: &Self,
        tab_off: usize,
        alibi: Option<(&Self, usize)>,
        a: &TqAttnArgs,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        if self.dtype() != DType::F32 {
            bail!(
                "turbo-quant attention: queries must be f32, got {:?}",
                self.dtype()
            )
        }
        let dev = &self.device;
        let rows = a.out_rows();
        let d = a.d;
        let mut out = alloc(dev, DType::F32, rows * d)?;
        if rows > 0 && d > 0 {
            let splits = a.splits.max(1);
            let stream = dev.cuda_stream();
            let prof = dev.accel_profile_ref();
            let tile_plan = if splits <= 1 {
                let version = crate::accel::config()
                    .flash_attn
                    .version
                    .unwrap_or(prof.flash_attn_version);
                tq_tile_launch(a, version, prof.max_shared_mem, dense.is_none())
            } else {
                None
            };
            let group = tq_attn_group(a.hq, a.hkv);
            let is_tile = tile_plan.is_some();
            let (kname, grid, smem) = if let Some((tname, br, _bc, tsmem)) = tile_plan {
                (
                    tname,
                    (
                        a.lq.div_ceil(br).clamp(1, MAX_GRID_X) as u32,
                        a.hq as u32,
                        a.b as u32,
                    ),
                    tsmem,
                )
            } else {
                (
                    tq_attn_kernel(&a.k, &a.v, dense.is_none(), group),
                    (
                        (a.lq * a.hq / group).clamp(1, MAX_GRID_X) as u32,
                        splits as u32,
                        a.b as u32,
                    ),
                    a.smem_bytes_for(group),
                )
            };
            let func = dev.accel_func(&kname)?;
            dev.ensure_func_max_smem(&kname, &func, smem)?;
            let part = if is_tile {
                None
            } else {
                alloc_parts(dev, splits, rows, d)?
            };
            {
                let (table, _gt) = match dense {
                    Some(_) => (None, Vec::new()),
                    None => {
                        let (t, g) = page_table(blocks, &stream)?;
                        if t.total_rows() != a.lk {
                            bail!(
                                "paged turbo-quant attention: blocks hold {} keys, args {}",
                                t.total_rows(),
                                a.lk
                            )
                        }
                        (Some(t), g)
                    }
                };
                let (q_p, _g1) = dptr(self, a.q_offset, &stream);
                let (k_p, v_p, _g23) = match dense {
                    Some((kd, vd)) => {
                        let (k_p, gk) = dptr(kd, a.k_offset, &stream);
                        let (v_p, gv) = dptr(vd, a.v_offset, &stream);
                        (k_p, v_p, Some((gk, gv)))
                    }
                    None => (0, 0, None),
                };
                let (t_p, _g4) = dptr(tab, tab_off, &stream);
                let (al_p, _g5) = match alibi {
                    Some((s, o)) => {
                        let (p, g) = dptr(s, o, &stream);
                        (p, Some(g))
                    }
                    None => (0, None),
                };
                let (o_p, _g6) = dptr_mut(&mut out, &stream);
                let (po_p, pl_p) = part.as_ref().map(|(_, a, b)| (*a, *b)).unwrap_or((0, 0));
                let ptrs: [u64; 8] = [q_p, k_p, v_p, t_p, al_p, o_p, po_p, pl_p];
                let dims: [u32; 5] = [
                    a.lq as u32,
                    a.lk as u32,
                    a.hq as u32,
                    a.hkv as u32,
                    d as u32,
                ];
                let strides: [u64; 9] = [
                    a.q_strides[0] as u64,
                    a.q_strides[1] as u64,
                    a.q_strides[2] as u64,
                    a.k_strides[0] as u64,
                    a.k_strides[1] as u64,
                    a.k_strides[2] as u64,
                    a.v_strides[0] as u64,
                    a.v_strides[1] as u64,
                    a.v_strides[2] as u64,
                ];
                let sides: [u32; 8] = [
                    a.k.mode,
                    a.k.code_bits,
                    a.k.code_words as u32,
                    a.k.rsc_offset as u32,
                    a.v.mode,
                    a.v.code_bits,
                    a.v.code_words as u32,
                    a.v.rsc_offset as u32,
                ];
                let fl: [f32; 2] = [a.scale, a.softcap];
                let win: [i32; 2] = [a.window_left, a.window_right];
                let tail: [u32; 2] = [splits as u32, a.alibi_bstride as u32];
                let rows64 = rows as u64;
                let mut b = func.builder();
                for x in ptrs.iter() {
                    b.arg(x);
                }
                for x in dims.iter() {
                    b.arg(x);
                }
                for x in strides.iter() {
                    b.arg(x);
                }
                for x in sides.iter() {
                    b.arg(x);
                }
                for x in fl.iter() {
                    b.arg(x);
                }
                for x in win.iter() {
                    b.arg(x);
                }
                for x in tail.iter() {
                    b.arg(x);
                }
                b.arg(&rows64);
                if let Some(t) = table.as_ref() {
                    b.arg(t);
                }
                // SAFETY: arguments match tq_attn* / tq_tile* (TQ_ATTN_PARAMS), followed by the
                // block table for the paged variants.
                unsafe { b.launch(cfg(grid, smem)) }.w()?;
            }
            launch_combine(dev, "tq_combine", part, &mut out, rows, d, splits)?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }
}
