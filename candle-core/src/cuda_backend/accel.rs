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
use crate::accel::profile::AccelProfile;
use crate::accel::sources;
use crate::accel::turbo::{tq_attn_group, tq_attn_kernel, TqAttnArgs, TqRowArgs, TQ_THREADS};
use crate::{bail, DType, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg, SyncOnDrop};
use std::collections::HashMap;
use std::sync::Mutex;

fn cc_arch(major: i32, minor: i32) -> &'static str {
    match (major, minor) {
        (7, 0) => "compute_70",
        (7, 2) => "compute_72",
        (7, 5) => "compute_75",
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 7) => "compute_87",
        (8, 9) => "compute_89",
        (9, 0) => "compute_90",
        (10, 0) => "compute_100",
        (10, 1) => "compute_101",
        (10, 3) => "compute_103",
        (11, 0) => "compute_110",
        (12, 0) => "compute_120",
        (12, 1) => "compute_121",
        (m, n) => Box::leak(format!("compute_{m}{n}").into_boxed_str()),
    }
}

impl CudaDevice {
    /// Fused attention / TurboQuant profile of this GPU (see `candle_core::accel::profile`).
    pub fn accel_profile(&self) -> AccelProfile {
        use cudarc::driver::sys::CUdevice_attribute as A;
        static CACHE: Mutex<Option<HashMap<super::DeviceId, AccelProfile>>> = Mutex::new(None);
        let mut g = CACHE.lock().unwrap_or_else(|p| p.into_inner());
        let map = g.get_or_insert_with(HashMap::new);
        if let Some(p) = map.get(&self.id()) {
            return p.clone();
        }
        let attr = |a| self.device_attribute(a).unwrap_or(0);
        let major = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        let minor = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        let sms = attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).max(1) as usize;
        let smem =
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).max(48 * 1024) as usize;
        let p = AccelProfile::cuda(major.max(0) as u32, minor.max(0) as u32, smem, sms);
        map.insert(self.id(), p.clone());
        p
    }

    fn accel_arch(&self) -> &'static str {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let major = self
            .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(7);
        let minor = self
            .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(5);
        cc_arch(major, minor)
    }

    /// Compiles `source` with NVRTC for this GPU (retrying without `--gpu-architecture` when the
    /// installed NVRTC does not know the GPU yet).
    fn accel_compile(&self, source: String) -> Result<cudarc::nvrtc::Ptx> {
        let opts = |arch: Option<&'static str>| cudarc::nvrtc::CompileOptions {
            arch,
            options: vec!["--std=c++17".into()],
            ..Default::default()
        };
        match cudarc::nvrtc::safe::compile_ptx_with_opts(&source, opts(Some(self.accel_arch()))) {
            Ok(p) => Ok(p),
            Err(_) => cudarc::nvrtc::safe::compile_ptx_with_opts(&source, opts(None)).w(),
        }
    }

    fn accel_func(&self, name: &str) -> Result<super::device::CudaFunc> {
        if name.starts_with("tq_") {
            return self.get_or_compile_func(name, "candle_accel_turbo_quant", || {
                self.accel_compile(sources::turbo_quant_cuda_source())
            });
        }
        let module = format!("candle_accel_{name}");
        self.get_or_compile_func(name, &module, || {
            let Some(src) = sources::flash_attn_cuda_source(name) else {
                bail!("flash-attn: unknown kernel {name}")
            };
            self.accel_compile(src)
        })
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

/// Partial outputs and log-sum-exps of a split-KV launch.
type Parts = Option<(CudaStorageSlice, CudaStorageSlice)>;

fn alloc_parts(dev: &CudaDevice, splits: usize, rows: usize, d: usize) -> Result<Parts> {
    Ok(if splits > 1 {
        Some((
            alloc(dev, DType::F32, splits * rows * d)?,
            alloc(dev, DType::F32, splits * rows)?,
        ))
    } else {
        None
    })
}

/// Merges the key splits in `part` into `out` with the combine kernel `name`.
fn launch_combine(
    dev: &CudaDevice,
    name: &str,
    part: &mut Parts,
    out: &mut CudaStorageSlice,
    rows: usize,
    d: usize,
    splits: usize,
) -> Result<()> {
    let Some((po, pl)) = part.as_mut() else {
        return Ok(());
    };
    let stream = dev.cuda_stream();
    let comb = dev.accel_func(name)?;
    let (po_p, _ga) = dptr_mut(po, &stream);
    let (pl_p, _gb) = dptr_mut(pl, &stream);
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
    // SAFETY: arguments match fa_combine / tq_combine; shared memory: one weight per split.
    unsafe { b.launch(cfg((grid_rows(rows), 1, 1), splits * 4)) }.w()?;
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
        let prof = dev.accel_profile();
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
            if smem > 48 * 1024 {
                func.set_max_dynamic_shared_memory(smem)?;
            }
            let stream = dev.cuda_stream();
            let mut part = alloc_parts(dev, splits, rows, d)?;
            {
                let (q_p, _gq) = dptr(self, geom.q_offset, &stream);
                let (k_p, _gk) = dptr(k, geom.k_offset, &stream);
                let (v_p, _gv) = dptr(v, geom.v_offset, &stream);
                let (o_p, _go) = dptr_mut(&mut out, &stream);
                let (po_p, pl_p, _gp) = match part.as_mut() {
                    Some((po, pl)) => {
                        let (a, ga) = dptr_mut(po, &stream);
                        let (b, gb) = dptr_mut(pl, &stream);
                        (a, b, Some((ga, gb)))
                    }
                    None => (0, 0, None),
                };
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
            let comb = format!("fa_combine_{sfx}");
            launch_combine(dev, &comb, &mut part, &mut out, rows, d, splits)?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    /// Decoding attention of `self` (queries, geometry from `AttnGeometry::paged`) over the K/V
    /// blocks of a paged cache, without concatenating them (row kernel `fa_rowp_*`). The keys
    /// are the blocks' rows in order; the output is contiguous in the query dtype.
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
        let prof = dev.accel_profile();
        let launch = att::row_launch(geom, p, prof.compute_units, self.dtype(), kv_dtype, true)?;
        let (grid, smem, splits) = (launch.grid, launch.shared_bytes, launch.splits);
        let rows = geom.out_rows();
        let d = geom.d;
        let mut out = alloc(dev, self.dtype(), rows * d)?;
        if rows > 0 && d > 0 {
            let func = dev.accel_func(&launch.name)?;
            let stream = dev.cuda_stream();
            let mut part = alloc_parts(dev, splits, rows, d)?;
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
                let (po_p, pl_p, _gp) = match part.as_mut() {
                    Some((po, pl)) => {
                        let (a, ga) = dptr_mut(po, &stream);
                        let (b, gb) = dptr_mut(pl, &stream);
                        (a, b, Some((ga, gb)))
                    }
                    None => (0, 0, None),
                };
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
            let comb = format!("fa_combine_{sfx}");
            launch_combine(dev, &comb, &mut part, &mut out, rows, d, splits)?;
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
            let mut part = alloc_parts(dev, splits, rows, d)?;
            {
                let group = tq_attn_group(a.hq, a.hkv);
                let func = dev.accel_func(&tq_attn_kernel(&a.k, &a.v, dense.is_none(), group))?;
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
                let (po_p, pl_p, _g7) = match part.as_mut() {
                    Some((po, pl)) => {
                        let (x, gx) = dptr_mut(po, &stream);
                        let (y, gy) = dptr_mut(pl, &stream);
                        (x, y, Some((gx, gy)))
                    }
                    None => (0, 0, None),
                };
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
                let grid = (
                    (a.lq * a.hq / group).clamp(1, MAX_GRID_X) as u32,
                    splits as u32,
                    a.b as u32,
                );
                // SAFETY: arguments match tq_attn* (TQ_ATTN_PARAMS), followed by the block table
                // for the paged tq_attnp* kernels.
                unsafe { b.launch(cfg(grid, a.smem_bytes_for(group))) }.w()?;
            }
            launch_combine(dev, "tq_combine", &mut part, &mut out, rows, d, splits)?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }
}
