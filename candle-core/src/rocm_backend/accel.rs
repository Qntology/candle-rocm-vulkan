//! Fused attention and TurboQuant kernels on the ROCm backend (`kernels/flash_attn.hip`,
//! `kernels/turbo_quant.hip`). Planning and argument layout are shared with the CUDA backend
//! in `crate::accel`.

use super::kernels::Module;
use super::utils::LaunchArgs;
use super::RocmStorage;
use crate::accel::attention::{
    self as att, AttnGeometry, FaExtra, FlashAttnParams, FA_THREADS, MAX_GRID_X,
};
use crate::accel::turbo::{TqAttnArgs, TqRowArgs, TQ_THREADS};
use crate::{bail, DType, Result};

fn null() -> *const u8 {
    std::ptr::null()
}

fn half_sfx(dtype: DType, op: &str) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("{op}: unsupported dtype {d:?} on the ROCm backend"),
    }
}

fn grid_rows(n: usize) -> u32 {
    n.clamp(1, MAX_GRID_X) as u32
}

fn grid_elems(n: usize) -> u32 {
    n.div_ceil(TQ_THREADS).clamp(1, 1 << 20) as u32
}

impl RocmStorage {
    /// Fused attention of `self` (queries) over `k` / `v` with the geometry computed from the
    /// layouts (`AttnGeometry::dense` / `varlen`). The output is contiguous in the query dtype.
    pub fn flash_attn(
        &self,
        k: &Self,
        v: &Self,
        geom: &AttnGeometry,
        p: &FlashAttnParams,
        extra: &FaExtra<'_, Self>,
    ) -> Result<Self> {
        if k.dtype != v.dtype {
            bail!(
                "flash-attn: k {:?} and v {:?} dtypes differ",
                k.dtype,
                v.dtype
            )
        }
        let prof = self.device.accel_profile();
        let plan = att::plan(geom, p, prof.max_shared_mem, prof.compute_units);
        let name = att::kernel_name(&plan, self.dtype, k.dtype)?;
        let rows = geom.out_rows();
        let d = geom.d;
        let out = self.device.alloc(rows * d * self.esz())?;
        if rows == 0 || d == 0 {
            return Ok(Self::new(out, self.dtype, self.device.clone()));
        }
        let func = self.device.get_func(Module::FlashAttn, &name)?;
        let (grid, smem, splits) = att::launch_dims(geom, &plan);
        let part = if splits > 1 {
            Some((
                self.device.alloc(splits * rows * d * 4)?,
                self.device.alloc(splits * rows * 4)?,
            ))
        } else {
            None
        };
        let (alibi, alibi_bs) = match extra.alibi {
            Some((s, off, bs)) => (s.ptr_at(off), bs),
            None => (null(), 0),
        };
        let (cu_q, cu_k) = match extra.cu_seqlens {
            Some((sq, oq, sk, ok)) => (sq.ptr_at(oq), sk.ptr_at(ok)),
            None => (null(), null()),
        };
        let mut a = LaunchArgs::new();
        a.ptr(self.ptr_at(geom.q_offset))
            .ptr(k.ptr_at(geom.k_offset))
            .ptr(v.ptr_at(geom.v_offset))
            .ptr(out.as_ptr())
            .ptr(part.as_ref().map(|p| p.0.as_ptr()).unwrap_or(null()))
            .ptr(part.as_ref().map(|p| p.1.as_ptr()).unwrap_or(null()))
            .ptr(alibi)
            .ptr(cu_q)
            .ptr(cu_k)
            .u32(geom.lq as u32)
            .u32(geom.lk as u32)
            .u32(geom.hq as u32)
            .u32(geom.hkv as u32)
            .u32(d as u32);
        for s in [geom.q_strides, geom.k_strides, geom.v_strides] {
            a.usize(s[0]).usize(s[1]).usize(s[2]);
        }
        a.f32(p.softmax_scale)
            .f32(p.softcap)
            .u32(p.window_left as u32)
            .u32(p.window_right as u32)
            .f32(p.k_descale)
            .f32(p.v_descale)
            .u32(splits as u32)
            .u32(alibi_bs as u32)
            .u32(geom.varlen as u32)
            .usize(rows);
        self.device
            .launch(func, grid, (FA_THREADS as u32, 1, 1), smem as u32, &mut a)?;
        if let Some((po, pl)) = part {
            let sfx = half_sfx(self.dtype, "flash-attn")?;
            let comb = self
                .device
                .get_func(Module::FlashAttn, &format!("fa_combine_{sfx}"))?;
            let mut a = LaunchArgs::new();
            a.ptr(po.as_ptr())
                .ptr(pl.as_ptr())
                .ptr(out.as_ptr())
                .usize(rows)
                .u32(d as u32)
                .u32(splits as u32);
            self.device.launch(
                comb,
                (grid_rows(rows), 1, 1),
                (FA_THREADS as u32, 1, 1),
                0,
                &mut a,
            )?;
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    /// Packs `a.rows` contiguous rows starting at element `offset` of `self` (f32 / f16 / bf16)
    /// into U32 rows. `tab` holds the codebook table (32 f32) at element `tab_off`.
    pub fn tq_quantize(
        &self,
        offset: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        let sfx = half_sfx(self.dtype, "turbo-quant")?;
        let out = self.device.alloc(a.rows * a.row_words * 4)?;
        if a.rows > 0 {
            let func = self
                .device
                .get_func(Module::TurboQuant, &format!("tq_quant_{sfx}"))?;
            let mut l = LaunchArgs::new();
            l.ptr(self.ptr_at(offset))
                .ptr(out.as_ptr())
                .ptr(tab.ptr_at(tab_off))
                .usize(a.rows)
                .u32(a.dim as u32)
                .usize(a.dim)
                .u32(a.mode)
                .u32(a.code_bits)
                .u32(a.code_words as u32)
                .u32(a.aux_words as u32)
                .u32(a.row_words as u32);
            self.device.launch(
                func,
                (grid_rows(a.rows), 1, 1),
                (TQ_THREADS as u32, 1, 1),
                a.quant_smem_bytes() as u32,
                &mut l,
            )?;
        }
        Ok(Self::new(out, DType::U32, self.device.clone()))
    }

    /// Residual of the unit vectors to their codebook values (TurboQuant prod), F32 rows.
    pub fn tq_residual(
        &self,
        offset: usize,
        data: &Self,
        data_off: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        let sfx = half_sfx(self.dtype, "turbo-quant")?;
        let n = a.rows * a.dim;
        let out = self.device.alloc(n * 4)?;
        if n > 0 {
            let func = self
                .device
                .get_func(Module::TurboQuant, &format!("tq_resid_{sfx}"))?;
            let mut l = LaunchArgs::new();
            l.ptr(self.ptr_at(offset))
                .ptr(data.ptr_at(data_off))
                .ptr(tab.ptr_at(tab_off))
                .ptr(out.as_ptr())
                .usize(a.rows)
                .u32(a.dim as u32)
                .usize(a.dim)
                .u32(a.code_bits)
                .u32(a.row_words as u32)
                .u32(a.rsc_offset() as u32);
            self.device.launch(
                func,
                (grid_elems(n), 1, 1),
                (TQ_THREADS as u32, 1, 1),
                0,
                &mut l,
            )?;
        }
        Ok(Self::new(out, DType::F32, self.device.clone()))
    }

    /// Copy of the packed rows `self` with the QJL sign bits of `proj` and the residual norms.
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
        let out = self.device.alloc(a.rows * a.row_words * 4)?;
        if a.rows > 0 {
            let func = self.device.get_func(Module::TurboQuant, "tq_signs")?;
            let mut l = LaunchArgs::new();
            l.ptr(self.ptr_at(data_off))
                .ptr(proj.ptr_at(proj_off))
                .ptr(resid.ptr_at(resid_off))
                .ptr(out.as_ptr())
                .usize(a.rows)
                .u32(a.dim as u32)
                .u32(a.code_words as u32)
                .u32(a.row_words as u32);
            self.device.launch(
                func,
                (grid_rows(a.rows), 1, 1),
                (TQ_THREADS as u32, 1, 1),
                (TQ_THREADS * 4) as u32,
                &mut l,
            )?;
        }
        Ok(Self::new(out, DType::U32, self.device.clone()))
    }

    /// Dequantizes packed rows (rotated domain) into F32 `rows * dim` values, followed by the
    /// QJL coefficients when `with_qjl`.
    pub fn tq_dequantize(
        &self,
        data_off: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
        with_qjl: bool,
    ) -> Result<Self> {
        let n = a.rows * a.dim;
        let parts = if with_qjl { 2 } else { 1 };
        let out = self.device.alloc(parts * n * 4)?;
        if n > 0 {
            let func = self.device.get_func(Module::TurboQuant, "tq_dequant")?;
            let zs = if with_qjl {
                unsafe { out.as_ptr().add(n * 4) }
            } else {
                null()
            };
            let mut l = LaunchArgs::new();
            l.ptr(self.ptr_at(data_off))
                .ptr(tab.ptr_at(tab_off))
                .ptr(out.as_ptr())
                .ptr(zs)
                .usize(a.rows)
                .u32(a.dim as u32)
                .u32(a.mode)
                .u32(a.code_bits)
                .u32(a.code_words as u32)
                .u32(a.row_words as u32)
                .u32(a.rsc_offset() as u32);
            self.device.launch(
                func,
                (grid_elems(n), 1, 1),
                (TQ_THREADS as u32, 1, 1),
                0,
                &mut l,
            )?;
        }
        Ok(Self::new(out, DType::F32, self.device.clone()))
    }

    /// Attention of the rotated query rows `self` (F32) over packed K/V rows. Output: F32
    /// `(b, hq, lq, d)` in the rotated domain.
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
        if self.dtype != DType::F32 {
            bail!(
                "turbo-quant attention: queries must be f32, got {:?}",
                self.dtype
            )
        }
        let rows = a.out_rows();
        let d = a.d;
        let out = self.device.alloc(rows * d * 4)?;
        if rows == 0 || d == 0 {
            return Ok(Self::new(out, DType::F32, self.device.clone()));
        }
        let splits = a.splits.max(1);
        let part = if splits > 1 {
            Some((
                self.device.alloc(splits * rows * d * 4)?,
                self.device.alloc(splits * rows * 4)?,
            ))
        } else {
            None
        };
        let func = self.device.get_func(Module::TurboQuant, "tq_attn")?;
        let mut l = LaunchArgs::new();
        l.ptr(self.ptr_at(a.q_offset))
            .ptr(kd.ptr_at(a.k_offset))
            .ptr(vd.ptr_at(a.v_offset))
            .ptr(tab.ptr_at(tab_off))
            .ptr(alibi.map(|(s, o)| s.ptr_at(o)).unwrap_or(null()))
            .ptr(out.as_ptr())
            .ptr(part.as_ref().map(|p| p.0.as_ptr()).unwrap_or(null()))
            .ptr(part.as_ref().map(|p| p.1.as_ptr()).unwrap_or(null()))
            .u32(a.lq as u32)
            .u32(a.lk as u32)
            .u32(a.hq as u32)
            .u32(a.hkv as u32)
            .u32(d as u32)
            .usize(a.q_strides[0])
            .usize(a.q_strides[1])
            .usize(a.q_strides[2])
            .usize(a.k_strides[0])
            .usize(a.k_strides[1])
            .usize(a.k_strides[2])
            .usize(a.v_strides[0])
            .usize(a.v_strides[1])
            .usize(a.v_strides[2])
            .u32(a.k.mode)
            .u32(a.k.code_bits)
            .u32(a.k.code_words as u32)
            .u32(a.k.rsc_offset as u32)
            .u32(a.v.mode)
            .u32(a.v.code_bits)
            .u32(a.v.code_words as u32)
            .u32(a.v.rsc_offset as u32)
            .f32(a.scale)
            .f32(a.softcap)
            .u32(a.window_left as u32)
            .u32(a.window_right as u32)
            .u32(splits as u32)
            .u32(a.alibi_bstride as u32)
            .usize(rows);
        let grid = (
            (a.lq * a.hq).clamp(1, MAX_GRID_X) as u32,
            splits as u32,
            a.b as u32,
        );
        self.device.launch(
            func,
            grid,
            (TQ_THREADS as u32, 1, 1),
            a.smem_bytes() as u32,
            &mut l,
        )?;
        if let Some((po, pl)) = part {
            let comb = self.device.get_func(Module::TurboQuant, "tq_combine")?;
            let mut l = LaunchArgs::new();
            l.ptr(po.as_ptr())
                .ptr(pl.as_ptr())
                .ptr(out.as_ptr())
                .usize(rows)
                .u32(d as u32)
                .u32(splits as u32);
            self.device.launch(
                comb,
                (grid_rows(rows), 1, 1),
                (TQ_THREADS as u32, 1, 1),
                0,
                &mut l,
            )?;
        }
        Ok(Self::new(out, DType::F32, self.device.clone()))
    }
}
