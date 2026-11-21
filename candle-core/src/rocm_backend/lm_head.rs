//! Fused `lm_head` Tile Argmax + Mean-Centered CS Guard kernels on the ROCm / HIP backend
//! (`kernels/lm_head_argmax.hip`). Planning is shared with CUDA in `crate::accel::lm_head`.

use super::kernels::Module;
use super::utils::LaunchArgs;
use super::RocmStorage;
use crate::accel::lm_head::{self as lmh, LM_NT};
use crate::{bail, DType, Result};

fn null() -> *const u8 {
    std::ptr::null()
}

fn float_sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("lm_head_argmax: unsupported dtype {d:?} on the ROCm backend"),
    }
}

impl RocmStorage {
    /// Mode A: 2-Stage Dense Fused Tile Argmax + Mean-Centered CS Guard on ROCm.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_linear_argmax_cs(
        &self,
        x_off: usize,
        w: &Self,
        w_off: usize,
        bias: Option<(&Self, usize)>,
        core_ids: Option<(&Self, usize)>,
        mu_rest: Option<(&Self, usize)>,
        w_rest_bound: f32,
        b_rest_max: f32,
        check_cs: bool,
        m: usize,
        v_valid: usize,
        k: usize,
    ) -> Result<(Self, Self, Self)> {
        let dtype = self.dtype;
        let sfx = float_sfx(dtype)?;
        let dev = &self.device;

        let out_ids = dev.alloc(m.max(1) * 4)?;
        let out_vals = dev.alloc(m.max(1) * 4)?;
        let out_rescue = dev.alloc(m.max(1))?;

        if m > 0 && v_valid > 0 && k > 0 {
            let plan = lmh::plan_stage1(m, v_valid);
            let n_tiles = plan.n_tiles as usize;
            let pv_bytes = ((m * n_tiles).max(1) * 4).div_ceil(256) * 256;
            let pi_bytes = (m * n_tiles).max(1) * 4;
            let f1 = dev.get_func(Module::LmHeadArgmax, &plan.kernel_name(sfx))?;
            let f2 = dev.get_func(Module::LmHeadArgmax, &format!("lm_stage2_reduce_cs_{sfx}"))?;
            // Keep workspace locked across Stage 1 + Stage 2 queueing.
            let _ws = dev.workspace_ensure("lm_head_parts", pv_bytes + pi_bytes)?;
            let part_val = _ws.as_ptr();
            let part_idx = unsafe { _ws.as_ptr().add(pv_bytes) };

            let mut la1 = LaunchArgs::new();
            la1.ptr(self.ptr_at(x_off))
                .ptr(w.ptr_at(w_off))
                .ptr(bias.map(|(bs, off)| bs.ptr_at(off)).unwrap_or(null()))
                .ptr(part_val)
                .ptr(part_idx)
                .u32(m as u32)
                .u32(v_valid as u32)
                .u32(k as u32)
                .u32(plan.n_tiles);
            dev.launch(f1, plan.grid, plan.block, plan.smem_bytes, &mut la1)?;

            let mut la2 = LaunchArgs::new();
            la2.ptr(part_val)
                .ptr(part_idx)
                .ptr(core_ids.map(|(cs, off)| cs.ptr_at(off)).unwrap_or(null()))
                .ptr(self.ptr_at(x_off))
                .ptr(mu_rest.map(|(ms, off)| ms.ptr_at(off)).unwrap_or(null()))
                .f32(w_rest_bound)
                .f32(b_rest_max)
                .ptr(out_ids.as_ptr())
                .ptr(out_vals.as_ptr())
                .ptr(out_rescue.as_ptr())
                .u32(m as u32)
                .u32(plan.n_tiles)
                .u32(k as u32)
                .u32(u32::from(check_cs));
            dev.launch(
                f2,
                (m as u32, 1, 1),
                (LM_NT, 1, 1),
                lmh::stage2_smem_bytes(),
                &mut la2,
            )?;
        }

        Ok((
            Self::new(out_ids, DType::U32, dev.clone()),
            Self::new(out_vals, DType::F32, dev.clone()),
            Self::new(out_rescue, DType::U8, dev.clone()),
        ))
    }

    /// Quantized Mode A: 2-Stage Fused Tile Argmax + CS Guard directly over quantized weights
    /// (`Q8_0`, `Q4_0`, `Q4K`, `Q5K`, `Q6K`) with 0-byte `[M, V]` logits materialization.
    /// Supports arbitrary `M` by chunking 8 rows at a time (`Q5-O07`).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_quant_argmax_cs(
        &self,
        x_off: usize,
        wq_ptr: *const u8,
        q_sfx: &str,
        row_bytes: usize,
        bias: Option<(&Self, usize)>,
        core_ids: Option<(&Self, usize)>,
        mu_rest: Option<(&Self, usize)>,
        w_rest_bound: f32,
        b_rest_max: f32,
        check_cs: bool,
        m: usize,
        v_valid: usize,
        k: usize,
    ) -> Result<(Self, Self, Self)> {
        let dtype = self.dtype;
        let sfx = float_sfx(dtype)?;
        let dev = &self.device;
        let out_ids = dev.alloc(m.max(1) * 4)?;
        let out_vals = dev.alloc(m.max(1) * 4)?;
        let out_rescue = dev.alloc(m.max(1))?;

        if m > 0 && v_valid > 0 && k > 0 {
            let n_tiles = v_valid.div_ceil(32);
            let max_chunk = m.min(8);
            let pv_bytes = ((max_chunk * n_tiles).max(1) * 4).div_ceil(256) * 256;
            let pi_bytes = (max_chunk * n_tiles).max(1) * 4;

            let f2 = dev.get_func(Module::LmHeadArgmax, &format!("lm_stage2_reduce_cs_{sfx}"))?;
            // Keep workspace locked across all chunk Stage 1 + Stage 2 queueings.
            let _ws = dev.workspace_ensure("lm_head_parts", pv_bytes + pi_bytes)?;
            let part_val = _ws.as_ptr();
            let part_idx = unsafe { _ws.as_ptr().add(pv_bytes) };

            let mut row0 = 0usize;
            while row0 < m {
                let cm = (m - row0).min(8);
                let m_tag = if cm <= 1 { "1" } else { "8" };
                let f1_name = format!("w8a16_lm{m_tag}_{q_sfx}_{sfx}");
                let f1 = dev.get_func(Module::W8A16, &f1_name)?;
                let x_ptr = self.ptr_at(x_off + row0 * k);

                let mut la1 = LaunchArgs::new();
                la1.ptr(x_ptr)
                    .ptr(wq_ptr)
                    .ptr(bias.map(|(bs, off)| bs.ptr_at(off)).unwrap_or(null()))
                    .ptr(part_val)
                    .ptr(part_idx)
                    .u32(cm as u32)
                    .u32(v_valid as u32)
                    .u32(k as u32)
                    .u32(n_tiles as u32)
                    .bits(row_bytes as u64);
                dev.launch(f1, (n_tiles as u32, 1, 1), (32, 8, 1), 0, &mut la1)?;

                let mut la2 = LaunchArgs::new();
                la2.ptr(part_val)
                    .ptr(part_idx)
                    .ptr(core_ids.map(|(cs, off)| cs.ptr_at(off)).unwrap_or(null()))
                    .ptr(x_ptr)
                    .ptr(mu_rest.map(|(ms, off)| ms.ptr_at(off)).unwrap_or(null()))
                    .f32(w_rest_bound)
                    .f32(b_rest_max)
                    .ptr(unsafe { out_ids.as_ptr().add(row0 * 4) })
                    .ptr(unsafe { out_vals.as_ptr().add(row0 * 4) })
                    .ptr(unsafe { out_rescue.as_ptr().add(row0) })
                    .u32(cm as u32)
                    .u32(n_tiles as u32)
                    .u32(k as u32)
                    .u32(u32::from(check_cs));
                dev.launch(
                    f2,
                    (cm as u32, 1, 1),
                    (LM_NT, 1, 1),
                    lmh::stage2_smem_bytes(),
                    &mut la2,
                )?;
                row0 += cm;
            }
        }

        Ok((
            Self::new(out_ids, DType::U32, dev.clone()),
            Self::new(out_vals, DType::F32, dev.clone()),
            Self::new(out_rescue, DType::U8, dev.clone()),
        ))
    }

    /// Mode B: 1-Stage Sub-Logits Argmax + Core-ID Remap + CS Guard on ROCm.
    #[allow(clippy::too_many_arguments)]
    pub fn logits_argmax_cs(
        &self,
        logits_off: usize,
        logits_row_stride: usize,
        core_ids: Option<(&Self, usize)>,
        hidden: Option<(&Self, usize)>,
        mu_rest: Option<(&Self, usize)>,
        w_rest_bound: f32,
        b_rest_max: f32,
        check_cs: bool,
        m: usize,
        v_valid: usize,
        k: usize,
    ) -> Result<(Self, Self, Self)> {
        let l_sfx = float_sfx(self.dtype)?;
        let h_sfx = match hidden {
            Some((h, _)) => float_sfx(h.dtype)?,
            None => l_sfx,
        };
        let fn_name = format!("lm_logits_argmax_cs_{l_sfx}_{h_sfx}");
        let dev = &self.device;

        let out_ids = dev.alloc(m.max(1) * 4)?;
        let out_vals = dev.alloc(m.max(1) * 4)?;
        let out_rescue = dev.alloc(m.max(1))?;

        if m > 0 && v_valid > 0 {
            let func = dev.get_func(Module::LmHeadArgmax, &fn_name)?;
            let mut la = LaunchArgs::new();
            la.ptr(self.ptr_at(logits_off))
                .usize(logits_row_stride)
                .ptr(core_ids.map(|(cs, off)| cs.ptr_at(off)).unwrap_or(null()))
                .ptr(hidden.map(|(hs, off)| hs.ptr_at(off)).unwrap_or(null()))
                .ptr(mu_rest.map(|(ms, off)| ms.ptr_at(off)).unwrap_or(null()))
                .f32(w_rest_bound)
                .f32(b_rest_max)
                .ptr(out_ids.as_ptr())
                .ptr(out_vals.as_ptr())
                .ptr(out_rescue.as_ptr())
                .u32(m as u32)
                .u32(v_valid as u32)
                .u32(k as u32)
                .u32(u32::from(check_cs && hidden.is_some()));
            dev.launch(
                func,
                (m as u32, 1, 1),
                (LM_NT, 1, 1),
                lmh::stage2_smem_bytes(),
                &mut la,
            )?;
        }

        Ok((
            Self::new(out_ids, DType::U32, dev.clone()),
            Self::new(out_vals, DType::F32, dev.clone()),
            Self::new(out_rescue, DType::U8, dev.clone()),
        ))
    }
}
