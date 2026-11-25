//! Dedicated Vision Encoder fused kernels on the ROCm backend (`kernels/vis_ops.hip`).

use super::kernels::Module;
use super::utils::LaunchArgs;
use super::RocmStorage;
use crate::accel::vis_ops::{VIS_LN_NT, VIS_LN_SMEM_BYTES, VIS_QKV_NT};
use crate::{bail, DType, Result};

fn null() -> *const u8 {
    std::ptr::null()
}

fn sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("vis_ops: unsupported dtype {d:?} on ROCm"),
    }
}

fn esz(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        _ => 2,
    }
}

impl RocmStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn vis_u8_resize_norm_patchify(
        &self,
        out_dtype: DType,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        patch_size: usize,
        bicubic: bool,
        chw_in: bool,
        mean: [f32; 3],
        inv_std: [f32; 3],
    ) -> Result<Self> {
        let sfx = sfx(out_dtype)?;
        let dev = &self.device;
        let total_px = h_out * w_out;
        let out = dev.alloc((total_px * 3).max(1) * esz(out_dtype))?;
        if total_px > 0 {
            let func = dev.get_func(
                Module::VisOps,
                &format!("vis_u8_resize_norm_patchify_{sfx}"),
            )?;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(out.as_mut_ptr())
                .u32(h_in as u32)
                .u32(w_in as u32)
                .u32(h_out as u32)
                .u32(w_out as u32)
                .u32(patch_size as u32)
                .u32(u32::from(bicubic))
                .u32(u32::from(chw_in))
                .f32(mean[0])
                .f32(mean[1])
                .f32(mean[2])
                .f32(inv_std[0])
                .f32(inv_std[1])
                .f32(inv_std[2]);
            dev.launch(
                func,
                (total_px.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut la,
            )?;
        }
        Ok(Self::new(out, out_dtype, dev.clone()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vis_res_layernorm_ls(
        &self,
        delta: Option<&Self>,
        ls_gamma: Option<&Self>,
        ln_weight: Option<&Self>,
        ln_bias: Option<&Self>,
        out_dtype: DType,
        rows: usize,
        d: usize,
        eps: f32,
        write_residual: bool,
    ) -> Result<(Option<Self>, Self)> {
        let res_dtype = self.dtype;
        let x_sfx = sfx(out_dtype)?;
        let fname = if res_dtype == DType::F32 && out_dtype != DType::F32 {
            format!("vis_res_ln_ls_f32res_{x_sfx}")
        } else if res_dtype == out_dtype {
            format!("vis_res_ln_ls_{x_sfx}")
        } else {
            bail!(
                "vis_res_layernorm_ls: residual dtype {res_dtype:?} incompatible with out {out_dtype:?}"
            )
        };
        let dev = &self.device;
        let norm_buf = dev.alloc((rows * d).max(1) * esz(out_dtype))?;
        let res_buf = if write_residual {
            Some(dev.alloc((rows * d).max(1) * esz(res_dtype))?)
        } else {
            None
        };
        if rows > 0 && d > 0 {
            let func = dev.get_func(Module::VisOps, &fname)?;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(delta.map(|s| s.device_ptr()).unwrap_or(null()))
                .ptr(ls_gamma.map(|s| s.device_ptr()).unwrap_or(null()))
                .ptr(ln_weight.map(|s| s.device_ptr()).unwrap_or(null()))
                .ptr(ln_bias.map(|s| s.device_ptr()).unwrap_or(null()))
                .ptr(res_buf.as_ref().map(|b| b.as_ptr()).unwrap_or(null()))
                .ptr(norm_buf.as_ptr())
                .usize(rows)
                .u32(d as u32)
                .f32(eps);
            dev.launch(
                func,
                (rows as u32, 1, 1),
                (VIS_LN_NT as u32, 1, 1),
                VIS_LN_SMEM_BYTES as u32,
                &mut la,
            )?;
        }
        Ok((
            res_buf.map(|b| Self::new(b, res_dtype, dev.clone())),
            Self::new(norm_buf, out_dtype, dev.clone()),
        ))
    }

    pub fn vis_bias_gelu(
        &self,
        bias: Option<&Self>,
        n_elems: usize,
        dim: usize,
        mode: u32,
    ) -> Result<Self> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let out = dev.alloc(n_elems.max(1) * esz(dtype))?;
        if n_elems > 0 {
            let func = dev.get_func(Module::VisOps, &format!("vis_bias_gelu_{sfx}"))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(bias.map(|b| b.device_ptr()).unwrap_or(null()))
                .ptr(out.as_mut_ptr())
                .usize(n_elems)
                .u32(dim as u32)
                .u32(mode);
            dev.launch(
                func,
                (n_elems.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut la,
            )?;
        }
        Ok(Self::new(out, dtype, dev.clone()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vis_qkv_2d_rope(
        &self,
        bias: Option<&Self>,
        q_norm_w: Option<&Self>,
        k_norm_w: Option<&Self>,
        patch_yx: &Self,
        inv_freq: &Self,
        n_tokens: usize,
        num_heads: usize,
        head_dim: usize,
        rot_dim: usize,
        flags: u32,
        eps: f32,
    ) -> Result<(Self, Self, Self)> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let out_bytes = (num_heads * n_tokens * head_dim).max(1) * esz(dtype);
        let q_out = dev.alloc(out_bytes)?;
        let k_out = dev.alloc(out_bytes)?;
        let v_out = dev.alloc(out_bytes)?;
        if n_tokens > 0 && num_heads > 0 && head_dim > 0 {
            let func = dev.get_func(Module::VisOps, &format!("vis_qkv_2d_rope_{sfx}"))?;
            let smem = (2 * head_dim + VIS_QKV_NT) * 4;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(bias.map(|b| b.device_ptr()).unwrap_or(null()))
                .ptr(q_norm_w.map(|w| w.device_ptr()).unwrap_or(null()))
                .ptr(k_norm_w.map(|w| w.device_ptr()).unwrap_or(null()))
                .ptr(patch_yx.device_ptr())
                .ptr(inv_freq.device_ptr())
                .ptr(q_out.as_mut_ptr())
                .ptr(k_out.as_mut_ptr())
                .ptr(v_out.as_mut_ptr())
                .u32(n_tokens as u32)
                .u32(num_heads as u32)
                .u32(head_dim as u32)
                .u32(rot_dim as u32)
                .u32(flags)
                .f32(eps);
            dev.launch(
                func,
                (n_tokens as u32, num_heads as u32, 1),
                (VIS_QKV_NT as u32, 1, 1),
                smem as u32,
                &mut la,
            )?;
        }
        Ok((
            Self::new(q_out, dtype, dev.clone()),
            Self::new(k_out, dtype, dev.clone()),
            Self::new(v_out, dtype, dev.clone()),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vis_spatial_merge_norm(
        &self,
        ln_weight: Option<&Self>,
        ln_bias: Option<&Self>,
        h_p: usize,
        w_p: usize,
        d: usize,
        merge: usize,
        mode: u32,
        eps: f32,
    ) -> Result<Self> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let out_toks = (h_p / merge) * (w_p / merge);
        let out_dim = if mode == 2 { d } else { merge * merge * d };
        let out = dev.alloc((out_toks * out_dim).max(1) * esz(dtype))?;
        if out_toks > 0 && d > 0 {
            let func = dev.get_func(Module::VisOps, &format!("vis_spatial_merge_norm_{sfx}"))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(ln_weight.map(|w| w.device_ptr()).unwrap_or(null()))
                .ptr(ln_bias.map(|b| b.device_ptr()).unwrap_or(null()))
                .ptr(out.as_mut_ptr())
                .u32(h_p as u32)
                .u32(w_p as u32)
                .u32(d as u32)
                .u32(merge as u32)
                .u32(mode)
                .f32(eps);
            dev.launch(
                func,
                (out_toks as u32, 1, 1),
                (VIS_LN_NT as u32, 1, 1),
                VIS_LN_SMEM_BYTES as u32,
                &mut la,
            )?;
        }
        Ok(Self::new(out, dtype, dev.clone()))
    }

    pub fn vis_scatter_embeds_inplace(
        &mut self,
        vis: &Self,
        indices: &Self,
        n_vis: usize,
        d: usize,
        l_total: usize,
    ) -> Result<()> {
        let sfx = sfx(self.dtype)?;
        let dev = &self.device;
        if n_vis > 0 && d > 0 {
            let func = dev.get_func(Module::VisOps, &format!("vis_scatter_embeds_{sfx}"))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(vis.device_ptr())
                .ptr(indices.device_ptr())
                .u32(n_vis as u32)
                .u32(d as u32)
                .u32(l_total as u32);
            dev.launch(func, (n_vis as u32, 1, 1), (256, 1, 1), 0, &mut la)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vis_patch_add_interp_pos2d_inplace(
        &mut self,
        pos_table: &Self,
        g_h: usize,
        g_w: usize,
        h_p: usize,
        w_p: usize,
        d: usize,
        bicubic: bool,
        align_corners: bool,
    ) -> Result<()> {
        let sfx = sfx(self.dtype)?;
        let dev = &self.device;
        let n_toks = h_p * w_p;
        if n_toks > 0 && d > 0 {
            let func = dev.get_func(
                Module::VisOps,
                &format!("vis_patch_add_interp_pos2d_{sfx}"),
            )?;
            let mut la = LaunchArgs::new();
            la.ptr(self.device_ptr())
                .ptr(pos_table.device_ptr())
                .u32(g_h as u32)
                .u32(g_w as u32)
                .u32(h_p as u32)
                .u32(w_p as u32)
                .u32(d as u32)
                .u32(u32::from(bicubic))
                .u32(u32::from(align_corners));
            dev.launch(func, (n_toks as u32, 1, 1), (256, 1, 1), 0, &mut la)?;
        }
        Ok(())
    }
}
