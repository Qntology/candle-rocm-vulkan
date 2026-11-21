//! Fused decoder kernels (`ResidualRmsNorm`, `SiluMul`, `RopeKvWrite`) on the ROCm / HIP backend
//! (`kernels/decoder_ops.hip`). Planning is shared with CUDA in `crate::accel::decoder_ops`.

use super::kernels::Module;
use super::utils::LaunchArgs;
use super::RocmStorage;
use crate::accel::decoder_ops::{
    self as dop, RopeKvLaunchArgs, DEC_NT, RES_RMS_PLUS_ONE_WEIGHT, RES_RMS_WRITE_RESIDUAL,
};
use crate::{bail, DType, Result};

fn null() -> *const u8 {
    std::ptr::null()
}

fn sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("decoder_ops: unsupported dtype {d:?} on the ROCm backend"),
    }
}

fn esz(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        _ => 2,
    }
}

impl RocmStorage {
    /// Out-of-place `ResidualRmsNorm`: returns `(residual, normed)`.
    #[allow(clippy::too_many_arguments)]
    pub fn res_rmsnorm(
        &self,
        x_off: usize,
        delta: Option<(&Self, usize)>,
        w: &Self,
        w_off: usize,
        rows: usize,
        d: usize,
        eps: f32,
        plus_one: bool,
        write_residual: bool,
    ) -> Result<(Option<Self>, Self)> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let r_buf = if write_residual {
            Some(dev.alloc((rows * d).max(1) * esz(dtype))?)
        } else {
            None
        };
        let y_buf = dev.alloc((rows * d).max(1) * esz(dtype))?;
        if rows > 0 && d > 0 {
            let func = dev.get_func(Module::DecoderOps, &format!("res_rmsnorm_{sfx}"))?;
            let mut flags = 0u32;
            if plus_one {
                flags |= RES_RMS_PLUS_ONE_WEIGHT;
            }
            if write_residual {
                flags |= RES_RMS_WRITE_RESIDUAL;
            }
            let mut la = LaunchArgs::new();
            la.ptr(self.ptr_at(x_off))
                .ptr(delta.map(|(ds, off)| ds.ptr_at(off)).unwrap_or(null()))
                .ptr(w.ptr_at(w_off))
                .ptr(r_buf.as_ref().map(|b| b.as_ptr()).unwrap_or(null()))
                .ptr(y_buf.as_ptr())
                .u32(rows as u32)
                .u32(d as u32)
                .f32(eps)
                .u32(flags);
            dev.launch(
                func,
                dop::res_rmsnorm_grid(rows),
                (DEC_NT, 1, 1),
                dop::res_rmsnorm_smem_bytes(),
                &mut la,
            )?;
        }
        Ok((
            r_buf.map(|b| Self::new(b, dtype, dev.clone())),
            Self::new(y_buf, dtype, dev.clone()),
        ))
    }

    /// In-place `ResidualRmsNorm`: updates `self` with `x + delta` and returns `normed`.
    #[allow(clippy::too_many_arguments)]
    pub fn res_rmsnorm_inplace(
        &mut self,
        delta: Option<(&Self, usize)>,
        w: &Self,
        w_off: usize,
        rows: usize,
        d: usize,
        eps: f32,
        plus_one: bool,
    ) -> Result<Self> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = self.device.clone();
        let y_buf = dev.alloc((rows * d).max(1) * esz(dtype))?;
        if rows > 0 && d > 0 {
            let func = dev.get_func(Module::DecoderOps, &format!("res_rmsnorm_{sfx}"))?;
            let flags = RES_RMS_WRITE_RESIDUAL | if plus_one { RES_RMS_PLUS_ONE_WEIGHT } else { 0 };
            let x_ptr = self.ptr_at(0);
            let mut la = LaunchArgs::new();
            la.ptr(x_ptr)
                .ptr(delta.map(|(ds, off)| ds.ptr_at(off)).unwrap_or(null()))
                .ptr(w.ptr_at(w_off))
                .ptr(x_ptr)
                .ptr(y_buf.as_ptr())
                .u32(rows as u32)
                .u32(d as u32)
                .f32(eps)
                .u32(flags);
            dev.launch(
                func,
                dop::res_rmsnorm_grid(rows),
                (DEC_NT, 1, 1),
                dop::res_rmsnorm_smem_bytes(),
                &mut la,
            )?;
        }
        Ok(Self::new(y_buf, dtype, dev))
    }

    /// Fused SwiGLU `silu(gate) * up` supporting strided `gate_up` views.
    #[allow(clippy::too_many_arguments)]
    pub fn silu_mul(
        &self,
        gate_off: usize,
        gate_row_stride: usize,
        up: &Self,
        up_off: usize,
        up_row_stride: usize,
        rows: usize,
        inter_dim: usize,
    ) -> Result<Self> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let out_buf = dev.alloc((rows * inter_dim).max(1) * esz(dtype))?;
        if rows > 0 && inter_dim > 0 {
            let func = dev.get_func(Module::DecoderOps, &format!("silu_mul_{sfx}"))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.ptr_at(gate_off))
                .usize(gate_row_stride)
                .ptr(up.ptr_at(up_off))
                .usize(up_row_stride)
                .ptr(out_buf.as_ptr())
                .u32(rows as u32)
                .u32(inter_dim as u32);
            dev.launch(
                func,
                dop::silu_mul_grid(rows, inter_dim),
                (DEC_NT, 1, 1),
                0,
                &mut la,
            )?;
        }
        Ok(Self::new(out_buf, dtype, dev.clone()))
    }

    /// Fused QK Head RMSNorm + RoPE + KV Cache Direct Write on ROCm.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv_write(
        &self,
        k_in: &Self,
        v_in: &Self,
        q_norm_w: Option<&Self>,
        k_norm_w: Option<&Self>,
        cos_tab: &Self,
        sin_tab: &Self,
        k_cache: &mut Self,
        v_cache: &mut Self,
        args: &RopeKvLaunchArgs,
    ) -> Result<Self> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let q_elems = (args.b as usize) * (args.t as usize) * (args.hq as usize) * (args.d as usize);
        let q_out = dev.alloc(q_elems.max(1) * esz(dtype))?;
        if args.b > 0 && args.t > 0 && (args.hq + args.hkv) > 0 && args.d > 0 {
            let func = dev.get_func(Module::DecoderOps, &format!("rope_kv_write_{sfx}"))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.ptr_at(args.q_off))
                .ptr(q_out.as_ptr())
                .ptr(k_in.ptr_at(args.k_off))
                .ptr(v_in.ptr_at(args.v_off))
                .ptr(q_norm_w.map(|w| w.ptr_at(args.qnw_off)).unwrap_or(null()))
                .ptr(k_norm_w.map(|w| w.ptr_at(args.knw_off)).unwrap_or(null()))
                .ptr(cos_tab.ptr_at(args.cos_off))
                .ptr(sin_tab.ptr_at(args.sin_off))
                .ptr(k_cache.ptr_at(args.kc_off))
                .ptr(v_cache.ptr_at(args.vc_off))
                .u32(args.b)
                .u32(args.t)
                .u32(args.hq)
                .u32(args.hkv)
                .u32(args.d)
                .u32(args.rotary_dim)
                .u32(args.past_len)
                .usize(args.q_strides[0] as usize)
                .usize(args.q_strides[1] as usize)
                .usize(args.q_strides[2] as usize)
                .usize(args.qo_strides[0] as usize)
                .usize(args.qo_strides[1] as usize)
                .usize(args.qo_strides[2] as usize)
                .usize(args.k_strides[0] as usize)
                .usize(args.k_strides[1] as usize)
                .usize(args.k_strides[2] as usize)
                .usize(args.v_strides[0] as usize)
                .usize(args.v_strides[1] as usize)
                .usize(args.v_strides[2] as usize)
                .usize(args.kc_strides[0] as usize)
                .usize(args.kc_strides[1] as usize)
                .usize(args.kc_strides[2] as usize)
                .usize(args.cos_strides[0] as usize)
                .usize(args.cos_strides[1] as usize)
                .f32(args.eps)
                .u32(args.flags);
            dev.launch(func, args.grid(), args.block(), args.smem_bytes(), &mut la)?;
        }
        Ok(Self::new(q_out, dtype, dev.clone()))
    }
}
