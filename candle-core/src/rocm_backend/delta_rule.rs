//! Fused Gated DeltaNet kernels on the ROCm backend (`kernels/delta_rule.hip`). Planning and
//! argument layout are shared with the CUDA backend in `crate::accel::delta`.

use super::kernels::Module;
use super::utils::LaunchArgs;
use super::RocmStorage;
use crate::accel::delta::{self as gd, GdnScanArgs, GNORM_THREADS};
use crate::{bail, DType, Result};

fn null() -> *const u8 {
    std::ptr::null()
}

fn sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("gated delta rule: unsupported dtype {d:?} on the ROCm backend"),
    }
}

fn esz(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        _ => 2,
    }
}

impl RocmStorage {
    /// Fused Gated DeltaNet layer step, see `crate::delta_rule::fused_gated_delta_rule` and
    /// `CudaStorage::gdn_scan`. Returns (out, conv_state, recurrent_state).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan(
        &self,
        a: &Self,
        b: &Self,
        conv_w: &Self,
        conv_b: Option<&Self>,
        conv_state: Option<&Self>,
        neg_a: &Self,
        dt_bias: &Self,
        state: Option<&Self>,
        p: &GdnScanArgs,
    ) -> Result<(Self, Self, Self)> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let Some(tile) = gd::scan_tile(p.dk) else {
            bail!("gated delta rule: head_k_dim {} is not supported by the fused kernel", p.dk)
        };
        let c = p.conv_dim();
        let km1 = p.kc.saturating_sub(1);
        let dev = &self.device;
        let out = dev.alloc((p.b * p.t * p.hv * p.dv).max(1) * esz(dtype))?;
        let cs_out = dev.alloc((p.b * c * km1).max(1) * esz(dtype))?;
        let s_out = dev.alloc((p.b * p.hv * p.dk * p.dv).max(1) * 4)?;
        if p.b > 0 && p.hv > 0 && p.dv > 0 {
            let func = dev.get_func(Module::DeltaRule, &tile.kernel_name(sfx))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.ptr_at(p.x[0]))
                .ptr(conv_state.map(|s| s.ptr_at(p.cs_off)).unwrap_or(null()))
                .ptr(conv_w.ptr_at(p.cw_off))
                .ptr(conv_b.map(|s| s.ptr_at(p.cb_off)).unwrap_or(null()))
                .ptr(a.ptr_at(p.a[0]))
                .ptr(b.ptr_at(p.bb[0]))
                .ptr(neg_a.ptr_at(p.na_off))
                .ptr(dt_bias.ptr_at(p.dtb_off))
                .ptr(state.map(|s| s.ptr_at(p.s_off)).unwrap_or(null()))
                .ptr(out.as_ptr())
                .ptr(cs_out.as_ptr())
                .ptr(s_out.as_ptr())
                .u32(p.t as u32)
                .u32(p.hk as u32)
                .u32(p.hv as u32)
                .u32(p.dk as u32)
                .u32(p.dv as u32)
                .u32(p.kc as u32)
                .u32(p.flags)
                .f32(p.eps)
                .f32(p.scale)
                .usize(p.x[1])
                .usize(p.x[2])
                .usize(p.a[1])
                .usize(p.a[2])
                .usize(p.bb[1])
                .usize(p.bb[2]);
            dev.launch(
                func,
                tile.grid(p.b, p.hv, p.dv),
                (tile.nt as u32, 1, 1),
                tile.smem_bytes() as u32,
                &mut la,
            )?;
        }
        Ok((
            Self::new(out, dtype, dev.clone()),
            Self::new(cs_out, dtype, dev.clone()),
            Self::new(s_out, DType::F32, dev.clone()),
        ))
    }

    /// Gated RMS norm, see `CudaStorage::gdn_gated_rms_norm`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gated_rms_norm(
        &self,
        x_off: usize,
        z: &Self,
        z_off: usize,
        w: &Self,
        w_off: usize,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<Self> {
        let dtype = self.dtype;
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let out = dev.alloc((rows * d).max(1) * esz(dtype))?;
        if rows > 0 && d > 0 {
            let func = dev.get_func(Module::DeltaRule, &format!("gdn_gnorm_{sfx}"))?;
            let mut la = LaunchArgs::new();
            la.ptr(self.ptr_at(x_off))
                .ptr(z.ptr_at(z_off))
                .ptr(w.ptr_at(w_off))
                .ptr(out.as_ptr())
                .usize(rows)
                .u32(d as u32)
                .f32(eps);
            dev.launch(
                func,
                (rows.div_ceil(4).clamp(1, 1 << 20) as u32, 1, 1),
                (GNORM_THREADS as u32, 1, 1),
                (GNORM_THREADS * 4) as u32,
                &mut la,
            )?;
        }
        Ok(Self::new(out, dtype, dev.clone()))
    }
}
