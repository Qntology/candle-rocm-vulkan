//! Fused decoder kernels (`ResidualRmsNorm`, `SiluMul`, `RopeKvWrite`) on the CUDA backend.
//!
//! Compiled once on first use as a single NVRTC module (`candle_decoder_ops`) from
//! `kernels/decoder_ops_impl.h` with `--std=c++17` (strictly no `--use_fast_math`).

use super::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::accel::decoder_ops::{
    self as dop, RopeKvLaunchArgs, DEC_NT, RES_RMS_PLUS_ONE_WEIGHT, RES_RMS_WRITE_RESIDUAL,
};
use crate::accel::sources::decoder_ops_cuda_source;
use crate::{bail, DType, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg, SyncOnDrop};

const MODULE_NAME: &str = "candle_decoder_ops";

impl CudaDevice {
    pub(crate) fn decoder_op_func(&self, fn_name: &str) -> Result<super::device::CudaFunc> {
        self.get_or_compile_func(fn_name, MODULE_NAME, || {
            let src = decoder_ops_cuda_source();
            let target_arch = super::delta_rule::arch(self);
            let make_opts = |arch: Option<&'static str>| cudarc::nvrtc::CompileOptions {
                arch,
                // Strict precision contract: --use_fast_math is forbidden.
                options: vec!["--std=c++17".into()],
                ..Default::default()
            };
            match cudarc::nvrtc::safe::compile_ptx_with_opts(&src, make_opts(Some(target_arch))) {
                Ok(ptx) => Ok(ptx),
                Err(_) => cudarc::nvrtc::safe::compile_ptx_with_opts(&src, make_opts(None)).w(),
            }
        })
    }
}

fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("decoder_ops: unsupported dtype {d:?} on the CUDA backend"),
    }
}

fn dptr<'a>(
    s: &'a CudaStorage,
    elem_offset: usize,
    stream: &'a cudarc::driver::CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    macro_rules! p {
        ($x:expr, $t:ty) => {{
            let (ptr, g) = $x.device_ptr(stream);
            (ptr + (elem_offset * std::mem::size_of::<$t>()) as u64, g)
        }};
    }
    Ok(match &s.slice {
        CudaStorageSlice::F32(x) => p!(x, f32),
        CudaStorageSlice::F16(x) => p!(x, half::f16),
        CudaStorageSlice::BF16(x) => p!(x, half::bf16),
        _ => bail!("decoder_ops: unsupported dtype on the CUDA backend"),
    })
}

fn dptr_mut<'a>(
    s: &'a mut CudaStorageSlice,
    elem_offset: usize,
    stream: &'a cudarc::driver::CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    macro_rules! p {
        ($x:expr, $t:ty) => {{
            let (ptr, g) = $x.device_ptr_mut(stream);
            (ptr + (elem_offset * std::mem::size_of::<$t>()) as u64, g)
        }};
    }
    Ok(match s {
        CudaStorageSlice::F32(x) => p!(x, f32),
        CudaStorageSlice::F16(x) => p!(x, half::f16),
        CudaStorageSlice::BF16(x) => p!(x, half::bf16),
        _ => bail!("decoder_ops: unsupported output dtype on the CUDA backend"),
    })
}

fn alloc(dev: &CudaDevice, dtype: DType, n: usize) -> Result<CudaStorageSlice> {
    let n = n.max(1);
    // SAFETY: every element read later is written by the kernel first.
    unsafe {
        Ok(match dtype {
            DType::F32 => CudaStorageSlice::F32(dev.alloc::<f32>(n)?),
            DType::F16 => CudaStorageSlice::F16(dev.alloc::<half::f16>(n)?),
            DType::BF16 => CudaStorageSlice::BF16(dev.alloc::<half::bf16>(n)?),
            d => bail!("decoder_ops: unsupported dtype {d:?} on the CUDA backend"),
        })
    }
}

impl CudaStorage {
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = dtype_suffix(dtype)?;
        let dev = &self.device;
        let mut r_slice = if write_residual {
            Some(alloc(dev, dtype, rows * d)?)
        } else {
            None
        };
        let mut y_slice = alloc(dev, dtype, rows * d)?;
        if rows > 0 && d > 0 {
            let func = dev.decoder_op_func(&format!("res_rmsnorm_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g0) = dptr(self, x_off, &stream)?;
            let (dl_p, _g1) = match delta {
                Some((ds, off)) => {
                    let (p, g) = dptr(ds, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (w_p, _g2) = dptr(w, w_off, &stream)?;
            let (r_p, _g3) = match r_slice.as_mut() {
                Some(rs) => {
                    let (p, g) = dptr_mut(rs, 0, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (y_p, _g4) = dptr_mut(&mut y_slice, 0, &stream)?;
            let rows32 = rows as u32;
            let d32 = d as u32;
            let mut flags = 0u32;
            if plus_one {
                flags |= RES_RMS_PLUS_ONE_WEIGHT;
            }
            if write_residual {
                flags |= RES_RMS_WRITE_RESIDUAL;
            }
            let mut bld = func.builder();
            bld.arg(&x_p)
                .arg(&dl_p)
                .arg(&w_p)
                .arg(&r_p)
                .arg(&y_p)
                .arg(&rows32)
                .arg(&d32)
                .arg(&eps)
                .arg(&flags);
            let cfg = LaunchConfig {
                grid_dim: dop::res_rmsnorm_grid(rows),
                block_dim: (DEC_NT, 1, 1),
                shared_mem_bytes: dop::res_rmsnorm_smem_bytes(),
            };
            // SAFETY: arguments match res_rmsnorm_{sfx}.
            unsafe { bld.launch(cfg) }.w()?;
        }
        let mk = |slice| CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((r_slice.map(mk), mk(y_slice)))
    }

    /// In-place `ResidualRmsNorm`: updates `self` with `x + delta` and returns `normed`.
    /// Caller MUST guarantee `self` is uniquely owned (`is_sole_owner_contiguous`) and disjoint from `delta`.
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = dtype_suffix(dtype)?;
        let dev = self.device.clone();
        let mut y_slice = alloc(&dev, dtype, rows * d)?;
        if rows > 0 && d > 0 {
            let func = dev.decoder_op_func(&format!("res_rmsnorm_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g0) = dptr_mut(&mut self.slice, 0, &stream)?;
            let (dl_p, _g1) = match delta {
                Some((ds, off)) => {
                    let (p, g) = dptr(ds, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (w_p, _g2) = dptr(w, w_off, &stream)?;
            let (y_p, _g3) = dptr_mut(&mut y_slice, 0, &stream)?;
            let rows32 = rows as u32;
            let d32 = d as u32;
            let flags = RES_RMS_WRITE_RESIDUAL | if plus_one { RES_RMS_PLUS_ONE_WEIGHT } else { 0 };
            let mut bld = func.builder();
            bld.arg(&x_p)
                .arg(&dl_p)
                .arg(&w_p)
                .arg(&x_p)
                .arg(&y_p)
                .arg(&rows32)
                .arg(&d32)
                .arg(&eps)
                .arg(&flags);
            let cfg = LaunchConfig {
                grid_dim: dop::res_rmsnorm_grid(rows),
                block_dim: (DEC_NT, 1, 1),
                shared_mem_bytes: dop::res_rmsnorm_smem_bytes(),
            };
            // SAFETY: x_p == r_p is safe in res_rmsnorm_impl (load before store, barrier before pass 2).
            unsafe { bld.launch(cfg) }.w()?;
        }
        Ok(CudaStorage {
            slice: y_slice,
            device: dev,
        })
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = dtype_suffix(dtype)?;
        let dev = &self.device;
        let mut out_slice = alloc(dev, dtype, rows * inter_dim)?;
        if rows > 0 && inter_dim > 0 {
            let func = dev.decoder_op_func(&format!("silu_mul_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (g_p, _g0) = dptr(self, gate_off, &stream)?;
            let (u_p, _g1) = dptr(up, up_off, &stream)?;
            let (o_p, _g2) = dptr_mut(&mut out_slice, 0, &stream)?;
            let gs = gate_row_stride as u64;
            let us = up_row_stride as u64;
            let rows32 = rows as u32;
            let inter32 = inter_dim as u32;
            let mut bld = func.builder();
            bld.arg(&g_p)
                .arg(&gs)
                .arg(&u_p)
                .arg(&us)
                .arg(&o_p)
                .arg(&rows32)
                .arg(&inter32);
            let cfg = LaunchConfig {
                grid_dim: dop::silu_mul_grid(rows, inter_dim),
                block_dim: (DEC_NT, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: arguments match silu_mul_{sfx}.
            unsafe { bld.launch(cfg) }.w()?;
        }
        Ok(CudaStorage {
            slice: out_slice,
            device: dev.clone(),
        })
    }

    /// Fused QK Head RMSNorm + RoPE + KV Cache Direct Write.
    /// Writes rotated K and raw V directly into `k_cache` / `v_cache` at `past_len..past_len+T`,
    /// and returns rotated `q_out`.
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = dtype_suffix(dtype)?;
        let dev = &self.device;
        let q_elems = (args.b as usize) * (args.t as usize) * (args.hq as usize) * (args.d as usize);
        let mut q_out = alloc(dev, dtype, q_elems)?;
        if args.b > 0 && args.t > 0 && (args.hq + args.hkv) > 0 && args.d > 0 {
            let func = dev.decoder_op_func(&format!("rope_kv_write_{sfx}"))?;
            let smem = args.smem_bytes();
            if smem > 48 * 1024 {
                func.set_max_dynamic_shared_memory(smem as usize)?;
            }
            let stream = dev.cuda_stream();
            let (q_p, _g0) = dptr(self, args.q_off, &stream)?;
            let (qo_p, _g1) = dptr_mut(&mut q_out, 0, &stream)?;
            let (k_p, _g2) = dptr(k_in, args.k_off, &stream)?;
            let (v_p, _g3) = dptr(v_in, args.v_off, &stream)?;
            let (qnw_p, _g4) = match q_norm_w {
                Some(w) => {
                    let (p, g) = dptr(w, args.qnw_off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (knw_p, _g5) = match k_norm_w {
                Some(w) => {
                    let (p, g) = dptr(w, args.knw_off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (cos_p, _g6) = dptr(cos_tab, args.cos_off, &stream)?;
            let (sin_p, _g7) = dptr(sin_tab, args.sin_off, &stream)?;
            let (kc_p, _g8) = dptr_mut(&mut k_cache.slice, args.kc_off, &stream)?;
            let (vc_p, _g9) = dptr_mut(&mut v_cache.slice, args.vc_off, &stream)?;

            let mut bld = func.builder();
            bld.arg(&q_p)
                .arg(&qo_p)
                .arg(&k_p)
                .arg(&v_p)
                .arg(&qnw_p)
                .arg(&knw_p)
                .arg(&cos_p)
                .arg(&sin_p)
                .arg(&kc_p)
                .arg(&vc_p)
                .arg(&args.b)
                .arg(&args.t)
                .arg(&args.hq)
                .arg(&args.hkv)
                .arg(&args.d)
                .arg(&args.rotary_dim)
                .arg(&args.past_len)
                .arg(&args.q_strides[0])
                .arg(&args.q_strides[1])
                .arg(&args.q_strides[2])
                .arg(&args.qo_strides[0])
                .arg(&args.qo_strides[1])
                .arg(&args.qo_strides[2])
                .arg(&args.k_strides[0])
                .arg(&args.k_strides[1])
                .arg(&args.k_strides[2])
                .arg(&args.v_strides[0])
                .arg(&args.v_strides[1])
                .arg(&args.v_strides[2])
                .arg(&args.kc_strides[0])
                .arg(&args.kc_strides[1])
                .arg(&args.kc_strides[2])
                .arg(&args.cos_strides[0])
                .arg(&args.cos_strides[1])
                .arg(&args.eps)
                .arg(&args.flags);
            let cfg = LaunchConfig {
                grid_dim: args.grid(),
                block_dim: args.block(),
                shared_mem_bytes: smem,
            };
            // SAFETY: arguments match rope_kv_write_{sfx}.
            unsafe { bld.launch(cfg) }.w()?;
        }
        Ok(CudaStorage {
            slice: q_out,
            device: dev.clone(),
        })
    }
}
