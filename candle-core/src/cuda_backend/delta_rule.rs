//! Fused Gated DeltaNet kernels on the CUDA backend.
//!
//! The kernels are the ones of the ROCm backend (`kernels/delta_rule_impl.h`), compiled with
//! NVRTC for the GPU's compute capability the first time they are used (one module per kernel).
//! See [`crate::delta_rule`] for the operation and [`crate::accel::delta`] for the planning.

use super::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::accel::delta::{self as gd, GdnScanArgs, GNORM_THREADS};
use crate::{bail, DType, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg, SyncOnDrop};

/// NVRTC `--gpu-architecture` of the device (interned: NVRTC options take `&'static str`).
fn arch(dev: &CudaDevice) -> &'static str {
    use cudarc::driver::sys::CUdevice_attribute as A;
    use std::collections::HashMap;
    use std::sync::Mutex;
    static NAMES: Mutex<Option<HashMap<(i32, i32), &'static str>>> = Mutex::new(None);
    let major = dev
        .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(7);
    let minor = dev
        .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(5);
    let mut g = NAMES.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(HashMap::new)
        .entry((major, minor))
        .or_insert_with(|| Box::leak(format!("compute_{major}{minor}").into_boxed_str()))
}

impl CudaDevice {
    /// Gated DeltaNet kernel `name`, compiled with NVRTC on first use (retrying without
    /// `--gpu-architecture` when the installed NVRTC does not know the GPU yet).
    fn gdn_func(&self, name: &str) -> Result<super::device::CudaFunc> {
        let module = format!("candle_gdn_{name}");
        self.get_or_compile_func(name, &module, || {
            let Some(src) = gd::gdn_cuda_source(name) else {
                bail!("gated delta rule: unknown kernel {name}")
            };
            let arch = arch(self);
            let opts = |arch: Option<&'static str>| cudarc::nvrtc::CompileOptions {
                arch,
                options: vec!["--std=c++17".into()],
                ..Default::default()
            };
            match cudarc::nvrtc::safe::compile_ptx_with_opts(&src, opts(Some(arch))) {
                Ok(p) => Ok(p),
                Err(_) => cudarc::nvrtc::safe::compile_ptx_with_opts(&src, opts(None)).w(),
            }
        })
    }
}

/// Raw device pointer at an element offset, with the guard that orders the access.
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
        CudaStorageSlice::BF16(x) => p!(x, half::bf16),
        CudaStorageSlice::F16(x) => p!(x, half::f16),
        CudaStorageSlice::F32(x) => p!(x, f32),
        _ => bail!("gated delta rule: unsupported dtype on the CUDA backend"),
    })
}

fn dptr_opt<'a>(
    s: Option<&'a CudaStorage>,
    elem_offset: usize,
    stream: &'a cudarc::driver::CudaStream,
) -> Result<(u64, Option<SyncOnDrop<'a>>)> {
    match s {
        Some(s) => {
            let (p, g) = dptr(s, elem_offset, stream)?;
            Ok((p, Some(g)))
        }
        None => Ok((0, None)),
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
        _ => unreachable!("gated delta rule outputs are f32 / f16 / bf16"),
    }
}

/// Uninitialized buffer of `n` elements (at least one, so empty outputs stay valid).
fn alloc(dev: &CudaDevice, dtype: DType, n: usize) -> Result<CudaStorageSlice> {
    let n = n.max(1);
    // SAFETY: every element that is read later is written by the kernels first.
    unsafe {
        Ok(match dtype {
            DType::F32 => CudaStorageSlice::F32(dev.alloc::<f32>(n)?),
            DType::F16 => CudaStorageSlice::F16(dev.alloc::<half::f16>(n)?),
            DType::BF16 => CudaStorageSlice::BF16(dev.alloc::<half::bf16>(n)?),
            d => bail!("gated delta rule: unsupported dtype {d:?} on the CUDA backend"),
        })
    }
}

fn sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("gated delta rule: unsupported dtype {d:?} on the CUDA backend"),
    }
}

impl CudaStorage {
    /// Fused Gated DeltaNet layer step, see `crate::delta_rule::fused_gated_delta_rule`.
    /// `self` is `mixed_qkv`; `a`, `b`, the conv state and the outputs share its dtype, the
    /// conv weight / bias, `neg_a`, `dt_bias` and the recurrent states are f32.
    /// Returns (out [B,T,Hv,Dv], conv_state [B,C,K-1], recurrent_state [B,Hv,Dk,Dv]).
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = sfx(dtype)?;
        let Some(tile) = gd::scan_tile(p.dk, p.dv) else {
            bail!("gated delta rule: head_k_dim {} is not supported by the fused kernel", p.dk)
        };
        let dev = &self.device;
        let c = p.conv_dim();
        let km1 = p.kc.saturating_sub(1);
        let mut out = alloc(dev, dtype, p.b * p.t * p.hv * p.dv)?;
        let mut cs_out = alloc(dev, dtype, p.b * c * km1)?;
        let mut s_out = alloc(dev, DType::F32, p.b * p.hv * p.dk * p.dv)?;
        if p.b > 0 && p.hv > 0 && p.dv > 0 {
            let func = dev.gdn_func(&tile.kernel_name(sfx))?;
            let smem = tile.smem_bytes();
            if smem > 48 * 1024 {
                func.set_max_dynamic_shared_memory(smem)?;
            }
            let stream = dev.cuda_stream();
            let (x_p, _g0) = dptr(self, p.x[0], &stream)?;
            let (cs_p, _g1) = dptr_opt(conv_state, p.cs_off, &stream)?;
            let (cw_p, _g2) = dptr(conv_w, p.cw_off, &stream)?;
            let (cb_p, _g3) = dptr_opt(conv_b, p.cb_off, &stream)?;
            let (a_p, _g4) = dptr(a, p.a[0], &stream)?;
            let (b_p, _g5) = dptr(b, p.bb[0], &stream)?;
            let (na_p, _g6) = dptr(neg_a, p.na_off, &stream)?;
            let (dt_p, _g7) = dptr(dt_bias, p.dtb_off, &stream)?;
            let (s_p, _g8) = dptr_opt(state, p.s_off, &stream)?;
            let (o_p, _g9) = dptr_mut(&mut out, &stream);
            let (co_p, _g10) = dptr_mut(&mut cs_out, &stream);
            let (so_p, _g11) = dptr_mut(&mut s_out, &stream);
            let ptrs: [u64; 12] = [
                x_p, cs_p, cw_p, cb_p, a_p, b_p, na_p, dt_p, s_p, o_p, co_p, so_p,
            ];
            let dims: [u32; 7] = [
                p.t as u32,
                p.hk as u32,
                p.hv as u32,
                p.dk as u32,
                p.dv as u32,
                p.kc as u32,
                p.flags,
            ];
            let fl: [f32; 2] = [p.eps, p.scale];
            let strides: [u64; 6] = [
                p.x[1] as u64,
                p.x[2] as u64,
                p.a[1] as u64,
                p.a[2] as u64,
                p.bb[1] as u64,
                p.bb[2] as u64,
            ];
            let mut bld = func.builder();
            for x in ptrs.iter() {
                bld.arg(x);
            }
            for x in dims.iter() {
                bld.arg(x);
            }
            for x in fl.iter() {
                bld.arg(x);
            }
            for x in strides.iter() {
                bld.arg(x);
            }
            let cfg = LaunchConfig {
                grid_dim: tile.grid(p.b, p.hv, p.dv),
                block_dim: (tile.threads() as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            // SAFETY: arguments match GDN_SCAN_PARAMS.
            unsafe { bld.launch(cfg) }.w()?;
        }
        let mk = |slice| CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((mk(out), mk(cs_out), mk(s_out)))
    }

    /// Gated RMS norm `rmsnorm(x) * w * silu(z)` over rows of `d` contiguous elements of
    /// `self` (x) and `z` (same dtype); `w` is f32. Output in the dtype of x.
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let mut out = alloc(dev, dtype, rows * d)?;
        if rows > 0 && d > 0 {
            let func = dev.gdn_func(&format!("gdn_gnorm_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g0) = dptr(self, x_off, &stream)?;
            let (z_p, _g1) = dptr(z, z_off, &stream)?;
            let (w_p, _g2) = dptr(w, w_off, &stream)?;
            let (y_p, _g3) = dptr_mut(&mut out, &stream);
            let ptrs: [u64; 4] = [x_p, z_p, w_p, y_p];
            let rows64 = rows as u64;
            let d32 = d as u32;
            let mut bld = func.builder();
            for x in ptrs.iter() {
                bld.arg(x);
            }
            bld.arg(&rows64).arg(&d32).arg(&eps);
            let cfg = LaunchConfig {
                grid_dim: (rows.div_ceil(4).clamp(1, 1 << 20) as u32, 1, 1),
                block_dim: (GNORM_THREADS as u32, 1, 1),
                shared_mem_bytes: (GNORM_THREADS * 4) as u32,
            };
            // SAFETY: arguments match GDN_GNORM_KERNEL.
            unsafe { bld.launch(cfg) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }
}
