//! Fused `lm_head` Tile Argmax + Mean-Centered CS Guard kernels on the CUDA backend.
//!
//! Compiled once on first use as a single NVRTC module (`candle_lm_head_argmax`) from
//! `kernels/lm_head_argmax_impl.h` with `--std=c++17` (no `--use_fast_math`).

use super::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::accel::lm_head::{self as lmh, LM_NT};
use crate::accel::sources::lm_head_argmax_cuda_source;
use crate::{bail, DType, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg, SyncOnDrop};

const MODULE_NAME: &str = "candle_lm_head_argmax";

impl CudaDevice {
    pub(crate) fn lm_head_func(&self, fn_name: &str) -> Result<super::device::CudaFunc> {
        self.get_or_compile_internal_func(
            fn_name,
            || MODULE_NAME.to_string(),
            || Ok(lm_head_argmax_cuda_source()),
        )
    }
}

fn stage2_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("lm_stage2_reduce_cs_f32"),
        DType::F16 => Ok("lm_stage2_reduce_cs_f16"),
        DType::BF16 => Ok("lm_stage2_reduce_cs_bf16"),
        d => bail!("lm_head_argmax: unsupported dtype {d:?} on the CUDA backend"),
    }
}

fn float_sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("lm_head_argmax: unsupported dtype {d:?} on the CUDA backend"),
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
        CudaStorageSlice::U32(x) => p!(x, u32),
        CudaStorageSlice::U8(x) => p!(x, u8),
        _ => bail!("lm_head_argmax: unsupported tensor dtype on the CUDA backend"),
    })
}

impl CudaStorage {
    /// Mode A: 2-Stage Dense Fused Tile Argmax + Mean-Centered CS Guard.
    /// Zero `[M, V]` logits buffer allocated in VRAM.
    /// Returns `(out_ids [M] u32, out_vals [M] f32, out_rescue [M] u8)`.
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = float_sfx(dtype)?;
        let dev = &self.device;

        // SAFETY: kernels write all m outputs before any read.
        let mut out_ids = unsafe { dev.alloc::<u32>(m.max(1))? };
        let mut out_vals = unsafe { dev.alloc::<f32>(m.max(1))? };
        let mut out_rescue = unsafe { dev.alloc::<u8>(m.max(1))? };

        if m > 0 && v_valid > 0 && k > 0 {
            let plan = lmh::plan_stage1(m, v_valid);
            let n_tiles = plan.n_tiles as usize;
            let pv_bytes = ((m * n_tiles).max(1) * 4).div_ceil(256) * 256;
            let pi_bytes = (m * n_tiles).max(1) * 4;

            let f1 = dev.lm_head_func(&plan.kernel_name(sfx))?;
            let f2 = dev.lm_head_func(stage2_name(dtype)?)?;
            let stream = dev.cuda_stream();

            // Hold the workspace lock across both Stage 1 and Stage 2 queueing so another
            // thread sharing this device cannot alias or reallocate `"lm_head_parts"`.
            let mut ws = dev.workspaces();
            let base = ws.ensure(dev, "lm_head_parts", pv_bytes + pi_bytes)?;
            let pv_p = base;
            let pi_p = base + pv_bytes as u64;

            // Stage 1: Tile matmul + in-register/smem tile argmax
            {
                let (x_p, _g0) = dptr(self, x_off, &stream)?;
                let (w_p, _g1) = dptr(w, w_off, &stream)?;
                let (b_p, _g2) = match bias {
                    Some((bs, off)) => {
                        let (p, g) = dptr(bs, off, &stream)?;
                        (p, Some(g))
                    }
                    None => (0u64, None),
                };
                let m32 = m as u32;
                let v32 = v_valid as u32;
                let k32 = k as u32;
                let nt32 = plan.n_tiles;
                let mut bld = f1.builder();
                bld.arg(&x_p)
                    .arg(&w_p)
                    .arg(&b_p)
                    .arg(&pv_p)
                    .arg(&pi_p)
                    .arg(&m32)
                    .arg(&v32)
                    .arg(&k32)
                    .arg(&nt32);
                let cfg = LaunchConfig {
                    grid_dim: plan.grid,
                    block_dim: plan.block,
                    shared_mem_bytes: plan.smem_bytes,
                };
                unsafe { bld.launch(cfg) }.w()?;
            }

            // Stage 2: Tile reduction + core_ids remap + CS upper-bound check
            {
                let (cid_p, _g2) = match core_ids {
                    Some((cs, off)) => {
                        let (p, g) = dptr(cs, off, &stream)?;
                        (p, Some(g))
                    }
                    None => (0u64, None),
                };
                let (h_p, _g3) = dptr(self, x_off, &stream)?;
                let (mu_p, _g4) = match mu_rest {
                    Some((ms, off)) => {
                        let (p, g) = dptr(ms, off, &stream)?;
                        (p, Some(g))
                    }
                    None => (0u64, None),
                };
                let (oid_p, _g5) = out_ids.device_ptr_mut(&stream);
                let (oval_p, _g6) = out_vals.device_ptr_mut(&stream);
                let (ores_p, _g7) = out_rescue.device_ptr_mut(&stream);
                let m32 = m as u32;
                let nt32 = plan.n_tiles;
                let k32 = k as u32;
                let cs32 = u32::from(check_cs);
                let mut bld = f2.builder();
                bld.arg(&pv_p)
                    .arg(&pi_p)
                    .arg(&cid_p)
                    .arg(&h_p)
                    .arg(&mu_p)
                    .arg(&w_rest_bound)
                    .arg(&b_rest_max)
                    .arg(&oid_p)
                    .arg(&oval_p)
                    .arg(&ores_p)
                    .arg(&m32)
                    .arg(&nt32)
                    .arg(&k32)
                    .arg(&cs32);
                let cfg = LaunchConfig {
                    grid_dim: (m as u32, 1, 1),
                    block_dim: (LM_NT, 1, 1),
                    shared_mem_bytes: lmh::stage2_smem_bytes(),
                };
                unsafe { bld.launch(cfg) }.w()?;
            }
        }

        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(out_ids),
                device: dev.clone(),
            },
            CudaStorage {
                slice: CudaStorageSlice::F32(out_vals),
                device: dev.clone(),
            },
            CudaStorage {
                slice: CudaStorageSlice::U8(out_rescue),
                device: dev.clone(),
            },
        ))
    }

    /// Quantized Mode A on CUDA: 2-Stage Fused Tile Argmax + Mean-Centered CS Guard directly over
    /// quantized weights (`Q8_0`, `Q4_0`, `Q4K`, `Q5K`, `Q6K`) with 0-byte `[M, V]` logits
    /// materialization (`P1-O03` / `Q5-O07`). Supports arbitrary `M` by chunking 8 rows per pass.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_quant_argmax_cs(
        &self,
        x_off: usize,
        wq: &cudarc::driver::CudaSlice<u8>,
        wtype: crate::quantized::GgmlDType,
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let s2 = stage2_name(dtype)?;
        let dev = &self.device;

        let mut out_ids = unsafe { dev.alloc::<u32>(m.max(1))? };
        let mut out_vals = unsafe { dev.alloc::<f32>(m.max(1))? };
        let mut out_rescue = unsafe { dev.alloc::<u8>(m.max(1))? };

        if m > 0 && v_valid > 0 && k > 0 {
            let n_tiles = v_valid.div_ceil(32);
            let max_chunk = m.min(8);
            let pv_bytes = ((max_chunk * n_tiles).max(1) * 4).div_ceil(256) * 256;
            let pi_bytes = (max_chunk * n_tiles).max(1) * 4;

            let f2 = dev.lm_head_func(s2)?;
            let stream = dev.cuda_stream();

            let mut ws = dev.workspaces();
            let base = ws.ensure(dev, "lm_head_parts", pv_bytes + pi_bytes)?;
            let pv_p = base;
            let pi_p = base + pv_bytes as u64;

            let (wq_p, _g_wq) = wq.device_ptr(&stream);
            let (b_p, _g_b) = match bias {
                Some((bs, off)) => {
                    let (p, g) = dptr(bs, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (cid_p, _g_c) = match core_ids {
                Some((cs, off)) => {
                    let (p, g) = dptr(cs, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (mu_p, _g_m) = match mu_rest {
                Some((ms, off)) => {
                    let (p, g) = dptr(ms, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };

            let mut row0 = 0usize;
            while row0 < m {
                let cm = (m - row0).min(8);
                let f1 = crate::quantized::w8a16::lm_stage1_quant_func(dev, wtype, dtype, cm)?;
                let (x_p, _g_x) = dptr(self, x_off + row0 * k, &stream)?;
                {
                    let m_i = cm as i32;
                    let v_i = v_valid as i32;
                    let k_i = k as i32;
                    let nt_i = n_tiles as i32;
                    let rb_i = row_bytes as i64;
                    let mut bld = stream.launch_builder(&f1);
                    bld.arg(&x_p)
                        .arg(&wq_p)
                        .arg(&b_p)
                        .arg(&pv_p)
                        .arg(&pi_p)
                        .arg(&m_i)
                        .arg(&v_i)
                        .arg(&k_i)
                        .arg(&nt_i)
                        .arg(&rb_i);
                    let cfg = LaunchConfig {
                        grid_dim: (n_tiles as u32, 1, 1),
                        block_dim: (32, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { bld.launch(cfg) }.w()?;
                }
                {
                    let (oid_p, _g5) = out_ids.device_ptr_mut(&stream);
                    let (oval_p, _g6) = out_vals.device_ptr_mut(&stream);
                    let (ores_p, _g7) = out_rescue.device_ptr_mut(&stream);
                    let oid_c = oid_p + (row0 * 4) as u64;
                    let oval_c = oval_p + (row0 * 4) as u64;
                    let ores_c = ores_p + row0 as u64;
                    let m32 = cm as u32;
                    let nt32 = n_tiles as u32;
                    let k32 = k as u32;
                    let cs32 = u32::from(check_cs);
                    let mut bld = f2.builder();
                    bld.arg(&pv_p)
                        .arg(&pi_p)
                        .arg(&cid_p)
                        .arg(&x_p)
                        .arg(&mu_p)
                        .arg(&w_rest_bound)
                        .arg(&b_rest_max)
                        .arg(&oid_c)
                        .arg(&oval_c)
                        .arg(&ores_c)
                        .arg(&m32)
                        .arg(&nt32)
                        .arg(&k32)
                        .arg(&cs32);
                    let cfg = LaunchConfig {
                        grid_dim: (cm as u32, 1, 1),
                        block_dim: (LM_NT, 1, 1),
                        shared_mem_bytes: lmh::stage2_smem_bytes(),
                    };
                    unsafe { bld.launch(cfg) }.w()?;
                }
                row0 += cm;
            }
        }

        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(out_ids),
                device: dev.clone(),
            },
            CudaStorage {
                slice: CudaStorageSlice::F32(out_vals),
                device: dev.clone(),
            },
            CudaStorage {
                slice: CudaStorageSlice::U8(out_rescue),
                device: dev.clone(),
            },
        ))
    }

    /// Mode B: 1-Stage Sub-Logits Argmax + Core-ID Remap + Mean-Centered CS Guard.
    /// Works directly on `sub_logits` produced by quantized `QMatMul` without full-vocab expansion.
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
        use crate::backend::BackendStorage;
        let l_sfx = float_sfx(self.dtype())?;
        let h_sfx = match hidden {
            Some((h, _)) => float_sfx(h.dtype())?,
            None => l_sfx,
        };
        let fn_name = format!("lm_logits_argmax_cs_{l_sfx}_{h_sfx}");
        let dev = &self.device;

        let mut out_ids = unsafe { dev.alloc::<u32>(m.max(1))? };
        let mut out_vals = unsafe { dev.alloc::<f32>(m.max(1))? };
        let mut out_rescue = unsafe { dev.alloc::<u8>(m.max(1))? };

        if m > 0 && v_valid > 0 {
            let func = dev.lm_head_func(&fn_name)?;
            let stream = dev.cuda_stream();
            let (l_p, _g0) = dptr(self, logits_off, &stream)?;
            let (cid_p, _g1) = match core_ids {
                Some((cs, off)) => {
                    let (p, g) = dptr(cs, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (h_p, _g2) = match hidden {
                Some((hs, off)) => {
                    let (p, g) = dptr(hs, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (mu_p, _g3) = match mu_rest {
                Some((ms, off)) => {
                    let (p, g) = dptr(ms, off, &stream)?;
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (oid_p, _g4) = out_ids.device_ptr_mut(&stream);
            let (oval_p, _g5) = out_vals.device_ptr_mut(&stream);
            let (ores_p, _g6) = out_rescue.device_ptr_mut(&stream);
            let stride64 = logits_row_stride as u64;
            let m32 = m as u32;
            let v32 = v_valid as u32;
            let k32 = k as u32;
            let cs32 = u32::from(check_cs && hidden.is_some());
            let mut bld = func.builder();
            bld.arg(&l_p)
                .arg(&stride64)
                .arg(&cid_p)
                .arg(&h_p)
                .arg(&mu_p)
                .arg(&w_rest_bound)
                .arg(&b_rest_max)
                .arg(&oid_p)
                .arg(&oval_p)
                .arg(&ores_p)
                .arg(&m32)
                .arg(&v32)
                .arg(&k32)
                .arg(&cs32);
            let cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (LM_NT, 1, 1),
                shared_mem_bytes: lmh::stage2_smem_bytes(),
            };
            unsafe { bld.launch(cfg) }.w()?;
        }

        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(out_ids),
                device: dev.clone(),
            },
            CudaStorage {
                slice: CudaStorageSlice::F32(out_vals),
                device: dev.clone(),
            },
            CudaStorage {
                slice: CudaStorageSlice::U8(out_rescue),
                device: dev.clone(),
            },
        ))
    }
}
