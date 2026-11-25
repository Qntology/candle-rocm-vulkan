//! Dedicated Vision Encoder fused kernels on the CUDA backend (`kernels/vis_ops_impl.h`, NVRTC).

use super::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::accel::vis_ops::{vis_cuda_source, VIS_LN_NT, VIS_LN_SMEM_BYTES, VIS_QKV_NT};
use crate::{bail, DType, Result};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg, SyncOnDrop};

impl CudaDevice {
    fn vis_func(&self, name: &str) -> Result<super::device::CudaFunc> {
        let module = format!("candle_vis_{name}");
        self.get_or_compile_source_func(name, &module, || {
            let Some(src) = vis_cuda_source(name) else {
                bail!("vis_ops: unknown kernel {name}")
            };
            Ok(src)
        })
    }
}

fn dptr<'a>(
    s: &'a CudaStorage,
    stream: &'a cudarc::driver::CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    Ok(match &s.slice {
        CudaStorageSlice::F32(x) => x.device_ptr(stream),
        CudaStorageSlice::F16(x) => x.device_ptr(stream),
        CudaStorageSlice::BF16(x) => x.device_ptr(stream),
        CudaStorageSlice::U8(x) => x.device_ptr(stream),
        CudaStorageSlice::U32(x) => x.device_ptr(stream),
        _ => bail!("vis_ops: unsupported dtype on CUDA"),
    })
}

fn dptr_opt<'a>(
    s: Option<&'a CudaStorage>,
    stream: &'a cudarc::driver::CudaStream,
) -> Result<(u64, Option<SyncOnDrop<'a>>)> {
    match s {
        Some(s) => {
            let (p, g) = dptr(s, stream)?;
            Ok((p, Some(g)))
        }
        None => Ok((0u64, None)),
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
        _ => unreachable!("vis_ops outputs are f32 / f16 / bf16"),
    }
}

fn alloc(dev: &CudaDevice, dtype: DType, n: usize) -> Result<CudaStorageSlice> {
    let n = n.max(1);
    unsafe {
        Ok(match dtype {
            DType::F32 => CudaStorageSlice::F32(dev.alloc::<f32>(n)?),
            DType::F16 => CudaStorageSlice::F16(dev.alloc::<half::f16>(n)?),
            DType::BF16 => CudaStorageSlice::BF16(dev.alloc::<half::bf16>(n)?),
            d => bail!("vis_ops: unsupported output dtype {d:?}"),
        })
    }
}

fn sfx(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        d => bail!("vis_ops: unsupported dtype {d:?} on CUDA"),
    }
}

impl CudaStorage {
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
        let mut out = alloc(dev, out_dtype, total_px * 3)?;
        if total_px > 0 {
            let func = dev.vis_func(&format!("vis_u8_resize_norm_patchify_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (in_p, _g0) = dptr(self, &stream)?;
            let (out_p, _g1) = dptr_mut(&mut out, &stream);
            let u32s: [u32; 7] = [
                h_in as u32,
                w_in as u32,
                h_out as u32,
                w_out as u32,
                patch_size as u32,
                u32::from(bicubic),
                u32::from(chw_in),
            ];
            let fs: [f32; 6] = [
                mean[0], mean[1], mean[2], inv_std[0], inv_std[1], inv_std[2],
            ];
            let mut bld = func.builder();
            bld.arg(&in_p).arg(&out_p);
            for v in u32s.iter() {
                bld.arg(v);
            }
            for v in fs.iter() {
                bld.arg(v);
            }
            let lc = LaunchConfig {
                grid_dim: (total_px.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { bld.launch(lc) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
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
        use crate::backend::BackendStorage;
        let res_dtype = self.dtype();
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
        let mut norm_out = alloc(dev, out_dtype, rows * d)?;
        let mut res_out = if write_residual {
            Some(alloc(dev, res_dtype, rows * d)?)
        } else {
            None
        };
        if rows > 0 && d > 0 {
            let func = dev.vis_func(&fname)?;
            let stream = dev.cuda_stream();
            let (rin_p, _g0) = dptr(self, &stream)?;
            let (del_p, _g1) = dptr_opt(delta, &stream)?;
            let (ls_p, _g2) = dptr_opt(ls_gamma, &stream)?;
            let (lnw_p, _g3) = dptr_opt(ln_weight, &stream)?;
            let (lnb_p, _g4) = dptr_opt(ln_bias, &stream)?;
            let (rout_p, _g5) = match res_out.as_mut() {
                Some(r) => {
                    let (p, g) = dptr_mut(r, &stream);
                    (p, Some(g))
                }
                None => (0u64, None),
            };
            let (nout_p, _g6) = dptr_mut(&mut norm_out, &stream);
            let rows64 = rows as u64;
            let d32 = d as u32;
            let mut bld = func.builder();
            bld.arg(&rin_p)
                .arg(&del_p)
                .arg(&ls_p)
                .arg(&lnw_p)
                .arg(&lnb_p)
                .arg(&rout_p)
                .arg(&nout_p)
                .arg(&rows64)
                .arg(&d32)
                .arg(&eps);
            let lc = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (VIS_LN_NT as u32, 1, 1),
                shared_mem_bytes: VIS_LN_SMEM_BYTES as u32,
            };
            unsafe { bld.launch(lc) }.w()?;
        }
        Ok((
            res_out.map(|slice| CudaStorage {
                slice,
                device: dev.clone(),
            }),
            CudaStorage {
                slice: norm_out,
                device: dev.clone(),
            },
        ))
    }

    pub fn vis_bias_gelu(
        &self,
        bias: Option<&Self>,
        n_elems: usize,
        dim: usize,
        mode: u32,
    ) -> Result<Self> {
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let mut out = alloc(dev, dtype, n_elems)?;
        if n_elems > 0 {
            let func = dev.vis_func(&format!("vis_bias_gelu_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g0) = dptr(self, &stream)?;
            let (b_p, _g1) = dptr_opt(bias, &stream)?;
            let (y_p, _g2) = dptr_mut(&mut out, &stream);
            let n64 = n_elems as u64;
            let d32 = dim as u32;
            let mut bld = func.builder();
            bld.arg(&x_p)
                .arg(&b_p)
                .arg(&y_p)
                .arg(&n64)
                .arg(&d32)
                .arg(&mode);
            let lc = LaunchConfig {
                grid_dim: (n_elems.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { bld.launch(lc) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let elems = num_heads * n_tokens * head_dim;
        let mut q_out = alloc(dev, dtype, elems)?;
        let mut k_out = alloc(dev, dtype, elems)?;
        let mut v_out = alloc(dev, dtype, elems)?;
        if n_tokens > 0 && num_heads > 0 && head_dim > 0 {
            let func = dev.vis_func(&format!("vis_qkv_2d_rope_{sfx}"))?;
            let smem = (2 * head_dim + VIS_QKV_NT) * 4;
            let stream = dev.cuda_stream();
            let (qkv_p, _g0) = dptr(self, &stream)?;
            let (b_p, _g1) = dptr_opt(bias, &stream)?;
            let (qn_p, _g2) = dptr_opt(q_norm_w, &stream)?;
            let (kn_p, _g3) = dptr_opt(k_norm_w, &stream)?;
            let (yx_p, _g4) = dptr(patch_yx, &stream)?;
            let (if_p, _g5) = dptr(inv_freq, &stream)?;
            let (qo_p, _g6) = dptr_mut(&mut q_out, &stream);
            let (ko_p, _g7) = dptr_mut(&mut k_out, &stream);
            let (vo_p, _g8) = dptr_mut(&mut v_out, &stream);
            let u32s: [u32; 5] = [
                n_tokens as u32,
                num_heads as u32,
                head_dim as u32,
                rot_dim as u32,
                flags,
            ];
            let mut bld = func.builder();
            bld.arg(&qkv_p)
                .arg(&b_p)
                .arg(&qn_p)
                .arg(&kn_p)
                .arg(&yx_p)
                .arg(&if_p)
                .arg(&qo_p)
                .arg(&ko_p)
                .arg(&vo_p);
            for v in u32s.iter() {
                bld.arg(v);
            }
            bld.arg(&eps);
            let lc = LaunchConfig {
                grid_dim: (n_tokens as u32, num_heads as u32, 1),
                block_dim: (VIS_QKV_NT as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe { bld.launch(lc) }.w()?;
        }
        Ok((
            CudaStorage {
                slice: q_out,
                device: dev.clone(),
            },
            CudaStorage {
                slice: k_out,
                device: dev.clone(),
            },
            CudaStorage {
                slice: v_out,
                device: dev.clone(),
            },
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
        use crate::backend::BackendStorage;
        let dtype = self.dtype();
        let sfx = sfx(dtype)?;
        let dev = &self.device;
        let out_toks = (h_p / merge) * (w_p / merge);
        let out_dim = if mode == 2 { d } else { merge * merge * d };
        let mut out = alloc(dev, dtype, out_toks * out_dim)?;
        if out_toks > 0 && d > 0 {
            let func = dev.vis_func(&format!("vis_spatial_merge_norm_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (x_p, _g0) = dptr(self, &stream)?;
            let (w_p_ptr, _g1) = dptr_opt(ln_weight, &stream)?;
            let (b_p_ptr, _g2) = dptr_opt(ln_bias, &stream)?;
            let (y_p, _g3) = dptr_mut(&mut out, &stream);
            let u32s: [u32; 5] = [h_p as u32, w_p as u32, d as u32, merge as u32, mode];
            let mut bld = func.builder();
            bld.arg(&x_p).arg(&w_p_ptr).arg(&b_p_ptr).arg(&y_p);
            for v in u32s.iter() {
                bld.arg(v);
            }
            bld.arg(&eps);
            let lc = LaunchConfig {
                grid_dim: (out_toks as u32, 1, 1),
                block_dim: (VIS_LN_NT as u32, 1, 1),
                shared_mem_bytes: VIS_LN_SMEM_BYTES as u32,
            };
            unsafe { bld.launch(lc) }.w()?;
        }
        Ok(CudaStorage {
            slice: out,
            device: dev.clone(),
        })
    }

    pub fn vis_scatter_embeds_inplace(
        &mut self,
        vis: &Self,
        indices: &Self,
        n_vis: usize,
        d: usize,
        l_total: usize,
    ) -> Result<()> {
        use crate::backend::BackendStorage;
        let sfx = sfx(self.dtype())?;
        let dev = &self.device;
        if n_vis > 0 && d > 0 {
            let func = dev.vis_func(&format!("vis_scatter_embeds_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (e_p, _g0) = dptr_mut(&mut self.slice, &stream);
            let (v_p, _g1) = dptr(vis, &stream)?;
            let (i_p, _g2) = dptr(indices, &stream)?;
            let u32s: [u32; 3] = [n_vis as u32, d as u32, l_total as u32];
            let mut bld = func.builder();
            bld.arg(&e_p).arg(&v_p).arg(&i_p);
            for v in u32s.iter() {
                bld.arg(v);
            }
            let lc = LaunchConfig {
                grid_dim: (n_vis as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { bld.launch(lc) }.w()?;
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
        use crate::backend::BackendStorage;
        let sfx = sfx(self.dtype())?;
        let dev = &self.device;
        let n_toks = h_p * w_p;
        if n_toks > 0 && d > 0 {
            let func = dev.vis_func(&format!("vis_patch_add_interp_pos2d_{sfx}"))?;
            let stream = dev.cuda_stream();
            let (p_p, _g0) = dptr_mut(&mut self.slice, &stream);
            let (t_p, _g1) = dptr(pos_table, &stream)?;
            let u32s: [u32; 7] = [
                g_h as u32,
                g_w as u32,
                h_p as u32,
                w_p as u32,
                d as u32,
                u32::from(bicubic),
                u32::from(align_corners),
            ];
            let mut bld = func.builder();
            bld.arg(&p_p).arg(&t_p);
            for v in u32s.iter() {
                bld.arg(v);
            }
            let lc = LaunchConfig {
                grid_dim: (n_toks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { bld.launch(lc) }.w()?;
        }
        Ok(())
    }
}
