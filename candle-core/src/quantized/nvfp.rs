//! Planar NVFP4 (E2M1 + 16-elem OCP E4M3 block scale + f32 row scale) and
//! Planar NVFP8 (OCP E4M3FN + f32 row scale) quantized linear weights.
//!
//! All data lives in a single 256-byte-aligned planar buffer `[qdata | bscale | rscale]`
//! compatible with:
//! * HuggingFace modelopt / compressed-tensors checkpoints (`from_safetensors_parts`,
//!   `from_safetensors_tensors`);
//! * `kernels/w8a16_impl.h` (`w8a16_nvfp4_{mv,dq}`, `w8a16_nvfp8_{mv,dq}`) on ROCm (hipcc) and
//!   CUDA (NVRTC), plus 32 MiB tiled prefill dequantization + GEMM on ROCm;
//! * CPU scalar dequantization & matmul (bit-exact with `accel::formats` OCP E4M3FN & E2M1).
//!   Vulkan currently falls back to host dequantization (emitting a one-time warning).

use crate::accel::formats::{e4m3_to_f32, f32_to_e4m3, f32_to_fp4, fp4_to_f32};
use crate::{bail, DType, Device, Result, Tensor};

/// Supported planar FP4 / FP8 linear weight formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvfpFormat {
    /// 4-bit E2M1 elements (2 per byte, LSB first) + 16-element OCP E4M3FN block scales + per-row f32 global scale.
    Nvfp4,
    /// 8-bit OCP E4M3FN elements + per-row f32 scale.
    Nvfp8,
}

/// Byte offsets of a 256-byte-aligned single-buffer planar NVFP4 / NVFP8 weight matrix `[rows, cols]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvfpPlanarLayout {
    pub format: NvfpFormat,
    pub rows: usize,
    pub cols: usize,
    pub qdata_bytes: usize,
    pub bscale_off: usize,
    pub bscale_bytes: usize,
    pub rscale_off: usize,
    pub total_bytes: usize,
}

impl NvfpPlanarLayout {
    #[inline]
    pub const fn align256(x: usize) -> usize {
        x.div_ceil(256) * 256
    }

    pub fn nvfp4(rows: usize, cols: usize) -> Result<Self> {
        if !cols.is_multiple_of(16) {
            bail!("Nvfp4 requires cols ({cols}) divisible by 16");
        }
        let qdata_bytes = rows * (cols / 2);
        let bscale_off = Self::align256(qdata_bytes);
        let bscale_bytes = rows * (cols / 16);
        let rscale_off = Self::align256(bscale_off + bscale_bytes);
        Ok(Self {
            format: NvfpFormat::Nvfp4,
            rows,
            cols,
            qdata_bytes,
            bscale_off,
            bscale_bytes,
            rscale_off,
            total_bytes: rscale_off + rows * 4,
        })
    }

    pub fn nvfp8(rows: usize, cols: usize) -> Result<Self> {
        if !cols.is_multiple_of(8) {
            bail!("Nvfp8 requires cols ({cols}) divisible by 8");
        }
        let qdata_bytes = rows * cols;
        let bscale_off = Self::align256(qdata_bytes);
        let rscale_off = bscale_off;
        Ok(Self {
            format: NvfpFormat::Nvfp8,
            rows,
            cols,
            qdata_bytes,
            bscale_off,
            bscale_bytes: 0,
            rscale_off,
            total_bytes: rscale_off + rows * 4,
        })
    }

    pub fn new(format: NvfpFormat, rows: usize, cols: usize) -> Result<Self> {
        match format {
            NvfpFormat::Nvfp4 => Self::nvfp4(rows, cols),
            NvfpFormat::Nvfp8 => Self::nvfp8(rows, cols),
        }
    }
}

/// Planar quantized NVFP4 / NVFP8 weight matrix stored as a single 256B-aligned byte tensor on any device.
#[derive(Debug, Clone)]
pub struct NvfpWeight {
    pub layout: NvfpPlanarLayout,
    /// Single 1D `U8` tensor of length `layout.total_bytes` on the target device.
    data: Tensor,
}

impl NvfpWeight {
    pub fn format(&self) -> NvfpFormat {
        self.layout.format
    }

    pub fn rows(&self) -> usize {
        self.layout.rows
    }

    pub fn cols(&self) -> usize {
        self.layout.cols
    }

    pub fn device(&self) -> &Device {
        self.data.device()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.layout.total_bytes
    }

    /// Packs pre-quantized safetensors components (`weight_packed`, optional `weight_scale`,
    /// per-row or scalar `rscale`) into a single planar buffer without re-quantizing.
    pub fn from_safetensors_parts(
        qdata: &[u8],
        bscale: Option<&[u8]>,
        rscale: &[f32],
        rows: usize,
        cols: usize,
        format: NvfpFormat,
        device: &Device,
    ) -> Result<Self> {
        let layout = NvfpPlanarLayout::new(format, rows, cols)?;
        if qdata.len() != layout.qdata_bytes {
            bail!(
                "nvfp qdata length {} != expected {}",
                qdata.len(),
                layout.qdata_bytes
            );
        }
        let mut buf = vec![0u8; layout.total_bytes];
        buf[..layout.qdata_bytes].copy_from_slice(qdata);
        if layout.bscale_bytes > 0 {
            let bs = bscale.ok_or_else(|| crate::Error::Msg("nvfp4 requires bscale".into()))?;
            if bs.len() != layout.bscale_bytes {
                bail!(
                    "nvfp4 bscale length {} != expected {}",
                    bs.len(),
                    layout.bscale_bytes
                );
            }
            buf[layout.bscale_off..layout.bscale_off + layout.bscale_bytes].copy_from_slice(bs);
        }
        let rscale_dst = &mut buf[layout.rscale_off..layout.rscale_off + rows * 4];
        match rscale.len() {
            1 => {
                let bytes = rscale[0].to_le_bytes();
                for r in 0..rows {
                    rscale_dst[r * 4..(r + 1) * 4].copy_from_slice(&bytes);
                }
            }
            n if n == rows => {
                for (r, &s) in rscale.iter().enumerate() {
                    rscale_dst[r * 4..(r + 1) * 4].copy_from_slice(&s.to_le_bytes());
                }
            }
            n => bail!("nvfp rscale length {n} must be 1 or rows ({rows})"),
        }
        let data = Tensor::from_vec(buf, (layout.total_bytes,), device)?;
        Ok(Self { layout, data })
    }

    /// Loads an [`NvfpWeight`] directly from safetensors-loaded tensors (`weight_packed` /
    /// `weight`, optional `weight_scale` for NVFP4 16-element blocks, and per-row or scalar
    /// `weight_scale_2` / global scale).
    ///
    /// Accepts `weight_packed` as `[rows, cols/2]` (`Nvfp4`, U8) or `[rows, cols]` (`Nvfp8`,
    /// U8/F8E4M3), `weight_scale` as `[rows, cols/16]` (U8/F8E4M3) for NVFP4, and `rscale` as
    /// scalar / `[1]` / `[rows]` float tensor (`Q1-F01` / `Q1-O01`).
    pub fn from_safetensors_tensors(
        weight_packed: &Tensor,
        weight_scale: Option<&Tensor>,
        weight_scale_2: &Tensor,
        format: NvfpFormat,
        device: &Device,
    ) -> Result<Self> {
        let (rows, packed_cols) = weight_packed.dims2()?;
        let cols = match format {
            NvfpFormat::Nvfp4 => packed_cols * 2,
            NvfpFormat::Nvfp8 => packed_cols,
        };
        let qdata = match weight_packed.dtype() {
            DType::U8 => weight_packed
                .contiguous()?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u8>()?,
            DType::F8E4M3 => weight_packed
                .contiguous()?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect(),
            dt => bail!("NvfpWeight::from_safetensors_tensors: unsupported qdata dtype {dt:?}"),
        };
        let bscale = match (format, weight_scale) {
            (NvfpFormat::Nvfp4, Some(bs)) => Some(match bs.dtype() {
                DType::U8 => bs
                    .contiguous()?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<u8>()?,
                DType::F8E4M3 => bs
                    .contiguous()?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<float8::F8E4M3>()?
                    .into_iter()
                    .map(|v| v.to_bits())
                    .collect(),
                dt => bail!("NvfpWeight::from_safetensors_tensors: unsupported bscale dtype {dt:?}"),
            }),
            (NvfpFormat::Nvfp4, None) => {
                bail!("NvfpWeight::from_safetensors_tensors: Nvfp4 requires weight_scale")
            }
            (NvfpFormat::Nvfp8, _) => None,
        };
        let rscale = weight_scale_2
            .contiguous()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Self::from_safetensors_parts(
            &qdata,
            bscale.as_deref(),
            &rscale,
            rows,
            cols,
            format,
            device,
        )
    }

    /// Quantizes a 2D weight tensor `[rows, cols]` into planar NVFP4 or NVFP8 on `tensor.device()`.
    pub fn quantize(tensor: &Tensor, format: NvfpFormat) -> Result<Self> {
        Self::quantize_onto(tensor, format, tensor.device())
    }

    /// Quantizes a 2D weight tensor `[rows, cols]` into planar NVFP4 or NVFP8 on `target_device`.
    pub fn quantize_onto(
        tensor: &Tensor,
        format: NvfpFormat,
        target_device: &Device,
    ) -> Result<Self> {
        let (rows, cols) = tensor.dims2()?;
        let layout = NvfpPlanarLayout::new(format, rows, cols)?;
        let vals = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut buf = vec![0u8; layout.total_bytes];

        match format {
            NvfpFormat::Nvfp4 => {
                let nblk = cols / 16;
                for r in 0..rows {
                    let row = &vals[r * cols..(r + 1) * cols];
                    let am = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let g = if am > 0.0 { am / 2688.0 } else { 1.0 };
                    let r_off = layout.rscale_off + r * 4;
                    buf[r_off..r_off + 4].copy_from_slice(&g.to_le_bytes());
                    for b in 0..nblk {
                        let sub = &row[b * 16..(b + 1) * 16];
                        let bm = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                        let sb = f32_to_e4m3(bm / 6.0 / g);
                        buf[layout.bscale_off + r * nblk + b] = sb;
                        let eff = e4m3_to_f32(sb) * g;
                        let q_base = r * (cols / 2) + b * 8;
                        for pair in 0..8 {
                            let c0 = if eff > 0.0 { f32_to_fp4(sub[pair * 2] / eff) } else { 0 };
                            let c1 = if eff > 0.0 { f32_to_fp4(sub[pair * 2 + 1] / eff) } else { 0 };
                            buf[q_base + pair] = ((c0 & 0x0f) | ((c1 & 0x0f) << 4)) as u8;
                        }
                    }
                }
            }
            NvfpFormat::Nvfp8 => {
                for r in 0..rows {
                    let row = &vals[r * cols..(r + 1) * cols];
                    let am = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let s = if am > 0.0 { am / 448.0 } else { 1.0 };
                    let r_off = layout.rscale_off + r * 4;
                    buf[r_off..r_off + 4].copy_from_slice(&s.to_le_bytes());
                    let q_base = r * cols;
                    for (j, &v) in row.iter().enumerate() {
                        buf[q_base + j] = f32_to_e4m3(v / s);
                    }
                }
            }
        }
        let data = Tensor::from_vec(buf, (layout.total_bytes,), target_device)?;
        Ok(Self { layout, data })
    }

    /// Dequantizes the full planar matrix `[rows, cols]` to `dtype` (`F32` / `F16` / `BF16`).
    pub fn dequantize_to(&self, dtype: DType) -> Result<Tensor> {
        let rows = self.layout.rows;
        let cols = self.layout.cols;
        #[cfg(feature = "rocm")]
        if let Device::Rocm(dev) = self.data.device() {
            if let Some(sfx) = match dtype {
                DType::F32 => Some("f32"),
                DType::F16 => Some("f16"),
                DType::BF16 => Some("bf16"),
                _ => None,
            } {
                use crate::rocm_backend::kernels::Module;
                use crate::rocm_backend::utils::LaunchArgs;
                use crate::rocm_backend::RocmStorage;
                let (st, _) = self.data.storage_and_layout();
                if let crate::Storage::Rocm(rs) = &*st {
                    let base = rs.device_ptr();
                    let out = dev.alloc((rows * cols).max(1) * dtype.size_in_bytes())?;
                    if rows > 0 && cols > 0 {
                        let mut args = LaunchArgs::new();
                        let (fname, total_work) = match self.layout.format {
                            NvfpFormat::Nvfp4 => {
                                let bs_ptr = unsafe { base.add(self.layout.bscale_off) };
                                let rs_ptr = unsafe { base.add(self.layout.rscale_off) };
                                args.ptr(base)
                                    .ptr(bs_ptr)
                                    .ptr(rs_ptr)
                                    .ptr(out.as_mut_ptr())
                                    .u32(cols as u32)
                                    .u32(rows as u32);
                                (format!("w8a16_nvfp4_dq_{sfx}"), rows * (cols / 16))
                            }
                            NvfpFormat::Nvfp8 => {
                                let rs_ptr = unsafe { base.add(self.layout.rscale_off) };
                                args.ptr(base)
                                    .ptr(rs_ptr)
                                    .ptr(out.as_mut_ptr())
                                    .u32(cols as u32)
                                    .u32(rows as u32);
                                (format!("w8a16_nvfp8_dq_{sfx}"), rows * (cols / 8))
                            }
                        };
                        if let Ok(func) = dev.get_func(Module::W8A16, &fname) {
                            let block = 256u32;
                            let grid = total_work.div_ceil(block as usize).clamp(1, 65535) as u32;
                            dev.launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut args)?;
                            let out_st = RocmStorage::new(out, dtype, dev.clone());
                            return Ok(crate::tensor::from_storage(
                                crate::Storage::Rocm(out_st),
                                (rows, cols),
                                crate::op::BackpropOp::none(),
                                false,
                            ));
                        }
                    }
                }
            }
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = self.data.device() {
            if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
                let (st, _) = self.data.storage_and_layout();
                if let crate::Storage::Cuda(cs) = &*st {
                    let data_slice = cs.as_cuda_slice::<u8>()?;
                    let out_st = super::w8a16::nvfp_dq(
                        dev,
                        data_slice,
                        self.layout.bscale_off,
                        self.layout.rscale_off,
                        self.layout.format == NvfpFormat::Nvfp4,
                        cols,
                        rows,
                        dtype,
                    )?;
                    return Ok(crate::tensor::from_storage(
                        crate::Storage::Cuda(out_st),
                        (rows, cols),
                        crate::op::BackpropOp::none(),
                        false,
                    ));
                }
            }
        }

        #[cfg(feature = "vulkan")]
        if matches!(self.data.device(), Device::Vulkan(_)) {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[candle-core] NvfpWeight::dequantize_to: Vulkan has no native NVFP shader yet; falling back to host dequantization"
                );
            }
        }

        // Portable CPU / fallback dequantization (bit-exact with GPU decoders).
        let raw = self.data.to_device(&Device::Cpu)?.to_vec1::<u8>()?;
        let mut out = vec![0.0f32; rows * cols];
        match self.layout.format {
            NvfpFormat::Nvfp4 => {
                let nblk = cols / 16;
                for r in 0..rows {
                    let ro = self.layout.rscale_off + r * 4;
                    let row_g = f32::from_le_bytes(raw[ro..ro + 4].try_into().unwrap());
                    for b in 0..nblk {
                        let sb = raw[self.layout.bscale_off + r * nblk + b];
                        let eff = e4m3_to_f32(sb) * row_g;
                        let q_base = r * (cols / 2) + b * 8;
                        let dst = &mut out[r * cols + b * 16..r * cols + (b + 1) * 16];
                        for pair in 0..8 {
                            let byte = raw[q_base + pair];
                            dst[pair * 2] = fp4_to_f32((byte & 0x0f) as u32) * eff;
                            dst[pair * 2 + 1] = fp4_to_f32((byte >> 4) as u32) * eff;
                        }
                    }
                }
            }
            NvfpFormat::Nvfp8 => {
                for r in 0..rows {
                    let ro = self.layout.rscale_off + r * 4;
                    let row_g = f32::from_le_bytes(raw[ro..ro + 4].try_into().unwrap());
                    let q_base = r * cols;
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    for j in 0..cols {
                        dst[j] = e4m3_to_f32(raw[q_base + j]) * row_g;
                    }
                }
            }
        }
        let t = Tensor::from_vec(out, (rows, cols), self.data.device())?;
        if dtype == DType::F32 {
            Ok(t)
        } else {
            t.to_dtype(dtype)
        }
    }

    /// Exact weight-only forward `y = x · dequant(W)ᵀ` (zero activation quantization).
    /// Dispatches to fused GPU matvec (`w8a16_nvfp{4,8}_mv*`) for decode (`M <= 8`) on ROCm,
    /// and dequantized GEMM for prefill (`M > 8`) or CPU/Vulkan/CUDA fallback.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        if x_dims.is_empty() {
            bail!("NvfpWeight::forward: rank-0 input");
        }
        let k = *x_dims.last().unwrap();
        if k != self.layout.cols {
            bail!(
                "NvfpWeight::forward: input last dim {k} != weight cols {}",
                self.layout.cols
            );
        }
        let mut out_dims = x_dims.to_vec();
        *out_dims.last_mut().unwrap() = self.layout.rows;
        let m = x.elem_count() / k.max(1);
        let in_dtype = x.dtype();

        #[cfg(feature = "rocm")]
        if let Device::Rocm(dev) = self.data.device() {
            if m > 0 && self.layout.rows > 0 {
                if let Some(sfx) = match in_dtype {
                    DType::F32 => Some("f32"),
                    DType::F16 => Some("f16"),
                    DType::BF16 => Some("bf16"),
                    _ => None,
                } {
                    use crate::rocm_backend::kernels::Module;
                    use crate::rocm_backend::utils::LaunchArgs;
                    use crate::rocm_backend::{GemmCall, GemmOutcome, RocmStorage};
                    use hip_runtime::blas::GemmType;
                    use std::ffi::c_void;
                    let esz = in_dtype.size_in_bytes();
                    let x_owned;
                    let x_ref = if x.is_contiguous()
                        && (x.layout().start_offset() * esz).is_multiple_of(16)
                    {
                        x
                    } else {
                        x_owned = x.force_contiguous()?;
                        &x_owned
                    };
                    let (wst, _) = self.data.storage_and_layout();
                    let (xst, xl) = x_ref.storage_and_layout();
                    if let (crate::Storage::Rocm(ws), crate::Storage::Rocm(xs)) = (&*wst, &*xst) {
                        let x_ptr = unsafe { xs.device_ptr().add(xl.start_offset() * esz) };
                        let base = ws.device_ptr();
                        if m <= 8 {
                            let m_tag = if m == 1 { "1" } else { "8" };
                            let fname = match self.layout.format {
                                NvfpFormat::Nvfp4 => format!("w8a16_nvfp4_mv{m_tag}_{sfx}"),
                                NvfpFormat::Nvfp8 => format!("w8a16_nvfp8_mv{m_tag}_{sfx}"),
                            };
                            if let Ok(func) = dev.get_func(Module::W8A16, &fname) {
                                let out = dev.alloc(m * self.layout.rows * esz)?;
                                let mut args = LaunchArgs::new();
                                match self.layout.format {
                                    NvfpFormat::Nvfp4 => {
                                        let bs_ptr = unsafe { base.add(self.layout.bscale_off) };
                                        let rs_ptr = unsafe { base.add(self.layout.rscale_off) };
                                        args.ptr(base)
                                            .ptr(bs_ptr)
                                            .ptr(rs_ptr)
                                            .ptr(x_ptr)
                                            .ptr(out.as_mut_ptr())
                                            .u32(k as u32)
                                            .u32(self.layout.rows as u32)
                                            .u32(m as u32);
                                    }
                                    NvfpFormat::Nvfp8 => {
                                        let rs_ptr = unsafe { base.add(self.layout.rscale_off) };
                                        args.ptr(base)
                                            .ptr(rs_ptr)
                                            .ptr(x_ptr)
                                            .ptr(out.as_mut_ptr())
                                            .u32(k as u32)
                                            .u32(self.layout.rows as u32)
                                            .u32(m as u32);
                                    }
                                }
                                let grid_x = self.layout.rows.div_ceil(4) as u32;
                                dev.launch(func, (grid_x, 1, 1), (32, 4, 1), 0, &mut args)?;
                                let out_st = RocmStorage::new(out, in_dtype, dev.clone());
                                return Ok(crate::tensor::from_storage(
                                    crate::Storage::Rocm(out_st),
                                    out_dims,
                                    crate::op::BackpropOp::none(),
                                    false,
                                ));
                            }
                        } else {
                            // Prefill (M > 8): 32 MiB tiled dequant into reusable workspace + GEMM
                            // so large weights never materialize a full `[N, K]` dense buffer (`Q1-O03`).
                            let dq_name = match self.layout.format {
                                NvfpFormat::Nvfp4 => format!("w8a16_nvfp4_dq_{sfx}"),
                                NvfpFormat::Nvfp8 => format!("w8a16_nvfp8_dq_{sfx}"),
                            };
                            if let Ok(dq_func) = dev.get_func(Module::W8A16, &dq_name) {
                                let n = self.layout.rows;
                                let tile_bytes: usize = 32 << 20;
                                let tile_n = (tile_bytes / (k * esz).max(1)).clamp(1, n);
                                let gemm_ty = match in_dtype {
                                    DType::F32 => GemmType::F32,
                                    DType::F16 => GemmType::F16,
                                    DType::BF16 => GemmType::BF16,
                                    _ => unreachable!(),
                                };
                                let out = dev.alloc(m * n * esz)?;
                                let tile_lease =
                                    dev.workspace_ensure("rocm_dq_tile", tile_n * k * esz)?;
                                let tile_ptr = tile_lease.as_mut_ptr();
                                let mut n0 = 0usize;
                                let mut ok = true;
                                while n0 < n {
                                    let tn = tile_n.min(n - n0);
                                    let mut args = LaunchArgs::new();
                                    let total_work = match self.layout.format {
                                        NvfpFormat::Nvfp4 => {
                                            let q_p = unsafe { base.add(n0 * (k / 2)) };
                                            let bs_p = unsafe {
                                                base.add(self.layout.bscale_off + n0 * (k / 16))
                                            };
                                            let rs_p =
                                                unsafe { base.add(self.layout.rscale_off + n0 * 4) };
                                            args.ptr(q_p)
                                                .ptr(bs_p)
                                                .ptr(rs_p)
                                                .ptr(tile_ptr)
                                                .u32(k as u32)
                                                .u32(tn as u32);
                                            tn * (k / 16)
                                        }
                                        NvfpFormat::Nvfp8 => {
                                            let q_p = unsafe { base.add(n0 * k) };
                                            let rs_p =
                                                unsafe { base.add(self.layout.rscale_off + n0 * 4) };
                                            args.ptr(q_p)
                                                .ptr(rs_p)
                                                .ptr(tile_ptr)
                                                .u32(k as u32)
                                                .u32(tn as u32);
                                            tn * (k / 8)
                                        }
                                    };
                                    let block = 256u32;
                                    let grid =
                                        total_work.div_ceil(block as usize).clamp(1, 65535) as u32;
                                    dev.launch(dq_func, (grid, 1, 1), (block, 1, 1), 0, &mut args)?;
                                    let c = unsafe { out.as_mut_ptr().add(n0 * esz) } as *mut c_void;
                                    let call = GemmCall {
                                        ty: gemm_ty,
                                        transa: true,
                                        transb: false,
                                        m: tn,
                                        n: m,
                                        k,
                                        a: tile_ptr as *const c_void,
                                        lda: k,
                                        stride_a: 0,
                                        b: x_ptr as *const c_void,
                                        ldb: k,
                                        stride_b: 0,
                                        c,
                                        ldc: n,
                                        stride_c: 0,
                                        batch: 1,
                                        batched: false,
                                    };
                                    match dev.gemm(&call)? {
                                        GemmOutcome::Done => {}
                                        GemmOutcome::Rocblas(_) => {
                                            if !matches!(dev.hip_gemm(&call)?, GemmOutcome::Done) {
                                                ok = false;
                                                break;
                                            }
                                        }
                                        GemmOutcome::NoKernel(_) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    n0 += tn;
                                }
                                drop(tile_lease);
                                if ok {
                                    let out_st = RocmStorage::new(out, in_dtype, dev.clone());
                                    return Ok(crate::tensor::from_storage(
                                        crate::Storage::Rocm(out_st),
                                        out_dims,
                                        crate::op::BackpropOp::none(),
                                        false,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = self.data.device() {
            if m > 0 && m <= 8 && self.layout.rows > 0 && matches!(in_dtype, DType::F32 | DType::F16 | DType::BF16) {
                let esz = in_dtype.size_in_bytes();
                let x_owned;
                let x_ref = if x.is_contiguous()
                    && (x.layout().start_offset() * esz).is_multiple_of(16)
                {
                    x
                } else {
                    x_owned = x.force_contiguous()?;
                    &x_owned
                };
                let (wst, _) = self.data.storage_and_layout();
                let (xst, xl) = x_ref.storage_and_layout();
                if let (crate::Storage::Cuda(ws), crate::Storage::Cuda(xs)) = (&*wst, &*xst) {
                    let out_st = super::w8a16::nvfp_mv(
                        dev,
                        ws.as_cuda_slice::<u8>()?,
                        self.layout.bscale_off,
                        self.layout.rscale_off,
                        self.layout.format == NvfpFormat::Nvfp4,
                        k,
                        self.layout.rows,
                        xs,
                        xl.start_offset(),
                        m,
                    )?;
                    return Ok(crate::tensor::from_storage(
                        crate::Storage::Cuda(out_st),
                        out_dims,
                        crate::op::BackpropOp::none(),
                        false,
                    ));
                }
            }
        }

        // Prefill / portable fallback: dequantize on device and multiply.
        let compute = if x.device().is_cpu() && in_dtype == DType::BF16 {
            DType::F32
        } else {
            in_dtype
        };
        let w = self.dequantize_to(compute)?;
        let x2 = x.to_dtype(compute)?.reshape((m, k))?;
        let y = x2.matmul(&w.t()?)?.reshape(out_dims)?;
        if y.dtype() == in_dtype {
            Ok(y)
        } else {
            y.to_dtype(in_dtype)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_and_nvfp8_cpu_roundtrip_and_forward() -> Result<()> {
        let rows = 32;
        let cols = 64;
        let w_data: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 37 + 13) % 97) as f32 - 48.0) * 0.03)
            .collect();
        let x_data: Vec<f32> = (0..4 * cols)
            .map(|i| (((i * 19 + 5) % 53) as f32 - 26.0) * 0.05)
            .collect();
        let w = Tensor::from_vec(w_data, (rows, cols), &Device::Cpu)?;
        let x = Tensor::from_vec(x_data, (4, cols), &Device::Cpu)?;

        for (fmt, max_rel) in [(NvfpFormat::Nvfp4, 0.16f32), (NvfpFormat::Nvfp8, 0.02f32)] {
            let qw = NvfpWeight::quantize(&w, fmt)?;
            assert_eq!(qw.layout.bscale_off % 256, 0);
            assert_eq!(qw.layout.rscale_off % 256, 0);
            let dq = qw.dequantize_to(DType::F32)?;
            let y_q = qw.forward(&x)?;
            let y_ref = x.matmul(&dq.t()?)?;
            let diff = (&y_q - &y_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 1e-5, "{fmt:?} forward vs dequant diff {diff}");

            let y_orig = x.matmul(&w.t()?)?;
            let rel = (&y_q - &y_orig)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt()
                / y_orig.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-6);
            assert!(rel < max_rel, "{fmt:?} relative error {rel} >= {max_rel}");
        }
        Ok(())
    }
}
