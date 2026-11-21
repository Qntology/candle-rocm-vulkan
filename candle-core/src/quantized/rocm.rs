//! Quantized (GGML block format) tensors stored on a ROCm device.
use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::rocm_backend::kernels::Module;
use crate::rocm_backend::utils::LaunchArgs;
use crate::rocm_backend::{GemmCall, GemmOutcome, RocmDevice, RocmStorage, WrapErr};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use hip_runtime::blas::GemmType;
use hip_runtime::memory::DeviceBuffer;
use std::ffi::c_void;

/// Inputs with at most this many rows use the fused dequantize + mat-vec kernel.
const QMV_MAX_ROWS: usize = 8;
/// Default upper bound for the temporary f32 weight tile used by the dequantize + gemm path.
/// Can be overridden via `CANDLE_ROCM_DEQUANT_TILE_MB` (or `CANDLE_W8A16_DEQUANT_TILE_MB`).
const DEFAULT_DEQUANT_TILE_BYTES: usize = 64 << 20;

fn dequant_tile_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("CANDLE_ROCM_DEQUANT_TILE_MB")
            .or_else(|_| std::env::var("CANDLE_W8A16_DEQUANT_TILE_MB"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .map(|mb| mb << 20)
            .unwrap_or(DEFAULT_DEQUANT_TILE_BYTES)
    })
}

pub struct QRocmStorage {
    data: DeviceBuffer<u8>,
    dtype: GgmlDType,
    device: RocmDevice,
}

fn type_suffix(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2k",
        GgmlDType::Q3K => "q3k",
        GgmlDType::Q4K => "q4k",
        GgmlDType::Q5K => "q5k",
        GgmlDType::Q6K => "q6k",
        GgmlDType::Q8K => "q8k",
    }
}

fn w8a16_q_suffix(dtype: GgmlDType) -> Option<&'static str> {
    match dtype {
        GgmlDType::Q8_0 => Some("q8_0"),
        GgmlDType::Q4_0 => Some("q4_0"),
        GgmlDType::Q4K => Some("q4k"),
        GgmlDType::Q5K => Some("q5k"),
        GgmlDType::Q6K => Some("q6k"),
        _ => None,
    }
}

fn w8a16_x_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        _ => None,
    }
}

impl QRocmStorage {
    pub fn zeros(device: &RocmDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let bytes = elem_count / dtype.block_size() * dtype.type_size();
        let data = device.alloc_zeros(bytes)?;
        Ok(Self {
            data,
            dtype,
            device: device.clone(),
        })
    }

    pub fn from_bytes(device: &RocmDevice, dtype: GgmlDType, bytes: &[u8]) -> Result<Self> {
        let data = device.upload(bytes)?;
        Ok(Self {
            data,
            dtype,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.byte_size()
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.as_ptr())
    }

    fn elem_count(&self) -> usize {
        self.data.byte_size() / self.dtype.type_size() * self.dtype.block_size()
    }

    fn dequantize_ptr(&self, src: *const u8, out: *mut u8, elem_count: usize, f16: bool) -> Result<()> {
        let name = format!(
            "dq_{}_{}",
            type_suffix(self.dtype),
            if f16 { "f16" } else { "f32" }
        );
        let mut args = LaunchArgs::new();
        args.ptr(src).ptr(out).usize(elem_count);
        self.device
            .launch_1d(Module::Quantized, &name, elem_count, &mut args)
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<RocmStorage> {
        let elem_count = elem_count.min(self.elem_count());
        let out = self.device.alloc(elem_count * 4)?;
        self.dequantize_ptr(self.data.as_ptr(), out.as_mut_ptr(), elem_count, false)?;
        Ok(RocmStorage::new(out, DType::F32, self.device.clone()))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<RocmStorage> {
        let elem_count = elem_count.min(self.elem_count());
        let out = self.device.alloc(elem_count * 2)?;
        self.dequantize_ptr(self.data.as_ptr(), out.as_mut_ptr(), elem_count, true)?;
        Ok(RocmStorage::new(out, DType::F16, self.device.clone()))
    }

    /// Dequantizes a block-aligned element slice `[elem0..elem0 + elem_count]` directly on GPU
    /// without allocating the full matrix (mirrors `QCudaStorage::dequantize_range_to`).
    pub fn dequantize_range_to(
        &self,
        elem0: usize,
        elem_count: usize,
        dtype: DType,
    ) -> Result<RocmStorage> {
        let bs = self.dtype.block_size();
        if elem0 % bs != 0 || elem_count % bs != 0 {
            crate::bail!(
                "rocm dequantize_range_to: range {elem0}..{} must align to block size {bs}",
                elem0 + elem_count
            );
        }
        let total = self.elem_count();
        if elem0.saturating_add(elem_count) > total {
            crate::bail!(
                "rocm dequantize_range_to: range {elem0}..{} exceeds {total}",
                elem0 + elem_count
            );
        }
        let byte_off = (elem0 / bs) * self.dtype.type_size();
        let src = unsafe { self.data.as_ptr().add(byte_off) };
        if let (Some(q_sfx), Some(x_sfx)) = (w8a16_q_suffix(self.dtype), w8a16_x_suffix(dtype)) {
            let esz = dtype.size_in_bytes();
            let out = self.device.alloc(elem_count.max(1) * esz)?;
            if elem_count > 0 {
                let nblocks = (elem_count / bs) as i64;
                let fname = format!("w8a16_dq_{q_sfx}_{x_sfx}");
                if let Ok(func) = self.device.get_func(Module::W8A16, &fname) {
                    let tasks_per_blk: usize = match self.dtype {
                        GgmlDType::Q8_0 => 4,
                        GgmlDType::Q4_0 => 2,
                        GgmlDType::Q4K | GgmlDType::Q5K => 16,
                        GgmlDType::Q6K => 8,
                        _ => unreachable!(),
                    };
                    let ntasks = (elem_count / bs) * tasks_per_blk;
                    let block = 256u32;
                    let grid = ntasks.div_ceil(block as usize).clamp(1, 65535) as u32;
                    let mut args = LaunchArgs::new();
                    args.ptr(src).ptr(out.as_mut_ptr()).bits(nblocks as u64);
                    self.device
                        .launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut args)?;
                    return Ok(RocmStorage::new(out, dtype, self.device.clone()));
                }
            } else {
                return Ok(RocmStorage::new(out, dtype, self.device.clone()));
            }
        }
        match dtype {
            DType::F32 => {
                let out = self.device.alloc(elem_count.max(1) * 4)?;
                if elem_count > 0 {
                    self.dequantize_ptr(src, out.as_mut_ptr(), elem_count, false)?;
                }
                Ok(RocmStorage::new(out, DType::F32, self.device.clone()))
            }
            DType::F16 => {
                let out = self.device.alloc(elem_count.max(1) * 2)?;
                if elem_count > 0 {
                    self.dequantize_ptr(src, out.as_mut_ptr(), elem_count, true)?;
                }
                Ok(RocmStorage::new(out, DType::F16, self.device.clone()))
            }
            other => {
                let f32_out = self.device.alloc(elem_count.max(1) * 4)?;
                if elem_count > 0 {
                    self.dequantize_ptr(src, f32_out.as_mut_ptr(), elem_count, false)?;
                }
                let s = RocmStorage::new(f32_out, DType::F32, self.device.clone());
                s.to_dtype(&Layout::contiguous((elem_count,)), other)
            }
        }
    }

    fn upload_quantized(&mut self, q: &dyn super::QuantizedType) -> Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(q.as_ptr(), q.storage_size_in_bytes()) };
        if bytes.len() != self.data.byte_size() {
            crate::bail!(
                "rocm quantize: size mismatch {} vs {}",
                bytes.len(),
                self.data.byte_size()
            )
        }
        self.data.copy_from_host(bytes).w()
    }

    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        let xs = src.as_slice::<f32>()?;
        let mut q = self.dtype.cpu_zeros(xs.len());
        q.from_float(xs);
        self.upload_quantized(q.as_ref())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let xs = src.as_slice::<f32>()?;
        let mut q = self.dtype.cpu_zeros(xs.len());
        q.from_float_imatrix(xs, imatrix_weights, n_per_row);
        self.upload_quantized(q.as_ref())
    }

    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()> {
        let cpu = src.to_cpu_storage()?;
        self.quantize_onto(&cpu)
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &RocmStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let cpu = src.to_cpu_storage()?;
        self.quantize_imatrix_onto(&cpu, imatrix_weights, n_per_row)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        self.data.to_vec().w()
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &RocmStorage,
        ids_l: &Layout,
    ) -> Result<RocmStorage> {
        if ids.dtype() != DType::U32 || !ids_l.is_contiguous() {
            crate::bail!("rocm quantized embedding expects contiguous u32 ids")
        }
        let n_ids = ids_l.shape().elem_count();
        let out = self.device.alloc(n_ids * hidden * 4)?;
        let name = format!("qembed_{}", type_suffix(self.dtype));
        let ids_ptr = unsafe { ids.device_ptr().add(ids_l.start_offset() * 4) };
        let mut args = LaunchArgs::new();
        args.ptr(self.data.as_ptr())
            .ptr(ids_ptr)
            .ptr(out.as_ptr())
            .usize(n_ids)
            .usize(hidden)
            .usize(rows);
        self.device
            .launch_1d(Module::Quantized, &name, n_ids * hidden, &mut args)?;
        Ok(RocmStorage::new(out, DType::F32, self.device.clone()))
    }

    fn w8a16_dequant_ptr(
        &self,
        src: *const u8,
        out: *mut u8,
        elem_count: usize,
        dtype: DType,
    ) -> Result<bool> {
        let (Some(q_sfx), Some(x_sfx)) = (w8a16_q_suffix(self.dtype), w8a16_x_suffix(dtype)) else {
            return Ok(false);
        };
        let bs = self.dtype.block_size();
        if !elem_count.is_multiple_of(bs) {
            return Ok(false);
        }
        let fname = format!("w8a16_dq_{q_sfx}_{x_sfx}");
        let Ok(func) = self.device.get_func(Module::W8A16, &fname) else {
            return Ok(false);
        };
        let tasks_per_blk: usize = match self.dtype {
            GgmlDType::Q8_0 => 4,
            GgmlDType::Q4_0 => 2,
            GgmlDType::Q4K | GgmlDType::Q5K => 16,
            GgmlDType::Q6K => 8,
            _ => return Ok(false),
        };
        let nblocks = (elem_count / bs) as i64;
        let ntasks = (elem_count / bs) * tasks_per_blk;
        let block = 256u32;
        let grid = ntasks.div_ceil(block as usize).clamp(1, 65535) as u32;
        let mut args = LaunchArgs::new();
        args.ptr(src).ptr(out).bits(nblocks as u64);
        self.device
            .launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut args)?;
        Ok(true)
    }

    /// Exact weight-only quantized forward (`y = x · dequant(W)ᵀ`) using single-source W8A16
    /// HIP kernels: direct F32/F16/BF16 matvec for `rows <= 8` (zero activation quantization and
    /// zero F32 round-trip casts), and direct tile dequantization + GEMM for `rows > 8`.
    pub fn fwd_exact(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        if src_shape.rank() < 1 {
            crate::bail!("input tensor has rank 0: {layout:?}")
        }
        let mut dst_dims = src_shape.dims().to_vec();
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let in_dtype = storage.dtype();
        let rows = src_shape.elem_count() / k.max(1);

        if let (Some(q_sfx), Some(x_sfx)) = (w8a16_q_suffix(self.dtype), w8a16_x_suffix(in_dtype)) {
            let bs = self.dtype.block_size();
            if k.is_multiple_of(bs) && n > 0 && rows > 0 {
                let esz = in_dtype.size_in_bytes();
                let x_owned;
                let align_req = if in_dtype == DType::F32 { 32 } else { 16 };
                let x_ptr: *const u8 = if layout.is_contiguous()
                    && (layout.start_offset() * esz).is_multiple_of(align_req)
                {
                    unsafe { storage.device_ptr().add(layout.start_offset() * esz) }
                } else {
                    x_owned = storage.contiguous_copy(layout)?;
                    x_owned.device_ptr()
                };

                if rows <= QMV_MAX_ROWS {
                    let m_tag = if rows == 1 { "1" } else { "8" };
                    let fname = format!("w8a16_mv{m_tag}_{q_sfx}_{x_sfx}");
                    if let Ok(func) = self.device.get_func(Module::W8A16, &fname) {
                        let out = self.device.alloc(rows * n * esz)?;
                        let row_bytes = (k / bs * self.dtype.type_size()) as i64;
                        let rpb = 2usize;
                        let wpr = 2usize;
                        let grid_x = n.div_ceil(rpb) as u32;
                        let mut args = LaunchArgs::new();
                        args.ptr(self.data.as_ptr())
                            .ptr(x_ptr)
                            .ptr(out.as_mut_ptr())
                            .u32(k as u32)
                            .u32(n as u32)
                            .u32(rows as u32)
                            .bits(row_bytes as u64);
                        self.device.launch(
                            func,
                            (grid_x, 1, 1),
                            (32, wpr as u32, rpb as u32),
                            0,
                            &mut args,
                        )?;
                        return Ok((RocmStorage::new(out, in_dtype, self.device.clone()), dst_shape));
                    }
                } else {
                    // Prefill (rows > 8): dequantize tiles directly to in_dtype into reusable
                    // workspace and run GemmType::{F32,F16,BF16} (F32 compute in rocBLAS/HIP).
                    let gemm_ty = match in_dtype {
                        DType::F32 => GemmType::F32,
                        DType::F16 => GemmType::F16,
                        DType::BF16 => GemmType::BF16,
                        _ => unreachable!(),
                    };
                    let row_bytes = k / bs * self.dtype.type_size();
                    let tile_n = (dequant_tile_bytes() / (k * esz).max(1)).clamp(1, n);
                    let tile_ptr = self
                        .device
                        .workspace_ensure("rocm_w8a16_tile", tile_n * k * esz)?;
                    let out = self.device.alloc(rows * n * esz)?;
                    let mut n0 = 0;
                    let mut half_gemm_ok = true;
                    while n0 < n {
                        let tn = tile_n.min(n - n0);
                        let src = unsafe { self.data.as_ptr().add(n0 * row_bytes) };
                        if !self.w8a16_dequant_ptr(src, tile_ptr, tn * k, in_dtype)? {
                            half_gemm_ok = false;
                            break;
                        }
                        let c = unsafe { out.as_mut_ptr().add(n0 * esz) } as *mut c_void;
                        let call = GemmCall {
                            ty: gemm_ty,
                            transa: true,
                            transb: false,
                            m: tn,
                            n: rows,
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
                        match self.device.gemm(&call)? {
                            GemmOutcome::Done => {}
                            GemmOutcome::Rocblas(_) | GemmOutcome::NoKernel(_) => {
                                half_gemm_ok = false;
                                break;
                            }
                        }
                        n0 += tn;
                    }
                    if half_gemm_ok {
                        return Ok((RocmStorage::new(out, in_dtype, self.device.clone()), dst_shape));
                    }
                }
            }
        }

        self.fwd_legacy(self_shape, storage, layout)
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        self.fwd_exact(self_shape, storage, layout)
    }

    fn fwd_legacy(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        if src_shape.rank() < 1 {
            crate::bail!("input tensor has rank 0 {layout:?}")
        }
        let mut dst_dims = src_shape.dims().to_vec();
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let in_dtype = storage.dtype();
        let x_owned;
        let x_ptr: *const u8 = if in_dtype == DType::F32 && layout.is_contiguous() {
            unsafe { storage.device_ptr().add(layout.start_offset() * 4) }
        } else {
            x_owned = storage.to_dtype(layout, DType::F32)?;
            x_owned.device_ptr()
        };
        let rows = src_shape.elem_count() / k;
        let out = self.device.alloc(rows * n * 4)?;
        if rows > 0 && n > 0 {
            if rows <= QMV_MAX_ROWS {
                let name = format!("qmv_{}", type_suffix(self.dtype));
                let func = self.device.get_func(Module::Quantized, &name)?;
                let block = k.next_power_of_two().clamp(32, 256) as u32;
                let mut args = LaunchArgs::new();
                args.ptr(self.data.as_ptr())
                    .ptr(x_ptr)
                    .ptr(out.as_ptr())
                    .usize(k)
                    .usize(n)
                    .usize(rows);
                let grid_x = n.min(1 << 30) as u32;
                self.device
                    .launch(func, (grid_x, rows as u32, 1), (block, 1, 1), block * 4, &mut args)?;
            } else {
                let row_bytes = k / self.dtype.block_size() * self.dtype.type_size();
                let tile_n = (dequant_tile_bytes() / (k * 4)).clamp(1, n);
                let tile_ptr = self
                    .device
                    .workspace_ensure("rocm_dq_tile_f32", tile_n * k * 4)?;
                let mut n0 = 0;
                while n0 < n {
                    let tn = tile_n.min(n - n0);
                    let src = unsafe { self.data.as_ptr().add(n0 * row_bytes) };
                    self.dequantize_ptr(src, tile_ptr, tn * k, false)?;
                    let c = unsafe { out.as_mut_ptr().add(n0 * 4) } as *mut c_void;
                    let call = GemmCall {
                        ty: GemmType::F32,
                        transa: true,
                        transb: false,
                        m: tn,
                        n: rows,
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
                    match self.device.gemm(&call)? {
                        GemmOutcome::Done => {}
                        GemmOutcome::Rocblas(e) => return Err(e).w(),
                        GemmOutcome::NoKernel(msg) => crate::bail!("rocm quantized matmul: {msg}"),
                    }
                    n0 += tn;
                }
            }
        }
        let out = RocmStorage::new(out, DType::F32, self.device.clone());
        if in_dtype == DType::F32 {
            Ok((out, dst_shape))
        } else {
            let out = out.to_dtype(&Layout::contiguous(&dst_shape), in_dtype)?;
            Ok((out, dst_shape))
        }
    }

    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        x: &RocmStorage,
        x_l: &Layout,
        ids: &RocmStorage,
        ids_l: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let (_n_experts, n, k) = self_shape.dims3()?;
        let (batch, topk) = ids_l.shape().dims2()?;
        if ids.dtype() != DType::U32 {
            crate::bail!("rocm indexed_moe_forward: ids must be U32, got {:?}", ids.dtype())
        }
        let x_dims = x_l.shape().dims();
        let x_has_topk = match x_dims {
            [b, k_in] if *b == batch && *k_in == k => false,
            [b, tk, k_in] if *b == batch && (*tk == 1 || *tk == topk) && *k_in == k => *tk == topk,
            _ => crate::bail!(
                "rocm indexed_moe_forward: incompatible x shape {:?} for batch={batch}, topk={topk}, k={k}",
                x_dims
            ),
        };
        let in_dtype = x.dtype();
        let out_shape = Shape::from((batch, topk, n));
        let slots = batch * topk;
        if slots == 0 || n == 0 {
            let out = self.device.alloc(0)?;
            return Ok((RocmStorage::new(out, in_dtype, self.device.clone()), out_shape));
        }

        let ids_owned;
        let ids_ptr = if ids_l.is_contiguous() {
            unsafe { ids.device_ptr().add(ids_l.start_offset() * 4) }
        } else {
            ids_owned = ids.contiguous_copy(ids_l)?;
            ids_owned.device_ptr()
        };

        if let (Some(q_sfx), Some(x_sfx)) = (w8a16_q_suffix(self.dtype), w8a16_x_suffix(in_dtype)) {
            let bs = self.dtype.block_size();
            if k.is_multiple_of(bs) {
                let fname = format!("w8a16_moe_{q_sfx}_{x_sfx}");
                if let Ok(func) = self.device.get_func(Module::W8A16, &fname) {
                    let esz = in_dtype.size_in_bytes();
                    let align_req = if in_dtype == DType::F32 { 32 } else { 16 };
                    let x_owned;
                    let x_ptr = if x_l.is_contiguous()
                        && (x_l.start_offset() * esz).is_multiple_of(align_req)
                    {
                        unsafe { x.device_ptr().add(x_l.start_offset() * esz) }
                    } else {
                        x_owned = x.contiguous_copy(x_l)?;
                        x_owned.device_ptr()
                    };
                    let out = self.device.alloc(slots * n * esz)?;
                    let row_bytes = (k / bs * self.dtype.type_size()) as i64;
                    let expert_bytes = (n as i64) * row_bytes;
                    let mut args = LaunchArgs::new();
                    args.ptr(self.data.as_ptr())
                        .ptr(x_ptr)
                        .ptr(ids_ptr)
                        .ptr(out.as_mut_ptr())
                        .u32(k as u32)
                        .u32(n as u32)
                        .u32(topk as u32)
                        .u32(u32::from(x_has_topk))
                        .bits(row_bytes as u64)
                        .bits(expert_bytes as u64);
                    let grid_x = n.div_ceil(4) as u32;
                    let grid_y = slots as u32;
                    self.device
                        .launch(func, (grid_x, grid_y, 1), (32, 4, 1), 0, &mut args)?;
                    return Ok((RocmStorage::new(out, in_dtype, self.device.clone()), out_shape));
                }
            }
        }

        // General fallback for remaining GgmlDTypes (Q2K, Q3K, Q8K, etc.): dequantize per unique
        // expert row slice on GPU and multiply via qmv/gemm without CPU round-trip.
        let ids_host = ids.to_cpu_storage()?;
        let ids_u32 = match &ids_host {
            CpuStorage::U32(v) => v,
            _ => unreachable!(),
        };
        let x_f32 = if in_dtype == DType::F32 && x_l.is_contiguous() {
            None
        } else {
            Some(x.to_dtype(x_l, DType::F32)?)
        };
        let x_base = match &x_f32 {
            Some(s) => s.device_ptr(),
            None => unsafe { x.device_ptr().add(x_l.start_offset() * 4) },
        };
        let bs = self.dtype.block_size();
        let row_bytes = k / bs * self.dtype.type_size();
        let expert_bytes = n * row_bytes;
        let out_f32 = self.device.alloc(slots * n * 4)?;
        let qmv_name = format!("qmv_{}", type_suffix(self.dtype));
        let qmv_fn = self.device.get_func(Module::Quantized, &qmv_name)?;
        let block = k.next_power_of_two().clamp(32, 256) as u32;
        for slot in 0..slots {
            let e_idx = ids_u32[ids_l.start_offset() + slot] as usize;
            let x_row = if x_has_topk { slot } else { slot / topk };
            let w_ptr = unsafe { self.data.as_ptr().add(e_idx * expert_bytes) };
            let xp = unsafe { x_base.add(x_row * k * 4) };
            let yp = unsafe { out_f32.as_mut_ptr().add(slot * n * 4) };
            let mut args = LaunchArgs::new();
            args.ptr(w_ptr).ptr(xp).ptr(yp).usize(k).usize(n).usize(1);
            self.device.launch(
                qmv_fn,
                (n.min(1 << 30) as u32, 1, 1),
                (block, 1, 1),
                block * 4,
                &mut args,
            )?;
        }
        let out = RocmStorage::new(out_f32, DType::F32, self.device.clone());
        if in_dtype == DType::F32 {
            Ok((out, out_shape))
        } else {
            Ok((out.to_dtype(&Layout::contiguous(&out_shape), in_dtype)?, out_shape))
        }
    }
}

pub fn load_quantized(device: &RocmDevice, dtype: GgmlDType, data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Rocm(QRocmStorage::from_bytes(device, dtype, data)?))
}
