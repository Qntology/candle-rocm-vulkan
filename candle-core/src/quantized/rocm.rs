//! Quantized (GGML block format) tensors stored on a ROCm device.
use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::rocm_backend::kernels::Module;
use crate::rocm_backend::utils::LaunchArgs;
use crate::rocm_backend::{RocmDevice, RocmStorage, WrapErr};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use hip_runtime::memory::DeviceBuffer;
use std::ffi::c_void;

/// Inputs with at most this many rows use the fused dequantize + mat-vec kernel.
const QMV_MAX_ROWS: usize = 8;
/// Upper bound for the temporary f32 weight tile used by the dequantize + gemm path.
const MAX_DEQUANT_TILE_BYTES: usize = 128 << 20;

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

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &RocmStorage,
        layout: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
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
                let tile_n = (MAX_DEQUANT_TILE_BYTES / (k * 4)).clamp(1, n);
                let tile = self.device.alloc(tile_n * k * 4)?;
                let mut n0 = 0;
                while n0 < n {
                    let tn = tile_n.min(n - n0);
                    let src = unsafe { self.data.as_ptr().add(n0 * row_bytes) };
                    self.dequantize_ptr(src, tile.as_mut_ptr(), tn * k, false)?;
                    let c = unsafe { out.as_mut_ptr().add(n0 * 4) } as *mut c_void;
                    self.device.with_blas(|blas| {
                        unsafe {
                            blas.sgemm_raw(
                                true,
                                false,
                                tn,
                                rows,
                                k,
                                1.0,
                                tile.as_void_ptr(),
                                k,
                                x_ptr as *const c_void,
                                k,
                                0.0,
                                c,
                                n,
                            )
                        }
                        .w()
                    })?;
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
        _: &Shape,
        _: &RocmStorage,
        _: &Layout,
        _: &RocmStorage,
        _: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        crate::bail!("indexed_moe_forward is not implemented on the ROCm backend")
    }
}

pub fn load_quantized(device: &RocmDevice, dtype: GgmlDType, data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Rocm(QRocmStorage::from_bytes(device, dtype, data)?))
}
