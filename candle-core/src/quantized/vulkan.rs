//! Quantized (GGML block format) tensors stored on a Vulkan device.
//!
//! `Q8_0` weights are repacked into GPU memory (VRAM on a discrete GPU, the carve-out
//! or shared memory on an integrated one, see `vulkan_backend::qgpu`) and run on GPU
//! kernels. Other block types, and `Q8_0` when no GPU memory is available, are kept as
//! raw bytes in a mapped host visible buffer and run on the CPU k-quant kernels directly
//! on the mapped memory.
use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType,
};
use super::{GgmlDType, QStorage, QuantizedType};
use crate::backend::BackendStorage;
use crate::vulkan_backend::qgpu::{self, GpuQ8};
use crate::{CpuStorage, DType, Layout, Result, Shape, VulkanDevice, VulkanStorage};
use half::{bf16, f16};

pub struct QVulkanStorage {
    /// GGML blocks in host visible memory, read by the CPU k-quant kernels.
    data: Option<VulkanStorage>,
    /// `Q8_0` blocks repacked into GPU memory, read by the GPU kernels.
    gpu: Option<GpuQ8>,
    dtype: GgmlDType,
    device: VulkanDevice,
}

macro_rules! with_blocks {
    ($dtype:expr, $bytes:expr, |$blocks:ident| $body:expr) => {
        match $dtype {
            GgmlDType::F32 => {
                let $blocks = cast::<f32>($bytes)?;
                $body
            }
            GgmlDType::F16 => {
                let $blocks = cast::<f16>($bytes)?;
                $body
            }
            GgmlDType::BF16 => {
                let $blocks = cast::<bf16>($bytes)?;
                $body
            }
            GgmlDType::Q4_0 => {
                let $blocks = cast::<BlockQ4_0>($bytes)?;
                $body
            }
            GgmlDType::Q4_1 => {
                let $blocks = cast::<BlockQ4_1>($bytes)?;
                $body
            }
            GgmlDType::Q5_0 => {
                let $blocks = cast::<BlockQ5_0>($bytes)?;
                $body
            }
            GgmlDType::Q5_1 => {
                let $blocks = cast::<BlockQ5_1>($bytes)?;
                $body
            }
            GgmlDType::Q8_0 => {
                let $blocks = cast::<BlockQ8_0>($bytes)?;
                $body
            }
            GgmlDType::Q8_1 => {
                let $blocks = cast::<BlockQ8_1>($bytes)?;
                $body
            }
            GgmlDType::Q2K => {
                let $blocks = cast::<BlockQ2K>($bytes)?;
                $body
            }
            GgmlDType::Q3K => {
                let $blocks = cast::<BlockQ3K>($bytes)?;
                $body
            }
            GgmlDType::Q4K => {
                let $blocks = cast::<BlockQ4K>($bytes)?;
                $body
            }
            GgmlDType::Q5K => {
                let $blocks = cast::<BlockQ5K>($bytes)?;
                $body
            }
            GgmlDType::Q6K => {
                let $blocks = cast::<BlockQ6K>($bytes)?;
                $body
            }
            GgmlDType::Q8K => {
                let $blocks = cast::<BlockQ8K>($bytes)?;
                $body
            }
        }
    };
}

fn cast<T>(bytes: &[u8]) -> Result<&[T]> {
    let size = std::mem::size_of::<T>();
    if bytes.len() % size != 0 {
        crate::bail!(
            "vulkan quantized storage: {} bytes is not a multiple of {size}",
            bytes.len()
        )
    }
    if bytes.is_empty() {
        return Ok(&[]);
    }
    if (bytes.as_ptr() as usize) % std::mem::align_of::<T>() != 0 {
        crate::bail!("vulkan quantized storage: misaligned block data")
    }
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size) })
}

fn dequantize_into<T: GgmlType>(blocks: &[T], out: &mut [f32]) -> Result<()> {
    let n = out.len();
    if n % T::BLCK_SIZE != 0 || n / T::BLCK_SIZE > blocks.len() {
        crate::bail!(
            "vulkan dequantize: {n} elements do not fit {} blocks of {}",
            blocks.len(),
            T::BLCK_SIZE
        )
    }
    T::to_float(&blocks[..n / T::BLCK_SIZE], out);
    Ok(())
}

impl QVulkanStorage {
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let bytes = elem_count / dtype.block_size() * dtype.type_size();
        let mut data = device.alloc_buffer(bytes, DType::U8)?;
        data.as_bytes_mut().fill(0);
        Ok(Self {
            data: Some(data),
            gpu: None,
            dtype,
            device: device.clone(),
        })
    }

    pub fn from_bytes(device: &VulkanDevice, dtype: GgmlDType, bytes: &[u8]) -> Result<Self> {
        let mut _p = crate::vulkan_backend::prof::scope("qtensor", if dtype == GgmlDType::Q8_0 { "load_q8" } else { "load_other(host)" });
        _p.work(bytes.len());
        if dtype == GgmlDType::Q8_0 && qgpu::want_q8_on_gpu(device, bytes.len()) {
            match GpuQ8::from_ggml(device, bytes) {
                Ok(g) => {
                    return Ok(Self {
                        data: None,
                        gpu: Some(g),
                        dtype,
                        device: device.clone(),
                    })
                }
                Err(e) => qgpu::note_fallback("q8 weights to GPU memory", &e),
            }
        }
        let data = device.upload_bytes(bytes, bytes.len(), DType::U8)?;
        Ok(Self {
            data: Some(data),
            gpu: None,
            dtype,
            device: device.clone(),
        })
    }

    /// Moves host `Q8_0` blocks into GPU memory when that is wanted and possible.
    fn maybe_promote(&mut self) {
        if self.gpu.is_some() || self.dtype != GgmlDType::Q8_0 {
            return;
        }
        let Some(data) = &self.data else { return };
        if !qgpu::want_q8_on_gpu(&self.device, data.capacity_bytes) {
            return;
        }
        match GpuQ8::from_ggml(&self.device, data.as_bytes()) {
            Ok(g) => {
                self.gpu = Some(g);
                self.data = None;
            }
            Err(e) => qgpu::note_fallback("q8 weights to GPU memory", &e),
        }
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Whether the blocks live in GPU memory and run on the GPU kernels.
    pub fn is_on_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        match (&self.data, &self.gpu) {
            (Some(d), _) => d.capacity_bytes,
            (None, Some(g)) => g.ggml_bytes(),
            (None, None) => 0,
        }
    }

    fn elem_count(&self) -> usize {
        self.storage_size_in_bytes() / self.dtype.type_size() * self.dtype.block_size()
    }

    /// Runs `f` on the GGML bytes, downloading them from GPU memory when needed.
    fn with_host_bytes<R>(&self, f: impl FnOnce(&[u8]) -> Result<R>) -> Result<R> {
        match (&self.data, &self.gpu) {
            (Some(d), _) => f(d.as_bytes()),
            (None, Some(g)) => {
                let v = g.to_ggml()?;
                // mapped memory: block aligned, unlike a plain Vec<u8>
                let tmp = self.device.upload_bytes(&v, v.len(), DType::U8)?;
                f(tmp.as_bytes())
            }
            (None, None) => crate::bail!("vulkan quantized storage holds no data"),
        }
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        let _p = crate::vulkan_backend::prof::scope("qtensor", "dequantize");
        let elem_count = elem_count.min(self.elem_count());
        if let Some(g) = &self.gpu {
            match g.dequantize(elem_count, DType::F32, &self.device) {
                Ok(s) => return Ok(s),
                Err(e) => qgpu::note_fallback("q8 dequantize", &e),
            }
        }
        let mut out = self.device.alloc_buffer(elem_count, DType::F32)?;
        let dtype = self.dtype;
        self.with_host_bytes(|bytes| {
            let dst = out.as_mut_slice::<f32>()?;
            with_blocks!(dtype, bytes, |blocks| dequantize_into(blocks, dst))
        })?;
        Ok(out)
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<VulkanStorage> {
        if let Some(g) = &self.gpu {
            let n = elem_count.min(self.elem_count());
            match g.dequantize(n, DType::F16, &self.device) {
                Ok(s) => return Ok(s),
                Err(e) => qgpu::note_fallback("q8 dequantize to f16", &e),
            }
        }
        let f32s = self.dequantize(elem_count)?;
        let src = f32s.as_slice::<f32>()?;
        let mut out = self.device.alloc_buffer(src.len(), DType::F16)?;
        for (d, s) in out.as_mut_slice::<f16>()?.iter_mut().zip(src.iter()) {
            *d = f16::from_f32(*s);
        }
        Ok(out)
    }

    fn write_quantized(&mut self, q: &dyn QuantizedType) -> Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(q.as_ptr(), q.storage_size_in_bytes()) };
        let expected = self.storage_size_in_bytes();
        if bytes.len() != expected {
            crate::bail!(
                "vulkan quantize: size mismatch {} vs {}",
                bytes.len(),
                expected
            )
        }
        // new content: the GPU copy (if any) is rebuilt from the host blocks
        self.gpu = None;
        match &mut self.data {
            Some(d) => d.as_bytes_mut().copy_from_slice(bytes),
            None => self.data = Some(self.device.upload_bytes(bytes, bytes.len(), DType::U8)?),
        }
        self.maybe_promote();
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        let xs = src.as_slice::<f32>()?;
        let mut q = self.dtype.cpu_zeros(xs.len());
        q.from_float(xs);
        self.write_quantized(q.as_ref())
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
        self.write_quantized(q.as_ref())
    }

    pub fn quantize(&mut self, src: &VulkanStorage) -> Result<()> {
        let xs = src.as_slice::<f32>()?;
        let mut q = self.dtype.cpu_zeros(xs.len());
        q.from_float(xs);
        self.write_quantized(q.as_ref())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &VulkanStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let xs = src.as_slice::<f32>()?;
        let mut q = self.dtype.cpu_zeros(xs.len());
        q.from_float_imatrix(xs, imatrix_weights, n_per_row);
        self.write_quantized(q.as_ref())
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        self.with_host_bytes(|b| Ok(b.to_vec()))
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &VulkanStorage,
        ids_l: &Layout,
    ) -> Result<VulkanStorage> {
        let ids: Vec<u32> = match ids.dtype {
            DType::U32 => {
                let s = ids.as_slice::<u32>()?;
                match ids_l.contiguous_offsets() {
                    Some((a, b)) => s[a..b].to_vec(),
                    None => ids_l.strided_index().map(|i| s[i]).collect(),
                }
            }
            dt => crate::bail!("vulkan quantized embedding expects u32 ids, got {dt:?}"),
        };
        let block = self.dtype.block_size();
        if hidden % block != 0 {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {block}"
            )
        }
        if let Some(g) = &self.gpu {
            match g.embedding(rows, hidden, &ids, &self.device) {
                Ok(s) => return Ok(s),
                Err(e) => qgpu::note_fallback("q8 embedding", &e),
            }
        }
        let row_bytes = hidden / block * self.dtype.type_size();
        if self.storage_size_in_bytes() != rows * row_bytes {
            crate::bail!("vulkan quantized embedding: storage does not hold {rows}x{hidden} values")
        }
        let mut out = self.device.alloc_buffer(ids.len() * hidden, DType::F32)?;
        let dtype = self.dtype;
        self.with_host_bytes(|src| {
            let dst = out.as_mut_slice::<f32>()?;
            for (o, &id) in ids.iter().enumerate() {
                let id = id as usize;
                if id >= rows {
                    crate::bail!("embedding id {id} is out of range for {rows} rows")
                }
                let row = &src[id * row_bytes..(id + 1) * row_bytes];
                let dst = &mut dst[o * hidden..(o + 1) * hidden];
                with_blocks!(dtype, row, |blocks| dequantize_into(blocks, dst))?;
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// `x @ self^T` where `self` is a `(n, k)` quantized matrix.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &VulkanStorage,
        layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        let (dst_shape, mkn) = qmatmul_shapes(self_shape, layout)?;
        let mut _p = crate::vulkan_backend::prof::scope("qtensor", "matmul_gpu");
        _p.work(2 * mkn.0 * mkn.1 * mkn.2);
        if let Some(g) = &self.gpu {
            match g.matmul(mkn, storage, layout) {
                Ok(out) => {
                    let in_dtype = storage.dtype();
                    if in_dtype == DType::F32 {
                        return Ok((out, dst_shape));
                    }
                    let out = out.to_dtype(&Layout::contiguous(&dst_shape), in_dtype)?;
                    return Ok((out, dst_shape));
                }
                Err(e) => qgpu::note_fallback("q8 matmul", &e),
            }
        }
        _p.rename("qtensor", "matmul_cpu");
        let dtype = self.dtype;
        self.with_host_bytes(|weights| {
            qmatmul(storage, layout, &dst_shape, mkn, |lhs, dst| {
                with_blocks!(dtype, weights, |blocks| super::k_quants::matmul(
                    mkn, lhs, blocks, dst
                ))
            })
        })
    }

    pub fn indexed_moe_forward(
        &self,
        _: &Shape,
        _: &VulkanStorage,
        _: &Layout,
        _: &VulkanStorage,
        _: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        crate::bail!("indexed_moe_forward is not implemented on the Vulkan backend")
    }
}


pub(crate) fn qmatmul_shapes(
    self_shape: &Shape,
    layout: &Layout,
) -> Result<(Shape, (usize, usize, usize))> {
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
    let m = dst_shape.elem_count() / n.max(1);
    Ok((dst_shape, (m, k, n)))
}

/// Runs `f(lhs_f32, dst_f32)` on the input activations and returns the result
/// in the input dtype. Contiguous f32 inputs are read in place.
pub(crate) fn qmatmul(
    storage: &VulkanStorage,
    layout: &Layout,
    dst_shape: &Shape,
    (m, _k, n): (usize, usize, usize),
    f: impl FnOnce(&[f32], &mut [f32]) -> Result<()>,
) -> Result<(VulkanStorage, Shape)> {
    let dev = storage.device().clone();
    let in_dtype = storage.dtype();
    let mut out = dev.alloc_buffer(m * n, DType::F32)?;
    let owned;
    let lhs: &[f32] = match (in_dtype, layout.contiguous_offsets()) {
        (DType::F32, Some((a, b))) => &storage.as_slice::<f32>()?[a..b],
        _ => {
            let view = storage.host_view();
            owned = view.to_dtype(layout, DType::F32)?;
            owned.as_slice::<f32>()?
        }
    };
    if m > 0 && n > 0 {
        f(lhs, out.as_mut_slice::<f32>()?)?;
    }
    if in_dtype == DType::F32 {
        return Ok((out, dst_shape.clone()));
    }
    let out = out.to_dtype(&Layout::contiguous(dst_shape), in_dtype)?;
    Ok((out, dst_shape.clone()))
}

pub fn load_quantized(device: &VulkanDevice, dtype: GgmlDType, data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Vulkan(QVulkanStorage::from_bytes(
        device, dtype, data,
    )?))
}

/// Matmul of activations stored on a Vulkan device with quantized weights that
/// stayed in CPU memory.
pub fn fwd_cpu_weights(
    weights: &dyn QuantizedType,
    self_shape: &Shape,
    storage: &VulkanStorage,
    layout: &Layout,
) -> Result<(VulkanStorage, Shape)> {
    let (dst_shape, mkn) = qmatmul_shapes(self_shape, layout)?;
    qmatmul(storage, layout, &dst_shape, mkn, |lhs, dst| {
        weights.matmul_t(mkn, lhs, dst)
    })
}
