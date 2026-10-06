//! Quantized (GGML block format) tensors stored on a Vulkan device.
//!
//! The blocks are kept as raw bytes in a mapped host visible buffer, so a
//! quantized weight uses the same amount of memory as in the GGUF file. The
//! matmul and dequantization run the CPU k-quant kernels directly on the
//! mapped memory, without copying the weights.
use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType,
};
use super::{GgmlDType, QStorage, QuantizedType};
use crate::backend::BackendStorage;
use crate::{CpuStorage, DType, Layout, Result, Shape, VulkanDevice, VulkanStorage};
use half::{bf16, f16};

pub struct QVulkanStorage {
    data: VulkanStorage,
    dtype: GgmlDType,
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
        Ok(Self { data, dtype })
    }

    pub fn from_bytes(device: &VulkanDevice, dtype: GgmlDType, bytes: &[u8]) -> Result<Self> {
        let data = device.upload_bytes(bytes, bytes.len(), DType::U8)?;
        Ok(Self { data, dtype })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.data.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.capacity_bytes
    }

    fn elem_count(&self) -> usize {
        self.data.capacity_bytes / self.dtype.type_size() * self.dtype.block_size()
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        let elem_count = elem_count.min(self.elem_count());
        let mut out = self.device().alloc_buffer(elem_count, DType::F32)?;
        let dst = out.as_mut_slice::<f32>()?;
        with_blocks!(self.dtype, self.data.as_bytes(), |blocks| dequantize_into(
            blocks, dst
        ))?;
        Ok(out)
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<VulkanStorage> {
        let f32s = self.dequantize(elem_count)?;
        let src = f32s.as_slice::<f32>()?;
        let mut out = self.device().alloc_buffer(src.len(), DType::F16)?;
        for (d, s) in out.as_mut_slice::<f16>()?.iter_mut().zip(src.iter()) {
            *d = f16::from_f32(*s);
        }
        Ok(out)
    }

    fn write_quantized(&mut self, q: &dyn QuantizedType) -> Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(q.as_ptr(), q.storage_size_in_bytes()) };
        if bytes.len() != self.data.capacity_bytes {
            crate::bail!(
                "vulkan quantize: size mismatch {} vs {}",
                bytes.len(),
                self.data.capacity_bytes
            )
        }
        self.data.as_bytes_mut().copy_from_slice(bytes);
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
        Ok(self.data.as_bytes().to_vec())
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
        let row_bytes = hidden / block * self.dtype.type_size();
        if self.data.capacity_bytes != rows * row_bytes {
            crate::bail!("vulkan quantized embedding: storage does not hold {rows}x{hidden} values")
        }
        let mut out = self.device().alloc_buffer(ids.len() * hidden, DType::F32)?;
        let dst = out.as_mut_slice::<f32>()?;
        let src = self.data.as_bytes();
        for (o, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= rows {
                crate::bail!("embedding id {id} is out of range for {rows} rows")
            }
            let row = &src[id * row_bytes..(id + 1) * row_bytes];
            let dst = &mut dst[o * hidden..(o + 1) * hidden];
            with_blocks!(self.dtype, row, |blocks| dequantize_into(blocks, dst))?;
        }
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
        let weights = self.data.as_bytes();
        qmatmul(storage, layout, &dst_shape, mkn, |lhs, dst| {
            with_blocks!(self.dtype, weights, |blocks| super::k_quants::matmul(
                mkn, lhs, blocks, dst
            ))
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
