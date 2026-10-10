use super::kernels::Module;
use super::utils::{
    cast_suffix, dtype_suffix, elem_bytes, grid_1d, index_suffix, is_float, size_suffix, LaunchArgs, StridedInfo,
};
use super::{GemmCall, GemmOutcome};
use super::{RocmDevice, RocmError, WrapErr};
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use hip_runtime::blas::GemmType;
use hip_runtime::memory::DeviceBuffer;
use std::ffi::c_void;

/// GPU resident tensor storage: a raw device allocation plus its element type.
pub struct RocmStorage {
    pub(crate) buf: DeviceBuffer<u8>,
    pub(crate) dtype: DType,
    pub(crate) device: RocmDevice,
}

impl std::fmt::Debug for RocmStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmStorage")
            .field("dtype", &self.dtype)
            .field("bytes", &self.buf.byte_size())
            .field("device", &self.device)
            .finish()
    }
}

/// Which rotary embedding layout to use in [`RocmStorage::rope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// `(b, h, t, d)` with interleaved pairs.
    Interleaved,
    /// `(b, h, t, d)` with the two halves of the last dim rotated together.
    Halves,
    /// `(b, t, h, d)` with the two halves of the last dim rotated together.
    Thd,
}

fn info1(layout: &Layout) -> Option<StridedInfo<1>> {
    if layout.is_contiguous() {
        return Some(StridedInfo::<1>::contiguous());
    }
    StridedInfo::<1>::new(layout.dims(), [layout.stride()])
}

fn cpu_bytes(s: &CpuStorage) -> &[u8] {
    fn b<T>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    match s {
        CpuStorage::U8(v) => v,
        CpuStorage::U32(v) => b(v),
        CpuStorage::I16(v) => b(v),
        CpuStorage::I32(v) => b(v),
        CpuStorage::I64(v) => b(v),
        CpuStorage::BF16(v) => b(v),
        CpuStorage::F16(v) => b(v),
        CpuStorage::F32(v) => b(v),
        CpuStorage::F64(v) => b(v),
        CpuStorage::F8E4M3(v) => b(v),
        CpuStorage::F6E2M3(v) | CpuStorage::F6E3M2(v) | CpuStorage::F4(v) | CpuStorage::F8E8M0(v) => {
            v
        }
    }
}

fn download<T: Copy>(src: *const u8, bytes: usize) -> Result<Vec<T>> {
    let n = bytes / std::mem::size_of::<T>();
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        hip_runtime::memory::memcpy_dtoh(v.as_mut_ptr() as *mut c_void, src as *const c_void, bytes).w()?;
        v.set_len(n);
    }
    Ok(v)
}

fn scalar_bits(s: crate::scalar::Scalar) -> u64 {
    use crate::scalar::Scalar;
    match s {
        Scalar::U8(v) => v as u64,
        Scalar::U32(v) => v as u64,
        Scalar::I16(v) => v as u16 as u64,
        Scalar::I32(v) => v as u32 as u64,
        Scalar::I64(v) => v as u64,
        Scalar::BF16(v) => v.to_bits() as u64,
        Scalar::F16(v) => v.to_bits() as u64,
        Scalar::F32(v) => v.to_bits() as u64,
        Scalar::F64(v) => v.to_bits(),
        Scalar::F8E4M3(v) => v.to_bits() as u64,
    }
}

struct GemmCfg {
    transa: bool,
    transb: bool,
    lda: usize,
    ldb: usize,
    stride_a: i64,
    stride_b: i64,
}

fn gemm_config(
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Option<GemmCfg> {
    let _ = b;
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    if lhs_stride.len() < 2 || rhs_stride.len() < 2 {
        return None;
    }
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k, true)
    } else {
        return None;
    };
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m, true)
    } else {
        return None;
    };
    let lhs_dims = lhs_l.dims();
    let rhs_dims = rhs_l.dims();
    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_dims[1] => stride,
        [_, stride] if lhs_dims[0] == 1 => stride,
        [stride, _] if lhs_dims[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => return None,
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_dims[1] => stride,
        [_, stride] if rhs_dims[0] == 1 => stride,
        [stride, _] if rhs_dims[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => return None,
    };
    Some(GemmCfg {
        transa,
        transb,
        lda: lda.max(1),
        ldb: ldb.max(1),
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
    })
}

impl RocmStorage {
    pub(crate) fn new(buf: DeviceBuffer<u8>, dtype: DType, device: RocmDevice) -> Self {
        Self { buf, dtype, device }
    }

    /// Raw device pointer to the start of the allocation.
    pub fn device_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    pub fn device_ptr_mut(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    pub fn byte_len(&self) -> usize {
        self.buf.byte_size()
    }

    pub fn elem_count(&self) -> usize {
        self.buf.byte_size() / elem_bytes(self.dtype).max(1)
    }

    pub fn rocm_device(&self) -> &RocmDevice {
        &self.device
    }

    fn esz(&self) -> usize {
        elem_bytes(self.dtype)
    }

    fn ptr_at(&self, elem_offset: usize) -> *const u8 {
        unsafe { self.buf.as_ptr().add(elem_offset * self.esz()) }
    }

    fn ptr_at_mut(&self, elem_offset: usize) -> *mut u8 {
        unsafe { self.buf.as_mut_ptr().add(elem_offset * self.esz()) }
    }

    pub(crate) fn from_cpu(device: &RocmDevice, storage: &CpuStorage) -> Result<Self> {
        let buf = device.upload(cpu_bytes(storage))?;
        Ok(Self::new(buf, storage.dtype(), device.clone()))
    }

    pub(crate) fn to_cpu(&self) -> Result<CpuStorage> {
        let src = self.buf.as_ptr();
        let bytes = self.buf.byte_size();
        let s = match self.dtype {
            DType::U8 => CpuStorage::U8(download::<u8>(src, bytes)?),
            DType::U32 => CpuStorage::U32(download::<u32>(src, bytes)?),
            DType::I16 => CpuStorage::I16(download::<i16>(src, bytes)?),
            DType::I32 => CpuStorage::I32(download::<i32>(src, bytes)?),
            DType::I64 => CpuStorage::I64(download::<i64>(src, bytes)?),
            DType::BF16 => CpuStorage::BF16(download::<half::bf16>(src, bytes)?),
            DType::F16 => CpuStorage::F16(download::<half::f16>(src, bytes)?),
            DType::F32 => CpuStorage::F32(download::<f32>(src, bytes)?),
            DType::F64 => CpuStorage::F64(download::<f64>(src, bytes)?),
            DType::F8E4M3 => CpuStorage::F8E4M3(download::<float8::F8E4M3>(src, bytes)?),
            DType::F6E2M3 => CpuStorage::F6E2M3(download::<u8>(src, bytes)?),
            DType::F6E3M2 => CpuStorage::F6E3M2(download::<u8>(src, bytes)?),
            DType::F4 => CpuStorage::F4(download::<u8>(src, bytes)?),
            DType::F8E8M0 => CpuStorage::F8E8M0(download::<u8>(src, bytes)?),
        };
        Ok(s)
    }

    /// Replaces the content of this storage with `cpu` (same dtype and size).
    pub(crate) fn overwrite_from_cpu(&mut self, cpu: &CpuStorage) -> Result<()> {
        let bytes = cpu_bytes(cpu);
        if cpu.dtype() != self.dtype || bytes.len() != self.buf.byte_size() {
            crate::bail!(
                "rocm overwrite_from_cpu: mismatch {:?}/{} vs {:?}/{}",
                cpu.dtype(),
                bytes.len(),
                self.dtype,
                self.buf.byte_size()
            )
        }
        self.buf.copy_from_host(bytes).w()
    }

    fn upload_cpu(&self, cpu: &CpuStorage) -> Result<Self> {
        Self::from_cpu(&self.device, cpu)
    }

    fn fallback1(&self, f: impl FnOnce(&CpuStorage) -> Result<CpuStorage>) -> Result<Self> {
        let cpu = self.to_cpu()?;
        let out = f(&cpu)?;
        self.upload_cpu(&out)
    }

    fn fallback2(
        &self,
        rhs: &Self,
        f: impl FnOnce(&CpuStorage, &CpuStorage) -> Result<CpuStorage>,
    ) -> Result<Self> {
        let a = self.to_cpu()?;
        let b = rhs.to_cpu()?;
        let out = f(&a, &b)?;
        self.upload_cpu(&out)
    }

    /// CPU fallback for ops whose CPU implementation relies on a matmul (no BF16 support on the
    /// CPU backend): half precision inputs are widened to F32 and the result narrowed back.
    fn fallback2_wide(
        &self,
        l: &Layout,
        rhs: &Self,
        rhs_l: &Layout,
        f: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<CpuStorage>,
    ) -> Result<Self> {
        let a = self.to_cpu()?;
        let b = rhs.to_cpu()?;
        if !matches!(self.dtype, DType::F16 | DType::BF16) {
            let out = f(&a, l, &b, rhs_l)?;
            return self.upload_cpu(&out);
        }
        let la = Layout::contiguous(l.shape());
        let lb = Layout::contiguous(rhs_l.shape());
        let a32 = a.to_dtype(l, DType::F32)?;
        let b32 = b.to_dtype(rhs_l, DType::F32)?;
        let out = f(&a32, &la, &b32, &lb)?;
        let n = out.as_slice::<f32>()?.len();
        let out = out.to_dtype(&Layout::contiguous(n), self.dtype)?;
        self.upload_cpu(&out)
    }

    fn fallback3(
        &self,
        s2: &Self,
        s3: &Self,
        f: impl FnOnce(&CpuStorage, &CpuStorage, &CpuStorage) -> Result<CpuStorage>,
    ) -> Result<Self> {
        let a = self.to_cpu()?;
        let b = s2.to_cpu()?;
        let c = s3.to_cpu()?;
        let out = f(&a, &b, &c)?;
        self.upload_cpu(&out)
    }

    /// Half precision GEMM through F32 when rocBLAS has no `gemm_ex` kernel for this GPU.
    fn half_matmul_via_f32(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "candle-rocm: rocBLAS has no {:?} gemm_ex kernel for this GPU, using F32 GEMM instead",
                self.dtype
            );
        }
        let (b, m, n, _) = bmnk;
        let lhs32 = self.to_dtype(lhs_l, DType::F32)?;
        let rhs32 = rhs.to_dtype(rhs_l, DType::F32)?;
        let out32 = lhs32.matmul(
            &rhs32,
            bmnk,
            &Layout::contiguous(lhs_l.shape()),
            &Layout::contiguous(rhs_l.shape()),
        )?;
        out32.to_dtype(&Layout::contiguous((b, m, n)), self.dtype)
    }

    /// Copies the elements described by `layout` into a new contiguous storage.
    pub fn contiguous_copy(&self, layout: &Layout) -> Result<Self> {
        let n = layout.shape().elem_count();
        let esz = self.esz();
        let out = self.device.alloc(n * esz)?;
        let mut dst = Self::new(out, self.dtype, self.device.clone());
        self.copy_strided_src(&mut dst, 0, layout)?;
        Ok(dst)
    }

    fn launch_unary_like(
        &self,
        layout: &Layout,
        module: Module,
        name: &str,
        out_dtype: DType,
        extra: impl FnOnce(&mut LaunchArgs),
    ) -> Result<Option<Self>> {
        let info = match info1(layout) {
            Some(i) => i,
            None => return Ok(None),
        };
        let n = layout.shape().elem_count();
        let func = match self.device.get_func(module, name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let out = self.device.alloc(n * elem_bytes(out_dtype))?;
        if n > 0 {
            let mut args = LaunchArgs::new();
            args.usize(n)
                .words(info.to_words())
                .ptr(self.ptr_at(layout.start_offset()))
                .ptr(out.as_ptr());
            extra(&mut args);
            let block = 256u32;
            self.device
                .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
        }
        Ok(Some(Self::new(out, out_dtype, self.device.clone())))
    }

    fn binary_like(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        name: &str,
        out_dtype: DType,
    ) -> Result<Option<Self>> {
        let info = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            StridedInfo::<2>::contiguous()
        } else {
            match StridedInfo::<2>::new(lhs_l.dims(), [lhs_l.stride(), rhs_l.stride()]) {
                Some(i) => i,
                None => return Ok(None),
            }
        };
        let func = match self.device.get_func(Module::Binary, name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let n = lhs_l.shape().elem_count();
        let out = self.device.alloc(n * elem_bytes(out_dtype))?;
        if n > 0 {
            let mut args = LaunchArgs::new();
            args.usize(n)
                .words(info.to_words())
                .ptr(self.ptr_at(lhs_l.start_offset()))
                .ptr(rhs.ptr_at(rhs_l.start_offset()))
                .ptr(out.as_ptr());
            let block = 256u32;
            self.device
                .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
        }
        Ok(Some(Self::new(out, out_dtype, self.device.clone())))
    }

    fn reduce_kernel(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Option<Self>> {
        let sfx = match dtype_suffix(self.dtype) {
            Some(s) => s,
            None => return Ok(None),
        };
        let dims = layout.dims();
        let strides = layout.stride();
        let mut nr_dims = Vec::new();
        let mut nr_strides = Vec::new();
        let mut r_dims = Vec::new();
        let mut r_strides = Vec::new();
        for (i, (&d, &s)) in dims.iter().zip(strides.iter()).enumerate() {
            if reduce_dims.contains(&i) {
                r_dims.push(d);
                r_strides.push(s);
            } else {
                nr_dims.push(d);
                nr_strides.push(s);
            }
        }
        let out_numel: usize = nr_dims.iter().product();
        let reduce_size: usize = r_dims.iter().product();
        let trailing = reduce_dims.iter().all(|&d| d + reduce_dims.len() >= dims.len());
        let info = if layout.is_contiguous() && trailing {
            StridedInfo::<1>::contiguous()
        } else {
            let perm_dims: Vec<usize> = nr_dims.iter().chain(r_dims.iter()).copied().collect();
            let perm_strides: Vec<usize> = nr_strides.iter().chain(r_strides.iter()).copied().collect();
            match StridedInfo::<1>::new(&perm_dims, [&perm_strides]) {
                Some(i) => i,
                None => return Ok(None),
            }
        };
        let (base, out_dtype) = match op {
            ReduceOp::Sum => ("rsum", self.dtype),
            ReduceOp::Max => ("rmax", self.dtype),
            ReduceOp::Min => ("rmin", self.dtype),
            ReduceOp::ArgMax => ("rargmax", DType::U32),
            ReduceOp::ArgMin => ("rargmin", DType::U32),
        };
        let out = self.device.alloc(out_numel * elem_bytes(out_dtype))?;
        if out_numel == 0 {
            return Ok(Some(Self::new(out, out_dtype, self.device.clone())));
        }
        if reduce_size == 0 {
            crate::bail!("rocm reduce: cannot reduce over an empty dimension")
        }
        let small = reduce_size <= 16;
        let name = if small {
            format!("{base}_small_{sfx}")
        } else {
            format!("{base}_{sfx}")
        };
        let func = match self.device.get_func(Module::Reduce, &name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let mut args = LaunchArgs::new();
        args.ptr(self.ptr_at(layout.start_offset()))
            .ptr(out.as_ptr())
            .words(info.to_words())
            .usize(out_numel)
            .usize(reduce_size);
        if small {
            let block = 256u32;
            self.device
                .launch(func, (grid_1d(out_numel, block), 1, 1), (block, 1, 1), 0, &mut args)?;
        } else {
            let block = reduce_size.next_power_of_two().clamp(32, 1024) as u32;
            let grid = out_numel.clamp(1, 1 << 20) as u32;
            let shmem = block * (8 + 4);
            self.device
                .launch(func, (grid, 1, 1), (block, 1, 1), shmem, &mut args)?;
        }
        Ok(Some(Self::new(out, out_dtype, self.device.clone())))
    }

    fn row_kernel_cfg(ncols: usize) -> u32 {
        ncols.next_power_of_two().clamp(32, 1024) as u32
    }

    fn require_float(&self, op: &'static str) -> Result<&'static str> {
        if !is_float(self.dtype) {
            return Err(RocmError::UnsupportedDtype {
                dtype: self.dtype,
                op,
            }
            .into());
        }
        Ok(dtype_suffix(self.dtype).unwrap())
    }

    /// `softmax` over the last dimension. Used by `candle_nn::ops::softmax_last_dim`.
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        let sfx = self.require_float("softmax")?;
        if !layout.is_contiguous() {
            let c = self.contiguous_copy(layout)?;
            return c.softmax_last_dim(&Layout::contiguous(layout.shape()));
        }
        let el = layout.shape().elem_count();
        let ncols = *layout.dims().last().unwrap_or(&1);
        let out = self.device.alloc(el * self.esz())?;
        if el > 0 && ncols > 0 {
            let nrows = el / ncols;
            let func = self.device.get_func(Module::Nn, &format!("softmax_{sfx}"))?;
            let block = Self::row_kernel_cfg(ncols);
            let mut args = LaunchArgs::new();
            args.ptr(self.ptr_at(layout.start_offset()))
                .ptr(out.as_ptr())
                .usize(nrows)
                .usize(ncols);
            self.device.launch(
                func,
                (nrows.clamp(1, 1 << 20) as u32, 1, 1),
                (block, 1, 1),
                block * 8,
                &mut args,
            )?;
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    /// RMS normalization over the last dimension. Used by `candle_nn::ops::rms_norm`.
    pub fn rms_norm(&self, layout: &Layout, alpha: &Self, alpha_l: &Layout, eps: f32) -> Result<Self> {
        let sfx = self.require_float("rms-norm")?;
        if alpha.dtype != self.dtype {
            crate::bail!("rms-norm dtype mismatch {:?} {:?}", self.dtype, alpha.dtype)
        }
        if !layout.is_contiguous() {
            let c = self.contiguous_copy(layout)?;
            return c.rms_norm(&Layout::contiguous(layout.shape()), alpha, alpha_l, eps);
        }
        if !alpha_l.is_contiguous() {
            let a = alpha.contiguous_copy(alpha_l)?;
            return self.rms_norm(layout, &a, &Layout::contiguous(alpha_l.shape()), eps);
        }
        let el = layout.shape().elem_count();
        let ncols = *layout.dims().last().unwrap_or(&1);
        let out = self.device.alloc(el * self.esz())?;
        if el > 0 && ncols > 0 {
            let nrows = el / ncols;
            let func = self.device.get_func(Module::Nn, &format!("rmsnorm_{sfx}"))?;
            let block = Self::row_kernel_cfg(ncols);
            let mut args = LaunchArgs::new();
            args.ptr(self.ptr_at(layout.start_offset()))
                .ptr(out.as_ptr())
                .ptr(alpha.ptr_at(alpha_l.start_offset()))
                .usize(nrows)
                .usize(ncols)
                .f32(eps);
            self.device.launch(
                func,
                (nrows.clamp(1, 1 << 20) as u32, 1, 1),
                (block, 1, 1),
                block * 8,
                &mut args,
            )?;
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    /// Layer normalization over the last dimension. Used by `candle_nn::ops::layer_norm`.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_l: &Layout,
        beta: &Self,
        beta_l: &Layout,
        eps: f32,
    ) -> Result<Self> {
        let sfx = self.require_float("layer-norm")?;
        if alpha.dtype != self.dtype || beta.dtype != self.dtype {
            crate::bail!("layer-norm dtype mismatch")
        }
        if !layout.is_contiguous() {
            let c = self.contiguous_copy(layout)?;
            return c.layer_norm(&Layout::contiguous(layout.shape()), alpha, alpha_l, beta, beta_l, eps);
        }
        if !alpha_l.is_contiguous() {
            let a = alpha.contiguous_copy(alpha_l)?;
            return self.layer_norm(layout, &a, &Layout::contiguous(alpha_l.shape()), beta, beta_l, eps);
        }
        if !beta_l.is_contiguous() {
            let b = beta.contiguous_copy(beta_l)?;
            return self.layer_norm(layout, alpha, alpha_l, &b, &Layout::contiguous(beta_l.shape()), eps);
        }
        let el = layout.shape().elem_count();
        let ncols = *layout.dims().last().unwrap_or(&1);
        let out = self.device.alloc(el * self.esz())?;
        if el > 0 && ncols > 0 {
            let nrows = el / ncols;
            let func = self.device.get_func(Module::Nn, &format!("layernorm_{sfx}"))?;
            let block = Self::row_kernel_cfg(ncols);
            let mut args = LaunchArgs::new();
            args.ptr(self.ptr_at(layout.start_offset()))
                .ptr(out.as_ptr())
                .ptr(alpha.ptr_at(alpha_l.start_offset()))
                .ptr(beta.ptr_at(beta_l.start_offset()))
                .usize(nrows)
                .usize(ncols)
                .f32(eps);
            self.device.launch(
                func,
                (nrows.clamp(1, 1 << 20) as u32, 1, 1),
                (block, 1, 1),
                block * 8,
                &mut args,
            )?;
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    /// Rotary embeddings. `layout` must be 4D, `cos`/`sin` 2D `(t, d/2)` or 3D `(b, t, d/2)`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        layout: &Layout,
        cos: &Self,
        cos_l: &Layout,
        sin: &Self,
        sin_l: &Layout,
        kind: RopeKind,
    ) -> Result<Self> {
        let sfx = self.require_float("rope")?;
        if cos.dtype != self.dtype || sin.dtype != self.dtype {
            crate::bail!("rope dtype mismatch {:?} {:?} {:?}", self.dtype, cos.dtype, sin.dtype)
        }
        if !layout.is_contiguous() {
            let c = self.contiguous_copy(layout)?;
            return c.rope(&Layout::contiguous(layout.shape()), cos, cos_l, sin, sin_l, kind);
        }
        if !cos_l.is_contiguous() {
            let c = cos.contiguous_copy(cos_l)?;
            return self.rope(layout, &c, &Layout::contiguous(cos_l.shape()), sin, sin_l, kind);
        }
        if !sin_l.is_contiguous() {
            let s = sin.contiguous_copy(sin_l)?;
            return self.rope(layout, cos, cos_l, &s, &Layout::contiguous(sin_l.shape()), kind);
        }
        let (d0, d1, d2, d3) = layout.shape().dims4()?;
        let unbatched = (cos_l.dims().len() == 3 && sin_l.dims().len() == 3) as u32;
        let el = layout.shape().elem_count();
        let out = self.device.alloc(el * self.esz())?;
        let pairs = el / 2;
        if pairs > 0 {
            let name = match kind {
                RopeKind::Interleaved => format!("rope_i_{sfx}"),
                RopeKind::Halves => format!("rope_{sfx}"),
                RopeKind::Thd => format!("rope_thd_{sfx}"),
            };
            let func = self.device.get_func(Module::Nn, &name)?;
            let mut args = LaunchArgs::new();
            args.ptr(self.ptr_at(layout.start_offset()))
                .ptr(cos.ptr_at(cos_l.start_offset()))
                .ptr(sin.ptr_at(sin_l.start_offset()))
                .ptr(out.as_ptr())
                .usize(d0)
                .usize(d1)
                .usize(d2)
                .usize(d3)
                .u32(unbatched);
            let block = 256u32;
            self.device
                .launch(func, (grid_1d(pairs, block), 1, 1), (block, 1, 1), 0, &mut args)?;
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    /// Element-wise sigmoid. Used by `candle_nn::ops::sigmoid`.
    pub fn sigmoid(&self, layout: &Layout) -> Result<Self> {
        let sfx = self.require_float("sigmoid")?;
        match self.launch_unary_like(layout, Module::Unary, &format!("usigmoid_{sfx}"), self.dtype, |_| {})? {
            Some(s) => Ok(s),
            None => {
                let c = self.contiguous_copy(layout)?;
                c.sigmoid(&Layout::contiguous(layout.shape()))
            }
        }
    }

    fn ids_contiguous(ids: &Self, ids_l: &Layout) -> Result<Option<Self>> {
        if ids_l.is_contiguous() {
            Ok(None)
        } else {
            Ok(Some(ids.contiguous_copy(ids_l)?))
        }
    }
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        let bytes = self.buf.byte_size();
        let out = self.device.alloc(bytes)?;
        unsafe {
            hip_runtime::memory::memcpy_dtod(out.as_void_ptr(), self.buf.as_void_ptr(), bytes).w()?;
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.to_cpu()
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if let Some(sfx) = dtype_suffix(self.dtype).filter(|_| is_float(self.dtype)) {
            let name = format!("affine_{sfx}");
            if let Some(s) = self.launch_unary_like(layout, Module::Unary, &name, self.dtype, |a| {
                a.f64(mul).f64(add);
            })? {
                return Ok(s);
            }
        }
        self.fallback1(|c| c.affine(layout, mul, add))
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if let Some(sfx) = dtype_suffix(self.dtype).filter(|_| is_float(self.dtype)) {
            let name = format!("upowf_{sfx}");
            if let Some(s) = self.launch_unary_like(layout, Module::Unary, &name, self.dtype, |a| {
                a.f64(e);
            })? {
                return Ok(s);
            }
        }
        self.fallback1(|c| c.powf(layout, e))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if let Some(sfx) = dtype_suffix(self.dtype).filter(|_| is_float(self.dtype)) {
            let name = format!("uelu_{sfx}");
            if let Some(s) = self.launch_unary_like(layout, Module::Unary, &name, self.dtype, |a| {
                a.f64(alpha);
            })? {
                return Ok(s);
            }
        }
        self.fallback1(|c| c.elu(layout, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if let Some(s) = self.reduce_kernel(op, layout, reduce_dims)? {
            return Ok(s);
        }
        self.fallback1(|c| c.reduce_op(op, layout, reduce_dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if let Some(sfx) = dtype_suffix(self.dtype) {
            let base = match op {
                CmpOp::Eq => "eq",
                CmpOp::Ne => "ne",
                CmpOp::Lt => "lt",
                CmpOp::Le => "le",
                CmpOp::Gt => "gt",
                CmpOp::Ge => "ge",
            };
            if let Some(s) = self.binary_like(rhs, lhs_l, rhs_l, &format!("{base}_{sfx}"), DType::U8)? {
                return Ok(s);
            }
        }
        self.fallback2(rhs, |a, b| a.cmp(op, b, lhs_l, rhs_l))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let is_dummy = |d: DType| matches!(d, DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0);
        if is_dummy(dtype) {
            return Err(crate::Error::UnsupportedDTypeForOp(dtype, "to_dtype").bt());
        }
        if is_dummy(self.dtype) {
            return Err(crate::Error::UnsupportedDTypeForOp(self.dtype, "to_dtype").bt());
        }
        if let (Some(src), Some(dst)) = (cast_suffix(self.dtype), cast_suffix(dtype)) {
            let name = format!("cast_{src}_{dst}");
            if let Some(s) = self.launch_unary_like(layout, Module::Cast, &name, dtype, |_| {})? {
                return Ok(s);
            }
        }
        self.fallback1(|c| c.to_dtype(layout, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if is_float(self.dtype) {
            let sfx = dtype_suffix(self.dtype).unwrap();
            let name = format!("{}_{sfx}", B::KERNEL);
            if let Some(s) = self.launch_unary_like(layout, Module::Unary, &name, self.dtype, |_| {})? {
                return Ok(s);
            }
        }
        self.fallback1(|c| c.unary_impl::<B>(layout))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if let Some(sfx) = dtype_suffix(self.dtype) {
            let name = format!("{}_{sfx}", B::KERNEL);
            if let Some(s) = self.binary_like(rhs, lhs_l, rhs_l, &name, self.dtype)? {
                return Ok(s);
            }
        }
        self.fallback2(rhs, |a, b| a.binary_impl::<B>(b, lhs_l, rhs_l))
    }

    fn where_cond(&self, layout: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        let csfx = index_suffix(self.dtype);
        let tsfx = size_suffix(t.esz());
        if let (Some(csfx), Some(tsfx)) = (csfx, tsfx) {
            let info = if layout.is_contiguous() && t_l.is_contiguous() && f_l.is_contiguous() {
                Some(StridedInfo::<3>::contiguous())
            } else {
                StridedInfo::<3>::new(layout.dims(), [layout.stride(), t_l.stride(), f_l.stride()])
            };
            let name = format!("where_{csfx}_{tsfx}");
            if let (Some(info), Ok(func)) = (info, self.device.get_func(Module::Ternary, &name)) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(n * t.esz())?;
                if n > 0 {
                    let mut args = LaunchArgs::new();
                    args.usize(n)
                        .words(info.to_words())
                        .ptr(self.ptr_at(layout.start_offset()))
                        .ptr(t.ptr_at(t_l.start_offset()))
                        .ptr(f.ptr_at(f_l.start_offset()))
                        .ptr(out.as_ptr());
                    let block = 256u32;
                    self.device
                        .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
                }
                return Ok(Self::new(out, t.dtype, self.device.clone()));
            }
        }
        let cond = self.to_cpu()?;
        let tc = t.to_cpu()?;
        let fc = f.to_cpu()?;
        let out = cond.where_cond(layout, &tc, t_l, &fc, f_l)?;
        self.upload_cpu(&out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.fallback2_wide(l, kernel, kernel_l, |a, la, b, lb| a.conv1d(la, b, lb, params))
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.fallback2_wide(l, kernel, kernel_l, |a, la, b, lb| a.conv_transpose1d(la, b, lb, params))
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.fallback2_wide(l, kernel, kernel_l, |a, la, b, lb| a.conv2d(la, b, lb, params))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.fallback2_wide(l, kernel, kernel_l, |a, la, b, lb| a.conv_transpose2d(la, b, lb, params))
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        self.fallback1(|c| c.avg_pool2d(l, k, s))
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        self.fallback1(|c| c.max_pool2d(l, k, s))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        self.fallback1(|c| c.upsample_nearest1d(l, sz))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        self.fallback1(|c| c.upsample_nearest2d(l, h, w))
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        self.fallback1(|c| c.upsample_bilinear2d(l, h, w, align_corners, scale_h, scale_w))
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let isfx = index_suffix(ids.dtype);
        let tsfx = size_suffix(self.esz());
        if let (Some(isfx), Some(tsfx)) = (isfx, tsfx) {
            if !l.is_contiguous() {
                let src = self.contiguous_copy(l)?;
                return src.gather(&Layout::contiguous(l.shape()), ids, ids_l, dim);
            }
            if let Some(ids_c) = Self::ids_contiguous(ids, ids_l)? {
                return self.gather(l, &ids_c, &Layout::contiguous(ids_l.shape()), dim);
            }
            let name = format!("gather_{isfx}_{tsfx}");
            let func = self.device.get_func(Module::Indexing, &name)?;
            let ids_dims = ids_l.dims();
            let left_size: usize = ids_dims[..dim].iter().product();
            let right_size: usize = ids_dims[dim + 1..].iter().product();
            let src_dim_size = l.dims()[dim];
            let ids_dim_size = ids_dims[dim];
            let n = ids_l.shape().elem_count();
            let out = self.device.alloc(n * self.esz())?;
            if n > 0 {
                let mut args = LaunchArgs::new();
                args.usize(n)
                    .ptr(ids.ptr_at(ids_l.start_offset()))
                    .ptr(self.ptr_at(l.start_offset()))
                    .ptr(out.as_ptr())
                    .usize(left_size)
                    .usize(src_dim_size)
                    .usize(ids_dim_size)
                    .usize(right_size);
                let block = 256u32;
                self.device
                    .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
            }
            return Ok(Self::new(out, self.dtype, self.device.clone()));
        }
        self.fallback2(ids, |s, i| s.gather(l, i, ids_l, dim))
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let isfx = index_suffix(ids.dtype);
        let tsfx = size_suffix(self.esz());
        if let (Some(isfx), Some(tsfx), true) = (isfx, tsfx, l.is_contiguous()) {
            if let Some(ids_c) = Self::ids_contiguous(ids, ids_l)? {
                return self.scatter_set(l, &ids_c, &Layout::contiguous(ids_l.shape()), src, src_l, dim);
            }
            if !src_l.is_contiguous() {
                let s = src.contiguous_copy(src_l)?;
                return self.scatter_set(l, ids, ids_l, &s, &Layout::contiguous(src_l.shape()), dim);
            }
            let func = self.device.get_func(Module::Indexing, &format!("s_{isfx}_{tsfx}"))?;
            let ids_dims = ids_l.dims();
            let left_size: usize = ids_dims[..dim].iter().product();
            let right_size: usize = ids_dims[dim + 1..].iter().product();
            let n = left_size * right_size;
            if n > 0 {
                let mut args = LaunchArgs::new();
                args.ptr(ids.ptr_at(ids_l.start_offset()))
                    .ptr(src.ptr_at(src_l.start_offset()))
                    .ptr(self.ptr_at_mut(l.start_offset()))
                    .usize(left_size)
                    .usize(ids_dims[dim])
                    .usize(l.dims()[dim])
                    .usize(right_size);
                let block = 256u32;
                self.device
                    .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
            }
            return Ok(());
        }
        let mut cpu = self.to_cpu()?;
        let ids_c = ids.to_cpu()?;
        let src_c = src.to_cpu()?;
        cpu.scatter_set(l, &ids_c, ids_l, &src_c, src_l, dim)?;
        self.overwrite_from_cpu(&cpu)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let isfx = index_suffix(ids.dtype);
        let tsfx = dtype_suffix(self.dtype);
        if let (Some(isfx), Some(tsfx), true) = (isfx, tsfx, l.is_contiguous()) {
            if let Some(ids_c) = Self::ids_contiguous(ids, ids_l)? {
                return self.scatter_add_set(l, &ids_c, &Layout::contiguous(ids_l.shape()), src, src_l, dim);
            }
            if !src_l.is_contiguous() {
                let s = src.contiguous_copy(src_l)?;
                return self.scatter_add_set(l, ids, ids_l, &s, &Layout::contiguous(src_l.shape()), dim);
            }
            let func = self.device.get_func(Module::Indexing, &format!("sa_{isfx}_{tsfx}"))?;
            let ids_dims = ids_l.dims();
            let left_size: usize = ids_dims[..dim].iter().product();
            let right_size: usize = ids_dims[dim + 1..].iter().product();
            let n = left_size * right_size;
            if n > 0 {
                let mut args = LaunchArgs::new();
                args.ptr(ids.ptr_at(ids_l.start_offset()))
                    .ptr(src.ptr_at(src_l.start_offset()))
                    .ptr(self.ptr_at_mut(l.start_offset()))
                    .usize(left_size)
                    .usize(ids_dims[dim])
                    .usize(l.dims()[dim])
                    .usize(right_size);
                let block = 256u32;
                self.device
                    .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
            }
            return Ok(());
        }
        let mut cpu = self.to_cpu()?;
        let ids_c = ids.to_cpu()?;
        let src_c = src.to_cpu()?;
        cpu.scatter_add_set(l, &ids_c, ids_l, &src_c, src_l, dim)?;
        self.overwrite_from_cpu(&cpu)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let isfx = index_suffix(ids.dtype);
        let tsfx = size_suffix(self.esz());
        if let (Some(isfx), Some(tsfx), 1) = (isfx, tsfx, ids_l.dims().len()) {
            if let Some(ids_c) = Self::ids_contiguous(ids, ids_l)? {
                return self.index_select(&ids_c, l, &Layout::contiguous(ids_l.shape()), dim);
            }
            let info = match info1(l) {
                Some(i) => i,
                None => {
                    let src = self.contiguous_copy(l)?;
                    return src.index_select(ids, &Layout::contiguous(l.shape()), ids_l, dim);
                }
            };
            let func = self.device.get_func(Module::Indexing, &format!("is_{isfx}_{tsfx}"))?;
            let dims = l.dims();
            let left_size: usize = dims[..dim].iter().product();
            let right_size: usize = dims[dim + 1..].iter().product();
            let src_dim_size = dims[dim];
            let ids_dim_size = ids_l.dims()[0];
            let n = left_size * ids_dim_size * right_size;
            let out = self.device.alloc(n * self.esz())?;
            if n > 0 {
                let mut args = LaunchArgs::new();
                args.usize(n)
                    .words(info.to_words())
                    .ptr(ids.ptr_at(ids_l.start_offset()))
                    .ptr(self.ptr_at(l.start_offset()))
                    .ptr(out.as_ptr())
                    .usize(left_size)
                    .usize(src_dim_size)
                    .usize(ids_dim_size)
                    .usize(right_size);
                let block = 256u32;
                self.device
                    .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
            }
            return Ok(Self::new(out, self.dtype, self.device.clone()));
        }
        self.fallback2(ids, |s, i| s.index_select(i, l, ids_l, dim))
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let isfx = index_suffix(ids.dtype);
        let tsfx = dtype_suffix(self.dtype);
        if let (Some(isfx), Some(tsfx), 1) = (isfx, tsfx, ids_l.dims().len()) {
            if let Some(ids_c) = Self::ids_contiguous(ids, ids_l)? {
                return self.index_add(l, &ids_c, &Layout::contiguous(ids_l.shape()), src, src_l, dim);
            }
            if !src_l.is_contiguous() {
                let s = src.contiguous_copy(src_l)?;
                return self.index_add(l, ids, ids_l, &s, &Layout::contiguous(src_l.shape()), dim);
            }
            let func = self.device.get_func(Module::Indexing, &format!("ia_{isfx}_{tsfx}"))?;
            let out = self.contiguous_copy(l)?;
            let src_dims = src_l.dims();
            let left_size: usize = src_dims[..dim].iter().product();
            let right_size: usize = src_dims[dim + 1..].iter().product();
            let n = left_size * right_size;
            if n > 0 {
                let mut args = LaunchArgs::new();
                args.ptr(ids.ptr_at(ids_l.start_offset()))
                    .usize(ids_l.dims()[0])
                    .ptr(src.ptr_at(src_l.start_offset()))
                    .ptr(out.buf.as_ptr())
                    .usize(left_size)
                    .usize(src_dims[dim])
                    .usize(l.dims()[dim])
                    .usize(right_size);
                let block = 256u32;
                self.device
                    .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args)?;
            }
            return Ok(out);
        }
        self.fallback3(ids, src, |s, i, x| s.index_add(l, i, ids_l, x, src_l, dim))
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let ty = match self.dtype {
            DType::F32 => GemmType::F32,
            DType::F64 => GemmType::F64,
            DType::F16 => GemmType::F16,
            DType::BF16 => GemmType::BF16,
            _ => {
                return self.fallback2(rhs, |a, c| a.matmul(c, (b, m, n, k), lhs_l, rhs_l));
            }
        };
        let esz = self.esz();
        let out_elems = b * m * n;
        if out_elems == 0 {
            let out = self.device.alloc(0)?;
            return Ok(Self::new(out, self.dtype, self.device.clone()));
        }
        if k == 0 {
            let out = self.device.alloc_zeros(out_elems * esz)?;
            return Ok(Self::new(out, self.dtype, self.device.clone()));
        }
        let cfg = match gemm_config((b, m, n, k), lhs_l, rhs_l) {
            Some(cfg) => cfg,
            None => {
                let lhs_c;
                let rhs_c;
                let (lhs, lhs_l2) = if gemm_config((b, m, n, k), lhs_l, &Layout::contiguous((b, k, n))).is_some() {
                    (self, lhs_l.clone())
                } else {
                    lhs_c = self.contiguous_copy(lhs_l)?;
                    (&lhs_c, Layout::contiguous(lhs_l.shape()))
                };
                let (rhs, rhs_l2) = if gemm_config((b, m, n, k), &Layout::contiguous((b, m, k)), rhs_l).is_some() {
                    (rhs, rhs_l.clone())
                } else {
                    rhs_c = rhs.contiguous_copy(rhs_l)?;
                    (&rhs_c, Layout::contiguous(rhs_l.shape()))
                };
                return match gemm_config((b, m, n, k), &lhs_l2, &rhs_l2) {
                    Some(_) => lhs.matmul(rhs, (b, m, n, k), &lhs_l2, &rhs_l2),
                    None => Err(crate::Error::MatMulUnexpectedStriding(Box::new(
                        crate::error::MatMulUnexpectedStriding {
                            lhs_l: lhs_l.clone(),
                            rhs_l: rhs_l.clone(),
                            bmnk: (b, m, n, k),
                            msg: "rocm matmul: unsupported strides",
                        },
                    ))
                    .bt()),
                };
            }
        };
        let out = self.device.alloc(out_elems * esz)?;
        let a = rhs.ptr_at(rhs_l.start_offset()) as *const c_void;
        let bp = self.ptr_at(lhs_l.start_offset()) as *const c_void;
        let c = out.as_void_ptr();
        let stride_c = (m * n) as i64;
        // Row-major C = lhs * rhs is computed as the column-major C^T = rhs^T * lhs^T.
        let status = self.device.gemm(&GemmCall {
            ty,
            transa: cfg.transa,
            transb: cfg.transb,
            m: n,
            n: m,
            k,
            a,
            lda: cfg.lda,
            stride_a: cfg.stride_a,
            b: bp,
            ldb: cfg.ldb,
            stride_b: cfg.stride_b,
            c,
            ldc: n,
            stride_c,
            batch: b,
            batched: true,
        })?;
        match status {
            GemmOutcome::Done => {}
            GemmOutcome::Rocblas(hip_runtime::error::HipError::RocblasError { code: 2 | 14 | 15 })
                if matches!(ty, GemmType::F16 | GemmType::BF16) =>
            {
                drop(out);
                return self.half_matmul_via_f32(rhs, (b, m, n, k), lhs_l, rhs_l);
            }
            GemmOutcome::Rocblas(e) => return Err::<Self, _>(e).w(),
            GemmOutcome::NoKernel(_) => {
                // No GEMM kernel for this GPU (and no rocBLAS kernels either): run on the CPU,
                // like the other ops whose kernels cannot be loaded.
                drop(out);
                return self.fallback2_wide(lhs_l, rhs, rhs_l, |a, la, c, lc| a.matmul(c, (b, m, n, k), la, lc));
            }
        }
        Ok(Self::new(out, self.dtype, self.device.clone()))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if dst.dtype != self.dtype {
            crate::bail!("rocm copy_strided_src dtype mismatch {:?} {:?}", self.dtype, dst.dtype)
        }
        let n = src_l.shape().elem_count();
        if n == 0 {
            return Ok(());
        }
        let esz = self.esz();
        if src_l.is_contiguous() {
            unsafe {
                hip_runtime::memory::memcpy_dtod(
                    dst.ptr_at_mut(dst_offset) as *mut c_void,
                    self.ptr_at(src_l.start_offset()) as *const c_void,
                    n * esz,
                )
                .w()?;
            }
            return Ok(());
        }
        let tsfx = size_suffix(esz);
        let info = info1(src_l);
        if let (Some(tsfx), Some(info)) = (tsfx, info) {
            let func = self.device.get_func(Module::Fill, &format!("copy_strided_{tsfx}"))?;
            let mut args = LaunchArgs::new();
            args.usize(n)
                .words(info.to_words())
                .ptr(self.ptr_at(src_l.start_offset()))
                .ptr(dst.ptr_at_mut(dst_offset));
            let block = 256u32;
            return self
                .device
                .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args);
        }
        let src_cpu = self.to_cpu()?;
        let mut dst_cpu = dst.to_cpu()?;
        src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        dst.overwrite_from_cpu(&dst_cpu)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if dst.dtype != self.dtype {
            crate::bail!("rocm copy2d dtype mismatch {:?} {:?}", self.dtype, dst.dtype)
        }
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let esz = self.esz();
        if (src_stride1 == d2 && dst_stride1 == d2) || d1 == 1 {
            unsafe {
                hip_runtime::memory::memcpy_dtod(
                    dst.ptr_at_mut(dst_offset) as *mut c_void,
                    self.ptr_at(src_offset) as *const c_void,
                    d1 * d2 * esz,
                )
                .w()?;
            }
            return Ok(());
        }
        let tsfx = match size_suffix(esz) {
            Some(s) => s,
            None => crate::bail!("rocm copy2d: unsupported element size {esz}"),
        };
        let func = self.device.get_func(Module::Fill, &format!("copy2d_{tsfx}"))?;
        let mut args = LaunchArgs::new();
        args.ptr(self.ptr_at(src_offset))
            .ptr(dst.ptr_at_mut(dst_offset))
            .usize(d1)
            .usize(d2)
            .usize(src_stride1)
            .usize(dst_stride1);
        let block = 256u32;
        self.device
            .launch(func, (grid_1d(d1 * d2, block), 1, 1), (block, 1, 1), 0, &mut args)
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let n = l.shape().elem_count();
        if n == 0 {
            return Ok(());
        }
        let tsfx = size_suffix(self.esz());
        let info = info1(l);
        if let (Some(tsfx), Some(info), true) = (tsfx, info, s.dtype() == self.dtype) {
            let func = self.device.get_func(Module::Fill, &format!("const_set_{tsfx}"))?;
            let mut args = LaunchArgs::new();
            args.usize(n)
                .words(info.to_words())
                .bits(scalar_bits(s))
                .ptr(self.ptr_at_mut(l.start_offset()));
            let block = 256u32;
            return self
                .device
                .launch(func, (grid_1d(n, block), 1, 1), (block, 1, 1), 0, &mut args);
        }
        let mut cpu = self.to_cpu()?;
        cpu.const_set(s, l)?;
        self.overwrite_from_cpu(&cpu)
    }
}

impl RocmStorage {
    /// Runs a cpu implementation of a custom op and moves the result back to this device.
    pub fn cpu_fallback_op(&self, out: CpuStorage) -> Result<Self> {
        self.upload_cpu(&out)
    }

    /// Default implementation used by `CustomOp1::rocm_fwd`.
    pub fn apply_cpu_op1(
        &self,
        layout: &Layout,
        f: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
    ) -> Result<(Self, Shape)> {
        let cpu = self.to_cpu()?;
        let (out, shape) = f(&cpu, layout)?;
        Ok((self.upload_cpu(&out)?, shape))
    }
}

