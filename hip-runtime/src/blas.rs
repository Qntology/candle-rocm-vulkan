//! Safe GEMM wrapper around rocBLAS.

use crate::error::{check_rocblas, Result};
use crate::memory::DeviceBuffer;
use hip_sys::rocblas;
use std::ffi::c_void;

pub struct RocBlas {
    handle: rocblas::rocblas_handle,
}

// A rocBLAS handle may be moved between threads; concurrent use must be serialized by the caller.
unsafe impl Send for RocBlas {}

/// Element type used by [`RocBlas::gemm_strided_batched_ex_raw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmType {
    F16,
    BF16,
    F32,
    F64,
}

impl GemmType {
    fn rocblas(self) -> rocblas::rocblas_datatype {
        match self {
            Self::F16 => rocblas::rocblas_datatype_f16_r,
            Self::BF16 => rocblas::rocblas_datatype_bf16_r,
            Self::F32 => rocblas::rocblas_datatype_f32_r,
            Self::F64 => rocblas::rocblas_datatype_f64_r,
        }
    }
}

fn op(t: bool) -> rocblas::rocblas_operation {
    if t {
        rocblas::rocblas_operation::rocblas_operation_transpose
    } else {
        rocblas::rocblas_operation::rocblas_operation_none
    }
}

impl RocBlas {
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        check_rocblas(unsafe { rocblas::rocblas_create_handle(&mut handle) })?;
        Ok(Self { handle })
    }

    /// C = alpha * op(A) * op(B) + beta * C, column-major.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: usize,
        b: &DeviceBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: usize,
    ) -> Result<()> {
        unsafe {
            self.sgemm_raw(
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a.as_void_ptr(),
                lda,
                b.as_void_ptr(),
                ldb,
                beta,
                c.as_void_ptr(),
                ldc,
            )
        }
    }

    /// Batched strided SGEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_strided_batched(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: usize,
        stride_a: i64,
        b: &DeviceBuffer<f32>,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        unsafe {
            self.sgemm_strided_batched_raw(
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a.as_void_ptr(),
                lda,
                stride_a,
                b.as_void_ptr(),
                ldb,
                stride_b,
                beta,
                c.as_void_ptr(),
                ldc,
                stride_c,
                batch_count,
            )
        }
    }

    /// # Safety
    /// Pointers must reference valid f32 device memory with the given dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const c_void,
        lda: usize,
        b: *const c_void,
        ldb: usize,
        beta: f32,
        c: *mut c_void,
        ldc: usize,
    ) -> Result<()> {
        check_rocblas(rocblas::rocblas_sgemm(
            self.handle,
            op(trans_a),
            op(trans_b),
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            &beta,
            c,
            ldc as i32,
        ))
    }

    /// # Safety
    /// Same requirements as `sgemm_raw`, plus stride/batch constraints.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm_strided_batched_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const c_void,
        lda: usize,
        stride_a: i64,
        b: *const c_void,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: *mut c_void,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        check_rocblas(rocblas::rocblas_sgemm_strided_batched(
            self.handle,
            op(trans_a),
            op(trans_b),
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            a,
            lda as i32,
            stride_a,
            b,
            ldb as i32,
            stride_b,
            &beta,
            c,
            ldc as i32,
            stride_c,
            batch_count as i32,
        ))
    }

    /// # Safety
    /// Pointers must reference valid f64 device memory with the given dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dgemm_strided_batched_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const c_void,
        lda: usize,
        stride_a: i64,
        b: *const c_void,
        ldb: usize,
        stride_b: i64,
        beta: f64,
        c: *mut c_void,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        check_rocblas(rocblas::rocblas_dgemm_strided_batched(
            self.handle,
            op(trans_a),
            op(trans_b),
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            a,
            lda as i32,
            stride_a,
            b,
            ldb as i32,
            stride_b,
            &beta,
            c,
            ldc as i32,
            stride_c,
            batch_count as i32,
        ))
    }

    /// Mixed precision strided batched GEMM, accumulating in f32 (f64 for `GemmType::F64`).
    ///
    /// # Safety
    /// Pointers must reference valid device memory of type `ty` with the given dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_strided_batched_ex_raw(
        &self,
        ty: GemmType,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: *const c_void,
        lda: usize,
        stride_a: i64,
        b: *const c_void,
        ldb: usize,
        stride_b: i64,
        c: *mut c_void,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let alpha32: f32 = 1.0;
        let beta32: f32 = 0.0;
        let alpha64: f64 = 1.0;
        let beta64: f64 = 0.0;
        let (alpha, beta, compute) = if ty == GemmType::F64 {
            (
                &alpha64 as *const f64 as *const c_void,
                &beta64 as *const f64 as *const c_void,
                rocblas::rocblas_datatype_f64_r,
            )
        } else {
            (
                &alpha32 as *const f32 as *const c_void,
                &beta32 as *const f32 as *const c_void,
                rocblas::rocblas_datatype_f32_r,
            )
        };
        let t = ty.rocblas();
        check_rocblas(rocblas::rocblas_gemm_strided_batched_ex(
            self.handle,
            op(trans_a),
            op(trans_b),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            t,
            lda as i32,
            stride_a,
            b,
            t,
            ldb as i32,
            stride_b,
            beta,
            c,
            t,
            ldc as i32,
            stride_c,
            c,
            t,
            ldc as i32,
            stride_c,
            batch_count as i32,
            compute,
            rocblas::rocblas_gemm_algo_standard,
            0,
            0,
        ))
    }
}

impl Drop for RocBlas {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { rocblas::rocblas_destroy_handle(self.handle) };
        }
    }
}
