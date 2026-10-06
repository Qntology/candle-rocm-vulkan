//! rocBLAS FFI bindings.

use std::ffi::c_void;
use std::os::raw::c_int;

pub type rocblas_handle = *mut c_void;
pub type rocblas_status = c_int;
pub type rocblas_datatype = c_int;
pub type rocblas_gemm_algo = c_int;

pub const ROCBLAS_STATUS_SUCCESS: rocblas_status = 0;

pub const rocblas_datatype_f16_r: rocblas_datatype = 150;
pub const rocblas_datatype_f32_r: rocblas_datatype = 151;
pub const rocblas_datatype_f64_r: rocblas_datatype = 152;
pub const rocblas_datatype_bf16_r: rocblas_datatype = 168;

pub const rocblas_gemm_algo_standard: rocblas_gemm_algo = 0;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum rocblas_operation {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}

extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;

    pub fn rocblas_sgemm(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
    ) -> rocblas_status;

    pub fn rocblas_sgemm_strided_batched(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        ldb: c_int,
        stride_b: i64,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> rocblas_status;

    pub fn rocblas_dgemm_strided_batched(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f64,
        a: *const c_void,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        ldb: c_int,
        stride_b: i64,
        beta: *const f64,
        c: *mut c_void,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> rocblas_status;

    pub fn rocblas_gemm_strided_batched_ex(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void,
        c: *const c_void,
        c_type: rocblas_datatype,
        ldc: c_int,
        stride_c: i64,
        d: *mut c_void,
        d_type: rocblas_datatype,
        ldd: c_int,
        stride_d: i64,
        batch_count: c_int,
        compute_type: rocblas_datatype,
        algo: rocblas_gemm_algo,
        solution_index: i32,
        flags: u32,
    ) -> rocblas_status;
}
