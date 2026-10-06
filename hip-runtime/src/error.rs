//! Error types for HIP runtime operations.

use std::fmt;

#[derive(Debug, Clone)]
pub enum HipError {
    HipRuntimeError { code: i32, msg: String },
    RocblasError { code: i32 },
    KernelNotFound { name: String },
    KernelCompileFailed { msg: String },
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HipRuntimeError { code, msg } => write!(f, "HIP error {code}: {msg}"),
            Self::RocblasError { code } => write!(f, "rocBLAS error {code}: {}", rocblas_status_name(*code)),
            Self::KernelNotFound { name } => write!(f, "kernel not found: {name}"),
            Self::KernelCompileFailed { msg } => write!(f, "kernel compile failed: {msg}"),
        }
    }
}

fn rocblas_status_name(code: i32) -> &'static str {
    match code {
        1 => "invalid handle",
        2 => "not implemented",
        3 => "invalid pointer",
        4 => "invalid size",
        5 => "out of memory",
        6 => "internal error",
        7 => "performance degraded",
        8 => "size query mismatch",
        9 => "size increased",
        10 => "size unchanged",
        11 => "invalid value",
        12 => "continue",
        13 => "check numerics fail",
        14 => "excluded from build",
        15 => "arch mismatch",
        _ => "unknown status",
    }
}

impl std::error::Error for HipError {}

pub type Result<T> = std::result::Result<T, HipError>;

pub fn error_string(code: i32) -> String {
    unsafe {
        let ptr = hip_sys::hip_runtime::hipGetErrorString(code);
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Check a HIP status code and convert to Result.
pub fn check_hip(code: i32) -> Result<()> {
    if code == hip_sys::hip_runtime::HIP_SUCCESS {
        Ok(())
    } else {
        Err(HipError::HipRuntimeError {
            code,
            msg: error_string(code),
        })
    }
}

pub fn check_rocblas(code: i32) -> Result<()> {
    if code == hip_sys::rocblas::ROCBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HipError::RocblasError { code })
    }
}
