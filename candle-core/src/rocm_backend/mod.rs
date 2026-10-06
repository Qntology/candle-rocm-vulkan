//! ROCm / HIP backend for AMD GPUs.
//!
//! Kernels are written in HIP (see `kernels/*.hip`), compiled by `build.rs` with `hipcc`
//! and embedded in the binary. Matrix multiplications go through rocBLAS.
mod device;
pub mod kernels;
mod storage;
pub mod utils;

pub use device::RocmDevice;
pub use hip_runtime;
pub use hip_sys;
pub use storage::{RocmStorage, RopeKind};

#[derive(Debug, Clone)]
pub enum RocmError {
    Hip(hip_runtime::error::HipError),
    Message(String),
    UnsupportedDtype {
        dtype: crate::DType,
        op: &'static str,
    },
}

impl std::fmt::Display for RocmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hip(e) => write!(f, "{e}"),
            Self::Message(m) => write!(f, "{m}"),
            Self::UnsupportedDtype { dtype, op } => {
                write!(f, "{op} is not supported for {dtype:?} on the ROCm backend")
            }
        }
    }
}

impl std::error::Error for RocmError {}

impl From<hip_runtime::error::HipError> for RocmError {
    fn from(e: hip_runtime::error::HipError) -> Self {
        Self::Hip(e)
    }
}

impl From<RocmError> for crate::Error {
    fn from(e: RocmError) -> Self {
        crate::Error::Rocm(Box::new(e)).bt()
    }
}

pub trait WrapErr<O> {
    fn w(self) -> crate::Result<O>;
}

impl<O, E: Into<RocmError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> crate::Result<O> {
        self.map_err(|e| crate::Error::Rocm(Box::new(e.into())).bt())
    }
}

/// Number of ROCm devices visible to the HIP runtime (0 when no AMD GPU or driver is present).
pub fn device_count() -> usize {
    hip_runtime::device::HipDevice::device_count().unwrap_or(0)
}

/// `(free, total)` memory in bytes of the device `ordinal`, without creating a candle device.
pub fn mem_info(ordinal: usize) -> crate::Result<(usize, usize)> {
    let dev = hip_runtime::device::HipDevice::new(ordinal).w()?;
    dev.mem_info().w()
}

/// Marketing name of the device `ordinal`.
pub fn device_name(ordinal: usize) -> crate::Result<String> {
    let dev = hip_runtime::device::HipDevice::new(ordinal).w()?;
    dev.name().w()
}
