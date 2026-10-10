//! ROCm/HIP backend for the candle ML framework.
//!
//! The backend itself lives in the patched `candle-core` (feature `rocm`); this crate only
//! re-exports it together with a few device helpers.
//!
//! ```rust,ignore
//! let dev = candle_rocm::device(0)?;
//! let t = candle_rocm::Tensor::zeros((2, 3), candle_rocm::DType::F32, &dev)?;
//! ```

pub use candle_core::*;

/// Create a ROCm device for the given GPU ordinal.
pub fn device(ordinal: usize) -> Result<Device> {
    Device::new_rocm(ordinal)
}

/// Number of ROCm-capable GPUs visible to the runtime.
pub fn device_count() -> Result<usize> {
    Ok(candle_core::rocm::device_count())
}

/// Human-readable name of the GPU (e.g. "AMD Radeon RX 7900 XTX").
pub fn device_name(ordinal: usize) -> Result<String> {
    candle_core::rocm::device_name(ordinal)
}

/// Total VRAM in bytes for the given GPU.
pub fn total_vram(ordinal: usize) -> Result<usize> {
    Ok(candle_core::rocm::mem_info(ordinal)?.1)
}

/// `(free, total)` VRAM in bytes for the given GPU.
pub fn mem_info(ordinal: usize) -> Result<(usize, usize)> {
    candle_core::rocm::mem_info(ordinal)
}

/// Returns true if at least one ROCm device is available.
pub fn is_available() -> bool {
    candle_core::rocm::device_count() > 0
}

/// GPU target of the device, e.g. `gfx1100`.
pub fn device_arch(ordinal: usize) -> Result<String> {
    candle_core::rocm::device_arch(ordinal)
}

/// Version of the loaded HIP runtime: `7.2.x` with ROCm 7.2, `7.16.0` with ROCm 10.1.
pub fn hip_version() -> Result<candle_core::rocm::hip_runtime::track::HipVersion> {
    candle_core::rocm::hip_version()
}

/// HIP runtime and track, GPU target, kernel targets and GEMM implementation of a device.
/// `Display` prints a one-line summary.
pub fn runtime_info(ordinal: usize) -> Result<candle_core::rocm::RocmRuntimeInfo> {
    candle_core::rocm::runtime_info(ordinal)
}
