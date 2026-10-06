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
