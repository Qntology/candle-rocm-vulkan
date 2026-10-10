//! Safe device management.

use crate::error::{check_hip, HipError, Result};
use crate::track::HipVersion;
use hip_sys::hip_runtime;

#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

/// GPU target of a device as reported by the HIP runtime (`hipDeviceProp_t::gcnArchName`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuArch {
    /// Full target id, e.g. `gfx1100` or `gfx90a:sramecc+:xnack-`.
    pub full: String,
    /// Processor name without target features, e.g. `gfx90a`.
    pub name: String,
}

impl std::fmt::Display for GpuArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.full)
    }
}

impl HipDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        crate::configure_runtime_env();
        let count = Self::device_count()?;
        if ordinal >= count {
            return Err(HipError::HipRuntimeError {
                code: hip_runtime::HIP_ERROR_NO_DEVICE,
                msg: format!("invalid device ordinal {ordinal}, {count} device(s) available"),
            });
        }
        check_hip(unsafe { hip_runtime::hipSetDevice(ordinal as i32) })?;
        Ok(Self { ordinal })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn set_current(&self) -> Result<()> {
        check_hip(unsafe { hip_runtime::hipSetDevice(self.ordinal as i32) })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        check_hip(unsafe { hip_runtime::hipDeviceSynchronize() })
    }

    pub fn device_count() -> Result<usize> {
        crate::configure_runtime_env();
        let mut count: i32 = 0;
        check_hip(unsafe { hip_runtime::hipGetDeviceCount(&mut count) })?;
        Ok(count.max(0) as usize)
    }

    fn handle(&self) -> Result<hip_runtime::hipDevice_t> {
        let mut dev: hip_runtime::hipDevice_t = 0;
        check_hip(unsafe { hip_runtime::hipDeviceGet(&mut dev, self.ordinal as i32) })?;
        Ok(dev)
    }

    pub fn name(&self) -> Result<String> {
        let dev = self.handle()?;
        let mut buf = [0 as std::os::raw::c_char; 256];
        check_hip(unsafe { hip_runtime::hipDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, dev) })?;
        let name = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        Ok(name.to_string_lossy().trim().to_string())
    }

    pub fn total_memory(&self) -> Result<usize> {
        let dev = self.handle()?;
        let mut bytes: usize = 0;
        check_hip(unsafe { hip_runtime::hipDeviceTotalMem(&mut bytes, dev) })?;
        Ok(bytes)
    }

    /// Returns `(free, total)` device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.set_current()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        check_hip(unsafe { hip_runtime::hipMemGetInfo(&mut free, &mut total) })?;
        Ok((free, total))
    }

    /// GPU target (`gfx...`) of this device.
    pub fn arch(&self) -> Result<GpuArch> {
        let mut props = hip_runtime::hipDevicePropR0600Storage::zeroed();
        check_hip(unsafe { hip_runtime::hipGetDevicePropertiesR0600(&mut props, self.ordinal as i32) })?;
        let full = String::from_utf8_lossy(props.gcn_arch_name()).trim().to_string();
        let name = full.split(':').next().unwrap_or("").trim().to_string();
        if !name.starts_with("gfx") {
            return Err(HipError::HipRuntimeError {
                code: hip_runtime::HIP_ERROR_NOT_FOUND,
                msg: format!("unexpected gcnArchName {full:?} for device {}", self.ordinal),
            });
        }
        Ok(GpuArch { full, name })
    }

    fn attribute(&self, attr: i32) -> Result<i32> {
        let mut v: i32 = 0;
        check_hip(unsafe { hip_runtime::hipDeviceGetAttribute(&mut v, attr, self.ordinal as i32) })?;
        Ok(v)
    }

    /// Wavefront size: 32 on RDNA, 64 on GCN / CDNA.
    pub fn warp_size(&self) -> Result<u32> {
        Ok(self.attribute(hip_runtime::hipDeviceAttributeWarpSize)?.max(0) as u32)
    }

    /// Number of compute units.
    pub fn compute_units(&self) -> Result<u32> {
        Ok(self.attribute(hip_runtime::hipDeviceAttributeMultiprocessorCount)?.max(0) as u32)
    }
}

pub fn runtime_version() -> Result<i32> {
    crate::configure_runtime_env();
    let mut v: i32 = 0;
    check_hip(unsafe { hip_runtime::hipRuntimeGetVersion(&mut v) })?;
    Ok(v)
}

/// Version of the loaded HIP runtime, e.g. `7.2.x` (legacy ROCm 7.2) or `7.16.0` (ROCm 10.1).
pub fn runtime_hip_version() -> Result<HipVersion> {
    runtime_version().map(HipVersion::from_raw)
}
