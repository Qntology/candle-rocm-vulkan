//! Safe device management.

use crate::error::{check_hip, HipError, Result};
use hip_sys::hip_runtime;

#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

impl HipDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
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
}

pub fn runtime_version() -> Result<i32> {
    let mut v: i32 = 0;
    check_hip(unsafe { hip_runtime::hipRuntimeGetVersion(&mut v) })?;
    Ok(v)
}
