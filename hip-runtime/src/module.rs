//! HIP module and kernel loading.

use crate::error::{check_hip, HipError, Result};
use hip_sys::hip_runtime;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::Path;
use std::sync::Mutex;

pub struct HipModule {
    module: hip_runtime::hipModule_t,
    functions: Mutex<HashMap<String, usize>>,
}

// A loaded module and its function handles are immutable and usable from any thread.
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    /// Load a compiled code object (or offload bundle) from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| {
            HipError::KernelCompileFailed {
                msg: format!("invalid module path {path:?}: {e}"),
            }
        })?;
        let mut module = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipModuleLoad(&mut module, c_path.as_ptr()) })?;
        Ok(Self {
            module,
            functions: Mutex::new(HashMap::new()),
        })
    }

    /// Load a code object (or offload bundle) from memory. The image must outlive the call.
    pub fn load_data(image: &[u8]) -> Result<Self> {
        let mut module = std::ptr::null_mut();
        check_hip(unsafe {
            hip_runtime::hipModuleLoadData(&mut module, image.as_ptr() as *const c_void)
        })?;
        Ok(Self {
            module,
            functions: Mutex::new(HashMap::new()),
        })
    }

    /// Get a kernel function by name. Caches the lookup.
    pub fn get_function(&self, name: &str) -> Result<hip_runtime::hipFunction_t> {
        let mut functions = self.functions.lock().unwrap();
        if let Some(&func) = functions.get(name) {
            return Ok(func as hip_runtime::hipFunction_t);
        }
        let c_name = CString::new(name).map_err(|_| HipError::KernelNotFound {
            name: name.to_string(),
        })?;
        let mut func = std::ptr::null_mut();
        let code = unsafe { hip_runtime::hipModuleGetFunction(&mut func, self.module, c_name.as_ptr()) };
        if code != hip_runtime::HIP_SUCCESS || func.is_null() {
            return Err(HipError::KernelNotFound {
                name: format!("{name} ({})", crate::error::error_string(code)),
            });
        }
        functions.insert(name.to_string(), func as usize);
        Ok(func)
    }

    /// Launch a kernel on the default stream.
    ///
    /// # Safety
    /// `params` must match the kernel signature and point to live values.
    pub unsafe fn launch(
        func: hip_runtime::hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        check_hip(hip_runtime::hipModuleLaunchKernel(
            func,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { hip_runtime::hipModuleUnload(self.module) };
        }
    }
}

/// Compile a .hip source file to a code object using hipcc.
pub fn compile_kernel(src: &Path, out: &Path, arch: &str) -> Result<()> {
    let hipcc = crate::toolchain::hipcc_path().ok_or_else(|| HipError::KernelCompileFailed {
        msg: "hipcc not found (set HIPCC, ROCM_PATH or HIP_PATH)".to_string(),
    })?;
    let mut cmd = std::process::Command::new(&hipcc);
    cmd.arg("--genco");
    for a in arch.split([',', ';', ' ']).filter(|s| !s.is_empty()) {
        cmd.arg(format!("--offload-arch={a}"));
    }
    cmd.arg("-O3").arg("-o").arg(out).arg(src);
    let status = cmd
        .status()
        .map_err(|e| HipError::KernelCompileFailed { msg: format!("{hipcc:?}: {e}") })?;
    if !status.success() {
        return Err(HipError::KernelCompileFailed {
            msg: format!("{hipcc:?} exited with {status}"),
        });
    }
    Ok(())
}
