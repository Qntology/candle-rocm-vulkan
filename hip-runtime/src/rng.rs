//! GPU random number generation using custom Philox kernel.
//!
//! Uses our own Philox 4x32-10 kernel instead of hipRAND,
//! because rocRAND lacks pre-compiled gfx1010 kernels.

use crate::error::{check_hip, Result};
use crate::memory::DeviceBuffer;
use crate::module::{compile_kernel, default_arch, HipModule};
use hip_sys::hip_runtime;
use std::path::{Path, PathBuf};

pub struct HipRng {
    module: HipModule,
    seed: u64,
    offset: u64,
}

impl HipRng {
    /// Create RNG with a seed. `kernel_dir` is the path to the kernels/ directory.
    ///
    /// `rng.hip` is compiled with hipcc for [`default_arch`] (`HIP_ARCH` or the current GPU) and
    /// cached per target as `rng-<arch>.hsaco`, rebuilt when `rng.hip` is newer.
    pub fn new(seed: u64, kernel_dir: &Path) -> Result<Self> {
        let src = kernel_dir.join("rng.hip");
        let arch = default_arch();
        let tag: String = arch
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
            .collect();
        let hsaco = kernel_dir.join(format!("rng-{tag}.hsaco"));

        let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let stale = match (modified(&hsaco), modified(&src)) {
            (Some(out), Some(src)) => out < src,
            (None, _) => true,
            (Some(_), None) => false,
        };
        if stale {
            compile_kernel(&src, &hsaco, &arch)?;
        }

        let module = HipModule::load(&hsaco)?;
        Ok(Self { module, seed, offset: 0 })
    }

    /// Load from a pre-compiled .hsaco file.
    pub fn from_hsaco(seed: u64, hsaco: &Path) -> Result<Self> {
        let module = HipModule::load(hsaco)?;
        Ok(Self { module, seed, offset: 0 })
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.offset = 0;
    }

    /// Fill buffer with uniform random values in (0, 1].
    pub fn uniform_f32(&mut self, buf: &mut DeviceBuffer<f32>) -> Result<()> {
        let func = self.module.get_function("rng_uniform_f32")?;
        let n = buf.len() as u32;
        let block_size: u32 = 256;
        let threads_needed = (n + 3) / 4;
        let grid_size = (threads_needed + block_size - 1) / block_size;

        let mut ptr = buf.as_void_ptr();
        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut ptr as *mut _ as *mut _,
            &n as *const _ as *mut _,
            &self.seed as *const _ as *mut _,
            &self.offset as *const _ as *mut _,
        ];

        unsafe {
            HipModule::launch(func, (grid_size, 1, 1), (block_size, 1, 1), 0, &mut params)?;
            check_hip(hip_runtime::hipDeviceSynchronize())?;
        }

        // Advance offset so subsequent calls produce different values
        self.offset += threads_needed as u64;
        Ok(())
    }

    /// Fill buffer with normal random values.
    pub fn normal_f32(&mut self, buf: &mut DeviceBuffer<f32>, mean: f32, std: f32) -> Result<()> {
        let func = self.module.get_function("rng_normal_f32")?;
        let n = buf.len() as u32;
        let block_size: u32 = 256;
        let threads_needed = (n + 3) / 4;
        let grid_size = (threads_needed + block_size - 1) / block_size;

        let mut ptr = buf.as_void_ptr();
        let mut params: [*mut std::ffi::c_void; 6] = [
            &mut ptr as *mut _ as *mut _,
            &n as *const _ as *mut _,
            &self.seed as *const _ as *mut _,
            &self.offset as *const _ as *mut _,
            &mean as *const _ as *mut _,
            &std as *const _ as *mut _,
        ];

        unsafe {
            HipModule::launch(func, (grid_size, 1, 1), (block_size, 1, 1), 0, &mut params)?;
            check_hip(hip_runtime::hipDeviceSynchronize())?;
        }

        self.offset += threads_needed as u64;
        Ok(())
    }

    /// Path to the kernels directory (for convenience).
    pub fn default_kernel_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kernels")
    }
}
