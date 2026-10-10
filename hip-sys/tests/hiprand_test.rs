//! Tests for custom Philox RNG kernel (replaces hipRAND which lacks gfx1010 support).

use hip_sys::hip_runtime::*;
use std::ffi::CString;
use std::ptr;

/// `HIP_ARCH`, else the gfx target of device 0, else `gfx1010`.
fn gpu_arch() -> String {
    if let Ok(a) = std::env::var("HIP_ARCH") {
        if !a.trim().is_empty() {
            return a.trim().to_string();
        }
    }
    let mut props = hipDevicePropR0600Storage::zeroed();
    if unsafe { hipGetDevicePropertiesR0600(&mut props, 0) } == HIP_SUCCESS {
        let name = String::from_utf8_lossy(props.gcn_arch_name()).to_string();
        let base = name.split(':').next().unwrap_or("").trim().to_string();
        if base.starts_with("gfx") {
            return base;
        }
    }
    "gfx1010".to_string()
}

/// `HIPCC`, else `bin/hipcc` of `ROCM_PATH` / `HIP_PATH` (legacy HIP SDK or ROCm Core SDK), else
/// `/opt/rocm/bin/hipcc`, else `hipcc` from `PATH`.
fn hipcc() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("HIPCC") {
        if !p.trim().is_empty() {
            return std::path::PathBuf::from(p.trim());
        }
    }
    let exe = if cfg!(windows) { "hipcc.exe" } else { "hipcc" };
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(root) = std::env::var(var) {
            let p = std::path::Path::new(root.trim().trim_matches('"')).join("bin").join(exe);
            if p.is_file() {
                return p;
            }
        }
    }
    let p = std::path::PathBuf::from("/opt/rocm/bin/hipcc");
    if p.is_file() {
        p
    } else {
        std::path::PathBuf::from(exe)
    }
}

fn compile_rng_kernel() -> std::path::PathBuf {
    let kernel_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels");
    let src = kernel_dir.join("rng.hip");
    let arch = gpu_arch();
    let out = kernel_dir.join(format!("rng-{arch}.hsaco"));

    if !out.exists() {
        let status = std::process::Command::new(hipcc())
            .args(["--genco", &format!("--offload-arch={arch}"), "-o"])
            .arg(&out)
            .arg(&src)
            .status()
            .expect("hipcc not found");
        assert!(status.success(), "hipcc failed to compile rng.hip");
    }
    out
}

fn load_function(hsaco: &std::path::Path, name: &str) -> hipFunction_t {
    let c_path = CString::new(hsaco.to_str().unwrap()).unwrap();
    let mut module: hipModule_t = ptr::null_mut();
    assert_eq!(
        unsafe { hipModuleLoad(&mut module, c_path.as_ptr()) },
        HIP_SUCCESS,
        "hipModuleLoad failed"
    );
    let c_name = CString::new(name).unwrap();
    let mut func: hipFunction_t = ptr::null_mut();
    assert_eq!(
        unsafe { hipModuleGetFunction(&mut func, module, c_name.as_ptr()) },
        HIP_SUCCESS,
        "hipModuleGetFunction failed for {name}"
    );
    func
}

#[test]
fn test_rng_uniform() {
    unsafe { hipSetDevice(0); }
    let hsaco = compile_rng_kernel();
    let func = load_function(&hsaco, "rng_uniform_f32");

    let n: u32 = 1024;
    let bytes = n as usize * std::mem::size_of::<f32>();
    let mut d_out: *mut std::ffi::c_void = ptr::null_mut();
    unsafe { hipMalloc(&mut d_out, bytes); }

    let seed: u64 = 42;
    let offset: u64 = 0;
    let block_size: u32 = 256;
    // Each thread generates 4 values
    let threads_needed = (n + 3) / 4;
    let grid_size = (threads_needed + block_size - 1) / block_size;

    let mut params: [*mut std::ffi::c_void; 4] = [
        &mut d_out as *mut _ as *mut _,
        &n as *const _ as *mut _,
        &seed as *const _ as *mut _,
        &offset as *const _ as *mut _,
    ];

    unsafe {
        let err = hipModuleLaunchKernel(
            func, grid_size, 1, 1, block_size, 1, 1,
            0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
        );
        assert_eq!(err, HIP_SUCCESS, "kernel launch failed");
        hipDeviceSynchronize();
    }

    let mut result = vec![0.0f32; n as usize];
    unsafe {
        hipMemcpy(
            result.as_mut_ptr() as *mut _,
            d_out, bytes,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        );
        hipFree(d_out);
    }

    for (i, &v) in result.iter().enumerate() {
        assert!(v > 0.0 && v <= 1.0, "Uniform value out of range at {i}: {v}");
    }

    // Basic statistical check: mean should be near 0.5
    let mean: f32 = result.iter().sum::<f32>() / n as f32;
    assert!(
        (mean - 0.5).abs() < 0.1,
        "Uniform mean {mean} too far from 0.5"
    );
}

#[test]
fn test_rng_normal() {
    unsafe { hipSetDevice(0); }
    let hsaco = compile_rng_kernel();
    let func = load_function(&hsaco, "rng_normal_f32");

    let n: u32 = 4096;
    let bytes = n as usize * std::mem::size_of::<f32>();
    let mut d_out: *mut std::ffi::c_void = ptr::null_mut();
    unsafe { hipMalloc(&mut d_out, bytes); }

    let seed: u64 = 42;
    let offset: u64 = 0;
    let mean_param: f32 = 0.0;
    let stddev_param: f32 = 1.0;
    let block_size: u32 = 256;
    let threads_needed = (n + 3) / 4;
    let grid_size = (threads_needed + block_size - 1) / block_size;

    let mut params: [*mut std::ffi::c_void; 6] = [
        &mut d_out as *mut _ as *mut _,
        &n as *const _ as *mut _,
        &seed as *const _ as *mut _,
        &offset as *const _ as *mut _,
        &mean_param as *const _ as *mut _,
        &stddev_param as *const _ as *mut _,
    ];

    unsafe {
        let err = hipModuleLaunchKernel(
            func, grid_size, 1, 1, block_size, 1, 1,
            0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
        );
        assert_eq!(err, HIP_SUCCESS, "kernel launch failed");
        hipDeviceSynchronize();
    }

    let mut result = vec![0.0f32; n as usize];
    unsafe {
        hipMemcpy(
            result.as_mut_ptr() as *mut _,
            d_out, bytes,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        );
        hipFree(d_out);
    }

    let mean: f32 = result.iter().sum::<f32>() / n as f32;
    let variance: f32 = result.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    assert!(mean.abs() < 0.1, "Normal mean {mean} too far from 0");
    assert!(
        (variance - 1.0).abs() < 0.2,
        "Normal variance {variance} too far from 1.0"
    );
}

#[test]
fn test_rng_seeded_reproducibility() {
    unsafe { hipSetDevice(0); }
    let hsaco = compile_rng_kernel();
    let func = load_function(&hsaco, "rng_uniform_f32");

    let n: u32 = 256;
    let bytes = n as usize * std::mem::size_of::<f32>();
    let block_size: u32 = 256;
    let threads_needed = (n + 3) / 4;
    let grid_size = (threads_needed + block_size - 1) / block_size;

    let generate = |seed: u64| -> Vec<f32> {
        let offset: u64 = 0;
        let mut d_out: *mut std::ffi::c_void = ptr::null_mut();
        unsafe { hipMalloc(&mut d_out, bytes); }

        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut d_out as *mut _ as *mut _,
            &n as *const _ as *mut _,
            &seed as *const _ as *mut _,
            &offset as *const _ as *mut _,
        ];

        unsafe {
            hipModuleLaunchKernel(
                func, grid_size, 1, 1, block_size, 1, 1,
                0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
            );
            hipDeviceSynchronize();
        }

        let mut result = vec![0.0f32; n as usize];
        unsafe {
            hipMemcpy(
                result.as_mut_ptr() as *mut _,
                d_out, bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost,
            );
            hipFree(d_out);
        }
        result
    };

    let run1 = generate(12345);
    let run2 = generate(12345);
    assert_eq!(run1, run2, "Same seed should produce identical results");

    let run3 = generate(99999);
    assert_ne!(run1, run3, "Different seeds should produce different results");
}
