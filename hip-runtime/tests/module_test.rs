use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::module::{compile_kernel, HipModule};
use std::path::Path;

#[test]
fn test_compile_and_load_kernel() {
    let _dev = HipDevice::new(0).unwrap();

    let kernel_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels");
    let src = kernel_dir.join("test_add.hip");
    let out = kernel_dir.join("test_add.hsaco");

    let arch = hip_runtime::module::default_arch();

    // Write a trivial kernel if it doesn't exist
    if !src.exists() {
        std::fs::write(&src, r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void add_scalar(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { data[i] += scalar; }
}
"#).unwrap();
    }

    compile_kernel(&src, &out, &arch).expect("Failed to compile kernel");

    let mut module = HipModule::load(&out).expect("Failed to load module");
    let func = module.get_function("add_scalar").expect("Failed to get function");
    assert!(!func.is_null());
}

#[test]
fn test_launch_add_scalar() {
    let _dev = HipDevice::new(0).unwrap();

    let kernel_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels");
    let src = kernel_dir.join("test_add.hip");
    let out = kernel_dir.join("test_add.hsaco");
    let arch = hip_runtime::module::default_arch();

    if !src.exists() {
        std::fs::write(&src, r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void add_scalar(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { data[i] += scalar; }
}
"#).unwrap();
    }

    compile_kernel(&src, &out, &arch).unwrap();
    let mut module = HipModule::load(&out).unwrap();
    let func = module.get_function("add_scalar").unwrap();

    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let buf = DeviceBuffer::from_slice(&data).unwrap();

    let scalar: f32 = 10.0;
    let n: i32 = 256;
    let mut data_ptr = buf.as_void_ptr();

    let block_size = 256u32;
    let grid_size = 1u32;

    // kernel_params: array of pointers to each argument's value
    unsafe {
        HipModule::launch(
            func,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            &mut [
                &mut data_ptr as *mut _ as *mut std::ffi::c_void,
                &scalar as *const _ as *mut std::ffi::c_void,
                &n as *const _ as *mut std::ffi::c_void,
            ],
        )
        .unwrap();
    }

    let result = buf.to_vec().unwrap();
    for (i, &v) in result.iter().enumerate() {
        let expected = i as f32 + 10.0;
        assert!(
            (v - expected).abs() < 1e-5,
            "Mismatch at {i}: got {v} expected {expected}"
        );
    }
}
