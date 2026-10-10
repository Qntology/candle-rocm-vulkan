//! HIP runtime FFI bindings.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

pub type hipError_t = c_int;
pub type hipDevice_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;
pub type hipMemPool_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;
pub const HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 0x4;
pub const HIP_ERROR_OUT_OF_MEMORY: hipError_t = 2;
pub const HIP_ERROR_INVALID_DEVICE_FUNCTION: hipError_t = 98;
pub const HIP_ERROR_NO_DEVICE: hipError_t = 100;
pub const HIP_ERROR_INVALID_IMAGE: hipError_t = 200;
pub const HIP_ERROR_NO_BINARY_FOR_GPU: hipError_t = 209;
pub const HIP_ERROR_SHARED_OBJECT_INIT_FAILED: hipError_t = 303;
pub const HIP_ERROR_NOT_FOUND: hipError_t = 500;

/// `hipDeviceAttribute_t` values used by this crate (same in the ROCm 7.2 and 10.1 headers).
pub const hipDeviceAttributeMultiprocessorCount: c_int = 63;
pub const hipDeviceAttributeWarpSize: c_int = 87;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
}

/// Legacy (pre ROCm 6.0) layout prefix of `hipDeviceProp_t`, kept for the tests of this crate.
/// Prefer `hipDeviceGetName`, `hipDeviceTotalMem`, `hipMemGetInfo`, `hipDeviceGetAttribute`, or
/// [`hipGetDevicePropertiesR0600`] with [`hipDevicePropR0600Storage`], which have a stable ABI.
#[repr(C)]
#[derive(Debug)]
pub struct hipDeviceProp_t {
    pub name: [c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub _padding: [u8; 4096],
}

/// Size of `hipDeviceProp_tR0600`, the `hipDeviceProp_t` of HIP 6.0 and later. The layout is frozen
/// by the `R0600` symbol version; it is identical in the ROCm 7.2 and ROCm 10.1 (HIP 7.16) headers,
/// for Linux and Windows (MSVC) targets.
pub const HIP_DEVICE_PROP_R0600_SIZE: usize = 1472;
/// Offset of `char gcnArchName[256]` (e.g. `gfx1100` or `gfx90a:sramecc+:xnack-`).
pub const HIP_DEVICE_PROP_R0600_GCN_ARCH_NAME_OFFSET: usize = 1160;
pub const HIP_DEVICE_PROP_R0600_GCN_ARCH_NAME_LEN: usize = 256;

/// Over-allocated, 8-byte aligned storage for a `hipDeviceProp_tR0600` (fields are read by offset).
#[repr(C, align(8))]
pub struct hipDevicePropR0600Storage {
    pub bytes: [u8; 4096],
}

impl hipDevicePropR0600Storage {
    pub fn zeroed() -> Self {
        Self { bytes: [0; 4096] }
    }

    /// `gcnArchName` as written by the runtime (up to the first NUL).
    pub fn gcn_arch_name(&self) -> &[u8] {
        let start = HIP_DEVICE_PROP_R0600_GCN_ARCH_NAME_OFFSET;
        let raw = &self.bytes[start..start + HIP_DEVICE_PROP_R0600_GCN_ARCH_NAME_LEN];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        &raw[..end]
    }
}

extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetDevice(device_id: *mut c_int) -> hipError_t;
    pub fn hipDeviceGet(device: *mut hipDevice_t, ordinal: c_int) -> hipError_t;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: hipDevice_t) -> hipError_t;
    pub fn hipDeviceTotalMem(bytes: *mut usize, device: hipDevice_t) -> hipError_t;
    pub fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device_id: c_int) -> hipError_t;
    pub fn hipGetDevicePropertiesR0600(
        prop: *mut hipDevicePropR0600Storage,
        device_id: c_int,
    ) -> hipError_t;
    pub fn hipDeviceGetAttribute(value: *mut c_int, attr: c_int, device_id: c_int) -> hipError_t;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    pub fn hipRuntimeGetVersion(version: *mut c_int) -> hipError_t;
    pub fn hipDriverGetVersion(version: *mut c_int) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMallocAsync(ptr: *mut *mut c_void, size: usize, stream: hipStream_t) -> hipError_t;
    pub fn hipFreeAsync(ptr: *mut c_void, stream: hipStream_t) -> hipError_t;
    pub fn hipDeviceGetDefaultMemPool(pool: *mut hipMemPool_t, device: c_int) -> hipError_t;
    pub fn hipMemPoolTrimTo(pool: hipMemPool_t, min_bytes_to_hold: usize) -> hipError_t;
    pub fn hipMemPoolSetAttribute(pool: hipMemPool_t, attr: c_int, value: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipGetLastError() -> hipError_t;
    pub fn hipModuleLoad(module: *mut hipModule_t, fname: *const c_char) -> hipError_t;
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
}
