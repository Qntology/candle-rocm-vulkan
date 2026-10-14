# candle-kernels in candle-rocm-vulkan

`candle-kernels/` is a copy of the crates.io release **candle-kernels 0.11.0**
(https://github.com/huggingface/candle, Copyright Hugging Face, Inc. and the candle contributors,
MIT OR Apache-2.0; see ../NOTICE and ../licenses/). `candle-core` depends on it by path instead of
through crates.io, so the fork can fix kernel sources without waiting for an upstream release.

Nothing else in the workspace uses this crate. Keep it the only copy in a build: it links the
`moe` static library, and two copies of it would define the same symbols twice.

## Changes against the crates.io release

### 1. CUDA 13: `__hmax_nan` / `__hmin_nan` for `__half` (src/compatibility.cuh, src/cuda_utils.cuh)

`compatibility.cuh` defined `__hmax_nan(__half, __half)` and `__hmin_nan(__half, __half)` for
`__CUDA_ARCH__ < 800`, because CUDA 12 and older only declare them for sm_80 and newer. CUDA 13 declares them for
every architecture, so building for any GPU below sm_80 with CUDA 13.2 failed:

    compatibility.cuh(11): error: function "__hmax_nan(__half, __half)" has already been defined
    (previous definition at line 3399 of ...\CUDA\v13.2\include\cuda_fp16.hpp)

Whether a toolkit already has the functions cannot be tested in the preprocessor, so the half helpers got names no
toolkit uses: `candle_hmax_nan` / `candle_hmin_nan`. They call the toolkit's function for sm_80 and newer (as before)
and the portable form below that (the same code the old shim had). The four uses in `atomicMaxf` / `atomicMinf` and the
`__half` `maxg` / `ming` in `cuda_utils.cuh` call them. The `__nv_bfloat16` versions (sm_80+ only) are unchanged.

Behaviour on CUDA 12 is identical (same function per architecture); CUDA 13 builds below sm_80 now compile.
Checked with the real headers replaced by mock ones on a host compiler (CUDA 12-like and CUDA 13-like toolkit,
sm_53 / 61 / 75 / 80 and the host pass; the original reproduces the redefinition error, the patched one compiles and
gives the same NaN-propagating results). It is built with a real nvcc by the cross-verify CI of logis-center.

Upstream candle-kernels has the same problem; this change can be offered to huggingface/candle.
