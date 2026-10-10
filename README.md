# candle-rocm-vulkan

AMD ROCm/HIP and Vulkan GPU backends for the [candle](https://github.com/huggingface/candle) ML
framework, built on **candle 0.11.0**.

The patched `candle-core` and `candle-nn` crates are drop-in replacements for the crates.io 0.11.0
releases and keep the same version number. Through `[patch.crates-io]`, `candle-transformers` and
the rest of the candle ecosystem work unchanged, and one patch serves the CPU, ROCm, Vulkan and
OpenCL backends.

| backend | feature | GPUs | notes |
| --- | --- | --- | --- |
| ROCm / HIP | `rocm` | AMD (RDNA2/3, CDNA, ...) | HIP kernels plus rocBLAS; needs the ROCm or HIP SDK |
| Vulkan | `vulkan` | Any Vulkan 1.1+ GPU: AMD, NVIDIA, Intel, integrated | No SDK needed; loads the system Vulkan loader at run time |
| OpenCL | `opencl` | Intel Arc / iGPU through the OpenCL ICD | Experimental |

## Usage

```toml
[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
candle-transformers = "0.11.0"
candle-rocm = { git = "https://github.com/Qntology/candle-rocm-vulkan.git", optional = true }

[features]
rocm = ["dep:candle-rocm", "candle-core/rocm", "candle-nn/rocm"]
vulkan = ["candle-core/vulkan", "candle-nn/vulkan"]

[patch.crates-io]
candle-core = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
candle-nn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
```

```rust
use candle_core::{Device, Tensor};

let dev = Device::new_rocm(0)?;      // or Device::new_vulkan(0)?
let a = Tensor::randn(0f32, 1., (128, 64), &dev)?;
let b = Tensor::randn(0f32, 1., (64, 256), &dev)?;
let c = a.matmul(&b)?;
```

## ROCm / HIP backend (`rocm` feature)

### Requirements

- An AMD GPU supported by ROCm or the HIP SDK, such as RDNA2 `gfx1030`, RDNA3 `gfx1100`-`gfx1103`, RDNA3.5
  `gfx1150`-`gfx1153`, RDNA4 `gfx1200`/`gfx1201`, or CDNA.
- A ROCm installation from either release stream ("track", see below):
  - `legacy`: ROCm 6.x - 7.2 on Linux (`/opt/rocm`), the AMD HIP SDK 6.x / 7.x on Windows (`HIP_PATH`).
  - `core`: the ROCm Core SDK 7.14 / 10.x built by TheRock (ROCm 10.1 ships HIP 7.16): native packages
    (`/opt/rocm/core-10.x`), tarballs, or the `rocm-sdk` Python wheels. On Windows it is currently distributed as
    tarballs and wheels: extract the tarball to `C:\Program Files\AMD\ROCm\10.1` (or anywhere and set
    `ROCM_PATH`).
- `hipcc` (or the SDK `clang++`) to compile the kernels at build time, plus `amdhip64` and `rocblas` to link.
- At run time, `rocblas` and its `rocblas/library` Tensile files must be next to the executable or on `PATH`.

### Two ROCm tracks

Both tracks ship HIP 7.x with the same `amdhip64_7` / `rocblas` ABI, so the FFI and the backend are shared and
one source tree builds against either. What differs is handled per track:

| | `legacy` (ROCm / HIP SDK 6.x - 7.2) | `core` (ROCm Core SDK 10.x) |
| --- | --- | --- |
| install layout | `bin/clang++`, `llvm/bin` | `lib/llvm/bin`, `core-10.x` folders, `rocm-sdk path --root` |
| compiler | classic offload driver | LLVM 24, new offload driver by default (output still a HIP fat binary, checked at build time) |
| default targets (no GPU detected) | `gfx1030,gfx1100,gfx1101,gfx1102` | every Radeon / Ryzen target of ROCm 10.1 |
| GPUs | everything the installed release supports | `gfx908`, `gfx90a`, `gfx942`, `gfx950`, `gfx1030`, `gfx110[0-3]`, `gfx115[0-3]`, `gfx120[01]` |

`hip-sys` picks the installation once and `candle-core` compiles its kernels with the same one: `ROCM_PATH` /
`HIP_PATH` win, otherwise `CANDLE_ROCM_TRACK=auto` (default) takes the newest `core` installation when every
detected GPU is supported by it and the newest `legacy` one otherwise; `CANDLE_ROCM_TRACK=legacy|core` forces a
track.

At run time the device reads its `gfx` target. When the rocBLAS of the loaded ROCm has no kernels for it (rocBLAS
would abort the process on the first GEMM), matmuls switch to the HIP GEMM kernel of `kernels/gemm.hip`; so do
F32/F64 GEMMs for which rocBLAS returns `not implemented` / `arch mismatch`. A GPU that the embedded kernels were
not compiled for is reported with the `HIP_ARCH` to rebuild with.

### Build configuration

| variable | meaning |
| --- | --- |
| `ROCM_PATH` / `HIP_PATH` | ROCm or HIP SDK root (auto-detected otherwise) |
| `CANDLE_ROCM_TRACK` | `auto` (default), `legacy` or `core`: which installation to use when `ROCM_PATH` is not set |
| `HIP_ARCH` or `CANDLE_ROCM_ARCHS` | GPU targets, e.g. `gfx1100` or `gfx1030,gfx1100`, or `all` (every ROCm 10.1 target, for redistributable builds). Auto-detected with `amdgpu-arch`/`hipInfo`; the defaults depend on the track (see above). Targets of the built-in lists that the compiler does not know are skipped |
| `HIPCC` | explicit compiler path |
| `CANDLE_ROCM_HIPCC_FLAGS` | extra compiler flags |
| `CANDLE_ROCM_LIB_DIR` | extra library search path for `amdhip64` / `rocblas` |
| `CANDLE_ROCM_SKIP_KERNEL_BUILD=1` | type-check only; no kernels are embedded |

### Run-time configuration

| variable | default | meaning |
| --- | --- | --- |
| `CANDLE_ROCM_ASYNC_ALLOC` | `1` | use the stream-ordered allocator (`hipMallocAsync`) |
| `CANDLE_ROCM_POOL_RELEASE_ZERO` | `1` | set the pool release threshold to 0 so freed memory goes back to the driver at sync points |
| `CANDLE_ROCM_TRIM_ON_SYNC` | `1` | trim the memory pool on `Device::synchronize` |
| `CANDLE_ROCM_RESOURCE_CACHE_MB` | `0` | value for the HIP runtime's `GPU_RESOURCE_CACHE_SIZE`, unless that variable is already set. The runtime default caches freed buffers as dedicated VRAM |
| `CANDLE_ROCM_GEMM` | `auto` | `rocblas`, `hip` (the tiled HIP kernel), or `auto`: rocBLAS unless it has no kernels for this GPU |

### Device utilities

```rust
candle_core::rocm::device_count()        // number of GPUs (0 when no driver)
candle_core::rocm::device_name(0)?       // e.g. "AMD Radeon RX 7900 XTX"
candle_core::rocm::mem_info(0)?          // (free, total) VRAM in bytes
candle_core::rocm::device_arch(0)?       // e.g. "gfx1100"
candle_core::rocm::hip_version()?        // e.g. 7.2.x (ROCm 7.2) or 7.16.0 (ROCm 10.1)
candle_core::rocm::runtime_info(0)?      // HIP version and track, GPU target, kernel targets, GEMM backend
let r = dev.as_rocm_device()?;
r.trim_memory_pool()?;                   // return the allocator pool to the driver
r.release_cached_resources()?;           // sync, trim, recreate the rocBLAS handle (its workspace)
r.gemm_backend() / r.set_gemm_backend(candle_core::rocm::GemmBackend::Hip)
candle_rocm::is_available() / device_count() / mem_info(0) / total_vram(0) / device_arch(0) / runtime_info(0)
```

### What runs on the GPU

- Unary, binary, comparison, affine, powf, elu, where, and casts, for all dtypes including `F8E4M3`.
- Reductions (sum, min, max, argmin, argmax), copies, cat, index_select, gather, scatter, and index_add.
- Matmul through rocBLAS, or the HIP GEMM kernel when rocBLAS cannot run on the GPU: f32, f64, and f16/bf16 with
  f32 accumulation.
- Quantized GGUF tensors (Q4_0 to Q8K): dequantize, matmul (`QMatMul`), and embedding.
- In candle-nn: `softmax_last_dim`, `rms_norm`, `layer_norm`, `sigmoid`, `rope`, `rope_i`, and `rope_thd`.

Convolutions, pooling, upsampling, arg-sort, and custom ops without a `rocm_fwd` run on the CPU, and the
result is copied back to the GPU.

## Vulkan backend (`vulkan` feature)

```rust
let dev = Device::new_vulkan(0)?;                      // ordinal 0 always returns the same logical device
candle_core::vulkan_backend::device_count()?;          // usable Vulkan devices, best first
let v = dev.as_vulkan_device()?;
v.name(); v.device_type(); v.mem_info()?;              // (free, total) bytes
v.allocated_bytes(); v.pooled_bytes();
v.trim_memory_pool()?;                                 // release the buffer reuse pool
```

- **No SDK is needed to build.** The loader (`vulkan-1.dll` / `libvulkan.so.1`) is opened at run time
  through `ash`. Kernels are GLSL compiled with `naga` the first time they are used.
- **Device selection.** Discrete GPUs come first, ordered by VRAM, then integrated GPUs, then the rest.
  CPU implementations such as lavapipe are skipped unless `CANDLE_VULKAN_ALLOW_CPU=1` is set.
- **Queue.** A compute-only (async compute) queue is used when the device has one, because it was
  measured to be faster. Otherwise, or with `CANDLE_VULKAN_QUEUE=graphics`, the graphics (3D) queue is used.
- **Quantized weights on the GPU.** Q8_0 GGUF weights are uploaded once to device-local memory (VRAM).
  `QMatMul` runs Q8 GEMV (decode) and GEMM (prefill) compute kernels on them, so the weights are not
  read from system memory on every token. Other quantization types use the CPU k-quant kernels on
  the mapped blocks.
- **Dense matmul.** Dense f32/f16 weights that are reused get a device-local mirror. Large
  activation-by-activation matmuls also run on the GPU.
- **Everything else** lives in host-visible memory: cached system memory, or unified memory on APUs.
  f32 unary/affine/binary ops, reductions, and the candle-nn softmax / rms-norm / layer-norm / sigmoid
  run as GPU kernels where it pays off. Other ops and dtypes run the CPU implementation directly on
  the mapped memory.
- **Memory budget.** Each upload is checked against the heap budget minus a reserve (at most 1/4 of
  the heap). Weights that do not fit stay in host memory and use the CPU path, so nothing spills into
  shared memory.

| variable | default | meaning |
| --- | --- | --- |
| `CANDLE_VULKAN_DEVICE` | – | pick the device whose name contains this substring (case-insensitive) |
| `CANDLE_VULKAN_ALLOW_CPU` | `0` | allow CPU Vulkan implementations (lavapipe, SwiftShader) |
| `CANDLE_VULKAN_QUEUE` | compute | `graphics` / `3d` / `universal` forces the graphics queue |
| `CANDLE_VULKAN_GPU_WEIGHTS` | `1` | keep Q8_0 weights in VRAM and run the Q8 GEMV/GEMM kernels |
| `CANDLE_VULKAN_DENSE_MIRROR` | `1` | mirror reused dense weights into VRAM |
| `CANDLE_VULKAN_MIRROR_AFTER` | `2` | number of uses before a dense weight is mirrored |
| `CANDLE_VULKAN_ACT_MATMUL` | `1` | run large activation-by-activation matmuls on the GPU |
| `CANDLE_VULKAN_ACT_MIN_MFLOP` | `256` | minimum size (MFLOP) for the above |
| `CANDLE_VULKAN_SCRATCH_MB` | `128` | cap of the device-local activation scratch buffer on discrete GPUs (large inputs are processed in bands) |
| `CANDLE_VULKAN_VRAM_RESERVE_MB` | `512` | VRAM kept free for the driver, display, and other apps (capped at 1/4 of the heap) |
| `CANDLE_VULKAN_PRETEND_INTEGRATED` | `0` | treat a discrete GPU like an integrated one (for testing) |
| `CANDLE_VULKAN_SPIN_US` | `1000` | microseconds to poll a fence before blocking on it |
| `CANDLE_VULKAN_NATIVE` | auto | force the element-wise GPU kernels on or off |
| `CANDLE_VULKAN_NATIVE_MIN_ELEMS` / `_MIN_FLOPS` | `32768` / 4 MFLOP | size below which element-wise ops and dense matmuls stay on the CPU path |
| `CANDLE_VULKAN_MEMORY=device` | – | prefer device-local host-visible memory (resizable BAR) for tensors |
| `CANDLE_VULKAN_POOL_MB` | 1/8 of heap, max 1 GiB | size of the buffer reuse pool (`0` disables it) |
| `CANDLE_VULKAN_SLAB` | auto | `1`/`0` always or never sub-allocate buffers up to 16 MiB from shared 64 MiB slabs |
| `CANDLE_VULKAN_DEBUG` | `0` | print why a kernel fell back to the CPU path |

## Layout

```
candle-rocm-vulkan/
├── candle-core/      # candle-core 0.11.0
│   ├── src/rocm_backend/     # ROCm/HIP backend (kernels in kernels/*.hip, built by build.rs)
│   ├── src/vulkan_backend/   # Vulkan backend (mod.rs, shaders.rs, qgpu.rs = Q8 / VRAM weights)
│   └── src/opencl_backend/   # OpenCL backend
├── candle-nn/        # candle-nn 0.11.0 + rocm_fwd / vulkan_fwd for the fused ops
├── candle-backend/   # convenience crate `candle-rocm` (device helpers)
├── hip-runtime/      # safe wrappers for the HIP runtime, rocBLAS, hipRAND
├── hip-sys/          # raw FFI bindings
├── kernels/          # runtime-compiled kernels used by hip-runtime's RNG and tests
├── poc/              # original HIP / rocBLAS proof-of-concept sources
├── licenses/         # third-party license texts
├── LICENSE           # Apache License 2.0
└── NOTICE            # attributions
```

## Tests

```
cargo test -p candle-core --features rocm   --test rocm_tests
cargo test -p candle-nn   --features rocm   --test rocm_ops
cargo test -p candle-core --features vulkan --test vulkan_tests
cargo test -p candle-nn   --features vulkan --test vulkan_ops
```

Every op is compared with the CPU backend. Mesa lavapipe works for the Vulkan tests when it is
enabled with `CANDLE_VULKAN_ALLOW_CPU=1`.

## License

Copyright 2026 Qntology

Licensed under the [Apache License, Version 2.0](LICENSE).

This project combines and modifies the following:

- [huggingface/candle](https://github.com/huggingface/candle), licensed MIT OR Apache-2.0 and used here under Apache-2.0
- the Vulkan/OpenCL backends of [rexlunae/candle-core](https://github.com/rexlunae/candle-core) and
  [rexlunae/candle-nn](https://github.com/rexlunae/candle-nn), licensed Apache-2.0
- [Nu11ified/candle-rocm](https://github.com/Nu11ified/candle-rocm), MIT, Copyright (c) 2026 Manas

Their notices are kept in [NOTICE](NOTICE) and [licenses/](licenses/).
