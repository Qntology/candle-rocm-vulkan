# candle-rocm

ROCm/HIP backend for the [candle](https://github.com/huggingface/candle) ML framework, ported to
**candle 0.11.0**. The patched `candle-core` / `candle-nn` are drop-in replacements for the crates.io
0.11.0 releases (same version number), so `candle-transformers` and the rest of the ecosystem keep
working unchanged through `[patch.crates-io]`.

## Requirements

- AMD GPU supported by ROCm / the HIP SDK (RDNA2 `gfx1030`, RDNA3 `gfx1100`-`gfx1102`, CDNA, ...)
- Linux: ROCm 6.x (`/opt/rocm`). Windows: AMD HIP SDK 6.x (`HIP_PATH`, e.g. `C:\Program Files\AMD\ROCm\6.2\`)
- `hipcc` (or the SDK `clang++`) to compile the kernels at build time, `amdhip64` and `rocblas` to link
- At run time `rocblas` (+ its `rocblas/library` Tensile files) must be found next to the
  executable or on `PATH`

## Using it from an existing candle 0.11 project

```toml
[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
candle-transformers = "0.11.0"
candle-rocm = { path = "../crates/candle-rocm/candle-backend", optional = true }

[features]
rocm = ["dep:candle-rocm", "candle-core/rocm", "candle-nn/rocm"]

[patch.crates-io]
candle-core = { path = "../crates/candle-rocm/candle-core" }
candle-nn = { path = "../crates/candle-rocm/candle-nn" }
```

```rust
use candle_core::{Device, Tensor};

let dev = Device::new_rocm(0)?;
let a = Tensor::randn(0f32, 1., (128, 64), &dev)?;
let b = Tensor::randn(0f32, 1., (64, 256), &dev)?;
let c = a.matmul(&b)?;
```

## Build configuration

| variable | meaning |
| --- | --- |
| `ROCM_PATH` / `HIP_PATH` | ROCm / HIP SDK root (auto-detected otherwise) |
| `HIP_ARCH` or `CANDLE_ROCM_ARCHS` | GPU targets, e.g. `gfx1100` or `gfx1030,gfx1100` (auto-detected with `amdgpu-arch`/`hipInfo`, default `gfx1030,gfx1100,gfx1101,gfx1102`) |
| `HIPCC` | explicit compiler path |
| `CANDLE_ROCM_HIPCC_FLAGS` | extra compiler flags |
| `CANDLE_ROCM_LIB_DIR` | extra library search path for `amdhip64` / `rocblas` |
| `CANDLE_ROCM_SKIP_KERNEL_BUILD=1` | type-check only (no kernels embedded) |

Run time switches: `CANDLE_ROCM_ASYNC_ALLOC=0` disables the stream ordered allocator,
`CANDLE_ROCM_TRIM_ON_SYNC=0` keeps the memory pool after `Device::synchronize`.

## Device utilities

```rust
candle_core::rocm::device_count()     // number of GPUs (0 when no driver)
candle_core::rocm::device_name(0)?    // e.g. "AMD Radeon RX 7900 XTX"
candle_core::rocm::mem_info(0)?       // (free, total) VRAM in bytes
dev.as_rocm_device()?.trim_memory_pool()?
candle_rocm::is_available() / device_count() / mem_info(0) / total_vram(0)
```

## What runs on the GPU

- unary / binary / comparison / affine / powf / elu / where / casts (all dtypes incl. `F8E4M3`)
- reductions (sum, min, max, argmin, argmax), copies, cat, index_select, gather, scatter, index_add
- matmul through rocBLAS (f32, f64, f16/bf16 with f32 accumulation)
- quantized GGUF tensors (Q4_0 ... Q8K): dequantize, matmul (`QMatMul`), embedding
- candle-nn: `softmax_last_dim`, `rms_norm`, `layer_norm`, `sigmoid`, `rope`, `rope_i`, `rope_thd`

Convolutions, pooling, upsampling, arg-sort and custom ops without a `rocm_fwd` run through the
CPU and the result is copied back to the GPU.

## Vulkan backend (`vulkan` feature)

`candle-core` / `candle-nn` also contain the Vulkan (and OpenCL) backends of
[rexlunae/candle-core](https://github.com/rexlunae/candle-core) and
[rexlunae/candle-nn](https://github.com/rexlunae/candle-nn), merged with the ROCm port so a single
`[patch.crates-io]` serves every backend.

```toml
[features]
vulkan = ["candle-core/vulkan", "candle-nn/vulkan"]
```

```rust
let dev = Device::new_vulkan(0)?;                      // same logical device for every call with 0
candle_core::vulkan_backend::device_count()?;          // usable Vulkan devices, GPUs first
let v = dev.as_vulkan_device()?;
v.name(); v.device_type(); v.mem_info()?;              // (free, total) bytes of the tensor heap
v.trim_memory_pool()?;                                 // release the buffer reuse pool
```

- No SDK is needed to build: the loader (`vulkan-1.dll` / `libvulkan.so.1`) is opened at run time
  through `ash`, kernels are GLSL compiled with `naga` the first time they are used.
- Tensors live in host visible memory (cached system memory, or unified memory on APUs). Kernels
  run on the GPU for f32 unary / affine / binary (strided, broadcast), sum / max / min, matmul
  and the candle-nn softmax / rms-norm / layer-norm / sigmoid; every other op and dtype runs the
  CPU implementation directly on the mapped memory.
- Quantized GGUF tensors keep their blocks on the device; `QMatMul` uses the CPU k-quant kernels
  on the mapped weights.

| variable | meaning |
| --- | --- |
| `CANDLE_VULKAN_NATIVE=1/0` (`JOSHUA_VULKAN_NATIVE`) | force the GPU kernels on / off (default: on for integrated GPUs and device local memory) |
| `CANDLE_VULKAN_NATIVE_MIN_ELEMS`, `CANDLE_VULKAN_NATIVE_MIN_FLOPS` | size below which ops stay on the CPU path (32768 elements, 4 MFLOP) |
| `CANDLE_VULKAN_MEMORY=device` | prefer device local host visible memory (resizable BAR) |
| `CANDLE_VULKAN_POOL_MB` | size of the buffer reuse pool (default 1/8 of the heap, at most 1 GiB, 0 disables) |
| `CANDLE_VULKAN_SLAB=1/0` | always / never place buffers up to 16 MiB in shared 64 MiB slabs (default: once half of `maxMemoryAllocationCount`, 4096 on most Windows drivers, is in use) |
| `CANDLE_VULKAN_DEBUG=1` | log why a kernel fell back to the CPU path |

`cargo test -p candle-core --features vulkan --test vulkan_tests` and
`cargo test -p candle-nn --features vulkan --test vulkan_ops` (Mesa lavapipe works for testing).

## Layout

```
candle-rocm/
├── candle-backend/   # convenience crate (candle-rocm)
├── candle-core/      # candle-core 0.11.0 + rocm backend (src/rocm_backend, kernels/*.hip, build.rs)
│                     #   + vulkan / opencl backends (src/vulkan_backend, src/opencl_backend)
├── candle-nn/        # candle-nn 0.11.0 + rocm_fwd / vulkan_fwd for the fused ops
├── hip-runtime/      # safe wrappers for the HIP runtime, rocBLAS, hipRAND
├── hip-sys/          # raw FFI bindings
├── kernels/          # original runtime-compiled kernels (only used by hip-runtime's own tests)
└── poc/              # original proof-of-concept programs
```

## Tests

`cargo test -p candle-core --features rocm --test rocm_tests` and
`cargo test -p candle-nn --features rocm --test rocm_ops` compare every op with the CPU backend.

## License

MIT
