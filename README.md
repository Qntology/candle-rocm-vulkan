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

On CUDA, ROCm and Vulkan the fork also provides fused attention with FlashAttention v1 / v2 / v3
tiers and TurboQuant / NVFP4 / MXFP4 / MXFP8 / FP8 KV-cache compression, selected per GPU and
switchable feature by feature (see [Fused attention and TurboQuant](#fused-attention-and-turboquant-flash-attn--turbo-quant-features)).

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

## Fused attention and TurboQuant (`flash-attn` / `turbo-quant` features)

Two accelerator paths shared by the CUDA, ROCm and Vulkan backends (and the CPU):

- **FlashAttention**: fused `softmax(Q Kᵀ · scale + bias) V` with the FlashAttention v1 / v2 / v3
  algorithm tiers, selected per GPU architecture. GQA / MQA, causal and sliding-window masks
  (bottom-right aligned like Dao-AILab FlashAttention), ALiBi, soft-capping, varlen and paged KV,
  split-KV (flash-decoding) for decoding, FP8 (e4m3) K/V with descale factors.
- **TurboQuant**: vector quantization of KV caches (or any rows) after a random rotation, with
  TurboQuant codebooks ([Zandieh et al. 2025](https://github.com/abdelstark/turboquant)) or the
  NVFP4 / MXFP4 / MXFP8 / FP8 formats, selected per GPU architecture, and attention computed
  directly on the compressed cache.

```rust
use candle_nn::flash_attn::{flash_attn, flash_attn_bhsd, FlashAttnConfig};
use candle_nn::turbo_quant::TurboKvCache;

// (batch, seq, heads, head_dim), as with candle-flash-attn
let out = flash_attn(&q, &k, &v, 1.0 / (head_dim as f32).sqrt(), /* causal */ true)?;
// (batch, heads, seq, head_dim), the layout of most candle models, without copies
let out = flash_attn_bhsd(&q, &k, &v, &FlashAttnConfig::new(scale).causal())?;

// compressed KV cache: formats, bits and rotation from the switches below / the GPU
let mut cache = TurboKvCache::new(head_dim, 1024, &device)?;
cache.append(&k, &v)?;                        // (batch, kv_heads, seq, head_dim)
let out = cache.attention(&q, &FlashAttnConfig::new(scale).causal())?;

println!("{}", candle_core::accel::describe(&device)); // what runs on this GPU
```

`candle-flash-attn/` is a drop-in replacement of the crates.io `candle-flash-attn` 0.11 (same
functions and `FlashAttn` op) built on these kernels, for applications that already call it.

### Switches

Everything can be turned on and off at three levels: Cargo features (what is compiled), environment
variables (read on first use), and `candle_core::accel::update_config` / per call options (which win).

| Cargo feature (candle-core, candle-nn, candle-flash-attn) | default | effect when disabled |
| --- | --- | --- |
| `flash-attn` | on | the fused kernels are not built (`flash_attn.hip`, NVRTC sources, shaders); the attention APIs run the unfused `matmul` + `softmax` path |
| `turbo-quant` | on | the quantization kernels are not built (`turbo_quant.hip`, ...); the TurboQuant APIs run the CPU codec |

On ROCm `flash_attn.hip` adds about 10 s of hipcc time per GPU target to the build (set `HIP_ARCH`
to your GPU, or turn the feature off with `default-features = false` on `candle-core` /
`candle-nn`). On CUDA each attention kernel is compiled by NVRTC the first time it is used.

| variable | default | meaning |
| --- | --- | --- |
| `CANDLE_FLASH_ATTN` | `on` | master switch of the fused attention; `off` runs the unfused path everywhere |
| `CANDLE_FLASH_ATTN_VERSION` | `auto` | `1`, `2`, `3` (also `fa2`, `v3`, ...): force a tier instead of the per-GPU choice |
| `CANDLE_FLASH_ATTN_SPLIT_KV` | `auto` | split-KV / flash-decoding: `auto`, `off`, or a fixed number of splits (max 64) |
| `CANDLE_FLASH_ATTN_FP8_KV` | `on` | kernels read F8E4M3 K/V directly; `off` casts them to the query dtype first |
| `CANDLE_FLASH_ATTN_HADAMARD` | `on` | Hadamard incoherent processing (rotate Q and K) in `flash_attn_fp8` |
| `CANDLE_FLASH_ATTN_BACKENDS` | `all` | backends that may run fused kernels: `all`, `none`, or e.g. `rocm,vulkan` |
| `CANDLE_FLASH_ATTN_DEBUG` | `0` | log the chosen path and fallbacks |
| `CANDLE_TURBO_QUANT` | `on` | master switch; `off` makes `TurboKvCache` a plain (uncompressed) cache |
| `CANDLE_TURBO_QUANT_FORMAT` / `_K_FORMAT` / `_V_FORMAT` | `auto` | `turbo[1-4]`, `prod[2-5]`, `nvfp4`, `mxfp4`, `mxfp8`, `fp8` (`nvfp8`); `auto` picks per GPU |
| `CANDLE_TURBO_QUANT_BITS` / `_K_BITS` / `_V_BITS` | `4` | width used by `auto` and by `turbo` / `prod` without a number |
| `CANDLE_TURBO_QUANT_ROTATION` | `auto` | `hadamard` (power-of-two dims), `dense` (Haar random, as the reference), `none`; `auto` = Hadamard when possible |
| `CANDLE_TURBO_QUANT_QJL` | `off` | 1-bit QJL residual sketch on TurboQuant keys (the `prod` variant, see below) |
| `CANDLE_TURBO_QUANT_FUSED` | `on` | attention directly on the compressed cache; `off` dequantizes and runs flash attention |
| `CANDLE_TURBO_QUANT_SEED` | `42` | seed of the rotation and QJL matrices |
| `CANDLE_TURBO_QUANT_BACKENDS` | `all` | backends that may run the quantization kernels |
| `CANDLE_TURBO_QUANT_DEBUG` | `0` | log the chosen formats and fallbacks |

```rust
use candle_core::accel::{self, FlashAttnVersion, QuantFormat, SplitKv};

accel::update_config(|c| {
    c.flash_attn.version = Some(FlashAttnVersion::V2);
    c.flash_attn.split_kv = SplitKv::Off;
    c.turbo_quant.key_format = Some(QuantFormat::Nvfp4);
    c.turbo_quant.value_format = Some(QuantFormat::TurboMse { bits: 3 });
});
// per call: FlashAttnConfig::new(scale).version(FlashAttnVersion::V1).split_kv(SplitKv::Fixed(4)).fused(false)
```

A fused call that cannot run (kernels not built for the GPU, an unsupported shape, Metal / OpenCL
devices) logs the reason once and falls back to the unfused path, so enabling the features never
breaks a model.

### FlashAttention tiers

| tier | algorithm | picked for |
| --- | --- | --- |
| V1 | FlashAttention-1 class: one workgroup per query row, keys streamed through a small shared memory window with an online softmax | NVIDIA sm < 8.0, AMD gfx908 / RDNA1-2 / older, Vulkan devices with < 32 KiB shared memory |
| V2 | FlashAttention-2 class: Q tiles per workgroup, K/V tiles in shared memory, delayed rescaling, causal / window tile skipping | NVIDIA sm 8.x and 12.x, AMD gfx90a and RDNA3 / 3.5, Vulkan |
| V3 | FlashAttention-3 class: V2 with 64 KiB tiles, FP8 K/V with descale factors, Hadamard incoherent processing | NVIDIA sm 9.x - 11.x (Hopper, Blackwell DC), AMD gfx940-942 / gfx950 / RDNA4 |

Short queries (decoding, up to 4 tokens) use the row kernel of every tier with split-KV. The tier
choice follows Dao-AILab: FA2 needs Ampere+ (on ROCm, the CDNA2+ / RDNA3+ targets of its
composable-kernel backend), FA3 is the Hopper generation. The kernels are portable
reimplementations of these algorithms (shared memory tiles, no wgmma / TMA / MFMA / WMMA
intrinsics), one source for HIP and CUDA and a GLSL port for Vulkan; they compute in f32 with
f32, f16, bf16 queries and K/V in the query dtype or FP8.

### TurboQuant and microscaling formats

| format | per vector of 128 | element | scales |
| --- | --- | --- | --- |
| `turbo{b}` (MSE) | `b + 0.5` bits / value | Lloyd-Max codebook of the rotated coordinates, `b` = 1-4 | f32 norm |
| `prod{b}` | `b + 0.5` bits | codebook of `b - 1` bits + 1-bit QJL sketch of the residual (unbiased inner products) | norm, residual norm |
| `nvfp4` | 5 bits | E2M1 | E4M3 per 16 values + f32 per vector |
| `mxfp4` | 4.75 bits | E2M1 | E8M0 per 32 values (OCP MX) |
| `mxfp8` | 8.75 bits | E4M3 | E8M0 per 32 values (OCP MX) |
| `fp8` (`nvfp8`) | 8.5 bits | E4M3 | f32 per vector |

`auto` picks, for the 4-bit class, the GPU's native FP4 format (NVFP4 on NVIDIA Blackwell, MXFP4
on AMD MI350 / gfx950) and TurboQuant codebooks elsewhere; for the 8-bit class FP8 (Ada, Hopper,
MI300, RDNA4), MXFP8 (Blackwell, MI350) and FP8 elsewhere. All formats are decoded by the same
kernels on every GPU; the native choice keeps the data in the layout of the vendor block-scaled
GEMM libraries.

Attention error over a compressed cache (random Gaussian Q/K/V, 256 keys, head dim 128, relative
RMS error of the output, mean of 6 seeds; `cargo test -p candle-nn --lib turbo_quant::accuracy -- --ignored --nocapture`):

| keys / values | bits per value | error |
| --- | --- | --- |
| `fp8` / `fp8` | 8.5 | 0.036 |
| `nvfp4` / `nvfp4` (Hadamard) | 5 | 0.133 |
| `turbo4` / `turbo4` | 4.5 | 0.137 |
| `mxfp4` / `mxfp4` (Hadamard) | 4.75 | 0.162 |
| `prod4` / `turbo4` | 4.5 | 0.247 |
| `turbo3` / `turbo3` | 3.5 | 0.253 |

The QJL stage makes key inner products unbiased but adds variance, which the softmax amplifies:
at the same width the MSE codebook gives the lower attention error, so QJL is off by default
(`CANDLE_TURBO_QUANT_QJL=on` enables the reference's `prod` keys).

### Using it from an application (e.g. logis-center)

```toml
[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
candle-flash-attn = { version = "0.11.0", optional = true }

[features]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
rocm = ["candle-core/rocm", "candle-nn/rocm"]
vulkan = ["candle-core/vulkan", "candle-nn/vulkan"]
flash-attn = ["dep:candle-flash-attn"]   # works with cuda, rocm and vulkan

[patch.crates-io]
candle-core = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
candle-nn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
candle-flash-attn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
```

Existing `candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)` calls then run on CUDA, ROCm and
Vulkan. Pass `causal = true` only for causal masks; arbitrary additive masks are not a FlashAttention
input (use `flash_attn_windowed` / ALiBi, or keep the unfused path for them). KV caches already
stored as `F8E4M3` can be passed as `k` / `v` directly (FP8 K/V), and `TurboKvCache` replaces a
`Tensor::cat` cache with a compressed one.

### Verification

`cargo test -p candle-core --test accel_kernels_host` compiles the HIP / CUDA kernel sources as
C++ (every GPU lane a host thread, `__syncthreads` a barrier) and checks them against the Rust
reference; `--features vulkan --test vulkan_accel_tests` and `candle-nn --features vulkan --test
accel_vulkan` run the shaders (lavapipe works). The HIP and NVRTC sources are also compiled for
gfx90a / gfx942 / gfx1030 / gfx1100 / gfx1201 and sm_75 / sm_80 / sm_90 without spills.

## Layout

```
candle-rocm-vulkan/
├── candle-core/      # candle-core 0.11.0
│   ├── src/accel/            # switches, per-GPU dispatch, formats, TurboQuant math, CPU reference
│   ├── kernels/flash_attn_impl.h, turbo_quant_impl.h  # fused kernels (HIP + NVRTC CUDA source)
│   ├── src/rocm_backend/     # ROCm/HIP backend (kernels in kernels/*.hip, built by build.rs)
│   ├── src/vulkan_backend/   # Vulkan backend (mod.rs, shaders.rs, qgpu.rs = Q8 / VRAM weights)
│   └── src/opencl_backend/   # OpenCL backend
├── candle-nn/        # candle-nn 0.11.0 + rocm_fwd / vulkan_fwd for the fused ops,
│                     # flash_attn.rs (fused attention), turbo_quant.rs (quantizer, KV cache)
├── candle-flash-attn/ # drop-in replacement of the crates.io candle-flash-attn 0.11
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
cargo test -p candle-core --test accel_kernels_host            # fused kernels, host emulation
cargo test -p candle-core --features vulkan --test vulkan_accel_tests
cargo test -p candle-nn   --features vulkan --test accel_vulkan
cargo test -p candle-nn   --test accel_config                  # the on/off switches
cargo test -p candle-flash-attn
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
