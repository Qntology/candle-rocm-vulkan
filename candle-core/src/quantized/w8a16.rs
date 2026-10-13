//! Exact weight-only quantized mat-vec products and direct dequantization on the CUDA backend.
//!
//! The kernels live in `w8a16.cu` and are compiled with NVRTC for the GPU's compute capability
//! the first time a (weight type, activation dtype, rows) combination is used, one small module
//! per kernel. They back [`super::QTensor::forward_exact`] and [`super::QTensor::dequantize_to`]:
//!
//! * `matvec`: `y = x · dequant(W)ᵀ` for at most [`MV_MAX_ROWS`] input rows. The weights are
//!   dequantized in f32 inside the kernel (bit exact with the CPU `to_float`), the f32 / f16 /
//!   bf16 activations are read directly and widened exactly, products are accumulated in f32 and
//!   the output is written in the activation dtype (one rounding). No activation quantization.
//! * `dequantize`: Q8_0 / Q4_0 / Q4K / Q5K / Q6K blocks or F32 / F16 / BF16 elements straight to
//!   an f32 / f16 / bf16 buffer, one rounding and no f32 intermediate.
//! * `gemm_tiles`: the prefill path, `x · Wᵀ` with `W` dequantized in row tiles into a transient
//!   buffer and multiplied by cuBLAS (f32 accumulation and f32 split-k reductions, f16 / bf16
//!   results rounded once).

use super::GgmlDType;
use crate::cuda_backend::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use crate::{bail, DType, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};

const SOURCE: &str = include_str!("w8a16.cu");

/// Largest number of input rows (product of the leading dims) handled by [`matvec`].
pub const MV_MAX_ROWS: usize = 8;

/// Launch shape of the mat-vec kernel: `warps_per_row` warps split the k range of one weight
/// row, a thread block covers `rows_per_block` weight rows, every lane unrolls its task loop
/// `unroll` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MvConfig {
    pub warps_per_row: usize,
    pub rows_per_block: usize,
    pub unroll: usize,
}

/// `CANDLE_W8A16_MV=<warps_per_row>x<rows_per_block>[x<unroll>]` overrides the tuned launch
/// shape (benchmarking aid; invalid values are ignored).
fn env_mv_config() -> Option<MvConfig> {
    static CFG: std::sync::OnceLock<Option<MvConfig>> = std::sync::OnceLock::new();
    *CFG.get_or_init(|| {
        let s = std::env::var("CANDLE_W8A16_MV").ok()?;
        let v: Vec<usize> = s
            .split('x')
            .map(|p| p.trim().parse::<usize>().ok())
            .collect::<Option<_>>()?;
        let (w, r, u) = match v[..] {
            [w, r] => (w, r, 2),
            [w, r, u] => (w, r, u),
            _ => return None,
        };
        let pow2 = |v: usize| matches!(v, 1 | 2 | 4 | 8);
        let ok = pow2(w) && pow2(r) && pow2(u) && w * r <= 16;
        ok.then_some(MvConfig {
            warps_per_row: w,
            rows_per_block: r,
            unroll: u,
        })
    })
}

/// Tuned launch shape (RTX 3050 Laptop / sm_86, `w8a16_bench_cuda` in
/// `tests/quantized_tests.rs`). One decode row of the Qwen3-0.6B tied lm_head (151936 x 1024
/// Q8_0, 165 MB) takes ~893 us, i.e. ~185 GB/s, the memory bandwidth of the part; the layer
/// shapes (1-3 MB) are bound by the per-launch cost. Shapes from 1x4 to 2x4 (warps per row x
/// rows per block) and unroll 1-4 measure within noise; 4 or more warps per row lose up to 2x
/// on rows of at most 3072 weights, so long rows only get more warps from 4 x 128 lane tasks.
pub fn mv_config(wtype: GgmlDType, k: usize, _n: usize) -> MvConfig {
    if let Some(c) = env_mv_config() {
        return c;
    }
    let tasks_per_row = match decoder(wtype) {
        Some(d) => k / d.qk * d.tasks,
        None => k / 8,
    };
    // Keep at least ~4 lane tasks per lane in flight; split long rows over several warps.
    let warps_per_row = if tasks_per_row >= 32 * 4 * 4 {
        4
    } else if tasks_per_row >= 32 * 4 * 2 {
        2
    } else {
        1
    };
    MvConfig {
        warps_per_row,
        rows_per_block: 4 / warps_per_row.min(4),
        unroll: 2,
    }
}

#[derive(Clone, Copy)]
struct Decoder {
    sfx: &'static str,
    ty: &'static str,
    qk: usize,
    tasks: usize,
}

fn decoder(dtype: GgmlDType) -> Option<Decoder> {
    let (sfx, ty, qk, tasks) = match dtype {
        GgmlDType::Q8_0 => ("q8_0", "DecQ8_0", 32, 4),
        GgmlDType::Q4_0 => ("q4_0", "DecQ4_0", 32, 2),
        GgmlDType::Q4K => ("q4k", "DecQ4K", 256, 16),
        GgmlDType::Q5K => ("q5k", "DecQ5K", 256, 16),
        GgmlDType::Q6K => ("q6k", "DecQ6K", 256, 8),
        _ => return None,
    };
    Some(Decoder { sfx, ty, qk, tasks })
}

/// Weight types with an exact mat-vec kernel.
pub fn mv_supported(dtype: GgmlDType) -> bool {
    decoder(dtype).is_some()
}

/// Weight types [`dequantize`] converts directly (block types with a decoder and the float
/// element types).
pub fn dq_supported(dtype: GgmlDType) -> bool {
    decoder(dtype).is_some() || matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

fn elem_ctype(dtype: DType) -> Result<(&'static str, &'static str)> {
    match dtype {
        DType::F32 => Ok(("f32", "float")),
        DType::F16 => Ok(("f16", "w_f16")),
        DType::BF16 => Ok(("bf16", "w_bf16")),
        d => bail!("w8a16: unsupported dtype {d:?} (expected f32, f16 or bf16)"),
    }
}

fn ggml_float_ctype(dtype: GgmlDType) -> Option<(&'static str, &'static str, DType)> {
    match dtype {
        GgmlDType::F32 => Some(("f32", "float", DType::F32)),
        GgmlDType::F16 => Some(("f16", "w_f16", DType::F16)),
        GgmlDType::BF16 => Some(("bf16", "w_bf16", DType::BF16)),
        _ => None,
    }
}

fn arch(dev: &CudaDevice) -> &'static str {
    use cudarc::driver::sys::CUdevice_attribute as A;
    use std::collections::HashMap;
    use std::sync::Mutex;
    static ARCHS: Mutex<Option<HashMap<(i32, i32), &'static str>>> = Mutex::new(None);
    let major = dev
        .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(7);
    let minor = dev
        .device_attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(5);
    let mut g = ARCHS.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(HashMap::new)
        .entry((major, minor))
        .or_insert_with(|| Box::leak(format!("compute_{major}{minor}").into_boxed_str()))
}

/// Kernel `name` of the module built from `w8a16.cu` + `inst` (compiled on first use).
fn func(dev: &CudaDevice, name: &str, inst: String) -> Result<CudaFunction> {
    let module = format!("candle_{name}");
    let f = dev.get_or_compile_func(name, &module, || {
        let src = format!("{SOURCE}\n{inst}\n");
        let opts = |arch: Option<&'static str>| cudarc::nvrtc::CompileOptions {
            arch,
            options: vec!["--std=c++17".into()],
            ..Default::default()
        };
        match cudarc::nvrtc::safe::compile_ptx_with_opts(&src, opts(Some(arch(dev)))) {
            Ok(p) => Ok(p),
            // An NVRTC older than the GPU: let the driver JIT the default target.
            Err(_) => cudarc::nvrtc::safe::compile_ptx_with_opts(&src, opts(None)).w(),
        }
    })?;
    Ok(f.into_cuda_function())
}

fn grid_1d(work: usize, threads: usize) -> u32 {
    work.div_ceil(threads).clamp(1, 1 << 16) as u32
}

/// `y[m, :] = dequant(W) · x[m, :]` for the `rows` contiguous input rows of `x` starting at
/// element `x_off`. `w` holds an `n × k` matrix of `wtype` blocks; the output is a fresh
/// `rows × n` buffer of the dtype of `x`. Requires `rows <= MV_MAX_ROWS`, `k % block == 0`
/// and an `x_off` whose byte offset is a multiple of 16 (the kernel loads 16 byte vectors).
#[allow(clippy::too_many_arguments)]
pub fn matvec(
    dev: &CudaDevice,
    w: &CudaSlice<u8>,
    wtype: GgmlDType,
    n: usize,
    k: usize,
    x: &CudaStorage,
    x_off: usize,
    rows: usize,
) -> Result<CudaStorage> {
    use crate::backend::BackendStorage;
    let Some(d) = decoder(wtype) else {
        bail!("w8a16 matvec: unsupported weight type {wtype:?}")
    };
    if rows == 0 || rows > MV_MAX_ROWS {
        bail!("w8a16 matvec: {rows} rows (expected 1..={MV_MAX_ROWS})")
    }
    if k == 0 || !k.is_multiple_of(d.qk) || n == 0 || n > i32::MAX as usize {
        bail!("w8a16 matvec: unsupported shape n={n} k={k} for {wtype:?}")
    }
    let (xs, xc) = elem_ctype(x.dtype())?;
    if (x_off * x.dtype().size_in_bytes()) % 16 != 0 {
        bail!("w8a16 matvec: input offset {x_off} is not 16 byte aligned")
    }
    let row_bytes = (k / d.qk * wtype.type_size()) as i64;
    let cfg = mv_config(wtype, k, n);
    let (wpr, rpb, unr) = (cfg.warps_per_row, cfg.rows_per_block, cfg.unroll);
    let name = format!("w8a16_mv_{}_{xs}_m{rows}_w{wpr}_r{rpb}_u{unr}", d.sfx);
    let inst = format!(
        "W8A16_MV_KERNEL({name}, {}, {xc}, {rows}, {wpr}, {rpb}, {unr})",
        d.ty
    );
    let f = func(dev, &name, inst)?;
    let lc = LaunchConfig {
        grid_dim: (n.div_ceil(rpb) as u32, 1, 1),
        block_dim: (32, wpr as u32, rpb as u32),
        shared_mem_bytes: 0,
    };
    let (k_i, n_i) = (k as i32, n as i32);
    macro_rules! launch {
        ($s:expr, $t:ty) => {{
            let xv = $s.slice(x_off..x_off + rows * k);
            // SAFETY: the kernel writes every one of the rows * n outputs.
            let mut out = unsafe { dev.alloc::<$t>(rows * n)? };
            let stream = dev.cuda_stream();
            let mut b = stream.launch_builder(&f);
            b.arg(w)
                .arg(&xv)
                .arg(&mut out)
                .arg(&k_i)
                .arg(&n_i)
                .arg(&row_bytes);
            // SAFETY: the arguments match W8A16_MV_KERNEL.
            unsafe { b.launch(lc) }.w()?;
            Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
        }};
    }
    match &x.slice {
        CudaStorageSlice::F32(s) => launch!(s, f32),
        CudaStorageSlice::F16(s) => launch!(s, half::f16),
        CudaStorageSlice::BF16(s) => launch!(s, half::bf16),
        _ => bail!("w8a16 matvec: unsupported input dtype {:?}", x.dtype()),
    }
}

/// Dequantizes `elems` weights of type `wtype` starting at the (block aligned) start of `w`
/// into a fresh buffer of `out` dtype (f32 / f16 / bf16), on the device.
pub fn dequantize(
    dev: &CudaDevice,
    w: &CudaView<u8>,
    wtype: GgmlDType,
    elems: usize,
    out: DType,
) -> Result<CudaStorage> {
    let (osfx, octype) = elem_ctype(out)?;
    macro_rules! alloc_out {
        ($body:ident) => {
            match out {
                DType::F32 => $body!(f32),
                DType::F16 => $body!(half::f16),
                _ => $body!(half::bf16),
            }
        };
    }
    if let Some(d) = decoder(wtype) {
        if !elems.is_multiple_of(d.qk) {
            bail!(
                "w8a16 dequantize: {elems} elements is not a multiple of {}",
                d.qk
            )
        }
        let nblocks = elems / d.qk;
        if w.len() < nblocks * wtype.type_size() {
            bail!(
                "w8a16 dequantize: {} bytes for {elems} {wtype:?} weights",
                w.len()
            )
        }
        let name = format!("w8a16_dq_{}_{osfx}", d.sfx);
        let inst = format!("W8A16_DQ_KERNEL({name}, {}, {octype})", d.ty);
        let f = func(dev, &name, inst)?;
        let threads = 256;
        let lc = LaunchConfig {
            grid_dim: (grid_1d(nblocks * d.tasks, threads), 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nb = nblocks as i64;
        macro_rules! run {
            ($t:ty) => {{
                // SAFETY: every output element is written by the kernel.
                let mut y = unsafe { dev.alloc::<$t>(elems.max(1))? };
                if nblocks > 0 {
                    let stream = dev.cuda_stream();
                    let mut b = stream.launch_builder(&f);
                    b.arg(w).arg(&mut y).arg(&nb);
                    // SAFETY: the arguments match W8A16_DQ_KERNEL.
                    unsafe { b.launch(lc) }.w()?;
                }
                Ok(CudaStorage::wrap_cuda_slice(y, dev.clone()))
            }};
        }
        return alloc_out!(run);
    }
    let Some((isfx, ictype, idtype)) = ggml_float_ctype(wtype) else {
        bail!("w8a16 dequantize: unsupported weight type {wtype:?}")
    };
    if w.len() < elems * idtype.size_in_bytes() {
        bail!(
            "w8a16 dequantize: {} bytes for {elems} {wtype:?} elements",
            w.len()
        )
    }
    if idtype == out {
        macro_rules! copy {
            ($t:ty) => {{
                // SAFETY: the bytes hold `elems` values of this type (checked above).
                let src = unsafe { w.transmute::<$t>(elems) }
                    .ok_or_else(|| crate::Error::Msg("w8a16 dequantize: bad view".into()))?;
                // SAFETY: filled by the copy below.
                let mut y = unsafe { dev.alloc::<$t>(elems.max(1))? };
                if elems > 0 {
                    dev.memcpy_dtod(&src, &mut y.slice_mut(..elems))?;
                }
                Ok(CudaStorage::wrap_cuda_slice(y, dev.clone()))
            }};
        }
        return alloc_out!(copy);
    }
    let name = format!("w8a16_cvt_{isfx}_{osfx}");
    let inst = format!("W8A16_CVT_KERNEL({name}, {ictype}, {octype})");
    let f = func(dev, &name, inst)?;
    let threads = 256;
    let lc = LaunchConfig {
        grid_dim: (grid_1d(elems, threads), 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let ne = elems as i64;
    macro_rules! cvt {
        ($t:ty) => {{
            // SAFETY: every output element is written by the kernel.
            let mut y = unsafe { dev.alloc::<$t>(elems.max(1))? };
            if elems > 0 {
                let stream = dev.cuda_stream();
                let mut b = stream.launch_builder(&f);
                b.arg(w).arg(&mut y).arg(&ne);
                // SAFETY: the arguments match W8A16_CVT_KERNEL.
                unsafe { b.launch(lc) }.w()?;
            }
            Ok(CudaStorage::wrap_cuda_slice(y, dev.clone()))
        }};
    }
    alloc_out!(cvt)
}

/// Byte budget of one transient dequantization tile (and of one f32 GEMM tile):
/// `CANDLE_W8A16_DEQUANT_TILE_MB` (default 32) MiB.
fn dequant_tile_bytes() -> usize {
    static TILE_BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TILE_BYTES.get_or_init(|| {
        std::env::var("CANDLE_W8A16_DEQUANT_TILE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(32)
            << 20
    })
}

/// Rows of an `(n, k)` weight dequantized at once by the GEMM path of `forward_exact`
/// (`CANDLE_W8A16_DEQUANT_TILE_MB`, default 32 MiB, of `dtype` values), at least one row.
pub fn dequant_tile_rows(k: usize, dtype: DType) -> usize {
    (dequant_tile_bytes() / (k.max(1) * dtype.size_in_bytes())).max(1)
}

/// How [`gemm_tiles`] issues f16 / bf16 GEMMs (`CANDLE_W8A16_PREFILL_GEMM`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillGemm {
    /// Default (`exact`): f16 / bf16 output, f32 compute, and the cuBLAS math mode
    /// `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` for the call, so split-k partial sums
    /// are reduced in f32 and every output is the f32 dot product rounded once.
    Exact,
    /// `staged`: each tile GEMM writes an f32 buffer that a small kernel rounds into the
    /// output (same numerics as `Exact`, one extra buffer and kernel per tile).
    Staged,
    /// `matmul`: the exact cuBLAS call `Tensor::matmul` issues (bit identical to a dense layer
    /// holding the dequantized weight, including the reduced precision split-k reductions
    /// cuBLAS may pick). Only for weights that fit one tile; several tiles use `Exact`.
    Matmul,
}

/// The f16 / bf16 prefill GEMM mode from `CANDLE_W8A16_PREFILL_GEMM` (`exact`, `staged` or
/// `matmul`; default `exact`).
pub fn prefill_gemm_mode() -> PrefillGemm {
    static MODE: std::sync::OnceLock<PrefillGemm> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("CANDLE_W8A16_PREFILL_GEMM")
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Ok("matmul") => PrefillGemm::Matmul,
            Ok("staged") => PrefillGemm::Staged,
            _ => PrefillGemm::Exact,
        }
    })
}

/// Runs `f` (which issues cuBLAS calls on the device's handle) with
/// `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` set on the handle, then restores the
/// previous math mode. candle never changes the math mode itself (it is `CUBLAS_DEFAULT_MATH`);
/// a mode that already forbids reduced precision reductions (the flag, or pedantic math) or an
/// unknown value is left untouched. Calls of other threads that land in the window only get
/// the more precise reduction.
fn with_f32_reductions<R>(dev: &CudaDevice, f: impl FnOnce() -> Result<R>) -> Result<R> {
    use cudarc::cublas::sys::{self, cublasMath_t as M};
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let handle = *dev.blas.handle();
    let mut prev = 0u32;
    // SAFETY: valid handle; the mode is written as a C int into `prev` (read as u32, never
    // materialized as a possibly invalid enum value).
    unsafe { sys::cublasGetMathMode(handle, &mut prev as *mut u32 as *mut M) }
        .result()
        .w()?;
    let restore = match prev {
        0 => Some(M::CUBLAS_DEFAULT_MATH),
        1 => Some(M::CUBLAS_TENSOR_OP_MATH),
        // TF32 only concerns f32 data, not the f16 / bf16 GEMMs issued here.
        3 => Some(M::CUBLAS_TF32_TENSOR_OP_MATH),
        _ => None,
    };
    if restore.is_some() {
        // SAFETY: valid handle and mode.
        unsafe {
            sys::cublasSetMathMode(handle, M::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
        }
        .result()
        .w()?;
    }
    let r = f();
    if let Some(mode) = restore {
        // SAFETY: valid handle; `mode` is the mode read above.
        let _ = unsafe { sys::cublasSetMathMode(handle, mode) };
    }
    r
}

/// `y = x · Wᵀ` for the `rows × k` contiguous input `x` starting at element `x_off` and an
/// `n × k` weight produced by `tile(row0, nrows)` as dense row tiles of at most `tile_rows`
/// rows in the dtype of `x` (f32 / f16 / bf16). A tile is freed (stream ordered) as soon as its
/// GEMM is queued, so at most one tile is alive. Each tile GEMM writes its columns
/// `row0..row0 + nrows` of the output directly (ldc = n) except in the `staged` mode.
///
/// * f16 / bf16, default ([`PrefillGemm::Exact`]): f32 compute with
///   `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, so every output is the f32 dot product
///   rounded once. A plain f16 / bf16 cuBLAS GEMM (what `Tensor::matmul` issues) can pick
///   split-k kernels that reduce partial sums in the output type: on an RTX 3050 the Qwen3
///   `down` projection (1024 x 3072) at 256 rows and skinny shapes (256 x 1024 at 9 rows) come
///   out with errors of 30-130 output ulps that way.
/// * f16 / bf16, `CANDLE_W8A16_PREFILL_GEMM=staged`: f32 output tiles rounded by a small kernel
///   (same numerics as the default, extra buffer and kernel).
/// * f16 / bf16, `CANDLE_W8A16_PREFILL_GEMM=matmul` and a single tile: the exact cuBLAS call
///   `Tensor::matmul` issues for `x.matmul(&w.t())`, i.e. bit identical to the dense
///   dequantize-to-bf16 layer (including its split-k behaviour). Several tiles use the default.
/// * f32: f32 compute (no TF32), except in the `matmul` mode, which mirrors `Tensor::matmul`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_tiles(
    dev: &CudaDevice,
    x: &CudaStorage,
    x_off: usize,
    rows: usize,
    n: usize,
    k: usize,
    tile_rows: usize,
    mut tile: impl FnMut(usize, usize) -> Result<CudaStorage>,
) -> Result<CudaStorage> {
    use crate::backend::BackendStorage;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dtype = x.dtype();
    if rows == 0 || n == 0 || k == 0 || tile_rows == 0 {
        bail!("w8a16 gemm: empty problem rows={rows} n={n} k={k} tile={tile_rows}")
    }
    let max = i32::MAX as usize;
    if rows > max || n > max || k > max {
        bail!("w8a16 gemm: shape rows={rows} n={n} k={k} too large")
    }
    let kind = match prefill_gemm_mode() {
        PrefillGemm::Matmul if tile_rows >= n => GemmKind::Matmul,
        PrefillGemm::Staged if dtype != DType::F32 => GemmKind::F32Out,
        _ => GemmKind::Exact,
    };
    // Staged: f32 GEMM output tiles, bounded by the same byte budget.
    let tile_rows = if kind == GemmKind::F32Out {
        tile_rows.min((dequant_tile_bytes() / (rows * 4)).max(1))
    } else {
        tile_rows
    };
    let cvt = if kind == GemmKind::F32Out {
        let (osfx, octype) = elem_ctype(dtype)?;
        let name = format!("w8a16_cvt2d_f32_{osfx}");
        Some(func(
            dev,
            &name,
            format!("W8A16_CVT2D_KERNEL({name}, {octype})"),
        )?)
    } else {
        None
    };
    let stream = dev.cuda_stream();
    macro_rules! run {
        ($s:expr, $t:ty, $variant:ident) => {{
            let xv = $s.slice(x_off..x_off + rows * k);
            // SAFETY: every output column is written by exactly one tile below.
            let mut out = unsafe { dev.alloc::<$t>(rows * n)? };
            let mut n0 = 0;
            while n0 < n {
                let tn = tile_rows.min(n - n0);
                let w = tile(n0, tn)?;
                let CudaStorageSlice::$variant(wv) = &w.slice else {
                    bail!(
                        "w8a16 gemm: weight tile is {:?}, expected {dtype:?}",
                        w.dtype()
                    )
                };
                if wv.len() < tn * k {
                    bail!(
                        "w8a16 gemm: weight tile has {} values, expected {}",
                        wv.len(),
                        tn * k
                    )
                }
                if let Some(cvt) = &cvt {
                    // SAFETY: fully written by the GEMM below.
                    let mut acc = unsafe { dev.alloc::<f32>(rows * tn)? };
                    {
                        let (a, _ga) = wv.device_ptr(&stream);
                        let (b, _gb) = xv.device_ptr(&stream);
                        let (c, _gc) = acc.device_ptr_mut(&stream);
                        // SAFETY: a = tn × k tile, b = rows × k input, c = rows × tn f32 buffer.
                        unsafe { gemm_ex(dev, dtype, kind, tn, rows, k, a, b, c, tn) }?;
                    }
                    let mut dst = out.slice_mut(n0..);
                    let (rows_i, tn_i, ld) = (rows as i32, tn as i32, n as i64);
                    let lc = LaunchConfig {
                        grid_dim: (grid_1d(rows * tn, 256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = stream.launch_builder(cvt);
                    b.arg(&acc).arg(&mut dst).arg(&rows_i).arg(&tn_i).arg(&ld);
                    // SAFETY: the arguments match W8A16_CVT2D_KERNEL; `dst` holds rows rows of
                    // n values from column n0 (the last row ends at column n0 + tn <= n).
                    unsafe { b.launch(lc) }.w()?;
                } else {
                    let (a, _ga) = wv.device_ptr(&stream);
                    let (b, _gb) = xv.device_ptr(&stream);
                    let (c, _gc) = out.device_ptr_mut(&stream);
                    let c = c + (n0 * std::mem::size_of::<$t>()) as u64;
                    // SAFETY: a = tn × k weight tile, b = rows × k input, c = column n0 of the
                    // rows × n output (ldc = n); all three live until the call is queued.
                    let call = || unsafe { gemm_ex(dev, dtype, kind, tn, rows, k, a, b, c, n) };
                    if kind == GemmKind::Exact && dtype != DType::F32 {
                        with_f32_reductions(dev, call)?;
                    } else {
                        call()?;
                    }
                }
                drop(w);
                n0 += tn;
            }
            Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
        }};
    }
    match &x.slice {
        CudaStorageSlice::F32(s) => run!(s, f32, F32),
        CudaStorageSlice::F16(s) => run!(s, half::f16, F16),
        CudaStorageSlice::BF16(s) => run!(s, half::bf16, BF16),
        _ => bail!("w8a16 gemm: unsupported input dtype {dtype:?}"),
    }
}

/// cuBLAS configuration of one [`gemm_ex`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmKind {
    /// C has the input dtype; issued exactly like `cuda_backend::gemm_strided_batched_*` for a
    /// 2D `Tensor::matmul` (batch 1, same strides, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, f32 compute
    /// unless the matching `gemm_reduced_precision_*` flag is set).
    Matmul,
    /// C has the input dtype, f32 compute regardless of the reduced precision flags (the
    /// caller also forbids reduced precision split-k reductions, see [`with_f32_reductions`]).
    Exact,
    /// C is f32, f32 compute.
    F32Out,
}

/// Column-major `C[tn × rows] = A[k × tn]ᵀ · B[k × rows]` (i.e. row-major
/// `y[rows × tn] = x[rows × k] · w[tn × k]ᵀ`) with leading dimension `ldc`; A and B have type
/// `dtype`, C and the compute type follow `kind`.
///
/// # Safety
/// `a`, `b` and `c` must be device pointers covering the described matrices.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_ex(
    dev: &CudaDevice,
    dtype: DType,
    kind: GemmKind,
    tn: usize,
    rows: usize,
    k: usize,
    a: u64,
    b: u64,
    c: u64,
    ldc: usize,
) -> Result<()> {
    use crate::cuda_backend::{
        gemm_reduced_precision_bf16, gemm_reduced_precision_f16, gemm_reduced_precision_f32,
    };
    use cudarc::cublas::sys;
    use std::ffi::c_void;
    let (one_f32, zero_f32) = (1f32, 0f32);
    let (one_f16, zero_f16) = (half::f16::ONE, half::f16::ZERO);
    let f32s = (
        &one_f32 as *const f32 as *const c_void,
        &zero_f32 as *const f32 as *const c_void,
    );
    let c32 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let mirror = kind == GemmKind::Matmul;
    let (data_type, compute, (alpha, beta)) = match dtype {
        DType::F32 => (
            sys::cudaDataType_t::CUDA_R_32F,
            if mirror && gemm_reduced_precision_f32() {
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
            } else {
                c32
            },
            f32s,
        ),
        DType::F16 if mirror && gemm_reduced_precision_f16() => (
            sys::cudaDataType_t::CUDA_R_16F,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            (
                &one_f16 as *const half::f16 as *const c_void,
                &zero_f16 as *const half::f16 as *const c_void,
            ),
        ),
        DType::F16 => (sys::cudaDataType_t::CUDA_R_16F, c32, f32s),
        DType::BF16 => (
            sys::cudaDataType_t::CUDA_R_16BF,
            if mirror && gemm_reduced_precision_bf16() {
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
            } else {
                c32
            },
            f32s,
        ),
        d => bail!("w8a16 gemm: unsupported dtype {d:?}"),
    };
    let c_type = if kind == GemmKind::F32Out {
        sys::cudaDataType_t::CUDA_R_32F
    } else {
        data_type
    };
    let (tn_i, rows_i, k_i) = (tn as i32, rows as i32, k as i32);
    cudarc::cublas::result::gemm_strided_batched_ex(
        *dev.blas.handle(),
        sys::cublasOperation_t::CUBLAS_OP_T,
        sys::cublasOperation_t::CUBLAS_OP_N,
        tn_i,
        rows_i,
        k_i,
        alpha,
        a as *const _,
        data_type,
        k_i,
        (tn * k) as i64,
        b as *const _,
        data_type,
        k_i,
        (rows * k) as i64,
        beta,
        c as *mut _,
        c_type,
        ldc as i32,
        (rows * tn) as i64,
        1,
        compute,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    .w()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instantiations_parse() {
        // Every instantiation macro is spelled as the Rust side emits it.
        for m in [
            "W8A16_MV_KERNEL(",
            "W8A16_DQ_KERNEL(",
            "W8A16_CVT_KERNEL(",
            "W8A16_CVT2D_KERNEL(",
        ] {
            assert!(SOURCE.contains(&format!("#define {m}")), "{m}");
        }
        for t in [
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let d = decoder(t).unwrap();
            assert_eq!(d.qk, t.block_size());
            assert!(SOURCE.contains(&format!("struct {} {{", d.ty)));
            assert!(SOURCE.contains(&format!("BYTES = {},", t.type_size())));
        }
    }
}
