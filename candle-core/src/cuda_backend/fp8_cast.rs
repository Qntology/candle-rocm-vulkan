//! Software OCP E4M3FN (`F8E4M3`) cast kernels on CUDA (`Q1-F06` / `Q1-O06`).
//!
//! `candle-kernels`'s precompiled `cast.cu` requires `__CUDA_ARCH__ >= 800` for FP8 and omits
//! several dtype pairs (e.g. `u32`/`i64` <-> `f8_e4m3`). This NVRTC module provides pure
//! software, headerless, bit-exact OCP E4M3FN conversions (matching CPU `float8::F8E4M3` and
//! ROCm `kernels/cast.hip` across all 256 codes, subnormals, ties-to-even, 448 saturation, ±Inf,
//! and NaN) on every CUDA architecture (`sm_70`..`sm_120`).

use super::device::{CudaDevice, CudaFunc};
use crate::Result;

const MODULE_NAME: &str = "candle_fp8_cast";

pub(crate) const FP8_CAST_CUDA_SOURCE: &str = r#"
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

__device__ __forceinline__ bool is_contig(size_t num_dims, const size_t *dims, const size_t *strides) {
    size_t acc = 1;
    for (size_t d = 0; d < num_dims; ++d) {
        size_t i = num_dims - 1 - d;
        if (dims[i] > 1 && strides[i] != acc) return false;
        acc *= dims[i];
    }
    return true;
}

__device__ __forceinline__ size_t strided_idx(
    size_t idx, size_t num_dims, const size_t *dims, const size_t *strides
) {
    size_t out = 0;
    for (size_t d = 0; d < num_dims; ++d) {
        size_t i = num_dims - 1 - d;
        out += (idx % dims[i]) * strides[i];
        idx /= dims[i];
    }
    return out;
}

// Exact OCP E4M3FN decode (all 256 codes, preserves -0.0, 0x7f/0xff -> NaN).
__device__ __forceinline__ float e4m3_to_f32(u8 c) {
    const u32 mag = (u32)(c & 0x7fu);
    const float nrm = __uint_as_float((mag << 20) + (120u << 23));
    const float sub = (float)mag * 0.001953125f;
    float v = mag >= 8u ? nrm : sub;
    v = (c & 0x80u) ? -v : v;
    return mag == 0x7fu ? __uint_as_float(0x7fc00000u) : v;
}

// Exact OCP E4M3FN encode from double (ties-to-even, 448 saturation, subnormals, NaN -> 0x7f).
// Matches `float8::F8E4M3` and ROCm `kernels/common.h::double_to_f8e4m3` bit-for-bit.
__device__ __forceinline__ u8 double_to_e4m3(double x) {
    const u64 xbits = (u64)__double_as_longlong(x);
    const u64 half_ulp = 1ull << 48;
    const u8 sign = (u8)((xbits >> 63) << 7);
    u16 e16 = (u16)(((u16)(xbits >> 52)) & 0x7FFu);
    e16 = (u16)(e16 - 1023u);
    e16 = (u16)(e16 + 7u);
    const u8 exp = (u8)e16;
    const u8 mantissa = (u8)((xbits >> 49) & 0x7u);
    const u64 absx = xbits & 0x7FFFFFFFFFFFFFFFull;
    u8 res;
    if (absx <= 0x3F50000000000000ull) {
        res = 0;
    } else if (absx > 0x7FF0000000000000ull) {
        res = 0x7F;
    } else if (absx > 0x407D000000000000ull) {
        res = 0x7E;
    } else if (absx >= 0x3F90000000000000ull) {
        res = (u8)((u8)(exp << 3) | mantissa);
        const u64 round = xbits & ((half_ulp << 1) - 1);
        if ((round > half_ulp) || ((round == half_ulp) && (mantissa & 1u))) {
            res = (u8)(res + 1);
        }
    } else {
        const u8 shift = (u8)(1u - exp);
        const u8 m = (u8)(mantissa | 0x8u);
        res = (u8)(m >> shift);
        const u64 round = (xbits | (1ull << 52)) & ((half_ulp << (shift + 1)) - 1);
        const u64 half = half_ulp << shift;
        if ((round > half) || ((round == half) && (res & 1u))) {
            res = (u8)(res + 1);
        }
    }
    return (u8)(res | sign);
}

__device__ __forceinline__ float f16_to_f32(u16 h) {
    const u32 s = (u32)(h & 0x8000u) << 16;
    const u32 e = (u32)(h >> 10) & 0x1fu;
    const u32 m = (u32)(h & 0x03ffu);
    if (e == 0u) {
        const float v = (float)m * 5.960464477539063e-8f; // 2^-24
        return s ? -v : v;
    }
    if (e == 31u) {
        return __uint_as_float(s | 0x7f800000u | (m << 13));
    }
    return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}

__device__ __forceinline__ u16 f32_to_f16(float f) {
    const u32 x = __float_as_uint(f);
    const u32 s = (x >> 16) & 0x8000u;
    const u32 ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {
        if (ax == 0x7f800000u) return (u16)(s | 0x7c00u);
        return (u16)(s | 0x7e00u | ((ax >> 13) & 0x01ffu));
    }
    if (ax >= 0x47800000u) return (u16)(s | 0x7c00u);
    if (ax < 0x38800000u) {
        if (ax < 0x33000000u) return (u16)s;
        const u32 e = ax >> 23;
        const u32 shift = 113u - e;
        const u32 mant = (ax & 0x007fffffu) | 0x00800000u;
        const u32 half_b = 1u << (shift + 12u);
        const u32 mask = (half_b << 1) - 1u;
        u32 r = mant >> (shift + 13u);
        const u32 rem = mant & mask;
        if (rem > half_b || (rem == half_b && (r & 1u))) r += 1u;
        return (u16)(s | r);
    }
    const u32 r = ax - 0x38000000u + 0x0fffu + ((ax >> 13) & 1u);
    return (u16)(s | (r >> 13));
}

__device__ __forceinline__ float bf16_to_f32(u16 b) {
    return __uint_as_float((u32)b << 16);
}

__device__ __forceinline__ u16 f32_to_bf16(float f) {
    const u32 x = __float_as_uint(f);
    if ((x & 0x7fffffffu) > 0x7f800000u) return (u16)((x >> 16) | 0x0040u);
    const u32 r = x + 0x7fffu + ((x >> 16) & 1u);
    return (u16)(r >> 16);
}

template <typename T> struct Fp8Cv;
template <> struct Fp8Cv<u8> {
    static __device__ __forceinline__ double to_d(u8 v) { return (double)v; }
    static __device__ __forceinline__ u8 from_f(float v) {
        if (!(v > 0.0f)) return 0;
        if (v >= 255.0f) return 255;
        return (u8)v;
    }
};
template <> struct Fp8Cv<u32> {
    static __device__ __forceinline__ double to_d(u32 v) { return (double)v; }
    static __device__ __forceinline__ u32 from_f(float v) {
        if (!(v > 0.0f)) return 0u;
        if (v >= 4294967295.0f) return 0xffffffffu;
        return (u32)v;
    }
};
template <> struct Fp8Cv<i64> {
    static __device__ __forceinline__ double to_d(i64 v) { return (double)v; }
    static __device__ __forceinline__ i64 from_f(float v) {
        if (v != v) return 0;
        return (i64)v;
    }
};
template <> struct Fp8Cv<float> {
    static __device__ __forceinline__ double to_d(float v) { return (double)v; }
    static __device__ __forceinline__ float from_f(float v) { return v; }
};
template <> struct Fp8Cv<double> {
    static __device__ __forceinline__ double to_d(double v) { return v; }
    static __device__ __forceinline__ double from_f(float v) { return (double)v; }
};

struct f16_bits { u16 b; };
struct bf16_bits { u16 b; };

template <> struct Fp8Cv<f16_bits> {
    static __device__ __forceinline__ double to_d(f16_bits v) { return (double)f16_to_f32(v.b); }
    static __device__ __forceinline__ f16_bits from_f(float v) { f16_bits r; r.b = f32_to_f16(v); return r; }
};
template <> struct Fp8Cv<bf16_bits> {
    static __device__ __forceinline__ double to_d(bf16_bits v) { return (double)bf16_to_f32(v.b); }
    static __device__ __forceinline__ bf16_bits from_f(float v) { bf16_bits r; r.b = f32_to_bf16(v); return r; }
};

template <typename S>
__device__ void cast_to_fp8_impl(
    const size_t numel, const size_t num_dims, const size_t *info, const S *inp, u8 *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contig(num_dims, dims, strides)) {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
             i += (size_t)blockDim.x * gridDim.x) {
            out[i] = double_to_e4m3(Fp8Cv<S>::to_d(inp[i]));
        }
    } else {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
             i += (size_t)blockDim.x * gridDim.x) {
            out[i] = double_to_e4m3(Fp8Cv<S>::to_d(inp[strided_idx(i, num_dims, dims, strides)]));
        }
    }
}

template <typename D>
__device__ void cast_from_fp8_impl(
    const size_t numel, const size_t num_dims, const size_t *info, const u8 *inp, D *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contig(num_dims, dims, strides)) {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
             i += (size_t)blockDim.x * gridDim.x) {
            out[i] = Fp8Cv<D>::from_f(e4m3_to_f32(inp[i]));
        }
    } else {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
             i += (size_t)blockDim.x * gridDim.x) {
            out[i] = Fp8Cv<D>::from_f(e4m3_to_f32(inp[strided_idx(i, num_dims, dims, strides)]));
        }
    }
}

extern "C" __global__ void cast_f8e4m3_f8e4m3(
    const size_t numel, const size_t num_dims, const size_t *info, const u8 *inp, u8 *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contig(num_dims, dims, strides)) {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
             i += (size_t)blockDim.x * gridDim.x) {
            out[i] = inp[i];
        }
    } else {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel;
             i += (size_t)blockDim.x * gridDim.x) {
            out[i] = inp[strided_idx(i, num_dims, dims, strides)];
        }
    }
}

#define DEF_TO_FP8(S, SFX)                                                                         \
    extern "C" __global__ void cast_##SFX##_f8e4m3(                                                \
        const size_t numel, const size_t num_dims, const size_t *info, const S *inp, u8 *out       \
    ) {                                                                                            \
        cast_to_fp8_impl<S>(numel, num_dims, info, inp, out);                                      \
    }

#define DEF_FROM_FP8(D, SFX)                                                                       \
    extern "C" __global__ void cast_f8e4m3_##SFX(                                                  \
        const size_t numel, const size_t num_dims, const size_t *info, const u8 *inp, D *out       \
    ) {                                                                                            \
        cast_from_fp8_impl<D>(numel, num_dims, info, inp, out);                                    \
    }

DEF_TO_FP8(u8, u8)
DEF_TO_FP8(u32, u32)
DEF_TO_FP8(i64, i64)
DEF_TO_FP8(f16_bits, f16)
DEF_TO_FP8(bf16_bits, bf16)
DEF_TO_FP8(float, f32)
DEF_TO_FP8(double, f64)

DEF_FROM_FP8(u8, u8)
DEF_FROM_FP8(u32, u32)
DEF_FROM_FP8(i64, i64)
DEF_FROM_FP8(f16_bits, f16)
DEF_FROM_FP8(bf16_bits, bf16)
DEF_FROM_FP8(float, f32)
DEF_FROM_FP8(double, f64)
"#;

impl CudaDevice {
    /// Software OCP E4M3FN cast kernel (`cast_<src>_f8_e4m3` / `cast_f8_e4m3_<dst>`), compiled
    /// once with NVRTC and cached on disk.
    pub(crate) fn fp8_cast_func(&self, kernel_name: &str) -> Result<CudaFunc> {
        self.get_or_compile_source_func(kernel_name, MODULE_NAME, || {
            Ok(FP8_CAST_CUDA_SOURCE.to_string())
        })
    }
}
