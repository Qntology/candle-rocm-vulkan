// Exact weight-only quantized matrix-vector products ("W8A16": int8/int4/NVFP4/NVFP8 weights,
// 16/32 bit activations), direct dequantization, exact MoE matvec, and Quantized Mode A lm_head
// Stage-1 kernels shared by BOTH CUDA (NVRTC) and ROCm (hipcc).
//
// * CUDA: compiled at run time by `candle-core/src/quantized/w8a16.rs` (NVRTC, no standard
//   headers, appends one `W8A16_*_KERNEL(...)` instantiation per module).
// * ROCm: compiled at build time via `candle-core/kernels/w8a16.hip` (hipcc).
//
// Numerics (see `QMatMul::forward_exact`):
//   * activations are NOT quantized: f16 / bf16 inputs are widened to f32 exactly;
//   * every weight is dequantized in f32 exactly like `k_quants::Block*::to_float` on the CPU
//     (separately rounded products, no FMA contraction - `__fmul_rn` / `__fsub_rn`), so the
//     dequantization kernels are bit exact with `QTensor::dequantize` on the CPU;
//   * products are accumulated in f32 and the result is rounded once to the output dtype
//     (round to nearest even).
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
// Prevent clang/hipcc from contracting `__fmul_rn` + `__fsub_rn` in DecQ4K/DecQ5K into FMA
// (preserving bit-exact dequantization with CPU `Block*::to_float`; explicit `fmaf()` calls in
// dot-product loops are unaffected by fp contract).
#pragma clang fp contract(off)
#define W_IS_HIP 1
#define W_LDG(p) (*(p))
#else
#define W_IS_HIP 0
#define W_LDG(p) __ldg(p)
#endif

typedef unsigned char w_u8;
typedef signed char w_i8;
typedef unsigned short w_u16;
typedef unsigned int w_u32;
typedef long long w_i64;

// Storage types of candle's f16 / bf16 (layout compatible with `half::f16` / `half::bf16`).
struct w_f16 {
    w_u16 b;
};
struct w_bf16 {
    w_u16 b;
};

struct __align__(8) w_v2 {
    w_u32 x, y;
};
struct __align__(16) w_v4 {
    w_u32 x, y, z, w;
};

#define W_DEV __device__ __forceinline__

// ---------------------------------------------------------------------------
// Portable 32-lane sub-warp reduction (CUDA warp32, HIP wave32 & wave64)
// ---------------------------------------------------------------------------

W_DEV float w_warp32_sum(float v) {
#if W_IS_HIP
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
#else
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
#endif
    return v;
}

// ---------------------------------------------------------------------------
// Scalar conversions (bit exact with `half`: IEEE round to nearest even, NaN kept)
// ---------------------------------------------------------------------------

#if W_IS_HIP
W_DEV float w_h2f(w_u16 h) {
    const w_u32 e = (h >> 10) & 0x1fu;
    const w_u32 m = h & 0x3ffu;
    const float sub = (float)m * 5.9604644775390625e-8f;
    const w_u32 big = (e == 31u ? 0x7f800000u : ((e + 112u) << 23)) | (m << 13);
    const float v = e == 0u ? sub : __uint_as_float(big);
    return (h & 0x8000u) ? -v : v;
}

W_DEV w_u16 w_f2h(float f) {
    const w_u32 x = __float_as_uint(f);
    const w_u32 s = (x >> 16) & 0x8000u;
    const w_u32 ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) return (w_u16)(s | 0x7e00u);
    if (ax >= 0x477ff000u) return (w_u16)(s | 0x7c00u);
    if (ax < 0x38800000u) {
        const float v = rintf(__uint_as_float(ax) * 16777216.0f);
        return (w_u16)(s | (w_u32)v);
    }
    const w_u32 r = ax + 0xfffu + ((ax >> 13) & 1u);
    return (w_u16)(s | ((r >> 13) - (112u << 10)));
}
#else
W_DEV float w_h2f(w_u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

W_DEV w_u16 w_f2h(float f) {
    w_u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
#endif

W_DEV float w_bf2f(w_u32 b) { return __uint_as_float(b << 16); }

W_DEV w_u16 w_f2bf(float f) {
    w_u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (w_u16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (w_u16)(u >> 16);
}

// Software OCP E4M3FN and E2M1 decoders (bit exact with accel::formats).
W_DEV float w_e4m3_to_f(w_u32 c) {
    const w_u32 mag = c & 0x7fu;
    const float nrm = __uint_as_float((mag << 20) + (120u << 23));
    const float sub = (float)mag * 0.001953125f;
    float v = mag >= 8u ? nrm : sub;
    v = (c & 0x80u) ? -v : v;
    return mag == 0x7fu ? __uint_as_float(0x7fc00000u) : v;
}

W_DEV float w_fp4_to_f(w_u32 c) {
    const w_u32 e = (c >> 1) & 3u;
    const w_u32 m = c & 1u;
    const float v = e == 0u ? 0.5f * (float)m : __uint_as_float(((e + 126u) << 23) | (m << 22));
    return (c & 8u) ? -v : v;
}

W_DEV w_u16 w_ld16(const w_u8 *p) { return W_LDG(reinterpret_cast<const w_u16 *>(p)); }

// 8 bytes at a 2 byte aligned address, as two little endian words.
W_DEV void w_ld8_a2(const w_u8 *p, w_u32 &lo, w_u32 &hi) {
    const w_u16 *q = reinterpret_cast<const w_u16 *>(p);
    lo = (w_u32)W_LDG(q) | ((w_u32)W_LDG(q + 1) << 16);
    hi = (w_u32)W_LDG(q + 2) | ((w_u32)W_LDG(q + 3) << 16);
}

// 8 bytes at an 8 byte aligned address.
W_DEV void w_ld8_a8(const w_u8 *p, w_u32 &lo, w_u32 &hi) {
    const w_v2 v = *reinterpret_cast<const w_v2 *>(p);
    lo = v.x;
    hi = v.y;
}

// Byte `i` (0..7) of the 8 byte group (lo, hi).
#define W_BYTE(lo, hi, i) ((((i) < 4 ? (lo) : (hi)) >> (8 * ((i) & 3))) & 0xffu)

// ---------------------------------------------------------------------------
// Element I/O: 8 consecutive values (16 byte aligned for 16 bit types, 32 for f32)
// ---------------------------------------------------------------------------

template <typename T> struct WIo;

template <> struct WIo<float> {
    static W_DEV void load8(const float *p, float *v) {
        const w_v4 a = *reinterpret_cast<const w_v4 *>(p);
        const w_v4 b = *(reinterpret_cast<const w_v4 *>(p) + 1);
        v[0] = __uint_as_float(a.x);
        v[1] = __uint_as_float(a.y);
        v[2] = __uint_as_float(a.z);
        v[3] = __uint_as_float(a.w);
        v[4] = __uint_as_float(b.x);
        v[5] = __uint_as_float(b.y);
        v[6] = __uint_as_float(b.z);
        v[7] = __uint_as_float(b.w);
    }
    static W_DEV void store8(float *p, const float *v) {
        w_v4 a, b;
        a.x = __float_as_uint(v[0]);
        a.y = __float_as_uint(v[1]);
        a.z = __float_as_uint(v[2]);
        a.w = __float_as_uint(v[3]);
        b.x = __float_as_uint(v[4]);
        b.y = __float_as_uint(v[5]);
        b.z = __float_as_uint(v[6]);
        b.w = __float_as_uint(v[7]);
        reinterpret_cast<w_v4 *>(p)[0] = a;
        reinterpret_cast<w_v4 *>(p)[1] = b;
    }
    static W_DEV float to_f(float v) { return v; }
    static W_DEV float from_f(float v) { return v; }
};

template <> struct WIo<w_bf16> {
    static W_DEV void load8(const w_bf16 *p, float *v) {
        const w_v4 a = *reinterpret_cast<const w_v4 *>(p);
        v[0] = w_bf2f(a.x & 0xffffu);
        v[1] = w_bf2f(a.x >> 16);
        v[2] = w_bf2f(a.y & 0xffffu);
        v[3] = w_bf2f(a.y >> 16);
        v[4] = w_bf2f(a.z & 0xffffu);
        v[5] = w_bf2f(a.z >> 16);
        v[6] = w_bf2f(a.w & 0xffffu);
        v[7] = w_bf2f(a.w >> 16);
    }
    static W_DEV void store8(w_bf16 *p, const float *v) {
        w_v4 a;
        a.x = (w_u32)w_f2bf(v[0]) | ((w_u32)w_f2bf(v[1]) << 16);
        a.y = (w_u32)w_f2bf(v[2]) | ((w_u32)w_f2bf(v[3]) << 16);
        a.z = (w_u32)w_f2bf(v[4]) | ((w_u32)w_f2bf(v[5]) << 16);
        a.w = (w_u32)w_f2bf(v[6]) | ((w_u32)w_f2bf(v[7]) << 16);
        *reinterpret_cast<w_v4 *>(p) = a;
    }
    static W_DEV float to_f(w_bf16 v) { return w_bf2f(v.b); }
    static W_DEV w_bf16 from_f(float v) {
        w_bf16 r;
        r.b = w_f2bf(v);
        return r;
    }
};

template <> struct WIo<w_f16> {
    static W_DEV void load8(const w_f16 *p, float *v) {
        const w_v4 a = *reinterpret_cast<const w_v4 *>(p);
        v[0] = w_h2f((w_u16)(a.x & 0xffffu));
        v[1] = w_h2f((w_u16)(a.x >> 16));
        v[2] = w_h2f((w_u16)(a.y & 0xffffu));
        v[3] = w_h2f((w_u16)(a.y >> 16));
        v[4] = w_h2f((w_u16)(a.z & 0xffffu));
        v[5] = w_h2f((w_u16)(a.z >> 16));
        v[6] = w_h2f((w_u16)(a.w & 0xffffu));
        v[7] = w_h2f((w_u16)(a.w >> 16));
    }
    static W_DEV void store8(w_f16 *p, const float *v) {
        w_v4 a;
        a.x = (w_u32)w_f2h(v[0]) | ((w_u32)w_f2h(v[1]) << 16);
        a.y = (w_u32)w_f2h(v[2]) | ((w_u32)w_f2h(v[3]) << 16);
        a.z = (w_u32)w_f2h(v[4]) | ((w_u32)w_f2h(v[5]) << 16);
        a.w = (w_u32)w_f2h(v[6]) | ((w_u32)w_f2h(v[7]) << 16);
        *reinterpret_cast<w_v4 *>(p) = a;
    }
    static W_DEV float to_f(w_f16 v) { return w_h2f(v.b); }
    static W_DEV w_f16 from_f(float v) {
        w_f16 r;
        r.b = w_f2h(v);
        return r;
    }
};

// ---------------------------------------------------------------------------
// Block decoders. A block of QK weights (BYTES bytes) is split into TASKS lane tasks; a task
// yields RUNS runs of 8 consecutive weights, run r starting at weight `xo[r]` of the block.
// ---------------------------------------------------------------------------

// llama.cpp / k_quants.rs `get_scale_min_k4`.
W_DEV void w_scale_min_k4(int j, const w_u8 *q, int &d, int &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct DecQ8_0 {
    enum { QK = 32, BYTES = 34, TASKS = 4, RUNS = 1 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        w_u32 lo, hi;
        w_ld8_a2(blk + 2 + 8 * t, lo, hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __fmul_rn((float)(w_i8)W_BYTE(lo, hi, i), d);
        xo[0] = 8 * t;
    }
};

struct DecQ4_0 {
    enum { QK = 32, BYTES = 18, TASKS = 2, RUNS = 2 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        w_u32 lo, hi;
        w_ld8_a2(blk + 2 + 8 * t, lo, hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 b = W_BYTE(lo, hi, i);
            w[i] = __fmul_rn((float)((int)(b & 0xFu) - 8), d);
            w[8 + i] = __fmul_rn((float)((int)(b >> 4) - 8), d);
        }
        xo[0] = 8 * t;
        xo[1] = 16 + 8 * t;
    }
};

struct DecQ4K {
    enum { QK = 256, BYTES = 144, TASKS = 16, RUNS = 2 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        const float dmin = w_h2f(w_ld16(blk + 2));
        const int j = t >> 2, l0 = (t & 3) * 8;
        int sc, m;
        w_scale_min_k4(2 * j, blk + 4, sc, m);
        const float d1 = __fmul_rn(d, (float)sc), m1 = __fmul_rn(dmin, (float)m);
        w_scale_min_k4(2 * j + 1, blk + 4, sc, m);
        const float d2 = __fmul_rn(d, (float)sc), m2 = __fmul_rn(dmin, (float)m);
        w_u32 lo, hi;
        w_ld8_a8(blk + 16 + 32 * j + l0, lo, hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 b = W_BYTE(lo, hi, i);
            w[i] = __fsub_rn(__fmul_rn(d1, (float)(b & 0xFu)), m1);
            w[8 + i] = __fsub_rn(__fmul_rn(d2, (float)(b >> 4)), m2);
        }
        xo[0] = 64 * j + l0;
        xo[1] = 64 * j + 32 + l0;
    }
};

struct DecQ5K {
    enum { QK = 256, BYTES = 176, TASKS = 16, RUNS = 2 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        const float dmin = w_h2f(w_ld16(blk + 2));
        const int j = t >> 2, l0 = (t & 3) * 8;
        int sc, m;
        w_scale_min_k4(2 * j, blk + 4, sc, m);
        const float d1 = __fmul_rn(d, (float)sc), m1 = __fmul_rn(dmin, (float)m);
        w_scale_min_k4(2 * j + 1, blk + 4, sc, m);
        const float d2 = __fmul_rn(d, (float)sc), m2 = __fmul_rn(dmin, (float)m);
        w_u32 lo, hi, hlo, hhi;
        w_ld8_a8(blk + 48 + 32 * j + l0, lo, hi);
        w_ld8_a8(blk + 16 + l0, hlo, hhi);
        const w_u32 u1 = 1u << (2 * j), u2 = 2u << (2 * j);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 b = W_BYTE(lo, hi, i);
            const w_u32 h = W_BYTE(hlo, hhi, i);
            w[i] = __fsub_rn(__fmul_rn(d1, __fadd_rn((float)(b & 0xFu), (h & u1) ? 16.0f : 0.0f)), m1);
            w[8 + i] = __fsub_rn(__fmul_rn(d2, __fadd_rn((float)(b >> 4), (h & u2) ? 16.0f : 0.0f)), m2);
        }
        xo[0] = 64 * j + l0;
        xo[1] = 64 * j + 32 + l0;
    }
};

struct DecQ6K {
    enum { QK = 256, BYTES = 210, TASKS = 8, RUNS = 4 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const int nh = t >> 2, l0 = (t & 3) * 8, is = l0 / 16;
        const float d = w_h2f(w_ld16(blk + 208));
        const w_u8 *sc = blk + 192 + 8 * nh;
        const float s1 = __fmul_rn(d, (float)(w_i8)sc[is]);
        const float s2 = __fmul_rn(d, (float)(w_i8)sc[is + 2]);
        const float s3 = __fmul_rn(d, (float)(w_i8)sc[is + 4]);
        const float s4 = __fmul_rn(d, (float)(w_i8)sc[is + 6]);
        w_u32 alo, ahi, blo, bhi, hlo, hhi;
        w_ld8_a2(blk + 64 * nh + l0, alo, ahi);
        w_ld8_a2(blk + 64 * nh + 32 + l0, blo, bhi);
        w_ld8_a2(blk + 128 + 32 * nh + l0, hlo, hhi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 a = W_BYTE(alo, ahi, i), b = W_BYTE(blo, bhi, i), h = W_BYTE(hlo, hhi, i);
            const int q1 = (int)((a & 0xFu) | ((h & 3u) << 4)) - 32;
            const int q2 = (int)((b & 0xFu) | (((h >> 2) & 3u) << 4)) - 32;
            const int q3 = (int)((a >> 4) | (((h >> 4) & 3u) << 4)) - 32;
            const int q4 = (int)((b >> 4) | (((h >> 6) & 3u) << 4)) - 32;
            w[i] = __fmul_rn(s1, (float)q1);
            w[8 + i] = __fmul_rn(s2, (float)q2);
            w[16 + i] = __fmul_rn(s3, (float)q3);
            w[24 + i] = __fmul_rn(s4, (float)q4);
        }
        xo[0] = 128 * nh + l0;
        xo[1] = 128 * nh + 32 + l0;
        xo[2] = 128 * nh + 64 + l0;
        xo[3] = 128 * nh + 96 + l0;
    }
};

// ---------------------------------------------------------------------------
// y[m, row] = sum_k dequant(W)[row, k] * x[m, k]   for m < M (M <= 8 input rows).
// ---------------------------------------------------------------------------

template <typename D, typename XT, int M, int WPR, int RPB, int UNR>
W_DEV void w8a16_mv(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y,
                    const int k, const int n, const w_i64 row_bytes) {
    const int lane = threadIdx.x, wy = threadIdx.y, rz = threadIdx.z;
    const int row = blockIdx.x * RPB + rz;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    if (row < n) {
        const w_u8 *wr = wq + (w_i64)row * row_bytes;
        const int ntasks = (k / D::QK) * D::TASKS;
#pragma unroll UNR
        for (int tk = wy * 32 + lane; tk < ntasks; tk += 32 * WPR) {
            const int b = tk / D::TASKS, t = tk % D::TASKS;
            float w[8 * D::RUNS];
            int xo[D::RUNS];
            D::decode(wr + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
            for (int r = 0; r < D::RUNS; ++r) {
                const XT *xr = x + b * D::QK + xo[r];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float xv[8];
                    WIo<XT>::load8(xr + (w_i64)m * k, xv);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[m] = fmaf(w[8 * r + i], xv[i], acc[m]);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = w_warp32_sum(acc[m]);
    if (WPR > 1) {
        __shared__ float red[RPB][WPR][M];
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < M; ++m) red[rz][wy][m] = acc[m];
        }
        __syncthreads();
        if (wy == 0 && lane < M && row < n) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; ++w) s += red[rz][w][lane];
            y[(w_i64)lane * n + row] = WIo<XT>::from_f(s);
        }
    } else {
        float v = acc[0];
#pragma unroll
        for (int m = 1; m < M; ++m) v = lane == m ? acc[m] : v;
        if (lane < M && row < n) y[(w_i64)lane * n + row] = WIo<XT>::from_f(v);
    }
}

#define W8A16_MV_KERNEL(name, D, XT, M, WPR, RPB, UNR)                                             \
    extern "C" __global__ void __launch_bounds__(32 * (WPR) * (RPB))                               \
        name(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y, int k,      \
             int n, w_i64 row_bytes) {                                                             \
        w8a16_mv<D, XT, M, WPR, RPB, UNR>(wq, x, y, k, n, row_bytes);                              \
    }

// Runtime-M variant for precompiled HIP modules (handles 1 <= rows <= M_MAX without 8x bloat).
template <typename D, typename XT, int M_MAX, int WPR, int RPB>
W_DEV void w8a16_mv_dyn(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y,
                        const int k, const int n, const int rows, const w_i64 row_bytes) {
    const int lane = threadIdx.x, wy = threadIdx.y, rz = threadIdx.z;
    const int row = blockIdx.x * RPB + rz;
    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;
    if (row < n) {
        const w_u8 *wr = wq + (w_i64)row * row_bytes;
        const int ntasks = (k / D::QK) * D::TASKS;
#pragma unroll 2
        for (int tk = wy * 32 + lane; tk < ntasks; tk += 32 * WPR) {
            const int b = tk / D::TASKS, t = tk % D::TASKS;
            float w[8 * D::RUNS];
            int xo[D::RUNS];
            D::decode(wr + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
            for (int r = 0; r < D::RUNS; ++r) {
                const XT *xr = x + b * D::QK + xo[r];
                for (int m = 0; m < rows; ++m) {
                    float xv[8];
                    WIo<XT>::load8(xr + (w_i64)m * k, xv);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[m] = fmaf(w[8 * r + i], xv[i], acc[m]);
                }
            }
        }
    }
    for (int m = 0; m < rows; ++m) acc[m] = w_warp32_sum(acc[m]);
    if (WPR > 1) {
        __shared__ float red[RPB][WPR][M_MAX];
        if (lane == 0) {
            for (int m = 0; m < rows; ++m) red[rz][wy][m] = acc[m];
        }
        __syncthreads();
        if (wy == 0 && lane < rows && row < n) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; ++w) s += red[rz][w][lane];
            y[(w_i64)lane * n + row] = WIo<XT>::from_f(s);
        }
    } else {
        float v = acc[0];
        for (int m = 1; m < rows; ++m) v = lane == m ? acc[m] : v;
        if (lane < rows && row < n) y[(w_i64)lane * n + row] = WIo<XT>::from_f(v);
    }
}

#define W8A16_MVDYN_KERNEL(name, D, XT, M_MAX, WPR, RPB)                                           \
    extern "C" __global__ void __launch_bounds__(32 * (WPR) * (RPB))                               \
        name(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y, int k,      \
             int n, int rows, w_i64 row_bytes) {                                                   \
        w8a16_mv_dyn<D, XT, M_MAX, WPR, RPB>(wq, x, y, k, n, rows, row_bytes);                     \
    }

// ---------------------------------------------------------------------------
// Dequantization of `nblocks` consecutive blocks into a dense row-major array.
// ---------------------------------------------------------------------------

template <typename D, typename OT>
W_DEV void w8a16_dq(const w_u8 *__restrict__ wq, OT *__restrict__ y, const w_i64 nblocks) {
    const w_i64 ntasks = nblocks * D::TASKS;
    for (w_i64 tk = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; tk < ntasks;
         tk += (w_i64)gridDim.x * blockDim.x) {
        const w_i64 b = tk / D::TASKS;
        const int t = (int)(tk % D::TASKS);
        float w[8 * D::RUNS];
        int xo[D::RUNS];
        D::decode(wq + b * D::BYTES, t, w, xo);
#pragma unroll
        for (int r = 0; r < D::RUNS; ++r) WIo<OT>::store8(y + b * D::QK + xo[r], w + 8 * r);
    }
}

#define W8A16_DQ_KERNEL(name, D, OT)                                                               \
    extern "C" __global__ void name(const w_u8 *__restrict__ wq, OT *__restrict__ y,               \
                                    w_i64 nblocks) {                                               \
        w8a16_dq<D, OT>(wq, y, nblocks);                                                           \
    }

// ---------------------------------------------------------------------------
// Element type conversion of F32 / F16 / BF16 GGUF tensors (one rounding, no f32 copy).
// ---------------------------------------------------------------------------

template <typename IT, typename OT>
W_DEV void w8a16_cvt(const IT *__restrict__ x, OT *__restrict__ y, const w_i64 n) {
    for (w_i64 i = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (w_i64)gridDim.x * blockDim.x)
        y[i] = WIo<OT>::from_f(WIo<IT>::to_f(x[i]));
}

#define W8A16_CVT_KERNEL(name, IT, OT)                                                             \
    extern "C" __global__ void name(const IT *__restrict__ x, OT *__restrict__ y, w_i64 n) {       \
        w8a16_cvt<IT, OT>(x, y, n);                                                                \
    }

// ---------------------------------------------------------------------------
// f32 GEMM tile (rows x cols, contiguous) rounded once into columns of a wider output.
// ---------------------------------------------------------------------------

template <typename OT>
W_DEV void w8a16_cvt2d(const float *__restrict__ x, OT *__restrict__ y, const int rows,
                       const int cols, const w_i64 ldy) {
    const w_i64 total = (w_i64)rows * cols;
    for (w_i64 i = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (w_i64)gridDim.x * blockDim.x) {
        const w_i64 r = i / cols, c = i - r * cols;
        y[r * ldy + c] = WIo<OT>::from_f(x[i]);
    }
}

#define W8A16_CVT2D_KERNEL(name, OT)                                                               \
    extern "C" __global__ void name(const float *__restrict__ x, OT *__restrict__ y, int rows,     \
                                    int cols, w_i64 ldy) {                                         \
        w8a16_cvt2d<OT>(x, y, rows, cols, ldy);                                                    \
    }

// ---------------------------------------------------------------------------
// Exact weight-only Indexed MoE MatVec (CUDA + ROCm, F32 / F16 / BF16 activations).
// Grid: (ceil(n / 4), batch * topk, 1), Block: (32, 4, 1) -> 4 output rows per block.
// ---------------------------------------------------------------------------

template <typename D, typename XT>
W_DEV void w8a16_indexed_moe(const w_u8 *__restrict__ wq, const XT *__restrict__ x,
                             const w_u32 *__restrict__ ids, XT *__restrict__ y, const int k,
                             const int n, const int topk, const int x_has_topk,
                             const w_i64 row_bytes, const w_i64 expert_bytes) {
    const int lane = threadIdx.x;
    const int rz = threadIdx.y;
    const int row = blockIdx.x * 4 + rz;
    const int slot = blockIdx.y; // 0 .. batch * topk - 1
    if (row >= n) return;
    const int b_idx = slot / topk;
    const int e_idx = (int)ids[slot];
    const XT *xb = x + (w_i64)(x_has_topk ? slot : b_idx) * k;
    const w_u8 *wr = wq + (w_i64)e_idx * expert_bytes + (w_i64)row * row_bytes;
    const int ntasks = (k / D::QK) * D::TASKS;
    float acc = 0.0f;
    for (int tk = lane; tk < ntasks; tk += 32) {
        const int b = tk / D::TASKS, t = tk % D::TASKS;
        float w[8 * D::RUNS];
        int xo[D::RUNS];
        D::decode(wr + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
        for (int r = 0; r < D::RUNS; ++r) {
            float xv[8];
            WIo<XT>::load8(xb + b * D::QK + xo[r], xv);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc = fmaf(w[8 * r + i], xv[i], acc);
        }
    }
    acc = w_warp32_sum(acc);
    if (lane == 0) y[(w_i64)slot * n + row] = WIo<XT>::from_f(acc);
}

#define W8A16_MOE_KERNEL(name, D, XT)                                                              \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        name(const w_u8 *__restrict__ wq, const XT *__restrict__ x, const w_u32 *__restrict__ ids, \
             XT *__restrict__ y, int k, int n, int topk, int x_has_topk, w_i64 row_bytes,          \
             w_i64 expert_bytes) {                                                                 \
        w8a16_indexed_moe<D, XT>(wq, x, ids, y, k, n, topk, x_has_topk, row_bytes, expert_bytes);  \
    }

// ---------------------------------------------------------------------------
// Quantized Mode A lm_head Stage-1 Tile Argmax (0-byte [M, V] logits materialization).
// Grid: (n_tiles, 1, 1), Block: (32, 8, 1) -> 8 warps x 4 vocab cols = 32 cols/tile.
// Output matches lm_stage2_reduce_cs_*: part_val[M, n_tiles], part_idx[M, n_tiles].
// ---------------------------------------------------------------------------

template <typename D, typename XT, int MAX_M>
W_DEV void w8a16_lm_stage1_quant(const XT *__restrict__ x, const w_u8 *__restrict__ wq,
                                 const float *__restrict__ bias, float *__restrict__ part_val,
                                 w_u32 *__restrict__ part_idx, const int M, const int V_valid,
                                 const int K, const int n_tiles, const w_i64 row_bytes) {
    const int tile_id = blockIdx.x;
    if (tile_id >= n_tiles || M <= 0 || M > MAX_M) return;
    const int lane = threadIdx.x;
    const int grp = threadIdx.y; // 0..7
    const int tid = grp * 32 + lane;
    const int col0 = tile_id * 32 + grp * 4;
    const int ntasks = (K / D::QK) * D::TASKS;

    float acc[MAX_M][4];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc[m][0] = 0.0f;
        acc[m][1] = 0.0f;
        acc[m][2] = 0.0f;
        acc[m][3] = 0.0f;
    }

    if (col0 < V_valid) {
        for (int tk = lane; tk < ntasks; tk += 32) {
            const int b = tk / D::TASKS, t = tk % D::TASKS;
            float w[8 * D::RUNS];
            int xo[D::RUNS];
            if constexpr (MAX_M == 1) {
                D::decode(wq + (w_i64)col0 * row_bytes + (w_i64)b * D::BYTES, t, w, xo);
                float xv[8 * D::RUNS];
#pragma unroll
                for (int r = 0; r < D::RUNS; ++r) {
                    WIo<XT>::load8(x + b * D::QK + xo[r], xv + 8 * r);
                }
#pragma unroll
                for (int i = 0; i < 8 * D::RUNS; ++i) acc[0][0] = fmaf(w[i], xv[i], acc[0][0]);
#pragma unroll
                for (int c = 1; c < 4; ++c) {
                    const int col = col0 + c;
                    if (col < V_valid) {
                        D::decode(wq + (w_i64)col * row_bytes + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
                        for (int i = 0; i < 8 * D::RUNS; ++i)
                            acc[0][c] = fmaf(w[i], xv[i], acc[0][c]);
                    }
                }
            } else {
                // For multi-row decode (M <= 8), stream x from L1 per (r, m) instead of keeping
                // MAX_M * 8 * D::RUNS floats live in registers (avoids VGPR spilling on RDNA/CDNA).
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int col = col0 + c;
                    if (col < V_valid) {
                        D::decode(wq + (w_i64)col * row_bytes + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
                        for (int r = 0; r < D::RUNS; ++r) {
                            const XT *xr = x + b * D::QK + xo[r];
                            for (int m = 0; m < M; ++m) {
                                float xv[8];
                                WIo<XT>::load8(xr + (w_i64)m * K, xv);
#pragma unroll
                                for (int i = 0; i < 8; ++i)
                                    acc[m][c] = fmaf(w[8 * r + i], xv[i], acc[m][c]);
                            }
                        }
                    }
                }
            }
        }
    }

    __shared__ float col_logits[MAX_M * 32];
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float sum = w_warp32_sum(acc[m][c]);
            if (lane == 0) {
                const int gc = col0 + c;
                float v = __uint_as_float(0xff800000u); // -inf
                if (gc < V_valid) {
                    v = sum;
                    if (bias != nullptr) v += bias[gc];
                }
                col_logits[m * 32 + grp * 4 + c] = v;
            }
        }
    }
    __syncthreads();

    if (tid < M) {
        float best_v = __uint_as_float(0xff800000u);
        w_u32 best_i = 0xffffffffu;
        for (int lc = 0; lc < 32; ++lc) {
            const int gc = tile_id * 32 + lc;
            if (gc < V_valid) {
                const float v = col_logits[tid * 32 + lc];
                if ((v > best_v) || (v == best_v && (w_u32)gc < best_i)) {
                    best_v = v;
                    best_i = (w_u32)gc;
                }
            }
        }
        part_val[(w_i64)tid * n_tiles + tile_id] = best_v;
        part_idx[(w_i64)tid * n_tiles + tile_id] = best_i;
    }
}

#define W8A16_LM_QUANT_KERNEL(name, D, XT, MAX_M)                                                  \
    extern "C" __global__ void __launch_bounds__(256)                                              \
        name(const XT *__restrict__ x, const w_u8 *__restrict__ wq,                                \
             const float *__restrict__ bias, float *__restrict__ part_val,                         \
             w_u32 *__restrict__ part_idx, int M, int V_valid, int K, int n_tiles,                 \
             w_i64 row_bytes) {                                                                    \
        w8a16_lm_stage1_quant<D, XT, MAX_M>(x, wq, bias, part_val, part_idx, M, V_valid, K,        \
                                            n_tiles, row_bytes);                                   \
    }

// ---------------------------------------------------------------------------
// Planar NVFP4 (E2M1 + 16-elem OCP E4M3 block scale + f32 row scale) and
// Planar NVFP8 (OCP E4M3 + f32 row scale) exact matvec & dequant kernels.
// ---------------------------------------------------------------------------

template <typename XT, int M_MAX>
W_DEV void w8a16_nvfp4_mv(const w_u8 *__restrict__ qdata, const w_u8 *__restrict__ bscale,
                          const float *__restrict__ rscale, const XT *__restrict__ x,
                          XT *__restrict__ y, const int k, const int n, const int rows) {
    const int lane = threadIdx.x;
    const int rz = threadIdx.y;
    const int row = blockIdx.x * 4 + rz;
    if (row >= n) return;
    const int nblk = k >> 4; // K divisible by 16
    const w_u8 *qr = qdata + (w_i64)row * (k >> 1);
    const w_u8 *bs = bscale + (w_i64)row * nblk;
    const float row_g = rscale[row];
    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;

    for (int b = lane; b < nblk; b += 32) {
        w_u32 lo, hi;
        w_ld8_a8(qr + (w_i64)b * 8, lo, hi);
        const float eff = __fmul_rn(w_e4m3_to_f(bs[b]), row_g);
        float w0[8], w1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w0[i] = __fmul_rn(w_fp4_to_f((lo >> (4 * i)) & 0xfu), eff);
            w1[i] = __fmul_rn(w_fp4_to_f((hi >> (4 * i)) & 0xfu), eff);
        }
        const XT *xr = x + b * 16;
        for (int m = 0; m < rows; ++m) {
            float xv0[8], xv1[8];
            WIo<XT>::load8(xr + (w_i64)m * k, xv0);
            WIo<XT>::load8(xr + (w_i64)m * k + 8, xv1);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[m] = fmaf(w0[i], xv0[i], acc[m]);
                acc[m] = fmaf(w1[i], xv1[i], acc[m]);
            }
        }
    }
    for (int m = 0; m < rows; ++m) {
        const float s = w_warp32_sum(acc[m]);
        if (lane == 0) y[(w_i64)m * n + row] = WIo<XT>::from_f(s);
    }
}

template <typename OT>
W_DEV void w8a16_nvfp4_dq(const w_u8 *__restrict__ qdata, const w_u8 *__restrict__ bscale,
                          const float *__restrict__ rscale, OT *__restrict__ y, const int k,
                          const int n) {
    const int nblk_row = k >> 4;
    const w_i64 total_blk = (w_i64)n * nblk_row;
    for (w_i64 idx = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; idx < total_blk;
         idx += (w_i64)gridDim.x * blockDim.x) {
        const int row = (int)(idx / nblk_row);
        const int b = (int)(idx - (w_i64)row * nblk_row);
        w_u32 lo, hi;
        w_ld8_a8(qdata + idx * 8, lo, hi);
        const float eff = __fmul_rn(w_e4m3_to_f(bscale[idx]), rscale[row]);
        float w0[8], w1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w0[i] = __fmul_rn(w_fp4_to_f((lo >> (4 * i)) & 0xfu), eff);
            w1[i] = __fmul_rn(w_fp4_to_f((hi >> (4 * i)) & 0xfu), eff);
        }
        OT *dst = y + (w_i64)row * k + b * 16;
        WIo<OT>::store8(dst, w0);
        WIo<OT>::store8(dst + 8, w1);
    }
}

template <typename XT, int M_MAX>
W_DEV void w8a16_nvfp8_mv(const w_u8 *__restrict__ qdata, const float *__restrict__ rscale,
                          const XT *__restrict__ x, XT *__restrict__ y, const int k, const int n,
                          const int rows) {
    const int lane = threadIdx.x;
    const int rz = threadIdx.y;
    const int row = blockIdx.x * 4 + rz;
    if (row >= n) return;
    const int ngrp = k >> 3; // groups of 8
    const w_u8 *qr = qdata + (w_i64)row * k;
    const float row_g = rscale[row];
    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;

    for (int g = lane; g < ngrp; g += 32) {
        w_u32 lo, hi;
        w_ld8_a8(qr + (w_i64)g * 8, lo, hi);
        float w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __fmul_rn(w_e4m3_to_f(W_BYTE(lo, hi, i)), row_g);
        const XT *xr = x + g * 8;
        for (int m = 0; m < rows; ++m) {
            float xv[8];
            WIo<XT>::load8(xr + (w_i64)m * k, xv);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[m] = fmaf(w[i], xv[i], acc[m]);
        }
    }
    for (int m = 0; m < rows; ++m) {
        const float s = w_warp32_sum(acc[m]);
        if (lane == 0) y[(w_i64)m * n + row] = WIo<XT>::from_f(s);
    }
}

template <typename OT>
W_DEV void w8a16_nvfp8_dq(const w_u8 *__restrict__ qdata, const float *__restrict__ rscale,
                          OT *__restrict__ y, const int k, const int n) {
    const int ngrp_row = k >> 3;
    const w_i64 total_grp = (w_i64)n * ngrp_row;
    for (w_i64 idx = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; idx < total_grp;
         idx += (w_i64)gridDim.x * blockDim.x) {
        const int row = (int)(idx / ngrp_row);
        const int g = (int)(idx - (w_i64)row * ngrp_row);
        w_u32 lo, hi;
        w_ld8_a8(qdata + idx * 8, lo, hi);
        const float row_g = rscale[row];
        float w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __fmul_rn(w_e4m3_to_f(W_BYTE(lo, hi, i)), row_g);
        WIo<OT>::store8(y + (w_i64)row * k + g * 8, w);
    }
}

#define W8A16_NVFP4_MV_KERNEL(name, XT, M_MAX)                                                     \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        name(const w_u8 *__restrict__ qdata, const w_u8 *__restrict__ bscale,                      \
             const float *__restrict__ rscale, const XT *__restrict__ x, XT *__restrict__ y,       \
             int k, int n, int rows) {                                                             \
        w8a16_nvfp4_mv<XT, M_MAX>(qdata, bscale, rscale, x, y, k, n, rows);                        \
    }

#define W8A16_NVFP4_DQ_KERNEL(name, OT)                                                            \
    extern "C" __global__ void name(const w_u8 *__restrict__ qdata,                                \
                                    const w_u8 *__restrict__ bscale,                               \
                                    const float *__restrict__ rscale, OT *__restrict__ y, int k,   \
                                    int n) {                                                       \
        w8a16_nvfp4_dq<OT>(qdata, bscale, rscale, y, k, n);                                        \
    }

#define W8A16_NVFP8_MV_KERNEL(name, XT, M_MAX)                                                     \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        name(const w_u8 *__restrict__ qdata, const float *__restrict__ rscale,                     \
             const XT *__restrict__ x, XT *__restrict__ y, int k, int n, int rows) {               \
        w8a16_nvfp8_mv<XT, M_MAX>(qdata, rscale, x, y, k, n, rows);                                \
    }

#define W8A16_NVFP8_DQ_KERNEL(name, OT)                                                            \
    extern "C" __global__ void name(const w_u8 *__restrict__ qdata,                                \
                                    const float *__restrict__ rscale, OT *__restrict__ y, int k,   \
                                    int n) {                                                       \
        w8a16_nvfp8_dq<OT>(qdata, rscale, y, k, n);                                                \
    }

// ---------------------------------------------------------------------------
// 3-Tier W8A16 On-Chip Fused Tile GEMM (8 < M <= 64, BM = 16, BN = 4/block).
// Eliminates the 32 MiB HBM dequantization staging buffer for medium batches
// (chunked prefill tails & speculative verification) while preserving exact
// F32 accumulation and single rounding on store.
// Grid: (ceil(n / 4), ceil(rows / 16), 1), Block: (32, 4, 1).
// ---------------------------------------------------------------------------
template <typename D, typename XT>
W_DEV void w8a16_fused_tile16(const w_u8 *__restrict__ wq, const XT *__restrict__ x,
                              XT *__restrict__ y, const int k, const int n, const int rows,
                              const w_i64 row_bytes) {
    const int lane = threadIdx.x;
    const int rz = threadIdx.y; // 0..3
    const int col = blockIdx.x * 4 + rz;
    const int m0 = blockIdx.y * 16;
    if (col >= n || m0 >= rows) return;
    const int m_cnt = (rows - m0 < 16) ? (rows - m0) : 16;
    const int ntasks = (k / D::QK) * D::TASKS;
    const w_u8 *wrow = wq + (w_i64)col * row_bytes;

    float acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0.0f;

    for (int tk = lane; tk < ntasks; tk += 32) {
        const int b = tk / D::TASKS, t = tk % D::TASKS;
        float w[8 * D::RUNS];
        int xo[D::RUNS];
        D::decode(wrow + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
        for (int r = 0; r < D::RUNS; ++r) {
            const XT *xr = x + (w_i64)m0 * k + b * D::QK + xo[r];
            for (int m = 0; m < m_cnt; ++m) {
                float xv[8];
                WIo<XT>::load8(xr + (w_i64)m * k, xv);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[m] = fmaf(w[8 * r + i], xv[i], acc[m]);
            }
        }
    }

    for (int m = 0; m < m_cnt; ++m) {
        const float s = w_warp32_sum(acc[m]);
        if (lane == 0) y[(w_i64)(m0 + m) * n + col] = WIo<XT>::from_f(s);
    }
}

#define W8A16_TILE16_KERNEL(name, D, XT)                                                           \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        name(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y, int k,     \
             int n, int rows, w_i64 row_bytes) {                                                   \
        w8a16_fused_tile16<D, XT>(wq, x, y, k, n, rows, row_bytes);                                \
    }

// ---------------------------------------------------------------------------
// Single-Rounding Dual-Projection SwiGLU: y = silu(x @ W_gate^T) * (x @ W_up^T)
// Loads input vector x ONCE per K-block, accumulates gate and up projections in
// F32 registers, and evaluates silu(acc_g) * acc_u in F32 before a single store
// rounding — eliminating intermediate [M, 2*inter] HBM traffic and BF16 double-rounding.
// Grid: (ceil(n / 4), 1, 1), Block: (32, 4, 1), M <= M_MAX.
// ---------------------------------------------------------------------------
template <typename D, typename XT, int M_MAX>
W_DEV void w8a16_gate_up_silu_mv(const w_u8 *__restrict__ w_gate, const w_u8 *__restrict__ w_up,
                                 const XT *__restrict__ x, XT *__restrict__ y, const int k,
                                 const int n, const int rows, const w_i64 row_bytes) {
    const int lane = threadIdx.x;
    const int rz = threadIdx.y;
    const int col = blockIdx.x * 4 + rz;
    if (col >= n || rows <= 0 || rows > M_MAX) return;
    const int ntasks = (k / D::QK) * D::TASKS;
    const w_u8 *wg_row = w_gate + (w_i64)col * row_bytes;
    const w_u8 *wu_row = w_up   + (w_i64)col * row_bytes;

    float acc_g[M_MAX];
    float acc_u[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    for (int tk = lane; tk < ntasks; tk += 32) {
        const int b = tk / D::TASKS, t = tk % D::TASKS;
        float wg[8 * D::RUNS];
        float wu[8 * D::RUNS];
        int xo[D::RUNS];
        D::decode(wg_row + (w_i64)b * D::BYTES, t, wg, xo);
        D::decode(wu_row + (w_i64)b * D::BYTES, t, wu, xo);
#pragma unroll
        for (int r = 0; r < D::RUNS; ++r) {
            const XT *xr = x + b * D::QK + xo[r];
            for (int m = 0; m < rows; ++m) {
                float xv[8];
                WIo<XT>::load8(xr + (w_i64)m * k, xv);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc_g[m] = fmaf(wg[8 * r + i], xv[i], acc_g[m]);
                    acc_u[m] = fmaf(wu[8 * r + i], xv[i], acc_u[m]);
                }
            }
        }
    }

    for (int m = 0; m < rows; ++m) {
        const float g = w_warp32_sum(acc_g[m]);
        const float u = w_warp32_sum(acc_u[m]);
        if (lane == 0) {
            const float out = (g / (1.0f + expf(-g))) * u;
            y[(w_i64)m * n + col] = WIo<XT>::from_f(out);
        }
    }
}

#define W8A16_GATE_UP_SILU_KERNEL(name, D, XT, M_MAX)                                              \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        name(const w_u8 *__restrict__ w_gate, const w_u8 *__restrict__ w_up,                       \
             const XT *__restrict__ x, XT *__restrict__ y, int k, int n, int rows,                 \
             w_i64 row_bytes) {                                                                    \
        w8a16_gate_up_silu_mv<D, XT, M_MAX>(w_gate, w_up, x, y, k, n, rows, row_bytes);            \
    }
