// Portable prelude of the fused attention and TurboQuant kernels.
//
// The same sources build as
//   * HIP: hipcc at build time (candle-core/build.rs, `flash_attn.hip`, `turbo_quant.hip`);
//   * CUDA: NVRTC at run time (candle-core/src/cuda_backend/accel.rs). NVRTC has no standard
//     headers, so nothing here includes one on that path;
//   * plain C++ (CANDLE_ACCEL_HOST): every workgroup lane runs as a host thread, which lets
//     candle-core/tests/accel_kernels_host.rs check the kernels without a GPU.
//
// The kernels only use dynamic shared memory, `__syncthreads` and plain arithmetic: no warp
// intrinsics, so the wave32 (RDNA), wave64 (GCN / CDNA) and warp32 (NVIDIA) targets behave
// the same. Floating point formats are converted with integer code, bit exact with the CPU
// codec in candle-core/src/accel/formats.rs.
#pragma once

#if defined(CANDLE_ACCEL_HOST)
#include <math.h>
#include "accel_host_emu.h"
#define AX_KERNEL extern "C"
#define AX_BOUNDS(n)
#define AX_DEV static inline
#define AX_TID (candle_emu::tid)
#define AX_BID_X (candle_emu::bid[0])
#define AX_BID_Y (candle_emu::bid[1])
#define AX_BID_Z (candle_emu::bid[2])
#define AX_NBLK_X (candle_emu::nblk[0])
#define AX_SYNC() candle_emu::sync()
AX_DEV float *ax_smem() { return reinterpret_cast<float *>(candle_emu::smem); }
#elif defined(__HIP__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define AX_KERNEL extern "C" __global__
#define AX_BOUNDS(n) __launch_bounds__(n)
#define AX_DEV __device__ __forceinline__
#define AX_TID ((unsigned)threadIdx.x)
#define AX_BID_X ((unsigned)blockIdx.x)
#define AX_BID_Y ((unsigned)blockIdx.y)
#define AX_BID_Z ((unsigned)blockIdx.z)
#define AX_NBLK_X ((unsigned)gridDim.x)
#define AX_SYNC() __syncthreads()
AX_DEV float *ax_smem() {
    extern __shared__ float ax_dyn_smem[];
    return ax_dyn_smem;
}
#else // CUDA (NVRTC, nvcc, clang -x cuda)
#define AX_KERNEL extern "C" __global__
#define AX_BOUNDS(n) __launch_bounds__(n)
#define AX_DEV __device__ __forceinline__
#define AX_TID ((unsigned)threadIdx.x)
#define AX_BID_X ((unsigned)blockIdx.x)
#define AX_BID_Y ((unsigned)blockIdx.y)
#define AX_BID_Z ((unsigned)blockIdx.z)
#define AX_NBLK_X ((unsigned)gridDim.x)
#define AX_SYNC() __syncthreads()
AX_DEV float *ax_smem() {
    extern __shared__ float ax_dyn_smem[];
    return ax_dyn_smem;
}
#endif

// Full unrolling of the loops over register arrays (keeps them out of local memory).
#if defined(CANDLE_ACCEL_HOST)
#define AX_UNROLL
#else
#define AX_UNROLL _Pragma("unroll")
#endif

typedef unsigned int ax_u32;
typedef unsigned long long ax_u64;
typedef unsigned short ax_u16;
typedef unsigned char ax_u8;

// Storage types of the element dtypes (layout compatible with candle's f16 / bf16 / F8E4M3).
struct ax_f16 {
    ax_u16 b;
};
struct ax_bf16 {
    ax_u16 b;
};
struct ax_fp8 {
    ax_u8 b;
};

union ax_fu {
    float f;
    ax_u32 u;
};

AX_DEV float ax_u2f(ax_u32 u) {
    ax_fu c;
    c.u = u;
    return c.f;
}

AX_DEV ax_u32 ax_f2u(float f) {
    ax_fu c;
    c.f = f;
    return c.u;
}

#define AX_INF ax_u2f(0x7f800000u)

AX_DEV long long ax_min_ll(long long a, long long b) { return a < b ? a : b; }
AX_DEV long long ax_max_ll(long long a, long long b) { return a > b ? a : b; }

// ---------------------------------------------------------------------------
// Paged K/V (block table passed by value, keep in sync with accel::attention::PageTable)
// ---------------------------------------------------------------------------

#define AX_MAX_PAGES 128

// Block i holds rows[i] keys: element / word pointers k[i] / v[i] of its (batch 0, kv head 0,
// row 0), laid out (batch, kv head, capacity, row) with a row stride shared by every block,
// head stride k_cap[i] (v_cap[i]) rows and batch stride Hkv head strides. The keys of the call
// are the concatenation of the blocks in order.
struct ax_pages {
    ax_u64 k[AX_MAX_PAGES];
    ax_u64 v[AX_MAX_PAGES];
    ax_u32 rows[AX_MAX_PAGES];
    ax_u32 k_cap[AX_MAX_PAGES];
    ax_u32 v_cap[AX_MAX_PAGES];
    ax_u32 n;
    ax_u32 pad;
};

// ---------------------------------------------------------------------------
// 16 bit floats
// ---------------------------------------------------------------------------

AX_DEV float ax_h2f(ax_u16 h) {
    const ax_u32 s = ((ax_u32)(h & 0x8000u)) << 16;
    const ax_u32 e = (h >> 10) & 0x1fu;
    const ax_u32 m = h & 0x3ffu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f; // m * 2^-24
        return s ? -v : v;
    }
    if (e == 31) return ax_u2f(s | 0x7f800000u | (m << 13));
    return ax_u2f(s | ((e + 112u) << 23) | (m << 13));
}

AX_DEV ax_u16 ax_f2h(float f) {
    const ax_u32 x = ax_f2u(f);
    const ax_u32 s = (x >> 16) & 0x8000u;
    const ax_u32 ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) return (ax_u16)(s | 0x7e00u);
    if (ax >= 0x477ff000u) return (ax_u16)(s | 0x7c00u); // rounds to infinity
    if (ax < 0x38800000u) {                               // half subnormal (or zero)
        const float v = rintf(ax_u2f(ax) * 16777216.0f);  // units of 2^-24
        return (ax_u16)(s | (ax_u32)v);
    }
    const ax_u32 r = ax + 0xfffu + ((ax >> 13) & 1u);
    return (ax_u16)(s | ((r >> 13) - (112u << 10)));
}

AX_DEV float ax_bf2f(ax_u16 b) { return ax_u2f(((ax_u32)b) << 16); }

AX_DEV ax_u16 ax_f2bf(float f) {
    ax_u32 u = ax_f2u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ax_u16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ax_u16)(u >> 16);
}

// ---------------------------------------------------------------------------
// 8 / 4 bit floats (OCP E4M3FN, E2M1, E8M0)
// ---------------------------------------------------------------------------

AX_DEV float ax_e4m3_to_f(ax_u32 c) {
    const ax_u32 e = (c >> 3) & 0xfu;
    const ax_u32 m = c & 7u;
    if (e == 15u && m == 7u) return ax_u2f(0x7fc00000u);
    const float v = e == 0 ? (float)m * 0.001953125f : ax_u2f(((e + 120u) << 23) | (m << 20));
    return (c & 0x80u) ? -v : v;
}

// Branch-free ax_e4m3_to_f for the decode loops: the same value, bit for bit, for all 256 codes
// (normals rebias the exponent with one add, subnormals use the same m * 2^-9 product, and the
// NaN code keeps the positive quiet NaN of ax_e4m3_to_f). No float subnormal is formed, so the
// result does not depend on flush-to-zero settings.
AX_DEV float ax_e4m3_to_f_fast(ax_u32 c) {
    const ax_u32 mag = c & 0x7fu;
    const float nrm = ax_u2f((mag << 20) + (120u << 23));
    const float sub = (float)mag * 0.001953125f;
    float v = mag >= 8u ? nrm : sub;
    v = (c & 0x80u) ? -v : v;
    return mag == 0x7fu ? ax_u2f(0x7fc00000u) : v;
}

AX_DEV ax_u32 ax_f_to_e4m3(float f) {
    const ax_u32 x = ax_f2u(f);
    const ax_u32 s = (x >> 24) & 0x80u;
    const ax_u32 ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) return s | 0x7fu;
    const float a = ax_u2f(ax);
    if (a >= 448.0f) return s | 0x7eu;
    if (a < 0.015625f) return s | (ax_u32)rintf(a * 512.0f);
    const ax_u32 r = ax + 0x7ffffu + ((ax >> 20) & 1u);
    ax_u32 code = (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u);
    if (code > 0x7eu) code = 0x7eu;
    return s | code;
}

AX_DEV float ax_fp4_to_f(ax_u32 c) {
    const ax_u32 e = (c >> 1) & 3u;
    const ax_u32 m = c & 1u;
    const float v = e == 0 ? 0.5f * (float)m : ax_u2f(((e + 126u) << 23) | (m << 22));
    return (c & 8u) ? -v : v;
}

AX_DEV ax_u32 ax_f_to_fp4(float f) {
    const ax_u32 s = (ax_f2u(f) >> 28) & 8u;
    const float a = fabsf(f);
    ax_u32 c;
    if (!(a > 0.25f)) c = 0; // also NaN
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return s | c;
}

AX_DEV float ax_e8m0_to_f(ax_u32 c) { return c == 0 ? ax_u2f(0x00400000u) : ax_u2f(c << 23); }

// OCP MX shared exponent of a block with maximum magnitude `amax` (emax_elem: 2 for E2M1, 8
// for E4M3), as the E8M0 code.
AX_DEV ax_u32 ax_mx_scale_code(float amax, int emax_elem) {
    int e = -127;
    const ax_u32 bits = ax_f2u(amax);
    const int be = (int)((bits >> 23) & 0xffu);
    if (amax > 0.0f && be != 0xff && be != 0) e = be - 127 - emax_elem;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (ax_u32)(e + 127);
}

// ---------------------------------------------------------------------------
// Element loads / stores
// ---------------------------------------------------------------------------

AX_DEV float ax_load(const float *p, ax_u64 i) { return p[i]; }
AX_DEV float ax_load(const ax_f16 *p, ax_u64 i) { return ax_h2f(p[i].b); }
AX_DEV float ax_load(const ax_bf16 *p, ax_u64 i) { return ax_bf2f(p[i].b); }
AX_DEV float ax_load(const ax_fp8 *p, ax_u64 i) { return ax_e4m3_to_f(p[i].b); }

AX_DEV void ax_store(float *p, ax_u64 i, float v) { p[i] = v; }
AX_DEV void ax_store(ax_f16 *p, ax_u64 i, float v) { p[i].b = ax_f2h(v); }
AX_DEV void ax_store(ax_bf16 *p, ax_u64 i, float v) { p[i].b = ax_f2bf(v); }

// ---------------------------------------------------------------------------
// Workgroup reductions (NT threads, NT a power of two, `red` holds NT floats)
// ---------------------------------------------------------------------------

template <int NT>
AX_DEV float ax_block_sum(float v, float *red) {
    const ax_u32 t = AX_TID;
    red[t] = v;
    AX_SYNC();
    for (ax_u32 s = NT / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = red[t] + red[t + s];
        AX_SYNC();
    }
    const float r = red[0];
    AX_SYNC();
    return r;
}

template <int NT>
AX_DEV float ax_block_max(float v, float *red) {
    const ax_u32 t = AX_TID;
    red[t] = v;
    AX_SYNC();
    for (ax_u32 s = NT / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        AX_SYNC();
    }
    const float r = red[0];
    AX_SYNC();
    return r;
}
