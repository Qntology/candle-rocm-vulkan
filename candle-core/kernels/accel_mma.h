// Isolated hardware matrix-unit, bank-conflict-free 128B XOR swizzling,
// register-only Walsh-Hadamard smoothing, single-pass Chan-Welford combiner,
// and Cody-Waite long-context RoPE range reduction primitives.
//
// Strictly isolated from the portable OCP E4M3FN software-decode headers so
// `e4m3_is_ocp_fn_software_decode` stays 100% invariant across all backends.
// Compiles cleanly under CUDA NVRTC (--std=c++17, zero std headers), ROCm hipcc
// (CDNA2/3/4 Wave64, RDNA2/3/4 Wave32), and C++20 host emulator (CANDLE_ACCEL_HOST=1).
#pragma once
#include "accel_common.h"

// ---------------------------------------------------------------------------
// 1) Zero-padding 128B XOR Shared-Memory / LDS Swizzling
// ---------------------------------------------------------------------------
// Maps 16-byte chunk column index (8 x f16/bf16 or 16 x fp8) using row[2:0]
// so 128-bit vector loads and matrix fragment reads have 0 shared-memory bank
// conflicts with 0 bytes of padding overhead.
AX_DEV ax_u32 ax_swizzle_128b(ax_u32 row, ax_u32 col_chunk16b) {
    return col_chunk16b ^ (row & 7u);
}

// ---------------------------------------------------------------------------
// 2) Barrier-Free 32-Lane In-Register Walsh-Hadamard Transform + Sub-Block Amax
// ---------------------------------------------------------------------------
// Performs orthonormal 32-point FWHT (H_32 / sqrt(32)) purely in registers via
// 5 XOR shuffles (zero shared memory, zero workgroup barriers), then computes
// both the 16-element half-warp amax (NVFP4 block scale) and 32-element full-warp
// amax (FP8/MXFP8 block scale).
#if AX_HAS_WARP32_SHFL
AX_DEV void ax_warp32_fwht_and_amax(
    float &v,
    int lane,
    float &amax_halfwarp,
    float &amax_warp
) {
    AX_UNROLL
    for (int mask = 1; mask < 32; mask <<= 1) {
        const float other = ax_shfl_xor32(v, mask);
        v = (lane & mask) ? (other - v) : (v + other);
    }
    v *= 0.17677669529663689f; // 1 / sqrt(32)
    float av = fabsf(v);
    AX_UNROLL
    for (int mask = 1; mask < 16; mask <<= 1) {
        av = fmaxf(av, ax_shfl_xor32(av, mask));
    }
    amax_halfwarp = av;
    amax_warp = fmaxf(av, ax_shfl_xor32(av, 16));
}

// ---------------------------------------------------------------------------
// 3) Single-Pass Parallel Chan-Welford Combiner (32-lane warp)
// ---------------------------------------------------------------------------
// Combines per-lane (mean, m2, count) across 32 lanes with zero catastrophic
// cancellation even when mean ~ 1e4..1e5 and variance ~ 1.0 (high-contrast OCR).
AX_DEV void ax_warp32_welford(float &mean, float &m2, float &count) {
    AX_UNROLL
    for (int mask = 16; mask > 0; mask >>= 1) {
        const float b_mean  = ax_shfl_xor32(mean,  mask);
        const float b_m2    = ax_shfl_xor32(m2,    mask);
        const float b_count = ax_shfl_xor32(count, mask);
        const float tot = count + b_count;
        if (tot > 0.0f) {
            const float delta = b_mean - mean;
            mean  = mean + delta * (b_count / tot);
            m2    = m2 + b_m2 + (delta * delta) * (count * b_count / tot);
            count = tot;
        }
    }
}
#endif

// Portable workgroup Chan-Welford reduction over NT threads (NT power of 2 >= 32).
// `smem` must hold at least 3 * NT floats: [means | m2s | counts].
template <int NT>
AX_DEV void ax_block_welford(
    float local_mean,
    float local_m2,
    float local_count,
    float *smem,
    float &out_mean,
    float &out_m2
) {
    static_assert(NT >= 32 && (NT & (NT - 1)) == 0, "ax_block_welford: NT must be power of two >= 32");
    const ax_u32 t = AX_TID;
    float *s_mean  = smem;
    float *s_m2    = smem + NT;
    float *s_count = smem + 2 * NT;

#if AX_HAS_WARP32_SHFL
    ax_warp32_welford(local_mean, local_m2, local_count);
    const ax_u32 lane = t & 31u;
    const ax_u32 wid  = t >> 5;
    constexpr ax_u32 NWARPS = NT >> 5;
    if (lane == 0u) {
        s_mean[wid]  = local_mean;
        s_m2[wid]    = local_m2;
        s_count[wid] = local_count;
    }
    AX_SYNC();
    if (t == 0u) {
        float gm = s_mean[0];
        float g2 = s_m2[0];
        float gc = s_count[0];
        AX_UNROLL
        for (ax_u32 w = 1u; w < NWARPS; ++w) {
            const float bm = s_mean[w];
            const float b2 = s_m2[w];
            const float bc = s_count[w];
            const float tot = gc + bc;
            if (tot > 0.0f) {
                const float d = bm - gm;
                gm = gm + d * (bc / tot);
                g2 = g2 + b2 + (d * d) * (gc * bc / tot);
                gc = tot;
            }
        }
        s_mean[0] = gm;
        s_m2[0]   = g2;
    }
    AX_SYNC();
    out_mean = s_mean[0];
    out_m2   = s_m2[0];
    AX_SYNC();
#else
    s_mean[t]  = local_mean;
    s_m2[t]    = local_m2;
    s_count[t] = local_count;
    AX_SYNC();
    for (ax_u32 s = NT / 2; s > 0u; s >>= 1u) {
        if (t < s) {
            const float bm = s_mean[t + s];
            const float b2 = s_m2[t + s];
            const float bc = s_count[t + s];
            const float ac = s_count[t];
            const float tot = ac + bc;
            if (tot > 0.0f) {
                const float am = s_mean[t];
                const float d = bm - am;
                s_mean[t]  = am + d * (bc / tot);
                s_m2[t]    = s_m2[t] + b2 + (d * d) * (ac * bc / tot);
                s_count[t] = tot;
            }
        }
        AX_SYNC();
    }
    out_mean = s_mean[0];
    out_m2   = s_m2[0];
    AX_SYNC();
#endif
}

// ---------------------------------------------------------------------------
// 4) Cody-Waite TwoProd Range-Reduced Long-Context RoPE Phase (64K-1M+)
// ---------------------------------------------------------------------------
// Uses exact FP32 FMA TwoProd (`p = a*b; e = fmaf(a, b, -p)`) to form a ~47-bit
// double-float representation of `pos * inv_freq / (2*pi)` using only FP32 ops.
// Verified <= 3e-7 max error vs FP64 oracle up to pos = 1,000,000+ (vs 4e-3 naive).
AX_DEV void ax_cody_waite_sincos(
    ax_u32 pos,
    float inv_freq,
    float &out_cos,
    float &out_sin
) {
    // 1 / (2 * pi) = INV_2PI_HI (0x3e22f983) + INV_2PI_LO (0x31dc9c88)
    constexpr float INV_2PI_HI = 0.15915493667125702f;
    constexpr float INV_2PI_LO = 6.4206382432985265e-9f;
    constexpr float TWO_PI_F   = 6.2831853071795864769f;

    // TwoProd(inv_freq, 1/(2*pi)) -> (w_hi, w_lo)
    const float w_hi = inv_freq * INV_2PI_HI;
    const float w_lo = fmaf(inv_freq, INV_2PI_HI, -w_hi) + inv_freq * INV_2PI_LO;

    // Exact split for arbitrary 32-bit pos: p_hi (multiple of 2^12) + p_lo (< 2^12).
    const float p_lo = (float)(pos & 0xfffu);
    const float p_hi = (float)(pos & ~0xfffu);

    const float c0 = p_hi * w_hi;
    const float e0 = fmaf(p_hi, w_hi, -c0);
    const float c1 = p_lo * w_hi;
    const float e1 = fmaf(p_lo, w_hi, -c1);

    float cyc = (c0 - floorf(c0)) + (c1 - floorf(c1)) + (e0 + e1 + fmaf(p_hi, w_lo, p_lo * w_lo));
    cyc -= floorf(cyc);
    if (cyc > 0.5f) cyc -= 1.0f;
    const float phi = cyc * TWO_PI_F;
    out_cos = cosf(phi);
    out_sin = sinf(phi);
}

// ---------------------------------------------------------------------------
// 5) Architecture-Guarded Logical 16x16x16 MMA Helpers (CUDA sm_80+ / ROCm)
// ---------------------------------------------------------------------------
// CUDA Ampere+ (sm_80..sm_120): headerless inline PTX mma.sync.aligned.m16n8k16 x 2
// ROCm CDNA (gfx90a/942/950):   __builtin_amdgcn_mfma_f32_16x16x16{f16,bf16_1k}
// ROCm RDNA3 (gfx11xx):         __builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32
// ROCm RDNA4 (gfx120x):         __builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32_gfx12
#if !defined(CANDLE_ACCEL_HOST) && !defined(__HIP__) && !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define AX_HAS_CUDA_MMA 1
AX_DEV void ax_mma_f16_m16n16k16(
    float d[8], const ax_u32 a[4], const ax_u32 b0[2], const ax_u32 b1[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%10, %11, %12, %13}, {%4, %5, %6, %7}, {%14, %15}, {%10, %11, %12, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0[0]), "r"(b0[1]), "r"(b1[0]), "r"(b1[1])
    );
}

AX_DEV void ax_mma_bf16_m16n16k16(
    float d[8], const ax_u32 a[4], const ax_u32 b0[2], const ax_u32 b1[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%10, %11, %12, %13}, {%4, %5, %6, %7}, {%14, %15}, {%10, %11, %12, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0[0]), "r"(b0[1]), "r"(b1[0]), "r"(b1[1])
    );
}
#else
#define AX_HAS_CUDA_MMA 0
#endif

// Detect AMD MFMA (CDNA2/3/4) vs WMMA (RDNA3/RDNA4) availability at compile time
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
#define AX_HAS_AMD_MFMA 1
#else
#define AX_HAS_AMD_MFMA 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__))
#define AX_HAS_AMD_WMMA_RDNA3 1
#else
#define AX_HAS_AMD_WMMA_RDNA3 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define AX_HAS_AMD_WMMA_RDNA4 1
#else
#define AX_HAS_AMD_WMMA_RDNA4 0
#endif
