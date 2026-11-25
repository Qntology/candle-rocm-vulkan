// Fused decoder kernels: ResidualRmsNorm, SiluMul, and RopeKvWrite.
//
// Built from one source by:
//   * ROCm / HIP: hipcc at build time (kernels/decoder_ops.hip)
//   * CUDA: NVRTC at run time (accel/sources.rs -> cuda_backend/decoder_ops.rs, --std=c++17, no fast-math)
//   * Host emulator: plain C++20 (CANDLE_ACCEL_HOST=1)
//
// Uses only accel_common.h primitives (dynamic shared memory + AX_SYNC, no warp shuffles,
// no standard headers on CUDA NVRTC) so NVIDIA warp32, AMD wave32/wave64, and host threads
// behave identically. All arithmetic promotes to F32 and rounds once on store.
#pragma once
#include "accel_common.h"

#define DEC_NT 256u
#define DEC_MR_NT 128u
#define DEC_ROWS_PER_WG 4u

// Flags for res_rmsnorm_*
#define RES_RMS_PLUS_ONE_WEIGHT 1u
#define RES_RMS_WRITE_RESIDUAL  2u

// Flags for rope_kv_write_* and rope_kv_write_indirect_*
#define ROPE_MODE_NEOX          0u
#define ROPE_MODE_ADJACENT      1u
#define ROPE_NORM_PLUS_ONE      2u
// When set on rope_kv_write_indirect_*, cos_tab holds `inv_freq[half_rot]` and
// sin_tab is ignored; exact TwoProd Cody-Waite range-reduced cos/sin are computed
// on-the-fly from `(d_rope_pos[b] + t_idx) * inv_freq[i]` (<= 3e-7 error at 1M pos).
#define ROPE_CODY_WAITE_ONTHEFLY 4u

// ---------------------------------------------------------------------------
// 0) Exact FP32 FMA TwoProd Cody-Waite range-reduced sin/cos (64K - 1M+ pos)
// ---------------------------------------------------------------------------
AX_DEV void dec_cody_waite_sincos(ax_u32 pos, float inv_freq, float &out_cos, float &out_sin) {
    constexpr float INV_2PI_HI = 0.15915493667125702f;
    constexpr float INV_2PI_LO = 6.4206382432985265e-9f;
    constexpr float TWO_PI_F   = 6.2831853071795864769f;

    const float w_hi = inv_freq * INV_2PI_HI;
    const float w_lo = fmaf(inv_freq, INV_2PI_HI, -w_hi) + inv_freq * INV_2PI_LO;

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
// 1) Residual + RMSNorm (1 workgroup of DEC_NT threads per row, 4-way pairwise sum)
//
// Computes:
//   r[col] = x[col] + (delta ? delta[col] : 0)
//   if write_res: r_out[col] = cast<TX>(r[col])
//   y_out[col] = cast<TX>((r_eff[col] / sqrt(mean(r^2) + eps)) * (plus_one ? 1 + w[col] : w[col]))
//
// Safe for in-place r_out == x because all loads of a column happen before its store in loop 1
// and ax_block_sum synchronizes the workgroup before loop 2 reads r_out.
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void res_rmsnorm_impl(
    const TX *x,
    const TX *delta,
    const float *w,
    TX *r_out,
    TX *y_out,
    ax_u32 rows,
    ax_u32 D,
    float eps,
    ax_u32 flags
) {
    const ax_u32 row = AX_BID_X;
    if (row >= rows || D == 0u) return;

    float *smem = ax_smem();
    const ax_u64 base = (ax_u64)row * (ax_u64)D;
    const bool write_res = ((flags & RES_RMS_WRITE_RESIDUAL) != 0u) && (r_out != nullptr);
    const bool plus_one = (flags & RES_RMS_PLUS_ONE_WEIGHT) != 0u;

    // 4-way pairwise accumulators reduce sequential rounding drift for wide hidden dims (D >= 8192).
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    ax_u32 step_idx = 0u;
    for (ax_u32 col = AX_TID; col < D; col += DEC_NT, ++step_idx) {
        float v = ax_load(x, base + col);
        if (delta != nullptr) {
            v += ax_load(delta, base + col);
        }
        if (write_res) {
            ax_store(r_out, base + col, v);
            // Re-read the rounded residual value for exact consistency with unfused
            // (r = cast(x + delta); y = rmsnorm(r)).
            v = ax_load(r_out, base + col);
        }
        switch (step_idx & 3u) {
            case 0u: sq0 = fmaf(v, v, sq0); break;
            case 1u: sq1 = fmaf(v, v, sq1); break;
            case 2u: sq2 = fmaf(v, v, sq2); break;
            default: sq3 = fmaf(v, v, sq3); break;
        }
    }
    const float local_sq = (sq0 + sq1) + (sq2 + sq3);

    const float total_sq = ax_block_sum<DEC_NT>(local_sq, smem);
    const float inv_rms = 1.0f / sqrtf(total_sq / (float)D + eps);

    for (ax_u32 col = AX_TID; col < D; col += DEC_NT) {
        float v;
        if (write_res) {
            v = ax_load(r_out, base + col);
        } else {
            v = ax_load(x, base + col);
            if (delta != nullptr) {
                v += ax_load(delta, base + col);
            }
        }
        const float wf = plus_one ? (1.0f + w[col]) : w[col];
        ax_store(y_out, base + col, (v * inv_rms) * wf);
    }
}

// ---------------------------------------------------------------------------
// 1b) Barrier-Free Multi-Row Residual + RMSNorm (4 rows/WG x 32 lanes = 128 threads)
// For prefill batches (M >= 32): each 32-lane warp owns 1 row independently with
// 0 workgroup barriers and 4-way pairwise sum accumulation.
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void res_rmsnorm_mr_impl(
    const TX *x,
    const TX *delta,
    const float *w,
    TX *r_out,
    TX *y_out,
    ax_u32 rows,
    ax_u32 D,
    float eps,
    ax_u32 flags
) {
    const ax_u32 wid = AX_TID >> 5;
    const ax_u32 lane = AX_TID & 31u;
    const ax_u32 row = AX_BID_X * DEC_ROWS_PER_WG + wid;
    const bool active = (row < rows) && (D > 0u);

    const bool write_res = ((flags & RES_RMS_WRITE_RESIDUAL) != 0u) && (r_out != nullptr);
    const bool plus_one = (flags & RES_RMS_PLUS_ONE_WEIGHT) != 0u;
    const ax_u64 base = (ax_u64)row * (ax_u64)D;

    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    if (active) {
        ax_u32 step_idx = 0u;
        for (ax_u32 col = lane; col < D; col += 32u, ++step_idx) {
            float v = ax_load(x, base + col);
            if (delta != nullptr) {
                v += ax_load(delta, base + col);
            }
            if (write_res) {
                ax_store(r_out, base + col, v);
                v = ax_load(r_out, base + col);
            }
            switch (step_idx & 3u) {
                case 0u: sq0 = fmaf(v, v, sq0); break;
                case 1u: sq1 = fmaf(v, v, sq1); break;
                case 2u: sq2 = fmaf(v, v, sq2); break;
                default: sq3 = fmaf(v, v, sq3); break;
            }
        }
    }
    const float local_sq = (sq0 + sq1) + (sq2 + sq3);

#if AX_HAS_WARP32_SHFL
    const float total_sq = ax_warp32_sum(local_sq);
#else
    float *smem = ax_smem();
    float *wbuf = smem + wid * 32u;
    wbuf[lane] = local_sq;
    AX_SYNC();
    for (ax_u32 s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) wbuf[lane] += wbuf[lane + s];
        AX_SYNC();
    }
    const float total_sq = wbuf[0];
    AX_SYNC();
#endif

    if (!active) return;
    const float inv_rms = 1.0f / sqrtf(total_sq / (float)D + eps);
    for (ax_u32 col = lane; col < D; col += 32u) {
        float v;
        if (write_res) {
            v = ax_load(r_out, base + col);
        } else {
            v = ax_load(x, base + col);
            if (delta != nullptr) {
                v += ax_load(delta, base + col);
            }
        }
        const float wf = plus_one ? (1.0f + w[col]) : w[col];
        ax_store(y_out, base + col, (v * inv_rms) * wf);
    }
}

// ---------------------------------------------------------------------------
// 1c) Outlier-Free Online Hadamard-32 Smoothing + OCP FP8 Activation Quantizer
// Fuses RMSNorm (or SiLU*Mul) + orthonormal H_32/sqrt(32) butterfly + per-32-block
// OCP E4M3FN dynamic quantization (`scales[row, blk] = amax / 448.0f`) in 1 pass.
// Grid: (rows, 1, 1), Block: (DEC_NT, 1, 1), D must be a multiple of 32.
// ---------------------------------------------------------------------------
AX_DEV void dec_block32_fwht_and_fp8(
    float v,
    ax_u32 lane,
    float *smem_w,
    ax_fp8 *y_fp8,
    float *y_scales,
    ax_u64 elem_off,
    ax_u64 blk_off
) {
#if AX_HAS_WARP32_SHFL
    AX_UNROLL
    for (int mask = 1; mask < 32; mask <<= 1) {
        const float other = ax_shfl_xor32(v, mask);
        v = (lane & (ax_u32)mask) ? (other - v) : (v + other);
    }
    v *= 0.17677669529663689f; // 1 / sqrt(32)
    float amax = ax_warp32_max(fabsf(v));
#else
    smem_w[lane] = v;
    AX_SYNC();
    for (ax_u32 mask = 1u; mask < 32u; mask <<= 1u) {
        const float a = smem_w[lane];
        const float b = smem_w[lane ^ mask];
        AX_SYNC();
        smem_w[lane] = (lane & mask) ? (b - a) : (a + b);
        AX_SYNC();
    }
    v = smem_w[lane] * 0.17677669529663689f;
    smem_w[lane] = fabsf(v);
    AX_SYNC();
    for (ax_u32 s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) smem_w[lane] = fmaxf(smem_w[lane], smem_w[lane + s]);
        AX_SYNC();
    }
    const float amax = smem_w[0];
    AX_SYNC();
#endif
    const float scale = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
    const float inv_scale = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
    if (lane == 0u) {
        y_scales[blk_off] = scale;
    }
    y_fp8[elem_off].b = (ax_u8)ax_f_to_e4m3(v * inv_scale);
}

template <typename TX>
AX_DEV void res_rmsnorm_hadamard_fp8_impl(
    const TX *x,
    const TX *delta,
    const float *w,
    TX *r_out,
    ax_fp8 *y_fp8,
    float *y_scales,
    ax_u32 rows,
    ax_u32 D,
    float eps,
    ax_u32 flags
) {
    const ax_u32 row = AX_BID_X;
    if (row >= rows || D == 0u || (D & 31u) != 0u) return;

    float *smem = ax_smem();
    const ax_u64 base = (ax_u64)row * (ax_u64)D;
    const ax_u64 n_blks = (ax_u64)(D >> 5);
    const bool write_res = ((flags & RES_RMS_WRITE_RESIDUAL) != 0u) && (r_out != nullptr);
    const bool plus_one = (flags & RES_RMS_PLUS_ONE_WEIGHT) != 0u;

    float sq0 = 0.0f, sq1 = 0.0f;
    ax_u32 step = 0u;
    for (ax_u32 col = AX_TID; col < D; col += DEC_NT, ++step) {
        float v = ax_load(x, base + col);
        if (delta != nullptr) v += ax_load(delta, base + col);
        if (write_res) {
            ax_store(r_out, base + col, v);
            v = ax_load(r_out, base + col);
        }
        if (step & 1u) sq1 = fmaf(v, v, sq1); else sq0 = fmaf(v, v, sq0);
    }
    const float total_sq = ax_block_sum<DEC_NT>(sq0 + sq1, smem);
    const float inv_rms = 1.0f / sqrtf(total_sq / (float)D + eps);

    const ax_u32 wid = AX_TID >> 5;
    const ax_u32 lane = AX_TID & 31u;
    constexpr ax_u32 NWARPS = DEC_NT >> 5;
    // Iterate over full 32-element blocks; on host without shfl, keep warps in lockstep.
    const ax_u32 n_iters = ((ax_u32)n_blks + NWARPS - 1u) / NWARPS;
    for (ax_u32 it = 0u; it < n_iters; ++it) {
        const ax_u32 blk = it * NWARPS + wid;
        const bool ok = blk < (ax_u32)n_blks;
        float y = 0.0f;
        const ax_u32 col = blk * 32u + lane;
        if (ok) {
            float v = write_res ? ax_load(r_out, base + col)
                                : (ax_load(x, base + col) + (delta != nullptr ? ax_load(delta, base + col) : 0.0f));
            const float wf = plus_one ? (1.0f + w[col]) : w[col];
            y = (v * inv_rms) * wf;
        }
#if AX_HAS_WARP32_SHFL
        if (ok) {
            dec_block32_fwht_and_fp8(
                y, lane, smem + wid * 32u, y_fp8, y_scales,
                base + col, (ax_u64)row * n_blks + blk);
        }
#else
        // Host emulator without shfl needs all threads at AX_SYNC().
        smem[wid * 32u + lane] = y;
        AX_SYNC();
        for (ax_u32 mask = 1u; mask < 32u; mask <<= 1u) {
            const float a = smem[wid * 32u + lane];
            const float b = smem[wid * 32u + (lane ^ mask)];
            AX_SYNC();
            smem[wid * 32u + lane] = (lane & mask) ? (b - a) : (a + b);
            AX_SYNC();
        }
        const float yr = smem[wid * 32u + lane] * 0.17677669529663689f;
        smem[wid * 32u + lane] = fabsf(yr);
        AX_SYNC();
        for (ax_u32 s = 16u; s > 0u; s >>= 1u) {
            if (lane < s) smem[wid * 32u + lane] = fmaxf(smem[wid * 32u + lane], smem[wid * 32u + lane + s]);
            AX_SYNC();
        }
        const float amax = smem[wid * 32u];
        AX_SYNC();
        if (ok) {
            const float scale = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
            const float inv_scale = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
            if (lane == 0u) y_scales[(ax_u64)row * n_blks + blk] = scale;
            y_fp8[base + col].b = (ax_u8)ax_f_to_e4m3(yr * inv_scale);
        }
#endif
    }
}

template <typename TX>
AX_DEV void silu_mul_hadamard_fp8_impl(
    const TX *gate,
    ax_u64 gate_row_stride,
    const TX *up,
    ax_u64 up_row_stride,
    ax_fp8 *y_fp8,
    float *y_scales,
    ax_u32 rows,
    ax_u32 inter_dim
) {
    const ax_u32 row = AX_BID_X;
    if (row >= rows || inter_dim == 0u || (inter_dim & 31u) != 0u) return;

    float *smem = ax_smem();
    const ax_u64 n_blks = (ax_u64)(inter_dim >> 5);
    const ax_u64 base = (ax_u64)row * (ax_u64)inter_dim;
    const ax_u32 wid = AX_TID >> 5;
    const ax_u32 lane = AX_TID & 31u;
    constexpr ax_u32 NWARPS = DEC_NT >> 5;
    const ax_u32 n_iters = ((ax_u32)n_blks + NWARPS - 1u) / NWARPS;

    for (ax_u32 it = 0u; it < n_iters; ++it) {
        const ax_u32 blk = it * NWARPS + wid;
        const bool ok = blk < (ax_u32)n_blks;
        const ax_u32 col = blk * 32u + lane;
        float y = 0.0f;
        if (ok) {
            const float g = ax_load(gate, (ax_u64)row * gate_row_stride + col);
            const float u = ax_load(up,   (ax_u64)row * up_row_stride + col);
            y = (g / (1.0f + expf(-g))) * u;
        }
#if AX_HAS_WARP32_SHFL
        if (ok) {
            dec_block32_fwht_and_fp8(
                y, lane, smem + wid * 32u, y_fp8, y_scales,
                base + col, (ax_u64)row * n_blks + blk);
        }
#else
        smem[wid * 32u + lane] = y;
        AX_SYNC();
        for (ax_u32 mask = 1u; mask < 32u; mask <<= 1u) {
            const float a = smem[wid * 32u + lane];
            const float b = smem[wid * 32u + (lane ^ mask)];
            AX_SYNC();
            smem[wid * 32u + lane] = (lane & mask) ? (b - a) : (a + b);
            AX_SYNC();
        }
        const float yr = smem[wid * 32u + lane] * 0.17677669529663689f;
        smem[wid * 32u + lane] = fabsf(yr);
        AX_SYNC();
        for (ax_u32 s = 16u; s > 0u; s >>= 1u) {
            if (lane < s) smem[wid * 32u + lane] = fmaxf(smem[wid * 32u + lane], smem[wid * 32u + lane + s]);
            AX_SYNC();
        }
        const float amax = smem[wid * 32u];
        AX_SYNC();
        if (ok) {
            const float scale = (amax > 0.0f) ? (amax * (1.0f / 448.0f)) : 1.0f;
            const float inv_scale = (amax > 0.0f) ? (448.0f / amax) : 0.0f;
            if (lane == 0u) y_scales[(ax_u64)row * n_blks + blk] = scale;
            y_fp8[base + col].b = (ax_u8)ax_f_to_e4m3(yr * inv_scale);
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// 2) SwiGLU SiLU * Mul (supports both separate tensors and zero-copy strided gate_up views)
//
// Computes:
//   out[r * inter_dim + c] = cast<TX>(silu(gate[r * gate_stride + c]) * up[r * up_stride + c])
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void silu_mul_impl(
    const TX *gate,
    ax_u64 gate_row_stride,
    const TX *up,
    ax_u64 up_row_stride,
    TX *out,
    ax_u32 rows,
    ax_u32 inter_dim
) {
    const ax_u64 total = (ax_u64)rows * (ax_u64)inter_dim;
    const ax_u64 step = (ax_u64)AX_NBLK_X * (ax_u64)DEC_NT;
    for (ax_u64 idx = (ax_u64)AX_BID_X * DEC_NT + AX_TID; idx < total; idx += step) {
        const ax_u64 r = idx / (ax_u64)inter_dim;
        const ax_u64 c = idx - r * (ax_u64)inter_dim;
        const float g = ax_load(gate, r * gate_row_stride + c);
        const float u = ax_load(up, r * up_row_stride + c);
        const float silu_g = g / (1.0f + expf(-g));
        ax_store(out, idx, silu_g * u);
    }
}

// ---------------------------------------------------------------------------
// 3) Fused QK Head RMSNorm + Partial/Full 1D/3D-M RoPE + KV Cache Direct Write
//
// Grid: (Hq + Hkv, B * T, 1), Block: (DEC_NT, 1, 1)
// Dynamic shared memory: (DEC_NT + D) * sizeof(float)
//
// Supports arbitrary BSHD or BHSD layouts on Q_in, Q_out, K_in, V_in, K_cache, V_cache
// via element strides (contiguous last dim D), optional Q/K per-head RMSNorm (standard or 1+w),
// partial rotary_dim <= D, NeoX or adjacent-pair RoPE, and both 1D RoPE (cos_sb=0) and
// 3D M-RoPE (cos_sb = T * half_rot). Safe for in-place q_out == q_in.
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void rope_kv_write_impl(
    const TX *q_in,
    TX *q_out,
    const TX *k_in,
    const TX *v_in,
    const float *q_norm_w,
    const float *k_norm_w,
    const float *cos_tab,
    const float *sin_tab,
    TX *k_cache,
    TX *v_cache,
    ax_u32 B,
    ax_u32 T,
    ax_u32 Hq,
    ax_u32 Hkv,
    ax_u32 D,
    ax_u32 rotary_dim,
    ax_u32 past_len,
    ax_u64 q_sb,  ax_u64 q_sh,  ax_u64 q_st,
    ax_u64 qo_sb, ax_u64 qo_sh, ax_u64 qo_st,
    ax_u64 k_sb,  ax_u64 k_sh,  ax_u64 k_st,
    ax_u64 v_sb,  ax_u64 v_sh,  ax_u64 v_st,
    ax_u64 kc_sb, ax_u64 kc_sh, ax_u64 kc_ss,
    ax_u64 cos_sb, ax_u64 cos_st,
    float eps,
    ax_u32 flags
) {
    const ax_u32 head_id = AX_BID_X;
    const ax_u32 bt = AX_BID_Y;
    if (bt >= B * T || head_id >= Hq + Hkv || D == 0u) return;

    float *smem = ax_smem();
    float *hbuf = smem + DEC_NT;
    const ax_u32 tid = AX_TID;
    const bool is_q = head_id < Hq;
    const ax_u32 h = is_q ? head_id : (head_id - Hq);
    const ax_u32 b_idx = bt / T;
    const ax_u32 t_idx = bt - b_idx * T;
    const float *nw = is_q ? q_norm_w : k_norm_w;
    const bool plus_one = (flags & ROPE_NORM_PLUS_ONE) != 0u;

    const ax_u64 src_off = is_q
        ? ((ax_u64)b_idx * q_sb + (ax_u64)h * q_sh + (ax_u64)t_idx * q_st)
        : ((ax_u64)b_idx * k_sb + (ax_u64)h * k_sh + (ax_u64)t_idx * k_st);
    const TX *src = is_q ? q_in : k_in;

    float sq = 0.0f;
    for (ax_u32 d = tid; d < D; d += DEC_NT) {
        const float v = ax_load(src, src_off + d);
        hbuf[d] = v;
        if (nw != nullptr) {
            sq = fmaf(v, v, sq);
        }
    }

    if (nw != nullptr) {
        const float total_sq = ax_block_sum<DEC_NT>(sq, smem);
        const float inv = 1.0f / sqrtf(total_sq / (float)D + eps);
        for (ax_u32 d = tid; d < D; d += DEC_NT) {
            const float wf = plus_one ? (1.0f + nw[d]) : nw[d];
            hbuf[d] = (hbuf[d] * inv) * wf;
        }
    }
    AX_SYNC();

    const ax_u32 half_rot = rotary_dim >> 1;
    const ax_u64 cs_off = (ax_u64)b_idx * cos_sb + (ax_u64)t_idx * cos_st;
    const float *cos_row = cos_tab + cs_off;
    const float *sin_row = sin_tab + cs_off;

    const ax_u64 dst_off = is_q
        ? ((ax_u64)b_idx * qo_sb + (ax_u64)h * qo_sh + (ax_u64)t_idx * qo_st)
        : ((ax_u64)b_idx * kc_sb + (ax_u64)h * kc_sh + (ax_u64)(past_len + t_idx) * kc_ss);
    TX *dst = is_q ? q_out : k_cache;

    for (ax_u32 i = tid; i < half_rot; i += DEC_NT) {
        const float c = cos_row[i];
        const float s = sin_row[i];
        if ((flags & 1u) == ROPE_MODE_NEOX) {
            const float x0 = hbuf[i];
            const float x1 = hbuf[i + half_rot];
            ax_store(dst, dst_off + i,            x0 * c - x1 * s);
            ax_store(dst, dst_off + i + half_rot, x1 * c + x0 * s);
        } else {
            const float x0 = hbuf[2u * i];
            const float x1 = hbuf[2u * i + 1u];
            ax_store(dst, dst_off + 2u * i,      x0 * c - x1 * s);
            ax_store(dst, dst_off + 2u * i + 1u, x1 * c + x0 * s);
        }
    }
    for (ax_u32 d = rotary_dim + tid; d < D; d += DEC_NT) {
        ax_store(dst, dst_off + d, hbuf[d]);
    }

    // K-head workers also write the matching V slice directly into v_cache.
    if (!is_q && v_in != nullptr && v_cache != nullptr) {
        const ax_u64 v_off = (ax_u64)b_idx * v_sb + (ax_u64)h * v_sh + (ax_u64)t_idx * v_st;
        for (ax_u32 d = tid; d < D; d += DEC_NT) {
            v_cache[dst_off + d] = v_in[v_off + d];
        }
    }
}

// ---------------------------------------------------------------------------
// 3b) Graph-Safe Indirect RoPE + Cody-Waite Long-Context + YaRN mscale + KV Write
//
// Reads physical KV write offset from `d_kv_pos[b]` (or `d_kv_pos[0]` if stride=0)
// and RoPE phase position from `d_rope_pos[b]` (supporting 1D collapsed 3D M-RoPE
// `past_len + delta_mrope` and Delta-RoPE offsets with zero host scalar updates).
// If `flags & ROPE_CODY_WAITE_ONTHEFLY`: `cos_tab` holds `inv_freq[half_rot]` and
// exact TwoProd Cody-Waite cos/sin are computed on-the-fly for pos up to 1M+.
// `mscale` (YaRN attention temperature scale, 1.0f = no-op) is fused into Q store.
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void rope_kv_write_indirect_impl(
    const TX *q_in,
    TX *q_out,
    const TX *k_in,
    const TX *v_in,
    const float *q_norm_w,
    const float *k_norm_w,
    const float *cos_tab,
    const float *sin_tab,
    TX *k_cache,
    TX *v_cache,
    const ax_u32 *d_kv_pos,
    const ax_u32 *d_rope_pos,
    ax_u32 pos_stride,
    ax_u32 B,
    ax_u32 T,
    ax_u32 Hq,
    ax_u32 Hkv,
    ax_u32 D,
    ax_u32 rotary_dim,
    ax_u64 q_sb,  ax_u64 q_sh,  ax_u64 q_st,
    ax_u64 qo_sb, ax_u64 qo_sh, ax_u64 qo_st,
    ax_u64 k_sb,  ax_u64 k_sh,  ax_u64 k_st,
    ax_u64 v_sb,  ax_u64 v_sh,  ax_u64 v_st,
    ax_u64 kc_sb, ax_u64 kc_sh, ax_u64 kc_ss,
    ax_u64 cos_st,
    float eps,
    float mscale,
    ax_u32 flags
) {
    const ax_u32 head_id = AX_BID_X;
    const ax_u32 bt = AX_BID_Y;
    if (bt >= B * T || head_id >= Hq + Hkv || D == 0u) return;

    float *smem = ax_smem();
    float *hbuf = smem + DEC_NT;
    const ax_u32 tid = AX_TID;
    const bool is_q = head_id < Hq;
    const ax_u32 h = is_q ? head_id : (head_id - Hq);
    const ax_u32 b_idx = bt / T;
    const ax_u32 t_idx = bt - b_idx * T;
    const float *nw = is_q ? q_norm_w : k_norm_w;
    const bool plus_one = (flags & ROPE_NORM_PLUS_ONE) != 0u;
    const bool onthefly_cw = (flags & ROPE_CODY_WAITE_ONTHEFLY) != 0u;

    const ax_u32 kv_base_pos   = d_kv_pos[b_idx * pos_stride];
    const ax_u32 rope_base_pos = (d_rope_pos != nullptr) ? d_rope_pos[b_idx * pos_stride] : kv_base_pos;
    const ax_u32 cur_rope_pos  = rope_base_pos + t_idx;

    const ax_u64 src_off = is_q
        ? ((ax_u64)b_idx * q_sb + (ax_u64)h * q_sh + (ax_u64)t_idx * q_st)
        : ((ax_u64)b_idx * k_sb + (ax_u64)h * k_sh + (ax_u64)t_idx * k_st);
    const TX *src = is_q ? q_in : k_in;

    float sq = 0.0f;
    for (ax_u32 d = tid; d < D; d += DEC_NT) {
        const float v = ax_load(src, src_off + d);
        hbuf[d] = v;
        if (nw != nullptr) sq = fmaf(v, v, sq);
    }
    if (nw != nullptr) {
        const float total_sq = ax_block_sum<DEC_NT>(sq, smem);
        const float inv = 1.0f / sqrtf(total_sq / (float)D + eps);
        for (ax_u32 d = tid; d < D; d += DEC_NT) {
            const float wf = plus_one ? (1.0f + nw[d]) : nw[d];
            hbuf[d] = (hbuf[d] * inv) * wf;
        }
    }
    AX_SYNC();

    const ax_u32 half_rot = rotary_dim >> 1;
    const ax_u64 dst_off = is_q
        ? ((ax_u64)b_idx * qo_sb + (ax_u64)h * qo_sh + (ax_u64)t_idx * qo_st)
        : ((ax_u64)b_idx * kc_sb + (ax_u64)h * kc_sh + (ax_u64)(kv_base_pos + t_idx) * kc_ss);
    TX *dst = is_q ? q_out : k_cache;
    const float q_scale = is_q ? mscale : 1.0f;

    const ax_u64 cs_off = (ax_u64)cur_rope_pos * cos_st;
    for (ax_u32 i = tid; i < half_rot; i += DEC_NT) {
        float c, s;
        if (onthefly_cw) {
            dec_cody_waite_sincos(cur_rope_pos, cos_tab[i], c, s);
        } else {
            c = cos_tab[cs_off + i];
            s = sin_tab[cs_off + i];
        }
        if ((flags & 1u) == ROPE_MODE_NEOX) {
            const float x0 = hbuf[i];
            const float x1 = hbuf[i + half_rot];
            ax_store(dst, dst_off + i,            (x0 * c - x1 * s) * q_scale);
            ax_store(dst, dst_off + i + half_rot, (x1 * c + x0 * s) * q_scale);
        } else {
            const float x0 = hbuf[2u * i];
            const float x1 = hbuf[2u * i + 1u];
            ax_store(dst, dst_off + 2u * i,      (x0 * c - x1 * s) * q_scale);
            ax_store(dst, dst_off + 2u * i + 1u, (x1 * c + x0 * s) * q_scale);
        }
    }
    for (ax_u32 d = rotary_dim + tid; d < D; d += DEC_NT) {
        ax_store(dst, dst_off + d, hbuf[d] * q_scale);
    }

    if (!is_q && v_in != nullptr && v_cache != nullptr) {
        const ax_u64 v_off = (ax_u64)b_idx * v_sb + (ax_u64)h * v_sh + (ax_u64)t_idx * v_st;
        for (ax_u32 d = tid; d < D; d += DEC_NT) {
            v_cache[dst_off + d] = v_in[v_off + d];
        }
    }
}

#define DEC_OPS_INSTANTIATE(SFX, TX) \
AX_KERNEL void AX_BOUNDS(DEC_NT) res_rmsnorm_##SFX( \
    const TX *x, const TX *delta, const float *w, TX *r_out, TX *y_out, \
    ax_u32 rows, ax_u32 D, float eps, ax_u32 flags) { \
    res_rmsnorm_impl<TX>(x, delta, w, r_out, y_out, rows, D, eps, flags); \
} \
AX_KERNEL void AX_BOUNDS(DEC_MR_NT) res_rmsnorm_mr_##SFX( \
    const TX *x, const TX *delta, const float *w, TX *r_out, TX *y_out, \
    ax_u32 rows, ax_u32 D, float eps, ax_u32 flags) { \
    res_rmsnorm_mr_impl<TX>(x, delta, w, r_out, y_out, rows, D, eps, flags); \
} \
AX_KERNEL void AX_BOUNDS(DEC_NT) res_rmsnorm_hadamard_fp8_##SFX( \
    const TX *x, const TX *delta, const float *w, TX *r_out, \
    ax_fp8 *y_fp8, float *y_scales, \
    ax_u32 rows, ax_u32 D, float eps, ax_u32 flags) { \
    res_rmsnorm_hadamard_fp8_impl<TX>(x, delta, w, r_out, y_fp8, y_scales, rows, D, eps, flags); \
} \
AX_KERNEL void AX_BOUNDS(DEC_NT) silu_mul_##SFX( \
    const TX *gate, ax_u64 gs, const TX *up, ax_u64 us, TX *out, \
    ax_u32 rows, ax_u32 inter) { \
    silu_mul_impl<TX>(gate, gs, up, us, out, rows, inter); \
} \
AX_KERNEL void AX_BOUNDS(DEC_NT) silu_mul_hadamard_fp8_##SFX( \
    const TX *gate, ax_u64 gs, const TX *up, ax_u64 us, \
    ax_fp8 *y_fp8, float *y_scales, \
    ax_u32 rows, ax_u32 inter) { \
    silu_mul_hadamard_fp8_impl<TX>(gate, gs, up, us, y_fp8, y_scales, rows, inter); \
} \
AX_KERNEL void AX_BOUNDS(DEC_NT) rope_kv_write_##SFX( \
    const TX *q_in, TX *q_out, const TX *k_in, const TX *v_in, \
    const float *qnw, const float *knw, const float *cos_t, const float *sin_t, \
    TX *kc, TX *vc, \
    ax_u32 B, ax_u32 T, ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u32 rot, ax_u32 past, \
    ax_u64 q_sb, ax_u64 q_sh, ax_u64 q_st, \
    ax_u64 qo_sb, ax_u64 qo_sh, ax_u64 qo_st, \
    ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_st, \
    ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_st, \
    ax_u64 kc_sb, ax_u64 kc_sh, ax_u64 kc_ss, \
    ax_u64 cos_sb, ax_u64 cos_st, \
    float eps, ax_u32 flags) { \
    rope_kv_write_impl<TX>(q_in, q_out, k_in, v_in, qnw, knw, cos_t, sin_t, kc, vc, \
        B, T, Hq, Hkv, D, rot, past, \
        q_sb, q_sh, q_st, qo_sb, qo_sh, qo_st, k_sb, k_sh, k_st, v_sb, v_sh, v_st, \
        kc_sb, kc_sh, kc_ss, cos_sb, cos_st, eps, flags); \
} \
AX_KERNEL void AX_BOUNDS(DEC_NT) rope_kv_write_indirect_##SFX( \
    const TX *q_in, TX *q_out, const TX *k_in, const TX *v_in, \
    const float *qnw, const float *knw, const float *cos_t, const float *sin_t, \
    TX *kc, TX *vc, \
    const ax_u32 *d_kv_pos, const ax_u32 *d_rope_pos, ax_u32 pos_stride, \
    ax_u32 B, ax_u32 T, ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u32 rot, \
    ax_u64 q_sb, ax_u64 q_sh, ax_u64 q_st, \
    ax_u64 qo_sb, ax_u64 qo_sh, ax_u64 qo_st, \
    ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_st, \
    ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_st, \
    ax_u64 kc_sb, ax_u64 kc_sh, ax_u64 kc_ss, \
    ax_u64 cos_st, float eps, float mscale, ax_u32 flags) { \
    rope_kv_write_indirect_impl<TX>(q_in, q_out, k_in, v_in, qnw, knw, cos_t, sin_t, kc, vc, \
        d_kv_pos, d_rope_pos, pos_stride, B, T, Hq, Hkv, D, rot, \
        q_sb, q_sh, q_st, qo_sb, qo_sh, qo_st, k_sb, k_sh, k_st, v_sb, v_sh, v_st, \
        kc_sb, kc_sh, kc_ss, cos_st, eps, mscale, flags); \
}
