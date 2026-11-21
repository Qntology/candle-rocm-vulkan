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

// Flags for res_rmsnorm_*
#define RES_RMS_PLUS_ONE_WEIGHT 1u
#define RES_RMS_WRITE_RESIDUAL  2u

// Flags for rope_kv_write_*
#define ROPE_MODE_NEOX          0u
#define ROPE_MODE_ADJACENT      1u
#define ROPE_NORM_PLUS_ONE      2u

// ---------------------------------------------------------------------------
// 1) Residual + RMSNorm (1 workgroup of DEC_NT threads per row)
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

    float local_sq = 0.0f;
    for (ax_u32 col = AX_TID; col < D; col += DEC_NT) {
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
        local_sq = fmaf(v, v, local_sq);
    }

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

#define DEC_OPS_INSTANTIATE(SFX, TX) \
AX_KERNEL void AX_BOUNDS(DEC_NT) res_rmsnorm_##SFX( \
    const TX *x, const TX *delta, const float *w, TX *r_out, TX *y_out, \
    ax_u32 rows, ax_u32 D, float eps, ax_u32 flags) { \
    res_rmsnorm_impl<TX>(x, delta, w, r_out, y_out, rows, D, eps, flags); \
} \
AX_KERNEL void AX_BOUNDS(DEC_NT) silu_mul_##SFX( \
    const TX *gate, ax_u64 gs, const TX *up, ax_u64 us, TX *out, \
    ax_u32 rows, ax_u32 inter) { \
    silu_mul_impl<TX>(gate, gs, up, us, out, rows, inter); \
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
}
