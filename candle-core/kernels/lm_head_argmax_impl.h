// Fused lm_head Tile Argmax + Mean-Centered Cauchy-Schwarz (CS) Guard kernels.
//
// Eliminates [M, V] logits materialization in HBM for greedy decoding (Dense F32/F16/BF16)
// and avoids full-vocab expansion / FP32 casting for quantized QMatMul sub-logits.
//
// Built from one source by:
//   * ROCm / HIP: hipcc at build time (kernels/lm_head_argmax.hip)
//   * CUDA: NVRTC at run time (accel/sources.rs -> cuda_backend/lm_head.rs, --std=c++17, no fast-math)
//   * Host emulator: plain C++20 (CANDLE_ACCEL_HOST=1)
//
// Uses only accel_common.h primitives (dynamic shared memory + AX_SYNC, no warp shuffles,
// no standard headers on CUDA NVRTC). Deterministic tie-break always prefers smaller index.
#pragma once
#include "accel_common.h"

#define LM_NT         256u
#define LM_DECODE_BN  32u
#define LM_PREFILL_BM 32u
#define LM_PREFILL_BN 64u
#define LM_PREFILL_BK 32u

AX_DEV bool lm_better(float v_cand, ax_u32 i_cand, float v_curr, ax_u32 i_curr) {
    return (v_cand > v_curr) || (v_cand == v_curr && i_cand < i_curr);
}

// Unified 4 KiB shared-memory tree reduction for (val, idx) argmax + (xx, xm) CS stats.
AX_DEV void lm_reduce_argmax_and_cs(
    float my_val,
    ax_u32 my_idx,
    float my_xx,
    float my_xm,
    bool check_cs,
    float *s_val,
    ax_u32 *s_idx,
    float *s_xx,
    float *s_xm
) {
    const ax_u32 tid = AX_TID;
    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    if (check_cs) {
        s_xx[tid] = my_xx;
        s_xm[tid] = my_xm;
    }
    AX_SYNC();

    for (ax_u32 s = LM_NT >> 1; s > 0u; s >>= 1) {
        if (tid < s) {
            if (lm_better(s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
            if (check_cs) {
                s_xx[tid] += s_xx[tid + s];
                s_xm[tid] += s_xm[tid + s];
            }
        }
        AX_SYNC();
    }
}

// ---------------------------------------------------------------------------
// Kernel 1a (Decode: M <= MAX_M <= 8, BN = 32)
// Grid: (n_tiles, 1, 1), Block: (256, 1, 1)
// Shared memory: (LM_NT * 4 + MAX_M * LM_DECODE_BN) * sizeof(float)
// ---------------------------------------------------------------------------
template <typename TX, ax_u32 MAX_M>
AX_DEV void lm_stage1_decode_impl(
    const TX *x,
    const TX *w,
    const float *bias,
    float *part_val,
    ax_u32 *part_idx,
    ax_u32 M,
    ax_u32 V_valid,
    ax_u32 K,
    ax_u32 n_tiles
) {
    const ax_u32 tile_id = AX_BID_X;
    if (tile_id >= n_tiles || M == 0u || M > MAX_M) return;

    const ax_u32 tid = AX_TID;
    const ax_u32 grp = tid >> 5;   // 0..7 (8 groups of 32 lanes)
    const ax_u32 lane = tid & 31u; // 0..31
    const ax_u32 col0 = tile_id * LM_DECODE_BN + grp * 4u;

    float acc[MAX_M][4];
    AX_UNROLL
    for (ax_u32 m = 0u; m < MAX_M; ++m) {
        acc[m][0] = 0.0f;
        acc[m][1] = 0.0f;
        acc[m][2] = 0.0f;
        acc[m][3] = 0.0f;
    }

    for (ax_u32 k = lane; k < K; k += 32u) {
        float xv[MAX_M];
        for (ax_u32 m = 0u; m < M; ++m) {
            xv[m] = ax_load(x, (ax_u64)m * (ax_u64)K + k);
        }
        AX_UNROLL
        for (ax_u32 c = 0u; c < 4u; ++c) {
            const ax_u32 n = col0 + c;
            if (n < V_valid) {
                const float wv = ax_load(w, (ax_u64)n * (ax_u64)K + k);
                for (ax_u32 m = 0u; m < M; ++m) {
                    acc[m][c] = fmaf(xv[m], wv, acc[m][c]);
                }
            }
        }
    }

    float *smem = ax_smem();
    float *col_logits = smem + LM_NT * 4u;

    for (ax_u32 m = 0u; m < M; ++m) {
        AX_UNROLL
        for (ax_u32 c = 0u; c < 4u; ++c) {
            smem[tid * 4u + c] = acc[m][c];
        }
        AX_SYNC();

        for (ax_u32 s = 16u; s > 0u; s >>= 1) {
            if (lane < s) {
                const ax_u32 base = (grp * 32u + lane) * 4u;
                const ax_u32 peer = (grp * 32u + lane + s) * 4u;
                AX_UNROLL
                for (ax_u32 c = 0u; c < 4u; ++c) {
                    smem[base + c] += smem[peer + c];
                }
            }
            AX_SYNC();
        }

        if (lane == 0u) {
            AX_UNROLL
            for (ax_u32 c = 0u; c < 4u; ++c) {
                const ax_u32 local_col = grp * 4u + c;
                const ax_u32 global_col = col0 + c;
                float v = -AX_INF;
                if (global_col < V_valid) {
                    v = smem[(grp * 32u) * 4u + c];
                    if (bias != nullptr) {
                        v += bias[global_col];
                    }
                }
                col_logits[m * LM_DECODE_BN + local_col] = v;
            }
        }
        AX_SYNC();
    }

    if (tid < M) {
        float best_v = -AX_INF;
        ax_u32 best_i = 0xffffffffu;
        for (ax_u32 lc = 0u; lc < LM_DECODE_BN; ++lc) {
            const ax_u32 gc = tile_id * LM_DECODE_BN + lc;
            if (gc < V_valid) {
                const float v = col_logits[tid * LM_DECODE_BN + lc];
                if (lm_better(v, gc, best_v, best_i)) {
                    best_v = v;
                    best_i = gc;
                }
            }
        }
        part_val[(ax_u64)tid * (ax_u64)n_tiles + tile_id] = best_v;
        part_idx[(ax_u64)tid * (ax_u64)n_tiles + tile_id] = best_i;
    }
}

// ---------------------------------------------------------------------------
// Kernel 1b (Prefill / Multi-Crop: M > 8, BM = 32, BN = 64, BK = 32)
// Grid: (n_tiles_n, ceil(M / 32), 1), Block: (256, 1, 1)
// Shared memory: (32 * 32 + 64 * 32) * sizeof(float) = 12 KiB
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void lm_stage1_prefill_impl(
    const TX *x,
    const TX *w,
    const float *bias,
    float *part_val,
    ax_u32 *part_idx,
    ax_u32 M,
    ax_u32 V_valid,
    ax_u32 K,
    ax_u32 n_tiles_n
) {
    const ax_u32 tile_n = AX_BID_X;
    const ax_u32 tile_m = AX_BID_Y;
    const ax_u32 m0 = tile_m * LM_PREFILL_BM;
    const ax_u32 n0 = tile_n * LM_PREFILL_BN;
    if (tile_n >= n_tiles_n || m0 >= M) return;

    const ax_u32 tid = AX_TID;
    const ax_u32 ty = tid >> 4;  // 0..15 (2 rows each -> 32 rows)
    const ax_u32 tx = tid & 15u; // 0..15 (4 cols each -> 64 cols)

    float *smem = ax_smem();
    float *sA = smem;                                 // [32 * 32]
    float *sB = smem + LM_PREFILL_BM * LM_PREFILL_BK; // [64 * 32]

    float acc[2][4];
    AX_UNROLL
    for (ax_u32 dr = 0u; dr < 2u; ++dr) {
        AX_UNROLL
        for (ax_u32 dc = 0u; dc < 4u; ++dc) {
            acc[dr][dc] = 0.0f;
        }
    }

    for (ax_u32 k0 = 0u; k0 < K; k0 += LM_PREFILL_BK) {
        // Load sA[32, 32]: 1024 elements / 256 threads = 4 elements per thread
        for (ax_u32 idx = tid; idx < LM_PREFILL_BM * LM_PREFILL_BK; idx += LM_NT) {
            const ax_u32 lr = idx / LM_PREFILL_BK;
            const ax_u32 lk = idx - lr * LM_PREFILL_BK;
            const ax_u32 gr = m0 + lr;
            const ax_u32 gk = k0 + lk;
            sA[idx] = (gr < M && gk < K) ? ax_load(x, (ax_u64)gr * (ax_u64)K + gk) : 0.0f;
        }
        // Load sB[64, 32]: 2048 elements / 256 threads = 8 elements per thread
        for (ax_u32 idx = tid; idx < LM_PREFILL_BN * LM_PREFILL_BK; idx += LM_NT) {
            const ax_u32 ln = idx / LM_PREFILL_BK;
            const ax_u32 lk = idx - ln * LM_PREFILL_BK;
            const ax_u32 gn = n0 + ln;
            const ax_u32 gk = k0 + lk;
            sB[idx] = (gn < V_valid && gk < K) ? ax_load(w, (ax_u64)gn * (ax_u64)K + gk) : 0.0f;
        }
        AX_SYNC();

        const ax_u32 k_lim = (k0 + LM_PREFILL_BK <= K) ? LM_PREFILL_BK : (K - k0);
        for (ax_u32 kk = 0u; kk < k_lim; ++kk) {
            const float a0 = sA[(ty * 2u + 0u) * LM_PREFILL_BK + kk];
            const float a1 = sA[(ty * 2u + 1u) * LM_PREFILL_BK + kk];
            AX_UNROLL
            for (ax_u32 dc = 0u; dc < 4u; ++dc) {
                const float bv = sB[(tx * 4u + dc) * LM_PREFILL_BK + kk];
                acc[0][dc] = fmaf(a0, bv, acc[0][dc]);
                acc[1][dc] = fmaf(a1, bv, acc[1][dc]);
            }
        }
        AX_SYNC();
    }

    // Reuse smem as sOut[32, 64] (8 KiB <= 12 KiB) for in-block row argmax.
    float *sOut = smem;
    AX_UNROLL
    for (ax_u32 dr = 0u; dr < 2u; ++dr) {
        const ax_u32 lr = ty * 2u + dr;
        AX_UNROLL
        for (ax_u32 dc = 0u; dc < 4u; ++dc) {
            const ax_u32 lc = tx * 4u + dc;
            const ax_u32 gn = n0 + lc;
            float v = -AX_INF;
            if (gn < V_valid) {
                v = acc[dr][dc];
                if (bias != nullptr) {
                    v += bias[gn];
                }
            }
            sOut[lr * LM_PREFILL_BN + lc] = v;
        }
    }
    AX_SYNC();

    if (tid < LM_PREFILL_BM) {
        const ax_u32 gr = m0 + tid;
        if (gr < M) {
            float best_v = -AX_INF;
            ax_u32 best_i = 0xffffffffu;
            for (ax_u32 lc = 0u; lc < LM_PREFILL_BN; ++lc) {
                const ax_u32 gn = n0 + lc;
                if (gn < V_valid) {
                    const float v = sOut[tid * LM_PREFILL_BN + lc];
                    if (lm_better(v, gn, best_v, best_i)) {
                        best_v = v;
                        best_i = gn;
                    }
                }
            }
            part_val[(ax_u64)gr * (ax_u64)n_tiles_n + tile_n] = best_v;
            part_idx[(ax_u64)gr * (ax_u64)n_tiles_n + tile_n] = best_i;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Stage 2 Tile Reduction + Core-ID Remap + Mean-Centered CS Guard
// Grid: (M, 1, 1), Block: (256, 1, 1), Shared memory: 4 * 256 * 4 = 4 KiB
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void lm_stage2_reduce_cs_impl(
    const float *part_val,
    const ax_u32 *part_idx,
    const ax_u32 *core_ids,
    const TX *hidden,
    const float *mu_rest,
    float w_rest_bound,
    float b_rest_max,
    ax_u32 *out_id,
    float *out_val,
    ax_u8 *out_rescue,
    ax_u32 M,
    ax_u32 n_tiles,
    ax_u32 K,
    ax_u32 check_cs
) {
    const ax_u32 m = AX_BID_X;
    if (m >= M) return;

    float *s_val = ax_smem();
    ax_u32 *s_idx = reinterpret_cast<ax_u32 *>(s_val + LM_NT);
    float *s_xx = reinterpret_cast<float *>(s_idx + LM_NT);
    float *s_xm = s_xx + LM_NT;

    float my_v = -AX_INF;
    ax_u32 my_i = 0xffffffffu;
    const ax_u64 p_base = (ax_u64)m * (ax_u64)n_tiles;
    for (ax_u32 t = AX_TID; t < n_tiles; t += LM_NT) {
        const float v = part_val[p_base + t];
        const ax_u32 i = part_idx[p_base + t];
        if (lm_better(v, i, my_v, my_i)) {
            my_v = v;
            my_i = i;
        }
    }

    float my_xx = 0.0f;
    float my_xm = 0.0f;
    const bool do_cs = (check_cs != 0u) && (hidden != nullptr);
    if (do_cs) {
        const ax_u64 h_base = (ax_u64)m * (ax_u64)K;
        for (ax_u32 k = AX_TID; k < K; k += LM_NT) {
            const float hv = ax_load(hidden, h_base + k);
            my_xx = fmaf(hv, hv, my_xx);
            if (mu_rest != nullptr) {
                my_xm = fmaf(hv, mu_rest[k], my_xm);
            }
        }
    }

    lm_reduce_argmax_and_cs(my_v, my_i, my_xx, my_xm, do_cs, s_val, s_idx, s_xx, s_xm);

    if (AX_TID == 0u) {
        const float best_v = s_val[0];
        const ax_u32 best_local = s_idx[0];
        out_id[m] = (core_ids != nullptr) ? core_ids[best_local] : best_local;
        if (out_val != nullptr) {
            out_val[m] = best_v;
        }
        if (out_rescue != nullptr) {
            if (do_cs) {
                const float upper = fmaf(sqrtf(s_xx[0]), w_rest_bound, s_xm[0]) + b_rest_max;
                out_rescue[m] = (best_v < upper) ? 1u : 0u;
            } else {
                out_rescue[m] = 0u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: 1-Stage Sub-Logits Argmax + Core-ID Remap + CS Guard (GGUF QMatMul)
// Grid: (M, 1, 1), Block: (256, 1, 1), Shared memory: 4 * 256 * 4 = 4 KiB
// ---------------------------------------------------------------------------
template <typename TL, typename TH>
AX_DEV void lm_logits_argmax_cs_impl(
    const TL *sub_logits,
    ax_u64 logits_row_stride,
    const ax_u32 *core_ids,
    const TH *hidden,
    const float *mu_rest,
    float w_rest_bound,
    float b_rest_max,
    ax_u32 *out_id,
    float *out_val,
    ax_u8 *out_rescue,
    ax_u32 M,
    ax_u32 V_valid,
    ax_u32 K,
    ax_u32 check_cs
) {
    const ax_u32 m = AX_BID_X;
    if (m >= M) return;

    float *s_val = ax_smem();
    ax_u32 *s_idx = reinterpret_cast<ax_u32 *>(s_val + LM_NT);
    float *s_xx = reinterpret_cast<float *>(s_idx + LM_NT);
    float *s_xm = s_xx + LM_NT;

    float my_v = -AX_INF;
    ax_u32 my_i = 0xffffffffu;
    const ax_u64 l_base = (ax_u64)m * logits_row_stride;
    for (ax_u32 j = AX_TID; j < V_valid; j += LM_NT) {
        const float v = ax_load(sub_logits, l_base + j);
        if (lm_better(v, j, my_v, my_i)) {
            my_v = v;
            my_i = j;
        }
    }

    float my_xx = 0.0f;
    float my_xm = 0.0f;
    const bool do_cs = (check_cs != 0u) && (hidden != nullptr);
    if (do_cs) {
        const ax_u64 h_base = (ax_u64)m * (ax_u64)K;
        for (ax_u32 k = AX_TID; k < K; k += LM_NT) {
            const float hv = ax_load(hidden, h_base + k);
            my_xx = fmaf(hv, hv, my_xx);
            if (mu_rest != nullptr) {
                my_xm = fmaf(hv, mu_rest[k], my_xm);
            }
        }
    }

    lm_reduce_argmax_and_cs(my_v, my_i, my_xx, my_xm, do_cs, s_val, s_idx, s_xx, s_xm);

    if (AX_TID == 0u) {
        const float best_v = s_val[0];
        const ax_u32 best_local = s_idx[0];
        out_id[m] = (core_ids != nullptr) ? core_ids[best_local] : best_local;
        if (out_val != nullptr) {
            out_val[m] = best_v;
        }
        if (out_rescue != nullptr) {
            if (do_cs) {
                const float upper = fmaf(sqrtf(s_xx[0]), w_rest_bound, s_xm[0]) + b_rest_max;
                out_rescue[m] = (best_v < upper) ? 1u : 0u;
            } else {
                out_rescue[m] = 0u;
            }
        }
    }
}

#define LM_HEAD_DENSE_INSTANTIATE(SFX, TX) \
AX_KERNEL void AX_BOUNDS(LM_NT) lm_stage1_decode1_##SFX( \
    const TX *x, const TX *w, const float *bias, \
    float *part_val, ax_u32 *part_idx, \
    ax_u32 M, ax_u32 V_valid, ax_u32 K, ax_u32 n_tiles) { \
    lm_stage1_decode_impl<TX, 1u>(x, w, bias, part_val, part_idx, M, V_valid, K, n_tiles); \
} \
AX_KERNEL void AX_BOUNDS(LM_NT) lm_stage1_decode8_##SFX( \
    const TX *x, const TX *w, const float *bias, \
    float *part_val, ax_u32 *part_idx, \
    ax_u32 M, ax_u32 V_valid, ax_u32 K, ax_u32 n_tiles) { \
    lm_stage1_decode_impl<TX, 8u>(x, w, bias, part_val, part_idx, M, V_valid, K, n_tiles); \
} \
AX_KERNEL void AX_BOUNDS(LM_NT) lm_stage1_prefill_##SFX( \
    const TX *x, const TX *w, const float *bias, \
    float *part_val, ax_u32 *part_idx, \
    ax_u32 M, ax_u32 V_valid, ax_u32 K, ax_u32 n_tiles_n) { \
    lm_stage1_prefill_impl<TX>(x, w, bias, part_val, part_idx, M, V_valid, K, n_tiles_n); \
} \
AX_KERNEL void AX_BOUNDS(LM_NT) lm_stage2_reduce_cs_##SFX( \
    const float *part_val, const ax_u32 *part_idx, const ax_u32 *core_ids, \
    const TX *hidden, const float *mu_rest, float w_rest_bound, float b_rest_max, \
    ax_u32 *out_id, float *out_val, ax_u8 *out_rescue, \
    ax_u32 M, ax_u32 n_tiles, ax_u32 K, ax_u32 check_cs) { \
    lm_stage2_reduce_cs_impl<TX>(part_val, part_idx, core_ids, hidden, mu_rest, \
        w_rest_bound, b_rest_max, out_id, out_val, out_rescue, M, n_tiles, K, check_cs); \
}

#define LM_HEAD_LOGITS_INSTANTIATE(LSFX, TL, HSFX, TH) \
AX_KERNEL void AX_BOUNDS(LM_NT) lm_logits_argmax_cs_##LSFX##_##HSFX( \
    const TL *sub_logits, ax_u64 logits_row_stride, const ax_u32 *core_ids, \
    const TH *hidden, const float *mu_rest, float w_rest_bound, float b_rest_max, \
    ax_u32 *out_id, float *out_val, ax_u8 *out_rescue, \
    ax_u32 M, ax_u32 V_valid, ax_u32 K, ax_u32 check_cs) { \
    lm_logits_argmax_cs_impl<TL, TH>(sub_logits, logits_row_stride, core_ids, hidden, mu_rest, \
        w_rest_bound, b_rest_max, out_id, out_val, out_rescue, M, V_valid, K, check_cs); \
}
