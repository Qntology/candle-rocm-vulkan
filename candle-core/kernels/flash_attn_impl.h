// Fused attention kernels (FlashAttention tiers), see candle-core/src/accel/attention.rs for
// the semantics and the launch planning.
//
// * fa_row  (V1 tier, and decoding for every tier): one workgroup per query row and key split.
//   The keys of a 128 key window are scored in parallel (one lane per key), the online softmax
//   statistics are reduced through shared memory and every lane accumulates one or two output
//   dims. With n_splits > 1 each split writes a normalized partial output and its log-sum-exp
//   (flash-decoding) and fa_combine merges them.
// * fa_tile (V2 / V3 tiers): one workgroup per BR query rows. K and V tiles of BC keys go
//   through shared memory, scores of the tile live in shared memory, every query row is owned
//   by NT / BR lanes that each keep DMAX / (NT / BR) output dims in registers (FA2 loop order:
//   Q outer, K/V inner, rescaling delayed to the end). Tiles outside the causal / sliding
//   window band are skipped.
//
// Kernel arguments (identical for both kernels):
//   q, k, v, o, part_o, part_lse, alibi, cu_q, cu_k,
//   Lq, Lk, Hq, Hkv, D,                          (max lengths for varlen)
//   q_sb, q_ss, q_sh, k_sb, k_ss, k_sh, v_sb, v_ss, v_sh,  (element strides: batch, seq, head)
//   scale, softcap, wl, wr, k_descale, v_descale,
//   n_splits, alibi_bs, varlen, rows_total
// Grid: fa_row (min(Lq * Hq, 2^31 - 1), n_splits, B); fa_tile (ceil(Lq / BR), Hq, B).
// Output row of query token `tok` (dense: b * Lq + i, varlen: cu_q[b] + i) and head h:
// `tok * Hq + h`, `D` contiguous elements.
#pragma once
#include "accel_common.h"

#define FA_NT 128

#define FA_PARAMS(TQ, TKV)                                                                         \
    const TQ *q, const TKV *k, const TKV *v, TQ *o, float *part_o, float *part_lse,                \
        const float *alibi, const ax_u32 *cu_q, const ax_u32 *cu_k, ax_u32 Lq, ax_u32 Lk,          \
        ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u64 q_sb, ax_u64 q_ss, ax_u64 q_sh, ax_u64 k_sb,       \
        ax_u64 k_ss, ax_u64 k_sh, ax_u64 v_sb, ax_u64 v_ss, ax_u64 v_sh, float scale,              \
        float softcap, int wl, int wr, float kd, float vd, ax_u32 n_splits, ax_u32 alibi_bs,       \
        ax_u32 varlen, ax_u64 rows_total

#define FA_ARGS                                                                                    \
    q, k, v, o, part_o, part_lse, alibi, cu_q, cu_k, Lq, Lk, Hq, Hkv, D, q_sb, q_ss, q_sh, k_sb,   \
        k_ss, k_sh, v_sb, v_ss, v_sh, scale, softcap, wl, wr, kd, vd, n_splits, alibi_bs, varlen,  \
        rows_total

AX_DEV float fa_logit(float dot, float softcap, float slope, long long dist) {
    float s = dot;
    if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
    if (slope != 0.0f) s -= slope * (float)(dist < 0 ? -dist : dist);
    return s;
}

template <typename TQ, typename TKV>
AX_DEV void fa_row_impl(FA_PARAMS(TQ, TKV)) {
    float *sm = ax_smem();
    float *qs = sm;          // D
    float *red = sm + D;     // FA_NT
    float *pr = red + FA_NT; // FA_NT
    const ax_u32 t = AX_TID;
    const ax_u32 split = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    ax_u32 lq = Lq, lk = Lk;
    ax_u64 q0 = (ax_u64)b * Lq, k0 = 0;
    if (varlen) {
        q0 = cu_q[b];
        lq = cu_q[b + 1] - cu_q[b];
        k0 = cu_k[b];
        lk = cu_k[b + 1] - cu_k[b];
    }
    const ax_u32 group = Hq / Hkv;
    for (ax_u64 rowi = AX_BID_X; rowi < (ax_u64)Lq * Hq; rowi += AX_NBLK_X) {
        const ax_u32 i = (ax_u32)(rowi / Hq);
        const ax_u32 h = (ax_u32)(rowi % Hq);
        if (i >= lq) continue; // uniform over the workgroup
        const ax_u32 hk = h / group;
        const TQ *qp = varlen ? q + (q0 + i) * q_ss + h * q_sh : q + b * q_sb + i * q_ss + h * q_sh;
        const TKV *kp = varlen ? k + k0 * k_ss + hk * k_sh : k + b * k_sb + hk * k_sh;
        const TKV *vp = varlen ? v + k0 * v_ss + hk * v_sh : v + b * v_sb + hk * v_sh;
        const ax_u64 orow = (q0 + i) * Hq + h;
        AX_SYNC();
        for (ax_u32 d = t; d < D; d += FA_NT) qs[d] = scale * ax_load(qp, d);
        AX_SYNC();
        const long long pos = (long long)i + (long long)lk - (long long)lq;
        long long lo = wl >= 0 ? ax_max_ll(pos - wl, 0) : 0;
        long long hi = wr >= 0 ? ax_min_ll(pos + wr + 1, (long long)lk) : (long long)lk;
        if (lo > (long long)lk) lo = lk;
        if (hi < lo) hi = lo;
        const long long chunk = (hi - lo + n_splits - 1) / n_splits;
        long long s_lo = lo + (long long)split * chunk;
        long long s_hi = ax_min_ll(s_lo + chunk, hi);
        if (s_lo > hi) s_lo = hi;
        if (s_hi < s_lo) s_hi = s_lo;
        const float slope = alibi ? alibi[(ax_u64)b * alibi_bs + h] : 0.0f;
        float m = -AX_INF, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
        for (long long kb = s_lo; kb < s_hi; kb += FA_NT) {
            const long long j = kb + t;
            float s = -AX_INF;
            if (j < s_hi) {
                const TKV *kr = kp + (ax_u64)j * k_ss;
                float dot = 0.0f;
                for (ax_u32 d = 0; d < D; ++d) dot += qs[d] * ax_load(kr, d);
                s = fa_logit(dot * kd, softcap, slope, pos - j);
            }
            const float m_new = fmaxf(m, ax_block_max<FA_NT>(s, red));
            const float p = s == -AX_INF ? 0.0f : expf(s - m_new);
            pr[t] = p;
            const float lt = ax_block_sum<FA_NT>(p, red); // also publishes pr
            const float alpha = m == -AX_INF ? 0.0f : expf(m - m_new);
            l = l * alpha + lt;
            acc0 *= alpha;
            acc1 *= alpha;
            const int cnt = (int)ax_min_ll(FA_NT, s_hi - kb);
            for (int u = 0; u < cnt; ++u) {
                const float pu = pr[u];
                if (pu != 0.0f) {
                    const TKV *vr = vp + (ax_u64)(kb + u) * v_ss;
                    if (t < D) acc0 += pu * ax_load(vr, t);
                    if (t + FA_NT < D) acc1 += pu * ax_load(vr, t + FA_NT);
                }
            }
            m = m_new;
            AX_SYNC();
        }
        const float inv = l > 0.0f ? vd / l : 0.0f;
        if (n_splits == 1) {
            if (t < D) ax_store(o, orow * D + t, acc0 * inv);
            if (t + FA_NT < D) ax_store(o, orow * D + t + FA_NT, acc1 * inv);
        } else {
            const ax_u64 prow = (ax_u64)split * rows_total + orow;
            if (t < D) part_o[prow * D + t] = acc0 * inv;
            if (t + FA_NT < D) part_o[prow * D + t + FA_NT] = acc1 * inv;
            if (t == 0) part_lse[prow] = l > 0.0f ? m + logf(l) : -AX_INF;
        }
    }
}

template <typename TQ, typename TKV, int BR, int BC, int DMAX>
AX_DEV void fa_tile_impl(FA_PARAMS(TQ, TKV)) {
    const int TPR = FA_NT / BR;
    const int NACC = DMAX / TPR;
    const ax_u32 DP = D + 1; // padded row stride against bank conflicts
    float *sm = ax_smem();
    float *Qs = sm;             // BR * DP
    float *KVs = Qs + BR * DP;  // BC * DP
    float *S = KVs + BC * DP;   // BR * BC
    const ax_u32 t = AX_TID;
    const ax_u32 r = t / TPR;
    const ax_u32 lane = t % TPR;
    const ax_u32 h = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    ax_u32 lq = Lq, lk = Lk;
    ax_u64 q0 = (ax_u64)b * Lq, k0 = 0;
    if (varlen) {
        q0 = cu_q[b];
        lq = cu_q[b + 1] - cu_q[b];
        k0 = cu_k[b];
        lk = cu_k[b + 1] - cu_k[b];
    }
    const ax_u32 hk = h / (Hq / Hkv);
    const TQ *qb = varlen ? q + q0 * q_ss + h * q_sh : q + b * q_sb + h * q_sh;
    const TKV *kb_ = varlen ? k + k0 * k_ss + hk * k_sh : k + b * k_sb + hk * k_sh;
    const TKV *vb_ = varlen ? v + k0 * v_ss + hk * v_sh : v + b * v_sb + hk * v_sh;
    const long long off = (long long)lk - (long long)lq;
    const float slope = alibi ? alibi[(ax_u64)b * alibi_bs + h] : 0.0f;
    for (ax_u64 tile = AX_BID_X; (long long)(tile * BR) < (long long)lq; tile += AX_NBLK_X) {
        const long long i0 = (long long)tile * BR;
        AX_SYNC();
        for (ax_u32 idx = t; idx < BR * D; idx += FA_NT) {
            const ax_u32 rr = idx / D, d = idx % D;
            const long long ii = i0 + rr;
            Qs[rr * DP + d] = ii < (long long)lq ? scale * ax_load(qb, (ax_u64)ii * q_ss + d) : 0.0f;
        }
        const long long last = ax_min_ll(i0 + BR, (long long)lq) - 1;
        long long lo = wl >= 0 ? ax_max_ll(i0 + off - wl, 0) : 0;
        long long hi = wr >= 0 ? ax_min_ll(last + off + wr + 1, (long long)lk) : (long long)lk;
        if (lo > (long long)lk) lo = lk;
        if (hi < lo) hi = lo;
        const long long i = i0 + r;
        float m = -AX_INF, l = 0.0f;
        float acc[NACC];
        AX_UNROLL
        for (int c = 0; c < NACC; ++c) acc[c] = 0.0f;
        for (long long kt = lo; kt < hi; kt += BC) {
            AX_SYNC();
            for (ax_u32 idx = t; idx < BC * D; idx += FA_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                KVs[c * DP + d] = j < hi ? kd * ax_load(kb_, (ax_u64)j * k_ss + d) : 0.0f;
            }
            AX_SYNC();
            for (ax_u32 idx = t; idx < BR * BC; idx += FA_NT) {
                const ax_u32 rr = idx / BC, c = idx % BC;
                const long long ii = i0 + rr, j = kt + c;
                const long long p2 = ii + off;
                float s = -AX_INF;
                if (ii < (long long)lq && j < hi && (wl < 0 || j >= p2 - wl) && (wr < 0 || j <= p2 + wr)) {
                    const float *qr = Qs + rr * DP;
                    const float *kr = KVs + c * DP;
                    float dot = 0.0f;
                    for (ax_u32 d = 0; d < D; ++d) dot += qr[d] * kr[d];
                    s = fa_logit(dot, softcap, slope, p2 - j);
                }
                S[idx] = s;
            }
            AX_SYNC();
            // V tile into shared memory while every lane finds its row maximum
            for (ax_u32 idx = t; idx < BC * D; idx += FA_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                KVs[c * DP + d] = j < hi ? ax_load(vb_, (ax_u64)j * v_ss + d) : 0.0f;
            }
            float mt = -AX_INF;
            for (int c = 0; c < BC; ++c) mt = fmaxf(mt, S[r * BC + c]);
            const float m_new = fmaxf(m, mt);
            AX_SYNC();
            for (int c = lane; c < BC; c += TPR) {
                const float s = S[r * BC + c];
                S[r * BC + c] = s == -AX_INF ? 0.0f : expf(s - m_new);
            }
            AX_SYNC();
            const float alpha = m == -AX_INF ? 0.0f : expf(m - m_new);
            float lt = 0.0f;
            for (int c = 0; c < BC; ++c) lt += S[r * BC + c];
            l = l * alpha + lt;
            AX_UNROLL
            for (int a = 0; a < NACC; ++a) acc[a] *= alpha;
            for (int c = 0; c < BC; ++c) {
                const float p = S[r * BC + c];
                if (p != 0.0f) {
                    const float *vr = KVs + c * DP;
                    AX_UNROLL
                    for (int a = 0; a < NACC; ++a) {
                        const ax_u32 d = lane + a * TPR;
                        if (d < D) acc[a] += p * vr[d];
                    }
                }
            }
            m = m_new;
        }
        if (i < (long long)lq) {
            const float inv = l > 0.0f ? vd / l : 0.0f;
            const ax_u64 orow = (q0 + (ax_u64)i) * Hq + h;
            AX_UNROLL
            for (int a = 0; a < NACC; ++a) {
                const ax_u32 d = lane + a * TPR;
                if (d < D) ax_store(o, orow * D + d, acc[a] * inv);
            }
        }
    }
}

// Merges the normalized partial outputs of n_splits key splits with their log-sum-exp.
template <typename TO>
AX_DEV void fa_combine_impl(const float *part_o, const float *part_lse, TO *o, ax_u64 rows,
                            ax_u32 D, ax_u32 n_splits) {
    const ax_u32 t = AX_TID;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        float mx = -AX_INF;
        for (ax_u32 s = 0; s < n_splits; ++s) mx = fmaxf(mx, part_lse[(ax_u64)s * rows + row]);
        float wsum = 0.0f;
        for (ax_u32 s = 0; s < n_splits; ++s) {
            const float e = part_lse[(ax_u64)s * rows + row];
            wsum += e == -AX_INF ? 0.0f : expf(e - mx);
        }
        for (ax_u32 d = t; d < D; d += FA_NT) {
            float acc = 0.0f;
            for (ax_u32 s = 0; s < n_splits; ++s) {
                const float e = part_lse[(ax_u64)s * rows + row];
                if (e != -AX_INF) acc += expf(e - mx) * part_o[((ax_u64)s * rows + row) * D + d];
            }
            ax_store(o, row * D + d, wsum > 0.0f ? acc / wsum : 0.0f);
        }
    }
}

// Kernel entry points. NAME is the exported symbol.
#define FA_ROW_KERNEL(NAME, TQ, TKV)                                                               \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV)) { fa_row_impl<TQ, TKV>(FA_ARGS); }

#define FA_TILE_KERNEL(NAME, TQ, TKV, BR, BC, DMAX)                                                \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV)) {                                     \
        fa_tile_impl<TQ, TKV, BR, BC, DMAX>(FA_ARGS);                                              \
    }

#define FA_COMBINE_KERNEL(NAME, TO)                                                                \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(const float *part_o, const float *part_lse, TO *o,        \
                                         ax_u64 rows, ax_u32 D, ax_u32 n_splits) {                 \
        fa_combine_impl<TO>(part_o, part_lse, o, rows, D, n_splits);                               \
    }

// Tile shapes per tier and head dim bucket (keep in sync with accel::attention::tile_config).
#define FA2_TILES(Q, KV, TQ, TKV)                                                                  \
    FA_TILE_KERNEL(fa2_d64_##Q##_##KV, TQ, TKV, 32, 32, 64)                                        \
    FA_TILE_KERNEL(fa2_d128_##Q##_##KV, TQ, TKV, 16, 32, 128)                                      \
    FA_TILE_KERNEL(fa2_d256_##Q##_##KV, TQ, TKV, 8, 16, 256)

#define FA3_TILES(Q, KV, TQ, TKV)                                                                  \
    FA_TILE_KERNEL(fa3_d64_##Q##_##KV, TQ, TKV, 64, 64, 64)                                        \
    FA_TILE_KERNEL(fa3_d128_##Q##_##KV, TQ, TKV, 32, 64, 128)                                      \
    FA_TILE_KERNEL(fa3_d256_##Q##_##KV, TQ, TKV, 16, 32, 256)

#define FA_ALL_FOR(Q, KV, TQ, TKV)                                                                 \
    FA_ROW_KERNEL(fa_row_##Q##_##KV, TQ, TKV)                                                      \
    FA2_TILES(Q, KV, TQ, TKV)                                                                      \
    FA3_TILES(Q, KV, TQ, TKV)
