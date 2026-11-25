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

// One window of up to FA_NT keys [kb, min(kb + FA_NT, e)) of the row kernel: key j sits at row
// j - j0 of kp / vp. Updates the online softmax state (m, l) and the two output accumulators.
template <typename TKV>
AX_DEV void fa_row_window(const float *qs, float *red, float *pr, const TKV *kp, const TKV *vp,
                          ax_u64 k_ss, ax_u64 v_ss, long long j0, long long kb, long long e,
                          ax_u32 D, float kd, float softcap, float slope, long long pos, float &m,
                          float &l, float &acc0, float &acc1) {
    const ax_u32 t = AX_TID;
    const long long j = kb + t;
    float s = -AX_INF;
    if (j < e) {
        const TKV *kr = kp + (ax_u64)(j - j0) * k_ss;
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
    const int cnt = (int)ax_min_ll(FA_NT, e - kb);
    for (int u = 0; u < cnt; ++u) {
        const float pu = pr[u];
        if (pu != 0.0f) {
            const TKV *vr = vp + (ax_u64)(kb + u - j0) * v_ss;
            if (t < D) acc0 += pu * ax_load(vr, t);
            if (t + FA_NT < D) acc1 += pu * ax_load(vr, t + FA_NT);
        }
    }
    m = m_new;
    AX_SYNC();
}

// Row kernel over dense K/V (PAGED = false) or over the blocks of `pg` (PAGED = true: k / v are
// unused, no varlen; k_ss / v_ss are the row strides shared by all blocks).
template <typename TQ, typename TKV, bool PAGED>
AX_DEV void fa_row_core(FA_PARAMS(TQ, TKV), const ax_pages *pg) {
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
        if (!PAGED) {
            const TKV *kp = varlen ? k + k0 * k_ss + hk * k_sh : k + b * k_sb + hk * k_sh;
            const TKV *vp = varlen ? v + k0 * v_ss + hk * v_sh : v + b * v_sb + hk * v_sh;
            for (long long kb = s_lo; kb < s_hi; kb += FA_NT)
                fa_row_window<TKV>(qs, red, pr, kp, vp, k_ss, v_ss, 0, kb, s_hi, D, kd, softcap,
                                   slope, pos, m, l, acc0, acc1);
        } else {
            // windows never straddle two blocks; the block loop is uniform over the workgroup
            long long j0 = 0;
            for (ax_u32 bi = 0; bi < pg->n && j0 < s_hi; ++bi) {
                const long long j1 = j0 + (long long)pg->rows[bi];
                const long long a = ax_max_ll(s_lo, j0), e = ax_min_ll(s_hi, j1);
                if (a < e) {
                    const ax_u64 kh = (ax_u64)pg->k_cap[bi] * k_ss;
                    const ax_u64 vh = (ax_u64)pg->v_cap[bi] * v_ss;
                    const ax_u64 bh = (ax_u64)b * Hkv + hk;
                    const TKV *kp = reinterpret_cast<const TKV *>(pg->k[bi]) + bh * kh;
                    const TKV *vp = reinterpret_cast<const TKV *>(pg->v[bi]) + bh * vh;
                    for (long long kb = a; kb < e; kb += FA_NT)
                        fa_row_window<TKV>(qs, red, pr, kp, vp, k_ss, v_ss, j0, kb, e, D, kd,
                                           softcap, slope, pos, m, l, acc0, acc1);
                }
                j0 = j1;
            }
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

template <typename TQ, typename TKV>
AX_DEV void fa_row_impl(FA_PARAMS(TQ, TKV)) {
    fa_row_core<TQ, TKV, false>(FA_ARGS, nullptr);
}

// Row kernel over a paged K/V block table (decoding without concatenating the blocks).
template <typename TQ, typename TKV>
AX_DEV void fa_rowp_impl(FA_PARAMS(TQ, TKV), const ax_pages &pg) {
    fa_row_core<TQ, TKV, true>(FA_ARGS, &pg);
}

// ---------------------------------------------------------------------------
// GQA-packed row kernel: one workgroup per (query row, pack of G query heads that share a kv
// head, split), so every K/V element is loaded and widened once for G heads (G divides
// Hq / Hkv). Each head's arithmetic is the one of fa_row (same products, same summation and
// reduction order), so the results are bit-identical to fa_row; only the loads are shared.
// Grid (min(Lq * Hq / G, 2^31 - 1), n_splits, B), no varlen. Shared memory: G * (D + 2 FA_NT)
// floats.
// ---------------------------------------------------------------------------

template <typename TKV, int G>
AX_DEV void fa_rowg_window(const float *qs, float *red, float *pr, const TKV *kp, const TKV *vp,
                           ax_u64 k_ss, ax_u64 v_ss, long long j0, long long kb, long long e,
                           ax_u32 D, float kd, float softcap, const float *slope, long long pos,
                           float *m, float *l, float *acc0, float *acc1) {
    const ax_u32 t = AX_TID;
    const long long j = kb + t;
    float s[G];
    AX_UNROLL
    for (int g = 0; g < G; ++g) s[g] = -AX_INF;
    if (j < e) {
        const TKV *kr = kp + (ax_u64)(j - j0) * k_ss;
        float dot[G];
        AX_UNROLL
        for (int g = 0; g < G; ++g) dot[g] = 0.0f;
        for (ax_u32 d = 0; d < D; ++d) {
            const float kv = ax_load(kr, d);
            AX_UNROLL
            for (int g = 0; g < G; ++g) dot[g] += qs[g * D + d] * kv;
        }
        AX_UNROLL
        for (int g = 0; g < G; ++g) s[g] = fa_logit(dot[g] * kd, softcap, slope[g], pos - j);
    }
    float mx[G], p[G], lt[G];
    ax_block_reduce_g<FA_NT, G, true>(s, red, mx);
    AX_UNROLL
    for (int g = 0; g < G; ++g) {
        mx[g] = fmaxf(m[g], mx[g]);
        p[g] = s[g] == -AX_INF ? 0.0f : expf(s[g] - mx[g]);
        pr[g * FA_NT + t] = p[g];
    }
    ax_block_reduce_g<FA_NT, G, false>(p, red, lt); // also publishes pr
    AX_UNROLL
    for (int g = 0; g < G; ++g) {
        const float alpha = m[g] == -AX_INF ? 0.0f : expf(m[g] - mx[g]);
        l[g] = l[g] * alpha + lt[g];
        acc0[g] *= alpha;
        acc1[g] *= alpha;
        m[g] = mx[g];
    }
    const int cnt = (int)ax_min_ll(FA_NT, e - kb);
    for (int u = 0; u < cnt; ++u) {
        bool any = false;
        AX_UNROLL
        for (int g = 0; g < G; ++g) any = any || pr[g * FA_NT + u] != 0.0f;
        if (any) {
            const TKV *vr = vp + (ax_u64)(kb + u - j0) * v_ss;
            const float v0 = t < D ? ax_load(vr, t) : 0.0f;
            const float v1 = t + FA_NT < D ? ax_load(vr, t + FA_NT) : 0.0f;
            AX_UNROLL
            for (int g = 0; g < G; ++g) {
                const float pu = pr[g * FA_NT + u];
                if (pu != 0.0f) {
                    acc0[g] += pu * v0;
                    acc1[g] += pu * v1;
                }
            }
        }
    }
    AX_SYNC();
}

template <typename TQ, typename TKV, int G, bool PAGED>
AX_DEV void fa_rowg_core(FA_PARAMS(TQ, TKV), const ax_pages *pg) {
    float *sm = ax_smem();
    float *qs = sm;              // G * D
    float *red = qs + G * D;     // G * FA_NT
    float *pr = red + G * FA_NT; // G * FA_NT
    const ax_u32 t = AX_TID;
    const ax_u32 split = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    const ax_u32 packs = Hq / G;
    const ax_u32 group = Hq / Hkv;
    for (ax_u64 rowi = AX_BID_X; rowi < (ax_u64)Lq * packs; rowi += AX_NBLK_X) {
        const ax_u32 i = (ax_u32)(rowi / packs);
        const ax_u32 h0 = (ax_u32)(rowi % packs) * G;
        const ax_u32 hk = h0 / group;
        AX_SYNC();
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            const TQ *qp = q + b * q_sb + i * q_ss + (h0 + g) * q_sh;
            for (ax_u32 d = t; d < D; d += FA_NT) qs[g * D + d] = scale * ax_load(qp, d);
        }
        AX_SYNC();
        const long long pos = (long long)i + (long long)Lk - (long long)Lq;
        long long lo = wl >= 0 ? ax_max_ll(pos - wl, 0) : 0;
        long long hi = wr >= 0 ? ax_min_ll(pos + wr + 1, (long long)Lk) : (long long)Lk;
        if (lo > (long long)Lk) lo = Lk;
        if (hi < lo) hi = lo;
        const long long chunk = (hi - lo + n_splits - 1) / n_splits;
        long long s_lo = lo + (long long)split * chunk;
        long long s_hi = ax_min_ll(s_lo + chunk, hi);
        if (s_lo > hi) s_lo = hi;
        if (s_hi < s_lo) s_hi = s_lo;
        float slope[G], m[G], l[G], acc0[G], acc1[G];
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            slope[g] = alibi ? alibi[(ax_u64)b * alibi_bs + h0 + g] : 0.0f;
            m[g] = -AX_INF;
            l[g] = 0.0f;
            acc0[g] = 0.0f;
            acc1[g] = 0.0f;
        }
        if (!PAGED) {
            const TKV *kp = k + b * k_sb + hk * k_sh;
            const TKV *vp = v + b * v_sb + hk * v_sh;
            for (long long kb = s_lo; kb < s_hi; kb += FA_NT)
                fa_rowg_window<TKV, G>(qs, red, pr, kp, vp, k_ss, v_ss, 0, kb, s_hi, D, kd, softcap,
                                       slope, pos, m, l, acc0, acc1);
        } else {
            long long j0 = 0;
            for (ax_u32 bi = 0; bi < pg->n && j0 < s_hi; ++bi) {
                const long long j1 = j0 + (long long)pg->rows[bi];
                const long long a = ax_max_ll(s_lo, j0), e = ax_min_ll(s_hi, j1);
                if (a < e) {
                    const ax_u64 bh = (ax_u64)b * Hkv + hk;
                    const TKV *kp = reinterpret_cast<const TKV *>(pg->k[bi]) +
                                    bh * ((ax_u64)pg->k_cap[bi] * k_ss);
                    const TKV *vp = reinterpret_cast<const TKV *>(pg->v[bi]) +
                                    bh * ((ax_u64)pg->v_cap[bi] * v_ss);
                    for (long long kb = a; kb < e; kb += FA_NT)
                        fa_rowg_window<TKV, G>(qs, red, pr, kp, vp, k_ss, v_ss, j0, kb, e, D, kd,
                                               softcap, slope, pos, m, l, acc0, acc1);
                }
                j0 = j1;
            }
        }
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            const ax_u64 orow = ((ax_u64)b * Lq + i) * Hq + h0 + g;
            const float inv = l[g] > 0.0f ? vd / l[g] : 0.0f;
            if (n_splits == 1) {
                if (t < D) ax_store(o, orow * D + t, acc0[g] * inv);
                if (t + FA_NT < D) ax_store(o, orow * D + t + FA_NT, acc1[g] * inv);
            } else {
                const ax_u64 prow = (ax_u64)split * rows_total + orow;
                if (t < D) part_o[prow * D + t] = acc0[g] * inv;
                if (t + FA_NT < D) part_o[prow * D + t + FA_NT] = acc1[g] * inv;
                if (t == 0) part_lse[prow] = l[g] > 0.0f ? m[g] + logf(l[g]) : -AX_INF;
            }
        }
    }
}

template <typename TKV>
AX_DEV float fa_load_paged(const ax_pages *pg, ax_u32 bi_start, long long j0_start, long long j,
                           ax_u32 d, ax_u64 bh, ax_u64 row_stride, bool is_v) {
    ax_u32 bi = bi_start;
    long long j0 = j0_start;
    while (bi + 1u < pg->n && j >= j0 + (long long)pg->rows[bi]) {
        j0 += (long long)pg->rows[bi];
        ++bi;
    }
    const ax_u64 cap = is_v ? (ax_u64)pg->v_cap[bi] : (ax_u64)pg->k_cap[bi];
    const ax_u64 ptr = is_v ? pg->v[bi] : pg->k[bi];
    const TKV *base = reinterpret_cast<const TKV *>(ptr) + bh * (cap * row_stride);
    return ax_load(base, (ax_u64)(j - j0) * row_stride + d);
}

template <typename TQ, typename TKV, bool PAGED, int BR, int BC, int DMAX>
AX_DEV void fa_tile_core(FA_PARAMS(TQ, TKV), const ax_pages *pg) {
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
    if (!PAGED && varlen) {
        q0 = cu_q[b];
        lq = cu_q[b + 1] - cu_q[b];
        k0 = cu_k[b];
        lk = cu_k[b + 1] - cu_k[b];
    }
    const ax_u32 hk = h / (Hq / Hkv);
    const ax_u64 bh = (ax_u64)b * Hkv + hk;
    const TQ *qb = (!PAGED && varlen) ? q + q0 * q_ss + h * q_sh : q + b * q_sb + h * q_sh;
    const TKV *kb_ = PAGED ? nullptr : (varlen ? k + k0 * k_ss + hk * k_sh : k + b * k_sb + hk * k_sh);
    const TKV *vb_ = PAGED ? nullptr : (varlen ? v + k0 * v_ss + hk * v_sh : v + b * v_sb + hk * v_sh);
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
        ax_u32 bi_start = 0;
        long long j0_start = 0;
        for (long long kt = lo; kt < hi; kt += BC) {
            if (PAGED) {
                while (bi_start + 1u < pg->n && j0_start + (long long)pg->rows[bi_start] <= kt) {
                    j0_start += (long long)pg->rows[bi_start];
                    ++bi_start;
                }
            }
            AX_SYNC();
            for (ax_u32 idx = t; idx < BC * D; idx += FA_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                float kv = 0.0f;
                if (j < hi) {
                    kv = PAGED ? fa_load_paged<TKV>(pg, bi_start, j0_start, j, d, bh, k_ss, false)
                               : ax_load(kb_, (ax_u64)j * k_ss + d);
                    kv *= kd;
                }
                KVs[c * DP + d] = kv;
            }
            AX_SYNC();
            const bool full_non_causal = (wl < 0 && wr < 0 && i0 + BR <= (long long)lq && kt + BC <= hi);
            if (full_non_causal) {
                // Branchless non-causal ViT interior tile fastpath (eliminates per-element window checks).
                for (ax_u32 idx = t; idx < BR * BC; idx += FA_NT) {
                    const ax_u32 rr = idx / BC, c = idx % BC;
                    const float *qr = Qs + rr * DP;
                    const float *kr = KVs + c * DP;
                    float dot = 0.0f;
                    for (ax_u32 d = 0; d < D; ++d) dot += qr[d] * kr[d];
                    if (softcap > 0.0f || slope != 0.0f) {
                        const long long p2 = (i0 + rr) + off;
                        dot = fa_logit(dot, softcap, slope, p2 - (kt + c));
                    }
                    S[idx] = dot;
                }
            } else {
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
            }
            AX_SYNC();
            // V tile into shared memory while every lane finds its row maximum
            for (ax_u32 idx = t; idx < BC * D; idx += FA_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                float vv = 0.0f;
                if (j < hi) {
                    vv = PAGED ? fa_load_paged<TKV>(pg, bi_start, j0_start, j, d, bh, v_ss, true)
                               : ax_load(vb_, (ax_u64)j * v_ss + d);
                }
                KVs[c * DP + d] = vv;
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

template <typename TQ, typename TKV, int BR, int BC, int DMAX>
AX_DEV void fa_tile_impl(FA_PARAMS(TQ, TKV)) {
    fa_tile_core<TQ, TKV, false, BR, BC, DMAX>(FA_ARGS, nullptr);
}

template <typename TQ, typename TKV, int BR, int BC, int DMAX>
AX_DEV void fa_tilep_impl(FA_PARAMS(TQ, TKV), const ax_pages &pg) {
    fa_tile_core<TQ, TKV, true, BR, BC, DMAX>(FA_ARGS, &pg);
}

// Merges the normalized partial outputs of n_splits key splits with their log-sum-exp. The
// split weights exp(lse_s - max) are computed once per row (shared memory: n_splits floats);
// the sums run in the same order as with per-element weights, so the result is unchanged.
template <typename TO>
AX_DEV void fa_combine_impl(const float *part_o, const float *part_lse, TO *o, ax_u64 rows,
                            ax_u32 D, ax_u32 n_splits) {
    float *w = ax_smem(); // n_splits
    const ax_u32 t = AX_TID;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        float mx = -AX_INF;
        for (ax_u32 s = 0; s < n_splits; ++s) mx = fmaxf(mx, part_lse[(ax_u64)s * rows + row]);
        AX_SYNC();
        for (ax_u32 s = t; s < n_splits; s += FA_NT) {
            const float e = part_lse[(ax_u64)s * rows + row];
            w[s] = e == -AX_INF ? 0.0f : expf(e - mx);
        }
        AX_SYNC();
        float wsum = 0.0f;
        for (ax_u32 s = 0; s < n_splits; ++s) wsum += w[s];
        for (ax_u32 d = t; d < D; d += FA_NT) {
            float acc = 0.0f;
            for (ax_u32 s = 0; s < n_splits; ++s) {
                const float ws = w[s];
                if (ws != 0.0f) acc += ws * part_o[((ax_u64)s * rows + row) * D + d];
            }
            ax_store(o, row * D + d, wsum > 0.0f ? acc / wsum : 0.0f);
        }
    }
}

// Split-Invariant deterministic combine: reduces `n_splits` using a fixed pairwise binary tree
// over Log-Sum-Exp so outputs are invariant to reduction order (opt-in; preserves `fa_combine_impl`
// verbatim for historical bit-exact contracts).
// Shared memory: (2 * 64 + 64) floats (supports up to FA_MAX_SPLITS = 64).
template <typename TO>
AX_DEV void fa_combine_tree_impl(const float *part_o, const float *part_lse, TO *o, ax_u64 rows,
                                 ax_u32 D, ax_u32 n_splits) {
    float *w = ax_smem();           // up to 64 split weights
    float *tree = w + 64;           // 64 floats for deterministic pairwise binary tree reduction
    const ax_u32 t = AX_TID;
    ax_u32 pow2 = 1u;
    while (pow2 < n_splits) pow2 <<= 1u;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        if (t < pow2) {
            tree[t] = (t < n_splits) ? part_lse[(ax_u64)t * rows + row] : -AX_INF;
        }
        AX_SYNC();
        for (ax_u32 s = pow2 >> 1u; s > 0u; s >>= 1u) {
            if (t < s) tree[t] = fmaxf(tree[t], tree[t + s]);
            AX_SYNC();
        }
        const float mx = tree[0];
        if (t < pow2) {
            const float e = (t < n_splits) ? part_lse[(ax_u64)t * rows + row] : -AX_INF;
            const float wt = (e == -AX_INF) ? 0.0f : expf(e - mx);
            w[t] = wt;
            tree[t] = wt;
        }
        AX_SYNC();
        for (ax_u32 s = pow2 >> 1u; s > 0u; s >>= 1u) {
            if (t < s) tree[t] = tree[t] + tree[t + s];
            AX_SYNC();
        }
        const float wsum = tree[0];
        AX_SYNC();
        for (ax_u32 d = t; d < D; d += FA_NT) {
            // Fixed pairwise reduction over split outputs (up to 64 splits in registers).
            float buf[64];
            for (ax_u32 s = 0u; s < pow2; ++s) {
                const float ws = (s < n_splits) ? w[s] : 0.0f;
                buf[s] = (ws != 0.0f) ? (ws * part_o[((ax_u64)s * rows + row) * D + d]) : 0.0f;
            }
            for (ax_u32 step = pow2 >> 1u; step > 0u; step >>= 1u) {
                for (ax_u32 s = 0u; s < step; ++s) {
                    buf[s] = buf[s] + buf[s + step];
                }
            }
            ax_store(o, row * D + d, wsum > 0.0f ? buf[0] / wsum : 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Coalesced Warp-Per-Key 2D GQA Decode Kernel (4 sub-warps x 32 lanes = 128 threads)
// ---------------------------------------------------------------------------
// Each 32-lane sub-warp reads contiguous D elements of one key j (100% coalesced across lanes),
// evaluates G_h query heads in registers via warp32 shuffle sum, accumulates V directly into
// per-lane registers with 0 shared-memory writes in the inner loop, and merges the 4 sub-warps
// once at loop exit. Supports optional `sink_logits[Hq]` (m = sink[h], l = 1.0f) and
// speculative `q_tree_mask[B * Lq]` (u64 bitmask over the trailing Lq draft tokens).
// Grid: (Lq * (Hq / G), n_splits, B), Block: (FA_NT = 128, 1, 1).
// Shared memory: (4 * G * (D + 2)) * sizeof(float).
template <typename TQ, typename TKV, int G, int DMAX>
AX_DEV void fa_decode_2d_core(
    const TQ *q, const TKV *k, const TKV *v, TQ *o, float *part_o, float *part_lse,
    const float *sink_logits, const ax_u64 *q_tree_mask,
    ax_u32 Lq, ax_u32 Lk, ax_u32 Hq, ax_u32 Hkv, ax_u32 D,
    ax_u64 q_sb, ax_u64 q_sh, ax_u64 q_ss,
    ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_ss,
    ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_ss,
    float scale, float softcap, int wl, int wr, float kd, float vd,
    ax_u32 n_splits, ax_u64 rows_total
) {
    constexpr int E = DMAX / 32;
    const ax_u32 t = AX_TID;
    const ax_u32 wid = t >> 5;
    const ax_u32 lane = t & 31u;
    const ax_u32 s_idx = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    const ax_u32 n_groups = Hq / (ax_u32)G;
    const ax_u32 total_tasks = Lq * n_groups;
    const ax_u32 task = AX_BID_X;
    if (task >= total_tasks || D > (ax_u32)DMAX) return;

    const ax_u32 qi = task / n_groups;
    const ax_u32 g_idx = task - qi * n_groups;
    const ax_u32 h0 = g_idx * (ax_u32)G;
    const ax_u32 hk = h0 / (Hq / Hkv);

    const long long pos = (long long)qi + (long long)Lk - (long long)Lq;
    long long lo = wl >= 0 ? ax_max_ll(pos - wl, 0) : 0;
    long long hi = wr >= 0 ? ax_min_ll(pos + wr + 1, (long long)Lk) : (long long)Lk;
    if (lo > (long long)Lk) lo = Lk;
    if (hi < lo) hi = lo;
    const long long span = hi - lo;
    const long long chunk = (span + (long long)n_splits - 1) / (long long)n_splits;
    const long long k_lo = lo + (long long)s_idx * chunk;
    const long long k_hi = ax_min_ll(k_lo + chunk, hi);

    const long long draft_start = (long long)Lk - (long long)Lq;
    const ax_u64 tree_mask = (q_tree_mask != nullptr) ? q_tree_mask[(ax_u64)b * Lq + qi] : 0ull;

    // Load G query slices (scaled by `scale * kd`) into per-lane registers.
    float q_reg[G][E];
    float acc[G][E];
    float m[G];
    float l[G];
    AX_UNROLL
    for (int g = 0; g < G; ++g) {
        const ax_u64 q_off = (ax_u64)b * q_sb + (ax_u64)(h0 + g) * q_sh + (ax_u64)qi * q_ss;
        AX_UNROLL
        for (int e = 0; e < E; ++e) {
            const ax_u32 d = lane + (ax_u32)(e * 32);
            q_reg[g][e] = (d < D) ? (ax_load(q, q_off + d) * scale * kd) : 0.0f;
            acc[g][e] = 0.0f;
        }
        // Zero-overhead Attention Sink initialization (only on split 0, warp 0 to avoid duplicate sink).
        if (sink_logits != nullptr && s_idx == 0u && wid == 0u) {
            m[g] = sink_logits[h0 + g];
            l[g] = 1.0f;
        } else {
            m[g] = -AX_INF;
            l[g] = 0.0f;
        }
    }

    const TKV *kb = k + (ax_u64)b * k_sb + (ax_u64)hk * k_sh;
    const TKV *vb = v + (ax_u64)b * v_sb + (ax_u64)hk * v_sh;

    for (long long j = k_lo + (long long)wid; j < k_hi; j += 4) {
        if (q_tree_mask != nullptr && j >= draft_start) {
            const ax_u32 bit = (ax_u32)(j - draft_start);
            if (bit < 64u && ((tree_mask >> bit) & 1ull) == 0ull) {
                continue;
            }
        }
        float k_reg[E];
        float v_reg[E];
        AX_UNROLL
        for (int e = 0; e < E; ++e) {
            const ax_u32 d = lane + (ax_u32)(e * 32);
            k_reg[e] = (d < D) ? ax_load(kb, (ax_u64)j * k_ss + d) : 0.0f;
            v_reg[e] = (d < D) ? ax_load(vb, (ax_u64)j * v_ss + d) : 0.0f;
        }
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            float dot = 0.0f;
            AX_UNROLL
            for (int e = 0; e < E; ++e) dot = fmaf(q_reg[g][e], k_reg[e], dot);
#if AX_HAS_WARP32_SHFL
            dot = ax_warp32_sum(dot);
#else
            float *ws = ax_smem() + wid * 32u;
            ws[lane] = dot;
            AX_SYNC();
            for (ax_u32 s = 16u; s > 0u; s >>= 1u) {
                if (lane < s) ws[lane] += ws[lane + s];
                AX_SYNC();
            }
            dot = ws[0];
            AX_SYNC();
#endif
            float sc = dot;
            if (softcap > 0.0f) sc = softcap * tanhf(sc / softcap);
            if (sc > m[g]) {
                const float alpha = (m[g] == -AX_INF) ? 0.0f : expf(m[g] - sc);
                l[g] = l[g] * alpha + 1.0f;
                AX_UNROLL
                for (int e = 0; e < E; ++e) acc[g][e] = fmaf(1.0f, v_reg[e], acc[g][e] * alpha);
                m[g] = sc;
            } else {
                const float p = expf(sc - m[g]);
                l[g] += p;
                AX_UNROLL
                for (int e = 0; e < E; ++e) acc[g][e] = fmaf(p, v_reg[e], acc[g][e]);
            }
        }
    }

    // Merge the 4 sub-warps once via shared memory.
    float *sm = ax_smem();
    float *sm_m = sm;                    // 4 * G
    float *sm_l = sm_m + 4 * G;          // 4 * G
    float *sm_o = sm_l + 4 * G;          // 4 * G * D
    AX_UNROLL
    for (int g = 0; g < G; ++g) {
        if (lane == 0u) {
            sm_m[wid * G + g] = m[g];
            sm_l[wid * G + g] = l[g];
        }
        AX_UNROLL
        for (int e = 0; e < E; ++e) {
            const ax_u32 d = lane + (ax_u32)(e * 32);
            if (d < D) sm_o[((wid * G + g) * D) + d] = acc[g][e];
        }
    }
    AX_SYNC();

    if (wid == 0u) {
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            float gm = -AX_INF;
            AX_UNROLL
            for (int w = 0; w < 4; ++w) gm = fmaxf(gm, sm_m[w * G + g]);
            float gl = 0.0f;
            float ws[4];
            AX_UNROLL
            for (int w = 0; w < 4; ++w) {
                const float mw = sm_m[w * G + g];
                ws[w] = (mw == -AX_INF) ? 0.0f : expf(mw - gm);
                gl += sm_l[w * G + g] * ws[w];
            }
            const ax_u64 row = ((ax_u64)b * Lq + qi) * Hq + (h0 + g);
            const float inv = (gl > 0.0f) ? (vd / gl) : 0.0f;
            if (n_splits == 1u) {
                AX_UNROLL
                for (int e = 0; e < E; ++e) {
                    const ax_u32 d = lane + (ax_u32)(e * 32);
                    if (d < D) {
                        float vsum = 0.0f;
                        AX_UNROLL
                        for (int w = 0; w < 4; ++w) {
                            if (ws[w] != 0.0f) vsum += ws[w] * sm_o[((w * G + g) * D) + d];
                        }
                        ax_store(o, row * D + d, vsum * inv);
                    }
                }
            } else {
                const ax_u64 pout = ((ax_u64)s_idx * rows_total + row) * D;
                AX_UNROLL
                for (int e = 0; e < E; ++e) {
                    const ax_u32 d = lane + (ax_u32)(e * 32);
                    if (d < D) {
                        float vsum = 0.0f;
                        AX_UNROLL
                        for (int w = 0; w < 4; ++w) {
                            if (ws[w] != 0.0f) vsum += ws[w] * sm_o[((w * G + g) * D) + d];
                        }
                        part_o[pout + d] = vsum * inv;
                    }
                }
                if (lane == 0u) {
                    part_lse[(ax_u64)s_idx * rows_total + row] = (gl > 0.0f) ? (gm + logf(gl)) : -AX_INF;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// O(1) Device Block-Table Paged Varlen FlashAttention (Zero Host Sync, Zero Copy)
// ---------------------------------------------------------------------------
// Eliminates `seqlens_k.to_device(Cpu)`, `block_table.to_device(Cpu)`, `Tensor::cat`,
// and the `AX_MAX_PAGES = 128` cap. Supports optional `mm_prefix_ranges[B * 2]` (per-batch
// [start, end) bidirectional span within causal attention) and `sink_logits[Hq]`.
// K/V pool layout: [num_blocks, page_size, Hkv, D] contiguous.
// Grid: (ceil(max_q / BR), Hq, B), Block: (FA_NT = 128, 1, 1).
template <typename TQ, typename TKV, int BR, int BC, int DMAX>
AX_DEV void fa_tile_bt_varlen_impl(
    const TQ *q, const TKV *k_pool, const TKV *v_pool, TQ *o,
    const ax_u32 *cu_q, const ax_u32 *cu_k, const int *block_table,
    const ax_u32 *mm_prefix_ranges, const float *sink_logits,
    ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u32 page_size, ax_u32 bt_stride,
    float scale, float softcap, int wl, int wr
) {
    const int TPR = FA_NT / BR;
    const int NACC = DMAX / TPR;
    const ax_u32 DP = D + 1;
    float *sm = ax_smem();
    float *Qs = sm;
    float *KVs = Qs + BR * DP;
    float *S = KVs + BC * DP;
    const ax_u32 t = AX_TID;
    const ax_u32 r = t / TPR;
    const ax_u32 lane = t % TPR;
    const ax_u32 h = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    const ax_u32 hk = h / (Hq / Hkv);

    const ax_u64 q0 = cu_q[b];
    const ax_u32 lq = cu_q[b + 1] - cu_q[b];
    const ax_u32 lk = cu_k[b + 1] - cu_k[b];
    if (lq == 0u || lk == 0u) return;

    const ax_u64 q_ss = (ax_u64)Hq * D;
    const ax_u64 q_sh = D;
    const ax_u64 blk_stride = (ax_u64)page_size * Hkv * D;
    const ax_u64 tok_stride = (ax_u64)Hkv * D;
    const ax_u64 h_off = (ax_u64)hk * D;

    const TQ *qb = q + q0 * q_ss + h * q_sh;
    const int *bt = block_table + (ax_u64)b * bt_stride;
    const long long off = (long long)lk - (long long)lq;
    const bool has_prefix = (mm_prefix_ranges != nullptr);
    const long long p_start = has_prefix ? (long long)mm_prefix_ranges[b * 2u] : 0;
    const long long p_end   = has_prefix ? (long long)mm_prefix_ranges[b * 2u + 1u] : 0;

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
        if (has_prefix && i0 + off < p_end && last + off >= p_start) {
            if (p_start < lo) lo = p_start;
            if (p_end > hi) hi = ax_min_ll(p_end, (long long)lk);
        }
        if (lo > (long long)lk) lo = lk;
        if (hi < lo) hi = lo;

        const long long i = i0 + r;
        float m = (sink_logits != nullptr) ? sink_logits[h] : -AX_INF;
        float l = (sink_logits != nullptr) ? 1.0f : 0.0f;
        float acc[NACC];
        AX_UNROLL
        for (int a = 0; a < NACC; ++a) acc[a] = 0.0f;

        for (long long kt = lo; kt < hi; kt += BC) {
            AX_SYNC();
            for (ax_u32 idx = t; idx < BC * D; idx += FA_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                float kv = 0.0f;
                if (j < hi) {
                    const ax_u32 lblk = (ax_u32)j / page_size;
                    const ax_u32 in_p = (ax_u32)j - lblk * page_size;
                    const ax_u64 pblk = (ax_u64)(ax_u32)bt[lblk];
                    kv = ax_load(k_pool, pblk * blk_stride + (ax_u64)in_p * tok_stride + h_off + d);
                }
                KVs[c * DP + d] = kv;
            }
            AX_SYNC();
            for (ax_u32 idx = t; idx < BR * BC; idx += FA_NT) {
                const ax_u32 rr = idx / BC, c = idx % BC;
                const long long ii = i0 + rr, j = kt + c;
                const long long p2 = ii + off;
                const bool in_win = (wl < 0 || j >= p2 - wl) && (wr < 0 || j <= p2 + wr);
                const bool in_pfx = has_prefix && (p2 >= p_start && p2 < p_end && j >= p_start && j < p_end);
                float s = -AX_INF;
                if (ii < (long long)lq && j < hi && (in_win || in_pfx)) {
                    const float *qr = Qs + rr * DP;
                    const float *kr = KVs + c * DP;
                    float dot = 0.0f;
                    for (ax_u32 d = 0; d < D; ++d) dot += qr[d] * kr[d];
                    s = (softcap > 0.0f) ? (softcap * tanhf(dot / softcap)) : dot;
                }
                S[idx] = s;
            }
            AX_SYNC();
            for (ax_u32 idx = t; idx < BC * D; idx += FA_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                float vv = 0.0f;
                if (j < hi) {
                    const ax_u32 lblk = (ax_u32)j / page_size;
                    const ax_u32 in_p = (ax_u32)j - lblk * page_size;
                    const ax_u64 pblk = (ax_u64)(ax_u32)bt[lblk];
                    vv = ax_load(v_pool, pblk * blk_stride + (ax_u64)in_p * tok_stride + h_off + d);
                }
                KVs[c * DP + d] = vv;
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
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            const ax_u64 orow = (q0 + (ax_u64)i) * Hq + h;
            AX_UNROLL
            for (int a = 0; a < NACC; ++a) {
                const ax_u32 d = lane + a * TPR;
                if (d < D) ax_store(o, orow * D + d, acc[a] * inv);
            }
        }
    }
}

// Kernel entry points. NAME is the exported symbol.
#define FA_ROW_KERNEL(NAME, TQ, TKV)                                                               \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV)) { fa_row_impl<TQ, TKV>(FA_ARGS); }

// Paged row kernel: the FA_PARAMS (k / v unused) followed by the block table by value.
#define FA_ROWP_KERNEL(NAME, TQ, TKV)                                                              \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV), const ax_pages pg) {                  \
        fa_rowp_impl<TQ, TKV>(FA_ARGS, pg);                                                        \
    }

// GQA-packed row kernels, dense and paged (G query heads per workgroup).
#define FA_ROWG_KERNEL(NAME, TQ, TKV, G)                                                           \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV)) {                                     \
        fa_rowg_core<TQ, TKV, G, false>(FA_ARGS, nullptr);                                         \
    }

#define FA_ROWPG_KERNEL(NAME, TQ, TKV, G)                                                          \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV), const ax_pages pg) {                  \
        fa_rowg_core<TQ, TKV, G, true>(FA_ARGS, &pg);                                              \
    }

#define FA_ROWG_ALL(Q, KV, TQ, TKV)                                                                \
    FA_ROWG_KERNEL(fa_rowg2_##Q##_##KV, TQ, TKV, 2)                                                \
    FA_ROWG_KERNEL(fa_rowg4_##Q##_##KV, TQ, TKV, 4)                                                \
    FA_ROWG_KERNEL(fa_rowg8_##Q##_##KV, TQ, TKV, 8)                                                \
    FA_ROWPG_KERNEL(fa_rowpg2_##Q##_##KV, TQ, TKV, 2)                                              \
    FA_ROWPG_KERNEL(fa_rowpg4_##Q##_##KV, TQ, TKV, 4)                                              \
    FA_ROWPG_KERNEL(fa_rowpg8_##Q##_##KV, TQ, TKV, 8)

#define FA_TILE_KERNEL(NAME, TQ, TKV, BR, BC, DMAX)                                                \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV)) {                                     \
        fa_tile_impl<TQ, TKV, BR, BC, DMAX>(FA_ARGS);                                              \
    }

#define FA_TILEP_KERNEL(NAME, TQ, TKV, BR, BC, DMAX)                                               \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(FA_PARAMS(TQ, TKV), const ax_pages pg) {                  \
        fa_tilep_impl<TQ, TKV, BR, BC, DMAX>(FA_ARGS, pg);                                         \
    }

#define FA_COMBINE_KERNEL(NAME, TO)                                                                \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(const float *part_o, const float *part_lse, TO *o,        \
                                         ax_u64 rows, ax_u32 D, ax_u32 n_splits) {                 \
        fa_combine_impl<TO>(part_o, part_lse, o, rows, D, n_splits);                               \
    }

#define FA_COMBINE_TREE_KERNEL(NAME, TO)                                                           \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(const float *part_o, const float *part_lse, TO *o,        \
                                         ax_u64 rows, ax_u32 D, ax_u32 n_splits) {                 \
        fa_combine_tree_impl<TO>(part_o, part_lse, o, rows, D, n_splits);                          \
    }

#define FA_TILE_BT_KERNEL(NAME, TQ, TKV, BR, BC, DMAX)                                             \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(                                                          \
        const TQ *q, const TKV *k_pool, const TKV *v_pool, TQ *o,                                  \
        const ax_u32 *cu_q, const ax_u32 *cu_k, const int *block_table,                            \
        const ax_u32 *mm_prefix_ranges, const float *sink_logits,                                  \
        ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u32 page_size, ax_u32 bt_stride,                       \
        float scale, float softcap, int wl, int wr) {                                              \
        fa_tile_bt_varlen_impl<TQ, TKV, BR, BC, DMAX>(                                             \
            q, k_pool, v_pool, o, cu_q, cu_k, block_table, mm_prefix_ranges, sink_logits,          \
            Hq, Hkv, D, page_size, bt_stride, scale, softcap, wl, wr);                             \
    }

#define FA_DECODE_2D_KERNEL(NAME, TQ, TKV, G, DMAX)                                                \
    AX_KERNEL void AX_BOUNDS(FA_NT) NAME(                                                          \
        const TQ *q, const TKV *k, const TKV *v, TQ *o, float *part_o, float *part_lse,            \
        const float *sink_logits, const ax_u64 *q_tree_mask,                                       \
        ax_u32 Lq, ax_u32 Lk, ax_u32 Hq, ax_u32 Hkv, ax_u32 D,                                     \
        ax_u64 q_sb, ax_u64 q_sh, ax_u64 q_ss,                                                     \
        ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_ss,                                                     \
        ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_ss,                                                     \
        float scale, float softcap, int wl, int wr, float kd, float vd,                            \
        ax_u32 n_splits, ax_u64 rows_total) {                                                      \
        fa_decode_2d_core<TQ, TKV, G, DMAX>(                                                       \
            q, k, v, o, part_o, part_lse, sink_logits, q_tree_mask,                                \
            Lq, Lk, Hq, Hkv, D, q_sb, q_sh, q_ss, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss,              \
            scale, softcap, wl, wr, kd, vd, n_splits, rows_total);                                 \
    }

// Tile shapes per tier and head dim bucket (keep in sync with accel::attention::tile_config).
#define FA2_TILES(Q, KV, TQ, TKV)                                                                  \
    FA_TILE_KERNEL(fa2_d64_##Q##_##KV, TQ, TKV, 32, 32, 64)                                        \
    FA_TILE_KERNEL(fa2_d128_##Q##_##KV, TQ, TKV, 16, 32, 128)                                      \
    FA_TILE_KERNEL(fa2_d256_##Q##_##KV, TQ, TKV, 8, 16, 256)

#define FA2_TILESP(Q, KV, TQ, TKV)                                                                 \
    FA_TILEP_KERNEL(fa2p_d64_##Q##_##KV, TQ, TKV, 32, 32, 64)                                      \
    FA_TILEP_KERNEL(fa2p_d128_##Q##_##KV, TQ, TKV, 16, 32, 128)                                    \
    FA_TILEP_KERNEL(fa2p_d256_##Q##_##KV, TQ, TKV, 8, 16, 256)

#define FA3_TILES(Q, KV, TQ, TKV)                                                                  \
    FA_TILE_KERNEL(fa3_d64_##Q##_##KV, TQ, TKV, 64, 64, 64)                                        \
    FA_TILE_KERNEL(fa3_d128_##Q##_##KV, TQ, TKV, 32, 64, 128)                                      \
    FA_TILE_KERNEL(fa3_d256_##Q##_##KV, TQ, TKV, 16, 32, 256)

#define FA3_TILESP(Q, KV, TQ, TKV)                                                                 \
    FA_TILEP_KERNEL(fa3p_d64_##Q##_##KV, TQ, TKV, 64, 64, 64)                                      \
    FA_TILEP_KERNEL(fa3p_d128_##Q##_##KV, TQ, TKV, 32, 64, 128)                                    \
    FA_TILEP_KERNEL(fa3p_d256_##Q##_##KV, TQ, TKV, 16, 32, 256)

#define FA_ALL_FOR(Q, KV, TQ, TKV)                                                                 \
    FA_ROW_KERNEL(fa_row_##Q##_##KV, TQ, TKV)                                                      \
    FA_ROWP_KERNEL(fa_rowp_##Q##_##KV, TQ, TKV)                                                    \
    FA_ROWG_ALL(Q, KV, TQ, TKV)                                                                    \
    FA2_TILES(Q, KV, TQ, TKV)                                                                      \
    FA2_TILESP(Q, KV, TQ, TKV)                                                                     \
    FA3_TILES(Q, KV, TQ, TKV)                                                                      \
    FA3_TILESP(Q, KV, TQ, TKV)
