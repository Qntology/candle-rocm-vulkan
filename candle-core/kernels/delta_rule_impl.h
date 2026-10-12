// Fused Gated DeltaNet kernels (Qwen3.5 / Qwen3-Next / Qwen4-Exp linear attention), see
// candle-core/src/delta_rule.rs for the semantics and candle-core/src/accel/delta.rs for the
// launch planning.
//
// * gdn_scan: one workgroup per (v column tile, v head, batch). The workgroup keeps its
//   [Dk, TV] slice of the recurrent state in registers for the whole sequence: lane
//   (col = tid % TV, grp = tid / TV) owns the RP = DK / (NT / TV) consecutive rows
//   grp * RP .. grp * RP + RP - 1 of column col. Tokens are processed in chunks of TC:
//     1. prologue (fully parallel): causal depthwise conv over [conv_state ; mixed_qkv]
//        (+ bias) and SiLU for the q / k channels of the workgroup's k head and the v channels
//        of its tile, staged in shared memory; per token beta = sigmoid(b),
//        exp(g) with g = neg_a * softplus(a + dt_bias) (torch threshold 20), and the
//        sums q.q, k.k, q.k that give the L2 norms (eps inside the square root) and q.k.
//     2. sequential scan, one barrier per token: with S the state before the token,
//          kv    = exp(g) * S^T k            (k, q L2 normalized, q scaled by 1/sqrt(Dk))
//          delta = beta * (v - kv)
//          o     = exp(g) * S^T q + delta * (k.q)      (= S'^T q)
//          S'    = exp(g) * S + k delta^T
//        S^T k and S^T q are reduced over the row groups through shared memory (double
//        buffered by token parity, so a single barrier per token suffices).
//   The new recurrent state and the new conv state are written out of place: every
//   workgroup owns its state tile; the q / k conv channels of a k head are written by the
//   first v head of the head group (tile 0), the v channels by the tile that reads them.
//   All math is f32 whatever the input dtype. Rows >= dk and columns >= dv are masked, so a
//   kernel built for DK serves any dk <= DK.
// * gdn_gnorm: gated RMS norm y = rmsnorm(x) * w * silu(z), 32 lanes per row, 4 rows per
//   workgroup of 128 lanes, f32 math.
//
// gdn_scan arguments:
//   x, cs_in, cw, cb, a, b, neg_a, dt_bias, s_in, out, cs_out, s_out,
//   T, Hk, Hv, dk, dv, K, flags (1: L2 norm q / k, 2: tiled head order, 4: a / b are f32
//   instead of the activation dtype), eps, scale,
//   x_sb, x_st, a_sb, a_st, b_sb, b_st                 (element strides: batch, token)
// Layouts: x [B, T, C] (channel stride 1), cs_in / cs_out [B, C, K-1] contiguous, cw [C, K]
// f32, cb [C] f32 (or null), a / b [B, T, Hv] (head stride 1), neg_a / dt_bias [Hv] f32,
// s_in / s_out [B, Hv, dk, dv] f32 contiguous (s_in may be null = zeros), out [B, T, Hv, dv].
// C = 2 * Hk * dk + Hv * dv; q channels [0, Hk*dk), k channels [Hk*dk, 2*Hk*dk), v after.
// Grid (ceil(dv / TV), Hv, B), NT lanes, dynamic shared memory gdn_scan_smem_floats * 4.
#pragma once
#include "accel_common.h"

#define GDN_FLAG_L2NORM 1u
#define GDN_FLAG_TILED 2u
#define GDN_FLAG_AB_F32 4u

#if defined(CANDLE_ACCEL_HOST)
#define GDN_BOUNDS(nt, minb)
#elif defined(__HIP__) || defined(__HIPCC__)
#define GDN_BOUNDS(nt, minb) __launch_bounds__(nt)
#else
#define GDN_BOUNDS(nt, minb) __launch_bounds__(nt, minb)
#endif

struct alignas(16) gdn_f4 {
    float x, y, z, w;
};

AX_DEV float gdn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
AX_DEV float gdn_silu(float x) { return x / (1.0f + expf(-x)); }
AX_DEV float gdn_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Shared memory of gdn_scan in floats (mirrored by accel::delta::scan_smem_floats):
// q [TC][DK], k [TC][DK], v [TC][TV], reductions [4 * NT] (norm partials alias the
// double-buffered row-group partials), per token scalars [5][TC].
#define GDN_SCAN_SMEM_FLOATS(DK, TV, NT, TC) (2 * (TC) * (DK) + (TC) * (TV) + 4 * (NT) + 5 * (TC))

#define GDN_SCAN_PARAMS(TX)                                                                        \
    const TX *x, const TX *cs_in, const float *cw, const float *cb, const void *a, const void *bb, \
        const float *neg_a, const float *dt_bias, const float *s_in, TX *out, TX *cs_out,          \
        float *s_out, ax_u32 seq, ax_u32 hk, ax_u32 hv, ax_u32 dk, ax_u32 dv, ax_u32 kc,           \
        ax_u32 flags, float eps, float scale, ax_u64 x_sb, ax_u64 x_st, ax_u64 a_sb, ax_u64 a_st,  \
        ax_u64 b_sb, ax_u64 b_st

// Element i of [conv_state ; tokens] for channel c of batch b (i < K-1: conv state).
template <typename TX>
AX_DEV float gdn_xe(const TX *x, const TX *cs_in, ax_u64 b, ax_u64 c, ax_u64 i, ax_u32 kc,
                    ax_u64 C, ax_u64 x_sb, ax_u64 x_st) {
    const ax_u64 km1 = kc - 1;
    if (i < km1) return cs_in ? ax_load(cs_in, (b * C + c) * km1 + i) : 0.0f;
    return ax_load(x, b * x_sb + (i - km1) * x_st + c);
}

// Gate input (a or b) element i, stored in the activation dtype or in f32.
template <typename TX>
AX_DEV float gdn_gate(const void *p, ax_u64 i, bool f32) {
    return f32 ? reinterpret_cast<const float *>(p)[i] : ax_load(reinterpret_cast<const TX *>(p), i);
}

template <typename TX, int DK, int TV, int NT, int TC>
AX_DEV void gdn_scan_body(GDN_SCAN_PARAMS(TX)) {
    constexpr int NG = NT / TV;      // row groups
    constexpr int RP = DK / NG;      // state rows per lane
    constexpr int NCH = 2 * DK + TV; // conv channels staged per chunk
    constexpr int NSEG = NT / TC;    // lanes per token in the norm reduction
    static_assert(NT % TV == 0 && DK % NG == 0 && RP % 4 == 0, "gdn_scan: bad tile");
    static_assert(NT % TC == 0 && DK % NSEG == 0, "gdn_scan: bad chunk");
    float *sm = ax_smem();
    float *s_q = sm;               // [TC][DK]
    float *s_k = s_q + TC * DK;    // [TC][DK]
    float *s_v = s_k + TC * DK;    // [TC][TV]
    float *s_red = s_v + TC * TV;  // [4 * NT]
    float *s_sc = s_red + 4 * NT;  // [5][TC]: exp(g), beta, q scale, k scale, q.k

    const ax_u32 tid = AX_TID;
    const ax_u32 col = tid % TV;
    const ax_u32 grp = tid / TV;
    const ax_u32 tile = AX_BID_X;
    const ax_u32 h = AX_BID_Y;
    const ax_u64 b = AX_BID_Z;
    const ax_u32 hg = hv / hk;
    const bool tiled = (flags & GDN_FLAG_TILED) != 0;
    const ax_u32 kh = tiled ? h % hk : h / hg;
    const ax_u32 vcol = tile * TV + col;
    const bool col_ok = vcol < dv;
    const ax_u64 key_dim = (ax_u64)hk * dk;
    const ax_u64 C = 2 * key_dim + (ax_u64)hv * dv;
    const ax_u64 km1 = kc - 1;
    const ax_u64 q_c0 = (ax_u64)kh * dk;
    const ax_u64 k_c0 = key_dim + (ax_u64)kh * dk;
    const ax_u64 v_c0 = 2 * key_dim + (ax_u64)h * dv + (ax_u64)tile * TV;
    const ax_u32 v_n = dv - tile * TV < (ax_u32)TV ? dv - tile * TV : (ax_u32)TV;

    // New conv state: the last K-1 entries of [conv_state ; tokens].
    if (km1 > 0) {
        const bool writes_qk = tile == 0 && (tiled ? h < hk : h % hg == 0);
        const ax_u64 n_qk = writes_qk ? 2ull * dk : 0ull;
        const ax_u64 n = (n_qk + v_n) * km1;
        for (ax_u64 e = tid; e < n; e += NT) {
            const ax_u64 ci = e / km1, m = e % km1;
            ax_u64 c;
            if (ci < n_qk) c = ci < dk ? q_c0 + ci : k_c0 + (ci - dk);
            else c = v_c0 + (ci - n_qk);
            const float v = gdn_xe(x, cs_in, b, c, (ax_u64)seq + m, kc, C, x_sb, x_st);
            ax_store(cs_out, (b * C + c) * km1 + m, v);
        }
    }

    const ax_u32 row0 = grp * RP;
    const ax_u64 s_base = ((b * hv + h) * (ax_u64)dk) * dv;
    float S[RP];
    AX_UNROLL
    for (int m = 0; m < RP; ++m) {
        const ax_u32 row = row0 + m;
        S[m] = (s_in && col_ok && row < dk) ? s_in[s_base + (ax_u64)row * dv + vcol] : 0.0f;
    }

    const bool l2 = (flags & GDN_FLAG_L2NORM) != 0;
    const float na = neg_a[h];
    const float dtb = dt_bias[h];
    ax_u32 par = 0;
    for (ax_u32 t0 = 0; t0 < seq; t0 += TC) {
        const ax_u32 tc = seq - t0 < (ax_u32)TC ? seq - t0 : (ax_u32)TC;
        // 1a. conv + SiLU of the chunk.
        for (ax_u32 idx = tid; idx < (ax_u32)(NCH * TC); idx += NT) {
            const ax_u32 ch = idx % NCH, tt = idx / NCH;
            float val = 0.0f;
            ax_u64 c = 0;
            bool ok = tt < tc;
            if (ch < (ax_u32)DK) {
                ok = ok && ch < dk;
                c = q_c0 + ch;
            } else if (ch < (ax_u32)(2 * DK)) {
                ok = ok && ch - DK < dk;
                c = k_c0 + (ch - DK);
            } else {
                ok = ok && ch - 2 * DK < v_n;
                c = v_c0 + (ch - 2 * DK);
            }
            if (ok) {
                float acc = cb ? cb[c] : 0.0f;
                const ax_u64 i0 = (ax_u64)t0 + tt;
                for (ax_u32 j = 0; j < kc; ++j)
                    acc = fmaf(cw[c * kc + j], gdn_xe(x, cs_in, b, c, i0 + j, kc, C, x_sb, x_st), acc);
                val = gdn_silu(acc);
            }
            if (ch < (ax_u32)DK) s_q[tt * DK + ch] = val;
            else if (ch < (ax_u32)(2 * DK)) s_k[tt * DK + (ch - DK)] = val;
            else s_v[tt * TV + (ch - 2 * DK)] = val;
        }
        // 1b. gates of the chunk.
        if (tid < (ax_u32)TC) {
            float eg = 1.0f, beta = 0.0f;
            if (tid < tc) {
                const ax_u64 t = (ax_u64)t0 + tid;
                const bool f32 = (flags & GDN_FLAG_AB_F32) != 0;
                const float av = gdn_gate<TX>(a, b * a_sb + t * a_st + h, f32);
                const float bv = gdn_gate<TX>(bb, b * b_sb + t * b_st + h, f32);
                eg = expf(na * gdn_softplus(av + dtb));
                beta = gdn_sigmoid(bv);
            }
            s_sc[tid] = eg;
            s_sc[TC + tid] = beta;
        }
        AX_SYNC();
        // 1c. q.q, k.k, q.k partial sums (NSEG interleaved lanes per token).
        {
            const ax_u32 tt = tid / NSEG, sg = tid % NSEG;
            float pq = 0.0f, pk = 0.0f, pqk = 0.0f;
            for (int i = sg; i < DK; i += NSEG) {
                const float qv = s_q[tt * DK + i], kv = s_k[tt * DK + i];
                pq = fmaf(qv, qv, pq);
                pk = fmaf(kv, kv, pk);
                pqk = fmaf(qv, kv, pqk);
            }
            s_red[tid] = pq;
            s_red[NT + tid] = pk;
            s_red[2 * NT + tid] = pqk;
        }
        AX_SYNC();
        if (tid < (ax_u32)TC) {
            float sq = 0.0f, sk = 0.0f, sqk = 0.0f;
            for (int i = 0; i < NSEG; ++i) {
                sq += s_red[tid * NSEG + i];
                sk += s_red[NT + tid * NSEG + i];
                sqk += s_red[2 * NT + tid * NSEG + i];
            }
            const float iq = l2 ? 1.0f / sqrtf(sq + eps) : 1.0f;
            const float ik = l2 ? 1.0f / sqrtf(sk + eps) : 1.0f;
            const float qs = iq * scale;
            s_sc[2 * TC + tid] = qs;
            s_sc[3 * TC + tid] = ik;
            s_sc[4 * TC + tid] = sqk * qs * ik;
        }
        AX_SYNC();
        // 2. sequential scan over the chunk.
        for (ax_u32 tt = 0; tt < tc; ++tt) {
            const float *qq = s_q + tt * DK + row0;
            const float *kk = s_k + tt * DK + row0;
            float pk0 = 0.0f, pk1 = 0.0f, pq0 = 0.0f, pq1 = 0.0f;
            AX_UNROLL
            for (int m = 0; m < RP; m += 4) {
                const gdn_f4 kv = *reinterpret_cast<const gdn_f4 *>(kk + m);
                const gdn_f4 qv = *reinterpret_cast<const gdn_f4 *>(qq + m);
                pk0 = fmaf(S[m], kv.x, pk0);
                pq0 = fmaf(S[m], qv.x, pq0);
                pk1 = fmaf(S[m + 1], kv.y, pk1);
                pq1 = fmaf(S[m + 1], qv.y, pq1);
                pk0 = fmaf(S[m + 2], kv.z, pk0);
                pq0 = fmaf(S[m + 2], qv.z, pq0);
                pk1 = fmaf(S[m + 3], kv.w, pk1);
                pq1 = fmaf(S[m + 3], qv.w, pq1);
            }
            float *red = s_red + par * 2 * NT;
            red[tid] = pk0 + pk1;
            red[NT + tid] = pq0 + pq1;
            AX_SYNC();
            float sk = 0.0f, sq = 0.0f;
            AX_UNROLL
            for (int g = 0; g < NG; ++g) {
                sk += red[g * TV + col];
                sq += red[NT + g * TV + col];
            }
            const float eg = s_sc[tt], beta = s_sc[TC + tt];
            const float qs = s_sc[2 * TC + tt], ks = s_sc[3 * TC + tt], kq = s_sc[4 * TC + tt];
            const float delta = (s_v[tt * TV + col] - eg * (sk * ks)) * beta;
            if (grp == 0 && col_ok) {
                const float o = fmaf(delta, kq, eg * (sq * qs));
                ax_store(out, ((b * seq + t0 + tt) * hv + h) * (ax_u64)dv + vcol, o);
            }
            const float dks = delta * ks;
            AX_UNROLL
            for (int m = 0; m < RP; m += 4) {
                const gdn_f4 kv = *reinterpret_cast<const gdn_f4 *>(kk + m);
                S[m] = fmaf(kv.x, dks, S[m] * eg);
                S[m + 1] = fmaf(kv.y, dks, S[m + 1] * eg);
                S[m + 2] = fmaf(kv.z, dks, S[m + 2] * eg);
                S[m + 3] = fmaf(kv.w, dks, S[m + 3] * eg);
            }
            par ^= 1u;
        }
        AX_SYNC();
    }

    if (col_ok) {
        AX_UNROLL
        for (int m = 0; m < RP; ++m) {
            const ax_u32 row = row0 + m;
            if (row < dk) s_out[s_base + (ax_u64)row * dv + vcol] = S[m];
        }
    }
}

// minb: workgroups per SM the CUDA register allocation targets (8 x 128 lanes on sm_8x).
#define GDN_SCAN_KERNEL(NAME, TX, DK, TV, NT, TC, MINB)                                            \
    AX_KERNEL void GDN_BOUNDS(NT, MINB) NAME(GDN_SCAN_PARAMS(TX)) {                                \
        gdn_scan_body<TX, DK, TV, NT, TC>(x, cs_in, cw, cb, a, bb, neg_a, dt_bias, s_in, out,      \
                                          cs_out, s_out, seq, hk, hv, dk, dv, kc, flags, eps,      \
                                          scale, x_sb, x_st, a_sb, a_st, b_sb, b_st);              \
    }

// ---------------------------------------------------------------------------
// Gated RMS norm: y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w * silu(z[r, :])
// Arguments: x, z, w (f32), y, rows, d, eps. Grid min(ceil(rows / 4), 2^20), 128 lanes,
// 128 floats of shared memory.
// ---------------------------------------------------------------------------

template <typename TX>
AX_DEV void gdn_gnorm_body(const TX *x, const TX *z, const float *w, TX *y, ax_u64 rows,
                           ax_u32 d, float eps) {
    float *red = ax_smem();
    const ax_u32 tid = AX_TID, lane = tid & 31u, sub = tid >> 5;
    for (ax_u64 r0 = (ax_u64)AX_BID_X * 4; r0 < rows; r0 += (ax_u64)AX_NBLK_X * 4) {
        const ax_u64 r = r0 + sub;
        const bool ok = r < rows;
        float ss = 0.0f;
        if (ok)
            for (ax_u32 j = lane; j < d; j += 32) {
                const float v = ax_load(x, r * d + j);
                ss = fmaf(v, v, ss);
            }
        red[tid] = ss;
        AX_SYNC();
        for (ax_u32 s = 16; s > 0; s >>= 1) {
            if (lane < s) red[tid] = red[tid] + red[tid + s];
            AX_SYNC();
        }
        const float inv = 1.0f / sqrtf(red[sub * 32] / (float)d + eps);
        AX_SYNC();
        if (ok)
            for (ax_u32 j = lane; j < d; j += 32) {
                const float v = ax_load(x, r * d + j) * inv * w[j];
                ax_store(y, r * d + j, v * gdn_silu(ax_load(z, r * d + j)));
            }
    }
}

#define GDN_GNORM_KERNEL(NAME, TX)                                                                 \
    AX_KERNEL void AX_BOUNDS(128) NAME(const TX *x, const TX *z, const float *w, TX *y,            \
                                       ax_u64 rows, ax_u32 d, float eps) {                         \
        gdn_gnorm_body<TX>(x, z, w, y, rows, d, eps);                                              \
    }
