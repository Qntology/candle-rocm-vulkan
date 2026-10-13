// Fused Gated DeltaNet kernels (Qwen3.5 / Qwen3-Next / Qwen4-Exp linear attention), see
// candle-core/src/delta_rule.rs for the semantics and candle-core/src/accel/delta.rs for the
// launch planning. Built by hipcc (kernels/delta_rule.hip), NVRTC (CUDA) and the host
// emulator (kernels/tests/delta_rule_host_test.cpp).
//
// gdn_scan: one workgroup of NW warps (32 lanes; 32-lane halves of a wave64) per (value
// column tile, value head, batch). The tile has TV = NW * CW value columns; warp w owns columns
// w * CW .. w * CW + CW - 1 and keeps their [DK, CW] slice of the recurrent state in registers
// for the whole sequence. With LC = CW / CPL lanes per row group and NG = 32 / LC row groups,
// lane l owns the CPL columns  w * CW + l % LC + c * LC  (c < CPL) and the RP = DK / NG rows
// j * 4 * NG + (l / LC) * 4 + e  (j < RP / 4, e < 4): the row groups of a warp read adjacent
// float4 of q / k in shared memory (no bank conflicts), every q / k value read is reused for
// CPL columns, and the reductions over the rows stay inside the warp (xor shuffles), so the
// token loop has no workgroup barrier.
// Tokens are processed in chunks of TC with two shared memory buffers:
//   1. conv phase: lane tid owns the conv channels tid + m * 32 * NW (q / k channels of the
//      workgroup's key head, then the v channels of its tile) and runs the causal depthwise
//      conv over [conv_state ; mixed_qkv] (+ bias) for the chunk with the taps in registers
//      (conv widths up to KMAX; wider kernels read the taps from global memory), then SiLU.
//      The last TC lanes compute the gates exp(g), g = neg_a * softplus(a + dt_bias) (torch
//      threshold 20), and beta = sigmoid(b).
//   2. barrier; norm phase: one warp per token reduces q.q, k.k and q.k (L2 norm with eps
//      inside the square root, q scaled by `scale` = 1 / sqrt(Dk)).
//   3. barrier; scan, per token with S the state before the token:
//          kv    = exp(g) * S^T k            (k, q L2 normalized, q scaled)
//          delta = beta * (v - kv)
//          o     = exp(g) * S^T q + delta * (k.q)      (= S'^T q)
//          S'    = exp(g) * S + k delta^T
//   The next chunk's conv phase writes the other buffer and its first barrier orders the
//   reuse of this one, so no barrier follows the scan.
// The new recurrent state and the new conv state are written out of place: every workgroup
// owns its state tile; the q / k conv channels of a key head are written by the first value
// head of the head group (tile 0), the v channels by the tile that reads them. All math is
// f32 whatever the input dtype. Rows >= dk and columns >= dv are masked, so a kernel built for
// DK serves any dk <= DK.
//
// gdn_gnorm: gated RMS norm y = rmsnorm(x) * w * silu(z), 32 lanes per row, 4 rows per
// workgroup of 128 lanes, f32 math.
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
// Grid (ceil(dv / TV), Hv, B), 32 * NW lanes, dynamic shared memory GDN_SCAN_SMEM_FLOATS * 4.
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

// v + v[lane ^ mask] over the 32 lanes of a warp (32-lane groups of a wave64). `scratch`
// (one float per workgroup lane) is only used by the host emulation, where every lane of
// the workgroup must call this uniformly.
#if defined(CANDLE_ACCEL_HOST)
AX_DEV float gdn_xor_add(float v, ax_u32 mask, float *scratch) {
    const ax_u32 t = AX_TID;
    scratch[t] = v;
    AX_SYNC();
    const float r = scratch[t ^ mask];
    AX_SYNC();
    return v + r;
}
#elif defined(__HIP__) || defined(__HIPCC__)
AX_DEV float gdn_xor_add(float v, ax_u32 mask, float *) {
    return v + __shfl_xor(v, (int)mask, 32);
}
#else
AX_DEV float gdn_xor_add(float v, ax_u32 mask, float *) {
    return v + __shfl_xor_sync(0xffffffffu, v, (int)mask, 32);
}
#endif

struct alignas(16) gdn_f4 {
    float x, y, z, w;
};

AX_DEV float gdn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
AX_DEV float gdn_silu(float x) { return x / (1.0f + expf(-x)); }
AX_DEV float gdn_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Shared memory of gdn_scan in floats (mirrored by accel::delta::ScanTile::smem_floats): two
// buffers of q [TC][DK], k [TC][DK], v [TC][TV] and per token scalars [8][TC], then one float
// per lane for the host emulation of the shuffles.
#define GDN_SCAN_BUF_FLOATS(DK, TV, TC) ((TC) * (2 * (DK) + (TV) + 8))
#define GDN_SCAN_SMEM_FLOATS(DK, TV, NT, TC) (2 * GDN_SCAN_BUF_FLOATS(DK, TV, TC) + (NT))

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

template <typename TX, int DK, int CW, int CPL, int NW, int TC, int KMAX>
AX_DEV void gdn_scan_body(GDN_SCAN_PARAMS(TX)) {
    constexpr int NT = 32 * NW;
    constexpr int TV = CW * NW;              // value columns per workgroup
    constexpr int LC = CW / CPL;             // lanes per row group (column stride of a lane)
    constexpr int NG = 32 / LC;              // row groups per warp
    constexpr int RP = DK / NG;              // state rows per lane
    constexpr int NCH = 2 * DK + TV;         // conv channels per workgroup
    constexpr int CPT = (NCH + NT - 1) / NT; // conv channels per lane
    constexpr int NIT = (TC + NW - 1) / NW;  // norm tokens per warp
    constexpr int BUF = GDN_SCAN_BUF_FLOATS(DK, TV, TC);
    static_assert(CW % CPL == 0 && 32 % LC == 0 && RP % 4 == 0 && DK % 32 == 0, "gdn_scan: bad tile");
    static_assert(KMAX >= 2 && TC <= NT, "gdn_scan: bad chunk");
    float *sm = ax_smem();
    float *scratch = sm + 2 * BUF;

    const ax_u32 tid = AX_TID;
    const ax_u32 lane = tid & 31u;
    const ax_u32 warp = tid >> 5;
    const ax_u32 rg = lane / LC;
    const ax_u32 tile = AX_BID_X;
    const ax_u32 h = AX_BID_Y;
    const ax_u64 b = AX_BID_Z;
    const ax_u32 hg = hv / hk;
    const bool tiled = (flags & GDN_FLAG_TILED) != 0;
    const ax_u32 kh = tiled ? h % hk : h / hg;
    const ax_u32 col0 = warp * CW + lane % LC; // first column of this lane inside the tile
    const ax_u32 vcol0 = tile * TV + col0;
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

    // Recurrent state slice.
    const ax_u64 s_base = ((b * hv + h) * (ax_u64)dk) * dv;
    float S[RP][CPL];
    AX_UNROLL
    for (int j = 0; j < RP / 4; ++j) {
        AX_UNROLL
        for (int e = 0; e < 4; ++e) {
            const ax_u32 row = j * 4 * NG + rg * 4 + e;
            AX_UNROLL
            for (int c = 0; c < CPL; ++c) {
                const ax_u32 vcol = vcol0 + c * LC;
                S[j * 4 + e][c] = (s_in && vcol < dv && row < dk)
                                      ? s_in[s_base + (ax_u64)row * dv + vcol]
                                      : 0.0f;
            }
        }
    }

    const bool reg_conv = kc <= (ax_u32)KMAX;
    const bool l2 = (flags & GDN_FLAG_L2NORM) != 0;
    const bool ab32 = (flags & GDN_FLAG_AB_F32) != 0;
    const float na = neg_a[h];
    const float dtb = dt_bias[h];
    ax_u32 p = 0;
    for (ax_u32 t0 = 0; t0 < seq; t0 += TC, p ^= 1u) {
        const ax_u32 tc = seq - t0 < (ax_u32)TC ? seq - t0 : (ax_u32)TC;
        float *s_q = sm + p * BUF;   // [TC][DK]
        float *s_k = s_q + TC * DK;  // [TC][DK]
        float *s_v = s_k + TC * DK;  // [TC][TV]
        float *s_sc = s_v + TC * TV; // [8][TC]: exp(g), beta, q scale, k scale, q.k

        // 1. conv + SiLU of the chunk, gates.
        AX_UNROLL
        for (int m = 0; m < CPT; ++m) {
            const ax_u32 ch = tid + m * NT;
            if (ch >= (ax_u32)NCH) continue;
            bool ok;
            ax_u64 c;
            float *dst;
            ax_u32 ld;
            if (ch < (ax_u32)DK) {
                ok = ch < dk;
                c = q_c0 + ch;
                dst = s_q + ch;
                ld = DK;
            } else if (ch < (ax_u32)(2 * DK)) {
                ok = ch - DK < dk;
                c = k_c0 + (ch - DK);
                dst = s_k + (ch - DK);
                ld = DK;
            } else {
                ok = ch - 2 * DK < v_n;
                c = v_c0 + (ch - 2 * DK);
                dst = s_v + (ch - 2 * DK);
                ld = TV;
            }
            if (!ok) {
                AX_UNROLL
                for (int tt = 0; tt < TC; ++tt) dst[tt * ld] = 0.0f;
                continue;
            }
            const float bias = cb ? cb[c] : 0.0f;
            if (reg_conv) {
                // Weights right aligned in KMAX slots (zero padded), inputs of the tokens
                // t0 - KMAX + 1 .. t0 + TC - 1 (the conv state before token 0).
                float w[KMAX];
                AX_UNROLL
                for (int j = 0; j < KMAX; ++j) {
                    const int jj = j - (KMAX - (int)kc);
                    w[j] = jj >= 0 ? cw[c * kc + jj] : 0.0f;
                }
                float xv[KMAX - 1 + TC];
                if (t0 >= (ax_u32)(KMAX - 1) && tc == (ax_u32)TC) {
                    const TX *xp = x + b * x_sb + (ax_u64)(t0 - (KMAX - 1)) * x_st + c;
                    AX_UNROLL
                    for (int i = 0; i < KMAX - 1 + TC; ++i) xv[i] = ax_load(xp, (ax_u64)i * x_st);
                } else {
                    const TX *xp = x + b * x_sb + c;
                    AX_UNROLL
                    for (int i = 0; i < KMAX - 1 + TC; ++i) {
                        const long long tok = (long long)t0 + i - (KMAX - 1);
                        float v = 0.0f;
                        if (tok >= 0) {
                            if (tok < (long long)seq) v = ax_load(xp, (ax_u64)tok * x_st);
                        } else if (cs_in && tok + (long long)km1 >= 0) {
                            v = ax_load(cs_in, (b * C + c) * km1 + (ax_u64)(tok + (long long)km1));
                        }
                        xv[i] = v;
                    }
                }
                AX_UNROLL
                for (int tt = 0; tt < TC; ++tt) {
                    float acc = bias;
                    AX_UNROLL
                    for (int j = 0; j < KMAX; ++j) acc = fmaf(w[j], xv[tt + j], acc);
                    dst[tt * ld] = (ax_u32)tt < tc ? gdn_silu(acc) : 0.0f;
                }
            } else {
                for (ax_u32 tt = 0; tt < (ax_u32)TC; ++tt) {
                    float acc = bias;
                    if (tt < tc)
                        for (ax_u32 j = 0; j < kc; ++j)
                            acc = fmaf(cw[c * kc + j],
                                       gdn_xe(x, cs_in, b, c, (ax_u64)t0 + tt + j, kc, C, x_sb, x_st), acc);
                    dst[tt * ld] = tt < tc ? gdn_silu(acc) : 0.0f;
                }
            }
        }
        if (tid >= (ax_u32)(NT - TC)) {
            const ax_u32 g = tid - (NT - TC);
            float eg = 1.0f, beta = 0.0f;
            if (g < tc) {
                const ax_u64 t = (ax_u64)t0 + g;
                const float av = gdn_gate<TX>(a, b * a_sb + t * a_st + h, ab32);
                const float bv = gdn_gate<TX>(bb, b * b_sb + t * b_st + h, ab32);
                eg = expf(na * gdn_softplus(av + dtb));
                beta = gdn_sigmoid(bv);
            }
            s_sc[g] = eg;
            s_sc[TC + g] = beta;
        }
        AX_SYNC();

        // 2. q.q, k.k, q.k: one warp per token.
        AX_UNROLL
        for (int it = 0; it < NIT; ++it) {
            const ax_u32 tt = warp + it * NW;
            const bool tok = tt < (ax_u32)TC;
            float pq = 0.0f, pk = 0.0f, pqk = 0.0f;
            if (tok) {
                AX_UNROLL
                for (int i = 0; i < DK / 32; ++i) {
                    const float qv = s_q[tt * DK + i * 32 + lane];
                    const float kv = s_k[tt * DK + i * 32 + lane];
                    pq = fmaf(qv, qv, pq);
                    pk = fmaf(kv, kv, pk);
                    pqk = fmaf(qv, kv, pqk);
                }
            }
            AX_UNROLL
            for (int msk = 16; msk >= 1; msk >>= 1) {
                pq = gdn_xor_add(pq, msk, scratch);
                pk = gdn_xor_add(pk, msk, scratch);
                pqk = gdn_xor_add(pqk, msk, scratch);
            }
            if (tok && lane == 0) {
                const float iq = l2 ? 1.0f / sqrtf(pq + eps) : 1.0f;
                const float ik = l2 ? 1.0f / sqrtf(pk + eps) : 1.0f;
                const float qs = iq * scale;
                s_sc[2 * TC + tt] = qs;
                s_sc[3 * TC + tt] = ik;
                s_sc[4 * TC + tt] = pqk * qs * ik;
            }
        }
        AX_SYNC();

        // 3. sequential scan over the chunk.
        for (ax_u32 tt = 0; tt < tc; ++tt) {
            const float *qq = s_q + tt * DK + rg * 4;
            const float *kk = s_k + tt * DK + rg * 4;
            float pk[2][CPL], pq[2][CPL];
            AX_UNROLL
            for (int c = 0; c < CPL; ++c) pk[0][c] = pk[1][c] = pq[0][c] = pq[1][c] = 0.0f;
            AX_UNROLL
            for (int j = 0; j < RP / 4; ++j) {
                const gdn_f4 kv = *reinterpret_cast<const gdn_f4 *>(kk + j * 4 * NG);
                const gdn_f4 qv = *reinterpret_cast<const gdn_f4 *>(qq + j * 4 * NG);
                AX_UNROLL
                for (int c = 0; c < CPL; ++c) {
                    pk[0][c] = fmaf(S[j * 4][c], kv.x, pk[0][c]);
                    pq[0][c] = fmaf(S[j * 4][c], qv.x, pq[0][c]);
                    pk[1][c] = fmaf(S[j * 4 + 1][c], kv.y, pk[1][c]);
                    pq[1][c] = fmaf(S[j * 4 + 1][c], qv.y, pq[1][c]);
                    pk[0][c] = fmaf(S[j * 4 + 2][c], kv.z, pk[0][c]);
                    pq[0][c] = fmaf(S[j * 4 + 2][c], qv.z, pq[0][c]);
                    pk[1][c] = fmaf(S[j * 4 + 3][c], kv.w, pk[1][c]);
                    pq[1][c] = fmaf(S[j * 4 + 3][c], qv.w, pq[1][c]);
                }
            }
            float sk[CPL], sq[CPL];
            AX_UNROLL
            for (int c = 0; c < CPL; ++c) {
                sk[c] = pk[0][c] + pk[1][c];
                sq[c] = pq[0][c] + pq[1][c];
            }
            AX_UNROLL
            for (int msk = LC; msk < 32; msk <<= 1) {
                AX_UNROLL
                for (int c = 0; c < CPL; ++c) {
                    sk[c] = gdn_xor_add(sk[c], msk, scratch);
                    sq[c] = gdn_xor_add(sq[c], msk, scratch);
                }
            }
            const float eg = s_sc[tt], beta = s_sc[TC + tt];
            const float qs = s_sc[2 * TC + tt], ks = s_sc[3 * TC + tt], kq = s_sc[4 * TC + tt];
            float dks[CPL];
            AX_UNROLL
            for (int c = 0; c < CPL; ++c) {
                const ax_u32 col = col0 + c * LC;
                const float delta = (s_v[tt * TV + col] - eg * (sk[c] * ks)) * beta;
                if (rg == 0 && tile * TV + col < dv) {
                    const float o = fmaf(delta, kq, eg * (sq[c] * qs));
                    ax_store(out, ((b * seq + t0 + tt) * hv + h) * (ax_u64)dv + tile * TV + col, o);
                }
                dks[c] = delta * ks;
            }
            AX_UNROLL
            for (int j = 0; j < RP / 4; ++j) {
                const gdn_f4 kv = *reinterpret_cast<const gdn_f4 *>(kk + j * 4 * NG);
                AX_UNROLL
                for (int c = 0; c < CPL; ++c) {
                    S[j * 4][c] = fmaf(kv.x, dks[c], S[j * 4][c] * eg);
                    S[j * 4 + 1][c] = fmaf(kv.y, dks[c], S[j * 4 + 1][c] * eg);
                    S[j * 4 + 2][c] = fmaf(kv.z, dks[c], S[j * 4 + 2][c] * eg);
                    S[j * 4 + 3][c] = fmaf(kv.w, dks[c], S[j * 4 + 3][c] * eg);
                }
            }
        }
    }

    AX_UNROLL
    for (int j = 0; j < RP / 4; ++j) {
        AX_UNROLL
        for (int e = 0; e < 4; ++e) {
            const ax_u32 row = j * 4 * NG + rg * 4 + e;
            AX_UNROLL
            for (int c = 0; c < CPL; ++c) {
                const ax_u32 vcol = vcol0 + c * LC;
                if (row < dk && vcol < dv) s_out[s_base + (ax_u64)row * dv + vcol] = S[j * 4 + e][c];
            }
        }
    }
}

// MINB: workgroups per multiprocessor the CUDA register allocation targets.
#define GDN_SCAN_KERNEL(NAME, TX, DK, CW, CPL, NW, TC, KMAX, MINB)                                 \
    AX_KERNEL void GDN_BOUNDS(32 * (NW), MINB) NAME(GDN_SCAN_PARAMS(TX)) {                         \
        gdn_scan_body<TX, DK, CW, CPL, NW, TC, KMAX>(x, cs_in, cw, cb, a, bb, neg_a, dt_bias,      \
                                                     s_in, out, cs_out, s_out, seq, hk, hv, dk,    \
                                                     dv, kc, flags, eps, scale, x_sb, x_st, a_sb,  \
                                                     a_st, b_sb, b_st);                            \
    }

// ---------------------------------------------------------------------------
// Gated RMS norm: y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w * silu(z[r, :])
// Arguments: x, z, w (f32), y, rows, d, eps. One warp per row (rows of up to 32 * GDN_GNORM_VPL
// elements stay in registers, longer rows are read twice), 4 rows per workgroup of 128 lanes,
// grid min(ceil(rows / 4), 2^20), 128 floats of shared memory (host emulation only).
// ---------------------------------------------------------------------------

#define GDN_GNORM_VPL 8

template <typename TX>
AX_DEV void gdn_gnorm_body(const TX *x, const TX *z, const float *w, TX *y, ax_u64 rows,
                           ax_u32 d, float eps) {
    float *scratch = ax_smem();
    const ax_u32 tid = AX_TID, lane = tid & 31u, sub = tid >> 5;
    for (ax_u64 r0 = (ax_u64)AX_BID_X * 4; r0 < rows; r0 += (ax_u64)AX_NBLK_X * 4) {
        const ax_u64 r = r0 + sub;
        const bool ok = r < rows;
        const TX *xr = x + r * d;
        float xv[GDN_GNORM_VPL];
        float ss = 0.0f;
        AX_UNROLL
        for (int i = 0; i < GDN_GNORM_VPL; ++i) {
            const ax_u32 j = lane + 32u * i;
            xv[i] = (ok && j < d) ? ax_load(xr, j) : 0.0f;
            ss = fmaf(xv[i], xv[i], ss);
        }
        if (ok)
            for (ax_u32 j = lane + 32u * GDN_GNORM_VPL; j < d; j += 32) {
                const float v = ax_load(xr, j);
                ss = fmaf(v, v, ss);
            }
        AX_UNROLL
        for (int msk = 16; msk >= 1; msk >>= 1) ss = gdn_xor_add(ss, msk, scratch);
        const float inv = 1.0f / sqrtf(ss / (float)d + eps);
        if (ok) {
            AX_UNROLL
            for (int i = 0; i < GDN_GNORM_VPL; ++i) {
                const ax_u32 j = lane + 32u * i;
                if (j < d) ax_store(y, r * d + j, xv[i] * inv * w[j] * gdn_silu(ax_load(z, r * d + j)));
            }
            for (ax_u32 j = lane + 32u * GDN_GNORM_VPL; j < d; j += 32)
                ax_store(y, r * d + j, ax_load(xr, j) * inv * w[j] * gdn_silu(ax_load(z, r * d + j)));
        }
    }
}

#define GDN_GNORM_KERNEL(NAME, TX)                                                                 \
    AX_KERNEL void AX_BOUNDS(128) NAME(const TX *x, const TX *z, const float *w, TX *y,            \
                                       ax_u64 rows, ax_u32 d, float eps) {                         \
        gdn_gnorm_body<TX>(x, z, w, y, rows, d, eps);                                              \
    }
