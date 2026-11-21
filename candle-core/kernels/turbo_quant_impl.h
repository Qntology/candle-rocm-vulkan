// TurboQuant and microscaling quantization kernels. Row layout and formats: see
// candle-core/src/accel/formats.rs; the CPU codec in candle-core/src/accel/turbo.rs is the
// reference of every kernel here.
//
// Rotations (TurboQuant Π, FA3 Hadamard) and QJL projections are applied by the host with
// regular matmuls, so the kernels only see rotated rows:
//   tq_quant   rows -> packed rows (codes, block scales, rsc0); prod rows get rsc1 / signs later
//   tq_resid   (rows, packed rows) -> residual of the unit vector to its codebook value (prod)
//   tq_signs   (packed rows, S·residual, residual) -> packed rows with QJL sign bits and rsc1
//   tq_dequant packed rows -> rsc0 * element (rotated domain); prod also writes the QJL
//              coefficients ±rsc0·rsc1·sqrt(pi/2)/D to multiply with S
//   tq_attn    attention of rotated queries over packed K/V rows (flash-decoding layout);
//              tq_attn_fp8 is the same kernel specialized for FP8 K and V rows, tq_attnp /
//              tq_attnp_fp8 read the rows from a paged block table instead of one tensor
//   tq_combine merges key splits of tq_attn
#pragma once
#include "accel_common.h"

#define TQ_NT 128
#define TQ_MSE 0u
#define TQ_PROD 1u
#define TQ_MXFP4 2u
#define TQ_NVFP4 3u
#define TQ_MXFP8 4u
#define TQ_FP8 5u
#define TQ_QJL 1.2533141f

AX_DEV ax_u32 tq_bits(const ax_u32 *w, ax_u32 pos, ax_u32 nb) {
    const ax_u32 wi = pos >> 5, sh = pos & 31u;
    ax_u32 v = w[wi] >> sh;
    if (sh + nb > 32u) v |= w[wi + 1] << (32u - sh);
    return v & ((1u << nb) - 1u);
}

AX_DEV ax_u32 tq_byte(const ax_u32 *w, ax_u32 kb) { return (w[kb >> 2] >> ((kb & 3u) * 8u)) & 0xffu; }

AX_DEV float tq_logit(float s, float softcap, float slope, long long dist) {
    if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
    if (slope != 0.0f) s -= slope * (float)(dist < 0 ? -dist : dist);
    return s;
}

// Element j of a packed row without the rsc0 factor.
AX_DEV float tq_elem(const ax_u32 *row, ax_u32 j, ax_u32 mode, ax_u32 cb, ax_u32 code_words,
                     const float *cent) {
    const ax_u32 c = tq_bits(row, j * cb, cb);
    const ax_u32 *aux = row + code_words;
    if (mode <= TQ_PROD) return cent[c];
    if (mode == TQ_MXFP4) return ax_fp4_to_f(c) * ax_e8m0_to_f(tq_byte(aux, j >> 5));
    if (mode == TQ_NVFP4) return ax_fp4_to_f(c) * ax_e4m3_to_f(tq_byte(aux, j >> 4));
    if (mode == TQ_MXFP8) return ax_e4m3_to_f(c) * ax_e8m0_to_f(tq_byte(aux, j >> 5));
    return ax_e4m3_to_f(c);
}

// Packs codes[e] (cb bits each, D of them) into the word w of the code region.
AX_DEV ax_u32 tq_pack_word(const ax_u32 *codes, ax_u32 w, ax_u32 cb, ax_u32 D) {
    const ax_u32 bit0 = w * 32u;
    const ax_u32 e0 = bit0 / cb;
    ax_u32 e1 = (bit0 + 31u) / cb + 1u;
    if (e1 > D) e1 = D;
    const ax_u32 mask = (1u << cb) - 1u;
    ax_u32 word = 0;
    for (ax_u32 e = e0; e < e1; ++e) {
        const int sh = (int)(e * cb) - (int)bit0;
        const ax_u32 c = codes[e] & mask;
        word |= sh >= 0 ? (c << sh) : (c >> (-sh));
    }
    return word;
}

// One workgroup per row. Shared memory: y[D], codes[D], red[TQ_NT], scales[D / 16].
template <typename T>
AX_DEV void tq_quant_impl(const T *src, ax_u32 *out, const float *tab, ax_u64 rows, ax_u32 D,
                          ax_u64 src_stride, ax_u32 mode, ax_u32 cb, ax_u32 code_words,
                          ax_u32 aux_words, ax_u32 row_words) {
    float *y = ax_smem();
    ax_u32 *codes = reinterpret_cast<ax_u32 *>(y + D);
    float *red = y + 2 * D;
    ax_u32 *sc = reinterpret_cast<ax_u32 *>(red + TQ_NT);
    const ax_u32 t = AX_TID;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        AX_SYNC();
        const T *x = src + row * src_stride;
        float ss = 0.0f, am = 0.0f;
        for (ax_u32 j = t; j < D; j += TQ_NT) {
            const float v = ax_load(x, j);
            y[j] = v;
            ss += v * v;
            am = fmaxf(am, fabsf(v));
        }
        // the norm is only used by the codebook modes, the row maximum by NVFP4 / FP8 (the mode
        // is uniform over the workgroup). The MX modes read y[] across lanes below, so they
        // still need the barrier that the skipped reductions would have given.
        if (mode <= TQ_PROD) ss = ax_block_sum<TQ_NT>(ss, red);
        else if (mode == TQ_NVFP4 || mode == TQ_FP8) am = ax_block_max<TQ_NT>(am, red);
        else AX_SYNC();
        float rsc0 = 1.0f;
        if (mode <= TQ_PROD) {
            const float norm = sqrtf(ss);
            const float inv = norm > 0.0f ? 1.0f / norm : 0.0f;
            const ax_u32 nb = (1u << cb) - 1u;
            rsc0 = norm;
            for (ax_u32 j = t; j < D; j += TQ_NT) {
                const float u = y[j] * inv;
                ax_u32 idx = 0;
                for (ax_u32 bi = 0; bi < nb; ++bi) idx += u >= tab[16 + bi] ? 1u : 0u;
                codes[j] = idx;
            }
        } else if (mode == TQ_MXFP4 || mode == TQ_MXFP8) {
            const int emax = mode == TQ_MXFP4 ? 2 : 8;
            for (ax_u32 bk = t; bk < (D >> 5); bk += TQ_NT) {
                float bm = 0.0f;
                for (ax_u32 e = 0; e < 32u; ++e) bm = fmaxf(bm, fabsf(y[bk * 32u + e]));
                sc[bk] = ax_mx_scale_code(bm, emax);
            }
            AX_SYNC();
            for (ax_u32 j = t; j < D; j += TQ_NT) {
                // a zero scale code means an all-zero (or subnormal) block: zero codes, no
                // division by the subnormal 2^-127 that some GPUs flush to zero
                const ax_u32 code = sc[j >> 5];
                const float s = ax_e8m0_to_f(code);
                codes[j] = code == 0 ? 0u : (mode == TQ_MXFP4 ? ax_f_to_fp4(y[j] / s) : ax_f_to_e4m3(y[j] / s));
            }
        } else if (mode == TQ_NVFP4) {
            const float g = am > 0.0f ? am / 2688.0f : 1.0f;
            rsc0 = g;
            for (ax_u32 bk = t; bk < (D >> 4); bk += TQ_NT) {
                float bm = 0.0f;
                for (ax_u32 e = 0; e < 16u; ++e) bm = fmaxf(bm, fabsf(y[bk * 16u + e]));
                sc[bk] = ax_f_to_e4m3(bm / 6.0f / g);
            }
            AX_SYNC();
            for (ax_u32 j = t; j < D; j += TQ_NT) {
                const float eff = ax_e4m3_to_f(sc[j >> 4]) * g;
                codes[j] = eff > 0.0f ? ax_f_to_fp4(y[j] / eff) : 0u;
            }
        } else {
            const float s = am > 0.0f ? am / 448.0f : 1.0f;
            rsc0 = s;
            for (ax_u32 j = t; j < D; j += TQ_NT) codes[j] = ax_f_to_e4m3(y[j] / s);
        }
        AX_SYNC();
        ax_u32 *orow = out + row * row_words;
        for (ax_u32 w = t; w < code_words; w += TQ_NT) orow[w] = tq_pack_word(codes, w, cb, D);
        const ax_u32 nblk = mode == TQ_NVFP4 ? (D >> 4) : (D >> 5);
        const bool scales = mode == TQ_MXFP4 || mode == TQ_MXFP8 || mode == TQ_NVFP4;
        for (ax_u32 w = t; w < aux_words; w += TQ_NT) {
            ax_u32 word = 0;
            if (scales)
                for (ax_u32 e = 0; e < 4u; ++e)
                    if (w * 4u + e < nblk) word |= (sc[w * 4u + e] & 0xffu) << (8u * e);
            orow[code_words + w] = word;
        }
        if (t == 0) {
            orow[code_words + aux_words] = ax_f2u(rsc0);
            orow[code_words + aux_words + 1] = 0u;
        }
    }
}

// Residual of the unit vector to its codebook value: resid = x / rsc0 - centroid.
template <typename T>
AX_DEV void tq_resid_impl(const T *src, const ax_u32 *data, const float *tab, float *resid,
                          ax_u64 rows, ax_u32 D, ax_u64 src_stride, ax_u32 cb, ax_u32 row_words,
                          ax_u32 rsc_off) {
    const ax_u64 n = rows * D;
    for (ax_u64 idx = (ax_u64)AX_BID_X * TQ_NT + AX_TID; idx < n; idx += (ax_u64)AX_NBLK_X * TQ_NT) {
        const ax_u64 row = idx / D;
        const ax_u32 j = (ax_u32)(idx % D);
        const ax_u32 *r = data + row * row_words;
        const float norm = ax_u2f(r[rsc_off]);
        const float inv = norm > 0.0f ? 1.0f / norm : 0.0f;
        const float u = ax_load(src + row * src_stride, j) * inv;
        resid[idx] = u - tab[tq_bits(r, j * cb, cb)];
    }
}

// Copies packed rows and fills in the QJL sign bits (sign of proj = S·resid) and rsc1 = |resid|.
AX_DEV void tq_signs_impl(const ax_u32 *in, const float *proj, const float *resid, ax_u32 *out,
                          ax_u64 rows, ax_u32 D, ax_u32 code_words, ax_u32 row_words) {
    float *red = ax_smem();
    const ax_u32 t = AX_TID;
    const ax_u32 sign_words = (D + 31u) / 32u;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        float ss = 0.0f;
        for (ax_u32 j = t; j < D; j += TQ_NT) {
            const float v = resid[row * D + j];
            ss += v * v;
        }
        ss = ax_block_sum<TQ_NT>(ss, red);
        const ax_u32 *ir = in + row * row_words;
        ax_u32 *orow = out + row * row_words;
        for (ax_u32 w = t; w < row_words; w += TQ_NT) {
            ax_u32 word = ir[w];
            if (w >= code_words && w < code_words + sign_words) {
                word = 0;
                const ax_u32 base = (w - code_words) * 32u;
                for (ax_u32 e = 0; e < 32u && base + e < D; ++e)
                    if (proj[row * D + base + e] >= 0.0f) word |= 1u << e;
            } else if (w == code_words + sign_words + 1u) {
                word = ax_f2u(sqrtf(ss));
            }
            orow[w] = word;
        }
    }
}

// rsc0 * element for every element; prod rows also give the QJL coefficients in zs.
AX_DEV void tq_dequant_impl(const ax_u32 *data, const float *tab, float *out, float *zs,
                            ax_u64 rows, ax_u32 D, ax_u32 mode, ax_u32 cb, ax_u32 code_words,
                            ax_u32 row_words, ax_u32 rsc_off) {
    const ax_u64 n = rows * D;
    for (ax_u64 idx = (ax_u64)AX_BID_X * TQ_NT + AX_TID; idx < n; idx += (ax_u64)AX_NBLK_X * TQ_NT) {
        const ax_u64 row = idx / D;
        const ax_u32 j = (ax_u32)(idx % D);
        const ax_u32 *r = data + row * row_words;
        const float r0 = ax_u2f(r[rsc_off]);
        out[idx] = r0 * tq_elem(r, j, mode, cb, code_words, tab);
        if (zs) {
            const float c = r0 * ax_u2f(r[rsc_off + 1]) * TQ_QJL / (float)D;
            zs[idx] = ((r[code_words + (j >> 5)] >> (j & 31u)) & 1u) ? c : -c;
        }
    }
}

// Element j of an FP8 row (bit-identical to tq_elem for TQ_FP8 rows).
AX_DEV float tq_fp8_elem(const ax_u32 *row, ax_u32 j) { return ax_e4m3_to_f_fast(tq_byte(row, j)); }

// Element j of an NVFP4 row (bit-identical to tq_elem for TQ_NVFP4 rows).
AX_DEV float tq_nvfp4_elem(const ax_u32 *row, ax_u32 j, ax_u32 cw) {
    const ax_u32 c = (row[j >> 3] >> ((j & 7u) * 4u)) & 0xfu;
    return ax_fp4_to_f(c) * ax_e4m3_to_f_fast(tq_byte(row + cw, j >> 4));
}

#define TQ_FM_GENERIC 0
#define TQ_FM_FP8     1
#define TQ_FM_NVFP4   2

// Single-pass TurboProd quantization: normalizes, quantizes to codebook, computes residual in
// shared memory, projects with S^T (coalesced across lanes), packs code words + QJL sign bits +
// rsc0/rsc1 in one kernel launch. Codes, residual norm, and rsc0/rsc1 match the 4-stage pipeline;
// QJL projection sums sequentially over j = 0..D matching the CPU reference `RowCodec::quantize_row`.
// Shared memory: (2 * D + TQ_NT) * sizeof(float).
template <typename T>
AX_DEV void tq_quant_prod_fused_impl(const T *src, const float *tab, const float *s_t,
                                     ax_u32 *out, ax_u64 rows, ax_u32 D, ax_u64 src_stride,
                                     ax_u32 cb, ax_u32 code_words, ax_u32 row_words) {
    float *y = ax_smem();
    ax_u32 *codes = reinterpret_cast<ax_u32 *>(y + D);
    float *red = y + 2 * D;
    const ax_u32 t = AX_TID;
    const ax_u32 nb = (1u << cb) - 1u;
    const ax_u32 sign_words = (D + 31u) / 32u;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        AX_SYNC();
        const T *x = src + row * src_stride;
        float ss = 0.0f;
        for (ax_u32 j = t; j < D; j += TQ_NT) {
            const float v = ax_load(x, j);
            y[j] = v;
            ss += v * v;
        }
        ss = ax_block_sum<TQ_NT>(ss, red);
        const float norm = sqrtf(ss);
        const float inv = norm > 0.0f ? 1.0f / norm : 0.0f;
        float ss_r = 0.0f;
        for (ax_u32 j = t; j < D; j += TQ_NT) {
            const float u = y[j] * inv;
            ax_u32 idx = 0;
            for (ax_u32 bi = 0; bi < nb; ++bi) idx += u >= tab[16 + bi] ? 1u : 0u;
            codes[j] = idx;
            const float r = u - tab[idx];
            y[j] = r;
            ss_r += r * r;
        }
        ss_r = ax_block_sum<TQ_NT>(ss_r, red); // also publishes y[0..D] and codes[0..D]
        ax_u32 *orow = out + row * row_words;
        for (ax_u32 w = t; w < code_words; w += TQ_NT) orow[w] = tq_pack_word(codes, w, cb, D);
        AX_SYNC(); // codes[] can now be reused for sign bits
        for (ax_u32 i = t; i < D; i += TQ_NT) {
            float dot = 0.0f;
            for (ax_u32 j = 0; j < D; ++j) dot += y[j] * s_t[(ax_u64)j * (ax_u64)D + i];
            codes[i] = dot >= 0.0f ? 1u : 0u;
        }
        AX_SYNC();
        for (ax_u32 w = t; w < sign_words; w += TQ_NT) {
            ax_u32 word = 0;
            const ax_u32 base = w * 32u;
            for (ax_u32 e = 0; e < 32u && base + e < D; ++e) {
                if (codes[base + e]) word |= 1u << e;
            }
            orow[code_words + w] = word;
        }
        if (t == 0) {
            orow[code_words + sign_words] = ax_f2u(norm);
            orow[code_words + sign_words + 1u] = ax_f2u(sqrtf(ss_r));
        }
    }
}

// On-chip Fast Walsh-Hadamard Transform kernel over rows of power-of-two length D.
// Shared memory: D * sizeof(float).
template <typename T>
AX_DEV void tq_fwht_impl(const T *src, float *dst, const ax_u32 *sign_bits, ax_u64 rows, ax_u32 D,
                         ax_u64 src_stride, float inv_sqrt_d, ax_u32 forward) {
    float *buf = ax_smem();
    const ax_u32 t = AX_TID;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        AX_SYNC();
        const T *x = src + row * src_stride;
        for (ax_u32 j = t; j < D; j += TQ_NT) buf[j] = ax_load(x, j);
        ax_fwht_inplace<TQ_NT>(buf, D, sign_bits, inv_sqrt_d, forward != 0u);
        float *y = dst + row * (ax_u64)D;
        for (ax_u32 j = t; j < D; j += TQ_NT) y[j] = buf[j];
    }
}

// Dot product of a packed key row with the rotated query, without the rsc0 factor.
// The FP8 and NVFP4 specializations read whole words and reuse block scales while preserving
// the exact product and summation order of the generic path (bit-identical).
template <int KM>
AX_DEV float tq_key_dot(const ax_u32 *kr, const float *qr, ax_u32 D, ax_u32 k_mode, ax_u32 k_cb,
                        ax_u32 k_cw, const float *kc) {
    float dot = 0.0f;
    if (KM == TQ_FM_FP8) {
        ax_u32 d = 0;
        for (; d + 4u <= D; d += 4u) {
            const ax_u32 x = kr[d >> 2];
            dot += ax_e4m3_to_f_fast(x & 0xffu) * qr[d];
            dot += ax_e4m3_to_f_fast((x >> 8) & 0xffu) * qr[d + 1];
            dot += ax_e4m3_to_f_fast((x >> 16) & 0xffu) * qr[d + 2];
            dot += ax_e4m3_to_f_fast(x >> 24) * qr[d + 3];
        }
        for (; d < D; ++d) dot += tq_fp8_elem(kr, d) * qr[d];
    } else if (KM == TQ_FM_NVFP4) {
        const ax_u32 *aux = kr + k_cw;
        ax_u32 d = 0;
        for (; d + 16u <= D; d += 16u) {
            const ax_u32 w0 = kr[d >> 3];
            const ax_u32 w1 = kr[(d >> 3) + 1u];
            const float bsc = ax_e4m3_to_f_fast(tq_byte(aux, d >> 4));
            AX_UNROLL
            for (ax_u32 i = 0; i < 8u; ++i) {
                dot += (ax_fp4_to_f((w0 >> (4u * i)) & 0xfu) * bsc) * qr[d + i];
            }
            AX_UNROLL
            for (ax_u32 i = 0; i < 8u; ++i) {
                dot += (ax_fp4_to_f((w1 >> (4u * i)) & 0xfu) * bsc) * qr[d + 8u + i];
            }
        }
        for (; d < D; ++d) dot += tq_nvfp4_elem(kr, d, k_cw) * qr[d];
    } else {
        for (ax_u32 d = 0; d < D; ++d) dot += tq_elem(kr, d, k_mode, k_cb, k_cw, kc) * qr[d];
    }
    return dot;
}

// Formats of the K / V rows of tq_attn.
struct tq_sides {
    ax_u32 k_mode, k_cb, k_cw, k_rsc, v_mode, v_cb, v_cw, v_rsc;
};

template <int VM>
AX_DEV float tq_val_elem(const ax_u32 *vr, ax_u32 j, const tq_sides &sd, const float *vc) {
    if (VM == TQ_FM_FP8) return tq_fp8_elem(vr, j);
    if (VM == TQ_FM_NVFP4) return tq_nvfp4_elem(vr, j, sd.v_cw);
    return tq_elem(vr, j, sd.v_mode, sd.v_cb, sd.v_cw, vc);
}

// One window of up to TQ_NT keys [kb, min(kb + TQ_NT, e)): key j sits at row j - j0 of kp / vp.
template <int KM, int VM>
AX_DEV void tq_attn_window(const float *qr, const float *qs, float *red, float *pr,
                           const float *kc, const float *vc, const ax_u32 *kp, const ax_u32 *vp,
                           ax_u64 k_ss, ax_u64 v_ss, long long j0, long long kb, long long e,
                           ax_u32 D, const tq_sides &sd, float scale, float softcap, float slope,
                           long long pos, float &m, float &l, float &acc0, float &acc1) {
    const ax_u32 t = AX_TID;
    const bool prod = (KM == TQ_FM_GENERIC) && (sd.k_mode == TQ_PROD);
    const long long j = kb + t;
    float s = -AX_INF, vscale = 0.0f;
    if (j < e) {
        const ax_u32 *kr = kp + (ax_u64)(j - j0) * k_ss;
        const float dot = tq_key_dot<KM>(kr, qr, D, sd.k_mode, sd.k_cb, sd.k_cw, kc);
        float sc = ax_u2f(kr[sd.k_rsc]) * dot;
        if (prod) {
            float sdot = 0.0f;
            for (ax_u32 d = 0; d < D; ++d)
                sdot += ((kr[sd.k_cw + (d >> 5)] >> (d & 31u)) & 1u) ? qs[d] : -qs[d];
            sc += ax_u2f(kr[sd.k_rsc]) * ax_u2f(kr[sd.k_rsc + 1]) * TQ_QJL / (float)D * sdot;
        }
        s = tq_logit(sc * scale, softcap, slope, pos - j);
        vscale = ax_u2f((vp + (ax_u64)(j - j0) * v_ss)[sd.v_rsc]);
    }
    const float m_new = fmaxf(m, ax_block_max<TQ_NT>(s, red));
    const float p = s == -AX_INF ? 0.0f : expf(s - m_new);
    pr[t] = p * vscale;
    const float lt = ax_block_sum<TQ_NT>(p, red);
    const float alpha = m == -AX_INF ? 0.0f : expf(m - m_new);
    l = l * alpha + lt;
    acc0 *= alpha;
    acc1 *= alpha;
    const int cnt = (int)ax_min_ll(TQ_NT, e - kb);
    for (int u = 0; u < cnt; ++u) {
        const float pu = pr[u];
        if (pu != 0.0f) {
            const ax_u32 *vr = vp + (ax_u64)(kb + u - j0) * v_ss;
            if (t < D) acc0 += pu * tq_val_elem<VM>(vr, t, sd, vc);
            if (t + TQ_NT < D) acc1 += pu * tq_val_elem<VM>(vr, t + TQ_NT, sd, vc);
        }
    }
    m = m_new;
    AX_SYNC();
}

// Attention of rotated queries over packed K/V rows.
template <int KM, int VM, bool PAGED>
AX_DEV void tq_attn_core(const float *qx, const ax_u32 *kd, const ax_u32 *vd, const float *tab,
                         const float *alibi, float *out, float *part_o, float *part_lse, ax_u32 Lq,
                         ax_u32 Lk, ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u64 q_sb, ax_u64 q_sh,
                         ax_u64 q_ss, ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_ss, ax_u64 v_sb,
                         ax_u64 v_sh, ax_u64 v_ss, const tq_sides &sd, float scale, float softcap,
                         int wl, int wr, ax_u32 n_splits, ax_u32 alibi_bs, ax_u64 rows_total,
                         const ax_pages *pg) {
    float *sm = ax_smem();
    float *qr = sm;            // D
    float *qs = qr + D;        // D
    float *red = qs + D;       // TQ_NT
    float *pr = red + TQ_NT;   // TQ_NT
    float *kc = pr + TQ_NT;    // 16
    float *vc = kc + 16;       // 16
    const ax_u32 t = AX_TID;
    const ax_u32 split = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    const ax_u32 group = Hq / Hkv;
    const bool prod = (KM == TQ_FM_GENERIC) && (sd.k_mode == TQ_PROD);
    for (ax_u64 rowi = AX_BID_X; rowi < (ax_u64)Lq * Hq; rowi += AX_NBLK_X) {
        const ax_u32 i = (ax_u32)(rowi / Hq);
        const ax_u32 h = (ax_u32)(rowi % Hq);
        const ax_u32 hk = h / group;
        const float *qp = qx + b * q_sb + h * q_sh + i * q_ss;
        AX_SYNC();
        for (ax_u32 d = t; d < D; d += TQ_NT) {
            qr[d] = qp[d];
            qs[d] = prod ? qp[D + d] : 0.0f;
        }
        if (t < 16u) {
            kc[t] = tab[t];
            vc[t] = tab[32u + t];
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
        const float slope = alibi ? alibi[(ax_u64)b * alibi_bs + h] : 0.0f;
        float m = -AX_INF, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
        if (!PAGED) {
            const ax_u32 *kbase = kd + (ax_u64)b * k_sb + (ax_u64)hk * k_sh;
            const ax_u32 *vbase = vd + (ax_u64)b * v_sb + (ax_u64)hk * v_sh;
            for (long long kb = s_lo; kb < s_hi; kb += TQ_NT)
                tq_attn_window<KM, VM>(qr, qs, red, pr, kc, vc, kbase, vbase, k_ss, v_ss, 0, kb,
                                       s_hi, D, sd, scale, softcap, slope, pos, m, l, acc0, acc1);
        } else {
            // windows never straddle two blocks; the block loop is uniform over the workgroup
            long long j0 = 0;
            for (ax_u32 bi = 0; bi < pg->n && j0 < s_hi; ++bi) {
                const long long j1 = j0 + (long long)pg->rows[bi];
                const long long a = ax_max_ll(s_lo, j0), e = ax_min_ll(s_hi, j1);
                if (a < e) {
                    const ax_u64 bh = (ax_u64)b * Hkv + hk;
                    const ax_u32 *kbase = reinterpret_cast<const ax_u32 *>(pg->k[bi]) +
                                          bh * ((ax_u64)pg->k_cap[bi] * k_ss);
                    const ax_u32 *vbase = reinterpret_cast<const ax_u32 *>(pg->v[bi]) +
                                          bh * ((ax_u64)pg->v_cap[bi] * v_ss);
                    for (long long kb = a; kb < e; kb += TQ_NT)
                        tq_attn_window<KM, VM>(qr, qs, red, pr, kc, vc, kbase, vbase, k_ss, v_ss,
                                               j0, kb, e, D, sd, scale, softcap, slope, pos, m, l,
                                               acc0, acc1);
                }
                j0 = j1;
            }
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        const ax_u64 orow = ((ax_u64)b * Hq + h) * Lq + i;
        if (n_splits == 1) {
            if (t < D) out[orow * D + t] = acc0 * inv;
            if (t + TQ_NT < D) out[orow * D + t + TQ_NT] = acc1 * inv;
        } else {
            const ax_u64 prow = (ax_u64)split * rows_total + orow;
            if (t < D) part_o[prow * D + t] = acc0 * inv;
            if (t + TQ_NT < D) part_o[prow * D + t + TQ_NT] = acc1 * inv;
            if (t == 0) part_lse[prow] = l > 0.0f ? m + logf(l) : -AX_INF;
        }
    }
}

// GQA-packed tq_attn: one workgroup per (query row, pack of G query heads sharing a kv head,
// split), so every packed K/V row is loaded and decoded once for G heads.
template <int KM, int VM, int G>
AX_DEV void tq_attng_window(const float *qr, const float *qs, float *red, float *pr,
                            const float *kc, const float *vc, const ax_u32 *kp, const ax_u32 *vp,
                            ax_u64 k_ss, ax_u64 v_ss, long long j0, long long kb, long long e,
                            ax_u32 D, const tq_sides &sd, float scale, float softcap,
                            const float *slope, long long pos, float *m, float *l, float *acc0,
                            float *acc1) {
    const ax_u32 t = AX_TID;
    const bool prod = (KM == TQ_FM_GENERIC) && (sd.k_mode == TQ_PROD);
    const long long j = kb + t;
    float s[G];
    AX_UNROLL
    for (int g = 0; g < G; ++g) s[g] = -AX_INF;
    float vscale = 0.0f;
    if (j < e) {
        const ax_u32 *kr = kp + (ax_u64)(j - j0) * k_ss;
        float dot[G];
        AX_UNROLL
        for (int g = 0; g < G; ++g) dot[g] = 0.0f;
        if (KM == TQ_FM_FP8) {
            ax_u32 d = 0;
            for (; d + 4u <= D; d += 4u) {
                const ax_u32 x = kr[d >> 2];
                const float c0 = ax_e4m3_to_f_fast(x & 0xffu);
                const float c1 = ax_e4m3_to_f_fast((x >> 8) & 0xffu);
                const float c2 = ax_e4m3_to_f_fast((x >> 16) & 0xffu);
                const float c3 = ax_e4m3_to_f_fast(x >> 24);
                AX_UNROLL
                for (int g = 0; g < G; ++g) {
                    const float *q = qr + g * D + d;
                    dot[g] += c0 * q[0];
                    dot[g] += c1 * q[1];
                    dot[g] += c2 * q[2];
                    dot[g] += c3 * q[3];
                }
            }
            for (; d < D; ++d) {
                const float c = tq_fp8_elem(kr, d);
                AX_UNROLL
                for (int g = 0; g < G; ++g) dot[g] += c * qr[g * D + d];
            }
        } else if (KM == TQ_FM_NVFP4) {
            const ax_u32 *aux = kr + sd.k_cw;
            ax_u32 d = 0;
            for (; d + 16u <= D; d += 16u) {
                const ax_u32 w0 = kr[d >> 3];
                const ax_u32 w1 = kr[(d >> 3) + 1u];
                const float bsc = ax_e4m3_to_f_fast(tq_byte(aux, d >> 4));
                AX_UNROLL
                for (ax_u32 i = 0; i < 8u; ++i) {
                    const float c = ax_fp4_to_f((w0 >> (4u * i)) & 0xfu) * bsc;
                    AX_UNROLL
                    for (int g = 0; g < G; ++g) dot[g] += c * qr[g * D + d + i];
                }
                AX_UNROLL
                for (ax_u32 i = 0; i < 8u; ++i) {
                    const float c = ax_fp4_to_f((w1 >> (4u * i)) & 0xfu) * bsc;
                    AX_UNROLL
                    for (int g = 0; g < G; ++g) dot[g] += c * qr[g * D + d + 8u + i];
                }
            }
            for (; d < D; ++d) {
                const float c = tq_nvfp4_elem(kr, d, sd.k_cw);
                AX_UNROLL
                for (int g = 0; g < G; ++g) dot[g] += c * qr[g * D + d];
            }
        } else {
            for (ax_u32 d = 0; d < D; ++d) {
                const float c = tq_elem(kr, d, sd.k_mode, sd.k_cb, sd.k_cw, kc);
                AX_UNROLL
                for (int g = 0; g < G; ++g) dot[g] += c * qr[g * D + d];
            }
        }
        const float rsc = ax_u2f(kr[sd.k_rsc]);
        float sdot[G];
        AX_UNROLL
        for (int g = 0; g < G; ++g) sdot[g] = 0.0f;
        if (prod) {
            for (ax_u32 d = 0; d < D; ++d) {
                const bool bit = ((kr[sd.k_cw + (d >> 5)] >> (d & 31u)) & 1u) != 0;
                AX_UNROLL
                for (int g = 0; g < G; ++g) sdot[g] += bit ? qs[g * D + d] : -qs[g * D + d];
            }
        }
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            float sc = rsc * dot[g];
            if (prod)
                sc += ax_u2f(kr[sd.k_rsc]) * ax_u2f(kr[sd.k_rsc + 1]) * TQ_QJL / (float)D * sdot[g];
            s[g] = tq_logit(sc * scale, softcap, slope[g], pos - j);
        }
        vscale = ax_u2f((vp + (ax_u64)(j - j0) * v_ss)[sd.v_rsc]);
    }
    float mx[G], p[G], lt[G];
    ax_block_reduce_g<TQ_NT, G, true>(s, red, mx);
    AX_UNROLL
    for (int g = 0; g < G; ++g) {
        mx[g] = fmaxf(m[g], mx[g]);
        p[g] = s[g] == -AX_INF ? 0.0f : expf(s[g] - mx[g]);
        pr[g * TQ_NT + t] = p[g] * vscale;
    }
    ax_block_reduce_g<TQ_NT, G, false>(p, red, lt); // also publishes pr
    AX_UNROLL
    for (int g = 0; g < G; ++g) {
        const float alpha = m[g] == -AX_INF ? 0.0f : expf(m[g] - mx[g]);
        l[g] = l[g] * alpha + lt[g];
        acc0[g] *= alpha;
        acc1[g] *= alpha;
        m[g] = mx[g];
    }
    const int cnt = (int)ax_min_ll(TQ_NT, e - kb);
    for (int u = 0; u < cnt; ++u) {
        bool any = false;
        AX_UNROLL
        for (int g = 0; g < G; ++g) any = any || pr[g * TQ_NT + u] != 0.0f;
        if (any) {
            const ax_u32 *vr = vp + (ax_u64)(kb + u - j0) * v_ss;
            float v0 = 0.0f, v1 = 0.0f;
            if (t < D) v0 = tq_val_elem<VM>(vr, t, sd, vc);
            if (t + TQ_NT < D) v1 = tq_val_elem<VM>(vr, t + TQ_NT, sd, vc);
            AX_UNROLL
            for (int g = 0; g < G; ++g) {
                const float pu = pr[g * TQ_NT + u];
                if (pu != 0.0f) {
                    acc0[g] += pu * v0;
                    acc1[g] += pu * v1;
                }
            }
        }
    }
    AX_SYNC();
}

template <int KM, int VM, bool PAGED, int G>
AX_DEV void tq_attng_core(const float *qx, const ax_u32 *kd, const ax_u32 *vd, const float *tab,
                          const float *alibi, float *out, float *part_o, float *part_lse,
                          ax_u32 Lq, ax_u32 Lk, ax_u32 Hq, ax_u32 Hkv, ax_u32 D, ax_u64 q_sb,
                          ax_u64 q_sh, ax_u64 q_ss, ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_ss,
                          ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_ss, const tq_sides &sd, float scale,
                          float softcap, int wl, int wr, ax_u32 n_splits, ax_u32 alibi_bs,
                          ax_u64 rows_total, const ax_pages *pg) {
    float *sm = ax_smem();
    float *qr = sm;               // G * D
    float *qs = qr + G * D;       // G * D
    float *red = qs + G * D;      // G * TQ_NT
    float *pr = red + G * TQ_NT;  // G * TQ_NT
    float *kc = pr + G * TQ_NT;   // 16
    float *vc = kc + 16;          // 16
    const ax_u32 t = AX_TID;
    const ax_u32 split = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    const ax_u32 packs = Hq / G;
    const ax_u32 group = Hq / Hkv;
    const bool prod = (KM == TQ_FM_GENERIC) && (sd.k_mode == TQ_PROD);
    for (ax_u64 rowi = AX_BID_X; rowi < (ax_u64)Lq * packs; rowi += AX_NBLK_X) {
        const ax_u32 i = (ax_u32)(rowi / packs);
        const ax_u32 h0 = (ax_u32)(rowi % packs) * G;
        const ax_u32 hk = h0 / group;
        AX_SYNC();
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            const float *qp = qx + b * q_sb + (h0 + g) * q_sh + i * q_ss;
            for (ax_u32 d = t; d < D; d += TQ_NT) {
                qr[g * D + d] = qp[d];
                qs[g * D + d] = prod ? qp[D + d] : 0.0f;
            }
        }
        if (t < 16u) {
            kc[t] = tab[t];
            vc[t] = tab[32u + t];
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
            const ax_u32 *kbase = kd + (ax_u64)b * k_sb + (ax_u64)hk * k_sh;
            const ax_u32 *vbase = vd + (ax_u64)b * v_sb + (ax_u64)hk * v_sh;
            for (long long kb = s_lo; kb < s_hi; kb += TQ_NT)
                tq_attng_window<KM, VM, G>(qr, qs, red, pr, kc, vc, kbase, vbase, k_ss, v_ss, 0, kb,
                                           s_hi, D, sd, scale, softcap, slope, pos, m, l, acc0,
                                           acc1);
        } else {
            long long j0 = 0;
            for (ax_u32 bi = 0; bi < pg->n && j0 < s_hi; ++bi) {
                const long long j1 = j0 + (long long)pg->rows[bi];
                const long long a = ax_max_ll(s_lo, j0), e = ax_min_ll(s_hi, j1);
                if (a < e) {
                    const ax_u64 bh = (ax_u64)b * Hkv + hk;
                    const ax_u32 *kbase = reinterpret_cast<const ax_u32 *>(pg->k[bi]) +
                                          bh * ((ax_u64)pg->k_cap[bi] * k_ss);
                    const ax_u32 *vbase = reinterpret_cast<const ax_u32 *>(pg->v[bi]) +
                                          bh * ((ax_u64)pg->v_cap[bi] * v_ss);
                    for (long long kb = a; kb < e; kb += TQ_NT)
                        tq_attng_window<KM, VM, G>(qr, qs, red, pr, kc, vc, kbase, vbase, k_ss,
                                                   v_ss, j0, kb, e, D, sd, scale, softcap, slope,
                                                   pos, m, l, acc0, acc1);
                }
                j0 = j1;
            }
        }
        AX_UNROLL
        for (int g = 0; g < G; ++g) {
            const float inv = l[g] > 0.0f ? 1.0f / l[g] : 0.0f;
            const ax_u64 orow = ((ax_u64)b * Hq + h0 + g) * Lq + i;
            if (n_splits == 1) {
                if (t < D) out[orow * D + t] = acc0[g] * inv;
                if (t + TQ_NT < D) out[orow * D + t + TQ_NT] = acc1[g] * inv;
            } else {
                const ax_u64 prow = (ax_u64)split * rows_total + orow;
                if (t < D) part_o[prow * D + t] = acc0[g] * inv;
                if (t + TQ_NT < D) part_o[prow * D + t + TQ_NT] = acc1[g] * inv;
                if (t == 0) part_lse[prow] = l[g] > 0.0f ? m[g] + logf(l[g]) : -AX_INF;
            }
        }
    }
}

// Cooperative-decoding tiled prefill attention over packed K/V rows (non-PROD modes).
// Decodes each BC key/value tile once into shared memory for BR query rows.
// Grid: (ceil(Lq / BR), Hq, B), Block: (TQ_NT, 1, 1).
// Shared memory: ((BR + BC) * (D + 1) + BR * BC + 32) * sizeof(float).
AX_DEV const ax_u32 *tq_paged_row(const ax_pages *pg, ax_u32 bi_start, long long j0_start,
                                  long long j, ax_u64 bh, ax_u64 row_stride, bool is_v) {
    ax_u32 bi = bi_start;
    long long j0 = j0_start;
    while (bi + 1u < pg->n && j >= j0 + (long long)pg->rows[bi]) {
        j0 += (long long)pg->rows[bi];
        ++bi;
    }
    const ax_u64 cap = is_v ? (ax_u64)pg->v_cap[bi] : (ax_u64)pg->k_cap[bi];
    const ax_u64 ptr = is_v ? pg->v[bi] : pg->k[bi];
    const ax_u32 *base = reinterpret_cast<const ax_u32 *>(ptr) + bh * (cap * row_stride);
    return base + (ax_u64)(j - j0) * row_stride;
}

template <bool PAGED, int BR, int BC, int DMAX>
AX_DEV void tq_tile_core(const float *qx, const ax_u32 *kd, const ax_u32 *vd, const float *tab,
                         const float *alibi, float *out, ax_u32 Lq, ax_u32 Lk, ax_u32 Hq,
                         ax_u32 Hkv, ax_u32 D, ax_u64 q_sb, ax_u64 q_sh, ax_u64 q_ss, ax_u64 k_sb,
                         ax_u64 k_sh, ax_u64 k_ss, ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_ss,
                         const tq_sides &sd, float scale, float softcap, int wl, int wr,
                         ax_u32 alibi_bs, const ax_pages *pg) {
    const int TPR = TQ_NT / BR;
    const int NACC = DMAX / TPR;
    const ax_u32 DP = D + 1;
    float *sm = ax_smem();
    float *Qs = sm;            // BR * DP
    float *KVs = Qs + BR * DP; // BC * DP
    float *S = KVs + BC * DP;  // BR * BC
    float *kc = S + BR * BC;   // 16
    float *vc = kc + 16;       // 16
    const ax_u32 t = AX_TID;
    const ax_u32 r = t / TPR;
    const ax_u32 lane = t % TPR;
    const ax_u32 h = AX_BID_Y;
    const ax_u32 b = AX_BID_Z;
    const ax_u32 hk = h / (Hq / Hkv);
    const ax_u64 bh = (ax_u64)b * Hkv + hk;
    const float *qb = qx + b * q_sb + h * q_sh;
    const ax_u32 *kbase = PAGED ? nullptr : kd + (ax_u64)b * k_sb + (ax_u64)hk * k_sh;
    const ax_u32 *vbase = PAGED ? nullptr : vd + (ax_u64)b * v_sb + (ax_u64)hk * v_sh;
    const long long off = (long long)Lk - (long long)Lq;
    const float slope = alibi ? alibi[(ax_u64)b * alibi_bs + h] : 0.0f;

    if (t < 16u) {
        kc[t] = tab[t];
        vc[t] = tab[32u + t];
    }
    AX_SYNC();

    for (ax_u64 tile = AX_BID_X; (long long)(tile * BR) < (long long)Lq; tile += AX_NBLK_X) {
        const long long i0 = (long long)tile * BR;
        AX_SYNC();
        for (ax_u32 idx = t; idx < BR * D; idx += TQ_NT) {
            const ax_u32 rr = idx / D, d = idx % D;
            const long long ii = i0 + rr;
            Qs[rr * DP + d] = ii < (long long)Lq ? scale * qb[(ax_u64)ii * q_ss + d] : 0.0f;
        }
        const long long last = ax_min_ll(i0 + BR, (long long)Lq) - 1;
        long long lo = wl >= 0 ? ax_max_ll(i0 + off - wl, 0) : 0;
        long long hi = wr >= 0 ? ax_min_ll(last + off + wr + 1, (long long)Lk) : (long long)Lk;
        if (lo > (long long)Lk) lo = Lk;
        if (hi < lo) hi = lo;
        const long long i = i0 + r;
        float m = -AX_INF, l = 0.0f;
        float acc[NACC];
        AX_UNROLL
        for (int a = 0; a < NACC; ++a) acc[a] = 0.0f;
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
            for (ax_u32 idx = t; idx < BC * D; idx += TQ_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                float kv = 0.0f;
                if (j < hi) {
                    const ax_u32 *kr = PAGED
                                           ? tq_paged_row(pg, bi_start, j0_start, j, bh, k_ss, false)
                                           : (kbase + (ax_u64)j * k_ss);
                    const float elem =
                        (sd.k_mode == TQ_FP8)
                            ? tq_fp8_elem(kr, d)
                            : ((sd.k_mode == TQ_NVFP4)
                                   ? tq_nvfp4_elem(kr, d, sd.k_cw)
                                   : tq_elem(kr, d, sd.k_mode, sd.k_cb, sd.k_cw, kc));
                    kv = ax_u2f(kr[sd.k_rsc]) * elem;
                }
                KVs[c * DP + d] = kv;
            }
            AX_SYNC();
            for (ax_u32 idx = t; idx < BR * BC; idx += TQ_NT) {
                const ax_u32 rr = idx / BC, c = idx % BC;
                const long long ii = i0 + rr, j = kt + c;
                const long long p2 = ii + off;
                float s = -AX_INF;
                if (ii < (long long)Lq && j < hi && (wl < 0 || j >= p2 - wl) &&
                    (wr < 0 || j <= p2 + wr)) {
                    const float *qr_row = Qs + rr * DP;
                    const float *kr_row = KVs + c * DP;
                    float dot = 0.0f;
                    for (ax_u32 d = 0; d < D; ++d) dot += qr_row[d] * kr_row[d];
                    s = tq_logit(dot, softcap, slope, p2 - j);
                }
                S[idx] = s;
            }
            AX_SYNC();
            for (ax_u32 idx = t; idx < BC * D; idx += TQ_NT) {
                const ax_u32 c = idx / D, d = idx % D;
                const long long j = kt + c;
                float vv = 0.0f;
                if (j < hi) {
                    const ax_u32 *vr = PAGED
                                           ? tq_paged_row(pg, bi_start, j0_start, j, bh, v_ss, true)
                                           : (vbase + (ax_u64)j * v_ss);
                    const float elem =
                        (sd.v_mode == TQ_FP8)
                            ? tq_fp8_elem(vr, d)
                            : ((sd.v_mode == TQ_NVFP4)
                                   ? tq_nvfp4_elem(vr, d, sd.v_cw)
                                   : tq_elem(vr, d, sd.v_mode, sd.v_cb, sd.v_cw, vc));
                    vv = ax_u2f(vr[sd.v_rsc]) * elem;
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
        if (i < (long long)Lq) {
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            const ax_u64 orow = ((ax_u64)b * Hq + h) * Lq + (ax_u64)i;
            AX_UNROLL
            for (int a = 0; a < NACC; ++a) {
                const ax_u32 d = lane + a * TPR;
                if (d < D) out[orow * D + d] = acc[a] * inv;
            }
        }
    }
}

#define TQ_ATTN_PARAMS                                                                             \
    const float *qx, const ax_u32 *kd, const ax_u32 *vd, const float *tab, const float *alibi,     \
        float *out, float *part_o, float *part_lse, ax_u32 Lq, ax_u32 Lk, ax_u32 Hq, ax_u32 Hkv,   \
        ax_u32 D, ax_u64 q_sb, ax_u64 q_sh, ax_u64 q_ss, ax_u64 k_sb, ax_u64 k_sh, ax_u64 k_ss,    \
        ax_u64 v_sb, ax_u64 v_sh, ax_u64 v_ss, ax_u32 k_mode, ax_u32 k_cb, ax_u32 k_cw,            \
        ax_u32 k_rsc, ax_u32 v_mode, ax_u32 v_cb, ax_u32 v_cw, ax_u32 v_rsc, float scale,          \
        float softcap, int wl, int wr, ax_u32 n_splits, ax_u32 alibi_bs, ax_u64 rows_total

#define TQ_ATTN_CALL_M(KM, VM, PAGED, PG)                                                          \
    {                                                                                              \
        const tq_sides sd = {k_mode, k_cb, k_cw, k_rsc, v_mode, v_cb, v_cw, v_rsc};                \
        tq_attn_core<KM, VM, PAGED>(qx, kd, vd, tab, alibi, out, part_o, part_lse, Lq, Lk, Hq,     \
                                    Hkv, D, q_sb, q_sh, q_ss, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss,  \
                                    sd, scale, softcap, wl, wr, n_splits, alibi_bs, rows_total,    \
                                    PG);                                                           \
    }

#define TQ_ATTNG_CALL_M(KM, VM, PAGED, G, PG)                                                      \
    {                                                                                              \
        const tq_sides sd = {k_mode, k_cb, k_cw, k_rsc, v_mode, v_cb, v_cw, v_rsc};                \
        tq_attng_core<KM, VM, PAGED, G>(qx, kd, vd, tab, alibi, out, part_o, part_lse, Lq, Lk, Hq, \
                                        Hkv, D, q_sb, q_sh, q_ss, k_sb, k_sh, k_ss, v_sb, v_sh,    \
                                        v_ss, sd, scale, softcap, wl, wr, n_splits, alibi_bs,      \
                                        rows_total, PG);                                           \
    }

#define TQ_TILE_CALL(PAGED, BR, BC, DMAX, PG)                                                      \
    {                                                                                              \
        const tq_sides sd = {k_mode, k_cb, k_cw, k_rsc, v_mode, v_cb, v_cw, v_rsc};                \
        tq_tile_core<PAGED, BR, BC, DMAX>(qx, kd, vd, tab, alibi, out, Lq, Lk, Hq, Hkv, D, q_sb,   \
                                          q_sh, q_ss, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss, sd,      \
                                          scale, softcap, wl, wr, alibi_bs, PG);                   \
    }

// GQA-packed entry (G query heads per workgroup), dense (pg == nullptr) or paged.
template <bool F8, bool PAGED, int G>
AX_DEV void tq_attng_impl(TQ_ATTN_PARAMS, const ax_pages *pg)
    TQ_ATTNG_CALL_M(F8 ? TQ_FM_FP8 : TQ_FM_GENERIC, F8 ? TQ_FM_FP8 : TQ_FM_GENERIC, PAGED, G, pg)

// Generic entry (every K / V format), kept with its historical signature.
AX_DEV void tq_attn_impl(TQ_ATTN_PARAMS) TQ_ATTN_CALL_M(TQ_FM_GENERIC, TQ_FM_GENERIC, false, nullptr)

// FP8 / NVFP4 / NVFP4+FP8 specialized entries.
AX_DEV void tq_attn_fp8_impl(TQ_ATTN_PARAMS) TQ_ATTN_CALL_M(TQ_FM_FP8, TQ_FM_FP8, false, nullptr)
AX_DEV void tq_attn_nvfp4_impl(TQ_ATTN_PARAMS)
    TQ_ATTN_CALL_M(TQ_FM_NVFP4, TQ_FM_NVFP4, false, nullptr)
AX_DEV void tq_attn_nvfp4_fp8_impl(TQ_ATTN_PARAMS)
    TQ_ATTN_CALL_M(TQ_FM_NVFP4, TQ_FM_FP8, false, nullptr)

// Paged block tables: the TQ_ATTN_PARAMS (kd / vd unused) followed by the table.
AX_DEV void tq_attnp_impl(TQ_ATTN_PARAMS, const ax_pages &pg)
    TQ_ATTN_CALL_M(TQ_FM_GENERIC, TQ_FM_GENERIC, true, &pg)
AX_DEV void tq_attnp_fp8_impl(TQ_ATTN_PARAMS, const ax_pages &pg)
    TQ_ATTN_CALL_M(TQ_FM_FP8, TQ_FM_FP8, true, &pg)
AX_DEV void tq_attnp_nvfp4_impl(TQ_ATTN_PARAMS, const ax_pages &pg)
    TQ_ATTN_CALL_M(TQ_FM_NVFP4, TQ_FM_NVFP4, true, &pg)
AX_DEV void tq_attnp_nvfp4_fp8_impl(TQ_ATTN_PARAMS, const ax_pages &pg)
    TQ_ATTN_CALL_M(TQ_FM_NVFP4, TQ_FM_FP8, true, &pg)

// Merges the key splits of tq_attn (weights once per row, see fa_combine_impl; shared memory:
// n_splits floats).
AX_DEV void tq_combine_impl(const float *part_o, const float *part_lse, float *o, ax_u64 rows,
                            ax_u32 D, ax_u32 n_splits) {
    float *w = ax_smem(); // n_splits
    const ax_u32 t = AX_TID;
    for (ax_u64 row = AX_BID_X; row < rows; row += AX_NBLK_X) {
        float mx = -AX_INF;
        for (ax_u32 s = 0; s < n_splits; ++s) mx = fmaxf(mx, part_lse[(ax_u64)s * rows + row]);
        AX_SYNC();
        for (ax_u32 s = t; s < n_splits; s += TQ_NT) {
            const float e = part_lse[(ax_u64)s * rows + row];
            w[s] = e == -AX_INF ? 0.0f : expf(e - mx);
        }
        AX_SYNC();
        float wsum = 0.0f;
        for (ax_u32 s = 0; s < n_splits; ++s) wsum += w[s];
        for (ax_u32 d = t; d < D; d += TQ_NT) {
            float acc = 0.0f;
            for (ax_u32 s = 0; s < n_splits; ++s) {
                const float ws = w[s];
                if (ws != 0.0f) acc += ws * part_o[((ax_u64)s * rows + row) * D + d];
            }
            o[row * D + d] = wsum > 0.0f ? acc / wsum : 0.0f;
        }
    }
}

#define TQ_QUANT_KERNELS(SFX, T)                                                                   \
    AX_KERNEL void AX_BOUNDS(TQ_NT)                                                                \
        tq_quant_##SFX(const T *src, ax_u32 *out, const float *tab, ax_u64 rows, ax_u32 D,         \
                       ax_u64 src_stride, ax_u32 mode, ax_u32 cb, ax_u32 code_words,               \
                       ax_u32 aux_words, ax_u32 row_words) {                                       \
        tq_quant_impl<T>(src, out, tab, rows, D, src_stride, mode, cb, code_words, aux_words,      \
                         row_words);                                                               \
    }                                                                                              \
    AX_KERNEL void AX_BOUNDS(TQ_NT)                                                                \
        tq_resid_##SFX(const T *src, const ax_u32 *data, const float *tab, float *resid,           \
                       ax_u64 rows, ax_u32 D, ax_u64 src_stride, ax_u32 cb, ax_u32 row_words,      \
                       ax_u32 rsc_off) {                                                           \
        tq_resid_impl<T>(src, data, tab, resid, rows, D, src_stride, cb, row_words, rsc_off);      \
    }                                                                                              \
    AX_KERNEL void AX_BOUNDS(TQ_NT)                                                                \
        tq_quant_prod_##SFX(const T *src, const float *tab, const float *s_mat, ax_u32 *out,       \
                            ax_u64 rows, ax_u32 D, ax_u64 src_stride, ax_u32 cb,                   \
                            ax_u32 code_words, ax_u32 row_words) {                                 \
        tq_quant_prod_fused_impl<T>(src, tab, s_mat, out, rows, D, src_stride, cb, code_words,     \
                                    row_words);                                                    \
    }                                                                                              \
    AX_KERNEL void AX_BOUNDS(TQ_NT)                                                                \
        tq_fwht_##SFX(const T *src, float *dst, const ax_u32 *sign_bits, ax_u64 rows, ax_u32 D,    \
                      ax_u64 src_stride, float inv_sqrt_d, ax_u32 forward) {                       \
        tq_fwht_impl<T>(src, dst, sign_bits, rows, D, src_stride, inv_sqrt_d, forward);            \
    }

#define TQ_ATTN_GROUP_SET(SFX, KM, VM)                                                             \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attn##SFX(TQ_ATTN_PARAMS)                                   \
        TQ_ATTN_CALL_M(KM, VM, false, nullptr)                                                     \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attnp##SFX(TQ_ATTN_PARAMS, const ax_pages pg)               \
        TQ_ATTN_CALL_M(KM, VM, true, &pg)                                                          \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attng2##SFX(TQ_ATTN_PARAMS)                                 \
        TQ_ATTNG_CALL_M(KM, VM, false, 2, nullptr)                                                 \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attnpg2##SFX(TQ_ATTN_PARAMS, const ax_pages pg)             \
        TQ_ATTNG_CALL_M(KM, VM, true, 2, &pg)                                                      \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attng4##SFX(TQ_ATTN_PARAMS)                                 \
        TQ_ATTNG_CALL_M(KM, VM, false, 4, nullptr)                                                 \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attnpg4##SFX(TQ_ATTN_PARAMS, const ax_pages pg)             \
        TQ_ATTNG_CALL_M(KM, VM, true, 4, &pg)                                                      \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attng8##SFX(TQ_ATTN_PARAMS)                                 \
        TQ_ATTNG_CALL_M(KM, VM, false, 8, nullptr)                                                 \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_attnpg8##SFX(TQ_ATTN_PARAMS, const ax_pages pg)             \
        TQ_ATTNG_CALL_M(KM, VM, true, 8, &pg)

#define TQ_COMMON_KERNELS                                                                          \
    AX_KERNEL void AX_BOUNDS(TQ_NT)                                                                \
        tq_signs(const ax_u32 *in, const float *proj, const float *resid, ax_u32 *out,             \
                 ax_u64 rows, ax_u32 D, ax_u32 code_words, ax_u32 row_words) {                     \
        tq_signs_impl(in, proj, resid, out, rows, D, code_words, row_words);                       \
    }                                                                                              \
    AX_KERNEL void AX_BOUNDS(TQ_NT)                                                                \
        tq_dequant(const ax_u32 *data, const float *tab, float *out, float *zs, ax_u64 rows,       \
                   ax_u32 D, ax_u32 mode, ax_u32 cb, ax_u32 code_words, ax_u32 row_words,          \
                   ax_u32 rsc_off) {                                                               \
        tq_dequant_impl(data, tab, out, zs, rows, D, mode, cb, code_words, row_words, rsc_off);    \
    }                                                                                              \
    TQ_ATTN_GROUP_SET(, TQ_FM_GENERIC, TQ_FM_GENERIC)                                              \
    TQ_ATTN_GROUP_SET(_fp8, TQ_FM_FP8, TQ_FM_FP8)                                                  \
    TQ_ATTN_GROUP_SET(_nvfp4, TQ_FM_NVFP4, TQ_FM_NVFP4)                                            \
    TQ_ATTN_GROUP_SET(_nvfp4_fp8, TQ_FM_NVFP4, TQ_FM_FP8)                                          \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tile2_d64(TQ_ATTN_PARAMS)                                   \
        TQ_TILE_CALL(false, 32, 32, 64, nullptr)                                                   \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tile2_d128(TQ_ATTN_PARAMS)                                  \
        TQ_TILE_CALL(false, 16, 32, 128, nullptr)                                                  \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tile2_d256(TQ_ATTN_PARAMS)                                  \
        TQ_TILE_CALL(false, 8, 16, 256, nullptr)                                                   \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tile3_d64(TQ_ATTN_PARAMS)                                   \
        TQ_TILE_CALL(false, 64, 64, 64, nullptr)                                                   \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tile3_d128(TQ_ATTN_PARAMS)                                  \
        TQ_TILE_CALL(false, 32, 64, 128, nullptr)                                                  \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tile3_d256(TQ_ATTN_PARAMS)                                  \
        TQ_TILE_CALL(false, 16, 32, 256, nullptr)                                                  \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tilep2_d64(TQ_ATTN_PARAMS, const ax_pages pg)               \
        TQ_TILE_CALL(true, 32, 32, 64, &pg)                                                        \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tilep2_d128(TQ_ATTN_PARAMS, const ax_pages pg)              \
        TQ_TILE_CALL(true, 16, 32, 128, &pg)                                                       \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tilep2_d256(TQ_ATTN_PARAMS, const ax_pages pg)              \
        TQ_TILE_CALL(true, 8, 16, 256, &pg)                                                        \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tilep3_d64(TQ_ATTN_PARAMS, const ax_pages pg)               \
        TQ_TILE_CALL(true, 64, 64, 64, &pg)                                                        \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tilep3_d128(TQ_ATTN_PARAMS, const ax_pages pg)              \
        TQ_TILE_CALL(true, 32, 64, 128, &pg)                                                       \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_tilep3_d256(TQ_ATTN_PARAMS, const ax_pages pg)              \
        TQ_TILE_CALL(true, 16, 32, 256, &pg)                                                       \
    AX_KERNEL void AX_BOUNDS(TQ_NT) tq_combine(const float *part_o, const float *part_lse,         \
                                               float *o, ax_u64 rows, ax_u32 D, ax_u32 n_splits) { \
        tq_combine_impl(part_o, part_lse, o, rows, D, n_splits);                                   \
    }
