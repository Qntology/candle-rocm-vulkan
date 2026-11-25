// Dedicated Vision Encoder Fused Ops (Part 4 Roadmap):
//   1. vis_u8_resize_norm_patchify  - Antialiased Bicubic/Bilinear u8 RGB -> norm -> patch grid
//   2. vis_res_ln_ls (+ f32res)     - Single-pass Chan-Welford Residual + LayerScale + LayerNorm
//   3. vis_bias_gelu                - Fused bias + 3-mode GELU (ExactErf / PytorchTanh / QuickGelu)
//   4. vis_qkv_2d_rope              - Fused Vision QKV unpack + optional norm + on-the-fly 2D RoPE
//   5. vis_spatial_merge_norm       - Zero-copy M x M spatial merge + Chan-Welford LayerNorm
//   6. vis_scatter_embeds           - Zero-copy in-place scatter of vision embeds into LLM embeds
//   7. vis_patch_add_interp_pos2d   - In-place F32 2D bilinear/bicubic learned pos embed addition
//
// Shared single-source header for HIP (vis_ops.hip), CUDA (NVRTC), and Host C++20 emulator.
#pragma once
#include "accel_common.h"
#include "accel_mma.h"

#define VIS_LN_NT 256
#define VIS_LN_SMEM_BYTES (3 * VIS_LN_NT * 4)

// Keys cubic interpolation kernel (a = -0.5, matches PIL / PyTorch bicubic).
AX_DEV float vis_cubic_weight(float x) {
    x = fabsf(x);
    if (x < 1.0f) {
        return (1.5f * x - 2.5f) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

AX_DEV float vis_linear_weight(float x) {
    x = fabsf(x);
    return x < 1.0f ? (1.0f - x) : 0.0f;
}

// ---------------------------------------------------------------------------
// 1. vis_u8_resize_norm_patchify
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void vis_u8_resize_norm_patchify_body(
    const ax_u8 *img_u8,
    TX *out,
    ax_u32 h_in,
    ax_u32 w_in,
    ax_u32 h_out,
    ax_u32 w_out,
    ax_u32 patch_size,
    ax_u32 bicubic,
    ax_u32 chw_in,
    float mean0,
    float mean1,
    float mean2,
    float inv_std0,
    float inv_std1,
    float inv_std2
) {
    const ax_u64 idx = (ax_u64)AX_BID_X * 256ull + (ax_u64)AX_TID;
    const ax_u64 total = (ax_u64)h_out * (ax_u64)w_out;
    if (idx >= total) return;

    const ax_u32 oy = (ax_u32)(idx / w_out);
    const ax_u32 ox = (ax_u32)(idx % w_out);

    const float sy = (float)h_in / (float)h_out;
    const float sx = (float)w_in / (float)w_out;
    const float r_base = bicubic ? 2.0f : 1.0f;
    const float fy = sy > 1.0f ? sy : 1.0f;
    const float fx = sx > 1.0f ? sx : 1.0f;
    const float ry = r_base * fy;
    const float rx = r_base * fx;
    const float inv_fy = 1.0f / fy;
    const float inv_fx = 1.0f / fx;

    const float cy = ((float)oy + 0.5f) * sy - 0.5f;
    const float cx = ((float)ox + 0.5f) * sx - 0.5f;

    int y0 = (int)floorf(cy - ry);
    int y1 = (int)ceilf(cy + ry);
    int x0 = (int)floorf(cx - rx);
    int x1 = (int)ceilf(cx + rx);
    if (y0 < 0) y0 = 0;
    if (y1 > (int)h_in - 1) y1 = (int)h_in - 1;
    if (x0 < 0) x0 = 0;
    if (x1 > (int)w_in - 1) x1 = (int)w_in - 1;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    float wsum = 0.0f;
    const ax_u64 hw_in = (ax_u64)h_in * (ax_u64)w_in;

    for (int iy = y0; iy <= y1; ++iy) {
        const float dy = ((float)iy - cy) * inv_fy;
        const float wy = bicubic ? vis_cubic_weight(dy) : vis_linear_weight(dy);
        if (wy == 0.0f) continue;
        for (int ix = x0; ix <= x1; ++ix) {
            const float dx = ((float)ix - cx) * inv_fx;
            const float wx = bicubic ? vis_cubic_weight(dx) : vis_linear_weight(dx);
            const float w = wy * wx;
            if (w == 0.0f) continue;
            float p0, p1, p2;
            if (chw_in) {
                const ax_u64 pix = (ax_u64)iy * w_in + (ax_u64)ix;
                p0 = (float)img_u8[pix];
                p1 = (float)img_u8[hw_in + pix];
                p2 = (float)img_u8[2ull * hw_in + pix];
            } else {
                const ax_u64 pix = ((ax_u64)iy * w_in + (ax_u64)ix) * 3ull;
                p0 = (float)img_u8[pix + 0];
                p1 = (float)img_u8[pix + 1];
                p2 = (float)img_u8[pix + 2];
            }
            acc0 = fmaf(w, p0, acc0);
            acc1 = fmaf(w, p1, acc1);
            acc2 = fmaf(w, p2, acc2);
            wsum += w;
        }
    }

    const float inv_w = wsum > 0.0f ? (1.0f / (wsum * 255.0f)) : 0.0f;
    const float v0 = (acc0 * inv_w - mean0) * inv_std0;
    const float v1 = (acc1 * inv_w - mean1) * inv_std1;
    const float v2 = (acc2 * inv_w - mean2) * inv_std2;

    if (patch_size <= 1u) {
        const ax_u64 hw_out = total;
        ax_store(out, idx, v0);
        ax_store(out, hw_out + idx, v1);
        ax_store(out, 2ull * hw_out + idx, v2);
    } else {
        const ax_u32 pw = w_out / patch_size;
        const ax_u32 py = oy / patch_size;
        const ax_u32 px = ox / patch_size;
        const ax_u32 ly = oy % patch_size;
        const ax_u32 lx = ox % patch_size;
        const ax_u64 p_area = (ax_u64)patch_size * (ax_u64)patch_size;
        const ax_u64 p_idx = (ax_u64)py * (ax_u64)pw + (ax_u64)px;
        const ax_u64 base = p_idx * 3ull * p_area + (ax_u64)ly * (ax_u64)patch_size + (ax_u64)lx;
        ax_store(out, base, v0);
        ax_store(out, base + p_area, v1);
        ax_store(out, base + 2ull * p_area, v2);
    }
}

#define VIS_RESIZE_PATCH_KERNEL(NAME, TX)                                                          \
    AX_KERNEL void NAME(const ax_u8 *img_u8, TX *out, ax_u32 h_in, ax_u32 w_in, ax_u32 h_out,      \
                        ax_u32 w_out, ax_u32 patch_size, ax_u32 bicubic, ax_u32 chw_in,            \
                        float mean0, float mean1, float mean2, float inv_std0, float inv_std1,     \
                        float inv_std2) {                                                          \
        vis_u8_resize_norm_patchify_body<TX>(img_u8, out, h_in, w_in, h_out, w_out, patch_size,    \
                                             bicubic, chw_in, mean0, mean1, mean2, inv_std0,       \
                                             inv_std1, inv_std2);                                  \
    }

// ---------------------------------------------------------------------------
// 2. vis_res_layernorm_ls (Single-Pass Chan-Welford Residual + LayerScale + LN)
// ---------------------------------------------------------------------------
template <typename TX, typename TRES>
AX_DEV void vis_res_ln_ls_body(
    const TRES *res_in,
    const TX *delta,
    const float *ls_gamma,
    const float *ln_weight,
    const float *ln_bias,
    TRES *res_out,
    TX *norm_out,
    ax_u64 rows,
    ax_u32 d,
    float eps
) {
    const ax_u64 row = (ax_u64)AX_BID_X;
    if (row >= rows) return;
    const ax_u32 t = AX_TID;
    const ax_u64 base = row * (ax_u64)d;

    float l_mean = 0.0f;
    float l_m2 = 0.0f;
    float l_cnt = 0.0f;

    for (ax_u32 i = t; i < d; i += VIS_LN_NT) {
        float v = ax_load(res_in, base + i);
        if (delta) {
            float dv = ax_load(delta, base + i);
            if (ls_gamma) dv *= ls_gamma[i];
            v += dv;
        }
        if (res_out) {
            ax_store(res_out, base + i, v);
        }
        l_cnt += 1.0f;
        const float diff = v - l_mean;
        l_mean += diff / l_cnt;
        l_m2 = fmaf(diff, v - l_mean, l_m2);
    }

    float g_mean = 0.0f, g_m2 = 0.0f;
    ax_block_welford<VIS_LN_NT>(l_mean, l_m2, l_cnt, ax_smem(), g_mean, g_m2);
    const float inv_std = 1.0f / sqrtf(g_m2 / (float)d + eps);

    for (ax_u32 i = t; i < d; i += VIS_LN_NT) {
        float v;
        if (res_out) {
            v = ax_load(res_out, base + i);
        } else {
            v = ax_load(res_in, base + i);
            if (delta) {
                float dv = ax_load(delta, base + i);
                if (ls_gamma) dv *= ls_gamma[i];
                v += dv;
            }
        }
        float y = (v - g_mean) * inv_std;
        if (ln_weight) y *= ln_weight[i];
        if (ln_bias) y += ln_bias[i];
        ax_store(norm_out, base + i, y);
    }
}

#define VIS_RES_LN_LS_KERNEL(NAME, TX, TRES)                                                       \
    AX_KERNEL void NAME(const TRES *res_in, const TX *delta, const float *ls_gamma,                \
                        const float *ln_weight, const float *ln_bias, TRES *res_out, TX *norm_out, \
                        ax_u64 rows, ax_u32 d, float eps) {                                        \
        vis_res_ln_ls_body<TX, TRES>(res_in, delta, ls_gamma, ln_weight, ln_bias, res_out,         \
                                     norm_out, rows, d, eps);                                      \
    }

// ---------------------------------------------------------------------------
// 3. vis_bias_gelu (mode 0 = ExactErf, 1 = PytorchTanh, 2 = QuickGelu)
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void vis_bias_gelu_body(
    const TX *x,
    const TX *bias,
    TX *y,
    ax_u64 n_elems,
    ax_u32 dim,
    ax_u32 mode
) {
    const ax_u64 idx = (ax_u64)AX_BID_X * 256ull + (ax_u64)AX_TID;
    if (idx >= n_elems) return;
    float v = ax_load(x, idx);
    if (bias) {
        v += ax_load(bias, idx % (ax_u64)dim);
    }
    float out;
    if (mode == 0u) {
        out = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    } else if (mode == 1u) {
        constexpr float SQRT_2_OVER_PI = 0.79788456080286535588f;
        const float inner = SQRT_2_OVER_PI * fmaf(0.044715f * v * v, v, v);
        out = 0.5f * v * (1.0f + tanhf(inner));
    } else {
        out = v / (1.0f + expf(-1.702f * v));
    }
    ax_store(y, idx, out);
}

#define VIS_BIAS_GELU_KERNEL(NAME, TX)                                                             \
    AX_KERNEL void NAME(const TX *x, const TX *bias, TX *y, ax_u64 n_elems, ax_u32 dim,            \
                        ax_u32 mode) {                                                             \
        vis_bias_gelu_body<TX>(x, bias, y, n_elems, dim, mode);                                    \
    }

// ---------------------------------------------------------------------------
// 4. vis_qkv_2d_rope
//    Input qkv: [N, 3 * H * D], optional bias: [3 * H * D], optional q/k head norm weights: [D],
//    patch_yx: [N, 2] u32 (y, x), inv_freq: [rot_dim / 4] f32.
//    Outputs q_out, k_out, v_out: [H, N, D].
//    flags: bit 0 = interleaved 2D pairs (else split-half), bit 1 = RMSNorm plus-one.
// ---------------------------------------------------------------------------
#define VIS_QKV_NT 128

template <typename TX>
AX_DEV void vis_qkv_2d_rope_body(
    const TX *qkv,
    const TX *bias,
    const float *q_norm_w,
    const float *k_norm_w,
    const ax_u32 *patch_yx,
    const float *inv_freq,
    TX *q_out,
    TX *k_out,
    TX *v_out,
    ax_u32 n_tokens,
    ax_u32 num_heads,
    ax_u32 head_dim,
    ax_u32 rot_dim,
    ax_u32 flags,
    float eps
) {
    const ax_u32 tok = AX_BID_X;
    const ax_u32 h = AX_BID_Y;
    if (tok >= n_tokens || h >= num_heads) return;
    const ax_u32 t = AX_TID;

    // Shared memory layout: q_buf[head_dim], k_buf[head_dim], scratch[VIS_QKV_NT]
    float *sm = ax_smem();
    float *q_buf = sm;
    float *k_buf = q_buf + head_dim;
    float *scratch = k_buf + head_dim;

    const ax_u64 hd_all = (ax_u64)num_heads * (ax_u64)head_dim;
    const ax_u64 qkv_base = (ax_u64)tok * 3ull * hd_all;
    const ax_u64 q_off = (ax_u64)h * head_dim;
    const ax_u64 k_off = hd_all + q_off;
    const ax_u64 v_off = 2ull * hd_all + q_off;
    const ax_u64 dst_base = ((ax_u64)h * (ax_u64)n_tokens + (ax_u64)tok) * (ax_u64)head_dim;

    float q_sq = 0.0f, k_sq = 0.0f;
    for (ax_u32 i = t; i < head_dim; i += VIS_QKV_NT) {
        float qv = ax_load(qkv, qkv_base + q_off + i);
        float kv = ax_load(qkv, qkv_base + k_off + i);
        float vv = ax_load(qkv, qkv_base + v_off + i);
        if (bias) {
            qv += ax_load(bias, q_off + i);
            kv += ax_load(bias, k_off + i);
            vv += ax_load(bias, v_off + i);
        }
        q_buf[i] = qv;
        k_buf[i] = kv;
        q_sq = fmaf(qv, qv, q_sq);
        k_sq = fmaf(kv, kv, k_sq);
        ax_store(v_out, dst_base + i, vv);
    }

    const bool plus_one = (flags & 2u) != 0u;
    if (q_norm_w) {
        scratch[t] = q_sq;
        AX_SYNC();
        for (ax_u32 s = VIS_QKV_NT / 2; s > 0u; s >>= 1u) {
            if (t < s) scratch[t] += scratch[t + s];
            AX_SYNC();
        }
        const float inv = 1.0f / sqrtf(scratch[0] / (float)head_dim + eps);
        for (ax_u32 i = t; i < head_dim; i += VIS_QKV_NT) {
            const float w = plus_one ? (1.0f + q_norm_w[i]) : q_norm_w[i];
            q_buf[i] = (q_buf[i] * inv) * w;
        }
        AX_SYNC();
    }
    if (k_norm_w) {
        scratch[t] = k_sq;
        AX_SYNC();
        for (ax_u32 s = VIS_QKV_NT / 2; s > 0u; s >>= 1u) {
            if (t < s) scratch[t] += scratch[t + s];
            AX_SYNC();
        }
        const float inv = 1.0f / sqrtf(scratch[0] / (float)head_dim + eps);
        for (ax_u32 i = t; i < head_dim; i += VIS_QKV_NT) {
            const float w = plus_one ? (1.0f + k_norm_w[i]) : k_norm_w[i];
            k_buf[i] = (k_buf[i] * inv) * w;
        }
        AX_SYNC();
    }
    if (!q_norm_w && !k_norm_w) {
        AX_SYNC();
    }

    const ax_u32 py = patch_yx[(ax_u64)tok * 2ull + 0ull];
    const ax_u32 px = patch_yx[(ax_u64)tok * 2ull + 1ull];
    const ax_u32 quarter = rot_dim >> 2;
    const ax_u32 half_rot = rot_dim >> 1;
    const bool adjacent = (flags & 1u) != 0u;

    // First half of rotary pairs uses Y coordinate, second half uses X coordinate.
    for (ax_u32 pair = t; pair < half_rot; pair += VIS_QKV_NT) {
        const bool is_x = pair >= quarter;
        const ax_u32 f_idx = is_x ? (pair - quarter) : pair;
        const ax_u32 pos = is_x ? px : py;
        float c = 1.0f, s = 0.0f;
        ax_cody_waite_sincos(pos, inv_freq[f_idx], c, s);
        const ax_u32 i0 = adjacent ? (2u * pair) : pair;
        const ax_u32 i1 = adjacent ? (2u * pair + 1u) : (pair + half_rot);
        const float q0 = q_buf[i0], q1 = q_buf[i1];
        const float k0 = k_buf[i0], k1 = k_buf[i1];
        ax_store(q_out, dst_base + i0, fmaf(q0, c, -q1 * s));
        ax_store(q_out, dst_base + i1, fmaf(q1, c, q0 * s));
        ax_store(k_out, dst_base + i0, fmaf(k0, c, -k1 * s));
        ax_store(k_out, dst_base + i1, fmaf(k1, c, k0 * s));
    }
    for (ax_u32 i = rot_dim + t; i < head_dim; i += VIS_QKV_NT) {
        ax_store(q_out, dst_base + i, q_buf[i]);
        ax_store(k_out, dst_base + i, k_buf[i]);
    }
}

#define VIS_QKV_2D_ROPE_KERNEL(NAME, TX)                                                           \
    AX_KERNEL void NAME(const TX *qkv, const TX *bias, const float *q_norm_w,                      \
                        const float *k_norm_w, const ax_u32 *patch_yx, const float *inv_freq,      \
                        TX *q_out, TX *k_out, TX *v_out, ax_u32 n_tokens, ax_u32 num_heads,        \
                        ax_u32 head_dim, ax_u32 rot_dim, ax_u32 flags, float eps) {                \
        vis_qkv_2d_rope_body<TX>(qkv, bias, q_norm_w, k_norm_w, patch_yx, inv_freq, q_out, k_out,  \
                                 v_out, n_tokens, num_heads, head_dim, rot_dim, flags, eps);       \
    }

// ---------------------------------------------------------------------------
// 5. vis_spatial_merge_norm
//    mode 0: PreMergeNorm  (LN each [D] patch, then concatenate M x M -> [M*M*D])
//    mode 1: PostMergeNorm (concatenate M x M -> [M*M*D], then single-pass Chan-Welford LN)
//    mode 2: AvgPoolNorm   (average M x M patches -> [D], then single-pass Chan-Welford LN)
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void vis_spatial_merge_norm_body(
    const TX *x,
    const float *ln_weight,
    const float *ln_bias,
    TX *y,
    ax_u32 h_p,
    ax_u32 w_p,
    ax_u32 d,
    ax_u32 merge,
    ax_u32 mode,
    float eps
) {
    const ax_u32 mh = h_p / merge;
    const ax_u32 mw = w_p / merge;
    const ax_u32 out_tok = AX_BID_X;
    if (out_tok >= mh * mw) return;
    const ax_u32 t = AX_TID;
    const ax_u32 oy = out_tok / mw;
    const ax_u32 ox = out_tok % mw;
    const ax_u32 m2 = merge * merge;

    if (mode == 0u) {
        // PreMergeNorm: normalize each of the M*M constituent patches independently over D,
        // then write directly into merged row [out_tok, M*M*D].
        const ax_u64 dst_row = (ax_u64)out_tok * (ax_u64)(m2 * d);
        for (ax_u32 sub = 0u; sub < m2; ++sub) {
            const ax_u32 dy = sub / merge;
            const ax_u32 dx = sub % merge;
            const ax_u64 in_tok = (ax_u64)(oy * merge + dy) * (ax_u64)w_p + (ax_u64)(ox * merge + dx);
            const ax_u64 src_base = in_tok * (ax_u64)d;
            float l_mean = 0.0f, l_m2 = 0.0f, l_cnt = 0.0f;
            for (ax_u32 i = t; i < d; i += VIS_LN_NT) {
                const float v = ax_load(x, src_base + i);
                l_cnt += 1.0f;
                const float diff = v - l_mean;
                l_mean += diff / l_cnt;
                l_m2 = fmaf(diff, v - l_mean, l_m2);
            }
            float g_mean = 0.0f, g_m2 = 0.0f;
            ax_block_welford<VIS_LN_NT>(l_mean, l_m2, l_cnt, ax_smem(), g_mean, g_m2);
            const float inv_std = 1.0f / sqrtf(g_m2 / (float)d + eps);
            const ax_u64 dst_sub = dst_row + (ax_u64)sub * (ax_u64)d;
            for (ax_u32 i = t; i < d; i += VIS_LN_NT) {
                float v = (ax_load(x, src_base + i) - g_mean) * inv_std;
                if (ln_weight) v *= ln_weight[i];
                if (ln_bias) v += ln_bias[i];
                ax_store(y, dst_sub + i, v);
            }
        }
    } else if (mode == 1u) {
        // PostMergeNorm: single Chan-Welford LN over all M*M*D elements directly from source grid.
        const ax_u32 d_out = m2 * d;
        const ax_u64 dst_row = (ax_u64)out_tok * (ax_u64)d_out;
        float l_mean = 0.0f, l_m2 = 0.0f, l_cnt = 0.0f;
        for (ax_u32 idx = t; idx < d_out; idx += VIS_LN_NT) {
            const ax_u32 sub = idx / d;
            const ax_u32 c = idx % d;
            const ax_u32 dy = sub / merge;
            const ax_u32 dx = sub % merge;
            const ax_u64 in_tok = (ax_u64)(oy * merge + dy) * (ax_u64)w_p + (ax_u64)(ox * merge + dx);
            const float v = ax_load(x, in_tok * (ax_u64)d + c);
            l_cnt += 1.0f;
            const float diff = v - l_mean;
            l_mean += diff / l_cnt;
            l_m2 = fmaf(diff, v - l_mean, l_m2);
        }
        float g_mean = 0.0f, g_m2 = 0.0f;
        ax_block_welford<VIS_LN_NT>(l_mean, l_m2, l_cnt, ax_smem(), g_mean, g_m2);
        const float inv_std = 1.0f / sqrtf(g_m2 / (float)d_out + eps);
        for (ax_u32 idx = t; idx < d_out; idx += VIS_LN_NT) {
            const ax_u32 sub = idx / d;
            const ax_u32 c = idx % d;
            const ax_u32 dy = sub / merge;
            const ax_u32 dx = sub % merge;
            const ax_u64 in_tok = (ax_u64)(oy * merge + dy) * (ax_u64)w_p + (ax_u64)(ox * merge + dx);
            float v = (ax_load(x, in_tok * (ax_u64)d + c) - g_mean) * inv_std;
            if (ln_weight) v *= ln_weight[idx];
            if (ln_bias) v += ln_bias[idx];
            ax_store(y, dst_row + idx, v);
        }
    } else {
        // AvgPoolNorm: average M*M neighbors -> [D], then Chan-Welford LN over D.
        const ax_u64 dst_row = (ax_u64)out_tok * (ax_u64)d;
        const float inv_m2 = 1.0f / (float)m2;
        float l_mean = 0.0f, l_m2 = 0.0f, l_cnt = 0.0f;
        for (ax_u32 c = t; c < d; c += VIS_LN_NT) {
            float sum = 0.0f;
            for (ax_u32 sub = 0u; sub < m2; ++sub) {
                const ax_u32 dy = sub / merge;
                const ax_u32 dx = sub % merge;
                const ax_u64 in_tok = (ax_u64)(oy * merge + dy) * (ax_u64)w_p + (ax_u64)(ox * merge + dx);
                sum += ax_load(x, in_tok * (ax_u64)d + c);
            }
            const float v = sum * inv_m2;
            l_cnt += 1.0f;
            const float diff = v - l_mean;
            l_mean += diff / l_cnt;
            l_m2 = fmaf(diff, v - l_mean, l_m2);
        }
        float g_mean = 0.0f, g_m2 = 0.0f;
        ax_block_welford<VIS_LN_NT>(l_mean, l_m2, l_cnt, ax_smem(), g_mean, g_m2);
        const float inv_std = 1.0f / sqrtf(g_m2 / (float)d + eps);
        for (ax_u32 c = t; c < d; c += VIS_LN_NT) {
            float sum = 0.0f;
            for (ax_u32 sub = 0u; sub < m2; ++sub) {
                const ax_u32 dy = sub / merge;
                const ax_u32 dx = sub % merge;
                const ax_u64 in_tok = (ax_u64)(oy * merge + dy) * (ax_u64)w_p + (ax_u64)(ox * merge + dx);
                sum += ax_load(x, in_tok * (ax_u64)d + c);
            }
            float v = (sum * inv_m2 - g_mean) * inv_std;
            if (ln_weight) v *= ln_weight[c];
            if (ln_bias) v += ln_bias[c];
            ax_store(y, dst_row + c, v);
        }
    }
}

#define VIS_SPATIAL_MERGE_KERNEL(NAME, TX)                                                         \
    AX_KERNEL void NAME(const TX *x, const float *ln_weight, const float *ln_bias, TX *y,          \
                        ax_u32 h_p, ax_u32 w_p, ax_u32 d, ax_u32 merge, ax_u32 mode, float eps) {  \
        vis_spatial_merge_norm_body<TX>(x, ln_weight, ln_bias, y, h_p, w_p, d, merge, mode, eps);  \
    }

// ---------------------------------------------------------------------------
// 6. vis_scatter_embeds (zero-copy in-place vision embedding scatter)
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void vis_scatter_embeds_body(
    TX *embeds,
    const TX *vis,
    const ax_u32 *indices,
    ax_u32 n_vis,
    ax_u32 d,
    ax_u32 l_total
) {
    const ax_u32 v_idx = AX_BID_X;
    if (v_idx >= n_vis) return;
    const ax_u32 dst_tok = indices[v_idx];
    if (dst_tok >= l_total) return;
    const ax_u64 src_base = (ax_u64)v_idx * (ax_u64)d;
    const ax_u64 dst_base = (ax_u64)dst_tok * (ax_u64)d;
    for (ax_u32 i = AX_TID; i < d; i += 256u) {
        embeds[dst_base + i] = vis[src_base + i];
    }
}

#define VIS_SCATTER_EMBEDS_KERNEL(NAME, TX)                                                        \
    AX_KERNEL void NAME(TX *embeds, const TX *vis, const ax_u32 *indices, ax_u32 n_vis, ax_u32 d,  \
                        ax_u32 l_total) {                                                          \
        vis_scatter_embeds_body<TX>(embeds, vis, indices, n_vis, d, l_total);                      \
    }

// ---------------------------------------------------------------------------
// 7. vis_patch_add_interp_pos2d (in-place F32 bilinear/bicubic 2D pos embed add)
// ---------------------------------------------------------------------------
template <typename TX>
AX_DEV void vis_patch_add_interp_pos2d_body(
    TX *patches,
    const TX *pos_table,
    ax_u32 g_h,
    ax_u32 g_w,
    ax_u32 h_p,
    ax_u32 w_p,
    ax_u32 d,
    ax_u32 bicubic,
    ax_u32 align_corners
) {
    const ax_u32 tok = AX_BID_X;
    if (tok >= h_p * w_p) return;
    const ax_u32 py = tok / w_p;
    const ax_u32 px = tok % w_p;

    float fy, fx;
    if (align_corners && h_p > 1u) {
        fy = (float)py * (float)(g_h - 1u) / (float)(h_p - 1u);
    } else {
        fy = ((float)py + 0.5f) * (float)g_h / (float)h_p - 0.5f;
    }
    if (align_corners && w_p > 1u) {
        fx = (float)px * (float)(g_w - 1u) / (float)(w_p - 1u);
    } else {
        fx = ((float)px + 0.5f) * (float)g_w / (float)w_p - 0.5f;
    }

    const ax_u64 dst_base = (ax_u64)tok * (ax_u64)d;
    if (!bicubic) {
        const int y0 = (int)floorf(fy);
        const int x0 = (int)floorf(fx);
        const float wy1 = fy - (float)y0;
        const float wx1 = fx - (float)x0;
        const float wy0 = 1.0f - wy1;
        const float wx0 = 1.0f - wx1;
        const int cy0 = y0 < 0 ? 0 : (y0 >= (int)g_h ? (int)g_h - 1 : y0);
        const int cy1 = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= (int)g_h ? (int)g_h - 1 : (y0 + 1));
        const int cx0 = x0 < 0 ? 0 : (x0 >= (int)g_w ? (int)g_w - 1 : x0);
        const int cx1 = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= (int)g_w ? (int)g_w - 1 : (x0 + 1));
        const ax_u64 p00 = ((ax_u64)cy0 * g_w + (ax_u64)cx0) * d;
        const ax_u64 p01 = ((ax_u64)cy0 * g_w + (ax_u64)cx1) * d;
        const ax_u64 p10 = ((ax_u64)cy1 * g_w + (ax_u64)cx0) * d;
        const ax_u64 p11 = ((ax_u64)cy1 * g_w + (ax_u64)cx1) * d;
        const float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
        for (ax_u32 c = AX_TID; c < d; c += 256u) {
            const float pe = w00 * ax_load(pos_table, p00 + c) +
                             w01 * ax_load(pos_table, p01 + c) +
                             w10 * ax_load(pos_table, p10 + c) +
                             w11 * ax_load(pos_table, p11 + c);
            ax_store(patches, dst_base + c, ax_load(patches, dst_base + c) + pe);
        }
    } else {
        const int y0 = (int)floorf(fy);
        const int x0 = (int)floorf(fx);
        float wy[4], wx[4];
        int cy[4], cx[4];
        float wsum_y = 0.0f, wsum_x = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const int iy = y0 - 1 + k;
            const int ix = x0 - 1 + k;
            wy[k] = vis_cubic_weight((float)iy - fy);
            wx[k] = vis_cubic_weight((float)ix - fx);
            wsum_y += wy[k];
            wsum_x += wx[k];
            cy[k] = iy < 0 ? 0 : (iy >= (int)g_h ? (int)g_h - 1 : iy);
            cx[k] = ix < 0 ? 0 : (ix >= (int)g_w ? (int)g_w - 1 : ix);
        }
        const float inv_w = (wsum_y * wsum_x != 0.0f) ? (1.0f / (wsum_y * wsum_x)) : 1.0f;
        for (ax_u32 c = AX_TID; c < d; c += 256u) {
            float acc = 0.0f;
            for (int ky = 0; ky < 4; ++ky) {
                for (int kx = 0; kx < 4; ++kx) {
                    const ax_u64 p = ((ax_u64)cy[ky] * g_w + (ax_u64)cx[kx]) * d + c;
                    acc = fmaf(wy[ky] * wx[kx], ax_load(pos_table, p), acc);
                }
            }
            ax_store(patches, dst_base + c, ax_load(patches, dst_base + c) + acc * inv_w);
        }
    }
}

#define VIS_PATCH_ADD_POS2D_KERNEL(NAME, TX)                                                       \
    AX_KERNEL void NAME(TX *patches, const TX *pos_table, ax_u32 g_h, ax_u32 g_w, ax_u32 h_p,      \
                        ax_u32 w_p, ax_u32 d, ax_u32 bicubic, ax_u32 align_corners) {              \
        vis_patch_add_interp_pos2d_body<TX>(patches, pos_table, g_h, g_w, h_p, w_p, d, bicubic,    \
                                            align_corners);                                        \
    }
