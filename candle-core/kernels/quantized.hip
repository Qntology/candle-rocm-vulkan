#include "common.h"

DEVFN uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
DEVFN float rd_f16(const uint8_t *p) { return f16_to_float((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }
DEVFN float rd_bf16(const uint8_t *p) { return bf16_to_float((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }
DEVFN float rd_f32(const uint8_t *p) {
    candle_fu32 v;
    v.u = rd_u32(p);
    return v.f;
}

DEVFN void scale_min_k4(const int j, const uint8_t *q, uint8_t &sc, uint8_t &m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

DEVFN int q3k_scale(const uint8_t *s, const int is) {
    if (is < 4) return (s[is] & 0xF) | ((s[8 + is] & 3) << 4);
    if (is < 8) return (s[is] & 0xF) | (((s[4 + is] >> 2) & 3) << 4);
    if (is < 12) return (s[is - 8] >> 4) | (((s[is] >> 4) & 3) << 4);
    return (s[is - 8] >> 4) | (((s[is - 4] >> 6) & 3) << 4);
}

struct DqF32 {
    static const int QK = 1;
    static const int BS = 4;
    static DEVFN float deq(const uint8_t *b, const int) { return rd_f32(b); }
};
struct DqF16 {
    static const int QK = 1;
    static const int BS = 2;
    static DEVFN float deq(const uint8_t *b, const int) { return rd_f16(b); }
};
struct DqBF16 {
    static const int QK = 1;
    static const int BS = 2;
    static DEVFN float deq(const uint8_t *b, const int) { return rd_bf16(b); }
};
struct DqQ4_0 {
    static const int QK = 32;
    static const int BS = 18;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const float d = rd_f16(b);
        const uint8_t q = b[2 + (e & 15)];
        const int v = (e < 16 ? (q & 0xF) : (q >> 4)) - 8;
        return (float)v * d;
    }
};
struct DqQ4_1 {
    static const int QK = 32;
    static const int BS = 20;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const float d = rd_f16(b);
        const float m = rd_f16(b + 2);
        const uint8_t q = b[4 + (e & 15)];
        const int v = e < 16 ? (q & 0xF) : (q >> 4);
        return (float)v * d + m;
    }
};
DEVFN int q5_value(const uint8_t *qs, const uint32_t qh, const int e) {
    const int j = e & 15;
    const uint8_t q = qs[j];
    if (e < 16) {
        const int xh = (int)(((qh >> j) << 4) & 0x10);
        return (q & 0xF) | xh;
    }
    const int xh = (int)((qh >> (j + 12)) & 0x10);
    return (q >> 4) | xh;
}
struct DqQ5_0 {
    static const int QK = 32;
    static const int BS = 22;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const float d = rd_f16(b);
        const uint32_t qh = rd_u32(b + 2);
        return (float)(q5_value(b + 6, qh, e) - 16) * d;
    }
};
struct DqQ5_1 {
    static const int QK = 32;
    static const int BS = 24;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const float d = rd_f16(b);
        const float m = rd_f16(b + 2);
        const uint32_t qh = rd_u32(b + 4);
        return (float)q5_value(b + 8, qh, e) * d + m;
    }
};
struct DqQ8_0 {
    static const int QK = 32;
    static const int BS = 34;
    static DEVFN float deq(const uint8_t *b, const int e) { return (float)(int8_t)b[2 + e] * rd_f16(b); }
};
struct DqQ8_1 {
    static const int QK = 32;
    static const int BS = 36;
    static DEVFN float deq(const uint8_t *b, const int e) { return (float)(int8_t)b[4 + e] * rd_f16(b); }
};
struct DqQ2K {
    static const int QK = 256;
    static const int BS = 84;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const int n = e >> 7;
        const int r = e & 127;
        const int j = r >> 5;
        const int r2 = r & 31;
        const int sub = r2 >> 4;
        const int l = r2 & 15;
        const uint8_t q = b[16 + 32 * n + 16 * sub + l];
        const uint8_t sc = b[8 * n + 2 * j + sub];
        const float d = rd_f16(b + 80);
        const float dmin = rd_f16(b + 82);
        const float dl = d * (float)(sc & 0xF);
        const float ml = dmin * (float)(sc >> 4);
        return dl * (float)((q >> (2 * j)) & 3) - ml;
    }
};
struct DqQ3K {
    static const int QK = 256;
    static const int BS = 110;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const int n = e >> 7;
        const int r = e & 127;
        const int sh = r >> 5;
        const int r2 = r & 31;
        const int si = r2 >> 4;
        const int i = r2 & 15;
        const uint8_t qb = b[32 + 32 * n + i + 16 * si];
        const uint8_t hb = b[i + 16 * si];
        const int m = 1 << (4 * n + sh);
        const int is = 8 * n + 2 * sh + si;
        const float d = rd_f16(b + 108);
        const float dl = d * ((float)q3k_scale(b + 96, is) - 32.0f);
        const int v = (int)((qb >> (2 * sh)) & 3) - ((hb & m) ? 0 : 4);
        return dl * (float)v;
    }
};
struct DqQ4K {
    static const int QK = 256;
    static const int BS = 144;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const int jj = e >> 6;
        const int r = e & 63;
        const int hi = r >> 5;
        const int l = r & 31;
        const uint8_t q = b[16 + 32 * jj + l];
        const int x = hi ? (q >> 4) : (q & 0xF);
        uint8_t sc, m;
        scale_min_k4(2 * jj + hi, b + 4, sc, m);
        const float d = rd_f16(b);
        const float dmin = rd_f16(b + 2);
        return (d * (float)sc) * (float)x - dmin * (float)m;
    }
};
struct DqQ5K {
    static const int QK = 256;
    static const int BS = 176;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const int jj = e >> 6;
        const int r = e & 63;
        const int hi = r >> 5;
        const int l = r & 31;
        const uint8_t q = b[48 + 32 * jj + l];
        float x = (float)(hi ? (q >> 4) : (q & 0xF));
        if (b[16 + l] & (1 << (2 * jj + hi))) x += 16.0f;
        uint8_t sc, m;
        scale_min_k4(2 * jj + hi, b + 4, sc, m);
        const float d = rd_f16(b);
        const float dmin = rd_f16(b + 2);
        return (d * (float)sc) * x - dmin * (float)m;
    }
};
struct DqQ6K {
    static const int QK = 256;
    static const int BS = 210;
    static DEVFN float deq(const uint8_t *b, const int e) {
        const int idx = e >> 7;
        const int r = e & 127;
        const int quarter = r >> 5;
        const int l = r & 31;
        const int is = l >> 4;
        const uint8_t qlb = b[64 * idx + l + (quarter & 1) * 32];
        const uint8_t qhb = b[128 + 32 * idx + l];
        const int low = quarter < 2 ? (qlb & 0xF) : (qlb >> 4);
        const int hi2 = (qhb >> (2 * quarter)) & 3;
        const int q = (low | (hi2 << 4)) - 32;
        const int sc = (int8_t)b[192 + 8 * idx + is + 2 * quarter];
        const float d = rd_f16(b + 208);
        return (d * (float)sc) * (float)q;
    }
};
struct DqQ8K {
    static const int QK = 256;
    static const int BS = 292;
    static DEVFN float deq(const uint8_t *b, const int e) { return rd_f32(b) * (float)(int8_t)b[4 + e]; }
};

template <typename D>
DEVFN float deq_at(const uint8_t *data, const size_t i) {
    return D::deq(data + (i / D::QK) * D::BS, (int)(i % D::QK));
}

template <typename D>
DEVFN void dequantize_f32_impl(const uint8_t *data, float *out, const size_t numel) {
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        out[i] = deq_at<D>(data, i);
    }
}

template <typename D>
DEVFN void dequantize_f16_impl(const uint8_t *data, f16_t *out, const size_t numel) {
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        out[i] = Ty<f16_t>::from(deq_at<D>(data, i));
    }
}

template <typename D>
DEVFN void qmv_impl(const uint8_t *w, const float *x, float *y, const size_t ncols, const size_t nrows,
                    const size_t nbatch) {
    HIP_DYNAMIC_SHARED(unsigned char, shr_raw)
    float *shr = reinterpret_cast<float *>(shr_raw);
    const size_t row_bytes = (ncols / D::QK) * D::BS;
    for (size_t bi = blockIdx.y; bi < nbatch; bi += gridDim.y) {
        for (size_t row = blockIdx.x; row < nrows; row += gridDim.x) {
            const uint8_t *wrow = w + row * row_bytes;
            const float *xr = x + bi * ncols;
            float acc = 0.0f;
            for (size_t k = threadIdx.x; k < ncols; k += blockDim.x) {
                acc += deq_at<D>(wrow, k) * xr[k];
            }
            shr[threadIdx.x] = acc;
            __syncthreads();
            for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) shr[threadIdx.x] += shr[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) y[bi * nrows + row] = shr[0];
            __syncthreads();
        }
    }
}

template <typename D>
DEVFN void qembed_impl(const uint8_t *w, const uint32_t *ids, float *out, const size_t n_ids, const size_t hidden,
                       const size_t rows) {
    const size_t numel = n_ids * hidden;
    const size_t row_bytes = (hidden / D::QK) * D::BS;
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        const size_t id_i = i / hidden;
        const size_t c = i % hidden;
        const uint32_t row = ids[id_i];
        out[i] = row < rows ? deq_at<D>(w + (size_t)row * row_bytes, c) : 0.0f;
    }
}

#define QUANT_FNS(D, SFX)                                                                                      \
    extern "C" __global__ void dq_##SFX##_f32(const uint8_t *data, float *out, const size_t numel) {           \
        dequantize_f32_impl<D>(data, out, numel);                                                              \
    }                                                                                                          \
    CANDLE_REGISTER(dq_##SFX##_f32)                                                                            \
    extern "C" __global__ void dq_##SFX##_f16(const uint8_t *data, f16_t *out, const size_t numel) {           \
        dequantize_f16_impl<D>(data, out, numel);                                                              \
    }                                                                                                          \
    CANDLE_REGISTER(dq_##SFX##_f16)                                                                            \
    extern "C" __global__ void qmv_##SFX(const uint8_t *w, const float *x, float *y, const size_t ncols,       \
                                         const size_t nrows, const size_t nbatch) {                            \
        qmv_impl<D>(w, x, y, ncols, nrows, nbatch);                                                            \
    }                                                                                                          \
    CANDLE_REGISTER(qmv_##SFX)                                                                                 \
    extern "C" __global__ void qembed_##SFX(const uint8_t *w, const uint32_t *ids, float *out,                 \
                                            const size_t n_ids, const size_t hidden, const size_t rows) {      \
        qembed_impl<D>(w, ids, out, n_ids, hidden, rows);                                                      \
    }                                                                                                          \
    CANDLE_REGISTER(qembed_##SFX)

QUANT_FNS(DqF32, f32)
QUANT_FNS(DqF16, f16)
QUANT_FNS(DqBF16, bf16)
QUANT_FNS(DqQ4_0, q4_0)
QUANT_FNS(DqQ4_1, q4_1)
QUANT_FNS(DqQ5_0, q5_0)
QUANT_FNS(DqQ5_1, q5_1)
QUANT_FNS(DqQ8_0, q8_0)
QUANT_FNS(DqQ8_1, q8_1)
QUANT_FNS(DqQ2K, q2k)
QUANT_FNS(DqQ3K, q3k)
QUANT_FNS(DqQ4K, q4k)
QUANT_FNS(DqQ5K, q5k)
QUANT_FNS(DqQ6K, q6k)
QUANT_FNS(DqQ8K, q8k)
