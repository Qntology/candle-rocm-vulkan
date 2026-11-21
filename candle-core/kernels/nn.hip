#include "common.h"

template <typename A>
DEVFN A block_reduce_sum(A v, A *shr) {
    const size_t tid = threadIdx.x;
    shr[tid] = v;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    const A r = shr[0];
    __syncthreads();
    return r;
}

template <typename A>
DEVFN A block_reduce_max(A v, A *shr) {
    const size_t tid = threadIdx.x;
    shr[tid] = v;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = shr[tid + s] > shr[tid] ? shr[tid + s] : shr[tid];
        __syncthreads();
    }
    const A r = shr[0];
    __syncthreads();
    return r;
}

template <typename T>
DEVFN void softmax_impl(const T *src, T *dst, const size_t nrows, const size_t ncols) {
    typedef typename Ty<T>::acc A;
    HIP_DYNAMIC_SHARED(unsigned char, shr_raw)
    A *shr = reinterpret_cast<A *>(shr_raw);
    for (size_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const T *x = src + row * ncols;
        T *y = dst + row * ncols;
        A m = Limits<A>::lowest();
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            const A v = Ty<T>::to(x[c]);
            m = v > m ? v : m;
        }
        m = block_reduce_max<A>(m, shr);
        A s = (A)0;
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            s += expg(Ty<T>::to(x[c]) - m);
        }
        s = block_reduce_sum<A>(s, shr);
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            y[c] = Ty<T>::from(expg(Ty<T>::to(x[c]) - m) / s);
        }
    }
}

template <typename T>
DEVFN void rmsnorm_impl(const T *src, T *dst, const T *alpha, const size_t nrows, const size_t ncols,
                        const float eps) {
    typedef typename Ty<T>::acc A;
    HIP_DYNAMIC_SHARED(unsigned char, shr_raw)
    A *shr = reinterpret_cast<A *>(shr_raw);
    for (size_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const T *x = src + row * ncols;
        T *y = dst + row * ncols;
        A s2 = (A)0;
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            const A v = Ty<T>::to(x[c]);
            s2 += v * v;
        }
        s2 = block_reduce_sum<A>(s2, shr);
        const A m = sqrtg(s2 / (A)ncols + (A)eps);
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            y[c] = Ty<T>::from(Ty<T>::to(x[c]) / m * Ty<T>::to(alpha[c]));
        }
    }
}

template <typename T>
DEVFN void layernorm_impl(const T *src, T *dst, const T *alpha, const T *beta, const size_t nrows,
                          const size_t ncols, const float eps) {
    typedef typename Ty<T>::acc A;
    HIP_DYNAMIC_SHARED(unsigned char, shr_raw)
    A *shr = reinterpret_cast<A *>(shr_raw);
    for (size_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const T *x = src + row * ncols;
        T *y = dst + row * ncols;
        A s = (A)0;
        A s2 = (A)0;
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            const A v = Ty<T>::to(x[c]);
            s += v;
            s2 += v * v;
        }
        s = block_reduce_sum<A>(s, shr);
        s2 = block_reduce_sum<A>(s2, shr);
        const A mean = s / (A)ncols;
        const A var = s2 / (A)ncols - mean * mean;
        const A inv_std = (A)1 / sqrtg(var + (A)eps);
        for (size_t c = threadIdx.x; c < ncols; c += blockDim.x) {
            const A v = (Ty<T>::to(x[c]) - mean) * inv_std * Ty<T>::to(alpha[c]) + Ty<T>::to(beta[c]);
            y[c] = Ty<T>::from(v);
        }
    }
}

template <typename T>
DEVFN void rope_i_impl(const T *src, const T *cos, const T *sin, T *dst, const size_t b, const size_t h,
                       const size_t t, const size_t d, const uint32_t unbatched) {
    typedef typename Ty<T>::acc A;
    const size_t half_td = t * d / 2;
    const size_t n = b * h * half_td;
    for (size_t p = GS_START; p < n; p += GS_STEP) {
        const size_t bh_i = p / half_td;
        const size_t rem = p % half_td;
        const size_t cs = rem + (unbatched ? (bh_i / h) * half_td : 0);
        const A x0 = Ty<T>::to(src[2 * p]);
        const A x1 = Ty<T>::to(src[2 * p + 1]);
        const A c = Ty<T>::to(cos[cs]);
        const A s = Ty<T>::to(sin[cs]);
        dst[2 * p] = Ty<T>::from(x0 * c - x1 * s);
        dst[2 * p + 1] = Ty<T>::from(x0 * s + x1 * c);
    }
}

template <typename T>
DEVFN void rope_impl(const T *src, const T *cos, const T *sin, T *dst, const size_t b, const size_t h,
                     const size_t t, const size_t d, const uint32_t unbatched) {
    typedef typename Ty<T>::acc A;
    const size_t half_d = d / 2;
    const size_t half_td = t * half_d;
    const size_t n = b * h * half_td;
    for (size_t p = GS_START; p < n; p += GS_STEP) {
        const size_t bh_i = p / half_td;
        const size_t rem = p % half_td;
        const size_t i_t = rem / half_d;
        const size_t i_d = rem % half_d;
        const size_t i1 = bh_i * t * d + i_t * d + i_d;
        const size_t i2 = i1 + half_d;
        const size_t cs = rem + (unbatched ? (bh_i / h) * half_td : 0);
        const A x1 = Ty<T>::to(src[i1]);
        const A x2 = Ty<T>::to(src[i2]);
        const A c = Ty<T>::to(cos[cs]);
        const A s = Ty<T>::to(sin[cs]);
        dst[i1] = Ty<T>::from(x1 * c - x2 * s);
        dst[i2] = Ty<T>::from(x1 * s + x2 * c);
    }
}

template <typename T>
DEVFN void rope_thd_impl(const T *src, const T *cos, const T *sin, T *dst, const size_t b, const size_t t,
                         const size_t h, const size_t d, const uint32_t unbatched) {
    typedef typename Ty<T>::acc A;
    const size_t half_d = d / 2;
    const size_t per_b = t * h * half_d;
    const size_t n = b * per_b;
    for (size_t p = GS_START; p < n; p += GS_STEP) {
        const size_t b_i = p / per_b;
        const size_t rem = p % per_b;
        const size_t i_t = rem / (h * half_d);
        const size_t rem2 = rem % (h * half_d);
        const size_t i_h = rem2 / half_d;
        const size_t i_d = rem2 % half_d;
        const size_t i1 = b_i * t * h * d + i_t * h * d + i_h * d + i_d;
        const size_t i2 = i1 + half_d;
        const size_t cs = i_t * half_d + i_d + (unbatched ? b_i * t * half_d : 0);
        const A x1 = Ty<T>::to(src[i1]);
        const A x2 = Ty<T>::to(src[i2]);
        const A c = Ty<T>::to(cos[cs]);
        const A s = Ty<T>::to(sin[cs]);
        dst[i1] = Ty<T>::from(x1 * c - x2 * s);
        dst[i2] = Ty<T>::from(x1 * s + x2 * c);
    }
}

template <typename T>
DEVFN void im2col_2d_impl(const T *src, T *dst, const size_t b, const size_t c_in, const size_t i_h, const size_t i_w,
                          const size_t k_h, const size_t k_w, const size_t o_h, const size_t o_w,
                          const size_t oh_start, const size_t oh_tile, const size_t padding, const size_t stride,
                          const size_t dilation) {
    typedef typename Ty<T>::acc A;
    const size_t col_rows = c_in * k_h * k_w;
    const size_t hw_tile = oh_tile * o_w;
    const size_t per_b = col_rows * hw_tile;
    const size_t n = b * per_b;
    for (size_t idx = GS_START; idx < n; idx += GS_STEP) {
        const size_t bi = idx / per_b;
        const size_t rem = idx % per_b;
        const size_t r = rem / hw_tile;
        const size_t q = rem % hw_tile;
        const size_t oh = oh_start + (q / o_w);
        const size_t ow = q % o_w;
        const size_t kw = r % k_w;
        const size_t r1 = r / k_w;
        const size_t kh = r1 % k_h;
        const size_t ic = r1 / k_h;
        const long long ih = (long long)(oh * stride + kh * dilation) - (long long)padding;
        const long long iw = (long long)(ow * stride + kw * dilation) - (long long)padding;
        if (ih >= 0 && ih < (long long)i_h && iw >= 0 && iw < (long long)i_w) {
            dst[idx] = src[((bi * c_in + ic) * i_h + (size_t)ih) * i_w + (size_t)iw];
        } else {
            dst[idx] = Ty<T>::from((A)0);
        }
    }
}

template <typename T>
DEVFN void depthwise_conv2d_impl(const T *src, const T *w, T *dst, const size_t b, const size_t c, const size_t i_h,
                                 const size_t i_w, const size_t k_h, const size_t k_w, const size_t o_h,
                                 const size_t o_w, const size_t padding, const size_t stride, const size_t dilation) {
    typedef typename Ty<T>::acc A;
    const size_t hw_out = o_h * o_w;
    const size_t n = b * c * hw_out;
    for (size_t idx = GS_START; idx < n; idx += GS_STEP) {
        const size_t ow = idx % o_w;
        const size_t t1 = idx / o_w;
        const size_t oh = t1 % o_h;
        const size_t t2 = t1 / o_h;
        const size_t ci = t2 % c;
        const size_t bi = t2 / c;
        const size_t src_base = (bi * c + ci) * i_h * i_w;
        const size_t w_base = ci * k_h * k_w;
        A acc = (A)0;
        for (size_t kh = 0; kh < k_h; ++kh) {
            const long long ih = (long long)(oh * stride + kh * dilation) - (long long)padding;
            if (ih < 0 || ih >= (long long)i_h) continue;
            const size_t row_off = src_base + (size_t)ih * i_w;
            const size_t w_row = w_base + kh * k_w;
            for (size_t kw = 0; kw < k_w; ++kw) {
                const long long iw = (long long)(ow * stride + kw * dilation) - (long long)padding;
                if (iw < 0 || iw >= (long long)i_w) continue;
                acc += Ty<T>::to(src[row_off + (size_t)iw]) * Ty<T>::to(w[w_row + kw]);
            }
        }
        dst[idx] = Ty<T>::from(acc);
    }
}

template <typename T>
DEVFN void col2im_2d_impl(const T *col, T *dst, const size_t b, const size_t c_out, const size_t i_h, const size_t i_w,
                          const size_t k_h, const size_t k_w, const size_t o_h, const size_t o_w, const size_t padding,
                          const size_t stride, const size_t dilation) {
    typedef typename Ty<T>::acc A;
    const size_t hw_in = i_h * i_w;
    const size_t kk = k_h * k_w;
    const size_t n = b * c_out * o_h * o_w;
    for (size_t idx = GS_START; idx < n; idx += GS_STEP) {
        const size_t ow = idx % o_w;
        const size_t t1 = idx / o_w;
        const size_t oh = t1 % o_h;
        const size_t t2 = t1 / o_h;
        const size_t co = t2 % c_out;
        const size_t bi = t2 / c_out;
        const size_t col_base = (bi * c_out + co) * kk * hw_in;
        A acc = (A)0;
        for (size_t kh = 0; kh < k_h; ++kh) {
            const long long h_num = (long long)(oh + padding) - (long long)(kh * dilation);
            if (h_num < 0 || (h_num % (long long)stride) != 0) continue;
            const size_t ih = (size_t)(h_num / (long long)stride);
            if (ih >= i_h) continue;
            for (size_t kw = 0; kw < k_w; ++kw) {
                const long long w_num = (long long)(ow + padding) - (long long)(kw * dilation);
                if (w_num < 0 || (w_num % (long long)stride) != 0) continue;
                const size_t iw = (size_t)(w_num / (long long)stride);
                if (iw >= i_w) continue;
                acc += Ty<T>::to(col[col_base + (kh * k_w + kw) * hw_in + ih * i_w + iw]);
            }
        }
        dst[idx] = Ty<T>::from(acc);
    }
}

template <typename T>
DEVFN void upsample_nearest2d_impl(const T *src, T *dst, const size_t b, const size_t c, const size_t i_h,
                                   const size_t i_w, const size_t o_h, const size_t o_w) {
    const double scale_h = (double)i_h / (double)o_h;
    const double scale_w = (double)i_w / (double)o_w;
    const size_t hw_out = o_h * o_w;
    const size_t hw_in = i_h * i_w;
    const size_t n = b * c * hw_out;
    for (size_t idx = GS_START; idx < n; idx += GS_STEP) {
        const size_t ow = idx % o_w;
        const size_t t1 = idx / o_w;
        const size_t oh = t1 % o_h;
        const size_t bc = t1 / o_h;
        size_t sh = (size_t)((double)oh * scale_h);
        if (sh >= i_h) sh = i_h - 1;
        size_t sw = (size_t)((double)ow * scale_w);
        if (sw >= i_w) sw = i_w - 1;
        dst[idx] = src[bc * hw_in + sh * i_w + sw];
    }
}

template <typename T>
DEVFN void upsample_bilinear2d_impl(const T *src, T *dst, const size_t b, const size_t c, const size_t i_h,
                                    const size_t i_w, const size_t o_h, const size_t o_w, const double scale_h,
                                    const double scale_w, const uint32_t align_corners) {
    typedef typename Ty<T>::acc A;
    const size_t hw_out = o_h * o_w;
    const size_t hw_in = i_h * i_w;
    const size_t n = b * c * hw_out;
    for (size_t idx = GS_START; idx < n; idx += GS_STEP) {
        const size_t ow = idx % o_w;
        const size_t t1 = idx / o_w;
        const size_t oh = t1 % o_h;
        const size_t bc = t1 / o_h;
        double src_h = align_corners ? (scale_h * (double)oh) : (scale_h * ((double)oh + 0.5) - 0.5);
        if (src_h < 0.0) src_h = 0.0;
        size_t h0 = (size_t)floor(src_h);
        if (h0 >= i_h) h0 = i_h - 1;
        const size_t h1 = (h0 + 1 < i_h) ? (h0 + 1) : (i_h - 1);
        double wh = src_h - (double)h0;
        if (wh < 0.0) wh = 0.0;
        if (wh > 1.0) wh = 1.0;

        double src_w = align_corners ? (scale_w * (double)ow) : (scale_w * ((double)ow + 0.5) - 0.5);
        if (src_w < 0.0) src_w = 0.0;
        size_t w0 = (size_t)floor(src_w);
        if (w0 >= i_w) w0 = i_w - 1;
        const size_t w1 = (w0 + 1 < i_w) ? (w0 + 1) : (i_w - 1);
        double ww = src_w - (double)w0;
        if (ww < 0.0) ww = 0.0;
        if (ww > 1.0) ww = 1.0;

        const size_t base = bc * hw_in;
        const double v00 = (double)Ty<T>::to(src[base + h0 * i_w + w0]);
        const double v10 = (double)Ty<T>::to(src[base + h0 * i_w + w1]);
        const double v01 = (double)Ty<T>::to(src[base + h1 * i_w + w0]);
        const double v11 = (double)Ty<T>::to(src[base + h1 * i_w + w1]);
        const double v_top = v00 * (1.0 - ww) + v10 * ww;
        const double v_bot = v01 * (1.0 - ww) + v11 * ww;
        const double val = v_top * (1.0 - wh) + v_bot * wh;
        dst[idx] = Ty<T>::from((A)val);
    }
}

#define NN_FNS(T, SFX)                                                                                         \
    extern "C" __global__ void softmax_##SFX(const T *src, T *dst, const size_t nrows, const size_t ncols) {   \
        softmax_impl<T>(src, dst, nrows, ncols);                                                               \
    }                                                                                                          \
    CANDLE_REGISTER(softmax_##SFX)                                                                             \
    extern "C" __global__ void rmsnorm_##SFX(const T *src, T *dst, const T *alpha, const size_t nrows,        \
                                             const size_t ncols, const float eps) {                            \
        rmsnorm_impl<T>(src, dst, alpha, nrows, ncols, eps);                                                   \
    }                                                                                                          \
    CANDLE_REGISTER(rmsnorm_##SFX)                                                                             \
    extern "C" __global__ void layernorm_##SFX(const T *src, T *dst, const T *alpha, const T *beta,           \
                                               const size_t nrows, const size_t ncols, const float eps) {      \
        layernorm_impl<T>(src, dst, alpha, beta, nrows, ncols, eps);                                           \
    }                                                                                                          \
    CANDLE_REGISTER(layernorm_##SFX)                                                                           \
    extern "C" __global__ void rope_i_##SFX(const T *src, const T *cos, const T *sin, T *dst, const size_t b, \
                                            const size_t h, const size_t t, const size_t d,                    \
                                            const uint32_t unbatched) {                                        \
        rope_i_impl<T>(src, cos, sin, dst, b, h, t, d, unbatched);                                             \
    }                                                                                                          \
    CANDLE_REGISTER(rope_i_##SFX)                                                                              \
    extern "C" __global__ void rope_##SFX(const T *src, const T *cos, const T *sin, T *dst, const size_t b,   \
                                          const size_t h, const size_t t, const size_t d,                      \
                                          const uint32_t unbatched) {                                          \
        rope_impl<T>(src, cos, sin, dst, b, h, t, d, unbatched);                                               \
    }                                                                                                          \
    CANDLE_REGISTER(rope_##SFX)                                                                                \
    extern "C" __global__ void rope_thd_##SFX(const T *src, const T *cos, const T *sin, T *dst,               \
                                              const size_t b, const size_t t, const size_t h, const size_t d,  \
                                              const uint32_t unbatched) {                                      \
        rope_thd_impl<T>(src, cos, sin, dst, b, t, h, d, unbatched);                                           \
    }                                                                                                          \
    CANDLE_REGISTER(rope_thd_##SFX)                                                                            \
    extern "C" __global__ void im2col_2d_##SFX(const T *src, T *dst, const size_t b, const size_t c_in,        \
                                               const size_t i_h, const size_t i_w, const size_t k_h,           \
                                               const size_t k_w, const size_t o_h, const size_t o_w,           \
                                               const size_t oh_start, const size_t oh_tile,                    \
                                               const size_t padding, const size_t stride,                      \
                                               const size_t dilation) {                                        \
        im2col_2d_impl<T>(src, dst, b, c_in, i_h, i_w, k_h, k_w, o_h, o_w, oh_start, oh_tile, padding, stride, \
                          dilation);                                                                           \
    }                                                                                                          \
    CANDLE_REGISTER(im2col_2d_##SFX)                                                                           \
    extern "C" __global__ void depthwise_conv2d_##SFX(const T *src, const T *w, T *dst, const size_t b,        \
                                                      const size_t c, const size_t i_h, const size_t i_w,      \
                                                      const size_t k_h, const size_t k_w, const size_t o_h,    \
                                                      const size_t o_w, const size_t padding,                  \
                                                      const size_t stride, const size_t dilation) {            \
        depthwise_conv2d_impl<T>(src, w, dst, b, c, i_h, i_w, k_h, k_w, o_h, o_w, padding, stride, dilation); \
    }                                                                                                          \
    CANDLE_REGISTER(depthwise_conv2d_##SFX)                                                                    \
    extern "C" __global__ void col2im_2d_##SFX(const T *col, T *dst, const size_t b, const size_t c_out,       \
                                               const size_t i_h, const size_t i_w, const size_t k_h,           \
                                               const size_t k_w, const size_t o_h, const size_t o_w,           \
                                               const size_t padding, const size_t stride,                      \
                                               const size_t dilation) {                                        \
        col2im_2d_impl<T>(col, dst, b, c_out, i_h, i_w, k_h, k_w, o_h, o_w, padding, stride, dilation);        \
    }                                                                                                          \
    CANDLE_REGISTER(col2im_2d_##SFX)                                                                           \
    extern "C" __global__ void upsample_nearest2d_##SFX(const T *src, T *dst, const size_t b, const size_t c,  \
                                                        const size_t i_h, const size_t i_w, const size_t o_h,  \
                                                        const size_t o_w) {                                    \
        upsample_nearest2d_impl<T>(src, dst, b, c, i_h, i_w, o_h, o_w);                                        \
    }                                                                                                          \
    CANDLE_REGISTER(upsample_nearest2d_##SFX)                                                                  \
    extern "C" __global__ void upsample_bilinear2d_##SFX(const T *src, T *dst, const size_t b, const size_t c, \
                                                         const size_t i_h, const size_t i_w, const size_t o_h, \
                                                         const size_t o_w, const double scale_h,               \
                                                         const double scale_w, const uint32_t align_corners) { \
        upsample_bilinear2d_impl<T>(src, dst, b, c, i_h, i_w, o_h, o_w, scale_h, scale_w, align_corners);      \
    }                                                                                                          \
    CANDLE_REGISTER(upsample_bilinear2d_##SFX)

NN_FNS(float, f32)
NN_FNS(double, f64)
NN_FNS(f16_t, f16)
NN_FNS(bf16_t, bf16)
