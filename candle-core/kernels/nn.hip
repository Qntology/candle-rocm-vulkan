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
    CANDLE_REGISTER(rope_thd_##SFX)

NN_FNS(float, f32)
NN_FNS(double, f64)
NN_FNS(f16_t, f16)
NN_FNS(bf16_t, bf16)
