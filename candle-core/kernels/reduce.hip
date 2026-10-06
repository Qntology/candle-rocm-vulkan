#include "common.h"

template <typename A> struct SumOp {
    static DEVFN A init() { return (A)0; }
    static DEVFN A combine(A a, A b) { return a + b; }
};
template <typename A> struct MaxOp {
    static DEVFN A init() { return Limits<A>::lowest(); }
    static DEVFN A combine(A a, A b) { return (b > a || isnang_any(b)) ? b : a; }
    static DEVFN bool isnang_any(A v) { return v != v; }
};
template <typename A> struct MinOp {
    static DEVFN A init() { return Limits<A>::highest(); }
    static DEVFN A combine(A a, A b) { return (b < a || isnang_any(b)) ? b : a; }
    static DEVFN bool isnang_any(A v) { return v != v; }
};

template <typename T> struct SumAcc { typedef typename Ty<T>::acc type; };
template <> struct SumAcc<uint8_t> { typedef uint64_t type; };
template <> struct SumAcc<uint32_t> { typedef uint64_t type; };
template <> struct SumAcc<int16_t> { typedef int64_t type; };
template <> struct SumAcc<int32_t> { typedef int64_t type; };

DEVFN size_t reduce_src_index(const size_t flat, const SI1 &info) {
    return info.ndim == 0 ? flat : get_strided_index(flat, info.ndim, info.dims, info.s0);
}

template <typename T, typename A, typename Op>
DEVFN void reduce_block(const T *inp, T *out, const SI1 &info, const size_t out_numel, const size_t reduce_size) {
    HIP_DYNAMIC_SHARED(unsigned char, shr_raw)
    A *shr = reinterpret_cast<A *>(shr_raw);
    const size_t tid = threadIdx.x;
    for (size_t o = blockIdx.x; o < out_numel; o += gridDim.x) {
        A acc = Op::init();
        for (size_t r = tid; r < reduce_size; r += blockDim.x) {
            const size_t src_i = reduce_src_index(o * reduce_size + r, info);
            acc = Op::combine(acc, (A)Ty<T>::to(inp[src_i]));
        }
        shr[tid] = acc;
        __syncthreads();
        for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shr[tid] = Op::combine(shr[tid], shr[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[o] = Ty<T>::from((typename Ty<T>::acc)shr[0]);
        }
        __syncthreads();
    }
}

template <typename T, typename A, typename Op>
DEVFN void reduce_small(const T *inp, T *out, const SI1 &info, const size_t out_numel, const size_t reduce_size) {
    for (size_t o = GS_START; o < out_numel; o += GS_STEP) {
        A acc = Op::init();
        for (size_t r = 0; r < reduce_size; ++r) {
            const size_t src_i = reduce_src_index(o * reduce_size + r, info);
            acc = Op::combine(acc, (A)Ty<T>::to(inp[src_i]));
        }
        out[o] = Ty<T>::from((typename Ty<T>::acc)acc);
    }
}

template <typename A, bool IS_MAX> DEVFN bool arg_better(A v, uint32_t i, A best, uint32_t best_i) {
    if (IS_MAX) {
        if (v > best) return true;
    } else {
        if (v < best) return true;
    }
    return v == best && i < best_i;
}

template <typename T, bool IS_MAX>
DEVFN void arg_reduce_block(const T *inp, uint32_t *out, const SI1 &info, const size_t out_numel, const size_t reduce_size) {
    typedef typename Ty<T>::acc A;
    HIP_DYNAMIC_SHARED(unsigned char, shr_raw)
    A *shr_val = reinterpret_cast<A *>(shr_raw);
    uint32_t *shr_idx = reinterpret_cast<uint32_t *>(shr_raw + blockDim.x * sizeof(A));
    const size_t tid = threadIdx.x;
    for (size_t o = blockIdx.x; o < out_numel; o += gridDim.x) {
        A best = IS_MAX ? Limits<A>::lowest() : Limits<A>::highest();
        uint32_t best_i = 0xFFFFFFFFu;
        for (size_t r = tid; r < reduce_size; r += blockDim.x) {
            const size_t src_i = reduce_src_index(o * reduce_size + r, info);
            const A v = Ty<T>::to(inp[src_i]);
            if (best_i == 0xFFFFFFFFu || arg_better<A, IS_MAX>(v, (uint32_t)r, best, best_i)) {
                best = v;
                best_i = (uint32_t)r;
            }
        }
        shr_val[tid] = best;
        shr_idx[tid] = best_i;
        __syncthreads();
        for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const uint32_t oi = shr_idx[tid + s];
                if (oi != 0xFFFFFFFFu &&
                    (shr_idx[tid] == 0xFFFFFFFFu || arg_better<A, IS_MAX>(shr_val[tid + s], oi, shr_val[tid], shr_idx[tid]))) {
                    shr_val[tid] = shr_val[tid + s];
                    shr_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[o] = shr_idx[0] == 0xFFFFFFFFu ? 0u : shr_idx[0];
        }
        __syncthreads();
    }
}

template <typename T, bool IS_MAX>
DEVFN void arg_reduce_small(const T *inp, uint32_t *out, const SI1 &info, const size_t out_numel, const size_t reduce_size) {
    typedef typename Ty<T>::acc A;
    for (size_t o = GS_START; o < out_numel; o += GS_STEP) {
        A best = IS_MAX ? Limits<A>::lowest() : Limits<A>::highest();
        uint32_t best_i = 0xFFFFFFFFu;
        for (size_t r = 0; r < reduce_size; ++r) {
            const size_t src_i = reduce_src_index(o * reduce_size + r, info);
            const A v = Ty<T>::to(inp[src_i]);
            if (best_i == 0xFFFFFFFFu || arg_better<A, IS_MAX>(v, (uint32_t)r, best, best_i)) {
                best = v;
                best_i = (uint32_t)r;
            }
        }
        out[o] = best_i == 0xFFFFFFFFu ? 0u : best_i;
    }
}

#define REDUCE_FNS(T, SFX)                                                                                     \
    extern "C" __global__ void rsum_##SFX(const T *inp, T *out, const SI1 info, const size_t out_numel,        \
                                          const size_t reduce_size) {                                          \
        reduce_block<T, SumAcc<T>::type, SumOp<SumAcc<T>::type> >(inp, out, info, out_numel, reduce_size);     \
    }                                                                                                          \
    CANDLE_REGISTER(rsum_##SFX)                                                                                \
    extern "C" __global__ void rmax_##SFX(const T *inp, T *out, const SI1 info, const size_t out_numel,        \
                                          const size_t reduce_size) {                                          \
        reduce_block<T, Ty<T>::acc, MaxOp<Ty<T>::acc> >(inp, out, info, out_numel, reduce_size);               \
    }                                                                                                          \
    CANDLE_REGISTER(rmax_##SFX)                                                                                \
    extern "C" __global__ void rmin_##SFX(const T *inp, T *out, const SI1 info, const size_t out_numel,        \
                                          const size_t reduce_size) {                                          \
        reduce_block<T, Ty<T>::acc, MinOp<Ty<T>::acc> >(inp, out, info, out_numel, reduce_size);               \
    }                                                                                                          \
    CANDLE_REGISTER(rmin_##SFX)                                                                                \
    extern "C" __global__ void rargmax_##SFX(const T *inp, uint32_t *out, const SI1 info,                      \
                                             const size_t out_numel, const size_t reduce_size) {               \
        arg_reduce_block<T, true>(inp, out, info, out_numel, reduce_size);                                     \
    }                                                                                                          \
    CANDLE_REGISTER(rargmax_##SFX)                                                                             \
    extern "C" __global__ void rargmin_##SFX(const T *inp, uint32_t *out, const SI1 info,                      \
                                             const size_t out_numel, const size_t reduce_size) {               \
        arg_reduce_block<T, false>(inp, out, info, out_numel, reduce_size);                                    \
    }                                                                                                          \
    CANDLE_REGISTER(rargmin_##SFX)                                                                             \
    extern "C" __global__ void rsum_small_##SFX(const T *inp, T *out, const SI1 info, const size_t out_numel,  \
                                                const size_t reduce_size) {                                    \
        reduce_small<T, SumAcc<T>::type, SumOp<SumAcc<T>::type> >(inp, out, info, out_numel, reduce_size);     \
    }                                                                                                          \
    CANDLE_REGISTER(rsum_small_##SFX)                                                                          \
    extern "C" __global__ void rmax_small_##SFX(const T *inp, T *out, const SI1 info, const size_t out_numel,  \
                                                const size_t reduce_size) {                                    \
        reduce_small<T, Ty<T>::acc, MaxOp<Ty<T>::acc> >(inp, out, info, out_numel, reduce_size);               \
    }                                                                                                          \
    CANDLE_REGISTER(rmax_small_##SFX)                                                                          \
    extern "C" __global__ void rmin_small_##SFX(const T *inp, T *out, const SI1 info, const size_t out_numel,  \
                                                const size_t reduce_size) {                                    \
        reduce_small<T, Ty<T>::acc, MinOp<Ty<T>::acc> >(inp, out, info, out_numel, reduce_size);               \
    }                                                                                                          \
    CANDLE_REGISTER(rmin_small_##SFX)                                                                          \
    extern "C" __global__ void rargmax_small_##SFX(const T *inp, uint32_t *out, const SI1 info,                \
                                                   const size_t out_numel, const size_t reduce_size) {         \
        arg_reduce_small<T, true>(inp, out, info, out_numel, reduce_size);                                     \
    }                                                                                                          \
    CANDLE_REGISTER(rargmax_small_##SFX)                                                                       \
    extern "C" __global__ void rargmin_small_##SFX(const T *inp, uint32_t *out, const SI1 info,                \
                                                   const size_t out_numel, const size_t reduce_size) {         \
        arg_reduce_small<T, false>(inp, out, info, out_numel, reduce_size);                                    \
    }                                                                                                          \
    CANDLE_REGISTER(rargmin_small_##SFX)

REDUCE_FNS(float, f32)
REDUCE_FNS(double, f64)
REDUCE_FNS(f16_t, f16)
REDUCE_FNS(bf16_t, bf16)
REDUCE_FNS(uint8_t, u8)
REDUCE_FNS(uint32_t, u32)
REDUCE_FNS(int16_t, i16)
REDUCE_FNS(int32_t, i32)
REDUCE_FNS(int64_t, i64)
