#include "common.h"

template <typename T, typename I>
DEVFN void index_select_impl(const size_t numel, const SI1 &info, const I *ids, const T *inp, T *out,
                             const size_t left_size, const size_t src_dim_size, const size_t ids_dim_size,
                             const size_t right_size) {
    for (size_t dst_i = GS_START; dst_i < numel; dst_i += GS_STEP) {
        const size_t left_i = dst_i / (ids_dim_size * right_size);
        const size_t id_i = dst_i / right_size % ids_dim_size;
        const size_t right_i = dst_i % right_size;
        const I idx = ids[id_i];
        if (idx == Limits<I>::highest() || (size_t)idx >= src_dim_size) {
            T zero;
            __builtin_memset(&zero, 0, sizeof(T));
            out[dst_i] = zero;
        } else {
            const size_t src_i = left_i * (src_dim_size * right_size) + (size_t)idx * right_size + right_i;
            const size_t strided_i =
                info.ndim == 0 ? src_i : get_strided_index(src_i, info.ndim, info.dims, info.s0);
            out[dst_i] = inp[strided_i];
        }
    }
}

template <typename T, typename I>
DEVFN void gather_impl(const size_t numel, const I *ids, const T *inp, T *out, const size_t left_size,
                       const size_t src_dim_size, const size_t ids_dim_size, const size_t right_size) {
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        const size_t post = i % right_size;
        const I idx = ids[i];
        if (idx == Limits<I>::highest() || (size_t)idx >= src_dim_size) {
            T zero;
            __builtin_memset(&zero, 0, sizeof(T));
            out[i] = zero;
        } else {
            const size_t pre = i / (right_size * ids_dim_size);
            const size_t src_i = (pre * src_dim_size + (size_t)idx) * right_size + post;
            out[i] = inp[src_i];
        }
    }
}

template <typename T, typename I>
DEVFN void scatter_set_impl(const I *ids, const T *inp, T *out, const size_t left_size, const size_t src_dim_size,
                            const size_t dst_dim_size, const size_t right_size) {
    const size_t numel = left_size * right_size;
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        const size_t pre = i / right_size;
        const size_t post = i % right_size;
        for (size_t j = 0; j < src_dim_size; ++j) {
            const size_t src_i = (pre * src_dim_size + j) * right_size + post;
            const I idx = ids[src_i];
            if (idx != Limits<I>::highest() && (size_t)idx < dst_dim_size) {
                const size_t dst_i = (pre * dst_dim_size + (size_t)idx) * right_size + post;
                out[dst_i] = inp[src_i];
            }
        }
    }
}

template <typename T, typename I>
DEVFN void scatter_add_impl(const I *ids, const T *inp, T *out, const size_t left_size, const size_t src_dim_size,
                            const size_t dst_dim_size, const size_t right_size) {
    const size_t numel = left_size * right_size;
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        const size_t pre = i / right_size;
        const size_t post = i % right_size;
        for (size_t j = 0; j < src_dim_size; ++j) {
            const size_t src_i = (pre * src_dim_size + j) * right_size + post;
            const I idx = ids[src_i];
            if (idx != Limits<I>::highest() && (size_t)idx < dst_dim_size) {
                const size_t dst_i = (pre * dst_dim_size + (size_t)idx) * right_size + post;
                out[dst_i] = Ty<T>::from(Ty<T>::to(out[dst_i]) + Ty<T>::to(inp[src_i]));
            }
        }
    }
}

template <typename T, typename I>
DEVFN void index_add_impl(const I *ids, const size_t ids_dim_size, const T *inp, T *out, const size_t left_size,
                          const size_t src_dim_size, const size_t dst_dim_size, const size_t right_size) {
    const size_t numel = left_size * right_size;
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        const size_t pre = i / right_size;
        const size_t post = i % right_size;
        for (size_t j = 0; j < ids_dim_size; ++j) {
            const I idx = ids[j];
            if (idx != Limits<I>::highest() && (size_t)idx < dst_dim_size) {
                const size_t src_i = (pre * src_dim_size + j) * right_size + post;
                const size_t dst_i = (pre * dst_dim_size + (size_t)idx) * right_size + post;
                out[dst_i] = Ty<T>::from(Ty<T>::to(out[dst_i]) + Ty<T>::to(inp[src_i]));
            }
        }
    }
}

#define MOVE_FNS(T, TS, I, IS)                                                                                 \
    extern "C" __global__ void is_##IS##_##TS(const size_t numel, const SI1 info, const I *ids, const T *inp,  \
                                              T *out, const size_t left_size, const size_t src_dim_size,       \
                                              const size_t ids_dim_size, const size_t right_size) {            \
        index_select_impl<T, I>(numel, info, ids, inp, out, left_size, src_dim_size, ids_dim_size,             \
                                right_size);                                                                   \
    }                                                                                                          \
    CANDLE_REGISTER(is_##IS##_##TS)                                                                            \
    extern "C" __global__ void gather_##IS##_##TS(const size_t numel, const I *ids, const T *inp, T *out,      \
                                                  const size_t left_size, const size_t src_dim_size,           \
                                                  const size_t ids_dim_size, const size_t right_size) {        \
        gather_impl<T, I>(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size);            \
    }                                                                                                          \
    CANDLE_REGISTER(gather_##IS##_##TS)                                                                        \
    extern "C" __global__ void s_##IS##_##TS(const I *ids, const T *inp, T *out, const size_t left_size,       \
                                             const size_t src_dim_size, const size_t dst_dim_size,             \
                                             const size_t right_size) {                                        \
        scatter_set_impl<T, I>(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size);              \
    }                                                                                                          \
    CANDLE_REGISTER(s_##IS##_##TS)

#define ADD_FNS(T, TS, I, IS)                                                                                  \
    extern "C" __global__ void sa_##IS##_##TS(const I *ids, const T *inp, T *out, const size_t left_size,      \
                                              const size_t src_dim_size, const size_t dst_dim_size,            \
                                              const size_t right_size) {                                       \
        scatter_add_impl<T, I>(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size);              \
    }                                                                                                          \
    CANDLE_REGISTER(sa_##IS##_##TS)                                                                            \
    extern "C" __global__ void ia_##IS##_##TS(const I *ids, const size_t ids_dim_size, const T *inp, T *out,   \
                                              const size_t left_size, const size_t src_dim_size,               \
                                              const size_t dst_dim_size, const size_t right_size) {            \
        index_add_impl<T, I>(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size);  \
    }                                                                                                          \
    CANDLE_REGISTER(ia_##IS##_##TS)

#define MOVE_ALL_IDS(T, TS)              \
    MOVE_FNS(T, TS, uint8_t, u8)         \
    MOVE_FNS(T, TS, uint32_t, u32)       \
    MOVE_FNS(T, TS, int64_t, i64)

#define ADD_ALL_IDS(T, TS)               \
    ADD_FNS(T, TS, uint8_t, u8)          \
    ADD_FNS(T, TS, uint32_t, u32)        \
    ADD_FNS(T, TS, int64_t, i64)

MOVE_ALL_IDS(uint8_t, b1)
MOVE_ALL_IDS(uint16_t, b2)
MOVE_ALL_IDS(uint32_t, b4)
MOVE_ALL_IDS(uint64_t, b8)

ADD_ALL_IDS(float, f32)
ADD_ALL_IDS(double, f64)
ADD_ALL_IDS(f16_t, f16)
ADD_ALL_IDS(bf16_t, bf16)
ADD_ALL_IDS(uint8_t, u8)
ADD_ALL_IDS(uint32_t, u32)
ADD_ALL_IDS(int16_t, i16)
ADD_ALL_IDS(int32_t, i32)
ADD_ALL_IDS(int64_t, i64)
