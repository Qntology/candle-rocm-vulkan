#include "common.h"

template <typename T>
DEVFN void const_set_impl(const size_t numel, const SI1 &info, const T value, T *out) {
    if (info.ndim == 0) {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[i] = value;
        }
    } else {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[get_strided_index(i, info.ndim, info.dims, info.s0)] = value;
        }
    }
}

template <typename T>
DEVFN void copy_strided_impl(const size_t numel, const SI1 &info, const T *src, T *dst) {
    if (info.ndim == 0) {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            dst[i] = src[i];
        }
    } else {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            dst[i] = src[get_strided_index(i, info.ndim, info.dims, info.s0)];
        }
    }
}

template <typename T>
DEVFN void copy2d_impl(const T *src, T *dst, const size_t d1, const size_t d2, const size_t src_stride1,
                       const size_t dst_stride1) {
    const size_t numel = d1 * d2;
    for (size_t i = GS_START; i < numel; i += GS_STEP) {
        const size_t r = i / d2;
        const size_t c = i - r * d2;
        dst[r * dst_stride1 + c] = src[r * src_stride1 + c];
    }
}

#define FILL_FNS(T, SFX)                                                                                       \
    extern "C" __global__ void const_set_##SFX(const size_t numel, const SI1 info, const T value, T *out) {     \
        const_set_impl<T>(numel, info, value, out);                                                            \
    }                                                                                                          \
    CANDLE_REGISTER(const_set_##SFX)                                                                           \
    extern "C" __global__ void copy_strided_##SFX(const size_t numel, const SI1 info, const T *src, T *dst) {  \
        copy_strided_impl<T>(numel, info, src, dst);                                                           \
    }                                                                                                          \
    CANDLE_REGISTER(copy_strided_##SFX)                                                                        \
    extern "C" __global__ void copy2d_##SFX(const T *src, T *dst, const size_t d1, const size_t d2,            \
                                            const size_t src_stride1, const size_t dst_stride1) {              \
        copy2d_impl<T>(src, dst, d1, d2, src_stride1, dst_stride1);                                            \
    }                                                                                                          \
    CANDLE_REGISTER(copy2d_##SFX)

FILL_FNS(uint8_t, b1)
FILL_FNS(uint16_t, b2)
FILL_FNS(uint32_t, b4)
FILL_FNS(uint64_t, b8)
