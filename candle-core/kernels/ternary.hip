#include "common.h"

template <typename C, typename T>
DEVFN void where_impl(const size_t numel, const SI3 &info, const C *ids, const T *t, const T *f, T *out) {
    if (info.ndim == 0) {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[i] = ids[i] ? t[i] : f[i];
        }
    } else {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            const size_t i_c = get_strided_index(i, info.ndim, info.dims, info.s0);
            const size_t i_t = get_strided_index(i, info.ndim, info.dims, info.s1);
            const size_t i_f = get_strided_index(i, info.ndim, info.dims, info.s2);
            out[i] = ids[i_c] ? t[i_t] : f[i_f];
        }
    }
}

#define WHERE_FN(C, CS, T, TS)                                                                                \
    extern "C" __global__ void where_##CS##_##TS(const size_t numel, const SI3 info, const C *ids, const T *t, \
                                                 const T *f, T *out) {                                        \
        where_impl<C, T>(numel, info, ids, t, f, out);                                                        \
    }                                                                                                         \
    CANDLE_REGISTER(where_##CS##_##TS)

#define WHERE_ALL(C, CS)              \
    WHERE_FN(C, CS, uint8_t, b1)      \
    WHERE_FN(C, CS, uint16_t, b2)     \
    WHERE_FN(C, CS, uint32_t, b4)     \
    WHERE_FN(C, CS, uint64_t, b8)

WHERE_ALL(uint8_t, u8)
WHERE_ALL(uint32_t, u32)
WHERE_ALL(int64_t, i64)
