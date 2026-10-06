#include "common.h"

template <typename T, typename O, typename F>
DEVFN void binary_map(const size_t numel, const SI2 &info, const T *lhs, const T *rhs, O *out, F f) {
    if (info.ndim == 0) {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[i] = f(Ty<T>::to(lhs[i]), Ty<T>::to(rhs[i]));
        }
    } else {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            const size_t lhs_i = get_strided_index(i, info.ndim, info.dims, info.s0);
            const size_t rhs_i = get_strided_index(i, info.ndim, info.dims, info.s1);
            out[i] = f(Ty<T>::to(lhs[lhs_i]), Ty<T>::to(rhs[rhs_i]));
        }
    }
}

template <typename T> struct BAdd { DEVFN T operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return Ty<T>::from(a + b); } };
template <typename T> struct BSub { DEVFN T operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return Ty<T>::from(a - b); } };
template <typename T> struct BMul { DEVFN T operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return Ty<T>::from(a * b); } };
template <typename T> struct BDiv { DEVFN T operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return Ty<T>::from(a / b); } };
template <typename T> struct BMin { DEVFN T operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return Ty<T>::from(a > b ? b : a); } };
template <typename T> struct BMax { DEVFN T operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return Ty<T>::from(a < b ? b : a); } };

template <typename T> struct CEq { DEVFN uint8_t operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return a == b; } };
template <typename T> struct CNe { DEVFN uint8_t operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return a != b; } };
template <typename T> struct CLt { DEVFN uint8_t operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return a < b; } };
template <typename T> struct CLe { DEVFN uint8_t operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return a <= b; } };
template <typename T> struct CGt { DEVFN uint8_t operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return a > b; } };
template <typename T> struct CGe { DEVFN uint8_t operator()(typename Ty<T>::acc a, typename Ty<T>::acc b) const { return a >= b; } };

#define BINARY_FN(NAME, OPSTRUCT, T, SFX)                                                                  \
    extern "C" __global__ void NAME##_##SFX(const size_t numel, const SI2 info, const T *lhs, const T *rhs, \
                                            T *out) {                                                       \
        binary_map<T, T>(numel, info, lhs, rhs, out, OPSTRUCT<T>());                                        \
    }                                                                                                       \
    CANDLE_REGISTER(NAME##_##SFX)

#define CMP_FN(NAME, OPSTRUCT, T, SFX)                                                                     \
    extern "C" __global__ void NAME##_##SFX(const size_t numel, const SI2 info, const T *lhs, const T *rhs, \
                                            uint8_t *out) {                                                 \
        binary_map<T, uint8_t>(numel, info, lhs, rhs, out, OPSTRUCT<T>());                                  \
    }                                                                                                       \
    CANDLE_REGISTER(NAME##_##SFX)

#define BINARY_ALL(T, SFX)              \
    BINARY_FN(badd, BAdd, T, SFX)       \
    BINARY_FN(bsub, BSub, T, SFX)       \
    BINARY_FN(bmul, BMul, T, SFX)       \
    BINARY_FN(bdiv, BDiv, T, SFX)       \
    BINARY_FN(bminimum, BMin, T, SFX)   \
    BINARY_FN(bmaximum, BMax, T, SFX)   \
    CMP_FN(eq, CEq, T, SFX)             \
    CMP_FN(ne, CNe, T, SFX)             \
    CMP_FN(lt, CLt, T, SFX)             \
    CMP_FN(le, CLe, T, SFX)             \
    CMP_FN(gt, CGt, T, SFX)             \
    CMP_FN(ge, CGe, T, SFX)

BINARY_ALL(float, f32)
BINARY_ALL(double, f64)
BINARY_ALL(f16_t, f16)
BINARY_ALL(bf16_t, bf16)
BINARY_ALL(uint8_t, u8)
BINARY_ALL(uint32_t, u32)
BINARY_ALL(int16_t, i16)
BINARY_ALL(int32_t, i32)
BINARY_ALL(int64_t, i64)
