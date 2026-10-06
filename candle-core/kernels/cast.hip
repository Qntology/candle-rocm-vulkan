#include "common.h"

template <typename T> struct IsFloat { static const bool value = false; };
template <> struct IsFloat<float> { static const bool value = true; };
template <> struct IsFloat<double> { static const bool value = true; };
template <> struct IsFloat<f16_t> { static const bool value = true; };
template <> struct IsFloat<bf16_t> { static const bool value = true; };
template <> struct IsFloat<f8e4m3_t> { static const bool value = true; };

template <typename D> DEVFN D sat_from_double(double x) {
    if (x != x) return (D)0;
    if (x <= (double)Limits<D>::lowest()) return Limits<D>::lowest();
    if (x >= (double)Limits<D>::highest()) return Limits<D>::highest();
    return (D)x;
}

template <typename S> DEVFN double float_src_to_double(S v) { return (double)Ty<S>::to(v); }

template <typename S, typename D, bool SF = IsFloat<S>::value, bool DF = IsFloat<D>::value> struct CastOp;

template <typename S, typename D> struct CastOp<S, D, true, true> {
    static DEVFN D apply(S v) { return Ty<D>::from((typename Ty<D>::acc)Ty<S>::to(v)); }
};
template <typename S, typename D> struct CastOp<S, D, true, false> {
    static DEVFN D apply(S v) { return sat_from_double<D>(float_src_to_double(v)); }
};
template <typename S, typename D> struct CastOp<S, D, false, true> {
    static DEVFN D apply(S v) { return Ty<D>::from((typename Ty<D>::acc)v); }
};
template <typename S, typename D> struct CastOp<S, D, false, false> {
    static DEVFN D apply(S v) { return (D)v; }
};

template <> struct CastOp<double, f16_t, true, true> {
    static DEVFN f16_t apply(double v) {
        candle_hu16 h;
        h.h = (_Float16)v;
        f16_t r;
        r.bits = h.u;
        return r;
    }
};

template <> struct CastOp<double, f8e4m3_t, true, true> {
    static DEVFN f8e4m3_t apply(double v) {
        f8e4m3_t r;
        r.bits = double_to_f8e4m3(v);
        return r;
    }
};

template <> struct CastOp<f8e4m3_t, f8e4m3_t, true, true> {
    static DEVFN f8e4m3_t apply(f8e4m3_t v) { return v; }
};

template <typename S, typename D>
DEVFN void cast_map(const size_t numel, const SI1 &info, const S *inp, D *out) {
    if (info.ndim == 0) {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[i] = CastOp<S, D>::apply(inp[i]);
        }
    } else {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[i] = CastOp<S, D>::apply(inp[get_strided_index(i, info.ndim, info.dims, info.s0)]);
        }
    }
}

#define CAST_FN(S, SS, D, DS)                                                                               \
    extern "C" __global__ void cast_##SS##_##DS(const size_t numel, const SI1 info, const S *inp, D *out) {   \
        cast_map<S, D>(numel, info, inp, out);                                                                \
    }                                                                                                         \
    CANDLE_REGISTER(cast_##SS##_##DS)

#define CAST_FROM(S, SS)                 \
    CAST_FN(S, SS, uint8_t, u8)          \
    CAST_FN(S, SS, uint32_t, u32)        \
    CAST_FN(S, SS, int16_t, i16)         \
    CAST_FN(S, SS, int32_t, i32)         \
    CAST_FN(S, SS, int64_t, i64)         \
    CAST_FN(S, SS, bf16_t, bf16)         \
    CAST_FN(S, SS, f16_t, f16)           \
    CAST_FN(S, SS, float, f32)           \
    CAST_FN(S, SS, double, f64)          \
    CAST_FN(S, SS, f8e4m3_t, f8e4m3)

CAST_FROM(uint8_t, u8)
CAST_FROM(uint32_t, u32)
CAST_FROM(int16_t, i16)
CAST_FROM(int32_t, i32)
CAST_FROM(int64_t, i64)
CAST_FROM(bf16_t, bf16)
CAST_FROM(f16_t, f16)
CAST_FROM(float, f32)
CAST_FROM(double, f64)
CAST_FROM(f8e4m3_t, f8e4m3)
