#include "common.h"

template <typename T, typename F>
DEVFN void unary_map(const size_t numel, const SI1 &info, const T *inp, T *out, F f) {
    if (info.ndim == 0) {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            out[i] = Ty<T>::from(f(Ty<T>::to(inp[i])));
        }
    } else {
        for (size_t i = GS_START; i < numel; i += GS_STEP) {
            const size_t src_i = get_strided_index(i, info.ndim, info.dims, info.s0);
            out[i] = Ty<T>::from(f(Ty<T>::to(inp[src_i])));
        }
    }
}

template <typename A> struct OpExp { DEVFN A operator()(A x) const { return expg(x); } };
template <typename A> struct OpLog { DEVFN A operator()(A x) const { return logg(x); } };
template <typename A> struct OpSin { DEVFN A operator()(A x) const { return sing(x); } };
template <typename A> struct OpCos { DEVFN A operator()(A x) const { return cosg(x); } };
template <typename A> struct OpTanh { DEVFN A operator()(A x) const { return tanhg(x); } };
template <typename A> struct OpNeg { DEVFN A operator()(A x) const { return -x; } };
template <typename A> struct OpRecip { DEVFN A operator()(A x) const { return (A)1 / x; } };
template <typename A> struct OpSqr { DEVFN A operator()(A x) const { return x * x; } };
template <typename A> struct OpSqrt { DEVFN A operator()(A x) const { return sqrtg(x); } };
template <typename A> struct OpGelu {
    DEVFN A operator()(A x) const {
        const A k = (A)0.79788456080286535587989211986876373;
        return (A)0.5 * x * ((A)1 + tanhg(k * x * ((A)1 + (A)0.044715 * x * x)));
    }
};
template <typename A> struct OpGeluErf {
    DEVFN A operator()(A x) const { return (A)0.5 * x * ((A)1 + erfg(x * (A)0.70710678118654752440084436210485)); }
};
template <typename A> struct OpErf { DEVFN A operator()(A x) const { return erfg(x); } };
template <typename A> struct OpSilu { DEVFN A operator()(A x) const { return x / ((A)1 + expg(-x)); } };
template <typename A> struct OpSigmoid { DEVFN A operator()(A x) const { return (A)1 / ((A)1 + expg(-x)); } };
template <typename A> struct OpAbs { DEVFN A operator()(A x) const { return absg(x); } };
template <typename A> struct OpCeil { DEVFN A operator()(A x) const { return ceilg(x); } };
template <typename A> struct OpFloor { DEVFN A operator()(A x) const { return floorg(x); } };
template <typename A> struct OpRound { DEVFN A operator()(A x) const { return roundg(x); } };
template <typename A> struct OpRelu { DEVFN A operator()(A x) const { return isnang(x) ? (A)0 : (x > (A)0 ? x : (A)0); } };
template <typename A> struct OpSign { DEVFN A operator()(A x) const { return (A)(x > (A)0) - (A)(x < (A)0); } };

template <typename A> struct OpAffine {
    A mul, add;
    DEVFN A operator()(A x) const { return x * mul + add; }
};
template <typename A> struct OpPowf {
    A e;
    DEVFN A operator()(A x) const { return powg(x, e); }
};
template <typename A> struct OpElu {
    A alpha;
    DEVFN A operator()(A x) const { return (x >= (A)0 && !(x == (A)0 && signbit(x))) ? x : (expg(x) - (A)1) * alpha; }
};

#define UNARY_FN(OPNAME, OPSTRUCT, T, SFX)                                                                  \
    extern "C" __global__ void OPNAME##_##SFX(const size_t numel, const SI1 info, const T *inp, T *out) {    \
        unary_map<T>(numel, info, inp, out, OPSTRUCT<Ty<T>::acc>());                                         \
    }                                                                                                        \
    CANDLE_REGISTER(OPNAME##_##SFX)

#define UNARY_ALL(T, SFX)                     \
    UNARY_FN(uexp, OpExp, T, SFX)             \
    UNARY_FN(ulog, OpLog, T, SFX)             \
    UNARY_FN(usin, OpSin, T, SFX)             \
    UNARY_FN(ucos, OpCos, T, SFX)             \
    UNARY_FN(utanh, OpTanh, T, SFX)           \
    UNARY_FN(uneg, OpNeg, T, SFX)             \
    UNARY_FN(urecip, OpRecip, T, SFX)         \
    UNARY_FN(usqr, OpSqr, T, SFX)             \
    UNARY_FN(usqrt, OpSqrt, T, SFX)           \
    UNARY_FN(ugelu, OpGelu, T, SFX)           \
    UNARY_FN(ugelu_erf, OpGeluErf, T, SFX)    \
    UNARY_FN(uerf, OpErf, T, SFX)             \
    UNARY_FN(usilu, OpSilu, T, SFX)           \
    UNARY_FN(usigmoid, OpSigmoid, T, SFX)     \
    UNARY_FN(uabs, OpAbs, T, SFX)             \
    UNARY_FN(uceil, OpCeil, T, SFX)           \
    UNARY_FN(ufloor, OpFloor, T, SFX)         \
    UNARY_FN(uround, OpRound, T, SFX)         \
    UNARY_FN(urelu, OpRelu, T, SFX)           \
    UNARY_FN(usign, OpSign, T, SFX)

UNARY_ALL(float, f32)
UNARY_ALL(double, f64)
UNARY_ALL(f16_t, f16)
UNARY_ALL(bf16_t, bf16)

#define PARAM_FNS(T, SFX)                                                                                    \
    extern "C" __global__ void affine_##SFX(const size_t numel, const SI1 info, const T *inp, T *out,        \
                                            const double mul, const double add) {                            \
        OpAffine<Ty<T>::acc> op;                                                                             \
        op.mul = (Ty<T>::acc)mul;                                                                            \
        op.add = (Ty<T>::acc)add;                                                                            \
        unary_map<T>(numel, info, inp, out, op);                                                             \
    }                                                                                                        \
    CANDLE_REGISTER(affine_##SFX)                                                                            \
    extern "C" __global__ void upowf_##SFX(const size_t numel, const SI1 info, const T *inp, T *out,         \
                                           const double e) {                                                 \
        OpPowf<Ty<T>::acc> op;                                                                               \
        op.e = (Ty<T>::acc)e;                                                                                \
        unary_map<T>(numel, info, inp, out, op);                                                             \
    }                                                                                                        \
    CANDLE_REGISTER(upowf_##SFX)                                                                             \
    extern "C" __global__ void uelu_##SFX(const size_t numel, const SI1 info, const T *inp, T *out,          \
                                          const double alpha) {                                              \
        OpElu<Ty<T>::acc> op;                                                                                \
        op.alpha = (Ty<T>::acc)alpha;                                                                        \
        unary_map<T>(numel, info, inp, out, op);                                                             \
    }                                                                                                        \
    CANDLE_REGISTER(uelu_##SFX)

PARAM_FNS(float, f32)
PARAM_FNS(double, f64)
PARAM_FNS(f16_t, f16)
PARAM_FNS(bf16_t, bf16)
