#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#ifdef CANDLE_ROCM_MOCK
#include "mock_register.h"
#define CANDLE_REGISTER(name) static const int candle_mock_reg_##name = ::candle_mock::register_kernel(#name, &name);
#else
#define CANDLE_REGISTER(name)
#endif

#ifdef CANDLE_ROCM_MOCK
#define DEVFN inline
#else
#define DEVFN __device__ __forceinline__
#endif
#define GS_START (blockIdx.x * (size_t)blockDim.x + threadIdx.x)
#define GS_STEP ((size_t)blockDim.x * gridDim.x)

struct f16_t { uint16_t bits; };
struct bf16_t { uint16_t bits; };

union candle_fu32 { float f; uint32_t u; };
union candle_hu16 { _Float16 h; uint16_t u; };

DEVFN float bf16_to_float(uint16_t b) {
    candle_fu32 v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

DEVFN uint16_t float_to_bf16(float f) {
    candle_fu32 v;
    v.f = f;
    uint32_t u = v.u;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((u >> 16) | 0x0040u);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

DEVFN float f16_to_float(uint16_t b) {
    candle_hu16 v;
    v.u = b;
    return (float)v.h;
}

DEVFN uint16_t float_to_f16(float f) {
    candle_hu16 v;
    v.h = (_Float16)f;
    return v.u;
}

struct f8e4m3_t { uint8_t bits; };

union candle_du64 { double d; uint64_t u; };

DEVFN uint16_t f8e4m3_to_f16_bits(uint8_t x) {
    uint16_t ur = (uint16_t)(((uint16_t)x) << 8);
    const uint16_t sign = (uint16_t)(ur & 0x8000u);
    uint16_t exponent = (uint16_t)((uint16_t)((ur & 0x7800u) >> 1) + 0x2000u);
    uint16_t mantissa = (uint16_t)((ur & 0x0700u) >> 1);
    if ((x & 0x7Fu) == 0x7Fu) {
        return 0x7FFFu;
    }
    if (exponent == 0x2000u) {
        if (mantissa != 0) {
            mantissa = (uint16_t)(mantissa << 1);
            while ((mantissa & 0x0400u) == 0) {
                mantissa = (uint16_t)(mantissa << 1);
                exponent = (uint16_t)(exponent - 0x0400u);
            }
            mantissa = (uint16_t)(mantissa & 0x03FFu);
        } else {
            exponent = 0;
        }
    }
    return (uint16_t)(sign | exponent | mantissa);
}

DEVFN uint8_t double_to_f8e4m3(double x) {
    candle_du64 v;
    v.d = x;
    const uint64_t xbits = v.u;
    const uint64_t half_ulp = 1ull << 48;
    const uint8_t sign = (uint8_t)((xbits >> 63) << 7);
    uint16_t e16 = (uint16_t)(((uint16_t)(xbits >> 52)) & 0x7FFu);
    e16 = (uint16_t)(e16 - 1023u);
    e16 = (uint16_t)(e16 + 7u);
    const uint8_t exp = (uint8_t)e16;
    const uint8_t mantissa = (uint8_t)((xbits >> 49) & 0x7u);
    const uint64_t absx = xbits & 0x7FFFFFFFFFFFFFFFull;
    uint8_t res;
    if (absx <= 0x3F50000000000000ull) {
        res = 0;
    } else if (absx > 0x7FF0000000000000ull) {
        res = 0x7F;
    } else if (absx > 0x407D000000000000ull) {
        res = 0x7E;
    } else if (absx >= 0x3F90000000000000ull) {
        res = (uint8_t)((uint8_t)(exp << 3) | mantissa);
        const uint64_t round = xbits & ((half_ulp << 1) - 1);
        if ((round > half_ulp) || ((round == half_ulp) && (mantissa & 1u))) {
            res = (uint8_t)(res + 1);
        }
    } else {
        const uint8_t shift = (uint8_t)(1u - exp);
        const uint8_t m = (uint8_t)(mantissa | 0x8u);
        res = (uint8_t)(m >> shift);
        const uint64_t round = (xbits | (1ull << 52)) & ((half_ulp << (shift + 1)) - 1);
        const uint64_t half = half_ulp << shift;
        if ((round > half) || ((round == half) && (res & 1u))) {
            res = (uint8_t)(res + 1);
        }
    }
    return (uint8_t)(res | sign);
}

template <typename T> struct Ty;

template <> struct Ty<float> {
    typedef float acc;
    static DEVFN float to(float v) { return v; }
    static DEVFN float from(float v) { return v; }
};
template <> struct Ty<double> {
    typedef double acc;
    static DEVFN double to(double v) { return v; }
    static DEVFN double from(double v) { return v; }
};
template <> struct Ty<f16_t> {
    typedef float acc;
    static DEVFN float to(f16_t v) { return f16_to_float(v.bits); }
    static DEVFN f16_t from(float v) { f16_t r; r.bits = float_to_f16(v); return r; }
};
template <> struct Ty<bf16_t> {
    typedef float acc;
    static DEVFN float to(bf16_t v) { return bf16_to_float(v.bits); }
    static DEVFN bf16_t from(float v) { bf16_t r; r.bits = float_to_bf16(v); return r; }
};
template <> struct Ty<f8e4m3_t> {
    typedef float acc;
    static DEVFN float to(f8e4m3_t v) { return f16_to_float(f8e4m3_to_f16_bits(v.bits)); }
    static DEVFN f8e4m3_t from(float v) { f8e4m3_t r; r.bits = double_to_f8e4m3((double)v); return r; }
};
template <> struct Ty<uint8_t> {
    typedef uint8_t acc;
    static DEVFN uint8_t to(uint8_t v) { return v; }
    static DEVFN uint8_t from(uint8_t v) { return v; }
};
template <> struct Ty<uint32_t> {
    typedef uint32_t acc;
    static DEVFN uint32_t to(uint32_t v) { return v; }
    static DEVFN uint32_t from(uint32_t v) { return v; }
};
template <> struct Ty<int16_t> {
    typedef int16_t acc;
    static DEVFN int16_t to(int16_t v) { return v; }
    static DEVFN int16_t from(int16_t v) { return v; }
};
template <> struct Ty<int32_t> {
    typedef int32_t acc;
    static DEVFN int32_t to(int32_t v) { return v; }
    static DEVFN int32_t from(int32_t v) { return v; }
};
template <> struct Ty<int64_t> {
    typedef int64_t acc;
    static DEVFN int64_t to(int64_t v) { return v; }
    static DEVFN int64_t from(int64_t v) { return v; }
};

#define CANDLE_MAX_DIMS 8

struct SI1 {
    size_t ndim;
    size_t dims[CANDLE_MAX_DIMS];
    size_t s0[CANDLE_MAX_DIMS];
};

struct SI2 {
    size_t ndim;
    size_t dims[CANDLE_MAX_DIMS];
    size_t s0[CANDLE_MAX_DIMS];
    size_t s1[CANDLE_MAX_DIMS];
};

struct SI3 {
    size_t ndim;
    size_t dims[CANDLE_MAX_DIMS];
    size_t s0[CANDLE_MAX_DIMS];
    size_t s1[CANDLE_MAX_DIMS];
    size_t s2[CANDLE_MAX_DIMS];
};

DEVFN size_t get_strided_index(size_t idx, const size_t num_dims, const size_t *dims, const size_t *strides) {
    size_t strided_i = 0;
    for (size_t d = 0; d < num_dims; d++) {
        const size_t dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

DEVFN float expg(float a) { return expf(a); }
DEVFN double expg(double a) { return exp(a); }
DEVFN float logg(float a) { return logf(a); }
DEVFN double logg(double a) { return log(a); }
DEVFN float sing(float a) { return sinf(a); }
DEVFN double sing(double a) { return sin(a); }
DEVFN float cosg(float a) { return cosf(a); }
DEVFN double cosg(double a) { return cos(a); }
DEVFN float tanhg(float a) { return tanhf(a); }
DEVFN double tanhg(double a) { return tanh(a); }
DEVFN float sqrtg(float a) { return sqrtf(a); }
DEVFN double sqrtg(double a) { return sqrt(a); }
DEVFN float erfg(float a) { return erff(a); }
DEVFN double erfg(double a) { return erf(a); }
DEVFN float absg(float a) { return fabsf(a); }
DEVFN double absg(double a) { return fabs(a); }
DEVFN float ceilg(float a) { return ceilf(a); }
DEVFN double ceilg(double a) { return ceil(a); }
DEVFN float floorg(float a) { return floorf(a); }
DEVFN double floorg(double a) { return floor(a); }
DEVFN float roundg(float a) { return roundf(a); }
DEVFN double roundg(double a) { return round(a); }
DEVFN float powg(float a, float b) { return powf(a, b); }
DEVFN double powg(double a, double b) { return pow(a, b); }
DEVFN float maxg(float a, float b) { return fmaxf(a, b); }
DEVFN double maxg(double a, double b) { return fmax(a, b); }
DEVFN float ming(float a, float b) { return fminf(a, b); }
DEVFN double ming(double a, double b) { return fmin(a, b); }
DEVFN bool isnang(float a) { return a != a; }
DEVFN bool isnang(double a) { return a != a; }

template <typename T> struct Limits;
template <> struct Limits<float> {
    static DEVFN float lowest() { return -INFINITY; }
    static DEVFN float highest() { return INFINITY; }
};
template <> struct Limits<double> {
    static DEVFN double lowest() { return -INFINITY; }
    static DEVFN double highest() { return INFINITY; }
};
template <> struct Limits<uint8_t> {
    static DEVFN uint8_t lowest() { return 0; }
    static DEVFN uint8_t highest() { return 0xFF; }
};
template <> struct Limits<uint32_t> {
    static DEVFN uint32_t lowest() { return 0; }
    static DEVFN uint32_t highest() { return 0xFFFFFFFFu; }
};
template <> struct Limits<int16_t> {
    static DEVFN int16_t lowest() { return (int16_t)-32768; }
    static DEVFN int16_t highest() { return (int16_t)32767; }
};
template <> struct Limits<int32_t> {
    static DEVFN int32_t lowest() { return (int32_t)0x80000000; }
    static DEVFN int32_t highest() { return 0x7FFFFFFF; }
};
template <> struct Limits<int64_t> {
    static DEVFN int64_t lowest() { return (int64_t)0x8000000000000000ULL; }
    static DEVFN int64_t highest() { return 0x7FFFFFFFFFFFFFFFLL; }
};
