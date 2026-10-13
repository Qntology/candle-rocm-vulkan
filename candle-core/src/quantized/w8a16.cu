// Exact weight-only quantized matrix-vector products ("W8A16": int8/int4 weights, 16/32 bit
// activations) and direct dequantization kernels of the CUDA backend.
//
// Compiled with NVRTC at run time by `candle-core/src/quantized/w8a16.rs`, which appends one
// `W8A16_*_KERNEL(...)` instantiation line per module. NVRTC has no standard headers, so this
// file includes none.
//
// Numerics (the reason these kernels exist, see `QMatMul::forward_exact`):
//   * activations are NOT quantized: f16 / bf16 inputs are widened to f32 exactly;
//   * every weight is dequantized in f32 exactly like `k_quants::Block*::to_float` on the CPU
//     (separately rounded products, no FMA contraction - `__fmul_rn` / `__fsub_rn`), so the
//     dequantization kernels are bit exact with `QTensor::dequantize` on the CPU;
//   * products are accumulated in f32 and the result is rounded once to the output dtype
//     (round to nearest even).

typedef unsigned char w_u8;
typedef signed char w_i8;
typedef unsigned short w_u16;
typedef unsigned int w_u32;
typedef long long w_i64;

// Storage types of candle's f16 / bf16 (layout compatible with `half::f16` / `half::bf16`).
struct w_f16 {
    w_u16 b;
};
struct w_bf16 {
    w_u16 b;
};

struct __align__(8) w_v2 {
    w_u32 x, y;
};
struct __align__(16) w_v4 {
    w_u32 x, y, z, w;
};

#define W_DEV __device__ __forceinline__

// ---------------------------------------------------------------------------
// Scalar conversions (bit exact with `half`: IEEE round to nearest even, NaN kept)
// ---------------------------------------------------------------------------

W_DEV float w_h2f(w_u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

W_DEV w_u16 w_f2h(float f) {
    w_u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

W_DEV float w_bf2f(w_u32 b) { return __uint_as_float(b << 16); }

W_DEV w_u16 w_f2bf(float f) {
    w_u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (w_u16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (w_u16)(u >> 16);
}

W_DEV w_u16 w_ld16(const w_u8 *p) { return __ldg(reinterpret_cast<const w_u16 *>(p)); }

// 8 bytes at a 2 byte aligned address, as two little endian words.
W_DEV void w_ld8_a2(const w_u8 *p, w_u32 &lo, w_u32 &hi) {
    const w_u16 *q = reinterpret_cast<const w_u16 *>(p);
    lo = (w_u32)__ldg(q) | ((w_u32)__ldg(q + 1) << 16);
    hi = (w_u32)__ldg(q + 2) | ((w_u32)__ldg(q + 3) << 16);
}

// 8 bytes at an 8 byte aligned address.
W_DEV void w_ld8_a8(const w_u8 *p, w_u32 &lo, w_u32 &hi) {
    const w_v2 v = *reinterpret_cast<const w_v2 *>(p);
    lo = v.x;
    hi = v.y;
}

// Byte `i` (0..7) of the 8 byte group (lo, hi).
#define W_BYTE(lo, hi, i) ((((i) < 4 ? (lo) : (hi)) >> (8 * ((i) & 3))) & 0xffu)

// ---------------------------------------------------------------------------
// Element I/O: 8 consecutive values (16 byte aligned for 16 bit types, 32 for f32)
// ---------------------------------------------------------------------------

template <typename T> struct WIo;

template <> struct WIo<float> {
    static W_DEV void load8(const float *p, float *v) {
        const w_v4 a = *reinterpret_cast<const w_v4 *>(p);
        const w_v4 b = *(reinterpret_cast<const w_v4 *>(p) + 1);
        v[0] = __uint_as_float(a.x);
        v[1] = __uint_as_float(a.y);
        v[2] = __uint_as_float(a.z);
        v[3] = __uint_as_float(a.w);
        v[4] = __uint_as_float(b.x);
        v[5] = __uint_as_float(b.y);
        v[6] = __uint_as_float(b.z);
        v[7] = __uint_as_float(b.w);
    }
    static W_DEV void store8(float *p, const float *v) {
        w_v4 a, b;
        a.x = __float_as_uint(v[0]);
        a.y = __float_as_uint(v[1]);
        a.z = __float_as_uint(v[2]);
        a.w = __float_as_uint(v[3]);
        b.x = __float_as_uint(v[4]);
        b.y = __float_as_uint(v[5]);
        b.z = __float_as_uint(v[6]);
        b.w = __float_as_uint(v[7]);
        reinterpret_cast<w_v4 *>(p)[0] = a;
        reinterpret_cast<w_v4 *>(p)[1] = b;
    }
    static W_DEV float to_f(float v) { return v; }
    static W_DEV float from_f(float v) { return v; }
};

template <> struct WIo<w_bf16> {
    static W_DEV void load8(const w_bf16 *p, float *v) {
        const w_v4 a = *reinterpret_cast<const w_v4 *>(p);
        v[0] = w_bf2f(a.x & 0xffffu);
        v[1] = w_bf2f(a.x >> 16);
        v[2] = w_bf2f(a.y & 0xffffu);
        v[3] = w_bf2f(a.y >> 16);
        v[4] = w_bf2f(a.z & 0xffffu);
        v[5] = w_bf2f(a.z >> 16);
        v[6] = w_bf2f(a.w & 0xffffu);
        v[7] = w_bf2f(a.w >> 16);
    }
    static W_DEV void store8(w_bf16 *p, const float *v) {
        w_v4 a;
        a.x = (w_u32)w_f2bf(v[0]) | ((w_u32)w_f2bf(v[1]) << 16);
        a.y = (w_u32)w_f2bf(v[2]) | ((w_u32)w_f2bf(v[3]) << 16);
        a.z = (w_u32)w_f2bf(v[4]) | ((w_u32)w_f2bf(v[5]) << 16);
        a.w = (w_u32)w_f2bf(v[6]) | ((w_u32)w_f2bf(v[7]) << 16);
        *reinterpret_cast<w_v4 *>(p) = a;
    }
    static W_DEV float to_f(w_bf16 v) { return w_bf2f(v.b); }
    static W_DEV w_bf16 from_f(float v) {
        w_bf16 r;
        r.b = w_f2bf(v);
        return r;
    }
};

template <> struct WIo<w_f16> {
    static W_DEV void load8(const w_f16 *p, float *v) {
        const w_v4 a = *reinterpret_cast<const w_v4 *>(p);
        v[0] = w_h2f((w_u16)(a.x & 0xffffu));
        v[1] = w_h2f((w_u16)(a.x >> 16));
        v[2] = w_h2f((w_u16)(a.y & 0xffffu));
        v[3] = w_h2f((w_u16)(a.y >> 16));
        v[4] = w_h2f((w_u16)(a.z & 0xffffu));
        v[5] = w_h2f((w_u16)(a.z >> 16));
        v[6] = w_h2f((w_u16)(a.w & 0xffffu));
        v[7] = w_h2f((w_u16)(a.w >> 16));
    }
    static W_DEV void store8(w_f16 *p, const float *v) {
        w_v4 a;
        a.x = (w_u32)w_f2h(v[0]) | ((w_u32)w_f2h(v[1]) << 16);
        a.y = (w_u32)w_f2h(v[2]) | ((w_u32)w_f2h(v[3]) << 16);
        a.z = (w_u32)w_f2h(v[4]) | ((w_u32)w_f2h(v[5]) << 16);
        a.w = (w_u32)w_f2h(v[6]) | ((w_u32)w_f2h(v[7]) << 16);
        *reinterpret_cast<w_v4 *>(p) = a;
    }
    static W_DEV float to_f(w_f16 v) { return w_h2f(v.b); }
    static W_DEV w_f16 from_f(float v) {
        w_f16 r;
        r.b = w_f2h(v);
        return r;
    }
};

// ---------------------------------------------------------------------------
// Block decoders. A block of QK weights (BYTES bytes) is split into TASKS lane tasks; a task
// yields RUNS runs of 8 consecutive weights, run r starting at weight `xo[r]` of the block.
// Consecutive tasks read consecutive bytes, so consecutive lanes load coalesced.
// ---------------------------------------------------------------------------

// llama.cpp / k_quants.rs `get_scale_min_k4`.
W_DEV void w_scale_min_k4(int j, const w_u8 *q, int &d, int &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct DecQ8_0 {
    enum { QK = 32, BYTES = 34, TASKS = 4, RUNS = 1 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        w_u32 lo, hi;
        w_ld8_a2(blk + 2 + 8 * t, lo, hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __fmul_rn((float)(w_i8)W_BYTE(lo, hi, i), d);
        xo[0] = 8 * t;
    }
};

struct DecQ4_0 {
    enum { QK = 32, BYTES = 18, TASKS = 2, RUNS = 2 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        w_u32 lo, hi;
        w_ld8_a2(blk + 2 + 8 * t, lo, hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 b = W_BYTE(lo, hi, i);
            w[i] = __fmul_rn((float)((int)(b & 0xFu) - 8), d);
            w[8 + i] = __fmul_rn((float)((int)(b >> 4) - 8), d);
        }
        xo[0] = 8 * t;
        xo[1] = 16 + 8 * t;
    }
};

struct DecQ4K {
    enum { QK = 256, BYTES = 144, TASKS = 16, RUNS = 2 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        const float dmin = w_h2f(w_ld16(blk + 2));
        const int j = t >> 2, l0 = (t & 3) * 8;
        int sc, m;
        w_scale_min_k4(2 * j, blk + 4, sc, m);
        const float d1 = __fmul_rn(d, (float)sc), m1 = __fmul_rn(dmin, (float)m);
        w_scale_min_k4(2 * j + 1, blk + 4, sc, m);
        const float d2 = __fmul_rn(d, (float)sc), m2 = __fmul_rn(dmin, (float)m);
        w_u32 lo, hi;
        w_ld8_a8(blk + 16 + 32 * j + l0, lo, hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 b = W_BYTE(lo, hi, i);
            w[i] = __fsub_rn(__fmul_rn(d1, (float)(b & 0xFu)), m1);
            w[8 + i] = __fsub_rn(__fmul_rn(d2, (float)(b >> 4)), m2);
        }
        xo[0] = 64 * j + l0;
        xo[1] = 64 * j + 32 + l0;
    }
};

struct DecQ5K {
    enum { QK = 256, BYTES = 176, TASKS = 16, RUNS = 2 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const float d = w_h2f(w_ld16(blk));
        const float dmin = w_h2f(w_ld16(blk + 2));
        const int j = t >> 2, l0 = (t & 3) * 8;
        int sc, m;
        w_scale_min_k4(2 * j, blk + 4, sc, m);
        const float d1 = __fmul_rn(d, (float)sc), m1 = __fmul_rn(dmin, (float)m);
        w_scale_min_k4(2 * j + 1, blk + 4, sc, m);
        const float d2 = __fmul_rn(d, (float)sc), m2 = __fmul_rn(dmin, (float)m);
        w_u32 lo, hi, hlo, hhi;
        w_ld8_a8(blk + 48 + 32 * j + l0, lo, hi);
        w_ld8_a8(blk + 16 + l0, hlo, hhi);
        const w_u32 u1 = 1u << (2 * j), u2 = 2u << (2 * j);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 b = W_BYTE(lo, hi, i);
            const w_u32 h = W_BYTE(hlo, hhi, i);
            w[i] = __fsub_rn(__fmul_rn(d1, __fadd_rn((float)(b & 0xFu), (h & u1) ? 16.0f : 0.0f)), m1);
            w[8 + i] = __fsub_rn(__fmul_rn(d2, __fadd_rn((float)(b >> 4), (h & u2) ? 16.0f : 0.0f)), m2);
        }
        xo[0] = 64 * j + l0;
        xo[1] = 64 * j + 32 + l0;
    }
};

struct DecQ6K {
    enum { QK = 256, BYTES = 210, TASKS = 8, RUNS = 4 };
    static W_DEV void decode(const w_u8 *blk, int t, float *w, int *xo) {
        const int nh = t >> 2, l0 = (t & 3) * 8, is = l0 / 16;
        const float d = w_h2f(w_ld16(blk + 208));
        const w_u8 *sc = blk + 192 + 8 * nh;
        const float s1 = __fmul_rn(d, (float)(w_i8)sc[is]);
        const float s2 = __fmul_rn(d, (float)(w_i8)sc[is + 2]);
        const float s3 = __fmul_rn(d, (float)(w_i8)sc[is + 4]);
        const float s4 = __fmul_rn(d, (float)(w_i8)sc[is + 6]);
        w_u32 alo, ahi, blo, bhi, hlo, hhi;
        w_ld8_a2(blk + 64 * nh + l0, alo, ahi);
        w_ld8_a2(blk + 64 * nh + 32 + l0, blo, bhi);
        w_ld8_a2(blk + 128 + 32 * nh + l0, hlo, hhi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const w_u32 a = W_BYTE(alo, ahi, i), b = W_BYTE(blo, bhi, i), h = W_BYTE(hlo, hhi, i);
            const int q1 = (int)((a & 0xFu) | ((h & 3u) << 4)) - 32;
            const int q2 = (int)((b & 0xFu) | (((h >> 2) & 3u) << 4)) - 32;
            const int q3 = (int)((a >> 4) | (((h >> 4) & 3u) << 4)) - 32;
            const int q4 = (int)((b >> 4) | (((h >> 6) & 3u) << 4)) - 32;
            w[i] = __fmul_rn(s1, (float)q1);
            w[8 + i] = __fmul_rn(s2, (float)q2);
            w[16 + i] = __fmul_rn(s3, (float)q3);
            w[24 + i] = __fmul_rn(s4, (float)q4);
        }
        xo[0] = 128 * nh + l0;
        xo[1] = 128 * nh + 32 + l0;
        xo[2] = 128 * nh + 64 + l0;
        xo[3] = 128 * nh + 96 + l0;
    }
};

// ---------------------------------------------------------------------------
// y[m, row] = sum_k dequant(W)[row, k] * x[m, k]   for m < M (M <= 8 input rows).
//
// Block = (32, WPR, RPB) threads: RPB weight rows per block, WPR warps share the k range of a
// row (reduced through shared memory), each lane walks the row's lane tasks with a stride of
// 32 * WPR (UNR tasks unrolled). `x` holds M contiguous rows of k values, `y` M contiguous
// rows of n values.
// ---------------------------------------------------------------------------

template <typename D, typename XT, int M, int WPR, int RPB, int UNR>
W_DEV void w8a16_mv(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y,
                    const int k, const int n, const w_i64 row_bytes) {
    const int lane = threadIdx.x, wy = threadIdx.y, rz = threadIdx.z;
    const int row = blockIdx.x * RPB + rz;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    if (row < n) {
        const w_u8 *wr = wq + (w_i64)row * row_bytes;
        const int ntasks = (k / D::QK) * D::TASKS;
#pragma unroll UNR
        for (int tk = wy * 32 + lane; tk < ntasks; tk += 32 * WPR) {
            const int b = tk / D::TASKS, t = tk % D::TASKS;
            float w[8 * D::RUNS];
            int xo[D::RUNS];
            D::decode(wr + (w_i64)b * D::BYTES, t, w, xo);
#pragma unroll
            for (int r = 0; r < D::RUNS; ++r) {
                const XT *xr = x + b * D::QK + xo[r];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float xv[8];
                    WIo<XT>::load8(xr + (w_i64)m * k, xv);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[m] = fmaf(w[8 * r + i], xv[i], acc[m]);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);
    }
    if (WPR > 1) {
        __shared__ float red[RPB][WPR][M];
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < M; ++m) red[rz][wy][m] = acc[m];
        }
        __syncthreads();
        if (wy == 0 && lane < M && row < n) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; ++w) s += red[rz][w][lane];
            y[(w_i64)lane * n + row] = WIo<XT>::from_f(s);
        }
    } else {
        float v = acc[0];
#pragma unroll
        for (int m = 1; m < M; ++m) v = lane == m ? acc[m] : v;
        if (lane < M && row < n) y[(w_i64)lane * n + row] = WIo<XT>::from_f(v);
    }
}

#define W8A16_MV_KERNEL(name, D, XT, M, WPR, RPB, UNR)                                             \
    extern "C" __global__ void __launch_bounds__(32 * (WPR) * (RPB))                               \
        name(const w_u8 *__restrict__ wq, const XT *__restrict__ x, XT *__restrict__ y, int k,      \
             int n, w_i64 row_bytes) {                                                             \
        w8a16_mv<D, XT, M, WPR, RPB, UNR>(wq, x, y, k, n, row_bytes);                              \
    }

// ---------------------------------------------------------------------------
// Dequantization of `nblocks` consecutive blocks into a dense row-major array.
// ---------------------------------------------------------------------------

template <typename D, typename OT>
W_DEV void w8a16_dq(const w_u8 *__restrict__ wq, OT *__restrict__ y, const w_i64 nblocks) {
    const w_i64 ntasks = nblocks * D::TASKS;
    for (w_i64 tk = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; tk < ntasks;
         tk += (w_i64)gridDim.x * blockDim.x) {
        const w_i64 b = tk / D::TASKS;
        const int t = (int)(tk % D::TASKS);
        float w[8 * D::RUNS];
        int xo[D::RUNS];
        D::decode(wq + b * D::BYTES, t, w, xo);
#pragma unroll
        for (int r = 0; r < D::RUNS; ++r) WIo<OT>::store8(y + b * D::QK + xo[r], w + 8 * r);
    }
}

#define W8A16_DQ_KERNEL(name, D, OT)                                                                \
    extern "C" __global__ void name(const w_u8 *__restrict__ wq, OT *__restrict__ y,               \
                                    w_i64 nblocks) {                                               \
        w8a16_dq<D, OT>(wq, y, nblocks);                                                           \
    }

// ---------------------------------------------------------------------------
// Element type conversion of F32 / F16 / BF16 GGUF tensors (one rounding, no f32 copy).
// ---------------------------------------------------------------------------

template <typename IT, typename OT>
W_DEV void w8a16_cvt(const IT *__restrict__ x, OT *__restrict__ y, const w_i64 n) {
    for (w_i64 i = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (w_i64)gridDim.x * blockDim.x)
        y[i] = WIo<OT>::from_f(WIo<IT>::to_f(x[i]));
}

#define W8A16_CVT_KERNEL(name, IT, OT)                                                              \
    extern "C" __global__ void name(const IT *__restrict__ x, OT *__restrict__ y, w_i64 n) {       \
        w8a16_cvt<IT, OT>(x, y, n);                                                                \
    }

// ---------------------------------------------------------------------------
// f32 GEMM tile (rows x cols, contiguous) rounded once into columns of a wider output whose
// rows are `ldy` values apart (the multi-tile prefill path).
// ---------------------------------------------------------------------------

template <typename OT>
W_DEV void w8a16_cvt2d(const float *__restrict__ x, OT *__restrict__ y, const int rows,
                       const int cols, const w_i64 ldy) {
    const w_i64 total = (w_i64)rows * cols;
    for (w_i64 i = (w_i64)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (w_i64)gridDim.x * blockDim.x) {
        const w_i64 r = i / cols, c = i - r * cols;
        y[r * ldy + c] = WIo<OT>::from_f(x[i]);
    }
}

#define W8A16_CVT2D_KERNEL(name, OT)                                                                \
    extern "C" __global__ void name(const float *__restrict__ x, OT *__restrict__ y, int rows,     \
                                    int cols, w_i64 ldy) {                                         \
        w8a16_cvt2d<OT>(x, y, rows, cols, ldy);                                                    \
    }
