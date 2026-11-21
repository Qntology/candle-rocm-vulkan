// Native ROCm GPU arg_sort_last_dim (LDS Bitonic Sort for ncols_pad <= 2048 and global
// multi-stage Bitonic Sort for larger rows such as full-vocab logits).
#include "common.h"

#define ASORT_LDS_MAX 2048u
#define ASORT_THREADS 256u

// Compare two elements at indices (ia, ib) of row `row_src`, where index >= ncols is a sentinel
// padding entry that always sorts to the end. Ties break deterministically by smaller index.
template <typename T, bool ASC>
DEVFN bool asort_should_swap(const T *row_src, uint32_t ia, uint32_t ib, uint32_t ncols, bool dir_asc) {
    const bool pa = ia >= ncols;
    const bool pb = ib >= ncols;
    if (pa && pb) return false;
    if (pa) return dir_asc;      // padding always sorts after valid elements in ascending order
    if (pb) return !dir_asc;

    const auto va = Ty<T>::to(row_src[ia]);
    const auto vb = Ty<T>::to(row_src[ib]);
    const bool na = (va != va);
    const bool nb = (vb != vb);
    bool a_before_b;
    if (na != nb) {
        // Match Rust f32::total_cmp: NaN sorts after all finite/inf values in ascending order
        a_before_b = ASC ? !na : na;
    } else if (va == vb || (na && nb)) {
        a_before_b = ia < ib;
    } else {
        a_before_b = ASC ? (va < vb) : (va > vb);
    }
    return dir_asc ? !a_before_b : a_before_b;
}

// Fast single-launch LDS Bitonic Sort for rows with ncols_pad <= 2048.
// Grid: (nrows, 1, 1), Block: (256, 1, 1). Shared memory: 2048 * 4 = 8 KiB.
template <typename T, bool ASC>
DEVFN void asort_lds_impl(const T *src, uint32_t *dst, const size_t nrows, const uint32_t ncols,
                          const uint32_t ncols_pad) {
    __shared__ uint32_t s_idx[ASORT_LDS_MAX];
    const uint32_t tid = threadIdx.x;
    for (size_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const T *row_src = src + row * (size_t)ncols;
        uint32_t *row_dst = dst + row * (size_t)ncols;
        for (uint32_t i = tid; i < ncols_pad; i += ASORT_THREADS) s_idx[i] = i;
        __syncthreads();

        for (uint32_t k = 2u; k <= ncols_pad; k <<= 1u) {
            for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
                for (uint32_t i = tid; i < ncols_pad; i += ASORT_THREADS) {
                    const uint32_t ixj = i ^ j;
                    if (ixj > i) {
                        const bool dir_asc = (i & k) == 0u;
                        const uint32_t ia = s_idx[i];
                        const uint32_t ib = s_idx[ixj];
                        if (asort_should_swap<T, ASC>(row_src, ia, ib, ncols, dir_asc)) {
                            s_idx[i] = ib;
                            s_idx[ixj] = ia;
                        }
                    }
                }
                __syncthreads();
            }
        }

        for (uint32_t i = tid; i < ncols; i += ASORT_THREADS) row_dst[i] = s_idx[i];
        __syncthreads();
    }
}

extern "C" __global__ void asort_init_pad(uint32_t *buf, const size_t nrows, const uint32_t ncols_pad) {
    const size_t total = nrows * (size_t)ncols_pad;
    for (size_t idx = GS_START; idx < total; idx += GS_STEP) {
        buf[idx] = (uint32_t)(idx % (size_t)ncols_pad);
    }
}
CANDLE_REGISTER(asort_init_pad)

extern "C" __global__ void asort_copy_out(const uint32_t *buf, uint32_t *dst, const size_t nrows,
                                          const uint32_t ncols, const uint32_t ncols_pad) {
    const size_t total = nrows * (size_t)ncols;
    for (size_t idx = GS_START; idx < total; idx += GS_STEP) {
        const size_t row = idx / (size_t)ncols;
        const size_t col = idx - row * (size_t)ncols;
        dst[idx] = buf[row * (size_t)ncols_pad + col];
    }
}
CANDLE_REGISTER(asort_copy_out)

// Global single bitonic compare-exchange stage for rows with ncols_pad > 2048.
template <typename T, bool ASC>
DEVFN void asort_global_step_impl(const T *src, uint32_t *buf, const size_t nrows, const uint32_t ncols,
                                  const uint32_t ncols_pad, const uint32_t k, const uint32_t j) {
    const size_t half = (size_t)(ncols_pad >> 1u);
    const size_t total = nrows * half;
    for (size_t p = GS_START; p < total; p += GS_STEP) {
        const size_t row = p / half;
        const uint32_t pair = (uint32_t)(p - row * half);
        // Map pair index 0..ncols_pad/2-1 to i with (i & j) == 0, ixj = i | j
        const uint32_t i = ((pair & ~(j - 1u)) << 1u) | (pair & (j - 1u));
        const uint32_t ixj = i | j;
        uint32_t *rbuf = buf + row * (size_t)ncols_pad;
        const uint32_t ia = rbuf[i];
        const uint32_t ib = rbuf[ixj];
        const bool dir_asc = (i & k) == 0u;
        if (asort_should_swap<T, ASC>(src + row * (size_t)ncols, ia, ib, ncols, dir_asc)) {
            rbuf[i] = ib;
            rbuf[ixj] = ia;
        }
    }
}

#define ASORT_FNS(T, SFX)                                                                                      \
    extern "C" __global__ void __launch_bounds__(ASORT_THREADS)                                                \
        asort_lds_asc_##SFX(const T *src, uint32_t *dst, const size_t nrows, const uint32_t ncols,             \
                            const uint32_t ncols_pad) {                                                        \
        asort_lds_impl<T, true>(src, dst, nrows, ncols, ncols_pad);                                            \
    }                                                                                                          \
    CANDLE_REGISTER(asort_lds_asc_##SFX)                                                                       \
    extern "C" __global__ void __launch_bounds__(ASORT_THREADS)                                                \
        asort_lds_desc_##SFX(const T *src, uint32_t *dst, const size_t nrows, const uint32_t ncols,            \
                             const uint32_t ncols_pad) {                                                       \
        asort_lds_impl<T, false>(src, dst, nrows, ncols, ncols_pad);                                           \
    }                                                                                                          \
    CANDLE_REGISTER(asort_lds_desc_##SFX)                                                                      \
    extern "C" __global__ void                                                                                 \
        asort_step_asc_##SFX(const T *src, uint32_t *buf, const size_t nrows, const uint32_t ncols,            \
                             const uint32_t ncols_pad, const uint32_t k, const uint32_t j) {                   \
        asort_global_step_impl<T, true>(src, buf, nrows, ncols, ncols_pad, k, j);                              \
    }                                                                                                          \
    CANDLE_REGISTER(asort_step_asc_##SFX)                                                                      \
    extern "C" __global__ void                                                                                 \
        asort_step_desc_##SFX(const T *src, uint32_t *buf, const size_t nrows, const uint32_t ncols,           \
                              const uint32_t ncols_pad, const uint32_t k, const uint32_t j) {                  \
        asort_global_step_impl<T, false>(src, buf, nrows, ncols, ncols_pad, k, j);                             \
    }                                                                                                          \
    CANDLE_REGISTER(asort_step_desc_##SFX)

ASORT_FNS(float, f32)
ASORT_FNS(double, f64)
ASORT_FNS(f16_t, f16)
ASORT_FNS(bf16_t, bf16)
ASORT_FNS(uint8_t, u8)
ASORT_FNS(uint32_t, u32)
ASORT_FNS(int16_t, i16)
ASORT_FNS(int32_t, i32)
ASORT_FNS(int64_t, i64)
