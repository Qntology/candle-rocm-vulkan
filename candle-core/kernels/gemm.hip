// GEMM used when rocBLAS cannot run on this GPU (no rocBLAS kernels for the target in the
// installed ROCm release, see `RocmDevice::gemm`), or when CANDLE_ROCM_GEMM=hip.
//
// Same contract as rocBLAS `<t>gemm_strided_batched` with alpha = 1 and beta = 0, column-major:
//   C[i + j*ldc] = sum_p op(A)(i, p) * op(B)(p, j)        for i < m, j < n, p < k
//   op(A)(i, p) = transa ? A[p + i*lda] : A[i + p*lda]
//   op(B)(p, j) = transb ? B[j + p*ldb] : B[p + j*ldb]
// for every batch b with A + b*stride_a, B + b*stride_b and C + b*stride_c.
//
// Tiled in shared memory: 256 threads compute a 64 x 64 tile of C, 4 x 4 values per thread,
// accumulating in f32 (f64 for double). Plain `__syncthreads` only, so it runs unchanged on wave32
// (RDNA) and wave64 (GCN / CDNA) GPUs.
#include "common.h"

#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 16
#define GEMM_TM 4
#define GEMM_TN 4
#define GEMM_THREADS ((GEMM_BM / GEMM_TM) * (GEMM_BN / GEMM_TN))

template <typename T>
DEVFN void gemm_impl(const T *A, const T *B, T *C, const uint32_t transa, const uint32_t transb, const size_t m,
                     const size_t n, const size_t k, const size_t lda, const size_t ldb, const size_t ldc,
                     const int64_t stride_a, const int64_t stride_b, const int64_t stride_c, const size_t batch) {
    typedef typename Ty<T>::acc Acc;
    __shared__ Acc As[GEMM_BK][GEMM_BM + 1];
    __shared__ Acc Bs[GEMM_BK][GEMM_BN + 1];
    const size_t tid = threadIdx.x;
    const size_t tx = tid % (GEMM_BM / GEMM_TM);
    const size_t ty = tid / (GEMM_BM / GEMM_TM);
    const size_t tiles_m = (m + GEMM_BM - 1) / GEMM_BM;
    const size_t tiles_n = (n + GEMM_BN - 1) / GEMM_BN;
    for (size_t bz = blockIdx.z; bz < batch; bz += gridDim.z) {
        const T *a = A + (int64_t)bz * stride_a;
        const T *b = B + (int64_t)bz * stride_b;
        T *c = C + (int64_t)bz * stride_c;
        for (size_t tj = blockIdx.y; tj < tiles_n; tj += gridDim.y) {
            for (size_t ti = blockIdx.x; ti < tiles_m; ti += gridDim.x) {
                const size_t i0 = ti * GEMM_BM;
                const size_t j0 = tj * GEMM_BN;
                Acc acc[GEMM_TM][GEMM_TN];
                for (int r = 0; r < GEMM_TM; ++r) {
                    for (int s = 0; s < GEMM_TN; ++s) {
                        acc[r][s] = (Acc)0;
                    }
                }
                for (size_t p0 = 0; p0 < k; p0 += GEMM_BK) {
                    // op(A)(i0 .. i0+BM, p0 .. p0+BK) -> As[p][i], reading along the contiguous dim.
                    for (size_t e = tid; e < GEMM_BM * GEMM_BK; e += GEMM_THREADS) {
                        const size_t ii = transa ? e / GEMM_BK : e % GEMM_BM;
                        const size_t kk = transa ? e % GEMM_BK : e / GEMM_BM;
                        const size_t i = i0 + ii;
                        const size_t p = p0 + kk;
                        Acc v = (Acc)0;
                        if (i < m && p < k) {
                            v = (Acc)Ty<T>::to(transa ? a[p + i * lda] : a[i + p * lda]);
                        }
                        As[kk][ii] = v;
                    }
                    // op(B)(p0 .. p0+BK, j0 .. j0+BN) -> Bs[p][j].
                    for (size_t e = tid; e < GEMM_BN * GEMM_BK; e += GEMM_THREADS) {
                        const size_t jj = transb ? e % GEMM_BN : e / GEMM_BK;
                        const size_t kk = transb ? e / GEMM_BN : e % GEMM_BK;
                        const size_t j = j0 + jj;
                        const size_t p = p0 + kk;
                        Acc v = (Acc)0;
                        if (j < n && p < k) {
                            v = (Acc)Ty<T>::to(transb ? b[j + p * ldb] : b[p + j * ldb]);
                        }
                        Bs[kk][jj] = v;
                    }
                    __syncthreads();
                    for (int kk = 0; kk < GEMM_BK; ++kk) {
                        Acc av[GEMM_TM];
                        Acc bv[GEMM_TN];
                        for (int r = 0; r < GEMM_TM; ++r) {
                            av[r] = As[kk][tx + r * (GEMM_BM / GEMM_TM)];
                        }
                        for (int s = 0; s < GEMM_TN; ++s) {
                            bv[s] = Bs[kk][ty + s * (GEMM_BN / GEMM_TN)];
                        }
                        for (int r = 0; r < GEMM_TM; ++r) {
                            for (int s = 0; s < GEMM_TN; ++s) {
                                acc[r][s] += av[r] * bv[s];
                            }
                        }
                    }
                    __syncthreads();
                }
                for (int s = 0; s < GEMM_TN; ++s) {
                    const size_t j = j0 + ty + s * (GEMM_BN / GEMM_TN);
                    if (j >= n) {
                        continue;
                    }
                    for (int r = 0; r < GEMM_TM; ++r) {
                        const size_t i = i0 + tx + r * (GEMM_BM / GEMM_TM);
                        if (i < m) {
                            c[i + j * ldc] = Ty<T>::from(acc[r][s]);
                        }
                    }
                }
            }
        }
    }
}

#define GEMM_FNS(T, SFX)                                                                                       \
    extern "C" __global__ void __launch_bounds__(GEMM_THREADS)                                                 \
        gemm_##SFX(const T *A, const T *B, T *C, const uint32_t transa, const uint32_t transb, const size_t m, \
                   const size_t n, const size_t k, const size_t lda, const size_t ldb, const size_t ldc,       \
                   const int64_t stride_a, const int64_t stride_b, const int64_t stride_c,                     \
                   const size_t batch) {                                                                       \
        gemm_impl<T>(A, B, C, transa, transb, m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch);   \
    }                                                                                                          \
    CANDLE_REGISTER(gemm_##SFX)

GEMM_FNS(float, f32)
GEMM_FNS(double, f64)
GEMM_FNS(f16_t, f16)
GEMM_FNS(bf16_t, bf16)
