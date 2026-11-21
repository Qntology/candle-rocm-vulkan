// ROCm HIP instantiations of exact weight-only W8A16 matvec, direct dequantization,
// exact indexed MoE matvec, Quantized Mode A lm_head Stage-1, and planar NVFP4/NVFP8 kernels.
#include "w8a16_impl.h"

#define W8A16_HIP_MV_SHAPES(SFX, DEC, XT_SFX, XT)                                                  \
    W8A16_MVDYN_KERNEL(w8a16_mv1_##SFX##_##XT_SFX##_1x4, DEC, XT, 1, 1, 4)                         \
    W8A16_MVDYN_KERNEL(w8a16_mv1_##SFX##_##XT_SFX##_2x2, DEC, XT, 1, 2, 2)                         \
    W8A16_MVDYN_KERNEL(w8a16_mv1_##SFX##_##XT_SFX##_4x1, DEC, XT, 1, 4, 1)                         \
    W8A16_MVDYN_KERNEL(w8a16_mv8_##SFX##_##XT_SFX##_1x4, DEC, XT, 8, 1, 4)                         \
    W8A16_MVDYN_KERNEL(w8a16_mv8_##SFX##_##XT_SFX##_2x2, DEC, XT, 8, 2, 2)                         \
    W8A16_MVDYN_KERNEL(w8a16_mv8_##SFX##_##XT_SFX##_4x1, DEC, XT, 8, 4, 1)

#define W8A16_HIP_ALL_FOR(SFX, DEC)                                                                \
    W8A16_HIP_MV_SHAPES(SFX, DEC, f32, float)                                                      \
    W8A16_HIP_MV_SHAPES(SFX, DEC, f16, w_f16)                                                      \
    W8A16_HIP_MV_SHAPES(SFX, DEC, bf16, w_bf16)                                                    \
    W8A16_DQ_KERNEL(w8a16_dq_##SFX##_f32, DEC, float)                                              \
    W8A16_DQ_KERNEL(w8a16_dq_##SFX##_f16, DEC, w_f16)                                              \
    W8A16_DQ_KERNEL(w8a16_dq_##SFX##_bf16, DEC, w_bf16)                                            \
    W8A16_MOE_KERNEL(w8a16_moe_##SFX##_f32, DEC, float)                                            \
    W8A16_MOE_KERNEL(w8a16_moe_##SFX##_f16, DEC, w_f16)                                            \
    W8A16_MOE_KERNEL(w8a16_moe_##SFX##_bf16, DEC, w_bf16)                                          \
    W8A16_LM_QUANT_KERNEL(w8a16_lm1_##SFX##_f32, DEC, float, 1)                                    \
    W8A16_LM_QUANT_KERNEL(w8a16_lm1_##SFX##_f16, DEC, w_f16, 1)                                    \
    W8A16_LM_QUANT_KERNEL(w8a16_lm1_##SFX##_bf16, DEC, w_bf16, 1)                                  \
    W8A16_LM_QUANT_KERNEL(w8a16_lm8_##SFX##_f32, DEC, float, 8)                                    \
    W8A16_LM_QUANT_KERNEL(w8a16_lm8_##SFX##_f16, DEC, w_f16, 8)                                    \
    W8A16_LM_QUANT_KERNEL(w8a16_lm8_##SFX##_bf16, DEC, w_bf16, 8)

W8A16_HIP_ALL_FOR(q8_0, DecQ8_0)
W8A16_HIP_ALL_FOR(q4_0, DecQ4_0)
W8A16_HIP_ALL_FOR(q4k, DecQ4K)
W8A16_HIP_ALL_FOR(q5k, DecQ5K)
W8A16_HIP_ALL_FOR(q6k, DecQ6K)

W8A16_CVT2D_KERNEL(w8a16_cvt2d_f16, w_f16)
W8A16_CVT2D_KERNEL(w8a16_cvt2d_bf16, w_bf16)

W8A16_NVFP4_MV_KERNEL(w8a16_nvfp4_mv1_f32, float, 1)
W8A16_NVFP4_MV_KERNEL(w8a16_nvfp4_mv1_f16, w_f16, 1)
W8A16_NVFP4_MV_KERNEL(w8a16_nvfp4_mv1_bf16, w_bf16, 1)
W8A16_NVFP4_MV_KERNEL(w8a16_nvfp4_mv8_f32, float, 8)
W8A16_NVFP4_MV_KERNEL(w8a16_nvfp4_mv8_f16, w_f16, 8)
W8A16_NVFP4_MV_KERNEL(w8a16_nvfp4_mv8_bf16, w_bf16, 8)
W8A16_NVFP4_DQ_KERNEL(w8a16_nvfp4_dq_f32, float)
W8A16_NVFP4_DQ_KERNEL(w8a16_nvfp4_dq_f16, w_f16)
W8A16_NVFP4_DQ_KERNEL(w8a16_nvfp4_dq_bf16, w_bf16)

W8A16_NVFP8_MV_KERNEL(w8a16_nvfp8_mv1_f32, float, 1)
W8A16_NVFP8_MV_KERNEL(w8a16_nvfp8_mv1_f16, w_f16, 1)
W8A16_NVFP8_MV_KERNEL(w8a16_nvfp8_mv1_bf16, w_bf16, 1)
W8A16_NVFP8_MV_KERNEL(w8a16_nvfp8_mv8_f32, float, 8)
W8A16_NVFP8_MV_KERNEL(w8a16_nvfp8_mv8_f16, w_f16, 8)
W8A16_NVFP8_MV_KERNEL(w8a16_nvfp8_mv8_bf16, w_bf16, 8)
W8A16_NVFP8_DQ_KERNEL(w8a16_nvfp8_dq_f32, float)
W8A16_NVFP8_DQ_KERNEL(w8a16_nvfp8_dq_f16, w_f16)
W8A16_NVFP8_DQ_KERNEL(w8a16_nvfp8_dq_bf16, w_bf16)
