// Fused attention kernels for the ROCm backend (implementation: flash_attn_impl.h).
#include "flash_attn_impl.h"

FA_ALL_FOR(f32, f32, float, float)
FA_ALL_FOR(f32, fp8, float, ax_fp8)
FA_ALL_FOR(f16, f16, ax_f16, ax_f16)
FA_ALL_FOR(f16, fp8, ax_f16, ax_fp8)
FA_ALL_FOR(bf16, bf16, ax_bf16, ax_bf16)
FA_ALL_FOR(bf16, fp8, ax_bf16, ax_fp8)

FA_COMBINE_KERNEL(fa_combine_f32, float)
FA_COMBINE_KERNEL(fa_combine_f16, ax_f16)
FA_COMBINE_KERNEL(fa_combine_bf16, ax_bf16)
