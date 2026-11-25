// Fused attention kernels for the ROCm backend (implementation: flash_attn_impl.h).
#include "flash_attn_impl.h"

FA_ALL_FOR(f32, f32, float, float)
FA_ALL_FOR(f32, fp8, float, ax_fp8)
// exact K/V stored in 16 bits with f32 queries (converted in the kernels)
FA_ALL_FOR(f32, f16, float, ax_f16)
FA_ALL_FOR(f32, bf16, float, ax_bf16)
FA_ALL_FOR(f16, f16, ax_f16, ax_f16)
FA_ALL_FOR(f16, fp8, ax_f16, ax_fp8)
FA_ALL_FOR(bf16, bf16, ax_bf16, ax_bf16)
FA_ALL_FOR(bf16, fp8, ax_bf16, ax_fp8)

FA_COMBINE_KERNEL(fa_combine_f32, float)
FA_COMBINE_KERNEL(fa_combine_f16, ax_f16)
FA_COMBINE_KERNEL(fa_combine_bf16, ax_bf16)

FA_COMBINE_TREE_KERNEL(fa_combine_tree_f32, float)
FA_COMBINE_TREE_KERNEL(fa_combine_tree_f16, ax_f16)
FA_COMBINE_TREE_KERNEL(fa_combine_tree_bf16, ax_bf16)

FA_TILE_BT_KERNEL(fa_bt_d64_f32, float, float, 32, 32, 64)
FA_TILE_BT_KERNEL(fa_bt_d128_f32, float, float, 16, 32, 128)
FA_TILE_BT_KERNEL(fa_bt_d256_f32, float, float, 8, 16, 256)
FA_TILE_BT_KERNEL(fa_bt_d64_f16, ax_f16, ax_f16, 32, 32, 64)
FA_TILE_BT_KERNEL(fa_bt_d128_f16, ax_f16, ax_f16, 16, 32, 128)
FA_TILE_BT_KERNEL(fa_bt_d256_f16, ax_f16, ax_f16, 8, 16, 256)
FA_TILE_BT_KERNEL(fa_bt_d64_bf16, ax_bf16, ax_bf16, 32, 32, 64)
FA_TILE_BT_KERNEL(fa_bt_d128_bf16, ax_bf16, ax_bf16, 16, 32, 128)
FA_TILE_BT_KERNEL(fa_bt_d256_bf16, ax_bf16, ax_bf16, 8, 16, 256)
