// Dedicated Vision Encoder Fused Ops for the ROCm backend (implementation: vis_ops_impl.h).
#include "vis_ops_impl.h"

VIS_RESIZE_PATCH_KERNEL(vis_u8_resize_norm_patchify_f32, float)
VIS_RESIZE_PATCH_KERNEL(vis_u8_resize_norm_patchify_f16, ax_f16)
VIS_RESIZE_PATCH_KERNEL(vis_u8_resize_norm_patchify_bf16, ax_bf16)

VIS_RES_LN_LS_KERNEL(vis_res_ln_ls_f32, float, float)
VIS_RES_LN_LS_KERNEL(vis_res_ln_ls_f16, ax_f16, ax_f16)
VIS_RES_LN_LS_KERNEL(vis_res_ln_ls_bf16, ax_bf16, ax_bf16)
VIS_RES_LN_LS_KERNEL(vis_res_ln_ls_f32res_f16, ax_f16, float)
VIS_RES_LN_LS_KERNEL(vis_res_ln_ls_f32res_bf16, ax_bf16, float)

VIS_BIAS_GELU_KERNEL(vis_bias_gelu_f32, float)
VIS_BIAS_GELU_KERNEL(vis_bias_gelu_f16, ax_f16)
VIS_BIAS_GELU_KERNEL(vis_bias_gelu_bf16, ax_bf16)

VIS_QKV_2D_ROPE_KERNEL(vis_qkv_2d_rope_f32, float)
VIS_QKV_2D_ROPE_KERNEL(vis_qkv_2d_rope_f16, ax_f16)
VIS_QKV_2D_ROPE_KERNEL(vis_qkv_2d_rope_bf16, ax_bf16)

VIS_SPATIAL_MERGE_KERNEL(vis_spatial_merge_norm_f32, float)
VIS_SPATIAL_MERGE_KERNEL(vis_spatial_merge_norm_f16, ax_f16)
VIS_SPATIAL_MERGE_KERNEL(vis_spatial_merge_norm_bf16, ax_bf16)

VIS_SCATTER_EMBEDS_KERNEL(vis_scatter_embeds_f32, float)
VIS_SCATTER_EMBEDS_KERNEL(vis_scatter_embeds_f16, ax_f16)
VIS_SCATTER_EMBEDS_KERNEL(vis_scatter_embeds_bf16, ax_bf16)

VIS_PATCH_ADD_POS2D_KERNEL(vis_patch_add_interp_pos2d_f32, float)
VIS_PATCH_ADD_POS2D_KERNEL(vis_patch_add_interp_pos2d_f16, ax_f16)
VIS_PATCH_ADD_POS2D_KERNEL(vis_patch_add_interp_pos2d_bf16, ax_bf16)
