//! Shared planning & NVRTC CUDA source synthesis for dedicated Vision Encoder fused ops
//! (`kernels/vis_ops_impl.h` / `kernels/vis_ops.hip`).

use super::sources::{strip_includes, ACCEL_COMMON_H, ACCEL_MMA_H};

pub const VIS_OPS_IMPL_H: &str = include_str!("../../kernels/vis_ops_impl.h");

/// Workgroup size for Chan-Welford LayerNorm & Spatial Merge kernels.
pub const VIS_LN_NT: usize = 256;
pub const VIS_LN_SMEM_BYTES: usize = 3 * VIS_LN_NT * 4;

/// Workgroup size for fused Vision QKV + 2D RoPE kernel.
pub const VIS_QKV_NT: usize = 128;

/// GELU variant for [`crate::vis_ops::fused_bias_gelu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluMode {
    /// Exact error-function GELU: `0.5 * x * (1 + erf(x / sqrt(2)))` (SigLIP / ViT).
    ExactErf = 0,
    /// PyTorch tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` (SigLIP2 / Gemma / Qwen-VL).
    PytorchTanh = 1,
    /// QuickGELU: `x * sigmoid(1.702 * x)` (OpenAI CLIP / Qwen2-VL).
    QuickGelu = 2,
}

/// Spatial merger normalization mode for [`crate::vis_ops::fused_spatial_merge_norm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialMergeMode {
    /// LayerNorm each `[D]` patch before concatenating `M x M` neighbors into `[M*M*D]` (Qwen2.5-VL / Qwen3-VL).
    PreMergeNorm = 0,
    /// Concatenate `M x M` neighbors into `[M*M*D]`, then apply single-pass Chan-Welford LayerNorm (InternVL / Qwen3-VL deepstack).
    PostMergeNorm = 1,
    /// Average `M x M` spatial neighbors into `[D]`, then apply Chan-Welford LayerNorm.
    AvgPoolNorm = 2,
}

fn ctype(sfx: &str) -> Option<&'static str> {
    match sfx {
        "f32" => Some("float"),
        "f16" => Some("ax_f16"),
        "bf16" => Some("ax_bf16"),
        _ => None,
    }
}

/// Kernel instantiation line for NVRTC compilation.
pub fn vis_instantiation(name: &str) -> Option<String> {
    if let Some(sfx) = name.strip_prefix("vis_u8_resize_norm_patchify_") {
        return Some(format!("VIS_RESIZE_PATCH_KERNEL({name}, {})", ctype(sfx)?));
    }
    if let Some(sfx) = name.strip_prefix("vis_res_ln_ls_f32res_") {
        return Some(format!(
            "VIS_RES_LN_LS_KERNEL({name}, {}, float)",
            ctype(sfx)?
        ));
    }
    if let Some(sfx) = name.strip_prefix("vis_res_ln_ls_") {
        let ty = ctype(sfx)?;
        return Some(format!("VIS_RES_LN_LS_KERNEL({name}, {ty}, {ty})"));
    }
    if let Some(sfx) = name.strip_prefix("vis_bias_gelu_") {
        return Some(format!("VIS_BIAS_GELU_KERNEL({name}, {})", ctype(sfx)?));
    }
    if let Some(sfx) = name.strip_prefix("vis_qkv_2d_rope_") {
        return Some(format!("VIS_QKV_2D_ROPE_KERNEL({name}, {})", ctype(sfx)?));
    }
    if let Some(sfx) = name.strip_prefix("vis_spatial_merge_norm_") {
        return Some(format!("VIS_SPATIAL_MERGE_KERNEL({name}, {})", ctype(sfx)?));
    }
    if let Some(sfx) = name.strip_prefix("vis_scatter_embeds_") {
        return Some(format!(
            "VIS_SCATTER_EMBEDS_KERNEL({name}, {})",
            ctype(sfx)?
        ));
    }
    if let Some(sfx) = name.strip_prefix("vis_patch_add_interp_pos2d_") {
        return Some(format!(
            "VIS_PATCH_ADD_POS2D_KERNEL({name}, {})",
            ctype(sfx)?
        ));
    }
    None
}

/// Full headerless NVRTC CUDA source for one `vis_ops` kernel.
pub fn vis_cuda_source(name: &str) -> Option<String> {
    let inst = vis_instantiation(name)?;
    Some(format!(
        "{}\n{}\n{}\n{inst}\n",
        strip_includes(ACCEL_COMMON_H),
        strip_includes(ACCEL_MMA_H),
        strip_includes(VIS_OPS_IMPL_H),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_source_has_no_includes_and_matches_hip() {
        let hip = include_str!("../../kernels/vis_ops.hip");
        let hip_lines: Vec<&str> = hip
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("VIS_"))
            .collect();
        assert!(!hip_lines.is_empty());
        for line in &hip_lines {
            let name = line
                .split('(')
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
                .trim();
            let inst = vis_instantiation(name).unwrap_or_else(|| panic!("missing inst for {name}"));
            assert_eq!(&inst, line);
            let src = vis_cuda_source(name).unwrap();
            assert!(!src.contains("#include \""));
            assert!(!src.contains("#pragma once"));
        }
    }
}
