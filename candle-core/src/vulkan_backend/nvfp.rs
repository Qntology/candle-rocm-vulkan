//! Native Vulkan compute shaders for planar NVFP4 (E2M1 + OCP E4M3FN block scales + f32 row scale)
//! and planar NVFP8 (OCP E4M3FN + f32 row scale) exact dequantization and decode matvec (`M <= 8`).
//!
//! Bit-exact with `accel::formats` OCP E4M3FN / E2M1 decoders and `kernels/w8a16_impl.h`.

use super::shaders::{grid_2d, kernel, run, GLSL_PACK16_HELPERS};
use super::{VulkanDevice, VulkanStorage};
use crate::{bail, DType, Result};

const NVFP_CODEC: &str = r#"
uint ld_byte(uint byte_off) {
    return (raw[byte_off >> 2u] >> ((byte_off & 3u) * 8u)) & 0xffu;
}
float e4m3_to_f(uint c) {
    uint e = (c >> 3u) & 15u;
    uint m = c & 7u;
    if (e == 15u && m == 7u) { return uintBitsToFloat(0x7fc00000u); }
    float v;
    if (e == 0u) { v = float(m) * 0.001953125; }
    else { v = uintBitsToFloat(((e + 120u) << 23u) | (m << 20u)); }
    if ((c & 0x80u) != 0u) { v = -v; }
    return v;
}
float fp4_to_f(uint c) {
    uint e = (c >> 1u) & 3u;
    uint m = c & 1u;
    float v;
    if (e == 0u) { v = 0.5 * float(m); }
    else { v = uintBitsToFloat(((e + 126u) << 23u) | (m << 22u)); }
    if ((c & 8u) != 0u) { v = -v; }
    return v;
}
"#;

fn dq_source(is_fp4: bool, out_dtype: DType) -> String {
    let (out_decl, store_pair) = match out_dtype {
        DType::F32 => (
            "float",
            "out_buf[elem] = v0; out_buf[elem + 1u] = v1;",
        ),
        DType::F16 => (
            "uint",
            "out_buf[pair] = packHalf2x16(vec2(v0, v1));",
        ),
        DType::BF16 => (
            "uint",
            "out_buf[pair] = pack_bf16x2(v0, v1);",
        ),
        _ => unreachable!(),
    };
    let decode_pair = if is_fp4 {
        r#"
        uint row = pair / half_cols;
        uint col_pair = pair - row * half_cols;
        uint col = col_pair * 2u;
        uint b_idx = col >> 4u;
        float row_g = uintBitsToFloat(raw[(pc.rscale_off >> 2u) + row]);
        uint sb = ld_byte(pc.bscale_off + row * (pc.cols >> 4u) + b_idx);
        float eff = e4m3_to_f(sb) * row_g;
        uint qbyte = ld_byte(row * half_cols + col_pair);
        float v0 = fp4_to_f(qbyte & 0x0fu) * eff;
        float v1 = fp4_to_f(qbyte >> 4u) * eff;
        "#
    } else {
        r#"
        uint row = pair / half_cols;
        uint col_pair = pair - row * half_cols;
        uint col = col_pair * 2u;
        float row_g = uintBitsToFloat(raw[(pc.rscale_off >> 2u) + row]);
        uint qbase = row * pc.cols + col;
        float v0 = e4m3_to_f(ld_byte(qbase)) * row_g;
        float v1 = e4m3_to_f(ld_byte(qbase + 1u)) * row_g;
        "#
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Raw {{ uint raw[]; }};
layout(set = 0, binding = 1) buffer Out {{ {out_decl} out_buf[]; }};
layout(push_constant, std430) uniform PC {{
    uint rows;
    uint cols;
    uint bscale_off;
    uint rscale_off;
}} pc;
{GLSL_PACK16_HELPERS}
{NVFP_CODEC}
void main() {{
    uint half_cols = pc.cols >> 1u;
    uint total_pairs = pc.rows * half_cols;
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint wg_stride = gl_NumWorkGroups.x * gl_NumWorkGroups.y;
    for (uint base_p = wg_id * 256u; base_p < total_pairs; base_p += wg_stride * 256u) {{
        uint pair = base_p + gl_LocalInvocationID.x;
        if (pair >= total_pairs) {{ return; }}
        uint elem = pair * 2u;
        {decode_pair}
        {store_pair}
    }}
}}
"#
    )
}

fn mv_source(is_fp4: bool, x_dtype: DType) -> String {
    let (x_decl, ld_x) = match x_dtype {
        DType::F32 => ("float", "x_buf[idx]"),
        DType::F16 => ("uint", "ld_f16(x_buf[idx >> 1u], idx)"),
        DType::BF16 => ("uint", "ld_bf16(x_buf[idx >> 1u], idx)"),
        _ => unreachable!(),
    };
    let inner_loop = if is_fp4 {
        r#"
        uint nblk = pc.cols >> 4u;
        uint half_cols = pc.cols >> 1u;
        for (uint b = tid; b < nblk; b += 64u) {
            uint sb = ld_byte(pc.bscale_off + row * nblk + b);
            float eff = e4m3_to_f(sb) * row_g;
            uint q_base = row * half_cols + b * 8u;
            uint col0 = b * 16u;
            for (uint pair = 0u; pair < 8u; pair++) {
                uint qb = ld_byte(q_base + pair);
                float w0 = fp4_to_f(qb & 0x0fu) * eff;
                float w1 = fp4_to_f(qb >> 4u) * eff;
                uint j0 = col0 + pair * 2u;
                uint j1 = j0 + 1u;
                for (uint m = 0u; m < pc.m; m++) {
                    uint xb = pc.x_off + m * pc.cols;
                    acc[m] = fma(w0, load_x(xb + j0), acc[m]);
                    acc[m] = fma(w1, load_x(xb + j1), acc[m]);
                }
            }
        }
        "#
    } else {
        r#"
        uint q_base = row * pc.cols;
        for (uint j = tid; j < pc.cols; j += 64u) {
            float w = e4m3_to_f(ld_byte(q_base + j)) * row_g;
            for (uint m = 0u; m < pc.m; m++) {
                acc[m] = fma(w, load_x(pc.x_off + m * pc.cols + j), acc[m]);
            }
        }
        "#
    };
    format!(
        r#"#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer Raw {{ uint raw[]; }};
layout(set = 0, binding = 1) readonly buffer Xb {{ {x_decl} x_buf[]; }};
layout(set = 0, binding = 2) buffer Ob {{ float y_f32[]; }};
layout(push_constant, std430) uniform PC {{
    uint rows;
    uint cols;
    uint m;
    uint x_off;
    uint bscale_off;
    uint rscale_off;
}} pc;
{GLSL_PACK16_HELPERS}
{NVFP_CODEC}
float load_x(uint idx) {{ return {ld_x}; }}
shared float red[64];
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint wg_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint wg_stride = gl_NumWorkGroups.x * gl_NumWorkGroups.y;
    for (uint row = wg_id; row < pc.rows; row += wg_stride) {{
        float row_g = uintBitsToFloat(raw[(pc.rscale_off >> 2u) + row]);
        float acc[8];
        for (uint mi = 0u; mi < 8u; mi++) {{ acc[mi] = 0.0; }}
        {inner_loop}
        for (uint mi = 0u; mi < pc.m; mi++) {{
            red[tid] = acc[mi];
            barrier();
            for (uint s = 32u; s > 0u; s >>= 1u) {{
                if (tid < s) {{ red[tid] += red[tid + s]; }}
                barrier();
            }}
            if (tid == 0u) {{
                y_f32[mi * pc.rows + row] = red[0];
            }}
            barrier();
        }}
    }}
}}
"#
    )
}

fn dtype_tag(dt: DType) -> Result<&'static str> {
    match dt {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => bail!("vulkan nvfp: unsupported dtype {other:?}"),
    }
}

/// Native Vulkan dequantization of planar NVFP4 / NVFP8 into `out_dtype` (`F32` / `F16` / `BF16`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequantize(
    dev: &VulkanDevice,
    raw: &VulkanStorage,
    bscale_off: usize,
    rscale_off: usize,
    is_fp4: bool,
    rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<VulkanStorage> {
    let tag = dtype_tag(out_dtype)?;
    let fmt = if is_fp4 { "nvfp4" } else { "nvfp8" };
    let out = dev.alloc_buffer(rows * cols, out_dtype)?;
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    let key = format!("vk_{fmt}_dq_{tag}");
    let kern = kernel(dev, &key, 2, 16, || dq_source(is_fp4, out_dtype))?;
    let mut push = Vec::with_capacity(16);
    push.extend_from_slice(&(rows as u32).to_le_bytes());
    push.extend_from_slice(&(cols as u32).to_le_bytes());
    push.extend_from_slice(&(bscale_off as u32).to_le_bytes());
    push.extend_from_slice(&(rscale_off as u32).to_le_bytes());
    let total_pairs = rows * (cols / 2);
    let groups = grid_2d(dev, total_pairs.div_ceil(256));
    run(dev, &kern, &[raw, &out], &push, groups)?;
    Ok(out)
}

/// Native Vulkan exact matvec (`M <= 8`) for planar NVFP4 / NVFP8 weights (`F32`/`F16`/`BF16` x).
/// Returns an `F32` storage of shape `[m, rows]` (caller casts to `x.dtype()` if F16/BF16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_f32(
    dev: &VulkanDevice,
    raw: &VulkanStorage,
    bscale_off: usize,
    rscale_off: usize,
    is_fp4: bool,
    rows: usize,
    cols: usize,
    x: &VulkanStorage,
    x_off: usize,
    m: usize,
) -> Result<VulkanStorage> {
    if m == 0 || m > 8 {
        bail!("vulkan nvfp matvec: m={m} must be in 1..=8");
    }
    let tag = dtype_tag(x.dtype)?;
    let fmt = if is_fp4 { "nvfp4" } else { "nvfp8" };
    let out = dev.alloc_buffer(m * rows, DType::F32)?;
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    let key = format!("vk_{fmt}_mv_{tag}");
    let xdt = x.dtype;
    let kern = kernel(dev, &key, 3, 24, || mv_source(is_fp4, xdt))?;
    let mut push = Vec::with_capacity(24);
    push.extend_from_slice(&(rows as u32).to_le_bytes());
    push.extend_from_slice(&(cols as u32).to_le_bytes());
    push.extend_from_slice(&(m as u32).to_le_bytes());
    push.extend_from_slice(&(x_off as u32).to_le_bytes());
    push.extend_from_slice(&(bscale_off as u32).to_le_bytes());
    push.extend_from_slice(&(rscale_off as u32).to_le_bytes());
    let groups = grid_2d(dev, rows);
    run(dev, &kern, &[raw, x, &out], &push, groups)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        use naga::front::glsl::{Frontend, Options};
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = Frontend::default()
            .parse(&Options::from(naga::ShaderStage::Compute), src)
            .unwrap_or_else(|e| panic!("nvfp glsl parse failed: {e:?}\n{src}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("nvfp glsl validation failed: {e:?}"));
    }

    #[test]
    fn nvfp_shaders_validate() {
        for is_fp4 in [false, true] {
            for dt in [DType::F32, DType::F16, DType::BF16] {
                validate(&dq_source(is_fp4, dt));
                validate(&mv_source(is_fp4, dt));
            }
        }
    }
}
