//! Fused decoder compute shaders on the Vulkan backend:
//! - `res_rmsnorm` / `res_rmsnorm_inplace` / `rmsnorm_fused`
//! - `silu_mul` (zero-copy strided narrow slices)
//! - `gdn_gated_rms_norm` (`x * rsqrt(mean(x^2)+eps) * w * silu(z)`)
//! - `rope_kv_write` (fused QK head RMSNorm + partial/full 1D/3D-M RoPE + KV cache direct write)
//!
//! All math executes in IEEE FP32. F16/BF16 outputs use race-free paired `uint` writes
//! (`packHalf2x16` / RNE `pack_bf16x2`) without requiring `GL_EXT_shader_16bit_storage`.

use super::shaders::{self, kernel, run, GLSL_PACK16_HELPERS};
use super::{VulkanDevice, VulkanStorage};
use crate::accel::decoder_ops::{RopeKvLaunchArgs, ROPE_MODE_ADJACENT, ROPE_NORM_PLUS_ONE};
use crate::{bail, DType, Result};

const MAX_GROUPS: usize = 65535;

fn fits(n: usize) -> Result<u32> {
    if n > u32::MAX as usize {
        bail!("vulkan decoder_ops: index {n} exceeds 32-bit kernel range")
    }
    Ok(n as u32)
}

fn dtype_tag(dt: DType) -> Result<&'static str> {
    match dt {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => bail!("vulkan decoder_ops: unsupported dtype {other:?}"),
    }
}

const REDUCE_256: &str = r#"
shared float red[256];
float wg_sum(float v) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}
"#;

const REDUCE_128: &str = r#"
shared float red[128];
float wg_sum128(float v) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}
"#;

// ---------------------------------------------------------------------------
// 1. Residual + RMSNorm (out-of-place & in-place)
// ---------------------------------------------------------------------------
// prm: 0 rows, 1 d, 2 x_off, 3 dl_off, 4 w_off, 5 has_delta, 6 write_res, 7 plus_one, 8 eps_bits

fn res_rmsnorm_f32_src() -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Xb {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer Db {{ float dl[]; }};
layout(set = 0, binding = 2) readonly buffer Wb {{ float w[]; }};
layout(set = 0, binding = 3) buffer Rb {{ float r_out[]; }};
layout(set = 0, binding = 4) buffer Yb {{ float y_out[]; }};
layout(set = 0, binding = 5) readonly buffer Pb {{ uint prm[]; }};
{REDUCE_256}
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1];
    uint x_off = prm[2]; uint dl_off = prm[3]; uint w_off = prm[4];
    bool has_delta = prm[5] != 0u;
    bool write_res = prm[6] != 0u;
    bool plus_one = prm[7] != 0u;
    float eps = uintBitsToFloat(prm[8]);
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {{
        uint xb = x_off + row * D;
        uint db = dl_off + row * D;
        uint ob = row * D;
        float sq = 0.0;
        for (uint c = t; c < D; c += 256u) {{
            float v = x[xb + c];
            if (has_delta) {{ v += dl[db + c]; }}
            if (write_res) {{ r_out[ob + c] = v; }}
            sq += v * v;
        }}
        float inv = inversesqrt(wg_sum(sq) / float(D) + eps);
        for (uint c = t; c < D; c += 256u) {{
            float v;
            if (write_res) {{
                v = r_out[ob + c];
            }} else {{
                v = x[xb + c];
                if (has_delta) {{ v += dl[db + c]; }}
            }}
            float wf = w[w_off + c];
            if (plus_one) {{ wf = 1.0 + wf; }}
            y_out[ob + c] = (v * inv) * wf;
        }}
    }}
}}
"#
    )
}

fn res_rmsnorm_p16_src(is_bf16: bool) -> String {
    let unpack = if is_bf16 {
        "vec2(uintBitsToFloat(v << 16u), uintBitsToFloat(v & 0xffff0000u))"
    } else {
        "unpackHalf2x16(v)"
    };
    let pack = if is_bf16 {
        "pack_bf16x2(a, b)"
    } else {
        "packHalf2x16(vec2(a, b))"
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Xb {{ uint x[]; }};
layout(set = 0, binding = 1) readonly buffer Db {{ uint dl[]; }};
layout(set = 0, binding = 2) readonly buffer Wb {{ float w[]; }};
layout(set = 0, binding = 3) buffer Rb {{ uint r_out[]; }};
layout(set = 0, binding = 4) buffer Yb {{ uint y_out[]; }};
layout(set = 0, binding = 5) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
{REDUCE_256}
vec2 unpack2(uint v) {{ return {unpack}; }}
uint pack2(float a, float b) {{ return {pack}; }}
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1]; uint dp = D >> 1u;
    uint x_off = prm[2] >> 1u; uint dl_off = prm[3] >> 1u; uint w_off = prm[4];
    bool has_delta = prm[5] != 0u;
    bool write_res = prm[6] != 0u;
    bool plus_one = prm[7] != 0u;
    float eps = uintBitsToFloat(prm[8]);
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {{
        uint xb = x_off + row * dp;
        uint db = dl_off + row * dp;
        uint ob = row * dp;
        float sq = 0.0;
        for (uint p = t; p < dp; p += 256u) {{
            vec2 v = unpack2(x[xb + p]);
            if (has_delta) {{ v += unpack2(dl[db + p]); }}
            if (write_res) {{ r_out[ob + p] = pack2(v.x, v.y); }}
            sq += v.x * v.x + v.y * v.y;
        }}
        float inv = inversesqrt(wg_sum(sq) / float(D) + eps);
        for (uint p = t; p < dp; p += 256u) {{
            vec2 v;
            if (write_res) {{
                v = unpack2(r_out[ob + p]);
            }} else {{
                v = unpack2(x[xb + p]);
                if (has_delta) {{ v += unpack2(dl[db + p]); }}
            }}
            uint c = p * 2u;
            float w0 = w[w_off + c];
            float w1 = w[w_off + c + 1u];
            if (plus_one) {{ w0 = 1.0 + w0; w1 = 1.0 + w1; }}
            y_out[ob + p] = pack2((v.x * inv) * w0, (v.y * inv) * w1);
        }}
    }}
}}
"#
    )
}

// ---------------------------------------------------------------------------
// 2. Fused SwiGLU `silu(gate) * up`
// ---------------------------------------------------------------------------
// prm: 0 rows, 1 inter_dim, 2 g_off, 3 g_stride, 4 u_off, 5 u_stride

fn silu_mul_f32_src() -> String {
    r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Gb { float gate[]; };
layout(set = 0, binding = 1) readonly buffer Ub { float up[]; };
layout(set = 0, binding = 2) buffer Ob { float outv[]; };
layout(set = 0, binding = 3) readonly buffer Pb { uint prm[]; };
void main() {
    uint rows = prm[0]; uint inter = prm[1];
    uint g_off = prm[2]; uint g_st = prm[3];
    uint u_off = prm[4]; uint u_st = prm[5];
    uint total = rows * inter;
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint idx = gl_GlobalInvocationID.x; idx < total; idx += stride) {
        uint r = idx / inter;
        uint c = idx - r * inter;
        float g = gate[g_off + r * g_st + c];
        float u = up[u_off + r * u_st + c];
        outv[idx] = (g / (1.0 + exp(-g))) * u;
    }
}
"#
    .to_string()
}

fn silu_mul_p16_src(is_bf16: bool) -> String {
    let ld = if is_bf16 { "ld_bf16" } else { "ld_f16" };
    let pack = if is_bf16 {
        "pack_bf16x2(y0, y1)"
    } else {
        "packHalf2x16(vec2(y0, y1))"
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Gb {{ uint gate[]; }};
layout(set = 0, binding = 1) readonly buffer Ub {{ uint up[]; }};
layout(set = 0, binding = 2) buffer Ob {{ uint outv[]; }};
layout(set = 0, binding = 3) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
void main() {{
    uint rows = prm[0]; uint inter = prm[1]; uint hp = inter >> 1u;
    uint g_off = prm[2]; uint g_st = prm[3];
    uint u_off = prm[4]; uint u_st = prm[5];
    uint total = rows * hp;
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint idx = gl_GlobalInvocationID.x; idx < total; idx += stride) {{
        uint r = idx / hp;
        uint c = (idx - r * hp) * 2u;
        uint gi0 = g_off + r * g_st + c;
        uint gi1 = gi0 + 1u;
        uint ui0 = u_off + r * u_st + c;
        uint ui1 = ui0 + 1u;
        float g0 = {ld}(gate[gi0 >> 1u], gi0);
        float g1 = {ld}(gate[gi1 >> 1u], gi1);
        float u0 = {ld}(up[ui0 >> 1u], ui0);
        float u1 = {ld}(up[ui1 >> 1u], ui1);
        float y0 = (g0 / (1.0 + exp(-g0))) * u0;
        float y1 = (g1 / (1.0 + exp(-g1))) * u1;
        outv[idx] = {pack};
    }}
}}
"#
    )
}

// ---------------------------------------------------------------------------
// 3. Fused Gated RMSNorm: `x / sqrt(mean(x^2)+eps) * weight * silu(z)`
// ---------------------------------------------------------------------------
// prm: 0 rows, 1 d, 2 x_off, 3 z_off, 4 w_off, 5 eps_bits

fn gated_rmsnorm_f32_src() -> String {
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Xb {{ float x[]; }};
layout(set = 0, binding = 1) readonly buffer Zb {{ float z[]; }};
layout(set = 0, binding = 2) readonly buffer Wb {{ float w[]; }};
layout(set = 0, binding = 3) buffer Yb {{ float y[]; }};
layout(set = 0, binding = 4) readonly buffer Pb {{ uint prm[]; }};
{REDUCE_256}
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1];
    uint x_off = prm[2]; uint z_off = prm[3]; uint w_off = prm[4];
    float eps = uintBitsToFloat(prm[5]);
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {{
        uint xb = x_off + row * D;
        uint zb = z_off + row * D;
        uint ob = row * D;
        float sq = 0.0;
        for (uint c = t; c < D; c += 256u) {{
            float v = x[xb + c];
            sq += v * v;
        }}
        float inv = inversesqrt(wg_sum(sq) / float(D) + eps);
        for (uint c = t; c < D; c += 256u) {{
            float zv = z[zb + c];
            float sz = zv / (1.0 + exp(-zv));
            y[ob + c] = (x[xb + c] * inv) * w[w_off + c] * sz;
        }}
    }}
}}
"#
    )
}

fn gated_rmsnorm_p16_src(is_bf16: bool) -> String {
    let ld = if is_bf16 { "ld_bf16" } else { "ld_f16" };
    let pack = if is_bf16 {
        "pack_bf16x2(y0, y1)"
    } else {
        "packHalf2x16(vec2(y0, y1))"
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Xb {{ uint x[]; }};
layout(set = 0, binding = 1) readonly buffer Zb {{ uint z[]; }};
layout(set = 0, binding = 2) readonly buffer Wb {{ float w[]; }};
layout(set = 0, binding = 3) buffer Yb {{ uint y[]; }};
layout(set = 0, binding = 4) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
{REDUCE_256}
void main() {{
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1]; uint dp = D >> 1u;
    uint x_off = prm[2]; uint z_off = prm[3]; uint w_off = prm[4];
    float eps = uintBitsToFloat(prm[5]);
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {{
        uint xb = x_off + row * D;
        uint zb = z_off + row * D;
        uint ob = row * dp;
        float sq = 0.0;
        for (uint p = t; p < dp; p += 256u) {{
            uint i0 = xb + p * 2u;
            uint i1 = i0 + 1u;
            float x0 = {ld}(x[i0 >> 1u], i0);
            float x1 = {ld}(x[i1 >> 1u], i1);
            sq += x0 * x0 + x1 * x1;
        }}
        float inv = inversesqrt(wg_sum(sq) / float(D) + eps);
        for (uint p = t; p < dp; p += 256u) {{
            uint c = p * 2u;
            uint xi0 = xb + c; uint xi1 = xi0 + 1u;
            uint zi0 = zb + c; uint zi1 = zi0 + 1u;
            float x0 = {ld}(x[xi0 >> 1u], xi0);
            float x1 = {ld}(x[xi1 >> 1u], xi1);
            float z0 = {ld}(z[zi0 >> 1u], zi0);
            float z1 = {ld}(z[zi1 >> 1u], zi1);
            float y0 = (x0 * inv) * w[w_off + c] * (z0 / (1.0 + exp(-z0)));
            float y1 = (x1 * inv) * w[w_off + c + 1u] * (z1 / (1.0 + exp(-z1)));
            y[ob + p] = {pack};
        }}
    }}
}}
"#
    )
}

// ---------------------------------------------------------------------------
// 4. Fused QK Head RMSNorm + Partial/Full RoPE + KV Cache Direct Write
// ---------------------------------------------------------------------------
// Bindings: 0 q_in, 1 k_in, 2 v_in, 3 qnw, 4 knw, 5 cos, 6 sin, 7 q_out, 8 k_cache, 9 v_cache, 10 prm
// prm layout (34 uints):
//   0 b, 1 t, 2 hq, 3 hkv, 4 d, 5 rotary_dim, 6 past_len, 7 eps_bits, 8 flags,
//   9 has_qnw, 10 has_knw,
//   11 q_off, 12 k_off, 13 v_off, 14 qnw_off, 15 knw_off, 16 cos_off, 17 sin_off, 18 kc_off, 19 vc_off,
//   20-22 q_st (b, h, t), 23-25 qo_st (b, h, t), 26-28 k_st (b, h, t), 29-31 v_st (b, h, t),
//   32-34 kc_st (b, h, s), 35-36 cos_st (b, t)

fn rope_kv_write_src(dtype: DType) -> String {
    let (decl_t, ld_q, ld_k, ld_v, store_q, store_k, store_v) = match dtype {
        DType::F32 => (
            "float",
            "q_in[i]",
            "k_in[i]",
            "v_in[i]",
            "for (uint i = tid; i < D; i += 128u) { q_out[qo_base + i] = sh_out[i]; }",
            "for (uint i = tid; i < D; i += 128u) { k_cache[kc_base + i] = sh_out[i]; }",
            "for (uint i = tid; i < D; i += 128u) { v_cache[vc_base + i] = v_in[v_base + i]; }",
        ),
        DType::F16 => (
            "uint",
            "ld_f16(q_in[i >> 1u], i)",
            "ld_f16(k_in[i >> 1u], i)",
            "ld_f16(v_in[i >> 1u], i)",
            "for (uint p = tid; p < (D >> 1u); p += 128u) { q_out[(qo_base >> 1u) + p] = packHalf2x16(vec2(sh_out[2u * p], sh_out[2u * p + 1u])); }",
            "for (uint p = tid; p < (D >> 1u); p += 128u) { k_cache[(kc_base >> 1u) + p] = packHalf2x16(vec2(sh_out[2u * p], sh_out[2u * p + 1u])); }",
            "for (uint p = tid; p < (D >> 1u); p += 128u) { uint i0 = v_base + 2u * p; float a = ld_f16(v_in[i0 >> 1u], i0); float b = ld_f16(v_in[(i0 + 1u) >> 1u], i0 + 1u); v_cache[(vc_base >> 1u) + p] = packHalf2x16(vec2(a, b)); }",
        ),
        DType::BF16 => (
            "uint",
            "ld_bf16(q_in[i >> 1u], i)",
            "ld_bf16(k_in[i >> 1u], i)",
            "ld_bf16(v_in[i >> 1u], i)",
            "for (uint p = tid; p < (D >> 1u); p += 128u) { q_out[(qo_base >> 1u) + p] = pack_bf16x2(sh_out[2u * p], sh_out[2u * p + 1u]); }",
            "for (uint p = tid; p < (D >> 1u); p += 128u) { k_cache[(kc_base >> 1u) + p] = pack_bf16x2(sh_out[2u * p], sh_out[2u * p + 1u]); }",
            "for (uint p = tid; p < (D >> 1u); p += 128u) { uint i0 = v_base + 2u * p; float a = ld_bf16(v_in[i0 >> 1u], i0); float b = ld_bf16(v_in[(i0 + 1u) >> 1u], i0 + 1u); v_cache[(vc_base >> 1u) + p] = pack_bf16x2(a, b); }",
        ),
        _ => unreachable!(),
    };
    let _ = ld_v;
    format!(
        r#"#version 450
layout(local_size_x = 128) in;
layout(set = 0, binding = 0) readonly buffer Qib {{ {decl_t} q_in[]; }};
layout(set = 0, binding = 1) readonly buffer Kib {{ {decl_t} k_in[]; }};
layout(set = 0, binding = 2) readonly buffer Vib {{ {decl_t} v_in[]; }};
layout(set = 0, binding = 3) readonly buffer Qnb {{ float qnw[]; }};
layout(set = 0, binding = 4) readonly buffer Knb {{ float knw[]; }};
layout(set = 0, binding = 5) readonly buffer Csb {{ float cos_t[]; }};
layout(set = 0, binding = 6) readonly buffer Snb {{ float sin_t[]; }};
layout(set = 0, binding = 7) buffer Qob {{ {decl_t} q_out[]; }};
layout(set = 0, binding = 8) buffer Kcb {{ {decl_t} k_cache[]; }};
layout(set = 0, binding = 9) buffer Vcb {{ {decl_t} v_cache[]; }};
layout(set = 0, binding = 10) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
{REDUCE_128}
shared float sh_head[512];
shared float sh_out[512];
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint head_id = gl_WorkGroupID.x;
    uint B = prm[0]; uint T = prm[1]; uint Hq = prm[2]; uint Hkv = prm[3];
    uint D = prm[4]; uint rot = prm[5]; uint half_rot = rot >> 1u;
    uint past_len = prm[6];
    float eps = uintBitsToFloat(prm[7]);
    uint flags = prm[8];
    bool adjacent = (flags & 1u) == {ROPE_MODE_ADJACENT}u;
    bool plus_one = (flags & {ROPE_NORM_PLUS_ONE}u) != 0u;
    bool has_qnw = prm[9] != 0u;
    bool has_knw = prm[10] != 0u;
    bool is_q = head_id < Hq;
    uint h = is_q ? head_id : (head_id - Hq);
    if (!is_q && h >= Hkv) {{ return; }}
    uint total_bt = B * T;
    for (uint bt = gl_WorkGroupID.y; bt < total_bt; bt += gl_NumWorkGroups.y) {{
        uint bi = bt / T;
        uint ti = bt - bi * T;
        uint cs_base = bi * prm[35] + ti * prm[36];
        if (is_q) {{
            uint q_base = prm[11] + bi * prm[20] + h * prm[21] + ti * prm[22];
            uint qo_base = bi * prm[23] + h * prm[24] + ti * prm[25];
            float sq = 0.0;
            for (uint d = tid; d < D; d += 128u) {{
                uint i = q_base + d;
                float v = {ld_q};
                sh_head[d] = v;
                sq += v * v;
            }}
            if (has_qnw) {{
                float inv = inversesqrt(wg_sum128(sq) / float(D) + eps);
                for (uint d = tid; d < D; d += 128u) {{
                    float wf = qnw[prm[14] + d];
                    if (plus_one) {{ wf = 1.0 + wf; }}
                    sh_head[d] = (sh_head[d] * inv) * wf;
                }}
            }}
            barrier();
            for (uint p = tid; p < half_rot; p += 128u) {{
                float c = cos_t[prm[16] + cs_base + p];
                float s = sin_t[prm[17] + cs_base + p];
                if (!adjacent) {{
                    float x0 = sh_head[p];
                    float x1 = sh_head[p + half_rot];
                    sh_out[p] = x0 * c - x1 * s;
                    sh_out[p + half_rot] = x1 * c + x0 * s;
                }} else {{
                    float x0 = sh_head[2u * p];
                    float x1 = sh_head[2u * p + 1u];
                    sh_out[2u * p] = x0 * c - x1 * s;
                    sh_out[2u * p + 1u] = x1 * c + x0 * s;
                }}
            }}
            for (uint d = rot + tid; d < D; d += 128u) {{
                sh_out[d] = sh_head[d];
            }}
            barrier();
            {store_q}
        }} else {{
            uint k_base = prm[12] + bi * prm[26] + h * prm[27] + ti * prm[28];
            uint v_base = prm[13] + bi * prm[29] + h * prm[30] + ti * prm[31];
            uint kv_step = bi * prm[32] + h * prm[33] + (past_len + ti) * prm[34];
            uint kc_base = prm[18] + kv_step;
            uint vc_base = prm[19] + kv_step;
            float sq = 0.0;
            for (uint d = tid; d < D; d += 128u) {{
                uint i = k_base + d;
                float v = {ld_k};
                sh_head[d] = v;
                sq += v * v;
            }}
            if (has_knw) {{
                float inv = inversesqrt(wg_sum128(sq) / float(D) + eps);
                for (uint d = tid; d < D; d += 128u) {{
                    float wf = knw[prm[15] + d];
                    if (plus_one) {{ wf = 1.0 + wf; }}
                    sh_head[d] = (sh_head[d] * inv) * wf;
                }}
            }}
            barrier();
            for (uint p = tid; p < half_rot; p += 128u) {{
                float c = cos_t[prm[16] + cs_base + p];
                float s = sin_t[prm[17] + cs_base + p];
                if (!adjacent) {{
                    float x0 = sh_head[p];
                    float x1 = sh_head[p + half_rot];
                    sh_out[p] = x0 * c - x1 * s;
                    sh_out[p + half_rot] = x1 * c + x0 * s;
                }} else {{
                    float x0 = sh_head[2u * p];
                    float x1 = sh_head[2u * p + 1u];
                    sh_out[2u * p] = x0 * c - x1 * s;
                    sh_out[2u * p + 1u] = x1 * c + x0 * s;
                }}
            }}
            for (uint d = rot + tid; d < D; d += 128u) {{
                sh_out[d] = sh_head[d];
            }}
            barrier();
            {store_k}
            {store_v}
        }}
        barrier();
    }}
}}
"#
    )
}

pub(crate) fn vk_fused_ready(dev: &VulkanDevice) -> bool {
    shaders::exec_mode() != shaders::ExecMode::Host
        && dev.ctx().native_ok.load(std::sync::atomic::Ordering::Relaxed)
        && dev.ctx().device_type != ash::vk::PhysicalDeviceType::CPU
}

impl VulkanStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn res_rmsnorm(
        &self,
        x_off: usize,
        delta: Option<(&Self, usize)>,
        weight_f32: &Self,
        w_off: usize,
        rows: usize,
        d: usize,
        eps: f32,
        plus_one: bool,
        write_residual: bool,
    ) -> Result<(Option<Self>, Self)> {
        let dev = &self.device;
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        let n = rows * d;
        let r_out = if write_residual {
            Some(dev.alloc_buffer(n, dtype)?)
        } else {
            None
        };
        let y_out = dev.alloc_buffer(n, dtype)?;
        if rows == 0 || d == 0 {
            return Ok((r_out, y_out));
        }
        let (dl_s, dl_off, has_delta) = match delta {
            Some((s, o)) => (s, o, 1u32),
            None => (self, 0, 0u32),
        };
        if dtype != DType::F32 && (d % 2 != 0 || x_off % 2 != 0 || (has_delta != 0 && dl_off % 2 != 0)) {
            bail!("vulkan res_rmsnorm {tag}: requires even d/offset for packed 16-bit write")
        }
        fits(x_off + n)?;
        fits(dl_off + n)?;
        fits(w_off + d)?;
        let key = format!("dec_res_rmsnorm_{tag}");
        let kern = kernel(dev, &key, 6, 0, || match dtype {
            DType::F32 => res_rmsnorm_f32_src(),
            DType::F16 => res_rmsnorm_p16_src(false),
            DType::BF16 => res_rmsnorm_p16_src(true),
            _ => unreachable!(),
        })?;
        let prm = dev.upload_u32_slice(&[
            rows as u32,
            d as u32,
            x_off as u32,
            dl_off as u32,
            w_off as u32,
            has_delta,
            write_residual as u32,
            plus_one as u32,
            eps.to_bits(),
        ])?;
        let r_ref = r_out.as_ref().unwrap_or(&y_out);
        let g = rows.clamp(1, MAX_GROUPS) as u32;
        run(dev, &kern, &[self, dl_s, weight_f32, r_ref, &y_out, &prm], &[], [g, 1, 1])?;
        Ok((r_out, y_out))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn res_rmsnorm_inplace(
        &mut self,
        delta: Option<(&Self, usize)>,
        weight_f32: &Self,
        w_off: usize,
        rows: usize,
        d: usize,
        eps: f32,
        plus_one: bool,
    ) -> Result<Self> {
        self.invalidate_mirror();
        let dev = self.device.clone();
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        let n = rows * d;
        let y_out = dev.alloc_buffer(n, dtype)?;
        if rows == 0 || d == 0 {
            return Ok(y_out);
        }
        let (dl_s, dl_off, has_delta) = match delta {
            Some((s, o)) => (s, o, 1u32),
            None => (&*self, 0, 0u32),
        };
        if dtype != DType::F32 && (d % 2 != 0 || (has_delta != 0 && dl_off % 2 != 0)) {
            bail!("vulkan res_rmsnorm_inplace {tag}: requires even d/offset")
        }
        fits(n)?;
        fits(dl_off + n)?;
        fits(w_off + d)?;
        let key = format!("dec_res_rmsnorm_{tag}");
        let kern = kernel(&dev, &key, 6, 0, || match dtype {
            DType::F32 => res_rmsnorm_f32_src(),
            DType::F16 => res_rmsnorm_p16_src(false),
            DType::BF16 => res_rmsnorm_p16_src(true),
            _ => unreachable!(),
        })?;
        let prm = dev.upload_u32_slice(&[
            rows as u32,
            d as u32,
            0u32,
            dl_off as u32,
            w_off as u32,
            has_delta,
            1u32,
            plus_one as u32,
            eps.to_bits(),
        ])?;
        let g = rows.clamp(1, MAX_GROUPS) as u32;
        run(&dev, &kern, &[&*self, dl_s, weight_f32, &*self, &y_out, &prm], &[], [g, 1, 1])?;
        Ok(y_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn silu_mul(
        &self,
        g_off: usize,
        g_stride: usize,
        up: &Self,
        u_off: usize,
        u_stride: usize,
        rows: usize,
        inter_dim: usize,
    ) -> Result<Self> {
        let dev = &self.device;
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        let total = rows * inter_dim;
        let out = dev.alloc_buffer(total, dtype)?;
        if total == 0 {
            return Ok(out);
        }
        if dtype != DType::F32 && inter_dim % 2 != 0 {
            bail!("vulkan silu_mul {tag}: requires even inter_dim")
        }
        fits(g_off + rows * g_stride)?;
        fits(u_off + rows * u_stride)?;
        fits(total)?;
        let key = format!("dec_silu_mul_{tag}");
        let kern = kernel(dev, &key, 4, 0, || match dtype {
            DType::F32 => silu_mul_f32_src(),
            DType::F16 => silu_mul_p16_src(false),
            DType::BF16 => silu_mul_p16_src(true),
            _ => unreachable!(),
        })?;
        let prm = dev.upload_u32_slice(&[
            rows as u32,
            inter_dim as u32,
            g_off as u32,
            g_stride as u32,
            u_off as u32,
            u_stride as u32,
        ])?;
        let work = if dtype == DType::F32 { total } else { total / 2 };
        let g = work.div_ceil(256).clamp(1, MAX_GROUPS) as u32;
        run(dev, &kern, &[self, up, &out, &prm], &[], [g, 1, 1])?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gated_rms_norm(
        &self,
        x_off: usize,
        z: &Self,
        z_off: usize,
        weight_f32: &Self,
        w_off: usize,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<Self> {
        let dev = &self.device;
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        let n = rows * d;
        let out = dev.alloc_buffer(n, dtype)?;
        if n == 0 {
            return Ok(out);
        }
        if dtype != DType::F32 && d % 2 != 0 {
            bail!("vulkan gated_rms_norm {tag}: requires even d")
        }
        fits(x_off + n)?;
        fits(z_off + n)?;
        fits(w_off + d)?;
        let key = format!("dec_gated_rmsnorm_{tag}");
        let kern = kernel(dev, &key, 5, 0, || match dtype {
            DType::F32 => gated_rmsnorm_f32_src(),
            DType::F16 => gated_rmsnorm_p16_src(false),
            DType::BF16 => gated_rmsnorm_p16_src(true),
            _ => unreachable!(),
        })?;
        let prm = dev.upload_u32_slice(&[
            rows as u32,
            d as u32,
            x_off as u32,
            z_off as u32,
            w_off as u32,
            eps.to_bits(),
        ])?;
        let g = rows.clamp(1, MAX_GROUPS) as u32;
        run(dev, &kern, &[self, z, weight_f32, &out, &prm], &[], [g, 1, 1])?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv_write(
        &self,
        k_in: &Self,
        v_in: &Self,
        q_norm_w: Option<&Self>,
        k_norm_w: Option<&Self>,
        cos_tab: &Self,
        sin_tab: &Self,
        k_cache: &mut Self,
        v_cache: &mut Self,
        args: &RopeKvLaunchArgs,
    ) -> Result<Self> {
        let dev = &self.device;
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        if dev.limits().max_storage_bindings < 11 {
            bail!("vulkan rope_kv_write: device supports fewer than 11 storage buffer bindings")
        }
        let d = args.d as usize;
        if d > 512 {
            bail!("vulkan rope_kv_write: head_dim {d} > 512")
        }
        if dtype != DType::F32 {
            let even_ok = d % 2 == 0
                && args.kc_off % 2 == 0
                && args.vc_off % 2 == 0
                && args.qo_strides.iter().all(|&s| s % 2 == 0)
                && args.kc_strides.iter().all(|&s| s % 2 == 0);
            if !even_ok {
                bail!("vulkan rope_kv_write {tag}: requires even head_dim and output/cache strides")
            }
        }
        k_cache.invalidate_mirror();
        v_cache.invalidate_mirror();
        let q_out_elems = args.b as usize * args.t as usize * args.hq as usize * d;
        let q_out = dev.alloc_buffer(q_out_elems, dtype)?;
        if q_out_elems == 0 {
            return Ok(q_out);
        }
        let key = format!("dec_rope_kv_write_{tag}");
        let kern = kernel(dev, &key, 11, 0, || rope_kv_write_src(dtype))?;
        let vals = [
            args.b,
            args.t,
            args.hq,
            args.hkv,
            args.d,
            args.rotary_dim,
            args.past_len,
            args.eps.to_bits(),
            args.flags,
            q_norm_w.is_some() as u32,
            k_norm_w.is_some() as u32,
            fits(args.q_off)?,
            fits(args.k_off)?,
            fits(args.v_off)?,
            fits(args.qnw_off)?,
            fits(args.knw_off)?,
            fits(args.cos_off)?,
            fits(args.sin_off)?,
            fits(args.kc_off)?,
            fits(args.vc_off)?,
            fits(args.q_strides[0] as usize)?,
            fits(args.q_strides[1] as usize)?,
            fits(args.q_strides[2] as usize)?,
            fits(args.qo_strides[0] as usize)?,
            fits(args.qo_strides[1] as usize)?,
            fits(args.qo_strides[2] as usize)?,
            fits(args.k_strides[0] as usize)?,
            fits(args.k_strides[1] as usize)?,
            fits(args.k_strides[2] as usize)?,
            fits(args.v_strides[0] as usize)?,
            fits(args.v_strides[1] as usize)?,
            fits(args.v_strides[2] as usize)?,
            fits(args.kc_strides[0] as usize)?,
            fits(args.kc_strides[1] as usize)?,
            fits(args.kc_strides[2] as usize)?,
            fits(args.cos_strides[0] as usize)?,
            fits(args.cos_strides[1] as usize)?,
        ];
        let prm = dev.upload_u32_slice(&vals)?;
        let qnw_ref = q_norm_w.unwrap_or(cos_tab);
        let knw_ref = k_norm_w.unwrap_or(cos_tab);
        let gx = args.hq + args.hkv;
        let gy = ((args.b as usize) * (args.t as usize)).clamp(1, MAX_GROUPS) as u32;
        run(
            dev,
            &kern,
            &[
                self, k_in, v_in, qnw_ref, knw_ref, cos_tab, sin_tab, &q_out, k_cache, v_cache,
                &prm,
            ],
            &[],
            [gx, gy, 1],
        )?;
        Ok(q_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        use naga::front::glsl::{Frontend, Options};
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = Frontend::default()
            .parse(&Options::from(naga::ShaderStage::Compute), src)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}\n{src}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("validation failed: {e:?}"));
    }

    #[test]
    fn decoder_ops_shaders_validate() {
        validate(&res_rmsnorm_f32_src());
        validate(&res_rmsnorm_p16_src(false));
        validate(&res_rmsnorm_p16_src(true));
        validate(&silu_mul_f32_src());
        validate(&silu_mul_p16_src(false));
        validate(&silu_mul_p16_src(true));
        validate(&gated_rmsnorm_f32_src());
        validate(&gated_rmsnorm_p16_src(false));
        validate(&gated_rmsnorm_p16_src(true));
        for dt in [DType::F32, DType::F16, DType::BF16] {
            validate(&rope_kv_write_src(dt));
        }
    }
}
