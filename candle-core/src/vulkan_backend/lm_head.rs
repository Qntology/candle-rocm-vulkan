//! Lossless `lm_head` fused tile argmax + Mean-Centered Cauchy-Schwarz (CS) Guard on Vulkan.
//!
//! - Mode A (`fused_linear_argmax_cs`): 2-Stage dense tile matmul + in-register argmax + CS guard
//!   with 0 bytes of `[M, V]` logits materialization.
//! - Mode B (`logits_argmax_cs`): 1-Stage argmax + `core_ids` remap + CS guard over `[M, V_core]`.
//!
//! All accumulations and tie-breaks (`v > bv || (v == bv && gi < bi)`) match the CUDA/ROCm/CPU
//! reference bit-for-bit in semantics. Both stages record into the shared async command batch.

use super::shaders::{kernel, run, GLSL_PACK16_HELPERS};
use super::{VulkanDevice, VulkanStorage};
use crate::{bail, DType, Result};

const MAX_GROUPS: usize = 65535;
const STAGE1_BN: usize = 32;
const STAGE1_BM: usize = 4;

fn fits(n: usize) -> Result<u32> {
    if n > u32::MAX as usize {
        bail!("vulkan lm_head: index {n} exceeds 32-bit kernel range")
    }
    Ok(n as u32)
}

fn dtype_tag(dt: DType) -> Result<&'static str> {
    match dt {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => bail!("vulkan lm_head: unsupported dtype {other:?}"),
    }
}

fn load_spec(dt: DType, buf: &str) -> (&'static str, String) {
    match dt {
        DType::F32 => ("float", format!("{buf}[idx]")),
        DType::F16 => ("uint", format!("ld_f16({buf}[idx >> 1u], idx)")),
        DType::BF16 => ("uint", format!("ld_bf16({buf}[idx >> 1u], idx)")),
        _ => unreachable!(),
    }
}

const LM_COMMON: &str = r#"
float ninf() { return uintBitsToFloat(0xff800000u); }
bool better(float vc, uint ic, float vb, uint ib) {
    return (vc > vb) || (vc == vb && ic < ib);
}
"#;

// Stage 1: [M, K] x [V, K]^T + bias -> partial argmax per 32-col tile [M, n_tiles]
// Bindings: 0 x, 1 w, 2 bias, 3 part_val, 4 part_idx, 5 prm
// prm: 0 m, 1 v_valid, 2 k, 3 n_tiles, 4 x_off, 5 w_off, 6 b_off, 7 has_bias
fn stage1_src(dtype: DType) -> String {
    let (decl, ld_x) = load_spec(dtype, "x");
    let (_, ld_w) = load_spec(dtype, "w");
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Xb {{ {decl} x[]; }};
layout(set = 0, binding = 1) readonly buffer Wb {{ {decl} w[]; }};
layout(set = 0, binding = 2) readonly buffer Bb {{ float bias[]; }};
layout(set = 0, binding = 3) buffer PVb {{ float part_val[]; }};
layout(set = 0, binding = 4) buffer PIb {{ uint part_idx[]; }};
layout(set = 0, binding = 5) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
{LM_COMMON}
float load_x(uint idx) {{ return {ld_x}; }}
float load_w(uint idx) {{ return {ld_w}; }}
shared float red[256];
shared float tile_val[128];
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint grp = tid >> 5u;
    uint lane = tid & 31u;
    uint M = prm[0]; uint V = prm[1]; uint K = prm[2]; uint n_tiles = prm[3];
    uint x_off = prm[4]; uint w_off = prm[5]; uint b_off = prm[6];
    bool has_bias = prm[7] != 0u;
    for (uint tile = gl_WorkGroupID.x; tile < n_tiles; tile += gl_NumWorkGroups.x) {{
        uint col0 = tile * 32u;
        for (uint r0 = gl_WorkGroupID.y * 4u; r0 < M; r0 += gl_NumWorkGroups.y * 4u) {{
            uint rm = min(4u, M - r0);
            for (uint pass = 0u; pass < 4u; pass++) {{
                uint lc = pass * 8u + grp;
                uint col = col0 + lc;
                float a0 = 0.0; float a1 = 0.0; float a2 = 0.0; float a3 = 0.0;
                if (col < V) {{
                    uint wb = w_off + col * K;
                    uint xb0 = x_off + r0 * K;
                    for (uint kk = lane; kk < K; kk += 32u) {{
                        float wv = load_w(wb + kk);
                        a0 += load_x(xb0 + kk) * wv;
                        if (rm > 1u) {{ a1 += load_x(xb0 + K + kk) * wv; }}
                        if (rm > 2u) {{ a2 += load_x(xb0 + 2u * K + kk) * wv; }}
                        if (rm > 3u) {{ a3 += load_x(xb0 + 3u * K + kk) * wv; }}
                    }}
                }}
                float bv = (has_bias && col < V) ? bias[b_off + col] : 0.0;
                // Reduce each active row across the 32 lanes of this subgroup
                red[tid] = a0;
                barrier();
                for (uint s = 16u; s > 0u; s >>= 1u) {{
                    if (lane < s) {{ red[tid] += red[tid + s]; }}
                    barrier();
                }}
                if (lane == 0u) {{ tile_val[lc] = (col < V) ? (red[tid] + bv) : ninf(); }}
                barrier();
                if (rm > 1u) {{
                    red[tid] = a1;
                    barrier();
                    for (uint s = 16u; s > 0u; s >>= 1u) {{
                        if (lane < s) {{ red[tid] += red[tid + s]; }}
                        barrier();
                    }}
                    if (lane == 0u) {{ tile_val[32u + lc] = (col < V) ? (red[tid] + bv) : ninf(); }}
                    barrier();
                }}
                if (rm > 2u) {{
                    red[tid] = a2;
                    barrier();
                    for (uint s = 16u; s > 0u; s >>= 1u) {{
                        if (lane < s) {{ red[tid] += red[tid + s]; }}
                        barrier();
                    }}
                    if (lane == 0u) {{ tile_val[64u + lc] = (col < V) ? (red[tid] + bv) : ninf(); }}
                    barrier();
                }}
                if (rm > 3u) {{
                    red[tid] = a3;
                    barrier();
                    for (uint s = 16u; s > 0u; s >>= 1u) {{
                        if (lane < s) {{ red[tid] += red[tid + s]; }}
                        barrier();
                    }}
                    if (lane == 0u) {{ tile_val[96u + lc] = (col < V) ? (red[tid] + bv) : ninf(); }}
                    barrier();
                }}
            }}
            // One thread per row in the BM=4 block reduces the 32 vocab columns deterministically
            if (tid < rm) {{
                float best_v = ninf();
                uint best_i = 0xffffffffu;
                uint base = tid * 32u;
                for (uint c = 0u; c < 32u; c++) {{
                    uint gi = col0 + c;
                    if (gi < V) {{
                        float v = tile_val[base + c];
                        if (better(v, gi, best_v, best_i)) {{
                            best_v = v;
                            best_i = gi;
                        }}
                    }}
                }}
                uint out_i = (r0 + tid) * n_tiles + tile;
                part_val[out_i] = best_v;
                part_idx[out_i] = best_i;
            }}
            barrier();
        }}
    }}
}}
"#
    )
}

// Stage 2: reduce [M, n_tiles] -> [M] (out_ids, out_vals, atomicOr out_rescue u8)
// Bindings: 0 part_val, 1 part_idx, 2 core_ids, 3 hidden, 4 mu_rest, 5 out_ids, 6 out_vals, 7 out_rescue, 8 prm
// prm: 0 m, 1 n_tiles, 2 v_valid, 3 k, 4 has_cids, 5 cids_off, 6 check_cs, 7 h_off, 8 has_mu, 9 mu_off, 10 w_bound_bits, 11 b_max_bits
fn stage2_src(h_dtype: DType) -> String {
    let (h_decl, ld_h) = load_spec(h_dtype, "hidden");
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer PVb {{ float part_val[]; }};
layout(set = 0, binding = 1) readonly buffer PIb {{ uint part_idx[]; }};
layout(set = 0, binding = 2) readonly buffer CIb {{ uint core_ids[]; }};
layout(set = 0, binding = 3) readonly buffer Hb {{ {h_decl} hidden[]; }};
layout(set = 0, binding = 4) readonly buffer Mb {{ float mu_rest[]; }};
layout(set = 0, binding = 5) buffer OIb {{ uint out_ids[]; }};
layout(set = 0, binding = 6) buffer OVb {{ float out_vals[]; }};
layout(set = 0, binding = 7) buffer ORb {{ uint out_rescue[]; }};
layout(set = 0, binding = 8) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
{LM_COMMON}
float load_h(uint idx) {{ return {ld_h}; }}
shared float s_val[256];
shared uint s_idx[256];
shared float s_xx[256];
shared float s_xm[256];
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint M = prm[0]; uint n_tiles = prm[1]; uint V = prm[2]; uint K = prm[3];
    bool has_cids = prm[4] != 0u; uint cids_off = prm[5];
    bool check_cs = prm[6] != 0u; uint h_off = prm[7];
    bool has_mu = prm[8] != 0u; uint mu_off = prm[9];
    float w_bound = uintBitsToFloat(prm[10]);
    float b_max = uintBitsToFloat(prm[11]);
    for (uint row = gl_WorkGroupID.x; row < M; row += gl_NumWorkGroups.x) {{
        float bv = ninf();
        uint bi = 0xffffffffu;
        uint base = row * n_tiles;
        for (uint t = tid; t < n_tiles; t += 256u) {{
            float v = part_val[base + t];
            uint idx = part_idx[base + t];
            if (better(v, idx, bv, bi)) {{ bv = v; bi = idx; }}
        }}
        s_val[tid] = bv;
        s_idx[tid] = bi;
        float xx = 0.0; float xm = 0.0;
        if (check_cs) {{
            uint hb = h_off + row * K;
            for (uint kk = tid; kk < K; kk += 256u) {{
                float hv = load_h(hb + kk);
                xx += hv * hv;
                if (has_mu) {{ xm += hv * mu_rest[mu_off + kk]; }}
            }}
        }}
        s_xx[tid] = xx;
        s_xm[tid] = xm;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {{
            if (tid < s) {{
                if (better(s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {{
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }}
                if (check_cs) {{
                    s_xx[tid] += s_xx[tid + s];
                    s_xm[tid] += s_xm[tid + s];
                }}
            }}
            barrier();
        }}
        if (tid == 0u) {{
            uint win_i = s_idx[0];
            float win_v = s_val[0];
            uint mapped = win_i;
            if (has_cids && win_i < V) {{ mapped = core_ids[cids_off + win_i]; }}
            out_ids[row] = mapped;
            out_vals[row] = win_v;
            if (check_cs) {{
                float upper = sqrt(s_xx[0]) * w_bound + s_xm[0] + b_max;
                out_rescue[row] = (win_v < upper) ? 1u : 0u;
            }}
        }}
        barrier();
    }}
}}
"#
    )
}

// Mode B (1-Stage): logits [M, V_core] -> out_ids, out_vals, out_rescue
// Bindings: 0 logits, 1 core_ids, 2 hidden, 3 mu_rest, 4 out_ids, 5 out_vals, 6 out_rescue, 7 prm
// prm: 0 m, 1 v_valid, 2 l_off, 3 l_stride, 4 has_cids, 5 cids_off, 6 check_cs, 7 has_h, 8 h_off, 9 k,
//      10 has_mu, 11 mu_off, 12 w_bound_bits, 13 b_max_bits
fn logits_argmax_cs_src(l_dtype: DType, h_dtype: DType) -> String {
    let (l_decl, ld_l) = load_spec(l_dtype, "logits");
    let (h_decl, ld_h) = load_spec(h_dtype, "hidden");
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Lb {{ {l_decl} logits[]; }};
layout(set = 0, binding = 1) readonly buffer CIb {{ uint core_ids[]; }};
layout(set = 0, binding = 2) readonly buffer Hb {{ {h_decl} hidden[]; }};
layout(set = 0, binding = 3) readonly buffer Mb {{ float mu_rest[]; }};
layout(set = 0, binding = 4) buffer OIb {{ uint out_ids[]; }};
layout(set = 0, binding = 5) buffer OVb {{ float out_vals[]; }};
layout(set = 0, binding = 6) buffer ORb {{ uint out_rescue[]; }};
layout(set = 0, binding = 7) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
{LM_COMMON}
float load_l(uint idx) {{ return {ld_l}; }}
float load_h(uint idx) {{ return {ld_h}; }}
shared float s_val[256];
shared uint s_idx[256];
shared float s_xx[256];
shared float s_xm[256];
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint M = prm[0]; uint V = prm[1]; uint l_off = prm[2]; uint l_st = prm[3];
    bool has_cids = prm[4] != 0u; uint cids_off = prm[5];
    bool check_cs = prm[6] != 0u; bool has_h = prm[7] != 0u;
    uint h_off = prm[8]; uint K = prm[9];
    bool has_mu = prm[10] != 0u; uint mu_off = prm[11];
    float w_bound = uintBitsToFloat(prm[12]);
    float b_max = uintBitsToFloat(prm[13]);
    for (uint row = gl_WorkGroupID.x; row < M; row += gl_NumWorkGroups.x) {{
        float bv = ninf();
        uint bi = 0xffffffffu;
        uint lb = l_off + row * l_st;
        for (uint j = tid; j < V; j += 256u) {{
            float v = load_l(lb + j);
            if (better(v, j, bv, bi)) {{ bv = v; bi = j; }}
        }}
        s_val[tid] = bv;
        s_idx[tid] = bi;
        float xx = 0.0; float xm = 0.0;
        if (check_cs && has_h) {{
            uint hb = h_off + row * K;
            for (uint kk = tid; kk < K; kk += 256u) {{
                float hv = load_h(hb + kk);
                xx += hv * hv;
                if (has_mu) {{ xm += hv * mu_rest[mu_off + kk]; }}
            }}
        }}
        s_xx[tid] = xx;
        s_xm[tid] = xm;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1u) {{
            if (tid < s) {{
                if (better(s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {{
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }}
                if (check_cs && has_h) {{
                    s_xx[tid] += s_xx[tid + s];
                    s_xm[tid] += s_xm[tid + s];
                }}
            }}
            barrier();
        }}
        if (tid == 0u) {{
            uint win_i = s_idx[0];
            float win_v = s_val[0];
            uint mapped = win_i;
            if (has_cids && win_i < V) {{ mapped = core_ids[cids_off + win_i]; }}
            out_ids[row] = mapped;
            out_vals[row] = win_v;
            if (check_cs && has_h) {{
                float upper = sqrt(s_xx[0]) * w_bound + s_xm[0] + b_max;
                out_rescue[row] = (win_v < upper) ? 1u : 0u;
            }}
        }}
        barrier();
    }}
}}
"#
    )
}

const PACK_U32_TO_U8_SRC: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Sb { uint src[]; };
layout(set = 0, binding = 1) buffer Db { uint dst[]; };
layout(set = 0, binding = 2) readonly buffer Pb { uint prm[]; };
void main() {
    uint m = prm[0];
    uint words = (m + 3u) >> 2u;
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint w = gl_GlobalInvocationID.x; w < words; w += stride) {
        uint i0 = w * 4u;
        uint v = (src[i0] & 0xffu);
        if (i0 + 1u < m) { v |= (src[i0 + 1u] & 0xffu) << 8u; }
        if (i0 + 2u < m) { v |= (src[i0 + 2u] & 0xffu) << 16u; }
        if (i0 + 3u < m) { v |= (src[i0 + 3u] & 0xffu) << 24u; }
        dst[w] = v;
    }
}
"#;

fn pack_u32_flags_to_u8(dev: &VulkanDevice, flags_u32: &VulkanStorage, m: usize) -> Result<VulkanStorage> {
    let dst = dev.alloc_buffer(m, DType::U8)?;
    if m == 0 {
        return Ok(dst);
    }
    let kern = kernel(dev, "lm_pack_u32_to_u8", 3, 0, || PACK_U32_TO_U8_SRC.to_string())?;
    let prm = dev.upload_u32_slice(&[m as u32])?;
    let words = m.div_ceil(4);
    let g = words.div_ceil(256).clamp(1, MAX_GROUPS) as u32;
    run(dev, &kern, &[flags_u32, &dst, &prm], &[], [g, 1, 1])?;
    Ok(dst)
}

fn alloc_zeroed_u8(dev: &VulkanDevice, m: usize) -> Result<VulkanStorage> {
    let storage = dev.alloc_buffer(m, DType::U8)?;
    let zero_bytes = m.div_ceil(4) * 4;
    if zero_bytes > 0 {
        unsafe { std::ptr::write_bytes(storage.mapped, 0, zero_bytes) };
    }
    Ok(storage)
}

impl VulkanStorage {
    /// Mode A: 2-Stage dense `lm_head` fused tile matmul + argmax + optional CS guard.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_linear_argmax_cs(
        &self,
        x_off: usize,
        weight: &Self,
        w_off: usize,
        bias: Option<(&Self, usize)>,
        core_ids: Option<(&Self, usize)>,
        mu_rest: Option<(&Self, usize)>,
        w_rest_bound: f32,
        b_rest_max: f32,
        check_cs: bool,
        m: usize,
        v_valid: usize,
        k: usize,
    ) -> Result<(Self, Self, Self)> {
        let dev = &self.device;
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        let out_ids = dev.alloc_buffer(m, DType::U32)?;
        let out_vals = dev.alloc_buffer(m, DType::F32)?;
        if m == 0 || v_valid == 0 {
            let out_rescue = alloc_zeroed_u8(dev, m)?;
            return Ok((out_ids, out_vals, out_rescue));
        }
        let rescue_u32 = dev.alloc_buffer(m, DType::U32)?;
        fits(x_off + m * k)?;
        fits(w_off + v_valid * k)?;
        let n_tiles = v_valid.div_ceil(STAGE1_BN).max(1);
        fits(m * n_tiles)?;
        let part_val = dev.alloc_buffer(m * n_tiles, DType::F32)?;
        let part_idx = dev.alloc_buffer(m * n_tiles, DType::U32)?;

        let (b_s, b_off, has_b) = match bias {
            Some((s, o)) => (s, o, 1u32),
            None => (&out_vals, 0, 0u32),
        };
        let k1 = kernel(dev, &format!("lm_stage1_{tag}"), 6, 0, || stage1_src(dtype))?;
        let prm1 = dev.upload_u32_slice(&[
            m as u32,
            v_valid as u32,
            k as u32,
            n_tiles as u32,
            x_off as u32,
            w_off as u32,
            b_off as u32,
            has_b,
        ])?;
        let gx = n_tiles.clamp(1, MAX_GROUPS) as u32;
        let gy = m.div_ceil(STAGE1_BM).clamp(1, MAX_GROUPS) as u32;
        run(
            dev,
            &k1,
            &[self, weight, b_s, &part_val, &part_idx, &prm1],
            &[],
            [gx, gy, 1],
        )?;

        let (c_s, c_off, has_c) = match core_ids {
            Some((s, o)) => (s, o, 1u32),
            None => (&out_ids, 0, 0u32),
        };
        let (mu_s, mu_off, has_mu) = match mu_rest {
            Some((s, o)) => (s, o, 1u32),
            None => (&out_vals, 0, 0u32),
        };
        let k2 = kernel(dev, &format!("lm_stage2_{tag}"), 9, 0, || stage2_src(dtype))?;
        let prm2 = dev.upload_u32_slice(&[
            m as u32,
            n_tiles as u32,
            v_valid as u32,
            k as u32,
            has_c,
            c_off as u32,
            check_cs as u32,
            x_off as u32,
            has_mu,
            mu_off as u32,
            w_rest_bound.to_bits(),
            b_rest_max.to_bits(),
        ])?;
        let g2 = m.clamp(1, MAX_GROUPS) as u32;
        run(
            dev,
            &k2,
            &[
                &part_val,
                &part_idx,
                c_s,
                self,
                mu_s,
                &out_ids,
                &out_vals,
                &rescue_u32,
                &prm2,
            ],
            &[],
            [g2, 1, 1],
        )?;
        let out_rescue = if check_cs {
            pack_u32_flags_to_u8(dev, &rescue_u32, m)?
        } else {
            alloc_zeroed_u8(dev, m)?
        };
        Ok((out_ids, out_vals, out_rescue))
    }

    /// Mode B: 1-Stage argmax + `core_ids` remap + optional CS guard over `sub_logits [M, V_core]`.
    #[allow(clippy::too_many_arguments)]
    pub fn logits_argmax_cs(
        &self,
        l_off: usize,
        l_stride: usize,
        core_ids: Option<(&Self, usize)>,
        hidden: Option<(&Self, usize)>,
        mu_rest: Option<(&Self, usize)>,
        w_rest_bound: f32,
        b_rest_max: f32,
        check_cs: bool,
        m: usize,
        v_valid: usize,
        k: usize,
    ) -> Result<(Self, Self, Self)> {
        let dev = &self.device;
        let l_dtype = self.dtype;
        let l_tag = dtype_tag(l_dtype)?;
        let (h_s, h_off, has_h, h_dtype) = match hidden {
            Some((s, o)) => (s, o, 1u32, s.dtype),
            None => (self, 0, 0u32, l_dtype),
        };
        let h_tag = dtype_tag(h_dtype)?;
        let out_ids = dev.alloc_buffer(m, DType::U32)?;
        let out_vals = dev.alloc_buffer(m, DType::F32)?;
        if m == 0 || v_valid == 0 {
            let out_rescue = alloc_zeroed_u8(dev, m)?;
            return Ok((out_ids, out_vals, out_rescue));
        }
        let rescue_u32 = dev.alloc_buffer(m, DType::U32)?;
        fits(l_off + m * l_stride)?;
        if has_h != 0 {
            fits(h_off + m * k)?;
        }
        let (c_s, c_off, has_c) = match core_ids {
            Some((s, o)) => (s, o, 1u32),
            None => (&out_ids, 0, 0u32),
        };
        let (mu_s, mu_off, has_mu) = match mu_rest {
            Some((s, o)) => (s, o, 1u32),
            None => (&out_vals, 0, 0u32),
        };
        let key = format!("lm_logits_argmax_cs_{l_tag}_{h_tag}");
        let kern = kernel(dev, &key, 8, 0, || logits_argmax_cs_src(l_dtype, h_dtype))?;
        let prm = dev.upload_u32_slice(&[
            m as u32,
            v_valid as u32,
            l_off as u32,
            l_stride as u32,
            has_c,
            c_off as u32,
            check_cs as u32,
            has_h,
            h_off as u32,
            k as u32,
            has_mu,
            mu_off as u32,
            w_rest_bound.to_bits(),
            b_rest_max.to_bits(),
        ])?;
        let g = m.clamp(1, MAX_GROUPS) as u32;
        run(
            dev,
            &kern,
            &[self, c_s, h_s, mu_s, &out_ids, &out_vals, &rescue_u32, &prm],
            &[],
            [g, 1, 1],
        )?;
        let out_rescue = if check_cs {
            pack_u32_flags_to_u8(dev, &rescue_u32, m)?
        } else {
            alloc_zeroed_u8(dev, m)?
        };
        Ok((out_ids, out_vals, out_rescue))
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
    fn lm_head_shaders_validate() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            validate(&stage1_src(dt));
            validate(&stage2_src(dt));
            validate(&logits_argmax_cs_src(dt, dt));
        }
        validate(&logits_argmax_cs_src(DType::F32, DType::F16));
        validate(&logits_argmax_cs_src(DType::F32, DType::BF16));
        validate(PACK_U32_TO_U8_SRC);
    }
}
