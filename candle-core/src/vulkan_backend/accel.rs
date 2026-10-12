//! Fused attention and TurboQuant kernels on the Vulkan backend.
//!
//! GLSL ports of `kernels/flash_attn_impl.h` and `kernels/turbo_quant_impl.h` (same algorithms,
//! same packed row layout), compiled with naga on first use. Queries and values are f32 (the
//! callers convert f16 / bf16); K/V may also be FP8 (e4m3) bytes, decoded in the shader.
//! Kernel parameters go through a small storage buffer because many devices only offer 128
//! bytes of push constants; indices are 32-bit.

use super::shaders::{kernel, run};
use super::{VulkanDevice, VulkanStorage};
use crate::accel::attention::{self as att, AttnGeometry, FaExtra, FaPlan, FlashAttnParams};
use crate::accel::profile::AccelProfile;
use crate::accel::turbo::{TqAttnArgs, TqRowArgs};
use crate::{bail, DType, Result};

const MAX_GROUPS: usize = 65535;

impl VulkanDevice {
    /// Fused attention / TurboQuant profile of this device (see `candle_core::accel::profile`).
    pub fn accel_profile(&self) -> AccelProfile {
        let l = self.limits();
        AccelProfile::vulkan(
            self.name(),
            l.max_shared_memory as usize,
            l.max_invocations as usize,
        )
    }
}

const COMMON: &str = r#"
float ninf() { return uintBitsToFloat(0xff800000u); }
float e4m3_to_f(uint c) {
    uint e = (c >> 3u) & 15u;
    uint m = c & 7u;
    if (e == 15u && m == 7u) { return uintBitsToFloat(0x7fc00000u); }
    float v;
    if (e == 0u) { v = float(m) * 0.001953125; } else { v = uintBitsToFloat(((e + 120u) << 23u) | (m << 20u)); }
    if ((c & 0x80u) != 0u) { v = -v; }
    return v;
}
uint f_to_e4m3(float f) {
    uint x = floatBitsToUint(f);
    uint s = (x >> 24u) & 0x80u;
    uint ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) { return s | 0x7fu; }
    float a = uintBitsToFloat(ax);
    if (a >= 448.0) { return s | 0x7eu; }
    if (a < 0.015625) { return s | uint(roundEven(a * 512.0)); }
    uint r = ax + 0x7ffffu + ((ax >> 20u) & 1u);
    uint code = (((r >> 23u) - 120u) << 3u) | ((r >> 20u) & 7u);
    if (code > 0x7eu) { code = 0x7eu; }
    return s | code;
}
float fp4_to_f(uint c) {
    uint e = (c >> 1u) & 3u;
    uint m = c & 1u;
    float v;
    if (e == 0u) { v = 0.5 * float(m); } else { v = uintBitsToFloat(((e + 126u) << 23u) | (m << 22u)); }
    if ((c & 8u) != 0u) { v = -v; }
    return v;
}
uint f_to_fp4(float f) {
    uint s = (floatBitsToUint(f) >> 28u) & 8u;
    float a = abs(f);
    uint c = 7u;
    if (!(a > 0.25)) { c = 0u; } else if (a < 0.75) { c = 1u; } else if (a <= 1.25) { c = 2u; }
    else if (a < 1.75) { c = 3u; } else if (a <= 2.5) { c = 4u; } else if (a < 3.5) { c = 5u; }
    else if (a <= 5.0) { c = 6u; }
    return s | c;
}
float e8m0_to_f(uint c) {
    if (c == 0u) { return uintBitsToFloat(0x00400000u); }
    return uintBitsToFloat(c << 23u);
}
uint mx_scale_code(float amax, int emax) {
    int e = -127;
    uint bits = floatBitsToUint(amax);
    int be = int((bits >> 23u) & 0xffu);
    if (amax > 0.0 && be != 255 && be != 0) { e = be - 127 - emax; }
    e = clamp(e, -127, 127);
    return uint(e + 127);
}
float logit(float s, float softcap, float slope, int dist) {
    if (softcap > 0.0) { s = softcap * tanh(s / softcap); }
    if (slope != 0.0) { s = s - slope * float(abs(dist)); }
    return s;
}
"#;

const REDUCE: &str = r#"
shared float red[128];
float block_sum(float v) {
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
float block_max(float v) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (t < s) { red[t] = max(red[t], red[t + s]); }
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}
"#;

/// Bindings 0-9 of the attention shaders; `KTYPE` is `float` or `uint` (fp8 bytes).
const FA_BINDINGS: &str = r#"
layout(set = 0, binding = 0) readonly buffer Qb { float q[]; };
layout(set = 0, binding = 1) readonly buffer Kb { KTYPE k[]; };
layout(set = 0, binding = 2) readonly buffer Vb { KTYPE v[]; };
layout(set = 0, binding = 3) buffer Ob { float o[]; };
layout(set = 0, binding = 4) buffer POb { float part_o[]; };
layout(set = 0, binding = 5) buffer PLb { float part_lse[]; };
layout(set = 0, binding = 6) readonly buffer ALb { float alibi[]; };
layout(set = 0, binding = 7) readonly buffer CQb { uint cu_q[]; };
layout(set = 0, binding = 8) readonly buffer CKb { uint cu_k[]; };
layout(set = 0, binding = 9) readonly buffer Pb { uint prm[]; };
"#;

const LOAD_F32: &str = r#"
float ldk(uint i) { return k[i]; }
float ldv(uint i) { return v[i]; }
"#;

const LOAD_FP8: &str = r#"
float ldk(uint i) { return e4m3_to_f((k[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu); }
float ldv(uint i) { return e4m3_to_f((v[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu); }
"#;

// prm: 0 Lq, 1 Lk, 2 Hq, 3 Hkv, 4 D, 5 q_off, 6 k_off, 7 v_off, 8-10 q strides (b, s, h),
// 11-13 k strides, 14-16 v strides, 17 scale, 18 softcap, 19 wl, 20 wr, 21 kd, 22 vd,
// 23 n_splits, 24 alibi_bs, 25 varlen, 26 rows_total, 27 has_alibi, 28 alibi_off,
// 29 cu_q_off, 30 cu_k_off
const FA_PRELUDE: &str = r#"
uint P_LQ() { return prm[0]; }
uint P_LK() { return prm[1]; }
uint P_HQ() { return prm[2]; }
uint P_HKV() { return prm[3]; }
uint P_D() { return prm[4]; }
float P_F(uint i) { return uintBitsToFloat(prm[i]); }
int P_I(uint i) { return int(prm[i]); }
"#;

const FA_ROW: &str = r#"
shared float qs[256];
shared float pr[128];
void main() {
    uint t = gl_LocalInvocationID.x;
    uint split = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint Lq = P_LQ(); uint Lk = P_LK(); uint Hq = P_HQ(); uint Hkv = P_HKV(); uint D = P_D();
    uint n_splits = prm[23];
    bool varlen = prm[25] != 0u;
    uint lq = Lq; uint lk = Lk; uint q0 = b * Lq; uint k0 = 0u;
    if (varlen) {
        q0 = cu_q[prm[29] + b]; lq = cu_q[prm[29] + b + 1u] - q0;
        k0 = cu_k[prm[30] + b]; lk = cu_k[prm[30] + b + 1u] - k0;
    }
    int wl = P_I(19); int wr = P_I(20);
    float scale = P_F(17); float softcap = P_F(18); float kd = P_F(21); float vd = P_F(22);
    uint group = Hq / Hkv;
    for (uint rowi = gl_WorkGroupID.x; rowi < Lq * Hq; rowi += gl_NumWorkGroups.x) {
        uint i = rowi / Hq;
        uint h = rowi % Hq;
        if (i >= lq) { continue; }
        uint hk = h / group;
        uint qb = varlen ? prm[5] + (q0 + i) * prm[9] + h * prm[10] : prm[5] + b * prm[8] + i * prm[9] + h * prm[10];
        uint kb0 = varlen ? prm[6] + k0 * prm[12] + hk * prm[13] : prm[6] + b * prm[11] + hk * prm[13];
        uint vb0 = varlen ? prm[7] + k0 * prm[15] + hk * prm[16] : prm[7] + b * prm[14] + hk * prm[16];
        uint orow = (q0 + i) * Hq + h;
        barrier();
        for (uint d = t; d < D; d += 128u) { qs[d] = scale * q[qb + d]; }
        barrier();
        int pos = int(i) + int(lk) - int(lq);
        int lo = 0; int hi = int(lk);
        if (wl >= 0) { lo = max(pos - wl, 0); }
        if (wr >= 0) { hi = min(pos + wr + 1, int(lk)); }
        if (lo > int(lk)) { lo = int(lk); }
        if (hi < lo) { hi = lo; }
        int chunk = (hi - lo + int(n_splits) - 1) / int(n_splits);
        int s_lo = lo + int(split) * chunk;
        int s_hi = min(s_lo + chunk, hi);
        if (s_lo > hi) { s_lo = hi; }
        if (s_hi < s_lo) { s_hi = s_lo; }
        float slope = 0.0;
        if (prm[27] != 0u) { slope = alibi[prm[28] + b * prm[24] + h]; }
        float m = ninf(); float l = 0.0; float acc0 = 0.0; float acc1 = 0.0;
        for (int kb = s_lo; kb < s_hi; kb += 128) {
            int j = kb + int(t);
            float s = ninf();
            if (j < s_hi) {
                uint kr = kb0 + uint(j) * prm[12];
                float dot = 0.0;
                for (uint d = 0u; d < D; d++) { dot += qs[d] * ldk(kr + d); }
                s = logit(dot * kd, softcap, slope, pos - j);
            }
            float m_new = max(m, block_max(s));
            float p = 0.0;
            if (s != ninf()) { p = exp(s - m_new); }
            pr[t] = p;
            float lt = block_sum(p);
            float alpha = 0.0;
            if (m != ninf()) { alpha = exp(m - m_new); }
            l = l * alpha + lt;
            acc0 *= alpha; acc1 *= alpha;
            int cnt = min(128, s_hi - kb);
            for (int u = 0; u < cnt; u++) {
                float pu = pr[u];
                if (pu != 0.0) {
                    uint vr = vb0 + uint(kb + u) * prm[15];
                    if (t < D) { acc0 += pu * ldv(vr + t); }
                    if (t + 128u < D) { acc1 += pu * ldv(vr + t + 128u); }
                }
            }
            m = m_new;
            barrier();
        }
        float inv = 0.0;
        if (l > 0.0) { inv = vd / l; }
        if (n_splits == 1u) {
            if (t < D) { o[orow * D + t] = acc0 * inv; }
            if (t + 128u < D) { o[orow * D + t + 128u] = acc1 * inv; }
        } else {
            uint prow = split * prm[26] + orow;
            if (t < D) { part_o[prow * D + t] = acc0 * inv; }
            if (t + 128u < D) { part_o[prow * D + t + 128u] = acc1 * inv; }
            if (t == 0u) { if (l > 0.0) { part_lse[prow] = m + log(l); } else { part_lse[prow] = ninf(); } }
        }
    }
}
"#;

const FA_TILE: &str = r#"
shared float Qs[BR * (DMAX + 1)];
shared float KVs[BC * (DMAX + 1)];
shared float S[BR * BC];
void main() {
    const uint TPR = 128u / BR;
    uint t = gl_LocalInvocationID.x;
    uint r = t / TPR;
    uint lane = t % TPR;
    uint h = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint Lq = P_LQ(); uint Lk = P_LK(); uint Hq = P_HQ(); uint Hkv = P_HKV(); uint D = P_D();
    uint DP = D + 1u;
    bool varlen = prm[25] != 0u;
    uint lq = Lq; uint lk = Lk; uint q0 = b * Lq; uint k0 = 0u;
    if (varlen) {
        q0 = cu_q[prm[29] + b]; lq = cu_q[prm[29] + b + 1u] - q0;
        k0 = cu_k[prm[30] + b]; lk = cu_k[prm[30] + b + 1u] - k0;
    }
    int wl = P_I(19); int wr = P_I(20);
    float scale = P_F(17); float softcap = P_F(18); float kd = P_F(21); float vd = P_F(22);
    uint hk = h / (Hq / Hkv);
    uint qb = varlen ? prm[5] + q0 * prm[9] + h * prm[10] : prm[5] + b * prm[8] + h * prm[10];
    uint kb0 = varlen ? prm[6] + k0 * prm[12] + hk * prm[13] : prm[6] + b * prm[11] + hk * prm[13];
    uint vb0 = varlen ? prm[7] + k0 * prm[15] + hk * prm[16] : prm[7] + b * prm[14] + hk * prm[16];
    int off = int(lk) - int(lq);
    float slope = 0.0;
    if (prm[27] != 0u) { slope = alibi[prm[28] + b * prm[24] + h]; }
    for (uint tile = gl_WorkGroupID.x; tile * BR < lq; tile += gl_NumWorkGroups.x) {
        int i0 = int(tile * BR);
        barrier();
        for (uint idx = t; idx < BR * D; idx += 128u) {
            uint rr = idx / D; uint d = idx % D;
            int ii = i0 + int(rr);
            float x = 0.0;
            if (ii < int(lq)) { x = scale * q[qb + uint(ii) * prm[9] + d]; }
            Qs[rr * DP + d] = x;
        }
        int last = min(i0 + int(BR), int(lq)) - 1;
        int lo = 0; int hi = int(lk);
        if (wl >= 0) { lo = max(i0 + off - wl, 0); }
        if (wr >= 0) { hi = min(last + off + wr + 1, int(lk)); }
        if (lo > int(lk)) { lo = int(lk); }
        if (hi < lo) { hi = lo; }
        int i = i0 + int(r);
        float m = ninf(); float l = 0.0;
        float acc[NACC];
        for (uint a = 0u; a < NACC; a++) { acc[a] = 0.0; }
        for (int kt = lo; kt < hi; kt += int(BC)) {
            barrier();
            for (uint idx = t; idx < BC * D; idx += 128u) {
                uint c = idx / D; uint d = idx % D;
                int j = kt + int(c);
                float x = 0.0;
                if (j < hi) { x = kd * ldk(kb0 + uint(j) * prm[12] + d); }
                KVs[c * DP + d] = x;
            }
            barrier();
            for (uint idx = t; idx < BR * BC; idx += 128u) {
                uint rr = idx / BC; uint c = idx % BC;
                int ii = i0 + int(rr); int j = kt + int(c);
                int p2 = ii + off;
                float s = ninf();
                if (ii < int(lq) && j < hi && (wl < 0 || j >= p2 - wl) && (wr < 0 || j <= p2 + wr)) {
                    float dot = 0.0;
                    for (uint d = 0u; d < D; d++) { dot += Qs[rr * DP + d] * KVs[c * DP + d]; }
                    s = logit(dot, softcap, slope, p2 - j);
                }
                S[idx] = s;
            }
            barrier();
            for (uint idx = t; idx < BC * D; idx += 128u) {
                uint c = idx / D; uint d = idx % D;
                int j = kt + int(c);
                float x = 0.0;
                if (j < hi) { x = ldv(vb0 + uint(j) * prm[15] + d); }
                KVs[c * DP + d] = x;
            }
            float mt = ninf();
            for (uint c = 0u; c < BC; c++) { mt = max(mt, S[r * BC + c]); }
            float m_new = max(m, mt);
            barrier();
            for (uint c = lane; c < BC; c += TPR) {
                float s = S[r * BC + c];
                float p = 0.0;
                if (s != ninf()) { p = exp(s - m_new); }
                S[r * BC + c] = p;
            }
            barrier();
            float alpha = 0.0;
            if (m != ninf()) { alpha = exp(m - m_new); }
            float lt = 0.0;
            for (uint c = 0u; c < BC; c++) { lt += S[r * BC + c]; }
            l = l * alpha + lt;
            for (uint a = 0u; a < NACC; a++) { acc[a] *= alpha; }
            for (uint c = 0u; c < BC; c++) {
                float p = S[r * BC + c];
                if (p != 0.0) {
                    for (uint a = 0u; a < NACC; a++) {
                        uint d = lane + a * TPR;
                        if (d < D) { acc[a] += p * KVs[c * DP + d]; }
                    }
                }
            }
            m = m_new;
        }
        if (i < int(lq)) {
            float inv = 0.0;
            if (l > 0.0) { inv = vd / l; }
            uint orow = (q0 + uint(i)) * Hq + h;
            for (uint a = 0u; a < NACC; a++) {
                uint d = lane + a * TPR;
                if (d < D) { o[orow * D + d] = acc[a] * inv; }
            }
        }
    }
}
"#;

// prm: 0 rows, 1 D, 2 n_splits
const COMBINE: &str = r#"
layout(set = 0, binding = 0) readonly buffer POb { float part_o[]; };
layout(set = 0, binding = 1) readonly buffer PLb { float part_lse[]; };
layout(set = 0, binding = 2) buffer Ob { float o[]; };
layout(set = 0, binding = 3) readonly buffer Pb { uint prm[]; };
void main() {
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1]; uint n = prm[2];
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {
        float mx = ninf();
        for (uint s = 0u; s < n; s++) { mx = max(mx, part_lse[s * rows + row]); }
        float wsum = 0.0;
        for (uint s = 0u; s < n; s++) {
            float e = part_lse[s * rows + row];
            if (e != ninf()) { wsum += exp(e - mx); }
        }
        for (uint d = t; d < D; d += 128u) {
            float acc = 0.0;
            for (uint s = 0u; s < n; s++) {
                float e = part_lse[s * rows + row];
                if (e != ninf()) { acc += exp(e - mx) * part_o[(s * rows + row) * D + d]; }
            }
            float r = 0.0;
            if (wsum > 0.0) { r = acc / wsum; }
            o[row * D + d] = r;
        }
    }
}
"#;

const TQ_HELPERS: &str = r#"
uint tq_bits_at(uint base, uint pos, uint nb) {
    uint wi = base + (pos >> 5u);
    uint sh = pos & 31u;
    uint val = WORDS[wi] >> sh;
    if (sh + nb > 32u) { val |= WORDS[wi + 1u] << (32u - sh); }
    return val & ((1u << nb) - 1u);
}
uint tq_byte_at(uint base, uint kb) { return (WORDS[base + (kb >> 2u)] >> ((kb & 3u) * 8u)) & 0xffu; }
float tq_elem(uint row, uint j, uint mode, uint cb, uint cw, uint cent) {
    uint c = tq_bits_at(row, j * cb, cb);
    uint aux = row + cw;
    if (mode <= 1u) { return CENT(cent, c); }
    if (mode == 2u) { return fp4_to_f(c) * e8m0_to_f(tq_byte_at(aux, j >> 5u)); }
    if (mode == 3u) { return fp4_to_f(c) * e4m3_to_f(tq_byte_at(aux, j >> 4u)); }
    if (mode == 4u) { return e4m3_to_f(c) * e8m0_to_f(tq_byte_at(aux, j >> 5u)); }
    return e4m3_to_f(c);
}
"#;

// prm: 0 rows, 1 D, 2 mode, 3 cb, 4 code_words, 5 aux_words, 6 row_words, 7 src_off, 8 tab_off
const TQ_QUANT: &str = r#"
layout(set = 0, binding = 0) readonly buffer Xb { float x[]; };
layout(set = 0, binding = 1) buffer Ob { uint outw[]; };
layout(set = 0, binding = 2) readonly buffer Tb { float tab[]; };
layout(set = 0, binding = 3) readonly buffer Pb { uint prm[]; };
shared float y[1024];
shared uint codes[1024];
shared uint sc[64];
uint pack_word(uint w, uint cb, uint D) {
    uint bit0 = w * 32u;
    uint e0 = bit0 / cb;
    uint e1 = min((bit0 + 31u) / cb + 1u, D);
    uint mask = (1u << cb) - 1u;
    uint word = 0u;
    for (uint e = e0; e < e1; e++) {
        int sh = int(e * cb) - int(bit0);
        uint c = codes[e] & mask;
        if (sh >= 0) { word |= c << uint(sh); } else { word |= c >> uint(-sh); }
    }
    return word;
}
void main() {
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1]; uint mode = prm[2]; uint cb = prm[3];
    uint cw = prm[4]; uint aw = prm[5]; uint rw = prm[6];
    uint tb = prm[8];
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {
        barrier();
        uint xb = prm[7] + row * D;
        float ss = 0.0; float am = 0.0;
        for (uint j = t; j < D; j += 128u) {
            float v = x[xb + j];
            y[j] = v;
            ss += v * v;
            am = max(am, abs(v));
        }
        ss = block_sum(ss);
        am = block_max(am);
        float rsc0 = 1.0;
        if (mode <= 1u) {
            float norm = sqrt(ss);
            float inv = 0.0;
            if (norm > 0.0) { inv = 1.0 / norm; }
            uint nb = (1u << cb) - 1u;
            rsc0 = norm;
            for (uint j = t; j < D; j += 128u) {
                float u = y[j] * inv;
                uint idx = 0u;
                for (uint bi = 0u; bi < nb; bi++) { if (u >= tab[tb + 16u + bi]) { idx += 1u; } }
                codes[j] = idx;
            }
        } else if (mode == 2u || mode == 4u) {
            int emax = 8;
            if (mode == 2u) { emax = 2; }
            for (uint bk = t; bk < (D >> 5u); bk += 128u) {
                float bm = 0.0;
                for (uint e = 0u; e < 32u; e++) { bm = max(bm, abs(y[bk * 32u + e])); }
                sc[bk] = mx_scale_code(bm, emax);
            }
            barrier();
            for (uint j = t; j < D; j += 128u) {
                uint code = sc[j >> 5u];
                uint c = 0u;
                if (code != 0u) {
                    float s = e8m0_to_f(code);
                    if (mode == 2u) { c = f_to_fp4(y[j] / s); } else { c = f_to_e4m3(y[j] / s); }
                }
                codes[j] = c;
            }
        } else if (mode == 3u) {
            float g = 1.0;
            if (am > 0.0) { g = am / 2688.0; }
            rsc0 = g;
            for (uint bk = t; bk < (D >> 4u); bk += 128u) {
                float bm = 0.0;
                for (uint e = 0u; e < 16u; e++) { bm = max(bm, abs(y[bk * 16u + e])); }
                sc[bk] = f_to_e4m3(bm / 6.0 / g);
            }
            barrier();
            for (uint j = t; j < D; j += 128u) {
                float eff = e4m3_to_f(sc[j >> 4u]) * g;
                uint c = 0u;
                if (eff > 0.0) { c = f_to_fp4(y[j] / eff); }
                codes[j] = c;
            }
        } else {
            float s = 1.0;
            if (am > 0.0) { s = am / 448.0; }
            rsc0 = s;
            for (uint j = t; j < D; j += 128u) { codes[j] = f_to_e4m3(y[j] / s); }
        }
        barrier();
        uint ob = row * rw;
        for (uint w = t; w < cw; w += 128u) { outw[ob + w] = pack_word(w, cb, D); }
        uint nblk = D >> 5u;
        if (mode == 3u) { nblk = D >> 4u; }
        bool scales = mode == 2u || mode == 3u || mode == 4u;
        for (uint w = t; w < aw; w += 128u) {
            uint word = 0u;
            if (scales) {
                for (uint e = 0u; e < 4u; e++) {
                    if (w * 4u + e < nblk) { word |= (sc[w * 4u + e] & 0xffu) << (8u * e); }
                }
            }
            outw[ob + cw + w] = word;
        }
        if (t == 0u) {
            outw[ob + cw + aw] = floatBitsToUint(rsc0);
            outw[ob + cw + aw + 1u] = 0u;
        }
    }
}
"#;

// prm: 0 rows, 1 D, 2 cb, 3 row_words, 4 rsc_off, 5 src_off, 6 data_off, 7 tab_off
const TQ_RESID: &str = r#"
layout(set = 0, binding = 0) readonly buffer Xb { float x[]; };
layout(set = 0, binding = 1) readonly buffer Db { uint data[]; };
layout(set = 0, binding = 2) readonly buffer Tb { float tab[]; };
layout(set = 0, binding = 3) buffer Rb { float resid[]; };
layout(set = 0, binding = 4) readonly buffer Pb { uint prm[]; };
void main() {
    uint rows = prm[0]; uint D = prm[1]; uint cb = prm[2]; uint rw = prm[3];
    uint n = rows * D;
    for (uint idx = gl_GlobalInvocationID.x; idx < n; idx += gl_NumWorkGroups.x * 128u) {
        uint row = idx / D; uint j = idx % D;
        uint rb = prm[6] + row * rw;
        float norm = uintBitsToFloat(data[rb + prm[4]]);
        float inv = 0.0;
        if (norm > 0.0) { inv = 1.0 / norm; }
        float u = x[prm[5] + idx] * inv;
        resid[idx] = u - tab[prm[7] + tq_bits_at(rb, j * cb, cb)];
    }
}
"#;

// prm: 0 rows, 1 D, 2 code_words, 3 row_words, 4 data_off, 5 proj_off, 6 resid_off
const TQ_SIGNS: &str = r#"
layout(set = 0, binding = 0) readonly buffer Ib { uint inw[]; };
layout(set = 0, binding = 1) readonly buffer PRb { float proj[]; };
layout(set = 0, binding = 2) readonly buffer Rb { float resid[]; };
layout(set = 0, binding = 3) buffer Ob { uint outw[]; };
layout(set = 0, binding = 4) readonly buffer Pb { uint prm[]; };
void main() {
    uint t = gl_LocalInvocationID.x;
    uint rows = prm[0]; uint D = prm[1]; uint cw = prm[2]; uint rw = prm[3];
    uint sw = (D + 31u) / 32u;
    for (uint row = gl_WorkGroupID.x; row < rows; row += gl_NumWorkGroups.x) {
        float ss = 0.0;
        for (uint j = t; j < D; j += 128u) { float v = resid[prm[6] + row * D + j]; ss += v * v; }
        ss = block_sum(ss);
        for (uint w = t; w < rw; w += 128u) {
            uint word = inw[prm[4] + row * rw + w];
            if (w >= cw && w < cw + sw) {
                word = 0u;
                uint base = (w - cw) * 32u;
                for (uint e = 0u; e < 32u && base + e < D; e++) {
                    if (proj[prm[5] + row * D + base + e] >= 0.0) { word |= 1u << e; }
                }
            } else if (w == cw + sw + 1u) {
                word = floatBitsToUint(sqrt(ss));
            }
            outw[row * rw + w] = word;
        }
    }
}
"#;

// prm: 0 rows, 1 D, 2 mode, 3 cb, 4 code_words, 5 row_words, 6 rsc_off, 7 data_off,
// 8 tab_off, 9 with_qjl
const TQ_DEQUANT: &str = r#"
layout(set = 0, binding = 0) readonly buffer Db { uint data[]; };
layout(set = 0, binding = 1) readonly buffer Tb { float tab[]; };
layout(set = 0, binding = 2) buffer Ob { float o[]; };
layout(set = 0, binding = 3) readonly buffer Pb { uint prm[]; };
void main() {
    uint rows = prm[0]; uint D = prm[1]; uint rw = prm[5]; uint ro = prm[6];
    uint n = rows * D;
    for (uint idx = gl_GlobalInvocationID.x; idx < n; idx += gl_NumWorkGroups.x * 128u) {
        uint row = idx / D; uint j = idx % D;
        uint rb = prm[7] + row * rw;
        float r0 = uintBitsToFloat(data[rb + ro]);
        o[idx] = r0 * tq_elem(rb, j, prm[2], prm[3], prm[4], prm[8]);
        if (prm[9] != 0u) {
            float c = r0 * uintBitsToFloat(data[rb + ro + 1u]) * 1.2533141 / float(D);
            uint bit = (data[rb + prm[4] + (j >> 5u)] >> (j & 31u)) & 1u;
            float z = -c;
            if (bit != 0u) { z = c; }
            o[n + idx] = z;
        }
    }
}
"#;

// prm: 0 Lq, 1 Lk, 2 Hq, 3 Hkv, 4 D, 5-7 q strides (b, h, s), 8-10 k strides (b, h, t),
// 11-13 v strides, 14 k_mode, 15 k_cb, 16 k_cw, 17 k_rsc, 18 v_mode, 19 v_cb, 20 v_cw, 21 v_rsc,
// 22 scale, 23 softcap, 24 wl, 25 wr, 26 n_splits, 27 alibi_bs, 28 rows_total, 29 q_off,
// 30 k_off, 31 v_off, 32 tab_off, 33 has_alibi, 34 alibi_off
const TQ_ATTN: &str = r#"
layout(set = 0, binding = 0) readonly buffer Qb { float qx[]; };
layout(set = 0, binding = 1) readonly buffer Kb { uint kd[]; };
layout(set = 0, binding = 2) readonly buffer Vb { uint vd[]; };
layout(set = 0, binding = 3) readonly buffer Tb { float tab[]; };
layout(set = 0, binding = 4) readonly buffer ALb { float alibi[]; };
layout(set = 0, binding = 5) buffer Ob { float o[]; };
layout(set = 0, binding = 6) buffer POb { float part_o[]; };
layout(set = 0, binding = 7) buffer PLb { float part_lse[]; };
layout(set = 0, binding = 8) readonly buffer Pb { uint prm[]; };
shared float qr[256];
shared float qsk[256];
shared float pr[128];
shared float cents[32];
uint kbits(uint base, uint pos, uint nb) {
    uint wi = base + (pos >> 5u); uint sh = pos & 31u;
    uint val = kd[wi] >> sh;
    if (sh + nb > 32u) { val |= kd[wi + 1u] << (32u - sh); }
    return val & ((1u << nb) - 1u);
}
uint vbits(uint base, uint pos, uint nb) {
    uint wi = base + (pos >> 5u); uint sh = pos & 31u;
    uint val = vd[wi] >> sh;
    if (sh + nb > 32u) { val |= vd[wi + 1u] << (32u - sh); }
    return val & ((1u << nb) - 1u);
}
uint kbyte(uint base, uint kb) { return (kd[base + (kb >> 2u)] >> ((kb & 3u) * 8u)) & 0xffu; }
uint vbyte(uint base, uint kb) { return (vd[base + (kb >> 2u)] >> ((kb & 3u) * 8u)) & 0xffu; }
float kelem(uint row, uint j, uint mode, uint cb, uint cw) {
    uint c = kbits(row, j * cb, cb);
    uint aux = row + cw;
    if (mode <= 1u) { return cents[c]; }
    if (mode == 2u) { return fp4_to_f(c) * e8m0_to_f(kbyte(aux, j >> 5u)); }
    if (mode == 3u) { return fp4_to_f(c) * e4m3_to_f(kbyte(aux, j >> 4u)); }
    if (mode == 4u) { return e4m3_to_f(c) * e8m0_to_f(kbyte(aux, j >> 5u)); }
    return e4m3_to_f(c);
}
float velem(uint row, uint j, uint mode, uint cb, uint cw) {
    uint c = vbits(row, j * cb, cb);
    uint aux = row + cw;
    if (mode <= 1u) { return cents[16u + c]; }
    if (mode == 2u) { return fp4_to_f(c) * e8m0_to_f(vbyte(aux, j >> 5u)); }
    if (mode == 3u) { return fp4_to_f(c) * e4m3_to_f(vbyte(aux, j >> 4u)); }
    if (mode == 4u) { return e4m3_to_f(c) * e8m0_to_f(vbyte(aux, j >> 5u)); }
    return e4m3_to_f(c);
}
void main() {
    uint t = gl_LocalInvocationID.x;
    uint split = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint Lq = prm[0]; uint Lk = prm[1]; uint Hq = prm[2]; uint Hkv = prm[3]; uint D = prm[4];
    uint k_mode = prm[14]; uint k_cb = prm[15]; uint k_cw = prm[16]; uint k_rsc = prm[17];
    uint v_mode = prm[18]; uint v_cb = prm[19]; uint v_cw = prm[20]; uint v_rsc = prm[21];
    float scale = uintBitsToFloat(prm[22]); float softcap = uintBitsToFloat(prm[23]);
    int wl = int(prm[24]); int wr = int(prm[25]);
    uint n_splits = prm[26];
    bool prod = k_mode == 1u;
    uint group = Hq / Hkv;
    for (uint rowi = gl_WorkGroupID.x; rowi < Lq * Hq; rowi += gl_NumWorkGroups.x) {
        uint i = rowi / Hq;
        uint h = rowi % Hq;
        uint hk = h / group;
        uint qb = prm[29] + b * prm[5] + h * prm[6] + i * prm[7];
        barrier();
        for (uint d = t; d < D; d += 128u) {
            qr[d] = qx[qb + d];
            float sq = 0.0;
            if (prod) { sq = qx[qb + D + d]; }
            qsk[d] = sq;
        }
        if (t < 16u) {
            cents[t] = tab[prm[32] + t];
            cents[16u + t] = tab[prm[32] + 32u + t];
        }
        barrier();
        int pos = int(i) + int(Lk) - int(Lq);
        int lo = 0; int hi = int(Lk);
        if (wl >= 0) { lo = max(pos - wl, 0); }
        if (wr >= 0) { hi = min(pos + wr + 1, int(Lk)); }
        if (lo > int(Lk)) { lo = int(Lk); }
        if (hi < lo) { hi = lo; }
        int chunk = (hi - lo + int(n_splits) - 1) / int(n_splits);
        int s_lo = lo + int(split) * chunk;
        int s_hi = min(s_lo + chunk, hi);
        if (s_lo > hi) { s_lo = hi; }
        if (s_hi < s_lo) { s_hi = s_lo; }
        float slope = 0.0;
        if (prm[33] != 0u) { slope = alibi[prm[34] + b * prm[27] + h]; }
        uint kbase = prm[30] + b * prm[8] + hk * prm[9];
        uint vbase = prm[31] + b * prm[11] + hk * prm[12];
        float m = ninf(); float l = 0.0; float acc0 = 0.0; float acc1 = 0.0;
        for (int kb = s_lo; kb < s_hi; kb += 128) {
            int j = kb + int(t);
            float s = ninf();
            float vscale = 0.0;
            if (j < s_hi) {
                uint kr = kbase + uint(j) * prm[10];
                float dot = 0.0;
                for (uint d = 0u; d < D; d++) { dot += kelem(kr, d, k_mode, k_cb, k_cw) * qr[d]; }
                float r0 = uintBitsToFloat(kd[kr + k_rsc]);
                float sc = r0 * dot;
                if (prod) {
                    float sd = 0.0;
                    for (uint d = 0u; d < D; d++) {
                        if (((kd[kr + k_cw + (d >> 5u)] >> (d & 31u)) & 1u) != 0u) { sd += qsk[d]; } else { sd -= qsk[d]; }
                    }
                    sc += r0 * uintBitsToFloat(kd[kr + k_rsc + 1u]) * 1.2533141 / float(D) * sd;
                }
                s = logit(sc * scale, softcap, slope, pos - j);
                vscale = uintBitsToFloat(vd[vbase + uint(j) * prm[13] + v_rsc]);
            }
            float m_new = max(m, block_max(s));
            float p = 0.0;
            if (s != ninf()) { p = exp(s - m_new); }
            pr[t] = p * vscale;
            float lt = block_sum(p);
            float alpha = 0.0;
            if (m != ninf()) { alpha = exp(m - m_new); }
            l = l * alpha + lt;
            acc0 *= alpha; acc1 *= alpha;
            int cnt = min(128, s_hi - kb);
            for (int u = 0; u < cnt; u++) {
                float pu = pr[u];
                if (pu != 0.0) {
                    uint vr = vbase + uint(kb + u) * prm[13];
                    if (t < D) { acc0 += pu * velem(vr, t, v_mode, v_cb, v_cw); }
                    if (t + 128u < D) { acc1 += pu * velem(vr, t + 128u, v_mode, v_cb, v_cw); }
                }
            }
            m = m_new;
            barrier();
        }
        float inv = 0.0;
        if (l > 0.0) { inv = 1.0 / l; }
        uint orow = (b * Hq + h) * Lq + i;
        if (n_splits == 1u) {
            if (t < D) { o[orow * D + t] = acc0 * inv; }
            if (t + 128u < D) { o[orow * D + t + 128u] = acc1 * inv; }
        } else {
            uint prow = split * prm[28] + orow;
            if (t < D) { part_o[prow * D + t] = acc0 * inv; }
            if (t + 128u < D) { part_o[prow * D + t + 128u] = acc1 * inv; }
            if (t == 0u) { if (l > 0.0) { part_lse[prow] = m + log(l); } else { part_lse[prow] = ninf(); } }
        }
    }
}
"#;

fn header() -> String {
    "#version 450\nlayout(local_size_x = 128) in;\n".to_string()
}

fn fa_row_source(fp8: bool) -> String {
    let ktype = if fp8 { "uint" } else { "float" };
    let load = if fp8 { LOAD_FP8 } else { LOAD_F32 };
    format!(
        "{}{}{}{}{}{}{}",
        header(),
        FA_BINDINGS.replace("KTYPE", ktype),
        COMMON,
        REDUCE,
        FA_PRELUDE,
        load,
        FA_ROW
    )
}

fn fa_tile_source(fp8: bool, br: usize, bc: usize, dmax: usize) -> String {
    let ktype = if fp8 { "uint" } else { "float" };
    let load = if fp8 { LOAD_FP8 } else { LOAD_F32 };
    let nacc = dmax / (128 / br);
    let consts = format!(
        "const uint BR = {br}u;\nconst uint BC = {bc}u;\nconst uint DMAX = {dmax}u;\nconst uint NACC = {nacc}u;\n"
    );
    format!(
        "{}{}{}{}{}{}{}",
        header(),
        consts,
        FA_BINDINGS.replace("KTYPE", ktype),
        COMMON,
        FA_PRELUDE,
        load,
        FA_TILE
    )
}

fn combine_source() -> String {
    format!("{}{}{}", header(), COMMON, COMBINE)
}

fn tq_source(body: &str, words: &str) -> String {
    let helpers = TQ_HELPERS
        .replace("WORDS", words)
        .replace("CENT(cent, c)", "tab[cent + c]");
    format!(
        "{}{}{}",
        header(),
        COMMON,
        body.replace("//HELPERS", &helpers)
    )
}

/// Params buffer for a dispatch.
fn params(dev: &VulkanDevice, vals: &[u32]) -> Result<VulkanStorage> {
    let mut p = dev.alloc_buffer(vals.len(), DType::U32)?;
    p.as_mut_slice::<u32>()?.copy_from_slice(vals);
    Ok(p)
}

fn fits(n: usize) -> Result<u32> {
    if n > u32::MAX as usize {
        bail!("vulkan accel: index {n} does not fit the 32-bit kernels")
    }
    Ok(n as u32)
}

fn groups(n: usize) -> [u32; 3] {
    [n.clamp(1, MAX_GROUPS) as u32, 1, 1]
}

impl VulkanStorage {
    /// Fused attention of `self` (f32 queries) over `k` / `v` (f32 or FP8). Output f32.
    pub fn flash_attn(
        &self,
        k: &Self,
        v: &Self,
        geom: &AttnGeometry,
        p: &FlashAttnParams,
        extra: &FaExtra<'_, Self>,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            bail!(
                "vulkan flash-attn: queries must be f32, got {:?}",
                self.dtype
            )
        }
        let fp8 = match (k.dtype, v.dtype) {
            (DType::F32, DType::F32) => false,
            (DType::F8E4M3, DType::F8E4M3) => true,
            (a, b) => bail!("vulkan flash-attn: unsupported k/v dtypes {a:?}/{b:?}"),
        };
        let dev = &self.device;
        let prof = dev.accel_profile();
        let plan = att::plan(geom, p, prof.max_shared_mem, prof.compute_units);
        let rows = geom.out_rows();
        let d = geom.d;
        let out = dev.alloc_buffer(rows * d, DType::F32)?;
        if rows == 0 || d == 0 {
            return Ok(out);
        }
        let (_, _, splits) = att::launch_dims(geom, &plan);
        fits(splits * rows * d)?;
        fits(self.numel)?;
        fits(k.numel.max(v.numel))?;
        let (part_o, part_lse) = if splits > 1 {
            (
                Some(dev.alloc_buffer(splits * rows * d, DType::F32)?),
                Some(dev.alloc_buffer(splits * rows, DType::F32)?),
            )
        } else {
            (None, None)
        };
        let b = |x: f32| x.to_bits();
        let (has_alibi, alibi_off, alibi_bs) = match extra.alibi {
            Some((_, off, bs)) => (1, off, bs),
            None => (0, 0, 0),
        };
        let (cq_off, ck_off) = match extra.cu_seqlens {
            Some((_, oq, _, ok)) => (oq, ok),
            None => (0, 0),
        };
        let mut vals = vec![
            geom.lq as u32,
            geom.lk as u32,
            geom.hq as u32,
            geom.hkv as u32,
            d as u32,
            fits(geom.q_offset)?,
            fits(geom.k_offset)?,
            fits(geom.v_offset)?,
        ];
        for s in [geom.q_strides, geom.k_strides, geom.v_strides] {
            for x in s {
                vals.push(fits(x)?);
            }
        }
        vals.extend([
            b(p.softmax_scale),
            b(p.softcap),
            p.window_left as u32,
            p.window_right as u32,
            b(p.k_descale),
            b(p.v_descale),
            splits as u32,
            alibi_bs as u32,
            geom.varlen as u32,
            fits(rows)?,
            has_alibi,
            fits(alibi_off)?,
            fits(cq_off)?,
            fits(ck_off)?,
        ]);
        let prm = params(dev, &vals)?;
        let alibi = extra.alibi.map(|(s, _, _)| s).unwrap_or(self);
        let (cu_q, cu_k) = match extra.cu_seqlens {
            Some((sq, _, sk, _)) => (sq, sk),
            None => (&prm, &prm),
        };
        let po = part_o.as_ref().unwrap_or(&out);
        let pl = part_lse.as_ref().unwrap_or(&out);
        let binds = [self, k, v, &out, po, pl, alibi, cu_q, cu_k, &prm];
        match plan {
            FaPlan::Row { splits } => {
                let key = format!("accel_fa_row_{}", if fp8 { "fp8" } else { "f32" });
                let kern = kernel(dev, &key, 10, 0, || fa_row_source(fp8))?;
                let grid = [
                    (geom.lq * geom.hq).clamp(1, MAX_GROUPS) as u32,
                    splits as u32,
                    geom.batch as u32,
                ];
                run(dev, &kern, &binds, &[], grid)?;
            }
            FaPlan::Tile { dmax, br, bc, .. } => {
                let key = format!(
                    "accel_fa_tile_{br}_{bc}_{dmax}_{}",
                    if fp8 { "fp8" } else { "f32" }
                );
                let kern = kernel(dev, &key, 10, 0, || fa_tile_source(fp8, br, bc, dmax))?;
                let grid = [
                    geom.lq.div_ceil(br).clamp(1, MAX_GROUPS) as u32,
                    geom.hq as u32,
                    geom.batch as u32,
                ];
                run(dev, &kern, &binds, &[], grid)?;
            }
        }
        if let (Some(po), Some(pl)) = (part_o.as_ref(), part_lse.as_ref()) {
            let kern = kernel(dev, "accel_fa_combine", 4, 0, combine_source)?;
            let prm = params(dev, &[fits(rows)?, d as u32, splits as u32])?;
            run(dev, &kern, &[po, pl, &out, &prm], &[], groups(rows))?;
        }
        Ok(out)
    }

    /// See `RocmStorage::tq_quantize`; `self` must be f32.
    pub fn tq_quantize(
        &self,
        offset: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            bail!(
                "vulkan turbo-quant: input must be f32, got {:?}",
                self.dtype
            )
        }
        let dev = &self.device;
        let out = dev.alloc_buffer(a.rows * a.row_words, DType::U32)?;
        if a.rows > 0 {
            fits(a.rows * a.row_words)?;
            let kern = kernel(dev, "accel_tq_quant", 4, 0, || {
                format!("{}{}{}{}", header(), COMMON, REDUCE, TQ_QUANT)
            })?;
            let prm = params(
                dev,
                &[
                    a.rows as u32,
                    a.dim as u32,
                    a.mode,
                    a.code_bits,
                    a.code_words as u32,
                    a.aux_words as u32,
                    a.row_words as u32,
                    fits(offset)?,
                    fits(tab_off)?,
                ],
            )?;
            run(dev, &kern, &[self, &out, tab, &prm], &[], groups(a.rows))?;
        }
        Ok(out)
    }

    /// See `RocmStorage::tq_residual`; `self` must be f32.
    pub fn tq_residual(
        &self,
        offset: usize,
        data: &Self,
        data_off: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            bail!(
                "vulkan turbo-quant: input must be f32, got {:?}",
                self.dtype
            )
        }
        let dev = &self.device;
        let n = a.rows * a.dim;
        let out = dev.alloc_buffer(n, DType::F32)?;
        if n > 0 {
            fits(n)?;
            let kern = kernel(dev, "accel_tq_resid", 5, 0, || {
                let body = format!("{TQ_RESID}").replace("void main()", "//HELPERS\nvoid main()");
                tq_source(&body, "data")
            })?;
            let prm = params(
                dev,
                &[
                    a.rows as u32,
                    a.dim as u32,
                    a.code_bits,
                    a.row_words as u32,
                    a.rsc_offset() as u32,
                    fits(offset)?,
                    fits(data_off)?,
                    fits(tab_off)?,
                ],
            )?;
            run(
                dev,
                &kern,
                &[self, data, tab, &out, &prm],
                &[],
                groups(n.div_ceil(128)),
            )?;
        }
        Ok(out)
    }

    /// See `RocmStorage::tq_signs`.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_signs(
        &self,
        data_off: usize,
        proj: &Self,
        proj_off: usize,
        resid: &Self,
        resid_off: usize,
        a: &TqRowArgs,
    ) -> Result<Self> {
        let dev = &self.device;
        let out = dev.alloc_buffer(a.rows * a.row_words, DType::U32)?;
        if a.rows > 0 {
            let kern = kernel(dev, "accel_tq_signs", 5, 0, || {
                format!("{}{}{}{}", header(), COMMON, REDUCE, TQ_SIGNS)
            })?;
            let prm = params(
                dev,
                &[
                    a.rows as u32,
                    a.dim as u32,
                    a.code_words as u32,
                    a.row_words as u32,
                    fits(data_off)?,
                    fits(proj_off)?,
                    fits(resid_off)?,
                ],
            )?;
            run(
                dev,
                &kern,
                &[self, proj, resid, &out, &prm],
                &[],
                groups(a.rows),
            )?;
        }
        Ok(out)
    }

    /// See `RocmStorage::tq_dequantize`.
    pub fn tq_dequantize(
        &self,
        data_off: usize,
        tab: &Self,
        tab_off: usize,
        a: &TqRowArgs,
        with_qjl: bool,
    ) -> Result<Self> {
        let dev = &self.device;
        let n = a.rows * a.dim;
        let parts = if with_qjl { 2 } else { 1 };
        let out = dev.alloc_buffer(parts * n, DType::F32)?;
        if n > 0 {
            fits(parts * n)?;
            let kern = kernel(dev, "accel_tq_dequant", 4, 0, || {
                let body = TQ_DEQUANT.replace("void main()", "//HELPERS\nvoid main()");
                tq_source(&body, "data")
            })?;
            let prm = params(
                dev,
                &[
                    a.rows as u32,
                    a.dim as u32,
                    a.mode,
                    a.code_bits,
                    a.code_words as u32,
                    a.row_words as u32,
                    a.rsc_offset() as u32,
                    fits(data_off)?,
                    fits(tab_off)?,
                    with_qjl as u32,
                ],
            )?;
            run(
                dev,
                &kern,
                &[self, tab, &out, &prm],
                &[],
                groups(n.div_ceil(128)),
            )?;
        }
        Ok(out)
    }

    /// See `RocmStorage::tq_attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn tq_attention(
        &self,
        kd: &Self,
        vd: &Self,
        tab: &Self,
        tab_off: usize,
        alibi: Option<(&Self, usize)>,
        a: &TqAttnArgs,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            bail!(
                "turbo-quant attention: queries must be f32, got {:?}",
                self.dtype
            )
        }
        let dev = &self.device;
        let rows = a.out_rows();
        let d = a.d;
        let out = dev.alloc_buffer(rows * d, DType::F32)?;
        if rows == 0 || d == 0 {
            return Ok(out);
        }
        let splits = a.splits.max(1);
        fits(splits * rows * d)?;
        fits(self.numel)?;
        fits(kd.numel.max(vd.numel))?;
        let (part_o, part_lse) = if splits > 1 {
            (
                Some(dev.alloc_buffer(splits * rows * d, DType::F32)?),
                Some(dev.alloc_buffer(splits * rows, DType::F32)?),
            )
        } else {
            (None, None)
        };
        let vals = [
            a.lq as u32,
            a.lk as u32,
            a.hq as u32,
            a.hkv as u32,
            d as u32,
            fits(a.q_strides[0])?,
            fits(a.q_strides[1])?,
            fits(a.q_strides[2])?,
            fits(a.k_strides[0])?,
            fits(a.k_strides[1])?,
            fits(a.k_strides[2])?,
            fits(a.v_strides[0])?,
            fits(a.v_strides[1])?,
            fits(a.v_strides[2])?,
            a.k.mode,
            a.k.code_bits,
            a.k.code_words as u32,
            a.k.rsc_offset as u32,
            a.v.mode,
            a.v.code_bits,
            a.v.code_words as u32,
            a.v.rsc_offset as u32,
            a.scale.to_bits(),
            a.softcap.to_bits(),
            a.window_left as u32,
            a.window_right as u32,
            splits as u32,
            a.alibi_bstride as u32,
            fits(rows)?,
            fits(a.q_offset)?,
            fits(a.k_offset)?,
            fits(a.v_offset)?,
            fits(tab_off)?,
            alibi.is_some() as u32,
            fits(alibi.map(|(_, o)| o).unwrap_or(0))?,
        ];
        let prm = params(dev, &vals)?;
        let kern = kernel(dev, "accel_tq_attn", 9, 0, || {
            format!("{}{}{}{}", header(), COMMON, REDUCE, TQ_ATTN)
        })?;
        let al = alibi.map(|(s, _)| s).unwrap_or(self);
        let po = part_o.as_ref().unwrap_or(&out);
        let pl = part_lse.as_ref().unwrap_or(&out);
        let grid = [
            (a.lq * a.hq).clamp(1, MAX_GROUPS) as u32,
            splits as u32,
            a.b as u32,
        ];
        run(
            dev,
            &kern,
            &[self, kd, vd, tab, al, &out, po, pl, &prm],
            &[],
            grid,
        )?;
        if let (Some(po), Some(pl)) = (part_o.as_ref(), part_lse.as_ref()) {
            let kern = kernel(dev, "accel_fa_combine", 4, 0, combine_source)?;
            let prm = params(dev, &[fits(rows)?, d as u32, splits as u32])?;
            run(dev, &kern, &[po, pl, &out, &prm], &[], groups(rows))?;
        }
        Ok(out)
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
    fn shaders_compile() {
        for fp8 in [false, true] {
            validate(&fa_row_source(fp8));
            for (br, bc, dmax) in [(32, 32, 64), (16, 32, 128), (8, 16, 256), (64, 64, 64)] {
                validate(&fa_tile_source(fp8, br, bc, dmax));
            }
        }
        validate(&combine_source());
        validate(&format!("{}{}{}{}", header(), COMMON, REDUCE, TQ_QUANT));
        validate(&tq_source(
            &TQ_RESID.replace("void main()", "//HELPERS\nvoid main()"),
            "data",
        ));
        validate(&format!("{}{}{}{}", header(), COMMON, REDUCE, TQ_SIGNS));
        validate(&tq_source(
            &TQ_DEQUANT.replace("void main()", "//HELPERS\nvoid main()"),
            "data",
        ));
        validate(&format!("{}{}{}{}", header(), COMMON, REDUCE, TQ_ATTN));
    }
}
