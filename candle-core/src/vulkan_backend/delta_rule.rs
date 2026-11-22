//! 2-Stage precision Gated DeltaNet (GDN) compute shaders on the Vulkan backend.
//!
//! - Stage 1 (`vk_gdn_conv_{f32,f16,bf16}`): causal depthwise conv1d + SiLU over
//!   `[conv_state ; mixed_qkv]` producing `qkv_conv [B, T, C]` (F32) and updated
//!   `conv_state_out [B, C, K-1]`.
//! - Stage 2 (`vk_gdn_scan_d{64,128,256}_{ab_tag}`): column-in-thread (`s_col[DK_CAP]`)
//!   sequential gated delta rule scan with `local_size_x = 64` (~1.25–2.25 KiB smem),
//!   exact IEEE FP32 softplus (`> 20.0`), L2 norm, and batched output write.
//!
//! Enabled per discrete Vulkan device after a one-time `self_check` against the CPU oracle.

use super::shaders::{self, kernel, run, GLSL_PACK16_HELPERS};
use super::{VulkanDevice, VulkanStorage};
use crate::accel::delta::{
    dk_cap, GdnScanArgs, GDN_FLAG_AB_F32, GDN_FLAG_L2NORM, GDN_FLAG_TILED, GDN_MAX_HEAD_K_DIM,
};
use crate::backend::BackendStorage;
use crate::{bail, DType, Result};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const MAX_GROUPS: usize = 65535;

fn fits(n: usize) -> Result<u32> {
    if n > u32::MAX as usize {
        bail!("vulkan delta_rule: index {n} exceeds 32-bit kernel range")
    }
    Ok(n as u32)
}

fn dtype_tag(dt: DType) -> Result<&'static str> {
    match dt {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => bail!("vulkan delta_rule: unsupported dtype {other:?}"),
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

// Stage 1: causal depthwise conv1d + SiLU + conv_state update
// Bindings: 0 x, 1 cw, 2 cb, 3 cs, 4 qkv_conv (f32), 5 cs_out (f32), 6 prm
// prm: 0 B, 1 T, 2 C, 3 K, 4 x_off, 5 xs0, 6 xs1, 7 cw_off, 8 has_cb, 9 cb_off, 10 has_cs, 11 cs_off
fn gdn_conv_src(dtype: DType) -> String {
    let (decl, ld_x) = load_spec(dtype, "x");
    let (_, ld_cs) = load_spec(dtype, "cs");
    format!(
        r#"#version 450
layout(local_size_x = 128) in;
layout(set = 0, binding = 0) readonly buffer Xb {{ {decl} x[]; }};
layout(set = 0, binding = 1) readonly buffer CWb {{ float cw[]; }};
layout(set = 0, binding = 2) readonly buffer CBb {{ float cb[]; }};
layout(set = 0, binding = 3) readonly buffer CSb {{ {decl} cs[]; }};
layout(set = 0, binding = 4) buffer QKVb {{ float qkv_conv[]; }};
layout(set = 0, binding = 5) buffer CSOb {{ float cs_out[]; }};
layout(set = 0, binding = 6) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
float load_x(uint idx) {{ return {ld_x}; }}
float load_cs(uint idx) {{ return {ld_cs}; }}
float xe(uint bi, uint ch, uint i, uint C, uint km1, bool has_cs) {{
    if (i < km1) {{
        if (!has_cs) {{ return 0.0; }}
        return load_cs(prm[11] + (bi * C + ch) * km1 + i);
    }}
    return load_x(prm[4] + bi * prm[5] + (i - km1) * prm[6] + ch);
}}
void main() {{
    uint B = prm[0]; uint T = prm[1]; uint C = prm[2]; uint K = prm[3];
    uint km1 = K - 1u;
    bool has_cb = prm[8] != 0u;
    bool has_cs = prm[10] != 0u;
    uint total = B * C;
    uint stride = gl_NumWorkGroups.x * 128u;
    for (uint bc = gl_GlobalInvocationID.x; bc < total; bc += stride) {{
        uint bi = bc / C;
        uint ch = bc - bi * C;
        float bias = has_cb ? cb[prm[9] + ch] : 0.0;
        uint wbase = prm[7] + ch * K;
        for (uint tt = 0u; tt < T; tt++) {{
            float acc = bias;
            for (uint j = 0u; j < K; j++) {{
                acc += cw[wbase + j] * xe(bi, ch, tt + j, C, km1, has_cs);
            }}
            qkv_conv[(bi * T + tt) * C + ch] = acc / (1.0 + exp(-acc));
        }}
        uint cs_dst = (bi * C + ch) * km1;
        for (uint m = 0u; m < km1; m++) {{
            cs_out[cs_dst + m] = xe(bi, ch, T + m, C, km1, has_cs);
        }}
    }}
}}
"#
    )
}

// Stage 2: column-in-thread Gated DeltaNet scan
// Bindings: 0 qkv_conv (f32), 1 a, 2 b, 3 neg_a (f32), 4 dt_bias (f32), 5 s_in (f32),
//           6 out_f32 (f32), 7 s_out (f32), 8 prm
// prm: 0 B, 1 T, 2 Hk, 3 Hv, 4 Dk, 5 Dv, 6 flags, 7 eps_bits, 8 scale_bits,
//      9 a_off, 10 as0, 11 as1, 12 b_off, 13 bs0, 14 bs1,
//      15 na_off, 16 dtb_off, 17 has_s, 18 s_off
fn gdn_scan_src(dk_cap_val: usize, ab_dtype: DType) -> String {
    let (ab_decl, ld_a) = load_spec(ab_dtype, "a_in");
    let (_, ld_b) = load_spec(ab_dtype, "b_in");
    format!(
        r#"#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer QKVb {{ float qkv[]; }};
layout(set = 0, binding = 1) readonly buffer Ab {{ {ab_decl} a_in[]; }};
layout(set = 0, binding = 2) readonly buffer Bb {{ {ab_decl} b_in[]; }};
layout(set = 0, binding = 3) readonly buffer NAb {{ float neg_a[]; }};
layout(set = 0, binding = 4) readonly buffer DTb {{ float dt_bias[]; }};
layout(set = 0, binding = 5) readonly buffer SIb {{ float s_in[]; }};
layout(set = 0, binding = 6) buffer Ob {{ float out_f32[]; }};
layout(set = 0, binding = 7) buffer SOb {{ float s_out[]; }};
layout(set = 0, binding = 8) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
float load_a(uint idx) {{ return {ld_a}; }}
float load_b(uint idx) {{ return {ld_b}; }}
shared float sh_q[{dk_cap_val}];
shared float sh_k[{dk_cap_val}];
shared float red_q[64];
shared float red_k[64];
vec2 wg_sum64_pair(float vq, float vk) {{
    uint tid = gl_LocalInvocationID.x;
    red_q[tid] = vq;
    red_k[tid] = vk;
    barrier();
    for (uint s = 32u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            red_q[tid] += red_q[tid + s];
            red_k[tid] += red_k[tid + s];
        }}
        barrier();
    }}
    vec2 r = vec2(red_q[0], red_k[0]);
    barrier();
    return r;
}}
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint c_tile = gl_WorkGroupID.x;
    uint h = gl_WorkGroupID.y;
    uint bi = gl_WorkGroupID.z;
    uint B = prm[0]; uint T = prm[1]; uint Hk = prm[2]; uint Hv = prm[3];
    uint Dk = prm[4]; uint Dv = prm[5];
    if (bi >= B || h >= Hv) {{ return; }}
    uint flags = prm[6];
    bool use_l2 = (flags & {GDN_FLAG_L2NORM}u) != 0u;
    bool tiled = (flags & {GDN_FLAG_TILED}u) != 0u;
    float eps = uintBitsToFloat(prm[7]);
    float scale = uintBitsToFloat(prm[8]);
    uint key_dim = Hk * Dk;
    uint C = 2u * key_dim + Hv * Dv;
    uint kh = tiled ? (h % Hk) : (h / (Hv / Hk));
    uint col = c_tile * 64u + tid;
    bool active = col < Dv;
    uint bh = bi * Hv + h;
    bool has_s = prm[17] != 0u;
    float s_col[{dk_cap_val}];
    for (uint i = 0u; i < {dk_cap_val}u; i++) {{
        if (i < Dk) {{
            s_col[i] = (active && has_s) ? s_in[prm[18] + (bh * Dk + i) * Dv + col] : 0.0;
        }} else {{
            s_col[i] = 0.0;
        }}
    }}
    float na_h = neg_a[prm[15] + h];
    float dtb_h = dt_bias[prm[16] + h];
    for (uint tt = 0u; tt < T; tt++) {{
        uint qkv_t = (bi * T + tt) * C;
        float sq = 0.0; float sk = 0.0;
        for (uint i = tid; i < Dk; i += 64u) {{
            float qv = qkv[qkv_t + kh * Dk + i];
            float kv = qkv[qkv_t + key_dim + kh * Dk + i];
            sh_q[i] = qv;
            sh_k[i] = kv;
            sq += qv * qv;
            sk += kv * kv;
        }}
        if (use_l2) {{
            vec2 sums = wg_sum64_pair(sq, sk);
            float inv_q = inversesqrt(sums.x + eps) * scale;
            float inv_k = inversesqrt(sums.y + eps);
            for (uint i = tid; i < Dk; i += 64u) {{
                sh_q[i] *= inv_q;
                sh_k[i] *= inv_k;
            }}
        }} else {{
            barrier();
            for (uint i = tid; i < Dk; i += 64u) {{
                sh_q[i] *= scale;
            }}
        }}
        barrier();
        if (active) {{
            float av = load_a(prm[9] + bi * prm[10] + tt * prm[11] + h);
            float bv = load_b(prm[12] + bi * prm[13] + tt * prm[14] + h);
            float x_sp = av + dtb_h;
            float ex = exp(x_sp);
            float u = 1.0 + ex;
            float sp = (x_sp > 20.0) ? x_sp : ((u == 1.0) ? ex : (log(u) * (ex / (u - 1.0))));
            float eg = exp(na_h * sp);
            float beta = 1.0 / (1.0 + exp(-bv));
            float vv = qkv[qkv_t + 2u * key_dim + h * Dv + col];
            float kv_mem = 0.0;
            for (uint i = 0u; i < {dk_cap_val}u; i++) {{
                if (i < Dk) {{
                    float s_i = s_col[i] * eg;
                    s_col[i] = s_i;
                    kv_mem = fma(s_i, sh_k[i], kv_mem);
                }}
            }}
            float delta = (vv - kv_mem) * beta;
            float ov = 0.0;
            for (uint i = 0u; i < {dk_cap_val}u; i++) {{
                if (i < Dk) {{
                    float s_i = fma(sh_k[i], delta, s_col[i]);
                    s_col[i] = s_i;
                    ov = fma(s_i, sh_q[i], ov);
                }}
            }}
            out_f32[((bi * T + tt) * Hv + h) * Dv + col] = ov;
        }}
        barrier();
    }}
    if (active) {{
        for (uint i = 0u; i < {dk_cap_val}u; i++) {{
            if (i < Dk) {{
                s_out[(bh * Dk + i) * Dv + col] = s_col[i];
            }}
        }}
    }}
}}
"#
    )
}

// Race-free paired f32 -> f16 / bf16 packer (works for any n; pads trailing odd element with 0)
fn pack_f32_to_p16_src(is_bf16: bool) -> String {
    let pack = if is_bf16 {
        "pack_bf16x2(a, b)"
    } else {
        "packHalf2x16(vec2(a, b))"
    };
    format!(
        r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Sb {{ float src[]; }};
layout(set = 0, binding = 1) buffer Db {{ uint dst[]; }};
layout(set = 0, binding = 2) readonly buffer Pb {{ uint prm[]; }};
{GLSL_PACK16_HELPERS}
void main() {{
    uint n = prm[0];
    uint pairs = (n + 1u) >> 1u;
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint p = gl_GlobalInvocationID.x; p < pairs; p += stride) {{
        uint i0 = 2u * p;
        float a = src[i0];
        float b = (i0 + 1u < n) ? src[i0 + 1u] : 0.0;
        dst[p] = {pack};
    }}
}}
"#
    )
}

fn pack_f32_to_dtype(dev: &VulkanDevice, src: &VulkanStorage, dtype: DType) -> Result<VulkanStorage> {
    if dtype == DType::F32 {
        return Ok(src.clone_shallow_or_copy()?);
    }
    let n = src.numel;
    let dst = dev.alloc_buffer(n, dtype)?;
    if n == 0 {
        return Ok(dst);
    }
    let is_bf16 = match dtype {
        DType::F16 => false,
        DType::BF16 => true,
        other => bail!("vulkan gdn pack: unsupported dtype {other:?}"),
    };
    let key = if is_bf16 {
        "gdn_pack_f32_bf16"
    } else {
        "gdn_pack_f32_f16"
    };
    let kern = kernel(dev, key, 3, 0, || pack_f32_to_p16_src(is_bf16))?;
    let prm = dev.upload_u32_slice(&[fits(n)?])?;
    let pairs = n.div_ceil(2);
    let g = pairs.div_ceil(256).clamp(1, MAX_GROUPS) as u32;
    run(dev, &kern, &[src, &dst, &prm], &[], [g, 1, 1])?;
    Ok(dst)
}

impl VulkanStorage {
    fn clone_shallow_or_copy(&self) -> Result<Self> {
        self.device
            .upload_bytes(self.as_bytes(), self.numel, self.dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan(
        &self,
        a: &Self,
        b: &Self,
        conv_w: &Self,
        conv_b: Option<&Self>,
        conv_s: Option<&Self>,
        neg_a: &Self,
        dt_bias: &Self,
        state: Option<&Self>,
        args: &GdnScanArgs,
    ) -> Result<(Self, Self, Self)> {
        let dev = &self.device;
        let dtype = self.dtype;
        let tag = dtype_tag(dtype)?;
        let Some(cap) = dk_cap(args.dk) else {
            bail!(
                "vulkan gdn_scan: head_k_dim {} > {GDN_MAX_HEAD_K_DIM}",
                args.dk
            )
        };
        let c = 2 * args.hk * args.dk + args.hv * args.dv;
        let km1 = args.kc - 1;
        let qkv_elems = args.b * args.t * c;
        let cs_elems = args.b * c * km1;
        let out_elems = args.b * args.t * args.hv * args.dv;
        let st_elems = args.b * args.hv * args.dk * args.dv;
        fits(qkv_elems.max(out_elems).max(st_elems).max(cs_elems))?;

        let qkv_conv = dev.alloc_buffer(qkv_elems, DType::F32)?;
        let cs_f32 = dev.alloc_buffer(cs_elems.max(1), DType::F32)?;
        let out_f32 = dev.alloc_buffer(out_elems, DType::F32)?;
        let s_out = dev.alloc_buffer(st_elems, DType::F32)?;

        // Stage 1: conv1d + SiLU + conv_state update
        let k_conv = kernel(dev, &format!("gdn_conv_{tag}"), 7, 0, || gdn_conv_src(dtype))?;
        let prm1 = dev.upload_u32_slice(&[
            args.b as u32,
            args.t as u32,
            c as u32,
            args.kc as u32,
            fits(args.x[0])?,
            fits(args.x[1])?,
            fits(args.x[2])?,
            fits(args.cw_off)?,
            conv_b.is_some() as u32,
            fits(args.cb_off)?,
            conv_s.is_some() as u32,
            fits(args.cs_off)?,
        ])?;
        let cb_ref = conv_b.unwrap_or(conv_w);
        let cs_ref = conv_s.unwrap_or(self);
        let g1 = (args.b * c).div_ceil(128).clamp(1, MAX_GROUPS) as u32;
        run(
            dev,
            &k_conv,
            &[self, conv_w, cb_ref, cs_ref, &qkv_conv, &cs_f32, &prm1],
            &[],
            [g1, 1, 1],
        )?;

        // Stage 2: column-in-thread GDN scan
        let ab_dtype = if (args.flags & GDN_FLAG_AB_F32) != 0 {
            DType::F32
        } else {
            dtype
        };
        let ab_tag = dtype_tag(ab_dtype)?;
        let k_scan = kernel(
            dev,
            &format!("gdn_scan_d{cap}_{ab_tag}"),
            9,
            0,
            || gdn_scan_src(cap, ab_dtype),
        )?;
        let prm2 = dev.upload_u32_slice(&[
            args.b as u32,
            args.t as u32,
            args.hk as u32,
            args.hv as u32,
            args.dk as u32,
            args.dv as u32,
            args.flags,
            args.eps.to_bits(),
            args.scale.to_bits(),
            fits(args.a[0])?,
            fits(args.a[1])?,
            fits(args.a[2])?,
            fits(args.bb[0])?,
            fits(args.bb[1])?,
            fits(args.bb[2])?,
            fits(args.na_off)?,
            fits(args.dtb_off)?,
            state.is_some() as u32,
            fits(args.s_off)?,
        ])?;
        let st_ref = state.unwrap_or(&s_out);
        let g2 = [
            args.dv.div_ceil(64).max(1) as u32,
            args.hv as u32,
            args.b as u32,
        ];
        run(
            dev,
            &k_scan,
            &[
                &qkv_conv, a, b, neg_a, dt_bias, st_ref, &out_f32, &s_out, &prm2,
            ],
            &[],
            g2,
        )?;

        let out = if dtype == DType::F32 {
            out_f32
        } else {
            pack_f32_to_dtype(dev, &out_f32, dtype)?
        };
        let cs_out = if cs_elems == 0 {
            dev.alloc_buffer(0, dtype)?
        } else if dtype == DType::F32 {
            cs_f32
        } else {
            pack_f32_to_dtype(dev, &cs_f32, dtype)?
        };
        Ok((out, cs_out, s_out))
    }
}

/// Runs a one-time accuracy self-check of the Vulkan GPU GDN kernels against the CPU oracle,
/// cached by `(ctx_id, dk_cap, x_dtype, ab_dtype)` so every distinct shader variant is verified
/// before activation (`P3c-F04`, `P3c-F10`).
pub(crate) fn self_check(dev: &VulkanDevice) -> bool {
    self_check_for(dev, 64, DType::F32, DType::F32)
}

pub(crate) fn self_check_for(
    dev: &VulkanDevice,
    dk: usize,
    x_dtype: DType,
    ab_dtype: DType,
) -> bool {
    if shaders::exec_mode() == shaders::ExecMode::Host
        || !dev.ctx().native_ok.load(std::sync::atomic::Ordering::Relaxed)
        || (dev.ctx().device_type == ash::vk::PhysicalDeviceType::CPU && !shaders::test_force_fused())
    {
        return false;
    }
    let Some(cap) = dk_cap(dk) else {
        return false;
    };
    static CACHE: OnceLock<Mutex<HashMap<(u64, usize, DType, DType), bool>>> = OnceLock::new();
    let key = (dev.ctx().ctx_id, cap, x_dtype, ab_dtype);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ok) = cache.lock().ok().and_then(|g| g.get(&key).copied()) {
        return ok;
    }
    let ok = run_self_check(dev, cap, x_dtype, ab_dtype).unwrap_or(false);
    if let Ok(mut g) = cache.lock() {
        g.insert(key, ok);
    }
    ok
}

fn run_self_check(dev: &VulkanDevice, dk: usize, x_dtype: DType, ab_dtype: DType) -> Result<bool> {
    use crate::delta_rule::{fused_gated_delta_rule, GdnConfig};
    use crate::{Device, Tensor};

    let cfg = GdnConfig::new(2, 4, dk, 64);
    let (bsz, t, c, k) = (1, 4, cfg.conv_dim(), cfg.conv_kernel);
    let hv = cfg.num_v_heads;
    let mut rng = 0x517cc1b727220a95u64;
    let mut next = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32) * 1.2 - 0.6
    };
    let mut mk = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };

    let x_cpu = Tensor::from_vec(mk(bsz * t * c), (bsz, t, c), &Device::Cpu)?.to_dtype(x_dtype)?;
    // `gdn_scan` always receives F32 `conv_weight` / `conv_bias` (matching `gpu_gated_delta_rule`).
    let cw_cpu = Tensor::from_vec(mk(c * k), (c, k), &Device::Cpu)?;
    let cb_cpu = Tensor::from_vec(mk(c), c, &Device::Cpu)?;
    let cs_cpu = Tensor::from_vec(mk(bsz * c * (k - 1)), (bsz, c, k - 1), &Device::Cpu)?.to_dtype(x_dtype)?;
    let a_cpu = Tensor::from_vec(mk(bsz * t * hv), (bsz, t, hv), &Device::Cpu)?.to_dtype(ab_dtype)?;
    let b_cpu = Tensor::from_vec(mk(bsz * t * hv), (bsz, t, hv), &Device::Cpu)?.to_dtype(ab_dtype)?;
    let na_cpu = Tensor::from_vec(vec![-0.4f32; hv], hv, &Device::Cpu)?;
    let dtb_cpu = Tensor::from_vec(vec![0.1f32; hv], hv, &Device::Cpu)?;
    let st_cpu = Tensor::from_vec(
        mk(bsz * hv * cfg.head_k_dim * cfg.head_v_dim),
        (bsz, hv, cfg.head_k_dim, cfg.head_v_dim),
        &Device::Cpu,
    )?;

    let ref_out = fused_gated_delta_rule(
        &x_cpu,
        &cw_cpu,
        Some(&cb_cpu),
        Some(&cs_cpu),
        &a_cpu,
        &b_cpu,
        &na_cpu,
        &dtb_cpu,
        Some(&st_cpu),
        &cfg,
    )?;

    let vdev = Device::Vulkan(dev.clone());
    let mut flags = GDN_FLAG_L2NORM;
    if ab_dtype == DType::F32 && x_dtype != DType::F32 {
        flags |= GDN_FLAG_AB_F32;
    }
    let args = GdnScanArgs {
        b: bsz,
        t,
        hk: cfg.num_k_heads,
        hv,
        dk: cfg.head_k_dim,
        dv: cfg.head_v_dim,
        kc: k,
        flags,
        eps: cfg.l2norm_eps,
        scale: (1.0 / (cfg.head_k_dim as f64).sqrt()) as f32,
        x: [0, t * c, c],
        a: [0, t * hv, hv],
        bb: [0, t * hv, hv],
        cw_off: 0,
        cb_off: 0,
        cs_off: 0,
        na_off: 0,
        dtb_off: 0,
        s_off: 0,
    };
    let to_vk = |t: &Tensor| -> Result<VulkanStorage> {
        let tv = t.to_device(&vdev)?;
        let (st, _) = tv.storage_and_layout();
        match &*st {
            crate::Storage::Vulkan(vs) => vs.try_clone(tv.layout()),
            _ => unreachable!(),
        }
    };
    let (xv, cwv, cbv, csv, av, bv, nav, dtbv, stv) = (
        to_vk(&x_cpu)?,
        to_vk(&cw_cpu)?,
        to_vk(&cb_cpu)?,
        to_vk(&cs_cpu)?,
        to_vk(&a_cpu)?,
        to_vk(&b_cpu)?,
        to_vk(&na_cpu)?,
        to_vk(&dtb_cpu)?,
        to_vk(&st_cpu)?,
    );
    let (ov, csv_o, stv_o) = xv.gdn_scan(
        &av,
        &bv,
        &cwv,
        Some(&cbv),
        Some(&csv),
        &nav,
        &dtbv,
        Some(&stv),
        &args,
    )?;
    let tol = match x_dtype {
        DType::F32 => 1e-3,
        DType::F16 => 1e-2,
        DType::BF16 => 3e-2,
        _ => 1e-2,
    };
    let check_close = |gpu: &VulkanStorage, cpu: &Tensor| -> Result<bool> {
        let g_cpu = gpu.to_cpu_storage()?;
        let g_t = crate::Tensor::from_storage(
            crate::Storage::Cpu(g_cpu),
            cpu.shape(),
            crate::op::BackpropOp::none(),
            false,
        )
        .to_dtype(DType::F32)?;
        let gs = g_t.flatten_all()?.to_vec1::<f32>()?;
        let cs = cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if gs.len() != cs.len() {
            return Ok(false);
        }
        let max_err = gs
            .iter()
            .zip(cs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        Ok(max_err.is_finite() && max_err < tol)
    };
    Ok(check_close(&ov, &ref_out.out)?
        && check_close(&csv_o, &ref_out.conv_state)?
        && check_close(&stv_o, &ref_out.recurrent_state)?)
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
    fn delta_rule_shaders_validate() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            validate(&gdn_conv_src(dt));
            for cap in [64, 128, 256] {
                validate(&gdn_scan_src(cap, dt));
            }
        }
        validate(&pack_f32_to_p16_src(false));
        validate(&pack_f32_to_p16_src(true));
    }
}
