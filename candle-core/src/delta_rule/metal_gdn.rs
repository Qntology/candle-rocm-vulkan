//! Metal Gated DeltaNet: two native kernels, all math f32 with precise exp / log / divide / sqrt
//! (the library is built without fast math; candle's own Metal kernels use fast math, whose `exp`
//! error compounds through the per-token decay over a sequence):
//!
//! 1. `gdn_conv_f32`: causal depthwise conv over `[conv_state ; tokens]` (+ bias) and SiLU, one
//!    thread per (channel, token, batch); the token-0 threads also write the new conv state (the
//!    last K-1 entries), so a decode step is two dispatches and no tensor ops.
//! 2. `gdn_scan_f32`: per (value head, batch), one thread per value column holding the `[dk]`
//!    column of the recurrent state; the threadgroup stages q / k of the token, reduces their
//!    squared norms (L2 norm, q scale), and every thread runs the CPU kernel's update order:
//!
//! ```text
//! g = neg_a * softplus(a + dt_bias) (torch threshold 20), beta = sigmoid(b)
//! S *= exp(g);  m = S^T k;  d = beta (v - m);  S += k d^T;  o = S^T q
//! ```
//!
//! A device enables the kernels only after [`self_check`] matched the CPU kernel (the test
//! oracle); otherwise callers keep the tensor path.

use super::{cpu, GdnConfig, GdnHeadOrder, GdnInputs, GdnOutput};
use crate::op::BackpropOp;
use crate::{bail, DType, Device, Layout, MetalDevice, MetalError, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLSize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GdnConvParams {
    uint t;
    uint c;
    uint kw;
    uint has_cs;
    uint has_bias;
};

struct GdnScanParams {
    uint t;
    uint c;
    uint hk;
    uint hv;
    uint dk;
    uint dv;
    uint tiled;
    uint l2;
    float eps;
    float scale;
};

inline float gdn_sigmoid(float x) {
    return precise::divide(1.0f, 1.0f + precise::exp(-x));
}

// log(1 + y) without the cancellation of log(1 + y) for small y (Goldberg).
inline float gdn_log1p(float y) {
    const float u = 1.0f + y;
    return u == 1.0f ? y : precise::log(u) * precise::divide(y, u - 1.0f);
}

inline float gdn_softplus(float x) {
    return x > 20.0f ? x : gdn_log1p(precise::exp(x));
}

kernel void gdn_conv_f32(
    device const float *x [[buffer(0)]],
    device const float *cs [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    device float *y [[buffer(4)]],
    device float *cs_out [[buffer(6)]],
    constant GdnConvParams &p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tl [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    const uint c = tg.x * tpg.x + tl.x;
    const uint t = tg.y;
    const uint b = tg.z;
    if (c >= p.c) {
        return;
    }
    const uint km1 = p.kw - 1u;
    float acc = p.has_bias != 0u ? bias[c] : 0.0f;
    for (uint j = 0u; j < p.kw; ++j) {
        const uint i = t + j;
        float xv;
        if (i < km1) {
            xv = p.has_cs != 0u ? cs[(ulong(b) * p.c + c) * km1 + i] : 0.0f;
        } else {
            xv = x[(ulong(b) * p.t + (i - km1)) * p.c + c];
        }
        acc += w[ulong(c) * p.kw + j] * xv;
    }
    y[(ulong(b) * p.t + t) * p.c + c] = acc * gdn_sigmoid(acc);
    if (t == 0u) {
        // New conv state: entries T .. T + K-2 of [conv_state ; tokens] (copies, no arithmetic).
        for (uint m = 0u; m < km1; ++m) {
            const uint i = p.t + m;
            float xv;
            if (i < km1) {
                xv = p.has_cs != 0u ? cs[(ulong(b) * p.c + c) * km1 + i] : 0.0f;
            } else {
                xv = x[(ulong(b) * p.t + (i - km1)) * p.c + c];
            }
            cs_out[(ulong(b) * p.c + c) * km1 + m] = xv;
        }
    }
}

kernel void gdn_scan_f32(
    device const float *y [[buffer(0)]],
    device const float *a [[buffer(1)]],
    device const float *bb [[buffer(2)]],
    device const float *neg_a [[buffer(3)]],
    device const float *dt_bias [[buffer(4)]],
    device const float *s_in [[buffer(5)]],
    device float *out [[buffer(6)]],
    device float *s_out [[buffer(7)]],
    constant GdnScanParams &p [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tl [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    threadgroup float qs[GDN_DK_MAX];
    threadgroup float ks[GDN_DK_MAX];
    threadgroup float red[2 * GDN_TPG_MAX];
    float s[GDN_DK_MAX];
    const uint j = tg.x * tpg.x + tl.x;
    const uint h = tg.y;
    const uint b = tg.z;
    const uint kh = p.tiled != 0u ? h % p.hk : h / (p.hv / p.hk);
    const bool col = j < p.dv;
    const uint key_dim = p.hk * p.dk;
    const ulong sbase = (ulong(b) * p.hv + h) * p.dk * p.dv;
    for (uint i = 0u; i < p.dk; ++i) {
        s[i] = col ? s_in[sbase + ulong(i) * p.dv + j] : 0.0f;
    }
    for (uint t = 0u; t < p.t; ++t) {
        const ulong row = (ulong(b) * p.t + t) * p.c;
        float pq = 0.0f;
        float pk = 0.0f;
        for (uint i = tl.x; i < p.dk; i += tpg.x) {
            const float qv = y[row + kh * p.dk + i];
            const float kv = y[row + key_dim + kh * p.dk + i];
            qs[i] = qv;
            ks[i] = kv;
            pq += qv * qv;
            pk += kv * kv;
        }
        red[tl.x] = pq;
        red[GDN_TPG_MAX + tl.x] = pk;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float iq = 1.0f;
        float ik = 1.0f;
        if (p.l2 != 0u) {
            float sq = 0.0f;
            float sk = 0.0f;
            for (uint u = 0u; u < tpg.x; ++u) {
                sq += red[u];
                sk += red[GDN_TPG_MAX + u];
            }
            iq = precise::divide(1.0f, precise::sqrt(sq + p.eps));
            ik = precise::divide(1.0f, precise::sqrt(sk + p.eps));
        }
        const ulong gi = (ulong(b) * p.t + t) * p.hv + h;
        const float g = neg_a[h] * gdn_softplus(a[gi] + dt_bias[h]);
        const float eg = precise::exp(g);
        const float beta = gdn_sigmoid(bb[gi]);
        if (col) {
            float m = 0.0f;
            for (uint i = 0u; i < p.dk; ++i) {
                s[i] *= eg;
                m += s[i] * (ks[i] * ik);
            }
            const float d = (y[row + 2u * key_dim + h * p.dv + j] - m) * beta;
            float o = 0.0f;
            for (uint i = 0u; i < p.dk; ++i) {
                s[i] += (ks[i] * ik) * d;
                o += s[i] * ((qs[i] * iq) * p.scale);
            }
            out[gi * p.dv + j] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (col) {
        for (uint i = 0u; i < p.dk; ++i) {
            s_out[sbase + ulong(i) * p.dv + j] = s[i];
        }
    }
}
"#;

/// Threads per threadgroup of the scan (value columns per group); also sizes `red`.
const SCAN_TPG: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnConvParams {
    t: u32,
    c: u32,
    kw: u32,
    has_cs: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnScanParams {
    t: u32,
    c: u32,
    hk: u32,
    hv: u32,
    dk: u32,
    dv: u32,
    tiled: u32,
    l2: u32,
    eps: f32,
    scale: f32,
}

/// The (conv, scan) pipelines for head_k_dim <= `dk_max` (the scan keeps the state column in
/// thread memory). Built without fast math.
fn pipelines(m: &MetalDevice, dk_max: usize) -> Result<(ComputePipeline, ComputePipeline)> {
    type Key = (crate::metal_backend::DeviceId, usize);
    static PIPES: OnceLock<Mutex<HashMap<Key, (ComputePipeline, ComputePipeline)>>> = OnceLock::new();
    let pipes = PIPES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (m.id(), dk_max);
    if let Some(p) = pipes.lock().ok().and_then(|g| g.get(&key).cloned()) {
        return Ok(p);
    }
    let src = SRC
        .replace("GDN_DK_MAX", &dk_max.to_string())
        .replace("GDN_TPG_MAX", &SCAN_TPG.to_string());
    let opts = MTLCompileOptions::new();
    #[allow(deprecated)]
    opts.setFastMathEnabled(false);
    let lib = m
        .device()
        .new_library_with_source(&src, Some(&*opts))
        .map_err(MetalError::from)?;
    let make = |name: &str| -> Result<ComputePipeline> {
        let func = lib.get_function(name, None).map_err(MetalError::from)?;
        Ok(m.device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?)
    };
    let p = (make("gdn_conv_f32")?, make("gdn_scan_f32")?);
    if let Ok(mut g) = pipes.lock() {
        g.insert(key, p.clone());
    }
    Ok(p)
}

fn metal_buffer<'a>(s: &'a Storage, l: &Layout) -> Result<(&'a Buffer, usize)> {
    match s {
        Storage::Metal(m) => Ok((m.buffer(), l.start_offset() * DType::F32.size_in_bytes())),
        _ => bail!("gated delta rule: inputs on different backends"),
    }
}

fn f32c(t: &Tensor) -> Result<Tensor> {
    t.to_dtype(DType::F32)?.contiguous()
}

pub(super) fn gated_delta_rule(inp: &GdnInputs, cfg: &GdnConfig) -> Result<GdnOutput> {
    let x = inp.mixed_qkv;
    let dtype = x.dtype();
    let dev = x.device().clone();
    let Device::Metal(m) = &dev else {
        bail!("gated delta rule: metal scan on {:?}", dev.location())
    };
    let (bsz, t, c) = x.dims3()?;
    let (hk, hv, dk, dv, kw) = (
        cfg.num_k_heads,
        cfg.num_v_heads,
        cfg.head_k_dim,
        cfg.head_v_dim,
        cfg.conv_kernel,
    );
    let dk_max = match dk {
        0..=64 => 64,
        65..=128 => 128,
        129..=256 => 256,
        _ => bail!("gated delta rule: head_k_dim {dk} > 256 has no Metal kernel"),
    };
    let km1 = kw - 1;
    let xf = f32c(x)?;
    let cs = inp
        .conv_state
        .filter(|s| s.elem_count() > 0)
        .map(f32c)
        .transpose()?;
    // New conv state: the last K-1 entries of [conv_state ; tokens] (copies only, no arithmetic).
    let w = f32c(inp.conv_weight)?;
    let bias = inp.conv_bias.map(f32c).transpose()?;
    let a = f32c(inp.a)?;
    let b = f32c(inp.b)?;
    let na = f32c(&inp.neg_a.flatten_all()?)?;
    let dtb = f32c(&inp.dt_bias.flatten_all()?)?;
    let s_in = match inp.recurrent_state {
        Some(s) => f32c(s)?,
        None => Tensor::zeros((bsz, hv, dk, dv), DType::F32, &dev)?,
    };

    let (conv_pipe, scan_pipe) = pipelines(m, dk_max)?;
    let n_conv = bsz * t * c;
    let n_out = bsz * t * hv * dv;
    let n_state = bsz * hv * dk * dv;
    let conv_buf = m.new_buffer(n_conv, DType::F32, "gdn_conv")?;
    let n_cs = bsz * c * km1;
    // (width-1 conv: an empty state; the kernel never writes it, the binding still needs a buffer)
    let cs_buf = m.new_buffer(n_cs.max(1), DType::F32, "gdn_conv_state")?;
    let out_buf = m.new_buffer(n_out, DType::F32, "gdn_out")?;
    let s_buf = m.new_buffer(n_state, DType::F32, "gdn_state")?;
    let conv_params = GdnConvParams {
        t: t as u32,
        c: c as u32,
        kw: kw as u32,
        has_cs: cs.is_some() as u32,
        has_bias: bias.is_some() as u32,
    };
    let scan_params = GdnScanParams {
        t: t as u32,
        c: c as u32,
        hk: hk as u32,
        hv: hv as u32,
        dk: dk as u32,
        dv: dv as u32,
        tiled: (cfg.v_head_order == GdnHeadOrder::Tiled) as u32,
        l2: cfg.use_qk_l2norm as u32,
        eps: cfg.l2norm_eps,
        scale: (1.0 / (dk as f64).sqrt()) as f32,
    };
    let conv_tpg = c.min(64).min(conv_pipe.max_total_threads_per_threadgroup()).max(1);
    let scan_tpg = SCAN_TPG.min(scan_pipe.max_total_threads_per_threadgroup()).max(1);
    {
        // Absent conv state / bias: bind x in their place (never read).
        let ins = [Some(&xf), cs.as_ref(), Some(&w), bias.as_ref(), Some(&a), Some(&b), Some(&na), Some(&dtb), Some(&s_in)];
        let guards: Vec<_> = ins
            .iter()
            .map(|t| t.unwrap_or(&xf).storage_and_layout())
            .collect();
        let bufs = guards
            .iter()
            .map(|(s, l)| metal_buffer(s, l))
            .collect::<Result<Vec<_>>>()?;
        let enc = m.command_encoder()?;
        let e: &ComputeCommandEncoder = enc.as_ref();
        e.set_compute_pipeline_state(&conv_pipe);
        for (i, (buf, off)) in bufs[..4].iter().enumerate() {
            e.set_input_buffer(i, Some(buf), *off);
        }
        e.set_output_buffer(4, Some(&conv_buf), 0);
        e.set_output_buffer(6, Some(&cs_buf), 0);
        e.set_bytes(5, &conv_params);
        e.dispatch_thread_groups(
            MTLSize {
                width: c.div_ceil(conv_tpg),
                height: t,
                depth: bsz,
            },
            MTLSize {
                width: conv_tpg,
                height: 1,
                depth: 1,
            },
        );
        e.set_compute_pipeline_state(&scan_pipe);
        e.set_input_buffer(0, Some(&conv_buf), 0);
        for (i, (buf, off)) in bufs[4..].iter().enumerate() {
            e.set_input_buffer(1 + i, Some(buf), *off);
        }
        e.set_output_buffer(6, Some(&out_buf), 0);
        e.set_output_buffer(7, Some(&s_buf), 0);
        e.set_bytes(8, &scan_params);
        e.dispatch_thread_groups(
            MTLSize {
                width: dv.div_ceil(scan_tpg),
                height: hv,
                depth: bsz,
            },
            MTLSize {
                width: scan_tpg,
                height: 1,
                depth: 1,
            },
        );
    }
    let mk = |buf, n, shape: (usize, usize, usize, usize)| {
        Tensor::from_storage(
            Storage::Metal(MetalStorage::new(buf, m.clone(), n, DType::F32)),
            shape,
            BackpropOp::none(),
            false,
        )
    };
    let conv_state = if km1 == 0 {
        Tensor::zeros((bsz, c, 0), dtype, &dev)?
    } else {
        Tensor::from_storage(
            Storage::Metal(MetalStorage::new(cs_buf, m.clone(), n_cs, DType::F32)),
            (bsz, c, km1),
            BackpropOp::none(),
            false,
        )
        .to_dtype(dtype)?
    };
    Ok(GdnOutput {
        out: mk(out_buf, n_out, (bsz, t, hv, dv)).to_dtype(dtype)?,
        conv_state,
        recurrent_state: mk(s_buf, n_state, (bsz, hv, dk, dv)),
    })
}

/// Runs a small problem (conv state and recurrent state given, 7 tokens, GQA 2 / 4 heads) through
/// the Metal kernel and the CPU kernel once per device; the Metal path is used only when the
/// outputs and both new states agree (max relative difference <= 1e-4).
pub(super) fn self_check(m: &MetalDevice) -> bool {
    static DONE: OnceLock<Mutex<HashMap<crate::metal_backend::DeviceId, bool>>> = OnceLock::new();
    let done = DONE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ok) = done.lock().ok().and_then(|g| g.get(&m.id()).copied()) {
        return ok;
    }
    let res = check(m);
    let ok = match &res {
        Ok(rel) if *rel <= 1e-4 => {
            eprintln!("candle gated delta rule: Metal scan kernel matches the CPU kernel (max rel {rel:.1e}), enabled");
            true
        }
        Ok(rel) => {
            eprintln!("candle gated delta rule: Metal scan kernel differs from the CPU kernel (max rel {rel:.2e}), using the tensor path");
            false
        }
        Err(e) => {
            eprintln!("candle gated delta rule: Metal scan kernel unavailable ({e}), using the tensor path");
            false
        }
    };
    if let Ok(mut g) = done.lock() {
        g.insert(m.id(), ok);
    }
    ok
}

fn check(m: &MetalDevice) -> Result<f32> {
    let cfg = GdnConfig::new(2, 4, 32, 32);
    let (bsz, t, c, kw) = (1usize, 7usize, cfg.conv_dim(), cfg.conv_kernel);
    let det = |n: usize, seed: u64, scale: f32| -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let cpu_t = |v: Vec<f32>, shape: &[usize]| Tensor::from_vec(v, shape, &Device::Cpu);
    let hv = cfg.num_v_heads;
    let host = [
        cpu_t(det(bsz * t * c, 1, 1.0), &[bsz, t, c])?,
        cpu_t(det(c * kw, 2, 0.5), &[c, kw])?,
        cpu_t(det(c, 3, 0.1), &[c])?,
        cpu_t(det(bsz * c * (kw - 1), 4, 1.0), &[bsz, c, kw - 1])?,
        cpu_t(det(bsz * t * hv, 5, 1.0), &[bsz, t, hv])?,
        cpu_t(det(bsz * t * hv, 6, 1.0), &[bsz, t, hv])?,
        cpu_t(det(hv, 7, 0.5).into_iter().map(|x| -(x.abs() + 0.1)).collect(), &[hv])?,
        cpu_t(det(hv, 8, 0.5), &[hv])?,
        cpu_t(det(bsz * hv * 32 * 32, 9, 0.2), &[bsz, hv, 32, 32])?,
    ];
    let dev = Device::Metal(m.clone());
    let gpu: Vec<Tensor> = host.iter().map(|x| x.to_device(&dev)).collect::<Result<_>>()?;
    let inputs = |v: &[Tensor]| -> [Tensor; 9] { std::array::from_fn(|i| v[i].clone()) };
    let run = |v: [Tensor; 9], metal: bool| -> Result<GdnOutput> {
        let inp = GdnInputs {
            mixed_qkv: &v[0],
            conv_weight: &v[1],
            conv_bias: Some(&v[2]),
            conv_state: Some(&v[3]),
            a: &v[4],
            b: &v[5],
            neg_a: &v[6],
            dt_bias: &v[7],
            recurrent_state: Some(&v[8]),
        };
        if metal {
            gated_delta_rule(&inp, &cfg)
        } else {
            cpu::gated_delta_rule(&inp, &cfg)
        }
    };
    let want = run(inputs(&host), false)?;
    let got = run(inputs(&gpu), true)?;
    let rel = |g: &Tensor, w: &Tensor| -> Result<f32> {
        let g = g.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let w = w.to_dtype(DType::F32)?;
        let d = (&g - &w)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let s = w.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        Ok(d / s)
    };
    Ok(rel(&got.out, &want.out)?
        .max(rel(&got.conv_state, &want.conv_state)?)
        .max(rel(&got.recurrent_state, &want.recurrent_state)?))
}
