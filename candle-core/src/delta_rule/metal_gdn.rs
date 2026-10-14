//! Metal Gated DeltaNet: the causal conv, SiLU, q / k L2 norms and the q scale run as tensor
//! ops (parallel over the tokens), then one native kernel runs the sequential gated delta rule
//! scan with the gates computed in place. Same semantics as the CPU kernel (the test oracle):
//!
//! ```text
//! g = neg_a * softplus(a + dt_bias)   (torch threshold 20),  beta = sigmoid(b)
//! S1 = exp(g) S;  delta = beta (v - S1^T k);  S' = S1 + k delta^T;  o = S'^T q
//! ```
//!
//! Kernel layout: one thread per value column `j` of a (value head, batch), holding the
//! `[dk]` column of the recurrent state; the threadgroup stages q / k of the token in
//! threadgroup memory. A device enables the kernel only after [`self_check`] matched the CPU
//! kernel; otherwise callers keep the tensor path.

use super::{cpu, GdnConfig, GdnHeadOrder, GdnInputs, GdnOutput};
use crate::op::BackpropOp;
use crate::{bail, DType, Device, Layout, MetalDevice, MetalError, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GdnScanParams {
    uint t;
    uint hk;
    uint hv;
    uint dk;
    uint dv;
    uint tiled;
};

kernel void gdn_scan_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *a [[buffer(3)]],
    device const float *bb [[buffer(4)]],
    device const float *neg_a [[buffer(5)]],
    device const float *dt_bias [[buffer(6)]],
    device const float *s_in [[buffer(7)]],
    device float *out [[buffer(8)]],
    device float *s_out [[buffer(9)]],
    constant GdnScanParams &p [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tl [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
{
    threadgroup float qs[GDN_DK_MAX];
    threadgroup float ks[GDN_DK_MAX];
    float s[GDN_DK_MAX];
    const uint j = tg.x * tpg.x + tl.x;
    const uint h = tg.y;
    const uint b = tg.z;
    const uint kh = p.tiled != 0u ? h % p.hk : h / (p.hv / p.hk);
    const bool col = j < p.dv;
    const ulong sbase = (ulong(b) * p.hv + h) * p.dk * p.dv;
    for (uint i = 0u; i < p.dk; ++i) {
        s[i] = col ? s_in[sbase + ulong(i) * p.dv + j] : 0.0f;
    }
    for (uint t = 0u; t < p.t; ++t) {
        const ulong qk = ((ulong(b) * p.t + t) * p.hk + kh) * p.dk;
        for (uint i = tl.x; i < p.dk; i += tpg.x) {
            qs[i] = q[qk + i];
            ks[i] = k[qk + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong gi = (ulong(b) * p.t + t) * p.hv + h;
        const float x = a[gi] + dt_bias[h];
        const float sp = x > 20.0f ? x : log(1.0f + exp(x));
        const float eg = exp(neg_a[h] * sp);
        const float beta = 1.0f / (1.0f + exp(-bb[gi]));
        if (col) {
            float kv = 0.0f;
            float sq = 0.0f;
            float kq = 0.0f;
            for (uint i = 0u; i < p.dk; ++i) {
                kv += s[i] * ks[i];
                sq += s[i] * qs[i];
                kq += ks[i] * qs[i];
            }
            const ulong vi = gi * p.dv + j;
            const float delta = beta * (v[vi] - eg * kv);
            out[vi] = eg * sq + delta * kq;
            for (uint i = 0u; i < p.dk; ++i) {
                s[i] = eg * s[i] + ks[i] * delta;
            }
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

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnScanParams {
    t: u32,
    hk: u32,
    hv: u32,
    dk: u32,
    dv: u32,
    tiled: u32,
}

/// The scan pipeline for head_k_dim <= `dk_max` (the state column lives in thread memory).
fn pipeline(m: &MetalDevice, dk_max: usize) -> Result<ComputePipeline> {
    static PIPES: OnceLock<Mutex<HashMap<(crate::metal_backend::DeviceId, usize), ComputePipeline>>> =
        OnceLock::new();
    let pipes = PIPES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (m.id(), dk_max);
    if let Some(p) = pipes.lock().ok().and_then(|g| g.get(&key).cloned()) {
        return Ok(p);
    }
    let src = SRC.replace("GDN_DK_MAX", &dk_max.to_string());
    let lib = m
        .device()
        .new_library_with_source(&src, None)
        .map_err(MetalError::from)?;
    let func = lib.get_function("gdn_scan_f32", None).map_err(MetalError::from)?;
    let pipe = m
        .device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(MetalError::from)?;
    if let Ok(mut g) = pipes.lock() {
        g.insert(key, pipe.clone());
    }
    Ok(pipe)
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
    // Causal depthwise conv over [conv_state ; tokens]: out[t, c] = bias[c] + sum_j w[c, j] x[t + j, c].
    let xf = x.to_dtype(DType::F32)?;
    let xin = if km1 == 0 {
        xf
    } else {
        let cs = match inp.conv_state.filter(|s| s.elem_count() > 0) {
            Some(cs) => cs.to_dtype(DType::F32)?.transpose(1, 2)?,
            None => Tensor::zeros((bsz, km1, c), DType::F32, &dev)?,
        };
        Tensor::cat(&[&cs, &xf], 1)?
    };
    let w = inp.conv_weight.to_dtype(DType::F32)?;
    let mut conv = match inp.conv_bias {
        Some(cb) => cb
            .to_dtype(DType::F32)?
            .reshape((1, 1, c))?
            .broadcast_as((bsz, t, c))?
            .contiguous()?,
        None => Tensor::zeros((bsz, t, c), DType::F32, &dev)?,
    };
    for j in 0..kw {
        let tap = w.narrow(1, j, 1)?.reshape((1, 1, c))?;
        conv = (conv + xin.narrow(1, j, t)?.broadcast_mul(&tap)?)?;
    }
    let conv = conv.silu()?;
    let conv_state = if km1 == 0 {
        Tensor::zeros((bsz, c, 0), dtype, &dev)?
    } else {
        xin.narrow(1, t, km1)?
            .transpose(1, 2)?
            .to_dtype(dtype)?
            .contiguous()?
    };
    let key_dim = hk * dk;
    let mut q = conv.narrow(2, 0, key_dim)?.reshape((bsz, t, hk, dk))?;
    let mut k = conv.narrow(2, key_dim, key_dim)?.reshape((bsz, t, hk, dk))?;
    let v = conv.narrow(2, 2 * key_dim, hv * dv)?.contiguous()?;
    if cfg.use_qk_l2norm {
        let eps = cfg.l2norm_eps as f64;
        let l2 = |u: &Tensor| -> Result<Tensor> {
            let inv = (u.sqr()?.sum_keepdim(3)? + eps)?.sqrt()?.recip()?;
            u.broadcast_mul(&inv)
        };
        q = l2(&q)?;
        k = l2(&k)?;
    }
    let q = (q * (1.0 / (dk as f64).sqrt()))?.contiguous()?;
    let k = k.contiguous()?;
    let a = f32c(inp.a)?;
    let b = f32c(inp.b)?;
    let na = f32c(&inp.neg_a.flatten_all()?)?;
    let dtb = f32c(&inp.dt_bias.flatten_all()?)?;
    let s_in = match inp.recurrent_state {
        Some(s) => f32c(s)?,
        None => Tensor::zeros((bsz, hv, dk, dv), DType::F32, &dev)?,
    };

    let pipe = pipeline(m, dk_max)?;
    let n_out = bsz * t * hv * dv;
    let n_state = bsz * hv * dk * dv;
    let out_buf = m.new_buffer(n_out, DType::F32, "gdn_out")?;
    let s_buf = m.new_buffer(n_state, DType::F32, "gdn_state")?;
    let params = GdnScanParams {
        t: t as u32,
        hk: hk as u32,
        hv: hv as u32,
        dk: dk as u32,
        dv: dv as u32,
        tiled: (cfg.v_head_order == GdnHeadOrder::Tiled) as u32,
    };
    let tpg = dv.min(64).min(pipe.max_total_threads_per_threadgroup()).max(1);
    {
        let ins = [&q, &k, &v, &a, &b, &na, &dtb, &s_in];
        let guards: Vec<_> = ins.iter().map(|t| t.storage_and_layout()).collect();
        let enc = m.command_encoder()?;
        let e: &ComputeCommandEncoder = enc.as_ref();
        e.set_compute_pipeline_state(&pipe);
        for (i, (s, l)) in guards.iter().enumerate() {
            let (buf, off) = metal_buffer(s, l)?;
            e.set_input_buffer(i, Some(buf), off);
        }
        e.set_output_buffer(8, Some(&out_buf), 0);
        e.set_output_buffer(9, Some(&s_buf), 0);
        e.set_bytes(10, &params);
        e.dispatch_thread_groups(
            MTLSize {
                width: dv.div_ceil(tpg),
                height: hv,
                depth: bsz,
            },
            MTLSize {
                width: tpg,
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
