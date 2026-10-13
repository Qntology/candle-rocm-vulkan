//! Fused Gated DeltaNet (`candle_core::delta_rule`): CPU implementation and GPU kernels
//! against a naive f64 reference, decode chaining, strided inputs, the gated RMS norm, and
//! (with `CANDLE_GDN_FIXTURES=<dir>`) the transformers fixtures of `gen_gdn_ref.py`.
//!
//! The CUDA / ROCm checks run when a GPU device can be created; `host_emulated_kernels` runs the
//! kernel source on the host emulator when a C++20 compiler is available. `bench_gdn_qwen35`
//! (ignored) times the kernel for the Qwen3.5-4B / 2B geometries: `--ignored --nocapture`.

use candle_core::delta_rule::{
    fused_gated_delta_rule, fused_gated_rms_norm, fused_gdn_supported, GdnConfig, GdnHeadOrder,
    GdnOutput,
};
use candle_core::{DType, Device, Result, Tensor};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    fn normals(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|_| (self.normal() * scale) as f32).collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct Case {
    b: usize,
    t: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    k: usize,
    state: bool,
    bias: bool,
    order: GdnHeadOrder,
    l2: bool,
    x_scale: f64,
}

impl Case {
    #[allow(clippy::too_many_arguments)]
    fn new(b: usize, t: usize, hk: usize, hv: usize, dk: usize, dv: usize, k: usize) -> Self {
        Self {
            b,
            t,
            hk,
            hv,
            dk,
            dv,
            k,
            state: true,
            bias: false,
            order: GdnHeadOrder::Grouped,
            l2: true,
            x_scale: 0.7,
        }
    }
    fn cfg(&self) -> GdnConfig {
        let mut c = GdnConfig::new(self.hk, self.hv, self.dk, self.dv);
        c.conv_kernel = self.k;
        c.v_head_order = self.order;
        c.use_qk_l2norm = self.l2;
        c
    }
    fn c(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
}

/// Raw inputs (f32 values).
#[derive(Clone)]
struct Data {
    mixed: Vec<f32>,
    conv_w: Vec<f32>,
    conv_b: Option<Vec<f32>>,
    conv_state: Option<Vec<f32>>,
    a: Vec<f32>,
    b: Vec<f32>,
    neg_a: Vec<f32>,
    dt_bias: Vec<f32>,
    state: Option<Vec<f32>>,
}

fn gen(cs: &Case, seed: u64) -> Data {
    let mut r = Rng(seed);
    let c = cs.c();
    Data {
        mixed: r.normals(cs.b * cs.t * c, cs.x_scale),
        conv_w: r.normals(c * cs.k, 0.4),
        conv_b: cs.bias.then(|| r.normals(c, 0.1)),
        conv_state: cs.state.then(|| r.normals(cs.b * c * (cs.k - 1), cs.x_scale)),
        a: r.normals(cs.b * cs.t * cs.hv, 1.0),
        b: r.normals(cs.b * cs.t * cs.hv, 1.0),
        neg_a: (0..cs.hv)
            .map(|_| -((r.uniform() * 2.0 - 1.0).exp()) as f32)
            .collect(),
        dt_bias: r.normals(cs.hv, 0.5),
        state: cs.state.then(|| r.normals(cs.b * cs.hv * cs.dk * cs.dv, 0.1)),
    }
}

struct Ref {
    out: Vec<f64>,
    conv_state: Vec<f64>,
    state: Vec<f64>,
}

/// Naive f64 Gated DeltaNet (transformers `torch_recurrent_gated_delta_rule` order).
fn reference(cs: &Case, d: &Data) -> Ref {
    let (bsz, t, hk, hv, dk, dv, k) = (cs.b, cs.t, cs.hk, cs.hv, cs.dk, cs.dv, cs.k);
    let c = cs.c();
    let km1 = k - 1;
    let key_dim = hk * dk;
    let xe = |bi: usize, ch: usize, i: usize| -> f64 {
        if i < km1 {
            d.conv_state
                .as_ref()
                .map(|s| s[(bi * c + ch) * km1 + i] as f64)
                .unwrap_or(0.0)
        } else {
            d.mixed[(bi * t + i - km1) * c + ch] as f64
        }
    };
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut out = vec![0f64; bsz * t * hv * dv];
    let mut state = vec![0f64; bsz * hv * dk * dv];
    let mut conv_state = vec![0f64; bsz * c * km1];
    for bi in 0..bsz {
        for ch in 0..c {
            for m in 0..km1 {
                conv_state[(bi * c + ch) * km1 + m] = xe(bi, ch, t + m);
            }
        }
        // conv + silu of every channel and token
        let mut y = vec![0f64; t * c];
        for tt in 0..t {
            for ch in 0..c {
                let mut acc = d.conv_b.as_ref().map(|b| b[ch] as f64).unwrap_or(0.0);
                for j in 0..k {
                    acc += d.conv_w[ch * k + j] as f64 * xe(bi, ch, tt + j);
                }
                y[tt * c + ch] = acc * sig(acc);
            }
        }
        for h in 0..hv {
            let kh = match cs.order {
                GdnHeadOrder::Grouped => h / (hv / hk),
                GdnHeadOrder::Tiled => h % hk,
            };
            let base = (bi * hv + h) * dk * dv;
            let s = &mut state[base..base + dk * dv];
            if let Some(s0) = &d.state {
                for (x, y) in s.iter_mut().zip(&s0[base..base + dk * dv]) {
                    *x = *y as f64;
                }
            }
            for tt in 0..t {
                let mut q: Vec<f64> = (0..dk).map(|i| y[tt * c + kh * dk + i]).collect();
                let mut kk: Vec<f64> = (0..dk).map(|i| y[tt * c + key_dim + kh * dk + i]).collect();
                let v: Vec<f64> = (0..dv)
                    .map(|j| y[tt * c + 2 * key_dim + h * dv + j])
                    .collect();
                if cs.l2 {
                    for u in [&mut q, &mut kk] {
                        let inv = 1.0 / (u.iter().map(|e| e * e).sum::<f64>() + 1e-6).sqrt();
                        u.iter_mut().for_each(|e| *e *= inv);
                    }
                }
                q.iter_mut().for_each(|e| *e /= (dk as f64).sqrt());
                let ab = (bi * t + tt) * hv + h;
                let x = d.a[ab] as f64 + d.dt_bias[h] as f64;
                let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
                let eg = (d.neg_a[h] as f64 * sp).exp();
                let beta = sig(d.b[ab] as f64);
                s.iter_mut().for_each(|e| *e *= eg);
                for j in 0..dv {
                    let kv: f64 = (0..dk).map(|i| s[i * dv + j] * kk[i]).sum();
                    let delta = (v[j] - kv) * beta;
                    for i in 0..dk {
                        s[i * dv + j] += kk[i] * delta;
                    }
                }
                for j in 0..dv {
                    out[((bi * t + tt) * hv + h) * dv + j] =
                        (0..dk).map(|i| s[i * dv + j] * q[i]).sum();
                }
            }
        }
    }
    Ref {
        out,
        conv_state,
        state,
    }
}

fn tensor(v: &[f32], shape: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(v.to_vec(), shape, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(dev)
}

/// Runs the fused op with `mixed_qkv`, `a`, `b` and the conv state in `dtype`.
fn run(cs: &Case, d: &Data, dev: &Device, dtype: DType) -> Result<GdnOutput> {
    let c = cs.c();
    let mixed = tensor(&d.mixed, &[cs.b, cs.t, c], dtype, dev)?;
    let w = tensor(&d.conv_w, &[c, cs.k], DType::F32, dev)?;
    let cb = d
        .conv_b
        .as_ref()
        .map(|v| tensor(v, &[c], DType::F32, dev))
        .transpose()?;
    let cst = d
        .conv_state
        .as_ref()
        .map(|v| tensor(v, &[cs.b, c, cs.k - 1], dtype, dev))
        .transpose()?;
    let a = tensor(&d.a, &[cs.b, cs.t, cs.hv], dtype, dev)?;
    let b = tensor(&d.b, &[cs.b, cs.t, cs.hv], dtype, dev)?;
    let na = tensor(&d.neg_a, &[cs.hv], DType::F32, dev)?;
    let dtb = tensor(&d.dt_bias, &[cs.hv], DType::F32, dev)?;
    let st = d
        .state
        .as_ref()
        .map(|v| tensor(v, &[cs.b, cs.hv, cs.dk, cs.dv], DType::F32, dev))
        .transpose()?;
    fused_gated_delta_rule(
        &mixed,
        &w,
        cb.as_ref(),
        cst.as_ref(),
        &a,
        &b,
        &na,
        &dtb,
        st.as_ref(),
        &cs.cfg(),
    )
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    if t.elem_count() == 0 {
        return Ok(vec![]);
    }
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// max |x - y| / max |y|
fn rel(x: &[f32], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in x.iter().zip(y) {
        num = num.max((*a as f64 - b).abs());
        den = den.max(b.abs());
    }
    if den == 0.0 {
        num
    } else {
        num / den
    }
}

fn rel32(x: &[f32], y: &[f32]) -> f64 {
    rel(x, &y.iter().map(|v| *v as f64).collect::<Vec<_>>())
}

fn gpu() -> Option<Device> {
    if let Ok(d) = Device::new_cuda(0) {
        return Some(d);
    }
    if let Ok(d) = Device::new_rocm(0) {
        if fused_gdn_supported(&d) {
            return Some(d);
        }
    }
    None
}

fn cases() -> Vec<Case> {
    let mut v = vec![];
    // decode step
    v.push(Case::new(1, 1, 2, 4, 64, 64, 4));
    // prefill with bias, two batches, Dk = Dv = 128
    let mut c = Case::new(2, 37, 2, 4, 128, 128, 4);
    c.bias = true;
    v.push(c);
    // no state, generic head dims (dk padded to 128, partial v tile)
    let mut c = Case::new(1, 20, 3, 3, 96, 80, 4);
    c.state = false;
    v.push(c);
    // tiled head order, dk = 256, conv width 3
    let mut c = Case::new(2, 13, 2, 6, 256, 64, 3);
    c.order = GdnHeadOrder::Tiled;
    v.push(c);
    // without L2 norm (small inputs keep the unnormalized rule stable), conv width 2
    let mut c = Case::new(1, 12, 1, 2, 32, 48, 2);
    c.l2 = false;
    c.x_scale = 0.2;
    v.push(c);
    // small odd geometry, tiled, more tokens than one chunk
    let mut c = Case::new(2, 19, 2, 4, 64, 32, 4);
    c.order = GdnHeadOrder::Tiled;
    v.push(c);
    // conv width 5 (taps read from global memory on the GPU), dv = 200 (two value tiles)
    let mut c = Case::new(1, 10, 1, 2, 64, 200, 5);
    c.bias = true;
    v.push(c);
    // conv width 1 (empty conv state), three batches, dv = 64 (4-warp tile)
    v.push(Case::new(3, 9, 2, 2, 128, 64, 1));
    v
}

fn check_f32(dev: &Device, tag: &str) -> Result<()> {
    for (i, cs) in cases().iter().enumerate() {
        let d = gen(cs, 100 + i as u64);
        let r = reference(cs, &d);
        let o = run(cs, &d, dev, DType::F32)?;
        assert_eq!(o.out.dims(), &[cs.b, cs.t, cs.hv, cs.dv]);
        assert_eq!(o.conv_state.dims(), &[cs.b, cs.c(), cs.k - 1]);
        assert_eq!(o.recurrent_state.dims(), &[cs.b, cs.hv, cs.dk, cs.dv]);
        assert_eq!(o.recurrent_state.dtype(), DType::F32);
        let eo = rel(&host(&o.out)?, &r.out);
        let es = rel(&host(&o.recurrent_state)?, &r.state);
        let ec = rel(&host(&o.conv_state)?, &r.conv_state);
        println!("{tag} f32 case {i} {cs:?}: out {eo:.2e} state {es:.2e} conv {ec:.2e}");
        assert!(eo <= 2e-5, "{tag} case {i}: out rel err {eo:.3e}");
        assert!(es <= 2e-5, "{tag} case {i}: state rel err {es:.3e}");
        assert_eq!(ec, 0.0, "{tag} case {i}: conv state");
    }
    Ok(())
}

#[test]
fn cpu_matches_f64_reference() -> Result<()> {
    assert!(fused_gdn_supported(&Device::Cpu));
    check_f32(&Device::Cpu, "cpu")
}

#[test]
fn gpu_matches_f64_reference() -> Result<()> {
    let Some(dev) = gpu() else {
        eprintln!("no GPU, skipping");
        return Ok(());
    };
    check_f32(&dev, "gpu")
}

/// f16 / bf16 inputs: within 1e-2 of the f32-input result, and the GPU kernel matches the CPU
/// implementation run on the same rounded inputs.
#[test]
fn half_inputs() -> Result<()> {
    let devs: Vec<Device> = std::iter::once(Device::Cpu).chain(gpu()).collect();
    for (i, cs) in cases().iter().enumerate() {
        let d = gen(cs, 200 + i as u64);
        let full = run(cs, &d, &Device::Cpu, DType::F32)?;
        let full_out = host(&full.out)?;
        let full_state = host(&full.recurrent_state)?;
        for dtype in [DType::F16, DType::BF16] {
            let cpu = run(cs, &d, &Device::Cpu, dtype)?;
            for dev in devs.iter() {
                let o = run(cs, &d, dev, dtype)?;
                assert_eq!(o.out.dtype(), dtype);
                assert_eq!(o.conv_state.dtype(), dtype);
                let eo = rel32(&host(&o.out)?, &full_out);
                let es = rel32(&host(&o.recurrent_state)?, &full_state);
                let po = rel32(&host(&o.out)?, &host(&cpu.out)?);
                let ps = rel32(&host(&o.recurrent_state)?, &host(&cpu.recurrent_state)?);
                println!(
                    "{dtype:?} {:?} case {i}: vs f32 out {eo:.2e} state {es:.2e}; vs cpu same dtype out {po:.2e} state {ps:.2e}",
                    dev.location()
                );
                assert!(eo <= 1e-2 && es <= 1e-2, "{dtype:?} case {i}: {eo:.3e} {es:.3e}");
                let tol_out = if dtype == DType::BF16 { 1e-2 } else { 2e-3 };
                assert!(po <= tol_out, "{dtype:?} case {i}: out vs cpu {po:.3e}");
                assert!(ps <= 2e-5, "{dtype:?} case {i}: state vs cpu {ps:.3e}");
                assert_eq!(host(&o.conv_state)?, host(&cpu.conv_state)?);
            }
        }
    }
    Ok(())
}

/// T single-token steps equal one call over T tokens (states round-trip exactly).
#[test]
fn decode_chaining() -> Result<()> {
    let devs: Vec<Device> = std::iter::once(Device::Cpu).chain(gpu()).collect();
    for dev in devs.iter() {
        for dtype in [DType::F32, DType::BF16] {
            let mut cs = Case::new(1, 11, 2, 4, 64, 64, 4);
            cs.bias = true;
            let d = gen(&cs, 7);
            let full = run(&cs, &d, dev, dtype)?;
            let c = cs.c();
            let mut conv_state = d.conv_state.clone().unwrap();
            let mut state = d.state.clone().unwrap();
            let mut outs = vec![];
            for t in 0..cs.t {
                let mut one = cs;
                one.t = 1;
                let step = Data {
                    mixed: d.mixed[t * c..(t + 1) * c].to_vec(),
                    a: d.a[t * cs.hv..(t + 1) * cs.hv].to_vec(),
                    b: d.b[t * cs.hv..(t + 1) * cs.hv].to_vec(),
                    conv_state: Some(conv_state.clone()),
                    state: Some(state.clone()),
                    ..d.clone()
                };
                let o = run(&one, &step, dev, dtype)?;
                outs.extend(host(&o.out)?);
                conv_state = host(&o.conv_state)?;
                state = host(&o.recurrent_state)?;
            }
            let eo = rel32(&outs, &host(&full.out)?);
            let es = rel32(&state, &host(&full.recurrent_state)?);
            println!("{:?} {dtype:?} chaining: out {eo:.2e} state {es:.2e}", dev.location());
            assert!(eo <= 1e-6 && es <= 1e-6, "chaining {eo:.3e} {es:.3e}");
            assert_eq!(conv_state, host(&full.conv_state)?);
        }
    }
    Ok(())
}

/// Strided `mixed_qkv` / `a` / `b` views (no copy on the GPU path) and `[C, 1, K]` weights.
#[test]
fn strided_inputs() -> Result<()> {
    let devs: Vec<Device> = std::iter::once(Device::Cpu).chain(gpu()).collect();
    let cs = Case::new(2, 9, 2, 4, 64, 32, 4);
    let d = gen(&cs, 11);
    let c = cs.c();
    for dev in devs.iter() {
        let base = run(&cs, &d, dev, DType::F32)?;
        // mixed as a narrow of a wider projection, a / b as narrows of a fused [.., 2 Hv]
        let mixed = tensor(&d.mixed, &[cs.b, cs.t, c], DType::F32, dev)?;
        let pad = Tensor::zeros((cs.b, cs.t, 5), DType::F32, dev)?;
        let wide = Tensor::cat(&[&pad, &mixed, &pad], 2)?;
        let mixed_v = wide.narrow(2, 5, c)?;
        let a = tensor(&d.a, &[cs.b, cs.t, cs.hv], DType::F32, dev)?;
        let b = tensor(&d.b, &[cs.b, cs.t, cs.hv], DType::F32, dev)?;
        let ab = Tensor::cat(&[&b, &a], 2)?;
        let w = tensor(&d.conv_w, &[c, 1, cs.k], DType::F32, dev)?;
        let cst = tensor(d.conv_state.as_ref().unwrap(), &[cs.b, c, cs.k - 1], DType::F32, dev)?;
        let st = tensor(
            d.state.as_ref().unwrap(),
            &[cs.b, cs.hv, cs.dk, cs.dv],
            DType::F32,
            dev,
        )?;
        let na = tensor(&d.neg_a, &[1, cs.hv], DType::F32, dev)?;
        let dtb = tensor(&d.dt_bias, &[cs.hv], DType::F32, dev)?;
        let o = fused_gated_delta_rule(
            &mixed_v,
            &w,
            None,
            Some(&cst),
            &ab.narrow(2, cs.hv, cs.hv)?,
            &ab.narrow(2, 0, cs.hv)?,
            &na,
            &dtb,
            Some(&st),
            &cs.cfg(),
        )?;
        assert_eq!(host(&o.out)?, host(&base.out)?);
        assert_eq!(host(&o.recurrent_state)?, host(&base.recurrent_state)?);
        assert_eq!(host(&o.conv_state)?, host(&base.conv_state)?);
        // a / b in f32 while mixed_qkv is bf16 (no lossy cast of the gates)
        let mixed_bf = mixed.to_dtype(DType::BF16)?;
        let o_bf = fused_gated_delta_rule(
            &mixed_bf,
            &w,
            None,
            Some(&cst.to_dtype(DType::BF16)?),
            &a,
            &b,
            &na,
            &dtb,
            Some(&st),
            &cs.cfg(),
        )?;
        let mut d_bf = d.clone();
        d_bf.mixed = host(&mixed_bf)?;
        d_bf.conv_state = Some(host(&cst.to_dtype(DType::BF16)?)?);
        let r = reference(&cs, &d_bf);
        let es = rel(&host(&o_bf.recurrent_state)?, &r.state);
        println!("{:?} bf16 mixed + f32 gates: state {es:.2e}", dev.location());
        assert!(es <= 2e-5, "bf16 mixed / f32 gates: {es:.3e}");
    }
    Ok(())
}

#[test]
fn empty_sequence() -> Result<()> {
    let cs = Case::new(1, 0, 2, 4, 64, 32, 4);
    let d = gen(&cs, 3);
    let o = run(&cs, &d, &Device::Cpu, DType::F32)?;
    assert_eq!(o.out.dims(), &[1, 0, 4, 32]);
    assert_eq!(host(&o.recurrent_state)?, d.state.unwrap());
    assert_eq!(host(&o.conv_state)?, d.conv_state.unwrap());
    Ok(())
}

fn gnorm_reference(x: &[f32], z: &[f32], w: &[f32], d: usize, eps: f64) -> Vec<f64> {
    let mut y = vec![0f64; x.len()];
    for r in 0..x.len() / d {
        let xr = &x[r * d..(r + 1) * d];
        let ms = xr.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        for j in 0..d {
            let zz = z[r * d + j] as f64;
            y[r * d + j] = xr[j] as f64 * inv * w[j] as f64 * zz / (1.0 + (-zz).exp());
        }
    }
    y
}

#[test]
fn gated_rms_norm() -> Result<()> {
    let devs: Vec<Device> = std::iter::once(Device::Cpu).chain(gpu()).collect();
    let mut r = Rng(5);
    for (shape, d) in [(vec![3usize, 7, 128], 128usize), (vec![5, 96], 96), (vec![2, 33], 33)] {
        let n: usize = shape.iter().product();
        let x = r.normals(n, 1.0);
        let z = r.normals(n, 1.5);
        let w = r.normals(d, 1.0);
        let want = gnorm_reference(&x, &z, &w, d, 1e-6);
        for dev in devs.iter() {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let xt = tensor(&x, &shape, dtype, dev)?;
                let zt = tensor(&z, &shape, dtype, dev)?;
                let wt = tensor(&w, &[d], DType::F32, dev)?;
                let y = fused_gated_rms_norm(&xt, &zt, &wt, 1e-6)?;
                assert_eq!(y.dtype(), dtype);
                assert_eq!(y.dims(), shape.as_slice());
                // reference on the rounded inputs
                let xr = host(&xt)?;
                let zr = host(&zt)?;
                let want_r = gnorm_reference(&xr, &zr, &w, d, 1e-6);
                let e = rel(&host(&y)?, &want_r);
                let tol = match dtype {
                    DType::F32 => 2e-6,
                    DType::F16 => 1e-3,
                    _ => 8e-3,
                };
                println!("gnorm {:?} {dtype:?} {shape:?}: {e:.2e}", dev.location());
                assert!(e <= tol, "gnorm {dtype:?} {shape:?}: {e:.3e}");
                if dtype == DType::F32 {
                    assert!(rel(&host(&y)?, &want) <= 2e-6);
                }
            }
        }
    }
    Ok(())
}

/// transformers fixtures (`gen_gdn_ref.py`): `CANDLE_GDN_FIXTURES=<dir of *.safetensors>`.
#[test]
fn transformers_fixtures() -> Result<()> {
    let Ok(dir) = std::env::var("CANDLE_GDN_FIXTURES") else {
        eprintln!("CANDLE_GDN_FIXTURES not set, skipping");
        return Ok(());
    };
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no fixtures in {dir}");
    let devs: Vec<Device> = std::iter::once(Device::Cpu).chain(gpu()).collect();
    for f in files {
        let t = candle_core::safetensors::load(&f, &Device::Cpu)?;
        let dims: Vec<usize> = t["dims"]
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let (bsz, tl, hk, hv, dk, dv, k) = (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
        let mut cfg = GdnConfig::new(hk, hv, dk, dv);
        cfg.conv_kernel = k;
        let truth_out = host(&t["out_f64"])?;
        let truth_state = host(&t["state_f64"])?;
        let truth_conv = host(&t["conv_state_out"])?;
        let as64 = |v: &[f32]| v.iter().map(|x| *x as f64).collect::<Vec<_>>();
        let (to, ts, tcv) = (as64(&truth_out), as64(&truth_state), as64(&truth_conv));
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let ref_rec = rel(&host(&t["out_rec_f32"])?, &to);
        let ref_chunk = rel(&host(&t["out_chunk_f32"])?, &to);
        let ref_rec_s = rel(&host(&t["state_rec_f32"])?, &ts);
        let ref_chunk_s = rel(&host(&t["state_chunk_f32"])?, &ts);
        println!(
            "{name} [B{bsz} T{tl} Hk{hk} Hv{hv} Dk{dk} Dv{dv} K{k}]: transformers f32 recurrent out {ref_rec:.2e} state {ref_rec_s:.2e}; chunk out {ref_chunk:.2e} state {ref_chunk_s:.2e}"
        );
        for dev in devs.iter() {
            let g = |n: &str| t[n].to_device(dev);
            let neg_a = g("A_log")?.exp()?.neg()?;
            let o = fused_gated_delta_rule(
                &g("mixed_qkv")?,
                &g("conv_w")?,
                None,
                Some(&g("conv_state")?),
                &g("a")?,
                &g("b")?,
                &neg_a,
                &g("dt_bias")?,
                Some(&g("rec_state")?),
                &cfg,
            )?;
            let eo = rel(&host(&o.out)?, &to);
            let es = rel(&host(&o.recurrent_state)?, &ts);
            let ec = rel(&host(&o.conv_state)?, &tcv);
            println!(
                "  {:?}: out {eo:.2e} state {es:.2e} conv_state {ec:.2e}",
                dev.location()
            );
            assert!(eo <= 5e-6, "{name}: out {eo:.3e}");
            assert!(es <= 5e-6, "{name}: state {es:.3e}");
            assert!(ec <= 1e-7, "{name}: conv state {ec:.3e}");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Host emulation of the GPU kernels (kernels/tests/delta_rule_host_test.cpp): covers the
// kernel logic (the same source hipcc compiles for ROCm) without a GPU. Needs a C++20
// compiler: `CXX` (g++ / clang++ style, or MSVC `cl` from a developer prompt), else `c++`;
// skipped when none works.
// ---------------------------------------------------------------------------

fn host_harness() -> Option<&'static std::path::Path> {
    use std::path::PathBuf;
    use std::process::Command;
    static BIN: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("tests")
            .join("delta_rule_host_test.cpp");
        let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
        let msvc = {
            let stem = std::path::Path::new(&cxx)
                .file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            stem == "cl" || stem == "clang-cl"
        };
        let bin = dir.join(if cfg!(windows) {
            "delta_rule_host_test.exe"
        } else {
            "delta_rule_host_test"
        });
        let mut cmd = Command::new(&cxx);
        if msvc {
            cmd.args(["/nologo", "/std:c++20", "/O2", "/EHsc"])
                .arg(format!("/Fe:{}", bin.display()))
                .arg(format!("/Fo:{}\\", dir.display()))
                .arg(&src);
        } else {
            cmd.args(["-std=c++20", "-O2", "-pthread", "-o"])
                .arg(&bin)
                .arg(&src);
        }
        match cmd.output() {
            Ok(o) if o.status.success() => Some(bin),
            Ok(o) => {
                eprintln!(
                    "skipping delta rule host kernel tests: {cxx} failed\n{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                );
                None
            }
            Err(e) => {
                eprintln!("skipping delta rule host kernel tests: no C++ compiler ({cxx}: {e})");
                None
            }
        }
    })
    .as_deref()
}

#[derive(Default)]
struct HostArrays(std::collections::BTreeMap<String, (u8, Vec<u8>)>);

impl HostArrays {
    fn f32(&mut self, name: &str, v: &[f32]) -> &mut Self {
        let b = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.0.insert(name.into(), (0, b));
        self
    }
    fn u32(&mut self, name: &str, v: &[u32]) -> &mut Self {
        let b = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.0.insert(name.into(), (1, b));
        self
    }
    fn str(&mut self, name: &str, s: &str) -> &mut Self {
        self.0.insert(name.into(), (4, s.as_bytes().to_vec()));
        self
    }
    fn get_f32(&self, name: &str) -> Vec<f32> {
        self.0[name]
            .1
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

fn host_run(bin: &std::path::Path, input: &HostArrays) -> HostArrays {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let pid = std::process::id();
    let inp = dir.join(format!("gdn_in_{pid}_{n}.bin"));
    let outp = dir.join(format!("gdn_out_{pid}_{n}.bin"));
    let mut bytes = Vec::new();
    for (name, (kind, data)) in input.0.iter() {
        bytes.extend((name.len() as u32).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.push(*kind);
        bytes.extend((data.len() as u64).to_le_bytes());
        bytes.extend(data);
    }
    std::fs::write(&inp, bytes).unwrap();
    let st = std::process::Command::new(bin)
        .arg(&inp)
        .arg(&outp)
        .status()
        .unwrap();
    assert!(st.success(), "delta rule host harness failed");
    let data = std::fs::read(&outp).unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
    let mut out = HostArrays::default();
    let mut p = 0;
    while p < data.len() {
        let nl = u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let name = String::from_utf8(data[p..p + nl].to_vec()).unwrap();
        p += nl;
        let kind = data[p];
        p += 1;
        let nb = u64::from_le_bytes(data[p..p + 8].try_into().unwrap()) as usize;
        p += 8;
        out.0.insert(name, (kind, data[p..p + nb].to_vec()));
        p += nb;
    }
    out
}

/// The kernels on the host emulator match the CPU implementation (f32 and bf16 activations,
/// grouped / tiled heads, every key head dim tile, conv widths 3 / 4 / 5, partial tiles).
#[test]
fn host_emulated_kernels() -> Result<()> {
    let Some(bin) = host_harness() else {
        return Ok(());
    };
    // (case, harness tile: 0 d64 / 1 d128 / 2 d256)
    let mut cases = vec![];
    let mut c = Case::new(2, 11, 1, 2, 64, 48, 4);
    c.bias = true;
    cases.push((c, 0u32));
    let mut c = Case::new(1, 9, 2, 4, 96, 80, 3);
    c.order = GdnHeadOrder::Tiled;
    c.state = false;
    cases.push((c, 1));
    let mut c = Case::new(1, 5, 1, 1, 256, 40, 5);
    c.l2 = false;
    c.x_scale = 0.2;
    cases.push((c, 2));
    for (i, (cs, tile)) in cases.iter().enumerate() {
        let d = gen(cs, 300 + i as u64);
        for (dtype, code) in [(DType::F32, 0u32), (DType::BF16, 2)] {
            let cpu = run(cs, &d, &Device::Cpu, dtype)?;
            let flags = cs.l2 as u32 | ((cs.order == GdnHeadOrder::Tiled) as u32) << 1;
            let mut inp = HostArrays::default();
            inp.str("op", "scan")
                .u32(
                    "dims",
                    &[
                        cs.b as u32,
                        cs.t as u32,
                        cs.hk as u32,
                        cs.hv as u32,
                        cs.dk as u32,
                        cs.dv as u32,
                        cs.k as u32,
                        flags,
                        code,
                        *tile,
                    ],
                )
                .f32("fparams", &[1e-6, 1.0 / (cs.dk as f32).sqrt()])
                .f32("x", &d.mixed)
                .f32("cw", &d.conv_w)
                .f32("a", &d.a)
                .f32("b", &d.b)
                .f32("neg_a", &d.neg_a)
                .f32("dt_bias", &d.dt_bias);
            if let Some(v) = &d.conv_b {
                inp.f32("cb", v);
            }
            if let Some(v) = &d.conv_state {
                inp.f32("cs_in", v);
            }
            if let Some(v) = &d.state {
                inp.f32("s_in", v);
            }
            let o = host_run(bin, &inp);
            let eo = rel32(&o.get_f32("out"), &host(&cpu.out)?);
            let es = rel32(&o.get_f32("s_out"), &host(&cpu.recurrent_state)?);
            println!("host emu {dtype:?} case {i} {cs:?}: out {eo:.2e} state {es:.2e}");
            let tol_out = if dtype == DType::F32 { 2e-6 } else { 1e-2 };
            assert!(eo <= tol_out, "host emu case {i} {dtype:?}: out {eo:.3e}");
            assert!(es <= 2e-6, "host emu case {i} {dtype:?}: state {es:.3e}");
            assert_eq!(o.get_f32("cs_out"), host(&cpu.conv_state)?, "host emu case {i} conv state");
        }
    }
    let mut r = Rng(9);
    for (rows, dd) in [(9usize, 40usize), (6, 300)] {
        let x = r.normals(rows * dd, 1.0);
        let z = r.normals(rows * dd, 1.5);
        let w = r.normals(dd, 1.0);
        for (dtype, code) in [(DType::F32, 0u32), (DType::F16, 1)] {
            let xt = tensor(&x, &[rows, dd], dtype, &Device::Cpu)?;
            let zt = tensor(&z, &[rows, dd], dtype, &Device::Cpu)?;
            let want = gnorm_reference(&host(&xt)?, &host(&zt)?, &w, dd, 1e-6);
            let mut inp = HostArrays::default();
            inp.str("op", "gnorm")
                .u32("dims", &[rows as u32, dd as u32, code])
                .f32("fparams", &[1e-6])
                .f32("x", &x)
                .f32("z", &z)
                .f32("w", &w);
            let o = host_run(bin, &inp);
            let e = rel(&o.get_f32("y"), &want);
            println!("host emu gnorm {dtype:?} [{rows}, {dd}]: {e:.2e}");
            assert!(e <= if dtype == DType::F32 { 2e-6 } else { 1e-3 });
        }
    }
    Ok(())
}

/// Median wall time per call (us) of `f` over `reps` timed batches of `iters` calls after a
/// warm-up of at least 300 ms (the GPU clocks ramp up).
fn time_us(dev: &Device, iters: usize, reps: usize, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < 300 {
        for _ in 0..iters.min(10) {
            f()?;
        }
        dev.synchronize()?;
    }
    let mut v = vec![];
    for _ in 0..reps {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        dev.synchronize()?;
        v.push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(v[v.len() / 2])
}

/// Per-layer latency of the fused kernel for the Qwen3.5-4B (Hk 16, Hv 32) and 2B (Hk = Hv =
/// 16) linear-attention geometries (Dk = Dv = 128, K 4): `--ignored --nocapture`.
/// `CANDLE_GDN_BENCH_TILES=c16x2w8t8,c16x2w4t8` compares instantiated tiles.
#[test]
#[ignore]
fn bench_gdn_qwen35() -> Result<()> {
    let Some(dev) = gpu() else {
        eprintln!("no GPU, skipping");
        return Ok(());
    };
    let tiles: Vec<String> = match std::env::var("CANDLE_GDN_BENCH_TILES") {
        Ok(s) => s.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => vec![String::new()],
    };
    for tile in tiles {
        if tile.is_empty() {
            std::env::remove_var("CANDLE_GDN_TILE");
        } else {
            std::env::set_var("CANDLE_GDN_TILE", &tile);
        }
        let tname = if tile.is_empty() { "default" } else { &tile };
        for (model, hk, hv) in [("4B", 16usize, 32usize), ("2B", 16, 16)] {
            for dtype in [DType::F32, DType::BF16] {
                for (t, iters) in [(1usize, 200usize), (902, 10), (2048, 5)] {
                    let mut cs = Case::new(1, t, hk, hv, 128, 128, 4);
                    cs.state = true;
                    let d = gen(&cs, 1);
                    let c = cs.c();
                    let mixed = tensor(&d.mixed, &[1, t, c], dtype, &dev)?;
                    let w = tensor(&d.conv_w, &[c, 4], DType::F32, &dev)?;
                    let cst = tensor(d.conv_state.as_ref().unwrap(), &[1, c, 3], dtype, &dev)?;
                    let a = tensor(&d.a, &[1, t, hv], dtype, &dev)?;
                    let b = tensor(&d.b, &[1, t, hv], dtype, &dev)?;
                    let na = tensor(&d.neg_a, &[hv], DType::F32, &dev)?;
                    let dtb = tensor(&d.dt_bias, &[hv], DType::F32, &dev)?;
                    let st = tensor(d.state.as_ref().unwrap(), &[1, hv, 128, 128], DType::F32, &dev)?;
                    let cfg = cs.cfg();
                    let us = time_us(&dev, iters, 5, &|| {
                        fused_gated_delta_rule(
                            &mixed,
                            &w,
                            None,
                            Some(&cst),
                            &a,
                            &b,
                            &na,
                            &dtb,
                            Some(&st),
                            &cfg,
                        )
                        .map(|_| ())
                    })?;
                    println!(
                        "gdn {model} tile {tname:>10} {dtype:?} T={t:>5}: {us:9.1} us / layer ({:.3} us / token)",
                        us / t as f64
                    );
                }
            }
        }
    }
    // gated norm on the prefill output size of the 4B (T = 2048, 32 heads of 128)
    for dtype in [DType::F32, DType::BF16] {
        let x = Tensor::randn(0f32, 1f32, (2048 * 32, 128), &dev)?.to_dtype(dtype)?;
        let z = Tensor::randn(0f32, 1f32, (2048 * 32, 128), &dev)?.to_dtype(dtype)?;
        let w = Tensor::ones(128, DType::F32, &dev)?;
        let us = time_us(&dev, 10, 5, &|| fused_gated_rms_norm(&x, &z, &w, 1e-6).map(|_| ()))?;
        println!("gated rms norm {dtype:?} [65536, 128]: {us:.1} us");
        let x1 = x.narrow(0, 0, 32)?;
        let z1 = z.narrow(0, 0, 32)?;
        let us = time_us(&dev, 200, 5, &|| fused_gated_rms_norm(&x1, &z1, &w, 1e-6).map(|_| ()))?;
        println!("gated rms norm {dtype:?} [32, 128] (decode): {us:.1} us");
    }
    Ok(())
}
