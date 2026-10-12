//! Fused attention / TurboQuant kernels of the Vulkan backend against the CPU reference.
//! Runs on any Vulkan device; set `CANDLE_VULKAN_ALLOW_CPU=1` to use lavapipe.
#![cfg(feature = "vulkan")]

use candle_core::accel::attention::{
    reference_attention, AttnGeometry, FaExtra, FlashAttnParams, ReferenceInputs,
};
use candle_core::accel::formats::{row_layout, QuantFormat};
use candle_core::accel::turbo::{RowCodec, SplitMix64, TqAttnArgs, TqRowArgs, TqSide};
use candle_core::accel::FlashAttnVersion;
use candle_core::{DType, Device, Storage, Tensor};

fn device() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping vulkan accel tests: {e}");
            None
        }
    }
}

fn randn(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_normal() as f32).collect()
}

fn vk(t: &Tensor) -> std::sync::RwLockReadGuard<'_, Storage> {
    t.storage_and_layout().0
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn run_fa(
    dev: &Device,
    version: FlashAttnVersion,
    b: usize,
    lq: usize,
    lk: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    causal: bool,
    fp8: bool,
) {
    let mut rng = SplitMix64::new((lq * 31 + lk + d) as u64);
    let q = randn(&mut rng, b * lq * hq * d);
    let k = randn(&mut rng, b * lk * hkv * d);
    let v = randn(&mut rng, b * lk * hkv * d);
    let qt = Tensor::from_vec(q.clone(), (b, lq, hq, d), dev).unwrap();
    let mut kt = Tensor::from_vec(k.clone(), (b, lk, hkv, d), dev).unwrap();
    let mut vt = Tensor::from_vec(v.clone(), (b, lk, hkv, d), dev).unwrap();
    let (mut kr, mut vr) = (k.clone(), v.clone());
    if fp8 {
        kt = kt.to_dtype(DType::F8E4M3).unwrap();
        vt = vt.to_dtype(DType::F8E4M3).unwrap();
        kr = kt
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        vr = vt
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
    }
    let mut p = FlashAttnParams::new(1.0 / (d as f32).sqrt());
    p.version = version;
    if causal {
        p.window_right = 0;
    }
    let geom = AttnGeometry::dense(qt.layout(), kt.layout(), vt.layout()).unwrap();
    let (qs, ks, vs) = (vk(&qt), vk(&kt), vk(&vt));
    let (Storage::Vulkan(qs), Storage::Vulkan(ks), Storage::Vulkan(vs)) = (&*qs, &*ks, &*vs) else {
        panic!("not vulkan")
    };
    let out = qs
        .flash_attn(ks, vs, &geom, &p, &FaExtra::default())
        .unwrap();
    let got = out.as_slice::<f32>().unwrap().to_vec();
    let want = reference_attention(
        &ReferenceInputs {
            q: &q,
            k: &kr,
            v: &vr,
            batch: b,
            lq,
            lk,
            hq,
            hkv,
            d,
            cu_q: None,
            cu_k: None,
            alibi: None,
        },
        &p,
    )
    .unwrap();
    let e = max_err(&got, &want);
    assert!(
        e < 1e-4,
        "{version:?} b{b} lq{lq} lk{lk} d{d} causal {causal} fp8 {fp8}: max err {e}"
    );
}

#[test]
fn vulkan_flash_attn_matches_reference() {
    let Some(dev) = device() else { return };
    run_fa(&dev, FlashAttnVersion::V1, 2, 9, 40, 4, 2, 64, true, false);
    run_fa(&dev, FlashAttnVersion::V2, 1, 70, 70, 2, 1, 64, true, false);
    run_fa(
        &dev,
        FlashAttnVersion::V2,
        1,
        33,
        50,
        2,
        2,
        128,
        false,
        false,
    );
    run_fa(&dev, FlashAttnVersion::V2, 1, 20, 20, 2, 2, 256, true, true);
    // decoding: row kernel with split-KV
    run_fa(&dev, FlashAttnVersion::V2, 1, 1, 700, 4, 1, 96, true, false);
}

#[test]
fn vulkan_turbo_quant_matches_cpu_codec() {
    let Some(dev) = device() else { return };
    let mut rng = SplitMix64::new(3);
    for (f, d) in [
        (QuantFormat::TurboMse { bits: 3 }, 128),
        (QuantFormat::TurboProd { bits: 4 }, 64),
        (QuantFormat::Nvfp4, 64),
        (QuantFormat::Mxfp4, 64),
        (QuantFormat::Mxfp8, 128),
        (QuantFormat::Fp8, 64),
    ] {
        let codec = RowCodec::new(f, d, 9).unwrap();
        let rows = 19;
        let x = randn(&mut rng, rows * d);
        let xt = Tensor::from_vec(x.clone(), (rows, d), &dev).unwrap();
        let tab: Vec<f32> = codec
            .codebook
            .as_ref()
            .map(|c| c.table().to_vec())
            .unwrap_or_else(|| vec![0f32; 32]);
        let tt = Tensor::from_vec(tab, 32, &dev).unwrap();
        let a = TqRowArgs::new(f, rows, d).unwrap();
        let (xs, ts) = (vk(&xt), vk(&tt));
        let (Storage::Vulkan(xs), Storage::Vulkan(ts)) = (&*xs, &*ts) else {
            panic!()
        };
        let mut data = xs.tq_quantize(0, ts, 0, &a).unwrap();
        if let Some(s) = &codec.qjl {
            let resid = xs.tq_residual(0, &data, 0, ts, 0, &a).unwrap();
            let r = resid.as_slice::<f32>().unwrap();
            let proj: Vec<f32> = r
                .chunks(d)
                .flat_map(|row| {
                    (0..d)
                        .map(|i| (0..d).map(|j| s[i * d + j] * row[j]).sum::<f32>())
                        .collect::<Vec<_>>()
                })
                .collect();
            let pt = Tensor::from_vec(proj, (rows, d), &dev).unwrap();
            let ps = vk(&pt);
            let Storage::Vulkan(ps) = &*ps else { panic!() };
            data = data.tq_signs(0, ps, 0, &resid, 0, &a).unwrap();
        }
        let words = data.as_slice::<u32>().unwrap().to_vec();
        let l = row_layout(f, d).unwrap();
        // dequantize on the GPU, compare with the CPU codec on the same words
        let deq = data
            .tq_dequantize(0, ts, 0, &a, codec.qjl.is_some())
            .unwrap();
        let deq = deq.as_slice::<f32>().unwrap().to_vec();
        let mut mismatched = 0;
        for r in 0..rows {
            let row = &words[r * l.row_words..(r + 1) * l.row_words];
            let mut cpu_words = vec![0u32; l.row_words];
            codec.quantize_row(&x[r * d..(r + 1) * d], &mut cpu_words);
            mismatched += row[..l.code_words]
                .iter()
                .zip(&cpu_words[..l.code_words])
                .filter(|(a, b)| a != b)
                .count();
            let (r0, _) = codec.rsc(row);
            let want_row: Vec<f32> = (0..d).map(|j| r0 * codec.elem(row, j)).collect();
            let e = max_err(&deq[r * d..(r + 1) * d], &want_row);
            assert!(e < 1e-5, "{f} row {r}: dequant err {e}");
        }
        // the 2.5 ulp division of SPIR-V may flip a rare code against the CPU codec
        assert!(
            mismatched <= rows * l.code_words / 50,
            "{f}: {mismatched} code words differ"
        );
    }
}

#[test]
fn vulkan_turbo_quant_attention() {
    let Some(dev) = device() else { return };
    let (b, lq, lk, hq, hkv, d) = (1usize, 2usize, 300usize, 4usize, 2usize, 64usize);
    let (fk, fv) = (QuantFormat::TurboMse { bits: 4 }, QuantFormat::Mxfp8);
    let ck = RowCodec::new(fk, d, 1).unwrap();
    let cv = RowCodec::new(fv, d, 2).unwrap();
    let (lkk, lvv) = (ck.layout, cv.layout);
    let rw = lkk.row_words.max(lvv.row_words);
    let mut rng = SplitMix64::new(42);
    let n_rows = b * hkv * lk;
    let mut kd = vec![0u32; n_rows * rw];
    let mut vd = vec![0u32; n_rows * rw];
    let mut kdeq = vec![0f32; n_rows * d];
    let mut vdeq = vec![0f32; n_rows * d];
    for r in 0..n_rows {
        let kx = randn(&mut rng, d);
        let vx = randn(&mut rng, d);
        ck.quantize_row(&kx, &mut kd[r * rw..r * rw + lkk.row_words]);
        cv.quantize_row(&vx, &mut vd[r * rw..r * rw + lvv.row_words]);
        ck.dequantize_row(
            &kd[r * rw..r * rw + lkk.row_words],
            &mut kdeq[r * d..(r + 1) * d],
        );
        cv.dequantize_row(
            &vd[r * rw..r * rw + lvv.row_words],
            &mut vdeq[r * d..(r + 1) * d],
        );
    }
    let q = randn(&mut rng, b * hq * lq * d);
    let mut tab = vec![0f32; 64];
    tab[..32].copy_from_slice(&ck.codebook.as_ref().unwrap().table());
    let qt = Tensor::from_vec(q.clone(), (b, hq, lq, d), &dev).unwrap();
    let kt = Tensor::from_vec(kd, n_rows * rw, &dev).unwrap();
    let vt = Tensor::from_vec(vd, n_rows * rw, &dev).unwrap();
    let tt = Tensor::from_vec(tab, 64, &dev).unwrap();
    let a = TqAttnArgs {
        b,
        lq,
        lk,
        hq,
        hkv,
        d,
        q_strides: [hq * lq * d, lq * d, d],
        q_offset: 0,
        k_strides: [hkv * lk * rw, lk * rw, rw],
        v_strides: [hkv * lk * rw, lk * rw, rw],
        k_offset: 0,
        v_offset: 0,
        k: TqSide::new(fk, d).unwrap(),
        v: TqSide::new(fv, d).unwrap(),
        scale: 0.125,
        softcap: 0.0,
        window_left: -1,
        window_right: 0,
        splits: 3,
        alibi_bstride: 0,
    };
    let (qs, ks, vs, ts) = (vk(&qt), vk(&kt), vk(&vt), vk(&tt));
    let (Storage::Vulkan(qs), Storage::Vulkan(ks), Storage::Vulkan(vs), Storage::Vulkan(ts)) =
        (&*qs, &*ks, &*vs, &*ts)
    else {
        panic!()
    };
    let out = qs.tq_attention(ks, vs, ts, 0, None, &a).unwrap();
    let got = out.as_slice::<f32>().unwrap().to_vec();
    // reference: (b, s, h, d) layouts on the dequantized rows
    let mut qr = vec![0f32; q.len()];
    for h in 0..hq {
        for s in 0..lq {
            qr[(s * hq + h) * d..(s * hq + h + 1) * d]
                .copy_from_slice(&q[(h * lq + s) * d..(h * lq + s + 1) * d]);
        }
    }
    let to_bshd = |x: &[f32]| {
        let mut o = vec![0f32; x.len()];
        for h in 0..hkv {
            for t in 0..lk {
                o[(t * hkv + h) * d..(t * hkv + h + 1) * d]
                    .copy_from_slice(&x[(h * lk + t) * d..(h * lk + t + 1) * d]);
            }
        }
        o
    };
    let (kr, vr) = (to_bshd(&kdeq), to_bshd(&vdeq));
    let mut p = FlashAttnParams::new(0.125);
    p.window_right = 0;
    let want = reference_attention(
        &ReferenceInputs {
            q: &qr,
            k: &kr,
            v: &vr,
            batch: b,
            lq,
            lk,
            hq,
            hkv,
            d,
            cu_q: None,
            cu_k: None,
            alibi: None,
        },
        &p,
    )
    .unwrap();
    let mut want_bhsd = vec![0f32; want.len()];
    for s in 0..lq {
        for h in 0..hq {
            want_bhsd[(h * lq + s) * d..(h * lq + s + 1) * d]
                .copy_from_slice(&want[(s * hq + h) * d..(s * hq + h + 1) * d]);
        }
    }
    let e = max_err(&got, &want_bhsd);
    assert!(e < 1e-4, "tq attention max err {e}");
}
