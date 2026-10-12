//! candle_nn::flash_attn / candle_nn::turbo_quant on a Vulkan device against the CPU path.
//! Set `CANDLE_VULKAN_ALLOW_CPU=1` to run on lavapipe.
#![cfg(feature = "vulkan")]

use candle::accel::turbo::SplitMix64;
use candle::accel::{FlashAttnVersion, RotationKind, SplitKv};
use candle::{DType, Device, Tensor};
use candle_nn::flash_attn::{
    flash_attn_bhsd, flash_attn_fp8, flash_attn_varlen_with_config, flash_attn_with_config,
    FlashAttnConfig,
};
use candle_nn::turbo_quant::{QuantFormat, TurboKvCache, TurboQuantizer};

fn vulkan() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn randn(shape: &[usize], seed: u64) -> Tensor {
    let mut rng = SplitMix64::new(seed);
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.next_normal() as f32).collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let b = b
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn flash_attn_vulkan_matches_cpu() {
    let Some(dev) = vulkan() else { return };
    let slopes = Tensor::new(&[0.05f32, 0.1, 0.2, 0.4], &Device::Cpu).unwrap();
    let cases: Vec<(&str, [usize; 6], FlashAttnConfig)> = vec![
        (
            "v2 causal gqa",
            [2, 40, 40, 4, 2, 64],
            FlashAttnConfig::new(0.125).causal(),
        ),
        (
            "v1 window softcap",
            [1, 17, 30, 4, 4, 32],
            FlashAttnConfig::new(0.2)
                .window(Some(5), Some(0))
                .softcap(8.0)
                .version(FlashAttnVersion::V1),
        ),
        (
            "decode split-kv alibi",
            [1, 1, 900, 4, 1, 128],
            FlashAttnConfig::new(0.09)
                .causal()
                .alibi(&slopes)
                .split_kv(SplitKv::Fixed(4)),
        ),
        (
            "cross attention",
            [1, 33, 47, 4, 2, 96],
            FlashAttnConfig::new(0.1),
        ),
    ];
    for (name, [b, lq, lk, hq, hkv, d], cfg) in cases {
        let q = randn(&[b, lq, hq, d], 1);
        let k = randn(&[b, lk, hkv, d], 2);
        let v = randn(&[b, lk, hkv, d], 3);
        let want = flash_attn_with_config(&q, &k, &v, &cfg).unwrap();
        let (qg, kg, vg) = (
            q.to_device(&dev).unwrap(),
            k.to_device(&dev).unwrap(),
            v.to_device(&dev).unwrap(),
        );
        let got = flash_attn_with_config(&qg, &kg, &vg, &cfg.clone().fused(true)).unwrap();
        assert!(got.device().is_vulkan());
        assert!(
            max_diff(&got, &want) < 2e-4,
            "{name}: {}",
            max_diff(&got, &want)
        );
        // f16 inputs: computed in f32 by the shaders, f16 output
        let got16 = flash_attn_with_config(
            &qg.to_dtype(DType::F16).unwrap(),
            &kg.to_dtype(DType::F16).unwrap(),
            &vg.to_dtype(DType::F16).unwrap(),
            &cfg,
        )
        .unwrap();
        assert_eq!(got16.dtype(), DType::F16);
        assert!(
            max_diff(&got16, &want) < 1e-2,
            "{name} f16: {}",
            max_diff(&got16, &want)
        );
    }
}

#[test]
fn flash_attn_vulkan_layouts_and_fp8() {
    let Some(dev) = vulkan() else { return };
    // (b, h, s, d) model layout through strides, no copies
    let q = randn(&[1, 4, 12, 64], 4);
    let k = randn(&[1, 2, 12, 64], 5);
    let v = randn(&[1, 2, 12, 64], 6);
    let cfg = FlashAttnConfig::new(0.125).causal();
    let want = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
    let got = flash_attn_bhsd(
        &q.to_device(&dev).unwrap(),
        &k.to_device(&dev).unwrap(),
        &v.to_device(&dev).unwrap(),
        &cfg,
    )
    .unwrap();
    assert!(max_diff(&got, &want) < 2e-4);
    // FP8 K/V (FlashAttention-3 style) agrees with the CPU path on the same quantized inputs
    let want = flash_attn_fp8(
        &q.transpose(1, 2).unwrap(),
        &k.transpose(1, 2).unwrap(),
        &v.transpose(1, 2).unwrap(),
        &cfg,
    )
    .unwrap();
    let got = flash_attn_fp8(
        &q.transpose(1, 2).unwrap().to_device(&dev).unwrap(),
        &k.transpose(1, 2).unwrap().to_device(&dev).unwrap(),
        &v.transpose(1, 2).unwrap().to_device(&dev).unwrap(),
        &cfg,
    )
    .unwrap();
    assert!(
        max_diff(&got, &want) < 1e-3,
        "fp8 {}",
        max_diff(&got, &want)
    );
    // varlen
    let q = randn(&[11, 2, 32], 7);
    let k = randn(&[14, 2, 32], 8);
    let v = randn(&[14, 2, 32], 9);
    let cq = Tensor::new(&[0u32, 6, 11], &Device::Cpu).unwrap();
    let ck = Tensor::new(&[0u32, 8, 14], &Device::Cpu).unwrap();
    let want = flash_attn_varlen_with_config(&q, &k, &v, &cq, &ck, 6, 8, &cfg).unwrap();
    let got = flash_attn_varlen_with_config(
        &q.to_device(&dev).unwrap(),
        &k.to_device(&dev).unwrap(),
        &v.to_device(&dev).unwrap(),
        &cq,
        &ck,
        6,
        8,
        &cfg,
    )
    .unwrap();
    assert!(max_diff(&got, &want) < 2e-4);
}

#[test]
fn turbo_quant_vulkan_matches_cpu() {
    let Some(dev) = vulkan() else { return };
    let x = randn(&[3, 7, 64], 10);
    for f in [
        QuantFormat::TurboMse { bits: 4 },
        QuantFormat::TurboProd { bits: 3 },
        QuantFormat::Nvfp4,
        QuantFormat::Mxfp4,
        QuantFormat::Mxfp8,
        QuantFormat::Fp8,
    ] {
        let cpu = TurboQuantizer::new(f, 64, RotationKind::Hadamard, 3, &Device::Cpu).unwrap();
        let gpu = TurboQuantizer::new(f, 64, RotationKind::Hadamard, 3, &dev).unwrap();
        let qc = cpu.quantize(&x).unwrap();
        let qg = gpu.quantize(&x.to_device(&dev).unwrap()).unwrap();
        let wc = qc.data.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let wg = qg
            .data
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        // codes: identical up to rare flips at decision boundaries (the rotation matmul and the
        // norms are summed in a different order on the GPU); row scalars: equal up to rounding
        let l = candle::accel::row_layout(gpu.format(), 64).unwrap();
        let (mut code_diff, mut codes) = (0, 0);
        for (rc, rg) in wc.chunks(l.row_words).zip(wg.chunks(l.row_words)) {
            for j in 0..64 {
                let (a, b) = (
                    candle::accel::formats::get_bits(rc, j * l.code_bits as usize, l.code_bits),
                    candle::accel::formats::get_bits(rg, j * l.code_bits as usize, l.code_bits),
                );
                code_diff += (a != b) as usize;
                codes += 1;
            }
            for k in 0..2 {
                let (a, b) = (
                    f32::from_bits(rc[l.rsc_offset() + k]),
                    f32::from_bits(rg[l.rsc_offset() + k]),
                );
                assert!(
                    (a - b).abs() <= 1e-4 * a.abs().max(1e-3),
                    "{f}: row scalar {a} vs {b}"
                );
            }
        }
        assert!(
            code_diff * 100 <= codes,
            "{f}: {code_diff} of {codes} codes differ"
        );
        // the GPU reconstruction of its own rows equals the CPU reconstruction of them
        let dg = gpu.dequantize(&qg).unwrap();
        let dc = cpu
            .dequantize(&candle_nn::turbo_quant::QuantizedTensor {
                data: qg.data.to_device(&Device::Cpu).unwrap(),
                shape: qg.shape.clone(),
                dtype: qg.dtype,
            })
            .unwrap();
        assert!(max_diff(&dg, &dc) < 1e-4, "{f}: {}", max_diff(&dg, &dc));
    }
}

#[test]
fn turbo_kv_cache_fused_vulkan() {
    let Some(dev) = vulkan() else { return };
    for (kf, vf) in [
        (
            QuantFormat::TurboMse { bits: 4 },
            QuantFormat::TurboMse { bits: 3 },
        ),
        (QuantFormat::TurboProd { bits: 4 }, QuantFormat::Mxfp8),
        (QuantFormat::Nvfp4, QuantFormat::Fp8),
    ] {
        let mut cache =
            TurboKvCache::with_formats(kf, vf, 64, RotationKind::Auto, 1, 32, &dev).unwrap();
        for step in 0..3u64 {
            let t = if step == 0 { 40 } else { 1 };
            let k = randn(&[1, 2, t, 64], 20 + step).to_device(&dev).unwrap();
            let v = randn(&[1, 2, t, 64], 30 + step).to_device(&dev).unwrap();
            cache.append(&k, &v).unwrap();
        }
        assert_eq!(cache.seq_len(), 42);
        let q = randn(&[1, 4, 2, 64], 40).to_device(&dev).unwrap();
        let cfg = FlashAttnConfig::new(0.125).causal();
        let fused = cache.attention(&q, &cfg.clone().fused(true)).unwrap();
        let unfused = cache.attention(&q, &cfg.clone().fused(false)).unwrap();
        assert_eq!(fused.dims(), &[1, 4, 2, 64]);
        assert!(
            max_diff(&fused, &unfused) < 2e-4,
            "{kf}/{vf}: {}",
            max_diff(&fused, &unfused)
        );
    }
}
