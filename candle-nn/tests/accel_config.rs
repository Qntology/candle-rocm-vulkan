//! The on/off switches of the fused attention / TurboQuant paths. One test function: the
//! configuration is process global.

use candle::accel::{self, FlashAttnVersion, QuantFormat, RotationKind, SplitKv};
use candle::{Device, Tensor};
use candle_nn::flash_attn::{flash_attn, flash_attn_unfused, FlashAttnConfig};
use candle_nn::turbo_quant::TurboKvCache;

#[test]
fn switches() {
    let cpu = Device::Cpu;

    // environment variables
    for (k, v) in [
        ("CANDLE_FLASH_ATTN", "off"),
        ("CANDLE_FLASH_ATTN_VERSION", "fa1"),
        ("CANDLE_FLASH_ATTN_SPLIT_KV", "8"),
        ("CANDLE_FLASH_ATTN_FP8_KV", "0"),
        ("CANDLE_FLASH_ATTN_BACKENDS", "rocm,vulkan"),
        ("CANDLE_TURBO_QUANT_K_FORMAT", "nvfp4"),
        ("CANDLE_TURBO_QUANT_V_BITS", "3"),
        ("CANDLE_TURBO_QUANT_ROTATION", "dense"),
        ("CANDLE_TURBO_QUANT_FUSED", "off"),
    ] {
        std::env::set_var(k, v);
    }
    accel::reset_config();
    let c = accel::config();
    assert!(!c.flash_attn.enabled);
    assert_eq!(c.flash_attn.version, Some(FlashAttnVersion::V1));
    assert_eq!(c.flash_attn.split_kv, SplitKv::Fixed(8));
    assert!(!c.flash_attn.fp8_kv);
    assert!(
        c.flash_attn.backends.rocm && c.flash_attn.backends.vulkan && !c.flash_attn.backends.cuda
    );
    assert_eq!(c.turbo_quant.key_format, Some(QuantFormat::Nvfp4));
    assert_eq!(c.turbo_quant.value_bits, 3);
    assert_eq!(c.turbo_quant.rotation, RotationKind::Dense);
    assert!(!c.turbo_quant.fused_attention);
    let report = accel::describe(&cpu);
    assert!(!report.flash_attn_active);
    assert_eq!(report.key_format, QuantFormat::Nvfp4);
    assert_eq!(report.value_format, QuantFormat::TurboMse { bits: 3 });
    let cache = TurboKvCache::new(64, 16, &cpu).unwrap();
    assert_eq!(
        cache.formats(),
        Some((QuantFormat::Nvfp4, QuantFormat::TurboMse { bits: 3 }))
    );

    // with the fused path off the results are the unfused ones
    let q = Tensor::randn(0f32, 1., (1, 5, 2, 32), &cpu).unwrap();
    let k = Tensor::randn(0f32, 1., (1, 5, 2, 32), &cpu).unwrap();
    let v = Tensor::randn(0f32, 1., (1, 5, 2, 32), &cpu).unwrap();
    let a = flash_attn(&q, &k, &v, 0.2, true).unwrap();
    let b = flash_attn_unfused(&q, &k, &v, &FlashAttnConfig::new(0.2).causal()).unwrap();
    let diff = (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(diff, 0.0);

    // programmatic changes win over the environment
    accel::update_config(|c| {
        c.flash_attn.enabled = true;
        c.flash_attn.backends = accel::BackendSwitches::all();
        c.turbo_quant.enabled = false;
    });
    assert!(accel::describe(&cpu).flash_attn_active);
    let cache = TurboKvCache::new(64, 16, &cpu).unwrap();
    assert!(!cache.is_quantized());

    // QJL on: TurboQuant keys use the prod format
    accel::update_config(|c| {
        c.turbo_quant.enabled = true;
        c.turbo_quant.key_format = None;
        c.turbo_quant.qjl = true;
    });
    let cache = TurboKvCache::new(64, 16, &cpu).unwrap();
    assert_eq!(
        cache.formats().unwrap().0,
        QuantFormat::TurboProd { bits: 4 }
    );

    // back to the defaults
    for k in [
        "CANDLE_FLASH_ATTN",
        "CANDLE_FLASH_ATTN_VERSION",
        "CANDLE_FLASH_ATTN_SPLIT_KV",
        "CANDLE_FLASH_ATTN_FP8_KV",
        "CANDLE_FLASH_ATTN_BACKENDS",
        "CANDLE_TURBO_QUANT_K_FORMAT",
        "CANDLE_TURBO_QUANT_V_BITS",
        "CANDLE_TURBO_QUANT_ROTATION",
        "CANDLE_TURBO_QUANT_FUSED",
    ] {
        std::env::remove_var(k);
    }
    accel::reset_config();
    assert_eq!(accel::config(), accel::AccelConfig::default());
    println!("{}", accel::describe(&cpu));
}
