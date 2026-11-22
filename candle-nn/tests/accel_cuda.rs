//! Fused attention / TurboQuant kernels on a CUDA GPU: the FP8 fast path of `tq_attn`, exact
//! F16 / BF16 K/V with F32 queries, the paged (block table) kernels and the split-KV combine.
//! Skipped when no CUDA device is available. The configuration is process global, so the tests
//! take a lock.
//!
//! Split heuristic benchmark (prints a table):
//! `cargo test -p candle-nn --features cuda --test accel_cuda -- --ignored --nocapture`
#![cfg(feature = "cuda")]

use candle::accel::{self, QuantFormat, RotationKind, SplitKv};
use candle::{DType, Device, Tensor};
use candle_nn::flash_attn::{flash_attn_bhsd, flash_attn_blocks, FlashAttnConfig, KvBlockRef};
use candle_nn::turbo_quant::{
    attention_packed, attention_packed_blocks, TurboKvCache, TurboQuantizer,
};
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

fn gpu() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping CUDA accel tests: {e}");
            None
        }
    }
}

fn randn(shape: &[usize], dev: &Device) -> Tensor {
    Tensor::randn(0f32, 1f32, shape, dev).unwrap()
}

/// max |a - b| / max |b|
fn rel(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap();
    let b = b.to_dtype(DType::F32).unwrap();
    let d = (&a - &b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let n = b
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    d / n.max(1e-30)
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// `t` in the K/V dtype `kvd` (uses on-GPU software OCP E4M3FN cast on `sm < 89`, no CPU roundtrip).
fn to_kv(t: &Tensor, kvd: DType) -> Tensor {
    t.to_dtype(kvd).unwrap()
}

fn strict(scale: f32) -> FlashAttnConfig {
    FlashAttnConfig::new(scale).causal().fused(true).strict(true)
}

fn pack(q: &TurboQuantizer, x: &Tensor) -> Tensor {
    let (b, h, t, d) = x.dims4().unwrap();
    q.quantize_rows(&x.reshape((b * h * t, d)).unwrap())
        .unwrap()
        .reshape((b, h, t, q.row_words()))
        .unwrap()
}

fn set_fp8_fast(on: bool) {
    accel::update_config(|c| c.turbo_quant.fp8_fast_path = on);
}

fn set_gqa(on: bool) {
    accel::update_config(|c| c.flash_attn.gqa_decode = on);
}

/// The GQA-packed row kernels (query heads of a kv head in one workgroup) give the same bits
/// as the plain row kernels, dense and paged, for every K/V dtype.
#[test]
fn gqa_packed_rows_are_bit_exact() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    for (hq, hkv, d, lk) in [(16, 4, 256, 1500), (16, 8, 128, 700), (32, 2, 64, 300), (12, 4, 128, 200)] {
        for (qd, kvd) in [
            (DType::F32, DType::F32),
            (DType::F32, DType::F16),
            (DType::F32, DType::BF16),
            (DType::F32, DType::F8E4M3),
            (DType::BF16, DType::BF16),
            (DType::F16, DType::F8E4M3),
        ] {
            let (cap, n) = (160usize, (lk as usize).div_ceil(160));
            let last = lk - (n - 1) * cap;
            let blocks = make_blocks(n, cap, last, hkv, d, &|t: &Tensor| to_kv(t, kvd), &dev);
            let (k, v) = cat_valid(&blocks);
            for lq in [1usize, 3] {
                let q = randn(&[1, hq, lq, d], &dev).to_dtype(qd).unwrap();
                for split in [SplitKv::Off, SplitKv::Fixed(5)] {
                    let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(split);
                    set_gqa(false);
                    let a = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                    let ap = flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap();
                    set_gqa(true);
                    let b = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                    let bp = flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap();
                    let what = format!("{qd:?}/{kvd:?} {hq}/{hkv}/{d} lq {lq} {split:?}");
                    assert_eq!(bits(&a), bits(&b), "dense {what}");
                    assert_eq!(bits(&ap), bits(&bp), "paged {what}");
                }
            }
        }
    }
    set_gqa(true);
}

/// The FP8 specialization of `tq_attn` gives the same bits as the generic kernel (dense and
/// paged, with and without key splits).
#[test]
fn tq_fp8_fast_path_is_bit_exact() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    for (hq, hkv, d, lk) in [(16, 4, 256, 1500), (16, 8, 128, 777), (4, 4, 64, 3)] {
        let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 42, &dev).unwrap();
        let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 44, &dev).unwrap();
        let (k, v) = (randn(&[1, hkv, lk, d], &dev), randn(&[1, hkv, lk, d], &dev));
        let (kr, vr) = (pack(&kq, &k), pack(&vq, &v));
        for lq in [1usize, 3, 9] {
            let q = randn(&[1, hq, lq, d], &dev);
            for split in [SplitKv::Off, SplitKv::Auto, SplitKv::Fixed(5)] {
                let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(split);
                let blocks = [KvBlockRef {
                    k: &kr,
                    v: &vr,
                    rows: lk,
                }];
                set_fp8_fast(false);
                let slow = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                let slow_p = attention_packed_blocks(&q, &blocks, &kq, &vq, &cfg).unwrap();
                set_fp8_fast(true);
                let fast = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                let fast_p = attention_packed_blocks(&q, &blocks, &kq, &vq, &cfg).unwrap();
                assert_eq!(bits(&fast), bits(&slow), "dense {hq}/{hkv}/{d} lk {lk} lq {lq}");
                assert_eq!(bits(&fast_p), bits(&slow_p), "paged {hq}/{hkv}/{d} lq {lq}");
            }
        }
    }
    set_fp8_fast(true);
}

/// The GQA-packed `tq_attng*` kernels give the bits of `tq_attn*` (dense and paged, FP8 fast
/// path and generic formats).
#[test]
fn tq_gqa_packed_is_bit_exact() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    for (hq, hkv, d, lk) in [(16, 4, 256, 1300), (16, 8, 128, 500), (32, 4, 64, 260)] {
        for (kf, vf, rot) in [
            (QuantFormat::Fp8, QuantFormat::Fp8, RotationKind::None),
            (
                QuantFormat::TurboMse { bits: 4 },
                QuantFormat::TurboMse { bits: 3 },
                RotationKind::Hadamard,
            ),
            (QuantFormat::Nvfp4, QuantFormat::Mxfp4, RotationKind::None),
        ] {
            let kq = TurboQuantizer::new(kf, d, rot, 5, &dev).unwrap();
            let vq = TurboQuantizer::new(vf, d, rot, 6, &dev).unwrap();
            let (cap, n) = (128usize, (lk as usize).div_ceil(128));
            let last = lk - (n - 1) * cap;
            let blocks: Vec<(Tensor, Tensor, usize)> = (0..n)
                .map(|i| {
                    let rows = if i + 1 == n { last } else { cap };
                    let k = pack(&kq, &randn(&[1, hkv, cap, d], &dev));
                    let v = pack(&vq, &randn(&[1, hkv, cap, d], &dev));
                    (k, v, rows)
                })
                .collect();
            let (kr, vr) = cat_valid(&blocks);
            for lq in [1usize, 3] {
                let q = randn(&[1, hq, lq, d], &dev);
                for split in [SplitKv::Off, SplitKv::Fixed(4)] {
                    let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(split);
                    let run = |gqa: bool| {
                        set_gqa(gqa);
                        (
                            bits(&attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap()),
                            bits(&attention_packed_blocks(&q, &refs(&blocks), &kq, &vq, &cfg).unwrap()),
                        )
                    };
                    let (a, ap) = run(false);
                    let (b, bp) = run(true);
                    assert_eq!(a, b, "dense {kf} {hq}/{hkv}/{d} lq {lq} {split:?}");
                    assert_eq!(ap, bp, "paged {kf} {hq}/{hkv}/{d} lq {lq} {split:?}");
                }
            }
        }
    }
    set_gqa(true);
}

/// Exact F16 / BF16 K/V with F32 queries run without a conversion and match the F32 kernels on
/// the widened K/V bit for bit (decode row kernel and prefill tile kernel); the error against the
/// unrounded F32 K/V is the storage rounding.
#[test]
fn half_kv_with_f32_queries() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    for (hq, hkv, d, lk) in [(16, 4, 256, 1100), (16, 8, 128, 900)] {
        let (k, v) = (randn(&[1, hkv, lk, d], &dev), randn(&[1, hkv, lk, d], &dev));
        for kvd in [DType::F16, DType::BF16] {
            let (kh, vh) = (k.to_dtype(kvd).unwrap(), v.to_dtype(kvd).unwrap());
            let (kw, vw) = (kh.to_dtype(DType::F32).unwrap(), vh.to_dtype(DType::F32).unwrap());
            for lq in [1usize, 4, 37] {
                let q = randn(&[1, hq, lq, d], &dev);
                let cfg = strict(1.0 / (d as f32).sqrt());
                let got = flash_attn_bhsd(&q, &kh, &vh, &cfg).unwrap();
                assert_eq!(got.dtype(), DType::F32);
                let widened = flash_attn_bhsd(&q, &kw, &vw, &cfg).unwrap();
                assert_eq!(bits(&got), bits(&widened), "{kvd:?} {hq}/{hkv}/{d} lq {lq}");
                let exact = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                let e = rel(&got, &exact);
                println!("{kvd:?} kv, f32 q, hq {hq} hkv {hkv} d {d} lq {lq}: max rel err vs f32 kv {e:.3e}");
                let tol = if kvd == DType::F16 { 2e-3 } else { 1.6e-2 };
                assert!(e < tol, "{kvd:?}: {e}");
            }
        }
    }
}

/// `n` blocks of `cap` rows (the last one partial), with junk in the spare capacity, as views of
/// larger buffers (head stride = capacity).
fn make_blocks(
    n: usize,
    cap: usize,
    last: usize,
    hkv: usize,
    w: usize,
    f: &dyn Fn(&Tensor) -> Tensor,
    dev: &Device,
) -> Vec<(Tensor, Tensor, usize)> {
    (0..n)
        .map(|i| {
            let rows = if i + 1 == n { last } else { cap };
            let k = f(&randn(&[1, hkv, cap, w], dev));
            let v = f(&randn(&[1, hkv, cap, w], dev));
            (k, v, rows)
        })
        .collect()
}

fn refs(blocks: &[(Tensor, Tensor, usize)]) -> Vec<KvBlockRef<'_>> {
    blocks
        .iter()
        .map(|(k, v, rows)| KvBlockRef { k, v, rows: *rows })
        .collect()
}

fn cat_valid(blocks: &[(Tensor, Tensor, usize)]) -> (Tensor, Tensor) {
    let ks: Vec<Tensor> = blocks
        .iter()
        .map(|(k, _, r)| k.narrow(2, 0, *r).unwrap())
        .collect();
    let vs: Vec<Tensor> = blocks
        .iter()
        .map(|(_, v, r)| v.narrow(2, 0, *r).unwrap())
        .collect();
    (Tensor::cat(&ks, 2).unwrap(), Tensor::cat(&vs, 2).unwrap())
}

/// Paged dense attention equals the attention over the concatenated valid rows (1..5 blocks,
/// partial last block, F32 / F16 / FP8 K/V, splits on and off).
#[test]
fn paged_dense_matches_concatenated() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let mut worst = 0f32;
    for (hq, hkv, d) in [(16, 4, 256), (16, 8, 128), (8, 8, 64)] {
        for kvd in [DType::F32, DType::F16, DType::F8E4M3] {
            let conv = |t: &Tensor| to_kv(t, kvd);
            for n in 1..=5 {
                let blocks = make_blocks(n, 160, 37 + 11 * n, hkv, d, &conv, &dev);
                let (k, v) = cat_valid(&blocks);
                for lq in [1usize, 2, 4, 6] {
                    let q = randn(&[1, hq, lq, d], &dev);
                    for split in [SplitKv::Off, SplitKv::Auto] {
                        let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(split);
                        let got = flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap();
                        let want = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                        let e = rel(&got, &want);
                        worst = worst.max(e);
                        assert!(e <= 1e-6, "{kvd:?} {hq}/{hkv}/{d} blocks {n} lq {lq}: {e}");
                    }
                }
            }
        }
    }
    println!("paged dense vs concatenated: worst max rel err {worst:.3e}");
}

/// Paged compressed attention equals `attention_packed` over the concatenated rows.
#[test]
fn paged_packed_matches_concatenated() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let mut worst = 0f32;
    for (hq, hkv, d) in [(16, 4, 256), (16, 8, 128)] {
        for (kf, vf, rot) in [
            (QuantFormat::Fp8, QuantFormat::Fp8, RotationKind::None),
            (
                QuantFormat::TurboMse { bits: 4 },
                QuantFormat::TurboMse { bits: 4 },
                RotationKind::Hadamard,
            ),
        ] {
            let kq = TurboQuantizer::new(kf, d, rot, 7, &dev).unwrap();
            let vq = TurboQuantizer::new(vf, d, rot, 9, &dev).unwrap();
            for n in 1..=5 {
                let cap = 96;
                let last = 1 + 17 * n;
                let blocks: Vec<(Tensor, Tensor, usize)> = (0..n)
                    .map(|i| {
                        let rows = if i + 1 == n { last } else { cap };
                        let k = pack(&kq, &randn(&[1, hkv, cap, d], &dev));
                        let v = pack(&vq, &randn(&[1, hkv, cap, d], &dev));
                        (k, v, rows)
                    })
                    .collect();
                let (kr, vr) = cat_valid(&blocks);
                let lk = kr.dim(2).unwrap();
                for lq in [1usize, 4, 13] {
                    let q = randn(&[1, hq, lq, d], &dev);
                    for split in [SplitKv::Off, SplitKv::Auto] {
                        let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(split);
                        let got =
                            attention_packed_blocks(&q, &refs(&blocks), &kq, &vq, &cfg).unwrap();
                        let want = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                        let e = rel(&got, &want);
                        worst = worst.max(e);
                        assert!(e <= 1e-6, "{kf} {hq}/{hkv}/{d} blocks {n} lq {lq}: {e}");
                    }
                }
            }
        }
    }
    println!("paged packed vs concatenated: worst max rel err {worst:.3e}");
}

/// Split-KV results (per-row combine weights) agree with the unsplit kernels up to the
/// summation order of the splits (the combine kernels compute the same float operations as
/// before the per-row weights, see `fa_combine_matches_previous_formula` in
/// candle-core/tests/accel_kernels_host.rs).
#[test]
fn split_combine_matches_unsplit() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let mut worst = 0f32;
    let (hq, hkv, d, lk) = (16, 4, 256, 2048);
    let (k, v) = (randn(&[1, hkv, lk, d], &dev), randn(&[1, hkv, lk, d], &dev));
    let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 1, &dev).unwrap();
    let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &dev).unwrap();
    let (kr, vr) = (pack(&kq, &k), pack(&vq, &v));
    let q = randn(&[1, hq, 1, d], &dev);
    let base = strict(1.0 / (d as f32).sqrt());
    let fa0 = flash_attn_bhsd(&q, &k, &v, &base.clone().split_kv(SplitKv::Off)).unwrap();
    let tq0 = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &base.clone().split_kv(SplitKv::Off))
        .unwrap();
    for n in [2usize, 3, 7, 16, 33, 64] {
        let cfg = base.clone().split_kv(SplitKv::Fixed(n));
        let fa = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
        let tq = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
        let (a, b) = (rel(&fa, &fa0), rel(&tq, &tq0));
        worst = worst.max(a).max(b);
        assert!(a <= 4e-6 && b <= 4e-6, "splits {n}: fa {a} tq {b}");
    }
    println!("split combine vs unsplit: worst max rel err {worst:.3e}");
}

/// A compressed cache with a planned capacity allocates once and attends like a growing one.
#[test]
fn turbo_cache_with_capacity() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let mut o = accel::config().turbo_quant;
    o.enabled = true;
    o.key_format = Some(QuantFormat::Fp8);
    o.value_format = Some(QuantFormat::Fp8);
    o.rotation = RotationKind::None;
    let (hq, hkv, d) = (16, 4, 256);
    let mut a = TurboKvCache::with_capacity(&o, d, 300, &dev).unwrap();
    let mut b = TurboKvCache::with_options(&o, d, 16, &dev).unwrap();
    let k = randn(&[1, hkv, 250, d], &dev);
    let v = randn(&[1, hkv, 250, d], &dev);
    a.append(&k, &v).unwrap();
    b.append(&k, &v).unwrap();
    let bytes = a.storage_bytes();
    for _ in 0..40 {
        let (k1, v1) = (randn(&[1, hkv, 1, d], &dev), randn(&[1, hkv, 1, d], &dev));
        a.append(&k1, &v1).unwrap();
        b.append(&k1, &v1).unwrap();
    }
    assert_eq!(a.storage_bytes(), bytes, "no growth within the capacity");
    let q = randn(&[1, hq, 1, d], &dev);
    let cfg = strict(1.0 / 16.0);
    let (x, y) = (a.attention(&q, &cfg).unwrap(), b.attention(&q, &cfg).unwrap());
    assert_eq!(bits(&x), bits(&y));
}

/// CUDA OCP E4M3FN software cast kernels (`sm < 89` / fallback): verifies bit-exact parity with
/// CPU `float8::F8E4M3` across all 256 codes, RNE midpoints, subnormals, 448 saturation, ±Inf,
/// NaN, strided layouts, integer/half pairs, and `flash_attn_fp8` (with & without Hadamard).
#[test]
fn e4m3_cast_kernels_match_cpu_cuda() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };

    // 1. All 256 E4M3 codes -> F32 (bit-exact including -0.0 and NaN)
    let raw: Vec<u8> = (0u8..=255u8).collect();
    let codes_cpu = Tensor::from_raw_buffer(&raw, DType::F8E4M3, &[256], &Device::Cpu).unwrap();
    let want_f32 = codes_cpu.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let codes_gpu = codes_cpu.to_device(&dev).unwrap();
    let got_f32 = codes_gpu
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (&g, &w)) in got_f32.iter().zip(want_f32.iter()).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "code {i:#04x}: expected NaN, got {g}");
        } else {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "code {i:#04x}: got {g} ({:#010x}) != want {w} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    // 2. F32 -> F8E4M3 across all codes, midpoints (ties-to-even), subnormals, saturation, Inf, NaN
    let mut xs: Vec<f32> = want_f32.iter().copied().filter(|x| x.is_finite()).collect();
    for c in 0..0x7eu8 {
        let m = 0.5 * (want_f32[c as usize] + want_f32[c as usize + 1]);
        xs.push(m);
        xs.push(-m);
        xs.push(m * (1.0 + 1e-6));
        xs.push(m * (1.0 - 1e-6));
    }
    xs.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        448.0,
        449.0,
        460.0,
        464.0,
        465.0,
        1e6,
        -1e6,
        0.0009765625, // 2^-10: half of smallest subnormal
        0.0009766,
        0.0029296875, // 1.5 * 2^-9
        1e-30,
        -1e-30,
    ]);
    let r = Tensor::randn(0f32, 100.0, xs.len(), &Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    xs.extend(r);
    let x_cpu = Tensor::from_vec(xs.clone(), xs.len(), &Device::Cpu).unwrap();
    let cpu_f8 = x_cpu
        .to_dtype(DType::F8E4M3)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let gpu_f8 = x_cpu
        .to_device(&dev)
        .unwrap()
        .to_dtype(DType::F8E4M3)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (&g, &w)) in gpu_f8.iter().zip(cpu_f8.iter()).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "f32->f8 idx {i} ({}): expected NaN, got {g}", xs[i]);
        } else {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "f32->f8 idx {i} ({}): got {g} != want {w}",
                xs[i]
            );
        }
    }

    // 3. Strided layout + F16/BF16/U32/I64 <-> F8E4M3 pairs
    let mat = Tensor::randn(0f32, 4.0, (16, 24), &dev).unwrap();
    let strided = mat.t().unwrap().narrow(0, 3, 10).unwrap();
    let s_gpu = strided
        .to_dtype(DType::F8E4M3)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let s_cpu = strided
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F8E4M3)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    assert_eq!(bits(&s_gpu), bits(&s_cpu), "strided f32->f8->f32 mismatch");
    for dt in [DType::F16, DType::BF16, DType::F64] {
        let a = mat.to_dtype(dt).unwrap();
        let g = a
            .to_dtype(DType::F8E4M3)
            .unwrap()
            .to_dtype(dt)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let w = a
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap()
            .to_dtype(dt)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(bits(&g), bits(&w), "cuda {dt:?}<->f8 mismatch");
    }

    // 4. `flash_attn_fp8` on CUDA (BSHD layout: exercises native GPU F8E4M3 cast + single-sync amax + Hadamard)
    let (q, k, v) = (
        randn(&[1, 16, 8, 64], &dev).to_dtype(DType::BF16).unwrap(),
        randn(&[1, 128, 4, 64], &dev).to_dtype(DType::BF16).unwrap(),
        randn(&[1, 128, 4, 64], &dev).to_dtype(DType::BF16).unwrap(),
    );
    let cfg = strict(1.0 / 8.0);
    let ref_out = candle_nn::flash_attn::flash_attn_with_config(&q, &k, &v, &cfg).unwrap();
    for had in [false, true] {
        accel::update_config(|c| c.flash_attn.hadamard = had);
        let fp8_out = candle_nn::flash_attn::flash_attn_fp8(&q, &k, &v, &cfg).unwrap();
        assert_eq!(fp8_out.dtype(), DType::BF16);
        let e = rel(&fp8_out, &ref_out);
        assert!(e < 0.15, "flash_attn_fp8 hadamard={had} rel err {e}");
    }
    accel::update_config(|c| c.flash_attn.hadamard = false);
}

/// Paged tile prefill (`fa2p_*`) on CUDA with non-BC-aligned pages (`[48, 40, 17]`), partial last
/// page, and sliding window (`window_left > 0`) matches concatenated dense tile prefill (`Q4-F03`).
#[test]
fn paged_tile_prefill_matches_concatenated_cuda() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    for d in [64usize, 128, 256] {
        for kvd in [DType::F32, DType::F16, DType::BF16, DType::F8E4M3] {
            let (hq, hkv, lq) = (8usize, 4usize, 16usize);
            let qd = if kvd == DType::F8E4M3 { DType::F16 } else { kvd };
            let page_rows = [48usize, 40, 17];
            let blocks: Vec<(Tensor, Tensor, usize)> = page_rows
                .iter()
                .map(|&r| {
                    let cap = 48usize;
                    let k = to_kv(&randn(&[1, hkv, cap, d], &dev), kvd);
                    let v = to_kv(&randn(&[1, hkv, cap, d], &dev), kvd);
                    (k, v, r)
                })
                .collect();
            let (k_cat, v_cat) = cat_valid(&blocks);
            let q = randn(&[1, hq, lq, d], &dev).to_dtype(qd).unwrap();
            for win in [-1i32, 40] {
                let mut cfg = strict(1.0 / (d as f32).sqrt()).split_kv(SplitKv::Off);
                if win >= 0 {
                    cfg = cfg.window(Some(win as usize), Some(0));
                }
                let got = flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap();
                let want = flash_attn_bhsd(&q, &k_cat, &v_cat, &cfg).unwrap();
                let e = rel(&got, &want);
                assert!(
                    e <= 1e-6,
                    "cuda fa2p d={d} kvd={kvd:?} win={win}: rel err {e}"
                );
            }
        }
    }
}

/// Real-GPU CUDA coverage for:
/// - NVFP4/NVFP4 (`_nvfp4`) & NVFP4/FP8 (`_nvfp4_fp8`) dense+paged fast-vs-generic bit-exactness
/// - CUDA `tq_fwht` (`d∈{64,128,256}`) vs dense `rotation_matrix` (`rel <= 1e-6`)
/// - CUDA `tq_quant_prod_fused` (`TurboProd` `d∈{64,128}`)
/// - CUDA `OffloadedTurboKvEngine` + `PinnedKvSlab` (both `<4 MiB` sync and `>=4 MiB` pinned async DMA + `kv_len < max_kv_len` compact view + 2-layer `run_layers`)
/// - CUDA `GuardedQuantLmHead` (adversarial REST spike, exact tie row, tail tile `V=97`, `M=1,4,12`, `download_ids_and_mask()`).
#[test]
fn tq_nvfp4_fast_fwht_prod_offload_and_quant_lm_head_cuda() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };

    // 1. NVFP4/NVFP4 and NVFP4/FP8 dense + paged fast-vs-generic bit-exactness
    for vf in [QuantFormat::Nvfp4, QuantFormat::Fp8] {
        let (hq, hkv, d, lk) = (8usize, 2usize, 128usize, 256usize);
        let kq = TurboQuantizer::new(QuantFormat::Nvfp4, d, RotationKind::Hadamard, 11, &dev).unwrap();
        let vq = TurboQuantizer::new(vf, d, RotationKind::Hadamard, 13, &dev).unwrap();
        let (k, v) = (randn(&[1, hkv, lk, d], &dev), randn(&[1, hkv, lk, d], &dev));
        let (kr, vr) = (pack(&kq, &k), pack(&vq, &v));
        let q = randn(&[1, hq, 1, d], &dev);
        let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(SplitKv::Off);
        let blocks = [KvBlockRef { k: &kr, v: &vr, rows: lk }];
        set_fp8_fast(false);
        let slow = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
        let slow_p = attention_packed_blocks(&q, &blocks, &kq, &vq, &cfg).unwrap();
        set_fp8_fast(true);
        let fast = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
        let fast_p = attention_packed_blocks(&q, &blocks, &kq, &vq, &cfg).unwrap();
        assert_eq!(bits(&fast), bits(&slow), "cuda nvfp4/{vf} dense fast vs generic");
        assert_eq!(bits(&fast_p), bits(&slow_p), "cuda nvfp4/{vf} paged fast vs generic");
    }

    // 2. CUDA FWHT vs dense rotation_matrix (Step 1 gate: rel <= 1e-6 for d∈{64,128,256})
    for d in [64usize, 128, 256] {
        let qz = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::Hadamard, 77, &dev).unwrap();
        let pi = accel::turbo::rotation_matrix(d, RotationKind::Hadamard, 77).unwrap();
        let pi_t = Tensor::from_vec(pi, (d, d), &dev).unwrap().t().unwrap().contiguous().unwrap();
        let x = randn(&[17, d], &dev);
        let y_fwht = qz.rotate(&x).unwrap();
        let y_dense = x.matmul(&pi_t).unwrap();
        let e = rel(&y_fwht, &y_dense);
        assert!(e <= 1e-6, "cuda FWHT d={d} vs dense rel err {e}");
        let rt = qz.unrotate(&y_fwht).unwrap();
        let e_rt = rel(&rt, &x);
        assert!(e_rt <= 1e-6, "cuda FWHT roundtrip d={d} rel err {e_rt}");
    }

    // 3. CUDA single-pass `tq_quant_prod_fused`
    for d in [64usize, 128] {
        let qp = TurboQuantizer::new(
            QuantFormat::TurboProd { bits: 4 },
            d,
            RotationKind::Hadamard,
            19,
            &dev,
        )
        .unwrap();
        let x = randn(&[32, d], &dev);
        let packed = qp.quantize_rows(&x).unwrap();
        let dq_gpu = qp.dequantize_rows(&packed).unwrap().to_device(&Device::Cpu).unwrap();
        let qp_cpu = TurboQuantizer::new(
            QuantFormat::TurboProd { bits: 4 },
            d,
            RotationKind::Hadamard,
            19,
            &Device::Cpu,
        )
        .unwrap();
        let p_cpu = qp_cpu.quantize_rows(&x.to_device(&Device::Cpu).unwrap()).unwrap();
        let dq_cpu = qp_cpu.dequantize_rows(&p_cpu).unwrap();
        let e = rel(&dq_gpu, &dq_cpu);
        assert!(e < 2e-2, "cuda tq_quant_prod_fused d={d} vs cpu rel err {e}");
    }

    // 4. CUDA `OffloadedTurboKvEngine`: both small sync and >=4 MiB pinned async DMA + compact view
    {
        use candle_nn::kv_offload::{OffloadedTurboKvEngine, PinnedKvSlab};
        let (hq, hkv, d) = (8usize, 4usize, 128usize);
        let kq = TurboQuantizer::new(QuantFormat::Nvfp4, d, RotationKind::Hadamard, 23, &dev).unwrap();
        let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::Hadamard, 29, &dev).unwrap();
        let cfg = strict(1.0 / (d as f32).sqrt()).split_kv(SplitKv::Off);

        // 4a. Small slab (sync direct upload) + compact kv_len < max_kv_len
        let (max_len, kv_len) = (256usize, 180usize);
        let (k0, v0) = (randn(&[1, hkv, kv_len, d], &dev), randn(&[1, hkv, kv_len, d], &dev));
        let (kr0, vr0) = (pack(&kq, &k0), pack(&vq, &v0));
        let slab_small = PinnedKvSlab::from_packed_tensors(&kr0, &vr0, &dev).unwrap();
        let mut eng_small =
            OffloadedTurboKvEngine::new(&dev, 1, hkv, max_len, kq.row_words(), vq.row_words())
                .unwrap();
        let q = randn(&[1, hq, 1, d], &dev);
        eng_small.prefetch_slab(0, &slab_small).unwrap();
        assert!(!eng_small.slot_used_async_dma(0), "small slab should take sync path");
        let out_small = eng_small.attend_slot(0, &q, &kq, &vq, &cfg).unwrap();
        let ref_small = attention_packed(&q, &kr0, &vr0, kv_len, &kq, &vq, &cfg).unwrap();
        assert_eq!(bits(&out_small), bits(&ref_small), "cuda offload small sync parity");

        // 4b. Large slab (> 4 MiB total so PinnedKvSlab allocates pinned host memory & runs async CopyStream DMA!)
        // hkv=4, kv_len=6000: k ~ 4*6000*20*4 = 1.92 MiB, v ~ 4*6000*34*4 = 3.26 MiB -> >5.1 MiB!
        let (big_max, big_len) = (8192usize, 6000usize);
        let (kb, vb) = (randn(&[1, hkv, big_len, d], &dev), randn(&[1, hkv, big_len, d], &dev));
        let (krb, vrb) = (pack(&kq, &kb), pack(&vq, &vb));
        let slab_big0 = PinnedKvSlab::from_packed_tensors(&krb, &vrb, &dev).unwrap();
        let slab_big1 = PinnedKvSlab::from_packed_tensors(&krb, &vrb, &dev).unwrap();
        assert!(slab_big0.is_pinned(), "large >4 MiB slab must use pinned host memory");
        let mut eng_big =
            OffloadedTurboKvEngine::new(&dev, 1, hkv, big_max, kq.row_words(), vq.row_words())
                .unwrap();
        let slabs = [slab_big0, slab_big1];
        let outs = eng_big
            .run_layers(&slabs, &kq, &vq, &cfg, |_| Ok(q.clone()))
            .unwrap();
        assert!(eng_big.slot_used_async_dma(0), "large slab 0 must use async DMA");
        assert!(eng_big.slot_used_async_dma(1), "large slab 1 must use async DMA");
        let ref_big = attention_packed(&q, &krb, &vrb, big_len, &kq, &vq, &cfg).unwrap();
        assert_eq!(bits(&outs[0]), bits(&ref_big), "cuda offload async DMA layer 0 parity");
        assert_eq!(bits(&outs[1]), bits(&ref_big), "cuda offload async DMA layer 1 parity");
    }

    // 5. CUDA `GuardedQuantLmHead` with adversarial REST spike, exact ties, tail tile V=97, M=1,4,12
    {
        use candle::quantized::{GgmlDType, QTensor};
        use candle_nn::trimmed_head::{GuardedQuantLmHead, LmWeight};
        use std::sync::Arc;
        let (v_core, v_rest, k_dim) = (97usize, 64usize, 256usize);
        let mut w_core_v: Vec<f32> = randn(&[v_core, k_dim], &Device::Cpu)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Force an exact tie between vocab index 11 and 73 so min-index tie-breaking is tested on GPU!
        for j in 0..k_dim {
            w_core_v[73 * k_dim + j] = w_core_v[11 * k_dim + j];
        }
        let w_core_cpu = Tensor::from_vec(w_core_v, (v_core, k_dim), &Device::Cpu).unwrap();
        let mut w_rest_v: Vec<f32> = randn(&[v_rest, k_dim], &Device::Cpu)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Inject an adversarial spike on REST row 7 so row 0 MUST be rescued.
        for j in 0..k_dim {
            w_rest_v[7 * k_dim + j] = if j % 2 == 0 { 6.0 } else { -6.0 };
        }
        let w_rest_cpu = Tensor::from_vec(w_rest_v, (v_rest, k_dim), &Device::Cpu).unwrap();

        let q_core = Arc::new(QTensor::quantize_onto(&w_core_cpu, GgmlDType::Q8_0, &dev).unwrap());
        let q_rest = Arc::new(QTensor::quantize_onto(&w_rest_cpu, GgmlDType::Q4K, &dev).unwrap());
        let core_ids = Tensor::arange(0u32, v_core as u32, &dev).unwrap();
        let rest_ids = Tensor::arange(v_core as u32, (v_core + v_rest) as u32, &dev).unwrap();
        let head = GuardedQuantLmHead::from_split_weights(
            LmWeight::Quantized(q_core.clone()),
            None,
            core_ids,
            Some((LmWeight::Quantized(q_rest.clone()), None, rest_ids)),
            1e-2,
            &dev,
        )
        .unwrap();

        for m in [1usize, 4, 12] {
            let mut h_v: Vec<f32> = randn(&[m, k_dim], &Device::Cpu)
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            // Row 0 aligns strongly with REST spike row 7
            for j in 0..k_dim {
                h_v[j] = if j % 2 == 0 { 2.0 } else { -2.0 };
            }
            let h = Tensor::from_vec(h_v, (m, k_dim), &dev).unwrap();
            let (got_ids_t, rescued) = head.forward_greedy(&h).unwrap();
            let got_ids = got_ids_t.to_vec1::<u32>().unwrap();
            assert!(rescued >= 1, "m={m}: adversarial REST spike must trigger rescue");
            let full_core = q_core.forward_exact(&h).unwrap();
            let full_rest = q_rest.forward_exact(&h).unwrap();
            let full = Tensor::cat(&[&full_core, &full_rest], 1)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for row in 0..m {
                let mut best_i = 0u32;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in full[row].iter().enumerate() {
                    if v > best_v || (v == best_v && (i as u32) < best_i) {
                        best_v = v;
                        best_i = i as u32;
                    }
                }
                assert_eq!(got_ids[row], best_i, "cuda GuardedQuantLmHead m={m} row={row}");
            }
            // Verify single-D2H `download_ids_and_mask` matches individual transfers
            let raw_out = candle::lm_head::fused_quant_argmax_cs(
                &h,
                &q_core,
                None,
                None,
                head.cs_guard.as_ref(),
            )
            .unwrap();
            let (d_ids, d_mask) = raw_out.download_ids_and_mask().unwrap();
            assert_eq!(d_ids, raw_out.ids.to_vec1::<u32>().unwrap());
            assert_eq!(
                d_mask,
                raw_out
                    .rescue_mask
                    .as_ref()
                    .map(|t| t.to_vec1::<u8>().unwrap())
            );
        }
    }
}

/// Multi-thread stress test on a shared `CudaDevice`: verifies RAII workspace leases
/// (`fa_split_kv` & `lm_head_parts`) prevent cross-thread aliasing or UAF (`P0-1` / `P1-F02`).
#[test]
fn cuda_multithread_workspace_lease_safety() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };

    let (hq, hkv, d, lk) = (8usize, 4usize, 64usize, 512usize);
    let q = randn(&[1, hq, 1, d], &dev);
    let k = randn(&[1, hkv, lk, d], &dev);
    let v = randn(&[1, hkv, lk, d], &dev);
    let cfg = strict(1.0 / 8.0).split_kv(SplitKv::Fixed(8));
    let golden_fa = bits(&flash_attn_bhsd(&q, &k, &v, &cfg).unwrap());

    let w_lm = randn(&[97, 256], &dev);
    let h_lm = randn(&[4, 256], &dev);
    let golden_lm = candle::lm_head::fused_linear_argmax_cs(&h_lm, &w_lm, None, None, None)
        .unwrap()
        .ids
        .to_vec1::<u32>()
        .unwrap();

    let handles: Vec<_> = (0..4)
        .map(|tid| {
            let (q, k, v, cfg, w_lm, h_lm) = (
                q.clone(),
                k.clone(),
                v.clone(),
                cfg.clone(),
                w_lm.clone(),
                h_lm.clone(),
            );
            let golden_fa = golden_fa.clone();
            let golden_lm = golden_lm.clone();
            std::thread::spawn(move || {
                for iter in 0..8 {
                    let out_fa = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                    assert_eq!(
                        bits(&out_fa),
                        golden_fa,
                        "thread {tid} iter {iter}: Split-KV fa_split_kv corrupted"
                    );
                    let out_lm =
                        candle::lm_head::fused_linear_argmax_cs(&h_lm, &w_lm, None, None, None)
                            .unwrap();
                    assert_eq!(
                        out_lm.ids.to_vec1::<u32>().unwrap(),
                        golden_lm,
                        "thread {tid} iter {iter}: lm_head_parts corrupted"
                    );
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// GPU milliseconds per call of `f`, without the host overhead of the (debug) test build: a
/// large matmul keeps the GPU busy while `iters` calls are queued behind it, and CUDA events
/// around the queued calls measure them back to back. Best of three.
fn time_ms(dev: &Device, iters: usize, mut f: impl FnMut()) -> f64 {
    use candle::cuda_backend::cudarc::driver::sys::CUevent_flags;
    let Device::Cuda(cd) = dev else {
        unreachable!("CUDA benchmark")
    };
    static BLOCKER: std::sync::OnceLock<(Tensor, Tensor)> = std::sync::OnceLock::new();
    let (a, b) = BLOCKER.get_or_init(|| (randn(&[4096, 4096], dev), randn(&[4096, 4096], dev)));
    let stream = cd.cuda_stream();
    let event = || {
        let e = stream
            .context()
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        e.record(&stream).unwrap();
        e
    };
    f();
    dev.synchronize().unwrap();
    // best of three measurements whose calls were all queued while the blocker ran (host
    // enqueue shorter than the blocker); runs with a slower host are retried
    let (mut best, mut any, mut clean) = (f64::INFINITY, f64::INFINITY, 0);
    for _ in 0..10 {
        let e_block = event();
        let _busy = a.matmul(b).unwrap();
        let e0 = event();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        let host = t.elapsed().as_secs_f64() * 1e3;
        let e1 = event();
        let gpu = e0.elapsed_ms(&e1).unwrap() as f64 / iters as f64;
        let blocker = e_block.elapsed_ms(&e0).unwrap() as f64;
        any = any.min(gpu);
        if host < blocker {
            best = best.min(gpu);
            clean += 1;
            if clean == 3 {
                break;
            }
        }
    }
    if clean == 0 {
        eprintln!("bench: host enqueue never fit under the blocker, GPU time may be inflated");
        return any;
    }
    best
}

const SPLITS: [usize; 12] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/// Decode attention time per split count (`SplitKv::Fixed`), and what `SplitKv::Auto` picks.
#[test]
#[ignore]
fn bench_split_heuristic() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let cus = accel::profile(&dev).compute_units;
    println!("compute units: {cus}");
    print!("| kernel | q | kv | hq/hkv/d/lq | lk | auto ms |");
    for s in SPLITS {
        print!(" s{s} |");
    }
    println!(" best |");
    for (hq, hkv, d, lq) in [(16usize, 4usize, 256usize, 1usize), (16, 8, 128, 1), (16, 4, 256, 4)] {
        for lk in [1024usize, 2048, 4096, 8192] {
            if lq > 1 && (lk == 1024 || lk == 4096) {
                continue;
            }
            let k32 = randn(&[1, hkv, lk, d], &dev);
            let v32 = randn(&[1, hkv, lk, d], &dev);
            let scale = 1.0 / (d as f32).sqrt();
            // dense kernels: (q dtype, kv dtype)
            for (qd, kvd) in [
                (DType::F32, DType::F32),
                (DType::F32, DType::F16),
                (DType::F32, DType::F8E4M3),
                (DType::BF16, DType::BF16),
                (DType::BF16, DType::F8E4M3),
            ] {
                let q = randn(&[1, hq, lq, d], &dev).to_dtype(qd).unwrap();
                let (k, v) = (to_kv(&k32, kvd), to_kv(&v32, kvd));
                for gqa in [false, true] {
                    set_gqa(gqa);
                    let run = |split: SplitKv| {
                        let cfg = strict(scale).split_kv(split);
                        time_ms(&dev, 10, || {
                            flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                        })
                    };
                    let name = if gqa { "fa_rowg" } else { "fa_row" };
                    bench_row(name, qd, kvd, (hq, hkv, d, lq), lk, &run);
                }
                set_gqa(true);
            }
            // compressed FP8 rows
            let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 1, &dev).unwrap();
            let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &dev).unwrap();
            let (kr, vr) = (pack(&kq, &k32), pack(&vq, &v32));
            let q = randn(&[1, hq, lq, d], &dev);
            for (fast, gqa) in [(false, false), (true, false), (true, true)] {
                set_fp8_fast(fast);
                set_gqa(gqa);
                let run = |split: SplitKv| {
                    let cfg = strict(scale).split_kv(split);
                    time_ms(&dev, 10, || {
                        attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                    })
                };
                let name = match (fast, gqa) {
                    (false, _) => "tq_attn",
                    (true, false) => "tq_attn_fp8",
                    (true, true) => "tq_attng_fp8",
                };
                bench_row(name, DType::F32, DType::F8E4M3, (hq, hkv, d, lq), lk, &run);
            }
            set_fp8_fast(true);
            set_gqa(true);
        }
    }
}

fn bench_row(
    name: &str,
    qd: DType,
    kvd: DType,
    (hq, hkv, d, lq): (usize, usize, usize, usize),
    lk: usize,
    run: &dyn Fn(SplitKv) -> f64,
) {
    let auto = run(SplitKv::Auto);
    print!("| {name} | {qd:?} | {kvd:?} | {hq}/{hkv}/{d}/{lq} | {lk} | {auto:.4} |");
    let mut best = (f64::INFINITY, 0);
    for s in SPLITS {
        let t = run(if s == 1 { SplitKv::Off } else { SplitKv::Fixed(s) });
        if t < best.0 {
            best = (t, s);
        }
        print!(" {t:.4} |");
    }
    println!(" s{} |", best.1);
}

/// Decode attention before / after this change set: "before" = the previous defaults (split
/// rule of 2 workgroups per CU with 256+ keys per split, plain row kernels, generic `tq_attn`,
/// F16 K/V converted to F32 first); "after" = the current defaults.
#[test]
#[ignore]
fn bench_decode_summary() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let cus = accel::profile(&dev).compute_units;
    let old_splits = |rows: usize, lk: usize| {
        let target = 2 * cus;
        if rows >= target {
            1
        } else {
            target.div_ceil(rows).min(lk.div_ceil(256)).clamp(1, 64)
        }
    };
    println!("| path | hq/hkv/d | lk | before ms | after ms | speedup |");
    for (hq, hkv, d) in [(16usize, 4usize, 256usize), (16, 8, 128)] {
        for lk in [1024usize, 2048, 4096, 8192] {
            let (k32, v32) = (randn(&[1, hkv, lk, d], &dev), randn(&[1, hkv, lk, d], &dev));
            let scale = 1.0 / (d as f32).sqrt();
            let before_cfg = strict(scale).split_kv(SplitKv::Fixed(old_splits(hq, lk)));
            let after_cfg = strict(scale);
            let row = |name: &str, before: f64, after: f64| {
                println!(
                    "| {name} | {hq}/{hkv}/{d} | {lk} | {before:.4} | {after:.4} | {:.2}x |",
                    before / after
                );
            };
            for (name, qd, kvd) in [
                ("exact f32 kv, f32 q", DType::F32, DType::F32),
                ("exact f16 kv, f32 q", DType::F32, DType::F16),
                ("exact bf16 kv, bf16 q", DType::BF16, DType::BF16),
                ("fp8 kv, f32 q", DType::F32, DType::F8E4M3),
            ] {
                let q = randn(&[1, hq, 1, d], &dev).to_dtype(qd).unwrap();
                let (k, v) = (to_kv(&k32, kvd), to_kv(&v32, kvd));
                set_gqa(false);
                let before = time_ms(&dev, 10, || {
                    // the previous kv_as_is converted f16 K/V to the query dtype first
                    if kvd == DType::F16 {
                        let (kc, vc) = (k.to_dtype(qd).unwrap(), v.to_dtype(qd).unwrap());
                        flash_attn_bhsd(&q, &kc, &vc, &before_cfg).unwrap();
                    } else {
                        flash_attn_bhsd(&q, &k, &v, &before_cfg).unwrap();
                    }
                });
                set_gqa(true);
                let after = time_ms(&dev, 10, || {
                    flash_attn_bhsd(&q, &k, &v, &after_cfg).unwrap();
                });
                row(name, before, after);
            }
            let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 1, &dev).unwrap();
            let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &dev).unwrap();
            let (kr, vr) = (pack(&kq, &k32), pack(&vq, &v32));
            let q = randn(&[1, hq, 1, d], &dev);
            set_gqa(false);
            set_fp8_fast(false);
            let before = time_ms(&dev, 10, || {
                attention_packed(&q, &kr, &vr, lk, &kq, &vq, &before_cfg).unwrap();
            });
            set_gqa(true);
            set_fp8_fast(true);
            let after = time_ms(&dev, 10, || {
                attention_packed(&q, &kr, &vr, lk, &kq, &vq, &after_cfg).unwrap();
            });
            row("turbo-quant fp8 rows", before, after);
        }
    }
}

/// Paged decode over blocks against concatenating the blocks first (the old path of callers).
#[test]
#[ignore]
fn bench_paged_vs_cat() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(dev) = gpu() else { return };
    let (hq, hkv, d) = (16, 4, 256);
    let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 1, &dev).unwrap();
    let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &dev).unwrap();
    println!("| kv | blocks x 1024 | paged ms | cat + attention ms |");
    for n in [1usize, 2, 4, 8] {
        let q = randn(&[1, hq, 1, d], &dev);
        let cfg = strict(1.0 / 16.0);
        let dense = make_blocks(n, 1024, 1000, hkv, d, &|t: &Tensor| t.clone(), &dev);
        let paged = time_ms(&dev, 10, || {
            flash_attn_blocks(&q, &refs(&dense), &cfg).unwrap();
        });
        let cat = time_ms(&dev, 10, || {
            let (k, v) = cat_valid(&dense);
            flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
        });
        println!("| f32 exact | {n} | {paged:.4} | {cat:.4} |");
        let packed: Vec<(Tensor, Tensor, usize)> = dense
            .iter()
            .map(|(k, v, r)| (pack(&kq, k), pack(&vq, v), *r))
            .collect();
        let paged = time_ms(&dev, 10, || {
            attention_packed_blocks(&q, &refs(&packed), &kq, &vq, &cfg).unwrap();
        });
        let cat = time_ms(&dev, 10, || {
            let (k, v) = cat_valid(&packed);
            let lk = k.dim(2).unwrap();
            attention_packed(&q, &k, &v, lk, &kq, &vq, &cfg).unwrap();
        });
        println!("| tq fp8 | {n} | {paged:.4} | {cat:.4} |");
    }
}
