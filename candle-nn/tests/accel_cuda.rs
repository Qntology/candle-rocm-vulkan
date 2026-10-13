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

/// `t` in the K/V dtype `kvd` (FP8 casts go through the CPU: the CUDA cast kernels of the
/// candle kernels crate need sm_89+ for FP8).
fn to_kv(t: &Tensor, kvd: DType) -> Tensor {
    if kvd == DType::F8E4M3 {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(kvd)
            .unwrap()
            .to_device(t.device())
            .unwrap()
    } else {
        t.to_dtype(kvd).unwrap()
    }
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
    let (a, b) = BLOCKER.get_or_init(|| (randn(&[3072, 3072], dev), randn(&[3072, 3072], dev)));
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
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let e_block = event();
        let _busy = a.matmul(b).unwrap();
        let e0 = event();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        let host = t.elapsed().as_secs_f64() * 1e3;
        let e1 = event();
        let gpu = e0.elapsed_ms(&e1).unwrap() as f64;
        let blocker = e_block.elapsed_ms(&e0).unwrap() as f64;
        if host > blocker {
            eprintln!("bench: host enqueue {host:.2} ms > blocker {blocker:.2} ms, GPU time inflated");
        }
        best = best.min(gpu / iters as f64);
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
                        time_ms(&dev, 20, || {
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
                    time_ms(&dev, 20, || {
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
        let paged = time_ms(&dev, 20, || {
            flash_attn_blocks(&q, &refs(&dense), &cfg).unwrap();
        });
        let cat = time_ms(&dev, 20, || {
            let (k, v) = cat_valid(&dense);
            flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
        });
        println!("| f32 exact | {n} | {paged:.4} | {cat:.4} |");
        let packed: Vec<(Tensor, Tensor, usize)> = dense
            .iter()
            .map(|(k, v, r)| (pack(&kq, k), pack(&vq, v), *r))
            .collect();
        let paged = time_ms(&dev, 20, || {
            attention_packed_blocks(&q, &refs(&packed), &kq, &vq, &cfg).unwrap();
        });
        let cat = time_ms(&dev, 20, || {
            let (k, v) = cat_valid(&packed);
            let lk = k.dim(2).unwrap();
            attention_packed(&q, &k, &v, lk, &kq, &vq, &cfg).unwrap();
        });
        println!("| tq fp8 | {n} | {paged:.4} | {cat:.4} |");
    }
}
