//! Fused attention / TurboQuant kernels on a ROCm (HIP) GPU: checklist §3-3 (I-3), the ROCm
//! counterpart of `accel_cuda.rs`. Every result is compared with the CPU path or with another
//! kernel of the same GPU, and the measured errors are printed (`--nocapture`).
//!
//! | test | what |
//! | --- | --- |
//! | `profile_tiers_and_lds` | `AccelProfile` of the GPU (tier V1/V2/V3, FP8 class, compute units, LDS); the kernel each plan picks for head dims 64 / 128 / 256 (decode and prefill, tier auto / V1 / V2 / V3); the forced V2 / V3 tile kernels (V3 d128 needs 57,728 B of LDS, d256 51,392 B) against the CPU reference |
//! | `e4m3_is_ocp_fn_software_decode` | the CPU reference decodes OCP E4M3FN (not FNUZ) and the kernel sources use no hardware FP8 conversion |
//! | `e4m3_attention_loads_match_cpu` | all 256 e4m3 codes through the FP8 K/V loads of `fa_row*` / `fa_rowg*` / `fa_rowp*` (branch-free `ax_e4m3_to_f_fast`, eeb9564) equal the CPU decode |
//! | `e4m3_cast_kernels_match_cpu` | candle's ROCm cast kernels F8E4M3 <-> F32 against the CPU casts |
//! | `half_loads_decode_every_code` | all 65,536 f16 / bf16 codes through the F16 / BF16 K/V loads (branch-free `ax_h2f`, eeb9564) |
//! | `gqa_packed_rows_are_bit_exact` | GQA-packed decode `fa_rowg{2,4,8}` / `fa_rowpg*` (eeb9564) == `fa_row` / `fa_rowp`, and vs the CPU |
//! | `half_kv_with_f32_queries` | F32 queries + F16 / BF16 K/V: same bits as the widened F32 K/V, error vs the f32 reference <= 1e-3 (F16) / 1e-2 (BF16) |
//! | `paged_dense_matches_concatenated` | paged `fa_rowp*` vs attention over the concatenated blocks <= 1e-6 |
//! | `split_rule_long_kv` | `split_count_grouped` (4ff0b05) with this GPU's compute units; long-KV decode with automatic / fixed splits vs unsplit and vs the CPU |
//! | `turbo::*` | `tq_attn_fp8` == `tq_attn` (FP8 fast path), `tq_attng*` == `tq_attn`, paged `tq_attnp*` vs concatenated <= 1e-6, e4m3 decode of TurboQuant FP8 rows, `TurboKvCache` fused vs unfused |
//!
//! `cargo test -p candle-nn --features rocm,flash-attn,turbo-quant --test accel_rocm -- --nocapture`
//!
//! Host-timed benchmarks (ignored; the host overhead of the debug build is included, so only
//! the relative numbers mean something): append `--ignored bench_` after `--` for the split
//! counts of decoding (`bench_split_heuristic`) and the tiers of prefill
//! (`bench_prefill_tiers`).
//!
//! The accelerator configuration is process global: the tests take a lock, start from
//! `AccelConfig::default()` (whatever `CANDLE_FLASH_ATTN*` / `CANDLE_TURBO_QUANT*` say) and
//! restore the previous configuration when they end. Inputs are generated on the CPU with a
//! seeded generator (reproducible) and converted to the K/V dtypes there, so the GPU and the
//! CPU reference see the same bits.
#![cfg(all(feature = "rocm", feature = "flash-attn"))]

use candle::accel::attention::{self as att, AttnGeometry, FaPlan, FlashAttnParams};
use candle::accel::turbo::SplitMix64;
use candle::accel::{
    self, AccelBackend, AccelProfile, FlashAttnVersion, QuantFormat, SplitKv,
};
use candle::{DType, Device, Tensor};
use candle_nn::flash_attn::{flash_attn_bhsd, flash_attn_blocks, FlashAttnConfig, KvBlockRef};
use std::sync::{Mutex, MutexGuard};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static LOCK: Mutex<()> = Mutex::new(());

/// Serializes the tests and restores the accelerator configuration when dropped.
struct Serial {
    saved: accel::AccelConfig,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for Serial {
    fn drop(&mut self) {
        // runs before the lock is released (fields drop after `drop`)
        accel::set_config(self.saved.clone());
    }
}

fn serial() -> Serial {
    let lock = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let saved = accel::config();
    accel::set_config(accel::AccelConfig::default());
    Serial { saved, _lock: lock }
}

fn gpu() -> Device {
    Device::new_rocm(0).expect("ROCm device 0 (HIP runtime / driver)")
}

/// Seeded standard normal values on the CPU.
fn randn(shape: &[usize], seed: u64) -> Tensor {
    let mut rng = SplitMix64::new(seed.wrapping_mul(0x9e37_79b9).wrapping_add(17));
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.next_normal() as f32).collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn up(t: &Tensor, dev: &Device) -> Tensor {
    t.to_device(dev).unwrap()
}

/// Values as f32 on the host (the dtype conversion runs on the CPU, exact for f16 / bf16 / e4m3).
fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// max |a - b| / max |b| (infinite when either side has a non-finite value).
fn rel(a: &Tensor, b: &Tensor) -> f32 {
    let (a, b) = (host(a), host(b));
    assert_eq!(a.len(), b.len(), "rel: lengths differ");
    let (mut d, mut n) = (0f32, 0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return f32::INFINITY;
        }
        d = d.max((x - y).abs());
        n = n.max(y.abs());
    }
    d / n.max(1e-30)
}

fn assert_bits_eq(a: &Tensor, b: &Tensor, what: &str) {
    let (a, b) = (host(a), host(b));
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    let bad: Vec<usize> = (0..a.len())
        .filter(|&i| a[i].to_bits() != b[i].to_bits())
        .collect();
    if !bad.is_empty() {
        let first: Vec<String> = bad
            .iter()
            .take(6)
            .map(|&i| format!("[{i}] {:e} vs {:e}", a[i], b[i]))
            .collect();
        panic!(
            "{what}: {} of {} values differ in bits, first: {}",
            bad.len(),
            a.len(),
            first.join(", ")
        );
    }
}

/// Compares decoded values: NaN must meet NaN, everything else the same bits (with `any_zero`,
/// -0 and +0 count as equal: the attention kernels accumulate into a zero, which turns -0 into
/// +0). Panics on mismatches, except where the CPU value is an f32 subnormal: those are only
/// counted and returned (a flush-to-zero build would zero them; irrelevant for attention).
fn check_values(got: &[f32], want: &[f32], any_zero: bool, what: &str) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let norm = |x: f32| if any_zero { x + 0.0 } else { x };
    let (mut bad, mut subnormal) = (vec![], 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ok = if w.is_nan() {
            g.is_nan()
        } else {
            norm(g).to_bits() == norm(w).to_bits()
        };
        if !ok {
            if w != 0.0 && w.abs() < f32::MIN_POSITIVE {
                subnormal += 1;
            } else {
                bad.push(i);
            }
        }
    }
    if !bad.is_empty() {
        let first: Vec<String> = bad
            .iter()
            .take(8)
            .map(|&i| {
                format!(
                    "[{i}] gpu {:e} ({:#010x}) cpu {:e} ({:#010x})",
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                )
            })
            .collect();
        panic!(
            "{what}: {} of {} decoded values differ from the CPU, first: {}",
            bad.len(),
            got.len(),
            first.join(", ")
        );
    }
    subnormal
}

fn strict(scale: f32) -> FlashAttnConfig {
    FlashAttnConfig::new(scale).causal().fused(true).strict(true)
}

fn set_gqa(on: bool) {
    accel::update_config(|c| c.flash_attn.gqa_decode = on);
}

#[cfg(feature = "turbo-quant")]
fn set_fp8_fast(on: bool) {
    accel::update_config(|c| c.turbo_quant.fp8_fast_path = on);
}

/// Planning geometry of a dense batch-1 call (strides do not matter for the plan).
fn geometry(lq: usize, lk: usize, hq: usize, hkv: usize, d: usize) -> AttnGeometry {
    AttnGeometry {
        batch: 1,
        lq,
        lk,
        hq,
        hkv,
        d,
        q_strides: [0; 3],
        k_strides: [0; 3],
        v_strides: [0; 3],
        q_offset: 0,
        k_offset: 0,
        v_offset: 0,
        varlen: false,
        total_q: lq,
    }
}

fn params(scale: f32, version: FlashAttnVersion, gqa: bool) -> FlashAttnParams {
    let mut p = FlashAttnParams::new(scale);
    p.version = version;
    p.gqa_pack = gqa;
    p
}

/// `(kernel, grid, LDS bytes, splits, query heads per workgroup)` of the dense launch the ROCm
/// backend makes for a plan (same calls as `RocmStorage::flash_attn`).
fn launch_of(
    g: &AttnGeometry,
    p: &FlashAttnParams,
    plan: &FaPlan,
    cus: usize,
    q: DType,
    kv: DType,
) -> (String, (u32, u32, u32), usize, usize, usize) {
    match plan {
        FaPlan::Row { .. } => {
            let l = att::row_launch(g, p, cus, q, kv, false).unwrap();
            (l.name, l.grid, l.shared_bytes, l.splits, l.group)
        }
        FaPlan::Tile { .. } => {
            let name = att::kernel_name(plan, q, kv).unwrap();
            let (grid, smem, splits) = att::launch_dims(g, plan);
            (name, grid, smem, splits, 1)
        }
    }
}

/// `n` blocks `(1, hkv, cap, w)` on the CPU in dtype `kvd` (the last one holds `last` valid
/// rows; the spare capacity is junk).
fn make_blocks(
    n: usize,
    cap: usize,
    last: usize,
    hkv: usize,
    w: usize,
    seed: u64,
    kvd: DType,
) -> Vec<(Tensor, Tensor, usize)> {
    (0..n)
        .map(|i| {
            let rows = if i + 1 == n { last } else { cap };
            let s = seed + 2 * i as u64;
            let k = randn(&[1, hkv, cap, w], s).to_dtype(kvd).unwrap();
            let v = randn(&[1, hkv, cap, w], s + 1).to_dtype(kvd).unwrap();
            (k, v, rows)
        })
        .collect()
}

fn upload_blocks(blocks: &[(Tensor, Tensor, usize)], dev: &Device) -> Vec<(Tensor, Tensor, usize)> {
    blocks
        .iter()
        .map(|(k, v, r)| (up(k, dev), up(v, dev), *r))
        .collect()
}

fn refs(blocks: &[(Tensor, Tensor, usize)]) -> Vec<KvBlockRef<'_>> {
    blocks
        .iter()
        .map(|(k, v, rows)| KvBlockRef { k, v, rows: *rows })
        .collect()
}

/// The valid rows of the blocks, concatenated on the CPU and placed on `dev` (keeps the
/// reference independent of the ROCm copy / cat kernels).
fn cat_valid(blocks: &[(Tensor, Tensor, usize)], dev: &Device) -> (Tensor, Tensor) {
    let cpu = |t: &Tensor, r: usize| {
        t.to_device(&Device::Cpu)
            .unwrap()
            .narrow(2, 0, r)
            .unwrap()
    };
    let ks: Vec<Tensor> = blocks.iter().map(|(k, _, r)| cpu(k, *r)).collect();
    let vs: Vec<Tensor> = blocks.iter().map(|(_, v, r)| cpu(v, *r)).collect();
    (
        up(&Tensor::cat(&ks, 2).unwrap(), dev),
        up(&Tensor::cat(&vs, 2).unwrap(), dev),
    )
}

/// The 256 e4m3 codes as an F8E4M3 tensor `(256)` on the CPU and their CPU decode
/// (`float8::F8E4M3`, the reference of candle's F8E4M3 dtype).
fn e4m3_codes_cpu() -> (Tensor, Vec<f32>) {
    let codes: Vec<u8> = (0..=255u8).collect();
    let t = Tensor::from_raw_buffer(&codes, DType::F8E4M3, &[256], &Device::Cpu).unwrap();
    let vals = host(&t);
    (t, vals)
}

// ---------------------------------------------------------------------------
// AccelProfile tier, plans, LDS
// ---------------------------------------------------------------------------

/// Which tier / kernel the dispatch picks on this GPU (printed), that the V2 / V3 tiles fit the
/// 64 KiB of LDS of a workgroup, and that every tier runs and matches the CPU reference for head
/// dims 64 / 128 / 256 with F32 / F16 / FP8 K/V.
#[test]
fn profile_tiers_and_lds() {
    let _s = serial();
    let dev = gpu();
    let Device::Rocm(rd) = &dev else {
        unreachable!("new_rocm returned another device")
    };
    let info = rd.runtime_info();
    let hip = rd.hip_device();
    println!(
        "[rocm] {} | {info}",
        rd.name().unwrap_or_else(|e| format!("<name: {e}>"))
    );
    println!(
        "[rocm] target {:?}, wave size {:?}, hipDeviceAttributeMultiprocessorCount {:?}, kernels for [{}], kernels_match {:?}, mem (free, total) {:?}",
        rd.arch().map(|a| a.full.clone()),
        hip.warp_size().ok(),
        hip.compute_units().ok(),
        candle::RocmDevice::compiled_archs(),
        info.kernels_match,
        rd.mem_info().ok()
    );
    assert_ne!(
        info.kernels_match,
        Some(false),
        "the embedded HIP kernels have no code object for this GPU: {info}"
    );
    assert!(accel::FLASH_ATTN_COMPILED, "candle-core built without flash-attn");
    let prof = accel::profile(&dev);
    println!("[accel] profile: {prof}");
    println!(
        "[accel] compute_units {}, max_shared_mem {} B, native fp8 {}, fp8 class {}",
        prof.compute_units, prof.max_shared_mem, prof.native_fp8, prof.fp8_format
    );
    println!("{}", accel::describe(&dev));
    assert_eq!(prof.backend, AccelBackend::Rocm);
    assert!(prof.arch.starts_with("gfx"), "arch {:?}", prof.arch);
    assert_eq!(
        prof,
        AccelProfile::rocm(&prof.arch, prof.compute_units),
        "the device profile does not follow AccelProfile::rocm"
    );
    assert_eq!(prof.max_shared_mem, 64 * 1024);
    if prof.arch.starts_with("gfx10") {
        // RDNA1 (gfx101x): FlashAttention-1 tier; RDNA2 (gfx103x): FlashAttention-2 tier.
        // Both decode FP8 in software.
        let want = if prof.arch.starts_with("gfx103") {
            FlashAttnVersion::V2
        } else {
            FlashAttnVersion::V1
        };
        assert_eq!(prof.flash_attn_version, want, "{prof}");
        assert!(!prof.native_fp8 && !prof.native_fp4 && !prof.native_mx, "{prof}");
        assert_eq!(prof.fp8_format, QuantFormat::Fp8);
    }

    // LDS of the V3 tiles (the numbers of the checklist) and of every tile config
    for (d, want) in [(128usize, 57_728usize), (256, 51_392)] {
        let (_, br, bc) = att::tile_config(FlashAttnVersion::V3, d);
        let bytes = att::tile_smem_bytes(d, br, bc);
        assert_eq!(bytes, want, "V3 d{d} tile LDS");
    }
    let cus = prof.compute_units;
    println!("| lq | d | tier asked | plan | kernel (f32 q, f16 kv) | grid | LDS B | splits | heads/wg |");
    for d in [64usize, 128, 256] {
        for lq in [1usize, 64] {
            for version in [
                None,
                Some(FlashAttnVersion::V1),
                Some(FlashAttnVersion::V2),
                Some(FlashAttnVersion::V3),
            ] {
                let tier = version.unwrap_or(prof.flash_attn_version);
                let g = geometry(lq, 1024, 16, 4, d);
                let p = params(1.0 / (d as f32).sqrt(), tier, true);
                let plan = att::plan(&g, &p, prof.max_shared_mem, cus);
                let (name, grid, smem, splits, group) =
                    launch_of(&g, &p, &plan, cus, DType::F32, DType::F16);
                let asked = match version {
                    Some(v) => v.to_string(),
                    None => format!("auto={tier}"),
                };
                println!(
                    "| {lq} | {d} | {asked} | {plan:?} | {name} | {grid:?} | {smem} | {splits} | {group} |"
                );
                assert!(smem <= prof.max_shared_mem, "{name}: {smem} B of LDS");
                if lq > att::FA_DECODE_MAX_Q && tier != FlashAttnVersion::V1 {
                    // the requested tile tier fits in 64 KiB: no downgrade
                    assert!(
                        matches!(plan, FaPlan::Tile { version: v, .. } if v == tier),
                        "d{d} {tier}: {plan:?}"
                    );
                } else {
                    assert!(matches!(plan, FaPlan::Row { .. }), "d{d} lq {lq}: {plan:?}");
                }
            }
        }
    }

    // every tier on this GPU against the CPU reference (prefill, causal, GQA 8/2)
    let (hq, hkv, lq, lk) = (8usize, 2usize, 64usize, 300usize);
    let mut worst = 0f32;
    println!("| d | kv | tier | kernel | LDS B | max rel err vs CPU |");
    for (case, d) in [64usize, 128, 256].into_iter().enumerate() {
        let scale = 1.0 / (d as f32).sqrt();
        let seed = 300 + 10 * case as u64;
        let q_cpu = randn(&[1, hq, lq, d], seed);
        let k32 = randn(&[1, hkv, lk, d], seed + 1);
        let v32 = randn(&[1, hkv, lk, d], seed + 2);
        let q = up(&q_cpu, &dev);
        for kvd in [DType::F32, DType::F16, DType::F8E4M3] {
            let (k_cpu, v_cpu) = (k32.to_dtype(kvd).unwrap(), v32.to_dtype(kvd).unwrap());
            let want = flash_attn_bhsd(&q_cpu, &k_cpu, &v_cpu, &strict(scale)).unwrap();
            let (k, v) = (up(&k_cpu, &dev), up(&v_cpu, &dev));
            for version in [
                None,
                Some(FlashAttnVersion::V1),
                Some(FlashAttnVersion::V2),
                Some(FlashAttnVersion::V3),
            ] {
                let tier = version.unwrap_or(prof.flash_attn_version);
                let cfg = match version {
                    Some(v) => strict(scale).version(v),
                    None => strict(scale),
                };
                let g = geometry(lq, lk, hq, hkv, d);
                let p = params(scale, tier, true);
                let plan = att::plan(&g, &p, prof.max_shared_mem, cus);
                let (name, _, smem, _, _) = launch_of(&g, &p, &plan, cus, DType::F32, kvd);
                let got = flash_attn_bhsd(&q, &k, &v, &cfg)
                    .unwrap_or_else(|e| panic!("{name} (d{d} {kvd:?} {tier}): {e}"));
                assert!(got.device().is_rocm());
                let e = rel(&got, &want);
                worst = worst.max(e);
                println!("| {d} | {kvd:?} | {tier} | {name} | {smem} | {e:.3e} |");
                assert!(e <= 1e-5, "{name} d{d} {kvd:?} {tier}: {e}");
            }
        }
    }
    println!("tiers vs CPU reference: worst max rel err {worst:.3e}");
}

// ---------------------------------------------------------------------------
// FP8 (OCP e4m3) decode
// ---------------------------------------------------------------------------

/// Kernel code that would replace the software e4m3 decode with a hardware conversion. MI300
/// (gfx94x) hardware FP8 is FNUZ (bias 8, 0x80 = NaN), not the OCP E4M3FN of candle's F8E4M3.
const HW_FP8: [&str; 8] = [
    "__builtin_amdgcn_cvt_f32_fp8",
    "__builtin_amdgcn_cvt_pk_f32_fp8",
    "__builtin_amdgcn_cvt_pk_fp8_f32",
    "__builtin_amdgcn_cvt_sr_fp8_f32",
    "__hip_fp8",
    "hip_fp8.h",
    "__nv_fp8",
    "cuda_fp8.h",
];

/// The CPU reference is OCP E4M3FN (NaN = S.1111.111, max 448, subnormals m * 2^-9, -0 kept),
/// `accel::formats::e4m3_to_f32` agrees with it for all codes, and the shared kernel headers
/// decode FP8 in software (`ax_e4m3_to_f_fast`). No GPU needed.
#[test]
fn e4m3_is_ocp_fn_software_decode() {
    let (_, cpu) = e4m3_codes_cpu();
    for c in 0..=255u8 {
        let ours = accel::formats::e4m3_to_f32(c);
        let theirs = cpu[c as usize];
        if c & 0x7f == 0x7f {
            assert!(ours.is_nan() && theirs.is_nan(), "code {c:#04x}");
        } else {
            assert_eq!(ours.to_bits(), theirs.to_bits(), "code {c:#04x}: {ours} vs {theirs}");
        }
    }
    let known: [(u8, f32); 14] = [
        (0x00, 0.0),
        (0x80, -0.0),
        (0x01, 0.001953125),  // min subnormal 2^-9
        (0x81, -0.001953125),
        (0x07, 0.013671875),  // max subnormal 7 * 2^-9
        (0x08, 0.015625),     // min normal 2^-6
        (0x38, 1.0),          // FNUZ would give 0.5
        (0xb8, -1.0),
        (0x40, 2.0),
        (0x78, 256.0),        // exponent 15 is a normal binade in E4M3FN
        (0x7d, 416.0),
        (0x7e, 448.0),        // max (FNUZ max is 240)
        (0xfe, -448.0),
        (0x77, 240.0),
    ];
    for (c, v) in known {
        assert_eq!(
            cpu[c as usize].to_bits(),
            v.to_bits(),
            "code {c:#04x}: {} expected {v}",
            cpu[c as usize]
        );
    }
    assert!(cpu[0x7f].is_nan() && cpu[0xff].is_nan());
    use accel::sources::{ACCEL_COMMON_H, FLASH_ATTN_IMPL_H, TURBO_QUANT_IMPL_H};
    for (name, src) in [
        ("accel_common.h", ACCEL_COMMON_H),
        ("flash_attn_impl.h", FLASH_ATTN_IMPL_H),
        ("turbo_quant_impl.h", TURBO_QUANT_IMPL_H),
    ] {
        for bad in HW_FP8 {
            assert!(!src.contains(bad), "{name} uses the hardware FP8 conversion {bad}");
        }
    }
    assert!(
        ACCEL_COMMON_H.contains("ax_e4m3_to_f_fast(p[i].b)"),
        "the FP8 K/V element load no longer goes through ax_e4m3_to_f_fast"
    );
    assert!(TURBO_QUANT_IMPL_H.contains("ax_e4m3_to_f_fast"));
    println!("e4m3: CPU reference is OCP E4M3FN; kernel headers decode FP8 in software");
}

/// All 256 e4m3 codes through the FP8 V loads of the attention row kernels: with one key the
/// softmax weight is exactly 1 (K = 0), so the output row is the decoded V row. Dense
/// `fa_row_*_fp8` / `fa_rowg{2,4,8}_*_fp8` and paged `fa_rowp*`, f32 / f16 / bf16 queries
/// (every e4m3 value is exact in f16 and bf16).
#[test]
fn e4m3_attention_loads_match_cpu() {
    let _s = serial();
    let dev = gpu();
    let d = 256;
    let (codes, want) = e4m3_codes_cpu();
    let v_cpu = codes.reshape((1, 1, 1, d)).unwrap();
    let k_cpu =
        Tensor::from_raw_buffer(&[0u8; 256], DType::F8E4M3, &[1, 1, 1, d], &Device::Cpu).unwrap();
    let (k, v) = (up(&k_cpu, &dev), up(&v_cpu, &dev));
    let mut runs = 0;
    for qd in [DType::F32, DType::F16, DType::BF16] {
        for g in [1usize, 2, 4, 8] {
            let q = up(&randn(&[1, g, 1, d], 40 + g as u64).to_dtype(qd).unwrap(), &dev);
            let exp: Vec<f32> = (0..g).flat_map(|_| want.iter().copied()).collect();
            for gqa in [false, true] {
                set_gqa(gqa);
                let cfg = strict(1.0 / 16.0).split_kv(SplitKv::Off);
                let p = params(1.0 / 16.0, FlashAttnVersion::V1, gqa);
                let name = att::row_launch(&geometry(1, 1, g, 1, d), &p, 1, qd, DType::F8E4M3, false)
                    .unwrap()
                    .name;
                let dense = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                let paged =
                    flash_attn_blocks(&q, &[KvBlockRef { k: &k, v: &v, rows: 1 }], &cfg).unwrap();
                assert_eq!(dense.dtype(), qd);
                let n0 = check_values(&host(&dense), &exp, true, &name);
                let n1 = check_values(&host(&paged), &exp, true, &format!("paged {name}"));
                assert_eq!(n0 + n1, 0, "{name}: e4m3 never decodes to an f32 subnormal");
                runs += 2;
            }
        }
    }
    println!("e4m3 attention loads: 256 codes x {runs} kernel runs equal the CPU decode (NaN 0x7f/0xff, max 448, subnormals)");
}

/// candle's ROCm cast kernels: F8E4M3 -> F32 for all 256 codes (bit for bit, -0 kept, NaN for
/// 0x7f / 0xff) and F32 -> F8E4M3 (round to nearest even, saturation to 448, ties, subnormals,
/// infinities, NaN) against the CPU casts (`float8` crate).
#[test]
fn e4m3_cast_kernels_match_cpu() {
    let _s = serial();
    let dev = gpu();
    let (codes, want) = e4m3_codes_cpu();
    let got = host(&up(&codes, &dev).to_dtype(DType::F32).unwrap());
    check_values(&got, &want, false, "rocm cast f8e4m3 -> f32");

    let mut xs: Vec<f32> = want.iter().copied().filter(|x| x.is_finite()).collect();
    for c in 0..0x7eu8 {
        // midpoints between neighbouring codes: ties to even
        let m = 0.5 * (want[c as usize] + want[c as usize + 1]);
        xs.push(m);
        xs.push(-m);
        xs.push(m * (1.0 + 1e-6));
        xs.push(m * (1.0 - 1e-6));
    }
    xs.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        449.0,
        460.0,
        464.0,
        465.0,
        1e6,
        -1e6,
        0.0009765625, // 2^-10: half of the smallest subnormal
        0.0009766,
        0.0029296875, // 1.5 * 2^-9
        1e-30,
        -1e-30,
    ]);
    let r = randn(&[4096], 77).to_vec1::<f32>().unwrap();
    xs.extend(r.iter().map(|x| x * 100.0));
    let x = Tensor::from_vec(xs.clone(), xs.len(), &Device::Cpu).unwrap();
    let cpu = host(&x.to_dtype(DType::F8E4M3).unwrap());
    let gpu = host(&up(&x, &dev).to_dtype(DType::F8E4M3).unwrap());
    check_values(&gpu, &cpu, false, "rocm cast f32 -> f8e4m3");
    println!(
        "e4m3 cast kernels: 256 codes decoded, {} values encoded like the CPU",
        xs.len()
    );
}

/// All 65,536 f16 / bf16 codes through the F16 / BF16 V loads of the row kernels with F32
/// queries (one key, weight 1, K = 0): `fa_row_f32_{f16,bf16}`, `fa_rowg{2,4,8}_*`, paged
/// `fa_rowp*`. The f16 load (`ax_h2f`) is the branch-free version of eeb9564 and must equal
/// the CPU conversion for every code; f16 never decodes to an f32 subnormal. bf16 subnormal
/// codes decode to f32 subnormals and are only counted (flush to zero).
#[test]
fn half_loads_decode_every_code() {
    let _s = serial();
    let dev = gpu();
    let (hkv, d) = (256usize, 256usize);
    let bytes: Vec<u8> = (0..=u16::MAX).flat_map(|c| c.to_le_bytes()).collect();
    for kvd in [DType::F16, DType::BF16] {
        let v_cpu = Tensor::from_raw_buffer(&bytes, kvd, &[1, hkv, 1, d], &Device::Cpu).unwrap();
        let want = host(&v_cpu);
        let k_cpu = Tensor::zeros((1, hkv, 1, d), kvd, &Device::Cpu).unwrap();
        let (k, v) = (up(&k_cpu, &dev), up(&v_cpu, &dev));
        let mut subnormal = 0;
        let mut kernels = std::collections::BTreeSet::new();
        for g in [1usize, 2, 4, 8] {
            let hq = hkv * g;
            let q = up(&randn(&[1, hq, 1, d], 50 + g as u64), &dev);
            let exp: Vec<f32> = (0..hq)
                .flat_map(|h| want[(h / g) * d..(h / g + 1) * d].iter().copied())
                .collect();
            for gqa in [false, true] {
                set_gqa(gqa);
                let cfg = strict(1.0 / 16.0).split_kv(SplitKv::Off);
                let p = params(1.0 / 16.0, FlashAttnVersion::V1, gqa);
                let l = att::row_launch(&geometry(1, 1, hq, hkv, d), &p, 1, DType::F32, kvd, false)
                    .unwrap();
                let dense = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                subnormal += check_values(&host(&dense), &exp, true, &l.name);
                let paged =
                    flash_attn_blocks(&q, &[KvBlockRef { k: &k, v: &v, rows: 1 }], &cfg).unwrap();
                subnormal += check_values(&host(&paged), &exp, true, &format!("paged {}", l.name));
                kernels.insert(l.name);
            }
        }
        println!(
            "{kvd:?} loads: 65536 codes equal the CPU conversion in {kernels:?} (+ paged); {subnormal} f32-subnormal results differ (flush to zero)"
        );
        if kvd == DType::F16 {
            assert_eq!(subnormal, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// GQA-packed decode, half K/V, paged attention, split rule
// ---------------------------------------------------------------------------

/// The GQA-packed row kernels (query heads of a kv head in one workgroup, eeb9564) give the
/// bits of the plain row kernels at the same split count, dense and paged, for every K/V
/// dtype, with and without key splits; with F32 queries both also match the CPU reference.
/// With `SplitKv::Auto` the packed kernel may pick more splits than the plain one (64 vs 128
/// keys per split, 4ff0b05): the plain kernel's own automatic result is checked to <= 4e-6
/// (plus one output ulp for f16 / bf16 outputs).
#[test]
fn gqa_packed_rows_are_bit_exact() {
    let _s = serial();
    let dev = gpu();
    let cus = accel::profile(&dev).compute_units;
    let mut worst = 0f32;
    let (mut worst_auto, mut worst_auto_half, mut auto_differs) = (0f32, 0f32, 0usize);
    // GQA ratios 4, 2, 16 (packs of 8) and 3 (no packing: falls back to fa_row)
    for (case, (hq, hkv, d, lk)) in [
        (16usize, 4usize, 256usize, 1500usize),
        (16, 8, 128, 700),
        (32, 2, 64, 300),
        (12, 4, 128, 200),
    ]
    .into_iter()
    .enumerate()
    {
        let scale = 1.0 / (d as f32).sqrt();
        for (pair, (qd, kvd)) in [
            (DType::F32, DType::F32),
            (DType::F32, DType::F16),
            (DType::F32, DType::BF16),
            (DType::F32, DType::F8E4M3),
            (DType::F16, DType::F16),
            (DType::BF16, DType::BF16),
            (DType::F16, DType::F8E4M3),
            (DType::BF16, DType::F8E4M3),
        ]
        .into_iter()
        .enumerate()
        {
            let (cap, n) = (160usize, lk.div_ceil(160));
            let last = lk - (n - 1) * cap;
            let seed = 1000 * case as u64 + 50 * pair as u64;
            let host_blocks = make_blocks(n, cap, last, hkv, d, seed, kvd);
            let (k_cpu, v_cpu) = cat_valid(&host_blocks, &Device::Cpu);
            let blocks = upload_blocks(&host_blocks, &dev);
            let (k, v) = (up(&k_cpu, &dev), up(&v_cpu, &dev));
            for lq in [1usize, 3] {
                let q_cpu = randn(&[1, hq, lq, d], seed + 40 + lq as u64)
                    .to_dtype(qd)
                    .unwrap();
                let q = up(&q_cpu, &dev);
                let reference = (qd == DType::F32)
                    .then(|| flash_attn_bhsd(&q_cpu, &k_cpu, &v_cpu, &strict(scale)).unwrap());
                for split in [SplitKv::Off, SplitKv::Auto, SplitKv::Fixed(5)] {
                    let geom = geometry(lq, lk, hq, hkv, d);
                    let mut p = params(scale, FlashAttnVersion::V1, true);
                    p.split_kv = split;
                    let l = att::row_launch(&geom, &p, cus, qd, kvd, false).unwrap();
                    let lp = att::row_launch(&geom, &FlashAttnParams { gqa_pack: false, ..p }, cus, qd, kvd, false)
                        .unwrap();
                    // `SplitKv::Auto` (4ff0b05) gives packed workgroups 64 keys per split and
                    // plain ones 128, so the two may split differently (another summation order).
                    // The bit comparison runs the plain kernel with the split count of the packed
                    // one; the plain kernel's own automatic count is compared with a tolerance.
                    let plain_split = if lp.splits == l.splits { split } else { SplitKv::Fixed(l.splits) };
                    let cfg = strict(scale).split_kv(split);
                    let cfg_plain = strict(scale).split_kv(plain_split);
                    set_gqa(false);
                    let a = flash_attn_bhsd(&q, &k, &v, &cfg_plain).unwrap();
                    let ap = flash_attn_blocks(&q, &refs(&blocks), &cfg_plain).unwrap();
                    let a_own = (plain_split != split).then(|| {
                        (
                            flash_attn_bhsd(&q, &k, &v, &cfg).unwrap(),
                            flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap(),
                        )
                    });
                    set_gqa(true);
                    let b = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                    let bp = flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap();
                    let what = format!(
                        "{qd:?}/{kvd:?} {hq}/{hkv}/{d} lk {lk} lq {lq} {split:?} ({} {} splits, {} {} splits, plain run as {plain_split:?})",
                        l.name, l.splits, lp.name, lp.splits
                    );
                    assert_bits_eq(&a, &b, &format!("dense {what}"));
                    assert_bits_eq(&ap, &bp, &format!("paged {what}"));
                    if let Some((ao, aop)) = &a_own {
                        // another summation order: 4e-6 in f32, plus one ulp of the output at
                        // the largest magnitude when the result is stored as f16 / bf16
                        let tol = match qd {
                            DType::F16 => 4e-6 + 1.0 / 1024.0,
                            DType::BF16 => 4e-6 + 1.0 / 128.0,
                            _ => 4e-6,
                        };
                        let e = rel(ao, &b).max(rel(aop, &bp));
                        if qd == DType::F32 {
                            worst_auto = worst_auto.max(e);
                        } else {
                            worst_auto_half = worst_auto_half.max(e);
                        }
                        auto_differs += 1;
                        assert!(e <= tol, "{what}: plain auto ({} splits) vs packed auto: {e} > {tol}", lp.splits);
                    }
                    if let Some(r) = &reference {
                        let e = rel(&b, r).max(rel(&bp, r));
                        worst = worst.max(e);
                        assert!(e <= 1e-5, "{what} vs CPU: {e}");
                    }
                }
            }
        }
    }
    println!("GQA-packed rows: bit-identical to the plain rows at the same split count; f32 queries vs CPU worst max rel err {worst:.3e}");
    println!(
        "GQA-packed rows, SplitKv::Auto: {auto_differs} cases with another split count than the plain rows, worst plain auto vs packed auto {worst_auto:.3e} (f32 out), {worst_auto_half:.3e} (f16 / bf16 out)"
    );
}

/// F32 queries with exact F16 / BF16 K/V run without a conversion: the results have the bits
/// of the F32 kernels on the widened K/V (decode row kernels, and for 37 queries the row
/// kernel of the automatic tier plus the forced V2 / V3 tile kernels), match the CPU reference
/// on the same K/V, and differ from the f32 reference on the unrounded K/V by the storage
/// rounding only: <= 1e-3 (F16) / 1e-2 (BF16), max |err| / max |ref|.
#[test]
fn half_kv_with_f32_queries() {
    let _s = serial();
    let dev = gpu();
    let auto = accel::profile(&dev).flash_attn_version;
    println!("| kv | hq/hkv/d | lk | lq | tier | vs CPU same kv | vs f32 kv | CPU rounding floor |");
    for (case, (hq, hkv, d, lk)) in [
        (16usize, 4usize, 256usize, 1100usize),
        (16, 8, 128, 900),
        (8, 2, 64, 700),
    ]
    .into_iter()
    .enumerate()
    {
        let scale = 1.0 / (d as f32).sqrt();
        let seed = 2000 + 100 * case as u64;
        let k = randn(&[1, hkv, lk, d], seed);
        let v = randn(&[1, hkv, lk, d], seed + 1);
        for lq in [1usize, 4, 37] {
            let q_cpu = randn(&[1, hq, lq, d], seed + 10 + lq as u64);
            let q = up(&q_cpu, &dev);
            let exact = flash_attn_bhsd(&q_cpu, &k, &v, &strict(scale)).unwrap();
            for kvd in [DType::F16, DType::BF16] {
                let (kh, vh) = (k.to_dtype(kvd).unwrap(), v.to_dtype(kvd).unwrap());
                let (kw, vw) = (kh.to_dtype(DType::F32).unwrap(), vh.to_dtype(DType::F32).unwrap());
                let same = flash_attn_bhsd(&q_cpu, &kw, &vw, &strict(scale)).unwrap();
                let floor = rel(&same, &exact);
                let (khg, vhg) = (up(&kh, &dev), up(&vh, &dev));
                let (kwg, vwg) = (up(&kw, &dev), up(&vw, &dev));
                for version in [None, Some(FlashAttnVersion::V2), Some(FlashAttnVersion::V3)] {
                    if lq <= att::FA_DECODE_MAX_Q && version.is_some() {
                        continue; // decoding always runs the row kernels
                    }
                    let cfg = match version {
                        Some(v) => strict(scale).version(v),
                        None => strict(scale),
                    };
                    let tier = version.unwrap_or(auto);
                    let got = flash_attn_bhsd(&q, &khg, &vhg, &cfg).unwrap();
                    assert_eq!(got.dtype(), DType::F32);
                    let widened = flash_attn_bhsd(&q, &kwg, &vwg, &cfg).unwrap();
                    let what = format!("{kvd:?} kv {hq}/{hkv}/{d} lk {lk} lq {lq} {tier}");
                    assert_bits_eq(&got, &widened, &what);
                    let e_same = rel(&got, &same);
                    let e = rel(&got, &exact);
                    println!(
                        "| {kvd:?} | {hq}/{hkv}/{d} | {lk} | {lq} | {tier} | {e_same:.3e} | {e:.3e} | {floor:.3e} |"
                    );
                    assert!(e_same <= 1e-5, "{what} vs CPU on the same K/V: {e_same}");
                    let tol = if kvd == DType::F16 { 1e-3 } else { 1e-2 };
                    assert!(e <= tol, "{what} vs f32 K/V: {e} > {tol}");
                }
            }
        }
    }
}

/// Paged dense attention (`fa_rowp*` / `fa_rowpg*` over a block table) equals the attention
/// over the concatenated valid rows: 1..5 blocks, partial last block, F32 / F16 / BF16 / FP8
/// K/V, splits off / automatic, GQA packing off / on.
#[test]
fn paged_dense_matches_concatenated() {
    let _s = serial();
    let dev = gpu();
    let mut worst = 0f32;
    for (case, (hq, hkv, d)) in [(16usize, 4usize, 256usize), (16, 8, 128), (8, 8, 64)]
        .into_iter()
        .enumerate()
    {
        let scale = 1.0 / (d as f32).sqrt();
        for (ki, kvd) in [DType::F32, DType::F16, DType::BF16, DType::F8E4M3]
            .into_iter()
            .enumerate()
        {
            for n in 1..=5usize {
                let seed = 3000 + 1000 * case as u64 + 100 * ki as u64 + 10 * n as u64;
                let host_blocks = make_blocks(n, 160, 37 + 11 * n, hkv, d, seed, kvd);
                let (k, v) = cat_valid(&host_blocks, &dev);
                let blocks = upload_blocks(&host_blocks, &dev);
                for lq in [1usize, 2, 4] {
                    let q = up(&randn(&[1, hq, lq, d], seed + 7 + lq as u64), &dev);
                    for split in [SplitKv::Off, SplitKv::Auto] {
                        for gqa in [false, true] {
                            set_gqa(gqa);
                            let cfg = strict(scale).split_kv(split);
                            let got = flash_attn_blocks(&q, &refs(&blocks), &cfg).unwrap();
                            let want = flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                            let e = rel(&got, &want);
                            worst = worst.max(e);
                            assert!(
                                e <= 1e-6,
                                "{kvd:?} {hq}/{hkv}/{d} blocks {n} lq {lq} {split:?} gqa {gqa}: {e}"
                            );
                        }
                    }
                }
            }
        }
    }
    println!("paged dense vs concatenated: worst max rel err {worst:.3e}");
}

/// The split rule of 4ff0b05 (`split_count_grouped`: one 128-key window per split, 64 keys for
/// GQA-packed workgroups, while under `SPLIT_WORKGROUPS_PER_CU` workgroups per compute unit,
/// at most 64) with this GPU's compute units, printed next to the RTX 3050 laptop it was tuned
/// on (16 SMs); then long-KV decode (head dims 64 / 128 / 256, GQA ratios 4 / 2 / 8 / 3) with
/// the automatic split count against the unsplit kernel (<= 4e-6), the CPU reference (<= 1e-5)
/// and fixed split counts 2..64 (<= 4e-6).
#[test]
fn split_rule_long_kv() {
    let _s = serial();
    let dev = gpu();
    let cus = accel::profile(&dev).compute_units;
    println!(
        "[split] compute_units {cus} (hipDeviceAttributeMultiprocessorCount), SPLIT_WORKGROUPS_PER_CU {}, FA_MAX_SPLITS {}",
        att::SPLIT_WORKGROUPS_PER_CU,
        att::FA_MAX_SPLITS
    );
    println!("| hq/hkv | lq | heads/wg | lk | splits here | splits @16 SMs |");
    for (hq, hkv) in [(16usize, 4usize), (16, 8), (32, 4), (12, 4)] {
        let g = att::row_group(hq, hkv);
        for lq in [1usize, 4] {
            for lk in [512usize, 1024, 2048, 4096, 8192, 32768] {
                let rows = hq * lq;
                let here = att::split_count_grouped(rows, g, lk, SplitKv::Auto, cus);
                let ref16 = att::split_count_grouped(rows, g, lk, SplitKv::Auto, 16);
                println!("| {hq}/{hkv} | {lq} | {g} | {lk} | {here} | {ref16} |");
                assert!((1..=att::FA_MAX_SPLITS).contains(&here));
            }
        }
    }

    let mut worst_cpu = 0f32;
    let mut worst_split = 0f32;
    println!("| hq/hkv/d | lk | kernel | auto splits | auto vs off | auto vs CPU | worst fixed vs off |");
    for (case, (hq, hkv, d)) in [
        (16usize, 4usize, 256usize),
        (16, 8, 128),
        (32, 4, 64),
        (12, 4, 128),
    ]
    .into_iter()
    .enumerate()
    {
        let scale = 1.0 / (d as f32).sqrt();
        for lk in [2048usize, 8192] {
            let seed = 4000 + 100 * case as u64 + lk as u64;
            let k_cpu = randn(&[1, hkv, lk, d], seed);
            let v_cpu = randn(&[1, hkv, lk, d], seed + 1);
            let q_cpu = randn(&[1, hq, 1, d], seed + 2);
            let cpu = flash_attn_bhsd(&q_cpu, &k_cpu, &v_cpu, &strict(scale)).unwrap();
            let (k, v, q) = (up(&k_cpu, &dev), up(&v_cpu, &dev), up(&q_cpu, &dev));
            for gqa in [false, true] {
                set_gqa(gqa);
                let p = params(scale, FlashAttnVersion::V1, gqa);
                let l = att::row_launch(&geometry(1, lk, hq, hkv, d), &p, cus, DType::F32, DType::F32, false)
                    .unwrap();
                assert!(l.splits > 1, "long-KV decode should split: {l:?}");
                let off = flash_attn_bhsd(&q, &k, &v, &strict(scale).split_kv(SplitKv::Off)).unwrap();
                let auto = flash_attn_bhsd(&q, &k, &v, &strict(scale).split_kv(SplitKv::Auto)).unwrap();
                let (e_off, e_cpu) = (rel(&auto, &off), rel(&auto, &cpu));
                let mut e_fixed = 0f32;
                for n in [2usize, 3, 7, 16, 33, 64] {
                    let fixed =
                        flash_attn_bhsd(&q, &k, &v, &strict(scale).split_kv(SplitKv::Fixed(n)))
                            .unwrap();
                    e_fixed = e_fixed.max(rel(&fixed, &off));
                }
                worst_cpu = worst_cpu.max(e_cpu);
                worst_split = worst_split.max(e_off).max(e_fixed);
                println!(
                    "| {hq}/{hkv}/{d} | {lk} | {} | {} | {e_off:.3e} | {e_cpu:.3e} | {e_fixed:.3e} |",
                    l.name, l.splits
                );
                assert!(e_off <= 4e-6, "{} auto vs off: {e_off}", l.name);
                assert!(e_fixed <= 4e-6, "{} fixed vs off: {e_fixed}", l.name);
                assert!(e_cpu <= 1e-5, "{} vs CPU: {e_cpu}", l.name);
            }
            #[cfg(feature = "turbo-quant")]
            {
                let (a, b) = turbo::fp8_split_errors(&q, &k, &v, d, cus);
                worst_split = worst_split.max(a).max(b);
            }
        }
    }
    println!(
        "split rule: worst auto/fixed vs unsplit {worst_split:.3e}, worst auto vs CPU {worst_cpu:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Split benchmark (host-timed, ignored)
// ---------------------------------------------------------------------------

/// Milliseconds per call of `f`, timed on the host around `iters` queued calls followed by a
/// device synchronization (best of three). Includes the host overhead of the build.
fn time_ms(dev: &Device, iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    dev.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        dev.synchronize().unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    best
}

/// Decode attention time per split count, and what `SplitKv::Auto` picks (data for retuning
/// `split_count_grouped` to the CU count of this GPU).
#[test]
#[ignore]
fn bench_split_heuristic() {
    let _s = serial();
    let dev = gpu();
    let cus = accel::profile(&dev).compute_units;
    const SPLITS: [usize; 10] = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64];
    println!("compute units {cus}; host-timed ms per call (20 queued calls, best of 3)");
    print!("| kernel | hq/hkv/d | lk | auto ms (splits) |");
    for s in SPLITS {
        print!(" s{s} |");
    }
    println!(" best |");
    for (case, (hq, hkv, d)) in [(16usize, 4usize, 256usize), (16, 8, 128)].into_iter().enumerate() {
        let scale = 1.0 / (d as f32).sqrt();
        for lk in [1024usize, 2048, 4096, 8192] {
            let seed = 9000 + 10 * case as u64 + lk as u64;
            let k = up(&randn(&[1, hkv, lk, d], seed), &dev);
            let v = up(&randn(&[1, hkv, lk, d], seed + 1), &dev);
            let q = up(&randn(&[1, hq, 1, d], seed + 2), &dev);
            for gqa in [false, true] {
                set_gqa(gqa);
                let run = |split: SplitKv| {
                    let cfg = strict(scale).split_kv(split);
                    time_ms(&dev, 20, || {
                        flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                    })
                };
                let p = params(scale, FlashAttnVersion::V1, gqa);
                let l = att::row_launch(&geometry(1, lk, hq, hkv, d), &p, cus, DType::F32, DType::F32, false)
                    .unwrap();
                let auto = run(SplitKv::Auto);
                print!("| {} | {hq}/{hkv}/{d} | {lk} | {auto:.4} (s{}) |", l.name, l.splits);
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
        }
    }
}

/// Prefill attention time per FlashAttention tier (V1 row kernel, V2 / V3 tile kernels), data
/// for checking the automatic tier of `AccelProfile::rocm` on this GPU.
#[test]
#[ignore]
fn bench_prefill_tiers() {
    let _s = serial();
    let dev = gpu();
    let prof = accel::profile(&dev);
    println!(
        "{} (auto tier {}); host-timed ms per call (5 queued calls, best of 3)",
        prof.arch, prof.flash_attn_version
    );
    println!("| hq/hkv/d | l | kv | V1 ms | V2 ms | V3 ms |");
    for (case, (hq, hkv, d)) in [(16usize, 4usize, 128usize), (16, 4, 256)].into_iter().enumerate() {
        let scale = 1.0 / (d as f32).sqrt();
        for l in [512usize, 2048] {
            let seed = 9500 + 10 * case as u64 + l as u64;
            let q = up(&randn(&[1, hq, l, d], seed), &dev);
            let k32 = randn(&[1, hkv, l, d], seed + 1);
            let v32 = randn(&[1, hkv, l, d], seed + 2);
            for kvd in [DType::F32, DType::F16] {
                let (k, v) = (
                    up(&k32.to_dtype(kvd).unwrap(), &dev),
                    up(&v32.to_dtype(kvd).unwrap(), &dev),
                );
                let t: Vec<f64> = [FlashAttnVersion::V1, FlashAttnVersion::V2, FlashAttnVersion::V3]
                    .into_iter()
                    .map(|ver| {
                        let cfg = strict(scale).version(ver);
                        time_ms(&dev, 5, || {
                            flash_attn_bhsd(&q, &k, &v, &cfg).unwrap();
                        })
                    })
                    .collect();
                println!(
                    "| {hq}/{hkv}/{d} | {l} | {kvd:?} | {:.3} | {:.3} | {:.3} |",
                    t[0], t[1], t[2]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TurboQuant
// ---------------------------------------------------------------------------

#[cfg(feature = "turbo-quant")]
mod turbo {
    use super::*;
    use candle::accel::turbo::{tq_attn_group, tq_attn_kernel, TqSide};
    use candle::accel::RotationKind;
    use candle_nn::turbo_quant::{
        attention_packed, attention_packed_blocks, TurboKvCache, TurboQuantizer,
    };

    fn pack(q: &TurboQuantizer, x: &Tensor) -> Tensor {
        let (b, h, t, d) = x.dims4().unwrap();
        q.quantize_rows(&x.reshape((b * h * t, d)).unwrap())
            .unwrap()
            .reshape((b, h, t, q.row_words()))
            .unwrap()
    }

    /// One FP8 row of `codes.len()` values with the row scale `scale` (layout of
    /// `RowCodec::quantize_row`: one byte per code, then `rsc0`, `rsc1`).
    fn fp8_row(codes: &[u8], scale: f32) -> Vec<u32> {
        let l = accel::row_layout(QuantFormat::Fp8, codes.len()).unwrap();
        let mut w = vec![0u32; l.row_words];
        for (j, &c) in codes.iter().enumerate() {
            accel::formats::put_byte(&mut w, j, c);
        }
        w[l.rsc_offset()] = scale.to_bits();
        w
    }

    fn fp8_kernel(d: usize, hq: usize, hkv: usize, paged: bool) -> String {
        let side = TqSide::new(QuantFormat::Fp8, d).unwrap();
        tq_attn_kernel(&side, &side, paged, tq_attn_group(hq, hkv))
    }

    /// Auto / fixed splits vs unsplit of the FP8 TurboQuant rows (used by `split_rule_long_kv`).
    pub(super) fn fp8_split_errors(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        d: usize,
        cus: usize,
    ) -> (f32, f32) {
        let (_, hq, _, _) = q.dims4().unwrap();
        let (_, hkv, lk, _) = k.dims4().unwrap();
        let dev = q.device();
        let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 1, dev).unwrap();
        let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, dev).unwrap();
        let (kr, vr) = (pack(&kq, k), pack(&vq, v));
        let scale = 1.0 / (d as f32).sqrt();
        let run = |split: SplitKv| {
            attention_packed(q, &kr, &vr, lk, &kq, &vq, &strict(scale).split_kv(split)).unwrap()
        };
        let off = run(SplitKv::Off);
        let auto = rel(&run(SplitKv::Auto), &off);
        let fixed = rel(&run(SplitKv::Fixed(7)), &off);
        let splits = att::split_count_grouped(hq, tq_attn_group(hq, hkv), lk, SplitKv::Auto, cus);
        println!(
            "| {hq}/{hkv}/{d} | {lk} | {} | {splits} | {auto:.3e} | - | {fixed:.3e} (s7) |",
            fp8_kernel(d, hq, hkv, false)
        );
        assert!(auto <= 4e-6 && fixed <= 4e-6, "tq fp8 splits: auto {auto}, fixed {fixed}");
        (auto, fixed)
    }

    /// The FP8 specialization of `tq_attn` (`tq_attn_fp8`, `tq_attng*_fp8`, paged `tq_attnp*`)
    /// gives the bits of the generic kernel, with and without key splits and GQA packing, and
    /// both match the CPU path (dequantized rows + CPU attention).
    #[test]
    fn tq_fp8_fast_path_is_bit_exact() {
        let _s = serial();
        let dev = gpu();
        assert!(accel::TURBO_QUANT_COMPILED, "candle-core built without turbo-quant");
        let mut worst = 0f32;
        let mut names = std::collections::BTreeSet::new();
        for (case, (hq, hkv, d, lk)) in [
            (16usize, 4usize, 256usize, 1500usize),
            (16, 8, 128, 777),
            (4, 4, 64, 3),
            (32, 4, 64, 2100),
        ]
        .into_iter()
        .enumerate()
        {
            let scale = 1.0 / (d as f32).sqrt();
            let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 42, &dev).unwrap();
            let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 44, &dev).unwrap();
            let kq_cpu =
                TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 42, &Device::Cpu).unwrap();
            let vq_cpu =
                TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 44, &Device::Cpu).unwrap();
            let seed = 5000 + 100 * case as u64;
            let k = up(&randn(&[1, hkv, lk, d], seed), &dev);
            let v = up(&randn(&[1, hkv, lk, d], seed + 1), &dev);
            let (kr, vr) = (pack(&kq, &k), pack(&vq, &v));
            let (kr_cpu, vr_cpu) = (
                kr.to_device(&Device::Cpu).unwrap(),
                vr.to_device(&Device::Cpu).unwrap(),
            );
            let blocks = [KvBlockRef {
                k: &kr,
                v: &vr,
                rows: lk,
            }];
            for lq in [1usize, 3, 9] {
                let q_cpu = randn(&[1, hq, lq, d], seed + 10 + lq as u64);
                let q = up(&q_cpu, &dev);
                // CPU path: the CPU codec dequantizes the same rows, CPU reference attention
                let cpu = attention_packed(&q_cpu, &kr_cpu, &vr_cpu, lk, &kq_cpu, &vq_cpu, &strict(scale))
                    .unwrap();
                for split in [SplitKv::Off, SplitKv::Auto, SplitKv::Fixed(5)] {
                    for gqa in [false, true] {
                        set_gqa(gqa);
                        let cfg = strict(scale).split_kv(split);
                        set_fp8_fast(false);
                        let slow = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                        let slow_p = attention_packed_blocks(&q, &blocks, &kq, &vq, &cfg).unwrap();
                        names.insert(fp8_kernel(d, hq, hkv, false));
                        set_fp8_fast(true);
                        let fast = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                        let fast_p = attention_packed_blocks(&q, &blocks, &kq, &vq, &cfg).unwrap();
                        names.insert(fp8_kernel(d, hq, hkv, false));
                        names.insert(fp8_kernel(d, hq, hkv, true));
                        let what = format!("{hq}/{hkv}/{d} lk {lk} lq {lq} {split:?} gqa {gqa}");
                        assert_bits_eq(&fast, &slow, &format!("dense {what}"));
                        assert_bits_eq(&fast_p, &slow_p, &format!("paged {what}"));
                        let e = rel(&fast, &cpu).max(rel(&fast_p, &cpu));
                        worst = worst.max(e);
                        assert!(e <= 1e-5, "{what} vs CPU path: {e}");
                    }
                }
            }
        }
        println!("tq FP8 fast path == generic ({names:?}); vs CPU path worst max rel err {worst:.3e}");
    }

    /// The GQA-packed `tq_attng*` / `tq_attnpg*` kernels give the bits of `tq_attn*` (dense and
    /// paged, FP8 fast path and generic formats).
    #[test]
    fn tq_gqa_packed_is_bit_exact() {
        let _s = serial();
        let dev = gpu();
        for (case, (hq, hkv, d, lk)) in [
            (16usize, 4usize, 256usize, 1300usize),
            (16, 8, 128, 500),
            (32, 4, 64, 260),
        ]
        .into_iter()
        .enumerate()
        {
            let scale = 1.0 / (d as f32).sqrt();
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
                let (cap, n) = (128usize, lk.div_ceil(128));
                let last = lk - (n - 1) * cap;
                let seed = 6000 + 100 * case as u64;
                let blocks: Vec<(Tensor, Tensor, usize)> = (0..n)
                    .map(|i| {
                        let rows = if i + 1 == n { last } else { cap };
                        let s = seed + 2 * i as u64;
                        let k = pack(&kq, &up(&randn(&[1, hkv, cap, d], s), &dev));
                        let v = pack(&vq, &up(&randn(&[1, hkv, cap, d], s + 1), &dev));
                        (k, v, rows)
                    })
                    .collect();
                let (kr, vr) = cat_valid(&blocks, &dev);
                for lq in [1usize, 3] {
                    let q = up(&randn(&[1, hq, lq, d], seed + 50 + lq as u64), &dev);
                    for split in [SplitKv::Off, SplitKv::Fixed(4)] {
                        let cfg = strict(scale).split_kv(split);
                        let run = |gqa: bool| {
                            set_gqa(gqa);
                            (
                                attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap(),
                                attention_packed_blocks(&q, &refs(&blocks), &kq, &vq, &cfg)
                                    .unwrap(),
                            )
                        };
                        let (a, ap) = run(false);
                        let (b, bp) = run(true);
                        let what = format!("{kf}/{vf} {hq}/{hkv}/{d} lq {lq} {split:?}");
                        assert_bits_eq(&a, &b, &format!("dense {what}"));
                        assert_bits_eq(&ap, &bp, &format!("paged {what}"));
                    }
                }
            }
        }
        println!("tq GQA-packed kernels: bit-identical to tq_attn* (fp8, turbo4/3 hadamard, nvfp4/mxfp4)");
    }

    /// Paged compressed attention (`tq_attnp*`) equals `attention_packed` over the concatenated
    /// rows.
    #[test]
    fn paged_packed_matches_concatenated() {
        let _s = serial();
        let dev = gpu();
        let mut worst = 0f32;
        for (case, (hq, hkv, d)) in [(16usize, 4usize, 256usize), (16, 8, 128), (16, 4, 64)]
            .into_iter()
            .enumerate()
        {
            let scale = 1.0 / (d as f32).sqrt();
            for (fi, (kf, vf, rot)) in [
                (QuantFormat::Fp8, QuantFormat::Fp8, RotationKind::None),
                (
                    QuantFormat::TurboMse { bits: 4 },
                    QuantFormat::TurboMse { bits: 4 },
                    RotationKind::Hadamard,
                ),
                (QuantFormat::Nvfp4, QuantFormat::Mxfp4, RotationKind::None),
            ]
            .into_iter()
            .enumerate()
            {
                let kq = TurboQuantizer::new(kf, d, rot, 7, &dev).unwrap();
                let vq = TurboQuantizer::new(vf, d, rot, 9, &dev).unwrap();
                for n in 1..=5usize {
                    let (cap, last) = (96usize, 1 + 17 * n);
                    let seed = 7000 + 1000 * case as u64 + 100 * fi as u64 + 10 * n as u64;
                    let blocks: Vec<(Tensor, Tensor, usize)> = (0..n)
                        .map(|i| {
                            let rows = if i + 1 == n { last } else { cap };
                            let s = seed + 2 * i as u64;
                            let k = pack(&kq, &up(&randn(&[1, hkv, cap, d], s), &dev));
                            let v = pack(&vq, &up(&randn(&[1, hkv, cap, d], s + 1), &dev));
                            (k, v, rows)
                        })
                        .collect();
                    let (kr, vr) = cat_valid(&blocks, &dev);
                    let lk = kr.dim(2).unwrap();
                    for lq in [1usize, 4, 13] {
                        let q = up(&randn(&[1, hq, lq, d], seed + 50 + lq as u64), &dev);
                        for split in [SplitKv::Off, SplitKv::Auto] {
                            let cfg = strict(scale).split_kv(split);
                            let got =
                                attention_packed_blocks(&q, &refs(&blocks), &kq, &vq, &cfg).unwrap();
                            let want = attention_packed(&q, &kr, &vr, lk, &kq, &vq, &cfg).unwrap();
                            let e = rel(&got, &want);
                            worst = worst.max(e);
                            assert!(
                                e <= 1e-6,
                                "{kf}/{vf} {hq}/{hkv}/{d} blocks {n} lq {lq} {split:?}: {e}"
                            );
                        }
                    }
                }
            }
        }
        println!("paged packed vs concatenated: worst max rel err {worst:.3e}");
    }

    /// TurboQuant FP8 rows holding all 256 e4m3 codes (row scale 1): the GPU dequantize kernel
    /// and the FP8 attention kernels (one key, weight 1: generic `tq_attn`, `tq_attn_fp8`,
    /// `tq_attng*`, paged) reproduce the CPU codec's decode.
    #[test]
    fn tq_fp8_rows_decode_like_cpu() {
        let _s = serial();
        let dev = gpu();
        let d = 256;
        let (_, want) = e4m3_codes_cpu();
        let codes: Vec<u8> = (0..=255u8).collect();
        let row = fp8_row(&codes, 1.0);
        let words = row.len();
        let v_cpu = Tensor::from_vec(row, (1, 1, 1, words), &Device::Cpu).unwrap();
        let k_cpu = Tensor::from_vec(fp8_row(&[0u8; 256], 1.0), (1, 1, 1, words), &Device::Cpu)
            .unwrap();
        let (k, v) = (up(&k_cpu, &dev), up(&v_cpu, &dev));
        let kq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 1, &dev).unwrap();
        let vq = TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &dev).unwrap();
        let vq_cpu =
            TurboQuantizer::new(QuantFormat::Fp8, d, RotationKind::None, 3, &Device::Cpu).unwrap();
        assert_eq!(vq.row_words(), words);

        // dequantize: CPU codec vs the e4m3 reference, GPU kernel vs CPU codec
        let flat_cpu = v_cpu.reshape((1, words)).unwrap();
        let codec = host(&vq_cpu.dequantize_rows(&flat_cpu).unwrap());
        check_values(&codec, &want, false, "CPU codec fp8 rows");
        let deq = host(&vq.dequantize_rows(&up(&flat_cpu, &dev)).unwrap());
        check_values(&deq, &want, true, "rocm tq dequantize fp8 rows");

        let mut names = std::collections::BTreeSet::new();
        for g in [1usize, 2, 4, 8] {
            let q = up(&randn(&[1, g, 1, d], 60 + g as u64), &dev);
            let exp: Vec<f32> = (0..g).flat_map(|_| want.iter().copied()).collect();
            for fast in [false, true] {
                for gqa in [false, true] {
                    set_fp8_fast(fast);
                    set_gqa(gqa);
                    let cfg = strict(1.0 / 16.0).split_kv(SplitKv::Off);
                    let name = fp8_kernel(d, g, 1, false);
                    let dense = attention_packed(&q, &k, &v, 1, &kq, &vq, &cfg).unwrap();
                    check_values(&host(&dense), &exp, true, &name);
                    let paged = attention_packed_blocks(
                        &q,
                        &[KvBlockRef {
                            k: &k,
                            v: &v,
                            rows: 1,
                        }],
                        &kq,
                        &vq,
                        &cfg,
                    )
                    .unwrap();
                    check_values(&host(&paged), &exp, true, &fp8_kernel(d, g, 1, true));
                    names.insert(name);
                }
            }
        }
        println!("tq FP8 rows: 256 codes decode like the CPU codec in tq_dequant and {names:?} (+ paged)");
    }

    /// `TurboKvCache` (the KV cache of the app): attention on the compressed rows (fused
    /// `tq_attn*`) against dequantize + fused flash attention, prefill and decode queries.
    #[test]
    fn turbo_kv_cache_fused_matches_unfused() {
        let _s = serial();
        let dev = gpu();
        let (hq, hkv, d) = (16usize, 4usize, 128usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut worst = 0f32;
        for (fi, (kf, vf, rot)) in [
            (
                QuantFormat::TurboMse { bits: 4 },
                QuantFormat::TurboMse { bits: 3 },
                RotationKind::Auto,
            ),
            (QuantFormat::Nvfp4, QuantFormat::Fp8, RotationKind::None),
            (QuantFormat::Fp8, QuantFormat::Fp8, RotationKind::None),
            (QuantFormat::Mxfp4, QuantFormat::Mxfp8, RotationKind::Hadamard),
        ]
        .into_iter()
        .enumerate()
        {
            let mut cache = TurboKvCache::with_formats(kf, vf, d, rot, 1, 64, &dev).unwrap();
            for step in 0..4u64 {
                let t = if step == 0 { 300 } else { 1 };
                let seed = 8000 + 100 * fi as u64 + 2 * step;
                let k = up(&randn(&[1, hkv, t, d], seed), &dev);
                let v = up(&randn(&[1, hkv, t, d], seed + 1), &dev);
                cache.append(&k, &v).unwrap();
            }
            assert_eq!(cache.seq_len(), 303);
            let mut unfused = cache.clone();
            unfused.set_fused(false);
            for lq in [1usize, 5] {
                let q = up(&randn(&[1, hq, lq, d], 8500 + 10 * fi as u64 + lq as u64), &dev);
                let fused = cache.attention(&q, &strict(scale)).unwrap();
                let plain = unfused
                    .attention(&q, &FlashAttnConfig::new(scale).causal().strict(true))
                    .unwrap();
                assert_eq!(fused.dims(), &[1, hq, lq, d]);
                let e = rel(&fused, &plain);
                worst = worst.max(e);
                println!("TurboKvCache {kf}/{vf} {rot:?} lq {lq}: fused vs dequantized max rel err {e:.3e}");
                assert!(e <= 1e-4, "{kf}/{vf} lq {lq}: {e}");
            }
        }
        println!("TurboKvCache fused vs unfused: worst max rel err {worst:.3e}");
    }
}
