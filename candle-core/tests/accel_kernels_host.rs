//! Runs the GPU kernels of the fused attention / TurboQuant paths (kernels/flash_attn_impl.h,
//! kernels/turbo_quant_impl.h) on the host emulator and checks them against the Rust
//! reference implementations in `candle_core::accel`. The same kernel sources are compiled by
//! hipcc (ROCm) and NVRTC (CUDA), so this covers their logic without a GPU.
//!
//! Needs a C++20 compiler (`CXX`, else `c++`); the tests are skipped when none works.

use candle_core::accel::attention::{reference_attention, FlashAttnParams, ReferenceInputs};
use candle_core::accel::formats::{e4m3_to_f32, f32_to_e4m3, row_layout, QuantFormat};
use candle_core::accel::turbo::{RowCodec, SplitMix64};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn harness() -> Option<&'static Path> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        let bin = dir.join("accel_host_test");
        let src =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/tests/accel_host_test.cpp");
        let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
        let out = Command::new(&cxx)
            .args(["-std=c++20", "-O2", "-pthread", "-o"])
            .arg(&bin)
            .arg(&src)
            .output();
        match out {
            Ok(o) if o.status.success() => Some(bin),
            Ok(o) => {
                eprintln!(
                    "skipping accel host kernel tests: {cxx} failed\n{}",
                    String::from_utf8_lossy(&o.stderr)
                );
                None
            }
            Err(e) => {
                eprintln!("skipping accel host kernel tests: no C++ compiler ({cxx}: {e})");
                None
            }
        }
    })
    .as_deref()
}

#[derive(Default)]
struct Arrays(BTreeMap<String, (u8, Vec<u8>)>);

impl Arrays {
    fn put_bytes(&mut self, name: &str, kind: u8, bytes: Vec<u8>) -> &mut Self {
        self.0.insert(name.into(), (kind, bytes));
        self
    }
    fn f32(&mut self, name: &str, v: &[f32]) -> &mut Self {
        self.put_bytes(name, 0, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn u32(&mut self, name: &str, v: &[u32]) -> &mut Self {
        self.put_bytes(name, 1, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn u64(&mut self, name: &str, v: &[u64]) -> &mut Self {
        self.put_bytes(name, 2, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn i32(&mut self, name: &str, v: &[i32]) -> &mut Self {
        self.put_bytes(name, 3, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn str(&mut self, name: &str, s: &str) -> &mut Self {
        self.put_bytes(name, 4, s.as_bytes().to_vec())
    }
    fn get_f32(&self, name: &str) -> Vec<f32> {
        self.0[name]
            .1
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn get_u32(&self, name: &str) -> Vec<u32> {
        self.0[name]
            .1
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

fn run(input: &Arrays) -> Arrays {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let pid = std::process::id();
    let inp = dir.join(format!("accel_in_{pid}_{n}.bin"));
    let outp = dir.join(format!("accel_out_{pid}_{n}.bin"));
    let mut bytes = Vec::new();
    for (name, (kind, data)) in input.0.iter() {
        bytes.extend((name.len() as u32).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.push(*kind);
        bytes.extend((data.len() as u64).to_le_bytes());
        bytes.extend(data);
    }
    std::fs::write(&inp, bytes).unwrap();
    let st = Command::new(harness().unwrap())
        .arg(&inp)
        .arg(&outp)
        .status()
        .unwrap();
    assert!(st.success(), "harness failed");
    let data = std::fs::read(&outp).unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
    let mut out = Arrays::default();
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
        out.put_bytes(&name, kind, data[p..p + nb].to_vec());
        p += nb;
    }
    out
}

fn randn(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_normal() as f32 * scale).collect()
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst = 0f32;
    let mut at = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() - rtol * w.abs();
        if err > worst || !g.is_finite() {
            worst = if g.is_finite() { err } else { f32::INFINITY };
            at = i;
        }
    }
    assert!(
        worst <= atol,
        "{name}: index {at}: got {} want {} (excess {worst})",
        got[at],
        want[at]
    );
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Row { splits: u32 },
    Tile { version: u32, dmax: u32 },
}

#[derive(Clone)]
struct FaCase {
    b: usize,
    lq: usize,
    lk: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    /// 0 f32, 1 f16, 2 bf16
    qd: u32,
    kv8: bool,
    wl: i32,
    wr: i32,
    softcap: f32,
    alibi: bool,
    varlen: Option<(Vec<u32>, Vec<u32>)>,
    kind: Kind,
    gx: u32,
    /// q stored as (b, h, s, d) and passed through strides
    q_bhsd: bool,
    /// exact 16-bit K/V with f32 queries: 2 f16, 3 bf16 (0: K/V in the query dtype or FP8)
    kvh: u32,
    /// paged row kernel over blocks of (valid rows, capacity); the rows add up to lk
    pages: Option<Vec<(usize, usize)>>,
}

impl FaCase {
    fn new(b: usize, lq: usize, lk: usize, hq: usize, hkv: usize, d: usize, kind: Kind) -> Self {
        Self {
            b,
            lq,
            lk,
            hq,
            hkv,
            d,
            qd: 0,
            kv8: false,
            wl: -1,
            wr: -1,
            softcap: 0.0,
            alibi: false,
            varlen: None,
            kind,
            gx: 0,
            q_bhsd: false,
            kvh: 0,
            pages: None,
        }
    }
}

fn round_q(v: f32, qd: u32) -> f32 {
    match qd {
        1 => half::f16::from_f32(v).to_f32(),
        2 => half::bf16::from_f32(v).to_f32(),
        _ => v,
    }
}

fn check_fa(c: &FaCase) {
    if harness().is_none() {
        return;
    }
    let mut rng = SplitMix64::new(
        (c.b * 131
            + c.lq * 7
            + c.lk * 3
            + c.d
            + c.qd as usize * 1000
            + c.kv8 as usize
            + c.kvh as usize * 10) as u64,
    );
    let (total_q, total_k) = match &c.varlen {
        Some((cq, ck)) => (*cq.last().unwrap() as usize, *ck.last().unwrap() as usize),
        None => (c.b * c.lq, c.b * c.lk),
    };
    let (hq, hkv, d) = (c.hq, c.hkv, c.d);
    let q_bshd = randn(&mut rng, total_q * hq * d, 1.0);
    let k = randn(&mut rng, total_k * hkv * d, 1.0);
    let v = randn(&mut rng, total_k * hkv * d, 1.0);
    let (kd, vd) = if c.kv8 { (0.5f32, 0.25f32) } else { (1.0, 1.0) };
    // kernel input values as seen after the storage rounding
    let qr: Vec<f32> = q_bshd.iter().map(|&x| round_q(x, c.qd)).collect();
    let kv_round = |x: f32| {
        if c.kv8 {
            e4m3_to_f32(f32_to_e4m3(x))
        } else if c.kvh > 0 {
            round_q(x, c.kvh - 1)
        } else {
            round_q(x, c.qd)
        }
    };
    let kr: Vec<f32> = k.iter().map(|&x| kv_round(x)).collect();
    let vr: Vec<f32> = v.iter().map(|&x| kv_round(x)).collect();
    let slopes: Vec<f32> = (0..hq).map(|h| 0.05 * (h + 1) as f32).collect();
    let mut p = FlashAttnParams::new(1.0 / (d as f32).sqrt());
    p.softcap = c.softcap;
    p.window_left = c.wl;
    p.window_right = c.wr;
    p.k_descale = kd;
    p.v_descale = vd;
    let inp = ReferenceInputs {
        q: &qr,
        k: &kr,
        v: &vr,
        batch: c.b,
        lq: c.lq,
        lk: c.lk,
        hq,
        hkv,
        d,
        cu_q: c.varlen.as_ref().map(|v| v.0.as_slice()),
        cu_k: c.varlen.as_ref().map(|v| v.1.as_slice()),
        alibi: c.alibi.then_some((slopes.as_slice(), 0)),
    };
    let want = reference_attention(&inp, &p).unwrap();
    let want: Vec<f32> = want.iter().map(|&x| round_q(x, c.qd)).collect();

    // q in memory, possibly (b, h, s, d)
    let (q_mem, qs) = if c.q_bhsd && c.varlen.is_none() {
        let mut m = vec![0f32; q_bshd.len()];
        for b in 0..c.b {
            for s in 0..c.lq {
                for h in 0..hq {
                    for e in 0..d {
                        m[((b * hq + h) * c.lq + s) * d + e] =
                            q_bshd[((b * c.lq + s) * hq + h) * d + e];
                    }
                }
            }
        }
        (m, [(hq * c.lq * d) as u64, d as u64, (c.lq * d) as u64])
    } else {
        (
            q_bshd.clone(),
            [(c.lq * hq * d) as u64, (hq * d) as u64, d as u64],
        )
    };
    let mut ks = [(c.lk * hkv * d) as u64, (hkv * d) as u64, d as u64];
    let (kind, dmax, version, splits) = match c.kind {
        Kind::Row { splits } => (0u32, 0u32, 0u32, splits),
        Kind::Tile { version, dmax } => (1, dmax, version, 1),
    };
    let gx = if c.gx > 0 {
        c.gx
    } else {
        match c.kind {
            Kind::Row { .. } => (c.lq * hq) as u32,
            Kind::Tile { version, dmax } => {
                let br = match (version, dmax) {
                    (3, 64) => 64,
                    (3, 128) => 32,
                    (3, _) => 16,
                    (_, 64) => 32,
                    (_, 128) => 16,
                    _ => 8,
                };
                c.lq.div_ceil(br) as u32
            }
        }
    };
    let mut a = Arrays::default();
    // paged: every block (b, hkv, capacity, d) one after the other, junk in the spare rows
    let (k, v) = match &c.pages {
        None => (k, v),
        Some(pages) => {
            assert_eq!(pages.iter().map(|p| p.0).sum::<usize>(), c.lk);
            let (mut km, mut vm) = (Vec::new(), Vec::new());
            let (mut offs, mut rows, mut caps) = (Vec::new(), Vec::new(), Vec::new());
            let mut j0 = 0;
            for &(r, cap) in pages {
                offs.push(km.len() as u64);
                rows.push(r as u32);
                caps.push(cap as u32);
                for b in 0..c.b {
                    for h in 0..hkv {
                        for t in 0..cap {
                            for e in 0..d {
                                let src = ((b * c.lk + j0 + t) * hkv + h) * d + e;
                                km.push(if t < r { k[src] } else { 7.0 });
                                vm.push(if t < r { v[src] } else { -7.0 });
                            }
                        }
                    }
                }
                j0 += r;
            }
            a.u64("pg_off_k", &offs)
                .u64("pg_off_v", &offs)
                .u32("pg_rows", &rows)
                .u32("pg_cap_k", &caps)
                .u32("pg_cap_v", &caps);
            ks = [0, d as u64, 0];
            (km, vm)
        }
    };
    a.str("op", "fa")
        .u32("dtypes", &[c.qd, if c.kv8 { 1 } else { c.kvh }])
        .u32(
            "dims",
            &[
                c.b as u32,
                c.lq as u32,
                c.lk as u32,
                hq as u32,
                hkv as u32,
                d as u32,
                total_q as u32,
            ],
        )
        .u64(
            "strides",
            &[
                qs[0], qs[1], qs[2], ks[0], ks[1], ks[2], ks[0], ks[1], ks[2],
            ],
        )
        .f32("fparams", &[p.softmax_scale, c.softcap, kd, vd])
        .i32("iparams", &[c.wl, c.wr])
        .u32(
            "cfg",
            &[
                kind,
                dmax,
                version,
                splits,
                c.varlen.is_some() as u32,
                gx,
                0,
            ],
        )
        .f32("q", &q_mem)
        .f32("k", &k)
        .f32("v", &v);
    if let Some((cq, ck)) = &c.varlen {
        a.u32("cu_q", cq).u32("cu_k", ck);
    }
    if c.alibi {
        a.f32("alibi", &slopes);
    }
    let got = run(&a).get_f32("o");
    let (atol, rtol) = match c.qd {
        0 => (2e-5, 1e-4),
        1 => (2e-3, 2e-3),
        _ => (1.6e-2, 1.6e-2),
    };
    assert_close("flash-attn", &got, &want, atol, rtol);
}

#[test]
fn fa_row_kernel() {
    let row = Kind::Row { splits: 1 };
    let mut c = FaCase::new(2, 3, 37, 4, 2, 64, row);
    c.wr = 0;
    check_fa(&c);
    // flash-decoding splits, softcap, alibi, sliding window, head dim > 128
    let mut c = FaCase::new(1, 1, 300, 8, 2, 160, Kind::Row { splits: 3 });
    c.softcap = 5.0;
    c.alibi = true;
    c.wl = 100;
    c.wr = 0;
    check_fa(&c);
    // grid smaller than the row count (grid-stride loop), bf16 + fp8 kv
    let mut c = FaCase::new(1, 6, 20, 4, 4, 32, Kind::Row { splits: 2 });
    c.gx = 3;
    c.qd = 2;
    c.kv8 = true;
    c.wr = 0;
    check_fa(&c);
}

#[test]
fn fa_tile_v2_kernels() {
    for (d, dmax) in [(40, 64), (128, 128), (256, 256)] {
        let mut c = FaCase::new(1, 70, 70, 2, 1, d, Kind::Tile { version: 2, dmax });
        c.wr = 0;
        check_fa(&c);
    }
    // non-causal cross attention, more keys than queries, f16, q in (b, h, s, d)
    let mut c = FaCase::new(
        2,
        33,
        50,
        4,
        2,
        64,
        Kind::Tile {
            version: 2,
            dmax: 64,
        },
    );
    c.qd = 1;
    c.q_bhsd = true;
    check_fa(&c);
    // more queries than keys with a causal mask: the first rows see no key
    let mut c = FaCase::new(
        1,
        40,
        25,
        2,
        2,
        64,
        Kind::Tile {
            version: 2,
            dmax: 64,
        },
    );
    c.wr = 0;
    check_fa(&c);
}

#[test]
fn fa_tile_v3_kernels() {
    // bf16 queries with fp8 keys / values and descale factors, sliding window, softcap
    let mut c = FaCase::new(
        1,
        77,
        90,
        4,
        1,
        128,
        Kind::Tile {
            version: 3,
            dmax: 128,
        },
    );
    c.qd = 2;
    c.kv8 = true;
    c.wl = 10;
    c.wr = 0;
    c.softcap = 30.0;
    check_fa(&c);
    let mut c = FaCase::new(
        1,
        65,
        65,
        1,
        1,
        64,
        Kind::Tile {
            version: 3,
            dmax: 64,
        },
    );
    c.alibi = true;
    c.wr = 0;
    check_fa(&c);
    let mut c = FaCase::new(
        1,
        20,
        33,
        2,
        2,
        256,
        Kind::Tile {
            version: 3,
            dmax: 256,
        },
    );
    c.qd = 1;
    check_fa(&c);
}

#[test]
fn fa_varlen_kernels() {
    let cu_q = vec![0u32, 5, 5, 17];
    let cu_k = vec![0u32, 9, 20, 31];
    for kind in [
        Kind::Tile {
            version: 2,
            dmax: 64,
        },
        Kind::Row { splits: 1 },
        Kind::Row { splits: 2 },
    ] {
        let mut c = FaCase::new(3, 12, 11, 4, 2, 48, kind);
        c.varlen = Some((cu_q.clone(), cu_k.clone()));
        c.wr = 0;
        check_fa(&c);
    }
}

#[test]
fn fa_half_kv_with_f32_queries() {
    for kvh in [2u32, 3] {
        // decode row kernel with splits and the V2 / V3 tile kernels
        let mut c = FaCase::new(1, 2, 70, 4, 2, 64, Kind::Row { splits: 2 });
        c.kvh = kvh;
        c.wr = 0;
        check_fa(&c);
        for (version, dmax, d) in [(2u32, 128u32, 96usize), (3, 256, 256)] {
            let mut c = FaCase::new(1, 21, 30, 2, 1, d, Kind::Tile { version, dmax });
            c.kvh = kvh;
            c.wr = 0;
            check_fa(&c);
        }
    }
}

#[test]
fn fa_paged_row_kernel() {
    // 1..5 blocks, partial last block, batch > 1 (batch stride = kv heads x capacity rows)
    for n in 1..=5usize {
        let mut pages: Vec<(usize, usize)> = (0..n).map(|i| (20 + 3 * i, 24 + 3 * i)).collect();
        pages.last_mut().unwrap().0 = 5;
        let lk: usize = pages.iter().map(|p| p.0).sum();
        for (splits, lq) in [(1u32, 1usize), (3, 2), (2, 4)] {
            let mut c = FaCase::new(2, lq, lk, 4, 2, 64, Kind::Row { splits });
            c.wr = 0;
            c.pages = Some(pages.clone());
            check_fa(&c);
        }
    }
    // FP8 K/V with bf16 queries, sliding window crossing block borders, softcap
    let mut c = FaCase::new(1, 1, 90, 8, 2, 128, Kind::Row { splits: 4 });
    c.qd = 2;
    c.kv8 = true;
    c.wl = 33;
    c.wr = 0;
    c.softcap = 20.0;
    c.pages = Some(vec![(40, 40), (30, 64), (20, 21)]);
    check_fa(&c);
    // f16 K/V with f32 queries
    let mut c = FaCase::new(1, 3, 50, 2, 1, 256, Kind::Row { splits: 2 });
    c.kvh = 2;
    c.pages = Some(vec![(17, 32), (33, 33)]);
    check_fa(&c);
}

#[test]
fn fa_combine_matches_previous_formula() {
    if harness().is_none() {
        return;
    }
    let mut rng = SplitMix64::new(5);
    for (rows, d, splits) in [(5usize, 64usize, 2u32), (9, 256, 7), (3, 160, 64), (130, 32, 3)] {
        let po = randn(&mut rng, splits as usize * rows * d, 1.0);
        let mut pl = randn(&mut rng, splits as usize * rows, 3.0);
        // an empty split (all keys masked) and one row without any key
        pl[1] = f32::NEG_INFINITY;
        for s in 0..splits as usize {
            pl[s * rows + rows - 1] = f32::NEG_INFINITY;
        }
        let mut a = Arrays::default();
        a.str("op", "combine")
            .u32("params", &[rows as u32, d as u32, splits])
            .f32("part_o", &po)
            .f32("part_lse", &pl);
        let out = run(&a);
        let bits = |n: &str| -> Vec<u32> { out.get_f32(n).iter().map(|x| x.to_bits()).collect() };
        let legacy = bits("legacy");
        assert_eq!(bits("o"), legacy, "fa_combine rows {rows} d {d} splits {splits}");
        assert_eq!(bits("t"), legacy, "tq_combine rows {rows} d {d} splits {splits}");
    }
}

/// Runs `tq_attn` on the host emulator over rows `(b, hkv, lk)` with a row stride of `rw`
/// words, or over the same rows as paged blocks of `(rows, capacity)` when `pages` is set (the
/// spare rows hold junk with a huge scale). `fp8` selects the FP8 specialization.
#[allow(clippy::too_many_arguments)]
fn host_tq_attn(
    ck: &RowCodec,
    cv: &RowCodec,
    kd: &[u32],
    vd: &[u32],
    qx: &[f32],
    tab: &[f32],
    dims: (usize, usize, usize, usize, usize, usize),
    rw: usize,
    splits: u32,
    fp8: bool,
    pages: Option<&[(usize, usize)]>,
) -> Vec<f32> {
    let (b, lq, lk, hq, hkv, d) = dims;
    let (lkk, lvv) = (ck.layout, cv.layout);
    let qw = qx.len() / (b * hq * lq);
    let mut a = Arrays::default();
    let (kd, vd, kst) = match pages {
        None => (
            kd.to_vec(),
            vd.to_vec(),
            [(hkv * lk * rw) as u64, (lk * rw) as u64, rw as u64],
        ),
        Some(pages) => {
            let (mut km, mut vm) = (Vec::new(), Vec::new());
            let (mut offs, mut rows, mut caps) = (Vec::new(), Vec::new(), Vec::new());
            let mut j0 = 0;
            for &(r, cap) in pages {
                offs.push(km.len() as u64);
                rows.push(r as u32);
                caps.push(cap as u32);
                for bi in 0..b {
                    for h in 0..hkv {
                        for t in 0..cap {
                            let src = ((bi * hkv + h) * lk + j0 + t.min(r - 1)) * rw;
                            for w in 0..rw {
                                let junk = t >= r && w == lkk.rsc_offset();
                                km.push(if junk { 1e4f32.to_bits() } else { kd[src + w] });
                                vm.push(vd[src + w]);
                            }
                        }
                    }
                }
                j0 += r;
            }
            a.u64("pg_off_k", &offs)
                .u64("pg_off_v", &offs)
                .u32("pg_rows", &rows)
                .u32("pg_cap_k", &caps)
                .u32("pg_cap_v", &caps);
            (km, vm, [0, 0, rw as u64])
        }
    };
    a.str("op", "tq_attn")
        .f32("q", qx)
        .u32("kd", &kd)
        .u32("vd", &vd)
        .f32("tab", tab)
        .u32(
            "uparams",
            &[
                b as u32,
                lq as u32,
                lk as u32,
                hq as u32,
                hkv as u32,
                d as u32,
                ck.format.mode_id(),
                lkk.code_bits,
                lkk.code_words as u32,
                lkk.rsc_offset() as u32,
                cv.format.mode_id(),
                lvv.code_bits,
                lvv.code_words as u32,
                lvv.rsc_offset() as u32,
                splits,
                (lq * hq).min(5) as u32,
                0,
                fp8 as u32,
            ],
        )
        .u64(
            "strides",
            &[
                (hq * lq * qw) as u64,
                (lq * qw) as u64,
                qw as u64,
                kst[0],
                kst[1],
                kst[2],
                kst[0],
                kst[1],
                kst[2],
            ],
        )
        .f32("fparams", &[1.0 / (d as f32).sqrt(), 0.0])
        .i32("iparams", &[-1, 0]);
    run(&a).get_f32("o")
}

#[test]
fn tq_attention_fp8_fast_and_paged() {
    if harness().is_none() {
        return;
    }
    let mut rng = SplitMix64::new(4321);
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (fk, fv, b, lq, hq, hkv, d) in [
        (QuantFormat::Fp8, QuantFormat::Fp8, 2usize, 1usize, 4usize, 2usize, 64usize),
        (QuantFormat::Fp8, QuantFormat::Fp8, 1, 3, 2, 1, 256),
        (
            QuantFormat::TurboMse { bits: 4 },
            QuantFormat::TurboMse { bits: 3 },
            1,
            2,
            4,
            2,
            128,
        ),
    ] {
        let pages = [(40usize, 48usize), (64, 64), (9, 30)];
        let lk: usize = pages.iter().map(|p| p.0).sum();
        let ck = RowCodec::new(fk, d, 1).unwrap();
        let cv = RowCodec::new(fv, d, 2).unwrap();
        let rw = ck.layout.row_words.max(cv.layout.row_words);
        let n_rows = b * hkv * lk;
        let (mut kd, mut vd) = (vec![0u32; n_rows * rw], vec![0u32; n_rows * rw]);
        for r in 0..n_rows {
            let (kx, vx) = (randn(&mut rng, d, 1.5), randn(&mut rng, d, 1.0));
            ck.quantize_row(&kx, &mut kd[r * rw..r * rw + ck.layout.row_words]);
            cv.quantize_row(&vx, &mut vd[r * rw..r * rw + cv.layout.row_words]);
        }
        let qx = randn(&mut rng, b * hq * lq * d, 1.0);
        let mut tab = vec![0f32; 64];
        tab[..32].copy_from_slice(&tq_tab(&ck));
        tab[32..].copy_from_slice(&tq_tab(&cv));
        let dims = (b, lq, lk, hq, hkv, d);
        for splits in [1u32, 3] {
            let run = |fp8: bool, pg: Option<&[(usize, usize)]>| {
                host_tq_attn(&ck, &cv, &kd, &vd, &qx, &tab, dims, rw, splits, fp8, pg)
            };
            let dense = run(false, None);
            let paged = run(false, Some(&pages));
            assert_close(&format!("paged {fk}"), &paged, &dense, 1e-6, 1e-5);
            if fk == QuantFormat::Fp8 && fv == QuantFormat::Fp8 {
                assert_eq!(bits(&run(true, None)), bits(&dense), "fp8 fast, splits {splits}");
                assert_eq!(
                    bits(&run(true, Some(&pages))),
                    bits(&paged),
                    "paged fp8 fast, splits {splits}"
                );
            }
        }
    }
}

fn tq_tab(codec: &RowCodec) -> Vec<f32> {
    codec
        .codebook
        .as_ref()
        .map(|c| c.table().to_vec())
        .unwrap_or_else(|| vec![0f32; 32])
}

fn matvec(s: &[f32], d: usize, x: &[f32]) -> Vec<f32> {
    (0..d)
        .map(|i| (0..d).map(|j| s[i * d + j] * x[j]).sum())
        .collect()
}

/// Quantizes `rows` rows with the kernels (prod rows through resid + signs) and returns the
/// packed words.
fn kernel_quantize(codec: &RowCodec, x: &[f32], rows: usize, src_dtype: u32, gx: u32) -> Vec<u32> {
    let l = codec.layout;
    let d = l.dim;
    let mut a = Arrays::default();
    a.str("op", "tq_quant")
        .f32("x", x)
        .f32("tab", &tq_tab(codec))
        .u32(
            "params",
            &[
                rows as u32,
                d as u32,
                codec.format.mode_id(),
                l.code_bits,
                l.code_words as u32,
                l.aux_words as u32,
                l.row_words as u32,
                gx,
                src_dtype,
            ],
        );
    let out = run(&a);
    let words = out.get_u32("out");
    match &codec.qjl {
        None => words,
        Some(s) => {
            let resid = out.get_f32("resid");
            let proj: Vec<f32> = resid.chunks(d).flat_map(|r| matvec(s, d, r)).collect();
            let mut a = Arrays::default();
            a.str("op", "tq_signs")
                .u32("data", &words)
                .f32("proj", &proj)
                .f32("resid", &resid)
                .u32(
                    "params",
                    &[
                        rows as u32,
                        d as u32,
                        l.code_words as u32,
                        l.row_words as u32,
                        gx,
                    ],
                );
            run(&a).get_u32("out")
        }
    }
}

#[test]
fn tq_quantize_matches_cpu_codec() {
    if harness().is_none() {
        return;
    }
    let formats = [
        (QuantFormat::TurboMse { bits: 4 }, 128),
        (QuantFormat::TurboMse { bits: 3 }, 96),
        (QuantFormat::TurboMse { bits: 1 }, 64),
        (QuantFormat::TurboProd { bits: 4 }, 128),
        (QuantFormat::TurboProd { bits: 2 }, 64),
        (QuantFormat::Nvfp4, 128),
        (QuantFormat::Mxfp4, 64),
        (QuantFormat::Mxfp8, 256),
        (QuantFormat::Fp8, 80),
    ];
    let mut rng = SplitMix64::new(77);
    for (f, d) in formats {
        let codec = RowCodec::new(f, d, 5).unwrap();
        let l = codec.layout;
        let rows = 37;
        let mut x = randn(&mut rng, rows * d, 1.0);
        x[3] = 0.0; // exact zeros and an all-zero row
        for v in x[d..2 * d].iter_mut() {
            *v = 0.0;
        }
        let got = kernel_quantize(&codec, &x, rows, 0, 5);
        let mut want = vec![0u32; rows * l.row_words];
        for r in 0..rows {
            codec.quantize_row(
                &x[r * d..(r + 1) * d],
                &mut want[r * l.row_words..(r + 1) * l.row_words],
            );
        }
        let mut code_mismatch = 0;
        for r in 0..rows {
            let g = &got[r * l.row_words..(r + 1) * l.row_words];
            let w = &want[r * l.row_words..(r + 1) * l.row_words];
            let o = l.rsc_offset();
            for k in 0..2 {
                let (a, b) = (f32::from_bits(g[o + k]), f32::from_bits(w[o + k]));
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1e-6),
                    "{f} row {r} rsc{k}: {a} vs {b}"
                );
            }
            for j in 0..d {
                let cb = l.code_bits;
                if candle_core::accel::formats::get_bits(g, j * cb as usize, cb)
                    != candle_core::accel::formats::get_bits(w, j * cb as usize, cb)
                {
                    code_mismatch += 1;
                }
            }
            if !matches!(f, QuantFormat::TurboProd { .. }) {
                assert_eq!(
                    &g[l.code_words..o],
                    &w[l.code_words..o],
                    "{f} row {r} aux words"
                );
            }
        }
        // the norms are summed in a different order: allow a rare code flip at a boundary
        let tolerance = if f.is_turbo() { rows * d / 1000 } else { 0 };
        assert!(
            code_mismatch <= tolerance,
            "{f}: {code_mismatch} codes differ"
        );

        // dequantize the kernel words with the kernel, compare with the CPU codec
        let mut a = Arrays::default();
        let prod = matches!(f, QuantFormat::TurboProd { .. });
        a.str("op", "tq_dequant")
            .u32("data", &got)
            .f32("tab", &tq_tab(&codec))
            .u32(
                "params",
                &[
                    rows as u32,
                    d as u32,
                    f.mode_id(),
                    l.code_bits,
                    l.code_words as u32,
                    l.row_words as u32,
                    l.rsc_offset() as u32,
                    3,
                    prod as u32,
                ],
            );
        let out = run(&a);
        let mut deq = out.get_f32("out");
        if prod {
            let zs = out.get_f32("zs");
            let s = codec.qjl.as_ref().unwrap();
            for r in 0..rows {
                for j in 0..d {
                    let corr: f32 = (0..d).map(|i| zs[r * d + i] * s[i * d + j]).sum();
                    deq[r * d + j] += corr;
                }
            }
        }
        let mut want = vec![0f32; rows * d];
        for r in 0..rows {
            codec.dequantize_row(
                &got[r * l.row_words..(r + 1) * l.row_words],
                &mut want[r * d..(r + 1) * d],
            );
        }
        assert_close(&format!("{f} dequant"), &deq, &want, 1e-5, 1e-5);
    }
}

#[test]
fn tq_quantize_half_inputs() {
    if harness().is_none() {
        return;
    }
    let d = 64;
    let rows = 9;
    let mut rng = SplitMix64::new(9);
    let x = randn(&mut rng, rows * d, 2.0);
    for (dt, round) in [(1u32, 1u32), (2, 2)] {
        let xr: Vec<f32> = x.iter().map(|&v| round_q(v, round)).collect();
        let codec = RowCodec::new(QuantFormat::Fp8, d, 1).unwrap();
        let got = kernel_quantize(&codec, &x, rows, dt, 2);
        let l = codec.layout;
        let mut want = vec![0u32; rows * l.row_words];
        for r in 0..rows {
            codec.quantize_row(
                &xr[r * d..(r + 1) * d],
                &mut want[r * l.row_words..(r + 1) * l.row_words],
            );
        }
        assert_eq!(got, want, "fp8 from dtype {dt}");
    }
}

#[test]
fn tq_attention_matches_reference() {
    if harness().is_none() {
        return;
    }
    let cases = [
        // (k format, v format, b, lq, lk, hq, hkv, d, splits, causal, softcap)
        (
            QuantFormat::TurboProd { bits: 4 },
            QuantFormat::TurboMse { bits: 4 },
            2,
            1,
            150,
            4,
            2,
            64,
            1,
            true,
            0.0,
        ),
        (
            QuantFormat::TurboMse { bits: 3 },
            QuantFormat::TurboMse { bits: 2 },
            1,
            3,
            70,
            2,
            1,
            128,
            3,
            true,
            0.0,
        ),
        (
            QuantFormat::Nvfp4,
            QuantFormat::Mxfp4,
            1,
            2,
            64,
            2,
            2,
            64,
            2,
            false,
            20.0,
        ),
        (
            QuantFormat::Fp8,
            QuantFormat::Mxfp8,
            1,
            4,
            33,
            4,
            1,
            256,
            1,
            true,
            0.0,
        ),
    ];
    let mut rng = SplitMix64::new(1234);
    for (fk, fv, b, lq, lk, hq, hkv, d, splits, causal, softcap) in cases {
        let ck = RowCodec::new(fk, d, 21).unwrap();
        let cv = RowCodec::new(fv, d, 22).unwrap();
        let (lkk, lvv) = (ck.layout, cv.layout);
        assert_eq!(lkk.row_words, row_layout(fk, d).unwrap().row_words);
        // K / V rows in (b, hkv, lk) order, both with the key row stride of their own layout:
        // the kernel takes one (b, h, t) stride set, so use the larger row size for both.
        let rw = lkk.row_words.max(lvv.row_words);
        let n_rows = b * hkv * lk;
        let mut kd = vec![0u32; n_rows * rw];
        let mut vd = vec![0u32; n_rows * rw];
        let mut kdeq = vec![0f32; n_rows * d];
        let mut vdeq = vec![0f32; n_rows * d];
        for r in 0..n_rows {
            let kx = randn(&mut rng, d, 1.5);
            let vx = randn(&mut rng, d, 1.0);
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
        // queries (b, hq, lq, d) followed by S·q for prod keys
        let qw = if ck.qjl.is_some() { 2 * d } else { d };
        let q = randn(&mut rng, b * hq * lq * d, 1.0);
        let mut qx = vec![0f32; b * hq * lq * qw];
        for r in 0..b * hq * lq {
            qx[r * qw..r * qw + d].copy_from_slice(&q[r * d..(r + 1) * d]);
            if let Some(s) = &ck.qjl {
                let sq = matvec(s, d, &q[r * d..(r + 1) * d]);
                qx[r * qw + d..(r + 1) * qw].copy_from_slice(&sq);
            }
        }
        let mut tab = vec![0f32; 64];
        tab[..32].copy_from_slice(&tq_tab(&ck));
        tab[32..].copy_from_slice(&tq_tab(&cv));
        let scale = 1.0 / (d as f32).sqrt();
        let (wl, wr) = (-1, if causal { 0 } else { -1 });
        let mut a = Arrays::default();
        a.str("op", "tq_attn")
            .f32("q", &qx)
            .u32("kd", &kd)
            .u32("vd", &vd)
            .f32("tab", &tab)
            .u32(
                "uparams",
                &[
                    b as u32,
                    lq as u32,
                    lk as u32,
                    hq as u32,
                    hkv as u32,
                    d as u32,
                    fk.mode_id(),
                    lkk.code_bits,
                    lkk.code_words as u32,
                    lkk.rsc_offset() as u32,
                    fv.mode_id(),
                    lvv.code_bits,
                    lvv.code_words as u32,
                    lvv.rsc_offset() as u32,
                    splits,
                    (lq * hq).min(5) as u32,
                    0,
                    0,
                ],
            )
            .u64(
                "strides",
                &[
                    (hq * lq * qw) as u64,
                    (lq * qw) as u64,
                    qw as u64,
                    (hkv * lk * rw) as u64,
                    (lk * rw) as u64,
                    rw as u64,
                    (hkv * lk * rw) as u64,
                    (lk * rw) as u64,
                    rw as u64,
                ],
            )
            .f32("fparams", &[scale, softcap])
            .i32("iparams", &[wl, wr]);
        let got = run(&a).get_f32("o");
        // reference in (b, s, h, d) layouts on the dequantized rows
        let mut qr = vec![0f32; b * lq * hq * d];
        for bi in 0..b {
            for h in 0..hq {
                for s in 0..lq {
                    let src = ((bi * hq + h) * lq + s) * d;
                    let dst = ((bi * lq + s) * hq + h) * d;
                    qr[dst..dst + d].copy_from_slice(&q[src..src + d]);
                }
            }
        }
        let to_bshd = |x: &[f32]| {
            let mut o = vec![0f32; x.len()];
            for bi in 0..b {
                for h in 0..hkv {
                    for t in 0..lk {
                        let src = ((bi * hkv + h) * lk + t) * d;
                        let dst = ((bi * lk + t) * hkv + h) * d;
                        o[dst..dst + d].copy_from_slice(&x[src..src + d]);
                    }
                }
            }
            o
        };
        let (kr, vr) = (to_bshd(&kdeq), to_bshd(&vdeq));
        let mut p = FlashAttnParams::new(scale);
        p.window_left = wl;
        p.window_right = wr;
        p.softcap = softcap;
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
        // kernel output is (b, hq, lq, d)
        let mut want_bhsd = vec![0f32; want.len()];
        for bi in 0..b {
            for s in 0..lq {
                for h in 0..hq {
                    let src = ((bi * lq + s) * hq + h) * d;
                    let dst = ((bi * hq + h) * lq + s) * d;
                    want_bhsd[dst..dst + d].copy_from_slice(&want[src..src + d]);
                }
            }
        }
        assert_close(
            &format!("tq attention {fk}/{fv}"),
            &got,
            &want_bhsd,
            3e-5,
            2e-4,
        );
    }
}
