//! Matmul row invariance (logis-center [EMBED-BATCHED] dynamic mode relies on it on the CPU).
//!
//! The app's CPU batched embedding forward groups texts of the exact same length and runs
//! `Linear` on [B·L, K] rows, so the same text is computed with M = L (per-text) or M = B·L
//! (batched, B varies per call). It is only cache-safe if every output row is a function of that
//! input row alone: independent of M (number of rows), of the row's position, and of the thread
//! count (no split-K, fixed-order k accumulation).
//!
//! - `linear_rows_are_m_and_position_invariant`: x[0..M] @ Wᵀ for M = 1..=300 (embedding Linear
//!   shapes 384→1152 / 384→384 / 384→3072 / 1536→384) compared bit-for-bit with M = 300, plus a
//!   reversed-row run (position invariance). M = 1 is reported separately (gemv path).
//! - `batched_attention_matmul_is_batch_invariant`: [B,12,L,32] @ [B,12,32,L] and
//!   [B,12,L,L] @ [B,12,L,32] for B = 1..=24, batch element 0 compared with B = 24.
//!
//! The CPU tests assert. The Metal / Vulkan variants (`--ignored`, need a device) only measure and
//! print the same numbers, to decide whether those backends need a fixed shape (like CUDA/ROCm) or
//! the dynamic mode is safe; they never count as an invariance pass.
//!
//! Prints a FNV-1a digest of the M = 300 outputs: run with RAYON_NUM_THREADS=1 and =N and compare
//! the digests to check thread-count invariance (CANDLE_MATMUL_INVARIANCE_DIGEST=<hex> asserts it).
use candle_core::{Device, Result, Tensor};

fn det(n: usize, salt: u64) -> Vec<f32> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn rows(t: &Tensor) -> Result<Vec<Vec<u32>>> {
    Ok(t.to_vec2::<f32>()?
        .into_iter()
        .map(|r| r.into_iter().map(f32::to_bits).collect())
        .collect())
}

fn fnv(h: &mut u64, bits: &[u32]) {
    for b in bits {
        for byte in b.to_le_bytes() {
            *h ^= byte as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Returns the number of (M ≥ 2 / position) mismatches; prints the per-shape report.
fn linear_m_report(dev: &Device, tag: &str) -> Result<usize> {
    let dev = dev.clone();
    const MAX_M: usize = 300;
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut m_ge2_bad_total = 0usize;
    for (k, n) in [(384usize, 1152usize), (384, 384), (384, 3072), (1536, 384)] {
        let x = Tensor::from_vec(det(MAX_M * k, (k * 7 + n) as u64), (MAX_M, k), &dev)?;
        // Linear keeps W as [N, K] and multiplies by its transposed view (candle-nn Linear).
        let w = Tensor::from_vec(det(n * k, (n * 13 + k) as u64), (n, k), &dev)?;
        let wt = w.t()?;
        let reference = rows(&x.matmul(&wt)?)?;
        for r in &reference {
            fnv(&mut digest, r);
        }
        let mut bad_m: Vec<usize> = Vec::new();
        let mut m1_rows_differ = false;
        for m in 1..=MAX_M {
            let y = rows(&x.narrow(0, 0, m)?.matmul(&wt)?)?;
            let differs = (0..m).any(|r| y[r] != reference[r]);
            if differs {
                if m == 1 {
                    m1_rows_differ = true;
                } else {
                    bad_m.push(m);
                }
            }
        }
        // Position: rows in reverse order must give the same per-row bits.
        let idx: Vec<u32> = (0..MAX_M as u32).rev().collect();
        let xr = x.index_select(&Tensor::new(idx.as_slice(), &dev)?, 0)?;
        let yr = rows(&xr.matmul(&wt)?)?;
        let pos_bad = (0..MAX_M)
            .filter(|&r| yr[MAX_M - 1 - r] != reference[r])
            .count();
        // A lone row at every position p of a 2-row batch [other, row] / [row, other].
        let mut pair_bad = 0usize;
        for r in (0..MAX_M).step_by(29) {
            let other = (r + 1) % MAX_M;
            for order in [[r as u32, other as u32], [other as u32, r as u32]] {
                let xp = x.index_select(&Tensor::new(&order[..], &dev)?, 0)?;
                let yp = rows(&xp.matmul(&wt)?)?;
                let at = if order[0] == r as u32 { 0 } else { 1 };
                if yp[at] != reference[r] {
                    pair_bad += 1;
                }
            }
        }
        println!(
            "[{tag}-M-INV] K={k:4} N={n:4}: M=2..={MAX_M} 행 결과가 M={MAX_M} 과 다른 M {}개 {:?} · M=1 {} · 역순 위치 불일치 {}/{MAX_M} · 2행 위치 불일치 {pair_bad}",
            bad_m.len(),
            &bad_m[..bad_m.len().min(16)],
            if m1_rows_differ { "다름(별도 경로)" } else { "같음" },
            pos_bad,
        );
        m_ge2_bad_total += bad_m.len() + pos_bad + pair_bad;
    }
    println!(
        "[{tag}-M-INV] arch {} · threads {} · 출력 digest {:016x}",
        std::env::consts::ARCH,
        candle_core::utils::get_num_threads(),
        digest
    );
    if dev.is_cpu() {
        if let Ok(want) = std::env::var("CANDLE_MATMUL_INVARIANCE_DIGEST") {
            assert_eq!(
                format!("{digest:016x}"),
                want.trim(),
                "출력 digest 가 스레드 수 / 실행에 따라 바뀌었습니다"
            );
        }
    }
    Ok(m_ge2_bad_total)
}

/// Returns the number of batch sizes whose element 0 differs from B = MAX_B.
fn batched_attn_report(dev: &Device, tag: &str) -> Result<usize> {
    let dev = dev.clone();
    const MAX_B: usize = 24;
    let mut bad = 0usize;
    for l in [3usize, 16, 64, 130] {
        let q = Tensor::from_vec(det(MAX_B * 12 * l * 32, l as u64), (MAX_B, 12, l, 32), &dev)?;
        let kt = Tensor::from_vec(
            det(MAX_B * 12 * 32 * l, 99 + l as u64),
            (MAX_B, 12, l, 32),
            &dev,
        )?
        .transpose(2, 3)?
        .contiguous()?;
        let att = (q.matmul(&kt)? / 32f64.sqrt())?;
        let p = candle_core::Tensor::from_vec(
            det(MAX_B * 12 * l * l, 7 + l as u64),
            (MAX_B, 12, l, l),
            &dev,
        )?;
        let v = Tensor::from_vec(
            det(MAX_B * 12 * l * 32, 31 + l as u64),
            (MAX_B, 12, l, 32),
            &dev,
        )?;
        let pv = p.matmul(&v)?;
        let ref_att = att.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let ref_pv = pv.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let mut bad_b = Vec::new();
        for b in 1..=MAX_B {
            let a = (q.narrow(0, 0, b)?.matmul(&kt.narrow(0, 0, b)?)? / 32f64.sqrt())?;
            let o = p.narrow(0, 0, b)?.matmul(&v.narrow(0, 0, b)?)?;
            let a0 = a.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
            let o0 = o.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
            let same = a0
                .iter()
                .zip(&ref_att)
                .all(|(x, y)| x.to_bits() == y.to_bits())
                && o0
                    .iter()
                    .zip(&ref_pv)
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            if !same {
                bad_b.push(b);
            }
        }
        println!(
            "[{tag}-B-INV] L={l:3}: QKᵀ · PV 배치 원소 0 이 B={MAX_B} 과 다른 B {:?}",
            bad_b
        );
        bad += bad_b.len();
    }
    Ok(bad)
}

#[test]
fn linear_rows_are_m_and_position_invariant() -> Result<()> {
    let bad = linear_m_report(&Device::Cpu, "CPU")?;
    assert_eq!(
        bad, 0,
        "CPU matmul 행 결과가 M(≥2) 또는 행 위치에 따라 바뀌었습니다"
    );
    Ok(())
}

#[test]
fn batched_attention_matmul_is_batch_invariant() -> Result<()> {
    let bad = batched_attn_report(&Device::Cpu, "CPU")?;
    assert_eq!(
        bad, 0,
        "배치 attention matmul 결과가 배치 크기에 따라 바뀌었습니다"
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "report-only: needs a Metal device (cargo test --features metal -- --ignored)"]
fn metal_matmul_m_invariance_report() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let a = linear_m_report(&dev, "METAL")?;
    let b = batched_attn_report(&dev, "METAL")?;
    println!("[METAL-M-INV] 측정 결과: Linear 불일치 {a} · 배치 attention 불일치 {b} (0 이면 동적 모드 가능)");
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "report-only: needs a Vulkan device (cargo test --features vulkan -- --ignored)"]
fn vulkan_matmul_m_invariance_report() -> Result<()> {
    use candle_core::vulkan_backend::shaders::{scoped_exec_mode, ExecMode};
    let dev = Device::new_vulkan(0)?;
    for mode in [ExecMode::Gpu, ExecMode::Host, ExecMode::Auto] {
        let _g = scoped_exec_mode(mode);
        let tag = format!("VK-{mode:?}");
        let a = linear_m_report(&dev, &tag)?;
        let b = batched_attn_report(&dev, &tag)?;
        println!("[{tag}-M-INV] 측정 결과: Linear 불일치 {a} · 배치 attention 불일치 {b}");
    }
    Ok(())
}
