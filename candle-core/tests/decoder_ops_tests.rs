//! Tests for fused decoder ops (`ResidualRmsNorm`, `SiluMul`, `RopeKvWrite`):
//! 1. Host-emulated GPU kernel tests (`CANDLE_ACCEL_HOST=1` C++20 thread+barrier runner)
//! 2. CPU reference + `Arc::strong_count > 1` aliasing protection tests
//! 3. Live GPU (ROCm / CUDA) accuracy tests across F32, F16, BF16

use candle_core::accel::decoder_ops::{
    self as dop, HeadLayout, RopeKvLaunchArgs, RopeMode, RES_RMS_PLUS_ONE_WEIGHT,
    RES_RMS_WRITE_RESIDUAL, ROPE_NORM_PLUS_ONE,
};
use candle_core::decoder_ops::{
    residual_rmsnorm, residual_rmsnorm_inplace, rmsnorm_fused, rope_kv_write, silu_mul,
};
use candle_core::{DType, Device, Result, Tensor};

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    if s == 0 {
        s = 1;
    }
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0f32;
    let mut max_ref = 1e-6f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        assert!(g.is_finite(), "non-finite output {g}");
        max_abs = max_abs.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
    }
    max_abs / max_ref
}

fn dtype_tol(dt: DType) -> f32 {
    match dt {
        DType::F32 => 1e-5,
        DType::F16 => 5e-3,
        DType::BF16 => 3e-2,
        _ => 1e-5,
    }
}

fn gpu_device() -> Option<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return Some(d);
    }
    #[cfg(feature = "rocm")]
    if let Ok(d) = Device::new_rocm(0) {
        return Some(d);
    }
    #[cfg(feature = "vulkan")]
    if let Ok(d) = Device::new_vulkan(0) {
        return Some(d);
    }
    None
}

// ---------------------------------------------------------------------------
// Host-emulated C++20 kernel tests
// ---------------------------------------------------------------------------

fn find_cxx() -> Option<String> {
    if let Ok(v) = std::env::var("CXX") {
        if !v.trim().is_empty() {
            return Some(v);
        }
    }
    let candidates = if cfg!(windows) {
        vec![
            r"C:\Program Files\AMD\ROCm\7.2\bin\clang++.exe",
            r"C:\Program Files\LLVM\bin\clang++.exe",
            "clang++",
            "g++",
            "cl",
        ]
    } else {
        vec!["c++", "clang++", "g++"]
    };
    for cand in candidates {
        if std::path::Path::new(cand).is_file() {
            return Some(cand.to_string());
        }
        if std::process::Command::new(cand)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(cand.to_string());
        }
    }
    None
}

fn host_harness() -> Option<&'static std::path::Path> {
    use std::path::PathBuf;
    use std::process::Command;
    static BIN: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        let cxx = find_cxx()?;
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("tests")
            .join("decoder_ops_host_test.cpp");
        let stem = std::path::Path::new(&cxx)
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let msvc = stem == "cl" || stem == "clang-cl";
        let bin = dir.join(if cfg!(windows) {
            "decoder_ops_host_test.exe"
        } else {
            "decoder_ops_host_test"
        });
        let mut cmd = Command::new(&cxx);
        if msvc {
            cmd.args(["/nologo", "/std:c++20", "/O2", "/EHsc"])
                .arg(format!("/Fe:{}", bin.display()))
                .arg(format!("/Fo:{}\\", dir.display()))
                .arg(&src);
        } else {
            cmd.args(["-std=c++20", "-O2", "-ffp-contract=off", "-pthread", "-o"])
                .arg(&bin)
                .arg(&src);
        }
        match cmd.output() {
            Ok(o) if o.status.success() => Some(bin),
            Ok(o) => {
                eprintln!(
                    "skipping decoder_ops host kernel tests: {cxx} failed\n{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                );
                None
            }
            Err(e) => {
                eprintln!("skipping decoder_ops host kernel tests ({cxx}: {e})");
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
        self.0
            .insert(name.into(), (0, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
        self
    }
    fn u32(&mut self, name: &str, v: &[u32]) -> &mut Self {
        self.0
            .insert(name.into(), (1, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
        self
    }
    fn u64(&mut self, name: &str, v: &[u64]) -> &mut Self {
        self.0
            .insert(name.into(), (2, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
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
    use std::io::{Read, Write};
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let pid = std::process::id();
    let inp = dir.join(format!("dec_in_{pid}_{n}.bin"));
    let outp = dir.join(format!("dec_out_{pid}_{n}.bin"));
    {
        let mut f = std::fs::File::create(&inp).unwrap();
        for (name, (kind, bytes)) in &input.0 {
            f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
            f.write_all(name.as_bytes()).unwrap();
            f.write_all(&[*kind]).unwrap();
            f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            f.write_all(bytes).unwrap();
        }
    }
    let st = std::process::Command::new(bin)
        .arg(&inp)
        .arg(&outp)
        .status()
        .unwrap();
    assert!(st.success(), "decoder_ops_host_test failed: {st}");
    let mut buf = Vec::new();
    std::fs::File::open(&outp)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);

    let mut out = HostArrays::default();
    let mut pos = 0usize;
    while pos < buf.len() {
        let nl = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let name = String::from_utf8(buf[pos..pos + nl].to_vec()).unwrap();
        pos += nl;
        let kind = buf[pos];
        pos += 1;
        let nb = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let bytes = buf[pos..pos + nb].to_vec();
        pos += nb;
        out.0.insert(name, (kind, bytes));
    }
    out
}

#[test]
fn host_emulated_kernels() {
    let Some(bin) = host_harness() else {
        eprintln!("no C++20 compiler available; skipping decoder_ops host emulation");
        return;
    };

    // 1) ResidualRmsNorm across non-aligned dims (72, 1152, 4304), standard & plus_one, in-place & out-of-place
    for &(rows, d, plus_one, inplace) in &[
        (3usize, 72usize, false, false),
        (2, 1152, true, true),
        (1, 4304, false, true),
    ] {
        for (dt_idx, dt) in [(0u32, DType::F32), (1, DType::F16), (2, DType::BF16)] {
            let x = det(rows * d, 10 + d as u64);
            let dl = det(rows * d, 20 + d as u64);
            let w = det(d, 30 + d as u64);
            let eps = 1e-6f32;
            let (ref_r, ref_y) = dop::cpu_res_rmsnorm_f32(&x, Some(&dl), &w, rows, d, eps, plus_one);
            let flags = RES_RMS_WRITE_RESIDUAL | if plus_one { RES_RMS_PLUS_ONE_WEIGHT } else { 0 };
            let mut inp = HostArrays::default();
            inp.str("op", "res_rmsnorm")
                .u32("dtype", &[dt_idx])
                .u32("dims", &[rows as u32, d as u32, flags, u32::from(inplace)])
                .f32("eps", &[eps])
                .f32("x", &x)
                .f32("delta", &dl)
                .f32("w", &w);
            let out = host_run(bin, &inp);
            let tol = dtype_tol(dt);
            let r_err = max_rel_err(&out.get_f32("r"), &ref_r);
            let y_err = max_rel_err(&out.get_f32("y"), &ref_y);
            assert!(r_err <= tol, "res_rmsnorm r err {r_err} > {tol} (d={d}, dt={dt:?})");
            assert!(y_err <= tol, "res_rmsnorm y err {y_err} > {tol} (d={d}, dt={dt:?})");
        }
    }

    // 2) SiluMul on merged strided [rows, 2 * inter] view (inter = 4304)
    {
        let (rows, inter) = (2usize, 4304usize);
        let merged = det(rows * 2 * inter, 77);
        let mut gate_c = Vec::with_capacity(rows * inter);
        let mut up_c = Vec::with_capacity(rows * inter);
        for r in 0..rows {
            gate_c.extend_from_slice(&merged[r * 2 * inter..r * 2 * inter + inter]);
            up_c.extend_from_slice(&merged[r * 2 * inter + inter..(r + 1) * 2 * inter]);
        }
        let ref_y = dop::cpu_silu_mul_f32(&gate_c, inter, &up_c, inter, rows, inter);
        for (dt_idx, dt) in [(0u32, DType::F32), (1, DType::F16), (2, DType::BF16)] {
            let mut inp = HostArrays::default();
            inp.str("op", "silu_mul")
                .u32("dtype", &[dt_idx])
                .u32("dims", &[rows as u32, inter as u32])
                .u64("strides", &[(2 * inter) as u64, (2 * inter) as u64, 0, inter as u64])
                .f32("gate", &merged)
                .f32("up", &merged);
            let out = host_run(bin, &inp);
            let err = max_rel_err(&out.get_f32("out"), &ref_y);
            assert!(err <= dtype_tol(dt), "silu_mul err {err} (dt={dt:?})");
        }
    }

    // 3) RopeKvWrite: partial RoPE (rot=32 < D=64), GQA (Hq=4, Hkv=2), plus_one QK-norm, NeoX & Adjacent
    for &(mode, plus_one, rot, d) in &[
        (RopeMode::NeoX, true, 32usize, 64usize),
        (RopeMode::Adjacent, false, 16, 64),
        (RopeMode::NeoX, false, 64, 64),
    ] {
        let (b, t, hq, hkv, max_seq, past) = (2usize, 3usize, 4usize, 2usize, 10usize, 4usize);
        let q_in = det(b * t * hq * d, 101);
        let k_in = det(b * t * hkv * d, 102);
        let v_in = det(b * t * hkv * d, 103);
        let qnw = det(d, 104);
        let knw = det(d, 105);
        let cos = det(b * t * (rot / 2), 106);
        let sin = det(b * t * (rot / 2), 107);
        let mut kc_ref = det(b * hkv * max_seq * d, 108);
        let mut vc_ref = det(b * hkv * max_seq * d, 109);
        let kc_init = kc_ref.clone();
        let vc_init = vc_ref.clone();

        let args = RopeKvLaunchArgs {
            b: b as u32,
            t: t as u32,
            hq: hq as u32,
            hkv: hkv as u32,
            d: d as u32,
            rotary_dim: rot as u32,
            past_len: past as u32,
            q_off: 0,
            k_off: 0,
            v_off: 0,
            qnw_off: 0,
            knw_off: 0,
            cos_off: 0,
            sin_off: 0,
            kc_off: 0,
            vc_off: 0,
            q_strides: [(t * hq * d) as u64, d as u64, (hq * d) as u64],
            qo_strides: [(hq * t * d) as u64, (t * d) as u64, d as u64],
            k_strides: [(t * hkv * d) as u64, d as u64, (hkv * d) as u64],
            v_strides: [(t * hkv * d) as u64, d as u64, (hkv * d) as u64],
            kc_strides: [(hkv * max_seq * d) as u64, (max_seq * d) as u64, d as u64],
            cos_strides: [(t * (rot / 2)) as u64, (rot / 2) as u64],
            eps: 1e-6,
            flags: mode.flag() | if plus_one { ROPE_NORM_PLUS_ONE } else { 0 },
        };
        let mut qo_ref = vec![0f32; b * hq * t * d];
        dop::cpu_rope_kv_write_f32(
            &q_in, &mut qo_ref, &k_in, &v_in, Some(&qnw), Some(&knw),
            &cos, &sin, &mut kc_ref, &mut vc_ref, &args,
        );

        let strides_u64 = [
            args.q_strides[0], args.q_strides[1], args.q_strides[2],
            args.qo_strides[0], args.qo_strides[1], args.qo_strides[2],
            args.k_strides[0], args.k_strides[1], args.k_strides[2],
            args.v_strides[0], args.v_strides[1], args.v_strides[2],
            args.kc_strides[0], args.kc_strides[1], args.kc_strides[2],
            args.cos_strides[0], args.cos_strides[1],
        ];
        for (dt_idx, dt) in [(0u32, DType::F32), (1, DType::F16), (2, DType::BF16)] {
            let mut inp = HostArrays::default();
            inp.str("op", "rope_kv_write")
                .u32("dtype", &[dt_idx])
                .u32("dims", &[args.b, args.t, args.hq, args.hkv, args.d, args.rotary_dim, args.past_len, args.flags])
                .u64("strides", &strides_u64)
                .f32("eps", &[args.eps])
                .f32("q_in", &q_in)
                .f32("k_in", &k_in)
                .f32("v_in", &v_in)
                .f32("q_norm_w", &qnw)
                .f32("k_norm_w", &knw)
                .f32("cos", &cos)
                .f32("sin", &sin)
                .f32("k_cache", &kc_init)
                .f32("v_cache", &vc_init);
            let out = host_run(bin, &inp);
            let tol = dtype_tol(dt);
            assert!(max_rel_err(&out.get_f32("q_out"), &qo_ref) <= tol);
            assert!(max_rel_err(&out.get_f32("k_cache"), &kc_ref) <= tol);
            assert!(max_rel_err(&out.get_f32("v_cache"), &vc_ref) <= tol);
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-level API tests (CPU + live ROCm/CUDA GPU + Arc aliasing safety)
// ---------------------------------------------------------------------------

fn verify_device(dev: &Device) -> Result<()> {
    for dt in [DType::F32, DType::F16, DType::BF16] {
        // Skip F16/BF16 CPU if unnecessary; CPU fallback works for all three!
        let tol = dtype_tol(dt);

        // 1) ResidualRmsNorm + aliasing safety (Arc::strong_count > 1 must NEVER mutate shared alias)
        let (rows, d) = (5usize, 1152usize);
        let x_cpu = Tensor::from_vec(det(rows * d, 1), (rows, d), &Device::Cpu)?;
        let dl_cpu = Tensor::from_vec(det(rows * d, 2), (rows, d), &Device::Cpu)?;
        let w_cpu = Tensor::from_vec(det(d, 3), (d,), &Device::Cpu)?;

        let x = x_cpu.to_dtype(dt)?.to_device(dev)?;
        let dl = dl_cpu.to_dtype(dt)?.to_device(dev)?;
        let w = w_cpu.to_dtype(dt)?.to_device(dev)?;

        let (ref_r, ref_y) = residual_rmsnorm(&x_cpu, Some(&dl_cpu), &w_cpu, 1e-6, true)?;
        let ref_rv = ref_r.flatten_all()?.to_vec1::<f32>()?;
        let ref_yv = ref_y.flatten_all()?.to_vec1::<f32>()?;

        // Shared alias test: clone x BEFORE calling residual_rmsnorm_inplace
        let x_alias = x.clone();
        let x_before = x_alias.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (r_shared, y_shared) = residual_rmsnorm_inplace(x, Some(&dl), &w, 1e-6, true)?;
        let x_after = x_alias.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(x_before, x_after, "shared Arc tensor MUST NOT be mutated in place!");
        assert!(max_rel_err(&r_shared.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, &ref_rv) <= tol);
        assert!(max_rel_err(&y_shared.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, &ref_yv) <= tol);

        // Sole-owner in-place path (copy() ensures distinct storage even when dev == Cpu && dt == F32)
        drop(x_alias);
        let x_sole = x_cpu.copy()?.to_dtype(dt)?.to_device(dev)?;
        assert!(x_sole.is_sole_owner_contiguous());
        let (r_in, y_in) = residual_rmsnorm_inplace(x_sole, Some(&dl), &w, 1e-6, true)?;
        assert!(max_rel_err(&r_in.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, &ref_rv) <= tol);
        assert!(max_rel_err(&y_in.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, &ref_yv) <= tol);

        // Standalone rmsnorm_fused
        let y_only = rmsnorm_fused(&r_in, &w, 1e-6, true)?;
        assert!(max_rel_err(&y_only.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, &ref_yv) <= tol);

        // 2) SiluMul on merged zero-copy strided gate_up view (dim = 4304)
        let inter = 4304usize;
        let gu_cpu = Tensor::from_vec(det(3 * 2 * inter, 4), (1, 3, 2 * inter), &Device::Cpu)?;
        let gu = gu_cpu.to_dtype(dt)?.to_device(dev)?;
        let gate_v = gu.narrow(2, 0, inter)?;
        let up_v = gu.narrow(2, inter, inter)?;
        let got_sm = silu_mul(&gate_v, &up_v)?;
        let ref_sm = silu_mul(&gu_cpu.narrow(2, 0, inter)?, &gu_cpu.narrow(2, inter, inter)?)?;
        assert!(
            max_rel_err(
                &got_sm.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &ref_sm.flatten_all()?.to_vec1::<f32>()?,
            ) <= tol
        );

        // 3) RopeKvWrite: prefill T=4 -> decode T=1 at past=4, checking non-overwritten slots too
        let (b, hq, hkv, hd, rot, max_seq) = (1usize, 8usize, 2usize, 64usize, 32usize, 12usize);
        let kc_cpu = Tensor::zeros((b, hkv, max_seq, hd), DType::F32, &Device::Cpu)?;
        let vc_cpu = Tensor::zeros((b, hkv, max_seq, hd), DType::F32, &Device::Cpu)?;
        let kc = Tensor::zeros((b, hkv, max_seq, hd), dt, dev)?;
        let vc = Tensor::zeros((b, hkv, max_seq, hd), dt, dev)?;
        let qnw_cpu = Tensor::from_vec(det(hd, 5), (hd,), &Device::Cpu)?;
        let knw_cpu = Tensor::from_vec(det(hd, 6), (hd,), &Device::Cpu)?;
        let qnw = qnw_cpu.to_device(dev)?;
        let knw = knw_cpu.to_device(dev)?;

        for (step, (past, t)) in [(0usize, 4usize), (4, 1)].into_iter().enumerate() {
            let q_cpu = Tensor::from_vec(det(b * t * hq * hd, 20 + step as u64), (b, t, hq, hd), &Device::Cpu)?;
            let k_cpu = Tensor::from_vec(det(b * t * hkv * hd, 30 + step as u64), (b, t, hkv, hd), &Device::Cpu)?;
            let v_cpu = Tensor::from_vec(det(b * t * hkv * hd, 40 + step as u64), (b, t, hkv, hd), &Device::Cpu)?;
            let cos_cpu = Tensor::from_vec(det(t * (rot / 2), 50 + step as u64), (t, rot / 2), &Device::Cpu)?;
            let sin_cpu = Tensor::from_vec(det(t * (rot / 2), 60 + step as u64), (t, rot / 2), &Device::Cpu)?;

            let q_ref = rope_kv_write(
                &q_cpu, &k_cpu, &v_cpu, HeadLayout::Bshd, HeadLayout::Bhsd,
                Some(&qnw_cpu), Some(&knw_cpu), &cos_cpu, &sin_cpu, &kc_cpu, &vc_cpu,
                past, 1e-6, RopeMode::NeoX, true,
            )?;
            let q_got = rope_kv_write(
                &q_cpu.to_dtype(dt)?.to_device(dev)?,
                &k_cpu.to_dtype(dt)?.to_device(dev)?,
                &v_cpu.to_dtype(dt)?.to_device(dev)?,
                HeadLayout::Bshd, HeadLayout::Bhsd,
                Some(&qnw), Some(&knw),
                &cos_cpu.to_device(dev)?, &sin_cpu.to_device(dev)?,
                &kc, &vc,
                past, 1e-6, RopeMode::NeoX, true,
            )?;
            assert_eq!(q_got.dims(), &[b, hq, t, hd]);
            assert!(
                max_rel_err(
                    &q_got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    &q_ref.flatten_all()?.to_vec1::<f32>()?,
                ) <= tol
            );
        }
        assert!(
            max_rel_err(
                &kc.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &kc_cpu.flatten_all()?.to_vec1::<f32>()?,
            ) <= tol
        );
        assert!(
            max_rel_err(
                &vc.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                &vc_cpu.flatten_all()?.to_vec1::<f32>()?,
            ) <= tol
        );
    }
    Ok(())
}

#[test]
fn decoder_ops_cpu_and_aliasing() -> Result<()> {
    verify_device(&Device::Cpu)
}

#[test]
fn decoder_ops_gpu_accuracy() -> Result<()> {
    let Some(dev) = gpu_device() else {
        eprintln!("no CUDA/ROCm/Vulkan GPU feature or device available; skipping GPU decoder_ops test");
        return Ok(());
    };
    verify_device(&dev)?;
    dev.synchronize()?;
    Ok(())
}
