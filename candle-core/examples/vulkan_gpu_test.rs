//! Vulkan GPU-weights path: correctness against the CPU, speed, and where the memory
//! goes (Windows' own per-process numbers via DXGI).
//!
//!   cargo run --release --features vulkan --example vulkan_gpu_test
#[path = "../../tools/dxgi_usage.rs"]
mod dxgi_usage;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use std::time::Instant;

fn mb(x: u64) -> f64 {
    x as f64 / 1048576.0
}

fn os() -> String {
    dxgi_usage::process_vram()
        .map(|v| {
            format!(
                "OS VRAM {:6.0} MB shared {:5.0} MB",
                mb(v.local_usage),
                mb(v.nonlocal_usage)
            )
        })
        .unwrap_or_else(|| "OS n/a".into())
}

fn rep(tag: &str, vd: &candle_core::VulkanDevice) {
    let (l, s) = vd.gpu_memory_bytes();
    let (free, total) = vd.mem_info().unwrap_or((0, 0));
    println!(
        "[VK-GPU] {tag:<36} | {} | gpu bufs local {:6.0} MB shared {:5.0} MB | mem_info free {:6.0}/{:6.0} MB | host bufs {:6.0} MB",
        os(),
        mb(l as u64),
        mb(s as u64),
        mb(free as u64),
        mb(total as u64),
        mb(vd.allocated_bytes() as u64)
    );
}

fn rel_err(got: &Tensor, want: &Tensor) -> candle_core::Result<f32> {
    let a = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
    let b = want.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
    let diff = (&a - &b)?.abs()?.max(0)?.to_scalar::<f32>()?;
    let scale = b.abs()?.max(0)?.to_scalar::<f32>()?.max(1e-6);
    Ok(diff / scale)
}

fn check(name: &str, e: f32, tol: f32, fails: &mut usize) {
    let ok = e.is_finite() && e <= tol;
    if !ok {
        *fails += 1;
    }
    println!("[VK-GPU] {} {name:<46} rel err {e:.2e} (tol {tol:.0e})", if ok { "PASS" } else { "FAIL" });
}

fn time_ms<F: FnMut() -> candle_core::Result<()>>(iters: usize, mut f: F) -> candle_core::Result<f64> {
    // two warm-up calls: pipelines are built on the first, dense GPU mirrors on the second
    f()?;
    f()?;
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> candle_core::Result<()> {
    println!(
        "[VK-GPU] CANDLE_VULKAN_GPU_WEIGHTS={:?} CANDLE_VULKAN_DEVICE={:?} CANDLE_VULKAN_PRETEND_INTEGRATED={:?}",
        std::env::var("CANDLE_VULKAN_GPU_WEIGHTS").ok(),
        std::env::var("CANDLE_VULKAN_DEVICE").ok(),
        std::env::var("CANDLE_VULKAN_PRETEND_INTEGRATED").ok()
    );
    println!("[VK-GPU] usable vulkan devices: {}", candle_core::vulkan_backend::device_count()?);
    let vd = candle_core::VulkanDevice::new(0)?;
    print!("{}", vd.memory_diagnostics());
    let dev = Device::Vulkan(vd.clone());
    let cpu = Device::Cpu;
    rep("init", &vd);
    let mut fails = 0usize;

    // ---------------- Q8_0 ----------------
    for &(n, k) in &[(512usize, 2048usize), (6144, 2048), (2048, 6144), (4096, 2048), (64, 1024)] {
        let w = Tensor::randn(0f32, 1.0, (n, k), &cpu)?;
        let q_cpu = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let w_deq = q_cpu.dequantize(&cpu)?;
        let before = vd.gpu_memory_bytes();
        let q_vk = QTensor::quantize(&w.to_device(&dev)?, GgmlDType::Q8_0)?;
        let after = vd.gpu_memory_bytes();
        let placed = (after.0 + after.1) > (before.0 + before.1);
        println!("[VK-GPU] q8 {n}x{k}: weights in GPU memory = {placed}");
        let mm = QMatMul::from_qtensor(q_vk)?;
        for &m in &[1usize, 3, 8, 9, 64, 130] {
            let x = Tensor::randn(0f32, 1.0, (m, k), &cpu)?;
            let want = x.matmul(&w_deq.t()?)?;
            let got = mm.forward(&x.to_device(&dev)?)?;
            check(&format!("q8 matmul m={m} n={n} k={k}"), rel_err(&got, &want)?, 1e-4, &mut fails);
        }
        // batched + strided inputs
        let x3 = Tensor::randn(0f32, 1.0, (2, 5, k), &cpu)?;
        let want = x3.broadcast_matmul(&w_deq.t()?)?;
        let got = mm.forward(&x3.to_device(&dev)?)?;
        check(&format!("q8 matmul 3d (2,5) n={n} k={k}"), rel_err(&got, &want)?, 1e-4, &mut fails);
        let xt = Tensor::randn(0f32, 1.0, (k, 7), &cpu)?;
        let want = xt.t()?.matmul(&w_deq.t()?)?;
        let got = mm.forward(&xt.to_device(&dev)?.t()?)?;
        check(&format!("q8 matmul strided x n={n} k={k}"), rel_err(&got, &want)?, 1e-4, &mut fails);
    }
    // embedding / dequantize
    {
        let (rows, hidden) = (3000usize, 2048usize);
        let w = Tensor::randn(0f32, 1.0, (rows, hidden), &cpu)?;
        let q_cpu = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let w_deq = q_cpu.dequantize(&cpu)?;
        let q_vk = QTensor::quantize(&w.to_device(&dev)?, GgmlDType::Q8_0)?;
        let ids = Tensor::new(&[0u32, 5, 2999, 17, 17, 1234], &cpu)?;
        let got = q_vk.embedding(&ids.to_device(&dev)?)?;
        let want = w_deq.index_select(&ids, 0)?;
        check("q8 embedding", rel_err(&got, &want)?, 1e-6, &mut fails);
        let got = q_vk.dequantize(&dev)?;
        check("q8 dequantize f32", rel_err(&got, &w_deq)?, 1e-6, &mut fails);
        let got = q_vk.dequantize_f16(&dev)?;
        check("q8 dequantize f16", rel_err(&got, &w_deq.to_dtype(DType::F16)?)?, 1e-3, &mut fails);
        let data = q_vk.data()?;
        let back = candle_core::quantized::ggml_file::qtensor_from_ggml(
            GgmlDType::Q8_0,
            &data,
            vec![rows, hidden],
            &cpu,
        )?;
        check("q8 data() round trip", rel_err(&back.dequantize(&cpu)?, &w_deq)?, 0.0, &mut fails);
        // the GGUF load path: raw GGML bytes straight onto the Vulkan device
        let raw = q_cpu.data()?;
        let q_load = candle_core::quantized::ggml_file::qtensor_from_ggml(
            GgmlDType::Q8_0,
            &raw,
            vec![rows, hidden],
            &dev,
        )?;
        let got = q_load.dequantize(&dev)?;
        check("q8 loaded from GGML bytes", rel_err(&got, &w_deq)?, 1e-6, &mut fails);
    }

    // ---------------- placement of many weights (fills a heap, then spills) ----------------
    {
        let mut keep = Vec::new();
        for i in 0..12 {
            let w = Tensor::randn(0f32, 1.0, (4096usize, 4096usize), &cpu)?; // 17.8 MB as Q8_0
            keep.push(QTensor::quantize(&w.to_device(&dev)?, GgmlDType::Q8_0)?);
            if i % 4 == 3 {
                rep(&format!("placement: {} q8 tensors (~{} MB)", i + 1, (i + 1) * 18), &vd);
            }
        }
        let x = Tensor::randn(0f32, 1.0, (3, 4096), &cpu)?;
        let w_last = keep.last().unwrap().dequantize(&cpu)?;
        let want = x.matmul(&w_last.t()?)?;
        let mm = QMatMul::from_arc(std::sync::Arc::new(keep.pop().unwrap()))?;
        let got = mm.forward(&x.to_device(&dev)?)?;
        check("q8 matmul on the last placed tensor", rel_err(&got, &want)?, 1e-4, &mut fails);
        drop(mm);
        drop(keep);
        rep("placement: all dropped", &vd);
    }

    // ---------------- dense rhs (mirrored) ----------------
    for (dt, tol) in [(DType::F32, 1e-4f32), (DType::F16, 4e-3), (DType::BF16, 3e-2)] {
        let (n, k) = (2048usize, 2048usize);
        let w = Tensor::randn(0f32, 1.0, (n, k), &cpu)?.to_dtype(dt)?;
        let w_vk = w.to_device(&dev)?;
        for &m in &[1usize, 5, 77] {
            let x = Tensor::randn(0f32, 1.0, (m, k), &cpu)?.to_dtype(dt)?;
            let want = x.to_dtype(DType::F32)?.matmul(&w.to_dtype(DType::F32)?.t()?)?;
            let x_vk = x.to_device(&dev)?;
            for pass in 0..2 {
                let got = x_vk.matmul(&w_vk.t()?)?;
                check(&format!("dense {dt:?} m={m} pass {pass}"), rel_err(&got, &want)?, tol, &mut fails);
            }
        }
        // broadcast like QMatMul::Tensor with a 3d input
        let x3 = Tensor::randn(0f32, 1.0, (2, 3, k), &cpu)?.to_dtype(dt)?;
        let want = x3.to_dtype(DType::F32)?.broadcast_matmul(&w.to_dtype(DType::F32)?.t()?)?;
        let got = x3.to_device(&dev)?.broadcast_matmul(&w_vk.t()?)?;
        check(&format!("dense {dt:?} broadcast 3d"), rel_err(&got, &want)?, tol, &mut fails);
        rep(&format!("dense {dt:?} weights alive"), &vd);
    }
    // ---------------- activation x activation (attention-like, batched, strided) ----------------
    {
        let (b, h, t, d) = (2usize, 8usize, 600usize, 64usize);
        let q = Tensor::randn(0f32, 1.0, (b, h, t, d), &cpu)?;
        let kk = Tensor::randn(0f32, 1.0, (b, h, t, d), &cpu)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, t, d), &cpu)?;
        let want_s = q.matmul(&kk.t()?)?;
        let qv = q.to_device(&dev)?;
        let kv = kk.to_device(&dev)?;
        let vv = v.to_device(&dev)?;
        let got_s = qv.matmul(&kv.t()?)?;
        check("act matmul q@k^T (2,8,600,64)", rel_err(&got_s, &want_s)?, 1e-4, &mut fails);
        let att = (&want_s / 8.0)?;
        let want_o = att.matmul(&v)?;
        let got_o = att.to_device(&dev)?.matmul(&vv)?;
        check("act matmul att@v (2,8,600,600)x(600,64)", rel_err(&got_o, &want_o)?, 1e-4, &mut fails);
        // GQA-style broadcast of k over heads via a stride-0 batch dim
        let k1 = Tensor::randn(0f32, 1.0, (b, 1, t, d), &cpu)?;
        let kb = k1.broadcast_as((b, h, t, d))?;
        let want_g = q.matmul(&kb.contiguous()?.t()?)?;
        let got_g = qv.matmul(&k1.to_device(&dev)?.broadcast_as((b, h, t, d))?.t()?)?;
        check("act matmul broadcast k", rel_err(&got_g, &want_g)?, 1e-4, &mut fails);
        let d0 = candle_core::vulkan_backend::shaders::native_exec_count();
        let tt = time_ms(5, || { let _ = qv.matmul(&kv.t()?)?; Ok(()) })?;
        let d1 = candle_core::vulkan_backend::shaders::native_exec_count();
        let tc = time_ms(3, || { let _ = q.matmul(&kk.t()?)?; Ok(()) })?;
        println!("[VK-GPU] SPEED act q@k^T (2,8,600,64): CPU {tc:8.3} ms | Vulkan GPU {tt:8.3} ms | x{:.1} | gpu dispatches +{}", tc / tt, d1 - d0);
    }

    rep("after correctness tests (all dropped)", &vd);
    vd.trim_memory_pool()?;
    rep("after trim", &vd);

    // ---------------- speed ----------------
    {
        let (n, k) = (6144usize, 2048usize);
        let w = Tensor::randn(0f32, 1.0, (n, k), &cpu)?;
        let q_cpu = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
        let q_vk = QMatMul::from_qtensor(QTensor::quantize(&w.to_device(&dev)?, GgmlDType::Q8_0)?)?;
        for &m in &[1usize, 512] {
            let x = Tensor::randn(0f32, 1.0, (m, k), &cpu)?;
            let xv = x.to_device(&dev)?;
            let it = if m == 1 { 50 } else { 5 };
            let tc = time_ms(it, || { let _ = q_cpu.forward(&x)?; Ok(()) })?;
            let tg = time_ms(it, || { let _ = q_vk.forward(&xv)?; Ok(()) })?;
            println!("[VK-GPU] SPEED q8 {m}x{k} @ {n}x{k}^T : CPU {tc:8.3} ms | Vulkan GPU {tg:8.3} ms | x{:.1}", tc / tg);
        }
        // GEMV scaling: fixed cost vs bytes (lm_head-like at the end)
        for (dt, n) in [(DType::F16, 6144usize), (DType::F16, 16384), (DType::F32, 32768), (DType::F16, 65536)] {
            let k = 2048usize;
            let w = Tensor::randn(0f32, 1.0, (n, k), &cpu)?.to_dtype(dt)?;
            let wv = w.to_device(&dev)?;
            let x = Tensor::randn(0f32, 1.0, (1, k), &cpu)?.to_dtype(dt)?;
            let xv = x.to_device(&dev)?;
            let tc = time_ms(5, || { let _ = x.matmul(&w.t()?)?; Ok(()) })?;
            let tg = time_ms(20, || { let _ = xv.matmul(&wv.t()?)?; Ok(()) })?;
            let mbytes = (n * k * dt.size_in_bytes()) as f64 / 1e6;
            println!(
                "[VK-GPU] SPEED {dt:?} gemv 1x{k} @ {n}x{k}^T ({mbytes:.0} MB): CPU {tc:8.3} ms | Vulkan GPU {tg:8.3} ms ({:.0} GB/s) | x{:.1}",
                mbytes / tg,
                tc / tg
            );
        }
        {
            let (n, k) = (65536usize, 2048usize);
            let w = Tensor::randn(0f32, 1.0, (n, k), &cpu)?;
            let q_vk = QMatMul::from_qtensor(QTensor::quantize(&w.to_device(&dev)?, GgmlDType::Q8_0)?)?;
            let xv = Tensor::randn(0f32, 1.0, (1, k), &dev)?;
            let tg = time_ms(20, || { let _ = q_vk.forward(&xv)?; Ok(()) })?;
            println!("[VK-GPU] SPEED q8 gemv 1x{k} @ {n}x{k}^T (143 MB): Vulkan GPU {tg:8.3} ms ({:.0} GB/s)", 143.0 / tg);
        }
        rep("speed weights alive", &vd);
    }
    rep("speed weights dropped", &vd);
    vd.trim_memory_pool()?;
    rep("end (trimmed)", &vd);
    print!("{}", vd.memory_diagnostics());
    println!("[VK-GPU] RESULT: {} failure(s)", fails);
    Ok(())
}
