//! Measures where the Vulkan backend really runs: memory placement, whether GPU
//! kernels dispatch, and timings of the op kinds an LLM uses (F32 matmul,
//! F16 matmul, quantized Q4K/Q8_0 matmul), with the GPU kernels off and on.
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::vulkan_backend::shaders;
use candle_core::{DType, Device, Module, Tensor};
use std::time::Instant;

fn mb(x: usize) -> f64 { x as f64 / 1048576.0 }

fn report(tag: &str, d: &candle_core::VulkanDevice) {
    let (free, total) = d.mem_info().unwrap_or((0, 0));
    println!(
        "[VK-PROBE] {tag:<32} | heap(of tensor type) used {:8.1} MB / {:8.1} MB | process alloc {:8.1} MB (pool {:6.1}) | gpu dispatches {}",
        mb(total.saturating_sub(free)), mb(total), mb(d.allocated_bytes()), mb(d.pooled_bytes()),
        shaders::native_exec_count()
    );
}

fn time<F: FnMut() -> candle_core::Result<()>>(iters: usize, mut f: F) -> candle_core::Result<f64> {
    f()?; // warm up (pipeline build)
    let t = Instant::now();
    for _ in 0..iters { f()?; }
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn bench(dev: &Device, label: &str) -> candle_core::Result<()> {
    let vd = if let Device::Vulkan(v) = dev { v.clone() } else { unreachable!() };
    let before = shaders::native_exec_count();
    // decode-like GEMV (1 x 2048 @ 2048 x 6144) and prefill-like GEMM (512 x 2048 @ 2048 x 2048)
    let x1 = Tensor::randn(0f32, 1., (1, 2048), dev)?;
    let xp = Tensor::randn(0f32, 1., (512, 2048), dev)?;
    let w = Tensor::randn(0f32, 1., (2048, 6144), dev)?;
    let w2 = Tensor::randn(0f32, 1., (2048, 2048), dev)?;
    let t_gemv = time(20, || { let _ = x1.matmul(&w)?; dev.synchronize() })?;
    let t_gemm = time(5, || { let _ = xp.matmul(&w2)?; dev.synchronize() })?;
    let x1h = x1.to_dtype(DType::F16)?; let wh = w.to_dtype(DType::F16)?;
    let t_f16 = time(20, || { let _ = x1h.matmul(&wh)?; dev.synchronize() })?;
    let wt = Tensor::randn(0f32, 1., (6144, 2048), dev)?; // (out, in) like a linear layer
    let q4 = QMatMul::from_qtensor(QTensor::quantize(&wt, GgmlDType::Q4K)?)?;
    let q8 = QMatMul::from_qtensor(QTensor::quantize(&wt, GgmlDType::Q8_0)?)?;
    let t_q4 = time(20, || { let _ = q4.forward(&x1)?; dev.synchronize() })?;
    let t_q8 = time(20, || { let _ = q8.forward(&x1)?; dev.synchronize() })?;
    let t_q4p = time(3, || { let _ = q4.forward(&xp)?; dev.synchronize() })?;
    println!(
        "[VK-PROBE] {label:<12} f32 gemv {:7.3} ms | f32 gemm512 {:8.3} ms | f16 gemv {:7.3} ms | q4k gemv {:7.3} ms | q8_0 gemv {:7.3} ms | q4k gemm512 {:8.3} ms | gpu dispatches +{}",
        t_gemv, t_gemm, t_f16, t_q4, t_q8, t_q4p, shaders::native_exec_count() - before
    );
    report(&format!("{label} tensors alive"), &vd);
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let vd = candle_core::VulkanDevice::new(0)?;
    println!("[VK-PROBE] CANDLE_VULKAN_MEMORY={:?} CANDLE_VULKAN_NATIVE={:?}",
        std::env::var("CANDLE_VULKAN_MEMORY").ok(), std::env::var("CANDLE_VULKAN_NATIVE").ok());
    print!("{}", vd.memory_diagnostics());
    let dev = Device::Vulkan(vd.clone());
    report("init", &vd);
    {
        let big = Tensor::zeros((256, 1024, 1024), DType::F32, &dev)?; // 1 GiB
        report("1 GiB tensor allocated", &vd);
        drop(big);
        report("1 GiB tensor dropped", &vd);
        vd.trim_memory_pool()?;
        report("after trim_memory_pool", &vd);
    }
    shaders::set_native_override(Some(false));
    bench(&dev, "kernels OFF")?;
    shaders::set_native_override(Some(true));
    bench(&dev, "kernels ON")?;
    shaders::set_native_override(None);
    vd.trim_memory_pool()?;
    report("end (trimmed)", &vd);
    // CPU reference
    let cpu = Device::Cpu;
    let x1 = Tensor::randn(0f32, 1., (1, 2048), &cpu)?;
    let wt = Tensor::randn(0f32, 1., (6144, 2048), &cpu)?;
    let q4 = QMatMul::from_qtensor(QTensor::quantize(&wt, GgmlDType::Q4K)?)?;
    let w = wt.t()?.contiguous()?;
    let t_cpu = time(20, || { let _ = x1.matmul(&w)?; Ok(()) })?;
    let t_cpu_q4 = time(20, || { let _ = q4.forward(&x1)?; Ok(()) })?;
    println!("[VK-PROBE] CPU device   f32 gemv {:7.3} ms | q4k gemv {:7.3} ms", t_cpu, t_cpu_q4);
    Ok(())
}
