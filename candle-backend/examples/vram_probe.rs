use candle_rocm::{DType, Device, Tensor};

fn mb(x: f64) -> f64 { x / 1048576.0 }

fn rep(tag: &str, dev: &Device) {
    if let Device::Rocm(d) = dev {
        let r = d.memory_report().unwrap();
        println!(
            "[PROBE] {:<28} | used(driver) {:8.1} MB | live {:8.1} MB ({} allocs) | peak {:8.1} | pool reserved {:8.1} used {:8.1}",
            tag,
            mb((r.total - r.free) as f64),
            mb(r.live_bytes as f64),
            r.live_allocs,
            mb(r.peak_bytes as f64),
            mb(r.pool_reserved as f64),
            mb(r.pool_used as f64),
        );
    }
}

fn workload(dev: &Device) -> candle_rocm::Result<()> {
    for dt in [DType::F32, DType::F16] {
        for &m in &[1usize, 17, 129, 512, 2347] {
            for &n in &[2048usize, 6144] {
                let a = Tensor::randn(0f32, 1f32, (m, 2048), dev)?.to_dtype(dt)?;
                let b = Tensor::randn(0f32, 1f32, (2048, n), dev)?.to_dtype(dt)?;
                let c = a.matmul(&b)?;
                let _ = c.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            }
        }
    }
    Ok(())
}

fn main() -> candle_rocm::Result<()> {
    let dev = candle_rocm::device(0)?;
    let Device::Rocm(d) = &dev else { panic!() };
    rep("init", &dev);
    for round in 1..=3 {
        let weights: Vec<Tensor> = (0..8)
            .map(|_| Tensor::randn(0f32, 1f32, (2048, 6144), &dev).unwrap().to_dtype(DType::F16).unwrap())
            .collect();
        rep(&format!("r{round} weights allocated"), &dev);
        workload(&dev)?;
        rep(&format!("r{round} after workload"), &dev);
        drop(weights);
        rep(&format!("r{round} weights dropped"), &dev);
        dev.synchronize()?;
        rep(&format!("r{round} after synchronize"), &dev);
        d.trim_memory_pool()?;
        rep(&format!("r{round} after trim"), &dev);
        d.release_cached_resources()?;
        rep(&format!("r{round} after release_cached"), &dev);
    }
    Ok(())
}
