//! ROCm VRAM release probe that mimics an LLM workload (BF16 weights, prefill +
//! decode with rms_norm / rope / attention / softmax / silu, KV cache growth via cat,
//! plus a burst of small allocations) and reports, after every step:
//!   OS     = what Windows charges to this process (DXGI, the Task Manager number)
//!   hip    = total - free from hipMemGetInfo
//!   live   = bytes still owned by live tensors
//!   pool   = stream-ordered pool reserved / used
#[path = "../../tools/dxgi_usage.rs"]
mod dxgi_usage;

use candle_rocm::{DType, Device, Tensor};

fn mb(x: f64) -> f64 {
    x / 1048576.0
}

fn rep(tag: &str, dev: &Device) {
    let Device::Rocm(d) = dev else { return };
    let r = match d.memory_report() {
        Ok(r) => r,
        Err(e) => {
            println!("[PROBE2] {tag}: memory_report failed: {e}");
            return;
        }
    };
    let os = dxgi_usage::process_vram()
        .map(|v| format!("{:8.1}", mb(v.local_usage as f64)))
        .unwrap_or_else(|| "     n/a".into());
    let top = d
        .live_size_histogram(4)
        .iter()
        .map(|(b, c)| format!("{:.2}MBx{}", mb(*b as f64), c))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "[PROBE2] {tag:<30} | OS {os} MB | hip {:8.1} | live {:8.1} ({:5}) | pool {:7.1}/{:7.1} | top {top}",
        mb((r.total - r.free) as f64),
        mb(r.live_bytes as f64),
        r.live_allocs,
        mb(r.pool_reserved as f64),
        mb(r.pool_used as f64),
    );
}

const H: usize = 2048; // hidden
const NH: usize = 16; // heads
const HD: usize = 128; // head dim
const NKV: usize = 4; // kv heads (GQA)
const FF: usize = 6144; // mlp
const LAYERS: usize = 12;

struct Layer {
    ln1: Tensor,
    ln2: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

fn w(dev: &Device, r: usize, c: usize) -> candle_rocm::Result<Tensor> {
    (Tensor::randn(0f32, 0.02, (r, c), dev)?).to_dtype(DType::BF16)
}

fn load_layers(dev: &Device) -> candle_rocm::Result<Vec<Layer>> {
    (0..LAYERS)
        .map(|_| {
            Ok(Layer {
                ln1: Tensor::ones(H, DType::BF16, dev)?,
                ln2: Tensor::ones(H, DType::BF16, dev)?,
                q: w(dev, NH * HD, H)?,
                k: w(dev, NKV * HD, H)?,
                v: w(dev, NKV * HD, H)?,
                o: w(dev, H, NH * HD)?,
                gate: w(dev, FF, H)?,
                up: w(dev, FF, H)?,
                down: w(dev, H, FF)?,
            })
        })
        .collect()
}

fn rope_tables(dev: &Device, t0: usize, t: usize) -> candle_rocm::Result<(Tensor, Tensor)> {
    let inv: Vec<f32> = (0..HD / 2).map(|i| 1f32 / 1e6f32.powf(2.0 * i as f32 / HD as f32)).collect();
    let inv = Tensor::from_vec(inv, (1, HD / 2), dev)?;
    let pos = Tensor::arange(t0 as u32, (t0 + t) as u32, dev)?.to_dtype(DType::F32)?.reshape((t, 1))?;
    let f = pos.broadcast_mul(&inv)?;
    Ok((f.cos()?.to_dtype(DType::BF16)?, f.sin()?.to_dtype(DType::BF16)?))
}

/// One transformer block; `kv` holds the cache of this layer and grows with `cat`.
fn block(
    l: &Layer,
    x: &Tensor,
    kv: &mut Option<(Tensor, Tensor)>,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_rocm::Result<Tensor> {
    let (b, t, _) = x.dims3()?;
    let h = candle_nn::ops::rms_norm(x, &l.ln1, 1e-6)?;
    let q = h.broadcast_matmul(&l.q.t()?)?.reshape((b, t, NH, HD))?.transpose(1, 2)?.contiguous()?;
    let k = h.broadcast_matmul(&l.k.t()?)?.reshape((b, t, NKV, HD))?.transpose(1, 2)?.contiguous()?;
    let v = h.broadcast_matmul(&l.v.t()?)?.reshape((b, t, NKV, HD))?.transpose(1, 2)?.contiguous()?;
    let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
    let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
    let (k, v) = match kv.take() {
        Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
        None => (k, v),
    };
    *kv = Some((k.clone(), v.clone()));
    let rep = NH / NKV;
    let k = k.repeat((1, rep, 1, 1))?;
    let v = v.repeat((1, rep, 1, 1))?;
    let att = (q.matmul(&k.t()?)? * (1.0 / (HD as f64).sqrt()))?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, NH * HD))?;
    let x = (x + y.broadcast_matmul(&l.o.t()?)?)?;
    let h = candle_nn::ops::rms_norm(&x, &l.ln2, 1e-6)?;
    let g = h.broadcast_matmul(&l.gate.t()?)?.silu()?;
    let u = h.broadcast_matmul(&l.up.t()?)?;
    let m = (g * u)?.broadcast_matmul(&l.down.t()?)?;
    x + m
}

fn run_model(dev: &Device, layers: &[Layer], prompt: usize, decode: usize) -> candle_rocm::Result<(f32, f64, f64)> {
    let mut kvs: Vec<Option<(Tensor, Tensor)>> = (0..layers.len()).map(|_| None).collect();
    let t0 = std::time::Instant::now();
    let mut x = Tensor::randn(0f32, 1.0, (1, prompt, H), dev)?.to_dtype(DType::BF16)?;
    let (cos, sin) = rope_tables(dev, 0, prompt)?;
    for (l, kv) in layers.iter().zip(kvs.iter_mut()) {
        x = block(l, &x, kv, &cos, &sin)?;
    }
    let mut last = x.narrow(1, prompt - 1, 1)?;
    dev.synchronize()?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    for s in 0..decode {
        let (cos, sin) = rope_tables(dev, prompt + s, 1)?;
        let mut y = last.clone();
        for (l, kv) in layers.iter().zip(kvs.iter_mut()) {
            y = block(l, &y, kv, &cos, &sin)?;
        }
        last = y;
        // like the app's generation loops: a device sync per token
        dev.synchronize()?;
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1e3 / decode.max(1) as f64;
    Ok((last.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?, prefill_ms, decode_ms))
}

fn small_alloc_storm(dev: &Device) -> candle_rocm::Result<()> {
    let mut v = Vec::with_capacity(20000);
    for i in 0..20000usize {
        let n = 1024 + (i % 61) * 512; // 4 KB .. 128 KB of f32
        v.push(Tensor::zeros(n, DType::F32, dev)?);
    }
    let _ = v.last().map(|t| t.sum_all());
    drop(v);
    Ok(())
}

fn main() -> candle_rocm::Result<()> {
    if std::env::var("PROBE_SET_ENV_IN_PROCESS").is_ok() {
        // set before the first HIP call, from inside the process
        std::env::set_var("GPU_RESOURCE_CACHE_SIZE", "0");
        println!("[PROBE2] GPU_RESOURCE_CACHE_SIZE=0 set in-process via std::env::set_var");
    }
    println!(
        "[PROBE2] env: CANDLE_ROCM_ASYNC_ALLOC={:?} GPU_RESOURCE_CACHE_SIZE={:?} GPU_MAX_SUBALLOC_SIZE={:?}",
        std::env::var("CANDLE_ROCM_ASYNC_ALLOC").ok(),
        std::env::var("GPU_RESOURCE_CACHE_SIZE").ok(),
        std::env::var("GPU_MAX_SUBALLOC_SIZE").ok()
    );
    let os0 = dxgi_usage::process_vram();
    println!("[PROBE2] OS usage before HIP init: {:?}", os0.map(|v| mb(v.local_usage as f64)));
    let dev = candle_rocm::device(0)?;
    let Device::Rocm(d) = &dev else { unreachable!() };
    rep("init", &dev);
    for round in 1..=2 {
        let layers = load_layers(&dev)?;
        rep(&format!("r{round} weights loaded"), &dev);
        match run_model(&dev, &layers, 1500, 48) {
            Ok((s, p, d)) => println!("[PROBE2] r{round} model ok (checksum {s:.3}) | prefill {p:.1} ms | decode {d:.2} ms/token (sync each)"),
            Err(e) => println!("[PROBE2] r{round} model FAILED: {e}"),
        }
        rep(&format!("r{round} after prefill+decode"), &dev);
        drop(layers);
        rep(&format!("r{round} weights dropped"), &dev);
        dev.synchronize()?;
        rep(&format!("r{round} synchronize (trim)"), &dev);
        d.release_cached_resources()?;
        rep(&format!("r{round} release_cached"), &dev);
        std::thread::sleep(std::time::Duration::from_secs(3));
        rep(&format!("r{round} +3s"), &dev);
        if let Err(e) = small_alloc_storm(&dev) {
            println!("[PROBE2] small alloc storm failed: {e}");
        }
        rep(&format!("r{round} small-alloc storm freed"), &dev);
        d.release_cached_resources()?;
        std::thread::sleep(std::time::Duration::from_secs(2));
        rep(&format!("r{round} storm released +2s"), &dev);
    }
    Ok(())
}
