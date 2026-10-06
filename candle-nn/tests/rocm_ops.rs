#![cfg(feature = "rocm")]
use candle::{DType, Device, Result, Tensor, D};

fn rocm() -> Device {
    Device::new_rocm(0).expect("rocm device")
}

fn tol(dtype: DType) -> f64 {
    match dtype {
        DType::F64 => 1e-9,
        DType::F32 => 1e-5,
        DType::F16 => 5e-3,
        DType::BF16 => 3e-2,
        _ => 0.0,
    }
}

fn assert_close(gpu: &Tensor, cpu: &Tensor, what: &str) -> Result<()> {
    assert_eq!(gpu.dims(), cpu.dims(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    assert!(gpu.device().is_rocm(), "{what}: result left the rocm device");
    let t = tol(gpu.dtype());
    let a = gpu.to_device(&Device::Cpu)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
    let b = cpu.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!((x - y).abs() <= t * scale, "{what}: {i}: rocm={x} cpu={y} ({:?})", gpu.dtype());
    }
    Ok(())
}

fn randn(shape: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?.to_dtype(dtype)
}

const FLOATS: [DType; 4] = [DType::F32, DType::F64, DType::F16, DType::BF16];

fn f64_ref(xs: &[&Tensor], dtype: DType, f: impl Fn(&[Tensor]) -> Result<Tensor>) -> Result<Tensor> {
    let xs = xs
        .iter()
        .map(|x| x.to_device(&Device::Cpu)?.to_dtype(DType::F64))
        .collect::<Result<Vec<_>>>()?;
    f(&xs)?.to_dtype(dtype)
}

#[test]
fn sigmoid_softmax() -> Result<()> {
    let dev = rocm();
    for dtype in FLOATS {
        let x = randn(&[3, 5, 70], dtype)?;
        let xg = x.to_device(&dev)?;
        let sig = |v: &[Tensor]| (v[0].neg()?.exp()? + 1.0)?.recip();
        assert_close(&candle_nn::ops::sigmoid(&xg)?, &f64_ref(&[&x], dtype, sig)?, "sigmoid")?;
        assert_close(
            &candle_nn::ops::sigmoid(&xg.transpose(0, 2)?)?,
            &f64_ref(&[&x.transpose(0, 2)?], dtype, sig)?,
            "sigmoid strided",
        )?;
        let sm = |v: &[Tensor]| candle_nn::ops::softmax(&v[0], D::Minus1);
        for cols in [1usize, 7, 64, 1000, 4099] {
            let x = (randn(&[4, cols], dtype)? * 4.0)?;
            let xg = x.to_device(&dev)?;
            assert_close(
                &candle_nn::ops::softmax_last_dim(&xg)?,
                &f64_ref(&[&x], dtype, sm)?,
                "softmax_last_dim",
            )?;
        }
        let x = randn(&[2, 6, 9], dtype)?;
        let xg = x.to_device(&dev)?;
        assert_close(
            &candle_nn::ops::softmax_last_dim(&xg.transpose(1, 2)?)?,
            &f64_ref(&[&x.transpose(1, 2)?], dtype, sm)?,
            "softmax_last_dim strided",
        )?;
    }
    Ok(())
}

#[test]
fn norms() -> Result<()> {
    let dev = rocm();
    let rms = |v: &[Tensor]| candle_nn::ops::rms_norm_slow(&v[0], &v[1], 1e-6);
    let ln = |v: &[Tensor]| candle_nn::ops::layer_norm_slow(&v[0], &v[1], &v[2], 1e-5);
    for dtype in FLOATS {
        for hidden in [8usize, 96, 1024, 2560] {
            let x = randn(&[3, 4, hidden], dtype)?;
            let alpha = (randn(&[hidden], dtype)? + 1.0)?;
            let beta = randn(&[hidden], dtype)?;
            let (xg, ag, bg) = (x.to_device(&dev)?, alpha.to_device(&dev)?, beta.to_device(&dev)?);
            assert_close(
                &candle_nn::ops::rms_norm(&xg, &ag, 1e-6)?,
                &f64_ref(&[&x, &alpha], dtype, rms)?,
                "rms_norm",
            )?;
            assert_close(
                &candle_nn::ops::layer_norm(&xg, &ag, &bg, 1e-5)?,
                &f64_ref(&[&x, &alpha, &beta], dtype, ln)?,
                "layer_norm",
            )?;
            use candle::Module;
            let rms_g = candle_nn::RmsNorm::new(ag.clone(), 1e-6);
            assert_close(&rms_g.forward(&xg)?, &f64_ref(&[&x, &alpha], dtype, rms)?, "RmsNorm module")?;
        }
        let x = randn(&[5, 4, 32], dtype)?;
        let alpha = randn(&[32], dtype)?;
        let xg = x.to_device(&dev)?;
        let ag = alpha.to_device(&dev)?;
        assert_close(
            &candle_nn::ops::rms_norm(&xg.transpose(0, 1)?, &ag, 1e-6)?,
            &f64_ref(&[&x.transpose(0, 1)?, &alpha], dtype, rms)?,
            "rms_norm strided",
        )?;
    }
    Ok(())
}

#[test]
fn rotary() -> Result<()> {
    let dev = rocm();
    for dtype in FLOATS {
        let (b, h, t, d) = (2usize, 3usize, 5usize, 16usize);
        let x = randn(&[b, h, t, d], dtype)?;
        let cos = randn(&[t + 3, d / 2], dtype)?;
        let sin = randn(&[t + 3, d / 2], dtype)?;
        let cos_t = cos.narrow(0, 0, t)?.contiguous()?;
        let sin_t = sin.narrow(0, 0, t)?.contiguous()?;
        let xg = x.to_device(&dev)?;
        let (cg, sg) = (cos.to_device(&dev)?, sin.to_device(&dev)?);
        let (ctg, stg) = (cos_t.to_device(&dev)?, sin_t.to_device(&dev)?);
        assert_close(
            &candle_nn::rotary_emb::rope_i(&xg, &ctg, &stg)?,
            &candle_nn::rotary_emb::rope_i(&x, &cos_t, &sin_t)?,
            "rope_i",
        )?;
        assert_close(
            &candle_nn::rotary_emb::rope(&xg, &ctg, &stg)?,
            &candle_nn::rotary_emb::rope(&x, &cos_t, &sin_t)?,
            "rope",
        )?;
        assert_close(
            &candle_nn::rotary_emb::rope(&xg, &cg, &sg)?,
            &candle_nn::rotary_emb::rope(&x, &cos, &sin)?,
            "rope longer cos",
        )?;
        let xt = randn(&[b, t, h, d], dtype)?;
        let xtg = xt.to_device(&dev)?;
        assert_close(
            &candle_nn::rotary_emb::rope_thd(&xtg, &ctg, &stg)?,
            &candle_nn::rotary_emb::rope_thd(&xt, &cos_t, &sin_t)?,
            "rope_thd",
        )?;
        let cos_b = randn(&[b, t, d / 2], dtype)?;
        let sin_b = randn(&[b, t, d / 2], dtype)?;
        let (cbg, sbg) = (cos_b.to_device(&dev)?, sin_b.to_device(&dev)?);
        assert_close(
            &candle_nn::rotary_emb::rope_i(&xg, &cbg, &sbg)?,
            &candle_nn::rotary_emb::rope_i(&x, &cos_b, &sin_b)?,
            "rope_i batched cs",
        )?;
        assert_close(
            &candle_nn::rotary_emb::rope(&xg, &cbg, &sbg)?,
            &candle_nn::rotary_emb::rope(&x, &cos_b, &sin_b)?,
            "rope batched cs",
        )?;
        assert_close(
            &candle_nn::rotary_emb::rope_thd(&xtg, &cbg, &sbg)?,
            &candle_nn::rotary_emb::rope_thd(&xt, &cos_b, &sin_b)?,
            "rope_thd batched cs",
        )?;
        let x1 = randn(&[1, h, 1, d], dtype)?;
        let c1 = cos.narrow(0, 4, 1)?.contiguous()?;
        let s1 = sin.narrow(0, 4, 1)?.contiguous()?;
        assert_close(
            &candle_nn::rotary_emb::rope(&x1.to_device(&dev)?, &c1.to_device(&dev)?, &s1.to_device(&dev)?)?,
            &candle_nn::rotary_emb::rope(&x1, &c1, &s1)?,
            "rope decode step",
        )?;
    }
    Ok(())
}

#[test]
fn kv_cache_and_linear() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mut cache_g = candle_nn::kv_cache::KvCache::new(2, 16);
        let mut cache_c = candle_nn::kv_cache::KvCache::new(2, 16);
        for step in [5usize, 1, 1, 3] {
            let k = randn(&[1, 2, step, 8], dtype)?;
            let v = randn(&[1, 2, step, 8], dtype)?;
            let (kg, vg) = cache_g.append(&k.to_device(&dev)?, &v.to_device(&dev)?)?;
            let (kc, vc) = cache_c.append(&k, &v)?;
            assert_close(&kg, &kc, "kv k")?;
            assert_close(&vg, &vc, "kv v")?;
        }
        let w = randn(&[12, 8], dtype)?;
        let bias = randn(&[12], dtype)?;
        let lin_c = candle_nn::Linear::new(w.clone(), Some(bias.clone()));
        let lin_g = candle_nn::Linear::new(w.to_device(&dev)?, Some(bias.to_device(&dev)?));
        let x = randn(&[2, 3, 8], dtype)?;
        use candle::Module;
        let yg = lin_g.forward(&x.to_device(&dev)?)?;
        if dtype == DType::F32 {
            assert_close(&yg, &lin_c.forward(&x)?, "linear")?;
        } else {
            let a = yg.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let r = candle_nn::Linear::new(w.to_dtype(DType::F32)?, Some(bias.to_dtype(DType::F32)?))
                .forward(&x.to_dtype(DType::F32)?)?;
            let diff = (a - r)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 0.1, "linear {dtype:?} diff {diff}");
        }
        let emb = candle_nn::Embedding::new(randn(&[10, 4], dtype)?.to_device(&dev)?, 4);
        let ids = Tensor::new(&[[1u32, 9, 0]], &dev)?;
        assert_eq!(emb.forward(&ids)?.dims(), &[1, 3, 4]);
    }
    Ok(())
}
