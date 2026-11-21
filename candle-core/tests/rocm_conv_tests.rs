#![cfg(feature = "rocm")]

use candle_core::{DType, Device, Result, Tensor};

fn det_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed ^ 0x9e3779b97f4a7c15;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((x >> 33) as f32) / ((1u32 << 31) as f32) * 2.0 - 1.0;
            u * scale
        })
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let va = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(va.len(), vb.len(), "shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    let mut m = 0f32;
    for (&x, &y) in va.iter().zip(vb.iter()) {
        assert!(x.is_finite() && y.is_finite(), "non-finite output: {x} vs {y}");
        m = m.max((x - y).abs());
    }
    Ok(m)
}

#[test]
fn rocm_conv2d_standard_and_tiled_matches_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for (dtype, tol) in [(DType::F32, 2e-4f32), (DType::F16, 2e-2), (DType::BF16, 8e-2)] {
        // Candle CPU conv2d uses matmul which has no BF16 kernel on CPU; run CPU oracle in F32 for BF16.
        let cpu_oracle_dtype = if dtype == DType::BF16 { DType::F32 } else { dtype };
        // 1) 1x1 stride-1 pad-0 zero-im2col fastpath
        let x_base = Tensor::from_vec(det_vec(2 * 64 * 17 * 19, 1, 0.5), (2, 64, 17, 19), &Device::Cpu)?.to_dtype(dtype)?;
        let w_base = Tensor::from_vec(det_vec(48 * 64, 2, 0.2), (48, 64, 1, 1), &Device::Cpu)?.to_dtype(dtype)?;
        let y_cpu = x_base.to_dtype(cpu_oracle_dtype)?.conv2d(&w_base.to_dtype(cpu_oracle_dtype)?, 0, 1, 1, 1)?;
        let y_gpu = x_base.to_device(&dev)?.conv2d(&w_base.to_device(&dev)?, 0, 1, 1, 1)?;
        let d = max_abs_diff(&y_cpu, &y_gpu)?;
        assert!(d <= tol, "conv2d 1x1 {dtype:?} diff={d} > {tol}");

        // 2) 3x3 pad=1 stride=1 + stride=2 dilation=2
        let x_base = Tensor::from_vec(det_vec(2 * 32 * 24 * 28, 3, 0.5), (2, 32, 24, 28), &Device::Cpu)?.to_dtype(dtype)?;
        let w_base = Tensor::from_vec(det_vec(40 * 32 * 9, 4, 0.15), (40, 32, 3, 3), &Device::Cpu)?.to_dtype(dtype)?;
        for (pad, stride, dil) in [(1, 1, 1), (1, 2, 1), (2, 1, 2)] {
            let y_cpu = x_base.to_dtype(cpu_oracle_dtype)?.conv2d(&w_base.to_dtype(cpu_oracle_dtype)?, pad, stride, dil, 1)?;
            let y_gpu = x_base.to_device(&dev)?.conv2d(&w_base.to_device(&dev)?, pad, stride, dil, 1)?;
            let d = max_abs_diff(&y_cpu, &y_gpu)?;
            assert!(d <= tol, "conv2d 3x3 p={pad} s={stride} d={dil} {dtype:?} diff={d} > {tol}");
        }
    }

    // 3) Large >64 MiB im2col tiled path (TIPSv2 DPT shape: B=1, C=256, 3x3, 192x144 -> im2col ~254 MiB)
    let x_cpu = Tensor::from_vec(det_vec(1 * 128 * 96 * 96, 5, 0.25), (1, 128, 96, 96), &Device::Cpu)?;
    let w_cpu = Tensor::from_vec(det_vec(64 * 128 * 9, 6, 0.1), (64, 128, 3, 3), &Device::Cpu)?;
    let y_cpu = x_cpu.conv2d(&w_cpu, 1, 1, 1, 1)?;
    let y_gpu = x_cpu.to_device(&dev)?.conv2d(&w_cpu.to_device(&dev)?, 1, 1, 1, 1)?;
    let d = max_abs_diff(&y_cpu, &y_gpu)?;
    assert!(d <= 2e-4, "conv2d tiled F32 diff={d}");
    Ok(())
}

#[test]
fn rocm_depthwise_conv2d_matches_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for (dtype, tol) in [(DType::F32, 2e-5f32), (DType::F16, 1e-2), (DType::BF16, 5e-2)] {
        let cpu_oracle_dtype = if dtype == DType::BF16 { DType::F32 } else { dtype };
        for (c, h, w, s) in [(64, 32, 32, 1), (160, 24, 24, 2), (320, 16, 16, 1)] {
            let x_base = Tensor::from_vec(det_vec(2 * c * h * w, 10 + c as u64, 0.5), (2, c, h, w), &Device::Cpu)?.to_dtype(dtype)?;
            let w_base = Tensor::from_vec(det_vec(c * 9, 20 + c as u64, 0.25), (c, 1, 3, 3), &Device::Cpu)?.to_dtype(dtype)?;
            let y_cpu = x_base.to_dtype(cpu_oracle_dtype)?.conv2d(&w_base.to_dtype(cpu_oracle_dtype)?, 1, s, 1, c)?;
            let y_gpu = x_base.to_device(&dev)?.conv2d(&w_base.to_device(&dev)?, 1, s, 1, c)?;
            let d = max_abs_diff(&y_cpu, &y_gpu)?;
            assert!(d <= tol, "depthwise c={c} s={s} {dtype:?} diff={d} > {tol}");
        }
    }
    Ok(())
}

#[test]
fn rocm_conv_transpose2d_matches_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for (dtype, tol) in [(DType::F32, 2e-4f32), (DType::F16, 2e-2), (DType::BF16, 8e-2)] {
        // TIPSv2 Up(4) k=4 s=4, Up(2) / SAM k=2 s=2, and overlapping k=3 s=2 p=1 op=1
        for (c_in, c_out, k, p, op, s) in [(32, 24, 4, 0, 0, 4), (48, 32, 2, 0, 0, 2), (24, 16, 3, 1, 1, 2)] {
            let x_cpu = Tensor::from_vec(det_vec(2 * c_in * 12 * 15, 30 + k as u64, 0.4), (2, c_in, 12, 15), &Device::Cpu)?.to_dtype(dtype)?;
            let w_cpu = Tensor::from_vec(det_vec(c_in * c_out * k * k, 40 + k as u64, 0.15), (c_in, c_out, k, k), &Device::Cpu)?.to_dtype(dtype)?;
            let y_cpu = x_cpu.conv_transpose2d(&w_cpu, p, op, s, 1)?;
            let y_gpu = x_cpu.to_device(&dev)?.conv_transpose2d(&w_cpu.to_device(&dev)?, p, op, s, 1)?;
            let d = max_abs_diff(&y_cpu, &y_gpu)?;
            assert!(d <= tol, "conv_transpose2d k={k} s={s} p={p} op={op} {dtype:?} diff={d} > {tol}");
        }
    }
    Ok(())
}

#[test]
fn rocm_upsample2d_matches_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for (dtype, tol) in [(DType::F32, 1e-5f32), (DType::F16, 2e-3), (DType::BF16, 1e-2)] {
        let x_cpu = Tensor::from_vec(det_vec(2 * 16 * 12 * 18, 50, 1.0), (2, 16, 12, 18), &Device::Cpu)?.to_dtype(dtype)?;
        let x_gpu = x_cpu.to_device(&dev)?;

        let yn_cpu = x_cpu.upsample_nearest2d(24, 36)?;
        let yn_gpu = x_gpu.upsample_nearest2d(24, 36)?;
        let dn = max_abs_diff(&yn_cpu, &yn_gpu)?;
        assert!(dn <= tol, "upsample_nearest2d {dtype:?} diff={dn} > {tol}");

        for align in [false, true] {
            for (oh, ow) in [(24, 36), (33, 47), (12, 18)] {
                let yb_cpu = x_cpu.upsample_bilinear2d(oh, ow, align)?;
                let yb_gpu = x_gpu.upsample_bilinear2d(oh, ow, align)?;
                let db = max_abs_diff(&yb_cpu, &yb_gpu)?;
                assert!(db <= tol, "upsample_bilinear2d align={align} ({oh},{ow}) {dtype:?} diff={db} > {tol}");
            }
        }
    }
    Ok(())
}
