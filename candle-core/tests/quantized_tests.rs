use candle_core::{
    bail,
    quantized::{self, GgmlDType},
    test_device,
    test_utils::to_vec2_round,
    DType, Device, IndexOp, Module, Result, Tensor, Var,
};
use quantized::{k_quants, GgmlType};
use rand::prelude::*;
use std::borrow::Cow;

const GGML_TEST_SIZE: usize = 32 * 128;

const GGML_MAX_QUANTIZATION_TOTAL_ERROR: f32 = 0.002;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS: f32 = 0.0040;
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

fn test_matmul(
    device: &Device,
    (b, m, n, k): (usize, usize, usize, usize),
    dtype: GgmlDType,
) -> Result<()> {
    if (device.is_cuda() || device.is_metal())
        && (dtype == GgmlDType::Q8_1 || dtype == GgmlDType::Q8K)
    {
        return Ok(());
    }

    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device)?;
    let rhs = Tensor::from_slice(&rhs, (k, n), device)?;
    let mm = lhs.matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;
    let error = error / (b * m * n) as f32;
    assert!(
        error <= 0.02,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_matmul_mm() -> Result<()> {
    let dtype = GgmlDType::Q8_0;
    let device = Device::new_metal(0)?;

    let m = 32;
    let n = 32;
    let k = 32;
    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (1, 1, k, n), &device)?.repeat((5, 20, 1, 1))?;
    let mm = lhs.broadcast_matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&lhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&rhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;

    let error = error / res.elem_count() as f32;
    assert!(
        error <= 0.001,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

fn quantized_matmul(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0,
            341876.0, 994283.0, 1655709.0, 2301518.0
        ]
    );
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        mm.to_vec2::<f32>()?,
        &[
            [85344.0, 214368.0, 343392.0, 472416.0],
            [214368.0, 605536.0, 996704.0, 1387872.0],
            [343392.0, 996704.0, 1650016.0, 2303328.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84946.0, 214126.0, 344757.0, 473798.0],
                [213458.0, 604350.0, 1000469.0, 1387990.0],
                [341970.0, 994574.0, 1656181.0, 2302182.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84866.0, 214045.0, 344676.0, 473707.0],
                [213425.0, 604313.0, 1000431.0, 1387960.0],
                [342030.0, 994630.0, 1656248.0, 2302250.0]
            ]
        ),
        Device::OpenCl(_) | Device::Cpu | Device::Vulkan(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [85120.0, 214562.0, 345455.0, 474748.0],
                [213475.0, 604465.0, 1000686.0, 1388317.0],
                [341876.0, 994283.0, 1655709.0, 2301518.0]
            ]
        ),
        Device::Rocm(_) => {
            let w = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?.dequantize(device)?;
            let expected = lhs.matmul(&w.t()?)?;
            let diff = (&res - &expected)?.abs()?.max_all()?.to_vec0::<f32>()?;
            assert!(diff < 1.0, "rocm qmatmul diff {diff}");
        }
    }
    test_matmul(device, (1, 3, 4, 256), GgmlDType::Q4_0)?;
    Ok(())
}

fn quantized_matmul_neg(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k))
        .map(|v| v as f32 - (m * k) as f32 / 2.0)
        .collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..k * n)
        .map(|v| v as f32 - (k * n) as f32 / 3.0)
        .collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0,
            -196472.0, 63012.0, 324585.0, 587902.0
        ]
    );
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        to_vec2_round(&mm, 0)?,
        &[
            [244064.0, -20128.0, -284320.0, -548512.0],
            [23563.0, 21515.0, 19467.0, 17419.0],
            [-196939.0, 63157.0, 323253.0, 583349.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243659.0, -19716.0, -285444.0, -550439.0],
                [23779.0, 21653.0, 19404.0, 18349.0],
                [-196101.0, 63021.0, 324252.0, 587137.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243740.0, -19762.0, -285476.0, -550498.0],
                [23774.0, 21645.0, 19395.0, 18364.0],
                [-196045.0, 63030.0, 324120.0, 587079.0]
            ]
        ),
        Device::OpenCl(_) | Device::Cpu | Device::Vulkan(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243524.0, -19596.0, -285051.0, -549815.0],
                [23777.0, 21651.0, 19398.0, 18367.0],
                [-196472.0, 63012.0, 324585.0, 587902.0]
            ]
        ),
        Device::Rocm(_) => {
            let w = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?.dequantize(device)?;
            let expected = lhs.matmul(&w.t()?)?;
            let diff = (&res - &expected)?.abs()?.max_all()?.to_vec0::<f32>()?;
            assert!(diff < 1.0, "rocm qmatmul diff {diff}");
        }
    }
    let lhs2 = Tensor::stack(&[&lhs, &lhs], 0)?;
    let res2 = matmul.forward(&lhs2)?;
    let res2 = res2.i(1)?;
    let diff = (&res - res2)?.abs()?.mean_all()?.to_vec0::<f32>()? / res.elem_count() as f32;
    if device.is_cuda() {
        assert!(diff < 0.1);
    } else {
        assert!(diff < 0.96);
    }
    Ok(())
}

fn qmm_batch(dev: &Device) -> Result<()> {
    let (lhs, rhs, _mm) = get_random_tensors(2, 256, 6, dev)?;
    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;
    assert_eq!(mm.shape().dims(), [2, 6]);
    let lhs2 = Tensor::cat(&[&lhs, &lhs], 0)?;
    let mm2 = rhs.forward(&lhs2)?;
    assert_eq!(mm2.shape().dims(), [4, 6]);
    let diff2 = (mm2.i(2..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff2, 0.0);
    let lhs3 = Tensor::cat(&[&lhs2, &lhs], 0)?;
    let mm3 = rhs.forward(&lhs3)?;
    assert_eq!(mm3.shape().dims(), [6, 6]);
    let diff3 = (mm3.i(2..4)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff3, 0.0);
    let diff3 = (mm3.i(4..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff3, 0.0);
    let lhs4 = Tensor::cat(&[&lhs3, &lhs3], 0)?;
    let mm4 = rhs.forward(&lhs4)?;
    assert_eq!(mm4.shape().dims(), [12, 6]);
    let diff4 = (mm4.i(..6)? - &mm3)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        // We use different fused kernels (MMVQ for batch<=8, MMQ for batch>8) on CUDA which accumulate differently than dequantize-then-matmul.
        // This can lead to small numerical differences especially for low-bit quants.
        assert!(0. < diff4 && diff4 < 0.5)
    } else if dev.is_rocm() {
        assert!(diff4 < 0.5)
    } else {
        assert_eq!(diff4, 0.0)
    };
    let diff4 = (mm4.i(6..)? - &mm4.i(..6)?)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff4, 0.0);
    Ok(())
}

test_device!(quantized_matmul, qmm_cpu, qmm_cuda, qmm_metal);
test_device!(quantized_matmul_neg, qmm_n_cpu, qmm_n_cuda, qmm_n_metal);
test_device!(qmm_batch, qmm_b_cpu, qmm_b_cuda, qmm_b_metal);

fn embedding_weight(device: &Device) -> Result<Tensor> {
    let values = (0..(8 * 256))
        .map(|i| {
            let x = i as f32;
            (x * 0.003).sin() * 0.5 + (x * 0.007).cos() * 0.25
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (8, 256), device)
}

fn assert_embedding_close(dtype: GgmlDType, a: &Tensor, b: &Tensor, tol: f32) -> Result<()> {
    let a = a.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (a, b)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{dtype:?} embedding mismatch at {idx}: {a} != {b}"
        );
    }
    Ok(())
}

fn run_quantized_embedding(device: &Device, dtype: GgmlDType, tol: f32) -> Result<()> {
    let w = embedding_weight(device)?;
    let ids = Tensor::from_vec(vec![3u32, 1, 3, 7], (2, 2), device)?;
    let q = quantized::QTensor::quantize(&w, dtype)?;
    let got = q.embedding(&ids)?;
    let expected = q
        .dequantize(device)?
        .index_select(&ids.flatten_all()?, 0)?
        .reshape((2, 2, 256))?;
    assert_embedding_close(dtype, &got, &expected, tol)
}

#[test]
fn quantized_embedding_cpu() -> Result<()> {
    let device = Device::Cpu;
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ] {
        run_quantized_embedding(&device, dtype, 1e-6)?;
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn quantized_embedding_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        run_quantized_embedding(&device, dtype, 1e-3)?;
    }
    Ok(())
}

fn quantize_q4_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();

    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        dst.to_vec1::<f32>()?,
        &[
            -0.0, -0.0, 3.875, 3.875, 3.875, 3.875, 7.75, 7.75, 7.75, 7.75, 11.625, 11.625, 11.625,
            11.625, 15.5, 15.5, 15.5, 15.5, 19.375, 19.375, 19.375, 19.375, 23.25, 23.25, 23.25,
            23.25, 27.125, 27.125, 27.125, 27.125, 31.0, 31.0, 31.5, 31.5, 31.5, 31.5, 39.375,
            39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 47.25, 47.25, 47.25, 47.25,
            47.25, 47.25, 47.25, 47.25, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125,
            55.125, 63.0, 63.0, 63.0, 63.0, 59.375, 59.375, 71.25, 71.25, 71.25, 71.25, 71.25,
            71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 83.125, 83.125, 83.125, 83.125,
            83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 95.0, 95.0, 95.0, 95.0,
            95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0,
            127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q4_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 0.0, 2.066, 2.066, 4.133, 4.133, 6.199, 6.199, 8.266, 8.266, 10.332, 10.332,
            12.398, 12.398, 14.465, 14.465, 16.531, 16.531, 18.598, 18.598, 20.664, 20.664, 22.73,
            22.73, 24.797, 24.797, 26.863, 26.863, 28.93, 28.93, 30.996, 30.996, 32.0, 32.0,
            34.066, 34.066, 36.133, 36.133, 38.199, 38.199, 40.266, 40.266, 42.332, 42.332, 44.398,
            44.398, 46.465, 46.465, 48.531, 48.531, 50.598, 50.598, 52.664, 52.664, 54.73, 54.73,
            56.797, 56.797, 58.863, 58.863, 60.93, 60.93, 62.996, 62.996, 64.0, 64.0, 66.066,
            66.066, 68.133, 68.133, 70.199, 70.199, 72.266, 72.266, 74.332, 74.332, 76.398, 76.398,
            78.465, 78.465, 80.531, 80.531, 82.598, 82.598, 84.664, 84.664, 86.73, 86.73, 88.797,
            88.797, 90.863, 90.863, 92.93, 92.93, 94.996, 94.996, 96.0, 96.0, 98.066, 98.066,
            100.133, 100.133, 102.199, 102.199, 104.266, 104.266, 106.332, 106.332, 108.398,
            108.398, 110.465, 110.465, 112.531, 112.531, 114.598, 114.598, 116.664, 116.664,
            118.73, 118.73, 120.797, 120.797, 122.863, 122.863, 124.93, 124.93, 126.996, 126.996
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            -0.0, 1.938, 1.938, 3.875, 3.875, 5.813, 5.813, 7.75, 7.75, 9.688, 9.688, 11.625,
            11.625, 13.563, 13.563, 15.5, 15.5, 17.438, 17.438, 19.375, 19.375, 21.313, 21.313,
            23.25, 23.25, 25.188, 25.188, 27.125, 27.125, 29.063, 29.063, 31.0, 31.5, 31.5, 35.438,
            35.438, 35.438, 35.438, 39.375, 39.375, 39.375, 39.375, 43.313, 43.313, 43.313, 43.313,
            47.25, 47.25, 47.25, 47.25, 51.188, 51.188, 51.188, 51.188, 55.125, 55.125, 55.125,
            55.125, 59.063, 59.063, 59.063, 59.063, 63.0, 63.0, 65.313, 65.313, 65.313, 65.313,
            65.313, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 77.188, 77.188, 77.188, 77.188,
            77.188, 77.188, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 89.063, 89.063, 89.063,
            89.063, 89.063, 89.063, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 103.188, 103.188,
            103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 119.063, 119.063, 119.063, 119.063,
            119.063, 119.063, 119.063, 119.063, 127.0, 127.0, 127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
            86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn get_test_vector2(bound: f32, size: usize, device: &Device) -> Result<Tensor> {
    assert!(
        size.is_multiple_of(crate::quantized::k_quants::QK_K),
        "size must be a multiple of {}",
        crate::quantized::k_quants::QK_K
    );

    let src = (0..size)
        .map(|v| (v as f32 - size as f32 / 2.) * bound / (size as f32 / 2.))
        .collect::<Vec<_>>();
    assert_eq!([src[0], src[size / 2]], [-bound, 0.0]);
    Tensor::from_vec(src, (size,), device)
}

/// Round a vector
fn round_vector(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>()
}

fn compare_with_error(values: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (value, expected_value)) in values.iter().zip(expected.iter()).enumerate() {
        let difference = (value - expected_value).abs();

        assert!(
            difference < tolerance,
            "Error at index {i}: value = {value}, expected = {expected_value}. Difference = {difference} exceeds tolerance = {tolerance}."
        );
    }
}

/// Creates a vector similar to the ones used in GGML unit tests:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// Calculates the root mean square error between two vectors
fn calculate_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    sum / a.len() as f32
}

/// Similar to the GGML quantization unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L43-L50
fn ggml_quantization_error_test(dtype: GgmlDType, device: &Device, max_error: f32) -> Result<()> {
    let src = create_ggml_like_vector(0.0);
    let src = Tensor::from_slice(&src, (GGML_TEST_SIZE,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    let error = calculate_rmse(&src.to_vec1::<f32>()?, &dst.to_vec1::<f32>()?);
    if error > max_error {
        bail!(
            "Quantization error {} exceeds max error {}",
            error,
            max_error
        );
    }
    Ok(())
}

#[test]
fn imatrix_quantize_q6k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q6K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q6K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q5k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q5K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q5K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q4k() -> Result<()> {
    // let data =
    //     quantized::imatrix_file::load_imatrix("../Llama-3.2-3B-Instruct.imatrix").unwrap();
    // for (name, weights) in &data {
    //     println!("{name}, {} elems", weights.len());
    // }
    // dbg!(&data["blk.0.attn_q.weight"].len());

    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q4K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q4K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q3k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q3K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q3K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q2k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q2K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q2K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

fn quantize_q2k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q2K;

    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.1);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.499, -0.366, -0.249, 0.0, 0.295, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 6.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS)?;
    Ok(())
}

fn quantize_q3k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q3K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.03);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.493, -0.37, -0.243, -0.0, 0.292, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 3.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS)?;
    Ok(())
}

fn quantize_q4k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q4K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.017);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.288, 0.498]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 4.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q5K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.009);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.279, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q6k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q6K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.497, -0.372, -0.25, -0.0, 0.284, 0.5]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q8k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q8K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.374, -0.25, -0.0, 0.283, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 0.6);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

test_device!(
    quantize_q4_0,
    quantize_q4_0_cpu,
    quantize_q4_0_cuda,
    quantize_q4_0_metal
);
test_device!(
    quantize_q4_1,
    quantize_q4_1_cpu,
    quantize_q4_1_cuda,
    quantize_q4_1_metal
);
test_device!(
    quantize_q5_0,
    quantize_q5_0_cpu,
    quantize_q5_0_cuda,
    quantize_q5_0_metal
);
test_device!(
    quantize_q5_1,
    quantize_q5_1_cpu,
    quantize_q5_1_cuda,
    quantize_q5_1_metal
);
test_device!(
    quantize_q2k,
    quantize_q2k_cpu,
    quantize_q2k_cuda,
    quantize_q2k_metal
);
test_device!(
    quantize_q3k,
    quantize_q3k_cpu,
    quantize_q3k_cuda,
    quantize_q3k_metal
);
test_device!(
    quantize_q4k,
    quantize_q4k_cpu,
    quantize_q4k_cuda,
    quantize_q4k_metal
);
test_device!(
    quantize_q5k,
    quantize_q5k_cpu,
    quantize_q5k_cuda,
    quantize_q5k_metal
);
test_device!(
    quantize_q6k,
    quantize_q6k_cpu,
    quantize_q6k_cuda,
    quantize_q6k_metal
);
test_device!(
    quantize_q8k,
    quantize_q8k_cpu,
    quantize_q8k_cuda,
    quantize_q8k_metal
);

/// Very simple dot product implementation
fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns the error achieved by the GGML matmul unit test.
fn ggml_reference_matmul_error(dtype: GgmlDType) -> Result<f32> {
    let err = match dtype {
        GgmlDType::F32 => 0.000000,
        GgmlDType::F16 => 0.000010,
        GgmlDType::BF16 => 0.000200,
        GgmlDType::Q2K => 0.004086,
        GgmlDType::Q3K => 0.017,
        GgmlDType::Q4K => 0.002425,
        GgmlDType::Q5K => 0.000740,
        GgmlDType::Q6K => 0.000952,
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4_1 => 0.008,
        GgmlDType::Q5_0 => 0.001353,
        GgmlDType::Q5_1 => 0.00149,
        GgmlDType::Q8_0 => 0.000092,
        GgmlDType::Q8_1 => 0.000092,

        // Not from the ggml repo.
        GgmlDType::Q8K => 0.00065,
    };
    Ok(err)
}

/// Similar to the GGML matmul unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
fn ggml_matmul_error_test<T: GgmlType>() -> Result<()> {
    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 1.0)?;
    // Another example that is more likely to trigger the overflow reported in #1526
    let a = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    let b = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 2.0)?;
    Ok(())
}

fn ggml_matmul_error_test_<T: GgmlType>(a: &[f32], b: &[f32], err_m: f32) -> Result<()> {
    let length = a.len();

    let mut a_quant = vec![T::zeros(); length / T::BLCK_SIZE];
    let mut b_quant = vec![T::VecDotType::zeros(); length / T::VecDotType::BLCK_SIZE];
    T::from_float(a, &mut a_quant);
    T::VecDotType::from_float(b, &mut b_quant);

    let result = T::vec_dot(length, &a_quant, &b_quant);
    let result_unopt = T::vec_dot_unopt(length, &a_quant, &b_quant);

    if (result - result_unopt).abs() / length as f32 > 1e-6 {
        bail!(
            "the opt and unopt vec-dot returned different values, opt: {result} vs unopt: {result_unopt}"
        )
    }

    let mut dst = vec![0.0f32; 1];
    crate::k_quants::matmul((1, length, 1), b, &a_quant, &mut dst)?;
    let result_matmul = dst[0];

    if (result_matmul - result).abs() / length as f32 > 1e-6 {
        bail!(
            "calling matmul vs calling vec-dot directly returned different values, matmul: {result_matmul} vs vec-dot: {result}"
        )
    }

    let reference_result = vec_dot_reference(a, b);

    let verify_result = |result: f32, source: &str| {
        let error = (result - reference_result).abs() / length as f32;
        let ggml_error = ggml_reference_matmul_error(T::DTYPE)? * err_m;
        if !error.is_finite() || error > GGML_MAX_DOT_PRODUCT_ERROR {
            bail!("Dot product with dtype {:?} error {error} exceeds max error {GGML_MAX_DOT_PRODUCT_ERROR}. Source: {source}", T::DTYPE);
        }
        // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
        // => we use a slightly higher error threshold
        const ERROR_LENIENCY: f32 = 0.00001;
        if error - ERROR_LENIENCY > ggml_error {
            bail!(
                "Dot product with dtype {:?} error {error} exceeds ggml reference error {ggml_error}. Source: {source}",
                T::DTYPE,
            );
        }
        Ok(())
    };

    verify_result(result, "vec-dot")?;
    verify_result(result_matmul, "matmul")?;
    Ok(())
}

#[test]
fn quantized_mm() -> Result<()> {
    ggml_matmul_error_test::<f32>()?;
    ggml_matmul_error_test::<half::f16>()?;
    ggml_matmul_error_test::<half::bf16>()?;
    ggml_matmul_error_test::<k_quants::BlockQ4_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ4_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_1>()?;
    Ok(())
}

/// generates random tensors of size `m x k` and `n x k` and calculates their expected matrix multiplication result.
fn get_random_tensors(
    m: usize,
    k: usize,
    n: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(314159265358979);

    let lhs = (0..m * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();
    let rhs = (0..n * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_vec(lhs, (m, k), device)?;
    let rhs = Tensor::from_vec(rhs, (n, k), device)?;

    let mm = lhs.matmul(&rhs.t()?)?;
    Ok((lhs, rhs, mm))
}

#[macro_export]
macro_rules! quantized_matmul {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $fn_name_cpu: ident, $fn_name_cuda: ident, $fn_name_metal: ident, $dtype: expr) => {
        fn $fn_name(device: &Device) -> Result<()> {
            test_matmul(device, (1, 3, 4, 256), $dtype)?;
            Ok(())
        }

        test_device!($fn_name, $fn_name_cpu, $fn_name_cuda, $fn_name_metal);
    };
}

quantized_matmul!(
    quantized_matmul_q4_0_bis,
    quantized_matmul_q4_0_cpu,
    quantized_matmul_q4_0_cuda,
    quantized_matmul_q4_0_metal,
    GgmlDType::Q4_0
);
quantized_matmul!(
    quantized_matmul_q4_1_bis,
    quantized_matmul_q4_1_cpu,
    quantized_matmul_q4_1_cuda,
    quantized_matmul_q4_1_metal,
    GgmlDType::Q4_1
);
quantized_matmul!(
    quantized_matmul_q5_0_bis,
    quantized_matmul_q5_0_cpu,
    quantized_matmul_q5_0_cuda,
    quantized_matmul_q5_0_metal,
    GgmlDType::Q5_0
);
quantized_matmul!(
    quantized_matmul_q5_1_bis,
    quantized_matmul_q5_1_cpu,
    quantized_matmul_q5_1_cuda,
    quantized_matmul_q5_1_metal,
    GgmlDType::Q5_1
);
quantized_matmul!(
    quantized_matmul_q8_0_bis,
    quantized_matmul_q8_0_cpu,
    quantized_matmul_q8_0_cuda,
    quantized_matmul_q8_0_metal,
    GgmlDType::Q8_0
);
quantized_matmul!(
    quantized_matmul_q8_1_bis,
    quantized_matmul_q8_1_cpu,
    quantized_matmul_q8_1_cuda,
    quantized_matmul_q8_1_metal,
    GgmlDType::Q8_1
);
quantized_matmul!(
    quantized_matmul_q2k_bis,
    quantized_matmul_q2k_cpu,
    quantized_matmul_q2k_cuda,
    quantized_matmul_q2k_metal,
    GgmlDType::Q2K
);
quantized_matmul!(
    quantized_matmul_q3k_bis,
    quantized_matmul_q3k_cpu,
    quantized_matmul_q3k_cuda,
    quantized_matmul_q3k_metal,
    GgmlDType::Q3K
);
quantized_matmul!(
    quantized_matmul_q4k_bis,
    quantized_matmul_q4k_cpu,
    quantized_matmul_q4k_cuda,
    quantized_matmul_q4k_metal,
    GgmlDType::Q4K
);
quantized_matmul!(
    quantized_matmul_q5k_bis,
    quantized_matmul_q5k_cpu,
    quantized_matmul_q5k_cuda,
    quantized_matmul_q5k_metal,
    GgmlDType::Q5K
);
quantized_matmul!(
    quantized_matmul_q6k_bis,
    quantized_matmul_q6k_cpu,
    quantized_matmul_q6k_cuda,
    quantized_matmul_q6k_metal,
    GgmlDType::Q6K
);
// Not implemented on metal
quantized_matmul!(
    quantized_matmul_q8k_bis,
    quantized_matmul_q8k_cpu,
    quantized_matmul_q8k_cuda,
    quantized_matmul_q8k_metal,
    GgmlDType::Q8K
);

#[test]
fn quantized_matmul_q2k() -> Result<()> {
    use k_quants::BlockQ2K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [0.887, 0.428, 0.219, 1.669]);

    ggml_matmul_error_test::<BlockQ2K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q3k() -> Result<()> {
    use k_quants::BlockQ3K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q3K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.001, 1.425, -0.311, 1.493]);

    ggml_matmul_error_test::<BlockQ3K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q4k() -> Result<()> {
    use k_quants::BlockQ4K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q4K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.094, 1.442, -0.196, 1.587]);

    ggml_matmul_error_test::<BlockQ4K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q5k() -> Result<()> {
    use k_quants::BlockQ5K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q5K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.161, 1.498, -0.175, 1.739]);

    //Expected: 0.000740408897
    ggml_matmul_error_test::<BlockQ5K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q6k() -> Result<()> {
    use k_quants::BlockQ6K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q6K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.293, 1.497, -0.159, 1.737]);

    ggml_matmul_error_test::<BlockQ6K>()?;
    Ok(())
}

#[test]
fn quantized_matmul_q8k() -> Result<()> {
    use k_quants::BlockQ8K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q8K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.241, 1.52, -0.2, 1.7]);

    ggml_matmul_error_test::<BlockQ8K>()?;
    Ok(())
}

fn from_data_dequant_matches_canonical_when_caller_passes_cow_owned(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let n = 1024usize;
    let src_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let src = Tensor::from_vec(src_data, (n,), &cpu)?;
    let qt_canonical = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let canonical_dequant = qt_canonical.dequantize(&cpu)?.to_vec1::<f32>()?;

    let bytes_owned: Vec<u8> = qt_canonical.data()?.to_vec();
    let storage = quantized::QStorage::from_data(Cow::Owned(bytes_owned), device, GgmlDType::Q4_0)?;
    let qt_via_from_data = quantized::QTensor::new(storage, (n,))?;
    let observed_dequant = qt_via_from_data.dequantize(device)?.to_vec1::<f32>()?;

    let max_diff = canonical_dequant
        .iter()
        .zip(observed_dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "QStorage::from_data dequant mismatch on {device:?} (max |Δ| = {max_diff})"
    );
    Ok(())
}

test_device!(
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned,
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned_cpu,
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned_cuda,
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned_metal
);

// ---------------------------------------------------------------------------
// Exact weight-only quantized matmul (`QTensor::forward_exact`) and `QTensor::dequantize_to`.
// ---------------------------------------------------------------------------

/// Small deterministic generator (splitmix64) so the W8A16 tests do not depend on `rand`.
struct W8Rng(u64);

impl W8Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Raw GGML bytes of a random `(n, k)` weight: random quants with small positive f16 scales
/// (plus random sub-block scales for the k-quants), or N(0, 0.05) values for the float types.
fn w8_weight_bytes(dtype: GgmlDType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = W8Rng(seed);
    let elems = n * k;
    let f16_scale = |rng: &mut W8Rng| half::f16::from_f32(2e-4 + 1.5e-3 * rng.uniform()).to_bits();
    match dtype {
        GgmlDType::F32 => (0..elems)
            .flat_map(|_| (0.05 * rng.normal()).to_le_bytes())
            .collect(),
        GgmlDType::F16 => (0..elems)
            .flat_map(|_| half::f16::from_f32(0.05 * rng.normal()).to_le_bytes())
            .collect(),
        GgmlDType::BF16 => (0..elems)
            .flat_map(|_| half::bf16::from_f32(0.05 * rng.normal()).to_le_bytes())
            .collect(),
        _ => {
            let bs = dtype.block_size();
            let ts = dtype.type_size();
            let mut bytes: Vec<u8> = (0..elems / bs * ts).map(|_| rng.next_u64() as u8).collect();
            // Byte offsets of the f16 scale fields inside a block.
            let scales: &[usize] = match dtype {
                GgmlDType::Q4_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0 => &[0],
                GgmlDType::Q4_1 | GgmlDType::Q5_1 => &[0, 2],
                GgmlDType::Q2K => &[80, 82],
                GgmlDType::Q3K => &[108],
                GgmlDType::Q4K | GgmlDType::Q5K => &[0, 2],
                GgmlDType::Q6K => &[208],
                d => panic!("w8_weight_bytes: unsupported {d:?}"),
            };
            for blk in bytes.chunks_exact_mut(ts) {
                for &o in scales {
                    blk[o..o + 2].copy_from_slice(&f16_scale(&mut rng).to_le_bytes());
                }
            }
            bytes
        }
    }
}

fn w8_qtensor(
    bytes: &[u8],
    dtype: GgmlDType,
    n: usize,
    k: usize,
    dev: &Device,
) -> Result<quantized::QTensor> {
    let storage = quantized::QStorage::from_data(Cow::Borrowed(bytes), dev, dtype)?;
    quantized::QTensor::new(storage, (n, k))
}

/// Activations with Qwen3-like "massive" channels: N(0, 1) plus a handful of channels around
/// +-(100..2000), the case where Q8_1 activation quantization loses accuracy.
fn w8_input(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = W8Rng(seed);
    let massive = [k / 7, k / 3 + 1, k / 2 + 5, k - 3];
    (0..rows * k)
        .map(|i| {
            let c = i % k;
            let v = rng.normal();
            if massive.contains(&c) {
                v.signum() * (100.0 + 1900.0 * rng.uniform())
            } else {
                v
            }
        })
        .collect()
}

/// `x` (rows × k, already rounded to the input dtype) times the dequantized weight `w` (n × k)
/// in f64, for the output columns `cols`; returns rows × cols.len() values.
fn w8_reference(x: &[f32], w: &[f32], rows: usize, k: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols.len());
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for &j in cols {
            let wr = &w[j * k..(j + 1) * k];
            out.push(xr.iter().zip(wr).map(|(&a, &b)| a as f64 * b as f64).sum());
        }
    }
    out
}

/// Spacing of `dtype` values around `v` (the larger neighbour gap).
fn w8_ulp(dtype: DType, v: f64) -> f64 {
    let v = v as f32;
    match dtype {
        DType::BF16 => {
            let b = half::bf16::from_f32(v.abs());
            (half::bf16::from_bits(b.to_bits() + 1).to_f32() - b.to_f32()) as f64
        }
        DType::F16 => {
            let b = half::f16::from_f32(v.abs());
            (half::f16::from_bits(b.to_bits() + 1).to_f32() - b.to_f32()) as f64
        }
        _ => (f32::from_bits(v.abs().to_bits() + 1) - v.abs()) as f64,
    }
}

/// Error statistics of `y` (rows × n, any float dtype) against the f64 reference on `cols`.
struct W8Err {
    /// max |y - ref| / max |ref|
    rel: f64,
    /// max |y - ref| / (0.5 ulp_dtype(ref) + 1e-5 max |ref|): <= 1 means "output rounding only".
    rounding: f64,
}

fn w8_err(
    y: &Tensor,
    reference: &[f64],
    rows: usize,
    cols: &[usize],
    dtype: DType,
) -> Result<W8Err> {
    let y = y
        .to_dtype(DType::F32)?
        .reshape((rows, ()))?
        .to_vec2::<f32>()?;
    let max_ref = reference
        .iter()
        .fold(0f64, |m, v| m.max(v.abs()))
        .max(1e-30);
    let (mut max_abs, mut rounding) = (0f64, 0f64);
    for r in 0..rows {
        for (ci, &j) in cols.iter().enumerate() {
            let rf = reference[r * cols.len() + ci];
            let d = (y[r][j] as f64 - rf).abs();
            max_abs = max_abs.max(d);
            rounding = rounding.max(d / (0.5 * w8_ulp(dtype, rf) + 1e-5 * max_ref));
        }
    }
    Ok(W8Err {
        rel: max_abs / max_ref,
        rounding,
    })
}

fn w8_round_to(x: &[f32], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::BF16 => x
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        DType::F16 => x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect(),
        _ => x.to_vec(),
    }
}

fn w8_bits(t: &Tensor) -> Result<Vec<u32>> {
    let t = t.flatten_all()?;
    Ok(match t.dtype() {
        DType::F32 => t.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect(),
        DType::F16 => t
            .to_vec1::<half::f16>()?
            .iter()
            .map(|v| v.to_bits() as u32)
            .collect(),
        DType::BF16 => t
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|v| v.to_bits() as u32)
            .collect(),
        d => bail!("w8_bits: unexpected {d:?}"),
    })
}

/// Columns used for the f64 reference: all of them up to `max`, otherwise an even spread that
/// includes the first and last column.
fn w8_cols(n: usize, max: usize) -> Vec<usize> {
    if n <= max {
        (0..n).collect()
    } else {
        (0..max).map(|i| i * (n - 1) / (max - 1)).collect()
    }
}

/// Runs `forward_exact` for one weight on `dev` over `ms` row counts and the three input dtypes,
/// checks the accuracy contract and prints one table row per case.
#[allow(clippy::too_many_arguments)]
fn w8_check_forward_exact(
    dev: &Device,
    label: &str,
    wtype: GgmlDType,
    n: usize,
    k: usize,
    ms: &[usize],
    cols_small: usize,
    cols_large: usize,
    compare_qmatmul: bool,
) -> Result<()> {
    let bytes = w8_weight_bytes(wtype, n, k, 0x5EED ^ (n * 31 + k) as u64);
    let q_cpu = w8_qtensor(&bytes, wtype, n, k, &Device::Cpu)?;
    let w_ref = q_cpu
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let q_dev = std::sync::Arc::new(w8_qtensor(&bytes, wtype, n, k, dev)?);
    let exact_mv = dev.is_cuda()
        && matches!(
            wtype,
            GgmlDType::Q8_0 | GgmlDType::Q4_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
        );
    for &m in ms {
        let cols = w8_cols(n, if m <= 8 { cols_small } else { cols_large });
        let x32 = w8_input(m, k, 0xACE ^ (m * 7919 + k) as u64);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let xr = w8_round_to(&x32, dtype);
            let reference = w8_reference(&xr, &w_ref, m, k, &cols);
            let x = Tensor::from_slice(&x32, (m, k), dev)?.to_dtype(dtype)?;
            let y = q_dev.forward_exact(&x)?;
            assert_eq!(y.dims(), [m, n]);
            assert_eq!(y.dtype(), dtype);
            let e = w8_err(&y, &reference, m, &cols, dtype)?;
            let path = if m <= 8 {
                if exact_mv {
                    "mv"
                } else {
                    "f32-dense"
                }
            } else {
                "dequant+gemm"
            };
            let mut line = format!(
                "[W8A16-ACC] {label} {wtype:?} {n}x{k} m={m} in={dtype:?} path={path} max_rel={:.2e} rounding={:.2}",
                e.rel, e.rounding
            );
            if m <= 8 || dtype == DType::F32 {
                // f32 dequantized weights, f32 accumulation: only the output rounding differs.
                if dtype == DType::F32 {
                    assert!(e.rel <= 1e-5, "{line}");
                } else {
                    assert!(e.rounding <= 1.0, "{line}");
                }
            } else {
                // Prefill in bf16 / f16: the weight is rounded to the activation dtype, then the
                // same GEMM as a dense layer. Must equal `x.matmul(dequantize_to(dtype).t())`.
                let tol = if dtype == DType::BF16 { 3e-2 } else { 5e-3 };
                assert!(e.rel <= tol, "{line}");
                if !dev.is_cuda() {
                    // CPU: bf16 prefill runs in f32 (no bf16 matmul on the CPU backend).
                    println!("{line}");
                    continue;
                }
                let dense = q_dev.dequantize_to(dtype, dev)?;
                let y_dense = x.matmul(&dense.t()?)?;
                // Against weights rounded to the activation dtype the GEMM must only add the
                // output rounding (f32 accumulation), for both paths.
                let w_rounded = w8_round_to(&w_ref, dtype);
                let ref_rounded = w8_reference(&xr, &w_rounded, m, k, &cols);
                let er = w8_err(&y, &ref_rounded, m, &cols, dtype)?;
                let erd = w8_err(&y_dense, &ref_rounded, m, &cols, dtype)?;
                line.push_str(&format!(
                    " rounding_vs_rounded_w={:.2} (dense {:.2})",
                    er.rounding, erd.rounding
                ));
                // Never worse than the dense layer, and a single rounding when tiled.
                assert!(er.rounding <= erd.rounding.max(1.0) + 0.05, "{line}");
                let (a, b) = (w8_bits(&y)?, w8_bits(&y_dense)?);
                let same = a.iter().zip(&b).filter(|(a, b)| a == b).count();
                let ed = w8_err(&y_dense, &reference, m, &cols, dtype)?;
                line.push_str(&format!(
                    " dense_bf16_path_max_rel={:.2e} bit_identical={}/{}",
                    ed.rel,
                    same,
                    a.len()
                ));
                let single_tile = n * k * dtype.size_in_bytes() <= 32 << 20;
                let matmul_mode = std::env::var("CANDLE_W8A16_PREFILL_GEMM")
                    .is_ok_and(|v| v.eq_ignore_ascii_case("matmul"));
                if single_tile && matmul_mode {
                    // Opt-in mode: the exact `Tensor::matmul` cuBLAS call.
                    assert_eq!(same, a.len(), "{line}");
                } else {
                    // Default: f32 accumulation rounded once, never worse than the dense layer.
                    assert!(er.rounding <= 1.05, "{line}");
                    assert!(e.rel <= ed.rel * 1.05 + 1e-6, "{line}");
                }
            }
            if compare_qmatmul {
                let mm = quantized::QMatMul::from_arc(q_dev.clone())?;
                if let Ok(yq) = mm.forward(&x) {
                    let eq = w8_err(&yq, &reference, m, &cols, dtype)?;
                    line.push_str(&format!(" | QMatMul::forward max_rel={:.2e}", eq.rel));
                }
            }
            println!("{line}");
        }
    }
    Ok(())
}

/// Qwen3-0.6B projection shapes (n = out, k = in).
#[cfg(feature = "cuda")]
const W8_QWEN3_SHAPES: [(&str, usize, usize); 5] = [
    ("q_proj", 2048, 1024),
    ("k/v_proj", 1024, 1024),
    ("o_proj", 1024, 2048),
    ("gate/up", 3072, 1024),
    ("down", 1024, 3072),
];
#[cfg(feature = "cuda")]
const W8_MS: [usize; 7] = [1, 2, 4, 8, 9, 64, 256];

#[test]
fn w8a16_forward_exact_cpu() -> Result<()> {
    let dev = &Device::Cpu;
    for wtype in [
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::F16,
    ] {
        w8_check_forward_exact(dev, "cpu", wtype, 96, 512, &[1, 3, 9], 96, 96, false)?;
    }
    Ok(())
}

/// Backend-generic contract of `forward_exact` and `dequantize_to`, generated for every compiled
/// backend by `test_device!` (CPU, CUDA, Metal, ROCm, Vulkan): weight types with and without a
/// CUDA mat-vec kernel, decode (m <= 8) and prefill row counts, F32 / BF16 / F16 inputs.
fn w8a16_generic(dev: &Device) -> Result<()> {
    let (n, k) = (192, 1024);
    let label = format!("{:?}", dev.location());
    for wtype in [
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::F16,
        GgmlDType::BF16,
    ] {
        w8_check_forward_exact(dev, &label, wtype, n, k, &[1, 3, 8, 9, 40], 1 << 20, 64, false)?;
        // `dequantize_to` on the device against the CPU dequantization: bit exact on CPU / Metal
        // and for the direct CUDA kernels, otherwise within one ulp of the target dtype (or of
        // f32 at the weight scale, for kernels that contract the k-quant `d * q - m` to an FMA).
        let bytes = w8_weight_bytes(wtype, n, k, 21);
        let want_f32 = w8_qtensor(&bytes, wtype, n, k, &Device::Cpu)?.dequantize(&Device::Cpu)?;
        let scale = want_f32.abs()?.max_all()?.to_scalar::<f32>()? as f64;
        let q = w8_qtensor(&bytes, wtype, n, k, dev)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let want = want_f32.to_dtype(dtype)?;
            let got = q.dequantize_to(dtype, dev)?;
            assert_eq!(got.dtype(), dtype);
            assert_eq!(got.dims(), [n, k]);
            assert!(got.device().same_device(dev));
            let a = w8_bits(&got.to_device(&Device::Cpu)?)?;
            let b = w8_bits(&want)?;
            let af = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?;
            let bf = want.to_dtype(DType::F32)?.flatten_all()?;
            let (af, bf) = (af.to_vec1::<f32>()?, bf.to_vec1::<f32>()?);
            let mut differ = 0;
            for i in 0..a.len() {
                if a[i] != b[i] {
                    differ += 1;
                    let d = (af[i] as f64 - bf[i] as f64).abs();
                    let tol = w8_ulp(dtype, bf[i] as f64).max(f32::EPSILON as f64 * scale);
                    assert!(
                        d <= tol,
                        "{label} {wtype:?} -> {dtype:?}: element {i} {} vs {}",
                        af[i],
                        bf[i]
                    );
                }
            }
            println!(
                "[W8A16-DQ] {label} {wtype:?} -> {dtype:?}: {differ}/{} values differ from the CPU",
                a.len()
            );
        }
    }
    Ok(())
}

test_device!(
    w8a16_generic,
    w8a16_generic_cpu,
    w8a16_generic_cuda,
    w8a16_generic_metal
);

#[test]
fn w8a16_dequantize_to_cpu() -> Result<()> {
    let (n, k) = (64, 512);
    for wtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::BF16] {
        let bytes = w8_weight_bytes(wtype, n, k, 7);
        let q = w8_qtensor(&bytes, wtype, n, k, &Device::Cpu)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let a = q.dequantize_to(dtype, &Device::Cpu)?;
            let b = q.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
            assert_eq!(a.dtype(), dtype);
            assert_eq!(w8_bits(&a)?, w8_bits(&b)?, "{wtype:?} -> {dtype:?}");
            let rows = q.dequantize_rows_to(5, 17, dtype, &Device::Cpu)?;
            assert_eq!(w8_bits(&rows)?, w8_bits(&b.narrow(0, 5, 17)?)?);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn w8a16_dequantize_to_cuda() -> Result<()> {
    let dev = &Device::new_cuda(0)?;
    let (n, k) = (160, 1024);
    for wtype in [
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        // No direct kernel: f32 kernel + device cast (not bit exact, see below).
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q3K,
    ] {
        let bytes = w8_weight_bytes(wtype, n, k, 11 + wtype.type_size() as u64);
        let q_cpu = w8_qtensor(&bytes, wtype, n, k, &Device::Cpu)?;
        let q_gpu = w8_qtensor(&bytes, wtype, n, k, dev)?;
        let direct = !matches!(wtype, GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q3K);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let want = q_cpu.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
            let got = q_gpu.dequantize_to(dtype, dev)?;
            assert_eq!(got.dtype(), dtype);
            assert!(got.device().is_cuda());
            let (a, b) = (w8_bits(&got)?, w8_bits(&want)?);
            let diff = a.iter().zip(&b).filter(|(a, b)| a != b).count();
            println!(
                "[W8A16-DQ] {wtype:?} -> {dtype:?}: {diff}/{} values differ from the CPU",
                a.len()
            );
            if direct {
                assert_eq!(diff, 0, "{wtype:?} -> {dtype:?} is not bit exact");
                // Row ranges (block aligned offsets) of the direct kernels.
                let rows = q_gpu.dequantize_rows_to(37, 101, dtype, dev)?;
                assert_eq!(w8_bits(&rows)?, w8_bits(&want.narrow(0, 37, 101)?)?);
            } else {
                let got = got.to_device(&Device::Cpu)?;
                let d = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(d <= 1e-3, "{wtype:?} -> {dtype:?}: max diff {d}");
            }
        }
        // `QTensor::dequantize` of the float GGUF types now stays on the device too.
        if matches!(wtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
            let a = q_gpu.dequantize(dev)?;
            let b = q_cpu.dequantize(&Device::Cpu)?;
            assert_eq!(w8_bits(&a)?, w8_bits(&b)?);
            let a = q_gpu.dequantize_f16(dev)?;
            assert_eq!(w8_bits(&a)?, w8_bits(&b.to_dtype(DType::F16)?)?);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn w8a16_forward_exact_qwen3_cuda() -> Result<()> {
    let dev = &Device::new_cuda(0)?;
    for (label, n, k) in W8_QWEN3_SHAPES {
        w8_check_forward_exact(dev, label, GgmlDType::Q8_0, n, k, &W8_MS, 1 << 20, 48, true)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn w8a16_forward_exact_lm_head_cuda() -> Result<()> {
    // A quarter of the 151936 x 1024 tied lm_head of Qwen3-0.6B (41 MB of Q8_0): the bf16 / f32
    // prefill weight no longer fits one 32 MiB dequantization tile.
    let dev = &Device::new_cuda(0)?;
    let n = 151936 / 4;
    w8_check_forward_exact(
        dev,
        "lm_head/4",
        GgmlDType::Q8_0,
        n,
        1024,
        &W8_MS,
        384,
        48,
        true,
    )
}

#[cfg(feature = "cuda")]
#[test]
fn w8a16_forward_exact_kquants_cuda() -> Result<()> {
    let dev = &Device::new_cuda(0)?;
    for wtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        for (n, k) in [(1024, 1024), (1024, 3072), (640, 2048)] {
            w8_check_forward_exact(
                dev,
                "kq",
                wtype,
                n,
                k,
                &[1, 3, 8, 9, 64],
                1 << 20,
                48,
                false,
            )?;
        }
    }
    // Weight types without a mat-vec kernel: f32 dequantization + f32 GEMM for <= 8 rows.
    for wtype in [
        GgmlDType::Q5_0,
        GgmlDType::Q3K,
        GgmlDType::F16,
        GgmlDType::BF16,
    ] {
        w8_check_forward_exact(
            dev,
            "other",
            wtype,
            256,
            1024,
            &[1, 8, 9],
            1 << 20,
            64,
            false,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn w8a16_forward_exact_layouts_cuda() -> Result<()> {
    let dev = &Device::new_cuda(0)?;
    let (n, k) = (384, 1024);
    let bytes = w8_weight_bytes(GgmlDType::Q8_0, n, k, 99);
    let q = std::sync::Arc::new(w8_qtensor(&bytes, GgmlDType::Q8_0, n, k, dev)?);
    let w = q.dequantize_to(DType::F32, dev)?;
    let x32 = Tensor::from_vec(w8_input(12, k, 5), (12, k), dev)?;
    let check = |x: &Tensor, what: &str| -> Result<()> {
        for dtype in [DType::F32, DType::BF16] {
            let x = x.to_dtype(dtype)?;
            let y = q.forward_exact(&x)?;
            let mut dims = x.dims().to_vec();
            *dims.last_mut().unwrap() = n;
            assert_eq!(y.dims(), dims.as_slice(), "{what}");
            let flat = x.to_dtype(DType::F32)?.reshape(((), k))?;
            let want = flat.matmul(&w.t()?)?;
            let got = y.to_dtype(DType::F32)?.reshape(((), n))?;
            let d = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let s = want.abs()?.max_all()?.to_scalar::<f32>()?;
            let rows = flat.dim(0)?;
            let tol = if dtype == DType::F32 {
                1e-5
            } else if rows <= 8 {
                4e-3
            } else {
                3e-2
            };
            assert!(d / s <= tol, "{what} {dtype:?}: rel {}", d / s);
        }
        Ok(())
    };
    check(&x32.get(0)?, "rank-1")?;
    check(&x32.narrow(0, 0, 6)?.reshape((2, 3, k))?, "rank-3, 6 rows")?;
    check(&x32.reshape((3, 4, k))?, "rank-3, 12 rows")?;
    check(&x32.reshape((2, 2, 3, k))?, "rank-4")?;
    // Misaligned start (one element in): the mat-vec input is re-packed.
    let flat = x32.flatten_all()?;
    check(&flat.narrow(0, 1, 2 * k)?.reshape((2, k))?, "misaligned")?;
    // Non-contiguous: every other row.
    check(
        &x32.reshape((6, 2, k))?.narrow(1, 0, 1)?.squeeze(1)?,
        "strided rows",
    )?;
    // Transposed storage.
    check(&x32.narrow(0, 0, 4)?.t()?.contiguous()?.t()?, "transposed")?;
    // Zero rows.
    let y = q.forward_exact(&Tensor::zeros((0, k), DType::BF16, dev)?)?;
    assert_eq!(y.dims(), [0, n]);
    // Wrong inner dimension.
    assert!(q
        .forward_exact(&Tensor::zeros((2, k + 32), DType::F32, dev)?)
        .is_err());

    // QMatMul::forward_exact on the three variants.
    let mm = quantized::QMatMul::from_arc(q.clone())?;
    let x = x32.narrow(0, 0, 2)?;
    let a = mm.forward_exact(&x)?;
    let b = q.forward_exact(&x)?;
    assert_eq!(w8_bits(&a)?, w8_bits(&b)?);
    let dense = quantized::QMatMul::Tensor(w.clone());
    let c = dense.forward_exact(&x.to_dtype(DType::BF16)?)?;
    assert_eq!(c.dtype(), DType::BF16);
    Ok(())
}

/// Average GPU time of `f` in microseconds. The GPU is kept busy with large f32 matmuls while
/// the timed calls are queued, so host-side overhead (this is a debug build) is excluded and the
/// result is the back-to-back device time per call (kernels + launch gaps).
#[cfg(feature = "cuda")]
fn w8_gpu_time_us(
    dev: &Device,
    iters: usize,
    mut f: impl FnMut() -> Result<Tensor>,
) -> Result<(f64, bool)> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUevent_flags;
    let cuda = dev.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    for _ in 0..3 {
        f()?;
    }
    dev.synchronize()?;
    let filler = Tensor::ones((2048, 2048), DType::F32, dev)?;
    let t_fill = std::time::Instant::now();
    let f0 = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    for _ in 0..10 {
        let _ = filler.matmul(&filler)?;
    }
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    let host = t0.elapsed().as_secs_f64() + (t0 - t_fill).as_secs_f64();
    let end = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(candle_core::Error::wrap)?;
    let ms = start.elapsed_ms(&end).map_err(candle_core::Error::wrap)?;
    let fill_ms = f0.elapsed_ms(&start).map_err(candle_core::Error::wrap)?;
    // Valid when the host finished queueing before the GPU drained the filler.
    let valid = host * 1e3 < fill_ms as f64;
    Ok((ms as f64 * 1e3 / iters as f64, valid))
}

/// Decode / prefill timing of `forward_exact` against `QMatMul::forward` (MMVQ / MMQ with Q8_1
/// activations) and a resident dense BF16 weight (today's dequantize-to-BF16 path).
/// Run with `cargo test --features cuda --test quantized_tests w8a16_bench -- --ignored --nocapture`
/// (`CANDLE_W8A16_MV=<warps>x<rows>` overrides the mat-vec launch shape).
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn w8a16_bench_cuda() -> Result<()> {
    let dev = &Device::new_cuda(0)?;
    let shapes: Vec<(&str, usize, usize)> = std::env::var("W8A16_BENCH_SHAPES")
        .ok()
        .map(|s| {
            W8_QWEN3_SHAPES
                .iter()
                .copied()
                .chain([("lm_head", 151936, 1024)])
                .filter(|(l, _, _)| s.split(',').any(|x| x == *l))
                .collect()
        })
        .unwrap_or_else(|| {
            W8_QWEN3_SHAPES
                .iter()
                .copied()
                .chain([("lm_head", 151936, 1024)])
                .collect()
        });
    let ms: Vec<usize> = std::env::var("W8A16_BENCH_M")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![1]);
    // Back-to-back cost of a trivial kernel: the per-call floor included in every number below.
    let tiny = Tensor::zeros(8, DType::F32, dev)?;
    let (floor, _) = w8_gpu_time_us(dev, 100, || tiny.affine(1.0, 0.0))?;
    println!("[W8A16-BENCH] floor (one trivial kernel per call) {floor:.1} us");
    for (label, n, k) in shapes {
        let bytes = w8_weight_bytes(GgmlDType::Q8_0, n, k, 3);
        let q = std::sync::Arc::new(w8_qtensor(&bytes, GgmlDType::Q8_0, n, k, dev)?);
        drop(bytes);
        let w_bf16 = q.dequantize_to(DType::BF16, dev)?;
        let mm = quantized::QMatMul::from_arc(q.clone())?;
        for &m in &ms {
            let iters = if n > 100_000 { 30 } else { 100 };
            let x = Tensor::from_vec(w8_input(m, k, 1), (m, k), dev)?;
            let xb = x.to_dtype(DType::BF16)?;
            let wt = w_bf16.t()?;
            let (exact_bf16, v1) = w8_gpu_time_us(dev, iters, || q.forward_exact(&xb))?;
            let (exact_f32, v2) = w8_gpu_time_us(dev, iters, || q.forward_exact(&x))?;
            let (mmvq, v3) = w8_gpu_time_us(dev, iters, || mm.forward(&xb))?;
            let (dense, v4) = w8_gpu_time_us(dev, iters, || xb.matmul(&wt))?;
            let valid = v1 && v2 && v3 && v4;
            println!(
                "[W8A16-BENCH] {label} {n}x{k} m={m}: forward_exact bf16 {exact_bf16:.1} us | forward_exact f32 {exact_f32:.1} us | QMatMul::forward (q8_1) {mmvq:.1} us | dense bf16 matmul {dense:.1} us{}",
                if valid { "" } else { " (host-bound, unreliable)" }
            );
        }
    }
    Ok(())
}
