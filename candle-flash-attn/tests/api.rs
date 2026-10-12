//! The crates.io `candle-flash-attn` API on the CPU against the unfused reference.

use candle::{DType, Device, Tensor};
use candle_flash_attn::*;

fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn dense_api() -> candle::Result<()> {
    let dev = Device::Cpu;
    let q = Tensor::randn(0f32, 1., (2, 7, 4, 32), &dev)?;
    let k = Tensor::randn(0f32, 1., (2, 9, 2, 32), &dev)?;
    let v = Tensor::randn(0f32, 1., (2, 9, 2, 32), &dev)?;
    let slopes = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &dev)?;
    let r = |cfg: FlashAttnConfig| flash_attn_unfused(&q, &k, &v, &cfg);
    assert!(
        max_diff(
            &flash_attn(&q, &k, &v, 0.3, true)?,
            &r(FlashAttnConfig::new(0.3).causal())?
        ) < 1e-5
    );
    assert!(
        max_diff(
            &flash_attn_windowed(&q, &k, &v, 0.3, Some(2), Some(0))?,
            &r(FlashAttnConfig::new(0.3).window(Some(2), Some(0)))?
        ) < 1e-5
    );
    assert!(
        max_diff(
            &flash_attn_alibi(&q, &k, &v, &slopes, 0.3, true)?,
            &r(FlashAttnConfig::new(0.3).causal().alibi(&slopes))?
        ) < 1e-5
    );
    assert!(
        max_diff(
            &flash_attn_alibi_windowed_softcap(
                &q,
                &k,
                &v,
                Some(&slopes),
                0.3,
                Some(3),
                Some(1),
                5.0
            )?,
            &r(FlashAttnConfig::new(0.3)
                .window(Some(3), Some(1))
                .alibi(&slopes)
                .softcap(5.0))?
        ) < 1e-5
    );
    // the op itself, as with the crates.io crate
    let op = FlashAttn {
        softmax_scale: 0.3,
        alibi_slopes: None,
        window_size_left: None,
        window_size_right: Some(0),
        softcap: None,
    };
    let o = q.apply_op3(&k, &v, op)?;
    assert!(max_diff(&o, &r(FlashAttnConfig::new(0.3).causal())?) < 1e-5);
    // bf16 in, bf16 out
    let o = flash_attn(
        &q.to_dtype(DType::BF16)?,
        &k.to_dtype(DType::BF16)?,
        &v.to_dtype(DType::BF16)?,
        0.3,
        true,
    )?;
    assert_eq!(o.dtype(), DType::BF16);
    Ok(())
}

#[test]
fn varlen_and_paged_api() -> candle::Result<()> {
    let dev = Device::Cpu;
    let (hq, hkv, d, page) = (2, 1, 16, 4);
    let q = Tensor::randn(0f32, 1., (9, hq, d), &dev)?;
    let cu_q = Tensor::new(&[0u32, 4, 9], &dev)?;
    let cu_k = Tensor::new(&[0u32, 6, 13], &dev)?;
    // paged cache: 5 pages of 4 tokens; sequence 0 uses pages [3, 0], sequence 1 [1, 4]
    let k_pages = Tensor::randn(0f32, 1., (5, page, hkv, d), &dev)?;
    let v_pages = Tensor::randn(0f32, 1., (5, page, hkv, d), &dev)?;
    let table = Tensor::new(&[[3u32, 0], [1, 4]], &dev)?;
    let paged = flash_attn_varlen_paged_windowed(
        &q,
        &k_pages,
        &v_pages,
        &cu_q,
        &cu_k,
        &table,
        None,
        5,
        7,
        0.25,
        None,
        Some(0),
        page,
        None,
    )?;
    // the same sequences, gathered by hand
    let seq = |pages: &[u32], len: usize| -> candle::Result<(Tensor, Tensor)> {
        let idx = Tensor::new(pages, &dev)?;
        let g = |x: &Tensor| {
            x.index_select(&idx, 0)?
                .reshape((pages.len() * page, hkv, d))?
                .narrow(0, 0, len)
        };
        Ok((g(&k_pages)?, g(&v_pages)?))
    };
    let (k0, v0) = seq(&[3, 0], 6)?;
    let (k1, v1) = seq(&[1, 4], 7)?;
    let k = Tensor::cat(&[&k0, &k1], 0)?;
    let v = Tensor::cat(&[&v0, &v1], 0)?;
    let dense = flash_attn_varlen(&q, &k, &v, &cu_q, &cu_k, 5, 7, 0.25, true)?;
    assert!(max_diff(&paged, &dense) < 1e-6);
    let first = flash_attn(
        &q.narrow(0, 0, 4)?.unsqueeze(0)?,
        &k0.unsqueeze(0)?,
        &v0.unsqueeze(0)?,
        0.25,
        true,
    )?;
    assert!(max_diff(&dense.narrow(0, 0, 4)?, &first.squeeze(0)?) < 1e-5);
    Ok(())
}
