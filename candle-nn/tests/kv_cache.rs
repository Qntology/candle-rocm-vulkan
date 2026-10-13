#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{Device, Result, Tensor};

#[test]
fn kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);
    for _ in [0, 1] {
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3.]);
        let t = Tensor::new(&[4f32], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0f32, 5., 6., 7.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 0., 5., 6., 7.]);
        assert_eq!(cache.current_seq_len(), 8);
        cache.reset();
    }
    Ok(())
}

/// Growth past the capacity (geometric, one copy) keeps the contents, zeroes the spare rows and
/// works along an inner dim.
#[test]
fn kv_cache_growth() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(1, 4);
    let mut want: Vec<Tensor> = vec![];
    let mut caps = vec![];
    for step in 0..12u32 {
        let len = 1 + (step as usize % 5);
        let vals: Vec<u32> = (0..(2 * len * 3) as u32).map(|i| i + 1 + 100 * step).collect();
        let t = Tensor::from_vec(vals, (2, len, 3), &Device::Cpu)?;
        cache.append(&t)?;
        want.push(t);
        caps.push(cache.max_seq_len());
        let got = cache.current_data()?.unwrap();
        assert_eq!(
            got.to_vec3::<u32>()?,
            Tensor::cat(&want, 1)?.to_vec3::<u32>()?,
            "step {step}"
        );
        let all = cache.all_data().as_ref().unwrap();
        assert_eq!(all.dim(1)?, cache.max_seq_len());
        let spare = cache.max_seq_len() - cache.current_seq_len();
        if spare > 0 {
            let z = all.narrow(1, cache.current_seq_len(), spare)?;
            assert_eq!(z.sum_all()?.to_scalar::<u32>()?, 0);
        }
    }
    // 4, then +4 (grow_by), then +half the capacity
    assert_eq!(caps.first(), Some(&4));
    assert!(caps.windows(2).all(|w| w[1] == w[0] || w[1] >= w[0] + 4));
    assert!(*caps.last().unwrap() < 2 * cache.current_seq_len() + 4);
    Ok(())
}

#[test]
fn rotating_kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingCache::new(0, 6);
    for _ in [0, 1] {
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        assert_eq!(cache.positions(1), &[0]);
        assert_eq!(cache.positions(2), &[0, 1]);
        let t = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3.]);
        assert_eq!(cache.positions(0), &[0, 1, 2]);
        assert_eq!(cache.positions(1), &[0, 1, 2, 3]);
        assert_eq!(cache.positions(2), &[0, 1, 2, 3, 4]);
        assert_eq!(cache.positions(3), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(cache.positions(4), &[6, 1, 2, 3, 4, 5]);
        let t = Tensor::new(&[4.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0., 5., 6., 7.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 3., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 8);
        assert_eq!(cache.offset(), 2);

        let t = Tensor::new(&[8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 9);
        assert_eq!(cache.offset(), 3);

        let t = Tensor::new(&[9., 10., 11.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 12);
        assert_eq!(cache.offset(), 0);

        let t = Tensor::new(&[12.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [12., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 13);
        assert_eq!(cache.offset(), 1);

        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        );
        assert_eq!(cache.positions(0), &[12, 7, 8, 9, 10, 11]);
        assert_eq!(cache.positions(2), &[12, 13, 14, 9, 10, 11]);
        assert_eq!(cache.positions(3), &[12, 13, 14, 15, 10, 11]);
        assert_eq!(cache.positions(8), &[13, 14, 15, 16, 17, 18, 19, 20]);
        let t = Tensor::new(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [0., 1., 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 22);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.positions(0), &[16, 17, 18, 19, 20, 21]);
        assert_eq!(cache.positions(1), &[22, 17, 18, 19, 20, 21]);

        let mask = cache.attn_mask(1, &Device::Cpu)?;
        assert!(mask.is_none());
        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let t = Tensor::new(&[42.], &Device::Cpu)?;

        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [42., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 23);
        assert_eq!(cache.offset(), 1);

        cache.reset();
    }
    Ok(())
}
