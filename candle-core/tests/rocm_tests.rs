#![cfg(feature = "rocm")]
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};

fn rocm() -> Device {
    Device::new_rocm(0).expect("rocm device")
}

fn tol(dtype: DType) -> f64 {
    match dtype {
        DType::F64 => 1e-9,
        DType::F32 => 1e-4,
        DType::F16 => 2e-2,
        DType::BF16 => 6e-2,
        _ => 0.0,
    }
}

fn assert_close(a: &Tensor, b: &Tensor, what: &str) -> Result<()> {
    assert_eq!(a.dims(), b.dims(), "{what}: shape mismatch");
    assert_eq!(a.dtype(), b.dtype(), "{what}: dtype mismatch");
    let t = tol(a.dtype());
    let va = a.to_device(&Device::Cpu)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
    let vb = b.to_device(&Device::Cpu)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
        if (x.is_nan() && y.is_nan()) || x == y {
            continue;
        }
        let diff = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            diff <= t * scale,
            "{what}: mismatch at {i}: rocm={x} cpu={y} (dtype {:?})",
            a.dtype()
        );
    }
    Ok(())
}

fn randn(shape: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?.to_dtype(dtype)
}

fn rand_pos(shape: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::rand(0.1f32, 2f32, shape, &Device::Cpu)?.to_dtype(dtype)
}

fn ints(shape: &[usize], dtype: DType, modulo: u32) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let v: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 13) % modulo).collect();
    Tensor::from_vec(v, shape, &Device::Cpu)?.to_dtype(dtype)
}

#[test]
fn transfer_roundtrip() -> Result<()> {
    let dev = rocm();
    for dtype in [
        DType::U8,
        DType::U32,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::F8E4M3,
    ] {
        let t = ints(&[3, 5, 7], DType::U32, 100)?.to_dtype(dtype)?;
        let g = t.to_device(&dev)?;
        assert!(g.device().is_rocm());
        let back = g.to_device(&Device::Cpu)?;
        assert_eq!(
            t.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?,
            back.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?,
            "{dtype:?}"
        );
    }
    let z = Tensor::zeros((4, 3), DType::F32, &dev)?;
    assert_eq!(z.sum_all()?.to_scalar::<f32>()?, 0.0);
    let o = Tensor::ones((4, 3), DType::BF16, &dev)?;
    assert_eq!(o.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?, 12.0);
    let f = Tensor::full(2.5f64, (2, 2), &dev)?;
    assert_eq!(f.to_vec2::<f64>()?, vec![vec![2.5, 2.5], vec![2.5, 2.5]]);
    Ok(())
}

#[test]
fn unary_ops() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let x = randn(&[4, 6, 5], dtype)?;
        let p = rand_pos(&[4, 6, 5], dtype)?;
        let xg = x.to_device(&dev)?;
        let pg = p.to_device(&dev)?;
        let xt = x.transpose(0, 2)?;
        let xgt = xg.transpose(0, 2)?;
        macro_rules! check {
            ($name:literal, $f:expr) => {{
                let f = $f;
                assert_close(&f(&xg)?, &f(&x)?, concat!($name, " contiguous"))?;
                assert_close(&f(&xgt)?, &f(&xt)?, concat!($name, " strided"))?;
            }};
        }
        check!("exp", |t: &Tensor| t.exp());
        check!("sin", |t: &Tensor| t.sin());
        check!("cos", |t: &Tensor| t.cos());
        check!("tanh", |t: &Tensor| t.tanh());
        check!("neg", |t: &Tensor| t.neg());
        check!("sqr", |t: &Tensor| t.sqr());
        check!("gelu", |t: &Tensor| t.gelu());
        check!("gelu_erf", |t: &Tensor| t.gelu_erf());
        check!("erf", |t: &Tensor| t.erf());
        check!("silu", |t: &Tensor| t.silu());
        check!("abs", |t: &Tensor| t.abs());
        check!("ceil", |t: &Tensor| t.ceil());
        check!("floor", |t: &Tensor| t.floor());
        check!("round", |t: &Tensor| t.round());
        check!("relu", |t: &Tensor| t.relu());
        check!("sign", |t: &Tensor| t.sign());
        check!("affine", |t: &Tensor| t.affine(1.5, -0.25));
        check!("elu", |t: &Tensor| t.elu(0.7));
        assert_close(&pg.log()?, &p.log()?, "log")?;
        assert_close(&pg.sqrt()?, &p.sqrt()?, "sqrt")?;
        assert_close(&pg.recip()?, &p.recip()?, "recip")?;
        assert_close(&pg.powf(1.7)?, &p.powf(1.7)?, "powf")?;
    }
    let i = ints(&[5, 4], DType::I64, 50)?;
    assert_close(&i.to_device(&dev)?.affine(2.0, 1.0)?, &i.affine(2.0, 1.0)?, "affine i64 (fallback)")?;
    Ok(())
}

#[test]
fn binary_and_cmp_ops() -> Result<()> {
    let dev = rocm();
    for dtype in [
        DType::F32,
        DType::F64,
        DType::F16,
        DType::BF16,
        DType::U8,
        DType::U32,
        DType::I64,
        DType::I32,
        DType::I16,
    ] {
        let (a, b) = if dtype.is_float() {
            (randn(&[3, 4, 5], dtype)?, rand_pos(&[3, 1, 5], dtype)?)
        } else {
            (ints(&[3, 4, 5], dtype, 10)?, ints(&[3, 1, 5], dtype, 5)?.affine(1.0, 1.0)?)
        };
        let ag = a.to_device(&dev)?;
        let bg = b.to_device(&dev)?;
        assert_close(&ag.broadcast_add(&bg)?, &a.broadcast_add(&b)?, "add")?;
        if dtype.is_float() {
            assert_close(&ag.broadcast_sub(&bg)?, &a.broadcast_sub(&b)?, "sub")?;
        }
        assert_close(&ag.broadcast_mul(&bg)?, &a.broadcast_mul(&b)?, "mul")?;
        assert_close(&ag.broadcast_div(&bg)?, &a.broadcast_div(&b)?, "div")?;
        assert_close(&ag.broadcast_maximum(&bg)?, &a.broadcast_maximum(&b)?, "maximum")?;
        assert_close(&ag.broadcast_minimum(&bg)?, &a.broadcast_minimum(&b)?, "minimum")?;
        assert_close(&ag.broadcast_lt(&bg)?, &a.broadcast_lt(&b)?, "lt")?;
        assert_close(&ag.broadcast_ge(&bg)?, &a.broadcast_ge(&b)?, "ge")?;
        assert_close(&ag.broadcast_eq(&bg)?, &a.broadcast_eq(&b)?, "eq")?;
        let at = a.transpose(1, 2)?;
        let agt = ag.transpose(1, 2)?;
        let bt = b.transpose(1, 2)?;
        let bgt = bg.transpose(1, 2)?;
        assert_close(&agt.broadcast_mul(&bgt)?, &at.broadcast_mul(&bt)?, "mul strided")?;
        assert_close(&(&agt + &agt)?, &(&at + &at)?, "add strided same")?;
    }
    Ok(())
}

#[test]
fn reductions() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16, DType::U32, DType::I64, DType::U8] {
        let x = if dtype.is_float() {
            randn(&[3, 37, 300], dtype)?
        } else {
            ints(&[3, 37, 300], dtype, 3)?
        };
        let xg = x.to_device(&dev)?;
        for d in 0..3 {
            assert_close(&xg.max_keepdim(d)?, &x.max_keepdim(d)?, "max")?;
            assert_close(&xg.min_keepdim(d)?, &x.min_keepdim(d)?, "min")?;
            assert_close(&xg.argmax_keepdim(d)?, &x.argmax_keepdim(d)?, "argmax")?;
            assert_close(&xg.argmin_keepdim(d)?, &x.argmin_keepdim(d)?, "argmin")?;
            if dtype != DType::U8 {
                let s1 = xg.sum_keepdim(d)?;
                let s2 = x.sum_keepdim(d)?;
                if dtype.is_float() {
                    let _ = s2;
                    let a = s1.to_device(&Device::Cpu)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
                    let b = x.to_dtype(DType::F64)?.sum_keepdim(d)?.flatten_all()?.to_vec1::<f64>()?;
                    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
                        let rel = (va - vb).abs() / vb.abs().max(1.0);
                        assert!(rel <= tol(dtype), "sum {dtype:?} dim {d} at {i}: rocm={va} exact={vb}");
                    }
                } else {
                    assert_close(&s1, &s2, "sum int")?;
                }
            }
        }
        let xt = x.transpose(0, 2)?;
        let xgt = xg.transpose(0, 2)?;
        assert_close(&xgt.max_keepdim(1)?, &xt.max_keepdim(1)?, "max strided")?;
        assert_close(&xgt.argmax_keepdim(2)?, &xt.argmax_keepdim(2)?, "argmax strided")?;
        assert_close(&xg.max_keepdim(2)?.max_keepdim(0)?, &x.max_keepdim(2)?.max_keepdim(0)?, "max chain")?;
    }
    let x = randn(&[2, 3], DType::F32)?;
    let xg = x.to_device(&dev)?;
    assert_close(&xg.sum_keepdim((0, 1))?, &x.sum_keepdim((0, 1))?, "sum all small")?;
    let big = randn(&[1, 151936], DType::F32)?;
    assert_close(&big.to_device(&dev)?.argmax_keepdim(1)?, &big.argmax_keepdim(1)?, "argmax vocab")?;
    Ok(())
}

#[test]
fn casts() -> Result<()> {
    let dev = rocm();
    let dtypes = [
        DType::U8,
        DType::U32,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::F64,
    ];
    let base = Tensor::new(&[[0.0f32, 1.5, -2.25, 300.7], [65535.0, 7.0, -0.5, 1e10]], &Device::Cpu)?;
    for &src in dtypes.iter() {
        let s = if src.is_float() { base.to_dtype(src)? } else { base.abs()?.to_dtype(DType::U8)?.to_dtype(src)? };
        let sg = s.to_device(&dev)?;
        for &dst in dtypes.iter() {
            let a = sg.to_dtype(dst)?;
            let b = s.to_dtype(dst)?;
            assert_close(&a, &b, &format!("cast {src:?}->{dst:?}"))?;
            let a = sg.t()?.to_dtype(dst)?;
            let b = s.t()?.to_dtype(dst)?;
            assert_close(&a, &b, &format!("cast strided {src:?}->{dst:?}"))?;
        }
    }
    let f8 = base.to_dtype(DType::F8E4M3)?;
    assert_close(&f8.to_device(&dev)?.to_dtype(DType::F32)?, &f8.to_dtype(DType::F32)?, "f8 fallback")?;
    Ok(())
}

#[test]
fn where_and_indexing() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::BF16, DType::F64, DType::U32, DType::U8] {
        let a = ints(&[4, 6], DType::U32, 9)?.to_dtype(dtype)?;
        let b = ints(&[1, 6], DType::U32, 5)?.to_dtype(dtype)?;
        let c = ints(&[4, 1], DType::U32, 2)?.to_dtype(DType::U8)?;
        let (ag, bg, cg) = (a.to_device(&dev)?, b.to_device(&dev)?, c.to_device(&dev)?);
        let r = c.broadcast_as((4, 6))?.where_cond(&a, &b.broadcast_as((4, 6))?)?;
        let rg = cg.broadcast_as((4, 6))?.where_cond(&ag, &bg.broadcast_as((4, 6))?)?;
        assert_close(&rg, &r, "where")?;

        for ids_dtype in [DType::U32, DType::I64, DType::U8] {
            let ids = Tensor::new(&[3u32, 0, 2, 2, 1], &Device::Cpu)?.to_dtype(ids_dtype)?;
            let idsg = ids.to_device(&dev)?;
            assert_close(&ag.index_select(&idsg, 0)?, &a.index_select(&ids, 0)?, "index_select 0")?;
            assert_close(&ag.index_select(&idsg, 1)?, &a.index_select(&ids, 1)?, "index_select 1")?;
            let at = a.t()?.contiguous()?;
            assert_close(&ag.t()?.index_select(&idsg, 1)?, &at.index_select(&ids, 1)?, "index_select strided")?;

            let gids = ints(&[4, 3], DType::U32, 6)?.to_dtype(ids_dtype)?;
            let gidsg = gids.to_device(&dev)?;
            assert_close(&ag.gather(&gidsg, 1)?, &a.gather(&gids, 1)?, "gather")?;

            if dtype.is_float() || dtype == DType::U32 {
                let src = ints(&[4, 3], DType::U32, 4)?.to_dtype(dtype)?;
                let srcg = src.to_device(&dev)?;
                let sids = Tensor::new(&[[0u32, 1, 2], [3, 4, 5], [5, 0, 1], [2, 2, 3]], &Device::Cpu)?
                    .to_dtype(ids_dtype)?;
                let sidsg = sids.to_device(&dev)?;
                assert_close(&ag.scatter(&sidsg, &srcg, 1)?, &a.scatter(&sids, &src, 1)?, "scatter")?;
                assert_close(&ag.scatter_add(&sidsg, &srcg, 1)?, &a.scatter_add(&sids, &src, 1)?, "scatter_add")?;
                let ia_ids = Tensor::new(&[1u32, 1, 3], &Device::Cpu)?.to_dtype(ids_dtype)?;
                let ia_idsg = ia_ids.to_device(&dev)?;
                let src2 = ints(&[3, 6], DType::U32, 3)?.to_dtype(dtype)?;
                let src2g = src2.to_device(&dev)?;
                assert_close(&ag.index_add(&ia_idsg, &src2g, 0)?, &a.index_add(&ia_ids, &src2, 0)?, "index_add")?;
            }
        }
    }
    let emb = randn(&[10, 4], DType::F32)?;
    let ids = Tensor::new(&[[1u32, 9], [0, 3]], &Device::Cpu)?;
    let r = candle_core::Tensor::embedding_like(&emb, &ids)?;
    let rg = candle_core::Tensor::embedding_like(&emb.to_device(&dev)?, &ids.to_device(&dev)?)?;
    assert_close(&rg, &r, "embedding")?;
    Ok(())
}

trait EmbeddingLike {
    fn embedding_like(w: &Tensor, ids: &Tensor) -> Result<Tensor>;
}

impl EmbeddingLike for Tensor {
    fn embedding_like(w: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let mut dims = ids.dims().to_vec();
        dims.push(w.dim(1)?);
        w.index_select(&ids.flatten_all()?, 0)?.reshape(dims)
    }
}

#[test]
fn copies_and_cat() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::F16, DType::I64, DType::U8, DType::F64] {
        let a = ints(&[3, 4, 5], DType::U32, 97)?.to_dtype(dtype)?;
        let b = ints(&[3, 2, 5], DType::U32, 31)?.to_dtype(dtype)?;
        let (ag, bg) = (a.to_device(&dev)?, b.to_device(&dev)?);
        assert_close(&Tensor::cat(&[&ag, &bg], 1)?, &Tensor::cat(&[&a, &b], 1)?, "cat 1")?;
        assert_close(&Tensor::cat(&[&ag, &ag], 2)?, &Tensor::cat(&[&a, &a], 2)?, "cat 2")?;
        assert_close(&Tensor::cat(&[&ag, &ag], 0)?, &Tensor::cat(&[&a, &a], 0)?, "cat 0")?;
        assert_close(&ag.transpose(0, 2)?.contiguous()?, &a.transpose(0, 2)?.contiguous()?, "contiguous")?;
        assert_close(&ag.narrow(1, 1, 2)?.contiguous()?, &a.narrow(1, 1, 2)?.contiguous()?, "narrow")?;
        assert_close(&ag.i((.., 2..4, 1))?, &a.i((.., 2..4, 1))?, "index op")?;
        let x = Tensor::stack(&[&ag, &ag], 0)?;
        let y = Tensor::stack(&[&a, &a], 0)?;
        assert_close(&x, &y, "stack")?;
        let s = ag.zeros_like()?;
        s.slice_set(&bg, 1, 1)?;
        let sc = a.zeros_like()?;
        sc.slice_set(&b, 1, 1)?;
        assert_close(&s, &sc, "slice_set")?;
        let big = ints(&[2, 3, 4, 5, 2, 3, 2, 2, 3, 2], DType::U32, 1000)?.to_dtype(dtype)?;
        let bigg = big.to_device(&dev)?;
        let perm = big.permute(vec![9usize, 0, 8, 1, 7, 2, 6, 3, 5, 4])?.contiguous()?;
        let permg = bigg.permute(vec![9usize, 0, 8, 1, 7, 2, 6, 3, 5, 4])?.contiguous()?;
        assert_close(&permg, &perm, "high rank permute")?;
    }
    Ok(())
}

fn mm_ref(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = a.dtype();
    if dt == DType::BF16 || dt == DType::F16 {
        a.to_dtype(DType::F32)?.matmul(&b.to_dtype(DType::F32)?)?.to_dtype(dt)
    } else {
        a.matmul(b)
    }
}

fn bmm_ref(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = a.dtype();
    if dt == DType::BF16 || dt == DType::F16 {
        a.to_dtype(DType::F32)?.broadcast_matmul(&b.to_dtype(DType::F32)?)?.to_dtype(dt)
    } else {
        a.broadcast_matmul(b)
    }
}

#[test]
fn matmuls() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let a = randn(&[2, 3, 7, 5], dtype)?;
        let b = randn(&[2, 3, 5, 4], dtype)?;
        let (ag, bg) = (a.to_device(&dev)?, b.to_device(&dev)?);
        assert_close(&ag.matmul(&bg)?, &mm_ref(&a, &b)?, "matmul batched")?;
        let bt = randn(&[2, 3, 4, 5], dtype)?;
        let btg = bt.to_device(&dev)?;
        assert_close(
            &ag.matmul(&btg.transpose(2, 3)?)?,
            &mm_ref(&a, &bt.transpose(2, 3)?)?,
            "matmul rhs transposed",
        )?;
        let at = randn(&[2, 3, 5, 7], dtype)?;
        let atg = at.to_device(&dev)?;
        assert_close(
            &atg.transpose(2, 3)?.matmul(&bg)?,
            &mm_ref(&at.transpose(2, 3)?, &b)?,
            "matmul lhs transposed",
        )?;
        let w = randn(&[9, 5], dtype)?;
        let wg = w.to_device(&dev)?;
        let x = randn(&[4, 6, 5], dtype)?;
        let xg = x.to_device(&dev)?;
        assert_close(
            &xg.broadcast_matmul(&wg.t()?)?,
            &bmm_ref(&x, &w.t()?)?,
            "linear broadcast",
        )?;
        let w3 = w.broadcast_left(4)?.t()?;
        let w3g = wg.broadcast_left(4)?.t()?;
        assert_close(&xg.matmul(&w3g)?, &mm_ref(&x, &w3)?, "qmatmul tensor style")?;
        let p = randn(&[3, 8, 6], dtype)?;
        let pg = p.to_device(&dev)?;
        let pt = pg.permute((1, 0, 2))?;
        let pc = p.permute((1, 0, 2))?;
        let v = randn(&[8, 6, 2], dtype)?;
        let vg = v.to_device(&dev)?;
        assert_close(&pt.matmul(&vg)?, &mm_ref(&pc, &v)?, "matmul permuted lhs")?;
        let v1 = randn(&[5], dtype)?;
        let m1 = randn(&[3, 5], dtype)?;
        assert_close(
            &m1.to_device(&dev)?.matmul(&v1.to_device(&dev)?.unsqueeze(1)?)?,
            &mm_ref(&m1, &v1.unsqueeze(1)?)?,
            "matvec",
        )?;
    }
    let a = ints(&[3, 4], DType::U32, 5)?.to_dtype(DType::I64)?;
    let b = ints(&[4, 2], DType::U32, 5)?.to_dtype(DType::I64)?;
    assert!(a.matmul(&b).is_err());
    assert!(a.to_device(&dev)?.matmul(&b.to_device(&dev)?).is_err());
    Ok(())
}

struct CpuOnlyDouble;

impl candle_core::CustomOp1 for CpuOnlyDouble {
    fn name(&self) -> &'static str {
        "cpu-only-double"
    }

    fn cpu_fwd(
        &self,
        s: &candle_core::CpuStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let v = s.as_slice::<f32>()?;
        let out: Vec<f32> = match l.contiguous_offsets() {
            Some((a, b)) => v[a..b].iter().map(|x| x * 2.0).collect(),
            None => candle_core::bail!("contiguous only"),
        };
        Ok((candle_core::CpuStorage::F32(out), l.shape().clone()))
    }
}

#[test]
fn custom_op_fallback_stays_on_device() -> Result<()> {
    let dev = rocm();
    let x = randn(&[3, 4], DType::F32)?;
    let xg = x.to_device(&dev)?.narrow(1, 1, 3)?.contiguous()?;
    let y = xg.apply_op1_no_bwd(&CpuOnlyDouble)?;
    assert!(y.device().is_rocm());
    assert_close(&y, &(x.narrow(1, 1, 3)?.contiguous()? * 2.0)?, "custom op")?;
    let z = (&y + &xg)?;
    assert!(z.device().is_rocm());
    Ok(())
}

#[test]
fn rand_seeded() -> Result<()> {
    let dev = rocm();
    dev.set_seed(42)?;
    let a = Tensor::randn(0f32, 1f32, (64,), &dev)?.to_vec1::<f32>()?;
    dev.set_seed(42)?;
    let b = Tensor::randn(0f32, 1f32, (64,), &dev)?.to_vec1::<f32>()?;
    assert_eq!(a, b);
    let u = Tensor::rand(0f32, 1f32, (1000,), &dev)?;
    let m = u.mean_all()?.to_scalar::<f32>()?;
    assert!((m - 0.5).abs() < 0.1);
    Ok(())
}

#[test]
fn quantized_dequantize_and_matmul() -> Result<()> {
    let dev = rocm();
    let types = [
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
        GgmlDType::Q8K,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::F32,
    ];
    let (n, k) = (40, 512);
    let w = randn(&[n, k], DType::F32)?;
    for dtype in types {
        let q_cpu = QTensor::quantize(&w, dtype)?;
        let deq_cpu = q_cpu.dequantize(&Device::Cpu)?;
        let q_gpu = QTensor::quantize_onto(&w, dtype, &dev)?;
        assert!(q_gpu.device().is_rocm());
        assert_eq!(q_gpu.data()?.as_ref(), q_cpu.data()?.as_ref(), "{dtype:?} data");
        let deq_gpu = q_gpu.dequantize(&dev)?;
        assert!(deq_gpu.device().is_rocm());
        let a = deq_gpu.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let b = deq_cpu.flatten_all()?.to_vec1::<f32>()?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= 1e-6 * y.abs().max(1.0), "{dtype:?} dequant {i}: {x} vs {y}");
        }
        let dq16 = q_gpu.dequantize_f16(&dev)?;
        assert_close(&dq16, &deq_cpu.to_dtype(DType::F16)?, "dequantize_f16")?;

        let mm_cpu = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
        let mm_gpu = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dtype, &dev)?)?;
        for rows in [1usize, 3, 8, 9, 33] {
            let x = randn(&[1, rows, k], DType::F32)?;
            let r_cpu = mm_cpu.forward(&x)?;
            let r_gpu = mm_gpu.forward(&x.to_device(&dev)?)?;
            let r_ref = x.matmul(&deq_cpu.t()?.unsqueeze(0)?)?;
            assert!(r_gpu.device().is_rocm());
            let diff = (r_gpu.to_device(&Device::Cpu)? - &r_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 2e-3, "{dtype:?} qmatmul rows {rows}: diff {diff}");
            let diff_cpu = (r_cpu - &r_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let _ = diff_cpu;
        }
        if !matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32) {
            let x16 = randn(&[2, k], DType::F16)?;
            let r16 = mm_gpu.forward(&x16.to_device(&dev)?)?;
            assert_eq!(r16.dtype(), DType::F16);
            let r16_ref = x16
                .to_dtype(DType::F32)?
                .matmul(&deq_cpu.t()?)?
                .to_dtype(DType::F16)?;
            assert_close(&r16, &r16_ref, "qmatmul f16 input")?;
        }

        if !matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32) {
            let ids = Tensor::new(&[[3u32, 0], [39, 7]], &Device::Cpu)?;
            let e_gpu = q_gpu.embedding(&ids.to_device(&dev)?)?;
            let e_cpu = q_cpu.embedding(&ids)?;
            assert_close(&e_gpu, &e_cpu, "quantized embedding")?;
        }
    }
    let lm_head = randn(&[3000, 256], DType::F32)?;
    let q = QMatMul::from_qtensor(QTensor::quantize_onto(&lm_head, GgmlDType::Q4K, &dev)?)?;
    let x = randn(&[1, 20, 256], DType::F32)?;
    let deq = QTensor::quantize(&lm_head, GgmlDType::Q4K)?.dequantize(&Device::Cpu)?;
    let r = q.forward(&x.to_device(&dev)?)?.to_device(&Device::Cpu)?;
    let r_ref = x.matmul(&deq.t()?.unsqueeze(0)?)?;
    let diff = (r - r_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 2e-3, "tiled dequant gemm diff {diff}");
    Ok(())
}

#[test]
fn device_info() -> Result<()> {
    let dev = rocm();
    let r = dev.as_rocm_device()?;
    let (free, total) = r.mem_info()?;
    assert!(total > 0 && free <= total);
    assert!(!r.name()?.is_empty());
    assert!(candle_core::rocm::device_count() >= 1);
    let d2 = Device::new_rocm(0)?;
    assert!(d2.same_device(&dev));
    let x = Tensor::new(&[1f32, 2., 3.], &dev)?;
    let y = Tensor::new(&[1f32, 2., 3.], &d2)?;
    assert_eq!((x + y)?.to_vec1::<f32>()?, vec![2., 4., 6.]);
    dev.synchronize()?;
    let s = format!("{}", Tensor::new(&[1f32], &dev)?);
    assert!(s.contains("rocm:0"), "{s}");
    let _ = D::Minus1;
    Ok(())
}

#[test]
fn conv_half_fallback() -> Result<()> {
    let dev = rocm();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let x = randn(&[2, 4, 9], dtype)?;
        let w = randn(&[6, 4, 3], dtype)?;
        let r = x.to_device(&dev)?.conv1d(&w.to_device(&dev)?, 1, 1, 1, 1)?;
        let r_ref = x.to_dtype(DType::F32)?.conv1d(&w.to_dtype(DType::F32)?, 1, 1, 1, 1)?.to_dtype(dtype)?;
        assert!(r.device().is_rocm());
        assert_close(&r, &r_ref, "conv1d")?;
        let x2 = randn(&[1, 3, 7, 6], dtype)?;
        let w2 = randn(&[5, 3, 3, 3], dtype)?;
        let xt = x2.to_device(&dev)?.transpose(2, 3)?;
        let r2 = xt.conv2d(&w2.to_device(&dev)?, 1, 1, 1, 1)?;
        let r2_ref = x2
            .to_dtype(DType::F32)?
            .transpose(2, 3)?
            .conv2d(&w2.to_dtype(DType::F32)?, 1, 1, 1, 1)?
            .to_dtype(dtype)?;
        assert_close(&r2, &r2_ref, "conv2d strided input")?;
    }
    Ok(())
}

#[test]
fn f8e4m3_casts() -> Result<()> {
    let dev = rocm();
    let mut vals: Vec<f32> = (0..=255u8)
        .map(|b| float8::F8E4M3::from_bits(b).to_f32())
        .filter(|v| v.is_finite())
        .collect();
    vals.extend_from_slice(&[
        0.0, -0.0, 1e-9, -1e-9, 0.0009765625, 0.001953125, 0.00146484375, 0.0029296875, 0.017578125, 3.3,
        -7.77, 240.0, 447.9, 448.0, 449.0, 463.9, 464.0, 464.1, 1000.0, -1e30, 1e30, 0.1, 0.2, 0.3,
    ]);
    let base = Tensor::new(vals.as_slice(), &Device::Cpu)?;
    for src in [DType::F32, DType::F64, DType::F16, DType::BF16, DType::U8, DType::U32, DType::I64, DType::I16, DType::I32] {
        let x = if src.is_float() { base.to_dtype(src)? } else { base.abs()?.clamp(0f32, 250f32)?.to_dtype(src)? };
        let cpu = x.to_dtype(DType::F8E4M3)?;
        let gpu = x.to_device(&dev)?.to_dtype(DType::F8E4M3)?;
        assert_eq!(gpu.dtype(), DType::F8E4M3);
        let a: Vec<u8> = gpu.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<float8::F8E4M3>()?.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u8> = cpu.flatten_all()?.to_vec1::<float8::F8E4M3>()?.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "{src:?} -> f8e4m3");
    }
    let all: Vec<float8::F8E4M3> = (0..=255u8).map(float8::F8E4M3::from_bits).collect();
    let f8 = Tensor::new(all.as_slice(), &Device::Cpu)?;
    let f8g = f8.to_device(&dev)?;
    for dst in [DType::F32, DType::F64, DType::F16, DType::BF16, DType::U8, DType::U32, DType::I64, DType::I16, DType::I32, DType::F8E4M3] {
        let a = f8g.to_dtype(dst)?.to_device(&Device::Cpu)?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        let b = f8.to_dtype(dst)?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x.is_nan() && y.is_nan()) || x == y, "f8e4m3 -> {dst:?} at {i}: {x} vs {y}");
        }
    }
    let k = randn(&[1, 2, 5, 8], DType::BF16)?;
    let kg = k.to_device(&dev)?.to_dtype(DType::F8E4M3)?;
    let cat = Tensor::cat(&[&kg, &kg.narrow(2, 1, 3)?], 2)?.contiguous()?;
    let back = cat.to_dtype(DType::BF16)?;
    let k8 = k.to_dtype(DType::F8E4M3)?;
    let cat_ref = Tensor::cat(&[&k8, &k8.narrow(2, 1, 3)?], 2)?.to_dtype(DType::BF16)?;
    assert_close(&back, &cat_ref, "f8 kv cat roundtrip")?;
    let strided = kg.transpose(1, 2)?.to_dtype(DType::F32)?;
    assert_close(&strided, &k8.transpose(1, 2)?.to_dtype(DType::F32)?, "f8 strided cast")?;
    Ok(())
}

#[test]
fn dummy_dtype_casts_fail_fast() -> Result<()> {
    let dev = rocm();
    let x = Tensor::ones((2, 3), DType::BF16, &dev)?;
    assert!(x.to_dtype(DType::F4).is_err());
    assert!(x.to_dtype(DType::F6E2M3).is_err());
    assert!(x.to_dtype(DType::F8E8M0).is_err());
    Ok(())
}

#[test]
fn half_matmul_without_gemm_ex() -> Result<()> {
    let dev = rocm();
    std::env::set_var("CANDLE_MOCK_NO_HALF_GEMM", "1");
    let res = (|| -> Result<()> {
        for dtype in [DType::F16, DType::BF16] {
            let a = randn(&[2, 3, 7, 5], dtype)?;
            let bt = randn(&[2, 3, 4, 5], dtype)?;
            let r = a.to_device(&dev)?.matmul(&bt.to_device(&dev)?.transpose(2, 3)?)?;
            assert_eq!(r.dtype(), dtype);
            assert_close(&r, &mm_ref(&a, &bt.transpose(2, 3)?)?, "half matmul via f32")?;
            let w = randn(&[9, 5], dtype)?;
            let x = randn(&[4, 6, 5], dtype)?;
            let r2 = x.to_device(&dev)?.broadcast_matmul(&w.to_device(&dev)?.t()?)?;
            assert_close(&r2, &bmm_ref(&x, &w.t()?)?, "half broadcast matmul via f32")?;
        }
        Ok(())
    })();
    std::env::remove_var("CANDLE_MOCK_NO_HALF_GEMM");
    res
}
