use crate::{bail, DType, Result, Tensor};
use rayon::prelude::*;

pub fn chunk_tril_recurrence(attn: &Tensor) -> Result<Tensor> {
    let dims = attn.dims().to_vec();
    let r = dims.len();
    if r < 2 {
        bail!("chunk_tril_recurrence: expected (..., n, n), got {dims:?}")
    }
    let n = dims[r - 1];
    if dims[r - 2] != n {
        bail!("chunk_tril_recurrence: last two dims must be square, got {dims:?}")
    }
    let dtype = attn.dtype();
    let mut v: Vec<f32> = attn.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mat = n * n;
    if mat > 0 {
        v.par_chunks_mut(mat).for_each(|a| {
            let mut row = vec![0f32; n];
            for i in 1..n {
                row[..i].copy_from_slice(&a[i * n..i * n + i]);
                for c in 0..i {
                    let mut acc = 0f32;
                    for (j, rj) in row[..i].iter().enumerate() {
                        acc += rj * a[j * n + c];
                    }
                    a[i * n + c] = acc + row[c];
                }
            }
        });
    }
    let out = Tensor::from_vec(v, dims, attn.device())?;
    if dtype == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(dtype)
    }
}

