//! Host side helpers for the gated delta rule (linear attention, e.g. Qwen3-Next /
//! Qwen3.5), for backends whose tensors live in host memory (CPU, Vulkan).
//!
//! The chunked form of the rule runs a row by row recurrence over every
//! `chunk x chunk` block. Written with tensor ops it becomes `chunk` rounds of
//! narrow / broadcast_mul / sum / slice_assign over the whole attention tensor,
//! which is cheap as GPU kernels but costs a full copy of the tensor per round
//! when every op runs on the CPU. These helpers do the same arithmetic in place.

use crate::{bail, DType, Result, Tensor};
use rayon::prelude::*;

/// Lower triangular recurrence of the chunked gated delta rule.
///
/// `attn` has shape `(..., n, n)`; every trailing `n x n` matrix `A` must be
/// strictly lower triangular. Rows are rewritten in order `i = 1..n` as
///
/// `A[i, :i] = A[i, :i] + A[i, :i] · A[:i, :i]`
///
/// where `A[:i, :i]` already holds the rewritten rows, i.e. the reference loop
/// `attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)`. The result is
/// returned with the dtype and on the device of `attn` (computed in f32 on the host).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, IndexOp, D};

    #[test]
    fn matches_reference_loop() -> Result<()> {
        let dev = Device::Cpu;
        let n = 8usize;
        let x = Tensor::randn(0f32, 0.3f32, (2, 3, n, n), &dev)?;
        let mask = Tensor::tril2(n, DType::F32, &dev)?.affine(1.0, 0.0)?;
        let eye = Tensor::eye(n, DType::F32, &dev)?;
        let strict = mask.broadcast_sub(&eye)?;
        let mut attn = x.broadcast_mul(&strict)?;
        let fast = chunk_tril_recurrence(&attn)?;
        let (d0, d1, _, _) = attn.dims4()?;
        for i in 1..n {
            let row = attn.i((.., .., i, ..i))?.contiguous()?;
            let sub = attn.i((.., .., ..i, ..i))?.contiguous()?;
            let attn_i = row
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&sub)?
                .sum(D::Minus2)?
                .add(&row)?
                .unsqueeze(D::Minus2)?;
            attn = attn.slice_assign(&[0..d0, 0..d1, i..i + 1, 0..i], &attn_i)?;
        }
        let diff = (fast - attn)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max diff {diff}");
        Ok(())
    }
}
