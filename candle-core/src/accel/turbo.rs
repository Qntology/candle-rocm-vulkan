//! TurboQuant ("Online Vector Quantization with Near-optimal Distortion Rate", Zandieh et al.
//! 2025) building blocks and the CPU reference codec for every [`QuantFormat`].
//!
//! * Rotation: a random orthogonal `Π` makes every coordinate of a unit vector follow the same
//!   known distribution (density `∝ (1 - x²)^((d-3)/2)`), so one scalar codebook fits all
//!   coordinates. `Π` is a randomized Hadamard transform (`H·diag(±1)/√d`) or, like the
//!   reference implementation, a Haar random matrix from the QR decomposition of a Gaussian one.
//! * MSE stage: Lloyd-Max codebook of that distribution, `b` bits per coordinate, one f32 norm
//!   per vector.
//! * Prod stage: MSE at `b - 1` bits plus a 1-bit QJL sketch `sign(S·r)` of the residual `r`,
//!   with the residual norm `γ`; `⟨x, y⟩ ≈ ‖x‖ (⟨x̃, y⟩ + γ √(π/2)/d ⟨z, S y⟩)` is unbiased.
//!
//! Rotation and QJL projections are dense `d x d` matrices applied with regular matmuls, so the
//! GPU kernels only implement the scalar part and work on every backend.

use super::config::RotationKind;
use super::formats::{
    e4m3_to_f32, e8m0_to_f32, f32_to_e4m3, f32_to_fp4, fp4_to_f32, get_bits, get_byte,
    mx_scale_code, put_bits, put_byte, row_layout, QuantFormat, RowLayout,
};
use crate::{bail, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// `sqrt(pi / 2)`, the QJL estimator constant.
pub const QJL_CONST: f32 = 1.253_314_1;

/// Lloyd-Max codebook for the coordinates of a random unit vector of length `dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct Codebook {
    pub dim: usize,
    pub bits: u32,
    /// `2^bits` centroids, ascending, in unit-vector scale.
    pub centroids: Vec<f32>,
    /// `2^bits - 1` decision boundaries (midpoints of the centroids).
    pub boundaries: Vec<f32>,
}

impl Codebook {
    /// Index of `u`: the number of boundaries `<= u` (the GPU kernels use the same rule).
    #[inline]
    pub fn index(&self, u: f32) -> u32 {
        self.boundaries.iter().filter(|&&b| u >= b).count() as u32
    }

    /// `[centroids; 16][boundaries; 16]`, the table layout read by the kernels.
    pub fn table(&self) -> [f32; 32] {
        let mut t = [0f32; 32];
        t[..self.centroids.len()].copy_from_slice(&self.centroids);
        t[16..16 + self.boundaries.len()].copy_from_slice(&self.boundaries);
        for v in t[16 + self.boundaries.len()..].iter_mut() {
            *v = f32::INFINITY;
        }
        t
    }
}

fn lloyd_max(dim: usize, bits: u32) -> Codebook {
    let k = 1usize << bits;
    // Density of one coordinate of a uniform random unit vector in R^dim. Integrate on the
    // region holding the mass: |x| < 12/sqrt(dim) (clamped to the support).
    let lim = (12.0 / (dim as f64).sqrt()).min(1.0);
    let n = 1usize << 16;
    let h = 2.0 * lim / n as f64;
    let expo = (dim as f64 - 3.0) / 2.0;
    let xs: Vec<f64> = (0..n).map(|i| -lim + (i as f64 + 0.5) * h).collect();
    let mut w: Vec<f64> = xs
        .iter()
        .map(|&x| {
            let t = (1.0 - x * x).max(1e-300);
            (expo * t.ln()).exp()
        })
        .collect();
    let total: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= total;
    }
    // prefix sums of the mass and of the first moment
    let mut cw = vec![0f64; n + 1];
    let mut cx = vec![0f64; n + 1];
    for i in 0..n {
        cw[i + 1] = cw[i] + w[i];
        cx[i + 1] = cx[i] + w[i] * xs[i];
    }
    // first grid index with x >= b
    let split = |b: f64| -> usize { xs.partition_point(|&x| x < b) };
    // quantile initialization
    let mut c: Vec<f64> = (0..k)
        .map(|i| {
            let p = (i as f64 + 0.5) / k as f64;
            let j = cw.partition_point(|&v| v < p).clamp(1, n);
            xs[j - 1]
        })
        .collect();
    for _ in 0..2000 {
        let bnd: Vec<f64> = (0..k - 1).map(|i| 0.5 * (c[i] + c[i + 1])).collect();
        let mut delta = 0f64;
        let mut lo = 0usize;
        for i in 0..k {
            let hi = if i + 1 < k { split(bnd[i]) } else { n };
            let mass = cw[hi] - cw[lo];
            if mass > 0.0 {
                let m = (cx[hi] - cx[lo]) / mass;
                delta = delta.max((m - c[i]).abs());
                c[i] = m;
            }
            lo = hi;
        }
        if delta < 1e-12 {
            break;
        }
    }
    // exact symmetry
    for i in 0..k / 2 {
        let v = 0.5 * (c[k - 1 - i] - c[i]);
        c[i] = -v;
        c[k - 1 - i] = v;
    }
    let centroids: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let boundaries: Vec<f32> = (0..k - 1)
        .map(|i| (0.5 * (c[i] + c[i + 1])) as f32)
        .collect();
    Codebook {
        dim,
        bits,
        centroids,
        boundaries,
    }
}

/// The (cached) codebook of `bits` bits for vectors of length `dim`.
pub fn codebook(dim: usize, bits: u32) -> Result<Arc<Codebook>> {
    if !(1..=4).contains(&bits) {
        bail!("turbo-quant codebooks have 1 to 4 bits, got {bits}")
    }
    if dim < 2 {
        bail!("turbo-quant needs vectors of length >= 2, got {dim}")
    }
    static CACHE: Mutex<Option<HashMap<(usize, u32), Arc<Codebook>>>> = Mutex::new(None);
    let mut g = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    let map = g.get_or_insert_with(HashMap::new);
    Ok(map
        .entry((dim, bits))
        .or_insert_with(|| Arc::new(lloyd_max(dim, bits)))
        .clone())
}

/// SplitMix64, the generator of the rotation and QJL matrices (stable across releases).
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Row-major `dim x dim` orthogonal matrix `Π` (forward rotation `y = Π x`).
/// `kind` must be resolved (see [`RotationKind::resolve`]); `None` gives the identity.
pub fn rotation_matrix(dim: usize, kind: RotationKind, seed: u64) -> Result<Vec<f32>> {
    let mut m = vec![0f32; dim * dim];
    match kind.resolve(dim) {
        RotationKind::None => {
            for i in 0..dim {
                m[i * dim + i] = 1.0;
            }
        }
        RotationKind::Hadamard => {
            if !dim.is_power_of_two() {
                bail!("hadamard rotation needs a power of two length, got {dim}")
            }
            let mut rng = SplitMix64::new(seed ^ 0x4841_4441_4d41_5244);
            let signs: Vec<f32> = (0..dim)
                .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
                .collect();
            let s = 1.0 / (dim as f32).sqrt();
            for i in 0..dim {
                for j in 0..dim {
                    let h = if (i & j).count_ones() % 2 == 0 { s } else { -s };
                    m[i * dim + j] = h * signs[j];
                }
            }
        }
        RotationKind::Dense | RotationKind::Auto => {
            let mut rng = SplitMix64::new(seed ^ 0x524f_5441_5445_2121);
            let mut rows: Vec<Vec<f64>> = (0..dim)
                .map(|_| (0..dim).map(|_| rng.next_normal()).collect())
                .collect();
            // modified Gram-Schmidt, two passes for orthogonality to f64 precision
            for _ in 0..2 {
                for i in 0..dim {
                    for j in 0..i {
                        let dot: f64 = (0..dim).map(|t| rows[i][t] * rows[j][t]).sum();
                        for t in 0..dim {
                            let v = rows[j][t];
                            rows[i][t] -= dot * v;
                        }
                    }
                    let n: f64 = rows[i].iter().map(|v| v * v).sum::<f64>().sqrt();
                    for v in rows[i].iter_mut() {
                        *v /= n;
                    }
                }
            }
            for i in 0..dim {
                for j in 0..dim {
                    m[i * dim + j] = rows[i][j] as f32;
                }
            }
        }
    }
    Ok(m)
}

/// Row-major `dim x dim` matrix of i.i.d. standard normal entries: the QJL projection `S`.
pub fn qjl_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed.wrapping_add(1) ^ 0x514a_4c5f_534b_4554);
    (0..dim * dim).map(|_| rng.next_normal() as f32).collect()
}

/// CPU codec of one format for vectors of a given length.
#[derive(Debug, Clone)]
pub struct RowCodec {
    pub format: QuantFormat,
    pub layout: RowLayout,
    pub codebook: Option<Arc<Codebook>>,
    /// QJL matrix (prod only), row-major `dim x dim`.
    pub qjl: Option<Arc<Vec<f32>>>,
}

impl RowCodec {
    pub fn new(format: QuantFormat, dim: usize, seed: u64) -> Result<Self> {
        let layout = row_layout(format, dim)?;
        let codebook = if format.is_turbo() {
            Some(codebook(dim, format.code_bits())?)
        } else {
            None
        };
        let qjl = match format {
            QuantFormat::TurboProd { .. } => Some(Arc::new(qjl_matrix(dim, seed))),
            _ => None,
        };
        Ok(Self {
            format,
            layout,
            codebook,
            qjl,
        })
    }

    /// The codec without the QJL matrix: stage one of prod rows (codes and norm, empty sketch),
    /// element access and the QJL coefficients.
    pub fn without_qjl(format: QuantFormat, dim: usize) -> Result<Self> {
        let layout = row_layout(format, dim)?;
        let codebook = if format.is_turbo() {
            Some(codebook(dim, format.code_bits())?)
        } else {
            None
        };
        Ok(Self {
            format,
            layout,
            codebook,
            qjl: None,
        })
    }

    pub fn dim(&self) -> usize {
        self.layout.dim
    }

    /// Quantizes one (already rotated) vector into `out` (`row_words` words).
    pub fn quantize_row(&self, y: &[f32], out: &mut [u32]) {
        let l = &self.layout;
        let d = l.dim;
        out.iter_mut().for_each(|w| *w = 0);
        let (rsc0, rsc1);
        match self.format {
            QuantFormat::TurboMse { .. } | QuantFormat::TurboProd { .. } => {
                let cb = self.codebook.as_ref().expect("codebook");
                let norm = y.iter().map(|v| v * v).sum::<f32>().sqrt();
                let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                let mut resid = vec![0f32; d];
                for (j, &v) in y.iter().enumerate() {
                    let u = v * inv;
                    let idx = cb.index(u);
                    put_bits(out, j * l.code_bits as usize, l.code_bits, idx);
                    resid[j] = u - cb.centroids[idx as usize];
                }
                rsc0 = norm;
                if let Some(s) = &self.qjl {
                    rsc1 = resid.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let signs = &mut out[l.code_words..l.code_words + l.aux_words];
                    for i in 0..d {
                        let p: f32 = (0..d).map(|j| s[i * d + j] * resid[j]).sum();
                        if p >= 0.0 {
                            signs[i >> 5] |= 1 << (i & 31);
                        }
                    }
                } else {
                    rsc1 = 0.0;
                }
            }
            QuantFormat::Mxfp4 | QuantFormat::Mxfp8 => {
                let fp4 = self.format == QuantFormat::Mxfp4;
                let bs = 32;
                for b in 0..d / bs {
                    let blk = &y[b * bs..(b + 1) * bs];
                    let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let code = mx_scale_code(amax, if fp4 { 2 } else { 8 });
                    let scale = e8m0_to_f32(code);
                    put_byte(&mut out[l.code_words..], b, code);
                    for (t, &v) in blk.iter().enumerate() {
                        // zero scale code: all-zero (or subnormal) block, zero codes
                        let e = if code == 0 {
                            0
                        } else if fp4 {
                            f32_to_fp4(v / scale)
                        } else {
                            f32_to_e4m3(v / scale) as u32
                        };
                        put_bits(out, (b * bs + t) * l.code_bits as usize, l.code_bits, e);
                    }
                }
                rsc0 = 1.0;
                rsc1 = 0.0;
            }
            QuantFormat::Nvfp4 => {
                let amax = y.iter().fold(0f32, |m, v| m.max(v.abs()));
                let g = if amax > 0.0 { amax / 2688.0 } else { 1.0 };
                let bs = 16;
                for b in 0..d / bs {
                    let blk = &y[b * bs..(b + 1) * bs];
                    let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let cs = f32_to_e4m3(bmax / 6.0 / g);
                    let eff = e4m3_to_f32(cs) * g;
                    put_byte(&mut out[l.code_words..], b, cs);
                    for (t, &v) in blk.iter().enumerate() {
                        let e = if eff > 0.0 { f32_to_fp4(v / eff) } else { 0 };
                        put_bits(out, (b * bs + t) * 4, 4, e);
                    }
                }
                rsc0 = g;
                rsc1 = 0.0;
            }
            QuantFormat::Fp8 => {
                let amax = y.iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
                for (j, &v) in y.iter().enumerate() {
                    put_bits(out, j * 8, 8, f32_to_e4m3(v / s) as u32);
                }
                rsc0 = s;
                rsc1 = 0.0;
            }
        }
        let o = l.rsc_offset();
        out[o] = rsc0.to_bits();
        out[o + 1] = rsc1.to_bits();
    }

    /// Element `j` of a row without the `rsc0` factor (the kernels' `tq_elem`).
    #[inline]
    pub fn elem(&self, row: &[u32], j: usize) -> f32 {
        let l = &self.layout;
        let code = get_bits(row, j * l.code_bits as usize, l.code_bits);
        let aux = &row[l.code_words..];
        match self.format {
            QuantFormat::TurboMse { .. } | QuantFormat::TurboProd { .. } => {
                self.codebook.as_ref().expect("codebook").centroids[code as usize]
            }
            QuantFormat::Mxfp4 => fp4_to_f32(code) * e8m0_to_f32(get_byte(aux, j / 32)),
            QuantFormat::Nvfp4 => fp4_to_f32(code) * e4m3_to_f32(get_byte(aux, j / 16)),
            QuantFormat::Mxfp8 => e4m3_to_f32(code as u8) * e8m0_to_f32(get_byte(aux, j / 32)),
            QuantFormat::Fp8 => e4m3_to_f32(code as u8),
        }
    }

    pub fn rsc(&self, row: &[u32]) -> (f32, f32) {
        let o = self.layout.rsc_offset();
        (f32::from_bits(row[o]), f32::from_bits(row[o + 1]))
    }

    /// Reconstructs one vector in the rotated domain (QJL correction included).
    pub fn dequantize_row(&self, row: &[u32], out: &mut [f32]) {
        let d = self.layout.dim;
        let (r0, r1) = self.rsc(row);
        for (j, o) in out.iter_mut().enumerate().take(d) {
            *o = r0 * self.elem(row, j);
        }
        if let Some(s) = &self.qjl {
            let c = r0 * r1 * QJL_CONST / d as f32;
            let signs = &row[self.layout.code_words..];
            for i in 0..d {
                let z = if (signs[i >> 5] >> (i & 31)) & 1 == 1 {
                    c
                } else {
                    -c
                };
                for (j, o) in out.iter_mut().enumerate().take(d) {
                    *o += z * s[i * d + j];
                }
            }
        }
    }

    /// The QJL part of the dequantized row as `±rsc0·rsc1·√(π/2)/d` per sketch bit (prod only):
    /// the vector the GPU path multiplies with `S` to add the correction.
    pub fn qjl_signs(&self, row: &[u32], out: &mut [f32]) {
        let d = self.layout.dim;
        let (r0, r1) = self.rsc(row);
        let c = r0 * r1 * QJL_CONST / d as f32;
        let signs = &row[self.layout.code_words..];
        for (i, o) in out.iter_mut().enumerate().take(d) {
            *o = if (signs[i >> 5] >> (i & 31)) & 1 == 1 {
                c
            } else {
                -c
            };
        }
    }
}

/// Kernel arguments of the row kernels (`tq_quant`, `tq_resid`, `tq_signs`, `tq_dequant`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TqRowArgs {
    pub rows: usize,
    pub dim: usize,
    pub mode: u32,
    pub code_bits: u32,
    pub code_words: usize,
    pub aux_words: usize,
    pub row_words: usize,
}

impl TqRowArgs {
    pub fn new(format: QuantFormat, rows: usize, dim: usize) -> Result<Self> {
        let l = row_layout(format, dim)?;
        Ok(Self {
            rows,
            dim,
            mode: format.mode_id(),
            code_bits: l.code_bits,
            code_words: l.code_words,
            aux_words: l.aux_words,
            row_words: l.row_words,
        })
    }

    pub fn rsc_offset(&self) -> usize {
        self.code_words + self.aux_words
    }

    /// Dynamic shared memory of `tq_quant`.
    pub fn quant_smem_bytes(&self) -> usize {
        (2 * self.dim + TQ_THREADS + self.dim / 16 + 1) * 4
    }
}

/// Threads per workgroup of the TurboQuant kernels.
pub const TQ_THREADS: usize = 128;

/// Format of one side (keys or values) of [`TqAttnArgs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TqSide {
    pub mode: u32,
    pub code_bits: u32,
    pub code_words: usize,
    pub rsc_offset: usize,
}

impl TqSide {
    pub fn new(format: QuantFormat, dim: usize) -> Result<Self> {
        let l = row_layout(format, dim)?;
        Ok(Self {
            mode: format.mode_id(),
            code_bits: l.code_bits,
            code_words: l.code_words,
            rsc_offset: l.rsc_offset(),
        })
    }
}

/// Kernel arguments of `tq_attn`: attention of rotated queries `(b, hq, lq, d)` (followed by
/// `S·q` when the keys are prod rows) over packed K/V rows `(b, hkv, t, words)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TqAttnArgs {
    pub b: usize,
    pub lq: usize,
    pub lk: usize,
    pub hq: usize,
    pub hkv: usize,
    pub d: usize,
    /// `(batch, head, seq)` strides of the query rows, in f32 elements.
    pub q_strides: [usize; 3],
    pub q_offset: usize,
    /// `(batch, head, seq)` strides of the key rows, in words.
    pub k_strides: [usize; 3],
    /// `(batch, head, seq)` strides of the value rows, in words.
    pub v_strides: [usize; 3],
    pub k_offset: usize,
    pub v_offset: usize,
    pub k: TqSide,
    pub v: TqSide,
    pub scale: f32,
    pub softcap: f32,
    pub window_left: i32,
    pub window_right: i32,
    pub splits: usize,
    pub alibi_bstride: usize,
}

/// Kernel of a `tq_attn` call: the FP8 specialization when keys and values are both per-vector
/// FP8 rows (bit-identical results to the generic kernel; switched off with
/// `TurboQuantOptions::fp8_fast_path` / `CANDLE_TURBO_QUANT_FP8_FAST=0`), the `tq_attnp*`
/// variants for paged block tables.
pub fn tq_attn_kernel(k: &TqSide, v: &TqSide, paged: bool) -> &'static str {
    use super::formats::mode;
    let fp8 = k.mode == mode::FP8 && v.mode == mode::FP8 && super::config().turbo_quant.fp8_fast_path;
    match (paged, fp8) {
        (false, false) => "tq_attn",
        (false, true) => "tq_attn_fp8",
        (true, false) => "tq_attnp",
        (true, true) => "tq_attnp_fp8",
    }
}

impl TqAttnArgs {
    /// Output rows `b * hq * lq` (layout `(b, hq, lq, d)`).
    pub fn out_rows(&self) -> usize {
        self.b * self.hq * self.lq
    }

    pub fn smem_bytes(&self) -> usize {
        (2 * self.d + 2 * TQ_THREADS + 32) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_matches_gaussian_lloyd_max() {
        // For large d the coordinates are ~N(0, 1/d); the 2 bit Gaussian Lloyd-Max centroids
        // are ±0.4528 and ±1.5104 (times 1/sqrt(d)).
        let d = 128;
        let cb = codebook(d, 2).unwrap();
        let s = (d as f32).sqrt();
        let c: Vec<f32> = cb.centroids.iter().map(|v| v * s).collect();
        assert!((c[2] - 0.4528).abs() < 0.01, "{c:?}");
        assert!((c[3] - 1.5104).abs() < 0.02, "{c:?}");
        assert!((c[0] + c[3]).abs() < 1e-6);
        let cb1 = codebook(d, 1).unwrap();
        assert!((cb1.centroids[1] * s - 0.7979).abs() < 0.01);
    }

    #[test]
    fn rotations_are_orthogonal() {
        for (d, kind) in [
            (64, RotationKind::Hadamard),
            (48, RotationKind::Dense),
            (16, RotationKind::None),
        ] {
            let m = rotation_matrix(d, kind, 7).unwrap();
            for i in 0..d {
                for j in 0..d {
                    let dot: f32 = (0..d).map(|t| m[i * d + t] * m[j * d + t]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-4, "{kind:?} {i} {j} {dot}");
                }
            }
        }
    }

    fn rand_vec(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
        (0..d).map(|_| rng.next_normal() as f32).collect()
    }

    #[test]
    fn roundtrip_errors() {
        let d = 128;
        let mut rng = SplitMix64::new(3);
        let formats = [
            (QuantFormat::TurboMse { bits: 4 }, 0.012),
            (QuantFormat::TurboMse { bits: 2 }, 0.13),
            (QuantFormat::TurboProd { bits: 4 }, 0.08),
            (QuantFormat::Nvfp4, 0.012),
            (QuantFormat::Mxfp4, 0.02),
            // the OCP power-of-two block scale can push the block maximum into (448, 512),
            // where E4M3 saturates
            (QuantFormat::Mxfp8, 0.0015),
            (QuantFormat::Fp8, 0.0008),
        ];
        for (f, max_rel_mse) in formats {
            let codec = RowCodec::new(f, d, 11).unwrap();
            let mut words = vec![0u32; codec.layout.row_words];
            let mut back = vec![0f32; d];
            let mut err = 0f64;
            let mut nrm = 0f64;
            for _ in 0..64 {
                let x = rand_vec(&mut rng, d);
                codec.quantize_row(&x, &mut words);
                codec.dequantize_row(&words, &mut back);
                err += x
                    .iter()
                    .zip(&back)
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>();
                nrm += x.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
            }
            let rel = err / nrm;
            assert!(rel < max_rel_mse, "{f}: relative mse {rel}");
        }
    }

    #[test]
    fn prod_inner_products_are_unbiased() {
        let d = 64;
        let mut rng = SplitMix64::new(5);
        let y = rand_vec(&mut rng, d);
        let x = rand_vec(&mut rng, d);
        let exact: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let mut mean_prod = 0f64;
        let mut mean_mse = 0f64;
        let n = 400;
        let mut back = vec![0f32; d];
        for seed in 0..n {
            // a fresh random rotation per trial: the estimator is unbiased over the randomness
            let rot = rotation_matrix(d, RotationKind::Dense, seed).unwrap();
            let rx: Vec<f32> = (0..d)
                .map(|i| (0..d).map(|j| rot[i * d + j] * x[j]).sum())
                .collect();
            let ry: Vec<f32> = (0..d)
                .map(|i| (0..d).map(|j| rot[i * d + j] * y[j]).sum())
                .collect();
            for (f, acc) in [
                (QuantFormat::TurboProd { bits: 2 }, &mut mean_prod),
                (QuantFormat::TurboMse { bits: 1 }, &mut mean_mse),
            ] {
                let codec = RowCodec::new(f, d, seed).unwrap();
                let mut words = vec![0u32; codec.layout.row_words];
                codec.quantize_row(&rx, &mut words);
                codec.dequantize_row(&words, &mut back);
                let ip: f32 = back.iter().zip(&ry).map(|(a, b)| a * b).sum();
                *acc += ip as f64 / n as f64;
            }
        }
        let scale = (x.iter().map(|v| v * v).sum::<f32>() * y.iter().map(|v| v * v).sum::<f32>())
            .sqrt() as f64;
        // prod is unbiased; 1-bit MSE shrinks inner products by 2/pi
        assert!(
            (mean_prod - exact as f64).abs() < 0.06 * scale,
            "prod {mean_prod} vs {exact}"
        );
        assert!(
            (mean_mse - exact as f64 * 2.0 / std::f64::consts::PI).abs() < 0.06 * scale,
            "mse {mean_mse} vs {exact}"
        );
    }
}
