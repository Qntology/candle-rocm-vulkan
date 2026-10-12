//! Low precision element formats and the packed row layout used by the TurboQuant kernels.
//!
//! Every quantized vector ("row" of `dim` values) is stored as little-endian `u32` words:
//!
//! ```text
//! [ codes: code_words ][ aux: aux_words ][ rsc0: f32 bits ][ rsc1: f32 bits ]
//! ```
//!
//! * codes: `code_bits` per element, LSB first, element `j` at bit `j * code_bits`.
//! * aux: QJL sign bits (TurboQuant prod), or one scale byte per block (E8M0 for MX, E4M3 for
//!   NVFP4), byte `k` of the region holding block `k`.
//! * rsc0: vector scale (TurboQuant norm, NVFP4 global scale, FP8 scale; 1 for MX).
//! * rsc1: TurboQuant prod residual norm, 0 otherwise.
//!
//! The functions here are bit exact with the GPU kernels (same algorithm, f32 arithmetic and
//! round-to-nearest-even).

use crate::{bail, Result};

/// Quantization format of a vector (KV cache row, activation row, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// TurboQuant MSE: random rotation, then a Lloyd-Max codebook for the coordinate
    /// distribution of a random unit vector. `bits` per coordinate (1-4) plus one f32 norm per
    /// vector. `bits == 0` means "the configured width".
    TurboMse { bits: u8 },
    /// TurboQuant prod: MSE at `bits - 1` plus a 1-bit QJL sketch of the residual, which makes
    /// inner products unbiased (keys). `bits` in 2-5; 0 means "the configured width".
    TurboProd { bits: u8 },
    /// NVIDIA NVFP4: E2M1 elements, one E4M3 scale per 16 elements, one f32 scale per vector.
    Nvfp4,
    /// OCP MXFP4: E2M1 elements, one E8M0 (power of two) scale per 32 elements.
    Mxfp4,
    /// OCP MXFP8: E4M3 elements, one E8M0 scale per 32 elements.
    Mxfp8,
    /// FP8 E4M3 with one f32 scale per vector (per-token FP8, a.k.a. "nvfp8").
    Fp8,
}

/// Kernel ids of the formats (`mode` argument of the kernels).
pub mod mode {
    pub const TURBO_MSE: u32 = 0;
    pub const TURBO_PROD: u32 = 1;
    pub const MXFP4: u32 = 2;
    pub const NVFP4: u32 = 3;
    pub const MXFP8: u32 = 4;
    pub const FP8: u32 = 5;
}

impl QuantFormat {
    /// Parses `auto` (→ `None`), `turbo` / `mse` / `turbo3`, `prod` / `turbo-prod4`, `nvfp4`,
    /// `mxfp4`, `mxfp8`, `fp8` / `nvfp8` / `e4m3`.
    pub fn parse(s: &str) -> std::result::Result<Option<Self>, String> {
        let t = s.trim().to_ascii_lowercase().replace(['-', '_', ' '], "");
        let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
        let name: String = t.chars().filter(|c| !c.is_ascii_digit()).collect();
        let bits = || -> std::result::Result<u8, String> {
            if digits.is_empty() {
                Ok(0)
            } else {
                digits
                    .parse::<u8>()
                    .map_err(|_| format!("invalid bit width in {s:?}"))
            }
        };
        let f = match name.as_str() {
            "" if digits.is_empty() => return Ok(None),
            "auto" => return Ok(None),
            "turbo" | "tq" | "mse" | "turbomse" | "tqmse" | "turboquant" | "lloydmax" => {
                Self::TurboMse { bits: bits()? }
            }
            "prod" | "turboprod" | "tqprod" | "qjl" | "turboqjl" => Self::TurboProd { bits: bits()? },
            "nvfp" if digits == "4" => Self::Nvfp4,
            "mxfp" if digits == "4" => Self::Mxfp4,
            "mxfp" if digits == "8" => Self::Mxfp8,
            "fp" | "nvfp" if digits == "8" => Self::Fp8,
            "em" if digits == "43" => Self::Fp8,
            _ => return Err(format!(
                "unknown quantization format {s:?} (auto, turbo[1-4], prod[2-5], nvfp4, mxfp4, mxfp8, fp8)"
            )),
        };
        f.check_bits()?;
        Ok(Some(f))
    }

    fn check_bits(&self) -> std::result::Result<(), String> {
        match *self {
            Self::TurboMse { bits } if bits > 4 => {
                Err(format!("turbo (mse) supports 1-4 bits, got {bits}"))
            }
            Self::TurboProd { bits } if bits == 1 || bits > 5 => {
                Err(format!("turbo prod supports 2-5 bits, got {bits}"))
            }
            _ => Ok(()),
        }
    }

    /// Fills in an unspecified (`0`) TurboQuant width with `bits`, clamped to the valid range.
    pub fn with_default_bits(self, bits: u8) -> Self {
        match self {
            Self::TurboMse { bits: 0 } => Self::TurboMse {
                bits: bits.clamp(1, 4),
            },
            Self::TurboProd { bits: 0 } => Self::TurboProd {
                bits: bits.clamp(2, 5),
            },
            f => f,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::TurboMse { bits } => format!("turbo-mse{bits}"),
            Self::TurboProd { bits } => format!("turbo-prod{bits}"),
            Self::Nvfp4 => "nvfp4".into(),
            Self::Mxfp4 => "mxfp4".into(),
            Self::Mxfp8 => "mxfp8".into(),
            Self::Fp8 => "fp8".into(),
        }
    }

    pub fn mode_id(&self) -> u32 {
        match self {
            Self::TurboMse { .. } => mode::TURBO_MSE,
            Self::TurboProd { .. } => mode::TURBO_PROD,
            Self::Mxfp4 => mode::MXFP4,
            Self::Nvfp4 => mode::NVFP4,
            Self::Mxfp8 => mode::MXFP8,
            Self::Fp8 => mode::FP8,
        }
    }

    pub fn is_turbo(&self) -> bool {
        matches!(self, Self::TurboMse { .. } | Self::TurboProd { .. })
    }

    /// Bits of the per element code.
    pub fn code_bits(&self) -> u32 {
        match *self {
            Self::TurboMse { bits } => bits as u32,
            Self::TurboProd { bits } => bits as u32 - 1,
            Self::Nvfp4 | Self::Mxfp4 => 4,
            Self::Mxfp8 | Self::Fp8 => 8,
        }
    }

    /// Elements per scale block (0 when there are no block scales).
    pub fn block_size(&self) -> usize {
        match self {
            Self::Mxfp4 | Self::Mxfp8 => 32,
            Self::Nvfp4 => 16,
            _ => 0,
        }
    }

    /// Codebook width of the TurboQuant MSE stage (0 for the floating point formats).
    pub fn codebook_bits(&self) -> u32 {
        if self.is_turbo() {
            self.code_bits()
        } else {
            0
        }
    }

    /// Checks the format against a vector length.
    pub fn validate(&self, dim: usize) -> Result<()> {
        if let Err(e) = self.check_bits() {
            bail!("{e}")
        }
        if let Self::TurboMse { bits: 0 } | Self::TurboProd { bits: 0 } = self {
            bail!("{}: bit width not resolved", self.name())
        }
        if dim == 0 || dim > MAX_ROW_DIM {
            bail!(
                "{}: vector length {dim} not in 1..={MAX_ROW_DIM}",
                self.name()
            )
        }
        let bs = self.block_size();
        if bs > 0 && dim % bs != 0 {
            bail!(
                "{}: vector length {dim} must be a multiple of the {bs} element block",
                self.name()
            )
        }
        Ok(())
    }

    /// Average storage per element in bits, scales included.
    pub fn bits_per_element(&self, dim: usize) -> f64 {
        row_layout(*self, dim)
            .map(|l| (l.row_words * 32) as f64 / dim as f64)
            .unwrap_or(f64::NAN)
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

/// Largest vector length handled by the row kernels.
pub const MAX_ROW_DIM: usize = 1024;

/// Word layout of one quantized row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    pub dim: usize,
    pub code_bits: u32,
    pub code_words: usize,
    pub aux_words: usize,
    /// Total words per row, the two f32 row scalars included.
    pub row_words: usize,
}

impl RowLayout {
    /// Word index of `rsc0` (followed by `rsc1`).
    pub fn rsc_offset(&self) -> usize {
        self.code_words + self.aux_words
    }
}

pub fn row_layout(format: QuantFormat, dim: usize) -> Result<RowLayout> {
    format.validate(dim)?;
    let code_bits = format.code_bits();
    let code_words = (dim * code_bits as usize).div_ceil(32);
    let aux_words = match format {
        QuantFormat::TurboProd { .. } => dim.div_ceil(32),
        QuantFormat::Mxfp4 | QuantFormat::Mxfp8 | QuantFormat::Nvfp4 => {
            (dim / format.block_size()).div_ceil(4)
        }
        _ => 0,
    };
    Ok(RowLayout {
        dim,
        code_bits,
        code_words,
        aux_words,
        row_words: code_words + aux_words + 2,
    })
}

// ---------------------------------------------------------------------------
// Element codecs (mirrors of kernels/accel_common.h)
// ---------------------------------------------------------------------------

/// `2^e` for `e` in `-126..=127`.
#[inline]
pub fn pow2i(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// `floor(log2(a))` for a positive finite normal `a`; -127 for zero and subnormals.
#[inline]
pub fn floor_log2(a: f32) -> i32 {
    let e = ((a.to_bits() >> 23) & 0xff) as i32;
    if e == 0 {
        -127
    } else {
        e - 127
    }
}

/// OCP E4M3 (FN: no infinities, NaN = 0x7f) to f32.
#[inline]
pub fn e4m3_to_f32(c: u8) -> f32 {
    let s = c & 0x80;
    let e = ((c >> 3) & 0xf) as u32;
    let m = (c & 7) as u32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 {
        m as f32 * 0.001953125
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if s != 0 {
        -v
    } else {
        v
    }
}

/// f32 to E4M3 with round-to-nearest-even and saturation to ±448.
#[inline]
pub fn f32_to_e4m3(f: f32) -> u8 {
    let x = f.to_bits();
    let s = ((x >> 24) & 0x80) as u8;
    let ax = x & 0x7fff_ffff;
    if ax > 0x7f80_0000 {
        return s | 0x7f;
    }
    let a = f32::from_bits(ax);
    if a >= 448.0 {
        return s | 0x7e;
    }
    if a < 0.015625 {
        let q = (a * 512.0).round_ties_even();
        return s | q as u8;
    }
    let r = ax + 0x7ffff + ((ax >> 20) & 1);
    let e = (r >> 23) - 120;
    let m = (r >> 20) & 7;
    let code = ((e << 3) | m).min(0x7e);
    s | code as u8
}

/// E2M1 code (4 bits, sign in bit 3) to f32.
#[inline]
pub fn fp4_to_f32(c: u32) -> f32 {
    let e = (c >> 1) & 3;
    let m = c & 1;
    let v = if e == 0 {
        0.5 * m as f32
    } else {
        f32::from_bits(((e + 126) << 23) | (m << 22))
    };
    if c & 8 != 0 {
        -v
    } else {
        v
    }
}

/// f32 to E2M1 with round-to-nearest-even and saturation to ±6.
#[inline]
pub fn f32_to_fp4(f: f32) -> u32 {
    let s = (f.to_bits() >> 28) & 8;
    let a = f.abs();
    let c = if a.is_nan() || a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    s | c
}

/// E8M0 scale code to f32 (`2^(c - 127)`).
#[inline]
pub fn e8m0_to_f32(c: u8) -> f32 {
    if c == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits((c as u32) << 23)
    }
}

/// OCP MX shared exponent code for a block with maximum magnitude `amax`, for elements whose
/// largest power of two is `2^emax_elem` (2 for E2M1, 8 for E4M3).
#[inline]
pub fn mx_scale_code(amax: f32, emax_elem: i32) -> u8 {
    let e = if amax > 0.0 && amax.is_finite() {
        floor_log2(amax) - emax_elem
    } else {
        -127
    };
    (e.clamp(-127, 127) + 127) as u8
}

/// Reads `bits` (<= 8) bits at bit offset `pos` of a little-endian word stream.
#[inline]
pub fn get_bits(words: &[u32], pos: usize, bits: u32) -> u32 {
    let w = pos >> 5;
    let sh = (pos & 31) as u32;
    let mask = (1u32 << bits) - 1;
    let mut v = words[w] >> sh;
    if sh + bits > 32 {
        v |= words[w + 1] << (32 - sh);
    }
    v & mask
}

/// Writes `bits` bits of `v` at bit offset `pos` (the target bits must be zero).
#[inline]
pub fn put_bits(words: &mut [u32], pos: usize, bits: u32, v: u32) {
    let w = pos >> 5;
    let sh = (pos & 31) as u32;
    let v = v & ((1u32 << bits) - 1);
    words[w] |= v << sh;
    if sh + bits > 32 {
        words[w + 1] |= v >> (32 - sh);
    }
}

#[inline]
pub fn get_byte(words: &[u32], k: usize) -> u8 {
    (words[k >> 2] >> ((k & 3) * 8)) as u8
}

#[inline]
pub fn put_byte(words: &mut [u32], k: usize, b: u8) {
    words[k >> 2] |= (b as u32) << ((k & 3) * 8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_all_codes() {
        for c in 0u8..=255 {
            let v = e4m3_to_f32(c);
            if v.is_nan() {
                continue;
            }
            let back = f32_to_e4m3(v);
            // +0 / -0 both decode to zero
            if v == 0.0 {
                assert_eq!(back & 0x7f, 0);
                continue;
            }
            assert_eq!(back, c, "code {c:#x} value {v}");
        }
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(f32_to_e4m3(1000.0), 0x7e);
        assert_eq!(f32_to_e4m3(-1000.0), 0xfe);
        assert_eq!(e4m3_to_f32(0x01), 0.001953125);
    }

    #[test]
    fn e4m3_matches_float8_crate() {
        // The float8 crate is the reference used by candle's F8E4M3 dtype.
        let mut x = -500.0f32;
        while x < 500.0 {
            let ours = e4m3_to_f32(f32_to_e4m3(x));
            let theirs = float8::F8E4M3::from_f32(x.clamp(-448.0, 448.0)).to_f32();
            assert_eq!(ours, theirs, "x = {x}");
            x += 0.37;
        }
    }

    #[test]
    fn fp4_codes() {
        let vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (c, v) in vals.iter().enumerate() {
            assert_eq!(fp4_to_f32(c as u32), *v);
            assert_eq!(fp4_to_f32(c as u32 | 8), -*v);
            assert_eq!(f32_to_fp4(*v), c as u32);
        }
        assert_eq!(f32_to_fp4(0.75), 2); // tie to even
        assert_eq!(f32_to_fp4(2.5), 4);
        assert_eq!(f32_to_fp4(5.0), 6);
        assert_eq!(f32_to_fp4(100.0), 7);
    }

    #[test]
    fn mx_scales() {
        assert_eq!(mx_scale_code(6.0, 2), 127); // 6 = 1.5 * 2^2 → scale 2^0
        assert_eq!(e8m0_to_f32(mx_scale_code(12.0, 2)), 2.0);
        assert_eq!(mx_scale_code(0.0, 2), 0);
    }

    #[test]
    fn layouts() {
        let l = row_layout(QuantFormat::TurboMse { bits: 3 }, 128).unwrap();
        assert_eq!((l.code_words, l.aux_words, l.row_words), (12, 0, 14));
        let l = row_layout(QuantFormat::TurboProd { bits: 4 }, 128).unwrap();
        assert_eq!((l.code_words, l.aux_words, l.row_words), (12, 4, 18));
        let l = row_layout(QuantFormat::Mxfp4, 64).unwrap();
        assert_eq!((l.code_words, l.aux_words), (8, 1));
        let l = row_layout(QuantFormat::Nvfp4, 128).unwrap();
        assert_eq!((l.code_words, l.aux_words), (16, 2));
        assert!(row_layout(QuantFormat::Mxfp8, 48).is_err());
        assert_eq!(QuantFormat::parse("auto").unwrap(), None);
        assert_eq!(
            QuantFormat::parse("turbo3").unwrap(),
            Some(QuantFormat::TurboMse { bits: 3 })
        );
        assert_eq!(
            QuantFormat::parse("NVFP4").unwrap(),
            Some(QuantFormat::Nvfp4)
        );
        assert_eq!(QuantFormat::parse("nvfp8").unwrap(), Some(QuantFormat::Fp8));
        assert_eq!(
            QuantFormat::parse("mx-fp8").unwrap(),
            Some(QuantFormat::Mxfp8)
        );
        assert!(QuantFormat::parse("turbo9").is_err());
    }

    #[test]
    fn bit_stream() {
        let mut w = vec![0u32; 4];
        for j in 0..40 {
            put_bits(&mut w, j * 3, 3, (j % 8) as u32);
        }
        for j in 0..40 {
            assert_eq!(get_bits(&w, j * 3, 3), (j % 8) as u32);
        }
    }
}
