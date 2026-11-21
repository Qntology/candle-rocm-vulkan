//! Shared planning, NVRTC source generation, and CPU F32 reference for the fused decoder
//! kernels (`kernels/decoder_ops_impl.h`: `ResidualRmsNorm`, `SiluMul`, `RopeKvWrite`).

use super::sources::{strip_includes, ACCEL_COMMON_H};

pub const DEC_OPS_IMPL_H: &str = include_str!("../../kernels/decoder_ops_impl.h");

pub const DEC_NT: u32 = 256;
pub const RES_RMS_PLUS_ONE_WEIGHT: u32 = 1;
pub const RES_RMS_WRITE_RESIDUAL: u32 = 2;
pub const ROPE_MODE_NEOX: u32 = 0;
pub const ROPE_MODE_ADJACENT: u32 = 1;
pub const ROPE_NORM_PLUS_ONE: u32 = 2;

/// RoPE pairing convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeMode {
    /// Split-half NeoX rotation: `(x[i], x[i + rotary_dim/2])`.
    NeoX,
    /// Interleaved adjacent-pair rotation: `(x[2i], x[2i + 1])`.
    Adjacent,
}

impl RopeMode {
    #[inline]
    pub fn flag(self) -> u32 {
        match self {
            RopeMode::NeoX => ROPE_MODE_NEOX,
            RopeMode::Adjacent => ROPE_MODE_ADJACENT,
        }
    }
}

/// 4D attention tensor layout for `rope_kv_write`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadLayout {
    /// `[B, T, H, D]`
    Bshd,
    /// `[B, H, T, D]`
    Bhsd,
}

/// Strides (in elements) and metadata for launching `rope_kv_write_*`.
#[derive(Debug, Clone, Copy)]
pub struct RopeKvLaunchArgs {
    pub b: u32,
    pub t: u32,
    pub hq: u32,
    pub hkv: u32,
    pub d: u32,
    pub rotary_dim: u32,
    pub past_len: u32,
    pub q_off: usize,
    pub k_off: usize,
    pub v_off: usize,
    pub qnw_off: usize,
    pub knw_off: usize,
    pub cos_off: usize,
    pub sin_off: usize,
    pub kc_off: usize,
    pub vc_off: usize,
    pub q_strides: [u64; 3],   // [sb, sh, st]
    pub qo_strides: [u64; 3],  // [sb, sh, st]
    pub k_strides: [u64; 3],   // [sb, sh, st]
    pub v_strides: [u64; 3],   // [sb, sh, st]
    pub kc_strides: [u64; 3],  // [sb, sh, ss]
    pub cos_strides: [u64; 2], // [sb, st]
    pub eps: f32,
    pub flags: u32,
}

impl RopeKvLaunchArgs {
    pub fn grid(&self) -> (u32, u32, u32) {
        (self.hq + self.hkv, self.b * self.t, 1)
    }

    pub fn block(&self) -> (u32, u32, u32) {
        (DEC_NT, 1, 1)
    }

    pub fn smem_bytes(&self) -> u32 {
        (DEC_NT + self.d) * 4
    }
}

pub fn res_rmsnorm_grid(rows: usize) -> (u32, u32, u32) {
    (rows as u32, 1, 1)
}

pub fn res_rmsnorm_smem_bytes() -> u32 {
    DEC_NT * 4
}

pub fn silu_mul_grid(rows: usize, inter_dim: usize) -> (u32, u32, u32) {
    let total = rows.saturating_mul(inter_dim);
    let blocks = total.div_ceil(DEC_NT as usize).clamp(1, 65535);
    (blocks as u32, 1, 1)
}

/// Complete NVRTC CUDA source of the `candle_decoder_ops` module (f32 / f16 / bf16).
pub fn decoder_ops_cuda_source() -> String {
    format!(
        "{}\n{}\nDEC_OPS_INSTANTIATE(f32, float)\nDEC_OPS_INSTANTIATE(f16, ax_f16)\nDEC_OPS_INSTANTIATE(bf16, ax_bf16)\n",
        strip_includes(ACCEL_COMMON_H),
        strip_includes(DEC_OPS_IMPL_H),
    )
}

/// CPU FP32 reference for `ResidualRmsNorm`.
/// Returns `(residual_vec, normed_vec)` each of length `rows * d`.
pub fn cpu_res_rmsnorm_f32(
    x: &[f32],
    delta: Option<&[f32]>,
    w: &[f32],
    rows: usize,
    d: usize,
    eps: f32,
    plus_one: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut r_out = vec![0f32; rows * d];
    let mut y_out = vec![0f32; rows * d];
    if rows == 0 || d == 0 {
        return (r_out, y_out);
    }
    for r in 0..rows {
        let base = r * d;
        let mut sq = 0.0f64;
        for c in 0..d {
            let v = x[base + c] + delta.map_or(0.0, |dl| dl[base + c]);
            r_out[base + c] = v;
            sq += (v as f64) * (v as f64);
        }
        let inv = (1.0 / (sq / d as f64 + eps as f64).sqrt()) as f32;
        for c in 0..d {
            let wf = if plus_one { 1.0 + w[c] } else { w[c] };
            y_out[base + c] = (r_out[base + c] * inv) * wf;
        }
    }
    (r_out, y_out)
}

/// CPU FP32 reference for `SiluMul` supporting arbitrary row strides.
pub fn cpu_silu_mul_f32(
    gate: &[f32],
    gate_row_stride: usize,
    up: &[f32],
    up_row_stride: usize,
    rows: usize,
    inter_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * inter_dim];
    for r in 0..rows {
        let g_row = &gate[r * gate_row_stride..r * gate_row_stride + inter_dim];
        let u_row = &up[r * up_row_stride..r * up_row_stride + inter_dim];
        let o_row = &mut out[r * inter_dim..(r + 1) * inter_dim];
        for c in 0..inter_dim {
            let g = g_row[c];
            o_row[c] = (g / (1.0 + (-g).exp())) * u_row[c];
        }
    }
    out
}

/// CPU FP32 reference for `RopeKvWrite`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_rope_kv_write_f32(
    q_in: &[f32],
    q_out: &mut [f32],
    k_in: &[f32],
    v_in: &[f32],
    q_norm_w: Option<&[f32]>,
    k_norm_w: Option<&[f32]>,
    cos_tab: &[f32],
    sin_tab: &[f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    args: &RopeKvLaunchArgs,
) {
    let b = args.b as usize;
    let t = args.t as usize;
    let hq = args.hq as usize;
    let hkv = args.hkv as usize;
    let d = args.d as usize;
    let rot = args.rotary_dim as usize;
    let half_rot = rot / 2;
    let past = args.past_len as usize;
    let adjacent = (args.flags & 1) == ROPE_MODE_ADJACENT;
    let plus_one = (args.flags & ROPE_NORM_PLUS_ONE) != 0;

    let mut hbuf = vec![0f32; d];

    for bi in 0..b {
        for ti in 0..t {
            let cs_off = bi * args.cos_strides[0] as usize + ti * args.cos_strides[1] as usize;
            let cos_row = &cos_tab[cs_off..cs_off + half_rot];
            let sin_row = &sin_tab[cs_off..cs_off + half_rot];

            // Q heads
            for h in 0..hq {
                let s_off = bi * args.q_strides[0] as usize
                    + h * args.q_strides[1] as usize
                    + ti * args.q_strides[2] as usize;
                let d_off = bi * args.qo_strides[0] as usize
                    + h * args.qo_strides[1] as usize
                    + ti * args.qo_strides[2] as usize;
                hbuf.copy_from_slice(&q_in[s_off..s_off + d]);
                if let Some(nw) = q_norm_w {
                    let sq: f64 = hbuf.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let inv = (1.0 / (sq / d as f64 + args.eps as f64).sqrt()) as f32;
                    for i in 0..d {
                        let wf = if plus_one { 1.0 + nw[i] } else { nw[i] };
                        hbuf[i] = (hbuf[i] * inv) * wf;
                    }
                }
                for i in 0..half_rot {
                    let (c, s) = (cos_row[i], sin_row[i]);
                    if !adjacent {
                        let (x0, x1) = (hbuf[i], hbuf[i + half_rot]);
                        q_out[d_off + i] = x0 * c - x1 * s;
                        q_out[d_off + i + half_rot] = x1 * c + x0 * s;
                    } else {
                        let (x0, x1) = (hbuf[2 * i], hbuf[2 * i + 1]);
                        q_out[d_off + 2 * i] = x0 * c - x1 * s;
                        q_out[d_off + 2 * i + 1] = x1 * c + x0 * s;
                    }
                }
                if rot < d {
                    q_out[d_off + rot..d_off + d].copy_from_slice(&hbuf[rot..d]);
                }
            }

            // K & V heads -> write directly into KV cache at past + ti
            for h in 0..hkv {
                let k_off = bi * args.k_strides[0] as usize
                    + h * args.k_strides[1] as usize
                    + ti * args.k_strides[2] as usize;
                let v_off = bi * args.v_strides[0] as usize
                    + h * args.v_strides[1] as usize
                    + ti * args.v_strides[2] as usize;
                let kv_off = bi * args.kc_strides[0] as usize
                    + h * args.kc_strides[1] as usize
                    + (past + ti) * args.kc_strides[2] as usize;
                hbuf.copy_from_slice(&k_in[k_off..k_off + d]);
                if let Some(nw) = k_norm_w {
                    let sq: f64 = hbuf.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let inv = (1.0 / (sq / d as f64 + args.eps as f64).sqrt()) as f32;
                    for i in 0..d {
                        let wf = if plus_one { 1.0 + nw[i] } else { nw[i] };
                        hbuf[i] = (hbuf[i] * inv) * wf;
                    }
                }
                for i in 0..half_rot {
                    let (c, s) = (cos_row[i], sin_row[i]);
                    if !adjacent {
                        let (x0, x1) = (hbuf[i], hbuf[i + half_rot]);
                        k_cache[kv_off + i] = x0 * c - x1 * s;
                        k_cache[kv_off + i + half_rot] = x1 * c + x0 * s;
                    } else {
                        let (x0, x1) = (hbuf[2 * i], hbuf[2 * i + 1]);
                        k_cache[kv_off + 2 * i] = x0 * c - x1 * s;
                        k_cache[kv_off + 2 * i + 1] = x1 * c + x0 * s;
                    }
                }
                if rot < d {
                    k_cache[kv_off + rot..kv_off + d].copy_from_slice(&hbuf[rot..d]);
                }
                v_cache[kv_off..kv_off + d].copy_from_slice(&v_in[v_off..v_off + d]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_source_has_no_includes_and_matches_hip() {
        let src = decoder_ops_cuda_source();
        assert!(!src.contains("#include \""));
        assert!(!src.contains("#pragma once"));
        let hip = include_str!("../../kernels/decoder_ops.hip");
        for line in [
            "DEC_OPS_INSTANTIATE(f32, float)",
            "DEC_OPS_INSTANTIATE(f16, ax_f16)",
            "DEC_OPS_INSTANTIATE(bf16, ax_bf16)",
        ] {
            assert!(src.contains(line), "cuda source missing {line}");
            assert!(hip.contains(line), "decoder_ops.hip missing {line}");
        }
    }
}
