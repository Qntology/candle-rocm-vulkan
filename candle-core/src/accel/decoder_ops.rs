//! Shared planning, NVRTC source generation, and CPU F32 reference for the fused decoder
//! kernels (`kernels/decoder_ops_impl.h`: `ResidualRmsNorm`, `SiluMul`, `RopeKvWrite`).

use super::sources::{strip_includes, ACCEL_COMMON_H};

pub const DEC_OPS_IMPL_H: &str = include_str!("../../kernels/decoder_ops_impl.h");

pub const DEC_NT: u32 = 256;
pub const DEC_MR_NT: u32 = 128;
pub const DEC_ROWS_PER_WG: u32 = 4;
pub const RES_RMS_PLUS_ONE_WEIGHT: u32 = 1;
pub const RES_RMS_WRITE_RESIDUAL: u32 = 2;
pub const ROPE_MODE_NEOX: u32 = 0;
pub const ROPE_MODE_ADJACENT: u32 = 1;
pub const ROPE_NORM_PLUS_ONE: u32 = 2;
pub const ROPE_CODY_WAITE_ONTHEFLY: u32 = 4;

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

/// Barrier-free multi-row RMSNorm (4 rows/WG x 32 lanes = 128 threads) is used for prefill batches.
#[inline]
pub fn res_rmsnorm_use_mr(rows: usize) -> bool {
    rows >= 32
}

pub fn res_rmsnorm_mr_grid(rows: usize) -> (u32, u32, u32) {
    (rows.div_ceil(DEC_ROWS_PER_WG as usize).max(1) as u32, 1, 1)
}

pub fn res_rmsnorm_mr_smem_bytes() -> u32 {
    // 0 bytes needed on GPU with warp shuffles; 128 floats only needed for host emu fallback.
    0
}

pub fn silu_mul_grid(rows: usize, inter_dim: usize) -> (u32, u32, u32) {
    let total = rows.saturating_mul(inter_dim);
    let blocks = total.div_ceil(DEC_NT as usize).clamp(1, 65535);
    (blocks as u32, 1, 1)
}

/// Exact FP32 FMA TwoProd Cody-Waite range-reduced `(cos, sin)` for long-context RoPE (64K - 1M+).
/// Matches `dec_cody_waite_sincos` in `kernels/decoder_ops_impl.h` (<= 3e-7 error vs FP64 at 1M pos).
#[inline]
pub fn cody_waite_sincos_f32(pos: u32, inv_freq: f32) -> (f32, f32) {
    const INV_2PI_HI: f32 = 0.15915493667125702f32;
    const INV_2PI_LO: f32 = 6.4206382432985265e-9f32;
    const TWO_PI_F: f32 = 6.2831853071795864769f32;

    let w_hi = inv_freq * INV_2PI_HI;
    let w_lo = inv_freq.mul_add(INV_2PI_HI, -w_hi) + inv_freq * INV_2PI_LO;

    let p_lo = (pos & 0xfff) as f32;
    let p_hi = (pos & !0xfff) as f32;

    let c0 = p_hi * w_hi;
    let e0 = p_hi.mul_add(w_hi, -c0);
    let c1 = p_lo * w_hi;
    let e1 = p_lo.mul_add(w_hi, -c1);

    let mut cyc = (c0 - c0.floor()) + (c1 - c1.floor()) + (e0 + e1 + p_hi.mul_add(w_lo, p_lo * w_lo));
    cyc -= cyc.floor();
    if cyc > 0.5 {
        cyc -= 1.0;
    }
    let phi = cyc * TWO_PI_F;
    (phi.cos(), phi.sin())
}

/// Applies orthonormal 32-point Walsh-Hadamard transform (`H_32 / sqrt(32)`) in-place
/// to each 32-element block of `data` (length must be a multiple of 32).
pub fn fwht32_blocks_inplace(data: &mut [f32]) {
    const INV_SQRT32: f32 = 0.17677669529663689f32;
    debug_assert_eq!(data.len() % 32, 0);
    for blk in data.chunks_exact_mut(32) {
        let mut h = 1usize;
        while h < 32 {
            let step = h << 1;
            let mut base = 0usize;
            while base < 32 {
                for j in 0..h {
                    let a = blk[base + j];
                    let b = blk[base + j + h];
                    blk[base + j] = a + b;
                    blk[base + j + h] = a - b;
                }
                base += step;
            }
            h = step;
        }
        for v in blk.iter_mut() {
            *v *= INV_SQRT32;
        }
    }
}

/// Online Hadamard-32 smoothing + per-32-block OCP E4M3FN dynamic quantization on CPU.
/// Returns `(fp8_codes, scales)` matching `res_rmsnorm_hadamard_fp8_*` / `silu_mul_hadamard_fp8_*`.
pub fn hadamard32_quant_fp8_cpu(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    use super::formats::f32_to_e4m3;
    assert_eq!(values.len() % 32, 0, "hadamard32_quant_fp8_cpu: length must be a multiple of 32");
    let mut rot = values.to_vec();
    fwht32_blocks_inplace(&mut rot);
    let n_blks = rot.len() / 32;
    let mut codes = Vec::with_capacity(rot.len());
    let mut scales = Vec::with_capacity(n_blks);
    for blk in rot.chunks_exact(32) {
        let mut amax = 0.0f32;
        for &v in blk {
            amax = amax.max(v.abs());
        }
        let scale = if amax > 0.0 { amax * (1.0 / 448.0) } else { 1.0 };
        let inv_scale = if amax > 0.0 { 448.0 / amax } else { 0.0 };
        scales.push(scale);
        for &v in blk {
            codes.push(f32_to_e4m3(v * inv_scale));
        }
    }
    (codes, scales)
}

/// Launch descriptor for Graph-safe indirect `rope_kv_write_indirect_*` (with optional
/// YaRN `mscale` and Cody-Waite on-the-fly range reduction).
#[derive(Debug, Clone, Copy)]
pub struct RopeKvIndirectLaunchArgs {
    pub b: u32,
    pub t: u32,
    pub hq: u32,
    pub hkv: u32,
    pub d: u32,
    pub rotary_dim: u32,
    pub pos_stride: u32,
    pub q_off: usize,
    pub k_off: usize,
    pub v_off: usize,
    pub qnw_off: usize,
    pub knw_off: usize,
    pub cos_off: usize,
    pub sin_off: usize,
    pub kc_off: usize,
    pub vc_off: usize,
    pub kv_pos_off: usize,
    pub rope_pos_off: usize,
    pub q_strides: [u64; 3],
    pub qo_strides: [u64; 3],
    pub k_strides: [u64; 3],
    pub v_strides: [u64; 3],
    pub kc_strides: [u64; 3],
    pub cos_st: u64,
    pub eps: f32,
    pub mscale: f32,
    pub flags: u32,
}

impl RopeKvIndirectLaunchArgs {
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

    #[test]
    fn cody_waite_long_context_matches_f64_oracle() {
        for &pos in &[0u32, 1024, 65_536, 131_072, 262_144, 1_000_000] {
            for &inv_f in &[1.0f32, 0.37f32, 0.0123456f32, 1e-4f32] {
                let (c, s) = cody_waite_sincos_f32(pos, inv_f);
                let theta = (pos as f64) * (inv_f as f64);
                let ec = (c as f64 - theta.cos()).abs();
                let es = (s as f64 - theta.sin()).abs();
                assert!(
                    ec <= 1e-6 && es <= 1e-6,
                    "pos={pos} inv_f={inv_f}: err_cos={ec:.3e} err_sin={es:.3e}"
                );
            }
        }
    }

    #[test]
    fn fwht32_orthonormal_and_outlier_smoothing() {
        use super::super::formats::e4m3_to_f32;
        // 1) Exact involution: ||x H32 H32^T - x||_inf <= 2e-7
        let orig: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.137).sin() * 3.5).collect();
        let mut buf = orig.clone();
        fwht32_blocks_inplace(&mut buf);
        fwht32_blocks_inplace(&mut buf);
        let max_err = orig
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err <= 2e-6, "fwht32 roundtrip err={max_err}");

        // 2) 100-sigma channel outlier vector: Hadamard-FP8 preserves cosine > 0.9995
        let mut x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.27).cos()).collect();
        x[3] = 100.0;
        x[67] = -95.0;
        let (codes, scales) = hadamard32_quant_fp8_cpu(&x);
        let mut deq: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| e4m3_to_f32(c) * scales[i / 32])
            .collect();
        fwht32_blocks_inplace(&mut deq);
        let dot: f64 = x.iter().zip(deq.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
        let nx: f64 = x.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        let nd: f64 = deq.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (nx * nd);
        assert!(cos > 0.9995, "outlier Hadamard-FP8 cosine={cos}");
    }
}
