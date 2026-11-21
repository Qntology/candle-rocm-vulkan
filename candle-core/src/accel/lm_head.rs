//! Shared tile planner, NVRTC source generation, and CPU FP32 oracle for the fused `lm_head`
//! tile argmax + Mean-Centered Cauchy-Schwarz (CS) Guard kernels (`kernels/lm_head_argmax_impl.h`).

use super::sources::{strip_includes, ACCEL_COMMON_H};

pub const LM_HEAD_IMPL_H: &str = include_str!("../../kernels/lm_head_argmax_impl.h");

pub const LM_NT: u32 = 256;
pub const LM_DECODE_BN: u32 = 32;
pub const LM_PREFILL_BM: u32 = 32;
pub const LM_PREFILL_BN: u32 = 64;
pub const LM_PREFILL_BK: u32 = 32;

/// Stage-1 launch plan selected from `(M, V_valid)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage1Kind {
    Decode1,
    Decode8,
    Prefill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage1Plan {
    pub kind: Stage1Kind,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub n_tiles: u32,
    pub smem_bytes: u32,
}

impl Stage1Plan {
    pub fn kernel_name(&self, dtype_sfx: &str) -> String {
        match self.kind {
            Stage1Kind::Decode1 => format!("lm_stage1_decode1_{dtype_sfx}"),
            Stage1Kind::Decode8 => format!("lm_stage1_decode8_{dtype_sfx}"),
            Stage1Kind::Prefill => format!("lm_stage1_prefill_{dtype_sfx}"),
        }
    }
}

/// Choose the Stage-1 tile kernel and launch configuration for `(m, v_valid)`.
pub fn plan_stage1(m: usize, v_valid: usize) -> Stage1Plan {
    if m <= 1 {
        let n_tiles = v_valid.div_ceil(LM_DECODE_BN as usize).max(1) as u32;
        let smem = (LM_NT * 4 + LM_DECODE_BN) * 4;
        Stage1Plan {
            kind: Stage1Kind::Decode1,
            grid: (n_tiles, 1, 1),
            block: (LM_NT, 1, 1),
            n_tiles,
            smem_bytes: smem,
        }
    } else if m <= 8 {
        let n_tiles = v_valid.div_ceil(LM_DECODE_BN as usize).max(1) as u32;
        let smem = (LM_NT * 4 + 8 * LM_DECODE_BN) * 4;
        Stage1Plan {
            kind: Stage1Kind::Decode8,
            grid: (n_tiles, 1, 1),
            block: (LM_NT, 1, 1),
            n_tiles,
            smem_bytes: smem,
        }
    } else {
        let n_tiles_n = v_valid.div_ceil(LM_PREFILL_BN as usize).max(1) as u32;
        let n_tiles_m = m.div_ceil(LM_PREFILL_BM as usize).max(1) as u32;
        let smem = (LM_PREFILL_BM * LM_PREFILL_BK + LM_PREFILL_BN * LM_PREFILL_BK) * 4;
        Stage1Plan {
            kind: Stage1Kind::Prefill,
            grid: (n_tiles_n, n_tiles_m, 1),
            block: (LM_NT, 1, 1),
            n_tiles: n_tiles_n,
            smem_bytes: smem,
        }
    }
}

/// Shared memory bytes for Stage 2 (`lm_stage2_reduce_cs_*`) and 1-Stage (`lm_logits_argmax_cs_*`).
pub fn stage2_smem_bytes() -> u32 {
    4 * LM_NT * 4 // [s_val][s_idx][s_xx][s_xm] = 4 KiB
}

/// Complete NVRTC CUDA source of the `candle_lm_head_argmax` module.
pub fn lm_head_argmax_cuda_source() -> String {
    format!(
        "{}\n{}\n\
         LM_HEAD_DENSE_INSTANTIATE(f32, float)\n\
         LM_HEAD_DENSE_INSTANTIATE(f16, ax_f16)\n\
         LM_HEAD_DENSE_INSTANTIATE(bf16, ax_bf16)\n\
         LM_HEAD_LOGITS_INSTANTIATE(f32, float, f32, float)\n\
         LM_HEAD_LOGITS_INSTANTIATE(f16, ax_f16, f16, ax_f16)\n\
         LM_HEAD_LOGITS_INSTANTIATE(bf16, ax_bf16, bf16, ax_bf16)\n\
         LM_HEAD_LOGITS_INSTANTIATE(f32, float, f16, ax_f16)\n\
         LM_HEAD_LOGITS_INSTANTIATE(f32, float, bf16, ax_bf16)\n",
        strip_includes(ACCEL_COMMON_H),
        strip_includes(LM_HEAD_IMPL_H),
    )
}

#[inline]
pub fn better_argmax(v_cand: f32, i_cand: u32, v_curr: f32, i_curr: u32) -> bool {
    (v_cand > v_curr) || (v_cand == v_curr && i_cand < i_curr)
}

/// CPU FP32 oracle for 2-Stage dense `fused_linear_argmax_cs`.
/// Returns `(out_ids, out_vals, out_rescue)`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_fused_linear_argmax_cs_f32(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    core_ids: Option<&[u32]>,
    mu_rest: Option<&[f32]>,
    w_rest_bound: f32,
    b_rest_max: f32,
    check_cs: bool,
    m: usize,
    v_valid: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>, Vec<u8>) {
    let mut out_ids = vec![0u32; m];
    let mut out_vals = vec![f32::NEG_INFINITY; m];
    let mut out_rescue = vec![0u8; m];

    for row in 0..m {
        let xr = &x[row * k..(row + 1) * k];
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = u32::MAX;
        for n in 0..v_valid {
            let wn = &w[n * k..(n + 1) * k];
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = xr[kk].mul_add(wn[kk], acc);
            }
            if let Some(b) = bias {
                acc += b[n];
            }
            if better_argmax(acc, n as u32, best_v, best_i) {
                best_v = acc;
                best_i = n as u32;
            }
        }
        out_ids[row] = match core_ids {
            Some(c) if (best_i as usize) < c.len() => c[best_i as usize],
            _ => best_i,
        };
        out_vals[row] = best_v;
        if check_cs {
            let mut xx = 0.0f32;
            let mut xm = 0.0f32;
            for kk in 0..k {
                let hv = xr[kk];
                xx = hv.mul_add(hv, xx);
                if let Some(mu) = mu_rest {
                    xm = hv.mul_add(mu[kk], xm);
                }
            }
            let upper = xx.sqrt().mul_add(w_rest_bound, xm) + b_rest_max;
            out_rescue[row] = u8::from(best_v < upper);
        }
    }
    (out_ids, out_vals, out_rescue)
}

/// CPU FP32 oracle for 1-Stage `logits_argmax_cs`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_logits_argmax_cs_f32(
    sub_logits: &[f32],
    logits_row_stride: usize,
    core_ids: Option<&[u32]>,
    hidden: Option<&[f32]>,
    mu_rest: Option<&[f32]>,
    w_rest_bound: f32,
    b_rest_max: f32,
    check_cs: bool,
    m: usize,
    v_valid: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>, Vec<u8>) {
    let mut out_ids = vec![0u32; m];
    let mut out_vals = vec![f32::NEG_INFINITY; m];
    let mut out_rescue = vec![0u8; m];

    for row in 0..m {
        let lr = &sub_logits[row * logits_row_stride..row * logits_row_stride + v_valid];
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = u32::MAX;
        for (j, &v) in lr.iter().enumerate() {
            if better_argmax(v, j as u32, best_v, best_i) {
                best_v = v;
                best_i = j as u32;
            }
        }
        out_ids[row] = match core_ids {
            Some(c) if (best_i as usize) < c.len() => c[best_i as usize],
            _ => best_i,
        };
        out_vals[row] = best_v;
        if check_cs {
            if let Some(h) = hidden {
                let hr = &h[row * k..(row + 1) * k];
                let mut xx = 0.0f32;
                let mut xm = 0.0f32;
                for kk in 0..k {
                    let hv = hr[kk];
                    xx = hv.mul_add(hv, xx);
                    if let Some(mu) = mu_rest {
                        xm = hv.mul_add(mu[kk], xm);
                    }
                }
                let upper = xx.sqrt().mul_add(w_rest_bound, xm) + b_rest_max;
                out_rescue[row] = u8::from(best_v < upper);
            }
        }
    }
    (out_ids, out_vals, out_rescue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_source_has_no_includes_and_matches_hip() {
        let src = lm_head_argmax_cuda_source();
        assert!(!src.contains("#include \""));
        assert!(!src.contains("#pragma once"));
        let hip = include_str!("../../kernels/lm_head_argmax.hip");
        for line in [
            "LM_HEAD_DENSE_INSTANTIATE(f32, float)",
            "LM_HEAD_DENSE_INSTANTIATE(f16, ax_f16)",
            "LM_HEAD_DENSE_INSTANTIATE(bf16, ax_bf16)",
            "LM_HEAD_LOGITS_INSTANTIATE(f32, float, f32, float)",
            "LM_HEAD_LOGITS_INSTANTIATE(f16, ax_f16, f16, ax_f16)",
            "LM_HEAD_LOGITS_INSTANTIATE(bf16, ax_bf16, bf16, ax_bf16)",
            "LM_HEAD_LOGITS_INSTANTIATE(f32, float, f16, ax_f16)",
            "LM_HEAD_LOGITS_INSTANTIATE(f32, float, bf16, ax_bf16)",
        ] {
            assert!(src.contains(line), "cuda source missing {line}");
            assert!(hip.contains(line), "lm_head_argmax.hip missing {line}");
        }
    }

    #[test]
    fn stage1_plan_switches_at_decode_and_prefill() {
        assert_eq!(plan_stage1(1, 151_936).kind, Stage1Kind::Decode1);
        assert_eq!(plan_stage1(8, 151_936).kind, Stage1Kind::Decode8);
        assert_eq!(plan_stage1(9, 151_936).kind, Stage1Kind::Prefill);
        assert_eq!(plan_stage1(64, 15_003).n_tiles, 15_003usize.div_ceil(64) as u32);
    }
}
