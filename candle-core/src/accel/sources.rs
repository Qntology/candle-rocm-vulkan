//! Kernel sources for the run-time compiled (NVRTC) CUDA path.
//!
//! The ROCm backend compiles `kernels/flash_attn.hip` / `kernels/turbo_quant.hip` with hipcc at
//! build time. CUDA compiles the same implementation headers with NVRTC the first time a
//! kernel is used: NVRTC cannot read include files, so the headers are concatenated here and a
//! single entry point is instantiated per compilation.

#![allow(dead_code)]

use super::attention::tile_config;
use super::config::FlashAttnVersion;

pub const ACCEL_COMMON_H: &str = include_str!("../../kernels/accel_common.h");
pub const FLASH_ATTN_IMPL_H: &str = include_str!("../../kernels/flash_attn_impl.h");
pub const TURBO_QUANT_IMPL_H: &str = include_str!("../../kernels/turbo_quant_impl.h");

fn strip(src: &str) -> String {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("#include \"") || t.starts_with("#pragma once"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn ctype(sfx: &str) -> Option<&'static str> {
    match sfx {
        "f32" => Some("float"),
        "f16" => Some("ax_f16"),
        "bf16" => Some("ax_bf16"),
        "fp8" => Some("ax_fp8"),
        _ => None,
    }
}

/// The instantiation line of a fused attention kernel name (`fa_row_bf16_fp8`,
/// `fa_rowp_f32_f16`, `fa_rowg4_f32_f16`, `fa_rowpg2_bf16_fp8`, `fa2_d128_f16_f16`,
/// `fa_combine_f32`, ...).
pub fn flash_attn_instantiation(name: &str) -> Option<String> {
    if let Some(o) = name.strip_prefix("fa_combine_") {
        return Some(format!("FA_COMBINE_KERNEL({name}, {})", ctype(o)?));
    }
    for (prefix, mac) in [("fa_rowpg", "FA_ROWPG_KERNEL"), ("fa_rowg", "FA_ROWG_KERNEL")] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let (g, rest) = rest.split_once('_')?;
            let g: usize = g.parse().ok().filter(|g| [2, 4, 8].contains(g))?;
            let (q, kv) = rest.split_once('_')?;
            return Some(format!("{mac}({name}, {}, {}, {g})", ctype(q)?, ctype(kv)?));
        }
    }
    if let Some(rest) = name.strip_prefix("fa_rowp_") {
        let (q, kv) = rest.split_once('_')?;
        return Some(format!(
            "FA_ROWP_KERNEL({name}, {}, {})",
            ctype(q)?,
            ctype(kv)?
        ));
    }
    if let Some(rest) = name.strip_prefix("fa_row_") {
        let (q, kv) = rest.split_once('_')?;
        return Some(format!(
            "FA_ROW_KERNEL({name}, {}, {})",
            ctype(q)?,
            ctype(kv)?
        ));
    }
    let rest = name.strip_prefix("fa")?;
    let (v, rest) = rest.split_once("_d")?;
    let version = match v {
        "2" => FlashAttnVersion::V2,
        "3" => FlashAttnVersion::V3,
        _ => return None,
    };
    let mut parts = rest.split('_');
    let dmax: usize = parts.next()?.parse().ok()?;
    let q = ctype(parts.next()?)?;
    let kv = ctype(parts.next()?)?;
    let (dm, br, bc) = tile_config(version, dmax);
    if dm != dmax {
        return None;
    }
    Some(format!(
        "FA_TILE_KERNEL({name}, {q}, {kv}, {br}, {bc}, {dmax})"
    ))
}

/// NVRTC source of one fused attention kernel.
pub fn flash_attn_cuda_source(name: &str) -> Option<String> {
    let inst = flash_attn_instantiation(name)?;
    Some(format!(
        "{}\n{}\n{inst}\n",
        strip(ACCEL_COMMON_H),
        strip(FLASH_ATTN_IMPL_H)
    ))
}

/// NVRTC source of every TurboQuant kernel.
pub fn turbo_quant_cuda_source() -> String {
    format!(
        "{}\n{}\nTQ_QUANT_KERNELS(f32, float)\nTQ_QUANT_KERNELS(f16, ax_f16)\nTQ_QUANT_KERNELS(bf16, ax_bf16)\nTQ_COMMON_KERNELS\n",
        strip(ACCEL_COMMON_H),
        strip(TURBO_QUANT_IMPL_H)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instantiations() {
        assert_eq!(
            flash_attn_instantiation("fa2_d128_bf16_fp8").unwrap(),
            "FA_TILE_KERNEL(fa2_d128_bf16_fp8, ax_bf16, ax_fp8, 16, 32, 128)"
        );
        assert_eq!(
            flash_attn_instantiation("fa3_d64_f32_f32").unwrap(),
            "FA_TILE_KERNEL(fa3_d64_f32_f32, float, float, 64, 64, 64)"
        );
        assert_eq!(
            flash_attn_instantiation("fa_row_f16_fp8").unwrap(),
            "FA_ROW_KERNEL(fa_row_f16_fp8, ax_f16, ax_fp8)"
        );
        assert_eq!(
            flash_attn_instantiation("fa_rowp_f32_bf16").unwrap(),
            "FA_ROWP_KERNEL(fa_rowp_f32_bf16, float, ax_bf16)"
        );
        assert_eq!(
            flash_attn_instantiation("fa_rowg4_f32_f16").unwrap(),
            "FA_ROWG_KERNEL(fa_rowg4_f32_f16, float, ax_f16, 4)"
        );
        assert_eq!(
            flash_attn_instantiation("fa_rowpg8_bf16_fp8").unwrap(),
            "FA_ROWPG_KERNEL(fa_rowpg8_bf16_fp8, ax_bf16, ax_fp8, 8)"
        );
        assert!(flash_attn_instantiation("fa_rowg3_f32_f32").is_none());
        assert_eq!(
            flash_attn_instantiation("fa2_d256_f32_f16").unwrap(),
            "FA_TILE_KERNEL(fa2_d256_f32_f16, float, ax_f16, 8, 16, 256)"
        );
        assert_eq!(
            flash_attn_instantiation("fa_combine_bf16").unwrap(),
            "FA_COMBINE_KERNEL(fa_combine_bf16, ax_bf16)"
        );
        assert!(flash_attn_instantiation("fa2_d100_f32_f32").is_none());
        let src = flash_attn_cuda_source("fa_row_f32_f32").unwrap();
        assert!(!src.contains("#include \""));
        assert!(turbo_quant_cuda_source().contains("TQ_COMMON_KERNELS"));
    }

    /// `CANDLE_ACCEL_DUMP_CUDA=<dir>` writes the NVRTC sources (for offline compile checks).
    #[test]
    fn dump_cuda_sources() {
        let Ok(dir) = std::env::var("CANDLE_ACCEL_DUMP_CUDA") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut names = vec![];
        for (q, kv) in [
            ("f32", "f32"),
            ("f16", "f16"),
            ("bf16", "bf16"),
            ("f32", "fp8"),
            ("f16", "fp8"),
            ("bf16", "fp8"),
            ("f32", "f16"),
            ("f32", "bf16"),
        ] {
            names.push(format!("fa_row_{q}_{kv}"));
            names.push(format!("fa_rowp_{q}_{kv}"));
            for g in [2, 4, 8] {
                names.push(format!("fa_rowg{g}_{q}_{kv}"));
                names.push(format!("fa_rowpg{g}_{q}_{kv}"));
            }
            for v in [2, 3] {
                for d in [64, 128, 256] {
                    names.push(format!("fa{v}_d{d}_{q}_{kv}"));
                }
            }
        }
        for o in ["f32", "f16", "bf16"] {
            names.push(format!("fa_combine_{o}"));
        }
        for n in names {
            std::fs::write(
                dir.join(format!("{n}.cu")),
                flash_attn_cuda_source(&n).unwrap(),
            )
            .unwrap();
        }
        std::fs::write(dir.join("turbo_quant.cu"), turbo_quant_cuda_source()).unwrap();
    }
}
