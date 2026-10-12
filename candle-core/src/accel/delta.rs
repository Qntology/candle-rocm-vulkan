//! Launch planning of the fused Gated DeltaNet kernels (`kernels/delta_rule_impl.h`), shared by
//! the CUDA (NVRTC) and ROCm (hipcc, `kernels/delta_rule.hip`) backends. The user facing
//! operation is [`crate::delta_rule::fused_gated_delta_rule`].

use super::sources::ACCEL_COMMON_H;

pub const DELTA_RULE_IMPL_H: &str = include_str!("../../kernels/delta_rule_impl.h");

/// Lanes of the gated RMS norm kernel (4 rows of 32 lanes).
pub const GNORM_THREADS: usize = 128;

/// Largest key head dimension of the fused scan (`dk` is padded to 64, 128 or 256).
pub const GDN_MAX_HEAD_K_DIM: usize = 256;

/// Tile of one `gdn_scan` instantiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScanTile {
    /// Key head dim the kernel is built for (`dk <= dk_cap`).
    pub dk_cap: usize,
    /// Value columns per workgroup.
    pub tv: usize,
    /// Lanes per workgroup.
    pub nt: usize,
    /// Tokens staged per chunk.
    pub tc: usize,
}

/// Instantiated tiles as (tv, nt, tc), for each `dk_cap` (the ROCm build compiles all of them,
/// CUDA compiles the one used on demand). The first entry is the default.
pub const SCAN_TILES: &[(usize, usize, usize)] = &[(32, 128, 8), (16, 128, 8), (64, 256, 8), (32, 256, 8)];

/// Key head dims the scan is instantiated for.
pub const SCAN_DK_CAPS: &[usize] = &[64, 128, 256];

impl ScanTile {
    pub fn row_groups(&self) -> usize {
        self.nt / self.tv
    }

    /// State rows per lane.
    pub fn rows_per_lane(&self) -> usize {
        self.dk_cap / self.row_groups()
    }

    /// Mirrors `GDN_SCAN_SMEM_FLOATS`.
    pub fn smem_floats(&self) -> usize {
        2 * self.tc * self.dk_cap + self.tc * self.tv + 4 * self.nt + 5 * self.tc
    }

    pub fn smem_bytes(&self) -> usize {
        4 * self.smem_floats()
    }

    /// Workgroups per multiprocessor the CUDA register allocation targets.
    pub fn min_blocks(&self) -> usize {
        // 64 registers per lane at 8 x 128 lanes; larger state slices get fewer workgroups.
        let regs = self.rows_per_lane() + 32;
        (65536 / (regs.next_power_of_two().max(64) * self.nt)).clamp(1, 16)
    }

    pub fn is_valid(&self) -> bool {
        let ng = self.nt / self.tv.max(1);
        self.tv > 0
            && self.nt % self.tv == 0
            && ng > 0
            && self.dk_cap % ng == 0
            && (self.dk_cap / ng) % 4 == 0
            && self.tc > 0
            && self.nt % self.tc == 0
            && self.dk_cap % (self.nt / self.tc) == 0
    }

    /// Grid of a launch over `b` batches, `hv` value heads of `dv` columns.
    pub fn grid(&self, b: usize, hv: usize, dv: usize) -> (u32, u32, u32) {
        (dv.div_ceil(self.tv) as u32, hv as u32, b as u32)
    }

    /// Kernel symbol, e.g. `gdn_scan_bf16_d128_v32_n128_c8`.
    pub fn kernel_name(&self, sfx: &str) -> String {
        format!(
            "gdn_scan_{sfx}_d{}_v{}_n{}_c{}",
            self.dk_cap, self.tv, self.nt, self.tc
        )
    }
}

/// `dk` rounded up to an instantiated key head dim.
pub fn dk_cap(dk: usize) -> Option<usize> {
    SCAN_DK_CAPS.iter().copied().find(|&c| dk <= c)
}

/// Tile used for key head dim `dk`: `CANDLE_GDN_TILE=v{tv}n{nt}c{tc}` selects one of
/// [`SCAN_TILES`] (for tuning), else the first.
pub fn scan_tile(dk: usize) -> Option<ScanTile> {
    let cap = dk_cap(dk)?;
    let (tv, nt, tc) = std::env::var("CANDLE_GDN_TILE")
        .ok()
        .and_then(|s| parse_tile(&s))
        .filter(|t| SCAN_TILES.contains(t))
        .unwrap_or(SCAN_TILES[0]);
    let t = ScanTile {
        dk_cap: cap,
        tv,
        nt,
        tc,
    };
    t.is_valid().then_some(t)
}

fn parse_tile(s: &str) -> Option<(usize, usize, usize)> {
    let s = s.trim().strip_prefix('v')?;
    let (tv, rest) = s.split_once('n')?;
    let (nt, tc) = rest.split_once('c')?;
    Some((tv.parse().ok()?, nt.parse().ok()?, tc.parse().ok()?))
}

/// Kernel arguments of `gdn_scan` that are not buffers (element offsets of the buffers are
/// applied by the backends).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnScanArgs {
    pub b: usize,
    pub t: usize,
    pub hk: usize,
    pub hv: usize,
    pub dk: usize,
    pub dv: usize,
    pub kc: usize,
    /// `GDN_FLAG_L2NORM` | `GDN_FLAG_TILED` | `GDN_FLAG_AB_F32`.
    pub flags: u32,
    pub eps: f32,
    pub scale: f32,
    /// Element offset, batch stride and token stride of `mixed_qkv`, `a` and `b`.
    pub x: [usize; 3],
    pub a: [usize; 3],
    pub bb: [usize; 3],
    /// Element offsets of conv weight, conv bias, conv state, neg_a, dt_bias, recurrent state.
    pub cw_off: usize,
    pub cb_off: usize,
    pub cs_off: usize,
    pub na_off: usize,
    pub dtb_off: usize,
    pub s_off: usize,
}

pub const GDN_FLAG_L2NORM: u32 = 1;
pub const GDN_FLAG_TILED: u32 = 2;
/// `a` / `b` are f32 (else they share the dtype of `mixed_qkv`).
pub const GDN_FLAG_AB_F32: u32 = 4;

impl GdnScanArgs {
    pub fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
}

fn ctype(sfx: &str) -> Option<&'static str> {
    match sfx {
        "f32" => Some("float"),
        "f16" => Some("ax_f16"),
        "bf16" => Some("ax_bf16"),
        _ => None,
    }
}

fn strip(src: &str) -> String {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("#include \"") || t.starts_with("#pragma once"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_scan_name(name: &str) -> Option<(&'static str, ScanTile)> {
    let rest = name.strip_prefix("gdn_scan_")?;
    let mut it = rest.split('_');
    let ty = ctype(it.next()?)?;
    let dk_cap = it.next()?.strip_prefix('d')?.parse().ok()?;
    let tv = it.next()?.strip_prefix('v')?.parse().ok()?;
    let nt = it.next()?.strip_prefix('n')?.parse().ok()?;
    let tc = it.next()?.strip_prefix('c')?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    let t = ScanTile {
        dk_cap,
        tv,
        nt,
        tc,
    };
    t.is_valid().then_some((ty, t))
}

/// The instantiation line of a Gated DeltaNet kernel name (`gdn_scan_f32_d128_v32_n128_c8`,
/// `gdn_gnorm_bf16`).
pub fn gdn_instantiation(name: &str) -> Option<String> {
    if let Some(sfx) = name.strip_prefix("gdn_gnorm_") {
        return Some(format!("GDN_GNORM_KERNEL({name}, {})", ctype(sfx)?));
    }
    let (ty, t) = parse_scan_name(name)?;
    Some(format!(
        "GDN_SCAN_KERNEL({name}, {ty}, {}, {}, {}, {}, {})",
        t.dk_cap,
        t.tv,
        t.nt,
        t.tc,
        t.min_blocks()
    ))
}

/// NVRTC source of one Gated DeltaNet kernel.
pub fn gdn_cuda_source(name: &str) -> Option<String> {
    let inst = gdn_instantiation(name)?;
    Some(format!(
        "{}\n{}\n{inst}\n",
        strip(ACCEL_COMMON_H),
        strip(DELTA_RULE_IMPL_H)
    ))
}

/// Every instantiation of `kernels/delta_rule.hip` (kept in sync by a unit test).
pub fn hip_instantiations() -> Vec<String> {
    let mut v = vec![];
    for sfx in ["f32", "f16", "bf16"] {
        for &cap in SCAN_DK_CAPS {
            for &(tv, nt, tc) in SCAN_TILES {
                let t = ScanTile {
                    dk_cap: cap,
                    tv,
                    nt,
                    tc,
                };
                if t.is_valid() {
                    v.push(gdn_instantiation(&t.kernel_name(sfx)).unwrap());
                }
            }
        }
        v.push(gdn_instantiation(&format!("gdn_gnorm_{sfx}")).unwrap());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_tiles() {
        let t = ScanTile {
            dk_cap: 128,
            tv: 32,
            nt: 128,
            tc: 8,
        };
        assert!(t.is_valid());
        assert_eq!(t.rows_per_lane(), 32);
        assert_eq!(t.kernel_name("bf16"), "gdn_scan_bf16_d128_v32_n128_c8");
        assert_eq!(
            gdn_instantiation("gdn_scan_bf16_d128_v32_n128_c8").unwrap(),
            "GDN_SCAN_KERNEL(gdn_scan_bf16_d128_v32_n128_c8, ax_bf16, 128, 32, 128, 8, 8)"
        );
        assert_eq!(
            gdn_instantiation("gdn_gnorm_f16").unwrap(),
            "GDN_GNORM_KERNEL(gdn_gnorm_f16, ax_f16)"
        );
        assert!(gdn_instantiation("gdn_scan_f32_d100_v32_n128_c8").is_none());
        assert_eq!(dk_cap(96), Some(128));
        assert_eq!(dk_cap(300), None);
        assert_eq!(parse_tile("v64n256c8"), Some((64, 256, 8)));
        for &cap in SCAN_DK_CAPS {
            for &(tv, nt, tc) in SCAN_TILES {
                let t = ScanTile {
                    dk_cap: cap,
                    tv,
                    nt,
                    tc,
                };
                assert!(t.is_valid(), "{t:?}");
            }
        }
        let src = gdn_cuda_source("gdn_scan_f32_d64_v32_n128_c8").unwrap();
        assert!(!src.contains("#include \""));
    }

    /// `kernels/delta_rule.hip` instantiates exactly [`hip_instantiations`].
    #[test]
    fn hip_list_in_sync() {
        let hip = include_str!("../../kernels/delta_rule.hip");
        let lines: Vec<&str> = hip
            .lines()
            .map(|l| l.trim())
            .filter(|l| l.starts_with("GDN_"))
            .collect();
        let want = hip_instantiations();
        assert_eq!(lines, want.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    }
}
