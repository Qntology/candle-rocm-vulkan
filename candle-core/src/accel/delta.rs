//! Launch planning of the fused Gated DeltaNet kernels (`kernels/delta_rule_impl.h`), shared by
//! the CUDA (NVRTC) and ROCm (hipcc, `kernels/delta_rule.hip`) backends. The user facing
//! operation is [`crate::delta_rule::fused_gated_delta_rule`].

use super::sources::ACCEL_COMMON_H;

pub const DELTA_RULE_IMPL_H: &str = include_str!("../../kernels/delta_rule_impl.h");

/// Lanes of the gated RMS norm kernel (4 rows of 32 lanes).
pub const GNORM_THREADS: usize = 128;

/// Largest key head dimension of the fused scan (`dk` is padded to 64, 128 or 256).
pub const GDN_MAX_HEAD_K_DIM: usize = 256;

/// Conv widths up to this keep the taps in registers; wider ones read them from global
/// memory (same results, slower).
pub const GDN_REG_CONV_MAX: usize = 4;

/// Tile of one `gdn_scan` instantiation (see `kernels/delta_rule_impl.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScanTile {
    /// Key head dim the kernel is built for (`dk <= dk_cap`).
    pub dk_cap: usize,
    /// Value columns per warp.
    pub cw: usize,
    /// Value columns per lane.
    pub cpl: usize,
    /// Warps (32 lanes) per workgroup.
    pub nw: usize,
    /// Tokens per chunk.
    pub tc: usize,
}

/// Instantiated tiles as (dk_cap, cw, cpl, nw, tc). The ROCm build compiles all of them, CUDA
/// compiles the one used on demand. Tuned on an RTX 3050 (sm_86) for the Qwen3.5 geometry
/// (Dk = Dv = 128): two value columns per lane halve the shared memory traffic of the scan,
/// whole-head tiles (TV = 128) compute the q / k conv of a key head once per value head.
pub const SCAN_TILES: &[(usize, usize, usize, usize, usize)] = &[
    (64, 16, 2, 8, 8),
    (64, 16, 2, 4, 8),
    (128, 16, 2, 8, 8),
    (128, 16, 2, 4, 8),
    (256, 8, 1, 8, 8),
    (256, 8, 1, 4, 8),
];

/// Key head dims the scan is instantiated for.
pub const SCAN_DK_CAPS: &[usize] = &[64, 128, 256];

impl ScanTile {
    /// Value columns per workgroup.
    pub fn tv(&self) -> usize {
        self.cw * self.nw
    }

    /// Lanes per workgroup.
    pub fn threads(&self) -> usize {
        32 * self.nw
    }

    /// Lanes sharing a row group (the column stride of a lane).
    pub fn lanes_per_row_group(&self) -> usize {
        self.cw / self.cpl.max(1)
    }

    /// State rows per lane.
    pub fn rows_per_lane(&self) -> usize {
        self.dk_cap * self.lanes_per_row_group() / 32
    }

    /// State registers per lane.
    pub fn state_regs(&self) -> usize {
        self.rows_per_lane() * self.cpl
    }

    /// Mirrors `GDN_SCAN_SMEM_FLOATS`.
    pub fn smem_floats(&self) -> usize {
        2 * self.tc * (2 * self.dk_cap + self.tv() + 8) + self.threads()
    }

    pub fn smem_bytes(&self) -> usize {
        4 * self.smem_floats()
    }

    /// Workgroups per multiprocessor the CUDA register allocation targets: the state slice
    /// plus about 48 registers must fit.
    pub fn min_blocks(&self) -> usize {
        let regs = (self.state_regs() + 48).next_power_of_two().max(64);
        (65536 / (regs * self.threads())).clamp(1, 16)
    }

    pub fn is_valid(&self) -> bool {
        let lc = self.lanes_per_row_group();
        self.cpl > 0
            && self.cw % self.cpl == 0
            && lc > 0
            && 32 % lc == 0
            && self.dk_cap % 32 == 0
            && self.rows_per_lane() % 4 == 0
            && self.rows_per_lane() > 0
            && self.nw > 0
            && self.tc > 0
            && self.tc <= self.threads()
    }

    /// Grid of a launch over `b` batches, `hv` value heads of `dv` columns.
    pub fn grid(&self, b: usize, hv: usize, dv: usize) -> (u32, u32, u32) {
        (dv.div_ceil(self.tv()) as u32, hv as u32, b as u32)
    }

    /// Kernel symbol, e.g. `gdn_scan_bf16_d128_c16x2_w8_t8`.
    pub fn kernel_name(&self, sfx: &str) -> String {
        format!(
            "gdn_scan_{sfx}_d{}_c{}x{}_w{}_t{}",
            self.dk_cap, self.cw, self.cpl, self.nw, self.tc
        )
    }

    /// Tuning name, e.g. `c16x2w8t8` (see [`scan_tile`]).
    pub fn short_name(&self) -> String {
        format!("c{}x{}w{}t{}", self.cw, self.cpl, self.nw, self.tc)
    }
}

/// `dk` rounded up to an instantiated key head dim.
pub fn dk_cap(dk: usize) -> Option<usize> {
    SCAN_DK_CAPS.iter().copied().find(|&c| dk <= c)
}

fn tiles_for(cap: usize) -> impl Iterator<Item = ScanTile> {
    SCAN_TILES
        .iter()
        .filter(move |t| t.0 == cap)
        .map(|&(dk_cap, cw, cpl, nw, tc)| ScanTile {
            dk_cap,
            cw,
            cpl,
            nw,
            tc,
        })
}

/// Tile for key head dim `dk` and value head dim `dv`: `CANDLE_GDN_TILE=c{cw}x{cpl}w{nw}t{tc}`
/// selects one of the instantiated tiles of this `dk` (for tuning); otherwise the 8-warp
/// tile, or the 4-warp one when it already covers `dv` (fewer idle lanes for small heads).
pub fn scan_tile(dk: usize, dv: usize) -> Option<ScanTile> {
    let cap = dk_cap(dk)?;
    if let Ok(s) = std::env::var("CANDLE_GDN_TILE") {
        if let Some(t) = tiles_for(cap).find(|t| t.short_name() == s.trim()) {
            return Some(t);
        }
    }
    let mut tiles: Vec<ScanTile> = tiles_for(cap).collect();
    tiles.sort_by_key(|t| std::cmp::Reverse(t.nw));
    let big = *tiles.first()?;
    Some(
        tiles
            .iter()
            .copied()
            .filter(|t| t.tv() >= dv && t.tv() < big.tv())
            .last()
            .unwrap_or(big),
    )
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
    let (cw, cpl) = it.next()?.strip_prefix('c')?.split_once('x')?;
    let nw = it.next()?.strip_prefix('w')?.parse().ok()?;
    let tc = it.next()?.strip_prefix('t')?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    let t = ScanTile {
        dk_cap,
        cw: cw.parse().ok()?,
        cpl: cpl.parse().ok()?,
        nw,
        tc,
    };
    t.is_valid().then_some((ty, t))
}

/// The instantiation line of a Gated DeltaNet kernel name (`gdn_scan_f32_d128_c16x2_w8_t8`,
/// `gdn_gnorm_bf16`).
pub fn gdn_instantiation(name: &str) -> Option<String> {
    if let Some(sfx) = name.strip_prefix("gdn_gnorm_") {
        return Some(format!("GDN_GNORM_KERNEL({name}, {})", ctype(sfx)?));
    }
    let (ty, t) = parse_scan_name(name)?;
    Some(format!(
        "GDN_SCAN_KERNEL({name}, {ty}, {}, {}, {}, {}, {}, {GDN_REG_CONV_MAX}, {})",
        t.dk_cap,
        t.cw,
        t.cpl,
        t.nw,
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
            for t in tiles_for(cap) {
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
            cw: 16,
            cpl: 2,
            nw: 8,
            tc: 8,
        };
        assert!(t.is_valid());
        assert_eq!(t.tv(), 128);
        assert_eq!(t.rows_per_lane(), 32);
        assert_eq!(t.state_regs(), 64);
        assert_eq!(t.min_blocks(), 2);
        assert_eq!(t.kernel_name("bf16"), "gdn_scan_bf16_d128_c16x2_w8_t8");
        assert_eq!(
            gdn_instantiation("gdn_scan_bf16_d128_c16x2_w8_t8").unwrap(),
            "GDN_SCAN_KERNEL(gdn_scan_bf16_d128_c16x2_w8_t8, ax_bf16, 128, 16, 2, 8, 8, 4, 2)"
        );
        assert_eq!(
            gdn_instantiation("gdn_gnorm_f16").unwrap(),
            "GDN_GNORM_KERNEL(gdn_gnorm_f16, ax_f16)"
        );
        assert!(gdn_instantiation("gdn_scan_f32_d100_c16x2_w8_t8").is_none());
        assert!(gdn_instantiation("gdn_scan_f32_d128_c16x3_w8_t8").is_none());
        assert_eq!(dk_cap(96), Some(128));
        assert_eq!(dk_cap(300), None);
        for &(dk_cap, cw, cpl, nw, tc) in SCAN_TILES {
            let t = ScanTile {
                dk_cap,
                cw,
                cpl,
                nw,
                tc,
            };
            assert!(t.is_valid(), "{t:?}");
            // two workgroups of the default tiles fit the 100 KiB of an sm_86 multiprocessor
            assert!(t.smem_bytes() <= 48 * 1024, "{t:?}");
            assert!(SCAN_DK_CAPS.contains(&dk_cap));
        }
        for &cap in SCAN_DK_CAPS {
            assert!(scan_tile(cap, cap).is_some());
        }
        assert_eq!(scan_tile(128, 128).unwrap().nw, 8);
        assert_eq!(scan_tile(128, 64).unwrap().nw, 4);
        assert_eq!(scan_tile(96, 200).unwrap().nw, 8);
        let src = gdn_cuda_source("gdn_scan_f32_d64_c16x2_w8_t8").unwrap();
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
