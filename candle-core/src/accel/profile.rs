//! Per architecture choices: FlashAttention tier and low precision formats.
//!
//! | GPU | FA tier | native low precision | FP4 class | FP8 class |
//! | --- | --- | --- | --- | --- |
//! | NVIDIA sm < 8.0 (Volta, Turing) | V1 | - | NVFP4 (emulated) | FP8 (emulated) |
//! | NVIDIA sm 8.0 - 8.7 (Ampere) | V2 | - | NVFP4 (emulated) | FP8 (emulated) |
//! | NVIDIA sm 8.9 (Ada) | V2 | FP8 | NVFP4 (emulated) | FP8 |
//! | NVIDIA sm 9.x (Hopper) | V3 | FP8 | NVFP4 (emulated) | FP8 |
//! | NVIDIA sm 10.x / 11.x (Blackwell DC, Thor) | V3 | FP8, NVFP4, MXFP4/6/8 | NVFP4 | MXFP8 |
//! | NVIDIA sm 12.x (Blackwell RTX) | V2 | FP8, NVFP4, MXFP4/6/8 | NVFP4 | MXFP8 |
//! | AMD gfx950 (MI350) | V3 | FP8, MXFP4/6/8 | MXFP4 | MXFP8 |
//! | AMD gfx940-942 (MI300) | V3 | FP8 (FNUZ) | MXFP4 (emulated) | FP8 |
//! | AMD gfx1200/1201 (RDNA4) | V3 | FP8 | MXFP4 (emulated) | FP8 |
//! | AMD gfx90a (MI200), gfx11xx (RDNA3/3.5) | V2 | - | MXFP4 (emulated) | FP8 (emulated) |
//! | AMD gfx908, gfx10xx (RDNA1/2), older | V1 | - | MXFP4 (emulated) | FP8 (emulated) |
//! | Vulkan | V2 if 32 KiB+ shared memory, else V1 | - | MXFP4 (emulated) | FP8 (emulated) |
//!
//! FA tiers follow Dao-AILab: FA2 needs Ampere+ (and on ROCm CDNA2+/RDNA3+, the targets of its
//! composable-kernel backend), FA3 is the Hopper generation; the FP8-capable AMD and Blackwell
//! data-center parts get the V3 tier (larger tiles + FP8 K/V). "emulated" means the format is
//! decoded in software by the kernels; on GPUs with native support the same data can also be
//! fed to the vendor block-scaled GEMM libraries.
//!
//! The automatic TurboQuant format picks the native FP4/FP8 format when the GPU has one for the
//! requested width, and TurboQuant codebooks otherwise (lower error at the same size when
//! everything is decoded in software anyway).

use super::config::{FlashAttnVersion, TurboQuantOptions};
use super::formats::QuantFormat;

/// Accelerator family of a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelBackend {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Other,
}

/// What a device supports and which variants the automatic choices use.
#[derive(Debug, Clone, PartialEq)]
pub struct AccelProfile {
    pub backend: AccelBackend,
    /// `sm_90`, `gfx942`, `vulkan:<device name>`, `cpu`.
    pub arch: String,
    /// FlashAttention tier picked when no version is forced.
    pub flash_attn_version: FlashAttnVersion,
    /// Native FP8 (e4m3) math.
    pub native_fp8: bool,
    /// Native FP4 (e2m1) math.
    pub native_fp4: bool,
    /// Native OCP microscaling (MX) block-scaled math.
    pub native_mx: bool,
    /// Format of the 4-bit class.
    pub fp4_format: QuantFormat,
    /// Format of the 8-bit class.
    pub fp8_format: QuantFormat,
    /// Shared memory available to one workgroup, in bytes.
    pub max_shared_mem: usize,
    /// Compute units / SMs (used by the split-KV heuristic).
    pub compute_units: usize,
}

impl AccelProfile {
    /// Profile of an NVIDIA GPU with compute capability `major.minor`.
    pub fn cuda(major: u32, minor: u32, max_shared_mem: usize, compute_units: usize) -> Self {
        let cc = major * 10 + minor;
        let version = if cc < 80 {
            FlashAttnVersion::V1
        } else if (90..120).contains(&cc) {
            FlashAttnVersion::V3
        } else {
            FlashAttnVersion::V2
        };
        let native_fp8 = cc >= 89;
        let blackwell = cc >= 100;
        Self {
            backend: AccelBackend::Cuda,
            arch: format!("sm_{cc}"),
            flash_attn_version: version,
            native_fp8,
            native_fp4: blackwell,
            native_mx: blackwell,
            fp4_format: QuantFormat::Nvfp4,
            fp8_format: if blackwell {
                QuantFormat::Mxfp8
            } else {
                QuantFormat::Fp8
            },
            max_shared_mem,
            compute_units,
        }
    }

    /// Profile of an AMD GPU (`gfx` processor name, target features ignored).
    pub fn rocm(arch: &str, compute_units: usize) -> Self {
        let name = arch
            .split(':')
            .next()
            .unwrap_or(arch)
            .trim()
            .to_ascii_lowercase();
        let (version, fp8, mx) = match name.as_str() {
            "gfx950" => (FlashAttnVersion::V3, true, true),
            "gfx940" | "gfx941" | "gfx942" => (FlashAttnVersion::V3, true, false),
            n if n.starts_with("gfx12") => (FlashAttnVersion::V3, true, false),
            "gfx90a" => (FlashAttnVersion::V2, false, false),
            n if n.starts_with("gfx11") => (FlashAttnVersion::V2, false, false),
            _ => (FlashAttnVersion::V1, false, false),
        };
        Self {
            backend: AccelBackend::Rocm,
            arch: name,
            flash_attn_version: version,
            native_fp8: fp8,
            native_fp4: mx,
            native_mx: mx,
            fp4_format: QuantFormat::Mxfp4,
            fp8_format: if mx {
                QuantFormat::Mxfp8
            } else {
                QuantFormat::Fp8
            },
            // 64 KiB of LDS per workgroup on every GCN / CDNA / RDNA generation
            max_shared_mem: 64 * 1024,
            compute_units,
        }
    }

    /// Profile of a Vulkan device.
    pub fn vulkan(name: &str, max_shared_mem: usize, max_invocations: usize) -> Self {
        let version = if max_shared_mem >= 32 * 1024 && max_invocations >= 128 {
            FlashAttnVersion::V2
        } else {
            FlashAttnVersion::V1
        };
        Self {
            backend: AccelBackend::Vulkan,
            arch: format!("vulkan:{name}"),
            flash_attn_version: version,
            native_fp8: false,
            native_fp4: false,
            native_mx: false,
            fp4_format: QuantFormat::Mxfp4,
            fp8_format: QuantFormat::Fp8,
            max_shared_mem,
            compute_units: 16,
        }
    }

    pub fn cpu() -> Self {
        Self {
            backend: AccelBackend::Cpu,
            arch: "cpu".into(),
            flash_attn_version: FlashAttnVersion::V2,
            native_fp8: false,
            native_fp4: false,
            native_mx: false,
            fp4_format: QuantFormat::Mxfp4,
            fp8_format: QuantFormat::Fp8,
            max_shared_mem: usize::MAX,
            compute_units: num_cpus::get(),
        }
    }

    pub fn other(name: &str) -> Self {
        Self {
            backend: AccelBackend::Other,
            arch: name.into(),
            ..Self::cpu()
        }
    }

    /// The automatic format for `bits` per element: native FP4 / FP8 when the GPU has it, else
    /// TurboQuant codebooks up to 5 bits and FP8 above.
    pub fn auto_format(&self, bits: u8, keys: bool, qjl: bool) -> QuantFormat {
        if bits >= 6 {
            return self.fp8_format;
        }
        if bits == 4 && self.native_fp4 {
            return self.fp4_format;
        }
        if keys && qjl && bits >= 2 {
            QuantFormat::TurboProd { bits: bits.min(5) }
        } else {
            QuantFormat::TurboMse { bits: bits.min(4) }
        }
    }

    /// Resolved key and value formats for a set of TurboQuant options.
    pub fn turbo_formats(&self, o: &TurboQuantOptions) -> (QuantFormat, QuantFormat) {
        let resolve = |f: Option<QuantFormat>, bits: u8, keys: bool| -> QuantFormat {
            let f = f.unwrap_or_else(|| self.auto_format(bits, keys, o.qjl));
            let f = f.with_default_bits(bits);
            match f {
                // QJL off: the prod format degrades to the MSE codebook of the same width
                QuantFormat::TurboProd { bits } if !o.qjl || !keys => {
                    QuantFormat::TurboMse { bits: bits.min(4) }
                }
                f => f,
            }
        };
        (
            resolve(o.key_format, o.key_bits, true),
            resolve(o.value_format, o.value_bits, false),
        )
    }
}

impl std::fmt::Display for AccelProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (flash-attn {}, fp8 {}, fp4 {}, mx {}, 4-bit {}, 8-bit {})",
            self.arch,
            self.flash_attn_version,
            self.native_fp8,
            self.native_fp4,
            self.native_mx,
            self.fp4_format,
            self.fp8_format
        )
    }
}

/// Profile of a device.
pub fn profile(device: &crate::Device) -> AccelProfile {
    match device {
        crate::Device::Cpu => AccelProfile::cpu(),
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(d) => d.accel_profile(),
        #[cfg(feature = "rocm")]
        crate::Device::Rocm(d) => d.accel_profile(),
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(d) => d.accel_profile(),
        #[allow(unreachable_patterns)]
        _ => AccelProfile::other("unsupported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_tables() {
        assert_eq!(
            AccelProfile::cuda(7, 5, 64 << 10, 40).flash_attn_version,
            FlashAttnVersion::V1
        );
        assert_eq!(
            AccelProfile::cuda(8, 6, 99 << 10, 80).flash_attn_version,
            FlashAttnVersion::V2
        );
        let h100 = AccelProfile::cuda(9, 0, 227 << 10, 132);
        assert_eq!(h100.flash_attn_version, FlashAttnVersion::V3);
        assert!(h100.native_fp8 && !h100.native_fp4);
        let b200 = AccelProfile::cuda(10, 0, 227 << 10, 148);
        assert_eq!(b200.auto_format(4, false, true), QuantFormat::Nvfp4);
        assert_eq!(b200.auto_format(8, false, true), QuantFormat::Mxfp8);
        assert_eq!(
            AccelProfile::cuda(12, 0, 99 << 10, 84).flash_attn_version,
            FlashAttnVersion::V2
        );

        let mi355 = AccelProfile::rocm("gfx950", 256);
        assert_eq!(mi355.auto_format(4, true, true), QuantFormat::Mxfp4);
        assert_eq!(mi355.flash_attn_version, FlashAttnVersion::V3);
        let mi300 = AccelProfile::rocm("gfx942:sramecc+:xnack-", 304);
        assert_eq!(mi300.arch, "gfx942");
        assert_eq!(mi300.auto_format(8, true, true), QuantFormat::Fp8);
        assert_eq!(
            mi300.auto_format(4, true, true),
            QuantFormat::TurboProd { bits: 4 }
        );
        assert_eq!(
            AccelProfile::rocm("gfx1100", 48).flash_attn_version,
            FlashAttnVersion::V2
        );
        assert_eq!(
            AccelProfile::rocm("gfx1030", 40).flash_attn_version,
            FlashAttnVersion::V1
        );
        assert_eq!(
            AccelProfile::rocm("gfx1201", 64).flash_attn_version,
            FlashAttnVersion::V3
        );

        assert_eq!(
            AccelProfile::vulkan("x", 16 << 10, 1024).flash_attn_version,
            FlashAttnVersion::V1
        );
    }

    #[test]
    fn turbo_resolution() {
        let p = AccelProfile::rocm("gfx1100", 48);
        let o = TurboQuantOptions::default();
        assert_eq!(
            p.turbo_formats(&o),
            (
                QuantFormat::TurboMse { bits: 4 },
                QuantFormat::TurboMse { bits: 4 }
            )
        );
        let o = TurboQuantOptions {
            qjl: true,
            ..Default::default()
        };
        assert_eq!(
            p.turbo_formats(&o),
            (
                QuantFormat::TurboProd { bits: 4 },
                QuantFormat::TurboMse { bits: 4 }
            )
        );
        let o = TurboQuantOptions {
            qjl: false,
            key_bits: 3,
            ..Default::default()
        };
        assert_eq!(p.turbo_formats(&o).0, QuantFormat::TurboMse { bits: 3 });
        let o = TurboQuantOptions {
            key_format: Some(QuantFormat::Nvfp4),
            value_format: Some(QuantFormat::TurboProd { bits: 3 }),
            ..Default::default()
        };
        // values never use the QJL stage
        assert_eq!(
            p.turbo_formats(&o),
            (QuantFormat::Nvfp4, QuantFormat::TurboMse { bits: 3 })
        );
    }
}
