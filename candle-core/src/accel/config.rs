//! Switches for the fused attention and TurboQuant paths.
//!
//! Every option has three layers, applied in this order:
//!
//! 1. Cargo features `flash-attn` / `turbo-quant` (default on) decide whether the GPU kernels are
//!    built at all. Without them the public APIs keep working through the unfused / CPU paths.
//! 2. Environment variables (`CANDLE_FLASH_ATTN*`, `CANDLE_TURBO_QUANT*`), read the first time the
//!    configuration is used.
//! 3. [`set_config`] / [`update_config`] at run time, and the per call `Option` fields of the
//!    candle-nn option structs, which win over the global values.

use super::formats::QuantFormat;
use std::sync::RwLock;

/// Whether the fused attention GPU kernels were compiled in (`flash-attn` Cargo feature).
pub const FLASH_ATTN_COMPILED: bool = cfg!(feature = "flash-attn");
/// Whether the TurboQuant / microscaling GPU kernels were compiled in (`turbo-quant` Cargo feature).
pub const TURBO_QUANT_COMPILED: bool = cfg!(feature = "turbo-quant");

/// Fused attention algorithm tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FlashAttnVersion {
    /// FlashAttention-1 class: one workgroup per query row streams K/V through a small shared
    /// memory window with an online softmax. Lowest shared memory use: pre-Ampere NVIDIA, GCN /
    /// RDNA2 AMD, Vulkan devices with little shared memory.
    V1,
    /// FlashAttention-2 class: Q tiles per workgroup, K/V tiles in shared memory, delayed
    /// rescaling, causal / window tile skipping, split-KV (flash-decoding) for short queries.
    V2,
    /// FlashAttention-3 class: the V2 pipeline with larger tiles for GPUs with 64 KiB+ of shared
    /// memory per workgroup, preferred FP8 (e4m3) K/V with descale factors and Hadamard
    /// incoherent processing when K is quantized to FP8.
    V3,
}

impl FlashAttnVersion {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
            Self::V3 => "v3",
        }
    }

    /// Parses `1`, `v1`, `fa1`, `flash-attention1`, ... (`auto` gives `Ok(None)`).
    pub fn parse(s: &str) -> std::result::Result<Option<Self>, String> {
        let t = s.trim().to_ascii_lowercase().replace(['-', '_', ' '], "");
        let t = t
            .trim_start_matches("flashattention")
            .trim_start_matches("flashattn")
            .trim_start_matches("fa")
            .trim_start_matches('v');
        match t {
            "" | "auto" | "uto" => Ok(None),
            "1" => Ok(Some(Self::V1)),
            "2" => Ok(Some(Self::V2)),
            "3" | "4" => Ok(Some(Self::V3)),
            _ => Err(format!(
                "unknown flash-attention version {s:?} (auto, 1, 2, 3)"
            )),
        }
    }
}

impl std::fmt::Display for FlashAttnVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Split-KV (flash-decoding) policy for short queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitKv {
    /// Split the keys across workgroups when the query side alone cannot fill the GPU.
    #[default]
    Auto,
    /// Never split.
    Off,
    /// Always use this many splits (clamped to `1..=64`).
    Fixed(usize),
}

impl SplitKv {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        if t.is_empty() || t == "auto" {
            return Ok(Self::Auto);
        }
        if let Some(b) = parse_bool(&t) {
            return Ok(if b { Self::Auto } else { Self::Off });
        }
        match t.parse::<usize>() {
            Ok(0) | Ok(1) => Ok(Self::Off),
            Ok(n) => Ok(Self::Fixed(n.min(64))),
            Err(_) => Err(format!(
                "invalid split-kv value {s:?} (auto, off, or a number)"
            )),
        }
    }
}

/// Orthogonal rotation applied before quantization (TurboQuant) or before FP8 quantization of
/// K (FlashAttention-3 incoherent processing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RotationKind {
    /// Randomized Hadamard when the head dim is a power of two, dense random orthogonal otherwise.
    #[default]
    Auto,
    /// `H * diag(random signs) / sqrt(d)`; requires a power of two head dim.
    Hadamard,
    /// Haar random orthogonal matrix (QR of a Gaussian matrix), as in the TurboQuant reference.
    Dense,
    /// No rotation.
    None,
}

impl RotationKind {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "hadamard" | "had" | "wht" | "fwht" => Ok(Self::Hadamard),
            "dense" | "orthogonal" | "qr" | "random" => Ok(Self::Dense),
            "none" | "off" | "0" | "false" | "no" | "identity" => Ok(Self::None),
            _ => Err(format!(
                "unknown rotation {s:?} (auto, hadamard, dense, none)"
            )),
        }
    }

    /// Resolves `Auto` for head dim `d`.
    pub fn resolve(self, d: usize) -> Self {
        match self {
            Self::Auto if d.is_power_of_two() => Self::Hadamard,
            Self::Auto => Self::Dense,
            Self::Hadamard if !d.is_power_of_two() => Self::Dense,
            k => k,
        }
    }
}

/// Backends on which a fused path may run. A disabled backend uses the unfused (FlashAttention)
/// or CPU reference (TurboQuant) implementation instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendSwitches {
    pub cuda: bool,
    pub rocm: bool,
    pub vulkan: bool,
    pub cpu: bool,
}

impl Default for BackendSwitches {
    fn default() -> Self {
        Self::all()
    }
}

impl BackendSwitches {
    pub const fn all() -> Self {
        Self {
            cuda: true,
            rocm: true,
            vulkan: true,
            cpu: true,
        }
    }

    pub const fn none() -> Self {
        Self {
            cuda: false,
            rocm: false,
            vulkan: false,
            cpu: false,
        }
    }

    /// `all`, `none`, or a comma separated list such as `cuda,rocm`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "" | "all" | "auto" => return Ok(Self::all()),
            "none" | "off" => return Ok(Self::none()),
            _ => {}
        }
        let mut out = Self::none();
        for part in t.split([',', ';', ' ', '+']).filter(|p| !p.is_empty()) {
            match part {
                "cuda" | "nvidia" => out.cuda = true,
                "rocm" | "hip" | "amd" => out.rocm = true,
                "vulkan" | "vk" => out.vulkan = true,
                "cpu" => out.cpu = true,
                _ => return Err(format!("unknown backend {part:?} in {s:?}")),
            }
        }
        Ok(out)
    }

    pub fn allows(&self, device: &crate::Device) -> bool {
        match device {
            crate::Device::Cpu => self.cpu,
            crate::Device::Cuda(_) => self.cuda,
            crate::Device::Rocm(_) => self.rocm,
            crate::Device::Vulkan(_) => self.vulkan,
            _ => false,
        }
    }
}

/// FlashAttention switches.
#[derive(Debug, Clone, PartialEq)]
pub struct FlashAttnOptions {
    /// Master switch. Off: every flash-attn entry point runs the unfused reference
    /// (`matmul` + `softmax`) instead of the fused kernels. Env: `CANDLE_FLASH_ATTN`.
    pub enabled: bool,
    /// Algorithm tier; `None` picks one per GPU architecture. Env: `CANDLE_FLASH_ATTN_VERSION`.
    pub version: Option<FlashAttnVersion>,
    /// Split-KV / flash-decoding. Env: `CANDLE_FLASH_ATTN_SPLIT_KV`.
    pub split_kv: SplitKv,
    /// Let the kernels read FP8 (e4m3) K/V directly. Off: FP8 K/V are cast to the query dtype
    /// first. Env: `CANDLE_FLASH_ATTN_FP8_KV`.
    pub fp8_kv: bool,
    /// Hadamard incoherent processing (rotate Q and K) when `flash_attn_fp8` quantizes K.
    /// Env: `CANDLE_FLASH_ATTN_HADAMARD`.
    pub hadamard: bool,
    /// Backends that may use the fused kernels. Env: `CANDLE_FLASH_ATTN_BACKENDS`.
    pub backends: BackendSwitches,
    /// Log the selected kernel once per configuration. Env: `CANDLE_FLASH_ATTN_DEBUG`.
    pub debug: bool,
}

impl Default for FlashAttnOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            version: None,
            split_kv: SplitKv::Auto,
            fp8_kv: true,
            hadamard: true,
            backends: BackendSwitches::all(),
            debug: false,
        }
    }
}

/// TurboQuant switches.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboQuantOptions {
    /// Master switch. Off: quantized KV caches store K/V unquantized and run plain attention.
    /// Explicit quantize / dequantize calls still work. Env: `CANDLE_TURBO_QUANT`.
    pub enabled: bool,
    /// Key format; `None` picks per GPU architecture. Env: `CANDLE_TURBO_QUANT_K_FORMAT`
    /// (or `CANDLE_TURBO_QUANT_FORMAT` for keys and values).
    pub key_format: Option<QuantFormat>,
    /// Value format; `None` picks per GPU architecture. Env: `CANDLE_TURBO_QUANT_V_FORMAT`.
    pub value_format: Option<QuantFormat>,
    /// Bits per coordinate used by the automatic format choice and by `turbo` formats without an
    /// explicit width (1-8; 4 by default). Env: `CANDLE_TURBO_QUANT_BITS`, `_K_BITS`, `_V_BITS`.
    pub key_bits: u8,
    pub value_bits: u8,
    /// Rotation before quantization. Env: `CANDLE_TURBO_QUANT_ROTATION`.
    pub rotation: RotationKind,
    /// 1-bit QJL residual sketch on keys (TurboQuant "prod", unbiased inner products). Applies
    /// when the key format is a TurboQuant codebook. Off by default: at the same number of bits
    /// the MSE codebook gives a lower attention error (the sketch removes the bias of the inner
    /// products but adds variance, which the softmax amplifies). Env: `CANDLE_TURBO_QUANT_QJL`.
    pub qjl: bool,
    /// Attention directly on the compressed cache. Off: dequantize, then flash-attn.
    /// Env: `CANDLE_TURBO_QUANT_FUSED`.
    pub fused_attention: bool,
    /// Specialized attention kernel for per-vector FP8 keys and values (same results bit for
    /// bit as the generic kernel, a fraction of its instructions per element). Off: the generic
    /// kernel. Env: `CANDLE_TURBO_QUANT_FP8_FAST`.
    pub fp8_fast_path: bool,
    /// Seed of the rotation and QJL matrices. Env: `CANDLE_TURBO_QUANT_SEED`.
    pub seed: u64,
    /// Backends that may use the GPU kernels. Env: `CANDLE_TURBO_QUANT_BACKENDS`.
    pub backends: BackendSwitches,
    /// Log the selected formats once per configuration. Env: `CANDLE_TURBO_QUANT_DEBUG`.
    pub debug: bool,
}

impl Default for TurboQuantOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            key_format: None,
            value_format: None,
            key_bits: 4,
            value_bits: 4,
            rotation: RotationKind::Auto,
            qjl: false,
            fused_attention: true,
            fp8_fast_path: true,
            seed: 42,
            backends: BackendSwitches::all(),
            debug: false,
        }
    }
}

/// All accelerator switches.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AccelConfig {
    pub flash_attn: FlashAttnOptions,
    pub turbo_quant: TurboQuantOptions,
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" | "y" | "enable" | "enabled" => Some(true),
        "0" | "off" | "false" | "no" | "n" | "disable" | "disabled" => Some(false),
        _ => None,
    }
}

fn warn_env(var: &str, msg: &str) {
    eprintln!("candle accel: ignoring {var}: {msg}");
}

fn env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.trim().is_empty())
}

fn env_bool(var: &str, target: &mut bool) {
    if let Some(v) = env(var) {
        match parse_bool(&v) {
            Some(b) => *target = b,
            None => warn_env(var, &format!("expected on/off, got {v:?}")),
        }
    }
}

fn env_parse<T>(var: &str, target: &mut T, f: impl Fn(&str) -> std::result::Result<T, String>) {
    if let Some(v) = env(var) {
        match f(&v) {
            Ok(x) => *target = x,
            Err(e) => warn_env(var, &e),
        }
    }
}

fn parse_bits(s: &str) -> std::result::Result<u8, String> {
    match s.trim().parse::<u8>() {
        Ok(b) if (1..=8).contains(&b) => Ok(b),
        _ => Err(format!("expected a bit width in 1..=8, got {s:?}")),
    }
}

fn parse_format(s: &str) -> std::result::Result<Option<QuantFormat>, String> {
    QuantFormat::parse(s)
}

impl AccelConfig {
    /// Defaults with the environment variables applied.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        c.apply_env();
        c
    }

    /// Applies the `CANDLE_FLASH_ATTN*` / `CANDLE_TURBO_QUANT*` environment variables.
    pub fn apply_env(&mut self) {
        let fa = &mut self.flash_attn;
        env_bool("CANDLE_FLASH_ATTN", &mut fa.enabled);
        env_parse(
            "CANDLE_FLASH_ATTN_VERSION",
            &mut fa.version,
            FlashAttnVersion::parse,
        );
        env_parse(
            "CANDLE_FLASH_ATTN_SPLIT_KV",
            &mut fa.split_kv,
            SplitKv::parse,
        );
        env_bool("CANDLE_FLASH_ATTN_FP8_KV", &mut fa.fp8_kv);
        env_bool("CANDLE_FLASH_ATTN_HADAMARD", &mut fa.hadamard);
        env_parse(
            "CANDLE_FLASH_ATTN_BACKENDS",
            &mut fa.backends,
            BackendSwitches::parse,
        );
        env_bool("CANDLE_FLASH_ATTN_DEBUG", &mut fa.debug);

        let tq = &mut self.turbo_quant;
        env_bool("CANDLE_TURBO_QUANT", &mut tq.enabled);
        if let Some(v) = env("CANDLE_TURBO_QUANT_FORMAT") {
            match parse_format(&v) {
                Ok(f) => {
                    tq.key_format = f;
                    tq.value_format = f;
                }
                Err(e) => warn_env("CANDLE_TURBO_QUANT_FORMAT", &e),
            }
        }
        env_parse(
            "CANDLE_TURBO_QUANT_K_FORMAT",
            &mut tq.key_format,
            parse_format,
        );
        env_parse(
            "CANDLE_TURBO_QUANT_V_FORMAT",
            &mut tq.value_format,
            parse_format,
        );
        if let Some(v) = env("CANDLE_TURBO_QUANT_BITS") {
            match parse_bits(&v) {
                Ok(b) => {
                    tq.key_bits = b;
                    tq.value_bits = b;
                }
                Err(e) => warn_env("CANDLE_TURBO_QUANT_BITS", &e),
            }
        }
        env_parse("CANDLE_TURBO_QUANT_K_BITS", &mut tq.key_bits, parse_bits);
        env_parse("CANDLE_TURBO_QUANT_V_BITS", &mut tq.value_bits, parse_bits);
        env_parse(
            "CANDLE_TURBO_QUANT_ROTATION",
            &mut tq.rotation,
            RotationKind::parse,
        );
        env_bool("CANDLE_TURBO_QUANT_QJL", &mut tq.qjl);
        env_bool("CANDLE_TURBO_QUANT_FUSED", &mut tq.fused_attention);
        env_bool("CANDLE_TURBO_QUANT_FP8_FAST", &mut tq.fp8_fast_path);
        env_parse("CANDLE_TURBO_QUANT_SEED", &mut tq.seed, |s| {
            s.trim().parse::<u64>().map_err(|e| e.to_string())
        });
        env_parse(
            "CANDLE_TURBO_QUANT_BACKENDS",
            &mut tq.backends,
            BackendSwitches::parse,
        );
        env_bool("CANDLE_TURBO_QUANT_DEBUG", &mut tq.debug);
    }

    /// Whether the fused attention kernels may run on `device` (compiled in, enabled, backend
    /// allowed). The CPU "fused" path is the multi-threaded reference implementation.
    pub fn flash_attn_allowed(&self, device: &crate::Device) -> bool {
        self.flash_attn.enabled
            && self.flash_attn.backends.allows(device)
            && (device.is_cpu() || FLASH_ATTN_COMPILED)
    }

    /// Whether the TurboQuant GPU kernels may run on `device`.
    pub fn turbo_quant_kernels_allowed(&self, device: &crate::Device) -> bool {
        self.turbo_quant.backends.allows(device) && (device.is_cpu() || TURBO_QUANT_COMPILED)
    }
}

static CONFIG: RwLock<Option<AccelConfig>> = RwLock::new(None);

/// The current configuration (environment on first use, then programmatic changes).
pub fn config() -> AccelConfig {
    if let Ok(g) = CONFIG.read() {
        if let Some(c) = g.as_ref() {
            return c.clone();
        }
    }
    let mut g = match CONFIG.write() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.get_or_insert_with(AccelConfig::from_env).clone()
}

/// Replaces the configuration for the whole process.
pub fn set_config(c: AccelConfig) {
    let mut g = match CONFIG.write() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    *g = Some(c);
}

/// Changes part of the configuration, e.g.
/// `update_config(|c| c.flash_attn.version = Some(FlashAttnVersion::V2))`.
pub fn update_config(f: impl FnOnce(&mut AccelConfig)) {
    let mut c = config();
    f(&mut c);
    set_config(c);
}

/// Drops programmatic changes and reads the environment again on next use.
pub fn reset_config() {
    let mut g = match CONFIG.write() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    *g = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_versions() {
        assert_eq!(FlashAttnVersion::parse("auto").unwrap(), None);
        assert_eq!(
            FlashAttnVersion::parse("2").unwrap(),
            Some(FlashAttnVersion::V2)
        );
        assert_eq!(
            FlashAttnVersion::parse("FA3").unwrap(),
            Some(FlashAttnVersion::V3)
        );
        assert_eq!(
            FlashAttnVersion::parse("flash-attention1").unwrap(),
            Some(FlashAttnVersion::V1)
        );
        assert!(FlashAttnVersion::parse("7").is_err());
    }

    #[test]
    fn parse_switches() {
        assert_eq!(SplitKv::parse("off").unwrap(), SplitKv::Off);
        assert_eq!(SplitKv::parse("8").unwrap(), SplitKv::Fixed(8));
        let b = BackendSwitches::parse("rocm,vulkan").unwrap();
        assert!(b.rocm && b.vulkan && !b.cuda && !b.cpu);
        assert_eq!(RotationKind::parse("none").unwrap(), RotationKind::None);
        assert_eq!(RotationKind::Auto.resolve(128), RotationKind::Hadamard);
        assert_eq!(RotationKind::Auto.resolve(96), RotationKind::Dense);
    }
}
