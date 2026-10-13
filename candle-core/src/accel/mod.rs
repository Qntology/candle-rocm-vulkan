//! Fused attention (FlashAttention v1/v2/v3 tiers) and TurboQuant / microscaling quantization
//! for the CUDA, ROCm and Vulkan backends.
//!
//! This module holds what the backends share: the on/off switches ([`config`]), the per
//! architecture dispatch ([`profile`]), kernel planning and the CPU reference of the attention
//! ([`attention`]), the FP4 / FP8 / microscaling codecs ([`formats`]) and the TurboQuant
//! codebooks, rotations and CPU codec ([`turbo`]). The user facing operations live in
//! `candle_nn::flash_attn` and `candle_nn::turbo_quant`, and the `candle-flash-attn` crate of
//! this workspace is a drop-in replacement of the crates.io one built on them.

pub mod attention;
pub mod config;
pub mod delta;
pub mod formats;
pub mod profile;
pub mod sources;
pub mod turbo;

pub use attention::{AttnGeometry, FaPlan, FlashAttnParams};
pub use config::{
    config, reset_config, set_config, update_config, AccelConfig, BackendSwitches,
    FlashAttnOptions, FlashAttnVersion, RotationKind, SplitKv, TurboQuantOptions,
    FLASH_ATTN_COMPILED, TURBO_QUANT_COMPILED,
};
pub use formats::{row_layout, QuantFormat, RowLayout};
pub use profile::{profile, AccelBackend, AccelProfile};

/// What the fused paths will do on a device with the current configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct AccelReport {
    pub profile: AccelProfile,
    pub flash_attn_compiled: bool,
    pub turbo_quant_compiled: bool,
    /// Fused attention kernels will run (else the unfused path).
    pub flash_attn_active: bool,
    /// FlashAttention tier that will run (forced or automatic).
    pub flash_attn_version: FlashAttnVersion,
    pub turbo_quant_active: bool,
    pub turbo_quant_kernels: bool,
    pub key_format: QuantFormat,
    pub value_format: QuantFormat,
    pub config: AccelConfig,
}

impl std::fmt::Display for AccelReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "device: {}", self.profile)?;
        writeln!(
            f,
            "flash-attn: {} (compiled {}, tier {}, split-kv {:?}, fp8 kv {}, hadamard {})",
            if self.flash_attn_active {
                "fused"
            } else {
                "unfused"
            },
            self.flash_attn_compiled,
            self.flash_attn_version,
            self.config.flash_attn.split_kv,
            self.config.flash_attn.fp8_kv,
            self.config.flash_attn.hadamard,
        )?;
        write!(
            f,
            "turbo-quant: {} (compiled {}, gpu kernels {}, k {}, v {}, rotation {:?}, fused attention {})",
            if self.turbo_quant_active { "on" } else { "off" },
            self.turbo_quant_compiled,
            self.turbo_quant_kernels,
            self.key_format,
            self.value_format,
            self.config.turbo_quant.rotation,
            self.config.turbo_quant.fused_attention,
        )
    }
}

/// Describes the fused attention / TurboQuant setup on `device` (useful for logs).
pub fn describe(device: &crate::Device) -> AccelReport {
    let config = config();
    let profile = profile(device);
    let (key_format, value_format) = profile.turbo_formats(&config.turbo_quant);
    AccelReport {
        flash_attn_compiled: FLASH_ATTN_COMPILED,
        turbo_quant_compiled: TURBO_QUANT_COMPILED,
        flash_attn_active: config.flash_attn_allowed(device)
            && profile.backend != AccelBackend::Other,
        flash_attn_version: config
            .flash_attn
            .version
            .unwrap_or(profile.flash_attn_version),
        turbo_quant_active: config.turbo_quant.enabled,
        turbo_quant_kernels: !device.is_cpu()
            && config.turbo_quant_kernels_allowed(device)
            && profile.backend != AccelBackend::Other,
        key_format,
        value_format,
        profile,
        config,
    }
}
