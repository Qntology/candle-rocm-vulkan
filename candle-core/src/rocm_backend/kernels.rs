//! HIP code objects compiled by `build.rs` and embedded in the binary.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Module {
    Unary,
    Binary,
    Cast,
    Fill,
    Indexing,
    Reduce,
    Ternary,
    Nn,
    Quantized,
    Gemm,
    /// Fused attention (`flash_attn.hip`, `flash-attn` feature).
    FlashAttn,
    /// TurboQuant / microscaling quantization (`turbo_quant.hip`, `turbo-quant` feature).
    TurboQuant,
    /// Fused Gated DeltaNet (`delta_rule.hip`).
    DeltaRule,
}

pub const UNARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary.co"));
pub const BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary.co"));
pub const CAST: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cast.co"));
pub const FILL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fill.co"));
pub const INDEXING: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/indexing.co"));
pub const REDUCE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce.co"));
pub const TERNARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ternary.co"));
pub const NN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nn.co"));
pub const QUANTIZED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantized.co"));
pub const GEMM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.co"));
pub const FLASH_ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn.co"));
pub const TURBO_QUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/turbo_quant.co"));
pub const DELTA_RULE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/delta_rule.co"));

/// GPU architectures the embedded code objects were built for.
pub const COMPILED_ARCHS: &str = env!("CANDLE_ROCM_ARCHS");

/// ROCm track of the toolchain the kernels were built with: `legacy` (ROCm / HIP SDK 6.x - 7.2)
/// or `core` (ROCm Core SDK 10.x, HIP 7.10+).
pub const BUILD_TRACK: &str = env!("CANDLE_ROCM_TRACK");

/// HIP version of the toolchain the kernels were built with (`unknown` when not found).
pub const BUILD_HIP_VERSION: &str = env!("CANDLE_ROCM_BUILD_HIP_VERSION");

/// Whether the embedded kernels contain code for `arch` (processor name, e.g. `gfx1100`).
pub fn compiled_for(arch: &str) -> bool {
    let arch = arch.split(':').next().unwrap_or(arch);
    COMPILED_ARCHS.split(',').any(|a| a.trim() == arch)
}

impl Module {
    pub const ALL: [Module; 13] = [
        Module::Unary,
        Module::Binary,
        Module::Cast,
        Module::Fill,
        Module::Indexing,
        Module::Reduce,
        Module::Ternary,
        Module::Nn,
        Module::Quantized,
        Module::Gemm,
        Module::FlashAttn,
        Module::TurboQuant,
        Module::DeltaRule,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Module::Unary => "unary",
            Module::Binary => "binary",
            Module::Cast => "cast",
            Module::Fill => "fill",
            Module::Indexing => "indexing",
            Module::Reduce => "reduce",
            Module::Ternary => "ternary",
            Module::Nn => "nn",
            Module::Quantized => "quantized",
            Module::Gemm => "gemm",
            Module::FlashAttn => "flash_attn",
            Module::TurboQuant => "turbo_quant",
            Module::DeltaRule => "delta_rule",
        }
    }

    pub fn image(self) -> &'static [u8] {
        match self {
            Module::Unary => UNARY,
            Module::Binary => BINARY,
            Module::Cast => CAST,
            Module::Fill => FILL,
            Module::Indexing => INDEXING,
            Module::Reduce => REDUCE,
            Module::Ternary => TERNARY,
            Module::Nn => NN,
            Module::Quantized => QUANTIZED,
            Module::Gemm => GEMM,
            Module::FlashAttn => FLASH_ATTN,
            Module::TurboQuant => TURBO_QUANT,
            Module::DeltaRule => DELTA_RULE,
        }
    }
}
