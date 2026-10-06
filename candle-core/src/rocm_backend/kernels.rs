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

/// GPU architectures the embedded code objects were built for.
pub const COMPILED_ARCHS: &str = env!("CANDLE_ROCM_ARCHS");

impl Module {
    pub const ALL: [Module; 9] = [
        Module::Unary,
        Module::Binary,
        Module::Cast,
        Module::Fill,
        Module::Indexing,
        Module::Reduce,
        Module::Ternary,
        Module::Nn,
        Module::Quantized,
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
        }
    }
}
