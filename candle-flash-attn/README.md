# candle-flash-attn (candle-rocm-vulkan)

Drop-in replacement of the crates.io [`candle-flash-attn`](https://crates.io/crates/candle-flash-attn)
0.11 for the CUDA, ROCm, Vulkan and CPU backends of
[candle-rocm-vulkan](https://github.com/Qntology/candle-rocm-vulkan): the same functions
(`flash_attn`, `flash_attn_windowed`, `flash_attn_alibi*`, `flash_attn_varlen*`,
`flash_attn_varlen_paged_windowed`) and `FlashAttn` op, running the fused kernels of
`candle_nn::flash_attn` (FlashAttention v1 / v2 / v3 tiers chosen per GPU) with an unfused
fallback.

```toml
[patch.crates-io]
candle-core = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
candle-nn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
candle-flash-attn = { git = "https://github.com/Qntology/candle-rocm-vulkan.git" }
```

The backend follows the `cuda` / `rocm` / `vulkan` features of `candle-nn` (or of this crate).
Switches (`CANDLE_FLASH_ATTN`, `CANDLE_FLASH_ATTN_VERSION`, ...) are listed in the repository
README.
