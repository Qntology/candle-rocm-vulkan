# candle-kernels in candle-rocm-vulkan

`candle-kernels/` is a copy of the crates.io release **candle-kernels 0.11.0**
(https://github.com/huggingface/candle, Copyright Hugging Face, Inc. and the candle contributors,
MIT OR Apache-2.0; see ../NOTICE and ../licenses/). `candle-core` depends on it by path instead of
through crates.io, so the fork can fix kernel sources without waiting for an upstream release.

Nothing else in the workspace uses this crate. Keep it the only copy in a build: it links the
`moe` static library, and two copies of it would define the same symbols twice.

## Changes against the crates.io release

None yet in the commit that added the copy. Later commits list their changes below.
