// Fused lm_head Tile Argmax + Mean-Centered CS Guard kernels for ROCm / HIP.
// Instantiated from the shared portable header `lm_head_argmax_impl.h`.
#include "lm_head_argmax_impl.h"

LM_HEAD_DENSE_INSTANTIATE(f32, float)
LM_HEAD_DENSE_INSTANTIATE(f16, ax_f16)
LM_HEAD_DENSE_INSTANTIATE(bf16, ax_bf16)

LM_HEAD_LOGITS_INSTANTIATE(f32, float, f32, float)
LM_HEAD_LOGITS_INSTANTIATE(f16, ax_f16, f16, ax_f16)
LM_HEAD_LOGITS_INSTANTIATE(bf16, ax_bf16, bf16, ax_bf16)
LM_HEAD_LOGITS_INSTANTIATE(f32, float, f16, ax_f16)
LM_HEAD_LOGITS_INSTANTIATE(f32, float, bf16, ax_bf16)
