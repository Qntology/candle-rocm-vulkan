// Fused decoder kernels for ROCm / HIP (ResidualRmsNorm, SiluMul, RopeKvWrite).
// Instantiated from the shared portable header `decoder_ops_impl.h`.
#include "decoder_ops_impl.h"

DEC_OPS_INSTANTIATE(f32, float)
DEC_OPS_INSTANTIATE(f16, ax_f16)
DEC_OPS_INSTANTIATE(bf16, ax_bf16)
