// TurboQuant / microscaling quantization kernels for the ROCm backend
// (implementation: turbo_quant_impl.h).
#include "turbo_quant_impl.h"

TQ_QUANT_KERNELS(f32, float)
TQ_QUANT_KERNELS(f16, ax_f16)
TQ_QUANT_KERNELS(bf16, ax_bf16)
TQ_COMMON_KERNELS
