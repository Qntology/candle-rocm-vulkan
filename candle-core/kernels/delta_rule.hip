// Fused Gated DeltaNet kernels for the ROCm backend (implementation: delta_rule_impl.h).
// The instantiation list must match candle_core::accel::delta::hip_instantiations (unit test
// accel::delta::tests::hip_list_in_sync).
#include "delta_rule_impl.h"

GDN_SCAN_KERNEL(gdn_scan_f32_d64_c16x2_w8_t8, float, 64, 16, 2, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_f32_d64_c16x2_w4_t8, float, 64, 16, 2, 4, 8, 4, 4)
GDN_SCAN_KERNEL(gdn_scan_f32_d128_c16x2_w8_t8, float, 128, 16, 2, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_f32_d128_c16x2_w4_t8, float, 128, 16, 2, 4, 8, 4, 4)
GDN_SCAN_KERNEL(gdn_scan_f32_d256_c8x1_w8_t8, float, 256, 8, 1, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_f32_d256_c8x1_w4_t8, float, 256, 8, 1, 4, 8, 4, 4)
GDN_GNORM_KERNEL(gdn_gnorm_f32, float)

GDN_SCAN_KERNEL(gdn_scan_f16_d64_c16x2_w8_t8, ax_f16, 64, 16, 2, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_f16_d64_c16x2_w4_t8, ax_f16, 64, 16, 2, 4, 8, 4, 4)
GDN_SCAN_KERNEL(gdn_scan_f16_d128_c16x2_w8_t8, ax_f16, 128, 16, 2, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_f16_d128_c16x2_w4_t8, ax_f16, 128, 16, 2, 4, 8, 4, 4)
GDN_SCAN_KERNEL(gdn_scan_f16_d256_c8x1_w8_t8, ax_f16, 256, 8, 1, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_f16_d256_c8x1_w4_t8, ax_f16, 256, 8, 1, 4, 8, 4, 4)
GDN_GNORM_KERNEL(gdn_gnorm_f16, ax_f16)

GDN_SCAN_KERNEL(gdn_scan_bf16_d64_c16x2_w8_t8, ax_bf16, 64, 16, 2, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_bf16_d64_c16x2_w4_t8, ax_bf16, 64, 16, 2, 4, 8, 4, 4)
GDN_SCAN_KERNEL(gdn_scan_bf16_d128_c16x2_w8_t8, ax_bf16, 128, 16, 2, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_bf16_d128_c16x2_w4_t8, ax_bf16, 128, 16, 2, 4, 8, 4, 4)
GDN_SCAN_KERNEL(gdn_scan_bf16_d256_c8x1_w8_t8, ax_bf16, 256, 8, 1, 8, 8, 4, 2)
GDN_SCAN_KERNEL(gdn_scan_bf16_d256_c8x1_w4_t8, ax_bf16, 256, 8, 1, 4, 8, 4, 4)
GDN_GNORM_KERNEL(gdn_gnorm_bf16, ax_bf16)
