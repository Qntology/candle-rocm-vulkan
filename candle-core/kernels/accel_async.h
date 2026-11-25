// Portable asynchronous global->shared tile loader & 2D L2 CTA swizzling.
//
// Compiles cleanly under CUDA NVRTC (--std=c++17, zero std headers), ROCm hipcc,
// and C++20 host emulator (CANDLE_ACCEL_HOST=1).
#pragma once
#include "accel_common.h"

// ---------------------------------------------------------------------------
// 1) Branchless 2D CTA L2 Cache Tile Swizzling (GROUP_M = 8)
// ---------------------------------------------------------------------------
// Groups consecutive M-tiles along the N sweep so concurrent SMs / CUs share
// activation tiles X in L2 cache instead of thrashing across full N columns.
AX_DEV void ax_swizzle_tile_2d(
    ax_u32 linear_bid,
    ax_u32 tiles_m,
    ax_u32 tiles_n,
    ax_u32 &tile_m,
    ax_u32 &tile_n
) {
    constexpr ax_u32 GROUP_M = 8u;
    const ax_u32 width = GROUP_M * tiles_n;
    const ax_u32 group_id = linear_bid / width;
    const ax_u32 first_m = group_id * GROUP_M;
    const ax_u32 rem_m = tiles_m - first_m;
    const ax_u32 group_size_m = rem_m < GROUP_M ? rem_m : GROUP_M;
    const ax_u32 in_group = linear_bid - group_id * width;
    tile_m = first_m + (in_group % group_size_m);
    tile_n = in_group / group_size_m;
}

// ---------------------------------------------------------------------------
// 2) Asynchronous 16-Byte Global -> Shared Copy Primitives
// ---------------------------------------------------------------------------
// On CUDA Ampere+ (sm_80..sm_120): uses headerless inline PTX `cp.async.cg.shared.global`
// with predicated zero-fill (`src_size = 0` when out of bounds) to bypass the register
// file and overlap next-tile HBM fetches with current-tile math.
// On ROCm & host emulator: falls back cleanly to synchronous 16-byte / scalar copy.
#if !defined(CANDLE_ACCEL_HOST) && !defined(__HIP__) && !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define AX_HAS_CP_ASYNC 1

AX_DEV void ax_cp_async_16b(void *smem_dst, const void *gmem_src, bool in_bounds) {
    const ax_u32 smem_addr = static_cast<ax_u32>(__cvta_generic_to_shared(smem_dst));
    const int src_size = in_bounds ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_src), "r"(src_size)
    );
}

AX_DEV void ax_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
AX_DEV void ax_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

AX_DEV void ax_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

#else
#define AX_HAS_CP_ASYNC 0

AX_DEV void ax_cp_async_16b(void *smem_dst, const void *gmem_src, bool in_bounds) {
    ax_u32 *dst = reinterpret_cast<ax_u32 *>(smem_dst);
    if (in_bounds) {
        const ax_u32 *src = reinterpret_cast<const ax_u32 *>(gmem_src);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    } else {
        dst[0] = 0u;
        dst[1] = 0u;
        dst[2] = 0u;
        dst[3] = 0u;
    }
}

AX_DEV void ax_cp_async_commit() {}

template <int N>
AX_DEV void ax_cp_async_wait() {}

AX_DEV void ax_cp_async_wait_all() {}
#endif
