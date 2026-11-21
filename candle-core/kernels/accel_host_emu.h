// Host emulation of the GPU execution model for the accel kernels (CANDLE_ACCEL_HOST).
// Workgroups run one after the other; every lane of a workgroup is a host thread and
// `__syncthreads` is a std::barrier, so the kernels run unmodified. Used by
// candle-core/tests/accel_kernels_host.rs through kernels/tests/accel_host_test.cpp.
#pragma once
#include <barrier>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace candle_emu {
inline thread_local unsigned tid = 0;
inline thread_local unsigned bid[3] = {0, 0, 0};
inline unsigned nblk[3] = {1, 1, 1};
inline unsigned char *smem = nullptr;
inline std::barrier<> *bar = nullptr;
inline std::vector<std::unique_ptr<std::barrier<>>> *warp_bars = nullptr;
inline uint32_t *shfl_slots = nullptr;

inline void sync() { bar->arrive_and_wait(); }

// Emulates a 32-lane sub-warp `__shfl_xor` / `__shfl_xor_sync(0xffffffffu, v, mask, 32)` so the
// host emulator can compile and bit-compare the GPU `AX_HAS_WARP32_SHFL=1` reduction branches
// against the shared-memory tree branches (`P5-O05` / `P2-O03`).
inline uint32_t shfl_xor_u32(uint32_t v, unsigned mask) {
    const unsigned t = tid;
    const unsigned wid = t >> 5;
    const unsigned lane = t & 31u;
    const unsigned base = wid << 5;
    shfl_slots[t] = v;
    (*warp_bars)[wid]->arrive_and_wait();
    const uint32_t out = shfl_slots[base + (lane ^ mask)];
    (*warp_bars)[wid]->arrive_and_wait();
    return out;
}

inline float shfl_xor_f32(float v, unsigned mask) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t out_bits = shfl_xor_u32(bits, mask);
    float out = 0.0f;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

// Runs `kernel` on a (gx, gy, gz) grid of `nt` lane workgroups with `smem_bytes` of dynamic
// shared memory. Shared memory starts filled with NaN bytes to expose reads of unwritten data.
template <typename F>
void launch(unsigned gx, unsigned gy, unsigned gz, unsigned nt, size_t smem_bytes, F kernel) {
    std::vector<unsigned char> buf(smem_bytes + 64, 0xff);
    const unsigned nwarps = (nt + 31u) / 32u;
    std::vector<std::unique_ptr<std::barrier<>>> wbars;
    wbars.reserve(nwarps);
    for (unsigned w = 0; w < nwarps; ++w) {
        const unsigned lanes_in_w = (w + 1u) * 32u <= nt ? 32u : (nt - w * 32u);
        wbars.push_back(std::make_unique<std::barrier<>>((std::ptrdiff_t)lanes_in_w));
    }
    std::vector<uint32_t> slots(nt, 0u);
    nblk[0] = gx;
    nblk[1] = gy;
    nblk[2] = gz;
    smem = buf.data();
    std::barrier<> b((std::ptrdiff_t)nt);
    bar = &b;
    warp_bars = &wbars;
    shfl_slots = slots.data();
    std::vector<std::thread> lanes;
    lanes.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
        lanes.emplace_back([&, t] {
            tid = t;
            for (unsigned z = 0; z < gz; ++z)
                for (unsigned y = 0; y < gy; ++y)
                    for (unsigned x = 0; x < gx; ++x) {
                        bid[0] = x;
                        bid[1] = y;
                        bid[2] = z;
                        kernel();
                        b.arrive_and_wait();
                    }
        });
    }
    for (auto &l : lanes) l.join();
    bar = nullptr;
    warp_bars = nullptr;
    shfl_slots = nullptr;
}
} // namespace candle_emu
