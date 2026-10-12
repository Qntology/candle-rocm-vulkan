// Host emulation of the GPU execution model for the accel kernels (CANDLE_ACCEL_HOST).
// Workgroups run one after the other; every lane of a workgroup is a host thread and
// `__syncthreads` is a std::barrier, so the kernels run unmodified. Used by
// candle-core/tests/accel_kernels_host.rs through kernels/tests/accel_host_test.cpp.
#pragma once
#include <barrier>
#include <thread>
#include <vector>

namespace candle_emu {
inline thread_local unsigned tid = 0;
inline thread_local unsigned bid[3] = {0, 0, 0};
inline unsigned nblk[3] = {1, 1, 1};
inline unsigned char *smem = nullptr;
inline std::barrier<> *bar = nullptr;

inline void sync() { bar->arrive_and_wait(); }

// Runs `kernel` on a (gx, gy, gz) grid of `nt` lane workgroups with `smem_bytes` of dynamic
// shared memory. Shared memory starts filled with NaN bytes to expose reads of unwritten data.
template <typename F>
void launch(unsigned gx, unsigned gy, unsigned gz, unsigned nt, size_t smem_bytes, F kernel) {
    std::vector<unsigned char> buf(smem_bytes + 64, 0xff);
    nblk[0] = gx;
    nblk[1] = gy;
    nblk[2] = gz;
    smem = buf.data();
    std::barrier<> b((std::ptrdiff_t)nt);
    bar = &b;
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
}
} // namespace candle_emu
