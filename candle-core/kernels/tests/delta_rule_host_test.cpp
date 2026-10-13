// Runs the fused Gated DeltaNet kernels (kernels/delta_rule_impl.h) on the host emulator.
//
// Driven by candle-core/tests/delta_rule_tests.rs (`host_emulated_kernels`):
// `delta_rule_host_test <in> <out>` reads named arrays from <in>, runs the operation named by
// the `op` array ("scan" or "gnorm") and writes the result arrays to <out>. File format (the
// one of accel_host_test.cpp): repeated [u32 name_len][name][u8 kind][u64 bytes][bytes].
#define CANDLE_ACCEL_HOST 1
#include "../delta_rule_impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Arr {
    uint8_t kind = 0; // 0 f32, 1 u32, 2 u64, 3 i32, 4 bytes / string
    std::vector<uint8_t> bytes;
    template <typename T> const T *as() const { return reinterpret_cast<const T *>(bytes.data()); }
    template <typename T> size_t len() const { return bytes.size() / sizeof(T); }
};

using Arrays = std::map<std::string, Arr>;

static Arrays read_arrays(const char *path) {
    Arrays out;
    FILE *f = fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open input");
    while (true) {
        uint32_t nl;
        if (fread(&nl, 4, 1, f) != 1) break;
        std::string name(nl, '\0');
        if (fread(name.data(), 1, nl, f) != nl) throw std::runtime_error("truncated name");
        Arr a;
        uint64_t nb;
        if (fread(&a.kind, 1, 1, f) != 1 || fread(&nb, 8, 1, f) != 1) throw std::runtime_error("truncated header");
        a.bytes.resize(nb);
        if (nb && fread(a.bytes.data(), 1, nb, f) != nb) throw std::runtime_error("truncated data");
        out[name] = std::move(a);
    }
    fclose(f);
    return out;
}

static void write_arrays(const char *path, const Arrays &arrs) {
    FILE *f = fopen(path, "wb");
    if (!f) throw std::runtime_error("cannot open output");
    for (auto &kv : arrs) {
        uint32_t nl = (uint32_t)kv.first.size();
        uint64_t nb = kv.second.bytes.size();
        fwrite(&nl, 4, 1, f);
        fwrite(kv.first.data(), 1, nl, f);
        fwrite(&kv.second.kind, 1, 1, f);
        fwrite(&nb, 8, 1, f);
        fwrite(kv.second.bytes.data(), 1, nb, f);
    }
    fclose(f);
}

static const Arr &need(const Arrays &a, const char *name) {
    auto it = a.find(name);
    if (it == a.end()) throw std::runtime_error(std::string("missing array ") + name);
    return it->second;
}

static const Arr *opt(const Arrays &a, const char *name) {
    auto it = a.find(name);
    return it == a.end() ? nullptr : &it->second;
}

template <typename T> static std::vector<T> convert(const float *x, size_t n);
template <> std::vector<float> convert<float>(const float *x, size_t n) { return std::vector<float>(x, x + n); }
template <> std::vector<ax_f16> convert<ax_f16>(const float *x, size_t n) {
    std::vector<ax_f16> v(n);
    for (size_t i = 0; i < n; ++i) v[i].b = ax_f2h(x[i]);
    return v;
}
template <> std::vector<ax_bf16> convert<ax_bf16>(const float *x, size_t n) {
    std::vector<ax_bf16> v(n);
    for (size_t i = 0; i < n; ++i) v[i].b = ax_f2bf(x[i]);
    return v;
}

static float to_f32(float v) { return v; }
static float to_f32(ax_f16 v) { return ax_h2f(v.b); }
static float to_f32(ax_bf16 v) { return ax_bf2f(v.b); }

template <typename T> static Arr f32_arr(const std::vector<T> &v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = to_f32(v[i]);
    Arr a;
    a.kind = 0;
    a.bytes.resize(f.size() * 4);
    if (!f.empty()) memcpy(a.bytes.data(), f.data(), a.bytes.size());
    return a;
}

// ---------------------------------------------------------------------------
// gdn_scan: dims = [B, T, Hk, Hv, dk, dv, K, flags, dtype (0 f32, 1 f16, 2 bf16), tile]
// with tile 0: d64 c16x2 w4, 1: d128 c16x2 w4, 2: d256 c8x1 w4 (TC 8, KMAX 4); a / b in the
// activation dtype; fparams = [eps, scale].
// ---------------------------------------------------------------------------

template <typename TX, int DK, int CW, int CPL, int NW>
static Arrays scan_impl(const Arrays &in) {
    constexpr int TC = 8;
    const ax_u32 *d = need(in, "dims").as<ax_u32>();
    const ax_u32 B = d[0], T = d[1], Hk = d[2], Hv = d[3], dk = d[4], dv = d[5], K = d[6], flags = d[7];
    const float *fp = need(in, "fparams").as<float>();
    const size_t C = 2ull * Hk * dk + (size_t)Hv * dv;
    auto load = [&](const char *n) {
        const Arr &a = need(in, n);
        return convert<TX>(a.as<float>(), a.len<float>());
    };
    std::vector<TX> x = load("x"), a = load("a"), b = load("b");
    const Arr *csa = opt(in, "cs_in");
    std::vector<TX> cs = csa ? convert<TX>(csa->as<float>(), csa->len<float>()) : std::vector<TX>();
    const float *cw = need(in, "cw").as<float>();
    const Arr *cba = opt(in, "cb");
    const float *cb = cba ? cba->as<float>() : nullptr;
    const float *na = need(in, "neg_a").as<float>();
    const float *dtb = need(in, "dt_bias").as<float>();
    const Arr *sa = opt(in, "s_in");
    const float *s_in = sa ? sa->as<float>() : nullptr;
    std::vector<TX> out((size_t)B * T * Hv * dv), cs_out(std::max<size_t>(1, (size_t)B * C * (K - 1)));
    std::vector<float> s_out((size_t)B * Hv * dk * dv);
    constexpr int TV = CW * NW, NT = 32 * NW;
    const size_t smem = GDN_SCAN_SMEM_FLOATS(DK, TV, NT, TC) * 4;
    const TX *csp = csa ? cs.data() : nullptr;
    candle_emu::launch((dv + TV - 1) / TV, Hv, B, NT, smem, [&] {
        gdn_scan_body<TX, DK, CW, CPL, NW, TC, 4>(
            x.data(), csp, cw, cb, a.data(), b.data(), na, dtb, s_in, out.data(), cs_out.data(),
            s_out.data(), T, Hk, Hv, dk, dv, K, flags, fp[0], fp[1], (ax_u64)T * C, (ax_u64)C,
            (ax_u64)T * Hv, (ax_u64)Hv, (ax_u64)T * Hv, (ax_u64)Hv);
    });
    cs_out.resize((size_t)B * C * (K - 1));
    Arrays r;
    r["out"] = f32_arr(out);
    r["cs_out"] = f32_arr(cs_out);
    r["s_out"] = f32_arr(s_out);
    return r;
}

template <typename TX> static Arrays scan_tile(const Arrays &in) {
    switch (need(in, "dims").as<ax_u32>()[9]) {
    case 0: return scan_impl<TX, 64, 16, 2, 4>(in);
    case 1: return scan_impl<TX, 128, 16, 2, 4>(in);
    case 2: return scan_impl<TX, 256, 8, 1, 4>(in);
    default: throw std::runtime_error("unknown tile");
    }
}

static Arrays op_scan(const Arrays &in) {
    switch (need(in, "dims").as<ax_u32>()[8]) {
    case 0: return scan_tile<float>(in);
    case 1: return scan_tile<ax_f16>(in);
    case 2: return scan_tile<ax_bf16>(in);
    default: throw std::runtime_error("unknown dtype");
    }
}

// ---------------------------------------------------------------------------
// gdn_gnorm: dims = [rows, d, dtype], fparams = [eps]
// ---------------------------------------------------------------------------

template <typename TX> static Arrays gnorm_impl(const Arrays &in) {
    const ax_u32 *d = need(in, "dims").as<ax_u32>();
    const ax_u64 rows = d[0];
    const ax_u32 dd = d[1];
    const float eps = need(in, "fparams").as<float>()[0];
    const Arr &xa = need(in, "x"), &za = need(in, "z");
    std::vector<TX> x = convert<TX>(xa.as<float>(), xa.len<float>());
    std::vector<TX> z = convert<TX>(za.as<float>(), za.len<float>());
    const float *w = need(in, "w").as<float>();
    std::vector<TX> y(rows * dd);
    // a small grid so that the grid-stride loop over the rows runs more than once
    const unsigned grid = (unsigned)std::min<ax_u64>(2, (rows + 3) / 4);
    candle_emu::launch(grid, 1, 1, 128, 128 * 4, [&] { gdn_gnorm_body<TX>(x.data(), z.data(), w, y.data(), rows, dd, eps); });
    Arrays r;
    r["y"] = f32_arr(y);
    return r;
}

static Arrays op_gnorm(const Arrays &in) {
    switch (need(in, "dims").as<ax_u32>()[2]) {
    case 0: return gnorm_impl<float>(in);
    case 1: return gnorm_impl<ax_f16>(in);
    case 2: return gnorm_impl<ax_bf16>(in);
    default: throw std::runtime_error("unknown dtype");
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in> <out>\n", argv[0]);
        return 2;
    }
    try {
        Arrays in = read_arrays(argv[1]);
        const Arr &opa = need(in, "op");
        std::string op(reinterpret_cast<const char *>(opa.bytes.data()), opa.bytes.size());
        Arrays out;
        if (op == "scan") out = op_scan(in);
        else if (op == "gnorm") out = op_gnorm(in);
        else throw std::runtime_error("unknown op " + op);
        write_arrays(argv[2], out);
    } catch (const std::exception &e) {
        fprintf(stderr, "delta_rule_host_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
