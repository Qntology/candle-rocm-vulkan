// Host-emulated tests for kernels/decoder_ops_impl.h (CANDLE_ACCEL_HOST=1).
// Every workgroup lane runs as a real host thread synced by std::barrier.
#define CANDLE_ACCEL_HOST 1
#include "../decoder_ops_impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Arr {
    uint8_t kind = 0; // 0 f32, 1 u32, 2 u64, 3 i32, 4 bytes/str
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

template <typename TX>
static void run_res_rmsnorm(const Arrays &in, Arrays &out, uint32_t rows, uint32_t D, float eps, uint32_t flags, bool inplace) {
    const Arr &xa = need(in, "x");
    const Arr *da = opt(in, "delta");
    const Arr &wa = need(in, "w");
    auto x = convert<TX>(xa.as<float>(), xa.len<float>());
    std::vector<TX> delta;
    if (da) delta = convert<TX>(da->as<float>(), da->len<float>());
    std::vector<TX> r_out(rows * D);
    std::vector<TX> y_out(rows * D);

    TX *r_ptr = inplace ? x.data() : ((flags & RES_RMS_WRITE_RESIDUAL) ? r_out.data() : nullptr);
    const TX *d_ptr = da ? delta.data() : nullptr;

    candle_emu::launch(rows, 1, 1, DEC_NT, DEC_NT * sizeof(float), [&]() {
        res_rmsnorm_impl<TX>(x.data(), d_ptr, wa.as<float>(), r_ptr, y_out.data(), rows, D, eps, flags);
    });

    out["y"] = f32_arr(y_out);
    if (inplace) {
        out["r"] = f32_arr(x);
    } else if (flags & RES_RMS_WRITE_RESIDUAL) {
        out["r"] = f32_arr(r_out);
    }
}

template <typename TX>
static void run_silu_mul(const Arrays &in, Arrays &out, uint32_t rows, uint32_t inter, uint64_t gs, uint64_t us, uint64_t g_off, uint64_t u_off) {
    const Arr &ga = need(in, "gate");
    const Arr &ua = need(in, "up");
    auto g = convert<TX>(ga.as<float>(), ga.len<float>());
    auto u = convert<TX>(ua.as<float>(), ua.len<float>());
    std::vector<TX> y(rows * inter);
    uint32_t gx = (uint32_t)((((uint64_t)rows * inter) + DEC_NT - 1) / DEC_NT);
    if (gx == 0) gx = 1;
    candle_emu::launch(gx, 1, 1, DEC_NT, 0, [&]() {
        silu_mul_impl<TX>(g.data() + g_off, gs, u.data() + u_off, us, y.data(), rows, inter);
    });
    out["out"] = f32_arr(y);
}

template <typename TX>
static void run_rope_kv_write(const Arrays &in, Arrays &out) {
    const Arr &dm = need(in, "dims");
    const uint32_t *d = dm.as<uint32_t>();
    const uint32_t B = d[0], T = d[1], Hq = d[2], Hkv = d[3], D = d[4], rot = d[5], past = d[6], flags = d[7];
    const Arr &st = need(in, "strides");
    const uint64_t *s = st.as<uint64_t>();
    const float eps = need(in, "eps").as<float>()[0];

    const Arr &qa = need(in, "q_in");
    const Arr &ka = need(in, "k_in");
    const Arr &va = need(in, "v_in");
    const Arr *qnw = opt(in, "q_norm_w");
    const Arr *knw = opt(in, "k_norm_w");
    const Arr &cosa = need(in, "cos");
    const Arr &sina = need(in, "sin");
    const Arr &kca = need(in, "k_cache");
    const Arr &vca = need(in, "v_cache");

    auto q_in = convert<TX>(qa.as<float>(), qa.len<float>());
    auto k_in = convert<TX>(ka.as<float>(), ka.len<float>());
    auto v_in = convert<TX>(va.as<float>(), va.len<float>());
    auto k_cache = convert<TX>(kca.as<float>(), kca.len<float>());
    auto v_cache = convert<TX>(vca.as<float>(), vca.len<float>());
    std::vector<TX> q_out(B * T * Hq * D);

    candle_emu::launch(Hq + Hkv, B * T, 1, DEC_NT, (DEC_NT + D) * sizeof(float), [&]() {
        rope_kv_write_impl<TX>(
            q_in.data(), q_out.data(), k_in.data(), v_in.data(),
            qnw ? qnw->as<float>() : nullptr,
            knw ? knw->as<float>() : nullptr,
            cosa.as<float>(), sina.as<float>(),
            k_cache.data(), v_cache.data(),
            B, T, Hq, Hkv, D, rot, past,
            s[0], s[1], s[2],
            s[3], s[4], s[5],
            s[6], s[7], s[8],
            s[9], s[10], s[11],
            s[12], s[13], s[14],
            s[15], s[16],
            eps, flags
        );
    });

    out["q_out"] = f32_arr(q_out);
    out["k_cache"] = f32_arr(k_cache);
    out["v_cache"] = f32_arr(v_cache);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in.bin> <out.bin>\n", argv[0]);
        return 2;
    }
    Arrays in = read_arrays(argv[1]);
    Arrays out;
    const Arr &op_arr = need(in, "op");
    std::string op(reinterpret_cast<const char *>(op_arr.bytes.data()), op_arr.bytes.size());
    uint32_t dt = need(in, "dtype").as<uint32_t>()[0]; // 0=f32, 1=f16, 2=bf16

    if (op == "res_rmsnorm") {
        const uint32_t *dm = need(in, "dims").as<uint32_t>();
        uint32_t rows = dm[0], D = dm[1], flags = dm[2], inplace = dm[3];
        float eps = need(in, "eps").as<float>()[0];
        if (dt == 0) run_res_rmsnorm<float>(in, out, rows, D, eps, flags, inplace != 0);
        else if (dt == 1) run_res_rmsnorm<ax_f16>(in, out, rows, D, eps, flags, inplace != 0);
        else run_res_rmsnorm<ax_bf16>(in, out, rows, D, eps, flags, inplace != 0);
    } else if (op == "silu_mul") {
        const uint32_t *dm = need(in, "dims").as<uint32_t>();
        const uint64_t *st = need(in, "strides").as<uint64_t>();
        if (dt == 0) run_silu_mul<float>(in, out, dm[0], dm[1], st[0], st[1], st[2], st[3]);
        else if (dt == 1) run_silu_mul<ax_f16>(in, out, dm[0], dm[1], st[0], st[1], st[2], st[3]);
        else run_silu_mul<ax_bf16>(in, out, dm[0], dm[1], st[0], st[1], st[2], st[3]);
    } else if (op == "rope_kv_write") {
        if (dt == 0) run_rope_kv_write<float>(in, out);
        else if (dt == 1) run_rope_kv_write<ax_f16>(in, out);
        else run_rope_kv_write<ax_bf16>(in, out);
    } else {
        throw std::runtime_error("unknown op: " + op);
    }

    write_arrays(argv[2], out);
    return 0;
}
