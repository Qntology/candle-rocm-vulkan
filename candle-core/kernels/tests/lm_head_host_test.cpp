// Host-emulated tests for kernels/lm_head_argmax_impl.h (CANDLE_ACCEL_HOST=1).
// Every workgroup lane runs as a real host thread synced by std::barrier.
#define CANDLE_ACCEL_HOST 1
#include "../lm_head_argmax_impl.h"

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

template <typename T> static Arr raw_arr(uint8_t kind, const std::vector<T> &v) {
    Arr a;
    a.kind = kind;
    a.bytes.resize(v.size() * sizeof(T));
    if (!v.empty()) memcpy(a.bytes.data(), v.data(), a.bytes.size());
    return a;
}

template <typename TX>
static void run_dense(const Arrays &in, Arrays &out) {
    const uint32_t *dm = need(in, "dims").as<uint32_t>();
    const uint32_t M = dm[0], V = dm[1], K = dm[2], check_cs = dm[3];
    const float *fp = need(in, "fparams").as<float>();
    const float w_bound = fp[0], b_max = fp[1];

    const Arr &xa = need(in, "x");
    const Arr &wa = need(in, "w");
    const Arr *ba = opt(in, "bias");
    const Arr *ca = opt(in, "core_ids");
    const Arr *mua = opt(in, "mu_rest");

    auto x = convert<TX>(xa.as<float>(), xa.len<float>());
    auto w = convert<TX>(wa.as<float>(), wa.len<float>());

    uint32_t n_tiles = 0;
    std::vector<float> part_val;
    std::vector<ax_u32> part_idx;

    if (M <= 1) {
        n_tiles = (V + LM_DECODE_BN - 1) / LM_DECODE_BN;
        part_val.resize((size_t)M * n_tiles);
        part_idx.resize((size_t)M * n_tiles);
        candle_emu::launch(n_tiles, 1, 1, LM_NT, (LM_NT * 4 + LM_DECODE_BN) * sizeof(float), [&]() {
            lm_stage1_decode_impl<TX, 1u>(
                x.data(), w.data(), ba ? ba->as<float>() : nullptr,
                part_val.data(), part_idx.data(), M, V, K, n_tiles);
        });
    } else if (M <= 8) {
        n_tiles = (V + LM_DECODE_BN - 1) / LM_DECODE_BN;
        part_val.resize((size_t)M * n_tiles);
        part_idx.resize((size_t)M * n_tiles);
        candle_emu::launch(n_tiles, 1, 1, LM_NT, (LM_NT * 4 + 8 * LM_DECODE_BN) * sizeof(float), [&]() {
            lm_stage1_decode_impl<TX, 8u>(
                x.data(), w.data(), ba ? ba->as<float>() : nullptr,
                part_val.data(), part_idx.data(), M, V, K, n_tiles);
        });
    } else {
        n_tiles = (V + LM_PREFILL_BN - 1) / LM_PREFILL_BN;
        uint32_t tiles_m = (M + LM_PREFILL_BM - 1) / LM_PREFILL_BM;
        part_val.resize((size_t)M * n_tiles);
        part_idx.resize((size_t)M * n_tiles);
        size_t smem = (LM_PREFILL_BM * LM_PREFILL_BK + LM_PREFILL_BN * LM_PREFILL_BK) * sizeof(float);
        candle_emu::launch(n_tiles, tiles_m, 1, LM_NT, smem, [&]() {
            lm_stage1_prefill_impl<TX>(
                x.data(), w.data(), ba ? ba->as<float>() : nullptr,
                part_val.data(), part_idx.data(), M, V, K, n_tiles);
        });
    }

    std::vector<ax_u32> out_id(M);
    std::vector<float> out_val(M);
    std::vector<ax_u8> out_res(M);
    candle_emu::launch(M, 1, 1, LM_NT, 4 * LM_NT * sizeof(float), [&]() {
        lm_stage2_reduce_cs_impl<TX>(
            part_val.data(), part_idx.data(),
            ca ? ca->as<ax_u32>() : nullptr,
            x.data(),
            mua ? mua->as<float>() : nullptr,
            w_bound, b_max,
            out_id.data(), out_val.data(), out_res.data(),
            M, n_tiles, K, check_cs);
    });

    out["ids"] = raw_arr(1, out_id);
    out["vals"] = raw_arr(0, out_val);
    out["rescue"] = raw_arr(4, out_res);
}

template <typename TL, typename TH>
static void run_logits(const Arrays &in, Arrays &out) {
    const uint32_t *dm = need(in, "dims").as<uint32_t>();
    const uint32_t M = dm[0], V = dm[1], K = dm[2], check_cs = dm[3], stride = dm[4];
    const float *fp = need(in, "fparams").as<float>();
    const float w_bound = fp[0], b_max = fp[1];

    const Arr &la = need(in, "sub_logits");
    const Arr *ha = opt(in, "hidden");
    const Arr *ca = opt(in, "core_ids");
    const Arr *mua = opt(in, "mu_rest");

    auto logits = convert<TL>(la.as<float>(), la.len<float>());
    std::vector<TH> hidden;
    if (ha) hidden = convert<TH>(ha->as<float>(), ha->len<float>());

    std::vector<ax_u32> out_id(M);
    std::vector<float> out_val(M);
    std::vector<ax_u8> out_res(M);
    candle_emu::launch(M, 1, 1, LM_NT, 4 * LM_NT * sizeof(float), [&]() {
        lm_logits_argmax_cs_impl<TL, TH>(
            logits.data(), stride,
            ca ? ca->as<ax_u32>() : nullptr,
            ha ? hidden.data() : nullptr,
            mua ? mua->as<float>() : nullptr,
            w_bound, b_max,
            out_id.data(), out_val.data(), out_res.data(),
            M, V, K, check_cs);
    });

    out["ids"] = raw_arr(1, out_id);
    out["vals"] = raw_arr(0, out_val);
    out["rescue"] = raw_arr(4, out_res);
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
    const uint32_t *dts = need(in, "dtype").as<uint32_t>(); // [l_dt, h_dt]

    if (op == "dense") {
        if (dts[0] == 0) run_dense<float>(in, out);
        else if (dts[0] == 1) run_dense<ax_f16>(in, out);
        else run_dense<ax_bf16>(in, out);
    } else if (op == "logits") {
        uint32_t ldt = dts[0], hdt = dts[1];
        if (ldt == 0 && hdt == 0) run_logits<float, float>(in, out);
        else if (ldt == 1 && hdt == 1) run_logits<ax_f16, ax_f16>(in, out);
        else if (ldt == 2 && hdt == 2) run_logits<ax_bf16, ax_bf16>(in, out);
        else if (ldt == 0 && hdt == 1) run_logits<float, ax_f16>(in, out);
        else run_logits<float, ax_bf16>(in, out);
    } else {
        throw std::runtime_error("unknown op: " + op);
    }

    write_arrays(argv[2], out);
    return 0;
}
