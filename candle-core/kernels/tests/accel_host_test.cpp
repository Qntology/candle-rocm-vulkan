// Runs the fused attention / TurboQuant kernels on the host emulator.
//
// Driven by candle-core/tests/accel_kernels_host.rs: `accel_host_test <in> <out>` reads named
// arrays from <in>, runs the operation named by the `op` array and writes the result arrays
// to <out>. File format: repeated [u32 name_len][name][u8 kind][u64 bytes][bytes].
#define CANDLE_ACCEL_HOST 1
#include "../flash_attn_impl.h"
#include "../turbo_quant_impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
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

template <typename T> static Arr make_arr(const std::vector<T> &v, uint8_t kind) {
    Arr a;
    a.kind = kind;
    a.bytes.resize(v.size() * sizeof(T));
    memcpy(a.bytes.data(), v.data(), a.bytes.size());
    return a;
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

// f32 values -> storage of the kernel element type
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
template <> std::vector<ax_fp8> convert<ax_fp8>(const float *x, size_t n) {
    std::vector<ax_fp8> v(n);
    for (size_t i = 0; i < n; ++i) v[i].b = (ax_u8)ax_f_to_e4m3(x[i]);
    return v;
}

static float to_f32(float v) { return v; }
static float to_f32(ax_f16 v) { return ax_h2f(v.b); }
static float to_f32(ax_bf16 v) { return ax_bf2f(v.b); }

// Block table from pg_off_k / pg_off_v (u64 element or word offsets into the k / v arrays),
// pg_rows, pg_cap_k and pg_cap_v (u32).
static ax_pages make_pages(const Arrays &in, ax_u64 kbase, ax_u64 vbase, size_t esz) {
    const Arr &rows = need(in, "pg_rows");
    const ax_u64 *ok = need(in, "pg_off_k").as<ax_u64>();
    const ax_u64 *ov = need(in, "pg_off_v").as<ax_u64>();
    const ax_u32 *ck = need(in, "pg_cap_k").as<ax_u32>();
    const ax_u32 *cv = need(in, "pg_cap_v").as<ax_u32>();
    ax_pages pg;
    memset(&pg, 0, sizeof(pg));
    pg.n = (ax_u32)rows.len<ax_u32>();
    if (pg.n > AX_MAX_PAGES) throw std::runtime_error("too many pages");
    for (ax_u32 i = 0; i < pg.n; ++i) {
        pg.k[i] = kbase + ok[i] * esz;
        pg.v[i] = vbase + ov[i] * esz;
        pg.rows[i] = rows.as<ax_u32>()[i];
        pg.k_cap[i] = ck[i];
        pg.v_cap[i] = cv[i];
    }
    return pg;
}

// ---------------------------------------------------------------------------
// flash attention
// ---------------------------------------------------------------------------

template <typename TQ, typename TKV, int BR, int BC, int DMAX>
static void run_tile(const Arrays &in, const TQ *q, const TKV *k, const TKV *v, TQ *o, const ax_u32 *cu_q,
                     const ax_u32 *cu_k, const float *alibi, const ax_u32 *dims, const ax_u64 *st,
                     const float *fp, const int *ip, ax_u32 varlen, ax_u32 alibi_bs, unsigned gx) {
    const ax_u32 B = dims[0], Lq = dims[1], Lk = dims[2], Hq = dims[3], Hkv = dims[4], D = dims[5];
    const size_t smem = ((BR + BC) * (D + 1) + BR * BC) * 4;
    candle_emu::launch(gx, Hq, B, FA_NT, smem, [&] {
        fa_tile_impl<TQ, TKV, BR, BC, DMAX>(q, k, v, o, nullptr, nullptr, alibi, cu_q, cu_k, Lq, Lk, Hq,
                                            Hkv, D, st[0], st[1], st[2], st[3], st[4], st[5], st[6],
                                            st[7], st[8], fp[0], fp[1], ip[0], ip[1], fp[2], fp[3], 1,
                                            alibi_bs, varlen, 0);
    });
    (void)in;
}

template <typename TQ, typename TKV> static Arrays op_fa(const Arrays &in) {
    const ax_u32 *dims = need(in, "dims").as<ax_u32>(); // B Lq Lk Hq Hkv D total_q
    const ax_u64 *st = need(in, "strides").as<ax_u64>();
    const float *fp = need(in, "fparams").as<float>();  // scale softcap kd vd
    const int *ip = need(in, "iparams").as<int>();      // wl wr
    const ax_u32 *cfg = need(in, "cfg").as<ax_u32>();   // kind dmax version splits varlen gx alibi_bs
    const Arr &qa = need(in, "q"), &ka = need(in, "k"), &va = need(in, "v");
    std::vector<TQ> q = convert<TQ>(qa.as<float>(), qa.len<float>());
    std::vector<TKV> k = convert<TKV>(ka.as<float>(), ka.len<float>());
    std::vector<TKV> v = convert<TKV>(va.as<float>(), va.len<float>());
    const ax_u32 B = dims[0], Lq = dims[1], Lk = dims[2], Hq = dims[3], Hkv = dims[4], D = dims[5];
    const ax_u64 total_q = dims[6];
    const ax_u64 rows = total_q * Hq;
    std::vector<TQ> o(rows * D);
    const Arr *cq = opt(in, "cu_q"), *ck = opt(in, "cu_k"), *al = opt(in, "alibi");
    const ax_u32 *cu_q = cq ? cq->as<ax_u32>() : nullptr;
    const ax_u32 *cu_k = ck ? ck->as<ax_u32>() : nullptr;
    const float *alibi = al ? al->as<float>() : nullptr;
    const ax_u32 kind = cfg[0], dmax = cfg[1], version = cfg[2], splits = cfg[3], varlen = cfg[4];
    const unsigned gx = cfg[5];
    const ax_u32 alibi_bs = cfg[6];
    const ax_u32 G = need(in, "cfg").len<ax_u32>() > 7 ? cfg[7] : 1; // GQA pack of the row kernel
    if (kind == 0 && G > 1) {
        std::vector<float> po(splits > 1 ? splits * rows * D : 1), pl(splits > 1 ? splits * rows : 1);
        const bool paged = opt(in, "pg_rows") != nullptr;
        ax_pages pg;
        memset(&pg, 0, sizeof(pg));
        if (paged)
            pg = make_pages(in, reinterpret_cast<ax_u64>(k.data()), reinterpret_cast<ax_u64>(v.data()),
                            sizeof(TKV));
        const TKV *kp = paged ? nullptr : k.data(), *vp = paged ? nullptr : v.data();
        auto run = [&](auto pack) {
            constexpr int GG = decltype(pack)::value;
            candle_emu::launch(gx, splits, B, FA_NT, GG * (D + 2 * FA_NT) * 4, [&] {
                if (paged)
                    fa_rowg_core<TQ, TKV, GG, true>(q.data(), kp, vp, o.data(), po.data(), pl.data(), alibi,
                                                    nullptr, nullptr, Lq, Lk, Hq, Hkv, D, st[0], st[1], st[2],
                                                    st[3], st[4], st[5], st[6], st[7], st[8], fp[0], fp[1],
                                                    ip[0], ip[1], fp[2], fp[3], splits, alibi_bs, 0, rows, &pg);
                else
                    fa_rowg_core<TQ, TKV, GG, false>(q.data(), kp, vp, o.data(), po.data(), pl.data(), alibi,
                                                     nullptr, nullptr, Lq, Lk, Hq, Hkv, D, st[0], st[1], st[2],
                                                     st[3], st[4], st[5], st[6], st[7], st[8], fp[0], fp[1],
                                                     ip[0], ip[1], fp[2], fp[3], splits, alibi_bs, 0, rows,
                                                     nullptr);
            });
        };
        if (G == 2) run(std::integral_constant<int, 2>{});
        else if (G == 4) run(std::integral_constant<int, 4>{});
        else if (G == 8) run(std::integral_constant<int, 8>{});
        else throw std::runtime_error("unknown gqa pack");
        if (splits > 1) {
            candle_emu::launch((unsigned)std::min<ax_u64>(rows, 7), 1, 1, FA_NT, splits * 4, [&] {
                fa_combine_impl<TQ>(po.data(), pl.data(), o.data(), rows, D, splits);
            });
        }
    } else if (kind == 0) {
        std::vector<float> po(splits > 1 ? splits * rows * D : 1), pl(splits > 1 ? splits * rows : 1);
        if (opt(in, "pg_rows")) {
            // paged row kernel: the blocks live inside k / v at the given element offsets
            const ax_pages pg = make_pages(in, reinterpret_cast<ax_u64>(k.data()),
                                           reinterpret_cast<ax_u64>(v.data()), sizeof(TKV));
            candle_emu::launch(gx, splits, B, FA_NT, (D + 2 * FA_NT) * 4, [&] {
                fa_rowp_impl<TQ, TKV>(q.data(), nullptr, nullptr, o.data(), po.data(), pl.data(), alibi,
                                      nullptr, nullptr, Lq, Lk, Hq, Hkv, D, st[0], st[1], st[2], st[3], st[4],
                                      st[5], st[6], st[7], st[8], fp[0], fp[1], ip[0], ip[1], fp[2], fp[3],
                                      splits, alibi_bs, 0, rows, pg);
            });
        } else {
            candle_emu::launch(gx, splits, B, FA_NT, (D + 2 * FA_NT) * 4, [&] {
                fa_row_impl<TQ, TKV>(q.data(), k.data(), v.data(), o.data(), po.data(), pl.data(), alibi, cu_q,
                                     cu_k, Lq, Lk, Hq, Hkv, D, st[0], st[1], st[2], st[3], st[4], st[5], st[6],
                                     st[7], st[8], fp[0], fp[1], ip[0], ip[1], fp[2], fp[3], splits, alibi_bs,
                                     varlen, rows);
            });
        }
        if (splits > 1) {
            candle_emu::launch((unsigned)std::min<ax_u64>(rows, 7), 1, 1, FA_NT, splits * 4, [&] {
                fa_combine_impl<TQ>(po.data(), pl.data(), o.data(), rows, D, splits);
            });
        }
    } else {
#define FA_TILE_CASE(V, DM, BR, BC)                                                                \
    if (version == V && dmax == DM) {                                                              \
        run_tile<TQ, TKV, BR, BC, DM>(in, q.data(), k.data(), v.data(), o.data(), cu_q, cu_k, alibi, \
                                      dims, st, fp, ip, varlen, alibi_bs, gx);                     \
    } else
        FA_TILE_CASE(2, 64, 32, 32)
        FA_TILE_CASE(2, 128, 16, 32)
        FA_TILE_CASE(2, 256, 8, 16)
        FA_TILE_CASE(3, 64, 64, 64)
        FA_TILE_CASE(3, 128, 32, 64)
        FA_TILE_CASE(3, 256, 16, 32)
        throw std::runtime_error("unknown tile config");
#undef FA_TILE_CASE
    }
    std::vector<float> out(o.size());
    for (size_t i = 0; i < o.size(); ++i) out[i] = to_f32(o[i]);
    Arrays res;
    res["o"] = make_arr(out, 0);
    return res;
}

static Arrays op_fa_dispatch(const Arrays &in) {
    // q: 0 f32 1 f16 2 bf16; kv: 0 same 1 fp8 (2 f16 / 3 bf16 with f32 queries)
    const ax_u32 *dt = need(in, "dtypes").as<ax_u32>();
    const ax_u32 qd = dt[0], kv8 = dt[1];
    if (qd == 0 && kv8 == 2) return op_fa<float, ax_f16>(in);
    if (qd == 0 && kv8 == 3) return op_fa<float, ax_bf16>(in);
    if (qd == 0) return kv8 ? op_fa<float, ax_fp8>(in) : op_fa<float, float>(in);
    if (qd == 1) return kv8 ? op_fa<ax_f16, ax_fp8>(in) : op_fa<ax_f16, ax_f16>(in);
    return kv8 ? op_fa<ax_bf16, ax_fp8>(in) : op_fa<ax_bf16, ax_bf16>(in);
}

// ---------------------------------------------------------------------------
// turbo quant
// ---------------------------------------------------------------------------

// p: rows D mode cb code_words aux_words row_words gx src_dtype
template <typename T> static Arrays op_tq_quant_t(const Arrays &in, const ax_u32 *p) {
    const Arr &xa = need(in, "x");
    std::vector<T> x = convert<T>(xa.as<float>(), xa.len<float>());
    const float *tab = need(in, "tab").as<float>();
    const ax_u64 rows = p[0];
    const ax_u32 D = p[1];
    std::vector<ax_u32> out(rows * p[6], 0xdeadbeefu);
    const size_t smem = (2 * D + TQ_NT + D / 16 + 1) * 4;
    candle_emu::launch(p[7], 1, 1, TQ_NT, smem, [&] {
        tq_quant_impl<T>(x.data(), out.data(), tab, rows, D, D, p[2], p[3], p[4], p[5], p[6]);
    });
    Arrays res;
    res["out"] = make_arr(out, 1);
    // residual for prod rows
    if (p[2] == TQ_PROD) {
        std::vector<float> resid(rows * D);
        candle_emu::launch(p[7], 1, 1, TQ_NT, 0, [&] {
            tq_resid_impl<T>(x.data(), out.data(), tab, resid.data(), rows, D, D, p[3], p[6], p[4] + p[5]);
        });
        res["resid"] = make_arr(resid, 0);
    }
    return res;
}

static Arrays op_tq_quant(const Arrays &in) {
    const ax_u32 *p = need(in, "params").as<ax_u32>();
    if (p[8] == 1) return op_tq_quant_t<ax_f16>(in, p);
    if (p[8] == 2) return op_tq_quant_t<ax_bf16>(in, p);
    return op_tq_quant_t<float>(in, p);
}

// p: rows D code_words row_words gx
static Arrays op_tq_signs(const Arrays &in) {
    const ax_u32 *p = need(in, "params").as<ax_u32>();
    const ax_u64 rows = p[0];
    const ax_u32 D = p[1];
    std::vector<ax_u32> out(rows * p[3], 0xdeadbeefu);
    candle_emu::launch(p[4], 1, 1, TQ_NT, TQ_NT * 4, [&] {
        tq_signs_impl(need(in, "data").as<ax_u32>(), need(in, "proj").as<float>(), need(in, "resid").as<float>(),
                      out.data(), rows, D, p[2], p[3]);
    });
    Arrays res;
    res["out"] = make_arr(out, 1);
    return res;
}

// p: rows D mode cb code_words row_words rsc_off gx want_zs
static Arrays op_tq_dequant(const Arrays &in) {
    const ax_u32 *p = need(in, "params").as<ax_u32>();
    const ax_u64 rows = p[0];
    const ax_u32 D = p[1];
    std::vector<float> out(rows * D), zs(p[8] ? rows * D : 0);
    candle_emu::launch(p[7], 1, 1, TQ_NT, 0, [&] {
        tq_dequant_impl(need(in, "data").as<ax_u32>(), need(in, "tab").as<float>(), out.data(),
                        p[8] ? zs.data() : nullptr, rows, D, p[2], p[3], p[4], p[5], p[6]);
    });
    Arrays res;
    res["out"] = make_arr(out, 0);
    if (p[8]) res["zs"] = make_arr(zs, 0);
    return res;
}

// u: B Lq Lk Hq Hkv D k_mode k_cb k_cw k_rsc v_mode v_cb v_cw v_rsc splits gx alibi_bs fp8
// s: q_sb q_sh q_ss k_sb k_sh k_ss v_sb v_sh v_ss ; f: scale softcap ; i: wl wr
// Optional block table pg_* (word offsets into kd / vd): the paged kernels.
static Arrays op_tq_attn(const Arrays &in) {
    const ax_u32 *u = need(in, "uparams").as<ax_u32>();
    const ax_u64 *s = need(in, "strides").as<ax_u64>();
    const float *f = need(in, "fparams").as<float>();
    const int *ip = need(in, "iparams").as<int>();
    const ax_u32 B = u[0], Lq = u[1], Lk = u[2], Hq = u[3], Hkv = u[4], D = u[5], splits = u[14];
    const ax_u64 rows = (ax_u64)B * Hq * Lq;
    std::vector<float> o(rows * D), po(splits > 1 ? splits * rows * D : 1), pl(splits > 1 ? splits * rows : 1);
    const Arr *al = opt(in, "alibi");
    const size_t smem = (2 * D + 2 * TQ_NT + 32) * 4;
    const ax_u32 *kd = need(in, "kd").as<ax_u32>(), *vd = need(in, "vd").as<ax_u32>();
    const float *qx = need(in, "q").as<float>(), *tab = need(in, "tab").as<float>();
    const float *alibi = al ? al->as<float>() : nullptr;
    // the FP8 specialization (both sides FP8 rows)
    const bool fp8 = need(in, "uparams").len<ax_u32>() > 17 && u[17] != 0;
    const bool paged = opt(in, "pg_rows") != nullptr;
    ax_pages pg;
    memset(&pg, 0, sizeof(pg));
    if (paged) pg = make_pages(in, reinterpret_cast<ax_u64>(kd), reinterpret_cast<ax_u64>(vd), 4);
    // GQA pack (query heads per workgroup, u[18]): the tq_attng kernels
    const ax_u32 G = need(in, "uparams").len<ax_u32>() > 18 ? u[18] : 1;
    const ax_u32 *k = paged ? nullptr : kd, *v = paged ? nullptr : vd;
#define TQ_HOST_ARGS                                                                               \
    qx, k, v, tab, alibi, o.data(), po.data(), pl.data(), Lq, Lk, Hq, Hkv, D, s[0], s[1], s[2], s[3], \
        s[4], s[5], s[6], s[7], s[8], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], f[0], f[1],  \
        ip[0], ip[1], splits, u[16], rows
    if (G > 1) {
        auto run = [&](auto pack) {
            constexpr int GG = decltype(pack)::value;
            candle_emu::launch(u[15], splits, B, TQ_NT, (2 * GG * D + 2 * GG * TQ_NT + 32) * 4, [&] {
                if (paged && fp8) tq_attng_impl<true, true, GG>(TQ_HOST_ARGS, &pg);
                else if (paged) tq_attng_impl<false, true, GG>(TQ_HOST_ARGS, &pg);
                else if (fp8) tq_attng_impl<true, false, GG>(TQ_HOST_ARGS, nullptr);
                else tq_attng_impl<false, false, GG>(TQ_HOST_ARGS, nullptr);
            });
        };
        if (G == 2) run(std::integral_constant<int, 2>{});
        else if (G == 4) run(std::integral_constant<int, 4>{});
        else if (G == 8) run(std::integral_constant<int, 8>{});
        else throw std::runtime_error("unknown gqa pack");
    } else {
        candle_emu::launch(u[15], splits, B, TQ_NT, smem, [&] {
            if (paged && fp8) tq_attnp_fp8_impl(TQ_HOST_ARGS, pg);
            else if (paged) tq_attnp_impl(TQ_HOST_ARGS, pg);
            else if (fp8) tq_attn_fp8_impl(TQ_HOST_ARGS);
            else tq_attn_impl(TQ_HOST_ARGS);
        });
    }
#undef TQ_HOST_ARGS
    if (splits > 1) {
        candle_emu::launch(5, 1, 1, TQ_NT, splits * 4,
                           [&] { tq_combine_impl(po.data(), pl.data(), o.data(), rows, D, splits); });
    }
    Arrays res;
    res["o"] = make_arr(o, 0);
    return res;
}

// fa_combine before the per-row weights (one expf per element and split), for the check that
// the per-row weights give the same bits.
static void fa_combine_legacy(const float *part_o, const float *part_lse, float *o, ax_u64 rows, ax_u32 D,
                              ax_u32 n_splits) {
    for (ax_u64 row = 0; row < rows; ++row) {
        float mx = -AX_INF;
        for (ax_u32 s = 0; s < n_splits; ++s) mx = fmaxf(mx, part_lse[(ax_u64)s * rows + row]);
        float wsum = 0.0f;
        for (ax_u32 s = 0; s < n_splits; ++s) {
            const float e = part_lse[(ax_u64)s * rows + row];
            wsum += e == -AX_INF ? 0.0f : expf(e - mx);
        }
        for (ax_u32 d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (ax_u32 s = 0; s < n_splits; ++s) {
                const float e = part_lse[(ax_u64)s * rows + row];
                if (e != -AX_INF) acc += expf(e - mx) * part_o[((ax_u64)s * rows + row) * D + d];
            }
            o[row * D + d] = wsum > 0.0f ? acc / wsum : 0.0f;
        }
    }
}

// p: rows D splits ; part_o, part_lse -> o (fa_combine), t (tq_combine), legacy
static Arrays op_combine(const Arrays &in) {
    const ax_u32 *p = need(in, "params").as<ax_u32>();
    const ax_u64 rows = p[0];
    const ax_u32 D = p[1], splits = p[2];
    const float *po = need(in, "part_o").as<float>(), *pl = need(in, "part_lse").as<float>();
    std::vector<float> o(rows * D), t(rows * D), legacy(rows * D);
    candle_emu::launch(3, 1, 1, FA_NT, splits * 4, [&] { fa_combine_impl<float>(po, pl, o.data(), rows, D, splits); });
    candle_emu::launch(2, 1, 1, TQ_NT, splits * 4, [&] { tq_combine_impl(po, pl, t.data(), rows, D, splits); });
    fa_combine_legacy(po, pl, legacy.data(), rows, D, splits);
    Arrays res;
    res["o"] = make_arr(o, 0);
    res["t"] = make_arr(t, 0);
    res["legacy"] = make_arr(legacy, 0);
    return res;
}

// ax_h2f before the branch-free version, for the check that both give the same bits.
static float h2f_legacy(ax_u16 h) {
    const ax_u32 s = ((ax_u32)(h & 0x8000u)) << 16;
    const ax_u32 e = (h >> 10) & 0x1fu;
    const ax_u32 m = h & 0x3ffu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f; // m * 2^-24
        return s ? -v : v;
    }
    if (e == 31) return ax_u2f(s | 0x7f800000u | (m << 13));
    return ax_u2f(s | ((e + 112u) << 23) | (m << 13));
}

// Every e4m3 code through ax_e4m3_to_f / ax_e4m3_to_f_fast and every f16 value through ax_h2f
// and the legacy decoder (the element loads of the kernels).
static Arrays op_decoders(const Arrays &) {
    std::vector<float> e_ref(256), e_fast(256), h(65536), h_legacy(65536);
    for (ax_u32 c = 0; c < 256; ++c) {
        e_ref[c] = ax_e4m3_to_f(c);
        e_fast[c] = ax_e4m3_to_f_fast(c);
    }
    for (ax_u32 x = 0; x < 65536; ++x) {
        h[x] = ax_h2f((ax_u16)x);
        h_legacy[x] = h2f_legacy((ax_u16)x);
    }
    Arrays res;
    res["e4m3_ref"] = make_arr(e_ref, 0);
    res["e4m3_fast"] = make_arr(e_fast, 0);
    res["h2f"] = make_arr(h, 0);
    res["h2f_legacy"] = make_arr(h_legacy, 0);
    return res;
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
        if (op == "fa") out = op_fa_dispatch(in);
        else if (op == "tq_quant") out = op_tq_quant(in);
        else if (op == "tq_signs") out = op_tq_signs(in);
        else if (op == "tq_dequant") out = op_tq_dequant(in);
        else if (op == "tq_attn") out = op_tq_attn(in);
        else if (op == "combine") out = op_combine(in);
        else if (op == "decoders") out = op_decoders(in);
        else throw std::runtime_error("unknown op " + op);
        write_arrays(argv[2], out);
    } catch (const std::exception &e) {
        fprintf(stderr, "accel_host_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
