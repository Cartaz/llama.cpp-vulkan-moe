#include "expert-pool.h"
#include "ggml-cpu.h"
#include "../../ggml/src/ggml-quants.h"
#include "../../ggml/src/ggml-backend-impl.h"
#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cstdint>
#include <memory>

bool check_expert_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output);
bool check_scheduler_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output);
bool check_expert_placement(ggml_backend_t target, ggml_backend_t cpu, FILE * output);
bool check_operator_numerics(ggml_backend_t target, FILE * output);
bool check_multilayer_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output);
bool check_transfer_cost(ggml_backend_t target);

namespace {
constexpr int experts = 256, topk = 8, width = 256, rows = 256, outputs = 8;
using values = std::array<std::vector<float>, 3>;

float source_value(int projection, int expert, int row, int col) {
    return float((projection * 3 + expert * 7 + row * 3 + col) % 17 - 8) / 128;
}
float activation_value(int token, int slot, int col) {
    return float((token * 5 + slot * 2 + col) % 13 - 6) / 32;
}

struct tensor_store {
    ggml_context * ctx = ggml_init({65536, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~tensor_store() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

struct graph_run {
    ggml_backend_sched_t sched = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input = nullptr, * storage = nullptr;
    std::array<ggml_tensor *, 3> result{};
    std::vector<std::array<ggml_tensor *, 3>> layer_results;
    ggml_cgraph * graph = nullptr;
    ~graph_run() {
        if (sched) { ggml_backend_sched_free(sched); }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    bool init(ggml_backend_t backend, const std::array<ggml_tensor *, 3> & weights, int tokens, bool broadcast, bool strided, ggml_backend_t cpu = nullptr, bool parallel = false, bool callback = false, bool placement = false, bool op_offload = true,
            const std::vector<std::array<ggml_tensor *, 3>> * layers = nullptr, ggml_backend_sched_t reuse = nullptr) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        if (!ctx) {
            return false;
        }
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, broadcast ? 1 : topk, tokens);
        storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, strided ? 16 : topk, tokens);
        auto * ids = strided ? ggml_view_2d(ctx, storage, topk, tokens, storage->nb[1], sizeof(int32_t)) : storage;
        graph = ggml_new_graph_custom(ctx, 128, false);
        const auto all_weights = layers ? *layers : std::vector<std::array<ggml_tensor *, 3>>{weights};
        auto * layer_input = input;
        for (const auto & layer : all_weights) {
            std::array<ggml_tensor *, 3> computed{};
            computed[0] = ggml_mul_mat_id(ctx, layer[0], layer_input, ids);
            // CPU bridges give each projection a separate scheduler split.
            auto * up_input = layers ? ggml_add(ctx,layer_input,ggml_scale(ctx,ggml_sum(ctx,computed[0]),0)) : layer_input;
            computed[1] = ggml_mul_mat_id(ctx, layer[1], up_input, ids);
            auto * intermediate = ggml_mul(ctx, ggml_silu(ctx, computed[0]), computed[1]);
            computed[2] = ggml_mul_mat_id(ctx, layer[2], intermediate, ids);
            if (layers) { layer_input = ggml_add(ctx,input,ggml_scale(ctx,ggml_sum(ctx,computed[2]),0)); }
            layer_results.push_back(computed);
            for (auto * tensor : computed) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(graph, tensor);
            }
        }
        result = layer_results.front();
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph,i);
            auto * expected = layers && cpu && node->op != GGML_OP_MUL_MAT_ID ? cpu : backend;
            if (!ggml_backend_supports_op(expected, node)) {
                fprintf(stderr, "pool fixture: unsupported op %s\n", ggml_op_name(ggml_graph_node(graph, i)->op));
                return false;
            }
        }
        if (cpu) {
            ggml_set_input(input);
            ggml_set_input(storage);
            if (layers) { ggml_set_output(storage); }
            std::vector<ggml_backend_t> backends = backend == cpu ? std::vector<ggml_backend_t>{cpu} : std::vector<ggml_backend_t>{backend, cpu};
            sched = reuse ? reuse : ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 128, parallel, op_offload);
            if (reuse) { ggml_backend_sched_reset(sched); }
            if (callback) {
                ggml_backend_sched_set_eval_callback(sched, [](ggml_tensor *, bool, void *) { return true; }, nullptr);
            }
            ggml_backend_sched_set_tensor_backend(sched, input, cpu);
            ggml_backend_sched_set_tensor_backend(sched, storage, cpu);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (layers && node->op != GGML_OP_MUL_MAT_ID) {
                    ggml_backend_sched_set_tensor_backend(sched,node,cpu);
                } else if (!placement) {
                    ggml_backend_sched_set_tensor_backend(sched, node, ggml_is_view(node) ? cpu : backend);
                }
            }
            return ggml_backend_sched_alloc_graph(sched, graph);
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        return buffer != nullptr;
    }
    bool submit(ggml_backend_t backend, const std::vector<int32_t> & ids, int epoch = 0) {
        std::vector<int32_t> id_data(ggml_nelements(storage), -1);
        const bool strided = storage->ne[0] != topk;
        for (int token = 0; token < storage->ne[1]; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                id_data[token * storage->ne[0] + slot + (strided ? 1 : 0)] = ids[token * topk + slot];
            }
        }
        std::vector<float> data(ggml_nelements(input));
        for (int token = 0; token < input->ne[2]; ++token) {
            for (int slot = 0; slot < input->ne[1]; ++slot) {
                for (int col = 0; col < width; ++col) {
                    data[(token * input->ne[1] + slot) * width + col] = activation_value(token + epoch, slot, col);
                }
            }
        }
        ggml_backend_tensor_set(storage, id_data.data(), 0, ggml_nbytes(storage));
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        return (sched ? ggml_backend_sched_graph_compute(sched, graph) : ggml_backend_graph_compute_async(backend, graph)) == GGML_STATUS_SUCCESS;
    }
    values read(ggml_backend_t backend) {
        ggml_backend_synchronize(backend);
        values data;
        for (int i = 0; i < 3; ++i) {
            data[i].resize(ggml_nelements(result[i]));
            ggml_backend_tensor_get(result[i], data[i].data(), 0, ggml_nbytes(result[i]));
        }
        return data;
    }
};

bool valid(const values & data) {
    for (const auto & projection : data) {
        bool nonzero = false;
        for (float value : projection) {
            if (!std::isfinite(value)) {
                return false;
            }
            nonzero = nonzero || value != 0;
        }
        if (!nonzero) {
            return false;
        }
    }
    return true;
}
bool identical(const values & a, const values & b) {
    for (int i = 0; i < 3; ++i) {
        if (a[i].size() != b[i].size()) {
            return false;
        }
        if (memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float))) {
            size_t differing = 0, first = a[i].size();
            double maximum = 0;
            for (size_t n = 0; n < a[i].size(); ++n) {
                if (memcmp(&a[i][n], &b[i][n], sizeof(float))) {
                    ++differing;
                    first = std::min(first, n);
                    maximum = std::max(maximum, std::abs(double(a[i][n]) - b[i][n]));
                }
            }
            fprintf(stderr, "pool mismatch projection=%d elements=%zu different=%zu first=%zu actual=%.9g reference=%.9g max_abs=%.9g finite_nonzero=%d/%d\n",
                i, a[i].size(), differing, first, a[i][first], b[i][first], maximum, valid(a), valid(b));
            return false;
        }
    }
    return valid(a) && valid(b);
}
bool same_counts(const moe_expert_pool::counters & a, const moe_expert_pool::counters & b) {
    return a.hits == b.hits && a.misses == b.misses && a.evictions == b.evictions && a.upload_bytes == b.upload_bytes;
}

struct saved_env {
    const char * key;
    bool present;
    std::string value;
    explicit saved_env(const char * key) : key(key), present(getenv(key) != nullptr), value(present ? getenv(key) : "") {}
    ~saved_env() { if (present) { setenv(key, value.c_str(), 1); } else { unsetenv(key); } }
};

struct fail_allocation {
    // Test process only. No graph runs while this allocator is replaced.
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t (*saved)(ggml_backend_buffer_type_t, size_t);
    size_t calls = 0;
    static fail_allocation * active;
    static ggml_backend_buffer_t reject(ggml_backend_buffer_type_t, size_t) {
        ++active->calls;
        return nullptr;
    }
    explicit fail_allocation(ggml_backend_t backend) : buft(ggml_backend_get_default_buffer_type(backend)), saved(buft->iface.alloc_buffer) {
        active = this;
        buft->iface.alloc_buffer = reject;
    }
    ~fail_allocation() { buft->iface.alloc_buffer = saved; active = nullptr; }
};
fail_allocation * fail_allocation::active = nullptr;

struct abort_pool_graph {
    ggml_backend_t backend;
    ggml_status (*saved)(ggml_backend_t, ggml_cgraph *);
    int calls = 0;
    static abort_pool_graph * active;
    static ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph,i);
            if (node->op == GGML_OP_MUL_MAT_ID && strncmp(node->src[0]->name,"pool.",5) == 0) {
                ++active->calls;
                return GGML_STATUS_ABORTED;
            }
        }
        return active->saved(backend,graph);
    }
    explicit abort_pool_graph(ggml_backend_t backend) : backend(backend), saved(backend->iface.graph_compute) {
        active = this;
        backend->iface.graph_compute = compute;
    }
    ~abort_pool_graph() { backend->iface.graph_compute = saved; active = nullptr; }
};
abort_pool_graph * abort_pool_graph::active = nullptr;

bool scheduler_case(ggml_backend_t target, ggml_backend_t cpu, const std::array<ggml_tensor *, 3> & source,
        int quant, int tokens, bool broadcast, bool strided, FILE * output, const char * config, bool parallel, bool callback, bool placement) {
    saved_env env("GGML_SCHED_EXPERT_POOL");
    saved_env place_env("GGML_SCHED_EXPERT_GPU_LAYER");
    if (placement) {
        setenv(place_env.key, "3", 1);
    }
    graph_run off, on;
    unsetenv(env.key);
    bool ok = off.init(target, source, tokens, broadcast, strided, cpu, parallel, callback);
    setenv(env.key, config, 1);
    ok = on.init(target, source, tokens, broadcast, strided, cpu, parallel, callback, placement) && ok;
    if (!ok) {
        return false;
    }
    for (auto * node : on.result) {
        ok = ggml_backend_sched_get_tensor_backend(on.sched, node) == target && ok;
    }
    for (int pass : {0, 1, 2, 3, 6, 7}) {
        std::vector<int32_t> ids(tokens * topk);
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                ids[token * topk + slot] = pass == 0 ? (slot + token) % topk :
                    pass == 1 ? (topk - 1 - slot + token) % topk : pass == 2 ? (slot + token) % topk + 4 :
                    pass == 3 ? (token * topk + slot) % experts :
                    pass == 6 ? 248 + (slot + token) % topk : 248 + (topk - 1 - slot + token) % topk;
            }
        }
        ok = off.submit(target, ids) && ok;
        const auto reference = off.read(target);
        ok = on.submit(target, ids) && ok;
        const auto actual = on.read(target);
        ok = identical(actual, reference) && ok;
        // The source IDs and graph arguments must survive repeated submissions.
        std::vector<int32_t> stored(ggml_nelements(on.storage));
        ggml_backend_tensor_get(on.storage, stored.data(), 0, ggml_nbytes(on.storage));
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                ok = stored[token * on.storage->ne[0] + slot + (strided ? 1 : 0)] == ids[token * topk + slot] && ok;
            }
        }
        if (quant == 0) {
            for (int i = 0; i < 2; ++i) {
                for (int token = 0; token < tokens; ++token) {
                    for (int slot = 0; slot < topk; ++slot) {
                        for (int row = 0; row < rows; ++row) {
                            float expected = 0;
                            for (int col = 0; col < width; ++col) {
                                expected += source_value(i, ids[token * topk + slot], row, col) * activation_value(token, broadcast ? 0 : slot, col);
                            }
                            ok = actual[i][(token * topk + slot) * rows + row] == expected && ok;
                        }
                    }
                }
            }
        }
        if (output) {
            for (const auto & projection : actual) {
                ok = fwrite(projection.data(), sizeof(float), projection.size(), output) == projection.size() && ok;
            }
        }
        if (!ok) {
            fprintf(stderr, "scheduler pool failed quant=%d tokens=%d pass=%d config=%s\n", quant, tokens, pass, config);
            break;
        }
    }
    printf("scheduler pool quant=%d tokens=%d broadcast=%d strided=%d config=%s parallel=%d callback=%d %s\n",
            quant, tokens, broadcast, strided, config, parallel, callback, ok ? "OK" : "FAIL");
    return ok;
}

bool check_pool_case(ggml_backend_t target, ggml_backend_t cpu, int quant, int tokens, bool broadcast, bool strided, FILE * output, const char * config = nullptr, bool parallel = false, bool callback = false, bool placement = false) {
    tensor_store source_store, full_store;
    auto * source_ctx = source_store.ctx;
    auto * full_ctx = full_store.ctx;
    if (!source_ctx || !full_ctx) {
        return false;
    }
    const std::array<ggml_type, 3> types = quant == 0 ? std::array<ggml_type, 3>{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32} :
        quant == 1 ? std::array<ggml_type, 3>{GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K} :
                     quant == 2 ? std::array<ggml_type, 3>{GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K} :
                                  std::array<ggml_type, 3>{GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    std::array<ggml_tensor *, 3> source{}, full{};
    for (int i = 0; i < 3; ++i) {
        source[i] = ggml_new_tensor_3d(source_ctx, types[i], width, i == 2 ? outputs : rows, experts);
        full[i] = ggml_new_tensor_3d(full_ctx, types[i], width, i == 2 ? outputs : rows, experts);
        if (config) {
            const char * names[] = {"gate", "up", "down"};
            ggml_format_name(source[i], "blk.3.ffn_%s_exps.weight", names[i]);
        } else {
            ggml_format_name(source[i], "blk.3.projection%d_exps.weight", i);
        }
    }
    auto source_buffer = source_store.buffer = ggml_backend_alloc_ctx_tensors(source_ctx, cpu);
    auto full_buffer = full_store.buffer = ggml_backend_alloc_ctx_tensors(full_ctx, target);
    if (!source_buffer || !full_buffer) {
        return false;
    }
    std::vector<float> row(width);
    for (int i = 0; i < 3; ++i) {
        std::vector<uint8_t> bytes(ggml_nbytes(source[i]));
        for (int expert = 0; expert < experts; ++expert) {
            for (int r = 0; r < source[i]->ne[1]; ++r) {
                for (int col = 0; col < width; ++col) {
                    row[col] = source_value(i, expert, r, col);
                }
                auto * dest = bytes.data() + expert * source[i]->nb[2] + r * source[i]->nb[1];
                if (types[i] == GGML_TYPE_F32) {
                    memcpy(dest, row.data(), width * sizeof(float));
                } else {
                    ggml_get_type_traits(types[i])->from_float_ref(row.data(), dest, width);
                }
            }
        }
        ggml_backend_tensor_set(source[i], bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(full[i], bytes.data(), 0, bytes.size());
    }
    if (config) {
        ggml_backend_buffer_set_usage(source_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        return scheduler_case(target, cpu, source, quant, tokens, broadcast, strided, output, config, parallel, callback, placement);
    }
    bool ok = true;
    uint64_t misses = 0, hits = 0, evictions = 0, uploads = 0;
    size_t payload = 0, allocated = 0;
    int ready = 0, bypass = 0;
    {
        moe_expert_pool pool;
        ok = pool.init(target, source, 12, 16 * 1024 * 1024) && ok;
        payload = pool.payload_bytes();
        allocated = pool.allocated_bytes();
        moe_expert_pool unavailable, tiny, zero;
        ok = !tiny.init(target, source, 12, 1) && !zero.init(target, source, 0, 16 * 1024 * 1024) && ok;
        std::vector<int32_t> remapped;
        using status = moe_expert_pool::status;
        ok = unavailable.acquire({0}, remapped) == status::unavailable && remapped.empty() && ok;
        if (!ok) {
            return false;
        }
        std::array<ggml_tensor *, 3> cached{pool.weight(0), pool.weight(1), pool.weight(2)};
        graph_run cached_run, target_run, cpu_run, fallback_run;
        ok = cached_run.init(target, cached, tokens, broadcast, strided) && target_run.init(target, full, tokens, broadcast, strided) &&
             cpu_run.init(cpu, source, tokens, broadcast, strided) && fallback_run.init(cpu, source, tokens, broadcast, strided) && ok;
        if (!ok) {
            return false;
        }
        for (int pass = 0; pass < 8; ++pass) {
            std::vector<int32_t> ids(tokens * topk);
            for (int token = 0; token < tokens; ++token) {
                for (int slot = 0; slot < topk; ++slot) {
                    ids[token * topk + slot] = pass == 0 ? (slot + token) % topk :
                        pass == 1 ? (topk - 1 - slot + token) % topk :
                        pass == 2 ? (slot + token) % topk + 4 :
                        pass == 3 ? (token * topk + slot) % experts :
                        pass == 4 ? 255 : pass == 5 ? (slot % 2 ? 255 : 0) :
                        pass == 6 ? 248 + (slot + token) % topk : 248 + (topk - 1 - slot + token) % topk;
                }
            }
            const auto before = pool.counts();
            const auto resident_before = pool.residents();
            ok = pool.acquire({-1}, remapped) == status::invalid && remapped.empty() && ok;
            ok = pool.acquire({256}, remapped) == status::invalid && remapped.empty() && ok;
            ok = pool.acquire({}, remapped) == status::invalid && remapped.empty() && ok;
            const auto state = pool.acquire(ids, remapped, topk);
            if (state == status::ready) {
                ++ready;
                ok = remapped.size() == ids.size() && ok;
                for (size_t n = 0; n < remapped.size(); ++n) {
                    ok = remapped[n] >= 0 && size_t(remapped[n]) < pool.residents().size() && pool.residents()[remapped[n]] == ids[n] && ok;
                }
                if (pass == 1 || pass == 7) {
                    ok = pool.counts().upload_bytes == before.upload_bytes && pool.counts().misses == before.misses && ok;
                }
                ok = cached_run.submit(target, remapped) && ok;
                const auto pending_residents = pool.residents();
                const auto pending_counts = pool.counts();
                std::vector<int32_t> rejected{99};
                ok = pool.acquire({42}, rejected) == status::busy && rejected.empty() &&
                     pool.residents() == pending_residents && same_counts(pool.counts(), pending_counts) && ok;
                pool.release();
                const auto actual = cached_run.read(target);
                ok = target_run.submit(target, ids) && ok;
                const auto reference = target_run.read(target);
                ok = identical(actual, reference) && ok;
                // Dyadic F32 inputs give an independent exact oracle for gate and up.
                if (quant == 0) {
                    for (int i = 0; i < 2; ++i) {
                        for (int token = 0; token < tokens; ++token) {
                            for (int slot = 0; slot < topk; ++slot) {
                                for (int r = 0; r < rows; ++r) {
                                    float expected = 0;
                                    for (int col = 0; col < width; ++col) {
                                        expected += source_value(i, ids[token * topk + slot], r, col) * activation_value(token, broadcast ? 0 : slot, col);
                                    }
                                    ok = actual[i][(token * topk + slot) * rows + r] == expected && ok;
                                }
                            }
                        }
                    }
                }
                if (output) {
                    for (const auto & projection : actual) {
                        ok = fwrite(projection.data(), sizeof(float), projection.size(), output) == projection.size() && ok;
                    }
                }
            } else if (state == status::capacity || state == status::routing) {
                ++bypass;
                ok = remapped.empty() && pool.residents() == resident_before && same_counts(pool.counts(), before) && ok;
                ok = fallback_run.submit(cpu, ids) && cpu_run.submit(cpu, ids) && ok;
                const auto actual = fallback_run.read(cpu), reference = cpu_run.read(cpu);
                ok = identical(actual, reference) && ok;
                if (output) {
                    for (const auto & projection : actual) {
                        ok = fwrite(projection.data(), sizeof(float), projection.size(), output) == projection.size() && ok;
                    }
                }
            } else {
                ok = false;
            }
            pool.release();
            if (!ok) {
                fprintf(stderr, "pool case failed quant=%d tokens=%d broadcast=%d strided=%d pass=%d\n", quant, tokens, broadcast, strided, pass);
                break;
            }
        }
        misses = pool.counts().misses;
        hits = pool.counts().hits;
        evictions = pool.counts().evictions;
        uploads = pool.counts().upload_bytes;
        const size_t per_slot = source[0]->nb[2] + source[1]->nb[2] + source[2]->nb[2];
        ok = uploads == misses * per_slot && payload == 12 * per_slot && allocated >= payload && ok;
        ok = misses == 20 && hits == uint64_t(tokens == 1 ? 28 : 20) && evictions == 8 && ready == (tokens == 1 ? 6 : 5) && bypass == (tokens == 1 ? 2 : 3) && ok;
    }
    printf("pool quant=%d tokens=%d broadcast=%d strided=%d ready=%d bypass=%d hits=%llu misses=%llu evictions=%llu upload_bytes=%llu payload=%zu allocated=%zu %s\n",
        quant, tokens, broadcast, strided, ready, bypass, (unsigned long long) hits, (unsigned long long) misses,
        (unsigned long long) evictions, (unsigned long long) uploads, payload, allocated, ok ? "OK" : "FAIL");
    return ok;
}
bool check_management(ggml_backend_t target, ggml_backend_t cpu) {
    tensor_store store;
    if (!store.ctx) {
        return false;
    }
    std::array<ggml_tensor *, 3> a{}, b{};
    for (int i = 0; i < 3; ++i) {
        a[i] = ggml_new_tensor_3d(store.ctx, GGML_TYPE_F32, 32, 8, 8);
        b[i] = ggml_new_tensor_3d(store.ctx, GGML_TYPE_F32, 32, 8, 8);
        ggml_format_name(a[i], "blk.0.projection%d", i);
        ggml_format_name(b[i], "blk.1.projection%d", i);
    }
    store.buffer = ggml_backend_alloc_ctx_tensors(store.ctx, cpu);
    if (!store.buffer) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        for (int expert = 0; expert < 8; ++expert) {
            std::vector<float> data(32 * 8, float(i * 16 + expert + 1));
            ggml_backend_tensor_set(a[i], data.data(), expert * a[i]->nb[2], a[i]->nb[2]);
            for (float & value : data) {
                value += 100;
            }
            ggml_backend_tensor_set(b[i], data.data(), expert * b[i]->nb[2], b[i]->nb[2]);
        }
    }
    using status = moe_expert_pool::status;
    bool ok = true;
    moe_expert_pool allocation_retry;
    ok = allocation_retry.init(target,a,4,65536,true) && ok;
    const auto retry_residents = allocation_retry.residents();
    const auto retry_counts = allocation_retry.counts();
    std::vector<int32_t> retry_ids;
    {
        fail_allocation fault(target);
        ok = allocation_retry.acquire({1},retry_ids) == status::unavailable && retry_ids.empty() && fault.calls == 1 &&
                allocation_retry.allocated_bytes() == 0 && allocation_retry.residents() == retry_residents &&
                same_counts(allocation_retry.counts(),retry_counts) && ok;
    }
    ok = allocation_retry.acquire({1},retry_ids) == status::ready && retry_ids == std::vector<int32_t>{0} && ok;
    allocation_retry.release();
    for (int i = 0; i < 3; ++i) {
        std::vector<float> actual(32*8);
        ggml_backend_tensor_get(allocation_retry.weight(i),actual.data(),0,a[i]->nb[2]);
        ok = std::all_of(actual.begin(),actual.end(),[i](float value) { return value == float(i*16+2); }) && ok;
    }
    printf("pool actual_allocator_failure=1 unchanged_state=1 retry=1 raw_payload_exact=1 %s\n",ok ? "OK" : "FAIL");
    for (int misses = 0; misses <= 4; ++misses) {
        moe_expert_pool mixed;
        std::vector<int32_t> mapped, requested;
        ok = mixed.init(target,a,4,65536) && mixed.acquire({0,1,2,3},mapped,4) == status::ready && ok;
        mixed.release();
        const auto before = mixed.counts();
        for (int i = 0; i < 4-misses; ++i) { requested.push_back(i); }
        for (int i = 0; i < misses; ++i) { requested.push_back(4+i); }
        ok = mixed.acquire(requested,mapped,4) == status::ready && ok;
        mixed.release();
        const auto after = mixed.counts();
        ok = after.misses-before.misses == uint64_t(misses) && after.hits-before.hits == uint64_t(4-misses) &&
                after.evictions-before.evictions == uint64_t(misses) &&
                after.upload_bytes-before.upload_bytes == uint64_t(misses)*3*a[0]->nb[2] && ok;
        for (int i = 0; i < 3; ++i) {
            for (size_t rank = 0; rank < requested.size(); ++rank) {
                std::vector<float> actual(32*8);
                ggml_backend_tensor_get(mixed.weight(i),actual.data(),mapped[rank]*a[i]->nb[2],a[i]->nb[2]);
                ok = std::all_of(actual.begin(),actual.end(),[&](float value) { return value == float(i*16+requested[rank]+1); }) && ok;
            }
        }
        printf("pool controlled_miss_percent=%d hits=%llu misses=%llu evictions=%llu raw_payload_exact=1 %s\n",misses*25,
                (unsigned long long)(after.hits-before.hits),(unsigned long long)(after.misses-before.misses),
                (unsigned long long)(after.evictions-before.evictions),ok ? "OK" : "FAIL");
    }
    for (int capacity : {1, 2, 4, 8}) {
        moe_expert_pool first, second;
        ok = first.init(target, a, capacity, 65536) && second.init(target, b, capacity, 65536) && ok;
        if (!ok) {
            return false;
        }
        std::vector<int32_t> remapped;
        ok = first.acquire({7, 7}, remapped) == status::ready && remapped == std::vector<int32_t>({0, 0}) && ok;
        first.release();
        ok = second.acquire({7}, remapped) == status::ready && ok;
        second.release();
        for (int i = 0; i < 3; ++i) {
            std::vector<float> one(32 * 8), two(32 * 8);
            ggml_backend_tensor_get(first.weight(i), one.data(), 0, one.size() * sizeof(float));
            ggml_backend_tensor_get(second.weight(i), two.data(), 0, two.size() * sizeof(float));
            ok = std::all_of(one.begin(), one.end(), [i](float value) { return value == float(i * 16 + 8); }) &&
                 std::all_of(two.begin(), two.end(), [i](float value) { return value == float(i * 16 + 108); }) && ok;
        }
        // Capacity below top-k must bypass without changing any slot.
        const auto before = first.counts();
        const auto resident_before = first.residents();
        std::vector<int32_t> overflow(capacity + 1);
        for (int i = 0; i <= capacity; ++i) {
            overflow[i] = i % 8;
        }
        if (capacity < 8) {
            ok = first.acquire(overflow, remapped) == status::capacity && remapped.empty() &&
                 first.residents() == resident_before && same_counts(first.counts(), before) && ok;
        }
        ok = first.acquire({7}, remapped) == status::ready && ok;
        first.release();
        // Cancel before submission, then retry. No reload of an admitted expert.
        ok = first.acquire({7}, remapped) == status::ready && ok;
        first.release();
        ok = first.counts().misses == 1 && first.counts().hits == 2 && second.counts().misses == 1 && ok;
    }
    moe_expert_pool lazy;
    ok = lazy.init(target, a, 4, 65536, true) && lazy.allocated_bytes() == 0 &&
         lazy.reserved_bytes() >= lazy.payload_bytes() && lazy.weight(0) == nullptr && ok;
    std::vector<int32_t> lazy_ids;
    const auto empty_counts = lazy.counts();
    const auto empty_residents = lazy.residents();
    for (const auto & request : std::vector<std::vector<int32_t>>{{0, 1, 2, 3, 4}, {1, 1}, {-1, 0}}) {
        const auto expected = request.size() == 5 ? status::capacity : request[0] < 0 ? status::invalid : status::routing;
        ok = lazy.request_status(request, request.size() == 5 ? 1 : 2) == expected &&
             lazy.acquire(request, lazy_ids, request.size() == 5 ? 1 : 2) == expected && lazy_ids.empty() &&
             lazy.allocated_bytes() == 0 && lazy.residents() == empty_residents && same_counts(lazy.counts(), empty_counts) && ok;
    }
    ok = lazy.acquire({7}, lazy_ids) == status::ready && lazy_ids == std::vector<int32_t>({0}) &&
         lazy.allocated_bytes() >= lazy.payload_bytes() && ok;
    lazy.release();
    for (int i = 0; i < 3 && ok; ++i) {
        std::vector<float> data(32 * 8 * 4);
        ggml_backend_tensor_get(lazy.weight(i), data.data(), 0, data.size() * sizeof(float));
        ok = std::all_of(data.begin(), data.begin() + 32 * 8, [i](float value) { return value == float(i * 16 + 8); }) &&
             std::all_of(data.begin() + 32 * 8, data.end(), [](float value) { return value == 0; }) && ok;
    }
    const auto lazy_bytes = lazy.allocated_bytes();
    ok = lazy.acquire({7}, lazy_ids) == status::ready && lazy.allocated_bytes() == lazy_bytes &&
         lazy.counts().hits == 1 && lazy.counts().misses == 1 && ok;
    lazy.release();
    printf("pool lazy capacity/routing/invalid_no_alloc=1 first_admission=1 zero_padding=1 all_hit_reuse=1 %s\n", ok ? "OK" : "FAIL");
    moe_expert_pool lru;
    ok = lru.init(target, a, 2, 65536) && ok;
    std::vector<int32_t> remapped;
    ok = lru.acquire({1, 0}, remapped) == status::ready && remapped == std::vector<int32_t>({1, 0}) && ok;
    lru.release();
    ok = lru.acquire({0}, remapped) == status::ready && ok;
    lru.release();
    ok = lru.acquire({7, 0}, remapped) == status::ready && remapped == std::vector<int32_t>({1, 0}) && ok;
    lru.release();
    ok = lru.acquire({6}, remapped) == status::ready && remapped == std::vector<int32_t>({0}) && ok;
    lru.release();
    ok = lru.residents() == std::vector<int32_t>({6, 7}) && lru.counts().misses == 4 && lru.counts().evictions == 2 && ok;
    moe_expert_pool invalid_geometry;
    auto bad_weight = *a[2];
    bad_weight.ne[2] = 7;
    auto bad_source = a;
    bad_source[2] = &bad_weight;
    ok = !invalid_geometry.init(target, bad_source, 2, 65536) && ok;
    ok = lru.acquire({0}, remapped, 0) == status::invalid && remapped.empty() && ok;
    ok = lru.acquire({0, 1, 2}, remapped, 2) == status::invalid && remapped.empty() && ok;
    // The graph and backend outlive the pool. Its destructor must wait for submitted work.
    tensor_store pending;
    if (!pending.ctx) {
        return false;
    }
    auto * input = ggml_new_tensor_3d(pending.ctx, GGML_TYPE_F32, 32, 1, 1);
    auto * ids = ggml_new_tensor_2d(pending.ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * result = nullptr;
    {
        moe_expert_pool temporary;
        ok = temporary.init(target, a, 1, 65536) && temporary.acquire({6}, remapped) == status::ready && ok;
        if (!ok) {
            return false;
        }
        result = ggml_mul_mat_id(pending.ctx, temporary.weight(0), input, ids);
        auto * graph = ggml_new_graph_custom(pending.ctx, 16, false);
        ggml_build_forward_expand(graph, result);
        pending.buffer = ggml_backend_alloc_ctx_tensors(pending.ctx, target);
        if (!pending.buffer) {
            return false;
        }
        std::vector<float> ones(32, 1);
        ggml_backend_tensor_set(input, ones.data(), 0, 32 * sizeof(float));
        ggml_backend_tensor_set(ids, remapped.data(), 0, sizeof(int32_t));
        ok = ggml_backend_graph_compute_async(target, graph) == GGML_STATUS_SUCCESS && ok;
    }
    std::vector<float> completed(8);
    ggml_backend_tensor_get(result, completed.data(), 0, completed.size() * sizeof(float));
    ok = std::all_of(completed.begin(), completed.end(), [](float value) { return value == 224; }) && ok;
    printf("pool management capacity=1,2,4,8 layer_isolation=1 lru=1 cancel_retry=1 destructor_wait=1 geometry=1 %s\n", ok ? "OK" : "FAIL");
    return ok;
}
} // namespace

bool check_multilayer_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output) {
    saved_env pool_env("GGML_SCHED_EXPERT_POOL"), place_env("GGML_SCHED_EXPERT_GPU_LAYER");
    bool ok = true;
    for (int count : {2, 4}) {
      for (int quant : {0, 1}) {
        tensor_store source_store, full_store;
        std::vector<std::array<ggml_tensor *, 3>> source(count), full(count);
        const char * names[] = {"gate", "up", "down"};
        std::string config, layers;
        for (int layer = 0; layer < count; ++layer) {
            config += (layer ? ";" : "") + std::to_string(layer+3) + ":12:16";
            layers += (layer ? "," : "") + std::to_string(layer+3);
            for (int i = 0; i < 3; ++i) {
                const auto type = quant == 0 ? GGML_TYPE_F32 : i == 2 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
                source[layer][i] = ggml_new_tensor_3d(source_store.ctx, type, width, i == 2 ? outputs : rows, experts);
                full[layer][i] = ggml_new_tensor_3d(full_store.ctx, type, width, i == 2 ? outputs : rows, experts);
                ggml_format_name(source[layer][i], "blk.%d.ffn_%s_exps.weight", layer+3, names[i]);
                ggml_format_name(full[layer][i], "blk.%d.ffn_%s_exps.weight", layer+3, names[i]);
            }
        }
        source_store.buffer = ggml_backend_alloc_ctx_tensors(source_store.ctx, cpu);
        full_store.buffer = ggml_backend_alloc_ctx_tensors(full_store.ctx, target);
        if (!source_store.buffer || !full_store.buffer) { return false; }
        ggml_backend_buffer_set_usage(source_store.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int layer = 0; layer < count; ++layer) {
            for (int i = 0; i < 3; ++i) {
                auto * weight = source[layer][i];
                std::vector<uint8_t> packed(ggml_nbytes(weight));
                std::vector<float> row(width);
                for (int expert = 0; expert < experts; ++expert) {
                    for (int r = 0; r < weight->ne[1]; ++r) {
                        for (int col = 0; col < width; ++col) { row[col] = source_value(layer*3+i, expert, r, col); }
                        auto * destination = packed.data()+expert*weight->nb[2]+r*weight->nb[1];
                        if (weight->type == GGML_TYPE_F32) { memcpy(destination,row.data(),width*sizeof(float)); }
                        else { ggml_get_type_traits(weight->type)->from_float_ref(row.data(),destination,width); }
                    }
                }
                ggml_backend_tensor_set(weight,packed.data(),0,packed.size());
                ggml_backend_tensor_set(full[layer][i],packed.data(),0,packed.size());
            }
        }
        std::unique_ptr<graph_run> previous;
        for (int tokens : {1, 3, 33, 3, 1}) {
            unsetenv(pool_env.key);
            unsetenv(place_env.key);
            graph_run off;
            auto current = std::make_unique<graph_run>();
            auto & on = *current;
            ggml_backend_sched_t reuse = previous ? previous->sched : nullptr;
            if (previous) { previous->sched = nullptr; }
            if (!off.init(target,full.front(),tokens,false,true,cpu,false,false,false,true,&full)) { return false; }
            setenv(pool_env.key,config.c_str(),1);
            setenv(place_env.key,layers.c_str(),1);
            if (!on.init(target,source.front(),tokens,false,true,cpu,false,false,true,true,&source,reuse)) { return false; }
            previous.reset();
            for (const auto & layer : on.layer_results) {
                for (auto * tensor : layer) {
                    auto * actual_backend = ggml_backend_sched_get_tensor_backend(on.sched,tensor);
                    if (actual_backend != target) { fprintf(stderr,"multilayer placement: %s backend=%s expected=%s\n",tensor->name,ggml_backend_name(actual_backend),ggml_backend_name(target)); }
                    ok = actual_backend == target && ok;
                }
            }
            for (int pass = 0; pass < 6; ++pass) {
                std::vector<int32_t> ids(tokens*topk);
                for (int token = 0; token < tokens; ++token) {
                    for (int rank = 0; rank < topk; ++rank) {
                        ids[token*topk+rank] = pass == 0 ? rank : pass == 1 ? 7-rank : pass == 2 ? rank+4 :
                                pass == 3 ? (token*topk+rank)%experts : pass == 4 ? 248+rank : 255-rank;
                    }
                }
                const bool off_submitted = off.submit(target,ids,pass), on_submitted = on.submit(target,ids,pass);
                if (!off_submitted || !on_submitted) { fprintf(stderr,"multilayer submit: off=%d on=%d\n",off_submitted,on_submitted); }
                ok = off_submitted && on_submitted && ok;
                ggml_backend_synchronize(target);
                for (int layer = 0; layer < count; ++layer) {
                    values actual, reference;
                    for (int i = 0; i < 3; ++i) {
                        auto * tensor = on.layer_results[layer][i];
                        actual[i].resize(ggml_nelements(tensor));
                        reference[i].resize(actual[i].size());
                        ggml_backend_tensor_get(tensor,actual[i].data(),0,ggml_nbytes(tensor));
                        ggml_backend_tensor_get(off.layer_results[layer][i],reference[i].data(),0,ggml_nbytes(tensor));
                        if (output) { ok = fwrite(actual[i].data(),sizeof(float),actual[i].size(),output) == actual[i].size() && ok; }
                    }
                    const bool same = identical(actual,reference);
                    if (!same) {
                        for (int projection=0;projection<3;++projection) {
                            double delta=0;size_t different=0;
                            for (size_t v=0;v<actual[projection].size();++v) { delta=std::max(delta,std::abs(double(actual[projection][v])-reference[projection][v]));different+=memcmp(&actual[projection][v],&reference[projection][v],sizeof(float))!=0; }
                            fprintf(stderr,"multilayer raw: count=%d quant=%d tokens=%d pass=%d layer=%d projection=%d diff=%zu maxabs=%g valid=%d/%d\n",count,quant,tokens,pass,layer,projection,different,delta,valid(actual),valid(reference));
                        }
                    }
                    ok = same && ok;
                }
                std::vector<int32_t> stored(ggml_nelements(on.storage));
                ggml_backend_tensor_get(on.storage,stored.data(),0,ggml_nbytes(on.storage));
                for (int token = 0; token < tokens; ++token) {
                    for (int rank = 0; rank < topk; ++rank) {
                        if (stored[token*16+rank+1] != ids[token*topk+rank]) { fprintf(stderr,"multilayer source ids: tokens=%d pass=%d token=%d rank=%d actual=%d expected=%d\n",tokens,pass,token,rank,stored[token*16+rank+1],ids[token*topk+rank]); }
                        ok = stored[token*16+rank+1] == ids[token*topk+rank] && ok;
                    }
                }
            }
            if (!ggml_backend_is_cpu(target)) {
                std::vector<int32_t> ids(tokens*topk);
                for (size_t i = 0; i < ids.size(); ++i) { ids[i] = i%topk; }
                ok = off.submit(target,ids,6) && ok;
                std::vector<std::array<ggml_tensor *,GGML_MAX_SRC>> original_sources;
                for (int i = 0; i < ggml_graph_n_nodes(on.graph); ++i) {
                    std::array<ggml_tensor *,GGML_MAX_SRC> sources;
                    std::copy_n(ggml_graph_node(on.graph,i)->src,GGML_MAX_SRC,sources.begin());
                    original_sources.push_back(sources);
                }
                {
                    abort_pool_graph fault(target);
                    const bool submitted = on.submit(target,ids,6);
                    fprintf(stderr,"abort fixture: submitted=%d calls=%d prior_ok=%d\n",submitted,fault.calls,ok);
                    ok = !submitted && fault.calls == 1 && ok;
                }
                for (int i = 0; i < ggml_graph_n_nodes(on.graph); ++i) {
                    const bool restored = std::equal(original_sources[i].begin(),original_sources[i].end(),ggml_graph_node(on.graph,i)->src);
                    if (!restored) { fprintf(stderr,"abort fixture: sources changed node=%d name=%s\n",i,ggml_graph_node(on.graph,i)->name); }
                    ok = restored && ok;
                }
                ok = on.submit(target,ids,6) && ok;
                ggml_backend_synchronize(target);
                for (int layer = 0; layer < count; ++layer) {
                    values actual,reference;
                    for (int i = 0; i < 3; ++i) {
                        auto * tensor = on.layer_results[layer][i];
                        actual[i].resize(ggml_nelements(tensor));reference[i].resize(actual[i].size());
                        ggml_backend_tensor_get(tensor,actual[i].data(),0,ggml_nbytes(tensor));
                        ggml_backend_tensor_get(off.layer_results[layer][i],reference[i].data(),0,ggml_nbytes(tensor));
                    }
                    ok = identical(actual,reference) && ok;
                }
                printf("pool abort_after_admission=1 graph_sources_restored=1 retry_raw_exact=1 %s\n",ok ? "OK" : "FAIL");
            }
            printf("multilayer pool layers=%d quant=%d tokens=%d changed_input=1 strided=1 resize_reused_scheduler=1 raw_exact=1 %s\n",count,quant,tokens,ok ? "OK" : "FAIL");
            previous = std::move(current);
        }
      }
    }
    return ok;
}

bool check_expert_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output) {
    bool ok = check_management(target, cpu);
    for (int quant : {0, 1, 2, 3}) {
        for (int tokens : {1, 3, 33}) {
            for (bool broadcast : {false, true}) {
                for (bool strided : {false, true}) {
                    ok = check_pool_case(target, cpu, quant, tokens, broadcast, strided, output) && ok;
                }
            }
        }
    }
    return ok;
}

bool check_scheduler_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output) {
    if (getenv("MOE_SCHED_POOL_FOCUS")) {
        return check_pool_case(target, cpu, 1, 3, false, true, output, "3:12:16");
    }
    bool ok = true;
    for (int quant : {0, 1, 2, 3}) {
        for (int tokens : {1, 33}) {
            for (bool broadcast : {false, true}) {
                ok = check_pool_case(target, cpu, quant, tokens, broadcast, true, output, "3:12:16") && ok;
            }
        }
    }
    for (const char * config : {"3:4:16", "3:12:1", "3:12:0", "3:12:16:1", "3:12:16;", ";3:12:16", "3:12:16;3:12:16", "3:12:16;4:12:2048", "3:12:16;4:12:0", "3:12:16;4:12:16"}) {
        ok = check_pool_case(target, cpu, 0, 1, false, false, output, config) && ok;
    }
    ok = check_pool_case(target, cpu, 1, 3, false, false, output, "3:12:16", true) && ok;
    ok = check_pool_case(target, cpu, 1, 3, false, false, output, "3:12:16", false, true) && ok;
    ok = check_pool_case(target, cpu, 0, 1, false, false, output, "3:128:65") && ok;
    return ok;
}

bool check_expert_placement(ggml_backend_t target, ggml_backend_t cpu, FILE * output) {
    bool ok = true;
    for (int quant : {0, 1, 2, 3}) {
        for (int tokens : {1, 33}) {
            for (bool broadcast : {false, true}) {
                ok = check_pool_case(target, cpu, quant, tokens, broadcast, true, output, "3:12:16", false, false, true) && ok;
            }
        }
    }
    saved_env pool_env("GGML_SCHED_EXPERT_POOL"), place_env("GGML_SCHED_EXPERT_GPU_LAYER");
    unsetenv(pool_env.key);
    tensor_store store;
    std::array<ggml_tensor *, 3> weights{};
    for (int i = 0; i < 3; ++i) {
        weights[i] = ggml_new_tensor_3d(store.ctx, GGML_TYPE_F32, width, i == 2 ? outputs : rows, experts);
    }
    store.buffer = ggml_backend_alloc_ctx_tensors(store.ctx, cpu);
    if (!store.buffer) {
        return false;
    }
    ggml_backend_buffer_set_usage(store.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int i = 0; i < 3; ++i) {
        std::vector<float> data(ggml_nelements(weights[i]));
        for (int e = 0; e < experts; ++e) {
            for (int r = 0; r < weights[i]->ne[1]; ++r) {
                for (int c = 0; c < width; ++c) {
                    data[(e * weights[i]->ne[1] + r) * width + c] = source_value(i, e, r, c);
                }
            }
        }
        ggml_backend_tensor_set(weights[i], data.data(), 0, ggml_nbytes(weights[i]));
    }
    for (const char * config : {"3", "", "-1", "3x", "4097", "999999999999999", "0", "4", "3,4", "4,3", "3,", ",3", "3,3", "3,4097"}) {
        for (bool shared : {false, true}) {
            const char * names[] = {"gate", "up", "down"};
            for (int i = 0; i < 3; ++i) {
                ggml_format_name(weights[i], "blk.3.ffn_%s_%s.weight", names[i], shared ? "shexp" : "exps");
            }
            setenv(place_env.key, config, 1);
            graph_run actual, reference;
            const bool selected = (strcmp(config, "3") == 0 || strcmp(config, "3,4") == 0 || strcmp(config, "4,3") == 0) && !shared;
            auto * expected = selected ? target : cpu;
            if (!actual.init(target, weights, 1, false, true, cpu, false, false, true) ||
                    !reference.init(expected, weights, 1, false, true, cpu)) {
                return false;
            }
            for (auto * node : actual.result) {
                ok = ggml_backend_sched_get_tensor_backend(actual.sched, node) == expected && ok;
            }
            std::vector<int32_t> ids{0, 1, 2, 3, 4, 5, 6, 7};
            ok = actual.submit(target, ids) && reference.submit(expected, ids) && ok;
            ok = identical(actual.read(target), reference.read(expected)) && ok;
            printf("expert placement config='%s' shared=%d backend=%s %s\n", config, shared,
                    ggml_backend_name(expected), ok ? "OK" : "FAIL");
        }
    }
    const char * names[] = {"gate", "up", "down"};
    for (int i = 0; i < 3; ++i) {
        ggml_format_name(weights[i], "blk.3.ffn_%s_exps.weight", names[i]);
    }
    setenv(place_env.key, "3", 1);
    graph_run disabled, reference;
    if (!disabled.init(target, weights, 1, false, true, cpu, false, false, true, false) ||
            !reference.init(cpu, weights, 1, false, true, cpu)) {
        return false;
    }
    for (auto * node : disabled.result) {
        ok = ggml_backend_sched_get_tensor_backend(disabled.sched, node) == cpu && ok;
    }
    std::vector<int32_t> ids{0, 1, 2, 3, 4, 5, 6, 7};
    ok = disabled.submit(target, ids) && reference.submit(cpu, ids) && ok;
    ok = identical(disabled.read(target), reference.read(cpu)) && ok;
    printf("expert placement op_offload=0 backend=CPU %s\n", ok ? "OK" : "FAIL");
    return ok;
}

namespace {
float numerics_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int32_t(state & 0xffff) - 32768) / 32768.0f;
}

bool numerics_dump(const char * prefix, const std::string & suffix, const void * data, size_t bytes) {
    if (!prefix || !*prefix) {
        return true;
    }
    FILE * file = fopen((std::string(prefix) + suffix).c_str(), "wb");
    if (!file) {
        return false;
    }
    const bool ok = fwrite(data, 1, bytes, file) == bytes;
    return fclose(file) == 0 && ok;
}

bool numerics_load(const char * prefix, const std::string & suffix, void * data, size_t bytes) {
    FILE * file = fopen((std::string(prefix) + suffix).c_str(), "rb");
    if (!file) {
        fprintf(stderr, "cannot read operator input %s\n", suffix.c_str());
        return false;
    }
    const bool ok = fread(data, 1, bytes, file) == bytes && fgetc(file) == EOF && !ferror(file);
    return fclose(file) == 0 && ok;
}

bool operator_cost(ggml_backend_t target, ggml_tensor * weights, const void * input_data, ggml_type input_type,
        const int32_t * route, int slots, const float * full_result, const char * name, int pass, FILE * cost) {
    if (!cost) { return true; }
    const bool cold_weights = getenv("MOE_OPERATOR_COLD_WEIGHTS") != nullptr;
    if (cold_weights) {
        if (!ggml_backend_is_cpu(target) || !ggml_backend_buffer_is_host(weights->buffer)) { return false; }
#if !defined(__x86_64__) && !defined(__i386__)
        fprintf(stderr,"cold weight timing requires x86 cache-line flush support\n");
        return false;
#endif
    }
    for (int selected : {0, 2, 4, 6, 8}) {
        tensor_store partial;
        ggml_tensor * result = nullptr;
        ggml_cgraph * graph = nullptr;
        if (selected) {
            auto * input = ggml_new_tensor_3d(partial.ctx,input_type,weights->ne[0],slots == 1 ? 1 : selected,1);
            auto * ids = ggml_new_tensor_2d(partial.ctx,GGML_TYPE_I32,selected,1);
            result = ggml_mul_mat_id(partial.ctx,weights,input,ids);
            graph = ggml_new_graph_custom(partial.ctx,16,false);
            ggml_build_forward_expand(graph,result);
            if (!ggml_backend_supports_op(target,result)) { return false; }
            partial.buffer = ggml_backend_alloc_ctx_tensors(partial.ctx,target);
            if (!partial.buffer) { return false; }
            ggml_backend_tensor_set(input,input_data,0,ggml_nbytes(input));
            ggml_backend_tensor_set(ids,route,0,selected*sizeof(int32_t));
            if (ggml_backend_graph_compute(target,graph) != GGML_STATUS_SUCCESS) { return false; }
            ggml_backend_synchronize(target);
        }
        std::vector<float> reference(selected*weights->ne[1]), actual(reference.size());
        if (selected) {
            ggml_backend_tensor_get(result,reference.data(),0,ggml_nbytes(result));
            if (memcmp(reference.data(),full_result,reference.size()*sizeof(float))) {
                fprintf(stderr,"operator cost subset differs %s pass=%d selected=%d\n",name,pass,selected);
                return false;
            }
        }
        for (int rep = 0; rep < 30; ++rep) {
#if defined(__x86_64__) || defined(__i386__)
            if (cold_weights) {
                ggml_backend_synchronize(target);
                for (int expert = 0; expert < selected; ++expert) {
                    if (route[expert] < 0 || route[expert] >= weights->ne[2]) { return false; }
                    const char * data = static_cast<const char *>(weights->data) + route[expert]*weights->nb[2];
                    for (size_t offset = 0; offset < weights->nb[2]; offset += 64) { _mm_clflush(data+offset); }
                }
                _mm_mfence();
            }
#endif
            const int64_t start = ggml_time_us();
            const auto status = selected ? ggml_backend_graph_compute_async(target,graph) : GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(target);
            const int64_t elapsed = ggml_time_us()-start;
            if (status != GGML_STATUS_SUCCESS) { return false; }
            if (selected) {
                ggml_backend_tensor_get(result,actual.data(),0,ggml_nbytes(result));
                if (actual != reference || std::any_of(actual.begin(),actual.end(),[](float v) { return !std::isfinite(v); }) ||
                        std::all_of(actual.begin(),actual.end(),[](float v) { return v == 0; })) { return false; }
            }
            if (fprintf(cost,"%s,%d,%s,%s,%d,%d,%lld\n",name,pass,ggml_backend_name(target),ggml_type_name(input_type),
                    selected,rep,(long long)elapsed) < 0) { return false; }
        }
    }
    return true;
}
}

bool check_operator_numerics(ggml_backend_t target, FILE * output) {
    const char * option = getenv("MOE_OPERATOR_INPUT_Q8_K");
    if (option && *option && strcmp(option, "0") != 0 && strcmp(option, "1") != 0) {
        fprintf(stderr, "MOE_OPERATOR_INPUT_Q8_K must be 0 or 1\n");
        return false;
    }
    const bool rounded_option = option && strcmp(option, "1") == 0;
    const char * captured_option = getenv("MOE_OPERATOR_CAPTURED_Q8_K");
    if (captured_option && *captured_option && strcmp(captured_option, "0") != 0 && strcmp(captured_option, "1") != 0) {
        fprintf(stderr, "MOE_OPERATOR_CAPTURED_Q8_K must be 0 or 1\n");
        return false;
    }
    const bool captured = captured_option && strcmp(captured_option, "1") == 0;
    const bool rounded = rounded_option || captured;
    const char * prefix = getenv("MOE_OPERATOR_DATA_PREFIX");
    const char * input_prefix = getenv("MOE_OPERATOR_INPUT_PREFIX");
    const bool external = input_prefix && *input_prefix;
    if (captured && (!external || (option && strcmp(option, "0") == 0))) {
        fprintf(stderr, "captured Q8_K needs external inputs and cannot use MOE_OPERATOR_INPUT_Q8_K=0\n");
        return false;
    }
    const bool direct_q8k = captured && ggml_backend_is_cpu(target);
    const auto * weights_trait = ggml_get_type_traits(GGML_TYPE_Q4_K);
    const auto convert = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    if (!convert || !weights_trait->from_float_ref || !weights_trait->to_float) {
        return false;
    }
    using file_ptr = std::unique_ptr<FILE, int (*)(FILE *)>;
    file_ptr cost(nullptr,fclose);
    if (const char * path = getenv("MOE_OPERATOR_COST_OUT")) {
        cost.reset(fopen(path,"wx"));
        if (!cost || fprintf(cost.get(),"projection,pass,backend,input_type,selected_experts,repetition,compute_sync_us\n") < 0) { return false; }
    }
    const char * oracle_prefix = getenv("MOE_OPERATOR_ORACLE_PREFIX");
    file_ptr fp32(nullptr, fclose), q8k(nullptr, fclose);
    if (oracle_prefix && *oracle_prefix) {
        fp32.reset(fopen((std::string(oracle_prefix) + "-fp32.bin").c_str(), "wb"));
        q8k.reset(fopen((std::string(oracle_prefix) + "-q8k.bin").c_str(), "wb"));
        if (!fp32 || !q8k) {
            return false;
        }
    }
    printf("operator,pass,k,m,input_slots,backend,rounded_input,fp32_max_abs,fp32_RMS,q8k_max_abs,q8k_RMS\n");
    const int selected[8] = {0, 1, 7, 31, 63, 127, 191, 255};
    for (int projection = 0; projection < 3; ++projection) {
        const char * name = projection == 0 ? "gate" : projection == 1 ? "up" : "down";
        const int k = projection == 2 ? 512 : 2048;
        const int m = projection == 2 ? 2048 : 512;
        const int slots = projection == 2 ? 8 : 1;
        tensor_store store;
        if (!store.ctx) {
            return false;
        }
        auto * weights = ggml_new_tensor_3d(store.ctx, GGML_TYPE_Q4_K, k, m, 256);
        auto * input = ggml_new_tensor_3d(store.ctx, direct_q8k ? GGML_TYPE_Q8_K : GGML_TYPE_F32, k, slots, 1);
        auto * ids = ggml_new_tensor_2d(store.ctx, GGML_TYPE_I32, 8, 1);
        auto * result = ggml_mul_mat_id(store.ctx, weights, input, ids);
        ggml_format_name(weights, "blk.17.ffn_%s_exps.weight", name);
        ggml_format_name(result, "operator-%s", name);
        auto * graph = ggml_new_graph_custom(store.ctx, 16, false);
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_supports_op(target, result)) {
            fprintf(stderr, "operator %s unsupported by %s\n", name, ggml_backend_name(target));
            return false;
        }
        store.buffer = ggml_backend_alloc_ctx_tensors(store.ctx, target);
        if (!store.buffer) {
            return false;
        }
        std::vector<uint8_t> packed(ggml_nbytes(weights), 0);
        std::vector<float> row(k);
        if (external && !numerics_load(input_prefix, std::string("-") + name + "-weights.bin", packed.data(), packed.size())) {
            return false;
        }
        // Synthetic mode populates only the selected experts.
        for (int expert : selected) {
            if (external) {
                break;
            }
            uint32_t state = 0x9e3779b9u + uint32_t(projection * 257 + expert);
            for (int r = 0; r < m; ++r) {
                for (float & value : row) {
                    value = numerics_random(state) / 16.0f;
                }
                weights_trait->from_float_ref(row.data(), packed.data() + expert * weights->nb[2] + r * weights->nb[1], k);
            }
        }
        ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
        if (!numerics_dump(prefix, std::string("-") + name + "-weights.bin", packed.data(), packed.size())) {
            return false;
        }
        fprintf(stderr, "operator numerics begin %s k=%d m=%d input_slots=%d backend=%s rounded=%d captured=%d direct_q8k=%d\n", name, k, m, slots, ggml_backend_name(target), int(rounded), int(captured), int(direct_q8k));
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<float> original(k * slots), converted(k * slots);
            const size_t q8k_row_size = ggml_row_size(GGML_TYPE_Q8_K, k);
            std::vector<uint8_t> quantized(q8k_row_size * slots);
            const std::string tag = std::string("-") + name + "-" + std::to_string(pass);
            uint32_t state = 0x12345678u + uint32_t(projection * 97 + pass * 65537);
            for (float & value : original) {
                value = numerics_random(state);
            }
            if (external && !numerics_load(input_prefix, tag + "-original.bin", original.data(), original.size() * sizeof(float))) {
                return false;
            }
            if (captured && !numerics_load(input_prefix, tag + "-quantized.bin", quantized.data(), quantized.size())) {
                return false;
            }
            if (captured) {
                const auto * blocks = reinterpret_cast<const block_q8_K *>(quantized.data());
                for (size_t i = 0; i < quantized.size() / sizeof(block_q8_K); ++i) {
                    if (!std::isfinite(blocks[i].d)) {
                        return false;
                    }
                    for (int group = 0; group < 16; ++group) {
                        int sum = 0;
                        for (int col = 0; col < 16; ++col) {
                            sum += blocks[i].qs[group * 16 + col];
                        }
                        if (sum != blocks[i].bsums[group]) {
                            return false;
                        }
                    }
                }
            }
            for (int slot = 0; slot < slots; ++slot) {
                bool nonzero = false;
                for (int col = 0; col < k; ++col) {
                    const float value = original[slot * k + col];
                    if (!std::isfinite(value)) {
                        return false;
                    }
                    nonzero = nonzero || value != 0;
                }
                if (!nonzero) {
                    return false;
                }
                auto * quantized_row = quantized.data() + slot * q8k_row_size;
                if (!captured) {
                    convert(original.data() + slot * k, quantized_row, k);
                }
                dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(quantized_row), converted.data() + slot * k, k);
                for (int col = 0; col < k; ++col) {
                    if (!std::isfinite(converted[slot * k + col])) {
                        return false;
                    }
                }
            }
            const auto & actual_input = rounded ? converted : original;
            int32_t id_data[8];
            for (int slot = 0; slot < 8; ++slot) {
                id_data[slot] = selected[(slot + pass) % 8];
            }
            if (external && !numerics_load(input_prefix, tag + "-ids.bin", id_data, sizeof(id_data))) {
                return false;
            }
            for (int id : id_data) {
                if (id < 0 || id >= 256) {
                    return false;
                }
            }
            ggml_backend_tensor_set(input, direct_q8k ? static_cast<const void *>(quantized.data()) : actual_input.data(), 0, ggml_nbytes(input));
            ggml_backend_tensor_set(ids, id_data, 0, sizeof(id_data));
            if (ggml_backend_graph_compute(target, graph) != GGML_STATUS_SUCCESS) {
                return false;
            }
            ggml_backend_synchronize(target);
            std::vector<float> actual(8 * m);
            ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
            if (!operator_cost(target,weights,direct_q8k ? static_cast<const void *>(quantized.data()) : actual_input.data(),
                    input->type,id_data,slots,actual.data(),name,pass,cost.get())) { return false; }
            std::vector<double> ref32(8 * m), ref8(8 * m);
            double max32 = 0, max8 = 0, sum32 = 0, sum8 = 0;
            for (int slot = 0; slot < 8; ++slot) {
                bool nonzero = false, ref_nonzero = false;
                for (int r = 0; r < m; ++r) {
                    weights_trait->to_float(packed.data() + id_data[slot] * weights->nb[2] + r * weights->nb[1], row.data(), k);
                    const int index = slot * m + r;
                    for (int col = 0; col < k; ++col) {
                        if (!std::isfinite(row[col])) {
                            return false;
                        }
                        ref32[index] += double(row[col]) * original[(slot % slots) * k + col];
                        ref8[index] += double(row[col]) * converted[(slot % slots) * k + col];
                    }
                    if (!std::isfinite(actual[index]) || !std::isfinite(ref32[index]) || !std::isfinite(ref8[index])) {
                        return false;
                    }
                    nonzero = nonzero || actual[index] != 0;
                    ref_nonzero = ref_nonzero || (ref32[index] != 0 && ref8[index] != 0);
                    const double d32 = actual[index] - ref32[index], d8 = actual[index] - ref8[index];
                    max32 = std::max(max32, std::abs(d32));
                    max8 = std::max(max8, std::abs(d8));
                    sum32 += d32 * d32;
                    sum8 += d8 * d8;
                }
                if (!nonzero || !ref_nonzero) {
                    return false;
                }
            }
            if (!numerics_dump(prefix, tag + "-original.bin", original.data(), original.size() * sizeof(float)) ||
                !numerics_dump(prefix, tag + "-input.bin", actual_input.data(), actual_input.size() * sizeof(float)) ||
                !numerics_dump(prefix, tag + "-quantized.bin", quantized.data(), quantized.size()) ||
                !numerics_dump(prefix, tag + "-ids.bin", id_data, sizeof(id_data))) {
                return false;
            }
            if ((output && fwrite(actual.data(), sizeof(float), actual.size(), output) != actual.size()) ||
                (fp32 && fwrite(ref32.data(), sizeof(double), ref32.size(), fp32.get()) != ref32.size()) ||
                (q8k && fwrite(ref8.data(), sizeof(double), ref8.size(), q8k.get()) != ref8.size())) {
                return false;
            }
            printf("%s,%d,%d,%d,%d,%s,%d,%.9g,%.9g,%.9g,%.9g\n", name, pass, k, m, slots, ggml_backend_name(target), int(rounded), max32, std::sqrt(sum32 / actual.size()), max8, std::sqrt(sum8 / actual.size()));
        }
        fprintf(stderr, "operator numerics end %s\n", name);
    }
    return (!fp32 || fflush(fp32.get()) == 0) && (!q8k || fflush(q8k.get()) == 0) && (!cost || fclose(cost.release()) == 0);
}

bool check_transfer_cost(ggml_backend_t target) {
    const char * prefix = getenv("MOE_OPERATOR_INPUT_PREFIX");
    const char * path = getenv("MOE_TRANSFER_COST_OUT");
    if (!prefix || !*prefix || !path || ggml_backend_is_cpu(target)) { return false; }
    FILE * cost = fopen(path,"wx");
    if (!cost) { return false; }
    tensor_store store;
    std::array<ggml_tensor *,3> weights{};
    std::array<std::vector<uint8_t>,3> host;
    std::array<size_t,3> stride{};
    int32_t route[8];
    bool ok = numerics_load(prefix,"-gate-0-ids.bin",route,sizeof(route));
    const char * names[] = {"gate","up","down"};
    for (int i = 0; i < 3 && ok; ++i) {
        const int k = i == 2 ? 512 : 2048, m = i == 2 ? 2048 : 512;
        weights[i] = ggml_new_tensor_3d(store.ctx,GGML_TYPE_Q4_K,k,m,9);
        stride[i] = weights[i]->nb[2];
        std::vector<uint8_t> full(stride[i]*256);
        ok = numerics_load(prefix,std::string("-")+names[i]+"-weights.bin",full.data(),full.size());
        host[i].assign(stride[i]*9,0);
        for (int expert = 0; expert < 8 && ok; ++expert) {
            ok = route[expert] >= 0 && route[expert] < 256;
            if (ok) { memcpy(host[i].data()+expert*stride[i],full.data()+route[expert]*stride[i],stride[i]); }
        }
    }
    if (!ok) { fclose(cost); return false; }
    store.buffer = ggml_backend_alloc_ctx_tensors(store.ctx,target);
    if (!store.buffer) { fclose(cost); return false; }
    ggml_backend_buffer_clear(store.buffer,0);
    fprintf(cost,"pattern,direction,selected_experts,repetition,payload_bytes,padding_bytes,api_calls,allocated_bytes,copy_sync_us\n");
    for (const char * pattern : {"grouped","per_expert","per_expert_padding512"}) {
      for (int selected : {0,2,4,6,8}) {
        for (int rep = -1; rep < 30; ++rep) {
            size_t payload = 0, padding = 0, calls = 0;
            ggml_backend_synchronize(target);
            const int64_t start = ggml_time_us();
            for (int expert = 0; expert < selected;) {
                const int count = strcmp(pattern,"grouped") == 0 ? selected : 1;
                for (int i = 0; i < 3; ++i) {
                    const size_t extra = strcmp(pattern,"per_expert_padding512") == 0 ? 512 : 0;
                    const size_t bytes = stride[i]*count;
                    ggml_backend_tensor_set(weights[i],host[i].data()+expert*stride[i],expert*stride[i],bytes+extra);
                    payload += bytes; padding += extra; ++calls;
                }
                expert += count;
            }
            ggml_backend_synchronize(target);
            const int64_t upload_us = ggml_time_us()-start;
            std::array<std::vector<uint8_t>,3> downloaded;
            for (int i = 0; i < 3; ++i) { downloaded[i].resize(selected*stride[i]); }
            const int64_t read_start = ggml_time_us();
            for (int expert = 0; expert < selected;) {
                const int count = strcmp(pattern,"grouped") == 0 ? selected : 1;
                for (int i = 0; i < 3; ++i) {
                    ggml_backend_tensor_get(weights[i],downloaded[i].data()+expert*stride[i],expert*stride[i],count*stride[i]);
                }
                expert += count;
            }
            ggml_backend_synchronize(target);
            const int64_t read_us = ggml_time_us()-read_start;
            for (int i = 0; i < 3; ++i) {
                ok = (!selected || memcmp(downloaded[i].data(),host[i].data(),downloaded[i].size()) == 0) && ok;
            }
            if (rep >= 0) {
                ok = fprintf(cost,"%s,H2D,%d,%d,%zu,%zu,%zu,%zu,%lld\n",pattern,selected,rep,payload,padding,calls,
                        ggml_backend_buffer_get_size(store.buffer),(long long)upload_us) >= 0 && ok;
                ok = fprintf(cost,"%s,D2H,%d,%d,%zu,0,%zu,%zu,%lld\n",pattern,selected,rep,payload,calls,
                        ggml_backend_buffer_get_size(store.buffer),(long long)read_us) >= 0 && ok;
            }
        }
      }
    }
    printf("expert triplet transfer actual_weight_bytes=1 raw_payload_exact=1 padding=0,512 %s\n",ok ? "OK" : "FAIL");
    return fclose(cost) == 0 && ok;
}
