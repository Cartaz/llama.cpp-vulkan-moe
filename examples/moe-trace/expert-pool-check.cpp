#include "expert-pool.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

bool check_expert_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output);
bool check_scheduler_pool(ggml_backend_t target, ggml_backend_t cpu, FILE * output);
bool check_expert_placement(ggml_backend_t target, ggml_backend_t cpu, FILE * output);

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
    ggml_cgraph * graph = nullptr;
    ~graph_run() {
        if (sched) { ggml_backend_sched_free(sched); }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    bool init(ggml_backend_t backend, const std::array<ggml_tensor *, 3> & weights, int tokens, bool broadcast, bool strided, ggml_backend_t cpu = nullptr, bool parallel = false, bool callback = false, bool placement = false, bool op_offload = true) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        if (!ctx) {
            return false;
        }
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, broadcast ? 1 : topk, tokens);
        storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, strided ? 16 : topk, tokens);
        auto * ids = strided ? ggml_view_2d(ctx, storage, topk, tokens, storage->nb[1], sizeof(int32_t)) : storage;
        result[0] = ggml_mul_mat_id(ctx, weights[0], input, ids);
        result[1] = ggml_mul_mat_id(ctx, weights[1], input, ids);
        auto * intermediate = ggml_mul(ctx, ggml_silu(ctx, result[0]), result[1]);
        result[2] = ggml_mul_mat_id(ctx, weights[2], intermediate, ids);
        graph = ggml_new_graph_custom(ctx, 128, false);
        for (auto * tensor : result) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
                fprintf(stderr, "pool fixture: unsupported op %s\n", ggml_op_name(ggml_graph_node(graph, i)->op));
                return false;
            }
        }
        if (cpu) {
            ggml_set_input(input);
            ggml_set_input(storage);
            std::vector<ggml_backend_t> backends = backend == cpu ? std::vector<ggml_backend_t>{cpu} : std::vector<ggml_backend_t>{backend, cpu};
            sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 128, parallel, op_offload);
            if (callback) {
                ggml_backend_sched_set_eval_callback(sched, [](ggml_tensor *, bool, void *) { return true; }, nullptr);
            }
            ggml_backend_sched_set_tensor_backend(sched, input, cpu);
            ggml_backend_sched_set_tensor_backend(sched, storage, cpu);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (!placement) {
                    ggml_backend_sched_set_tensor_backend(sched, node, ggml_is_view(node) ? cpu : backend);
                }
            }
            return ggml_backend_sched_alloc_graph(sched, graph);
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        return buffer != nullptr;
    }
    bool submit(ggml_backend_t backend, const std::vector<int32_t> & ids) {
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
                    data[(token * input->ne[1] + slot) * width + col] = activation_value(token, slot, col);
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
    for (const char * config : {"3:4:16", "3:12:1", "3:12:0", "3:12:16:1"}) {
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
    for (const char * config : {"3", "", "-1", "3x", "4097", "999999999999999", "0", "4"}) {
        for (bool shared : {false, true}) {
            const char * names[] = {"gate", "up", "down"};
            for (int i = 0; i < 3; ++i) {
                ggml_format_name(weights[i], "blk.3.ffn_%s_%s.weight", names[i], shared ? "shexp" : "exps");
            }
            setenv(place_env.key, config, 1);
            graph_run actual, reference;
            const bool selected = strcmp(config, "3") == 0 && !shared;
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
