#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "phase-profile.h"
#ifdef GGML_SCHED_CHECK_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float weight_value(int expert, int row, int col) {
    return float((expert * 7 + row * 3 + col) % 17 - 8) / 16;
}

static float input_value(int pass, int token, int slot, int col) {
    return float((pass * 3 + token * 5 + slot * 2 + col) % 13 - 6) / 16;
}

static bool check(ggml_backend_t target, ggml_backend_t cpu, int tokens, bool strided, bool callback, bool parallel, FILE * output, moe_phase_profile & profile, int rep) {
    const int k = 32, m = 8, experts = 8, topk = 2;
    ggml_context * leaves = ggml_init({65536, nullptr, true});
    ggml_context * ctx = ggml_init({65536, nullptr, true});
    ggml_tensor * weights = ggml_new_tensor_3d(leaves, GGML_TYPE_F32, k, m, experts);
    ggml_tensor * storage = ggml_new_tensor_2d(leaves, GGML_TYPE_I32, strided ? 16 : topk, tokens);
    ggml_tensor * ids = strided ? ggml_view_2d(ctx, storage, topk, tokens, storage->nb[1], sizeof(int32_t)) : storage;
    ggml_tensor * input = ggml_new_tensor_3d(leaves, GGML_TYPE_F32, k, topk, tokens);
    ggml_set_name(weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(ids, "ids,\"strided\"");
    ggml_set_name(input, "activation");
    ggml_set_input(storage);
    ggml_set_input(input);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(leaves, cpu);
    if (!buffer) {
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weights_data(k * m * experts);
    for (int expert = 0; expert < experts; ++expert) {
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < k; ++col) {
                weights_data[(expert * m + row) * k + col] = weight_value(expert, row, col);
            }
        }
    }
    ggml_backend_tensor_set(weights, weights_data.data(), 0, ggml_nbytes(weights));
    ggml_tensor * prepared = strided && callback ? ggml_scale(ctx, input, 1.0f) : input;
    ggml_tensor * result = ggml_mul_mat_id(ctx, weights, prepared, ids);
    ggml_set_name(result, "ffn_moe-0");
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, result);
    std::vector<ggml_backend_t> backends = target == cpu ? std::vector<ggml_backend_t>{cpu} : std::vector<ggml_backend_t>{target, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 128, parallel, true);
    if (callback) {
        ggml_backend_sched_set_eval_callback(sched, [](ggml_tensor *, bool, void *) { return true; }, nullptr);
    }
    bool ok = true;
    ggml_backend_sched_set_tensor_backend(sched, result, target);
    if (strided && callback) {
        ggml_backend_sched_set_tensor_backend(sched, ids, cpu);
        ggml_backend_sched_set_tensor_backend(sched, prepared, cpu);
    }
    ok = ggml_backend_sched_alloc_graph(sched, graph) && ok;
    if (!ok || ggml_backend_sched_get_tensor_backend(sched, result) != target) {
        fprintf(stderr, "requested compute backend was not applied\n");
        return false;
    }
    const int selected[3][2] = {{0, 1}, {1, 4}, {4, 7}};
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<int32_t> ids_data(ggml_nelements(storage), -1);
        std::vector<float> input_data(ggml_nelements(input));
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                ids_data[token * storage->ne[0] + slot + (strided ? 1 : 0)] = pass ? 7 : selected[token][slot];
                for (int col = 0; col < k; ++col) {
                    input_data[(token * topk + slot) * k + col] = input_value(pass, token, slot, col);
                }
            }
        }
        ggml_backend_tensor_set(storage, ids_data.data(), 0, ggml_nbytes(storage));
        ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
        const int64_t start = profile.enabled() ? ggml_time_us() : 0;
        const auto status = ggml_backend_sched_graph_compute(sched, graph);
        const int64_t end = profile.enabled() ? ggml_time_us() : 0;
        if (!profile.record(rep, pass ? "decode" : "prefill", pass ? tokens : 0, tokens, start, end, status) || status != GGML_STATUS_SUCCESS) {
            ok = false;
            break;
        }
        std::vector<float> actual(ggml_nelements(result));
        ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
        bool nonzero = false;
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                for (int row = 0; row < m; ++row) {
                    float expected = 0;
                    for (int col = 0; col < k; ++col) {
                        expected += weight_value(pass ? 7 : selected[token][slot], row, col) * input_value(pass, token, slot, col);
                    }
                    float value = actual[(token * topk + slot) * m + row];
                    ok = std::isfinite(value) && value == expected && ok;
                    nonzero = nonzero || value != 0;
                }
            }
        }
        ok = nonzero && ok;
        if (output) {
            ok = fwrite(actual.data(), sizeof(float), actual.size(), output) == actual.size() && ok;
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(leaves);
    printf("tokens=%d strided=%d callback=%d parallel=%d %s\n", tokens, strided, callback, parallel, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    moe_phase_profile profile;
    if (!profile.good()) {
        return 1;
    }
    bool vulkan = false;
    bool serial = false;
    const char * output_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--vulkan") == 0) {
            vulkan = true;
        } else if (strcmp(argv[i], "--serial") == 0) {
            serial = true;
        } else if (i + 1 < argc && strcmp(argv[i], "--output-bin") == 0) {
            output_path = argv[++i];
        } else {
            return 2;
        }
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_t target = cpu;
    if (vulkan) {
        ggml_backend_reg_t reg = nullptr;
#ifdef GGML_SCHED_CHECK_VULKAN
        reg = ggml_backend_vk_reg();
#endif
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            return 1;
        }
        auto device = ggml_backend_reg_dev_get(reg, 0);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                !strstr(ggml_backend_dev_description(device), "RX 6800")) {
            fprintf(stderr, "expected RX 6800 GPU; refusing fallback\n");
            return 1;
        }
        target = ggml_backend_dev_init(device, nullptr);
        if (!target) {
            return 1;
        }
    }
    FILE * output = output_path ? fopen(output_path, "wb") : nullptr;
    if (output_path && !output) {
        return 1;
    }
    bool ok = true;
    int rep = 0;
    for (bool parallel : {false, true}) {
        if (serial && parallel) {
            continue;
        }
        for (int tokens : {1, 3}) {
            for (bool strided : {false, true}) {
                for (bool callback : {false, true}) {
                    ok = check(target, cpu, tokens, strided, callback, parallel, output, profile, rep++) && ok;
                }
            }
        }
    }
    if (output) {
        ok = fclose(output) == 0 && ok;
    }
    if (target != cpu) {
        ggml_backend_free(target);
    }
    ggml_backend_free(cpu);
    return ok && profile.finish() ? 0 : 1;
}
