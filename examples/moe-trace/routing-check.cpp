#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>

static bool check_case(ggml_type type, int n, int n_threads, bool broadcast, bool repeated, bool strided, FILE * output) {
    const int k = 256;
    const int m = 32;
    const int n_experts = 256;
    const int n_used = 8;
    const int n_inputs = broadcast ? 1 : n_used;
    ggml_context * ctx = ggml_init({128*1024*1024, nullptr, false});
    if (!ctx) {
        return false;
    }
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, type, k, m, n_experts);
    ggml_tensor * ids_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts, n);
    ggml_tensor * ids = ggml_view_2d(ctx, ids_storage, n_used, n, ids_storage->nb[1], sizeof(int32_t));
    ggml_tensor * input_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, strided ? k + 16 : k, n_inputs, n);
    ggml_tensor * input = strided ? ggml_view_3d(ctx, input_storage, k, n_inputs, n, input_storage->nb[1], input_storage->nb[2], 0) : input_storage;
    ggml_tensor * result = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    std::vector<float> row(k);
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int r = 0; r < m; ++r) {
            for (int col = 0; col < k; ++col) {
                row[col] = float((expert*13 + r*7 + col*3) % 31 - 15)/16.0f;
            }
            char * dst = (char *) weights->data + expert*weights->nb[2] + r*weights->nb[1];
            if (type == GGML_TYPE_F32) {
                std::memcpy(dst, row.data(), k*sizeof(float));
            } else {
                ggml_get_type_traits(type)->from_float_ref(row.data(), dst, k);
            }
        }
    }

    ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);
    std::vector<uint8_t> workspace(plan.work_size + 128, 0xa5);
    plan.work_data = workspace.data() + 64;
    bool ok = true;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int pass = 0; pass < 2; ++pass) {
        for (int token = 0; token < n; ++token) {
            for (int slot = 0; slot < n_used; ++slot) {
                const int expert = pass == 0 ? (repeated ? 255 : (token*17 + slot*29) % n_experts) : ((token + slot) % 2 ? 255 : 0);
                ggml_set_i32_nd(ids, slot, token, 0, 0, expert);
            }
            for (int slot = 0; slot < n_inputs; ++slot) {
                for (int col = 0; col < k; ++col) {
                    const float value = float((token*5 + slot*11 + col*7) % 23 - 11)/32.0f;
                    ggml_set_f32_nd(input, col, slot, token, 0, pass == 0 ? value : -value);
                }
            }
        }
        ok = ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS && ok;
        for (int guard = 0; guard < 64; ++guard) {
            ok = workspace[guard] == 0xa5 && workspace[64 + plan.work_size + guard] == 0xa5 && ok;
        }
        if (type == GGML_TYPE_F32) {
            for (int token = 0; token < n; ++token) {
                for (int slot = 0; slot < n_used; ++slot) {
                    const int expert = ggml_get_i32_nd(ids, slot, token, 0, 0);
                    for (int r = 0; r < m; ++r) {
                        float expected = 0;
                        for (int col = 0; col < k; ++col) {
                            expected += ggml_get_f32_nd(weights, col, r, expert, 0)*ggml_get_f32_nd(input, col, slot % n_inputs, token, 0);
                        }
                        ok = ggml_get_f32_nd(result, r, slot, token, 0) == expected && ok;
                    }
                }
            }
        }
        const size_t size = ggml_nbytes(result);
        const uint8_t * bytes = (const uint8_t *) result->data;
        for (size_t i = 0; i < size; ++i) {
            hash = (hash ^ bytes[i])*UINT64_C(1099511628211);
        }
        if (output) {
            ok = std::fwrite(bytes, 1, size, output) == size && ok;
        }
    }
    std::printf("%s n=%d threads=%d broadcast=%d repeated=%d strided=%d %016llx %s\n", ggml_type_name(type), n, n_threads, broadcast, repeated, strided, (unsigned long long) hash, ok ? "OK" : "FAIL");
    std::fprintf(stderr, "%s n=%d threads=%d broadcast=%d workspace=%zu\n", ggml_type_name(type), n, n_threads, broadcast, plan.work_size);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    FILE * output = nullptr;
    if (argc == 3 && std::strcmp(argv[1], "--output-bin") == 0) {
        output = std::fopen(argv[2], "wb");
        if (!output) {
            return 1;
        }
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--output-bin PATH]\n", argv[0]);
        return 1;
    }
    ggml_cpu_init();
    bool ok = true;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (int n : {1, 7, 8, 9, 65}) {
            for (int n_threads : {1, 4, 8}) {
                for (bool broadcast : {false, true}) {
                    for (bool repeated : {false, true}) {
                        for (bool strided : {false, true}) {
                            ok = check_case(type, n, n_threads, broadcast, repeated, strided, output) && ok;
                        }
                    }
                }
            }
        }
    }
    ok = check_case(GGML_TYPE_Q4_K, 2048, 8, true, false, false, output) && ok;
    if (output) {
        ok = std::fclose(output) == 0 && ok;
    }
    return ok ? 0 : 1;
}
