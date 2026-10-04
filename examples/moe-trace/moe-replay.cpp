#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct replay_workload {
    std::vector<llama_token> prompt;
    std::vector<llama_token> decode;
};

static bool read_workload(const char * path, replay_workload & workload) {
    std::ifstream stream(path);
    std::string line;
    if (!std::getline(stream, line) || line != "phase,token,id") {
        LOG_ERR("expected phase,token,id header in replay input\n");
        return false;
    }
    bool decoding = false;
    size_t line_number = 1;
    while (std::getline(stream, line)) {
        ++line_number;
        std::istringstream row(line);
        std::string phase, position_text, id_text;
        if (!std::getline(row, phase, ',') || !std::getline(row, position_text, ',') || !std::getline(row, id_text) ||
            position_text.empty() || id_text.empty() || position_text.find_first_not_of("0123456789") != std::string::npos ||
            id_text.find_first_not_of("0123456789") != std::string::npos) {
            LOG_ERR("invalid replay row %zu\n", line_number);
            return false;
        }
        try {
            const auto position = std::stoull(position_text);
            const auto id = std::stoull(id_text);
            if (position != workload.prompt.size() + workload.decode.size() || id > INT32_MAX ||
                (phase != "prefill" && phase != "decode") || (decoding && phase == "prefill")) {
                LOG_ERR("invalid replay position, phase or token at row %zu\n", line_number);
                return false;
            }
            decoding = phase == "decode";
            (decoding ? workload.decode : workload.prompt).push_back(static_cast<llama_token>(id));
        } catch (const std::exception &) {
            LOG_ERR("replay integer out of range at row %zu\n", line_number);
            return false;
        }
    }
    if (!stream.eof() || workload.prompt.empty()) {
        LOG_ERR("replay needs a nonempty prefill followed by optional decode tokens\n");
        return false;
    }
    return true;
}

static uint64_t hash_logits(const float * logits, size_t count) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(logits);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static bool evaluate(llama_context * ctx, replay_workload & workload, int rep, int step, size_t n_vocab, FILE * raw) {
    llama_memory_clear(llama_get_memory(ctx), false);
    auto eval = [&](llama_token * tokens, int n, const char * phase, size_t position) {
        const int64_t start = ggml_time_us();
        if (llama_decode(ctx, llama_batch_get_one(tokens, n)) != 0) {
            LOG_ERR("replay decode failed at position %zu\n", position);
            return false;
        }
        llama_synchronize(ctx);
        const int64_t elapsed = ggml_time_us() - start;
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            LOG_ERR("no replay logits at position %zu\n", position);
            return false;
        }
        if (rep >= 0) {
            const uint64_t hash = hash_logits(logits, n_vocab);
            if (raw && std::fwrite(logits, sizeof(float), n_vocab, raw) != n_vocab) {
                LOG_ERR("cannot write replay logits\n");
                return false;
            }
            if (std::printf("%d,%s,%zu,%d,%" PRId64 ",%016" PRIx64 "\n", rep, phase, position, n, elapsed, hash) < 0) {
                return false;
            }
        }
        return true;
    };
    for (size_t position = 0; position < workload.prompt.size(); position += step) {
        const int n = static_cast<int>(std::min(static_cast<size_t>(step), workload.prompt.size() - position));
        if (!eval(workload.prompt.data() + position, n, "prefill", position)) {
            return false;
        }
    }
    for (size_t i = 0; i < workload.decode.size(); ++i) {
        if (!eval(workload.decode.data() + i, 1, "decode", workload.prompt.size() + i)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * input = std::getenv("MOE_REPLAY_IN");
    if (!input) {
        LOG_ERR("set MOE_REPLAY_IN to the phase,token,id CSV recorded by llama-moe-trace\n");
        return 1;
    }
    replay_workload workload;
    if (!read_workload(input, workload)) {
        return 1;
    }
    int reps = 3;
    if (const char * value = std::getenv("MOE_REPLAY_REPS")) {
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (*value == '\0' || *end != '\0' || parsed < 1 || parsed > INT_MAX) {
            LOG_ERR("MOE_REPLAY_REPS must be a positive integer\n");
            return 1;
        }
        reps = static_cast<int>(parsed);
    }
    params.warmup = false;
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto initialized = common_init_from_params(params);
    llama_model * model = initialized->model();
    llama_context * ctx = initialized->context();
    if (!model || !ctx) {
        return 1;
    }
    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (workload.prompt.size() + workload.decode.size() > llama_n_ctx_seq(ctx)) {
        LOG_ERR("replay workload exceeds context per sequence\n");
        return 1;
    }
    for (const auto * tokens : {&workload.prompt, &workload.decode}) {
        for (const llama_token id : *tokens) {
            if (static_cast<size_t>(id) >= n_vocab) {
                LOG_ERR("replay token %d is outside the model vocabulary\n", id);
                return 1;
            }
        }
    }
    const int step = static_cast<int>(std::min(llama_n_batch(ctx), llama_n_ubatch(ctx)));
    using file_ptr = std::unique_ptr<FILE, int (*)(FILE *)>;
    file_ptr raw(nullptr, &std::fclose);
    if (const char * output = std::getenv("MOE_REPLAY_LOGITS_OUT")) {
        raw.reset(std::fopen(output, "wb"));
        if (!raw) {
            LOG_ERR("cannot open replay logits output\n");
            return 1;
        }
    }
    LOG_INF("replay: %zu prompt tokens, %zu decode tokens, %d repetitions, step=%d\n",
            workload.prompt.size(), workload.decode.size(), reps, step);
    if (!evaluate(ctx, workload, -1, step, n_vocab, nullptr)) {
        return 1;
    }
    std::printf("rep,phase,position,n_tokens,elapsed_us,logits_hash\n");
    for (int rep = 0; rep < reps; ++rep) {
        if (!evaluate(ctx, workload, rep, step, n_vocab, raw.get())) {
            return 1;
        }
    }
    if (std::fflush(stdout) != 0 || (raw && std::fflush(raw.get()) != 0)) {
        return 1;
    }
    return 0;
}
