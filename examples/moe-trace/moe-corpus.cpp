#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "phase-profile.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;

struct corpus_route {
    FILE * file = nullptr;
    const char * phase = "prefill";
    size_t position = 0;
    bool after_up = false;
};

static bool corpus_callback(ggml_tensor * t, bool ask, void * data) {
    auto & route = *static_cast<corpus_route *>(data);
    const bool topk = route.after_up ? t->op == GGML_OP_MUL_MAT_ID && std::strncmp(t->name, "ffn_moe_up-", 11) == 0 :
            std::strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) { return topk; }
    auto * ids = route.after_up ? t->src[2] : t;
    if (!topk || !ids || ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1) { return true; }
    const char * dash = std::strrchr(t->name, '-');
    if (!route.file || !dash) { return false; }
    std::vector<uint8_t> storage(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, storage.data(), 0, storage.size());
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
            int32_t expert;
            std::memcpy(&expert, storage.data() + token*ids->nb[1] + rank*ids->nb[0], sizeof(expert));
            if (std::fprintf(route.file, "%s,%zu,%d,%lld,%d\n", route.phase, route.position + token,
                    std::atoi(dash+1), (long long) rank, expert) < 0) { return false; }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) { return 1; }
    const char * cases_path = std::getenv("MOE_CORPUS_CASES");
    const char * output_path = std::getenv("MOE_CORPUS_OUT");
    if (!cases_path || !output_path || params.n_predict <= 0) {
        LOG_ERR("set MOE_CORPUS_CASES, MOE_CORPUS_OUT and positive -n\n");
        return 1;
    }
    std::ifstream cases(cases_path);
    std::vector<json> items;
    std::string line;
    while (std::getline(cases, line)) {
        items.push_back(json::parse(line));
        const std::string id = items.back().at("id");
        if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos) { return 1; }
    }
    if (!cases.eof() || items.empty()) { return 1; }
    corpus_route route;
    route.after_up = std::getenv("MOE_CORPUS_CALLBACK_AFTER_UP") != nullptr;
    params.warmup = false;
    const bool callback = std::getenv("MOE_CORPUS_CALLBACK") != nullptr;
    params.cb_eval = callback ? corpus_callback : nullptr;
    params.cb_eval_user_data = callback ? &route : nullptr;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto initialized = common_init_from_params(params);
    auto * model = initialized->model();
    auto * ctx = initialized->context();
    if (!model || !ctx) { return 1; }
    const auto * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    using file_ptr = std::unique_ptr<FILE, int (*)(FILE *)>;
    file_ptr result(std::fopen((std::string(output_path)+"/responses.jsonl").c_str(), "wx"), std::fclose);
    moe_phase_profile phases;
    if (!result || !phases.good()) { return 1; }
    for (size_t index = 0; index < items.size(); ++index) {
        const auto & item = items[index];
        const std::string id = item.at("id");
        const std::string base = std::string(output_path)+"/"+id;
        const std::string prompt = item.value("prompt", std::string());
        if (!item.contains("prefill_ids") && prompt.empty()) { return 1; }
        auto tokens = item.contains("prefill_ids") ? item.at("prefill_ids").get<std::vector<llama_token>>() :
                common_tokenize(ctx, prompt, llama_vocab_get_add_bos(vocab), true);
        const std::vector<llama_token> forced = item.value("decode", std::vector<llama_token>{});
        const size_t limit = item.contains("decode") ? forced.size() : size_t(params.n_predict);
        if (tokens.empty() || tokens.size()+limit > llama_n_ctx_seq(ctx) ||
                std::any_of(tokens.begin(), tokens.end(), [nv](llama_token token) { return token < 0 || token >= nv; }) ||
                std::any_of(forced.begin(), forced.end(), [nv](llama_token token) { return token < 0 || token >= nv; })) { return 1; }
        file_ptr raw(std::fopen((base+"-logits.bin").c_str(), "wx"), std::fclose);
        file_ptr token_file(std::fopen((base+"-tokens.csv").c_str(), "wx"), std::fclose);
        file_ptr route_file(callback ? std::fopen((base+"-routes.csv").c_str(), "wx") : nullptr, std::fclose);
        if (!raw || !token_file || (callback && !route_file)) { return 1; }
        route.file = route_file.get();
        if (callback) { std::fprintf(route.file, "phase,token,layer,rank,expert\n"); }
        std::fprintf(token_file.get(), "phase,token,id\n");
        for (size_t i = 0; i < tokens.size(); ++i) { std::fprintf(token_file.get(), "prefill,%zu,%d\n", i, tokens[i]); }
        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
        auto sampling = params.sampling;
        sampling.seed = item.at("seed").get<uint32_t>();
        std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model, sampling), common_sampler_free);
        if (!sampler) { return 1; }
        for (llama_token token : tokens) { common_sampler_accept(sampler.get(), token, false); }
        size_t raw_vectors = 0;
        auto evaluate = [&](llama_token * input, int count, const char * phase, size_t position) {
            route.phase = phase;
            route.position = position;
            const int64_t start = ggml_time_us();
            const int status = llama_decode(ctx, llama_batch_get_one(input, count));
            llama_synchronize(ctx);
            if (!phases.record(int(index), phase, position, count, start, ggml_time_us(), status) || status) { return false; }
            const float * logits = llama_get_logits_ith(ctx, -1);
            bool nonzero = false;
            if (!logits) { return false; }
            for (int v = 0; v < nv; ++v) {
                if (!std::isfinite(logits[v])) { return false; }
                nonzero |= logits[v] != 0;
            }
            if (!nonzero || std::fwrite(logits, sizeof(float), nv, raw.get()) != size_t(nv)) { return false; }
            ++raw_vectors;
            return true;
        };
        const int step = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
        for (size_t position = 0; position < tokens.size(); position += step) {
            if (!evaluate(tokens.data()+position, int(std::min(size_t(step),tokens.size()-position)), "prefill", position)) { return 2; }
        }
        std::string text;
        std::vector<llama_token> generated;
        bool eog = false;
        for (size_t i = 0; i < limit; ++i) {
            llama_token token = item.contains("decode") ? forced[i] : common_sampler_sample(sampler.get(), ctx, -1);
            if (!item.contains("decode") && llama_vocab_is_eog(vocab, token)) { eog = true; break; }
            common_sampler_accept(sampler.get(), token, true);
            if (!evaluate(&token, 1, "decode", tokens.size()+i)) { return 2; }
            std::fprintf(token_file.get(), "decode,%zu,%d\n", tokens.size()+i, token);
            generated.push_back(token);
            text += common_token_to_piece(vocab, token, true);
        }
        file_ptr raw_text(std::fopen((base+"-text.bin").c_str(), "wx"), std::fclose);
        if (!raw_text || std::fwrite(text.data(), 1, text.size(), raw_text.get()) != text.size() || std::fclose(raw_text.release())) { return 1; }
        const json row = {{"id",id},{"case_index",index},{"prompt_ids",tokens},{"generated_ids",generated},
                {"text",text},{"eog",eog},{"raw_vectors",raw_vectors},{"seed",sampling.seed},{"temperature",sampling.temp},
                {"text_encoding","utf8_replace"},{"text_raw_file",id+"-text.bin"},{"text_bytes",text.size()},
                {"callback",callback},{"mode",item.contains("decode") ? "teacher_forced" : "autonomous"},
                {"callback_stage",route.after_up ? "up_projection" : "topk"},
                {"input",item.contains("prefill_ids") ? "fixed_token_ids" : "tokenized_prompt"},
                {"reset","synchronize then clear_data=true; scheduler pools persist between cases"}};
        if (std::fclose(raw.release()) || std::fclose(token_file.release())) { return 1; }
        if (route_file && std::fclose(route_file.release())) { return 1; }
        route.file = nullptr;
        const auto serialized = row.dump(-1, ' ', false, json::error_handler_t::replace);
        if (std::fprintf(result.get(), "%s\n", serialized.c_str()) < 0 || std::fflush(result.get())) { return 1; }
        LOG_INF("corpus: %s prompt=%zu decode=%zu vectors=%zu\n",id.c_str(),tokens.size(),generated.size(),raw_vectors);
    }
    const bool finished = phases.finish();
    const int closed = std::fclose(result.release());
    return finished && closed == 0 ? 0 : 1;
}
