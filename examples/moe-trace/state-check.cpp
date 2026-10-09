#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

int main(int argc, char ** argv) {
    common_init();
    common_params params;
    if (!common_params_parse(argc,argv,params,LLAMA_EXAMPLE_COMMON)) { return 1; }
    const char * input = std::getenv("MOE_STATE_CASE");
    const char * directory = std::getenv("MOE_STATE_OUT");
    if (!input || !directory) { return 1; }
    std::ifstream stream(input);
    const auto item = nlohmann::json::parse(stream);
    auto pp = item.at("prefill_ids").get<std::vector<llama_token>>();
    auto tg = item.at("decode").get<std::vector<llama_token>>();
    if (pp.empty() || tg.size() <= 16) { return 1; }
    params.warmup = false;
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto initialized = common_init_from_params(params);
    auto * model = initialized->model();
    auto * ctx = initialized->context();
    if (!model || !ctx || pp.size()+tg.size() > llama_n_ctx_seq(ctx)) { return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (const auto * tokens : {&pp,&tg}) {
        if (std::any_of(tokens->begin(),tokens->end(),[nv](llama_token token) { return token < 0 || token >= nv; })) { return 1; }
    }
    using file_ptr = std::unique_ptr<FILE,int (*)(FILE *)>;
    const std::string base(directory);
    file_ptr raw(std::fopen((base+"/logits.bin").c_str(),"wx"),std::fclose);
    file_ptr rewound(std::fopen((base+"/rewound.bin").c_str(),"wx"),std::fclose);
    if (!raw || !rewound) { return 1; }
    size_t vectors = 0;
    auto logits = [&]() {
        const float * source = llama_get_logits_ith(ctx,-1);
        if (!source) { return std::vector<float>{}; }
        std::vector<float> result(source,source+nv);
        if (std::any_of(result.begin(),result.end(),[](float value) { return !std::isfinite(value); }) ||
                std::all_of(result.begin(),result.end(),[](float value) { return value == 0; })) { result.clear(); }
        return result;
    };
    auto evaluate = [&](llama_token * tokens, int n, FILE * file) {
        if (llama_decode(ctx,llama_batch_get_one(tokens,n))) { return std::vector<float>{}; }
        llama_synchronize(ctx);
        auto result = logits();
        if (result.empty() || std::fwrite(result.data(),sizeof(float),nv,file) != size_t(nv)) { result.clear(); }
        if (file == raw.get() && !result.empty()) { ++vectors; }
        return result;
    };
    auto save = [&]() {
        llama_synchronize(ctx);
        const size_t size = llama_state_get_size(ctx);
        if (!size || size > 1024*1024*1024) { return std::vector<uint8_t>{}; }
        std::vector<uint8_t> state(size);
        if (llama_state_get_data(ctx,state.data(),state.size()) != size) { state.clear(); }
        return state;
    };
    auto restore = [&](const std::vector<uint8_t> & state,const std::vector<float> & reference) {
        llama_synchronize(ctx);
        if (llama_state_set_data(ctx,state.data(),state.size()) != state.size()) { return false; }
        auto current = logits();
        return current.size() == reference.size() && memcmp(current.data(),reference.data(),current.size()*sizeof(float)) == 0;
    };
    const int step = std::min(llama_n_batch(ctx),llama_n_ubatch(ctx));
    for (size_t i = 0; i < pp.size(); i += step) {
        if (evaluate(pp.data()+i,int(std::min(size_t(step),pp.size()-i)),raw.get()).empty()) { return 2; }
    }
    const auto pp_logits = logits();
    const auto pp_state = save();
    if (pp_state.empty()) { return 3; }
    std::vector<std::vector<float>> reference;
    for (int i = 0; i < 16; ++i) {
        auto value = evaluate(&tg[i],1,raw.get());
        if (value.empty()) { return 2; }
        reference.push_back(std::move(value));
    }
    if (!restore(pp_state,pp_logits)) { return 4; }
    for (int i = 0; i < 16; ++i) {
        auto value = evaluate(&tg[i],1,rewound.get());
        if (value.size() != reference[i].size() || memcmp(value.data(),reference[i].data(),value.size()*sizeof(float))) { return 5; }
    }
    const auto checkpoint_logits = logits();
    const auto checkpoint = save();
    if (checkpoint.empty()) { return 3; }
    int abort_calls = 0;
    llama_set_abort_callback(ctx,[](void * data) { ++*static_cast<int *>(data); return true; },&abort_calls);
    const int aborted = llama_decode(ctx,llama_batch_get_one(&tg[16],1));
    llama_synchronize(ctx);
    llama_set_abort_callback(ctx,nullptr,nullptr);
    if (aborted == 0 || abort_calls == 0 || !restore(checkpoint,checkpoint_logits)) { return 6; }
    for (size_t i = 16; i < tg.size(); ++i) {
        if (evaluate(&tg[i],1,raw.get()).empty()) { return 2; }
    }
    if (std::fclose(raw.release()) || std::fclose(rewound.release())) { return 1; }
    const nlohmann::json result = {{"prompt_tokens",pp.size()},{"decode_tokens",tg.size()},{"raw_vectors",vectors},
            {"rewound_vectors",16},{"pp_state_bytes",pp_state.size()},{"decode_state_bytes",checkpoint.size()},
            {"abort_return",aborted},{"abort_calls",abort_calls},{"state_logits_byte_exact",true},
            {"rewound_logits_byte_exact",true},{"clear_after_test",true}};
    llama_memory_clear(llama_get_memory(ctx),true);
    std::ofstream report(base+"/state-result.json");
    report << result.dump(2) << '\n';report.close();
    LOG_INF("state: save/restore,CPU abort/recover and rewind16 PASS; pools retained\n");
    return report ? 0 : 1;
}
