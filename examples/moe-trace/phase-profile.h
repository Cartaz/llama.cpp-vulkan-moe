#pragma once

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

class moe_phase_profile {
    FILE * file = nullptr;
    bool valid = true;

public:
    moe_phase_profile() {
        const char * path = std::getenv("MOE_REPLAY_PROFILE");
        if (path && *path) {
            file = std::fopen(path, "wx");
            valid = file && std::fprintf(file, "rep,phase,position,n_tokens,start_us,end_us,status\n") >= 0;
            if (!valid) {
                std::fprintf(stderr, "cannot create fresh replay phase profile '%s'\n", path);
            }
        }
    }

    moe_phase_profile(const moe_phase_profile &) = delete;
    moe_phase_profile & operator=(const moe_phase_profile &) = delete;

    ~moe_phase_profile() {
        if (file) {
            std::fclose(file);
        }
    }

    bool good() const { return valid; }
    bool enabled() const { return file != nullptr; }

    bool record(int rep, const char * phase, size_t position, int n_tokens, int64_t start, int64_t end, int status) {
        if (file) {
            valid = std::fprintf(file, "%d,%s,%zu,%d,%" PRId64 ",%" PRId64 ",%d\n",
                    rep, phase, position, n_tokens, start, end, status) >= 0 && valid;
        }
        return valid;
    }

    bool finish() {
        if (file) {
            valid = std::fprintf(file, "-1,profile_end,0,0,0,0,0\n") >= 0 && valid;
            valid = std::fclose(file) == 0 && valid;
            file = nullptr;
            if (!valid) {
                std::fprintf(stderr, "replay phase profile write failed\n");
            }
        }
        return valid;
    }
};
