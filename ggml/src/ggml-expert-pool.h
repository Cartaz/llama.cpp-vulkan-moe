#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// One immutable layer, one backend, one outstanding lease. Not thread safe.
// Sources and backend must outlive the pool. Release only after the last consumer is submitted.
class ggml_backend_expert_pool {
public:
    enum class status { ready, unavailable, capacity, busy, invalid, routing };
    struct counters {
        uint64_t hits = 0, misses = 0, evictions = 0, upload_bytes = 0;
    };

    ggml_backend_expert_pool() = default;
    ggml_backend_expert_pool(const ggml_backend_expert_pool &) = delete;
    ggml_backend_expert_pool & operator=(const ggml_backend_expert_pool &) = delete;

    ~ggml_backend_expert_pool() {
        release();
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    bool init(ggml_backend_t target, const std::array<ggml_tensor *, 3> & weights, int capacity, size_t byte_budget, bool lazy = false) {
        if (ctx || !target || capacity <= 0 || !weights[0]) {
            return false;
        }
        const int64_t experts = weights[0]->ne[2];
        if (capacity > experts) {
            return false;
        }
        size_t per_slot = 0;
        for (auto * weight : weights) {
            if (!weight || !weight->buffer || !weight->data || !ggml_backend_buffer_is_host(weight->buffer) ||
                    !ggml_is_contiguous(weight) || weight->ne[2] != experts || weight->ne[3] != 1 ||
                    (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_Q4_K && weight->type != GGML_TYPE_Q6_K) ||
                    weight->nb[2] > std::numeric_limits<size_t>::max() - per_slot) {
                return false;
            }
            per_slot += weight->nb[2];
        }
        if (per_slot > byte_budget / size_t(capacity)) {
            return false;
        }
        ctx = ggml_init({ggml_tensor_overhead() * 3 + 1024, nullptr, true});
        if (!ctx) {
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            tensors[i] = ggml_new_tensor_3d(ctx, weights[i]->type, weights[i]->ne[0], weights[i]->ne[1], capacity);
            ggml_format_name(tensors[i], "pool.%s", weights[i]->name);
        }
        const auto buft = ggml_backend_get_default_buffer_type(target);
        size_t allocated = 0;
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        for (auto * tensor : tensors) {
            const size_t bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
            const size_t pad = (alignment - bytes % alignment) % alignment;
            if (bytes > byte_budget - allocated || pad > byte_budget - allocated - bytes) {
                return false;
            }
            allocated += bytes + pad;
        }
        backend = target;
        sources = weights;
        slot_experts.assign(capacity, -1);
        ages.assign(capacity, 0);
        expert_slots.assign(experts, -1);
        slot_bytes = per_slot;
        reserved = allocated;
        budget = byte_budget;
        return lazy || allocate();
    }

    status request_status(const std::vector<int32_t> & ids, size_t top_k = 1) const {
        std::vector<int32_t> requested;
        return validate_request(ids, top_k, requested);
    }

    status acquire(const std::vector<int32_t> & ids, std::vector<int32_t> & remapped, size_t top_k = 1) {
        remapped.clear();
        std::vector<int32_t> requested;
        const auto checked = validate_request(ids, top_k, requested);
        if (checked != status::ready) {
            return checked;
        }
        if (!buffer && !allocate()) {
            return status::unavailable;
        }
        // Plan every slot before the first write. Never evict an expert needed by this request.
        auto planned = slot_experts;
        std::vector<std::pair<int32_t, int32_t>> uploads;
        for (int32_t expert : requested) {
            if (expert_slots[expert] >= 0) {
                continue;
            }
            int victim = -1;
            for (size_t slot = 0; slot < planned.size(); ++slot) {
                if (std::binary_search(requested.begin(), requested.end(), planned[slot])) {
                    continue;
                }
                if (victim < 0 || planned[slot] < 0 ||
                        (planned[victim] >= 0 && ages[slot] < ages[victim])) {
                    victim = int(slot);
                }
                if (planned[slot] < 0) {
                    break;
                }
            }
            GGML_ASSERT(victim >= 0);
            planned[victim] = expert;
            uploads.emplace_back(expert, victim);
        }
        for (const auto & upload : uploads) {
            const int expert = upload.first, slot = upload.second;
            for (int i = 0; i < 3; ++i) {
                const size_t bytes = sources[i]->nb[2];
                ggml_backend_tensor_set(tensors[i], (const uint8_t *) sources[i]->data + expert * bytes, slot * bytes, bytes);
            }
            if (slot_experts[slot] >= 0) {
                expert_slots[slot_experts[slot]] = -1;
                ++stats.evictions;
            }
            slot_experts[slot] = expert;
            expert_slots[expert] = slot;
        }
        stats.misses += uploads.size();
        stats.hits += requested.size() - uploads.size();
        stats.upload_bytes += uploads.size() * slot_bytes;
        ++clock;
        for (int32_t expert : requested) {
            ages[expert_slots[expert]] = clock;
        }
        remapped.reserve(ids.size());
        for (int32_t expert : ids) {
            remapped.push_back(expert_slots[expert]);
        }
        leased = true;
        return status::ready;
    }

    void release() {
        if (leased) {
            ggml_backend_synchronize(backend);
            leased = false;
        }
    }

    ggml_tensor * weight(int projection) const { return buffer ? tensors.at(projection) : nullptr; }
    const counters & counts() const { return stats; }
    const std::vector<int32_t> & residents() const { return slot_experts; }
    size_t payload_bytes() const { return slot_bytes * slot_experts.size(); }
    size_t reserved_bytes() const { return std::max(reserved, allocated_bytes()); }
    size_t allocated_bytes() const { return buffer ? ggml_backend_buffer_get_size(buffer) : 0; }

private:
    status validate_request(const std::vector<int32_t> & ids, size_t top_k, std::vector<int32_t> & requested) const {
        if (!backend) {
            return status::unavailable;
        }
        if (leased) {
            return status::busy;
        }
        if (ids.empty() || top_k == 0 || ids.size() % top_k != 0) {
            return status::invalid;
        }
        requested = ids;
        for (int32_t id : requested) {
            if (id < 0 || size_t(id) >= expert_slots.size()) {
                return status::invalid;
            }
        }
        // Vulkan matrix kernels can assume distinct experts within each token's top-k.
        for (size_t token = 0; token < ids.size(); token += top_k) {
            for (size_t i = 0; i < top_k; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if (ids[token + i] == ids[token + j]) {
                        return status::routing;
                    }
                }
            }
        }
        std::sort(requested.begin(), requested.end());
        requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
        if (requested.size() > slot_experts.size()) {
            return status::capacity;
        }
        return status::ready;
    }

    bool allocate() {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer || ggml_backend_buffer_get_size(buffer) > budget) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
            return false;
        }
        // Clear unused slots and backend padding before the first quantized kernel.
        ggml_backend_buffer_clear(buffer, 0);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        return true;
    }

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_t backend = nullptr;
    std::array<ggml_tensor *, 3> sources{}, tensors{};
    std::vector<int32_t> slot_experts, expert_slots;
    std::vector<uint64_t> ages;
    counters stats;
    uint64_t clock = 0;
    size_t slot_bytes = 0, reserved = 0, budget = 0;
    bool leased = false;
};
