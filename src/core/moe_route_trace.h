#pragma once

#include "fusionrt/json.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <span>
#include <stdexcept>

namespace fusionrt {

class MoeRouteTrace {
public:
    explicit MoeRouteTrace(const std::string& path) : out_(path, std::ios::binary) {
        if (!out_) throw std::runtime_error("Cannot open MoE route trace: " + path);
    }

    void begin(const void* owner) {
        owner_ = owner;
        ++request_;
        layers_.clear();
    }

    bool enabled_for(const void* owner) const { return owner_ == owner; }
    int32_t* staging(size_t n) { staging_.resize(n); return staging_.data(); }

    void record(int layer, int pos, int count, int top_k, const int32_t* ids) {
        auto& routes = layers_[layer];
        routes.top_k = top_k;
        const auto begin = static_cast<size_t>(pos) * top_k;
        const auto length = static_cast<size_t>(count) * top_k;
        if (routes.ids.size() < begin + length) routes.ids.resize(begin + length, -1);
        std::copy_n(ids, length, routes.ids.data() + begin);
    }

    void finish(int prompt_end, int end) {
        owner_ = nullptr;
        for (const auto& [layer, routes] : layers_) {
            const auto width = routes.top_k;
            auto token = [&](int pos) { return std::span(routes.ids.data() + static_cast<size_t>(pos) * width, width); };
            std::map<int, int64_t> counts;
            int64_t tokens = 0, pairs = 0, shared = 0, windows = 0, union_total = 0;
            int contiguous = 0;
            std::vector<int> expert_union;
            const int last = std::min(end, static_cast<int>(routes.ids.size() / width));
            for (int pos = 0; pos < last; ++pos) {
                const auto ids = token(pos);
                if (ids.front() < 0) { contiguous = 0; continue; }
                JsonArray experts;
                for (const int id : ids) experts.emplace_back(id);
                write({{"type", "routes"}, {"request", request_}, {"layer", layer},
                       {"position", pos}, {"phase", pos < prompt_end ? "prompt" : "decode"},
                       {"experts", std::move(experts)}});
                if (pos < prompt_end) continue;
                for (const int id : ids) ++counts[id];
                ++tokens;
                if (++contiguous >= 2) {
                    const auto prev = token(pos - 1);
                    for (const int id : ids) shared += std::find(prev.begin(), prev.end(), id) != prev.end();
                    ++pairs;
                }
                if (contiguous >= 5) {
                    expert_union.clear();
                    for (int past = pos - 4; past <= pos; ++past) {
                        const auto next = token(past);
                        expert_union.insert(expert_union.end(), next.begin(), next.end());
                    }
                    std::sort(expert_union.begin(), expert_union.end());
                    union_total += std::unique(expert_union.begin(), expert_union.end()) - expert_union.begin();
                    ++windows;
                }
            }
            if (!tokens) continue;
            double entropy = 0.0;
            for (const auto& entry : counts) {
                const double p = static_cast<double>(entry.second) / (tokens * width);
                entropy -= p * std::log2(p);
            }
            write({{"type", "summary"}, {"request", request_}, {"layer", layer}, {"decode_rows", tokens},
                   {"H", entropy}, {"effective_experts", std::exp2(entropy)},
                   {"adjacent_overlap", pairs ? JsonValue(static_cast<double>(shared) / (pairs * width)) : JsonValue()},
                   {"U_5", windows ? JsonValue(static_cast<double>(union_total) / windows) : JsonValue()}});
        }
        out_.flush();
        if (!out_) throw std::runtime_error("Cannot write MoE route trace");
    }

private:
    struct Layer { int top_k = 0; std::vector<int32_t> ids; };
    void write(JsonObject row) { out_ << JsonValue(std::move(row)).dump(0) << '\n'; }
    std::ofstream out_;
    const void* owner_ = nullptr;
    int64_t request_ = 0;
    std::map<int, Layer> layers_;
    std::vector<int32_t> staging_;
};

}  // namespace fusionrt
