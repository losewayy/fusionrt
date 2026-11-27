#pragma once

#include "fusionrt/cost_model.h"
#include "fusionrt/json.h"

#include <filesystem>
#include <fstream>
#include <optional>

namespace fusionrt {

inline std::vector<MoeLayerHeat> moe_layer_heat(std::span<const MoeLayerCost> layers,
                                              const std::filesystem::path& route_profile, bool dynamic) {
    if (route_profile.empty() && !dynamic) return {};
    std::vector<std::vector<double>> counts(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) counts[i].resize(layers[i].experts);
    if (!route_profile.empty()) {
        std::ifstream input(route_profile, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open MoE route profile");
        size_t line = 0;
        for (std::string row; std::getline(input, row);) {
            ++line;
            if (row.find_first_not_of(" \t\r") == std::string::npos) continue;
            try {
                const auto record = JsonValue::parse(row);
                const auto& fields = record.as_object();
                const auto phase = fields.find("phase");
                if (phase == fields.end() || phase->second.as_string() != "decode") continue;
                const auto& layer = fields.at("layer");
                if (!layer.is_int() || layer.as_int() < 0 || static_cast<uint64_t>(layer.as_int()) >= layers.size())
                    throw std::invalid_argument("Invalid layer");
                auto& experts = counts[layer.as_int()];
                const auto& routes = fields.at("experts").as_array();
                if (experts.empty() || routes.empty()) throw std::invalid_argument("Invalid routes");
                for (const auto& expert : routes) {
                    if (!expert.is_int() || expert.as_int() < 0 || static_cast<uint64_t>(expert.as_int()) >= experts.size())
                        throw std::invalid_argument("Invalid expert");
                    ++experts[expert.as_int()];
                }
            } catch (const std::exception& error) {
                throw std::invalid_argument("Invalid MoE route profile at line " + std::to_string(line) +
                                            ": " + error.what());
            }
        }
        if (input.bad()) throw std::runtime_error("Cannot read MoE route profile");
    }
    std::vector<MoeLayerHeat> result;
    result.reserve(layers.size());
    for (auto& experts : counts) {
        if (dynamic && std::none_of(experts.begin(), experts.end(), [](double n) { return n > 0; }))
            for (size_t n = 0; n < experts.size(); ++n) experts[n] = 1.0 / (n + 1);  // Zipf rank prior
        result.emplace_back(experts);
    }
    return result;
}

inline std::vector<int32_t> moe_expert_to_slot(int experts, std::span<const int32_t> slot_to_expert) {
    std::vector<int32_t> result(experts, -1);
    for (size_t slot = 0; slot < slot_to_expert.size(); ++slot)
        result[slot_to_expert[slot]] = static_cast<int32_t>(slot);
    return result;
}

class MoeDynamicCache {
public:
    struct Admission { int32_t expert, slot; };

    MoeDynamicCache(int experts, std::span<const int32_t> slots, double half_life = 32.0)
        : slot_to_expert_(slots.begin(), slots.end()), heat_(experts), decay_(std::exp2(-1.0 / half_life)) {}

    std::optional<Admission> observe(std::span<const int32_t> routes, std::span<const int32_t> expert_to_slot,
        const CostHardware& hw, size_t expert_bytes, int weight_copies) {
        for (auto& value : heat_) value *= decay_;
        int32_t candidate = -1;
        for (const auto expert : routes) {
            if (expert < 0 || static_cast<size_t>(expert) >= heat_.size()) continue;
            ++heat_[expert];
            if (expert_to_slot[expert] < 0 && (candidate < 0 || heat_[expert] > heat_[candidate]))
                candidate = expert;
        }
        if (candidate < 0) return {};

        int32_t victim = -1;
        for (const auto expert : slot_to_expert_) {
            if (victim >= 0 && heat_[expert] >= heat_[victim]) continue;
            if (std::find(routes.begin(), routes.end(), expert) == routes.end()) victim = expert;
        }
        if (victim < 0 || heat_[candidate] < 1.5 * heat_[victim] + 2.0) return {};
        // The EWMA rate is (1 - decay) * heat. Discounting future tokens over the same
        // horizon gives decay * heat expected hits. Replacing the victim costs its hits.
        const double future_hits = decay_ * (heat_[candidate] - heat_[victim]);
        if (!moe_cache_move_pays(hw, expert_bytes, expert_to_slot.size_bytes(), weight_copies + 1, future_hits))
            return {};
        return Admission{candidate, expert_to_slot[victim]};
    }

    void commit(Admission move, std::span<int32_t> expert_to_slot, size_t weight_bytes) {
        auto& expert = slot_to_expert_[move.slot];
        expert_to_slot[expert] = -1;
        expert_to_slot[move.expert] = move.slot;
        expert = move.expert;
        ++moves_;
        moved_bytes_ += weight_bytes + expert_to_slot.size_bytes();
    }

    uint64_t moves() const { return moves_; }
    uint64_t moved_bytes() const { return moved_bytes_; }

private:
    std::vector<int32_t> slot_to_expert_;
    std::vector<double> heat_;
    double decay_;
    uint64_t moves_ = 0, moved_bytes_ = 0;
};

// A single-row prefill is indistinguishable from decode at the layer boundary.
class MoeDecodeScope {
public:
    explicit MoeDecodeScope(const void* context) : previous_(active_) { active_ = context; }
    ~MoeDecodeScope() { active_ = previous_; }
    MoeDecodeScope(const MoeDecodeScope&) = delete;
    MoeDecodeScope& operator=(const MoeDecodeScope&) = delete;

    static bool active_for(const void* context) { return active_ == context; }

private:
    const void* previous_;
    static inline thread_local const void* active_ = nullptr;
};

}  // namespace fusionrt
