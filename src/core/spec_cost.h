#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace fusionrt {

inline bool prefix_restore_worthwhile(double restore_ms, double recompute_ms) {
    return restore_ms >= 0 && std::isfinite(recompute_ms) && restore_ms < recompute_ms;
}

inline double prefix_copy_ms(std::size_t bytes, std::size_t copies, double gbps, double latency_us) {
    if (bytes == 0) return 0;
    if (gbps <= 0) return std::numeric_limits<double>::infinity();
    return static_cast<double>(bytes) / (gbps * 1e6) + copies * latency_us / 1000;
}

class MtpDepthController {
public:
    static constexpr int max_depth = 7;

    // Acceptance at depth d is conditional on reaching that proposal.
    void observe_acceptance(int depth, double probability) { acceptance_[depth - 1].add(probability); }
    void observe_verify(int width, double ms) { verify_[width - 1].add(ms); }
    void observe_draft(double ms) { draft_.add(ms); }
    void observe_rewind(int rows, double ms) { rewind_.add(ms / rows); }

    int best_depth(int cap) const {
        int best = 0;
        double best_rate = 0;
        double survival = 1;
        double accepted = 0;
        for (int k = 0; k <= cap; ++k) {
            if (k > 0) {
                survival *= acceptance_[k - 1].mean;
                accepted += survival;
            }
            if (!verify_[k].measured || (k > 0 && !draft_.measured)) continue;
            const double ms = k * draft_.mean + verify_[k].mean + (k - accepted) * rewind_.mean;
            if (ms <= 0) continue;
            const double rate = (1 + accepted) / ms;
            if (rate > best_rate) {
                best = k;
                best_rate = rate;
            }
        }
        return best;
    }

    int choose_depth(int cap) {
        const auto round = rounds_++;
        if (cap == 0 || round == 0) return 0;
        if (round == 1) return 1;
        const int best = best_depth(cap);
        constexpr unsigned exploration_interval = 8;
        if (round % exploration_interval != 0) return best;
        // Visit unmeasured widths even when a local optimum borders a slow width.
        // A confidence cutoff can prevent a width from being measured on a probe.
        for (int offset = 0; offset <= cap; ++offset) {
            const int k = (next_probe_ + offset) % (cap + 1);
            if (!verify_[k].measured) {
                next_probe_ = k + 1;
                return k;
            }
        }
        return std::clamp(best + ((round / exploration_interval) % 2 ? 1 : -1), 0, cap);
    }

private:
    struct Average {
        double mean = 0;
        bool measured = false;
        void add(double value) {
            constexpr double alpha = 0.125;
            mean = measured ? std::lerp(mean, value, alpha) : value;
            measured = true;
        }
    };
    std::array<Average, max_depth> acceptance_{};
    std::array<Average, max_depth + 1> verify_{};
    Average draft_;
    Average rewind_;
    uint64_t rounds_ = 0;
    int next_probe_ = 2;
};

}  // namespace fusionrt
