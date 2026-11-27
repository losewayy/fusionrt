#pragma once

#include "fusionrt/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <vector>

namespace fusionrt {

constexpr size_t wddm_residency_reserve(size_t budget) {
    // Leave room for residency changes and allocations outside the model arenas.
    return budget / 8;
}

constexpr size_t wddm_model_available(size_t budget, size_t usage, size_t reclaimable) {
    const auto resident_limit = budget - wddm_residency_reserve(budget);
    usage -= std::min(usage, reclaimable);
    return resident_limit - std::min(resident_limit, usage);
}

struct CostHardware {
    int sm_count = 0;
    double hbm_bw_gbps = 0;
    double int8_tc_tflops = 0;
    double fp16_tc_tflops = 0;
    size_t l2_bytes = 0;
    double cpu_q4_gemv_gbps = 0;
    double pcie_h2d_gbps = 0;
    double pcie_latency_us = 0;
    double l2_bw_gbps = 0;
    double sm_clock_ghz = 0;
    int shared_bytes_per_sm = 100 * 1024;
    int shared_bytes_per_block = 96 * 1024;
};

struct MoeLayerCost {
    int experts = 0, top_k = 0, hidden = 0;
    std::array<size_t, 3> expert_tensor_bytes{};
    bool mapped_slots = false;

    size_t gpu_bytes(int resident) const {
        size_t bytes = 0;
        for (const auto unit : expert_tensor_bytes) bytes += (unit * resident + 255) & ~size_t{255};
        if (mapped_slots && resident > 0) bytes += (experts * sizeof(int32_t) + 255) & ~size_t{255};
        return bytes;
    }
};

struct MoePlacement {
    std::vector<int> gpu_experts;
    std::array<double, 4> candidate_us{};
    int choice = 0;
    std::vector<std::vector<int32_t>> ranked_experts;
};

struct MoeLayerHeat {
    std::vector<int32_t> ranked_experts;
    std::vector<double> cumulative_hits;

    explicit MoeLayerHeat(std::span<const double> frequencies) {
        ranked_experts.resize(frequencies.size());
        std::iota(ranked_experts.begin(), ranked_experts.end(), 0);
        std::stable_sort(ranked_experts.begin(), ranked_experts.end(),
            [&](int lhs, int rhs) { return frequencies[lhs] > frequencies[rhs]; });
        cumulative_hits.resize(frequencies.size() + 1);
        const double total = std::accumulate(frequencies.begin(), frequencies.end(), 0.0);
        for (size_t i = 0; i < frequencies.size(); ++i)
            cumulative_hits[i + 1] = cumulative_hits[i] + (total > 0 ?
                frequencies[ranked_experts[i]] / total : 1.0 / frequencies.size());
        if (!frequencies.empty()) cumulative_hits.back() = 1.0;
    }
};

namespace cost_detail {

inline auto moe_legacy_counts(std::span<const MoeLayerCost> layers, size_t available) {
    std::array<std::vector<int>, 3> counts;
    for (auto& candidate : counts) candidate.resize(layers.size());
    int common = 0, maximum = 0;
    for (const auto& layer : layers) maximum = std::max(maximum, layer.experts);
    auto uniform_bytes = [&](int resident) {
        size_t bytes = 0;
        for (const auto& layer : layers) bytes += layer.gpu_bytes(std::min(layer.experts, resident));
        return bytes;
    };
    for (int hi = maximum; common < hi;) {
        const int n = common + (hi - common + 1) / 2;
        if (uniform_bytes(n) <= available) common = n;
        else hi = n - 1;
    }
    size_t remainder = available - uniform_bytes(common);
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        auto& n = counts[0][i];
        n = std::min(common, layer.experts);
        if (n < layer.experts) {
            const auto extra = layer.gpu_bytes(n + 1) - layer.gpu_bytes(n);
            if (extra <= remainder) {
                ++n;
                remainder -= extra;
            }
        }
    }

    remainder = available;
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        const auto full = layer.gpu_bytes(layer.experts);
        if (full <= remainder) {
            counts[1][i] = counts[2][i] = layer.experts;
            remainder -= full;
        } else {
            while (counts[2][i] < layer.experts && layer.gpu_bytes(counts[2][i] + 1) <= remainder)
                ++counts[2][i];
            break;
        }
    }

    return counts;
}

inline double moe_layer_us(const MoeLayerCost& l, int gpu, double bytes, double hit, double all_hot,
                           const CostHardware& hw, double dispatch_us) {
    if (!l.experts) return 0.0;
    const double hot = l.top_k * hit;
    const double cpu_bw = std::max(1.0, hw.cpu_q4_gemv_gbps) * 1000;
    const double gpu_bw = std::max(1.0, hw.hbm_bw_gbps) * 1000;
    const double pci_bw = std::max(1.0, hw.pcie_h2d_gbps) * 1000;
    const double serial = gpu < l.experts ? hw.pcie_latency_us +
        (l.hidden * sizeof(float) + l.top_k * (sizeof(float) + sizeof(int32_t))) / pci_bw : 0.0;
    const double cpu = (l.top_k - hot) * bytes / cpu_bw + (1.0 - all_hot) * dispatch_us;
    return serial + std::max(cpu, hot * bytes / gpu_bw) +
        (1.0 - all_hot) * (hw.pcie_latency_us + l.hidden * sizeof(float) / pci_bw);
}

}  // namespace cost_detail

inline MoePlacement select_moe_placement(std::span<const MoeLayerCost> layers, size_t available,
                                        const CostHardware& hw, double cpu_dispatch_us) {
    auto counts = cost_detail::moe_legacy_counts(layers, available);
    MoePlacement result;
    for (size_t candidate = 0; candidate < counts.size(); ++candidate) {
        auto& us = result.candidate_us[candidate];
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& l = layers[i];
            if (l.experts == 0) continue;
            const int gpu = counts[candidate][i];
            const double bytes = static_cast<double>(l.gpu_bytes(l.experts)) / l.experts;
            double all_hot = 1.0;
            for (int k = 0; k < l.top_k; ++k)
                all_hot *= static_cast<double>(std::max(0, gpu - k)) / (l.experts - k);
            us += cost_detail::moe_layer_us(l, gpu, bytes, static_cast<double>(gpu) / l.experts, all_hot,
                                          hw, cpu_dispatch_us);
        }
        if (us < result.candidate_us[result.choice]) result.choice = static_cast<int>(candidate);
    }
    result.gpu_experts = std::move(counts[result.choice]);
    return result;
}

inline MoePlacement select_moe_placement(std::span<const MoeLayerCost> layers, size_t available,
    const CostHardware& hw, double dispatch_us, std::span<const MoeLayerHeat> heat, bool balanced = false) {
    if (heat.empty()) return select_moe_placement(layers, available, hw, dispatch_us);
    if (heat.size() != layers.size()) throw std::invalid_argument("MoE heat profile layer count mismatch");
    for (size_t i = 0; i < layers.size(); ++i)
        if (heat[i].ranked_experts.size() != static_cast<size_t>(layers[i].experts))
            throw std::invalid_argument("MoE heat profile expert count mismatch");
    auto legacy = cost_detail::moe_legacy_counts(layers, available);
    auto layer_us = [&](size_t i, int n) {
        const double hit = heat[i].cumulative_hits[n];
        // The histogram gives the marginal hit probability, not the joint top-k distribution.
        const double all_hot = n >= layers[i].top_k ? std::pow(hit, layers[i].top_k) : 0.0;
        const auto& l = layers[i];
        const double bytes = std::accumulate(l.expert_tensor_bytes.begin(), l.expert_tensor_bytes.end(), 0.0);
        return cost_detail::moe_layer_us(l, n, bytes, hit, all_hot, hw, dispatch_us);
    };
    auto score = [&](const std::vector<int>& counts) {
        double total = 0.0;
        for (size_t i = 0; i < layers.size(); ++i) total += layer_us(i, counts[i]);
        return total;
    };
    MoePlacement result;
    for (size_t j = 0; j < legacy.size(); ++j) {
        result.candidate_us[j] = score(legacy[j]);
        if (!balanced && result.candidate_us[j] < result.candidate_us[result.choice])
            result.choice = static_cast<int>(j);
    }

    // Score each next slot by expected cold-expert time saved per allocated byte.
    // Without a trace, reserve the same baseline count in every layer that can afford it.
    auto counts = balanced ? legacy[0] : std::vector<int>(layers.size());
    using Gain = std::pair<double, size_t>;
    std::priority_queue<Gain> next;
    auto push = [&](size_t i) {
        const int n = counts[i];
        const auto& l = layers[i];
        if (n == l.experts) return;
        const size_t bytes = l.gpu_bytes(n + 1) - l.gpu_bytes(n);
        const double hit_gain = heat[i].cumulative_hits[n + 1] - heat[i].cumulative_hits[n];
        const double compute_saved = l.top_k * hit_gain *
            std::accumulate(l.expert_tensor_bytes.begin(), l.expert_tensor_bytes.end(), 0.0);
        next.emplace(compute_saved / std::max<size_t>(1, bytes), i);
    };
    size_t remaining = available;
    for (size_t i = 0; i < layers.size(); ++i) {
        remaining -= layers[i].gpu_bytes(counts[i]);
        push(i);
    }
    while (!next.empty()) {
        const size_t i = next.top().second;
        next.pop();
        auto& n = counts[i];
        const size_t bytes = layers[i].gpu_bytes(n + 1) - layers[i].gpu_bytes(n);
        if (bytes > remaining) continue;
        remaining -= bytes;
        ++n;
        push(i);
    }
    result.candidate_us[3] = score(counts);
    if (result.candidate_us[3] <= result.candidate_us[result.choice]) {
        result.choice = 3;
        result.gpu_experts = std::move(counts);
    } else {
        result.gpu_experts = std::move(legacy[result.choice]);
    }
    for (const auto& layer : heat) result.ranked_experts.push_back(layer.ranked_experts);
    return result;
}

inline bool moe_cache_move_pays(const CostHardware& hw, size_t expert_bytes, size_t mapping_bytes,
                               int copies, double expected_future_hits) {
    if (!expert_bytes || !(hw.pcie_h2d_gbps > 0) || !(hw.cpu_q4_gemv_gbps > 0)) return false;
    const double move_us = (static_cast<double>(expert_bytes) + mapping_bytes) / (hw.pcie_h2d_gbps * 1000) +
        copies * hw.pcie_latency_us;
    const double miss_us = static_cast<double>(expert_bytes) / (hw.cpu_q4_gemv_gbps * 1000);
    return move_us < expected_future_hits * miss_us;
}

struct MoeKvPlacement {
    DType dtype;
    MoePlacement experts;
};

inline MoeKvPlacement select_moe_kv_precision(std::span<const MoeLayerCost> layers, size_t available,
    size_t arena_limit, size_t f16_bytes, size_t q8_bytes, const CostHardware& hw, double dispatch_us,
    std::span<const MoeLayerHeat> heat, bool balanced) {
    auto placement = [&](size_t kv) {
        return select_moe_placement(layers, std::min(available - kv, arena_limit), hw, dispatch_us, heat, balanced);
    };
    if (f16_bytes > available) {
        if (q8_bytes <= available) return {DType::Q8_0, placement(q8_bytes)};
        return {DType::F16, {}};
    }
    auto half = placement(f16_bytes);
    const auto resident = [](const MoePlacement& p) {
        return std::accumulate(p.gpu_experts.begin(), p.gpu_experts.end(), 0);
    };
    const int total = std::accumulate(layers.begin(), layers.end(), 0,
        [](int n, const MoeLayerCost& l) { return n + l.experts; });
    if (resident(half) < total && q8_bytes < f16_bytes) {
        auto quant = placement(q8_bytes);
        if (resident(quant) > resident(half) && quant.candidate_us[quant.choice] < half.candidate_us[half.choice])
            return {DType::Q8_0, std::move(quant)};
    }
    return {DType::F16, std::move(half)};
}

inline MoeKvPlacement select_moe_kv_precision(std::span<const MoeLayerCost> layers, size_t available,
    size_t arena_limit, size_t f16_bytes, size_t q8_bytes, const CostHardware& hw, double dispatch_us) {
    return select_moe_kv_precision(layers, available, arena_limit, f16_bytes, q8_bytes, hw, dispatch_us, {}, false);
}

enum class MmqTileConfig : uint8_t {
    M16N32, M32N32, M32N64, M64N32, M64N64, M64N64K256,
    M64N64W1, M64N128, M128N64, M128N96, M128N128
};

namespace cost_detail {

struct MmqTile {
    int m, n;
    int warp_m, warp_n;  // 16x8 MMA fragments per warp
    int k_groups;       // 32-element quant groups per pipeline stage
};

inline constexpr MmqTile mmq_tiles[] = {
    { 16,  32, 1, 1, 8}, { 32,  32, 2, 1, 8}, { 32,  64, 2, 2, 4},
    { 64,  32, 4, 1, 4}, { 64,  64, 2, 2, 4}, { 64,  64, 2, 2, 8},
    { 64,  64, 4, 1, 4}, { 64, 128, 4, 2, 4}, {128,  64, 2, 2, 4},
    {128,  96, 4, 2, 4}, {128, 128, 4, 2, 4},
};

constexpr int64_t ceil_div(int64_t n, int64_t d) {
    return n / d + (n % d != 0);
}

inline double l2_bandwidth(const CostHardware& hw, double hbm) {
    const double estimated = 32 * std::max(1, hw.sm_count) * hw.sm_clock_ghz;
    return std::max(hbm, hw.l2_bw_gbps > 0 ? hw.l2_bw_gbps : estimated);
}

}  // namespace cost_detail

inline MmqTileConfig select_mmq_tile(const CostHardware& hw, int M, int N, int K,
                                     size_t bytes_per_256, bool supports_128_tile, bool supports_async,
                                     unsigned compiled_tiles = ~0u, const int* shared_bytes = nullptr) {
    using namespace cost_detail;
    if (M <= 0 || N <= 0 || K <= 0) return MmqTileConfig::M16N32;

    const int sm_count = std::max(1, hw.sm_count);
    const double hbm = std::max(1.0, hw.hbm_bw_gbps);
    const double l2 = l2_bandwidth(hw, hbm);
    const double int8_tc = std::max(0.001, hw.int8_tc_tflops);
    const double x_row = K / 32.0 * sizeof(BlockQ8_1);
    const double w_row = K / 256.0 * static_cast<double>(bytes_per_256);
    const double x_bytes = M * x_row;
    const double w_bytes = N * w_row;
    const double output_bytes = static_cast<double>(M) * N * sizeof(float);
    const double cache_bytes = static_cast<double>(hw.l2_bytes);

    auto best = MmqTileConfig::M16N32;
    double best_us = std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < sizeof(mmq_tiles) / sizeof(mmq_tiles[0]); ++i) {
        const auto& tile = mmq_tiles[i];
        if (!(compiled_tiles & (1u << i)) || ((tile.m == 128 || tile.n == 128) && !supports_128_tile)) continue;

        const int shared = shared_bytes ? shared_bytes[i] : 2 * tile.k_groups * (tile.m + tile.n) * 40;
        if (shared > hw.shared_bytes_per_block) continue;
        const int warps = tile.m / (16 * tile.warp_m) * (tile.n / (8 * tile.warp_n));
        // Two ready warps per each of four warp schedulers can hide issue latency.
        constexpr int issue_warps = 8;
        const int occ = std::max(1, std::min(hw.shared_bytes_per_sm / shared,
                                            (issue_warps + warps - 1) / warps));
        const int64_t m_tiles = ceil_div(M, tile.m);
        const int64_t n_tiles = ceil_div(N, tile.n);
        const int64_t tiles = m_tiles * n_tiles;
        const int64_t slots = static_cast<int64_t>(sm_count) * occ;
        const double wave_eff = static_cast<double>(tiles) / (ceil_div(tiles, slots) * static_cast<double>(slots));
        const double issue_eff = std::min(1.0, static_cast<double>(warps * occ) / issue_warps);

        // All CTAs stage their operands. Repeated reads can hit L2 when the
        // operand fits; otherwise only the activation is likely to be reused.
        const double x_staged = x_bytes * static_cast<double>(n_tiles);
        const double w_staged = w_bytes * static_cast<double>(m_tiles);
        const double x_hbm = x_bytes <= cache_bytes ? x_bytes : x_staged;
        const double w_hbm = w_bytes <= cache_bytes ? w_bytes : w_staged;
        const double hbm_us = (x_hbm + w_hbm + output_bytes) / (hbm * 1e3);
        const double l2_us = (x_staged + w_staged + output_bytes) / (l2 * 1e3);
        // Charge padded MMA work and amortize A/B fragment loads over reuse
        // along the two warp dimensions.
        const double flops = 2.0 * tile.m * tile.n * K * static_cast<double>(tiles);
        const double mma_us = flops / (int8_tc * 1e6) *
            (1.0 + 0.5 / tile.warp_m + 0.5 / tile.warp_n) / issue_eff;
        // Async staging hides DRAM latency, but staging and MMA operand
        // loads still share on-chip resources.
        const double pipeline_us = supports_async ? std::max(hbm_us, l2_us + mma_us)
                                                   : std::max(hbm_us, l2_us) + mma_us;
        // Each stage ends with a CTA barrier. Account for its warp arrivals.
        const double sync_us = hw.sm_clock_ghz > 0
            ? static_cast<double>(ceil_div(K, 32 * tile.k_groups) * warps) / (hw.sm_clock_ghz * 1e3)
            : 0.0;
        const double time_us = (pipeline_us + sync_us * static_cast<double>(tiles) / sm_count) / wave_eff;
        if (time_us < best_us) {
            best_us = time_us;
            best = static_cast<MmqTileConfig>(i);
        }
    }
    return best;
}

inline bool should_repack_q6_a(const CostHardware& hw, int64_t N, int64_t K, bool is_vocab_head) {
    if (N <= 0 || K <= 0 || K % 256 != 0) return false;
    const double blocks = static_cast<double>(N) * (K / 256.0);
    const double hbm = std::max(1.0, hw.hbm_bw_gbps);
    const double decode_tax = blocks * (sizeof(BlockQ6_A) - sizeof(BlockQ6_K)) / (hbm * 1e9);
    // Per 32 weights, raw Q6 gathers four scalar words for each of four
    // lane groups. Q6_A issues two vector loads. The output head uses GEMV.
    constexpr size_t gathered_bytes = 8 * 4 * 4 * sizeof(uint32_t);
    constexpr size_t vector_bytes = 8 * 2 * 16;
    const double prefill_savings = is_vocab_head ? 0 :
        blocks * (gathered_bytes - vector_bytes) / (cost_detail::l2_bandwidth(hw, hbm) * 1e9);
    return prefill_savings > decode_tax;
}

inline int select_decode_splits(const CostHardware& hw, int kv_len, int n_kv_heads, int max_splits) {
    using cost_detail::ceil_div;
    if (kv_len <= 0 || n_kv_heads <= 0 || max_splits <= 1) return 1;
    constexpr int page_tokens = 128;
    if (hw.sm_count <= 0) return static_cast<int>(std::min<int64_t>(max_splits, ceil_div(kv_len, page_tokens)));

    // Bucket the context so decode graphs can reuse the same reduction size.
    // Each target split processes several 32-token attention tiles.
    constexpr int split_tokens = 2 * page_tokens;
    constexpr int bucket_tokens = 16 * split_tokens;
    const int64_t window = bucket_tokens * static_cast<int64_t>(
        std::bit_ceil(static_cast<unsigned>(ceil_div(kv_len, bucket_tokens))));
    // Short reductions have little KV traffic per CTA and pay for every
    // partial in the merge. Longer contexts benefit from more resident work.
    const int64_t ctas_per_sm = std::min<int64_t>(4, 1 + window / bucket_tokens);
    const int64_t fill = ceil_div(ctas_per_sm * hw.sm_count, n_kv_heads);
    return static_cast<int>(std::min({static_cast<int64_t>(max_splits), window / split_tokens,
                                      std::max(fill, window / (2 * split_tokens))}));
}

enum class ColdExpertPath { CpuInPlace, PcieStream };

inline ColdExpertPath select_cold_expert_path(const CostHardware& hw, size_t expert_bytes) {
    if (expert_bytes == 0 || hw.cpu_q4_gemv_gbps <= 0 || hw.pcie_h2d_gbps <= 0 || hw.hbm_bw_gbps <= 0) {
        return ColdExpertPath::CpuInPlace;
    }
    const double bytes = static_cast<double>(expert_bytes);
    const double cpu_s = bytes / (hw.cpu_q4_gemv_gbps * 1e9);
    const double pcie_s = hw.pcie_latency_us * 1e-6 + bytes / (hw.pcie_h2d_gbps * 1e9) +
                          bytes / (hw.hbm_bw_gbps * 1e9);
    return pcie_s < cpu_s ? ColdExpertPath::PcieStream : ColdExpertPath::CpuInPlace;
}

}  // namespace fusionrt
