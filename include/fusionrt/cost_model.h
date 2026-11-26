#pragma once

#include "fusionrt/types.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace fusionrt {

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
