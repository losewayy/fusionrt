#pragma once

#include "cpu_moe.h"
#include <immintrin.h>

namespace fusionrt::cpu_moe_x86 {
namespace {

inline __m256i load_bytes(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }

inline int horizontal_sum(__m256i value) {
    auto sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

struct QuantGroup {
    __m256i values, scales;
    int bias_low, bias_high;

    int bias(const CpuQ8K& x, int part) const {
        return bias_low * x.sums[2 * part] + bias_high * x.sums[2 * part + 1];
    }
};

inline QuantGroup group(const BlockQ4_K& w, int part) {
    const auto packed = load_bytes(w.qs + part / 2 * 32);
    const auto q = _mm256_and_si256(part & 1 ? _mm256_srli_epi16(packed, 4) : packed, _mm256_set1_epi8(15));
    const auto sm = unpack_k_scale_min(w.scales, part);
    return {q, _mm256_set1_epi32(sm & 63), static_cast<int>(sm >> 8), static_cast<int>(sm >> 8)};
}

inline QuantGroup group(const BlockQ2_K& w, int part) {
    const auto packed = load_bytes(w.qs + (part / 4) * 32);
    const auto shifted = _mm256_srl_epi16(packed, _mm_cvtsi32_si128((part % 4) * 2));
    const auto q = _mm256_and_si256(shifted, _mm256_set1_epi8(3));
    const int lo = w.scales[2 * part], hi = w.scales[2 * part + 1];
    return {q, _mm256_set_m128i(_mm_set1_epi32(hi & 15), _mm_set1_epi32(lo & 15)), lo >> 4, hi >> 4};
}

inline QuantGroup group(const BlockQ6_K& w, int part) {
    const int half = part / 4;
    const auto packed = load_bytes(w.ql + half * 64 + (part & 1) * 32);
    const auto low = _mm256_and_si256(part & 2 ? _mm256_srli_epi16(packed, 4) : packed, _mm256_set1_epi8(15));
    const auto high = _mm256_srl_epi16(load_bytes(w.qh + half * 32), _mm_cvtsi32_si128((part & 3) * 2));
    const auto q = _mm256_or_si256(low, _mm256_slli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(3)), 4));
    const int lo = w.scales[2 * part], hi = w.scales[2 * part + 1];
    return {q, _mm256_set_m128i(_mm_set1_epi32(hi), _mm_set1_epi32(lo)), 32 * lo, 32 * hi};
}

template<class Block>
float finish(const Block& w, const CpuQ8K& x, int sum, int bias) {
    if constexpr (std::is_same_v<Block, BlockQ2_K> || std::is_same_v<Block, BlockQ4_K>)
        return x.scale * (fp16_to_fp32(w.d) * sum - fp16_to_fp32(w.dmin) * bias);
    else
        return (fp16_to_fp32(w.d) * x.scale) * static_cast<float>(sum - bias);
}

template<class Block, bool Vnni, int M>
void dot_k_rows(const uint8_t* row, const CpuQ8K* const* inputs, int cols, float* out) {
    const auto* blocks = reinterpret_cast<const Block*>(row);
    float totals[M]{};
    for (int i = 0; i < cols / 256; ++i) {
        const auto& w = blocks[i];
        __m256i acc[M]{};
        int bias[M]{};
        for (int part = 0; part < 8; ++part) {
            const auto q = group(w, part);
            for (int m = 0; m < M; ++m) {
                const auto& x = inputs[m][i];
                const auto a = load_bytes(x.values + part * 32);
                __m256i dot;
                if constexpr (Vnni) dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q.values, a);
                else dot = _mm256_madd_epi16(_mm256_maddubs_epi16(q.values, a), _mm256_set1_epi16(1));
                acc[m] = _mm256_add_epi32(acc[m], _mm256_mullo_epi32(dot, q.scales));
                bias[m] += q.bias(x, part);
            }
        }
        for (int m = 0; m < M; ++m)
            totals[m] += finish(w, inputs[m][i], horizontal_sum(acc[m]), bias[m]);
    }
    for (int m = 0; m < M; ++m) out[m] = totals[m];
}

template<class Block, bool Vnni>
void dot_k_dispatch(const uint8_t* row, const CpuQ8K* const* input, int cols, int rows, float* out) {
    switch (rows) {
        case 4: dot_k_rows<Block, Vnni, 4>(row, input, cols, out); break;
        case 3: dot_k_rows<Block, Vnni, 3>(row, input, cols, out); break;
        case 2: dot_k_rows<Block, Vnni, 2>(row, input, cols, out); break;
        case 1: dot_k_rows<Block, Vnni, 1>(row, input, cols, out); break;
    }
}

}  // namespace
}  // namespace fusionrt::cpu_moe_x86
