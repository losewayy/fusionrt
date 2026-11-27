#include "cpu_moe_x86_k.h"

namespace fusionrt {
namespace {

using namespace cpu_moe_x86;

__m512i join(__m256i lo, __m256i hi) {
    return _mm512_maskz_inserti64x4(0xff, _mm512_castsi256_si512(lo), hi, 1);
}

int sum_wide(__m512i value) {
    return horizontal_sum(_mm512_castsi512_si256(value)) +
           horizontal_sum(_mm512_maskz_extracti64x4_epi64(0x0f, value, 1));
}

template<class Block>
float dot_k(const uint8_t* row, const CpuQ8K* input, int cols) {
    float result;
    dot_k_rows<Block, true, 1>(row, &input, cols, &result);
    return result;
}

template<class Block>
float dot_k_wide(const uint8_t* row, const CpuQ8K* input, int cols) {
    const auto* blocks = reinterpret_cast<const Block*>(row);
    float result = 0.0f;
    for (int i = 0; i < cols / 256; ++i) {
        const auto& w = blocks[i];
        const auto& x = input[i];
        int bias = 0;
        auto acc = _mm512_setzero_si512();
        for (int part = 0; part < 8; part += 2) {
            const auto lo = group(w, part);
            const auto hi = group(w, part + 1);
            const auto a = _mm512_loadu_si512(x.values + part * 32);
            const auto dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), join(lo.values, hi.values), a);
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, join(lo.scales, hi.scales)));
            bias += lo.bias(x, part) + hi.bias(x, part + 1);
        }
        result += finish(w, x, sum_wide(acc), bias);
    }
    return result;
}

template<class Block>
float dot_small(const uint8_t* row, const CpuQ8Small* input, int cols) {
    const auto* blocks = reinterpret_cast<const Block*>(row);
    float result = 0.0f;
    for (int i = 0; i < cols / 32; ++i) {
        const auto& w = blocks[i];
        const auto& x = input[i];
        __m256i q;
        int zero_point;
        if constexpr (std::is_same_v<Block, BlockQ4_0>) {
            const auto packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));
            const auto nibble = _mm_set1_epi8(15);
            q = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(packed, 4), nibble), _mm_and_si128(packed, nibble));
            zero_point = 8;
        } else {
            q = _mm256_xor_si256(load_bytes(w.qs), _mm256_set1_epi8(-128));
            zero_point = 128;
        }
        const auto dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q, load_bytes(x.values));
        result += (fp16_to_fp32(w.d) * x.scale) * static_cast<float>(horizontal_sum(dot) - zero_point * x.sum);
    }
    return result;
}

}  // namespace

CpuMoeQ8Dot cpu_moe_avx512_q8_dot(DType dtype, bool wide) {
    if (dtype == DType::Q2_K) return wide ? dot_k_wide<BlockQ2_K> : dot_k<BlockQ2_K>;
    if (dtype == DType::Q4_K) return wide ? dot_k_wide<BlockQ4_K> : dot_k<BlockQ4_K>;
    if (dtype == DType::Q6_K) return wide ? dot_k_wide<BlockQ6_K> : dot_k<BlockQ6_K>;
    return nullptr;
}

CpuMoeQ8Rows cpu_moe_avx512_q8_rows(DType dtype) {
    if (dtype == DType::Q2_K) return dot_k_dispatch<BlockQ2_K, true>;
    if (dtype == DType::Q4_K) return dot_k_dispatch<BlockQ4_K, true>;
    if (dtype == DType::Q6_K) return dot_k_dispatch<BlockQ6_K, true>;
    return nullptr;
}

CpuMoeQ8SmallDot cpu_moe_avx512_small_dot(DType dtype) {
    if (dtype == DType::Q4_0) return dot_small<BlockQ4_0>;
    if (dtype == DType::Q8_0) return dot_small<BlockQ8_0>;
    return nullptr;
}

}  // namespace fusionrt
