#include "cpu_moe.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace fusionrt {

#ifdef __AVX2__
namespace {

float horizontal_sum(__m256 sum) {
    __m128 lanes = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    lanes = _mm_hadd_ps(lanes, lanes);
    return _mm_cvtss_f32(_mm_hadd_ps(lanes, lanes));
}

__m256 load_nibbles(const uint8_t* data, bool high) {
    auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
    if (high) bytes = _mm_srli_epi16(bytes, 4);
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(bytes, _mm_set1_epi8(15))));
}

float dot_f32(const uint8_t* row, const float* input, int cols) {
    const auto* weights = reinterpret_cast<const float*>(row);
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= cols; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(weights + i), _mm256_loadu_ps(input + i), sum);
    }
    float result = horizontal_sum(sum);
    for (; i < cols; ++i) result += weights[i] * input[i];
    return result;
}

float dot_q4_0(const uint8_t* row, const float* input, int cols) {
    const auto* blocks = reinterpret_cast<const BlockQ4_0*>(row);
    __m256 sum = _mm256_setzero_ps();
    const auto bias = _mm256_set1_ps(8.0f);
    for (int i = 0; i < cols; i += 32) {
        const auto& block = blocks[i / 32];
        const auto scale = _mm256_set1_ps(fp16_to_fp32(block.d));
        for (int j = 0; j < 32; j += 8) {
            const auto quant = load_nibbles(block.qs + (j & 15), j >= 16);
            const auto values = _mm256_mul_ps(_mm256_sub_ps(quant, bias), scale);
            sum = _mm256_fmadd_ps(values, _mm256_loadu_ps(input + i + j), sum);
        }
    }
    return horizontal_sum(sum);
}

float dot_q4_k(const uint8_t* row, const float* input, int cols) {
    const auto* blocks = reinterpret_cast<const BlockQ4_K*>(row);
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < cols; i += 256) {
        const auto& block = blocks[i / 256];
        const float d = fp16_to_fp32(block.d);
        const float dm = fp16_to_fp32(block.dmin);
        for (int sub = 0; sub < 8; ++sub) {
            const unsigned sm = unpack_k_scale_min(block.scales, sub);
            const auto scale = _mm256_set1_ps(d * static_cast<float>(sm & 63));
            const auto min = _mm256_set1_ps(dm * static_cast<float>(sm >> 8));
            const auto* packed = block.qs + sub / 2 * 32;
            const auto* x = input + i + sub * 32;
            for (int j = 0; j < 32; j += 8) {
                const auto quant = load_nibbles(packed + j, (sub & 1) != 0);
                const auto values = _mm256_sub_ps(_mm256_mul_ps(quant, scale), min);
                sum = _mm256_fmadd_ps(values, _mm256_loadu_ps(x + j), sum);
            }
        }
    }
    return horizontal_sum(sum);
}

float dot_q8_0(const uint8_t* row, const float* input, int cols) {
    const auto* blocks = reinterpret_cast<const BlockQ8_0*>(row);
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < cols; i += 32) {
        const auto& block = blocks[i / 32];
        const auto scale = _mm256_set1_ps(fp16_to_fp32(block.d));
        for (int j = 0; j < 32; j += 8) {
            const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.qs + j));
            const auto quant = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            sum = _mm256_fmadd_ps(_mm256_mul_ps(quant, scale), _mm256_loadu_ps(input + i + j), sum);
        }
    }
    return horizontal_sum(sum);
}

}  // namespace
#endif

CpuMoeDot cpu_moe_avx2_dot(DType dtype) {
#ifdef __AVX2__
    switch (dtype) {
        case DType::F32: return dot_f32;
        case DType::Q4_0: return dot_q4_0;
        case DType::Q4_K: return dot_q4_k;
        case DType::Q8_0: return dot_q8_0;
        default: break;
    }
#else
    (void)dtype;
#endif
    return nullptr;
}

}  // namespace fusionrt
