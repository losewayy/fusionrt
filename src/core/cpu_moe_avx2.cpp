#include "cpu_moe.h"

#ifdef __AVX2__
#include "cpu_moe_x86_k.h"
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

float dot_q6_k(const uint8_t* row, const float* input, int cols) {
    const auto* blocks = reinterpret_cast<const BlockQ6_K*>(row);
    const auto low_mask = _mm_set1_epi8(15);
    const auto high_mask = _mm_set1_epi8(3);
    const auto bias = _mm_set1_epi8(32);
    __m256 lo_sum = _mm256_setzero_ps(), hi_sum = _mm256_setzero_ps();
    for (int i = 0; i < cols; i += 256) {
        const auto& block = blocks[i / 256];
        const float d = fp16_to_fp32(block.d);
        for (int sub = 0; sub < 16; ++sub) {
            const int half = sub / 8;
            const int part = (sub / 2) % 4;
            const int lane = (sub % 2) * 16;
            const auto* ql = block.ql + half * 64 + (part % 2) * 32 + lane;
            const auto* qh = block.qh + half * 32 + lane;
            auto low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql));
            const auto high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh));
            if (part >= 2) low = _mm_srli_epi16(low, 4);
            const auto high_bits = _mm_and_si128(_mm_srl_epi16(high, _mm_cvtsi32_si128(part * 2)), high_mask);
            const auto values = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(low, low_mask), _mm_slli_epi16(high_bits, 4)), bias);
            const auto scale = _mm256_set1_ps(d * static_cast<float>(block.scales[sub]));
            const auto lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values));
            const auto hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(values, 8)));
            const float* x = input + i + sub * 16;
            lo_sum = _mm256_fmadd_ps(_mm256_mul_ps(lo, scale), _mm256_loadu_ps(x), lo_sum);
            hi_sum = _mm256_fmadd_ps(_mm256_mul_ps(hi, scale), _mm256_loadu_ps(x + 8), hi_sum);
        }
    }
    return horizontal_sum(_mm256_add_ps(lo_sum, hi_sum));
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

int sum_i32(__m256i sum) {
    auto lanes = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    lanes = _mm_hadd_epi32(lanes, lanes);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(lanes, lanes));
}

float dot_q2_k_q8(const uint8_t* row, const CpuQ8K* input, int cols) {
    float value;
    cpu_moe_x86::dot_k_rows<BlockQ2_K, false, 1>(row, &input, cols, &value);
    return value;
}

float dot_q4_k_q8(const uint8_t* row, const CpuQ8K* input, int cols) {
    const auto* blocks = reinterpret_cast<const BlockQ4_K*>(row);
    const auto nibble = _mm256_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    float result = 0;
    for (int i = 0; i < cols / 256; ++i) {
        const auto& w = blocks[i];
        const auto& x = input[i];
        auto sum = _mm256_setzero_si256();
        int bias = 0;
        for (int pair = 0; pair < 4; ++pair) {
            const auto packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.qs + pair * 32));
            for (int j = 0; j < 2; ++j) {
                const int sub = 2 * pair + j;
                const auto q = _mm256_and_si256(j == 0 ? packed : _mm256_srli_epi16(packed, 4), nibble);
                const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.values + sub * 32));
                const unsigned sm = unpack_k_scale_min(w.scales, sub);
                const auto dot = _mm256_madd_epi16(_mm256_maddubs_epi16(q, a), ones);
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(dot, _mm256_set1_epi32(sm & 63)));
                bias += static_cast<int>(sm >> 8) * (x.sums[2 * sub] + x.sums[2 * sub + 1]);
            }
        }
        result += x.scale * (fp16_to_fp32(w.d) * sum_i32(sum) - fp16_to_fp32(w.dmin) * bias);
    }
    return result;
}

float dot_q6_k_q8(const uint8_t* row, const CpuQ8K* input, int cols) {
    const auto* blocks = reinterpret_cast<const BlockQ6_K*>(row);
    const auto nibble = _mm256_set1_epi8(15), two_bits = _mm256_set1_epi8(3);
    float result = 0;
    for (int i = 0; i < cols / 256; ++i) {
        const auto& w = blocks[i];
        const auto& x = input[i];
        auto sum = _mm256_setzero_si256();
        int bias = 0;
        for (int half = 0; half < 2; ++half) {
            const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.qh + half * 32));
            for (int part = 0; part < 4; ++part) {
                const auto packed = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(w.ql + half * 64 + (part % 2) * 32));
                const auto low = _mm256_and_si256(part < 2 ? packed : _mm256_srli_epi16(packed, 4), nibble);
                const auto bits = _mm256_and_si256(_mm256_srl_epi16(high, _mm_cvtsi32_si128(part * 2)), two_bits);
                const auto q = _mm256_or_si256(low, _mm256_slli_epi16(bits, 4));
                const int sub = half * 8 + part * 2;
                const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.values + sub * 16));
                const int lo = w.scales[sub], hi = w.scales[sub + 1];
                const auto scale = _mm256_set_m128i(_mm_set1_epi16(static_cast<short>(hi)), _mm_set1_epi16(static_cast<short>(lo)));
                // The unsigned six-bit codes have a zero point of 32.
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(q, a), scale));
                bias += lo * x.sums[sub] + hi * x.sums[sub + 1];
            }
        }
        result += (fp16_to_fp32(w.d) * x.scale) * static_cast<float>(sum_i32(sum) - 32 * bias);
    }
    return result;
}

}  // namespace
#endif

CpuMoeDot cpu_moe_avx2_dot(DType dtype) {
#ifdef __AVX2__
    switch (dtype) {
        case DType::F32: return dot_f32;
        case DType::Q4_0: return dot_q4_0;
        case DType::Q4_K: return dot_q4_k;
        case DType::Q6_K: return dot_q6_k;
        case DType::Q8_0: return dot_q8_0;
        default: break;
    }
#else
    (void)dtype;
#endif
    return nullptr;
}

CpuMoeQ8Dot cpu_moe_avx2_q8_dot(DType dtype) {
#ifdef __AVX2__
    if (dtype == DType::Q2_K) return dot_q2_k_q8;
    if (dtype == DType::Q4_K) return dot_q4_k_q8;
    if (dtype == DType::Q6_K) return dot_q6_k_q8;
#else
    (void)dtype;
#endif
    return nullptr;
}

CpuMoeQ8Rows cpu_moe_avx2_q8_rows(DType dtype) {
#ifdef __AVX2__
    if (dtype == DType::Q2_K) return cpu_moe_x86::dot_k_dispatch<BlockQ2_K, false>;
    if (dtype == DType::Q4_K) return cpu_moe_x86::dot_k_dispatch<BlockQ4_K, false>;
    if (dtype == DType::Q6_K) return cpu_moe_x86::dot_k_dispatch<BlockQ6_K, false>;
#else
    (void)dtype;
#endif
    return nullptr;
}

void cpu_moe_avx2_quantize_q8(const float* input, CpuQ8K* output, int cols) {
#ifdef __AVX2__
    const auto sign_clear = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const auto bias = _mm_set1_epi8(-128);
    const auto infinity = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
    const auto finite_load = [&](const float* p) {
        const auto x = _mm256_loadu_ps(p);
        const auto finite = _mm256_cmp_ps(_mm256_and_ps(x, sign_clear), infinity, _CMP_LT_OQ);
        return _mm256_and_ps(x, finite);
    };
    for (int i = 0; i < cols; i += 256) {
        auto& block = output[i / 256];
        auto maxima = _mm256_setzero_ps();
        for (int j = 0; j < 256; j += 8)
            maxima = _mm256_max_ps(maxima, _mm256_and_ps(finite_load(input + i + j), sign_clear));
        auto peak = _mm_max_ps(_mm256_castps256_ps128(maxima), _mm256_extractf128_ps(maxima, 1));
        peak = _mm_max_ps(peak, _mm_movehl_ps(peak, peak));
        peak = _mm_max_ss(peak, _mm_shuffle_ps(peak, peak, _MM_SHUFFLE(0, 0, 0, 1)));
        const float max = _mm_cvtss_f32(peak);
        block.scale = max / 127.0f;
        const auto inv = _mm256_set1_ps(max > 0 ? 127.0f / max : 0.0f);
        for (int sub = 0; sub < 16; ++sub) {
            const float* x = input + i + sub * 16;
            const auto a = _mm256_cvtps_epi32(_mm256_mul_ps(finite_load(x), inv));
            const auto b = _mm256_cvtps_epi32(_mm256_mul_ps(finite_load(x + 8), inv));
            const auto lo = _mm_packs_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
            const auto hi = _mm_packs_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
            const auto packed = _mm_packs_epi16(lo, hi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(block.values + sub * 16), packed);
            const auto sum = _mm_sad_epu8(_mm_xor_si128(packed, bias), _mm_setzero_si128());
            block.sums[sub] = static_cast<int16_t>(_mm_cvtsi128_si32(sum) + _mm_extract_epi16(sum, 4) - 16 * 128);
        }
    }
#else
    (void)input; (void)output; (void)cols;
#endif
}

}  // namespace fusionrt
