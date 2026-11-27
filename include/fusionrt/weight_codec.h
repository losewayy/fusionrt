#pragma once

#include "fusionrt/types.h"
#include <bit>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#define FUSIONRT_CODEC_INLINE __device__ __forceinline__
#define FUSIONRT_CODEBOOK(type, name, count) static __device__ const type name[count] = {
#else
#define FUSIONRT_CODEC_INLINE inline
#define FUSIONRT_CODEBOOK(type, name, count) inline constexpr type name[count] = {
#endif

namespace fusionrt {
namespace codec_detail {

#include "fusionrt/detail/weight_codebooks.inc"
#undef FUSIONRT_CODEBOOK

FUSIONRT_CODEC_INLINE uint16_t le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

FUSIONRT_CODEC_INLINE uint32_t le32(const uint8_t* p) {
    return le16(p) | (static_cast<uint32_t>(le16(p + 2)) << 16);
}

FUSIONRT_CODEC_INLINE float half(const uint8_t* p) {
#ifdef __CUDACC__
    return __half2float(__ushort_as_half(le16(p)));
#else
    return fp16_to_fp32(le16(p));
#endif
}

FUSIONRT_CODEC_INLINE float e8m0_half(uint8_t exponent) {
    const uint32_t bits = exponent < 2 ? 0x00200000u << exponent :
        static_cast<uint32_t>(exponent - 1) << 23;
#ifdef __CUDACC__
    return __uint_as_float(bits);
#else
    return std::bit_cast<float>(bits);
#endif
}

FUSIONRT_CODEC_INLINE int sign(uint8_t bits, int i) {
    return (bits >> i & 1) ? -1 : 1;
}

FUSIONRT_CODEC_INLINE int nibble(const uint8_t* p, int i) {
    return (p[i % 16] >> (4 * (i / 16))) & 15;
}

FUSIONRT_CODEC_INLINE int ternary(uint8_t packed, int digit) {
    constexpr int pow3[] = {1, 3, 9, 27, 81};
    const auto wrapped = static_cast<uint8_t>(packed * pow3[digit]);
    return ((wrapped * 3) >> 8) - 1;
}

}  // namespace codec_detail

// Block data is always little-endian. Access by byte so mapped and tiled rows
// can be read directly, including the 17-byte MXFP4 blocks.
template<DType Format>
struct WeightCodec {
    static_assert(is_weight_codec_dtype(Format));
    static constexpr int kBlockSize = static_cast<int>(dtype_block_size(Format));
    static constexpr int kBlockBytes = static_cast<int>(dtype_type_size(Format));

    static FUSIONRT_CODEC_INLINE float value(const uint8_t* b, int i) {
        using namespace codec_detail;
        if constexpr (Format == DType::Q3_K) {
            const int group = i / 16;
            const int low = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
            const int high = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
            const int quant = ((b[32 + 32 * (i / 128) + i % 32] >> (2 * (i / 32 % 4))) & 3)
                - ((b[i % 32] >> (i / 32) & 1) ? 0 : 4);
            return (half(b + 108) * ((low | (high << 4)) - 32)) * quant;
        } else if constexpr (Format == DType::IQ2_XXS) {
            const auto* group = b + 2 + (i / 32) * 8;
            const unsigned aux = le32(group + 4);
            const int sub = i / 8 % 4;
            const auto grid = iq2xxs_grid[group[sub]];
            const auto signs = ksigns_iq2xs[(aux >> (7 * sub)) & 127];
            const float scale = half(b) * (0.5f + (aux >> 28)) * 0.25f;
            return scale * static_cast<uint8_t>(grid >> (8 * (i % 8))) * sign(signs, i % 8);
        } else if constexpr (Format == DType::IQ2_XS) {
            const unsigned packed = le16(b + 2 + (i / 8) * 2);
            const auto grid = iq2xs_grid[packed & 511];
            const unsigned scale = (b[66 + i / 32] >> (4 * (i / 16 % 2))) & 15;
            return (half(b) * (0.5f + scale) * 0.25f)
                * static_cast<uint8_t>(grid >> (8 * (i % 8))) * sign(ksigns_iq2xs[packed >> 9], i % 8);
        } else if constexpr (Format == DType::IQ3_XXS) {
            const unsigned aux = le32(b + 66 + (i / 32) * 4);
            const auto grid = iq3xxs_grid[b[2 + i / 4]];
            const auto signs = ksigns_iq2xs[(aux >> (7 * (i / 8 % 4))) & 127];
            return (half(b) * (0.5f + (aux >> 28)) * 0.5f)
                * static_cast<uint8_t>(grid >> (8 * (i % 4))) * sign(signs, i % 8);
        } else if constexpr (Format == DType::IQ3_S) {
            const int group = i / 32;
            const int index = b[2 + i / 4] | (((b[66 + group] >> (i / 4 % 8)) & 1) << 8);
            const int scale = (b[106 + group / 2] >> (4 * (group % 2))) & 15;
            return (half(b) * (1 + 2 * scale))
                * static_cast<uint8_t>(iq3s_grid[index] >> (8 * (i % 4))) * sign(b[74 + i / 8], i % 8);
        } else if constexpr (Format == DType::IQ4_NL) {
            return half(b) * kvalues_iq4nl[nibble(b + 2, i)];
        } else if constexpr (Format == DType::IQ4_XS) {
            const int group = i / 32;
            const int low = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
            const int high = (le16(b + 2) >> (2 * group)) & 3;
            return (half(b) * ((low | (high << 4)) - 32)) * kvalues_iq4nl[nibble(b + 8 + group * 16, i % 32)];
        } else if constexpr (Format == DType::MXFP4) {
            return e8m0_half(b[0]) * kvalues_fp4[nibble(b + 1, i)];
        } else if constexpr (Format == DType::TQ1_0) {
            if (i < 160) return half(b + 52) * ternary(b[i % 32], i / 32);
            if (i < 240) return half(b + 52) * ternary(b[32 + (i - 160) % 16], (i - 160) / 16);
            return half(b + 52) * ternary(b[48 + (i - 240) % 4], (i - 240) / 4);
        } else if constexpr (Format == DType::TQ2_0) {
            const int packed = b[(i / 128) * 32 + i % 32];
            return half(b + 64) * (((packed >> (2 * (i / 32 % 4))) & 3) - 1);
        } else if constexpr (Format == DType::PTQ1_0) {
            if (i < 80) return half(b + 26) * ternary(b[i % 16], i / 16);
            if (i < 120) return half(b + 26) * ternary(b[16 + (i - 80) % 8], (i - 80) / 8);
            return half(b + 26) * ternary(b[24 + (i - 120) % 2], (i - 120) / 2);
        } else {
            static_assert(Format == DType::Q2_0 || Format == DType::PQ2_0);
            return half(b) * (((b[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
        }
    }

    static FUSIONRT_CODEC_INLINE void dequant_block(const uint8_t* b, float* out) {
        for (int i = 0; i < kBlockSize; ++i) out[i] = value(b, i);
    }

    static FUSIONRT_CODEC_INLINE float dot_block(const uint8_t* b, const float* x) {
        float sum = 0.0f;
        for (int i = 0; i < kBlockSize; ++i) sum += value(b, i) * x[i];
        return sum;
    }
};

#undef FUSIONRT_CODEC_INLINE

// Dispatch the format once outside the execution loop.
template<class Visitor>
decltype(auto) visit_weight_codec(DType dtype, Visitor&& visitor) {
    switch (dtype) {
#define FUSIONRT_CODEC_CASE(name, id, block, bytes) \
        case DType::name: return visitor.template operator()<WeightCodec<DType::name>>();
        FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_CASE)
#undef FUSIONRT_CODEC_CASE
        default: throw std::invalid_argument("Unsupported weight codec");
    }
}

}  // namespace fusionrt
