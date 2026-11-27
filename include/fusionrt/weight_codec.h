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

#ifdef __CUDACC__
    static constexpr int kWarpBlockLanes = Format == DType::PTQ1_0 ? 8 : 32;

    // Each lane unpacks the weights that share a packed byte or sub-block scale.
    // The caller reduces across the warp after accumulating the row's blocks.
    static __device__ __forceinline__ float warp_dot_block(const uint8_t* b, const float* x, int lane) {
        using namespace codec_detail;
        float sum = 0.0f;
        if constexpr (Format == DType::PTQ1_0) {
            // Eight lanes cover two primary bytes, one secondary byte, and
            // one tail trit each. A warp processes four complete blocks.
            const uint8_t lo = b[lane], hi = b[lane + 8], mid = b[lane + 16];
            float sum_hi = 0.f, sum_mid = 0.f;
#pragma unroll
            for (int digit = 0; digit < 5; ++digit) {
                sum += x[digit * 16 + lane] * ternary(lo, digit);
                sum_hi += x[digit * 16 + lane + 8] * ternary(hi, digit);
                sum_mid += x[80 + digit * 8 + lane] * ternary(mid, digit);
            }
            return half(b + 26) * (sum + sum_hi + sum_mid +
                                  x[120 + lane] * ternary(b[24 + (lane & 1)], lane / 2));
        } else if constexpr (Format == DType::TQ1_0) {
            const auto packed = b[lane];
#pragma unroll
            for (int digit = 0; digit < 5; ++digit) sum += x[digit * 32 + lane] * ternary(packed, digit);
            if (lane < 16) {
                const auto tail = b[32 + lane];
#pragma unroll
                for (int digit = 0; digit < 5; ++digit) sum += x[160 + digit * 16 + lane] * ternary(tail, digit);
                sum += x[240 + lane] * ternary(b[48 + (lane & 3)], lane / 4);
            }
            return half(b + 52) * sum;
        } else if constexpr (Format == DType::Q2_0 || Format == DType::PQ2_0) {
            if (lane < kBlockSize / 4) {
                const auto packed = b[2 + lane];
                const float4 a = reinterpret_cast<const float4*>(x)[lane];
                sum = a.x * (static_cast<int>(packed & 3) - 1) +
                      a.y * (static_cast<int>(packed >> 2 & 3) - 1) +
                      a.z * (static_cast<int>(packed >> 4 & 3) - 1) +
                      a.w * (static_cast<int>(packed >> 6) - 1);
            }
            return half(b) * sum;
        } else if constexpr (Format == DType::TQ2_0) {
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                const auto packed = b[lane + group * 32];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    sum += x[lane + group * 128 + i * 32] * (static_cast<int>((packed >> (i * 2)) & 3) - 1);
            }
            return half(b + 64) * sum;
        } else if constexpr (Format == DType::IQ4_NL || Format == DType::MXFP4 || Format == DType::IQ4_XS) {
            if (lane < 16) {
                if constexpr (Format == DType::IQ4_XS) {
                    const unsigned high = le16(b + 2);
#pragma unroll
                    for (int group = 0; group < 8; ++group) {
                        const int low = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
                        const int scale = (low | (static_cast<int>(high >> (2 * group) & 3) << 4)) - 32;
                        const auto packed = b[8 + group * 16 + lane];
                        sum += scale * (x[group * 32 + lane] * kvalues_iq4nl[packed & 15] +
                                        x[group * 32 + lane + 16] * kvalues_iq4nl[packed >> 4]);
                    }
                } else if constexpr (Format == DType::IQ4_NL) {
                    const auto packed = b[2 + lane];
                    sum = x[lane] * kvalues_iq4nl[packed & 15] + x[lane + 16] * kvalues_iq4nl[packed >> 4];
                } else {
                    const auto packed = b[1 + lane];
                    sum = x[lane] * kvalues_fp4[packed & 15] + x[lane + 16] * kvalues_fp4[packed >> 4];
                }
            }
            if constexpr (Format == DType::MXFP4) return e8m0_half(b[0]) * sum;
            else return half(b) * sum;
        } else if constexpr (Format == DType::Q3_K) {
            const int mask = b[lane];
#pragma unroll
            for (int group = 0; group < 8; ++group) {
                const int sub = group * 2 + lane / 16;
                const int low = (b[96 + sub % 8] >> (4 * (sub / 8))) & 15;
                const int high = (b[104 + sub % 4] >> (2 * (sub / 4))) & 3;
                const int quant = ((b[32 + group / 4 * 32 + lane] >> (2 * (group % 4))) & 3)
                    - ((mask >> group & 1) ? 0 : 4);
                sum += x[group * 32 + lane] * (((low | (high << 4)) - 32) * quant);
            }
            return half(b + 108) * sum;
        } else {
#pragma unroll
            for (int i = 0; i < kBlockSize; i += 32) sum += x[i + lane] * value(b, i + lane);
            return sum;
        }
    }
#endif

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
