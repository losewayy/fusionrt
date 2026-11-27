#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>
#include <cmath>

namespace fusionrt {

// Runtime dtype, GGML tensor ID, elements per block, bytes per block.
#define FUSIONRT_WEIGHT_CODEC_FORMATS(X) \
    X(Q3_K,    11, 256, 110) \
    X(IQ2_XXS,  16, 256,  66) \
    X(IQ2_XS,  17, 256,  74) \
    X(IQ3_XXS,  18, 256,  98) \
    X(IQ4_NL,  20,  32,  18) \
    X(IQ3_S,   21, 256, 110) \
    X(IQ4_XS,  23, 256, 136) \
    X(TQ1_0,   34, 256,  54) \
    X(TQ2_0,   35, 256,  66) \
    X(MXFP4,   39,  32,  17) \
    X(Q2_0,    42,  64,  18) \
    X(PQ2_0,  142, 128,  34) \
    X(PTQ1_0, 143, 128,  28)

enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q8_0 = 3,  // block size 32: fp16 scale + 32 x int8 (34 bytes)
    Q4_0 = 4,  // block size 32: fp16 scale + 16 x uint8 packed signed [-8..7] (18 bytes)
    I32 = 5,
    U8 = 6,
    Q4_1 = 7,  // block size 32: fp16 d + fp16 m + 16 x uint8 (20 bytes)
    Q4_K = 8,  // block size 256: k-quant 4-bit superblock (144 bytes)
    Q6_K = 9,  // block size 256: k-quant 6-bit superblock (210 bytes)
    Q5_K = 10, // block size 256: k-quant 5-bit superblock (176 bytes)
    Q4_A = 11, // block size 256: 16B-aligned repacked Q4_0 x8 (144 bytes)
    Q8_A = 12, // block size 256: 16B-aligned repacked Q8_0 x8 (272 bytes)
    Q8_1 = 13, // block size 32: packed activation quants, scale and sums (48 bytes)
    Q4_KA = 14, // CUDA: 256 weights in 144 aligned bytes
    Q6_A = 15,  // CUDA: 256 weights in 224 aligned bytes
    Q2_K = 16,  // 256 weights in 84 bytes
#define FUSIONRT_CODEC_ENUM(name, id, block, bytes) name,
    FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_ENUM)
#undef FUSIONRT_CODEC_ENUM
};

enum class Backend : uint8_t {
    CPU = 0,
    CUDA = 1,
    Vulkan = 2,
};

#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t d;      // fp16 scale
    int8_t qs[32];   // 32 int8 quants
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

struct BlockQ4_0 {
    uint16_t d;      // fp16 scale
    uint8_t qs[16];  // 32 4-bit unsigned nibbles [0..15] representing (q - 8) in [-8..7]
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

struct BlockQ4_1 {
    uint16_t d;      // fp16 delta
    uint16_t m;      // fp16 min
    uint8_t qs[16];  // 32 4-bit unsigned nibbles [0..15]
};
static_assert(sizeof(BlockQ4_1) == 20, "BlockQ4_1 must be 20 bytes");

struct BlockQ2_K {
    uint8_t scales[16];  // low nibble: scale, high nibble: minimum magnitude
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(BlockQ2_K) == 84);

struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

struct BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

struct BlockQ4_A {
    uint16_t d[8];    // fp16 scales for 8x32 sub-blocks
    uint32_t qs[32];  // 16B-aligned packed nibbles (4 words per sub-block)
};
static_assert(sizeof(BlockQ4_A) == 144, "BlockQ4_A must be 144 bytes");

struct BlockQ8_A {
    uint16_t d[8];    // fp16 scales for 8x32 sub-blocks
    int32_t qs[64];   // 16B-aligned packed int8s (8 words per sub-block)
};
static_assert(sizeof(BlockQ8_A) == 272, "BlockQ8_A must be 272 bytes");
struct alignas(16) BlockQ8_1 {
    int32_t qs[8];  // 32 signed int8 quants, packed four per word
    float d;        // amax / 127
    float s_lo;     // d * sum(q[0..15])
    float s_hi;     // d * sum(q[16..31])
    float s_all;    // d * sum(q[0..31])
};
static_assert(sizeof(BlockQ8_1) == 48);
#pragma pack(pop)

struct alignas(16) BlockQ4_KA {
    uint8_t q[8][16];
    uint16_t d, dmin;
    // Consecutive 12-bit pairs, each with a six-bit scale and minimum.
    uint8_t scales[12];
};
static_assert(sizeof(BlockQ4_KA) == 144);

// The low and high bytes hold the six-bit scale and minimum, respectively.
#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr unsigned unpack_k_scale_min(const uint8_t* s, int sub) {
    if (sub < 4) return (s[sub] & 63) | ((s[sub + 4] & 63) << 8);
    const unsigned low = s[sub + 4];
    const unsigned scale = (low & 15) | ((s[sub - 4] >> 6) << 4);
    const unsigned min = (low >> 4) | ((s[sub] >> 6) << 4);
    return scale | (min << 8);
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr unsigned unpack_q4_ka_scales(const BlockQ4_KA& b, int sub) {
    const int byte = sub + sub / 2;
    const unsigned pair = b.scales[byte] | (static_cast<unsigned>(b.scales[byte + 1]) << 8);
    return (pair >> ((sub & 1) * 4)) & 4095;
}

struct alignas(16) BlockQ6_A {
    uint8_t ql[8][16];
    uint8_t qh[4][16];
    int8_t scale[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_A) == 224);

inline constexpr bool is_weight_codec_dtype(DType dt) {
    switch (dt) {
#define FUSIONRT_CODEC_CASE(name, id, block, bytes) case DType::name: return true;
        FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_CASE)
#undef FUSIONRT_CODEC_CASE
        default: return false;
    }
}

inline constexpr size_t dtype_block_size(DType dt) {
    switch (dt) {
        case DType::Q8_0: return 32;
        case DType::Q8_1: return 32;
        case DType::Q4_0: return 32;
        case DType::Q4_1: return 32;
        case DType::Q2_K:
        case DType::Q4_K: return 256;
        case DType::Q5_K: return 256;
        case DType::Q6_K:
        case DType::Q4_KA:
        case DType::Q6_A: return 256;
        case DType::Q4_A: return 256;
        case DType::Q8_A: return 256;
#define FUSIONRT_CODEC_CASE(name, id, block, bytes) case DType::name: return block;
        FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_CASE)
#undef FUSIONRT_CODEC_CASE
        default: return 1;
    }
}

inline constexpr bool dtype_is_quantized(DType dt) {
    return dt == DType::Q8_0 || dt == DType::Q4_0 || dt == DType::Q4_1 ||
           dt == DType::Q2_K || dt == DType::Q4_K || dt == DType::Q5_K || dt == DType::Q6_K ||
           dt == DType::Q4_A || dt == DType::Q8_A || dt == DType::Q8_1 ||
           dt == DType::Q4_KA || dt == DType::Q6_A || is_weight_codec_dtype(dt);
}

inline constexpr size_t dtype_type_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::Q8_0: return sizeof(BlockQ8_0);
        case DType::Q4_0: return sizeof(BlockQ4_0);
        case DType::I32:  return 4;
        case DType::U8:   return 1;
        case DType::Q4_1: return sizeof(BlockQ4_1);
        case DType::Q2_K: return sizeof(BlockQ2_K);
        case DType::Q4_K: return sizeof(BlockQ4_K);
        case DType::Q5_K: return sizeof(BlockQ5_K);
        case DType::Q6_K: return sizeof(BlockQ6_K);
        case DType::Q4_KA: return sizeof(BlockQ4_KA);
        case DType::Q6_A: return sizeof(BlockQ6_A);
        case DType::Q4_A: return sizeof(BlockQ4_A);
        case DType::Q8_A: return sizeof(BlockQ8_A);
        case DType::Q8_1: return sizeof(BlockQ8_1);
#define FUSIONRT_CODEC_CASE(name, id, block, bytes) case DType::name: return bytes;
        FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_CASE)
#undef FUSIONRT_CODEC_CASE
    }
    return 0;
}

inline size_t row_nbytes(DType dt, int64_t cols) {
    size_t bs = dtype_block_size(dt);
    size_t ts = dtype_type_size(dt);
    if (cols % static_cast<int64_t>(bs) != 0) {
        throw std::runtime_error("Column count must be divisible by dtype block size");
    }
    return static_cast<size_t>(cols / static_cast<int64_t>(bs)) * ts;
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::BF16: return "BF16";
        case DType::Q8_0: return "Q8_0";
        case DType::Q4_0: return "Q4_0";
        case DType::I32:  return "I32";
        case DType::U8:   return "U8";
        case DType::Q4_1: return "Q4_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::Q4_KA: return "Q4_KA";
        case DType::Q6_A: return "Q6_A";
        case DType::Q4_A: return "Q4_A";
        case DType::Q8_A: return "Q8_A";
        case DType::Q8_1: return "Q8_1";
#define FUSIONRT_CODEC_CASE(name, id, block, bytes) case DType::name: return #name;
        FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_CASE)
#undef FUSIONRT_CODEC_CASE
    }
    return "UNKNOWN";
}

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CPU:    return "cpu";
        case Backend::CUDA:   return "cuda";
        case Backend::Vulkan: return "vulkan";
    }
    return "unknown";
}

// IEEE-754 float16 <-> float32 host helpers
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t rounded = (mant + ((1u << (shift - 1)) - 1) + ((mant >> shift) & 1u)) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint32_t rounded = mant + 0xFFFu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0;
        exp += 1;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x3FFu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((x >> 16) | 0x0040u);
    }
    uint32_t rounding_bias = 0x00007FFFu + ((x >> 16) & 1u);
    x += rounding_bias;
    return static_cast<uint16_t>(x >> 16);
}

inline float bf16_to_fp32(uint16_t b) {
    uint32_t out = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

}  // namespace fusionrt
