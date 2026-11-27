#ifndef FUSIONRT_CODEC_COMMON_GLSL
#define FUSIONRT_CODEC_COMMON_GLSL

// DType from fusionrt/types.h, in runtime enum order.
const int DTYPE_F32 = 0;
const int DTYPE_F16 = 1;
const int DTYPE_BF16 = 2;
const int DTYPE_Q8_0 = 3;
const int DTYPE_Q4_0 = 4;
const int DTYPE_I32 = 5;
const int DTYPE_U8 = 6;
const int DTYPE_Q4_1 = 7;
const int DTYPE_Q4_K = 8;
const int DTYPE_Q6_K = 9;
const int DTYPE_Q5_K = 10;
const int DTYPE_Q4_A = 11;
const int DTYPE_Q8_A = 12;
const int DTYPE_Q8_1 = 13;
const int DTYPE_Q4_KA = 14;
const int DTYPE_Q6_A = 15;
const int DTYPE_Q2_K = 16;
const int DTYPE_Q3_K = 17;
const int DTYPE_IQ2_XXS = 18;
const int DTYPE_IQ2_XS = 19;
const int DTYPE_IQ3_XXS = 20;
const int DTYPE_IQ4_NL = 21;
const int DTYPE_IQ3_S = 22;
const int DTYPE_IQ4_XS = 23;
const int DTYPE_TQ1_0 = 24;
const int DTYPE_TQ2_0 = 25;
const int DTYPE_MXFP4 = 26;
const int DTYPE_Q2_0 = 27;
const int DTYPE_PQ2_0 = 28;
const int DTYPE_PTQ1_0 = 29;

// The buffer contains tightly packed little-endian rows. Its last word is
// padded to four bytes; no padding is inserted between rows or blocks.
uint read_u8(uint offset) {
    return bitfieldExtract(w[offset >> 2u], int(offset & 3u) * 8, 8);
}

int read_i8(uint offset) {
    return bitfieldExtract(int(w[offset >> 2u]), int(offset & 3u) * 8, 8);
}

uint read_u16(uint offset) {
    uint word = w[offset >> 2u];
    uint byte_in_word = offset & 3u;
    if (byte_in_word == 3u) {
        return (word >> 24u) | (read_u8(offset + 1u) << 8u);
    }
    return bitfieldExtract(word, int(byte_in_word) * 8, 16);
}

uint read_u32(uint offset) {
    uint word = w[offset >> 2u];
    uint shift = (offset & 3u) * 8u;
    return shift == 0u ? word : (word >> shift) | (w[(offset >> 2u) + 1u] << (32u - shift));
}

float read_f16(uint offset) {
    return unpackHalf2x16(read_u16(offset)).x;
}

float read_bf16(uint offset) {
    return uintBitsToFloat(read_u16(offset) << 16u);
}

// Values and bytes per block, as in dtype_block_size and dtype_type_size.
uvec2 weight_block_layout(int dtype) {
    switch (dtype) {
        case DTYPE_F32:
        case DTYPE_I32:    return uvec2(1, 4);
        case DTYPE_F16:
        case DTYPE_BF16:   return uvec2(1, 2);
        case DTYPE_U8:     return uvec2(1, 1);
        case DTYPE_Q8_0:   return uvec2(32, 34);
        case DTYPE_Q4_0:
        case DTYPE_IQ4_NL: return uvec2(32, 18);
        case DTYPE_Q4_1:   return uvec2(32, 20);
        case DTYPE_Q4_K:
        case DTYPE_Q4_A:
        case DTYPE_Q4_KA:  return uvec2(256, 144);
        case DTYPE_Q6_K:   return uvec2(256, 210);
        case DTYPE_Q5_K:   return uvec2(256, 176);
        case DTYPE_Q8_A:   return uvec2(256, 272);
        case DTYPE_Q8_1:   return uvec2(32, 48);
        case DTYPE_Q6_A:   return uvec2(256, 224);
        case DTYPE_Q2_K:   return uvec2(256, 84);
        case DTYPE_Q3_K:
        case DTYPE_IQ3_S:  return uvec2(256, 110);
        case DTYPE_IQ2_XXS:
        case DTYPE_TQ2_0:  return uvec2(256, 66);
        case DTYPE_IQ2_XS: return uvec2(256, 74);
        case DTYPE_IQ3_XXS: return uvec2(256, 98);
        case DTYPE_IQ4_XS: return uvec2(256, 136);
        case DTYPE_TQ1_0:  return uvec2(256, 54);
        case DTYPE_MXFP4:  return uvec2(32, 17);
        case DTYPE_Q2_0:   return uvec2(64, 18);
        case DTYPE_PQ2_0:  return uvec2(128, 34);
        case DTYPE_PTQ1_0: return uvec2(128, 28);
        default:          return uvec2(1, 0); // Includes the K8V4 KV policy.
    }
}

uint row_bytes_for_dtype(int dtype, int K) {
    uvec2 block = weight_block_layout(dtype);
    // The host rejects invalid shapes in row_nbytes(); zero is the shader sentinel.
    return K >= 0 && uint(K) % block.x == 0u ? (uint(K) / block.x) * block.y : 0u;
}

int read_nibble(uint offset, uint i) {
    return int(bitfieldExtract(read_u8(offset + i % 16u), int(i / 16u) * 4, 4));
}

int read_packed_k2(uint offset, uint i) {
    uint packed = read_u8(offset + (i / 128u) * 32u + i % 32u);
    return int(bitfieldExtract(packed, int(i / 32u % 4u) * 2, 2));
}

uvec2 unpack_k_scale_min(uint offset, uint group) {
    if (group < 4u) {
        return uvec2(read_u8(offset + group), read_u8(offset + group + 4u)) & 63u;
    }
    uint low = read_u8(offset + group + 4u);
    uvec2 high = uvec2(read_u8(offset + group - 4u), read_u8(offset + group));
    return uvec2(low & 15u, low >> 4u) | ((high >> 6u) << 4u);
}

float codec_mul(float a, float b) {
    // Vulkan permits drivers to discard the sign of a floating-point zero.
    // Recover the product sign from the operands for CPU parity.
    uint magnitude = floatBitsToUint(a * b) & 0x7fffffffu;
    uint sign_bit = (floatBitsToUint(a) ^ floatBitsToUint(b)) & 0x80000000u;
    return uintBitsToFloat(magnitude | sign_bit);
}

float dequantize_affine(float scale, int quant, float bias) {
    float product = codec_mul(scale, float(quant));
    precise float value = product + bias;
    uint bits = floatBitsToUint(value);
    if ((bits & 0x7fffffffu) == 0u) {
        bits = (floatBitsToUint(product) & floatBitsToUint(bias)) & 0x80000000u;
    }
    return uintBitsToFloat(bits);
}

float dequantize_k45(uint b, uint i, bool has_high_bits) {
    uint group = i / 32u;
    uvec2 scale_min = unpack_k_scale_min(b + 4u, group);
    uint low_offset = has_high_bits ? 48u : 16u;
    uint low = read_u8(b + low_offset + (i / 64u) * 32u + i % 32u);
    int quant = int(bitfieldExtract(low, int(group & 1u) * 4, 4));
    if (has_high_bits) {
        quant |= int(bitfieldExtract(read_u8(b + 16u + i % 32u), int(group), 1)) << 4;
    }
    return dequantize_affine(codec_mul(read_f16(b), float(scale_min.x)), quant,
                            codec_mul(read_f16(b + 2u), -float(scale_min.y)));
}

float dequantize_mainline(int dtype, uint b, uint i) {
    switch (dtype) {
        case DTYPE_F32:  return uintBitsToFloat(w[b >> 2u]);
        case DTYPE_F16:  return read_f16(b);
        case DTYPE_BF16: return read_bf16(b);
        case DTYPE_I32:  return float(int(w[b >> 2u]));
        case DTYPE_U8:   return float(read_u8(b));
        case DTYPE_Q8_0: return codec_mul(read_f16(b), float(read_i8(b + 2u + i)));
        case DTYPE_Q4_0: return codec_mul(read_f16(b), float(read_nibble(b + 2u, i) - 8));
        case DTYPE_Q4_1: return dequantize_affine(read_f16(b), read_nibble(b + 4u, i), read_f16(b + 2u));
        case DTYPE_Q2_K: {
            uint sm = read_u8(b + i / 16u);
            return dequantize_affine(codec_mul(read_f16(b + 80u), float(sm & 15u)),
                                    read_packed_k2(b + 16u, i),
                                    codec_mul(read_f16(b + 82u), -float(sm >> 4u)));
        }
        case DTYPE_Q4_K: return dequantize_k45(b, i, false);
        case DTYPE_Q5_K: return dequantize_k45(b, i, true);
        case DTYPE_Q6_K: {
            uint group = i / 32u;
            uint low = read_u8(b + (i / 128u) * 64u + i % 64u);
            int ql = int(bitfieldExtract(low, int(group / 2u % 2u) * 4, 4));
            uint high = read_u8(b + 128u + (i / 128u) * 32u + i % 32u);
            int qh = int(bitfieldExtract(high, int(group % 4u) * 2, 2));
            float scale = codec_mul(read_f16(b + 208u), float(read_i8(b + 192u + i / 16u)));
            return codec_mul(scale, float((ql | (qh << 4)) - 32));
        }
        case DTYPE_Q4_A: {
            uint group = i / 32u;
            return codec_mul(read_f16(b + group * 2u), float(read_nibble(b + 16u + group * 16u, i % 32u) - 8));
        }
        case DTYPE_Q8_A:
            return codec_mul(read_f16(b + (i / 32u) * 2u), float(read_i8(b + 16u + i)));
        case DTYPE_Q8_1:
            return codec_mul(uintBitsToFloat(w[(b + 32u) >> 2u]), float(read_i8(b + i)));
        case DTYPE_Q4_KA: {
            uint group = i / 32u;
            uint sm = (read_u16(b + 132u + group + group / 2u) >> ((group & 1u) * 4u)) & 4095u;
            return dequantize_affine(codec_mul(read_f16(b + 128u), float(sm & 63u)),
                                    read_nibble(b + group * 16u, i % 32u),
                                    codec_mul(read_f16(b + 130u), -float(sm >> 6u)));
        }
        case DTYPE_Q6_A: {
            uint group = i / 32u;
            int low = read_nibble(b + group * 16u, i % 32u);
            uint packed = read_u8(b + 128u + (group / 2u) * 16u + i % 16u);
            int high = int(bitfieldExtract(packed, int(i / 16u % 4u) * 2, 2));
            float scale = codec_mul(read_f16(b + 208u), float(read_i8(b + 192u + i / 16u)));
            return codec_mul(scale, float((low | (high << 4)) - 32));
        }
        default: return 0.0;
    }
}

#if !defined(FUSIONRT_MAINLINE_ONLY)
// CMake translates the shared C++ codebooks into GLSL, splitting each
// 64-bit IQ2 grid word into a pair of 32-bit words for portable indexing.
#include "weight_codebooks.glsl"

// Stage only the active grid. This keeps dynamically indexed constant arrays
// out of the inner loop and bounds shared memory even in the tiled kernels.
shared uvec2 codec_grid[512];

// Call uniformly across the workgroup before decoding grid-based formats.
void load_codec_grid(int dtype) {
    uint step = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    uint lane = gl_LocalInvocationIndex;
    switch (dtype) {
        case DTYPE_IQ2_XXS:
            for (uint i = lane; i < iq2xxs_grid.length(); i += step) codec_grid[i] = iq2xxs_grid[i];
            break;
        case DTYPE_IQ2_XS:
            for (uint i = lane; i < iq2xs_grid.length(); i += step) codec_grid[i] = iq2xs_grid[i];
            break;
        case DTYPE_IQ3_XXS:
            for (uint i = lane; i < iq3xxs_grid.length(); i += step) codec_grid[i].x = iq3xxs_grid[i];
            break;
        case DTYPE_IQ3_S:
            for (uint i = lane; i < iq3s_grid.length(); i += step) codec_grid[i].x = iq3s_grid[i];
            break;
        default: return;
    }
    barrier();
}

int codec_sign(uint bits, uint i) {
    return ((bits >> i) & 1u) == 0u ? 1 : -1;
}

int codec_grid_byte(uint word, uint i) {
    return int(bitfieldExtract(word, int(i % 4u) * 8, 8));
}

int codec_ternary(uint packed, uint digit) {
    const uint pow3[5] = uint[5](1u, 3u, 9u, 27u, 81u);
    uint wrapped = (packed * pow3[digit]) & 255u;
    return int((wrapped * 3u) >> 8u) - 1;
}

float dequantize_mxfp4(uint exponent, int quant) {
    if (exponent >= 2u) return codec_mul(uintBitsToFloat((exponent - 1u) << 23u), float(quant));

    // E8M0 exponents 0 and 1 use a subnormal FP32 scale. Build the exact
    // product bits even on hardware that flushes FP32 subnormal operands.
    uint magnitude = uint(abs(quant)) << (21u + exponent);
    uint bits = magnitude;
    if (magnitude >= 0x00800000u) {
        uint shift = uint(findMSB(magnitude) - 23);
        bits = ((shift + 1u) << 23u) | ((magnitude >> shift) & 0x007fffffu);
    }
    return uintBitsToFloat(bits | (quant < 0 ? 0x80000000u : 0u));
}

// Mirrors WeightCodec<Format>::value in fusionrt/weight_codec.h.
float dequantize_weight_codec(int dtype, uint b, uint i) {
    switch (dtype) {
        case DTYPE_Q3_K: {
            uint group = i / 16u;
            uint low = (read_u8(b + 96u + group % 8u) >> (4u * (group / 8u))) & 15u;
            uint high = (read_u8(b + 104u + group % 4u) >> (2u * (group / 4u))) & 3u;
            int quant = read_packed_k2(b + 32u, i) -
                        (((read_u8(b + i % 32u) >> (i / 32u)) & 1u) == 0u ? 4 : 0);
            float scale = codec_mul(read_f16(b + 108u), float(int(low | (high << 4u)) - 32));
            return codec_mul(scale, float(quant));
        }
        case DTYPE_IQ2_XXS: {
            uint group = b + 2u + (i / 32u) * 8u;
            uint aux = read_u32(group + 4u);
            uint sub = i / 8u % 4u;
            uvec2 grid = codec_grid[read_u8(group + sub)];
            uint signs = ksigns_iq2xs[(aux >> (7u * sub)) & 127u];
            float scale = codec_mul(read_f16(b), (0.5 + float(aux >> 28u)) * 0.25);
            int quant = codec_grid_byte(grid[i / 4u % 2u], i) * codec_sign(signs, i % 8u);
            return codec_mul(scale, float(quant));
        }
        case DTYPE_IQ2_XS: {
            uint packed = read_u16(b + 2u + (i / 8u) * 2u);
            uvec2 grid = codec_grid[packed & 511u];
            uint scale = (read_u8(b + 66u + i / 32u) >> (4u * (i / 16u % 2u))) & 15u;
            float factor = codec_mul(read_f16(b), (0.5 + float(scale)) * 0.25);
            int quant = codec_grid_byte(grid[i / 4u % 2u], i) * codec_sign(ksigns_iq2xs[packed >> 9u], i % 8u);
            return codec_mul(factor, float(quant));
        }
        case DTYPE_IQ3_XXS: {
            uint aux = read_u32(b + 66u + (i / 32u) * 4u);
            uint grid = codec_grid[read_u8(b + 2u + i / 4u)].x;
            uint signs = ksigns_iq2xs[(aux >> (7u * (i / 8u % 4u))) & 127u];
            float scale = codec_mul(read_f16(b), (0.5 + float(aux >> 28u)) * 0.5);
            return codec_mul(scale, float(codec_grid_byte(grid, i) * codec_sign(signs, i % 8u)));
        }
        case DTYPE_IQ3_S: {
            uint group = i / 32u;
            uint high = (read_u8(b + 66u + group) >> (i / 4u % 8u)) & 1u;
            uint index = read_u8(b + 2u + i / 4u) | (high << 8u);
            uint scale = (read_u8(b + 106u + group / 2u) >> (4u * (group % 2u))) & 15u;
            float factor = codec_mul(read_f16(b), float(1u + 2u * scale));
            int quant = codec_grid_byte(codec_grid[index].x, i) * codec_sign(read_u8(b + 74u + i / 8u), i % 8u);
            return codec_mul(factor, float(quant));
        }
        case DTYPE_IQ4_NL:
            return codec_mul(read_f16(b), float(kvalues_iq4nl[read_nibble(b + 2u, i)]));
        case DTYPE_IQ4_XS: {
            uint group = i / 32u;
            uint low = (read_u8(b + 4u + group / 2u) >> (4u * (group % 2u))) & 15u;
            uint high = (read_u16(b + 2u) >> (2u * group)) & 3u;
            float scale = codec_mul(read_f16(b), float(int(low | (high << 4u)) - 32));
            return codec_mul(scale, float(kvalues_iq4nl[read_nibble(b + 8u + group * 16u, i % 32u)]));
        }
        case DTYPE_MXFP4:
            return dequantize_mxfp4(read_u8(b), kvalues_fp4[read_nibble(b + 1u, i)]);
        case DTYPE_TQ1_0: {
            float scale = read_f16(b + 52u);
            if (i < 160u) return codec_mul(scale, float(codec_ternary(read_u8(b + i % 32u), i / 32u)));
            if (i < 240u) return codec_mul(scale, float(codec_ternary(read_u8(b + 32u + (i - 160u) % 16u), (i - 160u) / 16u)));
            return codec_mul(scale, float(codec_ternary(read_u8(b + 48u + (i - 240u) % 4u), (i - 240u) / 4u)));
        }
        case DTYPE_TQ2_0:
            return codec_mul(read_f16(b + 64u), float(read_packed_k2(b, i) - 1));
        case DTYPE_Q2_0:
        case DTYPE_PQ2_0: {
            uint quant = (read_u8(b + 2u + i / 4u) >> (2u * (i % 4u))) & 3u;
            return codec_mul(read_f16(b), float(int(quant) - 1));
        }
        case DTYPE_PTQ1_0: {
            float scale = read_f16(b + 26u);
            if (i < 80u)  return codec_mul(scale, float(codec_ternary(read_u8(b + i % 16u), i / 16u)));
            if (i < 120u) return codec_mul(scale, float(codec_ternary(read_u8(b + 16u + (i - 80u) % 8u), (i - 80u) / 8u)));
            return codec_mul(scale, float(codec_ternary(read_u8(b + 24u + (i - 120u) % 2u), (i - 120u) / 2u)));
        }
        default: return 0.0;
    }
}

float dequantize_weight(int dtype, uint row, uint k, uvec2 block) {
    uint b = row + (k / block.x) * block.y;
    uint i = k % block.x;
    if (dtype >= DTYPE_Q3_K) return dequantize_weight_codec(dtype, b, i);
    return dequantize_mainline(dtype, b, i);
}
#endif

#if defined(FUSIONRT_GEMV_DEQUANTIZE)
shared float subgroup_sums[gl_WorkGroupSize.x];

void run_gemv() {
    uint n = gl_WorkGroupID.x;
    uint m = gl_WorkGroupID.y;
    if (n >= uint(pc.N) || m >= uint(pc.M)) return;

#if !defined(FUSIONRT_MAINLINE_ONLY)
    load_codec_grid(pc.dtype);
#endif
    uint stride = row_bytes_for_dtype(pc.dtype, pc.K);
    uvec2 block = weight_block_layout(pc.dtype);
    uint row = n * stride;
    uint x_row = m * uint(pc.K);
    float sum = 0.0;
    for (uint k = gl_LocalInvocationID.x; stride != 0u && k < uint(pc.K); k += gl_WorkGroupSize.x) {
        uint b = row + (k / block.x) * block.y;
        float weight = FUSIONRT_GEMV_DEQUANTIZE(pc.dtype, b, k % block.x);
        sum = fma(x[x_row + k], weight, sum);
    }

    sum = subgroupAdd(sum);
    if (subgroupElect()) subgroup_sums[gl_SubgroupID] = sum;
    barrier();

    if (gl_SubgroupID == 0u) {
        float partial = 0.0;
        for (uint s = gl_SubgroupInvocationID; s < gl_NumSubgroups; s += gl_SubgroupSize) {
            partial += subgroup_sums[s];
        }
        float result = subgroupAdd(partial);
        if (subgroupElect()) y[m * uint(pc.N) + n] = result;
    }
}
#endif

#endif
