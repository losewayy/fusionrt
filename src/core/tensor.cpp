#include "fusionrt/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace fusionrt {

void quantize_row_q8_0(const float* src, BlockQ8_0* dst, int64_t cols) {
    const int64_t nb = cols / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float* x = src + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(x[i]));
        }
        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        dst[b].d = fp32_to_fp16(d);
        for (int i = 0; i < 32; ++i) {
            int32_t q = static_cast<int32_t>(std::lround(x[i] * id));
            q = std::max(-127, std::min(127, q));
            dst[b].qs[i] = static_cast<int8_t>(q);
        }
    }
}

void dequantize_row_q8_0(const BlockQ8_0* src, float* dst, int64_t cols) {
    const int64_t nb = cols / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(src[b].d);
        for (int i = 0; i < 32; ++i) {
            dst[b * 32 + i] = static_cast<float>(src[b].qs[i]) * d;
        }
    }
}

void quantize_row_q4_0(const float* src, BlockQ4_0* dst, int64_t cols) {
    const int64_t nb = cols / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float* x = src + b * 32;
        float amax = 0.0f;
        float abs_signed_max = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float ax = std::fabs(x[i]);
            if (ax > amax) {
                amax = ax;
                abs_signed_max = x[i];
            }
        }
        const float d = abs_signed_max / -8.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        dst[b].d = fp32_to_fp16(d);
        for (int i = 0; i < 16; ++i) {
            float x0 = x[i] * id;
            float x1 = x[i + 16] * id;
            int32_t q0 = std::min(15, std::max(0, static_cast<int32_t>(x0 + 8.5f)));
            int32_t q1 = std::min(15, std::max(0, static_cast<int32_t>(x1 + 8.5f)));
            dst[b].qs[i] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

void dequantize_row_q4_0(const BlockQ4_0* src, float* dst, int64_t cols) {
    const int64_t nb = cols / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(src[b].d);
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = src[b].qs[i];
            int32_t q0 = static_cast<int32_t>(byte & 0x0F) - 8;
            int32_t q1 = static_cast<int32_t>(byte >> 4) - 8;
            dst[b * 32 + i] = static_cast<float>(q0) * d;
            dst[b * 32 + i + 16] = static_cast<float>(q1) * d;
        }
    }
}

void dequantize_row_q4_1(const BlockQ4_1* src, float* dst, int64_t cols) {
    const int64_t nb = cols / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(src[b].d);
        const float m = fp16_to_fp32(src[b].m);
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = src[b].qs[i];
            dst[b * 32 + i] = static_cast<float>(byte & 0x0F) * d + m;
            dst[b * 32 + i + 16] = static_cast<float>(byte >> 4) * d + m;
        }
    }
}

void quantize_row_q2_k(const float* src, BlockQ2_K* dst, int64_t cols) {
    for (int64_t block = 0; block < cols / 256; ++block) {
        auto& out = dst[block];
        const auto* x = src + block * 256;
        float scales[16], minima[16];
        float max_scale = 0.0f, max_min = 0.0f;
        for (int sub = 0; sub < 16; ++sub) {
            const auto* begin = x + sub * 16;
            const auto [low, high] = std::minmax_element(begin, begin + 16);
            minima[sub] = std::max(0.0f, -*low);
            scales[sub] = (*high + minima[sub]) / 3.0f;
            max_scale = std::max(max_scale, scales[sub]);
            max_min = std::max(max_min, minima[sub]);
        }
        out.d = fp32_to_fp16(max_scale / 15.0f);
        out.dmin = fp32_to_fp16(max_min / 15.0f);
        const float d = fp16_to_fp32(out.d), dmin = fp16_to_fp32(out.dmin);
        std::fill_n(out.qs, 64, uint8_t{0});
        auto code = [](float value, float scale, int maximum) {
            return scale > 0.0f ? static_cast<int>(std::clamp(std::nearbyint(value / scale), 0.0f,
                                                            static_cast<float>(maximum))) : 0;
        };
        for (int sub = 0; sub < 16; ++sub) {
            const int scale = code(scales[sub], d, 15), minimum = code(minima[sub], dmin, 15);
            out.scales[sub] = static_cast<uint8_t>(scale | (minimum << 4));
            for (int lane = 0; lane < 16; ++lane) {
                const int i = sub * 16 + lane;
                const int q = code(x[i] + dmin * minimum, d * scale, 3);
                out.qs[(i / 128) * 32 + i % 32] |= static_cast<uint8_t>(q << (2 * ((i / 32) % 4)));
            }
        }
    }
}

void dequantize_row_q2_k(const BlockQ2_K* src, float* dst, int64_t cols) {
    for (int64_t block = 0; block < cols / 256; ++block) {
        const auto& w = src[block];
        const float d = fp16_to_fp32(w.d), dmin = fp16_to_fp32(w.dmin);
        for (int sub = 0; sub < 16; ++sub) {
            const int sm = w.scales[sub];
            const float scale = d * (sm & 15), minimum = dmin * (sm >> 4);
            const auto* packed = w.qs + (sub / 8) * 32 + (sub % 2) * 16;
            const int shift = ((sub / 2) % 4) * 2;
            for (int i = 0; i < 16; ++i)
                dst[block * 256 + sub * 16 + i] = scale * ((packed[i] >> shift) & 3) - minimum;
        }
    }
}

void dequantize_row_q4_k(const BlockQ4_K* src, float* dst, int64_t cols) {
    const int64_t nb = cols / 256;
    auto get_scale_min = [](int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
        if (j < 4) {
            d_out = q[j] & 63;
            m_out = q[j + 4] & 63;
        } else {
            d_out = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m_out = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
        }
    };
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(src[b].d);
        const float min_val = fp16_to_fp32(src[b].dmin);
        const uint8_t* q = src[b].qs;
        float* y = dst + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min(is, src[b].scales, sc, m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = min_val * static_cast<float>(m);
            get_scale_min(is + 1, src[b].scales, sc, m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = min_val * static_cast<float>(m);
            for (int l = 0; l < 32; ++l) {
                y[j + l] = d1 * static_cast<float>(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                y[j + 32 + l] = d2 * static_cast<float>(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q6_k(const BlockQ6_K* src, float* dst, int64_t cols) {
    const int64_t nb = cols / 256;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(src[b].d);
        const uint8_t* ql = src[b].ql;
        const uint8_t* qh = src[b].qh;
        const int8_t* sc = src[b].scales;
        float* y = dst + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row_q5_k(const BlockQ5_K* src, float* dst, int64_t cols) {
    const int64_t nb = cols / 256;
    auto get_scale_min = [](int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
        if (j < 4) {
            d_out = q[j] & 63;
            m_out = q[j + 4] & 63;
        } else {
            d_out = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m_out = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
        }
    };
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(src[b].d);
        const float min_val = fp16_to_fp32(src[b].dmin);
        const uint8_t* ql = src[b].qs;
        const uint8_t* qh = src[b].qh;
        float* y = dst + b * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min(is, src[b].scales, sc, m);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = min_val * static_cast<float>(m);
            get_scale_min(is + 1, src[b].scales, sc, m);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = min_val * static_cast<float>(m);
            for (int l = 0; l < 32; ++l) {
                y[j + l] = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                y[j + 32 + l] = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void convert_f32_to_f16(const float* src, uint16_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = fp32_to_fp16(src[i]);
}

void convert_f16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
}

void convert_f32_to_bf16(const float* src, uint16_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = fp32_to_bf16(src[i]);
}

void convert_bf16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(src[i]);
}

void repack_q4_k_to_q4_ka(const BlockQ4_K* src, BlockQ4_KA* dst, int64_t cols) {
    for (int64_t b = 0; b < cols / 256; ++b) {
        const auto& in = src[b];
        auto& out = dst[b];
        out.d = in.d;
        out.dmin = in.dmin;
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned lo = unpack_k_scale_min(in.scales, 2 * pair);
            const unsigned hi = unpack_k_scale_min(in.scales, 2 * pair + 1);
            const unsigned packed = (lo & 63) | ((lo >> 8) << 6) | ((hi & 63) << 12) | ((hi >> 8) << 18);
            for (int byte = 0; byte < 3; ++byte) {
                out.scales[3 * pair + byte] = static_cast<uint8_t>(packed >> (8 * byte));
            }
        }
        for (int sub = 0; sub < 8; ++sub) {
            const auto* q = in.qs + (sub / 2) * 32;
            const int shift = (sub % 2) * 4;
            for (int i = 0; i < 16; ++i) {
                out.q[sub][i] = ((q[i] >> shift) & 15) | (((q[i + 16] >> shift) & 15) << 4);
            }
        }
    }
}

void repack_q6_k_to_q6_a(const BlockQ6_K* src, BlockQ6_A* dst, int64_t cols) {
    for (int64_t b = 0; b < cols / 256; ++b) {
        const auto& in = src[b];
        auto& out = dst[b];
        out.d = in.d;
        std::memcpy(out.scale, in.scales, sizeof(out.scale));
        for (int sub = 0; sub < 8; ++sub) {
            const auto* ql = in.ql + (sub / 4) * 64 + (sub % 2) * 32;
            const int shift = ((sub % 4) / 2) * 4;
            for (int i = 0; i < 16; ++i) {
                out.ql[sub][i] = ((ql[i] >> shift) & 15) | (((ql[i + 16] >> shift) & 15) << 4);
            }
        }
        // Each high plane serves two adjacent 32-value groups.
        for (int pair = 0; pair < 4; ++pair) {
            const auto* qh = in.qh + (pair / 2) * 32;
            const int shift = (pair % 2) * 4;
            for (int i = 0; i < 16; ++i) {
                out.qh[pair][i] = ((qh[i] >> shift) & 3) | (((qh[i + 16] >> shift) & 3) << 2) |
                    (((qh[i] >> (shift + 2)) & 3) << 4) | (((qh[i + 16] >> (shift + 2)) & 3) << 6);
            }
        }
    }
}

void dequantize_row_q4_ka(const BlockQ4_KA* src, float* dst, int64_t cols) {
    for (int64_t b = 0; b < cols / 256; ++b) {
        const auto& block = src[b];
        for (int sub = 0; sub < 8; ++sub) {
            const unsigned scale_min = unpack_q4_ka_scales(block, sub);
            const float d = fp16_to_fp32(block.d) * (scale_min & 63);
            const float m = fp16_to_fp32(block.dmin) * (scale_min >> 6);
            for (int i = 0; i < 32; ++i) {
                const int q = (block.q[sub][i % 16] >> ((i / 16) * 4)) & 15;
                dst[b * 256 + sub * 32 + i] = d * q - m;
            }
        }
    }
}

void dequantize_row_q6_a(const BlockQ6_A* src, float* dst, int64_t cols) {
    for (int64_t b = 0; b < cols / 256; ++b) {
        const auto& block = src[b];
        for (int sub = 0; sub < 8; ++sub) {
            for (int i = 0; i < 32; ++i) {
                const int low = (block.ql[sub][i % 16] >> ((i / 16) * 4)) & 15;
                const int high = (block.qh[sub / 2][i % 16] >> ((sub % 2) * 4 + (i / 16) * 2)) & 3;
                const float d = fp16_to_fp32(block.d) * block.scale[2 * sub + i / 16];
                dst[b * 256 + sub * 32 + i] = d * ((low | (high << 4)) - 32);
            }
        }
    }
}

void repack_q4_0_to_q4_a(const BlockQ4_0* src, BlockQ4_A* dst, int64_t cols) {
    const int64_t n_super = cols / 256;
    for (int64_t s = 0; s < n_super; ++s) {
        BlockQ4_A& out = dst[s];
        for (int sb = 0; sb < 8; ++sb) {
            const BlockQ4_0& in = src[s * 8 + sb];
            out.d[sb] = in.d;
            std::memcpy(&out.qs[sb * 4], in.qs, 16);
        }
    }
}

void repack_q8_0_to_q8_a(const BlockQ8_0* src, BlockQ8_A* dst, int64_t cols) {
    const int64_t n_super = cols / 256;
    for (int64_t s = 0; s < n_super; ++s) {
        BlockQ8_A& out = dst[s];
        for (int sb = 0; sb < 8; ++sb) {
            const BlockQ8_0& in = src[s * 8 + sb];
            out.d[sb] = in.d;
            std::memcpy(&out.qs[sb * 8], in.qs, 32);
        }
    }
}

void dequantize_row_q4_a(const BlockQ4_A* src, float* dst, int64_t cols) {
    const int64_t n_super = cols / 256;
    for (int64_t s = 0; s < n_super; ++s) {
        const BlockQ4_A& in = src[s];
        for (int sb = 0; sb < 8; ++sb) {
            const float d = fp16_to_fp32(in.d[sb]);
            const auto* qs = reinterpret_cast<const uint8_t*>(&in.qs[sb * 4]);
            float* out = dst + s * 256 + sb * 32;
            for (int i = 0; i < 16; ++i) {
                uint8_t byte = qs[i];
                out[i]      = static_cast<float>(static_cast<int32_t>(byte & 0x0F) - 8) * d;
                out[i + 16] = static_cast<float>(static_cast<int32_t>(byte >> 4) - 8) * d;
            }
        }
    }
}

void dequantize_row_q8_a(const BlockQ8_A* src, float* dst, int64_t cols) {
    const int64_t n_super = cols / 256;
    for (int64_t s = 0; s < n_super; ++s) {
        const BlockQ8_A& in = src[s];
        for (int sb = 0; sb < 8; ++sb) {
            const float d = fp16_to_fp32(in.d[sb]);
            const auto* qs = reinterpret_cast<const int8_t*>(&in.qs[sb * 8]);
            float* out = dst + s * 256 + sb * 32;
            for (int i = 0; i < 32; ++i) {
                out[i] = static_cast<float>(qs[i]) * d;
            }
        }
    }
}

}  // namespace fusionrt
