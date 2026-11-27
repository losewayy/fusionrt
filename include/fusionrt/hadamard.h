#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace fusionrt {

inline void validate_hadamard_shape(int rows, int cols, int block) {
    if (rows <= 0 || cols <= 0 || block < 64 || block > 1024 ||
        (block & (block - 1)) != 0 || cols % block != 0)
        throw std::invalid_argument("Invalid blockwise Hadamard dimensions");
}

namespace hadamard_detail {

inline void transform(const float* x, const float* weight, const int8_t* signs,
                      float* out, int rows, int cols, int block, float eps, bool inverse) {
    validate_hadamard_shape(rows, cols, block);
    const float block_scale = 1.0f / std::sqrt(static_cast<float>(block));
    for (int r = 0; r < rows; ++r, x += cols, out += cols) {
        float inv_rms = 1.0f;
        if (weight) {
            float sum = 0.0f;
            for (int i = 0; i < cols; ++i) sum += x[i] * x[i];
            inv_rms = 1.0f / std::sqrt(sum / cols + eps);
        }
        for (int start = 0; start < cols; start += block) {
            auto* buf = out + start;
            for (int i = start; i < start + block; ++i) {
                const float norm = weight ? x[i] * inv_rms * weight[i] : x[i];
                buf[i - start] = signs && !inverse ? norm * signs[i] : norm;
            }
            for (int span = 1; span < block; span *= 2) {
                for (int pair = 0; pair < block / 2; ++pair) {
                    const int lo = 2 * (pair / span) * span + pair % span;
                    const float a = buf[lo], b = buf[lo + span];
                    buf[lo] = a + b;
                    buf[lo + span] = a - b;
                }
            }
            for (int i = 0; i < block; ++i)
                buf[i] *= block_scale * (signs && inverse ? signs[start + i] : 1);
        }
    }
}

}  // namespace hadamard_detail

inline void blockwise_hadamard_f32(const float* x, const int8_t* signs, float* out,
                                  int rows, int cols, int block, bool inverse = false) {
    hadamard_detail::transform(x, nullptr, signs, out, rows, cols, block, 0.0f, inverse);
}

inline void fused_rmsnorm_hadamard_f32(const float* x, const float* weight, const int8_t* signs,
                                      float* out, int rows, int cols, int block, float eps) {
    hadamard_detail::transform(x, weight, signs, out, rows, cols, block, eps, false);
}

}  // namespace fusionrt
