#include "epilogue_common.cuh"
#include "fusionrt/hadamard.h"

namespace fusionrt::cuda_ops {

namespace {

template<bool Normalize>
__global__ void k_hadamard(const float* x, const float* weight, const int8_t* signs,
                           float* out, int cols, int block, float eps, bool inverse,
                           int head_dim, int key_heads) {
    __shared__ float values[1024];
    __shared__ float reduction[32];
    const int groups = cols / block;
    const int row = Normalize ? blockIdx.x : blockIdx.x / groups;
    const int first = Normalize ? 0 : blockIdx.x % groups;
    const int last = Normalize ? groups : first + 1;
    x += static_cast<int64_t>(row) * cols;
    out += static_cast<int64_t>(row) * cols;
    float inv_rms = 1.0f;
    if constexpr (Normalize) {
        float sum = 0;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += x[i] * x[i];
        inv_rms = rsqrtf(block_reduce_sum(sum, reduction) / cols + eps);
        __syncthreads();
    }
    const float scale = rsqrtf(static_cast<float>(block));
    for (int group = first; group < last; ++group) {
        const int start = group * block;
        for (int i = threadIdx.x; i < block; i += blockDim.x) {
            const int col = start + i;
            int src = col;
            if (key_heads) {
                // The GDN state is tiled by repeat, while the folded SSM projection groups repeats by key head.
                const int repeats = cols / head_dim / key_heads;
                const int head = col / head_dim;
                src = (head % repeats * key_heads + head / repeats) * head_dim + col % head_dim;
            }
            float v = x[src];
            if constexpr (Normalize) v *= inv_rms * weight[col];
            values[i] = signs && !inverse ? v * signs[col] : v;
        }
        __syncthreads();
        for (int span = 1; span < block; span *= 2) {
            for (int pair = threadIdx.x; pair < block / 2; pair += blockDim.x) {
                const int lo = 2 * (pair / span) * span + pair % span;
                const float a = values[lo], b = values[lo + span];
                values[lo] = a + b;
                values[lo + span] = a - b;
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < block; i += blockDim.x)
            out[start + i] = values[i] * scale * (signs && inverse ? signs[start + i] : 1);
        __syncthreads();
    }
}

}  // namespace

void blockwise_hadamard_f32(const float* x, const int8_t* signs, float* out, int rows, int cols,
                            int block, cudaStream_t stream, bool inverse,
                            int head_dim, int key_heads) {
    validate_hadamard_shape(rows, cols, block);
    const auto groups = static_cast<int64_t>(rows) * (cols / block);
    if (groups > 0x7fffffff) throw std::invalid_argument("Too many CUDA Hadamard groups");
    if (key_heads && (inverse || x == out || head_dim <= 0 || key_heads < 0 ||
                      cols % (static_cast<int64_t>(head_dim) * key_heads) != 0))
        throw std::invalid_argument("Invalid Hadamard value head permutation");
    k_hadamard<false><<<static_cast<unsigned>(groups), std::min(256, block / 2), 0, stream>>>(
        x, nullptr, signs, out, cols, block, 0.0f, inverse, head_dim, key_heads);
}

void fused_rmsnorm_hadamard_f32(const float* x, const float* weight, const int8_t* signs, float* out,
                               int rows, int cols, int block, float eps, cudaStream_t stream) {
    validate_hadamard_shape(rows, cols, block);
    k_hadamard<true><<<rows, std::min(256, block / 2), 0, stream>>>(
        x, weight, signs, out, cols, block, eps, false, 0, 0);
}

}  // namespace fusionrt::cuda_ops
