#include "../cuda_common.cuh"

namespace fusionrt::cuda_ops {
namespace {

__device__ void rotate_pair(const float* x, float* y, int pair, int pos,
                            const float* frequencies, float amplitude) {
    float s, c;
    sincosf(pos * frequencies[pair], &s, &c);
    const int i = 2 * pair;
    y[i] = (x[i] * c - x[i + 1] * s) * amplitude;
    y[i + 1] = (x[i] * s + x[i + 1] * c) * amplitude;
}

__global__ void k_mla_append(const float* kv, const float* norm, const float* frequencies,
                             CudaKvView cache, int rank, int rope, int start, const int* pos_dev,
                             float eps, float amplitude) {
    const int row = blockIdx.x;
    const int pos = (pos_dev ? *pos_dev : start) + row;
    const int width = rank + rope;
    const float* src = kv + static_cast<size_t>(row) * width;
    float* dst = kv_row<false>(static_cast<float*>(nullptr), cache, pos, 0, 1, width);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) sum += src[i] * src[i];
    __shared__ float reduction[32];
    const float rms = rsqrtf(block_reduce_sum(sum, reduction) / rank + eps);
    for (int i = threadIdx.x; i < rank; i += blockDim.x) dst[i] = src[i] * rms * norm[i];
    for (int i = threadIdx.x; i < rope / 2; i += blockDim.x)
        rotate_pair(src + rank, dst + rank, i, pos, frequencies, amplitude);
}

// Each warp projects one output channel of its head. The Q projection also
// copies the independently rotated positional channels beside the absorbed Q.
template <typename Weight, bool Query>
__global__ void k_mla_project(const float* x, const Weight* weights, float* out,
                              const float* frequencies, int heads, int in_dim, int out_dim,
                              int rope, int start, const int* pos_dev, float amplitude) {
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    const int lane = threadIdx.x & 31;
    const int input_stride = in_dim + (Query ? rope : 0);
    const int output_stride = out_dim + (Query ? rope : 0);
    const float* src = x + static_cast<size_t>(blockIdx.x) * input_stride;
    float* dst = out + static_cast<size_t>(blockIdx.x) * output_stride;
    const Weight* matrix = weights + static_cast<size_t>(head) * out_dim * in_dim;
    for (int o = threadIdx.x / 32; o < out_dim; o += blockDim.x / 32) {
        float sum = 0.0f;
        const Weight* w = matrix + static_cast<size_t>(o) * in_dim;
        for (int i = lane; i < in_dim; i += 32)
            sum = fmaf(src[i], static_cast<float>(w[i]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) dst[o] = sum;
    }
    if constexpr (Query) {
        const int pos = (pos_dev ? *pos_dev : start) + row;
        for (int i = threadIdx.x; i < rope / 2; i += blockDim.x)
            rotate_pair(src + in_dim, dst + out_dim, i, pos, frequencies, amplitude);
    }
}

__global__ void k_mla_causal(const float* queries, float* out, CudaKvView cache,
                             int heads, int rank, int width, int start, const int* pos_dev, float scale) {
    const int row = blockIdx.x / heads;
    const int last = (pos_dev ? *pos_dev : start) + row;
    const float* src = queries + static_cast<size_t>(blockIdx.x) * width;
    extern __shared__ float scratch[];
    float* q = scratch;
    float* acc = q + width;
    __shared__ float reduction[32];
    for (int i = threadIdx.x; i < width; i += blockDim.x) q[i] = src[i];
    for (int i = threadIdx.x; i < rank; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();
    float max_score = -INFINITY, mass = 0.0f;
    for (int token = 0; token <= last; ++token) {
        const float* kv = kv_row<false>(static_cast<float*>(nullptr), cache, token, 0, 1, width);
        float dot = 0.0f;
        for (int i = threadIdx.x; i < width; i += blockDim.x) dot = fmaf(q[i], kv[i], dot);
        const float score = block_reduce_sum(dot, reduction) * scale;
        const float peak = fmaxf(max_score, score);
        const float keep = expf(max_score - peak), add = expf(score - peak);
        for (int i = threadIdx.x; i < rank; i += blockDim.x) acc[i] = fmaf(add, kv[i], keep * acc[i]);
        mass = keep * mass + add;
        max_score = peak;
        __syncthreads();
    }
    float* dst = out + static_cast<size_t>(blockIdx.x) * rank;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) dst[i] = acc[i] / mass;
}

}  // namespace

void mla_attention_f32(const float* q, const float* kv, const float* kv_norm,
                       const Tensor& k_b, const Tensor& v_b, const float* frequencies,
                       float* q_absorbed, float* latent_out, float* out, CudaKvView cache,
                       int rows, int heads, int head_dim, int start_pos, const int* pos_dev,
                       float eps, float scale, float rotary_amplitude, cudaStream_t stream) {
    const int rank = static_cast<int>(k_b.shape[1]), nope = static_cast<int>(k_b.shape[2]);
    const int rope = head_dim - nope, width = rank + rope, value_dim = static_cast<int>(v_b.shape[1]);
    constexpr int threads = 128;
    const int blocks = rows * heads;
    k_mla_append<<<rows, threads, 0, stream>>>(kv, kv_norm, frequencies, cache, rank, rope,
                                             start_pos, pos_dev, eps, rotary_amplitude);
    if (k_b.dtype == DType::F32) {
        k_mla_project<float, true><<<blocks, threads, 0, stream>>>(q, k_b.ptr<float>(), q_absorbed, frequencies,
            heads, nope, rank, rope, start_pos, pos_dev, rotary_amplitude);
    } else {
        k_mla_project<half, true><<<blocks, threads, 0, stream>>>(q, k_b.ptr<half>(), q_absorbed, frequencies,
            heads, nope, rank, rope, start_pos, pos_dev, rotary_amplitude);
    }
    k_mla_causal<<<blocks, threads, (width + rank) * sizeof(float), stream>>>(
        q_absorbed, latent_out, cache, heads, rank, width, start_pos, pos_dev, scale);
    if (v_b.dtype == DType::F32) {
        k_mla_project<float, false><<<blocks, threads, 0, stream>>>(latent_out, v_b.ptr<float>(), out,
            nullptr, heads, rank, value_dim, 0, 0, nullptr, 1.0f);
    } else {
        k_mla_project<half, false><<<blocks, threads, 0, stream>>>(latent_out, v_b.ptr<half>(), out,
            nullptr, heads, rank, value_dim, 0, 0, nullptr, 1.0f);
    }
}

}  // namespace fusionrt::cuda_ops
