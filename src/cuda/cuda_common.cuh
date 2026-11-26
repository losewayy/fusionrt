#pragma once

#include "fusionrt/cuda_kernels.h"

#include <cuda_fp16.h>

#include <cmath>
#include <type_traits>

namespace fusionrt::cuda_ops {

namespace {

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float total = 0.0f;
    if (wid == 0) {
        total = (tid < num_warps) ? smem[tid] : 0.0f;
        total = warp_reduce_sum(total);
        if (tid == 0) smem[0] = total;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float half_bits_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

template <bool Value, typename T>
__device__ __forceinline__ T* kv_row(T* contiguous, CudaKvView cache, int token, int head, int heads, int dim) {
    if (!cache.pages) return contiguous + (static_cast<int64_t>(token) * heads + head) * dim;
    auto* page = cache.pages[token / kKvPageTokens];
    return reinterpret_cast<T*>(page + (Value ? cache.plane_bytes : 0)) +
           ((token % kKvPageTokens) * heads + head) * dim;
}

template <bool Value>
__device__ __forceinline__ half* kv_scales(CudaKvView cache, int token, int head, int heads, int dim) {
    auto* page = cache.pages[token / kKvPageTokens];
    return reinterpret_cast<half*>(page + (Value ? cache.plane_bytes : 0) + cache.elements) +
           ((token % kKvPageTokens) * heads + head) * (dim / 32);
}

template <bool Value, typename Cache>
__device__ __forceinline__ float kv_value(const Cache* row, CudaKvView cache,
                                         int token, int head, int heads, int dim, int d) {
    if constexpr (std::is_same_v<Cache, int8_t>) {
        return float(row[d]) * __half2float(kv_scales<Value>(cache, token, head, heads, dim)[d / 32]);
    } else {
        return static_cast<float>(row[d]);
    }
}

__device__ __forceinline__ void kv_quantize(float value, int8_t* bytes, half* scales, int i) {
    const float amax = __int_as_float(__reduce_max_sync(0xffffffffu, __float_as_int(fabsf(value))));
    const float d = amax * (1.0f / 127.0f);
    bytes[i] = static_cast<int8_t>(__float2int_rn(value * (amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f)));
    if ((i & 31) == 0) scales[i / 32] = __float2half_rn(d);
}

__device__ __forceinline__ void store_q8_1(float v, BlockQ8_1* out) {
    constexpr unsigned mask = 0xffffffffu;
    int lane = threadIdx.x & 31;
    float amax = __int_as_float(__reduce_max_sync(mask, __float_as_int(fabsf(v))));
    float d = amax * (1.0f / 127.0f);
    int q = __float2int_rn(v * (amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f));
    unsigned q0 = static_cast<unsigned>(q) & 255;
    unsigned q1 = __shfl_down_sync(mask, q0, 1);
    unsigned q2 = __shfl_down_sync(mask, q0, 2);
    unsigned q3 = __shfl_down_sync(mask, q0, 3);
    if ((lane & 3) == 0) out->qs[lane >> 2] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);

    unsigned half_warp = 0xffffu << (lane & 16);
    int sum = __reduce_add_sync(half_warp, q);
    int sum_hi = __shfl_sync(mask, sum, 16);
    if (lane == 0) {
        *reinterpret_cast<float4*>(&out->d) = make_float4(d, d * sum, d * sum_hi, d * (sum + sum_hi));
    }
}

}  // namespace

}  // namespace fusionrt::cuda_ops
