#include "fusionrt/cuda_kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float half_bits_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// =============================================================================
// Unfused Baseline Kernels
// =============================================================================

__global__ void k_rmsnorm_f32(const float* __restrict__ x, const float* __restrict__ w,
                              float* __restrict__ y, int cols, float eps) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    float sum_sq = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = xr[c];
        sum_sq += v * v;
    }
    float total_sq = block_reduce_sum(sum_sq, smem);
    float inv_rms = rsqrtf(total_sq / static_cast<float>(cols) + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        yr[c] = xr[c] * inv_rms * w[c];
    }
}

__global__ void k_rmsnorm_f16(const half* __restrict__ x, const half* __restrict__ w,
                              half* __restrict__ y, int cols, float eps) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    const half* xr = x + row * cols;
    half* yr = y + row * cols;

    float sum_sq = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(xr[c]);
        sum_sq += v * v;
    }
    float total_sq = block_reduce_sum(sum_sq, smem);
    float inv_rms = rsqrtf(total_sq / static_cast<float>(cols) + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        yr[c] = __float2half(__half2float(xr[c]) * inv_rms * __half2float(w[c]));
    }
}

// Unfused GEMV/GEMM for M=1..8 (warp-per-row for each output row)
__global__ void k_linear_gemv_f32(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ y, int M, int N, int K) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= N) return;

    const float* wr = w + static_cast<int64_t>(wid) * K;
    float acc[8] = {0.f};

    int k4 = K >> 2;
    const float4* wr4 = reinterpret_cast<const float4*>(wr);
    for (int i = lane; i < k4; i += 32) {
        float4 wv = wr4[i];
        int base_k = i << 2;
        for (int m = 0; m < M; ++m) {
            float4 xv = *reinterpret_cast<const float4*>(x + m * K + base_k);
            acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    }
    for (int k = (k4 << 2) + lane; k < K; k += 32) {
        float wv = wr[k];
        for (int m = 0; m < M; ++m) {
            acc[m] += x[m * K + k] * wv;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if (lane == 0) y[m * N + wid] = s;
    }
}

__global__ void k_linear_gemm_tiled_f32(const float* __restrict__ x, const float* __restrict__ w,
                                        float* __restrict__ y, int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float sx[TILE][TILE];
    __shared__ float sw[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // m
    int col = blockIdx.x * TILE + threadIdx.x;  // n
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int kx = t + threadIdx.x;
        int kw = t + threadIdx.y;
        sx[threadIdx.y][threadIdx.x] = (row < M && kx < K) ? x[row * K + kx] : 0.0f;
        sw[threadIdx.y][threadIdx.x] = (col < N && kw < K) ? w[static_cast<int64_t>(col) * K + kw] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            acc += sx[threadIdx.y][k] * sw[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        y[row * N + col] = acc;
    }
}

__global__ void k_linear_gemv_f16(const half* __restrict__ x, const half* __restrict__ w,
                                  half* __restrict__ y, int M, int N, int K) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= N) return;

    const half* wr = w + static_cast<int64_t>(wid) * K;
    float acc[8] = {0.f};
    for (int k = lane; k < K; k += 32) {
        float wv = __half2float(wr[k]);
        for (int m = 0; m < M; ++m) {
            acc[m] += __half2float(x[m * K + k]) * wv;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if (lane == 0) y[m * N + wid] = __float2half(s);
    }
}

__global__ void k_dequant_q8_0_to_f32(const BlockQ8_0* __restrict__ src, float* __restrict__ dst, int N, int K) {
    int row = blockIdx.y;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = K >> 5;
    if (row >= N || blk >= nb) return;
    const BlockQ8_0 b = src[static_cast<int64_t>(row) * nb + blk];
    float d = __half2float(*reinterpret_cast<const half*>(&b.d));
    float* out = dst + static_cast<int64_t>(row) * K + blk * 32;
    for (int i = 0; i < 32; ++i) {
        out[i] = static_cast<float>(b.qs[i]) * d;
    }
}

__global__ void k_dequant_q4_0_to_f32(const BlockQ4_0* __restrict__ src, float* __restrict__ dst, int N, int K) {
    int row = blockIdx.y;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = K >> 5;
    if (row >= N || blk >= nb) return;
    const BlockQ4_0 b = src[static_cast<int64_t>(row) * nb + blk];
    float d = __half2float(*reinterpret_cast<const half*>(&b.d));
    float* out = dst + static_cast<int64_t>(row) * K + blk * 32;
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = b.qs[i];
        int q0 = static_cast<int>(byte & 0x0Fu) - 8;
        int q1 = static_cast<int>(byte >> 4) - 8;
        out[i] = static_cast<float>(q0) * d;
        out[i + 16] = static_cast<float>(q1) * d;
    }
}

__global__ void k_rope_f32(float* __restrict__ x, int num_heads, int head_dim, int rot_dim,
                           int pos, float theta, bool neox) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half_rot = rot_dim >> 1;
    if (h >= num_heads || i >= half_rot) return;

    float* xh = x + h * head_dim;
    float freq = powf(theta, -static_cast<float>(2 * i) / static_cast<float>(rot_dim));
    float angle = static_cast<float>(pos) * freq;
    float cs = cosf(angle);
    float sn = sinf(angle);
    if (neox) {
        float x0 = xh[i];
        float x1 = xh[i + half_rot];
        xh[i] = x0 * cs - x1 * sn;
        xh[i + half_rot] = x0 * sn + x1 * cs;
    } else {
        float x0 = xh[2 * i];
        float x1 = xh[2 * i + 1];
        xh[2 * i] = x0 * cs - x1 * sn;
        xh[2 * i + 1] = x0 * sn + x1 * cs;
    }
}

__global__ void k_kv_append_f32(const float* __restrict__ k_in, const float* __restrict__ v_in,
                                float* __restrict__ k_cache, float* __restrict__ v_cache,
                                int num_kv_heads, int head_dim, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (idx >= total) return;
    int64_t dst_off = static_cast<int64_t>(pos) * total + idx;
    k_cache[dst_off] = k_in[idx];
    v_cache[dst_off] = v_in[idx];
}

__global__ void k_swiglu_f32(const float* __restrict__ gate, const float* __restrict__ up,
                             float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = silu_f(gate[idx]) * up[idx];
}

__global__ void k_swiglu_interleaved_f32(const float* __restrict__ gate_up, float* __restrict__ out,
                                         int rows, int inter_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * inter_dim;
    if (idx >= total) return;
    int r = idx / inter_dim;
    int c = idx % inter_dim;
    float g = gate_up[r * (2 * inter_dim) + c];
    float u = gate_up[r * (2 * inter_dim) + inter_dim + c];
    out[idx] = silu_f(g) * u;
}

__global__ void k_add_residual_f32(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void k_split_q_gate_f32(const float* __restrict__ q_full,
                                   float* __restrict__ q_out,
                                   float* __restrict__ gate_out,
                                   int M, int num_q_heads, int head_dim, int src_row_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_dim = num_q_heads * head_dim;
    int total = M * q_dim;
    if (idx >= total) return;
    int m = idx / q_dim;
    int rem = idx % q_dim;
    int h = rem / head_dim;
    int d = rem % head_dim;
    int src_off = m * src_row_stride + h * (2 * head_dim) + d;
    q_out[idx] = q_full[src_off];
    gate_out[idx] = q_full[src_off + head_dim];
}

__global__ void k_apply_sigmoid_gate_f32(float* __restrict__ attn_out,
                                         const float* __restrict__ gate,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float s = 1.0f / (1.0f + expf(-gate[idx]));
    attn_out[idx] *= s;
}

__global__ void k_argmax_f32(const float* __restrict__ x, int n, int32_t* __restrict__ out_idx,
                             float* partial_values, const int32_t* indices, int* pos, int* kv_len, int history_size) {
    __shared__ float sv[32];
    __shared__ int32_t si[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    float best_v = -INFINITY;
    int32_t best_i = 0;

    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        float v = __ldg(x + i);
        if (v > best_v) {
            best_v = v;
            best_i = indices ? indices[i] : i;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFFu, best_v, off);
        int32_t oi = __shfl_down_sync(0xFFFFFFFFu, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        sv[wid] = best_v;
        si[wid] = best_i;
    }
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -INFINITY;
        best_i = (lane < nwarps) ? si[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xFFFFFFFFu, best_v, off);
            int32_t oi = __shfl_down_sync(0xFFFFFFFFu, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            out_idx[blockIdx.x] = best_i;
            if (partial_values) partial_values[blockIdx.x] = best_v;
            if (pos) {
                int i = *pos;
                if (static_cast<unsigned>(i) < static_cast<unsigned>(history_size)) out_idx[1 + i] = best_i;
                *pos = i + 1;
            }
            if (kv_len) ++*kv_len;
        }
    }
}

// =============================================================================
// Fusion 1: Fused Pre-Norm (RMSNorm) + Linear GEMV (F32, Q8_0, Q4_0)
// =============================================================================
// Each CTA has WARPS_PER_BLOCK = 8 warps (256 threads) computing 8 output channels.
// Uses dot(x * inv_rms * norm_w, w) == inv_rms * dot(x * norm_w, w) so only
// inv_rms[M] (32B) needs shared memory, preserving full SM occupancy on sm_120.
template <int WARPS_PER_BLOCK>
__global__ void k_fused_rmsnorm_linear_f32(const float* __restrict__ x,
                                           const float* __restrict__ norm_w,
                                           const float* __restrict__ w,
                                           float* __restrict__ y,
                                           int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    for (int m = 0; m < M; ++m) {
        const float* xm = x + m * K;
        float sum_sq = 0.0f;
        int k4 = K >> 2;
        const float4* xm4 = reinterpret_cast<const float4*>(xm);
        for (int i = tid; i < k4; i += blockDim.x) {
            float4 v = xm4[i];
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int k = (k4 << 2) + tid; k < K; k += blockDim.x) {
            float v = xm[k];
            sum_sq += v * v;
        }
        float total_sq = block_reduce_sum(sum_sq, red_buf);
        if (tid == 0) {
            inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
    }
    __syncthreads();

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    const float* wr = w + static_cast<int64_t>(n) * K;
    float acc[8] = {0.f};

    int k4 = K >> 2;
    const float4* wr4 = reinterpret_cast<const float4*>(wr);
    const float4* nw4 = reinterpret_cast<const float4*>(norm_w);
    for (int i = lane; i < k4; i += 32) {
        float4 wv = wr4[i];
        float4 nw = nw4[i];
        wv.x *= nw.x; wv.y *= nw.y; wv.z *= nw.z; wv.w *= nw.w;
        for (int m = 0; m < M; ++m) {
            float4 xv = reinterpret_cast<const float4*>(x + m * K)[i];
            acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    }
    for (int k = (k4 << 2) + lane; k < K; k += 32) {
        float wv = wr[k] * norm_w[k];
        for (int m = 0; m < M; ++m) {
            acc[m] += x[m * K + k] * wv;
        }
    }

    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]) * inv_rms_sh[m];
        if (lane == 0) y[m * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_quant_gemv_q8_0(const float* __restrict__ x,
                                  const float* __restrict__ norm_w,
                                  const BlockQ8_0* __restrict__ w,
                                  float* __restrict__ y,
                                  int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            for (int k = (k4 << 2) + tid; k < K; k += blockDim.x) {
                float v = __ldg(xm + k);
                sum_sq += v * v;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 5;
    const BlockQ8_0* wb = w + static_cast<int64_t>(n) * nb;
    float acc[8] = {0.f};

    // 8 threads per 32-elem Q8_0 block -> 4 blocks processed in parallel per warp step
    int sub = lane & 7;
    int bi = lane >> 3;
    for (int b = bi; b < nb; b += 4) {
        const BlockQ8_0* bp = wb + b;
        float d = half_bits_to_f32(__ldg(&bp->d));
        const uint16_t* q16 = reinterpret_cast<const uint16_t*>(bp->qs) + (sub << 1);
        uint16_t u0 = __ldg(q16);
        uint16_t u1 = __ldg(q16 + 1);
        float q0 = static_cast<float>(static_cast<int8_t>(u0 & 0xFFu)) * d;
        float q1 = static_cast<float>(static_cast<int8_t>(u0 >> 8)) * d;
        float q2 = static_cast<float>(static_cast<int8_t>(u1 & 0xFFu)) * d;
        float q3 = static_cast<float>(static_cast<int8_t>(u1 >> 8)) * d;
        int k_off = (b << 5) + (sub << 2);
        if constexpr (HAS_NORM) {
            float4 nw4 = __ldg(reinterpret_cast<const float4*>(norm_w + k_off));
            q0 *= nw4.x; q1 *= nw4.y; q2 *= nw4.z; q3 *= nw4.w;
        }
        for (int m = 0; m < M; ++m) {
            float4 x4 = __ldg(reinterpret_cast<const float4*>(x + static_cast<int64_t>(m) * K + k_off));
            acc[m] += q0 * x4.x + q1 * x4.y + q2 * x4.z + q3 * x4.w;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_quant_gemv_q4_0(const float* __restrict__ x,
                                  const float* __restrict__ norm_w,
                                  const BlockQ4_0* __restrict__ w,
                                  float* __restrict__ y,
                                  int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            for (int k = (k4 << 2) + tid; k < K; k += blockDim.x) {
                float v = __ldg(xm + k);
                sum_sq += v * v;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 5;
    const BlockQ4_0* wb = w + static_cast<int64_t>(n) * nb;
    float acc[8] = {0.f};

    // 8 threads per 32-elem Q4_0 block -> each thread reads 1 aligned uint16_t (4 nibbles)
    int sub = lane & 7;
    int bi = lane >> 3;
    for (int b = bi; b < nb; b += 4) {
        const BlockQ4_0* bp = wb + b;
        float d = half_bits_to_f32(__ldg(&bp->d));
        uint16_t raw2 = __ldg(reinterpret_cast<const uint16_t*>(bp->qs) + sub);
        uint8_t b0 = raw2 & 0xFFu;
        uint8_t b1 = raw2 >> 8;
        float q0 = static_cast<float>(static_cast<int>(b0 & 0x0Fu) - 8) * d;
        float q1 = static_cast<float>(static_cast<int>(b1 & 0x0Fu) - 8) * d;
        float q2 = static_cast<float>(static_cast<int>(b0 >> 4) - 8) * d;
        float q3 = static_cast<float>(static_cast<int>(b1 >> 4) - 8) * d;
        int k_lo = (b << 5) + (sub << 1);
        int k_hi = k_lo + 16;
        if constexpr (HAS_NORM) {
            float2 nwl = __ldg(reinterpret_cast<const float2*>(norm_w + k_lo));
            float2 nwh = __ldg(reinterpret_cast<const float2*>(norm_w + k_hi));
            q0 *= nwl.x; q1 *= nwl.y;
            q2 *= nwh.x; q3 *= nwh.y;
        }
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float2 xl = __ldg(reinterpret_cast<const float2*>(xm + k_lo));
            float2 xh = __ldg(reinterpret_cast<const float2*>(xm + k_hi));
            acc[m] += q0 * xl.x + q1 * xl.y + q2 * xh.x + q3 * xh.y;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_quant_gemv_q4_1(const float* __restrict__ x,
                                  const float* __restrict__ norm_w,
                                  const BlockQ4_1* __restrict__ w,
                                  float* __restrict__ y,
                                  int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            for (int k = (k4 << 2) + tid; k < K; k += blockDim.x) {
                float v = __ldg(xm + k);
                sum_sq += v * v;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 5;
    const BlockQ4_1* wb = w + static_cast<int64_t>(n) * nb;
    float acc[8] = {0.f};

    int sub = lane & 7;
    int bi = lane >> 3;
    for (int b = bi; b < nb; b += 4) {
        const BlockQ4_1* bp = wb + b;
        float d = half_bits_to_f32(__ldg(&bp->d));
        float mb = half_bits_to_f32(__ldg(&bp->m));
        uint16_t raw2 = __ldg(reinterpret_cast<const uint16_t*>(bp->qs) + sub);
        uint8_t b0 = raw2 & 0xFFu;
        uint8_t b1 = raw2 >> 8;
        float q0 = static_cast<float>(b0 & 0x0Fu) * d + mb;
        float q1 = static_cast<float>(b1 & 0x0Fu) * d + mb;
        float q2 = static_cast<float>(b0 >> 4) * d + mb;
        float q3 = static_cast<float>(b1 >> 4) * d + mb;
        int k_lo = (b << 5) + (sub << 1);
        int k_hi = k_lo + 16;
        if constexpr (HAS_NORM) {
            float2 nwl = __ldg(reinterpret_cast<const float2*>(norm_w + k_lo));
            float2 nwh = __ldg(reinterpret_cast<const float2*>(norm_w + k_hi));
            q0 *= nwl.x; q1 *= nwl.y;
            q2 *= nwh.x; q3 *= nwh.y;
        }
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float2 xl = __ldg(reinterpret_cast<const float2*>(xm + k_lo));
            float2 xh = __ldg(reinterpret_cast<const float2*>(xm + k_hi));
            acc[m] += q0 * xl.x + q1 * xl.y + q2 * xh.x + q3 * xh.y;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM, typename W = BlockQ4_K>
__global__ void k_quant_gemv_q4_q5_k(const float* __restrict__ x,
                                  const float* __restrict__ norm_w,
                                  const W* __restrict__ w,
                                  float* __restrict__ y,
                                  int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 8;
    const W* wb = w + static_cast<int64_t>(n) * nb;
    float acc[8] = {0.f};
    auto get_sc = [](int j, const uint8_t* q, uint8_t& sc, uint8_t& mn) {
        if (j < 4) {
            sc = q[j] & 63u;
            mn = q[j + 4] & 63u;
        } else {
            sc = static_cast<uint8_t>((q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4));
            mn = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    };

    for (int b = 0; b < nb; ++b) {
        const W* bp = wb + b;
        float d = half_bits_to_f32(__ldg(&bp->d));
        float dmin = half_bits_to_f32(__ldg(&bp->dmin));
        for (int g = 0; g < 4; ++g) {
            uint8_t sc0, m0, sc1, m1;
            get_sc(2 * g, bp->scales, sc0, m0);
            get_sc(2 * g + 1, bp->scales, sc1, m1);
            float d1 = d * static_cast<float>(sc0), mv1 = dmin * static_cast<float>(m0);
            float d2 = d * static_cast<float>(sc1), mv2 = dmin * static_cast<float>(m1);
            uint8_t qb = __ldg(&bp->qs[g * 32 + lane]);
            int lo = qb & 15;
            int hi = qb >> 4;
            if constexpr (std::is_same_v<W, BlockQ5_K>) {
                int bits = __ldg(&bp->qh[lane]) >> (2 * g);
                lo |= (bits & 1) << 4;
                hi |= (bits & 2) << 3;
            }
            float w0 = d1 * lo - mv1;
            float w1 = d2 * hi - mv2;
            int k0 = (b << 8) + g * 64 + lane;
            int k1 = k0 + 32;
            if constexpr (HAS_NORM) {
                w0 *= __ldg(&norm_w[k0]);
                w1 *= __ldg(&norm_w[k1]);
            }
            for (int m = 0; m < M; ++m) {
                const float* xm = x + static_cast<int64_t>(m) * K;
                acc[m] += w0 * __ldg(&xm[k0]) + w1 * __ldg(&xm[k1]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_quant_gemv_q6_k(const float* __restrict__ x,
                                  const float* __restrict__ norm_w,
                                  const BlockQ6_K* __restrict__ w,
                                  float* __restrict__ y,
                                  int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 8;
    const BlockQ6_K* wb = w + static_cast<int64_t>(n) * nb;
    float acc[8] = {0.f};
    int is = lane >> 4;

    for (int b = 0; b < nb; ++b) {
        const BlockQ6_K* bp = wb + b;
        float d = half_bits_to_f32(__ldg(&bp->d));
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const uint8_t* ql = bp->ql + half_idx * 64;
            const uint8_t* qh = bp->qh + half_idx * 32;
            const int8_t* sc = bp->scales + half_idx * 8;
            uint8_t ql0 = __ldg(&ql[lane]);
            uint8_t ql1 = __ldg(&ql[lane + 32]);
            uint8_t qhv = __ldg(&qh[lane]);
            float s0 = d * static_cast<float>(__ldg(&sc[is + 0]));
            float s2 = d * static_cast<float>(__ldg(&sc[is + 2]));
            float s4 = d * static_cast<float>(__ldg(&sc[is + 4]));
            float s6 = d * static_cast<float>(__ldg(&sc[is + 6]));
            float w0 = s0 * static_cast<float>(static_cast<int>((ql0 & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32);
            float w1 = s2 * static_cast<float>(static_cast<int>((ql1 & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32);
            float w2 = s4 * static_cast<float>(static_cast<int>((ql0 >> 4) | (((qhv >> 4) & 3u) << 4)) - 32);
            float w3 = s6 * static_cast<float>(static_cast<int>((ql1 >> 4) | (((qhv >> 6) & 3u) << 4)) - 32);
            int kbase = (b << 8) + half_idx * 128 + lane;
            if constexpr (HAS_NORM) {
                w0 *= __ldg(&norm_w[kbase + 0]);
                w1 *= __ldg(&norm_w[kbase + 32]);
                w2 *= __ldg(&norm_w[kbase + 64]);
                w3 *= __ldg(&norm_w[kbase + 96]);
            }
            for (int m = 0; m < M; ++m) {
                const float* xm = x + static_cast<int64_t>(m) * K + kbase;
                acc[m] += w0 * __ldg(&xm[0]) + w1 * __ldg(&xm[32]) + w2 * __ldg(&xm[64]) + w3 * __ldg(&xm[96]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

// =============================================================================
// Fusion 2: Single-Kernel Register-Resident GatedDeltaNet Decode Step
// =============================================================================
// 1 CTA per k-head group (so conv_state for shared q/k channels is updated ONCE).
// blockDim.x = dv (must be <= 256, multiple of 32 or >= dv).
template <int MAX_DK>
__global__ void k_fused_gated_deltanet_step(
    const float* __restrict__ qkv_proj,
    const float* __restrict__ z,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w,
    float* __restrict__ conv_state,
    float* __restrict__ ssm_state,
    float* __restrict__ out,
    int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, bool tiled_value_heads)
{
    extern __shared__ float smem[];
    float* red_buf = smem;               // [32]
    float* q_sh = red_buf + 32;          // [dk]
    float* k_sh = q_sh + dk;             // [dk]
    float* v_sh = k_sh + dk;             // [dv]

    int kh = blockIdx.x;
    if (kh >= num_k_heads) return;
    int tid = threadIdx.x;
    int ratio = num_v_heads / num_k_heads;
    int q_dim = num_k_heads * dk;

    auto apply_conv_silu = [&](int ch) -> float {
        float* cs = conv_state + ch * 3;
        const float* cw = conv_w + ch * 4;
        float c0 = cs[0], c1 = cs[1], c2 = cs[2];
        float cur = qkv_proj[ch];
        float acc = c0 * cw[0] + c1 * cw[1] + c2 * cw[2] + cur * cw[3];
        if (conv_bias) acc += conv_bias[ch];
        cs[0] = c1;
        cs[1] = c2;
        cs[2] = cur;
        return silu_f(acc);
    };

    // 1. Causal Conv1d + SiLU on q[kh] and k[kh]
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    for (int i = tid; i < dk; i += blockDim.x) {
        float qv = apply_conv_silu(kh * dk + i);
        float kv = apply_conv_silu(q_dim + kh * dk + i);
        q_sh[i] = qv;
        k_sh[i] = kv;
        q_sq += qv * qv;
        k_sq += kv * kv;
    }

    // 2. L2 normalize q and k per head
    float q_inv = rsqrtf(block_reduce_sum(q_sq, red_buf) + 1e-6f);
    __syncthreads(); // All warps must read the Q sum before K reuses red_buf.
    float k_inv = rsqrtf(block_reduce_sum(k_sq, red_buf) + 1e-6f);
    for (int i = tid; i < dk; i += blockDim.x) {
        q_sh[i] = q_sh[i] * q_inv * q_scale;
        k_sh[i] = k_sh[i] * k_inv;
    }
    __syncthreads();

    // 3. Process each associated v-head in registers
    int v_offset = 2 * q_dim;
    for (int r = 0; r < ratio; ++r) {
        int vh = tiled_value_heads ? r * num_k_heads + kh : kh * ratio + r;
        if (tid < dv) {
            v_sh[tid] = apply_conv_silu(v_offset + vh * dv + tid);
        }
        __syncthreads();

        float g = expf(-expf(A_log[vh]) * softplus_f(a[vh] + dt_bias[vh]));
        float beta = 1.0f / (1.0f + expf(-b[vh]));

        float y_j = 0.0f;
        if (tid < dv) {
            float* S_vh = ssm_state + static_cast<int64_t>(vh) * dk * dv;
            float s_reg[MAX_DK];
            float kv_mem = 0.0f;
            for (int i = 0; i < dk; ++i) {
                float s_val = S_vh[i * dv + tid] * g;
                s_reg[i] = s_val;
                kv_mem += s_val * k_sh[i];
            }
            float delta = (v_sh[tid] - kv_mem) * beta;
            for (int i = 0; i < dk; ++i) {
                float s_new = s_reg[i] + k_sh[i] * delta;
                S_vh[i * dv + tid] = s_new;
                y_j += s_new * q_sh[i];
            }
        }

        // 4. Output Gated RMSNorm: rmsnorm(y) * silu(z)
        float y_sq = (tid < dv) ? (y_j * y_j) : 0.0f;
        float y_inv = rsqrtf(block_reduce_sum(y_sq, red_buf) / static_cast<float>(dv) + eps);
        if (tid < dv) {
            float z_val = z[vh * dv + tid];
            out[vh * dv + tid] = (y_j * y_inv * norm_w[tid]) * silu_f(z_val);
        }
        __syncthreads();
    }
}

// Unfused GatedDeltaNet kernels (for ablation & differential test)
__device__ __forceinline__ float gdn_conv_silu(const float* qkv_proj, const float* conv_w,
                                                  const float* conv_bias, float* conv_state, int ch) {
    float* cs = conv_state + ch * 3;
    const float* cw = conv_w + ch * 4;
    float c0 = cs[0], c1 = cs[1], c2 = cs[2];
    float cur = qkv_proj[ch];
    float acc = c0 * cw[0] + c1 * cw[1] + c2 * cw[2] + cur * cw[3];
    if (conv_bias) acc += conv_bias[ch];
    cs[0] = c1;
    cs[1] = c2;
    cs[2] = cur;
    return silu_f(acc);
}

__global__ void k_unfused_gdn_conv_silu(const float* __restrict__ qkv_proj,
                                        const float* __restrict__ conv_w,
                                        const float* __restrict__ conv_bias,
                                        float* __restrict__ conv_state,
                                        float* __restrict__ qkv_out,
                                        int conv_dim) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch < conv_dim) qkv_out[ch] = gdn_conv_silu(qkv_proj, conv_w, conv_bias, conv_state, ch);
}

constexpr int kGdnDvSplits = 4;
constexpr int kGdnDkWarps = 4;

__global__ void k_gdn_decode_conv_qk(const float* qkv_proj, const float* a, const float* b,
                                    const float* conv_w, const float* conv_bias, const float* A_log,
                                    const float* dt_bias, float* conv_state, float* qkv,
                                    float* alpha, float* beta, int nk, int dk, int dv, float q_scale) {
    __shared__ float reduce[32];
    int head = blockIdx.x;
    bool is_qk = head < 2 * nk;
    int width = is_qk ? dk : dv;
    int offset = is_qk ? head * dk : 2 * nk * dk + (head - 2 * nk) * dv;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        float v = gdn_conv_silu(qkv_proj, conv_w, conv_bias, conv_state, offset + i);
        qkv[offset + i] = v;
        sum += v * v;
    }
    if (is_qk) {
        float inv = rsqrtf(block_reduce_sum(sum, reduce) + 1e-6f);
        float scale = head < nk ? q_scale : 1.0f;
        for (int i = tid; i < width; i += blockDim.x) qkv[offset + i] *= inv * scale;
    } else if (tid == 0) {
        int vh = head - 2 * nk;
        alpha[vh] = expf(-expf(A_log[vh]) * softplus_f(a[vh] + dt_bias[vh]));
        beta[vh] = 1.0f / (1.0f + expf(-b[vh]));
    }
}

template <int DK>
__global__ void k_gdn_decode_ssm(const float* __restrict__ qkv, const float* __restrict__ alpha,
                                 const float* __restrict__ beta, float* __restrict__ state,
                                 float* __restrict__ y, int nk, int nv, int dv, bool tiled_value_heads) {
    __shared__ float partial[kGdnDkWarps][32];
    int vh = blockIdx.x / kGdnDvSplits;
    int kh = tiled_value_heads ? vh % nk : vh / (nv / nk);
    int slice = blockIdx.x % kGdnDvSplits;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    constexpr int chunk = DK / kGdnDkWarps;
    const float* q = qkv + kh * DK + warp * chunk;
    const float* k = q + nk * DK;
    const float* v = qkv + 2 * nk * DK + vh * dv;
    float* s = state + (static_cast<int64_t>(vh) * DK + warp * chunk) * dv;
    float decay = alpha[vh];
    float rate = beta[vh];
    int columns = (dv + kGdnDvSplits - 1) / kGdnDvSplits;

    for (int col = lane; col - lane < columns; col += 32) {
        int j = slice * columns + col;
        bool active = col < columns && j < dv;
        float saved[chunk];
        float kv = 0.0f;
        if (active) {
            #pragma unroll
            for (int i = 0; i < chunk; ++i) {
                float sd = s[i * dv + j] * decay;
                saved[i] = sd;
                kv += sd * k[i];
            }
        }
        partial[warp][lane] = kv;
        __syncthreads();
        float kv_total = partial[0][lane] + partial[1][lane] + partial[2][lane] + partial[3][lane];
        __syncthreads();

        float out = 0.0f;
        if (active) {
            float delta = (v[j] - kv_total) * rate;
            #pragma unroll
            for (int i = 0; i < chunk; ++i) {
                float next = saved[i] + k[i] * delta;
                s[i * dv + j] = next;
                out += next * q[i];
            }
        }
        partial[warp][lane] = out;
        __syncthreads();
        if (warp == 0 && active) {
            y[vh * dv + j] = partial[0][lane] + partial[1][lane] + partial[2][lane] + partial[3][lane];
        }
        __syncthreads();
    }
}

__global__ void k_unfused_gdn_l2norm(float* __restrict__ x, int num_heads, int dk, float scale) {
    __shared__ float smem[32];
    int h = blockIdx.x;
    if (h >= num_heads) return;
    float* xh = x + h * dk;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) ss += xh[i] * xh[i];
    float inv = rsqrtf(block_reduce_sum(ss, smem) + 1e-6f) * scale;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) xh[i] *= inv;
}

__global__ void k_unfused_gdn_state_step(const float* __restrict__ qkv,
                                         const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         const float* __restrict__ A_log,
                                         const float* __restrict__ dt_bias,
                                         float* __restrict__ ssm_state,
                                         float* __restrict__ y_out,
                                         int num_k_heads, int num_v_heads, int dk, int dv, bool tiled_value_heads) {
    int vh = blockIdx.x;
    int j = threadIdx.x;
    if (vh >= num_v_heads || j >= dv) return;
    int kh = tiled_value_heads ? vh % num_k_heads : vh / (num_v_heads / num_k_heads);
    const float* qh = qkv + kh * dk;
    const float* kh_p = qkv + num_k_heads * dk + kh * dk;
    const float* vh_p = qkv + 2 * num_k_heads * dk + vh * dv;

    float g = expf(-expf(A_log[vh]) * softplus_f(a[vh] + dt_bias[vh]));
    float beta = 1.0f / (1.0f + expf(-b[vh]));
    float* S_vh = ssm_state + static_cast<int64_t>(vh) * dk * dv;

    float kv_mem = 0.0f;
    for (int i = 0; i < dk; ++i) {
        float sv = S_vh[i * dv + j] * g;
        S_vh[i * dv + j] = sv;
        kv_mem += sv * kh_p[i];
    }
    float delta = (vh_p[j] - kv_mem) * beta;
    float y_j = 0.0f;
    for (int i = 0; i < dk; ++i) {
        float s_new = S_vh[i * dv + j] + kh_p[i] * delta;
        S_vh[i * dv + j] = s_new;
        y_j += s_new * qh[i];
    }
    y_out[vh * dv + j] = y_j;
}

__global__ void k_gdn_decode_gated_norm(const float* __restrict__ y,
                                            const float* __restrict__ z,
                                            const float* __restrict__ norm_w,
                                            float* __restrict__ out,
                                            int num_v_heads, int dv, float eps) {
    __shared__ float smem[32];
    int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const float* yh = y + vh * dv;
    const float* zh = z + vh * dv;
    float* oh = out + vh * dv;
    float ss = 0.0f;
    for (int j = threadIdx.x; j < dv; j += blockDim.x) ss += yh[j] * yh[j];
    float inv = rsqrtf(block_reduce_sum(ss, smem) / static_cast<float>(dv) + eps);
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        oh[j] = (yh[j] * inv * norm_w[j]) * silu_f(zh[j]);
    }
}

// =============================================================================
// Fusion 3: Fused Q/K RMSNorm + RoPE (Full/Partial) + KV Cache Append
// =============================================================================
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

template <typename Cache>
__global__ void k_qk_norm_rope_kv_append(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ q_norm_w,
    const float* __restrict__ k_norm_w,
    const float* __restrict__ rotary_factors,
    Cache* __restrict__ k_cache,
    Cache* __restrict__ v_cache,
    int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    const int* __restrict__ pos_dev, int pos_host,
    float theta, float eps, bool neox, int q_stride, int k_stride, int v_stride, CudaKvView cache)
{
    extern __shared__ float smem[];
    float* red_buf = smem;      // [32]
    float* h_buf = smem + 32;   // [head_dim]

    int h = blockIdx.x;
    int tid = threadIdx.x;
    const int m = blockIdx.y;
    int pos = pos_dev ? *pos_dev : pos_host + m;
    bool is_q = (h < num_q_heads);
    int kh = is_q ? h : (h - num_q_heads);
    if (!is_q && kh >= num_kv_heads) return;

    float* vec = is_q ? (q + m * q_stride + h * head_dim) : (k + m * k_stride + kh * head_dim);
    const float* nw = is_q ? q_norm_w : k_norm_w;

    float sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = vec[i];
        h_buf[i] = val;
        sq += val * val;
    }
    __syncthreads();

    if (nw != nullptr) {
        float inv = rsqrtf(block_reduce_sum(sq, red_buf) / static_cast<float>(head_dim) + eps);
        for (int i = tid; i < head_dim; i += blockDim.x) {
            h_buf[i] = h_buf[i] * inv * nw[i];
        }
        __syncthreads();
    }

    int half_rot = rot_dim >> 1;
    for (int i = tid; i < half_rot; i += blockDim.x) {
        float freq = powf(theta, -static_cast<float>(2 * i) / static_cast<float>(rot_dim));
        float angle = static_cast<float>(pos) * freq;
        if (rotary_factors) angle /= rotary_factors[i];
        float cs = cosf(angle);
        float sn = sinf(angle);
        if (neox) {
            float x0 = h_buf[i];
            float x1 = h_buf[i + half_rot];
            h_buf[i] = x0 * cs - x1 * sn;
            h_buf[i + half_rot] = x0 * sn + x1 * cs;
        } else {
            float x0 = h_buf[2 * i];
            float x1 = h_buf[2 * i + 1];
            h_buf[2 * i] = x0 * cs - x1 * sn;
            h_buf[2 * i + 1] = x0 * sn + x1 * cs;
        }
    }
    __syncthreads();

    if (is_q) {
        for (int i = tid; i < head_dim; i += blockDim.x) vec[i] = h_buf[i];
        return;
    }
    auto* keys = kv_row<false>(k_cache, cache, pos, kh, num_kv_heads, head_dim);
    auto* values = kv_row<true>(v_cache, cache, pos, kh, num_kv_heads, head_dim);
    const auto* src_v = v + m * v_stride + kh * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        vec[i] = h_buf[i];
        if constexpr (std::is_same_v<Cache, int8_t>) {
            kv_quantize(h_buf[i], keys, kv_scales<false>(cache, pos, kh, num_kv_heads, head_dim), i);
            kv_quantize(src_v[i], values, kv_scales<true>(cache, pos, kh, num_kv_heads, head_dim), i);
        } else {
            keys[i] = h_buf[i];
            values[i] = src_v[i];
        }
    }
}

// =============================================================================
// Fusion 4: Tensor Core (WMMA FP16) Prefill GEMM + Vectorized W16 & Embed Lookup
// =============================================================================
// Weight kind enum for unified WMMA prefill kernel:
// 0=F16, 1=BF16, 2=Q8_0, 3=Q4_0, 4=Q4_1, 5=Q4_K, 6=Q6_K, 7=Q5_K, 8=Q4_KA, 9=Q6_A
__device__ __forceinline__ float2 q4_ka_scales(const BlockQ4_KA& b, int sub) {
    const float2 base = __half22float2(*reinterpret_cast<const half2*>(&b.d));
    const unsigned factor = unpack_q4_ka_scales(b, sub);
    return make_float2(base.x * (factor & 63), base.y * (factor >> 6));
}

__device__ __forceinline__ float repacked_k_value(const BlockQ4_KA& b, int i) {
    const int sub = i / 32;
    const int q = (b.q[sub][i % 16] >> ((i / 16 % 2) * 4)) & 15;
    const float2 scales = q4_ka_scales(b, sub);
    return scales.x * q - scales.y;
}

__device__ __forceinline__ float repacked_k_value(const BlockQ6_A& b, int i) {
    const int sub = i / 32;
    const int half = i / 16 % 2;
    const int lo = (b.ql[sub][i % 16] >> (half * 4)) & 15;
    const int hi = (b.qh[sub / 2][i % 16] >> (sub % 2 * 4 + half * 2)) & 3;
    return half_bits_to_f32(b.d) * b.scale[i / 16] * ((lo | (hi << 4)) - 32);
}
template <int WK>
__device__ __forceinline__ float dequant_weight_elem(const void* w, int n, int k0, int c, int K) {
    if constexpr (WK == 0) {
        return __half2float(__ldg(reinterpret_cast<const half*>(w) + static_cast<int64_t>(n) * K + k0 + c));
    } else if constexpr (WK == 1) {
        return __bfloat162float(__ldg(reinterpret_cast<const __nv_bfloat16*>(w) + static_cast<int64_t>(n) * K + k0 + c));
    } else if constexpr (WK == 2) {
        int nb = K >> 5;
        const BlockQ8_0& blk = reinterpret_cast<const BlockQ8_0*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 5)];
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(__ldg(&blk.qs[c]));
    } else if constexpr (WK == 3) {
        int nb = K >> 5;
        const BlockQ4_0& blk = reinterpret_cast<const BlockQ4_0*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 5)];
        uint8_t byte = __ldg(&blk.qs[c & 15]);
        int q = (c < 16) ? (static_cast<int>(byte & 0x0Fu) - 8) : (static_cast<int>(byte >> 4) - 8);
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(q);
    } else if constexpr (WK == 4) {
        int nb = K >> 5;
        const BlockQ4_1& blk = reinterpret_cast<const BlockQ4_1*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 5)];
        uint8_t byte = __ldg(&blk.qs[c & 15]);
        int q = (c < 16) ? static_cast<int>(byte & 0x0Fu) : static_cast<int>(byte >> 4);
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(q) + half_bits_to_f32(__ldg(&blk.m));
    } else if constexpr (WK == 5 || WK == 7) {
        using W = std::conditional_t<WK == 5, BlockQ4_K, BlockQ5_K>;
        int nb = K >> 8;
        const W& blk = reinterpret_cast<const W*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 8)];
        int sub = k0 & 255;
        int grp = sub >> 6;
        int hi = (sub >> 5) & 1;
        int j = grp * 2 + hi;
        uint8_t sc = (j < 4) ? (blk.scales[j] & 63u) : static_cast<uint8_t>((blk.scales[j + 4] & 0xFu) | ((blk.scales[j - 4] >> 6) << 4));
        uint8_t mn = (j < 4) ? (blk.scales[j + 4] & 63u) : static_cast<uint8_t>((blk.scales[j + 4] >> 4) | ((blk.scales[j] >> 6) << 4));
        uint8_t qb = __ldg(&blk.qs[grp * 32 + c]);
        int q = hi ? (qb >> 4) : (qb & 0xFu);
        if constexpr (WK == 7) q |= ((__ldg(&blk.qh[c]) >> j) & 1) << 4;
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(sc) * static_cast<float>(q)
             - half_bits_to_f32(__ldg(&blk.dmin)) * static_cast<float>(mn);
    } else if constexpr (WK == 8 || WK == 9) {
        using W = std::conditional_t<WK == 8, BlockQ4_KA, BlockQ6_A>;
        const int64_t block = static_cast<int64_t>(n) * (K / 256) + k0 / 256;
        return repacked_k_value(static_cast<const W*>(w)[block], (k0 + c) % 256);
    } else {
        int nb = K >> 8;
        const BlockQ6_K& blk = reinterpret_cast<const BlockQ6_K*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 8)];
        int sub = k0 & 255;
        int half_idx = sub >> 7;
        int q_part = (sub >> 5) & 3;
        const uint8_t* ql = blk.ql + half_idx * 64 + ((q_part & 1) ? 32 : 0);
        const uint8_t* qh = blk.qh + half_idx * 32;
        const int8_t* sc = blk.scales + half_idx * 8;
        uint8_t qlv = __ldg(&ql[c]);
        uint8_t qhv = __ldg(&qh[c]);
        int lo4 = (q_part >= 2) ? (qlv >> 4) : (qlv & 0xFu);
        int hi2 = (qhv >> (q_part * 2)) & 3u;
        int q = (lo4 | (hi2 << 4)) - 32;
        int scv = static_cast<int>(__ldg(&sc[(c >> 4) + q_part * 2]));
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(scv) * static_cast<float>(q);
    }
}

template <int WK, bool HAS_NORM>
__global__ void k_gemm_wmma(const float* __restrict__ x,
                            const float* __restrict__ norm_w,
                            const void* __restrict__ w,
                            float* __restrict__ y,
                            int M, int N, int K, float eps) {
    // Tile 32(M) x 64(N) x 32(K) using 4 warps (128 threads), 2x2 warp grid of 16x32 each
    __shared__ half sA[32][40];
    __shared__ half sB[64][40];
    __shared__ float sC[4][256];
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[32];

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int bm = blockIdx.y * 32;
    int bn = blockIdx.x * 64;

    if constexpr (HAS_NORM) {
        int rows_here = min(32, M - bm);
        for (int lm = 0; lm < rows_here; ++lm) {
            const float* xm = x + static_cast<int64_t>(bm + lm) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += 128) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            for (int k = (k4 << 2) + tid; k < K; k += 128) {
                float v = __ldg(xm + k);
                sum_sq += v * v;
            }
            float tot = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[lm] = rsqrtf(tot / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int warp_m = wid >> 1;
    int warp_n = (wid & 1) * 2;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2];
    nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
    nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int idx = tid; idx < 32 * 32; idx += 128) {
            int lm = idx >> 5;
            int c = idx & 31;
            int gm = bm + lm;
            int gk = k0 + c;
            float v = (gm < M && gk < K) ? __ldg(&x[static_cast<int64_t>(gm) * K + gk]) : 0.0f;
            if constexpr (HAS_NORM) {
                if (gm < M && gk < K) v *= inv_rms_sh[lm] * __ldg(&norm_w[gk]);
            }
            sA[lm][c] = __float2half_rn(v);
        }
        for (int idx = tid; idx < 64 * 32; idx += 128) {
            int ln = idx >> 5;
            int c = idx & 31;
            int gn = bn + ln;
            float wv = (gn < N && (k0 + c) < K) ? dequant_weight_elem<WK>(w, gn, k0, c, K) : 0.0f;
            sB[ln][c] = __float2half_rn(wv);
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_m * 16][ks], 40);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
                nvcuda::wmma::load_matrix_sync(b_frag, &sB[(warp_n + j) * 16][ks], 40);
                nvcuda::wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
            }
        }
        __syncthreads();
    }

    int lane = tid & 31;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        nvcuda::wmma::store_matrix_sync(sC[wid], c_frag[j], 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 256; idx += 32) {
            int lm = warp_m * 16 + (idx >> 4);
            int ln = (warp_n + j) * 16 + (idx & 15);
            int gm = bm + lm;
            int gn = bn + ln;
            if (gm < M && gn < N) {
                y[static_cast<int64_t>(gm) * N + gn] = sC[wid][idx];
            }
        }
        __syncwarp();
    }
}

// =============================================================================
// GQA Flash / Split-K Decode (supports up to 64K seqlen)
// =============================================================================
// Split-K pass 1: grid = (num_q_heads, num_splits), blockDim = 128 (4 warps).
// Each thread computes dot product for one KV step across head_dim? Better:
// 1 CTA per (q_head, split), blockDim.x = head_dim (up to 256, or multiple of 32 >= head_dim).
// Every thread j < head_dim accumulates o_j in registers while looping over tokens in its split!
// Dot product q * k[t] is reduced across threads in shared memory!
template <typename Cache>
__global__ void k_gqa_splitk_part1(
    const float* __restrict__ q,
    const Cache* __restrict__ k_cache,
    const Cache* __restrict__ v_cache,
    float* __restrict__ partial_out,
    float* __restrict__ partial_ml,
    int num_q_heads, int num_kv_heads, int head_dim,
    const int* __restrict__ kv_len_dev, int kv_len_host,
    float scale, int num_splits, CudaKvView cache)
{
    extern __shared__ float smem[];
    float* red_buf = smem;       // [32]
    float* q_sh = red_buf + 32;  // [head_dim]

    int qh = blockIdx.x;
    int split_id = blockIdx.y;
    int tid = threadIdx.x;
    int kv_len = kv_len_dev ? *kv_len_dev : kv_len_host;
    if (qh >= num_q_heads) return;

    int group_size = num_q_heads / num_kv_heads;
    int kvh = qh / group_size;

    if (tid < head_dim) {
        q_sh[tid] = q[qh * head_dim + tid] * scale;
    }
    __syncthreads();

    int t_start = (kv_len * split_id) / num_splits;
    int t_end = (kv_len * (split_id + 1)) / num_splits;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float o_j = 0.0f;

    for (int t = t_start; t < t_end; ++t) {
        const auto* keys = kv_row<false>(k_cache, cache, t, kvh, num_kv_heads, head_dim);
        const auto* values = kv_row<true>(v_cache, cache, t, kvh, num_kv_heads, head_dim);
        const float qk_part = tid < head_dim
            ? q_sh[tid] * kv_value<false>(keys, cache, t, kvh, num_kv_heads, head_dim, tid) : 0.0f;
        float score = block_reduce_sum(qk_part, red_buf);

        float m_new = fmaxf(m_i, score);
        float alpha = (t == t_start) ? 0.0f : expf(m_i - m_new);
        float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        if (tid < head_dim) {
            o_j = o_j * alpha + p * kv_value<true>(values, cache, t, kvh, num_kv_heads, head_dim, tid);
        }
        m_i = m_new;
    }

    if (num_splits == 1) {
        if (tid < head_dim) {
            partial_out[qh * head_dim + tid] = (l_i > 0.0f) ? (o_j / l_i) : 0.0f;
        }
        return;
    }

    if (tid == 0) {
        int64_t ml_off = (static_cast<int64_t>(qh) * num_splits + split_id) * 2;
        partial_ml[ml_off + 0] = m_i;
        partial_ml[ml_off + 1] = l_i;
    }
    if (tid < head_dim) {
        int64_t o_off = (static_cast<int64_t>(qh) * num_splits + split_id) * head_dim + tid;
        partial_out[o_off] = o_j;
    }
}

__device__ __forceinline__ void store_q8_1(float v, BlockQ8_1* out);

template <bool Quantized>
__global__ void k_gqa_splitk_reduce(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_ml,
    float* __restrict__ out,
    BlockQ8_1* __restrict__ q8_out, const float* __restrict__ gate,
    int num_q_heads, int head_dim, int num_splits)
{
    int qh = blockIdx.x;
    int j = threadIdx.x;
    if (qh >= num_q_heads || j >= head_dim) return;

    float m_global = -1e30f;
    for (int s = 0; s < num_splits; ++s) {
        int64_t ml_off = (static_cast<int64_t>(qh) * num_splits + s) * 2;
        float l_s = partial_ml[ml_off + 1];
        if (l_s > 0.0f) {
            m_global = fmaxf(m_global, partial_ml[ml_off + 0]);
        }
    }

    float l_global = 0.0f;
    float o_acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        int64_t ml_off = (static_cast<int64_t>(qh) * num_splits + s) * 2;
        float m_s = partial_ml[ml_off + 0];
        float l_s = partial_ml[ml_off + 1];
        if (l_s > 0.0f) {
            float w = expf(m_s - m_global);
            l_global += l_s * w;
            int64_t o_off = (static_cast<int64_t>(qh) * num_splits + s) * head_dim + j;
            o_acc += partial_out[o_off] * w;
        }
    }
    const int col = qh * head_dim + j;
    float value = (l_global > 0.0f) ? (o_acc / l_global) : 0.0f;
    if constexpr (Quantized) {
        if (gate) value *= 1.0f / (1.0f + expf(-gate[col]));
        store_q8_1(value, q8_out + col / 32);
    } else {
        out[col] = value;
    }
}

// =============================================================================
// Sparse MoE Kernels
// =============================================================================
__global__ void k_moe_topk_router(const float* __restrict__ logits,
                                  int* __restrict__ topk_idx,
                                  float* __restrict__ topk_weight,
                                  int num_experts, int top_k, bool norm_topk) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Softmax over experts
    float max_l = -1e30f;
    for (int e = 0; e < num_experts; ++e) max_l = fmaxf(max_l, logits[e]);
    float sum_e = 0.0f;
    float probs[256];
    int ne = (num_experts < 256) ? num_experts : 256;
    for (int e = 0; e < ne; ++e) {
        probs[e] = expf(logits[e] - max_l);
        sum_e += probs[e];
    }
    for (int e = 0; e < ne; ++e) probs[e] /= sum_e;

    float weight_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int best_e = -1;
        float best_p = -1.0f;
        for (int e = 0; e < ne; ++e) {
            bool used = false;
            for (int prev = 0; prev < k; ++prev) {
                if (topk_idx[prev] == e) used = true;
            }
            if (!used && probs[e] > best_p) {
                best_p = probs[e];
                best_e = e;
            }
        }
        topk_idx[k] = best_e;
        topk_weight[k] = best_p;
        weight_sum += best_p;
    }
    if (norm_topk && weight_sum > 0.0f) {
        for (int k = 0; k < top_k; ++k) topk_weight[k] /= weight_sum;
    }
}

__global__ void k_moe_indexed_gate_up_swiglu(
    const float* __restrict__ x,
    const int* __restrict__ topk_idx,
    const float* __restrict__ gate_up_w,
    float* __restrict__ scratch_inter,
    int hidden, int inter, int top_k)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_rows = top_k * inter;
    if (wid >= total_rows) return;

    int k_slot = wid / inter;
    int c = wid % inter;
    int e = topk_idx[k_slot];
    const float* wg = gate_up_w + (static_cast<int64_t>(e) * 2 * inter + c) * hidden;
    const float* wu = gate_up_w + (static_cast<int64_t>(e) * 2 * inter + inter + c) * hidden;

    float g_acc = 0.0f, u_acc = 0.0f;
    for (int i = lane; i < hidden; i += 32) {
        float xv = x[i];
        g_acc += xv * wg[i];
        u_acc += xv * wu[i];
    }
    g_acc = warp_reduce_sum(g_acc);
    u_acc = warp_reduce_sum(u_acc);
    if (lane == 0) {
        scratch_inter[k_slot * inter + c] = silu_f(g_acc) * u_acc;
    }
}

__global__ void k_moe_indexed_down_combine(
    const float* __restrict__ scratch_inter,
    const int* __restrict__ topk_idx,
    const float* __restrict__ down_w,
    const float* __restrict__ topk_weight,
    float* __restrict__ out,
    int hidden, int inter, int top_k)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= hidden) return;

    float total = 0.0f;
    for (int k_slot = 0; k_slot < top_k; ++k_slot) {
        int e = topk_idx[k_slot];
        const float* wd = down_w + (static_cast<int64_t>(e) * hidden + wid) * inter;
        const float* act = scratch_inter + k_slot * inter;
        float acc = 0.0f;
        for (int i = lane; i < inter; i += 32) {
            acc += act[i] * wd[i];
        }
        total += warp_reduce_sum(acc) * topk_weight[k_slot];
    }
    if (lane == 0) out[wid] = total;
}

__global__ void k_update_pos_and_kvlen(int* pos_dev, int* kv_len_dev, int new_pos, int new_kv_len) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (pos_dev) *pos_dev = new_pos;
        if (kv_len_dev) *kv_len_dev = new_kv_len;
    }
}

// =============================================================================
// Native F16/BF16 weights, embedding lookup, and batched prefill
// =============================================================================

template <typename WType>
__device__ __forceinline__ float w_to_f32(WType v);

template <>
__device__ __forceinline__ float w_to_f32<half>(half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float w_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename WType, int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_linear_gemv_w16_vec(const float* __restrict__ x, const float* __restrict__ norm_w,
                                      const WType* __restrict__ w, float* __restrict__ y,
                                      int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            for (int k = (k4 << 2) + tid; k < K; k += blockDim.x) {
                float v = __ldg(xm + k);
                sum_sq += v * v;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;
    const WType* wr = w + static_cast<int64_t>(n) * K;
    float acc[8] = {0.f};

    if ((K & 7) == 0 && (reinterpret_cast<uintptr_t>(wr) & 15u) == 0 && (reinterpret_cast<uintptr_t>(x) & 15u) == 0) {
        int k8 = K >> 3;
        const uint4* wr8 = reinterpret_cast<const uint4*>(wr);
        for (int i = lane; i < k8; i += 32) {
            uint4 pack = __ldg(wr8 + i);
            const WType* wp = reinterpret_cast<const WType*>(&pack);
            float wv[8];
            int bk = i << 3;
            #pragma unroll
            for (int j = 0; j < 8; ++j) wv[j] = w_to_f32<WType>(wp[j]);
            if constexpr (HAS_NORM) {
                float4 nw0 = __ldg(reinterpret_cast<const float4*>(norm_w + bk));
                float4 nw1 = __ldg(reinterpret_cast<const float4*>(norm_w + bk + 4));
                wv[0] *= nw0.x; wv[1] *= nw0.y; wv[2] *= nw0.z; wv[3] *= nw0.w;
                wv[4] *= nw1.x; wv[5] *= nw1.y; wv[6] *= nw1.z; wv[7] *= nw1.w;
            }
            for (int m = 0; m < M; ++m) {
                const float* xm = x + static_cast<int64_t>(m) * K + bk;
                float4 x0 = __ldg(reinterpret_cast<const float4*>(xm));
                float4 x1 = __ldg(reinterpret_cast<const float4*>(xm + 4));
                acc[m] += x0.x * wv[0] + x0.y * wv[1] + x0.z * wv[2] + x0.w * wv[3]
                        + x1.x * wv[4] + x1.y * wv[5] + x1.z * wv[6] + x1.w * wv[7];
            }
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            float wv = w_to_f32<WType>(__ldg(&wr[k]));
            if constexpr (HAS_NORM) wv *= __ldg(&norm_w[k]);
            for (int m = 0; m < M; ++m) acc[m] += __ldg(&x[static_cast<int64_t>(m) * K + k]) * wv;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

__global__ void k_embed_lookup_f32(const uint8_t* __restrict__ table, int dtype,
                                   const int32_t* __restrict__ token_ids,
                                   float* __restrict__ out, int M, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * hidden;
    if (idx >= total) return;
    int m = idx / hidden;
    int c = idx % hidden;
    int tid = token_ids[m];

    float val = 0.0f;
    if (dtype == static_cast<int>(DType::F32)) {
        val = reinterpret_cast<const float*>(table)[static_cast<int64_t>(tid) * hidden + c];
    } else if (dtype == static_cast<int>(DType::F16)) {
        val = __half2float(reinterpret_cast<const half*>(table)[static_cast<int64_t>(tid) * hidden + c]);
    } else if (dtype == static_cast<int>(DType::BF16)) {
        val = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(table)[static_cast<int64_t>(tid) * hidden + c]);
    } else if (dtype == static_cast<int>(DType::Q8_0)) {
        int nb = hidden >> 5;
        int blk = c >> 5;
        int i = c & 31;
        const BlockQ8_0& b = reinterpret_cast<const BlockQ8_0*>(table)[static_cast<int64_t>(tid) * nb + blk];
        val = __half2float(*reinterpret_cast<const half*>(&b.d)) * static_cast<float>(b.qs[i]);
    } else if (dtype == static_cast<int>(DType::Q4_0)) {
        int nb = hidden >> 5;
        int blk = c >> 5;
        int i = c & 31;
        const BlockQ4_0& b = reinterpret_cast<const BlockQ4_0*>(table)[static_cast<int64_t>(tid) * nb + blk];
        float d = __half2float(*reinterpret_cast<const half*>(&b.d));
        uint8_t byte_v = b.qs[i & 15];
        int q = (i < 16) ? (static_cast<int>(byte_v & 0x0Fu) - 8) : (static_cast<int>(byte_v >> 4) - 8);
        val = d * static_cast<float>(q);
    } else if (dtype == static_cast<int>(DType::Q4_1)) {
        int nb = hidden >> 5;
        int blk = c >> 5;
        int i = c & 31;
        const BlockQ4_1& b = reinterpret_cast<const BlockQ4_1*>(table)[static_cast<int64_t>(tid) * nb + blk];
        float d = __half2float(*reinterpret_cast<const half*>(&b.d));
        float mb = __half2float(*reinterpret_cast<const half*>(&b.m));
        uint8_t byte_v = b.qs[i & 15];
        int q = (i < 16) ? static_cast<int>(byte_v & 0x0Fu) : static_cast<int>(byte_v >> 4);
        val = d * static_cast<float>(q) + mb;
    } else if (dtype == static_cast<int>(DType::Q4_K)) {
        val = dequant_weight_elem<5>(table, tid, c & ~31, c & 31, hidden);
    } else if (dtype == static_cast<int>(DType::Q5_K)) {
        val = dequant_weight_elem<7>(table, tid, c & ~31, c & 31, hidden);
    } else if (dtype == static_cast<int>(DType::Q6_K)) {
        int nb = hidden >> 8;
        int blk = c >> 8;
        int sc_c = c & 255;
        const BlockQ6_K& b = reinterpret_cast<const BlockQ6_K*>(table)[static_cast<int64_t>(tid) * nb + blk];
        int half_idx = sc_c >> 7;
        int q_part = (sc_c >> 5) & 3;
        int l = sc_c & 31;
        const uint8_t* ql = b.ql + half_idx * 64 + ((q_part & 1) ? 32 : 0);
        const uint8_t* qh = b.qh + half_idx * 32;
        const int8_t* sc = b.scales + half_idx * 8;
        uint8_t qlv = ql[l];
        uint8_t qhv = qh[l];
        int lo4 = (q_part >= 2) ? (qlv >> 4) : (qlv & 0xFu);
        int hi2 = (qhv >> (q_part * 2)) & 3u;
        int q = (lo4 | (hi2 << 4)) - 32;
        int scv = static_cast<int>(sc[(l >> 4) + q_part * 2]);
        val = __half2float(*reinterpret_cast<const half*>(&b.d)) * static_cast<float>(scv) * static_cast<float>(q);
    }
    out[idx] = val;
}

template<class Block>
__global__ void k_embed_lookup_repacked_f32(const Block* __restrict__ table,
                                            const int32_t* __restrict__ token_ids,
                                            float* __restrict__ out, int blocks_per_row) {
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    const int lane = threadIdx.x;
    const auto& b = table[static_cast<int64_t>(token_ids[row]) * blocks_per_row + col];
    float val;
    if constexpr (std::is_same_v<Block, BlockQ4_A>) {
        const int i = lane & 31;
        const uint8_t packed = reinterpret_cast<const uint8_t*>(b.qs)[(lane >> 5) * 16 + (i & 15)];
        val = half_bits_to_f32(b.d[lane >> 5]) * ((i < 16 ? packed & 15 : packed >> 4) - 8);
    } else if constexpr (std::is_same_v<Block, BlockQ8_A>) {
        val = half_bits_to_f32(b.d[lane >> 5]) * reinterpret_cast<const int8_t*>(b.qs)[lane];
    } else {
        val = repacked_k_value(b, lane);
    }
    out[(static_cast<int64_t>(row) * blocks_per_row + col) * 256 + lane] = val;
}

template <typename Cache>
__global__ void k_gqa_prefill_causal(
    const float* __restrict__ q, int q_stride,
    const Cache* __restrict__ k_cache,
    const Cache* __restrict__ v_cache,
    float* __restrict__ out,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int start_pos, float scale, CudaKvView cache)
{
    extern __shared__ float smem[];
    float* red_buf = smem;
    float* q_sh = red_buf + 32;

    int qh = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;
    if (qh >= num_q_heads || m >= M) return;

    int group_size = num_q_heads / num_kv_heads;
    int kvh = qh / group_size;
    int q_off = m * q_stride + qh * head_dim;
    int out_off = (m * num_q_heads + qh) * head_dim;

    if (tid < head_dim) {
        q_sh[tid] = q[q_off + tid] * scale;
    }
    __syncthreads();

    int kv_end = start_pos + m + 1;
    float m_i = -1e30f;
    float l_i = 0.0f;
    float o_j = 0.0f;

    for (int t = 0; t < kv_end; ++t) {
        const auto* keys = kv_row<false>(k_cache, cache, t, kvh, num_kv_heads, head_dim);
        const auto* values = kv_row<true>(v_cache, cache, t, kvh, num_kv_heads, head_dim);
        const float qk_part = tid < head_dim
            ? q_sh[tid] * kv_value<false>(keys, cache, t, kvh, num_kv_heads, head_dim, tid) : 0.0f;
        float score = block_reduce_sum(qk_part, red_buf);

        float m_new = fmaxf(m_i, score);
        float alpha = (t == 0) ? 0.0f : expf(m_i - m_new);
        float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        if (tid < head_dim) {
            o_j = o_j * alpha + p * kv_value<true>(values, cache, t, kvh, num_kv_heads, head_dim, tid);
        }
        m_i = m_new;
    }

    if (tid < head_dim) {
        out[out_off + tid] = (l_i > 0.0f) ? (o_j / l_i) : 0.0f;
    }
}

template <int HD, typename Cache, bool Decode = false, int Warps = 8, bool FirstPage = false>
__global__ void k_gqa_prefill_tiled(
    const float* __restrict__ q, int q_stride,
    const Cache* __restrict__ k_cache, const Cache* __restrict__ v_cache,
    float* __restrict__ out, int M, int num_q_heads, int num_kv_heads,
    int start_pos, float scale, const int* kv_len_dev = nullptr, int splits = 1,
    float* partial_ml = nullptr, CudaKvView cache = {})
{
    constexpr int BKV = std::is_same_v<Cache, float> ? 8 : 16, DP = HD / 32;
    constexpr int TokensPerWarp = Decode ? 4 : 1;
    constexpr int ScoreLanes = 32 / TokensPerWarp;
    // Separate the four token rows' shared-memory banks.
    alignas(16) __shared__ float keys[BKV][HD + (Decode ? ScoreLanes : 0)];
    alignas(16) __shared__ float values[BKV][HD];

    int kvh = blockIdx.x;
    int group = num_q_heads / num_kv_heads;
    int first = Decode ? 0 : blockIdx.y * Warps;
    int query = first + threadIdx.x / 32;
    int m = Decode ? 0 : query / group;
    int qh = kvh * group + query % group;
    int lane = threadIdx.x & 31;
    const int kv_len = Decode ? (kv_len_dev ? *kv_len_dev : start_pos) : 0;
    if constexpr (FirstPage || (Decode && !std::is_same_v<Cache, int8_t>)) {
        if (FirstPage || (cache.pages && kv_len <= kKvPageTokens)) {
            const auto* page = cache.pages[0];
            k_cache = reinterpret_cast<const Cache*>(page);
            v_cache = reinterpret_cast<const Cache*>(page + cache.plane_bytes);
            cache.pages = nullptr;
        }
    }
    const int tile_start = Decode ? (kv_len * blockIdx.y) / splits : 0;
    const int kv_end = Decode ? (kv_len * (blockIdx.y + 1)) / splits : start_pos + m + 1;
    const int tile_end = Decode ? kv_end : start_pos + min(M, (first + Warps - 1) / group + 1);
    const bool valid = Decode ? query < group : m < M;
    const float* qr = q + static_cast<int64_t>(min(m, M - 1)) * q_stride + qh * HD;

    float q_reg[DP][TokensPerWarp], o_reg[DP] = {};
    #pragma unroll
    for (int d = 0; d < DP; ++d) {
        #pragma unroll
        for (int j = 0; j < TokensPerWarp; ++j) {
            q_reg[d][j] = qr[d * 32 + j * ScoreLanes + lane % ScoreLanes] * scale;
        }
    }

    float max_score = -INFINITY, sum = 0.0f;
    for (int base = tile_start; base < tile_end; base += BKV) {
        constexpr int V = (Decode || FirstPage || std::is_same_v<Cache, int8_t>) ? 16 / sizeof(Cache) : 1;
        for (int i = threadIdx.x; i < BKV * HD / V; i += Warps * 32) {
            const int t = (V * i) / HD, d = (V * i) % HD;
            const int pos = min(base + t, tile_end - 1);
            const auto* kr = kv_row<false>(k_cache, cache, pos, kvh, num_kv_heads, HD);
            const auto* vr = kv_row<true>(v_cache, cache, pos, kvh, num_kv_heads, HD);
            if constexpr (std::is_same_v<Cache, int8_t>) {
                const uint4 k = *reinterpret_cast<const uint4*>(kr + d);
                const uint4 v = *reinterpret_cast<const uint4*>(vr + d);
                const float dk = __half2float(kv_scales<false>(cache, pos, kvh, num_kv_heads, HD)[d / 32]);
                const float dv = __half2float(kv_scales<true>(cache, pos, kvh, num_kv_heads, HD)[d / 32]);
                #pragma unroll
                for (int j = 0; j < V; ++j) {
                    keys[t][d+j] = float(reinterpret_cast<const int8_t*>(&k)[j]) * dk;
                    values[t][d+j] = float(reinterpret_cast<const int8_t*>(&v)[j]) * dv;
                }
            } else if constexpr ((Decode || FirstPage) && std::is_same_v<Cache, half>) {
                const uint4 k = *reinterpret_cast<const uint4*>(kr + d);
                const uint4 v = *reinterpret_cast<const uint4*>(vr + d);
                #pragma unroll
                for (int j = 0; j < V / 2; ++j) {
                    reinterpret_cast<float2*>(keys[t] + d)[j] = __half22float2(reinterpret_cast<const half2*>(&k)[j]);
                    reinterpret_cast<float2*>(values[t] + d)[j] = __half22float2(reinterpret_cast<const half2*>(&v)[j]);
                }
            } else if constexpr (FirstPage && std::is_same_v<Cache, float>) {
                *reinterpret_cast<float4*>(keys[t] + d) = *reinterpret_cast<const float4*>(kr + d);
                *reinterpret_cast<float4*>(values[t] + d) = *reinterpret_cast<const float4*>(vr + d);
            } else {
                keys[t][d] = static_cast<float>(kr[d]);
                values[t][d] = static_cast<float>(vr[d]);
            }
        }
        __syncthreads();
        if (valid && base < kv_end) {
            float scores[BKV];
            float next_max = max_score;
            #pragma unroll
            for (int t = 0; t < BKV; t += TokensPerWarp) {
                const int key_t = t + lane / ScoreLanes;
                float parts[TokensPerWarp] = {};
                #pragma unroll
                for (int d = 0; d < DP; ++d) {
                    #pragma unroll
                    for (int j = 0; j < TokensPerWarp; ++j) {
                        const int col = d * 32 + j * ScoreLanes + lane % ScoreLanes;
                        parts[j] = fmaf(q_reg[d][j], keys[key_t][col], parts[j]);
                    }
                }
                float dot = parts[0];
                if constexpr (Decode) dot = (dot + parts[2]) + (parts[1] + parts[3]);
                for (int off = ScoreLanes / 2; off > 0; off >>= 1) {
                    dot += __shfl_xor_sync(0xffffffffu, dot, off);
                }
                #pragma unroll
                for (int j = 0; j < TokensPerWarp; ++j) {
                    const float score = Decode ? __shfl_sync(0xffffffffu, dot, j * ScoreLanes) : dot;
                    scores[t + j] = base + t + j < kv_end ? score : -INFINITY;
                    next_max = fmaxf(next_max, scores[t + j]);
                }
            }
            float rescale = __expf(max_score - next_max);
            max_score = next_max;
            sum *= rescale;
            #pragma unroll
            for (int d = 0; d < DP; ++d) o_reg[d] *= rescale;
            #pragma unroll
            for (int t = 0; t < BKV; ++t) {
                float p = __expf(scores[t] - max_score);
                sum += p;
                #pragma unroll
                for (int d = 0; d < DP; ++d) o_reg[d] = fmaf(p, values[t][d * 32 + lane], o_reg[d]);
            }
        }
        __syncthreads();
    }
    if (valid) {
        const int64_t row = Decode ? static_cast<int64_t>(qh) * splits + blockIdx.y
                                   : static_cast<int64_t>(m) * num_q_heads + qh;
        float* dst = out + row * HD;
        float inv = Decode ? 1.0f : 1.0f / sum;
        if constexpr (Decode) {
            if (lane == 0) {
                partial_ml[2 * row] = max_score;
                partial_ml[2 * row + 1] = sum;
            }
        }
        #pragma unroll
        for (int d = 0; d < DP; ++d) dst[d * 32 + lane] = o_reg[d] * inv;
    }
}

template <bool Transpose>
__device__ __forceinline__ uint2 attention_load_b(const half* ptr) {
    uint2 b;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    if constexpr (Transpose) {
        asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(b.x), "=r"(b.y) : "r"(addr));
    } else {
        asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(b.x), "=r"(b.y) : "r"(addr));
    }
    return b;
}

__device__ __forceinline__ void attention_mma(float4& c, uint4 a, uint2 b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

template <int HD>
__device__ __forceinline__ int attention_offset(int row, int col) {
    return row * HD + (col ^ ((row & 7) * 8));
}

template <bool Value, typename Cache, int HD>
__device__ __forceinline__ void attention_stage_kv(half* tile, int r, CudaKvView cache, int token,
                                                    int head, int heads, int d) {
    half* dst = tile + attention_offset<HD>(r, d);
    const auto* row = kv_row<Value>(static_cast<const Cache*>(nullptr), cache, token, head, heads, HD);
    if constexpr (std::is_same_v<Cache, half>) {
        const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(shared), "l"(row + d) : "memory");
    } else if constexpr (std::is_same_v<Cache, int8_t>) {
        const auto packed = *reinterpret_cast<const uint4*>(row + d);
        const auto scale = kv_scales<Value>(cache, token, head, heads, HD)[d / 32];
        const auto scale2 = __halves2half2(scale, scale);
        const auto bias = __float2half2_rn(1152.0f);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned bytes = reinterpret_cast<const unsigned*>(&packed)[i] ^ 0x80808080u;
            // 0x6400 is 1024 with unit spacing. Subtracting 1024 + 128 restores the signed byte.
            const unsigned lo = __byte_perm(bytes, 0x64646464u, 0x5150);
            const unsigned hi = __byte_perm(bytes, 0x64646464u, 0x5352);
            auto* pair = reinterpret_cast<half2*>(tile + attention_offset<HD>(r, d + (i / 2) * 8)) + (i % 2) * 2;
            pair[0] = __hmul2(__hsub2(reinterpret_cast<const half2&>(lo), bias), scale2);
            pair[1] = __hmul2(__hsub2(reinterpret_cast<const half2&>(hi), bias), scale2);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            reinterpret_cast<half2*>(dst)[i / 2] = __floats2half2_rn(row[d + i], row[d + i + 1]);
        }
    }
}

__device__ __forceinline__ uint4 attention_load_q(const half* ptr) {
    uint4 a;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "r"(addr));
    return a;
}

__device__ __forceinline__ float attention_exp2(float x) {
    float value;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(x));
    return value;
}

__device__ __forceinline__ unsigned attention_prob(float a, float b) {
    const auto value = __floats2half2_rn(a, b);
    return reinterpret_cast<const unsigned&>(value);
}

// Keep scores, softmax state and output fragments on chip; K and V are shared
// by all query rows and heads in this KV group.
template <int HD, int Warps, typename Cache>
__global__ void k_gqa_prefill_online(const float* q, int q_stride, CudaKvView cache, float* out,
                                     int rows, int nq, int nkv, int start, float scale) {
    constexpr int Q = 16 * Warps, K = HD == 256 ? 32 : 64;
    constexpr bool KeepQ = HD <= 128;
    constexpr unsigned mask = 0xffffffffu;
    extern __shared__ __align__(16) half sh[];
    auto* sq = sh;
    auto* sk = sq + (KeepQ ? 0 : Q * HD);
    auto* sv = sk + K * HD;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int quad = lane / 4, col = 2 * (lane % 4);
    const int group = nq / nkv, head = blockIdx.x, qr = blockIdx.y * Q;
    const int end = rows * group;
    for (int i = tid; i < Q * HD / 8; i += blockDim.x) {
        const int r = i / (HD / 8), d = (i % (HD / 8)) * 8;
        const int j = min(qr + r, end - 1);
        const float* src = q + (j / group) * q_stride + (head * group + j % group) * HD + d;
        #pragma unroll
        for (int e = 0; e < 8; e += 2) {
            reinterpret_cast<half2*>(sq + attention_offset<HD>(r, d))[e / 2] = __floats2half2_rn(src[e], src[e + 1]);
        }
    }
    __syncthreads();
    const int j0 = qr + warp * 16 + quad, j1 = j0 + 8;
    const int last0 = start + min(j0, end - 1) / group, last1 = start + min(j1, end - 1) / group;
    const int last = start + min(qr + Q - 1, end - 1) / group;
    const int arow = warp * 16 + lane % 16, acol = 8 * (lane / 16);
    uint4 query[KeepQ ? HD / 16 : 1];
    if constexpr (KeepQ) {
        #pragma unroll
        for (int d = 0; d < HD / 16; ++d) {
            query[d] = attention_load_q(sq + attention_offset<HD>(arow, 16 * d + acol));
        }
        __syncthreads();
    }
    float max0 = -INFINITY, max1 = -INFINITY, sum0 = 0.0f, sum1 = 0.0f;
    float4 acc[HD / 8]{};
    for (int base = 0; base <= last; base += K) {
        constexpr int V = std::is_same_v<Cache, int8_t> ? 16 : 8;
        for (int i = tid; i < K * HD / V; i += blockDim.x) {
            const int r = i / (HD / V), d = (i % (HD / V)) * V;
            const int t = min(base + r, last);
            attention_stage_kv<false, Cache, HD>(sk, r, cache, t, head, nkv, d);
            attention_stage_kv<true, Cache, HD>(sv, r, cache, t, head, nkv, d);
        }
        if constexpr (std::is_same_v<Cache, half>) {
            asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        float4 scores[K / 8]{};
        #pragma unroll (HD == 256 ? 1 : HD / 16)
        for (int d = 0; d < HD / 16; ++d) {
            const auto a = KeepQ ? query[d] : attention_load_q(sq + attention_offset<HD>(arow, 16 * d + acol));
            #pragma unroll
            for (int t = 0; t < K / 8; ++t) {
                auto b = attention_load_b<false>(sk + attention_offset<HD>(t * 8 + lane % 8,
                                                                           16 * d + 8 * ((lane / 8) % 2)));
                attention_mma(scores[t], a, b);
            }
        }
        float next0 = max0, next1 = max1;
        #pragma unroll
        for (int i = 0; i < K / 8; ++i) {
            const int t = base + 8 * i + col;
            auto& s = scores[i];
            s = {t <= last0 ? s.x * scale : -INFINITY, t + 1 <= last0 ? s.y * scale : -INFINITY,
                 t <= last1 ? s.z * scale : -INFINITY, t + 1 <= last1 ? s.w * scale : -INFINITY};
            next0 = fmaxf(next0, fmaxf(s.x, s.y));
            next1 = fmaxf(next1, fmaxf(s.z, s.w));
        }
        #pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            next0 = fmaxf(next0, __shfl_xor_sync(mask, next0, x));
            next1 = fmaxf(next1, __shfl_xor_sync(mask, next1, x));
        }
        const float alpha0 = attention_exp2(max0 - next0), alpha1 = attention_exp2(max1 - next1);
        max0 = next0;
        max1 = next1;
        float tile0 = 0.0f, tile1 = 0.0f;
        uint4 p[K / 16];
        #pragma unroll
        for (int i = 0; i < K / 16; ++i) {
            const auto a = scores[2 * i], b = scores[2 * i + 1];
            const float ax = attention_exp2(a.x - max0), ay = attention_exp2(a.y - max0);
            const float az = attention_exp2(a.z - max1), aw = attention_exp2(a.w - max1);
            const float bx = attention_exp2(b.x - max0), by = attention_exp2(b.y - max0);
            const float bz = attention_exp2(b.z - max1), bw = attention_exp2(b.w - max1);
            tile0 += ax + ay + bx + by;
            tile1 += az + aw + bz + bw;
            p[i] = {attention_prob(ax, ay), attention_prob(az, aw),
                    attention_prob(bx, by), attention_prob(bz, bw)};
        }
        #pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            tile0 += __shfl_xor_sync(mask, tile0, x);
            tile1 += __shfl_xor_sync(mask, tile1, x);
        }
        sum0 = fmaf(sum0, alpha0, tile0);
        sum1 = fmaf(sum1, alpha1, tile1);
        if (__any_sync(mask, alpha0 != 1.0f || alpha1 != 1.0f)) {
            #pragma unroll
            for (auto& a : acc) a = {a.x * alpha0, a.y * alpha0, a.z * alpha1, a.w * alpha1};
        }
        #pragma unroll
        for (int d = 0; d < HD / 8; ++d) {
            auto& a = acc[d];
            #pragma unroll
            for (int i = 0; i < K / 16; ++i) {
                const auto b = attention_load_b<true>(sv + attention_offset<HD>(16 * i + lane % 16, 8 * d));
                attention_mma(a, p[i], b);
            }
        }
        __syncthreads();
    }
    const float inv0 = 1.0f / sum0, inv1 = 1.0f / sum1;
    #pragma unroll
    for (int d = 0; d < HD / 8; ++d) {
        if (j0 < end) {
            auto* dst = out + ((j0 / group) * nq + head * group + j0 % group) * HD + d * 8 + col;
            *reinterpret_cast<float2*>(dst) = {acc[d].x * inv0, acc[d].y * inv0};
        }
        if (j1 < end) {
            auto* dst = out + ((j1 / group) * nq + head * group + j1 % group) * HD + d * 8 + col;
            *reinterpret_cast<float2*>(dst) = {acc[d].z * inv1, acc[d].w * inv1};
        }
    }
}

// =============================================================================
// High-Speed Decode (__dp4a Q4_A / Q8_A, SMEM-Cached Q6_K), cuBLAS Prefill Helpers,
// and Single-Kernel Batched GatedDeltaNet Prefill
// =============================================================================

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

__device__ __forceinline__ void store_q8_1x4(float4 v, BlockQ8_1* out) {
    const int lane = threadIdx.x & 31;
    const unsigned group = 0xffu << (lane & ~7);
    const int local = max(max(__float_as_int(fabsf(v.x)), __float_as_int(fabsf(v.y))),
                          max(__float_as_int(fabsf(v.z)), __float_as_int(fabsf(v.w))));
    const float amax = __int_as_float(__reduce_max_sync(group, local));
    const float mul = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
    const int x = __float2int_rn(v.x * mul), y = __float2int_rn(v.y * mul);
    const int z = __float2int_rn(v.z * mul), w = __float2int_rn(v.w * mul);
    out->qs[lane & 7] = (x & 255u) | ((y & 255u) << 8) | ((z & 255u) << 16) | ((w & 255u) << 24);
    const int sum = __reduce_add_sync(0xfu << (lane & ~3), x + y + z + w);
    const int hi = __shfl_sync(group, sum, (lane & ~7) + 4);
    if ((lane & 7) == 0) {
        const float d = amax * (1.0f / 127.0f);
        *reinterpret_cast<float4*>(&out->d) = make_float4(d, d * sum, d * hi, d * (sum + hi));
    }
}

template <int K, bool Add>
__global__ void k_quantize_x_q8_1_decode(const float4* x, const float4* res_b, const float4* norm,
                                        float4* res_out, BlockQ8_1* out, float eps) {
    constexpr int threads = 256;
    constexpr int vectors = K / (4 * threads);
    __shared__ float reduce[32];
    float4 residual[vectors];
    float squares = 0.0f;
    #pragma unroll
    for (int i = 0; i < vectors; ++i) {
        const int col = threadIdx.x + i * threads;
        float4 v = x[col];
        if constexpr (Add) {
            const float4 r = res_b[col];
            v.x += r.x;
            v.y += r.y;
            v.z += r.z;
            v.w += r.w;
        }
        squares += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        residual[i] = v;
    }
    const float inv_rms = rsqrtf(block_reduce_sum(squares, reduce) / K + eps);
    #pragma unroll
    for (int i = 0; i < vectors; ++i) {
        const int col = threadIdx.x + i * threads;
        float4 v = residual[i];
        if constexpr (!Add) {
            v.x += 0.0f;
            v.y += 0.0f;
            v.z += 0.0f;
            v.w += 0.0f;
        }
        if (res_out) res_out[col] = v;
        const float4 w = norm[col];
        store_q8_1x4(make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y,
                                 v.z * inv_rms * w.z, v.w * inv_rms * w.w), out + col / 8);
    }
}

template <int StaticK = 0>
__global__ void k_quantize_x_q8_1(const float* x, const float* res_b, const float* norm_w,
                                  float* res_out, BlockQ8_1* out, int K, float eps) {
    __shared__ float reduce[32];
    if constexpr (StaticK != 0) K = StaticK;
    int row = blockIdx.x;
    x += static_cast<int64_t>(row) * K;
    if (res_b) res_b += static_cast<int64_t>(row) * K;
    if (res_out) res_out += static_cast<int64_t>(row) * K;
    out += static_cast<int64_t>(row) * (K / 32);
    if constexpr (StaticK != 0) {
        constexpr int threads = 256;
        constexpr int count = StaticK / threads;
        float residual[count];
        float squares = 0.0f;
        #pragma unroll
        for (int k = 0; k < count; ++k) {
            int col = threadIdx.x + k * threads;
            float v = x[col] + (res_b ? res_b[col] : 0.0f);
            if (res_out) res_out[col] = v;
            residual[k] = v;
            squares += v * v;
        }
        float inv_rms = norm_w ? rsqrtf(block_reduce_sum(squares, reduce) / StaticK + eps) : 1.0f;
        #pragma unroll
        for (int k = 0; k < count; ++k) {
            int col = threadIdx.x + k * threads;
            float v = residual[k];
            if (norm_w) v = v * inv_rms * norm_w[col];
            store_q8_1(v, out + col / 32);
        }
        return;
    }
    float inv_rms = 1.0f;
    if (norm_w) {
        float squares = 0.0f;
        uintptr_t alignment = reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(res_b);
        if ((alignment & 15) == 0) {
            auto* x4 = reinterpret_cast<const float4*>(x);
            auto* res4 = reinterpret_cast<const float4*>(res_b);
            for (int i = threadIdx.x; i < K / 4; i += blockDim.x) {
                float4 v = x4[i];
                if (res4) {
                    float4 r = res4[i];
                    v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
                }
                squares += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                float v = x[i] + (res_b ? res_b[i] : 0.0f);
                squares += v * v;
            }
        }
        inv_rms = rsqrtf(block_reduce_sum(squares, reduce) / K + eps);
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = x[i] + (res_b ? res_b[i] : 0.0f);
        if (res_out) res_out[i] = v;
        if (norm_w) v = v * inv_rms * norm_w[i];
        store_q8_1(v, out + i / 32);
    }
}

__global__ void k_swiglu_interleaved_q8_1(const float* gate_up, BlockQ8_1* out, int inter_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= inter_dim) return;
    int row = blockIdx.y;
    const float* gate = gate_up + static_cast<int64_t>(row) * 2 * inter_dim;
    store_q8_1(silu_f(gate[col]) * gate[inter_dim + col],
               out + (static_cast<int64_t>(row) * inter_dim + col) / 32);
}

__global__ void k_quantize_vector_q8_1(const float* x, BlockQ8_1* out, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < K) store_q8_1(x[col], out + col / 32);
}

constexpr unsigned kNibbleMask = 0x0f0f0f0fu;

__device__ __forceinline__ int dot_q8_words(uint4 w, uint4 x) {
    int dot = __dp4a(static_cast<int>(w.x), static_cast<int>(x.x), 0);
    dot = __dp4a(static_cast<int>(w.y), static_cast<int>(x.y), dot);
    dot = __dp4a(static_cast<int>(w.z), static_cast<int>(x.z), dot);
    return __dp4a(static_cast<int>(w.w), static_cast<int>(x.w), dot);
}

__device__ __forceinline__ uint4 unpack_q4_words(uint4 v, int shift) {
    return make_uint4((v.x >> shift) & kNibbleMask, (v.y >> shift) & kNibbleMask,
                      (v.z >> shift) & kNibbleMask, (v.w >> shift) & kNibbleMask);
}

__device__ __forceinline__ uint4 load_q6_words(const uint8_t* p) {
    // The 210-byte block stride can leave words two bytes off alignment.
    auto address = reinterpret_cast<uintptr_t>(p);
    auto* aligned = reinterpret_cast<const unsigned*>(address & ~uintptr_t{3});
    unsigned shift = static_cast<unsigned>(address & 2) * 8;
    unsigned a = __ldg(aligned);
    unsigned b = __ldg(aligned + 1);
    unsigned c = __ldg(aligned + 2);
    unsigned d = __ldg(aligned + 3);
    unsigned e = __ldg(aligned + 4);
    return make_uint4(__funnelshift_r(a, b, shift), __funnelshift_r(b, c, shift),
                      __funnelshift_r(c, d, shift), __funnelshift_r(d, e, shift));
}

__device__ __forceinline__ uint4 unpack_q6_words(const uint8_t* ql, const uint8_t* qh, int part) {
    uint4 lo = unpack_q4_words(load_q6_words(ql), (part >> 1) * 4);
    uint4 hi = load_q6_words(qh);
    int shift = part * 2;
    constexpr unsigned mask = 0x03030303u;
    return make_uint4(lo.x | (((hi.x >> shift) & mask) << 4),
                      lo.y | (((hi.y >> shift) & mask) << 4),
                      lo.z | (((hi.z >> shift) & mask) << 4),
                      lo.w | (((hi.w >> shift) & mask) << 4));
}

__device__ __forceinline__ unsigned unpack_q6_a(unsigned low, unsigned high, int half, int parity) {
    const unsigned lo = (low >> (half * 4)) & kNibbleMask;
    const unsigned hi = (high >> (parity * 4 + half * 2)) & 0x03030303u;
    return lo | (hi << 4);
}

__device__ __forceinline__ uint4 unpack_q6_a(uint4 low, uint4 high, int half, int parity) {
    return make_uint4(unpack_q6_a(low.x, high.x, half, parity), unpack_q6_a(low.y, high.y, half, parity),
                      unpack_q6_a(low.z, high.z, half, parity), unpack_q6_a(low.w, high.w, half, parity));
}

template <typename W>
__device__ __forceinline__ float dot_q8_1(const W& w, int sub, uint4 lo, uint4 hi, float4 x) {
    if constexpr (std::is_same_v<W, BlockQ4_A>) {
        uint4 qs = __ldg(reinterpret_cast<const uint4*>(&w.qs[4 * sub]));
        int dot = dot_q8_words(unpack_q4_words(qs, 0), lo) + dot_q8_words(unpack_q4_words(qs, 4), hi);
        return half_bits_to_f32(__ldg(&w.d[sub])) * (x.x * dot - 8.0f * x.w);
    } else if constexpr (std::is_same_v<W, BlockQ8_A>) {
        const auto* qs = reinterpret_cast<const uint4*>(&w.qs[8 * sub]);
        int dot = dot_q8_words(__ldg(qs), lo) + dot_q8_words(__ldg(qs + 1), hi);
        return half_bits_to_f32(__ldg(&w.d[sub])) * x.x * dot;
    } else if constexpr (std::is_same_v<W, BlockQ4_KA>) {
        const uint4 q = __ldg(reinterpret_cast<const uint4*>(w.q[sub]));
        int dot = dot_q8_words(unpack_q4_words(q, 0), lo) + dot_q8_words(unpack_q4_words(q, 4), hi);
        const float2 scales = q4_ka_scales(w, sub);
        return scales.x * x.x * dot - scales.y * x.w;
    } else if constexpr (std::is_same_v<W, BlockQ6_A>) {
        const uint4 low = __ldg(reinterpret_cast<const uint4*>(w.ql[sub]));
        const uint4 high = __ldg(reinterpret_cast<const uint4*>(w.qh[sub / 2]));
        int dot_lo = dot_q8_words(unpack_q6_a(low, high, 0, sub % 2), lo);
        int dot_hi = dot_q8_words(unpack_q6_a(low, high, 1, sub % 2), hi);
        return half_bits_to_f32(w.d) *
               (w.scale[2 * sub] * (x.x * dot_lo - 32.0f * x.y) +
                w.scale[2 * sub + 1] * (x.x * dot_hi - 32.0f * x.z));
    } else if constexpr (std::is_same_v<W, BlockQ6_K>) {
        int part = sub & 3;
        const uint8_t* ql = w.ql + (sub >> 2) * 64 + (part & 1) * 32;
        const uint8_t* qh = w.qh + (sub >> 2) * 32;
        int dot_lo = dot_q8_words(unpack_q6_words(ql, qh, part), lo);
        int dot_hi = dot_q8_words(unpack_q6_words(ql + 16, qh + 16, part), hi);
        return half_bits_to_f32(__ldg(&w.d)) *
               (__ldg(&w.scales[2 * sub]) * (x.x * dot_lo - 32.0f * x.y) +
                __ldg(&w.scales[2 * sub + 1]) * (x.x * dot_hi - 32.0f * x.z));
    } else {
        const auto* qs = reinterpret_cast<const uint4*>(w.qs + (sub >> 1) * 32);
        uint4 q_lo = unpack_q4_words(__ldg(qs), (sub & 1) * 4);
        uint4 q_hi = unpack_q4_words(__ldg(qs + 1), (sub & 1) * 4);
        if constexpr (std::is_same_v<W, BlockQ5_K>) {
            uint4 h_lo = __ldg(reinterpret_cast<const uint4*>(w.qh));
            uint4 h_hi = __ldg(reinterpret_cast<const uint4*>(w.qh) + 1);
            constexpr unsigned mask = 0x01010101u;
            q_lo.x |= ((h_lo.x >> sub) & mask) << 4;
            q_lo.y |= ((h_lo.y >> sub) & mask) << 4;
            q_lo.z |= ((h_lo.z >> sub) & mask) << 4;
            q_lo.w |= ((h_lo.w >> sub) & mask) << 4;
            q_hi.x |= ((h_hi.x >> sub) & mask) << 4;
            q_hi.y |= ((h_hi.y >> sub) & mask) << 4;
            q_hi.z |= ((h_hi.z >> sub) & mask) << 4;
            q_hi.w |= ((h_hi.w >> sub) & mask) << 4;
        }
        int dot = dot_q8_words(q_lo, lo) + dot_q8_words(q_hi, hi);
        int sc, mn;
        if (sub < 4) {
            sc = w.scales[sub] & 63;
            mn = w.scales[sub + 4] & 63;
        } else {
            sc = (w.scales[sub + 4] & 15) | ((w.scales[sub - 4] >> 6) << 4);
            mn = (w.scales[sub + 4] >> 4) | ((w.scales[sub] >> 6) << 4);
        }
        return half_bits_to_f32(w.d) * sc * x.x * dot - half_bits_to_f32(w.dmin) * mn * x.w;
    }
}

struct MmqWeight {
    unsigned lo, hi;
    float scale, other;
};

template <typename W>
constexpr bool mmq_packed_q4 =
    std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ4_K> || std::is_same_v<W, BlockQ4_KA>;

template <typename W>
constexpr bool mmq_async = mmq_packed_q4<W> || std::is_same_v<W, BlockQ6_A> || std::is_same_v<W, BlockQ5_K>;

template <typename W>
constexpr bool mmq_has_min =
    std::is_same_v<W, BlockQ4_K> || std::is_same_v<W, BlockQ4_KA> || std::is_same_v<W, BlockQ5_K>;

template <typename W>
__device__ __forceinline__ float2 mmq_k_scales(const W& w, int sub) {
    const unsigned scale_min = unpack_k_scale_min(w.scales, sub);
    return make_float2(half_bits_to_f32(w.d) * (scale_min & 255),
                      -half_bits_to_f32(w.dmin) * (scale_min >> 8));
}

__device__ __forceinline__ unsigned load_word_unaligned(const uint8_t* p) {
    auto address = reinterpret_cast<uintptr_t>(p);
    auto* aligned = reinterpret_cast<const unsigned*>(address & ~uintptr_t{3});
    return __funnelshift_r(__ldg(aligned), __ldg(aligned + 1), (address & 2) * 8);
}

__device__ __forceinline__ unsigned mmq_signed_q4(unsigned q) {
    return ((q & kNibbleMask) + 0x78787878u) ^ 0x80808080u;
}

__device__ __forceinline__ MmqWeight load_raw_q6_mmq_weight(const BlockQ6_K& w, int sub, int word) {
    MmqWeight v;
    int part = sub & 3;
    auto* ql = w.ql + (sub >> 2) * 64 + (part & 1) * 32 + 4 * word;
    auto* qh = w.qh + (sub >> 2) * 32 + 4 * word;
    unsigned lo = load_word_unaligned(ql);
    unsigned hi = load_word_unaligned(ql + 16);
    unsigned h0 = load_word_unaligned(qh);
    unsigned h1 = load_word_unaligned(qh + 16);
    v.lo = ((((lo >> ((part >> 1) * 4)) & kNibbleMask) |
             (((h0 >> (part * 2)) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
    v.hi = ((((hi >> ((part >> 1) * 4)) & kNibbleMask) |
             (((h1 >> (part * 2)) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
    float d = half_bits_to_f32(__ldg(&w.d));
    v.scale = d * __ldg(w.scales + 2 * sub);
    v.other = d * __ldg(w.scales + 2 * sub + 1);
    return v;
}

__device__ __forceinline__ int4 mma_s8(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
    int4 c;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
    return c;
}

__device__ __forceinline__ int4 mma_s8_k16(unsigned a0, unsigned a1, unsigned b0) {
    int4 c;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
    return c;
}

template <int Rows, int BK, int Words, bool Other>
struct MmqTile {
    // One extra 16-byte segment disjoins the eight rows of an ldmatrix across banks.
    alignas(16) unsigned qs[Rows][BK * Words + 4];
    float scale[BK][Rows + 32 / BK];
    float other[Other ? BK : 1][Other ? Rows + 32 / BK : 1];
};

template <int Bytes>
__device__ __forceinline__ void mmq_copy_async(void* dst, const void* src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(smem), "l"(src), "n"(Bytes) : "memory");
}

template <typename W, int BM, int BN, int BK>
__device__ __forceinline__ void load_mmq_tile(
    const BlockQ8_1* x, const W* w,
    MmqTile<BM, BK, 8, mmq_has_min<W>>& xs,
    MmqTile<BN, BK, mmq_packed_q4<W> ? 4 : 8,
            !std::is_same_v<W, BlockQ4_A> && !std::is_same_v<W, BlockQ8_A>>& ws,
    int m0, int n0, int M, int N, int K, int kb) {
    constexpr bool async = mmq_async<W>;
    int slot = threadIdx.x % BK;
    int xstride = K / 32, wstride = K / 256;
    int sub = (kb + slot) % 8;
    for (int m = threadIdx.x / BK; m < BM; m += blockDim.x / BK) {
        const auto* src = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(min(m0 + m, M - 1)) * xstride + kb + slot);
        auto* dst = reinterpret_cast<uint4*>(xs.qs[m] + slot * 8);
        if constexpr (async) {
            mmq_copy_async<16>(dst, src);
            mmq_copy_async<16>(dst + 1, src + 1);
            mmq_copy_async<4>(&xs.scale[slot][m], &src[2].x);
        } else {
            dst[0] = __ldg(src);
            dst[1] = __ldg(src + 1);
            xs.scale[slot][m] = __uint_as_float(__ldg(&src[2].x));
        }
        if constexpr (mmq_has_min<W>) {
            if constexpr (async) mmq_copy_async<4>(&xs.other[slot][m], &src[2].w);
            else xs.other[slot][m] = __uint_as_float(__ldg(&src[2].w));
        }
    }
    for (int n = threadIdx.x / BK; n < BN; n += blockDim.x / BK) {
        const W& src = w[static_cast<int64_t>(min(n0 + n, N - 1)) * wstride + (kb + slot) / 8];
        constexpr int words = mmq_packed_q4<W> ? 4 : 8;
        auto* dst = reinterpret_cast<uint4*>(ws.qs[n] + slot * words);
        if constexpr (std::is_same_v<W, BlockQ4_A>) {
            mmq_copy_async<16>(dst, reinterpret_cast<const uint4*>(src.qs) + sub);
            const auto* scale = reinterpret_cast<const unsigned*>(src.d) + sub / 2;
            mmq_copy_async<4>(&ws.scale[slot][n], scale);
        } else if constexpr (std::is_same_v<W, BlockQ4_K>) {
            // Adjacent slots stage the two aligned halves of a packed Q4_K pair.
            mmq_copy_async<16>(dst, reinterpret_cast<const uint4*>(src.qs) + sub);
            const float2 scales = mmq_k_scales(src, sub);
            ws.scale[slot][n] = scales.x;
            ws.other[slot][n] = scales.y;
        } else if constexpr (std::is_same_v<W, BlockQ4_KA>) {
            mmq_copy_async<16>(dst, src.q[sub]);
            const float2 scales = q4_ka_scales(src, sub);
            ws.scale[slot][n] = scales.x;
            ws.other[slot][n] = -scales.y;
        } else if constexpr (std::is_same_v<W, BlockQ6_A>) {
            mmq_copy_async<16>(dst, src.ql[sub]);
            mmq_copy_async<16>(dst + 1, src.qh[sub / 2]);
            const float d = half_bits_to_f32(src.d);
            ws.scale[slot][n] = d * src.scale[2 * sub];
            ws.other[slot][n] = d * src.scale[2 * sub + 1];
        } else if constexpr (std::is_same_v<W, BlockQ5_K>) {
            // Adjacent slots stage the low/high halves consumed by each 32-element dot.
            mmq_copy_async<16>(dst, reinterpret_cast<const uint4*>(src.qs) + sub);
            mmq_copy_async<16>(dst + 1, reinterpret_cast<const uint4*>(src.qh) + sub % 2);
            const float2 scales = mmq_k_scales(src, sub);
            ws.scale[slot][n] = scales.x;
            ws.other[slot][n] = scales.y;
        } else if constexpr (std::is_same_v<W, BlockQ8_A>) {
            auto* ptr = reinterpret_cast<const uint4*>(src.qs) + 2 * sub;
            dst[0] = __ldg(ptr);
            dst[1] = __ldg(ptr + 1);
            ws.scale[slot][n] = half_bits_to_f32(__ldg(src.d + sub));
        } else {
            MmqWeight a = load_raw_q6_mmq_weight(src, sub, 0);
            MmqWeight b = load_raw_q6_mmq_weight(src, sub, 1);
            MmqWeight c = load_raw_q6_mmq_weight(src, sub, 2);
            MmqWeight d = load_raw_q6_mmq_weight(src, sub, 3);
            dst[0] = make_uint4(a.lo, b.lo, c.lo, d.lo);
            dst[1] = make_uint4(a.hi, b.hi, c.hi, d.hi);
            ws.scale[slot][n] = a.scale;
            ws.other[slot][n] = a.other;
        }
    }
    if constexpr (async) asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ uint4 mmq_load_a(const unsigned* ptr) {
    uint4 r;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(shared));
    return r;
}

__device__ __forceinline__ uint2 mmq_load_b(const unsigned* ptr) {
    uint2 r;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r.x), "=r"(r.y) : "r"(shared));
    return r;
}

__device__ __forceinline__ unsigned mmq_load_packed_q4(const unsigned* ptr) {
    unsigned q;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(q) : "r"(shared));
    return q;
}

template <typename W, int BM, int BN, int BK>
struct MmqBuffers {
    static constexpr bool split_scale = std::is_same_v<W, BlockQ6_K> || std::is_same_v<W, BlockQ6_A>;
    static constexpr bool min_scale = mmq_has_min<W>;
    static constexpr int weight_words = mmq_packed_q4<W> ? 4 : 8;

    MmqTile<BM, BK, 8, min_scale> xs[2];
    MmqTile<BN, BK, weight_words, split_scale || min_scale> ws[2];
};

template <typename W, int BM, int BN, int WN, int BK, int WM = BM / 16>
__device__ __forceinline__ void mmq_q8_1_tile(const BlockQ8_1* __restrict__ x, const W* __restrict__ w,
    float* __restrict__ y, int M, int N, int K, int m0, int n0, MmqBuffers<W, BM, BN, BK>& storage) {
    constexpr int warps_n = BN / (WN * 8);
    constexpr bool split_scale = MmqBuffers<W, BM, BN, BK>::split_scale;
    constexpr bool min_scale = MmqBuffers<W, BM, BN, BK>::min_scale;
    auto& xs = storage.xs;
    auto& ws = storage.ws;
    int warp = threadIdx.x / 32;
    int warp_n = (warp % warps_n) * (WN * 8);
    int warp_m = (warp / warps_n) * (WM * 16);
    int lane = threadIdx.x % 32, group = lane / 4, word = lane % 4;
    float4 acc[WM][WN] = {};

    load_mmq_tile<W, BM, BN, BK>(x, w, xs[0], ws[0], m0, n0, M, N, K, 0);
    if constexpr (mmq_async<W>) asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int kb = 0; kb < K / 32; kb += BK) {
        const auto& a = xs[(kb / BK) & 1];
        const auto& b = ws[(kb / BK) & 1];
        if (kb + BK < K / 32) {
            load_mmq_tile<W, BM, BN, BK>(x, w, xs[1 - ((kb / BK) & 1)], ws[1 - ((kb / BK) & 1)],
                                      m0, n0, M, N, K, kb + BK);
        }

        #pragma unroll
        for (int slot = 0; slot < BK; ++slot) {
            uint2 weights[WN];
            float2 scales[WN], other[WN];
            #pragma unroll
            for (int ni = 0; ni < WN; ++ni) {
                int j = warp_n + ni * 8;
                int j0 = j + word * 2, j1 = j0 + 1;
                uint2 bv;
                if constexpr (std::is_same_v<W, BlockQ4_A>) {
                    const unsigned packed = mmq_load_packed_q4(b.qs[j + lane % 8] + slot * 4);
                    bv = make_uint2(mmq_signed_q4(packed), mmq_signed_q4(packed >> 4));
                } else if constexpr (std::is_same_v<W, BlockQ4_KA>) {
                    const unsigned packed = mmq_load_packed_q4(b.qs[j + lane % 8] + slot * 4);
                    bv = make_uint2(packed & kNibbleMask, (packed >> 4) & kNibbleMask);
                } else if constexpr (std::is_same_v<W, BlockQ6_A>) {
                    const unsigned* row = b.qs[j + lane % 8] + slot * 8;
                    const auto low = mmq_load_packed_q4(row);
                    const auto high = mmq_load_packed_q4(row + 4);
                    bv = make_uint2((unpack_q6_a(low, high, 0, slot % 2) + 0x60606060u) ^ 0x80808080u,
                                    (unpack_q6_a(low, high, 1, slot % 2) + 0x60606060u) ^ 0x80808080u);
                } else if constexpr (std::is_same_v<W, BlockQ4_K>) {
                    const auto* row = b.qs[j + lane % 8] + (slot & ~1) * 4;
                    const uint2 packed = mmq_load_b(row + ((lane / 8) % 2) * 4);
                    const int shift = (slot & 1) * 4;
                    bv = make_uint2((packed.x >> shift) & kNibbleMask, (packed.y >> shift) & kNibbleMask);
                } else if constexpr (std::is_same_v<W, BlockQ5_K>) {
                    const auto* row = b.qs[j + lane % 8] + ((slot & ~1) + (lane / 8) % 2) * 8;
                    const uint2 packed = mmq_load_b(row);
                    const uint2 high = mmq_load_b(row + 4);
                    const int shift = (slot & 1) * 4;
                    const int bit = (kb + slot) % 8;
                    bv = make_uint2(((packed.x >> shift) & kNibbleMask) | (((high.x >> bit) & 0x01010101u) << 4),
                                    ((packed.y >> shift) & kNibbleMask) | (((high.y >> bit) & 0x01010101u) << 4));
                } else {
                    bv = mmq_load_b(b.qs[j + lane % 8] + slot * 8 + ((lane / 8) % 2) * 4);
                }
                weights[ni] = bv;
                if constexpr (std::is_same_v<W, BlockQ4_A>) {
                    int shift = (slot % 2) * 16;
                    scales[ni] = make_float2(
                        half_bits_to_f32(static_cast<uint16_t>(__float_as_uint(b.scale[slot][j0]) >> shift)),
                        half_bits_to_f32(static_cast<uint16_t>(__float_as_uint(b.scale[slot][j1]) >> shift)));
                } else {
                    scales[ni] = make_float2(b.scale[slot][j0], b.scale[slot][j1]);
                }
                if constexpr (split_scale || min_scale) {
                    other[ni] = make_float2(b.other[slot][j0], b.other[slot][j1]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < WM; ++mi) {
                const int row = warp_m + mi * 16;
                const int i0 = row + group, i1 = i0 + 8;
                const uint4 av = mmq_load_a(a.qs[row + lane % 16] + slot * 8 + (lane / 16) * 4);
                const float ax = a.scale[slot][i0], ay = a.scale[slot][i1];
                float sx, sy;
                if constexpr (min_scale) {
                    sx = a.other[slot][i0];
                    sy = a.other[slot][i1];
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ++ni) {
                    const uint2 bv = weights[ni];
                    const float bx = scales[ni].x, by = scales[ni].y;
                    int4 c;
                    if constexpr (std::is_same_v<W, BlockQ6_A>) c = mma_s8_k16(av.x, av.y, bv.x);
                    else c = mma_s8(av.x, av.y, av.z, av.w, bv.x, split_scale ? 0u : bv.y);
                    auto& f = acc[mi][ni];
                    if constexpr (split_scale) {
                        int4 hi;
                        if constexpr (std::is_same_v<W, BlockQ6_A>) hi = mma_s8_k16(av.z, av.w, bv.y);
                        else hi = mma_s8(0, 0, av.z, av.w, 0, bv.y);
                        float hx = other[ni].x, hy = other[ni].y;
                        f.x = fmaf(ax, fmaf(hx, hi.x, bx * c.x), f.x);
                        f.y = fmaf(ax, fmaf(hy, hi.y, by * c.y), f.y);
                        f.z = fmaf(ay, fmaf(hx, hi.z, bx * c.z), f.z);
                        f.w = fmaf(ay, fmaf(hy, hi.w, by * c.w), f.w);
                    } else {
                        f.x = fmaf(ax * bx, c.x, f.x);
                        f.y = fmaf(ax * by, c.y, f.y);
                        f.z = fmaf(ay * bx, c.z, f.z);
                        f.w = fmaf(ay * by, c.w, f.w);
                        if constexpr (min_scale) {
                            float dx = other[ni].x, dy = other[ni].y;
                            f.x = fmaf(sx, dx, f.x);
                            f.y = fmaf(sx, dy, f.y);
                            f.z = fmaf(sy, dx, f.z);
                            f.w = fmaf(sy, dy, f.w);
                        }
                    }
                }
            }
        }
        if constexpr (mmq_async<W>) asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        int i0 = m0 + warp_m + mi * 16 + group, i1 = i0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            int j0 = n0 + warp_n + ni * 8 + word * 2, j1 = j0 + 1;
            const auto& f = acc[mi][ni];
            if (i0 < M && j0 < N) y[static_cast<int64_t>(i0) * N + j0] = f.x;
            if (i0 < M && j1 < N) y[static_cast<int64_t>(i0) * N + j1] = f.y;
            if (i1 < M && j0 < N) y[static_cast<int64_t>(i1) * N + j0] = f.z;
            if (i1 < M && j1 < N) y[static_cast<int64_t>(i1) * N + j1] = f.w;
        }
    }
}

template <typename W, int BM, int BN, int WN, int BK = 4, int WM = BM / 16>
__global__ void k_mmq_q8_1(const BlockQ8_1* x, const W* w, float* y, int M, int N, int K) {
    using Buffers = MmqBuffers<W, BM, BN, BK>;
    if constexpr (sizeof(Buffers) > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char mmq_smem[];
        auto& storage = *reinterpret_cast<Buffers*>(mmq_smem);
        mmq_q8_1_tile<W, BM, BN, WN, BK, WM>(x, w, y, M, N, K, blockIdx.y * BM, blockIdx.x * BN, storage);
    } else {
        __shared__ Buffers storage;
        mmq_q8_1_tile<W, BM, BN, WN, BK, WM>(x, w, y, M, N, K, blockIdx.y * BM, blockIdx.x * BN, storage);
    }
}

// Use the same tiles as the large Q4 projection and queue the narrow Q8 projection in the same launch.
__global__ void k_mmq_q4_a_and_q8_a(const BlockQ8_1* x, const BlockQ4_A* w4, const BlockQ8_A* w8,
                                     float* y4, float* y8, int M, int N4, int N8, int K,
                                     int q4_tiles, int q4_blocks) {
    __shared__ union {
        MmqBuffers<BlockQ4_A, 64, 128, 4> q4;
        MmqBuffers<BlockQ8_A, 16, 64, 4> q8;
    } storage;
    const int block = blockIdx.x;
    if (block < q4_blocks) {
        mmq_q8_1_tile<BlockQ4_A, 64, 128, 2, 4>(x, w4, y4, M, N4, K,
            block / q4_tiles * 64, block % q4_tiles * 128, storage.q4);
    } else {
        mmq_q8_1_tile<BlockQ8_A, 16, 64, 1, 4>(x, w8, y8, M, N8, K,
            (block - q4_blocks) * 16, 0, storage.q8);
    }
}

template <typename W, int BM, int BN, int WN, int BK = 4, int WM = BM / 16>
void launch_mmq_tile(const BlockQ8_1* x, const W* w, float* y, int M, int N, int K, cudaStream_t stream) {
    constexpr int threads = (BM / (WM * 16)) * (BN / (WN * 8)) * 32;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    constexpr int bytes = sizeof(MmqBuffers<W, BM, BN, BK>);
    constexpr int dynamic_bytes = bytes > 48 * 1024 ? bytes : 0;
    if constexpr (dynamic_bytes > 0) {
        static thread_local int configured_device = -1;
        static thread_local bool fits = false;
        int device = 0;
        FUSIONRT_CUDA_CHECK(cudaGetDevice(&device));
        if (configured_device != device) {
            int limit = 0;
            FUSIONRT_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            fits = dynamic_bytes <= limit;
            if (fits) {
                FUSIONRT_CUDA_CHECK(cudaFuncSetAttribute(k_mmq_q8_1<W, BM, BN, WN, BK, WM>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
            }
            configured_device = device;
        }
        if (!fits) {
            launch_mmq_tile<W, 64, 64, 2, 4, 2>(x, w, y, M, N, K, stream);
            return;
        }
    }
    k_mmq_q8_1<W, BM, BN, WN, BK, WM><<<grid, threads, dynamic_bytes, stream>>>(x, w, y, M, N, K);
}

template <typename W>
void launch_mmq(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    auto* qx = static_cast<const BlockQ8_1*>(x);
    auto* qw = static_cast<const W*>(w);
    if (M <= 16) {
        launch_mmq_tile<W, 16, 32, 1, 8>(qx, qw, y, M, N, K, stream);
        return;
    }
    if (M <= 32) {
        launch_mmq_tile<W, 32, 32, 1, 8>(qx, qw, y, M, N, K, stream);
        return;
    }
    if constexpr (std::is_same_v<W, BlockQ4_A>) {
        if (M >= 64 && N >= 4096) {
            launch_mmq_tile<W, 64, 128, 2>(qx, qw, y, M, N, K, stream);
            return;
        }
    }
    if constexpr (std::is_same_v<W, BlockQ4_KA> || std::is_same_v<W, BlockQ5_K>) {
        // A narrower output tile gives short wide projections more CTAs.
        if (M >= 128 && M < 256 && N >= 4096 && N <= 8192) {
            launch_mmq_tile<W, 128, 96, 2, 4, 4>(qx, qw, y, M, N, K, stream);
            return;
        }
        if ((std::is_same_v<W, BlockQ4_KA> && M >= 128 && N >= 4096) || (M >= 256 && N >= 1024)) {
            launch_mmq_tile<W, 128, 128, 2, 4, 4>(qx, qw, y, M, N, K, stream);
            return;
        }
        if (M >= 128 && N >= 2048) {
            launch_mmq_tile<W, 128, 64, 2, 4, 2>(qx, qw, y, M, N, K, stream);
            return;
        }
    }
    if constexpr (std::is_same_v<W, BlockQ6_A>) {
        if (M >= 256 && N >= 4096) {
            launch_mmq_tile<W, 128, 64, 2, 4, 2>(qx, qw, y, M, N, K, stream);
            return;
        }
    }
    if constexpr (std::is_same_v<W, BlockQ4_KA> || std::is_same_v<W, BlockQ5_K> ||
                  std::is_same_v<W, BlockQ6_A>) {
        if (M >= 64 && N >= 1024) {
            launch_mmq_tile<W, 64, 64, 2, 4, 2>(qx, qw, y, M, N, K, stream);
            return;
        }
    }
    if constexpr (std::is_same_v<W, BlockQ8_A>) {
        if (M >= 64 && N <= 128) {
            launch_mmq_tile<W, 64, 64, 2, 8, 2>(qx, qw, y, M, N, K, stream);
            return;
        }
    }
    if constexpr (std::is_same_v<W, BlockQ4_K>) {
        if (M >= 64 && N >= 1024) {
            launch_mmq_tile<W, 64, 64, 1>(qx, qw, y, M, N, K, stream);
            return;
        }
    }
    if (M >= 64 && K >= 4096) {
        launch_mmq_tile<W, 64, 32, 1>(qx, qw, y, M, N, K, stream);
        return;
    }
    launch_mmq_tile<W, 32, 64, 2>(qx, qw, y, M, N, K, stream);
}

template <typename W, int NR, int WARPS>
__device__ __forceinline__ void gemv_q8_1_rows(const BlockQ8_1* __restrict__ x, const W* __restrict__ w,
                                                float* __restrict__ y, int N, int K, int block) {
    int row = (block * WARPS + threadIdx.x / 32) * NR;
    if (row >= N) return;
    int lane = threadIdx.x & 31;
    int blocks = K / 256;
    w += static_cast<int64_t>(row) * blocks;
    float acc[NR] = {};
    for (int b = lane; b < K / 32; b += 32) {
        const auto* qs = reinterpret_cast<const uint4*>(x + b);
        uint4 lo = __ldg(qs);
        uint4 hi = __ldg(qs + 1);
        float4 stats = __ldg(reinterpret_cast<const float4*>(qs + 2));
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            if (r == 0 || row + r < N) {
                acc[r] += dot_q8_1(w[r * blocks + b / 8], b & 7, lo, hi, stats);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        float total = warp_reduce_sum(acc[r]);
        if (lane == 0 && row + r < N) y[row + r] = total;
    }
}

template <typename W, int NR, int WARPS>
__global__ void k_gemv_q8_1(const BlockQ8_1* x, const W* w, float* y, int N, int K) {
    gemv_q8_1_rows<W, NR, WARPS>(x, w, y, N, K, blockIdx.x);
}

template <int NR, int WARPS>
__global__ void k_gemv_q4_a_and_q8_a(const BlockQ8_1* x, const BlockQ4_A* w4, const BlockQ8_A* w8,
                                      float* y4, float* y8, int N4, int N8, int K, int q4_blocks) {
    const int block = blockIdx.x;
    if (block < q4_blocks) gemv_q8_1_rows<BlockQ4_A, NR, WARPS>(x, w4, y4, N4, K, block);
    else gemv_q8_1_rows<BlockQ8_A, NR, WARPS>(x, w8, y8, N8, K, block - q4_blocks);
}

template <typename W, int NR = 2, int WARPS = 4>
void launch_gemv_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream) {
    constexpr int rows = NR * WARPS;
    k_gemv_q8_1<W, NR, WARPS><<<(N + rows - 1) / rows, WARPS * 32, 0, stream>>>(
        static_cast<const BlockQ8_1*>(x), static_cast<const W*>(w), y, N, K);
}

template <typename W, int DefaultWarps = 4>
void launch_gemv_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                      GemvGeometry geometry) {
    if (geometry.warps != 4 && geometry.warps != 8) geometry.warps = DefaultWarps;
    if (geometry.warps == 4) {
        if (geometry.rows_per_warp == 1) launch_gemv_q8_1<W, 1, 4>(x, w, y, N, K, stream);
        else launch_gemv_q8_1<W, 2, 4>(x, w, y, N, K, stream);
    } else {
        if (geometry.rows_per_warp == 1) launch_gemv_q8_1<W, 1, 8>(x, w, y, N, K, stream);
        else launch_gemv_q8_1<W, 2, 8>(x, w, y, N, K, stream);
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM, bool HAS_RES = false>
__device__ __forceinline__ void prepare_q8_smem(
    const float* __restrict__ x,
    const float* __restrict__ res_b,
    const float* __restrict__ norm_w,
    float* __restrict__ res_out,
    int K, float eps,
    float* __restrict__ sh_d,
    float* __restrict__ sh_neg8sum,
    int32_t* __restrict__ sh_q8_qs)
{
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        float sum_sq = 0.0f;
        int k4 = K >> 2;
        const float4* xm4 = reinterpret_cast<const float4*>(x);
        const float4* bm4 = HAS_RES ? reinterpret_cast<const float4*>(res_b) : nullptr;
        float4* rm4 = (HAS_RES && blockIdx.x == 0 && res_out != nullptr) ? reinterpret_cast<float4*>(res_out) : nullptr;
        for (int i = tid; i < k4; i += blockDim.x) {
            float4 v = __ldg(xm4 + i);
            if constexpr (HAS_RES) {
                float4 bv = __ldg(bm4 + i);
                v.x += bv.x; v.y += bv.y; v.z += bv.z; v.w += bv.w;
                if (rm4) rm4[i] = v;
            }
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        float total_sq = block_reduce_sum(sum_sq, red_buf);
        if (tid == 0) inv_rms_sh = rsqrtf(total_sq / static_cast<float>(K) + eps);
        __syncthreads();
    }

    int nb = K >> 5;
    float inv = HAS_NORM ? inv_rms_sh : 1.0f;
    for (int b = wid; b < nb; b += WARPS_PER_BLOCK) {
        int idx = (b << 5) + lane;
        float v = __ldg(&x[idx]);
        if constexpr (HAS_RES) {
            v += __ldg(&res_b[idx]);
        }
        if constexpr (HAS_NORM) {
            v *= inv * __ldg(&norm_w[idx]);
        }
        float amax = fabsf(v);
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }
        float d_x = amax / 127.0f;
        float id_x = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = max(-127, min(127, __float2int_rn(v * id_x)));

        int q_sum = q;
        if (sh_neg8sum != nullptr) {
            for (int off = 16; off > 0; off >>= 1) {
                q_sum += __shfl_xor_sync(0xFFFFFFFFu, q_sum, off);
            }
        }
        int q0 = q;
        int q1 = __shfl_down_sync(0xFFFFFFFFu, q, 1);
        int q2 = __shfl_down_sync(0xFFFFFFFFu, q, 2);
        int q3 = __shfl_down_sync(0xFFFFFFFFu, q, 3);
        if ((lane & 3) == 0) {
            sh_q8_qs[b * 9 + (lane >> 2)] =
                (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
        if (lane == 0) {
            sh_d[b] = d_x;
            if (sh_neg8sum != nullptr) {
                sh_neg8sum[b] = -8.0f * d_x * static_cast<float>(q_sum);
            }
        }
    }
    __syncthreads();
}

template <int WARPS_PER_BLOCK, bool HAS_NORM, bool HAS_RES = false>
__global__ void k_gemv_q4_a_dp4a(const float* __restrict__ x,
                                 const float* __restrict__ res_b,
                                 const float* __restrict__ norm_w,
                                 const BlockQ4_A* __restrict__ w,
                                 float* __restrict__ res_out,
                                 float* __restrict__ y,
                                 int N, int K, float eps) {
    extern __shared__ uint8_t dyn_smem[];
    int nb = K >> 5;
    float* sh_d = reinterpret_cast<float*>(dyn_smem);
    float* sh_neg8sum = sh_d + nb;
    int32_t* sh_q8_qs = reinterpret_cast<int32_t*>(sh_neg8sum + nb);

    prepare_q8_smem<WARPS_PER_BLOCK, HAS_NORM, HAS_RES>(x, res_b, norm_w, res_out, K, eps, sh_d, sh_neg8sum, sh_q8_qs);

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int ns = K >> 8;
    const BlockQ4_A* wb = w + static_cast<int64_t>(n) * ns;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        int s = b >> 3;
        int sb = b & 7;
        uint4 qw = *reinterpret_cast<const uint4*>(&wb[s].qs[sb * 4]);
        const int32_t* xb = &sh_q8_qs[b * 9];
        int dot = 0;
        dot = __dp4a(static_cast<int>(qw.x & 0x0F0F0F0Fu), xb[0], dot);
        dot = __dp4a(static_cast<int>((qw.x >> 4) & 0x0F0F0F0Fu), xb[4], dot);
        dot = __dp4a(static_cast<int>(qw.y & 0x0F0F0F0Fu), xb[1], dot);
        dot = __dp4a(static_cast<int>((qw.y >> 4) & 0x0F0F0F0Fu), xb[5], dot);
        dot = __dp4a(static_cast<int>(qw.z & 0x0F0F0F0Fu), xb[2], dot);
        dot = __dp4a(static_cast<int>((qw.z >> 4) & 0x0F0F0F0Fu), xb[6], dot);
        dot = __dp4a(static_cast<int>(qw.w & 0x0F0F0F0Fu), xb[3], dot);
        dot = __dp4a(static_cast<int>((qw.w >> 4) & 0x0F0F0F0Fu), xb[7], dot);
        float d_w = half_bits_to_f32(__ldg(&wb[s].d[sb]));
        acc += d_w * (static_cast<float>(dot) * sh_d[b] + sh_neg8sum[b]);
    }
    float s = warp_reduce_sum(acc);
    if (lane == 0) y[n] = s;
}

template <int WARPS_PER_BLOCK, bool HAS_NORM, bool HAS_RES = false>
__global__ void k_gemv_q8_a_dp4a(const float* __restrict__ x,
                                 const float* __restrict__ res_b,
                                 const float* __restrict__ norm_w,
                                 const BlockQ8_A* __restrict__ w,
                                 float* __restrict__ res_out,
                                 float* __restrict__ y,
                                 int N, int K, float eps) {
    extern __shared__ uint8_t dyn_smem[];
    int nb = K >> 5;
    float* sh_d = reinterpret_cast<float*>(dyn_smem);
    int32_t* sh_q8_qs = reinterpret_cast<int32_t*>(sh_d + nb);

    prepare_q8_smem<WARPS_PER_BLOCK, HAS_NORM, HAS_RES>(x, res_b, norm_w, res_out, K, eps, sh_d, nullptr, sh_q8_qs);

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int ns = K >> 8;
    const BlockQ8_A* wb = w + static_cast<int64_t>(n) * ns;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        int s = b >> 3;
        int sb = b & 7;
        const uint4* wp = reinterpret_cast<const uint4*>(&wb[s].qs[sb * 8]);
        uint4 qw0 = wp[0];
        uint4 qw1 = wp[1];
        const int32_t* xb = &sh_q8_qs[b * 9];
        int dot = 0;
        dot = __dp4a(static_cast<int>(qw0.x), xb[0], dot);
        dot = __dp4a(static_cast<int>(qw0.y), xb[1], dot);
        dot = __dp4a(static_cast<int>(qw0.z), xb[2], dot);
        dot = __dp4a(static_cast<int>(qw0.w), xb[3], dot);
        dot = __dp4a(static_cast<int>(qw1.x), xb[4], dot);
        dot = __dp4a(static_cast<int>(qw1.y), xb[5], dot);
        dot = __dp4a(static_cast<int>(qw1.z), xb[6], dot);
        dot = __dp4a(static_cast<int>(qw1.w), xb[7], dot);
        float d_w = half_bits_to_f32(__ldg(&wb[s].d[sb]));
        acc += d_w * sh_d[b] * static_cast<float>(dot);
    }
    float s = warp_reduce_sum(acc);
    if (lane == 0) y[n] = s;
}

template <int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_quant_gemv_q4_a_f32(const float* __restrict__ x,
                                      const float* __restrict__ norm_w,
                                      const BlockQ4_A* __restrict__ w,
                                      float* __restrict__ y,
                                      int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 5;
    const BlockQ4_A* wb = w + static_cast<int64_t>(n) * (K >> 8);
    float acc[8] = {0.f};

    int sub = lane & 7;
    int bi = lane >> 3;
    for (int b = bi; b < nb; b += 4) {
        int s = b >> 3;
        int sb = b & 7;
        float d = half_bits_to_f32(__ldg(&wb[s].d[sb]));
        uint16_t raw2 = __ldg(reinterpret_cast<const uint16_t*>(&wb[s].qs[sb * 4]) + sub);
        uint8_t b0 = raw2 & 0xFFu;
        uint8_t b1 = raw2 >> 8;
        float q0 = static_cast<float>(static_cast<int>(b0 & 0x0Fu) - 8) * d;
        float q1 = static_cast<float>(static_cast<int>(b1 & 0x0Fu) - 8) * d;
        float q2 = static_cast<float>(static_cast<int>(b0 >> 4) - 8) * d;
        float q3 = static_cast<float>(static_cast<int>(b1 >> 4) - 8) * d;
        int k_lo = (b << 5) + (sub << 1);
        int k_hi = k_lo + 16;
        if constexpr (HAS_NORM) {
            float2 nwl = __ldg(reinterpret_cast<const float2*>(norm_w + k_lo));
            float2 nwh = __ldg(reinterpret_cast<const float2*>(norm_w + k_hi));
            q0 *= nwl.x; q1 *= nwl.y;
            q2 *= nwh.x; q3 *= nwh.y;
        }
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float2 xl = __ldg(reinterpret_cast<const float2*>(xm + k_lo));
            float2 xh = __ldg(reinterpret_cast<const float2*>(xm + k_hi));
            acc[m] += q0 * xl.x + q1 * xl.y + q2 * xh.x + q3 * xh.y;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM>
__global__ void k_quant_gemv_q8_a_f32(const float* __restrict__ x,
                                      const float* __restrict__ norm_w,
                                      const BlockQ8_A* __restrict__ w,
                                      float* __restrict__ y,
                                      int M, int N, int K, float eps) {
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    if constexpr (HAS_NORM) {
        for (int m = 0; m < M; ++m) {
            const float* xm = x + static_cast<int64_t>(m) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += blockDim.x) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            float total_sq = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[m] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 5;
    const BlockQ8_A* wb = w + static_cast<int64_t>(n) * (K >> 8);
    float acc[8] = {0.f};

    int sub = lane & 7;
    int bi = lane >> 3;
    for (int b = bi; b < nb; b += 4) {
        int s = b >> 3;
        int sb = b & 7;
        float d = half_bits_to_f32(__ldg(&wb[s].d[sb]));
        int32_t raw4 = __ldg(&wb[s].qs[sb * 8 + sub]);
        float q0 = static_cast<float>(static_cast<int8_t>(raw4 & 0xFF)) * d;
        float q1 = static_cast<float>(static_cast<int8_t>((raw4 >> 8) & 0xFF)) * d;
        float q2 = static_cast<float>(static_cast<int8_t>((raw4 >> 16) & 0xFF)) * d;
        float q3 = static_cast<float>(static_cast<int8_t>((raw4 >> 24) & 0xFF)) * d;
        int k_off = (b << 5) + (sub << 2);
        if constexpr (HAS_NORM) {
            float4 nw4 = __ldg(reinterpret_cast<const float4*>(norm_w + k_off));
            q0 *= nw4.x; q1 *= nw4.y; q2 *= nw4.z; q3 *= nw4.w;
        }
        for (int m = 0; m < M; ++m) {
            float4 x4 = __ldg(reinterpret_cast<const float4*>(x + static_cast<int64_t>(m) * K + k_off));
            acc[m] += q0 * x4.x + q1 * x4.y + q2 * x4.z + q3 * x4.w;
        }
    }
    for (int m = 0; m < M; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if constexpr (HAS_NORM) s *= inv_rms_sh[m];
        if (lane == 0) y[static_cast<int64_t>(m) * N + n] = s;
    }
}

template <int WARPS_PER_BLOCK, bool HAS_NORM, bool HAS_RES = false>
__global__ void k_quant_gemv_q6_k_m1(const float* __restrict__ x,
                                     const float* __restrict__ res_b,
                                     const float* __restrict__ norm_w,
                                     const BlockQ6_K* __restrict__ w,
                                     float* __restrict__ res_out,
                                     float* __restrict__ y,
                                     int N, int K, float eps) {
    extern __shared__ float sh_x[];
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    int k4 = K >> 2;
    const float4* xm4 = reinterpret_cast<const float4*>(x);
    const float4* bm4 = HAS_RES ? reinterpret_cast<const float4*>(res_b) : nullptr;
    float4* rm4 = (HAS_RES && blockIdx.x == 0 && res_out != nullptr) ? reinterpret_cast<float4*>(res_out) : nullptr;
    if constexpr (HAS_NORM) {
        float sum_sq = 0.0f;
        for (int i = tid; i < k4; i += blockDim.x) {
            float4 v = __ldg(xm4 + i);
            if constexpr (HAS_RES) {
                float4 bv = __ldg(bm4 + i);
                v.x += bv.x; v.y += bv.y; v.z += bv.z; v.w += bv.w;
                if (rm4) rm4[i] = v;
            }
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        float total_sq = block_reduce_sum(sum_sq, red_buf);
        if (tid == 0) inv_rms_sh = rsqrtf(total_sq / static_cast<float>(K) + eps);
        __syncthreads();
        float inv = inv_rms_sh;
        const float4* nw4 = reinterpret_cast<const float4*>(norm_w);
        for (int i = tid; i < k4; i += blockDim.x) {
            float4 xv = __ldg(xm4 + i);
            if constexpr (HAS_RES) {
                float4 bv = __ldg(bm4 + i);
                xv.x += bv.x; xv.y += bv.y; xv.z += bv.z; xv.w += bv.w;
            }
            float4 nwv = __ldg(nw4 + i);
            int k0 = i << 2;
            sh_x[k0 + 0] = xv.x * inv * nwv.x;
            sh_x[k0 + 1] = xv.y * inv * nwv.y;
            sh_x[k0 + 2] = xv.z * inv * nwv.z;
            sh_x[k0 + 3] = xv.w * inv * nwv.w;
        }
    } else {
        for (int i = tid; i < k4; i += blockDim.x) {
            float4 xv = __ldg(xm4 + i);
            int k0 = i << 2;
            sh_x[k0 + 0] = xv.x;
            sh_x[k0 + 1] = xv.y;
            sh_x[k0 + 2] = xv.z;
            sh_x[k0 + 3] = xv.w;
        }
    }
    __syncthreads();

    int n = blockIdx.x * WARPS_PER_BLOCK + wid;
    if (n >= N) return;

    int nb = K >> 8;
    const BlockQ6_K* wb = w + static_cast<int64_t>(n) * nb;
    float acc = 0.0f;
    int is = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const BlockQ6_K* bp = wb + b;
        float d = half_bits_to_f32(__ldg(&bp->d));
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const uint8_t* ql = bp->ql + half_idx * 64;
            const uint8_t* qh = bp->qh + half_idx * 32;
            const int8_t* sc = bp->scales + half_idx * 8;
            uint8_t ql0 = __ldg(&ql[lane]);
            uint8_t ql1 = __ldg(&ql[lane + 32]);
            uint8_t qhv = __ldg(&qh[lane]);
            float s0 = d * static_cast<float>(__ldg(&sc[is + 0]));
            float s2 = d * static_cast<float>(__ldg(&sc[is + 2]));
            float s4 = d * static_cast<float>(__ldg(&sc[is + 4]));
            float s6 = d * static_cast<float>(__ldg(&sc[is + 6]));
            int kbase = (b << 8) + half_idx * 128 + lane;
            acc += s0 * static_cast<float>(static_cast<int>((ql0 & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32) * sh_x[kbase + 0]
                 + s2 * static_cast<float>(static_cast<int>((ql1 & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32) * sh_x[kbase + 32]
                 + s4 * static_cast<float>(static_cast<int>((ql0 >> 4) | (((qhv >> 4) & 3u) << 4)) - 32) * sh_x[kbase + 64]
                 + s6 * static_cast<float>(static_cast<int>((ql1 >> 4) | (((qhv >> 6) & 3u) << 4)) - 32) * sh_x[kbase + 96];
        }
    }
    float s = warp_reduce_sum(acc);
    if (lane == 0) y[n] = s;
}

__global__ void k_dequant_q4_a_to_f16(const BlockQ4_A* __restrict__ w, half* __restrict__ out, int N, int K) {
    int ns = K >> 8;
    int s_idx = blockIdx.x * blockDim.y + threadIdx.y;  // 1 warp per superblock
    int lane = threadIdx.x;
    int total_s = N * ns;
    if (s_idx >= total_s) return;
    const BlockQ4_A& blk = w[s_idx];
    half* dst = out + static_cast<int64_t>(s_idx) * 256;
    for (int sb = 0; sb < 8; ++sb) {
        float d = half_bits_to_f32(__ldg(&blk.d[sb]));
        int byte_idx = lane & 15;
        uint8_t byte = reinterpret_cast<const uint8_t*>(&blk.qs[sb * 4])[byte_idx];
        int q = (lane < 16) ? (static_cast<int>(byte & 0x0Fu) - 8) : (static_cast<int>(byte >> 4) - 8);
        dst[sb * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
    }
}

__global__ void k_dequant_q8_a_to_f16(const BlockQ8_A* __restrict__ w, half* __restrict__ out, int N, int K) {
    int ns = K >> 8;
    int s_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int total_s = N * ns;
    if (s_idx >= total_s) return;
    const BlockQ8_A& blk = w[s_idx];
    half* dst = out + static_cast<int64_t>(s_idx) * 256;
    for (int sb = 0; sb < 8; ++sb) {
        float d = half_bits_to_f32(__ldg(&blk.d[sb]));
        int8_t q = reinterpret_cast<const int8_t*>(&blk.qs[sb * 8])[lane];
        dst[sb * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
    }
}

__global__ void k_dequant_q4_0_to_f16(const BlockQ4_0* __restrict__ w, half* __restrict__ out, int N, int K) {
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int total_b = N * (K >> 5);
    if (b >= total_b) return;
    const BlockQ4_0& blk = w[b];
    float d = half_bits_to_f32(__ldg(&blk.d));
    uint8_t byte = __ldg(&blk.qs[lane & 15]);
    int q = (lane < 16) ? (static_cast<int>(byte & 0x0Fu) - 8) : (static_cast<int>(byte >> 4) - 8);
    out[static_cast<int64_t>(b) * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
}

__global__ void k_dequant_q8_0_to_f16(const BlockQ8_0* __restrict__ w, half* __restrict__ out, int N, int K) {
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int total_b = N * (K >> 5);
    if (b >= total_b) return;
    const BlockQ8_0& blk = w[b];
    float d = half_bits_to_f32(__ldg(&blk.d));
    int8_t q = __ldg(&blk.qs[lane]);
    out[static_cast<int64_t>(b) * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
}

__global__ void k_dequant_q4_1_to_f16(const BlockQ4_1* __restrict__ w, half* __restrict__ out, int total_b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / 4, word = tid % 4;
    if (b >= total_b) return;
    const BlockQ4_1& blk = w[b];
    float d = half_bits_to_f32(__ldg(&blk.d));
    float m = half_bits_to_f32(__ldg(&blk.m));
    unsigned packed = __ldg(reinterpret_cast<const unsigned*>(blk.qs) + word);
    half2* dst = reinterpret_cast<half2*>(out + static_cast<int64_t>(b) * 32 + word * 4);
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        unsigned q = (packed >> (part * 4)) & kNibbleMask;
        dst[part * 8] = __floats2half2_rn(fmaf(d, q & 15u, m), fmaf(d, (q >> 8) & 15u, m));
        dst[part * 8 + 1] = __floats2half2_rn(fmaf(d, (q >> 16) & 15u, m), fmaf(d, q >> 24, m));
    }
}

template <int WK>
__global__ void k_dequant_generic_to_f16(const void* __restrict__ w, half* __restrict__ out, int N, int K) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * K;
    if (idx >= total) return;
    int n = static_cast<int>(idx / K);
    int k = static_cast<int>(idx % K);
    int k0 = k & ~31;
    int c = k & 31;
    out[idx] = __float2half_rn(dequant_weight_elem<WK>(w, n, k0, c, K));
}

__global__ void k_dequant_f32_or_bf16_to_f16(const void* __restrict__ w, int is_bf16, half* __restrict__ out, int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = is_bf16
        ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w)[idx])
        : reinterpret_cast<const float*>(w)[idx];
    out[idx] = __float2half_rn(v);
}

__global__ void k_rmsnorm_or_copy_to_f16(const float* __restrict__ x,
                                         const float* __restrict__ norm_w,
                                         half* __restrict__ out,
                                         int K, float eps) {
    __shared__ float smem[32];
    int m = blockIdx.x;
    int tid = threadIdx.x;
    const float* xm = x + static_cast<int64_t>(m) * K;
    half* ym = out + static_cast<int64_t>(m) * K;
    if (norm_w != nullptr) {
        float sum_sq = 0.0f;
        for (int c = tid; c < K; c += blockDim.x) {
            float v = __ldg(&xm[c]);
            sum_sq += v * v;
        }
        float total_sq = block_reduce_sum(sum_sq, smem);
        float inv = rsqrtf(total_sq / static_cast<float>(K) + eps);
        for (int c = tid; c < K; c += blockDim.x) {
            ym[c] = __float2half_rn(__ldg(&xm[c]) * inv * __ldg(&norm_w[c]));
        }
    } else {
        for (int c = tid; c < K; c += blockDim.x) {
            ym[c] = __float2half_rn(__ldg(&xm[c]));
        }
    }
}

template <int MAX_DK>
__global__ void k_fused_gated_deltanet_prefill_r1(
    const float* __restrict__ qkv_proj, int qkv_stride,
    const float* __restrict__ z, int z_stride,
    const float* __restrict__ a, int a_stride,
    const float* __restrict__ b, int b_stride,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w,
    float* __restrict__ conv_state,
    float* __restrict__ ssm_state,
    float* __restrict__ out,
    int M, int num_k_heads, int dk, float eps, float q_scale)
{
    extern __shared__ float smem[];
    float* red_buf = smem;       // [32]
    float* q_sh = red_buf + 32;  // [dk]
    float* k_sh = q_sh + dk;     // [dk]

    int kh = blockIdx.x;
    if (kh >= num_k_heads) return;
    int tid = threadIdx.x;
    int q_dim = num_k_heads * dk;
    int dv = dk;
    int q_ch = kh * dk + tid;
    int k_ch = q_dim + q_ch;
    int v_ch = 2 * q_dim + q_ch;

    float qc0 = 0.0f, qc1 = 0.0f, qc2 = 0.0f;
    float kc0 = 0.0f, kc1 = 0.0f, kc2 = 0.0f;
    float vc0 = 0.0f, vc1 = 0.0f, vc2 = 0.0f;
    float qcw0 = 0.0f, qcw1 = 0.0f, qcw2 = 0.0f, qcw3 = 0.0f, qcb = 0.0f;
    float kcw0 = 0.0f, kcw1 = 0.0f, kcw2 = 0.0f, kcw3 = 0.0f, kcb = 0.0f;
    float vcw0 = 0.0f, vcw1 = 0.0f, vcw2 = 0.0f, vcw3 = 0.0f, vcb = 0.0f;
    float s_reg[MAX_DK];
    float* S_vh = ssm_state + static_cast<int64_t>(kh) * dk * dv;
    float a_exp = expf(A_log[kh]);
    float dt_b = dt_bias[kh];
    float nw = 1.0f;

    if (tid < dk) {
        qc0 = conv_state[q_ch * 3 + 0]; qc1 = conv_state[q_ch * 3 + 1]; qc2 = conv_state[q_ch * 3 + 2];
        kc0 = conv_state[k_ch * 3 + 0]; kc1 = conv_state[k_ch * 3 + 1]; kc2 = conv_state[k_ch * 3 + 2];
        vc0 = conv_state[v_ch * 3 + 0]; vc1 = conv_state[v_ch * 3 + 1]; vc2 = conv_state[v_ch * 3 + 2];
        qcw0 = conv_w[q_ch * 4 + 0]; qcw1 = conv_w[q_ch * 4 + 1]; qcw2 = conv_w[q_ch * 4 + 2]; qcw3 = conv_w[q_ch * 4 + 3];
        kcw0 = conv_w[k_ch * 4 + 0]; kcw1 = conv_w[k_ch * 4 + 1]; kcw2 = conv_w[k_ch * 4 + 2]; kcw3 = conv_w[k_ch * 4 + 3];
        vcw0 = conv_w[v_ch * 4 + 0]; vcw1 = conv_w[v_ch * 4 + 1]; vcw2 = conv_w[v_ch * 4 + 2]; vcw3 = conv_w[v_ch * 4 + 3];
        if (conv_bias) {
            qcb = conv_bias[q_ch]; kcb = conv_bias[k_ch]; vcb = conv_bias[v_ch];
        }
        nw = norm_w[tid];
        for (int i = 0; i < dk; ++i) {
            s_reg[i] = S_vh[i * dv + tid];
        }
    }

    for (int m = 0; m < M; ++m) {
        const float* qp_m = qkv_proj + static_cast<int64_t>(m) * qkv_stride;
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        float v_cur_act = 0.0f;
        if (tid < dk) {
            float q_cur = qp_m[q_ch];
            float k_cur = qp_m[k_ch];
            float v_cur = qp_m[v_ch];
            float qv = silu_f(qc0 * qcw0 + qc1 * qcw1 + qc2 * qcw2 + q_cur * qcw3 + qcb);
            float kv = silu_f(kc0 * kcw0 + kc1 * kcw1 + kc2 * kcw2 + k_cur * kcw3 + kcb);
            v_cur_act = silu_f(vc0 * vcw0 + vc1 * vcw1 + vc2 * vcw2 + v_cur * vcw3 + vcb);
            qc0 = qc1; qc1 = qc2; qc2 = q_cur;
            kc0 = kc1; kc1 = kc2; kc2 = k_cur;
            vc0 = vc1; vc1 = vc2; vc2 = v_cur;
            q_sh[tid] = qv;
            k_sh[tid] = kv;
            q_sq = qv * qv;
            k_sq = kv * kv;
        }
        float q_inv = rsqrtf(block_reduce_sum(q_sq, red_buf) + 1e-6f);
        float k_inv = rsqrtf(block_reduce_sum(k_sq, red_buf) + 1e-6f);
        if (tid < dk) {
            q_sh[tid] *= q_inv * q_scale;
            k_sh[tid] *= k_inv;
        }
        __syncthreads();

        float y_j = 0.0f;
        if (tid < dk) {
            float g = expf(-a_exp * softplus_f(a[static_cast<int64_t>(m) * a_stride + kh] + dt_b));
            float beta = 1.0f / (1.0f + expf(-b[static_cast<int64_t>(m) * b_stride + kh]));
            float kv_mem = 0.0f;
            for (int i = 0; i < dk; ++i) {
                s_reg[i] *= g;
                kv_mem += s_reg[i] * k_sh[i];
            }
            float delta = (v_cur_act - kv_mem) * beta;
            for (int i = 0; i < dk; ++i) {
                s_reg[i] += k_sh[i] * delta;
                y_j += s_reg[i] * q_sh[i];
            }
        }
        float y_sq = (tid < dk) ? (y_j * y_j) : 0.0f;
        float y_inv = rsqrtf(block_reduce_sum(y_sq, red_buf) / static_cast<float>(dv) + eps);
        if (tid < dk) {
            float z_val = z[static_cast<int64_t>(m) * z_stride + q_ch];
            out[static_cast<int64_t>(m) * q_dim + q_ch] = (y_j * y_inv * nw) * silu_f(z_val);
        }
        __syncthreads();
    }

    if (tid < dk) {
        conv_state[q_ch * 3 + 0] = qc0; conv_state[q_ch * 3 + 1] = qc1; conv_state[q_ch * 3 + 2] = qc2;
        conv_state[k_ch * 3 + 0] = kc0; conv_state[k_ch * 3 + 1] = kc1; conv_state[k_ch * 3 + 2] = kc2;
        conv_state[v_ch * 3 + 0] = vc0; conv_state[v_ch * 3 + 1] = vc1; conv_state[v_ch * 3 + 2] = vc2;
        for (int i = 0; i < dk; ++i) {
            S_vh[i * dv + tid] = s_reg[i];
        }
    }
}

template <int MAX_DK>
__global__ void k_fused_gated_deltanet_prefill_generic(
    const float* __restrict__ qkv_proj, int qkv_stride,
    const float* __restrict__ z, int z_stride,
    const float* __restrict__ a, int a_stride,
    const float* __restrict__ b, int b_stride,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w,
    float* __restrict__ conv_state,
    float* __restrict__ ssm_state,
    float* __restrict__ out,
    int M, int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, bool tiled_value_heads)
{
    extern __shared__ float smem[];
    float* red_buf = smem;
    float* q_sh = red_buf + 32;
    float* k_sh = q_sh + dk;
    float* v_sh = k_sh + dk;

    int kh = blockIdx.x;
    if (kh >= num_k_heads) return;
    int tid = threadIdx.x;
    int ratio = num_v_heads / num_k_heads;
    int q_dim = num_k_heads * dk;
    int v_dim = num_v_heads * dv;

    for (int m = 0; m < M; ++m) {
        const float* qp_m = qkv_proj + static_cast<int64_t>(m) * qkv_stride;
        const float* zp_m = z + static_cast<int64_t>(m) * z_stride;
        const float* ap_m = a + static_cast<int64_t>(m) * a_stride;
        const float* bp_m = b + static_cast<int64_t>(m) * b_stride;
        float* out_m = out + static_cast<int64_t>(m) * v_dim;

        auto apply_conv_silu = [&](int ch) -> float {
            float* cs = conv_state + ch * 3;
            const float* cw = conv_w + ch * 4;
            float c0 = cs[0], c1 = cs[1], c2 = cs[2];
            float cur = qp_m[ch];
            float acc = c0 * cw[0] + c1 * cw[1] + c2 * cw[2] + cur * cw[3];
            if (conv_bias) acc += conv_bias[ch];
            cs[0] = c1; cs[1] = c2; cs[2] = cur;
            return silu_f(acc);
        };

        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (int i = tid; i < dk; i += blockDim.x) {
            float qv = apply_conv_silu(kh * dk + i);
            float kv = apply_conv_silu(q_dim + kh * dk + i);
            q_sh[i] = qv;
            k_sh[i] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
        }
        float q_inv = rsqrtf(block_reduce_sum(q_sq, red_buf) + 1e-6f);
        float k_inv = rsqrtf(block_reduce_sum(k_sq, red_buf) + 1e-6f);
        for (int i = tid; i < dk; i += blockDim.x) {
            q_sh[i] *= q_inv * q_scale;
            k_sh[i] *= k_inv;
        }
        __syncthreads();

        int v_offset = 2 * q_dim;
        for (int r = 0; r < ratio; ++r) {
            int vh = tiled_value_heads ? r * num_k_heads + kh : kh * ratio + r;
            if (tid < dv) {
                v_sh[tid] = apply_conv_silu(v_offset + vh * dv + tid);
            }
            __syncthreads();

            float g = expf(-expf(A_log[vh]) * softplus_f(ap_m[vh] + dt_bias[vh]));
            float beta = 1.0f / (1.0f + expf(-bp_m[vh]));
            float y_j = 0.0f;
            if (tid < dv) {
                float* S_vh = ssm_state + static_cast<int64_t>(vh) * dk * dv;
                float s_reg[MAX_DK];
                float kv_mem = 0.0f;
                for (int i = 0; i < dk; ++i) {
                    float s_val = S_vh[i * dv + tid] * g;
                    s_reg[i] = s_val;
                    kv_mem += s_val * k_sh[i];
                }
                float delta = (v_sh[tid] - kv_mem) * beta;
                for (int i = 0; i < dk; ++i) {
                    float s_new = s_reg[i] + k_sh[i] * delta;
                    S_vh[i * dv + tid] = s_new;
                    y_j += s_new * q_sh[i];
                }
            }
            float y_sq = (tid < dv) ? (y_j * y_j) : 0.0f;
            float y_inv = rsqrtf(block_reduce_sum(y_sq, red_buf) / static_cast<float>(dv) + eps);
            if (tid < dv) {
                out_m[vh * dv + tid] = (y_j * y_inv * norm_w[tid]) * silu_f(zp_m[vh * dv + tid]);
            }
            __syncthreads();
        }
    }
}

template <int DK>
__global__ void k_fused_gated_deltanet_step_tile(
    const float* __restrict__ qkv_proj,
    const float* __restrict__ z,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_w,
    float* __restrict__ conv_state,
    float* __restrict__ ssm_state,
    float* __restrict__ out,
    int num_k_heads, int num_v_heads, int dv,
    float eps, float q_scale, bool tiled_value_heads)
{
    extern __shared__ float smem[];
    float* red_buf = smem;               // [32]
    float* red_sh = red_buf + 32;        // [4 * 32]
    float* q_sh = red_sh + 128;          // [DK]
    float* k_sh = q_sh + DK;             // [DK]
    float* v_sh = k_sh + DK;             // [dv]
    float* y_sh = v_sh + dv;             // [dv]

    int kh = blockIdx.x;
    if (kh >= num_k_heads) return;
    int tid = threadIdx.x;  // 128 threads = 4 warps
    int wid = tid >> 5;
    int lane = tid & 31;
    int ratio = num_v_heads / num_k_heads;
    int q_dim = num_k_heads * DK;

    auto apply_conv_silu = [&](int ch) -> float {
        float* cs = conv_state + ch * 3;
        const float* cw = conv_w + ch * 4;
        float c0 = cs[0], c1 = cs[1], c2 = cs[2];
        float cur = qkv_proj[ch];
        float acc = c0 * cw[0] + c1 * cw[1] + c2 * cw[2] + cur * cw[3];
        if (conv_bias) acc += conv_bias[ch];
        cs[0] = c1; cs[1] = c2; cs[2] = cur;
        return silu_f(acc);
    };

    float q_sq = 0.0f, k_sq = 0.0f;
    for (int i = tid; i < DK; i += 128) {
        float qv = apply_conv_silu(kh * DK + i);
        float kv = apply_conv_silu(q_dim + kh * DK + i);
        q_sh[i] = qv;
        k_sh[i] = kv;
        q_sq += qv * qv;
        k_sq += kv * kv;
    }
    float q_inv = rsqrtf(block_reduce_sum(q_sq, red_buf) + 1e-6f);
    __syncthreads();
    float k_inv = rsqrtf(block_reduce_sum(k_sq, red_buf) + 1e-6f);
    for (int i = tid; i < DK; i += 128) {
        q_sh[i] *= q_inv * q_scale;
        k_sh[i] *= k_inv;
    }
    __syncthreads();

    int v_offset = 2 * q_dim;
    for (int r = 0; r < ratio; ++r) {
        int vh = tiled_value_heads ? r * num_k_heads + kh : kh * ratio + r;
        for (int i = tid; i < dv; i += 128) {
            v_sh[i] = apply_conv_silu(v_offset + vh * dv + i);
        }
        __syncthreads();

        float g = expf(-expf(A_log[vh]) * softplus_f(a[vh] + dt_bias[vh]));
        float beta = 1.0f / (1.0f + expf(-b[vh]));
        float* S_vh = ssm_state + static_cast<int64_t>(vh) * DK * dv;

        for (int jc = 0; jc < dv; jc += 32) {
            int j = jc + lane;
            constexpr int DK_CHUNK = DK / 4;
            float s_reg[DK_CHUNK];
            int i_base = wid * DK_CHUNK;
            float kv_part = 0.0f;
            #pragma unroll 8
            for (int ii = 0; ii < DK_CHUNK; ++ii) {
                int i = i_base + ii;
                float sv = S_vh[i * dv + j] * g;
                s_reg[ii] = sv;
                kv_part += sv * k_sh[i];
            }
            red_sh[wid * 32 + lane] = kv_part;
            __syncthreads();
            float kv_mem = red_sh[lane] + red_sh[32 + lane] + red_sh[64 + lane] + red_sh[96 + lane];
            float delta = (v_sh[j] - kv_mem) * beta;
            float y_part = 0.0f;
            #pragma unroll 8
            for (int ii = 0; ii < DK_CHUNK; ++ii) {
                int i = i_base + ii;
                float s_new = s_reg[ii] + k_sh[i] * delta;
                S_vh[i * dv + j] = s_new;
                y_part += s_new * q_sh[i];
            }
            red_sh[wid * 32 + lane] = y_part;
            __syncthreads();
            if (wid == 0) {
                y_sh[j] = red_sh[lane] + red_sh[32 + lane] + red_sh[64 + lane] + red_sh[96 + lane];
            }
            __syncthreads();
        }

        float y_sq = 0.0f;
        for (int j = tid; j < dv; j += 128) y_sq += y_sh[j] * y_sh[j];
        float y_inv = rsqrtf(block_reduce_sum(y_sq, red_buf) / static_cast<float>(dv) + eps);
        for (int j = tid; j < dv; j += 128) {
            out[vh * dv + j] = (y_sh[j] * y_inv * norm_w[j]) * silu_f(z[vh * dv + j]);
        }
        __syncthreads();
    }
}

__global__ void k_gdn_prefill_conv_silu(
    const float* __restrict__ qkv_proj, int qkv_stride,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_bias,
    float* __restrict__ conv_state,
    float* __restrict__ qkv_act,
    int M, int conv_dim)
{
    constexpr int Warps = 4;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int ch = min(static_cast<int>(blockIdx.x) * 32 + lane, conv_dim - 1);
    float* cs = conv_state + ch * 3;
    const float* cw = conv_w + ch * 4;
    int first = (M * warp) / Warps, end = (M * (warp + 1)) / Warps;
    auto prev = [&](int t) {
        return t < 0 ? cs[t + 3] : qkv_proj[static_cast<int64_t>(t) * qkv_stride + ch];
    };
    float c0 = prev(first - 3), c1 = prev(first - 2), c2 = prev(first - 1);
    __syncthreads();
    if (blockIdx.x * 32 + lane >= conv_dim) return;
    float w0 = cw[0], w1 = cw[1], w2 = cw[2], w3 = cw[3];
    float cb = conv_bias ? conv_bias[ch] : 0.0f;
    for (int m = first; m < end; ++m) {
        float cur = qkv_proj[static_cast<int64_t>(m) * qkv_stride + ch];
        qkv_act[static_cast<int64_t>(m) * conv_dim + ch] = silu_f(c0 * w0 + c1 * w1 + c2 * w2 + cur * w3 + cb);
        c0 = c1; c1 = c2; c2 = cur;
    }
    if (warp == Warps - 1) {
        cs[0] = c0; cs[1] = c1; cs[2] = c2;
    }
}

__global__ void k_gdn_prefill_l2_norm(
    float* __restrict__ qkv_act, int conv_dim, int num_k_heads, int dk, float q_scale)
{
    __shared__ float smem[32];
    int kh = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;
    float* row = qkv_act + static_cast<int64_t>(m) * conv_dim;
    float* qp = row + kh * dk;
    float* kp = row + (num_k_heads + kh) * dk;
    float q_sq = (tid < dk) ? (qp[tid] * qp[tid]) : 0.0f;
    float k_sq = (tid < dk) ? (kp[tid] * kp[tid]) : 0.0f;
    float q_inv = rsqrtf(block_reduce_sum(q_sq, smem) + 1e-6f) * q_scale;
    __syncthreads();
    float k_inv = rsqrtf(block_reduce_sum(k_sq, smem) + 1e-6f);
    if (tid < dk) {
        qp[tid] *= q_inv;
        kp[tid] *= k_inv;
    }
}

// Within a chunk, S[t] = D[t] S_prev + sum_j exp(logD[t] - logD[j]) k[j]^T delta[j].
// Substitution into the delta rule yields a unit lower-triangular system.
constexpr int kGdnChunk = 32;
constexpr int kGdnCoeffStride = 2 * kGdnChunk * kGdnChunk + 2 * kGdnChunk;

__device__ __forceinline__ unsigned round_tf32(float x) {
    unsigned q;
    asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(q) : "f"(x));
    return q;
}

__device__ __forceinline__ unsigned residual_tf32(float x, unsigned hi) {
    return round_tf32(x - __uint_as_float(hi));
}

__device__ __forceinline__ void mma_tf32(uint4 a, uint2 b, float4& c) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

__device__ __forceinline__ void mma_gdn_f32(float4 a, float2 b, float4& main, float4& correction) {
    uint4 ah = make_uint4(round_tf32(a.x), round_tf32(a.y), round_tf32(a.z), round_tf32(a.w));
    uint2 bh = make_uint2(round_tf32(b.x), round_tf32(b.y));
    uint4 al = make_uint4(residual_tf32(a.x, ah.x), residual_tf32(a.y, ah.y),
                          residual_tf32(a.z, ah.z), residual_tf32(a.w, ah.w));
    uint2 bl = make_uint2(residual_tf32(b.x, bh.x), residual_tf32(b.y, bh.y));
    // Keep both first-order rounding corrections to preserve FP32 state accuracy.
    mma_tf32(ah, bh, main);
    mma_tf32(al, bh, correction);
    mma_tf32(ah, bl, correction);
}

template <int DK>
__global__ void k_gdn_chunk_coefficients(
    const float* __restrict__ qkv, int conv_dim,
    const float* __restrict__ a, int a_stride,
    const float* __restrict__ b, int b_stride,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    float* __restrict__ coefficients, int M, int nk, int nv, bool tiled_value_heads)
{
    constexpr int C = kGdnChunk;
    __shared__ float keys[C][DK + 4], queries[C][DK + 4];
    __shared__ float lower[C][C + 4], inverse[C][C + 1];
    __shared__ float decay[C], rate[C];
    int vh = blockIdx.x, kh = tiled_value_heads ? vh % nk : vh / (nv / nk), base = blockIdx.y * C;
    int count = min(C, M - base);
    int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    float* coeff = coefficients + (static_cast<int64_t>(blockIdx.y) * nv + vh) * kGdnCoeffStride;
    float* attention = coeff + C * C;
    float* before = attention + C * C;
    float* after = before + C;

    for (int i = tid; i < C * DK; i += blockDim.x) {
        int row = i / DK, d = i % DK;
        float qt = 0.0f, kt = 0.0f;
        if (row < count) {
            const float* q = qkv + static_cast<int64_t>(base + row) * conv_dim + kh * DK + d;
            qt = q[0];
            kt = q[nk * DK];
        }
        queries[row][d] = qt;
        keys[row][d] = kt;
    }
    if (warp == 0) {
        float g = 0.0f;
        if (lane < count) {
            int m = base + lane;
            g = -expf(A_log[vh]) * softplus_f(a[static_cast<int64_t>(m) * a_stride + vh] + dt_bias[vh]);
            rate[lane] = 1.0f / (1.0f + expf(-b[static_cast<int64_t>(m) * b_stride + vh]));
        }
        for (int step = 1; step < C; step *= 2) {
            float prev = __shfl_up_sync(0xffffffffu, g, step);
            if (lane >= step) g += prev;
        }
        decay[lane] = g;
        float last = __shfl_sync(0xffffffffu, g, count - 1);
        if (lane < count) {
            before[lane] = expf(g);
            after[lane] = expf(last - g);
        }
    }
    __syncthreads();

    int row_base = (warp / 4) * 16, col_base = (warp % 4) * 8;
    if (row_base < count && col_base < count && col_base < row_base + 16) {
        int group = lane / 4, word = lane % 4;
        int row = row_base + group;
        float4 kk = {}, kk_error = {}, qk = {}, qk_error = {};
        #pragma unroll 2
        for (int d = 0; d < DK; d += 8) {
            int k = d + word;
            float4 kr = make_float4(keys[row][k], keys[row + 8][k], keys[row][k + 4], keys[row + 8][k + 4]);
            float4 qr = make_float4(queries[row][k], queries[row + 8][k], queries[row][k + 4], queries[row + 8][k + 4]);
            float2 kc = make_float2(keys[col_base + group][k], keys[col_base + group][k + 4]);
            mma_gdn_f32(kr, kc, kk, kk_error);
            mma_gdn_f32(qr, kc, qk, qk_error);
        }
        auto write = [&](int i, int j, float dot_kk, float dot_qk) {
            if (i < count && j <= i) {
                float g = expf(decay[i] - decay[j]);
                lower[i][j] = rate[i] * g * dot_kk;
                attention[i * C + j] = g * dot_qk;
            }
        };
        int col = col_base + 2 * word;
        write(row, col, kk.x + kk_error.x, qk.x + qk_error.x);
        write(row, col + 1, kk.y + kk_error.y, qk.y + qk_error.y);
        write(row + 8, col, kk.z + kk_error.z, qk.z + qk_error.z);
        write(row + 8, col + 1, kk.w + kk_error.w, qk.w + qk_error.w);
    }
    __syncthreads();

    // Invert the unit lower-triangular system, one column per lane. Multiplying
    // by beta here gives (I + L)^-1 diag(beta) for all value columns.
    if (warp == 0) {
        for (int row = 0; row < count; ++row) {
            if (lane <= row) {
                float t = lane == row ? rate[row] : 0.0f;
                for (int j = lane; j < row; ++j) t = fmaf(-lower[row][j], inverse[j][lane], t);
                inverse[row][lane] = t;
                coeff[row * C + lane] = t;
            }
            __syncwarp();
        }
    }
}

template <int DK>
__global__ void k_gdn_chunkwise_ssm(
    const float* __restrict__ qkv, int conv_dim,
    const float* __restrict__ coefficients, float* __restrict__ ssm_state,
    float* __restrict__ y_raw, int M, int nk, int nv, int dv, bool tiled_value_heads)
{
    constexpr int C = kGdnChunk, V = 16, Threads = 256;
    __shared__ float state[DK][V];
    __shared__ float keys[C][DK + 4], queries[C][DK + 4];
    __shared__ float values[C][V], delta[C][V], output[C][V];
    int vh = blockIdx.x, kh = tiled_value_heads ? vh % nk : vh / (nv / nk);
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int word = lane % 4, group = lane / 4;
    int row_base = (warp / 2 % 2) * 16, row = row_base + group;
    int tile_v = (warp % 2) * 8;
    int vbase = blockIdx.y * V;
    float* state_head = ssm_state + static_cast<int64_t>(vh) * DK * dv + vbase;
    int v_dim = nv * dv;
    // Swizzle the eight-column halves so the four K rows of a fragment use distinct banks.
    auto state_at = [&](int d, int j) -> float& { return state[d][j ^ ((d & 2) * 4)]; };
    bool nonzero = false;
    for (int i = tid; i < DK * V; i += Threads) {
        float s = state_head[(i / V) * dv + i % V];
        nonzero |= s != 0.0f;
        state_at(i / V, i % V) = s;
    }
    bool has_state = __syncthreads_or(nonzero);

    for (int base = 0; base < M; base += C) {
        int count = min(C, M - base);
        const float* coeff = coefficients + (static_cast<int64_t>(base / C) * nv + vh) * kGdnCoeffStride;
        const float* attention = coeff + C * C;
        const float* before = attention + C * C;
        const float* after = before + C;
        for (int i = tid; i < C * DK; i += Threads) {
            int t = i / DK, d = i % DK;
            float qt = 0.0f, kt = 0.0f;
            if (t < count) {
                const float* q = qkv + static_cast<int64_t>(base + t) * conv_dim + kh * DK + d;
                if (has_state) qt = q[0];
                kt = q[nk * DK];
            }
            keys[t][d] = kt;
            queries[t][d] = qt;
        }
        __syncthreads();

        // Each warp projects S_prev onto 16 queries or keys.
        if (has_state && row_base < count) {
            auto* rows = warp < 4 ? queries : keys;
            float4 main = {}, correction = {};
            #pragma unroll 2
            for (int d = 0; d < DK; d += 8) {
                int k = d + word;
                float4 a = make_float4(rows[row][k], rows[row + 8][k], rows[row][k + 4], rows[row + 8][k + 4]);
                float2 b = make_float2(state_at(k, tile_v + group), state_at(k + 4, tile_v + group));
                mma_gdn_f32(a, b, main, correction);
            }
            auto store_projection = [&](int t, int j, float result) {
                if (t >= count) return;
                float g = before[t];
                if (warp < 4) {
                    output[t][j] = g * result;
                } else {
                    const float* v = qkv + static_cast<int64_t>(base + t) * conv_dim + 2 * nk * DK + vh * dv + vbase;
                    values[t][j] = fmaf(-g, result, v[j]);
                }
            };
            store_projection(row, tile_v + 2 * word, main.x + correction.x);
            store_projection(row, tile_v + 2 * word + 1, main.y + correction.y);
            store_projection(row + 8, tile_v + 2 * word, main.z + correction.z);
            store_projection(row + 8, tile_v + 2 * word + 1, main.w + correction.w);
        }
        if (!has_state) {
            for (int i = tid; i < count * V; i += Threads) {
                int t = i / V, j = i % V;
                const float* v = qkv + static_cast<int64_t>(base + t) * conv_dim + 2 * nk * DK + vh * dv + vbase;
                values[t][j] = v[j];
                output[t][j] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < C * V / Threads; ++n) {
            int i = tid + n * Threads, t = i / V, j = i % V;
            float d = 0.0f;
            if (t < count) {
                for (int k = 0; k <= t; ++k) d = fmaf(coeff[t * C + k], values[k][j], d);
            }
            delta[t][j] = d;
        }
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < C * V / Threads; ++n) {
            int i = tid + n * Threads, t = i / V, j = i % V;
            float d = 0.0f;
            if (t < count) {
                float y = output[t][j];
                for (int k = 0; k <= t; ++k) y = fmaf(attention[t * C + k], delta[k][j], y);
                y_raw[static_cast<int64_t>(base + t) * v_dim + vh * dv + vbase + j] = y;
                d = after[t] * delta[t][j];
            }
            values[t][j] = d;
        }
        __syncthreads();

        // Contract the keys with the decay-weighted deltas to carry state forward.
        float g = before[count - 1];
        #pragma unroll
        for (int tile = 0; tile < DK / 16 / (Threads / 64); ++tile) {
            int d = (warp / 2 + tile * (Threads / 64)) * 16 + group;
            float4 updated = {}, error = {};
            #pragma unroll
            for (int k = 0; k < C; k += 8) {
                if (k >= count) break;
                int t = k + word;
                float4 a = make_float4(keys[t][d], keys[t][d + 8], keys[t + 4][d], keys[t + 4][d + 8]);
                float2 b = make_float2(values[t][tile_v + group], values[t + 4][tile_v + group]);
                mma_gdn_f32(a, b, updated, error);
            }
            int j = tile_v + 2 * word;
            state_at(d, j) = fmaf(g, state_at(d, j), updated.x + error.x);
            state_at(d, j + 1) = fmaf(g, state_at(d, j + 1), updated.y + error.y);
            state_at(d + 8, j) = fmaf(g, state_at(d + 8, j), updated.z + error.z);
            state_at(d + 8, j + 1) = fmaf(g, state_at(d + 8, j + 1), updated.w + error.w);
        }
        __syncthreads();
        has_state = true;
    }
    for (int i = tid; i < DK * V; i += Threads) state_head[(i / V) * dv + i % V] = state_at(i / V, i % V);
}

__global__ void k_gdn_prefill_gated_rmsnorm(
    const float* __restrict__ y_raw,
    const float* __restrict__ z, int z_stride,
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    int num_v_heads, int dv, float eps)
{
    __shared__ float smem[32];
    int vh = blockIdx.x;
    int m = blockIdx.y;
    int j = threadIdx.x;
    int v_dim = num_v_heads * dv;
    const float* yh = y_raw + static_cast<int64_t>(m) * v_dim + vh * dv;
    const float* zh = z + static_cast<int64_t>(m) * z_stride + vh * dv;
    float* oh = out + static_cast<int64_t>(m) * v_dim + vh * dv;
    float v = (j < dv) ? yh[j] : 0.0f;
    float inv = rsqrtf(block_reduce_sum(v * v, smem) / static_cast<float>(dv) + eps);
    if (j < dv) {
        oh[j] = (v * inv * norm_w[j]) * silu_f(zh[j]);
    }
}

__global__ void k_add_rmsnorm_f32(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  const float* __restrict__ norm_w,
                                  float* __restrict__ res_out,
                                  float* __restrict__ norm_out,
                                  int cols, float eps) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + static_cast<int64_t>(row) * cols;
    const float* br = b + static_cast<int64_t>(row) * cols;
    float* rr = res_out ? (res_out + static_cast<int64_t>(row) * cols) : nullptr;
    float* yr = norm_out + static_cast<int64_t>(row) * cols;

    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = __ldg(&ar[c]) + __ldg(&br[c]);
        if (rr) rr[c] = v;
        sum_sq += v * v;
    }
    float total_sq = block_reduce_sum(sum_sq, smem);
    float inv = rsqrtf(total_sq / static_cast<float>(cols) + eps);
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = rr ? rr[c] : (__ldg(&ar[c]) + __ldg(&br[c]));
        yr[c] = v * inv * __ldg(&norm_w[c]);
    }
}

}  // namespace

// =============================================================================
// Host Launch Wrappers
// =============================================================================

void rmsnorm_f32(const float* x, const float* weight, float* y,
                 int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((cols + 31) / 32) * 32));
    k_rmsnorm_f32<<<rows, threads, 0, stream>>>(x, weight, y, cols, eps);
}

void rmsnorm_f16(const uint16_t* x, const uint16_t* weight, uint16_t* y,
                 int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((cols + 31) / 32) * 32));
    k_rmsnorm_f16<<<rows, threads, 0, stream>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<half*>(y),
        cols, eps);
}

void linear_f32(const float* x, const float* w, float* y,
                int M, int N, int K, cudaStream_t stream) {
    if (M <= 8) {
        constexpr int WARPS = 8;
        int blocks = (N + WARPS - 1) / WARPS;
        k_linear_gemv_f32<<<blocks, WARPS * 32, 0, stream>>>(x, w, y, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        k_linear_gemm_tiled_f32<<<grid, block, 0, stream>>>(x, w, y, M, N, K);
    }
}

void linear_f16(const uint16_t* x, const uint16_t* w, uint16_t* y,
                int M, int N, int K, cudaStream_t stream) {
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    k_linear_gemv_f16<<<blocks, WARPS * 32, 0, stream>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(w),
        reinterpret_cast<half*>(y),
        M, N, K);
}

void linear_q8_0_unfused(const float* x, const void* w_q8, float* y,
                         int M, int N, int K, void* scratch_f32_w, cudaStream_t stream) {
    int nb = K >> 5;
    dim3 block(128, 1);
    dim3 grid((nb + 127) / 128, N);
    float* w_f32 = static_cast<float*>(scratch_f32_w);
    k_dequant_q8_0_to_f32<<<grid, block, 0, stream>>>(static_cast<const BlockQ8_0*>(w_q8), w_f32, N, K);
    linear_f32(x, w_f32, y, M, N, K, stream);
}

void linear_q4_0_unfused(const float* x, const void* w_q4, float* y,
                         int M, int N, int K, void* scratch_f32_w, cudaStream_t stream) {
    int nb = K >> 5;
    dim3 block(128, 1);
    dim3 grid((nb + 127) / 128, N);
    float* w_f32 = static_cast<float*>(scratch_f32_w);
    k_dequant_q4_0_to_f32<<<grid, block, 0, stream>>>(static_cast<const BlockQ4_0*>(w_q4), w_f32, N, K);
    linear_f32(x, w_f32, y, M, N, K, stream);
}

void rope_f32(float* x, int num_heads, int head_dim, int rot_dim, int pos,
              float theta, bool neox, cudaStream_t stream) {
    int half_rot = rot_dim >> 1;
    if (half_rot <= 0) return;
    k_rope_f32<<<num_heads, half_rot, 0, stream>>>(x, num_heads, head_dim, rot_dim, pos, theta, neox);
}

void kv_append_f32(const float* k_in, const float* v_in,
                   float* k_cache, float* v_cache,
                   int num_kv_heads, int head_dim, int pos, int /*max_seq*/,
                   cudaStream_t stream) {
    int total = num_kv_heads * head_dim;
    int blocks = (total + 255) / 256;
    k_kv_append_f32<<<blocks, 256, 0, stream>>>(k_in, v_in, k_cache, v_cache, num_kv_heads, head_dim, pos);
}

void swiglu_f32(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    k_swiglu_f32<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

void swiglu_interleaved_f32(const float* gate_up, float* out, int rows, int inter_dim, cudaStream_t stream) {
    int total = rows * inter_dim;
    k_swiglu_interleaved_f32<<<(total + 255) / 256, 256, 0, stream>>>(gate_up, out, rows, inter_dim);
}

void quantize_x_q8_1(const float* x, const float* res_b, const float* norm_w, float* res_out,
                     void* q8_out, int M, int K, float eps, cudaStream_t stream) {
    auto* out = static_cast<BlockQ8_1*>(q8_out);
    if (M == 1 && !res_b && !norm_w && !res_out && K >= 1024) {
        k_quantize_vector_q8_1<<<(K + 127) / 128, 128, 0, stream>>>(x, out, K);
        return;
    }
    const auto alignment = reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(res_b) |
                           reinterpret_cast<uintptr_t>(norm_w) | reinterpret_cast<uintptr_t>(res_out);
    if (M == 1 && K == 4096 && norm_w && (alignment & 15) == 0) {
        const auto* x4 = reinterpret_cast<const float4*>(x);
        const auto* n4 = reinterpret_cast<const float4*>(norm_w);
        auto* r4 = reinterpret_cast<float4*>(res_out);
        if (res_b) k_quantize_x_q8_1_decode<4096, true><<<1, 256, 0, stream>>>(
            x4, reinterpret_cast<const float4*>(res_b), n4, r4, out, eps);
        else k_quantize_x_q8_1_decode<4096, false><<<1, 256, 0, stream>>>(x4, nullptr, n4, r4, out, eps);
        return;
    }
    if (K == 1024) k_quantize_x_q8_1<1024><<<M, 256, 0, stream>>>(x, res_b, norm_w, res_out, out, K, eps);
    else if (K == 2048) k_quantize_x_q8_1<2048><<<M, 256, 0, stream>>>(x, res_b, norm_w, res_out, out, K, eps);
    else k_quantize_x_q8_1<<<M, 256, 0, stream>>>(x, res_b, norm_w, res_out, out, K, eps);
}

void swiglu_interleaved_q8_1(const float* gate_up, void* q8_out, int M, int inter_dim, cudaStream_t stream) {
    constexpr int threads = 128;
    k_swiglu_interleaved_q8_1<<<dim3((inter_dim + threads - 1) / threads, M), threads, 0, stream>>>(
        gate_up, static_cast<BlockQ8_1*>(q8_out), inter_dim);
}

void gemv_q4_a_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                    GemvGeometry geometry) {
    if (K <= 1024 || N <= 1024) launch_gemv_q8_1<BlockQ4_A, 1, 8>(x, w, y, N, K, stream);
    else launch_gemv_q8_1<BlockQ4_A, 8>(x, w, y, N, K, stream, geometry);
}

void gemv_q8_a_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                    GemvGeometry geometry) {
    launch_gemv_q8_1<BlockQ8_A>(x, w, y, N, K, stream, geometry);
}

void gemv_q6_k_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                    GemvGeometry geometry) {
    // The narrow Qwen3 vocabulary projection benefits from more warps per CTA.
    if (N >= 100000 && K <= 1024) launch_gemv_q8_1<BlockQ6_K, 1, 8>(x, w, y, N, K, stream);
    else launch_gemv_q8_1<BlockQ6_K>(x, w, y, N, K, stream, geometry);
}

void gemv_q4_k_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                    GemvGeometry geometry) {
    launch_gemv_q8_1<BlockQ4_K>(x, w, y, N, K, stream, geometry);
}

void gemv_q4_ka_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                     GemvGeometry geometry) {
    // Fewer rows per warp expose more parallel loads on wide projections.
    if (N >= 4096 && N < 100000 && K >= 4096) geometry.rows_per_warp = 1;
    launch_gemv_q8_1<BlockQ4_KA>(x, w, y, N, K, stream, geometry);
}

void gemv_q6_a_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                     GemvGeometry geometry) {
    // An extra output row per warp costs HBM throughput on large packed Q6 matrices.
    if ((N >= 4096 && K >= 4096) || N >= 100000) geometry.rows_per_warp = 1;
    launch_gemv_q8_1<BlockQ6_A>(x, w, y, N, K, stream, geometry);
}

void gemv_q5_k_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                    GemvGeometry geometry) {
    launch_gemv_q8_1<BlockQ5_K, 8>(x, w, y, N, K, stream, geometry);
}

bool project_q4_a_and_q8_a(const void* x, const void* q4, const void* q8, float* y4, float* y8,
                          int M, int N4, int N8, int K, cudaStream_t stream, GemvGeometry geometry) {
    const auto* xq = static_cast<const BlockQ8_1*>(x);
    const auto* w4 = static_cast<const BlockQ4_A*>(q4);
    const auto* w8 = static_cast<const BlockQ8_A*>(q8);
    if (M == 1 && K > 1024) {
        auto launch = [&]<int Rows, int Warps>() {
            constexpr int tile = Rows * Warps;
            const int q4_blocks = (N4 + tile - 1) / tile;
            const int blocks = q4_blocks + (N8 + tile - 1) / tile;
            k_gemv_q4_a_and_q8_a<Rows, Warps><<<blocks, Warps * 32, 0, stream>>>(
                xq, w4, w8, y4, y8, N4, N8, K, q4_blocks);
        };
        if (geometry.warps == 4) {
            if (geometry.rows_per_warp == 1) launch.template operator()<1, 4>();
            else launch.template operator()<2, 4>();
        } else {
            if (geometry.rows_per_warp == 1) launch.template operator()<1, 8>();
            else launch.template operator()<2, 8>();
        }
        return true;
    }
    if (M < 64 || N4 < 4096 || N8 > 64) return false;
    const int q4_tiles = (N4 + 127) / 128;
    const int q4_blocks = q4_tiles * ((M + 63) / 64);
    k_mmq_q4_a_and_q8_a<<<q4_blocks + (M + 15) / 16, 256, 0, stream>>>(
        xq, w4, w8, y4, y8, M, N4, N8, K, q4_tiles, q4_blocks);
    return true;
}

void mmq_q4_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ4_A>(x, w, y, M, N, K, stream);
}

void mmq_q8_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ8_A>(x, w, y, M, N, K, stream);
}

void mmq_q6_k_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ6_K>(x, w, y, M, N, K, stream);
}

void mmq_q4_k_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ4_K>(x, w, y, M, N, K, stream);
}

void mmq_q4_ka_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ4_KA>(x, w, y, M, N, K, stream);
}

void mmq_q6_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ6_A>(x, w, y, M, N, K, stream);
}

void mmq_q5_k_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    launch_mmq<BlockQ5_K>(x, w, y, M, N, K, stream);
}

__global__ void k_add_bias_f32(float* out, const float* bias, int rows, int cols, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[static_cast<size_t>(i / cols) * stride + i % cols] += bias[i % cols];
}

void add_bias_f32(float* out, const float* bias, int rows, int cols, int stride, cudaStream_t stream) {
    k_add_bias_f32<<<(rows * cols + 255) / 256, 256, 0, stream>>>(out, bias, rows, cols, stride);
}

void add_residual_f32(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    k_add_residual_f32<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
}

void argmax_f32(const float* x, int n, int32_t* out_idx, cudaStream_t stream,
                 void* scratch, int* pos, int* kv_len, int history_size) {
    if (scratch) {
        auto* values = static_cast<float*>(scratch);
        auto* indices = reinterpret_cast<int32_t*>(values + kArgmaxParts);
        k_argmax_f32<<<kArgmaxParts, 128, 0, stream>>>(x, n, indices, values, nullptr, nullptr, nullptr, 0);
        k_argmax_f32<<<1, kArgmaxParts, 0, stream>>>(values, kArgmaxParts, out_idx, nullptr, indices, pos, kv_len, history_size);
    } else {
        k_argmax_f32<<<1, 256, 0, stream>>>(x, n, out_idx, nullptr, nullptr, pos, kv_len, history_size);
    }
}

void fused_rmsnorm_linear_f32(const float* x, const float* norm_w, const float* weight,
                              float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_fused_rmsnorm_linear_f32<WARPS><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, weight,
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_q8_0(const float* x, const float* norm_w, const void* weight_q8,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<2, true><<<grid, block, 0, stream>>>(x, norm_w, weight_q8, y, M, N, K, eps);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q8_0<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ8_0*>(weight_q8),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_q4_0(const float* x, const float* norm_w, const void* weight_q4,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<3, true><<<grid, block, 0, stream>>>(x, norm_w, weight_q4, y, M, N, K, eps);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_0<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ4_0*>(weight_q4),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_q4_1(const float* x, const float* norm_w, const void* weight_q4_1,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<4, true><<<grid, block, 0, stream>>>(x, norm_w, weight_q4_1, y, M, N, K, eps);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_1<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ4_1*>(weight_q4_1),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_q4_k(const float* x, const float* norm_w, const void* weight_q4_k,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 255) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<5, true><<<grid, block, 0, stream>>>(x, norm_w, weight_q4_k, y, M, N, K, eps);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_q5_k<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ4_K*>(weight_q4_k),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_q6_k(const float* x, const float* norm_w, const void* weight_q6_k,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 255) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<6, true><<<grid, block, 0, stream>>>(x, norm_w, weight_q6_k, y, M, N, K, eps);
        return;
    }
    if (M == 1 && K <= 8192 && (K & 255) == 0) {
        size_t smem = static_cast<size_t>(K) * sizeof(float);
        if (N >= 16384) {
            constexpr int WARPS = 16;
            int blocks = (N + WARPS - 1) / WARPS;
            k_quant_gemv_q6_k_m1<WARPS, true, false><<<blocks, WARPS * 32, smem, stream>>>(
                x, nullptr, norm_w, static_cast<const BlockQ6_K*>(weight_q6_k), nullptr, y, N, K, eps);
        } else {
            constexpr int WARPS = 8;
            int blocks = (N + WARPS - 1) / WARPS;
            k_quant_gemv_q6_k_m1<WARPS, true, false><<<blocks, WARPS * 32, smem, stream>>>(
                x, nullptr, norm_w, static_cast<const BlockQ6_K*>(weight_q6_k), nullptr, y, N, K, eps);
        }
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q6_k<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ6_K*>(weight_q6_k),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

template <bool HAS_NORM, bool HAS_RES>
static void launch_dp4a_q4_a(const float* x, const float* res_b, const float* norm_w,
                             const BlockQ4_A* w, float* res_out, float* y,
                             int N, int K, float eps, cudaStream_t stream) {
    size_t smem = static_cast<size_t>(K >> 5) * (2 * sizeof(float) + 9 * sizeof(int32_t));
    if (N <= 512) {
        constexpr int WARPS = 4;
        k_gemv_q4_a_dp4a<WARPS, HAS_NORM, HAS_RES><<<(N + WARPS - 1) / WARPS, WARPS * 32, smem, stream>>>(
            x, res_b, norm_w, w, res_out, y, N, K, eps);
    } else {
        constexpr int WARPS = 8;
        k_gemv_q4_a_dp4a<WARPS, HAS_NORM, HAS_RES><<<(N + WARPS - 1) / WARPS, WARPS * 32, smem, stream>>>(
            x, res_b, norm_w, w, res_out, y, N, K, eps);
    }
}

template <bool HAS_NORM, bool HAS_RES>
static void launch_dp4a_q8_a(const float* x, const float* res_b, const float* norm_w,
                             const BlockQ8_A* w, float* res_out, float* y,
                             int N, int K, float eps, cudaStream_t stream) {
    size_t smem = static_cast<size_t>(K >> 5) * (sizeof(float) + 9 * sizeof(int32_t));
    if (N <= 512) {
        constexpr int WARPS = 4;
        k_gemv_q8_a_dp4a<WARPS, HAS_NORM, HAS_RES><<<(N + WARPS - 1) / WARPS, WARPS * 32, smem, stream>>>(
            x, res_b, norm_w, w, res_out, y, N, K, eps);
    } else {
        constexpr int WARPS = 8;
        k_gemv_q8_a_dp4a<WARPS, HAS_NORM, HAS_RES><<<(N + WARPS - 1) / WARPS, WARPS * 32, smem, stream>>>(
            x, res_b, norm_w, w, res_out, y, N, K, eps);
    }
}

void fused_rmsnorm_linear_q4_a(const float* x, const float* norm_w, const void* weight_q4_a,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M == 1 && K <= 8192 && (K & 255) == 0) {
        launch_dp4a_q4_a<true, false>(x, nullptr, norm_w, static_cast<const BlockQ4_A*>(weight_q4_a), nullptr, y, N, K, eps, stream);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_a_f32<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ4_A*>(weight_q4_a),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_q8_a(const float* x, const float* norm_w, const void* weight_q8_a,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M == 1 && K <= 8192 && (K & 255) == 0) {
        launch_dp4a_q8_a<true, false>(x, nullptr, norm_w, static_cast<const BlockQ8_A*>(weight_q8_a), nullptr, y, N, K, eps, stream);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q8_a_f32<WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, static_cast<const BlockQ8_A*>(weight_q8_a),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void add_rmsnorm_f32(const float* a, const float* b, const float* norm_w,
                     float* res_out, float* norm_out, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((cols + 31) / 32) * 32));
    k_add_rmsnorm_f32<<<rows, threads, 0, stream>>>(a, b, norm_w, res_out, norm_out, cols, eps);
}

void fused_add_rmsnorm_linear_q4_a(const float* a, const float* b, const float* norm_w,
                                   const void* weight_q4_a, float* res_out, float* y,
                                   int N, int K, float eps, cudaStream_t stream) {
    launch_dp4a_q4_a<true, true>(a, b, norm_w, static_cast<const BlockQ4_A*>(weight_q4_a), res_out, y, N, K, eps, stream);
}

void fused_add_rmsnorm_linear_q8_a(const float* a, const float* b, const float* norm_w,
                                   const void* weight_q8_a, float* res_out, float* y,
                                   int N, int K, float eps, cudaStream_t stream) {
    launch_dp4a_q8_a<true, true>(a, b, norm_w, static_cast<const BlockQ8_A*>(weight_q8_a), res_out, y, N, K, eps, stream);
}

void fused_add_rmsnorm_linear_q6_k(const float* a, const float* b, const float* norm_w,
                                   const void* weight_q6_k, float* res_out, float* y,
                                   int N, int K, float eps, cudaStream_t stream) {
    size_t smem = static_cast<size_t>(K) * sizeof(float);
    if (N >= 16384) {
        constexpr int WARPS = 16;
        k_quant_gemv_q6_k_m1<WARPS, true, true><<<(N + WARPS - 1) / WARPS, WARPS * 32, smem, stream>>>(
            a, b, norm_w, static_cast<const BlockQ6_K*>(weight_q6_k), res_out, y, N, K, eps);
    } else {
        constexpr int WARPS = 8;
        k_quant_gemv_q6_k_m1<WARPS, true, true><<<(N + WARPS - 1) / WARPS, WARPS * 32, smem, stream>>>(
            a, b, norm_w, static_cast<const BlockQ6_K*>(weight_q6_k), res_out, y, N, K, eps);
    }
}

void fused_gated_deltanet_step_f32(
    const float* qkv_proj, const float* z, const float* a, const float* b,
    const float* conv_w, const float* conv_bias, const float* A_log, const float* dt_bias,
    const float* norm_w, float* conv_state, float* ssm_state, float* out,
    int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, cudaStream_t stream, float* scratch, bool tiled_value_heads)
{
    if (scratch && dv >= 64 && (dk == 64 || dk == 128)) {
        float* alpha = scratch + 2 * num_k_heads * dk + num_v_heads * dv;
        float* beta = alpha + num_v_heads;
        float* y = beta + num_v_heads;
        k_gdn_decode_conv_qk<<<2 * num_k_heads + num_v_heads, 128, 0, stream>>>(
            qkv_proj, a, b, conv_w, conv_bias, A_log, dt_bias, conv_state, scratch, alpha, beta,
            num_k_heads, dk, dv, q_scale);
        if (dk == 128) {
            k_gdn_decode_ssm<128><<<num_v_heads * kGdnDvSplits, 32 * kGdnDkWarps, 0, stream>>>(
                scratch, alpha, beta, ssm_state, y, num_k_heads, num_v_heads, dv, tiled_value_heads);
        } else {
            k_gdn_decode_ssm<64><<<num_v_heads * kGdnDvSplits, 32 * kGdnDkWarps, 0, stream>>>(
                scratch, alpha, beta, ssm_state, y, num_k_heads, num_v_heads, dv, tiled_value_heads);
        }
        k_gdn_decode_gated_norm<<<num_v_heads, 128, 0, stream>>>(y, z, norm_w, out, num_v_heads, dv, eps);
        return;
    }
    if ((dk == 64 || dk == 128) && (dv % 32 == 0)) {
        size_t smem_bytes = (160 + 2 * static_cast<size_t>(dk) + 2 * static_cast<size_t>(dv)) * sizeof(float);
        if (dk == 64) {
            k_fused_gated_deltanet_step_tile<64><<<num_k_heads, 128, smem_bytes, stream>>>(
                qkv_proj, z, a, b, conv_w, conv_bias, A_log, dt_bias, norm_w,
                conv_state, ssm_state, out, num_k_heads, num_v_heads, dv, eps, q_scale, tiled_value_heads);
        } else {
            k_fused_gated_deltanet_step_tile<128><<<num_k_heads, 128, smem_bytes, stream>>>(
                qkv_proj, z, a, b, conv_w, conv_bias, A_log, dt_bias, norm_w,
                conv_state, ssm_state, out, num_k_heads, num_v_heads, dv, eps, q_scale, tiled_value_heads);
        }
        return;
    }
    int threads = std::min(256, std::max(32, ((std::max(dk, dv) + 31) / 32) * 32));
    size_t smem_bytes = (32 + 2 * static_cast<size_t>(dk) + static_cast<size_t>(dv)) * sizeof(float);
    if (dk <= 64) {
        k_fused_gated_deltanet_step<64><<<num_k_heads, threads, smem_bytes, stream>>>(
            qkv_proj, z, a, b, conv_w, conv_bias, A_log, dt_bias, norm_w,
            conv_state, ssm_state, out, num_k_heads, num_v_heads, dk, dv, eps, q_scale, tiled_value_heads);
    } else {
        k_fused_gated_deltanet_step<128><<<num_k_heads, threads, smem_bytes, stream>>>(
            qkv_proj, z, a, b, conv_w, conv_bias, A_log, dt_bias, norm_w,
            conv_state, ssm_state, out, num_k_heads, num_v_heads, dk, dv, eps, q_scale, tiled_value_heads);
    }
}

void unfused_gated_deltanet_step_f32(
    const float* qkv_proj, const float* z, const float* a, const float* b,
    const float* conv_w, const float* conv_bias, const float* A_log, const float* dt_bias,
    const float* norm_w, float* conv_state, float* ssm_state, float* out,
    float* scratch_qkv, float* scratch_y,
    int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, cudaStream_t stream, bool tiled_value_heads)
{
    int q_dim = num_k_heads * dk;
    int v_dim = num_v_heads * dv;
    int conv_dim = 2 * q_dim + v_dim;

    k_unfused_gdn_conv_silu<<<(conv_dim + 255) / 256, 256, 0, stream>>>(
        qkv_proj, conv_w, conv_bias, conv_state, scratch_qkv, conv_dim);

    int norm_threads = std::min(256, std::max(32, ((dk + 31) / 32) * 32));
    k_unfused_gdn_l2norm<<<num_k_heads, norm_threads, 0, stream>>>(scratch_qkv, num_k_heads, dk, q_scale);
    k_unfused_gdn_l2norm<<<num_k_heads, norm_threads, 0, stream>>>(scratch_qkv + q_dim, num_k_heads, dk, 1.0f);

    k_unfused_gdn_state_step<<<num_v_heads, dv, 0, stream>>>(
        scratch_qkv, a, b, A_log, dt_bias, ssm_state, scratch_y,
        num_k_heads, num_v_heads, dk, dv, tiled_value_heads);

    int out_threads = std::min(256, std::max(32, ((dv + 31) / 32) * 32));
    k_gdn_decode_gated_norm<<<num_v_heads, out_threads, 0, stream>>>(
        scratch_y, z, norm_w, out, num_v_heads, dv, eps);
}

void fused_qk_norm_rope_kv_append_f32(
    float* q, float* k, const float* v,
    const float* q_norm_w, const float* k_norm_w,
    void* k_cache, void* v_cache,
    int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    const int* pos_dev, int pos_host, int /*max_seq*/,
    float theta, float eps, bool neox, cudaStream_t stream, const float* rotary_factors, DType kv_dtype, CudaKvView cache)
{
    int total_heads = num_q_heads + num_kv_heads;
    int threads = std::min(256, std::max(32, ((head_dim + 31) / 32) * 32));
    size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);
    auto launch = [&]<typename Cache>() {
        k_qk_norm_rope_kv_append<<<total_heads, threads, smem_bytes, stream>>>(
            q, k, v, q_norm_w, k_norm_w, rotary_factors, static_cast<Cache*>(k_cache), static_cast<Cache*>(v_cache),
            num_q_heads, num_kv_heads, head_dim, rot_dim, pos_dev, pos_host, theta, eps, neox,
            num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim, cache);
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void fused_gemv_q8_0(const float* x, const void* w_q8, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<2, false><<<grid, block, 0, stream>>>(x, nullptr, w_q8, y, M, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q8_0<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ8_0*>(w_q8),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void fused_gemv_q4_0(const float* x, const void* w_q4, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<3, false><<<grid, block, 0, stream>>>(x, nullptr, w_q4, y, M, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_0<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ4_0*>(w_q4),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

template <int WK>
__global__ void k_repacked_k_linear(const float* x, const void* w, float* y, int N, int K) {
    const int n = blockIdx.x * 4 + threadIdx.x / 32;
    if (n >= N) return;
    const int m = blockIdx.y;
    const int lane = threadIdx.x % 32;
    float sum = 0;
    for (int k = lane; k < K; k += 32) {
        sum = fmaf(x[static_cast<int64_t>(m) * K + k], dequant_weight_elem<WK>(w, n, k - lane, lane, K), sum);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[static_cast<int64_t>(m) * N + n] = sum;
}

void linear_repacked_k_f32(const float* x, const void* w, DType dtype, float* y,
                           int M, int N, int K, cudaStream_t stream) {
    auto launch = [&]<int WK>() {
        if (M >= 16) {
            k_gemm_wmma<WK, false><<<dim3((N + 63) / 64, (M + 31) / 32), 128, 0, stream>>>(
                x, nullptr, w, y, M, N, K, 0);
        } else {
            k_repacked_k_linear<WK><<<dim3((N + 3) / 4, M), 128, 0, stream>>>(x, w, y, N, K);
        }
    };
    if (dtype == DType::Q4_KA) launch.template operator()<8>();
    else launch.template operator()<9>();
}

void fused_gemv_q4_1(const float* x, const void* w_q4_1, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<4, false><<<grid, block, 0, stream>>>(x, nullptr, w_q4_1, y, M, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_1<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ4_1*>(w_q4_1),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void fused_gemv_q4_k(const float* x, const void* w_q4_k, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 255) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<5, false><<<grid, block, 0, stream>>>(x, nullptr, w_q4_k, y, M, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_q5_k<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ4_K*>(w_q4_k),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void fused_gemv_q5_k(const float* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
    if (M <= 8) {
        constexpr int warps = 8;
        k_quant_gemv_q4_q5_k<warps, false, BlockQ5_K><<<(N + warps - 1) / warps, warps * 32, 0, stream>>>(
            x, nullptr, static_cast<const BlockQ5_K*>(w), y, M, N, K, 0.0f);
        return;
    }
    dim3 grid((N + 63) / 64, (M + 31) / 32);
    k_gemm_wmma<7, false><<<grid, 128, 0, stream>>>(x, nullptr, w, y, M, N, K, 0.0f);
}

void fused_gemv_q6_k(const float* x, const void* w_q6_k, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 255) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<6, false><<<grid, block, 0, stream>>>(x, nullptr, w_q6_k, y, M, N, K, 0.0f);
        return;
    }
    if (M == 1 && K <= 8192 && (K & 255) == 0) {
        constexpr int WARPS = 8;
        int blocks = (N + WARPS - 1) / WARPS;
        size_t smem = static_cast<size_t>(K) * sizeof(float);
        k_quant_gemv_q6_k_m1<WARPS, false, false><<<blocks, WARPS * 32, smem, stream>>>(
            x, nullptr, nullptr, static_cast<const BlockQ6_K*>(w_q6_k), nullptr, y, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q6_k<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ6_K*>(w_q6_k),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void fused_gemv_q4_a(const float* x, const void* w_q4_a, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M == 1 && K <= 8192 && (K & 255) == 0) {
        launch_dp4a_q4_a<false, false>(x, nullptr, nullptr, static_cast<const BlockQ4_A*>(w_q4_a), nullptr, y, N, K, 0.0f, stream);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q4_a_f32<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ4_A*>(w_q4_a),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void fused_gemv_q8_a(const float* x, const void* w_q8_a, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M == 1 && K <= 8192 && (K & 255) == 0) {
        launch_dp4a_q8_a<false, false>(x, nullptr, nullptr, static_cast<const BlockQ8_A*>(w_q8_a), nullptr, y, N, K, 0.0f, stream);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_quant_gemv_q8_a_f32<WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, static_cast<const BlockQ8_A*>(w_q8_a),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void gqa_splitk_decode_f32(
    const float* q, const void* k_cache, const void* v_cache,
    float* out, float* partial_out, float* partial_ml,
    int num_q_heads, int num_kv_heads, int head_dim,
    const int* kv_len_dev, int kv_len_host, int /*max_seq*/,
    float scale, int max_splits, cudaStream_t stream, bool fixed_splits, DType kv_dtype,
    BlockQ8_1* q8_out, const float* gate, CudaKvView cache)
{
    int num_splits = max_splits;
    if (!kv_len_dev && !fixed_splits && !q8_out) {
        if (kv_len_host <= 128) num_splits = 1;
        else num_splits = std::min(max_splits, std::max(1, (kv_len_host + 127) / 128));
    }
    int threads = std::min(256, std::max(32, ((head_dim + 31) / 32) * 32));
    size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);

    auto launch = [&]<typename Cache>() {
        const auto* keys = static_cast<const Cache*>(k_cache);
        const auto* values = static_cast<const Cache*>(v_cache);
        if constexpr (!std::is_same_v<Cache, float>) {
            if (num_splits > 1 && num_q_heads / num_kv_heads <= 8 &&
                (head_dim == 64 || head_dim == 128 || head_dim == 256)) {
                const dim3 grid(num_kv_heads, num_splits);
                auto grouped = [&]<int HD>() {
                    auto run = [&]<int Warps>() {
                        k_gqa_prefill_tiled<HD, Cache, true, Warps><<<grid, Warps * 32, 0, stream>>>(
                            q, 0, keys, values, partial_out, 1, num_q_heads, num_kv_heads,
                            kv_len_host, scale, kv_len_dev, num_splits, partial_ml, cache);
                    };
                    if (num_q_heads / num_kv_heads <= 4) run.template operator()<4>();
                    else run.template operator()<8>();
                };
                if (head_dim == 64) grouped.template operator()<64>();
                else if (head_dim == 128) grouped.template operator()<128>();
                else grouped.template operator()<256>();
                return;
            }
        }
        const dim3 grid(num_q_heads, num_splits);
        k_gqa_splitk_part1<<<grid, threads, smem_bytes, stream>>>(
            q, keys, values, num_splits == 1 ? out : partial_out, partial_ml,
            num_q_heads, num_kv_heads, head_dim, kv_len_dev, kv_len_host, scale, num_splits, cache);
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
    if (q8_out) {
        k_gqa_splitk_reduce<true><<<num_q_heads, threads, 0, stream>>>(
            partial_out, partial_ml, nullptr, q8_out, gate, num_q_heads, head_dim, num_splits);
    } else if (num_splits > 1) {
        k_gqa_splitk_reduce<false><<<num_q_heads, threads, 0, stream>>>(
            partial_out, partial_ml, out, nullptr, nullptr, num_q_heads, head_dim, num_splits);
    }
}

void moe_topk_router_f32(const float* logits, int* topk_idx, float* topk_weight,
                         int num_experts, int top_k, bool norm_topk, cudaStream_t stream) {
    k_moe_topk_router<<<1, 1, 0, stream>>>(logits, topk_idx, topk_weight, num_experts, top_k, norm_topk);
}

void moe_indexed_swiglu_gemv_f32(
    const float* x, const int* topk_idx, const float* gate_up_w, const float* down_w,
    const float* topk_weight, float* out, float* scratch_inter,
    int hidden, int inter, int top_k, cudaStream_t stream)
{
    constexpr int WARPS = 8;
    int total_up_rows = top_k * inter;
    int up_blocks = (total_up_rows + WARPS - 1) / WARPS;
    k_moe_indexed_gate_up_swiglu<<<up_blocks, WARPS * 32, 0, stream>>>(
        x, topk_idx, gate_up_w, scratch_inter, hidden, inter, top_k);

    int down_blocks = (hidden + WARPS - 1) / WARPS;
    k_moe_indexed_down_combine<<<down_blocks, WARPS * 32, 0, stream>>>(
        scratch_inter, topk_idx, down_w, topk_weight, out, hidden, inter, top_k);
}

void update_pos_and_kvlen_f32(int* pos_dev, int* kv_len_dev, int new_pos, int new_kv_len, cudaStream_t stream) {
    k_update_pos_and_kvlen<<<1, 1, 0, stream>>>(pos_dev, kv_len_dev, new_pos, new_kv_len);
}

void linear_w_f16_f32(const float* x, const uint16_t* w, float* y,
                      int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<0, false><<<grid, block, 0, stream>>>(x, nullptr, w, y, M, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_linear_gemv_w16_vec<half, WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, reinterpret_cast<const half*>(w),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void linear_w_bf16_f32(const float* x, const uint16_t* w, float* y,
                       int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<1, false><<<grid, block, 0, stream>>>(x, nullptr, w, y, M, N, K, 0.0f);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_linear_gemv_w16_vec<__nv_bfloat16, WARPS, false><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, nullptr, reinterpret_cast<const __nv_bfloat16*>(w),
            y + static_cast<int64_t>(m0) * N, mc, N, K, 0.0f);
    }
}

void fused_rmsnorm_linear_f16(const float* x, const float* norm_w, const uint16_t* weight_f16,
                              float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<0, true><<<grid, block, 0, stream>>>(x, norm_w, weight_f16, y, M, N, K, eps);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_linear_gemv_w16_vec<half, WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, reinterpret_cast<const half*>(weight_f16),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void fused_rmsnorm_linear_bf16(const float* x, const float* norm_w, const uint16_t* weight_bf16,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        dim3 block(128);
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        k_gemm_wmma<1, true><<<grid, block, 0, stream>>>(x, norm_w, weight_bf16, y, M, N, K, eps);
        return;
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int mc = std::min(8, M - m0);
        k_linear_gemv_w16_vec<__nv_bfloat16, WARPS, true><<<blocks, WARPS * 32, 0, stream>>>(
            x + static_cast<int64_t>(m0) * K, norm_w, reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
            y + static_cast<int64_t>(m0) * N, mc, N, K, eps);
    }
}

void embed_lookup_f32(const void* emb_table, DType table_dtype,
                      const int32_t* token_ids_dev, float* out,
                      int M, int hidden, cudaStream_t stream) {
    if (table_dtype == DType::Q4_A) {
        k_embed_lookup_repacked_f32<<<dim3(hidden / 256, M), 256, 0, stream>>>(
            static_cast<const BlockQ4_A*>(emb_table), token_ids_dev, out, hidden / 256);
        return;
    }
    if (table_dtype == DType::Q8_A) {
        k_embed_lookup_repacked_f32<<<dim3(hidden / 256, M), 256, 0, stream>>>(
            static_cast<const BlockQ8_A*>(emb_table), token_ids_dev, out, hidden / 256);
        return;
    }
    if (table_dtype == DType::Q4_KA) {
        k_embed_lookup_repacked_f32<<<dim3(hidden / 256, M), 256, 0, stream>>>(
            static_cast<const BlockQ4_KA*>(emb_table), token_ids_dev, out, hidden / 256);
        return;
    }
    if (table_dtype == DType::Q6_A) {
        k_embed_lookup_repacked_f32<<<dim3(hidden / 256, M), 256, 0, stream>>>(
            static_cast<const BlockQ6_A*>(emb_table), token_ids_dev, out, hidden / 256);
        return;
    }
    int total = M * hidden;
    k_embed_lookup_f32<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint8_t*>(emb_table), static_cast<int>(table_dtype),
        token_ids_dev, out, M, hidden);
}

void batched_qk_norm_rope_kv_append_f32(
    float* q, float* k, const float* v,
    const float* q_norm_w, const float* k_norm_w,
    void* k_cache, void* v_cache,
    int M, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    int start_pos, int /*max_seq*/, float theta, float eps, bool neox, cudaStream_t stream,
    int q_row_stride, int k_row_stride, int v_row_stride, const float* rotary_factors, DType kv_dtype, CudaKvView cache)
{
    int total_heads = num_q_heads + num_kv_heads;
    int threads = std::min(256, std::max(32, ((head_dim + 31) / 32) * 32));
    size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);
    dim3 grid(total_heads, M);
    int qs = (q_row_stride > 0) ? q_row_stride : (num_q_heads * head_dim);
    int ks = k_row_stride > 0 ? k_row_stride : num_kv_heads * head_dim;
    int vs = v_row_stride > 0 ? v_row_stride : ks;
    auto launch = [&]<typename Cache>() {
        k_qk_norm_rope_kv_append<<<grid, threads, smem_bytes, stream>>>(
            q, k, v, q_norm_w, k_norm_w, rotary_factors, static_cast<Cache*>(k_cache), static_cast<Cache*>(v_cache),
            num_q_heads, num_kv_heads, head_dim, rot_dim, nullptr, start_pos, theta, eps, neox, qs, ks, vs, cache);
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void gqa_prefill_causal_f32(
    const float* q, const void* k_cache, const void* v_cache, float* out,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int start_pos, int /*max_seq*/, float scale, cudaStream_t stream,
    int q_row_stride, DType kv_dtype, CudaKvView cache)
{
    int qs = (q_row_stride > 0) ? q_row_stride : (num_q_heads * head_dim);
    auto launch = [&]<typename Cache>() {
        const auto* keys = static_cast<const Cache*>(k_cache);
        const auto* values = static_cast<const Cache*>(v_cache);
        if (head_dim == 64 || head_dim == 128 || head_dim == 256) {
            const int group = num_q_heads / num_kv_heads;
            const dim3 grid(num_kv_heads, (M * group + 7) / 8);
            auto tiled = [&]<int HD>() {
                if constexpr (!std::is_same_v<Cache, int8_t>) {
                    if (cache.pages && start_pos + M <= kKvPageTokens) {
                        k_gqa_prefill_tiled<HD, Cache, false, 8, true><<<grid, 256, 0, stream>>>(
                            q, qs, keys, values, out, M, num_q_heads, num_kv_heads, start_pos, scale,
                            nullptr, 1, nullptr, cache);
                        return;
                    }
                }
                k_gqa_prefill_tiled<HD><<<grid, 256, 0, stream>>>(
                    q, qs, keys, values, out, M, num_q_heads, num_kv_heads, start_pos, scale, nullptr, 1, nullptr, cache);
            };
            if (head_dim == 64) tiled.template operator()<64>();
            else if (head_dim == 128) tiled.template operator()<128>();
            else tiled.template operator()<256>();
            return;
        }
        const int threads = std::min(256, std::max(32, ((head_dim + 31) / 32) * 32));
        const size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);
        const dim3 grid(num_q_heads, M);
        k_gqa_prefill_causal<<<grid, threads, smem_bytes, stream>>>(
            q, qs, keys, values, out, M, num_q_heads, num_kv_heads, head_dim, start_pos, scale, cache);
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void gqa_prefill_online_f32(const float* q, int q_stride, CudaKvView cache, float* out,
                            int rows, int q_heads, int kv_heads, int head_dim, int start_pos,
                            float scale, cudaStream_t stream) {
    auto launch = [&]<int HD, int Warps, typename Cache>() {
        constexpr int Q = 16 * Warps, K = HD == 256 ? 32 : 64;
        constexpr size_t bytes = ((HD <= 128 ? 0 : Q) + 2 * K) * HD * sizeof(half);
        if constexpr (bytes > 48 * 1024) {
            FUSIONRT_CUDA_CHECK(cudaFuncSetAttribute(k_gqa_prefill_online<HD, Warps, Cache>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        }
        const int queries = rows * (q_heads / kv_heads);
        const dim3 grid(kv_heads, (queries + Q - 1) / Q);
        k_gqa_prefill_online<HD, Warps, Cache><<<grid, 32 * Warps, bytes, stream>>>(
            q, q_stride, cache, out, rows, q_heads, kv_heads, start_pos, scale * 1.4426950408889634f);
    };
    auto dispatch = [&]<int HD, int Warps>() {
        if (cache.dtype == DType::Q8_0) launch.template operator()<HD, Warps, int8_t>();
        else if (cache.dtype == DType::F16) launch.template operator()<HD, Warps, half>();
        else launch.template operator()<HD, Warps, float>();
    };
    if (head_dim == 64) dispatch.template operator()<64, 4>();
    else if (head_dim == 128) dispatch.template operator()<128, 4>();
    else if (head_dim == 256) {
        int device, shared_limit;
        FUSIONRT_CUDA_CHECK(cudaGetDevice(&device));
        FUSIONRT_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (shared_limit >= 64 * 1024) dispatch.template operator()<256, 4>();
        else dispatch.template operator()<256, 2>();
    }
}

void split_q_gate_f32(const float* q_full, float* q_out, float* gate_out,
                      int M, int num_q_heads, int head_dim, cudaStream_t stream, int src_row_stride) {
    int total = M * num_q_heads * head_dim;
    int srs = (src_row_stride > 0) ? src_row_stride : (2 * num_q_heads * head_dim);
    k_split_q_gate_f32<<<(total + 255) / 256, 256, 0, stream>>>(q_full, q_out, gate_out, M, num_q_heads, head_dim, srs);
}

void apply_sigmoid_gate_f32(float* attn_out, const float* gate, int n, cudaStream_t stream) {
    k_apply_sigmoid_gate_f32<<<(n + 255) / 256, 256, 0, stream>>>(attn_out, gate, n);
}

void dequantize_to_f16(const void* w, DType dt, uint16_t* out_f16, int N, int K, cudaStream_t stream) {
    half* out = reinterpret_cast<half*>(out_f16);
    if (dt == DType::Q4_A) {
        int total_s = N * (K >> 8);
        dim3 block(32, 4);
        k_dequant_q4_a_to_f16<<<(total_s + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ4_A*>(w), out, N, K);
    } else if (dt == DType::Q8_A) {
        int total_s = N * (K >> 8);
        dim3 block(32, 4);
        k_dequant_q8_a_to_f16<<<(total_s + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ8_A*>(w), out, N, K);
    } else if (dt == DType::Q4_0) {
        int total_b = N * (K >> 5);
        dim3 block(32, 4);
        k_dequant_q4_0_to_f16<<<(total_b + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ4_0*>(w), out, N, K);
    } else if (dt == DType::Q8_0) {
        int total_b = N * (K >> 5);
        dim3 block(32, 4);
        k_dequant_q8_0_to_f16<<<(total_b + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ8_0*>(w), out, N, K);
    } else if (dt == DType::Q4_1) {
        int total_b = N * (K >> 5);
        k_dequant_q4_1_to_f16<<<(total_b + 63) / 64, 256, 0, stream>>>(
            static_cast<const BlockQ4_1*>(w), out, total_b);
    } else if (dt == DType::Q4_KA || dt == DType::Q6_A) {
        const auto grid = static_cast<unsigned>((static_cast<int64_t>(N) * K + 255) / 256);
        if (dt == DType::Q4_KA) k_dequant_generic_to_f16<8><<<grid, 256, 0, stream>>>(w, out, N, K);
        else k_dequant_generic_to_f16<9><<<grid, 256, 0, stream>>>(w, out, N, K);
    } else if (dt == DType::Q4_K) {
        int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_generic_to_f16<5><<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, out, N, K);
    } else if (dt == DType::Q5_K) {
        int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_generic_to_f16<7><<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, out, N, K);
    } else if (dt == DType::Q6_K) {
        int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_generic_to_f16<6><<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, out, N, K);
    } else if (dt == DType::F32 || dt == DType::BF16) {
        int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_f32_or_bf16_to_f16<<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, dt == DType::BF16 ? 1 : 0, out, total);
    } else if (dt == DType::F16) {
        cudaMemcpyAsync(out, w, static_cast<size_t>(N) * K * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    }
}

void rmsnorm_or_copy_to_f16(const float* x, const float* norm_w, uint16_t* out_f16,
                            int M, int K, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((K + 31) / 32) * 32));
    k_rmsnorm_or_copy_to_f16<<<M, threads, 0, stream>>>(x, norm_w, reinterpret_cast<half*>(out_f16), K, eps);
}

void cublas_linear_f16_f32(cublasHandle_t handle, const uint16_t* x_f16, const uint16_t* w_f16,
                           float* y, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w_f16, CUDA_R_16F, K,
        x_f16, CUDA_R_16F, K,
        &beta,
        y, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx failed with status " + std::to_string(static_cast<int>(st)));
    }
}

size_t gated_deltanet_prefill_scratch_floats(int M, int nk, int nv, int dk, int dv) {
    if (M <= 0) return 0;
    size_t rows = static_cast<size_t>(M) * (2 * nk * dk + 2 * nv * dv);
    size_t chunks = (static_cast<size_t>(M) + kGdnChunk - 1) / kGdnChunk;
    return rows + chunks * nv * kGdnCoeffStride;
}

void fused_gated_deltanet_prefill_f32(
    const float* qkv_proj, int qkv_stride,
    const float* z, int z_stride,
    const float* a, int a_stride,
    const float* b, int b_stride,
    const float* conv_w, const float* conv_bias, const float* A_log, const float* dt_bias,
    const float* norm_w, float* conv_state, float* ssm_state, float* out,
    int M, int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, cudaStream_t stream, float* scratch, size_t scratch_floats, bool tiled_value_heads)
{
    if (M <= 0) return;
    if (scratch != nullptr && M >= 4 && num_v_heads % num_k_heads == 0 && dv % 16 == 0 && (dk == 64 || dk == 128) &&
        scratch_floats >= gated_deltanet_prefill_scratch_floats(M, num_k_heads, num_v_heads, dk, dv)) {
        int q_dim = num_k_heads * dk;
        int v_dim = num_v_heads * dv;
        int conv_dim = 2 * q_dim + v_dim;
        float* qkv_act = scratch;
        float* y_raw = scratch + static_cast<size_t>(M) * conv_dim;

        k_gdn_prefill_conv_silu<<<(conv_dim + 31) / 32, 128, 0, stream>>>(
            qkv_proj, qkv_stride, conv_w, conv_bias, conv_state, qkv_act, M, conv_dim);
        k_gdn_prefill_l2_norm<<<dim3(num_k_heads, M), dk, 0, stream>>>(
            qkv_act, conv_dim, num_k_heads, dk, q_scale);
        float* coefficients = y_raw + static_cast<size_t>(M) * v_dim;
        dim3 chunks(num_v_heads, (M + kGdnChunk - 1) / kGdnChunk);
        dim3 state_tiles(num_v_heads, dv / 16);
        if (dk == 64) {
            k_gdn_chunk_coefficients<64><<<chunks, 256, 0, stream>>>(
                qkv_act, conv_dim, a, a_stride, b, b_stride, A_log, dt_bias, coefficients, M, num_k_heads, num_v_heads, tiled_value_heads);
            k_gdn_chunkwise_ssm<64><<<state_tiles, 256, 0, stream>>>(
                qkv_act, conv_dim, coefficients, ssm_state, y_raw, M, num_k_heads, num_v_heads, dv, tiled_value_heads);
        } else {
            k_gdn_chunk_coefficients<128><<<chunks, 256, 0, stream>>>(
                qkv_act, conv_dim, a, a_stride, b, b_stride, A_log, dt_bias, coefficients, M, num_k_heads, num_v_heads, tiled_value_heads);
            k_gdn_chunkwise_ssm<128><<<state_tiles, 256, 0, stream>>>(
                qkv_act, conv_dim, coefficients, ssm_state, y_raw, M, num_k_heads, num_v_heads, dv, tiled_value_heads);
        }
        k_gdn_prefill_gated_rmsnorm<<<dim3(num_v_heads, M), dv, 0, stream>>>(
            y_raw, z, z_stride, norm_w, out, num_v_heads, dv, eps);
        return;
    }
    if (num_k_heads == num_v_heads && dk == dv && dk <= 128) {
        int threads = std::min(256, std::max(32, ((dk + 31) / 32) * 32));
        size_t smem_bytes = (32 + 2 * static_cast<size_t>(dk)) * sizeof(float);
        if (dk <= 64) {
            k_fused_gated_deltanet_prefill_r1<64><<<num_k_heads, threads, smem_bytes, stream>>>(
                qkv_proj, qkv_stride, z, z_stride, a, a_stride, b, b_stride,
                conv_w, conv_bias, A_log, dt_bias, norm_w, conv_state, ssm_state, out,
                M, num_k_heads, dk, eps, q_scale);
        } else {
            k_fused_gated_deltanet_prefill_r1<128><<<num_k_heads, threads, smem_bytes, stream>>>(
                qkv_proj, qkv_stride, z, z_stride, a, a_stride, b, b_stride,
                conv_w, conv_bias, A_log, dt_bias, norm_w, conv_state, ssm_state, out,
                M, num_k_heads, dk, eps, q_scale);
        }
        return;
    }
    int threads = std::min(256, std::max(32, ((std::max(dk, dv) + 31) / 32) * 32));
    size_t smem_bytes = (32 + 2 * static_cast<size_t>(dk) + static_cast<size_t>(dv)) * sizeof(float);
    if (dk <= 64) {
        k_fused_gated_deltanet_prefill_generic<64><<<num_k_heads, threads, smem_bytes, stream>>>(
            qkv_proj, qkv_stride, z, z_stride, a, a_stride, b, b_stride,
            conv_w, conv_bias, A_log, dt_bias, norm_w, conv_state, ssm_state, out,
            M, num_k_heads, num_v_heads, dk, dv, eps, q_scale, tiled_value_heads);
    } else {
        k_fused_gated_deltanet_prefill_generic<128><<<num_k_heads, threads, smem_bytes, stream>>>(
            qkv_proj, qkv_stride, z, z_stride, a, a_stride, b, b_stride,
            conv_w, conv_bias, A_log, dt_bias, norm_w, conv_state, ssm_state, out,
            M, num_k_heads, num_v_heads, dk, dv, eps, q_scale, tiled_value_heads);
    }
}

}  // namespace fusionrt::cuda_ops
