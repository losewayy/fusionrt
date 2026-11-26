#include "weight_tiles.cuh"

#include <algorithm>

namespace fusionrt::cuda_ops {

namespace {

// Unfused GEMV for M=1..8 (warp-per-row for each output row)
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

}  // namespace

void linear_f32(const float* x, const float* w, float* y,
                int M, int N, int K, cudaStream_t stream) {
    if (M <= 8) {
        constexpr int WARPS = 8;
        int blocks = (N + WARPS - 1) / WARPS;
        k_linear_gemv_f32<<<blocks, WARPS * 32, 0, stream>>>(x, w, y, M, N, K);
    } else {
        matmul::launch_gemm_tiled_f32(x, w, y, M, N, K, stream);
    }
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
    matmul::launch_mmq_q4_a_and_q8_a(xq, w4, w8, y4, y8, M, N4, N8, K, stream);
    return true;
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
        matmul::launch_wmma<2, true>(x, norm_w, weight_q8, y, M, N, K, eps, stream);
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
        matmul::launch_wmma<3, true>(x, norm_w, weight_q4, y, M, N, K, eps, stream);
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
        matmul::launch_wmma<4, true>(x, norm_w, weight_q4_1, y, M, N, K, eps, stream);
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
        matmul::launch_wmma<5, true>(x, norm_w, weight_q4_k, y, M, N, K, eps, stream);
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
        matmul::launch_wmma<6, true>(x, norm_w, weight_q6_k, y, M, N, K, eps, stream);
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

void fused_gemv_q8_0(const float* x, const void* w_q8, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        matmul::launch_wmma<2, false>(x, nullptr, w_q8, y, M, N, K, 0.0f, stream);
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
        matmul::launch_wmma<3, false>(x, nullptr, w_q4, y, M, N, K, 0.0f, stream);
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
            matmul::launch_wmma<WK, false>(x, nullptr, w, y, M, N, K, 0, stream);
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
        matmul::launch_wmma<4, false>(x, nullptr, w_q4_1, y, M, N, K, 0.0f, stream);
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
        matmul::launch_wmma<5, false>(x, nullptr, w_q4_k, y, M, N, K, 0.0f, stream);
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
    matmul::launch_wmma<7, false>(x, nullptr, w, y, M, N, K, 0.0f, stream);
}

void fused_gemv_q6_k(const float* x, const void* w_q6_k, float* y,
                     int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 255) == 0) {
        matmul::launch_wmma<6, false>(x, nullptr, w_q6_k, y, M, N, K, 0.0f, stream);
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

void linear_w_f16_f32(const float* x, const uint16_t* w, float* y,
                      int M, int N, int K, cudaStream_t stream) {
    if (M >= 16 && (K & 31) == 0) {
        matmul::launch_wmma<0, false>(x, nullptr, w, y, M, N, K, 0.0f, stream);
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
        matmul::launch_wmma<1, false>(x, nullptr, w, y, M, N, K, 0.0f, stream);
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
        matmul::launch_wmma<0, true>(x, norm_w, weight_f16, y, M, N, K, eps, stream);
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
        matmul::launch_wmma<1, true>(x, norm_w, weight_bf16, y, M, N, K, eps, stream);
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

}  // namespace fusionrt::cuda_ops
