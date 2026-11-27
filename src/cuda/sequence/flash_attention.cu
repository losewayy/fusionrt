#include "../cuda_common.cuh"

#include <algorithm>

namespace fusionrt::cuda_ops {

namespace {

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
    float scale, int num_splits, CudaKvView cache, int sliding_window)
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

    const int first_key = sliding_window > 0 ? max(0, kv_len - sliding_window) : 0;
    const int visible = kv_len - first_key;
    const int t_start = first_key + (visible * split_id) / num_splits;
    const int t_end = first_key + (visible * (split_id + 1)) / num_splits;

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
        __syncthreads();
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

template <typename Cache, bool ImageMask = false>
__global__ void k_gqa_prefill_causal(
    const float* __restrict__ q, int q_stride,
    const Cache* __restrict__ k_cache,
    const Cache* __restrict__ v_cache,
    float* __restrict__ out,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int start_pos, float scale, CudaKvView cache, int sliding_window, bool causal, const int* image_bounds)
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

    int kv_end = start_pos + (causal ? m + 1 : M);
    int first_key = sliding_window > 0 ? max(0, kv_end - sliding_window) : 0;
    if constexpr (ImageMask) {
        if (image_bounds[2 * m] >= 0) {
            kv_end = image_bounds[2 * m + 1];
            first_key = min(first_key, image_bounds[2 * m]);
        }
    }
    float m_i = -1e30f;
    float l_i = 0.0f;
    float o_j = 0.0f;

    for (int t = first_key; t < kv_end; ++t) {
        const auto* keys = kv_row<false>(k_cache, cache, t, kvh, num_kv_heads, head_dim);
        const auto* values = kv_row<true>(v_cache, cache, t, kvh, num_kv_heads, head_dim);
        const float qk_part = tid < head_dim
            ? q_sh[tid] * kv_value<false>(keys, cache, t, kvh, num_kv_heads, head_dim, tid) : 0.0f;
        float score = block_reduce_sum(qk_part, red_buf);

        float m_new = fmaxf(m_i, score);
        float alpha = (t == first_key) ? 0.0f : expf(m_i - m_new);
        float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        if (tid < head_dim) {
            o_j = o_j * alpha + p * kv_value<true>(values, cache, t, kvh, num_kv_heads, head_dim, tid);
        }
        m_i = m_new;
        __syncthreads();
    }

    if (tid < head_dim) {
        out[out_off + tid] = (l_i > 0.0f) ? (o_j / l_i) : 0.0f;
    }
}

template <int HD, typename Cache, bool Decode = false, int Warps = 8, bool FirstPage = false,
          bool Windowed = false, bool Causal = true>
__global__ void k_gqa_prefill_tiled(
    const float* __restrict__ q, int q_stride,
    const Cache* __restrict__ k_cache, const Cache* __restrict__ v_cache,
    float* __restrict__ out, int M, int num_q_heads, int num_kv_heads,
    int start_pos, float scale, const int* kv_len_dev = nullptr, int splits = 1,
    float* partial_ml = nullptr, CudaKvView cache = {}, int sliding_window = 0)
{
    constexpr int BKV = HD == 512 || std::is_same_v<Cache, float> ? 8 : 16, DP = HD / 32;
    constexpr int TokensPerWarp = Decode ? 4 : 1;
    constexpr int ScoreLanes = 32 / TokensPerWarp;
    // Separate the four token rows' shared-memory banks.
    alignas(16) __shared__ float keys[BKV][HD + (Decode ? ScoreLanes : 0)];
    alignas(16) __shared__ float values[BKV][HD];

    int kvh = Decode ? blockIdx.x : blockIdx.y;
    int group = num_q_heads / num_kv_heads;
    int first = Decode ? 0 : blockIdx.x * Warps;
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
    int tile_start = Decode ? (kv_len * blockIdx.y) / splits : 0;
    int kv_end = Decode ? (kv_len * (blockIdx.y + 1)) / splits : start_pos + (Causal ? m + 1 : M);
    int first_key = 0;
    if constexpr (Windowed) {
        if constexpr (Decode) {
            first_key = max(0, kv_len - sliding_window);
            const int visible = kv_len - first_key;
            tile_start = first_key + (visible * blockIdx.y) / splits;
            kv_end = first_key + (visible * (blockIdx.y + 1)) / splits;
        } else {
            first_key = max(0, kv_end - sliding_window);
            tile_start = max(0, start_pos + first / group + 1 - sliding_window) / BKV * BKV;
        }
    }
    const int tile_end = Decode || !Causal ? kv_end : start_pos + min(M, (first + Warps - 1) / group + 1);
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
        if (valid && base < kv_end && (!Windowed || base + BKV > first_key)) {
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
                    const int pos = base + t + j;
                    scores[t + j] = pos < kv_end && (!Windowed || pos >= first_key) ? score : -INFINITY;
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

__device__ __forceinline__ void attention_mma_f16(uint2& c, uint4 a, uint2 b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c.x), "+r"(c.y)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

template <int HD>
__device__ __forceinline__ int attention_offset(int row, int col) {
    return row * HD + (col ^ ((row & 7) * 8));
}

struct AttentionQ8Pairs {
    half2 lo, hi;
};

__device__ __forceinline__ AttentionQ8Pairs attention_dequantize_q8(unsigned bytes, half2 scale) {
    bytes ^= 0x80808080u;
    // 0x6400 is 1024 with unit spacing. Subtracting 1024 + 128 restores the signed byte.
    const unsigned lo = __byte_perm(bytes, 0x64646464u, 0x5150);
    const unsigned hi = __byte_perm(bytes, 0x64646464u, 0x5352);
    const auto bias = __float2half2_rn(1152.0f);
    return {__hmul2(__hsub2(reinterpret_cast<const half2&>(lo), bias), scale),
            __hmul2(__hsub2(reinterpret_cast<const half2&>(hi), bias), scale)};
}

template <typename Cache, int HD>
__device__ __forceinline__ void attention_stage_kv_row(half* tile, int r, const Cache* row, int d, half scale) {
    half* dst = tile + attention_offset<HD>(r, d);
    if constexpr (std::is_same_v<Cache, half>) {
        const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            :: "r"(shared), "l"(row + d) : "memory");
    } else if constexpr (std::is_same_v<Cache, int8_t>) {
        const auto scale2 = __halves2half2(scale, scale);
        if constexpr (HD == 256) {
            // One lane per eight columns keeps each shared-memory store contiguous after swizzling.
            const auto packed = *reinterpret_cast<const uint2*>(row + d);
            const auto lo = attention_dequantize_q8(packed.x, scale2);
            const auto hi = attention_dequantize_q8(packed.y, scale2);
            *reinterpret_cast<uint4*>(dst) = {
                reinterpret_cast<const unsigned&>(lo.lo), reinterpret_cast<const unsigned&>(lo.hi),
                reinterpret_cast<const unsigned&>(hi.lo), reinterpret_cast<const unsigned&>(hi.hi)};
        } else {
            const auto packed = *reinterpret_cast<const uint4*>(row + d);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto value = attention_dequantize_q8(reinterpret_cast<const unsigned*>(&packed)[i], scale2);
                auto* pair = reinterpret_cast<half2*>(tile + attention_offset<HD>(r, d + (i / 2) * 8)) + (i % 2) * 2;
                pair[0] = value.lo;
                pair[1] = value.hi;
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            reinterpret_cast<half2*>(dst)[i / 2] = __floats2half2_rn(row[d + i], row[d + i + 1]);
        }
    }
}

template <bool Value, typename Cache, int HD>
__device__ __forceinline__ void attention_stage_kv(half* tile, int r, CudaKvView cache, int token,
                                                    int head, int heads, int d) {
    const auto* row = kv_row<Value>(static_cast<const Cache*>(nullptr), cache, token, head, heads, HD);
    half scale{};
    if constexpr (std::is_same_v<Cache, int8_t>) {
        scale = kv_scales<Value>(cache, token, head, heads, HD)[d / 32];
    }
    attention_stage_kv_row<Cache, HD>(tile, r, row, d, scale);
}

template <typename Cache, int HD, int K, int Warps>
__device__ __forceinline__ void attention_stage_kv_tile(half* sk, half* sv, CudaKvView cache, int base, int last,
                                                        int head, int heads) {
    static_assert(kKvPageTokens % K == 0);
    constexpr int Columns = HD / 8;
    constexpr int RowsPerStep = 32 * Warps / Columns;
    // A tile stays within one page, so both Q8 data and scales use the same page pointer.
    const auto* keys = cache.pages[base / kKvPageTokens];
    const auto* values = keys + cache.plane_bytes;
    const int first = base % kKvPageTokens;
    const int d = (threadIdx.x % Columns) * 8;
    #pragma unroll 1
    for (int r = threadIdx.x / Columns; r < K; r += RowsPerStep) {
        const int row = ((first + min(r, last - base)) * heads + head) * HD;
        half k_scale{}, v_scale{};
        if constexpr (std::is_same_v<Cache, int8_t>) {
            const int i = (row + d) / 32;
            k_scale = reinterpret_cast<const half*>(keys + cache.elements)[i];
            v_scale = reinterpret_cast<const half*>(values + cache.elements)[i];
        }
        attention_stage_kv_row<Cache, HD>(sk, r, reinterpret_cast<const Cache*>(keys) + row, d, k_scale);
        attention_stage_kv_row<Cache, HD>(sv, r, reinterpret_cast<const Cache*>(values) + row, d, v_scale);
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

__device__ __forceinline__ float4 attention_unpack(uint2 x) {
    const auto lo = __half22float2(reinterpret_cast<const half2&>(x.x));
    const auto hi = __half22float2(reinterpret_cast<const half2&>(x.y));
    return {lo.x, lo.y, hi.x, hi.y};
}

__device__ __forceinline__ unsigned attention_rescale(unsigned x, half2 alpha) {
    const auto y = __hmul2(reinterpret_cast<const half2&>(x), alpha);
    return reinterpret_cast<const unsigned&>(y);
}

template <bool Windowed>
__device__ __forceinline__ unsigned attention_mask_pair(unsigned x, int col, int last, int first) {
    constexpr unsigned neg_inf = 0xfc00u;
    if constexpr (Windowed) {
        if (col + 1 < first) return neg_inf | (neg_inf << 16);
        if (col < first) return (x & 0xffff0000u) | neg_inf;
    }
    if (col > last) return neg_inf | (neg_inf << 16);
    if (col == last) return (x & 0xffffu) | (neg_inf << 16);
    return x;
}

template <bool Windowed>
__device__ __forceinline__ float attention_weight(float score, float maximum) {
    if constexpr (Windowed) {
        if (maximum == -INFINITY) return 0.0f;
    }
    return attention_exp2(score - maximum);
}

template <int HD, int Warps>
struct OnlineAttentionTile {
    static constexpr int Q = 16 * Warps, K = HD == 256 ? 32 : 64;
    static constexpr size_t Bytes = 2 * K * HD * sizeof(half);
};

// Keep scores, softmax state and output fragments on chip; K and V are shared
// by all query rows and heads in this KV group.
template <int HD, int Warps, typename Cache, bool Windowed>
__device__ __forceinline__ void gqa_prefill_online(const float* q, int q_stride, CudaKvView cache, float* out,
                                                  int rows, int nq, int nkv, int start, float scale, half* sh, int sliding_window) {
    using Tile = OnlineAttentionTile<HD, Warps>;
    constexpr int Q = Tile::Q, K = Tile::K;
    constexpr bool HalfPV = false, HalfQK = HD == 256;
    constexpr unsigned mask = 0xffffffffu;
    auto* sk = sh;
    auto* sv = sk + K * HD;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int quad = lane / 4, col = 2 * (lane % 4);
    const int group = nq / nkv, head = blockIdx.x, qr = blockIdx.y * Q;
    const int end = rows * group;
    for (int i = tid; i < Q * HD / 8; i += blockDim.x) {
        const int r = i / (HD / 8), d = (i % (HD / 8)) * 8;
        const int j = min(qr + r, end - 1);
        const float* src = q + (j / group) * q_stride + (head * group + j % group) * HD + d;
        auto* dst = reinterpret_cast<half2*>(sh + attention_offset<HD>(r, d));
        #pragma unroll
        for (int e = 0; e < 8; e += 2) {
            if constexpr (HalfQK) dst[e / 2] = __floats2half2_rn(src[e] * scale, src[e + 1] * scale);
            else dst[e / 2] = __floats2half2_rn(src[e], src[e + 1]);
        }
    }
    __syncthreads();
    const int j0 = qr + warp * 16 + quad, j1 = j0 + 8;
    const int last0 = start + min(j0, end - 1) / group, last1 = start + min(j1, end - 1) / group;
    const int first = start + qr / group, last = start + min(qr + Q - 1, end - 1) / group;
    const int arow = warp * 16 + lane % 16, acol = 8 * (lane / 16);
    uint4 query[HD / 16];
    #pragma unroll
    for (int d = 0; d < HD / 16; ++d) {
        query[d] = attention_load_q(sh + attention_offset<HD>(arow, 16 * d + acol));
    }
    __syncthreads();
    float max0 = -INFINITY, max1 = -INFINITY, sum0 = 0.0f, sum1 = 0.0f;
    std::conditional_t<HalfPV, uint2, float4> acc[HD / 8]{};
    const int first0 = Windowed ? max(0, last0 + 1 - sliding_window) : 0;
    const int first1 = Windowed ? max(0, last1 + 1 - sliding_window) : 0;
    const int start_tile = Windowed ? max(0, first + 1 - sliding_window) / K * K : 0;
    for (int base = start_tile; base <= last; base += K) {
        if constexpr (HD == 256) {
            attention_stage_kv_tile<Cache, HD, K, Warps>(sk, sv, cache, base, last, head, nkv);
        } else {
            constexpr int V = std::is_same_v<Cache, int8_t> ? 16 : 8;
            for (int i = tid; i < K * HD / V; i += blockDim.x) {
                const int r = i / (HD / V), d = (i % (HD / V)) * V;
                const int t = min(base + r, last);
                attention_stage_kv<false, Cache, HD>(sk, r, cache, t, head, nkv, d);
                attention_stage_kv<true, Cache, HD>(sv, r, cache, t, head, nkv, d);
            }
        }
        if constexpr (std::is_same_v<Cache, half>) {
            asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        std::conditional_t<HalfQK, uint2, float4> scores[K / 8]{};
        #pragma unroll
        for (int d = 0; d < HD / 16; ++d) {
            const auto a = query[d];
            #pragma unroll
            for (int t = 0; t < K / 8; ++t) {
                auto b = attention_load_b<false>(sk + attention_offset<HD>(t * 8 + lane % 8,
                                                                           16 * d + 8 * ((lane / 8) % 2)));
                if constexpr (HalfQK) attention_mma_f16(scores[t], a, b);
                else attention_mma(scores[t], a, b);
            }
        }
        auto score = [](auto s) {
            if constexpr (HalfQK) return attention_unpack(s);
            else return s;
        };
        if constexpr (HalfQK) {
            // Only the tiles intersecting the causal boundary need per-query masks.
            if (Windowed || base + K - 1 > first) {
                #pragma unroll
                for (int i = 0; i < K / 8; ++i) {
                    const int t = base + 8 * i + col;
                    auto& s = scores[i];
                    s = {attention_mask_pair<Windowed>(s.x, t, last0, first0), attention_mask_pair<Windowed>(s.y, t, last1, first1)};
                }
            }
        }
        float next0 = max0, next1 = max1;
        #pragma unroll
        for (int i = 0; i < K / 8; ++i) {
            if constexpr (!HalfQK) {
                const int t = base + 8 * i + col;
                auto& s = scores[i];
                s = {t >= first0 && t <= last0 ? s.x * scale : -INFINITY,
                     t + 1 >= first0 && t + 1 <= last0 ? s.y * scale : -INFINITY,
                     t >= first1 && t <= last1 ? s.z * scale : -INFINITY,
                     t + 1 >= first1 && t + 1 <= last1 ? s.w * scale : -INFINITY};
            }
            const auto s = score(scores[i]);
            next0 = fmaxf(next0, fmaxf(s.x, s.y));
            next1 = fmaxf(next1, fmaxf(s.z, s.w));
        }
        #pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            next0 = fmaxf(next0, __shfl_xor_sync(mask, next0, x));
            next1 = fmaxf(next1, __shfl_xor_sync(mask, next1, x));
        }
        const float alpha0 = attention_weight<Windowed>(max0, next0), alpha1 = attention_weight<Windowed>(max1, next1);
        max0 = next0;
        max1 = next1;
        float tile0 = 0.0f, tile1 = 0.0f;
        uint4 p[K / 16];
        #pragma unroll
        for (int i = 0; i < K / 16; ++i) {
            const auto a = score(scores[2 * i]), b = score(scores[2 * i + 1]);
            const float ax = attention_weight<Windowed>(a.x, max0), ay = attention_weight<Windowed>(a.y, max0);
            const float az = attention_weight<Windowed>(a.z, max1), aw = attention_weight<Windowed>(a.w, max1);
            const float bx = attention_weight<Windowed>(b.x, max0), by = attention_weight<Windowed>(b.y, max0);
            const float bz = attention_weight<Windowed>(b.z, max1), bw = attention_weight<Windowed>(b.w, max1);
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
            if constexpr (HalfPV) {
                const auto alpha_lo = __float2half2_rn(alpha0), alpha_hi = __float2half2_rn(alpha1);
                #pragma unroll
                for (auto& a : acc) {
                    a = {attention_rescale(a.x, alpha_lo), attention_rescale(a.y, alpha_hi)};
                }
            } else {
                #pragma unroll
                for (auto& a : acc) a = {a.x * alpha0, a.y * alpha0, a.z * alpha1, a.w * alpha1};
            }
        }
        #pragma unroll
        for (int d = 0; d < HD / 8; ++d) {
            auto& a = acc[d];
            #pragma unroll
            for (int i = 0; i < K / 16; ++i) {
                const auto b = attention_load_b<true>(sv + attention_offset<HD>(16 * i + lane % 16, 8 * d));
                if constexpr (HalfPV) attention_mma_f16(a, p[i], b);
                else attention_mma(a, p[i], b);
            }
        }
        __syncthreads();
    }
    const float inv0 = 1.0f / sum0, inv1 = 1.0f / sum1;
    #pragma unroll
    for (int d = 0; d < HD / 8; ++d) {
        float4 a;
        if constexpr (HalfPV) a = attention_unpack(acc[d]);
        else a = acc[d];
        if (j0 < end) {
            auto* dst = out + ((j0 / group) * nq + head * group + j0 % group) * HD + d * 8 + col;
            *reinterpret_cast<float2*>(dst) = {a.x * inv0, a.y * inv0};
        }
        if (j1 < end) {
            auto* dst = out + ((j1 / group) * nq + head * group + j1 % group) * HD + d * 8 + col;
            *reinterpret_cast<float2*>(dst) = {a.z * inv1, a.w * inv1};
        }
    }
}

template <int HD, int Warps, typename Cache, bool Windowed>
    requires (HD != 256)
__global__ void k_gqa_prefill_online(const float* q, int q_stride, CudaKvView cache, float* out,
                                     int rows, int nq, int nkv, int start, float scale, int sliding_window) {
    extern __shared__ __align__(16) half sh[];
    gqa_prefill_online<HD, Warps, Cache, Windowed>(q, q_stride, cache, out, rows, nq, nkv, start, scale, sh, sliding_window);
}

template <int HD, int Warps, typename Cache, bool Windowed>
    requires (HD == 256)
__global__ __launch_bounds__(32 * Warps, 3)
void k_gqa_prefill_online(const float* q, int q_stride, CudaKvView cache, float* out,
                          int rows, int nq, int nkv, int start, float scale, int sliding_window) {
    extern __shared__ __align__(16) half sh[];
    gqa_prefill_online<HD, Warps, Cache, Windowed>(q, q_stride, cache, out, rows, nq, nkv, start, scale, sh, sliding_window);
}

}  // namespace

void gqa_splitk_decode_f32(
    const float* q, const void* k_cache, const void* v_cache,
    float* out, float* partial_out, float* partial_ml,
    int num_q_heads, int num_kv_heads, int head_dim,
    const int* kv_len_dev, int kv_len_host, int /*max_seq*/,
    float scale, int max_splits, cudaStream_t stream, bool fixed_splits, DType kv_dtype,
    BlockQ8_1* q8_out, const float* gate, CudaKvView cache, int sliding_window)
{
    int num_splits = max_splits;
    if (!kv_len_dev && !fixed_splits && !q8_out) {
        num_splits = select_decode_splits({}, kv_len_host, num_kv_heads, max_splits);
    }
    int threads = std::min(1024, std::max(32, ((head_dim + 31) / 32) * 32));
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
                        if (sliding_window > 0) {
                            k_gqa_prefill_tiled<HD, Cache, true, Warps, false, true><<<grid, Warps * 32, 0, stream>>>(
                                q, 0, keys, values, partial_out, 1, num_q_heads, num_kv_heads,
                                kv_len_host, scale, kv_len_dev, num_splits, partial_ml, cache, sliding_window);
                        } else {
                            k_gqa_prefill_tiled<HD, Cache, true, Warps><<<grid, Warps * 32, 0, stream>>>(
                                q, 0, keys, values, partial_out, 1, num_q_heads, num_kv_heads,
                                kv_len_host, scale, kv_len_dev, num_splits, partial_ml, cache);
                        }
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
            num_q_heads, num_kv_heads, head_dim, kv_len_dev, kv_len_host, scale, num_splits, cache, sliding_window);
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

void gqa_prefill_causal_f32(
    const float* q, const void* k_cache, const void* v_cache, float* out,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int start_pos, int /*max_seq*/, float scale, cudaStream_t stream,
    int q_row_stride, DType kv_dtype, CudaKvView cache, int sliding_window, bool causal, const int* image_bounds)
{
    int qs = (q_row_stride > 0) ? q_row_stride : (num_q_heads * head_dim);
    auto launch = [&]<typename Cache>() {
        const auto* keys = static_cast<const Cache*>(k_cache);
        const auto* values = static_cast<const Cache*>(v_cache);
        if (!image_bounds && (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512)) {
            const int group = num_q_heads / num_kv_heads;
            const dim3 grid(static_cast<unsigned>((int64_t{M} * group + 7) / 8), num_kv_heads);
            auto tiled = [&]<int HD>() {
                if (!causal) {
                    k_gqa_prefill_tiled<HD, Cache, false, 8, false, false, false><<<grid, 256, 0, stream>>>(
                        q, qs, keys, values, out, M, num_q_heads, num_kv_heads, start_pos, scale,
                        nullptr, 1, nullptr, cache);
                    return;
                }
                if (sliding_window > 0) {
                    k_gqa_prefill_tiled<HD, Cache, false, 8, false, true><<<grid, 256, 0, stream>>>(
                        q, qs, keys, values, out, M, num_q_heads, num_kv_heads, start_pos, scale,
                        nullptr, 1, nullptr, cache, sliding_window);
                    return;
                }
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
            else if (head_dim == 256) tiled.template operator()<256>();
            else tiled.template operator()<512>();
            return;
        }
        const int threads = std::min(1024, std::max(32, ((head_dim + 31) / 32) * 32));
        const size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);
        const dim3 grid(num_q_heads, M);
        if (image_bounds) {
            k_gqa_prefill_causal<Cache, true><<<grid, threads, smem_bytes, stream>>>(
                q, qs, keys, values, out, M, num_q_heads, num_kv_heads, head_dim, start_pos, scale,
                cache, sliding_window, true, image_bounds);
        } else {
            k_gqa_prefill_causal<Cache><<<grid, threads, smem_bytes, stream>>>(
                q, qs, keys, values, out, M, num_q_heads, num_kv_heads, head_dim, start_pos, scale,
                cache, causal ? sliding_window : 0, causal, nullptr);
        }
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void gqa_prefill_online_f32(const float* q, int q_stride, CudaKvView cache, float* out,
                            int rows, int q_heads, int kv_heads, int head_dim, int start_pos,
                            float scale, cudaStream_t stream, int sliding_window) {
    auto launch = [&]<int HD, int Warps, typename Cache>() {
        constexpr int Q = OnlineAttentionTile<HD, Warps>::Q;
        constexpr size_t bytes = OnlineAttentionTile<HD, Warps>::Bytes;
        const int queries = rows * (q_heads / kv_heads);
        const dim3 grid(kv_heads, (queries + Q - 1) / Q);
        const float scale2 = scale * 1.4426950408889634f;
        if (sliding_window > 0) {
            k_gqa_prefill_online<HD, Warps, Cache, true><<<grid, 32 * Warps, bytes, stream>>>(
                q, q_stride, cache, out, rows, q_heads, kv_heads, start_pos, scale2, sliding_window);
        } else {
            k_gqa_prefill_online<HD, Warps, Cache, false><<<grid, 32 * Warps, bytes, stream>>>(
                q, q_stride, cache, out, rows, q_heads, kv_heads, start_pos, scale2, 0);
        }
    };
    auto dispatch = [&]<int HD, int Warps>() {
        if (cache.dtype == DType::Q8_0) launch.template operator()<HD, Warps, int8_t>();
        else if (cache.dtype == DType::F16) launch.template operator()<HD, Warps, half>();
        else launch.template operator()<HD, Warps, float>();
    };
    if (head_dim == 64) dispatch.template operator()<64, 4>();
    else if (head_dim == 128) dispatch.template operator()<128, 4>();
    else if (head_dim == 256) dispatch.template operator()<256, 4>();
    else throw std::invalid_argument("Online GQA prefill requires head_dim 64, 128, or 256");
}

}  // namespace fusionrt::cuda_ops
