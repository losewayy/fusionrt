#include "epilogue_common.cuh"
#include "../matmul/weight_tiles.cuh"
#include "fusionrt/weight_codec.h"

#include <algorithm>

namespace fusionrt::cuda_ops {

namespace {

template<int StaticExperts>
__global__ void k_moe_topk_router(const float* __restrict__ logits,
                                  int* __restrict__ topk_idx,
                                  float* __restrict__ topk_weight,
                                  int num_experts, int top_k, bool norm_topk,
                                  const float* selection_bias, bool sigmoid, float scale,
                                  int groups, int groups_used) {
    logits += static_cast<size_t>(blockIdx.x) * num_experts;
    topk_idx += static_cast<size_t>(blockIdx.x) * top_k;
    topk_weight += static_cast<size_t>(blockIdx.x) * top_k;
    extern __shared__ float dynamic_probs[];
    float static_probs[StaticExperts > 0 ? StaticExperts : 1];
    float* probs = StaticExperts > 0 ? static_probs : dynamic_probs;
    const int ne = StaticExperts > 0 ? min(num_experts, StaticExperts) : num_experts;
    if (sigmoid) {
        for (int e = 0; e < ne; ++e) probs[e] = 1.0f / (1.0f + expf(-logits[e]));
    } else {
        float max_l = -INFINITY;
        for (int e = 0; e < ne; ++e) max_l = fmaxf(max_l, logits[e]);
        float sum_e = 0.0f;
        for (int e = 0; e < ne; ++e) {
            probs[e] = expf(logits[e] - max_l);
            sum_e += probs[e];
        }
        for (int e = 0; e < ne; ++e) probs[e] /= sum_e;
    }

    const int group_width = ne / groups;
    bool selected_groups[128]{};
    if (groups_used < groups) {
        float group_scores[128];
        for (int g = 0; g < groups; ++g) {
            float first = -INFINITY, second = -INFINITY;
            for (int e = g * group_width; e < (g + 1) * group_width; ++e) {
                const float score = selection_bias ? probs[e] + selection_bias[e] : probs[e];
                second = fmaxf(second, fminf(first, score));
                first = fmaxf(first, score);
            }
            group_scores[g] = first + second;
        }
        for (int i = 0; i < groups_used; ++i) {
            int best = -1;
            for (int g = 0; g < groups; ++g) {
                if (!selected_groups[g] && (best < 0 || group_scores[g] > group_scores[best])) best = g;
            }
            selected_groups[best] = true;
        }
    }

    float weight_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int best_e = -1;
        float best_score = -INFINITY;
        for (int e = 0; e < ne; ++e) {
            if (groups_used < groups && !selected_groups[e / group_width]) continue;
            bool used = false;
            for (int prev = 0; prev < k; ++prev) {
                if (topk_idx[prev] == e) used = true;
            }
            const float score = selection_bias ? probs[e] + selection_bias[e] : probs[e];
            if (!used && (best_e < 0 || score > best_score)) {
                best_score = score;
                best_e = e;
            }
        }
        if (best_e < 0) break;
        topk_idx[k] = best_e;
        topk_weight[k] = probs[best_e];
        weight_sum += topk_weight[k];
    }
    if (norm_topk) {
        weight_sum = fmaxf(weight_sum, 0x1p-14f);
        for (int k = 0; k < top_k; ++k) topk_weight[k] /= weight_sum;
    }
    if (scale != 1.0f) {
        for (int k = 0; k < top_k; ++k) topk_weight[k] *= scale;
    }
}

template <int Format>
__device__ float moe_warp_dot(const float* x, const void* weights, int row, int cols) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int base = 0; base + lane < cols; base += 32) {
        float w;
        if constexpr (Format == -1) w = static_cast<const float*>(weights)[static_cast<int64_t>(row) * cols + base + lane];
        else w = dequant_weight_elem<Format>(weights, row, base, lane, cols);
        acc += x[base + lane] * w;
    }
    return warp_reduce_sum(acc);
}

template<class Codec>
__device__ float codec_warp_dot(const float* x, const uint8_t* weight, int row, int cols) {
    const int lane = threadIdx.x & 31;
    const int blocks = cols / Codec::kBlockSize;
    const auto* b = weight + static_cast<size_t>(row) * blocks * Codec::kBlockBytes;
    float sum = 0.0f;
    constexpr int lanes = Codec::kWarpBlockLanes;
    constexpr int step = 32 / lanes;
    const int lane_block = lane / lanes;
    for (int block = lane_block; block < blocks; block += step)
        sum += Codec::warp_dot_block(b + block * Codec::kBlockBytes,
                                     x + block * Codec::kBlockSize, lane % lanes);
    return warp_reduce_sum(sum);
}

template<class Codec>
__global__ void k_codec_gemv(const float* x, const uint8_t* w, float* out, int rows, int n, int k) {
    const int col = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (col >= n) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const float sum = codec_warp_dot<Codec>(x + static_cast<int64_t>(row) * k, w, col, k);
        if ((threadIdx.x & 31) == 0) out[static_cast<int64_t>(row) * n + col] = sum;
    }
}

template<class Codec>
__global__ void k_codec_gemv_q8_1(const BlockQ8_1* __restrict__ x, const uint8_t* __restrict__ w,
                                  float* __restrict__ out, int n, int k) {
    constexpr int groups = Codec::kBlockSize / 32;
    const int col = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (col >= n) return;
    const int lane = threadIdx.x & 31;
    constexpr int block_lanes = Codec::kInt8BlockLanes;
    const int word = lane % block_lanes;
    const int blocks = k / Codec::kBlockSize;
    const auto* qx = x + static_cast<int64_t>(blockIdx.y) * (k / 32);
    const auto* b = w + static_cast<int64_t>(col) * blocks * Codec::kBlockBytes;
    float sum = 0.0f;
    for (int block = lane / block_lanes; block < blocks; block += 32 / block_lanes) {
        const auto* packed = b + block * Codec::kBlockBytes;
        if constexpr (Codec::kHasWarpInt8Block) {
            const auto basis = Codec::load_int8_block(packed, word);
            float inner = 0.0f;
            #pragma unroll
            for (int sub = 0; sub < groups; ++sub) {
                const auto& a = qx[block * groups + sub];
                const int lo = __dp4a(static_cast<int>(basis.word(sub, word)), a.qs[word], 0);
                const int dot = __dp4a(static_cast<int>(basis.word(sub, word + 4)), a.qs[word + 4], lo);
                inner = fmaf(a.d, dot, inner);
            }
            sum = fmaf(basis.scale(), inner, sum);
        } else {
            #pragma unroll
            for (int sub = 0; sub < groups; ++sub) {
                const auto& a = qx[block * groups + sub];
                const int dot = __dp4a(static_cast<int>(Codec::int8_word(packed, sub, word)), a.qs[word], 0);
                sum = fmaf(Codec::int8_scale(packed, sub) * a.d, dot, sum);
            }
        }
    }
    const float result = warp_reduce_sum(sum);
    if (lane == 0) out[static_cast<int64_t>(blockIdx.y) * n + col] = result;
}

// Interleave matching Q8 words from the same group so GEMV can load all
// activation rows for one packed weight with a vector read.
template<int Rows>
__global__ void k_pack_verify_activations(const BlockQ8_1* x, int* y, int m, int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups * 9 * Rows) return;
    const int row = i % Rows, word = i / Rows % 9, group = i / (Rows * 9);
    if (row >= m) {
        y[i] = 0;
    } else {
        const auto& src = x[static_cast<int64_t>(row) * groups + group];
        y[i] = word < 8 ? src.qs[word] : __float_as_int(src.d);
    }
}

template<int Rows>
__global__ void k_quantize_verify_activations(const float* x, int* y, int m, int k) {
    constexpr int warps = 4;
    __shared__ BlockQ8_1 blocks[warps];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int group = blockIdx.x * warps + warp;
    if (group >= k / 32) return;
    const int row = blockIdx.y;
    auto& block = blocks[warp];
    const float value = row < m ? x[static_cast<int64_t>(row) * k + group * 32 + lane] : 0.f;
    store_q8_1(value, &block);
    __syncwarp();
    if (lane < 9) y[(group * 9 + lane) * Rows + row] = lane < 8 ? block.qs[lane] : __float_as_int(block.d);
}

template<int Rows>
__device__ __forceinline__ void load_verify_words(const int* p, int (&values)[Rows]) {
    if constexpr (Rows == 2) {
        const auto v = __ldg(reinterpret_cast<const int2*>(p));
        values[0] = v.x;
        values[1] = v.y;
    } else {
        #pragma unroll
        for (int i = 0; i < Rows; i += 4) {
            const auto v = __ldg(reinterpret_cast<const int4*>(p + i));
            values[i + 0] = v.x;
            values[i + 1] = v.y;
            values[i + 2] = v.z;
            values[i + 3] = v.w;
        }
    }
}

template<class Codec, int Rows>
__global__ void k_codec_verify_q8_1(const int* __restrict__ x, const uint8_t* __restrict__ w,
                                   float* __restrict__ out, int m, int n, int k) {
    constexpr int groups = Codec::kBlockSize / 32;
    constexpr int lanes = Codec::kInt8BlockLanes;
    const int col = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (col >= n) return;
    const int lane = threadIdx.x % 32, word = lane % lanes;
    const int blocks = k / Codec::kBlockSize;
    const auto* b = w + static_cast<int64_t>(col) * blocks * Codec::kBlockBytes;
    float sums[Rows] = {};
    for (int block = lane / lanes; block < blocks; block += 32 / lanes) {
        const auto* packed = b + block * Codec::kBlockBytes;
        if constexpr (Codec::kHasWarpInt8Block) {
            const auto basis = Codec::load_int8_block(packed, word);
            float inner[Rows] = {};
            #pragma unroll
            for (int sub = 0; sub < groups; ++sub) {
                const int lo = static_cast<int>(basis.word(sub, word));
                const int hi = static_cast<int>(basis.word(sub, word + 4));
                const int* a = x + (block * groups + sub) * 9 * Rows;
                int alo[Rows], ahi[Rows], scale[Rows];
                load_verify_words(a + word * Rows, alo);
                load_verify_words(a + (word + 4) * Rows, ahi);
                load_verify_words(a + 8 * Rows, scale);
                #pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    const int dot = __dp4a(hi, ahi[row], __dp4a(lo, alo[row], 0));
                    inner[row] = fmaf(__int_as_float(scale[row]), dot, inner[row]);
                }
            }
            const float scale = basis.scale();
            #pragma unroll
            for (int row = 0; row < Rows; ++row) sums[row] = fmaf(scale, inner[row], sums[row]);
        } else {
            #pragma unroll
            for (int sub = 0; sub < groups; ++sub) {
                const int q = static_cast<int>(Codec::int8_word(packed, sub, word));
                const float scale = Codec::int8_scale(packed, sub);
                const int* a = x + (block * groups + sub) * 9 * Rows;
                int aq[Rows], ascale[Rows];
                load_verify_words(a + word * Rows, aq);
                load_verify_words(a + 8 * Rows, ascale);
                #pragma unroll
                for (int row = 0; row < Rows; ++row)
                    sums[row] = fmaf(scale * __int_as_float(ascale[row]), __dp4a(q, aq[row], 0), sums[row]);
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        const float value = warp_reduce_sum(sums[row]);
        if (lane == 0 && row < m) out[static_cast<int64_t>(row) * n + col] = value;
    }
}

template<class Codec, bool Embedding>
__global__ void k_codec_unpack(const uint8_t* w, const int32_t* ids,
                               void* out, int rows, int cols, int first) {
    constexpr int warps = 8;
    const int block = blockIdx.x * warps + threadIdx.x / 32;
    const int row_blocks = cols / Codec::kBlockSize;
    if (block >= row_blocks) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int64_t src_row = Embedding ? ids[row] : static_cast<int64_t>(row) + first;
        const auto* b = w + (src_row * row_blocks + block) * Codec::kBlockBytes;
        const int64_t pos = static_cast<int64_t>(row) * cols + block * Codec::kBlockSize;
#pragma unroll
        for (int offset = 0; offset < Codec::kBlockSize; offset += 32) {
            const int i = offset + (threadIdx.x & 31);
            const float v = Codec::value(b, i);
            if constexpr (Embedding) static_cast<float*>(out)[pos + i] = v;
            else static_cast<half*>(out)[pos + i] = __float2half_rn(v);
        }
    }
}

template<bool CodecWeights>
__device__ float moe_warp_dot(const float* x, const void* w, DType type, int row, int cols) {
    if constexpr (CodecWeights) {
        switch (type) {
#define FUSIONRT_CODEC_CASE(name, id, block, bytes) case DType::name: \
    return codec_warp_dot<WeightCodec<DType::name>>(x, static_cast<const uint8_t*>(w), row, cols);
            FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_CODEC_CASE)
#undef FUSIONRT_CODEC_CASE
            default: break;
        }
    }
    switch (type) {
        case DType::F32: return moe_warp_dot<-1>(x, w, row, cols);
        case DType::F16: return moe_warp_dot<0>(x, w, row, cols);
        case DType::BF16: return moe_warp_dot<1>(x, w, row, cols);
        case DType::Q8_0: return moe_warp_dot<2>(x, w, row, cols);
        case DType::Q4_0: return moe_warp_dot<3>(x, w, row, cols);
        case DType::Q4_1: return moe_warp_dot<4>(x, w, row, cols);
        case DType::Q2_K: return moe_warp_dot<10>(x, w, row, cols);
        case DType::Q4_K: return moe_warp_dot<5>(x, w, row, cols);
        case DType::Q6_K: return moe_warp_dot<6>(x, w, row, cols);
        case DType::Q5_K: return moe_warp_dot<7>(x, w, row, cols);
        default: return 0.0f;
    }
}

template<bool CodecWeights, bool ExpertOps>
__global__ void k_moe_indexed_gate_up_swiglu(
    const float* __restrict__ x,
    const int* __restrict__ topk_idx,
    const void* gate_w, DType gate_type,
    const void* up_w, DType up_type,
    float* __restrict__ scratch_inter,
    int hidden, int inter, int top_k, int resident, int first_expert,
    bool geglu, const float* gate_scale, const float* up_scale)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_rows = top_k * inter;
    if (wid >= total_rows) return;

    int k_slot = wid / inter;
    int c = wid % inter;
    int e = topk_idx[k_slot] - first_expert;
    if (e < 0 || e >= resident) return;
    const int gate_row = (up_w ? e : 2 * e) * inter + c;
    const float g_acc = moe_warp_dot<CodecWeights>(x, gate_w, gate_type, gate_row, hidden);
    const float u_acc = up_w ? moe_warp_dot<CodecWeights>(x, up_w, up_type, gate_row, hidden) :
                               moe_warp_dot<CodecWeights>(x, gate_w, gate_type, gate_row + inter, hidden);
    if (lane == 0) {
        float gate = g_acc, up = u_acc;
        if constexpr (ExpertOps) {
            if (gate_scale) gate *= gate_scale[e + first_expert];
            if (up_scale) up *= up_scale[e + first_expert];
        }
        scratch_inter[k_slot * inter + c] = ExpertOps && geglu ?
            gated_activation<true>(gate, up) : gated_activation<false>(gate, up);
    }
}

template<bool CodecWeights>
__global__ void k_moe_indexed_down_combine(
    const float* __restrict__ scratch_inter,
    const int* __restrict__ topk_idx,
    const void* __restrict__ down_w, DType down_type,
    const float* __restrict__ topk_weight,
    float* __restrict__ out,
    int hidden, int inter, int top_k, int resident, int first_expert)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= hidden) return;

    float total = 0.0f;
    for (int k_slot = 0; k_slot < top_k; ++k_slot) {
        int e = topk_idx[k_slot] - first_expert;
        if (e < 0 || e >= resident) continue;
        total += moe_warp_dot<CodecWeights>(scratch_inter + k_slot * inter, down_w, down_type,
                              e * hidden + wid, inter) * topk_weight[k_slot];
    }
    if (lane == 0) out[wid] = total;
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
    if (tid < 0) {
        out[idx] = 0.0f;
        return;
    }

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
    } else if (dtype == static_cast<int>(DType::Q2_K)) {
        val = dequant_weight_elem<10>(table, tid, c & ~31, c & 31, hidden);
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
    const auto index = (static_cast<int64_t>(row) * blocks_per_row + col) * 256 + lane;
    const int token = token_ids[row];
    if (token < 0) {
        out[index] = 0.0f;
        return;
    }
    const auto& b = table[static_cast<int64_t>(token) * blocks_per_row + col];
    float val;
    if constexpr (std::is_same_v<Block, BlockQ8_A>) {
        val = half_bits_to_f32(b.d[lane >> 5]) * reinterpret_cast<const int8_t*>(b.qs)[lane];
    } else {
        val = repacked_k_value(b, lane);
    }
    out[index] = val;
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

template <bool Gelu>
__global__ void k_gated_interleaved_q8_1(const float* gate_up, BlockQ8_1* out, int inter_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= inter_dim) return;
    int row = blockIdx.y;
    const float* gate = gate_up + static_cast<int64_t>(row) * 2 * inter_dim;
    store_q8_1(gated_activation<Gelu>(gate[col], gate[inter_dim + col]),
               out + (static_cast<int64_t>(row) * inter_dim + col) / 32);
}

__global__ void k_quantize_vector_q8_1(const float* x, BlockQ8_1* out, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < K) store_q8_1(x[col], out + col / 32);
}

}  // namespace

void geglu_interleaved_q8_1(const float* gate_up, void* q8_out, int M, int inter_dim, cudaStream_t stream) {
    const dim3 grid((inter_dim + 255) / 256, M);
    k_gated_interleaved_q8_1<true><<<grid, 256, 0, stream>>>(gate_up, static_cast<BlockQ8_1*>(q8_out), inter_dim);
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
    k_gated_interleaved_q8_1<false><<<dim3((inter_dim + threads - 1) / threads, M), threads, 0, stream>>>(
        gate_up, static_cast<BlockQ8_1*>(q8_out), inter_dim);
}

void moe_topk_router_f32(const float* logits, int* topk_idx, float* topk_weight,
                         int num_experts, int top_k, bool norm_topk, cudaStream_t stream,
                         const float* selection_bias, bool sigmoid, float scale, int groups, int groups_used, int rows) {
    if (rows <= 0 || num_experts <= 0 || static_cast<size_t>(num_experts) * sizeof(float) > 48 * 1024 || top_k <= 0 || groups <= 0 || groups > 128 ||
        groups_used <= 0 || groups_used > groups || num_experts % groups != 0 ||
        groups_used * (num_experts / groups) < top_k) {
        throw std::invalid_argument("Invalid CUDA MoE router dimensions");
    }
    constexpr int local_capacity = 256;
    if (num_experts <= local_capacity) {
        k_moe_topk_router<local_capacity><<<rows, 1, 0, stream>>>(logits, topk_idx, topk_weight, num_experts,
            top_k, norm_topk, selection_bias, sigmoid, scale, groups, groups_used);
    } else {
        k_moe_topk_router<0><<<rows, 1, num_experts * sizeof(float), stream>>>(logits, topk_idx, topk_weight,
            num_experts, top_k, norm_topk, selection_bias, sigmoid, scale, groups, groups_used);
    }
}

void moe_indexed_swiglu_gemv(
    const float* x, const int* topk_idx, const void* gate_w, DType gate_type,
    const void* up_w, DType up_type, const void* down_w, DType down_type,
    const float* topk_weight, float* out, float* scratch_inter,
    int hidden, int inter, int top_k, int resident, cudaStream_t stream, int first_expert,
    bool geglu, const float* gate_scale, const float* up_scale)
{
    constexpr int WARPS = 8;
    int total_up_rows = top_k * inter;
    int up_blocks = (total_up_rows + WARPS - 1) / WARPS;
    auto gate_up = [&]<bool CodecWeights, bool ExpertOps>() {
        k_moe_indexed_gate_up_swiglu<CodecWeights, ExpertOps><<<up_blocks, WARPS * 32, 0, stream>>>(
            x, topk_idx, gate_w, gate_type, up_w, up_type, scratch_inter, hidden, inter, top_k, resident,
            first_expert, geglu, gate_scale, up_scale);
    };
    const bool codec = is_weight_codec_dtype(gate_type) || (up_w && is_weight_codec_dtype(up_type));
    const bool transforms = geglu || gate_scale || up_scale;
    if (codec) {
        if (transforms) gate_up.template operator()<true, true>();
        else gate_up.template operator()<true, false>();
    } else {
        if (transforms) gate_up.template operator()<false, true>();
        else gate_up.template operator()<false, false>();
    }

    const int down_blocks = (hidden + WARPS - 1) / WARPS;
    auto down = [&]<bool CodecWeights>() {
        k_moe_indexed_down_combine<CodecWeights><<<down_blocks, WARPS * 32, 0, stream>>>(
            scratch_inter, topk_idx, down_w, down_type, topk_weight, out, hidden, inter, top_k, resident, first_expert);
    };
    if (is_weight_codec_dtype(down_type)) down.template operator()<true>();
    else down.template operator()<false>();
}

void codec_gemv_f32(const float* x, const void* weight, DType dtype, float* out,
                     int rows, int out_cols, int in_cols, cudaStream_t stream) {
    (void)row_nbytes(dtype, in_cols);
    constexpr int warps = 4;
    const dim3 blocks((out_cols - 1) / warps + 1, std::min(rows, 65535));
    visit_weight_codec(dtype, [&]<class Codec>() {
        k_codec_gemv<Codec><<<blocks, warps * 32, 0, stream>>>(
            x, static_cast<const uint8_t*>(weight), out, rows, out_cols, in_cols);
    });
}

void codec_dequant_tile_f16(const void* weight, DType dtype, uint16_t* out,
                             int first_row, int rows, int cols, cudaStream_t stream) {
    (void)row_nbytes(dtype, cols);
    visit_weight_codec(dtype, [&]<class Codec>() {
        const dim3 blocks((cols / Codec::kBlockSize + 7) / 8, std::min(rows, 65535));
        k_codec_unpack<Codec, false><<<blocks, 256, 0, stream>>>(
            static_cast<const uint8_t*>(weight), nullptr, out, rows, cols, first_row);
    });
}

void codec_gemv_q8_1(const void* x, const void* weight, DType dtype, float* out,
                      int M, int N, int K, cudaStream_t stream, int* verification_scratch,
                      const float* quantize_rows) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    visit_weight_codec(dtype, [&]<class Codec>() {
        if constexpr (Codec::kHasInt8Sub32) {
            if (K % Codec::kBlockSize) throw std::invalid_argument("Invalid WeightCodec GEMV column count");
            constexpr int warps = 4;
            const auto* qx = static_cast<const BlockQ8_1*>(x);
            const auto* w = static_cast<const uint8_t*>(weight);
            const int blocks = (N + warps - 1) / warps;
            if (M >= 2 && M <= 8) {
                if (!verification_scratch) throw std::invalid_argument("Multi-row WeightCodec GEMV requires scratch");
                auto launch = [&]<int Rows>() {
                    constexpr int threads = 128;
                    const int elements = K / 32 * 9 * Rows;
                    if (quantize_rows) {
                        const dim3 grid((K + threads - 1) / threads, Rows);
                        k_quantize_verify_activations<Rows><<<grid, threads, 0, stream>>>(
                            quantize_rows, verification_scratch, M, K);
                    } else {
                        k_pack_verify_activations<Rows><<<(elements + threads - 1) / threads, threads, 0, stream>>>(
                            qx, verification_scratch, M, K / 32);
                    }
                    k_codec_verify_q8_1<Codec, Rows><<<blocks, warps * 32, 0, stream>>>(
                        verification_scratch, w, out, M, N, K);
                };
                if (M == 2) launch.template operator()<2>();
                else if (M <= 4) launch.template operator()<4>();
                else launch.template operator()<8>();
                return;
            }
            const dim3 grid(blocks, M);
            k_codec_gemv_q8_1<Codec><<<grid, warps * 32, 0, stream>>>(qx, w, out, N, K);
        } else throw std::invalid_argument("WeightCodec has no INT8 GEMV basis");
    });
}

void embed_lookup_f32(const void* emb_table, DType table_dtype,
                      const int32_t* token_ids_dev, float* out,
                      int M, int hidden, cudaStream_t stream) {
    if (is_weight_codec_dtype(table_dtype)) {
        (void)row_nbytes(table_dtype, hidden);
        visit_weight_codec(table_dtype, [&]<class Codec>() {
            const dim3 blocks((hidden / Codec::kBlockSize + 7) / 8, std::min(M, 65535));
            k_codec_unpack<Codec, true><<<blocks, 256, 0, stream>>>(
                static_cast<const uint8_t*>(emb_table), token_ids_dev, out, M, hidden, 0);
        });
        return;
    }
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

__global__ void k_scale_moe_routes(float* weights, const int* indices, const float* scales, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) weights[i] *= scales[indices[i]];
}

void scale_moe_routes_f32(float* weights, const int* indices, const float* scales, int n, cudaStream_t stream) {
    k_scale_moe_routes<<<(n + 255) / 256, 256, 0, stream>>>(weights, indices, scales, n);
}

}  // namespace fusionrt::cuda_ops
