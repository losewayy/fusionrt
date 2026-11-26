#include "../matmul/weight_tiles.cuh"

#include <algorithm>

namespace fusionrt::cuda_ops {

namespace {

__global__ void k_layernorm(const float* x, const float* weight, const float* bias, float* out,
                            int cols, float eps) {
    __shared__ float reduce[32];
    const auto* src = x + blockIdx.x * cols;
    auto* dst = out + blockIdx.x * cols;
    float sum = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += src[i];
    const float mean = block_reduce_sum(sum, reduce) / cols;
    __syncthreads();
    float sq = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float d = src[i] - mean;
        sq = fmaf(d, d, sq);
    }
    const float inv = rsqrtf(block_reduce_sum(sq, reduce) / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dst[i] = (src[i] - mean) * inv * weight[i] + (bias ? bias[i] : 0.0f);
}

__global__ void k_vision_merge(const float* x, float* out, int count, int width, int cols, int merge, bool average) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int stride = cols * (average ? 1 : merge * merge);
    const int group = i / stride, lane = i % stride;
    const int base = (group / (width / merge) * width + group % (width / merge)) * merge;
    if (average) {
        float sum = 0;
        for (int y = 0; y < merge; ++y)
            for (int xx = 0; xx < merge; ++xx) sum += x[(base + y * width + xx) * cols + lane];
        out[i] = sum / (merge * merge);
    } else {
        const int patch = lane / cols;
        out[i] = x[(base + patch / merge * width + patch % merge) * cols + lane % cols];
    }
}

__global__ void k_rope_2d(float* x, int width, int heads, int half_dim, float theta, bool vision_layout) {
    const int patch = blockIdx.x / heads, pair = threadIdx.x;
    const int quarter = half_dim / 2;
    const bool row_axis = pair >= quarter;
    const int pos = row_axis != vision_layout ? patch / width : patch % width;
    const float angle = pos * powf(theta, -static_cast<float>(pair % quarter) / quarter);
    float s, c;
    sincosf(angle, &s, &c);
    const int lo = vision_layout ? pair : pair + (row_axis ? quarter : 0);
    const int hi = lo + (vision_layout ? half_dim : quarter);
    auto* v = x + blockIdx.x * (2 * half_dim);
    const float a = v[lo], b = v[hi];
    v[lo] = a * c - b * s;
    v[hi] = a * s + b * c;
}

__global__ void k_clamp(const float* x, float* out, int count, float lo, float hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = fminf(hi, fmaxf(lo, x[i]));
}

__global__ void k_vision_standardize(float* x, const float* bias, const float* scale, int count, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = (x[i] - bias[i % cols]) * scale[i % cols];
}

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

template <bool Gelu>
__device__ __forceinline__ float gated_activation(float gate, float up) {
    if constexpr (Gelu)
        return (0.5f * gate) * (1.0f + tanhf(0.7978845608028654f * gate * (1.0f + 0.044715f * gate * gate))) * up;
    else
        return silu_f(gate) * up;
}

__global__ void k_vision_activation(float* x, int count, bool gelu, bool quick) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float t = x[i];
        x[i] = quick ? t / (1.0f + expf(-1.702f * t)) :
               gelu ? gated_activation<true>(t, 1.f) : silu_f(t);
    }
}

template <bool Gelu>
__global__ void k_gated_f32(const float* __restrict__ gate, const float* __restrict__ up,
                             float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = gated_activation<Gelu>(gate[idx], up[idx]);
}

template <bool Gelu>
__global__ void k_gated_interleaved_f32(const float* __restrict__ gate_up, float* __restrict__ out,
                                         int rows, int inter_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * inter_dim;
    if (idx >= total) return;
    int r = idx / inter_dim;
    int c = idx % inter_dim;
    float g = gate_up[r * (2 * inter_dim) + c];
    float u = gate_up[r * (2 * inter_dim) + inter_dim + c];
    out[idx] = gated_activation<Gelu>(g, u);
}

template <bool Softcap>
__global__ void k_scale_f32(float* x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if constexpr (Softcap) x[i] = scale * tanhf(x[i] / scale);
        else x[i] *= scale;
    }
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

template <typename Cache, bool ValueNorm>
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
    const auto* src_v = ValueNorm && !v ? vec : v + m * v_stride + kh * head_dim;
    float v_inv = 1.0f;
    if constexpr (ValueNorm) {
        float v_sq = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) v_sq += src_v[i] * src_v[i];
        v_inv = rsqrtf(block_reduce_sum(v_sq, red_buf) / static_cast<float>(head_dim) + eps);
    }
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float value = ValueNorm ? src_v[i] * v_inv : src_v[i];
        vec[i] = h_buf[i];
        if constexpr (std::is_same_v<Cache, int8_t>) {
            kv_quantize(h_buf[i], keys, kv_scales<false>(cache, pos, kh, num_kv_heads, head_dim), i);
            kv_quantize(value, values, kv_scales<true>(cache, pos, kh, num_kv_heads, head_dim), i);
        } else {
            keys[i] = h_buf[i];
            values[i] = value;
        }
    }
}

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

__device__ float moe_warp_dot(const float* x, const void* w, DType type, int row, int cols) {
    switch (type) {
        case DType::F32: return moe_warp_dot<-1>(x, w, row, cols);
        case DType::F16: return moe_warp_dot<0>(x, w, row, cols);
        case DType::BF16: return moe_warp_dot<1>(x, w, row, cols);
        case DType::Q8_0: return moe_warp_dot<2>(x, w, row, cols);
        case DType::Q4_0: return moe_warp_dot<3>(x, w, row, cols);
        case DType::Q4_1: return moe_warp_dot<4>(x, w, row, cols);
        case DType::Q4_K: return moe_warp_dot<5>(x, w, row, cols);
        case DType::Q6_K: return moe_warp_dot<6>(x, w, row, cols);
        case DType::Q5_K: return moe_warp_dot<7>(x, w, row, cols);
        default: return 0.0f;
    }
}

__global__ void k_moe_indexed_gate_up_swiglu(
    const float* __restrict__ x,
    const int* __restrict__ topk_idx,
    const void* gate_w, DType gate_type,
    const void* up_w, DType up_type,
    float* __restrict__ scratch_inter,
    int hidden, int inter, int top_k, int resident)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_rows = top_k * inter;
    if (wid >= total_rows) return;

    int k_slot = wid / inter;
    int c = wid % inter;
    int e = topk_idx[k_slot];
    if (e < 0 || e >= resident) return;
    const int gate_row = (up_w ? e : 2 * e) * inter + c;
    const float g_acc = moe_warp_dot(x, gate_w, gate_type, gate_row, hidden);
    const float u_acc = up_w ? moe_warp_dot(x, up_w, up_type, gate_row, hidden) :
                               moe_warp_dot(x, gate_w, gate_type, gate_row + inter, hidden);
    if (lane == 0) {
        scratch_inter[k_slot * inter + c] = silu_f(g_acc) * u_acc;
    }
}

__global__ void k_moe_indexed_down_combine(
    const float* __restrict__ scratch_inter,
    const int* __restrict__ topk_idx,
    const void* __restrict__ down_w, DType down_type,
    const float* __restrict__ topk_weight,
    float* __restrict__ out,
    int hidden, int inter, int top_k, int resident)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= hidden) return;

    float total = 0.0f;
    for (int k_slot = 0; k_slot < top_k; ++k_slot) {
        int e = topk_idx[k_slot];
        if (e < 0 || e >= resident) continue;
        total += moe_warp_dot(scratch_inter + k_slot * inter, down_w, down_type,
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
    if constexpr (std::is_same_v<Block, BlockQ8_A>) {
        val = half_bits_to_f32(b.d[lane >> 5]) * reinterpret_cast<const int8_t*>(b.qs)[lane];
    } else {
        val = repacked_k_value(b, lane);
    }
    out[(static_cast<int64_t>(row) * blocks_per_row + col) * 256 + lane] = val;
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

void layernorm_f32(const float* x, const float* weight, const float* bias, float* out,
                   int rows, int cols, float eps, cudaStream_t stream) {
    const int threads = std::min(256, ((cols + 31) / 32) * 32);
    k_layernorm<<<rows, threads, 0, stream>>>(x, weight, bias, out, cols, eps);
}

void vision_merge_f32(const float* x, float* out, int height, int width, int cols, int merge,
                      bool average, cudaStream_t stream) {
    const int count = height / merge * (width / merge) * cols * (average ? 1 : merge * merge);
    k_vision_merge<<<(count + 255) / 256, 256, 0, stream>>>(x, out, count, width, cols, merge, average);
}

void rope_2d_f32(float* x, int height, int width, int heads, int dim, float theta,
                 cudaStream_t stream, bool vision_layout) {
    k_rope_2d<<<height * width * heads, dim / 2, 0, stream>>>(x, width, heads, dim / 2, theta, vision_layout);
}

void vision_activation_f32(float* x, int count, bool gelu, bool quick_gelu, cudaStream_t stream) {
    k_vision_activation<<<(count + 255) / 256, 256, 0, stream>>>(x, count, gelu, quick_gelu);
}

void clamp_f32(const float* x, float* out, int count, float lo, float hi, cudaStream_t stream) {
    k_clamp<<<(count + 255) / 256, 256, 0, stream>>>(x, out, count, lo, hi);
}

void vision_standardize_f32(float* x, const float* bias, const float* scale, int rows, int cols, cudaStream_t stream) {
    const int count = rows * cols;
    k_vision_standardize<<<(count + 255) / 256, 256, 0, stream>>>(x, bias, scale, count, cols);
}

void rmsnorm_f32(const float* x, const float* weight, float* y,
                 int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((cols + 31) / 32) * 32));
    k_rmsnorm_f32<<<rows, threads, 0, stream>>>(x, weight, y, cols, eps);
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

void scale_f32(float* x, int n, float scale, cudaStream_t stream) {
    k_scale_f32<false><<<(n + 255) / 256, 256, 0, stream>>>(x, n, scale);
}

void softcap_f32(float* x, int n, float cap, cudaStream_t stream) {
    k_scale_f32<true><<<(n + 255) / 256, 256, 0, stream>>>(x, n, cap);
}

void geglu_f32(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    k_gated_f32<true><<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

__global__ void k_gemma_ple_gate(const float* gate, const float* inputs, float* out,
                                 int count, int width, int stride, int offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = gate[i];
        const float gelu = (0.5f * x) * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
        out[i] = gelu * inputs[(i / width) * stride + offset + i % width];
    }
}

void gemma_ple_gate_f32(const float* gate, const float* inputs, float* out,
                       int rows, int width, int input_stride, int input_offset, cudaStream_t stream) {
    const int count = rows * width;
    k_gemma_ple_gate<<<(count + 255) / 256, 256, 0, stream>>>(
        gate, inputs, out, count, width, input_stride, input_offset);
}

void geglu_interleaved_f32(const float* gate_up, float* out, int rows, int inter_dim, cudaStream_t stream) {
    const int total = rows * inter_dim;
    k_gated_interleaved_f32<true><<<(total + 255) / 256, 256, 0, stream>>>(gate_up, out, rows, inter_dim);
}

void geglu_interleaved_q8_1(const float* gate_up, void* q8_out, int M, int inter_dim, cudaStream_t stream) {
    const dim3 grid((inter_dim + 255) / 256, M);
    k_gated_interleaved_q8_1<true><<<grid, 256, 0, stream>>>(gate_up, static_cast<BlockQ8_1*>(q8_out), inter_dim);
}

void swiglu_f32(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    k_gated_f32<false><<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

void swiglu_interleaved_f32(const float* gate_up, float* out, int rows, int inter_dim, cudaStream_t stream) {
    int total = rows * inter_dim;
    k_gated_interleaved_f32<false><<<(total + 255) / 256, 256, 0, stream>>>(gate_up, out, rows, inter_dim);
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

void add_rmsnorm_f32(const float* a, const float* b, const float* norm_w,
                     float* res_out, float* norm_out, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((cols + 31) / 32) * 32));
    k_add_rmsnorm_f32<<<rows, threads, 0, stream>>>(a, b, norm_w, res_out, norm_out, cols, eps);
}

void fused_qk_norm_rope_kv_append_f32(
    float* q, float* k, const float* v,
    const float* q_norm_w, const float* k_norm_w,
    void* k_cache, void* v_cache,
    int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    const int* pos_dev, int pos_host, int /*max_seq*/,
    float theta, float eps, bool neox, cudaStream_t stream, const float* rotary_factors, DType kv_dtype, CudaKvView cache, bool v_norm)
{
    int total_heads = num_q_heads + num_kv_heads;
    int threads = std::min(256, std::max(32, ((head_dim + 31) / 32) * 32));
    size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);
    auto launch = [&]<typename Cache>() {
        auto run = [&]<bool ValueNorm>() {
            k_qk_norm_rope_kv_append<Cache, ValueNorm><<<total_heads, threads, smem_bytes, stream>>>(
                q, k, v, q_norm_w, k_norm_w, rotary_factors, static_cast<Cache*>(k_cache), static_cast<Cache*>(v_cache),
                num_q_heads, num_kv_heads, head_dim, rot_dim, pos_dev, pos_host, theta, eps, neox,
                num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim, cache);
        };
        if (v_norm) run.template operator()<true>();
        else run.template operator()<false>();
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void moe_topk_router_f32(const float* logits, int* topk_idx, float* topk_weight,
                         int num_experts, int top_k, bool norm_topk, cudaStream_t stream) {
    k_moe_topk_router<<<1, 1, 0, stream>>>(logits, topk_idx, topk_weight, num_experts, top_k, norm_topk);
}

void moe_indexed_swiglu_gemv(
    const float* x, const int* topk_idx, const void* gate_w, DType gate_type,
    const void* up_w, DType up_type, const void* down_w, DType down_type,
    const float* topk_weight, float* out, float* scratch_inter,
    int hidden, int inter, int top_k, int resident, cudaStream_t stream)
{
    constexpr int WARPS = 8;
    int total_up_rows = top_k * inter;
    int up_blocks = (total_up_rows + WARPS - 1) / WARPS;
    k_moe_indexed_gate_up_swiglu<<<up_blocks, WARPS * 32, 0, stream>>>(
        x, topk_idx, gate_w, gate_type, up_w, up_type, scratch_inter, hidden, inter, top_k, resident);

    int down_blocks = (hidden + WARPS - 1) / WARPS;
    k_moe_indexed_down_combine<<<down_blocks, WARPS * 32, 0, stream>>>(
        scratch_inter, topk_idx, down_w, down_type, topk_weight, out, hidden, inter, top_k, resident);
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
    int q_row_stride, int k_row_stride, int v_row_stride, const float* rotary_factors, DType kv_dtype, CudaKvView cache, bool v_norm)
{
    int total_heads = num_q_heads + num_kv_heads;
    int threads = std::min(256, std::max(32, ((head_dim + 31) / 32) * 32));
    size_t smem_bytes = (32 + static_cast<size_t>(head_dim)) * sizeof(float);
    dim3 grid(total_heads, M);
    int qs = (q_row_stride > 0) ? q_row_stride : (num_q_heads * head_dim);
    int ks = k_row_stride > 0 ? k_row_stride : num_kv_heads * head_dim;
    int vs = v_row_stride > 0 ? v_row_stride : ks;
    auto launch = [&]<typename Cache>() {
        auto run = [&]<bool ValueNorm>() {
            k_qk_norm_rope_kv_append<Cache, ValueNorm><<<grid, threads, smem_bytes, stream>>>(
                q, k, v, q_norm_w, k_norm_w, rotary_factors, static_cast<Cache*>(k_cache), static_cast<Cache*>(v_cache),
                num_q_heads, num_kv_heads, head_dim, rot_dim, nullptr, start_pos, theta, eps, neox, qs, ks, vs, cache);
        };
        if (v_norm) run.template operator()<true>();
        else run.template operator()<false>();
    };
    if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
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

}  // namespace fusionrt::cuda_ops
