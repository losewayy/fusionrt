#include "epilogue_common.cuh"

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

__global__ void k_vision_activation(float* x, int count, bool gelu, bool quick) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float t = x[i];
        x[i] = quick ? t / (1.0f + expf(-1.702f * t)) :
               gelu ? gated_activation<true>(t, 1.f) : silu_f(t);
    }
}

template <bool Gelu>
__global__ void k_gated_f32(const float* gate, const float* up,
                             float* out, int n) {
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

__global__ void k_add_residual_f32(const float* a, const float* b,
                                   float* out, int n) {
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
    int32_t best_i = -1;

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
        best_i = (lane < nwarps) ? si[lane] : -1;
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

__device__ __forceinline__ void kv_quantize_q4(float value, BlockQ4_0& block, int lane) {
    constexpr unsigned mask = 0xffffffffu;
    const int magnitude = __float_as_int(fabsf(value));
    const int amax = __reduce_max_sync(mask, magnitude);
    const int peak_lane = __ffs(__ballot_sync(mask, magnitude == amax)) - 1;
    const float d = __shfl_sync(mask, value, peak_lane) * (-1.0f / 8.0f);
    const float scaled = d != 0.0f ? __fdividef(value, d) : 0.0f;
    const int q = min(15, max(0, __float2int_rz(scaled + 8.5f)));
    const int hi = __shfl_xor_sync(mask, q, 16);
    if (lane < 16) block.qs[lane] = static_cast<uint8_t>(q | (hi << 4));
    if (lane == 0) block.d = __half_as_ushort(__float2half_rn(d));
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
    int pos = (pos_dev ? *pos_dev : pos_host) + m;
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
    auto rows = [&]<bool Value>(Cache* contiguous) {
        if constexpr (std::is_same_v<Cache, BlockQ8_0>)
            return detail::k8v4_row<Value>(cache, pos, kh, num_kv_heads, head_dim);
        else
            return kv_row<Value>(contiguous, cache, pos, kh, num_kv_heads, head_dim);
    };
    auto* keys = rows.template operator()<false>(k_cache);
    auto* values = rows.template operator()<true>(v_cache);
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
        } else if constexpr (std::is_same_v<Cache, BlockQ8_0>) {
            const int lane = i % 32;
            auto& key = keys[i / 32];
            kv_quantize(h_buf[i], key.qs, reinterpret_cast<half*>(&key.d), lane);
            kv_quantize_q4(value, values[i / 32], lane);
        } else {
            keys[i] = h_buf[i];
            values[i] = value;
        }
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
                                 int64_t count, int width, int stride, int offset) {
    const int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = gate[i];
        const float gelu = (0.5f * x) * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
        out[i] = gelu * inputs[(i / width) * int64_t{stride} + offset + i % width];
    }
}

void gemma_ple_gate_f32(const float* gate, const float* inputs, float* out,
                       int rows, int width, int input_stride, int input_offset, cudaStream_t stream) {
    const int64_t count = int64_t{rows} * width;
    k_gemma_ple_gate<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        gate, inputs, out, count, width, input_stride, input_offset);
}

void geglu_interleaved_f32(const float* gate_up, float* out, int rows, int inter_dim, cudaStream_t stream) {
    const int total = rows * inter_dim;
    k_gated_interleaved_f32<true><<<(total + 255) / 256, 256, 0, stream>>>(gate_up, out, rows, inter_dim);
}

void swiglu_f32(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    k_gated_f32<false><<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

void swiglu_interleaved_f32(const float* gate_up, float* out, int rows, int inter_dim, cudaStream_t stream) {
    int total = rows * inter_dim;
    k_gated_interleaved_f32<false><<<(total + 255) / 256, 256, 0, stream>>>(gate_up, out, rows, inter_dim);
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

__global__ void k_argmax_mass(const float* x, int n, const int32_t* idx, float* parts) {
    __shared__ float reduce[32];
    const int best = *idx;
    const float mx = best >= 0 ? x[best] : -INFINITY;
    float mass = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
        if (best >= 0 && !isnan(v)) mass += isinf(mx) ? float(v == mx) : expf(v - mx);
    }
    const float sum = block_reduce_sum(mass, reduce);
    if (threadIdx.x == 0) parts[blockIdx.x] = sum;
}

__global__ void k_argmax_probability(const float* parts, float* probability) {
    __shared__ float reduce[32];
    const float total = block_reduce_sum(parts[threadIdx.x], reduce);
    if (threadIdx.x == 0) *probability = total > 0.f ? 1.f / total : 0.f;
}

void argmax_probability_f32(const float* x, int n, const int32_t* idx, float* probability,
                            float* scratch, cudaStream_t stream) {
    k_argmax_mass<<<kArgmaxParts, 128, 0, stream>>>(x, n, idx, scratch);
    k_argmax_probability<<<1, kArgmaxParts, 0, stream>>>(scratch, probability);
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
    if (kv_dtype == DType::K8V4) launch.template operator()<BlockQ8_0>();
    else if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void batched_qk_norm_rope_kv_append_f32(
    float* q, float* k, const float* v,
    const float* q_norm_w, const float* k_norm_w,
    void* k_cache, void* v_cache,
    int M, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    int start_pos, int /*max_seq*/, float theta, float eps, bool neox, cudaStream_t stream,
    int q_row_stride, int k_row_stride, int v_row_stride, const float* rotary_factors, DType kv_dtype,
    CudaKvView cache, bool v_norm, const int* start_dev)
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
                num_q_heads, num_kv_heads, head_dim, rot_dim, start_dev, start_pos, theta, eps, neox, qs, ks, vs, cache);
        };
        if (v_norm) run.template operator()<true>();
        else run.template operator()<false>();
    };
    if (kv_dtype == DType::K8V4) launch.template operator()<BlockQ8_0>();
    else if (kv_dtype == DType::Q8_0) launch.template operator()<int8_t>();
    else if (kv_dtype == DType::F16) launch.template operator()<half>();
    else launch.template operator()<float>();
}

void split_q_gate_f32(const float* q_full, float* q_out, float* gate_out,
                      int M, int num_q_heads, int head_dim, cudaStream_t stream, int src_row_stride) {
    int total = M * num_q_heads * head_dim;
    int srs = (src_row_stride > 0) ? src_row_stride : (2 * num_q_heads * head_dim);
    k_split_q_gate_f32<<<(total + 255) / 256, 256, 0, stream>>>(q_full, q_out, gate_out, M, num_q_heads, head_dim, srs);
}

__global__ void k_apply_row_sigmoid_gate(float* values, const float* gate, int n, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] /= 1.f + expf(-gate[i / cols]);
}

void apply_row_sigmoid_gate_f32(float* values, const float* gate, int rows, int cols, cudaStream_t stream) {
    const int n = rows * cols;
    k_apply_row_sigmoid_gate<<<(n + 255) / 256, 256, 0, stream>>>(values, gate, n, cols);
}

void apply_sigmoid_gate_f32(float* attn_out, const float* gate, int n, cudaStream_t stream) {
    k_apply_sigmoid_gate_f32<<<(n + 255) / 256, 256, 0, stream>>>(attn_out, gate, n);
}

}  // namespace fusionrt::cuda_ops
