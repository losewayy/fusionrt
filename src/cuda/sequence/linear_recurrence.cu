#include "../cuda_common.cuh"

#include <algorithm>

namespace fusionrt::cuda_ops {

namespace {

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

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

}  // namespace

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
