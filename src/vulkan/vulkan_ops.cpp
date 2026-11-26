#include "fusionrt/vulkan_kernels.h"
#include <cmath>
#include <cstring>

namespace fusionrt::vk_ops {

namespace {

template <typename T>
const void* store_push(std::vector<std::vector<uint8_t>>& storage, const T& val) {
    storage.emplace_back(sizeof(T));
    std::memcpy(storage.back().data(), &val, sizeof(T));
    return storage.back().data();
}

}  // namespace

double gemv_vec4(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int M, int N, int K) {
    PushGemm pc{M, N, K};
    VkBatchDispatch d;
    d.shader_name = "gemv_vec4.comp";
    d.gx = static_cast<uint32_t>(N);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {x, w, y};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

void append_linear_dispatch(std::vector<VkBatchDispatch>& ops,
                            const Tensor& x, const Tensor& w, Tensor& y,
                            int M, int N, int K,
                            std::vector<std::vector<uint8_t>>& push_storage) {
    PushGemm pc{M, N, K};
    std::string shader = "gemv_vec4.comp";
    if (w.dtype == DType::Q8_0) shader = "gemv_q8_0.comp";
    else if (w.dtype == DType::Q4_0) shader = "gemv_q4_0.comp";
    else if (M > 8 || (K % 4) != 0) shader = "gemm_tiled.comp";

    if (shader == "gemm_tiled.comp") {
        ops.push_back({shader,
                       static_cast<uint32_t>((N + 15) / 16),
                       static_cast<uint32_t>((M + 15) / 16), 1,
                       {x, w, y},
                       store_push(push_storage, pc), sizeof(pc)});
    } else {
        ops.push_back({shader,
                       static_cast<uint32_t>(N), 1, 1,
                       {x, w, y},
                       store_push(push_storage, pc), sizeof(pc)});
    }
}

void append_rmsnorm_linear_dispatch(std::vector<VkBatchDispatch>& ops,
                                    const Tensor& x, const Tensor& norm_w, const Tensor& w,
                                    Tensor& x_norm_scratch, Tensor& y,
                                    int M, int N, int K, float eps,
                                    std::vector<std::vector<uint8_t>>& push_storage) {
    if (w.dtype == DType::F32 && M <= 8 && (K % 4) == 0) {
        PushFusedNormGemv pc{M, N, K, eps};
        ops.push_back({"fused_rmsnorm_gemv_vec4.comp",
                       static_cast<uint32_t>(N), 1, 1,
                       {x, norm_w, w, y},
                       store_push(push_storage, pc), sizeof(pc)});
    } else {
        PushRmsNorm npc{M, K, eps};
        ops.push_back({"rmsnorm.comp",
                       static_cast<uint32_t>(M), 1, 1,
                       {x, norm_w, x_norm_scratch},
                       store_push(push_storage, npc), sizeof(npc)});
        append_linear_dispatch(ops, x_norm_scratch, w, y, M, N, K, push_storage);
    }
}

double linear_auto(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int M, int N, int K) {
    std::vector<VkBatchDispatch> ops;
    std::vector<std::vector<uint8_t>> push_storage;
    append_linear_dispatch(ops, x, w, y, M, N, K, push_storage);
    return ctx.dispatch_batched(ops, true);
}

double gemm_auto(VkContext& ctx, const Tensor& a, const Tensor& b, Tensor& c, int M, int N, int K, bool use_coopmat_if_avail) {
    PushGemm pc{M, N, K};
    VkBatchDispatch d;
    if (use_coopmat_if_avail && ctx.info().coop_mat.supported && ctx.info().coop_mat.fp16_in_fp32_acc &&
        a.dtype == DType::F16 && b.dtype == DType::F16 &&
        (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0)) {
        d.shader_name = "gemm_coopmat.comp";
        d.gx = static_cast<uint32_t>(N / 16);
        d.gy = static_cast<uint32_t>(M / 16);
    } else {
        d.shader_name = "gemm_tiled.comp";
        d.gx = static_cast<uint32_t>((N + 15) / 16);
        d.gy = static_cast<uint32_t>((M + 15) / 16);
    }
    d.gz = 1;
    d.buffers = {a, b, c};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

double rmsnorm(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int rows, int cols, float eps) {
    PushRmsNorm pc{rows, cols, eps};
    VkBatchDispatch d;
    d.shader_name = "rmsnorm.comp";
    d.gx = static_cast<uint32_t>(rows);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {x, w, y};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

double fused_rmsnorm_gemv(VkContext& ctx, const Tensor& x, const Tensor& norm_w, const Tensor& w, Tensor& y, int M, int N, int K, float eps) {
    PushFusedNormGemv pc{M, N, K, eps};
    VkBatchDispatch d;
    d.shader_name = "fused_rmsnorm_gemv_vec4.comp";
    d.gx = static_cast<uint32_t>(N);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {x, norm_w, w, y};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

std::vector<VkBatchDispatch> build_dense_layer_dispatches(
    const Tensor& x_in,
    const Tensor& attn_norm_w, const Tensor& qkv_w, const Tensor& q_norm_w, const Tensor& k_norm_w,
    const Tensor& o_w, const Tensor& mlp_norm_w, const Tensor& gate_up_w, const Tensor& down_w,
    Tensor& qkv_buf, Tensor& q_buf, Tensor& k_buf, Tensor& v_buf,
    Tensor& k_cache, Tensor& v_cache, Tensor& attn_out, Tensor& o_buf,
    Tensor& gate_up_buf, Tensor& swiglu_buf, Tensor& down_buf, Tensor& x_out,
    int hidden, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim, int inter,
    int pos, int kv_len, float theta, float eps, bool qk_norm, bool neox,
    std::vector<std::vector<uint8_t>>& push_storage)
{
    std::vector<VkBatchDispatch> ops;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    (void)qkv_buf;

    // 1. Fused Pre-Norm + QKV GEMV writing directly into q_buf, k_buf, v_buf
    {
        struct PushFusedQkv { int q_dim, kv_dim, K; float eps; } pc{q_dim, kv_dim, hidden, eps};
        ops.push_back({"fused_rmsnorm_qkv_vec4.comp",
                       static_cast<uint32_t>(qkv_dim), 1, 1,
                       {x_in, attn_norm_w, qkv_w, q_buf, k_buf, v_buf},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 2. Fused Q/K Norm + RoPE + KV Cache Append
    {
        int flags = (qk_norm ? 1 : 0) | (neox ? 2 : 0);
        PushRopeKv pc{num_q_heads, num_kv_heads, head_dim, rot_dim, pos, theta, eps, flags};
        ops.push_back({"rope_kv_append.comp",
                       static_cast<uint32_t>(num_q_heads + num_kv_heads), 1, 1,
                       {q_buf, k_buf, v_buf, q_norm_w, k_norm_w, k_cache, v_cache},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 3. GQA Decode
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        PushGqaDecode pc{num_q_heads, num_kv_heads, head_dim, kv_len, scale};
        ops.push_back({"gqa_decode.comp",
                       static_cast<uint32_t>(num_q_heads), 1, 1,
                       {q_buf, k_cache, v_cache, attn_out},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 4. O Proj
    append_linear_dispatch(ops, attn_out, o_w, o_buf, 1, hidden, q_dim, push_storage);

    // 5. Residual 1: x_out = x_in + o_buf
    {
        PushResidual pc{hidden};
        ops.push_back({"add_residual.comp",
                       static_cast<uint32_t>((hidden + 255) / 256), 1, 1,
                       {x_in, o_buf, x_out},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 6. Fused MLP Pre-Norm + Gate/Up GEMV
    {
        PushFusedNormGemv pc{1, 2 * inter, hidden, eps};
        ops.push_back({"fused_rmsnorm_gemv_vec4.comp",
                       static_cast<uint32_t>(2 * inter), 1, 1,
                       {x_out, mlp_norm_w, gate_up_w, gate_up_buf},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 7. SwiGLU
    {
        PushSwiglu pc{1, inter};
        ops.push_back({"swiglu.comp",
                       static_cast<uint32_t>((inter + 255) / 256), 1, 1,
                       {gate_up_buf, swiglu_buf},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 8. Down Proj
    append_linear_dispatch(ops, swiglu_buf, down_w, down_buf, 1, hidden, inter, push_storage);

    // 9. Residual 2: x_out = x_out + down_buf
    {
        PushResidual pc{hidden};
        ops.push_back({"add_residual.comp",
                       static_cast<uint32_t>((hidden + 255) / 256), 1, 1,
                       {x_out, down_buf, x_out},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    return ops;
}

double gated_deltanet_step(
    VkContext& ctx,
    const Tensor& qkv_proj, const Tensor& z, const Tensor& a, const Tensor& b,
    const Tensor& conv_w, const Tensor& conv_bias, const Tensor& A_log, const Tensor& dt_bias,
    const Tensor& norm_w, Tensor& conv_state, Tensor& ssm_state, Tensor& out,
    int num_k_heads, int num_v_heads, int dk, int dv, float eps, float q_scale, bool has_conv_bias)
{
    PushGdnStep pc{num_k_heads, num_v_heads, dk, dv, eps, q_scale, has_conv_bias ? 1 : 0, 0};
    VkBatchDispatch d;
    d.shader_name = "gated_deltanet_step.comp";
    d.gx = static_cast<uint32_t>(num_k_heads);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {qkv_proj, z, a, b, conv_w, conv_bias, A_log, dt_bias, norm_w, conv_state, ssm_state, out};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

}  // namespace fusionrt::vk_ops
