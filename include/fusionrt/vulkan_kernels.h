#pragma once

#include "fusionrt/vulkan_backend.h"
#include <vector>

namespace fusionrt::vk_ops {

struct PushRmsNorm { int rows, cols; float eps; };
struct PushGemm { int M, N, K; };
struct PushFusedNormGemv { int M, N, K; float eps; };
struct PushRopeKv { int num_q_heads, num_kv_heads, head_dim, rot_dim, pos; float theta, eps; int flags; };
struct PushGqaDecode { int num_q_heads, num_kv_heads, head_dim, kv_len; float scale; };
struct PushSwiglu { int rows, inter_dim; };
struct PushGdnStep { int num_k_heads, num_v_heads, dk, dv; float eps, q_scale; int has_conv_bias, tiled_value_heads; };
struct PushResidual { int n; };
struct PushEmbed { int M, hidden, dtype; };
struct PushMoeTopk { int num_experts, top_k, norm_topk; float scale; };
struct PushMoeSwiglu { int hidden, inter, top_k, mode; };

double gemv_vec4(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int M, int N, int K);
double linear_auto(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int M, int N, int K);
double gemm_auto(VkContext& ctx, const Tensor& a, const Tensor& b, Tensor& c, int M, int N, int K, bool use_coopmat_if_avail = true);
double rmsnorm(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int rows, int cols, float eps);
double fused_rmsnorm_gemv(VkContext& ctx, const Tensor& x, const Tensor& norm_w, const Tensor& w, Tensor& y, int M, int N, int K, float eps);

// Helper to append linear dispatch (F32 vec4, Q8_0, Q4_0) into a command buffer batch
void append_linear_dispatch(std::vector<VkBatchDispatch>& ops,
                            const Tensor& x, const Tensor& w, Tensor& y,
                            int M, int N, int K,
                            std::vector<std::vector<uint8_t>>& push_storage);

// Helper to append Pre-Norm + Linear (uses fused vec4 when F32 and K%4==0, else rmsnorm + linear)
void append_rmsnorm_linear_dispatch(std::vector<VkBatchDispatch>& ops,
                                    const Tensor& x, const Tensor& norm_w, const Tensor& w,
                                    Tensor& x_norm_scratch, Tensor& y,
                                    int M, int N, int K, float eps,
                                    std::vector<std::vector<uint8_t>>& push_storage);

std::vector<VkBatchDispatch> build_dense_layer_dispatches(
    const Tensor& x_in,
    const Tensor& attn_norm_w, const Tensor& qkv_w, const Tensor& q_norm_w, const Tensor& k_norm_w,
    const Tensor& o_w, const Tensor& mlp_norm_w, const Tensor& gate_up_w, const Tensor& down_w,
    Tensor& qkv_buf, Tensor& q_buf, Tensor& k_buf, Tensor& v_buf,
    Tensor& k_cache, Tensor& v_cache, Tensor& attn_out, Tensor& o_buf,
    Tensor& gate_up_buf, Tensor& swiglu_buf, Tensor& down_buf, Tensor& x_out,
    int hidden, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim, int inter,
    int pos, int kv_len, float theta, float eps, bool qk_norm, bool neox,
    std::vector<std::vector<uint8_t>>& push_storage);

double gated_deltanet_step(
    VkContext& ctx,
    const Tensor& qkv_proj, const Tensor& z, const Tensor& a, const Tensor& b,
    const Tensor& conv_w, const Tensor& conv_bias, const Tensor& A_log, const Tensor& dt_bias,
    const Tensor& norm_w, Tensor& conv_state, Tensor& ssm_state, Tensor& out,
    int num_k_heads, int num_v_heads, int dk, int dv, float eps, float q_scale, bool has_conv_bias);

}  // namespace fusionrt::vk_ops
