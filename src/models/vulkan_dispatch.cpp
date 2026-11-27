#include "fusionrt/vulkan_dispatch.h"
#include "vulkan_session.h"
#include "vulkan_weights.h"

#include "fusionrt/vulkan_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

namespace fusionrt {
namespace {

template <typename Push>
void append(VulkanDispatch::Batch& batch, const char* shader, uint32_t groups,
            std::initializer_list<Tensor> buffers, const Push& push) {
    auto& storage = batch.push.emplace_back(sizeof(push));
    std::memcpy(storage.data(), &push, sizeof(push));
    batch.ops.push_back({shader, groups, 1, 1, buffers, storage.data(), sizeof(push)});
}

uint32_t element_groups(int elements) {
    return static_cast<uint32_t>((elements + 255) / 256);
}

void append_residual(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& update, const Tensor& output, int width) {
    append(batch, "add_residual.comp", element_groups(width), {input, update, output}, vk_ops::PushResidual{width});
}

}  // namespace

void VulkanDispatch::append_embedding(Batch& batch, const Tensor& embeddings, const Tensor& token,
                                      const Tensor& output, int hidden) {
    append(batch, "embed_lookup.comp", element_groups(hidden), {embeddings, token, output},
           vk_ops::PushEmbed{1, hidden, static_cast<int>(embeddings.dtype)});
}

void VulkanDispatch::append_dense(Batch& batch, const VulkanDenseWeights& layer, const VulkanDenseKv& kv,
                                  DenseScratch& scratch, const Tensor& input, const Tensor& output, int hidden, int pos) {
    const auto& cfg = layer.cfg;
    const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads, hd = cfg.head_dim;
    const int q_dim = nq * hd, kv_dim = nkv * hd;
    const int rot_dim = std::max(2, static_cast<int>(hd * cfg.partial_rotary_factor) & ~1);

    if (layer.qkv_proj_fused.vk_buffer && layer.qkv_proj_fused.dtype == DType::F32 && (hidden % 4) == 0) {
        struct PushFusedQkv { int q_dim, kv_dim, K; float eps; };
        append(batch, "fused_rmsnorm_qkv_vec4.comp", static_cast<uint32_t>(q_dim + 2 * kv_dim),
               {input, layer.attn_norm_w, layer.qkv_proj_fused, scratch.query, scratch.key, scratch.value},
               PushFusedQkv{q_dim, kv_dim, hidden, cfg.rms_norm_eps});
    } else {
        append(batch, "rmsnorm.comp", 1, {input, layer.attn_norm_w, scratch.normalized},
               vk_ops::PushRmsNorm{1, hidden, cfg.rms_norm_eps});
        vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.q_proj,
                                       cfg.gated_q_attn ? scratch.query_full : scratch.query,
                                       1, cfg.gated_q_attn ? 2 * q_dim : q_dim, hidden, batch.push);
        vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.k_proj, scratch.key, 1, kv_dim, hidden, batch.push);
        vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.v_proj, scratch.value, 1, kv_dim, hidden, batch.push);
    }

    if (cfg.gated_q_attn) {
        if (layer.q_bias.vk_buffer) append_residual(batch, scratch.query_full, layer.q_bias, scratch.query_full, 2 * q_dim);
        const int push[] = {q_dim, hd};
        append(batch, "split_query_gate.comp", element_groups(q_dim), {scratch.query_full, scratch.query, scratch.query_gate}, push);
    } else if (layer.q_bias.vk_buffer) {
        append_residual(batch, scratch.query, layer.q_bias, scratch.query, q_dim);
    }
    if (layer.k_bias.vk_buffer) append_residual(batch, scratch.key, layer.k_bias, scratch.key, kv_dim);
    if (layer.v_bias.vk_buffer) append_residual(batch, scratch.value, layer.v_bias, scratch.value, kv_dim);

    const int flags = (cfg.qk_norm ? 1 : 0) | (cfg.rope_neox ? 2 : 0);
    append(batch, "rope_kv_append.comp", static_cast<uint32_t>(nq + nkv),
           {scratch.query, scratch.key, scratch.value, layer.q_norm_w, layer.k_norm_w, kv.key, kv.value},
           vk_ops::PushRopeKv{nq, nkv, hd, rot_dim, pos, cfg.rope_theta, cfg.rms_norm_eps, flags});
    append(batch, "gqa_decode.comp", static_cast<uint32_t>(nq), {scratch.query, kv.key, kv.value, scratch.attention},
           vk_ops::PushGqaDecode{nq, nkv, hd, pos + 1, 1.0f / std::sqrt(static_cast<float>(hd))});
    if (cfg.gated_q_attn) {
        const int push[] = {q_dim, 0};
        append(batch, "sigmoid_gate.comp", element_groups(q_dim), {scratch.attention, scratch.query_gate}, push);
    }
    vk_ops::append_linear_dispatch(batch.ops, scratch.attention, layer.o_proj, scratch.projected, 1, hidden, q_dim, batch.push);
    append_residual(batch, input, scratch.projected, output, hidden);
}

void VulkanDispatch::append_gdn(Batch& batch, const VulkanGdnWeights& layer, const VulkanGdnState& state,
                                GdnScratch& scratch, const Tensor& input, const Tensor& output, int hidden) {
    const auto& cfg = layer.cfg;
    const int nk = cfg.num_k_heads, nv = cfg.num_v_heads, dk = cfg.head_k_dim, dv = cfg.head_v_dim;
    const int v_dim = nv * dv, conv_dim = 2 * nk * dk + v_dim;

    append(batch, "rmsnorm.comp", 1, {input, layer.attn_norm_w, scratch.normalized},
           vk_ops::PushRmsNorm{1, hidden, cfg.rms_norm_eps});
    vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.qkv_proj, scratch.qkv, 1, conv_dim, hidden, batch.push);
    vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.z_proj, scratch.z, 1, v_dim, hidden, batch.push);
    vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.a_proj, scratch.a, 1, nv, hidden, batch.push);
    vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.b_proj, scratch.b, 1, nv, hidden, batch.push);
    append(batch, "gated_deltanet_step.comp", static_cast<uint32_t>(nk),
           {scratch.qkv, scratch.z, scratch.a, scratch.b, layer.conv_w, layer.conv_bias, layer.A_log,
            layer.dt_bias, layer.norm_w, state.conv, state.ssm, scratch.recurrent},
           vk_ops::PushGdnStep{nk, nv, dk, dv, cfg.rms_norm_eps, 1.0f / std::sqrt(static_cast<float>(dk)), 1,
                              cfg.tiled_value_heads ? 1 : 0});
    vk_ops::append_linear_dispatch(batch.ops, scratch.recurrent, layer.o_proj, scratch.projected, 1, hidden, v_dim, batch.push);
    append_residual(batch, input, scratch.projected, output, hidden);
}

void VulkanDispatch::append_ffn(Batch& batch, const Tensor& norm, const Tensor& gate_up, const Tensor& down,
                                Tensor& normalized, FfnScratch& scratch, const Tensor& activation, int hidden, int inter, float eps) {
    vk_ops::append_rmsnorm_linear_dispatch(batch.ops, activation, norm, gate_up, normalized, scratch.gate_up,
                                          1, 2 * inter, hidden, eps, batch.push);
    append(batch, "swiglu.comp", element_groups(inter), {scratch.gate_up, scratch.activated}, vk_ops::PushSwiglu{1, inter});
    vk_ops::append_linear_dispatch(batch.ops, scratch.activated, down, scratch.down, 1, hidden, inter, batch.push);
    append_residual(batch, activation, scratch.down, activation, hidden);
}

void VulkanDispatch::append_moe(Batch& batch, const VkSparseMoELayer& layer, MoeScratch& scratch,
                                const Tensor& input, const Tensor& output, int hidden) {
    const auto& cfg = layer.cfg;
    const int experts = cfg.num_experts, top_k = cfg.top_k, inter = cfg.intermediate_size;
    append(batch, "rmsnorm.comp", 1, {input, layer.norm_w, scratch.normalized},
           vk_ops::PushRmsNorm{1, hidden, cfg.rms_norm_eps});
    vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.router_w, scratch.router_logits,
                                   1, experts, hidden, batch.push);
    append(batch, "moe_topk.comp", 1, {scratch.router_logits, scratch.indices, scratch.weights},
           vk_ops::PushMoeTopk{experts, top_k, cfg.norm_topk_prob ? 1 : 0, cfg.expert_weights_scale});
    append(batch, "moe_swiglu_combine.comp", static_cast<uint32_t>(top_k * inter),
           {scratch.normalized, scratch.indices, layer.gpu_gate_up_w, scratch.weights, scratch.intermediate},
           vk_ops::PushMoeSwiglu{hidden, inter, top_k, 0});
    append(batch, "moe_swiglu_combine.comp", static_cast<uint32_t>(hidden),
           {scratch.intermediate, scratch.indices, layer.gpu_down_w, scratch.weights, scratch.output},
           vk_ops::PushMoeSwiglu{hidden, inter, top_k, 1});
    if (cfg.has_shared_expert) {
        const int shared = cfg.shared_intermediate_size;
        vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.shared_gate_up_w, scratch.shared_gate_up,
                                       1, 2 * shared, hidden, batch.push);
        append(batch, "swiglu.comp", element_groups(shared), {scratch.shared_gate_up, scratch.shared_activated},
               vk_ops::PushSwiglu{1, shared});
        vk_ops::append_linear_dispatch(batch.ops, scratch.shared_activated, layer.shared_down_w, scratch.shared_down,
                                       1, hidden, shared, batch.push);
        if (layer.shared_gate_w.vk_buffer) {
            vk_ops::append_linear_dispatch(batch.ops, scratch.normalized, layer.shared_gate_w, scratch.shared_gate,
                                           1, 1, hidden, batch.push);
            const int push[] = {hidden, 1};
            append(batch, "sigmoid_gate.comp", element_groups(hidden), {scratch.shared_down, scratch.shared_gate}, push);
        }
        append_residual(batch, scratch.output, scratch.shared_down, scratch.output, hidden);
    }
    append_residual(batch, input, scratch.output, output, hidden);
}

void VulkanDispatch::append_head(Batch& batch, const Tensor& norm, const Tensor& head, Tensor& normalized,
                                 Tensor& logits, const Tensor& input, int hidden, int vocab, float eps) {
    vk_ops::append_rmsnorm_linear_dispatch(batch.ops, input, norm, head, normalized, logits, 1, vocab, hidden, eps, batch.push);
}

}  // namespace fusionrt
