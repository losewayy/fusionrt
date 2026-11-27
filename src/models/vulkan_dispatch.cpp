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
            std::initializer_list<Tensor> buffers, const Push& push, uint32_t rows = 1) {
    auto& storage = batch.push.emplace_back(sizeof(push));
    std::memcpy(storage.data(), &push, sizeof(push));
    batch.ops.push_back({shader, groups, rows, 1, buffers, storage.data(), sizeof(push)});
}

uint32_t element_groups(int elements) {
    return static_cast<uint32_t>((elements + 255) / 256);
}

void append_residual(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& update, const Tensor& output, int width) {
    append(batch, "add_residual.comp", element_groups(width), {input, update, output}, vk_ops::PushResidual{width});
}

void append_linear(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& weights,
                   Tensor& output, int rows, int width, int extent) {
    vk_ops::append_linear_dispatch(batch.ops, input, weights, output, rows, width, extent, batch.push, &batch.ctx);
}

void append_pack_rows(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& packed,
                      int rows, int width, int packed_width, int packed_offset) {
    std::vector<VkBufferCopy> copies;
    copies.reserve(rows);
    const auto bytes = static_cast<VkDeviceSize>(width) * sizeof(float);
    for (int row = 0; row < rows; ++row)
        copies.push_back({row * bytes, (static_cast<VkDeviceSize>(row) * packed_width + packed_offset) * sizeof(float), bytes});
    VulkanDispatch::append_copy(batch, input, packed, std::move(copies));
}

void append_bias(VulkanDispatch::Batch& batch, const Tensor& bias, Tensor& projection, const Tensor& scratch,
                 int rows, int width) {
    if (!bias.vk_buffer) return;
    if (rows == 1) {
        append_residual(batch, projection, bias, projection, width);
        return;
    }

    const auto bytes = static_cast<VkDeviceSize>(width) * sizeof(float);
    std::vector<VkBufferCopy> copies;
    copies.reserve(rows);
    for (int row = 0; row < rows; ++row) copies.push_back({0, row * bytes, bytes});
    VulkanDispatch::append_copy(batch, bias, scratch, std::move(copies));
    append_residual(batch, projection, scratch, projection, rows * width);
}

}  // namespace

void VulkanDispatch::append_copy(Batch& batch, const Tensor& source, const Tensor& target,
                                 std::vector<VkBufferCopy> regions) {
    batch.ops.push_back({"copy", 0, 0, 0, {source, target}, nullptr, 0, std::move(regions)});
}

void VulkanDispatch::append_embedding(Batch& batch, const Tensor& embeddings, const Tensor& token,
                                      const Tensor& output, int hidden, int rows) {
    append(batch, "embed_lookup.comp", element_groups(rows * hidden), {embeddings, token, output},
           vk_ops::PushEmbed{rows, hidden, static_cast<int>(embeddings.dtype)});
}

void VulkanDispatch::append_dense(Batch& batch, const VulkanDenseWeights& layer, const VulkanDenseKv& kv,
                                  DenseScratch& scratch, const Tensor& input, const Tensor& output,
                                  int hidden, int pos, int rows) {
    const auto& cfg = layer.cfg;
    const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads, hd = cfg.head_dim;
    const int q_dim = nq * hd, kv_dim = nkv * hd, qkv_dim = q_dim + 2 * kv_dim;
    const int rot_dim = std::clamp(static_cast<int>(hd * cfg.partial_rotary_factor) & ~1, 0, hd & ~1);
    const bool separate_qkv = !layer.qkv_proj_fused.vk_buffer;

    if (!separate_qkv) {
        vk_ops::append_rmsnorm_linear_dispatch(batch.ops, input, layer.attn_norm_w, layer.qkv_proj_fused,
            scratch.normalized, scratch.qkv, rows, qkv_dim, hidden, cfg.rms_norm_eps, batch.push, &batch.ctx);
    } else {
        append(batch, "rmsnorm.comp", rows, {input, layer.attn_norm_w, scratch.normalized},
               vk_ops::PushRmsNorm{rows, hidden, cfg.rms_norm_eps});
        const int q_width = cfg.gated_q_attn ? 2 * q_dim : q_dim;
        auto& q = cfg.gated_q_attn ? scratch.query_full : scratch.query;
        append_linear(batch, scratch.normalized, layer.q_proj, q, rows, q_width, hidden);
        append_linear(batch, scratch.normalized, layer.k_proj, scratch.key, rows, kv_dim, hidden);
        append_linear(batch, scratch.normalized, layer.v_proj, scratch.value, rows, kv_dim, hidden);
        append_bias(batch, layer.q_bias, q, scratch.bias, rows, q_width);
        append_bias(batch, layer.k_bias, scratch.key, scratch.bias, rows, kv_dim);
        append_bias(batch, layer.v_bias, scratch.value, scratch.bias, rows, kv_dim);
        if (cfg.gated_q_attn) {
            const int push[] = {rows * q_dim, hd};
            append(batch, "split_query_gate.comp", element_groups(rows * q_dim),
                   {q, scratch.query, scratch.query_gate}, push);
        }
        append_pack_rows(batch, scratch.query, scratch.qkv, rows, q_dim, qkv_dim, 0);
        append_pack_rows(batch, scratch.key, scratch.qkv, rows, kv_dim, qkv_dim, q_dim);
        append_pack_rows(batch, scratch.value, scratch.qkv, rows, kv_dim, qkv_dim, q_dim + kv_dim);
    }

    const int flags = (cfg.qk_norm ? 1 : 0) | (cfg.rope_neox ? 2 : 0);
    const int kv_type = static_cast<int>(kv.key.dtype);
    append(batch, "sequence/rope_kv_append_batch.spv", nq + nkv,
           {scratch.qkv, layer.q_norm_w, layer.k_norm_w, scratch.query, kv.key, kv.value, layer.q_norm_w},
           vk_ops::PushRopeKvBatch{rows, nq, nkv, hd, rot_dim, pos, kv_type, flags, cfg.rope_theta, cfg.rms_norm_eps}, rows);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const int window = layer.sliding_window;
    if (rows == 1) {
        append(batch, "sequence/flash_attention_decode.spv", nq, {scratch.query, kv.key, kv.value, scratch.attention},
               vk_ops::PushFaDecode{nq, nkv, hd, pos + 1, kv_type, window, scale});
    } else {
        append(batch, "sequence/flash_attention_prefill.spv", nq, {scratch.query, kv.key, kv.value, scratch.attention},
               vk_ops::PushFaPrefill{rows, nq, nkv, hd, pos, pos + rows, kv_type, window, 1, scale}, rows);
    }
    if (cfg.gated_q_attn) {
        const int push[] = {rows * q_dim, 0};
        append(batch, "sigmoid_gate.comp", element_groups(rows * q_dim), {scratch.attention, scratch.query_gate}, push);
    }
    append_linear(batch, scratch.attention, layer.o_proj, scratch.projected, rows, hidden, q_dim);
    append_residual(batch, input, scratch.projected, output, rows * hidden);
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

void VulkanDispatch::append_ffn(Batch& batch, const Tensor& norm, const Tensor& gate_up, const Tensor& gate,
                                const Tensor& up, const Tensor& down, Tensor& normalized, FfnScratch& scratch,
                                const Tensor& activation, int hidden, int inter, int rows, float eps) {
    if (gate_up.vk_buffer) {
        vk_ops::append_rmsnorm_linear_dispatch(batch.ops, activation, norm, gate_up, normalized, scratch.gate_up,
                                               rows, 2 * inter, hidden, eps, batch.push, &batch.ctx);
    } else {
        append(batch, "rmsnorm.comp", rows, {activation, norm, normalized}, vk_ops::PushRmsNorm{rows, hidden, eps});
        append_linear(batch, normalized, gate, scratch.gate, rows, inter, hidden);
        append_linear(batch, normalized, up, scratch.up, rows, inter, hidden);
        append_pack_rows(batch, scratch.gate, scratch.gate_up, rows, inter, 2 * inter, 0);
        append_pack_rows(batch, scratch.up, scratch.gate_up, rows, inter, 2 * inter, inter);
    }
    append(batch, "swiglu.comp", element_groups(rows * inter), {scratch.gate_up, scratch.activated},
           vk_ops::PushSwiglu{rows, inter});
    append_linear(batch, scratch.activated, down, scratch.down, rows, hidden, inter);
    append_residual(batch, activation, scratch.down, activation, rows * hidden);
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
