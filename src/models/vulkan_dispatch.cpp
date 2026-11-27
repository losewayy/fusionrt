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
            std::initializer_list<Tensor> buffers, const Push& push, uint32_t rows = 1, uint32_t depth = 1) {
    auto& storage = batch.push.emplace_back(sizeof(push));
    std::memcpy(storage.data(), &push, sizeof(push));
    batch.ops.push_back({shader, groups, rows, depth, buffers, storage.data(), sizeof(push)});
    for (auto& tensor : batch.ops.back().buffers)
        if (!tensor.vk_buffer) tensor = batch.dummy;
}

uint32_t element_groups(int elements) {
    return static_cast<uint32_t>((elements + 255) / 256);
}

void append_residual(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& update, const Tensor& output, int width) {
    append(batch, "add_residual.comp", element_groups(width), {input, update, output}, vk_ops::PushResidual{width});
}

uint32_t vector_groups(int rows, int width) {
    return static_cast<uint32_t>((static_cast<int64_t>(rows) * width + 63) / 64);
}

void append_gemma(VulkanDispatch::Batch& batch, const Tensor& a, const Tensor& b, const Tensor& output,
                  const vk_ops::PushGemma& push, uint32_t groups) {
    append(batch, "epilogue/gemma4_ops.spv", groups, {a, b, {}, output, {}, {}, {}}, push);
}

Tensor hadamard_sign_buffer(const VulkanDispatch::Batch& batch, const Tensor& weight) {
    if (!batch.hadamard_signs) return {};
    const auto it = batch.hadamard_signs->find(weight.vk_offset);
    return it == batch.hadamard_signs->end() ? Tensor{} : it->second;
}

void append_norm_linear(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& norm,
                        const Tensor& weight, Tensor& normalized, Tensor& output,
                        int rows, int width, int extent, float eps) {
    if (weight.hadamard_block) {
        if (weight.hadamard_key_heads) {
            VulkanDispatch::append_norm(batch, input, norm, normalized, rows, extent, eps);
            VulkanDispatch::append_linear(batch, normalized, weight, output, rows, width, extent);
            return;
        }
        const auto signs = hadamard_sign_buffer(batch, weight);
        const vk_ops::PushHadamard push{1, rows, extent, weight.hadamard_block, signs.vk_buffer ? 1 : 0, 0, 0, eps};
        append(batch, "epilogue/hadamard_fwht.spv", 1, {input, batch.rotated, norm, signs}, push, rows);
        vk_ops::append_linear_dispatch(batch.ops, batch.rotated, weight, output, rows, width, extent, batch.push, &batch.ctx);
    } else {
        vk_ops::append_rmsnorm_linear_dispatch(batch.ops, input, norm, weight, normalized, output,
                                               rows, width, extent, eps, batch.push, &batch.ctx);
    }
}

// Copy a column range between dense row-major matrices without descriptor sub-offsets.
void append_columns(VulkanDispatch::Batch& batch, const Tensor& source, const Tensor& target,
                    int rows, int width, int src_stride, int dst_stride, int src_col, int dst_col) {
    std::vector<VkBufferCopy> copies;
    copies.reserve(rows);
    for (int row = 0; row < rows; ++row) {
        const VkDeviceSize src = static_cast<VkDeviceSize>(row) * src_stride + src_col;
        const VkDeviceSize dst = static_cast<VkDeviceSize>(row) * dst_stride + dst_col;
        copies.push_back({src * sizeof(float), dst * sizeof(float), static_cast<VkDeviceSize>(width) * sizeof(float)});
    }
    VulkanDispatch::append_copy(batch, source, target, std::move(copies));
}

void append_pack_rows(VulkanDispatch::Batch& batch, const Tensor& input, const Tensor& packed,
                      int rows, int width, int packed_width, int packed_offset) {
    append_columns(batch, input, packed, rows, width, width, packed_width, 0, packed_offset);
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

void VulkanDispatch::append_norm(Batch& batch, const Tensor& input, const Tensor& weight,
                                 const Tensor& output, int rows, int width, float eps) {
    append(batch, "rmsnorm.comp", rows, {input, weight, output}, vk_ops::PushRmsNorm{rows, width, eps});
}

void VulkanDispatch::append_gemma_norm(Batch& batch, const Tensor& input, const Tensor& weight,
                                       const Tensor& output, int rows, int width, float eps, bool add_one) {
    vk_ops::PushGemma push{1, rows, width};
    push.flags = weight.vk_buffer ? (add_one ? 1 : 0) : 2;
    push.eps = eps;
    append_gemma(batch, input, weight, output, push, rows);
}

void VulkanDispatch::append_scale_add(Batch& batch, const Tensor& a, const Tensor& b, const Tensor& output,
                                      int rows, int width, float scale_a, float scale_b) {
    vk_ops::PushGemma push{3, rows, width};
    push.scale_a = scale_a;
    push.scale_b = scale_b;
    append_gemma(batch, a, b, output, push, vector_groups(rows, width));
}

void VulkanDispatch::append_geglu(Batch& batch, const Tensor& gate, const Tensor& up,
                                  const Tensor& output, int rows, int width) {
    append_gemma(batch, gate, up, output, {0, rows, width}, vector_groups(rows, width));
}

void VulkanDispatch::append_hadamard(Batch& batch, const Tensor& weight, const Tensor& input,
                                     const Tensor& output, int rows, int width, bool inverse) {
    if (!weight.hadamard_block) return;
    const auto signs = hadamard_sign_buffer(batch, weight);
    const vk_ops::PushHadamard push{0, rows, width, weight.hadamard_block,
                                    (signs.vk_buffer ? 1 : 0) | (inverse ? 2 : 0),
                                    weight.hadamard_head_dim, weight.hadamard_key_heads, 0.0f};
    append(batch, "epilogue/hadamard_fwht.spv", width / push.block_size, {input, output, {}, signs}, push, rows);
}

void VulkanDispatch::append_linear(Batch& batch, const Tensor& input, const Tensor& weights,
                                   Tensor& output, int rows, int width, int extent) {
    const Tensor& x = weights.hadamard_block ? batch.rotated : input;
    append_hadamard(batch, weights, input, x, rows, extent, false);
    vk_ops::append_linear_dispatch(batch.ops, x, weights, output, rows, width, extent, batch.push, &batch.ctx);
}

void VulkanDispatch::append_dense(Batch& batch, const VulkanDenseWeights& layer, const VulkanDenseKv& kv,
                                  DenseScratch& scratch, const Tensor& input, const Tensor& output,
                                  int hidden, int pos, int rows, std::span<const AttentionSpan> image_spans) {
    const auto& cfg = layer.cfg;
    const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads, hd = cfg.head_dim;
    const int q_dim = nq * hd, kv_dim = nkv * hd, qkv_dim = q_dim + 2 * kv_dim;
    const int rot_dim = std::clamp(static_cast<int>(hd * cfg.partial_rotary_factor) & ~1, 0, hd & ~1);
    const bool separate_qkv = !layer.qkv_proj_fused.vk_buffer;
    const int append_heads = layer.kv_source >= 0 ? 0 : nkv;

    if (!separate_qkv) {
        append_norm_linear(batch, input, layer.attn_norm_w, layer.qkv_proj_fused,
                           scratch.normalized, scratch.qkv, rows, qkv_dim, hidden, cfg.rms_norm_eps);
    } else {
        append(batch, "rmsnorm.comp", rows, {input, layer.attn_norm_w, scratch.normalized},
               vk_ops::PushRmsNorm{rows, hidden, cfg.rms_norm_eps});
        const int q_width = cfg.gated_q_attn ? 2 * q_dim : q_dim;
        auto& q = cfg.gated_q_attn ? scratch.query_full : scratch.query;
        append_linear(batch, scratch.normalized, layer.q_proj, q, rows, q_width, hidden);
        if (append_heads) {
            append_linear(batch, scratch.normalized, layer.k_proj, scratch.key, rows, kv_dim, hidden);
            if (layer.v_proj.vk_buffer)
                append_linear(batch, scratch.normalized, layer.v_proj, scratch.value, rows, kv_dim, hidden);
        }
        append_bias(batch, layer.q_bias, q, scratch.bias, rows, q_width);
        append_bias(batch, layer.k_bias, scratch.key, scratch.bias, rows, kv_dim);
        append_bias(batch, layer.v_bias, scratch.value, scratch.bias, rows, kv_dim);
        if (cfg.gated_q_attn) {
            const int push[] = {rows * q_dim, hd};
            append(batch, "split_query_gate.comp", element_groups(rows * q_dim),
                   {q, scratch.query, scratch.query_gate}, push);
        }
        if (append_heads) {
            append_pack_rows(batch, scratch.query, scratch.qkv, rows, q_dim, qkv_dim, 0);
            append_pack_rows(batch, scratch.key, scratch.qkv, rows, kv_dim, qkv_dim, q_dim);
            append_pack_rows(batch, layer.v_proj.vk_buffer ? scratch.value : scratch.key,
                             scratch.qkv, rows, kv_dim, qkv_dim, q_dim + kv_dim);
        }
    }

    const int flags = (cfg.qk_norm ? 1 : 0) | (cfg.rope_neox ? 2 : 0);
    const int kv_type = static_cast<int>(kv.key.dtype);
    if (layer.gemma4 || layer.rope_factors.vk_buffer) {
        // Shared-KV layers rotate Q only. The shader takes separate input and output buffers.
        if (!append_heads)
            append_copy(batch, scratch.query, scratch.qkv, {{0, 0, scratch.query.nbytes()}});
        const vk_ops::PushGemma push{4, rows, 0, nq, append_heads, hd, rot_dim, pos, kv_type,
            flags | (layer.rope_factors.vk_buffer ? 4 : 0) | (layer.gemma4 ? 16 : 0),
            cfg.rope_theta, cfg.rms_norm_eps};
        append(batch, "epilogue/gemma4_ops.spv", nq + append_heads,
               {scratch.qkv, layer.q_norm_w, layer.k_norm_w, scratch.query, kv.key, kv.value, layer.rope_factors},
               push, rows);
    } else {
        append(batch, "sequence/rope_kv_append_batch.spv", nq + nkv,
               {scratch.qkv, layer.q_norm_w, layer.k_norm_w, scratch.query, kv.key, kv.value, {}},
               vk_ops::PushRopeKvBatch{rows, nq, nkv, hd, rot_dim, pos, kv_type, flags, cfg.rope_theta, cfg.rms_norm_eps}, rows);
    }
    const float scale = layer.gemma4 ? 1.0f : 1.0f / std::sqrt(static_cast<float>(hd));
    const int window = layer.sliding_window;
    if (rows == 1) {
        append(batch, "sequence/flash_attention_decode.spv", nq, {scratch.query, kv.key, kv.value, scratch.attention},
               vk_ops::PushFaDecode{nq, nkv, hd, pos + 1, kv_type, window, scale});
    } else {
        const auto attention = [&](int begin, int end, int kv_len, int swin, bool causal) {
            if (begin == end) return;
            auto query = scratch.query.slice_dim0(begin, end - begin);
            auto result = scratch.attention.slice_dim0(begin, end - begin);
            const auto offset = query.vk_offset - scratch.query.vk_offset;
            const auto bytes = query.nbytes();
            const bool aligned = query.vk_offset % batch.ctx.info().min_storage_offset_align == 0;
            if (!aligned) {
                query = scratch.image_query.slice_dim0(0, end - begin);
                result = scratch.image_attention.slice_dim0(0, end - begin);
                append_copy(batch, scratch.query, query, {{offset, 0, bytes}});
            }
            append(batch, "sequence/flash_attention_prefill.spv", nq, {query, kv.key, kv.value, result},
                   vk_ops::PushFaPrefill{end - begin, nq, nkv, hd, pos + begin, kv_len, kv_type, swin, causal, scale},
                   end - begin);
            if (!aligned) append_copy(batch, result, scratch.attention, {{0, offset, bytes}});
        };
        int begin = 0;
        for (const auto& span : image_spans) {
            attention(begin, span.begin, pos + span.begin, window, true);
            const int span_len = span.end - span.begin;
            const int first_end = window > 0 ? span.begin + std::min(span_len, window) : span.end;
            attention(span.begin, first_end, pos + span.end, window, false);
            // Clamp the lower SWA bound to the image's start, just as CUDA does.
            for (int row = first_end; row < span.end; ++row)
                attention(row, row + 1, pos + span.end, row - span.begin + 1, false);
            begin = span.end;
        }
        attention(begin, rows, pos + rows, window, true);
    }
    if (cfg.gated_q_attn) {
        const int push[] = {rows * q_dim, 0};
        append(batch, "sigmoid_gate.comp", element_groups(rows * q_dim), {scratch.attention, scratch.query_gate}, push);
    }
    append_linear(batch, scratch.attention, layer.o_proj, scratch.projected, rows, hidden, q_dim);
    if (layer.attn_post_norm.vk_buffer)
        append_gemma_norm(batch, scratch.projected, layer.attn_post_norm, scratch.projected, rows, hidden, cfg.rms_norm_eps);
    append_residual(batch, input, scratch.projected, output, rows * hidden);
}

void VulkanDispatch::append_mla(Batch& batch, const VulkanDenseWeights& layer, const VulkanDenseKv& kv,
                                MlaScratch& s, const Tensor& input, const Tensor& output,
                                int hidden, int pos, int rows) {
    const auto& d = layer.cfg;
    const auto& m = d.mla;
    const int heads = d.num_attention_heads, rank = m.latent_rank, rope = m.rope_dims;
    const int nope = d.head_dim - rope, dv = m.value_dim;
    append_norm(batch, input, layer.attn_norm_w, s.normalized, rows, hidden, d.rms_norm_eps);
    if (m.query_rank) {
        append_linear(batch, s.normalized, layer.mla_q_a, s.q_low, rows, m.query_rank, hidden);
        append_norm(batch, s.q_low, layer.mla_q_a_norm, s.q_low, rows, m.query_rank, d.rms_norm_eps);
    }
    append_linear(batch, m.query_rank ? s.q_low : s.normalized, layer.q_proj, s.q,
                  rows, heads * d.head_dim, m.query_rank ? m.query_rank : hidden);
    append_linear(batch, s.normalized, layer.mla_kv_a, s.kv, rows, rank + rope, hidden);
    append_columns(batch, s.q, s.q_pe, rows * heads, rope, d.head_dim, rope, nope, 0);
    // Pack each head temporarily for Q_nope * W_UK and restore token-major order.
    for (int head = 0; head < heads; ++head) {
        append_columns(batch, s.q, s.head_input, rows, nope, heads * d.head_dim, nope, head * d.head_dim, 0);
        append_linear(batch, s.head_input, layer.mla_k_b[head], s.head_output, rows, rank, nope);
        append_pack_rows(batch, s.head_output, s.q_absorbed, rows, rank, heads * rank, head * rank);
    }
    // The KV-A projection is packed. Extract its positional part for split RoPE.
    append_columns(batch, s.kv, s.head_input, rows, rope, rank + rope, rope, rank, 0);
    const auto scale = m.attention_scale > 0.0f ? m.attention_scale : 1.f / std::sqrt(static_cast<float>(d.head_dim));
    vk_ops::PushMla push{2, rows, heads, rank, rope, pos, layer.sliding_window, static_cast<int>(kv.latent.dtype),
                         layer.rope_factors.vk_buffer ? 4 : 0, scale, d.rope_theta, m.rope_amplitude, d.rms_norm_eps};
    append(batch, "sequence/mla_absorb_and_cache.spv", heads + 1,
           {s.q_pe, s.head_input, {}, {}, layer.rope_factors, {}}, push, rows);
    append_pack_rows(batch, s.head_input, s.kv, rows, rope, rank + rope, rank);
    push.mode = 0;
    push.flags = 1 | 8;
    append(batch, "sequence/mla_absorb_and_cache.spv", rows,
           {s.kv, {}, kv.latent, {}, {}, layer.mla_kv_a_norm}, push);
    push.mode = 1;
    push.flags = 0;
    append(batch, "sequence/mla_absorb_and_cache.spv", heads,
           {s.q_absorbed, s.q_pe, kv.latent, s.latent, {}, {}}, push, rows);
    for (int head = 0; head < heads; ++head) {
        append_columns(batch, s.latent, s.head_input, rows, rank, heads * rank, rank, head * rank, 0);
        append_linear(batch, s.head_input, layer.mla_v_b[head], s.head_output, rows, dv, rank);
        append_pack_rows(batch, s.head_output, s.value, rows, dv, heads * dv, head * dv);
    }
    append_linear(batch, s.value, layer.o_proj, s.projected, rows, hidden, heads * dv);
    append_residual(batch, input, s.projected, output, rows * hidden);
}

void VulkanDispatch::append_gdn(Batch& batch, const VulkanGdnWeights& layer, const VulkanGdnState& state,
                                GdnScratch& scratch, const Tensor& input, const Tensor& output, int hidden, int rows) {
    const auto& cfg = layer.cfg;
    const int nk = cfg.num_k_heads, nv = cfg.num_v_heads, dk = cfg.head_k_dim, dv = cfg.head_v_dim;
    const int v_dim = nv * dv, conv_dim = 2 * nk * dk + v_dim;
    append_norm(batch, input, layer.attn_norm_w, scratch.normalized, rows, hidden, cfg.rms_norm_eps);
    append_linear(batch, scratch.normalized, layer.qkv_proj, scratch.qkv, rows, conv_dim, hidden);
    append_linear(batch, scratch.normalized, layer.z_proj, scratch.z, rows, v_dim, hidden);
    append_linear(batch, scratch.normalized, layer.a_proj, scratch.a, rows, nv, hidden);
    append_linear(batch, scratch.normalized, layer.b_proj, scratch.b, rows, nv, hidden);
    vk_ops::PushGdnChunk push{0, rows, nk, nv, dk, dv, cfg.conv_kernel_size,
                              (layer.conv_bias.vk_buffer ? 1 : 0) | (cfg.tiled_value_heads ? 2 : 0),
                              cfg.rms_norm_eps, 1.0f / std::sqrt(static_cast<float>(dk))};
    const std::initializer_list<Tensor> buffers{
        scratch.qkv, scratch.z, scratch.a, scratch.b, layer.conv_w, layer.conv_bias, layer.A_log,
        layer.dt_bias, layer.norm_w, state.conv, state.ssm, scratch.recurrent, scratch.qkv};
    append(batch, "sequence/gdn_chunk_and_step.spv", vector_groups(1, conv_dim), buffers, push);
    push.mode = 1;
    append(batch, "sequence/gdn_chunk_and_step.spv", nv, buffers, push);
    append_linear(batch, scratch.recurrent, layer.o_proj, scratch.projected, rows, hidden, v_dim);
    append_residual(batch, input, scratch.projected, output, rows * hidden);
}

void VulkanDispatch::append_ffn(Batch& batch, const Tensor& norm, const Tensor& gate_up, const Tensor& gate,
                                const Tensor& up, const Tensor& down, Tensor& normalized, FfnScratch& scratch,
                                const Tensor& activation,
                                int hidden, int inter, int rows, float eps, bool geglu) {
    if (gate_up.vk_buffer) {
        append_norm_linear(batch, activation, norm, gate_up, normalized, scratch.gate_up,
                           rows, 2 * inter, hidden, eps);
        if (geglu) {
            append_columns(batch, scratch.gate_up, scratch.gate, rows, inter, 2 * inter, inter, 0, 0);
            append_columns(batch, scratch.gate_up, scratch.up, rows, inter, 2 * inter, inter, inter, 0);
        }
    } else {
        append_norm(batch, activation, norm, normalized, rows, hidden, eps);
        append_linear(batch, normalized, gate, scratch.gate, rows, inter, hidden);
        append_linear(batch, normalized, up, scratch.up, rows, inter, hidden);
        if (!geglu) {
            append_pack_rows(batch, scratch.gate, scratch.gate_up, rows, inter, 2 * inter, 0);
            append_pack_rows(batch, scratch.up, scratch.gate_up, rows, inter, 2 * inter, inter);
        }
    }
    if (geglu) append_geglu(batch, scratch.gate, scratch.up, scratch.activated, rows, inter);
    else append(batch, "swiglu.comp", element_groups(rows * inter), {scratch.gate_up, scratch.activated},
                vk_ops::PushSwiglu{rows, inter});
    append_linear(batch, scratch.activated, down, scratch.down, rows, hidden, inter);
}

void VulkanDispatch::append_finish_ffn(Batch& batch, const Tensor& activation, const Tensor& update,
                                      const Tensor& post_norm, int hidden, int rows, float eps, float scale) {
    if (post_norm.vk_buffer) append_norm(batch, update, post_norm, update, rows, hidden, eps);
    if (scale == 1.f) append_residual(batch, activation, update, activation, rows * hidden);
    else append_scale_add(batch, activation, update, activation, rows, hidden, scale, scale);
}

void VulkanDispatch::append_moe(Batch& batch, const VulkanMoeWeights& layer, MoeScratch& scratch,
                                const Tensor& input, const Tensor& output, int hidden, int rows) {
    const auto& cfg = layer.cfg;
    const int experts = cfg.num_experts, top_k = cfg.top_k, inter = cfg.intermediate_size;
    const bool geglu = layer.router_scale.vk_buffer != 0;
    append_norm(batch, input, layer.norm_w, scratch.normalized, rows, hidden, cfg.rms_norm_eps);
    if (geglu) append_norm(batch, input, layer.router_scale, scratch.router_input, rows, hidden, cfg.rms_norm_eps);
    append_linear(batch, geglu ? scratch.router_input : scratch.normalized, layer.router_w,
                  scratch.router_logits, rows, experts, hidden);
    vk_ops::PushMoeQuant push{0, rows, hidden, inter, experts, top_k,
        (cfg.sigmoid_router ? 1 : 0) | (layer.router_bias.vk_buffer ? 2 : 0) |
        (cfg.norm_topk_prob ? 4 : 0) | (layer.down_scale.vk_buffer ? 8 : 0),
        cfg.expert_groups, cfg.expert_groups_used, static_cast<int>(layer.gpu_gate_up_w.dtype),
        static_cast<int>(layer.up_dtype), static_cast<int>(layer.gpu_down_w.dtype), experts, 0, 0,
        layer.up_offset, 0, cfg.expert_weights_scale};
    append(batch, "epilogue/moe_quant_ops.spv", rows,
           {scratch.router_logits, scratch.indices, scratch.weights, layer.router_bias, {}, {}, {}, {}, layer.down_scale}, push);
    push.mode = 1;
    push.flags = (layer.up_offset ? 2 : 0) | (geglu ? 4 : 0) |
                 (layer.gate_scale.vk_buffer ? 8 : 0) | (layer.up_scale.vk_buffer ? 16 : 0);
    append(batch, "epilogue/moe_quant_ops.spv", inter,
           {scratch.normalized, scratch.indices, {}, layer.gate_scale, {}, layer.gate_region,
            scratch.intermediate, {}, layer.up_scale}, push, top_k, rows);
    if (cfg.has_shared_expert) {
        const int shared = cfg.shared_intermediate_size;
        append_linear(batch, scratch.normalized, layer.shared_gate_up_w, scratch.shared_gate_up, rows, 2 * shared, hidden);
        append(batch, "swiglu.comp", element_groups(rows * shared), {scratch.shared_gate_up, scratch.shared_activated},
               vk_ops::PushSwiglu{rows, shared});
        append_linear(batch, scratch.shared_activated, layer.shared_down_w, scratch.shared_down, rows, hidden, shared);
        if (layer.shared_gate_w.vk_buffer) {
            append_linear(batch, scratch.normalized, layer.shared_gate_w, scratch.shared_gate, rows, 1, hidden);
            const auto stride = static_cast<VkDeviceSize>(hidden) * sizeof(float);
            const int gate_push[] = {hidden, 1};
            for (int row = 0; row < rows; ++row) {
                const auto offset = row * stride;
                auto down = scratch.shared_down.slice_dim0(row, 1);
                const bool aligned = down.vk_offset % batch.ctx.info().min_storage_offset_align == 0;
                if (!aligned) {
                    append_copy(batch, scratch.shared_down, scratch.gated_row, {{offset, 0, stride}});
                    down = scratch.gated_row;
                }
                append_copy(batch, scratch.shared_gate, scratch.gate_scalar,
                            {{row * sizeof(float), 0, sizeof(float)}});
                append(batch, "sigmoid_gate.comp", element_groups(hidden), {down, scratch.gate_scalar}, gate_push);
                if (!aligned) append_copy(batch, down, scratch.shared_down, {{0, offset, stride}});
            }
        }
    }
    push.mode = 2;
    push.flags = (geglu ? 0 : 2) | (cfg.has_shared_expert ? 4 : 0);
    append(batch, "epilogue/moe_quant_ops.spv", hidden,
           {{}, scratch.indices, scratch.weights, input, {}, layer.gpu_down_w, scratch.intermediate, output,
            scratch.shared_down}, push, rows);
}

void VulkanDispatch::append_head(Batch& batch, const Tensor& norm, const Tensor& head, Tensor& normalized,
                                 Tensor& logits, const Tensor& input, int hidden, int vocab, float eps, float softcap) {
    append_norm_linear(batch, input, norm, head, normalized, logits, 1, vocab, hidden, eps);
    if (softcap > 0.f) {
        vk_ops::PushGemma push{2, 1, vocab};
        push.cap = softcap;
        append_gemma(batch, logits, {}, logits, push, vector_groups(1, vocab));
    }
}

}  // namespace fusionrt
