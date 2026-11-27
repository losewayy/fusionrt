#include "vulkan_executor.h"
#include "vulkan_session.h"
#include "vulkan_weights.h"

#include "fusionrt/sampler.h"
#include "fusionrt/vulkan_dispatch.h"

#include <algorithm>
#include <chrono>

namespace fusionrt {

void VulkanExecutor::allocate_workspace(VkContext& ctx, int hidden, int vocab) {
    auto& arena = ctx.weight_arena();
    hidden_a_ = arena.alloc_tensor({hidden}, DType::F32, "vk_ha");
    hidden_b_ = arena.alloc_tensor({hidden}, DType::F32, "vk_hb");
    logits_ = arena.alloc_tensor({vocab}, DType::F32, "vk_logits");
    token_slot_ = arena.alloc_tensor({1}, DType::I32, "vk_tid");
}

std::vector<float> VulkanExecutor::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    const auto& cfg = weights_.config();
    if (pos < 0 || pos >= cfg.dense.max_seq_len) throw std::invalid_argument("Token position exceeds context");
    validate_token_id(token_id, pos, cfg.vocab_size);
    const int hidden = static_cast<int>(hidden_a_.shape[0]);
    HostTensor host_token = HostTensor::create({1}, DType::I32);
    host_token.data<int32_t>()[0] = token_id;
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const Tensor token = ctx.upload(host_token, false);
    auto scratch = [&](int width) { return arena.alloc_tensor({width}, DType::F32); };

    VulkanDispatch::Batch batch;
    VulkanDispatch::append_embedding(batch, weights_.embeddings(), token, hidden_a_, hidden);
    Tensor* current = &hidden_a_;
    Tensor* next = &hidden_b_;

    auto append_moe = [&](const VkSparseMoELayer& layer, const Tensor& input, const Tensor& output) {
        VulkanDispatch::MoeScratch moe;
        moe.normalized = scratch(hidden);
        moe.router_logits = scratch(layer.cfg.num_experts);
        moe.indices = arena.alloc_tensor({layer.cfg.top_k}, DType::I32);
        moe.weights = scratch(layer.cfg.top_k);
        moe.intermediate = arena.alloc_tensor({layer.cfg.top_k, layer.cfg.intermediate_size}, DType::F32);
        moe.output = scratch(hidden);
        if (layer.cfg.has_shared_expert) {
            const int shared = layer.cfg.shared_intermediate_size;
            moe.shared_gate_up = scratch(2 * shared);
            moe.shared_activated = scratch(shared);
            moe.shared_down = scratch(hidden);
            if (layer.shared_gate_w.vk_buffer) moe.shared_gate = scratch(1);
        }
        VulkanDispatch::append_moe(batch, layer, moe, input, output, hidden);
    };
    auto append_ffn = [&](const auto& layer, Tensor& normalized, VulkanDispatch::FfnScratch& ffn) {
        if (layer.moe_index >= 0) {
            append_moe(weights_.moe_layers()[layer.moe_index], *next, *next);
        } else {
            VulkanDispatch::append_ffn(batch, layer.mlp_norm_w, layer.gate_up_proj, layer.down_proj, normalized,
                                       ffn, *next, hidden, layer.cfg.intermediate_size, layer.cfg.rms_norm_eps);
        }
    };

    for (size_t i = 0; i < weights_.layer_kinds().size(); ++i) {
        const int kind = weights_.layer_kinds()[i];
        const size_t index = weights_.layer_indices()[i];
        if (kind == 0) {
            const auto& layer = weights_.dense_layers()[index];
            const int q_dim = layer.cfg.num_attention_heads * layer.cfg.head_dim;
            const int kv_dim = layer.cfg.num_key_value_heads * layer.cfg.head_dim;
            const int inter = layer.cfg.intermediate_size;
            VulkanDispatch::DenseScratch dense;
            dense.normalized = scratch(hidden);
            dense.query = scratch(q_dim);
            if (layer.cfg.gated_q_attn) {
                dense.query_gate = scratch(q_dim);
                dense.query_full = scratch(2 * q_dim);
            }
            dense.key = scratch(kv_dim);
            dense.value = scratch(kv_dim);
            dense.attention = scratch(q_dim);
            dense.projected = scratch(hidden);
            VulkanDispatch::FfnScratch ffn{scratch(2 * inter), scratch(inter), scratch(hidden)};
            VulkanDispatch::append_dense(batch, layer, sessions_.dense(index), dense, *current, *next, hidden, pos);
            append_ffn(layer, dense.normalized, ffn);
        } else if (kind == 1) {
            const auto& layer = weights_.gdn_layers()[index];
            const auto& gdn_cfg = layer.cfg;
            const int v_dim = gdn_cfg.num_v_heads * gdn_cfg.head_v_dim;
            const int conv_dim = 2 * gdn_cfg.num_k_heads * gdn_cfg.head_k_dim + v_dim;
            const int inter = gdn_cfg.intermediate_size;
            VulkanDispatch::GdnScratch gdn{scratch(hidden), scratch(conv_dim), scratch(v_dim),
                                            scratch(gdn_cfg.num_v_heads), scratch(gdn_cfg.num_v_heads),
                                            scratch(v_dim), scratch(hidden)};
            VulkanDispatch::FfnScratch ffn{scratch(2 * inter), scratch(inter), scratch(hidden)};
            VulkanDispatch::append_gdn(batch, layer, sessions_.gdn(index), gdn, *current, *next, hidden);
            append_ffn(layer, gdn.normalized, ffn);
        } else {
            append_moe(weights_.moe_layers()[index], *current, *next);
        }
        std::swap(current, next);
    }

    Tensor final_normalized = scratch(hidden);
    VulkanDispatch::append_head(batch, weights_.final_norm(), weights_.lm_head(), final_normalized, logits_,
                                *current, hidden, cfg.vocab_size, cfg.dense.rms_norm_eps);
    ctx.dispatch_batched(batch.ops, false);
    HostTensor host_logits = ctx.download(logits_);
    arena.reset(mark);
    return {host_logits.data<float>(), host_logits.data<float>() + cfg.vocab_size};
}

std::vector<float> VulkanExecutor::prefill_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos) {
    std::vector<float> logits;
    for (int32_t token : tokens) logits = forward_token_vulkan(ctx, token, start_pos++);
    return logits;
}

std::vector<int32_t> VulkanExecutor::generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                     int max_new_tokens, GenerationStats* stats,
                                                     bool keep_state, int start_pos, int32_t stop_eos_id) {
    const auto& cfg = weights_.config();
    if (max_new_tokens < 0 || start_pos < 0) throw std::invalid_argument("Invalid generation length or start position");
    if (!keep_state) start_pos = 0;
    std::vector<int32_t> tokens = prompt_ids;
    if (tokens.empty()) tokens.push_back(std::min(1, cfg.vocab_size - 1));
    validate_token_ids(tokens, start_pos, cfg.vocab_size);
    const auto total_tokens = static_cast<size_t>(start_pos) + tokens.size();
    const int context = cfg.dense.max_seq_len;
    if (total_tokens > static_cast<size_t>(context)) throw prompt_context_error(total_tokens, context);
    max_new_tokens = std::min(max_new_tokens, context - static_cast<int>(total_tokens));
    const int prompt_len = static_cast<int>(tokens.size());
    if (!keep_state) sessions_.reset_kv_and_state_vulkan(ctx);

    auto tp0 = std::chrono::high_resolution_clock::now();
    auto logits = prefill_vulkan(ctx, tokens, start_pos);
    auto tp1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    int pos = start_pos + prompt_len;
    int gen_count = 0;
    int decode_steps = 0;
    auto td0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < max_new_tokens; ++step) {
        const int32_t next_tok = Sampler::argmax(logits.data(), static_cast<int>(logits.size()));
        tokens.push_back(next_tok);
        ++gen_count;
        const bool eos = next_tok == stop_eos_id;
        // A continued turn starts after the last emitted token, including EOS.
        if (keep_state || (!eos && step + 1 < max_new_tokens)) {
            logits = forward_token_vulkan(ctx, next_tok, pos++);
            ++decode_steps;
        }
        if (eos) break;
    }
    auto td1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();

    if (stats) {
        stats->prompt_tokens = static_cast<int>(prompt_ids.size());
        stats->generated_tokens = gen_count;
        stats->prefill_ms = prefill_ms;
        stats->decode_ms = decode_ms;
        stats->prefill_tok_per_s = (prefill_ms > 0.0) ? (static_cast<double>(prompt_len) * 1000.0 / prefill_ms) : 0.0;
        stats->decode_tok_per_s = decode_ms > 0.0 ? static_cast<double>(decode_steps) * 1000.0 / decode_ms : 0.0;
    }
    return tokens;
}

}  // namespace fusionrt
