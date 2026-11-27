#include "vulkan_executor.h"
#include "vulkan_session.h"
#include "vulkan_weights.h"

#include "fusionrt/sampler.h"
#include "fusionrt/vulkan_dispatch.h"
#include "fusionrt/vulkan_kernels.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

namespace fusionrt {

namespace {

void execute_batch(VkContext& ctx, const VulkanDispatch::Batch& batch) {
    auto* trace = ctx.op_trace();
    if (!trace) {
        ctx.dispatch_batched(batch.ops, false);
        return;
    }
    for (const auto& d : batch.ops) {
        const auto name = std::filesystem::path(d.shader_name).stem().string();
        auto op = std::find_if(trace->ops.begin(), trace->ops.end(), [&](const auto& entry) { return entry.name == name; });
        if (op == trace->ops.end()) {
            trace->ops.push_back({name, name});
            op = trace->ops.end() - 1;
        }
        op->total_ms += ctx.dispatch_single(d, true);
        ++op->calls;
        if (!d.copies.empty()) {
            for (const auto& region : d.copies) op->bytes += 2 * region.size;
        } else {
            for (const auto& tensor : d.buffers) op->bytes += tensor.nbytes();
            if (name.starts_with("gemv_") || name.starts_with("gemm_") || name == "fused_rmsnorm_gemv_vec4") {
                vk_ops::PushGemm shape{};
                std::memcpy(&shape, d.push_data, sizeof(shape));
                op->flops += int64_t{2} * shape.M * shape.N * shape.K;
            }
        }
    }
}

}  // namespace

bool VulkanExecutor::supports_batch_prefill() const {
    return weights_.gdn_layers().empty() && weights_.moe_layers().empty();
}

VkDeviceSize VulkanExecutor::scratch_bytes(const VkContext& ctx, int rows) const {
    const auto& cfg = weights_.config();
    const auto h = weights_.embeddings().shape[1];
    const auto align = ctx.info().min_storage_offset_align;
    const auto padded = [align](VkDeviceSize elements) {
        return ((elements * sizeof(float) + align - 1) / align) * align;
    };
    const auto tile = [&](int64_t width) { return padded(rows * width); };
    const auto ffn = [&](const auto& layer) {
        const auto inter = layer.cfg.intermediate_size;
        return tile(2 * inter) + tile(inter) + tile(h) + (layer.gate_proj.vk_buffer ? 2 * tile(inter) : 0);
    };

    VkDeviceSize moe_scratch = 0;
    for (const auto& layer : weights_.moe_layers()) {
        const auto& moe = layer.cfg;
        auto bytes = 2 * tile(h) + padded(moe.num_experts) + 2 * padded(moe.top_k) +
                     padded(static_cast<VkDeviceSize>(moe.top_k) * moe.intermediate_size);
        if (moe.has_shared_expert)
            bytes += tile(2 * moe.shared_intermediate_size) + tile(moe.shared_intermediate_size) + tile(h) + padded(1);
        moe_scratch = std::max(moe_scratch, bytes);
    }

    auto layer_scratch = moe_scratch;
    for (const auto& layer : weights_.dense_layers()) {
        const auto& d = layer.cfg;
        const int q = d.num_attention_heads * d.head_dim, kv = d.num_key_value_heads * d.head_dim;
        auto bytes = 2 * tile(h) + 2 * tile(q) + tile(q + 2 * kv);
        if (!layer.qkv_proj_fused.vk_buffer) {
            bytes += 2 * tile(kv);
            if (d.gated_q_attn) bytes += tile(2 * q) + tile(q);
            if (rows > 1 && (layer.q_bias.vk_buffer || layer.k_bias.vk_buffer || layer.v_bias.vk_buffer))
                bytes += tile(std::max(d.gated_q_attn ? 2 * q : q, kv));
        }
        layer_scratch = std::max(layer_scratch, bytes + (layer.moe_index >= 0 ? moe_scratch : ffn(layer)));
    }
    for (const auto& layer : weights_.gdn_layers()) {
        const auto& g = layer.cfg;
        const int v = g.num_v_heads * g.head_v_dim, conv = 2 * g.num_k_heads * g.head_k_dim + v;
        const auto bytes = 2 * tile(h) + tile(conv) + 2 * tile(v) + 2 * tile(g.num_v_heads);
        layer_scratch = std::max(layer_scratch, bytes + (layer.moe_index >= 0 ? moe_scratch : ffn(layer)));
    }
    const auto head = padded(cfg.vocab_size) + (rows > 1 ? 2 : 1) * padded(h);
    return 2 * tile(h) + padded(rows) + std::max(head, layer_scratch);
}

std::vector<float> VulkanExecutor::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    return forward_batch_vulkan(ctx, {&token_id, 1}, pos);
}

std::vector<float> VulkanExecutor::forward_batch_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos,
                                                       bool compute_logits) {
    const auto& cfg = weights_.config();
    if (start_pos < 0 || start_pos >= cfg.dense.max_seq_len || tokens.empty() ||
        tokens.size() > static_cast<size_t>(cfg.dense.max_seq_len - start_pos))
        throw std::invalid_argument("Token batch exceeds context or is empty");
    validate_token_ids(tokens, start_pos, cfg.vocab_size);
    const int rows = static_cast<int>(tokens.size());
    if (rows > 1 && !supports_batch_prefill())
        throw std::invalid_argument("Vulkan batch forward requires a Dense model");
    if (!ctx.try_reserve_scratch_memory(scratch_bytes(ctx, rows)))
        throw std::runtime_error("Vulkan scratch arena cannot fit the token batch");

    const int hidden = static_cast<int>(weights_.embeddings().shape[1]);
    auto& arena = ctx.scratch_arena();
    struct RestoreArena {
        VkArena& arena;
        VkDeviceSize mark;
        ~RestoreArena() { arena.reset(mark); }
    } restore{arena, arena.mark()};
    auto scratch = [&](int width, int count = 0) {
        return arena.alloc_tensor({count ? count : rows, width}, DType::F32);
    };

    VulkanDispatch::Batch batch{ctx};
    auto host_embeddings = weights_.gather_host_embeddings(tokens);
    Tensor hidden_a, hidden_b = scratch(hidden);
    if (host_embeddings.bytes().empty()) {
        HostTensor token_ids({rows}, DType::I32);
        std::copy(tokens.begin(), tokens.end(), token_ids.data<int32_t>());
        const Tensor token = ctx.upload(token_ids, false);
        hidden_a = scratch(hidden);
        VulkanDispatch::append_embedding(batch, weights_.embeddings(), token, hidden_a, hidden, rows);
    } else {
        hidden_a = ctx.upload(host_embeddings, false);
    }
    Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;

    auto append_moe = [&](const VkSparseMoELayer& layer, const Tensor& input, const Tensor& output) {
        VulkanDispatch::MoeScratch moe;
        moe.normalized = scratch(hidden);
        moe.router_logits = scratch(layer.cfg.num_experts);
        moe.indices = arena.alloc_tensor({layer.cfg.top_k}, DType::I32);
        moe.weights = scratch(layer.cfg.top_k);
        moe.intermediate = scratch(layer.cfg.intermediate_size, layer.cfg.top_k);
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
    auto append_ffn = [&](const auto& layer, Tensor& normalized) {
        if (layer.moe_index >= 0) {
            append_moe(weights_.moe_layers()[layer.moe_index], *next, *next);
            return;
        }
        const int inter = layer.cfg.intermediate_size;
        VulkanDispatch::FfnScratch ffn{scratch(2 * inter), scratch(inter), scratch(hidden)};
        if (layer.gate_proj.vk_buffer) {
            ffn.gate = scratch(inter);
            ffn.up = scratch(inter);
        }
        VulkanDispatch::append_ffn(batch, layer.mlp_norm_w, layer.gate_up_proj, layer.gate_proj, layer.up_proj,
                                   layer.down_proj, normalized, ffn, *next, hidden, inter, rows, layer.cfg.rms_norm_eps);
    };

    // Each layer's temporaries die before the next layer. The batch records barriers between all operations.
    const auto layer_mark = arena.mark();
    for (size_t i = 0; i < weights_.layer_kinds().size(); ++i) {
        const int kind = weights_.layer_kinds()[i];
        const size_t index = weights_.layer_indices()[i];
        if (kind == 0) {
            const auto& layer = weights_.dense_layers()[index];
            const int q_dim = layer.cfg.num_attention_heads * layer.cfg.head_dim;
            const int kv_dim = layer.cfg.num_key_value_heads * layer.cfg.head_dim;
            VulkanDispatch::DenseScratch dense;
            dense.normalized = scratch(hidden);
            dense.qkv = scratch(q_dim + 2 * kv_dim);
            dense.query = scratch(q_dim);
            if (!layer.qkv_proj_fused.vk_buffer) {
                dense.key = scratch(kv_dim);
                dense.value = scratch(kv_dim);
                const int q_width = layer.cfg.gated_q_attn ? 2 * q_dim : q_dim;
                if (layer.cfg.gated_q_attn) {
                    dense.query_gate = scratch(q_dim);
                    dense.query_full = scratch(q_width);
                }
                if (rows > 1 && (layer.q_bias.vk_buffer || layer.k_bias.vk_buffer || layer.v_bias.vk_buffer))
                    dense.bias = scratch(std::max(q_width, kv_dim));
            }
            dense.attention = scratch(q_dim);
            dense.projected = scratch(hidden);
            VulkanDispatch::append_dense(batch, layer, sessions_.dense(index), dense,
                                         *current, *next, hidden, start_pos, rows);
            append_ffn(layer, dense.normalized);
        } else if (kind == 1) {
            const auto& layer = weights_.gdn_layers()[index];
            const auto& gdn_cfg = layer.cfg;
            const int v_dim = gdn_cfg.num_v_heads * gdn_cfg.head_v_dim;
            const int conv_dim = 2 * gdn_cfg.num_k_heads * gdn_cfg.head_k_dim + v_dim;
            VulkanDispatch::GdnScratch gdn{scratch(hidden), scratch(conv_dim), scratch(v_dim),
                                            scratch(gdn_cfg.num_v_heads), scratch(gdn_cfg.num_v_heads),
                                            scratch(v_dim), scratch(hidden)};
            VulkanDispatch::append_gdn(batch, layer, sessions_.gdn(index), gdn, *current, *next, hidden);
            append_ffn(layer, gdn.normalized);
        } else {
            append_moe(weights_.moe_layers()[index], *current, *next);
        }
        std::swap(current, next);
        arena.reset(layer_mark);
    }

    Tensor logits;
    if (compute_logits) {
        Tensor last = *current;
        if (rows > 1) {
            last = scratch(hidden, 1);
            const auto row_bytes = static_cast<VkDeviceSize>(hidden) * sizeof(float);
            VulkanDispatch::append_copy(batch, *current, last, {{(rows - 1) * row_bytes, 0, row_bytes}});
        }
        auto final_normalized = scratch(hidden, 1);
        logits = scratch(cfg.vocab_size, 1);
        VulkanDispatch::append_head(batch, weights_.final_norm(), weights_.lm_head(), final_normalized, logits,
                                    last, hidden, cfg.vocab_size, cfg.dense.rms_norm_eps);
    }
    execute_batch(ctx, batch);
    if (!compute_logits) return {};
    const HostTensor host_logits = ctx.download(logits);
    return {host_logits.data<float>(), host_logits.data<float>() + cfg.vocab_size};
}

std::vector<float> VulkanExecutor::prefill_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos) {
    size_t chunk = supports_batch_prefill() ? std::min<size_t>(ctx.prefill_batch(), tokens.size()) : 1;
    while (chunk > 1 && !ctx.try_reserve_scratch_memory(scratch_bytes(ctx, static_cast<int>(chunk)))) chunk /= 2;
    std::vector<float> logits;
    for (size_t begin = 0; begin < tokens.size(); begin += chunk) {
        const auto rows = std::min(chunk, tokens.size() - begin);
        logits = forward_batch_vulkan(ctx, tokens.subspan(begin, rows), start_pos + static_cast<int>(begin),
                                     begin + rows == tokens.size());
    }
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
