#include "vulkan_executor.h"
#include "vulkan_session.h"
#include "vulkan_weights.h"

#include "fusionrt/models.h"
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

VkDeviceSize VulkanExecutor::scratch_bytes(const VkContext& ctx, int rows, bool embeds) const {
    const auto& cfg = weights_.config();
    const auto h = weights_.embeddings().shape[1];
    const auto align = ctx.info().min_storage_offset_align;
    const auto padded = [align](VkDeviceSize elements) {
        return ((elements * sizeof(float) + align - 1) / align) * align;
    };
    const auto tile = [&](int64_t width) { return padded(rows * width); };
    const auto ffn = [&](const auto& layer, bool geglu) {
        const auto inter = layer.cfg.intermediate_size;
        return tile(2 * inter) + tile(inter) + tile(h) + ((layer.gate_proj.vk_buffer || geglu) ? 2 * tile(inter) : 0);
    };

    VkDeviceSize moe_scratch = 0;
    for (const auto& layer : weights_.moe_layers()) {
        const auto& moe = layer.cfg;
        auto bytes = tile(h) + tile(moe.num_experts) + 2 * tile(moe.top_k) +
                     tile(static_cast<int64_t>(moe.top_k) * moe.intermediate_size);
        if (layer.router_scale.vk_buffer) bytes += tile(h);
        if (moe.has_shared_expert) {
            bytes += tile(2 * moe.shared_intermediate_size) + tile(moe.shared_intermediate_size) + tile(h);
            if (layer.shared_gate_w.vk_buffer) bytes += tile(1) + padded(1) + padded(h);
        }
        moe_scratch = std::max(moe_scratch, bytes);
    }

    auto layer_scratch = moe_scratch;
    for (const auto& layer : weights_.dense_layers()) {
        const auto& d = layer.cfg;
        const int heads = d.num_attention_heads, q = heads * d.head_dim, kv = d.num_key_value_heads * d.head_dim;
        auto bytes = 2 * tile(h);
        if (d.mla.latent_rank) {
            const auto& m = d.mla;
            const int rank = m.latent_rank, rope = m.rope_dims, nope = d.head_dim - rope;
            bytes += tile(m.query_rank) + tile(q) + tile(m.latent_width()) + tile(heads * rope) +
                     2 * tile(heads * rank) + tile(heads * m.value_dim) +
                     tile(std::max({nope, rank, rope})) + tile(std::max(rank, m.value_dim));
        } else {
            bytes += 2 * tile(q) + tile(layer.kv_source < 0 ? q + 2 * kv : q);
            if (!layer.qkv_proj_fused.vk_buffer) {
                if (layer.kv_source < 0) bytes += 2 * tile(kv);
                if (d.gated_q_attn) bytes += tile(2 * q) + tile(q);
                if (rows > 1 && (layer.q_bias.vk_buffer || layer.k_bias.vk_buffer || layer.v_bias.vk_buffer))
                    bytes += tile(std::max(d.gated_q_attn ? 2 * q : q, kv));
            }
            if (embeds && q * sizeof(float) % align != 0) bytes += 2 * tile(q);
        }
        if (layer.moe_index >= 0) bytes += moe_scratch;
        if (layer.moe_index < 0 || layer.dense_post_norm.vk_buffer) bytes += ffn(layer, layer.gemma4);
        if (layer.dense_post_norm.vk_buffer) bytes += tile(h);
        if (layer.ple_gate.vk_buffer) bytes += 2 * tile(layer.ple_gate.shape[0]) + tile(h);
        layer_scratch = std::max(layer_scratch, bytes);
    }
    for (const auto& layer : weights_.gdn_layers()) {
        const auto& g = layer.cfg;
        const int v = g.num_v_heads * g.head_v_dim, conv = 2 * g.num_k_heads * g.head_k_dim + v;
        const auto bytes = 2 * tile(h) + tile(conv) + 2 * tile(v) + 2 * tile(g.num_v_heads);
        layer_scratch = std::max(layer_scratch, bytes + (layer.moe_index >= 0 ? moe_scratch : ffn(layer, false)));
    }
    const auto head = padded(cfg.vocab_size) + (rows > 1 ? 2 : 1) * padded(h);
    const auto& ple = weights_.gemma_inputs().token_embeddings;
    const auto ple_cols = ple.vk_buffer ? ple.shape[1] : 0;
    const auto persistent = (embeds ? 3 : 2) * tile(h) + padded(rows) + tile(ple_cols) + tile(weights_.hadamard_width());
    return persistent + std::max({head, layer_scratch, tile(ple_cols)});
}

std::vector<float> VulkanExecutor::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    return forward_batch_vulkan(ctx, {&token_id, 1}, pos);
}

std::vector<float> VulkanExecutor::forward_batch_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos,
                                                       bool compute_logits, std::span<const InputEmbedding> embeds,
                                                       int prompt_offset) {
    const auto& cfg = weights_.config();
    if (start_pos < 0 || start_pos >= cfg.dense.max_seq_len || tokens.empty() ||
        tokens.size() > static_cast<size_t>(cfg.dense.max_seq_len - start_pos))
        throw std::invalid_argument("Token batch exceeds context or is empty");
    validate_token_ids(tokens, start_pos, cfg.vocab_size);
    const int rows = static_cast<int>(tokens.size());
    if (!ctx.try_reserve_scratch_memory(scratch_bytes(ctx, rows, !embeds.empty())))
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
    batch.dummy = weights_.dummy();
    batch.hadamard_signs = &weights_.hadamard_signs();
    if (weights_.hadamard_width()) batch.rotated = scratch(weights_.hadamard_width());

    HostTensor ids({rows}, DType::I32);
    std::copy(tokens.begin(), tokens.end(), ids.data<int32_t>());
    const auto token = ctx.upload(ids, false);
    const auto host_embeddings = weights_.gather_host_embeddings(tokens);
    Tensor hidden_a = host_embeddings.bytes().empty() ? scratch(hidden) : ctx.upload(host_embeddings, false);
    Tensor hidden_b = scratch(hidden);
    if (host_embeddings.bytes().empty())
        VulkanDispatch::append_embedding(batch, weights_.embeddings(), token, hidden_a, hidden, rows);
    VulkanDispatch::append_hadamard(batch, weights_.embeddings(), hidden_a, hidden_a, rows, hidden, true);
    if (cfg.embed_scale != 1.0f)
        VulkanDispatch::append_scale_add(batch, hidden_a, {}, hidden_a, rows, hidden, cfg.embed_scale, 0.f);

    std::vector<VulkanDispatch::AttentionSpan> image_spans;
    std::vector<VkBufferCopy> overrides;
    HostTensor override_rows;
    for (const auto& span : embeds) {
        const int end = span.start + static_cast<int>(span.host_rows.desc.shape[0]);
        const int begin_row = std::max(span.start - prompt_offset, 0);
        const int end_row = std::min(end - prompt_offset, rows);
        if (end_row <= begin_row) continue;
        if (override_rows.bytes().empty()) override_rows = HostTensor({rows, hidden}, DType::F32);
        const auto stride = static_cast<VkDeviceSize>(hidden) * sizeof(float);
        const auto offset = begin_row * stride;
        const auto bytes = (end_row - begin_row) * stride;
        std::memcpy(override_rows.storage.data() + offset, span.host_rows.bytes().data() +
                    (prompt_offset + begin_row - span.start) * stride, bytes);
        overrides.push_back({offset, offset, bytes});
        if (span.bidirectional) image_spans.push_back({begin_row, end_row});
    }
    if (!overrides.empty()) {
        const auto replacement = ctx.upload(override_rows, false);
        VulkanDispatch::append_copy(batch, replacement, hidden_a, std::move(overrides));
    }
    std::sort(image_spans.begin(), image_spans.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });

    Tensor ple;
    const auto& g = weights_.gemma_inputs();
    if (g.token_embeddings.vk_buffer) {
        const int cols = static_cast<int>(g.token_embeddings.shape[1]);
        const int width = static_cast<int>(g.norm.shape[0]);
        const auto host_ple = weights_.gather_host_ple(tokens);
        ple = host_ple.bytes().empty() ? scratch(cols) : ctx.upload(host_ple, false);
        const auto mark = arena.mark();
        auto projection = scratch(cols);
        VulkanDispatch::append_linear(batch, hidden_a, g.projection, projection, rows, cols, hidden);
        // Absorb the projection scale 1/sqrt(H) into RMSNorm epsilon, as in CUDA.
        VulkanDispatch::append_gemma_norm(batch, projection, g.norm, projection, rows * (cols / width), width,
                                    cfg.dense.rms_norm_eps * hidden);
        if (host_ple.bytes().empty())
            VulkanDispatch::append_embedding(batch, g.token_embeddings, token, ple, cols, rows);
        const float mix = std::sqrt(0.5f);
        VulkanDispatch::append_scale_add(batch, ple, projection, ple, rows, cols, std::sqrt(static_cast<float>(width)) * mix, mix);
        arena.reset(mark);
    }
    Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;

    auto append_moe = [&](const VulkanMoeWeights& layer, const Tensor& input, const Tensor& output) {
        VulkanDispatch::MoeScratch moe;
        moe.normalized = scratch(hidden);
        if (layer.router_scale.vk_buffer) moe.router_input = scratch(hidden);
        moe.router_logits = scratch(layer.cfg.num_experts);
        moe.indices = arena.alloc_tensor({rows, layer.cfg.top_k}, DType::I32);
        moe.weights = scratch(layer.cfg.top_k);
        moe.intermediate = scratch(layer.cfg.top_k * layer.cfg.intermediate_size);
        if (layer.cfg.has_shared_expert) {
            const int shared = layer.cfg.shared_intermediate_size;
            moe.shared_gate_up = scratch(2 * shared);
            moe.shared_activated = scratch(shared);
            moe.shared_down = scratch(hidden);
            if (layer.shared_gate_w.vk_buffer) {
                moe.shared_gate = scratch(1);
                moe.gate_scalar = scratch(1, 1);
                moe.gated_row = scratch(hidden, 1);
            }
        }
        VulkanDispatch::append_moe(batch, layer, moe, input, output, hidden, rows);
    };
    const auto dense_ffn = [&](const auto& layer, Tensor& normalized, bool geglu) {
        const int inter = layer.cfg.intermediate_size;
        VulkanDispatch::FfnScratch ffn{scratch(2 * inter), scratch(inter), scratch(hidden)};
        if (layer.gate_proj.vk_buffer || geglu) {
            ffn.gate = scratch(inter);
            ffn.up = scratch(inter);
        }
        VulkanDispatch::append_ffn(batch, layer.mlp_norm_w, layer.gate_up_proj, layer.gate_proj, layer.up_proj,
                                   layer.down_proj, normalized, ffn, *next, hidden, inter, rows, layer.cfg.rms_norm_eps, geglu);
        return ffn.down;
    };

    // Each layer's temporaries die before the next layer. The batch records barriers between all operations.
    const auto layer_mark = arena.mark();
    for (size_t i = 0; i < weights_.layer_kinds().size(); ++i) {
        const int kind = weights_.layer_kinds()[i];
        const size_t index = weights_.layer_indices()[i];
        if (kind == 0) {
            const auto& layer = weights_.dense_layers()[index];
            const auto& d = layer.cfg;
            const int heads = d.num_attention_heads;
            Tensor normalized = scratch(hidden);
            const auto& kv = sessions_.dense(index);
            if (d.mla.latent_rank > 0) {
                const auto& m = d.mla;
                const int rank = m.latent_rank, rope = m.rope_dims;
                VulkanDispatch::MlaScratch mla{
                    normalized, m.query_rank ? scratch(m.query_rank) : Tensor{}, scratch(heads * d.head_dim),
                    scratch(m.latent_width()), scratch(heads * rope), scratch(heads * rank), scratch(heads * rank),
                    scratch(heads * m.value_dim), scratch(hidden), scratch(std::max({d.head_dim - rope, rank, rope})),
                    scratch(std::max(rank, m.value_dim))};
                VulkanDispatch::append_mla(batch, layer, kv, mla, *current, *next, hidden, start_pos, rows);
            } else {
                const int q_dim = heads * d.head_dim, kv_dim = d.num_key_value_heads * d.head_dim;
                VulkanDispatch::DenseScratch dense;
                dense.normalized = normalized;
                dense.qkv = scratch(layer.kv_source < 0 ? q_dim + 2 * kv_dim : q_dim);
                dense.query = scratch(q_dim);
                if (!layer.qkv_proj_fused.vk_buffer) {
                    if (layer.kv_source < 0) {
                        dense.key = scratch(kv_dim);
                        dense.value = scratch(kv_dim);
                    }
                    const int q_width = d.gated_q_attn ? 2 * q_dim : q_dim;
                    if (d.gated_q_attn) {
                        dense.query_gate = scratch(q_dim);
                        dense.query_full = scratch(q_width);
                    }
                    if (rows > 1 && (layer.q_bias.vk_buffer || layer.k_bias.vk_buffer || layer.v_bias.vk_buffer))
                        dense.bias = scratch(std::max(q_width, kv_dim));
                }
                dense.attention = scratch(q_dim);
                dense.projected = scratch(hidden);
                if (!image_spans.empty() && q_dim * sizeof(float) % ctx.info().min_storage_offset_align != 0) {
                    dense.image_query = scratch(q_dim);
                    dense.image_attention = scratch(q_dim);
                }
                VulkanDispatch::append_dense(batch, layer, kv, dense, *current, *next, hidden, start_pos, rows, image_spans);
            }
            if (layer.moe_index >= 0 && !layer.dense_post_norm.vk_buffer) {
                append_moe(weights_.moe_layers()[layer.moe_index], *next, *next);
            } else {
                const auto update = dense_ffn(layer, normalized, layer.gemma4);
                if (layer.dense_post_norm.vk_buffer) {
                    const auto routed = scratch(hidden);
                    append_moe(weights_.moe_layers()[layer.moe_index], *next, routed);
                    VulkanDispatch::append_norm(batch, routed, layer.routed_post_norm, routed, rows, hidden, d.rms_norm_eps);
                    VulkanDispatch::append_norm(batch, update, layer.dense_post_norm, update, rows, hidden, d.rms_norm_eps);
                    VulkanDispatch::append_scale_add(batch, update, routed, update, rows, hidden, 1.f, 1.f);
                }
                VulkanDispatch::append_finish_ffn(batch, *next, update, layer.ffn_post_norm, hidden, rows,
                                                  d.rms_norm_eps, layer.ple_gate.vk_buffer ? 1.f : layer.output_scale);
            }
            if (layer.ple_gate.vk_buffer) {
                const int width = static_cast<int>(layer.ple_gate.shape[0]);
                auto gate = scratch(width);
                const auto input = scratch(width);
                auto update = scratch(hidden);
                const auto stride = static_cast<VkDeviceSize>(ple.shape[1]) * sizeof(float);
                const auto bytes = static_cast<VkDeviceSize>(width) * sizeof(float);
                std::vector<VkBufferCopy> copies;
                copies.reserve(rows);
                for (int row = 0; row < rows; ++row)
                    copies.push_back({row * stride + index * bytes, row * bytes, bytes});
                VulkanDispatch::append_copy(batch, ple, input, std::move(copies));
                VulkanDispatch::append_linear(batch, *next, layer.ple_gate, gate, rows, width, hidden);
                VulkanDispatch::append_geglu(batch, gate, input, gate, rows, width);
                VulkanDispatch::append_linear(batch, gate, layer.ple_projection, update, rows, hidden, width);
                VulkanDispatch::append_finish_ffn(batch, *next, update, layer.ple_post_norm, hidden, rows,
                                                  d.rms_norm_eps, layer.output_scale);
            }
        } else if (kind == 1) {
            const auto& layer = weights_.gdn_layers()[index];
            const auto& gdn_cfg = layer.cfg;
            const int v_dim = gdn_cfg.num_v_heads * gdn_cfg.head_v_dim;
            const int conv_dim = 2 * gdn_cfg.num_k_heads * gdn_cfg.head_k_dim + v_dim;
            VulkanDispatch::GdnScratch gdn{scratch(hidden), scratch(conv_dim), scratch(v_dim),
                                            scratch(gdn_cfg.num_v_heads), scratch(gdn_cfg.num_v_heads),
                                            scratch(v_dim), scratch(hidden)};
            VulkanDispatch::append_gdn(batch, layer, sessions_.gdn(index), gdn, *current, *next, hidden, rows);
            if (layer.moe_index >= 0) append_moe(weights_.moe_layers()[layer.moe_index], *next, *next);
            else VulkanDispatch::append_finish_ffn(batch, *next, dense_ffn(layer, gdn.normalized, false), {},
                                                   hidden, rows, gdn_cfg.rms_norm_eps, 1.f);
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
                                    last, hidden, cfg.vocab_size, cfg.dense.rms_norm_eps, cfg.final_logit_softcapping);
    }
    execute_batch(ctx, batch);
    if (!compute_logits) return {};
    const HostTensor host_logits = ctx.download(logits);
    return {host_logits.data<float>(), host_logits.data<float>() + cfg.vocab_size};
}

std::vector<float> VulkanExecutor::prefill_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos,
                                                 const GenerationOptions& options, int& prefilled, bool compute_logits) {
    const auto& cfg = weights_.config();
    const int hidden = static_cast<int>(weights_.embeddings().shape[1]);
    const auto cancel = [&] { return options.cancel && options.cancel->load(std::memory_order_relaxed); };
    std::vector<std::pair<int, int>> ranges;
    for (const auto& span : options.inputs_embeds) {
        const auto& t = span.host_rows;
        if (span.start < 0 || t.desc.ndim != 2 || t.desc.dtype != DType::F32 || t.desc.shape[0] <= 0 ||
            t.desc.shape[1] != hidden || t.bytes().size() != t.desc.nbytes() ||
            static_cast<size_t>(span.start) > tokens.size() ||
            static_cast<size_t>(t.desc.shape[0]) > tokens.size() - span.start)
            throw std::invalid_argument("Invalid Vulkan input embedding span");
        if (span.bidirectional && cfg.dense.mla.latent_rank > 0)
            throw std::invalid_argument("Vulkan MLA requires causal attention");
        ranges.emplace_back(span.start, span.start + static_cast<int>(t.desc.shape[0]));
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i - 1].second > ranges[i].first)
            throw std::invalid_argument("Vulkan input embedding spans overlap");

    if (cancel()) return {};
    int min_bidir = 0;
    for (const auto& span : options.inputs_embeds)
        if (span.bidirectional) min_bidir = std::max(min_bidir, static_cast<int>(span.host_rows.desc.shape[0]));
    int capacity = std::min(std::max(ctx.prefill_batch(), min_bidir), static_cast<int>(tokens.size()));
    const auto& ple = weights_.gemma_inputs().token_embeddings;
    if (ple.vk_buffer)
        capacity = std::min(capacity, static_cast<int>(65535LL * 64 / ple.shape[1]));
    capacity = std::max(1, capacity);
    while (capacity > 1 && !ctx.try_reserve_scratch_memory(scratch_bytes(ctx, capacity, !ranges.empty()))) capacity /= 2;
    for (const auto& span : options.inputs_embeds)
        if (span.bidirectional && span.host_rows.desc.shape[0] > capacity)
            throw std::invalid_argument("Vulkan bidirectional span exceeds the available prefill batch");

    std::vector<float> logits;
    while (prefilled < static_cast<int>(tokens.size()) && !cancel()) {
        int rows = std::min(capacity, static_cast<int>(tokens.size()) - prefilled);
        for (const auto& span : options.inputs_embeds) {
            if (!span.bidirectional || span.start < prefilled || span.start >= prefilled + rows) continue;
            const int end = span.start + static_cast<int>(span.host_rows.desc.shape[0]);
            if (end > prefilled + rows && span.start != prefilled) rows = span.start - prefilled;
        }
        logits = forward_batch_vulkan(ctx, tokens.subspan(prefilled, rows), start_pos + prefilled,
                                     compute_logits && prefilled + rows == tokens.size(), options.inputs_embeds, prefilled);
        prefilled += rows;
    }
    return logits;
}

std::vector<int32_t> VulkanExecutor::generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                     int max_new_tokens, GenerationStats* stats,
                                                     bool keep_state, int start_pos, int32_t stop_eos_id) {
    return generate_vulkan(ctx, prompt_ids, max_new_tokens, stats, keep_state, start_pos, stop_eos_id, {});
}

std::vector<int32_t> VulkanExecutor::generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                     int max_new_tokens, GenerationStats* stats,
                                                     bool keep_state, int start_pos, int32_t stop_eos_id,
                                                     const GenerationOptions& options) {
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
    options.sampling.validate();
    const auto* json = options.json_constraint;
    if (json && json->vocab_size() != static_cast<size_t>(cfg.vocab_size))
        throw std::invalid_argument("JSON grammar vocabulary does not match Vulkan model");
    if (json && max_new_tokens == 0) throw std::invalid_argument("JSON generation needs room for output tokens");
    auto grammar = json ? json->initial_state() : JsonGrammarState{};
    std::mt19937_64 rng(options.sampling.seed);
    const auto cancel = [&] { return options.cancel && options.cancel->load(std::memory_order_relaxed); };
    if (!keep_state && !cancel()) sessions_.reset_kv_and_state_vulkan(ctx);

    using Clock = std::chrono::steady_clock;
    const auto tp0 = Clock::now();
    int prefilled = 0;
    auto logits = prefill_vulkan(ctx, tokens, start_pos, options, prefilled, max_new_tokens > 0);
    const auto tp1 = Clock::now();
    const double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    int pos = start_pos + prompt_len;
    int gen_count = 0;
    int decode_steps = 0;
    bool cancelled = prefilled < prompt_len || cancel();
    const auto td0 = Clock::now();
    for (int step = 0; step < max_new_tokens && !cancelled; ++step) {
        if (json) json->mask(logits, grammar, max_new_tokens - step);
        const int32_t next_tok = options.sampling.temperature <= 0.f
            ? Sampler::argmax(logits.data(), static_cast<int>(logits.size()))
            : Sampler::sample(logits.data(), static_cast<int>(logits.size()), tokens, options.sampling, rng);
        cancelled = cancel();
        if (cancelled) break;
        if (json && !json->accept(grammar, next_tok)) throw std::logic_error("Sampled a token outside the JSON grammar");
        tokens.push_back(next_tok);
        ++gen_count;
        const bool eos = next_tok == stop_eos_id;
        const bool callback_stop = !eos && options.on_token && !options.on_token(next_tok);
        cancelled = cancel();
        const bool stop = eos || callback_stop || cancelled || step + 1 == max_new_tokens ||
                          (json && (grammar.sealed() || json->is_end(next_tok)));
        // Continued turns need the final emitted token in KV and recurrent state.
        if (keep_state || !stop) {
            logits = forward_batch_vulkan(ctx, {&next_tok, 1}, pos++, !stop);
            ++decode_steps;
        }
        cancelled = cancelled || cancel();
        if (stop) break;
    }
    const auto td1 = Clock::now();
    const double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();

    if (stats) {
        *stats = {};
        stats->prompt_tokens = static_cast<int>(prompt_ids.size());
        stats->prefilled_tokens = prefilled;
        stats->generated_tokens = gen_count;
        stats->prefill_ms = prefill_ms;
        stats->decode_ms = decode_ms;
        stats->prefill_tok_per_s = prefill_ms > 0.0 ? prefilled * 1000.0 / prefill_ms : 0.0;
        stats->decode_tok_per_s = decode_ms > 0.0 ? decode_steps * 1000.0 / decode_ms : 0.0;
        stats->cancelled = cancelled;
        stats->json_complete = json && grammar.complete();
    }
    return tokens;
}

}  // namespace fusionrt
