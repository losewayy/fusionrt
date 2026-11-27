#include "vulkan_executor.h"
#include "vulkan_session.h"
#include "vulkan_spec_moe.h"

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

VkDeviceSize VulkanExecutor::scratch_bytes(const VkContext& ctx, int rows, bool embeds, bool verify, bool mtp) const {
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
        if (!layer.cpu_down.bytes().empty()) bytes += tile(h);
        if (layer.cache) {
            for (const auto* t : {&layer.cpu_gate, &layer.cpu_up, &layer.cpu_down})
                bytes += padded((vulkan_detail::expert_bytes(*t) + sizeof(float) - 1) / sizeof(float));
            bytes += padded(moe.num_experts);
        }
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
            if ((embeds || verify) && q * sizeof(float) % align != 0) bytes += 2 * tile(q);
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
    const int head_rows = verify ? rows : 1;
    const auto& head_weight = mtp ? weights_.mtp().head : weights_.lm_head();
    const auto chunk = VulkanDispatch::head_chunk_width(ctx, head_weight, cfg.vocab_size);
    auto head = padded(int64_t{head_rows} * cfg.vocab_size) + padded(head_rows * h);
    if (rows > 1 && !verify) head += padded(h);
    if (chunk < cfg.vocab_size) head += padded(int64_t{head_rows} * chunk);
    const auto& ple = weights_.gemma_inputs().token_embeddings;
    const auto ple_cols = ple.vk_buffer ? ple.shape[1] : 0;
    const auto persistent = (embeds ? 3 : 2) * tile(h) + padded(rows) + tile(ple_cols) + tile(weights_.hadamard_width());
    return persistent + std::max({head, layer_scratch, tile(ple_cols), mtp ? 6 * tile(h) : VkDeviceSize{0}});
}

std::vector<float> VulkanExecutor::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    const MoeDecodeScope scope(&ctx);
    return forward_batch_vulkan(ctx, {&token_id, 1}, pos);
}

std::vector<float> VulkanExecutor::forward_verify_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos) {
    if (tokens.empty() || tokens.size() > MtpDepthController::max_depth + 1)
        throw std::invalid_argument("Vulkan verification requires 1 to 8 tokens");
    return forward_impl(ctx, tokens, start_pos, true, {}, 0, true, false, false);
}

std::vector<float> VulkanExecutor::forward_mtp_step_vulkan(VkContext& ctx, int32_t token, int pos, bool chained, bool predict) {
    if (!weights_.mtp().projection.vk_buffer) throw std::logic_error("Vulkan NextN weights are missing");
    return forward_impl(ctx, {&token, 1}, pos, predict, {}, 0, false, true, chained);
}

size_t VulkanExecutor::trunk_gdn_count() const {
    const auto& kinds = weights_.layer_kinds();
    const bool gdn_mtp = weights_.mtp().projection.vk_buffer && kinds.back() == 1;
    return sessions_.gdn_count() - gdn_mtp;
}

std::vector<float> VulkanExecutor::forward_batch_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int start_pos,
                                                       bool compute_logits, std::span<const InputEmbedding> embeds,
                                                       int prompt_offset) {
    return forward_impl(ctx, tokens, start_pos, compute_logits, embeds, prompt_offset, false, false, false);
}

std::vector<float> VulkanExecutor::forward_impl(VkContext& ctx, std::span<const int32_t> tokens, int start_pos,
                                               bool compute_logits, std::span<const InputEmbedding> embeds,
                                               int prompt_offset, bool verify, bool mtp, bool chained) {
    const auto& cfg = weights_.config();
    if (start_pos < 0 || start_pos >= cfg.dense.max_seq_len || tokens.empty() ||
        tokens.size() > static_cast<size_t>(cfg.dense.max_seq_len - start_pos))
        throw std::invalid_argument("Token batch exceeds context or is empty");
    validate_token_ids(tokens, start_pos, cfg.vocab_size);
    const int rows = static_cast<int>(tokens.size());
    if (!ctx.try_reserve_scratch_memory(scratch_bytes(ctx, rows, !embeds.empty(), verify, mtp)))
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
    const auto& nextn = weights_.mtp();
    const auto& embedding = mtp ? nextn.embedding : weights_.embeddings();
    const auto host_embeddings = mtp ? weights_.gather_mtp_embeddings(tokens) : weights_.gather_host_embeddings(tokens);
    Tensor hidden_a = host_embeddings.bytes().empty() || mtp ? scratch(hidden) : ctx.upload(host_embeddings, false);
    Tensor hidden_b = scratch(hidden);
    const auto embedding_mark = arena.mark();
    auto embedded = mtp ? (host_embeddings.bytes().empty() ? scratch(hidden) : ctx.upload(host_embeddings, false)) : hidden_a;
    if (host_embeddings.bytes().empty())
        VulkanDispatch::append_embedding(batch, embedding, token, embedded, hidden, rows);
    VulkanDispatch::append_hadamard(batch, embedding, embedded, embedded, rows, hidden, true);
    if (mtp) {
        auto normalized = scratch(hidden);
        const auto seed = scratch(hidden);
        auto concat = scratch(2 * hidden);
        const auto bytes = seed.nbytes();
        VulkanDispatch::append_norm(batch, embedded, nextn.embedding_norm, normalized, 1, hidden, cfg.dense.rms_norm_eps);
        VulkanDispatch::append_copy(batch, normalized, concat, {{0, 0, bytes}});
        if (start_pos > 0) {
            const auto& source = chained ? nextn.draft_hidden : nextn.trunk_hidden;
            VulkanDispatch::append_copy(batch, source, seed, {{chained ? 0 : (start_pos - 1) * bytes, 0, bytes}});
            VulkanDispatch::append_norm(batch, seed, nextn.hidden_norm, normalized, 1, hidden, cfg.dense.rms_norm_eps);
        } else {
            ctx.zero_buffers(std::span{&seed, 1});
            normalized = seed;
        }
        VulkanDispatch::append_copy(batch, normalized, concat, {{0, bytes, bytes}});
        VulkanDispatch::append_linear(batch, concat, nextn.projection, hidden_a, 1, hidden, 2 * hidden);
        if (nextn.projection_scale != 1.f)
            VulkanDispatch::append_scale_add(batch, hidden_a, {}, hidden_a, 1, hidden, nextn.projection_scale, 0.f);
        arena.reset(embedding_mark);
    } else if (cfg.embed_scale != 1.0f) {
        VulkanDispatch::append_scale_add(batch, hidden_a, {}, hidden_a, rows, hidden, cfg.embed_scale, 0.f);
    }

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
    if (!mtp && g.token_embeddings.vk_buffer) {
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
        if (layer.cpu_down.bytes().empty()) {
            VulkanDispatch::append_moe(batch, layer, moe, input, output, hidden, rows);
            return;
        }
        VulkanDispatch::append_moe_route(batch, layer, moe, input, hidden, rows);
        const auto flush = [&] {
            execute_batch(ctx, batch);
            batch.ops.clear();
            batch.push.clear();
        };
        flush();
        const auto ids = ctx.download(moe.indices);
        const auto* indices = ids.data<int32_t>();
        const std::span routes{indices, static_cast<size_t>(rows) * layer.cfg.top_k};
        const auto is_cold = [&](int id) {
            return id >= 0 && (layer.expert_to_slot.empty() || layer.expert_to_slot[id] < 0);
        };
        VulkanDispatch::append_moe_experts(batch, layer, moe, input, output, hidden, rows);
        if (std::any_of(routes.begin(), routes.end(), is_cold)) {
            const auto weights = ctx.download(moe.weights);
            const auto normalized = ctx.download(moe.normalized);
            HostTensor cold({rows, hidden}, DType::F32);
            if (!moe_workers_) moe_workers_ = std::make_unique<CpuWorkers>();
            const auto start = std::chrono::steady_clock::now();
            const int count = cpu_moe_accumulate_layer(moe_workers_.get(), moe_workspace_, layer.cpu_gate, layer.cpu_up,
                layer.cpu_down, indices, weights.data<float>(), 0, 0, rows, layer.cfg.top_k,
                normalized.data<float>(), cold.data<float>(), CpuMoeMode::Parallel, layer.router_scale.vk_buffer != 0,
                layer.cpu_gate_scale.data<float>(), layer.cpu_up_scale.data<float>(),
                layer.expert_to_slot.empty() ? nullptr : layer.expert_to_slot.data());
            if (layer.cache && layer.cache_hardware.cpu_q4_gemv_gbps <= 0) {
                const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
                if (ns > 0) layer.cache_hardware.cpu_q4_gemv_gbps = vulkan_detail::cache_copy_bytes(layer) * count / ns;
            }
            const auto update = ctx.upload(cold, false);
            VulkanDispatch::append_scale_add(batch, output, update, output, rows, hidden, 1.f, 1.f);
        }
        if (layer.cache && MoeDecodeScope::active_for(&ctx)) {
            flush();
            vulkan_detail::update_expert_cache(ctx, layer, routes);
        }
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
    const auto first_layer = mtp ? static_cast<size_t>(cfg.num_layers) : 0;
    const auto end_layer = mtp ? first_layer + 1 : static_cast<size_t>(cfg.num_layers);
    for (size_t i = first_layer; i < end_layer; ++i) {
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
                if ((verify || !image_spans.empty()) && q_dim * sizeof(float) % ctx.info().min_storage_offset_align != 0) {
                    dense.image_query = scratch(q_dim, verify ? 1 : rows);
                    dense.image_attention = scratch(q_dim, verify ? 1 : rows);
                }
                VulkanDispatch::append_dense(batch, layer, kv, dense, *current, *next, hidden, start_pos, rows, image_spans, verify);
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

    if ((mtp && compute_logits) || (!mtp && nextn.trunk_hidden.vk_buffer)) {
        // NextN is seeded from the head-normalized activation, as in the CUDA graph.
        const auto& norm = mtp ? nextn.head_norm : weights_.final_norm();
        VulkanDispatch::append_norm(batch, *current, norm, *next, rows, hidden, cfg.dense.rms_norm_eps);
        const auto offset = mtp ? 0 : static_cast<VkDeviceSize>(start_pos) * hidden * sizeof(float);
        VulkanDispatch::append_copy(batch, *next, mtp ? nextn.draft_hidden : nextn.trunk_hidden,
                                    {{0, offset, next->nbytes()}});
    }
    Tensor logits;
    const int head_rows = verify ? rows : 1;
    if (compute_logits) {
        Tensor last = *current;
        if (rows > 1 && !verify) {
            last = scratch(hidden, 1);
            const auto row_bytes = static_cast<VkDeviceSize>(hidden) * sizeof(float);
            VulkanDispatch::append_copy(batch, *current, last, {{(rows - 1) * row_bytes, 0, row_bytes}});
        }
        auto final_normalized = scratch(hidden, head_rows);
        logits = scratch(cfg.vocab_size, head_rows);
        const auto& head = mtp ? nextn.head : weights_.lm_head();
        const auto& norm = mtp ? nextn.head_norm : weights_.final_norm();
        const int width = VulkanDispatch::head_chunk_width(ctx, head, cfg.vocab_size);
        auto chunk = width == cfg.vocab_size ? Tensor{} : scratch(width, head_rows);
        VulkanDispatch::append_head(batch, norm, head, final_normalized, logits, last, hidden, cfg.vocab_size,
                                    cfg.dense.rms_norm_eps, mtp ? 0.f : cfg.final_logit_softcapping, head_rows, chunk);
        if (mtp && nextn.head_scale != 1.f)
            VulkanDispatch::append_scale_add(batch, logits, {}, logits, 1, cfg.vocab_size, nextn.head_scale, 0.f);
    }
    execute_batch(ctx, batch);
    if (!compute_logits) return {};
    const HostTensor host_logits = ctx.download(logits);
    return {host_logits.data<float>(), host_logits.data<float>() + static_cast<size_t>(head_rows) * cfg.vocab_size};
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
                                                     const GenerationOptions& options, VulkanExecutor* draft,
                                                     VkContext* draft_ctx) {
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
    const bool native_mtp = options.spec_mtp;
    const bool stochastic = options.sampling.temperature > 0.f;
    const bool auto_depth = native_mtp && (options.auto_draft_tokens || options.draft_tokens == 0);
    const int modes = (options.spec_ngram > 0) + (options.draft_model != nullptr) + native_mtp;
    if (options.spec_ngram < 0 || modes > 1 || options.draft_tokens < 0 || (!native_mtp && options.draft_tokens == 0))
        throw std::invalid_argument("Choose one speculative decoding mode. Auto draft depth requires native MTP");
    if (native_mtp && (!weights_.mtp().projection.vk_buffer || keep_state))
        throw std::invalid_argument("Vulkan native MTP requires a loaded NextN block and a fresh request");
    if (native_mtp && (!std::isfinite(options.spec_p_min) || options.spec_p_min < 0.f || options.spec_p_min > 1.f))
        throw std::invalid_argument("MTP draft probability cutoff must be in [0, 1]");
    if (options.draft_model && (!draft || !draft_ctx || draft == this || keep_state ||
        draft->weights_.config().vocab_size != cfg.vocab_size))
        throw std::invalid_argument("Vulkan draft model needs a compatible vocabulary and a fresh request");
    if (modes && !options.inputs_embeds.empty())
        throw std::invalid_argument("Vulkan speculative decoding requires a text prompt");
    const int draft_cap = options.spec_ngram > 0 ? options.spec_ngram :
        options.draft_tokens > 0 ? options.draft_tokens : MtpDepthController::max_depth;
    int batch_limit = modes ? std::min(max_new_tokens, 1 + std::min(draft_cap, MtpDepthController::max_depth)) : 1;
    while (batch_limit > 1 && !ctx.try_reserve_scratch_memory(scratch_bytes(ctx, batch_limit, false, true)))
        --batch_limit;
    const bool speculative = batch_limit > 1;
    GenerationStats spec_stats;
    spec_stats.speculative_active = speculative;
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
    const auto remaining = [&] { return max_new_tokens - gen_count; };
    const auto emit = [&](int32_t id) {
        cancelled = cancel();
        if (cancelled) return false;
        if (json && !json->accept(grammar, id)) throw std::logic_error("Sampled a token outside the JSON grammar");
        tokens.push_back(id);
        ++gen_count;
        const bool eos = id == stop_eos_id;
        const bool callback_stop = !eos && options.on_token && !options.on_token(id);
        cancelled = cancel();
        return !eos && !callback_stop && !cancelled && remaining() &&
            (!json || (!grammar.sealed() && !json->is_end(id)));
    };
    const auto choose = [&](std::span<float> row, const Sampler::Distribution* q, int32_t proposal) {
        if (json) json->mask(row, grammar, remaining());
        if (!stochastic) return Sampler::argmax(row.data(), cfg.vocab_size);
        if (!q) return Sampler::sample(row.data(), cfg.vocab_size, tokens, options.sampling, rng);
        const auto p = Sampler::distribution(row.data(), cfg.vocab_size, tokens, options.sampling);
        return vulkan_detail::verify_sample(p, *q, proposal, rng);
    };
    const auto decode = [&](int32_t id, bool predict) {
        const MoeDecodeScope scope(&ctx);
        ++decode_steps;
        return forward_batch_vulkan(ctx, {&id, 1}, pos++, predict);
    };

    bool running = max_new_tokens && !cancelled && emit(choose(logits, nullptr, -1));
    int draft_pos = 0, mtp_pos = 0;
    const int draft_context = draft ? draft->weights_.config().dense.max_seq_len : 0;
    if (speculative && running && draft && prompt_len < draft_context) {
        draft->sessions_.reset_kv_and_state_vulkan(*draft_ctx);
        int prefilled_draft = 0;
        GenerationOptions draft_options;
        draft_options.cancel = options.cancel;
        draft->prefill_vulkan(*draft_ctx, prompt_ids.empty() ? std::span(tokens).first(1) : std::span(prompt_ids),
                              0, draft_options, prefilled_draft, false);
        draft_pos = prefilled_draft;
    }
    MtpDepthController controller;
    std::mt19937_64 draft_rng(options.sampling.seed ^ UINT64_C(0x9e3779b97f4a7c15));
    const size_t trunk_gdn = trunk_gdn_count();
    const auto elapsed_ms = [](Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    while (running && !cancel()) {
        if (!speculative) {
            logits = decode(tokens.back(), true);
            running = emit(choose(logits, nullptr, -1));
            continue;
        }
        const int start = pos;
        int limit = std::min(remaining() - 1, batch_limit - 1);
        if (auto_depth) limit = controller.choose_depth(limit);
        std::vector<int32_t> input{tokens.back()};
        std::vector<Sampler::Distribution> draft_probs;
        if (limit > 0 && draft && start < draft_context) {
            limit = std::min(limit, draft_context - start);
            while (draft_pos < start && !cancel()) {
                draft->forward_batch_vulkan(*draft_ctx, {&tokens[draft_pos], 1}, draft_pos, false);
                ++draft_pos;
            }
            const auto gdn = draft->trunk_gdn_count();
            draft->sessions_.snapshot_gdn_state(*draft_ctx, 0, gdn);
            auto history = tokens;
            for (int i = 0; i < limit && !cancel(); ++i) {
                auto row = draft->forward_batch_vulkan(*draft_ctx, {&input.back(), 1}, draft_pos++);
                if (stochastic) {
                    auto q = Sampler::distribution(row.data(), cfg.vocab_size, history, options.sampling);
                    input.push_back(Sampler::draw(q, draft_rng));
                    history.push_back(input.back());
                    std::sort(q.begin(), q.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
                    draft_probs.push_back(std::move(q));
                } else {
                    input.push_back(Sampler::argmax(row.data(), cfg.vocab_size));
                }
            }
        } else if (limit > 0 && native_mtp) {
            while (mtp_pos < start && !cancel()) {
                forward_mtp_step_vulkan(ctx, tokens[mtp_pos], mtp_pos, false, false);
                ++mtp_pos;
            }
            const auto end_gdn = sessions_.gdn_count();
            int tentative = 0;
            for (int i = 0; i < limit && !cancel(); ++i) {
                const auto begin = Clock::now();
                if (i == 1) sessions_.snapshot_gdn_state(ctx, trunk_gdn, end_gdn);
                const auto row = forward_mtp_step_vulkan(ctx, input.back(), start + i, i != 0, true);
                if (i == 0) {
                    mtp_pos = start + 1;
                    ++spec_stats.mtp_reused_kv_rows;
                } else {
                    ++tentative;
                }
                if (auto_depth) controller.observe_draft(elapsed_ms(begin));
                const auto proposal = Sampler::argmax(row.data(), cfg.vocab_size);
                if (options.spec_p_min > 0.f && vulkan_detail::draft_confidence(row, proposal) < options.spec_p_min) {
                    ++spec_stats.mtp_cutoffs;
                    break;
                }
                input.push_back(proposal);
                ++spec_stats.mtp_proposed_by_depth[i];
            }
            if (tentative) sessions_.restore_gdn_state(ctx, trunk_gdn, end_gdn);
            spec_stats.mtp_rewound_kv_rows += tentative;
        } else if (limit > 0 && options.spec_ngram) {
            vulkan_detail::lookup_draft(tokens, limit, input);
        }
        if (cancel()) break;

        const int proposed = static_cast<int>(input.size()) - 1;
        const auto begin = Clock::now();
        if (!proposed) {
            logits = decode(input.front(), true);
            if (auto_depth) controller.observe_verify(1, elapsed_ms(begin));
            running = emit(choose(logits, nullptr, -1));
            continue;
        }

        spec_stats.spec_proposed += proposed;
        sessions_.snapshot_gdn_state(ctx, 0, trunk_gdn);
        logits = forward_verify_vulkan(ctx, input, start);
        if (auto_depth) controller.observe_verify(proposed + 1, elapsed_ms(begin));
        pos += proposed + 1;
        for (int i = 0; i <= proposed; ++i) {
            const bool bonus = i == proposed;
            const auto proposal = bonus ? -1 : input[i + 1];
            const Sampler::Distribution point_mass{{1.0, proposal}};
            const auto* q = !bonus && stochastic ? (draft ? &draft_probs[i] : &point_mass) : nullptr;
            auto row = std::span(logits).subspan(static_cast<size_t>(i) * cfg.vocab_size, cfg.vocab_size);
            const auto id = choose(row, q, proposal);
            const bool accepted = !bonus && id == proposal;
            const auto before = gen_count;
            running = emit(id);
            if (gen_count == before) break;
            spec_stats.spec_accepted += accepted;
            if (native_mtp && !bonus) {
                ++spec_stats.mtp_checked_by_depth[i];
                spec_stats.mtp_accepted_by_depth[i] += accepted;
                if (auto_depth) controller.observe_acceptance(i + 1, accepted);
            }
            if (!accepted || !running) break;
        }
        // The last emitted token is pending. Retain only the verified prefix before it.
        const int kept = std::max(start, start_pos + static_cast<int>(tokens.size()) - 1);
        if (kept < pos) {
            const auto rewind = Clock::now();
            if (trunk_gdn) {
                sessions_.restore_gdn_state(ctx, 0, trunk_gdn);
                if (kept > start)
                    forward_impl(ctx, std::span(input).first(kept - start), start, false, {}, 0, true, false, false);
            }
            if (auto_depth) controller.observe_rewind(pos - kept, elapsed_ms(rewind));
            pos = kept;
        }
        decode_steps += kept - start;
        if (draft && draft_pos > kept) {
            const auto gdn = draft->trunk_gdn_count();
            draft->sessions_.restore_gdn_state(*draft_ctx, 0, gdn);
            if (gdn)
                for (int i = 0; i < kept - start; ++i)
                    draft->forward_batch_vulkan(*draft_ctx, {&input[i], 1}, start + i, false);
            draft_pos = kept;
        }
    }
    if (keep_state && gen_count && pos < start_pos + static_cast<int>(tokens.size())) decode(tokens.back(), false);
    cancelled = cancelled || cancel();
    const auto td1 = Clock::now();
    const double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();

    if (stats) {
        *stats = spec_stats;
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
