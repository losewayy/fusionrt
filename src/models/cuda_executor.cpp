#include "cuda_executor.h"
#include "cuda_ir.h"
#include "cuda_profile.h"
#include "../core/moe_route_trace.h"

#ifdef FUSIONRT_ENABLE_CUDA

#include <algorithm>
#include <iostream>
#include <numeric>

namespace fusionrt {
namespace {

constexpr int kPrefillCarrySlots = 8;
constexpr int kMaxDecodeSplits = 128;

}  // namespace

static void decode_argmax(CudaContext& ctx, const Tensor& logits, Tensor& token, int vocab_size,
                          int* graph_pos = nullptr, int* graph_kv_len = nullptr) {
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    void* scratch = vocab_size >= 32768 ? arena.alloc(cuda_ops::kArgmaxScratchBytes) : nullptr;
    cuda_ops::argmax_f32(logits.ptr<float>(), vocab_size, token.ptr<int32_t>(), ctx.stream(),
                         scratch, graph_pos, graph_kv_len, static_cast<int>(token.shape[0] - 1));
    arena.reset(mark);
}

static bool all_experts_resident(const std::vector<CudaSparseMoELayer>& layers) {
    return std::all_of(layers.begin(), layers.end(), [](const auto& layer) {
        return layer.cfg.gpu_resident_experts == layer.cfg.num_experts;
    });
}

static std::optional<std::pair<int, int>> moe_graph_prefix(const MiniGraph& graph,
                                                          const std::vector<CudaSparseMoELayer>& layers) {
    if (graph.gemma_inputs.token_embeddings) return std::nullopt;
    std::optional<std::pair<int, int>> prefix;
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        const auto& node = graph.nodes[i];
        if (!node.alive || node.op != OpKind::MOE_FFN_BLOCK) continue;
        const auto& cfg = layers[node.layer_index].cfg;
        if (cfg.gpu_resident_experts != cfg.num_experts) return prefix;
        prefix = {i + 1, node.out};
    }
    return std::nullopt;
}

static void embed_tokens(CudaContext& ctx, const Tensor& table, const Tensor& ids, Tensor& hidden,
                         int rows, int width, float scale, int* invalid_token = nullptr) {
    cuda_ops::embed_lookup_f32(table.ptr(), table.dtype, ids.ptr<int32_t>(), hidden.ptr<float>(),
                               rows, width, static_cast<int>(table.shape[0]), ctx.stream(), invalid_token);
    if (table.hadamard_block) {
        cuda_ops::blockwise_hadamard_f32(hidden.ptr<float>(), table.hadamard_signs, hidden.ptr<float>(),
                                         rows, width, table.hadamard_block, ctx.stream(), true);
    }
    if (scale != 1.0f) cuda_ops::scale_f32(hidden.ptr<float>(), rows * width, scale, ctx.stream());
}

static bool run_forward_token_step_cuda(
    CudaContext& ctx, const HybridModelConfig& cfg,
    const Tensor& tok_embeddings, const MiniGraph& ir, CudaIRWorkspace& ws,
    Tensor& hidden_buf, Tensor& logits_buf, Tensor& tok_id_dev, Tensor& pos_dev, Tensor& kv_len_dev, int* invalid_token,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvView>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    CudaGraphExec& decode_graph, bool& graph_captured, bool& graph_fused_flag, bool& graph_has_argmax,
    int& graph_split_bucket,
    int pos, bool use_fused, bool use_graph, bool with_argmax, bool first_step = true,
    OpTrace* trace = nullptr, CudaOpTimer* timer = nullptr) {
    int H = static_cast<int>(hidden_buf.shape[0]);

    auto run_body = [&](bool graph_mode) {
        embed_tokens(ctx, tok_embeddings, tok_id_dev, hidden_buf, 1, H, cfg.embed_scale, invalid_token);
        int* p_dev = graph_mode ? pos_dev.ptr<int>() : nullptr;
        int* kl_dev = graph_mode ? kv_len_dev.ptr<int>() : nullptr;
        if (timer) {
            execute_ir_graph_cuda<true>(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos, p_dev, kl_dev,
                                        dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused, trace, timer);
        } else {
            execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos, p_dev, kl_dev,
                                  dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused);
        }
        if (with_argmax) {
            decode_argmax(ctx, logits_buf, tok_id_dev, cfg.vocab_size, p_dev, kl_dev);
        }
    };

    const auto capture = [&](const auto& body) {
        try {
            decode_graph.capture(ctx.stream(), body);
            return true;
        } catch (const std::runtime_error& error) {
            graph_captured = false;
            std::clog << "CUDA Graph capture failed, using eager: " << error.what() << '\n';
            run_body(false);
            return false;
        }
    };

    if (use_graph && all_experts_resident(moe_layers)) {
        // The generation graph advances both counters after it writes the next token.
        if (first_step || !with_argmax || !graph_captured) {
            int kv_len = pos + 1;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr(), &pos, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(kv_len_dev.ptr(), &kv_len, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
        }
        const int bucket = select_decode_splits(ctx.cost_hardware(), pos + 1,
                                                cfg.dense.num_key_value_heads, kMaxDecodeSplits);
        if (!graph_captured || graph_fused_flag != use_fused || graph_has_argmax != with_argmax ||
            graph_split_bucket != bucket) {
            if (!capture([&] { run_body(true); })) return false;
            graph_captured = true;
            graph_fused_flag = use_fused;
            graph_has_argmax = with_argmax;
            graph_split_bucket = bucket;
        }
        decode_graph.launch(ctx.stream());
        return true;
    }
    if (use_graph && !timer) {
        if (const auto prefix = moe_graph_prefix(ir, moe_layers)) {
            const auto [cut, boundary] = *prefix;
            const int kv_len = pos + 1;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr(), &pos, sizeof(pos), cudaMemcpyHostToDevice, ctx.stream()));
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(kv_len_dev.ptr(), &kv_len, sizeof(kv_len), cudaMemcpyHostToDevice, ctx.stream()));
            const int bucket = select_decode_splits(ctx.cost_hardware(), kv_len, cfg.dense.num_key_value_heads, kMaxDecodeSplits);
            if (!graph_captured || graph_fused_flag != use_fused || graph_has_argmax != with_argmax ||
                graph_split_bucket != bucket) {
                const CudaIRSegment captured{0, cut, ir.input_value, boundary, &hidden_buf};
                if (!capture([&] {
                    embed_tokens(ctx, tok_embeddings, tok_id_dev, hidden_buf, 1, H, cfg.embed_scale, invalid_token);
                    execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos,
                        pos_dev.ptr<int>(), kv_len_dev.ptr<int>(), dense_layers, gdn_layers, kv_states, gdn_states,
                        moe_layers, use_fused, nullptr, nullptr, nullptr, false, false, &captured);
                })) return false;
                graph_captured = true;
                graph_fused_flag = use_fused;
                graph_has_argmax = with_argmax;
                graph_split_bucket = bucket;
            }
            if (ctx.moe_timing) {
                for (int i = 0; i < cut; ++i) {
                    const auto& node = ir.nodes[i];
                    if (node.alive && node.op == OpKind::MOE_FFN_BLOCK)
                        ctx.moe_timing->layer(node.layer_index).hot += moe_layers[node.layer_index].cfg.top_k;
                }
            }
            decode_graph.launch(ctx.stream());
            const CudaIRSegment eager{cut, static_cast<int>(ir.nodes.size()), boundary};
            execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos, nullptr, nullptr,
                dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused,
                nullptr, nullptr, nullptr, true, false, &eager);
            if (with_argmax) decode_argmax(ctx, logits_buf, tok_id_dev, cfg.vocab_size);
            return true;
        }
    }
    run_body(false);
    return false;
}

int CudaExecutor::prefill_batch_limit(CudaContext& ctx, bool verify) const {
    const auto& d = weights_.config().dense;
    const auto& g = weights_.config().gdn;
    const int q = d.num_attention_heads * d.head_dim;
    const int kv = d.num_key_value_heads * d.head_dim;
    int64_t dense_extra = std::max<int64_t>(3LL * d.intermediate_size, 4LL * q + 2LL * kv);
    if (d.mla.latent_rank > 0) {
        const int64_t heads = d.num_attention_heads;
        dense_extra = std::max<int64_t>(dense_extra, q + d.mla.latent_width() + 2LL * d.mla.query_rank +
            heads * (d.mla.latent_width() + d.mla.latent_rank + d.mla.value_dim));
    }
    if (weights_.config().model_architecture == "gemma4") {
        for (const auto& layer : weights_.dense_layers()) {
            const auto& a = layer.cfg;
            dense_extra = std::max({dense_extra, int64_t{3} * a.intermediate_size,
                (int64_t{4} * a.num_attention_heads + int64_t{2} * a.num_key_value_heads) * a.head_dim});
        }
    }
    const auto& arena = ctx.scratch_arena();
    const size_t available = arena.capacity() - arena.used();
    int rows = std::min(ctx.prefill_batch(), d.max_seq_len);
    for (; rows > 1; rows /= 2) {
        const size_t layer_rows = static_cast<size_t>(rows);
        const size_t dense = dense_extra * layer_rows;
        size_t recurrent = 0;
        if (!weights_.gdn_layers().empty()) {
            const int recurrent_cols = 2 * g.num_k_heads * g.head_k_dim + 3 * g.num_v_heads * g.head_v_dim +
                                       2 * g.num_v_heads;
            recurrent = std::max(3 * g.intermediate_size * layer_rows, recurrent_cols * layer_rows +
                cuda_ops::gated_deltanet_prefill_scratch_floats(rows, g.num_k_heads, g.num_v_heads,
                                                              g.head_k_dim, g.head_v_dim));
        }
        const size_t moe = weights_.moe_layers().empty() ? 0 :
            (4 * static_cast<size_t>(d.hidden_size) + weights_.config().moe.num_experts + 2 * weights_.config().moe.top_k +
             3 * weights_.config().moe.shared_intermediate_size) * layer_rows +
            static_cast<size_t>(weights_.config().moe.top_k) * weights_.config().moe.intermediate_size;
        const auto layer = std::max({dense, recurrent, moe});
        const size_t ple = weights_.has_per_layer_embeddings() ?
            static_cast<size_t>(weights_.per_layer_embedding_width()) * layer_rows : 0;
        const size_t live = std::max((kPrefillCarrySlots + 2) * d.hidden_size * layer_rows + layer + ple,
                                    d.hidden_size * layer_rows + 3 * ple) * sizeof(float);
        const size_t verify_logits = verify ? layer_rows * weights_.config().vocab_size * sizeof(float) : 0;
        if (live + verify_logits + 64 * 1024 <= available) break;
    }
    return rows;
}

void CudaExecutor::validate_input_embeddings(std::span<const InputEmbedding> spans, size_t tokens, int width) {
    size_t end = 0;
    for (const auto& span : spans) {
        const auto& t = span.rows;
        if (span.start < 0 || static_cast<size_t>(span.start) < end || static_cast<size_t>(span.start) >= tokens ||
            t.ndim != 2 || t.shape[1] != width || t.shape[0] <= 0 ||
            static_cast<uint64_t>(t.shape[0]) > tokens - span.start ||
            t.dtype != DType::F32 || t.backend != Backend::CUDA || !t.data || !t.is_contiguous()) {
            throw std::invalid_argument("Invalid CUDA input embedding range or shape");
        }
        end = static_cast<size_t>(span.start) + static_cast<size_t>(t.shape[0]);
    }
}

void CudaExecutor::execute_prefill_chunks(CudaContext& ctx, const int32_t* token_ids, int count,
                                         int start_pos, bool use_fused, LayerTrace* layer_trace,
                                         OpTrace* op_trace, bool compute_logits, bool save_prefix_checkpoints,
                                         std::span<const InputEmbedding> inputs_embeds,
                                         const std::atomic_bool* cancel) {
    if (cancel && cancel->load(std::memory_order_relaxed)) throw CudaPrefillCancelled{};
    const int H = static_cast<int>(weights_.embeddings().shape[1]);
    auto& arena = ctx.scratch_arena();
    struct RestoreArena {
        CudaArena& arena;
        size_t mark;
        ~RestoreArena() { arena.reset(mark); }
    } restore{arena, arena.mark()};

    ensure_kv(ctx, start_pos, start_pos + count);
    const int capacity = std::min(count, prefill_batch_limit(ctx));
    const bool gemma_images = !inputs_embeds.empty() && weights_.config().model_architecture == "gemma4";
    if (gemma_images) for (const auto& span : inputs_embeds) {
        if (span.bidirectional && span.rows.shape[0] > capacity)
            throw std::invalid_argument("Gemma 4 image span exceeds the available prefill batch");
    }
    Tensor bounds;
    std::vector<int> image_bounds;
    if (gemma_images && std::any_of(inputs_embeds.begin(), inputs_embeds.end(), [](const auto& span) {
        return span.bidirectional;
    })) {
        bounds = cuda_scratch(ctx, {capacity, 2}, DType::I32);
        image_bounds.resize(static_cast<size_t>(capacity) * 2);
    }
    Tensor ids = cuda_scratch(ctx, {capacity}, DType::I32);
    Tensor hidden = cuda_scratch(ctx, {capacity, H});
    const auto& ir = use_fused ? weights_.fused_ir() : weights_.raw_ir();
    CudaIRWorkspace ws(ir);
    std::unique_ptr<CudaOpTimer> timer;
    if (op_trace) timer = std::make_unique<CudaOpTimer>(ir.nodes.size());
    auto& pa = prof_acc();

    const int end = start_pos + count;
    // Keep two recent page boundaries so a different question can reuse a long prompt.
    int checkpoint = save_prefix_checkpoints ?
        std::max((start_pos / kKvPageTokens + 1) * kKvPageTokens,
                 ((end - 1) / kKvPageTokens - 1) * kKvPageTokens) : end;
    for (int begin = 0; begin < count;) {
        if (cancel && cancel->load(std::memory_order_relaxed)) throw CudaPrefillCancelled{};
        const int position = start_pos + begin;
        int rows = std::min(capacity, std::min(end, checkpoint) - position);
        if (bounds.data) for (const auto& span : inputs_embeds) {
            if (!span.bidirectional || span.start < begin || span.start >= begin + rows) continue;
            const int span_end = span.start + static_cast<int>(span.rows.shape[0]);
            if (span_end > begin + rows) rows = span.start == begin ? span_end - begin : span.start - begin;
        }
        const int* image_mask = nullptr;
        if (bounds.data) {
            std::fill_n(image_bounds.begin(), 2 * rows, -1);
            for (const auto& span : inputs_embeds) {
                if (!span.bidirectional || span.start < begin || span.start >= begin + rows) continue;
                const int span_end = span.start + static_cast<int>(span.rows.shape[0]);
                for (int row = span.start - begin; row < span_end - begin; ++row) {
                    image_bounds[2 * row] = start_pos + span.start;
                    image_bounds[2 * row + 1] = start_pos + span_end;
                }
                image_mask = bounds.ptr<int>();
            }
            if (image_mask) FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(bounds.ptr(), image_bounds.data(),
                static_cast<size_t>(2 * rows) * sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
        }
        const bool last = begin + rows == count;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids.ptr(), token_ids + begin, rows * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, ctx.stream()));
        embed_tokens(ctx, weights_.embeddings(), ids, hidden, rows, H, weights_.config().embed_scale);
        for (const auto& span : inputs_embeds) {
            const int first = std::max(span.start, begin);
            const int span_end = std::min(span.start + static_cast<int>(span.rows.shape[0]), begin + rows);
            if (span_end <= first) continue;
            const auto src = span.rows.ptr<float>() + static_cast<size_t>(first - span.start) * H;
            const auto dst = hidden.ptr<float>() + static_cast<size_t>(first - begin) * H;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(span_end - first) * H * sizeof(float),
                                                cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        if (pa.enabled) pa.reset();
        auto* nextn = capture_mtp_hidden_ ? mtp_target_hidden_.ptr<float>() + static_cast<size_t>(position) * H : nullptr;
        if (layer_trace || op_trace) {
            execute_ir_graph_cuda<true>(ir, ws, ctx, hidden, ids, work_.logits, rows, position, position,
                nullptr, nullptr, weights_.dense_layers(), weights_.gdn_layers(), sessions_.kv(), sessions_.gdn(),
                weights_.moe_layers(), use_fused, op_trace, timer.get(), last ? layer_trace : nullptr,
                last && compute_logits, false, nullptr, nextn, nullptr, image_mask);
        } else {
            execute_ir_graph_cuda(ir, ws, ctx, hidden, ids, work_.logits, rows, position, position,
                nullptr, nullptr, weights_.dense_layers(), weights_.gdn_layers(), sessions_.kv(), sessions_.gdn(),
                weights_.moe_layers(), use_fused, nullptr, nullptr, nullptr, last && compute_logits, false, nullptr, nextn, nullptr, image_mask);
        }
        if (cancel) ctx.synchronize();
        begin += rows;
        if (position + rows == checkpoint && checkpoint < end) {
            sessions_.save_cuda_checkpoint(ctx, checkpoint, nullptr);
            checkpoint += kKvPageTokens;
        }
        if (pa.enabled) {
            std::cerr << "[PROF prefill M=" << rows << "] x16=" << pa.x16_ms << "ms w16=" << pa.w16_ms
                      << "ms cublas=" << pa.cublas_ms << "ms (calls=" << pa.calls_cublas
                      << ") total_gqa=" << pa.attn_ms << "ms total_gdn=" << pa.gdn_ms
                      << "ms lm=" << pa.lm_ms << "ms\n";
        }
    }
}

std::vector<float> CudaExecutor::prefill_cuda_impl(CudaContext& ctx, const int32_t* token_ids, int count,
                                                 int start_pos, bool use_fused, LayerTrace* layer_trace,
                                                 OpTrace* op_trace, bool save_prefix_checkpoints,
                                                 std::span<const InputEmbedding> inputs_embeds,
                                                 const std::atomic_bool* cancel) {
    validate_token_ids({token_ids, static_cast<size_t>(count)}, start_pos, weights_.config().vocab_size);
    execute_prefill_chunks(ctx, token_ids, count, start_pos, use_fused, layer_trace, op_trace, true,
                           save_prefix_checkpoints, inputs_embeds, cancel);
    HostTensor logits = ctx.download(work_.logits);
    return {logits.data<float>(), logits.data<float>() + weights_.config().vocab_size};
}

// Graphs are request-local. The device position, page-table addresses and scratch
// layout stay valid across replays; admission and host copies stay outside capture.
struct CudaExecutor::Verify {
    struct Entry {
        CudaGraphExec graph;
        uint64_t bucket = 0;
        uint64_t residency_generation = 0;
        bool failed = false;
    };
    std::array<Entry, 9> graphs;
    std::optional<GdnReplay> replay;
    bool enabled = false;
};

static bool launch_spec_graph(cudaStream_t stream, auto& entry, uint64_t bucket,
                              uint64_t residency_generation, bool eligible, const auto& body) {
    if (entry.residency_generation != residency_generation) {
        entry.graph.reset();
        entry.residency_generation = residency_generation;
        entry.failed = false;
    }
    if (!eligible || entry.failed) return false;
    if (!entry.graph.valid() || entry.bucket != bucket) {
        try {
            entry.graph.capture(stream, body);
            entry.bucket = bucket;
        } catch (const std::exception& e) {
            entry.failed = true;
            std::clog << "Speculative CUDA Graph capture failed, using eager: " << e.what() << '\n';
            return false;
        }
    }
    entry.graph.launch(stream);
    return true;
}

bool CudaExecutor::verify_cuda(CudaContext& ctx, const std::vector<int32_t>& input, int pos, bool use_fused,
                               std::vector<float>& logits, std::vector<int32_t>& argmax, bool greedy,
                               bool retain_last_logits, Verify* capture) {
    auto& arena = ctx.scratch_arena();
    struct RestoreArena {
        CudaArena& arena;
        size_t mark;
        ~RestoreArena() { arena.reset(mark); }
    } restore{arena, arena.mark()};
    const int rows = static_cast<int>(input.size());
    const int vocab = weights_.config().vocab_size;
    validate_token_ids(input, pos, vocab);
    const int H = static_cast<int>(weights_.embeddings().shape[1]);
    ensure_kv(ctx, pos, pos + rows);
    Tensor ids = cuda_scratch(ctx, {rows}, DType::I32);
    Tensor hidden = cuda_scratch(ctx, {rows, H});
    Tensor batch_logits = cuda_scratch(ctx, {rows, vocab});
    Tensor positions, nextn;
    std::array<int, 9> base{};
    if (capture) {
        positions = cuda_scratch(ctx, {rows + 1}, DType::I32);
        nextn = cuda_scratch(ctx, {rows, H});
        std::iota(base.begin(), base.begin() + rows + 1, pos);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(positions.ptr(), base.data(), positions.nbytes(),
                                            cudaMemcpyHostToDevice, ctx.stream()));
    }
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids.ptr(), input.data(), rows * sizeof(int32_t),
                                        cudaMemcpyHostToDevice, ctx.stream()));
    const auto& ir = use_fused ? weights_.fused_ir() : weights_.raw_ir();
    CudaIRWorkspace ws(ir);
    // Keep the target's Q8 activation path, even for short verification batches.
    struct RestoreCrossover {
        CudaContext& ctx;
        int rows;
        ~RestoreCrossover() { ctx.set_mmq_crossover_m(rows); }
    } crossover{ctx, ctx.mmq_crossover_m()};
    ctx.set_mmq_crossover_m(2);
    const auto body_mark = arena.mark();
    const auto body = [&] {
        embed_tokens(ctx, weights_.embeddings(), ids, hidden, rows, H, weights_.config().embed_scale);
        auto* position = positions.ptr<int>();
        execute_ir_graph_cuda(ir, ws, ctx, hidden, ids, batch_logits, rows, pos, pos,
            position, position ? position + 1 : nullptr, weights_.dense_layers(), weights_.gdn_layers(), sessions_.kv(),
            sessions_.gdn(), weights_.moe_layers(), use_fused, nullptr, nullptr, nullptr, true, true, nullptr,
            nextn.ptr<float>(), capture && capture->replay ? &*capture->replay : nullptr);
        if (greedy) for (int row = 0; row < rows; ++row) {
            Tensor row_logits = batch_logits.slice_dim0(row, 1);
            Tensor row_id = ids.slice_dim0(row, 1);
            decode_argmax(ctx, row_logits, row_id, vocab);
        }
    };
    bool used_graph = false;
    if (capture && capture->enabled) {
        uint64_t bucket = 0;
        for (int i = 0; i < rows; ++i) {
            bucket = bucket * (kMaxDecodeSplits + 1) + select_decode_splits(ctx.cost_hardware(),
                pos + i + 1, weights_.config().dense.num_key_value_heads, kMaxDecodeSplits);
        }
        used_graph = launch_spec_graph(ctx.stream(), capture->graphs[rows], bucket,
            sessions_.expert_residency_generation(), can_use_cuda_graph(), body);
    }
    if (!used_graph) {
        arena.reset(body_mark);
        body();
    }
    if (capture) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(mtp_target_hidden_.ptr<float>() + static_cast<size_t>(pos) * H,
            nextn.ptr(), nextn.nbytes(), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    if (retain_last_logits) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(work_.logits.ptr(), batch_logits.slice_dim0(rows - 1, 1).ptr(),
                                            vocab * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    if (greedy) {
        argmax.resize(rows);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(argmax.data(), ids.ptr(), rows * sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
    } else {
        logits.resize(static_cast<size_t>(rows) * vocab);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(logits.data(), batch_logits.ptr(), logits.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
    }
    ctx.synchronize();
    if (greedy && std::find(argmax.begin(), argmax.end(), -1) != argmax.end())
        throw std::runtime_error("Sampling found no selectable logits during speculative verification");
    return used_graph;
}

CudaExecutor::Request<CudaExecutor::Verify> CudaExecutor::begin_verify(
    CudaContext& ctx, bool native_mtp, bool use_graph, int batch_limit) {
    Request<Verify> verify(new Verify, [](Verify* p) { delete p; });
    verify->enabled = native_mtp && use_graph && can_use_cuda_graph() && !prof_acc().enabled;
    if (native_mtp && sessions_.recurrent_bytes())
        verify->replay.emplace(ctx, weights_.gdn_layers(), batch_limit);
    return verify;
}

bool CudaExecutor::restore_verified_recurrence(CudaContext& ctx, const Verify& verify, int rows, bool use_fused) {
    if (!verify.replay) return false;
    verify.replay->restore(ctx, weights_.gdn_layers(), sessions_.gdn(), rows, use_fused);
    return true;
}

bool CudaExecutor::can_use_cuda_graph() const { return all_experts_resident(weights_.moe_layers()); }

bool CudaExecutor::has_graph_prefix(bool fused) const {
    return moe_graph_prefix(weights_.ir(fused), weights_.moe_layers()).has_value();
}

void CudaExecutor::ensure_kv(CudaContext& ctx, int first_write, int tokens) {
    sessions_.ensure_cuda_kv(ctx, first_write, tokens, weights_.borrowable_experts());
}

void CudaExecutor::allocate_workspace(CudaArena& arena, int H, int vocab, int context) {
    work_.hidden = arena.alloc_tensor({H}, DType::F32, "hidden_buf");
    work_.logits = arena.alloc_tensor({vocab}, DType::F32, "logits_buf");
    work_.token_ids = arena.alloc_tensor({1 + static_cast<int64_t>(context)}, DType::I32, "tok_id_dev");
    work_.position = arena.alloc_tensor({1}, DType::I32, "pos_dev");
    work_.kv_length = arena.alloc_tensor({1}, DType::I32, "kv_len_dev");
    work_.invalid_token = arena.alloc_tensor({1}, DType::I32, "invalid_token");
}

struct CudaExecutor::Decode {
    const MiniGraph& ir;
    CudaIRWorkspace ws;
    std::unique_ptr<CudaOpTimer> timer;
    OpTrace* trace;
    bool fused;
    Decode(const MiniGraph& graph, OpTrace* t, bool f) : ir(graph), ws(graph), trace(t), fused(f) {}
};

CudaExecutor::Request<CudaExecutor::Decode> CudaExecutor::begin_decode(bool fused, OpTrace* trace, bool allocate_timer) {
    Request<Decode> request(new Decode(weights_.ir(fused), trace, fused), [](Decode* p) { delete p; });
    if (trace && allocate_timer) request->timer = std::make_unique<CudaOpTimer>(request->ir.nodes.size());
    return request;
}

bool CudaExecutor::decode(CudaContext& ctx, Decode& r, int pos, bool graph, bool argmax, bool first_step,
                          const int32_t* token, bool ensure) {
    if (ensure) ensure_kv(ctx, pos, pos + 1);
    if (token) set_token(ctx, token);
    return run_forward_token_step_cuda(ctx, weights_.config(), weights_.embeddings(), r.ir, r.ws,
        work_.hidden, work_.logits, work_.token_ids, work_.position, work_.kv_length, work_.invalid_token.ptr<int>(),
        weights_.dense_layers(), weights_.gdn_layers(), sessions_.kv(), sessions_.gdn(), weights_.moe_layers(),
        work_.decode_graph, work_.graph_captured, work_.graph_fused, work_.graph_has_argmax,
        work_.graph_split_bucket, pos, r.fused, graph, argmax, first_step, r.trace, r.timer.get());
}

void CudaExecutor::set_token(CudaContext& ctx, const int32_t* token) {
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(work_.token_ids.ptr(), token, sizeof(int32_t),
                                       cudaMemcpyHostToDevice, ctx.stream()));
}

void CudaExecutor::clear_invalid_token(CudaContext& ctx) {
    FUSIONRT_CUDA_CHECK(cudaMemsetAsync(work_.invalid_token.ptr(), 0, sizeof(int), ctx.stream()));
}

void CudaExecutor::download_token_history(CudaContext& ctx, int offset, std::span<int32_t> tokens,
                                             bool check_invalid) {
    int invalid = 0;
    if (check_invalid) FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(&invalid, work_.invalid_token.ptr(), sizeof(invalid),
                                                           cudaMemcpyDeviceToHost, ctx.stream()));
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(tokens.data(), work_.token_ids.ptr<int32_t>() + offset, tokens.size_bytes(),
                                       cudaMemcpyDeviceToHost, ctx.stream()));
    ctx.synchronize();
    if (invalid) throw std::runtime_error("Sampling found no selectable logits");
}

void CudaExecutor::store_token_history(CudaContext& ctx, int offset) {
    auto* ids = work_.token_ids.ptr<int32_t>();
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids + offset, ids, sizeof(int32_t), cudaMemcpyDeviceToDevice, ctx.stream()));
}

int32_t CudaExecutor::download_argmax(CudaContext& ctx) {
    int32_t tok = 0;
    download_token_history(ctx, 0, {&tok, 1});
    if (tok < 0) throw std::runtime_error("Sampling found no selectable logits");
    return tok;
}

int32_t CudaExecutor::sample_gpu(CudaContext& ctx) {
    decode_argmax(ctx, work_.logits, work_.token_ids, weights_.config().vocab_size);
    return download_argmax(ctx);
}

size_t CudaExecutor::inspect_scratch(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    struct RestoreProbe {
        CudaArena& arena;
        size_t mark;
        ~RestoreProbe() { arena.track_peak(false); arena.reset(mark); }
    } restore{arena, arena.mark()};
    CudaSessionManager probe_session(weights_, [] {});
    probe_session.start_sessions(sessions_.dtype(), 0);
    CudaExecutor probe(weights_, probe_session);
    probe.work_.logits = cuda_scratch(ctx, {weights_.config().vocab_size});
    probe.work_.token_ids = cuda_scratch(ctx, {1}, DType::I32);
    const int rows = probe.prefill_batch_limit(ctx);
    const std::vector<int32_t> tokens(rows, 0);
    const size_t mark = arena.mark();
    arena.track_peak(true);
    (void)probe.prefill_cuda_impl(ctx, tokens.data(), rows, 0, true);
    probe_session.reset_kv_and_state(ctx);
    (void)probe.prefill_cuda_impl(ctx, tokens.data(), 1, 0, true);
    decode_argmax(ctx, probe.work_.logits, probe.work_.token_ids, weights_.config().vocab_size);
    ctx.synchronize();
    return arena.peak() - mark;
}

struct CudaExecutor::MtpDraft {
    CudaIRWorkspace ws;
    Verify::Entry graph;
    Tensor args, seed, hidden, probability, mass;
    explicit MtpDraft(const MiniGraph& ir) : ws(ir) {}
};

CudaExecutor::Request<CudaExecutor::MtpDraft> CudaExecutor::begin_mtp(CudaContext& ctx, bool cutoff) {
    Request<MtpDraft> draft(new MtpDraft(weights_.mtp().graph), [](MtpDraft* p) { delete p; });
    const int H = weights_.config().dense.hidden_size;
    draft->args = cuda_scratch(ctx, {3}, DType::I32);
    draft->seed = cuda_scratch(ctx, {1, H});
    draft->hidden = cuda_scratch(ctx, {1, H});
    if (cutoff) {
        draft->probability = cuda_scratch(ctx, {1});
        draft->mass = cuda_scratch(ctx, {cuda_ops::kArgmaxParts});
    }
    return draft;
}

bool CudaExecutor::mtp_step(CudaContext& ctx, MtpDraft& draft, int position, int32_t token, int seed_position,
    bool use_draft_hidden, bool predict, bool use_fused, const Verify& verify) {
    const int H = weights_.config().dense.hidden_size;
    const auto* seed = use_draft_hidden ? draft.hidden.ptr<float>() :
        seed_position >= 0 ? mtp_target_hidden_.ptr<float>() + static_cast<size_t>(seed_position) * H : nullptr;
    auto& spec_arena = ctx.scratch_arena();
    const auto mark = spec_arena.mark();
    const int32_t args[] = {token, position, position + 1};
    auto* dev = draft.args.ptr<int32_t>();
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dev, args, sizeof(args), cudaMemcpyHostToDevice, ctx.stream()));
    if (predict && seed != draft.seed.ptr<float>())
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(draft.seed.ptr(), seed, H * sizeof(float),
                                            cudaMemcpyDeviceToDevice, ctx.stream()));
    auto ids = draft.args.slice_dim0(0, 1);
    auto body = [&] {
        auto emb = cuda_scratch(ctx, {1, H});
        auto concat = cuda_scratch(ctx, {1, 2 * H});
        auto projected = cuda_scratch(ctx, {1, H});
        embed_tokens(ctx, weights_.mtp().embedding, ids, emb, 1, H, 1.f);
        cuda_ops::rmsnorm_f32(emb.ptr<float>(), weights_.mtp().enorm.ptr<float>(), concat.ptr<float>(),
                              1, H, weights_.config().dense.rms_norm_eps, ctx.stream());
        auto* normalized_seed = concat.ptr<float>() + H;
        if (seed) {
            cuda_ops::rmsnorm_f32(predict ? draft.seed.ptr<float>() : seed, weights_.mtp().hnorm.ptr<float>(), normalized_seed,
                                  1, H, weights_.config().dense.rms_norm_eps, ctx.stream());
        } else {
            FUSIONRT_CUDA_CHECK(cudaMemsetAsync(normalized_seed, 0, H * sizeof(float), ctx.stream()));
        }
        dispatch_linear(ctx, concat.ptr<float>(), weights_.mtp().projection, projected.ptr<float>(), 1, H, H * 2, use_fused);
        if (weights_.mtp().projection_scale != 1.f)
            cuda_ops::scale_f32(projected.ptr<float>(), H, weights_.mtp().projection_scale, ctx.stream());
        execute_ir_graph_cuda(weights_.mtp().graph, draft.ws, ctx, projected, ids, work_.logits,
            1, position, position, dev + 1, dev + 2, weights_.dense_layers(), weights_.gdn_layers(), sessions_.kv(),
            sessions_.gdn(), weights_.moe_layers(), use_fused, nullptr, nullptr, nullptr, predict, false,
            nullptr, predict ? draft.hidden.ptr<float>() : nullptr);
        if (predict) {
            if (weights_.mtp().head_scale != 1.f)
                cuda_ops::scale_f32(work_.logits.ptr<float>(), weights_.config().vocab_size, weights_.mtp().head_scale, ctx.stream());
            decode_argmax(ctx, work_.logits, ids, weights_.config().vocab_size);
            if (draft.probability.data)
                cuda_ops::argmax_probability_f32(work_.logits.ptr<float>(), weights_.config().vocab_size, dev,
                    draft.probability.ptr<float>(), draft.mass.ptr<float>(), ctx.stream());
        }
    };
    const int bucket = select_decode_splits(ctx.cost_hardware(), position + 1,
        weights_.dense_layers().back().cfg.num_key_value_heads, kMaxDecodeSplits);
    const bool captured = predict && verify.enabled && launch_spec_graph(ctx.stream(), draft.graph, bucket,
        sessions_.expert_residency_generation(), can_use_cuda_graph(), body);
    if (!captured) {
        spec_arena.reset(mark);
        body();
    }
    spec_arena.reset(mark);
    return captured;
}

std::pair<int32_t, float> CudaExecutor::read_mtp(CudaContext& ctx, const MtpDraft& draft) {
    int32_t proposal = -1;
    float confidence = 1.f;
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(&proposal, draft.args.ptr(), sizeof(proposal),
                                       cudaMemcpyDeviceToHost, ctx.stream()));
    if (draft.probability.data)
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(&confidence, draft.probability.ptr(), sizeof(confidence),
                                           cudaMemcpyDeviceToHost, ctx.stream()));
    ctx.synchronize();
    if (proposal < 0) throw std::runtime_error("MTP draft found no selectable logits");
    return {proposal, confidence};
}

struct CudaExecutor::Bench {
    CudaIRWorkspace ws;
    std::unique_ptr<void, decltype(&cudaFreeHost)> host_logits{nullptr, cudaFreeHost};
    int32_t* host_args = nullptr;
    size_t logits_bytes = 0;
    Tensor args;
    CudaGraphExec graph;
    explicit Bench(const MiniGraph& ir) : ws(ir) {}
};

CudaExecutor::Request<CudaExecutor::Bench> CudaExecutor::begin_bench(CudaContext&, int pp) {
    Request<Bench> bench(new Bench(weights_.fused_ir()), [](Bench* p) { delete p; });
    void* host_ptr = nullptr;
    bench->logits_bytes = work_.logits.nbytes();
    FUSIONRT_CUDA_CHECK(cudaMallocHost(&host_ptr, bench->logits_bytes + sizeof(int32_t) * std::max(pp, 3)));
    bench->host_logits.reset(host_ptr);
    bench->host_args = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(host_ptr) + bench->logits_bytes);
    return bench;
}

int32_t* CudaExecutor::bench_args(Bench& b) { return b.host_args; }

void CudaExecutor::prepare_bench_decode(CudaContext& ctx, Bench& b) { b.args = cuda_scratch(ctx, {3}, DType::I32); }

void CudaExecutor::bench_prefill(CudaContext& ctx, Bench& b, int pp) {
    execute_prefill_chunks(ctx, b.host_args, pp, 0, true, nullptr, nullptr);
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(b.host_logits.get(), work_.logits.ptr(), b.logits_bytes,
                                       cudaMemcpyDeviceToHost, ctx.stream()));
}

void CudaExecutor::bench_decode(CudaContext& ctx, Bench& b, int pos, bool graph) {
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(b.args.ptr(), b.host_args, b.args.nbytes(), cudaMemcpyHostToDevice, ctx.stream()));
    const int H = static_cast<int>(weights_.embeddings().shape[1]);
    embed_tokens(ctx, weights_.embeddings(), b.args, work_.hidden, 1, H, weights_.config().embed_scale);
    execute_ir_graph_cuda(weights_.fused_ir(), b.ws, ctx, work_.hidden, b.args, work_.logits, 1, pos, pos,
        graph ? b.args.ptr<int32_t>() + 1 : nullptr, graph ? b.args.ptr<int32_t>() + 2 : nullptr,
        weights_.dense_layers(), weights_.gdn_layers(), sessions_.kv(), sessions_.gdn(), weights_.moe_layers(), true);
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(b.host_logits.get(), work_.logits.ptr(), b.logits_bytes,
                                       cudaMemcpyDeviceToHost, ctx.stream()));
}

void CudaExecutor::capture_bench(CudaContext& ctx, Bench& b, int size_or_pos, bool prefill) {
    b.graph.capture(ctx.stream(), [&] {
        if (prefill) bench_prefill(ctx, b, size_or_pos);
        else bench_decode(ctx, b, size_or_pos, true);
    });
}

void CudaExecutor::launch_bench(CudaContext& ctx, Bench& b) { b.graph.launch(ctx.stream()); }

void CudaExecutor::clear_bench_graph(Bench& b) { b.graph.reset(); }

}  // namespace fusionrt
#endif
