#include "cuda_ir.h"
#include "cuda_profile.h"
#include "../core/moe_route_trace.h"
#include "fusionrt/weight_codec.h"

#ifdef FUSIONRT_ENABLE_CUDA

#include <algorithm>

namespace fusionrt {
namespace {

constexpr int kPrefillCarrySlots = 8;
constexpr int kMaxDecodeSplits = 128;

static OpTiming operator_work(const IRNode& node, int rows, int pos, const CudaContext& ctx,
                               bool use_fused, bool packed_input, bool packed_output,
                               const CudaDenseGQALayer* dense, const CudaSparseMoELayer* moe,
                               DType kv_dtype) {
    const int64_t m = rows, f32 = sizeof(float);
    const int64_t n = node.op == OpKind::GQA_ATTN_BLOCK ? node.attention.query_heads :
                      node.op == OpKind::GDN_RECURRENT_BLOCK ? node.recurrent.key_heads : node.output_cols;
    const int64_t k = node.op == OpKind::GQA_ATTN_BLOCK ? node.attention.kv_heads :
                      node.op == OpKind::GDN_RECURRENT_BLOCK ? node.recurrent.value_heads : node.input_cols;
    OpTiming work;
    work.name = node.tag;
    work.kind = op_kind_name(node.op);
    work.calls = 1;
    auto norm = [&](int64_t width, int64_t count, bool output) {
        work.bytes += f32 * width * (1 + count * (output ? 2 : 1));
        work.flops += count * (4 * width + 2);
    };
    auto linear = [&](const Tensor& w, bool reads_activation) {
        const bool codec = has_weight_codec_int8(w.dtype);
        const bool q8 = codec || (supports_q8_1_decode(w.dtype) &&
                        (m == 1 || (use_fused && m >= ctx.mmq_crossover_m())));
        const bool mmq = codec ? use_codec_mmq(ctx, rows, use_fused) : q8 && m > 1;
        work.bytes += w.nbytes() + f32 * m * (n + (reads_activation && !packed_input ? k : 0));
        work.flops += 2 * m * n * k;
        if (q8) work.bytes += (packed_input ? 1 : 2) * m * row_nbytes(DType::Q8_1, k);
        if (mmq) work.kind += "_MMQ_INT8";
        else if (q8) work.kind += "_GEMV_INT8";
        else if (m == 1) work.kind += "_GEMV";
        else if (use_fused && m >= 4 && w.dtype != DType::F32 && !prefill_mmq(w.dtype) &&
                 (m >= 16 || w.dtype == DType::F16 || w.dtype == DType::BF16 || w.dtype == DType::Q4_1)) {
            work.bytes += 2 * sizeof(uint16_t) * m * k;
            if (w.dtype != DType::F16) work.bytes += 2 * sizeof(uint16_t) * n * k;
            work.kind += "_GEMM_FP16";
        } else work.kind += "_LINEAR";
    };
    switch (node.op) {
        case OpKind::FUSED_ADD_RMSNORM:
            work.flops += m * k;
            work.bytes += 2 * f32 * m * k;
            [[fallthrough]];
        case OpKind::RMSNORM:
            norm(k, m, !packed_output);
            if (packed_output) {
                work.bytes += m * row_nbytes(DType::Q8_1, k);
                work.kind += "_Q8_1";
            }
            break;
        case OpKind::SHARED_Q8_PROJECTIONS: {
            const auto cols = node.secondary_weight->shape[0];
            work.bytes += node.secondary_weight->nbytes() + f32 * m * cols;
            if (packed_input) work.bytes += m * row_nbytes(DType::Q8_1, k);
            work.flops += 2 * m * cols * k;
            [[fallthrough]];
        }
        case OpKind::LINEAR:
            linear(*node.weight, true);
            break;
        case OpKind::FUSED_ADD_RMSNORM_LINEAR:
            work.flops += m * k;
            work.bytes += f32 * m * k * (node.residual_out >= 0 && !node.output_logits ? 2 : 1);
            [[fallthrough]];
        case OpKind::FUSED_RMSNORM_LINEAR:
            norm(k, m, false);
            linear(*node.weight, false);
            break;
        case OpKind::ADD_RESIDUAL:
            work.bytes = 3 * f32 * m * n;
            work.flops = m * n;
            break;
        case OpKind::GEMMA_PLE_GATE:
        case OpKind::GEGLU_INTERLEAVED:
        case OpKind::SWIGLU_INTERLEAVED:
            work.bytes = 3 * f32 * m * n;
            work.flops = 5 * m * n;
            break;
        case OpKind::GEGLU_Q8_LINEAR:
        case OpKind::SWIGLU_Q8_LINEAR:
            work.bytes = 2 * f32 * m * k;
            work.flops = 5 * m * k;
            linear(*node.weight, false);
            break;
        case OpKind::MLA_ATTN_BLOCK: {
            const auto& d = dense->cfg;
            const int64_t heads = d.num_attention_heads, rank = d.mla.latent_rank, width = d.mla.latent_width();
            const int64_t visible = std::max<int64_t>(1, pos + 1);
            const int64_t projections = rank * (d.head_dim - d.mla.rope_dims + d.mla.value_dim);
            work.flops += 2 * m * heads * (projections + visible * (width + rank));
            work.bytes += m * (dense->mla_k_b.nbytes() + dense->mla_v_b.nbytes() +
                              f32 * heads * visible * width);
            break;
        }
        case OpKind::GQA_ATTN_BLOCK: {
            const int64_t hd = node.attention.head_dim;
            const int64_t pairs = m * pos + m * (m + 1) / 2;
            work.flops = n * pairs * (4 * hd + 4) + 3 * m * (n + k) * node.attention.rotary_dims;
            work.bytes = f32 * hd * 2 * m * (n + k) + row_nbytes(kv_dtype, hd) * 2 * (m + pairs) * k;
            if (node.attention.query_norm) norm(hd, m * n, false);
            if (node.attention.key_norm) norm(hd, m * k, false);
            if (node.attention.gated) {
                work.flops += 5 * m * n * hd;
                work.bytes += 4 * f32 * m * n * hd;
            }
            for (const Tensor* bias : {&dense->q_bias, &dense->k_bias, &dense->v_bias}) {
                if (!bias->data) continue;
                work.flops += m * bias->numel();
                work.bytes += (2 * m + 1) * bias->nbytes();
            }
            break;
        }
        case OpKind::GDN_RECURRENT_BLOCK: {
            const int64_t dk = node.recurrent.key_dim, dv = node.recurrent.value_dim;
            const int64_t conv = 2 * n * dk + k * dv;
            const int64_t state = k * dk * dv;
            work.bytes = f32 * (m * (conv + 2 * k * dv + 2 * k) + 10 * conv + 2 * state + 2 * k + dv);
            work.flops = m * (12 * conv + 6 * n * dk + k * (10 * dv + 12));
            if (m < 4 || (dk != 64 && dk != 128) || dv % 16 != 0 || k % n != 0) {
                work.flops += 7 * m * state;
            } else {
                for (int64_t start = 0; start < m; start += 32) {
                    const int64_t count = std::min<int64_t>(32, m - start);
                    const int64_t triangle = count * (count + 1) / 2;
                    const bool has_state = pos > 0 || start > 0;
                    work.flops += k * (4 * triangle * (dk + dv) + count * (count * count - 1) / 3)
                                  + (2 + (has_state ? 4 : 0)) * count * state + 2 * state;
                    work.bytes += f32 * k * (2 * triangle + 2 * count) * (1 + dv / 16);
                }
            }
            break;
        }
        case OpKind::MOE_FFN_BLOCK: {
            const int64_t experts = moe->cfg.num_experts, selected = moe->cfg.top_k;
            const int64_t inter = moe->cfg.intermediate_size;
            norm(n, m, true);
            work.flops += m * (2 * n * experts + selected * (6 * n * inter + 5 * inter + 2 * n));
            work.bytes += m * (moe->router_w.nbytes() + f32 * (selected * 3 * n * inter + experts + 2 * n));
            if (moe->shared_down_w.data) {
                work.bytes += m * (moe->shared_gate_up_w.nbytes() + moe->shared_down_w.nbytes());
                work.flops += m * (6 * n + 5) * moe->cfg.shared_intermediate_size;
            }
            break;
        }
    }
    return work;
}

}  // namespace
template<bool Observe>
void execute_ir_graph_cuda(
    const MiniGraph& g, CudaIRWorkspace& ws, CudaContext& ctx,
    const Tensor& x_input, const Tensor& token_ids, Tensor& logits_buf,
    int M, int pos_host, int start_pos,
    const int* pos_dev, const int* kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvView>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    bool use_fused, OpTrace* op_trace, CudaOpTimer* timer,
    LayerTrace* layer_trace, bool compute_logits, bool all_logits,
    const CudaIRSegment* segment, float* nextn_out, GdnReplay* replay) {
    size_t mark = ctx.scratch_arena().mark();
    int H = g.hidden_size;
    const int first_node = segment ? segment->first : 0;
    const int num_nodes = segment ? segment->end : static_cast<int>(g.nodes.size());
    const auto& last_use = ws.last_use;

    Tensor ple;
    if (const auto* embedding = g.gemma_inputs.token_embeddings) {
        const int width = static_cast<int>(g.gemma_inputs.norm->shape[0]);
        const int cols = static_cast<int>(embedding->shape[1]);
        ple = cuda_scratch(ctx, {M, cols});
        const auto ple_mark = ctx.scratch_arena().mark();
        auto projected = cuda_scratch(ctx, {M, cols});
        auto normalized = cuda_scratch(ctx, {M, cols});
        dispatch_linear(ctx, x_input.ptr<float>(), *g.gemma_inputs.projection, projected.ptr<float>(), M, cols, H, use_fused, all_logits);
        // Absorb the 1/sqrt(H) projection scale into the RMSNorm epsilon.
        cuda_ops::rmsnorm_f32(projected.ptr<float>(), g.gemma_inputs.norm->ptr<float>(), normalized.ptr<float>(),
                              M * cols / width, width, g.gemma_input_eps * H, ctx.stream());
        cuda_ops::embed_lookup_f32(embedding->ptr(), embedding->dtype, token_ids.ptr<int32_t>(),
                                  ple.ptr<float>(), M, cols, ctx.stream());
        cuda_ops::scale_f32(ple.ptr<float>(), M * cols, std::sqrt(static_cast<float>(width)), ctx.stream());
        cuda_ops::add_residual_f32(ple.ptr<float>(), normalized.ptr<float>(), ple.ptr<float>(), M * cols, ctx.stream());
        cuda_ops::scale_f32(ple.ptr<float>(), M * cols, 0.7071067811865475f, ctx.stream());
        ctx.scratch_arena().reset(ple_mark);
    }

    std::array<Tensor, kPrefillCarrySlots> carry_bufs;
    std::array<int, kPrefillCarrySlots> carry_busy_until;
    carry_busy_until.fill(-1);
    for (int s = 0; s < kPrefillCarrySlots; ++s) {
        carry_bufs[s] = cuda_scratch(ctx, {M, H});
    }
    size_t layer_reset_mark = ctx.scratch_arena().mark();
    int non_carry_busy_until = -1;

    auto& bufs = ws.bufs;
    std::fill(bufs.begin(), bufs.end(), Tensor{});
    bufs[segment ? segment->input_value : g.input_value] =
        Tensor::make_contiguous({M, H}, DType::F32, Backend::CUDA, x_input.ptr());
    auto& pa = prof_acc();

    for (int ni = first_node; ni < num_nodes; ++ni) {
        const IRNode& node = g.nodes[ni];
        if (!node.alive) continue;
        if (ni > non_carry_busy_until) {
            ctx.scratch_arena().reset(layer_reset_mark);
            non_carry_busy_until = -1;
        }
        const bool is_lm_head = node.output_logits;
        if (is_lm_head && !compute_logits && node.residual_out < 0) continue;
        const bool tail_logits = is_lm_head && compute_logits && !all_logits;
        const int rows = tail_logits ? 1 : M;

        auto get_out_buf = [&](int vid, int cols, int r_cnt) -> Tensor& {
            if (vid == g.output_value && is_lm_head) {
                bufs[vid] = logits_buf;
                return bufs[vid];
            }
            if (bufs[vid].data != nullptr) return bufs[vid];
            if (cols == H && r_cnt == M && !is_lm_head) {
                for (int s = 0; s < kPrefillCarrySlots; ++s) {
                    if (carry_busy_until[s] < ni) {
                        carry_busy_until[s] = std::max(ni, last_use[vid]);
                        bufs[vid] = carry_bufs[s];
                        return bufs[vid];
                    }
                }
            }
            bufs[vid] = cuda_scratch(ctx, {r_cnt, cols});
            non_carry_busy_until = std::max(non_carry_busy_until, last_use[vid]);
            return bufs[vid];
        };

        auto row_ptr = [&](int vid, bool tail_only) -> const float* {
            const Tensor& t = bufs[vid];
            if (!tail_only || M <= 1) return t.ptr<float>();
            return t.ptr<float>() + static_cast<size_t>(M - 1) * static_cast<size_t>(g.hidden_size);
        };

        if constexpr (Observe) {
            if (timer) FUSIONRT_CUDA_CHECK(cudaEventRecord(timer->event(ni, 0), ctx.stream()));
        }
        switch (node.op) {
            case OpKind::RMSNORM:
            case OpKind::FUSED_ADD_RMSNORM: {
                const bool add = node.op == OpKind::FUSED_ADD_RMSNORM;
                const float* a = bufs[node.inputs[0]].ptr<float>();
                const float* b = add ? bufs[node.inputs[1]].ptr<float>() : nullptr;
                float* res = add ? get_out_buf(node.residual_out, node.input_cols, rows).template ptr<float>() : nullptr;
                const float* weight = node.norm_weight->ptr<float>();
                if (node.share_quantized_activation && (rows == 1 || rows >= ctx.mmq_crossover_m())) {
                    bufs[node.out] = ctx.scratch_arena().alloc_tensor({rows, node.input_cols}, DType::Q8_1);
                    non_carry_busy_until = std::max(non_carry_busy_until, last_use[node.out]);
                    cuda_ops::quantize_x_q8_1(a, b, weight, res, bufs[node.out].ptr(), rows,
                                              node.input_cols, node.norm_eps, ctx.stream());
                } else {
                    float* out = get_out_buf(node.out, node.output_cols, rows).template ptr<float>();
                    if (add) cuda_ops::add_rmsnorm_f32(a, b, weight, res, out, rows, node.input_cols,
                                                       node.norm_eps, ctx.stream());
                    else cuda_ops::rmsnorm_f32(a, weight, out, rows, node.input_cols, node.norm_eps, ctx.stream());
                }
                break;
            }
            case OpKind::SHARED_Q8_PROJECTIONS:
            case OpKind::LINEAR: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* in_p = row_ptr(node.inputs[0], tail_logits);
                auto fn = [&]() {
                    const bool packed = bufs[node.inputs[0]].dtype == DType::Q8_1;
                    auto project = [&](const Tensor& w, float* out, int cols) {
                        if (!packed) {
                            if (pos_dev && rows > 1 && w.dtype == DType::F32) {
                                for (int r = 0; r < rows; ++r)
                                    dispatch_linear(ctx, in_p + static_cast<size_t>(r) * node.input_cols, w,
                                        out + static_cast<size_t>(r) * cols, 1, cols, node.input_cols, use_fused, true);
                            } else {
                                dispatch_linear(ctx, in_p, w, out, rows, cols, node.input_cols, use_fused, all_logits);
                            }
                        } else if (rows == 1) {
                            decode_gemv(w.dtype)(in_p, w.ptr(), out, cols, node.input_cols,
                                                  ctx.stream(), ctx.preferred_gemv_geometry());
                        } else {
                            prefill_mmq(w.dtype)(in_p, w.ptr(), out, rows, cols, node.input_cols,
                                                 ctx.stream(), ctx.cost_hardware());
                        }
                    };
                    const Tensor& w = *node.weight;
                    if (node.secondary_weight) {
                        const Tensor& other = *node.secondary_weight;
                        const int cols = static_cast<int>(other.shape[0]);
                        auto* extra = get_out_buf(node.secondary_out, cols, rows).template ptr<float>();
                        if (packed && cuda_ops::project_q4_a_and_q8_a(in_p, w.ptr(), other.ptr(), out_t.ptr<float>(),
                                extra, rows, node.output_cols, cols, node.input_cols, ctx.stream(),
                                ctx.preferred_gemv_geometry())) return;
                        project(other, extra, cols);
                    }
                    project(w, out_t.ptr<float>(), node.output_cols);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::FUSED_RMSNORM_LINEAR: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* in_p = row_ptr(node.inputs[0], tail_logits);
                auto fn = [&]() {
                    dispatch_rmsnorm_linear(ctx, in_p, *node.norm_weight, *node.weight, out_t.ptr<float>(),
                                            rows, node.output_cols, node.input_cols, node.norm_eps, use_fused, all_logits);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::ADD_RESIDUAL: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const int count = rows * node.output_cols;
                cuda_ops::add_residual_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                                           out_t.ptr<float>(), count, ctx.stream());
                if (node.output_scale != 1.0f) cuda_ops::scale_f32(out_t.ptr<float>(), count, node.output_scale, ctx.stream());
                break;
            }
            case OpKind::FUSED_ADD_RMSNORM_LINEAR: {
                if (is_lm_head && !compute_logits) {
                    Tensor& residual = get_out_buf(node.residual_out, node.input_cols, M);
                    cuda_ops::add_residual_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                                              residual.ptr<float>(), M * node.input_cols, ctx.stream());
                    break;
                }
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                float* res_upd_p = nullptr;
                if (node.residual_out >= 0 && !is_lm_head) {
                    Tensor& r_t = get_out_buf(node.residual_out, node.input_cols, rows);
                    res_upd_p = r_t.ptr<float>();
                }
                const float* a_p = row_ptr(node.inputs[0], tail_logits);
                const float* b_p = row_ptr(node.inputs[1], tail_logits);
                auto fn = [&]() {
                    dispatch_add_rmsnorm_linear(ctx, a_p, b_p, *node.norm_weight, *node.weight, res_upd_p,
                                                out_t.ptr<float>(), rows, node.output_cols, node.input_cols, node.norm_eps, use_fused, all_logits);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::MLA_ATTN_BLOCK: {
                const auto& layer = dense_layers[node.layer_index];
                const auto& d = layer.cfg;
                const int heads = d.num_attention_heads;
                const auto& mla = d.mla;
                auto& output = get_out_buf(node.out, node.output_cols, rows);
                auto query = cuda_scratch(ctx, {rows, heads, mla.latent_width()});
                auto latent = cuda_scratch(ctx, {rows, heads, mla.latent_rank});
                cuda_ops::mla_attention_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                    layer.mla_kv_a_norm.ptr<float>(), layer.mla_k_b, layer.mla_v_b, layer.rope_factors.ptr<float>(),
                    query.ptr<float>(), latent.ptr<float>(), output.template ptr<float>(), kv_states[node.layer_index],
                    rows, heads, d.head_dim, start_pos, pos_dev, d.rms_norm_eps, mla.attention_scale,
                    mla.rope_amplitude, ctx.stream());
                break;
            }
            case OpKind::GQA_ATTN_BLOCK: {
                auto run_gqa = [&]() {
                    const auto& L = dense_layers[static_cast<size_t>(node.layer_index)];
                    const auto& cfg = node.attention;
                    const bool shared = L.kv_source_layer >= 0;
                    const auto& state = kv_states[shared ? L.kv_source_layer : node.layer_index];
                    const int nq = cfg.query_heads, nkv = cfg.kv_heads, hd = cfg.head_dim;
                    const int q_dim = nq * hd, q_proj_dim = cfg.gated ? 2 * q_dim : q_dim;
                    const int kv_dim = nkv * hd;
                    float* q_full_p = bufs[node.inputs[0]].ptr<float>() + node.input_offsets[0];
                    const bool has_value = node.inputs.size() == 3;
                    float* k_p = shared ? nullptr : bufs[node.inputs[1]].ptr<float>() + node.input_offsets[1];
                    float* v_p = has_value ? bufs[node.inputs[2]].ptr<float>() + node.input_offsets[2] : nullptr;
                    int q_stride = g.values[node.inputs[0]].cols;
                    const int k_stride = shared ? 0 : g.values[node.inputs[1]].cols;
                    const int v_stride = has_value ? g.values[node.inputs[2]].cols : 0;
                    const int project_kv_heads = shared ? 0 : nkv;
                    if (L.q_bias.data) cuda_ops::add_bias_f32(q_full_p, L.q_bias.ptr<float>(), rows, q_proj_dim, q_stride, ctx.stream());
                    if (L.k_bias.data) cuda_ops::add_bias_f32(k_p, L.k_bias.ptr<float>(), rows, kv_dim, k_stride, ctx.stream());
                    if (L.v_bias.data) cuda_ops::add_bias_f32(v_p, L.v_bias.ptr<float>(), rows, kv_dim, v_stride, ctx.stream());

                    float* q_p = q_full_p;
                    float* gate_p = nullptr;
                    if (node.attention.gated) {
                        Tensor qb = cuda_scratch(ctx, {rows, q_dim});
                        Tensor gb = cuda_scratch(ctx, {rows, q_dim});
                        q_p = qb.ptr<float>();
                        gate_p = gb.ptr<float>();
                        cuda_ops::split_q_gate_f32(q_full_p, q_p, gate_p, rows, nq, hd, ctx.stream(), q_stride);
                        q_stride = q_dim;
                    }
                    const auto* next = ni + 1 < num_nodes ? &g.nodes[ni + 1] : nullptr;
                    const bool quantize_output = use_fused && rows == 1 && hd % 32 == 0 && hd <= 256 &&
                        next && next->alive && next->op == OpKind::LINEAR && next->inputs[0] == node.out &&
                        last_use[node.out] == ni + 1 && decode_gemv(next->weight->dtype);
                    if (quantize_output) {
                        bufs[node.out] = cuda_scratch(ctx, {q_dim}, DType::Q8_1);
                        non_carry_busy_until = std::max(non_carry_busy_until, last_use[node.out]);
                    }
                    Tensor& attn_out = quantize_output ? bufs[node.out] : get_out_buf(node.out, q_dim, rows);
                    const float scale = L.attention_scale > 0.0f ? L.attention_scale : 1.0f / std::sqrt(static_cast<float>(hd));
                    const float* qnw = cfg.query_norm ? cfg.query_norm->ptr<float>() : nullptr;
                    const float* knw = cfg.key_norm ? cfg.key_norm->ptr<float>() : nullptr;
                    const float* rotary = cfg.rotary_factors ? cfg.rotary_factors->ptr<float>() : nullptr;
                    if (rows == 1) {
                        cuda_ops::fused_qk_norm_rope_kv_append_f32(
                            q_p, k_p, v_p, qnw, knw,
                            nullptr, nullptr,
                            nq, project_kv_heads, hd, cfg.rotary_dims, pos_dev, pos_host, cfg.max_context,
                            cfg.rope_theta, node.norm_eps, cfg.rope_neox, ctx.stream(), rotary, state.dtype, state, L.v_norm);
                        const int split_heads = g.decode_split_kv_heads ? g.decode_split_kv_heads : nkv;
                        const int splits = select_decode_splits(ctx.cost_hardware(), pos_host + 1, split_heads, kMaxDecodeSplits);
                        Tensor part_out = cuda_scratch(ctx, {nq, splits, hd});
                        Tensor part_ml = cuda_scratch(ctx, {nq, splits, 2});
                        // Use the same reduction order for eager and captured decode.
                        cuda_ops::gqa_splitk_decode_f32(
                            q_p, nullptr, nullptr,
                            attn_out.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                            nq, nkv, hd, kv_len_dev, pos_host + 1, cfg.max_context,
                            scale, splits, ctx.stream(), true, state.dtype,
                            quantize_output ? attn_out.ptr<BlockQ8_1>() : nullptr, gate_p, state, L.sliding_window);
                    } else {
                        cuda_ops::batched_qk_norm_rope_kv_append_f32(
                            q_p, k_p, v_p, qnw, knw,
                            nullptr, nullptr,
                            rows, nq, project_kv_heads, hd, cfg.rotary_dims, start_pos, cfg.max_context,
                            cfg.rope_theta, node.norm_eps, cfg.rope_neox, ctx.stream(), q_stride, k_stride, v_stride,
                            rotary, state.dtype, state, L.v_norm, pos_dev);
                        if (pos_dev) {
                            // Verification uses the decode reduction for every row, with weights already projected in batch.
                            const int split_heads = g.decode_split_kv_heads ? g.decode_split_kv_heads : nkv;
                            const auto mark = ctx.scratch_arena().mark();
                            for (int i = 0; i < rows; ++i) {
                                const int splits = select_decode_splits(ctx.cost_hardware(), start_pos + i + 1,
                                                                         split_heads, kMaxDecodeSplits);
                                auto part_out = cuda_scratch(ctx, {nq, splits, hd});
                                auto part_ml = cuda_scratch(ctx, {nq, splits, 2});
                                cuda_ops::gqa_splitk_decode_f32(q_p + static_cast<size_t>(i) * q_stride,
                                    nullptr, nullptr, attn_out.ptr<float>() + static_cast<size_t>(i) * q_dim,
                                    part_out.ptr<float>(), part_ml.ptr<float>(), nq, nkv, hd, kv_len_dev + i,
                                    start_pos + i + 1, cfg.max_context, scale, splits, ctx.stream(), true, state.dtype,
                                    nullptr, gate_p ? gate_p + static_cast<size_t>(i) * q_dim : nullptr,
                                    state, L.sliding_window);
                                ctx.scratch_arena().reset(mark);
                            }
                        } else if ((hd == 64 || hd == 128 || hd == 256) && start_pos + rows >= 512) {
                            cuda_ops::gqa_prefill_online_f32(q_p, q_stride, state, attn_out.ptr<float>(),
                                rows, nq, nkv, hd, start_pos, scale, ctx.stream(), L.sliding_window);
                        } else {
                            cuda_ops::gqa_prefill_causal_f32(q_p, nullptr, nullptr, attn_out.ptr<float>(),
                                rows, nq, nkv, hd, start_pos, cfg.max_context, scale, ctx.stream(),
                                q_stride, state.dtype, state, L.sliding_window);
                        }
                    }
                    if (node.attention.gated && !quantize_output) {
                        cuda_ops::apply_sigmoid_gate_f32(attn_out.ptr<float>(), gate_p, rows * q_dim, ctx.stream());
                    }
                };
                if (pa.enabled && M > 1) pa.attn_ms += time_cuda_fn(ctx.stream(), run_gqa);
                else run_gqa();
                break;
            }
            case OpKind::GDN_RECURRENT_BLOCK: {
                auto run_gdn = [&]() {
                    const auto& L = gdn_layers[static_cast<size_t>(node.layer_index)];
                    const auto& cfg = node.recurrent;
                    const auto& state = gdn_states[node.layer_index];
                    const int nk = cfg.key_heads, nv = cfg.value_heads, dk = cfg.key_dim, dv = cfg.value_dim;
                    const int conv_dim = 2 * nk * dk + nv * dv;
                    const int v_dim = nv * dv;
                    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
                    const float* qkv_p = bufs[node.inputs[0]].ptr<float>() + node.input_offsets[0];
                    const float* z_p = bufs[node.inputs[1]].ptr<float>() + node.input_offsets[1];
                    const float* a_p = bufs[node.inputs[2]].ptr<float>() + node.input_offsets[2];
                    const float* b_p = bufs[node.inputs[3]].ptr<float>() + node.input_offsets[3];
                    Tensor& gdn_out = get_out_buf(node.out, v_dim, rows);
                    if (replay && rows > 1) {
                        const std::array<int, 4> strides{g.values[node.inputs[0]].cols, g.values[node.inputs[1]].cols,
                            g.values[node.inputs[2]].cols, g.values[node.inputs[3]].cols};
                        const std::array<const float*, 4> projections{qkv_p, z_p, a_p, b_p};
                        replay->save(ctx, node.layer_index, rows, projections, {conv_dim, v_dim, nv, nv}, strides);
                        GdnReplay::steps(ctx, L, state, projections, strides, gdn_out.ptr<float>(),
                                         rows, node.norm_eps, use_fused);
                    } else if (rows == 1) {
                        if (use_fused) {
                            float* scratch = nullptr;
                            if (dv >= 64 && (dk == 64 || dk == 128)) {
                                scratch = cuda_scratch(ctx, {conv_dim + v_dim + 2 * nv}).ptr<float>();
                            }
                            cuda_ops::fused_gated_deltanet_step_f32(
                                qkv_p, z_p, a_p, b_p,
                                L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                                L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                                gdn_out.ptr<float>(), nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(), scratch,
                                cfg.tiled_value_heads, L.cfg.conv_kernel_size);
                        } else {
                            Tensor sqkv = cuda_scratch(ctx, {conv_dim});
                            Tensor sy = cuda_scratch(ctx, {v_dim});
                            cuda_ops::unfused_gated_deltanet_step_f32(
                                qkv_p, z_p, a_p, b_p,
                                L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                                L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                                gdn_out.ptr<float>(), sqkv.ptr<float>(), sy.ptr<float>(),
                                nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(), cfg.tiled_value_heads, L.cfg.conv_kernel_size);
                        }
                    } else {
                        int s_qkv = g.values[node.inputs[0]].cols;
                        int s_z = g.values[node.inputs[1]].cols;
                        int s_a = g.values[node.inputs[2]].cols;
                        int s_b = g.values[node.inputs[3]].cols;
                        Tensor gdn_ws = cuda_scratch(ctx, {static_cast<int64_t>(
                            cuda_ops::gated_deltanet_prefill_scratch_floats(rows, nk, nv, dk, dv))});
                        cuda_ops::fused_gated_deltanet_prefill_f32(
                            qkv_p, s_qkv, z_p, s_z, a_p, s_a, b_p, s_b,
                            L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                            L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                            gdn_out.ptr<float>(), rows, nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(),
                            gdn_ws.ptr<float>(), gdn_ws.numel(), cfg.tiled_value_heads, L.cfg.conv_kernel_size);
                    }
                };
                if (pa.enabled && M > 1) pa.gdn_ms += time_cuda_fn(ctx.stream(), run_gdn);
                else run_gdn();
                break;
            }
            case OpKind::GEMMA_PLE_GATE: {
                const int width = node.output_cols;
                auto& out = get_out_buf(node.out, width, rows);
                cuda_ops::gemma_ple_gate_f32(bufs[node.inputs[0]].ptr<float>(), ple.ptr<float>(),
                    out.template ptr<float>(), rows, width, static_cast<int>(ple.shape[1]),
                    node.layer_index * width, ctx.stream());
                break;
            }
            case OpKind::GEGLU_INTERLEAVED:
            case OpKind::SWIGLU_INTERLEAVED: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* gate = bufs[node.inputs[0]].ptr<float>();
                const bool geglu = node.op == OpKind::GEGLU_INTERLEAVED;
                if (node.inputs.size() == 1) {
                    (geglu ? cuda_ops::geglu_interleaved_f32 : cuda_ops::swiglu_interleaved_f32)(gate, out_t.ptr<float>(), rows, node.output_cols, ctx.stream());
                } else {
                    (geglu ? cuda_ops::geglu_f32 : cuda_ops::swiglu_f32)(gate, bufs[node.inputs[1]].ptr<float>(), out_t.ptr<float>(),
                                        rows * node.output_cols, ctx.stream());
                }
                break;
            }
            case OpKind::GEGLU_Q8_LINEAR:
            case OpKind::SWIGLU_Q8_LINEAR: {
                const bool geglu = node.op == OpKind::GEGLU_Q8_LINEAR;
                const auto quantize = geglu ? cuda_ops::geglu_interleaved_q8_1 : cuda_ops::swiglu_interleaved_q8_1;
                const float* gate_up = bufs[node.inputs[0]].ptr<float>();
                float* out = get_out_buf(node.out, node.output_cols, rows).template ptr<float>();
                auto& arena = ctx.scratch_arena();
                size_t scratch_mark = arena.mark();
                if (rows == 1) {
                    void* q8 = arena.alloc(row_nbytes(DType::Q8_1, node.input_cols));
                    quantize(gate_up, q8, rows, node.input_cols, ctx.stream());
                    decode_gemv(node.weight->dtype)(q8, node.weight->ptr(), out, node.output_cols, node.input_cols, ctx.stream(),
                                               ctx.preferred_gemv_geometry());
                } else if (auto mmq = rows >= ctx.mmq_crossover_m() ? prefill_mmq(node.weight->dtype) : nullptr) {
                    void* q8 = arena.alloc(static_cast<size_t>(rows) * row_nbytes(DType::Q8_1, node.input_cols));
                    quantize(gate_up, q8, rows, node.input_cols, ctx.stream());
                    mmq(q8, node.weight->ptr(), out, rows, node.output_cols, node.input_cols,
                        ctx.stream(), ctx.cost_hardware());
                } else {
                    Tensor act = cuda_scratch(ctx, {rows, node.input_cols});
                    (geglu ? cuda_ops::geglu_interleaved_f32 : cuda_ops::swiglu_interleaved_f32)(gate_up, act.ptr<float>(), rows, node.input_cols, ctx.stream());
                    dispatch_linear(ctx, act.ptr<float>(), *node.weight, out, rows, node.output_cols, node.input_cols, use_fused, all_logits);
                }
                arena.reset(scratch_mark);
                break;
            }
            case OpKind::MOE_FFN_BLOCK: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(out_t.ptr(), bufs[node.inputs[0]].ptr(),
                                                    static_cast<size_t>(rows) * node.output_cols * sizeof(float),
                                                    cudaMemcpyDeviceToDevice, ctx.stream()));
                Tensor x_v = (rows == 1) ? out_t.view({node.output_cols}) : out_t;
                const auto& L = moe_layers[static_cast<size_t>(node.layer_index)];
                const bool trace_routes = ctx.moe_route_trace && ctx.moe_route_trace->enabled_for(&moe_layers);
                const int index = trace_routes || (rows == 1 && !(segment && segment->boundary)) ? node.layer_index : -1;
                L.forward_prefill_cuda(ctx, x_v, rows, use_fused, index, trace_routes ? start_pos : -1);
                break;
            }
        }
        if (nextn_out && node.out == g.nextn_hidden) {
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(nextn_out, bufs[node.out].ptr(),
                                                static_cast<size_t>(M) * H * sizeof(float),
                                                cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        if (is_lm_head && compute_logits && g.final_logit_softcapping > 0.0f) {
            cuda_ops::softcap_f32(logits_buf.ptr<float>(), rows * node.output_cols, g.final_logit_softcapping, ctx.stream());
        }
        if constexpr (Observe) {
            if (timer) FUSIONRT_CUDA_CHECK(cudaEventRecord(timer->event(ni, 1), ctx.stream()));
            if (layer_trace) {
                int value = node.residual_out >= 0 ? node.residual_out : node.out;
                const std::string& name = g.values[value].name;
                if (name.starts_with("L") &&
                    (name.ends_with("_h2") || name.ends_with("_h2_upd") || name.ends_with("_moe_out"))) {
                    const size_t layer = std::stoul(name.substr(1, name.find('_') - 1));
                    if (layer_trace->outputs.size() <= layer) layer_trace->outputs.resize(layer + 1);
                    auto& output = layer_trace->outputs[layer];
                    output.resize(H);
                    auto read = [&](float* dst, int id) {
                        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, row_ptr(id, true), H * sizeof(float),
                                                            cudaMemcpyDeviceToHost, ctx.stream()));
                    };
                    if (is_lm_head && compute_logits) {
                        // The final fused head materializes only logits.
                        std::vector<float> residual(H);
                        read(output.data(), node.inputs[0]);
                        read(residual.data(), node.inputs[1]);
                        ctx.synchronize();
                        for (int j = 0; j < H; ++j) output[j] += residual[j];
                    } else {
                        read(output.data(), value);
                        ctx.synchronize();
                    }
                }
            }
        }
    }
    if constexpr (Observe) {
        if (timer) {
            ctx.synchronize();
            for (int ni = 0; ni < num_nodes; ++ni) {
                const auto& node = g.nodes[ni];
                if (!node.alive || (node.output_logits && !compute_logits && node.residual_out < 0)) continue;
                float ms = 0.0f;
                FUSIONRT_CUDA_CHECK(cudaEventElapsedTime(&ms, timer->event(ni, 0), timer->event(ni, 1)));
                if (ms <= 0) continue;
                const bool residual_only = node.output_logits && !compute_logits;
                OpTiming work;
                if (residual_only) {
                    work.name = node.tag + "_residual";
                    work.kind = op_kind_name(OpKind::ADD_RESIDUAL);
                    work.calls = 1;
                    work.flops = static_cast<int64_t>(M) * node.input_cols;
                    work.bytes = 3 * sizeof(float) * work.flops;
                } else {
                    work = operator_work(node, node.output_logits ? 1 : M, pos_host, ctx, use_fused,
                        !node.inputs.empty() && bufs[node.inputs[0]].dtype == DType::Q8_1,
                        node.out >= 0 && bufs[node.out].dtype == DType::Q8_1,
                        node.op == OpKind::GQA_ATTN_BLOCK || node.op == OpKind::MLA_ATTN_BLOCK
                            ? &dense_layers[node.layer_index] : nullptr,
                        node.op == OpKind::MOE_FFN_BLOCK ? &moe_layers[node.layer_index] : nullptr,
                        node.op == OpKind::GQA_ATTN_BLOCK ? kv_states[dense_layers[node.layer_index].kv_source_layer >= 0
                            ? dense_layers[node.layer_index].kv_source_layer : node.layer_index].dtype : DType::F32);
                }
                work.total_ms = ms;
                auto op = std::find_if(op_trace->ops.begin(), op_trace->ops.end(), [&](const OpTiming& entry) {
                    return entry.name == work.name && entry.kind == work.kind;
                });
                if (op == op_trace->ops.end()) {
                    op_trace->ops.push_back(std::move(work));
                } else {
                    ++op->calls;
                    op->total_ms += ms;
                    op->bytes += work.bytes;
                    op->flops += work.flops;
                }
            }
        }
    }
    if (segment && segment->boundary) {
        const auto& output = bufs[segment->boundary_value];
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(segment->boundary->ptr(), output.ptr(), output.nbytes(),
                                            cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    ctx.scratch_arena().reset(mark);
}

template void execute_ir_graph_cuda<false>(
    const MiniGraph& g, CudaIRWorkspace& ws, CudaContext& ctx,
    const Tensor& x_input, const Tensor& token_ids, Tensor& logits_buf,
    int M, int pos_host, int start_pos,
    const int* pos_dev, const int* kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvView>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    bool use_fused, OpTrace* op_trace, CudaOpTimer* timer,
    LayerTrace* layer_trace, bool compute_logits, bool all_logits,
    const CudaIRSegment* segment, float* nextn_out, GdnReplay* replay);
template void execute_ir_graph_cuda<true>(
    const MiniGraph& g, CudaIRWorkspace& ws, CudaContext& ctx,
    const Tensor& x_input, const Tensor& token_ids, Tensor& logits_buf,
    int M, int pos_host, int start_pos,
    const int* pos_dev, const int* kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvView>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    bool use_fused, OpTrace* op_trace, CudaOpTimer* timer,
    LayerTrace* layer_trace, bool compute_logits, bool all_logits,
    const CudaIRSegment* segment, float* nextn_out, GdnReplay* replay);

}  // namespace fusionrt
#endif
