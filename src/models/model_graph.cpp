#include "fusionrt/models.h"
#include <algorithm>
#include <span>

namespace fusionrt {

FusionModel::FusionModel(HybridModelConfig cfg) : cfg_(std::move(cfg)) {}

#ifdef FUSIONRT_ENABLE_CUDA
namespace {

int norm(MiniGraph& graph, int input, const Tensor& weight, int width, float eps,
         const std::string& name, const std::string& tag) {
    const int output = graph.new_value(width, name);
    IRNode node;
    node.op = OpKind::RMSNORM;
    node.inputs = {input};
    node.out = output;
    node.input_cols = node.output_cols = width;
    node.norm_weight = &weight;
    node.norm_eps = eps;
    node.tag = tag;
    graph.add_node(std::move(node));
    return output;
}

int linear(MiniGraph& graph, int input, const Tensor& weight, const std::string& tag,
           const Tensor* packed_weight = nullptr, bool output_logits = false) {
    const int output = graph.new_value(static_cast<int>(weight.shape[0]), tag);
    IRNode node;
    node.inputs = {input};
    node.out = output;
    node.input_cols = static_cast<int>(weight.shape[1]);
    node.output_cols = static_cast<int>(weight.shape[0]);
    node.weight = &weight;
    node.packed_weight = packed_weight;
    node.output_logits = output_logits;
    node.tag = tag;
    graph.add_node(std::move(node));
    return output;
}

template <size_t Count>
std::vector<int> projections(MiniGraph& graph, int input, const std::array<const Tensor*, Count>& weights,
                             std::span<const Tensor> packs, const std::array<std::string, Count>& tags) {
    std::vector<int> values;
    const Tensor* pack = nullptr;
    int remaining = 0;
    for (size_t i = 0; i < Count; ++i) {
        if (remaining == 0) {
            pack = packs[i].data ? &packs[i] : nullptr;
            remaining = pack ? static_cast<int>(pack->shape[0]) : static_cast<int>(weights[i]->shape[0]);
        }
        values.push_back(linear(graph, input, *weights[i], tags[i], pack));
        remaining -= static_cast<int>(weights[i]->shape[0]);
    }
    return values;
}

int residual(MiniGraph& graph, int input, int update, const std::string& name, const std::string& tag,
             float scale = 1.0f) {
    const int output = graph.new_value(graph.hidden_size, name);
    IRNode node;
    node.op = OpKind::ADD_RESIDUAL;
    node.output_scale = scale;
    node.inputs = {input, update};
    node.out = output;
    node.output_cols = graph.hidden_size;
    node.tag = tag;
    graph.add_node(std::move(node));
    return output;
}

template <typename Layer>
int append_ffn(MiniGraph& graph, int input, const Layer& layer, const std::string& prefix,
               const Tensor* post_norm = nullptr, bool geglu = false, float scale = 1.0f,
               bool add_residual = true) {
    const int normalized = norm(graph, input, layer.mlp_norm_w, graph.hidden_size,
                                layer.cfg.rms_norm_eps, prefix + "n2", prefix + "mnorm");
    const int width = layer.cfg.intermediate_size;
    const int activated = graph.new_value(width, prefix + "sw");
    IRNode swiglu;
    swiglu.op = geglu ? OpKind::GEGLU_INTERLEAVED : OpKind::SWIGLU_INTERLEAVED;
    if (layer.gate_up_proj.data) {
        swiglu.inputs = {linear(graph, normalized, layer.gate_up_proj, prefix + "gu")};
    } else {
        swiglu.inputs = {linear(graph, normalized, layer.gate_proj, prefix + "gate"),
                         linear(graph, normalized, layer.up_proj, prefix + "up")};
    }
    swiglu.out = activated;
    swiglu.output_cols = width;
    swiglu.tag = prefix + (geglu ? "geglu" : "swiglu");
    graph.add_node(std::move(swiglu));
    int down = linear(graph, activated, layer.down_proj, prefix + "down");
    if (post_norm && post_norm->data) {
        down = norm(graph, down, *post_norm, graph.hidden_size, layer.cfg.rms_norm_eps,
                    prefix + "ffn_post_norm", prefix + "ffn_post_norm");
    }
    return add_residual ? residual(graph, input, down, prefix + "h2", prefix + "add2", scale) : down;
}

int append_moe(MiniGraph& graph, int input, int index, const std::string& prefix) {
    const int output = graph.new_value(graph.hidden_size, prefix + "moe_out");
    IRNode node;
    node.op = OpKind::MOE_FFN_BLOCK;
    node.inputs = {input};
    node.out = output;
    node.layer_index = index;
    node.output_cols = graph.hidden_size;
    node.tag = prefix + "moe";
    graph.add_node(std::move(node));
    return output;
}

int append_dense(MiniGraph& graph, int input, int index, const CudaDenseGQALayer& layer,
                 const std::string& prefix) {
    const auto& cfg = layer.cfg;
    const int normalized = norm(graph, input, layer.attn_norm_w, graph.hidden_size, cfg.rms_norm_eps,
                                prefix + "n1", prefix + "anorm");
    const bool mla = cfg.mla.latent_rank > 0;
    const int output_cols = cfg.num_attention_heads * (mla ? cfg.mla.value_dim : cfg.head_dim);
    const int output = graph.new_value(output_cols, prefix + "attn");
    IRNode node;
    node.op = mla ? OpKind::MLA_ATTN_BLOCK : OpKind::GQA_ATTN_BLOCK;
    if (mla) {
        int query_input = normalized;
        if (layer.mla_q_a.data) {
            const int low_rank = linear(graph, normalized, layer.mla_q_a, prefix + "q_a");
            query_input = norm(graph, low_rank, layer.mla_q_a_norm, cfg.mla.query_rank,
                               cfg.rms_norm_eps, prefix + "q_a_norm", prefix + "q_a_norm");
        }
        node.inputs = {linear(graph, query_input, layer.q_proj, prefix + "q"),
                       linear(graph, normalized, layer.mla_kv_a, prefix + "kv_a")};
        node.output_cols = output_cols;
    } else if (layer.kv_source_layer >= 0) {
        node.inputs = {linear(graph, normalized, layer.q_proj, prefix + "q")};
    } else if (!layer.v_proj.data) {
        node.inputs = projections<2>(graph, normalized, {&layer.q_proj, &layer.k_proj}, layer.projection_packs,
                                  {prefix + "qk", prefix + "k"});
    } else {
        node.inputs = projections<3>(graph, normalized, {&layer.q_proj, &layer.k_proj, &layer.v_proj},
                                  layer.projection_packs, {prefix + "qkv", prefix + "k", prefix + "v"});
    }
    node.out = output;
    node.layer_index = index;
    node.attention = {cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                      std::max(2, static_cast<int>(cfg.head_dim * cfg.partial_rotary_factor) & ~1),
                      cfg.max_seq_len, cfg.rope_theta, cfg.gated_q_attn, cfg.rope_neox,
                      layer.q_norm_w.data ? &layer.q_norm_w : nullptr,
                      layer.k_norm_w.data ? &layer.k_norm_w : nullptr,
                      layer.rope_factors.data ? &layer.rope_factors : nullptr};
    node.norm_eps = cfg.rms_norm_eps;
    node.tag = prefix + (mla ? "mla" : "gqa");
    graph.add_node(std::move(node));
    int projected = linear(graph, output, layer.o_proj, prefix + "o");
    if (layer.attn_post_norm_w.data) {
        projected = norm(graph, projected, layer.attn_post_norm_w, graph.hidden_size, cfg.rms_norm_eps,
                         prefix + "attn_post_norm", prefix + "attn_post_norm");
    }
    const int next = residual(graph, input, projected, prefix + "h1", prefix + "add1");
    if (layer.moe_index >= 0 && !layer.dense_post_norm_w.data) return append_moe(graph, next, layer.moe_index, prefix);
    int ffn;
    if (layer.dense_post_norm_w.data) {
        const int shared = append_ffn(graph, next, layer, prefix, &layer.dense_post_norm_w, true, 1.f, false);
        const int routed = append_moe(graph, next, layer.moe_index, prefix);
        const int routed_norm = norm(graph, routed, layer.routed_post_norm_w, graph.hidden_size, cfg.rms_norm_eps,
                                     prefix + "routed_norm", prefix + "routed_norm");
        const int combined = residual(graph, shared, routed_norm, prefix + "ffn_sum", prefix + "ffn_sum");
        const int post = norm(graph, combined, layer.ffn_post_norm_w, graph.hidden_size, cfg.rms_norm_eps,
                              prefix + "ffn_post_norm", prefix + "ffn_post_norm");
        ffn = residual(graph, next, post, prefix + "h2", prefix + "add2",
                       layer.ple_gate.data ? 1.f : layer.output_scale);
    } else {
        if (!layer.ple_gate.data)
            return append_ffn(graph, next, layer, prefix, &layer.ffn_post_norm_w, layer.geglu, layer.output_scale);
        ffn = append_ffn(graph, next, layer, prefix, &layer.ffn_post_norm_w, layer.geglu);
    }
    if (!layer.ple_gate.data) return ffn;
    const int gate = linear(graph, ffn, layer.ple_gate, prefix + "ple_gate");
    IRNode gated;
    gated.op = OpKind::GEMMA_PLE_GATE;
    gated.inputs = {gate};
    gated.output_cols = static_cast<int>(layer.ple_gate.shape[0]);
    gated.out = graph.new_value(gated.output_cols, prefix + "ple_gate_out");
    gated.layer_index = index;
    gated.tag = prefix + "ple_gelu_mul";
    const int activated = gated.out;
    graph.add_node(std::move(gated));
    int update = linear(graph, activated, layer.ple_projection, prefix + "ple_proj");
    update = norm(graph, update, layer.ple_post_norm, graph.hidden_size, cfg.rms_norm_eps,
                  prefix + "ple_post_norm", prefix + "ple_post_norm");
    return residual(graph, ffn, update, prefix + "h2_ple", prefix + "ple_add", layer.output_scale);
}

int append_recurrent(MiniGraph& graph, int input, int index, const CudaGatedDeltaNetLayer& layer,
                     const std::string& prefix) {
    const auto& cfg = layer.cfg;
    const int normalized = norm(graph, input, layer.attn_norm_w, graph.hidden_size, cfg.rms_norm_eps,
                                prefix + "n1", prefix + "anorm");
    const int output = graph.new_value(cfg.num_v_heads * cfg.head_v_dim, prefix + "gdn");
    IRNode node;
    node.op = OpKind::GDN_RECURRENT_BLOCK;
    node.inputs = projections<4>(graph, normalized, {&layer.qkv_proj, &layer.z_proj, &layer.a_proj, &layer.b_proj},
                              layer.projection_packs, {prefix + "qkvzab", prefix + "z", prefix + "a", prefix + "b"});
    node.out = output;
    node.layer_index = index;
    node.recurrent = {cfg.num_k_heads, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.tiled_value_heads};
    node.norm_eps = cfg.rms_norm_eps;
    node.tag = prefix + "gdn";
    graph.add_node(std::move(node));
    const int projected = linear(graph, output, layer.o_proj, prefix + "o");
    const int next = residual(graph, input, projected, prefix + "h1", prefix + "add1");
    if (layer.moe_index >= 0) return append_moe(graph, next, layer.moe_index, prefix);
    return append_ffn(graph, next, layer, prefix);
}

}  // namespace

void FusionModel::rebuild_cuda_ir() {
    MiniGraph graph;
    graph.hidden_size = static_cast<int>(lm_head_w_.shape[1]);
    graph.final_logit_softcapping = cfg_.final_logit_softcapping;
    if (cfg_.model_architecture == "gemma4") graph.decode_split_kv_heads = cfg_.dense.num_key_value_heads;
    if (gemma_inputs_.token_embeddings.data) {
        graph.gemma_inputs = {&gemma_inputs_.token_embeddings, &gemma_inputs_.projection, &gemma_inputs_.norm};
        graph.gemma_input_eps = cfg_.dense.rms_norm_eps;
    }
    graph.input_value = graph.new_value(graph.hidden_size, "hidden_in");
    int hidden = graph.input_value;
    for (int i = 0; i < cfg_.num_layers; ++i) {
        const auto index = static_cast<int>(layer_indices_[i]);
        const std::string prefix = "L" + std::to_string(i) + "_";
        if (layer_kinds_[i] == 0) hidden = append_dense(graph, hidden, index, dense_layers_[index], prefix);
        else if (layer_kinds_[i] == 1) hidden = append_recurrent(graph, hidden, index, gdn_layers_[index], prefix);
        else hidden = append_moe(graph, hidden, index, prefix);
    }
    const int normalized = norm(graph, hidden, final_norm_w_, graph.hidden_size, cfg_.dense.rms_norm_eps,
                                "final_norm", "fnorm");
    graph.output_value = linear(graph, normalized, lm_head_w_, "lm_head", nullptr, true);
    if (has_native_mtp()) {
        graph.nextn_hidden = normalized;
        MiniGraph draft;
        draft.hidden_size = graph.hidden_size;
        draft.input_value = draft.new_value(draft.hidden_size, "nextn_input");
        const int layer = static_cast<int>(layer_indices_.back());
        const int next = append_dense(draft, draft.input_value, layer, dense_layers_[layer], "nextn_");
        draft.nextn_hidden = norm(draft, next, mtp_.head_norm, draft.hidden_size,
                                  cfg_.dense.rms_norm_eps, "nextn_hidden", "nextn_norm");
        draft.output_value = linear(draft, draft.nextn_hidden, mtp_.head, "nextn_head", nullptr, true);
        optimize_graph(draft);
        mtp_.graph = std::move(draft);
    }
    graph.stats.raw_nodes = graph.stats.opt_nodes = static_cast<int>(graph.nodes.size());
    raw_ir_graph_ = std::move(graph);
    fused_ir_graph_ = raw_ir_graph_;
    optimize_graph(fused_ir_graph_);
}
#endif

}  // namespace fusionrt
