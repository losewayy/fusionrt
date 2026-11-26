#include "fusionrt/models.h"
#include <algorithm>

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
                             const std::array<Tensor, Count>& packs, const std::array<std::string, Count>& tags) {
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

int residual(MiniGraph& graph, int input, int update, const std::string& name, const std::string& tag) {
    const int output = graph.new_value(graph.hidden_size, name);
    IRNode node;
    node.op = OpKind::ADD_RESIDUAL;
    node.inputs = {input, update};
    node.out = output;
    node.output_cols = graph.hidden_size;
    node.tag = tag;
    graph.add_node(std::move(node));
    return output;
}

template <typename Layer>
int append_ffn(MiniGraph& graph, int input, const Layer& layer, const std::string& prefix) {
    const int normalized = norm(graph, input, layer.mlp_norm_w, graph.hidden_size,
                                layer.cfg.rms_norm_eps, prefix + "n2", prefix + "mnorm");
    const int width = layer.cfg.intermediate_size;
    const int activated = graph.new_value(width, prefix + "sw");
    IRNode swiglu;
    swiglu.op = OpKind::SWIGLU_INTERLEAVED;
    if (layer.gate_up_proj.data) {
        swiglu.inputs = {linear(graph, normalized, layer.gate_up_proj, prefix + "gu")};
    } else {
        swiglu.inputs = {linear(graph, normalized, layer.gate_proj, prefix + "gate"),
                         linear(graph, normalized, layer.up_proj, prefix + "up")};
    }
    swiglu.out = activated;
    swiglu.output_cols = width;
    swiglu.tag = prefix + "swiglu";
    graph.add_node(std::move(swiglu));
    const int down = linear(graph, activated, layer.down_proj, prefix + "down");
    return residual(graph, input, down, prefix + "h2", prefix + "add2");
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
    const int output = graph.new_value(cfg.num_attention_heads * cfg.head_dim, prefix + "attn");
    IRNode node;
    node.op = OpKind::GQA_ATTN_BLOCK;
    node.inputs = projections(graph, normalized, {&layer.q_proj, &layer.k_proj, &layer.v_proj},
                              layer.projection_packs, {prefix + "qkv", prefix + "k", prefix + "v"});
    node.out = output;
    node.layer_index = index;
    node.attention = {cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                      std::max(2, static_cast<int>(cfg.head_dim * cfg.partial_rotary_factor) & ~1),
                      cfg.max_seq_len, cfg.rope_theta, cfg.gated_q_attn, cfg.rope_neox,
                      layer.q_norm_w.data ? &layer.q_norm_w : nullptr,
                      layer.k_norm_w.data ? &layer.k_norm_w : nullptr,
                      layer.rope_factors.data ? &layer.rope_factors : nullptr};
    node.norm_eps = cfg.rms_norm_eps;
    node.tag = prefix + "gqa";
    graph.add_node(std::move(node));
    const int projected = linear(graph, output, layer.o_proj, prefix + "o");
    const int next = residual(graph, input, projected, prefix + "h1", prefix + "add1");
    return layer.moe_index >= 0 ? append_moe(graph, next, layer.moe_index, prefix)
                                : append_ffn(graph, next, layer, prefix);
}

int append_recurrent(MiniGraph& graph, int input, int index, const CudaGatedDeltaNetLayer& layer,
                     const std::string& prefix) {
    const auto& cfg = layer.cfg;
    const int normalized = norm(graph, input, layer.attn_norm_w, graph.hidden_size, cfg.rms_norm_eps,
                                prefix + "n1", prefix + "anorm");
    const int output = graph.new_value(cfg.num_v_heads * cfg.head_v_dim, prefix + "gdn");
    IRNode node;
    node.op = OpKind::GDN_RECURRENT_BLOCK;
    node.inputs = projections(graph, normalized, {&layer.qkv_proj, &layer.z_proj, &layer.a_proj, &layer.b_proj},
                              layer.projection_packs, {prefix + "qkvzab", prefix + "z", prefix + "a", prefix + "b"});
    node.out = output;
    node.layer_index = index;
    node.recurrent = {cfg.num_k_heads, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.tiled_value_heads};
    node.norm_eps = cfg.rms_norm_eps;
    node.tag = prefix + "gdn";
    graph.add_node(std::move(node));
    const int projected = linear(graph, output, layer.o_proj, prefix + "o");
    const int next = residual(graph, input, projected, prefix + "h1", prefix + "add1");
    return append_ffn(graph, next, layer, prefix);
}

}  // namespace

void FusionModel::rebuild_cuda_ir() {
    MiniGraph graph;
    graph.hidden_size = static_cast<int>(lm_head_w_.shape[1]);
    graph.input_value = graph.new_value(graph.hidden_size, "hidden_in");
    int hidden = graph.input_value;
    for (size_t i = 0; i < layer_kinds_.size(); ++i) {
        const auto index = static_cast<int>(layer_indices_[i]);
        const std::string prefix = "L" + std::to_string(i) + "_";
        if (layer_kinds_[i] == 0) hidden = append_dense(graph, hidden, index, dense_layers_[index], prefix);
        else if (layer_kinds_[i] == 1) hidden = append_recurrent(graph, hidden, index, gdn_layers_[index], prefix);
        else hidden = append_moe(graph, hidden, index, prefix);
    }
    const int normalized = norm(graph, hidden, final_norm_w_, graph.hidden_size, cfg_.dense.rms_norm_eps,
                                "final_norm", "fnorm");
    graph.output_value = linear(graph, normalized, lm_head_w_, "lm_head", nullptr, true);
    graph.stats.raw_nodes = graph.stats.opt_nodes = static_cast<int>(graph.nodes.size());
    raw_ir_graph_ = std::move(graph);
    fused_ir_graph_ = raw_ir_graph_;
    optimize_graph(fused_ir_graph_);
}
#endif

}  // namespace fusionrt
