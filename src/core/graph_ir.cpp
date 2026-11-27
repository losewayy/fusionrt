#include "fusionrt/graph_ir.h"
#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace fusionrt {

int MiniGraph::new_value(int cols, std::string name) {
    int id = static_cast<int>(values.size());
    values.push_back(ValueMeta{id, cols, std::move(name)});
    return id;
}

int MiniGraph::add_node(IRNode node) {
    node.id = static_cast<int>(nodes.size());
    node.alive = true;
    nodes.push_back(std::move(node));
    return nodes.back().id;
}

void MiniGraph::compact_dead_nodes() {
    std::vector<IRNode> kept;
    kept.reserve(nodes.size());
    for (auto& n : nodes) {
        if (n.alive) {
            n.id = static_cast<int>(kept.size());
            kept.push_back(std::move(n));
        }
    }
    nodes = std::move(kept);
}

std::string MiniGraph::summary() const {
    std::ostringstream oss;
    oss << "MiniGraph(raw=" << stats.raw_nodes
        << " -> opt=" << stats.opt_nodes
        << ", sibling_merges=" << stats.sibling_merges
        << ", norm_linear=" << stats.norm_linear_fusions
        << ", add_norm_linear=" << stats.add_norm_linear_fusions << ")";
    return oss.str();
}

namespace {

std::vector<int> compute_use_counts(const MiniGraph& g) {
    std::vector<int> uses(g.values.size(), 0);
    for (const auto& n : g.nodes) {
        if (!n.alive) continue;
        for (int vid : n.inputs) {
            if (vid >= 0 && vid < static_cast<int>(uses.size())) uses[vid]++;
        }
    }
    if (g.output_value >= 0 && g.output_value < static_cast<int>(uses.size())) {
        uses[g.output_value]++;
    }
    if (g.nextn_hidden >= 0) uses[g.nextn_hidden]++;
    return uses;
}

void replace_value_uses_after(MiniGraph& g, size_t start_idx, int old_val, int new_val) {
    for (size_t i = start_idx; i < g.nodes.size(); ++i) {
        if (!g.nodes[i].alive) continue;
        for (int& v : g.nodes[i].inputs) {
            if (v == old_val) v = new_val;
        }
    }
    if (g.output_value == old_val) g.output_value = new_val;
}

}  // namespace

void optimize_graph(MiniGraph& g) {
    int alive0 = 0;
    for (const auto& n : g.nodes) {
        if (n.alive) ++alive0;
    }
    g.stats.raw_nodes = alive0;

    // Merge contiguous projections whose consumers can read packed row slices.
    const auto initial_uses = compute_use_counts(g);
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        auto& first = g.nodes[i];
        if (!first.alive || !first.packed_weight) continue;
        size_t end = i + 1;
        while (end < g.nodes.size() && g.nodes[end].packed_weight == first.packed_weight &&
               g.nodes[end].inputs == first.inputs) ++end;
        if (end == i + 1) continue;

        // A different packed group can precede the attention/recurrent node.
        size_t consumer_index = end;
        while (consumer_index < g.nodes.size() && g.nodes[consumer_index].op == OpKind::LINEAR) ++consumer_index;
        if (consumer_index == g.nodes.size()) continue;
        auto& block = g.nodes[consumer_index];
        if (block.op != OpKind::GQA_ATTN_BLOCK && block.op != OpKind::GDN_RECURRENT_BLOCK) continue;

        bool valid = true;
        for (size_t j = i; j < end; ++j) {
            const int value = g.nodes[j].out;
            valid &= initial_uses[value] == 1 &&
                     std::find(block.inputs.begin(), block.inputs.end(), value) != block.inputs.end();
        }
        if (!valid) continue;
        const int cols = static_cast<int>(first.packed_weight->shape[0]);
        const int packed = g.new_value(cols, first.tag + "_packed");
        int offset = 0;
        for (size_t j = i; j < end; ++j) {
            auto& node = g.nodes[j];
            const auto slot = std::find(block.inputs.begin(), block.inputs.end(), node.out) - block.inputs.begin();
            block.inputs[slot] = packed;
            block.input_offsets[slot] = offset;
            offset += node.output_cols;
            if (j != i) node.alive = false;
        }
        first.weight = first.packed_weight;
        first.output_cols = cols;
        first.out = packed;
        first.tag += "_merged";
        ++g.stats.sibling_merges;
    }

    // Pass 2: Fold RMSNORM -> LINEAR into FUSED_RMSNORM_LINEAR
    {
        auto uses = compute_use_counts(g);
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            IRNode& norm = g.nodes[i];
            if (!norm.alive || norm.op != OpKind::RMSNORM) continue;
            int norm_out = norm.out;
            if (norm_out < 0 || uses[norm_out] != 1) continue;
            for (size_t j = i + 1; j < g.nodes.size(); ++j) {
                IRNode& lin = g.nodes[j];
                if (!lin.alive || lin.op != OpKind::LINEAR) continue;
                if (lin.inputs.size() == 1 && lin.inputs[0] == norm_out) {
                    lin.op = OpKind::FUSED_RMSNORM_LINEAR;
                    lin.inputs[0] = norm.inputs[0];
                    lin.norm_weight = norm.norm_weight;
                    lin.norm_eps = norm.norm_eps;
                    norm.alive = false;
                    g.stats.norm_linear_fusions++;
                    break;
                }
            }
        }
    }

    for (auto& norm : g.nodes) {
        if (!norm.alive || norm.op != OpKind::RMSNORM || norm.out == g.nextn_hidden) continue;
        int consumers = 0;
        bool supported = true;
        for (const auto& node : g.nodes) {
            if (!node.alive || std::find(node.inputs.begin(), node.inputs.end(), norm.out) == node.inputs.end()) continue;
            ++consumers;
            supported &= node.op == OpKind::LINEAR && node.weight && supports_q8_1_decode(node.weight->dtype);
        }
        norm.share_quantized_activation = supported && consumers > 1;
    }

    // Pass 3: Fold ADD_RESIDUAL(a, b -> v_res) + FUSED_RMSNORM_LINEAR(v_res -> y)
    // into FUSED_ADD_RMSNORM_LINEAR(a, b -> y, aux_out = v_res_upd) when no other reader precedes FUSED_RMSNORM_LINEAR
    {
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            IRNode& add = g.nodes[i];
            if (!add.alive || add.op != OpKind::ADD_RESIDUAL || add.output_scale != 1.0f) continue;
            int v_res = add.out;
            int first_use_idx = -1;
            for (size_t j = i + 1; j < g.nodes.size(); ++j) {
                const IRNode& c = g.nodes[j];
                if (!c.alive) continue;
                bool reads_v = std::find(c.inputs.begin(), c.inputs.end(), v_res) != c.inputs.end();
                if (reads_v) {
                    first_use_idx = static_cast<int>(j);
                    break;
                }
            }
            if (first_use_idx < 0) continue;
            IRNode& fus = g.nodes[static_cast<size_t>(first_use_idx)];
            if ((fus.op != OpKind::FUSED_RMSNORM_LINEAR && !fus.share_quantized_activation) ||
                fus.inputs.size() != 1 || fus.inputs[0] != v_res) {
                continue;
            }
            // Ensure add inputs are distinct from updated residual target
            int v_res_upd = g.new_value(g.values[static_cast<size_t>(v_res)].cols, g.values[static_cast<size_t>(v_res)].name + "_upd");
            fus.op = fus.share_quantized_activation ? OpKind::FUSED_ADD_RMSNORM : OpKind::FUSED_ADD_RMSNORM_LINEAR;
            fus.inputs = add.inputs;  // {res_a, res_b}
            fus.residual_out = v_res_upd;
            add.alive = false;
            replace_value_uses_after(g, static_cast<size_t>(first_use_idx) + 1, v_res, v_res_upd);
            g.stats.add_norm_linear_fusions++;
        }
    }

    auto uses = compute_use_counts(g);
    for (size_t i = 1; i < g.nodes.size(); ++i) {
        IRNode& act = g.nodes[i - 1];
        IRNode& lin = g.nodes[i];
        const bool geglu = act.op == OpKind::GEGLU_INTERLEAVED;
        if (!act.alive || (act.op != OpKind::SWIGLU_INTERLEAVED && !geglu) || act.inputs.size() != 1 || uses[act.out] != 1 ||
            !lin.alive || lin.op != OpKind::LINEAR || lin.inputs.size() != 1 ||
            lin.inputs[0] != act.out || lin.input_cols != act.output_cols || !lin.weight ||
            !supports_q8_1_decode(lin.weight->dtype)) continue;
        lin.op = geglu ? OpKind::GEGLU_Q8_LINEAR : OpKind::SWIGLU_Q8_LINEAR;
        lin.inputs[0] = act.inputs[0];
        act.alive = false;
    }

    g.compact_dead_nodes();
    std::vector<bool> quantized(g.values.size());
    for (const auto& node : g.nodes) {
        if (node.alive && node.share_quantized_activation) quantized[node.out] = true;
    }
    for (size_t i = 1; i < g.nodes.size(); ++i) {
        auto& a = g.nodes[i - 1];
        auto& b = g.nodes[i];
        if (!a.alive || !b.alive || a.op != OpKind::LINEAR || b.op != OpKind::LINEAR ||
            a.inputs != b.inputs || !quantized[a.inputs[0]] ||
            a.weight->dtype != DType::Q4_A || b.weight->dtype != DType::Q8_A ||
            a.output_cols < 4096 || b.output_cols > 64) continue;
        a.op = OpKind::SHARED_Q8_PROJECTIONS;
        a.secondary_weight = b.weight;
        a.secondary_out = b.out;
        b.alive = false;
    }

    g.compact_dead_nodes();
    g.stats.opt_nodes = static_cast<int>(g.nodes.size());
}

}  // namespace fusionrt
