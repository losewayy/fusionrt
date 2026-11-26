#pragma once

#include "fusionrt/tensor.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace fusionrt {

enum class OpKind : uint8_t {
    RMSNORM = 0,
    LINEAR = 1,
    FUSED_RMSNORM_LINEAR = 2,
    ADD_RESIDUAL = 3,
    FUSED_ADD_RMSNORM_LINEAR = 4,
    GQA_ATTN_BLOCK = 5,
    GDN_RECURRENT_BLOCK = 6,
    SWIGLU_INTERLEAVED = 7,
    MOE_FFN_BLOCK = 8,
    SWIGLU_Q8_LINEAR = 9,
    FUSED_ADD_RMSNORM = 10,
    SHARED_Q8_PROJECTIONS = 11,
    GEGLU_INTERLEAVED = 12,
    GEGLU_Q8_LINEAR = 13,
};

inline const char* op_kind_name(OpKind op) {
    switch (op) {
        case OpKind::RMSNORM:                  return "RMSNORM";
        case OpKind::LINEAR:                   return "LINEAR";
        case OpKind::FUSED_RMSNORM_LINEAR:     return "FUSED_RMSNORM_LINEAR";
        case OpKind::ADD_RESIDUAL:             return "ADD_RESIDUAL";
        case OpKind::FUSED_ADD_RMSNORM_LINEAR: return "FUSED_ADD_RMSNORM_LINEAR";
        case OpKind::GQA_ATTN_BLOCK:           return "GQA_ATTN_BLOCK";
        case OpKind::GDN_RECURRENT_BLOCK:      return "GDN_RECURRENT_BLOCK";
        case OpKind::SWIGLU_INTERLEAVED:       return "SWIGLU_INTERLEAVED";
        case OpKind::FUSED_ADD_RMSNORM:        return "FUSED_ADD_RMSNORM";
        case OpKind::SHARED_Q8_PROJECTIONS:    return "SHARED_Q8_PROJECTIONS";
        case OpKind::SWIGLU_Q8_LINEAR:         return "SWIGLU_Q8_LINEAR";
        case OpKind::GEGLU_INTERLEAVED:        return "GEGLU_INTERLEAVED";
        case OpKind::GEGLU_Q8_LINEAR:          return "GEGLU_Q8_LINEAR";
        case OpKind::MOE_FFN_BLOCK:            return "MOE_FFN_BLOCK";
    }
    return "UNKNOWN";
}

struct ValueMeta {
    int id = -1;
    int cols = 0;
    std::string name;
};

struct AttentionParams {
    int query_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int rotary_dims = 0;
    int max_context = 0;
    float rope_theta = 0.0f;
    bool gated = false;
    bool rope_neox = false;
    const Tensor* query_norm = nullptr;
    const Tensor* key_norm = nullptr;
    const Tensor* rotary_factors = nullptr;
};

struct RecurrentParams {
    int key_heads = 0;
    int value_heads = 0;
    int key_dim = 0;
    int value_dim = 0;
    bool tiled_value_heads = false;
};

struct IRNode {
    int id = -1;
    OpKind op = OpKind::LINEAR;
    bool alive = true;
    std::vector<int> inputs;
    int out = -1;
    int residual_out = -1;
    int output_cols = 0;
    int input_cols = 0;
    float norm_eps = 1e-6f;
    const Tensor* weight = nullptr;
    const Tensor* norm_weight = nullptr;
    bool share_quantized_activation = false;
    bool output_logits = false;
    float output_scale = 1.0f;

    // Attention and recurrent nodes refer to session state by layer index.
    int layer_index = -1;
    AttentionParams attention;
    RecurrentParams recurrent;
    std::array<int, 4> input_offsets{};

    const Tensor* packed_weight = nullptr;
    const Tensor* secondary_weight = nullptr;
    int secondary_out = -1;
    std::string tag;
};

struct FusionPassStats {
    int raw_nodes = 0;
    int opt_nodes = 0;
    int sibling_merges = 0;
    int norm_linear_fusions = 0;
    int add_norm_linear_fusions = 0;
};

struct MiniGraph {
    int input_value = -1;
    int output_value = -1;
    int hidden_size = 0;
    float final_logit_softcapping = 0.0f;
    int decode_split_kv_heads = 0;
    std::vector<ValueMeta> values;
    std::vector<IRNode> nodes;
    FusionPassStats stats;

    int new_value(int cols, std::string name = "");
    int add_node(IRNode node);
    void compact_dead_nodes();
    std::string summary() const;
};

inline constexpr bool supports_q8_1_decode(DType dtype) {
    return dtype == DType::Q4_A || dtype == DType::Q8_A || dtype == DType::Q6_K ||
           dtype == DType::Q4_K || dtype == DType::Q5_K ||
           dtype == DType::Q4_KA || dtype == DType::Q6_A;
}

// Compiler optimization passes:
// 1. Horizontal merge of sibling LINEAR ops sharing a normalized activation and packed weight
// 2. Vertical fusion of RMSNORM -> LINEAR into FUSED_RMSNORM_LINEAR
// 3. Triple fusion of ADD_RESIDUAL -> FUSED_RMSNORM_LINEAR into FUSED_ADD_RMSNORM_LINEAR
// 4. Fuse a single-use SwiGLU with its quantized down projection
void optimize_graph(MiniGraph& graph);

}  // namespace fusionrt
