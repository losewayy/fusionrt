#pragma once

#include "fusionrt/models.h"
#include <random>

namespace fusionrt::model_io {

struct LoadedLayerRaw {
    int kind = 0;  // 0=dense, 1=gdn, 2=moe
    bool gated_q_attn = false;
    int kv_source_layer = -1;
    int sliding_window = 0;
    int query_heads = 0, kv_heads = 0, head_dim = 0, rotary_dims = 0;
    float rope_theta = 0.0f;
    float output_scale = 1.0f;
    HostTensor attn_post_norm, ffn_post_norm;
    HostTensor ple_gate, ple_projection, ple_post_norm;
    // Dense
    HostTensor attn_norm, q_proj, k_proj, v_proj, qkv_proj, q_norm, k_norm, o_proj, mlp_norm, gate_up, gate, up, down;
    HostTensor q_bias, k_bias, v_bias;
    HostTensor mla_q_a, mla_q_a_norm, mla_kv_a, mla_kv_a_norm, mla_kv_b, mla_k_b, mla_v_b;
    // GDN extras
    HostTensor z_proj, a_proj, b_proj, conv_w, conv_bias, A_log, dt_bias, gdn_norm_w;
    // MoE extras
    HostTensor router_w, experts_gate_up, experts_up, experts_down, shared_gate_up, shared_down;
    HostTensor router_bias;
    bool has_shared = false;
};

struct LoadedModelRaw {
    HybridModelConfig cfg;
    HostTensor tok_embeddings;
    HostTensor final_norm_w;
    HostTensor lm_head_w;
    HostTensor rope_factors;
    GemmaLayerInputWeights<HostTensor> gemma_inputs;
    bool hadamard_folded = false;
    std::vector<LoadedLayerRaw> layers;
};


HostTensor make_rand_f32(const std::vector<int64_t>& shape, std::mt19937_64& rng, float scale = 0.05f);
HostTensor make_ones_f32(const std::vector<int64_t>& shape);
HostTensor make_zeros_f32(const std::vector<int64_t>& shape);
HostTensor quantize_host_matrix(const HostTensor&, DType);
inline bool same_weight_basis(const HostTensor& a, const HostTensor& b) {
    return a.desc.hadamard_block == b.desc.hadamard_block && a.hadamard_signs == b.hadamard_signs;
}
HostTensor concat_rows_same_dtype(const std::vector<HostTensor>& parts);
HostTensor to_f32_host(const HostTensor&);
bool parse_gguf_model_raw(const std::string&, int max_seq_override, LoadedModelRaw&, std::string* err);
bool parse_safetensors_model_raw(const std::string&, int max_seq_override, LoadedModelRaw&, std::string* err);
#ifdef FUSIONRT_ENABLE_CUDA
void vision_linear_cuda(CudaContext& ctx, const Tensor& input, const Tensor& weight, Tensor& output, int rows);
#endif

}  // namespace fusionrt::model_io
