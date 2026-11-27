#pragma once

#include "fusionrt/tensor.h"
#include "fusionrt/graph_ir.h"
#include "fusionrt/device_profile.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#endif
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusionrt {

struct MlaConfig {
    int query_rank = 0;
    int latent_rank = 0;
    int rope_dims = 0;
    int value_dim = 0;
    int leading_dense_layers = 0;
    float attention_scale = 0.0f;
    float rope_amplitude = 1.0f;

    int latent_width() const { return latent_rank + rope_dims; }
};

struct DenseGQAConfig {
    int hidden_size = 512;
    int num_attention_heads = 8;
    int num_key_value_heads = 2;
    int head_dim = 64;
    int intermediate_size = 1536;
    int max_seq_len = 4096;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    float partial_rotary_factor = 1.0f;  // 1.0 = full RoPE; <1.0 = partial RoPE
    bool qk_norm = true;                 // Qwen3/Gemma3 style Q/K RMSNorm before RoPE
    bool rope_neox = true;               // NeoX split-half RoPE vs interleaved
    bool gated_q_attn = false;           // Qwen3.5 full-attn joint [Q, gate] projection
    MlaConfig mla;
    DType weight_dtype = DType::F32;     // F32, F16, BF16, Q8_0, or Q4_0
};

struct GatedDeltaNetConfig {
    int hidden_size = 512;
    int num_k_heads = 4;
    int num_v_heads = 8;
    int head_k_dim = 64;
    int head_v_dim = 64;
    int conv_kernel_size = 4;
    int intermediate_size = 1536;
    float rms_norm_eps = 1e-6f;
    DType weight_dtype = DType::F32;
    bool tiled_value_heads = false;
};

struct MoEConfig {
    int hidden_size = 512;
    int intermediate_size = 512;
    int num_experts = 8;
    int top_k = 2;
    bool norm_topk_prob = true;
    bool sigmoid_router = false;
    float expert_weights_scale = 1.0f;
    int expert_groups = 1;
    int expert_groups_used = 1;
    int gpu_resident_experts = 8;
    float rms_norm_eps = 1e-6f;
    bool has_shared_expert = true;
    int shared_intermediate_size = 512;
    DType weight_dtype = DType::F32;
};

struct HybridModelConfig {
    std::string arch = "dense";  // "dense", "hybrid", "moe"
    int vocab_size = 512;
    int num_layers = 4;
    DenseGQAConfig dense;
    GatedDeltaNetConfig gdn;
    MoEConfig moe;
    std::vector<int> gpu_experts;  // Per transformer layer, zero where there are no routed experts.
    std::string model_architecture;
    float embed_scale = 1.0f;
    float final_logit_softcapping = 0.0f;
};

struct PromptContextError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

inline PromptContextError prompt_context_error(size_t tokens, int context) {
    return PromptContextError("Prompt (" + std::to_string(tokens) + " tokens) exceeds model context (" +
                                 std::to_string(context) + ")");
}

inline void validate_token_id(int32_t id, size_t position, int vocab_size) {
    if (id < 0 || id >= vocab_size)
        throw std::invalid_argument("Token ID " + std::to_string(id) + " at position " + std::to_string(position) +
                                    " is outside vocabulary size " + std::to_string(vocab_size));
}

inline void validate_token_ids(std::span<const int32_t> ids, size_t start, int vocab_size) {
    for (size_t i = 0; i < ids.size(); ++i) validate_token_id(ids[i], start + i, vocab_size);
}

struct GenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double prefill_tok_per_s = 0.0;
    double decode_tok_per_s = 0.0;
    int prefilled_tokens = 0;
    int spec_proposed = 0;
    int spec_accepted = 0;
    bool used_cuda_graph = false;
    bool cancelled = false;
    bool json_complete = false;
    bool speculative_active = false;
    std::array<int, 7> mtp_proposed_by_depth{}, mtp_accepted_by_depth{};
    std::array<int, 7> mtp_checked_by_depth{};
    int mtp_cutoffs = 0;
    int mtp_verify_graph_steps = 0, mtp_draft_graph_steps = 0;
    int mtp_reused_kv_rows = 0, mtp_rewound_kv_rows = 0;
};

struct OpTiming {
    std::string name;
    std::string kind;
    int64_t calls = 0;
    double total_ms = 0.0;
    int64_t bytes = 0;
    int64_t flops = 0;
};

struct OpTrace {
    std::string mode;
    std::vector<OpTiming> ops;
    void write(const std::string& path, const std::string& model) const;
};

struct LayerTrace {
    std::vector<std::vector<float>> outputs;
    void write(const std::string& path, const std::string& model) const;
};

struct InputEmbedding {
    int start = 0;  // Offset into prompt token IDs.
    Tensor rows;
    bool bidirectional = false;
};

struct ModelMemory {
    size_t static_scratch_bytes = 0;
    size_t kv_bytes = 0;
    int kv_pages = 0;
    size_t kv_reserved_bytes = 0;
    size_t host_kv_bytes = 0;
    int host_kv_pages = 0;
    size_t ssm_bytes = 0;
    size_t persistent_bytes = 0;
    std::map<DType, size_t> weight_bytes;
};

#ifdef FUSIONRT_ENABLE_CUDA
struct CudaDenseGQALayer {
    DenseGQAConfig cfg;
    int moe_index = -1;
    int kv_source_layer = -1;
    int sliding_window = 0;
    float attention_scale = 0.0f;  // 0 selects 1 / sqrt(head_dim).
    float output_scale = 1.0f;
    bool v_norm = false;
    bool geglu = false;
    Tensor attn_post_norm_w, ffn_post_norm_w, dense_post_norm_w, routed_post_norm_w;
    Tensor attn_norm_w;     // [hidden]
    Tensor q_proj;          // [q_dim, hidden]
    Tensor k_proj;          // [kv_dim, hidden]
    Tensor v_proj;          // [kv_dim, hidden]
    std::array<Tensor, 3> projection_packs;
    Tensor q_bias, k_bias, v_bias;
    Tensor mla_q_a, mla_q_a_norm, mla_kv_a, mla_kv_a_norm, mla_k_b, mla_v_b;
    Tensor q_norm_w;        // [head_dim] (optional)
    Tensor k_norm_w;        // [head_dim] (optional)
    Tensor rope_factors;    // [rotary_dims / 2] (optional)
    Tensor o_proj;          // [hidden, q_dim]
    Tensor mlp_norm_w;      // [hidden]
    Tensor gate_up_proj;    // [2*inter, hidden]
    Tensor gate_proj, up_proj;  // Separate when the gate and up weights use different formats.
    Tensor down_proj;       // [hidden, inter]
    Tensor ple_gate, ple_projection, ple_post_norm;


};

struct CudaGatedDeltaNetLayer {
    GatedDeltaNetConfig cfg;
    int moe_index = -1;
    Tensor attn_norm_w;     // [hidden]
    Tensor qkv_proj;        // [conv_dim, hidden]
    Tensor z_proj;          // [nv*dv, hidden]
    Tensor a_proj;          // [nv, hidden]
    Tensor b_proj;          // [nv, hidden]
    std::array<Tensor, 4> projection_packs;
    Tensor conv_w;          // [conv_dim, 4]
    Tensor conv_bias;       // [conv_dim] (optional)
    Tensor A_log;           // [nv]
    Tensor dt_bias;         // [nv]
    Tensor norm_w;          // [dv]
    Tensor o_proj;          // [hidden, nv*dv]
    Tensor mlp_norm_w;      // [hidden]
    Tensor gate_up_proj;    // [2*inter, hidden]
    Tensor gate_proj, up_proj;
    Tensor down_proj;       // [hidden, inter]
};

struct CudaGdnState {
    Tensor convolution, recurrence;
};

struct CudaSparseMoELayer {
    MoEConfig cfg;
    int model_layer_index = -1;
    Tensor norm_w;                // [hidden]
    Tensor router_w;              // [num_experts, hidden]
    Tensor router_bias;           // Optional correction for expert selection.
    Tensor router_scale, shared_gate_w, gate_scale, up_scale, down_scale;
    HostTensor cpu_gate_scale, cpu_up_scale;
    bool geglu = false;
    Tensor gpu_gate_up_w;         // [gpu_resident_experts, (2*)inter, hidden]
    Tensor gpu_up_w;              // Separate up projection for unpacked gate/up weights.
    Tensor gpu_down_w;            // [gpu_resident_experts, hidden, inter]
    Tensor borrowed_gate_up_w, borrowed_up_w, borrowed_down_w;
    HostTensor cpu_gate_up_w;     // [offloaded, (2*)inter, hidden]
    HostTensor cpu_up_w;
    HostTensor cpu_down_w;        // [offloaded, hidden, inter]
    Tensor shared_gate_up_w;      // [2*shared_inter, hidden]
    Tensor shared_down_w;         // [hidden, shared_inter]

    void forward_cuda(CudaContext& ctx, Tensor& x_in_out, bool use_fused = true, int moe_index = -1) const {
        forward_prefill_cuda(ctx, x_in_out, 1, use_fused, moe_index);
    }
    void forward_prefill_cuda(CudaContext& ctx, Tensor& x_in_out, int M, bool use_fused = true,
                              int moe_index = -1, int route_pos = -1) const;
};
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
struct VkDenseGQALayer {
    DenseGQAConfig cfg;
    int moe_index = -1;
    int kv_source_layer = -1;
    int sliding_window = 0;
    float attention_scale = 0.0f;  // 0 selects 1 / sqrt(head_dim).
    float output_scale = 1.0f;
    bool v_norm = false;
    bool geglu = false;
    Tensor attn_post_norm_w, ffn_post_norm_w;
    Tensor attn_norm_w;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor qkv_proj_fused;
    Tensor q_bias, k_bias, v_bias;
    Tensor q_norm_w;
    Tensor k_norm_w;
    Tensor o_proj;
    Tensor mlp_norm_w;
    Tensor gate_up_proj;
    Tensor down_proj;
    Tensor k_cache;
    Tensor v_cache;
};

struct VkGatedDeltaNetLayer {
    GatedDeltaNetConfig cfg;
    int moe_index = -1;
    Tensor attn_norm_w;
    Tensor qkv_proj;
    Tensor z_proj;
    Tensor a_proj;
    Tensor b_proj;
    Tensor conv_w;
    Tensor conv_bias;
    Tensor A_log;
    Tensor dt_bias;
    Tensor norm_w;
    Tensor o_proj;
    Tensor mlp_norm_w;
    Tensor gate_up_proj;
    Tensor down_proj;
    Tensor conv_state;
    Tensor ssm_state;
};

struct VkSparseMoELayer {
    MoEConfig cfg;
    Tensor shared_gate_w;
    Tensor norm_w;
    Tensor router_w;
    Tensor gpu_gate_up_w;
    Tensor gpu_down_w;
    Tensor shared_gate_up_w;
    Tensor shared_down_w;
};
#endif

}  // namespace fusionrt
