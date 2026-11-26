#pragma once

#include "fusionrt/tensor.h"
#include "fusionrt/graph_ir.h"
#include "fusionrt/device_profile.h"
#include "fusionrt/sampler.h"
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
#include <string>
#include <vector>

namespace fusionrt {

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
    std::string model_architecture;
};

struct GenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double prefill_tok_per_s = 0.0;
    double decode_tok_per_s = 0.0;
    int prefilled_tokens = 0;
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

struct GenerationOptions {
    SamplingParams sampling;
    bool reuse_prefix = false;
    std::function<bool(int32_t)> on_token;  // Return false to stop.
    OpTrace* op_trace = nullptr;
    LayerTrace* layer_trace = nullptr;
};

struct ModelMemory {
    size_t static_scratch_bytes = 0;
    size_t kv_bytes = 0;
    size_t ssm_bytes = 0;
    size_t persistent_bytes = 0;
    std::map<DType, size_t> weight_bytes;
};

#ifdef FUSIONRT_ENABLE_CUDA
struct CudaDenseGQALayer {
    DenseGQAConfig cfg;
    int moe_index = -1;
    Tensor attn_norm_w;     // [hidden]
    Tensor q_proj;          // [q_dim, hidden]
    Tensor k_proj;          // [kv_dim, hidden]
    Tensor v_proj;          // [kv_dim, hidden]
    std::array<Tensor, 3> projection_packs;
    Tensor q_bias, k_bias, v_bias;
    Tensor q_norm_w;        // [head_dim] (optional)
    Tensor k_norm_w;        // [head_dim] (optional)
    Tensor rope_factors;    // [rotary_dims / 2] (optional)
    Tensor o_proj;          // [hidden, q_dim]
    Tensor mlp_norm_w;      // [hidden]
    Tensor gate_up_proj;    // [2*inter, hidden]
    Tensor gate_proj, up_proj;  // Separate when the gate and up weights use different formats.
    Tensor down_proj;       // [hidden, inter]


};

struct CudaGatedDeltaNetLayer {
    GatedDeltaNetConfig cfg;
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

struct CudaKvState {
    Tensor keys, values;
};

struct CudaGdnState {
    Tensor convolution, recurrence;
};

struct CudaSparseMoELayer {
    MoEConfig cfg;
    Tensor norm_w;                // [hidden]
    Tensor router_w;              // [num_experts, hidden]
    Tensor gpu_gate_up_w;         // [gpu_resident_experts, 2*inter, hidden]
    Tensor gpu_down_w;            // [gpu_resident_experts, hidden, inter]
    HostTensor cpu_gate_up_w;     // [offloaded, 2*inter, hidden]
    HostTensor cpu_down_w;        // [offloaded, hidden, inter]
    Tensor shared_gate_up_w;      // [2*shared_inter, hidden]
    Tensor shared_down_w;         // [hidden, shared_inter]

    void forward_cuda(CudaContext& ctx, Tensor& x_in_out, bool use_fused = true) const;
    void forward_prefill_cuda(CudaContext& ctx, Tensor& x_in_out, int M, bool use_fused = true) const;
};
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
struct VkDenseGQALayer {
    DenseGQAConfig cfg;
    int moe_index = -1;
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
    Tensor norm_w;
    Tensor router_w;
    Tensor gpu_gate_up_w;
    Tensor gpu_down_w;
    Tensor shared_gate_up_w;
    Tensor shared_down_w;
};
#endif

class FusionModel {
public:
    explicit FusionModel(HybridModelConfig cfg);

    const HybridModelConfig& config() const { return cfg_; }

#ifdef FUSIONRT_ENABLE_CUDA
    void init_synthetic(CudaContext& ctx, uint64_t seed = 42);
    bool load_gguf(CudaContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);
    bool load_safetensors(CudaContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);

    void reset_kv_and_state(CudaContext& ctx);
    std::vector<float> forward_token_cuda(CudaContext& ctx, int32_t token_id, int pos,
                                          bool use_fused = true, bool use_graph = false);
    std::vector<float> forward_prefill_cuda(CudaContext& ctx, const std::vector<int32_t>& token_ids,
                                            int start_pos, bool use_fused = true);
    std::vector<float> prefill_cached_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                          bool use_fused = true, int* prefilled_tokens = nullptr,
                                          LayerTrace* trace = nullptr, OpTrace* op_trace = nullptr);
    std::vector<int32_t> generate_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                       int max_new_tokens, bool use_fused = true, bool use_graph = false,
                                       GenerationStats* stats = nullptr, bool keep_state = false, int start_pos = 0,
                                       int32_t stop_eos_id = -1, const GenerationOptions& options = {});
    std::vector<GenerationStats> bench_model_cuda(CudaContext& ctx, int pp, int tg, int reps, bool use_graph,
                                                  OpTrace* trace = nullptr, int decode_prefix = 0);
    void set_device_profile(CudaContext& ctx, const DeviceProfile& profile);
    bool recommended_cuda_graph() const { return recommend_cuda_graph_; }
    ModelMemory inspect_cuda(CudaContext& ctx);
    int prefill_batch_limit(CudaContext& ctx) const;

    const std::vector<CudaDenseGQALayer>& dense_layers() const { return dense_layers_; }
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers() const { return gdn_layers_; }
    const std::vector<CudaSparseMoELayer>& moe_layers() const { return moe_layers_; }
    const MiniGraph& raw_ir_graph() const { return raw_ir_graph_; }
    const MiniGraph& fused_ir_graph() const { return fused_ir_graph_; }
    void rebuild_cuda_ir();
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
    void init_synthetic_vulkan(VkContext& ctx, uint64_t seed = 42);
    bool load_gguf_vulkan(VkContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);
    bool load_safetensors_vulkan(VkContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);

    void reset_kv_and_state_vulkan(VkContext& ctx);
    std::vector<float> forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos);
    std::vector<int32_t> generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                         int max_new_tokens, GenerationStats* stats = nullptr,
                                         bool keep_state = false, int start_pos = 0,
                                         int32_t stop_eos_id = -1);
#endif

private:
    HybridModelConfig cfg_;
    std::vector<int> layer_kinds_;
    std::vector<size_t> layer_indices_;

#ifdef FUSIONRT_ENABLE_CUDA
    Tensor tok_embeddings_;
    Tensor final_norm_w_;
    Tensor lm_head_w_;
    struct CudaSession {
        Tensor hidden, logits, token_ids, position, kv_length;
        std::vector<CudaKvState> kv;
        std::vector<CudaGdnState> gdn;
        std::vector<int32_t> cached_prompt_tokens;
        bool cached_fused = true;
        CudaGraphExec decode_graph;
        bool graph_captured = false;
        bool graph_fused = true;
        bool graph_has_argmax = false;
    } cuda_session_;

    std::vector<CudaDenseGQALayer> dense_layers_;
    std::vector<CudaGatedDeltaNetLayer> gdn_layers_;
    std::vector<CudaSparseMoELayer> moe_layers_;
    MiniGraph raw_ir_graph_;
    MiniGraph fused_ir_graph_;
    bool recommend_cuda_graph_ = false;
    void execute_prefill_chunks(CudaContext& ctx, const int32_t* token_ids, int count, int start_pos,
                                bool use_fused, LayerTrace* layer_trace, OpTrace* op_trace, bool compute_logits = true);
    std::vector<float> prefill_cuda_impl(CudaContext& ctx, const int32_t* token_ids, int count,
                                         int start_pos, bool use_fused, LayerTrace* layer_trace = nullptr,
                                         OpTrace* op_trace = nullptr);

#endif

#ifdef FUSIONRT_ENABLE_VULKAN
    Tensor vk_tok_embeddings_;
    Tensor vk_final_norm_w_;
    Tensor vk_lm_head_w_;
    Tensor vk_hidden_a_;
    Tensor vk_hidden_b_;
    Tensor vk_logits_;
    Tensor vk_tok_id_;
    Tensor vk_dummy_norm_;

    std::vector<VkDenseGQALayer> vk_dense_layers_;
    std::vector<VkGatedDeltaNetLayer> vk_gdn_layers_;
    std::vector<VkSparseMoELayer> vk_moe_layers_;
#endif
};

}  // namespace fusionrt
