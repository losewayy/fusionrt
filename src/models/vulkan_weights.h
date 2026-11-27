#pragma once

#include "fusionrt/model_types.h"
#include <unordered_map>
#include "../core/moe_residency.h"

namespace fusionrt {

namespace model_io { struct LoadedModelRaw; }

struct VulkanLoadHooks {
    std::function<void(const DenseGQAConfig&, DType)> allocate_dense;
    std::function<void(const GatedDeltaNetConfig&, int)> allocate_gdn;
    std::function<void(int)> share_dense;
};

struct VulkanDenseWeights {
    DenseGQAConfig cfg;
    int moe_index = -1;
    int sliding_window = 0;
    int kv_source = -1;
    bool gemma4 = false;
    float output_scale = 1.0f;
    Tensor rope_factors, attn_post_norm, ffn_post_norm;
    Tensor ple_gate, ple_projection, ple_post_norm;
    Tensor dense_post_norm, routed_post_norm;
    Tensor mla_q_a, mla_q_a_norm, mla_kv_a, mla_kv_a_norm;
    std::vector<Tensor> mla_k_b, mla_v_b;
    Tensor attn_norm_w;
    Tensor q_proj, k_proj, v_proj, qkv_proj_fused;
    Tensor q_bias, k_bias, v_bias;
    Tensor q_norm_w, k_norm_w, o_proj;
    Tensor mlp_norm_w, gate_up_proj, gate_proj, up_proj, down_proj;
};

struct VulkanGdnWeights {
    GatedDeltaNetConfig cfg;
    int moe_index = -1;
    Tensor attn_norm_w;
    Tensor qkv_proj, z_proj, a_proj, b_proj;
    Tensor conv_w, conv_bias, A_log, dt_bias, norm_w, o_proj;
    Tensor mlp_norm_w, gate_up_proj, gate_proj, up_proj, down_proj;
};

struct VulkanMoeWeights : VkSparseMoELayer {
    Tensor router_bias, router_scale, gate_scale, up_scale, down_scale, gate_region;
    Tensor gpu_up_w, slot_map;
    HostTensor cpu_gate, cpu_up, cpu_down, cpu_gate_scale, cpu_up_scale;
    mutable std::vector<int32_t> expert_to_slot;
    mutable std::optional<MoeDynamicCache> cache;
    mutable CostHardware cache_hardware;
    uint32_t up_offset = 0;
    DType up_dtype = DType::F32;
};

struct VulkanMtpWeights {
    Tensor projection, embedding, embedding_norm, hidden_norm, head, head_norm, trunk_hidden, draft_hidden;
    HostTensor host_embedding;
    float projection_scale = 1.f, head_scale = 1.f;
};

class VulkanModelWeights {
public:
    explicit VulkanModelWeights(const HybridModelConfig& cfg, int gpu_experts = -1, bool mtp = false)
        : cfg_(cfg), gpu_experts_(gpu_experts), load_mtp_(mtp) {}

    void init_synthetic_vulkan(VkContext&, uint64_t seed, const VulkanLoadHooks&);
    bool load_gguf_vulkan(VkContext&, const std::string&, std::string*, int, const VulkanLoadHooks&);
    bool load_safetensors_vulkan(VkContext&, const std::string&, std::string*, int, const VulkanLoadHooks&);

    const HybridModelConfig& config() const { return cfg_; }
    const Tensor& embeddings() const { return tok_embeddings_; }
    HostTensor gather_host_embeddings(std::span<const int32_t> tokens) const;
    const auto& hadamard_signs() const { return hadamard_signs_; }
    int hadamard_width() const { return hadamard_width_; }
    const auto& gemma_inputs() const { return gemma_inputs_; }
    HostTensor gather_host_ple(std::span<const int32_t> tokens) const;
    const Tensor& dummy() const { return dummy_; }
    const Tensor& final_norm() const { return final_norm_; }
    const Tensor& lm_head() const { return lm_head_; }
    const auto& layer_kinds() const { return layer_kinds_; }
    const auto& layer_indices() const { return layer_indices_; }
    const auto& dense_layers() const { return dense_layers_; }
    const auto& gdn_layers() const { return gdn_layers_; }
    const auto& moe_layers() const { return moe_layers_; }
    const auto& mtp() const { return mtp_; }
    HostTensor gather_mtp_embeddings(std::span<const int32_t> tokens) const;

    void upload(VkContext&, const model_io::LoadedModelRaw&, const VulkanLoadHooks&, DType default_kv_type);

private:
    HybridModelConfig cfg_;
    int gpu_experts_;
    bool load_mtp_;
    VulkanMtpWeights mtp_;
    Tensor dummy_;
    std::unordered_map<uint64_t, Tensor> hadamard_signs_;
    int hadamard_width_ = 0;
    GemmaLayerInputWeights<Tensor> gemma_inputs_;
    HostTensor host_ple_;
    Tensor tok_embeddings_, final_norm_, lm_head_;
    HostTensor host_embeddings_;
    std::vector<int> layer_kinds_;
    std::vector<size_t> layer_indices_;
    std::vector<VulkanDenseWeights> dense_layers_;
    std::vector<VulkanGdnWeights> gdn_layers_;
    std::vector<VulkanMoeWeights> moe_layers_;
};

}  // namespace fusionrt
