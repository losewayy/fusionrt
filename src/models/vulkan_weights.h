#pragma once

#include "fusionrt/model_types.h"

namespace fusionrt {

namespace model_io { struct LoadedModelRaw; }

// Keep the existing arena allocation order while each subsystem owns its own views.
struct VulkanLoadHooks {
    std::function<void(int, int)> allocate_workspace;
    std::function<void(const DenseGQAConfig&)> allocate_dense;
    std::function<void(const GatedDeltaNetConfig&, int)> allocate_gdn;
};

struct VulkanDenseWeights {
    DenseGQAConfig cfg;
    int moe_index = -1;
    Tensor attn_norm_w;
    Tensor q_proj, k_proj, v_proj, qkv_proj_fused;
    Tensor q_bias, k_bias, v_bias;
    Tensor q_norm_w, k_norm_w, o_proj;
    Tensor mlp_norm_w, gate_up_proj, down_proj;
};

struct VulkanGdnWeights {
    GatedDeltaNetConfig cfg;
    int moe_index = -1;
    Tensor attn_norm_w;
    Tensor qkv_proj, z_proj, a_proj, b_proj;
    Tensor conv_w, conv_bias, A_log, dt_bias, norm_w, o_proj;
    Tensor mlp_norm_w, gate_up_proj, down_proj;
};

class VulkanModelWeights {
public:
    explicit VulkanModelWeights(const HybridModelConfig& cfg) : cfg_(cfg) {}

    void init_synthetic_vulkan(VkContext&, uint64_t seed, const VulkanLoadHooks&);
    bool load_gguf_vulkan(VkContext&, const std::string&, std::string*, int, const VulkanLoadHooks&);
    bool load_safetensors_vulkan(VkContext&, const std::string&, std::string*, int, const VulkanLoadHooks&);

    const HybridModelConfig& config() const { return cfg_; }
    const Tensor& embeddings() const { return tok_embeddings_; }
    const Tensor& final_norm() const { return final_norm_; }
    const Tensor& lm_head() const { return lm_head_; }
    const auto& layer_kinds() const { return layer_kinds_; }
    const auto& layer_indices() const { return layer_indices_; }
    const auto& dense_layers() const { return dense_layers_; }
    const auto& gdn_layers() const { return gdn_layers_; }
    const auto& moe_layers() const { return moe_layers_; }

private:
    void upload(VkContext&, const model_io::LoadedModelRaw&, const VulkanLoadHooks&);

    HybridModelConfig cfg_;
    Tensor tok_embeddings_, final_norm_, lm_head_;
    std::vector<int> layer_kinds_;
    std::vector<size_t> layer_indices_;
    std::vector<VulkanDenseWeights> dense_layers_;
    std::vector<VulkanGdnWeights> gdn_layers_;
    std::vector<VkSparseMoELayer> moe_layers_;
};

}  // namespace fusionrt
