#include "vulkan_session.h"

namespace fusionrt {

void VulkanSessionManager::allocate_dense(VkContext& ctx, const DenseGQAConfig& cfg) {
    const std::vector<int64_t> shape{cfg.max_seq_len, cfg.num_key_value_heads, cfg.head_dim};
    auto& arena = ctx.weight_arena();
    dense_kv_.push_back({arena.alloc_tensor(shape, DType::F32, "vk_kc"),
                         arena.alloc_tensor(shape, DType::F32, "vk_vc")});
}

void VulkanSessionManager::allocate_gdn(VkContext& ctx, const GatedDeltaNetConfig& cfg, int conv_dim) {
    auto& arena = ctx.weight_arena();
    gdn_states_.push_back({arena.alloc_tensor({conv_dim, 3}, DType::F32, "vk_cs"),
                           arena.alloc_tensor({cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim}, DType::F32, "vk_ss")});
}

void VulkanSessionManager::reset_kv_and_state_vulkan(VkContext& ctx) const {
    std::vector<Tensor> state;
    state.reserve(2 * (dense_kv_.size() + gdn_states_.size()));
    for (const auto& kv : dense_kv_) {
        state.push_back(kv.key);
        state.push_back(kv.value);
    }
    for (const auto& gdn : gdn_states_) {
        state.push_back(gdn.conv);
        state.push_back(gdn.ssm);
    }
    ctx.zero_buffers(state);
}

}  // namespace fusionrt
