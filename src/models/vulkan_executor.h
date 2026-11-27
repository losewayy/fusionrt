#pragma once

#include "fusionrt/model_types.h"

namespace fusionrt {

class VulkanModelWeights;
class VulkanSessionManager;

class VulkanExecutor {
public:
    VulkanExecutor(const VulkanModelWeights& weights, const VulkanSessionManager& sessions)
        : weights_(weights), sessions_(sessions) {}

    void allocate_workspace(VkContext&, int hidden, int vocab);
    std::vector<float> forward_token_vulkan(VkContext&, int32_t token_id, int pos);
    std::vector<int32_t> generate_vulkan(VkContext&, const std::vector<int32_t>& prompt_ids, int max_new_tokens,
                                         GenerationStats*, bool keep_state, int start_pos, int32_t stop_eos_id);

private:
    std::vector<float> prefill_vulkan(VkContext&, std::span<const int32_t>, int start_pos);

    const VulkanModelWeights& weights_;
    const VulkanSessionManager& sessions_;
    Tensor hidden_a_, hidden_b_, logits_, token_slot_;
};

}  // namespace fusionrt
