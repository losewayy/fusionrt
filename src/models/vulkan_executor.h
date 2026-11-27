#pragma once

#include "fusionrt/model_types.h"
#include "../core/cpu_moe.h"

namespace fusionrt {

struct GenerationOptions;

class VulkanModelWeights;
class VulkanSessionManager;

class VulkanExecutor {
public:
    VulkanExecutor(const VulkanModelWeights& weights, const VulkanSessionManager& sessions)
        : weights_(weights), sessions_(sessions) {}

    std::vector<float> forward_batch_vulkan(VkContext&, std::span<const int32_t>, int start_pos,
                                            bool compute_logits = true, std::span<const InputEmbedding> embeds = {},
                                            int prompt_offset = 0);
    std::vector<float> forward_verify_vulkan(VkContext&, std::span<const int32_t>, int start_pos);
    std::vector<float> forward_mtp_step_vulkan(VkContext&, int32_t token, int pos, bool chained, bool predict);
    std::vector<float> forward_token_vulkan(VkContext&, int32_t token_id, int pos);
    std::vector<int32_t> generate_vulkan(VkContext&, const std::vector<int32_t>& prompt_ids, int max_new_tokens,
                                         GenerationStats*, bool keep_state, int start_pos, int32_t stop_eos_id);
    std::vector<int32_t> generate_vulkan(VkContext&, const std::vector<int32_t>& prompt_ids, int max_new_tokens,
                                         GenerationStats*, bool keep_state, int start_pos, int32_t stop_eos_id,
                                         const GenerationOptions&, VulkanExecutor* draft = nullptr, VkContext* draft_ctx = nullptr);

private:
    std::vector<float> prefill_vulkan(VkContext&, std::span<const int32_t>, int start_pos,
                                      const GenerationOptions&, int& prefilled, bool compute_logits);

    std::vector<float> forward_impl(VkContext&, std::span<const int32_t>, int start_pos,
                                    bool compute_logits, std::span<const InputEmbedding>, int prompt_offset,
                                    bool verify, bool mtp, bool chained);
    size_t trunk_gdn_count() const;

    const VulkanModelWeights& weights_;
    const VulkanSessionManager& sessions_;
    std::unique_ptr<CpuWorkers> moe_workers_;
    CpuMoeWorkspace moe_workspace_;
    VkDeviceSize scratch_bytes(const VkContext&, int rows, bool embeds, bool verify = false, bool mtp = false) const;
};

}  // namespace fusionrt
