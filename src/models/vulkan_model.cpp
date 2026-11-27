#include "fusionrt/models.h"

#ifdef FUSIONRT_ENABLE_VULKAN
#include "vulkan_executor.h"
#include "vulkan_session.h"
#include "vulkan_weights.h"

namespace fusionrt {

struct VulkanModelRuntime {
    explicit VulkanModelRuntime(const HybridModelConfig& config) : weights(config), executor(weights, sessions) {}

    VulkanLoadHooks load_hooks(VkContext& ctx) {
        return {
            [&](const DenseGQAConfig& cfg, DType kv_type) { sessions.allocate_dense(ctx, cfg, kv_type); },
            [&](const GatedDeltaNetConfig& cfg, int conv_dim) { sessions.allocate_gdn(ctx, cfg, conv_dim); }
        };
    }

    VulkanModelWeights weights;
    VulkanSessionManager sessions;
    VulkanExecutor executor;
};

void FusionModel::init_synthetic_vulkan(VkContext& ctx, uint64_t seed) {
    auto runtime = std::make_shared<VulkanModelRuntime>(cfg_);
    runtime->weights.init_synthetic_vulkan(ctx, seed, runtime->load_hooks(ctx));
    vulkan_ = std::move(runtime);
}

bool FusionModel::load_gguf_vulkan(VkContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    auto runtime = std::make_shared<VulkanModelRuntime>(cfg_);
    if (!runtime->weights.load_gguf_vulkan(ctx, path, err, max_seq_override, runtime->load_hooks(ctx))) return false;
    cfg_ = runtime->weights.config();
    vulkan_ = std::move(runtime);
    return true;
}

bool FusionModel::load_safetensors_vulkan(VkContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    auto runtime = std::make_shared<VulkanModelRuntime>(cfg_);
    if (!runtime->weights.load_safetensors_vulkan(ctx, path, err, max_seq_override, runtime->load_hooks(ctx))) return false;
    cfg_ = runtime->weights.config();
    vulkan_ = std::move(runtime);
    return true;
}

void FusionModel::reset_kv_and_state_vulkan(VkContext& ctx) {
    if (vulkan_) vulkan_->sessions.reset_kv_and_state_vulkan(ctx);
}

std::vector<float> FusionModel::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    if (!vulkan_) throw std::logic_error("Vulkan model is not loaded");
    return vulkan_->executor.forward_token_vulkan(ctx, token_id, pos);
}

std::vector<int32_t> FusionModel::generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                  int max_new_tokens, GenerationStats* stats, bool keep_state,
                                                  int start_pos, int32_t stop_eos_id) {
    if (!vulkan_) throw std::logic_error("Vulkan model is not loaded");
    return vulkan_->executor.generate_vulkan(ctx, prompt_ids, max_new_tokens, stats, keep_state, start_pos, stop_eos_id);
}

}  // namespace fusionrt
#endif
