#include "fusionrt/models.h"

#ifdef FUSIONRT_ENABLE_VULKAN
#include "vulkan_executor.h"
#include "vulkan_session.h"
#include "vulkan_weights.h"

namespace fusionrt {

struct VulkanModelRuntime {
    VulkanModelRuntime(VkContext& ctx, const HybridModelConfig& config, int gpu_experts, bool mtp)
        : context(ctx), weights(config, gpu_experts, mtp), executor(weights, sessions) {}

    VulkanLoadHooks load_hooks(VkContext& ctx) {
        return {
            [&](const DenseGQAConfig& cfg, DType kv_type) { sessions.allocate_dense(ctx, cfg, kv_type); },
            [&](const GatedDeltaNetConfig& cfg, int conv_dim) { sessions.allocate_gdn(ctx, cfg, conv_dim); },
            [&](int source) { sessions.allocate_dense(ctx, {}, DType::F32, source); }
        };
    }

    VkContext& context;
    GenerationOptions speculation;
    VulkanModelWeights weights;
    VulkanSessionManager sessions;
    VulkanExecutor executor;
};

void FusionModel::set_moe_gpu_experts(VkContext&, int count) {
    if (count < -1) throw std::invalid_argument("Expected --moe-gpu-experts >= -1");
    vulkan_gpu_experts_ = count;
}

void FusionModel::enable_native_mtp(VkContext&, bool enabled) { vulkan_load_mtp_ = enabled; }

bool FusionModel::has_native_mtp(const VkContext& ctx) const {
    return vulkan_ && &vulkan_->context == &ctx && vulkan_->weights.mtp().projection.vk_buffer;
}

void FusionModel::set_vulkan_speculation(int ngram, FusionModel* draft, int depth, bool mtp, float p_min) {
    if (!vulkan_) throw std::logic_error("Vulkan model is not loaded");
    auto& options = vulkan_->speculation;
    options.spec_ngram = ngram;
    options.draft_model = draft;
    options.draft_tokens = depth;
    options.spec_mtp = mtp;
    options.spec_p_min = p_min;
    options.auto_draft_tokens = depth == 0;
}

void FusionModel::init_synthetic_vulkan(VkContext& ctx, uint64_t seed) {
    auto runtime = std::make_shared<VulkanModelRuntime>(ctx, cfg_, vulkan_gpu_experts_, vulkan_load_mtp_);
    runtime->weights.init_synthetic_vulkan(ctx, seed, runtime->load_hooks(ctx));
    cfg_ = runtime->weights.config();
    vulkan_ = std::move(runtime);
}

bool FusionModel::load_gguf_vulkan(VkContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    auto runtime = std::make_shared<VulkanModelRuntime>(ctx, cfg_, vulkan_gpu_experts_, vulkan_load_mtp_);
    if (!runtime->weights.load_gguf_vulkan(ctx, path, err, max_seq_override, runtime->load_hooks(ctx))) return false;
    cfg_ = runtime->weights.config();
    vulkan_ = std::move(runtime);
    return true;
}

bool FusionModel::load_safetensors_vulkan(VkContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    auto runtime = std::make_shared<VulkanModelRuntime>(ctx, cfg_, vulkan_gpu_experts_, vulkan_load_mtp_);
    if (!runtime->weights.load_safetensors_vulkan(ctx, path, err, max_seq_override, runtime->load_hooks(ctx))) return false;
    cfg_ = runtime->weights.config();
    vulkan_ = std::move(runtime);
    return true;
}

void FusionModel::reset_kv_and_state_vulkan(VkContext& ctx) {
    if (!vulkan_) return;
    if (&vulkan_->context != &ctx) throw std::invalid_argument("Vulkan model was loaded on a different context");
    vulkan_->sessions.reset_kv_and_state_vulkan(ctx);
}

std::vector<float> FusionModel::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    if (!vulkan_) throw std::logic_error("Vulkan model is not loaded");
    if (&vulkan_->context != &ctx) throw std::invalid_argument("Vulkan model was loaded on a different context");
    return vulkan_->executor.forward_token_vulkan(ctx, token_id, pos);
}

std::vector<float> FusionModel::forward_verify_vulkan(VkContext& ctx, std::span<const int32_t> tokens, int pos) {
    if (!vulkan_) throw std::logic_error("Vulkan model is not loaded");
    if (&vulkan_->context != &ctx) throw std::invalid_argument("Vulkan model was loaded on a different context");
    return vulkan_->executor.forward_verify_vulkan(ctx, tokens, pos);
}

std::vector<int32_t> FusionModel::generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                  int max_new_tokens, GenerationStats* stats, bool keep_state,
                                                  int start_pos, int32_t stop_eos_id, const GenerationOptions& options) {
    if (!vulkan_) throw std::logic_error("Vulkan model is not loaded");
    if (&vulkan_->context != &ctx) throw std::invalid_argument("Vulkan model was loaded on a different context");
    auto generation = options;
    const auto& defaults = vulkan_->speculation;
    if (max_new_tokens > 0 && options.inputs_embeds.empty() && !options.spec_ngram && !options.spec_mtp && !options.draft_model) {
        generation.spec_ngram = defaults.spec_ngram;
        generation.spec_mtp = defaults.spec_mtp;
        generation.draft_model = defaults.draft_model;
        if (defaults.spec_ngram || defaults.spec_mtp || defaults.draft_model) {
            generation.draft_tokens = defaults.draft_tokens;
            generation.spec_p_min = defaults.spec_p_min;
            generation.auto_draft_tokens = defaults.auto_draft_tokens;
        }
    }
    auto* draft = generation.draft_model ? generation.draft_model->vulkan_.get() : nullptr;
    return vulkan_->executor.generate_vulkan(ctx, prompt_ids, max_new_tokens, stats, keep_state, start_pos, stop_eos_id,
        generation, draft ? &draft->executor : nullptr, draft ? &draft->context : nullptr);
}

}  // namespace fusionrt
#endif
