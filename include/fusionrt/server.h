#pragma once

#include "fusionrt/models.h"
#include "fusionrt/tokenizer.h"
#include "fusionrt/image.h"
#include <optional>
#include <stop_token>

namespace fusionrt {

struct ServeSpeculation {
    int ngram = 0;
    FusionModel* draft = nullptr;
    int draft_tokens = 4;
    bool mtp = false;
    float p_min = 0.1f;
    bool auto_draft_tokens = false;

    const char* mode() const { return mtp ? "mtp" : draft ? "draft" : ngram > 0 ? "ngram" : "none"; }
    bool enabled() const { return mtp || draft || ngram > 0; }

    GenerationOptions for_inputs(std::span<const InputEmbedding> embeds) const {
        GenerationOptions options;
        options.inputs_embeds = embeds;
        if (embeds.empty()) {
            options.spec_ngram = ngram;
            options.spec_mtp = mtp;
            options.draft_model = draft;
            options.draft_tokens = draft_tokens;
            options.auto_draft_tokens = auto_draft_tokens;
            options.spec_p_min = p_min;
        }
        return options;
    }
};

#ifdef FUSIONRT_ENABLE_CUDA
class VisionEncoder;
void serve_http(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                const std::string& model_name, const std::string& host, int port, std::optional<bool> graph,
                const VisionEncoder* vision = nullptr, std::span<const Image> preset_images = {},
                ServeSpeculation speculation = {}, std::stop_token shutdown = {});
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
class VkContext;
class VulkanVisionEncoder;
void serve_http(VkContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                const std::string& model_name, const std::string& host, int port,
                const VulkanVisionEncoder* vision = nullptr, std::span<const Image> preset_images = {},
                ServeSpeculation speculation = {}, std::stop_token shutdown = {});
#endif

}  // namespace fusionrt
