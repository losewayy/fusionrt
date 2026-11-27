#pragma once

#include "fusionrt/models.h"
#include "fusionrt/tokenizer.h"
#include "fusionrt/image.h"
#include <optional>

namespace fusionrt {

struct ServeSpeculation {
    int ngram = 0;
    FusionModel* draft = nullptr;
    int draft_tokens = 4;
    bool mtp = false;

    GenerationOptions for_inputs(std::span<const InputEmbedding> embeds) const {
        GenerationOptions options;
        options.inputs_embeds = embeds;
        if (embeds.empty()) {
            options.spec_ngram = ngram;
            options.spec_mtp = mtp;
            options.draft_model = draft;
            options.draft_tokens = draft_tokens;
        }
        return options;
    }
};

#ifdef FUSIONRT_ENABLE_CUDA
class VisionEncoder;
void serve_http(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                const std::string& model_name, const std::string& host, int port, std::optional<bool> graph,
                const VisionEncoder* vision = nullptr, std::span<const Image> preset_images = {},
                ServeSpeculation speculation = {});
#endif

}  // namespace fusionrt
