#pragma once

#include "fusionrt/models.h"
#include "fusionrt/tokenizer.h"
#include "fusionrt/image.h"

namespace fusionrt {

#ifdef FUSIONRT_ENABLE_CUDA
class VisionEncoder;
void serve_http(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                const std::string& model_name, const std::string& host, int port, bool use_graph,
                const VisionEncoder* vision = nullptr, std::span<const Image> preset_images = {});
#endif

}  // namespace fusionrt
