#pragma once

#include "fusionrt/models.h"
#include "fusionrt/tokenizer.h"

namespace fusionrt {

#ifdef FUSIONRT_ENABLE_CUDA
void serve_http(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                const std::string& model_name, const std::string& host, int port, bool use_graph);
#endif

}  // namespace fusionrt
