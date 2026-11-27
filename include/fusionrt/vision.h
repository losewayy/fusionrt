#pragma once

#include "fusionrt/gguf.h"
#include "fusionrt/image.h"
#include "fusionrt/models.h"
#include "fusionrt/tokenizer.h"

namespace fusionrt {

struct VisionInfo {
    std::string architecture, projector;
    int patch_size = 0, image_size = 0, hidden_size = 0, heads = 0, layers = 0;
    int output_size = 0, spatial_merge = 1;

    static VisionInfo from_gguf(const GgufFile& file);
};

#ifdef FUSIONRT_ENABLE_CUDA
struct VisionEmbedding {
    std::unique_ptr<CudaArena> storage;
    Tensor rows;
};

class VisionEncoder {
public:
    VisionEncoder(CudaContext& ctx, const std::string& path, int text_hidden_size);
    ~VisionEncoder();
    VisionEmbedding encode(CudaContext& ctx, const Image& image, int max_tokens) const;
    const VisionInfo& info() const;

private:
    struct Weights;
    std::unique_ptr<Weights> weights_;
};

struct VisionPrompt {
    std::vector<int32_t> tokens;
    std::vector<VisionEmbedding> images;
    std::vector<InputEmbedding> inputs_embeds;
};

VisionPrompt encode_vision_prompt(CudaContext& ctx, const VisionEncoder& encoder, const BpeTokenizer& tokenizer,
                                   std::string_view text, std::span<const Image* const> images, int context);
VisionPrompt encode_vision_prompt(CudaContext& ctx, const VisionEncoder& encoder, const BpeTokenizer& tokenizer,
                                   std::string_view text, std::span<const Image* const> images,
                                   std::span<const size_t> offsets, int context);
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
class VkContext;

class VulkanVisionEncoder {
public:
    VulkanVisionEncoder(VkContext& ctx, const std::string& path, int text_hidden_size);
    ~VulkanVisionEncoder();
    HostTensor encode(VkContext& ctx, const Image& image, int max_tokens) const;
    const VisionInfo& info() const;

private:
    struct Weights;
    std::unique_ptr<Weights> weights_;
};

struct VulkanVisionPrompt {
    std::vector<int32_t> tokens;
    std::vector<InputEmbedding> inputs_embeds;
};

VulkanVisionPrompt encode_vision_prompt_vulkan(VkContext& ctx, const VulkanVisionEncoder& encoder,
    const BpeTokenizer& tokenizer, std::string_view text, std::span<const Image* const> images,
    std::span<const size_t> offsets, int context);

inline VulkanVisionPrompt encode_vision_prompt_vulkan(VkContext& ctx, const VulkanVisionEncoder& encoder,
    const BpeTokenizer& tokenizer, std::string_view text, std::span<const Image* const> images, int context) {
    std::vector<size_t> offsets;
    for (size_t pos = text.find(image_placeholder); pos != text.npos;
         pos = text.find(image_placeholder, pos + image_placeholder.size())) offsets.push_back(pos);
    return encode_vision_prompt_vulkan(ctx, encoder, tokenizer, text, images, offsets, context);
}
#endif

}  // namespace fusionrt
