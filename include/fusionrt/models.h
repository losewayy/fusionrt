#pragma once

#include "fusionrt/model_types.h"
#include "fusionrt/sampler.h"
#include <atomic>

namespace fusionrt {

#ifdef FUSIONRT_ENABLE_CUDA
class CudaModelWeights;
class CudaSessionManager;
class CudaExecutor;
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
struct VulkanModelRuntime;
#endif

class FusionModel;

struct GenerationOptions {
    SamplingParams sampling;
    bool reuse_prefix = false;
    std::function<bool(int32_t)> on_token;  // Return false to stop.
    OpTrace* op_trace = nullptr;
    LayerTrace* layer_trace = nullptr;
    const JsonConstraint* json_constraint = nullptr;
    int spec_ngram = 0;
    bool spec_mtp = false;
    FusionModel* draft_model = nullptr;
    int draft_tokens = 4;
    float spec_p_min = 0.1f;
    std::span<const InputEmbedding> inputs_embeds;
    bool warn_on_graph_fallback = false;
    const std::atomic_bool* cancel = nullptr;
    bool auto_draft_tokens = false;  // Native MTP calibrates up to draft_tokens. Zero uses the full supported range.
};

class FusionModel {
public:
    explicit FusionModel(HybridModelConfig cfg);

    const HybridModelConfig& config() const { return cfg_; }

#ifdef FUSIONRT_ENABLE_CUDA
    ~FusionModel();
    void set_num_slots(int n);
    int num_slots() const;
    void set_kv_vram_budget_bytes(size_t bytes);
    void set_kv_commitment_tokens(int tokens);
    void enable_native_mtp(bool enabled);
    bool has_native_mtp() const;
    void validate_cuda_kv_capacity(int tokens) const;
    void set_kv_host_budget_bytes(size_t bytes);
    void set_moe_gpu_experts(int count);
    void init_synthetic(CudaContext& ctx, uint64_t seed = 42);
    bool load_gguf(CudaContext& ctx, const std::string& path, std::string* err = nullptr,
                   int max_seq_override = 0, bool own_weights = false);
    bool load_safetensors(CudaContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);

    void reset_kv_and_state(CudaContext& ctx);
    std::vector<float> forward_token_cuda(CudaContext& ctx, int32_t token_id, int pos,
                                          bool use_fused = true, bool use_graph = false);
    std::vector<float> forward_prefill_cuda(CudaContext& ctx, const std::vector<int32_t>& token_ids,
                                            int start_pos, bool use_fused = true,
                                            std::span<const InputEmbedding> inputs_embeds = {});
    std::vector<float> prefill_cached_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                          bool use_fused = true, int* prefilled_tokens = nullptr,
                                          LayerTrace* trace = nullptr, OpTrace* op_trace = nullptr,
                                          const std::atomic_bool* cancel = nullptr);
    std::vector<int32_t> generate_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                       int max_new_tokens, bool use_fused = true, bool use_graph = false,
                                       GenerationStats* stats = nullptr, bool keep_state = false, int start_pos = 0,
                                       int32_t stop_eos_id = -1, const GenerationOptions& options = {});
    std::vector<GenerationStats> bench_model_cuda(CudaContext& ctx, int pp, int tg, int reps, bool use_graph,
                                                  OpTrace* trace = nullptr, int decode_prefix = 0);
    void set_device_profile(CudaContext& ctx, const DeviceProfile& profile);
    bool recommended_cuda_graph() const;
    ModelMemory cuda_cache_memory() const;
    int cuda_max_active_tokens() const;
    DType cuda_kv_dtype() const;
    ModelMemory inspect_cuda(CudaContext& ctx);
    int prefill_batch_limit(CudaContext& ctx, bool verify = false) const;

    const std::vector<CudaDenseGQALayer>& dense_layers() const;
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers() const;
    const std::vector<CudaSparseMoELayer>& moe_layers() const;
    size_t cuda_borrowed_expert_bytes() const;
    const MiniGraph& raw_ir_graph() const;
    const MiniGraph& fused_ir_graph() const;
    void rebuild_cuda_ir();
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
    void init_synthetic_vulkan(VkContext& ctx, uint64_t seed = 42);
    bool load_gguf_vulkan(VkContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);
    bool load_safetensors_vulkan(VkContext& ctx, const std::string& path, std::string* err = nullptr, int max_seq_override = 0);

    void reset_kv_and_state_vulkan(VkContext& ctx);
    std::vector<float> forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos);
    std::vector<int32_t> generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                         int max_new_tokens, GenerationStats* stats = nullptr,
                                         bool keep_state = false, int start_pos = 0,
                                         int32_t stop_eos_id = -1);
#endif

private:
    HybridModelConfig cfg_;

#ifdef FUSIONRT_ENABLE_CUDA
    std::unique_ptr<CudaModelWeights> weights_;
    std::unique_ptr<CudaSessionManager> sessions_;
    std::unique_ptr<CudaExecutor> executor_;
    friend struct CudaReviewTestAccess;
#endif

#ifdef FUSIONRT_ENABLE_VULKAN
    std::shared_ptr<VulkanModelRuntime> vulkan_;
#endif
};

}  // namespace fusionrt
