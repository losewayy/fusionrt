#pragma once

#include "fusionrt/model_types.h"

#ifdef FUSIONRT_ENABLE_CUDA
namespace fusionrt {

// The facade wires runtime resources into the loader without making the loader depend on them.
struct CudaLoadSettings {
    size_t requested_kv_bytes;
    int commitment_tokens;
    int slots;
};

struct CudaLoadHooks {
    std::function<void()> release_sessions;
    std::function<void(DType, size_t)> set_placement;
    std::function<void(CudaArena&, int, int, int)> allocate_workspace;
    std::function<void(DType, size_t)> start_sessions;
    std::function<void(CudaArena&, int, int)> allocate_mtp_hidden;
};

class CudaModelWeights {
public:
    explicit CudaModelWeights(HybridModelConfig& config) : cfg_(config) {}

    void init_synthetic(CudaContext&, uint64_t seed, const CudaLoadHooks&);
    bool load_gguf(CudaContext&, const std::string&, std::string*, int, bool,
                   CudaLoadSettings, const CudaLoadHooks&);
    bool load_safetensors(CudaContext&, const std::string&, std::string*, int, CudaLoadSettings,
                          const CudaLoadHooks&);
    void set_moe_gpu_experts(int count);
    void set_device_profile(CudaContext&, const DeviceProfile&);
    void rebuild_ir();
    void inspect_cuda(CudaContext&, ModelMemory&) const;
    void enable_native_mtp(bool enabled) { load_mtp_ = enabled; }
    bool has_native_mtp() const { return mtp_.projection.data != nullptr; }
    bool recommended_cuda_graph() const { return recommend_cuda_graph_; }
    const HybridModelConfig& config() const { return cfg_; }
    const Tensor& embeddings() const { return tok_embeddings_; }
    const auto& dense_layers() const { return dense_layers_; }
    const auto& gdn_layers() const { return gdn_layers_; }
    const auto& moe_layers() const { return moe_layers_; }
    std::span<CudaSparseMoELayer> borrowable_experts() { return moe_layers_; }
    const auto& raw_ir() const { return raw_ir_graph_; }
    const auto& fused_ir() const { return fused_ir_graph_; }
    const auto& ir(bool fused) const { return fused ? fused_ir_graph_ : raw_ir_graph_; }
    bool has_per_layer_embeddings() const { return gemma_inputs_.token_embeddings.data != nullptr; }

    int64_t per_layer_embedding_width() const { return gemma_inputs_.token_embeddings.shape[1]; }

    struct NativeMtp {
        Tensor projection, embedding, enorm, hnorm, head_norm, head;
        float projection_scale = 1.f, head_scale = 1.f;
        MiniGraph graph;
    };
    const NativeMtp& mtp() const { return mtp_; }

private:
    HybridModelConfig& cfg_;
    CudaArena gguf_weights_;
    Tensor tok_embeddings_, final_norm_w_, lm_head_w_;
    GemmaLayerInputWeights<Tensor> gemma_inputs_;
    std::vector<int> layer_kinds_;
    std::vector<size_t> layer_indices_;
    std::vector<CudaDenseGQALayer> dense_layers_;
    std::vector<CudaGatedDeltaNetLayer> gdn_layers_;
    std::vector<CudaSparseMoELayer> moe_layers_;
    NativeMtp mtp_;
    MiniGraph raw_ir_graph_, fused_ir_graph_;
    CudaContext* owned_gguf_arena_ = nullptr;
    int moe_gpu_experts_override_ = -1;
    bool load_mtp_ = false;
    bool recommend_cuda_graph_ = false;
};

}  // namespace fusionrt
#endif
