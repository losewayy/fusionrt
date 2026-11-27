#pragma once

#include "fusionrt/model_types.h"

#ifdef FUSIONRT_ENABLE_CUDA
namespace fusionrt {

class CudaModelWeights;
struct CudaLoadSettings;

class CudaSessionManager {
public:
    CudaSessionManager(const CudaModelWeights& weights, std::function<void()> invalidate_graph)
        : weights_(weights), invalidate_graph_(std::move(invalidate_graph)) {}
    void set_num_slots(int n);
    int num_slots() const { return static_cast<int>(cuda_slots_.size()); }
    CudaLoadSettings load_settings() const;
    void configure_placement(DType, size_t);
    void start_sessions(DType, size_t slab_count);
    void release_cuda_sessions();
    void set_kv_vram_budget_bytes(size_t bytes) {
        cuda_kv_pool_.set_kv_vram_budget_bytes(bytes);
        requested_kv_budget_ = bytes;
    }
    void set_kv_host_budget_bytes(size_t bytes) { cuda_kv_pool_.set_kv_host_budget_bytes(bytes); }
    void set_kv_commitment_tokens(int tokens);
    DType dtype() const { return cuda_kv_dtype_; }
    size_t borrowed_bytes() const { return expert_slab_bytes_; }
    uint64_t expert_residency_generation() const { return expert_residency_gen_; }
    void require_loaded() const {
        if (!initialized()) throw std::logic_error("CUDA model is not loaded");
    }
    bool initialized() const { return cuda_slots_[0] != nullptr; }
    const std::vector<CudaKvView>& kv() const { return active().kv; }
    const std::vector<CudaGdnState>& gdn() const { return active().gdn; }
    size_t recurrent_bytes() const { return active().ssm_bytes; }
    void save_recurrent(CudaContext&, void* dst) const;
    void restore_recurrent(CudaContext&, const void* src);
    using HostRecurrentState = std::vector<std::pair<HostTensor, HostTensor>>;
    void download_recurrent(CudaContext&, HostRecurrentState&) const;
    void upload_recurrent(CudaContext&, const HostRecurrentState&);
    void clear_cuda_history();
    void clear_histories();
    void forget_tokens() { active().cached_tokens.clear(); }
    void retain_tokens(std::span<const int32_t> tokens, bool fused) {
        active().cached_tokens.assign(tokens.begin(), tokens.end());
        active().cached_fused = fused;
    }
    void truncate(int tokens) { cuda_kv_pool_.truncate(active().pages, tokens); }
    void zero_cuda_gdn(CudaContext&);
    void reset_kv_and_state(CudaContext&);
    void ensure_cuda_kv(CudaContext&, int first_write, int tokens, std::span<CudaSparseMoELayer>);
    void validate_cuda_kv_capacity(int tokens) const;
    void promote_cuda_expert_slabs(CudaContext&, int budgeted_tokens, std::span<CudaSparseMoELayer>);
    void save_cuda_checkpoint(CudaContext&, int tokens, const Tensor* logits);
    ModelMemory cache_memory(size_t logits_bytes) const;
    int cuda_max_active_tokens() const;
    struct Prefix { int length; bool checkpoint_pages; };
    Prefix prepare_prefix(CudaContext&, const std::vector<int32_t>&, bool fused,
                          bool allow_cached_logits, const Tensor& logits);

private:
    friend struct CudaReviewTestAccess;
    const CudaModelWeights& weights_;
    std::function<void()> invalidate_graph_;
    struct GdnSnapshot {
        std::unique_ptr<uint8_t, decltype(&cudaFree)> data{nullptr, cudaFree};
        bool has_logits = false;
    };
    struct GdnCheckpoint {
        int tokens;
        std::shared_ptr<const GdnSnapshot> snapshot;
    };
    struct CudaSession {
        CudaArena storage;
        std::vector<CudaKvView> kv;
        CudaKvPool::Table pages;
        std::vector<CudaGdnState> gdn;
        void* gdn_data = nullptr;
        size_t ssm_bytes = 0;
        std::vector<GdnCheckpoint> checkpoints;
        // Checkpoints belong to exact prefixes of this token history.
        std::vector<int32_t> cached_tokens;
        bool cached_fused = true;
        uint64_t last_used = 0;
    };
    CudaSession& active() { require_loaded(); return *cuda_slots_[active_slot_]; }
    const CudaSession& active() const { require_loaded(); return *cuda_slots_[active_slot_]; }
    std::unique_ptr<CudaSession> make_cuda_session() const;
    size_t cuda_kv_page_bytes() const;
    void reclaim_cuda_expert_slabs(CudaContext&, size_t, bool, std::span<CudaSparseMoELayer>);

    CudaKvPool cuda_kv_pool_;
    std::vector<std::unique_ptr<CudaSession>> cuda_slots_{1};
    DType cuda_kv_dtype_ = DType::F32;
    size_t active_slot_ = 0;
    uint64_t slot_clock_ = 0;
    size_t requested_kv_budget_ = 0;
    int kv_commitment_tokens_ = 0;
    bool may_evict_paused_cuda_slots_ = false;
    struct ExpertSlab {
        std::unique_ptr<uint8_t, decltype(&cudaFree)> storage{nullptr, cudaFree};
        size_t bytes = 0;
    };
    std::vector<ExpertSlab> expert_slabs_;
    size_t expert_slab_bytes_ = 0;
    uint64_t expert_residency_gen_ = 0;
};

}  // namespace fusionrt
#endif
