#pragma once

#include "cuda_session.h"
#include "cuda_weights.h"
#include <atomic>

#ifdef FUSIONRT_ENABLE_CUDA
namespace fusionrt {

struct CudaPrefillCancelled {};

class CudaExecutor {
public:
    CudaExecutor(CudaModelWeights& weights, CudaSessionManager& sessions)
        : weights_(weights), sessions_(sessions) {}

    void allocate_workspace(CudaArena&, int hidden, int vocab, int context);
    void allocate_mtp_hidden(CudaArena& arena, int context, int hidden) {
        mtp_target_hidden_ = arena.alloc_tensor({context, hidden}, DType::F32);
    }
    void invalidate_graph() { work_.decode_graph.reset(); work_.graph_captured = false; }
    void invalidate_capture() { work_.graph_captured = false; }
    bool can_use_cuda_graph() const;
    bool has_graph_prefix(bool fused) const;
    const Tensor& logits() const { return work_.logits; }
    void set_mtp_hidden_capture(bool capture) { capture_mtp_hidden_ = capture; }
    int prefill_batch_limit(CudaContext&, bool verify = false) const;
    static void validate_input_embeddings(std::span<const InputEmbedding>, size_t tokens, int width);
    void ensure_kv(CudaContext&, int first_write, int tokens);
    void execute_prefill_chunks(CudaContext&, const int32_t*, int count, int start_pos, bool fused,
                                LayerTrace*, OpTrace*, bool compute_logits = true, bool checkpoint_pages = false,
                                std::span<const InputEmbedding> = {}, const std::atomic_bool* cancel = nullptr);
    std::vector<float> prefill_cuda_impl(CudaContext&, const int32_t*, int count, int start_pos, bool fused,
                                       LayerTrace* = nullptr, OpTrace* = nullptr, bool checkpoint_pages = false,
                                       std::span<const InputEmbedding> = {}, const std::atomic_bool* cancel = nullptr);
    size_t inspect_scratch(CudaContext&);
    int32_t download_argmax(CudaContext&);
    int32_t sample_gpu(CudaContext&);
    void set_token(CudaContext&, const int32_t*);
    int64_t token_history_capacity() const { return work_.token_ids.shape[0]; }
    void clear_invalid_token(CudaContext&);
    void download_token_history(CudaContext&, int offset, std::span<int32_t>, bool check_invalid = false);
    void store_token_history(CudaContext&, int offset);

    // These request handles own transient graph and execution state.
    struct Decode;
    struct Verify;
    struct MtpDraft;
    struct Bench;
    template<class T> using Request = std::unique_ptr<T, void(*)(T*)>;
    Request<Decode> begin_decode(bool fused, OpTrace* trace = nullptr, bool allocate_timer = false);
    bool decode(CudaContext&, Decode&, int pos, bool graph, bool argmax, bool first_step = true,
                const int32_t* token = nullptr, bool ensure = true);

    Request<Verify> begin_verify(CudaContext&, bool native_mtp, bool graph, int batch_limit);
    bool verify_cuda(CudaContext&, const std::vector<int32_t>&, int pos, bool fused,
                     std::vector<float>&, std::vector<int32_t>&, bool greedy, bool retain_last_logits,
                     Verify* = nullptr);
    bool restore_verified_recurrence(CudaContext&, const Verify&, int rows, bool fused);
    Request<MtpDraft> begin_mtp(CudaContext&, bool cutoff);
    bool mtp_step(CudaContext&, MtpDraft&, int position, int32_t token, int seed_position,
                  bool use_draft_hidden, bool predict, bool fused, const Verify&);
    std::pair<int32_t, float> read_mtp(CudaContext&, const MtpDraft&);

    Request<Bench> begin_bench(CudaContext&, int pp);
    int32_t* bench_args(Bench&);
    void prepare_bench_decode(CudaContext&, Bench&);
    void bench_prefill(CudaContext&, Bench&, int pp);
    void bench_decode(CudaContext&, Bench&, int pos, bool graph);
    void capture_bench(CudaContext&, Bench&, int size_or_pos, bool prefill);
    void launch_bench(CudaContext&, Bench&);
    void clear_bench_graph(Bench&);

private:
    friend struct CudaReviewTestAccess;
    CudaModelWeights& weights_;
    CudaSessionManager& sessions_;
    struct Workspace {
        Tensor hidden, logits, token_ids, position, kv_length, invalid_token;
        CudaGraphExec decode_graph;
        bool graph_captured = false;
        bool graph_fused = true;
        bool graph_has_argmax = false;
        int graph_split_bucket = 0;
    } work_;
    Tensor mtp_target_hidden_;
    bool capture_mtp_hidden_ = false;
};

}  // namespace fusionrt
#endif
