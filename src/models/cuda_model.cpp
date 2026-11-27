#include "fusionrt/models.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include "cuda_executor.h"
#include "cuda_session.h"
#include "cuda_weights.h"
#endif

namespace fusionrt {

FusionModel::FusionModel(HybridModelConfig cfg) : cfg_(std::move(cfg))
#ifdef FUSIONRT_ENABLE_CUDA
    , weights_(std::make_unique<CudaModelWeights>(cfg_)),
      sessions_(std::make_unique<CudaSessionManager>(*weights_, [this] { executor_->invalidate_graph(); })),
      executor_(std::make_unique<CudaExecutor>(*weights_, *sessions_))
#endif
{}

#ifdef FUSIONRT_ENABLE_CUDA
FusionModel::~FusionModel() = default;

static CudaLoadHooks load_hooks(CudaSessionManager& sessions, CudaExecutor& executor) {
    return {
        [&] { sessions.release_cuda_sessions(); },
        [&](DType dtype, size_t budget) { sessions.configure_placement(dtype, budget); },
        [&](CudaArena& arena, int hidden, int vocab, int context) {
            executor.allocate_workspace(arena, hidden, vocab, context);
        },
        [&](DType dtype, size_t slabs) { sessions.start_sessions(dtype, slabs); },
        [&](CudaArena& arena, int context, int hidden) { executor.allocate_mtp_hidden(arena, context, hidden); }
    };
}

void FusionModel::init_synthetic(CudaContext& ctx, uint64_t seed) {
    weights_->init_synthetic(ctx, seed, load_hooks(*sessions_, *executor_));
}

bool FusionModel::load_gguf(CudaContext& ctx, const std::string& path, std::string* err, int context, bool own_weights) {
    return weights_->load_gguf(ctx, path, err, context, own_weights, sessions_->load_settings(),
                               load_hooks(*sessions_, *executor_));
}

bool FusionModel::load_safetensors(CudaContext& ctx, const std::string& path, std::string* err, int context) {
    return weights_->load_safetensors(ctx, path, err, context, sessions_->load_settings(),
                                      load_hooks(*sessions_, *executor_));
}

void FusionModel::set_num_slots(int n) { sessions_->set_num_slots(n); }
int FusionModel::num_slots() const { return sessions_->num_slots(); }
void FusionModel::set_kv_vram_budget_bytes(size_t bytes) { sessions_->set_kv_vram_budget_bytes(bytes); }
void FusionModel::set_kv_host_budget_bytes(size_t bytes) { sessions_->set_kv_host_budget_bytes(bytes); }
void FusionModel::set_kv_commitment_tokens(int tokens) { sessions_->set_kv_commitment_tokens(tokens); }
void FusionModel::enable_native_mtp(bool enabled) { weights_->enable_native_mtp(enabled); }
bool FusionModel::has_native_mtp() const { return sessions_->initialized() && weights_->has_native_mtp(); }
void FusionModel::validate_cuda_kv_capacity(int tokens) const { sessions_->validate_cuda_kv_capacity(tokens); }
void FusionModel::set_moe_gpu_experts(int count) { weights_->set_moe_gpu_experts(count); }
void FusionModel::reset_kv_and_state(CudaContext& ctx) { sessions_->reset_kv_and_state(ctx); }
int FusionModel::cuda_max_active_tokens() const { return sessions_->cuda_max_active_tokens(); }
DType FusionModel::cuda_kv_dtype() const { return sessions_->dtype(); }
bool FusionModel::recommended_cuda_graph() const { return weights_->recommended_cuda_graph(); }
const std::vector<CudaDenseGQALayer>& FusionModel::dense_layers() const { return weights_->dense_layers(); }
const std::vector<CudaGatedDeltaNetLayer>& FusionModel::gdn_layers() const { return weights_->gdn_layers(); }
const std::vector<CudaSparseMoELayer>& FusionModel::moe_layers() const { return weights_->moe_layers(); }
size_t FusionModel::cuda_borrowed_expert_bytes() const { return sessions_->borrowed_bytes(); }
const MiniGraph& FusionModel::raw_ir_graph() const { return weights_->raw_ir(); }
const MiniGraph& FusionModel::fused_ir_graph() const { return weights_->fused_ir(); }
void FusionModel::rebuild_cuda_ir() { weights_->rebuild_ir(); }
int FusionModel::prefill_batch_limit(CudaContext& ctx, bool verify) const {
    return executor_->prefill_batch_limit(ctx, verify);
}

void FusionModel::set_device_profile(CudaContext& ctx, const DeviceProfile& profile) {
    weights_->set_device_profile(ctx, profile);
    executor_->invalidate_capture();
    sessions_->clear_histories();
}

ModelMemory FusionModel::cuda_cache_memory() const {
    return sessions_->cache_memory();
}

ModelMemory FusionModel::inspect_cuda(CudaContext& ctx) {
    sessions_->require_loaded();
    auto result = cuda_cache_memory();
    weights_->inspect_cuda(ctx, result);
    result.static_scratch_bytes = executor_->inspect_scratch(ctx);
    return result;
}

std::vector<float> FusionModel::forward_token_cuda(CudaContext& ctx, int32_t token_id, int pos,
                                                   bool use_fused, bool use_graph) {
    if (pos < 0 || pos >= cfg_.dense.max_seq_len) throw std::invalid_argument("Token position exceeds context");
    validate_token_id(token_id, pos, cfg_.vocab_size);
    sessions_->clear_cuda_history();
    executor_->ensure_kv(ctx, pos, pos + 1);
    auto request = executor_->begin_decode(use_fused);
    executor_->decode(ctx, *request, pos, use_graph, false, true, &token_id, false);
    HostTensor h_logits = ctx.download(executor_->logits());
    return {h_logits.data<float>(), h_logits.data<float>() + cfg_.vocab_size};
}

std::vector<float> FusionModel::forward_prefill_cuda(CudaContext& ctx, const std::vector<int32_t>& token_ids,
                                                     int start_pos, bool use_fused,
                                                     std::span<const InputEmbedding> inputs_embeds) {
    CudaExecutor::validate_input_embeddings(inputs_embeds, token_ids.size(), static_cast<int>(weights_->embeddings().shape[1]));
    if (token_ids.empty()) return {};
    if (start_pos < 0) throw std::invalid_argument("Invalid prefill start position");
    const auto total_tokens = static_cast<size_t>(start_pos) + token_ids.size();
    if (total_tokens > static_cast<size_t>(cfg_.dense.max_seq_len)) {
        throw prompt_context_error(total_tokens, cfg_.dense.max_seq_len);
    }
    validate_token_ids(token_ids, start_pos, cfg_.vocab_size);
    sessions_->clear_cuda_history();
    return executor_->prefill_cuda_impl(ctx, token_ids.data(), static_cast<int>(token_ids.size()), start_pos, use_fused,
                             nullptr, nullptr, false, inputs_embeds);
}


#endif

}  // namespace fusionrt
