#include "cuda_session.h"
#include "cuda_weights.h"
#include "../core/spec_cost.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include <algorithm>
#include <iostream>
#include <tuple>
#include <unordered_set>

namespace fusionrt {

void CudaSessionManager::set_kv_commitment_tokens(int tokens) {
    if (tokens < 0) throw std::invalid_argument("KV commitment must be nonnegative");
    kv_commitment_tokens_ = tokens;
}

void CudaSessionManager::set_num_slots(int n) {
    if (n < 1) throw std::invalid_argument("Expected --slots >= 1");
    const auto count = static_cast<size_t>(n);
    if (active_slot_ >= count) {
        invalidate_graph_();
        active_slot_ = 0;
    }
    for (size_t i = count; i < cuda_slots_.size(); ++i) {
        if (cuda_slots_[i]) cuda_kv_pool_.truncate(cuda_slots_[i]->pages, 0);
    }
    cuda_slots_.resize(count);
}

std::unique_ptr<CudaSessionManager::CudaSession> CudaSessionManager::make_cuda_session() const {
    auto slot = std::make_unique<CudaSession>();
    auto aligned = [](size_t bytes) { return (bytes + 255) & ~size_t{255}; };
    size_t table_bytes = 0;
    for (const auto& layer : weights_.dense_layers()) {
        if (layer.kv_source_layer >= 0) continue;
        table_bytes += aligned((1 + (layer.cfg.max_seq_len - 1) / kKvPageTokens) * sizeof(uint8_t*));
    }
    size_t state_bytes = 0;
    for (const auto& layer : weights_.gdn_layers()) {
        const auto& g = layer.cfg;
        slot->gdn.push_back({
            Tensor::from_shape({layer.conv_w.shape[0], 3}, DType::F32, Backend::CUDA),
            Tensor::from_shape({g.num_v_heads, g.head_k_dim, g.head_v_dim}, DType::F32, Backend::CUDA)});
        state_bytes += aligned(slot->gdn.back().convolution.nbytes()) + aligned(slot->gdn.back().recurrence.nbytes());
    }
    slot->storage.init(table_bytes + state_bytes);
    for (const auto& layer : weights_.dense_layers()) {
        const auto& d = layer.cfg;
        const bool mla = d.mla.latent_rank > 0;
        slot->kv.push_back(layer.kv_source_layer >= 0 ? CudaKvView{} :
            make_cuda_kv_view(slot->storage, cuda_kv_dtype_, d.max_seq_len, d.num_key_value_heads,
                              mla ? d.mla.latent_width() : d.head_dim, mla));
    }
    if (state_bytes != 0) {
        slot->gdn_data = slot->storage.alloc(state_bytes);
        slot->ssm_bytes = state_bytes;
        auto* data = static_cast<uint8_t*>(slot->gdn_data);
        for (auto& state : slot->gdn) {
            state.convolution.data = data;
            data += aligned(state.convolution.nbytes());
            state.recurrence.data = data;
            data += aligned(state.recurrence.nbytes());
        }
    }
    return slot;
}

void CudaSessionManager::release_cuda_sessions() {
    invalidate_graph_();
    for (auto& slot : cuda_slots_) slot.reset();
    cuda_kv_pool_.release();
    if (expert_slab_bytes_) ++expert_residency_gen_;
    expert_slabs_.clear();
    expert_slab_bytes_ = 0;
    active_slot_ = 0;
    slot_clock_ = 0;
    prefix_prefill_ms_per_token_ = 0;
}

void CudaSessionManager::clear_cuda_history() {
    auto& slot = active();
    slot.cached_tokens.clear();
    slot.checkpoints.clear();
    slot.mtp_kv_valid_len = 0;
}

void CudaSessionManager::truncate(int tokens) {
    auto& slot = active();
    cuda_kv_pool_.truncate(slot.pages, tokens);
    slot.mtp_kv_valid_len = std::min(slot.mtp_kv_valid_len, tokens);
    std::erase_if(slot.checkpoints, [=](const auto& entry) { return entry.tokens > tokens; });
}

void CudaSessionManager::zero_cuda_gdn(CudaContext& ctx) {
    auto& slot = active();
    if (slot.ssm_bytes != 0) {
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(slot.gdn_data, 0, slot.ssm_bytes, ctx.stream()));
    }
}

void CudaSessionManager::reset_kv_and_state(CudaContext& ctx) {
    clear_cuda_history();
    truncate(0);
    zero_cuda_gdn(ctx);
    ctx.synchronize();
}

size_t CudaSessionManager::cuda_kv_page_bytes() const {
    size_t page_bytes = 0;
    for (const auto& layer : active().kv) page_bytes += layer.page_bytes();
    return page_bytes;
}

int CudaSessionManager::cuda_max_active_tokens() const {
    const int context = weights_.config().dense.max_seq_len;
    const auto budget = cuda_kv_pool_.vram_budget_bytes();
    const auto page_bytes = cuda_kv_page_bytes();
    if (!budget || !page_bytes) return context;
    const auto pages = std::min(budget / page_bytes, static_cast<size_t>(1 + (context - 1) / kKvPageTokens));
    return static_cast<int>(std::min(pages * kKvPageTokens, static_cast<size_t>(context)));
}

void CudaSessionManager::validate_cuda_kv_capacity(int tokens) const {
    const auto budget = cuda_kv_pool_.vram_budget_bytes();
    if (tokens <= 0 || !budget) return;
    const auto page_bytes = cuda_kv_page_bytes();
    const size_t needed = static_cast<size_t>(1 + (tokens - 1) / kKvPageTokens) * page_bytes;
    if (needed > budget)
        throw std::invalid_argument("Active KV for " + std::to_string(tokens) + " tokens needs " +
            std::to_string(needed) + " bytes, exceeding the " + std::to_string(budget) + "-byte KV budget");
}

void CudaSessionManager::promote_cuda_expert_slabs(CudaContext& ctx, int budgeted_tokens,
                                                   std::span<CudaSparseMoELayer> moe_layers) {
    if (expert_slabs_.empty()) return;
    if (cuda_kv_pool_.bytes() < cuda_kv_pool_.reserved_bytes()) {
        ctx.synchronize();
        cuda_kv_pool_.release_unused_tail();
    }
    size_t per_page = 0;
    for (const auto& layer : active().kv) per_page += layer.page_bytes();
    const auto pages = 1 + (budgeted_tokens - 1) / kKvPageTokens;
    const auto required = std::max(cuda_kv_pool_.reserved_bytes(), per_page * pages);
    const auto commitment = cuda_kv_pool_.vram_budget_bytes();
    if (required >= commitment || expert_slab_bytes_ >= commitment - required) return;
    size_t available = commitment - required - expert_slab_bytes_;
    const auto align = [](size_t bytes) { return (bytes + 255) & ~size_t{255}; };
    int promoted = 0;
    for (size_t i = 0; i < moe_layers.size(); ++i) {
        auto& layer = moe_layers[i];
        auto& slab = expert_slabs_[i];
        const int base = layer.cfg.gpu_resident_experts;
        const int tail = layer.cfg.num_experts - base;
        if (tail == 0 || slab.storage) continue;
        const auto gate = layer.cpu_gate_up_w.bytes().size() / tail;
        const auto up = layer.cpu_up_w.bytes().size() / tail;
        const auto down = layer.cpu_down_w.bytes().size() / tail;
        if (!gate || !down) continue;
        auto bytes_for = [&](int n) { return align(gate * n) + align(up * n) + align(down * n); };
        int count = tail;
        while (count > 0 && bytes_for(count) > available) --count;
        if (!count) continue;
        const size_t bytes = bytes_for(count);
        uint8_t* ptr = nullptr;
        const auto status = cudaMalloc(&ptr, bytes);
        if (status == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            break;
        }
        FUSIONRT_CUDA_CHECK(status);
        ExpertSlab candidate;
        candidate.storage.reset(ptr);
        candidate.bytes = bytes;
        auto upload = [&](const HostTensor& source, size_t unit, uint8_t*& dest) {
            if (!unit) return Tensor{};
            const auto& shape = source.desc.shape;
            auto tensor = Tensor::make_contiguous({count, shape[1], shape[2]}, source.desc.dtype, Backend::CUDA, dest);
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dest, source.bytes().data(), unit * count,
                                                cudaMemcpyHostToDevice, ctx.stream()));
            dest += align(unit * count);
            return tensor;
        };
        auto g = upload(layer.cpu_gate_up_w, gate, ptr);
        auto u = upload(layer.cpu_up_w, up, ptr);
        auto d = upload(layer.cpu_down_w, down, ptr);
        if (promoted++ == 0) {
            invalidate_graph_();
        }
        slab = std::move(candidate);
        layer.borrowed_gate_up_w = std::move(g);
        layer.borrowed_up_w = std::move(u);
        layer.borrowed_down_w = std::move(d);
        layer.cfg.gpu_resident_experts += count;
        available -= bytes;
        expert_slab_bytes_ += bytes;
        ++expert_residency_gen_;
    }
    if (promoted) {
        int full = 0;
        for (const auto& layer : moe_layers) full += layer.cfg.gpu_resident_experts == layer.cfg.num_experts;
        std::clog << "MoE slabs promoted=" << promoted << " borrowed_bytes=" << expert_slab_bytes_
                  << " full_gpu_layers=" << full << " kv_reserved_bytes=" << cuda_kv_pool_.reserved_bytes() << '\n';
    }
}

void CudaSessionManager::reclaim_cuda_expert_slabs(CudaContext& ctx, size_t target_kv_bytes, bool allocation_failed, std::span<CudaSparseMoELayer> moe_layers) {
    const auto commitment = cuda_kv_pool_.vram_budget_bytes();
    const size_t allowed = allocation_failed ? 0 : commitment - std::min(commitment, target_kv_bytes);
    if (expert_slab_bytes_ <= allowed) return;
    ctx.synchronize();
    invalidate_graph_();
    int evicted = 0;
    for (size_t i = expert_slabs_.size(); i > 0 && expert_slab_bytes_ > allowed;) {
        --i;
        auto& slab = expert_slabs_[i];
        if (!slab.storage) continue;
        auto& layer = moe_layers[i];
        auto* ptr = slab.storage.release();
        FUSIONRT_CUDA_CHECK(cudaFree(ptr));
        expert_slab_bytes_ -= slab.bytes;
        slab.bytes = 0;
        layer.cfg.gpu_resident_experts -= static_cast<int>(layer.borrowed_gate_up_w.shape[0]);
        layer.borrowed_gate_up_w = {};
        layer.borrowed_up_w = {};
        layer.borrowed_down_w = {};
        ++evicted;
        ++expert_residency_gen_;
    }
    std::clog << "MoE slabs evicted=" << evicted << " borrowed_bytes=" << expert_slab_bytes_
              << " target_kv_bytes=" << target_kv_bytes << '\n';
}

void CudaSessionManager::ensure_cuda_kv(CudaContext& ctx, int first_write, int tokens,
                                       std::span<CudaSparseMoELayer> moe_layers) {
    auto& slot = active();
    std::function<void(size_t, bool)> reclaim;
    if (expert_slab_bytes_) reclaim = [&](size_t target, bool failed) { reclaim_cuda_expert_slabs(ctx, target, failed, moe_layers); };
    if (cuda_slots_.size() == 1 || (!may_evict_paused_cuda_slots_ && cuda_kv_pool_.host_budget_bytes() == 0)) {
        cuda_kv_pool_.ensure(ctx, slot.kv, slot.pages, first_write, tokens, {}, reclaim);
        return;
    }
    if (cuda_kv_pool_.host_budget_bytes() == 0) {
        try {
            cuda_kv_pool_.ensure(ctx, slot.kv, slot.pages, first_write, tokens, {}, reclaim);
            return;
        } catch (const KvVramExhausted&) {
            // A paused prefix can be evicted and prefilled again when the active request needs its pages.
        }
    }
    std::vector<CudaSession*> paused;
    for (const auto& candidate : cuda_slots_) {
        if (candidate && candidate.get() != &slot) paused.push_back(candidate.get());
    }
    std::sort(paused.begin(), paused.end(), [](const CudaSession* a, const CudaSession* b) {
        return a->last_used < b->last_used;
    });
    std::vector<CudaKvPool::Table*> victims;
    victims.reserve(paused.size());
    for (auto* candidate : paused) victims.push_back(&candidate->pages);
    for (size_t evicted = 0;; ++evicted) {
        try {
            cuda_kv_pool_.ensure(ctx, slot.kv, slot.pages, first_write, tokens,
                cuda_kv_pool_.host_budget_bytes() ? std::span(victims).subspan(evicted) :
                                                  std::span<CudaKvPool::Table*>{}, reclaim);
            break;
        } catch (const KvVramExhausted&) {
            if (!may_evict_paused_cuda_slots_ || evicted >= paused.size()) throw;
            auto& victim = *paused[evicted];
            cuda_kv_pool_.truncate(victim.pages, 0);
            victim.cached_tokens.clear();
            victim.checkpoints.clear();
            victim.mtp_kv_valid_len = 0;
        }
    }
}

void CudaSessionManager::save_cuda_checkpoint(CudaContext& ctx, int tokens, const Tensor* logits,
                                              const Tensor& mtp_hidden) {
    auto& slot = active();
    if (slot.gdn.empty() && !mtp_hidden.data) return;
    constexpr size_t max_checkpoints = 6;
    auto& checkpoints = slot.checkpoints;
    std::erase_if(checkpoints, [=](const auto& entry) { return entry.tokens == tokens; });
    if (checkpoints.size() == max_checkpoints) {
        // A non-MTP continuation can still share the earlier verified NextN boundary.
        const auto oldest = std::find_if(checkpoints.begin(), checkpoints.end(), [&](const auto& entry) {
            return entry.tokens != slot.mtp_kv_valid_len;
        });
        checkpoints.erase(oldest);
    }
    auto snapshot = std::make_shared<GdnSnapshot>();
    snapshot->logits_bytes = logits ? logits->nbytes() : 0;
    snapshot->mtp_hidden_bytes = mtp_hidden.data ? mtp_hidden.nbytes() : 0;
    uint8_t* data = nullptr;
    FUSIONRT_CUDA_CHECK(cudaMalloc(&data, slot.ssm_bytes + snapshot->logits_bytes + snapshot->mtp_hidden_bytes));
    snapshot->data.reset(data);
    if (slot.ssm_bytes) FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data, slot.gdn_data, slot.ssm_bytes,
                                                          cudaMemcpyDeviceToDevice, ctx.stream()));
    if (logits) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data + slot.ssm_bytes, logits->ptr(), snapshot->logits_bytes,
                                            cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    if (mtp_hidden.data) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data + slot.ssm_bytes + snapshot->logits_bytes, mtp_hidden.ptr(),
                                            snapshot->mtp_hidden_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    checkpoints.push_back({tokens, std::move(snapshot)});
}

ModelMemory CudaSessionManager::cache_memory() const {
    ModelMemory result;
    result.kv_pages = cuda_kv_pool_.pages();
    result.kv_bytes = cuda_kv_pool_.bytes();
    result.kv_reserved_bytes = cuda_kv_pool_.reserved_bytes();
    result.host_kv_bytes = cuda_kv_pool_.host_bytes();
    result.host_kv_pages = cuda_kv_pool_.host_pages();
    std::unordered_set<const GdnSnapshot*> counted;
    for (const auto& slot : cuda_slots_) {
        if (!slot) continue;
        result.ssm_bytes += slot->ssm_bytes;
        result.persistent_bytes += slot->storage.capacity();
        for (const auto& checkpoint : slot->checkpoints) {
            if (!counted.insert(checkpoint.snapshot.get()).second) continue;
            result.ssm_bytes += slot->ssm_bytes;
            result.persistent_bytes += slot->ssm_bytes + checkpoint.snapshot->logits_bytes +
                                       checkpoint.snapshot->mtp_hidden_bytes;
        }
    }
    return result;
}

void CudaSessionManager::observe_prefix_prefill(int tokens, double ms) {
    const double per_token = ms / tokens;
    prefix_prefill_ms_per_token_ = prefix_prefill_ms_per_token_ > 0
        ? std::lerp(prefix_prefill_ms_per_token_, per_token, 0.125) : per_token;
}

CudaSessionManager::Prefix CudaSessionManager::prepare_prefix(CudaContext& ctx,
    const std::vector<int32_t>& prompt, bool use_fused, bool allow_cached_logits, const Tensor& logits,
    const Tensor* mtp_hidden) {
    const int length = static_cast<int>(prompt.size());
    const auto& hw = ctx.cost_hardware();
    const auto page_bytes = mtp_hidden ? cuda_kv_page_bytes() : 0;
    struct Match {
        size_t index = 0;
        int common = 0;
        int prefix = 0;
        bool append = false;
        uint64_t last_used = 0;
        std::shared_ptr<const GdnSnapshot> snapshot;
    } best;
    for (size_t i = 0; i < cuda_slots_.size(); ++i) {
        const auto& slot = cuda_slots_[i];
        if (!slot || slot->cached_fused != use_fused || slot->cached_tokens.empty()) continue;
        const int common = static_cast<int>(std::mismatch(prompt.begin(), prompt.end(), slot->cached_tokens.begin(),
                                                          slot->cached_tokens.end()).first - prompt.begin());
        const bool append = common == static_cast<int>(slot->cached_tokens.size());
        Match candidate{i, common, 0, append, slot->last_used, {}};
        if (mtp_hidden) {
            const int valid = std::min(common, slot->mtp_kv_valid_len);
            for (const auto& checkpoint : slot->checkpoints) {
                const auto& snapshot = *checkpoint.snapshot;
                if (checkpoint.tokens > valid || checkpoint.tokens <= candidate.prefix || !snapshot.mtp_hidden_bytes)
                    continue;
                const bool exact = checkpoint.tokens == length;
                if (exact && (!snapshot.logits_bytes || !allow_cached_logits)) continue;
                const int pages = 1 + (checkpoint.tokens - 1) / kKvPageTokens;
                // The pool exposes total spill occupancy. Bound the candidate's host pages by that total.
                const auto spills = std::min(cuda_kv_pool_.host_pages(), pages);
                const bool live_append = append && checkpoint.tokens == common;
                const size_t state_bytes = live_append ? 0 : slot->ssm_bytes;
                const size_t cow_bytes = !append && !exact && cuda_slots_.size() > 1 &&
                    checkpoint.tokens % kKvPageTokens != 0 ? page_bytes : 0;
                const size_t logits_bytes = exact ? snapshot.logits_bytes : 0;
                const double restore_ms = prefix_copy_ms(static_cast<size_t>(spills) * page_bytes, spills,
                    hw.pcie_h2d_gbps, hw.pcie_latency_us) + prefix_copy_ms(
                    state_bytes + logits_bytes + snapshot.mtp_hidden_bytes + cow_bytes,
                    (state_bytes != 0) + (logits_bytes != 0) + 1 + (cow_bytes != 0),
                    hw.hbm_bw_gbps, ctx.device_profile().launch_floor_us);
                if (!prefix_restore_worthwhile(restore_ms, checkpoint.tokens * prefix_prefill_ms_per_token_))
                    continue;
                candidate.prefix = checkpoint.tokens;
                candidate.snapshot = checkpoint.snapshot;
            }
        } else if (slot->gdn.empty()) {
            candidate.prefix = std::min(common, length - 1);
        } else if (append && common < length) {
            candidate.prefix = common;
        } else {
            for (const auto& checkpoint : slot->checkpoints) {
                if (checkpoint.tokens > common || checkpoint.tokens <= candidate.prefix) continue;
                if (checkpoint.tokens == length && (!checkpoint.snapshot->logits_bytes || !allow_cached_logits)) continue;
                candidate.prefix = checkpoint.tokens;
                candidate.snapshot = checkpoint.snapshot;
            }
        }
        if (mtp_hidden && candidate.prefix > 0) {
            if (std::tie(candidate.prefix, candidate.append, candidate.last_used) >
                std::tie(best.prefix, best.append, best.last_used)) best = std::move(candidate);
        } else if (!mtp_hidden && (candidate.prefix > 0 || (candidate.append && candidate.common == length))) {
            if (std::tie(candidate.append, candidate.prefix, candidate.common, candidate.last_used) >
                std::tie(best.append, best.prefix, best.common, best.last_used)) {
                best = std::move(candidate);
            }
        }
    }

    size_t target = best.index;
    if (!best.append && cuda_slots_.size() > 1) {
        size_t oldest = cuda_slots_.size();
        for (size_t i = 0; i < cuda_slots_.size(); ++i) {
            if (best.prefix > 0 && i == best.index) continue;
            const auto& slot = cuda_slots_[i];
            if (!slot || slot->cached_tokens.empty()) {
                oldest = i;
                break;
            }
            if (oldest == cuda_slots_.size() || slot->last_used < cuda_slots_[oldest]->last_used) oldest = i;
        }
        target = oldest;
    }
    if (!cuda_slots_[target]) cuda_slots_[target] = make_cuda_session();
    if (active_slot_ != target) {
        invalidate_graph_();
        active_slot_ = target;
    }
    auto& slot = active();
    slot.cached_tokens.clear();
    slot.last_used = ++slot_clock_;
    const int prefix = best.prefix;
    if (target != best.index && prefix > 0) {
        const auto& source = *cuda_slots_[best.index];
        slot.checkpoints.clear();
        for (const auto& entry : source.checkpoints) {
            if (entry.tokens <= prefix) slot.checkpoints.push_back(entry);
        }
        cuda_kv_pool_.fork(slot.pages, source.pages, prefix);
        slot.mtp_kv_valid_len = std::min(source.mtp_kv_valid_len, prefix);
    } else {
        truncate(prefix);
    }
    if (best.snapshot) {
        const auto* data = best.snapshot->data.get();
        if (slot.ssm_bytes && !(best.append && prefix == best.common)) {
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(slot.gdn_data, data, slot.ssm_bytes,
                                                cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        if (prefix == length) {
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(logits.ptr(), data + slot.ssm_bytes,
                                                logits.nbytes(), cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        if (mtp_hidden) {
            const auto row = mtp_hidden->slice_dim0(prefix - 1, 1);
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(row.ptr(), data + slot.ssm_bytes + best.snapshot->logits_bytes,
                                                row.nbytes(), cudaMemcpyDeviceToDevice, ctx.stream()));
        }
    } else if (prefix == 0) {
        zero_cuda_gdn(ctx);
    }
    return {prefix, cuda_slots_.size() > 1 && !slot.gdn.empty()};
}

CudaLoadSettings CudaSessionManager::load_settings() const {
    size_t releasable = cuda_kv_pool_.reserved_bytes() + expert_slab_bytes_;
    for (const auto& slot : cuda_slots_) if (slot) releasable += slot->storage.capacity();
    return {requested_kv_budget_, releasable, kv_commitment_tokens_, num_slots()};
}

void CudaSessionManager::configure_placement(DType dtype, size_t kv_budget) {
    cuda_kv_dtype_ = dtype;
    cuda_kv_pool_.set_kv_vram_budget_bytes(kv_budget);
    may_evict_paused_cuda_slots_ = true;
}

void CudaSessionManager::start_sessions(DType dtype, size_t slab_count) {
    cuda_kv_dtype_ = dtype;
    if (slab_count) expert_slabs_.resize(slab_count);
    cuda_slots_[0] = make_cuda_session();
}

void CudaSessionManager::clear_histories() {
    for (const auto& slot : cuda_slots_) {
        if (!slot) continue;
        slot->cached_tokens.clear();
        slot->checkpoints.clear();
        slot->mtp_kv_valid_len = 0;
    }
    prefix_prefill_ms_per_token_ = 0;
}

void CudaSessionManager::download_recurrent(CudaContext& ctx, HostRecurrentState& snapshot) const {
    for (const auto& state : active().gdn) {
        snapshot.emplace_back(ctx.download(state.convolution), ctx.download(state.recurrence));
    }
}

void CudaSessionManager::upload_recurrent(CudaContext& ctx, const HostRecurrentState& snapshot) {
    for (size_t i = 0; i < snapshot.size(); ++i) {
        ctx.upload_into(snapshot[i].first, active().gdn[i].convolution);
        ctx.upload_into(snapshot[i].second, active().gdn[i].recurrence);
    }
}

void CudaSessionManager::save_recurrent(CudaContext& ctx, void* dst) const {
    if (active().ssm_bytes) FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, active().gdn_data,
        active().ssm_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
}

void CudaSessionManager::restore_recurrent(CudaContext& ctx, const void* src) {
    if (active().ssm_bytes) FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(active().gdn_data, src,
        active().ssm_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
}

}  // namespace fusionrt
#endif
