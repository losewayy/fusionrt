#pragma once

#include "model_loader.h"
#include "vulkan_weights.h"
#include "fusionrt/vulkan_dispatch.h"
#include "fusionrt/sampler.h"
#include "../core/cpu_moe.h"
#include "../core/spec_cost.h"

#include <chrono>
#include <cstring>

namespace fusionrt::vulkan_detail {

inline CostHardware moe_hardware(const VkContext& ctx) {
    const auto& p = ctx.device_profile();
    CostHardware hw;
    hw.hbm_bw_gbps = p.measured_hbm_bw_gbps;
    hw.cpu_q4_gemv_gbps = p.cpu_q4_gemv_gbps > 0 ? p.cpu_q4_gemv_gbps : p.cpu_ddr_bw_gbps;
    hw.pcie_h2d_gbps = p.pcie_h2d_gbps;
    hw.pcie_latency_us = p.pcie_latency_us;
    return hw;
}

inline std::vector<MoeLayerCost> moe_costs(const model_io::LoadedModelRaw& raw) {
    std::vector<MoeLayerCost> costs(raw.layers.size());
    for (size_t i = 0; i < raw.layers.size(); ++i) {
        const auto& layer = raw.layers[i];
        if (layer.router_w.bytes().empty()) continue;
        auto& cost = costs[i];
        cost.experts = static_cast<int>(layer.router_w.desc.shape[0]);
        cost.hidden = static_cast<int>(layer.router_w.desc.shape[1]);
        cost.top_k = raw.cfg.moe.top_k;
        cost.mapped_slots = true;
        cost.expert_tensor_bytes = {layer.experts_gate_up.bytes().size() / cost.experts,
                                    layer.experts_up.bytes().size() / cost.experts,
                                    layer.experts_down.bytes().size() / cost.experts};
    }
    return costs;
}

inline VkDeviceSize moe_scratch_reserve(const VkContext& ctx, const model_io::LoadedModelRaw& raw,
                                              std::span<const MoeLayerCost> costs) {
    const auto align = ctx.info().min_storage_offset_align;
    const auto aligned = [align](VkDeviceSize bytes) { return (bytes + align - 1) & ~(align - 1); };
    constexpr int rows = MtpDepthController::max_depth + 1;
    const auto& d = raw.cfg.dense;
    const auto& g = raw.cfg.gdn;
    const auto& m = raw.cfg.moe;
    const int64_t h = raw.tok_embeddings.desc.shape[1];
    const auto q = int64_t{d.num_attention_heads} * d.head_dim;
    const auto kv = int64_t{d.num_key_value_heads} * d.head_dim;
    const auto v = int64_t{g.num_v_heads} * g.head_v_dim;
    const auto conv = 2LL * g.num_k_heads * g.head_k_dim + v;
    const auto attn = std::max<int64_t>(9 * q + 5 * kv, conv + 2 * v + 2 * g.num_v_heads);
    const auto ffn = std::max<int64_t>(5LL * d.intermediate_size,
        m.num_experts + 2LL * m.top_k + int64_t{m.top_k} * m.intermediate_size + 3LL * m.shared_intermediate_size);
    // Keep a decode/verify tile and one expert swap available before filling the weight arena.
    auto bytes = aligned(rows * sizeof(float) * (12 * h + attn + ffn + 2LL * raw.cfg.vocab_size)) + 64 * align;
    if (m.dynamic_cache) {
        VkDeviceSize staging = 0;
        for (const auto& cost : costs)
            staging = std::max(staging, aligned(cost.experts * sizeof(int32_t)) +
                aligned(cost.expert_tensor_bytes[0]) + aligned(cost.expert_tensor_bytes[1]) +
                aligned(cost.expert_tensor_bytes[2]));
        bytes += staging;
    }
    return bytes;
}

inline size_t model_budget(VkContext& ctx) {
    const auto& p = ctx.device_profile();
    const auto& arena = ctx.weight_arena();
    if (arena.used()) {
        const auto align = ctx.info().min_storage_offset_align;
        return arena.capacity() - std::min(arena.capacity(), (arena.used() + align - 1) & ~(align - 1));
    }
    auto total = ctx.info().device_local_vram_bytes;
    if (p.vram_total_mb > 0 && p.vram_total_mb * 1048576.0 < static_cast<double>(total))
        total = static_cast<uint64_t>(p.vram_total_mb * 1048576.0);
    return std::max<VkDeviceSize>(arena.capacity(), wddm_model_available(total, ctx.scratch_arena().capacity(), 0));
}

inline HostTensor pack_experts(const HostTensor& source, std::span<const int32_t> slots) {
    if (source.bytes().empty()) return {};
    const auto& s = source.desc.shape;
    auto packed = HostTensor::create({static_cast<int64_t>(slots.size()), s[1], s[2]}, source.desc.dtype);
    const auto unit = source.bytes().size() / s[0];
    for (size_t slot = 0; slot < slots.size(); ++slot)
        std::memcpy(packed.storage.data() + slot * unit, source.bytes().data() + slots[slot] * unit, unit);
    return packed;
}

inline void upload_experts(VkContext& ctx, VulkanMoeWeights& layer, const model_io::LoadedLayerRaw& raw,
                           int resident, std::span<const int32_t> ranking) {
    const int experts = layer.cfg.num_experts;
    layer.cfg.gpu_resident_experts = resident;
    if (resident < experts) {
        layer.cpu_gate = raw.experts_gate_up;
        layer.cpu_up = raw.experts_up;
        layer.cpu_down = raw.experts_down;
        if (!raw.expert_gate_scale.bytes().empty()) layer.cpu_gate_scale = model_io::to_f32_host(raw.expert_gate_scale);
        if (!raw.expert_up_scale.bytes().empty()) layer.cpu_up_scale = model_io::to_f32_host(raw.expert_up_scale);
    }
    if (!resident) return;
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto upload = [&](const HostTensor& source) {
        if (source.bytes().empty()) return Tensor{};
        return ctx.upload(resident == experts ? source : pack_experts(source, ranking.first(resident)), true);
    };
    layer.gpu_gate_up_w = upload(raw.experts_gate_up);
    layer.gpu_up_w = upload(raw.experts_up);
    layer.gpu_down_w = upload(raw.experts_down);
    if (resident == experts) return;

    const auto slots = ranking.first(resident);
    layer.expert_to_slot = moe_expert_to_slot(experts, slots);
    HostTensor map({experts}, DType::I32);
    std::copy(layer.expert_to_slot.begin(), layer.expert_to_slot.end(), map.data<int32_t>());
    layer.slot_map = ctx.upload(map, true);
    if (layer.cfg.dynamic_cache) {
        layer.cache.emplace(experts, slots);
        layer.cache_hardware = moe_hardware(ctx);
        if (layer.cache_hardware.pcie_h2d_gbps <= 0) {
            const double ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
            const auto bytes = layer.gpu_gate_up_w.nbytes() + layer.gpu_up_w.nbytes() + layer.gpu_down_w.nbytes();
            if (ns > 0) layer.cache_hardware.pcie_h2d_gbps = bytes / ns;
        }
    }
}

inline size_t expert_bytes(const HostTensor& matrix) {
    return matrix.bytes().empty() ? 0 : matrix.bytes().size() / matrix.desc.shape[0];
}

inline size_t cache_copy_bytes(const VulkanMoeWeights& layer) {
    return expert_bytes(layer.cpu_gate) + expert_bytes(layer.cpu_up) + expert_bytes(layer.cpu_down);
}

inline void update_expert_cache(VkContext& ctx, const VulkanMoeWeights& layer, std::span<const int32_t> routes) {
    if (!layer.cache || !MoeDecodeScope::active_for(&ctx)) return;
    const size_t bytes = cache_copy_bytes(layer);
    const auto move = layer.cache->observe(routes, layer.expert_to_slot, layer.cache_hardware, bytes,
                                           layer.gpu_up_w.vk_buffer ? 3 : 2);
    if (!move) return;

    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    struct Restore { VkArena& arena; VkDeviceSize mark; ~Restore() { arena.reset(mark); } } restore{arena, mark};
    VulkanDispatch::Batch batch{ctx};
    const auto copy = [&](const HostTensor& source, const Tensor& target) {
        if (source.bytes().empty()) return;
        const auto unit = expert_bytes(source);
        auto part = source.slice_dim0(move->expert, 1);
        const auto staging = ctx.upload(part, false);
        VulkanDispatch::append_copy(batch, staging, target, {{0, unit * move->slot, unit}});
    };
    copy(layer.cpu_gate, layer.gpu_gate_up_w);
    copy(layer.cpu_up, layer.gpu_up_w);
    copy(layer.cpu_down, layer.gpu_down_w);
    auto updated = layer.expert_to_slot;
    const auto old = std::find(updated.begin(), updated.end(), move->slot);
    *old = -1;
    updated[move->expert] = move->slot;
    HostTensor map({layer.cfg.num_experts}, DType::I32);
    std::copy(updated.begin(), updated.end(), map.data<int32_t>());
    const auto staging = ctx.upload(map, false);
    VulkanDispatch::append_copy(batch, staging, layer.slot_map, {{0, 0, map.desc.nbytes()}});
    ctx.dispatch_batched(batch.ops, false);
    layer.cache->commit(*move, layer.expert_to_slot, bytes);
}

inline void lookup_draft(const std::vector<int32_t>& history, int limit, std::vector<int32_t>& input) {
    const int count = static_cast<int>(history.size());
    for (int n = std::min(3, count); n >= 2; --n) {
        for (int end = count - 1; end >= n; --end) {
            if (std::equal(history.end() - n, history.end(), history.begin() + end - n)) {
                input.insert(input.end(), history.begin() + end, history.begin() + end + std::min(limit, count - end));
                return;
            }
        }
    }
}

inline int32_t verify_sample(const Sampler::Distribution& target, const Sampler::Distribution& draft,
                             int32_t proposal, std::mt19937_64& rng) {
    // Draft entries are sorted by token ID for residual probability lookups.
    const auto probability = [&](int32_t id) {
        const auto it = std::lower_bound(draft.begin(), draft.end(), id,
                                         [](const auto& entry, int32_t token) { return entry.second < token; });
        return it == draft.end() || it->second != id ? 0.0 : it->first;
    };
    const auto candidate = std::find_if(target.begin(), target.end(),
        [proposal](const auto& entry) { return entry.second == proposal; });
    if (candidate != target.end() &&
        std::generate_canonical<double, 53>(rng) * probability(proposal) < candidate->first)
        return proposal;

    Sampler::Distribution residual;
    residual.reserve(target.size());
    for (const auto& [weight, id] : target) {
        const double difference = weight - probability(id);
        if (difference > 0) residual.emplace_back(difference, id);
    }
    return Sampler::draw(residual, rng);
}

inline double draft_confidence(std::span<const float> logits, int32_t best) {
    double mass = 0;
    for (const float value : logits) mass += std::exp(static_cast<double>(value) - logits[best]);
    return 1.0 / mass;
}

}  // namespace fusionrt::vulkan_detail
