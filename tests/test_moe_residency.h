#pragma once

#include "test_support.h"
#include "../src/core/moe_residency.h"

namespace fusionrt::test {

struct MoeProfileFixture {
    std::string path = "test-moe-residency.jsonl";
    ~MoeProfileFixture() { std::error_code error; std::filesystem::remove(path, error); }
};

inline void test_decode_heat_placement(const CostHardware& hw) {
    constexpr size_t unit = 1024 * 1024;
    std::array<MoeLayerCost, 3> costs;
    costs.fill(MoeLayerCost{4, 1, 512, {unit, 0, 0}, true});
    const MoeProfileFixture fixture;
    {
        std::ofstream trace(fixture.path);
        for (int row = 0; row < 32; ++row)
            trace << R"({"type":"routes","layer":0,"phase":"prompt","experts":[0]})" << '\n';
        for (int row = 0; row < 4; ++row) {
            trace << R"({"type":"routes","layer":0,"phase":"decode","experts":[3]})" << '\n';
            trace << R"({"type":"routes","layer":1,"phase":"decode","experts":[1]})" << '\n';
            trace << R"({"type":"routes","layer":1,"phase":"decode","experts":[2]})" << '\n';
        }
        for (int expert : {0, 2, 3})
            trace << "{\"layer\":2,\"phase\":\"decode\",\"experts\":[" << expert << "]}\n";
        trace << R"({"type":"summary","layer":0,"decode_rows":4})" << '\n';
    }
    const auto heat = moe_layer_heat(costs, fixture.path, false);
    const size_t budget = 3 * costs.front().gpu_bytes(2);
    const auto ranked = select_moe_placement(costs, budget, hw, 50, heat);
    if (ranked.gpu_experts != std::vector<int>{1, 2, 3} || ranked.ranked_experts[0][0] != 3 ||
        heat[0].cumulative_hits[1] != 1.0 || ranked.choice != 3 ||
        ranked.candidate_us[3] >= ranked.candidate_us[0] || ranked.candidate_us[3] >= ranked.candidate_us[2])
        throw std::runtime_error("Decode heat must beat uniform and prefix-boundary placement within the same budget");
    const auto prior = moe_layer_heat(costs, {}, true);
    const auto balanced = select_moe_placement(costs, budget, hw, 50, prior, true);
    if (balanced.gpu_experts != std::vector<int>{2, 2, 2} || prior[0].cumulative_hits[1] <= 0.25)
        throw std::runtime_error("Dynamic placement without a trace must balance the skewed prior across layers");
    std::cout << "  [PASS] Decode route heat ignores prompt rows and improves expert placement within budget\n";
}

inline void test_moe_admission() {
    constexpr size_t bytes = 1024;
    CostHardware hw;
    hw.cpu_q4_gemv_gbps = 10;
    hw.pcie_h2d_gbps = 20;
    if (moe_cache_move_pays(hw, bytes, 0, 1, 0.5) || !moe_cache_move_pays(hw, bytes, 0, 1, 0.51))
        throw std::runtime_error("Expert move must be strictly cheaper than future misses");
    hw.pcie_latency_us = 100.;
    if (moe_cache_move_pays(hw, bytes, 0, 1, 0.51))
        throw std::runtime_error("Expert move must pay for PCIe latency");

    const std::array<int32_t, 2> slots{0, 1}, cold{2, 3}, active{0, 1};
    auto mapping = moe_expert_to_slot(4, slots);
    MoeDynamicCache cache(4, slots, 4);
    auto observe = [&](std::span<const int32_t> routes) { return cache.observe(routes, mapping, hw, bytes, 2); };
    for (int tick = 0; tick < 5; ++tick)
        if (observe(cold)) throw std::runtime_error("Hysteresis alone must not admit an expensive move");
    hw.pcie_latency_us = 0.;
    const std::array<int32_t, 3> full_routes{0, 1, 2};
    if (observe(full_routes)) throw std::runtime_error("Cannot evict an expert used by this token");
    const auto move = observe(cold);
    if (!move || move->expert != 2 || move->slot != 0)
        throw std::runtime_error("Expected the hottest cold expert to replace the coldest inactive slot");
    cache.commit(*move, mapping, bytes);
    if (mapping != std::vector<int32_t>{-1, 1, 0, -1} || cache.moves() != 1 ||
        cache.moved_bytes() != bytes + 4 * sizeof(int32_t))
        throw std::runtime_error("Admission must update the slot map and transfer counters");

    MoeDynamicCache decaying(4, slots, 1);
    mapping = moe_expert_to_slot(4, slots);
    for (int tick = 0; tick < 8; ++tick) (void)decaying.observe(active, mapping, hw, bytes, 2);
    for (int tick = 0; tick < 16; ++tick)
        if (decaying.observe(cold, mapping, hw, bytes, 2))
            throw std::runtime_error("Old heat must decay and repeated misses below the hysteresis floor must stay cold");
    std::cout << "  [PASS] MoE cache admission enforces transfer cost, decay, hysteresis and active-slot protection\n";
}

inline void test_cpu_moe_slot_parity() {
    std::mt19937_64 rng(142);
    const auto gate = rand_f32({3, 4, 4}, rng, 0.3f), down = rand_f32({3, 4, 2}, rng, 0.3f);
    const auto first = gate.slice_dim0(1, 2), last = down.slice_dim0(1, 2);
    const std::array<int32_t, 3> routes{2, 0, 1};
    const std::array<float, 3> weights{0.2f, 0.5f, 0.3f};
    const std::array<float, 4> input{0.25f, 1.f, -0.5f, 2.f};
    const std::array<int32_t, 1> initial{0};
    auto mapping = moe_expert_to_slot(3, initial);
    CpuMoeWorkspace work;
    auto run = [&](const HostTensor& g, const HostTensor& d, const auto& ids, int cold, const int32_t* map) {
        std::array<float, 4> output{};
        const int count = cpu_moe_accumulate_layer(nullptr, work, g, {}, d, ids.data(), weights.data(),
            cold, cold, 1, 3, input.data(), output.data(), CpuMoeMode::Scalar, false, nullptr, nullptr, map);
        if (count != 2) throw std::runtime_error("CPU slot map must route only the cold experts");
        return output;
    };
    if (run(gate, down, routes, 0, mapping.data()) != run(first, last, routes, 1, nullptr))
        throw std::runtime_error("CPU identity map changed cold output");
    MoeDynamicCache cache(3, initial);
    cache.commit({2, 0}, mapping, gate.bytes().size() / 3 + down.bytes().size() / 3);
    auto masked = routes;
    masked[0] = -1;
    if (run(gate, down, routes, 0, mapping.data()) != run(gate, down, masked, 0, nullptr))
        throw std::runtime_error("CPU remapped slot changed cold output");
    std::cout << "  [PASS] CPU cold expert outputs match exactly before and after a slot swap\n";
}

#ifdef FUSIONRT_ENABLE_CUDA
inline void test_cuda_moe_slot_swap(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    for (const bool separate : {false, true}) {
        std::mt19937_64 rng(142);
        const auto gate = rand_f32({3, separate ? 2 : 4, 4}, rng, 0.3f);
        const auto up = separate ? rand_f32({3, 2, 4}, rng, 0.3f) : HostTensor{};
        const auto down = rand_f32({3, 4, 2}, rng, 0.3f);
        HostTensor router({3, 4}, DType::F32);
        router.data<float>()[4] = 4.f;
        router.data<float>()[8] = -4.f;
        auto scales = make_ones_f32({3});
        scales.data<float>()[1] = 0.7f;
        scales.data<float>()[2] = 1.4f;
        CudaSparseMoELayer cpu;
        cpu.cfg.hidden_size = 4;
        cpu.cfg.intermediate_size = 2;
        cpu.cfg.num_experts = 3;
        cpu.cfg.top_k = 1;
        cpu.cfg.gpu_resident_experts = 0;
        cpu.cfg.has_shared_expert = false;
        cpu.norm_w = ctx.upload(make_ones_f32({4}), false);
        cpu.router_w = ctx.upload(router, false);
        cpu.gate_scale = cpu.up_scale = ctx.upload(scales, false);
        cpu.cpu_gate_scale = cpu.cpu_up_scale = scales;
        cpu.cpu_gate_up_w = gate;
        cpu.cpu_up_w = up;
        cpu.cpu_down_w = down;
        auto gpu = cpu;
        gpu.cfg.gpu_resident_experts = 3;
        gpu.gpu_gate_up_w = ctx.upload(gate, false);
        gpu.gpu_down_w = ctx.upload(down, false);
        if (separate) gpu.gpu_up_w = ctx.upload(up, false);
        auto mapped = cpu;
        mapped.cfg.gpu_resident_experts = 1;
        mapped.gpu_gate_up_w = ctx.upload(gate.slice_dim0(2, 1), false);
        mapped.gpu_down_w = ctx.upload(down.slice_dim0(2, 1), false);
        if (separate) mapped.gpu_up_w = ctx.upload(up.slice_dim0(2, 1), false);
        const std::array<int32_t, 1> slots{2};
        mapped.expert_to_slot = moe_expert_to_slot(3, slots);
        HostTensor host_map({3}, DType::I32);
        std::copy_n(mapped.expert_to_slot.data(), 3, host_map.data<int32_t>());
        mapped.d_expert_to_slot = ctx.upload(host_map, false);
        mapped.dynamic_cache = std::make_shared<MoeDynamicCache>(3, slots);
        mapped.measured_h2d_gbps = 100;
        mapped.measured_cpu_gbps = 1;
        auto input = make_ones_f32({1, 4});
        auto x = ctx.upload(input, false);
        auto run = [&](const CudaSparseMoELayer& layer, float sign) {
            input.data<float>()[0] = sign;
            ctx.upload_into(input, x);
            layer.forward_prefill_cuda(ctx, x, 1);
            const auto result = ctx.download(x);
            return std::vector<float>(result.data<float>(), result.data<float>() + 4);
        };
        if (run(mapped, -1.f) != run(gpu, -1.f)) throw std::runtime_error("CUDA slot must use logical expert scales");
        for (int step = 0; step < 6; ++step)
            if (run(mapped, 1.f) != run(cpu, 1.f)) throw std::runtime_error("Mapped cold expert parity");
        if (mapped.dynamic_cache->moves() != 0) throw std::runtime_error("Single-row prefill must never move experts");
        {
            const MoeDecodeScope decode(&ctx);
            for (int step = 0; step < 32 && mapped.dynamic_cache->moves() == 0; ++step) (void)run(mapped, 1.f);
        }
        const auto uploaded = ctx.download(mapped.d_expert_to_slot);
        if (mapped.dynamic_cache->moves() != 1 || mapped.expert_to_slot != std::vector<int32_t>{-1, 0, -1} ||
            !std::equal(mapped.expert_to_slot.begin(), mapped.expert_to_slot.end(), uploaded.data<int32_t>()) ||
            run(mapped, 1.f) != run(gpu, 1.f) || run(mapped, -1.f) != run(cpu, -1.f))
            throw std::runtime_error("CUDA slot swap must update weights, both maps and CPU/GPU ownership");
        arena.reset(mark);
    }
    std::cout << "  [PASS] CUDA and CPU exact expert parity through packed and split weight slot swaps\n";
}
#endif

}  // namespace fusionrt::test
