#pragma once

#include "test_vulkan_v2.h"

namespace fusionrt::test {
namespace {

void test_vk_v3_tiered_moe(VkContext& vk) {
    constexpr int H = 64, I = 64, E = 4, V = 8;
    std::mt19937_64 rng(17);
    auto raw = v2_raw(H, V, 1, rng);
    raw.cfg.arch = "moe";
    raw.cfg.moe.hidden_size = H;
    raw.cfg.moe.num_experts = E;
    raw.cfg.moe.top_k = 2;
    raw.cfg.moe.intermediate_size = I;
    raw.tok_embeddings = model_io::make_ones_f32({V, H});
    auto& layer = raw.layers.front();
    layer.kind = 2;
    layer.attn_norm = model_io::make_ones_f32({H});
    layer.router_w = model_io::make_zeros_f32({E, H});
    std::fill_n(layer.router_w.data<float>(), H, .015f);
    std::fill_n(layer.router_w.data<float>() + 3 * H, H, .02f);
    const auto weights = [&](int rows, int cols) {
        return model_io::quantize_host_matrix(model_io::make_rand_f32({E, rows, cols}, rng, .03f), DType::Q4_0);
    };
    layer.experts_gate_up = weights(2 * I, H);
    layer.experts_down = weights(H, I);
    const VulkanLoadHooks hooks;
    VulkanSessionManager sessions;
    VulkanModelWeights all(raw.cfg);
    all.upload(vk, raw, hooks, DType::F32);
    VulkanExecutor gpu(all, sessions);
    const auto reference = gpu.forward_token_vulkan(vk, 2, 0);
    auto cpu = v2_moe(raw, layer, v2_row(raw.tok_embeddings, 2));
    v2_norm(cpu, raw.final_norm_w.data<float>(), raw.cfg.dense.rms_norm_eps);
    cpu = v2_linear(cpu, raw.lm_head_w);
    require_vk_close(DType::Q4_0, "full GPU MoE vs CPU", reference, cpu, 5e-5);

    auto profile = vk.device_profile();
    auto cache_profile = profile;
    cache_profile.cpu_q4_gemv_gbps = 1;
    cache_profile.pcie_h2d_gbps = 100;
    vk.set_device_profile(cache_profile);
    raw.cfg.moe.dynamic_cache = true;
    VulkanModelWeights mixed(raw.cfg, 2);
    mixed.upload(vk, raw, hooks, DType::F32);
    VulkanExecutor tiered(mixed, sessions);
    const auto& slots = mixed.moe_layers().front();
    if (!slots.cache || slots.cfg.gpu_resident_experts != 2 || slots.expert_to_slot[3] != -1 ||
        slots.gpu_gate_up_w.dtype != DType::Q4_0 || slots.gpu_gate_up_w.nbytes() != layer.experts_gate_up.bytes().size() / 2)
        throw std::runtime_error("Vulkan Tiered MoE did not upload two compact expert slots");
    for (int pos = 0; pos < 8; ++pos) {
        const auto actual = tiered.forward_token_vulkan(vk, 2, pos);
        require_vk_close(DType::Q4_0, "Tiered MoE cache parity", actual, reference, 5e-5);
        require_vk_close(DType::Q4_0, "Tiered MoE CPU reference", actual, cpu, 5e-5);
    }
    const auto device_map = vk.download(slots.slot_map);
    if (slots.cache->moves() != 1 || slots.expert_to_slot[3] < 0 ||
        !std::equal(slots.expert_to_slot.begin(), slots.expert_to_slot.end(), device_map.data<int32_t>()))
        throw std::runtime_error("Vulkan expert cache did not admit the repeatedly routed expert");

    VulkanModelWeights cold(raw.cfg, 0);
    cold.upload(vk, raw, hooks, DType::F32);
    VulkanExecutor host(cold, sessions);
    require_vk_close(DType::Q4_0, "CPU-only expert layer", host.forward_token_vulkan(vk, 2, 0), cpu, 5e-5);
    vk.set_device_profile(profile);
}

HybridModelConfig v3_config(bool hybrid) {
    HybridModelConfig cfg;
    cfg.arch = hybrid ? "hybrid" : "dense";
    cfg.vocab_size = 33;
    cfg.num_layers = hybrid ? 4 : 1;
    cfg.dense.max_seq_len = 40;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 32;
    cfg.dense.num_attention_heads = 4;
    cfg.dense.num_key_value_heads = 2;
    cfg.dense.head_dim = 8;
    cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 64;
    cfg.gdn.num_k_heads = 2;
    cfg.gdn.num_v_heads = 4;
    cfg.gdn.head_k_dim = cfg.gdn.head_v_dim = 8;
    return cfg;
}

void test_vk_v3_verify(VkContext& vk, bool hybrid) {
    FusionModel model(v3_config(hybrid));
    model.init_synthetic_vulkan(vk, 813);
    const std::array<int32_t, 4> tokens{6, 3, 2, 1};
    model.reset_kv_and_state_vulkan(vk);
    model.forward_token_vulkan(vk, 9, 0);
    const auto batch = model.forward_verify_vulkan(vk, tokens, 1);
    const auto tail = model.forward_token_vulkan(vk, 7, 5);
    model.reset_kv_and_state_vulkan(vk);
    model.forward_token_vulkan(vk, 9, 0);
    V2Vector steps;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto row = model.forward_token_vulkan(vk, tokens[i], static_cast<int>(i) + 1);
        steps.insert(steps.end(), row.begin(), row.end());
    }
    require_vk_close(DType::F32, "batched Vulkan verify", batch, steps, 2e-5);
    require_vk_close(DType::F32, "Vulkan verify continuation", tail, model.forward_token_vulkan(vk, 7, 5), 2e-5);
}

void test_vk_v3_wide_head(VkContext& vk) {
    constexpr int H = 4, V = 66000;
    std::mt19937_64 rng(132);
    const auto raw = v2_raw(H, V, 0, rng);
    VulkanV2Model model(vk, raw);
    const std::array<int32_t, 2> tokens{3, V - 1};
    const auto rows = model.executor.forward_verify_vulkan(vk, tokens, 0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto input = v2_row(raw.tok_embeddings, tokens[i]);
        v2_norm(input, raw.final_norm_w.data<float>(), raw.cfg.dense.rms_norm_eps);
        require_vk_close(DType::F32, "Vulkan wide vocabulary", std::span(rows).subspan(i * V, V),
                         v2_linear(input, raw.lm_head_w), 2e-5);
    }
}

void test_vk_v3_speculation(VkContext& vk) {
    auto cfg = v3_config(true);
    FusionModel target(cfg), draft(cfg);
    target.enable_native_mtp(vk, true);
    target.init_synthetic_vulkan(vk, 812);
    draft.init_synthetic_vulkan(vk, 521);
    if (!target.has_native_mtp(vk)) throw std::runtime_error("Synthetic Vulkan MTP head failed to load");

    std::vector<int32_t> prompt;
    for (int i = 0; i < 4; ++i) {
        prompt.insert(prompt.end(), {1, 2});
        const auto next = target.generate_vulkan(vk, prompt, 1);
        prompt.push_back(next.back());
        prompt.push_back((next.back() + 1) % cfg.vocab_size);
    }
    prompt.insert(prompt.end(), {1, 2});
    constexpr int generated = 14;
    const auto baseline = target.generate_vulkan(vk, prompt, generated);
    const auto verify = [&](const GenerationOptions& options) {
        GenerationStats stats;
        const auto ids = target.generate_vulkan(vk, prompt, generated, &stats, false, 0, -1, options);
        if (ids != baseline || !stats.speculative_active || !stats.spec_proposed)
            throw std::runtime_error("Vulkan speculative mode changed tokens or did not exercise verification");
        return stats;
    };
    GenerationOptions options;
    options.spec_ngram = 4;
    const auto ngram_stats = verify(options);
    if (ngram_stats.spec_accepted >= ngram_stats.spec_proposed)
        throw std::runtime_error("N-gram fixture did not exercise target recurrent rollback");
    options = {};
    options.draft_model = &draft;
    const auto draft_stats = verify(options);
    if (draft_stats.spec_accepted >= draft_stats.spec_proposed)
        throw std::runtime_error("Draft fixture did not exercise recurrent rollback");
    options = {};
    options.spec_mtp = true;
    options.spec_p_min = 0.f;
    options.draft_tokens = 3;
    if (!verify(options).mtp_rewound_kv_rows)
        throw std::runtime_error("MTP fixture did not exercise NextN recurrent rollback");
    options.draft_tokens = 0;
    verify(options);
}

void test_vk_v3_recurrent_snapshot(VkContext& vk) {
    auto cfg = v3_config(true);
    cfg.num_layers = 1;
    VulkanSessionManager sessions;
    VulkanModelWeights weights(cfg);
    weights.init_synthetic_vulkan(vk, 47, {{}, [&](const auto& gdn, int dim) { sessions.allocate_gdn(vk, gdn, dim); }, {}});
    VulkanExecutor executor(weights, sessions);
    sessions.reset_kv_and_state_vulkan(vk);
    executor.forward_token_vulkan(vk, 3, 0);
    const auto& gdn = sessions.gdn(0);
    const auto before_conv = vk.download(gdn.conv);
    const auto before_ssm = vk.download(gdn.ssm);
    sessions.snapshot_gdn_state(vk, 0, 1);
    const std::array<int32_t, 4> rejected{1, 2, 3, 4};
    executor.forward_verify_vulkan(vk, rejected, 1);
    if (vk.download(gdn.ssm).storage == before_ssm.storage)
        throw std::runtime_error("Vulkan rollback test did not change recurrent state");
    sessions.restore_gdn_state(vk, 0, 1);
    if (vk.download(gdn.conv).storage != before_conv.storage || vk.download(gdn.ssm).storage != before_ssm.storage)
        throw std::runtime_error("Vulkan recurrent snapshot did not restore the exact state");
}

void test_vk_v3_speculative_json(VkContext& vk) {
    auto cfg = v3_config(true);
    cfg.num_layers = 1;
    BpeTokenizer tokenizer;
    tokenizer.init_byte_fallback();
    cfg.vocab_size = tokenizer.vocab_size();
    FusionModel model(cfg);
    model.enable_native_mtp(vk, true);
    model.init_synthetic_vulkan(vk, 445);
    model.set_vulkan_speculation(0, nullptr, 3, true, 0.f);
    const JsonConstraint grammar(tokenizer, cfg.vocab_size, JsonRoot::null);
    GenerationOptions options;
    options.sampling.temperature = .8f;
    options.json_constraint = &grammar;
    const std::vector<int32_t> prompt{2, 9};
    GenerationStats stats;
    const auto result = model.generate_vulkan(vk, prompt, 8, &stats, false, 0, -1, options);
    auto state = grammar.initial_state();
    for (size_t i = prompt.size(); i < result.size(); ++i)
        if (!grammar.accept(state, result[i])) throw std::runtime_error("Vulkan MTP sampled outside the JSON grammar");
    if (!state.complete() || !stats.json_complete || !stats.spec_proposed)
        throw std::runtime_error("Vulkan stochastic MTP did not finish constrained speculation");
}

void test_vulkan_v3() {
    VkContext vk(0, 32, 32);
    const auto run = [&](auto check) {
        const auto mark = vk.weight_arena().mark();
        check(vk);
        vk.weight_arena().reset(mark);
    };
    run(test_vk_v3_tiered_moe);
    run([](VkContext& ctx) { test_vk_v3_verify(ctx, false); });
    run([](VkContext& ctx) { test_vk_v3_verify(ctx, true); });
    run(test_vk_v3_wide_head);
    run(test_vk_v3_recurrent_snapshot);
    run(test_vk_v3_speculation);
    run(test_vk_v3_speculative_json);
    std::cout << "  [PASS] Vulkan V3 tiered MoE, expert cache, batched verify and speculative decoding\n";
}

}  // namespace
}  // namespace fusionrt::test
