#include "test_support.h"
#include "../src/models/cuda_session.h"

#ifdef FUSIONRT_ENABLE_CUDA
namespace fusionrt {
decltype(CudaKvPool::allocate_)& CudaReviewTestAccess::allocator(FusionModel& model) {
    return model.sessions_->cuda_kv_pool_.allocate_;
}
}  // namespace fusionrt
#endif

namespace fusionrt::test {

void test_wave3() {
    std::cout << "Testing sampling, UTF-8, device profile, GGUF biases and prefix reuse...\n";
    auto check = [](bool passed, const char* error) {
        if (!passed) throw std::runtime_error(error);
    };
    const float logits[] = {3.0f, 2.0f, -4.0f};
    std::mt19937_64 rng(42);
    SamplingParams params;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Greedy sampler");
    params.repeat_penalty = 2.0f;
    check(Sampler::sample(logits, 3, {0, 0}, params, rng) == 1, "Repetition penalty");
    params.repeat_penalty = 1.0f;
    params.temperature = 0.7f;
    params.top_k = 1;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Top-k sampler");
    params.top_k = 0;
    params.top_p = 0.1f;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Top-p sampler");
    params.top_p = 1.0f;
    params.min_p = 0.5f;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Min-p sampler");
    params.min_p = 0.0f;
    std::mt19937_64 first(42), second(42);
    bool saw_second = false;
    for (int i = 0; i < 32; ++i) {
        int32_t a = Sampler::sample(logits, 3, {}, params, first);
        check(a == Sampler::sample(logits, 3, {}, params, second), "Seeded sampler reproducibility");
        saw_second |= a == 1;
    }
    check(saw_second, "Temperature sampling yielded only the argmax");
    Utf8StreamDecoder utf8;
    check(utf8.push("A\xe4").compare("A") == 0 && utf8.push("\xb8").empty() &&
          utf8.push("\xad" "B") == "\xe4\xb8\xad" "B" && utf8.finish().empty(), "UTF-8 token split");
    check(utf8.push("\xed\xa0\x80") == "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd", "Invalid UTF-8");

    struct Fixtures {
        std::filesystem::path profile = "test-profile.json";
        std::filesystem::path model = "test-model.gguf";
        ~Fixtures() {
            std::error_code ec;
            std::filesystem::remove(profile, ec);
            std::filesystem::remove(model, ec);
        }
    } files;
    std::ofstream(files.profile) << R"({"schema_version":"device_profile.v1","mmq_crossover_m":7,
        "recommend_cuda_graph":true,"empirical_l2_bytes":33554432,
        "preferred_gemv_warps":4,"preferred_gemv_rows_per_warp":2})";
    const auto profile = DeviceProfile::load(files.profile.string());
    check(profile.mmq_crossover_m == 7 && profile.recommend_cuda_graph &&
          profile.empirical_l2_bytes == 33554432 && profile.preferred_gemv_warps == 4 &&
          profile.preferred_gemv_rows_per_warp == 2, "Device profile fields");

#ifdef FUSIONRT_ENABLE_CUDA
    CudaContext ctx(0, 64, 64);
    std::map<std::string, GgufMetaValue> metadata{
        {"general.architecture", std::string("qwen2")}, {"qwen2.block_count", uint32_t{1}},
        {"qwen2.embedding_length", uint32_t{256}}, {"qwen2.attention.head_count", uint32_t{8}},
        {"qwen2.attention.head_count_kv", uint32_t{2}}, {"qwen2.feed_forward_length", uint32_t{256}},
        {"qwen2.rope.freq_base", 12345.0f}, {"qwen2.rope.dimension_count", uint32_t{16}},
        {"tokenizer.ggml.model", std::string("gpt2")},
        {"tokenizer.ggml.tokens", std::vector<std::string>{"<|im_start|>", "<|im_end|>", "a"}},
        {"tokenizer.chat_template", std::string("{{ '<|im_start|>' }}{{ '<|im_end|>' }}")}
    };
    const auto emb = rand_f32({32, 256}, rng, 0.1f);
    HostTensor q_emb = HostTensor::create({32, 256}, DType::Q4_0);
    for (int i = 0; i < 32; ++i) {
        quantize_row_q4_0(emb.data<float>() + i * 256, q_emb.data<BlockQ4_0>() + i * 8, 256);
    }
    std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", q_emb}, {"output_norm.weight", make_ones_f32({256})},
        {"blk.0.attn_norm.weight", make_ones_f32({256})}, {"blk.0.ffn_norm.weight", make_ones_f32({256})},
        {"blk.0.attn_q.weight", rand_f32({256, 256}, rng, 0.03f)}, {"blk.0.attn_q.bias", rand_f32({256}, rng)},
        {"blk.0.attn_k.weight", rand_f32({64, 256}, rng, 0.03f)}, {"blk.0.attn_k.bias", rand_f32({64}, rng)},
        {"blk.0.attn_v.weight", rand_f32({64, 256}, rng, 0.03f)}, {"blk.0.attn_v.bias", rand_f32({64}, rng)},
        {"blk.0.attn_output.weight", rand_f32({256, 256}, rng, 0.03f)},
        {"blk.0.ffn_gate.weight", rand_f32({256, 256}, rng, 0.03f)},
        {"blk.0.ffn_up.weight", rand_f32({256, 256}, rng, 0.03f)},
        {"blk.0.ffn_down.weight", rand_f32({256, 256}, rng, 0.03f)}
    };
    FusionModel tied(HybridModelConfig{}), untied(HybridModelConfig{});
    std::string error;
    const auto model_path = files.model.string();
    auto load = [&](FusionModel& model) {
        if (!GgufFile::write_v3(model_path, metadata, weights, &error) ||
            !model.load_gguf(ctx, model_path, &error, 32)) throw std::runtime_error(error);
    };
    load(tied);
    GgufFile gguf;
    BpeTokenizer tokenizer;
    check(gguf.load(model_path) && tokenizer.load_from_gguf(gguf) &&
          tokenizer.format_chat_messages({{"user", "a"}}) == "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n" &&
          tokenizer.is_end_of_generation(1), "ChatML metadata");
    tied.set_device_profile(ctx, profile);
    const auto geometry = ctx.preferred_gemv_geometry();
    check(ctx.mmq_crossover_m() == 7 && tied.recommended_cuda_graph() && ctx.empirical_l2_bytes() == 33554432 &&
          geometry.warps == 4 && geometry.rows_per_warp == 2, "Device profile applied");
    const auto& layer = tied.dense_layers().front();
    check(layer.q_bias.data && layer.k_bias.data && layer.v_bias.data && layer.cfg.rope_neox &&
          layer.cfg.partial_rotary_factor == 0.5f && layer.cfg.rope_theta == 12345.0f, "Qwen2 metadata and QKV biases");
    weights.emplace_back("output.weight", q_emb);
    gguf = {};
    load(untied);
    const std::vector<int32_t> prefix{1, 2, 3}, branched{1, 2, 5, 6};
    untied.reset_kv_and_state(ctx);
    const auto explicit_logits = untied.forward_prefill_cuda(ctx, branched, 0);
    tied.prefill_cached_cuda(ctx, prefix);
    int prefilled = 0;
    const auto reused = tied.prefill_cached_cuda(ctx, branched, true, &prefilled);
    check(prefilled == 2, "Dense prefix should reuse KV before a divergence");
    expect_close(reused, std::vector<double>(explicit_logits.begin(), explicit_logits.end()),
                 1e-5, 1e-5, "Biased Qwen2 with reused KV and tied embeddings");
    const std::vector<int32_t> repeated{1, 2, 5, 6, 7};
    const auto first_logits = tied.prefill_cached_cuda(ctx, repeated);
    const auto repeat = tied.prefill_cached_cuda(ctx, repeated, true, &prefilled);
    check(prefilled == 1 && repeat == first_logits, "Dense exact repeat should replay one token with identical logits");
    ctx.weight_arena().reset();
    std::erase_if(weights, [](const auto& row) { return row.first.ends_with(".bias"); });
    FusionModel unbiased(HybridModelConfig{});
    load(unbiased);
    const auto no_bias = unbiased.prefill_cached_cuda(ctx, branched);
    check(std::abs(no_bias[0] - explicit_logits[0]) > 1e-4, "QKV biases were ignored in the graph");

    for (const auto& [key, value] : std::map(metadata)) {
        if (key.starts_with("qwen2.")) metadata["qwen3moe." + key.substr(6)] = value;
    }
    metadata["general.architecture"] = std::string("qwen3moe");
    weights.emplace_back("blk.0.ffn_gate_inp.weight", rand_f32({2, 256}, rng));
    weights.emplace_back("blk.0.ffn_gate_up_exps.weight", rand_f32({2, 64, 256}, rng, 0.03f));
    weights.emplace_back("blk.0.ffn_down_exps.weight", rand_f32({2, 256, 32}, rng, 0.03f));
    FusionModel routed(HybridModelConfig{});
    load(routed);
    check(routed.dense_layers().size() == 1 && routed.moe_layers().size() == 1 &&
          std::isfinite(routed.prefill_cached_cuda(ctx, branched)[0]), "GGUF MoE retains its GQA attention");

    HybridModelConfig hybrid;
    hybrid.arch = "hybrid";
    hybrid.num_layers = 1;
    hybrid.vocab_size = 32;
    hybrid.dense.max_seq_len = 32;
    FusionModel recurrent(hybrid);
    recurrent.init_synthetic(ctx, 111);
    GenerationOptions options;
    options.reuse_prefix = true;
    recurrent.generate_cuda(ctx, prefix, 0, true, false, nullptr, false, 0, -1, options);
    const std::vector<int32_t> extended{1, 2, 3, 4, 5};
    const auto appended = recurrent.prefill_cached_cuda(ctx, extended, true, &prefilled);
    check(prefilled == 2, "Hybrid prefix should append to live state");
    const auto reset = recurrent.prefill_cached_cuda(ctx, branched, true, &prefilled);
    check(prefilled == static_cast<int>(branched.size()), "Hybrid divergence must reset state");
    auto fresh = [&](const auto& ids) {
        recurrent.reset_kv_and_state(ctx);
        const auto scores = recurrent.forward_prefill_cuda(ctx, ids, 0);
        return std::vector<double>(scores.begin(), scores.end());
    };
    expect_close(appended, fresh(extended), 1e-5, 1e-5, "Hybrid appending cached state");
    expect_close(reset, fresh(branched), 1e-5, 1e-5, "Hybrid divergence resets cached state");
#endif
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_session_slots() {
    std::cout << "Testing CUDA session slots, KV sharing, GDN checkpoints and LRU...\n";
    CudaContext ctx(0, 48, 48);
    ctx.set_kv_dtype(DType::Q8_0);
    HybridModelConfig cfg;
    cfg.num_layers = 2;
    cfg.vocab_size = 64;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 128;
    cfg.dense.num_attention_heads = 2;
    cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 128;
    cfg.dense.max_seq_len = 512;
    const auto mark = ctx.weight_arena().mark();
    int prefilled = 0;

    {
        FusionModel model(cfg);
        model.set_num_slots(2);
        model.init_synthetic(ctx);
        const std::vector<int32_t> a(170, 1);
        const std::vector<int32_t> a_append{3};
        model.forward_prefill_cuda(ctx, a, 0);
        model.forward_prefill_cuda(ctx, a_append, static_cast<int>(a.size()));
        const auto cold = model.forward_token_cuda(ctx, 4, static_cast<int>(a.size() + 1), true, true);
        model.reset_kv_and_state(ctx);
        model.prefill_cached_cuda(ctx, a);
        const auto a_cache = model.cuda_cache_memory();
        auto b = a;
        std::fill(b.begin() + 145, b.end(), 2);
        model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto b_cache = model.cuda_cache_memory();
        if (prefilled != 25 || a_cache.kv_pages != 2 || b_cache.kv_pages != 3 ||
            b_cache.kv_bytes * 2 != a_cache.kv_bytes * 3) {
            throw std::runtime_error("Dense fork must share the complete page and copy only the partial page");
        }
        auto a_next = a;
        a_next.push_back(a_append.front());
        model.prefill_cached_cuda(ctx, a_next, true, &prefilled);
        const auto resumed = model.forward_token_cuda(ctx, 4, static_cast<int>(a_next.size()), true, true);
        if (prefilled != 1 || resumed != cold) {
            throw std::runtime_error("Dense slot A decode after the fork must match cold logits exactly");
        }
    }
    ctx.weight_arena().reset(mark);

    {
        auto hybrid = cfg;
        hybrid.arch = "hybrid";
        hybrid.num_layers = 4;
        FusionModel model(hybrid);
        model.set_num_slots(2);
        model.init_synthetic(ctx);
        const auto initial_ssm = model.cuda_cache_memory().ssm_bytes;
        std::vector<int32_t> a(147, 1);
        GenerationOptions options;
        options.reuse_prefix = true;
        GenerationStats stats;
        auto turn = model.generate_cuda(ctx, a, 3, true, true, &stats, false, 0, -1, options);
        turn.pop_back();  // Only the forwarded tokens have a checkpoint.
        auto next = turn;
        next.insert(next.end(), 7, 2);
        model.prefill_cached_cuda(ctx, next, true, &prefilled);
        if (prefilled != 7) throw std::runtime_error("Hybrid turn must append to the live recurrent state");

        auto b = turn;
        b.insert(b.end(), 8, 3);
        const auto forked = model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto cache = model.cuda_cache_memory();
        if (prefilled != 8 || cache.kv_pages != 3 || initial_ssm == 0 || cache.ssm_bytes != 7 * initial_ssm) {
            throw std::runtime_error("Hybrid fork must restore the turn checkpoint and share immutable snapshots");
        }
        model.reset_kv_and_state(ctx);
        const auto cold = model.forward_prefill_cuda(ctx, b, 0);
        expect_close(forked, {cold.begin(), cold.end()}, 1e-5, 1e-5,
                     "Hybrid GDN turn checkpoint fork against cold prefill");
    }
    ctx.weight_arena().reset(mark);

    {
        FusionModel model(cfg);
        model.set_num_slots(2);
        model.init_synthetic(ctx);
        const std::vector<int32_t> a(300, 1), b(150, 2), c(80, 3);
        model.prefill_cached_cuda(ctx, a);
        model.prefill_cached_cuda(ctx, b);
        model.prefill_cached_cuda(ctx, a);
        const auto before = model.cuda_cache_memory();
        model.prefill_cached_cuda(ctx, c, true, &prefilled);
        const auto after = model.cuda_cache_memory();
        if (prefilled != 80 || before.kv_pages != 5 || after.kv_pages != 4 ||
            after.kv_reserved_bytes != before.kv_reserved_bytes) {
            throw std::runtime_error("LRU eviction must return unreferenced pages to the free list");
        }
        model.prefill_cached_cuda(ctx, a, true, &prefilled);
        if (prefilled != 1) throw std::runtime_error("LRU eviction must keep the most recently used slot");
        model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto reused = model.cuda_cache_memory();
        if (prefilled != 150 || reused.kv_pages != 5 || reused.kv_reserved_bytes != before.kv_reserved_bytes) {
            throw std::runtime_error("An evicted prompt must start cold and recycle the released pages");
        }
    }
}

void test_tiered_cuda_kv() {
    CudaContext ctx(0, 32, 32);
    ctx.set_kv_dtype(DType::Q8_0);
    HybridModelConfig cfg;
    cfg.num_layers = 2;
    cfg.vocab_size = 64;
    cfg.dense.hidden_size = cfg.dense.intermediate_size = 128;
    cfg.dense.num_attention_heads = cfg.dense.num_key_value_heads = 2;
    cfg.dense.head_dim = 64;
    cfg.dense.max_seq_len = 384;
    const size_t page_bytes = 2 * cfg.num_layers * row_nbytes(DType::Q8_0,
        kKvPageTokens * cfg.dense.num_key_value_heads * cfg.dense.head_dim);
    const size_t vram_budget = 3 * page_bytes;
    const std::vector<int32_t> a(300, 1);
    auto b = a;
    std::fill(b.begin() + kKvPageTokens, b.end(), 2);
    auto resumed_prompt = a;
    resumed_prompt.push_back(3);

    auto run = [&](size_t budget, size_t host_budget) {
        const auto mark = ctx.weight_arena().mark();
        FusionModel model(cfg);
        model.set_num_slots(2);
        model.set_kv_vram_budget_bytes(budget);
        model.set_kv_host_budget_bytes(host_budget);
        model.init_synthetic(ctx, 41);
        int prefilled = 0;
        model.prefill_cached_cuda(ctx, a);
        model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto paused = model.cuda_cache_memory();
        if (prefilled != 300 - kKvPageTokens ||
            (budget && (paused.kv_pages != 3 || paused.kv_reserved_bytes != budget ||
                        paused.host_kv_pages != 2 || paused.host_kv_bytes != 2 * page_bytes)) ||
            (!budget && (paused.kv_pages != 5 || paused.host_kv_bytes != 0))) {
            throw std::runtime_error("Tiered KV must spill only the paused slot's two private pages");
        }
        model.prefill_cached_cuda(ctx, resumed_prompt, true, &prefilled);
        const auto active = model.cuda_cache_memory();
        if (prefilled != 1 || (budget && (active.kv_reserved_bytes > budget || active.host_kv_pages != 2))) {
            throw std::runtime_error("Tiered KV must restore the prefix and prefill only the new token");
        }
        std::vector<float> logits = model.forward_token_cuda(ctx, 4, static_cast<int>(resumed_prompt.size()), true, true);
        ctx.weight_arena().reset(mark);
        return logits;
    };
    const auto unspilled = run(0, 0);
    const auto spilled = run(vram_budget, 4 * page_bytes);
    const auto full_host = run(vram_budget, 2 * page_bytes);
    if (spilled != unspilled || full_host != unspilled) {
        throw std::runtime_error("Tiered KV decode logits differ after restore");
    }
    std::cout << "  [PASS] Tiered KV: GPU " << vram_budget << " B, pinned host " << 2 * page_bytes
              << " B (2 pages), reused 300 tokens, max_abs=0\n";
}

HostTensor moe_test_weights(std::vector<int64_t> shape, DType dtype, std::mt19937_64& rng) {
    if (dtype == DType::F32) return rand_f32(shape, rng, 0.04f);
    if (dtype == DType::Q2_K) return model_io::quantize_host_matrix(rand_f32(shape, rng, 0.04f), dtype);
    auto result = HostTensor::create(shape, dtype);
    if (dtype == DType::Q4_K) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ4_K); ++i) {
            auto& block = result.data<BlockQ4_K>()[i];
            block.d = fp32_to_fp16(0.00018f);
            block.dmin = fp32_to_fp16(0.00135f);
            for (auto& scale : block.scales) scale = static_cast<uint8_t>(rng());
            for (auto& q : block.qs) q = static_cast<uint8_t>(rng());
        }
    } else if (dtype == DType::Q6_K) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ6_K); ++i) {
            auto& block = result.data<BlockQ6_K>()[i];
            block.d = fp32_to_fp16(0.000023f);
            for (auto& scale : block.scales) scale = static_cast<int8_t>(rng());
            for (auto& q : block.ql) q = static_cast<uint8_t>(rng());
            for (auto& q : block.qh) q = static_cast<uint8_t>(rng());
        }
    } else if (dtype == DType::Q4_0) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ4_0); ++i) {
            auto& block = result.data<BlockQ4_0>()[i];
            block.d = fp32_to_fp16(0.009f);
            for (auto& q : block.qs) q = static_cast<uint8_t>(rng());
        }
    } else if (dtype == DType::Q8_0) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ8_0); ++i) {
            auto& block = result.data<BlockQ8_0>()[i];
            block.d = fp32_to_fp16(0.0005f);
            for (auto& q : block.qs) q = static_cast<int8_t>(rng());
        }
    }
    return result;
}

void test_cuda_native_quantized_moe() {
    constexpr int H = 256, I = 256, E = 8, resident = 3, V = 64;
    constexpr double mib = 1024.0 * 1024.0;
    struct Fixture {
        std::filesystem::path path = "test-u4-moe.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    CudaContext ctx(0, 0, 32);
    ctx.set_cpu_moe_q8(false);
    for (const auto dtype : {DType::Q4_0, DType::Q2_K, DType::Q4_K, DType::Q6_K, DType::F32}) {
        const std::string arch = dtype == DType::Q2_K ? "llama" : "qwen3moe";
        const auto prefix = arch + ".";
        std::map<std::string, GgufMetaValue> meta{
            {"general.architecture", arch}, {prefix + "block_count", uint32_t{1}},
            {prefix + "embedding_length", uint32_t{H}}, {prefix + "attention.head_count", uint32_t{8}},
            {prefix + "attention.head_count_kv", uint32_t{2}}, {prefix + "feed_forward_length", uint32_t{I}},
            {prefix + "expert_count", uint32_t{E}}, {prefix + "expert_used_count", uint32_t{2}}
        };
        FusionModel model(HybridModelConfig{});
        model.set_device_profile(ctx, {});
        const bool normalize = dtype != DType::Q6_K;
        meta[prefix + "expert_weights_norm"] = normalize;
        std::mt19937_64 rng(841);
        auto embeddings = rand_f32({V, H}, rng, 0.1f);
        for (int t = 0; t < V; ++t) embeddings.data<float>()[t * H] = 0.35f;
        HostTensor router({E, H}, DType::F32);
        // With a zero attention output, the positive first hidden feature routes to one hot and one cold expert.
        router.data<float>()[H] = 0.9f;
        router.data<float>()[6 * H] = 0.7f;
        std::vector<std::pair<std::string, HostTensor>> weights{
            {"token_embd.weight", std::move(embeddings)}, {"output_norm.weight", make_ones_f32({H})},
            {"blk.0.attn_norm.weight", make_ones_f32({H})}, {"blk.0.ffn_norm.weight", make_ones_f32({H})},
            {"blk.0.attn_q.weight", rand_f32({H, H}, rng)}, {"blk.0.attn_k.weight", rand_f32({64, H}, rng)},
            {"blk.0.attn_q_norm.weight", make_ones_f32({32})}, {"blk.0.attn_k_norm.weight", make_ones_f32({32})},
            {"blk.0.attn_v.weight", rand_f32({64, H}, rng)}, {"blk.0.attn_output.weight", HostTensor({H, H}, DType::F32)},
            {"blk.0.ffn_gate_inp.weight", std::move(router)},
            {"blk.0.ffn_gate_shexp.weight", rand_f32({I, H}, rng, 0.035f)},
            {"blk.0.ffn_up_shexp.weight", rand_f32({I, H}, rng, 0.035f)},
            {"blk.0.ffn_down_shexp.weight", rand_f32({H, I}, rng, 0.035f)},
            {"blk.0.ffn_down_exps.weight", moe_test_weights({E, H, I}, dtype, rng)}
        };
        const bool split = dtype == DType::Q4_K || dtype == DType::Q6_K;
        const auto gate_type = dtype == DType::Q6_K ? DType::Q4_K : dtype;
        const auto up_type = dtype == DType::Q4_K ? DType::Q8_0 : gate_type;
        weights.emplace_back(split ? "blk.0.ffn_gate_exps.weight" : "blk.0.ffn_gate_up_exps.weight",
                             moe_test_weights({E, split ? I : 2 * I, H}, gate_type, rng));
        if (split) weights.emplace_back("blk.0.ffn_up_exps.weight", moe_test_weights({E, I, H}, up_type, rng));
        std::string error;
        if (!GgufFile::write_v3(fixture.path.string(), meta, weights, &error)) throw std::runtime_error(error);

        const size_t gate_bytes = (split ? I : 2 * I) * row_nbytes(gate_type, H);
        const size_t down_bytes = H * row_nbytes(dtype, I);
        const size_t up_bytes = split ? I * row_nbytes(up_type, H) : 0;
        auto load = [&](int expert_override, int expected, int context = 32) {
            model.set_moe_gpu_experts(expert_override);
            if (!model.load_gguf(ctx, fixture.path.string(), &error, context)) throw std::runtime_error(error);
            const auto& layer = model.moe_layers().at(0);
            const int cold = E - expected;
            if (!layer.cfg.has_shared_expert || layer.cfg.top_k != 2 ||
                layer.cfg.norm_topk_prob != normalize || !model.config().dense.qk_norm ||
                model.config().moe.gpu_resident_experts != expected || layer.cfg.gpu_resident_experts != expected ||
                layer.gpu_gate_up_w.dtype != gate_type || layer.gpu_down_w.dtype != dtype ||
                layer.gpu_gate_up_w.nbytes() != expected * gate_bytes ||
                layer.gpu_down_w.nbytes() != expected * down_bytes ||
                layer.gpu_up_w.nbytes() != expected * up_bytes ||
                layer.cpu_gate_up_w.bytes().size() != cold * gate_bytes ||
                layer.cpu_down_w.bytes().size() != cold * down_bytes ||
                layer.cpu_up_w.bytes().size() != cold * up_bytes ||
                !layer.cpu_gate_up_w.storage.empty() || !layer.cpu_down_w.storage.empty() || !layer.cpu_up_w.storage.empty() ||
                (cold && (layer.cpu_gate_up_w.desc.dtype != gate_type || layer.cpu_down_w.desc.dtype != dtype)) ||
                (split && (layer.gpu_up_w.dtype != up_type || (cold && layer.cpu_up_w.desc.dtype != up_type)))) {
                throw std::runtime_error("MoE placement or native expert dtype/byte count is incorrect");
            }
        };
        auto logits = [&] {
            const std::vector<int32_t> prompt{2, 3, 4, 5, 6, 7, 8, 9};
            auto prefill = model.forward_prefill_cuda(ctx, prompt, 0);
            auto decode = model.forward_token_cuda(ctx, 10, static_cast<int>(prompt.size()), true, true);
            return std::pair{std::move(prefill), std::move(decode)};
        };
        load(E, E);
        const size_t full_arena = ctx.weight_arena().capacity();
        const auto gpu = logits();
        load(resident, resident);
        const auto hybrid = logits();
        DeviceProfile profile;
        const size_t kv_bytes = 2 * kKvPageTokens * 64 * sizeof(float);
        profile.vram_total_mb = (ctx.scratch_arena().capacity() + kv_bytes + 256 + full_arena -
                                 (E - resident) * (gate_bytes + up_bytes + down_bytes)) / mib;
        profile.cpu_q4_gemv_gbps = 51.38;
        profile.pcie_h2d_gbps = 14.24;
        model.set_device_profile(ctx, profile);
        load(-1, resident);
        const auto automatic = logits();
        const double tolerance = dtype == DType::F32 ? 1e-6 : 1e-4;
        auto compare = [&](const std::vector<float>& a, const std::vector<float>& b) {
            double diff_sq = 0.0, ref_sq = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double diff = static_cast<double>(a[i]) - b[i];
                diff_sq += diff * diff;
                ref_sq += static_cast<double>(a[i]) * a[i];
            }
            const double rel_l2 = std::sqrt(diff_sq / ref_sq);
            if (!(rel_l2 < tolerance)) throw std::runtime_error("GPU vs hybrid MoE logits differ");
            return rel_l2;
        };
        if (automatic != hybrid) throw std::runtime_error("MoE auto placement must match the explicit split");
        if (dtype == DType::Q2_K || dtype == DType::Q4_K || dtype == DType::Q6_K) {
            ctx.set_cpu_moe_q8(true);
            const auto integer = logits();
            ctx.set_cpu_moe_q8(false);
            expect_close(integer.first, {hybrid.first.begin(), hybrid.first.end()}, 0.02, 0.02,
                         "Q8_K hybrid prefill vs F32 activation");
            expect_close(integer.second, {hybrid.second.begin(), hybrid.second.end()}, 0.02, 0.02,
                         "Q8_K hybrid decode vs F32 activation");
        }
        std::cout << "  [PASS] Native " << dtype_name(dtype) << " MoE 8x top-2 + shared: gpu_resident=3 cpu_cold=5"
                  << " prefill(M=8) rel_l2=" << compare(gpu.first, hybrid.first)
                  << " decode(M=1) rel_l2=" << compare(gpu.second, hybrid.second) << '\n';
        if (dtype == DType::F32) {
            constexpr int context = 4096, base = E - 2;
            model.set_device_profile(ctx, {});
            load(E, E, context);
            profile.vram_total_mb = (ctx.scratch_arena().capacity() + ctx.weight_arena().capacity() + 256 +
                (context / kKvPageTokens) * kv_bytes - (E - base) * (gate_bytes + down_bytes)) / mib;
            model.set_device_profile(ctx, profile);
            load(-1, base, context);
            const auto short_result = model.generate_cuda(ctx, {1, 2, 3}, 4);
            const auto borrowed = model.cuda_borrowed_expert_bytes();
            if (!borrowed || model.moe_layers()[0].cfg.gpu_resident_experts != E)
                throw std::runtime_error("Short request must allow experts to borrow unused KV memory");
            const auto long_result = model.generate_cuda(ctx, std::vector<int32_t>(1152, 1), 4);
            if (model.cuda_borrowed_expert_bytes() != 0 || model.moe_layers()[0].cfg.gpu_resident_experts != base ||
                long_result.size() != 1152 + 4)
                throw std::runtime_error("Growing KV cache must reclaim borrowed expert memory");
            if (model.generate_cuda(ctx, {1, 2, 3}, 4) != short_result || model.cuda_borrowed_expert_bytes() != borrowed)
                throw std::runtime_error("After the KV cache resets, expert re-promotion must preserve output");
            auto& allocate = CudaReviewTestAccess::allocator(model);
            const auto normal = allocate;
            static int attempts = 0;
            static cudaError_t oom = cudaSuccess;
            attempts = 0;
            allocate = [](void** address, size_t bytes) {
                if (++attempts != 1) return cudaMalloc(address, bytes);
                // Larger than the GPU virtual address space, so this fails without consuming VRAM.
                oom = cudaMalloc(address, std::numeric_limits<size_t>::max() / 2);
                return oom;
            };
            const auto recovered = model.generate_cuda(ctx, {1, 2, 3}, 4);
            allocate = normal;
            if (oom != cudaErrorMemoryAllocation || attempts != 2 || model.cuda_borrowed_expert_bytes() != 0 ||
                recovered.size() != short_result.size() || model.moe_layers()[0].cfg.gpu_resident_experts != base)
                throw std::runtime_error("Actual CUDA KV allocation failure must reclaim the borrowed slab and retry");
            std::cout << "  [PASS] CUDA allocation OOM evicts borrowed experts before retry\n";

            model.set_device_profile(ctx, {});
            model.set_num_slots(2);
            model.set_kv_vram_budget_bytes(3 * kv_bytes);
            model.set_kv_host_budget_bytes(kv_bytes);
            load(E, E, context);
            model.prefill_cached_cuda(ctx, std::vector<int32_t>(kKvPageTokens + 1, 1));
            int prefilled = 0;
            const std::vector<int32_t> active(2 * kKvPageTokens + 1, 2);
            const auto active_logits = model.prefill_cached_cuda(ctx, active, true, &prefilled);
            if (prefilled != static_cast<int>(active.size()) || active_logits.size() != V ||
                model.cuda_cache_memory().kv_pages != 3 || model.cuda_cache_memory().host_kv_pages != 0)
                throw std::runtime_error("When host KV cannot fit the spill, reclaim the paused prefix and retry");
            std::cout << "  [PASS] Full host KV recovery evicts the paused slot\n";
            FusionModel explicit_model(HybridModelConfig{});
            ctx.set_kv_dtype(std::nullopt);
            explicit_model.set_moe_gpu_experts(1);
            explicit_model.set_kv_vram_budget_bytes(2 * row_nbytes(DType::Q8_0, kKvPageTokens * 64));
            if (!explicit_model.load_gguf(ctx, fixture.path.string(), &error, 32, true) ||
                explicit_model.cuda_kv_dtype() != DType::Q8_0 ||
                explicit_model.moe_layers()[0].cfg.gpu_resident_experts != 1)
                throw std::runtime_error("Explicit expert count must still allow auto KV precision fallback");
            ctx.set_kv_dtype(DType::F32);
            std::cout << "  [PASS] Explicit GPU expert placement retains auto Q8_0 KV fallback\n";
            std::cout << "  [PASS] MoE expert slabs promote, yield to KV, and re-promote: " << borrowed << " bytes\n";
        }
        ctx.weight_arena().release();
    }
}

void test_cpu_moe_q8_dots() {
    constexpr int cols = 768, rows = 64;
    std::mt19937_64 rng(0x84);
    const auto input = rand_f32({cols}, rng);
    std::array<CpuQ8K, cols / 256> q8;
    cpu_moe_quantize_q8(input.data<float>(), q8.data(), cols);
    std::array<float, cols> quantized{};
    for (int i = 0; i < cols; ++i) quantized[i] = q8[i / 256].scale * q8[i / 256].values[i % 256];
    for (const auto dtype : {DType::Q2_K, DType::Q4_K, DType::Q6_K}) {
        const auto dot = cpu_moe_q8_dot(dtype);
        if (!dot) continue;
        const auto w = moe_test_weights({1, rows, cols}, dtype, rng);
        const auto values = model_io::to_f32_host(w);
        const auto stride = row_nbytes(dtype, cols);
        const auto narrow = cpu_moe_vnni_dot(dtype);
        const auto wide = cpu_moe_vnni_dot(dtype, true);
        double error = 0.0, reference = 0.0, quant_error = 0.0;
        for (int row = 0; row < rows; ++row) {
            const auto* x = values.data<float>() + row * cols;
            const double expected = std::inner_product(quantized.begin(), quantized.end(), x, 0.0);
            const double exact = std::inner_product(input.data<float>(), input.data<float>() + cols, x, 0.0);
            const auto* weight = w.bytes().data() + row * stride;
            const double actual = dot(weight, q8.data(), cols);
            if (wide && narrow && narrow(weight, q8.data(), cols) != wide(weight, q8.data(), cols))
                throw std::runtime_error("256-bit and 512-bit VNNI must produce identical dots");
            error += (actual - expected) * (actual - expected);
            quant_error += (actual - exact) * (actual - exact);
            reference += expected * expected;
        }
        const double rel = std::sqrt(error / reference), quant_rel = std::sqrt(quant_error / reference);
        if (!(rel < 2e-6 && quant_rel < 0.02))
            throw std::runtime_error("Integer Q8_K dot parity failed");
        const char* kernel = dot == wide ? "VNNI-512" : dot == narrow ? "VNNI-256" : "AVX2";
        std::cout << "  [PASS] " << kernel << ' ' << dtype_name(dtype) << " x Q8_K scalar rel_l2=" << rel
                  << " F32_activation_rel_l2=" << quant_rel << '\n';
    }
    std::array<CpuQ8Small, cols / 32> small;
    cpu_moe_quantize_q8_small(input.data<float>(), small.data(), cols);
    for (int i = 0; i < cols; ++i) quantized[i] = small[i / 32].scale * small[i / 32].values[i % 32];
    for (const auto dtype : {DType::Q4_0, DType::Q8_0}) {
        const auto dot = cpu_moe_vnni_small_dot(dtype);
        if (!dot) continue;
        const auto weights = moe_test_weights({1, rows, cols}, dtype, rng);
        const auto values = model_io::to_f32_host(weights);
        const auto stride = row_nbytes(dtype, cols);
        double squared_error = 0.0, squared_reference = 0.0;
        for (int row = 0; row < rows; ++row) {
            const double expected = std::inner_product(quantized.begin(), quantized.end(), values.data<float>() + row * cols, 0.0);
            const double actual = dot(weights.bytes().data() + row * stride, small.data(), cols);
            squared_error += (actual - expected) * (actual - expected);
            squared_reference += expected * expected;
        }
        const double rel = std::sqrt(squared_error / squared_reference);
        if (!(rel < 2e-6)) throw std::runtime_error("VNNI Q8_0 dot parity failed");
        std::cout << "  [PASS] VNNI " << dtype_name(dtype) << " x Q8_0 scalar rel_l2=" << rel << '\n';
    }
}

void test_cpu_moe_multirow() {
    constexpr int hidden = 256, inter = 256, experts = 4, rows = 7, top_k = 2;
    const std::array<int32_t, rows * top_k> ids{1,2, 2,1, 2,3, 1,3, 2,3, 2,1, 3,1};
    std::array<float, rows * top_k> weights{};
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (i % 2 == 0) ? 0.7f : 0.3f;
    std::mt19937_64 rng(123);
    const auto x = rand_f32({rows, hidden}, rng);
    CpuWorkers pool(4);
    CpuMoeWorkspace scratch;
    for (const auto dtype : {DType::Q2_K, DType::Q4_K, DType::Q6_K}) {
        if (!cpu_moe_q8_rows(dtype)) continue;
        const auto gate = moe_test_weights({experts, 2 * inter, hidden}, dtype, rng);
        const auto down = moe_test_weights({experts, hidden, inter}, dtype, rng);
        std::vector<float> reference(rows * hidden), result(rows * hidden);
        for (int row = 0; row < rows; ++row)
            cpu_moe_accumulate_layer(&pool, scratch, gate, {}, down, ids.data() + row * top_k,
                weights.data() + row * top_k, 0, 0, 1, top_k, x.data<float>() + row * hidden,
                reference.data() + row * hidden);
        cpu_moe_accumulate_layer(&pool, scratch, gate, {}, down, ids.data(), weights.data(), 0, 0, rows,
            top_k, x.data<float>(), result.data());
        double error = 0.0, norm = 0.0;
        for (size_t i = 0; i < result.size(); ++i) {
            const double delta = result[i] - reference[i];
            error += delta * delta;
            norm += static_cast<double>(reference[i]) * reference[i];
        }
        const double rel = std::sqrt(error / norm);
        if (!(rel < 1e-6)) throw std::runtime_error("Multi-row cold expert output differs from single-row");
        std::cout << "  [PASS] " << dtype_name(dtype) << " cold expert multi-row rel_l2=" << rel << '\n';
    }
}

void test_long_context_kv_admission() {
    HybridModelConfig cfg;
    cfg.num_layers = 1;
    cfg.vocab_size = 32;
    cfg.dense.hidden_size = 64;
    cfg.dense.intermediate_size = 128;
    cfg.dense.num_attention_heads = 2;
    cfg.dense.num_key_value_heads = 1;
    cfg.dense.head_dim = 32;
    cfg.dense.max_seq_len = 32768;
    constexpr size_t page = 2 * kKvPageTokens * 32 * sizeof(uint16_t);
    CudaContext ctx(0, 4, 16);
    ctx.set_kv_dtype(std::nullopt);
    FusionModel model(cfg);
    model.set_num_slots(2);
    model.set_kv_vram_budget_bytes(page);
    model.init_synthetic(ctx);
    if (model.cuda_kv_dtype() != DType::F16 || model.cuda_cache_memory().kv_reserved_bytes != 0 ||
        model.cuda_max_active_tokens() != kKvPageTokens)
        throw std::runtime_error("32K context must start with zero physical KV pages");
    const auto short_request = model.generate_cuda(ctx, {1, 2, 3}, 4);
    if (short_request.size() != 7 || model.cuda_cache_memory().kv_reserved_bytes != page)
        throw std::runtime_error("Short request must only allocate a single KV page");
    bool rejected = false;
    try { (void)model.generate_cuda(ctx, {1, 2, 3}, 126); }
    catch (const std::invalid_argument& e) {
        const std::string_view message = e.what();
        rejected = message.find("129 tokens") != message.npos && message.find("KV budget") != message.npos;
    }
    if (!rejected || model.cuda_cache_memory().kv_reserved_bytes != page)
        throw std::runtime_error("The over-budget 32K request must be rejected before allocating or mutating KV");
    std::cout << "  [PASS] 32K two-slot short KV allocation and over-budget admission\n";
}

void test_dense_auto_kv_fallback() {
    constexpr int H = 64, context = 2 * kKvPageTokens;
    struct Fixture {
        std::filesystem::path path = "test-dense-auto-kv.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("llama")}, {"llama.block_count", uint32_t{1}},
        {"llama.embedding_length", uint32_t{H}}, {"llama.attention.head_count", uint32_t{2}},
        {"llama.attention.head_count_kv", uint32_t{1}}, {"llama.feed_forward_length", uint32_t{H}}
    };
    const std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", HostTensor({32, H}, DType::F32)}, {"output_norm.weight", make_ones_f32({H})},
        {"blk.0.attn_norm.weight", make_ones_f32({H})}, {"blk.0.ffn_norm.weight", make_ones_f32({H})},
        {"blk.0.attn_q.weight", HostTensor({H,H}, DType::F32)}, {"blk.0.attn_k.weight", HostTensor({H/2,H}, DType::F32)},
        {"blk.0.attn_v.weight", HostTensor({H/2,H}, DType::F32)}, {"blk.0.attn_output.weight", HostTensor({H,H}, DType::F32)},
        {"blk.0.ffn_gate_up.weight", HostTensor({2 * H,H}, DType::F32)}, {"blk.0.ffn_down.weight", HostTensor({H,H}, DType::F32)}
    };
    CudaContext ctx(0, 0, 32);
    ctx.set_kv_dtype(std::nullopt);
    FusionModel model(HybridModelConfig{});
    std::string error;
    if (!GgufFile::write_v3(fixture.path.string(), meta, weights, &error) ||
        !model.load_gguf(ctx, fixture.path.string(), &error, context) || model.cuda_kv_dtype() != DType::F16)
        throw std::runtime_error("Dense auto KV baseline: " + error);
    const size_t q8 = 2 * row_nbytes(DType::Q8_0, context * (H / 2));
    DeviceProfile profile;
    profile.vram_total_mb = (ctx.weight_arena().capacity() + ctx.scratch_arena().capacity() + 256 + q8) / 1048576.0;
    model.set_device_profile(ctx, profile);
    if (!model.load_gguf(ctx, fixture.path.string(), &error, context) ||
        model.cuda_kv_dtype() != DType::Q8_0 || model.cuda_max_active_tokens() != context)
        throw std::runtime_error("Dense auto KV must use Q8_0 under a profile between the F16 and Q8_0 footprints");
    std::cout << "  [PASS] Dense auto KV uses Q8_0 when F16 cannot fit the full context\n";
}

void test_cpu_quantized_moe_speed() {
    constexpr int width = 1536;
    std::mt19937_64 rng(161);
    const auto input = rand_f32({width}, rng, 0.25f);
    const HostTensor no_separate_up;
    for (const auto dtype : {DType::Q4_0, DType::Q4_K, DType::Q6_K}) {
        const auto gate_up = moe_test_weights({1, 2 * width, width}, dtype, rng);
        const auto down = moe_test_weights({1, width, width}, dtype, rng);
        auto run = [&](CpuMoeMode mode, std::vector<float>& output) {
            auto execute = [&] {
                std::fill(output.begin(), output.end(), 0.0f);
                cpu_moe_accumulate(gate_up, no_separate_up, down, 0, input.data<float>(), 0.5f, output.data(), mode);
            };
            execute();
            double best_seconds = std::numeric_limits<double>::infinity();
            for (int sample = 0; sample < 5; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                execute();
                best_seconds = std::min(best_seconds, std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count());
            }
            return best_seconds;
        };
        std::vector<float> scalar(width), parallel(width);
        const double scalar_seconds = run(CpuMoeMode::Scalar, scalar);
        const double parallel_seconds = run(CpuMoeMode::ParallelF32, parallel);
        expect_close(parallel, {scalar.begin(), scalar.end()}, 3e-4, 1e-4, "CPU native expert vs scalar");
        if (std::thread::hardware_concurrency() > 1 && !(parallel_seconds < scalar_seconds)) {
            throw std::runtime_error("Parallel CPU expert must beat single-thread scalar");
        }
        std::cout << "  [PASS] CPU " << dtype_name(dtype) << " "
                  << (gate_up.storage.size() + down.storage.size()) / parallel_seconds / 1e9 << " GB/s, "
                  << scalar_seconds / parallel_seconds << "x vs single-thread scalar\n";
    }
}

void test_chunked_prefill_and_k_quant_gdn() {
    std::cout << "Testing chunked prefill and K-quant GDN GGUF...\n";
    CudaContext ctx(0, 64, 64);
    ctx.set_prefill_batch(128);
    const auto mark = ctx.weight_arena().mark();
    for (const std::string arch : {"dense", "hybrid"}) {
        HybridModelConfig cfg;
        cfg.arch = arch;
        cfg.num_layers = 2;
        cfg.vocab_size = 64;
        cfg.dense.hidden_size = cfg.gdn.hidden_size = 256;
        cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 256;
        cfg.dense.max_seq_len = 160;
        FusionModel model(cfg);
        model.init_synthetic(ctx, 13);
        std::vector<int32_t> prompt(129);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<int32_t>(i % 63 + 1);
        for (int count : {127, 128, 129}) {
            const std::vector<int32_t> tokens(prompt.begin(), prompt.begin() + count);
            const std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + 65);
            model.reset_kv_and_state(ctx);
            OpTrace ops;
            LayerTrace layers;
            const auto whole = model.prefill_cached_cuda(ctx, tokens, true, nullptr, &layers, &ops);
            const auto next = model.forward_token_cuda(ctx, 8, count);
            const auto head = std::find_if(ops.ops.begin(), ops.ops.end(),
                [](const OpTiming& op) { return op.name == "lm_head"; });
            if (head == ops.ops.end() || head->calls != 1 || layers.outputs.size() != 2) {
                throw std::runtime_error("Chunked prefill must project logits once and trace the final chunk");
            }
            model.reset_kv_and_state(ctx);
            model.prefill_cached_cuda(ctx, prefix);
            int processed = 0;
            const auto resumed = model.prefill_cached_cuda(ctx, tokens, true, &processed);
            const auto resumed_next = model.forward_token_cuda(ctx, 8, count);
            if (processed != count - static_cast<int>(prefix.size())) {
                throw std::runtime_error("Chunked prefill did not reuse the cached prefix");
            }
            expect_close(resumed, {whole.begin(), whole.end()}, 1e-4, 1e-4,
                         arch + " prefix prefill " + std::to_string(count));
            expect_close(resumed_next, {next.begin(), next.end()}, 1e-4, 1e-4,
                         arch + " first decode after " + std::to_string(count));
        }
        ctx.weight_arena().reset(mark);
    }

    struct Fixture {
        std::filesystem::path path = "k-quant-gdn-test.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    ctx.weight_arena().init(0);
    constexpr int H = 256, K = 2, V = 4, D = 64;
    constexpr int conv = 2 * K * D + V * D;
    std::mt19937_64 rng(748);
    auto quantized = [&](int rows) {
        HostTensor result = HostTensor::create({rows, H}, DType::Q4_K);
        for (int row = 0; row < rows; ++row) {
            auto& block = result.data<BlockQ4_K>()[row];
            block.d = fp32_to_fp16(0.002f);
            block.dmin = fp32_to_fp16(0.012f);
            std::fill(std::begin(block.scales), std::end(block.scales), uint8_t{1});
            for (auto& q : block.qs) q = static_cast<uint8_t>(rng());
        }
        return result;
    };
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("qwen35")}, {"qwen35.block_count", uint32_t{1}},
        {"qwen35.embedding_length", uint32_t{H}}, {"qwen35.feed_forward_length", uint32_t{H}},
        {"qwen35.ssm.group_count", uint32_t{K}}, {"qwen35.ssm.time_step_rank", uint32_t{V}},
        {"qwen35.ssm.state_size", uint32_t{D}}, {"qwen35.ssm.inner_size", uint32_t{V * D}},
    };
    const std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", rand_f32({32, H}, rng, 0.1f)}, {"output_norm.weight", make_ones_f32({H})},
        {"blk.0.attn_norm.weight", make_ones_f32({H})}, {"blk.0.attn_qkv.weight", quantized(conv)},
        {"blk.0.attn_gate.weight", quantized(V * D)}, {"blk.0.ssm_alpha.weight", rand_f32({V, H}, rng, 0.03f)},
        {"blk.0.ssm_beta.weight", rand_f32({V, H}, rng, 0.03f)}, {"blk.0.ssm_conv1d.weight", rand_f32({conv, 4}, rng)},
        {"blk.0.ssm_a", rand_f32({V}, rng)}, {"blk.0.ssm_dt.bias", rand_f32({V}, rng)},
        {"blk.0.ssm_norm.weight", make_ones_f32({D})}, {"blk.0.ssm_out.weight", rand_f32({H, V * D}, rng, 0.03f)},
        {"blk.0.ffn_norm.weight", make_ones_f32({H})}, {"blk.0.ffn_gate.weight", rand_f32({H, H}, rng, 0.03f)},
        {"blk.0.ffn_up.weight", rand_f32({H, H}, rng, 0.03f)}, {"blk.0.ffn_down.weight", rand_f32({H, H}, rng, 0.03f)},
    };
    FusionModel model(HybridModelConfig{});
    std::string error;
    const auto path = fixture.path.string();
    const std::tuple<const char*, uint32_t, const char*> bad_gdn_shapes[] = {
        {"qwen35.ssm.conv_kernel", 5, "conv_kernel=5"},
        {"qwen35.ssm.state_size", 129, "dk=129"},
        {"qwen35.ssm.inner_size", V * 257, "dv=257"}};
    for (const auto& [key, value, detail] : bad_gdn_shapes) {
        auto invalid = meta;
        invalid[key] = value;
        model_io::LoadedModelRaw raw;
        if (!GgufFile::write_v3(path, invalid, weights, &error) ||
            model_io::parse_gguf_model_raw(path, 32, raw, &error) ||
            error.find("Unsupported GDN shape") == std::string::npos ||
            error.find(detail) == std::string::npos)
            throw std::runtime_error("Unsupported GDN shape must be rejected before upload");
    }
    if (!GgufFile::write_v3(path, meta, weights, &error) || !model.load_gguf(ctx, path, &error, 32)) {
        throw std::runtime_error("K-quant GDN GGUF load failed: " + error);
    }
    const auto& layer = model.gdn_layers().front();
    if (layer.qkv_proj.dtype != DType::Q4_KA || layer.z_proj.dtype != DType::Q4_KA ||
        layer.a_proj.dtype != DType::F32 || layer.b_proj.dtype != DType::F32) {
        throw std::runtime_error("GDN projections must preserve their weight precision");
    }
    const std::vector<int32_t> prompt{1, 2, 3, 4, 5};
    model.reset_kv_and_state(ctx);
    const auto logits = model.forward_prefill_cuda(ctx, prompt, 0);
    const auto next = model.forward_token_cuda(ctx, 6, static_cast<int>(prompt.size()));
    if (logits.size() != 32 || next.size() != 32 ||
        !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(next.begin(), next.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("K-quant GDN must produce finite prefill and decode logits");
    }
}
#endif

void test_stage7c_mla() {
    constexpr int hidden = 32, vocab = 48, layers = 2, heads = 8;
    constexpr int rank = 160, rope = 8, nope = 24, key = nope + rope, value = 24;
    constexpr int experts = 4, top_k = 2, expert_width = 16, shared = 2;
    constexpr int query_rank = 24;
    constexpr float eps = 1e-6f, router_scale = 1.35f, yarn_log_mul = 0.07f;
    struct Fixture {
        std::filesystem::path path = "test-stage7c-mla.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
#ifdef FUSIONRT_ENABLE_CUDA
    CudaContext ctx(0, 0, 32);
#endif
    for (const bool lora : {false, true}) {
        std::map<std::string, GgufMetaValue> meta{
            {"general.architecture", std::string("deepseek2")}, {"deepseek2.block_count", uint32_t{layers}},
            {"deepseek2.embedding_length", uint32_t{hidden}}, {"deepseek2.attention.head_count", uint32_t{heads}},
            {"deepseek2.attention.head_count_kv", uint32_t{1}}, {"deepseek2.context_length", uint32_t{256}},
            {"deepseek2.attention.key_length", static_cast<uint32_t>(lora ? rank + rope : key)},
            {"deepseek2.attention.value_length", static_cast<uint32_t>(lora ? rank : value)},
            {"deepseek2.attention.kv_lora_rank", uint32_t{rank}}, {"deepseek2.rope.dimension_count", uint32_t{rope}},
            {"deepseek2.rope.freq_base", 100.0f}, {"deepseek2.attention.layer_norm_rms_epsilon", eps},
            {"deepseek2.feed_forward_length", uint32_t{hidden}}, {"deepseek2.leading_dense_block_count", uint32_t{1}},
            {"deepseek2.expert_count", uint32_t{experts}}, {"deepseek2.expert_used_count", uint32_t{top_k}},
            {"deepseek2.expert_shared_count", uint32_t{shared}}, {"deepseek2.expert_feed_forward_length", uint32_t{expert_width}},
            {"deepseek2.expert_gating_func", uint32_t{2}}, {"deepseek2.expert_weights_norm", true},
            {"deepseek2.expert_weights_scale", router_scale}
        };
        if (lora) {
            meta["deepseek2.attention.q_lora_rank"] = uint32_t{query_rank};
            meta["deepseek2.attention.key_length_mla"] = uint32_t{key};
            meta["deepseek2.attention.value_length_mla"] = uint32_t{value};
            meta["deepseek2.rope.scaling.type"] = std::string("yarn");
            meta["deepseek2.rope.scaling.factor"] = 4.0f;
            meta["deepseek2.rope.scaling.original_context_length"] = uint32_t{256};
            meta["deepseek2.rope.scaling.yarn_log_multiplier"] = yarn_log_mul;
            meta["deepseek2.expert_group_count"] = uint32_t{2};
            meta["deepseek2.expert_group_used_count"] = uint32_t{1};
        }
        std::mt19937_64 rng(7003);
        std::vector<std::pair<std::string, HostTensor>> tensors;
        auto add = [&](const std::string& name, std::vector<int64_t> shape, float scale = 0.09f) {
            tensors.emplace_back(name, model_io::make_rand_f32(shape, rng, scale));
        };
        auto norm_weight = [&](const std::string& name, int width) {
            add(name, {width}, 0.04f);
            float* w = tensors.back().second.data<float>();
            for (int i = 0; i < width; ++i) w[i] += 1.0f;
        };
        add("token_embd.weight", {vocab, hidden});
        add("output.weight", {vocab, hidden});
        norm_weight("output_norm.weight", hidden);
        std::array<HostTensor, layers> combined;
        for (int i = 0; i < layers; ++i) {
            const std::string p = "blk." + std::to_string(i) + ".";
            norm_weight(p + "attn_norm.weight", hidden);
            norm_weight(p + "ffn_norm.weight", hidden);
            norm_weight(p + "attn_kv_a_norm.weight", rank);
            add(p + "attn_kv_a_mqa.weight", {rank + rope, hidden});
            if (lora) {
                add(p + "attn_q_a.weight", {query_rank, hidden});
                norm_weight(p + "attn_q_a_norm.weight", query_rank);
                add(p + "attn_q_b.weight", {heads * key, query_rank});
            } else {
                add(p + "attn_q.weight", {heads * key, hidden});
            }
            add(p + "attn_output.weight", {hidden, heads * value}, 0.04f);
            combined[i] = model_io::make_rand_f32({heads, nope + value, rank}, rng, 0.04f);
            if (lora) {
                // Store split KV-B in FP16 and use the same rounded weights in the expanded reference.
                combined[i] = model_io::to_f32_host(model_io::quantize_host_matrix(combined[i], DType::F16));
                HostTensor k({heads, rank, nope}, DType::F32), v({heads, value, rank}, DType::F32);
                for (int h = 0; h < heads; ++h) {
                    const float* src = combined[i].data<float>() + h * (nope + value) * rank;
                    for (int d = 0; d < nope; ++d)
                        for (int c = 0; c < rank; ++c) k.data<float>()[(h * rank + c) * nope + d] = src[d * rank + c];
                    std::copy_n(src + nope * rank, value * rank, v.data<float>() + h * value * rank);
                }
                tensors.emplace_back(p + "attn_k_b.weight", model_io::quantize_host_matrix(k, DType::F16));
                tensors.emplace_back(p + "attn_v_b.weight", model_io::quantize_host_matrix(v, DType::F16));
            } else {
                auto packed = combined[i];
                packed.desc = Tensor::from_shape({heads * (nope + value), rank}, DType::F32, Backend::CPU);
                tensors.emplace_back(p + "attn_kv_b.weight", std::move(packed));
            }
            if (i == 0) {
                add(p + "ffn_gate.weight", {hidden, hidden});
                add(p + "ffn_up.weight", {hidden, hidden});
                add(p + "ffn_down.weight", {hidden, hidden});
            } else {
                add(p + "ffn_gate_inp.weight", {experts, hidden});
                add(p + "exp_probs_b.bias", {experts});
                auto* bias = tensors.back().second.data<float>();
                bias[0] = 3.6f;
                bias[1] = -3.0f;
                bias[2] = 1.8f;
                bias[3] = 0.7f;
                add(p + "ffn_gate_exps.weight", {experts, expert_width, hidden});
                add(p + "ffn_up_exps.weight", {experts, expert_width, hidden});
                add(p + "ffn_down_exps.weight", {experts, hidden, expert_width});
                add(p + "ffn_gate_shexp.weight", {shared * expert_width, hidden});
                add(p + "ffn_up_shexp.weight", {shared * expert_width, hidden});
                add(p + "ffn_down_shexp.weight", {hidden, shared * expert_width});
            }
        }
        std::string error;
        const auto path = fixture.path.string();
        model_io::LoadedModelRaw raw;
        if (!GgufFile::write_v3(path, meta, tensors, &error) ||
            !model_io::parse_gguf_model_raw(path, 0, raw, &error)) {
            throw std::runtime_error("DeepSeek MLA GGUF roundtrip: " + error);
        }
        const auto& mla = raw.cfg.dense.mla;
        if (mla.query_rank != (lora ? query_rank : 0) || mla.latent_width() != rank + rope ||
            mla.value_dim != value || mla.leading_dense_layers != 1 || raw.cfg.dense.head_dim != key ||
            raw.cfg.dense.num_key_value_heads != 1 || raw.cfg.moe.num_experts != experts ||
            !raw.cfg.moe.sigmoid_router || raw.cfg.moe.top_k != top_k || raw.cfg.moe.expert_weights_scale != router_scale ||
            raw.cfg.moe.expert_groups != (lora ? 2 : 1) || raw.cfg.moe.expert_groups_used != 1 ||
            !raw.layers.at(0).router_w.bytes().empty() || raw.layers.at(1).router_bias.bytes().empty() ||
            !raw.layers.at(1).has_shared || raw.cfg.moe.shared_intermediate_size != shared * expert_width) {
            throw std::runtime_error("DeepSeek MLA roundtrip metadata and tensors");
        }
        std::cout << "  [PASS] DeepSeek MLA GGUF " << (lora ? "Q-LoRA/split KV-B/YaRN" : "direct Q/combined KV-B")
                  << ", dense + MoE, shared and sigmoid+bias routing\n";
#ifdef FUSIONRT_ENABLE_CUDA
        FusionModel model(HybridModelConfig{});
        model.set_moe_gpu_experts(experts);
        if (!model.load_gguf(ctx, path, &error, 0, true)) throw std::runtime_error(error);
        const std::vector<int32_t> prompt{2, 9, 11, 3, 5, 24, 7, 8, 16};
        using Row = std::vector<double>;
        auto weight = [&](const std::string& name) -> const HostTensor& {
            return std::find_if(tensors.begin(), tensors.end(), [&](const auto& entry) { return entry.first == name; })->second;
        };
        auto linear = [](const HostTensor& w, const Row& input, int group = 0) {
            const int64_t n = w.desc.shape[w.desc.ndim - 2], k = w.desc.shape[w.desc.ndim - 1];
            const float* p = w.data<float>() + group * n * k;
            Row out(n);
            for (int64_t i = 0; i < n; ++i)
                out[i] = std::inner_product(input.begin(), input.end(), p + i * k, 0.0);
            return out;
        };
        auto norm = [&](const std::string& name, Row input) {
            const double rms = std::sqrt(std::inner_product(input.begin(), input.end(), input.begin(), 0.0) / input.size() + eps);
            const float* w = weight(name).data<float>();
            for (size_t i = 0; i < input.size(); ++i) input[i] *= w[i] / rms;
            return input;
        };
        auto rotate = [&](double* x, size_t t) {
            for (int j = 0; j < rope / 2; ++j) {
                // For this YaRN fixture the correction interval is [0, 4].
                const double rate = lora ? (1 - 0.75 * j / 4) : 1;
                const double a = t * std::pow(100.0, -2.0 * j / rope) * rate;
                const double c = std::cos(a), s = std::sin(a), re = x[2 * j], im = x[2 * j + 1];
                x[2 * j] = c * re - s * im;
                x[2 * j + 1] = s * re + c * im;
            }
        };
        auto ffn = [&](const std::string& p, const std::string& suffix, const Row& input, int group = 0) {
            auto gate = linear(weight(p + "ffn_gate" + suffix + ".weight"), input, group);
            auto up = linear(weight(p + "ffn_up" + suffix + ".weight"), input, group);
            for (size_t j = 0; j < gate.size(); ++j) gate[j] *= up[j] / (1.0 + std::exp(-gate[j]));
            return linear(weight(p + "ffn_down" + suffix + ".weight"), gate, group);
        };
        std::vector<Row> h;
        for (const int token : prompt) {
            const auto* emb = weight("token_embd.weight").data<float>() + token * hidden;
            h.emplace_back(emb, emb + hidden);
        }
        const double mscale = lora ? 1 + yarn_log_mul * std::log(4.0) : 1;
        const double attention_scale = mscale * mscale / std::sqrt(key);
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            std::vector<Row> queries, keys, expanded_kv;
            for (size_t t = 0; t < h.size(); ++t) {
                auto input = norm(p + "attn_norm.weight", h[t]);
                auto q = lora ? linear(weight(p + "attn_q_b.weight"),
                                       norm(p + "attn_q_a_norm.weight", linear(weight(p + "attn_q_a.weight"), input)))
                              : linear(weight(p + "attn_q.weight"), input);
                for (int a = 0; a < heads; ++a) rotate(q.data() + a * key + nope, t);
                queries.push_back(std::move(q));
                auto latent = linear(weight(p + "attn_kv_a_mqa.weight"), input);
                keys.emplace_back(latent.begin() + rank, latent.end());
                rotate(keys.back().data(), t);
                latent.resize(rank);
                latent = norm(p + "attn_kv_a_norm.weight", std::move(latent));
                expanded_kv.emplace_back();
                for (int a = 0; a < heads; ++a) {
                    const auto full = linear(combined[l], latent, a);
                    expanded_kv.back().insert(expanded_kv.back().end(), full.begin(), full.end());
                }
            }
            for (size_t t = 0; t < h.size(); ++t) {
                Row attended(heads * value, 0.0);
                for (int a = 0; a < heads; ++a) {
                    const double* q = queries[t].data() + a * key;
                    Row scores(t + 1);
                    for (size_t j = 0; j <= t; ++j) {
                        const double* k = expanded_kv[j].data() + a * (nope + value);
                        scores[j] = attention_scale * (std::inner_product(q, q + nope, k, 0.0) +
                            std::inner_product(q + nope, q + key, keys[j].begin(), 0.0));
                    }
                    const double peak = *std::max_element(scores.begin(), scores.end());
                    double mass = 0;
                    for (size_t j = 0; j <= t; ++j) {
                        const double probability = std::exp(scores[j] - peak);
                        mass += probability;
                        const double* v = expanded_kv[j].data() + a * (nope + value) + nope;
                        for (int d = 0; d < value; ++d) attended[a * value + d] += probability * v[d];
                    }
                    for (int d = 0; d < value; ++d) attended[a * value + d] /= mass;
                }
                auto projected = linear(weight(p + "attn_output.weight"), attended);
                for (int d = 0; d < hidden; ++d) h[t][d] += projected[d];
                const auto input = norm(p + "ffn_norm.weight", h[t]);
                auto ff = ffn(p, l == 0 ? "" : "_shexp", input);
                if (l != 0) {
                    auto gates = linear(weight(p + "ffn_gate_inp.weight"), input);
                    for (double& g : gates) g = 1.0 / (1.0 + std::exp(-g));
                    // Group routing selects group 1 even though expert 0 has the highest individual score.
                    const int other = lora ? 3 : 0;
                    const double sum = gates[other] + gates[2];
                    for (const int e : {other, 2}) {
                        const auto update = ffn(p, "_exps", input, e);
                        for (int d = 0; d < hidden; ++d) ff[d] += router_scale * gates[e] / sum * update[d];
                    }
                }
                for (int d = 0; d < hidden; ++d) h[t][d] += ff[d];
            }
        }
        auto rel_l2 = [](const auto& actual, const auto& reference) {
            double diff = 0, magnitude = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                const double err = double(actual[i]) - reference[i];
                diff += err * err;
                magnitude += double(reference[i]) * reference[i];
            }
            const double result = std::sqrt(diff / magnitude);
            if (!(result < 1e-5)) throw std::runtime_error("DeepSeek MLA rel_l2=" + std::to_string(result));
            return result;
        };
        const auto reference = [&](size_t t) { return linear(weight("output.weight"), norm("output_norm.weight", h[t])); };
        const auto batch = model.forward_prefill_cuda(ctx, prompt, 0);
        constexpr size_t latent_bytes = layers * kKvPageTokens * (rank + rope) * sizeof(float);
        constexpr size_t expanded_bytes = layers * kKvPageTokens * heads * (key + value) * sizeof(float);
        static_assert(latent_bytes < expanded_bytes);
        if (model.cuda_cache_memory().kv_bytes != latent_bytes) throw std::runtime_error("DeepSeek MLA latent KV bytes");
        model.reset_kv_and_state(ctx);
        std::vector<float> step;
        double step_error = 0;
        for (size_t t = 0; t < prompt.size(); ++t) {
            step = model.forward_token_cuda(ctx, prompt[t], static_cast<int>(t));
            step_error = std::max(step_error, rel_l2(step, reference(t)));
        }
        const auto eager = model.generate_cuda(ctx, prompt, 16, true, false);
        const auto graph = model.generate_cuda(ctx, prompt, 16, true, true);
        if (eager != graph || graph.size() != prompt.size() + 16) throw std::runtime_error("DeepSeek MLA graph tokens");

        std::vector<int32_t> prefix(kKvPageTokens + 9);
        for (auto& token : prefix) token = static_cast<int32_t>(rng() % vocab);
        auto branch = prefix;
        std::fill(branch.begin() + kKvPageTokens + 4, branch.end(), 21);
        prefix.push_back(31);
        model.reset_kv_and_state(ctx);
        const auto fresh_a = model.forward_prefill_cuda(ctx, prefix, 0);
        model.reset_kv_and_state(ctx);
        const auto fresh_b = model.forward_prefill_cuda(ctx, branch, 0);
        model.reset_kv_and_state(ctx);
        model.set_num_slots(2);
        model.set_kv_vram_budget_bytes(3 * latent_bytes);
        model.prefill_cached_cuda(ctx, {prefix.begin(), prefix.end() - 1});
        int prefilled = 0;
        const auto reuse_b = model.prefill_cached_cuda(ctx, branch, true, &prefilled);
        if (prefilled != 5 || model.cuda_cache_memory().kv_bytes != 3 * latent_bytes)
            throw std::runtime_error("DeepSeek MLA slots must share the complete page and fork the partial page");
        const auto reuse_a = model.prefill_cached_cuda(ctx, prefix, true, &prefilled);
        if (prefilled != 1) throw std::runtime_error("DeepSeek MLA slot resume did not reuse the prefix");
        std::cout << "  [PASS] DeepSeek MLA batch/FP64=" << rel_l2(batch, reference(h.size() - 1))
                  << " step/FP64(max)=" << step_error << " batch/step=" << rel_l2(batch, step)
                  << " latent_bytes=" << latent_bytes << " expanded_bytes=" << expanded_bytes
                  << " Graph/eager=16/16 slot_fork=" << rel_l2(reuse_b, fresh_b)
                  << " slot_resume=" << rel_l2(reuse_a, fresh_a) << "\n";
#endif
    }
}

}  // namespace fusionrt::test
