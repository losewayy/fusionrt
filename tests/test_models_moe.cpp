#include "test_support.h"

namespace fusionrt::test {

struct FrontierFixture {
    static constexpr int hidden = 64, vocab = 16, intermediate = 16, shared = 8;
    std::string arch, path;
    int experts, used = 2, trunk;
    bool mtp;
    std::map<std::string, GgufMetaValue> meta;
    std::vector<std::pair<std::string, HostTensor>> tensors;

    FrontierFixture(bool gemma, bool nextn) :
        arch(gemma ? "gemma4" : "qwen35moe"), path("test-" + arch + (nextn ? "-mtp" : "") + ".gguf"),
        experts(gemma ? 5 : 300), trunk(gemma ? 1 : 2), mtp(nextn) {
        meta["general.architecture"] = arch;
        auto hparam = [&](const char* key, int n) { meta[arch + "." + key] = uint32_t(n); };
        hparam("block_count", trunk + mtp);
        hparam("embedding_length", hidden);
        hparam("feed_forward_length", intermediate);
        hparam("attention.head_count", 1);
        hparam("attention.head_count_kv", 1);
        hparam("attention.key_length", hidden);
        hparam("expert_count", experts);
        hparam("expert_used_count", used);
        hparam("expert_feed_forward_length", intermediate);
        hparam("context_length", 64);
        std::mt19937_64 rng(2821);
        auto normal = [&](const std::string& name, int len, float value) {
            auto t = model_io::make_ones_f32({len});
            std::fill_n(t.data<float>(), len, value);
            tensors.emplace_back(name, std::move(t));
        };
        auto w = [&](const std::string& name, std::vector<int64_t> shape, float scale) {
            tensors.emplace_back(name, scale == 0.0f ?
                model_io::make_zeros_f32(shape) : model_io::make_rand_f32(shape, rng, scale));
        };
        w("token_embd.weight", {vocab, hidden}, 0.2f);
        if (mtp) for (int id = 0; id < vocab; ++id) tensors.back().second.data<float>()[id * hidden] += 2.f;
        normal("output_norm.weight", hidden, 1.1f);
        w("output.weight", {vocab, hidden}, mtp ? 0.0f : 0.1f);
        if (mtp) tensors.back().second.data<float>()[0] = 1.f;
        if (gemma) {
            meta[arch + ".attention.sliding_window_pattern"] = std::vector<bool>{false};
            meta[arch + ".final_logit_softcapping"] = 0.5f;
        } else {
            hparam("expert_shared_feed_forward_length", shared);
            hparam("ssm.conv_kernel", 4);
            hparam("ssm.state_size", 16);
            hparam("ssm.group_count", 1);
            hparam("ssm.time_step_rank", 2);
            hparam("ssm.inner_size", 32);
            hparam("nextn_predict_layers", mtp ? 1 : 0);
            meta[arch + ".expert_weights_scale"] = 0.85f;
        }
        for (int i = 0; i < trunk + mtp; ++i) {
            const auto p = "blk." + std::to_string(i) + ".";
            normal(p + "attn_norm.weight", hidden, 0.95f);
            if (i == 0 && !gemma) {
                w(p + "attn_qkv.weight", {64, hidden}, 0.1f);
                w(p + "attn_gate.weight", {32, hidden}, 0.1f);
                w(p + "ssm_alpha.weight", {2, hidden}, 0.1f);
                w(p + "ssm_beta.weight", {2, hidden}, 0.1f);
                w(p + "ssm_conv1d.weight", {64, 4}, 0.1f);
                normal(p + "ssm_a", 2, -0.75f);
                normal(p + "ssm_dt.bias", 2, 0.01f);
                normal(p + "ssm_norm.weight", 16, 1.f);
                w(p + "ssm_out.weight", {hidden, 32}, 0.f);
            } else {
                w(p + "attn_q.weight", {gemma ? hidden : 2 * hidden, hidden}, 0.1f);
                w(p + "attn_k.weight", {hidden, hidden}, 0.1f);
                w(p + "attn_v.weight", {hidden, hidden}, 0.1f);
                w(p + "attn_output.weight", {hidden, hidden}, i == trunk ? 0.05f : 0.f);
                normal(p + "attn_q_norm.weight", hidden, 1.f);
                normal(p + "attn_k_norm.weight", hidden, 1.f);
            }
            if (gemma) {
                normal(p + "post_attention_norm.weight", hidden, 1.05f);
                normal(p + "ffn_norm.weight", hidden, 0.81f);
                normal(p + "post_ffw_norm.weight", hidden, 0.25f);
                normal(p + "post_ffw_norm_1.weight", hidden, 0.75f);
                normal(p + "post_ffw_norm_2.weight", hidden, 0.91f);
                normal(p + "pre_ffw_norm_2.weight", hidden, 1.23f);
                normal(p + "ffn_gate_inp.scale", hidden, 0.68f);
                normal(p + "layer_output_scale.weight", 1, 0.88f);
                w(p + "ffn_gate.weight", {intermediate, hidden}, 0.08f);
                w(p + "ffn_up.weight", {intermediate, hidden}, 0.09f);
                w(p + "ffn_down.weight", {hidden, intermediate}, 0.08f);
            } else {
                normal(p + "post_attention_norm.weight", hidden, 1.15f);
                w(p + "ffn_gate_inp_shexp.weight", {hidden}, 0.15f);
                w(p + "ffn_gate_shexp.weight", {shared, hidden}, 0.08f);
                w(p + "ffn_up_shexp.weight", {shared, hidden}, 0.08f);
                w(p + "ffn_down_shexp.weight", {hidden, shared}, 0.08f);
            }
            w(p + "ffn_gate_inp.weight", {experts, hidden}, 0.15f);
            auto* router = tensors.back().second.data<float>();
            // Two opposite gates above expert 256 must remain eligible.
            router[(experts - 1) * hidden] = 12.f;
            router[(experts - 2) * hidden] = -12.f;
            w(p + "ffn_gate_exps.weight", {experts, intermediate, hidden}, 0.07f);
            w(p + "ffn_up_exps.weight", {experts, intermediate, hidden}, 0.07f);
            w(p + "ffn_down_exps.weight", {experts, hidden, intermediate}, 0.065f);
            normal(p + "ffn_gate_exps.scale", experts, 0.81f);
            normal(p + "ffn_up_exps.scale", experts, 1.05f);
            normal(p + "ffn_down_exps.scale", experts, 1.21f);
            for (size_t j = tensors.size() - 3; j < tensors.size(); ++j) {
                auto* s = tensors[j].second.data<float>();
                for (int e = 0; e < experts; ++e) s[e] += 0.02f * (e % 5);
            }
            if (i == trunk) {
                normal(p + "nextn.enorm.weight", hidden, 1.f);
                normal(p + "nextn.hnorm.weight", hidden, 0.7f);
                w(p + "nextn.eh_proj.weight", {hidden, 2 * hidden}, 0.0f);
                auto* proj = tensors.back().second.data<float>();
                for (int j = 0; j < hidden; ++j) proj[j * (2 * hidden) + hidden + j] = 1.f;
                normal(p + "nextn.eh_proj.scale", 1, 1.25f);
            }
        }
        std::string err;
        if (!GgufFile::write_v3(path, meta, tensors, &err)) throw std::runtime_error(err);
    }

    ~FrontierFixture() { std::error_code ec; std::filesystem::remove(path, ec); }

    const HostTensor& tensor(const std::string& name) const {
        const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const auto& t) { return t.first == name; });
        if (it == tensors.end()) throw std::runtime_error("Missing fixture tensor " + name);
        return it->second;
    }
};

std::vector<float> frontier_reference(const FrontierFixture& f, int32_t token) {
    const bool gemma = f.arch == "gemma4";
    constexpr int H = FrontierFixture::hidden, I = FrontierFixture::intermediate;
    const auto* emb = f.tensor("token_embd.weight").data<float>() + token * H;
    std::vector<float> hidden(emb, emb + H);
    if (gemma) for (auto& x : hidden) x *= std::sqrt(static_cast<float>(H));

    auto rms = [](std::vector<float> in, const HostTensor& w) {
        const auto factor = 1.0 / std::sqrt(codec_reference_dot(in.data(), in.data(), static_cast<int>(in.size())) /
                                           in.size() + 1e-6);
        for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(in[i] * factor * w.data<float>()[i]);
        return in;
    };
    auto activate = [&](double x) {
        return gemma ? x * 0.5 * (1.0 + std::tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))
                     : x / (1.0 + std::exp(-x));
    };
    auto branch = [&](const std::vector<float>& input, const std::string& prefix, const char* suffix,
                       int width, int expert, bool scaled) {
        const auto* gate = f.tensor(prefix + "ffn_gate" + suffix + ".weight").data<float>() + expert * width * H;
        const auto* up = f.tensor(prefix + "ffn_up" + suffix + ".weight").data<float>() + expert * width * H;
        const auto* down = f.tensor(prefix + "ffn_down" + suffix + ".weight").data<float>() + expert * width * H;
        const double gs = scaled ? f.tensor(prefix + "ffn_gate_exps.scale").data<float>()[expert] : 1.0;
        const double us = scaled ? f.tensor(prefix + "ffn_up_exps.scale").data<float>()[expert] : 1.0;
        std::vector<float> act(width), output(H);
        for (int i = 0; i < width; ++i)
            act[i] = static_cast<float>(activate(gs * codec_reference_dot(gate + i * H, input.data(), H)) *
                                       us * codec_reference_dot(up + i * H, input.data(), H));
        for (int h = 0; h < H; ++h)
            output[h] = static_cast<float>(codec_reference_dot(down + h * width, act.data(), width));
        return output;
    };

    // Attention output projections on the trunk are zero. This isolates the
    // FFN math while CUDA and Vulkan still execute the underlying blocks.
    for (int layer = 0; layer < f.trunk; ++layer) {
        const auto p = "blk." + std::to_string(layer) + ".";
        const auto normalized = rms(hidden, f.tensor(p + (gemma ? "ffn_norm.weight" : "post_attention_norm.weight")));
        const auto x = gemma ? rms(hidden, f.tensor(p + "pre_ffw_norm_2.weight")) : normalized;
        auto route = gemma ? rms(hidden, f.tensor(p + "ffn_gate_inp.scale")) : normalized;
        if (gemma) for (auto& r : route) r /= std::sqrt(static_cast<float>(H));
        const auto* router = f.tensor(p + "ffn_gate_inp.weight").data<float>();
        std::vector<std::pair<double, int>> ranked;
        for (int e = 0; e < f.experts; ++e)
            ranked.emplace_back(codec_reference_dot(router + e * H, route.data(), H), e);
        std::partial_sort(ranked.begin(), ranked.begin() + f.used, ranked.end(), [](const auto& a, const auto& b) {
            return a.first == b.first ? a.second < b.second : a.first > b.first;
        });
        double mass = 0.0;
        for (int k = 0; k < f.used; ++k) mass += std::exp(ranked[k].first - ranked[0].first);
        std::vector<float> routed(H, 0.f);
        for (int k = 0; k < f.used; ++k) {
            const auto [v, e] = ranked[k];
            const double dw = f.tensor(p + "ffn_down_exps.scale").data<float>()[e];
            const double weight = std::exp(v - ranked[0].first) / mass * dw *
                (gemma ? 1.0 : std::get<float>(f.meta.at(f.arch + ".expert_weights_scale")));
            const auto expert = branch(x, p, "_exps", I, e, true);
            for (int h = 0; h < H; ++h) routed[h] += static_cast<float>(weight * expert[h]);
        }
        if (gemma) {
            auto shared = rms(branch(normalized, p, "", I, 0, false), f.tensor(p + "post_ffw_norm_1.weight"));
            routed = rms(std::move(routed), f.tensor(p + "post_ffw_norm_2.weight"));
            for (int h = 0; h < H; ++h) routed[h] += shared[h];
            routed = rms(std::move(routed), f.tensor(p + "post_ffw_norm.weight"));
            for (int h = 0; h < H; ++h) hidden[h] = (hidden[h] + routed[h]) * 0.88f;
        } else {
            const double g = codec_reference_dot(f.tensor(p + "ffn_gate_inp_shexp.weight").data<float>(), x.data(), H);
            const auto shared = branch(x, p, "_shexp", FrontierFixture::shared, 0, false);
            for (int h = 0; h < H; ++h) hidden[h] += routed[h] + static_cast<float>(shared[h] / (1.0 + std::exp(-g)));
        }
    }
    const auto n = rms(hidden, f.tensor("output_norm.weight"));
    std::vector<float> logits(FrontierFixture::vocab);
    for (int id = 0; id < FrontierFixture::vocab; ++id) {
        const auto z = codec_reference_dot(f.tensor("output.weight").data<float>() + id * H, n.data(), H);
        logits[id] = static_cast<float>(gemma ? 0.5 * std::tanh(z / 0.5) : z);
    }
    return logits;
}

void test_frontier_gguf() {
    for (const bool gemma : {false, true}) {
        const FrontierFixture f(gemma, false);
        model_io::LoadedModelRaw raw;
        std::string error;
        if (!model_io::parse_gguf_model_raw(f.path, 64, raw, &error)) throw std::runtime_error(error);
        if (raw.cfg.moe.num_experts != f.experts || raw.cfg.num_layers != f.trunk ||
            (gemma ? raw.layers[0].router_scale.bytes().empty() : raw.layers[0].kind != 1))
            throw std::runtime_error("Frontier GGUF geometry");
        const auto ref = frontier_reference(f, 3);
        if (!std::isfinite(ref[0])) throw std::runtime_error("Frontier CPU reference is finite");
    }
    const FrontierFixture f(false, true);
    model_io::LoadedModelRaw raw;
    std::string error;
    if (!model_io::parse_gguf_model_raw(f.path, 64, raw, &error) || raw.cfg.num_layers != 2 ||
        raw.layers.size() != 3 || raw.mtp_projection.desc.shape[1] != 2 * f.hidden || raw.mtp_projection_scale != 1.25f)
        throw std::runtime_error("Qwen NextN GGUF roundtrip " + error);
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_frontier(CudaContext& ctx) {
    {
        auto& arena = ctx.scratch_arena();
        const auto mark = arena.mark();
        auto host = model_io::make_zeros_f32({513});
        host.data<float>()[512] = 10.f;
        host.data<float>()[257] = 9.f;
        const auto input = ctx.upload(host, false);
        auto ids = arena.alloc_tensor({2}, DType::I32);
        auto scores = arena.alloc_tensor({2}, DType::F32);
        cuda_ops::moe_topk_router_f32(input.ptr<float>(), ids.ptr<int32_t>(), scores.ptr<float>(), 513, 2, true, ctx.stream());
        const auto routed_ids = ctx.download(ids);
        if (routed_ids.data<int32_t>()[0] != 512 || routed_ids.data<int32_t>()[1] != 257)
            throw std::runtime_error("CUDA preserves expert IDs beyond 256");
        arena.reset(mark);
    }
    const std::vector<int32_t> prompt{1, 3, 2};
    for (const bool gemma : {false, true}) {
        const FrontierFixture f(gemma, false);
        const auto ref = frontier_reference(f, prompt.back());
        std::vector<int32_t> expected;
        double max_error = 0;
        for (const int resident : {f.experts, gemma ? 3 : 270, 0}) {
            FusionModel model(HybridModelConfig{});
            model.set_moe_gpu_experts(resident);
            std::string error;
            if (!model.load_gguf(ctx, f.path, &error, 64, true)) throw std::runtime_error(error);
            const auto logits = model.forward_prefill_cuda(ctx, prompt, 0, true);
            max_error = std::max(max_error, codec_rel_l2(logits, ref));
            require_codec_close(DType::F32, "Frontier MoE CPU reference", logits, ref);
            for (const bool graph : {false, true}) {
                const auto ids = model.generate_cuda(ctx, prompt, 8, true, graph);
                if (expected.empty()) expected = ids;
                if (ids != expected) throw std::runtime_error("Frontier MoE greedy ID parity");
            }
        }
#ifdef FUSIONRT_ENABLE_VULKAN
        if (!gemma && !VkContext::enumerate_devices().empty()) {
            VkContext vk(0, 2, 8);
            FusionModel model(HybridModelConfig{});
            std::string error;
            if (!model.load_gguf_vulkan(vk, f.path, &error, 64)) throw std::runtime_error(error);
            if (model.generate_vulkan(vk, prompt, 8) != expected)
                throw std::runtime_error("Qwen Vulkan greedy ID parity");
        }
#endif
        std::cout << "  [PASS] " << f.arch << " CPU reference / CUDA eager, graph and tiered exact tokens, max rel_l2="
                  << max_error << '\n';
    }

    FrontierFixture f(false, true);
    FusionModel model(HybridModelConfig{});
    model.enable_native_mtp(true);
    std::string error;
    if (!model.load_gguf(ctx, f.path, &error, 64, true)) throw std::runtime_error(error);
    GenerationOptions mtp;
    mtp.spec_mtp = true;
    mtp.draft_tokens = 3;
    mtp.spec_p_min = 0.f;
    GenerationStats stats;
    const auto normal = model.generate_cuda(ctx, prompt, 16);
    const auto speculative = model.generate_cuda(ctx, prompt, 16, true, true, &stats, false, 0, -1, mtp);
    if (normal != speculative || stats.spec_proposed == 0 || stats.spec_accepted == 0 ||
        !stats.used_cuda_graph || stats.mtp_verify_graph_steps == 0 || stats.mtp_draft_graph_steps == 0 ||
        stats.mtp_reused_kv_rows == 0 || stats.mtp_rewound_kv_rows == 0)
        throw std::runtime_error("Qwen integrated MTP verification or acceptance");
    std::cout << "  [PASS] Qwen native MTP exact greedy IDs, accepted " << stats.spec_accepted
              << '/' << stats.spec_proposed << '\n';
    mtp.spec_p_min = 1.f;
    const auto cut_off = model.generate_cuda(ctx, prompt, 16, true, true, &stats, false, 0, -1, mtp);
    if (cut_off != normal || stats.spec_proposed != 0 || stats.mtp_cutoffs == 0 ||
        stats.mtp_verify_graph_steps == 0 || stats.mtp_draft_graph_steps == 0)
        throw std::runtime_error("Native MTP confidence cutoff must fall back to exact single-row verification");
    std::cout << "  [PASS] Native MTP cutoff, graph verification and KV reuse preserve exact greedy IDs\n";
    mtp.spec_p_min = 0.f;

    ServeSpeculation serve;
    serve.mtp = true;
    serve.draft_tokens = 3;
    serve.p_min = 0.f;
    const auto image = ctx.upload(make_ones_f32({1, f.hidden}), false);
    const InputEmbedding span{0, image};
    const auto image_opts = serve.for_inputs(std::span(&span, 1));
    (void)model.generate_cuda(ctx, prompt, 16, true, true, &stats, false, 0, -1, image_opts);
    if (stats.speculative_active || stats.spec_proposed)
        throw std::runtime_error("Vision generation must bypass native MTP");
    auto text_opts = serve.for_inputs({});
    if (model.generate_cuda(ctx, prompt, 16, true, true, &stats, false, 0, -1, text_opts) != normal ||
        !stats.speculative_active || stats.spec_accepted == 0)
        throw std::runtime_error("Text generation must resume native MTP after a vision request");
    text_opts.sampling.temperature = 0.7f;
    text_opts.sampling.top_k = 4;
    text_opts.sampling.seed = 421;
    const auto sampled = model.generate_cuda(ctx, prompt, 16, true, true, &stats, false, 0, -1, text_opts);
    if (!stats.speculative_active || sampled.size() != prompt.size() + 16 ||
        model.generate_cuda(ctx, prompt, 16, true, true, nullptr, false, 0, -1, text_opts) != sampled)
        throw std::runtime_error("Seeded stochastic native MTP must reproduce its own output");
    std::cout << "  [PASS] Native MTP resumes after vision and seeded sampling is reproducible\n";

    auto conflicting_head = model_io::make_zeros_f32({f.vocab, f.hidden});
    // Rows 1 and 2 have opposite logits. The trunk's greedy choice is 0.
    conflicting_head.data<float>()[f.hidden] = 1.f;
    conflicting_head.data<float>()[2 * f.hidden] = -1.f;
    f.tensors.emplace_back("blk.2.nextn.shared_head_head.weight", std::move(conflicting_head));
    std::filesystem::remove(f.path);
    f.path = "test-qwen35moe-mtp-reject.gguf";
    if (!GgufFile::write_v3(f.path, f.meta, f.tensors, &error) ||
        !model.load_gguf(ctx, f.path, &error, 64, true)) throw std::runtime_error(error);
    const auto rejected = model.generate_cuda(ctx, prompt, 16, true, true, &stats, false, 0, -1, mtp);
    if (rejected != normal || stats.spec_proposed == 0 || stats.spec_accepted == stats.spec_proposed)
        throw std::runtime_error("Native MTP rollback after rejected drafts");
    std::cout << "  [PASS] Qwen native MTP rejected drafts preserve the target IDs, accepted "
              << stats.spec_accepted << '/' << stats.spec_proposed << '\n';
}
#endif

void test_moe_autofit() {
    // CUDA availability already accounts for external allocations. The DXGI residency cap is independent.
    if (std::min(size_t{600}, wddm_model_available(1000, 100, 0)) != 600 ||
        wddm_model_available(1000, 950, 100) != 25 || wddm_model_available(1000, 950, 0) != 0)
        throw std::runtime_error("WDDM budget must deduct process usage and credit reclaimable bytes once");
    constexpr size_t unit = 1024 * 1024;
    const std::array costs{MoeLayerCost{4, 2, 512, {unit, 0, 0}}, MoeLayerCost{4, 2, 512, {unit, 0, 0}},
                           MoeLayerCost{4, 2, 512, {unit, 0, 0}}};
    CostHardware hw;
    hw.hbm_bw_gbps = 512;
    hw.cpu_q4_gemv_gbps = 30;
    hw.pcie_h2d_gbps = 14;
    hw.pcie_latency_us = 12;
    const auto mixed = select_moe_placement(costs, 7 * unit, hw, 50);
    if (mixed.gpu_experts != std::vector<int>{4, 3, 0} || mixed.choice != 2 ||
        mixed.candidate_us[2] >= mixed.candidate_us[0] || mixed.candidate_us[2] >= mixed.candidate_us[1])
        throw std::runtime_error("MoE placement must score all layouts against the same byte budget");
    if (select_moe_placement(costs, unit - 1, hw, 50).gpu_experts != std::vector<int>{0, 0, 0} ||
        select_moe_placement(costs, 12 * unit, hw, 50).gpu_experts != std::vector<int>{4, 4, 4})
        throw std::runtime_error("MoE placement mishandled unfittable or full residency");
    const auto precision = select_moe_kv_precision(costs, 10 * unit, 12 * unit, 5 * unit, 3 * unit, hw, 50);
    if (precision.dtype != DType::Q8_0 || precision.experts.gpu_experts != std::vector<int>{4, 3, 0} ||
        select_moe_kv_precision(costs, 20 * unit, 12 * unit, 5 * unit, 3 * unit, hw, 50).dtype != DType::F16 ||
        select_moe_kv_precision(costs, 20 * unit, 7 * unit, 5 * unit, 3 * unit, hw, 50).dtype != DType::F16)
        throw std::runtime_error("Auto KV precision requires spill and strictly improved expert placement");
    std::cout << "  [PASS] Per-layer MoE auto-fit budget and candidate scores\n";
}

void test_gdn_loader_shapes() {
    constexpr int H = 64, D = 16, N = 2, C = 3 * N * D;
    struct Fixture {
        std::filesystem::path dir = "test-review-gdn-shapes";
        Fixture() { std::filesystem::create_directory(dir); }
        ~Fixture() {
            std::error_code ec;
            for (const char* file : {"model.gguf", "model.safetensors", "config.json"})
                std::filesystem::remove(dir / file, ec);
            std::filesystem::remove(dir, ec);
        }
    } fixture;
    const auto gguf = (fixture.dir / "model.gguf").string(), st = (fixture.dir / "model.safetensors").string();
    {
        std::ofstream cfg(fixture.dir / "config.json");
        cfg << R"({"model_type":"qwen3_5","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":2,
                   "intermediate_size":64,"vocab_size":32,"linear_num_key_heads":2,"linear_num_value_heads":2,
                   "linear_key_head_dim":16,"linear_value_head_dim":16,"linear_conv_kernel_dim":4})";
    }
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("qwen35")}, {"qwen35.block_count", uint32_t{1}},
        {"qwen35.embedding_length", uint32_t{H}}, {"qwen35.feed_forward_length", uint32_t{H}},
        {"qwen35.attention.head_count", uint32_t{N}},
        {"qwen35.ssm.group_count", uint32_t{N}}, {"qwen35.ssm.time_step_rank", uint32_t{N}},
        {"qwen35.ssm.state_size", uint32_t{D}}, {"qwen35.ssm.inner_size", uint32_t{N * D}}
    };
    using Weights = std::vector<std::pair<std::string, HostTensor>>;
    Weights gw{{"token_embd.weight", HostTensor({32, H}, DType::F32)},
               {"output_norm.weight", make_ones_f32({H})}};
    Weights sw{{"model.embed_tokens.weight", HostTensor({32, H}, DType::F32)},
               {"model.norm.weight", make_ones_f32({H})}};
    struct TensorSpec { const char* field; const char* gguf; const char* st; std::vector<int64_t> shape; };
    const TensorSpec tensors[]{
        {"conv_w", "ssm_conv1d.weight", "linear_attn.conv1d.weight", {C, 4}},
        {"conv_bias", "ssm_conv1d.bias", "linear_attn.conv1d.bias", {C}},
        {"A_log", "ssm_a", "linear_attn.A_log", {N}},
        {"dt_bias", "ssm_dt.bias", "linear_attn.dt_bias", {N}},
        {"gdn_norm_w", "ssm_norm.weight", "linear_attn.norm.weight", {D}},
        {"qkv_proj", "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", {C, H}},
        {"z_proj", "attn_gate.weight", "linear_attn.in_proj_z.weight", {N * D, H}},
        {"a_proj", "ssm_alpha.weight", "linear_attn.in_proj_a.weight", {N, H}},
        {"b_proj", "ssm_beta.weight", "linear_attn.in_proj_b.weight", {N, H}},
        {"o_proj", "ssm_out.weight", "linear_attn.out_proj.weight", {H, N * D}},
        {"attn_norm", "attn_norm.weight", "input_layernorm.weight", {H}},
        {"mlp_norm", "ffn_norm.weight", "post_attention_layernorm.weight", {H}},
        {"gate", "ffn_gate.weight", "mlp.gate_proj.weight", {H, H}},
        {"up", "ffn_up.weight", "mlp.up_proj.weight", {H, H}},
        {"down", "ffn_down.weight", "mlp.down_proj.weight", {H, H}}
    };
    for (const auto& spec : tensors) {
        gw.emplace_back(std::string("blk.0.") + spec.gguf, HostTensor::create(spec.shape, DType::F32));
        const auto shape = std::string_view(spec.field) == "conv_w" ? std::vector<int64_t>{C, 1, 4} : spec.shape;
        sw.emplace_back(std::string("model.layers.0.") + spec.st, HostTensor::create(shape, DType::F32));
    }
    std::string error;
    auto check = [&](const Weights& gguf_weights, const Weights& st_weights, std::string_view bad_field) {
        model_io::LoadedModelRaw raw, safe;
        if (!GgufFile::write_v3(gguf, meta, gguf_weights, &error)) throw std::runtime_error(error);
        const bool gguf_ok = model_io::parse_gguf_model_raw(gguf, 32, raw, &error);
        if (gguf_ok != bad_field.empty() || (!gguf_ok && error.find(bad_field) == error.npos))
            throw std::runtime_error("GGUF GDN tensor validation: " + error);
        if (!SafetensorsFile::write(st, {}, st_weights, &error)) throw std::runtime_error(error);
        const bool st_ok = model_io::parse_safetensors_model_raw(st, 32, safe, &error);
        if (st_ok != bad_field.empty() || (!st_ok && error.find(bad_field) == error.npos))
            throw std::runtime_error("Safetensors GDN tensor validation: " + error);
    };
    check(gw, sw, {});
    for (size_t i = 0; i < 12; ++i) {
        auto bad_gguf = gw, bad_st = sw;
        bad_gguf[i + 2].second = HostTensor({1}, DType::F32);
        bad_st[i + 2].second = HostTensor({1}, DType::F32);
        check(bad_gguf, bad_st, tensors[i].field);
    }
    std::cout << "  [PASS] GGUF and Safetensors GDN tensor shapes, including convolution bias\n";
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_gemma4_gguf() {
    constexpr int layers = 4, hidden = 64, vocab = 64, nq = 4, window = 32, seq = 136, ple = 64;
    constexpr float cap = 0.5f;
    struct Fixture {
        std::filesystem::path path = "test-gemma4.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    const std::vector<bool> swa{true, false, true, false};
    const std::vector<uint32_t> ff{64, 96, 128, 96};
    std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("gemma4")}, {"gemma4.block_count", uint32_t{layers}},
        {"gemma4.embedding_length", uint32_t{hidden}}, {"gemma4.attention.head_count", uint32_t{nq}},
        {"gemma4.attention.head_count_kv", std::vector<uint32_t>{2, 1, 0, 0}},
        {"gemma4.attention.key_length", uint32_t{512}}, {"gemma4.attention.key_length_swa", uint32_t{256}},
        {"gemma4.attention.sliding_window", uint32_t{window}},
        {"gemma4.attention.sliding_window_pattern", swa}, {"gemma4.attention.shared_kv_layers", uint32_t{2}},
        {"gemma4.rope.freq_base", 1000000.0f}, {"gemma4.rope.freq_base_swa", 10000.0f},
        {"gemma4.feed_forward_length", ff}, {"gemma4.final_logit_softcapping", cap},
        {"gemma4.embedding_length_per_layer_input", uint32_t{ple}}
    };

    std::mt19937_64 rng(391);
    auto output_factors = make_ones_f32({256});
    output_factors.data<float>()[255] = 1.0e9f;
    HostTensor ple_tokens({vocab, layers * ple}, DType::Q5_K);
    for (int id = 0; id < vocab; ++id) {
        auto& block = ple_tokens.data<BlockQ5_K>()[id];
        block.d = fp32_to_fp16(0.00004f);
        block.dmin = fp32_to_fp16(0.0003f);
        for (auto& x : block.scales) x = static_cast<uint8_t>(rng());
        for (auto& x : block.qh) x = static_cast<uint8_t>(rng());
        for (auto& x : block.qs) x = static_cast<uint8_t>(rng());
    }
    HostTensor ple_model({layers * ple, hidden}, DType::BF16);
    const auto source = rand_f32({layers * ple, hidden}, rng, 0.035f);
    convert_f32_to_bf16(source.data<float>(), ple_model.data<uint16_t>(), source.desc.numel());
    std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", rand_f32({vocab, hidden}, rng, 0.1f)},
        {"output_norm.weight", make_ones_f32({hidden})},
        {"rope_freqs.weight", std::move(output_factors)},
        {"per_layer_token_embd.weight", std::move(ple_tokens)},
        {"per_layer_model_proj.weight", std::move(ple_model)},
        {"per_layer_proj_norm.weight", make_ones_f32({ple})}
    };
    for (int i = 0; i < layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const int hd = swa[i] ? 256 : 512, kv_heads = swa[i] ? 2 : 1;
        const int width = static_cast<int>(ff[i]);
        auto matrix = [&](const char* name, int r, int c) {
            weights.emplace_back(p + name, rand_f32({r, c}, rng, 0.035f));
        };
        auto norm = [&](const char* name, int n, float value = 1.0f) {
            auto weight = make_ones_f32({n});
            for (int j = 0; j < n; ++j) weight.data<float>()[j] = value;
            weights.emplace_back(p + name, std::move(weight));
        };
        norm("attn_norm.weight", hidden);
        norm("attn_q_norm.weight", hd);
        if (i < 2) {
            matrix("attn_k.weight", kv_heads * hd, hidden);
            if (swa[i]) matrix("attn_v.weight", kv_heads * hd, hidden);
            norm("attn_k_norm.weight", hd);
        }
        matrix("attn_q.weight", nq * hd, hidden);
        matrix("attn_output.weight", hidden, nq * hd);
        norm("post_attention_norm.weight", hidden, 0.35f);
        norm("ffn_norm.weight", hidden);
        matrix("ffn_gate.weight", width, hidden);
        matrix("ffn_up.weight", width, hidden);
        matrix("ffn_down.weight", hidden, width);
        norm("post_ffw_norm.weight", hidden, 0.25f);
        matrix("inp_gate.weight", ple, hidden);
        matrix("proj.weight", hidden, ple);
        norm("post_norm.weight", hidden, 0.13f);
        norm("layer_output_scale.weight", 1, 0.9f);
    }
    CudaContext ctx(0, 0, 64);
    FusionModel model(HybridModelConfig{});
    std::string err;
    if (!GgufFile::write_v3(fixture.path.string(), meta, weights, &err) ||
        !model.load_gguf(ctx, fixture.path.string(), &err, 160)) {
        throw std::runtime_error("Gemma 4 GGUF load: " + err);
    }
    const auto& loaded = model.dense_layers();
    if (model.config().embed_scale != 8.0f || model.config().final_logit_softcapping != cap ||
        loaded.size() != layers || loaded[0].cfg.head_dim != 256 ||
        loaded[0].sliding_window != window || loaded[0].rope_factors.data ||
        loaded[1].cfg.head_dim != 512 || loaded[1].v_proj.data || !loaded[1].rope_factors.data ||
        loaded[2].kv_source_layer != 0 || loaded[3].kv_source_layer != 1 ||
        loaded[2].k_proj.data || loaded[3].v_proj.data || !loaded[3].ffn_post_norm_w.data ||
        !loaded[3].ple_gate.data || !loaded[3].ple_post_norm.data || loaded[3].output_scale != 0.9f) {
        throw std::runtime_error("Gemma 4 GGUF layer wiring");
    }

    std::vector<int32_t> prompt(seq);
    for (auto& token : prompt) token = static_cast<int32_t>(rng() % vocab);
    const int prefill_batch = ctx.prefill_batch();
    // Small chunks keep the BF16 projection on the FP32-input GEMV path.
    ctx.set_prefill_batch(8);
    const auto batched = model.forward_prefill_cuda(ctx, prompt, 0, false);
    ctx.set_prefill_batch(prefill_batch);
    const size_t expected_kv_bytes = 2 * 2 * kKvPageTokens * (2 * 256 + 512) * sizeof(float);
    if (model.cuda_cache_memory().kv_bytes != expected_kv_bytes) {
        throw std::runtime_error("Gemma 4 shared layers allocated their own KV pages");
    }
    model.reset_kv_and_state(ctx);
    std::vector<float> sequential;
    for (int i = 0; i < seq; ++i) sequential = model.forward_token_cuda(ctx, prompt[i], i, false);
    double squared_error = 0.0, squared_ref = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const double diff = batched[i] - sequential[i];
        squared_error += diff * diff;
        squared_ref += double(sequential[i]) * sequential[i];
        if (std::abs(batched[i]) > cap) throw std::runtime_error("Gemma 4 logits exceed softcap");
    }
    const double rel_l2 = std::sqrt(squared_error / squared_ref);
    if (!(rel_l2 < 1e-5)) throw std::runtime_error("Gemma 4 batched/sequential rel_l2: " + std::to_string(rel_l2));
    model.reset_kv_and_state(ctx);
    const auto optimized = model.forward_prefill_cuda(ctx, prompt, 0);
    // Batched cuBLAS uses FP16 activations for the BF16 PLE projection.
    expect_close(optimized, {batched.begin(), batched.end()}, 3e-4, 3e-4, "Gemma 4 optimized prefill");

    const auto eager = model.generate_cuda(ctx, prompt, 16, true, false);
    const auto graph = model.generate_cuda(ctx, prompt, 16, true, true);
    if (eager != graph || graph.size() != seq + 16) {
        throw std::runtime_error("Gemma 4 CUDA Graph/eager tokens differ");
    }
    std::cout << "  [PASS] Gemma 4 four-layer GGUF, prefill rel_l2=" << rel_l2
              << ", Graph/eager 16/16 tokens, shared KV bytes=" << expected_kv_bytes << "\n";
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_graph_and_moe_offload(CudaContext& ctx) {
    std::cout << "[5/7] Testing CUDA Graph full-decode parity & Sparse MoE CPU-offload parity...\n";
    // 5a. Hybrid 3:1 CUDA Graph vs Eager parity
    {
        HybridModelConfig cfg;
        cfg.arch = "hybrid";
        cfg.vocab_size = 256;
        cfg.num_layers = 4;
        cfg.dense.hidden_size = 256;
        cfg.dense.num_attention_heads = 4;
        cfg.dense.num_key_value_heads = 2;
        cfg.dense.head_dim = 64;
        cfg.dense.intermediate_size = 512;
        cfg.dense.max_seq_len = 256;
        cfg.gdn.hidden_size = 256;
        cfg.gdn.num_k_heads = 4;
        cfg.gdn.num_v_heads = 4;
        cfg.gdn.head_k_dim = 64;
        cfg.gdn.head_v_dim = 64;
        cfg.gdn.intermediate_size = 512;

        FusionModel m(cfg);
        m.init_synthetic(ctx, 505);
        const auto& st = m.fused_ir_graph().stats;
        if (st.sibling_merges < 4 || st.norm_linear_fusions < 8 || st.add_norm_linear_fusions < 4 || st.opt_nodes >= st.raw_nodes) {
            throw std::runtime_error("MiniGraph optimize_graph did not fuse expected ops: " + m.fused_ir_graph().summary());
        }
        std::cout << "  [PASS/CHECK] " << m.fused_ir_graph().summary() << "\n";
        m.reset_kv_and_state(ctx);
        auto logits_fused = m.forward_token_cuda(ctx, 1, 0, true, false);
        m.reset_kv_and_state(ctx);
        auto logits_raw = m.forward_token_cuda(ctx, 1, 0, false, false);
        std::vector<double> ref_l(logits_raw.begin(), logits_raw.end());
        expect_close(logits_fused, ref_l, 1e-4, 1e-4, "MiniGraph fused vs raw IR logits parity");

        m.reset_kv_and_state(ctx);
        auto out_eager = m.generate_cuda(ctx, {1, 10, 20}, 4, true, false);
        GenerationStats graph_stats;
        auto out_graph = m.generate_cuda(ctx, {1, 10, 20}, 4, true, true, &graph_stats);
        if (out_eager != out_graph || !graph_stats.used_cuda_graph) {
            throw std::runtime_error("CUDA Graph token generation diverged from eager!");
        }
        GenerationOptions sampling_options;
        sampling_options.sampling.temperature = 0.7f;
        sampling_options.warn_on_graph_fallback = true;
        (void)m.generate_cuda(ctx, {1, 10, 20}, 4, true, true, &graph_stats, false, 0, -1, sampling_options);
        if (graph_stats.used_cuda_graph) throw std::runtime_error("Stochastic decoding must fall back to eager");
        std::cout << "  [PASS/CHECK] CUDA Graph 4-layer Hybrid 3:1 token parity exact match\n";
    }

    // 5a-2. Sparse MoE CUDA Graph vs Eager parity
    ctx.weight_arena().reset();
    ctx.scratch_arena().reset();
    {
        HybridModelConfig cfg;
        cfg.arch = "moe";
        cfg.vocab_size = 256;
        cfg.num_layers = 2;
        cfg.moe.hidden_size = 128;
        cfg.moe.intermediate_size = 128;
        cfg.moe.num_experts = 4;
        cfg.moe.top_k = 2;
        cfg.moe.gpu_resident_experts = 4;
        cfg.moe.shared_intermediate_size = 128;

        FusionModel m(cfg);
        m.init_synthetic(ctx, 506);
        auto out_eager = m.generate_cuda(ctx, {1, 5, 9}, 4, true, false);
        auto out_graph = m.generate_cuda(ctx, {1, 5, 9}, 4, true, true);
        if (out_eager != out_graph) {
            throw std::runtime_error("CUDA Graph MoE token generation diverged from eager!");
        }
        std::cout << "  [PASS/CHECK] CUDA Graph 2-layer Sparse MoE token parity exact match\n";

        cfg.num_layers = 3;
        cfg.gpu_experts = {4, 2, 0};
        FusionModel mixed(cfg);
        mixed.init_synthetic(ctx, 506);
        GenerationOptions options;
        options.warn_on_graph_fallback = true;
        GenerationStats graph_stats;
        const auto eager = mixed.generate_cuda(ctx, {1, 5, 9}, 8, true, false);
        const auto prefix = mixed.generate_cuda(ctx, {1, 5, 9}, 8, true, true, &graph_stats, false, 0, -1, options);
        if (!graph_stats.used_cuda_graph || eager != prefix)
            throw std::runtime_error("CUDA MoE prefix graph must preserve greedy tokens across full, partial and CPU layers");
        mixed.reset_kv_and_state(ctx);
        const auto eager_logits = mixed.forward_token_cuda(ctx, 1, 0, true, false);
        mixed.reset_kv_and_state(ctx);
        if (eager_logits != mixed.forward_token_cuda(ctx, 1, 0, true, true))
            throw std::runtime_error("CUDA MoE prefix graph must preserve logits");
        std::cout << "  [PASS] MoE full/hybrid/CPU prefix Graph logits and 8/8 token parity\n";
    }

    // 5b. Sparse MoE all-GPU vs partial CPU-offload parity
    ctx.weight_arena().reset();
    ctx.scratch_arena().reset();
    {
        MoEConfig mc;
        mc.hidden_size = 128;
        mc.intermediate_size = 128;
        mc.num_experts = 6;
        mc.top_k = 2;
        mc.has_shared_expert = true;
        mc.shared_intermediate_size = 128;

        std::mt19937_64 rng(606);
        HostTensor h_x = rand_f32({mc.hidden_size}, rng, 0.3f);
        HostTensor h_nw = rand_f32({mc.hidden_size}, rng, 0.5f);
        HostTensor h_rw = rand_f32({mc.num_experts, mc.hidden_size}, rng, 0.2f);
        HostTensor h_gu = rand_f32({mc.num_experts, 2 * mc.intermediate_size, mc.hidden_size}, rng, 0.1f);
        HostTensor h_dn = rand_f32({mc.num_experts, mc.hidden_size, mc.intermediate_size}, rng, 0.1f);
        HostTensor h_sgu = rand_f32({2 * mc.shared_intermediate_size, mc.hidden_size}, rng, 0.1f);
        HostTensor h_sdn = rand_f32({mc.hidden_size, mc.shared_intermediate_size}, rng, 0.1f);

        CudaSparseMoELayer l_gpu;
        l_gpu.cfg = mc;
        l_gpu.cfg.gpu_resident_experts = 6;
        l_gpu.norm_w = ctx.upload(h_nw, true);
        l_gpu.router_w = ctx.upload(h_rw, true);
        l_gpu.gpu_gate_up_w = ctx.upload(h_gu, true);
        l_gpu.gpu_down_w = ctx.upload(h_dn, true);
        l_gpu.shared_gate_up_w = ctx.upload(h_sgu, true);
        l_gpu.shared_down_w = ctx.upload(h_sdn, true);

        CudaSparseMoELayer l_off;
        l_off.cfg = mc;
        l_off.cfg.gpu_resident_experts = 3;  // experts 3..5 offloaded to CPU!
        l_off.norm_w = l_gpu.norm_w;
        l_off.router_w = l_gpu.router_w;
        l_off.gpu_gate_up_w = ctx.upload(HostTensor::create({3, 2 * mc.intermediate_size, mc.hidden_size}, DType::F32), true);
        l_off.gpu_down_w = ctx.upload(HostTensor::create({3, mc.hidden_size, mc.intermediate_size}, DType::F32), true);
        // Copy first 3 experts to GPU, last 3 to CPU
        FUSIONRT_CUDA_CHECK(cudaMemcpy(l_off.gpu_gate_up_w.ptr(), l_gpu.gpu_gate_up_w.ptr(), l_off.gpu_gate_up_w.nbytes(), cudaMemcpyDeviceToDevice));
        FUSIONRT_CUDA_CHECK(cudaMemcpy(l_off.gpu_down_w.ptr(), l_gpu.gpu_down_w.ptr(), l_off.gpu_down_w.nbytes(), cudaMemcpyDeviceToDevice));
        l_off.cpu_gate_up_w = HostTensor::create({3, 2 * mc.intermediate_size, mc.hidden_size}, DType::F32);
        l_off.cpu_down_w = HostTensor::create({3, mc.hidden_size, mc.intermediate_size}, DType::F32);
        std::memcpy(l_off.cpu_gate_up_w.storage.data(), h_gu.storage.data() + l_off.gpu_gate_up_w.nbytes(), l_off.cpu_gate_up_w.storage.size());
        std::memcpy(l_off.cpu_down_w.storage.data(), h_dn.storage.data() + l_off.gpu_down_w.nbytes(), l_off.cpu_down_w.storage.size());
        l_off.shared_gate_up_w = l_gpu.shared_gate_up_w;
        l_off.shared_down_w = l_gpu.shared_down_w;

        Tensor x1 = ctx.upload(h_x, false);
        Tensor x2 = ctx.upload(h_x, false);
        l_gpu.forward_cuda(ctx, x1);
        l_off.forward_cuda(ctx, x2);
        HostTensor h1 = ctx.download(x1);
        HostTensor h2 = ctx.download(x2);
        std::vector<double> ref(h1.data<float>(), h1.data<float>() + mc.hidden_size);
        expect_close(std::vector<float>(h2.data<float>(), h2.data<float>() + mc.hidden_size), ref, 1e-4, 1e-4, "MoE GPU-resident vs CPU-offload parity");
    }
}
#endif

#if defined(FUSIONRT_ENABLE_CUDA) && defined(FUSIONRT_ENABLE_VULKAN)
void test_phase2_loaders_prefill_and_vulkan_e2e(CudaContext& cuda_ctx) {
    std::cout << "[8/8] Testing full GGUF/Safetensors loading, batched prefill, and Vulkan E2E parity...\n";
    std::mt19937_64 rng(909);
    auto tmp_dir = std::filesystem::temp_directory_path() / "fusionrt_phase2_test";
    std::filesystem::create_directories(tmp_dir);

    // 8a. Full 2-layer Qwen3/Llama-style GGUF (Q4_0 weights!) + explicit missing-tensor rejection
    {
        int H = 64, nq = 4, nkv = 2, hd = 16, inter = 128, vocab = 64, nlayers = 2;
        int q_dim = nq * hd, kv_dim = nkv * hd;
        auto q4_mat = [&](int rows, int cols) {
            HostTensor f = rand_f32({rows, cols}, rng, 0.1f);
            HostTensor q = HostTensor::create({rows, cols}, DType::Q4_0);
            int nb = cols / 32;
            for (int r = 0; r < rows; ++r) {
                quantize_row_q4_0(f.data<float>() + r * cols,
                                  reinterpret_cast<BlockQ4_0*>(q.storage.data()) + r * nb, cols);
            }
            return q;
        };

        std::map<std::string, GgufMetaValue> meta;
        meta["general.architecture"] = std::string("llama");
        meta["llama.block_count"] = static_cast<int32_t>(nlayers);
        meta["llama.embedding_length"] = static_cast<int32_t>(H);
        meta["llama.attention.head_count"] = static_cast<int32_t>(nq);
        meta["llama.attention.head_count_kv"] = static_cast<int32_t>(nkv);
        meta["llama.attention.key_length"] = static_cast<int32_t>(hd);
        meta["llama.feed_forward_length"] = static_cast<int32_t>(inter);
        meta["llama.context_length"] = static_cast<int32_t>(256);
        meta["llama.rope.freq_base"] = 10000.0f;
        meta["llama.rope.dimension_count"] = static_cast<int32_t>(hd);
        meta["llama.attention.layer_norm_rms_epsilon"] = 1e-6f;
        std::vector<std::string> toks(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) toks[static_cast<size_t>(i)] = "t" + std::to_string(i);
        meta["tokenizer.ggml.tokens"] = toks;

        std::vector<std::pair<std::string, HostTensor>> tensors;
        tensors.emplace_back("token_embd.weight", q4_mat(vocab, H));
        tensors.emplace_back("output_norm.weight", make_ones_f32({H}));
        tensors.emplace_back("output.weight", q4_mat(vocab, H));
        for (int l = 0; l < nlayers; ++l) {
            std::string p = "blk." + std::to_string(l) + ".";
            tensors.emplace_back(p + "attn_norm.weight", make_ones_f32({H}));
            tensors.emplace_back(p + "attn_q.weight", q4_mat(q_dim, H));
            tensors.emplace_back(p + "attn_k.weight", q4_mat(kv_dim, H));
            tensors.emplace_back(p + "attn_v.weight", q4_mat(kv_dim, H));
            tensors.emplace_back(p + "attn_output.weight", q4_mat(H, q_dim));
            tensors.emplace_back(p + "ffn_norm.weight", make_ones_f32({H}));
            tensors.emplace_back(p + "ffn_gate.weight", q4_mat(inter, H));
            tensors.emplace_back(p + "ffn_up.weight", q4_mat(inter, H));
            tensors.emplace_back(p + "ffn_down.weight", q4_mat(H, inter));
        }

        std::string gguf_p = (tmp_dir / "full_q4.gguf").string();
        std::string err;
        if (!GgufFile::write_v3(gguf_p, meta, tensors, &err)) throw std::runtime_error(err);

        cuda_ctx.weight_arena().reset();
        cuda_ctx.scratch_arena().reset();
        FusionModel m_cuda(HybridModelConfig{});
        if (!m_cuda.load_gguf(cuda_ctx, gguf_p, &err)) throw std::runtime_error("Full GGUF CUDA load failed: " + err);
        if (m_cuda.dense_layers().size() != 2 || m_cuda.dense_layers()[0].o_proj.dtype != DType::Q4_0) {
            throw std::runtime_error("GGUF loader failed to preserve Q4_0 weights on GPU!");
        }
        auto cuda_ids = m_cuda.generate_cuda(cuda_ctx, {1, 5, 9}, 4, true, false);

        VkContext vk(0, 64, 64);
        FusionModel m_vk(HybridModelConfig{});
        if (!m_vk.load_gguf_vulkan(vk, gguf_p, &err)) throw std::runtime_error("Full GGUF Vulkan load failed: " + err);
        auto vk_ids = m_vk.generate_vulkan(vk, {1, 5, 9}, 4);
        if (cuda_ids != vk_ids) throw std::runtime_error("Full Q4_0 GGUF CUDA vs Vulkan token mismatch!");
        std::cout << "  [PASS/CHECK] Full 2-layer Q4_0 GGUF CUDA vs Vulkan exact token parity\n";

        // Test explicit missing-tensor rejection (drop last tensor)
        tensors.pop_back();
        std::string bad_gguf = (tmp_dir / "incomplete.gguf").string();
        GgufFile::write_v3(bad_gguf, meta, tensors, &err);
        FusionModel m_bad(HybridModelConfig{});
        if (m_bad.load_gguf(cuda_ctx, bad_gguf, &err)) {
            throw std::runtime_error("Expected load_gguf to fail on incomplete GGUF!");
        }
        std::cout << "  [PASS/CHECK] Incomplete GGUF explicitly rejected: " << err << "\n";
    }

    // 8b. Full Safetensors (F16 + config.json) load test
    {
        int H = 64, nq = 4, nkv = 2, hd = 16, inter = 128, vocab = 64;
        int q_dim = nq * hd, kv_dim = nkv * hd;
        auto f16_mat = [&](int rows, int cols) {
            HostTensor f = rand_f32({rows, cols}, rng, 0.1f);
            HostTensor h = HostTensor::create({rows, cols}, DType::F16);
            convert_f32_to_f16(f.data<float>(), reinterpret_cast<uint16_t*>(h.storage.data()), rows * cols);
            return h;
        };
        auto ones_f16 = [&](int n) {
            HostTensor f = make_ones_f32({n});
            HostTensor h = HostTensor::create({n}, DType::F16);
            convert_f32_to_f16(f.data<float>(), reinterpret_cast<uint16_t*>(h.storage.data()), n);
            return h;
        };
        std::string st_p = (tmp_dir / "model.safetensors").string();
        std::string cfg_p = (tmp_dir / "config.json").string();
        {
            std::ofstream ofs(cfg_p);
            ofs << "{\"model_type\":\"llama\",\"hidden_size\":64,\"num_hidden_layers\":1,"
                << "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":16,"
                << "\"intermediate_size\":128,\"vocab_size\":64,\"tie_word_embeddings\":false}";
        }
        std::vector<std::pair<std::string, HostTensor>> tensors;
        tensors.emplace_back("model.embed_tokens.weight", f16_mat(vocab, H));
        tensors.emplace_back("model.norm.weight", ones_f16(H));
        tensors.emplace_back("lm_head.weight", f16_mat(vocab, H));
        tensors.emplace_back("model.layers.0.input_layernorm.weight", ones_f16(H));
        tensors.emplace_back("model.layers.0.self_attn.q_proj.weight", f16_mat(q_dim, H));
        tensors.emplace_back("model.layers.0.self_attn.k_proj.weight", f16_mat(kv_dim, H));
        tensors.emplace_back("model.layers.0.self_attn.v_proj.weight", f16_mat(kv_dim, H));
        tensors.emplace_back("model.layers.0.self_attn.o_proj.weight", f16_mat(H, q_dim));
        tensors.emplace_back("model.layers.0.post_attention_layernorm.weight", ones_f16(H));
        tensors.emplace_back("model.layers.0.mlp.gate_proj.weight", f16_mat(inter, H));
        tensors.emplace_back("model.layers.0.mlp.up_proj.weight", f16_mat(inter, H));
        tensors.emplace_back("model.layers.0.mlp.down_proj.weight", f16_mat(H, inter));

        std::string err;
        if (!SafetensorsFile::write(st_p, {{"format", "pt"}}, tensors, &err)) throw std::runtime_error(err);
        cuda_ctx.weight_arena().reset();
        cuda_ctx.scratch_arena().reset();
        FusionModel m_st(HybridModelConfig{});
        if (!m_st.load_safetensors(cuda_ctx, st_p, &err)) throw std::runtime_error("Safetensors load failed: " + err);
        if (m_st.dense_layers()[0].o_proj.dtype != DType::F16) {
            throw std::runtime_error("Safetensors loader failed to preserve F16 weights on GPU!");
        }
        auto st_ids = m_st.generate_cuda(cuda_ctx, {1, 3, 7}, 3, true, false);
        if (st_ids.size() != 6) throw std::runtime_error("Safetensors generation length mismatch");
        std::cout << "  [PASS/CHECK] Full F16 Safetensors + config.json load & generation OK\n";
    }

    // 8c. Batched Prefill (M=5) vs Token-by-Token + Vulkan E2E parity across architectures
    for (const std::string& arch : {"dense", "hybrid", "moe"}) {
        cuda_ctx.weight_arena().reset();
        cuda_ctx.scratch_arena().reset();
        HybridModelConfig cfg;
        cfg.arch = arch;
        cfg.vocab_size = 128;
        cfg.num_layers = 4;
        cfg.dense.hidden_size = 128;
        cfg.dense.num_attention_heads = 4;
        cfg.dense.num_key_value_heads = 2;
        cfg.dense.head_dim = 32;
        cfg.dense.intermediate_size = 256;
        cfg.dense.max_seq_len = 256;
        cfg.gdn.hidden_size = 128;
        cfg.gdn.num_k_heads = 2;
        cfg.gdn.num_v_heads = 4;
        cfg.gdn.head_k_dim = 32;
        cfg.gdn.head_v_dim = 32;
        cfg.gdn.intermediate_size = 256;
        cfg.moe.hidden_size = 128;
        cfg.moe.intermediate_size = 128;
        cfg.moe.num_experts = 4;
        cfg.moe.top_k = 2;
        cfg.moe.gpu_resident_experts = 4;
        cfg.moe.shared_intermediate_size = 128;

        FusionModel m_cuda(cfg);
        m_cuda.init_synthetic(cuda_ctx, 999);
        std::vector<int32_t> prompt = {1, 11, 22, 33, 44};

        // Token-by-token reference
        m_cuda.reset_kv_and_state(cuda_ctx);
        std::vector<float> ref_logits;
        for (int p = 0; p < static_cast<int>(prompt.size()); ++p) {
            ref_logits = m_cuda.forward_token_cuda(cuda_ctx, prompt[static_cast<size_t>(p)], p, true, false);
        }
        // Batched prefill
        m_cuda.reset_kv_and_state(cuda_ctx);
        auto pf_logits = m_cuda.forward_prefill_cuda(cuda_ctx, prompt, 0, true);
        std::vector<double> ref_d(ref_logits.begin(), ref_logits.end());
        expect_close(pf_logits, ref_d, 1e-4, 1e-4, "Batched prefill M=5 vs sequential (" + arch + ")");

        auto cuda_gen = m_cuda.generate_cuda(cuda_ctx, prompt, 3, true, false);
        auto vk_devs = VkContext::enumerate_devices();
        for (const auto& vd : vk_devs) {
            VkContext vk(vd.device_index, 128, 128);
            FusionModel m_vk(cfg);
            m_vk.init_synthetic_vulkan(vk, 999);
            auto vk_gen = m_vk.generate_vulkan(vk, prompt, 3);
            if (cuda_gen != vk_gen) {
                throw std::runtime_error("Vulkan E2E generation diverged from CUDA on " + vd.name + " (" + arch + ")");
            }
            std::cout << "  [PASS/CHECK] Vulkan[" << vd.device_index << "] E2E " << arch << " token parity vs CUDA\n";
        }
    }

    std::filesystem::remove_all(tmp_dir);
}
#endif

void test_review_gguf_validation() {
    struct Fixture {
        std::string path = "test-review-gguf.gguf";
        ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    } fixture;
    std::string error;
    for (const auto dtype : {DType::F16, DType::BF16, DType::F32, DType::I32, DType::Q6_K}) {
        const int cols = dtype == DType::Q6_K ? 256 : 4;
        const HostTensor weight({cols}, dtype);
        GgufFile gguf;
        if (!GgufFile::write_v3(fixture.path, {}, {{"v", weight}, {"next", weight}}, &error) ||
            !gguf.load(fixture.path, &error)) throw std::runtime_error(error);
        gguf = {};
        constexpr int first_tensor_offset = 24 + 8 + 1 + 4 + 8 + 4;
        {
            std::fstream file(fixture.path, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(first_tensor_offset);
            file.put(dtype == DType::F32 || dtype == DType::I32 ? '\x02' : '\x01');
            if (!file) throw std::runtime_error("Could not construct misaligned GGUF fixture");
        }
        if (gguf.load(fixture.path, &error) || !error.starts_with("Misaligned GGUF tensor offset"))
            throw std::runtime_error("GGUF must reject misaligned data");
    }
    {
        const std::map<std::string, GgufMetaValue> meta{
            {"tokenizer.ggml.model", std::string("gemma4")},
            {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "\xe2\x96\x81", "b"}}
        };
        GgufFile gguf;
        BpeTokenizer tokenizer;
        if (!GgufFile::write_v3(fixture.path, meta, {}, &error) || !gguf.load(fixture.path, &error) ||
            !tokenizer.load_from_gguf(gguf) || tokenizer.encode("a b", false) != std::vector<int32_t>{0,1,2})
            throw std::runtime_error("Gemma 4 must normalize spaces with an empty merge table");
    }
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("moe")}, {"moe.block_count", uint32_t{1}},
        {"moe.embedding_length", uint32_t{8}}, {"moe.attention.head_count", uint32_t{1}},
        {"moe.expert_count", uint32_t{2}}, {"moe.expert_feed_forward_length", uint32_t{4}}
    };
    std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", HostTensor({8,8}, DType::F32)}, {"output_norm.weight", HostTensor({8}, DType::F32)},
        {"blk.0.ffn_norm.weight", HostTensor({8}, DType::F32)}, {"blk.0.ffn_gate_inp.weight", HostTensor({2,8}, DType::F32)}
    };
    for (int e = 0; e < 2; ++e) {
        const auto suffix = std::to_string(e) + ".weight";
        weights.emplace_back("blk.0.ffn_gate." + suffix, HostTensor({4,8}, DType::F32));
        weights.emplace_back("blk.0.ffn_up." + suffix, HostTensor({4,8}, DType::F32));
        weights.emplace_back("blk.0.ffn_down." + suffix, HostTensor({8,4}, DType::F32));
    }
    if (!GgufFile::write_v3(fixture.path, meta, weights, &error)) throw std::runtime_error(error);
    {
        model_io::LoadedModelRaw raw;
        if (!model_io::parse_gguf_model_raw(fixture.path, 16, raw, &error)) throw std::runtime_error(error);
    }
    weights[7].second = HostTensor({3,8}, DType::F32);
    weights[8].second = HostTensor({5,8}, DType::F32);
    model_io::LoadedModelRaw raw;
    if (!GgufFile::write_v3(fixture.path, meta, weights, &error) ||
        model_io::parse_gguf_model_raw(fixture.path, 16, raw, &error) ||
        !error.starts_with("Invalid GGUF individual expert FFN shapes"))
        throw std::runtime_error("GGUF loader accepted mismatched per-expert gate/up widths");
    std::cout << "  [PASS] GGUF scalar/block alignment, individual expert shapes and Gemma 4 empty merges\n";
}

}  // namespace fusionrt::test
