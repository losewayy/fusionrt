#pragma once

#include "test_vulkan_v1.h"
#include "fusionrt/vulkan_dispatch.h"

namespace fusionrt::test {
namespace {

using V2Vector = std::vector<float>;

struct VulkanV2Model {
    VulkanSessionManager sessions;
    VulkanModelWeights weights;
    VulkanExecutor executor;

    VulkanV2Model(VkContext& vk, const model_io::LoadedModelRaw& raw, DType kv = DType::F32)
        : weights(raw.cfg), executor(weights, sessions) {
        weights.upload(vk, raw, {
            [&](const DenseGQAConfig& d, DType type) { sessions.allocate_dense(vk, d, type); },
            [&](const GatedDeltaNetConfig& g, int conv) { sessions.allocate_gdn(vk, g, conv); },
            [&](int source) { sessions.allocate_dense(vk, {}, DType::F32, source); }
        }, kv);
        sessions.reset_kv_and_state_vulkan(vk);
    }
};

V2Vector v2_row(const HostTensor& t, int row) {
    const auto width = t.desc.shape[1];
    return {t.data<float>() + row * width, t.data<float>() + (row + 1) * width};
}

V2Vector v2_linear(std::span<const float> input, const float* w, int n) {
    V2Vector output(n);
    for (int i = 0; i < n; ++i)
        output[i] = static_cast<float>(codec_reference_dot(w + i * input.size(), input.data(), static_cast<int>(input.size())));
    return output;
}

V2Vector v2_linear(std::span<const float> input, const HostTensor& w) {
    return v2_linear(input, w.data<float>(), static_cast<int>(w.desc.shape[0]));
}

void v2_norm(std::span<float> x, const float* w, float eps) {
    const auto mean = codec_reference_dot(x.data(), x.data(), static_cast<int>(x.size())) / x.size();
    const auto inv = static_cast<float>(1.0 / std::sqrt(mean + eps));
    for (size_t i = 0; i < x.size(); ++i) x[i] *= inv * (w ? w[i] : 1.f);
}

void v2_add(V2Vector& x, const V2Vector& y, float scale = 1.f) {
    for (size_t i = 0; i < x.size(); ++i) x[i] = (x[i] + y[i]) * scale;
}

float v2_gate(float x, bool geglu) {
    return geglu ? 0.5f * x * (1.f + std::tanh(0.7978845608f * x * (1.f + 0.044715f * x * x)))
                 : x / (1.f + std::exp(-x));
}

V2Vector v2_glu(const V2Vector& gu, bool geglu) {
    const auto width = gu.size() / 2;
    V2Vector result(width);
    for (size_t i = 0; i < width; ++i) result[i] = v2_gate(gu[i], geglu) * gu[width + i];
    return result;
}

void v2_rope(std::span<float> x, int width, int pos, float theta, bool neox,
             const float* factors = nullptr, bool frequency = false, float amplitude = 1.f) {
    for (int i = 0; i < width / 2; ++i) {
        const auto base = std::pow(theta, -2.f * i / width);
        const auto f = !factors ? base : frequency ? factors[i] : base / factors[i];
        const int a = neox ? i : 2 * i, b = neox ? i + width / 2 : a + 1;
        const float c = std::cos(pos * f) * amplitude, s = std::sin(pos * f) * amplitude;
        const auto u = x[a], v = x[b];
        x[a] = u * c - v * s;
        x[b] = u * s + v * c;
    }
}

V2Vector v2_attend(std::span<const float> query, std::span<const V2Vector> keys,
                    std::span<const V2Vector> values, float scale) {
    V2Vector scores;
    for (const auto& key : keys)
        scores.push_back(static_cast<float>(codec_reference_dot(query.data(), key.data(), static_cast<int>(query.size()))) * scale);
    const float peak = *std::max_element(scores.begin(), scores.end());
    V2Vector result(values.front().size());
    float mass = 0.f;
    for (size_t i = 0; i < scores.size(); ++i) {
        const float p = std::exp(scores[i] - peak);
        mass += p;
        for (size_t j = 0; j < result.size(); ++j) result[j] += p * values[i][j];
    }
    for (auto& x : result) x /= mass;
    return result;
}

model_io::LoadedModelRaw v2_raw(int hidden, int vocab, int layers, std::mt19937_64& rng) {
    model_io::LoadedModelRaw raw;
    raw.cfg.dense.hidden_size = hidden;
    raw.cfg.dense.max_seq_len = 24;
    raw.cfg.vocab_size = vocab;
    raw.cfg.num_layers = layers;
    raw.tok_embeddings = model_io::make_rand_f32({vocab, hidden}, rng, .25f);
    raw.final_norm_w = model_io::make_ones_f32({hidden});
    raw.lm_head_w = model_io::make_rand_f32({vocab, hidden}, rng, .12f);
    raw.layers.resize(layers);
    return raw;
}

void test_vk_v2_gdn(VkContext& vk) {
    HybridModelConfig cfg;
    cfg.arch = "hybrid";
    cfg.vocab_size = 33;
    cfg.num_layers = 1;
    cfg.dense.max_seq_len = 24;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 32;
    cfg.gdn.num_k_heads = 2;
    cfg.gdn.num_v_heads = 4;
    cfg.gdn.head_k_dim = cfg.gdn.head_v_dim = 8;
    cfg.gdn.intermediate_size = 64;
    cfg.gdn.tiled_value_heads = true;
    VulkanSessionManager sessions;
    VulkanModelWeights weights(cfg);
    weights.init_synthetic_vulkan(vk, 903, {
        {},
        [&](const GatedDeltaNetConfig& g, int conv) { sessions.allocate_gdn(vk, g, conv); },
        {}
    });
    VulkanExecutor executor(weights, sessions);
    const std::array<int32_t, 8> prompt{1, 4, 7, 2, 31, 5, 12, 6};
    sessions.reset_kv_and_state_vulkan(vk);
    const auto batch = executor.forward_batch_vulkan(vk, prompt, 0);
    const auto batch_tail = executor.forward_token_vulkan(vk, 13, 8);
    const auto state = vk.download(sessions.gdn(0).ssm);

    sessions.reset_kv_and_state_vulkan(vk);
    V2Vector sequential;
    for (int i = 0; i < static_cast<int>(prompt.size()); ++i)
        sequential = executor.forward_token_vulkan(vk, prompt[i], i);
    const auto sequential_tail = executor.forward_token_vulkan(vk, 13, 8);
    const auto seq_state = vk.download(sessions.gdn(0).ssm);
    require_vk_close(DType::F32, "GDN M=8 logits", batch, sequential, 2e-5);
    require_vk_close(DType::F32, "GDN continuation", batch_tail, sequential_tail, 2e-5);
    require_vk_close(DType::F32, "GDN recurrent state",
                     {state.data<float>(), static_cast<size_t>(state.desc.numel())},
                     {seq_state.data<float>(), static_cast<size_t>(seq_state.desc.numel())}, 2e-5);
}

void test_vk_v2_fwht(VkContext& vk) {
    constexpr int H = 128, V = 16, block = 64;
    std::mt19937_64 rng(37);
    auto raw = v2_raw(H, V, 0, rng);
    raw.hadamard_folded = true;
    raw.tok_embeddings = native_weight_fixture(DType::PTQ1_0, V, H);
    const auto decoded = model_io::to_f32_host(raw.tok_embeddings);
    raw.lm_head_w = {};
    auto signs = std::make_shared<std::vector<int8_t>>(H);
    for (int i = 0; i < H; ++i) {
        (*signs)[i] = i % 3 ? 1 : -1;
        raw.final_norm_w.data<float>()[i] = i % 2 ? 0.8f : 1.3f;
    }
    raw.tok_embeddings.hadamard_signs = signs;
    raw.tok_embeddings.desc.hadamard_block = block;
    VulkanV2Model model(vk, raw);
    V2Vector hidden(H), rotated(H);
    blockwise_hadamard_f32(decoded.data<float>() + 3 * H, signs->data(), hidden.data(), 1, H, block, true);
    v2_norm(hidden, raw.final_norm_w.data<float>(), raw.cfg.dense.rms_norm_eps);
    blockwise_hadamard_f32(hidden.data(), signs->data(), rotated.data(), 1, H, block);
    require_vk_close(DType::PTQ1_0, "signed folded embedding and tied head",
                     model.executor.forward_token_vulkan(vk, 3, 0), v2_linear(rotated, decoded), 2e-5);

    // GDN O projections may need repeat-major heads grouped before rotation.
    const auto mark = vk.scratch_arena().mark();
    constexpr int dim = 16, keys = 2, repeats = H / (dim * keys);
    auto input = model_io::make_rand_f32({1, H}, rng);
    auto grouped = input;
    for (int k = 0; k < keys; ++k)
        for (int r = 0; r < repeats; ++r)
            std::copy_n(input.data<float>() + (r * keys + k) * dim, dim,
                        grouped.data<float>() + (k * repeats + r) * dim);
    blockwise_hadamard_f32(grouped.data<float>(), signs->data(), rotated.data(), 1, H, block);
    auto weight = model.weights.lm_head();
    weight.hadamard_head_dim = dim;
    weight.hadamard_key_heads = keys;
    const auto x = vk.upload(input, false);
    auto y = vk.scratch_arena().alloc_tensor({1, V}, DType::F32);
    VulkanDispatch::Batch batch{vk};
    batch.rotated = vk.scratch_arena().alloc_tensor({1, H}, DType::F32);
    batch.dummy = model.weights.dummy();
    batch.hadamard_signs = &model.weights.hadamard_signs();
    VulkanDispatch::append_linear(batch, x, weight, y, 1, V, H);
    vk.dispatch_batched(batch.ops, false);
    const auto actual = vk.download(y);
    require_vk_close(DType::PTQ1_0, "repeat-major FWHT projection", {actual.data<float>(), V},
                     v2_linear(rotated, decoded), 2e-5);
    vk.scratch_arena().reset(mark);
}

V2Vector v2_moe(const model_io::LoadedModelRaw& raw, const model_io::LoadedLayerRaw& l, V2Vector x) {
    const auto& cfg = raw.cfg.moe;
    const int h = static_cast<int>(x.size()), e = cfg.num_experts, n = cfg.intermediate_size;
    const auto& norm = l.mlp_norm.bytes().empty() ? l.attn_norm : l.mlp_norm;
    const bool geglu = !l.router_scale.bytes().empty();
    auto normalized = x, router = x;
    v2_norm(normalized, geglu ? l.routed_norm.data<float>() : norm.data<float>(), cfg.rms_norm_eps);
    if (geglu) v2_norm(router, l.router_scale.data<float>(), cfg.rms_norm_eps);
    auto p = v2_linear(geglu ? router : normalized, l.router_w);
    const float peak = *std::max_element(p.begin(), p.end());
    float mass = 0.f;
    for (auto& v : p) {
        v = cfg.sigmoid_router ? 1.f / (1.f + std::exp(-v)) : std::exp(v - peak);
        mass += v;
    }
    if (!cfg.sigmoid_router) for (auto& v : p) v /= mass;
    const auto ranking = [&](int i) { return p[i] + (l.router_bias.bytes().empty() ? 0.f : l.router_bias.data<float>()[i]); };
    const auto better = [&](int a, int b) {
        return ranking(a) == ranking(b) ? a < b : ranking(a) > ranking(b);
    };
    std::vector<int> candidates(e);
    std::iota(candidates.begin(), candidates.end(), 0);
    if (cfg.expert_groups > 1) {
        const int width = e / cfg.expert_groups;
        std::vector<int> groups(cfg.expert_groups);
        std::iota(groups.begin(), groups.end(), 0);
        V2Vector scores(cfg.expert_groups);
        for (int g = 0; g < cfg.expert_groups; ++g) {
            auto first = candidates.begin() + g * width;
            std::sort(first, first + width, better);
            scores[g] = ranking(first[0]) + ranking(first[1]);
        }
        std::stable_sort(groups.begin(), groups.end(), [&](int a, int b) { return scores[a] > scores[b]; });
        candidates.clear();
        for (int g = 0; g < cfg.expert_groups_used; ++g)
            for (int i = 0; i < width; ++i) candidates.push_back(groups[g] * width + i);
    }
    std::sort(candidates.begin(), candidates.end(), better);
    mass = 0.f;
    for (int k = 0; k < cfg.top_k; ++k) mass += p[candidates[k]];
    const auto gu = model_io::to_f32_host(l.experts_gate_up), down = model_io::to_f32_host(l.experts_down);
    const bool split = !l.experts_up.bytes().empty();
    const auto up = split ? model_io::to_f32_host(l.experts_up) : HostTensor{};
    if (geglu) std::fill(x.begin(), x.end(), 0.f);
    for (int k = 0; k < cfg.top_k; ++k) {
        const int i = candidates[k];
        auto fused = v2_linear(normalized, gu.data<float>() + i * n * h * (split ? 1 : 2), split ? n : 2 * n);
        if (split) {
            const auto u = v2_linear(normalized, up.data<float>() + i * n * h, n);
            fused.insert(fused.end(), u.begin(), u.end());
        }
        if (!l.expert_gate_scale.bytes().empty())
            for (int j = 0; j < n; ++j) fused[j] *= l.expert_gate_scale.data<float>()[i];
        if (!l.expert_up_scale.bytes().empty())
            for (int j = n; j < 2 * n; ++j) fused[j] *= l.expert_up_scale.data<float>()[i];
        const auto value = v2_linear(v2_glu(fused, geglu), down.data<float>() + i * h * n, h);
        float route = p[i] * cfg.expert_weights_scale / (cfg.norm_topk_prob ? std::max(mass, 0x1p-14f) : 1.f);
        if (!l.expert_down_scale.bytes().empty()) route *= l.expert_down_scale.data<float>()[i];
        for (int j = 0; j < h; ++j) x[j] += route * value[j];
    }
    if (l.has_shared) {
        const auto gu_shared = v2_linear(normalized, l.shared_gate_up);
        const auto value = v2_linear(v2_glu(gu_shared, false), l.shared_down);
        const auto factor = l.shared_gate.bytes().empty() ? 1.f :
            1.f / (1.f + std::exp(-v2_linear(normalized, l.shared_gate)[0]));
        for (int j = 0; j < h; ++j) x[j] += factor * value[j];
    }
    return x;
}

void test_vk_v2_gemma(VkContext& vk) {
    constexpr int H = 64, I = 32, P = 8, V = 24, L = 3;
    std::mt19937_64 rng(80);
    const auto rand = [&](std::initializer_list<int64_t> dims) { return model_io::make_rand_f32(dims, rng, .12f); };
    auto raw = v2_raw(H, V, L, rng);
    raw.cfg.model_architecture = "gemma4";
    raw.cfg.embed_scale = std::sqrt(static_cast<float>(H));
    raw.cfg.final_logit_softcapping = 1.25f;
    raw.cfg.moe.intermediate_size = I;
    raw.cfg.moe.num_experts = 3;
    raw.gemma_inputs = {rand({V, L * P}), rand({L * P, H}), model_io::make_ones_f32({P})};
    raw.rope_factors = model_io::make_ones_f32({64});
    raw.rope_factors.data<float>()[0] = 3.f;
    raw.rope_factors.data<float>()[7] = std::numeric_limits<float>::infinity();
    for (int i = 0; i < L; ++i) {
        auto& l = raw.layers[i];
        const int d = i == 1 ? 512 : 256;
        l.query_heads = l.kv_heads = 1;
        l.head_dim = d;
        l.rotary_dims = 128;
        l.rope_theta = i == 1 ? 1e6f : 1e4f;
        l.sliding_window = i == 1 ? 0 : 2;
        l.kv_source_layer = i == 2 ? 0 : -1;
        l.output_scale = 0.8f + i * 0.1f;
        l.q_proj = rand({d, H});
        l.q_norm = model_io::make_ones_f32({d});
        if (i != 2) {
            l.k_proj = rand({d, H});
            l.k_norm = model_io::make_ones_f32({d});
            if (i == 0) l.v_proj = rand({d, H});
        }
        l.o_proj = rand({H, d});
        l.attn_norm = l.attn_post_norm = l.mlp_norm = l.ffn_post_norm = model_io::make_ones_f32({H});
        l.gate_up = rand({2 * I, H});
        l.down = rand({H, I});
        l.ple_gate = rand({P, H});
        l.ple_projection = rand({H, P});
        l.ple_post_norm = model_io::make_ones_f32({H});
        if (i == 1) {
            l.router_w = rand({3, H});
            l.router_scale = rand({H});
            l.routed_norm = l.dense_post_norm = l.routed_post_norm = model_io::make_ones_f32({H});
            l.experts_gate_up = rand({3, 2 * I, H});
            l.experts_down = rand({3, H, I});
            l.expert_down_scale = rand({3});
            l.expert_gate_scale = rand({3});
            l.expert_up_scale = rand({3});
        }
    }
    VulkanV2Model model(vk, raw);
    if (model.sessions.dense(0).key.vk_offset != model.sessions.dense(2).key.vk_offset)
        throw std::runtime_error("Gemma shared KV allocated a second key cache");

    const std::vector<int32_t> prompt{2, 9, 8, 12, 19};
    const InputEmbedding span{1, {}, true, rand({3, H})};
    std::vector<V2Vector> hidden, ple;
    const float eps = raw.cfg.dense.rms_norm_eps;
    for (int row = 0; row < static_cast<int>(prompt.size()); ++row) {
        auto x = v2_row(raw.tok_embeddings, prompt[row]);
        for (auto& v : x) v *= raw.cfg.embed_scale;
        if (row >= 1 && row < 4) x = v2_row(span.host_rows, row - 1);
        auto projection = v2_linear(x, raw.gemma_inputs.projection);
        auto p = v2_row(raw.gemma_inputs.token_embeddings, prompt[row]);
        for (int i = 0; i < L; ++i)
            v2_norm(std::span{projection}.subspan(i * P, P), raw.gemma_inputs.norm.data<float>(), eps * H);
        for (auto& v : p) v *= std::sqrt(static_cast<float>(P));
        v2_add(p, projection, std::sqrt(0.5f));
        hidden.push_back(std::move(x));
        ple.push_back(std::move(p));
    }

    std::array<std::vector<V2Vector>, L> keys, values;
    for (int i = 0; i < L; ++i) {
        const auto& l = raw.layers[i];
        const int source = l.kv_source_layer < 0 ? i : l.kv_source_layer;
        const float* factors = l.sliding_window ? nullptr : raw.rope_factors.data<float>();
        std::vector<V2Vector> queries;
        for (int row = 0; row < static_cast<int>(prompt.size()); ++row) {
            auto n = hidden[row];
            v2_norm(n, l.attn_norm.data<float>(), eps);
            auto q = v2_linear(n, l.q_proj);
            v2_norm(q, l.q_norm.data<float>(), eps);
            v2_rope(q, l.rotary_dims, row, l.rope_theta, true, factors);
            queries.push_back(std::move(q));
            if (source != i) continue;
            auto k = v2_linear(n, l.k_proj);
            auto v = l.v_proj.bytes().empty() ? k : v2_linear(n, l.v_proj);
            v2_norm(k, l.k_norm.data<float>(), eps);
            v2_norm(v, nullptr, eps);
            v2_rope(k, l.rotary_dims, row, l.rope_theta, true, factors);
            keys[i].push_back(std::move(k));
            values[i].push_back(std::move(v));
        }
        for (int row = 0; row < static_cast<int>(prompt.size()); ++row) {
            int first = l.sliding_window ? std::max(0, row + 1 - l.sliding_window) : 0;
            int end = row + 1;
            if (row >= 1 && row < 4) {
                first = std::min(first, 1);
                end = 4;
            }
            const auto a = v2_attend(queries[row], std::span{keys[source]}.subspan(first, end - first),
                                     std::span{values[source]}.subspan(first, end - first), 1.f);
            auto& x = hidden[row];
            auto update = v2_linear(a, l.o_proj);
            v2_norm(update, l.attn_post_norm.data<float>(), eps);
            v2_add(x, update);
            auto n = x;
            v2_norm(n, l.mlp_norm.data<float>(), eps);
            update = v2_linear(v2_glu(v2_linear(n, l.gate_up), true), l.down);
            if (!l.dense_post_norm.bytes().empty()) {
                auto routed = v2_moe(raw, l, x);
                v2_norm(routed, l.routed_post_norm.data<float>(), eps);
                v2_norm(update, l.dense_post_norm.data<float>(), eps);
                v2_add(update, routed);
            }
            v2_norm(update, l.ffn_post_norm.data<float>(), eps);
            v2_add(x, update);
            auto gate = v2_linear(x, l.ple_gate);
            for (int j = 0; j < P; ++j) gate[j] = v2_gate(gate[j], true) * ple[row][i * P + j];
            update = v2_linear(gate, l.ple_projection);
            v2_norm(update, l.ple_post_norm.data<float>(), eps);
            v2_add(x, update, l.output_scale);
        }
    }
    v2_norm(hidden.back(), raw.final_norm_w.data<float>(), eps);
    auto expected = v2_linear(hidden.back(), raw.lm_head_w);
    const float cap = raw.cfg.final_logit_softcapping;
    for (auto& x : expected) x = cap * std::tanh(x / cap);
    const auto actual = model.executor.forward_batch_vulkan(vk, prompt, 0, true, {&span, 1});
    require_vk_close(DType::F32, "Gemma 4 SWA/shared KV/GeGLU/MoE/softcap/PLE/RoPE", actual, expected, 5e-5);

    // Force a chunk boundary inside the image. Prefill must move it before the span.
    vk.set_prefill_batch(3);
    GenerationOptions options;
    options.inputs_embeds = {&span, 1};
    const auto result = model.executor.generate_vulkan(vk, prompt, 1, nullptr, false, 0, -1, options);
    vk.set_prefill_batch(512);
    if (result.back() != Sampler::argmax(expected.data(), V))
        throw std::runtime_error("Gemma 4 image-aware chunked prefill changed the greedy token");

    // Offset-form RMSNorm remains available for callers supplying zero-centered weights.
    const auto mark = vk.scratch_arena().mark();
    const auto scale = rand({H});
    auto row = rand({1, H});
    auto norm_expected = v2_row(row, 0);
    v2_norm(norm_expected, nullptr, eps);
    for (int j = 0; j < H; ++j) norm_expected[j] *= 1.f + scale.data<float>()[j];
    const auto input = vk.upload(row, false), weight = vk.upload(scale, false);
    VulkanDispatch::Batch batch{vk};
    batch.dummy = model.weights.dummy();
    VulkanDispatch::append_gemma_norm(batch, input, weight, input, 1, H, eps, true);
    vk.dispatch_batched(batch.ops, false);
    row = vk.download(input);
    require_vk_close(DType::F32, "Gemma 4 one-plus RMSNorm", v2_row(row, 0), norm_expected, 2e-6);
    vk.scratch_arena().reset(mark);
}


void test_vk_v2_mla(VkContext& vk, bool split) {
    constexpr int H = 32, heads = 2, rank = 12, nope = 8, rope = 4, dv = 4, V = 23, E = 4, I = 32;
    std::mt19937_64 rng(81);
    const auto rand = [&](std::initializer_list<int64_t> dims) { return model_io::make_rand_f32(dims, rng, .15f); };
    auto raw = v2_raw(H, V, 1, rng);
    auto& d = raw.cfg.dense;
    d.num_attention_heads = heads;
    d.head_dim = nope + rope;
    d.mla = {split ? 0 : 8, rank, rope, dv, 0, .4f, 1.2f};
    auto& m = raw.cfg.moe;
    m.num_experts = E;
    m.intermediate_size = I;
    m.sigmoid_router = true;
    m.expert_groups = 2;
    m.expert_groups_used = 1;
    m.expert_weights_scale = 1.25f;
    raw.rope_factors = rand({rope / 2});
    raw.rope_factors.data<float>()[0] = .33f;
    raw.rope_factors.data<float>()[1] = .09f;
    auto& l = raw.layers.front();
    l.attn_norm = l.mlp_norm = model_io::make_ones_f32({H});
    if (d.mla.query_rank) {
        l.mla_q_a = rand({d.mla.query_rank, H});
        l.mla_q_a_norm = model_io::make_ones_f32({d.mla.query_rank});
    }
    l.q_proj = rand({heads * (nope + rope), d.mla.query_rank ? d.mla.query_rank : H});
    l.mla_kv_a = rand({rank + rope, H});
    l.mla_kv_a_norm = model_io::make_ones_f32({rank});
    auto k_b = rand({heads * nope, rank});
    auto v_b = rand({heads * dv, rank});
    if (split) {
        l.mla_k_b = HostTensor({heads, rank, nope}, DType::F32);
        l.mla_v_b = HostTensor({heads, dv, rank}, DType::F32);
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < rank; ++r)
                for (int n = 0; n < nope; ++n)
                    l.mla_k_b.data<float>()[(h * rank + r) * nope + n] = k_b.data<float>()[(h * nope + n) * rank + r];
        std::copy(v_b.data<float>(), v_b.data<float>() + heads * dv * rank, l.mla_v_b.data<float>());
    } else {
        l.mla_kv_b = HostTensor({heads * (nope + dv), rank}, DType::F32);
        for (int h = 0; h < heads; ++h) {
            auto* dst = l.mla_kv_b.data<float>() + h * (nope + dv) * rank;
            std::copy_n(k_b.data<float>() + h * nope * rank, nope * rank, dst);
            std::copy_n(v_b.data<float>() + h * dv * rank, dv * rank, dst + nope * rank);
        }
    }
    l.o_proj = rand({H, heads * dv});
    l.router_w = rand({E, H});
    l.router_bias = rand({E});
    l.router_bias.data<float>()[2] = .7f;
    l.router_bias.data<float>()[3] = .8f;
    l.experts_gate_up = model_io::quantize_host_matrix(rand({E, 2 * I, H}), DType::Q8_0);
    l.experts_down = model_io::quantize_host_matrix(rand({E, H, I}), DType::Q8_0);

    VulkanV2Model model(vk, raw);
    const auto& cache = model.sessions.dense(0);
    if (cache.key.vk_buffer || cache.value.vk_buffer || cache.latent.nbytes() != d.max_seq_len * (rank + rope) * sizeof(float))
        throw std::runtime_error("MLA must retain a single compressed cache");

    const std::array<int32_t, 4> tokens{1, 3, 4, 12};
    std::array<std::vector<V2Vector>, heads> keys, values;
    V2Vector expected;
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        auto x = v2_row(raw.tok_embeddings, tokens[pos]);
        auto n = x;
        v2_norm(n, l.attn_norm.data<float>(), d.rms_norm_eps);
        auto kv = v2_linear(n, l.mla_kv_a);
        auto latent = std::span{kv}.first(rank);
        auto k_pe = std::span{kv}.last(rope);
        v2_norm(latent, l.mla_kv_a_norm.data<float>(), d.rms_norm_eps);
        v2_rope(k_pe, rope, pos, d.rope_theta, false, raw.rope_factors.data<float>(), true, d.mla.rope_amplitude);
        auto query_in = d.mla.query_rank ? v2_linear(n, l.mla_q_a) : n;
        if (d.mla.query_rank) v2_norm(query_in, l.mla_q_a_norm.data<float>(), d.rms_norm_eps);
        auto q = v2_linear(query_in, l.q_proj);
        V2Vector context;
        // Expand keys and values here, independent of the absorbed Vulkan attention.
        for (int head = 0; head < heads; ++head) {
            auto k = v2_linear(latent, k_b.data<float>() + head * nope * rank, nope);
            k.insert(k.end(), k_pe.begin(), k_pe.end());
            keys[head].push_back(std::move(k));
            values[head].push_back(v2_linear(latent, v_b.data<float>() + head * dv * rank, dv));
            auto query = std::span{q}.subspan(head * (nope + rope), nope + rope);
            v2_rope(query.last(rope), rope, pos, d.rope_theta, false, raw.rope_factors.data<float>(), true, d.mla.rope_amplitude);
            const auto out = v2_attend(query, keys[head], values[head], d.mla.attention_scale);
            context.insert(context.end(), out.begin(), out.end());
        }
        v2_add(x, v2_linear(context, l.o_proj));
        x = v2_moe(raw, l, std::move(x));
        v2_norm(x, raw.final_norm_w.data<float>(), d.rms_norm_eps);
        expected = v2_linear(x, raw.lm_head_w);
    }
    model.executor.forward_batch_vulkan(vk, std::span{tokens}.first(3), 0, false);
    const auto actual = model.executor.forward_token_vulkan(vk, tokens.back(), 3);
    require_vk_close(DType::F32, "MLA absorbed decode and sigmoid/grouped MoE", actual, expected, 3e-5);
}

void test_vk_v2_compact_moe(VkContext& vk) {
    constexpr int H = 256, V = 12, I = 256, E = 3;
    std::mt19937_64 rng(16);
    auto raw = v2_raw(H, V, 1, rng);
    raw.cfg.arch = "moe";
    raw.cfg.moe.hidden_size = H;
    raw.cfg.moe.num_experts = E;
    raw.cfg.moe.top_k = 2;
    raw.cfg.moe.intermediate_size = I;
    auto& l = raw.layers.front();
    l.kind = 2;
    l.attn_norm = model_io::make_ones_f32({H});
    l.router_w = model_io::make_rand_f32({E, H}, rng, .1f);
    l.experts_gate_up = native_weight_fixture(DType::Q4_K, E * I, H);
    l.experts_gate_up.desc = l.experts_gate_up.desc.view({E, I, H});
    l.experts_up = model_io::quantize_host_matrix(model_io::make_rand_f32({E, I, H}, rng, .05f), DType::Q8_0);
    l.experts_down = model_io::quantize_host_matrix(model_io::make_rand_f32({E, H, I}, rng, .05f), DType::Q8_0);
    l.has_shared = true;
    l.shared_gate_up = model_io::make_rand_f32({2 * 64, H}, rng, .05f);
    l.shared_down = model_io::make_rand_f32({H, 64}, rng, .05f);
    l.shared_gate = model_io::make_rand_f32({1, H}, rng, .05f);
    VulkanV2Model model(vk, raw);
    const auto& gpu = model.weights.moe_layers().front();
    const auto align = vk.info().min_storage_offset_align;
    if (gpu.gpu_gate_up_w.dtype != DType::Q4_K || gpu.gpu_down_w.dtype != DType::Q8_0 || gpu.up_dtype != DType::Q8_0 ||
        gpu.gpu_gate_up_w.nbytes() != l.experts_gate_up.bytes().size() ||
        gpu.gpu_down_w.nbytes() != l.experts_down.bytes().size() ||
        gpu.gate_region.nbytes() >= l.experts_gate_up.bytes().size() + l.experts_up.bytes().size() + align)
        throw std::runtime_error("MoE expert weights did not preserve their packed allocation");

    auto expected = v2_moe(raw, l, v2_row(raw.tok_embeddings, 4));
    v2_norm(expected, raw.final_norm_w.data<float>(), raw.cfg.dense.rms_norm_eps);
    const std::array<int32_t, 4> prompt{9, 2, 7, 4};
    require_vk_close(DType::Q4_K, "compact split Q4_K/Q8_0 expert output",
                     model.executor.forward_batch_vulkan(vk, prompt, 0), v2_linear(expected, raw.lm_head_w), 4e-5);
}

void test_vk_v2_generation(VkContext& vk) {
    BpeTokenizer tokenizer;
    tokenizer.init_byte_fallback();
    const int vocab = static_cast<int>(tokenizer.vocab_size());
    std::mt19937_64 rng(96);
    const auto raw = v2_raw(32, vocab, 0, rng);
    VulkanV2Model model(vk, raw);
    const std::vector<int32_t> prompt{1, 2};
    auto generate = [&](GenerationOptions options, GenerationStats* stats = nullptr) {
        return model.executor.generate_vulkan(vk, prompt, 8, stats, false, 0, -1, options);
    };
    GenerationOptions options;
    options.sampling.temperature = .8f;
    options.sampling.top_k = 5;
    options.sampling.seed = 279;
    if (generate(options) != generate(options))
        throw std::runtime_error("Vulkan sampling changed with a fixed seed");
    const JsonConstraint constraint(tokenizer, vocab, JsonRoot::null);
    options.json_constraint = &constraint;
    GenerationStats stats;
    auto result = generate(options, &stats);
    auto state = constraint.initial_state();
    for (size_t i = prompt.size(); i < result.size(); ++i)
        if (!constraint.accept(state, result[i])) throw std::runtime_error("Vulkan sampled outside JSON grammar");
    if (!state.complete() || !stats.json_complete)
        throw std::runtime_error("Vulkan did not complete the constrained JSON value");

    options = {};
    options.on_token = [](int32_t) { return false; };
    result = generate(options, &stats);
    if (result.size() != prompt.size() + 1 || stats.generated_tokens != 1)
        throw std::runtime_error("Vulkan continued after its token callback stopped");
    std::atomic_bool cancel{false};
    options.cancel = &cancel;
    options.on_token = [&](int32_t) { cancel = true; return true; };
    result = generate(options, &stats);
    if (result.size() != prompt.size() + 1 || !stats.cancelled)
        throw std::runtime_error("Vulkan ignored decode cancellation");
    if (generate(options, &stats) != prompt || !stats.cancelled || stats.prefilled_tokens != 0)
        throw std::runtime_error("Vulkan ignored prefill cancellation");
}

void test_vulkan_v2() {
    VkContext vk(0, 32, 32);
    const auto test = [&](auto run) {
        const auto mark = vk.weight_arena().mark();
        run(vk);
        vk.weight_arena().reset(mark);
    };
    test(test_vk_v2_gdn);
    test(test_vk_v2_fwht);
    test(test_vk_v2_gemma);
    test([](VkContext& ctx) { test_vk_v2_mla(ctx, false); });
    test([](VkContext& ctx) { test_vk_v2_mla(ctx, true); });
    test(test_vk_v2_compact_moe);
    test(test_vk_v2_generation);
    std::cout << "  [PASS] Vulkan V2 GDN, FWHT, Gemma 4, MLA, compact MoE and generation\n";
}

}  // namespace
}  // namespace fusionrt::test
