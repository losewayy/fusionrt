#include "fusionrt/models.h"
#include "model_loader.h"

#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_kernels.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace fusionrt {
using namespace model_io;

namespace {

bool expand_vk_weight(DType dtype) {
    return is_weight_codec_dtype(dtype) || dtype == DType::F16 || dtype == DType::BF16 || dtype == DType::Q4_1 ||
           dtype == DType::Q2_K || dtype == DType::Q4_K || dtype == DType::Q5_K || dtype == DType::Q6_K;
}

bool expand_vk_embedding(DType dtype, bool tied) {
    return expand_vk_weight(dtype) && (tied || (dtype != DType::F16 && dtype != DType::BF16));
}

bool pack_vk_qkv(const LoadedLayerRaw& layer, int hidden) {
    return !layer.gated_q_attn && hidden % 4 == 0 && layer.q_proj.desc.dtype != DType::Q4_0 && layer.q_proj.desc.dtype != DType::Q8_0;
}

void reserve_vulkan_model(VkContext& ctx, const LoadedModelRaw& raw) {
    const auto& cfg = raw.cfg;
    const auto& dense = cfg.dense;
    const int H = dense.hidden_size;
    const auto align = std::max<VkDeviceSize>(256, ctx.info().min_storage_offset_align);
    VkDeviceSize bytes = 0, state = 0;
    auto add = [&](VkDeviceSize n) { bytes = ((bytes + align - 1) & ~(align - 1)) + std::max<VkDeviceSize>(16, n); };
    const auto f32 = [](const HostTensor& t) { return static_cast<VkDeviceSize>(t.desc.numel()) * sizeof(float); };
    auto matrix = [&](const HostTensor& t) { add(expand_vk_weight(t.desc.dtype) ? f32(t) : t.desc.nbytes()); };
    auto state_tensor = [&](VkDeviceSize n) { const auto before = bytes; add(n); state += bytes - before; };
    const bool tied = raw.lm_head_w.bytes().empty();
    add(expand_vk_embedding(raw.tok_embeddings.desc.dtype, tied) ? f32(raw.tok_embeddings) : raw.tok_embeddings.desc.nbytes());
    add(f32(raw.final_norm_w));
    if (!tied) matrix(raw.lm_head_w);
    add(H * sizeof(float));
    add(H * sizeof(float));
    add(cfg.vocab_size * sizeof(float));
    add(sizeof(int32_t));
    add(std::max(1, dense.head_dim) * sizeof(float));
    auto routed = [&](const LoadedLayerRaw& l) {
        add(f32(l.mlp_norm.bytes().empty() ? l.attn_norm : l.mlp_norm));
        add(f32(l.router_w));
        add(f32(l.experts_gate_up) + (l.experts_up.bytes().empty() ? 0 : f32(l.experts_up)));
        add(f32(l.experts_down));
        if (l.has_shared) { matrix(l.shared_gate_up); matrix(l.shared_down); }
        if (!l.shared_gate.bytes().empty()) matrix(l.shared_gate);
    };
    auto dense_ffn = [&](const LoadedLayerRaw& l) {
        add(f32(l.mlp_norm));
        if (l.gate_up.bytes().empty()) add(f32(l.gate) + f32(l.up));
        else matrix(l.gate_up);
        matrix(l.down);
    };
    for (const auto& l : raw.layers) {
        if (l.kind == 0) {
            add(f32(l.attn_norm));
            if (pack_vk_qkv(l, H)) add(f32(l.q_proj) + f32(l.k_proj) + f32(l.v_proj));
            else { matrix(l.q_proj); matrix(l.k_proj); matrix(l.v_proj); }
            for (const auto* bias : {&l.q_bias, &l.k_bias, &l.v_bias})
                if (!bias->bytes().empty()) add(bias->desc.nbytes());
            if (!l.q_norm.bytes().empty() && !l.k_norm.bytes().empty()) { add(f32(l.q_norm)); add(f32(l.k_norm)); }
            matrix(l.o_proj);
            if (l.router_w.bytes().empty()) dense_ffn(l);
            else routed(l);
            const auto plane = static_cast<VkDeviceSize>(dense.max_seq_len) * dense.num_key_value_heads *
                dense.head_dim * sizeof(float);
            state_tensor(plane);
            state_tensor(plane);
        } else if (l.kind == 1) {
            add(f32(l.attn_norm));
            for (const auto* w : {&l.qkv_proj, &l.z_proj, &l.a_proj, &l.b_proj}) matrix(*w);
            for (const auto* w : {&l.conv_w, &l.conv_bias, &l.A_log, &l.dt_bias, &l.gdn_norm_w}) add(f32(*w));
            matrix(l.o_proj);
            if (l.router_w.bytes().empty()) dense_ffn(l);
            else routed(l);
            const auto& g = cfg.gdn;
            state_tensor(static_cast<VkDeviceSize>(l.conv_w.desc.shape[0]) * 3 * sizeof(float));
            state_tensor(static_cast<VkDeviceSize>(g.num_v_heads) * g.head_k_dim * g.head_v_dim * sizeof(float));
        } else routed(l);
    }
    ctx.reserve_model_memory(bytes - state, state);
}

}  // namespace

static void upload_raw_model_to_vulkan(VkContext& ctx, const LoadedModelRaw& raw,
                                       HybridModelConfig& cfg_out,
                                       Tensor& tok_emb, Tensor& final_norm, Tensor& lm_head,
                                       Tensor& hidden_a, Tensor& hidden_b, Tensor& logits,
                                       Tensor& tok_id, Tensor& dummy_norm,
                                       std::vector<int>& layer_kinds, std::vector<size_t>& layer_indices,
                                       std::vector<VkDenseGQALayer>& dense_layers,
                                       std::vector<VkGatedDeltaNetLayer>& gdn_layers,
                                       std::vector<VkSparseMoELayer>& moe_layers) {
    reserve_vulkan_model(ctx, raw);
    cfg_out = raw.cfg;
    int H = cfg_out.dense.hidden_size;
    // Vulkan shader embed_lookup / gemv supports F32, F16, BF16, Q8_0, Q4_0;
    // for F16/BF16 weights in linear projections on Vulkan, convert 2D matrices to F32 if needed
    auto upload_vk_weight = [&](const HostTensor& ht) -> Tensor {
        if (expand_vk_weight(ht.desc.dtype)) {
            return ctx.upload(to_f32_host(ht), true);
        }
        return ctx.upload(ht, true);
    };

    const bool tied = raw.lm_head_w.bytes().empty();
    tok_emb = ctx.upload(
        expand_vk_embedding(raw.tok_embeddings.desc.dtype, tied)
            ? to_f32_host(raw.tok_embeddings) : raw.tok_embeddings,
        true);
    final_norm = ctx.upload(to_f32_host(raw.final_norm_w), true);
    lm_head = tied ? tok_emb : upload_vk_weight(raw.lm_head_w);
    hidden_a = ctx.weight_arena().alloc_tensor({H}, DType::F32, "vk_ha");
    hidden_b = ctx.weight_arena().alloc_tensor({H}, DType::F32, "vk_hb");
    logits = ctx.weight_arena().alloc_tensor({cfg_out.vocab_size}, DType::F32, "vk_logits");
    tok_id = ctx.weight_arena().alloc_tensor({1}, DType::I32, "vk_tid");
    dummy_norm = ctx.upload(make_ones_f32({std::max(1, cfg_out.dense.head_dim)}), true);

    layer_kinds.clear();
    layer_indices.clear();
    dense_layers.clear();
    gdn_layers.clear();
    moe_layers.clear();

    auto upload_moe = [&](const LoadedLayerRaw& lr) {
        VkSparseMoELayer L;
        L.cfg = cfg_out.moe;
        L.cfg.has_shared_expert = lr.has_shared;
        L.cfg.num_experts = L.cfg.gpu_resident_experts = static_cast<int>(lr.router_w.desc.shape[0]);
        L.cfg.intermediate_size = static_cast<int>(lr.experts_down.desc.shape[2]);
        L.norm_w = ctx.upload(to_f32_host(lr.mlp_norm.bytes().empty() ? lr.attn_norm : lr.mlp_norm), true);
        L.router_w = ctx.upload(to_f32_host(lr.router_w), true);
        if (!lr.shared_gate.bytes().empty()) L.shared_gate_w = ctx.upload(lr.shared_gate, true);
        auto scale_experts = [](HostTensor& w, const HostTensor& scales, int offset, int width) {
            if (scales.bytes().empty()) return;
            const auto cols = w.desc.shape[2];
            for (int e = 0; e < w.desc.shape[0]; ++e) {
                auto* start = w.data<float>() + (e * w.desc.shape[1] + offset) * cols;
                for (int64_t i = 0; i < width * cols; ++i) start[i] *= scales.data<float>()[e];
            }
        };
        auto gate = to_f32_host(lr.experts_gate_up);
        const int I = L.cfg.intermediate_size;
        scale_experts(gate, lr.expert_gate_scale, 0, I);
        if (!lr.experts_up.bytes().empty()) {
            auto up = to_f32_host(lr.experts_up);
            scale_experts(up, lr.expert_up_scale, 0, I);
            const int E = L.cfg.num_experts;
            auto packed = HostTensor::create({E, 2 * I, H}, DType::F32);
            const size_t bytes = gate.bytes().size() / E;
            for (int e = 0; e < E; ++e) {
                std::memcpy(packed.storage.data() + (2 * e) * bytes, gate.bytes().data() + e * bytes, bytes);
                std::memcpy(packed.storage.data() + (2 * e + 1) * bytes, up.bytes().data() + e * bytes, bytes);
            }
            gate = std::move(packed);
        } else {
            scale_experts(gate, lr.expert_up_scale, I, I);
        }
        auto down = to_f32_host(lr.experts_down);
        scale_experts(down, lr.expert_down_scale, 0, H);
        L.gpu_gate_up_w = ctx.upload(gate, true);
        L.gpu_down_w = ctx.upload(down, true);
        if (lr.has_shared) {
            L.cfg.shared_intermediate_size = static_cast<int>(lr.shared_down.desc.shape[1]);
            L.shared_gate_up_w = upload_vk_weight(lr.shared_gate_up);
            L.shared_down_w = upload_vk_weight(lr.shared_down);
        }
        moe_layers.push_back(std::move(L));
        return static_cast<int>(moe_layers.size() - 1);
    };

    for (const auto& lr : raw.layers) {
        layer_kinds.push_back(lr.kind);
        if (lr.kind == 0) {
            VkDenseGQALayer L;
            L.cfg = cfg_out.dense;
            L.cfg.gated_q_attn = lr.gated_q_attn;
            L.cfg.qk_norm = !lr.q_norm.bytes().empty() && !lr.k_norm.bytes().empty();
            L.attn_norm_w = ctx.upload(to_f32_host(lr.attn_norm), true);
            const bool packed_fp32 = pack_vk_qkv(lr, H);
            if (packed_fp32) {
                L.qkv_proj_fused = ctx.upload(concat_rows_same_dtype(
                    {to_f32_host(lr.q_proj), to_f32_host(lr.k_proj), to_f32_host(lr.v_proj)}), true);
            } else {
                L.q_proj = upload_vk_weight(lr.q_proj);
                L.k_proj = upload_vk_weight(lr.k_proj);
                L.v_proj = upload_vk_weight(lr.v_proj);
            }
            if (!lr.q_bias.bytes().empty()) L.q_bias = ctx.upload(lr.q_bias, true);
            if (!lr.k_bias.bytes().empty()) L.k_bias = ctx.upload(lr.k_bias, true);
            if (!lr.v_bias.bytes().empty()) L.v_bias = ctx.upload(lr.v_bias, true);
            if (L.cfg.qk_norm) {
                L.q_norm_w = ctx.upload(to_f32_host(lr.q_norm), true);
                L.k_norm_w = ctx.upload(to_f32_host(lr.k_norm), true);
            } else {
                L.q_norm_w = dummy_norm;
                L.k_norm_w = dummy_norm;
            }
            L.o_proj = upload_vk_weight(lr.o_proj);
            if (!lr.router_w.bytes().empty()) {
                L.moe_index = upload_moe(lr);
            } else {
                L.mlp_norm_w = ctx.upload(to_f32_host(lr.mlp_norm), true);
                L.gate_up_proj = lr.gate_up.bytes().empty()
                    ? ctx.upload(concat_rows_same_dtype({to_f32_host(lr.gate), to_f32_host(lr.up)}), true)
                    : upload_vk_weight(lr.gate_up);
                L.down_proj = upload_vk_weight(lr.down);
            }
            L.k_cache = ctx.weight_arena().alloc_tensor({L.cfg.max_seq_len, L.cfg.num_key_value_heads, L.cfg.head_dim}, DType::F32, "vk_kc");
            L.v_cache = ctx.weight_arena().alloc_tensor({L.cfg.max_seq_len, L.cfg.num_key_value_heads, L.cfg.head_dim}, DType::F32, "vk_vc");
            layer_indices.push_back(dense_layers.size());
            dense_layers.push_back(std::move(L));
        } else if (lr.kind == 1) {
            VkGatedDeltaNetLayer L;
            L.cfg = cfg_out.gdn;
            L.attn_norm_w = ctx.upload(to_f32_host(lr.attn_norm), true);
            L.qkv_proj = upload_vk_weight(lr.qkv_proj);
            L.z_proj = upload_vk_weight(lr.z_proj);
            L.a_proj = upload_vk_weight(lr.a_proj);
            L.b_proj = upload_vk_weight(lr.b_proj);
            L.conv_w = ctx.upload(to_f32_host(lr.conv_w), true);
            L.conv_bias = ctx.upload(to_f32_host(lr.conv_bias), true);
            L.A_log = ctx.upload(to_f32_host(lr.A_log), true);
            L.dt_bias = ctx.upload(to_f32_host(lr.dt_bias), true);
            L.norm_w = ctx.upload(to_f32_host(lr.gdn_norm_w), true);
            L.o_proj = upload_vk_weight(lr.o_proj);
            if (lr.router_w.bytes().empty()) {
                L.mlp_norm_w = ctx.upload(to_f32_host(lr.mlp_norm), true);
                L.gate_up_proj = lr.gate_up.bytes().empty()
                        ? ctx.upload(concat_rows_same_dtype({to_f32_host(lr.gate), to_f32_host(lr.up)}), true)
                        : upload_vk_weight(lr.gate_up);
                L.down_proj = upload_vk_weight(lr.down);
            } else {
                L.moe_index = upload_moe(lr);
            }
            int conv_dim = static_cast<int>(lr.conv_w.desc.shape[0]);
            L.conv_state = ctx.weight_arena().alloc_tensor({conv_dim, 3}, DType::F32, "vk_cs");
            L.ssm_state = ctx.weight_arena().alloc_tensor({L.cfg.num_v_heads, L.cfg.head_k_dim, L.cfg.head_v_dim}, DType::F32, "vk_ss");
            layer_indices.push_back(gdn_layers.size());
            gdn_layers.push_back(std::move(L));
        } else {
            layer_indices.push_back(upload_moe(lr));
        }
    }
    ctx.synchronize();
}

void FusionModel::init_synthetic_vulkan(VkContext& ctx, uint64_t seed) {
    std::mt19937_64 rng(seed);
    int H = (cfg_.arch == "hybrid") ? cfg_.gdn.hidden_size
          : (cfg_.arch == "moe") ? cfg_.moe.hidden_size
          : cfg_.dense.hidden_size;

    LoadedModelRaw raw;
    raw.cfg = cfg_;
    raw.tok_embeddings = make_rand_f32({cfg_.vocab_size, H}, rng, 0.1f);
    raw.final_norm_w = make_ones_f32({H});
    raw.lm_head_w = make_rand_f32({cfg_.vocab_size, H}, rng, 0.05f);
    for (int l = 0; l < cfg_.num_layers; ++l) {
        LoadedLayerRaw lr;
        if (cfg_.arch == "hybrid") lr.kind = ((l % 4) == 3) ? 0 : 1;
        else if (cfg_.arch == "moe") lr.kind = 2;
        else lr.kind = 0;

        if (lr.kind == 0) {
            DType wdt = cfg_.dense.weight_dtype;
            int q_dim = cfg_.dense.num_attention_heads * cfg_.dense.head_dim;
            int kv_dim = cfg_.dense.num_key_value_heads * cfg_.dense.head_dim;
            int I = cfg_.dense.intermediate_size;
            lr.attn_norm = make_ones_f32({H});
            HostTensor hq = make_rand_f32({q_dim, H}, rng, 0.04f);
            HostTensor hk = make_rand_f32({kv_dim, H}, rng, 0.04f);
            HostTensor hv = make_rand_f32({kv_dim, H}, rng, 0.04f);
            lr.q_proj = quantize_host_matrix(hq, wdt);
            lr.k_proj = quantize_host_matrix(hk, wdt);
            lr.v_proj = quantize_host_matrix(hv, wdt);
            lr.qkv_proj = quantize_host_matrix(concat_rows_same_dtype({hq, hk, hv}), wdt);
            if (cfg_.dense.qk_norm) {
                lr.q_norm = make_ones_f32({cfg_.dense.head_dim});
                lr.k_norm = make_ones_f32({cfg_.dense.head_dim});
            }
            lr.o_proj = quantize_host_matrix(make_rand_f32({H, q_dim}, rng, 0.04f), wdt);
            lr.mlp_norm = make_ones_f32({H});
            lr.gate_up = quantize_host_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt);
            lr.down = quantize_host_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt);
        } else if (lr.kind == 1) {
            DType wdt = cfg_.gdn.weight_dtype;
            int nk = cfg_.gdn.num_k_heads, nv = cfg_.gdn.num_v_heads, dk = cfg_.gdn.head_k_dim, dv = cfg_.gdn.head_v_dim;
            int conv_dim = 2 * nk * dk + nv * dv, v_dim = nv * dv, I = cfg_.gdn.intermediate_size;
            lr.attn_norm = make_ones_f32({H});
            lr.qkv_proj = quantize_host_matrix(make_rand_f32({conv_dim, H}, rng, 0.04f), wdt);
            lr.z_proj = quantize_host_matrix(make_rand_f32({v_dim, H}, rng, 0.04f), wdt);
            lr.a_proj = quantize_host_matrix(make_rand_f32({nv, H}, rng, 0.04f), wdt);
            lr.b_proj = quantize_host_matrix(make_rand_f32({nv, H}, rng, 0.04f), wdt);
            lr.conv_w = make_rand_f32({conv_dim, 4}, rng, 0.1f);
            lr.conv_bias = make_rand_f32({conv_dim}, rng, 0.01f);
            lr.A_log = HostTensor::create({nv}, DType::F32);
            for (int i = 0; i < nv; ++i) lr.A_log.data<float>()[i] = -2.0f;
            lr.dt_bias = make_rand_f32({nv}, rng, 0.02f);
            lr.gdn_norm_w = make_ones_f32({dv});
            lr.o_proj = quantize_host_matrix(make_rand_f32({H, v_dim}, rng, 0.04f), wdt);
            lr.mlp_norm = make_ones_f32({H});
            lr.gate_up = quantize_host_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt);
            lr.down = quantize_host_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt);
        } else {
            int I = cfg_.moe.intermediate_size, E = cfg_.moe.num_experts;
            lr.attn_norm = make_ones_f32({H});
            lr.router_w = make_rand_f32({E, H}, rng, 0.1f);
            lr.experts_gate_up = make_rand_f32({E, 2 * I, H}, rng, 0.04f);
            lr.experts_down = make_rand_f32({E, H, I}, rng, 0.04f);
            if (cfg_.moe.has_shared_expert) {
                lr.has_shared = true;
                lr.shared_gate_up = make_rand_f32({2 * cfg_.moe.shared_intermediate_size, H}, rng, 0.04f);
                lr.shared_down = make_rand_f32({H, cfg_.moe.shared_intermediate_size}, rng, 0.04f);
            }
        }
        raw.layers.push_back(std::move(lr));
    }
    upload_raw_model_to_vulkan(ctx, raw, cfg_, vk_tok_embeddings_, vk_final_norm_w_, vk_lm_head_w_,
                               vk_hidden_a_, vk_hidden_b_, vk_logits_, vk_tok_id_, vk_dummy_norm_,
                               layer_kinds_, layer_indices_, vk_dense_layers_, vk_gdn_layers_, vk_moe_layers_);
}

bool FusionModel::load_gguf_vulkan(VkContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    LoadedModelRaw raw;
    if (!parse_gguf_model_raw(path, max_seq_override, raw, err)) return false;
    raw.layers.resize(raw.cfg.num_layers);
    if (raw.hadamard_folded) {
        if (err) *err = "Hadamard-folded GGUF weights require the CUDA backend";
        return false;
    }
    if (raw.cfg.model_architecture == "gemma4") {
        if (err) *err = "Gemma 4 GGUF requires the CUDA backend";
        return false;
    }
    if (raw.cfg.dense.mla.latent_rank > 0) {
        if (err) *err = "DeepSeek MLA requires the CUDA backend";
        return false;
    }
    if (!raw.rope_factors.bytes().empty()) {
        if (err) *err = "Vulkan does not support GGUF RoPE frequency factors";
        return false;
    }
    upload_raw_model_to_vulkan(ctx, raw, cfg_, vk_tok_embeddings_, vk_final_norm_w_, vk_lm_head_w_,
                               vk_hidden_a_, vk_hidden_b_, vk_logits_, vk_tok_id_, vk_dummy_norm_,
                               layer_kinds_, layer_indices_, vk_dense_layers_, vk_gdn_layers_, vk_moe_layers_);
    return true;
}

bool FusionModel::load_safetensors_vulkan(VkContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    LoadedModelRaw raw;
    if (!parse_safetensors_model_raw(path, max_seq_override, raw, err)) return false;
    upload_raw_model_to_vulkan(ctx, raw, cfg_, vk_tok_embeddings_, vk_final_norm_w_, vk_lm_head_w_,
                               vk_hidden_a_, vk_hidden_b_, vk_logits_, vk_tok_id_, vk_dummy_norm_,
                               layer_kinds_, layer_indices_, vk_dense_layers_, vk_gdn_layers_, vk_moe_layers_);
    return true;
}

void FusionModel::reset_kv_and_state_vulkan(VkContext& ctx) {
    for (auto& L : vk_dense_layers_) {
        std::vector<int64_t> ks(L.k_cache.shape.begin(), L.k_cache.shape.begin() + L.k_cache.ndim);
        HostTensor z = HostTensor::create(ks, DType::F32);
        // Zero caches cleanly via staging copy if small, or scratch upload
        (void)z;
    }
    for (auto& L : vk_gdn_layers_) {
        std::vector<int64_t> css(L.conv_state.shape.begin(), L.conv_state.shape.begin() + L.conv_state.ndim);
        std::vector<int64_t> sss(L.ssm_state.shape.begin(), L.ssm_state.shape.begin() + L.ssm_state.ndim);
        Tensor tmp_cs = ctx.upload(make_zeros_f32(css), false);
        Tensor tmp_ss = ctx.upload(make_zeros_f32(sss), false);
        // Copy via add_residual? Better: upload zeros only needed for GDN states
        std::vector<VkBatchDispatch> ops;
        std::vector<std::vector<uint8_t>> ps;
        vk_ops::PushResidual pc1{static_cast<int>(L.conv_state.numel())};
        auto sp = [&](const auto& v) { ps.emplace_back(sizeof(v)); std::memcpy(ps.back().data(), &v, sizeof(v)); return ps.back().data(); };
        ops.push_back({"add_residual.comp", static_cast<uint32_t>((pc1.n + 255) / 256), 1, 1, {tmp_cs, tmp_cs, L.conv_state}, sp(pc1), sizeof(pc1)});
        vk_ops::PushResidual pc2{static_cast<int>(L.ssm_state.numel())};
        ops.push_back({"add_residual.comp", static_cast<uint32_t>((pc2.n + 255) / 256), 1, 1, {tmp_ss, tmp_ss, L.ssm_state}, sp(pc2), sizeof(pc2)});
        ctx.dispatch_batched(ops, false);
        ctx.scratch_arena().reset();
    }
}

std::vector<float> FusionModel::forward_token_vulkan(VkContext& ctx, int32_t token_id, int pos) {
    if (pos < 0 || pos >= cfg_.dense.max_seq_len) throw std::invalid_argument("Token position exceeds context");
    validate_token_id(token_id, pos, cfg_.vocab_size);
    int H = static_cast<int>(vk_hidden_a_.shape[0]);
    HostTensor h_tid = HostTensor::create({1}, DType::I32);
    h_tid.data<int32_t>()[0] = token_id;
    // Upload token id into scratch and run ALL layers + final head in a single batched command buffer!
    size_t smark = ctx.scratch_arena().mark();
    Tensor d_tid = ctx.upload(h_tid, false);

    std::vector<VkBatchDispatch> ops;
    std::vector<std::vector<uint8_t>> ps;
    auto sp = [&](const auto& v) -> const void* {
        ps.emplace_back(sizeof(v));
        std::memcpy(ps.back().data(), &v, sizeof(v));
        return ps.back().data();
    };

    // Embedding lookup -> vk_hidden_a_
    {
        vk_ops::PushEmbed pc{1, H, static_cast<int>(vk_tok_embeddings_.dtype)};
        ops.push_back({"embed_lookup.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1,
                       {vk_tok_embeddings_, d_tid, vk_hidden_a_}, sp(pc), sizeof(pc)});
    }

    Tensor* cur = &vk_hidden_a_;
    Tensor* nxt = &vk_hidden_b_;

    auto append_moe = [&](const VkSparseMoELayer& L, const Tensor& input, const Tensor& output) {
        int E = L.cfg.num_experts, K = L.cfg.top_k, I = L.cfg.intermediate_size;
        Tensor xn = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
        Tensor rl = ctx.scratch_arena().alloc_tensor({E}, DType::F32);
        Tensor tidx = ctx.scratch_arena().alloc_tensor({K}, DType::I32);
        Tensor tw = ctx.scratch_arena().alloc_tensor({K}, DType::F32);
        Tensor sinter = ctx.scratch_arena().alloc_tensor({K, I}, DType::F32);
        Tensor mout = ctx.scratch_arena().alloc_tensor({H}, DType::F32);

        {
            vk_ops::PushRmsNorm npc{1, H, L.cfg.rms_norm_eps};
            ops.push_back({"rmsnorm.comp", 1, 1, 1, {input, L.norm_w, xn}, sp(npc), sizeof(npc)});
        }
        vk_ops::append_linear_dispatch(ops, xn, L.router_w, rl, 1, E, H, ps);
        {
            vk_ops::PushMoeTopk pc{E, K, L.cfg.norm_topk_prob ? 1 : 0, L.cfg.expert_weights_scale};
            ops.push_back({"moe_topk.comp", 1, 1, 1, {rl, tidx, tw}, sp(pc), sizeof(pc)});
        }
        {
            vk_ops::PushMoeSwiglu pc{H, I, K, 0};
            ops.push_back({"moe_swiglu_combine.comp", static_cast<uint32_t>(K * I), 1, 1,
                           {xn, tidx, L.gpu_gate_up_w, tw, sinter}, sp(pc), sizeof(pc)});
        }
        {
            vk_ops::PushMoeSwiglu pc{H, I, K, 1};
            ops.push_back({"moe_swiglu_combine.comp", static_cast<uint32_t>(H), 1, 1,
                           {sinter, tidx, L.gpu_down_w, tw, mout}, sp(pc), sizeof(pc)});
        }
        if (L.cfg.has_shared_expert) {
            int shI = L.cfg.shared_intermediate_size;
            Tensor shgu = ctx.scratch_arena().alloc_tensor({2 * shI}, DType::F32);
            Tensor shsw = ctx.scratch_arena().alloc_tensor({shI}, DType::F32);
            Tensor shd = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
            vk_ops::append_linear_dispatch(ops, xn, L.shared_gate_up_w, shgu, 1, 2 * shI, H, ps);
            vk_ops::PushSwiglu spc{1, shI};
            ops.push_back({"swiglu.comp", static_cast<uint32_t>((shI + 255) / 256), 1, 1, {shgu, shsw}, sp(spc), sizeof(spc)});
            vk_ops::append_linear_dispatch(ops, shsw, L.shared_down_w, shd, 1, H, shI, ps);
            if (L.shared_gate_w.vk_buffer) {
                auto gate = ctx.scratch_arena().alloc_tensor({1}, DType::F32);
                vk_ops::append_linear_dispatch(ops, xn, L.shared_gate_w, gate, 1, 1, H, ps);
                const int pc[] = {H, 1};
                ops.push_back({"sigmoid_gate.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1,
                               {shd, gate}, sp(pc), sizeof(pc)});
            }
            vk_ops::PushResidual rpc{H};
            ops.push_back({"add_residual.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1, {mout, shd, mout}, sp(rpc), sizeof(rpc)});
        }
        {
            vk_ops::PushResidual pc{H};
            ops.push_back({"add_residual.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1, {input, mout, output}, sp(pc), sizeof(pc)});
        }
    };

    for (size_t i = 0; i < layer_kinds_.size(); ++i) {
        int kind = layer_kinds_[i];
        size_t idx = layer_indices_[i];
        if (kind == 0) {
            auto& L = vk_dense_layers_[idx];
            int nq = L.cfg.num_attention_heads, nkv = L.cfg.num_key_value_heads, hd = L.cfg.head_dim;
            int q_dim = nq * hd, kv_dim = nkv * hd, inter = L.cfg.intermediate_size;
            int rot_dim = std::max(2, static_cast<int>(hd * L.cfg.partial_rotary_factor) & ~1);
            Tensor xn = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
            Tensor qb = ctx.scratch_arena().alloc_tensor({q_dim}, DType::F32);
            Tensor q_gate, q_full;
            if (L.cfg.gated_q_attn) {
                q_gate = ctx.scratch_arena().alloc_tensor({q_dim}, DType::F32);
                q_full = ctx.scratch_arena().alloc_tensor({2 * q_dim}, DType::F32);
            }
            Tensor kb = ctx.scratch_arena().alloc_tensor({kv_dim}, DType::F32);
            Tensor vb = ctx.scratch_arena().alloc_tensor({kv_dim}, DType::F32);
            Tensor aout = ctx.scratch_arena().alloc_tensor({q_dim}, DType::F32);
            Tensor ob = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
            Tensor gub = ctx.scratch_arena().alloc_tensor({2 * inter}, DType::F32);
            Tensor swb = ctx.scratch_arena().alloc_tensor({inter}, DType::F32);
            Tensor db = ctx.scratch_arena().alloc_tensor({H}, DType::F32);

            if (L.qkv_proj_fused.vk_buffer && L.qkv_proj_fused.dtype == DType::F32 && (H % 4) == 0) {
                struct PushFusedQkv { int q_dim, kv_dim, K; float eps; } pc{q_dim, kv_dim, H, L.cfg.rms_norm_eps};
                ops.push_back({"fused_rmsnorm_qkv_vec4.comp", static_cast<uint32_t>(q_dim + 2 * kv_dim), 1, 1,
                               {*cur, L.attn_norm_w, L.qkv_proj_fused, qb, kb, vb}, sp(pc), sizeof(pc)});
            } else {
                vk_ops::PushRmsNorm npc{1, H, L.cfg.rms_norm_eps};
                ops.push_back({"rmsnorm.comp", 1, 1, 1, {*cur, L.attn_norm_w, xn}, sp(npc), sizeof(npc)});
                vk_ops::append_linear_dispatch(ops, xn, L.q_proj, L.cfg.gated_q_attn ? q_full : qb,
                                               1, L.cfg.gated_q_attn ? 2 * q_dim : q_dim, H, ps);
                vk_ops::append_linear_dispatch(ops, xn, L.k_proj, kb, 1, kv_dim, H, ps);
                vk_ops::append_linear_dispatch(ops, xn, L.v_proj, vb, 1, kv_dim, H, ps);
            }
            auto add_bias = [&](const Tensor& value, const Tensor& bias, int cols) {
                if (!bias.vk_buffer) return;
                vk_ops::PushResidual pc{cols};
                ops.push_back({"add_residual.comp", static_cast<uint32_t>((cols + 255) / 256), 1, 1,
                               {value, bias, value}, sp(pc), sizeof(pc)});
            };
            if (L.cfg.gated_q_attn) {
                add_bias(q_full, L.q_bias, 2 * q_dim);
                const int pc[] = {q_dim, hd};
                ops.push_back({"split_query_gate.comp", static_cast<uint32_t>((q_dim + 255) / 256), 1, 1,
                               {q_full, qb, q_gate}, sp(pc), sizeof(pc)});
            } else add_bias(qb, L.q_bias, q_dim);
            add_bias(kb, L.k_bias, kv_dim);
            add_bias(vb, L.v_bias, kv_dim);
            {
                int flags = (L.cfg.qk_norm ? 1 : 0) | (L.cfg.rope_neox ? 2 : 0);
                vk_ops::PushRopeKv pc{nq, nkv, hd, rot_dim, pos, L.cfg.rope_theta, L.cfg.rms_norm_eps, flags};
                ops.push_back({"rope_kv_append.comp", static_cast<uint32_t>(nq + nkv), 1, 1,
                               {qb, kb, vb, L.q_norm_w, L.k_norm_w, L.k_cache, L.v_cache}, sp(pc), sizeof(pc)});
            }
            {
                vk_ops::PushGqaDecode pc{nq, nkv, hd, pos + 1, 1.0f / std::sqrt(static_cast<float>(hd))};
                ops.push_back({"gqa_decode.comp", static_cast<uint32_t>(nq), 1, 1,
                               {qb, L.k_cache, L.v_cache, aout}, sp(pc), sizeof(pc)});
            }
            if (L.cfg.gated_q_attn) {
                const int pc[] = {q_dim, 0};
                ops.push_back({"sigmoid_gate.comp", static_cast<uint32_t>((q_dim + 255) / 256), 1, 1,
                               {aout, q_gate}, sp(pc), sizeof(pc)});
            }
            vk_ops::append_linear_dispatch(ops, aout, L.o_proj, ob, 1, H, q_dim, ps);
            {
                vk_ops::PushResidual pc{H};
                ops.push_back({"add_residual.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1, {*cur, ob, *nxt}, sp(pc), sizeof(pc)});
            }
            if (L.moe_index >= 0) {
                append_moe(vk_moe_layers_[L.moe_index], *nxt, *nxt);
                std::swap(cur, nxt);
                continue;
            }
            vk_ops::append_rmsnorm_linear_dispatch(ops, *nxt, L.mlp_norm_w, L.gate_up_proj, xn, gub, 1, 2 * inter, H, L.cfg.rms_norm_eps, ps);
            {
                vk_ops::PushSwiglu pc{1, inter};
                ops.push_back({"swiglu.comp", static_cast<uint32_t>((inter + 255) / 256), 1, 1, {gub, swb}, sp(pc), sizeof(pc)});
            }
            vk_ops::append_linear_dispatch(ops, swb, L.down_proj, db, 1, H, inter, ps);
            {
                vk_ops::PushResidual pc{H};
                ops.push_back({"add_residual.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1, {*nxt, db, *nxt}, sp(pc), sizeof(pc)});
            }
            std::swap(cur, nxt);
        } else if (kind == 1) {
            auto& L = vk_gdn_layers_[idx];
            int nk = L.cfg.num_k_heads, nv = L.cfg.num_v_heads, dk = L.cfg.head_k_dim, dv = L.cfg.head_v_dim;
            int conv_dim = 2 * nk * dk + nv * dv, v_dim = nv * dv, inter = L.cfg.intermediate_size;
            Tensor xn = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
            Tensor qkvb = ctx.scratch_arena().alloc_tensor({conv_dim}, DType::F32);
            Tensor zb = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
            Tensor ab = ctx.scratch_arena().alloc_tensor({nv}, DType::F32);
            Tensor bb = ctx.scratch_arena().alloc_tensor({nv}, DType::F32);
            Tensor gdn_o = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
            Tensor ob = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
            Tensor gub = ctx.scratch_arena().alloc_tensor({2 * inter}, DType::F32);
            Tensor swb = ctx.scratch_arena().alloc_tensor({inter}, DType::F32);
            Tensor db = ctx.scratch_arena().alloc_tensor({H}, DType::F32);

            {
                vk_ops::PushRmsNorm npc{1, H, L.cfg.rms_norm_eps};
                ops.push_back({"rmsnorm.comp", 1, 1, 1, {*cur, L.attn_norm_w, xn}, sp(npc), sizeof(npc)});
            }
            vk_ops::append_linear_dispatch(ops, xn, L.qkv_proj, qkvb, 1, conv_dim, H, ps);
            vk_ops::append_linear_dispatch(ops, xn, L.z_proj, zb, 1, v_dim, H, ps);
            vk_ops::append_linear_dispatch(ops, xn, L.a_proj, ab, 1, nv, H, ps);
            vk_ops::append_linear_dispatch(ops, xn, L.b_proj, bb, 1, nv, H, ps);
            {
                vk_ops::PushGdnStep pc{nk, nv, dk, dv, L.cfg.rms_norm_eps, 1.0f / std::sqrt(static_cast<float>(dk)), 1,
                                      L.cfg.tiled_value_heads ? 1 : 0};
                ops.push_back({"gated_deltanet_step.comp", static_cast<uint32_t>(nk), 1, 1,
                               {qkvb, zb, ab, bb, L.conv_w, L.conv_bias, L.A_log, L.dt_bias, L.norm_w, L.conv_state, L.ssm_state, gdn_o},
                               sp(pc), sizeof(pc)});
            }
            vk_ops::append_linear_dispatch(ops, gdn_o, L.o_proj, ob, 1, H, v_dim, ps);
            {
                vk_ops::PushResidual pc{H};
                ops.push_back({"add_residual.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1, {*cur, ob, *nxt}, sp(pc), sizeof(pc)});
            }
            if (L.moe_index >= 0) {
                append_moe(vk_moe_layers_[L.moe_index], *nxt, *nxt);
                std::swap(cur, nxt);
                continue;
            }
            vk_ops::append_rmsnorm_linear_dispatch(ops, *nxt, L.mlp_norm_w, L.gate_up_proj, xn, gub, 1, 2 * inter, H, L.cfg.rms_norm_eps, ps);
            {
                vk_ops::PushSwiglu pc{1, inter};
                ops.push_back({"swiglu.comp", static_cast<uint32_t>((inter + 255) / 256), 1, 1, {gub, swb}, sp(pc), sizeof(pc)});
            }
            vk_ops::append_linear_dispatch(ops, swb, L.down_proj, db, 1, H, inter, ps);
            {
                vk_ops::PushResidual pc{H};
                ops.push_back({"add_residual.comp", static_cast<uint32_t>((H + 255) / 256), 1, 1, {*nxt, db, *nxt}, sp(pc), sizeof(pc)});
            }
            std::swap(cur, nxt);
        } else {
            append_moe(vk_moe_layers_[idx], *cur, *nxt);
            std::swap(cur, nxt);
        }
    }

    // Final RMSNorm + lm_head
    Tensor final_xn = ctx.scratch_arena().alloc_tensor({H}, DType::F32);
    vk_ops::append_rmsnorm_linear_dispatch(ops, *cur, vk_final_norm_w_, vk_lm_head_w_, final_xn, vk_logits_,
                                           1, cfg_.vocab_size, H, cfg_.dense.rms_norm_eps, ps);

    ctx.dispatch_batched(ops, false);
    HostTensor h_logits = ctx.download(vk_logits_);
    ctx.scratch_arena().reset(smark);
    return std::vector<float>(h_logits.data<float>(), h_logits.data<float>() + cfg_.vocab_size);
}

std::vector<int32_t> FusionModel::generate_vulkan(VkContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                  int max_new_tokens, GenerationStats* stats,
                                                  bool keep_state, int start_pos,
                                                  int32_t stop_eos_id) {
    if (max_new_tokens < 0 || start_pos < 0) throw std::invalid_argument("Invalid generation length or start position");
    if (!keep_state) start_pos = 0;
    std::vector<int32_t> tokens = prompt_ids;
    if (tokens.empty()) tokens.push_back(std::min(1, cfg_.vocab_size - 1));
    validate_token_ids(tokens, start_pos, cfg_.vocab_size);
    const auto total_tokens = static_cast<size_t>(start_pos) + tokens.size();
    const int context = cfg_.dense.max_seq_len;
    if (total_tokens > static_cast<size_t>(context)) throw prompt_context_error(total_tokens, context);
    max_new_tokens = std::min(max_new_tokens, context - static_cast<int>(total_tokens));
    const int prompt_len = static_cast<int>(tokens.size());
    if (!keep_state) reset_kv_and_state_vulkan(ctx);

    std::vector<float> logits;
    int pos = start_pos;
    auto tp0 = std::chrono::high_resolution_clock::now();
    for (int32_t t : tokens) {
        logits = forward_token_vulkan(ctx, t, pos++);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    int gen_count = 0;
    int decode_steps = 0;
    auto td0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < max_new_tokens; ++step) {
        const int32_t next_tok = Sampler::argmax(logits.data(), static_cast<int>(logits.size()));
        tokens.push_back(next_tok);
        ++gen_count;
        const bool eos = next_tok == stop_eos_id;
        // A continued turn starts after the last emitted token, including EOS.
        if (keep_state || (!eos && step + 1 < max_new_tokens)) {
            logits = forward_token_vulkan(ctx, next_tok, pos++);
            ++decode_steps;
        }
        if (eos) break;
    }
    auto td1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();

    if (stats) {
        stats->prompt_tokens = static_cast<int>(prompt_ids.size());
        stats->generated_tokens = gen_count;
        stats->prefill_ms = prefill_ms;
        stats->decode_ms = decode_ms;
        stats->prefill_tok_per_s = (prefill_ms > 0.0) ? (static_cast<double>(prompt_len) * 1000.0 / prefill_ms) : 0.0;
        stats->decode_tok_per_s = decode_ms > 0.0 ? static_cast<double>(decode_steps) * 1000.0 / decode_ms : 0.0;
    }
    return tokens;
}

}  // namespace fusionrt
#endif
