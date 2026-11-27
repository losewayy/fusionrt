#include "vulkan_weights.h"
#include "model_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace fusionrt {
using namespace model_io;

namespace {

bool shader_embedding(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16 ||
           dtype == DType::Q4_0 || dtype == DType::Q8_0;
}

bool pack_vk_qkv(const LoadedLayerRaw& layer) {
    return !layer.gated_q_attn && layer.q_bias.bytes().empty() && layer.k_bias.bytes().empty() &&
           layer.v_bias.bytes().empty() && layer.q_proj.desc.dtype == layer.k_proj.desc.dtype &&
           layer.q_proj.desc.dtype == layer.v_proj.desc.dtype;
}

void reserve_vulkan_model(VkContext& ctx, const LoadedModelRaw& raw, DType kv_type) {
    const auto& cfg = raw.cfg;
    const auto& dense = cfg.dense;
    const auto align = ctx.info().min_storage_offset_align;
    VkDeviceSize bytes = 0, state = 0;
    auto add = [&](VkDeviceSize n) {
        bytes = ((bytes + align - 1) & ~(align - 1)) + std::max<VkDeviceSize>(16, (n + 3) & ~VkDeviceSize{3});
    };
    const auto f32 = [](const HostTensor& t) { return static_cast<VkDeviceSize>(t.desc.numel()) * sizeof(float); };
    auto matrix = [&](const HostTensor& t) { add(t.desc.nbytes()); };
    auto state_tensor = [&](VkDeviceSize n) { const auto before = bytes; add(n); state += bytes - before; };
    matrix(raw.tok_embeddings);
    add(f32(raw.final_norm_w));
    if (!raw.lm_head_w.bytes().empty()) matrix(raw.lm_head_w);
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
        if (l.gate_up.bytes().empty()) { matrix(l.gate); matrix(l.up); }
        else matrix(l.gate_up);
        matrix(l.down);
    };
    for (const auto& l : raw.layers) {
        if (l.kind == 0) {
            add(f32(l.attn_norm));
            if (pack_vk_qkv(l)) add(l.q_proj.desc.nbytes() + l.k_proj.desc.nbytes() + l.v_proj.desc.nbytes());
            else { matrix(l.q_proj); matrix(l.k_proj); matrix(l.v_proj); }
            for (const auto* bias : {&l.q_bias, &l.k_bias, &l.v_bias})
                if (!bias->bytes().empty()) add(f32(*bias));
            if (!l.q_norm.bytes().empty() && !l.k_norm.bytes().empty()) { add(f32(l.q_norm)); add(f32(l.k_norm)); }
            matrix(l.o_proj);
            if (l.router_w.bytes().empty()) dense_ffn(l);
            else routed(l);
            const auto plane = static_cast<VkDeviceSize>(dense.max_seq_len) * dense.num_key_value_heads *
                (dense.head_dim / dtype_block_size(kv_type)) * dtype_type_size(kv_type);
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

HostTensor synthetic_matrix(const HostTensor& fp32, DType dtype) {
    if (dtype != DType::Q4_K) return quantize_host_matrix(fp32, dtype);
    if (fp32.desc.shape[1] % 256 != 0)
        throw std::invalid_argument("Synthetic Q4_K matrices require a multiple of 256 columns");
    HostTensor quant({fp32.desc.shape[0], fp32.desc.shape[1]}, dtype);
    // A common scale and zero point for each 256-element synthetic block.
    for (int64_t b = 0; b < fp32.desc.numel() / 256; ++b) {
        auto& block = quant.data<BlockQ4_K>()[b];
        const auto* row = fp32.data<float>() + b * 256;
        const auto [lo, hi] = std::minmax_element(row, row + 256);
        block.d = fp32_to_fp16(std::max(std::abs(*lo), std::abs(*hi)) / 7.0f);
        const float step = fp16_to_fp32(block.d);
        block.dmin = fp32_to_fp16(step * 8.0f);
        std::fill_n(block.scales, 8, 1);
        std::fill_n(block.scales + 8, 4, 0x11);
        const auto code = [step](float v) {
            return step == 0.0f ? 8 : std::clamp(static_cast<int>(std::round(v / step)) + 8, 0, 15);
        };
        for (int i = 0; i < 256; i += 64)
            for (int j = 0; j < 32; ++j)
                block.qs[i / 2 + j] = static_cast<uint8_t>(code(row[i + j]) | (code(row[i + j + 32]) << 4));
    }
    return quant;
}

}  // namespace

HostTensor VulkanModelWeights::gather_host_embeddings(std::span<const int32_t> tokens) const {
    if (host_embeddings_.bytes().empty()) return {};
    const auto hidden = host_embeddings_.desc.shape[1];
    HostTensor rows({static_cast<int64_t>(tokens.size()), hidden}, DType::F32);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto row = to_f32_host(host_embeddings_.slice_dim0(tokens[i], 1));
        std::copy_n(row.data<float>(), hidden, rows.data<float>() + i * hidden);
    }
    return rows;
}

void VulkanModelWeights::upload(VkContext& ctx, const LoadedModelRaw& raw, const VulkanLoadHooks& hooks,
                                DType default_kv_type) {
    const auto& cfg = raw.cfg;
    const int H = cfg.dense.hidden_size;
    const int head_dim = cfg.dense.head_dim;
    const bool has_dense = std::any_of(raw.layers.begin(), raw.layers.end(), [](const auto& l) { return l.kind == 0; });
    DType kv_type = ctx.kv_dtype().value_or(default_kv_type);
    if (has_dense && (head_dim < 1 || head_dim > 512))
        throw std::invalid_argument("Vulkan attention supports head dimensions from 1 to 512");
    if (has_dense && kv_type == DType::Q8_0 && head_dim % 32 != 0)
        throw std::invalid_argument("Vulkan Q8_0 KV requires a head dimension divisible by 32");

    try {
        reserve_vulkan_model(ctx, raw, kv_type);
    } catch (const std::runtime_error&) {
        if (!has_dense || ctx.kv_dtype() || kv_type != DType::F16 || head_dim % 64 != 0) throw;
        kv_type = DType::Q8_0;
        reserve_vulkan_model(ctx, raw, kv_type);
    }

    const auto upload_vk_weight = [&](const HostTensor& ht) { return ctx.upload(ht, true); };
    const bool tied = raw.lm_head_w.bytes().empty();
    tok_embeddings_ = upload_vk_weight(raw.tok_embeddings);
    if (!shader_embedding(raw.tok_embeddings.desc.dtype)) host_embeddings_ = raw.tok_embeddings;
    final_norm_ = ctx.upload(to_f32_host(raw.final_norm_w), true);
    lm_head_ = tied ? tok_embeddings_ : upload_vk_weight(raw.lm_head_w);
    const Tensor dummy_norm = ctx.upload(make_ones_f32({std::max(1, cfg.dense.head_dim)}), true);

    layer_kinds_.clear();
    layer_indices_.clear();
    dense_layers_.clear();
    gdn_layers_.clear();
    moe_layers_.clear();

    auto upload_moe = [&](const LoadedLayerRaw& lr) {
        VkSparseMoELayer L;
        L.cfg = cfg.moe;
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
        moe_layers_.push_back(std::move(L));
        return static_cast<int>(moe_layers_.size() - 1);
    };

    const auto upload_ffn = [&](auto& layer, const LoadedLayerRaw& lr) {
        layer.mlp_norm_w = ctx.upload(to_f32_host(lr.mlp_norm), true);
        if (lr.gate_up.bytes().empty()) {
            layer.gate_proj = upload_vk_weight(lr.gate);
            layer.up_proj = upload_vk_weight(lr.up);
        } else {
            layer.gate_up_proj = upload_vk_weight(lr.gate_up);
        }
        layer.down_proj = upload_vk_weight(lr.down);
    };

    for (const auto& lr : raw.layers) {
        layer_kinds_.push_back(lr.kind);
        if (lr.kind == 0) {
            VulkanDenseWeights L;
            L.cfg = cfg.dense;
            L.cfg.gated_q_attn = lr.gated_q_attn;
            L.cfg.qk_norm = !lr.q_norm.bytes().empty() && !lr.k_norm.bytes().empty();
            L.attn_norm_w = ctx.upload(to_f32_host(lr.attn_norm), true);
            L.sliding_window = lr.sliding_window;
            if (pack_vk_qkv(lr)) {
                L.qkv_proj_fused = upload_vk_weight(lr.qkv_proj.bytes().empty()
                    ? concat_rows_same_dtype({lr.q_proj, lr.k_proj, lr.v_proj}) : lr.qkv_proj);
            } else {
                L.q_proj = upload_vk_weight(lr.q_proj);
                L.k_proj = upload_vk_weight(lr.k_proj);
                L.v_proj = upload_vk_weight(lr.v_proj);
            }
            if (!lr.q_bias.bytes().empty()) L.q_bias = ctx.upload(to_f32_host(lr.q_bias), true);
            if (!lr.k_bias.bytes().empty()) L.k_bias = ctx.upload(to_f32_host(lr.k_bias), true);
            if (!lr.v_bias.bytes().empty()) L.v_bias = ctx.upload(to_f32_host(lr.v_bias), true);
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
                upload_ffn(L, lr);
            }
            hooks.allocate_dense(L.cfg, kv_type);
            layer_indices_.push_back(dense_layers_.size());
            dense_layers_.push_back(std::move(L));
        } else if (lr.kind == 1) {
            VulkanGdnWeights L;
            L.cfg = cfg.gdn;
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
                upload_ffn(L, lr);
            } else {
                L.moe_index = upload_moe(lr);
            }
            hooks.allocate_gdn(L.cfg, static_cast<int>(lr.conv_w.desc.shape[0]));
            layer_indices_.push_back(gdn_layers_.size());
            gdn_layers_.push_back(std::move(L));
        } else {
            layer_indices_.push_back(upload_moe(lr));
        }
    }
    ctx.synchronize();
    cfg_ = cfg;
}

void VulkanModelWeights::init_synthetic_vulkan(VkContext& ctx, uint64_t seed, const VulkanLoadHooks& hooks) {
    std::mt19937_64 rng(seed);
    int H = (cfg_.arch == "hybrid") ? cfg_.gdn.hidden_size
          : (cfg_.arch == "moe") ? cfg_.moe.hidden_size
          : cfg_.dense.hidden_size;

    LoadedModelRaw raw;
    raw.cfg = cfg_;
    raw.tok_embeddings = make_rand_f32({cfg_.vocab_size, H}, rng, 0.1f);
    raw.final_norm_w = make_ones_f32({H});
    raw.lm_head_w = make_rand_f32({cfg_.vocab_size, H}, rng, 0.05f);
    if (cfg_.arch == "dense" && cfg_.dense.weight_dtype == DType::Q4_K) {
        raw.tok_embeddings = synthetic_matrix(raw.tok_embeddings, DType::Q4_K);
        raw.lm_head_w = synthetic_matrix(raw.lm_head_w, DType::Q4_K);
    }
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
            lr.q_proj = synthetic_matrix(hq, wdt);
            lr.k_proj = synthetic_matrix(hk, wdt);
            lr.v_proj = synthetic_matrix(hv, wdt);
            lr.qkv_proj = synthetic_matrix(concat_rows_same_dtype({hq, hk, hv}), wdt);
            if (cfg_.dense.qk_norm) {
                lr.q_norm = make_ones_f32({cfg_.dense.head_dim});
                lr.k_norm = make_ones_f32({cfg_.dense.head_dim});
            }
            lr.o_proj = synthetic_matrix(make_rand_f32({H, q_dim}, rng, 0.04f), wdt);
            lr.mlp_norm = make_ones_f32({H});
            lr.gate_up = synthetic_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt);
            lr.down = synthetic_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt);
        } else if (lr.kind == 1) {
            DType wdt = cfg_.gdn.weight_dtype;
            int nk = cfg_.gdn.num_k_heads, nv = cfg_.gdn.num_v_heads, dk = cfg_.gdn.head_k_dim, dv = cfg_.gdn.head_v_dim;
            int conv_dim = 2 * nk * dk + nv * dv, v_dim = nv * dv, I = cfg_.gdn.intermediate_size;
            lr.attn_norm = make_ones_f32({H});
            lr.qkv_proj = synthetic_matrix(make_rand_f32({conv_dim, H}, rng, 0.04f), wdt);
            lr.z_proj = synthetic_matrix(make_rand_f32({v_dim, H}, rng, 0.04f), wdt);
            lr.a_proj = synthetic_matrix(make_rand_f32({nv, H}, rng, 0.04f), wdt);
            lr.b_proj = synthetic_matrix(make_rand_f32({nv, H}, rng, 0.04f), wdt);
            lr.conv_w = make_rand_f32({conv_dim, 4}, rng, 0.1f);
            lr.conv_bias = make_rand_f32({conv_dim}, rng, 0.01f);
            lr.A_log = HostTensor::create({nv}, DType::F32);
            for (int i = 0; i < nv; ++i) lr.A_log.data<float>()[i] = -2.0f;
            lr.dt_bias = make_rand_f32({nv}, rng, 0.02f);
            lr.gdn_norm_w = make_ones_f32({dv});
            lr.o_proj = synthetic_matrix(make_rand_f32({H, v_dim}, rng, 0.04f), wdt);
            lr.mlp_norm = make_ones_f32({H});
            lr.gate_up = synthetic_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt);
            lr.down = synthetic_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt);
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
    upload(ctx, raw, hooks, DType::F32);
}

bool VulkanModelWeights::load_gguf_vulkan(VkContext& ctx, const std::string& path, std::string* err,
                                         int max_seq_override, const VulkanLoadHooks& hooks) {
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
    upload(ctx, raw, hooks, DType::F16);
    return true;
}

bool VulkanModelWeights::load_safetensors_vulkan(VkContext& ctx, const std::string& path, std::string* err,
                                         int max_seq_override, const VulkanLoadHooks& hooks) {
    LoadedModelRaw raw;
    if (!parse_safetensors_model_raw(path, max_seq_override, raw, err)) return false;
    upload(ctx, raw, hooks, DType::F16);
    return true;
}

}  // namespace fusionrt
