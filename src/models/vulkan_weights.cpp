#include "vulkan_weights.h"
#include "model_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace fusionrt {
using namespace model_io;

namespace {

bool shader_embedding(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16 ||
           dtype == DType::Q4_0 || dtype == DType::Q8_0;
}

bool pack_vk_qkv(const LoadedLayerRaw& l) {
    return !l.gated_q_attn && !l.k_proj.bytes().empty() && !l.v_proj.bytes().empty() &&
           l.q_bias.bytes().empty() && l.k_bias.bytes().empty() && l.v_bias.bytes().empty() &&
           l.q_proj.desc.dtype == l.k_proj.desc.dtype && l.q_proj.desc.dtype == l.v_proj.desc.dtype &&
           same_weight_basis(l.q_proj, l.k_proj) && same_weight_basis(l.q_proj, l.v_proj);
}

DenseGQAConfig dense_config(const HybridModelConfig& cfg, const LoadedLayerRaw& l) {
    auto d = cfg.dense;
    d.gated_q_attn = l.gated_q_attn;
    d.qk_norm = !l.q_norm.bytes().empty();
    if (cfg.model_architecture == "gemma4") {
        d.num_attention_heads = l.query_heads;
        d.num_key_value_heads = l.kv_heads;
        d.head_dim = l.head_dim;
        d.rope_theta = l.rope_theta;
        d.partial_rotary_factor = static_cast<float>(l.rotary_dims) / l.head_dim;
        d.intermediate_size = static_cast<int>(l.down.desc.shape[1]);
    }
    return d;
}

VkDeviceSize aligned(VkDeviceSize bytes, VkDeviceSize alignment) {
    return (bytes + alignment - 1) & ~(alignment - 1);
}

DType mla_matrix_dtype(const HostTensor& source) {
    return source.desc.dtype == DType::F32 ? DType::F32 : DType::F16;
}

void reserve_vulkan_model(VkContext& ctx, const LoadedModelRaw& raw, DType kv_type) {
    const auto align = ctx.info().min_storage_offset_align;
    VkDeviceSize bytes = 0, state = 0;
    auto add = [&](VkDeviceSize n) {
        if (n) bytes = aligned(bytes, align) + std::max<VkDeviceSize>(16, aligned(n, 4));
    };
    std::unordered_set<const void*> signs;
    const auto matrix = [&](const HostTensor& t) {
        add(t.desc.nbytes());
        if (t.hadamard_signs && signs.insert(t.hadamard_signs.get()).second) add(t.hadamard_signs->size());
    };
    const auto f32 = [&](const HostTensor& t) { add(static_cast<VkDeviceSize>(t.desc.numel()) * sizeof(float)); };
    const auto state_tensor = [&](VkDeviceSize n) {
        const auto before = bytes;
        add(n);
        state += bytes - before;
    };

    add(16);
    matrix(raw.tok_embeddings);
    matrix(raw.lm_head_w);
    f32(raw.final_norm_w);
    f32(raw.rope_factors);
    matrix(raw.gemma_inputs.token_embeddings);
    matrix(raw.gemma_inputs.projection);
    f32(raw.gemma_inputs.norm);
    for (const auto& l : raw.layers) {
        f32(l.attn_norm);
        f32(l.mlp_norm);
        for (const auto* t : {&l.o_proj, &l.gate_up, &l.gate, &l.up, &l.down, &l.ple_gate, &l.ple_projection})
            matrix(*t);
        for (const auto* t : {&l.attn_post_norm, &l.ffn_post_norm, &l.ple_post_norm, &l.dense_post_norm, &l.routed_post_norm})
            f32(*t);
        if (!l.router_w.bytes().empty()) {
            for (const auto* t : {&l.router_w, &l.router_bias, &l.router_scale, &l.routed_norm,
                                 &l.expert_gate_scale, &l.expert_up_scale, &l.expert_down_scale})
                f32(*t);
            for (const auto* t : {&l.experts_gate_up, &l.experts_up, &l.experts_down,
                                  &l.shared_gate_up, &l.shared_down, &l.shared_gate})
                matrix(*t);
        }
        if (l.kind == 0) {
            const auto d = dense_config(raw.cfg, l);
            if (pack_vk_qkv(l)) {
                if (!l.qkv_proj.bytes().empty()) matrix(l.qkv_proj);
                else {
                    add(l.q_proj.desc.nbytes() + l.k_proj.desc.nbytes() + l.v_proj.desc.nbytes());
                    if (l.q_proj.hadamard_signs && signs.insert(l.q_proj.hadamard_signs.get()).second)
                        add(l.q_proj.hadamard_signs->size());
                }
            } else {
                for (const auto* t : {&l.q_proj, &l.k_proj, &l.v_proj}) matrix(*t);
            }
            for (const auto* t : {&l.q_norm, &l.k_norm, &l.q_bias, &l.k_bias, &l.v_bias}) f32(*t);
            if (d.mla.latent_rank > 0) {
                for (const auto* t : {&l.mla_q_a, &l.mla_kv_a}) matrix(*t);
                f32(l.mla_q_a_norm);
                f32(l.mla_kv_a_norm);
                const auto& m = d.mla;
                const auto& k = l.mla_kv_b.bytes().empty() ? l.mla_k_b : l.mla_kv_b;
                const auto& v = l.mla_kv_b.bytes().empty() ? l.mla_v_b : l.mla_kv_b;
                const auto plane = [&](int width, const HostTensor& t) {
                    return d.num_attention_heads * aligned(static_cast<VkDeviceSize>(m.latent_rank) * width *
                        dtype_type_size(mla_matrix_dtype(t)), align);
                };
                add(plane(d.head_dim - m.rope_dims, k));
                add(plane(m.value_dim, v));
                state_tensor(static_cast<VkDeviceSize>(d.max_seq_len) * m.latent_width() * dtype_type_size(kv_type));
            } else if (l.kv_source_layer < 0) {
                const auto plane = static_cast<VkDeviceSize>(d.max_seq_len) * d.num_key_value_heads *
                    row_nbytes(kv_type, d.head_dim);
                state_tensor(plane);
                state_tensor(plane);
            }
        } else if (l.kind == 1) {
            const auto& g = raw.cfg.gdn;
            for (const auto* t : {&l.qkv_proj, &l.z_proj, &l.a_proj, &l.b_proj}) matrix(*t);
            for (const auto* t : {&l.conv_w, &l.conv_bias, &l.A_log, &l.dt_bias, &l.gdn_norm_w}) f32(*t);
            state_tensor(static_cast<VkDeviceSize>(l.conv_w.desc.shape[0]) * std::max(1, g.conv_kernel_size - 1) * sizeof(float));
            state_tensor(static_cast<VkDeviceSize>(g.num_v_heads) * g.head_k_dim * g.head_v_dim * sizeof(float));
        }
    }
    // The final state allocation need not end at the same alignment as the counted order.
    add(align);
    ctx.reserve_model_memory(bytes - state, state);
}

std::vector<Tensor> upload_mla_matrix(VkContext& ctx, const HostTensor& fp, DType dtype) {
    const int heads = static_cast<int>(fp.desc.shape[0]);
    const auto matrix_bytes = static_cast<VkDeviceSize>(fp.desc.shape[1] * fp.desc.shape[2]) * dtype_type_size(dtype);
    const auto stride = aligned(matrix_bytes, ctx.info().min_storage_offset_align);
    const auto source = dtype == fp.desc.dtype ? fp : quantize_host_matrix(fp, dtype);
    HostTensor packed({static_cast<int64_t>(heads * stride)}, DType::U8);
    for (int head = 0; head < heads; ++head)
        std::memcpy(packed.data<uint8_t>() + head * stride, source.bytes().data() + head * matrix_bytes, matrix_bytes);
    const auto storage = ctx.upload(packed, true);
    std::vector<Tensor> views;
    views.reserve(heads);
    for (int head = 0; head < heads; ++head) {
        auto view = Tensor::make_contiguous({fp.desc.shape[1], fp.desc.shape[2]}, dtype, Backend::Vulkan);
        view.vk_buffer = storage.vk_buffer;
        view.vk_offset = storage.vk_offset + head * stride;
        views.push_back(view);
    }
    return views;
}

void upload_mla_b(VkContext& ctx, const LoadedLayerRaw& raw, VulkanDenseWeights& layer) {
    if (raw.mla_kv_b.bytes().empty()) {
        layer.mla_k_b = upload_mla_matrix(ctx, to_f32_host(raw.mla_k_b), mla_matrix_dtype(raw.mla_k_b));
        layer.mla_v_b = upload_mla_matrix(ctx, to_f32_host(raw.mla_v_b), mla_matrix_dtype(raw.mla_v_b));
        return;
    }
    const auto& d = layer.cfg;
    const int heads = d.num_attention_heads, rank = d.mla.latent_rank, dv = d.mla.value_dim;
    const int nope = d.head_dim - d.mla.rope_dims;
    const auto combined = to_f32_host(raw.mla_kv_b);
    HostTensor k({heads, rank, nope}, DType::F32), v({heads, dv, rank}, DType::F32);
    for (int h = 0; h < heads; ++h) {
        const auto* src = combined.data<float>() + static_cast<size_t>(h) * (nope + dv) * rank;
        auto* dst = k.data<float>() + static_cast<size_t>(h) * rank * nope;
        for (int r = 0; r < rank; ++r)
            for (int i = 0; i < nope; ++i)
                dst[r * nope + i] = src[i * rank + r];
        std::copy_n(src + static_cast<size_t>(nope) * rank, static_cast<size_t>(dv) * rank,
                    v.data<float>() + static_cast<size_t>(h) * dv * rank);
    }
    const auto dtype = mla_matrix_dtype(raw.mla_kv_b);
    layer.mla_k_b = upload_mla_matrix(ctx, k, dtype);
    layer.mla_v_b = upload_mla_matrix(ctx, v, dtype);
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

HostTensor gather_host_rows(const HostTensor& table, std::span<const int32_t> tokens) {
    if (table.bytes().empty()) return {};
    const auto hidden = table.desc.shape[1];
    HostTensor rows({static_cast<int64_t>(tokens.size()), hidden}, DType::F32);
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto encoded = table.slice_dim0(tokens[i], 1);
        // Gather the folded values. The executor applies the inverse basis after lookup.
        encoded.desc.hadamard_block = 0;
        const auto row = to_f32_host(encoded);
        std::copy_n(row.data<float>(), hidden, rows.data<float>() + i * hidden);
    }
    return rows;
}

}  // namespace

HostTensor VulkanModelWeights::gather_host_embeddings(std::span<const int32_t> tokens) const {
    return gather_host_rows(host_embeddings_, tokens);
}

HostTensor VulkanModelWeights::gather_host_ple(std::span<const int32_t> tokens) const {
    return gather_host_rows(host_ple_, tokens);
}

void VulkanModelWeights::upload(VkContext& ctx, const LoadedModelRaw& raw, const VulkanLoadHooks& hooks,
                                DType default_kv_type) {
    const auto& cfg = raw.cfg;
    const bool mla = cfg.dense.mla.latent_rank > 0;
    DType kv_type = ctx.kv_dtype().value_or(mla ? DType::F32 : default_kv_type);
    if (mla && (kv_type != DType::F32 && kv_type != DType::F16))
        throw std::invalid_argument("Vulkan MLA cache supports F32 or F16");
    bool can_fallback = false;
    for (const auto& l : raw.layers) {
        if (l.kind == 1) {
            const auto& g = cfg.gdn;
            if (g.conv_kernel_size < 1 || g.conv_kernel_size > 4 || g.head_k_dim < 1 || g.head_k_dim > 128 ||
                g.head_v_dim < 1 || g.head_v_dim > 256 || g.num_k_heads < 1 || g.num_v_heads < 1 ||
                g.num_v_heads % g.num_k_heads != 0)
                throw std::invalid_argument("Unsupported Vulkan GDN geometry");
        }
        if (l.kind != 0) continue;
        const auto d = dense_config(cfg, l);
        if (mla) {
            if (d.mla.latent_rank > 1024 || d.mla.rope_dims <= 0 || d.mla.rope_dims % 2 != 0 ||
                d.mla.rope_dims >= d.head_dim || d.mla.value_dim <= 0)
                throw std::invalid_argument("Unsupported Vulkan MLA geometry");
            continue;
        }
        if (d.head_dim < 1 || d.head_dim > 512)
            throw std::invalid_argument("Vulkan attention supports head dimensions from 1 to 512");
        if (kv_type == DType::Q8_0 && d.head_dim % 32 != 0)
            throw std::invalid_argument("Vulkan Q8_0 KV requires a head dimension divisible by 32");
        can_fallback = true;
    }
    can_fallback = can_fallback && std::all_of(raw.layers.begin(), raw.layers.end(), [&](const auto& l) {
        return l.kind != 0 || dense_config(cfg, l).head_dim % 32 == 0;
    });
    try {
        reserve_vulkan_model(ctx, raw, kv_type);
    } catch (const std::runtime_error&) {
        if (!can_fallback || ctx.kv_dtype() || kv_type != DType::F16) throw;
        kv_type = DType::Q8_0;
        reserve_vulkan_model(ctx, raw, kv_type);
    }

    std::unordered_map<const void*, Tensor> sign_buffers;
    hadamard_signs_.clear();
    hadamard_width_ = 0;
    const auto upload_vk_weight = [&](const HostTensor& ht) {
        if (ht.bytes().empty()) return Tensor{};
        auto t = ctx.upload(ht, true);
        t.hadamard_block = ht.desc.hadamard_block;
        t.hadamard_head_dim = ht.desc.hadamard_head_dim;
        t.hadamard_key_heads = ht.desc.hadamard_key_heads;
        if (t.hadamard_block) hadamard_width_ = std::max(hadamard_width_, static_cast<int>(t.shape[t.ndim - 1]));
        if (ht.hadamard_signs) {
            const auto& signs = *ht.hadamard_signs;
            auto [it, inserted] = sign_buffers.try_emplace(&signs);
            if (inserted) {
                HostTensor packed({static_cast<int64_t>(signs.size())}, DType::U8);
                std::memcpy(packed.storage.data(), signs.data(), signs.size());
                it->second = ctx.upload(packed, true);
            }
            hadamard_signs_.emplace(t.vk_offset, it->second);
        }
        return t;
    };
    const auto upload_f32 = [&](const HostTensor& ht) {
        return ht.bytes().empty() ? Tensor{} : ctx.upload(to_f32_host(ht), true);
    };
    dummy_ = ctx.upload(make_zeros_f32({4}), true);
    tok_embeddings_ = upload_vk_weight(raw.tok_embeddings);
    host_embeddings_ = shader_embedding(raw.tok_embeddings.desc.dtype) ? HostTensor{} : raw.tok_embeddings;
    final_norm_ = upload_f32(raw.final_norm_w);
    lm_head_ = raw.lm_head_w.bytes().empty() ? tok_embeddings_ : upload_vk_weight(raw.lm_head_w);
    const auto rope_factors = upload_f32(raw.rope_factors);
    gemma_inputs_ = {upload_vk_weight(raw.gemma_inputs.token_embeddings),
                     upload_vk_weight(raw.gemma_inputs.projection), upload_f32(raw.gemma_inputs.norm)};
    host_ple_ = shader_embedding(raw.gemma_inputs.token_embeddings.desc.dtype) ? HostTensor{} : raw.gemma_inputs.token_embeddings;

    layer_kinds_.clear();
    layer_indices_.clear();
    dense_layers_.clear();
    gdn_layers_.clear();
    moe_layers_.clear();

    auto upload_moe = [&](const LoadedLayerRaw& lr) {
        VulkanMoeWeights L;
        L.cfg = cfg.moe;
        L.cfg.has_shared_expert = lr.has_shared;
        L.cfg.num_experts = L.cfg.gpu_resident_experts = static_cast<int>(lr.router_w.desc.shape[0]);
        L.cfg.intermediate_size = static_cast<int>(lr.experts_down.desc.shape[2]);
        L.norm_w = upload_f32(!lr.router_scale.bytes().empty() ? lr.routed_norm :
                              lr.mlp_norm.bytes().empty() ? lr.attn_norm : lr.mlp_norm);
        L.router_w = upload_f32(lr.router_w);
        L.router_bias = upload_f32(lr.router_bias);
        L.router_scale = upload_f32(lr.router_scale);
        L.gate_scale = upload_f32(lr.expert_gate_scale);
        L.up_scale = upload_f32(lr.expert_up_scale);
        L.down_scale = upload_f32(lr.expert_down_scale);
        L.shared_gate_w = upload_vk_weight(lr.shared_gate);
        L.gpu_gate_up_w = upload_vk_weight(lr.experts_gate_up);
        L.gate_region = L.gpu_gate_up_w;
        if (!lr.experts_up.bytes().empty()) {
            // Both native matrices live in the same arena. Bind one byte range across them.
            const auto up = upload_vk_weight(lr.experts_up);
            const auto offset = up.vk_offset - L.gpu_gate_up_w.vk_offset;
            if (offset + up.nbytes() > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("Vulkan MoE gate/up weights exceed shader address range");
            L.up_offset = static_cast<uint32_t>(offset);
            L.up_dtype = up.dtype;
            L.gate_region = Tensor::make_contiguous({static_cast<int64_t>(offset + up.nbytes())}, DType::U8, Backend::Vulkan);
            L.gate_region.vk_buffer = L.gpu_gate_up_w.vk_buffer;
            L.gate_region.vk_offset = L.gpu_gate_up_w.vk_offset;
        }
        L.gpu_down_w = upload_vk_weight(lr.experts_down);
        if (lr.has_shared) {
            L.cfg.shared_intermediate_size = static_cast<int>(lr.shared_down.desc.shape[1]);
            L.shared_gate_up_w = upload_vk_weight(lr.shared_gate_up);
            L.shared_down_w = upload_vk_weight(lr.shared_down);
        }
        moe_layers_.push_back(std::move(L));
        return static_cast<int>(moe_layers_.size() - 1);
    };

    const auto upload_ffn = [&](auto& layer, const LoadedLayerRaw& lr) {
        layer.mlp_norm_w = upload_f32(lr.mlp_norm);
        if (lr.gate_up.bytes().empty()) {
            layer.gate_proj = upload_vk_weight(lr.gate);
            layer.up_proj = upload_vk_weight(lr.up);
        } else {
            layer.gate_up_proj = upload_vk_weight(lr.gate_up);
        }
        layer.down_proj = upload_vk_weight(lr.down);
    };

    for (const auto& lr : raw.layers) {
        if (lr.kind == 0) {
            VulkanDenseWeights L;
            L.cfg = dense_config(cfg, lr);
            L.gemma4 = cfg.model_architecture == "gemma4";
            L.sliding_window = lr.sliding_window;
            L.output_scale = lr.output_scale;
            L.attn_norm_w = upload_f32(lr.attn_norm);
            L.attn_post_norm = upload_f32(lr.attn_post_norm);
            L.ffn_post_norm = upload_f32(lr.ffn_post_norm);
            L.ple_gate = upload_vk_weight(lr.ple_gate);
            L.ple_projection = upload_vk_weight(lr.ple_projection);
            L.ple_post_norm = upload_f32(lr.ple_post_norm);
            L.dense_post_norm = upload_f32(lr.dense_post_norm);
            L.routed_post_norm = upload_f32(lr.routed_post_norm);
            if (!L.sliding_window) L.rope_factors = rope_factors;
            if (lr.kv_source_layer >= 0) {
                const auto source = static_cast<size_t>(lr.kv_source_layer);
                if (source >= layer_indices_.size() || layer_kinds_[source] != 0)
                    throw std::invalid_argument("Invalid Vulkan shared KV layer");
                L.kv_source = static_cast<int>(layer_indices_[source]);
                L.q_proj = upload_vk_weight(lr.q_proj);
            } else if (mla) {
                L.q_proj = upload_vk_weight(lr.q_proj);
                L.mla_q_a = upload_vk_weight(lr.mla_q_a);
                L.mla_q_a_norm = upload_f32(lr.mla_q_a_norm);
                L.mla_kv_a = upload_vk_weight(lr.mla_kv_a);
                L.mla_kv_a_norm = upload_f32(lr.mla_kv_a_norm);
                upload_mla_b(ctx, lr, L);
            } else if (pack_vk_qkv(lr)) {
                L.qkv_proj_fused = upload_vk_weight(lr.qkv_proj.bytes().empty()
                    ? concat_rows_same_dtype({lr.q_proj, lr.k_proj, lr.v_proj}) : lr.qkv_proj);
            } else {
                L.q_proj = upload_vk_weight(lr.q_proj);
                L.k_proj = upload_vk_weight(lr.k_proj);
                L.v_proj = upload_vk_weight(lr.v_proj);
            }
            L.q_bias = upload_f32(lr.q_bias);
            L.k_bias = upload_f32(lr.k_bias);
            L.v_bias = upload_f32(lr.v_bias);
            L.q_norm_w = upload_f32(lr.q_norm);
            L.k_norm_w = upload_f32(lr.k_norm);
            L.o_proj = upload_vk_weight(lr.o_proj);
            if (!lr.router_w.bytes().empty()) L.moe_index = upload_moe(lr);
            if (L.moe_index < 0 || L.dense_post_norm.vk_buffer) upload_ffn(L, lr);
            if (L.kv_source < 0) hooks.allocate_dense(L.cfg, kv_type);
            else hooks.share_dense(L.kv_source);
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
            L.conv_bias = upload_f32(lr.conv_bias);
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
        layer_kinds_.push_back(lr.kind);
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
            lr.conv_w = make_rand_f32({conv_dim, cfg_.gdn.conv_kernel_size}, rng, 0.1f);
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
