#include "cuda_weights.h"
#include "model_loader.h"
#include "../core/cpu_moe.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#ifdef _WIN32
#include <dxgi1_4.h>
#endif

namespace fusionrt {
using namespace model_io;

namespace {

DType cuda_matrix_dtype(const CostHardware& hw, const HostTensor& t, bool is_vocab_head) {
    const auto dtype = t.desc.dtype;
    if (t.desc.ndim != 2 || t.desc.shape[1] % 256 != 0) return dtype;
    if (dtype == DType::Q4_K) return DType::Q4_KA;
    if (dtype == DType::Q6_K && should_repack_q6_a(hw, t.desc.shape[0], t.desc.shape[1], is_vocab_head)) {
        return DType::Q6_A;
    }
    if (t.desc.shape[0] < 32) return dtype;
    if (dtype == DType::Q4_0) return DType::Q4_A;
    if (dtype == DType::Q8_0) return DType::Q8_A;
    return dtype;
}

template<class Src, class Dst>
void repack_matrix_blocks(CpuWorkers& workers, const HostTensor& src, HostTensor& dst,
                          void (*repack)(const Src*, Dst*, int64_t)) {
    const int64_t block_count = src.desc.numel() / 256;
    workers.run(block_count, [&](int64_t first, int64_t last) {
        repack(src.data<Src>() + first * (256 / dtype_block_size(src.desc.dtype)),
               dst.data<Dst>() + first, (last - first) * 256);
    });
}

HostTensor repack_matrix_for_cuda(CpuWorkers& workers, const HostTensor& src, DType dtype) {
    auto dst = HostTensor::create({src.desc.shape[0], src.desc.shape[1]}, dtype);
    dst.desc.name = src.desc.name;
    switch (dtype) {
        case DType::Q4_KA: repack_matrix_blocks(workers, src, dst, repack_q4_k_to_q4_ka); break;
        case DType::Q6_A: repack_matrix_blocks(workers, src, dst, repack_q6_k_to_q6_a); break;
        case DType::Q4_A: repack_matrix_blocks(workers, src, dst, repack_q4_0_to_q4_a); break;
        case DType::Q8_A: repack_matrix_blocks(workers, src, dst, repack_q8_0_to_q8_a); break;
        default: throw std::logic_error("Expected a repacked CUDA matrix format");
    }
    return dst;
}

Tensor upload_cuda_weight(CudaContext& ctx, CudaArena& arena, const HostTensor& src) {
    auto dst = arena.alloc_tensor({src.desc.shape.begin(), src.desc.shape.begin() + src.desc.ndim},
                                  src.desc.dtype, src.desc.name);
    ctx.upload_into(src, dst);
    dst.hadamard_block = src.desc.hadamard_block;
    dst.hadamard_head_dim = src.desc.hadamard_head_dim;
    dst.hadamard_key_heads = src.desc.hadamard_key_heads;
    if (src.hadamard_signs) {
        const auto& signs = *src.hadamard_signs;
        auto* device = static_cast<int8_t*>(arena.alloc(signs.size()));
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(device, signs.data(), signs.size(),
                                            cudaMemcpyHostToDevice, ctx.stream()));
        dst.hadamard_signs = device;
    }
    return dst;
}

Tensor upload_cuda_matrix(CudaContext& ctx, CudaArena& arena, const HostTensor& src, bool is_vocab_head = false) {
    const auto dtype = cuda_matrix_dtype(ctx.cost_hardware(), src, is_vocab_head);
    return dtype == src.desc.dtype ? upload_cuda_weight(ctx, arena, src) :
                                     upload_cuda_weight(ctx, arena, repack_matrix_for_cuda(ctx.cpu_workers(), src, dtype));
}

DType mla_weight_dtype(DType source) {
    return source == DType::F32 ? DType::F32 : DType::F16;
}

void upload_mla_weights(CudaContext& ctx, CudaArena& arena, const LoadedLayerRaw& raw, CudaDenseGQALayer& layer) {
    auto upload_b = [&](const HostTensor& source, DType dtype) {
        if (source.desc.dtype == dtype) return upload_cuda_weight(ctx, arena, source);
        return upload_cuda_weight(ctx, arena, quantize_host_matrix(to_f32_host(source), dtype));
    };
    if (!raw.mla_kv_b.bytes().empty()) {
        const auto& d = layer.cfg;
        const int heads = d.num_attention_heads, rank = d.mla.latent_rank, dv = d.mla.value_dim;
        const int nope = d.head_dim - d.mla.rope_dims;
        const auto combined = to_f32_host(raw.mla_kv_b);
        auto k = HostTensor::create({heads, rank, nope}, DType::F32);
        auto v = HostTensor::create({heads, dv, rank}, DType::F32);
        for (int h = 0; h < heads; ++h) {
            const float* src = combined.data<float>() + static_cast<size_t>(h) * (nope + dv) * rank;
            float* dst = k.data<float>() + static_cast<size_t>(h) * rank * nope;
            for (int r = 0; r < rank; ++r)
                for (int i = 0; i < nope; ++i) dst[r * nope + i] = src[i * rank + r];
            std::copy_n(src + static_cast<size_t>(nope) * rank, static_cast<size_t>(dv) * rank,
                        v.data<float>() + static_cast<size_t>(h) * dv * rank);
        }
        const auto dtype = mla_weight_dtype(raw.mla_kv_b.desc.dtype);
        layer.mla_k_b = upload_b(k, dtype);
        layer.mla_v_b = upload_b(v, dtype);
    } else {
        layer.mla_k_b = upload_b(raw.mla_k_b, mla_weight_dtype(raw.mla_k_b.desc.dtype));
        layer.mla_v_b = upload_b(raw.mla_v_b, mla_weight_dtype(raw.mla_v_b.desc.dtype));
    }
}

template <size_t Count>
void upload_projections(CudaContext& ctx, CudaArena& arena, const std::array<const HostTensor*, Count>& sources,
                        const std::array<Tensor*, Count>& weights, std::array<Tensor, Count>& packs) {
    for (size_t begin = 0; begin < Count;) {
        size_t end = begin + 1;
        while (end < Count && sources[end]->desc.dtype == sources[begin]->desc.dtype &&
               same_weight_basis(*sources[end], *sources[begin])) ++end;
        if (end == begin + 1) {
            *weights[begin] = upload_cuda_matrix(ctx, arena, *sources[begin]);
        } else {
            std::vector<HostTensor> parts;
            for (size_t i = begin; i < end; ++i) parts.push_back(*sources[i]);
            packs[begin] = upload_cuda_matrix(ctx, arena, concat_rows_same_dtype(parts));
            int64_t offset = 0;
            for (size_t i = begin; i < end; ++i) {
                const int64_t rows = sources[i]->desc.shape[0];
                *weights[i] = packs[begin].slice_dim0(offset, rows);
                offset += rows;
            }
        }
        begin = end;
    }
}

}  // namespace

void CudaModelWeights::init_synthetic(CudaContext& ctx, uint64_t seed, const CudaLoadHooks& hooks) {
    mtp_ = {};
    gemma_inputs_ = {};
    hooks.release_sessions();
    std::mt19937_64 rng(seed);
    int H = (cfg_.arch == "hybrid") ? cfg_.gdn.hidden_size
          : (cfg_.arch == "moe") ? cfg_.moe.hidden_size
          : cfg_.dense.hidden_size;

    tok_embeddings_ = ctx.upload(make_rand_f32({cfg_.vocab_size, H}, rng, 0.1f), true);
    final_norm_w_ = ctx.upload(make_ones_f32({H}), true);
    lm_head_w_ = ctx.upload(make_rand_f32({cfg_.vocab_size, H}, rng, 0.05f), true);
    hooks.allocate_workspace(ctx.weight_arena(), H, cfg_.vocab_size, cfg_.dense.max_seq_len);

    layer_kinds_.clear();
    layer_indices_.clear();
    dense_layers_.clear();
    gdn_layers_.clear();
    moe_layers_.clear();

    for (int l = 0; l < cfg_.num_layers; ++l) {
        int kind = 0;
        if (cfg_.arch == "hybrid") kind = ((l % 4) == 3) ? 0 : 1;
        else if (cfg_.arch == "moe") kind = 2;
        layer_kinds_.push_back(kind);

        if (kind == 0) {
            CudaDenseGQALayer L;
            L.cfg = cfg_.dense;
            L.cfg.hidden_size = H;
            int q_dim = L.cfg.num_attention_heads * L.cfg.head_dim;
            int kv_dim = L.cfg.num_key_value_heads * L.cfg.head_dim;
            int I = L.cfg.intermediate_size;
            DType wdt = L.cfg.weight_dtype;

            L.attn_norm_w = ctx.upload(make_ones_f32({H}), true);
            HostTensor hq = make_rand_f32({q_dim, H}, rng, 0.04f);
            HostTensor hk = make_rand_f32({kv_dim, H}, rng, 0.04f);
            HostTensor hv = make_rand_f32({kv_dim, H}, rng, 0.04f);
            const auto q = quantize_host_matrix(hq, wdt);
            const auto k = quantize_host_matrix(hk, wdt);
            const auto v = quantize_host_matrix(hv, wdt);
            upload_projections(ctx, ctx.weight_arena(), {&q, &k, &v}, {&L.q_proj, &L.k_proj, &L.v_proj}, L.projection_packs);
            if (L.cfg.qk_norm) {
                L.q_norm_w = ctx.upload(make_ones_f32({L.cfg.head_dim}), true);
                L.k_norm_w = ctx.upload(make_ones_f32({L.cfg.head_dim}), true);
            }
            L.o_proj = ctx.upload(quantize_host_matrix(make_rand_f32({H, q_dim}, rng, 0.04f), wdt), true);
            L.mlp_norm_w = ctx.upload(make_ones_f32({H}), true);
            L.gate_up_proj = ctx.upload(quantize_host_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt), true);
            L.down_proj = ctx.upload(quantize_host_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt), true);

            layer_indices_.push_back(dense_layers_.size());
            dense_layers_.push_back(std::move(L));
        } else if (kind == 1) {
            CudaGatedDeltaNetLayer L;
            L.cfg = cfg_.gdn;
            L.cfg.hidden_size = H;
            int nk = L.cfg.num_k_heads, nv = L.cfg.num_v_heads, dk = L.cfg.head_k_dim, dv = L.cfg.head_v_dim;
            int conv_dim = 2 * nk * dk + nv * dv;
            int v_dim = nv * dv;
            int I = L.cfg.intermediate_size;
            DType wdt = L.cfg.weight_dtype;

            L.attn_norm_w = ctx.upload(make_ones_f32({H}), true);
            HostTensor hqkv = make_rand_f32({conv_dim, H}, rng, 0.04f);
            HostTensor hz = make_rand_f32({v_dim, H}, rng, 0.04f);
            HostTensor ha = make_rand_f32({nv, H}, rng, 0.04f);
            HostTensor hb = make_rand_f32({nv, H}, rng, 0.04f);
            const auto qkv = quantize_host_matrix(hqkv, wdt);
            const auto z = quantize_host_matrix(hz, wdt);
            const auto a = quantize_host_matrix(ha, wdt);
            const auto b = quantize_host_matrix(hb, wdt);
            upload_projections(ctx, ctx.weight_arena(), {&qkv, &z, &a, &b}, {&L.qkv_proj, &L.z_proj, &L.a_proj, &L.b_proj},
                               L.projection_packs);
            L.conv_w = ctx.upload(make_rand_f32({conv_dim, 4}, rng, 0.1f), true);
            L.conv_bias = ctx.upload(make_rand_f32({conv_dim}, rng, 0.01f), true);
            HostTensor hA = HostTensor::create({nv}, DType::F32);
            for (int i = 0; i < nv; ++i) hA.data<float>()[i] = -2.0f;
            L.A_log = ctx.upload(hA, true);
            L.dt_bias = ctx.upload(make_rand_f32({nv}, rng, 0.02f), true);
            L.norm_w = ctx.upload(make_ones_f32({dv}), true);
            L.o_proj = ctx.upload(quantize_host_matrix(make_rand_f32({H, v_dim}, rng, 0.04f), wdt), true);
            L.mlp_norm_w = ctx.upload(make_ones_f32({H}), true);
            L.gate_up_proj = ctx.upload(quantize_host_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt), true);
            L.down_proj = ctx.upload(quantize_host_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt), true);

            layer_indices_.push_back(gdn_layers_.size());
            gdn_layers_.push_back(std::move(L));
        } else {
            CudaSparseMoELayer L;
            L.cfg = cfg_.moe;
            L.model_layer_index = l;
            L.cfg.hidden_size = H;
            int I = L.cfg.intermediate_size;
            int E = L.cfg.num_experts;
            const int configured = cfg_.gpu_experts.empty() ? L.cfg.gpu_resident_experts : cfg_.gpu_experts.at(l);
            const int gE = std::clamp(moe_gpu_experts_override_ < 0 ? configured : moe_gpu_experts_override_, 0, E);
            int cE = E - gE;
            L.cfg.gpu_resident_experts = gE;

            L.norm_w = ctx.upload(make_ones_f32({H}), true);
            L.router_w = ctx.upload(make_rand_f32({E, H}, rng, 0.1f), true);
            if (gE > 0) {
                L.gpu_gate_up_w = ctx.upload(make_rand_f32({gE, 2 * I, H}, rng, 0.04f), true);
                L.gpu_down_w = ctx.upload(make_rand_f32({gE, H, I}, rng, 0.04f), true);
            }
            if (cE > 0) {
                L.cpu_gate_up_w = make_rand_f32({cE, 2 * I, H}, rng, 0.04f);
                L.cpu_down_w = make_rand_f32({cE, H, I}, rng, 0.04f);
            }
            if (L.cfg.has_shared_expert) {
                L.shared_gate_up_w = ctx.upload(make_rand_f32({2 * L.cfg.shared_intermediate_size, H}, rng, 0.04f), true);
                L.shared_down_w = ctx.upload(make_rand_f32({H, L.cfg.shared_intermediate_size}, rng, 0.04f), true);
            }

            layer_indices_.push_back(moe_layers_.size());
            moe_layers_.push_back(std::move(L));
        }
    }
    ctx.synchronize();
    rebuild_ir();
    hooks.start_sessions(ctx.kv_dtype(), 0);
}

static bool upload_raw_model_to_cuda(CudaContext& ctx, CudaArena& arena, LoadedModelRaw& raw,
                                     HybridModelConfig& cfg_out,
                                     Tensor& tok_emb, Tensor& final_norm, Tensor& lm_head,
                                     GemmaLayerInputWeights<Tensor>& gemma_inputs,
                                     const CudaLoadHooks& hooks,
                                     std::vector<int>& layer_kinds, std::vector<size_t>& layer_indices,
                                     std::vector<CudaDenseGQALayer>& dense_layers,
                                     std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
                                     std::vector<CudaSparseMoELayer>& moe_layers) {
    cfg_out = raw.cfg;
    const Tensor rope_factors = raw.rope_factors.bytes().empty() ? Tensor{} : upload_cuda_weight(ctx, arena, raw.rope_factors);
    int H = cfg_out.dense.hidden_size;
    const bool tied = raw.lm_head_w.bytes().empty();
    tok_emb = tied ? upload_cuda_matrix(ctx, arena, raw.tok_embeddings, true) :
                    upload_cuda_weight(ctx, arena, raw.tok_embeddings);
    final_norm = upload_cuda_weight(ctx, arena, raw.final_norm_w);
    lm_head = tied ? tok_emb : upload_cuda_matrix(ctx, arena, raw.lm_head_w, true);
    gemma_inputs = {};
    if (!raw.gemma_inputs.token_embeddings.bytes().empty()) {
        gemma_inputs.token_embeddings = upload_cuda_weight(ctx, arena, raw.gemma_inputs.token_embeddings);
        gemma_inputs.projection = upload_cuda_matrix(ctx, arena, raw.gemma_inputs.projection);
        gemma_inputs.norm = upload_cuda_weight(ctx, arena, raw.gemma_inputs.norm);
    }
    cfg_out.dense.weight_dtype = lm_head.dtype;
    hooks.allocate_workspace(arena, H, cfg_out.vocab_size, cfg_out.dense.max_seq_len);

    layer_kinds.clear();
    layer_indices.clear();
    dense_layers.clear();
    gdn_layers.clear();
    moe_layers.clear();

    auto upload_moe = [&](LoadedLayerRaw& lr) {
        CudaSparseMoELayer L;
        L.cfg = cfg_out.moe;
        L.cfg.has_shared_expert = lr.has_shared;
        L.cfg.num_experts = static_cast<int>(lr.router_w.desc.shape[0]);
        const auto layer_index = static_cast<size_t>(&lr - raw.layers.data());
        L.model_layer_index = static_cast<int>(layer_index);
        const int resident = cfg_out.gpu_experts[layer_index];
        L.cfg.gpu_resident_experts = resident;
        L.cfg.intermediate_size = static_cast<int>(lr.experts_down.desc.shape[2]);
        L.geglu = !lr.router_scale.bytes().empty();
        L.norm_w = upload_cuda_weight(ctx, arena, L.geglu ? lr.routed_norm :
                                      lr.mlp_norm.bytes().empty() ? lr.attn_norm : lr.mlp_norm);
        L.router_w = upload_cuda_weight(ctx, arena, lr.router_w);
        if (L.geglu) L.router_scale = upload_cuda_weight(ctx, arena, lr.router_scale);
        if (!lr.shared_gate.bytes().empty()) L.shared_gate_w = upload_cuda_weight(ctx, arena, lr.shared_gate);
        if (!lr.expert_gate_scale.bytes().empty()) L.gate_scale = upload_cuda_weight(ctx, arena, lr.expert_gate_scale);
        if (!lr.expert_up_scale.bytes().empty()) L.up_scale = upload_cuda_weight(ctx, arena, lr.expert_up_scale);
        if (!lr.expert_down_scale.bytes().empty()) L.down_scale = upload_cuda_weight(ctx, arena, lr.expert_down_scale);
        L.cpu_gate_scale = std::move(lr.expert_gate_scale);
        L.cpu_up_scale = std::move(lr.expert_up_scale);
        if (!lr.router_bias.bytes().empty()) L.router_bias = upload_cuda_weight(ctx, arena, lr.router_bias);
        auto split = [&](HostTensor& weights, Tensor& gpu, HostTensor& cpu) {
            if (weights.bytes().empty()) return;
            const auto& shape = weights.desc.shape;
            const auto type = weights.desc.dtype;
            if (resident == 0) {
                cpu = std::move(weights);
                return;
            }
            gpu = arena.alloc_tensor({resident, shape[1], shape[2]}, type);
            const size_t hot_bytes = gpu.nbytes();
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(gpu.ptr(), weights.bytes().data(), hot_bytes,
                                                cudaMemcpyHostToDevice, ctx.stream()));
            ctx.synchronize();
            if (resident < shape[0]) cpu = weights.slice_dim0(resident, shape[0] - resident);
            weights = {};
        };
        split(lr.experts_gate_up, L.gpu_gate_up_w, L.cpu_gate_up_w);
        split(lr.experts_up, L.gpu_up_w, L.cpu_up_w);
        split(lr.experts_down, L.gpu_down_w, L.cpu_down_w);
        if (lr.has_shared) {
            L.cfg.shared_intermediate_size = static_cast<int>(lr.shared_down.desc.shape[1]);
            L.shared_gate_up_w = upload_cuda_weight(ctx, arena, lr.shared_gate_up);
            L.shared_down_w = upload_cuda_weight(ctx, arena, lr.shared_down);
        }
        moe_layers.push_back(std::move(L));
        return static_cast<int>(moe_layers.size() - 1);
    };

    for (auto& lr : raw.layers) {
        layer_kinds.push_back(lr.kind);
        if (lr.kind == 0) {
            CudaDenseGQALayer L;
            L.cfg = cfg_out.dense;
            L.cfg.gated_q_attn = lr.gated_q_attn;
            const bool gemma4 = cfg_out.model_architecture == "gemma4";
            L.kv_source_layer = lr.kv_source_layer;
            L.sliding_window = lr.sliding_window;
            L.output_scale = lr.output_scale;
            L.geglu = L.v_norm = gemma4;
            if (gemma4) {
                L.attention_scale = 1.0f;
                L.cfg.num_attention_heads = lr.query_heads;
                L.cfg.num_key_value_heads = lr.kv_heads;
                L.cfg.head_dim = lr.head_dim;
                L.cfg.rope_theta = lr.rope_theta;
                L.cfg.partial_rotary_factor = static_cast<float>(lr.rotary_dims) / lr.head_dim;
                L.cfg.intermediate_size = static_cast<int>(lr.down.desc.shape[1]);
            }
            if (L.sliding_window == 0) L.rope_factors = rope_factors;
            L.attn_norm_w = upload_cuda_weight(ctx, arena, lr.attn_norm);
            if (!lr.attn_post_norm.bytes().empty()) L.attn_post_norm_w = upload_cuda_weight(ctx, arena, lr.attn_post_norm);
            if (!lr.ffn_post_norm.bytes().empty()) L.ffn_post_norm_w = upload_cuda_weight(ctx, arena, lr.ffn_post_norm);
            if (!lr.ple_gate.bytes().empty()) {
                L.ple_gate = upload_cuda_matrix(ctx, arena, lr.ple_gate);
                L.ple_projection = upload_cuda_matrix(ctx, arena, lr.ple_projection);
                L.ple_post_norm = upload_cuda_weight(ctx, arena, lr.ple_post_norm);
            }
            if (L.cfg.mla.latent_rank > 0) {
                L.q_proj = upload_cuda_matrix(ctx, arena, lr.q_proj);
                if (L.cfg.mla.query_rank > 0) {
                    L.mla_q_a = upload_cuda_matrix(ctx, arena, lr.mla_q_a);
                    L.mla_q_a_norm = upload_cuda_weight(ctx, arena, lr.mla_q_a_norm);
                }
                L.mla_kv_a = upload_cuda_matrix(ctx, arena, lr.mla_kv_a);
                L.mla_kv_a_norm = upload_cuda_weight(ctx, arena, lr.mla_kv_a_norm);
                upload_mla_weights(ctx, arena, lr, L);
            } else if (L.kv_source_layer >= 0) {
                L.q_proj = upload_cuda_matrix(ctx, arena, lr.q_proj);
            } else if (lr.v_proj.bytes().empty()) {
                std::array<Tensor, 2> packs;
                upload_projections(ctx, arena, {&lr.q_proj, &lr.k_proj}, {&L.q_proj, &L.k_proj}, packs);
                std::move(packs.begin(), packs.end(), L.projection_packs.begin());
            } else {
                upload_projections(ctx, arena, {&lr.q_proj, &lr.k_proj, &lr.v_proj},
                                   {&L.q_proj, &L.k_proj, &L.v_proj}, L.projection_packs);
            }
            if (!lr.q_bias.bytes().empty()) L.q_bias = upload_cuda_weight(ctx, arena, lr.q_bias);
            if (!lr.k_bias.bytes().empty()) L.k_bias = upload_cuda_weight(ctx, arena, lr.k_bias);
            if (!lr.v_bias.bytes().empty()) L.v_bias = upload_cuda_weight(ctx, arena, lr.v_bias);
            if (L.cfg.qk_norm && !lr.q_norm.bytes().empty()) {
                L.q_norm_w = upload_cuda_weight(ctx, arena, lr.q_norm);
                if (!lr.k_norm.bytes().empty()) L.k_norm_w = upload_cuda_weight(ctx, arena, lr.k_norm);
            }
            L.o_proj = upload_cuda_matrix(ctx, arena, lr.o_proj);
            if (!lr.router_w.bytes().empty()) {
                L.moe_index = upload_moe(lr);
                if (!lr.dense_post_norm.bytes().empty()) {
                    L.dense_post_norm_w = upload_cuda_weight(ctx, arena, lr.dense_post_norm);
                    L.routed_post_norm_w = upload_cuda_weight(ctx, arena, lr.routed_post_norm);
                }
            }
            if (L.moe_index < 0 || L.dense_post_norm_w.data) {
                L.mlp_norm_w = upload_cuda_weight(ctx, arena, lr.mlp_norm);
                if (!lr.gate_up.bytes().empty()) {
                    L.gate_up_proj = upload_cuda_matrix(ctx, arena, lr.gate_up);
                } else {
                    L.gate_proj = upload_cuda_matrix(ctx, arena, lr.gate);
                    L.up_proj = upload_cuda_matrix(ctx, arena, lr.up);
                }
                L.down_proj = upload_cuda_matrix(ctx, arena, lr.down);
            }
            layer_indices.push_back(dense_layers.size());
            dense_layers.push_back(std::move(L));
        } else if (lr.kind == 1) {
            CudaGatedDeltaNetLayer L;
            L.cfg = cfg_out.gdn;
            L.attn_norm_w = upload_cuda_weight(ctx, arena, lr.attn_norm);
            upload_projections(ctx, arena, {&lr.qkv_proj, &lr.z_proj, &lr.a_proj, &lr.b_proj},
                               {&L.qkv_proj, &L.z_proj, &L.a_proj, &L.b_proj}, L.projection_packs);
            L.conv_w = upload_cuda_weight(ctx, arena, lr.conv_w);
            L.conv_bias = upload_cuda_weight(ctx, arena, lr.conv_bias);
            L.A_log = upload_cuda_weight(ctx, arena, lr.A_log);
            L.dt_bias = upload_cuda_weight(ctx, arena, lr.dt_bias);
            L.norm_w = upload_cuda_weight(ctx, arena, lr.gdn_norm_w);
            L.o_proj = upload_cuda_matrix(ctx, arena, lr.o_proj);
            if (!lr.router_w.bytes().empty()) {
                L.moe_index = upload_moe(lr);
            } else {
                L.mlp_norm_w = upload_cuda_weight(ctx, arena, lr.mlp_norm);
                if (!lr.gate_up.bytes().empty()) {
                    L.gate_up_proj = upload_cuda_matrix(ctx, arena, lr.gate_up);
                } else {
                    L.gate_proj = upload_cuda_matrix(ctx, arena, lr.gate);
                    L.up_proj = upload_cuda_matrix(ctx, arena, lr.up);
                }
                L.down_proj = upload_cuda_matrix(ctx, arena, lr.down);
            }
            layer_indices.push_back(gdn_layers.size());
            gdn_layers.push_back(std::move(L));
        } else {
            layer_indices.push_back(upload_moe(lr));
        }
    }
    ctx.synchronize();
    return true;
}

static size_t cuda_model_arena_bytes(const LoadedModelRaw& raw, const CostHardware& hw,
                                     std::span<const int> placements = {}) {
    size_t bytes = 0;
    auto reserve = [&](size_t n) { bytes += (n + 255) & ~size_t{255}; };
    auto weight = [&](const HostTensor& t, bool is_vocab_head = false) {
        if (t.bytes().empty()) return;
        const auto dtype = cuda_matrix_dtype(hw, t, is_vocab_head);
        reserve(t.desc.numel() / dtype_block_size(dtype) * dtype_type_size(dtype));
        if (t.hadamard_signs) reserve(t.hadamard_signs->size());
    };
    if (raw.lm_head_w.bytes().empty()) weight(raw.tok_embeddings, true);
    else {
        reserve(raw.tok_embeddings.bytes().size());
        if (raw.tok_embeddings.hadamard_signs) reserve(raw.tok_embeddings.hadamard_signs->size());
    }
    reserve(raw.gemma_inputs.token_embeddings.bytes().size());
    for (const auto* t : {&raw.final_norm_w, &raw.rope_factors, &raw.gemma_inputs.projection, &raw.gemma_inputs.norm}) weight(*t);
    weight(raw.lm_head_w, true);
    const auto& d = raw.cfg.dense;
    reserve(sizeof(float) * static_cast<size_t>(d.hidden_size));
    reserve(sizeof(float) * static_cast<size_t>(raw.cfg.vocab_size));
    reserve(sizeof(int32_t) * (1 + static_cast<size_t>(d.max_seq_len)));
    reserve(sizeof(int32_t));
    reserve(sizeof(int32_t));
    reserve(sizeof(int32_t));
    if (raw.layers.size() > static_cast<size_t>(raw.cfg.num_layers)) {
        for (const auto* tensor : {&raw.mtp_projection, &raw.mtp_embedding, &raw.mtp_enorm, &raw.mtp_hnorm,
                                   &raw.mtp_head_norm}) weight(*tensor);
        weight(raw.mtp_head, true);
        reserve(static_cast<size_t>(d.max_seq_len) * d.hidden_size * sizeof(float));
    }
    for (const auto& l : raw.layers) {
        for (const auto* t : {&l.attn_norm, &l.q_proj, &l.k_proj, &l.v_proj, &l.q_norm, &l.k_norm,
                             &l.q_bias, &l.k_bias, &l.v_bias, &l.o_proj, &l.mlp_norm, &l.gate_up,
                             &l.attn_post_norm, &l.ffn_post_norm,
                             &l.ple_gate, &l.ple_projection, &l.ple_post_norm,
                             &l.mla_q_a, &l.mla_q_a_norm, &l.mla_kv_a, &l.mla_kv_a_norm,
                             &l.gate, &l.up, &l.down, &l.z_proj, &l.a_proj, &l.b_proj,
                             &l.conv_w, &l.conv_bias, &l.A_log, &l.dt_bias, &l.gdn_norm_w,
                             &l.router_w, &l.router_bias, &l.shared_gate_up, &l.shared_down, &l.shared_gate,
                             &l.router_scale, &l.routed_norm, &l.dense_post_norm, &l.routed_post_norm,
                             &l.expert_gate_scale, &l.expert_up_scale, &l.expert_down_scale}) {
            weight(*t);
        }
        if (d.mla.latent_rank > 0) {
            const auto split_bytes = [](const HostTensor& t) {
                return t.desc.numel() * dtype_type_size(mla_weight_dtype(t.desc.dtype));
            };
            if (l.mla_kv_b.bytes().empty()) {
                reserve(split_bytes(l.mla_k_b));
                reserve(split_bytes(l.mla_v_b));
            } else {
                const size_t unit = static_cast<size_t>(d.num_attention_heads) * d.mla.latent_rank *
                    dtype_type_size(mla_weight_dtype(l.mla_kv_b.desc.dtype));
                reserve(unit * (d.head_dim - d.mla.rope_dims));
                reserve(unit * d.mla.value_dim);
            }
        }
        for (const auto* t : {&l.experts_gate_up, &l.experts_up, &l.experts_down}) {
            if (t->bytes().empty()) continue;
            const auto experts = static_cast<size_t>(t->desc.shape[0]);
            const int resident = placements.empty() ? 0 : placements[&l - raw.layers.data()];
            reserve(t->bytes().size() / experts * resident);
        }
        if (l.kind == 1) weight(l.qkv_proj);
    }
    return bytes;
}

#ifdef _WIN32
static size_t wddm_available_vram(int device, DXGI_QUERY_VIDEO_MEMORY_INFO& memory) {
    constexpr size_t unknown = static_cast<size_t>(-1);
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess || !properties.luidDeviceNodeMask)
        return unknown;
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(properties.luid));
    std::memcpy(&luid, properties.luid, sizeof(luid));
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return unknown;
    IDXGIAdapter3* adapter = nullptr;
    const auto found = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
    factory->Release();
    if (FAILED(found)) return unknown;
    const auto queried = adapter->QueryVideoMemoryInfo(std::countr_zero(properties.luidDeviceNodeMask),
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory);
    adapter->Release();
    if (FAILED(queried) || !memory.Budget) return unknown;
    return static_cast<size_t>(memory.Budget - std::min(memory.Budget, memory.CurrentUsage));
}
#endif

static size_t place_cuda_experts(CudaContext& ctx, LoadedModelRaw& raw, int override_count,
                                 size_t requested_kv_bytes, int active_tokens, int slots,
                                 DType& dtype, bool owns_arena, size_t reclaim_bytes = 0) {
    const auto& d = raw.cfg.dense;
    const auto requested_dtype = ctx.requested_kv_dtype();
    const bool mla = d.mla.latent_rank > 0;
    if (mla && requested_dtype && *requested_dtype != DType::F32)
        throw std::invalid_argument("MLA requires F32 latent KV");
    dtype = mla ? DType::F32 : ctx.kv_dtype();
    bool q8_compatible = !mla;
    size_t kv_page_bytes = 0, q8_page_bytes = 0, session_bytes = 0;
    const auto align = [](size_t bytes) { return (bytes + 255) & ~size_t{255}; };
    const size_t max_pages = 1 + (d.max_seq_len - 1) / kKvPageTokens;
    std::vector<MoeLayerCost> costs(raw.layers.size());
    int expert_layers = 0;
    for (size_t i = 0; i < raw.layers.size(); ++i) {
        const auto& layer = raw.layers[i];
        if (layer.kind == 0 && layer.kv_source_layer < 0) {
            const int heads = layer.kv_heads > 0 ? layer.kv_heads : d.num_key_value_heads;
            const int dim = layer.head_dim > 0 ? layer.head_dim : d.head_dim;
            const auto elements = kKvPageTokens * static_cast<int64_t>(heads) * dim;
            q8_compatible &= dim % 32 == 0;
            kv_page_bytes += mla ?
                kKvPageTokens * static_cast<size_t>(d.mla.latent_width()) * sizeof(float) :
                2 * row_nbytes(dtype, elements);
            if (q8_compatible) q8_page_bytes += 2 * row_nbytes(DType::Q8_0, elements);
            session_bytes += align(max_pages * sizeof(uint8_t*));
        } else if (layer.kind == 1) {
            const auto& g = raw.cfg.gdn;
            session_bytes += align(layer.conv_w.desc.shape[0] * 3 * sizeof(float)) +
                align(static_cast<size_t>(g.num_v_heads) * g.head_k_dim * g.head_v_dim * sizeof(float));
        }
        if (layer.router_w.bytes().empty()) continue;
        auto& c = costs[i];
        c.experts = static_cast<int>(layer.router_w.desc.shape[0]);
        c.top_k = raw.cfg.moe.top_k;
        c.hidden = raw.cfg.moe.hidden_size;
        c.expert_tensor_bytes = {layer.experts_gate_up.bytes().size() / c.experts, layer.experts_up.bytes().size() / c.experts,
                                 layer.experts_down.bytes().size() / c.experts};
        ++expert_layers;
    }

    if (active_tokens <= 0) active_tokens = d.max_seq_len;
    active_tokens = std::min(active_tokens, d.max_seq_len);
    auto bytes_for = [&](size_t page) {
        if (!page) return size_t{0};
        const size_t pages = requested_kv_bytes ? requested_kv_bytes / page : 1 + (active_tokens - 1) / kKvPageTokens;
        return pages * page;
    };
    if (dtype == DType::Q8_0 && !q8_compatible)
        throw std::invalid_argument("Q8_0 KV requires head dimensions divisible by 32");
    size_t kv_bytes = bytes_for(kv_page_bytes);
    session_bytes *= static_cast<size_t>(slots);
    auto log_kv = [&] {
        if (kv_page_bytes && !kv_bytes)
            throw std::invalid_argument("KV budget requires one page of " + std::to_string(kv_page_bytes) +
                                        " bytes, but configured " + std::to_string(requested_kv_bytes) + " bytes");
        std::clog << "KV placement kv_dtype_requested=" << (requested_dtype ? dtype_name(*requested_dtype) : "auto")
              << " kv_dtype=" << dtype_name(dtype)
              << " committed_bytes=" << kv_bytes
              << " page_table_and_state_bytes=" << session_bytes << " slots=" << slots
              << " active_token_capacity=" << (kv_page_bytes == 0 ? static_cast<size_t>(d.max_seq_len) :
                  std::min(static_cast<size_t>(d.max_seq_len), kv_bytes / kv_page_bytes * kKvPageTokens)) << '\n';
    };

    raw.cfg.gpu_experts.assign(raw.layers.size(), 0);
    size_t cuda_free = 0, cuda_total = 0;
    FUSIONRT_CUDA_CHECK(cudaMemGetInfo(&cuda_free, &cuda_total));
    size_t available = cuda_free + reclaim_bytes;
#ifdef _WIN32
    DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
    const auto wddm_free = wddm_available_vram(ctx.info().device_id, memory);
    if (wddm_free != static_cast<size_t>(-1))
        available = std::min(available, wddm_model_available(memory.Budget, memory.CurrentUsage, reclaim_bytes));
#endif
    const size_t scratch = ctx.scratch_arena().capacity();
    const size_t profile_bytes = static_cast<size_t>(std::min(static_cast<double>(cuda_total),
        ctx.device_profile().vram_total_mb * 1024.0 * 1024.0));
    const size_t profile_limit = profile_bytes - std::min(profile_bytes, scratch);
    available = std::min(available, profile_limit);
    const size_t fixed = cuda_model_arena_bytes(raw, ctx.cost_hardware());
    const size_t overhead = (owns_arena ? fixed : 0) + session_bytes;
    size_t arena_limit = std::numeric_limits<size_t>::max();
    if (!owns_arena) {
        const auto& arena = ctx.weight_arena();
        const size_t remaining = arena.capacity() - arena.used();
        if (fixed > remaining) throw std::runtime_error("CUDA weight arena cannot fit mandatory model weights");
        arena_limit = remaining - fixed;
    }
    size_t explicit_expert_bytes = 0;
    if (override_count >= 0) {
        for (size_t i = 0; i < costs.size(); ++i) {
            raw.cfg.gpu_experts[i] = std::min(override_count, costs[i].experts);
            explicit_expert_bytes += costs[i].gpu_bytes(raw.cfg.gpu_experts[i]);
        }
    }
    const size_t fixed_budget = overhead + (owns_arena ? explicit_expert_bytes : 0);
    const auto q8_bytes = bytes_for(q8_page_bytes);
    const auto desired_pages = static_cast<size_t>(1 + (active_tokens - 1) / kKvPageTokens);
    auto fits = [&](size_t bytes) { return fixed_budget <= available && bytes <= available - fixed_budget; };
    auto select_q8 = [&] {
        dtype = DType::Q8_0;
        kv_bytes = q8_bytes;
        kv_page_bytes = q8_page_bytes;
    };
    if (!requested_dtype && q8_compatible && q8_bytes && fits(q8_bytes) &&
        (!fits(kv_bytes) || !kv_bytes ||
         (kv_bytes / kv_page_bytes < desired_pages && q8_bytes / q8_page_bytes >= desired_pages))) {
        select_q8();
    }
    double dispatch_us = 0;
    MoePlacement choice;
    if (expert_layers && override_count < 0) {
        auto& pool = ctx.cpu_workers();
        auto empty = [](int64_t, int64_t) {};
        pool.run(pool.count(), empty);
        const auto start = std::chrono::steady_clock::now();
        constexpr int trials = 32;
        for (int i = 0; i < trials; ++i) pool.run(pool.count(), empty);
        dispatch_us = 2.0 * std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / trials;
        if (owns_arena && !requested_dtype && dtype == DType::F16 && q8_compatible && q8_bytes && kv_bytes &&
            overhead <= available) {
            auto fit = select_moe_kv_precision(costs, available - overhead, arena_limit,
                kv_bytes, q8_bytes, ctx.cost_hardware(), dispatch_us);
            if (fit.dtype == DType::Q8_0) select_q8();
            choice = std::move(fit.experts);
        }
    }
    log_kv();
    const size_t mandatory = overhead + kv_bytes;
    if (mandatory > available) throw std::runtime_error("Model shared weights, KV commitment, and session state need " +
        std::to_string(mandatory) + " VRAM bytes, but the current budget is " + std::to_string(available));
    if (!expert_layers) return kv_bytes;
    const size_t expert_budget = owns_arena ? available - mandatory : arena_limit;
    if (override_count < 0) {
        if (choice.gpu_experts.empty())
            choice = select_moe_placement(costs, expert_budget, ctx.cost_hardware(), dispatch_us);
        raw.cfg.gpu_experts = std::move(choice.gpu_experts);
    }

    size_t expert_gpu = 0, expert_cpu = 0;
    int full_layers = 0, boundary = -1, full_prefix = 0, experts_total = 0, experts_gpu = 0;
    for (size_t i = 0; i < costs.size(); ++i) {
        const auto& c = costs[i];
        if (!c.experts) continue;
        const int n = raw.cfg.gpu_experts[i];
        expert_gpu += c.gpu_bytes(n);
        expert_cpu += static_cast<size_t>(c.experts - n) *
            std::accumulate(c.expert_tensor_bytes.begin(), c.expert_tensor_bytes.end(), size_t{0});
        experts_total += c.experts;
        experts_gpu += n;
        if (n == c.experts) {
            ++full_layers;
            if (boundary < 0) ++full_prefix;
        } else if (boundary < 0) boundary = static_cast<int>(i);
    }
    if (expert_gpu > expert_budget) throw std::invalid_argument("Requested GPU expert placement needs " +
        std::to_string(expert_gpu) + " bytes, but the expert budget is " + std::to_string(expert_budget));
    raw.cfg.moe.gpu_resident_experts = *std::max_element(raw.cfg.gpu_experts.begin(), raw.cfg.gpu_experts.end());
    std::clog << "MoE VRAM snapshot cuda_free_bytes=" << cuda_free << " cuda_total_bytes=" << cuda_total;
#ifdef _WIN32
    if (wddm_free != static_cast<size_t>(-1))
        std::clog << " wddm_budget_bytes=" << memory.Budget << " wddm_current_usage_bytes=" << memory.CurrentUsage
                  << " wddm_residency_reserve_bytes=" << wddm_residency_reserve(memory.Budget);
#endif
    std::clog << " reclaimable_bytes=" << reclaim_bytes << " allocated_scratch_bytes=" << scratch
              << " effective_budget_bytes=" << available << " shared_bytes=" << fixed
              << " expert_budget_bytes=" << expert_budget << " expert_gpu_bytes=" << expert_gpu
              << " expert_cpu_bytes=" << expert_cpu << " margin_bytes=" << expert_budget - expert_gpu << '\n';
    constexpr const char* names[] = {"uniform", "prefix", "prefix_boundary"};
    std::clog << "MoE placement gpu_resident_total=" << experts_gpu << " cpu_cold_total=" << experts_total - experts_gpu
              << " full_gpu_layers=" << full_layers << " full_gpu_prefix=" << full_prefix << " boundary_layer=" << boundary
              << " threads=" << ctx.cpu_workers().count() << " layout=" << (override_count < 0 ? names[choice.choice] : "explicit");
    if (override_count < 0) {
        const auto& hw = ctx.cost_hardware();
        std::clog << " uniform_us=" << choice.candidate_us[0] << " prefix_us=" << choice.candidate_us[1]
                  << " prefix_boundary_us=" << choice.candidate_us[2] << " cpu_dispatch_us=" << dispatch_us
                  << " cpu_gbps=" << hw.cpu_q4_gemv_gbps << " gpu_gbps=" << hw.hbm_bw_gbps;
    }
    std::clog << " gpu_experts=";
    for (size_t i = 0; i < raw.cfg.gpu_experts.size(); ++i) std::clog << (i ? "," : "") << raw.cfg.gpu_experts[i];
    std::clog << '\n';
    return kv_bytes;
}

void CudaModelWeights::set_moe_gpu_experts(int count) {
    if (count < -1) throw std::invalid_argument("Expected --moe-gpu-experts >= -1");
    moe_gpu_experts_override_ = count;
}

bool CudaModelWeights::load_gguf(CudaContext& ctx, const std::string& path, std::string* err,
                            int max_seq_override, bool own_weights, CudaLoadSettings settings,
                            const CudaLoadHooks& hooks) {
    LoadedModelRaw raw;
    if (!parse_gguf_model_raw(path, max_seq_override, raw, err)) return false;
    if (load_mtp_ && raw.mtp_projection.bytes().empty()) {
        if (err) *err = "--spec-mtp requires a Qwen GGUF with an integrated NextN block";
        return false;
    }
    if (!load_mtp_) raw.layers.resize(raw.cfg.num_layers);
    auto& weights = own_weights ? gguf_weights_ : ctx.weight_arena();
    if (!own_weights) settings.releasable_session_bytes += gguf_weights_.capacity();
    const bool resize = own_weights || weights.capacity() == 0 || owned_gguf_arena_ == &ctx;
    DType kv_dtype;
    const auto kv_budget = place_cuda_experts(ctx, raw, moe_gpu_experts_override_, settings.requested_kv_bytes,
        settings.commitment_tokens, settings.slots, kv_dtype, resize,
        settings.releasable_session_bytes + (resize ? weights.capacity() : 0));
    const size_t bytes = resize ? cuda_model_arena_bytes(raw, ctx.cost_hardware(), raw.cfg.gpu_experts) : 0;
    ctx.synchronize();
    hooks.release_sessions();
    mtp_ = {};
    if (!own_weights) gguf_weights_.release();
    hooks.set_placement(kv_dtype, kv_budget);
    if (resize) {
        if (weights.capacity() != bytes) weights.init(bytes);
        else weights.reset();
        if (!own_weights) owned_gguf_arena_ = &ctx;
    }
    if (!upload_raw_model_to_cuda(ctx, weights, raw, cfg_, tok_embeddings_, final_norm_w_, lm_head_w_, gemma_inputs_,
                                    hooks,
                                    layer_kinds_, layer_indices_, dense_layers_, gdn_layers_, moe_layers_)) {
        return false;
    }
    if (load_mtp_) {
        mtp_.projection = upload_cuda_matrix(ctx, weights, raw.mtp_projection);
        mtp_.projection_scale = raw.mtp_projection_scale;
        mtp_.head_scale = raw.mtp_head_scale;
        mtp_.enorm = upload_cuda_weight(ctx, weights, raw.mtp_enorm);
        mtp_.hnorm = upload_cuda_weight(ctx, weights, raw.mtp_hnorm);
        mtp_.embedding = raw.mtp_embedding.bytes().empty() ?
            tok_embeddings_ : upload_cuda_weight(ctx, weights, raw.mtp_embedding);
        mtp_.head = raw.mtp_head.bytes().empty() ? lm_head_w_ : upload_cuda_matrix(ctx, weights, raw.mtp_head, true);
        mtp_.head_norm = raw.mtp_head_norm.bytes().empty() ?
            final_norm_w_ : upload_cuda_weight(ctx, weights, raw.mtp_head_norm);
        hooks.allocate_mtp_hidden(weights, cfg_.dense.max_seq_len, cfg_.dense.hidden_size);
        ctx.synchronize();
    }
    rebuild_ir();
    hooks.start_sessions(kv_dtype, resize && moe_gpu_experts_override_ < 0 && kv_budget ? moe_layers_.size() : 0);
    return true;
}

bool CudaModelWeights::load_safetensors(CudaContext& ctx, const std::string& path, std::string* err, int max_seq_override,
                                    CudaLoadSettings settings, const CudaLoadHooks& hooks) {
    LoadedModelRaw raw;
    if (load_mtp_) {
        if (err) *err = "--spec-mtp requires a GGUF model";
        return false;
    }
    if (!parse_safetensors_model_raw(path, max_seq_override, raw, err)) return false;
    auto& weights = ctx.weight_arena();
    const bool resize = weights.capacity() == 0 || owned_gguf_arena_ == &ctx;
    DType kv_dtype;
    const auto kv_budget = place_cuda_experts(ctx, raw, moe_gpu_experts_override_, settings.requested_kv_bytes,
        settings.commitment_tokens, settings.slots, kv_dtype, resize,
        settings.releasable_session_bytes + (resize ? weights.capacity() : 0));
    const size_t bytes = resize ? cuda_model_arena_bytes(raw, ctx.cost_hardware(), raw.cfg.gpu_experts) : 0;
    ctx.synchronize();
    hooks.release_sessions();
    mtp_ = {};
    hooks.set_placement(kv_dtype, kv_budget);
    if (resize) {
        if (weights.capacity() != bytes) weights.init(bytes);
        else weights.reset();
        owned_gguf_arena_ = &ctx;
    }
    if (!upload_raw_model_to_cuda(ctx, weights, raw, cfg_, tok_embeddings_, final_norm_w_, lm_head_w_, gemma_inputs_,
                                    hooks,
                                    layer_kinds_, layer_indices_, dense_layers_, gdn_layers_, moe_layers_)) {
        return false;
    }
    rebuild_ir();
    hooks.start_sessions(kv_dtype, resize && moe_gpu_experts_override_ < 0 && kv_budget ? moe_layers_.size() : 0);
    return true;
}

void CudaModelWeights::set_device_profile(CudaContext& ctx, const DeviceProfile& profile) {
    ctx.set_device_profile(profile);
    recommend_cuda_graph_ = profile.recommend_cuda_graph;
}

void CudaModelWeights::inspect_cuda(CudaContext& ctx, ModelMemory& result) const {
    result.persistent_bytes += gguf_weights_.capacity() ? gguf_weights_.used() : ctx.weight_arena().used();
    std::unordered_set<const void*> counted;
    auto count = [&](std::initializer_list<const Tensor*> weights) {
        for (const auto* w : weights) {
            if (w->data && counted.insert(w->ptr()).second) result.weight_bytes[w->dtype] += w->nbytes();
        }
    };
    count({&tok_embeddings_, &lm_head_w_, &final_norm_w_, &gemma_inputs_.token_embeddings,
           &gemma_inputs_.projection, &gemma_inputs_.norm});
    for (const auto& l : dense_layers_) {
        count({&l.attn_norm_w, &l.q_proj, &l.k_proj, &l.v_proj, &l.q_bias, &l.k_bias, &l.v_bias,
               &l.q_norm_w, &l.k_norm_w, &l.rope_factors, &l.o_proj, &l.mlp_norm_w,
               &l.gate_up_proj, &l.gate_proj, &l.up_proj, &l.down_proj, &l.attn_post_norm_w, &l.ffn_post_norm_w,
               &l.ple_gate, &l.ple_projection, &l.ple_post_norm,
               &l.mla_q_a, &l.mla_q_a_norm, &l.mla_kv_a, &l.mla_kv_a_norm, &l.mla_k_b, &l.mla_v_b});
    }
    for (const auto& l : gdn_layers_) {
        count({&l.attn_norm_w, &l.qkv_proj, &l.z_proj, &l.a_proj, &l.b_proj, &l.conv_w, &l.conv_bias,
               &l.A_log, &l.dt_bias, &l.norm_w, &l.o_proj, &l.mlp_norm_w, &l.gate_up_proj,
               &l.gate_proj, &l.up_proj, &l.down_proj});
    }
    for (const auto& l : moe_layers_) {
        count({&l.norm_w, &l.router_w, &l.router_bias, &l.gpu_gate_up_w, &l.gpu_up_w, &l.gpu_down_w,
               &l.shared_gate_up_w, &l.shared_down_w});
    }
}

}  // namespace fusionrt
#endif
