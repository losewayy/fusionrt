#include "model_loader.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace fusionrt {
using namespace model_io;

namespace {

float silu_host(float x) { return x / (1.0f + std::exp(-x)); }

struct ProfAccum {
    bool enabled = false;
    double x16_ms = 0, w16_ms = 0, cublas_ms = 0, attn_ms = 0, gdn_ms = 0, other_ms = 0, lm_ms = 0;
    int calls_cublas = 0;
    void reset() { x16_ms = w16_ms = cublas_ms = attn_ms = gdn_ms = other_ms = lm_ms = 0; calls_cublas = 0; }
};
static ProfAccum& prof_acc() {
    static ProfAccum p = []() {
        ProfAccum x;
        const char* e = std::getenv("FUSIONRT_PROFILE");
        x.enabled = (e && e[0] == '1');
        return x;
    }();
    return p;
}
static double time_cuda_fn(cudaStream_t s, const std::function<void()>& fn) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    fn();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return static_cast<double>(ms);
}

static Tensor cuda_scratch(CudaContext& ctx, std::initializer_list<int64_t> shape, DType dtype = DType::F32) {
    Tensor t = Tensor::make_contiguous(shape, dtype, Backend::CUDA);
    t.data = ctx.scratch_arena().alloc(t.nbytes());
    return t;
}

using DecodeGemv = decltype(&cuda_ops::gemv_q4_a_q8_1);

static DecodeGemv decode_gemv(DType dtype) {
    switch (dtype) {
        case DType::Q4_A: return cuda_ops::gemv_q4_a_q8_1;
        case DType::Q8_A: return cuda_ops::gemv_q8_a_q8_1;
        case DType::Q6_K: return cuda_ops::gemv_q6_k_q8_1;
        case DType::Q4_K: return cuda_ops::gemv_q4_k_q8_1;
        case DType::Q5_K: return cuda_ops::gemv_q5_k_q8_1;
        default: return nullptr;
    }
}

using PrefillMmq = decltype(&cuda_ops::mmq_q4_a_q8_1);

static PrefillMmq prefill_mmq(DType dtype) {
    switch (dtype) {
        case DType::Q4_A: return cuda_ops::mmq_q4_a_q8_1;
        case DType::Q8_A: return cuda_ops::mmq_q8_a_q8_1;
        case DType::Q6_K: return cuda_ops::mmq_q6_k_q8_1;
        case DType::Q4_K: return cuda_ops::mmq_q4_k_q8_1;
        case DType::Q5_K: return cuda_ops::mmq_q5_k_q8_1;
        default: return nullptr;
    }
}

static bool try_q8_1_linear(CudaContext& ctx, const float* x, const float* res_b,
                            const float* norm_w, float* res_out, const Tensor& w,
                            float* y, int M, int N, int K, float eps, bool use_fused) {
    auto gemv = M == 1 ? decode_gemv(w.dtype) : nullptr;
    auto mmq = M > 1 && M >= ctx.mmq_crossover_m() && use_fused ? prefill_mmq(w.dtype) : nullptr;
    if (!gemv && !mmq) return false;
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    void* q8 = arena.alloc(static_cast<size_t>(M) * row_nbytes(DType::Q8_1, K));
    cuda_ops::quantize_x_q8_1(x, res_b, norm_w, res_out, q8, M, K, eps, ctx.stream());
    if (gemv) gemv(q8, w.ptr(), y, N, K, ctx.stream(), ctx.preferred_gemv_geometry());
    else mmq(q8, w.ptr(), y, M, N, K, ctx.stream());
    arena.reset(mark);
    return true;
}

static bool try_cublas_prefill(CudaContext& ctx, const float* x, const float* norm_w, const Tensor& w,
                               float* y, int M, int N, int K, float eps) {
    if (M < 4 || ctx.cublas() == nullptr || w.dtype == DType::F32 || prefill_mmq(w.dtype)) return false;
    if (M < 16 && w.dtype != DType::F16 && w.dtype != DType::BF16 && w.dtype != DType::Q4_1) return false;
    size_t x_bytes = static_cast<size_t>(M) * K * sizeof(uint16_t);
    size_t w_bytes = (w.dtype == DType::F16) ? 0 : (static_cast<size_t>(N) * K * sizeof(uint16_t));
    size_t need = x_bytes + w_bytes + 4096;
    if (ctx.scratch_arena().used() + need > ctx.scratch_arena().capacity()) return false;

    size_t mark = ctx.scratch_arena().mark();
    Tensor x_f16 = ctx.scratch_arena().alloc_tensor({M, K}, DType::F16, "cb_x16");
    auto& pa = prof_acc();
    if (pa.enabled) {
        pa.x16_ms += time_cuda_fn(ctx.stream(), [&]() {
            cuda_ops::rmsnorm_or_copy_to_f16(x, norm_w, x_f16.ptr<uint16_t>(), M, K, eps, ctx.stream());
        });
    } else {
        cuda_ops::rmsnorm_or_copy_to_f16(x, norm_w, x_f16.ptr<uint16_t>(), M, K, eps, ctx.stream());
    }
    const uint16_t* w_f16_ptr = nullptr;
    if (w.dtype == DType::F16) {
        w_f16_ptr = w.ptr<uint16_t>();
    } else {
        Tensor w_f16 = ctx.scratch_arena().alloc_tensor({N, K}, DType::F16, "cb_w16");
        if (pa.enabled) {
            pa.w16_ms += time_cuda_fn(ctx.stream(), [&]() {
                cuda_ops::dequantize_to_f16(w.ptr(), w.dtype, w_f16.ptr<uint16_t>(), N, K, ctx.stream());
            });
        } else {
            cuda_ops::dequantize_to_f16(w.ptr(), w.dtype, w_f16.ptr<uint16_t>(), N, K, ctx.stream());
        }
        w_f16_ptr = w_f16.ptr<uint16_t>();
    }
    if (pa.enabled) {
        pa.calls_cublas++;
        pa.cublas_ms += time_cuda_fn(ctx.stream(), [&]() {
            cuda_ops::cublas_linear_f16_f32(ctx.cublas(), x_f16.ptr<uint16_t>(), w_f16_ptr, y, M, N, K);
        });
    } else {
        cuda_ops::cublas_linear_f16_f32(ctx.cublas(), x_f16.ptr<uint16_t>(), w_f16_ptr, y, M, N, K);
    }
    ctx.scratch_arena().reset(mark);
    return true;
}

void dispatch_linear(CudaContext& ctx, const float* x, const Tensor& w, float* y, int M, int N, int K, bool use_fused) {
    if (try_q8_1_linear(ctx, x, nullptr, nullptr, nullptr, w, y, M, N, K, 0.0f, use_fused) ||
        (use_fused && try_cublas_prefill(ctx, x, nullptr, w, y, M, N, K, 0.0f))) {
        return;
    }
    if (w.dtype == DType::F32) {
        cuda_ops::linear_f32(x, w.ptr<float>(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::F16) {
        cuda_ops::linear_w_f16_f32(x, w.ptr<uint16_t>(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::BF16) {
        cuda_ops::linear_w_bf16_f32(x, w.ptr<uint16_t>(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::Q8_0) {
        if (use_fused) {
            cuda_ops::fused_gemv_q8_0(x, w.ptr(), y, M, N, K, ctx.stream());
        } else {
            Tensor scratch_w = cuda_scratch(ctx, {N, K});
            cuda_ops::linear_q8_0_unfused(x, w.ptr(), y, M, N, K, scratch_w.ptr(), ctx.stream());
        }
    } else if (w.dtype == DType::Q4_0) {
        if (use_fused) {
            cuda_ops::fused_gemv_q4_0(x, w.ptr(), y, M, N, K, ctx.stream());
        } else {
            Tensor scratch_w = cuda_scratch(ctx, {N, K});
            cuda_ops::linear_q4_0_unfused(x, w.ptr(), y, M, N, K, scratch_w.ptr(), ctx.stream());
        }
    } else if (w.dtype == DType::Q4_1) {
        cuda_ops::fused_gemv_q4_1(x, w.ptr(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::Q4_K) {
        cuda_ops::fused_gemv_q4_k(x, w.ptr(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::Q5_K) {
        cuda_ops::fused_gemv_q5_k(x, w.ptr(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::Q6_K) {
        cuda_ops::fused_gemv_q6_k(x, w.ptr(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::Q4_A) {
        cuda_ops::fused_gemv_q4_a(x, w.ptr(), y, M, N, K, ctx.stream());
    } else if (w.dtype == DType::Q8_A) {
        cuda_ops::fused_gemv_q8_a(x, w.ptr(), y, M, N, K, ctx.stream());
    } else {
        throw std::runtime_error("Unsupported linear weight dtype");
    }
}

void dispatch_rmsnorm_linear(CudaContext& ctx, const float* x, const Tensor& norm_w, const Tensor& w,
                             float* y, int M, int N, int K, float eps, bool use_fused) {
    if (try_q8_1_linear(ctx, x, nullptr, norm_w.ptr<float>(), nullptr, w, y, M, N, K, eps, use_fused) ||
        (use_fused && try_cublas_prefill(ctx, x, norm_w.ptr<float>(), w, y, M, N, K, eps))) {
        return;
    }
    if (use_fused) {
        if (w.dtype == DType::F32) {
            cuda_ops::fused_rmsnorm_linear_f32(x, norm_w.ptr<float>(), w.ptr<float>(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::F16) {
            cuda_ops::fused_rmsnorm_linear_f16(x, norm_w.ptr<float>(), w.ptr<uint16_t>(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::BF16) {
            cuda_ops::fused_rmsnorm_linear_bf16(x, norm_w.ptr<float>(), w.ptr<uint16_t>(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q8_0) {
            cuda_ops::fused_rmsnorm_linear_q8_0(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q4_0) {
            cuda_ops::fused_rmsnorm_linear_q4_0(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q4_1) {
            cuda_ops::fused_rmsnorm_linear_q4_1(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q4_K) {
            cuda_ops::fused_rmsnorm_linear_q4_k(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q6_K) {
            cuda_ops::fused_rmsnorm_linear_q6_k(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q4_A) {
            cuda_ops::fused_rmsnorm_linear_q4_a(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        } else if (w.dtype == DType::Q8_A) {
            cuda_ops::fused_rmsnorm_linear_q8_a(x, norm_w.ptr<float>(), w.ptr(), y, M, N, K, eps, ctx.stream());
            return;
        }
    }
    Tensor x_norm = cuda_scratch(ctx, {M, K});
    cuda_ops::rmsnorm_f32(x, norm_w.ptr<float>(), x_norm.ptr<float>(), M, K, eps, ctx.stream());
    dispatch_linear(ctx, x_norm.ptr<float>(), w, y, M, N, K, use_fused);
}

void dispatch_add_rmsnorm_linear(CudaContext& ctx, const float* a, const float* b,
                                 const Tensor& norm_w, const Tensor& w,
                                 float* res_out, float* y, int M, int N, int K, float eps, bool use_fused) {
    if (try_q8_1_linear(ctx, a, b, norm_w.ptr<float>(), res_out, w, y, M, N, K, eps, use_fused)) return;
    Tensor x_norm = cuda_scratch(ctx, {M, K});
    cuda_ops::add_rmsnorm_f32(a, b, norm_w.ptr<float>(), res_out, x_norm.ptr<float>(), M, K, eps, ctx.stream());
    dispatch_linear(ctx, x_norm.ptr<float>(), w, y, M, N, K, use_fused);
}

}  // namespace

void CudaSparseMoELayer::forward_cuda(CudaContext& ctx, Tensor& x_in_out, bool use_fused) const {
    size_t mark = ctx.scratch_arena().mark();
    int H = cfg.hidden_size;
    int I = cfg.intermediate_size;
    int E = cfg.num_experts;
    int K = cfg.top_k;
    int gpu_E = cfg.gpu_resident_experts;

    Tensor x_norm = cuda_scratch(ctx, {H});
    cuda_ops::rmsnorm_f32(x_in_out.ptr<float>(), norm_w.ptr<float>(), x_norm.ptr<float>(), 1, H, cfg.rms_norm_eps, ctx.stream());

    Tensor logits = cuda_scratch(ctx, {E});
    if (use_fused) {
        cuda_ops::fused_rmsnorm_linear_f32(x_in_out.ptr<float>(), norm_w.ptr<float>(), router_w.ptr<float>(),
                                           logits.ptr<float>(), 1, E, H, cfg.rms_norm_eps, ctx.stream());
    } else {
        cuda_ops::linear_f32(x_norm.ptr<float>(), router_w.ptr<float>(), logits.ptr<float>(), 1, E, H, ctx.stream());
    }

    Tensor topk_idx = cuda_scratch(ctx, {K}, DType::I32);
    Tensor topk_w = cuda_scratch(ctx, {K});
    cuda_ops::moe_topk_router_f32(logits.ptr<float>(), topk_idx.ptr<int>(), topk_w.ptr<float>(),
                                  E, K, cfg.norm_topk_prob, ctx.stream());

    Tensor moe_out = cuda_scratch(ctx, {H});
    FUSIONRT_CUDA_CHECK(cudaMemsetAsync(moe_out.ptr(), 0, moe_out.nbytes(), ctx.stream()));

    if (gpu_E == E) {
        Tensor scratch_inter = cuda_scratch(ctx, {K, I});
        cuda_ops::moe_indexed_swiglu_gemv_f32(
            x_norm.ptr<float>(), topk_idx.ptr<int>(), gpu_gate_up_w.ptr<float>(), gpu_down_w.ptr<float>(),
            topk_w.ptr<float>(), moe_out.ptr<float>(), scratch_inter.ptr<float>(), H, I, K, ctx.stream());
    } else {
        HostTensor h_idx = ctx.download(topk_idx);
        HostTensor h_wt = ctx.download(topk_w);
        HostTensor h_x = ctx.download(x_norm);
        std::vector<float> acc_host(static_cast<size_t>(H), 0.0f);
        std::vector<int> gpu_slots;
        for (int s = 0; s < K; ++s) {
            int e = h_idx.data<int32_t>()[s];
            float wt = h_wt.data<float>()[s];
            if (e < gpu_E) {
                gpu_slots.push_back(s);
            } else {
                int cpu_e = e - gpu_E;
                const float* gu_w = cpu_gate_up_w.data<float>() + static_cast<size_t>(cpu_e) * (2 * I) * H;
                const float* d_w = cpu_down_w.data<float>() + static_cast<size_t>(cpu_e) * H * I;
                const float* xp = h_x.data<float>();
                std::vector<float> act(static_cast<size_t>(I), 0.0f);
                for (int c = 0; c < I; ++c) {
                    const float* wg = gu_w + static_cast<size_t>(c) * H;
                    const float* wu = gu_w + static_cast<size_t>(I + c) * H;
                    float g_acc = 0.0f, u_acc = 0.0f;
                    for (int i = 0; i < H; ++i) {
                        g_acc += xp[i] * wg[i];
                        u_acc += xp[i] * wu[i];
                    }
                    act[c] = silu_host(g_acc) * u_acc;
                }
                for (int h_i = 0; h_i < H; ++h_i) {
                    const float* wd = d_w + static_cast<size_t>(h_i) * I;
                    float d_acc = 0.0f;
                    for (int c = 0; c < I; ++c) d_acc += act[c] * wd[c];
                    acc_host[h_i] += d_acc * wt;
                }
            }
        }
        if (!gpu_slots.empty()) {
            int gk = static_cast<int>(gpu_slots.size());
            HostTensor h_gidx = HostTensor::create({gk}, DType::I32);
            HostTensor h_gwt = HostTensor::create({gk}, DType::F32);
            for (int i = 0; i < gk; ++i) {
                h_gidx.data<int32_t>()[i] = h_idx.data<int32_t>()[gpu_slots[i]];
                h_gwt.data<float>()[i] = h_wt.data<float>()[gpu_slots[i]];
            }
            Tensor d_gidx = ctx.upload(h_gidx, false);
            Tensor d_gwt = ctx.upload(h_gwt, false);
            Tensor scratch_inter = cuda_scratch(ctx, {gk, I});
            cuda_ops::moe_indexed_swiglu_gemv_f32(
                x_norm.ptr<float>(), d_gidx.ptr<int>(), gpu_gate_up_w.ptr<float>(), gpu_down_w.ptr<float>(),
                d_gwt.ptr<float>(), moe_out.ptr<float>(), scratch_inter.ptr<float>(), H, I, gk, ctx.stream());
        }
        HostTensor h_gpu_out = ctx.download(moe_out);
        for (int i = 0; i < H; ++i) h_gpu_out.data<float>()[i] += acc_host[i];
        ctx.upload_into(h_gpu_out, moe_out);
    }

    if (cfg.has_shared_expert) {
        Tensor sh_gu = cuda_scratch(ctx, {2 * cfg.shared_intermediate_size});
        dispatch_linear(ctx, x_norm.ptr<float>(), shared_gate_up_w, sh_gu.ptr<float>(), 1, 2 * cfg.shared_intermediate_size, H, use_fused);
        Tensor sh_act = cuda_scratch(ctx, {cfg.shared_intermediate_size});
        cuda_ops::swiglu_interleaved_f32(sh_gu.ptr<float>(), sh_act.ptr<float>(), 1, cfg.shared_intermediate_size, ctx.stream());
        Tensor sh_down = cuda_scratch(ctx, {H});
        dispatch_linear(ctx, sh_act.ptr<float>(), shared_down_w, sh_down.ptr<float>(), 1, H, cfg.shared_intermediate_size, use_fused);
        cuda_ops::add_residual_f32(moe_out.ptr<float>(), sh_down.ptr<float>(), moe_out.ptr<float>(), H, ctx.stream());
    }

    cuda_ops::add_residual_f32(x_in_out.ptr<float>(), moe_out.ptr<float>(), x_in_out.ptr<float>(), H, ctx.stream());
    ctx.scratch_arena().reset(mark);
}

void CudaSparseMoELayer::forward_prefill_cuda(CudaContext& ctx, Tensor& x_in_out, int M, bool use_fused) const {
    int H = cfg.hidden_size;
    for (int m = 0; m < M; ++m) {
        Tensor row = x_in_out.slice_dim0(m, 1).view({H});
        forward_cuda(ctx, row, use_fused);
    }
}

namespace {

HostTensor repack_matrix_for_cuda(const HostTensor& ht) {
    int64_t rows = (ht.desc.ndim >= 2) ? ht.desc.shape[0] : 0;
    int64_t cols = (ht.desc.ndim >= 2) ? ht.desc.shape[ht.desc.ndim - 1] : 0;
    if (ht.desc.dtype == DType::Q4_0 && cols >= 256 && (cols % 256) == 0 && rows >= 32) {
        HostTensor out = HostTensor::create({rows, cols}, DType::Q4_A);
        const auto* src = reinterpret_cast<const BlockQ4_0*>(ht.storage.data());
        auto* dst = reinterpret_cast<BlockQ4_A*>(out.storage.data());
        int64_t nb4 = cols / 32;
        int64_t nsa = cols / 256;
        for (int64_t r = 0; r < rows; ++r) repack_q4_0_to_q4_a(src + r * nb4, dst + r * nsa, cols);
        return out;
    }
    if (ht.desc.dtype == DType::Q8_0 && cols >= 256 && (cols % 256) == 0 && rows >= 32) {
        HostTensor out = HostTensor::create({rows, cols}, DType::Q8_A);
        const auto* src = reinterpret_cast<const BlockQ8_0*>(ht.storage.data());
        auto* dst = reinterpret_cast<BlockQ8_A*>(out.storage.data());
        int64_t nb8 = cols / 32;
        int64_t nsa = cols / 256;
        for (int64_t r = 0; r < rows; ++r) repack_q8_0_to_q8_a(src + r * nb8, dst + r * nsa, cols);
        return out;
    }
    return ht;
}

template <size_t Count>
void upload_projections(CudaContext& ctx, const std::array<const HostTensor*, Count>& sources,
                        const std::array<Tensor*, Count>& weights, std::array<Tensor, Count>& packs) {
    for (size_t begin = 0; begin < Count;) {
        size_t end = begin + 1;
        while (end < Count && sources[end]->desc.dtype == sources[begin]->desc.dtype) ++end;
        if (end == begin + 1) {
            *weights[begin] = ctx.upload(repack_matrix_for_cuda(*sources[begin]), true);
        } else {
            std::vector<HostTensor> parts;
            for (size_t i = begin; i < end; ++i) parts.push_back(*sources[i]);
            packs[begin] = ctx.upload(repack_matrix_for_cuda(concat_rows_same_dtype(parts)), true);
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

void FusionModel::init_synthetic(CudaContext& ctx, uint64_t seed) {
    cuda_session_.cached_prompt_tokens.clear();
    cuda_session_.graph_captured = false;
    std::mt19937_64 rng(seed);
    int H = (cfg_.arch == "hybrid") ? cfg_.gdn.hidden_size
          : (cfg_.arch == "moe") ? cfg_.moe.hidden_size
          : cfg_.dense.hidden_size;

    tok_embeddings_ = ctx.upload(make_rand_f32({cfg_.vocab_size, H}, rng, 0.1f), true);
    final_norm_w_ = ctx.upload(make_ones_f32({H}), true);
    lm_head_w_ = ctx.upload(make_rand_f32({cfg_.vocab_size, H}, rng, 0.05f), true);
    cuda_session_.hidden = ctx.weight_arena().alloc_tensor({H}, DType::F32, "hidden_buf");
    cuda_session_.logits = ctx.weight_arena().alloc_tensor({cfg_.vocab_size}, DType::F32, "logits_buf");
    cuda_session_.token_ids = ctx.weight_arena().alloc_tensor({1 + static_cast<int64_t>(cfg_.dense.max_seq_len)}, DType::I32, "tok_id_dev");
    cuda_session_.position = ctx.weight_arena().alloc_tensor({1}, DType::I32, "pos_dev");
    cuda_session_.kv_length = ctx.weight_arena().alloc_tensor({1}, DType::I32, "kv_len_dev");

    layer_kinds_.clear();
    layer_indices_.clear();
    dense_layers_.clear();
    gdn_layers_.clear();
    cuda_session_.kv.clear();
    cuda_session_.gdn.clear();
    moe_layers_.clear();
    cuda_session_.graph_captured = false;

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
            upload_projections(ctx, {&q, &k, &v}, {&L.q_proj, &L.k_proj, &L.v_proj}, L.projection_packs);
            if (L.cfg.qk_norm) {
                L.q_norm_w = ctx.upload(make_ones_f32({L.cfg.head_dim}), true);
                L.k_norm_w = ctx.upload(make_ones_f32({L.cfg.head_dim}), true);
            }
            L.o_proj = ctx.upload(quantize_host_matrix(make_rand_f32({H, q_dim}, rng, 0.04f), wdt), true);
            L.mlp_norm_w = ctx.upload(make_ones_f32({H}), true);
            L.gate_up_proj = ctx.upload(quantize_host_matrix(make_rand_f32({2 * I, H}, rng, 0.04f), wdt), true);
            L.down_proj = ctx.upload(quantize_host_matrix(make_rand_f32({H, I}, rng, 0.04f), wdt), true);
            cuda_session_.kv.push_back(CudaKvState{
                ctx.weight_arena().alloc_tensor({L.cfg.max_seq_len, L.cfg.num_key_value_heads, L.cfg.head_dim}, ctx.kv_dtype(), "k_cache"),
                ctx.weight_arena().alloc_tensor({L.cfg.max_seq_len, L.cfg.num_key_value_heads, L.cfg.head_dim}, ctx.kv_dtype(), "v_cache")});

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
            upload_projections(ctx, {&qkv, &z, &a, &b}, {&L.qkv_proj, &L.z_proj, &L.a_proj, &L.b_proj},
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
            cuda_session_.gdn.push_back(CudaGdnState{
                ctx.weight_arena().alloc_tensor({conv_dim, 3}, DType::F32, "conv_state"),
                ctx.weight_arena().alloc_tensor({nv, dk, dv}, DType::F32, "ssm_state")});

            layer_indices_.push_back(gdn_layers_.size());
            gdn_layers_.push_back(std::move(L));
        } else {
            CudaSparseMoELayer L;
            L.cfg = cfg_.moe;
            L.cfg.hidden_size = H;
            int I = L.cfg.intermediate_size;
            int E = L.cfg.num_experts;
            int gE = std::min(E, L.cfg.gpu_resident_experts);
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
    rebuild_cuda_ir();
}

static bool upload_raw_model_to_cuda(CudaContext& ctx, const LoadedModelRaw& raw,
                                     HybridModelConfig& cfg_out,
                                     Tensor& tok_emb, Tensor& final_norm, Tensor& lm_head,
                                     Tensor& hidden_buf, Tensor& logits_buf,
                                     Tensor& tok_id_dev, Tensor& pos_dev, Tensor& kv_len_dev,
                                     std::vector<int>& layer_kinds, std::vector<size_t>& layer_indices,
                                     std::vector<CudaDenseGQALayer>& dense_layers,
                                     std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
                                     std::vector<CudaKvState>& kv_states, std::vector<CudaGdnState>& gdn_states,
                                     std::vector<CudaSparseMoELayer>& moe_layers) {
    cfg_out = raw.cfg;
    const Tensor rope_factors = raw.rope_factors.storage.empty() ? Tensor{} : ctx.upload(raw.rope_factors, true);
    int H = cfg_out.dense.hidden_size;
    const bool tied = raw.lm_head_w.storage.empty();
    tok_emb = tied ? ctx.upload(repack_matrix_for_cuda(raw.tok_embeddings), true) :
                    ctx.upload(raw.tok_embeddings, true);
    final_norm = ctx.upload(raw.final_norm_w, true);
    lm_head = tied ? tok_emb : ctx.upload(repack_matrix_for_cuda(raw.lm_head_w), true);
    cfg_out.dense.weight_dtype = lm_head.dtype;
    hidden_buf = ctx.weight_arena().alloc_tensor({H}, DType::F32, "hidden_buf");
    logits_buf = ctx.weight_arena().alloc_tensor({cfg_out.vocab_size}, DType::F32, "logits_buf");
    tok_id_dev = ctx.weight_arena().alloc_tensor({1 + static_cast<int64_t>(cfg_out.dense.max_seq_len)}, DType::I32, "tok_id_dev");
    pos_dev = ctx.weight_arena().alloc_tensor({1}, DType::I32, "pos_dev");
    kv_len_dev = ctx.weight_arena().alloc_tensor({1}, DType::I32, "kv_len_dev");

    layer_kinds.clear();
    layer_indices.clear();
    dense_layers.clear();
    gdn_layers.clear();
    kv_states.clear();
    gdn_states.clear();
    moe_layers.clear();

    auto upload_moe = [&](const LoadedLayerRaw& lr) {
        CudaSparseMoELayer L;
        L.cfg = cfg_out.moe;
        L.cfg.has_shared_expert = lr.has_shared;
        L.cfg.num_experts = L.cfg.gpu_resident_experts = static_cast<int>(lr.router_w.desc.shape[0]);
        L.cfg.intermediate_size = static_cast<int>(lr.experts_down.desc.shape[2]);
        L.norm_w = ctx.upload(lr.mlp_norm.storage.empty() ? lr.attn_norm : lr.mlp_norm, true);
        L.router_w = ctx.upload(lr.router_w, true);
        L.gpu_gate_up_w = ctx.upload(lr.experts_gate_up, true);
        L.gpu_down_w = ctx.upload(lr.experts_down, true);
        if (lr.has_shared) {
            L.cfg.shared_intermediate_size = static_cast<int>(lr.shared_down.desc.shape[1]);
            L.shared_gate_up_w = ctx.upload(lr.shared_gate_up, true);
            L.shared_down_w = ctx.upload(lr.shared_down, true);
        }
        moe_layers.push_back(std::move(L));
        return static_cast<int>(moe_layers.size() - 1);
    };

    for (const auto& lr : raw.layers) {
        layer_kinds.push_back(lr.kind);
        if (lr.kind == 0) {
            CudaDenseGQALayer L;
            L.cfg = cfg_out.dense;
            L.cfg.gated_q_attn = lr.gated_q_attn;
            if (!raw.rope_factors.storage.empty()) L.rope_factors = rope_factors;
            L.attn_norm_w = ctx.upload(lr.attn_norm, true);
            upload_projections(ctx, {&lr.q_proj, &lr.k_proj, &lr.v_proj},
                               {&L.q_proj, &L.k_proj, &L.v_proj}, L.projection_packs);
            if (!lr.q_bias.storage.empty()) L.q_bias = ctx.upload(lr.q_bias, true);
            if (!lr.k_bias.storage.empty()) L.k_bias = ctx.upload(lr.k_bias, true);
            if (!lr.v_bias.storage.empty()) L.v_bias = ctx.upload(lr.v_bias, true);
            if (L.cfg.qk_norm && !lr.q_norm.storage.empty()) {
                L.q_norm_w = ctx.upload(lr.q_norm, true);
                L.k_norm_w = ctx.upload(lr.k_norm, true);
            }
            L.o_proj = ctx.upload(repack_matrix_for_cuda(lr.o_proj), true);
            if (!lr.router_w.storage.empty()) {
                L.moe_index = upload_moe(lr);
            } else {
                L.mlp_norm_w = ctx.upload(lr.mlp_norm, true);
                if (!lr.gate_up.storage.empty()) {
                    L.gate_up_proj = ctx.upload(repack_matrix_for_cuda(lr.gate_up), true);
                } else {
                    L.gate_proj = ctx.upload(repack_matrix_for_cuda(lr.gate), true);
                    L.up_proj = ctx.upload(repack_matrix_for_cuda(lr.up), true);
                }
                L.down_proj = ctx.upload(repack_matrix_for_cuda(lr.down), true);
            }
            kv_states.push_back(CudaKvState{
                ctx.weight_arena().alloc_tensor({L.cfg.max_seq_len, L.cfg.num_key_value_heads, L.cfg.head_dim}, ctx.kv_dtype(), "k_cache"),
                ctx.weight_arena().alloc_tensor({L.cfg.max_seq_len, L.cfg.num_key_value_heads, L.cfg.head_dim}, ctx.kv_dtype(), "v_cache")});
            layer_indices.push_back(dense_layers.size());
            dense_layers.push_back(std::move(L));
        } else if (lr.kind == 1) {
            CudaGatedDeltaNetLayer L;
            L.cfg = cfg_out.gdn;
            L.attn_norm_w = ctx.upload(lr.attn_norm, true);
            upload_projections(ctx, {&lr.qkv_proj, &lr.z_proj, &lr.a_proj, &lr.b_proj},
                               {&L.qkv_proj, &L.z_proj, &L.a_proj, &L.b_proj}, L.projection_packs);
            L.conv_w = ctx.upload(lr.conv_w, true);
            L.conv_bias = ctx.upload(lr.conv_bias, true);
            L.A_log = ctx.upload(lr.A_log, true);
            L.dt_bias = ctx.upload(lr.dt_bias, true);
            L.norm_w = ctx.upload(lr.gdn_norm_w, true);
            L.o_proj = ctx.upload(repack_matrix_for_cuda(lr.o_proj), true);
            L.mlp_norm_w = ctx.upload(lr.mlp_norm, true);
            if (!lr.gate_up.storage.empty()) {
                L.gate_up_proj = ctx.upload(repack_matrix_for_cuda(lr.gate_up), true);
            } else {
                L.gate_proj = ctx.upload(repack_matrix_for_cuda(lr.gate), true);
                L.up_proj = ctx.upload(repack_matrix_for_cuda(lr.up), true);
            }
            L.down_proj = ctx.upload(repack_matrix_for_cuda(lr.down), true);
            int conv_dim = static_cast<int>(lr.conv_w.desc.shape[0]);
            gdn_states.push_back(CudaGdnState{
                ctx.weight_arena().alloc_tensor({conv_dim, 3}, DType::F32, "conv_state"),
                ctx.weight_arena().alloc_tensor({L.cfg.num_v_heads, L.cfg.head_k_dim, L.cfg.head_v_dim}, DType::F32, "ssm_state")});
            layer_indices.push_back(gdn_layers.size());
            gdn_layers.push_back(std::move(L));
        } else {
            layer_indices.push_back(upload_moe(lr));
        }
    }
    ctx.synchronize();
    return true;
}

static size_t gguf_cuda_arena_bytes(const LoadedModelRaw& raw, DType kv_dtype) {
    size_t bytes = 0;
    auto reserve = [&](size_t n) { bytes += (n + 255) & ~size_t{255}; };
    auto weight = [&](const HostTensor& t) { reserve(t.storage.size()); };
    for (const auto* t : {&raw.tok_embeddings, &raw.final_norm_w, &raw.lm_head_w, &raw.rope_factors}) weight(*t);
    const auto& d = raw.cfg.dense;
    const auto& g = raw.cfg.gdn;
    reserve(sizeof(float) * static_cast<size_t>(d.hidden_size));
    reserve(sizeof(float) * static_cast<size_t>(raw.cfg.vocab_size));
    reserve(sizeof(int32_t) * (1 + static_cast<size_t>(d.max_seq_len)));
    reserve(sizeof(int32_t));
    reserve(sizeof(int32_t));
    for (const auto& l : raw.layers) {
        for (const auto* t : {&l.attn_norm, &l.q_proj, &l.k_proj, &l.v_proj, &l.q_norm, &l.k_norm,
                             &l.q_bias, &l.k_bias, &l.v_bias, &l.o_proj, &l.mlp_norm, &l.gate_up,
                             &l.gate, &l.up, &l.down, &l.z_proj, &l.a_proj, &l.b_proj,
                             &l.conv_w, &l.conv_bias, &l.A_log, &l.dt_bias, &l.gdn_norm_w,
                             &l.router_w, &l.experts_gate_up, &l.experts_down, &l.shared_gate_up, &l.shared_down}) {
            weight(*t);
        }
        if (l.kind == 0) {
            const size_t kv = dtype_type_size(kv_dtype) * static_cast<size_t>(d.max_seq_len) * d.num_key_value_heads * d.head_dim;
            reserve(kv);
            reserve(kv);
        } else if (l.kind == 1) {
            weight(l.qkv_proj);
            reserve(sizeof(float) * static_cast<size_t>(l.conv_w.desc.shape[0]) * 3);
            reserve(sizeof(float) * static_cast<size_t>(g.num_v_heads) * g.head_k_dim * g.head_v_dim);
        }
    }
    return bytes;
}

bool FusionModel::load_gguf(CudaContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    LoadedModelRaw raw;
    if (!parse_gguf_model_raw(path, max_seq_override, raw, err)) return false;
    if (ctx.weight_arena().capacity() == 0) {
        ctx.weight_arena().init(gguf_cuda_arena_bytes(raw, ctx.kv_dtype()));
    }
    cuda_session_.graph_captured = false;
    cuda_session_.cached_prompt_tokens.clear();
    if (!upload_raw_model_to_cuda(ctx, raw, cfg_, tok_embeddings_, final_norm_w_, lm_head_w_,
                                    cuda_session_.hidden, cuda_session_.logits, cuda_session_.token_ids, cuda_session_.position, cuda_session_.kv_length,
                                    layer_kinds_, layer_indices_, dense_layers_, gdn_layers_,
                                    cuda_session_.kv, cuda_session_.gdn, moe_layers_)) {
        return false;
    }
    rebuild_cuda_ir();
    return true;
}

bool FusionModel::load_safetensors(CudaContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    LoadedModelRaw raw;
    if (!parse_safetensors_model_raw(path, max_seq_override, raw, err)) return false;
    cuda_session_.graph_captured = false;
    cuda_session_.cached_prompt_tokens.clear();
    if (!upload_raw_model_to_cuda(ctx, raw, cfg_, tok_embeddings_, final_norm_w_, lm_head_w_,
                                    cuda_session_.hidden, cuda_session_.logits, cuda_session_.token_ids, cuda_session_.position, cuda_session_.kv_length,
                                    layer_kinds_, layer_indices_, dense_layers_, gdn_layers_,
                                    cuda_session_.kv, cuda_session_.gdn, moe_layers_)) {
        return false;
    }
    rebuild_cuda_ir();
    return true;
}

void FusionModel::reset_kv_and_state(CudaContext& ctx) {
    cuda_session_.cached_prompt_tokens.clear();
    for (auto& state : cuda_session_.kv) {
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(state.keys.ptr(), 0, state.keys.nbytes(), ctx.stream()));
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(state.values.ptr(), 0, state.values.nbytes(), ctx.stream()));
    }
    for (auto& state : cuda_session_.gdn) {
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(state.convolution.ptr(), 0, state.convolution.nbytes(), ctx.stream()));
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(state.recurrence.ptr(), 0, state.recurrence.nbytes(), ctx.stream()));
    }
    ctx.synchronize();
}

struct CudaIRWorkspace {
    std::vector<int> last_use;
    std::vector<Tensor> bufs;

    explicit CudaIRWorkspace(const MiniGraph& graph)
        : last_use(graph.values.size(), -1), bufs(graph.values.size()) {
        int num_nodes = static_cast<int>(graph.nodes.size());
        for (int i = 0; i < num_nodes; ++i) {
            if (!graph.nodes[i].alive) continue;
            for (int vid : graph.nodes[i].inputs) last_use[vid] = i;
        }
        last_use[graph.output_value] = num_nodes;
    }
};

namespace {

class CudaOpTimer {
public:
    explicit CudaOpTimer(size_t count) : events_(count) {
        try {
            for (auto& pair : events_) {
                FUSIONRT_CUDA_CHECK(cudaEventCreate(&pair[0]));
                FUSIONRT_CUDA_CHECK(cudaEventCreate(&pair[1]));
            }
        } catch (...) {
            release();
            throw;
        }
    }
    ~CudaOpTimer() { release(); }
    cudaEvent_t event(int node, int endpoint) const { return events_[node][endpoint]; }
private:
    void release() {
        for (auto& pair : events_) for (auto e : pair) if (e) cudaEventDestroy(e);
    }
    std::vector<std::array<cudaEvent_t, 2>> events_;
};

// Count useful arithmetic and logical device-buffer traffic, including packed
// weights and intermediate buffers. Cache-line traffic depends on the GPU.
static OpTiming operator_work(const IRNode& node, int rows, int pos, const CudaContext& ctx,
                               bool use_fused, bool packed_input, bool packed_output,
                               const CudaDenseGQALayer* dense, const CudaSparseMoELayer* moe,
                               size_t kv_element_bytes) {
    const int64_t m = rows, f32 = sizeof(float);
    const int64_t n = node.op == OpKind::GQA_ATTN_BLOCK ? node.attention.query_heads :
                      node.op == OpKind::GDN_RECURRENT_BLOCK ? node.recurrent.key_heads : node.output_cols;
    const int64_t k = node.op == OpKind::GQA_ATTN_BLOCK ? node.attention.kv_heads :
                      node.op == OpKind::GDN_RECURRENT_BLOCK ? node.recurrent.value_heads : node.input_cols;
    OpTiming work;
    work.name = node.tag;
    work.kind = op_kind_name(node.op);
    work.calls = 1;
    auto norm = [&](int64_t width, int64_t count, bool output) {
        work.bytes += f32 * width * (1 + count * (output ? 2 : 1));
        work.flops += count * (4 * width + 2);
    };
    auto linear = [&](const Tensor& w, bool reads_activation) {
        const bool q8 = supports_q8_1_decode(w.dtype) &&
                        (m == 1 || (use_fused && m >= ctx.mmq_crossover_m()));
        work.bytes += w.nbytes() + f32 * m * (n + (reads_activation && !packed_input ? k : 0));
        work.flops += 2 * m * n * k;
        if (q8) work.bytes += (packed_input ? 1 : 2) * m * row_nbytes(DType::Q8_1, k);
        if (q8 && m > 1) work.kind += "_MMQ_INT8";
        else if (q8) work.kind += "_GEMV_INT8";
        else if (m == 1) work.kind += "_GEMV";
        else if (use_fused && m >= 4 && w.dtype != DType::F32 && !prefill_mmq(w.dtype) &&
                 (m >= 16 || w.dtype == DType::F16 || w.dtype == DType::BF16 || w.dtype == DType::Q4_1)) {
            work.bytes += 2 * sizeof(uint16_t) * m * k;
            if (w.dtype != DType::F16) work.bytes += 2 * sizeof(uint16_t) * n * k;
            work.kind += "_GEMM_FP16";
        } else work.kind += "_LINEAR";
    };
    switch (node.op) {
        case OpKind::FUSED_ADD_RMSNORM:
            work.flops += m * k;
            work.bytes += 2 * f32 * m * k;
            [[fallthrough]];
        case OpKind::RMSNORM:
            norm(k, m, !packed_output);
            if (packed_output) {
                work.bytes += m * row_nbytes(DType::Q8_1, k);
                work.kind += "_Q8_1";
            }
            break;
        case OpKind::SHARED_Q8_PROJECTIONS: {
            const auto cols = node.secondary_weight->shape[0];
            work.bytes += node.secondary_weight->nbytes() + f32 * m * cols;
            if (packed_input) work.bytes += m * row_nbytes(DType::Q8_1, k);
            work.flops += 2 * m * cols * k;
            [[fallthrough]];
        }
        case OpKind::LINEAR:
            linear(*node.weight, true);
            break;
        case OpKind::FUSED_ADD_RMSNORM_LINEAR:
            work.flops += m * k;
            work.bytes += f32 * m * k * (node.residual_out >= 0 && !node.output_logits ? 2 : 1);
            [[fallthrough]];
        case OpKind::FUSED_RMSNORM_LINEAR:
            norm(k, m, false);
            linear(*node.weight, false);
            break;
        case OpKind::ADD_RESIDUAL:
            work.bytes = 3 * f32 * m * n;
            work.flops = m * n;
            break;
        case OpKind::SWIGLU_INTERLEAVED:
            work.bytes = 3 * f32 * m * n;
            work.flops = 5 * m * n;
            break;
        case OpKind::SWIGLU_Q8_LINEAR:
            work.bytes = 2 * f32 * m * k;
            work.flops = 5 * m * k;
            linear(*node.weight, false);
            break;
        case OpKind::GQA_ATTN_BLOCK: {
            const int64_t hd = node.attention.head_dim;
            const int64_t pairs = m * pos + m * (m + 1) / 2;
            work.flops = n * pairs * (4 * hd + 4) + 3 * m * (n + k) * node.attention.rotary_dims;
            work.bytes = f32 * hd * 2 * m * (n + k) + kv_element_bytes * hd * 2 * (m + pairs) * k;
            if (node.attention.query_norm) norm(hd, m * n, false);
            if (node.attention.key_norm) norm(hd, m * k, false);
            if (node.attention.gated) {
                work.flops += 5 * m * n * hd;
                work.bytes += 4 * f32 * m * n * hd;
            }
            for (const Tensor* bias : {&dense->q_bias, &dense->k_bias, &dense->v_bias}) {
                if (!bias->data) continue;
                work.flops += m * bias->numel();
                work.bytes += (2 * m + 1) * bias->nbytes();
            }
            break;
        }
        case OpKind::GDN_RECURRENT_BLOCK: {
            const int64_t dk = node.recurrent.key_dim, dv = node.recurrent.value_dim;
            const int64_t conv = 2 * n * dk + k * dv;
            const int64_t state = k * dk * dv;
            work.bytes = f32 * (m * (conv + 2 * k * dv + 2 * k) + 10 * conv + 2 * state + 2 * k + dv);
            work.flops = m * (12 * conv + 6 * n * dk + k * (10 * dv + 12));
            if (m < 4 || (dk != 64 && dk != 128) || dv % 16 != 0 || k % n != 0) {
                work.flops += 7 * m * state;
            } else {
                for (int64_t start = 0; start < m; start += 32) {
                    const int64_t count = std::min<int64_t>(32, m - start);
                    const int64_t triangle = count * (count + 1) / 2;
                    const bool has_state = pos > 0 || start > 0;
                    work.flops += k * (4 * triangle * (dk + dv) + count * (count * count - 1) / 3)
                                  + (2 + (has_state ? 4 : 0)) * count * state + 2 * state;
                    work.bytes += f32 * k * (2 * triangle + 2 * count) * (1 + dv / 16);
                }
            }
            break;
        }
        case OpKind::MOE_FFN_BLOCK: {
            const int64_t experts = moe->cfg.num_experts, selected = moe->cfg.top_k;
            const int64_t inter = moe->cfg.intermediate_size;
            norm(n, m, true);
            work.flops += m * (2 * n * experts + selected * (6 * n * inter + 5 * inter + 2 * n));
            work.bytes += m * (moe->router_w.nbytes() + f32 * (selected * 3 * n * inter + experts + 2 * n));
            if (moe->shared_down_w.data) {
                work.bytes += m * (moe->shared_gate_up_w.nbytes() + moe->shared_down_w.nbytes());
                work.flops += m * (6 * n + 5) * moe->cfg.shared_intermediate_size;
            }
            break;
        }
    }
    return work;
}

}  // namespace

template<bool Observe = false>
static void execute_ir_graph_cuda(
    const MiniGraph& g, CudaIRWorkspace& ws, CudaContext& ctx,
    const Tensor& x_input, Tensor& logits_buf,
    int M, int pos_host, int start_pos,
    const int* pos_dev, const int* kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvState>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    bool use_fused, OpTrace* op_trace = nullptr, CudaOpTimer* timer = nullptr,
    LayerTrace* layer_trace = nullptr, bool compute_logits = true) {
    size_t mark = ctx.scratch_arena().mark();
    int H = g.hidden_size;
    int num_nodes = static_cast<int>(g.nodes.size());
    const auto& last_use = ws.last_use;

    constexpr int kCarrySlots = 8;
    std::array<Tensor, kCarrySlots> carry_bufs;
    std::array<int, kCarrySlots> carry_busy_until;
    carry_busy_until.fill(-1);
    for (int s = 0; s < kCarrySlots; ++s) {
        carry_bufs[s] = cuda_scratch(ctx, {M, H});
    }
    size_t layer_reset_mark = ctx.scratch_arena().mark();
    int non_carry_busy_until = -1;

    auto& bufs = ws.bufs;
    std::fill(bufs.begin(), bufs.end(), Tensor{});
    bufs[g.input_value] = Tensor::make_contiguous({M, H}, DType::F32, Backend::CUDA, x_input.ptr());
    auto& pa = prof_acc();

    for (int ni = 0; ni < num_nodes; ++ni) {
        const IRNode& node = g.nodes[ni];
        if (!node.alive) continue;
        if (ni > non_carry_busy_until) {
            ctx.scratch_arena().reset(layer_reset_mark);
            non_carry_busy_until = -1;
        }
        const bool is_lm_head = node.output_logits;
        if (is_lm_head && !compute_logits && node.residual_out < 0) continue;
        const int rows = is_lm_head && compute_logits ? 1 : M;

        auto get_out_buf = [&](int vid, int cols, int r_cnt) -> Tensor& {
            if (vid == g.output_value && r_cnt == 1) {
                bufs[vid] = Tensor::make_contiguous({cols}, DType::F32, Backend::CUDA, logits_buf.ptr());
                return bufs[vid];
            }
            if (bufs[vid].data != nullptr) return bufs[vid];
            if (cols == H && r_cnt == M && !is_lm_head) {
                for (int s = 0; s < kCarrySlots; ++s) {
                    if (carry_busy_until[s] < ni) {
                        carry_busy_until[s] = std::max(ni, last_use[vid]);
                        bufs[vid] = carry_bufs[s];
                        return bufs[vid];
                    }
                }
            }
            bufs[vid] = cuda_scratch(ctx, {r_cnt, cols});
            non_carry_busy_until = std::max(non_carry_busy_until, last_use[vid]);
            return bufs[vid];
        };

        auto row_ptr = [&](int vid, bool tail_only) -> const float* {
            const Tensor& t = bufs[vid];
            if (!tail_only || M <= 1) return t.ptr<float>();
            return t.ptr<float>() + static_cast<size_t>(M - 1) * static_cast<size_t>(g.hidden_size);
        };

        if constexpr (Observe) {
            if (timer) FUSIONRT_CUDA_CHECK(cudaEventRecord(timer->event(ni, 0), ctx.stream()));
        }
        switch (node.op) {
            case OpKind::RMSNORM:
            case OpKind::FUSED_ADD_RMSNORM: {
                const bool add = node.op == OpKind::FUSED_ADD_RMSNORM;
                const float* a = bufs[node.inputs[0]].ptr<float>();
                const float* b = add ? bufs[node.inputs[1]].ptr<float>() : nullptr;
                float* res = add ? get_out_buf(node.residual_out, node.input_cols, rows).ptr<float>() : nullptr;
                const float* weight = node.norm_weight->ptr<float>();
                if (node.share_quantized_activation && (rows == 1 || rows >= ctx.mmq_crossover_m())) {
                    bufs[node.out] = ctx.scratch_arena().alloc_tensor({rows, node.input_cols}, DType::Q8_1);
                    non_carry_busy_until = std::max(non_carry_busy_until, last_use[node.out]);
                    cuda_ops::quantize_x_q8_1(a, b, weight, res, bufs[node.out].ptr(), rows,
                                              node.input_cols, node.norm_eps, ctx.stream());
                } else {
                    float* out = get_out_buf(node.out, node.output_cols, rows).ptr<float>();
                    if (add) cuda_ops::add_rmsnorm_f32(a, b, weight, res, out, rows, node.input_cols,
                                                       node.norm_eps, ctx.stream());
                    else cuda_ops::rmsnorm_f32(a, weight, out, rows, node.input_cols, node.norm_eps, ctx.stream());
                }
                break;
            }
            case OpKind::SHARED_Q8_PROJECTIONS:
            case OpKind::LINEAR: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* in_p = row_ptr(node.inputs[0], is_lm_head);
                auto fn = [&]() {
                    const bool packed = bufs[node.inputs[0]].dtype == DType::Q8_1;
                    auto project = [&](const Tensor& w, float* out, int cols) {
                        if (!packed) {
                            dispatch_linear(ctx, in_p, w, out, rows, cols, node.input_cols, use_fused);
                        } else if (rows == 1) {
                            decode_gemv(w.dtype)(in_p, w.ptr(), out, cols, node.input_cols,
                                                  ctx.stream(), ctx.preferred_gemv_geometry());
                        } else {
                            prefill_mmq(w.dtype)(in_p, w.ptr(), out, rows, cols, node.input_cols, ctx.stream());
                        }
                    };
                    const Tensor& w = *node.weight;
                    if (node.secondary_weight) {
                        const Tensor& other = *node.secondary_weight;
                        const int cols = static_cast<int>(other.shape[0]);
                        auto* extra = get_out_buf(node.secondary_out, cols, rows).ptr<float>();
                        if (packed && cuda_ops::project_q4_a_and_q8_a(in_p, w.ptr(), other.ptr(), out_t.ptr<float>(),
                                extra, rows, node.output_cols, cols, node.input_cols, ctx.stream(),
                                ctx.preferred_gemv_geometry())) return;
                        project(other, extra, cols);
                    }
                    project(w, out_t.ptr<float>(), node.output_cols);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::FUSED_RMSNORM_LINEAR: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* in_p = row_ptr(node.inputs[0], is_lm_head);
                auto fn = [&]() {
                    dispatch_rmsnorm_linear(ctx, in_p, *node.norm_weight, *node.weight, out_t.ptr<float>(), rows, node.output_cols, node.input_cols, node.norm_eps, use_fused);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::ADD_RESIDUAL: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                cuda_ops::add_residual_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                                           out_t.ptr<float>(), rows * node.output_cols, ctx.stream());
                break;
            }
            case OpKind::FUSED_ADD_RMSNORM_LINEAR: {
                if (is_lm_head && !compute_logits) {
                    auto& residual = get_out_buf(node.residual_out, node.input_cols, M);
                    cuda_ops::add_residual_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                                              residual.ptr<float>(), M * node.input_cols, ctx.stream());
                    break;
                }
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                float* res_upd_p = nullptr;
                if (node.residual_out >= 0 && !is_lm_head) {
                    Tensor& r_t = get_out_buf(node.residual_out, node.input_cols, rows);
                    res_upd_p = r_t.ptr<float>();
                }
                const float* a_p = row_ptr(node.inputs[0], is_lm_head);
                const float* b_p = row_ptr(node.inputs[1], is_lm_head);
                auto fn = [&]() {
                    dispatch_add_rmsnorm_linear(ctx, a_p, b_p, *node.norm_weight, *node.weight, res_upd_p,
                                                out_t.ptr<float>(), rows, node.output_cols, node.input_cols, node.norm_eps, use_fused);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::GQA_ATTN_BLOCK: {
                auto run_gqa = [&]() {
                    const auto& L = dense_layers[static_cast<size_t>(node.layer_index)];
                    const auto& cfg = node.attention;
                    const auto& state = kv_states[node.layer_index];
                    const int nq = cfg.query_heads, nkv = cfg.kv_heads, hd = cfg.head_dim;
                    const int q_dim = nq * hd, q_proj_dim = cfg.gated ? 2 * q_dim : q_dim;
                    const int kv_dim = nkv * hd;
                    float* q_full_p = bufs[node.inputs[0]].ptr<float>() + node.input_offsets[0];
                    float* k_p = bufs[node.inputs[1]].ptr<float>() + node.input_offsets[1];
                    float* v_p = bufs[node.inputs[2]].ptr<float>() + node.input_offsets[2];
                    int q_stride = g.values[node.inputs[0]].cols;
                    const int k_stride = g.values[node.inputs[1]].cols;
                    const int v_stride = g.values[node.inputs[2]].cols;
                    if (L.q_bias.data) cuda_ops::add_bias_f32(q_full_p, L.q_bias.ptr<float>(), rows, q_proj_dim, q_stride, ctx.stream());
                    if (L.k_bias.data) cuda_ops::add_bias_f32(k_p, L.k_bias.ptr<float>(), rows, kv_dim, k_stride, ctx.stream());
                    if (L.v_bias.data) cuda_ops::add_bias_f32(v_p, L.v_bias.ptr<float>(), rows, kv_dim, v_stride, ctx.stream());

                    float* q_p = q_full_p;
                    float* gate_p = nullptr;
                    if (node.attention.gated) {
                        Tensor qb = cuda_scratch(ctx, {rows, q_dim});
                        Tensor gb = cuda_scratch(ctx, {rows, q_dim});
                        q_p = qb.ptr<float>();
                        gate_p = gb.ptr<float>();
                        cuda_ops::split_q_gate_f32(q_full_p, q_p, gate_p, rows, nq, hd, ctx.stream(), q_stride);
                        q_stride = q_dim;
                    }
                    Tensor& attn_out = get_out_buf(node.out, q_dim, rows);
                    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
                    const float* qnw = cfg.query_norm ? cfg.query_norm->ptr<float>() : nullptr;
                    const float* knw = cfg.key_norm ? cfg.key_norm->ptr<float>() : nullptr;
                    const float* rotary = cfg.rotary_factors ? cfg.rotary_factors->ptr<float>() : nullptr;
                    if (rows == 1) {
                        cuda_ops::fused_qk_norm_rope_kv_append_f32(
                            q_p, k_p, v_p, qnw, knw,
                            state.keys.ptr(), state.values.ptr(),
                            nq, nkv, hd, cfg.rotary_dims, pos_dev, pos_host, cfg.max_context,
                            cfg.rope_theta, node.norm_eps, cfg.rope_neox, ctx.stream(), rotary, state.keys.dtype);
                        constexpr int kMaxSplits = 16;
                        Tensor part_out = cuda_scratch(ctx, {nq, kMaxSplits, hd});
                        Tensor part_ml = cuda_scratch(ctx, {nq, kMaxSplits, 2});
                        // Use the same reduction order for eager and captured decode.
                        cuda_ops::gqa_splitk_decode_f32(
                            q_p, state.keys.ptr(), state.values.ptr(),
                            attn_out.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                            nq, nkv, hd, kv_len_dev, pos_host + 1, cfg.max_context,
                            scale, kMaxSplits, ctx.stream(), true, state.keys.dtype);
                    } else {
                        cuda_ops::batched_qk_norm_rope_kv_append_f32(
                            q_p, k_p, v_p, qnw, knw,
                            state.keys.ptr(), state.values.ptr(),
                            rows, nq, nkv, hd, cfg.rotary_dims, start_pos, cfg.max_context,
                            cfg.rope_theta, node.norm_eps, cfg.rope_neox, ctx.stream(), q_stride, k_stride, v_stride,
                            rotary, state.keys.dtype);
                        auto& arena = ctx.scratch_arena();
                        const size_t workspace = state.keys.dtype == DType::F16 && ctx.cublas() &&
                                                 rows >= 64 && start_pos >= 512 && hd % 32 == 0
                            ? cuda_ops::gqa_f16_prefill_workspace(rows, nq, hd, start_pos + rows) : 0;
                        const size_t aligned = (arena.used() + 255) & ~size_t{255};
                        if (workspace && aligned <= arena.capacity() && workspace <= arena.capacity() - aligned) {
                            const size_t mark = arena.mark();
                            cuda_ops::gqa_f16_prefill(
                                ctx.cublas(), q_p, q_stride, state.keys.ptr(), state.values.ptr(), attn_out.ptr<float>(),
                                rows, nq, nkv, hd, start_pos, scale, arena.alloc(workspace), ctx.stream());
                            arena.reset(mark);
                        } else {
                            cuda_ops::gqa_prefill_causal_f32(
                                q_p, state.keys.ptr(), state.values.ptr(), attn_out.ptr<float>(),
                                rows, nq, nkv, hd, start_pos, cfg.max_context, scale, ctx.stream(), q_stride, state.keys.dtype);
                        }
                    }
                    if (node.attention.gated) {
                        cuda_ops::apply_sigmoid_gate_f32(attn_out.ptr<float>(), gate_p, rows * q_dim, ctx.stream());
                    }
                };
                if (pa.enabled && M > 1) pa.attn_ms += time_cuda_fn(ctx.stream(), run_gqa);
                else run_gqa();
                break;
            }
            case OpKind::GDN_RECURRENT_BLOCK: {
                auto run_gdn = [&]() {
                    const auto& L = gdn_layers[static_cast<size_t>(node.layer_index)];
                    const auto& cfg = node.recurrent;
                    const auto& state = gdn_states[node.layer_index];
                    const int nk = cfg.key_heads, nv = cfg.value_heads, dk = cfg.key_dim, dv = cfg.value_dim;
                    const int conv_dim = 2 * nk * dk + nv * dv;
                    const int v_dim = nv * dv;
                    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
                    const float* qkv_p = bufs[node.inputs[0]].ptr<float>() + node.input_offsets[0];
                    const float* z_p = bufs[node.inputs[1]].ptr<float>() + node.input_offsets[1];
                    const float* a_p = bufs[node.inputs[2]].ptr<float>() + node.input_offsets[2];
                    const float* b_p = bufs[node.inputs[3]].ptr<float>() + node.input_offsets[3];
                    Tensor& gdn_out = get_out_buf(node.out, v_dim, rows);
                    if (rows == 1) {
                        if (use_fused) {
                            float* scratch = nullptr;
                            if (dv >= 64 && (dk == 64 || dk == 128)) {
                                scratch = cuda_scratch(ctx, {conv_dim + v_dim + 2 * nv}).ptr<float>();
                            }
                            cuda_ops::fused_gated_deltanet_step_f32(
                                qkv_p, z_p, a_p, b_p,
                                L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                                L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                                gdn_out.ptr<float>(), nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(), scratch,
                                cfg.tiled_value_heads);
                        } else {
                            Tensor sqkv = cuda_scratch(ctx, {conv_dim});
                            Tensor sy = cuda_scratch(ctx, {v_dim});
                            cuda_ops::unfused_gated_deltanet_step_f32(
                                qkv_p, z_p, a_p, b_p,
                                L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                                L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                                gdn_out.ptr<float>(), sqkv.ptr<float>(), sy.ptr<float>(),
                                nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(), cfg.tiled_value_heads);
                        }
                    } else {
                        int s_qkv = g.values[node.inputs[0]].cols;
                        int s_z = g.values[node.inputs[1]].cols;
                        int s_a = g.values[node.inputs[2]].cols;
                        int s_b = g.values[node.inputs[3]].cols;
                        Tensor gdn_ws = cuda_scratch(ctx, {static_cast<int64_t>(
                            cuda_ops::gated_deltanet_prefill_scratch_floats(rows, nk, nv, dk, dv))});
                        cuda_ops::fused_gated_deltanet_prefill_f32(
                            qkv_p, s_qkv, z_p, s_z, a_p, s_a, b_p, s_b,
                            L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                            L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                            gdn_out.ptr<float>(), rows, nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(),
                            gdn_ws.ptr<float>(), gdn_ws.numel(), cfg.tiled_value_heads);
                    }
                };
                if (pa.enabled && M > 1) pa.gdn_ms += time_cuda_fn(ctx.stream(), run_gdn);
                else run_gdn();
                break;
            }
            case OpKind::SWIGLU_INTERLEAVED: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* gate = bufs[node.inputs[0]].ptr<float>();
                if (node.inputs.size() == 1) {
                    cuda_ops::swiglu_interleaved_f32(gate, out_t.ptr<float>(), rows, node.output_cols, ctx.stream());
                } else {
                    cuda_ops::swiglu_f32(gate, bufs[node.inputs[1]].ptr<float>(), out_t.ptr<float>(),
                                        rows * node.output_cols, ctx.stream());
                }
                break;
            }
            case OpKind::SWIGLU_Q8_LINEAR: {
                const float* gate_up = bufs[node.inputs[0]].ptr<float>();
                float* out = get_out_buf(node.out, node.output_cols, rows).ptr<float>();
                auto& arena = ctx.scratch_arena();
                size_t scratch_mark = arena.mark();
                if (rows == 1) {
                    void* q8 = arena.alloc(row_nbytes(DType::Q8_1, node.input_cols));
                    cuda_ops::swiglu_interleaved_q8_1(gate_up, q8, rows, node.input_cols, ctx.stream());
                    decode_gemv(node.weight->dtype)(q8, node.weight->ptr(), out, node.output_cols, node.input_cols, ctx.stream(),
                                               ctx.preferred_gemv_geometry());
                } else if (auto mmq = rows >= ctx.mmq_crossover_m() ? prefill_mmq(node.weight->dtype) : nullptr) {
                    void* q8 = arena.alloc(static_cast<size_t>(rows) * row_nbytes(DType::Q8_1, node.input_cols));
                    cuda_ops::swiglu_interleaved_q8_1(gate_up, q8, rows, node.input_cols, ctx.stream());
                    mmq(q8, node.weight->ptr(), out, rows, node.output_cols, node.input_cols, ctx.stream());
                } else {
                    Tensor act = cuda_scratch(ctx, {rows, node.input_cols});
                    cuda_ops::swiglu_interleaved_f32(gate_up, act.ptr<float>(), rows, node.input_cols, ctx.stream());
                    dispatch_linear(ctx, act.ptr<float>(), *node.weight, out, rows, node.output_cols, node.input_cols, use_fused);
                }
                arena.reset(scratch_mark);
                break;
            }
            case OpKind::MOE_FFN_BLOCK: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(out_t.ptr(), bufs[node.inputs[0]].ptr(),
                                                    static_cast<size_t>(rows) * node.output_cols * sizeof(float),
                                                    cudaMemcpyDeviceToDevice, ctx.stream()));
                Tensor x_v = (rows == 1) ? out_t.view({node.output_cols}) : out_t;
                const auto& L = moe_layers[static_cast<size_t>(node.layer_index)];
                if (rows == 1) L.forward_cuda(ctx, x_v, use_fused);
                else L.forward_prefill_cuda(ctx, x_v, rows, use_fused);
                break;
            }
        }
        if constexpr (Observe) {
            if (timer) FUSIONRT_CUDA_CHECK(cudaEventRecord(timer->event(ni, 1), ctx.stream()));
            if (layer_trace) {
                int value = node.residual_out >= 0 ? node.residual_out : node.out;
                const std::string& name = g.values[value].name;
                if (name.starts_with("L") &&
                    (name.ends_with("_h2") || name.ends_with("_h2_upd") || name.ends_with("_moe_out"))) {
                    const size_t layer = std::stoul(name.substr(1, name.find('_') - 1));
                    if (layer_trace->outputs.size() <= layer) layer_trace->outputs.resize(layer + 1);
                    auto& output = layer_trace->outputs[layer];
                    output.resize(H);
                    auto read = [&](float* dst, int id) {
                        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, row_ptr(id, true), H * sizeof(float),
                                                            cudaMemcpyDeviceToHost, ctx.stream()));
                    };
                    if (is_lm_head && compute_logits) {
                        // The final fused head materializes only logits.
                        std::vector<float> residual(H);
                        read(output.data(), node.inputs[0]);
                        read(residual.data(), node.inputs[1]);
                        ctx.synchronize();
                        for (int j = 0; j < H; ++j) output[j] += residual[j];
                    } else {
                        read(output.data(), value);
                        ctx.synchronize();
                    }
                }
            }
        }
    }
    if constexpr (Observe) {
        if (timer) {
            ctx.synchronize();
            for (int ni = 0; ni < num_nodes; ++ni) {
                const auto& node = g.nodes[ni];
                if (!node.alive || (node.output_logits && !compute_logits && node.residual_out < 0)) continue;
                float ms = 0.0f;
                FUSIONRT_CUDA_CHECK(cudaEventElapsedTime(&ms, timer->event(ni, 0), timer->event(ni, 1)));
                if (ms <= 0) continue;
                const bool residual_only = node.output_logits && !compute_logits;
                OpTiming work;
                if (residual_only) {
                    work.name = node.tag + "_residual";
                    work.kind = op_kind_name(OpKind::ADD_RESIDUAL);
                    work.calls = 1;
                    work.flops = static_cast<int64_t>(M) * node.input_cols;
                    work.bytes = 3 * sizeof(float) * work.flops;
                } else {
                    work = operator_work(node, node.output_logits ? 1 : M, pos_host, ctx, use_fused,
                        !node.inputs.empty() && bufs[node.inputs[0]].dtype == DType::Q8_1,
                        node.out >= 0 && bufs[node.out].dtype == DType::Q8_1,
                        node.op == OpKind::GQA_ATTN_BLOCK ? &dense_layers[node.layer_index] : nullptr,
                        node.op == OpKind::MOE_FFN_BLOCK ? &moe_layers[node.layer_index] : nullptr,
                        node.op == OpKind::GQA_ATTN_BLOCK ? dtype_type_size(kv_states[node.layer_index].keys.dtype) : 0);
                }
                work.total_ms = ms;
                auto op = std::find_if(op_trace->ops.begin(), op_trace->ops.end(), [&](const OpTiming& entry) {
                    return entry.name == work.name && entry.kind == work.kind;
                });
                if (op == op_trace->ops.end()) {
                    op_trace->ops.push_back(std::move(work));
                } else {
                    ++op->calls;
                    op->total_ms += ms;
                    op->bytes += work.bytes;
                    op->flops += work.flops;
                }
            }
        }
    }
    ctx.scratch_arena().reset(mark);
}

static void decode_argmax(CudaContext& ctx, const Tensor& logits, Tensor& token, int vocab_size,
                          int* graph_pos = nullptr, int* graph_kv_len = nullptr) {
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    void* scratch = vocab_size >= 32768 ? arena.alloc(cuda_ops::kArgmaxScratchBytes) : nullptr;
    cuda_ops::argmax_f32(logits.ptr<float>(), vocab_size, token.ptr<int32_t>(), ctx.stream(),
                         scratch, graph_pos, graph_kv_len, static_cast<int>(token.shape[0] - 1));
    arena.reset(mark);
}

static bool can_use_cuda_graph(const HybridModelConfig& cfg) {
    return cfg.arch != "moe" || cfg.moe.gpu_resident_experts == cfg.moe.num_experts;
}

static void run_forward_token_step_cuda(
    CudaContext& ctx, const HybridModelConfig& cfg,
    const Tensor& tok_embeddings, const MiniGraph& ir, CudaIRWorkspace& ws,
    Tensor& hidden_buf, Tensor& logits_buf, Tensor& tok_id_dev, Tensor& pos_dev, Tensor& kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvState>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    CudaGraphExec& decode_graph, bool& graph_captured, bool& graph_fused_flag, bool& graph_has_argmax,
    int pos, bool use_fused, bool use_graph, bool with_argmax, bool first_step = true,
    OpTrace* trace = nullptr, CudaOpTimer* timer = nullptr) {
    int H = static_cast<int>(hidden_buf.shape[0]);

    auto run_body = [&](bool graph_mode) {
        cuda_ops::embed_lookup_f32(tok_embeddings.ptr(), tok_embeddings.dtype,
                                   tok_id_dev.ptr<int32_t>(), hidden_buf.ptr<float>(), 1, H, ctx.stream());
        int* p_dev = graph_mode ? pos_dev.ptr<int>() : nullptr;
        int* kl_dev = graph_mode ? kv_len_dev.ptr<int>() : nullptr;
        if (timer) {
            execute_ir_graph_cuda<true>(ir, ws, ctx, hidden_buf, logits_buf, 1, pos, pos, p_dev, kl_dev,
                                        dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused, trace, timer);
        } else {
            execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, logits_buf, 1, pos, pos, p_dev, kl_dev,
                                  dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused);
        }
        if (with_argmax) {
            decode_argmax(ctx, logits_buf, tok_id_dev, cfg.vocab_size, p_dev, kl_dev);
        }
    };

    if (use_graph && can_use_cuda_graph(cfg)) {
        // The generation graph advances both counters after it writes the next token.
        if (first_step) {
            int kv_len = pos + 1;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr(), &pos, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(kv_len_dev.ptr(), &kv_len, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
        }
        if (!graph_captured || graph_fused_flag != use_fused || graph_has_argmax != with_argmax) {
            decode_graph.capture(ctx.stream(), [&]() { run_body(true); });
            graph_captured = true;
            graph_fused_flag = use_fused;
            graph_has_argmax = with_argmax;
        }
        decode_graph.launch(ctx.stream());
    } else {
        run_body(false);
    }
}

std::vector<float> FusionModel::forward_token_cuda(CudaContext& ctx, int32_t token_id, int pos,
                                                   bool use_fused, bool use_graph) {
    cuda_session_.cached_prompt_tokens.clear();
    if (pos < 0 || pos >= cfg_.dense.max_seq_len) throw std::invalid_argument("Token position exceeds context");
    const MiniGraph& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    int32_t tid = std::clamp(token_id, int32_t{0}, cfg_.vocab_size - 1);
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_session_.token_ids.ptr(), &tid, sizeof(tid), cudaMemcpyHostToDevice, ctx.stream()));
    run_forward_token_step_cuda(ctx, cfg_, tok_embeddings_, ir, ws,
                                cuda_session_.hidden, cuda_session_.logits, cuda_session_.token_ids, cuda_session_.position, cuda_session_.kv_length,
                                dense_layers_, gdn_layers_, cuda_session_.kv, cuda_session_.gdn, moe_layers_,
                                cuda_session_.decode_graph, cuda_session_.graph_captured, cuda_session_.graph_fused, cuda_session_.graph_has_argmax,
                                pos, use_fused, use_graph, false);
    HostTensor h_logits = ctx.download(cuda_session_.logits);
    return std::vector<float>(h_logits.data<float>(), h_logits.data<float>() + cfg_.vocab_size);
}

namespace {
constexpr int kPrefillBatch = 128;
}

void FusionModel::set_device_profile(CudaContext& ctx, const DeviceProfile& profile) {
    ctx.set_mmq_crossover_m(profile.mmq_crossover_m);
    ctx.set_preferred_gemv_geometry({profile.preferred_gemv_warps, profile.preferred_gemv_rows_per_warp});
    ctx.set_empirical_l2_bytes(profile.empirical_l2_bytes);
    recommend_cuda_graph_ = profile.recommend_cuda_graph;
    cuda_session_.graph_captured = false;
    cuda_session_.cached_prompt_tokens.clear();
}

ModelMemory FusionModel::inspect_cuda(CudaContext& ctx) {
    ModelMemory result;
    for (const auto& state : cuda_session_.kv) result.kv_bytes += state.keys.nbytes() + state.values.nbytes();
    for (const auto& state : cuda_session_.gdn) result.ssm_bytes += state.recurrence.nbytes() + state.convolution.nbytes();
    result.persistent_bytes = ctx.weight_arena().used();
    std::unordered_set<const void*> counted;
    auto count = [&](std::initializer_list<const Tensor*> weights) {
        for (const auto* w : weights) {
            if (w->data && counted.insert(w->ptr()).second) result.weight_bytes[w->dtype] += w->nbytes();
        }
    };
    count({&tok_embeddings_, &lm_head_w_, &final_norm_w_});
    for (const auto& l : dense_layers_) {
        count({&l.attn_norm_w, &l.q_proj, &l.k_proj, &l.v_proj, &l.q_bias, &l.k_bias, &l.v_bias,
               &l.q_norm_w, &l.k_norm_w, &l.rope_factors, &l.o_proj, &l.mlp_norm_w,
               &l.gate_up_proj, &l.gate_proj, &l.up_proj, &l.down_proj});
    }
    for (const auto& l : gdn_layers_) {
        count({&l.attn_norm_w, &l.qkv_proj, &l.z_proj, &l.a_proj, &l.b_proj, &l.conv_w, &l.conv_bias,
               &l.A_log, &l.dt_bias, &l.norm_w, &l.o_proj, &l.mlp_norm_w, &l.gate_up_proj,
               &l.gate_proj, &l.up_proj, &l.down_proj});
    }
    for (const auto& l : moe_layers_) {
        count({&l.norm_w, &l.router_w, &l.gpu_gate_up_w, &l.gpu_down_w, &l.shared_gate_up_w, &l.shared_down_w});
    }

    const int rows = std::min(kPrefillBatch, cfg_.dense.max_seq_len);
    const std::vector<int32_t> tokens(rows, 0);
    auto& arena = ctx.scratch_arena();
    const size_t mark = arena.mark();
    arena.track_peak(true);
    try {
        reset_kv_and_state(ctx);
        (void)prefill_cuda_impl(ctx, tokens.data(), rows, 0, true);
        reset_kv_and_state(ctx);
        (void)prefill_cuda_impl(ctx, tokens.data(), 1, 0, true);
        decode_argmax(ctx, cuda_session_.logits, cuda_session_.token_ids, cfg_.vocab_size);
        ctx.synchronize();
        result.static_scratch_bytes = arena.peak() - mark;
    } catch (...) {
        arena.track_peak(false);
        arena.reset(mark);
        throw;
    }
    arena.track_peak(false);
    reset_kv_and_state(ctx);
    return result;
}

std::vector<float> FusionModel::forward_prefill_cuda(CudaContext& ctx, const std::vector<int32_t>& token_ids,
                                                     int start_pos, bool use_fused) {
    if (token_ids.empty()) return {};
    cuda_session_.cached_prompt_tokens.clear();
    if (start_pos < 0 || token_ids.size() > static_cast<size_t>(cfg_.dense.max_seq_len - std::min(start_pos, cfg_.dense.max_seq_len))) {
        throw std::invalid_argument("Prompt exceeds model context");
    }
    return prefill_cuda_impl(ctx, token_ids.data(), static_cast<int>(token_ids.size()), start_pos, use_fused);
}

std::vector<float> FusionModel::prefill_cached_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt,
                                                    bool use_fused, int* prefilled_tokens, LayerTrace* trace,
                                                    OpTrace* op_trace) {
    if (prompt.empty() || prompt.size() > static_cast<size_t>(cfg_.dense.max_seq_len)) {
        throw std::invalid_argument("Prompt is empty or exceeds model context");
    }
    size_t lcp = 0;
    if (cuda_session_.cached_fused == use_fused) {
        lcp = static_cast<size_t>(std::mismatch(prompt.begin(), prompt.end(), cuda_session_.cached_prompt_tokens.begin(),
                                               cuda_session_.cached_prompt_tokens.end()).first - prompt.begin());
    }
    if (gdn_layers_.empty() && lcp == prompt.size()) {
        --lcp;
    }
    if (lcp == 0 || (!gdn_layers_.empty() && (lcp != cuda_session_.cached_prompt_tokens.size() || lcp == prompt.size()))) {
        reset_kv_and_state(ctx);
        lcp = 0;
    }
    cuda_session_.cached_prompt_tokens.clear();
    auto logits = prefill_cuda_impl(ctx, prompt.data() + lcp, static_cast<int>(prompt.size() - lcp),
                                    static_cast<int>(lcp), use_fused, trace, op_trace);
    cuda_session_.cached_prompt_tokens = prompt;
    cuda_session_.cached_fused = use_fused;
    if (prefilled_tokens) *prefilled_tokens = static_cast<int>(prompt.size() - lcp);
    return logits;
}

void FusionModel::execute_prefill_chunks(CudaContext& ctx, const int32_t* token_ids, int count,
                                         int start_pos, bool use_fused, LayerTrace* layer_trace,
                                         OpTrace* op_trace, bool compute_logits) {
    const int H = static_cast<int>(tok_embeddings_.shape[1]);
    auto& arena = ctx.scratch_arena();
    struct RestoreArena {
        CudaArena& arena;
        size_t mark;
        ~RestoreArena() { arena.reset(mark); }
    } restore{arena, arena.mark()};

    const int capacity = std::min(count, kPrefillBatch);
    Tensor ids = cuda_scratch(ctx, {capacity}, DType::I32);
    Tensor hidden = cuda_scratch(ctx, {capacity, H});
    const auto& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    std::unique_ptr<CudaOpTimer> timer;
    if (op_trace) timer = std::make_unique<CudaOpTimer>(ir.nodes.size());
    auto& pa = prof_acc();

    for (int begin = 0; begin < count; begin += capacity) {
        const int rows = std::min(capacity, count - begin);
        const int position = start_pos + begin;
        const bool last = begin + rows == count;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids.ptr(), token_ids + begin, rows * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, ctx.stream()));
        cuda_ops::embed_lookup_f32(tok_embeddings_.ptr(), tok_embeddings_.dtype,
                                   ids.ptr<int32_t>(), hidden.ptr<float>(), rows, H, ctx.stream());
        if (pa.enabled) pa.reset();
        if (layer_trace || op_trace) {
            execute_ir_graph_cuda<true>(ir, ws, ctx, hidden, cuda_session_.logits, rows, position, position,
                nullptr, nullptr, dense_layers_, gdn_layers_, cuda_session_.kv, cuda_session_.gdn,
                moe_layers_, use_fused, op_trace, timer.get(), last ? layer_trace : nullptr, last && compute_logits);
        } else {
            execute_ir_graph_cuda(ir, ws, ctx, hidden, cuda_session_.logits, rows, position, position,
                nullptr, nullptr, dense_layers_, gdn_layers_, cuda_session_.kv, cuda_session_.gdn,
                moe_layers_, use_fused, nullptr, nullptr, nullptr, last && compute_logits);
        }
        if (pa.enabled) {
            std::cerr << "[PROF prefill M=" << rows << "] x16=" << pa.x16_ms << "ms w16=" << pa.w16_ms
                      << "ms cublas=" << pa.cublas_ms << "ms (calls=" << pa.calls_cublas
                      << ") total_gqa=" << pa.attn_ms << "ms total_gdn=" << pa.gdn_ms
                      << "ms lm=" << pa.lm_ms << "ms\n";
        }
    }
}

std::vector<float> FusionModel::prefill_cuda_impl(CudaContext& ctx, const int32_t* token_ids, int count,
                                                 int start_pos, bool use_fused, LayerTrace* layer_trace,
                                                 OpTrace* op_trace) {
    std::vector<int32_t> tokens(token_ids, token_ids + count);
    for (auto& token : tokens) token = std::clamp(token, int32_t{0}, cfg_.vocab_size - 1);
    execute_prefill_chunks(ctx, tokens.data(), count, start_pos, use_fused, layer_trace, op_trace);
    HostTensor logits = ctx.download(cuda_session_.logits);
    return {logits.data<float>(), logits.data<float>() + cfg_.vocab_size};
}

std::vector<int32_t> FusionModel::generate_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                int max_new_tokens, bool use_fused, bool use_graph,
                                                GenerationStats* stats, bool keep_state, int start_pos,
                                                int32_t stop_eos_id, const GenerationOptions& options) {
    options.sampling.validate();
    if (max_new_tokens < 0 || start_pos < 0) throw std::invalid_argument("Invalid generation length or start position");
    if (!keep_state) start_pos = 0;
    std::vector<int32_t> tokens = prompt_ids;
    if (tokens.empty()) tokens.push_back(std::min(1, cfg_.vocab_size - 1));
    const int prompt_len = static_cast<int>(tokens.size());
    if (static_cast<int64_t>(start_pos) + prompt_len > cfg_.dense.max_seq_len) {
        throw std::invalid_argument("Prompt exceeds model context");
    }
    max_new_tokens = std::min(max_new_tokens, cfg_.dense.max_seq_len - start_pos - prompt_len);
    bool reuse = options.reuse_prefix && !keep_state;
    if (!reuse) {
        cuda_session_.cached_prompt_tokens.clear();
        if (!keep_state) reset_kv_and_state(ctx);
    }
    const bool greedy = options.sampling.greedy();
    use_graph = use_graph && greedy && !options.op_trace;
    std::mt19937_64 rng;
    if (!greedy) rng.seed(options.sampling.seed);
    OpTrace prefill_profile;
    if (options.op_trace) prefill_profile.mode = "prefill";

    auto download_argmax_tok = [&]() -> int32_t {
        int32_t tok = 0;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(&tok, cuda_session_.token_ids.ptr<int32_t>(), sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
        ctx.synchronize();
        return tok;
    };
    auto sample_gpu = [&]() {
        decode_argmax(ctx, cuda_session_.logits, cuda_session_.token_ids, cfg_.vocab_size);
        return download_argmax_tok();
    };

    auto tp0 = std::chrono::high_resolution_clock::now();
    int prefilled = prompt_len;
    OpTrace* prefill_trace = options.op_trace ? &prefill_profile : nullptr;
    auto logits = reuse ? prefill_cached_cuda(ctx, tokens, use_fused, &prefilled, options.layer_trace, prefill_trace)
                        : prefill_cuda_impl(ctx, tokens.data(), prompt_len, start_pos, use_fused,
                                            options.layer_trace, prefill_trace);
    // A failing callback or CUDA operation must not leave a reusable recurrent state.
    cuda_session_.cached_prompt_tokens.clear();
    int32_t next_tok = -1;
    if (max_new_tokens > 0) {
        next_tok = greedy ? sample_gpu() :
            Sampler::sample(logits.data(), cfg_.vocab_size, tokens, options.sampling, rng);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    int pos = start_pos + prompt_len;
    int gen_count = 0;
    const MiniGraph& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    std::unique_ptr<CudaOpTimer> timer;
    if (options.op_trace && max_new_tokens > 1) {
        timer = std::make_unique<CudaOpTimer>(ir.nodes.size());
        options.op_trace->mode = "decode";
    }
    auto decode = [&](bool first_step) {
        run_forward_token_step_cuda(ctx, cfg_, tok_embeddings_, ir, ws,
                                    cuda_session_.hidden, cuda_session_.logits, cuda_session_.token_ids, cuda_session_.position, cuda_session_.kv_length,
                                    dense_layers_, gdn_layers_, cuda_session_.kv, cuda_session_.gdn, moe_layers_,
                                    cuda_session_.decode_graph, cuda_session_.graph_captured, cuda_session_.graph_fused, cuda_session_.graph_has_argmax,
                                    pos++, use_fused, use_graph, greedy, first_step, options.op_trace, timer.get());
    };
    auto td0 = std::chrono::high_resolution_clock::now();
    if (use_graph && can_use_cuda_graph(cfg_) && !options.on_token && stop_eos_id < 0 && max_new_tokens > 1 &&
        static_cast<int64_t>(pos) + max_new_tokens <= cuda_session_.token_ids.shape[0]) {
        // Graph replay feeds the next greedy step and records IDs on the device.
        int first_pos = pos;
        for (int step = 1; step < max_new_tokens; ++step) decode(step == 1);
        tokens.push_back(next_tok);
        tokens.resize(static_cast<size_t>(prompt_len) + max_new_tokens);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(tokens.data() + prompt_len + 1, cuda_session_.token_ids.ptr<int32_t>() + 1 + first_pos,
                                            static_cast<size_t>(max_new_tokens - 1) * sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
        ctx.synchronize();
        gen_count = max_new_tokens;
    } else {
        for (int step = 0; step < max_new_tokens; ++step) {
            tokens.push_back(next_tok);
            ++gen_count;
            if (stop_eos_id >= 0 && next_tok == stop_eos_id) break;
            if (options.on_token && !options.on_token(next_tok)) break;
            if (step + 1 == max_new_tokens) break;
            if (!greedy) {
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_session_.token_ids.ptr(), &next_tok, sizeof(next_tok),
                                                    cudaMemcpyHostToDevice, ctx.stream()));
            }
            decode(step == 0);
            if (greedy) {
                next_tok = download_argmax_tok();
            } else {
                HostTensor host_logits = ctx.download(cuda_session_.logits);
                next_tok = Sampler::sample(host_logits.data<float>(), cfg_.vocab_size, tokens, options.sampling, rng);
            }
        }
    }
    if (reuse) {
        // The most recently sampled token has not been forwarded into KV/SSM state.
        cuda_session_.cached_prompt_tokens.assign(tokens.begin(), tokens.begin() + pos);
        cuda_session_.cached_fused = use_fused;
    }
    if (options.op_trace && options.op_trace->ops.empty()) *options.op_trace = std::move(prefill_profile);
    auto td1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();

    if (stats) {
        stats->prompt_tokens = static_cast<int>(prompt_ids.size());
        stats->generated_tokens = gen_count;
        stats->prefilled_tokens = prefilled;
        stats->prefill_ms = prefill_ms;
        stats->decode_ms = decode_ms;
        stats->prefill_tok_per_s = (prefill_ms > 0.0) ? (static_cast<double>(prefilled) * 1000.0 / prefill_ms) : 0.0;
        stats->decode_tok_per_s = (decode_ms > 0.0 && gen_count > 1) ? (static_cast<double>(gen_count - 1) * 1000.0 / decode_ms) : 0.0;
    }
    return tokens;
}

std::vector<GenerationStats> FusionModel::bench_model_cuda(CudaContext& ctx, int pp, int tg, int reps, bool use_graph,
                                                           OpTrace* trace, int decode_prefix) {
    if (decode_prefix < 0 || tg < 0 || decode_prefix > cfg_.dense.max_seq_len - tg) {
        throw std::invalid_argument("Decode benchmark exceeds model context");
    }
    using Clock = std::chrono::steady_clock;
    const auto& ir = fused_ir_graph_;
    CudaIRWorkspace ws(ir);
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    const int H = static_cast<int>(tok_embeddings_.shape[1]);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> token_id(0, cfg_.vocab_size - 1);
    std::vector<int32_t> prompt(pp), steps(tg), prefix(decode_prefix);
    std::generate(prompt.begin(), prompt.end(), [&] { return token_id(rng); });
    std::generate(steps.begin(), steps.end(), [&] { return token_id(rng); });
    std::generate(prefix.begin(), prefix.end(), [&] { return token_id(rng); });
    std::vector<GenerationStats> results(reps);

    void* host_ptr = nullptr;
    size_t logits_bytes = cuda_session_.logits.nbytes();
    FUSIONRT_CUDA_CHECK(cudaMallocHost(&host_ptr, logits_bytes + sizeof(int32_t) * std::max(pp, 3)));
    std::unique_ptr<void, decltype(&cudaFreeHost)> host_logits(host_ptr, cudaFreeHost);
    auto* host_args = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(host_ptr) + logits_bytes);
    auto read_logits = [&] {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host_logits.get(), cuda_session_.logits.ptr(), logits_bytes,
                                           cudaMemcpyDeviceToHost, ctx.stream()));
    };
    use_graph = use_graph && can_use_cuda_graph(cfg_);

    if (pp > 0) {
        std::copy(prompt.begin(), prompt.end(), host_args);
        auto prefill = [&] {
            execute_prefill_chunks(ctx, host_args, pp, 0, true, nullptr, nullptr);
            read_logits();
        };
        CudaGraphExec graph;
        if (use_graph) graph.capture(ctx.stream(), prefill);
        for (int rep = -1; rep < reps; ++rep) {
            reset_kv_and_state(ctx);
            auto start = Clock::now();
            if (use_graph) graph.launch(ctx.stream());
            else prefill();
            ctx.synchronize();
            if (rep >= 0) {
                auto& s = results[rep];
                s.prompt_tokens = pp;
                s.prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                s.prefill_tok_per_s = 1000.0 * pp / s.prefill_ms;
            }
        }
        arena.reset(mark);
    }

    if (tg > 0) {
        Tensor args = cuda_scratch(ctx, {3}, DType::I32);
        auto decode = [&](int pos) {
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(args.ptr(), host_args, args.nbytes(), cudaMemcpyHostToDevice, ctx.stream()));
            cuda_ops::embed_lookup_f32(tok_embeddings_.ptr(), tok_embeddings_.dtype,
                                       args.ptr<int32_t>(), cuda_session_.hidden.ptr<float>(), 1, H, ctx.stream());
            execute_ir_graph_cuda(ir, ws, ctx, cuda_session_.hidden, cuda_session_.logits, 1, pos, pos,
                                  use_graph ? args.ptr<int32_t>() + 1 : nullptr,
                                  use_graph ? args.ptr<int32_t>() + 2 : nullptr,
                                  dense_layers_, gdn_layers_, cuda_session_.kv, cuda_session_.gdn, moe_layers_, true);
            read_logits();
        };
        CudaGraphExec graph;
        if (use_graph) graph.capture(ctx.stream(), [&] { decode(decode_prefix); });
        std::vector<std::pair<HostTensor, HostTensor>> recurrent_prefix;
        if (!prefix.empty()) {
            reset_kv_and_state(ctx);
            execute_prefill_chunks(ctx, prefix.data(), decode_prefix, 0, true, nullptr, nullptr, false);
            for (const auto& state : cuda_session_.gdn) {
                recurrent_prefix.emplace_back(ctx.download(state.convolution), ctx.download(state.recurrence));
            }
            ctx.synchronize();
        }
        for (int rep = -1; rep < reps; ++rep) {
            if (prefix.empty()) reset_kv_and_state(ctx);
            // Decode only overwrites KV after the prefix. GDN state must be restored in place.
            for (size_t i = 0; i < recurrent_prefix.size(); ++i) {
                ctx.upload_into(recurrent_prefix[i].first, cuda_session_.gdn[i].convolution);
                ctx.upload_into(recurrent_prefix[i].second, cuda_session_.gdn[i].recurrence);
            }
            auto start = Clock::now();
            for (int pos = 0; pos < tg; ++pos) {
                host_args[0] = steps[pos];
                host_args[1] = decode_prefix + pos;
                host_args[2] = decode_prefix + pos + 1;
                if (use_graph) graph.launch(ctx.stream());
                else decode(decode_prefix + pos);
                ctx.synchronize();
            }
            if (rep >= 0) {
                auto& s = results[rep];
                s.generated_tokens = tg;
                s.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                s.decode_tok_per_s = 1000.0 * tg / s.decode_ms;
            }
        }
    }
    arena.reset(mark);
    if (trace) {
        reset_kv_and_state(ctx);
        if (tg > 0) {
            trace->mode = "decode";
            if (!prefix.empty()) execute_prefill_chunks(ctx, prefix.data(), decode_prefix, 0, true, nullptr, nullptr, false);
            CudaOpTimer timer(ir.nodes.size());
            for (int pos = 0; pos < tg; ++pos) {
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_session_.token_ids.ptr(), &steps[pos], sizeof(int32_t),
                                                    cudaMemcpyHostToDevice, ctx.stream()));
                run_forward_token_step_cuda(ctx, cfg_, tok_embeddings_, ir, ws, cuda_session_.hidden, cuda_session_.logits,
                                            cuda_session_.token_ids, cuda_session_.position, cuda_session_.kv_length, dense_layers_, gdn_layers_, cuda_session_.kv, cuda_session_.gdn, moe_layers_,
                                            cuda_session_.decode_graph, cuda_session_.graph_captured, cuda_session_.graph_fused, cuda_session_.graph_has_argmax,
                                            decode_prefix + pos, true, false, false, true, trace, &timer);
            }
        } else {
            trace->mode = "prefill";
            (void)prefill_cuda_impl(ctx, prompt.data(), pp, 0, true, nullptr, trace);
        }
    }
    return results;
}

}  // namespace fusionrt
#endif
