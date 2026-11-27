#include "cuda_dispatch.h"
#include "cuda_profile.h"
#include "../core/cpu_moe.h"
#include "../core/moe_route_trace.h"
#include "fusionrt/weight_codec.h"

#ifdef FUSIONRT_ENABLE_CUDA

#include <chrono>
#include <cstdlib>

namespace fusionrt {

ProfAccum& prof_acc() {
    static ProfAccum p = []() {
        ProfAccum x;
        const char* e = std::getenv("FUSIONRT_PROFILE");
        x.enabled = (e && e[0] == '1');
        return x;
    }();
    return p;
}
double time_cuda_fn(cudaStream_t s, const std::function<void()>& fn) {
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

DecodeGemv decode_gemv(DType dtype) {
    switch (dtype) {
        case DType::Q4_A: return cuda_ops::gemv_q4_a_q8_1;
        case DType::Q8_A: return cuda_ops::gemv_q8_a_q8_1;
        case DType::Q6_K: return cuda_ops::gemv_q6_k_q8_1;
        case DType::Q4_KA: return cuda_ops::gemv_q4_ka_q8_1;
        case DType::Q6_A: return cuda_ops::gemv_q6_a_q8_1;
        case DType::Q4_K: return cuda_ops::gemv_q4_k_q8_1;
        case DType::Q5_K: return cuda_ops::gemv_q5_k_q8_1;
        default: return nullptr;
    }
}
PrefillMmq prefill_mmq(DType dtype) {
    switch (dtype) {
        case DType::Q4_A: return cuda_ops::mmq_q4_a_q8_1;
        case DType::Q8_A: return cuda_ops::mmq_q8_a_q8_1;
        case DType::Q6_K: return cuda_ops::mmq_q6_k_q8_1;
        case DType::Q4_KA: return cuda_ops::mmq_q4_ka_q8_1;
        case DType::Q6_A: return cuda_ops::mmq_q6_a_q8_1;
        case DType::Q4_K: return cuda_ops::mmq_q4_k_q8_1;
        case DType::Q5_K: return cuda_ops::mmq_q5_k_q8_1;
        default: return nullptr;
    }
}

static bool try_q8_1_linear(CudaContext& ctx, const float* x, const float* res_b,
                            const float* norm_w, float* res_out, const Tensor& w,
                            float* y, int M, int N, int K, float eps, bool allow_mmq) {
    auto gemv = M == 1 ? decode_gemv(w.dtype) : nullptr;
    auto mmq = M > 1 && M >= ctx.mmq_crossover_m() && allow_mmq ? prefill_mmq(w.dtype) : nullptr;
    if (!gemv && !mmq) return false;
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    void* q8 = arena.alloc(static_cast<size_t>(M) * row_nbytes(DType::Q8_1, K));
    cuda_ops::quantize_x_q8_1(x, res_b, norm_w, res_out, q8, M, K, eps, ctx.stream());
    if (gemv) gemv(q8, w.ptr(), y, N, K, ctx.stream(), ctx.preferred_gemv_geometry());
    else mmq(q8, w.ptr(), y, M, N, K, ctx.stream(), ctx.cost_hardware());
    arena.reset(mark);
    return true;
}

static bool try_cublas_prefill(CudaContext& ctx, const float* x, const float* norm_w, const Tensor& w,
                               float* y, int M, int N, int K, float eps) {
    if (M < 4 || ctx.cublas() == nullptr || w.dtype == DType::F32 || w.dtype == DType::Q2_K || prefill_mmq(w.dtype)) return false;
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

bool use_codec_mmq(const CudaContext& ctx, int rows, bool allow_prefill) {
    return allow_prefill && rows >= std::max(16, ctx.mmq_crossover_m());
}

static void codec_linear(CudaContext& ctx, const float* x, const float* norm, const Tensor& w,
                         float* y, int M, int N, int K, float eps, bool allow_prefill) {
    auto& arena = ctx.scratch_arena();
    const size_t mark = arena.mark();
    if (has_weight_codec_int8(w.dtype)) {
        const bool mmq = use_codec_mmq(ctx, M, allow_prefill);
        const size_t bytes = mmq ? 0 : cuda_ops::codec_verify_scratch_bytes(M, K);
        const auto* quantize_rows = bytes && !norm ? x : nullptr;
        void* q8 = nullptr;
        if (!quantize_rows) {
            q8 = arena.alloc(static_cast<size_t>(M) * row_nbytes(DType::Q8_1, K));
            cuda_ops::quantize_x_q8_1(x, nullptr, norm, nullptr, q8, M, K, eps, ctx.stream());
        }
        if (mmq) {
            cuda_ops::codec_mmq_q8_1(q8, w.ptr(), w.dtype, y, M, N, K, ctx.stream());
        } else {
            auto* scratch = bytes ? static_cast<int*>(arena.alloc(bytes)) : nullptr;
            cuda_ops::codec_gemv_q8_1(q8, w.ptr(), w.dtype, y, M, N, K, ctx.stream(), scratch, quantize_rows);
        }
        arena.reset(mark);
        return;
    }
    const size_t row_bytes = static_cast<size_t>(K) * sizeof(uint16_t);
    const size_t x_bytes = row_bytes * M;
    const size_t available = arena.capacity() - arena.used();
    // For short prompts, rereading packed weights costs less traffic than
    // writing and rereading the expanded FP16 tile.
    const size_t packed_row = row_nbytes(w.dtype, K);
    const int traffic_crossover = 1 + static_cast<int>((2 * row_bytes + packed_row - 1) / packed_row);
    const int crossover = std::max(traffic_crossover, ctx.mmq_crossover_m());
    if (allow_prefill && M >= crossover && ctx.cublas() &&
        available >= x_bytes + row_bytes + 512) {
        const int tile = static_cast<int>(std::min({size_t{128}, static_cast<size_t>(N),
                                                   (available - x_bytes - 512) / row_bytes}));
        auto* x16 = static_cast<uint16_t*>(arena.alloc(x_bytes));
        auto* w16 = static_cast<uint16_t*>(arena.alloc(row_bytes * tile));
        cuda_ops::rmsnorm_or_copy_to_f16(x, norm, x16, M, K, eps, ctx.stream());
        for (int first = 0; first < N; first += tile) {
            const int count = std::min(tile, N - first);
            cuda_ops::codec_dequant_tile_f16(w.ptr(), w.dtype, w16, first, count, K, ctx.stream(), w.codec_tile_scale);
            cuda_ops::cublas_linear_f16_f32(ctx.cublas(), x16, w16, y + first, M, count, K, N, w.codec_tile_scale);
        }
    } else {
        if (norm) {
            auto* normalized = static_cast<float*>(arena.alloc(static_cast<size_t>(M) * K * sizeof(float)));
            cuda_ops::rmsnorm_f32(x, norm, normalized, M, K, eps, ctx.stream());
            x = normalized;
        }
        cuda_ops::codec_gemv_f32(x, w.ptr(), w.dtype, y, M, N, K, ctx.stream());
    }
    arena.reset(mark);
}

void dispatch_linear(CudaContext& ctx, const float* x, const Tensor& w, float* y, int M, int N, int K,
                     bool use_fused, bool verify) {
    if (is_weight_codec_dtype(w.dtype)) {
        auto& arena = ctx.scratch_arena();
        const auto mark = arena.mark();
        if (w.hadamard_block) {
            auto rotated = cuda_scratch(ctx, {M, K});
            cuda_ops::blockwise_hadamard_f32(x, w.hadamard_signs, rotated.ptr<float>(),
                                             M, K, w.hadamard_block, ctx.stream(), false,
                                             w.hadamard_head_dim, w.hadamard_key_heads);
            x = rotated.ptr<float>();
        }
        codec_linear(ctx, x, nullptr, w, y, M, N, K, 0.0f, use_fused && !verify);
        arena.reset(mark);
        return;
    }
    if (try_q8_1_linear(ctx, x, nullptr, nullptr, nullptr, w, y, M, N, K, 0.0f, use_fused || verify) ||
        (use_fused && !verify && try_cublas_prefill(ctx, x, nullptr, w, y, M, N, K, 0.0f))) {
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
    } else if (w.dtype == DType::Q2_K) {
        cuda_ops::linear_q2_k_f32(x, w.ptr(), y, M, N, K, ctx.stream(), use_fused && !verify);
    } else if (w.dtype == DType::Q4_KA || w.dtype == DType::Q6_A) {
        cuda_ops::linear_repacked_k_f32(x, w.ptr(), w.dtype, y, M, N, K, ctx.stream());
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
                             float* y, int M, int N, int K, float eps, bool use_fused, bool verify) {
    if (is_weight_codec_dtype(w.dtype)) {
        auto& arena = ctx.scratch_arena();
        const auto mark = arena.mark();
        const auto* norm = norm_w.ptr<float>();
        if (w.hadamard_block) {
            auto rotated = cuda_scratch(ctx, {M, K});
            cuda_ops::fused_rmsnorm_hadamard_f32(x, norm, w.hadamard_signs, rotated.ptr<float>(),
                                                  M, K, w.hadamard_block, eps, ctx.stream());
            x = rotated.ptr<float>();
            norm = nullptr;
        }
        codec_linear(ctx, x, norm, w, y, M, N, K, eps, use_fused && !verify);
        arena.reset(mark);
        return;
    }
    if (try_q8_1_linear(ctx, x, nullptr, norm_w.ptr<float>(), nullptr, w, y, M, N, K, eps, use_fused || verify) ||
        (use_fused && !verify && try_cublas_prefill(ctx, x, norm_w.ptr<float>(), w, y, M, N, K, eps))) {
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
    dispatch_linear(ctx, x_norm.ptr<float>(), w, y, M, N, K, use_fused, verify);
}

void dispatch_add_rmsnorm_linear(CudaContext& ctx, const float* a, const float* b,
                                 const Tensor& norm_w, const Tensor& w,
                                 float* res_out, float* y, int M, int N, int K, float eps, bool use_fused, bool verify) {
    if (try_q8_1_linear(ctx, a, b, norm_w.ptr<float>(), res_out, w, y, M, N, K, eps, use_fused || verify)) return;
    Tensor x_norm = cuda_scratch(ctx, {M, K});
    cuda_ops::add_rmsnorm_f32(a, b, norm_w.ptr<float>(), res_out, x_norm.ptr<float>(), M, K, eps, ctx.stream());
    dispatch_linear(ctx, x_norm.ptr<float>(), w, y, M, N, K, use_fused, verify);
}

namespace model_io {
void vision_linear_cuda(CudaContext& ctx, const Tensor& input, const Tensor& weight, Tensor& output, int rows) {
    dispatch_linear(ctx, input.ptr<float>(), weight, output.ptr<float>(), rows,
                    static_cast<int>(weight.shape[0]), static_cast<int>(weight.shape[1]), true);
}

}  // namespace model_io

void CudaSparseMoELayer::forward_prefill_cuda(CudaContext& ctx, Tensor& x_in_out, int M, bool use_fused,
                                             int moe_index, int route_pos) const {
    auto& arena = ctx.scratch_arena();
    struct ResetMoEScratch {
        CudaArena& arena;
        size_t mark;
        ~ResetMoEScratch() { arena.reset(mark); }
    } reset{arena, arena.mark()};
    const int H = cfg.hidden_size, I = cfg.intermediate_size, E = cfg.num_experts, K = cfg.top_k;
    const int gpu_E = cfg.gpu_resident_experts;
    const int borrowed = borrowed_gate_up_w.data ? static_cast<int>(borrowed_gate_up_w.shape[0]) : 0;
    const int base = gpu_E - borrowed;
    const auto stream = ctx.stream();
    auto* timing = ctx.moe_timing && moe_index >= 0 ? &ctx.moe_timing->layer(moe_index) : nullptr;

    Tensor x_norm = cuda_scratch(ctx, {M, H});
    cuda_ops::rmsnorm_f32(x_in_out.ptr<float>(), norm_w.ptr<float>(), x_norm.ptr<float>(), M, H, cfg.rms_norm_eps, stream);
    Tensor router_input = x_norm;
    if (router_scale.data) {
        router_input = cuda_scratch(ctx, {M, H});
        cuda_ops::rmsnorm_f32(x_in_out.ptr<float>(), router_scale.ptr<float>(), router_input.ptr<float>(),
                              M, H, cfg.rms_norm_eps, stream);
    }
    Tensor logits = cuda_scratch(ctx, {M, E});
    cuda_ops::linear_f32(router_input.ptr<float>(), router_w.ptr<float>(), logits.ptr<float>(), M, E, H, stream);
    Tensor topk_idx = cuda_scratch(ctx, {M, K}, DType::I32);
    Tensor topk_w = cuda_scratch(ctx, {M, K});
    cuda_ops::moe_topk_router_f32(logits.ptr<float>(), topk_idx.ptr<int>(), topk_w.ptr<float>(),
        E, K, cfg.norm_topk_prob, stream, router_bias.ptr<float>(), cfg.sigmoid_router,
        cfg.expert_weights_scale, cfg.expert_groups, cfg.expert_groups_used, M);
    if (down_scale.data)
        cuda_ops::scale_moe_routes_f32(topk_w.ptr<float>(), topk_idx.ptr<int>(),
                                       down_scale.ptr<float>(), M * K, stream);

    float* host_x = nullptr;
    float* host_out = nullptr;
    float* host_weights = nullptr;
    int32_t* host_indices = nullptr;
    if (gpu_E < E) {
        host_x = ctx.moe_transfer_buffer(2 * x_norm.nbytes() + topk_w.nbytes() + topk_idx.nbytes());
        host_out = host_x + static_cast<size_t>(M) * H;
        host_weights = host_out + static_cast<size_t>(M) * H;
        host_indices = reinterpret_cast<int32_t*>(host_weights + static_cast<size_t>(M) * K);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host_x, x_norm.ptr(), x_norm.nbytes(), cudaMemcpyDeviceToHost, stream));
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host_weights, topk_w.ptr(), topk_w.nbytes(), cudaMemcpyDeviceToHost, stream));
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host_indices, topk_idx.ptr(), topk_idx.nbytes(), cudaMemcpyDeviceToHost, stream));
        FUSIONRT_CUDA_CHECK(cudaEventRecord(ctx.moe_input_ready(), stream));
    } else if (route_pos >= 0) {
        auto* trace = ctx.moe_route_trace;
        auto* ids = trace->staging(static_cast<size_t>(M) * K);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids, topk_idx.ptr(), topk_idx.nbytes(), cudaMemcpyDeviceToHost, stream));
        ctx.synchronize();
        trace->record(model_layer_index, route_pos, M, K, ids);
    }

    Tensor moe_out = cuda_scratch(ctx, {M, H});
    if (gpu_E > 0) {
        Tensor scratch_inter = cuda_scratch(ctx, {K, I});
        auto run_hot = [&](const Tensor& gate, const Tensor& up, const Tensor& down, Tensor& out, int first, int count) {
            for (int row = 0; row < M; ++row) {
                const auto hidden_offset = static_cast<size_t>(row) * H;
                const auto route_offset = static_cast<size_t>(row) * K;
                cuda_ops::moe_indexed_swiglu_gemv(
                    x_norm.ptr<float>() + hidden_offset, topk_idx.ptr<int>() + route_offset, gate.ptr(), gate.dtype,
                    up.data ? up.ptr() : nullptr, up.dtype, down.ptr(), down.dtype, topk_w.ptr<float>() + route_offset,
                    out.ptr<float>() + hidden_offset, scratch_inter.ptr<float>(), H, I, K, count, stream, first,
                    geglu, gate_scale.ptr<float>(), up_scale.ptr<float>());
            }
        };
        if (base) run_hot(gpu_gate_up_w, gpu_up_w, gpu_down_w, moe_out, 0, base);
        if (borrowed) {
            const size_t mark = arena.mark();
            Tensor tail = base ? cuda_scratch(ctx, {M, H}) : moe_out;
            run_hot(borrowed_gate_up_w, borrowed_up_w, borrowed_down_w, tail, base, borrowed);
            if (base) cuda_ops::add_residual_f32(moe_out.ptr<float>(), tail.ptr<float>(), moe_out.ptr<float>(), M * H, stream);
            arena.reset(mark);
        }
    } else {
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(moe_out.ptr(), 0, moe_out.nbytes(), stream));
    }

    if (cfg.has_shared_expert) {
        const int width = cfg.shared_intermediate_size;
        Tensor sh_gu = cuda_scratch(ctx, {M, 2 * width});
        dispatch_linear(ctx, x_norm.ptr<float>(), shared_gate_up_w, sh_gu.ptr<float>(), M, 2 * width, H, use_fused);
        Tensor sh_act = cuda_scratch(ctx, {M, width});
        cuda_ops::swiglu_interleaved_f32(sh_gu.ptr<float>(), sh_act.ptr<float>(), M, width, stream);
        Tensor sh_down = cuda_scratch(ctx, {M, H});
        dispatch_linear(ctx, sh_act.ptr<float>(), shared_down_w, sh_down.ptr<float>(), M, H, width, use_fused);
        if (shared_gate_w.data) {
            Tensor shared_logits = cuda_scratch(ctx, {M});
            cuda_ops::linear_f32(x_norm.ptr<float>(), shared_gate_w.ptr<float>(), shared_logits.ptr<float>(), M, 1, H, stream);
            cuda_ops::apply_row_sigmoid_gate_f32(sh_down.ptr<float>(), shared_logits.ptr<float>(), M, H, stream);
        }
        cuda_ops::add_residual_f32(moe_out.ptr<float>(), sh_down.ptr<float>(), moe_out.ptr<float>(), M * H, stream);
    }

    if (host_x) {
        using Clock = std::chrono::steady_clock;
        if (timing) ctx.moe_timing->begin_gap(*timing, stream);
        const auto wait_start = timing ? Clock::now() : Clock::time_point{};
        // The event covers only the inputs. Hot and shared experts can run while the CPU works.
        FUSIONRT_CUDA_CHECK(cudaEventSynchronize(ctx.moe_input_ready()));
        if (route_pos >= 0) ctx.moe_route_trace->record(model_layer_index, route_pos, M, K, host_indices);
        const auto compute_start = timing ? Clock::now() : Clock::time_point{};
        const int cold = cpu_moe_accumulate_layer(&ctx.cpu_workers(), ctx.cpu_moe_workspace(), cpu_gate_up_w,
            cpu_up_w, cpu_down_w, host_indices, host_weights, gpu_E, base, M, K, host_x, host_out,
            ctx.cpu_moe_q8() ? CpuMoeMode::Parallel : CpuMoeMode::ParallelF32, geglu,
            cpu_gate_scale.bytes().empty() ? nullptr : cpu_gate_scale.data<float>(),
            cpu_up_scale.bytes().empty() ? nullptr : cpu_up_scale.data<float>());
        if (timing) {
            ctx.moe_timing->end_gap(*timing, stream);
            timing->hot += M * K - cold;
            timing->cold += cold;
            timing->input_wait_ms += std::chrono::duration<double, std::milli>(compute_start - wait_start).count();
            timing->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - compute_start).count();
        }
        if (cold > 0) {
            Tensor cold_out = cuda_scratch(ctx, {M, H});
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cold_out.ptr(), host_out, cold_out.nbytes(), cudaMemcpyHostToDevice, stream));
            cuda_ops::add_residual_f32(moe_out.ptr<float>(), cold_out.ptr<float>(), moe_out.ptr<float>(), M * H, stream);
        }
    } else if (timing) timing->hot += M * K;
    if (router_scale.data)
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(x_in_out.ptr(), moe_out.ptr(), moe_out.nbytes(),
                                            cudaMemcpyDeviceToDevice, stream));
    else
        cuda_ops::add_residual_f32(x_in_out.ptr<float>(), moe_out.ptr<float>(), x_in_out.ptr<float>(), M * H, stream);
}


}  // namespace fusionrt
#endif
