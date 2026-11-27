#include "model_loader.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include "../core/cpu_moe.h"
#include "../core/moe_route_trace.h"
#include "fusionrt/cuda_kernels.h"

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

constexpr int kPrefillCarrySlots = 8;
constexpr int kMaxDecodeSplits = 128;

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
        case DType::Q4_KA: return cuda_ops::gemv_q4_ka_q8_1;
        case DType::Q6_A: return cuda_ops::gemv_q6_a_q8_1;
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

void dispatch_linear(CudaContext& ctx, const float* x, const Tensor& w, float* y, int M, int N, int K,
                     bool use_fused, bool verify = false) {
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

}  // namespace

void model_io::vision_linear_cuda(CudaContext& ctx, const Tensor& input, const Tensor& weight, Tensor& output, int rows) {
    dispatch_linear(ctx, input.ptr<float>(), weight, output.ptr<float>(), rows,
                    static_cast<int>(weight.shape[0]), static_cast<int>(weight.shape[1]), true);
}

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
    Tensor logits = cuda_scratch(ctx, {M, E});
    cuda_ops::linear_f32(x_norm.ptr<float>(), router_w.ptr<float>(), logits.ptr<float>(), M, E, H, stream);
    Tensor topk_idx = cuda_scratch(ctx, {M, K}, DType::I32);
    Tensor topk_w = cuda_scratch(ctx, {M, K});
    cuda_ops::moe_topk_router_f32(logits.ptr<float>(), topk_idx.ptr<int>(), topk_w.ptr<float>(),
        E, K, cfg.norm_topk_prob, stream, router_bias.ptr<float>(), cfg.sigmoid_router,
        cfg.expert_weights_scale, cfg.expert_groups, cfg.expert_groups_used, M);

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
                    out.ptr<float>() + hidden_offset, scratch_inter.ptr<float>(), H, I, K, count, stream, first);
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
            ctx.cpu_moe_q8() ? CpuMoeMode::Parallel : CpuMoeMode::ParallelF32);
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
    cuda_ops::add_residual_f32(x_in_out.ptr<float>(), moe_out.ptr<float>(), x_in_out.ptr<float>(), M * H, stream);
}

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
        while (end < Count && sources[end]->desc.dtype == sources[begin]->desc.dtype) ++end;
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

void FusionModel::init_synthetic(CudaContext& ctx, uint64_t seed) {
    gemma_inputs_ = {};
    release_cuda_sessions();
    std::mt19937_64 rng(seed);
    int H = (cfg_.arch == "hybrid") ? cfg_.gdn.hidden_size
          : (cfg_.arch == "moe") ? cfg_.moe.hidden_size
          : cfg_.dense.hidden_size;

    tok_embeddings_ = ctx.upload(make_rand_f32({cfg_.vocab_size, H}, rng, 0.1f), true);
    final_norm_w_ = ctx.upload(make_ones_f32({H}), true);
    lm_head_w_ = ctx.upload(make_rand_f32({cfg_.vocab_size, H}, rng, 0.05f), true);
    cuda_workspace_.hidden = ctx.weight_arena().alloc_tensor({H}, DType::F32, "hidden_buf");
    cuda_workspace_.logits = ctx.weight_arena().alloc_tensor({cfg_.vocab_size}, DType::F32, "logits_buf");
    cuda_workspace_.token_ids = ctx.weight_arena().alloc_tensor({1 + static_cast<int64_t>(cfg_.dense.max_seq_len)}, DType::I32, "tok_id_dev");
    cuda_workspace_.position = ctx.weight_arena().alloc_tensor({1}, DType::I32, "pos_dev");
    cuda_workspace_.kv_length = ctx.weight_arena().alloc_tensor({1}, DType::I32, "kv_len_dev");

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
    cuda_kv_dtype_ = ctx.kv_dtype();
    cuda_slots_[0] = make_cuda_session();
    rebuild_cuda_ir();
}

static bool upload_raw_model_to_cuda(CudaContext& ctx, CudaArena& arena, LoadedModelRaw& raw,
                                     HybridModelConfig& cfg_out,
                                     Tensor& tok_emb, Tensor& final_norm, Tensor& lm_head,
                                     GemmaLayerInputWeights<Tensor>& gemma_inputs,
                                     Tensor& hidden_buf, Tensor& logits_buf,
                                     Tensor& tok_id_dev, Tensor& pos_dev, Tensor& kv_len_dev,
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
    hidden_buf = arena.alloc_tensor({H}, DType::F32, "hidden_buf");
    logits_buf = arena.alloc_tensor({cfg_out.vocab_size}, DType::F32, "logits_buf");
    tok_id_dev = arena.alloc_tensor({1 + static_cast<int64_t>(cfg_out.dense.max_seq_len)}, DType::I32, "tok_id_dev");
    pos_dev = arena.alloc_tensor({1}, DType::I32, "pos_dev");
    kv_len_dev = arena.alloc_tensor({1}, DType::I32, "kv_len_dev");

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
        L.norm_w = upload_cuda_weight(ctx, arena, lr.mlp_norm.bytes().empty() ? lr.attn_norm : lr.mlp_norm);
        L.router_w = upload_cuda_weight(ctx, arena, lr.router_w);
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
            L.mlp_norm_w = upload_cuda_weight(ctx, arena, lr.mlp_norm);
            if (!lr.gate_up.bytes().empty()) {
                L.gate_up_proj = upload_cuda_matrix(ctx, arena, lr.gate_up);
            } else {
                L.gate_proj = upload_cuda_matrix(ctx, arena, lr.gate);
                L.up_proj = upload_cuda_matrix(ctx, arena, lr.up);
            }
            L.down_proj = upload_cuda_matrix(ctx, arena, lr.down);
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
    };
    if (raw.lm_head_w.bytes().empty()) weight(raw.tok_embeddings, true);
    else reserve(raw.tok_embeddings.bytes().size());
    reserve(raw.gemma_inputs.token_embeddings.bytes().size());
    for (const auto* t : {&raw.final_norm_w, &raw.rope_factors, &raw.gemma_inputs.projection, &raw.gemma_inputs.norm}) weight(*t);
    weight(raw.lm_head_w, true);
    const auto& d = raw.cfg.dense;
    reserve(sizeof(float) * static_cast<size_t>(d.hidden_size));
    reserve(sizeof(float) * static_cast<size_t>(raw.cfg.vocab_size));
    reserve(sizeof(int32_t) * (1 + static_cast<size_t>(d.max_seq_len)));
    reserve(sizeof(int32_t));
    reserve(sizeof(int32_t));
    for (const auto& l : raw.layers) {
        for (const auto* t : {&l.attn_norm, &l.q_proj, &l.k_proj, &l.v_proj, &l.q_norm, &l.k_norm,
                             &l.q_bias, &l.k_bias, &l.v_bias, &l.o_proj, &l.mlp_norm, &l.gate_up,
                             &l.attn_post_norm, &l.ffn_post_norm,
                             &l.ple_gate, &l.ple_projection, &l.ple_post_norm,
                             &l.mla_q_a, &l.mla_q_a_norm, &l.mla_kv_a, &l.mla_kv_a_norm,
                             &l.gate, &l.up, &l.down, &l.z_proj, &l.a_proj, &l.b_proj,
                             &l.conv_w, &l.conv_bias, &l.A_log, &l.dt_bias, &l.gdn_norm_w,
                             &l.router_w, &l.router_bias, &l.shared_gate_up, &l.shared_down}) {
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
        if (layer.router_w.bytes().empty() || layer.kind == 1) continue;
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

void FusionModel::set_kv_commitment_tokens(int tokens) {
    if (tokens < 0) throw std::invalid_argument("KV commitment must be nonnegative");
    kv_commitment_tokens_ = tokens;
}

void FusionModel::set_moe_gpu_experts(int count) {
    if (count < -1) throw std::invalid_argument("Expected --moe-gpu-experts >= -1");
    moe_gpu_experts_override_ = count;
}

bool FusionModel::load_gguf(CudaContext& ctx, const std::string& path, std::string* err,
                            int max_seq_override, bool own_weights) {
    LoadedModelRaw raw;
    if (!parse_gguf_model_raw(path, max_seq_override, raw, err)) return false;
    release_cuda_sessions();
    if (!own_weights) gguf_weights_.release();
    auto& weights = own_weights ? gguf_weights_ : ctx.weight_arena();
    const bool resize = own_weights || weights.capacity() == 0 || owned_gguf_arena_ == &ctx;
    const auto kv_budget = place_cuda_experts(ctx, raw, moe_gpu_experts_override_, requested_kv_budget_,
        kv_commitment_tokens_, num_slots(), cuda_kv_dtype_, resize, resize ? weights.capacity() : 0);
    cuda_kv_pool_.set_kv_vram_budget_bytes(kv_budget);
    may_evict_paused_cuda_slots_ = true;
    if (resize) {
        const size_t bytes = cuda_model_arena_bytes(raw, ctx.cost_hardware(), raw.cfg.gpu_experts);
        if (weights.capacity() != bytes) weights.init(bytes);
        else weights.reset();
        if (!own_weights) owned_gguf_arena_ = &ctx;
    }
    if (!upload_raw_model_to_cuda(ctx, weights, raw, cfg_, tok_embeddings_, final_norm_w_, lm_head_w_, gemma_inputs_,
                                    cuda_workspace_.hidden, cuda_workspace_.logits, cuda_workspace_.token_ids, cuda_workspace_.position, cuda_workspace_.kv_length,
                                    layer_kinds_, layer_indices_, dense_layers_, gdn_layers_, moe_layers_)) {
        return false;
    }
    if (resize && moe_gpu_experts_override_ < 0 && kv_budget) expert_slabs_.resize(moe_layers_.size());
    cuda_slots_[0] = make_cuda_session();
    rebuild_cuda_ir();
    return true;
}

bool FusionModel::load_safetensors(CudaContext& ctx, const std::string& path, std::string* err, int max_seq_override) {
    LoadedModelRaw raw;
    if (!parse_safetensors_model_raw(path, max_seq_override, raw, err)) return false;
    release_cuda_sessions();
    auto& weights = ctx.weight_arena();
    const bool resize = weights.capacity() == 0 || owned_gguf_arena_ == &ctx;
    const auto kv_budget = place_cuda_experts(ctx, raw, moe_gpu_experts_override_, requested_kv_budget_,
        kv_commitment_tokens_, num_slots(), cuda_kv_dtype_, resize, resize ? weights.capacity() : 0);
    cuda_kv_pool_.set_kv_vram_budget_bytes(kv_budget);
    may_evict_paused_cuda_slots_ = true;
    if (resize) {
        const auto bytes = cuda_model_arena_bytes(raw, ctx.cost_hardware(), raw.cfg.gpu_experts);
        if (weights.capacity() != bytes) weights.init(bytes);
        else weights.reset();
        owned_gguf_arena_ = &ctx;
    }
    if (!upload_raw_model_to_cuda(ctx, weights, raw, cfg_, tok_embeddings_, final_norm_w_, lm_head_w_, gemma_inputs_,
                                    cuda_workspace_.hidden, cuda_workspace_.logits, cuda_workspace_.token_ids, cuda_workspace_.position, cuda_workspace_.kv_length,
                                    layer_kinds_, layer_indices_, dense_layers_, gdn_layers_, moe_layers_)) {
        return false;
    }
    if (resize && moe_gpu_experts_override_ < 0 && kv_budget) expert_slabs_.resize(moe_layers_.size());
    cuda_slots_[0] = make_cuda_session();
    rebuild_cuda_ir();
    return true;
}

void FusionModel::set_num_slots(int n) {
    if (n < 1) throw std::invalid_argument("Expected --slots >= 1");
    const auto count = static_cast<size_t>(n);
    if (active_slot_ >= count) {
        cuda_workspace_.decode_graph.reset();
        cuda_workspace_.graph_captured = false;
        active_slot_ = 0;
    }
    for (size_t i = count; i < cuda_slots_.size(); ++i) {
        if (cuda_slots_[i]) cuda_kv_pool_.truncate(cuda_slots_[i]->pages, 0);
    }
    cuda_slots_.resize(count);
}

std::unique_ptr<FusionModel::CudaSession> FusionModel::make_cuda_session() const {
    auto slot = std::make_unique<CudaSession>();
    auto aligned = [](size_t bytes) { return (bytes + 255) & ~size_t{255}; };
    size_t table_bytes = 0;
    for (const auto& layer : dense_layers_) {
        if (layer.kv_source_layer >= 0) continue;
        table_bytes += aligned((1 + (layer.cfg.max_seq_len - 1) / kKvPageTokens) * sizeof(uint8_t*));
    }
    size_t state_bytes = 0;
    for (const auto& layer : gdn_layers_) {
        const auto& g = layer.cfg;
        slot->gdn.push_back({
            Tensor::from_shape({layer.conv_w.shape[0], 3}, DType::F32, Backend::CUDA),
            Tensor::from_shape({g.num_v_heads, g.head_k_dim, g.head_v_dim}, DType::F32, Backend::CUDA)});
        state_bytes += aligned(slot->gdn.back().convolution.nbytes()) + aligned(slot->gdn.back().recurrence.nbytes());
    }
    slot->storage.init(table_bytes + state_bytes);
    for (const auto& layer : dense_layers_) {
        const auto& d = layer.cfg;
        const bool mla = d.mla.latent_rank > 0;
        slot->kv.push_back(layer.kv_source_layer >= 0 ? CudaKvView{} :
            make_cuda_kv_view(slot->storage, cuda_kv_dtype_, d.max_seq_len, d.num_key_value_heads,
                              mla ? d.mla.latent_width() : d.head_dim, mla));
    }
    if (state_bytes != 0) {
        slot->gdn_data = slot->storage.alloc(state_bytes);
        slot->ssm_bytes = state_bytes;
        auto* data = static_cast<uint8_t*>(slot->gdn_data);
        for (auto& state : slot->gdn) {
            state.convolution.data = data;
            data += aligned(state.convolution.nbytes());
            state.recurrence.data = data;
            data += aligned(state.recurrence.nbytes());
        }
    }
    return slot;
}

void FusionModel::release_cuda_sessions() {
    cuda_workspace_.decode_graph.reset();
    cuda_workspace_.graph_captured = false;
    for (auto& slot : cuda_slots_) slot.reset();
    cuda_kv_pool_.release();
    expert_slabs_.clear();
    expert_slab_bytes_ = 0;
    active_slot_ = 0;
    slot_clock_ = 0;
}

void FusionModel::clear_cuda_history() {
    auto& slot = cuda_session();
    slot.cached_tokens.clear();
    slot.checkpoints.clear();
}

void FusionModel::zero_cuda_gdn(CudaContext& ctx) {
    auto& slot = cuda_session();
    if (slot.ssm_bytes != 0) {
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(slot.gdn_data, 0, slot.ssm_bytes, ctx.stream()));
    }
}

void FusionModel::reset_kv_and_state(CudaContext& ctx) {
    clear_cuda_history();
    cuda_kv_pool_.truncate(cuda_session().pages, 0);
    zero_cuda_gdn(ctx);
    ctx.synchronize();
}

size_t FusionModel::cuda_kv_page_bytes() const {
    size_t page_bytes = 0;
    for (const auto& layer : cuda_slots_[active_slot_]->kv) page_bytes += layer.page_bytes();
    return page_bytes;
}

int FusionModel::cuda_max_active_tokens() const {
    const int context = cfg_.dense.max_seq_len;
    const auto budget = cuda_kv_pool_.vram_budget_bytes();
    const auto page_bytes = cuda_kv_page_bytes();
    if (!budget || !page_bytes) return context;
    const auto pages = std::min(budget / page_bytes, static_cast<size_t>(1 + (context - 1) / kKvPageTokens));
    return static_cast<int>(std::min(pages * kKvPageTokens, static_cast<size_t>(context)));
}

void FusionModel::validate_cuda_kv_capacity(int tokens) const {
    const auto budget = cuda_kv_pool_.vram_budget_bytes();
    if (tokens <= 0 || !budget) return;
    const auto page_bytes = cuda_kv_page_bytes();
    const size_t needed = static_cast<size_t>(1 + (tokens - 1) / kKvPageTokens) * page_bytes;
    if (needed > budget)
        throw std::invalid_argument("Active KV for " + std::to_string(tokens) + " tokens needs " +
            std::to_string(needed) + " bytes, exceeding the " + std::to_string(budget) + "-byte KV budget");
}

void FusionModel::promote_cuda_expert_slabs(CudaContext& ctx, int budgeted_tokens) {
    if (expert_slabs_.empty()) return;
    if (cuda_kv_pool_.bytes() < cuda_kv_pool_.reserved_bytes()) {
        ctx.synchronize();
        cuda_kv_pool_.release_unused_tail();
    }
    size_t per_page = 0;
    for (const auto& layer : cuda_session().kv) per_page += layer.page_bytes();
    const auto pages = 1 + (budgeted_tokens - 1) / kKvPageTokens;
    const auto required = std::max(cuda_kv_pool_.reserved_bytes(), per_page * pages);
    const auto commitment = cuda_kv_pool_.vram_budget_bytes();
    if (required >= commitment || expert_slab_bytes_ >= commitment - required) return;
    size_t available = commitment - required - expert_slab_bytes_;
    const auto align = [](size_t bytes) { return (bytes + 255) & ~size_t{255}; };
    int promoted = 0;
    for (size_t i = 0; i < moe_layers_.size(); ++i) {
        auto& layer = moe_layers_[i];
        auto& slab = expert_slabs_[i];
        const int base = layer.cfg.gpu_resident_experts;
        const int tail = layer.cfg.num_experts - base;
        if (tail == 0 || slab.storage) continue;
        const auto gate = layer.cpu_gate_up_w.bytes().size() / tail;
        const auto up = layer.cpu_up_w.bytes().size() / tail;
        const auto down = layer.cpu_down_w.bytes().size() / tail;
        if (!gate || !down) continue;
        auto bytes_for = [&](int n) { return align(gate * n) + align(up * n) + align(down * n); };
        int count = tail;
        while (count > 0 && bytes_for(count) > available) --count;
        if (!count) continue;
        const size_t bytes = bytes_for(count);
        uint8_t* ptr = nullptr;
        const auto status = cudaMalloc(&ptr, bytes);
        if (status == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            break;
        }
        FUSIONRT_CUDA_CHECK(status);
        ExpertSlab candidate;
        candidate.storage.reset(ptr);
        candidate.bytes = bytes;
        auto upload = [&](const HostTensor& source, size_t unit, uint8_t*& dest) {
            if (!unit) return Tensor{};
            const auto& shape = source.desc.shape;
            auto tensor = Tensor::make_contiguous({count, shape[1], shape[2]}, source.desc.dtype, Backend::CUDA, dest);
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dest, source.bytes().data(), unit * count,
                                                cudaMemcpyHostToDevice, ctx.stream()));
            dest += align(unit * count);
            return tensor;
        };
        auto g = upload(layer.cpu_gate_up_w, gate, ptr);
        auto u = upload(layer.cpu_up_w, up, ptr);
        auto d = upload(layer.cpu_down_w, down, ptr);
        if (promoted++ == 0) {
            cuda_workspace_.decode_graph.reset();
            cuda_workspace_.graph_captured = false;
        }
        slab = std::move(candidate);
        layer.borrowed_gate_up_w = std::move(g);
        layer.borrowed_up_w = std::move(u);
        layer.borrowed_down_w = std::move(d);
        layer.cfg.gpu_resident_experts += count;
        available -= bytes;
        expert_slab_bytes_ += bytes;
    }
    if (promoted) {
        int full = 0;
        for (const auto& layer : moe_layers_) full += layer.cfg.gpu_resident_experts == layer.cfg.num_experts;
        std::clog << "MoE slabs promoted=" << promoted << " borrowed_bytes=" << expert_slab_bytes_
                  << " full_gpu_layers=" << full << " kv_reserved_bytes=" << cuda_kv_pool_.reserved_bytes() << '\n';
    }
}

void FusionModel::reclaim_cuda_expert_slabs(CudaContext& ctx, size_t target_kv_bytes, bool allocation_failed) {
    const auto commitment = cuda_kv_pool_.vram_budget_bytes();
    const size_t allowed = allocation_failed ? 0 : commitment - std::min(commitment, target_kv_bytes);
    if (expert_slab_bytes_ <= allowed) return;
    ctx.synchronize();
    cuda_workspace_.decode_graph.reset();
    cuda_workspace_.graph_captured = false;
    int evicted = 0;
    for (size_t i = expert_slabs_.size(); i > 0 && expert_slab_bytes_ > allowed;) {
        --i;
        auto& slab = expert_slabs_[i];
        if (!slab.storage) continue;
        auto& layer = moe_layers_[i];
        auto* ptr = slab.storage.release();
        FUSIONRT_CUDA_CHECK(cudaFree(ptr));
        expert_slab_bytes_ -= slab.bytes;
        slab.bytes = 0;
        layer.cfg.gpu_resident_experts -= static_cast<int>(layer.borrowed_gate_up_w.shape[0]);
        layer.borrowed_gate_up_w = {};
        layer.borrowed_up_w = {};
        layer.borrowed_down_w = {};
        ++evicted;
    }
    std::clog << "MoE slabs evicted=" << evicted << " borrowed_bytes=" << expert_slab_bytes_
              << " target_kv_bytes=" << target_kv_bytes << '\n';
}

void FusionModel::ensure_cuda_kv(CudaContext& ctx, int first_write, int tokens) {
    auto& slot = cuda_session();
    std::function<void(size_t, bool)> reclaim;
    if (expert_slab_bytes_) reclaim = [&](size_t target, bool failed) { reclaim_cuda_expert_slabs(ctx, target, failed); };
    if (cuda_slots_.size() == 1 || (!may_evict_paused_cuda_slots_ && cuda_kv_pool_.host_budget_bytes() == 0)) {
        cuda_kv_pool_.ensure(ctx, slot.kv, slot.pages, first_write, tokens, {}, reclaim);
        return;
    }
    if (cuda_kv_pool_.host_budget_bytes() == 0) {
        try {
            cuda_kv_pool_.ensure(ctx, slot.kv, slot.pages, first_write, tokens, {}, reclaim);
            return;
        } catch (const KvVramExhausted&) {
            // A paused prefix can be evicted and prefilled again when the active request needs its pages.
        }
    }
    std::vector<CudaSession*> paused;
    for (const auto& candidate : cuda_slots_) {
        if (candidate && candidate.get() != &slot) paused.push_back(candidate.get());
    }
    std::sort(paused.begin(), paused.end(), [](const CudaSession* a, const CudaSession* b) {
        return a->last_used < b->last_used;
    });
    std::vector<CudaKvPool::Table*> victims;
    victims.reserve(paused.size());
    for (auto* candidate : paused) victims.push_back(&candidate->pages);
    for (size_t evicted = 0;; ++evicted) {
        try {
            cuda_kv_pool_.ensure(ctx, slot.kv, slot.pages, first_write, tokens,
                cuda_kv_pool_.host_budget_bytes() ? std::span(victims).subspan(evicted) :
                                                  std::span<CudaKvPool::Table*>{}, reclaim);
            break;
        } catch (const KvVramExhausted&) {
            if (!may_evict_paused_cuda_slots_ || evicted >= paused.size()) throw;
            auto& victim = *paused[evicted];
            cuda_kv_pool_.truncate(victim.pages, 0);
            victim.cached_tokens.clear();
            victim.checkpoints.clear();
        }
    }
}

void FusionModel::save_cuda_checkpoint(CudaContext& ctx, int tokens, bool logits) {
    auto& slot = cuda_session();
    if (slot.gdn.empty()) return;
    constexpr size_t max_checkpoints = 6;
    auto& checkpoints = slot.checkpoints;
    std::erase_if(checkpoints, [=](const auto& entry) { return entry.tokens == tokens; });
    if (checkpoints.size() == max_checkpoints) checkpoints.erase(checkpoints.begin());
    auto snapshot = std::make_shared<GdnSnapshot>();
    const size_t logits_bytes = logits ? cuda_workspace_.logits.nbytes() : 0;
    uint8_t* data = nullptr;
    FUSIONRT_CUDA_CHECK(cudaMalloc(&data, slot.ssm_bytes + logits_bytes));
    snapshot->data.reset(data);
    snapshot->has_logits = logits;
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data, slot.gdn_data, slot.ssm_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
    if (logits) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data + slot.ssm_bytes, cuda_workspace_.logits.ptr(), logits_bytes,
                                            cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    checkpoints.push_back({tokens, std::move(snapshot)});
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
                               DType kv_dtype) {
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
        case OpKind::GEMMA_PLE_GATE:
        case OpKind::GEGLU_INTERLEAVED:
        case OpKind::SWIGLU_INTERLEAVED:
            work.bytes = 3 * f32 * m * n;
            work.flops = 5 * m * n;
            break;
        case OpKind::GEGLU_Q8_LINEAR:
        case OpKind::SWIGLU_Q8_LINEAR:
            work.bytes = 2 * f32 * m * k;
            work.flops = 5 * m * k;
            linear(*node.weight, false);
            break;
        case OpKind::MLA_ATTN_BLOCK: {
            const auto& d = dense->cfg;
            const int64_t heads = d.num_attention_heads, rank = d.mla.latent_rank, width = d.mla.latent_width();
            const int64_t visible = std::max<int64_t>(1, pos + 1);
            const int64_t projections = rank * (d.head_dim - d.mla.rope_dims + d.mla.value_dim);
            work.flops += 2 * m * heads * (projections + visible * (width + rank));
            work.bytes += m * (dense->mla_k_b.nbytes() + dense->mla_v_b.nbytes() +
                              f32 * heads * visible * width);
            break;
        }
        case OpKind::GQA_ATTN_BLOCK: {
            const int64_t hd = node.attention.head_dim;
            const int64_t pairs = m * pos + m * (m + 1) / 2;
            work.flops = n * pairs * (4 * hd + 4) + 3 * m * (n + k) * node.attention.rotary_dims;
            work.bytes = f32 * hd * 2 * m * (n + k) + row_nbytes(kv_dtype, hd) * 2 * (m + pairs) * k;
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

struct CudaIRSegment {
    int first, end;
    int input_value;
    int boundary_value = -1;
    Tensor* boundary = nullptr;
};

template<bool Observe = false>
static void execute_ir_graph_cuda(
    const MiniGraph& g, CudaIRWorkspace& ws, CudaContext& ctx,
    const Tensor& x_input, const Tensor& token_ids, Tensor& logits_buf,
    int M, int pos_host, int start_pos,
    const int* pos_dev, const int* kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvView>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    bool use_fused, OpTrace* op_trace = nullptr, CudaOpTimer* timer = nullptr,
    LayerTrace* layer_trace = nullptr, bool compute_logits = true, bool all_logits = false,
    const CudaIRSegment* segment = nullptr) {
    size_t mark = ctx.scratch_arena().mark();
    int H = g.hidden_size;
    const int first_node = segment ? segment->first : 0;
    const int num_nodes = segment ? segment->end : static_cast<int>(g.nodes.size());
    const auto& last_use = ws.last_use;

    Tensor ple;
    if (const auto* embedding = g.gemma_inputs.token_embeddings) {
        const int width = static_cast<int>(g.gemma_inputs.norm->shape[0]);
        const int cols = static_cast<int>(embedding->shape[1]);
        ple = cuda_scratch(ctx, {M, cols});
        const auto ple_mark = ctx.scratch_arena().mark();
        auto projected = cuda_scratch(ctx, {M, cols});
        auto normalized = cuda_scratch(ctx, {M, cols});
        dispatch_linear(ctx, x_input.ptr<float>(), *g.gemma_inputs.projection, projected.ptr<float>(), M, cols, H, use_fused, all_logits);
        // Absorb the 1/sqrt(H) projection scale into the RMSNorm epsilon.
        cuda_ops::rmsnorm_f32(projected.ptr<float>(), g.gemma_inputs.norm->ptr<float>(), normalized.ptr<float>(),
                              M * cols / width, width, g.gemma_input_eps * H, ctx.stream());
        cuda_ops::embed_lookup_f32(embedding->ptr(), embedding->dtype, token_ids.ptr<int32_t>(),
                                  ple.ptr<float>(), M, cols, ctx.stream());
        cuda_ops::scale_f32(ple.ptr<float>(), M * cols, std::sqrt(static_cast<float>(width)), ctx.stream());
        cuda_ops::add_residual_f32(ple.ptr<float>(), normalized.ptr<float>(), ple.ptr<float>(), M * cols, ctx.stream());
        cuda_ops::scale_f32(ple.ptr<float>(), M * cols, 0.7071067811865475f, ctx.stream());
        ctx.scratch_arena().reset(ple_mark);
    }

    std::array<Tensor, kPrefillCarrySlots> carry_bufs;
    std::array<int, kPrefillCarrySlots> carry_busy_until;
    carry_busy_until.fill(-1);
    for (int s = 0; s < kPrefillCarrySlots; ++s) {
        carry_bufs[s] = cuda_scratch(ctx, {M, H});
    }
    size_t layer_reset_mark = ctx.scratch_arena().mark();
    int non_carry_busy_until = -1;

    auto& bufs = ws.bufs;
    std::fill(bufs.begin(), bufs.end(), Tensor{});
    bufs[segment ? segment->input_value : g.input_value] =
        Tensor::make_contiguous({M, H}, DType::F32, Backend::CUDA, x_input.ptr());
    auto& pa = prof_acc();

    for (int ni = first_node; ni < num_nodes; ++ni) {
        const IRNode& node = g.nodes[ni];
        if (!node.alive) continue;
        if (ni > non_carry_busy_until) {
            ctx.scratch_arena().reset(layer_reset_mark);
            non_carry_busy_until = -1;
        }
        const bool is_lm_head = node.output_logits;
        if (is_lm_head && !compute_logits && node.residual_out < 0) continue;
        const bool tail_logits = is_lm_head && compute_logits && !all_logits;
        const int rows = tail_logits ? 1 : M;

        auto get_out_buf = [&](int vid, int cols, int r_cnt) -> Tensor& {
            if (vid == g.output_value && is_lm_head) {
                bufs[vid] = logits_buf;
                return bufs[vid];
            }
            if (bufs[vid].data != nullptr) return bufs[vid];
            if (cols == H && r_cnt == M && !is_lm_head) {
                for (int s = 0; s < kPrefillCarrySlots; ++s) {
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
                float* res = add ? get_out_buf(node.residual_out, node.input_cols, rows).template ptr<float>() : nullptr;
                const float* weight = node.norm_weight->ptr<float>();
                if (node.share_quantized_activation && (rows == 1 || rows >= ctx.mmq_crossover_m())) {
                    bufs[node.out] = ctx.scratch_arena().alloc_tensor({rows, node.input_cols}, DType::Q8_1);
                    non_carry_busy_until = std::max(non_carry_busy_until, last_use[node.out]);
                    cuda_ops::quantize_x_q8_1(a, b, weight, res, bufs[node.out].ptr(), rows,
                                              node.input_cols, node.norm_eps, ctx.stream());
                } else {
                    float* out = get_out_buf(node.out, node.output_cols, rows).template ptr<float>();
                    if (add) cuda_ops::add_rmsnorm_f32(a, b, weight, res, out, rows, node.input_cols,
                                                       node.norm_eps, ctx.stream());
                    else cuda_ops::rmsnorm_f32(a, weight, out, rows, node.input_cols, node.norm_eps, ctx.stream());
                }
                break;
            }
            case OpKind::SHARED_Q8_PROJECTIONS:
            case OpKind::LINEAR: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* in_p = row_ptr(node.inputs[0], tail_logits);
                auto fn = [&]() {
                    const bool packed = bufs[node.inputs[0]].dtype == DType::Q8_1;
                    auto project = [&](const Tensor& w, float* out, int cols) {
                        if (!packed) {
                            dispatch_linear(ctx, in_p, w, out, rows, cols, node.input_cols, use_fused, all_logits);
                        } else if (rows == 1) {
                            decode_gemv(w.dtype)(in_p, w.ptr(), out, cols, node.input_cols,
                                                  ctx.stream(), ctx.preferred_gemv_geometry());
                        } else {
                            prefill_mmq(w.dtype)(in_p, w.ptr(), out, rows, cols, node.input_cols,
                                                 ctx.stream(), ctx.cost_hardware());
                        }
                    };
                    const Tensor& w = *node.weight;
                    if (node.secondary_weight) {
                        const Tensor& other = *node.secondary_weight;
                        const int cols = static_cast<int>(other.shape[0]);
                        auto* extra = get_out_buf(node.secondary_out, cols, rows).template ptr<float>();
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
                const float* in_p = row_ptr(node.inputs[0], tail_logits);
                auto fn = [&]() {
                    dispatch_rmsnorm_linear(ctx, in_p, *node.norm_weight, *node.weight, out_t.ptr<float>(),
                                            rows, node.output_cols, node.input_cols, node.norm_eps, use_fused, all_logits);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::ADD_RESIDUAL: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const int count = rows * node.output_cols;
                cuda_ops::add_residual_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                                           out_t.ptr<float>(), count, ctx.stream());
                if (node.output_scale != 1.0f) cuda_ops::scale_f32(out_t.ptr<float>(), count, node.output_scale, ctx.stream());
                break;
            }
            case OpKind::FUSED_ADD_RMSNORM_LINEAR: {
                if (is_lm_head && !compute_logits) {
                    Tensor& residual = get_out_buf(node.residual_out, node.input_cols, M);
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
                const float* a_p = row_ptr(node.inputs[0], tail_logits);
                const float* b_p = row_ptr(node.inputs[1], tail_logits);
                auto fn = [&]() {
                    dispatch_add_rmsnorm_linear(ctx, a_p, b_p, *node.norm_weight, *node.weight, res_upd_p,
                                                out_t.ptr<float>(), rows, node.output_cols, node.input_cols, node.norm_eps, use_fused, all_logits);
                };
                if (pa.enabled && is_lm_head && M > 1) pa.lm_ms += time_cuda_fn(ctx.stream(), fn);
                else fn();
                break;
            }
            case OpKind::MLA_ATTN_BLOCK: {
                const auto& layer = dense_layers[node.layer_index];
                const auto& d = layer.cfg;
                const int heads = d.num_attention_heads;
                const auto& mla = d.mla;
                auto& output = get_out_buf(node.out, node.output_cols, rows);
                auto query = cuda_scratch(ctx, {rows, heads, mla.latent_width()});
                auto latent = cuda_scratch(ctx, {rows, heads, mla.latent_rank});
                cuda_ops::mla_attention_f32(bufs[node.inputs[0]].ptr<float>(), bufs[node.inputs[1]].ptr<float>(),
                    layer.mla_kv_a_norm.ptr<float>(), layer.mla_k_b, layer.mla_v_b, layer.rope_factors.ptr<float>(),
                    query.ptr<float>(), latent.ptr<float>(), output.template ptr<float>(), kv_states[node.layer_index],
                    rows, heads, d.head_dim, start_pos, pos_dev, d.rms_norm_eps, mla.attention_scale,
                    mla.rope_amplitude, ctx.stream());
                break;
            }
            case OpKind::GQA_ATTN_BLOCK: {
                auto run_gqa = [&]() {
                    const auto& L = dense_layers[static_cast<size_t>(node.layer_index)];
                    const auto& cfg = node.attention;
                    const bool shared = L.kv_source_layer >= 0;
                    const auto& state = kv_states[shared ? L.kv_source_layer : node.layer_index];
                    const int nq = cfg.query_heads, nkv = cfg.kv_heads, hd = cfg.head_dim;
                    const int q_dim = nq * hd, q_proj_dim = cfg.gated ? 2 * q_dim : q_dim;
                    const int kv_dim = nkv * hd;
                    float* q_full_p = bufs[node.inputs[0]].ptr<float>() + node.input_offsets[0];
                    const bool has_value = node.inputs.size() == 3;
                    float* k_p = shared ? nullptr : bufs[node.inputs[1]].ptr<float>() + node.input_offsets[1];
                    float* v_p = has_value ? bufs[node.inputs[2]].ptr<float>() + node.input_offsets[2] : nullptr;
                    int q_stride = g.values[node.inputs[0]].cols;
                    const int k_stride = shared ? 0 : g.values[node.inputs[1]].cols;
                    const int v_stride = has_value ? g.values[node.inputs[2]].cols : 0;
                    const int project_kv_heads = shared ? 0 : nkv;
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
                    const auto* next = ni + 1 < num_nodes ? &g.nodes[ni + 1] : nullptr;
                    const bool quantize_output = use_fused && rows == 1 && hd % 32 == 0 && hd <= 256 &&
                        next && next->alive && next->op == OpKind::LINEAR && next->inputs[0] == node.out &&
                        last_use[node.out] == ni + 1 && decode_gemv(next->weight->dtype);
                    if (quantize_output) {
                        bufs[node.out] = cuda_scratch(ctx, {q_dim}, DType::Q8_1);
                        non_carry_busy_until = std::max(non_carry_busy_until, last_use[node.out]);
                    }
                    Tensor& attn_out = quantize_output ? bufs[node.out] : get_out_buf(node.out, q_dim, rows);
                    const float scale = L.attention_scale > 0.0f ? L.attention_scale : 1.0f / std::sqrt(static_cast<float>(hd));
                    const float* qnw = cfg.query_norm ? cfg.query_norm->ptr<float>() : nullptr;
                    const float* knw = cfg.key_norm ? cfg.key_norm->ptr<float>() : nullptr;
                    const float* rotary = cfg.rotary_factors ? cfg.rotary_factors->ptr<float>() : nullptr;
                    if (rows == 1) {
                        cuda_ops::fused_qk_norm_rope_kv_append_f32(
                            q_p, k_p, v_p, qnw, knw,
                            nullptr, nullptr,
                            nq, project_kv_heads, hd, cfg.rotary_dims, pos_dev, pos_host, cfg.max_context,
                            cfg.rope_theta, node.norm_eps, cfg.rope_neox, ctx.stream(), rotary, state.dtype, state, L.v_norm);
                        const int split_heads = g.decode_split_kv_heads ? g.decode_split_kv_heads : nkv;
                        const int splits = select_decode_splits(ctx.cost_hardware(), pos_host + 1, split_heads, kMaxDecodeSplits);
                        Tensor part_out = cuda_scratch(ctx, {nq, splits, hd});
                        Tensor part_ml = cuda_scratch(ctx, {nq, splits, 2});
                        // Use the same reduction order for eager and captured decode.
                        cuda_ops::gqa_splitk_decode_f32(
                            q_p, nullptr, nullptr,
                            attn_out.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                            nq, nkv, hd, kv_len_dev, pos_host + 1, cfg.max_context,
                            scale, splits, ctx.stream(), true, state.dtype,
                            quantize_output ? attn_out.ptr<BlockQ8_1>() : nullptr, gate_p, state, L.sliding_window);
                    } else {
                        cuda_ops::batched_qk_norm_rope_kv_append_f32(
                            q_p, k_p, v_p, qnw, knw,
                            nullptr, nullptr,
                            rows, nq, project_kv_heads, hd, cfg.rotary_dims, start_pos, cfg.max_context,
                            cfg.rope_theta, node.norm_eps, cfg.rope_neox, ctx.stream(), q_stride, k_stride, v_stride,
                            rotary, state.dtype, state, L.v_norm);
                        if ((hd == 64 || hd == 128 || hd == 256) && start_pos + rows >= 512) {
                            cuda_ops::gqa_prefill_online_f32(q_p, q_stride, state, attn_out.ptr<float>(),
                                rows, nq, nkv, hd, start_pos, scale, ctx.stream(), L.sliding_window);
                        } else {
                            cuda_ops::gqa_prefill_causal_f32(q_p, nullptr, nullptr, attn_out.ptr<float>(),
                                rows, nq, nkv, hd, start_pos, cfg.max_context, scale, ctx.stream(),
                                q_stride, state.dtype, state, L.sliding_window);
                        }
                    }
                    if (node.attention.gated && !quantize_output) {
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
                                cfg.tiled_value_heads, L.cfg.conv_kernel_size);
                        } else {
                            Tensor sqkv = cuda_scratch(ctx, {conv_dim});
                            Tensor sy = cuda_scratch(ctx, {v_dim});
                            cuda_ops::unfused_gated_deltanet_step_f32(
                                qkv_p, z_p, a_p, b_p,
                                L.conv_w.ptr<float>(), L.conv_bias.ptr<float>(), L.A_log.ptr<float>(), L.dt_bias.ptr<float>(),
                                L.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                                gdn_out.ptr<float>(), sqkv.ptr<float>(), sy.ptr<float>(),
                                nk, nv, dk, dv, node.norm_eps, q_scale, ctx.stream(), cfg.tiled_value_heads, L.cfg.conv_kernel_size);
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
                            gdn_ws.ptr<float>(), gdn_ws.numel(), cfg.tiled_value_heads, L.cfg.conv_kernel_size);
                    }
                };
                if (pa.enabled && M > 1) pa.gdn_ms += time_cuda_fn(ctx.stream(), run_gdn);
                else run_gdn();
                break;
            }
            case OpKind::GEMMA_PLE_GATE: {
                const int width = node.output_cols;
                auto& out = get_out_buf(node.out, width, rows);
                cuda_ops::gemma_ple_gate_f32(bufs[node.inputs[0]].ptr<float>(), ple.ptr<float>(),
                    out.template ptr<float>(), rows, width, static_cast<int>(ple.shape[1]),
                    node.layer_index * width, ctx.stream());
                break;
            }
            case OpKind::GEGLU_INTERLEAVED:
            case OpKind::SWIGLU_INTERLEAVED: {
                Tensor& out_t = get_out_buf(node.out, node.output_cols, rows);
                const float* gate = bufs[node.inputs[0]].ptr<float>();
                const bool geglu = node.op == OpKind::GEGLU_INTERLEAVED;
                if (node.inputs.size() == 1) {
                    (geglu ? cuda_ops::geglu_interleaved_f32 : cuda_ops::swiglu_interleaved_f32)(gate, out_t.ptr<float>(), rows, node.output_cols, ctx.stream());
                } else {
                    (geglu ? cuda_ops::geglu_f32 : cuda_ops::swiglu_f32)(gate, bufs[node.inputs[1]].ptr<float>(), out_t.ptr<float>(),
                                        rows * node.output_cols, ctx.stream());
                }
                break;
            }
            case OpKind::GEGLU_Q8_LINEAR:
            case OpKind::SWIGLU_Q8_LINEAR: {
                const bool geglu = node.op == OpKind::GEGLU_Q8_LINEAR;
                const auto quantize = geglu ? cuda_ops::geglu_interleaved_q8_1 : cuda_ops::swiglu_interleaved_q8_1;
                const float* gate_up = bufs[node.inputs[0]].ptr<float>();
                float* out = get_out_buf(node.out, node.output_cols, rows).template ptr<float>();
                auto& arena = ctx.scratch_arena();
                size_t scratch_mark = arena.mark();
                if (rows == 1) {
                    void* q8 = arena.alloc(row_nbytes(DType::Q8_1, node.input_cols));
                    quantize(gate_up, q8, rows, node.input_cols, ctx.stream());
                    decode_gemv(node.weight->dtype)(q8, node.weight->ptr(), out, node.output_cols, node.input_cols, ctx.stream(),
                                               ctx.preferred_gemv_geometry());
                } else if (auto mmq = rows >= ctx.mmq_crossover_m() ? prefill_mmq(node.weight->dtype) : nullptr) {
                    void* q8 = arena.alloc(static_cast<size_t>(rows) * row_nbytes(DType::Q8_1, node.input_cols));
                    quantize(gate_up, q8, rows, node.input_cols, ctx.stream());
                    mmq(q8, node.weight->ptr(), out, rows, node.output_cols, node.input_cols,
                        ctx.stream(), ctx.cost_hardware());
                } else {
                    Tensor act = cuda_scratch(ctx, {rows, node.input_cols});
                    (geglu ? cuda_ops::geglu_interleaved_f32 : cuda_ops::swiglu_interleaved_f32)(gate_up, act.ptr<float>(), rows, node.input_cols, ctx.stream());
                    dispatch_linear(ctx, act.ptr<float>(), *node.weight, out, rows, node.output_cols, node.input_cols, use_fused, all_logits);
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
                const bool trace_routes = ctx.moe_route_trace && ctx.moe_route_trace->enabled_for(&moe_layers);
                const int index = trace_routes || (rows == 1 && !(segment && segment->boundary)) ? node.layer_index : -1;
                L.forward_prefill_cuda(ctx, x_v, rows, use_fused, index, trace_routes ? start_pos : -1);
                break;
            }
        }
        if (is_lm_head && compute_logits && g.final_logit_softcapping > 0.0f) {
            cuda_ops::softcap_f32(logits_buf.ptr<float>(), rows * node.output_cols, g.final_logit_softcapping, ctx.stream());
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
                        node.op == OpKind::GQA_ATTN_BLOCK || node.op == OpKind::MLA_ATTN_BLOCK
                            ? &dense_layers[node.layer_index] : nullptr,
                        node.op == OpKind::MOE_FFN_BLOCK ? &moe_layers[node.layer_index] : nullptr,
                        node.op == OpKind::GQA_ATTN_BLOCK ? kv_states[dense_layers[node.layer_index].kv_source_layer >= 0
                            ? dense_layers[node.layer_index].kv_source_layer : node.layer_index].dtype : DType::F32);
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
    if (segment && segment->boundary) {
        const auto& output = bufs[segment->boundary_value];
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(segment->boundary->ptr(), output.ptr(), output.nbytes(),
                                            cudaMemcpyDeviceToDevice, ctx.stream()));
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

static bool can_use_cuda_graph(const std::vector<CudaSparseMoELayer>& layers) {
    return std::all_of(layers.begin(), layers.end(), [](const auto& layer) {
        return layer.cfg.gpu_resident_experts == layer.cfg.num_experts;
    });
}

static std::optional<std::pair<int, int>> moe_graph_prefix(const MiniGraph& graph,
                                                          const std::vector<CudaSparseMoELayer>& layers) {
    if (graph.gemma_inputs.token_embeddings) return std::nullopt;
    std::optional<std::pair<int, int>> prefix;
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        const auto& node = graph.nodes[i];
        if (!node.alive || node.op != OpKind::MOE_FFN_BLOCK) continue;
        const auto& cfg = layers[node.layer_index].cfg;
        if (cfg.gpu_resident_experts != cfg.num_experts) return prefix;
        prefix = {i + 1, node.out};
    }
    return std::nullopt;
}

static void embed_tokens(CudaContext& ctx, const Tensor& table, const Tensor& ids, Tensor& hidden,
                         int rows, int width, float scale) {
    cuda_ops::embed_lookup_f32(table.ptr(), table.dtype, ids.ptr<int32_t>(), hidden.ptr<float>(),
                               rows, width, ctx.stream());
    if (scale != 1.0f) cuda_ops::scale_f32(hidden.ptr<float>(), rows * width, scale, ctx.stream());
}

static bool run_forward_token_step_cuda(
    CudaContext& ctx, const HybridModelConfig& cfg,
    const Tensor& tok_embeddings, const MiniGraph& ir, CudaIRWorkspace& ws,
    Tensor& hidden_buf, Tensor& logits_buf, Tensor& tok_id_dev, Tensor& pos_dev, Tensor& kv_len_dev,
    const std::vector<CudaDenseGQALayer>& dense_layers,
    const std::vector<CudaGatedDeltaNetLayer>& gdn_layers,
    const std::vector<CudaKvView>& kv_states, const std::vector<CudaGdnState>& gdn_states,
    const std::vector<CudaSparseMoELayer>& moe_layers,
    CudaGraphExec& decode_graph, bool& graph_captured, bool& graph_fused_flag, bool& graph_has_argmax,
    int& graph_split_bucket,
    int pos, bool use_fused, bool use_graph, bool with_argmax, bool first_step = true,
    OpTrace* trace = nullptr, CudaOpTimer* timer = nullptr) {
    int H = static_cast<int>(hidden_buf.shape[0]);

    auto run_body = [&](bool graph_mode) {
        embed_tokens(ctx, tok_embeddings, tok_id_dev, hidden_buf, 1, H, cfg.embed_scale);
        int* p_dev = graph_mode ? pos_dev.ptr<int>() : nullptr;
        int* kl_dev = graph_mode ? kv_len_dev.ptr<int>() : nullptr;
        if (timer) {
            execute_ir_graph_cuda<true>(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos, p_dev, kl_dev,
                                        dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused, trace, timer);
        } else {
            execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos, p_dev, kl_dev,
                                  dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused);
        }
        if (with_argmax) {
            decode_argmax(ctx, logits_buf, tok_id_dev, cfg.vocab_size, p_dev, kl_dev);
        }
    };

    if (use_graph && can_use_cuda_graph(moe_layers)) {
        // The generation graph advances both counters after it writes the next token.
        if (first_step || !with_argmax) {
            int kv_len = pos + 1;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr(), &pos, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(kv_len_dev.ptr(), &kv_len, sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
        }
        const int bucket = select_decode_splits(ctx.cost_hardware(), pos + 1,
                                                cfg.dense.num_key_value_heads, kMaxDecodeSplits);
        if (!graph_captured || graph_fused_flag != use_fused || graph_has_argmax != with_argmax ||
            graph_split_bucket != bucket) {
            decode_graph.capture(ctx.stream(), [&]() { run_body(true); });
            graph_captured = true;
            graph_fused_flag = use_fused;
            graph_has_argmax = with_argmax;
            graph_split_bucket = bucket;
        }
        decode_graph.launch(ctx.stream());
        return true;
    }
    if (use_graph && !timer) {
        if (const auto prefix = moe_graph_prefix(ir, moe_layers)) {
            const auto [cut, boundary] = *prefix;
            const int kv_len = pos + 1;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr(), &pos, sizeof(pos), cudaMemcpyHostToDevice, ctx.stream()));
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(kv_len_dev.ptr(), &kv_len, sizeof(kv_len), cudaMemcpyHostToDevice, ctx.stream()));
            const int bucket = select_decode_splits(ctx.cost_hardware(), kv_len, cfg.dense.num_key_value_heads, kMaxDecodeSplits);
            if (!graph_captured || graph_fused_flag != use_fused || graph_has_argmax != with_argmax ||
                graph_split_bucket != bucket) {
                const CudaIRSegment captured{0, cut, ir.input_value, boundary, &hidden_buf};
                decode_graph.capture(ctx.stream(), [&] {
                    embed_tokens(ctx, tok_embeddings, tok_id_dev, hidden_buf, 1, H, cfg.embed_scale);
                    execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos,
                        pos_dev.ptr<int>(), kv_len_dev.ptr<int>(), dense_layers, gdn_layers, kv_states, gdn_states,
                        moe_layers, use_fused, nullptr, nullptr, nullptr, false, false, &captured);
                });
                graph_captured = true;
                graph_fused_flag = use_fused;
                graph_has_argmax = with_argmax;
                graph_split_bucket = bucket;
            }
            if (ctx.moe_timing) {
                for (int i = 0; i < cut; ++i) {
                    const auto& node = ir.nodes[i];
                    if (node.alive && node.op == OpKind::MOE_FFN_BLOCK)
                        ctx.moe_timing->layer(node.layer_index).hot += moe_layers[node.layer_index].cfg.top_k;
                }
            }
            decode_graph.launch(ctx.stream());
            const CudaIRSegment eager{cut, static_cast<int>(ir.nodes.size()), boundary};
            execute_ir_graph_cuda(ir, ws, ctx, hidden_buf, tok_id_dev, logits_buf, 1, pos, pos, nullptr, nullptr,
                dense_layers, gdn_layers, kv_states, gdn_states, moe_layers, use_fused,
                nullptr, nullptr, nullptr, true, false, &eager);
            if (with_argmax) decode_argmax(ctx, logits_buf, tok_id_dev, cfg.vocab_size);
            return true;
        }
    }
    run_body(false);
    return false;
}

std::vector<float> FusionModel::forward_token_cuda(CudaContext& ctx, int32_t token_id, int pos,
                                                   bool use_fused, bool use_graph) {
    if (pos < 0 || pos >= cfg_.dense.max_seq_len) throw std::invalid_argument("Token position exceeds context");
    validate_token_id(token_id, pos, cfg_.vocab_size);
    clear_cuda_history();
    ensure_cuda_kv(ctx, pos, pos + 1);
    const MiniGraph& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_workspace_.token_ids.ptr(), &token_id, sizeof(token_id),
                                        cudaMemcpyHostToDevice, ctx.stream()));
    run_forward_token_step_cuda(ctx, cfg_, tok_embeddings_, ir, ws,
                                cuda_workspace_.hidden, cuda_workspace_.logits, cuda_workspace_.token_ids, cuda_workspace_.position, cuda_workspace_.kv_length,
                                dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn, moe_layers_,
                                cuda_workspace_.decode_graph, cuda_workspace_.graph_captured, cuda_workspace_.graph_fused, cuda_workspace_.graph_has_argmax,
                                cuda_workspace_.graph_split_bucket,
                                pos, use_fused, use_graph, false);
    HostTensor h_logits = ctx.download(cuda_workspace_.logits);
    return std::vector<float>(h_logits.data<float>(), h_logits.data<float>() + cfg_.vocab_size);
}

int FusionModel::prefill_batch_limit(CudaContext& ctx, bool verify) const {
    const auto& d = cfg_.dense;
    const auto& g = cfg_.gdn;
    const int q = d.num_attention_heads * d.head_dim;
    const int kv = d.num_key_value_heads * d.head_dim;
    int64_t dense_extra = std::max<int64_t>(3LL * d.intermediate_size, 4LL * q + 2LL * kv);
    if (d.mla.latent_rank > 0) {
        const int64_t heads = d.num_attention_heads;
        dense_extra = std::max<int64_t>(dense_extra, q + d.mla.latent_width() + 2LL * d.mla.query_rank +
            heads * (d.mla.latent_width() + d.mla.latent_rank + d.mla.value_dim));
    }
    if (cfg_.model_architecture == "gemma4") {
        for (const auto& layer : dense_layers_) {
            const auto& a = layer.cfg;
            dense_extra = std::max({dense_extra, int64_t{3} * a.intermediate_size,
                (int64_t{4} * a.num_attention_heads + int64_t{2} * a.num_key_value_heads) * a.head_dim});
        }
    }
    const auto& arena = ctx.scratch_arena();
    const size_t available = arena.capacity() - arena.used();
    int rows = std::min(ctx.prefill_batch(), d.max_seq_len);
    for (; rows > 1; rows /= 2) {
        const size_t layer_rows = static_cast<size_t>(rows);
        const size_t dense = dense_extra * layer_rows;
        size_t recurrent = 0;
        if (!gdn_layers_.empty()) {
            const int recurrent_cols = 2 * g.num_k_heads * g.head_k_dim + 3 * g.num_v_heads * g.head_v_dim +
                                       2 * g.num_v_heads;
            recurrent = std::max(3 * g.intermediate_size * layer_rows, recurrent_cols * layer_rows +
                cuda_ops::gated_deltanet_prefill_scratch_floats(rows, g.num_k_heads, g.num_v_heads,
                                                              g.head_k_dim, g.head_v_dim));
        }
        const size_t moe = moe_layers_.empty() ? 0 :
            (4 * static_cast<size_t>(d.hidden_size) + cfg_.moe.num_experts + 2 * cfg_.moe.top_k +
             3 * cfg_.moe.shared_intermediate_size) * layer_rows +
            static_cast<size_t>(cfg_.moe.top_k) * cfg_.moe.intermediate_size;
        const auto layer = std::max({dense, recurrent, moe});
        const size_t ple = gemma_inputs_.token_embeddings.data ?
            static_cast<size_t>(gemma_inputs_.token_embeddings.shape[1]) * layer_rows : 0;
        const size_t live = std::max((kPrefillCarrySlots + 2) * d.hidden_size * layer_rows + layer + ple,
                                    d.hidden_size * layer_rows + 3 * ple) * sizeof(float);
        const size_t verify_logits = verify ? layer_rows * cfg_.vocab_size * sizeof(float) : 0;
        if (live + verify_logits + 64 * 1024 <= available) break;
    }
    return rows;
}

void FusionModel::set_device_profile(CudaContext& ctx, const DeviceProfile& profile) {
    ctx.set_device_profile(profile);
    recommend_cuda_graph_ = profile.recommend_cuda_graph;
    cuda_workspace_.graph_captured = false;
    for (const auto& slot : cuda_slots_) {
        if (!slot) continue;
        slot->cached_tokens.clear();
        slot->checkpoints.clear();
    }
}

ModelMemory FusionModel::cuda_cache_memory() const {
    ModelMemory result;
    result.kv_pages = cuda_kv_pool_.pages();
    result.kv_bytes = cuda_kv_pool_.bytes();
    result.kv_reserved_bytes = cuda_kv_pool_.reserved_bytes();
    result.host_kv_bytes = cuda_kv_pool_.host_bytes();
    result.host_kv_pages = cuda_kv_pool_.host_pages();
    std::unordered_set<const GdnSnapshot*> counted;
    for (const auto& slot : cuda_slots_) {
        if (!slot) continue;
        result.ssm_bytes += slot->ssm_bytes;
        result.persistent_bytes += slot->storage.capacity();
        for (const auto& checkpoint : slot->checkpoints) {
            if (!counted.insert(checkpoint.snapshot.get()).second) continue;
            result.ssm_bytes += slot->ssm_bytes;
            result.persistent_bytes += slot->ssm_bytes;
            if (checkpoint.snapshot->has_logits) result.persistent_bytes += cuda_workspace_.logits.nbytes();
        }
    }
    return result;
}

ModelMemory FusionModel::inspect_cuda(CudaContext& ctx) {
    ModelMemory result = cuda_cache_memory();
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

    const int rows = prefill_batch_limit(ctx);
    const std::vector<int32_t> tokens(rows, 0);
    auto& arena = ctx.scratch_arena();
    const size_t mark = arena.mark();
    auto probe_slot = make_cuda_session();
    CudaKvPool probe_pool;
    std::swap(probe_slot, cuda_slots_[active_slot_]);
    std::swap(probe_pool, cuda_kv_pool_);
    arena.track_peak(true);
    try {
        (void)prefill_cuda_impl(ctx, tokens.data(), rows, 0, true);
        reset_kv_and_state(ctx);
        (void)prefill_cuda_impl(ctx, tokens.data(), 1, 0, true);
        decode_argmax(ctx, cuda_workspace_.logits, cuda_workspace_.token_ids, cfg_.vocab_size);
        ctx.synchronize();
        result.static_scratch_bytes = arena.peak() - mark;
    } catch (...) {
        arena.track_peak(false);
        arena.reset(mark);
        std::swap(probe_pool, cuda_kv_pool_);
        std::swap(probe_slot, cuda_slots_[active_slot_]);
        throw;
    }
    arena.track_peak(false);
    std::swap(probe_pool, cuda_kv_pool_);
    std::swap(probe_slot, cuda_slots_[active_slot_]);
    return result;
}

static void validate_input_embeddings(std::span<const InputEmbedding> spans, size_t tokens, int width) {
    size_t end = 0;
    for (const auto& span : spans) {
        const auto& t = span.rows;
        if (span.start < 0 || static_cast<size_t>(span.start) < end || static_cast<size_t>(span.start) >= tokens ||
            t.ndim != 2 || t.shape[1] != width || t.shape[0] <= 0 ||
            static_cast<uint64_t>(t.shape[0]) > tokens - span.start ||
            t.dtype != DType::F32 || t.backend != Backend::CUDA || !t.data || !t.is_contiguous()) {
            throw std::invalid_argument("Invalid CUDA input embedding range or shape");
        }
        end = static_cast<size_t>(span.start) + static_cast<size_t>(t.shape[0]);
    }
}

std::vector<float> FusionModel::forward_prefill_cuda(CudaContext& ctx, const std::vector<int32_t>& token_ids,
                                                     int start_pos, bool use_fused,
                                                     std::span<const InputEmbedding> inputs_embeds) {
    validate_input_embeddings(inputs_embeds, token_ids.size(), static_cast<int>(tok_embeddings_.shape[1]));
    if (token_ids.empty()) return {};
    if (start_pos < 0) throw std::invalid_argument("Invalid prefill start position");
    const auto total_tokens = static_cast<size_t>(start_pos) + token_ids.size();
    if (total_tokens > static_cast<size_t>(cfg_.dense.max_seq_len)) {
        throw prompt_context_error(total_tokens, cfg_.dense.max_seq_len);
    }
    validate_token_ids(token_ids, start_pos, cfg_.vocab_size);
    clear_cuda_history();
    return prefill_cuda_impl(ctx, token_ids.data(), static_cast<int>(token_ids.size()), start_pos, use_fused,
                             nullptr, nullptr, false, inputs_embeds);
}

std::vector<float> FusionModel::prefill_cached_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt,
                                                    bool use_fused, int* prefilled_tokens, LayerTrace* trace,
                                                    OpTrace* op_trace) {
    if (prompt.empty()) throw std::invalid_argument("Prompt is empty");
    if (prompt.size() > static_cast<size_t>(cfg_.dense.max_seq_len))
        throw prompt_context_error(prompt.size(), cfg_.dense.max_seq_len);
    validate_token_ids(prompt, 0, cfg_.vocab_size);
    const int length = static_cast<int>(prompt.size());
    struct Match {
        size_t index = 0;
        int common = 0;
        int prefix = 0;
        bool append = false;
        uint64_t last_used = 0;
        std::shared_ptr<const GdnSnapshot> snapshot;
    } best;
    for (size_t i = 0; i < cuda_slots_.size(); ++i) {
        const auto& slot = cuda_slots_[i];
        if (!slot || slot->cached_fused != use_fused || slot->cached_tokens.empty()) continue;
        const int common = static_cast<int>(std::mismatch(prompt.begin(), prompt.end(), slot->cached_tokens.begin(),
                                                          slot->cached_tokens.end()).first - prompt.begin());
        const bool append = common == static_cast<int>(slot->cached_tokens.size());
        Match candidate{i, common, 0, append, slot->last_used, {}};
        if (slot->gdn.empty()) {
            candidate.prefix = std::min(common, length - 1);
        } else if (append && common < length) {
            candidate.prefix = common;
        } else {
            for (const auto& checkpoint : slot->checkpoints) {
                if (checkpoint.tokens > common || checkpoint.tokens <= candidate.prefix) continue;
                if (checkpoint.tokens == length && (!checkpoint.snapshot->has_logits || trace || op_trace)) continue;
                candidate.prefix = checkpoint.tokens;
                candidate.snapshot = checkpoint.snapshot;
            }
        }
        if (candidate.prefix > 0 || (candidate.append && candidate.common == length)) {
            if (std::tie(candidate.append, candidate.prefix, candidate.common, candidate.last_used) >
                std::tie(best.append, best.prefix, best.common, best.last_used)) {
                best = std::move(candidate);
            }
        }
    }

    size_t target = best.index;
    if (!best.append && cuda_slots_.size() > 1) {
        size_t oldest = cuda_slots_.size();
        for (size_t i = 0; i < cuda_slots_.size(); ++i) {
            if (best.prefix > 0 && i == best.index) continue;
            const auto& slot = cuda_slots_[i];
            if (!slot || slot->cached_tokens.empty()) {
                oldest = i;
                break;
            }
            if (oldest == cuda_slots_.size() || slot->last_used < cuda_slots_[oldest]->last_used) oldest = i;
        }
        target = oldest;
    }
    if (!cuda_slots_[target]) cuda_slots_[target] = make_cuda_session();
    if (active_slot_ != target) {
        cuda_workspace_.decode_graph.reset();
        cuda_workspace_.graph_captured = false;
        active_slot_ = target;
    }
    auto& slot = cuda_session();
    slot.cached_tokens.clear();
    slot.last_used = ++slot_clock_;
    const int prefix = best.prefix;
    if (target != best.index && prefix > 0) {
        const auto& source = *cuda_slots_[best.index];
        slot.checkpoints.clear();
        for (const auto& entry : source.checkpoints) {
            if (entry.tokens <= prefix) slot.checkpoints.push_back(entry);
        }
        cuda_kv_pool_.fork(slot.pages, source.pages, prefix);
    } else {
        std::erase_if(slot.checkpoints, [=](const auto& entry) { return entry.tokens > prefix; });
        cuda_kv_pool_.truncate(slot.pages, prefix);
    }
    if (best.snapshot) {
        const auto* data = best.snapshot->data.get();
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(slot.gdn_data, data, slot.ssm_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
        if (prefix == length) {
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_workspace_.logits.ptr(), data + slot.ssm_bytes,
                                                cuda_workspace_.logits.nbytes(), cudaMemcpyDeviceToDevice, ctx.stream()));
        }
    } else if (prefix == 0) {
        zero_cuda_gdn(ctx);
    }

    std::vector<float> logits;
    if (prefix == length) {
        ensure_cuda_kv(ctx, length, length);
        auto saved_logits = ctx.download(cuda_workspace_.logits);
        logits.assign(saved_logits.data<float>(), saved_logits.data<float>() + cfg_.vocab_size);
    } else {
        const bool checkpoint_pages = cuda_slots_.size() > 1 && !slot.gdn.empty();
        logits = prefill_cuda_impl(ctx, prompt.data() + prefix, length - prefix, prefix, use_fused, trace, op_trace,
                                   checkpoint_pages);
        save_cuda_checkpoint(ctx, length, true);
    }
    slot.cached_tokens = prompt;
    slot.cached_fused = use_fused;
    if (prefilled_tokens) *prefilled_tokens = length - prefix;
    return logits;
}

void FusionModel::execute_prefill_chunks(CudaContext& ctx, const int32_t* token_ids, int count,
                                         int start_pos, bool use_fused, LayerTrace* layer_trace,
                                         OpTrace* op_trace, bool compute_logits, bool save_prefix_checkpoints,
                                         std::span<const InputEmbedding> inputs_embeds) {
    const int H = static_cast<int>(tok_embeddings_.shape[1]);
    auto& arena = ctx.scratch_arena();
    struct RestoreArena {
        CudaArena& arena;
        size_t mark;
        ~RestoreArena() { arena.reset(mark); }
    } restore{arena, arena.mark()};

    ensure_cuda_kv(ctx, start_pos, start_pos + count);
    const int capacity = std::min(count, prefill_batch_limit(ctx));
    Tensor ids = cuda_scratch(ctx, {capacity}, DType::I32);
    Tensor hidden = cuda_scratch(ctx, {capacity, H});
    const auto& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    std::unique_ptr<CudaOpTimer> timer;
    if (op_trace) timer = std::make_unique<CudaOpTimer>(ir.nodes.size());
    auto& pa = prof_acc();

    const int end = start_pos + count;
    // Keep two recent page boundaries so a different question can reuse a long prompt.
    int checkpoint = save_prefix_checkpoints ?
        std::max((start_pos / kKvPageTokens + 1) * kKvPageTokens,
                 ((end - 1) / kKvPageTokens - 1) * kKvPageTokens) : end;
    for (int begin = 0; begin < count;) {
        const int position = start_pos + begin;
        const int rows = std::min(capacity, std::min(end, checkpoint) - position);
        const bool last = begin + rows == count;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids.ptr(), token_ids + begin, rows * sizeof(int32_t),
                                            cudaMemcpyHostToDevice, ctx.stream()));
        embed_tokens(ctx, tok_embeddings_, ids, hidden, rows, H, cfg_.embed_scale);
        for (const auto& span : inputs_embeds) {
            const int first = std::max(span.start, begin);
            const int span_end = std::min(span.start + static_cast<int>(span.rows.shape[0]), begin + rows);
            if (span_end <= first) continue;
            const auto src = span.rows.ptr<float>() + static_cast<size_t>(first - span.start) * H;
            const auto dst = hidden.ptr<float>() + static_cast<size_t>(first - begin) * H;
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(span_end - first) * H * sizeof(float),
                                                cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        if (pa.enabled) pa.reset();
        if (layer_trace || op_trace) {
            execute_ir_graph_cuda<true>(ir, ws, ctx, hidden, ids, cuda_workspace_.logits, rows, position, position,
                nullptr, nullptr, dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn,
                moe_layers_, use_fused, op_trace, timer.get(), last ? layer_trace : nullptr, last && compute_logits);
        } else {
            execute_ir_graph_cuda(ir, ws, ctx, hidden, ids, cuda_workspace_.logits, rows, position, position,
                nullptr, nullptr, dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn,
                moe_layers_, use_fused, nullptr, nullptr, nullptr, last && compute_logits);
        }
        begin += rows;
        if (position + rows == checkpoint && checkpoint < end) {
            save_cuda_checkpoint(ctx, checkpoint, false);
            checkpoint += kKvPageTokens;
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
                                                 OpTrace* op_trace, bool save_prefix_checkpoints,
                                                 std::span<const InputEmbedding> inputs_embeds) {
    validate_token_ids({token_ids, static_cast<size_t>(count)}, start_pos, cfg_.vocab_size);
    execute_prefill_chunks(ctx, token_ids, count, start_pos, use_fused, layer_trace, op_trace, true,
                           save_prefix_checkpoints, inputs_embeds);
    HostTensor logits = ctx.download(cuda_workspace_.logits);
    return {logits.data<float>(), logits.data<float>() + cfg_.vocab_size};
}

void FusionModel::verify_cuda(CudaContext& ctx, const std::vector<int32_t>& input, int pos, bool use_fused,
                               std::vector<float>& logits, std::vector<int32_t>& argmax, bool greedy,
                               bool retain_last_logits) {
    auto& arena = ctx.scratch_arena();
    struct RestoreArena {
        CudaArena& arena;
        size_t mark;
        ~RestoreArena() { arena.reset(mark); }
    } restore{arena, arena.mark()};
    const int rows = static_cast<int>(input.size());
    const int vocab = cfg_.vocab_size;
    validate_token_ids(input, pos, vocab);
    const int hidden_size = static_cast<int>(tok_embeddings_.shape[1]);
    ensure_cuda_kv(ctx, pos, pos + rows);
    Tensor ids = cuda_scratch(ctx, {rows}, DType::I32);
    Tensor hidden = cuda_scratch(ctx, {rows, hidden_size});
    Tensor batch_logits = cuda_scratch(ctx, {rows, vocab});
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids.ptr(), input.data(), rows * sizeof(int32_t),
                                        cudaMemcpyHostToDevice, ctx.stream()));
    embed_tokens(ctx, tok_embeddings_, ids, hidden, rows, hidden_size, cfg_.embed_scale);
    const auto& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    // Match the decode path's quantized activations even for short verification batches.
    struct RestoreCrossover {
        CudaContext& ctx;
        int rows;
        ~RestoreCrossover() { ctx.set_mmq_crossover_m(rows); }
    } crossover{ctx, ctx.mmq_crossover_m()};
    ctx.set_mmq_crossover_m(2);
    execute_ir_graph_cuda(ir, ws, ctx, hidden, ids, batch_logits, rows, pos, pos, nullptr, nullptr,
                          dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn, moe_layers_,
                          use_fused, nullptr, nullptr, nullptr, true, true);
    if (retain_last_logits) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_workspace_.logits.ptr(), batch_logits.slice_dim0(rows - 1, 1).ptr(),
                                            vocab * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    if (greedy) {
        argmax.resize(rows);
        for (int row = 0; row < rows; ++row) {
            Tensor row_logits = batch_logits.slice_dim0(row, 1);
            Tensor row_id = ids.slice_dim0(row, 1);
            decode_argmax(ctx, row_logits, row_id, vocab);
        }
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(argmax.data(), ids.ptr(), rows * sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
    } else {
        logits.resize(static_cast<size_t>(rows) * vocab);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(logits.data(), batch_logits.ptr(), logits.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
    }
    ctx.synchronize();
    if (greedy && std::find(argmax.begin(), argmax.end(), -1) != argmax.end())
        throw std::runtime_error("Sampling found no selectable logits during speculative verification");
}

static void lookup_draft(const std::vector<int32_t>& history, int limit, std::vector<int32_t>& input) {
    const int count = static_cast<int>(history.size());
    for (int n = std::min(3, count); n >= 2; --n) {
        for (int end = count - 1; end >= n; --end) {
            if (std::equal(history.end() - n, history.end(), history.begin() + end - n)) {
                input.insert(input.end(), history.begin() + end, history.begin() + end + std::min(limit, count - end));
                return;
            }
        }
    }
}

static int32_t verify_sample(const Sampler::Distribution& target, const Sampler::Distribution& draft,
                             int32_t proposal, std::mt19937_64& rng) {
    const auto probability = [](const Sampler::Distribution& distribution, int32_t id) {
        const auto it = std::lower_bound(distribution.begin(), distribution.end(), id,
                                         [](const auto& entry, int32_t token) { return entry.second < token; });
        return it == distribution.end() || it->second != id ? 0.0 : it->first;
    };
    const auto candidate = std::find_if(target.begin(), target.end(),
                                        [proposal](const auto& entry) { return entry.second == proposal; });
    if (candidate != target.end() &&
        std::generate_canonical<double, 53>(rng) * probability(draft, proposal) < candidate->first)
        return proposal;

    // A point-mass lookup proposal uses the same residual rule as a draft model.
    Sampler::Distribution residual;
    residual.reserve(target.size());
    for (const auto& [weight, id] : target) {
        const double difference = weight - probability(draft, id);
        if (difference > 0) residual.emplace_back(difference, id);
    }
    return Sampler::draw(residual, rng);
}

std::vector<int32_t> FusionModel::generate_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                int max_new_tokens, bool use_fused, bool use_graph,
                                                GenerationStats* stats, bool keep_state, int start_pos,
                                                int32_t stop_eos_id, const GenerationOptions& options) {
    options.sampling.validate();
    if (max_new_tokens < 0 || start_pos < 0) throw std::invalid_argument("Invalid generation length or start position");
    auto* draft = options.draft_model;
    const auto* json = options.json_constraint;
    if (json && json->vocab_size() != static_cast<size_t>(cfg_.vocab_size))
        throw std::invalid_argument("JSON constraint vocabulary does not match model");
    if (options.spec_ngram < 0 || options.draft_tokens <= 0 || (draft && options.spec_ngram > 0)) {
        throw std::invalid_argument("Choose one speculative decoding mode with a positive token limit");
    }
    if (draft && (draft == this || !draft->cuda_slots_[0] || draft->cfg_.vocab_size != cfg_.vocab_size || keep_state)) {
        throw std::invalid_argument("Draft generation needs a separate CUDA model with matching vocabulary and a fresh prompt");
    }
    if (!keep_state) start_pos = 0;
    std::vector<int32_t> tokens = prompt_ids;
    if (tokens.empty()) tokens.push_back(std::min(1, cfg_.vocab_size - 1));
    validate_token_ids(tokens, start_pos, cfg_.vocab_size);
    const auto embeds = options.inputs_embeds;
    validate_input_embeddings(embeds, prompt_ids.size(), static_cast<int>(tok_embeddings_.shape[1]));
    if (!embeds.empty() && (keep_state || draft || options.spec_ngram > 0))
        throw std::invalid_argument("Image input requires a fresh prompt and standard decoding");
    const auto total_tokens = static_cast<size_t>(start_pos) + tokens.size();
    if (total_tokens > static_cast<size_t>(cfg_.dense.max_seq_len)) {
        throw prompt_context_error(total_tokens, cfg_.dense.max_seq_len);
    }
    const int prompt_len = static_cast<int>(tokens.size());
    max_new_tokens = std::min(max_new_tokens, cfg_.dense.max_seq_len - start_pos - prompt_len);
    validate_cuda_kv_capacity(start_pos + prompt_len + max_new_tokens);
    if (json && max_new_tokens == 0) throw std::invalid_argument("JSON generation needs room for output tokens");
    const int batch_limit = (draft || options.spec_ngram > 0) && max_new_tokens > 1
        ? prefill_batch_limit(ctx, true) : 1;
    const bool speculative = batch_limit >= 2;
    if (speculative && options.op_trace) {
        throw std::invalid_argument("Operation profiling is unavailable during speculative decoding");
    }
    const bool reuse = options.reuse_prefix && !keep_state && embeds.empty();
    if (keep_state) clear_cuda_history();
    else if (!reuse) reset_kv_and_state(ctx);
    promote_cuda_expert_slabs(ctx, start_pos + prompt_len + max_new_tokens);
    auto* route_trace = ctx.moe_route_trace;
    if (route_trace) route_trace->begin(&moe_layers_);
    const bool greedy = options.sampling.greedy() && !json;
    const bool stochastic = options.sampling.temperature > 0;
    if (use_graph) {
        const char* fallback = nullptr;
        if (options.op_trace) fallback = "operation tracing is enabled";
        else if (route_trace) fallback = "MoE route tracing is enabled";
        else if (json) fallback = "JSON grammar is enabled";
        else if (stochastic) fallback = "temperature is greater than zero";
        else if (options.sampling.repeat_penalty != 1.0f) fallback = "repeat_penalty is not 1";
        else if (!can_use_cuda_graph(moe_layers_) &&
                 (!options.warn_on_graph_fallback ||
                  !moe_graph_prefix(use_fused ? fused_ir_graph_ : raw_ir_graph_, moe_layers_)))
            fallback = "no eligible MoE GPU prefix";
        if (fallback && options.warn_on_graph_fallback)
            std::clog << "--graph requested, using eager decoding because " << fallback << '\n';
        use_graph = fallback == nullptr;
    }
    std::mt19937_64 rng;
    if (!greedy) rng.seed(options.sampling.seed);
    std::mt19937_64 draft_rng(options.sampling.seed ^ UINT64_C(0x9e3779b97f4a7c15));
    auto grammar = json ? json->initial_state() : JsonGrammarState{};
    OpTrace prefill_profile;
    if (options.op_trace) prefill_profile.mode = "prefill";

    auto download_argmax_tok = [&]() -> int32_t {
        int32_t tok = 0;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(&tok, cuda_workspace_.token_ids.ptr<int32_t>(), sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
        ctx.synchronize();
        if (tok < 0) throw std::runtime_error("Sampling found no selectable logits");
        return tok;
    };
    auto sample_gpu = [&]() {
        decode_argmax(ctx, cuda_workspace_.logits, cuda_workspace_.token_ids, cfg_.vocab_size);
        return download_argmax_tok();
    };
    auto sample_cpu = [&](float* row, int remaining) {
        if (json) json->mask({row, static_cast<size_t>(cfg_.vocab_size)}, grammar, remaining);
        return Sampler::sample(row, cfg_.vocab_size, tokens, options.sampling, rng);
    };

    auto tp0 = std::chrono::high_resolution_clock::now();
    int prefilled = prompt_len;
    OpTrace* prefill_trace = options.op_trace ? &prefill_profile : nullptr;
    auto logits = reuse ? prefill_cached_cuda(ctx, tokens, use_fused, &prefilled, options.layer_trace, prefill_trace)
                        : prefill_cuda_impl(ctx, tokens.data(), prompt_len, start_pos, use_fused,
                                            options.layer_trace, prefill_trace, false, embeds);
    // A failing callback or CUDA operation must not leave a reusable recurrent state.
    cuda_session().cached_tokens.clear();
    int draft_pos = 0;
    if (speculative && draft && prompt_len < draft->cfg_.dense.max_seq_len) {
        draft->reset_kv_and_state(ctx);
        // Leave the first generated token pending, as on the target.
        draft->execute_prefill_chunks(ctx, tokens.data(), prompt_len, 0, use_fused, nullptr, nullptr, false);
        draft_pos = prompt_len;
    }
    int32_t next_tok = -1;
    if (max_new_tokens > 0) {
        next_tok = greedy ? sample_gpu() : sample_cpu(logits.data(), max_new_tokens);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    int pos = start_pos + prompt_len;
    int gen_count = 0;
    int spec_proposed = 0, spec_accepted = 0;
    const MiniGraph& ir = use_fused ? fused_ir_graph_ : raw_ir_graph_;
    CudaIRWorkspace ws(ir);
    std::unique_ptr<CudaOpTimer> timer;
    if (options.op_trace && max_new_tokens > 1) {
        timer = std::make_unique<CudaOpTimer>(ir.nodes.size());
        options.op_trace->mode = "decode";
    }
    bool used_cuda_graph = false;
    auto decode = [&](bool first_step) {
        ensure_cuda_kv(ctx, pos, pos + 1);
        used_cuda_graph = run_forward_token_step_cuda(ctx, cfg_, tok_embeddings_, ir, ws,
                                    cuda_workspace_.hidden, cuda_workspace_.logits, cuda_workspace_.token_ids, cuda_workspace_.position, cuda_workspace_.kv_length,
                                    dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn, moe_layers_,
                                    cuda_workspace_.decode_graph, cuda_workspace_.graph_captured, cuda_workspace_.graph_fused, cuda_workspace_.graph_has_argmax,
                                    cuda_workspace_.graph_split_bucket,
                                    pos++, use_fused, use_graph, greedy, first_step, options.op_trace, timer.get());
    };
    auto emit = [&](int32_t token) {
        tokens.push_back(token);
        ++gen_count;
        if (json && !json->accept(grammar, token)) throw std::logic_error("Sampled a token outside the JSON grammar");
        if (stop_eos_id >= 0 && token == stop_eos_id) return false;
        if (options.on_token && !options.on_token(token)) return false;
        return gen_count < max_new_tokens && !(json && (grammar.sealed() || json->is_end(token)));
    };
    auto td0 = std::chrono::high_resolution_clock::now();
    if (!speculative && greedy && use_graph && can_use_cuda_graph(moe_layers_) && !options.on_token && stop_eos_id < 0 && max_new_tokens > 1 &&
        static_cast<int64_t>(pos) + max_new_tokens <= cuda_workspace_.token_ids.shape[0]) {
        // Graph replay feeds the next greedy step and records IDs on the device.
        int first_pos = pos;
        for (int step = 1; step < max_new_tokens; ++step) decode(step == 1);
        tokens.push_back(next_tok);
        tokens.resize(static_cast<size_t>(prompt_len) + max_new_tokens);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(tokens.data() + prompt_len + 1, cuda_workspace_.token_ids.ptr<int32_t>() + 1 + first_pos,
                                            static_cast<size_t>(max_new_tokens - 1) * sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, ctx.stream()));
        ctx.synchronize();
        if (std::find(tokens.begin() + prompt_len, tokens.end(), -1) != tokens.end())
            throw std::runtime_error("Sampling found no selectable logits");
        gen_count = max_new_tokens;
    } else if (!speculative) {
        for (int step = 0; step < max_new_tokens; ++step) {
            if (!emit(next_tok)) break;
            if (!greedy) {
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_workspace_.token_ids.ptr(), &next_tok, sizeof(next_tok),
                                                    cudaMemcpyHostToDevice, ctx.stream()));
            }
            decode(step == 0);
            if (greedy) {
                next_tok = download_argmax_tok();
            } else {
                HostTensor host_logits = ctx.download(cuda_workspace_.logits);
                next_tok = sample_cpu(host_logits.data<float>(), max_new_tokens - gen_count);
            }
        }
    } else {
        auto& target_slot = cuda_session();
        CudaSession* draft_slot = draft_pos > 0 ? &draft->cuda_session() : nullptr;
        const size_t draft_ssm = draft_slot ? draft_slot->ssm_bytes : 0;
        CudaArena snapshots;
        void* target_snapshot = nullptr;
        void* draft_snapshot = nullptr;
        if (target_slot.ssm_bytes || draft_ssm) {
            snapshots.init(target_slot.ssm_bytes + draft_ssm);
            if (target_slot.ssm_bytes) target_snapshot = snapshots.alloc(target_slot.ssm_bytes);
            if (draft_ssm) draft_snapshot = snapshots.alloc(draft_ssm);
        }
        auto copy_state = [&](void* dst, const void* src, size_t bytes) {
            if (bytes) FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
        };
        const MiniGraph& draft_ir = draft ? (use_fused ? draft->fused_ir_graph_ : draft->raw_ir_graph_) : ir;
        CudaIRWorkspace draft_ws(draft_ir);
        const bool draft_greedy = options.sampling.greedy();
        const bool draft_graph = use_graph && draft && can_use_cuda_graph(draft->moe_layers_);
        auto draft_step = [&](const int32_t* token, bool need_argmax, bool first_step = true) {
            auto& work = draft->cuda_workspace_;
            draft->ensure_cuda_kv(ctx, draft_pos, draft_pos + 1);
            if (token) {
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(work.token_ids.ptr(), token, sizeof(int32_t),
                                                    cudaMemcpyHostToDevice, ctx.stream()));
            }
            run_forward_token_step_cuda(ctx, draft->cfg_, draft->tok_embeddings_, draft_ir, draft_ws,
                work.hidden, work.logits, work.token_ids, work.position, work.kv_length,
                draft->dense_layers_, draft->gdn_layers_, draft_slot->kv, draft_slot->gdn, draft->moe_layers_,
                work.decode_graph, work.graph_captured, work.graph_fused, work.graph_has_argmax, work.graph_split_bucket,
                draft_pos++, use_fused, draft_graph, need_argmax, first_step);
        };
        std::vector<int32_t> input, row_ids, draft_history;
        std::vector<float> rows;
        std::vector<Sampler::Distribution> draft_probs;
        int ngram_depth = 1, ngram_misses = 0, ngram_pause = 0;
        bool pending_on_device = greedy, reset_decode_position = true;
        bool running = max_new_tokens > 0 && emit(next_tok);
        while (running) {
            const int remaining = max_new_tokens - gen_count;
            const int start = pos;
            const bool ngram_probe = !draft && ngram_pause == 0;
            if (ngram_pause > 0) --ngram_pause;
            int limit = std::min({remaining - 1, batch_limit - 1, draft ? options.draft_tokens :
                                  ngram_probe ? ngram_depth : 0});
            input.assign(1, tokens.back());
            draft_probs.clear();
            if (draft_slot && limit > 0 && start < draft->cfg_.dense.max_seq_len) {
                limit = std::min(limit, draft->cfg_.dense.max_seq_len - start);
                while (draft_pos < start) draft_step(&tokens[draft_pos], false);
                copy_state(draft_snapshot, draft_slot->gdn_data, draft_ssm);
                if (draft_greedy) {
                    auto* device_ids = draft->cuda_workspace_.token_ids.ptr<int32_t>();
                    for (int i = 0; i < limit; ++i) {
                        draft_step(i == 0 ? input.data() : nullptr, true, i == 0);
                        if (!draft_graph) {
                            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(device_ids + 1 + start + i, device_ids,
                                sizeof(int32_t), cudaMemcpyDeviceToDevice, ctx.stream()));
                        }
                    }
                    input.resize(1 + limit);
                    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(input.data() + 1, device_ids + 1 + start,
                        limit * sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.stream()));
                    ctx.synchronize();
                } else {
                    draft_history = tokens;
                    for (int i = 0; i < limit; ++i) {
                        draft_step(&input.back(), false);
                        auto host = ctx.download(draft->cuda_workspace_.logits);
                        auto q = Sampler::distribution(host.data<float>(), cfg_.vocab_size, draft_history, options.sampling);
                        const auto proposal = stochastic ? Sampler::draw(q, draft_rng) : q.front().second;
                        if (stochastic) {
                            std::sort(q.begin(), q.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
                            draft_probs.push_back(std::move(q));
                        }
                        draft_history.push_back(proposal);
                        input.push_back(proposal);
                    }
                }
            } else if (!draft && limit > 0) {
                lookup_draft(tokens, limit, input);
            }
            const int proposed = static_cast<int>(input.size()) - 1;
            if (proposed == 0) {
                if (ngram_probe && limit > 0 && ++ngram_misses >= 3) {
                    ngram_pause = 8;
                    ngram_misses = 0;
                }
                if (!pending_on_device) {
                    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_workspace_.token_ids.ptr(), input.data(), sizeof(int32_t),
                                                        cudaMemcpyHostToDevice, ctx.stream()));
                }
                decode(reset_decode_position);
                pending_on_device = greedy;
                reset_decode_position = false;
                if (greedy) next_tok = download_argmax_tok();
                else {
                    auto host = ctx.download(cuda_workspace_.logits);
                    next_tok = sample_cpu(host.data<float>(), remaining);
                }
                running = emit(next_tok);
                continue;
            }

            spec_proposed += proposed;
            copy_state(target_snapshot, target_slot.gdn_data, target_slot.ssm_bytes);
            verify_cuda(ctx, input, start, use_fused, rows, row_ids, greedy, reuse && target_snapshot);
            pending_on_device = false;
            reset_decode_position = true;
            int accepted_count = 0;
            for (int i = 0; i <= proposed; ++i) {
                const bool bonus = i == proposed;
                if (greedy) next_tok = row_ids[i];
                else {
                    auto* row = rows.data() + static_cast<size_t>(i) * cfg_.vocab_size;
                    if (!stochastic || bonus) next_tok = sample_cpu(row, max_new_tokens - gen_count);
                    else {
                        if (json) json->mask({row, static_cast<size_t>(cfg_.vocab_size)}, grammar, max_new_tokens - gen_count);
                        const auto p = Sampler::distribution(row, cfg_.vocab_size, tokens, options.sampling);
                        const auto proposal = input[i + 1];
                        next_tok = draft ? verify_sample(p, draft_probs[i], proposal, rng) :
                                           verify_sample(p, {{1.0, proposal}}, proposal, rng);
                    }
                }
                const bool accepted = !bonus && next_tok == input[i + 1];
                spec_accepted += accepted;
                accepted_count += accepted;
                running = emit(next_tok);
                if (!accepted || !running) break;
            }
            if (!draft) {
                if (accepted_count == proposed) {
                    ngram_misses = 0;
                    if (proposed == ngram_depth && ngram_depth < options.spec_ngram) ++ngram_depth;
                } else {
                    ngram_depth = std::max(1, accepted_count);
                    ngram_pause = accepted_count == 0 ? 8 : 0;
                    ngram_misses = 0;
                }
            }
            // The last emitted token is pending. Only earlier tokens belong to the live KV/SSM prefix.
            const int kept = start_pos + static_cast<int>(tokens.size()) - 1;
            pos = start + proposed + 1;
            if (kept < pos) {
                cuda_kv_pool_.truncate(target_slot.pages, kept);
                if (target_snapshot) {
                    copy_state(target_slot.gdn_data, target_snapshot, target_slot.ssm_bytes);
                    pos = start;
                    execute_prefill_chunks(ctx, input.data(), kept - start, start, use_fused, nullptr, nullptr, reuse);
                }
                pos = kept;
            }
            if (draft_slot && draft_pos > kept) {
                draft->cuda_kv_pool_.truncate(draft_slot->pages, kept);
                if (draft_snapshot) {
                    copy_state(draft_slot->gdn_data, draft_snapshot, draft_ssm);
                    draft_pos = start;
                    while (draft_pos < kept) draft_step(&tokens[draft_pos], false);
                } else draft_pos = kept;
            }
        }
        ctx.synchronize();
    }
    if (reuse) {
        // The most recently sampled token has not been forwarded into KV/SSM state.
        if (pos > prompt_len) save_cuda_checkpoint(ctx, pos, true);
        cuda_session().cached_tokens.assign(tokens.begin(), tokens.begin() + pos);
        cuda_session().cached_fused = use_fused;
    }
    if (options.op_trace && options.op_trace->ops.empty()) *options.op_trace = std::move(prefill_profile);
    auto td1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    if (route_trace) route_trace->finish(start_pos + prompt_len, pos);

    if (stats) {
        stats->prompt_tokens = static_cast<int>(prompt_ids.size());
        stats->generated_tokens = gen_count;
        stats->prefilled_tokens = prefilled;
        stats->spec_proposed = spec_proposed;
        stats->spec_accepted = spec_accepted;
        stats->used_cuda_graph = used_cuda_graph;
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
    size_t logits_bytes = cuda_workspace_.logits.nbytes();
    FUSIONRT_CUDA_CHECK(cudaMallocHost(&host_ptr, logits_bytes + sizeof(int32_t) * std::max(pp, 3)));
    std::unique_ptr<void, decltype(&cudaFreeHost)> host_logits(host_ptr, cudaFreeHost);
    auto* host_args = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(host_ptr) + logits_bytes);
    auto read_logits = [&] {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host_logits.get(), cuda_workspace_.logits.ptr(), logits_bytes,
                                           cudaMemcpyDeviceToHost, ctx.stream()));
    };
    use_graph = use_graph && can_use_cuda_graph(moe_layers_);
    reset_kv_and_state(ctx);

    if (pp > 0) {
        ensure_cuda_kv(ctx, 0, pp);
        std::copy(prompt.begin(), prompt.end(), host_args);
        auto prefill = [&] {
            execute_prefill_chunks(ctx, host_args, pp, 0, true, nullptr, nullptr);
            read_logits();
        };
        CudaGraphExec graph;
        if (use_graph) graph.capture(ctx.stream(), prefill);
        for (int rep = -1; rep < reps; ++rep) {
            zero_cuda_gdn(ctx);
            ctx.synchronize();
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
        ensure_cuda_kv(ctx, 0, decode_prefix + tg);
        Tensor args = cuda_scratch(ctx, {3}, DType::I32);
        auto decode = [&](int pos) {
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(args.ptr(), host_args, args.nbytes(), cudaMemcpyHostToDevice, ctx.stream()));
            embed_tokens(ctx, tok_embeddings_, args, cuda_workspace_.hidden, 1, H, cfg_.embed_scale);
            execute_ir_graph_cuda(ir, ws, ctx, cuda_workspace_.hidden, args, cuda_workspace_.logits, 1, pos, pos,
                                  use_graph ? args.ptr<int32_t>() + 1 : nullptr,
                                  use_graph ? args.ptr<int32_t>() + 2 : nullptr,
                                  dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn, moe_layers_, true);
            read_logits();
        };
        CudaGraphExec graph;
        if (use_graph) graph.capture(ctx.stream(), [&] { decode(decode_prefix); });
        std::vector<std::pair<HostTensor, HostTensor>> recurrent_prefix;
        if (!prefix.empty()) {
            zero_cuda_gdn(ctx);
            execute_prefill_chunks(ctx, prefix.data(), decode_prefix, 0, true, nullptr, nullptr, false);
            for (const auto& state : cuda_session().gdn) {
                recurrent_prefix.emplace_back(ctx.download(state.convolution), ctx.download(state.recurrence));
            }
            ctx.synchronize();
        }
        for (int rep = -1; rep < reps; ++rep) {
            if (prefix.empty()) {
                zero_cuda_gdn(ctx);
                ctx.synchronize();
            }
            // Decode only overwrites KV after the prefix. GDN state must be restored in place.
            for (size_t i = 0; i < recurrent_prefix.size(); ++i) {
                ctx.upload_into(recurrent_prefix[i].first, cuda_session().gdn[i].convolution);
                ctx.upload_into(recurrent_prefix[i].second, cuda_session().gdn[i].recurrence);
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
            ensure_cuda_kv(ctx, 0, decode_prefix + tg);
            if (!prefix.empty()) execute_prefill_chunks(ctx, prefix.data(), decode_prefix, 0, true, nullptr, nullptr, false);
            CudaOpTimer timer(ir.nodes.size());
            for (int pos = 0; pos < tg; ++pos) {
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cuda_workspace_.token_ids.ptr(), &steps[pos], sizeof(int32_t),
                                                    cudaMemcpyHostToDevice, ctx.stream()));
                run_forward_token_step_cuda(ctx, cfg_, tok_embeddings_, ir, ws, cuda_workspace_.hidden, cuda_workspace_.logits,
                                            cuda_workspace_.token_ids, cuda_workspace_.position, cuda_workspace_.kv_length, dense_layers_, gdn_layers_, cuda_session().kv, cuda_session().gdn, moe_layers_,
                                            cuda_workspace_.decode_graph, cuda_workspace_.graph_captured, cuda_workspace_.graph_fused, cuda_workspace_.graph_has_argmax,
                                cuda_workspace_.graph_split_bucket,
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
