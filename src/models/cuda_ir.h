#pragma once

#include "cuda_dispatch.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include <cmath>

namespace fusionrt {

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
struct CudaIRSegment {
    int first, end;
    int input_value;
    int boundary_value = -1;
    Tensor* boundary = nullptr;
};

struct GdnReplay {
    std::vector<Tensor> projections;

    explicit GdnReplay(CudaContext& ctx, const std::vector<CudaGatedDeltaNetLayer>& layers, int rows) {
        for (const auto& layer : layers) {
            const auto& c = layer.cfg;
            const int qkv = 2 * c.num_k_heads * c.head_k_dim + c.num_v_heads * c.head_v_dim;
            projections.push_back(cuda_scratch(ctx, {rows, qkv + c.num_v_heads * (c.head_v_dim + 2)}));
        }
    }

    void save(CudaContext& ctx, int layer, int rows, std::array<const float*, 4> inputs,
              std::array<int, 4> widths, std::array<int, 4> strides) {
        const auto& storage = projections[layer];
        auto* dst = storage.ptr<float>();
        for (size_t i = 0; i < widths.size(); ++i) {
            FUSIONRT_CUDA_CHECK(cudaMemcpy2DAsync(dst, storage.shape[1] * sizeof(float), inputs[i],
                strides[i] * sizeof(float), widths[i] * sizeof(float), rows, cudaMemcpyDeviceToDevice, ctx.stream()));
            dst += widths[i];
        }
    }

    static void steps(CudaContext& ctx, const CudaGatedDeltaNetLayer& layer, const CudaGdnState& state,
                      std::array<const float*, 4> inputs, std::array<int, 4> strides, float* out,
                      int rows, float eps, bool use_fused) {
        const auto& c = layer.cfg;
        const int nk = c.num_k_heads, nv = c.num_v_heads, dk = c.head_k_dim, dv = c.head_v_dim;
        const int v_dim = nv * dv, qkv = 2 * nk * dk + v_dim;
        const float scale = 1.f / std::sqrt(static_cast<float>(dk));
        const auto mark = ctx.scratch_arena().mark();
        Tensor scratch, scratch_out;
        if (use_fused && dv >= 64 && (dk == 64 || dk == 128))
            scratch = cuda_scratch(ctx, {qkv + v_dim + 2 * nv});
        else if (!use_fused) {
            scratch = cuda_scratch(ctx, {qkv});
            scratch_out = cuda_scratch(ctx, {v_dim});
        }
        for (int m = 0; m < rows; ++m) {
            const auto* q = inputs[0] + static_cast<size_t>(m) * strides[0];
            const auto* z = inputs[1] + static_cast<size_t>(m) * strides[1];
            const auto* a = inputs[2] + static_cast<size_t>(m) * strides[2];
            const auto* b = inputs[3] + static_cast<size_t>(m) * strides[3];
            float* y = out + static_cast<size_t>(m) * v_dim;
            if (use_fused) {
                cuda_ops::fused_gated_deltanet_step_f32(q, z, a, b, layer.conv_w.ptr<float>(),
                    layer.conv_bias.ptr<float>(), layer.A_log.ptr<float>(), layer.dt_bias.ptr<float>(),
                    layer.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                    y, nk, nv, dk, dv, eps, scale, ctx.stream(), scratch.ptr<float>(),
                    c.tiled_value_heads, c.conv_kernel_size);
            } else {
                cuda_ops::unfused_gated_deltanet_step_f32(q, z, a, b, layer.conv_w.ptr<float>(),
                    layer.conv_bias.ptr<float>(), layer.A_log.ptr<float>(), layer.dt_bias.ptr<float>(),
                    layer.norm_w.ptr<float>(), state.convolution.ptr<float>(), state.recurrence.ptr<float>(),
                    y, scratch.ptr<float>(), scratch_out.ptr<float>(), nk, nv, dk, dv, eps, scale, ctx.stream(),
                    c.tiled_value_heads, c.conv_kernel_size);
            }
        }
        ctx.scratch_arena().reset(mark);
    }

    void restore(CudaContext& ctx, const std::vector<CudaGatedDeltaNetLayer>& layers,
                 const std::vector<CudaGdnState>& states, int rows, bool use_fused) const {
        const auto mark = ctx.scratch_arena().mark();
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            const auto& c = layer.cfg;
            const int v_dim = c.num_v_heads * c.head_v_dim;
            const int qkv = 2 * c.num_k_heads * c.head_k_dim + v_dim;
            const auto* src = projections[i].ptr<float>();
            const int stride = static_cast<int>(projections[i].shape[1]);
            auto out = cuda_scratch(ctx, {rows, v_dim});
            steps(ctx, layer, states[i], {src, src + qkv, src + qkv + v_dim, src + qkv + v_dim + c.num_v_heads},
                  {stride, stride, stride, stride}, out.ptr<float>(), rows, c.rms_norm_eps, use_fused);
            ctx.scratch_arena().reset(mark);
        }
    }
};

template<bool Observe = false>
void execute_ir_graph_cuda(
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
    const CudaIRSegment* segment = nullptr, float* nextn_out = nullptr, GdnReplay* replay = nullptr, const int* image_bounds = nullptr);

}  // namespace fusionrt
#endif
