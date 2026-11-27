#pragma once

#include "fusionrt/vulkan_backend.h"
#include <unordered_map>

namespace fusionrt {

struct VulkanMoeWeights;
struct VulkanDenseWeights;
struct VulkanDenseKv;
struct VulkanGdnWeights;
struct VulkanGdnState;

struct VulkanDispatch {
    struct Batch {
        const VkContext& ctx;
        std::vector<VkBatchDispatch> ops;
        std::vector<std::vector<uint8_t>> push;
        Tensor dummy, rotated;
        const std::unordered_map<uint64_t, Tensor>* hadamard_signs = nullptr;
    };

    struct DenseScratch {
        Tensor normalized, qkv, query, query_gate, query_full, key, value, attention, projected, bias;
        Tensor image_query, image_attention;
    };
    struct AttentionSpan { int begin, end; };
    struct MlaScratch {
        Tensor normalized, q_low, q, kv, q_pe, q_absorbed, latent, value, projected;
        Tensor head_input, head_output;
    };
    struct GdnScratch {
        Tensor normalized, qkv, z, a, b, recurrent, projected;
    };
    struct FfnScratch {
        Tensor gate_up, activated, down, gate, up;
    };
    struct MoeScratch {
        Tensor normalized, router_input, router_logits, indices, weights, intermediate;
        Tensor shared_gate_up, shared_activated, shared_down, shared_gate, gate_scalar, gated_row;
    };

    static void append_embedding(Batch&, const Tensor& embeddings, const Tensor& token, const Tensor& output, int hidden, int rows);
    static void append_copy(Batch&, const Tensor& source, const Tensor& target, std::vector<VkBufferCopy>);
    static void append_linear(Batch&, const Tensor& input, const Tensor& weights,
                              Tensor& output, int rows, int width, int extent);
    static void append_norm(Batch&, const Tensor& input, const Tensor& weight, const Tensor& output,
                            int rows, int width, float eps);
    static void append_gemma_norm(Batch&, const Tensor& input, const Tensor& weight, const Tensor& output,
                                  int rows, int width, float eps, bool add_one = false);
    static void append_scale_add(Batch&, const Tensor& a, const Tensor& b, const Tensor& output,
                                 int rows, int width, float scale_a, float scale_b);
    static void append_geglu(Batch&, const Tensor& gate, const Tensor& up, const Tensor& output,
                             int rows, int width);
    static void append_hadamard(Batch&, const Tensor& weight, const Tensor& input, const Tensor& output,
                                int rows, int width, bool inverse);
    static void append_dense(Batch&, const VulkanDenseWeights&, const VulkanDenseKv&, DenseScratch&,
                             const Tensor& input, const Tensor& output, int hidden, int pos, int rows, std::span<const AttentionSpan> image_spans = {});
    static void append_mla(Batch&, const VulkanDenseWeights&, const VulkanDenseKv&, MlaScratch&,
                           const Tensor& input, const Tensor& output, int hidden, int pos, int rows);
    static void append_gdn(Batch&, const VulkanGdnWeights&, const VulkanGdnState&, GdnScratch&,
                           const Tensor& input, const Tensor& output, int hidden, int rows);
    static void append_ffn(Batch&, const Tensor& norm, const Tensor& gate_up, const Tensor& gate,
                           const Tensor& up, const Tensor& down,
                           Tensor& normalized, FfnScratch&, const Tensor& activation,
                           int hidden, int inter, int rows, float eps, bool geglu = false);
    static void append_finish_ffn(Batch&, const Tensor& activation, const Tensor& update,
                                  const Tensor& post_norm, int hidden, int rows, float eps, float scale);
    static void append_moe(Batch&, const VulkanMoeWeights&, MoeScratch&,
                           const Tensor& input, const Tensor& output, int hidden, int rows);
    static void append_head(Batch&, const Tensor& norm, const Tensor& head, Tensor& normalized, Tensor& logits,
                            const Tensor& input, int hidden, int vocab, float eps, float softcap);
};

}  // namespace fusionrt
