#pragma once

#include "fusionrt/vulkan_backend.h"

namespace fusionrt {

struct VkSparseMoELayer;
struct VulkanDenseWeights;
struct VulkanDenseKv;
struct VulkanGdnWeights;
struct VulkanGdnState;

struct VulkanDispatch {
    struct Batch {
        std::vector<VkBatchDispatch> ops;
        std::vector<std::vector<uint8_t>> push;
    };

    struct DenseScratch {
        Tensor normalized, query, query_gate, query_full, key, value, attention, projected;
    };
    struct GdnScratch {
        Tensor normalized, qkv, z, a, b, recurrent, projected;
    };
    struct FfnScratch {
        Tensor gate_up, activated, down;
    };
    struct MoeScratch {
        Tensor normalized, router_logits, indices, weights, intermediate, output;
        Tensor shared_gate_up, shared_activated, shared_down, shared_gate;
    };

    static void append_embedding(Batch&, const Tensor& embeddings, const Tensor& token, const Tensor& output, int hidden);
    static void append_dense(Batch&, const VulkanDenseWeights&, const VulkanDenseKv&, DenseScratch&,
                             const Tensor& input, const Tensor& output, int hidden, int pos);
    static void append_gdn(Batch&, const VulkanGdnWeights&, const VulkanGdnState&, GdnScratch&,
                           const Tensor& input, const Tensor& output, int hidden);
    static void append_ffn(Batch&, const Tensor& norm, const Tensor& gate_up, const Tensor& down,
                           Tensor& normalized, FfnScratch&, const Tensor& activation, int hidden, int inter, float eps);
    static void append_moe(Batch&, const VkSparseMoELayer&, MoeScratch&,
                           const Tensor& input, const Tensor& output, int hidden);
    static void append_head(Batch&, const Tensor& norm, const Tensor& head, Tensor& normalized, Tensor& logits,
                            const Tensor& input, int hidden, int vocab, float eps);
};

}  // namespace fusionrt
