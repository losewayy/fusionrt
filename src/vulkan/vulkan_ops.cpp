#include "fusionrt/vulkan_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace fusionrt::vk_ops {

namespace {

template <typename T>
const void* store_push(std::vector<std::vector<uint8_t>>& storage, const T& val) {
    storage.emplace_back(sizeof(T));
    std::memcpy(storage.back().data(), &val, sizeof(T));
    return storage.back().data();
}

}  // namespace

double gemv_vec4(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int M, int N, int K) {
    PushGemm pc{M, N, K};
    VkBatchDispatch d;
    d.shader_name = "gemv_vec4.comp";
    d.gx = static_cast<uint32_t>(N);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {x, w, y};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

void append_linear_dispatch(std::vector<VkBatchDispatch>& ops,
                            const Tensor& x, const Tensor& w, Tensor& y,
                            int M, int N, int K,
                            std::vector<std::vector<uint8_t>>& push_storage, const VkContext* ctx) {
    const auto n = static_cast<uint32_t>(N);
    const auto m = static_cast<uint32_t>(M);
    if (M <= 8) {
        const char* shader = nullptr;
        if (w.dtype == DType::F32 && K % 4 == 0) shader = "gemv_vec4.comp";
        else if (M == 1 && w.dtype == DType::Q8_0) shader = "gemv_q8_0.comp";
        else if (M == 1 && w.dtype == DType::Q4_0) shader = "gemv_q4_0.comp";
        if (shader) {
            const PushGemm push{M, N, K};
            ops.push_back({shader, n, 1, 1, {x, w, y}, store_push(push_storage, push), sizeof(push)});
            return;
        }
        if (w.dtype == DType::F32) {
            const PushGemm push{M, N, K};
            ops.push_back({"gemm_tiled.comp", (n + 15) / 16, 1, 1, {x, w, y},
                           store_push(push_storage, push), sizeof(push)});
            return;
        }
    }

    const PushQuantMatmul push{M, N, K, static_cast<int>(w.dtype)};
    if (M <= 8) {
        const char* shader = is_weight_codec_dtype(w.dtype)
            ? "matmul/gemv_weight_codec.spv" : "matmul/gemv_quant_k.spv";
        ops.push_back({shader, n, m, 1, {x, w, y}, store_push(push_storage, push), sizeof(push)});
    } else {
        const bool coop = ctx && ctx->supports_coopmat() && ctx->subgroup_size() == 32 &&
                          M >= std::max(16, ctx->device_profile().mmq_crossover_m);
        const char* shader = coop ? "matmul/gemm_coopmat_quant.spv" : "matmul/gemm_quant_prefill.spv";
        ops.push_back({shader, (n + 31) / 32, (m + 31) / 32, 1, {x, w, y},
                       store_push(push_storage, push), sizeof(push)});
    }
}

void append_rmsnorm_linear_dispatch(std::vector<VkBatchDispatch>& ops,
                                    const Tensor& x, const Tensor& norm_w, const Tensor& w,
                                    Tensor& x_norm_scratch, Tensor& y,
                                    int M, int N, int K, float eps,
                                    std::vector<std::vector<uint8_t>>& push_storage, const VkContext* ctx) {
    if (w.dtype == DType::F32 && M <= 8 && (K % 4) == 0) {
        PushFusedNormGemv pc{M, N, K, eps};
        ops.push_back({"fused_rmsnorm_gemv_vec4.comp",
                       static_cast<uint32_t>(N), 1, 1,
                       {x, norm_w, w, y},
                       store_push(push_storage, pc), sizeof(pc)});
    } else {
        PushRmsNorm npc{M, K, eps};
        ops.push_back({"rmsnorm.comp",
                       static_cast<uint32_t>(M), 1, 1,
                       {x, norm_w, x_norm_scratch},
                       store_push(push_storage, npc), sizeof(npc)});
        append_linear_dispatch(ops, x_norm_scratch, w, y, M, N, K, push_storage, ctx);
    }
}

double linear_auto(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int M, int N, int K) {
    std::vector<VkBatchDispatch> ops;
    std::vector<std::vector<uint8_t>> push_storage;
    append_linear_dispatch(ops, x, w, y, M, N, K, push_storage, &ctx);
    return ctx.dispatch_batched(ops, true);
}

double gemm_auto(VkContext& ctx, const Tensor& a, const Tensor& b, Tensor& c, int M, int N, int K, bool use_coopmat_if_avail) {
    PushGemm pc{M, N, K};
    VkBatchDispatch d;
    if (use_coopmat_if_avail && ctx.info().coop_mat.supported && ctx.info().coop_mat.fp16_in_fp32_acc &&
        a.dtype == DType::F16 && b.dtype == DType::F16 &&
        (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0)) {
        d.shader_name = "gemm_coopmat.comp";
        d.gx = static_cast<uint32_t>(N / 16);
        d.gy = static_cast<uint32_t>(M / 16);
    } else {
        d.shader_name = "gemm_tiled.comp";
        d.gx = static_cast<uint32_t>((N + 15) / 16);
        d.gy = static_cast<uint32_t>((M + 15) / 16);
    }
    d.gz = 1;
    d.buffers = {a, b, c};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

double rmsnorm(VkContext& ctx, const Tensor& x, const Tensor& w, Tensor& y, int rows, int cols, float eps) {
    PushRmsNorm pc{rows, cols, eps};
    VkBatchDispatch d;
    d.shader_name = "rmsnorm.comp";
    d.gx = static_cast<uint32_t>(rows);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {x, w, y};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

double fused_rmsnorm_gemv(VkContext& ctx, const Tensor& x, const Tensor& norm_w, const Tensor& w, Tensor& y, int M, int N, int K, float eps) {
    PushFusedNormGemv pc{M, N, K, eps};
    VkBatchDispatch d;
    d.shader_name = "fused_rmsnorm_gemv_vec4.comp";
    d.gx = static_cast<uint32_t>(N);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {x, norm_w, w, y};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

std::vector<VkBatchDispatch> build_dense_layer_dispatches(
    const Tensor& x_in,
    const Tensor& attn_norm_w, const Tensor& qkv_w, const Tensor& q_norm_w, const Tensor& k_norm_w,
    const Tensor& o_w, const Tensor& mlp_norm_w, const Tensor& gate_up_w, const Tensor& down_w,
    Tensor& qkv_buf, Tensor& q_buf, Tensor& k_buf, Tensor& v_buf,
    Tensor& k_cache, Tensor& v_cache, Tensor& attn_out, Tensor& o_buf,
    Tensor& gate_up_buf, Tensor& swiglu_buf, Tensor& down_buf, Tensor& x_out,
    int hidden, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim, int inter,
    int pos, int kv_len, float theta, float eps, bool qk_norm, bool neox,
    std::vector<std::vector<uint8_t>>& push_storage)
{
    std::vector<VkBatchDispatch> ops;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    (void)qkv_buf;

    // 1. Fused Pre-Norm + QKV GEMV writing directly into q_buf, k_buf, v_buf
    {
        struct PushFusedQkv { int q_dim, kv_dim, K; float eps; } pc{q_dim, kv_dim, hidden, eps};
        ops.push_back({"fused_rmsnorm_qkv_vec4.comp",
                       static_cast<uint32_t>(qkv_dim), 1, 1,
                       {x_in, attn_norm_w, qkv_w, q_buf, k_buf, v_buf},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 2. Fused Q/K Norm + RoPE + KV Cache Append
    {
        int flags = (qk_norm ? 1 : 0) | (neox ? 2 : 0);
        PushRopeKv pc{num_q_heads, num_kv_heads, head_dim, rot_dim, pos, theta, eps, flags};
        ops.push_back({"rope_kv_append.comp",
                       static_cast<uint32_t>(num_q_heads + num_kv_heads), 1, 1,
                       {q_buf, k_buf, v_buf, q_norm_w, k_norm_w, k_cache, v_cache},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 3. GQA Decode
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        PushGqaDecode pc{num_q_heads, num_kv_heads, head_dim, kv_len, scale};
        ops.push_back({"gqa_decode.comp",
                       static_cast<uint32_t>(num_q_heads), 1, 1,
                       {q_buf, k_cache, v_cache, attn_out},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 4. O Proj
    append_linear_dispatch(ops, attn_out, o_w, o_buf, 1, hidden, q_dim, push_storage);

    // 5. Residual 1: x_out = x_in + o_buf
    {
        PushResidual pc{hidden};
        ops.push_back({"add_residual.comp",
                       static_cast<uint32_t>((hidden + 255) / 256), 1, 1,
                       {x_in, o_buf, x_out},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 6. Fused MLP Pre-Norm + Gate/Up GEMV
    {
        PushFusedNormGemv pc{1, 2 * inter, hidden, eps};
        ops.push_back({"fused_rmsnorm_gemv_vec4.comp",
                       static_cast<uint32_t>(2 * inter), 1, 1,
                       {x_out, mlp_norm_w, gate_up_w, gate_up_buf},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 7. SwiGLU
    {
        PushSwiglu pc{1, inter};
        ops.push_back({"swiglu.comp",
                       static_cast<uint32_t>((inter + 255) / 256), 1, 1,
                       {gate_up_buf, swiglu_buf},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    // 8. Down Proj
    append_linear_dispatch(ops, swiglu_buf, down_w, down_buf, 1, hidden, inter, push_storage);

    // 9. Residual 2: x_out = x_out + down_buf
    {
        PushResidual pc{hidden};
        ops.push_back({"add_residual.comp",
                       static_cast<uint32_t>((hidden + 255) / 256), 1, 1,
                       {x_out, down_buf, x_out},
                       store_push(push_storage, pc), sizeof(pc)});
    }

    return ops;
}

double gated_deltanet_step(
    VkContext& ctx,
    const Tensor& qkv_proj, const Tensor& z, const Tensor& a, const Tensor& b,
    const Tensor& conv_w, const Tensor& conv_bias, const Tensor& A_log, const Tensor& dt_bias,
    const Tensor& norm_w, Tensor& conv_state, Tensor& ssm_state, Tensor& out,
    int num_k_heads, int num_v_heads, int dk, int dv, float eps, float q_scale, bool has_conv_bias)
{
    PushGdnStep pc{num_k_heads, num_v_heads, dk, dv, eps, q_scale, has_conv_bias ? 1 : 0, 0};
    VkBatchDispatch d;
    d.shader_name = "gated_deltanet_step.comp";
    d.gx = static_cast<uint32_t>(num_k_heads);
    d.gy = 1;
    d.gz = 1;
    d.buffers = {qkv_proj, z, a, b, conv_w, conv_bias, A_log, dt_bias, norm_w, conv_state, ssm_state, out};
    d.push_data = &pc;
    d.push_size = sizeof(pc);
    return ctx.dispatch_single(d, true);
}

}  // namespace fusionrt::vk_ops

namespace fusionrt {

double VkContext::dispatch_single(const VkBatchDispatch& d, bool measure_timestamp) {
    return dispatch_batched({d}, measure_timestamp);
}

double VkContext::dispatch_batched(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp) {
    if (dispatches.empty()) return 0.0;

    measure_timestamp = measure_timestamp && timestamp_bits_ > 0;
    uint32_t set_count = 0, descriptor_count = 0;
    for (const auto& d : dispatches) {
        if (!d.copies.empty()) continue;
        ++set_count;
        descriptor_count += static_cast<uint32_t>(d.buffers.size());
    }
    if (set_count > descriptor_set_capacity_ || descriptor_count > descriptor_capacity_) {
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max(descriptor_count, 4096u)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets = std::max(set_count, 1024u);
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &size;
        VkDescriptorPool pool;
        FUSIONRT_VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool));
        if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        desc_pool_ = pool;
        descriptor_capacity_ = size.descriptorCount;
        descriptor_set_capacity_ = ci.maxSets;
    }
    struct PoolReset {
        VkDevice device;
        VkDescriptorPool pool;
        ~PoolReset() { if (pool) vkResetDescriptorPool(device, pool, 0); }
    } reset{device_, desc_pool_};

    std::vector<VkDescriptorSet> dsets(dispatches.size());
    for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& d = dispatches[i];
        if (!d.copies.empty()) continue;
        const auto& pipe = get_pipeline(d.shader_name, static_cast<uint32_t>(d.buffers.size()), d.push_size);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = desc_pool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &pipe.desc_layout;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        FUSIONRT_VK_CHECK(vkAllocateDescriptorSets(device_, &dsai, &ds));

        std::vector<VkDescriptorBufferInfo> dbis(d.buffers.size());
        std::vector<VkWriteDescriptorSet> writes(d.buffers.size());
        for (size_t i = 0; i < d.buffers.size(); ++i) {
            dbis[i].buffer = reinterpret_cast<VkBuffer>(d.buffers[i].vk_buffer);
            dbis[i].offset = d.buffers[i].vk_offset;
            dbis[i].range = std::max<VkDeviceSize>((d.buffers[i].nbytes() + 3) & ~VkDeviceSize{3}, 16);

            writes[i] = VkWriteDescriptorSet{};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &dbis[i];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        dsets[i] = ds;
    }

    VkCommandBuffer cmd = begin_one_time_commands();

    if (measure_timestamp) {
        vkCmdResetQueryPool(cmd, query_pool_, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
    }

    constexpr auto compute_and_transfer = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& d = dispatches[i];
        if (!d.copies.empty()) {
            auto regions = d.copies;
            for (auto& r : regions) {
                r.srcOffset += d.buffers[0].vk_offset;
                r.dstOffset += d.buffers[1].vk_offset;
            }
            vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(d.buffers[0].vk_buffer),
                            reinterpret_cast<VkBuffer>(d.buffers[1].vk_buffer),
                            static_cast<uint32_t>(regions.size()), regions.data());
        } else {
            const auto& pipe = get_pipeline(d.shader_name, static_cast<uint32_t>(d.buffers.size()), d.push_size);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipe_layout, 0, 1, &dsets[i], 0, nullptr);
            if (d.push_size) {
                vkCmdPushConstants(cmd, pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, d.push_size, d.push_data);
            }
            vkCmdDispatch(cmd, d.gx, d.gy, d.gz);
        }

        // Include the last operation so a subsequent submission can read the results or KV.
        VkMemoryBarrier bar{};
        bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, compute_and_transfer, compute_and_transfer, 0, 1, &bar, 0, nullptr, 0, nullptr);
    }

    if (measure_timestamp) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
    }

    submit_and_wait(cmd);

    double gpu_ms = 0.0;
    if (measure_timestamp) {
        uint64_t ts[2]{0, 0};
        FUSIONRT_VK_CHECK(vkGetQueryPoolResults(device_, query_pool_, 0, 2, sizeof(ts), ts,
                                                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        const uint64_t elapsed = (ts[1] - ts[0]) & (UINT64_MAX >> (64 - timestamp_bits_));
        gpu_ms = static_cast<double>(elapsed) * static_cast<double>(info_.timestamp_period_ns) * 1e-6;
    }

    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    return gpu_ms;
}

double VkContext::dispatch_unbatched(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp) {
    double total_ms = 0.0;
    for (const auto& d : dispatches) {
        total_ms += dispatch_batched({d}, measure_timestamp);
    }
    return total_ms;
}

}  // namespace fusionrt
