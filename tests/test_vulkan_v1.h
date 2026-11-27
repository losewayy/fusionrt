#pragma once

#include "test_support.h"
#include "../src/models/vulkan_executor.h"
#include "../src/models/vulkan_session.h"
#include "../src/models/vulkan_weights.h"

namespace fusionrt::test {
namespace {

HostTensor native_weight_fixture(DType type, int rows, int width) {
    if (type == DType::F32 || type == DType::F16 || type == DType::BF16) {
        HostTensor fp32({rows, width}, DType::F32);
        const auto values = codec_input(rows, width);
        std::copy(values.begin(), values.end(), fp32.data<float>());
        return model_io::quantize_host_matrix(fp32, type);
    }

    HostTensor result({rows, width}, type);
    const auto bytes = dtype_type_size(type);
    std::mt19937 rng(918);
    for (size_t offset = 0; offset < result.storage.size(); offset += bytes) {
        auto* block = result.storage.data() + offset;
        std::generate_n(block, bytes, [&] { return static_cast<uint8_t>(rng()); });
        const auto scale = [&](size_t index, float value) {
            const uint16_t bits = fp32_to_fp16(value);
            block[index] = static_cast<uint8_t>(bits);
            block[index + 1] = static_cast<uint8_t>(bits >> 8);
        };
        switch (type) {
            case DType::Q2_K: scale(80, 0.015625f); scale(82, 0.03125f); break;
            case DType::Q4_K: case DType::Q5_K:
            case DType::Q4_1: scale(0, 0.0078125f); scale(2, 0.015625f); break;
            case DType::Q6_K: scale(208, 0.0009765625f); break;
            case DType::Q3_K: scale(108, 0.015625f); break;
            case DType::TQ1_0: scale(52, 0.125f); break;
            case DType::TQ2_0: scale(64, 0.125f); break;
            case DType::PTQ1_0: scale(26, 0.125f); break;
            case DType::MXFP4: block[0] = static_cast<uint8_t>(115 + rng() % 9); break;
            default: scale(0, 0.0078125f); break;
        }
    }
    return result;
}

void require_vk_close(DType type, const char* op, std::span<const float> actual,
                      std::span<const float> expected, double threshold) {
    const double error = codec_rel_l2(actual, expected);
    if (!(error <= threshold)) {
        throw std::runtime_error(std::string(dtype_name(type)) + " " + op + " relative L2=" + std::to_string(error));
    }
}

#ifdef FUSIONRT_ENABLE_CUDA
void check_cuda_native_weight(CudaContext& cuda, const HostTensor& src, const HostTensor& x,
                              const HostTensor& decoded, std::span<const float> expected) {
    auto& arena = cuda.scratch_arena();
    const auto mark = arena.mark();
    const int rows = static_cast<int>(x.desc.shape[0]);
    const int n = static_cast<int>(src.desc.shape[0]), k = static_cast<int>(src.desc.shape[1]);
    const auto packed = cuda.upload(src, false), input = cuda.upload(x, false);
    HostTensor ids({n}, DType::I32);
    std::iota(ids.data<int32_t>(), ids.data<int32_t>() + n, 0);
    const auto indices = cuda.upload(ids, false);
    auto weights = arena.alloc_tensor({n, k}, DType::F32);
    auto output = arena.alloc_tensor({rows, n}, DType::F32);
    cuda_ops::embed_lookup_f32(packed.ptr(), src.desc.dtype, indices.ptr<int32_t>(), weights.ptr<float>(),
                               n, k, n, cuda.stream());
    const auto unpacked = cuda.download(weights);
    require_codec_close(src.desc.dtype, "CUDA compact decode", {unpacked.data<float>(), static_cast<size_t>(n) * k},
                        {decoded.data<float>(), static_cast<size_t>(n) * k});
    cuda_ops::linear_f32(input.ptr<float>(), weights.ptr<float>(), output.ptr<float>(), rows, n, k, cuda.stream());
    const auto product = cuda.download(output);
    require_codec_close(src.desc.dtype, "CUDA compact weight product", {product.data<float>(), expected.size()}, expected);
    if (is_weight_codec_dtype(src.desc.dtype)) {
        cuda_ops::codec_gemv_f32(input.ptr<float>(), packed.ptr(), src.desc.dtype, output.ptr<float>(),
                                 8, n, k, cuda.stream());
        const auto gemv = cuda.download(output);
        require_codec_close(src.desc.dtype, "CUDA WeightCodec GEMV", {gemv.data<float>(), static_cast<size_t>(8 * n)},
                            expected.first(8 * n));
    }
    arena.reset(mark);
}
#endif

void test_vk_native_matmul(VkContext& vk
#ifdef FUSIONRT_ENABLE_CUDA
                          , CudaContext& cuda
#endif
) {
    constexpr DType types[] = {
        DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q8_0, DType::Q4_1,
        DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::Q2_K,
#define VULKAN_CODEC_TEST_FORMAT(name, id, block, bytes) DType::name,
        FUSIONRT_WEIGHT_CODEC_FORMATS(VULKAN_CODEC_TEST_FORMAT)
#undef VULKAN_CODEC_TEST_FORMAT
    };
    constexpr int max_rows = 35, n = 37;
    const bool coop = vk.supports_coopmat() && vk.subgroup_size() == 32;
    auto& arena = vk.scratch_arena();
    for (const DType type : types) {
        const auto mark = arena.mark();
        const int block = static_cast<int>(dtype_block_size(type));
        const int k = block == 1 ? 68 : (block <= 128 ? 3 : 2) * block;
        const auto src = native_weight_fixture(type, n, k);
        const auto decoded = model_io::to_f32_host(src);
        const auto values = codec_input(max_rows, k);
        HostTensor x({max_rows, k}, DType::F32);
        std::copy(values.begin(), values.end(), x.data<float>());
        std::vector<float> reference(max_rows * n);
        for (int m = 0; m < max_rows; ++m)
            for (int i = 0; i < n; ++i)
                reference[m * n + i] = static_cast<float>(codec_reference_dot(decoded.data<float>() + i * k,
                                                                              values.data() + m * k, k));
        const auto weights = vk.upload(src, false), input = vk.upload(x, false);
        for (const int m : {1, 8, 17, 35}) {
            auto output = arena.alloc_tensor({m, n}, DType::F32);
            const auto check = [&](const VkContext* policy, const char* shader, double tolerance) {
                std::vector<VkBatchDispatch> ops;
                std::vector<std::vector<uint8_t>> push;
                vk_ops::append_linear_dispatch(ops, input, weights, output, m, n, k, push, policy);
                if (shader && ops.front().shader_name != shader)
                    throw std::runtime_error("Unexpected Vulkan matmul dispatch for " + std::string(dtype_name(type)));
                vk.dispatch_batched(ops, false);
                const auto actual = vk.download(output);
                const auto length = static_cast<size_t>(m * n);
                require_vk_close(type, ops.front().shader_name.c_str(), {actual.data<float>(), length},
                                 std::span(reference).first(length), tolerance);
            };
            check(nullptr, m > 8 ? "matmul/gemm_quant_prefill.spv" : nullptr, 2e-5);
            if (m >= 16 && coop) check(&vk, "matmul/gemm_coopmat_quant.spv", 2e-3);
        }
#ifdef FUSIONRT_ENABLE_CUDA
        check_cuda_native_weight(cuda, src, x, decoded, reference);
#endif
        arena.reset(mark);
    }
    std::cout << "  [PASS] CPU"
#ifdef FUSIONRT_ENABLE_CUDA
              << "/CUDA"
#endif
              << "/Vulkan compact weights, M=1/8 GEMV and M=17/35 tiled prefill"
              << (coop ? " and CoopMat" : "") << '\n';
}

std::vector<float> vk_attention_reference(const HostTensor& qkv, const HostTensor& key, const HostTensor& value,
                                          int q_heads, int kv_heads, int dim, int start, int rows, int window) {
    const int q_width = q_heads * dim, kv_width = kv_heads * dim;
    const float scale = 1.f / std::sqrt(static_cast<float>(dim));
    std::vector<float> output(rows * q_width);
    for (int row = 0; row < rows; ++row) {
        const int pos = start + row;
        const int first = window > 0 ? std::max(0, pos - window + 1) : 0;
        std::vector<double> scores(pos - first + 1);
        for (int h = 0; h < q_heads; ++h) {
            const int kv_head_offset = h / (q_heads / kv_heads) * dim;
            const auto* q = qkv.data<float>() + pos * (q_width + 2 * kv_width) + h * dim;
            for (int j = first; j <= pos; ++j)
                scores[j - first] = scale * codec_reference_dot(q, key.data<float>() + j * kv_width + kv_head_offset, dim);
            const double peak = *std::max_element(scores.begin(), scores.end());
            double norm = 0.0;
            for (auto& score : scores) norm += score = std::exp(score - peak);
            for (int d = 0; d < dim; ++d) {
                double sum = 0.0;
                for (int j = first; j <= pos; ++j)
                    sum += scores[j - first] * value.data<float>()[j * kv_width + kv_head_offset + d];
                output[row * q_width + h * dim + d] = static_cast<float>(sum / norm);
            }
        }
    }
    return output;
}

void test_vk_attention_precision(VkContext& vk) {
    constexpr int start = 5, length = 73, rows = length - start, nq = 4, nkv = 1, hd = 96;
    constexpr int q_width = nq * hd, kv_width = nkv * hd, width = q_width + 2 * kv_width;
    std::mt19937_64 rng(601);
    const auto qkv = rand_f32({length, width}, rng, 0.5f);
    HostTensor keys({length, nkv, hd}, DType::F32), values({length, nkv, hd}, DType::F32);
    for (int row = 0; row < length; ++row) {
        const auto* k = qkv.data<float>() + row * width + q_width;
        std::copy_n(k, kv_width, keys.data<float>() + row * kv_width);
        std::copy_n(k + kv_width, kv_width, values.data<float>() + row * kv_width);
    }
    auto& arena = vk.scratch_arena();
    const auto initial = arena.mark();
    const auto input = vk.upload(qkv, false), norm = vk.upload(make_ones_f32({hd}), false);
    const auto q = arena.alloc_tensor({length, q_width}, DType::F32);
    const auto prefill = arena.alloc_tensor({rows, q_width}, DType::F32);
    const auto decode = arena.alloc_tensor({1, q_width}, DType::F32);
    const auto mark = arena.mark();
    for (const DType type : {DType::F32, DType::F16, DType::Q8_0}) {
        const auto key = arena.alloc_tensor({length, nkv, hd}, type);
        const auto value = arena.alloc_tensor({length, nkv, hd}, type);
        const int kv_type = static_cast<int>(type);
        const vk_ops::PushRopeKvBatch rope{length, nq, nkv, hd, 0, 0, kv_type, 0, 10000.f, 1e-6f};
        vk.dispatch_single({"sequence/rope_kv_append_batch.spv", nq + nkv, length, 1,
                            {input, norm, norm, q, key, value, norm}, &rope, sizeof(rope)});
        const auto key_ref = model_io::to_f32_host(model_io::quantize_host_matrix(keys, type));
        const auto value_ref = model_io::to_f32_host(model_io::quantize_host_matrix(values, type));
        for (const auto* cache : {&key, &value}) {
            const auto host = model_io::to_f32_host(vk.download(*cache));
            const auto& ref = cache == &key ? key_ref : value_ref;
            require_vk_close(type, "KV append", {host.data<float>(), length * kv_width},
                             {ref.data<float>(), length * kv_width}, type == DType::Q8_0 ? 2e-3 : 1e-6);
        }
        for (const int window : {0, 11}) {
            const float scale = 1.f / std::sqrt(static_cast<float>(hd));
            const vk_ops::PushFaPrefill p{rows, nq, nkv, hd, start, length, kv_type, window, 1, scale};
            const vk_ops::PushFaDecode d{nq, nkv, hd, length, kv_type, window, scale};
            vk.dispatch_single({"sequence/flash_attention_prefill.spv", nq, rows, 1,
                                {q.slice_dim0(start, rows), key, value, prefill}, &p, sizeof(p)});
            vk.dispatch_single({"sequence/flash_attention_decode.spv", nq, 1, 1,
                                {q.slice_dim0(length - 1, 1), key, value, decode}, &d, sizeof(d)});
            const auto expected = vk_attention_reference(qkv, key_ref, value_ref, nq, nkv, hd, start, rows, window);
            const auto p_actual = vk.download(prefill), d_actual = vk.download(decode);
            const double tolerance = type == DType::Q8_0 ? 3e-3 : 2e-5;
            require_vk_close(type, "Vulkan FA prefill", {p_actual.data<float>(), expected.size()}, expected, tolerance);
            require_vk_close(type, "Vulkan FA decode", {d_actual.data<float>(), q_width},
                             std::span(expected).last(q_width), tolerance);
        }
        arena.reset(mark);
    }
    arena.reset(initial);
    std::cout << "  [PASS] Vulkan KV append and FA prefill/decode vs CPU for F32, F16 and Q8_0\n";
}

void test_vk_dense_q4k_prefill(VkContext& vk) {
    HybridModelConfig cfg;
    cfg.vocab_size = 59;
    cfg.num_layers = 2;
    cfg.dense.hidden_size = cfg.dense.intermediate_size = 256;
    cfg.dense.num_attention_heads = 4;
    cfg.dense.num_key_value_heads = 2;
    cfg.dense.head_dim = 64;
    cfg.dense.max_seq_len = 64;
    cfg.dense.weight_dtype = DType::Q4_K;
    vk.set_kv_dtype(DType::F16);
    vk.set_prefill_batch(17);
    VulkanModelWeights weights(cfg);
    VulkanSessionManager sessions;
    weights.init_synthetic_vulkan(vk, 71, {
        [&](const DenseGQAConfig& dense, DType type) { sessions.allocate_dense(vk, dense, type); },
        [&](const GatedDeltaNetConfig& gdn, int conv_dim) { sessions.allocate_gdn(vk, gdn, conv_dim); }
    });
    if (weights.embeddings().dtype != DType::Q4_K || weights.lm_head().dtype != DType::Q4_K ||
        vk.weight_arena().capacity() != vk.weight_arena().used() || sessions.dense(0).key.dtype != DType::F16)
        throw std::runtime_error("Vulkan Dense Q4_K weights and F16 KV must occupy the exact compact arena");
    for (const auto& layer : weights.dense_layers()) {
        for (const auto* matrix : {&layer.qkv_proj_fused, &layer.o_proj, &layer.gate_up_proj, &layer.down_proj}) {
            if (matrix->dtype != DType::Q4_K || matrix->nbytes() != matrix->numel() / 256 * sizeof(BlockQ4_K))
                throw std::runtime_error("Vulkan Dense matrix expanded out of Q4_K");
        }
    }
    VulkanExecutor executor(weights, sessions);
    std::vector<int32_t> prompt(39);
    std::iota(prompt.begin(), prompt.end(), 1);
    sessions.reset_kv_and_state_vulkan(vk);
    const auto batch_logits = executor.forward_batch_vulkan(vk, prompt, 0);
    sessions.reset_kv_and_state_vulkan(vk);
    std::vector<float> serial_logits;
    for (int pos = 0; pos < static_cast<int>(prompt.size()); ++pos)
        serial_logits = executor.forward_token_vulkan(vk, prompt[pos], pos);
    require_vk_close(DType::Q4_K, "Vulkan Dense batch vs token", batch_logits, serial_logits, 3e-3);
    const auto expected_next = Sampler::argmax(serial_logits.data(), cfg.vocab_size);
    const auto continued = executor.forward_token_vulkan(vk, expected_next, static_cast<int>(prompt.size()));
    const auto generated = executor.generate_vulkan(vk, prompt, 1, nullptr, false, 0, -1);
    if (generated.back() != expected_next)
        throw std::runtime_error("Vulkan chunked Dense prefill changed the greedy token");
    (void)executor.generate_vulkan(vk, prompt, 0, nullptr, false, 0, -1);
    const auto after_chunks = executor.forward_token_vulkan(vk, expected_next, static_cast<int>(prompt.size()));
    require_vk_close(DType::Q4_K, "Vulkan Dense chunked continuation", after_chunks, continued, 3e-3);
    std::cout << "  [PASS] Vulkan Dense Q4_K native arena and batch/chunked prefill vs token-by-token\n";
}

void test_vulkan_v1(
#ifdef FUSIONRT_ENABLE_CUDA
    CudaContext& cuda
#endif
) {
    VkContext vk(0, 0, 4);
    test_vk_native_matmul(vk
#ifdef FUSIONRT_ENABLE_CUDA
                          , cuda
#endif
    );
    test_vk_attention_precision(vk);
    test_vk_dense_q4k_prefill(vk);
}

}  // namespace
}  // namespace fusionrt::test
