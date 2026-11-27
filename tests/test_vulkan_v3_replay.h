#pragma once

#include "test_vulkan_v1.h"

namespace fusionrt::test {
namespace {

void test_vk_v3_batch(VkContext& vk) {
    const auto check = [](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    };
    auto& arena = vk.scratch_arena();
    auto input = vk.upload(make_ones_f32({64}), false);
    auto a = arena.alloc_tensor({64}, DType::F32);
    auto b = arena.alloc_tensor({64}, DType::F32);
    auto out = arena.alloc_tensor({64}, DType::F32);
    int count = 64;
    const auto add = [&](const Tensor& x, const Tensor& y, const Tensor& z) {
        return VkBatchDispatch{"add_residual.comp", 1, 1, 1, {x, y, z}, &count, sizeof(count), {}};
    };
    const auto copy = [](const Tensor& x, const Tensor& y, VkDeviceSize offset = 0) {
        VkBatchDispatch op;
        op.buffers = {x, y};
        op.copies = {{offset, offset, 64}};
        return op;
    };
    const auto expect = [&](const Tensor& tensor, size_t index, float value) {
        check(vk.download(tensor).data<float>()[index] == value, "Vulkan replay produced stale data");
    };
    const auto run = [&](const std::vector<VkBatchDispatch>& ops) { vk.dispatch_batched(ops, false); };
    std::vector ops{add(input, input, a), add(input, input, b), copy(a, out), add(b, input, a), copy(a, out, 64)};
    run(ops);
    check(vk.last_batch_descriptor_allocs() == 3, "Cold batch did not allocate three unique descriptor sets");
    check(vk.last_batch_barriers() == 4, "Batch did not prune disjoint writes or protect RAW/WAR dependencies");
    expect(out, 0, 2.0f);
    expect(out, 16, 3.0f);
    vk.zero_buffers(std::span(&out, 1));
    run(ops);
    check(vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 0, "Identical batch missed replay");
    expect(out, 16, 3.0f);
    run({copy(input, out), copy(input, out, 64)});
    check(vk.last_batch_barriers() == 1, "Disjoint copy ranges emitted a spurious barrier");
    run({copy(input, out), copy(input, out)});
    check(vk.last_batch_barriers() == 2, "Overlapping copies lost their WAW barrier");
    run(ops);
    count = 32;
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 0,
          "Changing push constants should reuse only the descriptor sets");
    ops[1].gx = 2;
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 0,
          "Changing dispatch size should reuse only the descriptor sets");
    ops.back().copies[0].dstOffset = 128;
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 0, "Copy change replayed an old region");
    expect(out, 32, 3.0f);
    ops[0].buffers[0].shape[0] = 32;
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 1, "Descriptor range change was ignored");
    ops[0].buffers[0].vk_offset = b.vk_offset;
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 1, "Descriptor offset change was ignored");
    vk.set_replay_enabled(false);
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 0, "Replay toggle lost cached descriptors");
    vk.set_replay_enabled(true);
    vk.invalidate_replay();
    run(ops);
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 3, "Invalidation retained Vulkan state");

    ops.clear();
    for (int i = 0; i < 1100; ++i) ops.push_back(add(input, input, arena.alloc_tensor({64}, DType::F32)));
    run(ops);
    check(vk.last_batch_descriptor_allocs() == ops.size() && vk.last_batch_barriers() == 1,
          "Descriptor pool growth or independent dispatch handling failed");
    run(ops);
    check(vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 0, "Replay failed across pool pages");
    expect(ops.back().buffers[2], 0, 2.0f);

    vk.reset_arenas();
    input = vk.upload(make_ones_f32({64}), false);
    a = arena.alloc_tensor({64}, DType::F32);
    run({add(input, input, a)});
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 1, "Arena reset retained descriptors");
    arena.reset();
    check(vk.try_reserve_scratch_memory(8 * 1024 * 1024), "Test scratch arena could not grow");
    input = vk.upload(make_ones_f32({64}), false);
    a = arena.alloc_tensor({64}, DType::F32);
    run({add(input, input, a)});
    check(!vk.last_batch_replay_hit() && vk.last_batch_descriptor_allocs() == 1, "Scratch replacement retained replay");
    expect(a, 0, 2.0f);
}

void test_vk_v3_quant_gemv(VkContext& vk) {
    constexpr DType types[] = {
        DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q8_0, DType::Q4_1,
        DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::Q2_K,
#define VULKAN_V3_CODEC_TYPE(name, id, block, bytes) DType::name,
        FUSIONRT_WEIGHT_CODEC_FORMATS(VULKAN_V3_CODEC_TYPE)
#undef VULKAN_V3_CODEC_TYPE
    };
    constexpr int n = 17, rows = 8;
    auto& arena = vk.scratch_arena();
    for (const auto type : types) {
        const auto mark = arena.mark();
        const auto block = static_cast<int>(dtype_block_size(type));
        const int k = block == 1 ? 135 : 3 * block;
        const auto packed = native_weight_fixture(type, n, k);
        const auto decoded = model_io::to_f32_host(packed);
        const auto values = codec_input(rows, k);
        HostTensor x({rows, k}, DType::F32);
        std::copy(values.begin(), values.end(), x.data<float>());
        const auto w = vk.upload(packed, false), input = vk.upload(x, false);
        for (const int m : {1, rows}) {
            const auto out = arena.alloc_tensor({m, n}, DType::F32);
            const vk_ops::PushQuantMatmul pc{m, n, k, static_cast<int>(type)};
            const char* shader = is_weight_codec_dtype(type) ? "matmul/gemv_weight_codec.spv" : "matmul/gemv_quant_k.spv";
            vk.dispatch_single({shader, n, static_cast<uint32_t>(m), 1, {input, w, out}, &pc, sizeof(pc), {}});
            const auto actual = vk.download(out);
            std::vector<float> expected(m * n);
            for (int row = 0; row < m; ++row)
                for (int col = 0; col < n; ++col)
                    expected[row * n + col] = static_cast<float>(codec_reference_dot(
                        decoded.data<float>() + col * k, values.data() + row * k, k));
            require_vk_close(type, shader, {actual.data<float>(), expected.size()}, expected, 2e-5);
        }
        arena.reset(mark);
    }
}

}  // namespace

inline void test_vulkan_v3_replay() {
    for (const auto& device : VkContext::enumerate_devices()) {
        VkContext vk(device.device_index, 0, 4);
        test_vk_v3_batch(vk);
        test_vk_v3_quant_gemv(vk);
        std::cout << "  [PASS] Vulkan V3 replay, barriers, and 23 GEMV formats on " << device.name << '\n';
    }
}

}  // namespace fusionrt::test
