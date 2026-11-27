#include "test_support.h"
#include "../src/models/cuda_dispatch.h"
#include "fusionrt/repack.h"

namespace fusionrt::test {

// Packed blocks and expected FP32 fingerprints from the ggml reference decoder.
struct CodecFixture {
    DType type;
    int elements, bytes, scale_offset;
    uint64_t hash;
};

constexpr CodecFixture codec_fixtures[] = {
    {DType::Q3_K,   256, 110, 108, 0xb9b5b8f40b5f9aa5ull},
    {DType::IQ2_XXS,256,  66,   0, 0xace5e35ee5c54bedull},
    {DType::IQ2_XS, 256,  74,   0, 0x4d509c24a3554e1dull},
    {DType::IQ3_XXS,256,  98,   0, 0x22cb41796a0613edull},
    {DType::IQ4_NL,  32,  18,   0, 0x86d46fe0aacde9e5ull},
    {DType::IQ3_S,  256, 110,   0, 0x0fb884f038702015ull},
    {DType::IQ4_XS, 256, 136,   0, 0x65c86b925a03a197ull},
    {DType::TQ1_0,  256,  54,  52, 0x77701590e1236325ull},
    {DType::TQ2_0,  256,  66,  64, 0x365290aba4c06325ull},
    {DType::MXFP4,   32,  17,   0, 0x71a10ada5b054b25ull},
    {DType::Q2_0,    64,  18,   0, 0xa6cb4cc8f0073325ull},
    {DType::PQ2_0,  128,  34,   0, 0xab129627b52e2325ull},
    {DType::PTQ1_0, 128,  28,  26, 0x6b64a37572418325ull},
};

HostTensor packed_fixture(const CodecFixture& f, const std::vector<int64_t>& shape, bool reference_bits = false) {
    auto result = HostTensor::create(shape, f.type);
    constexpr uint16_t reference_scales[] = {0x2c00, 0xa800, 0, 0x3555};
    constexpr uint16_t op_scales[] = {0x1800, 0x9400, 0, 0x1400};
    constexpr uint8_t reference_exponents[] = {0, 1, 127, 131};
    constexpr uint8_t op_exponents[] = {119, 117, 118, 120};
    for (int64_t j = 0; j < result.desc.numel() / f.elements; ++j) {
        auto* b = result.storage.data() + j * f.bytes;
        // Repeat the block pattern to keep the fixture independent of the row shape.
        const int seed = static_cast<int>(j % 16);
        for (int i = 0; i < f.bytes; ++i)
            b[i] = static_cast<uint8_t>((i * 73 + seed * 29) ^ ((i + 17 * seed) * (i + seed) >> 3));
        if (!reference_bits && f.type == DType::IQ4_XS) {
            // Restrict subscales to [-16, 15] so the tile path is exactly representable in FP16.
            b[2] = 0x99;
            b[3] = 0x66;
        }
        if (f.type == DType::MXFP4) b[0] = (reference_bits ? reference_exponents : op_exponents)[j % 4];
        else {
            const uint16_t h = (reference_bits ? reference_scales : op_scales)[j % 4];
            b[f.scale_offset] = static_cast<uint8_t>(h);
            b[f.scale_offset + 1] = static_cast<uint8_t>(h >> 8);
        }
    }
    return result;
}

std::vector<float> codec_reference_moe(const HostTensor& gate, const HostTensor& up, const HostTensor& down,
                                       std::span<const float> x, std::span<const int32_t> ids,
                                       std::span<const float> routes, int rows, int top_k) {
    const auto g = model_io::to_f32_host(gate), u = model_io::to_f32_host(up), d = model_io::to_f32_host(down);
    const int H = static_cast<int>(gate.desc.shape[2]), I = static_cast<int>(gate.desc.shape[1]);
    std::vector<float> out(static_cast<size_t>(rows) * H);
    std::vector<float> act(I);
    for (int m = 0; m < rows; ++m) for (int k = 0; k < top_k; ++k) {
        const int expert = ids[m * top_k + k];
        const auto* input = x.data() + static_cast<size_t>(m) * H;
        for (int i = 0; i < I; ++i) {
            const int index = (expert * I + i) * H;
            const double a = codec_reference_dot(g.data<float>() + index, input, H);
            const double b = codec_reference_dot(u.data<float>() + index, input, H);
            act[i] = static_cast<float>(a / (1.0 + std::exp(-a)) * b);
        }
        for (int h = 0; h < H; ++h)
            out[m * H + h] += routes[m * top_k + k] *
                static_cast<float>(codec_reference_dot(d.data<float>() + (expert * H + h) * I, act.data(), I));
    }
    return out;
}

void test_universal_weight_codecs() {
    struct FixtureFile {
        const char* path = "test-weight-codecs.gguf";
        ~FixtureFile() { std::error_code ec; std::filesystem::remove(path, ec); }
    } file;
    std::vector<std::pair<std::string, HostTensor>> weights;
    constexpr int rows = 5, top_k = 2;
    const int32_t ids[] = {0, 1, 0, 1, 1, 0, 0, 1, 0, 1};
    const float routes[] = {0.75f, 0.25f, 0.625f, 0.375f, 0.5f, 0.5f, 0.125f, 0.875f, 0.25f, 0.75f};
    CpuMoeWorkspace scratch;
    for (const auto& f : codec_fixtures) {
        const auto bits = packed_fixture(f, {16, f.elements}, true);
        if (dtype_block_size(f.type) != static_cast<size_t>(f.elements) ||
            dtype_type_size(f.type) != static_cast<size_t>(f.bytes) ||
            bits.desc.nbytes() != static_cast<size_t>(16 * f.bytes))
            throw std::runtime_error("Universal packed tensor geometry");
        const auto unpacked = model_io::to_f32_host(bits);
        uint64_t hash = 14695981039346656037ull;
        for (int i = 0; i < bits.desc.numel(); ++i) {
            const float v = unpacked.data<float>()[i];
            hash = (hash ^ (v == 0.0f ? 0u : std::bit_cast<uint32_t>(v))) * 1099511628211ull;
        }
        if (hash != f.hash) throw std::runtime_error(std::string(dtype_name(f.type)) + " ggml reference fingerprint");
        weights.emplace_back(dtype_name(f.type), bits);

        const int H = f.elements * 2, I = f.elements;
        const auto gate = packed_fixture(f, {2, I, H}), up = packed_fixture(f, {2, I, H});
        const auto down = packed_fixture(f, {2, H, I});
        const auto input = codec_input(rows, H);
        const auto expected = codec_reference_moe(gate, up, down, input, ids, routes, rows, top_k);
        std::vector<float> output(expected.size());
        for (const auto mode : {CpuMoeMode::Scalar, CpuMoeMode::Parallel}) {
            cpu_moe_accumulate_layer(nullptr, scratch, gate, up, down, ids, routes, 0, 0, rows, top_k,
                                      input.data(), output.data(), mode);
            require_codec_close(f.type, "CPU cold experts", output, expected);
        }

        MoeLayerCost placement{2, 1, H, {gate.desc.nbytes() / 2, up.desc.nbytes() / 2, down.desc.nbytes() / 2}};
        const auto bytes = placement.gpu_bytes(1);
        CostHardware hardware;
        hardware.hbm_bw_gbps = 512;
        hardware.cpu_q4_gemv_gbps = 25;
        const std::array costs{placement};
        if (select_moe_placement(costs, bytes - 1, hardware, 1).gpu_experts[0] != 0 ||
            select_moe_placement(costs, placement.gpu_bytes(2), hardware, 1).gpu_experts[0] != 2)
            throw std::runtime_error("Universal codec expert byte budget");
    }
    std::string error;
    GgufFile mapped;
    if (!GgufFile::write_v3(file.path, {}, weights, &error) || !mapped.load(file.path, &error))
        throw std::runtime_error(error);
    for (const auto& [name, weight] : weights) {
        const auto actual = mapped.view_tensor(name);
        if (actual.desc.dtype != weight.desc.dtype || !std::ranges::equal(actual.bytes(), weight.bytes()))
            throw std::runtime_error("Universal codec GGUF roundtrip");
    }
    std::cout << "  [PASS] 13 GGML codecs, bit-exact fingerprints, mapped GGUF, CPU multi-row MoE and placement\n";
}

void test_hadamard_cpu() {
    double max_orthogonal = 0.0;
    for (const int block : {64, 128, 256, 512, 1024}) {
        std::vector<int8_t> signs(block);
        auto weight = codec_input(1, block);
        for (int i = 0; i < block; ++i) signs[i] = (i % 3) ? 1 : -1;
        std::vector<float> basis(static_cast<size_t>(block) * block);
        for (int i = 0; i < block; ++i) basis[i * block + i] = 1.f;
        blockwise_hadamard_f32(basis.data(), signs.data(), basis.data(), block, block, block);
        blockwise_hadamard_f32(basis.data(), signs.data(), basis.data(), block, block, block, true);
        for (int i = 0; i < block * block; ++i)
            max_orthogonal = std::max(max_orthogonal, std::abs(static_cast<double>(basis[i]) - (i % (block + 1) == 0)));

        const auto x = codec_input(1, block);
        auto rx = x, rw = weight, fused = x, unfused = x;
        blockwise_hadamard_f32(x.data(), signs.data(), rx.data(), 1, block, block);
        blockwise_hadamard_f32(weight.data(), signs.data(), rw.data(), 1, block, block);
        const auto reference = codec_reference_dot(weight.data(), x.data(), block);
        if (!(std::abs(codec_reference_dot(rw.data(), rx.data(), block) - reference) < 1e-5 * std::abs(reference)))
            throw std::runtime_error("Rotated linear equivalence");

        constexpr float eps = 1e-5f;
        const auto inv = 1.0 / std::sqrt(codec_reference_dot(x.data(), x.data(), block) / block + eps);
        for (int i = 0; i < block; ++i) unfused[i] = static_cast<float>(x[i] * inv * weight[i]);
        blockwise_hadamard_f32(unfused.data(), signs.data(), unfused.data(), 1, block, block);
        fused_rmsnorm_hadamard_f32(x.data(), weight.data(), signs.data(), fused.data(), 1, block, block, eps);
        require_codec_close(DType::F32, "CPU fused FWHT", fused, unfused);
    }
    if (!(max_orthogonal < 1e-6)) throw std::runtime_error("Hadamard basis is not orthogonal");
    std::cout << "  [PASS] FWHT 64-1024 CPU orthogonality max_err=" << max_orthogonal << '\n';
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_weight_codecs(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const int crossover = ctx.mmq_crossover_m();
    ctx.set_mmq_crossover_m(2);
    double max_gemv = 0, max_gemm = 0, max_moe = 0;
    for (const auto& f : codec_fixtures) {
        constexpr int M = 23, N = 131;
        const int K = 2 * f.elements;
        const auto src = packed_fixture(f, {N, K});
        const auto fp = model_io::to_f32_host(src);
        const auto values = codec_input(M, K);
        HostTensor x({M, K}, DType::F32);
        std::copy(values.begin(), values.end(), x.data<float>());
        const auto dx = ctx.upload(x, false), w = ctx.upload(src, false);
        auto out = arena.alloc_tensor({M, N}, DType::F32);
        const auto check = [&](bool gemm) {
            if (has_weight_codec_int8(f.type)) return;
            model_io::vision_linear_cuda(ctx, dx, w, out, gemm ? M : 1);
            const auto read = ctx.download(out);
            const auto length = static_cast<size_t>((gemm ? M : 1) * N);
            std::vector<float> ref(length);
            for (int m = 0; m < (gemm ? M : 1); ++m) for (int n = 0; n < N; ++n) {
                ref[m * N + n] = static_cast<float>(codec_reference_dot(fp.data<float>() + n * K, values.data() + m * K, K));
            }
            const std::span<const float> actual(read.data<float>(), length);
            require_codec_close(f.type, gemm ? "tiled cuBLAS" : "CUDA GEMV", actual, ref);
            auto& max_error = gemm ? max_gemm : max_gemv;
            max_error = std::max(max_error, codec_rel_l2(actual, ref));
        };
        check(false);
        check(true);

        HostTensor ids({3}, DType::I32);
        ids.data<int32_t>()[0] = N - 1;
        ids.data<int32_t>()[1] = 3;
        ids.data<int32_t>()[2] = 0;
        const auto dev_ids = ctx.upload(ids, false);
        const auto emb = arena.alloc_tensor({3, K}, DType::F32);
        cuda_ops::embed_lookup_f32(w.ptr(), f.type, dev_ids.ptr<int32_t>(), emb.ptr<float>(), 3, K, N, ctx.stream());
        const auto embedded = ctx.download(emb);
        for (int i = 0; i < 3; ++i)
            require_codec_close(f.type, "CUDA packed embedding",
                {embedded.data<float>() + i * K, static_cast<size_t>(K)},
                {fp.data<float>() + ids.data<int32_t>()[i] * K, static_cast<size_t>(K)});

        const int I = f.elements;
        const auto gate = packed_fixture(f, {2, I, K}), up = packed_fixture(f, {2, I, K});
        const auto down = packed_fixture(f, {2, K, I});
        auto dg = ctx.upload(gate, false), du = ctx.upload(up, false), dd = ctx.upload(down, false);
        HostTensor route_ids({2}, DType::I32), route_weights({2}, DType::F32);
        route_ids.data<int32_t>()[0] = 1;
        route_ids.data<int32_t>()[1] = 0;
        route_weights.data<float>()[0] = 0.625f;
        route_weights.data<float>()[1] = 0.375f;
        const auto rid = ctx.upload(route_ids, false), rw = ctx.upload(route_weights, false);
        const auto act = arena.alloc_tensor({2, I}, DType::F32);
        const auto moe_out = arena.alloc_tensor({1, K}, DType::F32);
        const auto ref = codec_reference_moe(gate, up, down, values, {route_ids.data<int32_t>(), 2},
                                             {route_weights.data<float>(), 2}, 1, 2);
        for (int resident : {2, 1}) {
            cuda_ops::moe_indexed_swiglu_gemv(dx.ptr<float>(), rid.ptr<int32_t>(), dg.ptr(), f.type,
                du.ptr(), f.type, dd.ptr(), f.type, rw.ptr<float>(), moe_out.ptr<float>(),
                act.ptr<float>(), K, I, 2, resident, ctx.stream());
            const auto moe_host = ctx.download(moe_out);
            std::vector<float> mixed(moe_host.data<float>(), moe_host.data<float>() + K);
            if (resident < 2) {
                std::vector<float> cold(K);
                CpuMoeWorkspace workspace;
                cpu_moe_accumulate_layer(nullptr, workspace, gate, up, down, route_ids.data<int32_t>(),
                    route_weights.data<float>(), resident, 0, 1, 2, values.data(), cold.data(), CpuMoeMode::Parallel);
                for (int i = 0; i < K; ++i) mixed[i] += cold[i];
            }
            require_codec_close(f.type, "CUDA indexed or hybrid MoE", mixed, ref);
            max_moe = std::max(max_moe, codec_rel_l2(mixed, ref));
        }
        arena.reset(mark);
    }
    ctx.set_mmq_crossover_m(crossover);
    std::cout << "  [PASS] 13 packed CUDA codecs, 5 fallback formats max_rel_l2 GEMV=" << max_gemv
              << " tiled cuBLAS=" << max_gemm << " indexed/hybrid MoE=" << max_moe << '\n';
}

void test_cuda_codec_int8(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const int crossover = ctx.mmq_crossover_m();
    ctx.set_mmq_crossover_m(16);
    double max_gemv = 0.0, max_mmq = 0.0, max_integer = 0.0;
    for (const auto& f : codec_fixtures) {
        if (!has_weight_codec_int8(f.type)) continue;
        constexpr int N = 71, max_rows = 256;
        const int K = 3 * f.elements;
        const auto src = packed_fixture(f, {N, K});
        const auto w32 = model_io::to_f32_host(src);
        auto values = codec_input(max_rows, K);
        // Include a completely zero activation block and exercise general Q8 rounding.
        std::fill_n(values.begin(), 32, 0.0f);
        HostTensor input({max_rows, K}, DType::F32);
        std::copy(values.begin(), values.end(), input.data<float>());
        const auto x = ctx.upload(input, false), w = ctx.upload(src, false);
        const auto q = arena.alloc_tensor({max_rows, K}, DType::Q8_1);
        cuda_ops::quantize_x_q8_1(x.ptr<float>(), nullptr, nullptr, nullptr, q.ptr(),
                                 max_rows, K, 0, ctx.stream());
        const auto qhost = ctx.download(q);
        const auto* qblocks = qhost.data<BlockQ8_1>();
        std::vector<float> rounded(values.size());
        for (size_t i = 0; i < rounded.size(); ++i) {
            const auto& b = qblocks[i / 32];
            rounded[i] = b.d * reinterpret_cast<const int8_t*>(b.qs)[i % 32];
        }
        double format_gemv = 0.0, format_mmq = 0.0;
        for (const int rows : {1, 2, 3, 4, 8, 16, 37, 128, 256}) {
            const auto saved = arena.mark();
            auto out = arena.alloc_tensor({rows, N}, DType::F32);
            auto linear = [&] { model_io::vision_linear_cuda(ctx, x, w, out, rows); };
            linear();
            const auto eager = ctx.download(out);
            CudaGraphExec graph;
            graph.capture(ctx.stream(), linear);
            graph.launch(ctx.stream());
            const auto replay = ctx.download(out);
            const size_t count = static_cast<size_t>(rows) * N;
            if (!std::equal(eager.data<float>(), eager.data<float>() + count, replay.data<float>()))
                throw std::runtime_error(std::string(dtype_name(f.type)) + " INT8 graph differs from eager");
            if (rows > 1 && rows <= 8) {
                auto single = arena.alloc_tensor({1, N}, DType::F32);
                for (int row = 0; row < rows; ++row) {
                    const auto x_row = x.slice_dim0(row, 1);
                    model_io::vision_linear_cuda(ctx, x_row, w, single, 1);
                    const auto ref = ctx.download(single);
                    require_codec_close(f.type, "single-pass GEMV versus single row",
                        {eager.data<float>() + row * N, N}, {ref.data<float>(), N});
                }
            }
            std::vector<float> ref(count), int_ref(count);
            for (int m = 0; m < rows; ++m) for (int n = 0; n < N; ++n) {
                const auto* weights = w32.data<float>() + n * K;
                ref[m * N + n] = static_cast<float>(codec_reference_dot(weights, values.data() + m * K, K));
                int_ref[m * N + n] = static_cast<float>(codec_reference_dot(weights, rounded.data() + m * K, K));
            }
            std::span<const float> actual(eager.data<float>(), count);
            const double error = codec_rel_l2(actual, ref);
            if (!(error < 0.009))
                throw std::runtime_error(std::string(dtype_name(f.type)) + " INT8 relative L2 vs FP32=" +
                                         std::to_string(error) + " at M=" + std::to_string(rows));
            require_codec_close(f.type, "INT8 vs dequantized Q8_1 reference", actual, int_ref);
            max_integer = std::max(max_integer, codec_rel_l2(actual, int_ref));
            auto& max_error = rows <= 8 ? format_gemv : format_mmq;
            max_error = std::max(max_error, error);
            arena.reset(saved);
        }
        max_gemv = std::max(max_gemv, format_gemv);
        max_mmq = std::max(max_mmq, format_mmq);
        std::cout << "    " << dtype_name(f.type) << " FP32 rel_l2 GEMV=" << format_gemv
                  << " MMQ=" << format_mmq << '\n';
        arena.reset(mark);
    }
    ctx.set_mmq_crossover_m(crossover);
    std::cout << "  [PASS] 8 integer codecs GEMV rel_l2=" << max_gemv << " MMQ rel_l2=" << max_mmq
              << " exact Q8_1 reference rel_l2=" << max_integer << " and exact graph replay\n";
}

void test_cuda_codec_model(CudaContext& ctx) {
    struct File {
        const char* path = "test-codec-model.gguf";
        ~File() { std::error_code ec; std::filesystem::remove(path, ec); }
    } file;
    const std::vector<int32_t> ids{1, 9, 2};
    for (const auto& f : codec_fixtures) {
        const int H = f.elements * 2;
        constexpr int V = 37;
        const bool folded = f.type == DType::PTQ1_0;
        constexpr int block = 128;
        const std::vector<int8_t> signs = [&] {
            std::vector<int8_t> result(H);
            for (int i = 0; i < H; ++i) result[i] = (i % 3) ? 1 : -1;
            return result;
        }();

        std::map<std::string, GgufMetaValue> meta{
            {"general.architecture", std::string("llama")}, {"llama.block_count", 1u},
            {"llama.embedding_length", static_cast<uint32_t>(H)}, {"llama.feed_forward_length", static_cast<uint32_t>(H)},
            {"llama.attention.head_count", static_cast<uint32_t>(H / 64)},
            {"llama.context_length", 96u}
        };
        std::vector<std::pair<std::string, HostTensor>> weights{
            {"token_embd.weight", packed_fixture(f, {V, H})},
            {"output_norm.weight", model_io::make_ones_f32({H})},
            {"output.weight", packed_fixture(f, {V, H})},
            {"blk.0.attn_norm.weight", model_io::make_ones_f32({H})},
            {"blk.0.ffn_norm.weight", model_io::make_ones_f32({H})},
            {"blk.0.attn_q.weight", model_io::make_zeros_f32({H, H})},
            {"blk.0.attn_k.weight", model_io::make_zeros_f32({H, H})},
            {"blk.0.attn_v.weight", model_io::make_zeros_f32({H, H})},
            {"blk.0.attn_output.weight", model_io::make_zeros_f32({H, H})},
            {"blk.0.ffn_gate.weight", packed_fixture(f, {H, H})},
            {"blk.0.ffn_up.weight", packed_fixture(f, {H, H})},
            {"blk.0.ffn_down.weight", packed_fixture(f, {H, H})},
        };
        const std::vector<std::string> names{"output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"};
        const auto actual_logits = [&] {
            std::string error;
            auto model_meta = meta;
            if (folded) {
                model_meta["prism.hadamard.version"] = 1u;
                model_meta["prism.hadamard.block_size"] = static_cast<uint32_t>(block);
                model_meta["prism.hadamard.transform"] = std::string("normalized-sylvester-walsh-hadamard");
                model_meta["prism.hadamard.axis"] = std::string("input-last-dimension");
                model_meta["prism.hadamard.sign_mode"] = std::string("explicit");
                model_meta["prism.hadamard.sign_widths"] = std::vector<uint32_t>{static_cast<uint32_t>(H)};
                model_meta["prism.hadamard.sign_values"] = std::vector<uint32_t>(signs.begin(), signs.end());
                model_meta["prism.hadamard.weight_names"] = names;
                model_meta["prism.hadamard.inverse_weight_names"] = std::vector<std::string>{"token_embd.weight"};
            }
            if (folded) {
                auto bad = model_meta;
                bad["prism.hadamard.sign_mode"] = std::string("seeded");
                model_io::LoadedModelRaw raw;
                if (!GgufFile::write_v3(file.path, bad, weights, &error) ||
                    model_io::parse_gguf_model_raw(file.path, 96, raw, &error) ||
                    error.find("sign mode") == std::string::npos)
                    throw std::runtime_error("Unknown Prism sign mode should be rejected");
            }
            if (!GgufFile::write_v3(file.path, model_meta, weights, &error)) throw std::runtime_error(error);
            FusionModel model(HybridModelConfig{});
            if (!model.load_gguf(ctx, file.path, &error, 96)) throw std::runtime_error(error);
            return model.forward_prefill_cuda(ctx, ids, 0);
        }();

        for (auto& [name, weight] : weights) {
            if (!is_weight_codec_dtype(weight.desc.dtype)) continue;
            weight = model_io::to_f32_host(weight);
            if (folded && (name == "token_embd.weight" || std::ranges::find(names, name) != names.end()))
                blockwise_hadamard_f32(weight.data<float>(), signs.data(), weight.data<float>(),
                                        static_cast<int>(weight.desc.shape[0]), H, block, true);
        }
        std::string error;
        FusionModel reference(HybridModelConfig{});
        if (!GgufFile::write_v3(file.path, meta, weights, &error) ||
            !reference.load_gguf(ctx, file.path, &error, 96)) throw std::runtime_error(error);
        const auto expected = reference.forward_prefill_cuda(ctx, ids, 0);
        const double tolerance = has_weight_codec_int8(f.type) ? 0.015 : 1e-5;
        if (!(codec_rel_l2(actual_logits, expected) < tolerance))
            throw std::runtime_error(std::string(dtype_name(f.type)) + " loaded CUDA model logits relative L2=" +
                                     std::to_string(codec_rel_l2(actual_logits, expected)));
    }
    std::cout << "  [PASS] 13 loaded CUDA models, embeddings/LM head and Prism explicit rotation\n";
}

void test_cuda_hadamard(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    double max_error = 0;
    for (const auto block : {64, 128, 256, 512, 1024}) {
        constexpr int M = 3;
        const int cols = 2 * block;
        HostTensor x({M, cols}, DType::F32), weight({cols}, DType::F32), signs({cols}, DType::U8);
        const auto input = codec_input(M, cols);
        std::copy(input.begin(), input.end(), x.data<float>());
        for (int i = 0; i < cols; ++i) {
            signs.data<int8_t>()[i] = (i % 3) ? 1 : -1;
            weight.data<float>()[i] = 0.75f + i % 8 / 16.f;
        }
        const auto dx = ctx.upload(x, false), dw = ctx.upload(weight, false), ds = ctx.upload(signs, false);
        const auto y = arena.alloc_tensor({M, cols}, DType::F32);
        std::vector<float> ref(input.size());
        for (const bool fused : {false, true}) {
            if (fused) {
                fused_rmsnorm_hadamard_f32(input.data(), weight.data<float>(), signs.data<int8_t>(), ref.data(), M, cols, block, 1e-5f);
                cuda_ops::fused_rmsnorm_hadamard_f32(dx.ptr<float>(), dw.ptr<float>(), ds.ptr<int8_t>(), y.ptr<float>(),
                                                       M, cols, block, 1e-5f, ctx.stream());
            } else {
                blockwise_hadamard_f32(input.data(), signs.data<int8_t>(), ref.data(), M, cols, block);
                cuda_ops::blockwise_hadamard_f32(dx.ptr<float>(), ds.ptr<int8_t>(), y.ptr<float>(),
                                                  M, cols, block, ctx.stream());
            }
            const auto h = ctx.download(y);
            const std::span<const float> actual(h.data<float>(), input.size());
            require_codec_close(DType::F32, "CUDA FWHT", actual, ref);
            max_error = std::max(max_error, codec_rel_l2(actual, ref));
        }
        const int head_dim = block / 4;
        constexpr int key_heads = 2, repeats = 4;
        auto grouped = input;
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < key_heads; ++k) {
                for (int r = 0; r < repeats; ++r) {
                    const int src = m * cols + (r * key_heads + k) * head_dim;
                    const int dst = m * cols + (k * repeats + r) * head_dim;
                    std::copy_n(input.data() + src, head_dim, grouped.data() + dst);
                }
            }
        }
        blockwise_hadamard_f32(grouped.data(), signs.data<int8_t>(), ref.data(), M, cols, block);
        cuda_ops::blockwise_hadamard_f32(dx.ptr<float>(), ds.ptr<int8_t>(), y.ptr<float>(), M, cols,
                                          block, ctx.stream(), false, head_dim, key_heads);
        const auto h = ctx.download(y);
        require_codec_close(DType::F32, "CUDA grouped GDN FWHT", {h.data<float>(), input.size()}, ref);
        arena.reset(mark);
    }
    std::cout << "  [PASS] FWHT 64-1024 CUDA max_rel_l2=" << max_error << '\n';
}
#endif
void test_q2_k_format() {
    constexpr int rows = 3, cols = 512;
    HostTensor weights({rows, cols}, DType::Q2_K);
    std::vector<float> decoded(rows * cols);
    std::vector<double> reference(rows * cols);
    for (int b = 0; b < rows * cols / 256; ++b) {
        auto& block = weights.data<BlockQ2_K>()[b];
        block.d = fp32_to_fp16(0.125f);
        block.dmin = fp32_to_fp16(0.0625f);
        std::fill_n(block.qs, 32, uint8_t{0xe4});
        std::fill_n(block.qs + 32, 32, uint8_t{0x1b});
        for (int s = 0; s < 16; ++s) {
            const int scale = 1 + (s + b) % 15, minimum = (2 * s + b) % 16;
            block.scales[s] = static_cast<uint8_t>(scale + 16 * minimum);
            const int q = s < 8 ? s / 2 : 7 - s / 2;
            for (int i = 0; i < 16; ++i) reference[b * 256 + s * 16 + i] = scale * q / 8.0 - minimum / 16.0;
        }
    }
    dequantize_row_q2_k(weights.data<BlockQ2_K>(), decoded.data(), rows * cols);
    if (!std::equal(decoded.begin(), decoded.end(), reference.begin()))
        throw std::runtime_error("Q2_K block packing mismatch");

    std::mt19937_64 rng(2027);
    const auto input = rand_f32({cols}, rng);
    std::array<CpuQ8K, cols / 256> activation;
    cpu_moe_quantize_q8(input.data<float>(), activation.data(), cols);
    std::array<float, cols> x{};
    for (int i = 0; i < cols; ++i) x[i] = activation[i / 256].scale * activation[i / 256].values[i % 256];
    for (const auto& [name, dot] : std::array{
             std::pair{"AVX2", cpu_moe_avx2_integer_dot(DType::Q2_K)},
             std::pair{"VNNI-256", cpu_moe_vnni_dot(DType::Q2_K)},
             std::pair{"VNNI-512", cpu_moe_vnni_dot(DType::Q2_K, true)}}) {
        if (!dot) continue;
        std::vector<double> expected;
        std::vector<float> actual;
        for (int r = 0; r < rows; ++r) {
            expected.push_back(std::inner_product(x.begin(), x.end(), reference.begin() + r * cols, 0.0));
            actual.push_back(dot(weights.bytes().data() + r * row_nbytes(DType::Q2_K, cols), activation.data(), cols));
        }
        expect_close(actual, expected, 1e-5, 1e-6, std::string("Q2_K x Q8_K ") + name);
    }

    HostTensor source({1, 256}, DType::F32);
    for (int i = 0; i < 256; ++i) source.data<float>()[i] = static_cast<float>(i % 4 - 2);
    auto encoded = model_io::quantize_host_matrix(source, DType::Q2_K);
    const auto roundtrip = model_io::to_f32_host(encoded);
    expect_close({roundtrip.data<float>(), roundtrip.data<float>() + 256},
                 {source.data<float>(), source.data<float>() + 256}, 0.001, 0.001, "Q2_K quantize/dequantize");
    struct Fixture {
        std::filesystem::path path = "test-q2-k.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    std::string error;
    if (!GgufFile::write_v3(fixture.path.string(), {}, {{"weights", weights}}, &error)) throw std::runtime_error(error);
    GgufFile gguf;
    if (!gguf.load(fixture.path.string(), &error)) throw std::runtime_error(error);
    const auto view = gguf.view_tensor("weights");
    if (!view.storage.empty() || view.desc.dtype != DType::Q2_K ||
        !std::equal(view.bytes().begin(), view.bytes().end(), weights.bytes().begin(), weights.bytes().end()))
        throw std::runtime_error("Q2_K mapped GGUF roundtrip differs");
}

static void test_host_repacking() {
    constexpr int rows = 32, cols = 512;
    std::mt19937_64 rng(102);
    auto check = [&]<typename Src, typename Dst>(DType src_dtype, DType dst_dtype,
                                                 void (*legacy)(const Src*, Dst*, int64_t)) {
        HostTensor src({rows, cols}, src_dtype);
        for (auto& byte : src.storage) byte = static_cast<uint8_t>(rng());
        HostTensor expected({rows, cols}, dst_dtype);
        legacy(src.data<Src>(), expected.data<Dst>(), rows * cols);
        HostTensor actual({rows, cols}, dst_dtype);
        std::fill(actual.storage.begin(), actual.storage.end(), uint8_t{0xff});
        if (!repack::can_repack_for_mmq(src_dtype, rows, cols) ||
            repack::repacked_dtype_for(src_dtype, rows, cols, true) != dst_dtype ||
            repack::repacked_nbytes(dst_dtype, rows, cols) != actual.storage.size() ||
            !repack::repack_matrix_blocks(src.storage.data(), src_dtype, actual.storage.data(), dst_dtype, rows, cols) ||
            actual.storage != expected.storage) {
            throw std::runtime_error("Host repacking differs from the block codec");
        }
    };
    check(DType::Q4_0, DType::Q4_A, repack_q4_0_to_q4_a);
    check(DType::Q8_0, DType::Q8_A, repack_q8_0_to_q8_a);
    check(DType::Q4_K, DType::Q4_KA, repack_q4_k_to_q4_ka);
    check(DType::Q6_K, DType::Q6_A, repack_q6_k_to_q6_a);

    if (repack::repacked_dtype_for(DType::Q6_K, rows, cols) != DType::Q6_K ||
        repack::can_repack_for_mmq(DType::Q4_0, rows - 1, cols) ||
        repack::can_repack_for_mmq(DType::Q8_0, rows, cols - 32) ||
        repack::can_repack_for_mmq(DType::F16, rows, cols) ||
        repack::repacked_nbytes(DType::Q4_A, 0, cols) != 0) {
        throw std::runtime_error("Host repacking accepted an unsuitable MMQ format or shape");
    }

    HostTensor src({1, cols}, DType::Q4_K);
    HostTensor dst({1, cols}, DType::Q4_KA);
    if (repack::repack_matrix_blocks(src.storage.data(), DType::Q6_K, dst.storage.data(), DType::Q4_KA, 1, cols) ||
        repack::repack_matrix_blocks(src.storage.data(), DType::Q4_K, dst.storage.data(), DType::Q4_KA, 1, cols - 32) ||
        !repack::repack_matrix_blocks(src.storage.data(), DType::Q4_K, dst.storage.data(), DType::Q4_KA, 1, cols)) {
        throw std::runtime_error("Host repacking mishandled a caller-owned buffer");
    }
}

void test_quant_and_io() {
    std::cout << "[1/7] Testing Q8_0 / Q4_0 quantization, GGUF v3, Safetensors, BPE...\n";
    test_host_repacking();
    std::mt19937_64 rng(101);
    HostTensor src = rand_f32({4, 128}, rng, 0.5f);
    std::vector<BlockQ8_0> q8(4 * 4);
    std::vector<BlockQ4_0> q4(4 * 4);
    std::vector<float> dq8(512), dq4(512);
    for (int r = 0; r < 4; ++r) {
        quantize_row_q8_0(src.data<float>() + r * 128, q8.data() + r * 4, 128);
        dequantize_row_q8_0(q8.data() + r * 4, dq8.data() + r * 128, 128);
        quantize_row_q4_0(src.data<float>() + r * 128, q4.data() + r * 4, 128);
        dequantize_row_q4_0(q4.data() + r * 4, dq4.data() + r * 128, 128);
    }
    std::vector<double> ref(src.data<float>(), src.data<float>() + 512);
    expect_close(dq8, ref, 0.03, 0.02, "Q8_0 roundtrip");
    expect_close(dq4, ref, 0.35, 0.15, "Q4_0 roundtrip");

    auto tmp_dir = std::filesystem::temp_directory_path();
    std::string gguf_path = (tmp_dir / "fusionrt_test.gguf").string();
    std::string st_path = (tmp_dir / "fusionrt_test.safetensors").string();

    std::map<std::string, GgufMetaValue> meta;
    meta["general.architecture"] = std::string("dense");
    meta["tokenizer.ggml.tokens"] = std::vector<std::string>{"<pad>", "<bos>", "<eos>", "<unk>", "h", "e", "l", "o", "he", "ll", "hello"};
    meta["tokenizer.ggml.merges"] = std::vector<std::string>{"h e", "l l", "he ll", "hell o"};
    std::string err;
    if (!GgufFile::write_v3(gguf_path, meta, {{"test.weight", src}}, &err)) {
        throw std::runtime_error("GGUF write failed: " + err);
    }
    GgufFile gf;
    if (!gf.load(gguf_path, &err)) throw std::runtime_error("GGUF load failed: " + err);
    HostTensor gf_w = gf.read_tensor("test.weight");
    expect_close(std::vector<float>(gf_w.data<float>(), gf_w.data<float>() + 512), ref, 1e-6, 1e-6, "GGUF v3 roundtrip");
    {
        GgufFile file;
        if (!file.load(gguf_path, &err, false)) throw std::runtime_error(err);
        auto view = file.view_tensor("test.weight");
        const auto* bytes = view.bytes().data();
        const size_t row_bytes = view.bytes().size() / view.desc.shape[0];
        const auto slice = view.slice_dim0(1, 1);
        file = {};
        view = {};
        if (!slice.storage.empty() || slice.bytes().data() != bytes + row_bytes ||
            std::memcmp(slice.bytes().data(), gf_w.bytes().data() + row_bytes, row_bytes) != 0) {
            throw std::runtime_error("GGUF mapped slice must retain its owner and alias file bytes");
        }
    }

    BpeTokenizer tok;
    if (!tok.load_from_gguf(gf)) throw std::runtime_error("Tokenizer load from GGUF failed");
    auto ids = tok.encode("hello", true);
    if (ids.empty() || ids[0] != 1) throw std::runtime_error("Tokenizer BOS check failed");
    {
        const auto path = (tmp_dir / "fusionrt_gemma4_vocab.gguf").string();
        const std::string space = "\xe2\x96\x81";
        const std::map<std::string, GgufMetaValue> gemma{
            {"tokenizer.ggml.model", std::string("gemma4")},
            {"tokenizer.ggml.tokens", std::vector<std::string>{
                "<pad>", "<bos>", "<unk>", "<|turn>", "<turn|>", "\n", "\n\n", "a", "b", space, space + "b", "<0x09>"}},
            {"tokenizer.ggml.merges", std::vector<std::string>{space + " b"}},
            {"tokenizer.ggml.bos_token_id", uint32_t{1}},
            {"tokenizer.ggml.eos_token_id", uint32_t{4}},
            {"tokenizer.ggml.unknown_token_id", uint32_t{2}},
            {"tokenizer.ggml.add_bos_token", true}
        };
        GgufFile file;
        BpeTokenizer tokenizer;
        if (!GgufFile::write_v3(path, gemma, {{"test.weight", src}}, &err) ||
            !file.load(path, &err) || !tokenizer.load_from_gguf(file)) throw std::runtime_error(err);
        file = {};
        std::filesystem::remove(path);
        const std::vector<int32_t> expected{1, 7, 10, 6, 11};
        if (tokenizer.encode("a b\n\n\t") != expected || tokenizer.decode(expected) != "a b\n\n\t" ||
            !tokenizer.is_end_of_generation(4) ||
            tokenizer.format_chat_messages({{"user", " a "}}) != "<|turn>user\na<turn|>\n<|turn>model\n") {
            throw std::runtime_error("Gemma 4 metaspace BPE and turn format");
        }
    }

    // GPT-2 / Qwen byte-level BPE + special token regression test
    {
        std::string gpt2_path = (tmp_dir / "fusionrt_gpt2_test.gguf").string();
        std::map<std::string, GgufMetaValue> gmeta;
        gmeta["general.architecture"] = std::string("qwen3");
        gmeta["tokenizer.ggml.model"] = std::string("gpt2");
        gmeta["tokenizer.ggml.eos_token_id"] = int32_t(8);
        gmeta["tokenizer.ggml.add_bos_token"] = false;
        gmeta["tokenizer.ggml.tokens"] = std::vector<std::string>{
            "h", "e", "l", "o", "\xC4\xA0", "\xC4\xA0o", "ll", "<|im_start|>", "<|im_end|>"
        };
        gmeta["tokenizer.ggml.merges"] = std::vector<std::string>{"l l", "\xC4\xA0 o"};
        if (!GgufFile::write_v3(gpt2_path, gmeta, {{"test.weight", src}}, &err)) {
            throw std::runtime_error("GPT2 GGUF write failed: " + err);
        }
        GgufFile ggf;
        if (!ggf.load(gpt2_path, &err)) throw std::runtime_error("GPT2 GGUF load failed: " + err);
        BpeTokenizer gtok;
        if (!gtok.load_from_gguf(ggf)) throw std::runtime_error("GPT2 tokenizer load failed");
        auto gids = gtok.encode("<|im_start|>hello o<|im_end|>", true);
        // Should NOT prepend BOS, should match <|im_start|>=7, h=0,e=1,ll=6,o=3,臓o=5,<|im_end|>=8
        std::vector<int32_t> expect_gids = {7, 0, 1, 6, 3, 5, 8};
        if (gids != expect_gids) {
            throw std::runtime_error("GPT2 byte BPE encode mismatch");
        }
        if (gtok.decode(gids) != "<|im_start|>hello o") {
            throw std::runtime_error("GPT2 byte BPE decode mismatch: " + gtok.decode(gids));
        }
        ggf = {};
        std::filesystem::remove(gpt2_path);

        gmeta["tokenizer.ggml.pre"] = std::string("smollm");
        gmeta["tokenizer.ggml.tokens"] = std::vector<std::string>{
            "a", "'", "s", "'s", "\xC4\x8A", "\xC4\x8A\xC4\x8A", "1", "2", "<|im_end|>"
        };
        gmeta["tokenizer.ggml.merges"] = std::vector<std::string>{"' s", "\xC4\x8A \xC4\x8A"};
        if (!GgufFile::write_v3(gpt2_path, gmeta, {{"test.weight", src}}, &err) ||
            !ggf.load(gpt2_path, &err) || !gtok.load_from_gguf(ggf) ||
            gtok.encode("a's\n\n12\n\na") != std::vector<int32_t>{0, 3, 5, 6, 7, 4, 4, 0}) {
            throw std::runtime_error("SmolLM pre-tokenization mismatch");
        }
        ggf = {};
        std::filesystem::remove(gpt2_path);
    }

    if (!SafetensorsFile::write(st_path, {{"format", "pt"}}, {{"test.weight", src}}, &err)) {
        throw std::runtime_error("Safetensors write failed: " + err);
    }
    SafetensorsFile sf;
    if (!sf.load(st_path, &err)) throw std::runtime_error("Safetensors load failed: " + err);
    HostTensor sf_w = sf.read_tensor("test.weight");
    expect_close(std::vector<float>(sf_w.data<float>(), sf_w.data<float>() + 512), ref, 1e-6, 1e-6, "Safetensors roundtrip");
    gf = {};
    std::filesystem::remove(gguf_path);
    std::filesystem::remove(st_path);
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_q2_k(CudaContext& ctx) {
    constexpr int rows = 19, width = 512, out = 35;
    std::mt19937_64 rng(2027);
    const auto x = rand_f32({rows, width}, rng);
    const auto w = model_io::quantize_host_matrix(rand_f32({out, width}, rng), DType::Q2_K);
    const auto w_f32 = model_io::to_f32_host(w);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const auto dx = ctx.upload(x, false), dw = ctx.upload(w, false);
    auto check = [&](int m, double tol, const char* name) {
        auto y = arena.alloc_tensor({m, out}, DType::F32);
        cuda_ops::linear_q2_k_f32(dx.ptr<float>(), dw.ptr(), y.ptr<float>(), m, out, width, ctx.stream());
        const auto actual = ctx.download(y);
        std::vector<double> expected(m * out);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < out; ++j) {
                const auto* xv = x.data<float>() + i * width;
                expected[i * out + j] = std::inner_product(xv, xv + width, w_f32.data<float>() + j * width, 0.0);
            }
        expect_close({actual.data<float>(), actual.data<float>() + m * out}, expected, tol * 6, tol, name);
    };
    check(1, 1e-6, "Q2_K CUDA decode vs FP64");
    check(rows, 0.001, "Q2_K CUDA prefill vs FP64");
    auto ids = arena.alloc_tensor({1}, DType::I32);
    const int32_t id = 4;
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ids.ptr(), &id, sizeof(id), cudaMemcpyHostToDevice, ctx.stream()));
    auto embedding = arena.alloc_tensor({1, width}, DType::F32);
    cuda_ops::embed_lookup_f32(dw.ptr(), DType::Q2_K, ids.ptr<int32_t>(), embedding.ptr<float>(), 1, width, out, ctx.stream());
    const auto downloaded = ctx.download(embedding);
    if (!std::equal(downloaded.data<float>(), downloaded.data<float>() + width, w_f32.data<float>() + id * width))
        throw std::runtime_error("Q2_K embedding lookup differs from scalar");
    arena.reset(mark);
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
void test_repacked_quants(CudaContext& ctx) {
    constexpr int M = 129, N = 4097, K = 256;
    std::mt19937_64 rng(91);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const auto x = ctx.upload(rand_f32({M, K}, rng), false);
    const auto qx = arena.alloc_tensor({M, K}, DType::Q8_1);
    const auto y = arena.alloc_tensor({M, N}, DType::F32);
    cuda_ops::quantize_x_q8_1(x.ptr<float>(), nullptr, nullptr, nullptr, qx.ptr(), M, K, 0, ctx.stream());
    const auto activation = ctx.download(qx);
    std::vector<float> full_x(M * K);
    for (int i = 0; i < M * K; ++i) {
        const auto& blk = activation.data<BlockQ8_1>()[i / 32];
        full_x[i] = blk.d * reinterpret_cast<const int8_t*>(blk.qs)[i % 32];
    }
    auto ids = HostTensor::create({2}, DType::I32);
    ids.data<int32_t>()[0] = N - 1;
    ids.data<int32_t>()[1] = 0;
    const auto d_ids = ctx.upload(ids, false);
    const auto embeddings = arena.alloc_tensor({2, K}, DType::F32);
    auto check = [&]<typename Raw, typename Packed>(DType raw_type, DType packed_type,
        void (*repack)(const Raw*, Packed*, int64_t), void (*dequant)(const Raw*, float*, int64_t),
        void (*unpack)(const Packed*, float*, int64_t), auto gemv, auto mmq) {
        auto raw = HostTensor::create({N, K}, raw_type);
        auto packed = HostTensor::create({N, K}, packed_type);
        for (auto& byte : raw.storage) byte = static_cast<uint8_t>(rng());
        for (size_t n = 0; n < static_cast<size_t>(raw.desc.numel()) / dtype_block_size(raw_type); ++n) {
            raw.data<Raw>()[n].d = fp32_to_fp16(0.00021f);
            if constexpr (std::is_same_v<Raw, BlockQ4_K>) raw.data<Raw>()[n].dmin = fp32_to_fp16(0.00031f);
        }
        std::vector<float> reference(N * K), restored(N * K);
        repack(raw.data<Raw>(), packed.data<Packed>(), N * K);
        dequant(raw.data<Raw>(), reference.data(), N * K);
        unpack(packed.data<Packed>(), restored.data(), N * K);
        if (reference != restored) throw std::runtime_error("K-quant repacking changed dequantized weights");

        const std::string label = dtype_name(packed_type);
        const auto w = ctx.upload(packed, false);
        cuda_ops::embed_lookup_f32(w.ptr(), packed_type, d_ids.ptr<int32_t>(), embeddings.ptr<float>(), 2, K, N, ctx.stream());
        const auto embed = ctx.download(embeddings);
        std::vector<double> expect_embed(reference.end() - K, reference.end());
        expect_embed.insert(expect_embed.end(), reference.begin(), reference.begin() + K);
        expect_close({embed.data<float>(), embed.data<float>() + 2 * K}, expect_embed, 1e-6, 1e-6, label + " embedding");

        std::vector<double> expected(M * N);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) expected[m * N + n] += double(full_x[m * K + k]) * reference[n * K + k];
            }
        }
        gemv(qx.ptr(), w.ptr(), y.ptr<float>(), N, K, ctx.stream(), GemvGeometry{4, 2});
        auto got = ctx.download(y);
        expect_close({got.data<float>(), got.data<float>() + N}, {expected.begin(), expected.begin() + N},
                     3e-5, 1e-5, label + " GEMV");
        mmq(qx.ptr(), w.ptr(), y.ptr<float>(), M, N, K, ctx.stream(), ctx.cost_hardware());
        got = ctx.download(y);
        expect_close({got.data<float>(), got.data<float>() + M * N}, expected, 3e-5, 1e-5, label + " MMQ");
    };
    check(DType::Q4_0, DType::Q4_A, repack_q4_0_to_q4_a, dequantize_row_q4_0, dequantize_row_q4_a,
          cuda_ops::gemv_q4_a_q8_1, cuda_ops::mmq_q4_a_q8_1);
    check(DType::Q4_K, DType::Q4_KA, repack_q4_k_to_q4_ka, dequantize_row_q4_k, dequantize_row_q4_ka,
          cuda_ops::gemv_q4_ka_q8_1, cuda_ops::mmq_q4_ka_q8_1);
    std::cout << "  [PASS] TileQ4_144 Q4_A and Q4_KA GEMV/MMQ parity\n";
    check(DType::Q6_K, DType::Q6_A, repack_q6_k_to_q6_a, dequantize_row_q6_k, dequantize_row_q6_a,
          cuda_ops::gemv_q6_a_q8_1, cuda_ops::mmq_q6_a_q8_1);
    arena.reset(mark);
}
#endif

}  // namespace fusionrt::test
