#ifdef FUSIONRT_ENABLE_CUDA
#include "../src/server/http_transport.h"
#include "../src/models/cuda_dispatch.h"
#endif
#include "test_support.h"
#include <future>

namespace fusionrt::test {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TestFile {
    std::string path;
    ~TestFile() { std::filesystem::remove(path); }
};

}  // namespace

void test_wave1_inputs() {
    require(fp32_to_fp16(1.f + std::ldexp(1.f, -11)) == 0x3c00 &&
            fp32_to_fp16(1.f + 3 * std::ldexp(1.f, -11)) == 0x3c02 &&
            fp32_to_fp16(std::ldexp(1.f, -25)) == 0 &&
            fp32_to_fp16(-3 * std::ldexp(1.f, -25)) == 0x8002,
            "FP16 conversion must round normal and subnormal ties to even");

    const auto up = make_ones_f32({1, 2, 2});
    float input[] = {1, 1}, output[] = {0, 0};
    cpu_moe_accumulate(up, up, up, 0, input, 1, output, CpuMoeMode::Scalar);
    const float contribution = output[0];
    require(contribution > 0, "MoE fixture did not contribute");
    for (int id : {-1, 1, INT32_MAX})
        cpu_moe_accumulate(up, up, up, id, input, 1, output, CpuMoeMode::Scalar);
    require(output[0] == contribution && output[1] == contribution, "Invalid CPU MoE route changed the output");

    std::array<float, 256> activation{};
    activation[0] = std::numeric_limits<float>::infinity();
    activation[1] = -activation[0];
    activation[2] = std::numeric_limits<float>::quiet_NaN();
    activation[3] = 1.f;
    CpuQ8K large{};
    CpuQ8Small small[8]{};
    cpu_moe_quantize_q8(activation.data(), &large, 256);
    cpu_moe_quantize_q8_small(activation.data(), small, 256);
    for (const auto* packed : {large.values, small[0].values})
        require(packed[0] == 0 && packed[1] == 0 && packed[2] == 0 && packed[3] == 127,
                "CPU activation quantizer must zero non-finite values before scaling");

    TestFile tokenizer_json{"wave1-tokenizer.json"};
    BpeTokenizer tokenizer;
    tokenizer.init_byte_fallback();
    const auto vocab_count = tokenizer.vocab_size();
    for (const auto* body : {
            R"({"model":{"vocab":{"a":-1}}})",
            R"({"model":{"vocab":{"a":0.5}}})",
            R"({"model":{"vocab":{"a":2147483647}}})",
            R"({"model":{"vocab":{"a":0}},"added_tokens":[{"id":-1,"content":"x"}]})"}) {
        std::ofstream(tokenizer_json.path) << body;
        require(!tokenizer.load_from_json(tokenizer_json.path) && tokenizer.vocab_size() == vocab_count,
                "Malformed tokenizer ID was accepted or published");
    }

    TestFile tensor_file{"wave1.safetensors"};
    SafetensorsFile tensors;
    for (const std::string_view header : {
            R"({"w":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}})",
            R"({"w":{"dtype":"F32","shape":[1,1,1,1,1],"data_offsets":[0,4]}})",
            R"({"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
            R"({"w":{"dtype":"F32","shape":[2147483648,2147483648,16],"data_offsets":[0,4]}})",
            R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})",
            R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})"}) {
        const uint64_t size = header.size();
        {
            std::ofstream file(tensor_file.path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            file << header;
            const uint32_t payload = 0;
            file.write(reinterpret_cast<const char*>(&payload), sizeof(payload));
        }
        require(!tensors.load(tensor_file.path), "Malformed Safetensors header accepted");
    }

    bool rejected = false;
    try {
        (void)agent::parse_generation(JsonObject{{"repeat_penalty", 0.99}}, "max_tokens");
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "HTTP repeat_penalty below 1 accepted");

    std::vector<std::string> frames;
    agent::AgentResponse messages(agent::Protocol::messages, "test", "model", 0, false,
        [&](std::string_view event, std::string_view value) {
            if (event == "message_delta") frames.emplace_back(value);
            return true;
        });
    messages.finish(agent::Finish::stop_sequence, 3, "END");
    require(frames.size() == 1, "Anthropic stop_sequence terminal event missing");
    const auto delta = JsonValue::parse(frames[0]).as_object().at("delta").as_object();
    const auto body = messages.body().as_object();
    require(delta.at("stop_reason").as_string() == "stop_sequence" && delta.at("stop_sequence").as_string() == "END" &&
            body.at("stop_reason").as_string() == "stop_sequence" && body.at("stop_sequence").as_string() == "END",
            "Anthropic stop_sequence must be present in JSON and streaming");

    std::string finish_event;
    agent::AgentResponse cancelled(agent::Protocol::responses, "test", "model", 0, false,
        [&](std::string_view event, std::string_view) { finish_event = event; return true; });
    cancelled.finish(agent::Finish::cancelled, 1);
    require(finish_event == "response.incomplete" &&
            cancelled.body().as_object().at("status").as_string() == "incomplete",
            "A cancelled response was reported as completed");
    std::cout << "  [PASS] Wave 1 CPU quantization, metadata validation and terminal protocol contracts\n";
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_wave1_cuda(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const size_t mark = arena.mark();
    const auto ids = [&] {
        HostTensor tokens({3}, DType::I32);
        const int32_t values[] = {-1, 1, 2};
        std::copy(std::begin(values), std::end(values), tokens.data<int32_t>());
        return ctx.upload(tokens, false);
    }();

    constexpr int width = 256;
    HostTensor quant({2, width}, DType::IQ3_S);
    const auto iq3 = fp32_to_fp16(65504.f);
    for (size_t offset = 0; offset < quant.storage.size(); offset += dtype_type_size(quant.desc.dtype)) {
        std::memcpy(quant.storage.data() + offset, &iq3, sizeof(iq3));
        std::fill_n(quant.storage.data() + offset + 106, 4, uint8_t{0x11});
    }
    const auto packed = ctx.upload(quant, false);
    for (const auto& table : {ctx.upload(make_ones_f32({2, width}), false), packed}) {
        auto out = arena.alloc_tensor({3, width}, DType::F32);
        auto invalid = arena.alloc_tensor({1}, DType::I32);
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(invalid.ptr(), 0, sizeof(int32_t), ctx.stream()));
        CudaGraphExec graph;
        graph.capture(ctx.stream(), [&] {
            cuda_ops::embed_lookup_f32(table.ptr(), table.dtype, ids.ptr<int32_t>(), out.ptr<float>(),
                                       3, width, 2, ctx.stream(), invalid.ptr<int>());
        });
        graph.launch(ctx.stream());
        const auto read = ctx.download(out);
        const auto* values = read.data<float>();
        require(std::all_of(values, values + width, [](float v) { return v == 0; }) &&
                std::all_of(values + width * 2, values + width * 3, [](float v) { return v == 0; }) &&
                values[width] > 0 && ctx.download(invalid).data<int>()[0] == 1,
                "CUDA Graph embedding must flag and zero invalid token rows");
    }
    const auto x = ctx.upload(make_ones_f32({32, width}), false);
    auto y = arena.alloc_tensor({32, 2}, DType::F32);
    model_io::vision_linear_cuda(ctx, x, packed, y, 32);
    const auto actual = ctx.download(y);
    const auto expanded = model_io::to_f32_host(quant);
    const float reference = std::accumulate(expanded.data<float>(), expanded.data<float>() + width, 0.f);
    require(std::abs(actual.data<float>()[0] / reference - 1.f) < 1e-3f,
            "Extreme finite IQ3_S weight was distorted or overflowed in tiled FP16 GEMM");
    cuda_ops::codec_gemv_f32(nullptr, nullptr, DType::IQ3_S, nullptr, 0, 2, width, ctx.stream());
    cuda_ops::codec_dequant_tile_f16(nullptr, DType::IQ3_S, nullptr, 0, 0, width, ctx.stream());

    for (const int cols : {7, 8}) {
        const auto unaligned = ctx.upload(make_ones_f32({1 + 2 * cols}), false);
        auto result = arena.alloc_tensor({2, 2}, DType::F32);
        cuda_ops::linear_f32(unaligned.ptr<float>() + 1, unaligned.ptr<float>() + 1,
                             result.ptr<float>(), 2, 2, cols, ctx.stream());
        const auto read = ctx.download(result);
        require(std::all_of(read.data<float>(), read.data<float>() + 4, [cols](float v) { return v == cols; }),
                "Unaligned F32 matrix rows must use safe loads");
        for (const auto type : {DType::F16, DType::BF16}) {
            HostTensor low({2 * cols}, type);
            std::fill_n(low.data<uint16_t>(), 2 * cols, type == DType::F16 ? 0x3c00 : 0x3f80);
            const auto weight = ctx.upload(low, false);
            auto fused = type == DType::F16 ? cuda_ops::fused_rmsnorm_linear_f16 : cuda_ops::fused_rmsnorm_linear_bf16;
            fused(unaligned.ptr<float>() + 1, unaligned.ptr<float>() + 1, weight.ptr<uint16_t>(),
                  result.ptr<float>(), 2, 2, cols, 1e-6f, ctx.stream());
            const auto normalized = ctx.download(result);
            require(std::all_of(normalized.data<float>(), normalized.data<float>() + 4,
                                [cols](float v) { return std::abs(v - cols) < 1e-5f; }),
                    "Unaligned F16/BF16 normalization must use safe loads");
        }
    }

    constexpr int length = 512;
    CudaKvPool pool;
    CudaKvPool::Table pages;
    const auto cache = make_cuda_kv_view(arena, DType::F16, length, 1, width);
    pool.ensure(ctx, std::span(&cache, 1), pages, 0, length);
    const auto q = ctx.upload(model_io::make_zeros_f32({length, width}), false);
    auto values = make_ones_f32({length, width});
    std::fill_n(values.data<float>(), length * width, 256.f);
    auto v = ctx.upload(values, false);
    auto out = arena.alloc_tensor({length, width}, DType::F32);
    auto queries = ctx.upload(model_io::make_zeros_f32({length, width}), false);
    auto keys = ctx.upload(model_io::make_zeros_f32({length, width}), false);
    cuda_ops::batched_qk_norm_rope_kv_append_f32(queries.ptr<float>(), keys.ptr<float>(), v.ptr<float>(),
        nullptr, nullptr, nullptr, nullptr, length, 1, 1, width, width, 0, length, 10000.f, 1e-6f,
        true, ctx.stream(), width, width, width, nullptr, DType::F16, cache);
    cuda_ops::gqa_prefill_online_f32(q.ptr<float>(), width, cache, out.ptr<float>(), length, 1, 1, width, 0,
                                     1.f, ctx.stream());
    require(ctx.download(out).data<float>()[(length - 1) * width] == 256.f,
            "FP16 V values overflowed the running online numerator");

    constexpr int rows = 4, head = 64;
    auto iq = ctx.upload(model_io::make_zeros_f32({rows, head}), false);
    auto ivals = make_ones_f32({rows, head});
    for (int row = 0; row < rows; ++row) std::fill_n(ivals.data<float>() + row * head, head, 1.f + row * 2.f);
    const auto iv = ctx.upload(ivals, false);
    auto image_out = arena.alloc_tensor({rows, head}, DType::F32);
    HostTensor bounds({rows, 2}, DType::I32);
    const int mask[] = {-1,-1, 1,3, 1,3, -1,-1};
    std::copy(std::begin(mask), std::end(mask), bounds.data<int>());
    const auto masked = ctx.upload(bounds, false);
    cuda_ops::gqa_prefill_causal_f32(iq.ptr<float>(), iq.ptr<float>(), iv.ptr<float>(), image_out.ptr<float>(),
        rows, 1, 1, head, 0, rows, 1.f, ctx.stream(), head, DType::F32, {}, 0, true, masked.ptr<int>());
    const auto image = ctx.download(image_out);
    const float expected[] = {1, 3, 3, 4};
    for (int row = 0; row < rows; ++row)
        require(std::abs(image.data<float>()[row * head] - expected[row]) < 1e-5f,
                "Image attention must extend visibility only within its span");
    arena.reset(mark);

    ctx.weight_arena().reset();
    HybridModelConfig config;
    config.num_layers = 1;
    FusionModel model(config);
    model.init_synthetic(ctx);
    const std::vector<int32_t> prompt{5,6,7};
    const auto reference_ids = model.generate_cuda(ctx, prompt, 8);
    auto& instantiate = CudaReviewTestAccess::instantiate(CudaReviewTestAccess::graph(model));
    const auto original = instantiate;
    instantiate = [](cudaGraphExec_t*, cudaGraph_t) { return cudaErrorMemoryAllocation; };
    GenerationStats stats;
    const auto fallback = model.generate_cuda(ctx, prompt, 8, true, true, &stats);
    instantiate = original;
    require(fallback == reference_ids && !stats.used_cuda_graph && arena.mark() == mark,
            "CUDA capture fallback must preserve IDs and restore the IR scratch watermark");
    std::atomic_bool cancel{true};
    GenerationOptions opts;
    opts.cancel = &cancel;
    const auto aborted = model.generate_cuda(ctx, prompt, 8, true, true, &stats, false, 0, -1, opts);
    require(stats.cancelled && aborted == prompt, "CUDA prefill ignored cancellation");

    auto& allocator = CudaReviewTestAccess::allocator(model);
    const auto normal_allocate = allocator;
    allocator = [](void**, size_t) { return cudaErrorMemoryAllocation; };
    const auto cache_before = model.cuda_cache_memory();
    const auto memory = model.inspect_cuda(ctx);
    allocator = normal_allocate;
    require(memory.static_scratch_bytes > 0 && model.cuda_cache_memory().kv_bytes == cache_before.kv_bytes &&
            model.generate_cuda(ctx, prompt, 8) == reference_ids, "Inspect probe reused live KV allocation state");
    std::cout << "  [PASS] Wave 1 CUDA embedding, attention, finite GEMM, cancellation and probe isolation\n";
}

void test_wave1_http(CudaContext& ctx) {
    using namespace std::chrono_literals;
    using namespace fusionrt::http;
    [[maybe_unused]] SocketRuntime sockets;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    {
        Socket probe(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
        require(bind(probe.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Bind test port");
        SocketLength len = sizeof(address);
        require(getsockname(probe.get(), reinterpret_cast<sockaddr*>(&address), &len) == 0, "Read test port");
    }
    BpeTokenizer tokenizer;
    tokenizer.init_byte_fallback();
    HybridModelConfig cfg;
    cfg.dense.max_seq_len = 2048;
    cfg.vocab_size = tokenizer.vocab_size();
    ctx.weight_arena().reset();
    FusionModel model(cfg);
    model.init_synthetic(ctx);

    std::promise<void> exit;
    auto outcome = exit.get_future();
    std::jthread server([&](std::stop_token stop) {
        try {
            serve_http(ctx, model, tokenizer, "fixture", "127.0.0.1", ntohs(address.sin_port), true,
                       nullptr, {}, {}, stop);
            exit.set_value();
        } catch (...) { exit.set_exception(std::current_exception()); }
    });
    auto open = [&] {
        const auto deadline = Clock::now() + 3s;
        do {
            Socket client(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
            if (connect(client.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
                require(set_nonblocking(client.get()), "Set client nonblocking");
                return client;
            }
            if (outcome.wait_for(0ms) == std::future_status::ready) outcome.get();
            std::this_thread::sleep_for(10ms);
        } while (Clock::now() < deadline);
        throw std::runtime_error("Loopback HTTP server failed to listen");
    };
    const auto send_request = [&](Socket& s, std::string_view route, std::string body = {}) {
        std::string req = body.empty() ? "GET " : "POST ";
        req += std::string(route) + " HTTP/1.1\r\nHost: localhost\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body;
        require(send_all(s.get(), req, Clock::now() + 2s, {}), "Send loopback HTTP request");
    };
    const auto read = [&](Socket& socket, std::chrono::milliseconds time, bool header_only = false) {
        std::string result;
        char block[4096];
        const auto deadline = Clock::now() + time;
        while (true) {
            const int count = receive(socket.get(), block, sizeof(block), deadline, {});
            require(count >= 0, "Loopback HTTP response timed out");
            if (count == 0) break;
            result.append(block, count);
            if (header_only && result.find("\r\n\r\n") != std::string::npos) break;
        }
        return result;
    };
    for (const char* stream : {"true", "false"}) {
        {
            auto active = open();
            send_request(active, "/v1/completions",
                std::string(R"({"prompt":"hello world","stream":)") + stream + R"(,"max_tokens":1800})");
            if (std::string_view(stream) == "true")
                require(read(active, 2s, true).starts_with("HTTP/1.1 200"), "Streaming request did not start");
            auto health = open();
            send_request(health, "/health");
            const auto status = read(health, 750ms);
            require(status.starts_with("HTTP/1.1 200") && status.find("\"max_active_tokens\"") != std::string::npos &&
                    status.find("\"speculative_mode\":\"none\"") != std::string::npos,
                    "Health stalled or missed the concurrent snapshot");
        }  // The client disconnects while its generation may still be running.
        auto next = open();
        send_request(next, "/v1/completions", R"({"prompt":"reply","max_tokens":2})");
        const auto reply = read(next, 3s);
        require(reply.starts_with("HTTP/1.1 200") && reply.find("\"fusionrt_diagnostics\"") != std::string::npos,
                "Inference worker did not recover after disconnect");
    }
    auto context = open();
    send_request(context, "/v1/completions", JsonValue(JsonObject{{"prompt", std::string(2048, 'a')}}).dump(0));
    const auto bad = read(context, 2s);
    require(bad.starts_with("HTTP/1.1 400") && bad.find("context_length_exceeded") != std::string::npos,
            "Overflow response lacks its context error code");
    server.request_stop();
    server.join();
    outcome.get();
    std::cout << "  [PASS] Loopback concurrent health, streaming/non-streaming disconnects, and context diagnostics\n";
}
#endif

}  // namespace fusionrt::test
