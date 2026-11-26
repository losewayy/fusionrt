#include "fusionrt/bench.h"
#include "fusionrt/gguf.h"
#include "fusionrt/models.h"
#include "fusionrt/safetensors.h"
#include "fusionrt/tokenizer.h"
#include "agent.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#include "fusionrt/cuda_kernels.h"
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#include "fusionrt/vulkan_kernels.h"
#endif
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace fusionrt;

namespace {

void test_agent_protocols() {
    using namespace agent;
    auto check = [](bool value, const char* label) {
        if (!value) throw std::runtime_error(std::string("Agent protocol: ") + label);
    };
    auto parse = [](const char* json, Protocol protocol) {
        return parse_request(JsonValue::parse(json).as_object(), protocol);
    };
    auto chat = parse(R"({
        "messages":[{"role":"developer","content":[{"type":"text","text":"Be concise"}]},
            {"role":"user","content":"Calculate"},
            {"role":"assistant","tool_calls":[{"type":"function","id":"a","function":{"name":"calc","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"a","content":"391"}],
        "tools":[{"type":"function","function":{"name":"calc"}}],
        "tool_choice":{"type":"function","function":{"name":"calc"}},
        "stream_options":{"include_usage":true}
    })", Protocol::chat);
    check(chat.choice == Choice::named && chat.tool_name == "calc" && chat.stream_usage &&
          chat.messages[0].role == Role::system && std::get<ToolResult>(chat.messages.back().content[0]).name == "calc",
          "Chat maps developer, named choice and tool result");
    const auto responses = parse(R"({"instructions":"Be concise","max_output_tokens":64,"input":[
        {"role":"user","content":[{"type":"input_text","text":"Calculate"}]},
        {"type":"function_call","name":"calc","call_id":"b","arguments":"{}"},
        {"type":"function_call_output","call_id":"b","output":"391"}]})", Protocol::responses);
    check(responses.generation.max_tokens == 64 &&
          std::get<ToolResult>(responses.messages.back().content[0]).content == "391", "Responses input replay");
    const auto anthropic = parse(R"({"system":[{"type":"text","text":"Be concise"}],"messages":[
        {"role":"user","content":"Calculate"},
        {"role":"assistant","content":[{"type":"tool_use","name":"calc","id":"c","input":{}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"c","is_error":true,"content":"Unavailable"}]}
    ]})", Protocol::messages);
    check(std::get<ToolResult>(anthropic.messages.back().content[0]).is_error, "Messages preserves tool failure");
    bool rejected = false;
    try {
        parse(R"({"input":[{"type":"function_call_output","call_id":"other","output":"data"}]})", Protocol::responses);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unknown result ID rejected");

    struct Frame {
        std::string event, data;
    };
    for (const auto protocol : {Protocol::chat, Protocol::responses, Protocol::messages}) {
        std::vector<Frame> frames;
        AgentResponse output(protocol, "sample", "model", 0, true, [&](std::string_view e, std::string_view d) {
            frames.push_back({std::string(e), std::string(d)});
            return true;
        });
        output.start(10);
        ToolParser parser(chat, {}, [&](std::string_view text) { output.text(text); },
                          [&](ToolCall call) { output.call(std::move(call)); });
        const std::string generated = "<think>secret</think>ok<tool_call>"
            R"({"name":"calc","arguments":{"text":"</tool_call>"}}</tool_call><tool_ca)";
        for (const char c : generated) parser.push(std::string_view(&c, 1));
        parser.finish();
        output.finish(Finish::stop, 5);
        const auto& blocks = output.result().output;
        check(blocks.size() == 2 && std::get<std::string>(blocks[0]) == "ok" &&
              std::get<ToolCall>(blocks[1]).arguments.at("text").as_string() == "</tool_call>",
              "fragmented markup and quoted delimiters");
        const auto& id = std::get<ToolCall>(blocks[1]).id;
        const auto body = output.body().as_object();
        if (protocol == Protocol::chat) {
            const auto& choice = body.at("choices").as_array()[0].as_object();
            const auto& message = choice.at("message").as_object();
            check(choice.at("finish_reason").as_string() == "tool_calls" && frames.back().data == "[DONE]" &&
                  message.at("tool_calls").as_array()[0].as_object().at("id").as_string() == id, "Chat completion");
        } else if (protocol == Protocol::responses) {
            const auto& call = body.at("output").as_array()[1].as_object();
            check(frames.front().event == "response.created" && frames.back().event == "response.completed" &&
                  call.at("call_id").as_string() == id && call.at("id").as_string() != id, "Responses completion");
        } else {
            const auto& call = body.at("content").as_array()[1].as_object();
            check(frames.front().event == "message_start" && frames.back().event == "message_stop" &&
                  body.at("stop_reason").as_string() == "tool_use" && call.at("id").as_string() == id,
                  "Messages completion");
        }
    }
    int count = 0;
    AgentPrompt llama;
    llama.llama_calls = true;
    ToolParser llama_parser(chat, llama, [](std::string_view) {}, [&](ToolCall) { ++count; });
    llama_parser.push(R"(<|python_tag|>{"name":"calc","parameters":{"number":391}})");
    llama_parser.finish();
    check(count == 1, "Llama native JSON tool call");
    std::cout << "[agent] Request replay, tool extraction, and SSE completion passed\n";
}

void expect_close(const std::vector<float>& got, const std::vector<double>& ref,
                  double max_abs_tol, double rel_l2_tol, const std::string& label) {
    auto audit = bench::audit_arrays_f32(got.data(), ref.data(), {static_cast<int64_t>(got.size())},
                                         "fp32", max_abs_tol, rel_l2_tol);
    std::cout << "  [PASS/CHECK] " << label
              << " max_abs=" << audit.max_abs
              << " rel_l2=" << audit.rel_l2
              << " cos=" << audit.cosine_similarity << "\n";
    if (!audit.passed) {
        throw std::runtime_error("Differential check failed for " + label);
    }
}

HostTensor rand_f32(const std::vector<int64_t>& sh, std::mt19937_64& rng, float scale = 0.25f) {
    HostTensor ht = HostTensor::create(sh, DType::F32);
    std::normal_distribution<float> d(0.0f, scale);
    for (int64_t i = 0; i < ht.desc.numel(); ++i) ht.data<float>()[i] = d(rng);
    return ht;
}

HostTensor make_ones_f32(const std::vector<int64_t>& sh) {
    HostTensor ht = HostTensor::create(sh, DType::F32);
    for (int64_t i = 0; i < ht.desc.numel(); ++i) ht.data<float>()[i] = 1.0f;
    return ht;
}

void test_quant_and_io() {
    std::cout << "[1/7] Testing Q8_0 / Q4_0 quantization, GGUF v3, Safetensors, BPE...\n";
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

    BpeTokenizer tok;
    if (!tok.load_from_gguf(gf)) throw std::runtime_error("Tokenizer load from GGUF failed");
    auto ids = tok.encode("hello", true);
    if (ids.empty() || ids[0] != 1) throw std::runtime_error("Tokenizer BOS check failed");

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
        std::filesystem::remove(gpt2_path);
    }

    if (!SafetensorsFile::write(st_path, {{"format", "pt"}}, {{"test.weight", src}}, &err)) {
        throw std::runtime_error("Safetensors write failed: " + err);
    }
    SafetensorsFile sf;
    if (!sf.load(st_path, &err)) throw std::runtime_error("Safetensors load failed: " + err);
    HostTensor sf_w = sf.read_tensor("test.weight");
    expect_close(std::vector<float>(sf_w.data<float>(), sf_w.data<float>() + 512), ref, 1e-6, 1e-6, "Safetensors roundtrip");
    std::filesystem::remove(gguf_path);
    std::filesystem::remove(st_path);
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_fused_rmsnorm_linear(CudaContext& ctx) {
    std::cout << "[2/7] Testing CUDA Fused Pre-Norm + Linear (F32, Q8_0, Q4_0)...\n";
    std::mt19937_64 rng(202);
    int M = 3, K = 512, N = 256;
    HostTensor hx = rand_f32({M, K}, rng, 0.5f);
    HostTensor hnw = rand_f32({K}, rng, 0.5f);
    HostTensor hw = rand_f32({N, K}, rng, 0.2f);

    // FP64 reference
    std::vector<double> ref(static_cast<size_t>(M * N), 0.0);
    for (int m = 0; m < M; ++m) {
        double ss = 0.0;
        for (int k = 0; k < K; ++k) {
            double v = hx.data<float>()[m * K + k];
            ss += v * v;
        }
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (hx.data<float>()[m * K + k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            }
            ref[static_cast<size_t>(m * N + n)] = acc;
        }
    }

    Tensor dx = ctx.upload(hx, false);
    Tensor dnw = ctx.upload(hnw, false);
    Tensor dw = ctx.upload(hw, false);
    Tensor dy = ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
    cuda_ops::fused_rmsnorm_linear_f32(dx.ptr<float>(), dnw.ptr<float>(), dw.ptr<float>(), dy.ptr<float>(), M, N, K, 1e-6f, ctx.stream());
    HostTensor hy = ctx.download(dy);
    expect_close(std::vector<float>(hy.data<float>(), hy.data<float>() + M * N), ref, 1e-4, 1e-4, "CUDA fused_rmsnorm_linear_f32");

    // Q8_0 and Q4_0 fused vs unfused
    HostTensor hq8 = HostTensor::create({N, K}, DType::Q8_0);
    HostTensor hq4 = HostTensor::create({N, K}, DType::Q4_0);
    for (int n = 0; n < N; ++n) {
        quantize_row_q8_0(hw.data<float>() + n * K, reinterpret_cast<BlockQ8_0*>(hq8.storage.data()) + n * (K / 32), K);
        quantize_row_q4_0(hw.data<float>() + n * K, reinterpret_cast<BlockQ4_0*>(hq4.storage.data()) + n * (K / 32), K);
    }
    Tensor dq8 = ctx.upload(hq8, false);
    Tensor dq4 = ctx.upload(hq4, false);
    Tensor dy_q8 = ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
    Tensor dy_q4 = ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
    cuda_ops::fused_rmsnorm_linear_q8_0(dx.ptr<float>(), dnw.ptr<float>(), dq8.ptr(), dy_q8.ptr<float>(), M, N, K, 1e-6f, ctx.stream());
    cuda_ops::fused_rmsnorm_linear_q4_0(dx.ptr<float>(), dnw.ptr<float>(), dq4.ptr(), dy_q4.ptr<float>(), M, N, K, 1e-6f, ctx.stream());
    HostTensor hy8 = ctx.download(dy_q8);
    HostTensor hy4 = ctx.download(dy_q4);
    expect_close(std::vector<float>(hy8.data<float>(), hy8.data<float>() + M * N), ref, 0.08, 0.02, "CUDA fused_rmsnorm_linear_q8_0");
    expect_close(std::vector<float>(hy4.data<float>(), hy4.data<float>() + M * N), ref, 0.65, 0.12, "CUDA fused_rmsnorm_linear_q4_0");

    // Verify M >= 16 WMMA Tensor Core prefill path (Q8_0, Q4_0, Q4_1, F16) against FP64 reference
    {
        int MP = 32;
        HostTensor hxp = rand_f32({MP, K}, rng, 0.4f);
        std::vector<double> ref_p(static_cast<size_t>(MP * N), 0.0);
        for (int m = 0; m < MP; ++m) {
            double ss = 0.0;
            for (int k = 0; k < K; ++k) {
                double v = hxp.data<float>()[m * K + k];
                ss += v * v;
            }
            double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += (hxp.data<float>()[m * K + k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
                }
                ref_p[static_cast<size_t>(m * N + n)] = acc;
            }
        }
        Tensor dxp = ctx.upload(hxp, false);
        Tensor dyp_q8 = ctx.scratch_arena().alloc_tensor({MP, N}, DType::F32);
        Tensor dyp_q4 = ctx.scratch_arena().alloc_tensor({MP, N}, DType::F32);
        cuda_ops::fused_rmsnorm_linear_q8_0(dxp.ptr<float>(), dnw.ptr<float>(), dq8.ptr(), dyp_q8.ptr<float>(), MP, N, K, 1e-6f, ctx.stream());
        cuda_ops::fused_rmsnorm_linear_q4_0(dxp.ptr<float>(), dnw.ptr<float>(), dq4.ptr(), dyp_q4.ptr<float>(), MP, N, K, 1e-6f, ctx.stream());
        HostTensor hyp8 = ctx.download(dyp_q8);
        HostTensor hyp4 = ctx.download(dyp_q4);
        expect_close(std::vector<float>(hyp8.data<float>(), hyp8.data<float>() + MP * N), ref_p, 0.15, 0.02, "CUDA WMMA prefill M=32 Q8_0");
        expect_close(std::vector<float>(hyp4.data<float>(), hyp4.data<float>() + MP * N), ref_p, 0.85, 0.14, "CUDA WMMA prefill M=32 Q4_0");
    }
    ctx.scratch_arena().reset();
}

void test_cuda_step1_fast_paths(CudaContext& ctx) {
    std::cout << "[2b/7] Testing CUDA Q4_A/Q8_A dp4a, cuBLAS FP16 prefill, and 3-stage GDN prefill...\n";
    std::mt19937_64 rng(2026);
    int N = 256, K = 512;
    HostTensor hx1 = rand_f32({1, K}, rng, 0.5f);
    HostTensor hnw = rand_f32({K}, rng, 0.5f);
    HostTensor hw = rand_f32({N, K}, rng, 0.25f);

    // FP64 reference M=1 normalized
    std::vector<double> ref1(static_cast<size_t>(N), 0.0);
    {
        double ss = 0.0;
        for (int k = 0; k < K; ++k) ss += hx1.data<float>()[k] * hx1.data<float>()[k];
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (hx1.data<float>()[k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            }
            ref1[static_cast<size_t>(n)] = acc;
        }
    }

    HostTensor hq8a = HostTensor::create({N, K}, DType::Q8_A);
    HostTensor hq4a = HostTensor::create({N, K}, DType::Q4_A);
    {
        std::vector<BlockQ8_0> tmp8(static_cast<size_t>(K / 32));
        std::vector<BlockQ4_0> tmp4(static_cast<size_t>(K / 32));
        for (int n = 0; n < N; ++n) {
            quantize_row_q8_0(hw.data<float>() + n * K, tmp8.data(), K);
            repack_q8_0_to_q8_a(tmp8.data(), reinterpret_cast<BlockQ8_A*>(hq8a.storage.data()) + n * (K / 256), K);
            quantize_row_q4_0(hw.data<float>() + n * K, tmp4.data(), K);
            repack_q4_0_to_q4_a(tmp4.data(), reinterpret_cast<BlockQ4_A*>(hq4a.storage.data()) + n * (K / 256), K);
        }
    }
    Tensor dx1 = ctx.upload(hx1, false);
    Tensor dnw = ctx.upload(hnw, false);
    Tensor dq8a = ctx.upload(hq8a, false);
    Tensor dq4a = ctx.upload(hq4a, false);
    Tensor dy8a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
    Tensor dy4a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
    cuda_ops::fused_rmsnorm_linear_q8_a(dx1.ptr<float>(), dnw.ptr<float>(), dq8a.ptr(), dy8a.ptr<float>(), 1, N, K, 1e-6f, ctx.stream());
    cuda_ops::fused_rmsnorm_linear_q4_a(dx1.ptr<float>(), dnw.ptr<float>(), dq4a.ptr(), dy4a.ptr<float>(), 1, N, K, 1e-6f, ctx.stream());
    HostTensor hy8a = ctx.download(dy8a);
    HostTensor hy4a = ctx.download(dy4a);
    expect_close(std::vector<float>(hy8a.data<float>(), hy8a.data<float>() + N), ref1, 0.08, 0.02, "CUDA Q8_A dp4a M=1");
    expect_close(std::vector<float>(hy4a.data<float>(), hy4a.data<float>() + N), ref1, 0.65, 0.12, "CUDA Q4_A dp4a M=1");

    // cuBLAS prefill M=32 parity on Q4_A and Q8_A
    {
        int MP = 32;
        HostTensor hxp = rand_f32({MP, K}, rng, 0.4f);
        std::vector<double> refp(static_cast<size_t>(MP * N), 0.0);
        for (int m = 0; m < MP; ++m) {
            double ss = 0.0;
            for (int k = 0; k < K; ++k) ss += hxp.data<float>()[m * K + k] * hxp.data<float>()[m * K + k];
            double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += (hxp.data<float>()[m * K + k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
                }
                refp[static_cast<size_t>(m * N + n)] = acc;
            }
        }
        Tensor dxp = ctx.upload(hxp, false);
        Tensor dx16 = ctx.scratch_arena().alloc_tensor({MP, K}, DType::F16);
        Tensor dw16 = ctx.scratch_arena().alloc_tensor({N, K}, DType::F16);
        Tensor dycb = ctx.scratch_arena().alloc_tensor({MP, N}, DType::F32);
        cuda_ops::rmsnorm_or_copy_to_f16(dxp.ptr<float>(), dnw.ptr<float>(), dx16.ptr<uint16_t>(), MP, K, 1e-6f, ctx.stream());
        cuda_ops::dequantize_to_f16(dq8a.ptr(), DType::Q8_A, dw16.ptr<uint16_t>(), N, K, ctx.stream());
        cuda_ops::cublas_linear_f16_f32(ctx.cublas(), dx16.ptr<uint16_t>(), dw16.ptr<uint16_t>(), dycb.ptr<float>(), MP, N, K);
        HostTensor hycb = ctx.download(dycb);
        expect_close(std::vector<float>(hycb.data<float>(), hycb.data<float>() + MP * N), refp, 0.15, 0.02, "CUDA cuBLAS prefill M=32 Q8_A");
    }

    // GDN prefill r1 (M=12, nk=nv=4, dk=dv=64) 3-stage vs M x single-step
    {
        int M = 12, nk = 4, nv = 4, dk = 64, dv = 64;
        int conv_dim = 2 * nk * dk + nv * dv;
        int v_dim = nv * dv;
        HostTensor hqkv = rand_f32({M, conv_dim}, rng, 0.25f);
        HostTensor hz = rand_f32({M, v_dim}, rng, 0.25f);
        HostTensor ha = rand_f32({M, nv}, rng, 0.15f);
        HostTensor hb = rand_f32({M, nv}, rng, 0.15f);
        HostTensor hcw = rand_f32({conv_dim, 4}, rng, 0.2f);
        HostTensor hcb = rand_f32({conv_dim}, rng, 0.05f);
        HostTensor hal = HostTensor::create({nv}, DType::F32);
        for (int i = 0; i < nv; ++i) hal.data<float>()[i] = -1.2f;
        HostTensor hdt = rand_f32({nv}, rng, 0.05f);
        HostTensor hnw_g = rand_f32({dv}, rng, 0.5f);

        Tensor d_qkv = ctx.upload(hqkv, false), d_z = ctx.upload(hz, false);
        Tensor d_a = ctx.upload(ha, false), d_b = ctx.upload(hb, false);
        Tensor d_cw = ctx.upload(hcw, false), d_cb = ctx.upload(hcb, false);
        Tensor d_al = ctx.upload(hal, false), d_dt = ctx.upload(hdt, false), d_nwg = ctx.upload(hnw_g, false);

        Tensor d_cs1 = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
        Tensor d_ss1 = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
        Tensor d_cs2 = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
        Tensor d_ss2 = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
        FUSIONRT_CUDA_CHECK(cudaMemset(d_cs1.ptr(), 0, d_cs1.nbytes()));
        FUSIONRT_CUDA_CHECK(cudaMemset(d_ss1.ptr(), 0, d_ss1.nbytes()));
        FUSIONRT_CUDA_CHECK(cudaMemset(d_cs2.ptr(), 0, d_cs2.nbytes()));
        FUSIONRT_CUDA_CHECK(cudaMemset(d_ss2.ptr(), 0, d_ss2.nbytes()));

        Tensor out_step = ctx.scratch_arena().alloc_tensor({M, v_dim}, DType::F32);
        Tensor out_pf = ctx.scratch_arena().alloc_tensor({M, v_dim}, DType::F32);
        Tensor gdn_ws = ctx.scratch_arena().alloc_tensor({static_cast<int64_t>(
            cuda_ops::gated_deltanet_prefill_scratch_floats(M, nk, nv, dk, dv))}, DType::F32);
        float qs = 1.0f / std::sqrt(static_cast<float>(dk));
        for (int m = 0; m < M; ++m) {
            cuda_ops::fused_gated_deltanet_step_f32(
                d_qkv.ptr<float>() + m * conv_dim, d_z.ptr<float>() + m * v_dim,
                d_a.ptr<float>() + m * nv, d_b.ptr<float>() + m * nv,
                d_cw.ptr<float>(), d_cb.ptr<float>(), d_al.ptr<float>(), d_dt.ptr<float>(),
                d_nwg.ptr<float>(), d_cs1.ptr<float>(), d_ss1.ptr<float>(),
                out_step.ptr<float>() + m * v_dim, nk, nv, dk, dv, 1e-6f, qs, ctx.stream());
        }
        cuda_ops::fused_gated_deltanet_prefill_f32(
            d_qkv.ptr<float>(), conv_dim, d_z.ptr<float>(), v_dim,
            d_a.ptr<float>(), nv, d_b.ptr<float>(), nv,
            d_cw.ptr<float>(), d_cb.ptr<float>(), d_al.ptr<float>(), d_dt.ptr<float>(),
            d_nwg.ptr<float>(), d_cs2.ptr<float>(), d_ss2.ptr<float>(),
            out_pf.ptr<float>(), M, nk, nv, dk, dv, 1e-6f, qs, ctx.stream(), gdn_ws.ptr<float>(), gdn_ws.numel());
        HostTensor h1 = ctx.download(out_step);
        HostTensor h2 = ctx.download(out_pf);
        std::vector<double> ref_v(h1.data<float>(), h1.data<float>() + M * v_dim);
        expect_close(std::vector<float>(h2.data<float>(), h2.data<float>() + M * v_dim),
                     ref_v, 1e-4, 1e-4, "CUDA GDN 3-stage prefill M=12 vs step-by-step");
    }

    // Step 2: Fused Add+RMSNorm+Linear Q4_A/Q8_A + MiniGraph optimization stats & IR parity
    {
        HostTensor hres_b = rand_f32({1, K}, rng, 0.3f);
        Tensor db = ctx.upload(hres_b, false);
        Tensor d_res_out = ctx.scratch_arena().alloc_tensor({1, K}, DType::F32);
        Tensor dy_add_q4a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
        Tensor dy_add_q8a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
        cuda_ops::fused_add_rmsnorm_linear_q4_a(dx1.ptr<float>(), db.ptr<float>(), dnw.ptr<float>(),
                                                dq4a.ptr(), d_res_out.ptr<float>(), dy_add_q4a.ptr<float>(),
                                                N, K, 1e-6f, ctx.stream());
        cuda_ops::fused_add_rmsnorm_linear_q8_a(dx1.ptr<float>(), db.ptr<float>(), dnw.ptr<float>(),
                                                dq8a.ptr(), d_res_out.ptr<float>(), dy_add_q8a.ptr<float>(),
                                                N, K, 1e-6f, ctx.stream());

        std::vector<double> ref_sum(static_cast<size_t>(K));
        double ss = 0.0;
        for (int k = 0; k < K; ++k) {
            ref_sum[k] = static_cast<double>(hx1.data<float>()[k]) + static_cast<double>(hres_b.data<float>()[k]);
            ss += ref_sum[k] * ref_sum[k];
        }
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        std::vector<double> ref_add_y(static_cast<size_t>(N), 0.0);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (ref_sum[k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            }
            ref_add_y[static_cast<size_t>(n)] = acc;
        }
        HostTensor hres_chk = ctx.download(d_res_out);
        HostTensor hy_add8 = ctx.download(dy_add_q8a);
        HostTensor hy_add4 = ctx.download(dy_add_q4a);
        expect_close(std::vector<float>(hres_chk.data<float>(), hres_chk.data<float>() + K), ref_sum, 1e-5, 1e-5, "CUDA fused_add_rmsnorm res_out");
        expect_close(std::vector<float>(hy_add8.data<float>(), hy_add8.data<float>() + N), ref_add_y, 0.08, 0.02, "CUDA fused_add_rmsnorm_linear_q8_a");
        expect_close(std::vector<float>(hy_add4.data<float>(), hy_add4.data<float>() + N), ref_add_y, 0.75, 0.14, "CUDA fused_add_rmsnorm_linear_q4_a");
    }
    ctx.scratch_arena().reset();
}

void test_cuda_gated_deltanet_fused_vs_unfused_and_fp64(CudaContext& ctx) {
    std::cout << "[3/7] Testing CUDA Fused Register-Resident GatedDeltaNet vs FP64 ref...\n";
    std::mt19937_64 rng(303);
    int nk = 4, nv = 8, dk = 64, dv = 64;
    int q_dim = nk * dk, v_dim = nv * dv, conv_dim = 2 * q_dim + v_dim;

    HostTensor h_cw = rand_f32({conv_dim, 4}, rng, 0.2f);
    HostTensor h_cb = rand_f32({conv_dim}, rng, 0.05f);
    HostTensor h_alog = HostTensor::create({nv}, DType::F32);
    for (int i = 0; i < nv; ++i) h_alog.data<float>()[i] = -1.5f;
    HostTensor h_dtb = rand_f32({nv}, rng, 0.05f);
    HostTensor h_nw = rand_f32({dv}, rng, 0.5f);

    Tensor d_cw = ctx.upload(h_cw, false);
    Tensor d_cb = ctx.upload(h_cb, false);
    Tensor d_alog = ctx.upload(h_alog, false);
    Tensor d_dtb = ctx.upload(h_dtb, false);
    Tensor d_nw = ctx.upload(h_nw, false);

    Tensor d_cs_f = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
    Tensor d_ss_f = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
    Tensor d_cs_u = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
    Tensor d_ss_u = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
    FUSIONRT_CUDA_CHECK(cudaMemset(d_cs_f.ptr(), 0, d_cs_f.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(d_ss_f.ptr(), 0, d_ss_f.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(d_cs_u.ptr(), 0, d_cs_u.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(d_ss_u.ptr(), 0, d_ss_u.nbytes()));

    // Host FP64 reference state
    std::vector<double> r_cs(static_cast<size_t>(conv_dim * 3), 0.0);
    std::vector<double> r_ss(static_cast<size_t>(nv * dk * dv), 0.0);
    std::vector<double> r_out(static_cast<size_t>(v_dim), 0.0);

    float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
    int steps = 6;
    Tensor d_out_f = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
    Tensor d_out_u = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
    Tensor sqkv = ctx.scratch_arena().alloc_tensor({conv_dim}, DType::F32);
    Tensor sy = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);

    for (int step = 0; step < steps; ++step) {
        HostTensor h_qkv = rand_f32({conv_dim}, rng, 0.3f);
        HostTensor h_z = rand_f32({v_dim}, rng, 0.3f);
        HostTensor h_a = rand_f32({nv}, rng, 0.2f);
        HostTensor h_b = rand_f32({nv}, rng, 0.2f);

        Tensor dqkv = ctx.upload(h_qkv, false);
        Tensor dz = ctx.upload(h_z, false);
        Tensor da = ctx.upload(h_a, false);
        Tensor db = ctx.upload(h_b, false);

        cuda_ops::fused_gated_deltanet_step_f32(
            dqkv.ptr<float>(), dz.ptr<float>(), da.ptr<float>(), db.ptr<float>(),
            d_cw.ptr<float>(), d_cb.ptr<float>(), d_alog.ptr<float>(), d_dtb.ptr<float>(),
            d_nw.ptr<float>(), d_cs_f.ptr<float>(), d_ss_f.ptr<float>(), d_out_f.ptr<float>(),
            nk, nv, dk, dv, 1e-6f, q_scale, ctx.stream(), nullptr, true);

        cuda_ops::unfused_gated_deltanet_step_f32(
            dqkv.ptr<float>(), dz.ptr<float>(), da.ptr<float>(), db.ptr<float>(),
            d_cw.ptr<float>(), d_cb.ptr<float>(), d_alog.ptr<float>(), d_dtb.ptr<float>(),
            d_nw.ptr<float>(), d_cs_u.ptr<float>(), d_ss_u.ptr<float>(), d_out_u.ptr<float>(),
            sqkv.ptr<float>(), sy.ptr<float>(), nk, nv, dk, dv, 1e-6f, q_scale, ctx.stream(), true);

        // FP64 step
        std::vector<double> qkv_silu(static_cast<size_t>(conv_dim), 0.0);
        for (int ch = 0; ch < conv_dim; ++ch) {
            double c0 = r_cs[ch * 3 + 0], c1 = r_cs[ch * 3 + 1], c2 = r_cs[ch * 3 + 2];
            double cur = h_qkv.data<float>()[ch];
            double acc = c0 * h_cw.data<float>()[ch * 4 + 0] + c1 * h_cw.data<float>()[ch * 4 + 1]
                       + c2 * h_cw.data<float>()[ch * 4 + 2] + cur * h_cw.data<float>()[ch * 4 + 3]
                       + h_cb.data<float>()[ch];
            r_cs[ch * 3 + 0] = c1; r_cs[ch * 3 + 1] = c2; r_cs[ch * 3 + 2] = cur;
            qkv_silu[ch] = acc / (1.0 + std::exp(-acc));
        }
        for (int kh = 0; kh < nk; ++kh) {
            double qss = 0.0, kss = 0.0;
            for (int i = 0; i < dk; ++i) {
                qss += qkv_silu[kh * dk + i] * qkv_silu[kh * dk + i];
                kss += qkv_silu[q_dim + kh * dk + i] * qkv_silu[q_dim + kh * dk + i];
            }
            double qinv = 1.0 / std::sqrt(qss + 1e-6) * q_scale;
            double kinv = 1.0 / std::sqrt(kss + 1e-6);
            for (int i = 0; i < dk; ++i) {
                qkv_silu[kh * dk + i] *= qinv;
                qkv_silu[q_dim + kh * dk + i] *= kinv;
            }
        }
        for (int vh = 0; vh < nv; ++vh) {
            int kh = vh % nk;
            double x_sp = h_a.data<float>()[vh] + h_dtb.data<float>()[vh];
            double sp = (x_sp > 20.0) ? x_sp : std::log1p(std::exp(x_sp));
            double g = std::exp(-std::exp(h_alog.data<float>()[vh]) * sp);
            double beta = 1.0 / (1.0 + std::exp(-h_b.data<float>()[vh]));
            std::vector<double> y(static_cast<size_t>(dv), 0.0);
            for (int j = 0; j < dv; ++j) {
                double kv_mem = 0.0;
                for (int i = 0; i < dk; ++i) {
                    size_t idx = (static_cast<size_t>(vh) * dk + i) * dv + j;
                    r_ss[idx] *= g;
                    kv_mem += r_ss[idx] * qkv_silu[q_dim + kh * dk + i];
                }
                double v_j = qkv_silu[2 * q_dim + vh * dv + j];
                double delta = (v_j - kv_mem) * beta;
                for (int i = 0; i < dk; ++i) {
                    size_t idx = (static_cast<size_t>(vh) * dk + i) * dv + j;
                    r_ss[idx] += qkv_silu[q_dim + kh * dk + i] * delta;
                    y[j] += r_ss[idx] * qkv_silu[kh * dk + i];
                }
            }
            double yss = 0.0;
            for (int j = 0; j < dv; ++j) yss += y[j] * y[j];
            double yinv = 1.0 / std::sqrt(yss / dv + 1e-6);
            for (int j = 0; j < dv; ++j) {
                double zv = h_z.data<float>()[vh * dv + j];
                double sz = zv / (1.0 + std::exp(-zv));
                r_out[vh * dv + j] = (y[j] * yinv * h_nw.data<float>()[j]) * sz;
            }
        }
    }

    HostTensor hf = ctx.download(d_out_f);
    HostTensor hu = ctx.download(d_out_u);
    expect_close(std::vector<float>(hf.data<float>(), hf.data<float>() + v_dim), r_out, 1e-4, 1e-4, "CUDA fused GatedDeltaNet 6-step vs FP64");
    expect_close(std::vector<float>(hu.data<float>(), hu.data<float>() + v_dim), r_out, 1e-4, 1e-4, "CUDA unfused GatedDeltaNet 6-step vs FP64");
    ctx.scratch_arena().reset();
}

void test_cuda_wave1_decode(CudaContext& ctx) {
    std::cout << "Testing Q8_1 decode kernels and CUDA Graph replay...\n";
    std::mt19937_64 rng(451);
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    constexpr int M = 2, N = 19, K = 1280;
    constexpr int blocks = K / 256;
    const float eps = 1e-6f;
    HostTensor x = rand_f32({M, K}, rng);
    HostTensor res = rand_f32({M, K}, rng);
    HostTensor norm = rand_f32({K}, rng, 1.0f);
    HostTensor gate_up = rand_f32({M, 2 * K}, rng);
    std::fill_n(x.data<float>(), 32, 0.0f);
    std::fill_n(res.data<float>(), 32, 0.0f);
    std::fill_n(gate_up.data<float>(), 32, 0.0f);
    Tensor dx = ctx.upload(x, false);
    Tensor dr = ctx.upload(res, false);
    Tensor dn = ctx.upload(norm, false);
    Tensor dgu = ctx.upload(gate_up, false);
    Tensor res_out = arena.alloc_tensor({M, K}, DType::F32);
    Tensor qx = arena.alloc_tensor({M, K}, DType::Q8_1);
    Tensor qsw = arena.alloc_tensor({M, K}, DType::Q8_1);
    cuda_ops::quantize_x_q8_1(dx.ptr<float>(), dr.ptr<float>(), dn.ptr<float>(), res_out.ptr<float>(),
                             qx.ptr(), M, K, eps, ctx.stream());
    cuda_ops::swiglu_interleaved_q8_1(dgu.ptr<float>(), qsw.ptr(), M, K, ctx.stream());

    std::vector<double> activation(M * K), swiglu(M * K), residual(M * K);
    for (int m = 0; m < M; ++m) {
        double squares = 0.0;
        for (int k = 0; k < K; ++k) {
            int j = m * K + k;
            residual[j] = x.data<float>()[j] + res.data<float>()[j];
            squares += residual[j] * residual[j];
            float gate = gate_up.data<float>()[2 * m * K + k];
            float up = gate_up.data<float>()[2 * m * K + K + k];
            swiglu[j] = gate / (1.0 + std::exp(-gate)) * up;
        }
        double inv = 1.0 / std::sqrt(squares / K + eps);
        for (int k = 0; k < K; ++k) activation[m * K + k] = residual[m * K + k] * inv * norm.data<float>()[k];
    }
    auto hr = ctx.download(res_out);
    expect_close({hr.data<float>(), hr.data<float>() + M * K}, residual, 1e-7, 1e-7, "Q8_1 residual");
    auto check_q8 = [&](const Tensor& t, const std::vector<double>& ref, const char* label) {
        auto host = ctx.download(t);
        std::vector<float> reconstructed(M * K);
        for (int b = 0; b < M * K / 32; ++b) {
            const auto& q = host.data<BlockQ8_1>()[b];
            const auto* bytes = reinterpret_cast<const int8_t*>(q.qs);
            int lo = 0, hi = 0;
            for (int i = 0; i < 32; ++i) {
                reconstructed[32 * b + i] = q.d * bytes[i];
                (i < 16 ? lo : hi) += bytes[i];
            }
            if (q.s_lo != q.d * lo || q.s_hi != q.d * hi || q.s_all != q.d * (lo + hi)) {
                throw std::runtime_error("Incorrect Q8_1 activation sums");
            }
        }
        expect_close(reconstructed, ref, 0.025, 0.012, label);
        return reconstructed;
    };
    auto quantized = check_q8(qx, activation, "residual + RMSNorm + Q8_1");
    check_q8(qsw, swiglu, "SwiGLU + Q8_1");

    {
        constexpr int wide = 4096;
        const auto wide_x = ctx.upload(rand_f32({2, wide}, rng), false);
        const auto wide_res = ctx.upload(rand_f32({2, wide}, rng), false);
        const auto wide_norm = ctx.upload(rand_f32({wide}, rng), false);
        const auto output = arena.alloc_tensor({1, wide}, DType::Q8_1);
        const auto reference = arena.alloc_tensor({2, wide}, DType::Q8_1);
        const auto out_res = arena.alloc_tensor({1, wide}, DType::F32);
        const auto ref_res = arena.alloc_tensor({2, wide}, DType::F32);
        for (const float* add : std::array<const float*, 2>{nullptr, wide_res.ptr<float>()}) {
            cuda_ops::quantize_x_q8_1(wide_x.ptr<float>(), add, wide_norm.ptr<float>(),
                                     ref_res.ptr<float>(), reference.ptr(), 2, wide, eps, ctx.stream());
            cuda_ops::quantize_x_q8_1(wide_x.ptr<float>(), add, wide_norm.ptr<float>(),
                                     out_res.ptr<float>(), output.ptr(), 1, wide, eps, ctx.stream());
            if (std::memcmp(ctx.download(output).storage.data(), ctx.download(reference).storage.data(), output.nbytes()) ||
                std::memcmp(ctx.download(out_res).storage.data(), ctx.download(ref_res).storage.data(), out_res.nbytes())) {
                throw std::runtime_error("Single-row Q8_1 RMSNorm changed quantized or residual values");
            }
        }
    }

    auto check_gemv = [&]<typename W>(DType dtype, auto gemv, auto dequantize) {
        HostTensor w = HostTensor::create({N, K}, dtype);
        for (uint8_t& byte : w.storage) byte = static_cast<uint8_t>(rng());
        for (int b = 0; b < N * blocks; ++b) {
            W& block = w.data<W>()[b];
            if constexpr (std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ8_A>) {
                for (auto& d : block.d) d = fp32_to_fp16(0.002f + 0.0001f * (b % 7));
            } else {
                block.d = fp32_to_fp16(0.00002f + 0.000001f * (b % 7));
                if constexpr (!std::is_same_v<W, BlockQ6_K>) block.dmin = fp32_to_fp16(0.0001f);
            }
        }
        Tensor dw = ctx.upload(w, false);
        Tensor dy = arena.alloc_tensor({N}, DType::F32);
        gemv(qx.ptr(), dw.ptr(), dy.ptr<float>(), N, K, ctx.stream(), {});
        auto hy = ctx.download(dy);
        std::vector<float> wf(K);
        std::vector<double> ref(N), ref_quantized(N);
        for (int n = 0; n < N; ++n) {
            dequantize(w.data<W>() + n * blocks, wf.data(), K);
            for (int k = 0; k < K; ++k) {
                ref[n] += wf[k] * activation[k];
                ref_quantized[n] += static_cast<double>(wf[k]) * quantized[k];
            }
        }
        std::vector<float> got(hy.data<float>(), hy.data<float>() + N);
        expect_close(got, ref_quantized, 2e-5, 1e-5, dtype_name(dtype));
        expect_close(got, ref, 0.05, 0.022, std::string(dtype_name(dtype)) + " vs FP32 activations");
        if constexpr (std::is_same_v<W, BlockQ5_K>) {
            Tensor prefill = arena.alloc_tensor({M, N}, DType::F32);
            cuda_ops::fused_gemv_q5_k(dx.ptr<float>(), dw.ptr(), prefill.ptr<float>(), M, N, K, ctx.stream());
            auto prefill_out = ctx.download(prefill);
            Tensor dw16 = arena.alloc_tensor({N, K}, DType::F16);
            cuda_ops::dequantize_to_f16(dw.ptr(), dtype, dw16.ptr<uint16_t>(), N, K, ctx.stream());
            auto hw16 = ctx.download(dw16);
            std::vector<double> prefill_ref(M * N, 0.0);
            for (int n = 0; n < N; ++n) {
                dequantize(w.data<W>() + n * blocks, wf.data(), K);
                for (int k = 0; k < K; ++k) {
                    uint16_t half = hw16.data<uint16_t>()[n * K + k];
                    if (half != fp32_to_fp16(wf[k]) && std::abs(fp16_to_fp32(half) - wf[k]) > 2e-5f) {
                        throw std::runtime_error("Incorrect Q5_K to F16 conversion");
                    }
                    for (int m = 0; m < M; ++m) {
                        prefill_ref[m * N + n] += static_cast<double>(wf[k]) * x.data<float>()[m * K + k];
                    }
                }
            }
            expect_close({prefill_out.data<float>(), prefill_out.data<float>() + M * N},
                         prefill_ref, 1e-5, 1e-5, "Q5_K short prefill");
        }
    };
    check_gemv.operator()<BlockQ4_A>(DType::Q4_A, cuda_ops::gemv_q4_a_q8_1, dequantize_row_q4_a);
    check_gemv.operator()<BlockQ8_A>(DType::Q8_A, cuda_ops::gemv_q8_a_q8_1, dequantize_row_q8_a);
    check_gemv.operator()<BlockQ6_K>(DType::Q6_K, cuda_ops::gemv_q6_k_q8_1, dequantize_row_q6_k);
    check_gemv.operator()<BlockQ4_K>(DType::Q4_K, cuda_ops::gemv_q4_k_q8_1, dequantize_row_q4_k);
    check_gemv.operator()<BlockQ5_K>(DType::Q5_K, cuda_ops::gemv_q5_k_q8_1, dequantize_row_q5_k);
    arena.reset(mark);

    constexpr int nk = 4, nv = 16, dk = 128, dv = 128;
    constexpr int vdim = nv * dv, conv = 2 * nk * dk + vdim;
    auto upload = [&](const std::vector<int64_t>& shape, float scale) {
        return ctx.upload(rand_f32(shape, rng, scale), false);
    };
    Tensor cw = upload({conv, 4}, 0.2f), cb = upload({conv}, 0.05f);
    Tensor alog = upload({nv}, 0.4f), dt = upload({nv}, 0.1f), nw = upload({dv}, 0.5f);
    auto init_conv = rand_f32({conv, 3}, rng, 0.2f);
    auto init_ssm = rand_f32({nv, dk, dv}, rng, 0.1f);
    Tensor conv_state[] = {ctx.upload(init_conv, false), ctx.upload(init_conv, false)};
    Tensor ssm[] = {ctx.upload(init_ssm, false), ctx.upload(init_ssm, false)};
    Tensor out[] = {arena.alloc_tensor({vdim}, DType::F32), arena.alloc_tensor({vdim}, DType::F32)};
    Tensor ws = arena.alloc_tensor({conv + vdim + 2 * nv}, DType::F32);
    for (int step = 0; step < 3; ++step) {
        Tensor qkv = upload({conv}, 0.3f), z = upload({vdim}, 0.3f);
        Tensor a = upload({nv}, 0.2f), b = upload({nv}, 0.2f);
        for (int i = 0; i < 2; ++i) {
            cuda_ops::fused_gated_deltanet_step_f32(
                qkv.ptr<float>(), z.ptr<float>(), a.ptr<float>(), b.ptr<float>(), cw.ptr<float>(), cb.ptr<float>(),
                alog.ptr<float>(), dt.ptr<float>(), nw.ptr<float>(), conv_state[i].ptr<float>(), ssm[i].ptr<float>(),
                out[i].ptr<float>(), nk, nv, dk, dv, eps, 1.0f / std::sqrt(float(dk)), ctx.stream(),
                i == 0 ? nullptr : ws.ptr<float>());
        }
    }
    auto gdn_ref = ctx.download(out[0]), gdn_split = ctx.download(out[1]);
    for (int i = 0; i < vdim; ++i) {
        if (!(std::abs(gdn_split.data<float>()[i] - gdn_ref.data<float>()[i]) < 1e-4f)) {
            throw std::runtime_error("64-CTA GDN decode exceeded max_abs");
        }
    }
    expect_close({gdn_split.data<float>(), gdn_split.data<float>() + vdim},
                 {gdn_ref.data<float>(), gdn_ref.data<float>() + vdim}, 9.99e-5, 1e-4, "64-CTA GDN decode");
    arena.reset(mark);

    HybridModelConfig cfg;
    cfg.arch = "hybrid";
    cfg.vocab_size = 256;
    cfg.num_layers = 4;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 256;
    cfg.dense.num_attention_heads = 4;
    cfg.dense.num_key_value_heads = 2;
    cfg.dense.head_dim = 64;
    cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 512;
    cfg.dense.max_seq_len = 256;
    cfg.dense.weight_dtype = cfg.gdn.weight_dtype = DType::Q4_A;
    cfg.gdn.num_k_heads = 4;
    cfg.gdn.num_v_heads = 16;
    cfg.gdn.head_k_dim = cfg.gdn.head_v_dim = 128;
    FusionModel model(cfg);
    size_t weights_mark = ctx.weight_arena().mark();
    model.init_synthetic(ctx, 452);
    int swiglu_fusions = 0;
    for (const auto& node : model.fused_ir_graph().nodes) {
        if (node.op == OpKind::SWIGLU_Q8_LINEAR) ++swiglu_fusions;
    }
    if (swiglu_fusions != cfg.num_layers) throw std::runtime_error("Missing Q8_1 SwiGLU fusion");
    const std::vector<int32_t> prompt{1, 10, 20};
    auto eager = model.generate_cuda(ctx, prompt, 16, true, false);
    auto graph = model.generate_cuda(ctx, prompt, 16, true, true);
    auto replay = model.generate_cuda(ctx, prompt, 16, true, true);
    if (graph != eager || replay != eager || eager.size() != prompt.size() + 16) {
        throw std::runtime_error("Q8_1 CUDA Graph token-ID replay mismatch");
    }
    std::cout << "  [PASS/CHECK] Q8_1 hybrid eager and CUDA Graph token IDs match exactly\n";
    ctx.weight_arena().reset(weights_mark);
}

void test_repacked_k_quants(CudaContext& ctx) {
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
        for (int n = 0; n < N; ++n) {
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
        cuda_ops::embed_lookup_f32(w.ptr(), packed_type, d_ids.ptr<int32_t>(), embeddings.ptr<float>(), 2, K, ctx.stream());
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
        mmq(qx.ptr(), w.ptr(), y.ptr<float>(), M, N, K, ctx.stream());
        got = ctx.download(y);
        expect_close({got.data<float>(), got.data<float>() + M * N}, expected, 3e-5, 1e-5, label + " MMQ");
    };
    check(DType::Q4_K, DType::Q4_KA, repack_q4_k_to_q4_ka, dequantize_row_q4_k, dequantize_row_q4_ka,
          cuda_ops::gemv_q4_ka_q8_1, cuda_ops::mmq_q4_ka_q8_1);
    check(DType::Q6_K, DType::Q6_A, repack_q6_k_to_q6_a, dequantize_row_q6_k, dequantize_row_q6_a,
          cuda_ops::gemv_q6_a_q8_1, cuda_ops::mmq_q6_a_q8_1);
    arena.reset(mark);
}

void test_cuda_wave2_prefill(CudaContext& ctx) {
    std::cout << "Testing INT8 MMQ, chunkwise GatedDeltaNet, and tiled GQA prefill...\n";
    std::mt19937_64 rng(452);
    auto& arena = ctx.scratch_arena();
    auto upload_random = [&](std::initializer_list<int64_t> shape, float scale = 0.25f) {
        return ctx.upload(rand_f32(shape, rng, scale), false);
    };
    auto read = [&](const Tensor& t) {
        auto h = ctx.download(t);
        return std::vector<float>(h.data<float>(), h.data<float>() + h.desc.numel());
    };
    size_t mark = arena.mark();
    constexpr int M = 128, N = 73, K = 768;
    HostTensor x = rand_f32({M, K}, rng);
    auto dx = ctx.upload(x, false);
    auto qx = arena.alloc_tensor({M, K}, DType::Q8_1);
    cuda_ops::quantize_x_q8_1(dx.ptr<float>(), nullptr, nullptr, nullptr, qx.ptr(), M, K, 0.0f, ctx.stream());
    auto hq = ctx.download(qx);
    std::vector<float> quantized(M * K);
    for (int block = 0; block < M * K / 32; ++block) {
        const auto& q = hq.data<BlockQ8_1>()[block];
        const auto* qs = reinterpret_cast<const int8_t*>(q.qs);
        for (int k = 0; k < 32; ++k) quantized[block * 32 + k] = q.d * qs[k];
    }

    auto check_mmq = [&]<typename W>(DType dtype, auto mmq, auto dequantize) {
        constexpr int blocks = K / 256;
        HostTensor w = HostTensor::create({N, K}, dtype);
        for (auto& byte : w.storage) byte = static_cast<uint8_t>(rng());
        for (int i = 0; i < N * blocks; ++i) {
            auto& block = w.data<W>()[i];
            if constexpr (std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ8_A>) {
                for (int sub = 0; sub < 8; ++sub) block.d[sub] = fp32_to_fp16(0.002f + 0.0001f * sub);
            } else {
                block.d = fp32_to_fp16(0.00002f + 0.000001f * (i % 7));
                if constexpr (!std::is_same_v<W, BlockQ6_K>) block.dmin = fp32_to_fp16(0.0001f);
            }
        }
        auto dw = ctx.upload(w, false);
        std::vector<float> weights(N * K);
        for (int n = 0; n < N; ++n) dequantize(w.data<W>() + n * blocks, weights.data() + n * K, K);
        for (int rows : {4, 17, 64, M}) {
            auto dy = arena.alloc_tensor({rows, N}, DType::F32);
            mmq(qx.ptr(), dw.ptr(), dy.ptr<float>(), rows, N, K, ctx.stream());
            std::vector<double> ref(rows * N), ref_quantized(rows * N);
            for (int m = 0; m < rows; ++m) {
                for (int n = 0; n < N; ++n) {
                    for (int k = 0; k < K; ++k) {
                        double weight = weights[n * K + k];
                        ref[m * N + n] += weight * x.data<float>()[m * K + k];
                        ref_quantized[m * N + n] += weight * quantized[m * K + k];
                    }
                }
            }
            std::string label = std::string("MMQ ") + dtype_name(dtype) + " M=" + std::to_string(rows);
            auto got = read(dy);
            expect_close(got, ref_quantized, 2e-5, 1e-5, label + " with dequantized inputs");
            expect_close(got, ref, 0.0, 0.015, label + " with FP32 activations");
        }
    };
    check_mmq.operator()<BlockQ4_A>(DType::Q4_A, cuda_ops::mmq_q4_a_q8_1, dequantize_row_q4_a);
    check_mmq.operator()<BlockQ8_A>(DType::Q8_A, cuda_ops::mmq_q8_a_q8_1, dequantize_row_q8_a);
    check_mmq.operator()<BlockQ6_K>(DType::Q6_K, cuda_ops::mmq_q6_k_q8_1, dequantize_row_q6_k);
    check_mmq.operator()<BlockQ4_K>(DType::Q4_K, cuda_ops::mmq_q4_k_q8_1, dequantize_row_q4_k);
    check_mmq.operator()<BlockQ5_K>(DType::Q5_K, cuda_ops::mmq_q5_k_q8_1, dequantize_row_q5_k);
    arena.reset(mark);

    constexpr int nk = 2, nv = 4, dk = 128, dv = 128;
    constexpr int vdim = nv * dv, conv = 2 * nk * dk + vdim;
    const float qs = 1.0f / std::sqrt(static_cast<float>(dk));
    auto qkv = upload_random({M, conv}), z = upload_random({M, vdim});
    auto a = upload_random({M, nv}), b = upload_random({M, nv});
    auto cw = upload_random({conv, 4}), cb = upload_random({conv}, 0.05f);
    auto al = upload_random({nv}), dt = upload_random({nv}), norm = upload_random({dv});
    auto init_conv = rand_f32({conv, 3}, rng), init_state = rand_f32({nv, dk, dv}, rng, 0.01f);
    auto cs_ref = ctx.upload(init_conv, false), ss_ref = ctx.upload(init_state, false);
    auto cs = ctx.upload(init_conv, false), ss = ctx.upload(init_state, false);
    auto ref = arena.alloc_tensor({M, vdim}, DType::F32), out = arena.alloc_tensor({M, vdim}, DType::F32);
    auto scratch = arena.alloc_tensor({static_cast<int64_t>(
        cuda_ops::gated_deltanet_prefill_scratch_floats(M, nk, nv, dk, dv))}, DType::F32);
    for (int m = 0; m < M; ++m) {
        cuda_ops::fused_gated_deltanet_step_f32(
            qkv.ptr<float>() + m * conv, z.ptr<float>() + m * vdim, a.ptr<float>() + m * nv, b.ptr<float>() + m * nv,
            cw.ptr<float>(), cb.ptr<float>(), al.ptr<float>(), dt.ptr<float>(), norm.ptr<float>(),
            cs_ref.ptr<float>(), ss_ref.ptr<float>(), ref.ptr<float>() + m * vdim,
            nk, nv, dk, dv, 1e-6f, qs, ctx.stream(), nullptr, true);
    }
    cuda_ops::fused_gated_deltanet_prefill_f32(
        qkv.ptr<float>(), conv, z.ptr<float>(), vdim, a.ptr<float>(), nv, b.ptr<float>(), nv,
        cw.ptr<float>(), cb.ptr<float>(), al.ptr<float>(), dt.ptr<float>(), norm.ptr<float>(),
        cs.ptr<float>(), ss.ptr<float>(), out.ptr<float>(), M, nk, nv, dk, dv, 1e-6f, qs, ctx.stream(),
        scratch.ptr<float>(), scratch.numel(), true);
    auto expected = read(ref);
    expect_close(read(out), {expected.begin(), expected.end()}, 9.99e-5, 9.99e-5, "GDN prefill output M=128");
    expected = read(ss_ref);
    expect_close(read(ss), {expected.begin(), expected.end()}, 9.99e-5, 9.99e-5, "GDN recurrent state M=128");
    expected = read(cs_ref);
    expect_close(read(cs), {expected.begin(), expected.end()}, 0.0, 0.0, "GDN convolution state M=128");
    arena.reset(mark);

    constexpr int nq = 6, nkv = 2, start = 5;
    for (int hd : {128, 256}) {
        int qcols = nq * hd, stride = qcols + 64;
        float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        auto q = upload_random({M, stride}, 0.8f);
        auto k = upload_random({start + M, nkv, hd}, 0.8f);
        auto v = upload_random({start + M, nkv, hd});
        for (int rows : {17, M}) {
            auto seq = arena.alloc_tensor({rows, qcols}, DType::F32);
            auto tiled = arena.alloc_tensor({rows, qcols}, DType::F32);
            for (int m = 0; m < rows; ++m) {
                cuda_ops::gqa_splitk_decode_f32(q.ptr<float>() + m * stride, k.ptr<float>(), v.ptr<float>(),
                    seq.ptr<float>() + m * qcols, nullptr, nullptr, nq, nkv, hd, nullptr, start + m + 1,
                    start + M, scale, 1, ctx.stream());
            }
            cuda_ops::gqa_prefill_causal_f32(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), tiled.ptr<float>(),
                rows, nq, nkv, hd, start, start + M, scale, ctx.stream(), stride);
            expected = read(seq);
            expect_close(read(tiled), {expected.begin(), expected.end()}, 9.99e-6, 0.0,
                "GQA prefill head_dim=" + std::to_string(hd) + " M=" + std::to_string(rows));
            if (rows == M && hd == 128) {
                Tensor kh = arena.alloc_tensor({start + M, nkv, hd}, DType::F16);
                Tensor vh = arena.alloc_tensor({start + M, nkv, hd}, DType::F16);
                cuda_ops::dequantize_to_f16(k.ptr(), DType::F32, kh.ptr<uint16_t>(), start + M, nkv * hd, ctx.stream());
                cuda_ops::dequantize_to_f16(v.ptr(), DType::F32, vh.ptr<uint16_t>(), start + M, nkv * hd, ctx.stream());
                cuda_ops::gqa_prefill_causal_f32(q.ptr<float>(), kh.ptr(), vh.ptr(), tiled.ptr<float>(),
                    M, nq, nkv, hd, start, start + M, scale, ctx.stream(), stride, DType::F16);
                expect_close(read(tiled), {expected.begin(), expected.end()}, 4e-4, 1e-3, "GQA FP16 KV prefill");
                void* workspace = arena.alloc(cuda_ops::gqa_f16_prefill_workspace(M, nq, hd, start + M));
                cuda_ops::gqa_f16_prefill(ctx.cublas(), q.ptr<float>(), stride, kh.ptr(), vh.ptr(), tiled.ptr<float>(),
                    M, nq, nkv, hd, start, scale, workspace, ctx.stream());
                expect_close(read(tiled), {expected.begin(), expected.end()}, 5e-4, 1e-3, "GQA FP16 Tensor Core prefill");
                constexpr int splits = 4;
                auto partial = arena.alloc_tensor({nq, splits, hd}, DType::F32);
                auto stats = arena.alloc_tensor({nq, splits, 2}, DType::F32);
                auto step = arena.alloc_tensor({qcols}, DType::F32);
                cuda_ops::gqa_splitk_decode_f32(q.ptr<float>() + (M - 1) * stride, kh.ptr(), vh.ptr(),
                    step.ptr<float>(), partial.ptr<float>(), stats.ptr<float>(), nq, nkv, hd,
                    nullptr, start + M, start + M, scale, splits, ctx.stream(), true, DType::F16);
                expect_close(read(step), {expected.end() - qcols, expected.end()}, 4e-4, 1e-3,
                    "Grouped decode after FP16 KV prefill");
                const auto reference = arena.alloc_tensor({1, qcols}, DType::Q8_1);
                const auto packed = arena.alloc_tensor({1, qcols}, DType::Q8_1);
                cuda_ops::apply_sigmoid_gate_f32(step.ptr<float>(), q.ptr<float>(), qcols, ctx.stream());
                cuda_ops::quantize_x_q8_1(step.ptr<float>(), nullptr, nullptr, nullptr, reference.ptr(),
                                          1, qcols, 0, ctx.stream());
                cuda_ops::gqa_splitk_decode_f32(q.ptr<float>() + (M - 1) * stride, kh.ptr(), vh.ptr(),
                    nullptr, partial.ptr<float>(), stats.ptr<float>(), nq, nkv, hd,
                    nullptr, start + M, start + M, scale, splits, ctx.stream(), true, DType::F16,
                    packed.ptr<BlockQ8_1>(), q.ptr<float>());
                if (ctx.download(reference).storage != ctx.download(packed).storage) {
                    throw std::runtime_error("Fused attention and gate must produce the same Q8_1 blocks");
                }
            }
        }
        arena.reset(mark);
    }
}

void test_cuda_qk_rope_and_gqa_64k(CudaContext& ctx) {
    std::cout << "[4/7] Testing CUDA Fused Q/K Norm + Partial RoPE + GQA 64K Split-K...\n";
    std::mt19937_64 rng(404);
    int nq = 8, nkv = 2, hd = 64, rot_dim = 32, pos = 65535, max_seq = 65536;
    HostTensor hq = rand_f32({nq, hd}, rng, 0.3f);
    HostTensor hk = rand_f32({nkv, hd}, rng, 0.3f);
    HostTensor hv = rand_f32({nkv, hd}, rng, 0.3f);
    HostTensor hqn = rand_f32({hd}, rng, 0.5f);
    HostTensor hkn = rand_f32({hd}, rng, 0.5f);

    Tensor dq = ctx.upload(hq, false);
    Tensor dk = ctx.upload(hk, false);
    Tensor dv = ctx.upload(hv, false);
    Tensor dqn = ctx.upload(hqn, false);
    Tensor dkn = ctx.upload(hkn, false);
    Tensor dkc = ctx.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
    Tensor dvc = ctx.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
    FUSIONRT_CUDA_CHECK(cudaMemset(dkc.ptr(), 0, dkc.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(dvc.ptr(), 0, dvc.nbytes()));

    // Fused Q/K Norm + Partial RoPE (rot_dim=32) + Append at pos=65535
    cuda_ops::fused_qk_norm_rope_kv_append_f32(
        dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(),
        dqn.ptr<float>(), dkn.ptr<float>(), dkc.ptr<float>(), dvc.ptr<float>(),
        nq, nkv, hd, rot_dim, nullptr, pos, max_seq, 1000000.0f, 1e-6f, true, ctx.stream());

    Tensor out_s1 = ctx.scratch_arena().alloc_tensor({nq, hd}, DType::F32);
    Tensor out_s16 = ctx.scratch_arena().alloc_tensor({nq, hd}, DType::F32);
    Tensor part_out = ctx.scratch_arena().alloc_tensor({nq, 16, hd}, DType::F32);
    Tensor part_ml = ctx.scratch_arena().alloc_tensor({nq, 16, 2}, DType::F32);
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>(), dkc.ptr<float>(), dvc.ptr<float>(),
                                    out_s1.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                                    nq, nkv, hd, nullptr, 65536, max_seq, scale, 1, ctx.stream());
    cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>(), dkc.ptr<float>(), dvc.ptr<float>(),
                                    out_s16.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                                    nq, nkv, hd, nullptr, 65536, max_seq, scale, 16, ctx.stream());
    HostTensor h1 = ctx.download(out_s1);
    HostTensor h16 = ctx.download(out_s16);
    std::vector<double> ref(h1.data<float>(), h1.data<float>() + nq * hd);
    expect_close(std::vector<float>(h16.data<float>(), h16.data<float>() + nq * hd), ref, 1e-5, 1e-5, "CUDA GQA 64K Split-K=16 vs Split-K=1");
    ctx.scratch_arena().reset();
}

void test_cuda_graph_and_moe_offload(CudaContext& ctx) {
    std::cout << "[5/7] Testing CUDA Graph full-decode parity & Sparse MoE CPU-offload parity...\n";
    // 5a. Hybrid 3:1 CUDA Graph vs Eager parity
    {
        HybridModelConfig cfg;
        cfg.arch = "hybrid";
        cfg.vocab_size = 256;
        cfg.num_layers = 4;
        cfg.dense.hidden_size = 256;
        cfg.dense.num_attention_heads = 4;
        cfg.dense.num_key_value_heads = 2;
        cfg.dense.head_dim = 64;
        cfg.dense.intermediate_size = 512;
        cfg.dense.max_seq_len = 256;
        cfg.gdn.hidden_size = 256;
        cfg.gdn.num_k_heads = 4;
        cfg.gdn.num_v_heads = 4;
        cfg.gdn.head_k_dim = 64;
        cfg.gdn.head_v_dim = 64;
        cfg.gdn.intermediate_size = 512;

        FusionModel m(cfg);
        m.init_synthetic(ctx, 505);
        const auto& st = m.fused_ir_graph().stats;
        if (st.sibling_merges < 4 || st.norm_linear_fusions < 8 || st.add_norm_linear_fusions < 4 || st.opt_nodes >= st.raw_nodes) {
            throw std::runtime_error("MiniGraph optimize_graph did not fuse expected ops: " + m.fused_ir_graph().summary());
        }
        std::cout << "  [PASS/CHECK] " << m.fused_ir_graph().summary() << "\n";
        m.reset_kv_and_state(ctx);
        auto logits_fused = m.forward_token_cuda(ctx, 1, 0, true, false);
        m.reset_kv_and_state(ctx);
        auto logits_raw = m.forward_token_cuda(ctx, 1, 0, false, false);
        std::vector<double> ref_l(logits_raw.begin(), logits_raw.end());
        expect_close(logits_fused, ref_l, 1e-4, 1e-4, "MiniGraph fused vs raw IR logits parity");

        m.reset_kv_and_state(ctx);
        auto out_eager = m.generate_cuda(ctx, {1, 10, 20}, 4, true, false);
        auto out_graph = m.generate_cuda(ctx, {1, 10, 20}, 4, true, true);
        if (out_eager != out_graph) {
            throw std::runtime_error("CUDA Graph token generation diverged from eager!");
        }
        std::cout << "  [PASS/CHECK] CUDA Graph 4-layer Hybrid 3:1 token parity exact match\n";
    }

    // 5a-2. Sparse MoE CUDA Graph vs Eager parity
    ctx.weight_arena().reset();
    ctx.scratch_arena().reset();
    {
        HybridModelConfig cfg;
        cfg.arch = "moe";
        cfg.vocab_size = 256;
        cfg.num_layers = 2;
        cfg.moe.hidden_size = 128;
        cfg.moe.intermediate_size = 128;
        cfg.moe.num_experts = 4;
        cfg.moe.top_k = 2;
        cfg.moe.gpu_resident_experts = 4;
        cfg.moe.shared_intermediate_size = 128;

        FusionModel m(cfg);
        m.init_synthetic(ctx, 506);
        auto out_eager = m.generate_cuda(ctx, {1, 5, 9}, 4, true, false);
        auto out_graph = m.generate_cuda(ctx, {1, 5, 9}, 4, true, true);
        if (out_eager != out_graph) {
            throw std::runtime_error("CUDA Graph MoE token generation diverged from eager!");
        }
        std::cout << "  [PASS/CHECK] CUDA Graph 2-layer Sparse MoE token parity exact match\n";
    }

    // 5b. Sparse MoE all-GPU vs partial CPU-offload parity
    ctx.weight_arena().reset();
    ctx.scratch_arena().reset();
    {
        MoEConfig mc;
        mc.hidden_size = 128;
        mc.intermediate_size = 128;
        mc.num_experts = 6;
        mc.top_k = 2;
        mc.has_shared_expert = true;
        mc.shared_intermediate_size = 128;

        std::mt19937_64 rng(606);
        HostTensor h_x = rand_f32({mc.hidden_size}, rng, 0.3f);
        HostTensor h_nw = rand_f32({mc.hidden_size}, rng, 0.5f);
        HostTensor h_rw = rand_f32({mc.num_experts, mc.hidden_size}, rng, 0.2f);
        HostTensor h_gu = rand_f32({mc.num_experts, 2 * mc.intermediate_size, mc.hidden_size}, rng, 0.1f);
        HostTensor h_dn = rand_f32({mc.num_experts, mc.hidden_size, mc.intermediate_size}, rng, 0.1f);
        HostTensor h_sgu = rand_f32({2 * mc.shared_intermediate_size, mc.hidden_size}, rng, 0.1f);
        HostTensor h_sdn = rand_f32({mc.hidden_size, mc.shared_intermediate_size}, rng, 0.1f);

        CudaSparseMoELayer l_gpu;
        l_gpu.cfg = mc;
        l_gpu.cfg.gpu_resident_experts = 6;
        l_gpu.norm_w = ctx.upload(h_nw, true);
        l_gpu.router_w = ctx.upload(h_rw, true);
        l_gpu.gpu_gate_up_w = ctx.upload(h_gu, true);
        l_gpu.gpu_down_w = ctx.upload(h_dn, true);
        l_gpu.shared_gate_up_w = ctx.upload(h_sgu, true);
        l_gpu.shared_down_w = ctx.upload(h_sdn, true);

        CudaSparseMoELayer l_off;
        l_off.cfg = mc;
        l_off.cfg.gpu_resident_experts = 3;  // experts 3..5 offloaded to CPU!
        l_off.norm_w = l_gpu.norm_w;
        l_off.router_w = l_gpu.router_w;
        l_off.gpu_gate_up_w = ctx.upload(HostTensor::create({3, 2 * mc.intermediate_size, mc.hidden_size}, DType::F32), true);
        l_off.gpu_down_w = ctx.upload(HostTensor::create({3, mc.hidden_size, mc.intermediate_size}, DType::F32), true);
        // Copy first 3 experts to GPU, last 3 to CPU
        FUSIONRT_CUDA_CHECK(cudaMemcpy(l_off.gpu_gate_up_w.ptr(), l_gpu.gpu_gate_up_w.ptr(), l_off.gpu_gate_up_w.nbytes(), cudaMemcpyDeviceToDevice));
        FUSIONRT_CUDA_CHECK(cudaMemcpy(l_off.gpu_down_w.ptr(), l_gpu.gpu_down_w.ptr(), l_off.gpu_down_w.nbytes(), cudaMemcpyDeviceToDevice));
        l_off.cpu_gate_up_w = HostTensor::create({3, 2 * mc.intermediate_size, mc.hidden_size}, DType::F32);
        l_off.cpu_down_w = HostTensor::create({3, mc.hidden_size, mc.intermediate_size}, DType::F32);
        std::memcpy(l_off.cpu_gate_up_w.storage.data(), h_gu.storage.data() + l_off.gpu_gate_up_w.nbytes(), l_off.cpu_gate_up_w.storage.size());
        std::memcpy(l_off.cpu_down_w.storage.data(), h_dn.storage.data() + l_off.gpu_down_w.nbytes(), l_off.cpu_down_w.storage.size());
        l_off.shared_gate_up_w = l_gpu.shared_gate_up_w;
        l_off.shared_down_w = l_gpu.shared_down_w;

        Tensor x1 = ctx.upload(h_x, false);
        Tensor x2 = ctx.upload(h_x, false);
        l_gpu.forward_cuda(ctx, x1);
        l_off.forward_cuda(ctx, x2);
        HostTensor h1 = ctx.download(x1);
        HostTensor h2 = ctx.download(x2);
        std::vector<double> ref(h1.data<float>(), h1.data<float>() + mc.hidden_size);
        expect_close(std::vector<float>(h2.data<float>(), h2.data<float>() + mc.hidden_size), ref, 1e-4, 1e-4, "MoE GPU-resident vs CPU-offload parity");
    }
}
#endif  // FUSIONRT_ENABLE_CUDA

#ifdef FUSIONRT_ENABLE_VULKAN
void test_vulkan_all_devices() {
    auto vk_devs = VkContext::enumerate_devices();
    std::cout << "[6/7] Testing Vulkan compute & CoopMat & Batched Layer across " << vk_devs.size() << " device(s)...\n";
    std::mt19937_64 rng(707);

    for (const auto& info : vk_devs) {
        std::cout << "  Device [" << info.device_index << "] " << info.name << " (" << info.device_type << ")\n";
        VkContext vk(info.device_index, 64, 64);

        // Test 1: Fused Pre-Norm + GEMV vs FP64
        int K = 256, N = 128;
        HostTensor hx = rand_f32({1, K}, rng, 0.4f);
        HostTensor hnw = rand_f32({K}, rng, 0.5f);
        HostTensor hw = rand_f32({N, K}, rng, 0.2f);
        std::vector<double> ref(static_cast<size_t>(N), 0.0);
        double ss = 0.0;
        for (int k = 0; k < K; ++k) ss += static_cast<double>(hx.data<float>()[k]) * hx.data<float>()[k];
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (hx.data<float>()[k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            ref[static_cast<size_t>(n)] = acc;
        }
        Tensor dx = vk.upload(hx, false);
        Tensor dnw = vk.upload(hnw, false);
        Tensor dw = vk.upload(hw, false);
        Tensor dy = vk.scratch_arena().alloc_tensor({1, N}, DType::F32);
        vk_ops::fused_rmsnorm_gemv(vk, dx, dnw, dw, dy, 1, N, K, 1e-6f);
        HostTensor hy = vk.download(dy);
        expect_close(std::vector<float>(hy.data<float>(), hy.data<float>() + N), ref, 1e-4, 1e-4, "Vulkan fused_rmsnorm_gemv");

        // Test 2: Cooperative Matrix FP16 GEMM (64x64x64) if supported, else tiled FP32 fallback
        if (vk.info().coop_mat.supported && vk.info().coop_mat.fp16_in_fp32_acc) {
            int M = 64, GN = 64, GK = 64;
            HostTensor ha32 = rand_f32({M, GK}, rng, 0.2f);
            HostTensor hb32 = rand_f32({GK, GN}, rng, 0.2f);
            HostTensor ha16 = HostTensor::create({M, GK}, DType::F16);
            HostTensor hb16 = HostTensor::create({GK, GN}, DType::F16);
            convert_f32_to_f16(ha32.data<float>(), reinterpret_cast<uint16_t*>(ha16.storage.data()), M * GK);
            convert_f32_to_f16(hb32.data<float>(), reinterpret_cast<uint16_t*>(hb16.storage.data()), GK * GN);

            std::vector<double> cref(static_cast<size_t>(M * GN), 0.0);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < GN; ++n) {
                    double acc = 0.0;
                    for (int k = 0; k < GK; ++k) {
                        acc += static_cast<double>(fp16_to_fp32(reinterpret_cast<uint16_t*>(ha16.storage.data())[m * GK + k]))
                             * static_cast<double>(fp16_to_fp32(reinterpret_cast<uint16_t*>(hb16.storage.data())[k * GN + n]));
                    }
                    cref[static_cast<size_t>(m * GN + n)] = acc;
                }
            }
            Tensor va = vk.upload(ha16, false);
            Tensor vb = vk.upload(hb16, false);
            Tensor vc = vk.scratch_arena().alloc_tensor({M, GN}, DType::F32);
            vk_ops::gemm_auto(vk, va, vb, vc, M, GN, GK, true);
            HostTensor hvc = vk.download(vc);
            expect_close(std::vector<float>(hvc.data<float>(), hvc.data<float>() + M * GN), cref, 1e-2, 1e-2, "Vulkan CoopMat 16x16x16");
        } else {
            int M = 64, GN = 64, GK = 64;
            HostTensor ha32 = rand_f32({M, GK}, rng, 0.2f);
            HostTensor hb32 = rand_f32({GN, GK}, rng, 0.2f);
            std::vector<double> cref(static_cast<size_t>(M * GN), 0.0);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < GN; ++n) {
                    double acc = 0.0;
                    for (int k = 0; k < GK; ++k) {
                        acc += static_cast<double>(ha32.data<float>()[m * GK + k])
                             * static_cast<double>(hb32.data<float>()[n * GK + k]);
                    }
                    cref[static_cast<size_t>(m * GN + n)] = acc;
                }
            }
            Tensor va = vk.upload(ha32, false);
            Tensor vb = vk.upload(hb32, false);
            Tensor vc = vk.scratch_arena().alloc_tensor({M, GN}, DType::F32);
            vk_ops::gemm_auto(vk, va, vb, vc, M, GN, GK, false);
            HostTensor hvc = vk.download(vc);
            expect_close(std::vector<float>(hvc.data<float>(), hvc.data<float>() + M * GN), cref, 1e-4, 1e-4, "Vulkan Tiled GEMM fallback");
        }
        vk.scratch_arena().reset();
    }
}

void test_vulkan_batched_vs_unbatched() {
    std::cout << "[7/7] Testing Vulkan Single-Command-Buffer Batched vs Unbatched layer parity...\n";
    VkContext vk(0, 64, 64);
    std::mt19937_64 rng(808);
    int H = 128, nq = 4, nkv = 2, hd = 32, inter = 256, max_seq = 64;
    int q_dim = nq * hd, kv_dim = nkv * hd, qkv_dim = q_dim + 2 * kv_dim;

    Tensor x_in = vk.upload(rand_f32({H}, rng, 0.2f), true);
    Tensor anw = vk.upload(rand_f32({H}, rng, 0.5f), true);
    Tensor qkvw = vk.upload(rand_f32({qkv_dim, H}, rng, 0.1f), true);
    Tensor qnw = vk.upload(rand_f32({hd}, rng, 0.5f), true);
    Tensor knw = vk.upload(rand_f32({hd}, rng, 0.5f), true);
    Tensor ow = vk.upload(rand_f32({H, q_dim}, rng, 0.1f), true);
    Tensor mnw = vk.upload(rand_f32({H}, rng, 0.5f), true);
    Tensor guw = vk.upload(rand_f32({2 * inter, H}, rng, 0.1f), true);
    Tensor dw = vk.upload(rand_f32({H, inter}, rng, 0.1f), true);

    Tensor qkv_buf = vk.scratch_arena().alloc_tensor({qkv_dim}, DType::F32);
    Tensor q_buf = vk.scratch_arena().alloc_tensor({q_dim}, DType::F32);
    Tensor k_buf = vk.scratch_arena().alloc_tensor({kv_dim}, DType::F32);
    Tensor v_buf = vk.scratch_arena().alloc_tensor({kv_dim}, DType::F32);
    Tensor kc = vk.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
    Tensor vc = vk.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
    Tensor attn_out = vk.scratch_arena().alloc_tensor({q_dim}, DType::F32);
    Tensor o_buf = vk.scratch_arena().alloc_tensor({H}, DType::F32);
    Tensor gu_buf = vk.scratch_arena().alloc_tensor({2 * inter}, DType::F32);
    Tensor sw_buf = vk.scratch_arena().alloc_tensor({inter}, DType::F32);
    Tensor d_buf = vk.scratch_arena().alloc_tensor({H}, DType::F32);
    Tensor x_out1 = vk.scratch_arena().alloc_tensor({H}, DType::F32);
    Tensor x_out2 = vk.scratch_arena().alloc_tensor({H}, DType::F32);

    std::vector<std::vector<uint8_t>> p1, p2;
    auto ops1 = vk_ops::build_dense_layer_dispatches(
        x_in, anw, qkvw, qnw, knw, ow, mnw, guw, dw,
        qkv_buf, q_buf, k_buf, v_buf, kc, vc, attn_out, o_buf, gu_buf, sw_buf, d_buf, x_out1,
        H, nq, nkv, hd, hd, inter, 0, 1, 1000000.0f, 1e-6f, true, true, p1);
    auto ops2 = vk_ops::build_dense_layer_dispatches(
        x_in, anw, qkvw, qnw, knw, ow, mnw, guw, dw,
        qkv_buf, q_buf, k_buf, v_buf, kc, vc, attn_out, o_buf, gu_buf, sw_buf, d_buf, x_out2,
        H, nq, nkv, hd, hd, inter, 0, 1, 1000000.0f, 1e-6f, true, true, p2);

    vk.dispatch_unbatched(ops1, true);
    vk.dispatch_batched(ops2, true);
    HostTensor h1 = vk.download(x_out1);
    HostTensor h2 = vk.download(x_out2);
    std::vector<double> ref(h1.data<float>(), h1.data<float>() + H);
    expect_close(std::vector<float>(h2.data<float>(), h2.data<float>() + H), ref, 1e-6, 1e-6, "Vulkan batched vs unbatched layer");
}
#endif  // FUSIONRT_ENABLE_VULKAN

#if defined(FUSIONRT_ENABLE_CUDA) && defined(FUSIONRT_ENABLE_VULKAN)
void test_phase2_loaders_prefill_and_vulkan_e2e(CudaContext& cuda_ctx) {
    std::cout << "[8/8] Testing full GGUF/Safetensors loading, batched prefill, and Vulkan E2E parity...\n";
    std::mt19937_64 rng(909);
    auto tmp_dir = std::filesystem::temp_directory_path() / "fusionrt_phase2_test";
    std::filesystem::create_directories(tmp_dir);

    // 8a. Full 2-layer Qwen3/Llama-style GGUF (Q4_0 weights!) + explicit missing-tensor rejection
    {
        int H = 64, nq = 4, nkv = 2, hd = 16, inter = 128, vocab = 64, nlayers = 2;
        int q_dim = nq * hd, kv_dim = nkv * hd;
        auto q4_mat = [&](int rows, int cols) {
            HostTensor f = rand_f32({rows, cols}, rng, 0.1f);
            HostTensor q = HostTensor::create({rows, cols}, DType::Q4_0);
            int nb = cols / 32;
            for (int r = 0; r < rows; ++r) {
                quantize_row_q4_0(f.data<float>() + r * cols,
                                  reinterpret_cast<BlockQ4_0*>(q.storage.data()) + r * nb, cols);
            }
            return q;
        };

        std::map<std::string, GgufMetaValue> meta;
        meta["general.architecture"] = std::string("llama");
        meta["llama.block_count"] = static_cast<int32_t>(nlayers);
        meta["llama.embedding_length"] = static_cast<int32_t>(H);
        meta["llama.attention.head_count"] = static_cast<int32_t>(nq);
        meta["llama.attention.head_count_kv"] = static_cast<int32_t>(nkv);
        meta["llama.attention.key_length"] = static_cast<int32_t>(hd);
        meta["llama.feed_forward_length"] = static_cast<int32_t>(inter);
        meta["llama.context_length"] = static_cast<int32_t>(256);
        meta["llama.rope.freq_base"] = 10000.0f;
        meta["llama.rope.dimension_count"] = static_cast<int32_t>(hd);
        meta["llama.attention.layer_norm_rms_epsilon"] = 1e-6f;
        std::vector<std::string> toks(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) toks[static_cast<size_t>(i)] = "t" + std::to_string(i);
        meta["tokenizer.ggml.tokens"] = toks;

        std::vector<std::pair<std::string, HostTensor>> tensors;
        tensors.emplace_back("token_embd.weight", q4_mat(vocab, H));
        tensors.emplace_back("output_norm.weight", make_ones_f32({H}));
        tensors.emplace_back("output.weight", q4_mat(vocab, H));
        for (int l = 0; l < nlayers; ++l) {
            std::string p = "blk." + std::to_string(l) + ".";
            tensors.emplace_back(p + "attn_norm.weight", make_ones_f32({H}));
            tensors.emplace_back(p + "attn_q.weight", q4_mat(q_dim, H));
            tensors.emplace_back(p + "attn_k.weight", q4_mat(kv_dim, H));
            tensors.emplace_back(p + "attn_v.weight", q4_mat(kv_dim, H));
            tensors.emplace_back(p + "attn_output.weight", q4_mat(H, q_dim));
            tensors.emplace_back(p + "ffn_norm.weight", make_ones_f32({H}));
            tensors.emplace_back(p + "ffn_gate.weight", q4_mat(inter, H));
            tensors.emplace_back(p + "ffn_up.weight", q4_mat(inter, H));
            tensors.emplace_back(p + "ffn_down.weight", q4_mat(H, inter));
        }

        std::string gguf_p = (tmp_dir / "full_q4.gguf").string();
        std::string err;
        if (!GgufFile::write_v3(gguf_p, meta, tensors, &err)) throw std::runtime_error(err);

        cuda_ctx.weight_arena().reset();
        cuda_ctx.scratch_arena().reset();
        FusionModel m_cuda(HybridModelConfig{});
        if (!m_cuda.load_gguf(cuda_ctx, gguf_p, &err)) throw std::runtime_error("Full GGUF CUDA load failed: " + err);
        if (m_cuda.dense_layers().size() != 2 || m_cuda.dense_layers()[0].o_proj.dtype != DType::Q4_0) {
            throw std::runtime_error("GGUF loader failed to preserve Q4_0 weights on GPU!");
        }
        auto cuda_ids = m_cuda.generate_cuda(cuda_ctx, {1, 5, 9}, 4, true, false);

        VkContext vk(0, 64, 64);
        FusionModel m_vk(HybridModelConfig{});
        if (!m_vk.load_gguf_vulkan(vk, gguf_p, &err)) throw std::runtime_error("Full GGUF Vulkan load failed: " + err);
        auto vk_ids = m_vk.generate_vulkan(vk, {1, 5, 9}, 4);
        if (cuda_ids != vk_ids) throw std::runtime_error("Full Q4_0 GGUF CUDA vs Vulkan token mismatch!");
        std::cout << "  [PASS/CHECK] Full 2-layer Q4_0 GGUF CUDA vs Vulkan exact token parity\n";

        // Test explicit missing-tensor rejection (drop last tensor)
        tensors.pop_back();
        std::string bad_gguf = (tmp_dir / "incomplete.gguf").string();
        GgufFile::write_v3(bad_gguf, meta, tensors, &err);
        FusionModel m_bad(HybridModelConfig{});
        if (m_bad.load_gguf(cuda_ctx, bad_gguf, &err)) {
            throw std::runtime_error("Expected load_gguf to fail on incomplete GGUF!");
        }
        std::cout << "  [PASS/CHECK] Incomplete GGUF explicitly rejected: " << err << "\n";
    }

    // 8b. Full Safetensors (F16 + config.json) load test
    {
        int H = 64, nq = 4, nkv = 2, hd = 16, inter = 128, vocab = 64;
        int q_dim = nq * hd, kv_dim = nkv * hd;
        auto f16_mat = [&](int rows, int cols) {
            HostTensor f = rand_f32({rows, cols}, rng, 0.1f);
            HostTensor h = HostTensor::create({rows, cols}, DType::F16);
            convert_f32_to_f16(f.data<float>(), reinterpret_cast<uint16_t*>(h.storage.data()), rows * cols);
            return h;
        };
        auto ones_f16 = [&](int n) {
            HostTensor f = make_ones_f32({n});
            HostTensor h = HostTensor::create({n}, DType::F16);
            convert_f32_to_f16(f.data<float>(), reinterpret_cast<uint16_t*>(h.storage.data()), n);
            return h;
        };
        std::string st_p = (tmp_dir / "model.safetensors").string();
        std::string cfg_p = (tmp_dir / "config.json").string();
        {
            std::ofstream ofs(cfg_p);
            ofs << "{\"model_type\":\"llama\",\"hidden_size\":64,\"num_hidden_layers\":1,"
                << "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":16,"
                << "\"intermediate_size\":128,\"vocab_size\":64,\"tie_word_embeddings\":false}";
        }
        std::vector<std::pair<std::string, HostTensor>> tensors;
        tensors.emplace_back("model.embed_tokens.weight", f16_mat(vocab, H));
        tensors.emplace_back("model.norm.weight", ones_f16(H));
        tensors.emplace_back("lm_head.weight", f16_mat(vocab, H));
        tensors.emplace_back("model.layers.0.input_layernorm.weight", ones_f16(H));
        tensors.emplace_back("model.layers.0.self_attn.q_proj.weight", f16_mat(q_dim, H));
        tensors.emplace_back("model.layers.0.self_attn.k_proj.weight", f16_mat(kv_dim, H));
        tensors.emplace_back("model.layers.0.self_attn.v_proj.weight", f16_mat(kv_dim, H));
        tensors.emplace_back("model.layers.0.self_attn.o_proj.weight", f16_mat(H, q_dim));
        tensors.emplace_back("model.layers.0.post_attention_layernorm.weight", ones_f16(H));
        tensors.emplace_back("model.layers.0.mlp.gate_proj.weight", f16_mat(inter, H));
        tensors.emplace_back("model.layers.0.mlp.up_proj.weight", f16_mat(inter, H));
        tensors.emplace_back("model.layers.0.mlp.down_proj.weight", f16_mat(H, inter));

        std::string err;
        if (!SafetensorsFile::write(st_p, {{"format", "pt"}}, tensors, &err)) throw std::runtime_error(err);
        cuda_ctx.weight_arena().reset();
        cuda_ctx.scratch_arena().reset();
        FusionModel m_st(HybridModelConfig{});
        if (!m_st.load_safetensors(cuda_ctx, st_p, &err)) throw std::runtime_error("Safetensors load failed: " + err);
        if (m_st.dense_layers()[0].o_proj.dtype != DType::F16) {
            throw std::runtime_error("Safetensors loader failed to preserve F16 weights on GPU!");
        }
        auto st_ids = m_st.generate_cuda(cuda_ctx, {1, 3, 7}, 3, true, false);
        if (st_ids.size() != 6) throw std::runtime_error("Safetensors generation length mismatch");
        std::cout << "  [PASS/CHECK] Full F16 Safetensors + config.json load & generation OK\n";
    }

    // 8c. Batched Prefill (M=5) vs Token-by-Token + Vulkan E2E parity across architectures
    for (const std::string& arch : {"dense", "hybrid", "moe"}) {
        cuda_ctx.weight_arena().reset();
        cuda_ctx.scratch_arena().reset();
        HybridModelConfig cfg;
        cfg.arch = arch;
        cfg.vocab_size = 128;
        cfg.num_layers = 4;
        cfg.dense.hidden_size = 128;
        cfg.dense.num_attention_heads = 4;
        cfg.dense.num_key_value_heads = 2;
        cfg.dense.head_dim = 32;
        cfg.dense.intermediate_size = 256;
        cfg.dense.max_seq_len = 256;
        cfg.gdn.hidden_size = 128;
        cfg.gdn.num_k_heads = 2;
        cfg.gdn.num_v_heads = 4;
        cfg.gdn.head_k_dim = 32;
        cfg.gdn.head_v_dim = 32;
        cfg.gdn.intermediate_size = 256;
        cfg.moe.hidden_size = 128;
        cfg.moe.intermediate_size = 128;
        cfg.moe.num_experts = 4;
        cfg.moe.top_k = 2;
        cfg.moe.gpu_resident_experts = 4;
        cfg.moe.shared_intermediate_size = 128;

        FusionModel m_cuda(cfg);
        m_cuda.init_synthetic(cuda_ctx, 999);
        std::vector<int32_t> prompt = {1, 11, 22, 33, 44};

        // Token-by-token reference
        m_cuda.reset_kv_and_state(cuda_ctx);
        std::vector<float> ref_logits;
        for (int p = 0; p < static_cast<int>(prompt.size()); ++p) {
            ref_logits = m_cuda.forward_token_cuda(cuda_ctx, prompt[static_cast<size_t>(p)], p, true, false);
        }
        // Batched prefill
        m_cuda.reset_kv_and_state(cuda_ctx);
        auto pf_logits = m_cuda.forward_prefill_cuda(cuda_ctx, prompt, 0, true);
        std::vector<double> ref_d(ref_logits.begin(), ref_logits.end());
        expect_close(pf_logits, ref_d, 1e-4, 1e-4, "Batched prefill M=5 vs sequential (" + arch + ")");

        auto cuda_gen = m_cuda.generate_cuda(cuda_ctx, prompt, 3, true, false);
        auto vk_devs = VkContext::enumerate_devices();
        for (const auto& vd : vk_devs) {
            VkContext vk(vd.device_index, 128, 128);
            FusionModel m_vk(cfg);
            m_vk.init_synthetic_vulkan(vk, 999);
            auto vk_gen = m_vk.generate_vulkan(vk, prompt, 3);
            if (cuda_gen != vk_gen) {
                throw std::runtime_error("Vulkan E2E generation diverged from CUDA on " + vd.name + " (" + arch + ")");
            }
            std::cout << "  [PASS/CHECK] Vulkan[" << vd.device_index << "] E2E " << arch << " token parity vs CUDA\n";
        }
    }

    std::filesystem::remove_all(tmp_dir);
}
#endif

void test_wave3() {
    std::cout << "Testing sampling, UTF-8, device profile, GGUF biases and prefix reuse...\n";
    auto check = [](bool passed, const char* error) {
        if (!passed) throw std::runtime_error(error);
    };
    const float logits[] = {3.0f, 2.0f, -4.0f};
    std::mt19937_64 rng(42);
    SamplingParams params;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Greedy sampler");
    params.repeat_penalty = 2.0f;
    check(Sampler::sample(logits, 3, {0, 0}, params, rng) == 1, "Repetition penalty");
    params.repeat_penalty = 1.0f;
    params.temperature = 0.7f;
    params.top_k = 1;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Top-k sampler");
    params.top_k = 0;
    params.top_p = 0.1f;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Top-p sampler");
    params.top_p = 1.0f;
    params.min_p = 0.5f;
    check(Sampler::sample(logits, 3, {}, params, rng) == 0, "Min-p sampler");
    params.min_p = 0.0f;
    std::mt19937_64 first(42), second(42);
    bool saw_second = false;
    for (int i = 0; i < 32; ++i) {
        int32_t a = Sampler::sample(logits, 3, {}, params, first);
        check(a == Sampler::sample(logits, 3, {}, params, second), "Seeded sampler reproducibility");
        saw_second |= a == 1;
    }
    check(saw_second, "Temperature sampling yielded only the argmax");
    Utf8StreamDecoder utf8;
    check(utf8.push("A\xe4").compare("A") == 0 && utf8.push("\xb8").empty() &&
          utf8.push("\xad" "B") == "\xe4\xb8\xad" "B" && utf8.finish().empty(), "UTF-8 token split");
    check(utf8.push("\xed\xa0\x80") == "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd", "Invalid UTF-8");

    struct Fixtures {
        std::filesystem::path profile = "test-profile.json";
        std::filesystem::path model = "test-model.gguf";
        ~Fixtures() {
            std::error_code ec;
            std::filesystem::remove(profile, ec);
            std::filesystem::remove(model, ec);
        }
    } files;
    std::ofstream(files.profile) << R"({"schema_version":"device_profile.v1","mmq_crossover_m":7,
        "recommend_cuda_graph":true,"empirical_l2_bytes":33554432,
        "preferred_gemv_warps":4,"preferred_gemv_rows_per_warp":2})";
    const auto profile = DeviceProfile::load(files.profile.string());
    check(profile.mmq_crossover_m == 7 && profile.recommend_cuda_graph &&
          profile.empirical_l2_bytes == 33554432 && profile.preferred_gemv_warps == 4 &&
          profile.preferred_gemv_rows_per_warp == 2, "Device profile fields");

#ifdef FUSIONRT_ENABLE_CUDA
    CudaContext ctx(0, 64, 64);
    std::map<std::string, GgufMetaValue> metadata{
        {"general.architecture", std::string("qwen2")}, {"qwen2.block_count", uint32_t{1}},
        {"qwen2.embedding_length", uint32_t{256}}, {"qwen2.attention.head_count", uint32_t{8}},
        {"qwen2.attention.head_count_kv", uint32_t{2}}, {"qwen2.feed_forward_length", uint32_t{256}},
        {"qwen2.rope.freq_base", 12345.0f}, {"qwen2.rope.dimension_count", uint32_t{16}},
        {"tokenizer.ggml.model", std::string("gpt2")},
        {"tokenizer.ggml.tokens", std::vector<std::string>{"<|im_start|>", "<|im_end|>", "a"}},
        {"tokenizer.chat_template", std::string("{{ '<|im_start|>' }}{{ '<|im_end|>' }}")}
    };
    const auto emb = rand_f32({32, 256}, rng, 0.1f);
    HostTensor q_emb = HostTensor::create({32, 256}, DType::Q4_0);
    for (int i = 0; i < 32; ++i) {
        quantize_row_q4_0(emb.data<float>() + i * 256, q_emb.data<BlockQ4_0>() + i * 8, 256);
    }
    std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", q_emb}, {"output_norm.weight", make_ones_f32({256})},
        {"blk.0.attn_norm.weight", make_ones_f32({256})}, {"blk.0.ffn_norm.weight", make_ones_f32({256})},
        {"blk.0.attn_q.weight", rand_f32({256, 256}, rng, 0.03f)}, {"blk.0.attn_q.bias", rand_f32({256}, rng)},
        {"blk.0.attn_k.weight", rand_f32({64, 256}, rng, 0.03f)}, {"blk.0.attn_k.bias", rand_f32({64}, rng)},
        {"blk.0.attn_v.weight", rand_f32({64, 256}, rng, 0.03f)}, {"blk.0.attn_v.bias", rand_f32({64}, rng)},
        {"blk.0.attn_output.weight", rand_f32({256, 256}, rng, 0.03f)},
        {"blk.0.ffn_gate.weight", rand_f32({256, 256}, rng, 0.03f)},
        {"blk.0.ffn_up.weight", rand_f32({256, 256}, rng, 0.03f)},
        {"blk.0.ffn_down.weight", rand_f32({256, 256}, rng, 0.03f)}
    };
    FusionModel tied(HybridModelConfig{}), untied(HybridModelConfig{});
    std::string error;
    const auto model_path = files.model.string();
    auto load = [&](FusionModel& model) {
        if (!GgufFile::write_v3(model_path, metadata, weights, &error) ||
            !model.load_gguf(ctx, model_path, &error, 32)) throw std::runtime_error(error);
    };
    load(tied);
    GgufFile gguf;
    BpeTokenizer tokenizer;
    check(gguf.load(model_path) && tokenizer.load_from_gguf(gguf) &&
          tokenizer.format_chat_messages({{"user", "a"}}) == "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n" &&
          tokenizer.is_end_of_generation(1), "ChatML metadata");
    tied.set_device_profile(ctx, profile);
    const auto geometry = ctx.preferred_gemv_geometry();
    check(ctx.mmq_crossover_m() == 7 && tied.recommended_cuda_graph() && ctx.empirical_l2_bytes() == 33554432 &&
          geometry.warps == 4 && geometry.rows_per_warp == 2, "Device profile applied");
    const auto& layer = tied.dense_layers().front();
    check(layer.q_bias.data && layer.k_bias.data && layer.v_bias.data && layer.cfg.rope_neox &&
          layer.cfg.partial_rotary_factor == 0.5f && layer.cfg.rope_theta == 12345.0f, "Qwen2 metadata and QKV biases");
    weights.emplace_back("output.weight", q_emb);
    load(untied);
    const std::vector<int32_t> prefix{1, 2, 3}, branched{1, 2, 5, 6};
    untied.reset_kv_and_state(ctx);
    const auto explicit_logits = untied.forward_prefill_cuda(ctx, branched, 0);
    tied.prefill_cached_cuda(ctx, prefix);
    int prefilled = 0;
    const auto reused = tied.prefill_cached_cuda(ctx, branched, true, &prefilled);
    check(prefilled == 2, "Dense prefix should reuse KV before a divergence");
    expect_close(reused, std::vector<double>(explicit_logits.begin(), explicit_logits.end()),
                 1e-5, 1e-5, "Biased Qwen2 with reused KV and tied embeddings");
    const std::vector<int32_t> repeated{1, 2, 5, 6, 7};
    const auto first_logits = tied.prefill_cached_cuda(ctx, repeated);
    const auto repeat = tied.prefill_cached_cuda(ctx, repeated, true, &prefilled);
    check(prefilled == 1 && repeat == first_logits, "Dense exact repeat should replay one token with identical logits");
    ctx.weight_arena().reset();
    std::erase_if(weights, [](const auto& row) { return row.first.ends_with(".bias"); });
    FusionModel unbiased(HybridModelConfig{});
    load(unbiased);
    const auto no_bias = unbiased.prefill_cached_cuda(ctx, branched);
    check(std::abs(no_bias[0] - explicit_logits[0]) > 1e-4, "QKV biases were ignored in the graph");

    for (const auto& [key, value] : std::map(metadata)) {
        if (key.starts_with("qwen2.")) metadata["qwen3moe." + key.substr(6)] = value;
    }
    metadata["general.architecture"] = std::string("qwen3moe");
    weights.emplace_back("blk.0.ffn_gate_inp.weight", rand_f32({2, 256}, rng));
    weights.emplace_back("blk.0.ffn_gate_up_exps.weight", rand_f32({2, 64, 256}, rng, 0.03f));
    weights.emplace_back("blk.0.ffn_down_exps.weight", rand_f32({2, 256, 32}, rng, 0.03f));
    FusionModel routed(HybridModelConfig{});
    load(routed);
    check(routed.dense_layers().size() == 1 && routed.moe_layers().size() == 1 &&
          std::isfinite(routed.prefill_cached_cuda(ctx, branched)[0]), "GGUF MoE retains its GQA attention");

    HybridModelConfig hybrid;
    hybrid.arch = "hybrid";
    hybrid.num_layers = 1;
    hybrid.vocab_size = 32;
    hybrid.dense.max_seq_len = 32;
    FusionModel recurrent(hybrid);
    recurrent.init_synthetic(ctx, 111);
    GenerationOptions options;
    options.reuse_prefix = true;
    recurrent.generate_cuda(ctx, prefix, 0, true, false, nullptr, false, 0, -1, options);
    const std::vector<int32_t> extended{1, 2, 3, 4, 5};
    const auto appended = recurrent.prefill_cached_cuda(ctx, extended, true, &prefilled);
    check(prefilled == 2, "Hybrid prefix should append to live state");
    const auto reset = recurrent.prefill_cached_cuda(ctx, branched, true, &prefilled);
    check(prefilled == static_cast<int>(branched.size()), "Hybrid divergence must reset state");
    auto fresh = [&](const auto& ids) {
        recurrent.reset_kv_and_state(ctx);
        const auto scores = recurrent.forward_prefill_cuda(ctx, ids, 0);
        return std::vector<double>(scores.begin(), scores.end());
    };
    expect_close(appended, fresh(extended), 1e-5, 1e-5, "Hybrid appending cached state");
    expect_close(reset, fresh(branched), 1e-5, 1e-5, "Hybrid divergence resets cached state");
#endif
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_chunked_prefill_and_k_quant_gdn() {
    std::cout << "Testing chunked prefill and K-quant GDN GGUF...\n";
    CudaContext ctx(0, 64, 64);
    ctx.set_prefill_batch(128);
    const auto mark = ctx.weight_arena().mark();
    for (const std::string arch : {"dense", "hybrid"}) {
        HybridModelConfig cfg;
        cfg.arch = arch;
        cfg.num_layers = 2;
        cfg.vocab_size = 64;
        cfg.dense.hidden_size = cfg.gdn.hidden_size = 256;
        cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 256;
        cfg.dense.max_seq_len = 160;
        FusionModel model(cfg);
        model.init_synthetic(ctx, 13);
        std::vector<int32_t> prompt(129);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<int32_t>(i % 63 + 1);
        for (int count : {127, 128, 129}) {
            const std::vector<int32_t> tokens(prompt.begin(), prompt.begin() + count);
            const std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + 65);
            model.reset_kv_and_state(ctx);
            OpTrace ops;
            LayerTrace layers;
            const auto whole = model.prefill_cached_cuda(ctx, tokens, true, nullptr, &layers, &ops);
            const auto next = model.forward_token_cuda(ctx, 8, count);
            const auto head = std::find_if(ops.ops.begin(), ops.ops.end(),
                [](const OpTiming& op) { return op.name == "lm_head"; });
            if (head == ops.ops.end() || head->calls != 1 || layers.outputs.size() != 2) {
                throw std::runtime_error("Chunked prefill must project logits once and trace the final chunk");
            }
            model.reset_kv_and_state(ctx);
            model.prefill_cached_cuda(ctx, prefix);
            int processed = 0;
            const auto resumed = model.prefill_cached_cuda(ctx, tokens, true, &processed);
            const auto resumed_next = model.forward_token_cuda(ctx, 8, count);
            if (processed != count - static_cast<int>(prefix.size())) {
                throw std::runtime_error("Chunked prefill did not reuse the cached prefix");
            }
            expect_close(resumed, {whole.begin(), whole.end()}, 1e-4, 1e-4,
                         arch + " prefix prefill " + std::to_string(count));
            expect_close(resumed_next, {next.begin(), next.end()}, 1e-4, 1e-4,
                         arch + " first decode after " + std::to_string(count));
        }
        ctx.weight_arena().reset(mark);
    }

    struct Fixture {
        std::filesystem::path path = "k-quant-gdn-test.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    ctx.weight_arena().init(0);
    constexpr int H = 256, K = 2, V = 4, D = 64;
    constexpr int conv = 2 * K * D + V * D;
    std::mt19937_64 rng(748);
    auto quantized = [&](int rows) {
        HostTensor result = HostTensor::create({rows, H}, DType::Q4_K);
        for (int row = 0; row < rows; ++row) {
            auto& block = result.data<BlockQ4_K>()[row];
            block.d = fp32_to_fp16(0.002f);
            block.dmin = fp32_to_fp16(0.012f);
            std::fill(std::begin(block.scales), std::end(block.scales), uint8_t{1});
            for (auto& q : block.qs) q = static_cast<uint8_t>(rng());
        }
        return result;
    };
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("qwen35")}, {"qwen35.block_count", uint32_t{1}},
        {"qwen35.embedding_length", uint32_t{H}}, {"qwen35.feed_forward_length", uint32_t{H}},
        {"qwen35.ssm.group_count", uint32_t{K}}, {"qwen35.ssm.time_step_rank", uint32_t{V}},
        {"qwen35.ssm.state_size", uint32_t{D}}, {"qwen35.ssm.inner_size", uint32_t{V * D}},
    };
    const std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", rand_f32({32, H}, rng, 0.1f)}, {"output_norm.weight", make_ones_f32({H})},
        {"blk.0.attn_norm.weight", make_ones_f32({H})}, {"blk.0.attn_qkv.weight", quantized(conv)},
        {"blk.0.attn_gate.weight", quantized(V * D)}, {"blk.0.ssm_alpha.weight", rand_f32({V, H}, rng, 0.03f)},
        {"blk.0.ssm_beta.weight", rand_f32({V, H}, rng, 0.03f)}, {"blk.0.ssm_conv1d.weight", rand_f32({conv, 4}, rng)},
        {"blk.0.ssm_a", rand_f32({V}, rng)}, {"blk.0.ssm_dt.bias", rand_f32({V}, rng)},
        {"blk.0.ssm_norm.weight", make_ones_f32({D})}, {"blk.0.ssm_out.weight", rand_f32({H, V * D}, rng, 0.03f)},
        {"blk.0.ffn_norm.weight", make_ones_f32({H})}, {"blk.0.ffn_gate.weight", rand_f32({H, H}, rng, 0.03f)},
        {"blk.0.ffn_up.weight", rand_f32({H, H}, rng, 0.03f)}, {"blk.0.ffn_down.weight", rand_f32({H, H}, rng, 0.03f)},
    };
    FusionModel model(HybridModelConfig{});
    std::string error;
    const auto path = fixture.path.string();
    if (!GgufFile::write_v3(path, meta, weights, &error) || !model.load_gguf(ctx, path, &error, 32)) {
        throw std::runtime_error("K-quant GDN GGUF load failed: " + error);
    }
    const auto& layer = model.gdn_layers().front();
    if (layer.qkv_proj.dtype != DType::Q4_KA || layer.z_proj.dtype != DType::Q4_KA ||
        layer.a_proj.dtype != DType::F32 || layer.b_proj.dtype != DType::F32) {
        throw std::runtime_error("GDN projections must preserve their weight precision");
    }
    const std::vector<int32_t> prompt{1, 2, 3, 4, 5};
    model.reset_kv_and_state(ctx);
    const auto logits = model.forward_prefill_cuda(ctx, prompt, 0);
    const auto next = model.forward_token_cuda(ctx, 6, static_cast<int>(prompt.size()));
    if (logits.size() != 32 || next.size() != 32 ||
        !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(next.begin(), next.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("K-quant GDN must produce finite prefill and decode logits");
    }
}
#endif

}  // namespace

int main() {
    std::cout << std::unitbuf;
    try {
        test_quant_and_io();
        test_agent_protocols();
#ifdef FUSIONRT_ENABLE_CUDA
        CudaContext cuda_ctx(0, 256, 512);
        test_cuda_fused_rmsnorm_linear(cuda_ctx);
        test_cuda_step1_fast_paths(cuda_ctx);
        test_cuda_gated_deltanet_fused_vs_unfused_and_fp64(cuda_ctx);
        test_cuda_qk_rope_and_gqa_64k(cuda_ctx);
        test_cuda_graph_and_moe_offload(cuda_ctx);
        test_cuda_wave1_decode(cuda_ctx);
        test_repacked_k_quants(cuda_ctx);
        test_cuda_wave2_prefill(cuda_ctx);
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
        test_vulkan_all_devices();
        test_vulkan_batched_vs_unbatched();
#endif
#if defined(FUSIONRT_ENABLE_CUDA) && defined(FUSIONRT_ENABLE_VULKAN)
        test_phase2_loaders_prefill_and_vulkan_e2e(cuda_ctx);
#endif
        test_wave3();
#ifdef FUSIONRT_ENABLE_CUDA
        test_chunked_prefill_and_k_quant_gdn();
#endif
        std::cout << "\nALL TESTS PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
