#include "fusionrt/bench.h"
#include "fusionrt/cost_model.h"
#include "fusionrt/gguf.h"
#include "fusionrt/models.h"
#include "fusionrt/safetensors.h"
#include "fusionrt/tokenizer.h"
#include "fusionrt/vision.h"
#include "agent.h"
#include "http_request.h"
#include "../src/cli/cli_args.h"
#include "../src/models/model_loader.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "../src/core/cpu_moe.h"
#include "fusionrt/cuda_backend.h"
#include "fusionrt/cuda_kernels.h"
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#include "fusionrt/vulkan_kernels.h"
#endif
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

using namespace fusionrt;

namespace {

void test_device_profile_and_cost_model() {
    auto check = [](bool ok, const char* label) {
        if (!ok) throw std::runtime_error(std::string("Device cost model: ") + label);
    };
    struct Fixture {
        std::filesystem::path path = "test-device-cost-profile.json";
        ~Fixture() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } fixture;
    auto load = [&](const char* json) {
        std::ofstream(fixture.path) << json;
        return DeviceProfile::load(fixture.path.string());
    };
    const auto wave1 = load(R"({"schema_version":"device_profile.v1","mmq_crossover_m":7,
        "recommend_cuda_graph":true,"empirical_l2_bytes":33554432,"measured_dp4a_gemv_gbps":430.5,
        "empirical_l1_kb":48,"graph_replay_us":3.125})");
    check(wave1.mmq_crossover_m == 7 && wave1.recommend_cuda_graph && wave1.empirical_l2_bytes == 33554432 &&
          wave1.measured_dp4a_gemv_gbps == 430.5 && wave1.empirical_l1_kb == 48 && wave1.graph_replay_us == 3.125 &&
          wave1.sm_count == 0 && wave1.cpu_q4_gemv_gbps == 0, "Wave 1 measurements and defaults");
    const auto wave2 = load(R"({"schema_version":"device_profile.v1","mmq_crossover_m":0,
        "sm_count":46,"vram_total_mb":12199.75,"measured_hbm_bw_gbps":548.5,"measured_l2_bw_gbps":2100,
        "reported_l2_bytes":37748736,"measured_fp16_tc_tflops":50.125,"measured_int8_tc_tflops":162.625,
        "launch_floor_us":4.125,"host_ram_mb":31965.8125,"cpu_ddr_bw_gbps":39.875,"cpu_q4_gemv_gbps":51.375,
        "pcie_h2d_gbps":14.25,"pcie_d2h_gbps":14.5,"pcie_latency_us":12.0625})");
    check(wave2.mmq_crossover_m == DeviceProfile{}.mmq_crossover_m && wave2.sm_count == 46 &&
          wave2.vram_total_mb == 12199.75 && wave2.measured_hbm_bw_gbps == 548.5 &&
          wave2.measured_l2_bw_gbps == 2100 && wave2.reported_l2_bytes == 37748736 &&
          wave2.measured_fp16_tc_tflops == 50.125 && wave2.measured_int8_tc_tflops == 162.625 &&
          wave2.launch_floor_us == 4.125 && wave2.host_ram_mb == 31965.8125 &&
          wave2.cpu_ddr_bw_gbps == 39.875 && wave2.cpu_q4_gemv_gbps == 51.375 &&
          wave2.pcie_h2d_gbps == 14.25 && wave2.pcie_d2h_gbps == 14.5 &&
          wave2.pcie_latency_us == 12.0625, "Wave 2 measurements and unmeasured crossover");
    if (const char* path = std::getenv("FUSIONRT_TEST_WAVE2_PROFILE"); path && *path) {
        const auto measured = DeviceProfile::load(path);
        check(measured.sm_count > 0 && measured.vram_total_mb > 0 && measured.measured_hbm_bw_gbps > 0 &&
              measured.measured_fp16_tc_tflops > 0 && measured.measured_int8_tc_tflops > 0 &&
              measured.cpu_ddr_bw_gbps > 0 && measured.cpu_q4_gemv_gbps > 0 &&
              measured.pcie_h2d_gbps > 0 && measured.pcie_d2h_gbps > 0 && measured.pcie_latency_us > 0,
              "inferpeak Wave 2 fixture");
        std::cout << "inferpeak Wave 2 device profile loaded\n";
    }

    CostHardware hw{.sm_count = 46, .hbm_bw_gbps = 549, .int8_tc_tflops = 160, .l2_bytes = 36 * 1024 * 1024,
                    .cpu_q4_gemv_gbps = 51.38, .pcie_h2d_gbps = 14.24, .pcie_latency_us = 12, .l2_bw_gbps = 2400};
    auto tile_for = [&](int sms, int rows) {
        hw.sm_count = sms;
        return cost_detail::mmq_tiles[static_cast<unsigned>(
            select_mmq_tile(hw, rows, 4096, 4096, sizeof(BlockQ4_KA), true, true))];
    };
    const auto small_gpu = tile_for(28, 128);
    const auto mid_gpu = tile_for(46, 128);
    const auto large_gpu = tile_for(128, 128);
    check(small_gpu.m >= tile_for(28, 16).m && mid_gpu.m > tile_for(46, 16).m &&
          large_gpu.m >= tile_for(128, 16).m, "MMQ tile scales with prompt rows");
    check(large_gpu.m * large_gpu.n < mid_gpu.m * mid_gpu.n, "MMQ tiles expose more CTAs on a larger GPU");
    hw.sm_count = 46;
    check(!should_repack_q6_a(hw, 151936, 4096, true) && should_repack_q6_a(hw, 4096, 4096, false),
          "Q6 decode head and prefill projection");
    check(select_decode_splits(hw, 64, 8, 128) == select_decode_splits(hw, 128, 8, 128) &&
          select_decode_splits(hw, 64, 4, 128) > select_decode_splits(hw, 64, 8, 128) &&
          select_decode_splits(hw, 8192, 8, 128) > select_decode_splits(hw, 128, 8, 128) &&
          select_decode_splits(hw, 8192, 8, 8) == 8, "Decode split buckets and cap");
    check(select_cold_expert_path(hw, 8 * 1024 * 1024) == ColdExpertPath::CpuInPlace,
          "5070 Ti cold expert");
    hw.cpu_q4_gemv_gbps = 10;
    hw.pcie_h2d_gbps = 64;
    check(select_cold_expert_path(hw, 8 * 1024 * 1024) == ColdExpertPath::PcieStream &&
          select_cold_expert_path(hw, 4096) == ColdExpertPath::CpuInPlace, "Fast PCIe and transfer startup cost");
}

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
        {"type":"reasoning","id":"rs_b","summary":[],"content":[{"type":"reasoning_text","text":"hidden thought"}]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Checking"}]},
        {"type":"function_call","name":"calc","call_id":"b","arguments":"{}"},
        {"type":"function_call_output","call_id":"b","output":"391"}]})", Protocol::responses);
    check(responses.generation.max_tokens == 64 &&
          std::get<ToolResult>(responses.messages.back().content[0]).content == "391", "Responses input replay");
    const auto anthropic = parse(R"({"system":[{"type":"text","text":"Be concise"}],"messages":[
        {"role":"user","content":"Calculate"},
        {"role":"assistant","content":[{"type":"thinking","thinking":"hidden thought"},{"type":"text","text":"Checking"},
            {"type":"tool_use","name":"calc","id":"c","input":{}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"c","is_error":true,"content":"Unavailable"}]}
    ]})", Protocol::messages);
    check(std::get<ToolResult>(anthropic.messages.back().content[0]).is_error, "Messages preserves tool failure");
    BpeTokenizer tokenizer;
    tokenizer.init_byte_fallback();
    for (const auto* request : {&responses, &anthropic}) {
        const auto rendered = render_prompt(*request, tokenizer);
        check(request->messages.size() == 4 && request->messages[2].content.size() == 2 &&
              std::holds_alternative<ToolCall>(request->messages[2].content[1]) &&
              rendered.text.find("Checking") != std::string::npos && rendered.text.find("<tool_call>") != std::string::npos &&
              rendered.text.find("<tool_response>") != std::string::npos && rendered.text.find("hidden thought") == std::string::npos,
              "Echoed reasoning keeps assistant text, tools and tool results");
    }
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

void test_agent_reasoning() {
    using namespace agent;
    auto check = [](bool ok, const char* label) {
        if (!ok) throw std::runtime_error(std::string("Agent reasoning: ") + label);
    };
    AgentRequest request;
    request.tools.push_back({"calc", "", {}});
    constexpr std::string_view thought = R"(plan <tool_call>{"name":"calc","arguments":{"hidden":true}}</tool_call>)";
    constexpr std::string_view call = R"(<tool_call>{"name":"calc","arguments":{"visible":true}}</tool_call>)";
    enum class Input { explicit_tags, open_thought, plain, empty_thought };
    for (const auto protocol : {Protocol::chat, Protocol::messages, Protocol::responses}) {
        for (const bool stream : {false, true}) {
            for (const auto input : {Input::explicit_tags, Input::open_thought, Input::plain, Input::empty_thought}) {
                const bool thinking = input == Input::explicit_tags || input == Input::open_thought;
                std::vector<std::pair<std::string, JsonValue>> frames;
                EventSink events;
                if (stream) events = [&](std::string_view event, std::string_view data) {
                    if (data != "[DONE]") frames.emplace_back(event, JsonValue::parse(std::string(data)));
                    return true;
                };
                AgentResponse response(protocol, "sample", "model", 0, false, std::move(events));
                response.start(10);
                const size_t initial_frames = frames.size();
                AgentPrompt prompt;
                prompt.open_thought = input == Input::open_thought;
                std::string parsed_thought;
                ToolParser parser(request, prompt, [&](std::string_view text) { response.text(text); },
                    [&](ToolCall tool) { response.call(std::move(tool)); },
                    [&](std::string_view text) { parsed_thought += text; response.thought(text); });
                if (input == Input::explicit_tags) {
                    parser.push("<th");
                    parser.push("ink>");
                }
                if (thinking) {
                    parser.push(thought.substr(0, 5));
                    check(parsed_thought == thought.substr(0, 5) && (!stream || frames.size() > initial_frames),
                          "safe bytes streamed before the closing tag");
                    parser.push(thought.substr(5));
                    parser.push("</thi");
                    check(parsed_thought == thought && !response.result().has_calls(),
                          "partial delimiter held and inner tool ignored");
                    parser.push("nk>");
                } else if (input == Input::empty_thought) parser.push("<think></think>");
                parser.push("answer");
                parser.push(call);
                parser.finish();
                response.finish(Finish::stop, 5);
                check(parsed_thought == (thinking ? thought : ""), "isolated thought text");
                const auto& blocks = response.result().output;
                const size_t first_text = thinking ? 1 : 0;
                check(blocks.size() == first_text + 2 && std::get<std::string>(blocks[first_text]) == "answer" &&
                      std::get<ToolCall>(blocks.back()).arguments.contains("visible"), "clean text and one real tool");

                const auto body = response.body();
                const auto& obj = body.as_object();
                std::string item_id;
                if (protocol == Protocol::chat) {
                    const auto& message = obj.at("choices").as_array()[0].as_object().at("message").as_object();
                    check(message.at("content").as_string() == "answer" && message.at("tool_calls").as_array().size() == 1 &&
                          message.contains("reasoning_content") == thinking &&
                          (!thinking || message.at("reasoning_content").as_string() == thought), "Chat message");
                } else {
                    const auto& items = obj.at(protocol == Protocol::messages ? "content" : "output").as_array();
                    const bool anthropic = protocol == Protocol::messages;
                    check(items.size() == first_text + 2 && items[first_text].as_object().at("type").as_string() ==
                          (anthropic ? "text" : "message") && items.back().as_object().at("type").as_string() ==
                          (anthropic ? "tool_use" : "function_call"), "answer and tool order");
                    if (thinking) {
                        const auto& reason = items[0].as_object();
                        if (anthropic) {
                            check(reason.at("type").as_string() == "thinking" &&
                                  reason.at("thinking").as_string() == thought, "Messages thinking block");
                        } else {
                            const auto& part = reason.at("content").as_array()[0].as_object();
                            item_id = reason.at("id").as_string();
                            check(reason.at("type").as_string() == "reasoning" && part.at("type").as_string() ==
                                  "reasoning_text" && part.at("text").as_string() == thought, "Responses reasoning item");
                        }
                    }
                }
                if (!stream) continue;
                std::string streamed_thought, streamed_text;
                std::vector<std::string> starts, stops;
                int reasoning_done = 0, tool_deltas = 0;
                for (const auto& [event, data] : frames) {
                    const auto& frame = data.as_object();
                    if (protocol == Protocol::chat) {
                        const auto& delta = frame.at("choices").as_array()[0].as_object().at("delta").as_object();
                        if (delta.contains("reasoning_content")) {
                            check(!delta.contains("content") && streamed_text.empty(), "Chat thought-only delta");
                            streamed_thought += delta.at("reasoning_content").as_string();
                        }
                        if (delta.contains("content")) streamed_text += delta.at("content").as_string();
                        tool_deltas += delta.contains("tool_calls");
                    } else if (protocol == Protocol::messages) {
                        if (event == "content_block_start") {
                            const auto& block = frame.at("content_block").as_object();
                            starts.push_back(block.at("type").as_string());
                            if (starts.back() == "thinking")
                                check(block.at("thinking").as_string().empty(), "Messages thinking starts empty");
                        } else if (event == "content_block_delta") {
                            const auto& delta = frame.at("delta").as_object();
                            if (delta.at("type").as_string() == "thinking_delta") {
                                check(frame.at("index").as_int() == 0 && stops.empty(), "Messages thinking delta index");
                                streamed_thought += delta.at("thinking").as_string();
                            }
                            if (delta.contains("text")) streamed_text += delta.at("text").as_string();
                            tool_deltas += delta.at("type").as_string() == "input_json_delta";
                        } else if (event == "content_block_stop") {
                            stops.push_back(starts.at(static_cast<size_t>(frame.at("index").as_int())));
                        }
                    } else {
                        if (event == "response.output_item.added" || event == "response.output_item.done") {
                            auto& order = event == "response.output_item.added" ? starts : stops;
                            order.push_back(frame.at("item").as_object().at("type").as_string());
                        } else if (event == "response.reasoning_text.delta" || event == "response.reasoning_text.done") {
                            check(frame.at("item_id").as_string() == item_id && frame.at("output_index").as_int() == 0 &&
                                  frame.at("content_index").as_int() == 0 && streamed_text.empty(), "Responses reasoning event indices");
                            if (event.ends_with(".delta")) streamed_thought += frame.at("delta").as_string();
                            else {
                                ++reasoning_done;
                                check(frame.at("text").as_string() == thought, "Responses final reasoning text");
                            }
                        } else if (event == "response.output_text.delta") streamed_text += frame.at("delta").as_string();
                        else if (event == "response.function_call_arguments.delta") ++tool_deltas;
                    }
                }
                check(streamed_thought == (thinking ? thought : "") && streamed_text == "answer" && tool_deltas == 1,
                      "stream keeps reasoning separate from answer and tools");
                if (protocol != Protocol::chat) {
                    const bool anthropic = protocol == Protocol::messages;
                    std::vector<std::string> expected;
                    if (thinking) expected.push_back(anthropic ? "thinking" : "reasoning");
                    expected.push_back(anthropic ? "text" : "message");
                    expected.push_back(anthropic ? "tool_use" : "function_call");
                    check(starts == expected && stops == expected && reasoning_done == (!anthropic && thinking ? 1 : 0),
                          "reasoning closes before subsequent output blocks");
                }
            }
        }
    }
    std::cout << "[agent] Thinking tags, reasoning streaming and non-streaming for all three protocols passed\n";
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

void test_cuda_sliding_window(CudaContext& ctx) {
    constexpr int seq = 192, window = 32, nq = 4, nkv = 2, splits = 8;
    std::mt19937_64 rng(343);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    for (const int hd : {64, 256}) {
        const int q_dim = nq * hd, kv_dim = nkv * hd;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const auto q = rand_f32({seq, q_dim}, rng, 0.5f);
        const auto k = rand_f32({seq, kv_dim}, rng, 0.5f);
        const auto v = rand_f32({seq, kv_dim}, rng, 0.25f);
        const auto dq = ctx.upload(q, false), dk = ctx.upload(k, false), dv = ctx.upload(v, false);
        const auto out = arena.alloc_tensor({seq, q_dim}, DType::F32);
        const auto decoded = arena.alloc_tensor({q_dim}, DType::F32);
        const auto partial = arena.alloc_tensor({nq, splits, hd}, DType::F32);
        const auto stats = arena.alloc_tensor({nq, splits, 2}, DType::F32);

        std::vector<double> reference(seq * q_dim);
        for (int pos = 0; pos < seq; ++pos) {
            const int first = std::max(0, pos + 1 - window);
            for (int h = 0; h < nq; ++h) {
                const int kh = h / (nq / nkv);
                const auto* query = q.data<float>() + pos * q_dim + h * hd;
                auto* dst = reference.data() + pos * q_dim + h * hd;
                float probability[window];
                float sum = 0.0f;
                for (int t = first; t <= pos; ++t) {
                    const auto* key = k.data<float>() + t * kv_dim + kh * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; ++i) dot += query[i] * key[i];
                    const float p = std::exp(dot * scale);
                    probability[t - first] = p;
                    sum += p;
                }
                for (int t = first; t <= pos; ++t) {
                    const auto* value = v.data<float>() + t * kv_dim + kh * hd;
                    for (int i = 0; i < hd; ++i) dst[i] += (probability[t - first] / sum) * value[i];
                }
            }
        }
        for (const auto dtype : {DType::F32, DType::F16}) {
            const auto cache = make_cuda_kv_view(arena, dtype, seq, nkv, hd);
            CudaKvPool pool;
            CudaKvPool::Table pages;
            pool.ensure(ctx, std::span(&cache, 1), pages, 0, seq);
            cuda_ops::batched_qk_norm_rope_kv_append_f32(dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(),
                nullptr, nullptr, nullptr, nullptr, seq, nq, nkv, hd, 0, 0, seq, 10000.0f, 1e-6f, true,
                ctx.stream(), q_dim, kv_dim, kv_dim, nullptr, dtype, cache);
            auto read = [&](const Tensor& t) {
                const auto host = ctx.download(t);
                return std::vector<float>(host.data<float>(), host.data<float>() + host.desc.numel());
            };

            cuda_ops::gqa_prefill_online_f32(dq.ptr<float>(), q_dim, cache, out.ptr<float>(), seq, nq, nkv,
                hd, 0, scale, ctx.stream(), window);
            expect_close(read(out), reference, 2.5e-4, 2e-3, "SWA paged online prefill");
            cuda_ops::gqa_prefill_causal_f32(dq.ptr<float>(), nullptr, nullptr, out.ptr<float>(), seq, nq, nkv,
                hd, 0, seq, scale, ctx.stream(), q_dim, dtype, cache, window);
            expect_close(read(out), reference, 8e-5, 1e-3, "SWA paged tiled prefill");
            for (const int length : {3, kKvPageTokens + 3, seq}) {
                cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>() + (length - 1) * q_dim, nullptr, nullptr,
                    decoded.ptr<float>(), partial.ptr<float>(), stats.ptr<float>(), nq, nkv, hd, nullptr,
                    length, seq, scale, splits, ctx.stream(), true, dtype, nullptr, nullptr, cache, window);
                auto first = reference.begin() + (length - 1) * q_dim;
                expect_close(read(decoded), {first, first + q_dim}, 8e-5, 1e-3, "SWA paged Split-KV decode");
            }
        }
        arena.reset(mark);
    }
    std::cout << "  [PASS] SWA window 32, 192 tokens, online/tiled prefill and Split-KV decode vs FP32\n";
}

void test_cuda_gemma_epilogues(CudaContext& ctx) {
    constexpr int rows = 3, width = 64;
    constexpr float eps = 1e-6f, cap = 0.5f;
    std::mt19937_64 rng(419);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const auto gate_up = rand_f32({rows, 2 * width}, rng, 2.0f);
    const auto w = rand_f32({width}, rng, 1.0f);
    const auto dgu = ctx.upload(gate_up, false), dw = ctx.upload(w, false);
    const auto out = arena.alloc_tensor({rows, width}, DType::F32);
    const auto q8 = arena.alloc_tensor({rows, width}, DType::Q8_1);
    std::vector<double> act(rows * width), post(rows * width);
    for (int r = 0; r < rows; ++r) {
        const auto* gate = gate_up.data<float>() + r * 2 * width;
        float sq = 0.0f;
        for (int i = 0; i < width; ++i) {
            const float x = gate[i];
            const float y = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265359f) *
                (x + 0.044715f * x * x * x))) * gate[width + i];
            act[r * width + i] = y;
            sq += y * y;
        }
        const float inv = 1.0f / std::sqrt(sq / width + eps);
        for (int i = 0; i < width; ++i) post[r * width + i] = act[r * width + i] * inv * w.data<float>()[i];
    }
    auto read = [&]() {
        const auto h = ctx.download(out);
        return std::vector<float>(h.data<float>(), h.data<float>() + rows * width);
    };
    cuda_ops::geglu_interleaved_f32(dgu.ptr<float>(), out.ptr<float>(), rows, width, ctx.stream());
    expect_close(read(), act, 2e-6, 1e-6, "GeGLU F32");
    cuda_ops::geglu_interleaved_q8_1(dgu.ptr<float>(), q8.ptr(), rows, width, ctx.stream());
    const auto packed = ctx.download(q8);
    for (int i = 0; i < rows * width; i += 32) {
        const auto& block = packed.data<BlockQ8_1>()[i / 32];
        const auto* bytes = reinterpret_cast<const int8_t*>(block.qs);
        float maximum = 0.0f;
        int lo = 0, hi = 0;
        for (int j = 0; j < 32; ++j) {
            maximum = std::max(maximum, static_cast<float>(std::abs(act[i + j])));
            (j < 16 ? lo : hi) += bytes[j];
            if (std::abs(block.d * bytes[j] - act[i + j]) > block.d * 0.5f + 3e-6f) {
                throw std::runtime_error("GeGLU Q8_1 quantization parity");
            }
        }
        if (std::abs(block.d - maximum / 127.0f) > 1e-7f ||
            std::abs(block.s_lo - block.d * lo) > 1e-6f ||
            std::abs(block.s_hi - block.d * hi) > 1e-6f ||
            std::abs(block.s_all - block.d * (lo + hi)) > 1e-6f) {
            throw std::runtime_error("GeGLU Q8_1 block metadata parity");
        }
    }
    const auto normalized = arena.alloc_tensor({rows, width}, DType::F32);
    cuda_ops::rmsnorm_f32(out.ptr<float>(), dw.ptr<float>(), normalized.ptr<float>(), rows, width, eps, ctx.stream());
    auto host = ctx.download(normalized);
    expect_close({host.data<float>(), host.data<float>() + rows * width}, post, 2e-6, 1e-6, "Gemma post RMSNorm");
    cuda_ops::softcap_f32(normalized.ptr<float>(), rows * width, cap, ctx.stream());
    for (auto& x : post) x = cap * std::tanh(x / cap);
    host = ctx.download(normalized);
    expect_close({host.data<float>(), host.data<float>() + rows * width}, post, 2e-6, 1e-6, "Gemma logit softcap");
    arena.reset(mark);
    std::cout << "  [PASS] GeGLU F32/Q8_1, post RMSNorm and logit softcap vs FP32\n";
}

void test_cuda_gemma4_gguf() {
    constexpr int layers = 4, hidden = 64, vocab = 64, nq = 4, window = 32, seq = 136, ple = 64;
    constexpr float cap = 0.5f;
    struct Fixture {
        std::filesystem::path path = "test-gemma4.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    const std::vector<bool> swa{true, false, true, false};
    const std::vector<uint32_t> ff{64, 96, 128, 96};
    std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("gemma4")}, {"gemma4.block_count", uint32_t{layers}},
        {"gemma4.embedding_length", uint32_t{hidden}}, {"gemma4.attention.head_count", uint32_t{nq}},
        {"gemma4.attention.head_count_kv", std::vector<uint32_t>{2, 1, 0, 0}},
        {"gemma4.attention.key_length", uint32_t{512}}, {"gemma4.attention.key_length_swa", uint32_t{256}},
        {"gemma4.attention.sliding_window", uint32_t{window}},
        {"gemma4.attention.sliding_window_pattern", swa}, {"gemma4.attention.shared_kv_layers", uint32_t{2}},
        {"gemma4.rope.freq_base", 1000000.0f}, {"gemma4.rope.freq_base_swa", 10000.0f},
        {"gemma4.feed_forward_length", ff}, {"gemma4.final_logit_softcapping", cap},
        {"gemma4.embedding_length_per_layer_input", uint32_t{ple}}
    };

    std::mt19937_64 rng(391);
    auto output_factors = make_ones_f32({256});
    output_factors.data<float>()[255] = 1.0e9f;
    HostTensor ple_tokens({vocab, layers * ple}, DType::Q5_K);
    for (int id = 0; id < vocab; ++id) {
        auto& block = ple_tokens.data<BlockQ5_K>()[id];
        block.d = fp32_to_fp16(0.00004f);
        block.dmin = fp32_to_fp16(0.0003f);
        for (auto& x : block.scales) x = static_cast<uint8_t>(rng());
        for (auto& x : block.qh) x = static_cast<uint8_t>(rng());
        for (auto& x : block.qs) x = static_cast<uint8_t>(rng());
    }
    HostTensor ple_model({layers * ple, hidden}, DType::BF16);
    const auto source = rand_f32({layers * ple, hidden}, rng, 0.035f);
    convert_f32_to_bf16(source.data<float>(), ple_model.data<uint16_t>(), source.desc.numel());
    std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", rand_f32({vocab, hidden}, rng, 0.1f)},
        {"output_norm.weight", make_ones_f32({hidden})},
        {"rope_freqs.weight", std::move(output_factors)},
        {"per_layer_token_embd.weight", std::move(ple_tokens)},
        {"per_layer_model_proj.weight", std::move(ple_model)},
        {"per_layer_proj_norm.weight", make_ones_f32({ple})}
    };
    for (int i = 0; i < layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const int hd = swa[i] ? 256 : 512, kv_heads = swa[i] ? 2 : 1;
        const int width = static_cast<int>(ff[i]);
        auto matrix = [&](const char* name, int r, int c) {
            weights.emplace_back(p + name, rand_f32({r, c}, rng, 0.035f));
        };
        auto norm = [&](const char* name, int n, float value = 1.0f) {
            auto weight = make_ones_f32({n});
            for (int j = 0; j < n; ++j) weight.data<float>()[j] = value;
            weights.emplace_back(p + name, std::move(weight));
        };
        norm("attn_norm.weight", hidden);
        norm("attn_q_norm.weight", hd);
        if (i < 2) {
            matrix("attn_k.weight", kv_heads * hd, hidden);
            if (swa[i]) matrix("attn_v.weight", kv_heads * hd, hidden);
            norm("attn_k_norm.weight", hd);
        }
        matrix("attn_q.weight", nq * hd, hidden);
        matrix("attn_output.weight", hidden, nq * hd);
        norm("post_attention_norm.weight", hidden, 0.35f);
        norm("ffn_norm.weight", hidden);
        matrix("ffn_gate.weight", width, hidden);
        matrix("ffn_up.weight", width, hidden);
        matrix("ffn_down.weight", hidden, width);
        norm("post_ffw_norm.weight", hidden, 0.25f);
        matrix("inp_gate.weight", ple, hidden);
        matrix("proj.weight", hidden, ple);
        norm("post_norm.weight", hidden, 0.13f);
        norm("layer_output_scale.weight", 1, 0.9f);
    }
    CudaContext ctx(0, 0, 64);
    FusionModel model(HybridModelConfig{});
    std::string err;
    if (!GgufFile::write_v3(fixture.path.string(), meta, weights, &err) ||
        !model.load_gguf(ctx, fixture.path.string(), &err, 160)) {
        throw std::runtime_error("Gemma 4 GGUF load: " + err);
    }
    const auto& loaded = model.dense_layers();
    if (model.config().embed_scale != 8.0f || model.config().final_logit_softcapping != cap ||
        loaded.size() != layers || loaded[0].cfg.head_dim != 256 ||
        loaded[0].sliding_window != window || loaded[0].rope_factors.data ||
        loaded[1].cfg.head_dim != 512 || loaded[1].v_proj.data || !loaded[1].rope_factors.data ||
        loaded[2].kv_source_layer != 0 || loaded[3].kv_source_layer != 1 ||
        loaded[2].k_proj.data || loaded[3].v_proj.data || !loaded[3].ffn_post_norm_w.data ||
        !loaded[3].ple_gate.data || !loaded[3].ple_post_norm.data || loaded[3].output_scale != 0.9f) {
        throw std::runtime_error("Gemma 4 GGUF layer wiring");
    }

    std::vector<int32_t> prompt(seq);
    for (auto& token : prompt) token = static_cast<int32_t>(rng() % vocab);
    const int prefill_batch = ctx.prefill_batch();
    // Small chunks keep the BF16 projection on the FP32-input GEMV path.
    ctx.set_prefill_batch(8);
    const auto batched = model.forward_prefill_cuda(ctx, prompt, 0, false);
    ctx.set_prefill_batch(prefill_batch);
    const size_t expected_kv_bytes = 2 * 2 * kKvPageTokens * (2 * 256 + 512) * sizeof(float);
    if (model.cuda_cache_memory().kv_bytes != expected_kv_bytes) {
        throw std::runtime_error("Gemma 4 shared layers allocated their own KV pages");
    }
    model.reset_kv_and_state(ctx);
    std::vector<float> sequential;
    for (int i = 0; i < seq; ++i) sequential = model.forward_token_cuda(ctx, prompt[i], i, false);
    double squared_error = 0.0, squared_ref = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const double diff = batched[i] - sequential[i];
        squared_error += diff * diff;
        squared_ref += double(sequential[i]) * sequential[i];
        if (std::abs(batched[i]) > cap) throw std::runtime_error("Gemma 4 logits exceed softcap");
    }
    const double rel_l2 = std::sqrt(squared_error / squared_ref);
    if (!(rel_l2 < 1e-5)) throw std::runtime_error("Gemma 4 batched/sequential rel_l2: " + std::to_string(rel_l2));
    model.reset_kv_and_state(ctx);
    const auto optimized = model.forward_prefill_cuda(ctx, prompt, 0);
    // Batched cuBLAS uses FP16 activations for the BF16 PLE projection.
    expect_close(optimized, {batched.begin(), batched.end()}, 3e-4, 3e-4, "Gemma 4 optimized prefill");

    const auto eager = model.generate_cuda(ctx, prompt, 16, true, false);
    const auto graph = model.generate_cuda(ctx, prompt, 16, true, true);
    if (eager != graph || graph.size() != seq + 16) {
        throw std::runtime_error("Gemma 4 CUDA Graph/eager tokens differ");
    }
    std::cout << "  [PASS] Gemma 4 four-layer GGUF, prefill rel_l2=" << rel_l2
              << ", Graph/eager 16/16 tokens, shared KV bytes=" << expected_kv_bytes << "\n";
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
            mmq(qx.ptr(), dw.ptr(), dy.ptr<float>(), rows, N, K, ctx.stream(), ctx.cost_hardware());
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
    const auto old_kv_dtype = ctx.kv_dtype();
    for (int hd : {64, 128, 256}) {
        const int prefix = kKvPageTokens - 3, rows = 23, total = prefix + rows;
        const int stride = nq * hd + 64;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const auto q = upload_random({total, stride}, 0.5f);
        const auto k = upload_random({total, nkv * hd}, 0.5f);
        const auto v = upload_random({total, nkv * hd}, 0.25f);
        const auto reference_out = arena.alloc_tensor({rows, nq * hd}, DType::F32);
        const auto online_out = arena.alloc_tensor({rows, nq * hd}, DType::F32);
        std::vector<double> f16;
        for (auto dtype : {DType::F16, DType::F32, DType::Q8_0}) {
            ctx.set_kv_dtype(dtype);
            const auto persistent_mark = ctx.weight_arena().mark();
            {
                const auto cache = make_cuda_kv_view(ctx.weight_arena(), dtype, 262144, nkv, hd);
                CudaKvPool pool;
                CudaKvPool::Table pages;
                auto append = [&](int from, int count) {
                    pool.ensure(ctx, std::span(&cache, 1), pages, from, from + count);
                    cuda_ops::batched_qk_norm_rope_kv_append_f32(
                        q.ptr<float>() + from * stride, k.ptr<float>() + from * nkv * hd, v.ptr<float>() + from * nkv * hd,
                        nullptr, nullptr, nullptr, nullptr, count, nq, nkv, hd, 0, from, total, 10000.0f, 1e-6f,
                        true, ctx.stream(), stride, nkv * hd, nkv * hd, nullptr, dtype, cache);
                };
                append(0, prefix);
                if (pool.pages() != 1) throw std::runtime_error("KV must allocate only the first page");
                if (dtype == DType::F16) {
                    cuda_ops::gqa_prefill_causal_f32(q.ptr<float>(), nullptr, nullptr, reference_out.ptr<float>(),
                        rows, nq, nkv, hd, 0, total, scale, ctx.stream(), stride, dtype, cache);
                    cuda_ops::gqa_prefill_online_f32(q.ptr<float>(), stride, cache, online_out.ptr<float>(), rows,
                        nq, nkv, hd, 0, scale, ctx.stream());
                    const auto first_page = read(reference_out);
                    expect_close(read(online_out), {first_page.begin(), first_page.end()}, 2e-4, 1e-3,
                                 "Online attention against the short FP16 path");
                }
                append(prefix, rows);
                const auto bytes = 4 * kKvPageTokens * nkv * hd * dtype_type_size(dtype) / dtype_block_size(dtype);
                if (pool.pages() != 2 || pool.bytes() != bytes) throw std::runtime_error("KV page payload size mismatch");
                const float* query = q.ptr<float>() + prefix * stride;
                cuda_ops::gqa_prefill_causal_f32(query, nullptr, nullptr, reference_out.ptr<float>(), rows,
                    nq, nkv, hd, prefix, total, scale, ctx.stream(), stride, dtype, cache);
                cuda_ops::gqa_prefill_online_f32(query, stride, cache, online_out.ptr<float>(), rows,
                    nq, nkv, hd, prefix, scale, ctx.stream());
                const auto reference = read(reference_out);
                const auto online = read(online_out);
                expect_close(online, {reference.begin(), reference.end()}, 2e-4, 1e-3, "Paged online attention");
                if (dtype == DType::F16) f16.assign(reference.begin(), reference.end());
                if (dtype == DType::Q8_0) expect_close(online, f16, 1.5e-3, 1e-2, "Q8 KV attention accuracy");
            }
            ctx.weight_arena().reset(persistent_mark);
        }
        arena.reset(mark);
    }
    ctx.set_kv_dtype(old_kv_dtype);
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
    const auto old_kv_dtype = ctx.kv_dtype();
    const auto mark = ctx.weight_arena().mark();
    ctx.set_kv_dtype(DType::Q8_0);
    {
        CudaKvPool pool;
        CudaKvPool::Table pages;
        const auto cache = make_cuda_kv_view(ctx.weight_arena(), ctx.kv_dtype(), max_seq, nkv, hd);
        pool.ensure(ctx, std::span(&cache, 1), pages, pos, max_seq);
        ctx.upload_into(hq, dq);
        ctx.upload_into(hk, dk);
        cuda_ops::fused_qk_norm_rope_kv_append_f32(dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(),
            dqn.ptr<float>(), dkn.ptr<float>(), nullptr, nullptr, nq, nkv, hd, rot_dim, nullptr, pos, max_seq,
            1000000.0f, 1e-6f, true, ctx.stream(), nullptr, DType::Q8_0, cache);
        constexpr int splits = 128;
        auto parts = ctx.scratch_arena().alloc_tensor({nq, splits, hd}, DType::F32);
        auto ml = ctx.scratch_arena().alloc_tensor({nq, splits, 2}, DType::F32);
        cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>(), nullptr, nullptr, out_s16.ptr<float>(),
            parts.ptr<float>(), ml.ptr<float>(), nq, nkv, hd, nullptr, max_seq, max_seq, scale,
            splits, ctx.stream(), true, DType::Q8_0, nullptr, nullptr, cache);
        const auto actual = ctx.download(out_s16);
        expect_close({actual.data<float>(), actual.data<float>() + nq * hd}, ref,
                     1e-7, 1e-2, "Paged Q8 attention with 128 splits at 64K");
    }
    ctx.set_kv_dtype(old_kv_dtype);
    ctx.weight_arena().reset(mark);
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
        GenerationStats graph_stats;
        auto out_graph = m.generate_cuda(ctx, {1, 10, 20}, 4, true, true, &graph_stats);
        if (out_eager != out_graph || !graph_stats.used_cuda_graph) {
            throw std::runtime_error("CUDA Graph token generation diverged from eager!");
        }
        GenerationOptions sampling_options;
        sampling_options.sampling.temperature = 0.7f;
        sampling_options.warn_on_graph_fallback = true;
        (void)m.generate_cuda(ctx, {1, 10, 20}, 4, true, true, &graph_stats, false, 0, -1, sampling_options);
        if (graph_stats.used_cuda_graph) throw std::runtime_error("Stochastic decoding must fall back to eager");
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

void test_vulkan_continued_turn() {
    VkContext vk(0, 32, 32);
    HybridModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.num_layers = 1;
    cfg.dense.hidden_size = cfg.dense.intermediate_size = 64;
    cfg.dense.num_attention_heads = 2;
    cfg.dense.num_key_value_heads = 1;
    cfg.dense.head_dim = 32;
    cfg.dense.max_seq_len = 32;
    constexpr uint64_t seed = 781;
    const std::vector<int32_t> prompt{1, 5, 9};
    FusionModel fresh(cfg);
    fresh.init_synthetic_vulkan(vk, seed);
    const auto first = fresh.generate_vulkan(vk, prompt, 1);
    const int next_pos = static_cast<int>(first.size());
    const int32_t next_user_token = (first.back() + 1) % cfg.vocab_size;
    for (int i = 0; i < next_pos; ++i) (void)fresh.forward_token_vulkan(vk, first[i], i);
    const auto expected = fresh.forward_token_vulkan(vk, next_user_token, next_pos);
    for (const auto eos : {int32_t{-1}, first.back()}) {
        FusionModel continued(cfg);
        continued.init_synthetic_vulkan(vk, seed);
        const auto answer = continued.generate_vulkan(vk, prompt, eos < 0 ? 1 : 4, nullptr, true, 0, eos);
        if (answer != first || continued.forward_token_vulkan(vk, next_user_token, next_pos) != expected)
            throw std::runtime_error("Vulkan must cache the final token before continuing a turn");
    }
    std::cout << "  [PASS] Vulkan continued-turn logits after EOS and max steps\n";
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

void test_stage7b_images_and_protocols() {
    using namespace agent;
    const std::string ppm = "UDYKMSAxCjI1NQr/AAA=";
    const std::vector<uint8_t> colors{255,0,0,0,255,0,0,0,255,255,255,255};
    for (const auto* b64 : {
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAGUlEQVR4AQEOAPH/AP8AAAD/AAAAAP////8f7gX73t3sKwAAAABJRU5ErkJggg==",
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFElEQVR4AWP4z8DAAMIM/////w8AH+4F+w7A5KwAAAAASUVORK5CYII="}) {
        const auto decoded = decode_image_base64(b64);
        if (decoded.width != 2 || decoded.height != 2 || decoded.rgb != colors)
            throw std::runtime_error("Stored/fixed PNG decoding");
    }
    const auto dynamic = decode_image_base64("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAT0lEQVR42u2VuQ0AQAjDvErmYn8pY9wUviolFYI8BkiSu7u2ZfP3GXsf9j3Y/8LWA1tvbD9h+xU7D9h5w84zdl+wvhwPxoPxYDwYD37w4AFHlCw9N3VFOgAAAABJRU5ErkJggg==");
    if (dynamic.width != 32 || dynamic.height != 32) throw std::runtime_error("Dynamic PNG geometry");
    constexpr std::array<uint8_t, 4> tones{0, 33, 91, 232};
    for (int y = 0; y < 32; ++y) for (int x = 0; x < 96; ++x) {
        if (dynamic.rgb[y * 96 + x] != tones[((y * 97 + x + 1) / 4) % 4])
            throw std::runtime_error("Dynamic PNG pixels");
    }
    std::vector<uint8_t> bmp(58);
    bmp[0] = 'B'; bmp[1] = 'M'; bmp[2] = 58; bmp[10] = 54; bmp[14] = 40;
    bmp[18] = 1; bmp[22] = 1; bmp[26] = 1; bmp[28] = 24; bmp[56] = 255;
    if (decode_image(bmp).rgb != std::vector<uint8_t>{255,0,0} ||
        decode_image_base64("UDUKMSAxCjI1NQoR").rgb != std::vector<uint8_t>{17,17,17})
        throw std::runtime_error("BMP or PGM decoding");

    const std::string uri = "data:image/x-portable-pixmap;base64," + ppm;
    BpeTokenizer tok;
    tok.init_byte_fallback();
    const std::vector<std::pair<Protocol, std::string>> requests{
        {Protocol::chat, R"({"messages":[{"role":"user","content":[{"type":"text","text":"literal <image> before"},{"type":"image_url","image_url":{"url":")" + uri + R"("}},{"type":"text","text":"after"}]}]})"},
        {Protocol::responses, R"({"input":[{"role":"user","content":[{"type":"input_image","image_url":")" + uri + R"("}]}]})"},
        {Protocol::messages, R"({"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/x-portable-pixmap","data":")" + ppm + R"("}}]}]})"},
        {Protocol::messages, R"({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call","name":"draw","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call","content":[{"type":"text","text":"literal <image> before"},{"type":"image","source":{"type":"base64","media_type":"image/x-portable-pixmap","data":")" + ppm + R"("}},{"type":"text","text":"after"}]}]}]})"}
    };
    auto bad_request = [](const auto& obj, Protocol protocol, bool loaded) {
        try { (void)parse_request(obj, protocol, loaded); }
        catch (const std::invalid_argument&) { return; } // The HTTP dispatcher returns 400 for this exception.
        throw std::runtime_error("Invalid image request was accepted");
    };
    for (const auto& [protocol, json] : requests) {
        const auto obj = JsonValue::parse(json).as_object();
        const auto parsed = parse_request(obj, protocol, true);
        const auto prompt = render_prompt(parsed, tok);
        if (prompt.images.size() != 1 || prompt.images[0]->rgb != std::vector<uint8_t>{255,0,0} ||
            prompt.image_offsets.size() != 1 ||
            prompt.text.substr(prompt.image_offsets[0], image_placeholder.size()) != image_placeholder ||
            (json.find("literal <image>") != std::string::npos &&
             (prompt.text.find(image_placeholder) >= prompt.image_offsets[0] ||
              prompt.text.substr(prompt.image_offsets[0] - 6, 18) != "before<image>after")))
            throw std::runtime_error("Image request or prompt order");
        bad_request(obj, protocol, false);
        auto broken = json;
        broken.replace(broken.find(ppm), ppm.size(), "!!!!");
        bad_request(JsonValue::parse(broken).as_object(), protocol, true);
        broken = json;
        broken.replace(broken.find(ppm), ppm.size(), "QUJDRA==");
        bad_request(JsonValue::parse(broken).as_object(), protocol, true);
    }
    const std::vector<uint8_t> cr_raster{'P','6','\n','1',' ','1','\n','2','5','5','\r',0x0a,20,30};
    if (decode_image(cr_raster).rgb != std::vector<uint8_t>{10,20,30})
        throw std::runtime_error("Binary PPM CR delimiter must preserve first LF pixel byte");
    std::cout << "  [PASS] PNG, BMP/PPM/PGM, CR raster, literal image marker and Anthropic tool-result image\n";
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_stage7b_cuda_vision() {
    CudaContext ctx(0, 24, 32);
    auto& scratch = ctx.scratch_arena();
    std::mt19937_64 rng(712);
    auto compare = [&](const Tensor& result, const std::vector<double>& reference, const char* name) {
        const auto host = ctx.download(result);
        const std::vector<float> values(host.data<float>(), host.data<float>() + host.desc.numel());
        double diff = 0, base = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            const double d = values[i] - reference[i];
            diff += d * d;
            base += reference[i] * reference[i];
        }
        const double rel = std::sqrt(diff / base);
        if (!(rel < 1e-5)) throw std::runtime_error(std::string(name) + " relative error: " + std::to_string(rel));
        std::cout << "  [PASS] " << name << " rel_l2=" << rel << '\n';
    };
    {
        constexpr int rows = 6, dim = 64;
        const auto q = rand_f32({rows, dim}, rng), k = rand_f32({rows, dim}, rng), v = rand_f32({rows, dim}, rng);
        auto dq = ctx.upload(q, false), dk = ctx.upload(k, false), dv = ctx.upload(v, false);
        auto output = scratch.alloc_tensor({rows, dim}, DType::F32);
        const float scale = 1.f / std::sqrt(static_cast<float>(dim));
        cuda_ops::gqa_prefill_causal_f32(dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(), output.ptr<float>(),
            rows, 1, 1, dim, 0, rows, scale, ctx.stream(), 0, DType::F32, {}, 0, false);
        std::vector<double> ref(rows * dim);
        for (int i = 0; i < rows; ++i) {
            double sum = 0;
            for (int j = 0; j < rows; ++j) {
                double score = 0;
                for (int d = 0; d < dim; ++d) score += double(q.data<float>()[i*dim+d]) * k.data<float>()[j*dim+d];
                const double p = std::exp(score * scale);
                sum += p;
                for (int d = 0; d < dim; ++d) ref[i*dim+d] += p * v.data<float>()[j*dim+d];
            }
            for (int d = 0; d < dim; ++d) ref[i*dim+d] /= sum;
        }
        compare(output, ref, "Bidirectional FA2 vs FP64");
        auto w = make_ones_f32({dim}), b = rand_f32({dim}, rng);
        auto dw = ctx.upload(w, false), db = ctx.upload(b, false);
        cuda_ops::layernorm_f32(dq.ptr<float>(), dw.ptr<float>(), db.ptr<float>(), output.ptr<float>(), rows, dim, 1e-5f, ctx.stream());
        for (int row = 0; row < rows; ++row) {
            const auto* x = q.data<float>() + row * dim;
            double sum = 0, sq = 0;
            for (int d = 0; d < dim; ++d) sum += x[d];
            const double mean = sum / dim;
            for (int d = 0; d < dim; ++d) sq += (x[d] - mean) * (x[d] - mean);
            for (int d = 0; d < dim; ++d) ref[row*dim+d] = (x[d] - mean) / std::sqrt(sq / dim + 1e-5) + b.data<float>()[d];
        }
        compare(output, ref, "LayerNorm vs FP64");
        constexpr int width = 3, quarter = dim / 4;
        for (const bool qwen : {false, true}) {
            auto d_rope = ctx.upload(q, false);
            cuda_ops::rope_2d_f32(d_rope.ptr<float>(), 2, width, 1, dim, 10000.f, ctx.stream(), qwen);
            for (int row = 0; row < rows; ++row) for (int axis = 0; axis < 2; ++axis) for (int d = 0; d < quarter; ++d) {
                const int lo = row*dim + (qwen ? axis*quarter : axis*2*quarter) + d;
                const int hi = lo + (qwen ? 2*quarter : quarter);
                const double angle = ((axis != 0) != qwen ? row / width : row % width) /
                                     std::pow(10000.0, double(d)/quarter);
                ref[lo] = q.data<float>()[lo]*std::cos(angle) - q.data<float>()[hi]*std::sin(angle);
                ref[hi] = q.data<float>()[lo]*std::sin(angle) + q.data<float>()[hi]*std::cos(angle);
            }
            compare(d_rope, ref, qwen ? "2D RoPE vision layout vs FP64" : "2D RoPE split spatial axes vs FP64");
        }
    }
    scratch.reset();
    constexpr int H = 64;
    struct Fixture {
        std::string path = "stage7b-test-mmproj.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("clip")}, {"clip.vision.patch_size", uint32_t{1}},
        {"clip.vision.image_size", uint32_t{4}}, {"clip.vision.embedding_length", uint32_t{H}},
        {"clip.vision.attention.head_count", uint32_t{1}}, {"clip.vision.block_count", uint32_t{1}},
        {"clip.vision.spatial_merge_size", uint32_t{2}}, {"clip.vision.rope.freq_base", 10000.0f}
    };
    const std::vector<std::pair<std::string, HostTensor>> tensors{
        {"v.patch_embd.weight", rand_f32({H,3,1,1}, rng)}, {"v.patch_embd.bias", HostTensor({H}, DType::F32)},
        {"v.position_embd.weight", rand_f32({4,H}, rng, 0.02f)},
        {"v.blk.0.ln1.weight", make_ones_f32({H})}, {"v.blk.0.ln2.weight", make_ones_f32({H})},
        {"v.blk.0.attn_qkv.weight", rand_f32({3*H,H}, rng, 0.05f)},
        {"v.blk.0.attn_out.weight", rand_f32({H,H}, rng, 0.05f)},
        {"v.blk.0.ffn_up.weight", rand_f32({2*H,H}, rng, 0.05f)},
        {"v.blk.0.ffn_down.weight", rand_f32({H,2*H}, rng, 0.05f)},
        {"mm.0.weight", rand_f32({H,4*H}, rng, 0.08f)}, {"mm.2.weight", rand_f32({H,H}, rng, 0.12f)}
    };
    std::string error;
    if (!GgufFile::write_v3(fixture.path, meta, tensors, &error)) throw std::runtime_error(error);
    VisionEncoder vision(ctx, fixture.path, H);
    if (vision.info().layers != 1 || vision.info().heads != 1 || vision.info().spatial_merge != 2)
        throw std::runtime_error("mmproj geometry roundtrip");
    const auto red = decode_image_base64("UDYKMSAxCjI1NQr/AAA=");
    const auto green = decode_image_base64("UDYKMSAxCjI1NQoA/wA=");
    const Image* images[] = {&red};
    BpeTokenizer tokenizer;
    tokenizer.init_byte_fallback();
    const auto a = encode_vision_prompt(ctx, vision, tokenizer, "A<image>Z", images, 64);
    images[0] = &green;
    const auto b = encode_vision_prompt(ctx, vision, tokenizer, "A<image>Z", images, 64);
    if (a.tokens != b.tokens || a.inputs_embeds[0].rows.shape[0] != 4) throw std::runtime_error("Image placeholder geometry");
    {
        agent::AgentRequest request;
        request.messages.push_back({agent::Role::user, {std::string("literal <image> "), red, std::string("end")}});
        const auto rendered = agent::render_prompt(request, tokenizer);
        const auto with_literal = encode_vision_prompt(ctx, vision, tokenizer, rendered.text, rendered.images,
                                                       rendered.image_offsets, 64);
        const auto plain = tokenizer.encode("user: literal <image> ");
        if (with_literal.inputs_embeds.size() != 1 || with_literal.inputs_embeds[0].start != static_cast<int>(plain.size()) ||
            !std::equal(plain.begin(), plain.end(), with_literal.tokens.begin()))
            throw std::runtime_error("Literal <image> must remain text before the CUDA image splice");
    }
    HybridModelConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size();
    cfg.num_layers = 2;
    cfg.dense.hidden_size = H;
    cfg.dense.intermediate_size = 2 * H;
    cfg.dense.num_attention_heads = 2;
    cfg.dense.num_key_value_heads = 1;
    cfg.dense.head_dim = 32;
    cfg.dense.max_seq_len = 64;
    FusionModel model(cfg);
    model.set_num_slots(2);
    model.init_synthetic(ctx, 71);
    ctx.set_prefill_batch(3);
    model.reset_kv_and_state(ctx);
    const auto logits_a = model.forward_prefill_cuda(ctx, a.tokens, 0, true, a.inputs_embeds);
    const auto logits_b = model.forward_prefill_cuda(ctx, b.tokens, 0, true, b.inputs_embeds);
    if (logits_a == logits_b) throw std::runtime_error("Different images produced identical logits");
    GenerationStats stats;
    auto generate = [&](const VisionPrompt& p) {
        GenerationOptions op;
        op.reuse_prefix = true;
        op.inputs_embeds = p.inputs_embeds;
        auto ids = model.generate_cuda(ctx, p.tokens, 6, true, true, &stats, false, 0, -1, op);
        if (stats.prefilled_tokens != static_cast<int>(p.tokens.size())) throw std::runtime_error("Vision prefix reused text KV");
        return ids;
    };
    model.prefill_cached_cuda(ctx, a.tokens);
    auto text_alternative = a.tokens;
    text_alternative[0] = 35;
    model.prefill_cached_cuda(ctx, text_alternative);
    const auto out_a = generate(a), out_b = generate(b);
    if (out_a != generate(a) || out_a == out_b) throw std::runtime_error("Image generation or slot isolation");
    int prefilled = 0;
    model.prefill_cached_cuda(ctx, a.tokens, true, &prefilled);
    if (prefilled != 1) throw std::runtime_error("Vision request invalidated unrelated text slot");
    std::cout << "  [PASS] mmproj vision encode and chunked CUDA splice, distinct image tokens, isolated prefix slots\n";
}
#endif

void test_stage7a_json() {
    std::cout << "Testing incremental JSON grammar and constrained sampling...\n";
    auto check = [](bool passed, const char* error) {
        if (!passed) throw std::runtime_error(error);
    };
    JsonGrammarState state;
    const std::vector<std::string> pieces = {
        " \r\n{\"utf8\":\"\xf0\x9f", "\x98\x83", "\\ud83d", "\\ude03",
        "\",\"list\":[-0.3e", "+2,true,false,null,{},[]]}\t"
    };
    std::string valid;
    for (const auto& piece : pieces) {
        check(!state.complete() && state.consume(piece), "A valid JSON token split was rejected");
        valid += piece;
    }
    check(state.complete() && JsonValue::parse(valid).is_object(), "Completed JSON object");
    for (const std::string_view invalid : {"[1,]", "{\"a\":}", "{\"a\":1,}", "00", "1 2",
                                           "\"\\udc00\"", "\"\\ud800x\"", "\"\xed\xa0\x80\"", "\vnull"}) {
        JsonGrammarState fresh;
        check(!fresh.consume(invalid) && fresh.consume("null") && fresh.complete(),
              "Invalid JSON token must be rejected without changing state");
    }
    for (const std::string_view partial : {"1.", "1e-", "-", "[true", "\"\\ud800\\u", "\"\xe2"}) {
        JsonGrammarState prefix;
        check(prefix.consume(partial) && !prefix.complete(), "Incomplete JSON prefix");
    }

    BpeTokenizer tok;
    tok.init_byte_fallback();
    const JsonConstraint constraint(tok, tok.vocab_size(), true);
    const auto invalid = tok.encode("]", false).front();
    std::vector<float> logits(tok.vocab_size());
    std::uniform_real_distribution<float> score(-20, 20);
    for (uint64_t seed = 0; seed < 48; ++seed) {
        auto grammar = constraint.initial_state();
        std::mt19937_64 rng(seed);
        SamplingParams params;
        params.temperature = seed % 2 ? 0.7f : 0.0f;
        std::vector<int32_t> ids;
        for (int budget = 32; budget > 0 && !grammar.sealed(); --budget) {
            for (auto& logit : logits) logit = score(rng);
            logits[tok.eos_id()] = logits[invalid] = std::numeric_limits<float>::infinity();
            constraint.mask(logits, grammar, budget);
            const auto token = Sampler::sample(logits.data(), tok.vocab_size(), ids, params, rng);
            check(constraint.accept(grammar, token), "Sampler chose a masked JSON token");
            ids.push_back(token);
        }
        check(grammar.complete() && JsonValue::parse(tok.decode(ids)).is_object(), "Constrained sample must parse");
    }

    using agent::Protocol;
    const auto parse = [](std::string_view json, Protocol protocol) {
        return agent::parse_request(JsonValue::parse(std::string(json)).as_object(), protocol).generation.json;
    };
    check(parse(R"({"messages":[{"role":"user","content":"test"}],"response_format":{"type":"json_object"}})",
                Protocol::chat) == JsonRoot::object, "Chat JSON object format");
    check(parse(R"({"input":"test","response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}})",
                Protocol::responses) == JsonRoot::object, "Responses JSON schema format");
    check(parse(R"({"input":"test","text":{"format":{"type":"json_schema","schema":{"type":"array"}}}})",
                Protocol::responses) == JsonRoot::array, "Responses native text format");
}

void test_review_input_validation() {
    using agent::Protocol;
    auto reject = [](auto operation) {
        try { operation(); }
        catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("Invalid input was accepted");
    };
    for (const bool object : {false, true}) {
        std::string opening;
        for (int depth = 0; depth < 128; ++depth) opening += object ? "{\"k\":" : "[";
        const std::string closing(128, object ? '}' : ']');
        const auto valid = opening + "null" + closing;
        (void)JsonValue::parse(valid);
        const std::string nested = opening + (object ? "{}" : "[]") + closing;
        bool exceeds = false;
        try { (void)JsonValue::parse(nested); }
        catch (const std::runtime_error& e) { exceeds = std::string_view(e.what()) == "JSON nesting depth exceeded"; }
        JsonGrammarState grammar;
        if (!exceeds || grammar.consume(nested) || !grammar.consume(valid) || !grammar.complete())
            throw std::runtime_error("JSON nesting limit mismatch");
    }
    auto parse = [](const char* text, Protocol protocol) {
        return agent::parse_request(JsonValue::parse(text).as_object(), protocol);
    };
    const auto request = JsonValue::parse(R"({"model":"gpt-4o","n":1,"logprobs":false,"top_logprobs":null,"presence_penalty":0,"frequency_penalty":0,"logit_bias":{},"messages":[{"role":"user","content":"test"}]})").as_object();
    (void)agent::parse_request(request, Protocol::chat);
    for (const auto& [key, value] : JsonObject{{"n", 2}, {"logprobs", true}, {"top_logprobs", 0},
                                              {"presence_penalty", 0.5}, {"frequency_penalty", -0.5},
                                              {"logit_bias", JsonObject{{"42", 1}}}}) {
        auto unsupported = request;
        unsupported[key] = value;
        bool rejected = false;
        try { (void)agent::parse_request(unsupported, Protocol::chat); }
        catch (const std::invalid_argument& e) { rejected = std::string_view(e.what()).find(key) != std::string_view::npos; }
        if (!rejected) throw std::runtime_error("Expected a named bad-request error for " + key);
    }
    HybridModelConfig health_cfg;
    health_cfg.dense.max_seq_len = 4096;
    if (http::health_body("local", health_cfg, 1, {}).at("max_seq").as_int() != 4096)
        throw std::runtime_error("/health must include the model context");
    BpeTokenizer http_tokenizer;
    http_tokenizer.init_byte_fallback();
    try {
        (void)http::encode_prompt(http_tokenizer, std::string(4199, 'x'), 4096);
        throw std::runtime_error("HTTP accepted an oversized prompt");
    } catch (const std::invalid_argument& e) {
        if (std::string_view(e.what()) != "Prompt (4200 tokens) exceeds model context (4096)")
            throw std::runtime_error("HTTP context error must report actual token counts");
    }
    reject([&] { parse(R"({"input":"test","text":{"format":{"type":"json_schema","schema":{"type":"object","required":["x"]}}}})", Protocol::responses); });
    reject([&] { parse(R"({"messages":[{"role":"user","content":"test"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"demo"}}],"tool_choice":"auto"})", Protocol::chat); });
    const auto omitted = parse(R"({"messages":[{"role":"user","content":"test"}]})", Protocol::chat);
    const auto capped = parse(R"({"messages":[{"role":"user","content":"test"}],"max_completion_tokens":128})", Protocol::chat);
    if (agent::generation_limit(omitted.generation, 2048, 127) != 1921 ||
        agent::generation_limit(capped.generation, 2048, 127) != 128 ||
        agent::generation_limit(capped.generation, 200, 127) != 73)
        throw std::runtime_error("HTTP output limits must use remaining context and respect explicit caps");
    reject([&] { parse(R"({"messages":[{"role":"user","content":"test"}],"max_completion_tokens":-1})", Protocol::chat); });
    const auto zero = parse(R"({"messages":[{"role":"user","content":"test"}],"max_completion_tokens":0})", Protocol::chat);
    if (agent::generation_limit(zero.generation, 2048, 127) != 0)
        throw std::runtime_error("An explicit zero output limit must be preserved");
    auto params = parse(R"({"messages":[{"role":"user","content":"test"}],"response_format":{"type":"json_object"},"max_completion_tokens":0})", Protocol::chat).generation;
    for (const bool stream : {false, true}) {
        params.stream = stream;
        params.max_tokens = 0;
        reject([&] { (void)agent::generation_limit(params, 32, 10); });
        params.max_tokens = 5;
        reject([&] { (void)agent::generation_limit(params, 32, 32); });
        if (agent::generation_limit(params, 32, 31) != 1) throw std::runtime_error("Effective JSON output limit");
    }
    JsonGrammarState array(JsonRoot::array), integer(JsonRoot::integer);
    if (array.consume("{}") || !array.consume("[]") || !array.complete() ||
        integer.consume("1.5") || !integer.consume("-42") || !integer.complete())
        throw std::runtime_error("JSON schema root constraint");
    BpeTokenizer tok;
    tok.init_byte_fallback();
    JsonConstraint constraint(tok, tok.vocab_size(), JsonRoot::array);
    std::vector<float> logits(tok.vocab_size() - 1);
    reject([&] { constraint.mask(logits, constraint.initial_state(), 2); });
    logits.resize(tok.vocab_size(), 0.0f);
    auto grammar = constraint.initial_state();
    constraint.mask(logits, grammar, 2);
    if (logits[tok.encode("[", false).front()] != 0.0f || std::isfinite(logits[tok.encode("{", false).front()]))
        throw std::runtime_error("JSON array mask admits a different root");

    std::optional<Protocol> route;
    const auto accepted = http::parse_request_head("POST /v1/messages HTTP/1.1\r\nContent-Length: 16777216", route);
    if (accepted.content_length != 16777216 || route != Protocol::messages) throw std::runtime_error("HTTP body limit boundary");
    for (const auto* length : {"16777217", "18446744073709551616"}) {
        for (const auto* path : {"/v1/messages", "/v1/responses"}) {
            const bool anthropic = std::string_view(path) == "/v1/messages";
            bool oversized = false;
            try {
                (void)http::parse_request_head(std::string("POST ") + path + "?mode=test HTTP/1.1\r\nContent-Length: " + length, route);
            } catch (const http::PayloadTooLarge& e) {
                const auto body = http::error_body(e.status, e.what(), route);
                const auto& detail = body.at("error").as_object();
                oversized = std::string_view(e.status) == "413 Payload Too Large" &&
                    route == (anthropic ? Protocol::messages : Protocol::responses) &&
                    body.contains("type") == anthropic &&
                    detail.at("type").as_string() == (anthropic ? "request_too_large" : "invalid_request_error");
            }
            if (!oversized) throw std::runtime_error("HTTP 413 must preserve the route's error envelope");
        }
    }
    for (const auto* option : {"--spec-ngram", "--draft-tokens", "--max-seq", "--steps", "--port"}) {
        for (const auto* value : {"0junk", "4junk", "2147483648"})
            reject([&] { (void)cli::parse_integer(option, value); });
        if (cli::parse_integer(option, "4") != 4) throw std::runtime_error("CLI integer parser rejected a valid value");
    }
    std::cout << "  [PASS] JSON nesting, schema/root and tool choice, JSON output budget, HTTP 413, and strict CLI integers\n";
}

void test_review_gguf_validation() {
    struct Fixture {
        std::string path = "test-review-gguf.gguf";
        ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    } fixture;
    std::string error;
    for (const auto dtype : {DType::F16, DType::BF16, DType::F32, DType::I32, DType::Q6_K}) {
        const int cols = dtype == DType::Q6_K ? 256 : 4;
        const HostTensor weight({cols}, dtype);
        GgufFile gguf;
        if (!GgufFile::write_v3(fixture.path, {}, {{"v", weight}, {"next", weight}}, &error) ||
            !gguf.load(fixture.path, &error)) throw std::runtime_error(error);
        gguf = {};
        constexpr int first_tensor_offset = 24 + 8 + 1 + 4 + 8 + 4;
        {
            std::fstream file(fixture.path, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(first_tensor_offset);
            file.put(dtype == DType::F32 || dtype == DType::I32 ? '\x02' : '\x01');
            if (!file) throw std::runtime_error("Could not construct misaligned GGUF fixture");
        }
        if (gguf.load(fixture.path, &error) || !error.starts_with("Misaligned GGUF tensor offset"))
            throw std::runtime_error("GGUF must reject misaligned data");
    }
    {
        const std::map<std::string, GgufMetaValue> meta{
            {"tokenizer.ggml.model", std::string("gemma4")},
            {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "\xe2\x96\x81", "b"}}
        };
        GgufFile gguf;
        BpeTokenizer tokenizer;
        if (!GgufFile::write_v3(fixture.path, meta, {}, &error) || !gguf.load(fixture.path, &error) ||
            !tokenizer.load_from_gguf(gguf) || tokenizer.encode("a b", false) != std::vector<int32_t>{0,1,2})
            throw std::runtime_error("Gemma 4 must normalize spaces with an empty merge table");
    }
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("moe")}, {"moe.block_count", uint32_t{1}},
        {"moe.embedding_length", uint32_t{8}}, {"moe.attention.head_count", uint32_t{1}},
        {"moe.expert_count", uint32_t{2}}, {"moe.expert_feed_forward_length", uint32_t{4}}
    };
    std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", HostTensor({8,8}, DType::F32)}, {"output_norm.weight", HostTensor({8}, DType::F32)},
        {"blk.0.ffn_norm.weight", HostTensor({8}, DType::F32)}, {"blk.0.ffn_gate_inp.weight", HostTensor({2,8}, DType::F32)}
    };
    for (int e = 0; e < 2; ++e) {
        const auto suffix = std::to_string(e) + ".weight";
        weights.emplace_back("blk.0.ffn_gate." + suffix, HostTensor({4,8}, DType::F32));
        weights.emplace_back("blk.0.ffn_up." + suffix, HostTensor({4,8}, DType::F32));
        weights.emplace_back("blk.0.ffn_down." + suffix, HostTensor({8,4}, DType::F32));
    }
    if (!GgufFile::write_v3(fixture.path, meta, weights, &error)) throw std::runtime_error(error);
    {
        model_io::LoadedModelRaw raw;
        if (!model_io::parse_gguf_model_raw(fixture.path, 16, raw, &error)) throw std::runtime_error(error);
    }
    weights[7].second = HostTensor({3,8}, DType::F32);
    weights[8].second = HostTensor({5,8}, DType::F32);
    model_io::LoadedModelRaw raw;
    if (!GgufFile::write_v3(fixture.path, meta, weights, &error) ||
        model_io::parse_gguf_model_raw(fixture.path, 16, raw, &error) ||
        !error.starts_with("Invalid GGUF individual expert FFN shapes"))
        throw std::runtime_error("GGUF loader accepted mismatched per-expert gate/up widths");
    std::cout << "  [PASS] GGUF scalar/block alignment, individual expert shapes and Gemma 4 empty merges\n";
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_review_cuda_kernels(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    auto buffer = arena.alloc_tensor({4}, DType::F32);
    auto captured_grid = [&](auto launch) {
        FUSIONRT_CUDA_CHECK(cudaStreamBeginCapture(ctx.stream(), cudaStreamCaptureModeThreadLocal));
        launch(buffer.ptr<float>());
        cudaGraph_t raw = nullptr;
        FUSIONRT_CUDA_CHECK(cudaStreamEndCapture(ctx.stream(), &raw));
        std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph(raw, cudaGraphDestroy);
        cudaGraphNode_t node = nullptr;
        size_t count = 1;
        FUSIONRT_CUDA_CHECK(cudaGraphGetNodes(graph.get(), &node, &count));
        if (count != 1) throw std::runtime_error("Expected a single captured CUDA kernel");
        cudaKernelNodeParams params{};
        FUSIONRT_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
        return params.gridDim;
    };
    // Inspect the actual launch without allocating gigabytes of attention inputs.
    const auto gqa = captured_grid([&](float* p) {
        cuda_ops::gqa_prefill_causal_f32(p, p, p, p, 32768, 32, 2, 512, 0, 32768, 1.0f, ctx.stream());
    });
    if (gqa.x != 65536 || gqa.y != 2) throw std::runtime_error("HD512 prefill grid must fit CUDA's Y dimension");
    const auto ple = captured_grid([&](float* p) {
        cuda_ops::gemma_ple_gate_f32(p, p, p, 65536, 32769, 32769, 0, ctx.stream());
    });
    if (ple.x != (uint64_t{65536} * 32769 + 255) / 256) throw std::runtime_error("Large Gemma PLE grid overflow");

    bool invalid_router = false;
    try {
        cuda_ops::moe_topk_router_f32(nullptr, nullptr, nullptr, 8, 5, true, ctx.stream(), nullptr, true, 1.0f, 2, 1);
    } catch (const std::invalid_argument&) {
        invalid_router = true;
    }
    if (!invalid_router) throw std::runtime_error("Grouped router accepted more experts than the selected groups contain");

    auto upload = [&](std::initializer_list<float> values, std::vector<int64_t> dims) {
        auto host = HostTensor::create(dims, DType::F32);
        std::copy(values.begin(), values.end(), host.data<float>());
        return ctx.upload(host, false);
    };
    const auto gate = upload({1, 2}, {2}), up = upload({2, 3}, {2});
    cuda_ops::swiglu_f32(gate.ptr<float>(), up.ptr<float>(), up.ptr<float>(), 2, ctx.stream());
    cuda_ops::add_residual_f32(gate.ptr<float>(), up.ptr<float>(), up.ptr<float>(), 2, ctx.stream());
    const auto updated = ctx.download(up);
    expect_close({updated.data<float>(), updated.data<float>() + 2},
                 {1 + 2 / (1 + std::exp(-1.0)), 2 + 6 / (1 + std::exp(-2.0))},
                 1e-6, 1e-6, "Aliased up/output and residual/output");

    const auto q = upload({-1e20f, 0, 0}, {1,3});
    const auto kv = upload({1, 0, 0}, {1,3});
    const auto norm = upload({1e20f}, {1});
    const auto k_b = upload({1}, {1,1,1}), v_b = upload({1}, {1,1,1});
    const auto frequency = upload({1}, {1});
    const auto cache = make_cuda_kv_view(arena, DType::F32, 1, 1, 3, true);
    auto* page = static_cast<uint8_t*>(arena.alloc(cache.page_bytes()));
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cache.pages, &page, sizeof(page), cudaMemcpyHostToDevice, ctx.stream()));
    auto absorbed = arena.alloc_tensor({3}, DType::F32), latent = arena.alloc_tensor({1}, DType::F32);
    cuda_ops::mla_attention_f32(q.ptr<float>(), kv.ptr<float>(), norm.ptr<float>(), k_b, v_b, frequency.ptr<float>(),
        absorbed.ptr<float>(), latent.ptr<float>(), buffer.ptr<float>(), cache, 1, 1, 3, 0, nullptr, 1e-6f, 1.0f, 1.0f, ctx.stream());
    if (ctx.download(buffer).data<float>()[0] != 0.0f) throw std::runtime_error("All -Inf MLA scores must produce zero");
    arena.reset(mark);
    std::cout << "  [PASS] HD512 grid.x=65536, 64-bit PLE, MoE launch validation and all-masked MLA zero output\n";
}

void test_stage7a_speculative() {
    std::cout << "Testing prompt lookup and draft model CUDA token parity...\n";
    CudaContext ctx(0, 32, 32);
    HybridModelConfig cfg;
    cfg.vocab_size = 8;
    cfg.num_layers = 4;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 128;
    cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 192;
    cfg.dense.num_attention_heads = cfg.dense.num_key_value_heads = 2;
    cfg.gdn.num_k_heads = cfg.gdn.num_v_heads = 2;
    cfg.dense.max_seq_len = 256;
    std::vector<int32_t> prompt;
    for (int32_t a = 0; a < cfg.vocab_size; ++a) {
        for (int32_t b = 0; b < cfg.vocab_size; ++b) {
            prompt.push_back(a);
            prompt.push_back(b);
        }
    }
    constexpr int steps = 40;
    for (const char* arch : {"dense", "hybrid"}) {
        const auto mark = ctx.weight_arena().mark();
        {
            cfg.arch = arch;
            FusionModel target(cfg);
            target.init_synthetic(ctx, 505);
            const auto baseline = target.generate_cuda(ctx, prompt, steps, true, true);
            GenerationOptions options;
            options.spec_ngram = 4;
            GenerationStats stats;
            const auto actual = target.generate_cuda(ctx, prompt, steps, true, true, &stats, false, 0, -1, options);
            if (actual != baseline || stats.spec_proposed == 0 || stats.spec_accepted == stats.spec_proposed) {
                throw std::runtime_error(std::string(arch) + " prompt lookup must exercise rejection and match greedy IDs");
            }

            FusionModel draft(cfg);
            draft.init_synthetic(ctx, 505);
            options.spec_ngram = 0;
            options.draft_model = &draft;
            const auto drafted = target.generate_cuda(ctx, prompt, steps, true, true, &stats, false, 0, -1, options);
            if (drafted != baseline || stats.spec_accepted == 0) {
                throw std::runtime_error(std::string(arch) + " draft model must accept proposals and match greedy IDs");
            }
        }
        ctx.weight_arena().reset(mark);
    }
}

void test_review_speculation() {
    CudaContext ctx(0, 32, 8);
    constexpr int H = 256, V = 32;
    struct Fixture {
        std::string path = "test-review-speculation.gguf";
        ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    } fixture;
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("qwen3")}, {"qwen3.block_count", uint32_t{1}},
        {"qwen3.embedding_length", uint32_t{H}}, {"qwen3.attention.head_count", uint32_t{4}},
        {"qwen3.attention.head_count_kv", uint32_t{1}}, {"qwen3.feed_forward_length", uint32_t{H}}
    };
    HostTensor embeddings({V,H}, DType::F32), lm_head({V,H}, DType::Q4_0);
    for (int i = 0; i < V; ++i) {
        auto* row = embeddings.data<float>() + i * H;
        row[0] = 1.0f;
        row[1] = 0.501f;
        row[2] = 0.5001f;
    }
    auto* blocks = lm_head.data<BlockQ4_0>();
    for (int i = 0; i < V * H / 32; ++i) {
        blocks[i].d = 0x3c00;
        std::fill_n(blocks[i].qs, 16, uint8_t{0x88});
    }
    // Token 1 wins in FP32, but channels 1 and 2 quantize to the same value in the decode activation.
    blocks[H / 32].qs[1] = 0x89;
    blocks[H / 32].qs[2] = 0x87;
    const std::vector<std::pair<std::string, HostTensor>> weights{
        {"token_embd.weight", std::move(embeddings)}, {"output.weight", std::move(lm_head)},
        {"output_norm.weight", make_ones_f32({H})},
        {"blk.0.attn_norm.weight", make_ones_f32({H})}, {"blk.0.ffn_norm.weight", make_ones_f32({H})},
        {"blk.0.attn_q.weight", HostTensor({H,H}, DType::F32)}, {"blk.0.attn_k.weight", HostTensor({64,H}, DType::F32)},
        {"blk.0.attn_v.weight", HostTensor({64,H}, DType::F32)}, {"blk.0.attn_output.weight", HostTensor({H,H}, DType::F32)},
        {"blk.0.ffn_gate.weight", HostTensor({H,H}, DType::F32)}, {"blk.0.ffn_up.weight", HostTensor({H,H}, DType::F32)},
        {"blk.0.ffn_down.weight", HostTensor({H,H}, DType::F32)}
    };
    std::string error;
    FusionModel target(HybridModelConfig{}), draft(HybridModelConfig{});
    if (!GgufFile::write_v3(fixture.path, meta, weights, &error) ||
        !target.load_gguf(ctx, fixture.path, &error, 64, true) ||
        !draft.load_gguf(ctx, fixture.path, &error, 64, true)) throw std::runtime_error(error);
    const std::vector<int32_t> invalid{-9, INT32_MAX, 0, INT32_MAX, -9, V - 1}, valid{0,V - 1,0,V - 1,0,V - 1};
    const auto baseline = target.generate_cuda(ctx, valid, 10, false, false);
    if (!std::all_of(baseline.begin() + valid.size(), baseline.end(), [](int32_t id) { return id == 0; }))
        throw std::runtime_error("Quantized speculative regression fixture must tie at token 0");
    GenerationStats stats;
    GenerationOptions options;
    options.draft_model = &draft;
    if (target.generate_cuda(ctx, invalid, 10, false, false, &stats, false, 0, -1, options) != baseline ||
        stats.spec_proposed == 0 || stats.spec_accepted != stats.spec_proposed)
        throw std::runtime_error("Unfused draft verification changed quantized decode or did not clamp prompt IDs");
    options.draft_model = nullptr;
    options.spec_ngram = 4;
    if (target.generate_cuda(ctx, invalid, 10, false, false, &stats, false, 0, -1, options) != baseline ||
        stats.spec_proposed == 0)
        throw std::runtime_error("Prompt lookup must use clamped IDs");
    BpeTokenizer tok;
    tok.init_byte_fallback();
    JsonConstraint mismatched(tok, V + 1, true);
    options.json_constraint = &mismatched;
    bool rejected = false;
    try { (void)target.generate_cuda(ctx, valid, 10, false, false, nullptr, false, 0, -1, options); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("CUDA generation accepted a mismatched JSON vocabulary");

    HybridModelConfig cfg;
    cfg.vocab_size = 8192;
    cfg.num_layers = 1;
    cfg.dense.hidden_size = cfg.dense.head_dim = 64;
    cfg.dense.num_attention_heads = cfg.dense.num_key_value_heads = 1;
    cfg.dense.intermediate_size = 64;
    cfg.dense.max_seq_len = 64;
    FusionModel wide(cfg);
    wide.init_synthetic(ctx, 9);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    ctx.set_prefill_batch(2);
    arena.alloc(arena.capacity() - arena.used() - 96 * 1024);
    const auto ordinary = wide.generate_cuda(ctx, valid, 4, true, false);
    options.json_constraint = nullptr;
    if (wide.prefill_batch_limit(ctx) != 2 || wide.prefill_batch_limit(ctx, true) != 1 ||
        wide.generate_cuda(ctx, valid, 4, true, false, &stats, false, 0, -1, options) != ordinary ||
        stats.spec_proposed != 0)
        throw std::runtime_error("Speculation must fall back when two verification logits rows do not fit");
    arena.reset(mark);
    std::cout << "  [PASS] Unfused quantized verification, clamped draft/lookup IDs, vocabulary guard and scratch fallback\n";
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
    gguf = {};
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
void test_cuda_session_slots() {
    std::cout << "Testing CUDA session slots, KV sharing, GDN checkpoints and LRU...\n";
    CudaContext ctx(0, 48, 48);
    ctx.set_kv_dtype(DType::Q8_0);
    HybridModelConfig cfg;
    cfg.num_layers = 2;
    cfg.vocab_size = 64;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 128;
    cfg.dense.num_attention_heads = 2;
    cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 128;
    cfg.dense.max_seq_len = 512;
    const auto mark = ctx.weight_arena().mark();
    int prefilled = 0;

    {
        FusionModel model(cfg);
        model.set_num_slots(2);
        model.init_synthetic(ctx);
        const std::vector<int32_t> a(170, 1);
        const std::vector<int32_t> a_append{3};
        model.forward_prefill_cuda(ctx, a, 0);
        model.forward_prefill_cuda(ctx, a_append, static_cast<int>(a.size()));
        const auto cold = model.forward_token_cuda(ctx, 4, static_cast<int>(a.size() + 1), true, true);
        model.reset_kv_and_state(ctx);
        model.prefill_cached_cuda(ctx, a);
        const auto a_cache = model.cuda_cache_memory();
        auto b = a;
        std::fill(b.begin() + 145, b.end(), 2);
        model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto b_cache = model.cuda_cache_memory();
        if (prefilled != 25 || a_cache.kv_pages != 2 || b_cache.kv_pages != 3 ||
            b_cache.kv_bytes * 2 != a_cache.kv_bytes * 3) {
            throw std::runtime_error("Dense fork must share the complete page and copy only the partial page");
        }
        auto a_next = a;
        a_next.push_back(a_append.front());
        model.prefill_cached_cuda(ctx, a_next, true, &prefilled);
        const auto resumed = model.forward_token_cuda(ctx, 4, static_cast<int>(a_next.size()), true, true);
        if (prefilled != 1 || resumed != cold) {
            throw std::runtime_error("Dense slot A decode after the fork must match cold logits exactly");
        }
    }
    ctx.weight_arena().reset(mark);

    {
        auto hybrid = cfg;
        hybrid.arch = "hybrid";
        hybrid.num_layers = 4;
        FusionModel model(hybrid);
        model.set_num_slots(2);
        model.init_synthetic(ctx);
        const auto initial_ssm = model.cuda_cache_memory().ssm_bytes;
        std::vector<int32_t> a(147, 1);
        GenerationOptions options;
        options.reuse_prefix = true;
        GenerationStats stats;
        auto turn = model.generate_cuda(ctx, a, 3, true, true, &stats, false, 0, -1, options);
        turn.pop_back();  // Only the forwarded tokens have a checkpoint.
        auto next = turn;
        next.insert(next.end(), 7, 2);
        model.prefill_cached_cuda(ctx, next, true, &prefilled);
        if (prefilled != 7) throw std::runtime_error("Hybrid turn must append to the live recurrent state");

        auto b = turn;
        b.insert(b.end(), 8, 3);
        const auto forked = model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto cache = model.cuda_cache_memory();
        if (prefilled != 8 || cache.kv_pages != 3 || initial_ssm == 0 || cache.ssm_bytes != 7 * initial_ssm) {
            throw std::runtime_error("Hybrid fork must restore the turn checkpoint and share immutable snapshots");
        }
        model.reset_kv_and_state(ctx);
        const auto cold = model.forward_prefill_cuda(ctx, b, 0);
        expect_close(forked, {cold.begin(), cold.end()}, 1e-5, 1e-5,
                     "Hybrid GDN turn checkpoint fork against cold prefill");
    }
    ctx.weight_arena().reset(mark);

    {
        FusionModel model(cfg);
        model.set_num_slots(2);
        model.init_synthetic(ctx);
        const std::vector<int32_t> a(300, 1), b(150, 2), c(80, 3);
        model.prefill_cached_cuda(ctx, a);
        model.prefill_cached_cuda(ctx, b);
        model.prefill_cached_cuda(ctx, a);
        const auto before = model.cuda_cache_memory();
        model.prefill_cached_cuda(ctx, c, true, &prefilled);
        const auto after = model.cuda_cache_memory();
        if (prefilled != 80 || before.kv_pages != 5 || after.kv_pages != 4 ||
            after.kv_reserved_bytes != before.kv_reserved_bytes) {
            throw std::runtime_error("LRU eviction must return unreferenced pages to the free list");
        }
        model.prefill_cached_cuda(ctx, a, true, &prefilled);
        if (prefilled != 1) throw std::runtime_error("LRU eviction must keep the most recently used slot");
        model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto reused = model.cuda_cache_memory();
        if (prefilled != 150 || reused.kv_pages != 5 || reused.kv_reserved_bytes != before.kv_reserved_bytes) {
            throw std::runtime_error("An evicted prompt must start cold and recycle the released pages");
        }
    }
}

void test_tiered_cuda_kv() {
    CudaContext ctx(0, 32, 32);
    ctx.set_kv_dtype(DType::Q8_0);
    HybridModelConfig cfg;
    cfg.num_layers = 2;
    cfg.vocab_size = 64;
    cfg.dense.hidden_size = cfg.dense.intermediate_size = 128;
    cfg.dense.num_attention_heads = cfg.dense.num_key_value_heads = 2;
    cfg.dense.head_dim = 64;
    cfg.dense.max_seq_len = 384;
    const size_t page_bytes = 2 * cfg.num_layers * row_nbytes(DType::Q8_0,
        kKvPageTokens * cfg.dense.num_key_value_heads * cfg.dense.head_dim);
    const size_t vram_budget = 3 * page_bytes;
    const std::vector<int32_t> a(300, 1);
    auto b = a;
    std::fill(b.begin() + kKvPageTokens, b.end(), 2);
    auto resumed_prompt = a;
    resumed_prompt.push_back(3);

    auto run = [&](size_t budget, size_t host_budget) {
        const auto mark = ctx.weight_arena().mark();
        FusionModel model(cfg);
        model.set_num_slots(2);
        model.set_kv_vram_budget_bytes(budget);
        model.set_kv_host_budget_bytes(host_budget);
        model.init_synthetic(ctx, 41);
        int prefilled = 0;
        model.prefill_cached_cuda(ctx, a);
        model.prefill_cached_cuda(ctx, b, true, &prefilled);
        const auto paused = model.cuda_cache_memory();
        if (prefilled != 300 - kKvPageTokens ||
            (budget && (paused.kv_pages != 3 || paused.kv_reserved_bytes != budget ||
                        paused.host_kv_pages != 2 || paused.host_kv_bytes != 2 * page_bytes)) ||
            (!budget && (paused.kv_pages != 5 || paused.host_kv_bytes != 0))) {
            throw std::runtime_error("Tiered KV must spill only the paused slot's two private pages");
        }
        model.prefill_cached_cuda(ctx, resumed_prompt, true, &prefilled);
        const auto active = model.cuda_cache_memory();
        if (prefilled != 1 || (budget && (active.kv_reserved_bytes > budget || active.host_kv_pages != 2))) {
            throw std::runtime_error("Tiered KV must restore the prefix and prefill only the new token");
        }
        std::vector<float> logits = model.forward_token_cuda(ctx, 4, static_cast<int>(resumed_prompt.size()), true, true);
        ctx.weight_arena().reset(mark);
        return logits;
    };
    const auto unspilled = run(0, 0);
    const auto spilled = run(vram_budget, 4 * page_bytes);
    const auto full_host = run(vram_budget, 2 * page_bytes);
    if (spilled != unspilled || full_host != unspilled) {
        throw std::runtime_error("Tiered KV decode logits differ after restore");
    }
    std::cout << "  [PASS] Tiered KV: GPU " << vram_budget << " B, pinned host " << 2 * page_bytes
              << " B (2 pages), reused 300 tokens, max_abs=0\n";
}

HostTensor moe_test_weights(std::vector<int64_t> shape, DType dtype, std::mt19937_64& rng) {
    if (dtype == DType::F32) return rand_f32(shape, rng, 0.04f);
    auto result = HostTensor::create(shape, dtype);
    if (dtype == DType::Q4_K) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ4_K); ++i) {
            auto& block = result.data<BlockQ4_K>()[i];
            block.d = fp32_to_fp16(0.00018f);
            block.dmin = fp32_to_fp16(0.00135f);
            for (auto& scale : block.scales) scale = static_cast<uint8_t>(rng());
            for (auto& q : block.qs) q = static_cast<uint8_t>(rng());
        }
    } else if (dtype == DType::Q6_K) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ6_K); ++i) {
            auto& block = result.data<BlockQ6_K>()[i];
            block.d = fp32_to_fp16(0.000023f);
            for (auto& scale : block.scales) scale = static_cast<int8_t>(rng());
            for (auto& q : block.ql) q = static_cast<uint8_t>(rng());
            for (auto& q : block.qh) q = static_cast<uint8_t>(rng());
        }
    } else if (dtype == DType::Q4_0) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ4_0); ++i) {
            auto& block = result.data<BlockQ4_0>()[i];
            block.d = fp32_to_fp16(0.009f);
            for (auto& q : block.qs) q = static_cast<uint8_t>(rng());
        }
    } else if (dtype == DType::Q8_0) {
        for (size_t i = 0; i < result.storage.size() / sizeof(BlockQ8_0); ++i) {
            auto& block = result.data<BlockQ8_0>()[i];
            block.d = fp32_to_fp16(0.0005f);
            for (auto& q : block.qs) q = static_cast<int8_t>(rng());
        }
    }
    return result;
}

void test_cuda_native_quantized_moe() {
    constexpr int H = 256, I = 256, E = 8, resident = 3, V = 64;
    constexpr double mib = 1024.0 * 1024.0;
    struct Fixture {
        std::filesystem::path path = "test-u4-moe.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    CudaContext ctx(0, 0, 32);
    std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("qwen3moe")}, {"qwen3moe.block_count", uint32_t{1}},
        {"qwen3moe.embedding_length", uint32_t{H}}, {"qwen3moe.attention.head_count", uint32_t{8}},
        {"qwen3moe.attention.head_count_kv", uint32_t{2}}, {"qwen3moe.feed_forward_length", uint32_t{I}},
        {"qwen3moe.expert_count", uint32_t{E}}, {"qwen3moe.expert_used_count", uint32_t{2}}
    };
    for (const auto dtype : {DType::Q4_0, DType::Q4_K, DType::Q6_K, DType::F32}) {
        FusionModel model(HybridModelConfig{});
        const bool normalize = dtype != DType::Q6_K;
        meta["qwen3moe.expert_weights_norm"] = normalize;
        std::mt19937_64 rng(841);
        auto embeddings = rand_f32({V, H}, rng, 0.1f);
        for (int t = 0; t < V; ++t) embeddings.data<float>()[t * H] = 0.35f;
        HostTensor router({E, H}, DType::F32);
        // With a zero attention output, the positive first hidden feature routes to one hot and one cold expert.
        router.data<float>()[H] = 0.9f;
        router.data<float>()[6 * H] = 0.7f;
        std::vector<std::pair<std::string, HostTensor>> weights{
            {"token_embd.weight", std::move(embeddings)}, {"output_norm.weight", make_ones_f32({H})},
            {"blk.0.attn_norm.weight", make_ones_f32({H})}, {"blk.0.ffn_norm.weight", make_ones_f32({H})},
            {"blk.0.attn_q.weight", rand_f32({H, H}, rng)}, {"blk.0.attn_k.weight", rand_f32({64, H}, rng)},
            {"blk.0.attn_q_norm.weight", make_ones_f32({32})}, {"blk.0.attn_k_norm.weight", make_ones_f32({32})},
            {"blk.0.attn_v.weight", rand_f32({64, H}, rng)}, {"blk.0.attn_output.weight", HostTensor({H, H}, DType::F32)},
            {"blk.0.ffn_gate_inp.weight", std::move(router)},
            {"blk.0.ffn_gate_shexp.weight", rand_f32({I, H}, rng, 0.035f)},
            {"blk.0.ffn_up_shexp.weight", rand_f32({I, H}, rng, 0.035f)},
            {"blk.0.ffn_down_shexp.weight", rand_f32({H, I}, rng, 0.035f)},
            {"blk.0.ffn_down_exps.weight", moe_test_weights({E, H, I}, dtype, rng)}
        };
        const bool split = dtype == DType::Q4_K || dtype == DType::Q6_K;
        const auto gate_type = dtype == DType::Q6_K ? DType::Q4_K : dtype;
        const auto up_type = dtype == DType::Q4_K ? DType::Q8_0 : gate_type;
        weights.emplace_back(split ? "blk.0.ffn_gate_exps.weight" : "blk.0.ffn_gate_up_exps.weight",
                             moe_test_weights({E, split ? I : 2 * I, H}, gate_type, rng));
        if (split) weights.emplace_back("blk.0.ffn_up_exps.weight", moe_test_weights({E, I, H}, up_type, rng));
        std::string error;
        if (!GgufFile::write_v3(fixture.path.string(), meta, weights, &error)) throw std::runtime_error(error);

        const size_t gate_bytes = (split ? I : 2 * I) * row_nbytes(gate_type, H);
        const size_t down_bytes = H * row_nbytes(dtype, I);
        const size_t up_bytes = split ? I * row_nbytes(up_type, H) : 0;
        auto load = [&](int expert_override, int expected) {
            model.set_moe_gpu_experts(expert_override);
            if (!model.load_gguf(ctx, fixture.path.string(), &error, 32)) throw std::runtime_error(error);
            const auto& layer = model.moe_layers().at(0);
            const int cold = E - expected;
            if (!layer.cfg.has_shared_expert || layer.cfg.top_k != 2 ||
                layer.cfg.norm_topk_prob != normalize || !model.config().dense.qk_norm ||
                model.config().moe.gpu_resident_experts != expected || layer.cfg.gpu_resident_experts != expected ||
                layer.gpu_gate_up_w.dtype != gate_type || layer.gpu_down_w.dtype != dtype ||
                layer.gpu_gate_up_w.nbytes() != expected * gate_bytes ||
                layer.gpu_down_w.nbytes() != expected * down_bytes ||
                layer.gpu_up_w.nbytes() != expected * up_bytes ||
                layer.cpu_gate_up_w.bytes().size() != cold * gate_bytes ||
                layer.cpu_down_w.bytes().size() != cold * down_bytes ||
                layer.cpu_up_w.bytes().size() != cold * up_bytes ||
                !layer.cpu_gate_up_w.storage.empty() || !layer.cpu_down_w.storage.empty() || !layer.cpu_up_w.storage.empty() ||
                (cold && (layer.cpu_gate_up_w.desc.dtype != gate_type || layer.cpu_down_w.desc.dtype != dtype)) ||
                (split && (layer.gpu_up_w.dtype != up_type || (cold && layer.cpu_up_w.desc.dtype != up_type)))) {
                throw std::runtime_error("MoE placement or native expert dtype/byte count is incorrect");
            }
        };
        auto logits = [&] {
            const std::vector<int32_t> prompt{2, 3, 4, 5, 6, 7, 8, 9};
            auto prefill = model.forward_prefill_cuda(ctx, prompt, 0);
            auto decode = model.forward_token_cuda(ctx, 10, static_cast<int>(prompt.size()), true, true);
            return std::pair{std::move(prefill), std::move(decode)};
        };
        load(E, E);
        const size_t full_arena = ctx.weight_arena().capacity();
        const auto gpu = logits();
        load(resident, resident);
        const auto hybrid = logits();
        DeviceProfile profile;
        const size_t kv_bytes = 2 * kKvPageTokens * 64 * sizeof(float);
        profile.vram_total_mb = (ctx.scratch_arena().capacity() + kv_bytes + 256 + full_arena -
                                 (E - resident) * (gate_bytes + up_bytes + down_bytes)) / mib;
        profile.cpu_q4_gemv_gbps = 51.38;
        profile.pcie_h2d_gbps = 14.24;
        model.set_device_profile(ctx, profile);
        load(-1, resident);
        const auto automatic = logits();
        const double tolerance = dtype == DType::F32 ? 1e-6 : 1e-4;
        auto compare = [&](const std::vector<float>& a, const std::vector<float>& b) {
            double diff_sq = 0.0, ref_sq = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double diff = static_cast<double>(a[i]) - b[i];
                diff_sq += diff * diff;
                ref_sq += static_cast<double>(a[i]) * a[i];
            }
            const double rel_l2 = std::sqrt(diff_sq / ref_sq);
            if (!(rel_l2 < tolerance)) throw std::runtime_error("GPU vs hybrid MoE logits differ");
            return rel_l2;
        };
        if (automatic != hybrid) throw std::runtime_error("MoE auto placement must match the explicit split");
        std::cout << "  [PASS] Native " << dtype_name(dtype) << " MoE 8x top-2 + shared: gpu_resident=3 cpu_cold=5"
                  << " prefill(M=8) rel_l2=" << compare(gpu.first, hybrid.first)
                  << " decode(M=1) rel_l2=" << compare(gpu.second, hybrid.second) << '\n';
        ctx.weight_arena().release();
    }
}

void test_cpu_quantized_moe_speed() {
    constexpr int width = 1536;
    std::mt19937_64 rng(161);
    const auto input = rand_f32({width}, rng, 0.25f);
    const HostTensor no_separate_up;
    for (const auto dtype : {DType::Q4_0, DType::Q4_K, DType::Q6_K}) {
        const auto gate_up = moe_test_weights({1, 2 * width, width}, dtype, rng);
        const auto down = moe_test_weights({1, width, width}, dtype, rng);
        auto run = [&](CpuMoeMode mode, std::vector<float>& output) {
            auto execute = [&] {
                std::fill(output.begin(), output.end(), 0.0f);
                cpu_moe_accumulate(gate_up, no_separate_up, down, 0, input.data<float>(), 0.5f, output.data(), mode);
            };
            execute();
            double best_seconds = std::numeric_limits<double>::infinity();
            for (int sample = 0; sample < 5; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                execute();
                best_seconds = std::min(best_seconds, std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count());
            }
            return best_seconds;
        };
        std::vector<float> scalar(width), parallel(width);
        const double scalar_seconds = run(CpuMoeMode::Scalar, scalar);
        const double parallel_seconds = run(CpuMoeMode::Parallel, parallel);
        expect_close(parallel, {scalar.begin(), scalar.end()}, 3e-4, 1e-4, "CPU native expert vs scalar");
        if (std::thread::hardware_concurrency() > 1 && !(parallel_seconds < scalar_seconds)) {
            throw std::runtime_error("Parallel CPU expert must beat single-thread scalar");
        }
        std::cout << "  [PASS] CPU " << dtype_name(dtype) << " "
                  << (gate_up.storage.size() + down.storage.size()) / parallel_seconds / 1e9 << " GB/s, "
                  << scalar_seconds / parallel_seconds << "x vs single-thread scalar\n";
    }
}

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

void test_stage7c_mla() {
    constexpr int hidden = 32, vocab = 48, layers = 2, heads = 8;
    constexpr int rank = 160, rope = 8, nope = 24, key = nope + rope, value = 24;
    constexpr int experts = 4, top_k = 2, expert_width = 16, shared = 2;
    constexpr int query_rank = 24;
    constexpr float eps = 1e-6f, router_scale = 1.35f, yarn_log_mul = 0.07f;
    struct Fixture {
        std::filesystem::path path = "test-stage7c-mla.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
#ifdef FUSIONRT_ENABLE_CUDA
    CudaContext ctx(0, 0, 32);
#endif
    for (const bool lora : {false, true}) {
        std::map<std::string, GgufMetaValue> meta{
            {"general.architecture", std::string("deepseek2")}, {"deepseek2.block_count", uint32_t{layers}},
            {"deepseek2.embedding_length", uint32_t{hidden}}, {"deepseek2.attention.head_count", uint32_t{heads}},
            {"deepseek2.attention.head_count_kv", uint32_t{1}}, {"deepseek2.context_length", uint32_t{256}},
            {"deepseek2.attention.key_length", static_cast<uint32_t>(lora ? rank + rope : key)},
            {"deepseek2.attention.value_length", static_cast<uint32_t>(lora ? rank : value)},
            {"deepseek2.attention.kv_lora_rank", uint32_t{rank}}, {"deepseek2.rope.dimension_count", uint32_t{rope}},
            {"deepseek2.rope.freq_base", 100.0f}, {"deepseek2.attention.layer_norm_rms_epsilon", eps},
            {"deepseek2.feed_forward_length", uint32_t{hidden}}, {"deepseek2.leading_dense_block_count", uint32_t{1}},
            {"deepseek2.expert_count", uint32_t{experts}}, {"deepseek2.expert_used_count", uint32_t{top_k}},
            {"deepseek2.expert_shared_count", uint32_t{shared}}, {"deepseek2.expert_feed_forward_length", uint32_t{expert_width}},
            {"deepseek2.expert_gating_func", uint32_t{2}}, {"deepseek2.expert_weights_norm", true},
            {"deepseek2.expert_weights_scale", router_scale}
        };
        if (lora) {
            meta["deepseek2.attention.q_lora_rank"] = uint32_t{query_rank};
            meta["deepseek2.attention.key_length_mla"] = uint32_t{key};
            meta["deepseek2.attention.value_length_mla"] = uint32_t{value};
            meta["deepseek2.rope.scaling.type"] = std::string("yarn");
            meta["deepseek2.rope.scaling.factor"] = 4.0f;
            meta["deepseek2.rope.scaling.original_context_length"] = uint32_t{256};
            meta["deepseek2.rope.scaling.yarn_log_multiplier"] = yarn_log_mul;
            meta["deepseek2.expert_group_count"] = uint32_t{2};
            meta["deepseek2.expert_group_used_count"] = uint32_t{1};
        }
        std::mt19937_64 rng(7003);
        std::vector<std::pair<std::string, HostTensor>> tensors;
        auto add = [&](const std::string& name, std::vector<int64_t> shape, float scale = 0.09f) {
            tensors.emplace_back(name, model_io::make_rand_f32(shape, rng, scale));
        };
        auto norm_weight = [&](const std::string& name, int width) {
            add(name, {width}, 0.04f);
            float* w = tensors.back().second.data<float>();
            for (int i = 0; i < width; ++i) w[i] += 1.0f;
        };
        add("token_embd.weight", {vocab, hidden});
        add("output.weight", {vocab, hidden});
        norm_weight("output_norm.weight", hidden);
        std::array<HostTensor, layers> combined;
        for (int i = 0; i < layers; ++i) {
            const std::string p = "blk." + std::to_string(i) + ".";
            norm_weight(p + "attn_norm.weight", hidden);
            norm_weight(p + "ffn_norm.weight", hidden);
            norm_weight(p + "attn_kv_a_norm.weight", rank);
            add(p + "attn_kv_a_mqa.weight", {rank + rope, hidden});
            if (lora) {
                add(p + "attn_q_a.weight", {query_rank, hidden});
                norm_weight(p + "attn_q_a_norm.weight", query_rank);
                add(p + "attn_q_b.weight", {heads * key, query_rank});
            } else {
                add(p + "attn_q.weight", {heads * key, hidden});
            }
            add(p + "attn_output.weight", {hidden, heads * value}, 0.04f);
            combined[i] = model_io::make_rand_f32({heads, nope + value, rank}, rng, 0.04f);
            if (lora) {
                // Store split KV-B in FP16 and use the same rounded weights in the expanded reference.
                combined[i] = model_io::to_f32_host(model_io::quantize_host_matrix(combined[i], DType::F16));
                HostTensor k({heads, rank, nope}, DType::F32), v({heads, value, rank}, DType::F32);
                for (int h = 0; h < heads; ++h) {
                    const float* src = combined[i].data<float>() + h * (nope + value) * rank;
                    for (int d = 0; d < nope; ++d)
                        for (int c = 0; c < rank; ++c) k.data<float>()[(h * rank + c) * nope + d] = src[d * rank + c];
                    std::copy_n(src + nope * rank, value * rank, v.data<float>() + h * value * rank);
                }
                tensors.emplace_back(p + "attn_k_b.weight", model_io::quantize_host_matrix(k, DType::F16));
                tensors.emplace_back(p + "attn_v_b.weight", model_io::quantize_host_matrix(v, DType::F16));
            } else {
                auto packed = combined[i];
                packed.desc = Tensor::from_shape({heads * (nope + value), rank}, DType::F32, Backend::CPU);
                tensors.emplace_back(p + "attn_kv_b.weight", std::move(packed));
            }
            if (i == 0) {
                add(p + "ffn_gate.weight", {hidden, hidden});
                add(p + "ffn_up.weight", {hidden, hidden});
                add(p + "ffn_down.weight", {hidden, hidden});
            } else {
                add(p + "ffn_gate_inp.weight", {experts, hidden});
                add(p + "exp_probs_b.bias", {experts});
                auto* bias = tensors.back().second.data<float>();
                bias[0] = 3.6f;
                bias[1] = -3.0f;
                bias[2] = 1.8f;
                bias[3] = 0.7f;
                add(p + "ffn_gate_exps.weight", {experts, expert_width, hidden});
                add(p + "ffn_up_exps.weight", {experts, expert_width, hidden});
                add(p + "ffn_down_exps.weight", {experts, hidden, expert_width});
                add(p + "ffn_gate_shexp.weight", {shared * expert_width, hidden});
                add(p + "ffn_up_shexp.weight", {shared * expert_width, hidden});
                add(p + "ffn_down_shexp.weight", {hidden, shared * expert_width});
            }
        }
        std::string error;
        const auto path = fixture.path.string();
        model_io::LoadedModelRaw raw;
        if (!GgufFile::write_v3(path, meta, tensors, &error) ||
            !model_io::parse_gguf_model_raw(path, 0, raw, &error)) {
            throw std::runtime_error("DeepSeek MLA GGUF roundtrip: " + error);
        }
        const auto& mla = raw.cfg.dense.mla;
        if (mla.query_rank != (lora ? query_rank : 0) || mla.latent_width() != rank + rope ||
            mla.value_dim != value || mla.leading_dense_layers != 1 || raw.cfg.dense.head_dim != key ||
            raw.cfg.dense.num_key_value_heads != 1 || raw.cfg.moe.num_experts != experts ||
            !raw.cfg.moe.sigmoid_router || raw.cfg.moe.top_k != top_k || raw.cfg.moe.expert_weights_scale != router_scale ||
            raw.cfg.moe.expert_groups != (lora ? 2 : 1) || raw.cfg.moe.expert_groups_used != 1 ||
            !raw.layers.at(0).router_w.bytes().empty() || raw.layers.at(1).router_bias.bytes().empty() ||
            !raw.layers.at(1).has_shared || raw.cfg.moe.shared_intermediate_size != shared * expert_width) {
            throw std::runtime_error("DeepSeek MLA roundtrip metadata and tensors");
        }
        std::cout << "  [PASS] DeepSeek MLA GGUF " << (lora ? "Q-LoRA/split KV-B/YaRN" : "direct Q/combined KV-B")
                  << ", dense + MoE, shared and sigmoid+bias routing\n";
#ifdef FUSIONRT_ENABLE_CUDA
        FusionModel model(HybridModelConfig{});
        model.set_moe_gpu_experts(experts);
        if (!model.load_gguf(ctx, path, &error, 0, true)) throw std::runtime_error(error);
        const std::vector<int32_t> prompt{2, 9, 11, 3, 5, 24, 7, 8, 16};
        using Row = std::vector<double>;
        auto weight = [&](const std::string& name) -> const HostTensor& {
            return std::find_if(tensors.begin(), tensors.end(), [&](const auto& entry) { return entry.first == name; })->second;
        };
        auto linear = [](const HostTensor& w, const Row& input, int group = 0) {
            const int64_t n = w.desc.shape[w.desc.ndim - 2], k = w.desc.shape[w.desc.ndim - 1];
            const float* p = w.data<float>() + group * n * k;
            Row out(n);
            for (int64_t i = 0; i < n; ++i)
                out[i] = std::inner_product(input.begin(), input.end(), p + i * k, 0.0);
            return out;
        };
        auto norm = [&](const std::string& name, Row input) {
            const double rms = std::sqrt(std::inner_product(input.begin(), input.end(), input.begin(), 0.0) / input.size() + eps);
            const float* w = weight(name).data<float>();
            for (size_t i = 0; i < input.size(); ++i) input[i] *= w[i] / rms;
            return input;
        };
        auto rotate = [&](double* x, size_t t) {
            for (int j = 0; j < rope / 2; ++j) {
                // For this YaRN fixture the correction interval is [0, 4].
                const double rate = lora ? (1 - 0.75 * j / 4) : 1;
                const double a = t * std::pow(100.0, -2.0 * j / rope) * rate;
                const double c = std::cos(a), s = std::sin(a), re = x[2 * j], im = x[2 * j + 1];
                x[2 * j] = c * re - s * im;
                x[2 * j + 1] = s * re + c * im;
            }
        };
        auto ffn = [&](const std::string& p, const std::string& suffix, const Row& input, int group = 0) {
            auto gate = linear(weight(p + "ffn_gate" + suffix + ".weight"), input, group);
            auto up = linear(weight(p + "ffn_up" + suffix + ".weight"), input, group);
            for (size_t j = 0; j < gate.size(); ++j) gate[j] *= up[j] / (1.0 + std::exp(-gate[j]));
            return linear(weight(p + "ffn_down" + suffix + ".weight"), gate, group);
        };
        std::vector<Row> h;
        for (const int token : prompt) {
            const auto* emb = weight("token_embd.weight").data<float>() + token * hidden;
            h.emplace_back(emb, emb + hidden);
        }
        const double mscale = lora ? 1 + yarn_log_mul * std::log(4.0) : 1;
        const double attention_scale = mscale * mscale / std::sqrt(key);
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            std::vector<Row> queries, keys, expanded_kv;
            for (size_t t = 0; t < h.size(); ++t) {
                auto input = norm(p + "attn_norm.weight", h[t]);
                auto q = lora ? linear(weight(p + "attn_q_b.weight"),
                                       norm(p + "attn_q_a_norm.weight", linear(weight(p + "attn_q_a.weight"), input)))
                              : linear(weight(p + "attn_q.weight"), input);
                for (int a = 0; a < heads; ++a) rotate(q.data() + a * key + nope, t);
                queries.push_back(std::move(q));
                auto latent = linear(weight(p + "attn_kv_a_mqa.weight"), input);
                keys.emplace_back(latent.begin() + rank, latent.end());
                rotate(keys.back().data(), t);
                latent.resize(rank);
                latent = norm(p + "attn_kv_a_norm.weight", std::move(latent));
                expanded_kv.emplace_back();
                for (int a = 0; a < heads; ++a) {
                    const auto full = linear(combined[l], latent, a);
                    expanded_kv.back().insert(expanded_kv.back().end(), full.begin(), full.end());
                }
            }
            for (size_t t = 0; t < h.size(); ++t) {
                Row attended(heads * value, 0.0);
                for (int a = 0; a < heads; ++a) {
                    const double* q = queries[t].data() + a * key;
                    Row scores(t + 1);
                    for (size_t j = 0; j <= t; ++j) {
                        const double* k = expanded_kv[j].data() + a * (nope + value);
                        scores[j] = attention_scale * (std::inner_product(q, q + nope, k, 0.0) +
                            std::inner_product(q + nope, q + key, keys[j].begin(), 0.0));
                    }
                    const double peak = *std::max_element(scores.begin(), scores.end());
                    double mass = 0;
                    for (size_t j = 0; j <= t; ++j) {
                        const double probability = std::exp(scores[j] - peak);
                        mass += probability;
                        const double* v = expanded_kv[j].data() + a * (nope + value) + nope;
                        for (int d = 0; d < value; ++d) attended[a * value + d] += probability * v[d];
                    }
                    for (int d = 0; d < value; ++d) attended[a * value + d] /= mass;
                }
                auto projected = linear(weight(p + "attn_output.weight"), attended);
                for (int d = 0; d < hidden; ++d) h[t][d] += projected[d];
                const auto input = norm(p + "ffn_norm.weight", h[t]);
                auto ff = ffn(p, l == 0 ? "" : "_shexp", input);
                if (l != 0) {
                    auto gates = linear(weight(p + "ffn_gate_inp.weight"), input);
                    for (double& g : gates) g = 1.0 / (1.0 + std::exp(-g));
                    // Group routing selects group 1 even though expert 0 has the highest individual score.
                    const int other = lora ? 3 : 0;
                    const double sum = gates[other] + gates[2];
                    for (const int e : {other, 2}) {
                        const auto update = ffn(p, "_exps", input, e);
                        for (int d = 0; d < hidden; ++d) ff[d] += router_scale * gates[e] / sum * update[d];
                    }
                }
                for (int d = 0; d < hidden; ++d) h[t][d] += ff[d];
            }
        }
        auto rel_l2 = [](const auto& actual, const auto& reference) {
            double diff = 0, magnitude = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                const double err = double(actual[i]) - reference[i];
                diff += err * err;
                magnitude += double(reference[i]) * reference[i];
            }
            const double result = std::sqrt(diff / magnitude);
            if (!(result < 1e-5)) throw std::runtime_error("DeepSeek MLA rel_l2=" + std::to_string(result));
            return result;
        };
        const auto reference = [&](size_t t) { return linear(weight("output.weight"), norm("output_norm.weight", h[t])); };
        const auto batch = model.forward_prefill_cuda(ctx, prompt, 0);
        constexpr size_t latent_bytes = layers * kKvPageTokens * (rank + rope) * sizeof(float);
        constexpr size_t expanded_bytes = layers * kKvPageTokens * heads * (key + value) * sizeof(float);
        static_assert(latent_bytes < expanded_bytes);
        if (model.cuda_cache_memory().kv_bytes != latent_bytes) throw std::runtime_error("DeepSeek MLA latent KV bytes");
        model.reset_kv_and_state(ctx);
        std::vector<float> step;
        double step_error = 0;
        for (size_t t = 0; t < prompt.size(); ++t) {
            step = model.forward_token_cuda(ctx, prompt[t], static_cast<int>(t));
            step_error = std::max(step_error, rel_l2(step, reference(t)));
        }
        const auto eager = model.generate_cuda(ctx, prompt, 16, true, false);
        const auto graph = model.generate_cuda(ctx, prompt, 16, true, true);
        if (eager != graph || graph.size() != prompt.size() + 16) throw std::runtime_error("DeepSeek MLA graph tokens");

        std::vector<int32_t> prefix(kKvPageTokens + 9);
        for (auto& token : prefix) token = static_cast<int32_t>(rng() % vocab);
        auto branch = prefix;
        std::fill(branch.begin() + kKvPageTokens + 4, branch.end(), 21);
        prefix.push_back(31);
        model.reset_kv_and_state(ctx);
        const auto fresh_a = model.forward_prefill_cuda(ctx, prefix, 0);
        model.reset_kv_and_state(ctx);
        const auto fresh_b = model.forward_prefill_cuda(ctx, branch, 0);
        model.reset_kv_and_state(ctx);
        model.set_num_slots(2);
        model.prefill_cached_cuda(ctx, {prefix.begin(), prefix.end() - 1});
        int prefilled = 0;
        const auto reuse_b = model.prefill_cached_cuda(ctx, branch, true, &prefilled);
        if (prefilled != 5 || model.cuda_cache_memory().kv_bytes != 3 * latent_bytes)
            throw std::runtime_error("DeepSeek MLA slots must share the complete page and fork the partial page");
        const auto reuse_a = model.prefill_cached_cuda(ctx, prefix, true, &prefilled);
        if (prefilled != 1) throw std::runtime_error("DeepSeek MLA slot resume did not reuse the prefix");
        std::cout << "  [PASS] DeepSeek MLA batch/FP64=" << rel_l2(batch, reference(h.size() - 1))
                  << " step/FP64(max)=" << step_error << " batch/step=" << rel_l2(batch, step)
                  << " latent_bytes=" << latent_bytes << " expanded_bytes=" << expanded_bytes
                  << " Graph/eager=16/16 slot_fork=" << rel_l2(reuse_b, fresh_b)
                  << " slot_resume=" << rel_l2(reuse_a, fresh_a) << "\n";
#endif
    }
}

}  // namespace

int main() {
    std::cout << std::unitbuf;
    try {
        test_quant_and_io();
        test_agent_protocols();
        test_agent_reasoning();
        test_device_profile_and_cost_model();
#ifdef FUSIONRT_ENABLE_CUDA
        CudaContext cuda_ctx(0, 256, 512);
        if (cuda_ctx.device_profile().sm_count != cuda_ctx.info().multi_processor_count ||
            cuda_ctx.device_profile().vram_total_mb !=
                static_cast<double>(cuda_ctx.info().total_vram_bytes) / (1024.0 * 1024.0) ||
            cuda_ctx.cost_hardware().hbm_bw_gbps != cuda_ctx.info().peak_bandwidth_gbps) {
            throw std::runtime_error("CUDA device profile runtime defaults");
        }
        test_cuda_fused_rmsnorm_linear(cuda_ctx);
        test_cuda_step1_fast_paths(cuda_ctx);
        test_cuda_gated_deltanet_fused_vs_unfused_and_fp64(cuda_ctx);
        test_cuda_qk_rope_and_gqa_64k(cuda_ctx);
        test_cuda_graph_and_moe_offload(cuda_ctx);
        test_cuda_wave1_decode(cuda_ctx);
        test_repacked_quants(cuda_ctx);
        test_cuda_wave2_prefill(cuda_ctx);
        test_cuda_sliding_window(cuda_ctx);
        test_cuda_gemma_epilogues(cuda_ctx);
        test_cuda_gemma4_gguf();
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
        test_vulkan_all_devices();
        test_vulkan_batched_vs_unbatched();
        test_vulkan_continued_turn();
#endif
#if defined(FUSIONRT_ENABLE_CUDA) && defined(FUSIONRT_ENABLE_VULKAN)
        test_phase2_loaders_prefill_and_vulkan_e2e(cuda_ctx);
#endif
        test_wave3();
        test_stage7a_json();
        test_stage7b_images_and_protocols();
        test_review_input_validation();
        test_review_gguf_validation();
        test_stage7c_mla();
#ifdef FUSIONRT_ENABLE_CUDA
        test_review_cuda_kernels(cuda_ctx);
        test_stage7a_speculative();
        test_review_speculation();
        test_stage7b_cuda_vision();
        test_chunked_prefill_and_k_quant_gdn();
        test_cuda_session_slots();
        test_tiered_cuda_kv();
        test_cuda_native_quantized_moe();
        test_cpu_quantized_moe_speed();
#endif
        std::cout << "\nALL TESTS PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
