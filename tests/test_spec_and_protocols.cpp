#include "test_support.h"
#include "../src/core/spec_cost.h"

namespace fusionrt::test {

void test_sampling_and_benchmark_gates() {
    auto rejected = [](auto&& fn) {
        try { fn(); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 2> bad{nan, nan};
    const std::array<float, 3> mixed{nan, -1.f, -2.f};
    std::mt19937_64 rng(7);
    if (!rejected([&] { (void)Sampler::sample(bad.data(), 2, {}, {}, rng); }) ||
        !rejected([&] { (void)Sampler::distribution(nullptr, 0, {}, {}); }) ||
        !rejected([&] { (void)Sampler::argmax(bad.data(), 2); }) ||
        !rejected([&] { (void)Sampler::draw({}, rng); }) || Sampler::argmax(mixed.data(), 3) != 1)
        throw std::runtime_error("Sampling must reject empty and all-NaN logits");
    if (bench::benchmark_passed(0, 0) || bench::benchmark_passed(2, 1) || !bench::benchmark_passed(1, 0))
        throw std::runtime_error("Benchmark gate must reject empty and failing reports");
    struct Fixture {
        std::string path = "test-failed-benchmark.json";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    auto check_suite = [&](const char* suite, std::optional<int> vk_device = {}) {
        if (bench::run_benchmark_suite(suite, fixture.path, vk_device) == 0)
            throw std::runtime_error("An unavailable suite must return a nonzero exit code");
        std::ifstream input(fixture.path);
        const std::string json((std::istreambuf_iterator<char>(input)), {});
        const auto report = JsonValue::parse(json).as_object();
        if (report.at("summary").as_object().at("overall_status").as_string() != "fail")
            throw std::runtime_error("An unavailable suite must write a failing report");
        if (vk_device && report.at("error").as_string().find("Unavailable Vulkan device") == std::string::npos)
            throw std::runtime_error("An unavailable explicit Vulkan device must be reported");
    };
    check_suite("nonexistent-suite");
    check_suite("all", std::numeric_limits<int>::max());
    struct FailedFlush : std::stringbuf { int sync() override { return -1; } } buffer;
    std::ostream incomplete(&buffer);
    if (bench::write_benchmark_report(incomplete, {{"test", true}}) || buffer.str().empty())
        throw std::runtime_error("Benchmark report must detect a failed flush after writing");
#ifndef FUSIONRT_ENABLE_CUDA
    check_suite("blocks");
#endif
    std::cout << "  [PASS] Empty and all-NaN sampling, empty/failing/unknown benchmark gate\n";
}

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

    const auto resident = prefix_copy_ms(1 << 20, 1, 592, 4);
    const auto spilled = resident + prefix_copy_ms(1 << 20, 1, 14, 12);
    check(prefix_restore_worthwhile(resident, 0.05) && !prefix_restore_worthwhile(spilled, 0.05) &&
          !prefix_restore_worthwhile(0.05, 0.05), "A prefix restore must save time after spill and snapshot copies");

    MtpDepthController depth;
    // Bonsai 2.5 PTQ1_0: measured widths 2, 4, 6 and 8, with intermediate widths interpolated.
    const double verify_ms[] = {1000 / 43.26, 27.97, 31, 33.72, 60, 87.04, 79, 71.30};
    // Conditional probabilities reconstructed from the observed acceptance fractions.
    const double acceptance[] = {0.78, 0.75, 0.73, 0.60, 0.58, 0.30, 0.28};
    for (int i = 0; i < 8; ++i) depth.observe_verify(i + 1, verify_ms[i]);
    for (int i = 0; i < 7; ++i) depth.observe_acceptance(i + 1, acceptance[i]);
    depth.observe_draft(0.8);
    depth.observe_rewind(1, 0.25);
    check(depth.best_depth(7) == 3, "MTP must avoid the measured verification cliff");
    for (int i = 0; i < 16; ++i) depth.observe_acceptance(1, false);
    check(depth.best_depth(7) == 0, "MTP must back off when early proposals are rejected");

    MtpDepthController probe;
    check(probe.choose_depth(7) == 0 && probe.choose_depth(7) == 1, "MTP measures its baseline and first draft");
    probe.observe_verify(1, verify_ms[0]);
    probe.observe_verify(2, verify_ms[1]);
    for (int i = 2; i < 8; ++i) check(probe.choose_depth(7) == 0, "MTP uses measured widths between probes");
    check(probe.choose_depth(7) == 2, "MTP probes an unmeasured width");
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
        check(blocks.size() == 3 && std::get<std::string>(blocks[0]) == "ok" &&
              std::get<ToolCall>(blocks[1]).arguments.at("text").as_string() == "</tool_call>" &&
              std::get<std::string>(blocks[2]) == "<tool_ca",
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

        AgentResponse truncated(protocol, "truncated", "model", 0, false);
        truncated.start(10);
        ToolParser incomplete(chat, {}, [&](std::string_view text) { truncated.text(text); },
                              [&](ToolCall call) { truncated.call(std::move(call)); });
        const std::string tail = R"(<tool_call>{"name":"calc","arguments":{"value":)";
        incomplete.push(R"(<tool_call>{"name":"calc","arguments":{}}</tool_call>)");
        incomplete.push(tail);
        incomplete.finish();
        check(incomplete.truncated_call() && truncated.result().output.size() == 2 &&
              std::get<std::string>(truncated.result().output.back()) == tail, "Incomplete tool call must retain raw text");
        truncated.finish(Finish::truncated_tool_call, 5);
        const auto truncated_body = truncated.body().as_object();
        if (protocol == Protocol::chat)
            check(truncated_body.at("choices").as_array()[0].as_object().at("finish_reason").as_string() == "length",
                  "Chat tool truncation must finish with length");
        else if (protocol == Protocol::responses)
            check(truncated_body.at("status").as_string() == "incomplete", "Responses tool truncation must be incomplete");
        else
            check(truncated_body.at("stop_reason").as_string() == "max_tokens", "Messages tool truncation must be max_tokens");
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
    for (const auto partial : {"<thi", "<tool_cal"}) {
        std::string text;
        const AgentPrompt prompt;
        ToolParser parser(request, prompt, [&](std::string_view chunk) { text += chunk; },
            [&](ToolCall) { throw std::runtime_error("An incomplete delimiter is plain text"); });
        parser.push("answer ");
        parser.push(partial);
        check(text == "answer ", "partial plain-text delimiter held before EOF");
        parser.finish();
        check(text == std::string("answer ") + partial, "plain-text delimiter suffix flushed verbatim");
    }
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
    try {
        (void)fresh.generate_vulkan(vk, {1, 32}, 1);
        throw std::runtime_error("Vulkan accepted an out-of-range token");
    } catch (const std::invalid_argument& e) {
        if (std::string_view(e.what()).find("Token ID 32 at position 1 is outside vocabulary size 32") ==
            std::string_view::npos) throw;
    }
    const int next_pos = static_cast<int>(first.size());
    const int32_t next_user_token = (first.back() + 1) % cfg.vocab_size;
    for (int i = 0; i < next_pos; ++i) (void)fresh.forward_token_vulkan(vk, first[i], i);
    const auto expected = fresh.forward_token_vulkan(vk, next_user_token, next_pos);
    const std::vector<double> expected_ref(expected.begin(), expected.end());
    for (const auto eos : {int32_t{-1}, first.back()}) {
        FusionModel continued(cfg);
        continued.init_synthetic_vulkan(vk, seed);
        const auto answer = continued.generate_vulkan(vk, prompt, eos < 0 ? 1 : 4, nullptr, true, 0, eos);
        if (answer != first)
            throw std::runtime_error("Vulkan must cache the final token before continuing a turn");
        const auto got = continued.forward_token_vulkan(vk, next_user_token, next_pos);
        expect_close(got, expected_ref, 1e-5, 1e-5, "Vulkan continued-turn logits");
    }
    std::cout << "  [PASS] Vulkan continued-turn logits after EOS and max steps\n";
    VkPhysicalDeviceMemoryProperties heaps{};
    heaps.memoryHeapCount = heaps.memoryTypeCount = 2;
    heaps.memoryTypes[1].heapIndex = 1;
    heaps.memoryHeaps[0].size = 4096;
    heaps.memoryHeaps[1].size = 512;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT heap_budget{};
    heap_budget.heapBudget[0] = 4096;
    heap_budget.heapBudget[1] = 512;
    heap_budget.heapUsage[1] = 500;
    if (vulkan_detail::available_heap_bytes(heaps, &heap_budget, 1, 0) != 12 ||
        vulkan_detail::available_heap_bytes(heaps, nullptr, 1, 500) != 12)
        throw std::runtime_error("Vulkan preflight must use the selected memory type's heap");
    VkContext capacity(0, 0, 32);
    FusionModel exact(cfg);
    exact.init_synthetic_vulkan(capacity, seed);
    if (capacity.weight_arena().used() != capacity.weight_arena().capacity())
        throw std::runtime_error("Vulkan arenas must match the calculated backend bytes exactly");
    try {
        capacity.reserve_model_memory(0, capacity.info().device_local_vram_bytes + 1);
        throw std::runtime_error("Vulkan accepted an oversized model footprint");
    } catch (const std::runtime_error& e) {
        if (std::string_view(e.what()).find("KV/state bytes") == std::string_view::npos) throw;
    }
    std::cout << "  [PASS] Vulkan exact arena, OOM accounting, and out-of-range token rejection\n";
}
#endif  // FUSIONRT_ENABLE_VULKAN

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
        auto op = ServeSpeculation{4}.for_inputs(p.inputs_embeds);
        op.reuse_prefix = true;
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

void test_gemma4v_vision() {
    CudaContext ctx(0, 0, 32);
    constexpr int H = 64, side = 32;
    struct Fixture {
        std::string path = "test-gemma4v-mmproj.gguf";
        ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    } fixture;
    const std::map<std::string, GgufMetaValue> meta{
        {"general.architecture", std::string("clip")}, {"clip.vision.projector_type", std::string("gemma4v")},
        {"clip.vision.patch_size", uint32_t{1}}, {"clip.vision.image_size", uint32_t{12}},
        {"clip.vision.embedding_length", uint32_t{H}}, {"clip.vision.block_count", uint32_t{1}},
        {"tokenizer.ggml.model", std::string("gemma4")},
        {"tokenizer.ggml.tokens", std::vector<std::string>{
            "<pad>", "<bos>", "<unk>", "<|image>", "<image|>", "<|image|>", "a", "z"}},
        {"tokenizer.ggml.bos_token_id", uint32_t{1}}, {"tokenizer.ggml.add_bos_token", true}
    };
    HostTensor patch({H, 3, 1, 1}, DType::F16), projection({H, H}, DType::F16);
    patch.data<uint16_t>()[0] = 0x3c00;
    for (int i = 0; i < H; ++i) projection.data<uint16_t>()[i * H + i] = 0x3c00;
    HostTensor positions({2, side, H}, DType::F32), post_norm({H}, DType::F32), clamp({1}, DType::F32);
    for (int i = 0; i < side; ++i) {
        positions.data<float>()[i * H + 1] = static_cast<float>(i) / side;
        positions.data<float>()[(side + i) * H + 2] = static_cast<float>(i) / side;
    }
    post_norm.data<float>()[3] = 1.f;
    clamp.data<float>()[0] = 0.25f;
    const auto ones = make_ones_f32({H});
    const HostTensor zero({H, H}, DType::F32);
    const std::vector<std::pair<std::string, HostTensor>> tensors{
        {"v.patch_embd.weight", patch}, {"v.position_embd.weight", positions},
        {"v.blk.0.ln1.weight", ones}, {"v.blk.0.ln2.weight", ones},
        {"v.blk.0.attn_q.weight", zero}, {"v.blk.0.attn_k.weight", zero}, {"v.blk.0.attn_v.weight", zero},
        {"v.blk.0.attn_q_norm.weight", ones}, {"v.blk.0.attn_k_norm.weight", ones},
        {"v.blk.0.attn_out.weight", zero}, {"v.blk.0.attn_out.output_min", clamp},
        {"v.blk.0.attn_out.output_max", clamp}, {"v.blk.0.attn_post_norm.weight", post_norm},
        {"v.blk.0.ffn_gate.weight", zero}, {"v.blk.0.ffn_up.weight", zero}, {"v.blk.0.ffn_down.weight", zero},
        {"v.blk.0.ffn_post_norm.weight", ones}, {"mm.input_projection.weight", projection},
        {"a.test.weight", zero}, {"mm.a.test.weight", zero}
    };
    std::string error;
    GgufFile file;
    BpeTokenizer tokenizer;
    if (!GgufFile::write_v3(fixture.path, meta, tensors, &error) ||
        !file.load(fixture.path, &error) || !tokenizer.load_from_gguf(file))
        throw std::runtime_error("Gemma 4 vision fixture: " + error);
    VisionEncoder encoder(ctx, fixture.path, H);
    const auto red = decode_image_base64("UDYKMSAxCjI1NQr/AAA=");
    const auto green = decode_image_base64("UDYKMSAxCjI1NQoA/wA=");
    constexpr int grid = 9, count = grid * grid;
    double diff = 0, base = 0;
    for (const auto* image : {&red, &green}) {
        const auto prompt = encode_vision_prompt(ctx, encoder, tokenizer, "a<image>z", {&image, 1}, 128);
        std::vector<int32_t> expected{1, 6, 3};
        expected.insert(expected.end(), count, 0);
        expected.insert(expected.end(), {4, 7});
        if (prompt.tokens != expected || prompt.inputs_embeds.size() != 1 || prompt.inputs_embeds[0].start != 3 ||
            prompt.inputs_embeds[0].rows.shape[0] != count)
            throw std::runtime_error("Gemma 4 vision grid or image framing");
        const auto output = ctx.download(prompt.inputs_embeds[0].rows);
        for (int y = 0; y < grid; ++y) for (int x = 0; x < grid; ++x) {
            std::array<double, H> ref{image->rgb[0] > 0 ? 1.0 : -1.0, (x * 3 + 1.0) / side, (y * 3 + 1.0) / side,
                                      0.25 / std::sqrt(0.0625 + 1e-6)};
            const auto rms = std::sqrt(std::inner_product(ref.begin(), ref.end(), ref.begin(), 0.0) / H + 1e-6 / H);
            for (int d = 0; d < H; ++d) {
                const double want = ref[d] / rms;
                const double delta = output.data<float>()[(y * grid + x) * H + d] - want;
                diff += delta * delta;
                base += want * want;
            }
        }
    }
    const double rel = std::sqrt(diff / base);
    if (!(rel < 5e-4)) throw std::runtime_error("Gemma 4 vision projection relative error: " + std::to_string(rel));
    std::cout << "  [PASS] Gemma 4 vision resizing, RGB projection, pooling and image framing rel_l2=" << rel << '\n';
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
    const auto health = http::health_body("local", health_cfg, 1, 128, {});
    if (health.at("max_seq").as_int() != 4096 || health.at("max_active_tokens").as_int() != 128)
        throw std::runtime_error("/health must include model context and active KV capacity");
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
    if (agent::generation_limit(omitted.generation, static_cast<int>(health.at("max_active_tokens").as_int()), 127) != 1 ||
        agent::generation_limit(omitted.generation, 2048, 127) != 1921 ||
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
    cli::SpeculativeArgs speculative;
    if (!speculative.parse("--spec-ngram", "4")) throw std::runtime_error("Missing --spec-ngram");
    speculative.validate(true);
    const ServeSpeculation lookup{speculative.ngram};
    const InputEmbedding image{};
    const auto lookup_text = lookup.for_inputs({});
    const auto lookup_image = lookup.for_inputs({&image, 1});
    if (lookup_text.spec_ngram != 4 || lookup_image.spec_ngram != 0 || lookup_image.inputs_embeds.data() != &image)
        throw std::runtime_error("Serve prompt lookup must bypass image embeddings");
    if (!speculative.parse("--draft-model", "draft.gguf") || !speculative.parse("--draft-tokens", "6"))
        throw std::runtime_error("Missing serve draft arguments");
    reject([&] { speculative.validate(true); });
    speculative.parse("--spec-ngram", "0");
    reject([&] { speculative.validate(false); });
    speculative.validate(true);
    FusionModel draft(HybridModelConfig{});
    const ServeSpeculation companion{speculative.ngram, &draft, speculative.draft_tokens};
    const auto draft_text = companion.for_inputs({});
    const auto draft_image = companion.for_inputs({&image, 1});
    if (draft_text.draft_model != &draft || draft_text.draft_tokens != 6 ||
        draft_image.draft_model || draft_image.inputs_embeds.data() != &image)
        throw std::runtime_error("Serve draft model must bypass image embeddings");
    speculative.draft_path.clear();
    reject([&] { speculative.validate(true); });
    std::cout << "  [PASS] JSON nesting, schema/root and tool choice, JSON output budget, HTTP 413, and strict CLI integers\n";
    std::cout << "  [PASS] Serve speculative flags, mode validation and multimodal bypass\n";
}

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_failed_graph_recapture(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    auto value = arena.alloc_tensor({1}, DType::I32);
    CudaGraphExec graph;
    auto capture = [&](int byte) {
        graph.capture(ctx.stream(), [&] {
            FUSIONRT_CUDA_CHECK(cudaMemsetAsync(value.ptr(), byte, value.nbytes(), ctx.stream()));
        });
    };
    auto check = [&](int pattern) {
        graph.launch(ctx.stream());
        if (*ctx.download(value).data<int32_t>() != pattern)
            throw std::runtime_error("CUDA Graph used an unexpected executable after recapture");
    };
    capture(1);
    check(0x01010101);
    auto& instantiate = CudaReviewTestAccess::instantiate(graph);
    auto real_instantiate = instantiate;
    instantiate = [](cudaGraphExec_t*, cudaGraph_t) { return cudaErrorInvalidValue; };
    bool failed = false;
    try { capture(2); } catch (const std::runtime_error&) { failed = true; }
    if (!failed || !graph.valid()) throw std::runtime_error("Failed recapture must preserve the previous CUDA Graph");
    check(0x01010101);
    instantiate = real_instantiate;
    capture(3);
    check(0x03030303);
    ctx.synchronize();
    graph.reset();
    arena.reset(mark);
    std::cout << "  [PASS] CUDA Graph preserves its executable after failed instantiation and supports recapture\n";
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
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
    auto reject_invalid = [&] {
        try { (void)target.generate_cuda(ctx, invalid, 10, false, false, &stats, false, 0, -1, options); }
        catch (const std::invalid_argument& e) {
            if (std::string_view(e.what()).find("Token ID -9 at position 0 is outside vocabulary size 32") !=
                std::string_view::npos) return;
            throw;
        }
        throw std::runtime_error("Speculative generation accepted an invalid token");
    };
    reject_invalid();
    if (target.generate_cuda(ctx, valid, 10, false, false, &stats, false, 0, -1, options) != baseline ||
        stats.spec_proposed == 0 || stats.spec_accepted != stats.spec_proposed)
        throw std::runtime_error("Unfused draft verification changed quantized decode");
    options.draft_model = nullptr;
    options.spec_ngram = 4;
    reject_invalid();
    if (target.generate_cuda(ctx, valid, 10, false, false, &stats, false, 0, -1, options) != baseline ||
        stats.spec_proposed == 0)
        throw std::runtime_error("Prompt lookup must preserve quantized decode");
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

}  // namespace fusionrt::test
