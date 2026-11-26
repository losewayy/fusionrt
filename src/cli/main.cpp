#include "fusionrt/bench.h"
#include "fusionrt/gguf.h"
#include "fusionrt/models.h"
#include "fusionrt/safetensors.h"
#include "fusionrt/server.h"
#include "fusionrt/tokenizer.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using namespace fusionrt;

namespace {

void print_usage() {
    std::cout << "fusionrt v" << FUSIONRT_VERSION << " - Direct CUDA sm_120 + Vulkan LLM Inference Engine\n"
              << "Usage:\n"
              << "  fusionrt info\n"
              << "  fusionrt run [--arch dense|hybrid|moe] [--dtype f32|q8|q4] [--backend cuda|vulkan]\n"
              << "               [--vk-device 0] [--model <gguf|safetensors>] [--prompt \"text\"]\n"
              << "               [--steps 16] [--max-seq 4096] [--graph|--no-graph] [--unfused] [--stream]\n"
              << "               [--temp 0] [--top-k 40] [--top-p 0.95] [--min-p 0.05] [--repeat-penalty 1] [--seed 42]\n"
              << "               [--profile <json>] [--profile-ops <json>] [--trace-layers <json>]\n"
              << "  fusionrt inspect --model <path> [--max-seq 4096]\n"
              << "  fusionrt chat --model <path> [--system <text>] [--temp 0.7] [--max-tokens 512]\n"
              << "  fusionrt serve --model <path> [--host 127.0.0.1] [--port 8080] [--max-seq 4096] [--profile <json>]\n"
              << "  fusionrt bench-model --model <path> [--pp 128] [--tg 64] [--decode-prefix 0] [--ubatch 512] [--reps 5] [--graph] [--profile-ops <json>]\n"
              << "  CUDA run, chat, serve, inspect and bench-model accept --kv-type {f32|f16|q8_0}.\n"
              << "  fusionrt bench [--suite all|fusions|blocks|vulkan] [--output <path>] [--vk-device 0]\n";
}

int cmd_info() {
#ifdef FUSIONRT_ENABLE_CUDA
    auto cuda_devs = CudaContext::enumerate_devices();
    std::cout << "CUDA Devices (" << cuda_devs.size() << "):\n";
    for (const auto& d : cuda_devs) {
        std::cout << "  [" << d.device_id << "] " << d.name
                  << " (sm_" << d.sm_major << d.sm_minor
                  << ", SMs=" << d.multi_processor_count
                  << ", VRAM=" << (d.total_vram_bytes / (1024 * 1024)) << " MiB"
                  << ", L2=" << (d.l2_cache_bytes / 1024) << " KiB"
                  << ", BW=" << d.peak_bandwidth_gbps << " GB/s"
                  << ", drv=" << d.driver_version << ")\n";
    }
#else
    std::cout << "CUDA Devices: disabled at build time\n";
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
    auto vk_devs = VkContext::enumerate_devices();
    std::cout << "Vulkan Devices (" << vk_devs.size() << "):\n";
    for (const auto& d : vk_devs) {
        std::cout << "  [" << d.device_index << "] " << d.name
                  << " (" << d.device_type
                  << ", subgroup=" << d.subgroup_size
                  << ", shmem=" << d.shared_mem_bytes << " B"
                  << ", fp16=" << (d.has_fp16_int8 ? "yes" : "no")
                  << ", coopmat=" << ((d.coop_mat.supported && d.coop_mat.fp16_in_fp32_acc) ? "16x16x16" : "no") << ")\n";
    }
#else
    std::cout << "Vulkan Devices: disabled at build time\n";
#endif
    return 0;
}

std::filesystem::path utf8_path(const std::string& path) {
    return std::filesystem::path(std::u8string(path.begin(), path.end()));
}

size_t estimate_arena_mb(const std::string& model_path) {
    if (model_path.empty()) return 512;
    std::error_code ec;
    auto sz = std::filesystem::file_size(utf8_path(model_path), ec);
    if (ec) return 1024;
    size_t file_mb = static_cast<size_t>(sz / (1024ULL * 1024ULL)) + 1;
    return std::min<size_t>(8192, std::max<size_t>(2048, file_mb * 3 + 768));
}

DType parse_kv_type(const std::string& value) {
    if (value == "f16") return DType::F16;
    if (value == "f32") return DType::F32;
    if (value == "q8_0") return DType::Q8_0;
    throw std::invalid_argument("Expected f32, f16 or q8_0 for --kv-type");
}

bool is_gguf(const std::string& path) {
    return utf8_path(path).extension() == ".gguf";
}

std::string model_filename(const std::string& path) {
    if (path.empty()) return "synthetic";
    auto name = utf8_path(path).filename().u8string();
    return std::string(name.begin(), name.end());
}

bool parse_sampling(std::string_view option, const std::string& value, SamplingParams& params) {
    size_t end = 0;
    if (option == "--temp") params.temperature = std::stof(value, &end);
    else if (option == "--top-k") params.top_k = std::stoi(value, &end);
    else if (option == "--top-p") params.top_p = std::stof(value, &end);
    else if (option == "--min-p") params.min_p = std::stof(value, &end);
    else if (option == "--repeat-penalty") params.repeat_penalty = std::stof(value, &end);
    else if (option == "--seed" && !value.starts_with("-")) params.seed = std::stoull(value, &end);
    else return false;
    if (end != value.size()) throw std::invalid_argument("Invalid value for " + std::string(option));
    return true;
}

bool load_tokenizer(BpeTokenizer& tok, const std::string& model_path) {
    if (is_gguf(model_path)) {
        GgufFile gf;
        return gf.load(model_path) && tok.load_from_gguf(gf);
    }
    auto path = (utf8_path(model_path).parent_path() / "tokenizer.json").u8string();
    return !model_path.empty() && tok.load_from_json(std::string(path.begin(), path.end()));
}

#ifdef FUSIONRT_ENABLE_CUDA
void load_cuda_model(CudaContext& ctx, FusionModel& model, const std::string& path, int max_seq,
                     const std::string& profile) {
    const auto device = DeviceProfile::load(profile);
    std::string err;
    bool ok = is_gguf(path) ? model.load_gguf(ctx, path, &err, max_seq)
                           : model.load_safetensors(ctx, path, &err, max_seq);
    if (!ok) throw std::runtime_error("Error loading model: " + err);
    model.set_device_profile(ctx, device);
}

int interactive_chat(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tok,
                     const std::string& system, const SamplingParams& sampling,
                     int steps, bool use_fused, bool use_graph, bool stop_on_eos) {
    std::vector<ChatMessage> messages;
    if (!system.empty()) messages.push_back({"system", system});
    std::cout << "fusionrt interactive chat (type 'exit' to quit)\n> " << std::flush;
    for (std::string line; std::getline(std::cin, line);) {
        if (line == "exit" || line == "quit") break;
        messages.push_back({"user", line});
        auto prompt = tok.encode(tok.format_chat_messages(messages));
        if (prompt.size() >= static_cast<size_t>(model.config().dense.max_seq_len)) {
            messages.pop_back();
            std::cerr << "Conversation exceeds the model context. Start a new chat or use --max-seq.\n";
            std::cout << "> " << std::flush;
            continue;
        }
        Utf8StreamDecoder utf8;
        std::string answer;
        auto emit = [&](const std::string& text) {
            answer += text;
            std::cout << text << std::flush;
        };
        GenerationOptions options;
        options.sampling = sampling;
        options.reuse_prefix = true;
        options.on_token = [&](int32_t id) {
            if (stop_on_eos && tok.is_end_of_generation(id)) return false;
            emit(utf8.push(tok.decode_token(id)));
            return true;
        };
        model.generate_cuda(ctx, prompt, steps, use_fused, use_graph, nullptr, false, 0, -1, options);
        emit(utf8.finish());
        messages.push_back({"assistant", std::move(answer)});
        std::cout << "\n> " << std::flush;
    }
    std::cout << '\n';
    return 0;
}
#endif

int cmd_daily(const std::string& cmd, int argc, char** argv) {
#ifndef FUSIONRT_ENABLE_CUDA
    (void)cmd;
    (void)argc;
    (void)argv;
    std::cerr << "CUDA backend was not enabled in this build.\n";
    return 1;
#else
    std::string path, system, profile, host = "127.0.0.1";
    int max_seq = 4096, steps = 512, port = 8080;
    std::optional<bool> graph;
    SamplingParams sampling;
    sampling.temperature = 0.7f;
    sampling.top_p = 0.9f;
    sampling.repeat_penalty = 1.1f;
    DType kv_dtype = DType::F32;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--graph" || arg == "--no-graph") {
            graph = arg == "--graph";
            continue;
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--model") path = value;
        else if (arg == "--max-seq") max_seq = std::stoi(value);
        else if (arg == "--kv-type") kv_dtype = parse_kv_type(value);
        else if (arg == "--profile") profile = value;
        else if (cmd == "serve" && arg == "--host") host = value;
        else if (cmd == "serve" && arg == "--port") port = std::stoi(value);
        else if (cmd == "chat" && arg == "--system") system = value;
        else if (cmd == "chat" && arg == "--max-tokens") steps = std::stoi(value);
        else if (cmd != "chat" || !parse_sampling(arg, value, sampling)) {
            throw std::invalid_argument("Unknown " + cmd + " option: " + arg);
        }
    }
    if (path.empty() || max_seq <= 0 || steps <= 0 || port < 1 || port > 65535) {
        throw std::invalid_argument("Expected --model and positive context, generation length and port");
    }
    sampling.validate();
    CudaContext ctx(0, is_gguf(path) ? 0 : estimate_arena_mb(path), 256);
    ctx.set_kv_dtype(kv_dtype);
    FusionModel model(HybridModelConfig{});
    load_cuda_model(ctx, model, path, max_seq, profile);
    const bool use_graph = graph.value_or(model.recommended_cuda_graph());
    if (cmd == "inspect") {
        std::map<DType, std::pair<uint64_t, uint64_t>> breakdown;
        uint64_t parameters = 0, disk_bytes = 0;
        auto count = [&](DType dtype, const std::vector<int64_t>& shape, uint64_t bytes) {
            const auto size = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<>{});
            parameters += size;
            disk_bytes += bytes;
            auto& entry = breakdown[dtype];
            entry.first += size;
            entry.second += bytes;
        };
        std::string err;
        if (is_gguf(path)) {
            GgufFile gguf;
            if (!gguf.load(path, &err)) throw std::runtime_error(err);
            for (const auto& t : gguf.tensors()) count(t.dtype, t.shape, t.nbytes);
        } else {
            SafetensorsFile sf;
            if (!sf.load(path, &err)) throw std::runtime_error(err);
            for (const auto& t : sf.tensors()) count(t.dtype, t.shape, t.end - t.begin);
        }
        const auto& cfg = model.config();
        const auto mem = model.inspect_cuda(ctx);
        std::cout << "model=" << model_filename(path)
                  << " architecture=" << (cfg.model_architecture.empty() ? cfg.arch : cfg.model_architecture)
                  << " layout=" << cfg.arch << " layers=" << cfg.num_layers << " max_seq=" << cfg.dense.max_seq_len
                  << "\nparameters=" << parameters << " weight_bytes=" << disk_bytes << '\n';
        for (const auto& [dtype, amounts] : breakdown) {
            std::cout << "  " << dtype_name(dtype) << " parameters=" << amounts.first << " bytes=" << amounts.second << '\n';
        }
        size_t device_weight_bytes = 0;
        for (const auto& [dtype, bytes] : mem.weight_bytes) {
            device_weight_bytes += bytes;
            std::cout << "  device_weights " << dtype_name(dtype) << " bytes=" << bytes << '\n';
        }
        std::cout << "device_weight_bytes=" << device_weight_bytes << '\n';
        std::cout << model.fused_ir_graph().summary()
                  << "\nstatic_scratch_peak_bytes=" << mem.static_scratch_bytes
                  << " kv_cache_dtype=" << dtype_name(kv_dtype) << " kv_cache_bytes=" << mem.kv_bytes
                  << " kv_cache_pages=" << mem.kv_pages
                  << " weight_arena_reserved_bytes=" << ctx.weight_arena().capacity()
                  << " ssm_cache_bytes=" << mem.ssm_bytes
                  << " persistent_bytes=" << mem.persistent_bytes
                  << " total_vram_estimate_bytes=" << mem.persistent_bytes + mem.kv_bytes + mem.static_scratch_bytes << '\n';
        return 0;
    }
    BpeTokenizer tokenizer;
    if (!load_tokenizer(tokenizer, path)) throw std::runtime_error("Model tokenizer could not be loaded");
    if (cmd == "chat") return interactive_chat(ctx, model, tokenizer, system, sampling, steps, true, use_graph, true);
    serve_http(ctx, model, tokenizer, model_filename(path), host, port, use_graph);
    return 0;
#endif
}

int cmd_bench_model(int argc, char** argv) {
#ifdef FUSIONRT_ENABLE_CUDA
    std::string path;
    int pp = 128, tg = 64, reps = 5, decode_prefix = 0, ubatch = 512;
    DType kv_dtype = DType::F32;
    std::optional<bool> graph_option;
    std::string profile, profile_ops;
    auto parse_int = [](const std::string& arg) {
        size_t end = 0;
        int value = std::stoi(arg, &end);
        if (end != arg.size() || value < 0) throw std::invalid_argument("Expected a nonnegative integer: " + arg);
        return value;
    };
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--graph" || a == "--no-graph") graph_option = a == "--graph";
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--profile-ops" && i + 1 < argc) profile_ops = argv[++i];
        else if (a == "--kv-type" && i + 1 < argc) kv_dtype = parse_kv_type(argv[++i]);
        else if (a == "--model" && i + 1 < argc) path = argv[++i];
        else if (a == "--pp" && i + 1 < argc) pp = parse_int(argv[++i]);
        else if (a == "--tg" && i + 1 < argc) tg = parse_int(argv[++i]);
        else if (a == "--ubatch" && i + 1 < argc) ubatch = parse_int(argv[++i]);
        else if (a == "--decode-prefix" && i + 1 < argc) decode_prefix = parse_int(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = parse_int(argv[++i]);
        else throw std::invalid_argument("Unknown or incomplete bench-model option: " + a);
    }
    if (path.empty() || reps == 0 || (pp == 0 && tg == 0)) {
        throw std::invalid_argument("bench-model requires --model, positive --reps and at least one nonzero token count");
    }
    if (ubatch <= 0 || ubatch > 512) throw std::invalid_argument("--ubatch must be between 1 and 512");
    if (decode_prefix > std::numeric_limits<int>::max() - tg || (decode_prefix != 0 && tg == 0)) {
        throw std::invalid_argument("--decode-prefix requires a positive generation count and a valid context length");
    }
    CudaContext ctx(0, is_gguf(path) ? 0 : estimate_arena_mb(path), 256);
    ctx.set_kv_dtype(kv_dtype);
    FusionModel model(HybridModelConfig{});
    ctx.set_prefill_batch(ubatch);
    const auto load_start = std::chrono::steady_clock::now();
    load_cuda_model(ctx, model, path, std::max(pp, decode_prefix + tg), profile);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    const bool graph = graph_option.value_or(model.recommended_cuda_graph());
    OpTrace trace;
    auto samples = model.bench_model_cuda(ctx, pp, tg, reps, graph, profile_ops.empty() ? nullptr : &trace, decode_prefix);
    if (!profile_ops.empty()) trace.write(profile_ops, model_filename(path));
    auto report = [&](const char* name, int tokens, double GenerationStats::* rate) {
        if (tokens == 0) return;
        std::vector<double> speeds;
        for (const auto& sample : samples) speeds.push_back(sample.*rate);
        std::sort(speeds.begin(), speeds.end());
        double median = (speeds[(reps - 1) / 2] + speeds[reps / 2]) * 0.5;
        double mean = std::accumulate(speeds.begin(), speeds.end(), 0.0) / reps;
        std::cout << name << tokens << " median=" << median << " tok/s mean=" << mean << " tok/s\n";
    };
    std::cout << "model=" << std::filesystem::path(path).filename().string()
              << " backend=cuda graph=" << (graph ? "true" : "false") << " reps=" << reps
              << " ubatch=" << model.prefill_batch_limit(ctx) << " load_ms=" << load_ms
              << '\n' << std::fixed << std::setprecision(2);
    report("pp", pp, &GenerationStats::prefill_tok_per_s);
    report("tg", tg, &GenerationStats::decode_tok_per_s);
    return 0;
#else
    (void)argc;
    (void)argv;
    std::cerr << "CUDA backend was not enabled in this build.\n";
    return 1;
#endif
}

int cmd_run(int argc, char** argv) {
    std::string arch = "hybrid";
    std::string dtype_s = "f32";
    std::string backend_s = "cuda";
    std::string model_path;
    std::string prompt = "Hello fusionrt";
    int steps = 16;
    int max_seq = 4096;
    int vk_dev = 0;
    std::optional<bool> graph_option;
    std::optional<DType> kv_dtype;
    SamplingParams sampling;
    std::string profile, profile_ops, layer_trace;
    bool stream = false, sampling_set = false;
    bool use_fused = true;
    bool chat_mode = false;
    bool ignore_eos = false;
    bool use_chatml = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--arch" && i + 1 < argc) arch = argv[++i];
        else if (a == "--dtype" && i + 1 < argc) dtype_s = argv[++i];
        else if (a == "--backend" && i + 1 < argc) backend_s = argv[++i];
        else if (a == "--vk-device" && i + 1 < argc) vk_dev = std::stoi(argv[++i]);
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (a == "--max-seq" && i + 1 < argc) max_seq = std::stoi(argv[++i]);
        else if (a == "--kv-type" && i + 1 < argc) kv_dtype = parse_kv_type(argv[++i]);
        else if (a == "--graph" || a == "--no-graph") graph_option = a == "--graph";
        else if (a == "--stream") stream = true;
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--profile-ops" && i + 1 < argc) profile_ops = argv[++i];
        else if (a == "--trace-layers" && i + 1 < argc) layer_trace = argv[++i];
        else if (a == "--unfused" || a == "--no-fuse") use_fused = false;
        else if (a == "--chat") chat_mode = true;
        else if (a == "--ignore-eos") ignore_eos = true;
        else if (a == "--chatml") use_chatml = true;
        else if (i + 1 < argc && parse_sampling(a, argv[i + 1], sampling)) {
            ++i;
            sampling_set = true;
        } else throw std::invalid_argument("Unknown or incomplete run option: " + a);
    }
    sampling.validate();
    if (kv_dtype && backend_s != "cuda") throw std::invalid_argument("--kv-type requires the CUDA backend");
    if (steps < 0 || max_seq <= 0) throw std::invalid_argument("Invalid --steps or --max-seq");

    DType wdt = DType::F32;
    if (dtype_s == "q8") wdt = DType::Q8_0;
    else if (dtype_s == "q4") wdt = DType::Q4_0;
    else if (dtype_s == "f16") wdt = DType::F16;
    else if (dtype_s == "bf16") wdt = DType::BF16;

    HybridModelConfig cfg;
    cfg.arch = arch;
    cfg.vocab_size = 512;
    cfg.num_layers = 4;
    cfg.dense.max_seq_len = max_seq;
    cfg.dense.weight_dtype = wdt;
    cfg.gdn.weight_dtype = wdt;
    cfg.moe.weight_dtype = wdt;

    FusionModel model(cfg);
    BpeTokenizer tok;
    tok.init_byte_fallback(cfg.vocab_size);
    size_t weight_mb = estimate_arena_mb(model_path);
    (void)weight_mb;

    auto load_tok_if_present = [&] {
        if (!load_tokenizer(tok, model_path)) tok.init_byte_fallback(model.config().vocab_size);
    };

    if (backend_s == "cuda") {
#ifndef FUSIONRT_ENABLE_CUDA
        std::cerr << "CUDA backend was not enabled in this build.\n";
        return 1;
#else
        CudaContext ctx(0, is_gguf(model_path) ? 0 : weight_mb, 256);
        if (kv_dtype) ctx.set_kv_dtype(*kv_dtype);
        if (!model_path.empty()) {
            load_cuda_model(ctx, model, model_path, max_seq, profile);
        } else {
            model.init_synthetic(ctx, 42);
            model.set_device_profile(ctx, DeviceProfile::load(profile));
        }
        load_tok_if_present();
        const bool use_graph = graph_option.value_or(model.recommended_cuda_graph()) &&
                               sampling.greedy() && profile_ops.empty();
        const bool stop_on_eos = !model_path.empty() && !ignore_eos;
        if (chat_mode) {
            return interactive_chat(ctx, model, tok, "", sampling, steps, use_fused, use_graph, stop_on_eos);
        }

        std::string final_prompt = use_chatml && tok.has_chatml()
            ? "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            : prompt;
        auto prompt_ids = tok.encode(final_prompt, true);
        if (prompt_ids.empty()) prompt_ids.push_back(1);
        OpTrace op_trace;
        LayerTrace trace;
        Utf8StreamDecoder utf8;
        GenerationOptions options;
        options.sampling = sampling;
        options.op_trace = profile_ops.empty() ? nullptr : &op_trace;
        options.layer_trace = layer_trace.empty() ? nullptr : &trace;
        if (stream || stop_on_eos) {
            options.on_token = [&](int32_t id) {
                if (stop_on_eos && tok.is_end_of_generation(id)) return false;
                if (stream) std::cerr << utf8.push(tok.decode_token(id)) << std::flush;
                return true;
            };
        }
        GenerationStats st;
        auto out_ids = model.generate_cuda(ctx, prompt_ids, steps, use_fused, use_graph, &st, false, 0,
                                           stop_on_eos ? tok.eos_id() : -1, options);
        if (stream) std::cerr << utf8.finish() << '\n';
        const auto name = model_filename(model_path);
        if (options.op_trace) op_trace.write(profile_ops, name);
        if (options.layer_trace) trace.write(layer_trace, name);
        const auto& ir_st = (use_fused ? model.fused_ir_graph() : model.raw_ir_graph()).stats;
        std::cout << "backend=cuda arch=" << model.config().arch
                  << " dtype=" << dtype_name(model.config().dense.weight_dtype)
                  << " fused=" << (use_fused ? "true" : "false")
                  << " graph=" << (use_graph ? "true" : "false")
                  << " ir_ops=" << ir_st.raw_nodes << "->" << ir_st.opt_nodes
                  << " prompt_toks=" << prompt_ids.size()
                  << " gen_toks=" << st.generated_tokens
                  << " prefill_ms=" << st.prefill_ms
                  << " prefill_tok/s=" << st.prefill_tok_per_s
                  << " decode_ms=" << st.decode_ms
                  << " decode_tok/s=" << st.decode_tok_per_s << "\n";
        std::cout << "output_ids:";
        for (int32_t id : out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::vector<int32_t> gen_only(out_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()), out_ids.end());
        std::cout << "output_text: " << tok.decode(gen_only) << "\n";
        return 0;
#endif
    } else if (backend_s == "vulkan") {
#ifndef FUSIONRT_ENABLE_VULKAN
        std::cerr << "Vulkan backend was not enabled in this build.\n";
        return 1;
#else
        if (sampling_set || stream || !profile.empty() || !profile_ops.empty() || !layer_trace.empty()) {
            throw std::invalid_argument("Sampling, streaming and profiles require the CUDA backend");
        }
        VkContext ctx(static_cast<uint32_t>(vk_dev), weight_mb, 256);
        if (!model_path.empty()) {
            std::string err;
            bool ok = (model_path.size() >= 5 && model_path.substr(model_path.size() - 5) == ".gguf")
                ? model.load_gguf_vulkan(ctx, model_path, &err, max_seq)
                : model.load_safetensors_vulkan(ctx, model_path, &err, max_seq);
            if (!ok) {
                std::cerr << "Error loading model on Vulkan: " << err << "\n";
                return 1;
            }
        } else {
            model.init_synthetic_vulkan(ctx, 42);
        }
        load_tok_if_present();
        int32_t eos_stop = (!model_path.empty() && !ignore_eos) ? tok.eos_id() : -1;
        auto format_chatml_turn = [&](const std::string& msg) {
            return (tok.has_chatml())
                ? ("<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n")
                : msg;
        };

        if (chat_mode) {
            model.reset_kv_and_state_vulkan(ctx);
            int cur_pos = 0;
            std::string line;
            std::cout << "fusionrt interactive chat (arch=" << model.config().arch
                      << ", backend=vulkan:" << vk_dev << ", type 'exit' to quit)\n> " << std::flush;
            while (std::getline(std::cin, line)) {
                if (line == "exit" || line == "quit") break;
                auto p_ids = tok.encode(format_chatml_turn(line), cur_pos == 0);
                GenerationStats st;
                auto out_ids = model.generate_vulkan(ctx, p_ids, steps, &st, true, cur_pos, eos_stop);
                cur_pos += static_cast<int>(out_ids.size());
                std::vector<int32_t> gen_only(out_ids.begin() + static_cast<std::ptrdiff_t>(p_ids.size()), out_ids.end());
                std::cout << tok.decode(gen_only)
                          << "\n[prefill=" << st.prefill_ms << "ms, decode=" << st.decode_ms << "ms]\n> " << std::flush;
            }
            return 0;
        }

        std::string final_prompt = use_chatml ? format_chatml_turn(prompt) : prompt;
        auto prompt_ids = tok.encode(final_prompt, true);
        GenerationStats st;
        auto out_ids = model.generate_vulkan(ctx, prompt_ids, steps, &st, false, 0, eos_stop);
        std::cout << "backend=vulkan:" << vk_dev << " device=\"" << ctx.info().name << "\""
                  << " arch=" << model.config().arch
                  << " dtype=" << dtype_name(model.config().dense.weight_dtype)
                  << " prompt_toks=" << prompt_ids.size()
                  << " gen_toks=" << st.generated_tokens
                  << " prefill_ms=" << st.prefill_ms
                  << " prefill_tok/s=" << st.prefill_tok_per_s
                  << " decode_ms=" << st.decode_ms
                  << " decode_tok/s=" << st.decode_tok_per_s << "\n";
        std::cout << "output_ids:";
        for (int32_t id : out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::vector<int32_t> gen_only(out_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()), out_ids.end());
        std::cout << "output_text: " << tok.decode(gen_only) << "\n";
        return 0;
#endif
    } else {
        std::cerr << "Unknown --backend: " << backend_s << "\n";
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    std::vector<std::string> utf8_args;
    std::vector<char*> utf8_ptrs;
    if (wargv && wargc > 0) {
        utf8_args.reserve(static_cast<size_t>(wargc));
        utf8_ptrs.reserve(static_cast<size_t>(wargc));
        for (int i = 0; i < wargc; ++i) {
            int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(len > 0 ? static_cast<size_t>(len - 1) : 0, '\0');
            if (len > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), len, nullptr, nullptr);
            utf8_args.push_back(std::move(s));
        }
        for (auto& s : utf8_args) utf8_ptrs.push_back(s.data());
        LocalFree(wargv);
        argc = wargc;
        argv = utf8_ptrs.data();
    }
#endif
    try {
        if (argc < 2) {
            print_usage();
            return 0;
        }
        std::string cmd = argv[1];
        if (cmd == "--version" || cmd == "-v") {
            std::cout << "fusionrt " << FUSIONRT_VERSION << "\n";
            return 0;
        }
        if (cmd == "info") return cmd_info();
        if (cmd == "run") return cmd_run(argc, argv);
        if (cmd == "inspect" || cmd == "chat" || cmd == "serve") return cmd_daily(cmd, argc, argv);
        if (cmd == "bench-model") return cmd_bench_model(argc, argv);
        if (cmd == "bench") {
            std::string suite = "all";
            std::string output = "bench_report.v1.json";
            int vk_dev = 0;
            for (int i = 2; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--suite" && i + 1 < argc) suite = argv[++i];
                else if (a == "--output" && i + 1 < argc) output = argv[++i];
                else if (a == "--vk-device" && i + 1 < argc) vk_dev = std::stoi(argv[++i]);
            }
            return bench::run_benchmark_suite(suite, output, vk_dev);
        }
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 2;
    }
}
