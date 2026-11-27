#include "fusionrt/bench.h"
#include "fusionrt/gguf.h"
#include "fusionrt/json.h"
#include "fusionrt/models.h"
#include "fusionrt/safetensors.h"
#include "fusionrt/server.h"
#include "fusionrt/tokenizer.h"
#include "fusionrt/vision.h"
#include "cli_args.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#include "../core/moe_route_trace.h"
#include "../core/moe_residency.h"
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
#include <random>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using namespace fusionrt;
using fusionrt::cli::parse_integer;

namespace {

void print_usage(std::string_view command = {}) {
    const auto show = [&](std::string_view name) { return command.empty() || command == name; };
    const auto sampling = "  [--temp N] [--top-k N] [--top-p N] [--min-p N] [--repeat-penalty N] [--seed N]\n";
    const auto placement = "  CUDA: [--kv-dtype auto|f32|f16|q8_0|k8v4] [--profile FILE] [--threads N] [--cpu-affinity spread]\n"
                           "  CUDA: [--kv-vram-mb MiB] [--host-kv-mb MiB]\n"
                           "  [--moe-gpu-experts N] [--moe-route-profile FILE] [--moe-dynamic-cache]\n";
    const auto vulkan = "  Vulkan: [--kv-dtype auto|f32|f16|q8_0] [--profile FILE]\n";
    const auto speculative = "  [--spec-ngram K | --draft-model GGUF | --spec-mtp] [--draft-tokens N|auto] [--spec-p-min N]\n"
                             "  --draft-tokens auto (or 0) requires --spec-mtp.\n";
    if (command.empty()) std::cout << "fusionrt " << FUSIONRT_VERSION
        << "\nUsage: fusionrt <info|run|chat|serve|inspect|bench-model|bench> [options]\n"
        << "Use fusionrt <command> --help for details. generate is an alias for run.\n";
    if (show("run")) std::cout << "fusionrt run [--model PATH] [--arch dense|hybrid|moe] [--dtype f32|f16|bf16|q4|q8]\n"
        << "  [--backend cuda|vulkan] [--vk-device N] [--prompt TEXT] [--steps N] [--max-seq N] [--chat]\n"
        << "  [--stream] [--ignore-eos] [--chatml] [--graph|--no-graph] [--unfused]\n"
        << "  [--json|--json-grammar] [--mmproj GGUF] [--image PATH]...\n"
        << "  CUDA: [--slots N] [--warmup N]\n"
        << "  CUDA: [--profile-ops FILE] [--trace-layers FILE] [--trace-moe] [--trace-moe-routes FILE] [--cpu-moe-f32]\n"
        << sampling << placement << vulkan << speculative << "  --steps defaults to 512 for real models and 16 for synthetic models.\n";
    if (show("chat")) std::cout << "fusionrt chat --model PATH [--system TEXT] [--max-tokens N] [--max-seq N] [--graph|--no-graph]\n"
        << sampling << placement;
    if (show("serve")) std::cout << "fusionrt serve --model PATH [--host HOST] [--port N] [--max-seq N] [--slots N]\n"
        << "  [--backend cuda|vulkan] [--vk-device N] [--mmproj GGUF] [--image PATH]...\n"
        << "  CUDA: [--graph|--no-graph] [--trace-moe-routes FILE]\n"
        << placement << vulkan << speculative;
    if (show("inspect")) std::cout << "fusionrt inspect --model PATH [--max-seq N] [--device-memory]\n"
        << "  CUDA placement flags require --device-memory.\n" << placement;
    if (show("bench-model")) std::cout << "fusionrt bench-model --model PATH [--pp N] [--tg N] [--decode-prefix N]\n"
        << "  [--backend cuda|vulkan] [--vk-device N] [--ubatch N] [--reps N] [--graph|--no-graph] [--profile-ops FILE]\n"
        << placement << vulkan << "  Vulkan benchmark speculation:\n" << speculative;
    if (show("bench")) std::cout << "fusionrt bench [--suite all|fusions|blocks|vulkan] [--output FILE] [--vk-device N]\n";
    if (command == "info") std::cout << "fusionrt info\n";
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

std::optional<DType> parse_kv_type(const std::string& value) {
    if (value == "auto") return std::nullopt;
    if (value == "f16") return DType::F16;
    if (value == "f32") return DType::F32;
    if (value == "q8_0") return DType::Q8_0;
    if (value == "k8v4") return DType::K8V4;
    throw std::invalid_argument("Expected auto, f32, f16, q8_0 or k8v4 for --kv-dtype");
}

struct SpeculativeArgs : cli::SpeculativeArgs {
    bool parse(std::string_view option, std::string_view value) {
        return cli::SpeculativeArgs::parse(option, option == "--draft-tokens" && value == "auto" ? "0" : value);
    }

    bool auto_draft_tokens() const { return draft_tokens == 0; }

#ifdef FUSIONRT_ENABLE_VULKAN
    void apply(FusionModel& target, FusionModel* draft) const {
        if (enabled()) target.set_vulkan_speculation(ngram, draft, draft_tokens, mtp, p_min);
    }
#endif

    void validate(bool has_target) const {
        if (!auto_draft_tokens()) {
            cli::SpeculativeArgs::validate(has_target);
            return;
        }
        if (!mtp) throw std::invalid_argument("--draft-tokens auto (or 0) requires --spec-mtp");
        auto fixed_depth = static_cast<const cli::SpeculativeArgs&>(*this);
        fixed_depth.draft_tokens = 1;
        fixed_depth.validate(has_target);
    }
};

struct CudaPlacementOptions {
    static constexpr size_t mib = 1024 * 1024;
    size_t vram = 0, host = 0;
    int experts = -1, threads = 0;
    bool specified = false, cuda_only = false, spread_affinity = false, dynamic_cache = false;
    std::string route_profile;

    bool parse_flag(std::string_view option) {
        if (option != "--moe-dynamic-cache") return false;
        return specified = dynamic_cache = true;
    }

    HybridModelConfig configure(HybridModelConfig config = {}) const {
        config.moe.route_profile = route_profile;
        config.moe.dynamic_cache = dynamic_cache;
        return config;
    }

    bool parse(std::string_view option, std::string_view value) {
        if (option == "--moe-route-profile") {
            if (value.empty() || value.starts_with("--")) throw std::invalid_argument("Expected a file for --moe-route-profile");
            route_profile = value;
            specified = true;
            return true;
        }
        if (option == "--cpu-affinity") {
            if (value != "spread" && value != "none") throw std::invalid_argument("Expected spread or none for --cpu-affinity");
            spread_affinity = value == "spread";
            specified = cuda_only = true;
            return true;
        }
        if (option != "--kv-vram-mb" && option != "--host-kv-mb" && option != "--moe-gpu-experts" && option != "--threads")
            return false;
        if (option == "--threads") {
            threads = parse_integer(option, value);
            if (threads < 0) throw std::invalid_argument("Expected --threads >= 0");
            specified = cuda_only = true;
            return true;
        }
        if (option == "--moe-gpu-experts") {
            experts = parse_integer(option, value);
            if (experts < -1) throw std::invalid_argument("Expected --moe-gpu-experts >= -1");
        } else {
            const size_t mb = parse_integer<size_t>(option, value);
            if (mb > std::numeric_limits<size_t>::max() / mib) throw std::invalid_argument("KV budget is too large");
            (option == "--kv-vram-mb" ? vram : host) = mb * mib;
            cuda_only = true;
        }
        specified = true;
        return true;
    }

#ifdef FUSIONRT_ENABLE_CUDA
    void apply(CudaContext& ctx, FusionModel& model) const {
        ctx.set_cpu_threads(threads, spread_affinity);
        model.set_kv_vram_budget_bytes(vram);
        model.set_kv_host_budget_bytes(host);
        model.set_moe_gpu_experts(experts);
    }
#endif
};

bool is_gguf(const std::string& path) {
    return utf8_path(path).extension() == ".gguf";
}

std::string model_filename(const std::string& path) {
    if (path.empty()) return "synthetic";
    auto name = utf8_path(path).filename().u8string();
    return std::string(name.begin(), name.end());
}

std::vector<int64_t> per_layer_integers(const GgufFile& file, const std::string& key) {
    const auto it = file.metadata().find(key);
    if (it == file.metadata().end()) return {};
    if (const auto* values = std::get_if<std::vector<int32_t>>(&it->second)) return {values->begin(), values->end()};
    if (const auto* values = std::get_if<std::vector<uint32_t>>(&it->second)) return {values->begin(), values->end()};
    if (const auto* values = std::get_if<std::vector<bool>>(&it->second)) return {values->begin(), values->end()};
    return {};
}

bool parse_sampling(std::string_view option, const std::string& value, SamplingParams& params) {
    if (option == "--top-k") {
        params.top_k = parse_integer(option, value);
        return true;
    }
    if (option == "--seed") {
        params.seed = parse_integer<uint64_t>(option, value);
        return true;
    }
    size_t end = 0;
    if (option == "--temp") params.temperature = std::stof(value, &end);
    else if (option == "--top-p") params.top_p = std::stof(value, &end);
    else if (option == "--min-p") params.min_p = std::stof(value, &end);
    else if (option == "--repeat-penalty") params.repeat_penalty = std::stof(value, &end);
    else return false;
    if (end != value.size()) throw std::invalid_argument("Invalid value for " + std::string(option));
    return true;
}

#if defined(FUSIONRT_ENABLE_CUDA) || defined(FUSIONRT_ENABLE_VULKAN)
void load_tokenizer(BpeTokenizer& tok, const std::string& model_path) {
    std::string error;
    if (is_gguf(model_path)) {
        GgufFile gf;
        if (gf.load(model_path, &error)) {
            if (tok.load_from_gguf(gf)) return;
            error = "missing or empty tokenizer.ggml.tokens metadata";
        }
        throw std::runtime_error("Cannot load tokenizer from GGUF " + model_path + ": " + error);
    }
    const auto model = utf8_path(model_path);
    std::error_code ec;
    const auto path = ((std::filesystem::is_directory(model, ec) ? model : model.parent_path()) / "tokenizer.json").u8string();
    const std::string expected(path.begin(), path.end());
    if (tok.load_from_json(expected, &error)) return;
    if (error.empty()) error = "expected a supported model.vocab object";
    throw std::runtime_error("Cannot load tokenizer at " + expected + ": " + error);
}

std::vector<Image> load_images(const std::vector<std::string>& paths) {
    std::vector<Image> images;
    images.reserve(paths.size());
    for (const auto& path : paths) images.push_back(load_image(path));
    return images;
}

std::string format_run_prompt(std::string prompt, const BpeTokenizer& tok, bool chatml,
                              size_t images, std::string_view architecture) {
    if (images && prompt.find(image_placeholder) == std::string::npos) {
        std::string prefix;
        for (size_t i = 0; i < images; ++i) prefix += image_placeholder;
        prompt.insert(0, prefix);
    }
    if (images && architecture == "gemma4" && !prompt.starts_with("<|turn>"))
        return tok.format_chat_messages({{"user", prompt}});
    if (chatml && tok.has_chatml()) return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

std::vector<const Image*> image_refs(std::span<const Image> images) {
    std::vector<const Image*> refs;
    refs.reserve(images.size());
    for (const auto& image : images) refs.push_back(&image);
    return refs;
}

const char* stop_reason(int generated, int steps, bool eos, bool context_full) {
    if (eos) return "eos";
    if (generated == steps) return "max_steps";
    return context_full ? "context_limit" : "json_complete";
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
std::unique_ptr<FusionModel> load_draft_model(CudaContext& ctx, const BpeTokenizer& target,
                                              const cli::SpeculativeArgs& args, int max_seq) {
    if (args.draft_path.empty()) return {};
    BpeTokenizer tokenizer;
    load_tokenizer(tokenizer, args.draft_path);
    if (!target.has_compatible_vocab(tokenizer))
        throw std::invalid_argument("Draft tokenizer must have the same vocabulary and EOS as the target");
    auto draft = std::make_unique<FusionModel>(HybridModelConfig{});
    std::string error;
    if (!draft->load_gguf(ctx, args.draft_path, &error, max_seq, true))
        throw std::runtime_error("Error loading draft model: " + error);
    return draft;
}

void load_cuda_model(CudaContext& ctx, FusionModel& model, const std::string& path, int max_seq,
                     const std::string& profile) {
    model.set_device_profile(ctx, DeviceProfile::load(profile));
    std::string err;
    bool ok = is_gguf(path) ? model.load_gguf(ctx, path, &err, max_seq)
                           : model.load_safetensors(ctx, path, &err, max_seq);
    if (!ok) throw std::runtime_error("Error loading model: " + err);
}

int interactive_chat(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tok,
                     const std::string& system, const SamplingParams& sampling,
                     int steps, bool use_fused, std::optional<bool> graph, bool stop_on_eos) {
    std::vector<ChatMessage> messages;
    if (!system.empty()) messages.push_back({"system", system});
    std::cout << "fusionrt interactive chat (type 'exit' to quit)\n> " << std::flush;
    for (std::string line; std::getline(std::cin, line);) {
        if (line == "exit" || line == "quit") break;
        messages.push_back({"user", line});
        auto prompt = tok.encode(tok.format_chat_messages(messages));
        const int context = model.config().dense.max_seq_len;
        if (prompt.size() >= static_cast<size_t>(context)) {
            messages.pop_back();
            std::cerr << prompt_context_error(prompt.size(), context).what() << ". Start a new chat or use --max-seq.\n";
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
        options.warn_on_graph_fallback = graph.value_or(false);
        options.on_token = [&](int32_t id) {
            if (stop_on_eos && tok.is_end_of_generation(id)) return false;
            emit(utf8.push(tok.decode_token(id)));
            return true;
        };
        model.generate_cuda(ctx, prompt, steps, use_fused, graph.value_or(model.recommended_cuda_graph()),
                            nullptr, false, 0, -1, options);
        emit(utf8.finish());
        messages.push_back({"assistant", std::move(answer)});
        std::cout << "\n> " << std::flush;
    }
    std::cout << '\n';
    return 0;
}
#endif

struct WeightSummary {
    std::map<DType, std::pair<uint64_t, uint64_t>> by_type;
    uint64_t parameters = 0, bytes = 0;

    void add(DType type, const std::vector<int64_t>& shape, uint64_t size) {
        const auto count = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<>{});
        parameters += count;
        bytes += size;
        auto& entry = by_type[type];
        entry.first += count;
        entry.second += size;
    }

    void print() const {
        std::cout << "parameters=" << parameters << " weight_bytes=" << bytes << '\n';
        for (const auto& [type, amounts] : by_type) {
            std::cout << "  " << dtype_name(type) << " parameters=" << amounts.first << " bytes=" << amounts.second << '\n';
        }
    }
};

#ifdef FUSIONRT_ENABLE_VULKAN
void load_vulkan_model(VkContext& ctx, FusionModel& model, const std::string& path, int max_seq);

struct VulkanDraft {
    std::unique_ptr<VkContext> context;
    std::unique_ptr<FusionModel> model;
};

VulkanDraft load_vulkan_draft(VkContext& ctx, const BpeTokenizer& target, const SpeculativeArgs& args, int max_seq) {
    VulkanDraft draft;
    if (!args.draft_path.empty()) {
        BpeTokenizer tokenizer;
        load_tokenizer(tokenizer, args.draft_path);
        if (!target.has_compatible_vocab(tokenizer))
            throw std::invalid_argument("Draft tokenizer must have the same vocabulary and EOS as the target");
        draft.context = std::make_unique<VkContext>(ctx.info().device_index, 0, 0);
        draft.context->set_prefill_batch(ctx.prefill_batch());
        draft.context->set_device_profile(ctx.device_profile());
        draft.model = std::make_unique<FusionModel>(HybridModelConfig{});
        load_vulkan_model(*draft.context, *draft.model, args.draft_path, max_seq);
    }
    return draft;
}
#endif

int cmd_daily(const std::string& cmd, int argc, char** argv) {
    std::string path, system, profile, host = "127.0.0.1";
#if defined(FUSIONRT_ENABLE_VULKAN) && !defined(FUSIONRT_ENABLE_CUDA)
    std::string backend = "vulkan";
#else
    std::string backend = "cuda";
#endif
    std::string mmproj, route_trace_path;
    std::vector<std::string> image_paths;
    int max_seq = 4096, steps = 512, port = 8080, slots = 1;
    std::optional<int> vk_device;
    std::optional<bool> graph;
    bool inspect_device = false, kv_dtype_specified = false;
    SamplingParams sampling;
    sampling.temperature = 0.7f;
    sampling.top_p = 0.9f;
    sampling.repeat_penalty = 1.1f;
    std::optional<DType> kv_dtype;
    CudaPlacementOptions placement;
    SpeculativeArgs speculative;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (placement.parse_flag(arg)) continue;
        if (arg == "--graph" || arg == "--no-graph") {
            if (cmd == "inspect") throw std::invalid_argument("inspect does not accept " + arg);
            graph = arg == "--graph";
            continue;
        }
        if (cmd == "serve" && arg == "--spec-mtp") {
            speculative.mtp = true;
            continue;
        }
        if (cmd == "inspect" && arg == "--device-memory") {
            inspect_device = true;
            continue;
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--model") path = value;
        else if (arg == "--max-seq") max_seq = parse_integer(arg, value);
        else if (arg == "--kv-type" || arg == "--kv-dtype") {
            kv_dtype = parse_kv_type(value);
            kv_dtype_specified = true;
        }
        else if (arg == "--profile") profile = value;
        else if (placement.parse(arg, value)) continue;
        else if (cmd == "serve" && arg == "--backend") backend = value;
        else if (cmd == "serve" && arg == "--vk-device") vk_device = parse_integer(arg, value);
        else if (cmd == "serve" && arg == "--host") host = value;
        else if (cmd == "serve" && arg == "--port") port = parse_integer(arg, value);
        else if (cmd == "serve" && arg == "--mmproj") mmproj = value;
        else if (cmd == "serve" && arg == "--trace-moe-routes") route_trace_path = value;
        else if (cmd == "serve" && arg == "--image") image_paths.push_back(value);
        else if (cmd == "serve" && speculative.parse(arg, value)) continue;
        else if (cmd == "serve" && arg == "--slots") {
            slots = parse_integer(arg, value);
            if (slots < 1) throw std::invalid_argument("Expected --slots >= 1");
        }
        else if (cmd == "chat" && arg == "--system") system = value;
        else if (cmd == "chat" && arg == "--max-tokens") steps = parse_integer(arg, value);
        else if (cmd != "chat" || !parse_sampling(arg, value, sampling)) {
            throw std::invalid_argument("Unknown " + cmd + " option: " + arg);
        }
    }
    if (path.empty() || max_seq <= 0 || steps <= 0 || port < 1 || port > 65535) {
        throw std::invalid_argument("Expected --model and positive context, generation length and port");
    }
    if (cmd == "inspect" && !inspect_device && (kv_dtype_specified || !profile.empty() || placement.specified))
        throw std::invalid_argument("inspect device options require --device-memory");
    sampling.validate();
    speculative.validate(!path.empty());
    if (!image_paths.empty() && mmproj.empty()) throw std::invalid_argument("--image requires --mmproj");
    GgufFile header;
    const bool inspect_gguf = cmd == "inspect" && is_gguf(path);
    if (inspect_gguf && !inspect_device) {
        const auto start = std::chrono::steady_clock::now();
        std::string error;
        if (!header.load(path, &error, false)) throw std::runtime_error(error);
        if (header.find_tensor("v.patch_embd.weight")) {
            const auto vision = VisionInfo::from_gguf(header);
            std::cout << "model=" << model_filename(path) << " architecture=" << vision.architecture
                      << " projector=" << vision.projector << " layout=vision"
                      << " patch_size=" << vision.patch_size << " image_size=" << vision.image_size
                      << " hidden_size=" << vision.hidden_size << " heads=" << vision.heads
                      << " layers=" << vision.layers << " output_size=" << vision.output_size
                      << " spatial_merge=" << vision.spatial_merge << '\n';
            return 0;
        }
        const auto arch = header.get_string("general.architecture", "llama");
        const auto prefix = arch + ".";
        const auto hidden = header.get_int(prefix + "embedding_length");
        const auto heads = header.get_int(prefix + "attention.head_count");
        const auto layers = header.get_int(prefix + "block_count");
        const auto experts = header.get_int(prefix + "expert_count");
        const bool mla = arch == "deepseek2";
        const bool gemma4 = arch == "gemma4";
        auto layer_kv = gemma4 ? per_layer_integers(header, prefix + "attention.head_count_kv") : std::vector<int64_t>{};
        const auto swa = gemma4 ? per_layer_integers(header, prefix + "attention.sliding_window_pattern") : std::vector<int64_t>{};
        const auto shared = gemma4 ? header.get_int(prefix + "attention.shared_kv_layers") : 0;
        if (!layer_kv.empty()) {
            if (static_cast<int64_t>(layer_kv.size()) != layers || (!swa.empty() && swa.size() != layer_kv.size()))
                throw std::invalid_argument("Invalid Gemma 4 per-layer attention metadata");
            int64_t last_source[2]{};
            for (size_t i = 0; i < layer_kv.size(); ++i) {
                const bool sliding = swa.empty() ? header.get_bool(prefix + "attention.sliding_window_pattern", false) : swa[i] != 0;
                if (static_cast<int64_t>(i) < layers - shared) last_source[sliding] = layer_kv[i];
                else if (layer_kv[i] == 0) layer_kv[i] = last_source[sliding];
            }
        }
        const auto kv_heads = mla ? 1 : layer_kv.empty() ? header.get_int(prefix + "attention.head_count_kv", heads)
                                                       : layer_kv.front();
        auto head_dim = header.get_int(prefix + "attention.key_length", hidden / std::max<int64_t>(1, heads));
        if (mla) head_dim = header.get_int(prefix + "attention.key_length_mla", head_dim);
        const auto qk_norm = header.find_tensor("blk.0.attn_q_norm.weight") && header.find_tensor("blk.0.attn_k_norm.weight");
        std::cout << "model=" << model_filename(path) << " architecture=" << arch
                  << " layout=" << (mla ? "mla" : experts ? "moe" : header.find_tensor("blk.0.ssm_conv1d.weight") ? "hybrid" : "dense")
                  << " layers=" << layers << " max_seq=" << max_seq
                  << " trained_context=" << header.get_int(prefix + "context_length") << '\n'
                  << "gguf_version=" << header.version() << " file_bytes=" << header.file_bytes()
                  << " tensors=" << header.tensors().size() << " hidden_size=" << hidden
                  << " attention_heads=" << heads << " kv_heads=" << kv_heads
                  << " head_dim=" << head_dim
                  << " qk_norm=" << (qk_norm ? "true" : "false")
                  << " rope_theta=" << header.get_float(prefix + "rope.freq_base", 10000.0) << '\n';
        if (experts) {
            std::cout << "experts=" << experts << " top_k=" << header.get_int(prefix + "expert_used_count")
                      << " expert_width=" << header.get_int(prefix + "expert_feed_forward_length")
                      << " norm_topk_prob=" << (header.get_bool(prefix + "expert_weights_norm",
                             header.get_bool(prefix + "norm_topk_prob", !mla)) ? "true" : "false") << '\n';
        }
        if (mla) {
            const auto rank = header.get_int(prefix + "attention.kv_lora_rank");
            const auto rope = header.get_int(prefix + "rope.dimension_count");
            const auto gate = header.get_int(prefix + "expert_gating_func", 1);
            std::cout << "q_lora_rank=" << header.get_int(prefix + "attention.q_lora_rank", 0)
                      << " kv_lora_rank=" << rank << " d_rope=" << rope << " d_nope=" << head_dim - rope
                      << " d_v=" << header.get_int(prefix + "attention.value_length_mla",
                                                   header.get_int(prefix + "attention.value_length"))
                      << " latent_width=" << rank + rope
                      << " leading_dense=" << header.get_int(prefix + "leading_dense_block_count",
                          experts ? 0 : header.get_int(prefix + "block_count"))
                      << " shared_experts=" << header.get_int(prefix + "expert_shared_count")
                      << " expert_gating=" << (gate == 2 ? "sigmoid" : "softmax")
                      << " expert_groups=" << header.get_int(prefix + "expert_group_count", 1)
                      << " expert_groups_used=" << header.get_int(prefix + "expert_group_used_count",
                          header.get_int(prefix + "expert_group_count", 1))
                      << " expert_weights_scale=" << header.get_float(prefix + "expert_weights_scale", 1.0)
                      << " rope_scaling=" << header.get_string(prefix + "rope.scaling.type", "none")
                      << " kv_cache_dtype=" << dtype_name(DType::F32) << '\n';
        }
        if (gemma4) {
            const auto sliding_layers = swa.empty() ? (header.get_bool(prefix + "attention.sliding_window_pattern", false) ? layers : 0)
                                                    : std::count(swa.begin(), swa.end(), 1);
            std::cout << "global_layers=" << layers - sliding_layers
                      << " sliding_layers=" << sliding_layers
                      << " sliding_head_dim=" << header.get_int(prefix + "attention.key_length_swa")
                      << " sliding_window=" << header.get_int(prefix + "attention.sliding_window")
                      << " shared_kv_layers=" << shared
                      << " ple_width=" << header.get_int(prefix + "embedding_length_per_layer_input") << '\n';
            if (!layer_kv.empty()) {
                std::cout << "kv_heads_per_layer=";
                for (size_t i = 0; i < layer_kv.size(); ++i) {
                    if (i != 0) std::cout << ',';
                    std::cout << layer_kv[i];
                }
                std::cout << '\n';
            }
        }
        WeightSummary sizes;
        for (const auto& tensor : header.tensors()) sizes.add(tensor.dtype, tensor.shape, tensor.nbytes);
        sizes.print();
        std::cout << "inspect_ms=" << std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() << '\n';
        return 0;
    }
    if (cmd == "serve") {
        if (backend != "cuda" && backend != "vulkan") throw std::invalid_argument("Unknown --backend: " + backend);
        if (vk_device && backend != "vulkan") throw std::invalid_argument("--vk-device requires the Vulkan backend");
        if (backend == "vulkan") {
#ifndef FUSIONRT_ENABLE_VULKAN
            throw std::invalid_argument("Vulkan backend was not enabled in this build");
#else
            const int vk_dev = vk_device.value_or(0);
            if (vk_dev < 0) throw std::invalid_argument("Expected --vk-device >= 0");
            if (placement.cuda_only || slots != 1 || !route_trace_path.empty())
                throw std::invalid_argument("KV budgets, CPU thread settings, multiple slots and MoE route tracing require CUDA");
            if (graph.value_or(false)) std::clog << "--graph requested, using eager decoding on Vulkan\n";
            VkContext ctx(static_cast<uint32_t>(vk_dev), 0, 0);
            ctx.set_kv_dtype(kv_dtype);
            ctx.set_device_profile(DeviceProfile::load(profile));
            FusionModel model(placement.configure());
            model.set_moe_gpu_experts(ctx, placement.experts);
            model.enable_native_mtp(ctx, speculative.mtp);
            BpeTokenizer tokenizer;
            load_tokenizer(tokenizer, path);
            const auto draft = load_vulkan_draft(ctx, tokenizer, speculative, max_seq);
            load_vulkan_model(ctx, model, path, max_seq);
            speculative.apply(model, draft.model.get());
            std::unique_ptr<VulkanVisionEncoder> vision;
            if (!mmproj.empty())
                vision = std::make_unique<VulkanVisionEncoder>(ctx, mmproj, model.config().dense.hidden_size);
            const auto preset_images = load_images(image_paths);
            serve_http(ctx, model, tokenizer, model_filename(path), host, port, vision.get(), preset_images);
            return 0;
#endif
        }
    }
#ifndef FUSIONRT_ENABLE_CUDA
    std::cerr << "CUDA backend was not enabled in this build.\n";
    return 1;
#else
    CudaContext ctx(0, 0, 256);
    ctx.set_kv_dtype(kv_dtype);
    FusionModel model(placement.configure());
    model.set_num_slots(slots);
    placement.apply(ctx, model);
    model.enable_native_mtp(speculative.mtp);
    load_cuda_model(ctx, model, path, max_seq, profile);
    if (cmd == "inspect") {
        WeightSummary sizes;
        std::string err;
        if (inspect_gguf) {
            if (!header.load(path, &err, false)) throw std::runtime_error(err);
            for (const auto& t : header.tensors()) sizes.add(t.dtype, t.shape, t.nbytes);
        } else {
            SafetensorsFile sf;
            if (!sf.load(path, &err)) throw std::runtime_error(err);
            for (const auto& t : sf.tensors()) sizes.add(t.dtype, t.shape, t.end - t.begin);
        }
        const auto& cfg = model.config();
        const auto mem = model.inspect_cuda(ctx);
        std::cout << "model=" << model_filename(path)
                  << " architecture=" << (cfg.model_architecture.empty() ? cfg.arch : cfg.model_architecture)
                  << " layout=" << cfg.arch << " layers=" << cfg.num_layers << " max_seq=" << cfg.dense.max_seq_len << '\n';
        sizes.print();
        size_t device_weight_bytes = 0;
        for (const auto& [dtype, bytes] : mem.weight_bytes) {
            device_weight_bytes += bytes;
            std::cout << "  device_weights " << dtype_name(dtype) << " bytes=" << bytes << '\n';
        }
        std::cout << "device_weight_bytes=" << device_weight_bytes << '\n';
        std::cout << model.fused_ir_graph().summary()
                  << "\nstatic_scratch_peak_bytes=" << mem.static_scratch_bytes
                  << " kv_cache_dtype=" << dtype_name(model.cuda_kv_dtype()) << " kv_cache_bytes=" << mem.kv_bytes
                  << " kv_cache_pages=" << mem.kv_pages
                  << " kv_cache_reserved_bytes=" << mem.kv_reserved_bytes
                  << " host_kv_bytes=" << mem.host_kv_bytes << " host_kv_pages=" << mem.host_kv_pages
                  << " weight_arena_reserved_bytes=" << ctx.weight_arena().capacity()
                  << " ssm_cache_bytes=" << mem.ssm_bytes
                  << " persistent_bytes=" << mem.persistent_bytes
                  << " total_vram_estimate_bytes=" << mem.persistent_bytes + mem.kv_reserved_bytes + mem.static_scratch_bytes << '\n';
        return 0;
    }
    BpeTokenizer tokenizer;
    load_tokenizer(tokenizer, path);
    if (cmd == "chat") return interactive_chat(ctx, model, tokenizer, system, sampling, steps, true, graph, true);
    std::unique_ptr<VisionEncoder> vision;
    if (!mmproj.empty()) vision = std::make_unique<VisionEncoder>(ctx, mmproj, model.config().dense.hidden_size);
    auto draft = load_draft_model(ctx, tokenizer, speculative, max_seq);
    const auto preset_images = load_images(image_paths);
    std::unique_ptr<MoeRouteTrace> route_trace;
    if (!route_trace_path.empty()) route_trace = std::make_unique<MoeRouteTrace>(route_trace_path);
    ctx.moe_route_trace = route_trace.get();
    serve_http(ctx, model, tokenizer, model_filename(path), host, port, graph, vision.get(), preset_images,
               {speculative.ngram, draft.get(), speculative.draft_tokens, speculative.mtp, speculative.p_min,
                speculative.auto_draft_tokens()});
    return 0;
#endif
}

#ifdef FUSIONRT_ENABLE_VULKAN
void load_vulkan_model(VkContext& ctx, FusionModel& model, const std::string& path, int max_seq) {
    std::string error;
    const bool ok = is_gguf(path) ? model.load_gguf_vulkan(ctx, path, &error, max_seq)
                                  : model.load_safetensors_vulkan(ctx, path, &error, max_seq);
    if (!ok) throw std::runtime_error("Error loading model on Vulkan: " + error);
}

std::vector<GenerationStats> bench_model_vulkan(VkContext& ctx, FusionModel& model, int pp, int tg,
                                                 int reps, int decode_prefix, OpTrace* trace, bool speculative) {
    using Clock = std::chrono::steady_clock;
    if (std::max(pp, decode_prefix + tg) > model.config().dense.max_seq_len)
        throw std::invalid_argument("Benchmark exceeds model context");
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> pick(0, model.config().vocab_size - 1);
    std::vector<int32_t> prompt(pp), steps(tg), prefix(decode_prefix);
    for (auto* ids : {&prompt, &steps, &prefix})
        std::generate(ids->begin(), ids->end(), [&] { return pick(rng); });

    const auto prefill = [&](const std::vector<int32_t>& ids, GenerationStats* stats) {
        if (!ids.empty()) model.generate_vulkan(ctx, ids, 0, stats, true);
    };
    const auto decode = [&] {
        for (int pos = 0; pos < tg; ++pos)
            model.forward_token_vulkan(ctx, steps[pos], decode_prefix + pos);
    };
    std::vector<GenerationStats> samples(reps);
    for (int rep = -1; rep < reps && pp > 0; ++rep) {
        model.reset_kv_and_state_vulkan(ctx);
        prefill(prompt, rep < 0 ? nullptr : &samples[rep]);
    }
    for (int rep = -1; rep < reps && tg > 0; ++rep) {
        if (speculative) {
            GenerationStats decode_stats;
            model.generate_vulkan(ctx, prefix, tg, &decode_stats);
            if (rep >= 0) {
                const auto& prefill_stats = samples[rep];
                decode_stats.prefill_ms = prefill_stats.prefill_ms;
                decode_stats.prefill_tok_per_s = prefill_stats.prefill_tok_per_s;
                samples[rep] = decode_stats;
            }
            continue;
        }
        model.reset_kv_and_state_vulkan(ctx);
        prefill(prefix, nullptr);
        const auto begin = Clock::now();
        decode();
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (rep >= 0) {
            samples[rep].generated_tokens = tg;
            samples[rep].decode_ms = ms;
            samples[rep].decode_tok_per_s = 1000.0 * tg / ms;
        }
    }
    if (trace) {
        trace->mode = tg > 0 ? "decode" : "prefill";
        model.reset_kv_and_state_vulkan(ctx);
        if (tg > 0) prefill(prefix, nullptr);
        ctx.set_op_trace(trace);
        if (tg > 0) decode();
        else prefill(prompt, nullptr);
        ctx.set_op_trace(nullptr);
    }
    return samples;
}
#endif

int cmd_bench_model(int argc, char** argv) {
    std::string path, backend = "cuda";
    int vk_dev = 0;
    int pp = 128, tg = 64, reps = 5, decode_prefix = 0, ubatch = 512;
    std::optional<DType> kv_dtype;
    std::optional<bool> graph_option;
    std::string profile, profile_ops;
    CudaPlacementOptions placement;
    SpeculativeArgs speculative;
    auto parse_int = [](std::string_view option, std::string_view arg) {
        const int value = parse_integer(option, arg);
        if (value < 0) throw std::invalid_argument("Expected a nonnegative integer for " + std::string(option));
        return value;
    };
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (placement.parse_flag(a)) continue;
        if (a == "--graph" || a == "--no-graph") graph_option = a == "--graph";
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--profile-ops" && i + 1 < argc) profile_ops = argv[++i];
        else if ((a == "--kv-type" || a == "--kv-dtype") && i + 1 < argc) kv_dtype = parse_kv_type(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--vk-device" && i + 1 < argc) vk_dev = parse_int(a, argv[++i]);
        else if (a == "--model" && i + 1 < argc) path = argv[++i];
        else if (a == "--pp" && i + 1 < argc) pp = parse_int(a, argv[++i]);
        else if (a == "--tg" && i + 1 < argc) tg = parse_int(a, argv[++i]);
        else if (a == "--ubatch" && i + 1 < argc) ubatch = parse_int(a, argv[++i]);
        else if (a == "--decode-prefix" && i + 1 < argc) decode_prefix = parse_int(a, argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = parse_int(a, argv[++i]);
        else if (a == "--spec-mtp") speculative.mtp = true;
        else if (i + 1 < argc && speculative.parse(a, argv[i + 1])) ++i;
        else if (i + 1 < argc && placement.parse(a, argv[i + 1])) ++i;
        else throw std::invalid_argument("Unknown or incomplete bench-model option: " + a);
    }
    if (path.empty() || reps == 0 || (pp == 0 && tg == 0)) {
        throw std::invalid_argument("bench-model requires --model, positive --reps and at least one nonzero token count");
    }
    if (ubatch <= 0 || ubatch > 512) throw std::invalid_argument("--ubatch must be between 1 and 512");
    if (decode_prefix > std::numeric_limits<int>::max() - tg || (decode_prefix != 0 && tg == 0)) {
        throw std::invalid_argument("--decode-prefix requires a positive generation count and a valid context length");
    }
    speculative.validate(true);
    if (speculative.enabled() && (backend != "vulkan" || tg == 0 || !profile_ops.empty()))
        throw std::invalid_argument("Speculative benchmark requires Vulkan generation without operation profiling");
    const int prefix_len = speculative.enabled() ? std::max(1, decode_prefix) : decode_prefix;
    if (prefix_len > std::numeric_limits<int>::max() - tg) throw std::invalid_argument("Benchmark context is too large");
    const auto max_seq = std::max(pp, prefix_len + tg);
    std::vector<GenerationStats> samples;
    double load_ms = 0.0;
    bool graph_used = false;
    OpTrace trace;
    auto* op_trace = profile_ops.empty() ? nullptr : &trace;
    if (backend == "cuda") {
#ifdef FUSIONRT_ENABLE_CUDA
        CudaContext ctx(0, 0, 256);
        ctx.set_kv_dtype(kv_dtype);
        FusionModel model(placement.configure());
        placement.apply(ctx, model);
        ctx.set_prefill_batch(ubatch);
        const auto load_start = std::chrono::steady_clock::now();
        load_cuda_model(ctx, model, path, max_seq, profile);
        load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
        samples = model.bench_model_cuda(ctx, pp, tg, reps, graph_option.value_or(model.recommended_cuda_graph()),
                                         op_trace, decode_prefix);
        graph_used = samples.front().used_cuda_graph;
        if (graph_option.value_or(false) && !graph_used) {
            const auto& layers = model.moe_layers();
            const bool cold = std::any_of(layers.begin(), layers.end(), [](const auto& layer) {
                return layer.cfg.gpu_resident_experts < layer.cfg.num_experts;
            });
            std::clog << "--graph requested, benchmark used eager because "
                      << (cold ? "some MoE experts are CPU-resident" : "CUDA Graph capture failed") << '\n';
        }
        ubatch = model.prefill_batch_limit(ctx);
#else
        throw std::runtime_error("CUDA backend was not enabled in this build");
#endif
    } else if (backend == "vulkan") {
#ifdef FUSIONRT_ENABLE_VULKAN
        if (placement.cuda_only) throw std::invalid_argument("KV budgets and CPU thread settings require CUDA");
        VkContext ctx(static_cast<uint32_t>(vk_dev), 0, 0);
        ctx.set_kv_dtype(kv_dtype);
        ctx.set_prefill_batch(ubatch);
        ctx.set_device_profile(DeviceProfile::load(profile));
        FusionModel model(placement.configure());
        model.set_moe_gpu_experts(ctx, placement.experts);
        model.enable_native_mtp(ctx, speculative.mtp);
        const auto load_start = std::chrono::steady_clock::now();
        BpeTokenizer tokenizer;
        if (!speculative.draft_path.empty()) load_tokenizer(tokenizer, path);
        const auto draft = load_vulkan_draft(ctx, tokenizer, speculative, max_seq);
        load_vulkan_model(ctx, model, path, max_seq);
        speculative.apply(model, draft.model.get());
        load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
        if (graph_option.value_or(false))
            std::clog << "--graph requested, using eager decoding because the backend is Vulkan\n";
        samples = bench_model_vulkan(ctx, model, pp, tg, reps, prefix_len, op_trace, speculative.enabled());
        ubatch = ctx.prefill_batch();
#else
        throw std::runtime_error("Vulkan backend was not enabled in this build");
#endif
    } else {
        throw std::invalid_argument("Unknown backend: " + backend);
    }
    if (op_trace) trace.write(profile_ops, model_filename(path));
    auto report = [&](const char* name, int tokens, double GenerationStats::* rate) {
        if (tokens == 0) return;
        std::vector<double> speeds;
        for (const auto& sample : samples) speeds.push_back(sample.*rate);
        std::sort(speeds.begin(), speeds.end());
        double median = (speeds[(reps - 1) / 2] + speeds[reps / 2]) * 0.5;
        double mean = std::accumulate(speeds.begin(), speeds.end(), 0.0) / reps;
        std::cout << name << tokens << " median=" << median << " tok/s mean=" << mean << " tok/s\n";
    };
    std::cout << "model=" << model_filename(path)
              << " backend=" << backend << " graph=" << (graph_used ? "true" : "false") << " reps=" << reps
              << " ubatch=" << ubatch << " load_ms=" << load_ms
              << '\n' << std::fixed << std::setprecision(2);
    report("pp", pp, &GenerationStats::prefill_tok_per_s);
    report("tg", tg, &GenerationStats::decode_tok_per_s);
    return 0;
}

int cmd_run(int argc, char** argv) {
    std::string arch = "hybrid";
    std::string dtype_s = "f32";
#if defined(FUSIONRT_ENABLE_VULKAN) && !defined(FUSIONRT_ENABLE_CUDA)
    std::string backend_s = "vulkan";
#else
    std::string backend_s = "cuda";
#endif
    std::string model_path;
    std::string mmproj;
    std::vector<std::string> image_paths;
    std::string prompt = "Hello fusionrt";
    std::optional<int> steps_option;
    SpeculativeArgs speculative;
    int max_seq = 4096, slots = 1, warmup = 0;
    int vk_dev = 0;
    std::optional<bool> graph_option;
    std::optional<DType> kv_dtype;
    CudaPlacementOptions placement;
    SamplingParams sampling;
    std::string profile, profile_ops, layer_trace, route_trace_path;
    bool stream = false, json_grammar = false, trace_moe = false, cpu_moe_f32 = false;
    bool use_fused = true;
    bool chat_mode = false;
    bool ignore_eos = false;
    bool use_chatml = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (placement.parse_flag(a)) continue;
        if (a == "--arch" && i + 1 < argc) arch = argv[++i];
        else if (a == "--dtype" && i + 1 < argc) dtype_s = argv[++i];
        else if (a == "--backend" && i + 1 < argc) backend_s = argv[++i];
        else if (a == "--vk-device" && i + 1 < argc) vk_dev = parse_integer(a, argv[++i]);
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--json" || a == "--json-grammar") json_grammar = true;
        else if (a == "--spec-mtp") speculative.mtp = true;
        else if (i + 1 < argc && speculative.parse(a, argv[i + 1])) ++i;
        else if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
        else if (a == "--image" && i + 1 < argc) image_paths.push_back(argv[++i]);
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps_option = parse_integer(a, argv[++i]);
        else if (a == "--max-seq" && i + 1 < argc) max_seq = parse_integer(a, argv[++i]);
        else if (a == "--slots" && i + 1 < argc) slots = parse_integer(a, argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = parse_integer(a, argv[++i]);
        else if (a == "--trace-moe") trace_moe = true;
        else if (a == "--trace-moe-routes" && i + 1 < argc) route_trace_path = argv[++i];
        else if (a == "--cpu-moe-f32") cpu_moe_f32 = true;
        else if ((a == "--kv-type" || a == "--kv-dtype") && i + 1 < argc) kv_dtype = parse_kv_type(argv[++i]);
        else if (a == "--graph" || a == "--no-graph") graph_option = a == "--graph";
        else if (a == "--stream") stream = true;
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--profile-ops" && i + 1 < argc) profile_ops = argv[++i];
        else if (a == "--trace-layers" && i + 1 < argc) layer_trace = argv[++i];
        else if (a == "--unfused" || a == "--no-fuse") use_fused = false;
        else if (a == "--chat") chat_mode = true;
        else if (a == "--ignore-eos") ignore_eos = true;
        else if (a == "--chatml") use_chatml = true;
        else if (i + 1 < argc && placement.parse(a, argv[i + 1])) ++i;
        else if (i + 1 < argc && parse_sampling(a, argv[i + 1], sampling)) {
            ++i;
        } else throw std::invalid_argument("Unknown or incomplete run option: " + a);
    }
    const int steps = steps_option.value_or(model_path.empty() ? 16 : 512);
    sampling.validate();
    if (placement.cuda_only && backend_s != "cuda") throw std::invalid_argument("KV budgets and CPU thread settings require CUDA");
    if (steps < 0 || max_seq <= 0) throw std::invalid_argument("Invalid --steps or --max-seq");
    if (slots < 1 || warmup < 0) throw std::invalid_argument("Expected --slots >= 1 and --warmup >= 0");
    if ((slots != 1 || warmup || trace_moe || !route_trace_path.empty() || cpu_moe_f32) && (backend_s != "cuda" || chat_mode))
        throw std::invalid_argument("--slots, --warmup, --trace-moe and --cpu-moe-f32 require non-interactive CUDA run");
    if (trace_moe && speculative.enabled()) throw std::invalid_argument("--trace-moe requires standard decoding");
    if (!image_paths.empty() && mmproj.empty()) throw std::invalid_argument("--image requires --mmproj");
    if (!mmproj.empty() && chat_mode)
        throw std::invalid_argument("--mmproj requires non-interactive run");
    if (!image_paths.empty() && speculative.enabled())
        throw std::invalid_argument("Image input requires standard decoding");
    speculative.validate(!model_path.empty());
    if (json_grammar && chat_mode) throw std::invalid_argument("JSON grammar requires non-interactive run");
    if (speculative.enabled() && chat_mode)
        throw std::invalid_argument("Speculative decoding requires non-interactive run");

    DType wdt = DType::F32;
    if (dtype_s == "q8") wdt = DType::Q8_0;
    else if (dtype_s == "q4") wdt = DType::Q4_0;
    else if (dtype_s == "f16") wdt = DType::F16;
    else if (dtype_s == "bf16") wdt = DType::BF16;
    else if (dtype_s != "f32") throw std::invalid_argument("Unknown --dtype: " + dtype_s);

    HybridModelConfig cfg;
    cfg.arch = arch;
    cfg.vocab_size = 512;
    cfg.num_layers = 4;
    cfg.dense.max_seq_len = max_seq;
    cfg.dense.weight_dtype = wdt;
    cfg.gdn.weight_dtype = wdt;
    cfg.moe.weight_dtype = wdt;

    FusionModel model(placement.configure(cfg));
    BpeTokenizer tok;
    if (model_path.empty()) tok.init_byte_fallback(cfg.vocab_size);

    if (backend_s == "cuda") {
#ifndef FUSIONRT_ENABLE_CUDA
        std::cerr << "CUDA backend was not enabled in this build.\n";
        return 1;
#else
        CudaContext ctx(0, model_path.empty() ? 512 : 0, 256);
        const auto load_start = std::chrono::steady_clock::now();
        placement.apply(ctx, model);
        ctx.set_cpu_moe_q8(!cpu_moe_f32);
        model.set_num_slots(slots);
        ctx.set_kv_dtype(kv_dtype);
        std::vector<int32_t> text_ids;
        if (!model_path.empty()) {
            load_tokenizer(tok, model_path);
            if (image_paths.empty() && !chat_mode) {
                text_ids = tok.encode(format_run_prompt(prompt, tok, use_chatml, 0, {}), true);
                const auto requested = std::max(size_t{1}, text_ids.size()) + std::max(steps, warmup);
                model.set_kv_commitment_tokens(static_cast<int>(std::min(requested, static_cast<size_t>(max_seq))));
            }
            model.enable_native_mtp(speculative.mtp);
            load_cuda_model(ctx, model, model_path, max_seq, profile);
        } else {
            model.init_synthetic(ctx, 42);
            model.set_device_profile(ctx, DeviceProfile::load(profile));
        }
        std::unique_ptr<VisionEncoder> vision;
        if (!mmproj.empty()) vision = std::make_unique<VisionEncoder>(ctx, mmproj, model.config().dense.hidden_size);
        auto draft = load_draft_model(ctx, tok, speculative, max_seq);
        const bool use_graph = graph_option.value_or(model.recommended_cuda_graph());
        const bool stop_on_eos = !model_path.empty() && !ignore_eos;
        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_start).count();
        if (chat_mode) {
            return interactive_chat(ctx, model, tok, "", sampling, steps, use_fused, graph_option, stop_on_eos);
        }

        const auto final_prompt = format_run_prompt(prompt, tok, use_chatml, image_paths.size(),
                                                    model.config().model_architecture);
        VisionPrompt prepared;
        double vision_ms = 0;
        if (image_paths.empty()) prepared.tokens = text_ids.empty() ? tok.encode(final_prompt, true) : std::move(text_ids);
        else {
            const auto image_start = std::chrono::steady_clock::now();
            const auto images = load_images(image_paths);
            prepared = encode_vision_prompt(ctx, *vision, tok, final_prompt, image_refs(images),
                                            model.config().dense.max_seq_len);
            vision_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - image_start).count();
        }
        auto& prompt_ids = prepared.tokens;
        if (prompt_ids.empty()) prompt_ids.push_back(1);
        OpTrace op_trace;
        LayerTrace trace;
        Utf8StreamDecoder utf8;
        std::optional<JsonConstraint> constraint;
        if (json_grammar) constraint.emplace(tok, model.config().vocab_size);
        GenerationOptions options;
        options.sampling = sampling;
        options.json_constraint = constraint ? &*constraint : nullptr;
        options.spec_ngram = speculative.ngram;
        options.spec_mtp = speculative.mtp;
        options.spec_p_min = speculative.p_min;
        options.draft_model = draft.get();
        options.draft_tokens = speculative.draft_tokens;
        options.auto_draft_tokens = speculative.auto_draft_tokens();
        options.inputs_embeds = prepared.inputs_embeds;
        options.op_trace = profile_ops.empty() ? nullptr : &op_trace;
        options.layer_trace = layer_trace.empty() ? nullptr : &trace;
        options.warn_on_graph_fallback = graph_option.value_or(false);
        if (warmup > 0) {
            auto warm_options = options;
            warm_options.op_trace = nullptr;
            warm_options.layer_trace = nullptr;
            model.generate_cuda(ctx, prompt_ids, warmup, use_fused, use_graph, nullptr, false, 0, -1, warm_options);
        }
        CudaMoeTiming moe_timing(trace_moe ? model.config().num_layers : 0);
        std::unique_ptr<MoeRouteTrace> route_trace;
        if (!route_trace_path.empty()) route_trace = std::make_unique<MoeRouteTrace>(route_trace_path);
        ctx.moe_route_trace = route_trace.get();
        using Clock = std::chrono::steady_clock;
        Clock::time_point prev_token;
        std::vector<double> decode_samples;
        if (warmup || trace_moe) decode_samples.reserve(steps);
        if (stream || stop_on_eos || warmup || trace_moe) {
            options.on_token = [&](int32_t id) {
                if (warmup || trace_moe) {
                    if (trace_moe) moe_timing.collect();
                    const auto now = Clock::now();
                    if (prev_token != Clock::time_point{})
                        decode_samples.push_back(std::chrono::duration<double, std::milli>(now - prev_token).count());
                    prev_token = now;
                    if (trace_moe) ctx.moe_timing = &moe_timing;
                }
                if (stop_on_eos && tok.is_end_of_generation(id)) return false;
                if (stream) std::cerr << utf8.push(tok.decode_token(id)) << std::flush;
                return true;
            };
        }
        GenerationStats st;
        auto out_ids = model.generate_cuda(ctx, prompt_ids, steps, use_fused, use_graph, &st, false, 0,
                                           stop_on_eos ? tok.eos_id() : -1, options);
        ctx.moe_timing = nullptr;
        if (stream) std::cerr << utf8.finish() << '\n';
        const auto name = model_filename(model_path);
        if (options.op_trace) op_trace.write(profile_ops, name);
        if (options.layer_trace) trace.write(layer_trace, name);
        const auto& ir_st = (use_fused ? model.fused_ir_graph() : model.raw_ir_graph()).stats;
        const bool eos = st.generated_tokens > 0 && (stop_on_eos || constraint) && tok.is_end_of_generation(out_ids.back());
        std::cout << "backend=cuda arch=" << model.config().arch
                  << " dtype=" << dtype_name(model.config().dense.weight_dtype)
                  << " fused=" << (use_fused ? "true" : "false")
                  << " graph=" << (st.used_cuda_graph ? "true" : "false")
                  << " ir_ops=" << ir_st.raw_nodes << "->" << ir_st.opt_nodes
                  << " prompt_toks=" << prompt_ids.size()
                  << " load_ms=" << load_ms
                  << " gen_toks=" << st.generated_tokens
                  << " stop_reason=" << stop_reason(st.generated_tokens, steps, eos, out_ids.size() >=
                                                  static_cast<size_t>(model.config().dense.max_seq_len))
                  << " prefill_ms=" << st.prefill_ms
                  << " prefill_tok/s=" << st.prefill_tok_per_s
                  << " decode_ms=" << st.decode_ms
                  << " decode_tok/s=" << st.decode_tok_per_s;
        if (!decode_samples.empty()) {
            std::sort(decode_samples.begin(), decode_samples.end());
            auto percentile = [&](size_t p) { return decode_samples[(decode_samples.size() * p + 99) / 100 - 1]; };
            const auto kv = model.cuda_cache_memory();
            std::cout << " decode_p50_ms=" << percentile(50) << " decode_p95_ms=" << percentile(95)
                      << " kv_reserved_bytes=" << kv.kv_reserved_bytes << " kv_live_bytes=" << kv.kv_bytes;
        }
        if (!image_paths.empty()) {
            int64_t image_tokens = 0;
            for (const auto& embedding : prepared.inputs_embeds) image_tokens += embedding.rows.shape[0];
            std::cout << " image_tokens=" << image_tokens << " vision_ms=" << vision_ms;
        }
        if (speculative.enabled()) {
            const double rate = st.spec_proposed ? static_cast<double>(st.spec_accepted) / st.spec_proposed : 0.0;
            std::cout << " spec_proposed=" << st.spec_proposed << " spec_accepted=" << st.spec_accepted
                      << " accept_rate=" << rate;
            if (speculative.mtp) {
                std::cout << " mtp_cutoffs=" << st.mtp_cutoffs << " mtp_kv_reused=" << st.mtp_reused_kv_rows
                          << " mtp_kv_rewound=" << st.mtp_rewound_kv_rows
                          << " mtp_graph_verify=" << st.mtp_verify_graph_steps << " mtp_graph_draft=" << st.mtp_draft_graph_steps;
                for (size_t i = 0; i < st.mtp_proposed_by_depth.size() && st.mtp_proposed_by_depth[i]; ++i)
                    std::cout << " mtp_depth" << i + 1 << '=' << st.mtp_accepted_by_depth[i]
                              << '/' << st.mtp_checked_by_depth[i] << '/' << st.mtp_proposed_by_depth[i];
            }
        }
        std::cout << '\n';
        std::cout << "output_ids:";
        for (int32_t id : out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::vector<int32_t> gen_only(out_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()), out_ids.end());
        const auto output_text = tok.decode(gen_only);
        if (constraint) (void)JsonValue::parse(output_text);
        std::cout << "output_text: " << output_text << "\n";
        if (trace_moe) {
            const auto& layers = model.moe_layers();
            for (size_t i = 0; i < layers.size(); ++i) {
                const auto& layer = moe_timing.layers()[i];
                const auto& cache = layers[i].dynamic_cache;
                if (layer.hot || layer.cold || cache) {
                    std::cout << "moe_layer=" << layers[i].model_layer_index << " hot=" << layer.hot
                              << " cold=" << layer.cold << " d2h_wait_ms=" << layer.input_wait_ms
                              << " cpu_ms=" << layer.cpu_ms << " exposed_gpu_gap_ms=" << layer.gpu_gap_ms;
                    if (cache) std::cout << " cache_moves=" << cache->moves() << " cache_move_bytes=" << cache->moved_bytes();
                    std::cout << '\n';
                }
            }
        }
        return 0;
#endif
    } else if (backend_s == "vulkan") {
#ifndef FUSIONRT_ENABLE_VULKAN
        std::cerr << "Vulkan backend was not enabled in this build.\n";
        return 1;
#else
        if (!profile_ops.empty() || !layer_trace.empty()) {
            throw std::invalid_argument("Operator/layer tracing requires the CUDA backend for run");
        }
        if (vk_dev < 0) throw std::invalid_argument("Expected --vk-device >= 0");
        VkContext ctx(static_cast<uint32_t>(vk_dev), 0, 0);
        ctx.set_kv_dtype(kv_dtype);
        ctx.set_device_profile(DeviceProfile::load(profile));
        model.set_moe_gpu_experts(ctx, placement.experts);
        model.enable_native_mtp(ctx, speculative.mtp);
        if (!model_path.empty()) load_tokenizer(tok, model_path);
        const auto draft = load_vulkan_draft(ctx, tok, speculative, max_seq);
        if (model_path.empty()) model.init_synthetic_vulkan(ctx, 42);
        else load_vulkan_model(ctx, model, model_path, max_seq);
        speculative.apply(model, draft.model.get());
        if (graph_option.value_or(false))
            std::clog << "--graph requested, using eager decoding because the backend is Vulkan\n";
        const bool stop_on_eos = !model_path.empty() && !ignore_eos;
        const int32_t eos_stop = stop_on_eos ? tok.eos_id() : -1;
        Utf8StreamDecoder utf8;
        std::optional<JsonConstraint> constraint;
        if (json_grammar) constraint.emplace(tok, model.config().vocab_size);
        GenerationOptions options;
        options.sampling = sampling;
        options.json_constraint = constraint ? &*constraint : nullptr;
        if (stream || stop_on_eos) {
            options.on_token = [&](int32_t id) {
                if (stop_on_eos && tok.is_end_of_generation(id)) return false;
                if (stream) std::cerr << utf8.push(tok.decode_token(id)) << std::flush;
                return true;
            };
        }
        auto finish_stream = [&] {
            if (stream) std::cerr << utf8.finish() << '\n';
        };

        if (chat_mode) {
            model.reset_kv_and_state_vulkan(ctx);
            int cur_pos = 0;
            std::string line;
            std::cout << "fusionrt interactive chat (arch=" << model.config().arch
                      << ", backend=vulkan:" << vk_dev << ", type 'exit' to quit)\n> " << std::flush;
            while (std::getline(std::cin, line)) {
                if (line == "exit" || line == "quit") break;
                auto p_ids = tok.encode(format_run_prompt(line, tok, true, 0, {}), cur_pos == 0);
                GenerationStats st;
                auto out_ids = model.generate_vulkan(ctx, p_ids, steps, &st, true, cur_pos, eos_stop, options);
                finish_stream();
                cur_pos += static_cast<int>(out_ids.size());
                std::vector<int32_t> gen_only(out_ids.end() - st.generated_tokens, out_ids.end());
                std::cout << tok.decode(gen_only)
                          << "\n[prefill=" << st.prefill_ms << "ms, decode=" << st.decode_ms << "ms]\n> " << std::flush;
            }
            return 0;
        }

        const auto final_prompt = format_run_prompt(prompt, tok, use_chatml, image_paths.size(),
                                                    model.config().model_architecture);
        VulkanVisionPrompt prepared;
        std::unique_ptr<VulkanVisionEncoder> vision;
        if (!mmproj.empty())
            vision = std::make_unique<VulkanVisionEncoder>(ctx, mmproj, model.config().dense.hidden_size);
        double vision_ms = 0;
        if (image_paths.empty()) prepared.tokens = tok.encode(final_prompt, true);
        else {
            const auto image_start = std::chrono::steady_clock::now();
            const auto images = load_images(image_paths);
            prepared = encode_vision_prompt_vulkan(ctx, *vision, tok, final_prompt, image_refs(images),
                                                   model.config().dense.max_seq_len);
            vision_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - image_start).count();
        }
        options.inputs_embeds = prepared.inputs_embeds;
        const auto& prompt_ids = prepared.tokens;
        GenerationStats st;
        auto out_ids = model.generate_vulkan(ctx, prompt_ids, steps, &st, false, 0, eos_stop, options);
        finish_stream();
        const bool eos = st.generated_tokens > 0 && (stop_on_eos || constraint) && tok.is_end_of_generation(out_ids.back());
        std::cout << "backend=vulkan:" << vk_dev << " device=\"" << ctx.info().name << "\""
                  << " arch=" << model.config().arch
                  << " dtype=" << dtype_name(model.config().dense.weight_dtype)
                  << " prompt_toks=" << prompt_ids.size()
                  << " gen_toks=" << st.generated_tokens
                  << " stop_reason=" << stop_reason(st.generated_tokens, steps, eos, out_ids.size() >=
                                                  static_cast<size_t>(model.config().dense.max_seq_len))
                  << " prefill_ms=" << st.prefill_ms
                  << " prefill_tok/s=" << st.prefill_tok_per_s
                  << " decode_ms=" << st.decode_ms
                  << " decode_tok/s=" << st.decode_tok_per_s;
        if (speculative.enabled()) {
            const auto rate = st.spec_proposed ? static_cast<double>(st.spec_accepted) / st.spec_proposed : 0.0;
            std::cout << " spec_proposed=" << st.spec_proposed << " spec_accepted=" << st.spec_accepted
                      << " spec_accept_rate=" << rate;
        }
        if (!prepared.inputs_embeds.empty()) {
            int64_t image_tokens = 0;
            for (const auto& embedding : prepared.inputs_embeds) image_tokens += embedding.host_rows.desc.shape[0];
            std::cout << " image_tokens=" << image_tokens << " vision_ms=" << vision_ms;
        }
        std::cout << '\n';
        std::cout << "output_ids:";
        for (int32_t id : out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::vector<int32_t> gen_only(out_ids.end() - st.generated_tokens, out_ids.end());
        const auto output_text = tok.decode(gen_only);
        if (constraint) (void)JsonValue::parse(output_text);
        std::cout << "output_text: " << output_text << "\n";
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
        const auto is_help = [](std::string_view arg) { return arg == "-h" || arg == "--help"; };
        if (is_help(cmd) || (argc >= 3 && is_help(argv[2]))) {
            print_usage(is_help(cmd) ? "" : cmd == "generate" ? "run" : cmd);
            return 0;
        }
        if (cmd == "--version" || cmd == "-v") {
            std::cout << "fusionrt " << FUSIONRT_VERSION << "\n";
            return 0;
        }
        if (cmd == "info") return cmd_info();
        if (cmd == "run" || cmd == "generate") return cmd_run(argc, argv);
        if (cmd == "inspect" || cmd == "chat" || cmd == "serve") return cmd_daily(cmd, argc, argv);
        if (cmd == "bench-model") return cmd_bench_model(argc, argv);
        if (cmd == "bench") {
            std::string suite = "all";
            std::string output = "bench_report.v1.json";
            std::optional<int> vk_dev;
            for (int i = 2; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--suite" && i + 1 < argc) suite = argv[++i];
                else if (a == "--output" && i + 1 < argc) output = argv[++i];
                else if (a == "--vk-device" && i + 1 < argc) vk_dev = parse_integer(a, argv[++i]);
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
