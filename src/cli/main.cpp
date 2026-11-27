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
using fusionrt::cli::parse_integer;

namespace {

void print_usage() {
    std::cout << "fusionrt v" << FUSIONRT_VERSION << " - Direct CUDA sm_120 + Vulkan LLM Inference Engine\n"
              << "Usage:\n"
              << "  fusionrt info\n"
              << "  fusionrt run [--arch dense|hybrid|moe] [--dtype f32|q8|q4] [--backend cuda|vulkan]\n"
              << "               [--vk-device 0] [--model <gguf|safetensors>] [--prompt \"text\"]\n"
              << "               [--steps <n>] [--max-seq 4096] [--graph|--no-graph] [--unfused] [--stream]\n"
              << "               [--temp 0] [--top-k 40] [--top-p 0.95] [--min-p 0.05] [--repeat-penalty 1] [--seed 42]\n"
              << "               [--profile <json>] [--profile-ops <json>] [--trace-layers <json>]\n"
              << "               [--json-grammar] [--spec-ngram <k> | --draft-model <gguf> [--draft-tokens 4]]\n"
              << "               [--mmproj <gguf> --image <path> ...] (use <image> to place images in the prompt)\n"
              << "               --steps defaults to 512 with --model and 16 for synthetic models.\n"
              << "  fusionrt inspect --model <path> [--max-seq 4096] [--device-memory]\n"
              << "  fusionrt chat --model <path> [--system <text>] [--temp 0.7] [--max-tokens 512]\n"
              << "  fusionrt serve --model <path> [--mmproj <gguf>] [--image <preset> ...]\n"
              << "                 [--host 127.0.0.1] [--port 8080] [--max-seq 4096] [--slots 1] [--profile <json>]\n"
              << "  fusionrt bench-model --model <path> [--pp 128] [--tg 64] [--decode-prefix 0] [--ubatch 512] [--reps 5] [--graph] [--profile-ops <json>]\n"
              << "  CUDA run, chat, serve, inspect and bench-model accept --kv-type {f32|f16|q8_0}.\n"
              << "  CUDA placement options: --kv-vram-mb <MiB> (0: auto), --host-kv-mb <MiB> (0: disabled),\n"
              << "                          --moe-gpu-experts <n> (-1: auto).\n"
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

struct CudaPlacementOptions {
    static constexpr size_t mib = 1024 * 1024;
    size_t vram = 0, host = 0;
    int experts = -1;
    bool specified = false;

    bool parse(std::string_view option, std::string_view value) {
        if (option != "--kv-vram-mb" && option != "--host-kv-mb" && option != "--moe-gpu-experts") return false;
        if (option == "--moe-gpu-experts") {
            experts = parse_integer(option, value);
            if (experts < -1) throw std::invalid_argument("Expected --moe-gpu-experts >= -1");
        } else {
            const size_t mb = parse_integer<size_t>(option, value);
            if (mb > std::numeric_limits<size_t>::max() / mib) throw std::invalid_argument("KV budget is too large");
            (option == "--kv-vram-mb" ? vram : host) = mb * mib;
        }
        specified = true;
        return true;
    }

#ifdef FUSIONRT_ENABLE_CUDA
    void apply(FusionModel& model) const {
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

const char* stop_reason(int generated, int steps, bool eos, bool context_full) {
    if (eos) return "eos";
    if (generated == steps) return "max_steps";
    return context_full ? "context_limit" : "json_complete";
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
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

int cmd_daily(const std::string& cmd, int argc, char** argv) {
    std::string path, system, profile, host = "127.0.0.1";
    std::string mmproj;
    std::vector<std::string> image_paths;
    int max_seq = 4096, steps = 512, port = 8080, slots = 1;
    std::optional<bool> graph;
    bool inspect_device = false;
    SamplingParams sampling;
    sampling.temperature = 0.7f;
    sampling.top_p = 0.9f;
    sampling.repeat_penalty = 1.1f;
    DType kv_dtype = DType::F32;
    CudaPlacementOptions placement;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--graph" || arg == "--no-graph") {
            graph = arg == "--graph";
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
        else if (arg == "--kv-type") kv_dtype = parse_kv_type(value);
        else if (arg == "--profile") profile = value;
        else if (placement.parse(arg, value)) continue;
        else if (cmd == "serve" && arg == "--host") host = value;
        else if (cmd == "serve" && arg == "--port") port = parse_integer(arg, value);
        else if (cmd == "serve" && arg == "--mmproj") mmproj = value;
        else if (cmd == "serve" && arg == "--image") image_paths.push_back(value);
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
    sampling.validate();
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
#ifndef FUSIONRT_ENABLE_CUDA
    std::cerr << "CUDA backend was not enabled in this build.\n";
    return 1;
#else
    CudaContext ctx(0, is_gguf(path) ? 0 : estimate_arena_mb(path), 256);
    ctx.set_kv_dtype(kv_dtype);
    FusionModel model(HybridModelConfig{});
    model.set_num_slots(slots);
    placement.apply(model);
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
    std::vector<Image> preset_images;
    for (const auto& image_path : image_paths) preset_images.push_back(load_image(image_path));
    serve_http(ctx, model, tokenizer, model_filename(path), host, port, graph, vision.get(), preset_images);
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
    CudaPlacementOptions placement;
    auto parse_int = [](std::string_view option, std::string_view arg) {
        const int value = parse_integer(option, arg);
        if (value < 0) throw std::invalid_argument("Expected a nonnegative integer for " + std::string(option));
        return value;
    };
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--graph" || a == "--no-graph") graph_option = a == "--graph";
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--profile-ops" && i + 1 < argc) profile_ops = argv[++i];
        else if (a == "--kv-type" && i + 1 < argc) kv_dtype = parse_kv_type(argv[++i]);
        else if (a == "--model" && i + 1 < argc) path = argv[++i];
        else if (a == "--pp" && i + 1 < argc) pp = parse_int(a, argv[++i]);
        else if (a == "--tg" && i + 1 < argc) tg = parse_int(a, argv[++i]);
        else if (a == "--ubatch" && i + 1 < argc) ubatch = parse_int(a, argv[++i]);
        else if (a == "--decode-prefix" && i + 1 < argc) decode_prefix = parse_int(a, argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = parse_int(a, argv[++i]);
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
    CudaContext ctx(0, is_gguf(path) ? 0 : estimate_arena_mb(path), 256);
    ctx.set_kv_dtype(kv_dtype);
    FusionModel model(HybridModelConfig{});
    placement.apply(model);
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
    std::string draft_path;
    std::string mmproj;
    std::vector<std::string> image_paths;
    std::string prompt = "Hello fusionrt";
    std::optional<int> steps_option;
    int spec_ngram = 0, draft_tokens = 4;
    int max_seq = 4096;
    int vk_dev = 0;
    std::optional<bool> graph_option;
    std::optional<DType> kv_dtype;
    CudaPlacementOptions placement;
    SamplingParams sampling;
    std::string profile, profile_ops, layer_trace;
    bool stream = false, sampling_set = false, json_grammar = false, draft_tokens_set = false;
    bool use_fused = true;
    bool chat_mode = false;
    bool ignore_eos = false;
    bool use_chatml = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--arch" && i + 1 < argc) arch = argv[++i];
        else if (a == "--dtype" && i + 1 < argc) dtype_s = argv[++i];
        else if (a == "--backend" && i + 1 < argc) backend_s = argv[++i];
        else if (a == "--vk-device" && i + 1 < argc) vk_dev = parse_integer(a, argv[++i]);
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--json-grammar") json_grammar = true;
        else if (a == "--spec-ngram" && i + 1 < argc) spec_ngram = parse_integer(a, argv[++i]);
        else if (a == "--draft-model" && i + 1 < argc) draft_path = argv[++i];
        else if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
        else if (a == "--image" && i + 1 < argc) image_paths.push_back(argv[++i]);
        else if (a == "--draft-tokens" && i + 1 < argc) {
            draft_tokens = parse_integer(a, argv[++i]);
            draft_tokens_set = true;
        }
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps_option = parse_integer(a, argv[++i]);
        else if (a == "--max-seq" && i + 1 < argc) max_seq = parse_integer(a, argv[++i]);
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
        else if (i + 1 < argc && placement.parse(a, argv[i + 1])) ++i;
        else if (i + 1 < argc && parse_sampling(a, argv[i + 1], sampling)) {
            ++i;
            sampling_set = true;
        } else throw std::invalid_argument("Unknown or incomplete run option: " + a);
    }
    const int steps = steps_option.value_or(model_path.empty() ? 16 : 512);
    sampling.validate();
    if (kv_dtype && backend_s != "cuda") throw std::invalid_argument("--kv-type requires the CUDA backend");
    if (placement.specified && backend_s != "cuda") throw std::invalid_argument("Placement options require the CUDA backend");
    if (steps < 0 || max_seq <= 0) throw std::invalid_argument("Invalid --steps or --max-seq");
    if (!image_paths.empty() && mmproj.empty()) throw std::invalid_argument("--image requires --mmproj");
    if (!mmproj.empty() && (backend_s != "cuda" || chat_mode))
        throw std::invalid_argument("--mmproj requires non-interactive CUDA run");
    if (!image_paths.empty() && (spec_ngram > 0 || !draft_path.empty()))
        throw std::invalid_argument("Image input requires standard decoding");
    if (spec_ngram < 0 || draft_tokens <= 0 || (draft_tokens_set && draft_path.empty())) {
        throw std::invalid_argument("Expected --spec-ngram >= 0 and --draft-tokens > 0 with --draft-model");
    }
    if (!draft_path.empty() && (!is_gguf(draft_path) || model_path.empty() || spec_ngram > 0)) {
        throw std::invalid_argument("--draft-model requires a GGUF draft, a target model, and no --spec-ngram");
    }
    if ((json_grammar || spec_ngram > 0 || !draft_path.empty()) && (backend_s != "cuda" || chat_mode)) {
        throw std::invalid_argument("JSON grammar and speculative decoding require non-interactive CUDA run");
    }

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
    if (model_path.empty()) tok.init_byte_fallback(cfg.vocab_size);
    size_t weight_mb = estimate_arena_mb(model_path);
    (void)weight_mb;

    if (backend_s == "cuda") {
#ifndef FUSIONRT_ENABLE_CUDA
        std::cerr << "CUDA backend was not enabled in this build.\n";
        return 1;
#else
        CudaContext ctx(0, is_gguf(model_path) ? 0 : weight_mb, 256);
        const auto load_start = std::chrono::steady_clock::now();
        placement.apply(model);
        if (kv_dtype) ctx.set_kv_dtype(*kv_dtype);
        if (!model_path.empty()) {
            load_tokenizer(tok, model_path);
            load_cuda_model(ctx, model, model_path, max_seq, profile);
        } else {
            model.init_synthetic(ctx, 42);
            model.set_device_profile(ctx, DeviceProfile::load(profile));
        }
        std::unique_ptr<VisionEncoder> vision;
        if (!mmproj.empty()) vision = std::make_unique<VisionEncoder>(ctx, mmproj, model.config().dense.hidden_size);
        std::unique_ptr<FusionModel> draft;
        if (!draft_path.empty()) {
            BpeTokenizer draft_tok;
            load_tokenizer(draft_tok, draft_path);
            if (!tok.has_compatible_vocab(draft_tok)) {
                throw std::invalid_argument("Draft tokenizer must have the same vocabulary and EOS as the target");
            }
            draft = std::make_unique<FusionModel>(HybridModelConfig{});
            std::string error;
            if (!draft->load_gguf(ctx, draft_path, &error, max_seq, true)) {
                throw std::runtime_error("Error loading draft model: " + error);
            }
        }
        const bool use_graph = graph_option.value_or(model.recommended_cuda_graph());
        const bool stop_on_eos = !model_path.empty() && !ignore_eos;
        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_start).count();
        if (chat_mode) {
            return interactive_chat(ctx, model, tok, "", sampling, steps, use_fused, graph_option, stop_on_eos);
        }

        if (!image_paths.empty() && prompt.find(image_placeholder) == std::string::npos) {
            std::string prefix;
            for (size_t i = 0; i < image_paths.size(); ++i) prefix += image_placeholder;
            prompt = prefix + prompt;
        }
        std::string final_prompt = use_chatml && tok.has_chatml()
            ? "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            : prompt;
        VisionPrompt prepared;
        if (image_paths.empty()) prepared.tokens = tok.encode(final_prompt, true);
        else {
            std::vector<Image> images;
            std::vector<const Image*> refs;
            for (const auto& path : image_paths) images.push_back(load_image(path));
            for (const auto& image : images) refs.push_back(&image);
            prepared = encode_vision_prompt(ctx, *vision, tok, final_prompt, refs, model.config().dense.max_seq_len);
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
        options.spec_ngram = spec_ngram;
        options.draft_model = draft.get();
        options.draft_tokens = draft_tokens;
        options.inputs_embeds = prepared.inputs_embeds;
        options.op_trace = profile_ops.empty() ? nullptr : &op_trace;
        options.layer_trace = layer_trace.empty() ? nullptr : &trace;
        options.warn_on_graph_fallback = graph_option.value_or(false);
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
        if (spec_ngram > 0 || draft) {
            const double rate = st.spec_proposed ? static_cast<double>(st.spec_accepted) / st.spec_proposed : 0.0;
            std::cout << " spec_proposed=" << st.spec_proposed << " spec_accepted=" << st.spec_accepted
                      << " accept_rate=" << rate;
        }
        std::cout << '\n';
        std::cout << "output_ids:";
        for (int32_t id : out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::vector<int32_t> gen_only(out_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()), out_ids.end());
        const auto output_text = tok.decode(gen_only);
        if (constraint) (void)JsonValue::parse(output_text);
        std::cout << "output_text: " << output_text << "\n";
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
            load_tokenizer(tok, model_path);
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
        if (graph_option.value_or(false))
            std::clog << "--graph requested, using eager decoding because the backend is Vulkan\n";
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
                std::vector<int32_t> gen_only(out_ids.end() - st.generated_tokens, out_ids.end());
                std::cout << tok.decode(gen_only)
                          << "\n[prefill=" << st.prefill_ms << "ms, decode=" << st.decode_ms << "ms]\n> " << std::flush;
            }
            return 0;
        }

        std::string final_prompt = use_chatml ? format_chatml_turn(prompt) : prompt;
        auto prompt_ids = tok.encode(final_prompt, true);
        GenerationStats st;
        auto out_ids = model.generate_vulkan(ctx, prompt_ids, steps, &st, false, 0, eos_stop);
        const bool eos = st.generated_tokens > 0 && out_ids.back() == eos_stop;
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
                  << " decode_tok/s=" << st.decode_tok_per_s << "\n";
        std::cout << "output_ids:";
        for (int32_t id : out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::vector<int32_t> gen_only(out_ids.end() - st.generated_tokens, out_ids.end());
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
