#include "http_transport.h"
#include "fusionrt/server.h"
#include "fusionrt/vision.h"
#include "agent.h"
#include "http_request.h"

#include <algorithm>
#include <csignal>
#include <ctime>
#include <future>
#include <iostream>
#include <thread>
#include <optional>
#include <stdexcept>

namespace fusionrt {
namespace {

using namespace http;
using namespace std::chrono_literals;

class StopHandler {
public:
    StopHandler() {
        stopping = 0;
        old_int_ = std::signal(SIGINT, stop);
        old_term_ = std::signal(SIGTERM, stop);
#ifdef _WIN32
        old_break_ = std::signal(SIGBREAK, stop);
#else
        old_pipe_ = std::signal(SIGPIPE, SIG_IGN);
#endif
    }
    ~StopHandler() {
        std::signal(SIGINT, old_int_);
        std::signal(SIGTERM, old_term_);
#ifdef _WIN32
        std::signal(SIGBREAK, old_break_);
#else
        std::signal(SIGPIPE, old_pipe_);
#endif
    }
    static_assert(std::atomic_bool::is_always_lock_free);
    static std::atomic_bool stopping;
private:
    static void stop(int) { stopping = 1; }
    using Handler = void (*)(int);
    Handler old_int_, old_term_;
#ifdef _WIN32
    Handler old_break_;
#else
    Handler old_pipe_;
#endif
};

std::atomic_bool StopHandler::stopping{false};

constexpr const char* cors =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta\r\n";

void reply(const Write& write, const char* status, const JsonValue& body) {
    std::string text = body.dump(0);
    write(std::string("HTTP/1.1 ") + status + "\r\nConnection: close\r\n" + cors +
                 "Content-Type: application/json\r\nContent-Length: " + std::to_string(text.size()) +
                 "\r\n\r\n" + text);
}

void error(const Write& write, const char* status, const std::string& message,
           std::optional<agent::Protocol> protocol = {}, std::string_view code = {}) {
    reply(write, status, http::error_body(status, message, protocol, code));
}

struct Request {
    std::string method;
    std::string path;
    std::string body;
};

std::optional<Request> read_request(SocketHandle fd, std::optional<agent::Protocol>& protocol,
                                    std::stop_token stop) {
    const auto deadline = Clock::now() + 5s;
    constexpr size_t max_headers = 16 * 1024;
    std::string data;
    auto receive = [&] {
        char buf[4096];
        const int n = http::receive(fd, buf, sizeof(buf), deadline, stop);
        if (n <= 0) return false;
        data.append(buf, n);
        return true;
    };
    size_t end;
    while ((end = data.find("\r\n\r\n")) == std::string::npos) {
        if (data.size() >= max_headers) throw std::invalid_argument("HTTP headers are too large");
        if (!receive()) return std::nullopt;
    }
    if (end > max_headers) throw std::invalid_argument("HTTP headers are too large");
    auto head = http::parse_request_head(std::string_view(data).substr(0, end), protocol);
    while (data.size() - end - 4 < head.content_length) {
        if (!receive()) throw std::invalid_argument("Incomplete HTTP request body");
    }
    return Request{std::move(head.method), std::move(head.path), data.substr(end + 4, head.content_length)};
}

bool start_stream(const Write& write) {
    return write(std::string("HTTP/1.1 200 OK\r\nConnection: close\r\n") + cors +
                    "Cache-Control: no-cache\r\nContent-Type: text/event-stream\r\n\r\n");
}

bool send_event(const Write& write, std::string_view event, std::string_view data) {
    std::string frame;
    if (!event.empty()) frame = "event: " + std::string(event) + "\n";
    frame += "data: ";
    frame += data;
    frame += "\n\n";
    return write(frame);
}

// Hold only the suffix that could become a stop sequence on a later token.
struct StopBuffer {
    const std::vector<std::string>& stops;
    std::string pending;
    const std::string* matched = nullptr;

    std::string push(std::string text) {
        pending += text;
        size_t match = std::string::npos;
        for (const auto& stop : stops) {
            const auto position = pending.find(stop);
            if (position < match) {
                match = position;
                matched = &stop;
            }
        }
        if (matched) {
            std::string out = pending.substr(0, match);
            pending.clear();
            return out;
        }
        size_t hold = 0;
        for (const auto& stop : stops) {
            for (size_t n = std::min(stop.size() - 1, pending.size()); n > hold; --n) {
                if (std::string_view(pending).ends_with(std::string_view(stop).substr(0, n))) {
                    hold = n;
                    break;
                }
            }
        }
        std::string out = pending.substr(0, pending.size() - hold);
        pending.erase(0, out.size());
        return out;
    }
};

struct Completion {
    GenerationStats stats;
    agent::Finish finish = agent::Finish::length;
    std::string stop_sequence;
};

JsonObject diagnostics(const Completion& result, const ServeSpeculation& speculation, bool image) {
    const bool active = result.stats.speculative_active;
    return {{"speculative_mode", speculation.mode()}, {"speculative_active", active},
            {"spec_proposed", result.stats.spec_proposed}, {"spec_accepted", result.stats.spec_accepted},
            {"speculative_disabled_reason", active ? "" : image && speculation.enabled() ? "vision" : "disabled"}};
}

template<class Emit>
Completion generate(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                    const std::vector<int32_t>& prompt, const agent::GenerationParams& params,
                    int limit, std::optional<bool> graph, bool reuse_prefix, Emit emit,
                    const JsonConstraint* constraint, std::span<const InputEmbedding> embeds,
                    const ServeSpeculation& speculation, const std::atomic_bool& cancelled) {
    StopBuffer stop{params.stops, {}, nullptr};
    Utf8StreamDecoder utf8;
    Completion completion;
    bool connected = true;
    auto opts = speculation.for_inputs(embeds);
    opts.sampling = params.sampling;
    opts.reuse_prefix = reuse_prefix;
    opts.json_constraint = constraint;
    opts.cancel = &cancelled;
    opts.warn_on_graph_fallback = graph.value_or(false);
    opts.on_token = [&](int32_t id) {
        if (tokenizer.is_end_of_generation(id)) {
            completion.finish = agent::Finish::stop;
            return false;
        }
        connected = emit(stop.push(utf8.push(tokenizer.decode_token(id))));
        return connected && !stop.matched && !cancelled.load(std::memory_order_relaxed);
    };
    model.generate_cuda(ctx, prompt, limit, true, graph.value_or(model.recommended_cuda_graph()),
                        &completion.stats, false, 0, -1, opts);
    const auto& stats = completion.stats;
    std::clog << "prompt_tokens=" << stats.prompt_tokens << " prefilled_tokens=" << stats.prefilled_tokens
              << " cached_tokens=" << stats.prompt_tokens - stats.prefilled_tokens
              << " prefill_ms=" << stats.prefill_ms << " decode_ms=" << stats.decode_ms
              << " decode_tok/s=" << stats.decode_tok_per_s << " spec_proposed=" << stats.spec_proposed
              << " spec_accepted=" << stats.spec_accepted << " generated_tokens=" << stats.generated_tokens
              << " cancelled=" << (stats.cancelled ? "true" : "false") << std::endl;
    if (!connected || stats.cancelled || cancelled.load(std::memory_order_relaxed)) {
        completion.finish = agent::Finish::cancelled;
        return completion;
    }
    if (!stop.matched && emit(stop.push(utf8.finish())) && !stop.matched) emit(stop.pending);
    if (stop.matched) {
        completion.finish = agent::Finish::stop_sequence;
        completion.stop_sequence = *stop.matched;
    }
    if (constraint) completion.finish = stats.json_complete ? agent::Finish::stop : agent::Finish::length;
    return completion;
}

void complete(const Write& write, CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
              const std::string& name, const std::vector<int32_t>& prompt, const agent::GenerationParams& params,
              int context, std::optional<bool> graph, uint64_t request_id, std::span<const InputEmbedding> embeds,
              const ServeSpeculation& speculation, const std::atomic_bool& cancelled) {
    const int64_t created = std::time(nullptr);
    const std::string id = "cmpl-" + std::to_string(created) + "-" + std::to_string(request_id);
    auto response = [&](JsonObject choice) {
        choice["index"] = 0;
        return JsonObject{{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", name},
                          {"choices", JsonArray{std::move(choice)}}};
    };
    auto event = [&](const JsonValue& v) { return send_event(write, "", v.dump(0)); };
    const int limit = agent::generation_limit(params, context, prompt.size());
    model.validate_cuda_kv_capacity(static_cast<int>(prompt.size()) + limit);
    if (params.stream && !start_stream(write)) return;
    std::string result;
    bool connected = true;
    auto emit = [&](const std::string& text) {
        if (text.empty()) return connected;
        if (params.stream) connected = event(response({{"finish_reason", nullptr}, {"text", text}}));
        else result += text;
        return connected;
    };
    Completion completion;
    try {
        completion = generate(ctx, model, tokenizer, prompt, params, limit, graph, true, emit, nullptr, embeds, speculation, cancelled);
    } catch (const std::exception& e) {
        if (!params.stream) throw;
        if (connected && event(JsonObject{{"error", JsonObject{{"message", e.what()}, {"type", "server_error"}}}})) {
            send_event(write, "", "[DONE]");
        }
        return;
    }
    if (!connected || completion.finish == agent::Finish::cancelled) return;
    const bool stopped = completion.finish == agent::Finish::stop || completion.finish == agent::Finish::stop_sequence;
    auto payload = response({{"finish_reason", stopped ? "stop" : "length"},
                             {"text", std::move(result)}});
    payload["fusionrt_diagnostics"] = diagnostics(completion, speculation, !embeds.empty());
    if (params.stream) {
        if (event(payload)) send_event(write, "", "[DONE]");
    } else {
        const auto& stats = completion.stats;
        payload["usage"] = JsonObject{{"prompt_tokens", stats.prompt_tokens}, {"completion_tokens", stats.generated_tokens},
                                      {"total_tokens", stats.prompt_tokens + stats.generated_tokens}};
        reply(write, "200 OK", payload);
    }
}

void complete_agent(const Write& write, CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                    const std::string& name, agent::Protocol protocol, const agent::AgentRequest& request,
                    const agent::AgentPrompt& rendered, const std::vector<int32_t>& prompt,
                    int context, std::optional<bool> graph, uint64_t request_id, const JsonConstraint* constraint,
                    std::span<const InputEmbedding> embeds, const ServeSpeculation& speculation, const std::atomic_bool& cancelled) {
    const int64_t created = std::time(nullptr);
    const char* prefix = protocol == agent::Protocol::chat ? "chatcmpl-" :
                         protocol == agent::Protocol::responses ? "resp_" : "msg_";
    const auto id = prefix + std::to_string(created) + "-" + std::to_string(request_id);
    const int limit = agent::generation_limit(request.generation, context, prompt.size());
    model.validate_cuda_kv_capacity(static_cast<int>(prompt.size()) + limit);
    agent::EventSink event;
    if (request.generation.stream) {
        if (!start_stream(write)) return;
        event = [&write](std::string_view type, std::string_view data) { return send_event(write, type, data); };
    }
    agent::AgentResponse response(protocol, id, name, created, request.stream_usage, std::move(event));
    response.start(static_cast<int>(prompt.size()));
    if (!response.connected()) return;
    agent::ToolParser parser(request, rendered, [&](std::string_view text) { response.text(text); },
                             [&](agent::ToolCall call) { response.call(std::move(call)); },
                             [&](std::string_view thought) { response.thought(thought); });
    try {
        const auto completion = generate(ctx, model, tokenizer, prompt, request.generation, limit, graph, true,
            [&](const std::string& text) {
                if (constraint) response.text(text);
                else parser.push(text);
                return response.connected();
            }, constraint, embeds, speculation, cancelled);
        if (!response.connected() || completion.finish == agent::Finish::cancelled) return;
        parser.finish();
        if ((request.choice == agent::Choice::required || request.choice == agent::Choice::named) &&
            !response.result().has_calls() && completion.finish == agent::Finish::stop) {
            throw std::runtime_error("The model ended its turn without producing the requested tool call");
        }
        const auto reason = completion.finish == agent::Finish::length && parser.truncated_call()
            ? agent::Finish::truncated_tool_call : completion.finish;
        response.finish(reason, completion.stats.generated_tokens, completion.stop_sequence,
                        diagnostics(completion, speculation, !embeds.empty()));
        if (!request.generation.stream) reply(write, "200 OK", response.body());
    } catch (const std::exception& e) {
        if (!request.generation.stream) throw;
        response.error(e.what());
    }
}

SocketHandle listen_on(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found) != 0) {
        throw std::runtime_error("Cannot resolve bind address: " + host);
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(found, freeaddrinfo);
    for (auto* a = found; a; a = a->ai_next) {
        Socket sock(socket(a->ai_family, a->ai_socktype, a->ai_protocol));
        if (sock.get() == invalid_socket) continue;
#ifdef _WIN32
        const int enabled = 1;
        setsockopt(sock.get(), SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&enabled), sizeof(enabled));
#else
        const int enabled = 1;
        setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
#endif
        if (bind(sock.get(), a->ai_addr, static_cast<SocketLength>(a->ai_addrlen)) == 0 &&
            listen(sock.get(), SOMAXCONN) == 0) return sock.release();
    }
    throw std::runtime_error("Cannot listen on " + host + ":" + std::to_string(port));
}

}  // namespace

void serve_http(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                const std::string& model_name, const std::string& host, int port, std::optional<bool> graph,
                const VisionEncoder* vision, std::span<const Image> preset_images, ServeSpeculation speculation,
                std::stop_token shutdown) {
    if (!vision && !preset_images.empty()) throw std::invalid_argument("Preset images require --mmproj");
    [[maybe_unused]] SocketRuntime sockets;
    Socket listener(listen_on(host, port));
    StopHandler stop_handler;
    uint64_t request_id = 0;
    const auto& cfg = model.config();
    std::map<JsonRoot, JsonConstraint> json_constraints;
    std::mutex health_mutex;
    JsonObject snapshot;
    auto refresh_health = [&] {
        auto health = http::health_body(model_name, cfg, model.num_slots(), model.cuda_max_active_tokens(),
                                        model.cuda_cache_memory());
        health["vision"] = vision != nullptr;
        health["speculative"] = speculation.enabled();
        health["speculative_mode"] = speculation.mode();
        health["kv_dtype_requested"] = ctx.requested_kv_dtype() ? dtype_name(*ctx.requested_kv_dtype()) : "auto";
        health["kv_dtype"] = dtype_name(model.cuda_kv_dtype());
        health["borrowed_expert_bytes"] = static_cast<int64_t>(model.cuda_borrowed_expert_bytes());
        int full_gpu = 0, resident = 0, slabs = 0;
        for (const auto& layer : model.moe_layers()) {
            full_gpu += layer.cfg.gpu_resident_experts == layer.cfg.num_experts;
            resident += layer.cfg.gpu_resident_experts;
            slabs += layer.borrowed_gate_up_w.data != nullptr;
        }
        health["full_gpu_expert_layers"] = full_gpu;
        health["gpu_resident_experts"] = resident;
        health["borrowed_expert_slabs"] = slabs;
        std::lock_guard lock(health_mutex);
        snapshot = std::move(health);
    };
    refresh_health();

    auto post = [&](const Write& write, const Request& req, std::optional<agent::Protocol> protocol,
                    const std::atomic_bool& cancelled) {
    const JsonValue parsed = [&] {
        try {
            return JsonValue::parse(req.body);
        } catch (const std::runtime_error& e) {
            throw std::invalid_argument(e.what());
        }
    }();
    if (!parsed.is_object()) throw std::invalid_argument("Expected a JSON object");
    const auto& obj = parsed.as_object();
    const int context = std::min(cfg.dense.max_seq_len, model.cuda_max_active_tokens());
    if (protocol) {
        const auto input = agent::parse_request(obj, *protocol, vision != nullptr);
        const auto rendered = agent::render_prompt(input, tokenizer, preset_images);
        VisionPrompt prompt;
        if (rendered.images.empty()) prompt.tokens = http::encode_prompt(tokenizer, rendered.text, context);
        else prompt = encode_vision_prompt(ctx, *vision, tokenizer, rendered.text, rendered.images,
                                           rendered.image_offsets, context);
        const JsonConstraint* constraint = nullptr;
        if (input.generation.json) {
            const auto root = *input.generation.json;
            constraint = &json_constraints.try_emplace(root, tokenizer, cfg.vocab_size, root).first->second;
        }
        complete_agent(write, ctx, model, tokenizer, model_name, *protocol, input, rendered, prompt.tokens,
                       context, graph, request_id, constraint, prompt.inputs_embeds, speculation, cancelled);
    } else {
        const auto& text = agent::field(obj, "prompt");
        if (!text.is_string()) throw std::invalid_argument("prompt must be a string");
        const auto params = agent::parse_generation(obj, "max_tokens");
        VisionPrompt prompt;
        if (preset_images.empty()) prompt.tokens = http::encode_prompt(tokenizer, text.as_string(), context);
        else {
            std::vector<const Image*> images;
            std::vector<size_t> offsets;
            std::string marked;
            for (const auto& image : preset_images) {
                images.push_back(&image);
                offsets.push_back(marked.size());
                marked += image_placeholder;
            }
            prompt = encode_vision_prompt(ctx, *vision, tokenizer, marked + text.as_string(), images, offsets, context);
        }
        complete(write, ctx, model, tokenizer, model_name, prompt.tokens, params, context, graph,
                 request_id, prompt.inputs_embeds, speculation, cancelled);
    }
    };
    InferenceQueue queue;
    std::promise<void> ready;
    auto startup = ready.get_future();
    std::jthread worker([&](std::stop_token stop) {
        try {
            FUSIONRT_CUDA_CHECK(cudaSetDevice(ctx.info().device_id));
            ready.set_value();
        } catch (...) {
            ready.set_exception(std::current_exception());
            return;
        }
        while (auto job = queue.take(stop)) job();
    });
    startup.get();

    auto handle = [&](SocketHandle fd, std::stop_token stop) {
        if (!set_nonblocking(fd)) return;
        const auto deadline = Clock::now() + 6s;
        const Write write = [&](std::string_view bytes) { return send_all(fd, bytes, deadline, stop); };
        std::optional<agent::Protocol> protocol;
        try {
            auto req = read_request(fd, protocol, stop);
            if (!req) return;
            if (req->method == "OPTIONS") {
                write(std::string("HTTP/1.1 204 No Content\r\nConnection: close\r\n") + cors + "\r\n");
            } else if (req->method == "GET" && req->path == "/health") {
                JsonObject health;
                {
                    std::lock_guard lock(health_mutex);
                    health = snapshot;
                }
                reply(write, "200 OK", health);
            } else if (req->method == "GET" && req->path == "/v1/models") {
                reply(write, "200 OK", JsonObject{{"object", "list"}, {"data", JsonArray{JsonObject{
                      {"id", model_name}, {"object", "model"}, {"created", int64_t{0}}, {"owned_by", "fusionrt"}}}}});
            } else if (req->method == "POST" && (protocol || req->path == "/v1/completions")) {
                auto pipe = std::make_shared<ResponsePipe>();
                const bool queued = queue.try_push([&, pipe, request = std::move(*req), protocol] {
                    if (!pipe->cancelled.load(std::memory_order_relaxed)) {
                        const Write enqueue = [&](std::string_view bytes) { return pipe->write(bytes); };
                        try {
                            post(enqueue, request, protocol, pipe->cancelled);
                        } catch (const PromptContextError& e) {
                            error(enqueue, "400 Bad Request", e.what(), protocol, "context_length_exceeded");
                        } catch (const std::invalid_argument& e) {
                            error(enqueue, "400 Bad Request", e.what(), protocol);
                        } catch (const std::exception& e) {
                            error(enqueue, "500 Internal Server Error", e.what(), protocol);
                        }
                        refresh_health();
                    }
                    pipe->finish();
                });
                if (queued) pipe->flush(fd, stop);
                else error(write, "503 Service Unavailable", "Inference queue is full", protocol);
            } else error(write, "404 Not Found", "Unknown route", protocol);
        } catch (const http::PayloadTooLarge& e) {
            error(write, http::PayloadTooLarge::status, e.what(), protocol);
        } catch (const std::invalid_argument& e) {
            error(write, "400 Bad Request", e.what(), protocol);
        } catch (const std::exception& e) {
            error(write, "500 Internal Server Error", e.what(), protocol);
        }
    };

    struct Connection {
        std::shared_ptr<std::atomic_bool> done;
        std::jthread io;
    };
    std::vector<Connection> connections;
    std::cout << "fusionrt listening on " << host << ':' << port << " max_seq=" << cfg.dense.max_seq_len
              << " kv_type=" << dtype_name(model.cuda_kv_dtype()) << '\n' << std::flush;
    while (!StopHandler::stopping.load(std::memory_order_relaxed) && !shutdown.stop_requested()) {
        std::erase_if(connections, [](const auto& connection) { return connection.done->load(std::memory_order_acquire); });
        const int incoming = socket_ready(listener.get(), false, 100);
        if (incoming < 0 && !would_block()) throw std::runtime_error("HTTP listener wait failed");
        if (incoming <= 0) continue;
        Socket client(accept(listener.get(), nullptr, nullptr));
        if (client.get() == invalid_socket) continue;
        constexpr size_t max_connections = 32;
        if (connections.size() >= max_connections) {
            if (set_nonblocking(client.get())) {
                const Write write = [&](std::string_view bytes) {
                    return send_all(client.get(), bytes, Clock::now() + 100ms, shutdown);
                };
                error(write, "503 Service Unavailable", "Too many connections");
            }
            continue;
        }
        auto done = std::make_shared<std::atomic_bool>(false);
        connections.push_back({done, std::jthread([&, done, client = std::move(client)](std::stop_token stop) {
            handle(client.get(), stop);
            done->store(true, std::memory_order_release);
        })});
    }
    for (auto& connection : connections) connection.io.request_stop();
    connections.clear();
    worker.request_stop();
}

}  // namespace fusionrt
