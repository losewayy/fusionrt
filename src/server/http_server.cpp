#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "fusionrt/server.h"
#include "fusionrt/vision.h"
#include "agent.h"
#include "http_request.h"

#include <algorithm>
#include <csignal>
#include <ctime>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace fusionrt {
namespace {

#ifdef _WIN32
using SocketHandle = SOCKET;
using SocketLength = int;
constexpr SocketHandle invalid_socket = INVALID_SOCKET;
void close_socket(SocketHandle fd) { closesocket(fd); }

struct SocketRuntime {
    SocketRuntime() {
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) throw std::runtime_error("WSAStartup failed");
    }
    ~SocketRuntime() { WSACleanup(); }
};
#else
using SocketHandle = int;
using SocketLength = socklen_t;
constexpr SocketHandle invalid_socket = -1;
void close_socket(SocketHandle fd) { close(fd); }
struct SocketRuntime {};
#endif

class Socket {
public:
    explicit Socket(SocketHandle fd) : fd_(fd) {}
    ~Socket() { if (fd_ != invalid_socket) close_socket(fd_); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    SocketHandle get() const { return fd_; }
    SocketHandle release() {
        auto fd = fd_;
        fd_ = invalid_socket;
        return fd;
    }
private:
    SocketHandle fd_;
};

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
    static volatile std::sig_atomic_t stopping;
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

volatile std::sig_atomic_t StopHandler::stopping = 0;

constexpr const char* cors =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta\r\n";

bool send_all(SocketHandle fd, std::string_view text) {
    while (!text.empty()) {
        const int n = static_cast<int>(send(fd, text.data(), static_cast<int>(text.size()), 0));
        if (n <= 0) return false;
        text.remove_prefix(n);
    }
    return true;
}

void reply(SocketHandle fd, const char* status, const JsonValue& body) {
    std::string text = body.dump(0);
    send_all(fd, std::string("HTTP/1.1 ") + status + "\r\nConnection: close\r\n" + cors +
                 "Content-Type: application/json\r\nContent-Length: " + std::to_string(text.size()) +
                 "\r\n\r\n" + text);
}

void error(SocketHandle fd, const char* status, const std::string& message,
           std::optional<agent::Protocol> protocol = {}) {
    reply(fd, status, http::error_body(status, message, protocol));
}

struct Request {
    std::string method;
    std::string path;
    std::string body;
};

std::optional<Request> read_request(SocketHandle fd, std::optional<agent::Protocol>& protocol) {
    constexpr size_t max_headers = 16 * 1024;
    std::string data;
    auto receive = [&] {
        char buf[4096];
        const int n = static_cast<int>(recv(fd, buf, sizeof(buf), 0));
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

bool start_stream(SocketHandle fd) {
    return send_all(fd, std::string("HTTP/1.1 200 OK\r\nConnection: close\r\n") + cors +
                    "Cache-Control: no-cache\r\nContent-Type: text/event-stream\r\n\r\n");
}

bool send_event(SocketHandle fd, std::string_view event, std::string_view data) {
    std::string frame;
    if (!event.empty()) frame = "event: " + std::string(event) + "\n";
    frame += "data: ";
    frame += data;
    frame += "\n\n";
    return send_all(fd, frame);
}

// Hold only the suffix that could become a stop sequence on a later token.
struct StopBuffer {
    const std::vector<std::string>& stops;
    std::string pending;
    bool matched = false;

    std::string push(std::string text) {
        pending += text;
        size_t match = std::string::npos;
        for (const auto& stop : stops) match = std::min(match, pending.find(stop));
        if (match != std::string::npos) {
            matched = true;
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
};

template<class Emit>
Completion generate(CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                    const std::vector<int32_t>& prompt, const agent::GenerationParams& params,
                    int limit, std::optional<bool> graph, bool reuse_prefix, Emit emit,
                    const JsonConstraint* constraint, std::span<const InputEmbedding> embeds,
                    const ServeSpeculation& speculation) {
    StopBuffer stop{params.stops, {}, false};
    Utf8StreamDecoder utf8;
    Completion completion;
    bool connected = true;
    auto opts = speculation.for_inputs(embeds);
    opts.sampling = params.sampling;
    opts.reuse_prefix = reuse_prefix;
    opts.json_constraint = constraint;
    opts.warn_on_graph_fallback = graph.value_or(false);
    opts.on_token = [&](int32_t id) {
        if (tokenizer.is_end_of_generation(id)) {
            completion.finish = agent::Finish::stop;
            return false;
        }
        connected = emit(stop.push(utf8.push(tokenizer.decode_token(id))));
        return connected && !stop.matched && !StopHandler::stopping;
    };
    model.generate_cuda(ctx, prompt, limit, true, graph.value_or(model.recommended_cuda_graph()),
                        &completion.stats, false, 0, -1, opts);
    const auto& stats = completion.stats;
    std::clog << "prompt_tokens=" << stats.prompt_tokens << " prefilled_tokens=" << stats.prefilled_tokens
              << " cached_tokens=" << stats.prompt_tokens - stats.prefilled_tokens
              << " prefill_ms=" << stats.prefill_ms << std::endl;
    if (connected && !stop.matched && emit(stop.push(utf8.finish())) && !stop.matched) emit(stop.pending);
    if (stop.matched || constraint) completion.finish = agent::Finish::stop;
    return completion;
}

void complete(SocketHandle fd, CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
              const std::string& name, const std::vector<int32_t>& prompt, const agent::GenerationParams& params,
              int context, std::optional<bool> graph, uint64_t request_id, std::span<const InputEmbedding> embeds,
              const ServeSpeculation& speculation) {
    const int64_t created = std::time(nullptr);
    const std::string id = "cmpl-" + std::to_string(created) + "-" + std::to_string(request_id);
    auto response = [&](JsonObject choice) {
        choice["index"] = 0;
        return JsonObject{{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", name},
                          {"choices", JsonArray{std::move(choice)}}};
    };
    auto event = [&](const JsonValue& v) { return send_event(fd, "", v.dump(0)); };
    const int limit = agent::generation_limit(params, context, prompt.size());
    if (params.stream && !start_stream(fd)) return;
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
        completion = generate(ctx, model, tokenizer, prompt, params, limit, graph, true, emit, nullptr, embeds, speculation);
    } catch (const std::exception& e) {
        if (!params.stream) throw;
        if (connected && event(JsonObject{{"error", JsonObject{{"message", e.what()}, {"type", "server_error"}}}})) {
            send_event(fd, "", "[DONE]");
        }
        return;
    }
    if (!connected) return;
    auto payload = response({{"finish_reason", completion.finish == agent::Finish::stop ? "stop" : "length"},
                             {"text", std::move(result)}});
    if (params.stream) {
        if (event(payload)) send_event(fd, "", "[DONE]");
    } else {
        const auto& stats = completion.stats;
        payload["usage"] = JsonObject{{"prompt_tokens", stats.prompt_tokens}, {"completion_tokens", stats.generated_tokens},
                                      {"total_tokens", stats.prompt_tokens + stats.generated_tokens}};
        reply(fd, "200 OK", payload);
    }
}

void complete_agent(SocketHandle fd, CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
                    const std::string& name, agent::Protocol protocol, const agent::AgentRequest& request,
                    const agent::AgentPrompt& rendered, const std::vector<int32_t>& prompt,
                    int context, std::optional<bool> graph, uint64_t request_id, const JsonConstraint* constraint,
                    std::span<const InputEmbedding> embeds, const ServeSpeculation& speculation) {
    const int64_t created = std::time(nullptr);
    const char* prefix = protocol == agent::Protocol::chat ? "chatcmpl-" :
                         protocol == agent::Protocol::responses ? "resp_" : "msg_";
    const auto id = prefix + std::to_string(created) + "-" + std::to_string(request_id);
    const int limit = agent::generation_limit(request.generation, context, prompt.size());
    agent::EventSink event;
    if (request.generation.stream) {
        if (!start_stream(fd)) return;
        event = [fd](std::string_view type, std::string_view data) { return send_event(fd, type, data); };
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
            }, constraint, embeds, speculation);
        if (!response.connected()) return;
        parser.finish();
        if ((request.choice == agent::Choice::required || request.choice == agent::Choice::named) &&
            !response.result().has_calls() && completion.finish == agent::Finish::stop) {
            throw std::runtime_error("The model ended its turn without producing the requested tool call");
        }
        response.finish(completion.finish, completion.stats.generated_tokens);
        if (!request.generation.stream) reply(fd, "200 OK", response.body());
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
                const VisionEncoder* vision, std::span<const Image> preset_images, ServeSpeculation speculation) {
    if (!vision && !preset_images.empty()) throw std::invalid_argument("Preset images require --mmproj");
    [[maybe_unused]] SocketRuntime sockets;
    Socket listener(listen_on(host, port));
    StopHandler stop_handler;
    uint64_t request_id = 0;
    const auto& cfg = model.config();
    std::map<JsonRoot, JsonConstraint> json_constraints;
    std::cout << "fusionrt listening on " << host << ':' << port << " max_seq=" << cfg.dense.max_seq_len
              << " kv_type=" << dtype_name(model.cuda_kv_dtype()) << '\n' << std::flush;
    while (!StopHandler::stopping) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(listener.get(), &read_fds);
        timeval wait{0, 250000};
#ifdef _WIN32
        const int ready = select(0, &read_fds, nullptr, nullptr, &wait);
#else
        const int ready = select(listener.get() + 1, &read_fds, nullptr, nullptr, &wait);
#endif
        if (StopHandler::stopping) break;
        if (ready == 0) continue;
        if (ready < 0) throw std::runtime_error("HTTP listener select failed");
        Socket client(accept(listener.get(), nullptr, nullptr));
        auto fd = client.get();
        if (fd == invalid_socket) continue;
#ifdef _WIN32
        const DWORD timeout = 5000;
#else
        const timeval timeout{5, 0};
#endif
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        std::optional<agent::Protocol> protocol;
        try {
            auto req = read_request(fd, protocol);
            if (!req) continue;
            if (req->method == "OPTIONS") {
                send_all(fd, std::string("HTTP/1.1 204 No Content\r\nConnection: close\r\n") + cors + "\r\n");
            } else if (req->method == "GET" && req->path == "/health") {
                auto health = http::health_body(model_name, cfg, model.num_slots(), model.cuda_cache_memory());
                health["vision"] = vision != nullptr;
                health["speculative"] = speculation.ngram > 0 || speculation.draft != nullptr;
                reply(fd, "200 OK", health);
            } else if (req->method == "GET" && req->path == "/v1/models") {
                reply(fd, "200 OK", JsonObject{{"object", "list"}, {"data", JsonArray{JsonObject{
                      {"id", model_name}, {"object", "model"}, {"created", int64_t{0}}, {"owned_by", "fusionrt"}}}}});
            } else if (req->method == "POST" && (protocol || req->path == "/v1/completions")) {
                const JsonValue parsed = [&] {
                    try {
                        return JsonValue::parse(req->body);
                    } catch (const std::runtime_error& e) {
                        throw std::invalid_argument(e.what());
                    }
                }();
                if (!parsed.is_object()) throw std::invalid_argument("Expected a JSON object");
                const auto& obj = parsed.as_object();
                const int context = cfg.dense.max_seq_len;
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
                    complete_agent(fd, ctx, model, tokenizer, model_name, *protocol, input, rendered, prompt.tokens,
                                   context, graph, ++request_id, constraint, prompt.inputs_embeds, speculation);
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
                    complete(fd, ctx, model, tokenizer, model_name, prompt.tokens, params, context, graph,
                             ++request_id, prompt.inputs_embeds, speculation);
                }
            } else error(fd, "404 Not Found", "Unknown route", protocol);
        } catch (const http::PayloadTooLarge& e) {
            error(fd, http::PayloadTooLarge::status, e.what(), protocol);
        } catch (const std::invalid_argument& e) {
            error(fd, "400 Bad Request", e.what(), protocol);
        } catch (const std::exception& e) {
            error(fd, "500 Internal Server Error", e.what(), protocol);
        }
    }
}

}  // namespace fusionrt
