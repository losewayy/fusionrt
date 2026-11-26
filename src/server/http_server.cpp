#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "fusionrt/server.h"
#include "fusionrt/json.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <csignal>
#include <ctime>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
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
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

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

void error(SocketHandle fd, const char* status, const std::string& message) {
    reply(fd, status, JsonObject{{"error", JsonObject{{"message", message}, {"type", "invalid_request_error"}}}});
}

struct Request {
    std::string method;
    std::string path;
    std::string body;
};

std::optional<Request> read_request(SocketHandle fd) {
    constexpr size_t max_headers = 16 * 1024;
    constexpr size_t max_body = 2 * 1024 * 1024;
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
    Request request;
    std::istringstream headers(data.substr(0, end));
    std::string first, version;
    std::getline(headers, first);
    std::istringstream start(first);
    if (!(start >> request.method >> request.path >> version) ||
        (version != "HTTP/1.1" && version != "HTTP/1.0")) {
        throw std::invalid_argument("Expected an HTTP/1.0 or HTTP/1.1 request");
    }
    request.path = request.path.substr(0, request.path.find('?'));
    size_t length = 0;
    bool has_length = false;
    for (std::string line; std::getline(headers, line);) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("Invalid HTTP header");
        std::string name = line.substr(0, colon);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        if (name == "transfer-encoding") throw std::invalid_argument("Use Content-Length for the request body");
        if (name != "content-length") continue;
        const size_t begin = line.find_first_not_of(" \t", colon + 1);
        const size_t last = line.find_last_not_of(" \t\r");
        if (has_length || begin == std::string::npos || begin > last) throw std::invalid_argument("Invalid Content-Length");
        auto [ptr, ec] = std::from_chars(line.data() + begin, line.data() + last + 1, length);
        if (ec != std::errc{} || ptr != line.data() + last + 1 || length > max_body) {
            throw std::invalid_argument("Invalid or oversized Content-Length");
        }
        has_length = true;
    }
    while (data.size() - end - 4 < length) {
        if (!receive()) throw std::invalid_argument("Incomplete HTTP request body");
    }
    request.body = data.substr(end + 4, length);
    return request;
}

const JsonValue& field(const JsonObject& obj, const char* key) {
    static const JsonValue null;
    auto it = obj.find(key);
    return it == obj.end() ? null : it->second;
}

std::string require_string(const JsonValue& v, const char* field_name) {
    if (!v.is_string()) throw std::invalid_argument(std::string(field_name) + " must be a string");
    return v.as_string();
}

struct CompletionRequest {
    std::vector<int32_t> prompt;
    std::vector<std::string> stops;
    SamplingParams sampling;
    int max_tokens = 512;
    bool stream = false;
};

CompletionRequest parse_completion(const std::string& body, const BpeTokenizer& tokenizer, bool chat, int context) {
    const JsonValue parsed = JsonValue::parse(body);
    if (!parsed.is_object()) throw std::invalid_argument("Expected a JSON object");
    const auto& obj = parsed.as_object();
    CompletionRequest req;
    std::string prompt;
    if (chat) {
        const auto& value = field(obj, "messages");
        if (!value.is_array() || value.as_array().empty()) throw std::invalid_argument("Expected messages");
        std::vector<ChatMessage> messages;
        for (const auto& item : value.as_array()) {
            if (!item.is_object()) throw std::invalid_argument("Expected a message object");
            const auto& msg = item.as_object();
            auto role = require_string(field(msg, "role"), "role");
            if (role != "system" && role != "user" && role != "assistant") {
                throw std::invalid_argument("Unsupported message role: " + role);
            }
            messages.push_back({std::move(role), require_string(field(msg, "content"), "content")});
        }
        prompt = tokenizer.format_chat_messages(messages);
    } else {
        prompt = require_string(field(obj, "prompt"), "prompt");
    }
    req.prompt = tokenizer.encode(prompt);
    if (req.prompt.empty()) req.prompt.push_back(tokenizer.bos_id() >= 0 ? tokenizer.bos_id() : 0);
    if (req.prompt.size() >= static_cast<size_t>(context)) throw std::invalid_argument("Prompt exceeds model context");

    auto number = [&](const char* key, float& out) {
        const auto& v = field(obj, key);
        if (v.is_null()) return;
        if (!v.is_number()) throw std::invalid_argument(std::string(key) + " must be a number");
        out = static_cast<float>(v.as_double());
    };
    auto integer = [&](const char* key, int64_t fallback, int64_t max) {
        const auto& v = field(obj, key);
        if (v.is_null()) return fallback;
        if (!v.is_int() || v.as_int() < 0 || v.as_int() > max) {
            throw std::invalid_argument(std::string(key) + " must be a nonnegative integer in range");
        }
        return v.as_int();
    };
    number("temperature", req.sampling.temperature);
    number("top_p", req.sampling.top_p);
    number("min_p", req.sampling.min_p);
    number("repeat_penalty", req.sampling.repeat_penalty);
    req.sampling.top_k = static_cast<int>(integer("top_k", req.sampling.top_k, std::numeric_limits<int>::max()));
    req.sampling.seed = static_cast<uint64_t>(integer("seed", req.sampling.seed, INT64_MAX));
    req.sampling.validate();
    req.max_tokens = static_cast<int>(integer("max_tokens", req.max_tokens, std::numeric_limits<int>::max()));
    req.max_tokens = std::min(req.max_tokens, context - static_cast<int>(req.prompt.size()));
    const auto& stream = field(obj, "stream");
    if (!stream.is_null() && !stream.is_bool()) throw std::invalid_argument("stream must be a boolean");
    req.stream = stream.is_bool() && std::get<bool>(stream.val);
    const auto& stop = field(obj, "stop");
    if (stop.is_string()) req.stops.push_back(stop.as_string());
    else if (stop.is_array()) {
        for (const auto& entry : stop.as_array()) req.stops.push_back(require_string(entry, "stop"));
    } else if (!stop.is_null()) throw std::invalid_argument("stop must be a string or string array");
    if (req.stops.size() > 4 || std::any_of(req.stops.begin(), req.stops.end(), [](const auto& s) { return s.empty(); })) {
        throw std::invalid_argument("Expected up to four nonempty stop strings");
    }
    return req;
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

void complete(SocketHandle fd, CudaContext& ctx, FusionModel& model, const BpeTokenizer& tokenizer,
              const std::string& name, const CompletionRequest& req, bool chat, bool graph, uint64_t request_id) {
    const int64_t created = std::time(nullptr);
    const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(created) +
                           "-" + std::to_string(request_id);
    const char* object = chat ? (req.stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";
    auto response = [&](JsonObject choice) {
        choice["index"] = 0;
        return JsonObject{{"id", id}, {"object", object}, {"created", created}, {"model", name},
                          {"choices", JsonArray{std::move(choice)}}};
    };
    auto event = [&](const JsonValue& v) { return send_all(fd, "data: " + v.dump(0) + "\n\n"); };
    if (req.stream) {
        if (!send_all(fd, std::string("HTTP/1.1 200 OK\r\nConnection: close\r\n") + cors +
                          "Cache-Control: no-cache\r\nContent-Type: text/event-stream\r\n\r\n")) return;
        if (chat && !event(response({{"delta", JsonObject{{"role", "assistant"}}}, {"finish_reason", nullptr}}))) return;
    }
    std::string result;
    bool connected = true;
    bool end_token = false;
    StopBuffer stop{req.stops, {}, false};
    Utf8StreamDecoder utf8;
    auto emit = [&](const std::string& text) {
        if (text.empty()) return;
        if (!req.stream) result += text;
        else {
            JsonObject choice{{"finish_reason", nullptr}};
            if (chat) choice["delta"] = JsonObject{{"content", text}};
            else choice["text"] = text;
            connected = event(response(std::move(choice)));
        }
    };
    GenerationOptions opts;
    opts.sampling = req.sampling;
    opts.reuse_prefix = true;
    opts.on_token = [&](int32_t id) {
        if (tokenizer.is_end_of_generation(id)) {
            end_token = true;
            return false;
        }
        emit(stop.push(utf8.push(tokenizer.decode_token(id))));
        return connected && !stop.matched && !StopHandler::stopping;
    };
    GenerationStats stats;
    try {
        model.generate_cuda(ctx, req.prompt, req.max_tokens, true, graph, &stats, false, 0, -1, opts);
    } catch (const std::exception& e) {
        if (!req.stream) throw;
        if (connected) {
            event(JsonObject{{"error", JsonObject{{"message", e.what()}, {"type", "server_error"}}}});
            send_all(fd, "data: [DONE]\n\n");
        }
        return;
    }
    if (!connected) return;
    if (!stop.matched) {
        emit(stop.push(utf8.finish()));
        if (connected) emit(stop.pending);
    }
    if (!connected) return;
    const char* reason = end_token || stop.matched ? "stop" : "length";
    JsonObject choice{{"finish_reason", reason}};
    if (chat) {
        if (req.stream) choice["delta"] = JsonObject{};
        else choice["message"] = JsonObject{{"role", "assistant"}, {"content", std::move(result)}};
    } else choice["text"] = std::move(result);
    auto payload = response(std::move(choice));
    if (req.stream) {
        if (event(payload)) send_all(fd, "data: [DONE]\n\n");
    } else {
        payload["usage"] = JsonObject{{"prompt_tokens", stats.prompt_tokens}, {"completion_tokens", stats.generated_tokens},
                                      {"total_tokens", stats.prompt_tokens + stats.generated_tokens}};
        reply(fd, "200 OK", payload);
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
                const std::string& model_name, const std::string& host, int port, bool use_graph) {
    [[maybe_unused]] SocketRuntime sockets;
    Socket listener(listen_on(host, port));
    StopHandler stop_handler;
    std::mutex engine;
    uint64_t request_id = 0;
    const auto& cfg = model.config();
    const std::string& arch = cfg.model_architecture.empty() ? cfg.arch : cfg.model_architecture;
    std::cout << "fusionrt listening on " << host << ':' << port << '\n' << std::flush;
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
        try {
            auto req = read_request(fd);
            if (!req) continue;
            const bool chat = req->path == "/v1/chat/completions";
            if (req->method == "OPTIONS") {
                send_all(fd, std::string("HTTP/1.1 204 No Content\r\nConnection: close\r\n") + cors + "\r\n");
            } else if (req->method == "GET" && req->path == "/health") {
                reply(fd, "200 OK", JsonObject{{"status", "ok"}, {"model", model_name}, {"arch", arch}, {"backend", "cuda"}});
            } else if (req->method == "GET" && req->path == "/v1/models") {
                reply(fd, "200 OK", JsonObject{{"object", "list"}, {"data", JsonArray{JsonObject{
                      {"id", model_name}, {"object", "model"}, {"created", int64_t{0}}, {"owned_by", "fusionrt"}}}}});
            } else if (req->method == "POST" && (chat || req->path == "/v1/completions")) {
                CompletionRequest input;
                try {
                    input = parse_completion(req->body, tokenizer, chat, cfg.dense.max_seq_len);
                } catch (const std::exception& e) {
                    error(fd, "400 Bad Request", e.what());
                    continue;
                }
                std::lock_guard lock(engine);
                complete(fd, ctx, model, tokenizer, model_name, input, chat, use_graph, ++request_id);
            } else error(fd, "404 Not Found", "Unknown route");
        } catch (const std::invalid_argument& e) {
            error(fd, "400 Bad Request", e.what());
        } catch (const std::exception& e) {
            error(fd, "500 Internal Server Error", e.what());
        }
    }
}

}  // namespace fusionrt
