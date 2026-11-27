#include "http_request.h"
#include "fusionrt/models.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <sstream>

namespace fusionrt::http {

RequestHead parse_request_head(std::string_view text, std::optional<agent::Protocol>& protocol) {
    constexpr size_t max_body = 16 * 1024 * 1024;
    RequestHead request;
    std::istringstream headers{std::string(text)};
    std::string first, version;
    std::getline(headers, first);
    std::istringstream start(first);
    if (!(start >> request.method >> request.path >> version) ||
        (version != "HTTP/1.1" && version != "HTTP/1.0")) {
        throw std::invalid_argument("Expected an HTTP/1.0 or HTTP/1.1 request");
    }
    request.path.resize(std::min(request.path.find('?'), request.path.size()));
    protocol.reset();
    if (request.path == "/v1/chat/completions") protocol = agent::Protocol::chat;
    else if (request.path == "/v1/responses") protocol = agent::Protocol::responses;
    else if (request.path == "/v1/messages") protocol = agent::Protocol::messages;

    bool has_length = false, oversized = false;
    for (std::string line; std::getline(headers, line);) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("Invalid HTTP header");
        std::string name = line.substr(0, colon);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (name == "transfer-encoding") throw std::invalid_argument("Use Content-Length for the request body");
        if (name != "content-length") continue;
        const size_t begin = line.find_first_not_of(" \t", colon + 1);
        const size_t last = line.find_last_not_of(" \t\r");
        if (has_length || begin == std::string::npos || begin > last) throw std::invalid_argument("Invalid Content-Length");
        const auto* end = line.data() + last + 1;
        auto [ptr, ec] = std::from_chars(line.data() + begin, end, request.content_length);
        if (ptr != end || (ec != std::errc{} && ec != std::errc::result_out_of_range))
            throw std::invalid_argument("Invalid Content-Length");
        oversized = ec == std::errc::result_out_of_range || request.content_length > max_body;
        has_length = true;
    }
    if (oversized) throw PayloadTooLarge{};
    return request;
}

JsonObject error_body(std::string_view status, const std::string& message, std::optional<agent::Protocol> protocol) {
    const bool anthropic = protocol == agent::Protocol::messages;
    const char* type = status.starts_with('5') ? (anthropic ? "api_error" : "server_error") : "invalid_request_error";
    if (anthropic && status == PayloadTooLarge::status) type = "request_too_large";
    JsonObject body{{"error", JsonObject{{"message", message}, {"type", type}}}};
    if (anthropic) body["type"] = "error";
    return body;
}

JsonObject health_body(const std::string& model, const HybridModelConfig& cfg, int slots, const ModelMemory& cache) {
    const auto& arch = cfg.model_architecture.empty() ? cfg.arch : cfg.model_architecture;
    return {{"status", "ok"}, {"model", model}, {"arch", arch}, {"backend", "cuda"},
            {"max_seq", cfg.dense.max_seq_len}, {"slots", slots}, {"kv_pages", cache.kv_pages},
            {"kv_bytes", static_cast<int64_t>(cache.kv_bytes)},
            {"kv_reserved_bytes", static_cast<int64_t>(cache.kv_reserved_bytes)},
            {"host_kv_bytes", static_cast<int64_t>(cache.host_kv_bytes)},
            {"host_kv_pages", cache.host_kv_pages},
            {"ssm_bytes", static_cast<int64_t>(cache.ssm_bytes)}};
}

std::vector<int32_t> encode_prompt(const BpeTokenizer& tokenizer, const std::string& text, int context) {
    auto prompt = tokenizer.encode(text);
    if (prompt.empty()) prompt.push_back(tokenizer.bos_id() >= 0 ? tokenizer.bos_id() : 0);
    if (prompt.size() >= static_cast<size_t>(context)) throw prompt_context_error(prompt.size(), context);
    return prompt;
}

}  // namespace fusionrt::http
