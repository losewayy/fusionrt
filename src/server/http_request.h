#pragma once

#include "agent.h"
#include <stdexcept>

namespace fusionrt::http {

struct RequestHead {
    std::string method, path;
    size_t content_length = 0;
};

struct PayloadTooLarge : std::invalid_argument {
    static constexpr const char* status = "413 Payload Too Large";
    PayloadTooLarge() : std::invalid_argument("HTTP body exceeds the 16 MiB limit") {}
};

RequestHead parse_request_head(std::string_view text, std::optional<agent::Protocol>& protocol);
JsonObject error_body(std::string_view status, const std::string& message, std::optional<agent::Protocol> protocol);

}  // namespace fusionrt::http
