#pragma once

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fusionrt::cli {

template <typename T = int>
T parse_integer(std::string_view option, std::string_view value) {
    T number{};
    const auto* end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, number);
    if (result.ec != std::errc{} || result.ptr != end)
        throw std::invalid_argument("Invalid integer for " + std::string(option));
    return number;
}

}  // namespace fusionrt::cli
