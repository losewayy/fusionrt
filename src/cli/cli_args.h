#pragma once

#include <charconv>
#include <filesystem>
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

struct SpeculativeArgs {
    int ngram = 0;
    std::string draft_path;
    int draft_tokens = 4;
    bool draft_tokens_set = false;
    bool mtp = false;

    bool parse(std::string_view option, std::string_view value) {
        if (option == "--spec-ngram") ngram = parse_integer(option, value);
        else if (option == "--draft-model") draft_path = value;
        else if (option == "--draft-tokens") {
            draft_tokens = parse_integer(option, value);
            draft_tokens_set = true;
        } else return false;
        return true;
    }

    bool enabled() const { return ngram > 0 || !draft_path.empty() || mtp; }

    void validate(bool has_target) const {
        if (ngram < 0 || draft_tokens <= 0 || (draft_tokens_set && draft_path.empty() && !mtp))
            throw std::invalid_argument("Expected --spec-ngram >= 0 and --draft-tokens > 0 with --draft-model or --spec-mtp");
        if (mtp && (!has_target || ngram > 0 || !draft_path.empty()))
            throw std::invalid_argument("--spec-mtp requires a target GGUF and excludes --spec-ngram and --draft-model");
        if (!draft_path.empty() && (!has_target || ngram > 0 ||
            std::filesystem::path(std::u8string(draft_path.begin(), draft_path.end())).extension() != ".gguf"))
            throw std::invalid_argument("--draft-model requires a GGUF draft, a target model, and no --spec-ngram");
    }
};

}  // namespace fusionrt::cli
