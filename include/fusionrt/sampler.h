#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fusionrt {

class BpeTokenizer;

struct SamplingParams {
    float temperature = 0.0f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    uint64_t seed = 42;

    bool greedy() const { return temperature <= 0.0f && repeat_penalty <= 1.0f; }
    void validate() const;
};

struct Sampler {
    using Distribution = std::vector<std::pair<double, int32_t>>;

    static Distribution distribution(const float* logits, int vocab_size,
                                     const std::vector<int32_t>& recent_tokens, const SamplingParams& params);
    static int32_t draw(const Distribution& distribution, std::mt19937_64& rng);
    static int32_t argmax(const float* logits, int vocab_size);
    static int32_t sample(const float* logits, int vocab_size, const std::vector<int32_t>& recent_tokens,
                          const SamplingParams& params, std::mt19937_64& rng);
};

enum class JsonRoot { value, object, array, string, number, integer, boolean, null };

class JsonGrammarState {
public:
    explicit JsonGrammarState(bool object_only = false);
    explicit JsonGrammarState(JsonRoot root);
    bool consume(std::string_view bytes);
    bool complete() const;
    bool sealed() const;

private:
    friend class JsonConstraint;
    enum class Expect : uint8_t { value, key_or_end, key, colon, item_or_end, object_end, array_end, done };
    enum class Lex : uint8_t { syntax, string, escape, hex, low_slash, low_u, low_hex, literal,
                               minus, zero, integer, dot, fraction, exponent, exponent_sign, exponent_digits };

    std::vector<Expect> stack_;
    JsonRoot root_;
    Lex lex_ = Lex::syntax;
    std::string_view literal_;
    uint16_t hex_ = 0;
    uint8_t hex_digits_ = 0;
    uint8_t utf8_left_ = 0, utf8_min_ = 0x80, utf8_max_ = 0xbf;
    bool key_ = false;

    bool advance(unsigned char c);
    bool advance(std::string_view bytes);
    void value_done();
    bool number_complete() const;
    std::string closing_suffix() const;
    bool operator==(const JsonGrammarState&) const = default;
};

class JsonConstraint {
public:
    explicit JsonConstraint(const BpeTokenizer& tokenizer, int vocab_size, bool object_only = false);
    JsonConstraint(const BpeTokenizer& tokenizer, int vocab_size, JsonRoot root);
    size_t vocab_size() const { return pieces_.size(); }
    JsonGrammarState initial_state() const { return JsonGrammarState(root_); }
    void mask(std::span<float> logits, const JsonGrammarState& state, int remaining_tokens) const;
    bool accept(JsonGrammarState& state, int32_t id) const;
    bool is_end(int32_t id) const { return end_[id]; }

private:
    bool can_finish(std::string_view suffix, int budget) const;

    std::vector<std::string> pieces_;
    std::vector<bool> end_;
    std::array<std::vector<int32_t>, 256> by_first_byte_;
    std::array<bool, 256> single_byte_{};
    JsonRoot root_;
};

class Utf8StreamDecoder {
public:
    std::string push(std::string_view bytes);
    std::string finish();

private:
    std::string pending_;
};

}  // namespace fusionrt
