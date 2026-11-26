#include "fusionrt/sampler.h"
#include "fusionrt/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace fusionrt {

void SamplingParams::validate() const {
    if (!std::isfinite(temperature) || top_k < 0 ||
        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 ||
        !std::isfinite(min_p) || min_p < 0 || min_p > 1 ||
        !std::isfinite(repeat_penalty) || repeat_penalty <= 0 || repeat_last_n < 0) {
        throw std::invalid_argument("Invalid sampling parameters");
    }
}

static Sampler::Distribution sampling_weights(const float* logits, int vocab_size,
                                               const std::vector<int32_t>& recent_tokens,
                                               const SamplingParams& params, bool& draw_required) {
    if (!logits || vocab_size <= 0) throw std::invalid_argument("Sampling requires logits");
    std::unordered_set<int32_t> repeats;
    if (params.repeat_penalty > 1.0f && params.repeat_last_n > 0) {
        size_t count = std::min(recent_tokens.size(), static_cast<size_t>(params.repeat_last_n));
        repeats.insert(recent_tokens.end() - count, recent_tokens.end());
    }

    using Candidate = std::pair<double, int32_t>;
    std::vector<Candidate> candidates;
    candidates.reserve(vocab_size);
    for (int32_t id = 0; id < vocab_size; ++id) {
        double score = logits[id];
        if (std::isnan(score) || score == -std::numeric_limits<double>::infinity()) continue;
        if (repeats.contains(id)) {
            score = score < 0 ? score * params.repeat_penalty : score / params.repeat_penalty;
        }
        candidates.emplace_back(score, id);
    }
    if (candidates.empty()) return {{1.0, 0}};
    auto better = [](const Candidate& a, const Candidate& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    size_t k = params.temperature <= 0 ? 1 : candidates.size();
    if (params.top_k > 0) k = std::min(k, static_cast<size_t>(params.top_k));
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), better);
    if (k == 1) return {{1.0, candidates.front().second}};
    draw_required = true;
    const double best = candidates.front().first;
    double mass = 0.0;
    for (size_t i = 0; i < k; ++i) {
        auto& score = candidates[i].first;
        score = score == best ? 1.0 : std::exp((score - best) / params.temperature);
        mass += score;
    }

    // Keep the shortest nucleus reaching top_p, then discard weights below min_p * max_weight.
    const double cutoff = mass * params.top_p;
    size_t kept = 0;
    double retained = 0.0;
    do {
        retained += candidates[kept++].first;
    } while (kept < k && retained < cutoff && candidates[kept].first >= params.min_p);

    candidates.resize(kept);
    return candidates;
}

Sampler::Distribution Sampler::distribution(const float* logits, int vocab_size,
                                            const std::vector<int32_t>& recent_tokens,
                                            const SamplingParams& params) {
    bool draw_required = false;
    auto weights = sampling_weights(logits, vocab_size, recent_tokens, params, draw_required);
    const double mass = std::accumulate(weights.begin(), weights.end(), 0.0,
        [](double sum, const auto& candidate) { return sum + candidate.first; });
    for (auto& candidate : weights) candidate.first /= mass;
    return weights;
}

int32_t Sampler::draw(const Distribution& distribution, std::mt19937_64& rng) {
    // Rejection sampling can pass unnormalized residual weights.
    double mass = 0.0;
    for (const auto& [weight, id] : distribution) mass += weight;
    double draw = std::generate_canonical<double, 53>(rng) * mass;
    for (const auto& [weight, id] : distribution) {
        draw -= weight;
        if (draw < 0) return id;
    }
    return distribution.back().second;
}

int32_t Sampler::sample(const float* logits, int vocab_size, const std::vector<int32_t>& recent_tokens,
                        const SamplingParams& params, std::mt19937_64& rng) {
    bool draw_required = false;
    const auto weights = sampling_weights(logits, vocab_size, recent_tokens, params, draw_required);
    return draw_required ? draw(weights, rng) : weights.front().second;
}

JsonGrammarState::JsonGrammarState(bool object_only)
    : stack_{object_only ? Expect::object_root : Expect::value} {}

bool JsonGrammarState::number_complete() const {
    return lex_ == Lex::zero || lex_ == Lex::integer || lex_ == Lex::fraction || lex_ == Lex::exponent_digits;
}

bool JsonGrammarState::sealed() const {
    return stack_.back() == Expect::done && lex_ == Lex::syntax;
}

bool JsonGrammarState::complete() const {
    return stack_.back() == Expect::done && (lex_ == Lex::syntax || number_complete());
}

void JsonGrammarState::value_done() {
    switch (stack_.back()) {
        case Expect::value:
        case Expect::object_root: stack_.back() = Expect::done; break;
        case Expect::item_or_end: stack_.back() = Expect::array_end; break;
        default: break;
    }
}

bool JsonGrammarState::advance(unsigned char c) {
    const bool digit = c >= '0' && c <= '9';
    switch (lex_) {
        case Lex::string:
            if (utf8_left_) {
                if (c < utf8_min_ || c > utf8_max_) return false;
                --utf8_left_;
                utf8_min_ = 0x80;
                utf8_max_ = 0xbf;
            } else if (c == '"') {
                lex_ = Lex::syntax;
                if (key_) stack_.back() = Expect::colon;
                key_ = false;
            } else if (c == '\\') {
                lex_ = Lex::escape;
            } else if (c >= 0xc2 && c <= 0xf4) {
                utf8_left_ = c < 0xe0 ? 1 : c < 0xf0 ? 2 : 3;
                utf8_min_ = c == 0xe0 ? 0xa0 : c == 0xf0 ? 0x90 : 0x80;
                utf8_max_ = c == 0xed ? 0x9f : c == 0xf4 ? 0x8f : 0xbf;
            } else if (c < 0x20 || c >= 0x80) {
                return false;
            }
            return true;
        case Lex::escape:
            if (c == 'u') lex_ = Lex::hex;
            else if (std::string_view("\"\\/bfnrt").find(static_cast<char>(c)) != std::string_view::npos)
                lex_ = Lex::string;
            else return false;
            return true;
        case Lex::low_slash:
            if (c != '\\') return false;
            lex_ = Lex::low_u;
            return true;
        case Lex::low_u:
            if (c != 'u') return false;
            lex_ = Lex::low_hex;
            return true;
        case Lex::hex:
        case Lex::low_hex: {
            const int nibble = digit ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                               c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (nibble < 0) return false;
            const bool low = lex_ == Lex::low_hex;
            hex_ = static_cast<uint16_t>((hex_ << 4) | nibble);
            ++hex_digits_;
            const int shift = 4 * (4 - hex_digits_);
            const int first = hex_ << shift;
            const int last = first + (1 << shift) - 1;
            if (low ? first > 0xdfff || last < 0xdc00 : first >= 0xdc00 && last <= 0xdfff) return false;
            if (hex_digits_ == 4) {
                lex_ = !low && hex_ >= 0xd800 && hex_ <= 0xdbff ? Lex::low_slash : Lex::string;
                hex_digits_ = 0;
                hex_ = 0;
            }
            return true;
        }
        case Lex::literal:
            if (c != static_cast<unsigned char>(literal_.front())) return false;
            literal_.remove_prefix(1);
            if (literal_.empty()) lex_ = Lex::syntax;
            return true;
        case Lex::minus:
            if (!digit) return false;
            lex_ = c == '0' ? Lex::zero : Lex::integer;
            return true;
        case Lex::zero:
        case Lex::integer:
            if (digit) return lex_ == Lex::integer;
            if (c == '.') { lex_ = Lex::dot; return true; }
            if (c == 'e' || c == 'E') { lex_ = Lex::exponent; return true; }
            break;
        case Lex::dot:
            if (!digit) return false;
            lex_ = Lex::fraction;
            return true;
        case Lex::fraction:
            if (digit) return true;
            if (c == 'e' || c == 'E') { lex_ = Lex::exponent; return true; }
            break;
        case Lex::exponent:
            if (c == '-' || c == '+') { lex_ = Lex::exponent_sign; return true; }
            [[fallthrough]];
        case Lex::exponent_sign:
            if (!digit) return false;
            lex_ = Lex::exponent_digits;
            return true;
        case Lex::exponent_digits:
            if (digit) return true;
            break;
        case Lex::syntax: break;
    }

    lex_ = Lex::syntax;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return true;
    switch (stack_.back()) {
        case Expect::done: return false;
        case Expect::key_or_end:
            if (c == '}') { stack_.pop_back(); return true; }
            [[fallthrough]];
        case Expect::key:
            if (c != '"') return false;
            key_ = true;
            lex_ = Lex::string;
            return true;
        case Expect::colon:
            if (c != ':') return false;
            stack_.back() = Expect::object_end;
            stack_.push_back(Expect::value);
            return true;
        case Expect::object_end:
            if (c == '}') { stack_.pop_back(); return true; }
            if (c != ',') return false;
            stack_.back() = Expect::key;
            return true;
        case Expect::array_end:
            if (c == ']') { stack_.pop_back(); return true; }
            if (c != ',') return false;
            stack_.push_back(Expect::value);
            return true;
        case Expect::item_or_end:
            if (c == ']') { stack_.pop_back(); return true; }
            break;
        case Expect::object_root:
            if (c != '{') return false;
            break;
        case Expect::value: break;
    }
    if (stack_.size() > 1 && stack_.back() == Expect::value) stack_.pop_back();
    else value_done();
    if (c == '{') stack_.push_back(Expect::key_or_end);
    else if (c == '[') stack_.push_back(Expect::item_or_end);
    else if (c == '"') lex_ = Lex::string;
    else if (c == '-') lex_ = Lex::minus;
    else if (digit) lex_ = c == '0' ? Lex::zero : Lex::integer;
    else {
        if (c == 't') literal_ = "rue";
        else if (c == 'f') literal_ = "alse";
        else if (c == 'n') literal_ = "ull";
        else return false;
        lex_ = Lex::literal;
    }
    return true;
}

bool JsonGrammarState::advance(std::string_view bytes) {
    for (unsigned char c : bytes) {
        if (!advance(c)) return false;
    }
    return true;
}

bool JsonGrammarState::consume(std::string_view bytes) {
    auto next = *this;
    if (!next.advance(bytes)) return false;
    *this = std::move(next);
    return true;
}

std::string JsonGrammarState::closing_suffix() const {
    auto state = *this;
    std::string suffix;
    while (!state.complete()) {
        unsigned char c;
        switch (state.lex_) {
            case Lex::string: c = state.utf8_left_ ? state.utf8_min_ : '"'; break;
            case Lex::escape: c = '"'; break;
            case Lex::hex: c = '0'; break;
            case Lex::low_slash: c = '\\'; break;
            case Lex::low_u: c = 'u'; break;
            case Lex::low_hex: c = state.hex_digits_ == 0 ? 'd' : state.hex_digits_ == 1 ? 'c' : '0'; break;
            case Lex::literal: c = static_cast<unsigned char>(state.literal_.front()); break;
            case Lex::minus:
            case Lex::dot:
            case Lex::exponent:
            case Lex::exponent_sign: c = '0'; break;
            default:
                switch (state.stack_.back()) {
                    case Expect::object_root: c = '{'; break;
                    case Expect::key_or_end:
                    case Expect::object_end: c = '}'; break;
                    case Expect::item_or_end:
                    case Expect::array_end: c = ']'; break;
                    case Expect::key: c = '"'; break;
                    case Expect::colon: c = ':'; break;
                    default: c = '0'; break;
                }
                break;
        }
        if (!state.advance(c)) throw std::logic_error("Invalid JSON completion state");
        suffix.push_back(static_cast<char>(c));
    }
    return suffix;
}

JsonConstraint::JsonConstraint(const BpeTokenizer& tokenizer, int vocab_size, bool object_only)
    : pieces_(vocab_size), end_(vocab_size), object_only_(object_only) {
    for (int32_t id = 0; id < vocab_size; ++id) {
        end_[id] = tokenizer.is_end_of_generation(id);
        if (end_[id]) continue;
        auto& piece = pieces_[id] = tokenizer.decode_token(id);
        if (piece.empty()) continue;
        const auto first = static_cast<unsigned char>(piece.front());
        by_first_byte_[first].push_back(id);
        if (piece.size() == 1) single_byte_[first] = true;
    }
}

bool JsonConstraint::can_finish(std::string_view suffix, int budget) const {
    if (suffix.empty()) return true;
    if (budget == 0) return false;
    if (suffix.size() <= static_cast<size_t>(budget) &&
        std::all_of(suffix.begin(), suffix.end(), [&](unsigned char c) { return single_byte_[c]; })) return true;
    std::vector<int> steps(suffix.size() + 1, budget);
    steps[0] = 0;
    for (size_t i = 0; i < suffix.size(); ++i) {
        if (steps[i] >= budget) continue;
        const auto tail = suffix.substr(i);
        for (int32_t id : by_first_byte_[static_cast<unsigned char>(tail.front())]) {
            const auto& piece = pieces_[id];
            if (!tail.starts_with(piece)) continue;
            if (piece.size() == tail.size()) return true;
            steps[i + piece.size()] = std::min(steps[i + piece.size()], steps[i] + 1);
        }
    }
    return false;
}

void JsonConstraint::mask(float* logits, const JsonGrammarState& state, int remaining_tokens) const {
    constexpr float masked = -std::numeric_limits<float>::infinity();
    if (remaining_tokens <= 0) throw std::invalid_argument("JSON generation needs a positive token budget");
    auto next = state;
    const bool same_state_fits = can_finish(state.closing_suffix(), remaining_tokens - 1);
    bool any = false;
    int fallback = -1;
    for (size_t id = 0; id < pieces_.size(); ++id) {
        next = state;
        const bool valid = end_[id] ? state.complete() :
            !pieces_[id].empty() && next.advance(pieces_[id]) &&
            (next == state ? same_state_fits : can_finish(next.closing_suffix(), remaining_tokens - 1));
        if (!valid) logits[id] = masked;
        else {
            fallback = static_cast<int>(id);
            any |= logits[id] > masked;
        }
    }
    if (!any) {
        if (fallback < 0) throw std::runtime_error("The vocabulary cannot finish JSON within the token budget");
        logits[fallback] = 0;
    }
}

bool JsonConstraint::accept(JsonGrammarState& state, int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) return false;
    if (end_[id]) return state.complete();
    return !pieces_[id].empty() && state.consume(pieces_[id]);
}

std::string Utf8StreamDecoder::push(std::string_view bytes) {
    pending_.append(bytes);
    std::string out;
    size_t pos = 0;
    while (pos < pending_.size()) {
        const auto lead = static_cast<unsigned char>(pending_[pos]);
        int n = lead < 0x80 ? 1 : lead >= 0xc2 && lead <= 0xdf ? 2 :
                lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
        int valid = 1;
        for (; valid < n && pos + valid < pending_.size(); ++valid) {
            const auto next = static_cast<unsigned char>(pending_[pos + valid]);
            if (next < 0x80 || next > 0xbf) break;
            if (valid == 1 && ((lead == 0xe0 && next < 0xa0) || (lead == 0xed && next > 0x9f) ||
                               (lead == 0xf0 && next < 0x90) || (lead == 0xf4 && next > 0x8f))) break;
        }
        if (valid == n) {
            out.append(pending_, pos, n);
        } else {
            if (n > 0 && pos + valid == pending_.size()) break;
            out += "\xef\xbf\xbd";
        }
        pos += valid;
    }
    pending_.erase(0, pos);
    return out;
}

std::string Utf8StreamDecoder::finish() {
    std::string out = pending_.empty() ? "" : "\xef\xbf\xbd";
    pending_.clear();
    return out;
}

}  // namespace fusionrt
