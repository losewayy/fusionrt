#include "fusionrt/tokenizer.h"
#include "fusionrt/json.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <string_view>
#include <sstream>

namespace fusionrt {

namespace {

std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp <= 0x7Fu) {
        s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        s.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        s.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        s.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
    return s;
}

struct Gpt2ByteMaps {
    std::array<std::string, 256> b2u;
    std::unordered_map<std::string, uint8_t> u2b;
    Gpt2ByteMaps() {
        std::array<bool, 256> direct{};
        for (int b = 33; b <= 126; ++b) direct[b] = true;
        for (int b = 161; b <= 172; ++b) direct[b] = true;
        for (int b = 174; b <= 255; ++b) direct[b] = true;
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
            std::string u = codepoint_to_utf8(cp);
            b2u[b] = u;
            u2b[u] = static_cast<uint8_t>(b);
        }
    }
};

const Gpt2ByteMaps& gpt2_maps() {
    static const Gpt2ByteMaps maps;
    return maps;
}

std::vector<std::string> split_utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0xE0u) == 0xC0u) len = 2;
        else if ((c & 0xF0u) == 0xE0u) len = 3;
        else if ((c & 0xF8u) == 0xF0u) len = 4;
        len = std::min(len, s.size() - i);
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::vector<std::string> pretokenize_gpt2(const std::string& text) {
    std::vector<std::string> chunks;
    size_t i = 0;
    const size_t n = text.size();
    auto is_ascii_alpha = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    auto is_ascii_digit = [](unsigned char c) { return (c >= '0' && c <= '9'); };
    auto is_space_no_nl = [](unsigned char c) { return c == ' ' || c == '\t'; };

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '\r' || c == '\n') {
            size_t j = i;
            while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
            chunks.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        size_t start = i;
        if (c == ' ' && i + 1 < n && text[i + 1] != '\r' && text[i + 1] != '\n' && !is_space_no_nl(static_cast<unsigned char>(text[i + 1]))) {
            ++i;
            c = static_cast<unsigned char>(text[i]);
        } else if (is_space_no_nl(c)) {
            while (i + 1 < n && is_space_no_nl(static_cast<unsigned char>(text[i + 1])) &&
                   (i + 2 < n && is_space_no_nl(static_cast<unsigned char>(text[i + 2])))) {
                ++i;
            }
            if (i == start) {
                while (i < n && is_space_no_nl(static_cast<unsigned char>(text[i]))) ++i;
            } else {
                ++i;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        if (is_ascii_digit(c)) {
            ++i;
            chunks.push_back(text.substr(start, i - start));
        } else if (is_ascii_alpha(c) || c >= 0x80u) {
            while (i < n) {
                unsigned char nc = static_cast<unsigned char>(text[i]);
                if (is_ascii_alpha(nc) || nc >= 0x80u) ++i;
                else break;
            }
            chunks.push_back(text.substr(start, i - start));
        } else {
            while (i < n) {
                unsigned char nc = static_cast<unsigned char>(text[i]);
                if (nc == ' ' || nc == '\t' || nc == '\r' || nc == '\n' || is_ascii_alpha(nc) || is_ascii_digit(nc) || nc >= 0x80u) break;
                ++i;
            }
            chunks.push_back(text.substr(start, i - start));
        }
    }
    return chunks;
}

// Qwen and Llama use the same pre-tokenization rules with different digit group widths.
std::vector<std::string> pretokenize_modern_bpe(const std::string& text, int digit_width, bool combine_marks) {
    enum class Kind { letter, digit, space, newline, symbol };
    const auto chars = split_utf8_chars(text);
    static const std::locale locale("");
    const auto& unicode = std::use_facet<std::ctype<wchar_t>>(locale);
    std::vector<Kind> kinds;
    kinds.reserve(chars.size());
    for (const auto& c : chars) {
        constexpr std::array<unsigned, 5> masks{0, 127, 31, 15, 7};
        uint32_t cp = static_cast<unsigned char>(c[0]) & masks[c.size()];
        for (size_t j = 1; j < c.size(); ++j) cp = cp * 64 + (static_cast<unsigned char>(c[j]) & 63);
        const bool fits = cp <= static_cast<uint32_t>(std::numeric_limits<wchar_t>::max());
        const auto wc = static_cast<wchar_t>(cp);
        if (cp == '\n' || cp == '\r') kinds.push_back(Kind::newline);
        else if (fits && unicode.is(std::ctype_base::space, wc)) kinds.push_back(Kind::space);
        else if ((cp >= '0' && cp <= '9') || (cp >= 0xff10 && cp <= 0xff19) ||
                 (cp >= 0x660 && cp <= 0x669) || (cp >= 0x6f0 && cp <= 0x6f9)) kinds.push_back(Kind::digit);
        else if ((fits && unicode.is(std::ctype_base::alpha, wc)) ||
                 (combine_marks && cp >= 0x300 && cp <= 0x36f)) kinds.push_back(Kind::letter);
        else kinds.push_back(Kind::symbol);
    }
    auto is_space = [](Kind kind) { return kind == Kind::space || kind == Kind::newline; };
    std::vector<std::string> chunks;
    for (size_t i = 0; i < chars.size();) {
        const size_t start = i;
        if (chars[i] == "'") {
            for (std::string_view suffix : {"s", "t", "re", "ve", "m", "ll", "d"}) {
                if (i + suffix.size() >= chars.size()) continue;
                size_t j = 0;
                while (j < suffix.size() && chars[i + j + 1].size() == 1 &&
                       std::tolower(static_cast<unsigned char>(chars[i + j + 1][0])) == suffix[j]) ++j;
                if (j == suffix.size()) {
                    i += j + 1;
                    break;
                }
            }
        }
        if (i == start) {
            const auto kind = kinds[i];
            if (kind == Kind::letter || (kind != Kind::newline && kind != Kind::digit &&
                                         i + 1 < chars.size() && kinds[i + 1] == Kind::letter)) {
                do { ++i; } while (i < chars.size() && kinds[i] == Kind::letter);
            } else if (kind == Kind::digit) {
                do { ++i; } while (i < chars.size() && kinds[i] == Kind::digit &&
                                    i - start < static_cast<size_t>(digit_width));
            } else if (kind == Kind::symbol || (chars[i] == " " && i + 1 < chars.size() &&
                                                kinds[i + 1] == Kind::symbol)) {
                do { ++i; } while (i < chars.size() && kinds[i] == Kind::symbol);
                while (i < chars.size() && kinds[i] == Kind::newline) ++i;
            } else {
                size_t last_newline = start;
                while (i < chars.size() && is_space(kinds[i])) {
                    if (kinds[i] == Kind::newline) last_newline = i + 1;
                    ++i;
                }
                if (last_newline > start) i = last_newline;
                else if (i < chars.size() && i - start > 1) --i;
            }
        }
        std::string chunk;
        for (size_t j = start; j < i; ++j) chunk += chars[j];
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

std::string trim_chat(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    return std::string(text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1));
}

std::string chat_date() {
    const auto now = std::time(nullptr);
    std::tm time{};
#ifdef _WIN32
    localtime_s(&time, &now);
#else
    localtime_r(&now, &time);
#endif
    constexpr const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    char date[24];
    std::snprintf(date, sizeof(date), "%02d %s %d", time.tm_mday, months[time.tm_mon], 1900 + time.tm_year);
    return date;
}

}  // namespace

void BpeTokenizer::init_byte_fallback(int vocab_size) {
    id_to_token_.clear();
    token_to_id_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    chat_template_.clear();
    pretokenizer_.clear();
    end_ids_.clear();
    token_types_.clear();
    token_scores_.clear();
    sentencepiece_ = false;
    bos_id_ = 1;
    eos_id_ = 2;
    add_bos_default_ = true;
    byte_fallback_only_ = true;
    gpt2_byte_bpe_ = false;

    id_to_token_.reserve(static_cast<size_t>(std::max(260, vocab_size)));
    id_to_token_.push_back("<pad>");
    id_to_token_.push_back("<bos>");
    id_to_token_.push_back("<eos>");
    id_to_token_.push_back("<unk>");
    for (int b = 0; b < 256; ++b) {
        id_to_token_.push_back(std::string(1, static_cast<char>(b)));
    }
    while (static_cast<int>(id_to_token_.size()) < vocab_size) {
        id_to_token_.push_back("<tok_" + std::to_string(id_to_token_.size()) + ">");
    }
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<int32_t>(i);
    }
}

bool BpeTokenizer::load_from_gguf(const GgufFile& gguf) {
    auto tokens = gguf.get_string_array("tokenizer.ggml.tokens");
    if (tokens.empty()) return false;

    id_to_token_ = std::move(tokens);
    token_to_id_.clear();
    special_tokens_.clear();
    chat_template_ = gguf.get_string("tokenizer.chat_template", gguf.get_string("tokenizer.chat_template.default"));
    pretokenizer_ = gguf.get_string("tokenizer.ggml.pre");
    token_types_ = gguf.get_int_array("tokenizer.ggml.token_type");
    token_scores_ = gguf.get_float_array("tokenizer.ggml.scores");
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        const std::string& t = id_to_token_[i];
        token_to_id_[t] = static_cast<int32_t>(i);
        if ((t.size() >= 5 && t.rfind("<|", 0) == 0 && t.substr(t.size() - 2) == "|>") ||
            t == "<think>" || t == "</think>" || t == "<bos>" || t == "<eos>" || t == "<s>" || t == "</s>" ||
            t == "<start_of_turn>" || t == "<end_of_turn>" ||
            (i < token_types_.size() && (token_types_[i] == 3 || token_types_[i] == 4) && !t.empty())) {
            special_tokens_.push_back(t);
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

    std::string tok_model = gguf.get_string("tokenizer.ggml.model", "");
    gpt2_byte_bpe_ = (tok_model == "gpt2");
    sentencepiece_ = tok_model == "llama" || tok_model == "spm";
    add_space_prefix_ = gguf.get_bool("tokenizer.ggml.add_space_prefix", sentencepiece_);
    bos_id_ = static_cast<int32_t>(gguf.get_int("tokenizer.ggml.bos_token_id", gpt2_byte_bpe_ ? -1 : 1));
    eos_id_ = static_cast<int32_t>(gguf.get_int("tokenizer.ggml.eos_token_id", 2));
    unk_id_ = static_cast<int32_t>(gguf.get_int("tokenizer.ggml.unknown_token_id", 0));
    end_ids_.clear();
    for (const char* key : {"eot_token_id", "eom_token_id"}) {
        const auto id = gguf.get_int(std::string("tokenizer.ggml.") + key, -1);
        if (id >= 0 && id < vocab_size()) end_ids_.push_back(static_cast<int32_t>(id));
    }
    for (const char* marker : {"<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|end|>"}) {
        if (auto it = token_to_id_.find(marker); it != token_to_id_.end()) end_ids_.push_back(it->second);
    }
    add_bos_default_ = gguf.get_bool("tokenizer.ggml.add_bos_token", !gpt2_byte_bpe_ || pretokenizer_ == "llama-bpe");

    merge_ranks_.clear();
    auto merges = gguf.get_string_array("tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); ++i) {
        merge_ranks_[merges[i]] = static_cast<int>(i);
    }
    byte_fallback_only_ = !sentencepiece_ && merge_ranks_.empty() && special_tokens_.empty();
    return true;
}

bool BpeTokenizer::load_from_json(const std::string& path, std::string* err) {
    try {
        std::ifstream ifs(std::filesystem::path(std::u8string(path.begin(), path.end())));
        if (!ifs) {
            if (err) *err = "Cannot open tokenizer.json: " + path;
            return false;
        }
        std::ostringstream ss;
        ss << ifs.rdbuf();
        JsonValue root = JsonValue::parse(ss.str());
        if (!root.is_object()) return false;
        const auto& robj = root.as_object();
        if (!robj.count("model") || !robj.at("model").is_object()) return false;
        const auto& model = robj.at("model").as_object();
        if (!model.count("vocab") || !model.at("vocab").is_object()) return false;

        const auto& vocab = model.at("vocab").as_object();
        int64_t max_id = 0;
        for (const auto& [tok, id_v] : vocab) {
            max_id = std::max(max_id, id_v.as_int());
        }
        id_to_token_.assign(static_cast<size_t>(max_id + 1), "<unk>");
        token_to_id_.clear();
        special_tokens_.clear();
        chat_template_.clear();
    pretokenizer_.clear();
        end_ids_.clear();
        token_types_.clear();
        token_scores_.clear();
        sentencepiece_ = false;
        for (const auto& [tok, id_v] : vocab) {
            int32_t id = static_cast<int32_t>(id_v.as_int());
            id_to_token_[static_cast<size_t>(id)] = tok;
            token_to_id_[tok] = id;
        }
        if (robj.count("added_tokens") && robj.at("added_tokens").is_array()) {
            for (const auto& at : robj.at("added_tokens").as_array()) {
                if (!at.is_object()) continue;
                const auto& aobj = at.as_object();
                if (aobj.count("content") && aobj.count("id")) {
                    int32_t id = static_cast<int32_t>(aobj.at("id").as_int());
                    std::string content = aobj.at("content").as_string();
                    if (id >= static_cast<int32_t>(id_to_token_.size())) {
                        id_to_token_.resize(static_cast<size_t>(id + 1), "<unk>");
                    }
                    id_to_token_[static_cast<size_t>(id)] = content;
                    token_to_id_[content] = id;
                    if (aobj.count("special") && aobj.at("special").is_bool() && std::get<bool>(aobj.at("special").val)) {
                        special_tokens_.push_back(content);
                    }
                }
            }
        }
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

        gpt2_byte_bpe_ = token_to_id_.count("\xC4\xA0" "the") || token_to_id_.count("\xC4\xA0" "and") || token_to_id_.count("\xC4\x8A");
        add_bos_default_ = !gpt2_byte_bpe_;

        merge_ranks_.clear();
        if (model.count("merges") && model.at("merges").is_array()) {
            const auto& merges = model.at("merges").as_array();
            for (size_t i = 0; i < merges.size(); ++i) {
                if (merges[i].is_string()) {
                    merge_ranks_[merges[i].as_string()] = static_cast<int>(i);
                } else if (merges[i].is_array()) {
                    const auto& pair = merges[i].as_array();
                    if (pair.size() == 2 && pair[0].is_string() && pair[1].is_string()) {
                        merge_ranks_[pair[0].as_string() + " " + pair[1].as_string()] = static_cast<int>(i);
                    }
                }
            }
        }
        byte_fallback_only_ = merge_ranks_.empty();
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

std::vector<int32_t> BpeTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && add_bos_default_ && bos_id_ >= 0 && bos_id_ < vocab_size() &&
        (pretokenizer_ != "llama-bpe" || !text.starts_with(id_to_token_[bos_id_]))) {
        ids.push_back(bos_id_);
    }
    if (text.empty()) return ids;

    if (byte_fallback_only_) {
        for (unsigned char c : text) {
            std::string key(1, static_cast<char>(c));
            auto it = token_to_id_.find(key);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(4 + static_cast<int32_t>(c));
            }
        }
        return ids;
    }

    auto bpe_encode_symbols = [&](std::vector<std::string> syms) {
        while (syms.size() >= 2) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_idx = std::string::npos;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                std::string pair = syms[i] + " " + syms[i + 1];
                auto it = merge_ranks_.find(pair);
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_idx == std::string::npos) break;
            syms[best_idx] += syms[best_idx + 1];
            syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
        }
        for (const auto& s : syms) {
            auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                for (const auto& ch : split_utf8_chars(s)) {
                    auto bit = token_to_id_.find(ch);
                    ids.push_back(bit != token_to_id_.end() ? bit->second : 3);
                }
            }
        }
    };

    auto encode_plain = [&](const std::string& seg) {
        if (seg.empty()) return;
        if (gpt2_byte_bpe_) {
            const auto& b2u = gpt2_maps().b2u;
            const bool llama = pretokenizer_ == "llama-bpe";
            const bool qwen35 = pretokenizer_ == "qwen35";
            const bool modern = llama || qwen35 || pretokenizer_ == "qwen2";
            const auto chunks = modern ? pretokenize_modern_bpe(seg, llama ? 3 : 1, qwen35) : pretokenize_gpt2(seg);
            for (const auto& chunk : chunks) {
                std::vector<std::string> syms;
                syms.reserve(chunk.size());
                for (unsigned char c : chunk) {
                    syms.push_back(b2u[c]);
                }
                if (llama) {
                    std::string whole;
                    for (const auto& symbol : syms) whole += symbol;
                    if (auto it = token_to_id_.find(whole); it != token_to_id_.end()) {
                        ids.push_back(it->second);
                        continue;
                    }
                }
                bpe_encode_symbols(std::move(syms));
            }
        } else if (sentencepiece_) {
            std::string normalized;
            if (add_space_prefix_ && (ids.empty() || (ids.size() == 1 && ids[0] == bos_id_))) {
                normalized = "\xe2\x96\x81";
            }
            for (char ch : seg) {
                if (ch == ' ') normalized += "\xe2\x96\x81";
                else normalized += ch;
            }
            auto syms = split_utf8_chars(normalized);
            while (syms.size() > 1) {
                size_t best = syms.size();
                float score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i + 1 < syms.size(); ++i) {
                    const auto it = token_to_id_.find(syms[i] + syms[i + 1]);
                    if (it == token_to_id_.end()) continue;
                    const size_t id = static_cast<size_t>(it->second);
                    if (id < token_types_.size() && token_types_[id] != 1) continue;
                    float next = id < token_scores_.size() ? token_scores_[id] : 0.0f;
                    if (next > score) { best = i; score = next; }
                }
                if (best == syms.size()) break;
                syms[best] += syms[best + 1];
                syms.erase(syms.begin() + best + 1);
            }
            for (const auto& sym : syms) {
                if (auto it = token_to_id_.find(sym); it != token_to_id_.end()) {
                    ids.push_back(it->second);
                    continue;
                }
                for (unsigned char ch : sym) {
                    char hex[7];
                    std::snprintf(hex, sizeof(hex), "<0x%02X>", static_cast<unsigned int>(ch));
                    auto it = token_to_id_.find(hex);
                    ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
                }
            }
        } else {
            std::vector<std::string> syms;
            syms.reserve(seg.size());
            for (unsigned char c : seg) {
                if (c == ' ') syms.emplace_back(" ");
                else syms.emplace_back(1, static_cast<char>(c));
            }
            bpe_encode_symbols(std::move(syms));
        }
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        const std::string* best_sp = nullptr;
        for (const auto& sp : special_tokens_) {
            size_t f = text.find(sp, pos);
            if (f != std::string::npos && (f < best_pos || (f == best_pos && best_sp && sp.size() > best_sp->size()))) {
                best_pos = f;
                best_sp = &sp;
            }
        }
        if (!best_sp) {
            encode_plain(text.substr(pos));
            break;
        }
        if (best_pos > pos) {
            encode_plain(text.substr(pos, best_pos - pos));
        }
        ids.push_back(token_to_id_.at(*best_sp));
        pos = best_pos + best_sp->size();
    }
    return ids;
}

std::string BpeTokenizer::decode_token(int32_t id) const {
    if (id < 0 || id >= vocab_size()) return "";
    if (id == bos_id_ || is_end_of_generation(id)) return "";
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    if (sentencepiece_) {
        if (tok.size() == 6 && tok.starts_with("<0x") && tok.back() == '>') {
            const int hi = hex_digit(tok[3]), lo = hex_digit(tok[4]);
            if (hi >= 0 && lo >= 0) return std::string(1, static_cast<char>(hi * 16 + lo));
        }
        std::string text = tok;
        const std::string space = "\xe2\x96\x81";
        for (size_t pos = 0; (pos = text.find(space, pos)) != std::string::npos; ++pos) {
            text.replace(pos, space.size(), 1, ' ');
        }
        return text;
    }
    if (!gpt2_byte_bpe_) return tok;
    if ((tok.size() >= 5 && tok.rfind("<|", 0) == 0 && tok.substr(tok.size() - 2) == "|>") ||
        tok == "<think>" || tok == "</think>") {
        return tok;
    }
    const auto& u2b = gpt2_maps().u2b;
    std::string out;
    out.reserve(tok.size());
    for (const auto& ch : split_utf8_chars(tok)) {
        auto it = u2b.find(ch);
        if (it != u2b.end()) out.push_back(static_cast<char>(it->second));
        else out += ch;
    }
    return out;
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        out += decode_token(id);
    }
    return out;
}

bool BpeTokenizer::is_end_of_generation(int32_t id) const {
    return id == eos_id_ || std::find(end_ids_.begin(), end_ids_.end(), id) != end_ids_.end();
}

std::string BpeTokenizer::format_chat_messages(const std::vector<ChatMessage>& msgs,
                                               bool add_generation_prompt) const {
    const bool has_known_template = chat_template_.find("<|im_start|>") != std::string::npos ||
                                    chat_template_.find("<|start_header_id|>") != std::string::npos ||
                                    chat_template_.find("<start_of_turn>") != std::string::npos;
    auto uses = [&](const char* marker) {
        return token_to_id_.contains(marker) &&
               (!has_known_template || chat_template_.find(marker) != std::string::npos);
    };
    std::string out;
    if (uses("<|im_start|>") && has_chatml()) {
        constexpr std::string_view qwen_system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        if (chat_template_.find(qwen_system) != std::string::npos && (msgs.empty() || msgs[0].role != "system")) {
            out += "<|im_start|>system\n" + std::string(qwen_system) + "<|im_end|>\n";
        }
        for (const auto& msg : msgs) out += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        if (add_generation_prompt) {
            out += "<|im_start|>assistant\n";
            if (pretokenizer_ == "qwen35" && chat_template_.find("enable_thinking is true") != std::string::npos) {
                out += "<think>\n";
            }
        }
    } else if (uses("<|start_header_id|>") && token_to_id_.contains("<|end_header_id|>") &&
               token_to_id_.contains("<|eot_id|>")) {
        auto header = [](const std::string& role) {
            return "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
        };
        size_t begin = 0;
        if (chat_template_.find("Cutting Knowledge Date: December 2023") != std::string::npos) {
            if (!add_bos_default_ && bos_id_ >= 0 && bos_id_ < vocab_size()) out += id_to_token_[bos_id_];
            out += header("system") + "Cutting Knowledge Date: December 2023\nToday Date: " + chat_date() + "\n\n";
            if (!msgs.empty() && msgs[0].role == "system") out += trim_chat(msgs[begin++].content);
            out += "<|eot_id|>";
        }
        for (; begin < msgs.size(); ++begin) {
            out += header(msgs[begin].role) + trim_chat(msgs[begin].content) + "<|eot_id|>";
        }
        if (add_generation_prompt) out += header("assistant");
    } else if (uses("<start_of_turn>") && token_to_id_.contains("<end_of_turn>")) {
        std::string system;
        for (const auto& msg : msgs) {
            if (msg.role == "system") {
                system += msg.content + "\n\n";
                continue;
            }
            out += std::string("<start_of_turn>") + (msg.role == "assistant" ? "model" : "user") +
                   "\n" + system + msg.content + "<end_of_turn>\n";
            system.clear();
        }
        if (add_generation_prompt) out += "<start_of_turn>model\n";
    } else {
        for (const auto& msg : msgs) out += msg.role + ": " + msg.content + "\n";
        if (add_generation_prompt) out += "assistant: ";
    }
    return out;
}

}  // namespace fusionrt
