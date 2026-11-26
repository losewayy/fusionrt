#include "fusionrt/json.h"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace fusionrt {

namespace {

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    JsonValue parse_value(int depth = 0) {
        skip_ws();
        if (pos_ >= s_.size()) throw std::runtime_error("Unexpected end of JSON");
        char c = s_[pos_];
        if (c == '{' || c == '[') {
            if (depth >= kMaxJsonNesting) throw std::runtime_error("JSON nesting depth exceeded");
            return c == '{' ? parse_object(depth + 1) : parse_array(depth + 1);
        }
        if (c == '"') return JsonValue(parse_string());
        if (c == 't') { expect("true"); return JsonValue(true); }
        if (c == 'f') { expect("false"); return JsonValue(false); }
        if (c == 'n') { expect("null"); return JsonValue(nullptr); }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw std::runtime_error(std::string("Unexpected JSON char: ") + c);
    }

    JsonValue parse_document() {
        auto value = parse_value();
        skip_ws();
        if (pos_ != s_.size()) throw std::runtime_error("Trailing characters after JSON value");
        return value;
    }

private:
    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    const std::string& s_;
    size_t pos_ = 0;

    void expect(const char* kw) {
        for (const char* p = kw; *p; ++p) {
            if (pos_ >= s_.size() || s_[pos_] != *p) {
                throw std::runtime_error(std::string("Expected ") + kw);
            }
            ++pos_;
        }
    }

    unsigned int read_hex4() {
        if (pos_ + 4 > s_.size()) throw std::runtime_error("Invalid Unicode escape");
        unsigned int code = 0;
        for (int i = 0; i < 4; ++i) {
            char digit = s_[pos_++];
            code <<= 4;
            if (digit >= '0' && digit <= '9') code |= digit - '0';
            else if (digit >= 'a' && digit <= 'f') code |= digit - 'a' + 10;
            else if (digit >= 'A' && digit <= 'F') code |= digit - 'A' + 10;
            else throw std::runtime_error("Invalid Unicode escape");
        }
        return code;
    }

    std::string parse_string() {
        if (s_[pos_] != '"') throw std::runtime_error("Expected '\"'");
        ++pos_;
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) throw std::runtime_error("Invalid escape");
                char e = s_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        unsigned int cp = read_hex4();
                        if (cp >= 0xd800 && cp <= 0xdbff) {
                            expect("\\u");
                            const unsigned int low = read_hex4();
                            if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("Invalid Unicode surrogate");
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                        } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                            throw std::runtime_error("Invalid Unicode surrogate");
                        }
                        if (cp < 0x80) {
                            out.push_back(static_cast<char>(cp));
                        } else if (cp < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else if (cp < 0x10000) {
                            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default: throw std::runtime_error("Invalid JSON escape");
                }
            } else {
                if (static_cast<unsigned char>(c) < 0x20) throw std::runtime_error("Unescaped JSON control character");
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated string");
    }

    JsonValue parse_number() {
        size_t start = pos_;
        bool is_float = false;
        if (s_[pos_] == '-') ++pos_;
        auto digits = [&] {
            const size_t first = pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
            if (pos_ == first) throw std::runtime_error("Invalid JSON number");
        };
        if (pos_ < s_.size() && s_[pos_] == '0') ++pos_;
        else digits();
        if (pos_ < s_.size() && s_[pos_] == '.') {
            is_float = true;
            ++pos_;
            digits();
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            digits();
        }
        std::string sub = s_.substr(start, pos_ - start);
        if (is_float) return JsonValue(std::stod(sub));
        return JsonValue(static_cast<int64_t>(std::stoll(sub)));
    }

    JsonValue parse_array(int depth) {
        ++pos_;
        JsonArray arr;
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return JsonValue(std::move(arr)); }
        while (true) {
            arr.push_back(parse_value(depth));
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
            if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; break; }
            throw std::runtime_error("Expected ',' or ']' in array");
        }
        return JsonValue(std::move(arr));
    }

    JsonValue parse_object(int depth) {
        ++pos_;
        JsonObject obj;
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return JsonValue(std::move(obj)); }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') throw std::runtime_error("Expected ':'");
            ++pos_;
            obj[std::move(key)] = parse_value(depth);
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
            if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; break; }
            throw std::runtime_error("Expected ',' or '}' in object");
        }
        return JsonValue(std::move(obj));
    }
};

void escape_json_str(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else {
                    os << static_cast<char>(c);
                }
        }
    }
    os << '"';
}

void dump_impl(std::ostringstream& os, const JsonValue& v, int indent, int depth) {
    if (v.is_null()) { os << "null"; return; }
    if (v.is_bool()) { os << (std::get<bool>(v.val) ? "true" : "false"); return; }
    if (v.is_int()) { os << std::get<int64_t>(v.val); return; }
    if (v.is_double()) {
        double d = std::get<double>(v.val);
        if (!std::isfinite(d)) { os << "null"; return; }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", d);
        os << buf;
        return;
    }
    if (v.is_string()) { escape_json_str(os, v.as_string()); return; }
    if (v.is_array()) {
        const auto& arr = v.as_array();
        if (arr.empty()) { os << "[]"; return; }
        os << '[';
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) os << ',';
            if (indent > 0) os << '\n' << std::string((depth + 1) * indent, ' ');
            dump_impl(os, arr[i], indent, depth + 1);
        }
        if (indent > 0) os << '\n' << std::string(depth * indent, ' ');
        os << ']';
        return;
    }
    if (v.is_object()) {
        const auto& obj = v.as_object();
        if (obj.empty()) { os << "{}"; return; }
        os << '{';
        size_t i = 0;
        for (const auto& [k, val] : obj) {
            if (i++ > 0) os << ',';
            if (indent > 0) os << '\n' << std::string((depth + 1) * indent, ' ');
            escape_json_str(os, k);
            os << (indent > 0 ? ": " : ":");
            dump_impl(os, val, indent, depth + 1);
        }
        if (indent > 0) os << '\n' << std::string(depth * indent, ' ');
        os << '}';
    }
}

}  // namespace

JsonValue JsonValue::parse(const std::string& text) {
    JsonParser p(text);
    return p.parse_document();
}

std::string JsonValue::dump(int indent) const {
    std::ostringstream os;
    dump_impl(os, *this, indent, 0);
    return os.str();
}

}  // namespace fusionrt
