#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace fusionrt {

struct JsonValue;
using JsonArray = std::vector<JsonValue>;
using JsonObject = std::map<std::string, JsonValue>;

struct JsonValue {
    using Variant = std::variant<std::nullptr_t, bool, int64_t, double, std::string, JsonArray, JsonObject>;
    Variant val;

    JsonValue() : val(nullptr) {}
    JsonValue(std::nullptr_t) : val(nullptr) {}
    JsonValue(bool b) : val(b) {}
    JsonValue(int i) : val(static_cast<int64_t>(i)) {}
    JsonValue(int64_t i) : val(i) {}
    JsonValue(double d) : val(d) {}
    JsonValue(const char* s) : val(std::string(s)) {}
    JsonValue(std::string s) : val(std::move(s)) {}
    JsonValue(JsonArray a) : val(std::move(a)) {}
    JsonValue(JsonObject o) : val(std::move(o)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(val); }
    bool is_bool() const { return std::holds_alternative<bool>(val); }
    bool is_int() const { return std::holds_alternative<int64_t>(val); }
    bool is_double() const { return std::holds_alternative<double>(val); }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return std::holds_alternative<std::string>(val); }
    bool is_array() const { return std::holds_alternative<JsonArray>(val); }
    bool is_object() const { return std::holds_alternative<JsonObject>(val); }

    double as_double(double def = 0.0) const {
        if (is_double()) return std::get<double>(val);
        if (is_int()) return static_cast<double>(std::get<int64_t>(val));
        return def;
    }
    int64_t as_int(int64_t def = 0) const {
        if (is_int()) return std::get<int64_t>(val);
        if (is_double()) return static_cast<int64_t>(std::get<double>(val));
        return def;
    }
    const std::string& as_string() const { return std::get<std::string>(val); }
    const JsonArray& as_array() const { return std::get<JsonArray>(val); }
    const JsonObject& as_object() const { return std::get<JsonObject>(val); }

    static JsonValue parse(const std::string& text);
    std::string dump(int indent = 2) const;
};

}  // namespace fusionrt
