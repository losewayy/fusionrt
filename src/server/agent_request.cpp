#include "agent.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fusionrt::agent {

const JsonValue& field(const JsonObject& obj, const char* key) {
    static const JsonValue null;
    const auto it = obj.find(key);
    return it == obj.end() ? null : it->second;
}

namespace {

const JsonObject& object(const JsonValue& v, const char* label) {
    if (!v.is_object()) throw std::invalid_argument(std::string(label) + " must be an object");
    return v.as_object();
}

const JsonArray& array(const JsonValue& v, const char* label) {
    if (!v.is_array()) throw std::invalid_argument(std::string(label) + " must be an array");
    return v.as_array();
}

std::string string(const JsonValue& v, const char* label) {
    if (!v.is_string()) throw std::invalid_argument(std::string(label) + " must be a string");
    return v.as_string();
}

std::string optional_string(const JsonObject& v, const char* key) {
    return field(v, key).is_null() ? std::string{} : string(field(v, key), key);
}

std::string identifier(const JsonObject& v, const char* key) {
    std::string id = string(field(v, key), key);
    if (id.empty()) throw std::invalid_argument(std::string(key) + " must not be empty");
    return id;
}

bool boolean(const JsonObject& obj, const char* key) {
    const auto& v = field(obj, key);
    if (v.is_null()) return false;
    if (!v.is_bool()) throw std::invalid_argument(std::string(key) + " must be a boolean");
    return std::get<bool>(v.val);
}

int64_t integer(const JsonObject& obj, const char* key, int64_t fallback, int64_t max) {
    const auto& v = field(obj, key);
    if (v.is_null()) return fallback;
    if (!v.is_int() || v.as_int() < 0 || v.as_int() > max) {
        throw std::invalid_argument(std::string(key) + " must be a nonnegative integer in range");
    }
    return v.as_int();
}

Role role(std::string_view name) {
    if (name == "system" || name == "developer") return Role::system;
    if (name == "user") return Role::user;
    if (name == "assistant") return Role::assistant;
    if (name == "tool") return Role::tool;
    throw std::invalid_argument("Unsupported message role");
}

std::string text_content(const JsonValue& v) {
    if (v.is_string()) return v.as_string();
    std::string out;
    for (const auto& entry : array(v, "content")) {
        const auto& block = object(entry, "content block");
        const auto type = identifier(block, "type");
        if (type != "text" && type != "input_text" && type != "output_text") {
            throw std::invalid_argument("Only text content blocks are supported");
        }
        out += string(field(block, "text"), "text");
    }
    return out;
}

struct ImageBudget {
    bool allowed;
    int count = 0;
    size_t bytes = 0;

    void check(Role role) const {
        if (!allowed) throw std::invalid_argument("Image input requires a loaded --mmproj");
        if (role != Role::user) throw std::invalid_argument("Image input requires a user message");
        if (count >= 8) throw std::invalid_argument("Too many images in request");
    }
    void append(Message& msg, Image image) {
        if (image.rgb.size() > 64 * 1024 * 1024 - bytes)
            throw std::invalid_argument("Decoded images exceed request size limit");
        bytes += image.rgb.size();
        ++count;
        msg.content.emplace_back(std::move(image));
    }
};

void openai_content(Message& msg, const JsonValue& value, Protocol protocol, ImageBudget& budget) {
    if (!value.is_array()) {
        msg.content.emplace_back(text_content(value));
        return;
    }
    for (const auto& item : value.as_array()) {
        const auto& block = object(item, "content block");
        const auto type = identifier(block, "type");
        if (type == "text" || type == "input_text" || type == "output_text") {
            auto text = string(field(block, "text"), "text");
            auto* previous = msg.content.empty() ? nullptr : std::get_if<std::string>(&msg.content.back());
            if (previous) *previous += text;
            else msg.content.emplace_back(std::move(text));
        } else if (protocol == Protocol::chat && type == "image_url") {
            budget.check(msg.role);
            const auto& image = object(field(block, "image_url"), "image_url");
            budget.append(msg, decode_image_data_uri(identifier(image, "url")));
        } else if (protocol == Protocol::responses && type == "input_image") {
            budget.check(msg.role);
            budget.append(msg, decode_image_data_uri(identifier(block, "image_url")));
        } else throw std::invalid_argument("Unsupported OpenAI content block: " + type);
    }
}

JsonObject arguments(const JsonObject& obj) {
    const auto encoded = string(field(obj, "arguments"), "arguments");
    try {
        return object(JsonValue::parse(encoded), "arguments");
    } catch (const std::runtime_error&) {
        throw std::invalid_argument("arguments must encode a valid JSON object");
    }
}

void chat_messages(AgentRequest& req, const JsonValue& input, ImageBudget& budget) {
    for (const auto& item : array(input, "messages")) {
        const auto& o = object(item, "message");
        Message msg{role(identifier(o, "role")), {}};
        const auto& content = field(o, "content");
        if (msg.role == Role::tool) {
            msg.content.emplace_back(ToolResult{identifier(o, "tool_call_id"), optional_string(o, "name"),
                                                text_content(content)});
        } else if (!content.is_null() || msg.role != Role::assistant) {
            openai_content(msg, content, Protocol::chat, budget);
        }
        const auto& calls = field(o, "tool_calls");
        if (!calls.is_null()) {
            if (msg.role != Role::assistant) throw std::invalid_argument("tool_calls require an assistant message");
            for (const auto& call : array(calls, "tool_calls")) {
                const auto& c = object(call, "tool call");
                if (identifier(c, "type") != "function") throw std::invalid_argument("Expected a function tool call");
                const auto& fn = object(field(c, "function"), "function");
                msg.content.emplace_back(ToolCall{identifier(c, "id"), identifier(fn, "name"), arguments(fn)});
            }
        }
        req.messages.push_back(std::move(msg));
    }
}

void responses_input(AgentRequest& req, const JsonValue& input, ImageBudget& budget) {
    if (input.is_string()) {
        req.messages.push_back({Role::user, {input.as_string()}});
        return;
    }
    for (const auto& item : array(input, "input")) {
        const auto& o = object(item, "input item");
        const auto type = optional_string(o, "type");
        if (type.empty() || type == "message") {
            auto r = role(identifier(o, "role"));
            if (r == Role::tool) throw std::invalid_argument("Use function_call_output for tool results");
            Message msg{r, {}};
            openai_content(msg, field(o, "content"), Protocol::responses, budget);
            req.messages.push_back(std::move(msg));
        } else if (type == "function_call") {
            ToolCall call{identifier(o, "call_id"), identifier(o, "name"), arguments(o)};
            if (req.messages.empty() || req.messages.back().role != Role::assistant) {
                req.messages.push_back({Role::assistant, {}});
            }
            req.messages.back().content.emplace_back(std::move(call));
        } else if (type == "function_call_output") {
            req.messages.push_back({Role::tool, {ToolResult{identifier(o, "call_id"), {},
                                                           text_content(field(o, "output"))}}});
        } else {
            throw std::invalid_argument("Unsupported Responses input item: " + type);
        }
    }
}

void anthropic_messages(AgentRequest& req, const JsonValue& input, ImageBudget& budget) {
    for (const auto& item : array(input, "messages")) {
        const auto& o = object(item, "message");
        Message msg{role(identifier(o, "role")), {}};
        if (msg.role != Role::user && msg.role != Role::assistant) {
            throw std::invalid_argument("Messages expects user and assistant roles");
        }
        const auto& content = field(o, "content");
        if (content.is_string()) {
            msg.content.emplace_back(content.as_string());
        } else {
            for (const auto& entry : array(content, "content")) {
                const auto& b = object(entry, "content block");
                const auto type = identifier(b, "type");
                if (type == "text") {
                    msg.content.emplace_back(string(field(b, "text"), "text"));
                } else if (type == "image") {
                    budget.check(msg.role);
                    const auto& source = object(field(b, "source"), "image source");
                    if (identifier(source, "type") != "base64" || !identifier(source, "media_type").starts_with("image/"))
                        throw std::invalid_argument("Expected an Anthropic base64 image source");
                    budget.append(msg, decode_image_base64(identifier(source, "data")));
                } else if (type == "tool_use" && msg.role == Role::assistant) {
                    msg.content.emplace_back(ToolCall{identifier(b, "id"), identifier(b, "name"),
                                                     object(field(b, "input"), "input")});
                } else if (type == "tool_result" && msg.role == Role::user) {
                    msg.content.emplace_back(ToolResult{identifier(b, "tool_use_id"), {},
                        field(b, "content").is_null() ? std::string{} : text_content(field(b, "content")),
                        boolean(b, "is_error")});
                } else {
                    throw std::invalid_argument("Unsupported Messages content block: " + type);
                }
            }
        }
        req.messages.push_back(std::move(msg));
    }
}

void parse_tools(AgentRequest& req, const JsonObject& obj, Protocol protocol) {
    const auto& tools = field(obj, "tools");
    if (!tools.is_null()) {
        for (const auto& entry : array(tools, "tools")) {
            const auto& t = object(entry, "tool");
            if (protocol != Protocol::messages && identifier(t, "type") != "function") {
                throw std::invalid_argument("Only function tools are supported");
            }
            const auto& fn = protocol == Protocol::chat ? object(field(t, "function"), "function") : t;
            const auto& schema = field(fn, protocol == Protocol::messages ? "input_schema" : "parameters");
            ToolDef tool{identifier(fn, "name"), optional_string(fn, "description"),
                         schema.is_null() ? JsonObject{{"type", "object"}} : object(schema, "tool schema")};
            if (std::any_of(req.tools.begin(), req.tools.end(), [&](const ToolDef& other) {
                return other.name == tool.name;
            })) throw std::invalid_argument("Duplicate tool name: " + tool.name);
            req.tools.push_back(std::move(tool));
        }
    }
    const auto& choice = field(obj, "tool_choice");
    if (!choice.is_null()) {
        const auto type = choice.is_string() ? choice.as_string() : identifier(object(choice, "tool_choice"), "type");
        if (type == "auto") req.choice = Choice::automatic;
        else if (type == "none") req.choice = Choice::none;
        else if ((type == "any" && protocol == Protocol::messages) || type == "required") req.choice = Choice::required;
        else if (type == (protocol == Protocol::messages ? "tool" : "function") && choice.is_object()) {
            const auto& c = choice.as_object();
            req.tool_name = protocol == Protocol::chat ? identifier(object(field(c, "function"), "function"), "name")
                                                      : identifier(c, "name");
            req.choice = Choice::named;
        } else throw std::invalid_argument("Unsupported tool_choice");
    }
    if (req.choice == Choice::required && req.tools.empty()) throw std::invalid_argument("tool_choice requires tools");
    if (req.choice == Choice::named && std::none_of(req.tools.begin(), req.tools.end(), [&](const ToolDef& t) {
        return t.name == req.tool_name;
    })) throw std::invalid_argument("tool_choice names an unknown tool");
}

void match_results(AgentRequest& req) {
    std::map<std::string, std::string> outstanding;
    for (auto& msg : req.messages) {
        for (auto& block : msg.content) {
            if (const auto* call = std::get_if<ToolCall>(&block)) {
                if (!outstanding.emplace(call->id, call->name).second) {
                    throw std::invalid_argument("Duplicate pending tool call ID");
                }
            } else if (auto* result = std::get_if<ToolResult>(&block)) {
                const auto match = outstanding.find(result->call_id);
                if (match == outstanding.end() || (!result->name.empty() && result->name != match->second)) {
                    throw std::invalid_argument("Tool result does not match a preceding tool call");
                }
                result->name = match->second;
                outstanding.erase(match);
            }
        }
    }
}

}  // namespace

GenerationParams parse_generation(const JsonObject& obj, const char* limit_key) {
    GenerationParams params;
    params.max_tokens = static_cast<int>(integer(obj, limit_key, params.max_tokens, std::numeric_limits<int>::max()));
    auto number = [&](const char* key, float& out) {
        const auto& v = field(obj, key);
        if (v.is_null()) return;
        if (!v.is_number()) throw std::invalid_argument(std::string(key) + " must be a number");
        out = static_cast<float>(v.as_double());
    };
    number("temperature", params.sampling.temperature);
    number("top_p", params.sampling.top_p);
    number("min_p", params.sampling.min_p);
    number("repeat_penalty", params.sampling.repeat_penalty);
    params.sampling.top_k = static_cast<int>(integer(obj, "top_k", params.sampling.top_k, std::numeric_limits<int>::max()));
    params.sampling.seed = static_cast<uint64_t>(integer(obj, "seed", params.sampling.seed, INT64_MAX));
    params.sampling.validate();
    params.stream = boolean(obj, "stream");
    const auto& configured_stops = field(obj, "stop_sequences");
    const auto& stop = configured_stops.is_null() ? field(obj, "stop") : configured_stops;
    if (stop.is_string()) params.stops.push_back(stop.as_string());
    else if (!stop.is_null()) {
        for (const auto& v : array(stop, "stop sequences")) params.stops.push_back(string(v, "stop sequence"));
    }
    if (params.stops.size() > 4 || std::any_of(params.stops.begin(), params.stops.end(), [](const auto& s) {
        return s.empty();
    })) throw std::invalid_argument("Expected up to four nonempty stop strings");
    return params;
}

AgentRequest parse_request(const JsonObject& obj, Protocol protocol, bool allow_images) {
    AgentRequest req;
    ImageBudget budget{allow_images};
    const char* limit = protocol == Protocol::responses ? "max_output_tokens" : "max_tokens";
    if (protocol == Protocol::chat && !field(obj, "max_completion_tokens").is_null()) limit = "max_completion_tokens";
    req.generation = parse_generation(obj, limit);
    if (!field(obj, "model").is_null()) (void)identifier(obj, "model");
    if (protocol == Protocol::responses) {
        if (!field(obj, "previous_response_id").is_null()) {
            throw std::invalid_argument("previous_response_id is unavailable; resend the full input history");
        }
        const auto instructions = optional_string(obj, "instructions");
        if (!instructions.empty()) req.messages.push_back({Role::system, {instructions}});
        responses_input(req, field(obj, "input"), budget);
    } else if (protocol == Protocol::messages) {
        const auto& system = field(obj, "system");
        if (!system.is_null()) req.messages.push_back({Role::system, {text_content(system)}});
        anthropic_messages(req, field(obj, "messages"), budget);
    } else {
        chat_messages(req, field(obj, "messages"), budget);
        const auto& options = field(obj, "stream_options");
        if (!options.is_null()) req.stream_usage = boolean(object(options, "stream_options"), "include_usage");
    }
    if (std::none_of(req.messages.begin(), req.messages.end(), [](const Message& m) { return m.role != Role::system; })) {
        throw std::invalid_argument("Expected conversation input");
    }
    parse_tools(req, obj, protocol);
    if (protocol != Protocol::messages) {
        const auto* format = &field(obj, "response_format");
        bool nested_schema = true;
        if (protocol == Protocol::responses && format->is_null() && !field(obj, "text").is_null()) {
            format = &field(object(field(obj, "text"), "text"), "format");
            nested_schema = false;
        }
        if (!format->is_null()) {
            const auto& config = object(*format, "response format");
            const auto type = identifier(config, "type");
            if (type == "json_object" || type == "json_schema") {
                auto instruction = std::string("Respond with a single JSON object.");
                req.generation.json = JsonMode::object;
                if (type == "json_schema") {
                    const auto& schema_format = nested_schema && !field(config, "json_schema").is_null()
                        ? object(field(config, "json_schema"), "json_schema") : config;
                    const auto& schema = field(schema_format, "schema");
                    const auto& root = field(object(schema, "schema"), "type");
                    if (!root.is_string() || root.as_string() != "object") req.generation.json = JsonMode::value;
                    instruction = "Respond with a single JSON value matching this schema: " + schema.dump(0);
                }
                if (!req.generation.stops.empty() || req.choice == Choice::required || req.choice == Choice::named) {
                    throw std::invalid_argument("JSON response formats cannot be combined with stop sequences or required tools");
                }
                req.choice = Choice::none;
                req.messages.insert(req.messages.begin(), {Role::system, {std::move(instruction)}});
            } else if (type != "text") {
                throw std::invalid_argument("Unsupported response format: " + type);
            }
        }
    }
    match_results(req);
    return req;
}

}  // namespace fusionrt::agent
