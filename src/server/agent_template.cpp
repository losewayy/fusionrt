#include "agent.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

namespace fusionrt::agent {
namespace {

constexpr std::string_view tool_open = "<tool_call>", tool_close = "</tool_call>";
constexpr std::string_view think_open = "<think>", think_close = "</think>";
constexpr std::string_view python_tag = "<|python_tag|>";

size_t partial_suffix(std::string_view s, std::string_view marker) {
    for (size_t n = std::min(s.size(), marker.size() - 1); n; --n) {
        if (s.ends_with(marker.substr(0, n))) return n;
    }
    return 0;
}

std::string dump_call(const ToolCall& call, bool llama) {
    return JsonValue(JsonObject{{"name", call.name},
                                {llama ? "parameters" : "arguments", call.arguments}}).dump(0);
}

std::string tool_instructions(const AgentRequest& req, bool llama) {
    if (req.tools.empty() || req.choice == Choice::none) return {};
    std::string tools;
    for (const auto& tool : req.tools) {
        JsonObject def{{"name", tool.name}, {"description", tool.description}, {"parameters", tool.parameters}};
        tools += JsonValue(JsonObject{{"type", "function"}, {"function", std::move(def)}}).dump(llama ? 4 : 0);
        tools += llama ? "\n\n" : "\n";
    }
    if (llama) {
        return "Given the following functions, please respond with a JSON for a function call with its proper "
               "arguments that best answers the given prompt.\n\n"
               "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name "
               "and its value}.Do not use variables.\n\n" + tools;
    }
    return "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
           tools + "</tools>\n\nFor each function call, return a json object with function name and arguments "
           "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
           "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
}

}  // namespace

AgentPrompt render_prompt(const AgentRequest& req, const BpeTokenizer& tokenizer,
                           std::span<const Image> preset_images) {
    const auto format = tokenizer.chat_format();
    const bool llama = format == ChatFormat::llama;
    std::vector<ChatMessage> messages;
    AgentPrompt prompt;
    std::string system;
    bool has_system = false;
    for (const auto& msg : req.messages) {
        if (msg.role != Role::system) continue;
        for (const auto& block : msg.content) {
            if (has_system) system += "\n\n";
            system += std::get<std::string>(block);
            has_system = true;
        }
    }
    const auto instructions = tool_instructions(req, llama);
    const auto tmpl = tokenizer.chat_template();
    const bool llama_tools_in_user = llama && tmpl.find("tools_in_user_message") != std::string_view::npos;
    if (!instructions.empty() && !llama_tools_in_user) {
        if (!system.empty()) system += "\n\n";
        system += instructions;
        has_system = true;
    }
    if (has_system) messages.push_back({"system", std::move(system)});
    bool last_was_result = false;
    for (const auto& msg : req.messages) {
        if (msg.role == Role::system) continue;
        std::string content;
        std::vector<size_t> image_offsets;
        bool pending = false;
        auto add_image = [&](const Image& image) {
            image_offsets.push_back(content.size());
            content += image_placeholder;
            prompt.images.push_back(&image);
            pending = true;
        };
        if (msg.role == Role::user) {
            for (const auto& image : preset_images) add_image(image);
            preset_images = {};
        }
        auto flush = [&] {
            if (!pending) return;
            messages.push_back({msg.role == Role::assistant ? "assistant" : "user", std::move(content),
                                std::move(image_offsets)});
            content.clear();
            image_offsets.clear();
            pending = false;
            last_was_result = false;
        };
        for (const auto& block : msg.content) {
            if (const auto* t = std::get_if<std::string>(&block)) {
                content += *t;
                pending = true;
            } else if (const auto* image = std::get_if<Image>(&block)) {
                add_image(*image);
            } else if (const auto* call = std::get_if<ToolCall>(&block)) {
                if (!content.empty()) content += '\n';
                if (llama) content += dump_call(*call, true);
                else content += "<tool_call>\n" + dump_call(*call, false) + "\n</tool_call>";
                pending = true;
            } else if (const auto* result = std::get_if<ToolResult>(&block)) {
                flush();
                ChatMessage reply{llama ? "ipython" : "user", llama ? "\"" : "<tool_response>\n"};
                auto append_text = [&](std::string_view text) {
                    if (llama) {
                        const auto quoted = JsonValue(std::string(text)).dump(0);
                        reply.content.append(quoted, 1, quoted.size() - 2);
                    } else reply.content += text;
                };
                if (result->is_error) append_text("Tool error: ");
                size_t pos = 0;
                const std::string_view value = result->content;
                for (const auto& attachment : result->images) {
                    append_text(value.substr(pos, attachment.offset - pos));
                    reply.image_offsets.push_back(reply.content.size());
                    reply.content += image_placeholder;
                    prompt.images.push_back(&attachment.image);
                    pos = attachment.offset;
                }
                append_text(value.substr(pos));
                reply.content += llama ? "\"" : "\n</tool_response>";
                if (!llama && last_was_result) {
                    auto& previous = messages.back();
                    previous.content += '\n';
                    for (const auto offset : reply.image_offsets) previous.image_offsets.push_back(previous.content.size() + offset);
                    previous.content += reply.content;
                } else messages.push_back(std::move(reply));
                last_was_result = true;
            }
        }
        flush();
    }
    if (llama_tools_in_user && !instructions.empty()) {
        const auto user = std::find_if(messages.begin(), messages.end(), [](const ChatMessage& msg) {
            return msg.role == "user";
        });
        if (user != messages.end()) {
            user->content.insert(0, instructions);
            for (auto& offset : user->image_offsets) offset += instructions.size();
        }
    }
    prompt.text = tokenizer.format_chat_messages(messages, true, &prompt.image_offsets);
    if (llama_tools_in_user && !instructions.empty()) {
        const auto date = prompt.text.find("Cutting Knowledge Date: December 2023");
        if (date != std::string::npos) {
            constexpr std::string_view environment = "Environment: ipython\n";
            prompt.text.insert(date, environment);
            for (auto& offset : prompt.image_offsets) if (offset >= date) offset += environment.size();
        }
    }
    prompt.llama_calls = llama && !req.tools.empty() && req.choice != Choice::none;
    prompt.open_thought = prompt.text.ends_with("<think>\n");
    if (prompt.open_thought && req.generation.json) {
        prompt.text += "\n</think>\n\n";
        prompt.open_thought = false;
    }
    if (req.choice == Choice::required || req.choice == Choice::named) {
        if (prompt.open_thought) {
            prompt.text += "\n</think>\n\n";
            prompt.open_thought = false;
        }
        if (!llama) prompt.call_prefix = "<tool_call>\n";
        if (req.choice == Choice::named) {
            prompt.call_prefix += "{\"name\":" + JsonValue(req.tool_name).dump(0) +
                                  (llama ? ",\"parameters\":" : ",\"arguments\":");
        } else if (llama) prompt.call_prefix = "{";
        prompt.text += prompt.call_prefix;
    }
    return prompt;
}

ToolParser::ToolParser(const AgentRequest& request, const AgentPrompt& prompt, TextSink text, CallSink call, TextSink thought)
    : request_(request), text_(std::move(text)), call_(std::move(call)), thought_(std::move(thought)),
      mode_(prompt.open_thought ? Mode::thought : Mode::text), llama_calls_(prompt.llama_calls) {
    push(prompt.call_prefix);
}

void ToolParser::push(std::string_view chunk) {
    pending_ += chunk;
    drain(false);
}

void ToolParser::finish() {
    drain(true);
}

void ToolParser::emit_text(std::string_view text) {
    if (text.empty()) return;
    saw_text_ |= text.find_first_not_of(" \t\r\n") != std::string_view::npos;
    text_(text);
}

bool ToolParser::emit_call(std::string_view raw) {
    if (request_.choice == Choice::none || request_.tools.empty()) return false;
    ToolCall call;
    try {
        const auto value = JsonValue::parse(std::string(raw));
        if (!value.is_object()) return false;
        const auto& obj = value.as_object();
        const auto& name = field(obj, "name");
        if (!name.is_string()) return false;
        call.name = name.as_string();
        if (request_.choice == Choice::named && call.name != request_.tool_name) return false;
        if (std::none_of(request_.tools.begin(), request_.tools.end(),
                         [&](const ToolDef& tool) { return tool.name == call.name; })) return false;
        auto arguments = field(obj, "arguments");
        if (arguments.is_null()) arguments = field(obj, "parameters");
        if (arguments.is_string()) arguments = JsonValue::parse(arguments.as_string());
        if (!arguments.is_object()) return false;
        call.arguments = arguments.as_object();
    } catch (const std::runtime_error&) {
        return false;
    }
    call_(std::move(call));
    return true;
}

void ToolParser::drain(bool final) {
    const std::array markers{think_open, think_close, tool_open, tool_close, python_tag};
    while (!pending_.empty()) {
        const std::string_view view(pending_);
        if (mode_ == Mode::thought) {
            const auto end = view.find(think_close);
            const auto length = end != view.npos ? end : view.size() - (final ? 0 : partial_suffix(view, think_close));
            if (thought_ && length != 0) thought_(view.substr(0, length));
            if (end == view.npos) {
                pending_.erase(0, length);
                return;
            }
            pending_.erase(0, end + think_close.size());
            mode_ = Mode::text;
            continue;
        }
        if (mode_ == Mode::tagged || mode_ == Mode::json) {
            size_t end = 0, payload = 0;
            for (; scanned_ < view.size(); ++scanned_) {
                const char c = view[scanned_];
                if (quoted_) {
                    if (escape_) escape_ = false;
                    else if (c == '\\') escape_ = true;
                    else if (c == '"') quoted_ = false;
                    continue;
                }
                if (c == '"') quoted_ = true;
                if (mode_ == Mode::tagged && c == '<') {
                    const auto tail = view.substr(scanned_);
                    if (tail.starts_with(tool_close)) {
                        payload = scanned_;
                        end = scanned_ + tool_close.size();
                        break;
                    }
                    if (!final && tool_close.starts_with(tail)) return;
                } else if (mode_ == Mode::json) {
                    if (c == '{') ++depth_;
                    else if (c == '}' && --depth_ == 0) {
                        end = payload = scanned_ + 1;
                        break;
                    } else if (!depth_ && !std::isspace(static_cast<unsigned char>(c))) {
                        end = scanned_ + 1;
                        break;
                    }
                }
            }
            if (!end && !final) return;
            if (!end) end = view.size();
            const bool call = payload && emit_call(view.substr(0, payload));
            if (!call && mode_ == Mode::json && !explicit_json_) emit_text(view.substr(0, end));
            pending_.erase(0, end);
            mode_ = Mode::text;
            scanned_ = 0;
            depth_ = 0;
            quoted_ = escape_ = explicit_json_ = false;
            continue;
        }

        size_t offset = view.size(), hold = 0;
        std::string_view marker;
        for (const auto candidate : markers) {
            const size_t found = view.find(candidate);
            if (found < offset) {
                offset = found;
                marker = candidate;
            }
            hold = std::max(hold, partial_suffix(view, candidate));
        }
        const auto first = view.find_first_not_of(" \t\r\n");
        if (llama_calls_ && !saw_text_ && first < offset && view[first] == '{') {
            mode_ = Mode::json;
            continue;
        }
        if (marker.empty()) {
            emit_text(view.substr(0, view.size() - hold));
            pending_.erase(0, final ? view.size() : view.size() - hold);
            return;
        }
        emit_text(view.substr(0, offset));
        pending_.erase(0, offset + marker.size());
        if (marker == think_open) mode_ = Mode::thought;
        else if (marker == tool_open) mode_ = Mode::tagged;
        else if (marker == python_tag) {
            mode_ = Mode::json;
            explicit_json_ = true;
        }
    }
}

}  // namespace fusionrt::agent
