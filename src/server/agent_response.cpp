#include "agent.h"

#include <algorithm>

namespace fusionrt::agent {
namespace {

JsonObject function(const ToolCall& call) {
    return {{"name", call.name}, {"arguments", JsonValue(call.arguments).dump(0)}};
}

JsonObject chat_call(const ToolCall& call) {
    return {{"id", call.id}, {"type", "function"}, {"function", function(call)}};
}

JsonObject response_item(const Block& block, const std::string& id, bool done) {
    JsonObject item{{"id", id}, {"status", done ? "completed" : "in_progress"}};
    if (const auto* call = std::get_if<ToolCall>(&block)) {
        item["type"] = "function_call";
        item["call_id"] = call->id;
        item["name"] = call->name;
        item["arguments"] = done ? JsonValue(call->arguments).dump(0) : "";
    } else {
        item["type"] = "message";
        item["role"] = "assistant";
        item["content"] = done ? JsonArray{JsonObject{{"type", "output_text"},
                                        {"text", std::get<std::string>(block)}, {"annotations", JsonArray{}}}}
                               : JsonArray{};
    }
    return item;
}

JsonObject messages_block(const Block& block, bool done) {
    if (const auto* call = std::get_if<ToolCall>(&block)) {
        return {{"type", "tool_use"}, {"id", call->id}, {"name", call->name},
                {"input", done ? call->arguments : JsonObject{}}};
    }
    return {{"type", "text"}, {"text", done ? std::get<std::string>(block) : ""}};
}

JsonObject chat_usage(const AgentResult& result) {
    return {{"prompt_tokens", result.input_tokens}, {"completion_tokens", result.output_tokens},
            {"total_tokens", result.input_tokens + result.output_tokens}};
}

const char* chat_reason(const AgentResult& result) {
    if (result.has_calls()) return "tool_calls";
    return result.finish == Finish::length ? "length" : "stop";
}

const char* messages_reason(const AgentResult& result) {
    if (result.has_calls()) return "tool_use";
    return result.finish == Finish::length ? "max_tokens" : "end_turn";
}

}  // namespace

bool AgentResult::has_calls() const {
    return std::any_of(output.begin(), output.end(), [](const Block& b) { return std::holds_alternative<ToolCall>(b); });
}

AgentResponse::AgentResponse(Protocol protocol, std::string id, std::string model, int64_t created,
                             bool stream_usage, EventSink event)
    : protocol_(protocol), id_(std::move(id)), model_(std::move(model)), created_(created),
      event_(std::move(event)), stream_usage_(stream_usage) {}

void AgentResponse::send(std::string_view event, JsonObject payload) {
    if (!event_ || !connected_) return;
    if (!event.empty()) payload["type"] = std::string(event);
    if (protocol_ == Protocol::responses) payload["sequence_number"] = sequence_++;
    const std::string data = JsonValue(std::move(payload)).dump(0);
    send(event, std::string_view(data));
}

void AgentResponse::send(std::string_view event, std::string_view data) {
    if (event_ && connected_) connected_ = event_(event, data);
}

std::string AgentResponse::item_id(size_t index) const {
    const char* prefix = std::holds_alternative<ToolCall>(result_.output[index]) ? "fc_" : "msg_";
    return prefix + id_ + "_" + std::to_string(index);
}

JsonObject AgentResponse::chat_chunk(JsonObject delta, JsonValue finish_reason) const {
    JsonObject result{{"id", id_}, {"model", model_}, {"created", created_}, {"object", "chat.completion.chunk"},
        {"choices", JsonArray{JsonObject{{"index", 0}, {"delta", std::move(delta)},
                                         {"finish_reason", std::move(finish_reason)}}}}};
    if (stream_usage_) result["usage"] = nullptr;
    return result;
}

JsonObject AgentResponse::responses_body(const char* status) const {
    JsonArray output;
    for (size_t i = 0; i < result_.output.size(); ++i) {
        output.emplace_back(response_item(result_.output[i], item_id(i), true));
    }
    JsonObject body{{"id", id_}, {"object", "response"}, {"created_at", created_}, {"model", model_},
                    {"status", status}, {"output", std::move(output)}, {"error", nullptr},
                    {"incomplete_details", nullptr}, {"usage", nullptr}};
    if (std::string_view(status) != "in_progress") {
        body["usage"] = JsonObject{{"input_tokens", result_.input_tokens},
            {"input_tokens_details", JsonObject{{"cached_tokens", 0}}},
            {"output_tokens", result_.output_tokens}, {"output_tokens_details", JsonObject{{"reasoning_tokens", 0}}},
            {"total_tokens", result_.input_tokens + result_.output_tokens}};
        if (std::string_view(status) == "incomplete") {
            body["incomplete_details"] = JsonObject{{"reason", "max_output_tokens"}};
        }
    }
    return body;
}

JsonObject AgentResponse::messages_body(bool starting) const {
    JsonArray content;
    if (!starting) {
        for (const auto& block : result_.output) content.emplace_back(messages_block(block, true));
    }
    return {{"id", id_}, {"type", "message"}, {"role", "assistant"}, {"model", model_},
            {"content", std::move(content)}, {"stop_reason", starting ? JsonValue() : messages_reason(result_)},
            {"stop_sequence", nullptr}, {"usage", JsonObject{{"input_tokens", result_.input_tokens},
                                            {"output_tokens", starting ? 0 : result_.output_tokens}}}};
}

void AgentResponse::start(int input_tokens) {
    result_.input_tokens = input_tokens;
    if (!event_) return;
    if (protocol_ == Protocol::chat) send("", chat_chunk({{"role", "assistant"}}, nullptr));
    else if (protocol_ == Protocol::responses) send("response.created", {{"response", responses_body("in_progress")}});
    else send("message_start", {{"message", messages_body(true)}});
}

void AgentResponse::text(std::string_view text) {
    if (text.empty()) return;
    const auto index = static_cast<int64_t>(text_open_ ? result_.output.size() - 1 : result_.output.size());
    if (!text_open_) {
        result_.output.emplace_back(std::string{});
        text_open_ = true;
        if (event_ && protocol_ == Protocol::responses) {
            const auto item = item_id(index);
            send("response.output_item.added", {{"output_index", index},
                {"item", response_item(result_.output.back(), item, false)}});
            send("response.content_part.added", {{"output_index", index}, {"item_id", item}, {"content_index", 0},
                {"part", JsonObject{{"type", "output_text"}, {"text", ""}, {"annotations", JsonArray{}}}}});
        } else if (event_ && protocol_ == Protocol::messages) {
            send("content_block_start", {{"index", index}, {"content_block", messages_block(result_.output.back(), false)}});
        }
    }
    std::get<std::string>(result_.output.back()) += text;
    if (!event_) return;
    if (protocol_ == Protocol::chat) send("", chat_chunk({{"content", std::string(text)}}, nullptr));
    else if (protocol_ == Protocol::responses) {
        send("response.output_text.delta", {{"output_index", index}, {"item_id", item_id(index)},
            {"content_index", 0}, {"delta", std::string(text)}, {"logprobs", JsonArray{}}});
    } else {
        send("content_block_delta", {{"index", index}, {"delta", JsonObject{{"type", "text_delta"},
                                                                         {"text", std::string(text)}}}});
    }
}

void AgentResponse::close_text() {
    if (!text_open_) return;
    text_open_ = false;
    if (!event_) return;
    const auto index = static_cast<int64_t>(result_.output.size() - 1);
    if (protocol_ == Protocol::responses) {
        const auto item = item_id(index);
        const auto& text = std::get<std::string>(result_.output.back());
        send("response.output_text.done", {{"output_index", index}, {"item_id", item}, {"content_index", 0},
            {"text", text}, {"logprobs", JsonArray{}}});
        send("response.content_part.done", {{"output_index", index}, {"item_id", item}, {"content_index", 0},
            {"part", JsonObject{{"type", "output_text"}, {"text", text}, {"annotations", JsonArray{}}}}});
        send("response.output_item.done", {{"output_index", index},
            {"item", response_item(result_.output.back(), item, true)}});
    } else if (protocol_ == Protocol::messages) {
        send("content_block_stop", JsonObject{{"index", index}});
    }
}

void AgentResponse::call(ToolCall call) {
    close_text();
    const int call_index = next_call_++;
    call.id = (protocol_ == Protocol::messages ? "toolu_" : "call_") + id_ + "_" + std::to_string(call_index);
    const auto index = static_cast<int64_t>(result_.output.size());
    result_.output.emplace_back(std::move(call));
    if (!event_) return;
    const auto& saved = std::get<ToolCall>(result_.output.back());
    const auto arguments = JsonValue(saved.arguments).dump(0);
    if (protocol_ == Protocol::chat) {
        auto value = chat_call(saved);
        value["index"] = call_index;
        send("", chat_chunk({{"tool_calls", JsonArray{std::move(value)}}}, nullptr));
    } else if (protocol_ == Protocol::responses) {
        const auto item = item_id(index);
        send("response.output_item.added", {{"output_index", index}, {"item", response_item(saved, item, false)}});
        send("response.function_call_arguments.delta", {{"output_index", index}, {"item_id", item}, {"delta", arguments}});
        send("response.function_call_arguments.done", {{"output_index", index}, {"item_id", item},
                                                       {"arguments", arguments}, {"name", saved.name}});
        send("response.output_item.done", {{"output_index", index}, {"item", response_item(saved, item, true)}});
    } else {
        send("content_block_start", {{"index", index}, {"content_block", messages_block(saved, false)}});
        send("content_block_delta", {{"index", index},
            {"delta", JsonObject{{"type", "input_json_delta"}, {"partial_json", arguments}}}});
        send("content_block_stop", JsonObject{{"index", index}});
    }
}

void AgentResponse::finish(Finish reason, int output_tokens) {
    result_.finish = reason;
    result_.output_tokens = output_tokens;
    close_text();
    if (!event_) return;
    if (protocol_ == Protocol::chat) {
        send("", chat_chunk({}, chat_reason(result_)));
        if (stream_usage_) {
            auto chunk = chat_chunk({}, nullptr);
            chunk["choices"] = JsonArray{};
            chunk["usage"] = chat_usage(result_);
            send("", std::move(chunk));
        }
        send("", std::string_view("[DONE]"));
    } else if (protocol_ == Protocol::responses) {
        const bool incomplete = reason == Finish::length && !result_.has_calls();
        send(incomplete ? "response.incomplete" : "response.completed",
            {{"response", responses_body(incomplete ? "incomplete" : "completed")}});
    } else {
        send("message_delta", {{"delta", JsonObject{{"stop_reason", messages_reason(result_)}, {"stop_sequence", nullptr}}},
            {"usage", JsonObject{{"output_tokens", output_tokens}}}});
        send("message_stop", JsonObject{});
    }
}

void AgentResponse::error(std::string_view message) {
    JsonObject detail{{"message", std::string(message)}, {"type", "server_error"}};
    if (protocol_ == Protocol::chat) {
        send("", {{"error", std::move(detail)}});
        send("", std::string_view("[DONE]"));
    } else if (protocol_ == Protocol::responses) {
        send("error", {{"code", "server_error"}, {"message", std::string(message)}, {"param", nullptr}});
    } else {
        detail["type"] = "api_error";
        send("error", {{"error", std::move(detail)}});
    }
}

JsonValue AgentResponse::body() const {
    if (protocol_ == Protocol::messages) return messages_body(false);
    if (protocol_ == Protocol::responses) {
        return responses_body(result_.finish == Finish::length && !result_.has_calls() ? "incomplete" : "completed");
    }
    std::string text;
    JsonArray calls;
    for (const auto& block : result_.output) {
        if (const auto* call = std::get_if<ToolCall>(&block)) calls.emplace_back(chat_call(*call));
        else text += std::get<std::string>(block);
    }
    JsonObject message{{"role", "assistant"}, {"content", text.empty() && !calls.empty() ? JsonValue() : JsonValue(text)}};
    if (!calls.empty()) message["tool_calls"] = std::move(calls);
    return JsonObject{{"id", id_}, {"object", "chat.completion"}, {"created", created_}, {"model", model_},
        {"choices", JsonArray{JsonObject{{"index", 0}, {"message", std::move(message)},
                                         {"finish_reason", chat_reason(result_)}}}}, {"usage", chat_usage(result_)}};
}

}  // namespace fusionrt::agent
