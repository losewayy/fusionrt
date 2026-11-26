#pragma once

#include "fusionrt/json.h"
#include "fusionrt/image.h"
#include "fusionrt/sampler.h"
#include "fusionrt/tokenizer.h"

#include <functional>
#include <string_view>

namespace fusionrt::agent {

enum class Protocol { chat, responses, messages };
enum class Role { system, user, assistant, tool };

struct ToolDef {
    std::string name, description;
    JsonObject parameters;
};

struct ToolCall {
    std::string id, name;
    JsonObject arguments;
};

struct ToolResult {
    std::string call_id, name, content;
    bool is_error = false;
};

using Block = std::variant<std::string, ToolCall, ToolResult, Image>;

struct Message {
    Role role;
    std::vector<Block> content;
};

enum class Choice { automatic, none, required, named };
enum class JsonMode { none, value, object };

struct GenerationParams {
    int max_tokens = 512;
    SamplingParams sampling;
    std::vector<std::string> stops;
    bool stream = false;
    JsonMode json = JsonMode::none;
};

struct AgentRequest {
    std::vector<Message> messages;
    std::vector<ToolDef> tools;
    Choice choice = Choice::automatic;
    std::string tool_name;
    GenerationParams generation;
    bool stream_usage = false;
};

struct AgentPrompt {
    std::string text;
    std::string call_prefix;
    bool open_thought = false;
    bool llama_calls = false;
    std::vector<const Image*> images;
};

const JsonValue& field(const JsonObject& obj, const char* key);
GenerationParams parse_generation(const JsonObject& obj, const char* limit_key);
AgentRequest parse_request(const JsonObject& obj, Protocol protocol, bool allow_images = false);
AgentPrompt render_prompt(const AgentRequest& req, const BpeTokenizer& tokenizer,
                           std::span<const Image> preset_images = {});

class ToolParser {
public:
    using TextSink = std::function<void(std::string_view)>;
    using CallSink = std::function<void(ToolCall)>;

    ToolParser(const AgentRequest& request, const AgentPrompt& prompt, TextSink text, CallSink call);
    void push(std::string_view chunk);
    void finish();

private:
    enum class Mode { text, thought, tagged, json };
    void emit_text(std::string_view text);
    bool emit_call(std::string_view json);
    void drain(bool final);

    const AgentRequest& request_;
    TextSink text_;
    CallSink call_;
    std::string pending_;
    Mode mode_ = Mode::text;
    size_t scanned_ = 0;
    int depth_ = 0;
    bool quoted_ = false, escape_ = false;
    bool llama_calls_ = false, explicit_json_ = false, saw_text_ = false;
};

enum class Finish { stop, length };

struct AgentResult {
    std::vector<Block> output;
    int input_tokens = 0, output_tokens = 0;
    Finish finish = Finish::stop;

    bool has_calls() const;
};

// An empty event name denotes a data-only SSE frame.
using EventSink = std::function<bool(std::string_view, std::string_view)>;
class AgentResponse {
public:
    AgentResponse(Protocol protocol, std::string id, std::string model, int64_t created,
                  bool stream_usage, EventSink event = {});
    void start(int input_tokens);
    void text(std::string_view text);
    void call(ToolCall call);
    void finish(Finish reason, int output_tokens);
    void error(std::string_view message);
    JsonValue body() const;
    bool connected() const { return connected_; }
    const AgentResult& result() const { return result_; }

private:
    void send(std::string_view event, JsonObject payload);
    void send(std::string_view event, std::string_view data);
    void close_text();
    JsonObject chat_chunk(JsonObject delta, JsonValue finish_reason) const;
    JsonObject responses_body(const char* status) const;
    JsonObject messages_body(bool starting) const;
    std::string item_id(size_t index) const;

    Protocol protocol_;
    std::string id_, model_;
    int64_t created_;
    EventSink event_;
    AgentResult result_;
    int sequence_ = 0, next_call_ = 0;
    bool connected_ = true, text_open_ = false, stream_usage_ = false;
};

}  // namespace fusionrt::agent
