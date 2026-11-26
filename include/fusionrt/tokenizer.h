#pragma once

#include "fusionrt/gguf.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusionrt {

struct ChatMessage {
    std::string role;
    std::string content;
};

class BpeTokenizer {
public:
    BpeTokenizer() = default;

    // Initialize from GGUF metadata (tokenizer.ggml.tokens / merges)
    bool load_from_gguf(const GgufFile& gguf);

    // Initialize from HuggingFace tokenizer.json
    bool load_from_json(const std::string& path, std::string* err = nullptr);

    // Initialize a minimal byte-level fallback vocab (256 byte tokens + special tokens)
    void init_byte_fallback(int vocab_size = 512);

    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string decode_token(int32_t id) const;
    std::string format_chat_messages(const std::vector<ChatMessage>& msgs,
                                     bool add_generation_prompt = true) const;
    bool is_end_of_generation(int32_t id) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    bool add_bos_default() const { return add_bos_default_; }
    bool has_chatml() const { return token_to_id_.count("<|im_start|>") && token_to_id_.count("<|im_end|>"); }
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int> merge_ranks_;
    std::vector<std::string> special_tokens_;
    std::string chat_template_;
    std::string pretokenizer_;
    std::vector<int32_t> end_ids_;
    std::vector<int32_t> token_types_;
    std::vector<float> token_scores_;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t unk_id_ = 3;
    bool add_bos_default_ = true;
    bool add_space_prefix_ = false;
    bool sentencepiece_ = false;
    bool byte_fallback_only_ = false;
    bool gpt2_byte_bpe_ = false;
};

}  // namespace fusionrt
