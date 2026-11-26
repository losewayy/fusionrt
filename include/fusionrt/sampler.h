#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace fusionrt {

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
    static int32_t sample(const float* logits, int vocab_size, const std::vector<int32_t>& recent_tokens,
                          const SamplingParams& params, std::mt19937_64& rng);
};

class Utf8StreamDecoder {
public:
    std::string push(std::string_view bytes);
    std::string finish();

private:
    std::string pending_;
};

}  // namespace fusionrt
