#include "fusionrt/sampler.h"

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

int32_t Sampler::sample(const float* logits, int vocab_size, const std::vector<int32_t>& recent_tokens,
                        const SamplingParams& params, std::mt19937_64& rng) {
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
    if (candidates.empty()) return 0;
    auto better = [](const Candidate& a, const Candidate& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    size_t k = params.temperature <= 0 ? 1 : candidates.size();
    if (params.top_k > 0) k = std::min(k, static_cast<size_t>(params.top_k));
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), better);
    if (k == 1) return candidates.front().second;
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

    double draw = std::generate_canonical<double, 53>(rng) * retained;
    for (size_t i = 0; i < kept; ++i) {
        draw -= candidates[i].first;
        if (draw < 0) return candidates[i].second;
    }
    return candidates[kept - 1].second;
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
