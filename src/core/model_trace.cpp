#include "fusionrt/models.h"
#include "fusionrt/json.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fusionrt {
namespace {

void write_trace(const std::string& path, const JsonValue& document) {
    std::ofstream out(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
    if (!(out << document.dump() << '\n')) throw std::runtime_error("Cannot write trace: " + path);
}

}  // namespace

void OpTrace::write(const std::string& path, const std::string& model) const {
    JsonArray rows;
    double total_ms = 0.0;
    for (const auto& op : ops) {
        if (op.total_ms <= 0) continue;
        total_ms += op.total_ms;
        rows.emplace_back(JsonObject{{"name", op.name}, {"kind", op.kind}, {"calls", op.calls},
                                     {"total_ms", op.total_ms}, {"bytes", op.bytes}, {"flops", op.flops}});
    }
    if (rows.empty()) throw std::runtime_error("No CUDA operator timings were recorded");
    write_trace(path, JsonObject{{"model", model}, {"mode", mode}, {"total_ms", total_ms}, {"ops", std::move(rows)}});
}

void LayerTrace::write(const std::string& path, const std::string& model) const {
    JsonArray rows;
    for (size_t layer = 0; layer < outputs.size(); ++layer) {
        const auto& output = outputs[layer];
        if (output.empty()) throw std::runtime_error("Missing output for layer " + std::to_string(layer));
        double sum = 0.0, squares = 0.0, max_abs = 0.0;
        JsonArray sample;
        for (double value : output) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite activation in layer trace");
            sum += value;
            squares += value * value;
            max_abs = std::max(max_abs, std::abs(value));
            if (sample.size() < 16) sample.emplace_back(value);
        }
        rows.emplace_back(JsonObject{{"layer", static_cast<int64_t>(layer)}, {"op", "layer_out"},
                                     {"norm_l2", std::sqrt(squares)}, {"mean", sum / output.size()},
                                     {"max_abs", max_abs}, {"sample", std::move(sample)}});
    }
    write_trace(path, JsonObject{{"model", model}, {"layers", std::move(rows)}});
}

}  // namespace fusionrt
