#include "fusionrt/device_profile.h"
#include "fusionrt/json.h"

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fusionrt {

DeviceProfile DeviceProfile::load(std::string path) {
    if (path.empty()) {
        if (const char* env = std::getenv("FUSIONRT_DEVICE_PROFILE")) path = env;
    }
    DeviceProfile out;
    if (path.empty()) return out;
    std::ifstream input(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open device profile: " + path);
    std::ostringstream text;
    text << input.rdbuf();
    const auto root = JsonValue::parse(text.str());
    if (!root.is_object()) throw std::runtime_error("Expected a device_profile.v1 object");
    const auto& fields = root.as_object();
    auto schema = fields.find("schema_version");
    if (schema == fields.end() || !schema->second.is_string() ||
        schema->second.as_string() != "device_profile.v1") {
        throw std::runtime_error("Expected schema_version device_profile.v1: " + path);
    }
    auto integer = [&](const char* name, int64_t fallback, int64_t min, int64_t max) {
        auto it = fields.find(name);
        if (it == fields.end()) return fallback;
        const auto& value = it->second;
        if (!value.is_int() || value.as_int() < min || value.as_int() > max) {
            throw std::runtime_error(std::string("Invalid device profile field: ") + name);
        }
        return value.as_int();
    };
    out.mmq_crossover_m = static_cast<int>(integer("mmq_crossover_m", out.mmq_crossover_m, 1, INT_MAX));
    out.empirical_l2_bytes = static_cast<uint64_t>(integer("empirical_l2_bytes", 0, 0, INT64_MAX));
    out.preferred_gemv_warps = static_cast<int>(integer("preferred_gemv_warps", 0, 0, INT_MAX));
    out.preferred_gemv_rows_per_warp = static_cast<int>(integer("preferred_gemv_rows_per_warp", 0, 0, INT_MAX));
    if (auto it = fields.find("recommend_cuda_graph"); it != fields.end()) {
        if (!it->second.is_bool()) throw std::runtime_error("Invalid recommend_cuda_graph");
        out.recommend_cuda_graph = std::get<bool>(it->second.val);
    }
    return out;
}

}  // namespace fusionrt
