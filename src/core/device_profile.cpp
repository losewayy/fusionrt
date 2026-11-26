#include "fusionrt/device_profile.h"
#include "fusionrt/json.h"

#include <climits>
#include <cmath>
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
    auto measurement = [&](const char* name) {
        auto it = fields.find(name);
        if (it == fields.end()) return 0.0;
        const auto& value = it->second;
        const double number = value.as_double();
        if (!value.is_number() || !std::isfinite(number) || number < 0) {
            throw std::runtime_error(std::string("Invalid device profile field: ") + name);
        }
        return number;
    };
    const auto crossover = integer("mmq_crossover_m", 0, 0, INT_MAX);
    if (crossover > 0) out.mmq_crossover_m = static_cast<int>(crossover);
    out.empirical_l2_bytes = static_cast<uint64_t>(integer("empirical_l2_bytes", 0, 0, INT64_MAX));
    out.preferred_gemv_warps = static_cast<int>(integer("preferred_gemv_warps", 0, 0, INT_MAX));
    out.preferred_gemv_rows_per_warp = static_cast<int>(integer("preferred_gemv_rows_per_warp", 0, 0, INT_MAX));
    out.sm_count = static_cast<int>(integer("sm_count", 0, 0, INT_MAX));
    out.vram_total_mb = measurement("vram_total_mb");
    out.measured_hbm_bw_gbps = measurement("measured_hbm_bw_gbps");
    out.measured_l2_bw_gbps = measurement("measured_l2_bw_gbps");
    out.measured_fp16_tc_tflops = measurement("measured_fp16_tc_tflops");
    out.measured_int8_tc_tflops = measurement("measured_int8_tc_tflops");
    out.measured_dp4a_gemv_gbps = measurement("measured_dp4a_gemv_gbps");
    out.launch_floor_us = measurement("launch_floor_us");
    out.graph_replay_us = measurement("graph_replay_us");
    out.empirical_l1_kb = static_cast<int>(integer("empirical_l1_kb", 0, 0, INT_MAX));
    out.reported_l2_bytes = static_cast<uint64_t>(integer("reported_l2_bytes", 0, 0, INT64_MAX));
    out.host_ram_mb = measurement("host_ram_mb");
    out.cpu_ddr_bw_gbps = measurement("cpu_ddr_bw_gbps");
    out.cpu_q4_gemv_gbps = measurement("cpu_q4_gemv_gbps");
    out.pcie_h2d_gbps = measurement("pcie_h2d_gbps");
    out.pcie_d2h_gbps = measurement("pcie_d2h_gbps");
    out.pcie_latency_us = measurement("pcie_latency_us");
    if (auto it = fields.find("recommend_cuda_graph"); it != fields.end()) {
        if (!it->second.is_bool()) throw std::runtime_error("Invalid recommend_cuda_graph");
        out.recommend_cuda_graph = std::get<bool>(it->second.val);
    }
    return out;
}

}  // namespace fusionrt
