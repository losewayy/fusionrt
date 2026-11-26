#pragma once

#include <cstdint>
#include <string>

namespace fusionrt {

struct DeviceProfile {
    int mmq_crossover_m = 4;
    bool recommend_cuda_graph = false;
    uint64_t empirical_l2_bytes = 0;
    int preferred_gemv_warps = 0;
    int preferred_gemv_rows_per_warp = 0;

    int sm_count = 0;
    double vram_total_mb = 0;
    double measured_hbm_bw_gbps = 0;
    double measured_l2_bw_gbps = 0;
    double measured_fp16_tc_tflops = 0;
    double measured_int8_tc_tflops = 0;
    double measured_dp4a_gemv_gbps = 0;
    double launch_floor_us = 0;
    double graph_replay_us = 0;
    int empirical_l1_kb = 0;
    uint64_t reported_l2_bytes = 0;
    double host_ram_mb = 0;
    double cpu_ddr_bw_gbps = 0;
    double cpu_q4_gemv_gbps = 0;
    double pcie_h2d_gbps = 0;
    double pcie_d2h_gbps = 0;
    double pcie_latency_us = 0;

    // An empty path checks FUSIONRT_DEVICE_PROFILE, then uses built-in defaults.
    static DeviceProfile load(std::string path = {});
};

}  // namespace fusionrt
