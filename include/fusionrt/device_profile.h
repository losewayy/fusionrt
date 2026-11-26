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

    // An empty path checks FUSIONRT_DEVICE_PROFILE, then uses built-in defaults.
    static DeviceProfile load(std::string path = {});
};

}  // namespace fusionrt
