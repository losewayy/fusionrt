#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fusionrt {

inline void validate_gdn_shape(int64_t taps, int64_t dk, int64_t dv, int nk, int nv) {
    if (taps == 4 && dk >= 1 && dk <= 128 && dv >= 1 && dv <= 256 &&
        nk > 0 && nv > 0 && nv % nk == 0) return;
    throw std::invalid_argument("Unsupported GDN shape (conv_kernel=" + std::to_string(taps) +
        ", dk=" + std::to_string(dk) + ", dv=" + std::to_string(dv) + ", key_heads=" + std::to_string(nk) +
        ", value_heads=" + std::to_string(nv) +
        "). Supported: conv_kernel=4, dk=1..128, dv=1..256, positive heads with value_heads divisible by key_heads");
}

}  // namespace fusionrt
