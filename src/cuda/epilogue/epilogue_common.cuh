#pragma once

#include "../cuda_common.cuh"

namespace fusionrt::cuda_ops {
namespace {

template <bool Gelu>
__device__ __forceinline__ float gated_activation(float gate, float up) {
    if constexpr (Gelu)
        return (0.5f * gate) * (1.0f + tanhf(0.7978845608028654f * gate * (1.0f + 0.044715f * gate * gate))) * up;
    else
        return silu_f(gate) * up;
}

}  // namespace
}  // namespace fusionrt::cuda_ops
