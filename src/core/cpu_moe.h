#pragma once

#include "fusionrt/tensor.h"

namespace fusionrt {

enum class CpuMoeMode { Parallel, Scalar };

// The gate tensor also holds up when the separate up tensor is empty.
void cpu_moe_accumulate(const HostTensor& gate, const HostTensor& up, const HostTensor& down,
                        int expert, const float* input, float weight, float* output,
                        CpuMoeMode mode = CpuMoeMode::Parallel);

using CpuMoeDot = float (*)(const uint8_t*, const float*, int);
#ifdef FUSIONRT_CPU_MOE_AVX2
CpuMoeDot cpu_moe_avx2_dot(DType dtype);
#endif

}  // namespace fusionrt
