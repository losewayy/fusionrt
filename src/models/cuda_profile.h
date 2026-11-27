#pragma once

#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#include <functional>

namespace fusionrt {

struct ProfAccum {
    bool enabled = false;
    double x16_ms = 0, w16_ms = 0, cublas_ms = 0, attn_ms = 0, gdn_ms = 0, other_ms = 0, lm_ms = 0;
    int calls_cublas = 0;
    void reset() { x16_ms = w16_ms = cublas_ms = attn_ms = gdn_ms = other_ms = lm_ms = 0; calls_cublas = 0; }
};
ProfAccum& prof_acc();
double time_cuda_fn(cudaStream_t stream, const std::function<void()>& fn);

}  // namespace fusionrt
#endif
