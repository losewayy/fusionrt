#pragma once

#include "fusionrt/bench.h"
#include "fusionrt/weight_codec.h"
#include "fusionrt/hadamard.h"
#include <algorithm>
#include "fusionrt/cost_model.h"
#include "fusionrt/gguf.h"
#include "fusionrt/models.h"
#include "fusionrt/safetensors.h"
#include "fusionrt/server.h"
#include "fusionrt/tokenizer.h"
#include "fusionrt/vision.h"
#include "agent.h"
#include "http_request.h"
#include "../src/cli/cli_args.h"
#include "../src/core/cpu_moe.h"
#include "../src/models/model_loader.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#include "fusionrt/cuda_kernels.h"
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#include "fusionrt/vulkan_kernels.h"
#include "../src/vulkan/vulkan_memory.h"
#endif
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace fusionrt;

#ifdef FUSIONRT_ENABLE_CUDA
namespace fusionrt {
struct CudaReviewTestAccess {
    static auto& instantiate(CudaGraphExec& graph) { return graph.instantiate_; }
    static decltype(CudaKvPool::allocate_)& allocator(FusionModel& model);
};
}  // namespace fusionrt
#endif

namespace fusionrt::test {

inline double codec_rel_l2(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) throw std::runtime_error("Codec vector size mismatch");
    double error2 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - b[i];
        error2 += diff * diff;
        norm2 += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(error2 / std::max(norm2, 1e-30));
}

inline void require_codec_close(DType type, const char* op, std::span<const float> actual, std::span<const float> expected) {
    const double rel = codec_rel_l2(actual, expected);
    if (!(rel < 1e-5))
        throw std::runtime_error(std::string(dtype_name(type)) + " " + op + " relative L2=" + std::to_string(rel));
}

inline std::vector<float> codec_input(int rows, int cols) {
    std::vector<float> x(static_cast<size_t>(rows) * cols);
    std::mt19937 rng(512);
    for (auto& value : x) value = (static_cast<int>(rng() % 49) - 24) / 128.0f;
    return x;
}

inline double codec_reference_dot(const float* w, const float* x, int cols) {
    double sum = 0;
    for (int k = 0; k < cols; ++k) sum += static_cast<double>(w[k]) * x[k];
    return sum;
}

inline void expect_close(const std::vector<float>& got, const std::vector<double>& ref,
                  double max_abs_tol, double rel_l2_tol, const std::string& label) {
    auto audit = bench::audit_arrays_f32(got.data(), ref.data(), {static_cast<int64_t>(got.size())},
                                         "fp32", max_abs_tol, rel_l2_tol);
    std::cout << "  [PASS/CHECK] " << label
              << " max_abs=" << audit.max_abs
              << " rel_l2=" << audit.rel_l2
              << " cos=" << audit.cosine_similarity << "\n";
    if (!audit.passed) {
        throw std::runtime_error("Differential check failed for " + label);
    }
}

inline HostTensor rand_f32(const std::vector<int64_t>& sh, std::mt19937_64& rng, float scale = 0.25f) {
    HostTensor ht = HostTensor::create(sh, DType::F32);
    std::normal_distribution<float> d(0.0f, scale);
    for (int64_t i = 0; i < ht.desc.numel(); ++i) ht.data<float>()[i] = d(rng);
    return ht;
}

inline HostTensor make_ones_f32(const std::vector<int64_t>& sh) {
    HostTensor ht = HostTensor::create(sh, DType::F32);
    for (int64_t i = 0; i < ht.desc.numel(); ++i) ht.data<float>()[i] = 1.0f;
    return ht;
}

}  // namespace fusionrt::test
