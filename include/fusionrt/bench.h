#pragma once

#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#endif
#include "fusionrt/json.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fusionrt::bench {

struct TimingStats {
    std::string unit = "ms";
    int samples_count = 0;
    double median = 0.0;
    double iqr = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
    double ci_level = 0.95;
    double ci_low = 0.0;
    double ci_high = 0.0;
    int bootstrap_resamples = 1000;
    // Optional CUDA Graph split
    bool has_graph_split = false;
    double host_launch_overhead_us = 0.0;
    double pure_kernel_us = 0.0;
    double end_to_end_us = 0.0;
    double speedup_factor = 1.0;
};

struct NumericalAudit {
    std::string reference_dtype = "fp64";
    std::string target_dtype = "fp32";
    int64_t num_elements = 0;
    bool passed = true;
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    double cosine_similarity = 1.0;
    double sqnr_db = 120.0;
    double ulp_p50 = 0.0;
    double ulp_p99 = 0.0;
    double ulp_max = 0.0;
    std::vector<int64_t> max_error_coords;
};

struct BenchmarkEntry {
    std::string id;
    std::string suite;
    std::string name;
    std::string backend;  // cuda, vulkan, cpu
    int device_index = 0;
    std::string dtype = "fp32";
    std::vector<int64_t> shape;
    TimingStats timing;
    bool has_numerical_audit = false;
    NumericalAudit audit;
    double achieved_tflops = 0.0;
    double achieved_bandwidth_gbps = 0.0;
    double theoretical_ceiling_pct = 0.0;
    double arithmetic_intensity = 0.0;
    std::string bottleneck = "bandwidth_bound";
    double decode_tok_per_s = 0.0;
    double speedup_vs_baseline = 0.0;
    std::string baseline_id;
    std::string status = "pass";
    std::string notes;
};

TimingStats compute_timing_stats(std::vector<double> samples_ms, uint64_t seed = 1337);

NumericalAudit audit_arrays_f32(const float* target, const double* ref_f64,
                                const std::vector<int64_t>& shape,
                                const std::string& target_dtype = "fp32",
                                double max_abs_tol = 1e-3, double rel_l2_tol = 1e-3);

// Interleaved A-B-A-B timing with L2 cache flush on CUDA
struct AbTimingResult {
    TimingStats a_stats;
    TimingStats b_stats;
};

#ifdef FUSIONRT_ENABLE_CUDA
AbTimingResult measure_cuda_ab_interleaved(
    CudaContext& ctx,
    const std::function<void(cudaStream_t)>& fn_a,
    const std::function<void(cudaStream_t)>& fn_b,
    int warmup_iters, int measured_iters, bool flush_l2 = true);
#endif

// Run full benchmark suite and write bench_report.v1.json
int run_benchmark_suite(const std::string& suite, const std::string& output_json_path, int vk_device_idx = 0);

}  // namespace fusionrt::bench
