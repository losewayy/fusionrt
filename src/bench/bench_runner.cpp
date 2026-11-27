#include "fusionrt/bench.h"
#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_backend.h"
#include "fusionrt/cuda_kernels.h"
#endif
#include "fusionrt/models.h"
#ifdef FUSIONRT_ENABLE_VULKAN
#include "fusionrt/vulkan_backend.h"
#include "fusionrt/vulkan_kernels.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef FUSIONRT_HAS_NVML
#include <nvml.h>
#endif

namespace fusionrt::bench {

namespace {

std::string utc_now_iso() {
    std::time_t t = std::time(nullptr);
    std::tm gmt{};
#ifdef _WIN32
    gmtime_s(&gmt, &t);
#else
    gmtime_r(&t, &gmt);
#endif
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gmt);
    return std::string(buf);
}

double percentile_sorted(const std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    double idx = p * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct NvmlSample {
    bool available = false;
    double sm_clock_mhz = 0.0;
    double mem_clock_mhz = 0.0;
    double temp_c = 0.0;
    double power_w = 0.0;
};

NvmlSample read_nvml(int dev_idx) {
    NvmlSample s;
#ifdef FUSIONRT_HAS_NVML
    if (nvmlInit_v2() == NVML_SUCCESS) {
        nvmlDevice_t dev{};
        if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(dev_idx), &dev) == NVML_SUCCESS) {
            unsigned int sm_c = 0, mem_c = 0, temp = 0, pwr = 0;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_c) == NVML_SUCCESS) s.sm_clock_mhz = sm_c;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &mem_c) == NVML_SUCCESS) s.mem_clock_mhz = mem_c;
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) s.temp_c = temp;
            if (nvmlDeviceGetPowerUsage(dev, &pwr) == NVML_SUCCESS) s.power_w = pwr / 1000.0;
            s.available = true;
        }
        nvmlShutdown();
    }
#else
    (void)dev_idx;
#endif
    return s;
}

JsonValue timing_to_json(const TimingStats& ts, const NvmlSample& nv) {
    JsonObject obj;
    obj["unit"] = JsonValue(ts.unit);
    obj["samples_count"] = JsonValue(ts.samples_count);
    obj["median"] = JsonValue(ts.median);
    obj["iqr"] = JsonValue(ts.iqr);
    obj["p95"] = JsonValue(ts.p95);
    obj["p99"] = JsonValue(ts.p99);
    obj["mean"] = JsonValue(ts.mean);
    obj["min"] = JsonValue(ts.min);
    obj["max"] = JsonValue(ts.max);
    obj["std"] = JsonValue(ts.stddev);

    JsonObject ci;
    ci["level"] = JsonValue(ts.ci_level);
    ci["low"] = JsonValue(ts.ci_low);
    ci["high"] = JsonValue(ts.ci_high);
    ci["resamples"] = JsonValue(ts.bootstrap_resamples);
    obj["bootstrap_ci"] = JsonValue(std::move(ci));

    if (ts.has_graph_split) {
        JsonObject gs;
        gs["host_launch_overhead_us"] = JsonValue(ts.host_launch_overhead_us);
        gs["pure_kernel_us"] = JsonValue(ts.pure_kernel_us);
        gs["end_to_end_us"] = JsonValue(ts.end_to_end_us);
        gs["speedup_factor"] = JsonValue(ts.speedup_factor);
        obj["graph_split"] = JsonValue(std::move(gs));
    }

    JsonArray samples;
    if (nv.available) {
        JsonObject cs;
        cs["timestamp_ms"] = JsonValue(0.0);
        cs["sm_clock_mhz"] = JsonValue(nv.sm_clock_mhz);
        cs["mem_clock_mhz"] = JsonValue(nv.mem_clock_mhz);
        cs["temp_c"] = JsonValue(nv.temp_c);
        cs["power_w"] = JsonValue(nv.power_w);
        samples.push_back(JsonValue(std::move(cs)));
    }
    obj["clock_temp_samples"] = JsonValue(std::move(samples));
    return JsonValue(std::move(obj));
}

JsonValue entry_to_json(const BenchmarkEntry& e, const NvmlSample& nv) {
    JsonObject obj;
    obj["id"] = JsonValue(e.id);
    obj["suite"] = JsonValue(e.suite);
    obj["name"] = JsonValue(e.name);
    obj["backend"] = JsonValue(e.backend);
    obj["device_index"] = JsonValue(e.device_index);
    obj["dtype"] = JsonValue(e.dtype);
    JsonArray sh;
    for (int64_t d : e.shape) sh.emplace_back(d);
    obj["shape"] = JsonValue(std::move(sh));
    obj["timing"] = timing_to_json(e.timing, nv);

    if (e.has_numerical_audit) {
        JsonObject na;
        na["reference_dtype"] = JsonValue(e.audit.reference_dtype);
        na["target_dtype"] = JsonValue(e.audit.target_dtype);
        na["num_elements"] = JsonValue(e.audit.num_elements);
        na["passed"] = JsonValue(e.audit.passed);
        na["max_abs"] = JsonValue(e.audit.max_abs);
        na["rel_l2"] = JsonValue(e.audit.rel_l2);
        na["cosine_similarity"] = JsonValue(e.audit.cosine_similarity);
        na["sqnr_db"] = JsonValue(e.audit.sqnr_db);
        JsonObject ulp;
        ulp["p50"] = JsonValue(e.audit.ulp_p50);
        ulp["p99"] = JsonValue(e.audit.ulp_p99);
        ulp["max"] = JsonValue(e.audit.ulp_max);
        na["ulp"] = JsonValue(std::move(ulp));
        JsonArray coords;
        for (int64_t c : e.audit.max_error_coords) coords.emplace_back(c);
        na["max_error_coords"] = JsonValue(std::move(coords));
        obj["numerical_audit"] = JsonValue(std::move(na));
    }

    JsonObject metrics;
    if (e.achieved_tflops > 0.0) metrics["achieved_tflops"] = JsonValue(e.achieved_tflops);
    if (e.achieved_bandwidth_gbps > 0.0) metrics["achieved_bandwidth_gbps"] = JsonValue(e.achieved_bandwidth_gbps);
    if (e.theoretical_ceiling_pct > 0.0) metrics["theoretical_ceiling_pct"] = JsonValue(e.theoretical_ceiling_pct);
    if (e.arithmetic_intensity > 0.0) metrics["arithmetic_intensity_flops_per_byte"] = JsonValue(e.arithmetic_intensity);
    metrics["bottleneck"] = JsonValue(e.bottleneck);
    if (e.decode_tok_per_s > 0.0) metrics["decode_tok_per_s"] = JsonValue(e.decode_tok_per_s);
    if (e.speedup_vs_baseline > 0.0) metrics["speedup_vs_baseline"] = JsonValue(e.speedup_vs_baseline);
    if (!e.baseline_id.empty()) metrics["baseline_id"] = JsonValue(e.baseline_id);
    obj["metrics"] = JsonValue(std::move(metrics));

    obj["status"] = JsonValue(e.status);
    if (!e.notes.empty()) obj["notes"] = JsonValue(e.notes);
    return JsonValue(std::move(obj));
}

}  // namespace

TimingStats compute_timing_stats(std::vector<double> samples_ms, uint64_t seed) {
    TimingStats ts;
    if (samples_ms.empty()) return ts;
    ts.unit = "ms";
    ts.samples_count = static_cast<int>(samples_ms.size());

    double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    ts.mean = sum / static_cast<double>(samples_ms.size());
    double sq_sum = 0.0;
    for (double x : samples_ms) sq_sum += (x - ts.mean) * (x - ts.mean);
    ts.stddev = std::sqrt(sq_sum / static_cast<double>(samples_ms.size()));

    std::sort(samples_ms.begin(), samples_ms.end());
    ts.min = samples_ms.front();
    ts.max = samples_ms.back();
    ts.median = percentile_sorted(samples_ms, 0.50);
    double q25 = percentile_sorted(samples_ms, 0.25);
    double q75 = percentile_sorted(samples_ms, 0.75);
    ts.iqr = q75 - q25;
    ts.p95 = percentile_sorted(samples_ms, 0.95);
    ts.p99 = percentile_sorted(samples_ms, 0.99);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, samples_ms.size() - 1);
    std::vector<double> boot_medians(static_cast<size_t>(ts.bootstrap_resamples));
    std::vector<double> resamp(samples_ms.size());
    for (int r = 0; r < ts.bootstrap_resamples; ++r) {
        for (size_t i = 0; i < samples_ms.size(); ++i) resamp[i] = samples_ms[dist(rng)];
        std::sort(resamp.begin(), resamp.end());
        boot_medians[static_cast<size_t>(r)] = percentile_sorted(resamp, 0.50);
    }
    std::sort(boot_medians.begin(), boot_medians.end());
    ts.ci_low = percentile_sorted(boot_medians, 0.025);
    ts.ci_high = percentile_sorted(boot_medians, 0.975);
    return ts;
}

NumericalAudit audit_arrays_f32(const float* target, const double* ref_f64,
                                const std::vector<int64_t>& shape,
                                const std::string& target_dtype,
                                double max_abs_tol, double rel_l2_tol) {
    NumericalAudit na;
    na.reference_dtype = "fp64";
    na.target_dtype = target_dtype;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    na.num_elements = n;

    double diff_sq = 0.0, ref_sq = 0.0, tgt_sq = 0.0, dot = 0.0;
    int64_t max_idx = 0;
    std::vector<double> ulps(static_cast<size_t>(n), 0.0);

    for (int64_t i = 0; i < n; ++i) {
        double r = ref_f64[i];
        double t = static_cast<double>(target[i]);
        double d = std::fabs(t - r);
        if (d > na.max_abs) {
            na.max_abs = d;
            max_idx = i;
        }
        diff_sq += d * d;
        ref_sq += r * r;
        tgt_sq += t * t;
        dot += r * t;

        float rf = static_cast<float>(r);
        int32_t ri = 0, ti = 0;
        std::memcpy(&ri, &rf, 4);
        std::memcpy(&ti, &target[i], 4);
        if (ri < 0) ri = static_cast<int32_t>(0x80000000u) - ri;
        if (ti < 0) ti = static_cast<int32_t>(0x80000000u) - ti;
        ulps[static_cast<size_t>(i)] = static_cast<double>(std::llabs(static_cast<int64_t>(ti) - static_cast<int64_t>(ri)));
    }

    na.rel_l2 = (ref_sq > 0.0) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
    na.cosine_similarity = (ref_sq > 0.0 && tgt_sq > 0.0) ? (dot / (std::sqrt(ref_sq) * std::sqrt(tgt_sq))) : 1.0;
    na.sqnr_db = (diff_sq > 0.0 && ref_sq > 0.0) ? (10.0 * std::log10(ref_sq / diff_sq)) : 140.0;
    std::sort(ulps.begin(), ulps.end());
    na.ulp_p50 = percentile_sorted(ulps, 0.50);
    na.ulp_p99 = percentile_sorted(ulps, 0.99);
    na.ulp_max = ulps.empty() ? 0.0 : ulps.back();

    na.max_error_coords.resize(shape.size());
    int64_t rem = max_idx;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        na.max_error_coords[static_cast<size_t>(d)] = rem % shape[static_cast<size_t>(d)];
        rem /= shape[static_cast<size_t>(d)];
    }
    na.passed = (na.max_abs <= max_abs_tol) || (na.rel_l2 <= rel_l2_tol);
    return na;
}

#ifdef FUSIONRT_ENABLE_CUDA
AbTimingResult measure_cuda_ab_interleaved(
    CudaContext& ctx,
    const std::function<void(cudaStream_t)>& fn_a,
    const std::function<void(cudaStream_t)>& fn_b,
    int warmup_iters, int measured_iters, bool flush_l2)
{
    cudaEvent_t ev0{}, ev1{};
    FUSIONRT_CUDA_CHECK(cudaEventCreate(&ev0));
    FUSIONRT_CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < warmup_iters; ++i) {
        fn_a(ctx.stream());
        fn_b(ctx.stream());
    }
    ctx.synchronize();

    std::vector<double> a_ms, b_ms;
    a_ms.reserve(static_cast<size_t>(measured_iters));
    b_ms.reserve(static_cast<size_t>(measured_iters));

    for (int i = 0; i < measured_iters; ++i) {
        // Alternate A-B and B-A order across even/odd iterations
        bool a_first = (i % 2 == 0);
        for (int pass = 0; pass < 2; ++pass) {
            bool run_a = (a_first ? (pass == 0) : (pass == 1));
            if (flush_l2) ctx.flush_l2();
            FUSIONRT_CUDA_CHECK(cudaEventRecord(ev0, ctx.stream()));
            if (run_a) fn_a(ctx.stream());
            else fn_b(ctx.stream());
            FUSIONRT_CUDA_CHECK(cudaEventRecord(ev1, ctx.stream()));
            FUSIONRT_CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            FUSIONRT_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (run_a) a_ms.push_back(static_cast<double>(ms));
            else b_ms.push_back(static_cast<double>(ms));
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return {compute_timing_stats(a_ms), compute_timing_stats(b_ms)};
}
#endif

bool write_benchmark_report(std::ostream& output, const JsonObject& report) {
    output << JsonValue(report).dump(2) << '\n';
    return static_cast<bool>(output.flush());
}

int run_benchmark_suite(const std::string& suite, const std::string& output_json_path, std::optional<int> vk_device) {
    auto write = [&](const JsonObject& report) {
        std::ofstream output(output_json_path, std::ios::binary);
        if (!output) {
            std::cerr << "Failed to open output path: " << output_json_path << '\n';
            return false;
        }
        const bool flushed = write_benchmark_report(output, report);
        output.close();
        if (flushed && output) return true;
        std::cerr << "Failed to write benchmark report: " << output_json_path << '\n';
        return false;
    };
    bool supported = false;
#ifdef FUSIONRT_ENABLE_CUDA
    supported = suite == "all" || suite == "fusions" || suite == "blocks";
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
    supported = supported || suite == "all" || suite == "vulkan";
#endif
    std::string reason;
    if (!supported) reason = "Unknown or unavailable benchmark suite: " + suite;
    if (vk_device && (suite == "all" || suite == "vulkan")) {
        size_t devices = 0;
#ifdef FUSIONRT_ENABLE_VULKAN
        devices = VkContext::enumerate_devices().size();
#endif
        if (*vk_device < 0 || static_cast<size_t>(*vk_device) >= devices)
            reason = "Unavailable Vulkan device " + std::to_string(*vk_device) + " (" + std::to_string(devices) + " available)";
    }
    if (!reason.empty()) {
        std::cerr << reason << '\n';
        JsonObject report{{"schema_version", "bench_report.v1"}, {"generated_at", utc_now_iso()}, {"error", reason},
                          {"benchmarks", JsonArray{}},
                          {"summary", JsonObject{{"total_cases", 0}, {"passed", 0}, {"failed", 0},
                                                   {"skipped", 0}, {"overall_status", "fail"}}}};
        write(report);
        return 1;
    }
#if !defined(FUSIONRT_ENABLE_CUDA) && !defined(FUSIONRT_ENABLE_VULKAN)
    return 1;
#else
    size_t l2_bytes = 32ULL * 1024ULL * 1024ULL;
#ifdef FUSIONRT_ENABLE_CUDA
    std::unique_ptr<CudaContext> cuda;
    if (suite != "vulkan") {
        cuda = std::make_unique<CudaContext>(0, 512, 256);
        l2_bytes = cuda->info().l2_cache_bytes;
    }
#endif
    NvmlSample nv_start = read_nvml(0);
    std::vector<BenchmarkEntry> entries;
    std::mt19937_64 rng(2026);

    auto make_host_rand = [&](const std::vector<int64_t>& sh) {
        HostTensor ht = HostTensor::create(sh, DType::F32);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (int64_t i = 0; i < ht.desc.numel(); ++i) ht.data<float>()[i] = dist(rng);
        return ht;
    };

    auto run_all = (suite == "all");

#ifdef FUSIONRT_ENABLE_CUDA

    // -------------------------------------------------------------------------
    // 1. CUDA Fusions Suite
    // -------------------------------------------------------------------------
    if (run_all || suite == "fusions") {
        auto& cuda_ctx = *cuda;
        // 1a. Fused Pre-Norm + Linear (K=4096, N=6144, M=1, F32)
        {
            int M = 1, K = 4096, N = 6144;
            Tensor x = cuda_ctx.upload(make_host_rand({M, K}), false);
            Tensor nw = cuda_ctx.upload(make_host_rand({K}), false);
            Tensor w = cuda_ctx.upload(make_host_rand({N, K}), false);
            Tensor x_norm = cuda_ctx.scratch_arena().alloc_tensor({M, K}, DType::F32);
            Tensor y_unf = cuda_ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
            Tensor y_fus = cuda_ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);

            auto ab = measure_cuda_ab_interleaved(
                cuda_ctx,
                [&](cudaStream_t s) {
                    cuda_ops::rmsnorm_f32(x.ptr<float>(), nw.ptr<float>(), x_norm.ptr<float>(), M, K, 1e-6f, s);
                    cuda_ops::linear_f32(x_norm.ptr<float>(), w.ptr<float>(), y_unf.ptr<float>(), M, N, K, s);
                },
                [&](cudaStream_t s) {
                    cuda_ops::fused_rmsnorm_linear_f32(x.ptr<float>(), nw.ptr<float>(), w.ptr<float>(), y_fus.ptr<float>(), M, N, K, 1e-6f, s);
                },
                5, 30, true);

            double bytes = static_cast<double>(N) * K * 4.0;
            BenchmarkEntry e_unf;
            e_unf.id = "cuda_rmsnorm_linear_unfused_f32";
            e_unf.suite = "fusions";
            e_unf.name = "Unfused RMSNorm + GEMV (F32, 1x6144x4096)";
            e_unf.backend = "cuda";
            e_unf.dtype = "fp32";
            e_unf.shape = {M, N, K};
            e_unf.timing = ab.a_stats;
            e_unf.achieved_bandwidth_gbps = (bytes / (ab.a_stats.median * 1e-3)) / 1e9;
            e_unf.theoretical_ceiling_pct = 100.0 * e_unf.achieved_bandwidth_gbps / cuda_ctx.info().peak_bandwidth_gbps;
            entries.push_back(e_unf);

            BenchmarkEntry e_fus = e_unf;
            e_fus.id = "cuda_rmsnorm_linear_fused_f32";
            e_fus.name = "Fused Pre-Norm + GEMV (F32, 1x6144x4096)";
            e_fus.timing = ab.b_stats;
            e_fus.achieved_bandwidth_gbps = (bytes / (ab.b_stats.median * 1e-3)) / 1e9;
            e_fus.theoretical_ceiling_pct = 100.0 * e_fus.achieved_bandwidth_gbps / cuda_ctx.info().peak_bandwidth_gbps;
            e_fus.speedup_vs_baseline = ab.a_stats.median / ab.b_stats.median;
            e_fus.baseline_id = e_unf.id;
            entries.push_back(e_fus);
            cuda_ctx.scratch_arena().reset();
        }

        // 1b. Fused W4 (Q4_0) Dequantize-GEMV (K=4096, N=4096, M=1)
        {
            int M = 1, K = 4096, N = 4096;
            HostTensor hx = make_host_rand({M, K});
            HostTensor hw_f32 = make_host_rand({N, K});
            HostTensor hw_q4 = HostTensor::create({N, K}, DType::Q4_0);
            for (int r = 0; r < N; ++r) {
                quantize_row_q4_0(hw_f32.data<float>() + r * K,
                                  reinterpret_cast<BlockQ4_0*>(hw_q4.storage.data()) + r * (K / 32), K);
            }
            Tensor x = cuda_ctx.upload(hx, false);
            Tensor w_q4 = cuda_ctx.upload(hw_q4, false);
            Tensor w_scratch = cuda_ctx.scratch_arena().alloc_tensor({N, K}, DType::F32);
            Tensor y_unf = cuda_ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
            Tensor y_fus = cuda_ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);

            auto ab = measure_cuda_ab_interleaved(
                cuda_ctx,
                [&](cudaStream_t s) {
                    cuda_ops::linear_q4_0_unfused(x.ptr<float>(), w_q4.ptr(), y_unf.ptr<float>(), M, N, K, w_scratch.ptr(), s);
                },
                [&](cudaStream_t s) {
                    cuda_ops::fused_gemv_q4_0(x.ptr<float>(), w_q4.ptr(), y_fus.ptr<float>(), M, N, K, s);
                },
                5, 30, true);

            double q4_bytes = static_cast<double>(w_q4.nbytes());
            BenchmarkEntry e_unf;
            e_unf.id = "cuda_gemv_q4_unfused";
            e_unf.suite = "fusions";
            e_unf.name = "Unfused Global Dequant + GEMV (Q4_0, 1x4096x4096)";
            e_unf.backend = "cuda";
            e_unf.dtype = "q4_0";
            e_unf.shape = {M, N, K};
            e_unf.timing = ab.a_stats;
            e_unf.achieved_bandwidth_gbps = (q4_bytes / (ab.a_stats.median * 1e-3)) / 1e9;
            entries.push_back(e_unf);

            BenchmarkEntry e_fus = e_unf;
            e_fus.id = "cuda_gemv_q4_fused";
            e_fus.name = "Fused On-the-Fly Dequant-GEMV (Q4_0, 1x4096x4096)";
            e_fus.timing = ab.b_stats;
            e_fus.achieved_bandwidth_gbps = (q4_bytes / (ab.b_stats.median * 1e-3)) / 1e9;
            e_fus.theoretical_ceiling_pct = 100.0 * e_fus.achieved_bandwidth_gbps / cuda_ctx.info().peak_bandwidth_gbps;
            e_fus.speedup_vs_baseline = ab.a_stats.median / ab.b_stats.median;
            e_fus.baseline_id = e_unf.id;
            entries.push_back(e_fus);
            cuda_ctx.scratch_arena().reset();
        }

        // 1c. Fused Single-Kernel Register-Resident GatedDeltaNet Step vs Unfused 5-Kernel
        {
            int nk = 8, nv = 16, dk = 64, dv = 64;
            int conv_dim = 2 * nk * dk + nv * dv;
            int v_dim = nv * dv;
            Tensor qkv = cuda_ctx.upload(make_host_rand({conv_dim}), false);
            Tensor z = cuda_ctx.upload(make_host_rand({v_dim}), false);
            Tensor a = cuda_ctx.upload(make_host_rand({nv}), false);
            Tensor b = cuda_ctx.upload(make_host_rand({nv}), false);
            Tensor cw = cuda_ctx.upload(make_host_rand({conv_dim, 4}), false);
            Tensor cb = cuda_ctx.upload(make_host_rand({conv_dim}), false);
            Tensor alog = cuda_ctx.upload(make_host_rand({nv}), false);
            Tensor dtb = cuda_ctx.upload(make_host_rand({nv}), false);
            Tensor nw = cuda_ctx.upload(make_host_rand({dv}), false);
            Tensor cs_u = cuda_ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
            Tensor ss_u = cuda_ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
            Tensor cs_f = cuda_ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
            Tensor ss_f = cuda_ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
            Tensor out_u = cuda_ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
            Tensor out_f = cuda_ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
            Tensor sqkv = cuda_ctx.scratch_arena().alloc_tensor({conv_dim}, DType::F32);
            Tensor sy = cuda_ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);

            auto ab = measure_cuda_ab_interleaved(
                cuda_ctx,
                [&](cudaStream_t s) {
                    cuda_ops::unfused_gated_deltanet_step_f32(
                        qkv.ptr<float>(), z.ptr<float>(), a.ptr<float>(), b.ptr<float>(),
                        cw.ptr<float>(), cb.ptr<float>(), alog.ptr<float>(), dtb.ptr<float>(),
                        nw.ptr<float>(), cs_u.ptr<float>(), ss_u.ptr<float>(), out_u.ptr<float>(),
                        sqkv.ptr<float>(), sy.ptr<float>(), nk, nv, dk, dv, 1e-6f, 0.125f, s);
                },
                [&](cudaStream_t s) {
                    cuda_ops::fused_gated_deltanet_step_f32(
                        qkv.ptr<float>(), z.ptr<float>(), a.ptr<float>(), b.ptr<float>(),
                        cw.ptr<float>(), cb.ptr<float>(), alog.ptr<float>(), dtb.ptr<float>(),
                        nw.ptr<float>(), cs_f.ptr<float>(), ss_f.ptr<float>(), out_f.ptr<float>(),
                        nk, nv, dk, dv, 1e-6f, 0.125f, s);
                },
                5, 40, true);

            double state_rw_bytes = 2.0 * static_cast<double>(nv) * dk * dv * 4.0;
            BenchmarkEntry e_unf;
            e_unf.id = "cuda_gdn_step_unfused";
            e_unf.suite = "fusions";
            e_unf.name = "Unfused GatedDeltaNet Step (5 kernels, nv=16, dk=dv=64)";
            e_unf.backend = "cuda";
            e_unf.dtype = "fp32";
            e_unf.shape = {nv, dk, dv};
            e_unf.timing = ab.a_stats;
            e_unf.achieved_bandwidth_gbps = (state_rw_bytes / (ab.a_stats.median * 1e-3)) / 1e9;
            entries.push_back(e_unf);

            BenchmarkEntry e_fus = e_unf;
            e_fus.id = "cuda_gdn_step_fused";
            e_fus.name = "Fused Register-Resident GatedDeltaNet Step (1 kernel)";
            e_fus.timing = ab.b_stats;
            e_fus.achieved_bandwidth_gbps = (state_rw_bytes / (ab.b_stats.median * 1e-3)) / 1e9;
            e_fus.speedup_vs_baseline = ab.a_stats.median / ab.b_stats.median;
            e_fus.baseline_id = e_unf.id;
            entries.push_back(e_fus);
            cuda_ctx.scratch_arena().reset();
        }

        // 1d. Fused Q/K RMSNorm + RoPE + KV Cache Append vs Unfused 5 launches
        {
            int nq = 32, nkv = 8, hd = 128, max_seq = 2048;
            Tensor q = cuda_ctx.upload(make_host_rand({nq, hd}), false);
            Tensor k = cuda_ctx.upload(make_host_rand({nkv, hd}), false);
            Tensor v = cuda_ctx.upload(make_host_rand({nkv, hd}), false);
            Tensor qn = cuda_ctx.upload(make_host_rand({hd}), false);
            Tensor kn = cuda_ctx.upload(make_host_rand({hd}), false);
            Tensor kc = cuda_ctx.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
            Tensor vc = cuda_ctx.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);

            auto ab = measure_cuda_ab_interleaved(
                cuda_ctx,
                [&](cudaStream_t s) {
                    cuda_ops::rmsnorm_f32(q.ptr<float>(), qn.ptr<float>(), q.ptr<float>(), nq, hd, 1e-6f, s);
                    cuda_ops::rmsnorm_f32(k.ptr<float>(), kn.ptr<float>(), k.ptr<float>(), nkv, hd, 1e-6f, s);
                    cuda_ops::rope_f32(q.ptr<float>(), nq, hd, hd, 123, 1000000.0f, true, s);
                    cuda_ops::rope_f32(k.ptr<float>(), nkv, hd, hd, 123, 1000000.0f, true, s);
                    cuda_ops::kv_append_f32(k.ptr<float>(), v.ptr<float>(), kc.ptr<float>(), vc.ptr<float>(), nkv, hd, 123, max_seq, s);
                },
                [&](cudaStream_t s) {
                    cuda_ops::fused_qk_norm_rope_kv_append_f32(
                        q.ptr<float>(), k.ptr<float>(), v.ptr<float>(),
                        qn.ptr<float>(), kn.ptr<float>(), kc.ptr<float>(), vc.ptr<float>(),
                        nq, nkv, hd, hd, nullptr, 123, max_seq, 1000000.0f, 1e-6f, true, s);
                },
                5, 40, true);

            BenchmarkEntry e_unf;
            e_unf.id = "cuda_qk_norm_rope_kv_unfused";
            e_unf.suite = "fusions";
            e_unf.name = "Unfused Q/K Norm + RoPE + KV Append (5 launches)";
            e_unf.backend = "cuda";
            e_unf.dtype = "fp32";
            e_unf.shape = {nq, nkv, hd};
            e_unf.timing = ab.a_stats;
            e_unf.bottleneck = "launch_bound";
            entries.push_back(e_unf);

            BenchmarkEntry e_fus = e_unf;
            e_fus.id = "cuda_qk_norm_rope_kv_fused";
            e_fus.name = "Fused Q/K Norm + RoPE + KV Append (1 launch)";
            e_fus.timing = ab.b_stats;
            e_fus.speedup_vs_baseline = ab.a_stats.median / ab.b_stats.median;
            e_fus.baseline_id = e_unf.id;
            entries.push_back(e_fus);
            cuda_ctx.scratch_arena().reset();
        }

        // 1e. GQA Split-K Decode at 64K seqlen (max_splits=1 vs max_splits=16)
        {
            int nq = 16, nkv = 4, hd = 64, kv_len = 65536;
            Tensor q = cuda_ctx.upload(make_host_rand({nq, hd}), false);
            Tensor kc = cuda_ctx.scratch_arena().alloc_tensor({kv_len, nkv, hd}, DType::F32);
            Tensor vc = cuda_ctx.scratch_arena().alloc_tensor({kv_len, nkv, hd}, DType::F32);
            Tensor out1 = cuda_ctx.scratch_arena().alloc_tensor({nq, hd}, DType::F32);
            Tensor out2 = cuda_ctx.scratch_arena().alloc_tensor({nq, hd}, DType::F32);
            Tensor part_out = cuda_ctx.scratch_arena().alloc_tensor({nq, 16, hd}, DType::F32);
            Tensor part_ml = cuda_ctx.scratch_arena().alloc_tensor({nq, 16, 2}, DType::F32);

            auto ab = measure_cuda_ab_interleaved(
                cuda_ctx,
                [&](cudaStream_t s) {
                    cuda_ops::gqa_splitk_decode_f32(q.ptr<float>(), kc.ptr<float>(), vc.ptr<float>(),
                                                    out1.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                                                    nq, nkv, hd, nullptr, kv_len, kv_len, 0.125f, 1, s);
                },
                [&](cudaStream_t s) {
                    cuda_ops::gqa_splitk_decode_f32(q.ptr<float>(), kc.ptr<float>(), vc.ptr<float>(),
                                                    out2.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                                                    nq, nkv, hd, nullptr, kv_len, kv_len, 0.125f, 16, s);
                },
                3, 20, true);

            double kv_bytes = 2.0 * static_cast<double>(kv_len) * nkv * hd * 4.0;
            BenchmarkEntry e1;
            e1.id = "cuda_gqa_64k_single_split";
            e1.suite = "fusions";
            e1.name = "GQA Decode 64K SeqLen (single-CTA/head baseline)";
            e1.backend = "cuda";
            e1.dtype = "fp32";
            e1.shape = {nq, nkv, kv_len, hd};
            e1.timing = ab.a_stats;
            e1.achieved_bandwidth_gbps = (kv_bytes / (ab.a_stats.median * 1e-3)) / 1e9;
            entries.push_back(e1);

            BenchmarkEntry e2 = e1;
            e2.id = "cuda_gqa_64k_splitk16";
            e2.name = "GQA Flash/Split-K Decode 64K SeqLen (16 splits)";
            e2.timing = ab.b_stats;
            e2.achieved_bandwidth_gbps = (kv_bytes / (ab.b_stats.median * 1e-3)) / 1e9;
            e2.theoretical_ceiling_pct = 100.0 * e2.achieved_bandwidth_gbps / cuda_ctx.info().peak_bandwidth_gbps;
            e2.speedup_vs_baseline = ab.a_stats.median / ab.b_stats.median;
            e2.baseline_id = e1.id;
            entries.push_back(e2);
            cuda_ctx.scratch_arena().reset();
        }
    }

    // -------------------------------------------------------------------------
    // 2. End-to-End Architecture Blocks Suite (Dense, Hybrid 3:1, Sparse MoE)
    // -------------------------------------------------------------------------
    if (run_all || suite == "blocks") {
        auto& cuda_ctx = *cuda;
        for (const std::string& arch : {"dense", "hybrid", "moe"}) {
            cuda_ctx.weight_arena().reset();
            cuda_ctx.scratch_arena().reset();

            HybridModelConfig cfg;
            cfg.arch = arch;
            cfg.vocab_size = 512;
            cfg.num_layers = 4;
            cfg.dense.hidden_size = 512;
            cfg.dense.max_seq_len = 2048;
            cfg.gdn.hidden_size = 512;
            cfg.moe.hidden_size = 512;
            cfg.moe.gpu_resident_experts = 8;

            FusionModel model(cfg);
            model.init_synthetic(cuda_ctx, 77);

            // Warmup & measure unfused vs fused vs CUDA Graph (using real cudaEvent + wall-clock split)
            auto measure_model = [&](bool use_fused, bool use_graph, double* pure_gpu_ms_out = nullptr) {
                model.reset_kv_and_state(cuda_ctx);
                for (int w = 0; w < 5; ++w) {
                    model.forward_token_cuda(cuda_ctx, 1 + w, w, use_fused, use_graph);
                }
                cudaEvent_t ev0{}, ev1{};
                FUSIONRT_CUDA_CHECK(cudaEventCreate(&ev0));
                FUSIONRT_CUDA_CHECK(cudaEventCreate(&ev1));
                std::vector<double> wall_ms;
                std::vector<double> gpu_ms;
                for (int i = 0; i < 20; ++i) {
                    cuda_ctx.flush_l2();
                    cuda_ctx.synchronize();
                    auto t0 = std::chrono::high_resolution_clock::now();
                    FUSIONRT_CUDA_CHECK(cudaEventRecord(ev0, cuda_ctx.stream()));
                    model.forward_token_cuda(cuda_ctx, 1 + (i % 10), 5 + i, use_fused, use_graph);
                    FUSIONRT_CUDA_CHECK(cudaEventRecord(ev1, cuda_ctx.stream()));
                    FUSIONRT_CUDA_CHECK(cudaEventSynchronize(ev1));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    float gms = 0.0f;
                    FUSIONRT_CUDA_CHECK(cudaEventElapsedTime(&gms, ev0, ev1));
                    gpu_ms.push_back(static_cast<double>(gms));
                }
                cudaEventDestroy(ev0);
                cudaEventDestroy(ev1);
                if (pure_gpu_ms_out) {
                    *pure_gpu_ms_out = compute_timing_stats(gpu_ms).median;
                }
                return compute_timing_stats(wall_ms);
            };

            if (arch == "moe") {
                double pure_eager_ms = 0.0, pure_graph_ms = 0.0;
                TimingStats t_eager = measure_model(true, false, &pure_eager_ms);
                TimingStats t_gr = measure_model(true, true, &pure_graph_ms);
                t_gr.has_graph_split = true;
                t_gr.end_to_end_us = t_gr.median * 1000.0;
                t_gr.pure_kernel_us = pure_graph_ms * 1000.0;
                t_gr.host_launch_overhead_us = std::max(0.0, (t_eager.median - pure_graph_ms) * 1000.0);
                t_gr.speedup_factor = t_eager.median / t_gr.median;

                BenchmarkEntry ee;
                ee.id = "cuda_block_moe_fused";
                ee.suite = "blocks";
                ee.name = "4-Layer moe Decode (Fused Eager)";
                ee.backend = "cuda";
                ee.dtype = "fp32";
                ee.shape = {4, 512};
                ee.timing = t_eager;
                ee.decode_tok_per_s = 1000.0 / t_eager.median;
                entries.push_back(ee);

                BenchmarkEntry eg = ee;
                eg.id = "cuda_block_moe_graph";
                eg.name = "4-Layer moe Decode (Fused + CUDA Graph)";
                eg.timing = t_gr;
                eg.decode_tok_per_s = 1000.0 / t_gr.median;
                eg.speedup_vs_baseline = t_eager.median / t_gr.median;
                eg.baseline_id = ee.id;
                entries.push_back(eg);
            } else {
                TimingStats t_unf = measure_model(false, false);
                TimingStats t_fus = measure_model(true, false);
                double pure_graph_ms = 0.0;
                TimingStats t_gr = measure_model(true, true, &pure_graph_ms);
                t_gr.has_graph_split = true;
                t_gr.end_to_end_us = t_gr.median * 1000.0;
                t_gr.pure_kernel_us = pure_graph_ms * 1000.0;
                t_gr.host_launch_overhead_us = std::max(0.0, (t_fus.median - pure_graph_ms) * 1000.0);
                t_gr.speedup_factor = t_fus.median / t_gr.median;

                BenchmarkEntry eu;
                eu.id = "cuda_block_" + arch + "_unfused";
                eu.suite = "blocks";
                eu.name = "4-Layer " + arch + " Decode (Unfused Eager)";
                eu.backend = "cuda";
                eu.dtype = "fp32";
                eu.shape = {4, 512};
                eu.timing = t_unf;
                eu.decode_tok_per_s = 1000.0 / t_unf.median;
                entries.push_back(eu);

                BenchmarkEntry ef = eu;
                ef.id = "cuda_block_" + arch + "_fused";
                ef.name = "4-Layer " + arch + " Decode (Fused Eager)";
                ef.timing = t_fus;
                ef.decode_tok_per_s = 1000.0 / t_fus.median;
                ef.speedup_vs_baseline = t_unf.median / t_fus.median;
                ef.baseline_id = eu.id;
                entries.push_back(ef);

                BenchmarkEntry eg = eu;
                eg.id = "cuda_block_" + arch + "_graph";
                eg.name = "4-Layer " + arch + " Decode (Fused + CUDA Graph)";
                eg.timing = t_gr;
                eg.decode_tok_per_s = 1000.0 / t_gr.median;
                eg.speedup_vs_baseline = t_unf.median / t_gr.median;
                eg.baseline_id = eu.id;
                entries.push_back(eg);

                // Batched Prefill (M=32 tokens)
                std::vector<int32_t> pf_toks(32, 1);
                model.reset_kv_and_state(cuda_ctx);
                model.forward_prefill_cuda(cuda_ctx, pf_toks, 0, true);
                std::vector<double> pf_ms;
                for (int i = 0; i < 10; ++i) {
                    model.reset_kv_and_state(cuda_ctx);
                    cuda_ctx.flush_l2();
                    cuda_ctx.synchronize();
                    auto t0 = std::chrono::high_resolution_clock::now();
                    model.forward_prefill_cuda(cuda_ctx, pf_toks, 0, true);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    pf_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
                TimingStats tpf = compute_timing_stats(pf_ms);
                BenchmarkEntry epf = eu;
                epf.id = "cuda_block_" + arch + "_prefill32";
                epf.name = "4-Layer " + arch + " Batched Prefill (M=32)";
                epf.shape = {4, 32, 512};
                epf.timing = tpf;
                epf.decode_tok_per_s = 32000.0 / tpf.median;
                epf.speedup_vs_baseline = 0.0;
                epf.baseline_id.clear();
                entries.push_back(epf);
            }
        }
    }
#endif  // FUSIONRT_ENABLE_CUDA

#ifdef FUSIONRT_ENABLE_VULKAN
    // -------------------------------------------------------------------------
    // 3. Vulkan Suite (Cooperative Matrix vs Tiled + Single-Command-Buffer Batch)
    // -------------------------------------------------------------------------
    if (run_all || suite == "vulkan") {
        const int vk_device_idx = vk_device.value_or(0);
        auto vk_devs = VkContext::enumerate_devices();
        if (vk_device_idx >= 0 && vk_device_idx < static_cast<int>(vk_devs.size())) {
            VkContext vk(static_cast<uint32_t>(vk_device_idx), 256, 256);

            // 3a. Single-Command-Buffer Batching vs Per-Op Submission for Dense GQA Layer
            {
                int H = 512, nq = 8, nkv = 2, hd = 64, inter = 1536, max_seq = 512;
                int q_dim = nq * hd, kv_dim = nkv * hd, qkv_dim = q_dim + 2 * kv_dim;
                Tensor x_in = vk.upload(make_host_rand({H}), true);
                Tensor anw = vk.upload(make_host_rand({H}), true);
                Tensor qkvw = vk.upload(make_host_rand({qkv_dim, H}), true);
                Tensor qnw = vk.upload(make_host_rand({hd}), true);
                Tensor knw = vk.upload(make_host_rand({hd}), true);
                Tensor ow = vk.upload(make_host_rand({H, q_dim}), true);
                Tensor mnw = vk.upload(make_host_rand({H}), true);
                Tensor guw = vk.upload(make_host_rand({2 * inter, H}), true);
                Tensor dw = vk.upload(make_host_rand({H, inter}), true);

                Tensor qkv_buf = vk.scratch_arena().alloc_tensor({qkv_dim}, DType::F32);
                Tensor q_buf = vk.scratch_arena().alloc_tensor({q_dim}, DType::F32);
                Tensor k_buf = vk.scratch_arena().alloc_tensor({kv_dim}, DType::F32);
                Tensor v_buf = vk.scratch_arena().alloc_tensor({kv_dim}, DType::F32);
                Tensor kc = vk.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
                Tensor vc = vk.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
                Tensor attn_out = vk.scratch_arena().alloc_tensor({q_dim}, DType::F32);
                Tensor o_buf = vk.scratch_arena().alloc_tensor({H}, DType::F32);
                Tensor gu_buf = vk.scratch_arena().alloc_tensor({2 * inter}, DType::F32);
                Tensor sw_buf = vk.scratch_arena().alloc_tensor({inter}, DType::F32);
                Tensor d_buf = vk.scratch_arena().alloc_tensor({H}, DType::F32);
                Tensor x_out = vk.scratch_arena().alloc_tensor({H}, DType::F32);

                std::vector<std::vector<uint8_t>> push_storage;
                auto ops = vk_ops::build_dense_layer_dispatches(
                    x_in, anw, qkvw, qnw, knw, ow, mnw, guw, dw,
                    qkv_buf, q_buf, k_buf, v_buf, kc, vc, attn_out, o_buf,
                    gu_buf, sw_buf, d_buf, x_out,
                    H, nq, nkv, hd, hd, inter, 0, 16, 1000000.0f, 1e-6f, true, true, push_storage);

                // Warmup
                vk.dispatch_batched(ops, true);
                std::vector<double> unb_ms, bat_ms;
                for (int i = 0; i < 15; ++i) {
                    vk.flush_l2();
                    auto t0 = std::chrono::high_resolution_clock::now();
                    vk.dispatch_unbatched(ops, false);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    unb_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

                    vk.flush_l2();
                    auto t2 = std::chrono::high_resolution_clock::now();
                    vk.dispatch_batched(ops, false);
                    auto t3 = std::chrono::high_resolution_clock::now();
                    bat_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
                }

                TimingStats tu = compute_timing_stats(unb_ms);
                TimingStats tb = compute_timing_stats(bat_ms);

                BenchmarkEntry eu;
                eu.id = "vulkan_dense_layer_unbatched";
                eu.suite = "vulkan";
                eu.name = "Vulkan Dense GQA Layer (Unbatched 9 Submissions)";
                eu.backend = "vulkan";
                eu.device_index = vk_device_idx;
                eu.dtype = "fp32";
                eu.shape = {H, inter};
                eu.timing = tu;
                eu.bottleneck = "launch_bound";
                entries.push_back(eu);

                BenchmarkEntry eb = eu;
                eb.id = "vulkan_dense_layer_batched";
                eb.name = "Vulkan Dense GQA Layer (Single-CmdBuf Batched)";
                eb.timing = tb;
                eb.speedup_vs_baseline = tu.median / tb.median;
                eb.baseline_id = eu.id;
                entries.push_back(eb);
                vk.scratch_arena().reset();
            }

            // 3b. Vulkan Cooperative Matrix FP16 GEMM vs Tiled FP32 GEMM (1024x1024x1024)
            {
                int M = 1024, N = 1024, K = 1024;
                HostTensor ha_f32 = make_host_rand({M, K});
                HostTensor hb_f32 = make_host_rand({N, K});
                HostTensor ha_f16 = HostTensor::create({M, K}, DType::F16);
                HostTensor hb_f16 = HostTensor::create({K, N}, DType::F16);
                convert_f32_to_f16(ha_f32.data<float>(), reinterpret_cast<uint16_t*>(ha_f16.storage.data()), M * K);
                convert_f32_to_f16(hb_f32.data<float>(), reinterpret_cast<uint16_t*>(hb_f16.storage.data()), K * N);

                Tensor a32 = vk.upload(ha_f32, false);
                Tensor b32 = vk.upload(hb_f32, false);
                Tensor c32 = vk.scratch_arena().alloc_tensor({M, N}, DType::F32);
                Tensor a16 = vk.upload(ha_f16, false);
                Tensor b16 = vk.upload(hb_f16, false);
                Tensor c_cm = vk.scratch_arena().alloc_tensor({M, N}, DType::F32);

                vk_ops::gemm_auto(vk, a32, b32, c32, M, N, K, false);
                std::vector<double> tiled_ms, cm_ms;
                for (int i = 0; i < 15; ++i) {
                    tiled_ms.push_back(vk_ops::gemm_auto(vk, a32, b32, c32, M, N, K, false));
                    if (vk.info().coop_mat.supported) {
                        cm_ms.push_back(vk_ops::gemm_auto(vk, a16, b16, c_cm, M, N, K, true));
                    }
                }
                double flops = 2.0 * M * N * K;
                TimingStats tt = compute_timing_stats(tiled_ms);
                BenchmarkEntry et;
                et.id = "vulkan_gemm_tiled_1024";
                et.suite = "vulkan";
                et.name = "Vulkan Tiled FP32 GEMM (1024x1024x1024)";
                et.backend = "vulkan";
                et.device_index = vk_device_idx;
                et.dtype = "fp32";
                et.shape = {M, N, K};
                et.timing = tt;
                et.achieved_tflops = (flops / (tt.median * 1e-3)) / 1e12;
                et.bottleneck = "compute_bound";
                entries.push_back(et);

                if (vk.info().coop_mat.supported) {
                    TimingStats tc = compute_timing_stats(cm_ms);
                    BenchmarkEntry ec = et;
                    ec.id = "vulkan_gemm_coopmat_1024";
                    ec.name = "Vulkan VK_KHR_cooperative_matrix FP16 GEMM (1024x1024x1024)";
                    ec.dtype = "fp16";
                    ec.timing = tc;
                    ec.achieved_tflops = (flops / (tc.median * 1e-3)) / 1e12;
                    ec.speedup_vs_baseline = tt.median / tc.median;
                    ec.baseline_id = et.id;
                    entries.push_back(ec);
                }
                vk.scratch_arena().reset();
            }
        }
    }
#endif  // FUSIONRT_ENABLE_VULKAN

    NvmlSample nv_end = read_nvml(0);

    // Build canonical bench_report.v1.json
    JsonObject root;
    root["schema_version"] = JsonValue("bench_report.v1");
    root["generated_at"] = JsonValue(utc_now_iso());

    JsonObject tool;
    tool["name"] = JsonValue("fusionrt");
    tool["version"] = JsonValue(FUSIONRT_VERSION);
    tool["command"] = JsonValue("fusionrt bench --suite " + suite);
    root["tool"] = JsonValue(std::move(tool));

    JsonObject host;
#ifdef _WIN32
    host["os"] = JsonValue("Windows");
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        host["ram_mb"] = JsonValue(static_cast<double>(ms.ullTotalPhys) / (1024.0 * 1024.0));
    }
#else
    host["os"] = JsonValue("Linux");
#endif
    host["arch"] = JsonValue("x86_64");
    host["cpu_cores"] = JsonValue(static_cast<int64_t>(std::thread::hardware_concurrency()));

    JsonArray devs;
#ifdef FUSIONRT_ENABLE_CUDA
    for (const auto& cd : CudaContext::enumerate_devices()) {
        JsonObject d;
        d["index"] = JsonValue(cd.device_id);
        d["name"] = JsonValue(cd.name);
        d["backend"] = JsonValue("cuda");
        d["vendor"] = JsonValue("NVIDIA");
        d["driver_version"] = JsonValue(cd.driver_version);
        d["compute_capability"] = JsonValue(std::to_string(cd.sm_major) + "." + std::to_string(cd.sm_minor));
        d["l2_cache_bytes"] = JsonValue(static_cast<int64_t>(cd.l2_cache_bytes));
        d["vram_total_mb"] = JsonValue(static_cast<double>(cd.total_vram_bytes) / (1024.0 * 1024.0));
        d["sm_count"] = JsonValue(cd.multi_processor_count);
        d["peak_bandwidth_gbps"] = JsonValue(cd.peak_bandwidth_gbps);
        devs.push_back(JsonValue(std::move(d)));
    }
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
    for (const auto& vd : VkContext::enumerate_devices()) {
        JsonObject d;
        d["index"] = JsonValue(static_cast<int64_t>(vd.device_index));
        d["name"] = JsonValue(vd.name);
        d["backend"] = JsonValue("vulkan");
        d["vram_total_mb"] = JsonValue(static_cast<double>(vd.device_local_vram_bytes) / (1024.0 * 1024.0));
        d["subgroup_size"] = JsonValue(static_cast<int64_t>(vd.subgroup_size));
        d["cooperative_matrix"] = JsonValue(vd.coop_mat.supported);
        devs.push_back(JsonValue(std::move(d)));
    }
#endif

    JsonObject hw;
    hw["host"] = JsonValue(std::move(host));
    hw["devices"] = JsonValue(std::move(devs));
    if (nv_start.available) {
        JsonObject nvo;
        nvo["available"] = JsonValue(true);
        nvo["temp_c_start"] = JsonValue(nv_start.temp_c);
        nvo["temp_c_end"] = JsonValue(nv_end.temp_c);
        nvo["sm_clock_mhz_start"] = JsonValue(nv_start.sm_clock_mhz);
        nvo["sm_clock_mhz_end"] = JsonValue(nv_end.sm_clock_mhz);
        nvo["mem_clock_mhz_start"] = JsonValue(nv_start.mem_clock_mhz);
        nvo["mem_clock_mhz_end"] = JsonValue(nv_end.mem_clock_mhz);
        nvo["power_w_avg"] = JsonValue((nv_start.power_w + nv_end.power_w) * 0.5);
        hw["nvml_telemetry"] = JsonValue(std::move(nvo));
    }
    root["hardware"] = JsonValue(std::move(hw));

    JsonObject l2f;
    l2f["enabled"] = JsonValue(true);
    l2f["buffer_bytes"] = JsonValue(static_cast<int64_t>(std::max<size_t>(l2_bytes * 2, 64ULL * 1024ULL * 1024ULL)));
    l2f["device_l2_bytes"] = JsonValue(static_cast<int64_t>(l2_bytes));
    JsonObject iab;
    iab["enabled"] = JsonValue(true);
    iab["pattern"] = JsonValue("ABAB");
    iab["warmup_iters"] = JsonValue(5);
    iab["measured_iters"] = JsonValue(30);
    JsonObject meth;
    meth["l2_flush"] = JsonValue(std::move(l2f));
    meth["interleaved_ab"] = JsonValue(std::move(iab));
    meth["timer_backend"] = JsonValue("hybrid");
    root["methodology"] = JsonValue(std::move(meth));

    JsonArray b_arr;
    int passed = 0, failed = 0;
    for (const auto& e : entries) {
        b_arr.push_back(entry_to_json(e, nv_end));
        if (e.status == "pass") ++passed;
        else ++failed;
    }
    root["benchmarks"] = JsonValue(std::move(b_arr));

    JsonObject summary;
    summary["total_cases"] = JsonValue(static_cast<int64_t>(entries.size()));
    summary["passed"] = JsonValue(passed);
    summary["failed"] = JsonValue(failed);
    summary["skipped"] = JsonValue(0);
    const bool success = benchmark_passed(entries.size(), failed);
    summary["overall_status"] = JsonValue(success ? "pass" : "fail");
    root["summary"] = JsonValue(std::move(summary));

    if (!write(root)) return 1;

    std::cout << "Benchmark summary (" << entries.size() << " entries) -> " << output_json_path << "\n";
    for (const auto& e : entries) {
        std::cout << "  [" << e.backend << "] " << e.name << ": median="
                  << std::fixed << std::setprecision(4) << e.timing.median << " ms";
        if (e.speedup_vs_baseline > 0.0) std::cout << " (speedup=" << std::setprecision(2) << e.speedup_vs_baseline << "x)";
        if (e.achieved_bandwidth_gbps > 0.0) std::cout << " (" << std::setprecision(1) << e.achieved_bandwidth_gbps << " GB/s)";
        if (e.achieved_tflops > 0.0) std::cout << " (" << std::setprecision(2) << e.achieved_tflops << " TFLOPS)";
        if (e.decode_tok_per_s > 0.0) std::cout << " (" << std::setprecision(1) << e.decode_tok_per_s << " tok/s)";
        std::cout << "\n";
    }
    return success ? 0 : 1;
#endif
}

}  // namespace fusionrt::bench
