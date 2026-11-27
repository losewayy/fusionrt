#if defined(FUSIONRT_ENABLE_CUDA) || defined(FUSIONRT_NVFP4_TEST_MAIN)

#include "fusionrt/nvfp4_ops.h"

#include "fusionrt/dit_ops.h"
#include "fusionrt/tensor.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace fusionrt;
using namespace fusionrt::cuda_ops;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasGetStatusName(status));
}

struct Stream {
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    cudaStream_t value{};
};

struct Blas {
    explicit Blas(cudaStream_t stream) {
        check(cublasCreate(&handle));
        const auto status = cublasSetStream(handle, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            check(status);
        }
    }
    ~Blas() { cublasDestroy(handle); }
    cublasHandle_t handle{};
};

template<class T = uint8_t>
class Buffer {
public:
    explicit Buffer(size_t count) { check(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void upload(const std::vector<T>& data, cudaStream_t stream) const {
        check(cudaMemcpyAsync(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
    }

    std::vector<T> download(size_t count, cudaStream_t stream, size_t offset = 0) const {
        std::vector<T> data(count);
        check(cudaMemcpyAsync(data.data(), ptr + offset, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return data;
    }

    T* ptr{};
};

uint16_t encode(float value, DType dtype) {
    return dtype == DType::BF16 ? fp32_to_bf16(value) : fp32_to_fp16(value);
}

float decode(uint16_t value, DType dtype) {
    return dtype == DType::BF16 ? bf16_to_fp32(value) : fp16_to_fp32(value);
}

std::vector<uint16_t> narrow(const std::vector<float>& data, DType dtype) {
    std::vector<uint16_t> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) result[i] = encode(data[i], dtype);
    return result;
}

std::vector<float> normal_values(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> values(count);
    for (auto& value : values) value = normal(rng);
    return values;
}

float tensor_scale(const std::vector<uint16_t>& data, DType dtype) {
    float amax = 0.0f;
    for (const uint16_t value : data) amax = std::max(amax, std::abs(decode(value, dtype)));
    return amax / (6.0f * 448.0f);
}

float e4m3(uint8_t bits) {
    const int exponent = bits >> 3;
    const float mantissa = (bits & 7) / 8.0f;
    return exponent ? std::ldexp(1.0f + mantissa, exponent - 7) : std::ldexp(mantissa, -6);
}

constexpr std::array<float, 16> kFp4{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Mirrors to_blocked's reshape/transpose order by traversing physical tiles.
std::vector<uint8_t> blocked(const std::vector<uint8_t>& linear, int rows, int cols) {
    std::vector<uint8_t> result;
    for (int tile_row = 0; tile_row < rows; tile_row += 128) {
        for (int tile_col = 0; tile_col < cols; tile_col += 4) {
            for (int inner_row = 0; inner_row < 32; ++inner_row) {
                for (int group = 0; group < 4; ++group) {
                    for (int inner_col = 0; inner_col < 4; ++inner_col) {
                        const int row = tile_row + group * 32 + inner_row;
                        const int col = tile_col + inner_col;
                        result.push_back(row < rows && col < cols ? linear[row * cols + col] : 0);
                    }
                }
            }
        }
    }
    return result;
}

void test_format(cudaStream_t stream) {
    constexpr int rows = 129, cols = 80;
    constexpr float global = 0.25f;
    const size_t count = rows * cols;
    std::vector<float> values(count);
    std::vector<uint8_t> packed(count / 2), linear(count / 16);

    for (size_t block = 0; block < linear.size(); ++block) {
        const auto scale = static_cast<uint8_t>(32 + block % 38);
        linear[block] = scale;
        for (size_t j = 0; j < 16; j += 2) {
            const unsigned first = static_cast<unsigned>((block + j) % 16);
            const unsigned second = (first + 1) % 16;
            values[block * 16 + j] = (kFp4[first] * e4m3(scale)) * global;
            values[block * 16 + j + 1] = (kFp4[second] * e4m3(scale)) * global;
            packed[block * 8 + j / 2] = static_cast<uint8_t>((first << 4) | second);
        }
    }

    const auto scales = blocked(linear, rows, cols / 16);
    require(scales.size() == 2048 && scales.size() == nvfp4_swizzled_scale_bytes(rows, cols),
            "NVFP4 scale size differs from comfy-kitchen");
    Buffer<float> x(count), output(count);
    Buffer<uint16_t> x16(count), output16(count);
    Buffer<> qdata(packed.size()), scale_out(scales.size()), scale_in(linear.size());
    scale_in.upload(linear, stream);
    nvfp4_swizzle_scales(scale_in.ptr, scale_out.ptr, rows, cols, stream);
    require(scale_out.download(scales.size(), stream) == scales, "NVFP4 scale swizzle or padding mismatch");

    for (const auto dtype : {DType::F32, DType::BF16, DType::F16}) {
        if (dtype == DType::F32) x.upload(values, stream);
        else x16.upload(narrow(values, dtype), stream);
        const void* input = dtype == DType::F32 ? static_cast<void*>(x.ptr) : x16.ptr;
        void* out = dtype == DType::F32 ? static_cast<void*>(output.ptr) : output16.ptr;
        nvfp4_quantize_2d(input, dtype, global, qdata.ptr, scale_out.ptr, rows, cols, stream);
        require(qdata.download(packed.size(), stream) == packed, "NVFP4 nibble packing mismatch");
        require(scale_out.download(scales.size(), stream) == scales, "NVFP4 fused scale swizzle mismatch");
        nvfp4_dequantize_2d(qdata.ptr, scale_out.ptr, global, out, dtype, rows, cols, stream);
        if (dtype == DType::F32) {
            const auto actual = output.download(count, stream);
            for (size_t i = 0; i < count; ++i) {
                require(actual[i] == values[i] && std::signbit(actual[i]) == std::signbit(values[i]),
                        "NVFP4 F32 roundtrip mismatch");
            }
        } else {
            require(output16.download(count, stream) == narrow(values, dtype), "NVFP4 narrow roundtrip mismatch");
        }
    }

    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> edges{
        0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 6.0f,
        -0.25f, -0.75f, -1.25f, -1.75f, -2.5f, -3.5f, -5.0f, -6.0f,
    };
    edges.resize(64, 0.0f);
    edges[16] = 4096.0f;
    edges[17] = -4096.0f;
    edges[18] = inf;
    edges[19] = nan;
    edges[32] = -0.001f;
    edges[48] = 6.0f / 512.0f;
    x.upload(edges, stream);
    nvfp4_quantize_2d(x.ptr, DType::F32, 1.0f, qdata.ptr, scale_out.ptr, 1, 64, stream);
    std::vector<uint8_t> expected{
        0x02, 0x24, 0x46, 0x67, 0x8a, 0xac, 0xce, 0xef, 0x7f, 0x00, 0, 0, 0, 0, 0, 0,
    };
    expected.resize(32, 0);
    expected[24] = 0x70;
    require(qdata.download(expected.size(), stream) == expected, "NVFP4 E2M1 ties, zeros or saturation mismatch");
    require(scale_out.download(4, stream) == std::vector<uint8_t>{0x38, 0x7e, 0, 1},
            "NVFP4 E4M3 subnormal scale or saturation mismatch");
    std::cout << "  [FORMAT] exact E2M1, UE4M3, nibble order and 128x4 scale bytes\n";
}

struct Shape {
    int m, n, k;
};

void dense_gemm(cublasHandle_t handle, Shape shape, DType dtype, const void* x, const void* w,
                void* y, cudaDataType_t output) {
    const auto type = dtype == DType::BF16 ? CUDA_R_16BF : CUDA_R_16F;
    const float one = 1.0f, zero = 0.0f;
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.n, shape.m, shape.k,
                       &one, w, type, shape.k, x, type, shape.k, &zero, y, output, shape.n,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void test_small_linear(cudaStream_t stream) {
    constexpr Shape s{3, 5, 48};
    constexpr float x_global = 0.25f, w_global = 0.25f;
    const size_t x_count = s.m * s.k, y_count = s.m * s.n;
    std::vector<float> x(x_count), w(s.n * s.k);
    std::vector<uint8_t> raw_weights(w.size() / 2), linear_scales(s.n * s.k / 16);
    std::vector<float> bias{0.125f, -0.25f, 0.125f, 1.0f, -0.5f};
    for (size_t block = 0; block < x.size() / 16; ++block) {
        const float scale = (block + 1) * x_global;
        for (size_t i = 0; i < 16; ++i) x[block * 16 + i] = kFp4[(i + block) % 16] * scale;
    }
    for (size_t block = 0; block < linear_scales.size(); ++block) {
        const auto scale = static_cast<uint8_t>(0x30 + block);
        linear_scales[block] = scale | (block % 2 ? 0x80u : 0u);
        for (size_t i = 0; i < 16; i += 2) {
            const size_t first = (i + block * 3) % 16, second = (first + 7) % 16;
            raw_weights[block * 8 + i / 2] = static_cast<uint8_t>((first << 4) | second);
            w[block * 16 + i] = kFp4[first] * e4m3(scale) * w_global;
            w[block * 16 + i + 1] = kFp4[second] * e4m3(scale) * w_global;
        }
    }

    Buffer<uint16_t> dx(x_count);
    Buffer<uint16_t> db16(s.n);
    Buffer<float> dy(y_count + 1), db(s.n), dw(w.size());
    Buffer<> qw(raw_weights.size()), scales(nvfp4_swizzled_scale_bytes(s.n, s.k));
    Buffer<> scales_linear(linear_scales.size());
    const size_t workspace_bytes = nvfp4_linear_workspace_bytes(s.m, s.n, s.k);
    Buffer<> workspace(workspace_bytes);
    dx.upload(narrow(x, DType::BF16), stream);
    db.upload(bias, stream);
    qw.upload(raw_weights, stream);
    scales_linear.upload(linear_scales, stream);
    nvfp4_swizzle_scales(scales_linear.ptr, scales.ptr, s.n, s.k, stream);
    nvfp4_dequantize_2d(qw.ptr, scales.ptr, w_global, dw.ptr, DType::F32, s.n, s.k, stream);
    const auto dequantized = dw.download(w.size(), stream);
    require(dequantized == w, "NVFP4 dequantization must ignore UE4M3 bit 7");

    auto run = [&](float global, size_t bytes) {
        nvfp4_linear(dx.ptr, DType::BF16, qw.ptr, scales.ptr, w_global, global, db.ptr, DType::F32,
                     dy.ptr + 1, DType::F32, s.m, s.n, s.k, workspace.ptr, bytes, stream);
    };
    bool rejected = false;
    try {
        run(x_global, workspace_bytes - 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "NVFP4 accepted undersized workspace");
    cudaGraph_t cold_graph = nullptr;
    bool needs_warmup = false;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        run(x_global, workspace_bytes);
    } catch (const std::runtime_error& e) {
        needs_warmup = std::string_view(e.what()).find("before CUDA Graph capture") != std::string_view::npos;
    }
    check(cudaStreamEndCapture(stream, &cold_graph));
    check(cudaGraphDestroy(cold_graph));
    require(needs_warmup, "Cold NVFP4 CUDA Graph capture must ask for warmup");

    run(x_global, workspace_bytes);
    const auto actual = dy.download(y_count, stream, 1);
    for (int row = 0; row < s.m; ++row) {
        for (int col = 0; col < s.n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < s.k; ++k) sum += x[row * s.k + k] * dequantized[col * s.k + k];
            require(actual[row * s.n + col] == sum + bias[col], "NVFP4 UE4M3 bit-7 native GEMM parity mismatch");
        }
    }

    struct Graph {
        ~Graph() {
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
        }
        cudaGraph_t graph{};
        cudaGraphExec_t exec{};
    } graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        run(0.0f, workspace_bytes);
    } catch (...) {
        cudaStreamEndCapture(stream, &graph.graph);
        throw;
    }
    check(cudaStreamEndCapture(stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.exec, graph.graph, 0));
    check(cudaGraphLaunch(graph.exec, stream));
    const auto captured = dy.download(y_count, stream, 1);
    run(0.0f, workspace_bytes);
    require(dy.download(y_count, stream, 1) == captured, "NVFP4 graph and eager output differ");
    check(cudaMemsetAsync(dx.ptr, 0, x_count * sizeof(uint16_t), stream));
    check(cudaGraphLaunch(graph.exec, stream));
    const auto zero = dy.download(y_count, stream, 1);
    for (size_t i = 0; i < y_count; ++i) require(zero[i] == bias[i % s.n], "NVFP4 zero activation mismatch");
    for (const auto dtype : {DType::BF16, DType::F16}) {
        db16.upload(narrow(bias, dtype), stream);
        nvfp4_linear(dx.ptr, DType::BF16, qw.ptr, scales.ptr, w_global, 0.0f, db16.ptr, dtype,
                     dy.ptr, DType::F32, s.m, s.n, s.k, workspace.ptr, workspace_bytes, stream);
        require(dy.download(y_count, stream) == zero, "NVFP4 narrow bias mismatch");
    }
    require(nvfp4_linear_workspace_bytes(0, 5, 0) == 0, "NVFP4 empty output requires workspace");
    std::cout << "  [PASS] NVFP4 UE4M3 bit-7 GEMM parity, cold capture rejection and warm graph replay\n";
}

void test_accuracy(cudaStream_t stream) {
    const Blas blas(stream);
    constexpr std::array shapes{
        Shape{16, 3072, 12288}, Shape{128, 5376, 3072}, Shape{1024, 4096, 5376},
        Shape{4096, 4096, 4096}, Shape{4096, 12288, 4096},
    };
    for (const auto s : shapes) {
        const size_t xc = static_cast<size_t>(s.m) * s.k;
        const size_t wc = static_cast<size_t>(s.n) * s.k;
        const size_t yc = static_cast<size_t>(s.m) * s.n;
        const auto x = normal_values(xc, 927), w = normal_values(wc, 981);
        const auto bias = normal_values(s.n, 249);
        const size_t bytes = nvfp4_linear_workspace_bytes(s.m, s.n, s.k);
        Buffer<uint16_t> dx(xc), dw(wc), dy(yc);
        Buffer<float> reference(yc), db(s.n);
        Buffer<> qweight(wc / 2), scales(nvfp4_swizzled_scale_bytes(s.n, s.k)), work(bytes);
        db.upload(bias, stream);
        for (const auto dtype : {DType::BF16, DType::F16}) {
            const auto weights = narrow(w, dtype);
            const auto inputs = narrow(x, dtype);
            const float g_w = tensor_scale(weights, dtype);
            dw.upload(weights, stream);
            dx.upload(inputs, stream);
            nvfp4_quantize_2d(dw.ptr, dtype, g_w, qweight.ptr, scales.ptr, s.n, s.k, stream);
            dense_gemm(blas.handle, s, dtype, dx.ptr, dw.ptr, reference.ptr, CUDA_R_32F);
            const auto ref = reference.download(yc, stream);
            for (const bool with_bias : {false, true}) {
                nvfp4_linear(dx.ptr, dtype, qweight.ptr, scales.ptr, g_w, 0.0f,
                             with_bias ? db.ptr : nullptr, DType::F32, dy.ptr, dtype,
                             s.m, s.n, s.k, work.ptr, bytes, stream);
                const auto actual = dy.download(yc, stream);
                double dot = 0.0, aa = 0.0, rr = 0.0, error = 0.0;
                for (size_t i = 0; i < yc; ++i) {
                    const double a = decode(actual[i], dtype);
                    const double r = ref[i] + (with_bias ? bias[i % s.n] : 0.0f);
                    dot += a * r;
                    aa += a * a;
                    rr += r * r;
                    error += (a - r) * (a - r);
                }
                const double rel_l2 = std::sqrt(error / rr), cosine = dot / std::sqrt(aa * rr);
                std::cout << std::fixed << std::setprecision(6) << "  [ACCURACY] "
                          << s.m << 'x' << s.n << 'x' << s.k
                          << (dtype == DType::BF16 ? " BF16" : " FP16")
                          << (with_bias ? " +bias" : " no-bias")
                          << " rel_l2=" << rel_l2 << " cos=" << cosine << std::endl;
                require(rel_l2 < 0.145 && cosine > 0.99, "NVFP4 accuracy outside tolerance");
            }
        }
    }
}

#ifdef FUSIONRT_NVFP4_TEST_MAIN
template<class Fn>
double time_ms(cudaStream_t stream, Fn&& fn) {
    constexpr int repeats = 10;
    struct Events {
        ~Events() {
            if (start) cudaEventDestroy(start);
            if (end) cudaEventDestroy(end);
        }
        cudaEvent_t start{}, end{};
    } events;
    check(cudaEventCreate(&events.start));
    check(cudaEventCreate(&events.end));
    check(cudaEventRecord(events.start, stream));
    for (int i = 0; i < repeats; ++i) fn();
    check(cudaEventRecord(events.end, stream));
    check(cudaEventSynchronize(events.end));
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, events.start, events.end));
    return elapsed / repeats;
}

void benchmark(Shape s, cudaStream_t stream, const cudaDeviceProp& props, int device) {
    const Blas blas(stream);
    const size_t xc = static_cast<size_t>(s.m) * s.k;
    const size_t wc = static_cast<size_t>(s.n) * s.k;
    const size_t yc = static_cast<size_t>(s.m) * s.n;
    const auto source_x = narrow(normal_values(xc, 341), DType::BF16);
    std::vector<BlockQ4_K> q4(wc / 256);
    std::mt19937 rng(459);
    for (auto& b : q4) {
        b.d = fp32_to_fp16(0.0078125f);
        b.dmin = fp32_to_fp16(0.0625f);
        for (auto& scale : b.scales) scale = 16;
        for (auto& q : b.qs) q = static_cast<uint8_t>(rng());
    }
    std::vector<BlockQ4_KA> q4ka(q4.size());
    repack_q4_k_to_q4_ka(q4.data(), q4ka.data(), static_cast<int64_t>(wc));
    std::vector<float> q4_dense(wc);
    dequantize_row_q4_ka(q4ka.data(), q4_dense.data(), static_cast<int64_t>(wc));
    const auto source_w = narrow(q4_dense, DType::BF16);
    const float w_global = tensor_scale(source_w, DType::BF16);

    Buffer<uint16_t> x(xc), w(wc), y(yc);
    Buffer<BlockQ4_KA> w_q4ka(q4ka.size());
    Buffer<> w_fp4(wc / 2), scales(nvfp4_swizzled_scale_bytes(s.n, s.k));
    const size_t fp4_bytes = nvfp4_linear_workspace_bytes(s.m, s.n, s.k);
    const size_t q4_bytes = dit_linear_workspace_bytes(s.m, s.n, s.k, DType::Q4_KA, DType::BF16, DType::BF16);
    Buffer<> fp4_work(fp4_bytes), q4_work(q4_bytes);
    x.upload(source_x, stream);
    w.upload(source_w, stream);
    w_q4ka.upload(q4ka, stream);
    nvfp4_quantize_2d(w.ptr, DType::BF16, w_global, w_fp4.ptr, scales.ptr, s.n, s.k, stream);

    int sm_khz = 0, mem_khz = 0;
    check(cudaDeviceGetAttribute(&sm_khz, cudaDevAttrClockRate, device));
    check(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, device));
    const CostHardware hw{
        .sm_count = props.multiProcessorCount,
        .hbm_bw_gbps = 2.0 * mem_khz * (props.memoryBusWidth / 8.0) / 1e6,
        .int8_tc_tflops = 2048.0 * props.multiProcessorCount * sm_khz / 1e9,
        .l2_bytes = static_cast<size_t>(props.l2CacheSize),
        .sm_clock_ghz = sm_khz / 1e6,
        .shared_bytes_per_sm = static_cast<int>(props.sharedMemPerMultiprocessor),
        .shared_bytes_per_block = static_cast<int>(props.sharedMemPerBlockOptin),
    };
    auto nvfp4 = [&] {
        nvfp4_linear(x.ptr, DType::BF16, w_fp4.ptr, scales.ptr, w_global, 0.0f, nullptr, DType::F32,
                     y.ptr, DType::BF16, s.m, s.n, s.k, fp4_work.ptr, fp4_bytes, stream);
    };
    auto bf16 = [&] { dense_gemm(blas.handle, s, DType::BF16, x.ptr, w.ptr, y.ptr, CUDA_R_16BF); };
    auto q4_ka = [&] {
        dit_linear_quantized(x.ptr, DType::BF16, w_q4ka.ptr, DType::Q4_KA, nullptr, DType::F32,
                             y.ptr, DType::BF16, s.m, s.n, s.k, q4_work.ptr, q4_bytes, stream, hw);
    };
    for (int i = 0; i < 5; ++i) {
        nvfp4();
        bf16();
        q4_ka();
    }
    check(cudaStreamSynchronize(stream));

    std::array<double, 3> fp4_ms{}, bf16_ms{}, q4_ms{};
    for (size_t i = 0; i < fp4_ms.size(); ++i) {
        fp4_ms[i] = time_ms(stream, nvfp4);
        bf16_ms[i] = time_ms(stream, bf16);
        q4_ms[i] = time_ms(stream, q4_ka);
    }
    const double ops = 2.0 * s.m * s.n * s.k;
    auto report = [&](const char* name, auto samples) {
        std::sort(samples.begin(), samples.end());
        const double ms = samples[1];
        std::cout << std::fixed << std::setprecision(6) << "  [BENCH] "
                  << s.m << 'x' << s.n << 'x' << s.k << ' ' << name << ' '
                  << ms << " ms, " << ops / ms / 1e9 << " effective TFLOP/s\n";
    };
    report("NVFP4 dynamic", fp4_ms);
    report("cuBLAS BF16", bf16_ms);
    report("DiT Q4_KA", q4_ms);
}
#endif

}  // namespace

void run_nvfp4_ops_tests() {
    int device = 0;
    check(cudaGetDevice(&device));
    if (!nvfp4_device_supported(device)) {
        std::cout << "  [SKIP] NVFP4 requires sm_120+ and FP4-capable cuBLASLt\n";
        return;
    }
    const Stream stream;
    test_format(stream.value);
    test_small_linear(stream.value);
    test_accuracy(stream.value);
}

#ifdef FUSIONRT_NVFP4_TEST_MAIN
int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        require(argc <= 2 && (mode.empty() || mode == "--tests-only" || mode == "--bench-only"),
                "Usage: test_nvfp4_ops [--tests-only | --bench-only]");
        int device = 0;
        cudaDeviceProp props{};
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&props, device));
        require(nvfp4_device_supported(device), "NVFP4 verification requires sm_120+ and FP4-capable cuBLASLt");
        std::cout << props.name << " (sm_" << props.major << props.minor << ")\n";
        if (mode != "--bench-only") run_nvfp4_ops_tests();
        if (mode != "--tests-only") {
            const Stream stream;
            benchmark({4096, 4096, 4096}, stream.value, props, device);
            benchmark({4096, 12288, 4096}, stream.value, props, device);
        }
        std::cout << "ALL NVFP4 CHECKS PASSED!\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
#endif

#endif
