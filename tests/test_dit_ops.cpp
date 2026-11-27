#if defined(FUSIONRT_ENABLE_CUDA) || defined(FUSIONRT_DIT_TEST_MAIN)

#include "fusionrt/dit_ops.h"

#include "fusionrt/cuda_kernels.h"
#include "fusionrt/weight_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace fusionrt;
using namespace fusionrt::cuda_ops;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class Fn>
void require_invalid(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

struct DeviceBuffer {
    explicit DeviceBuffer(size_t bytes) {
        if (bytes) FUSIONRT_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    template<class T = uint8_t>
    T* data(size_t offset = 0) const { return reinterpret_cast<T*>(static_cast<uint8_t*>(ptr) + offset); }

    void upload(const void* src, size_t bytes, cudaStream_t stream, size_t offset = 0) const {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data(offset), src, bytes, cudaMemcpyHostToDevice, stream));
    }

    template<class T>
    std::vector<T> download(size_t count, cudaStream_t stream, size_t offset = 0) const {
        std::vector<T> host(count);
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host.data(), data(offset), count * sizeof(T),
                                            cudaMemcpyDeviceToHost, stream));
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream));
        return host;
    }

    void* ptr = nullptr;
};

struct Stream {
    Stream() { FUSIONRT_CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    cudaStream_t value = nullptr;
};

std::vector<float> values(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> v(count);
    for (auto& value : v)
        value = (static_cast<int>(rng() & 65535u) - 32768) * (1.0f / 32768.0f);
    return v;
}

uint16_t encode_narrow(float v, DType dtype) {
    return dtype == DType::F16 ? fp32_to_fp16(v) : fp32_to_bf16(v);
}

float decode_narrow(uint16_t v, DType dtype) {
    return dtype == DType::F16 ? fp16_to_fp32(v) : bf16_to_fp32(v);
}

void upload_floats(const DeviceBuffer& buffer, std::vector<float>& v, DType dtype,
                   cudaStream_t stream, size_t offset = 0) {
    if (dtype == DType::F32) {
        buffer.upload(v.data(), v.size() * sizeof(float), stream, offset);
    } else {
        std::vector<uint16_t> packed(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            packed[i] = encode_narrow(v[i], dtype);
            v[i] = decode_narrow(packed[i], dtype);
        }
        buffer.upload(packed.data(), packed.size() * sizeof(uint16_t), stream, offset);
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

std::vector<float> download_floats(const DeviceBuffer& buffer, size_t count, DType dtype,
                                   cudaStream_t stream, size_t offset) {
    if (dtype == DType::F32) return buffer.download<float>(count, stream, offset);
    const auto packed = buffer.download<uint16_t>(count, stream, offset);
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = decode_narrow(packed[i], dtype);
    return result;
}

struct WeightFixture {
    std::vector<uint8_t> packed;
    std::vector<float> reference;
};

template<class In, class Out>
std::vector<Out> transform_weight_rows(const std::vector<In>& src, DType from, DType to,
                                       int N, int K, void (*convert)(const In*, Out*, int64_t)) {
    const size_t in_row = K / dtype_block_size(from), out_row = K / dtype_block_size(to);
    std::vector<Out> dst(N * out_row);
    for (int n = 0; n < N; ++n) convert(src.data() + n * in_row, dst.data() + n * out_row, K);
    return dst;
}

WeightFixture make_weight(DType dtype, int N, int K, bool reference = true) {
    WeightFixture result;
    std::mt19937 rng(234);
    auto randomize = [&](auto& blocks) {
        for (auto& byte : std::as_writable_bytes(std::span(blocks)))
            byte = static_cast<std::byte>(rng() & 255u);
    };
    auto pack = [&](const auto& blocks, auto dequant) {
        const auto bytes = std::as_bytes(std::span(blocks));
        result.packed.resize(bytes.size());
        std::memcpy(result.packed.data(), bytes.data(), bytes.size());
        if (reference) result.reference = transform_weight_rows(blocks, dtype, DType::F32, N, K, dequant);
    };
    const size_t blocks = static_cast<size_t>(N) * (K / dtype_block_size(dtype));

    if (dtype == DType::Q4_A || dtype == DType::Q4_0) {
        const auto src = transform_weight_rows(values(static_cast<size_t>(N) * K, 234),
                                               DType::F32, DType::Q4_0, N, K, quantize_row_q4_0);
        if (dtype == DType::Q4_0) pack(src, dequantize_row_q4_0);
        else pack(transform_weight_rows(src, DType::Q4_0, dtype, N, K, repack_q4_0_to_q4_a), dequantize_row_q4_a);
    } else if (dtype == DType::Q8_A || dtype == DType::Q8_0) {
        const auto src = transform_weight_rows(values(static_cast<size_t>(N) * K, 234),
                                               DType::F32, DType::Q8_0, N, K, quantize_row_q8_0);
        if (dtype == DType::Q8_0) pack(src, dequantize_row_q8_0);
        else pack(transform_weight_rows(src, DType::Q8_0, dtype, N, K, repack_q8_0_to_q8_a), dequantize_row_q8_a);
    } else if (dtype == DType::Q4_1) {
        std::vector<BlockQ4_1> src(blocks);
        randomize(src);
        for (auto& b : src) {
            b.d = fp32_to_fp16(0.0859375f);
            b.m = fp32_to_fp16(-0.625f);
        }
        pack(src, dequantize_row_q4_1);
    } else if (dtype == DType::Q4_K || dtype == DType::Q4_KA) {
        std::vector<BlockQ4_K> src(blocks);
        randomize(src);
        for (auto& b : src) {
            b.d = fp32_to_fp16(0.0029296875f);
            b.dmin = fp32_to_fp16(0.015625f);
        }
        if (dtype == DType::Q4_K) pack(src, dequantize_row_q4_k);
        else pack(transform_weight_rows(src, DType::Q4_K, dtype, N, K, repack_q4_k_to_q4_ka), dequantize_row_q4_ka);
    } else if (dtype == DType::Q5_K) {
        std::vector<BlockQ5_K> src(blocks);
        randomize(src);
        for (auto& b : src) {
            b.d = fp32_to_fp16(0.00146484375f);
            b.dmin = fp32_to_fp16(0.015625f);
        }
        pack(src, dequantize_row_q5_k);
    } else if (dtype == DType::Q6_K || dtype == DType::Q6_A) {
        std::vector<BlockQ6_K> src(blocks);
        randomize(src);
        for (auto& b : src) b.d = fp32_to_fp16(0.000244140625f);
        if (dtype == DType::Q6_K) pack(src, dequantize_row_q6_k);
        else pack(transform_weight_rows(src, DType::Q6_K, dtype, N, K, repack_q6_k_to_q6_a), dequantize_row_q6_a);
    } else if (dtype == DType::Q2_K) {
        pack(transform_weight_rows(values(static_cast<size_t>(N) * K, 234),
                                    DType::F32, dtype, N, K, quantize_row_q2_k), dequantize_row_q2_k);
    } else {
        visit_weight_codec(dtype, [&]<class Codec>() {
            result.packed.resize(blocks * Codec::kBlockBytes);
            result.reference.resize(blocks * Codec::kBlockSize);
            randomize(result.packed);
            int scale_offset = 0;
            if (dtype == DType::Q3_K) scale_offset = 108;
            if (dtype == DType::TQ1_0) scale_offset = 52;
            if (dtype == DType::TQ2_0) scale_offset = 64;
            if (dtype == DType::PTQ1_0) scale_offset = 26;
            for (size_t b = 0; b < blocks; ++b) {
                auto* block = result.packed.data() + b * Codec::kBlockBytes;
                if (dtype == DType::MXFP4) block[0] = 122 + static_cast<uint8_t>(b % 4);
                else {
                    const uint16_t scale = fp32_to_fp16((1.0f + static_cast<float>(b % 5)) / 512.0f);
                    block[scale_offset] = static_cast<uint8_t>(scale);
                    block[scale_offset + 1] = static_cast<uint8_t>(scale >> 8);
                }
                Codec::dequant_block(block, result.reference.data() + b * Codec::kBlockSize);
            }
        });
    }
    return result;
}

struct Accuracy {
    double max_abs = 0, rel_l2 = 0, cosine = 1;
};

Accuracy accuracy_case(const WeightFixture& weights, const DeviceBuffer& w, DType w_dtype, DType x_dtype,
                       DType y_dtype, DType bias_dtype, int M, int N, int K, cudaStream_t stream,
                       bool use_bias = true, bool unaligned = false) {
    const size_t x_size = dtype_type_size(x_dtype), y_size = dtype_type_size(y_dtype);
    const size_t bias_size = dtype_type_size(bias_dtype);
    auto x = values(static_cast<size_t>(M) * K, 14);
    auto bias = use_bias ? values(N, 42) : std::vector<float>{};
    for (auto& b : bias) b *= 0.25f;
    const size_t x_offset = unaligned ? x_size : 0;
    const size_t y_offset = unaligned ? y_size : 0;
    const size_t bias_offset = unaligned ? bias_size : 0;
    const DeviceBuffer dx(x.size() * x_size + x_offset), dy(static_cast<size_t>(M) * N * y_size + y_offset);
    const DeviceBuffer db(use_bias ? bias.size() * bias_size + bias_offset : 0);
    const size_t bytes = dit_linear_workspace_bytes(M, N, K, w_dtype, x_dtype, y_dtype);
    const DeviceBuffer workspace(bytes);
    upload_floats(dx, x, x_dtype, stream, x_offset);
    if (use_bias) upload_floats(db, bias, bias_dtype, stream, bias_offset);

    dit_linear_quantized(dx.data(x_offset), x_dtype, w.ptr, w_dtype,
                         use_bias ? db.data(bias_offset) : nullptr, bias_dtype, dy.data(y_offset), y_dtype,
                         M, N, K, workspace.ptr, bytes, stream);
    const auto got = download_floats(dy, static_cast<size_t>(M) * N, y_dtype, stream, y_offset);
    Accuracy a;
    double dot = 0, ref_l2 = 0, got_l2 = 0, diff_l2 = 0, ref_max = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k)
                expected += x[static_cast<size_t>(m) * K + k] * weights.reference[static_cast<size_t>(n) * K + k];
            if (use_bias) expected += bias[n];
            const double actual = got[static_cast<size_t>(m) * N + n];
            require(std::isfinite(actual), "Non-finite DiT output");
            const double diff = actual - expected;
            a.max_abs = std::max(a.max_abs, std::abs(diff));
            ref_max = std::max(ref_max, std::abs(static_cast<double>(expected)));
            dot += actual * expected;
            ref_l2 += static_cast<double>(expected) * expected;
            got_l2 += actual * actual;
            diff_l2 += diff * diff;
        }
    }
    a.rel_l2 = std::sqrt(diff_l2 / ref_l2);
    a.cosine = dot / std::sqrt(ref_l2 * got_l2);
    if (!(a.cosine > 0.995 && a.rel_l2 < 0.02 && a.max_abs < 0.03 * ref_max + 1e-4)) {
        std::cerr << dtype_name(w_dtype) << ' ' << dtype_name(x_dtype) << " -> " << dtype_name(y_dtype)
                  << " M=" << M << " max_abs=" << a.max_abs << " rel_l2=" << a.rel_l2 << " cos=" << a.cosine << '\n';
        throw std::runtime_error("DiT result differs from CPU FP32 reference");
    }
    return a;
}

void test_dit_accuracy(cudaStream_t stream) {
    constexpr DType weights[] = {
        DType::Q4_A, DType::Q4_KA, DType::Q8_A, DType::Q6_A, DType::Q4_K, DType::Q5_K,
        DType::Q6_K, DType::Q2_K, DType::Q4_0, DType::Q4_1, DType::Q8_0,
#define FUSIONRT_DIT_FORMAT(name, id, block, bytes) DType::name,
        FUSIONRT_WEIGHT_CODEC_FORMATS(FUSIONRT_DIT_FORMAT)
#undef FUSIONRT_DIT_FORMAT
    };
    constexpr int N = 67;
    Accuracy worst;
    auto record = [&](Accuracy a) {
        worst.max_abs = std::max(worst.max_abs, a.max_abs);
        worst.rel_l2 = std::max(worst.rel_l2, a.rel_l2);
        worst.cosine = std::min(worst.cosine, a.cosine);
    };
    for (DType w_dtype : weights) {
        require(dit_supports_weight_dtype(w_dtype), "Missing DiT quantized dtype");
        const auto block = dtype_block_size(w_dtype);
        const int K = block == 32 ? 96 : 512;
        const auto fixture = make_weight(w_dtype, N, K);
        const DeviceBuffer w(fixture.packed.size());
        w.upload(fixture.packed.data(), fixture.packed.size(), stream);
        for (DType dtype : {DType::BF16, DType::F16}) {
            const auto a = accuracy_case(fixture, w, w_dtype, dtype, dtype, dtype, 17, N, K, stream);
            record(a);
            std::cout << std::setprecision(8) << "  [PASS] DiT " << dtype_name(w_dtype) << ' ' << dtype_name(dtype)
                      << " M=17 max_abs=" << a.max_abs << " rel_l2=" << a.rel_l2 << " cos=" << a.cosine << '\n';
        }
        record(accuracy_case(fixture, w, w_dtype, DType::F32, DType::F32, DType::F32, 1, N, K, stream, false));
        record(accuracy_case(fixture, w, w_dtype, DType::BF16, DType::F16, DType::F32, 5, N, K, stream, true, true));
        record(accuracy_case(fixture, w, w_dtype, DType::F16, DType::BF16, DType::BF16, 9, N, K, stream, false));
        if (has_weight_codec_int8(w_dtype))
            record(accuracy_case(fixture, w, w_dtype, DType::BF16, DType::F32, DType::F16, 16, N, K, stream));
    }
    std::cout << "  [PASS] DiT aggregate max_abs=" << worst.max_abs << " max_rel_l2=" << worst.rel_l2
              << " min_cos=" << worst.cosine << '\n';
}

void test_dit_helpers(cudaStream_t stream) {
    constexpr int K = 96;
    constexpr float inf = std::numeric_limits<float>::infinity();
    for (DType dtype : {DType::F32, DType::BF16, DType::F16}) {
        std::vector<float> src(K);
        for (int i = 0; i < 32; ++i) src[i] = i - 16.0f;
        src[13] = 127.0f;
        src[15] = -127.0f;
        src[28] = std::numeric_limits<float>::quiet_NaN();
        src[29] = inf;
        src[30] = -inf;
        src[64] = dtype == DType::F16 ? 0x1p-24f : 0x1p-126f;
        src[65] = -src[64];
        const size_t size = dtype_type_size(dtype);
        const DeviceBuffer x((K + 1) * size), q8((K / 32) * sizeof(BlockQ8_1)), cast(K * sizeof(float));
        upload_floats(x, src, dtype, stream, size);
        dit_quantize_x_q8_1(x.data(size), dtype, q8.ptr, 1, K, stream);
        dit_cast_to_f32(x.data(size), dtype, cast.data<float>(), K, stream);
        const auto got = q8.download<BlockQ8_1>(K / 32, stream);
        const auto restored = cast.download<float>(K, stream);
        const auto* quants = reinterpret_cast<const int8_t*>(got.front().qs);
        int lo = 0, hi = 0;
        for (int i = 0; i < K; ++i) {
            if (i < 32) {
                const int q = std::isfinite(src[i]) ? static_cast<int>(src[i]) : 0;
                require(quants[i] == q, "DiT Q8_1 value or non-finite handling");
                (i < 16 ? lo : hi) += q;
            }
            require(std::isnan(src[i]) ? std::isnan(restored[i]) : restored[i] == src[i], "DiT F32 cast value");
        }
        require(got[0].d == 1.0f && got[0].s_lo == lo && got[0].s_hi == hi && got[0].s_all == lo + hi,
                "DiT Q8_1 scale and sums");
        require(got[1].d == 0.0f && got[1].s_all == 0.0f &&
                std::all_of(std::begin(got[1].qs), std::end(got[1].qs), [](int q) { return q == 0; }),
                "DiT Q8_1 all-zero block");
        require(got[2].d > 0.0f && got[2].s_all == 0.0f && (got[2].qs[0] & 65535) == 0x817f,
                "DiT Q8_1 tiny finite block");
    }

    require(!dit_supports_weight_dtype(DType::F16) && !dit_supports_weight_dtype(DType::Q8_1),
            "Unexpected DiT weight dtype");
    for (const auto [M, N] : {std::array{0, 67}, std::array{17, 0}}) {
        require(dit_linear_workspace_bytes(M, N, 0, DType::U8, DType::U8, DType::U8) == 0,
                "DiT empty workspace");
        dit_linear_quantized(nullptr, DType::U8, nullptr, DType::U8, nullptr, DType::U8,
                             nullptr, DType::U8, M, N, 0, nullptr, 0, stream);
    }
    require_invalid([] {
        (void)dit_linear_workspace_bytes(17, 67, 510, DType::Q4_K, DType::BF16, DType::BF16);
    }, "DiT accepted an incomplete weight block");
    require_invalid([] {
        (void)dit_linear_workspace_bytes(17, 67, 512, DType::Q4_K, DType::I32, DType::BF16);
    }, "DiT accepted an unsupported activation dtype");
    require_invalid([&] {
        dit_quantize_x_q8_1(nullptr, DType::F16, nullptr, 1, 33, stream);
    }, "DiT accepted an incomplete Q8_1 block");
    std::cout << "  [PASS] DiT Q8_1 quantization, non-finite inputs, scalar alignment, F32 casting and validation\n";
}

void test_dit_tiles(cudaStream_t stream) {
    constexpr int M = 2500, N = 512, K = 512;
    const auto workspace_bytes = [](int m) {
        return dit_linear_workspace_bytes(m, N, K, DType::Q4_KA, DType::BF16, DType::BF16);
    };
    const size_t bytes = workspace_bytes(M);
    require(bytes == workspace_bytes(1024) && bytes == workspace_bytes(16384) && bytes == workspace_bytes(65536),
            "DiT workspace grows past one row tile");
    require(bytes < workspace_bytes(512) * 3 && bytes <= 48 * 1024 * 1024, "DiT workspace bound");
    const size_t wide = dit_linear_workspace_bytes(65536, 16384, 16384, DType::IQ4_XS, DType::BF16, DType::BF16);
    require(wide <= 48 * 1024 * 1024, "DiT wide workspace exceeds the tile budget");

    const auto fixture = make_weight(DType::Q4_KA, N, K, false);
    auto src = values(static_cast<size_t>(M) * K, 17), bias = values(N, 19);
    const DeviceBuffer x(src.size() * sizeof(uint16_t)), w(fixture.packed.size()), b(N * sizeof(float));
    const DeviceBuffer y(static_cast<size_t>(M) * N * sizeof(uint16_t)), single(static_cast<size_t>(M) * N * sizeof(uint16_t));
    const DeviceBuffer workspace(bytes);
    const DeviceBuffer q8(static_cast<size_t>(M) * (K / 32) * sizeof(BlockQ8_1));
    const DeviceBuffer acc(static_cast<size_t>(M) * N * sizeof(float));
    upload_floats(x, src, DType::BF16, stream);
    upload_floats(b, bias, DType::F32, stream);
    w.upload(fixture.packed.data(), fixture.packed.size(), stream);

    auto launch = [&](size_t scratch_bytes) {
        dit_linear_quantized(x.ptr, DType::BF16, w.ptr, DType::Q4_KA, b.ptr, DType::F32, y.ptr, DType::BF16,
                             M, N, K, workspace.ptr, scratch_bytes, stream);
    };
    require_invalid([&] { launch(bytes - 1); }, "DiT accepted insufficient workspace");
    launch(bytes);
    dit_quantize_x_q8_1(x.ptr, DType::BF16, q8.ptr, M, K, stream);
    mmq_q4_ka_q8_1(q8.ptr, w.ptr, acc.data<float>(), M, N, K, stream);
    dit_f32_to_out_bias(acc.data<float>(), b.ptr, DType::F32, single.ptr, DType::BF16, M, N, stream);
    const auto expected = single.download<uint16_t>(static_cast<size_t>(M) * N, stream);
    require(y.download<uint16_t>(expected.size(), stream) == expected, "Tiled DiT differs from one-shot MMQ");

    struct Graph {
        cudaGraph_t definition = nullptr;
        cudaGraphExec_t executable = nullptr;
        ~Graph() {
            if (executable) cudaGraphExecDestroy(executable);
            if (definition) cudaGraphDestroy(definition);
        }
    } graph;
    FUSIONRT_CUDA_CHECK(cudaMemsetAsync(y.ptr, 0, expected.size() * sizeof(uint16_t), stream));
    FUSIONRT_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(bytes);
    FUSIONRT_CUDA_CHECK(cudaStreamEndCapture(stream, &graph.definition));
    FUSIONRT_CUDA_CHECK(cudaGraphInstantiate(&graph.executable, graph.definition, 0));
    FUSIONRT_CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    require(y.download<uint16_t>(expected.size(), stream) == expected, "Captured DiT differs from one-shot MMQ");
    std::cout << "  [PASS] DiT M=" << M << " bit-exact against one-shot MMQ, eager and CUDA Graph"
              << ", workspace=" << bytes << " bytes through M=65536, wide workspace=" << wide << " bytes\n";
}

#ifdef FUSIONRT_DIT_TEST_MAIN
template<class Fn>
double time_ms(cudaStream_t stream, Fn&& fn) {
    constexpr int repeats = 20;
    struct Events {
        cudaEvent_t start = nullptr, stop = nullptr;
        ~Events() {
            if (start) cudaEventDestroy(start);
            if (stop) cudaEventDestroy(stop);
        }
    } events;
    for (int i = 0; i < 5; ++i) fn();
    FUSIONRT_CUDA_CHECK(cudaEventCreate(&events.start));
    FUSIONRT_CUDA_CHECK(cudaEventCreate(&events.stop));
    FUSIONRT_CUDA_CHECK(cudaEventRecord(events.start, stream));
    for (int i = 0; i < repeats; ++i) fn();
    FUSIONRT_CUDA_CHECK(cudaEventRecord(events.stop, stream));
    FUSIONRT_CUDA_CHECK(cudaEventSynchronize(events.stop));
    float ms = 0;
    FUSIONRT_CUDA_CHECK(cudaEventElapsedTime(&ms, events.start, events.stop));
    return ms / repeats;
}

void benchmark_dit(cudaStream_t stream, const cudaDeviceProp& props) {
    constexpr int M = 4096, N = 4096, K = 4096;
    const auto fixture = make_weight(DType::Q4_KA, N, K, false);
    auto src = values(static_cast<size_t>(M) * K, 31);
    const DeviceBuffer x(src.size() * sizeof(uint16_t)), w(fixture.packed.size());
    const DeviceBuffer y(static_cast<size_t>(M) * N * sizeof(uint16_t));
    const size_t bytes = dit_linear_workspace_bytes(M, N, K, DType::Q4_KA, DType::BF16, DType::BF16);
    const DeviceBuffer workspace(bytes);
    const DeviceBuffer q8(static_cast<size_t>(M) * (K / 32) * sizeof(BlockQ8_1));
    const DeviceBuffer acc(static_cast<size_t>(M) * N * sizeof(float));
    upload_floats(x, src, DType::BF16, stream);
    w.upload(fixture.packed.data(), fixture.packed.size(), stream);
    dit_quantize_x_q8_1(x.ptr, DType::BF16, q8.ptr, M, K, stream);

    int device = 0, clock_khz = 0, mem_clock_khz = 0;
    FUSIONRT_CUDA_CHECK(cudaGetDevice(&device));
    FUSIONRT_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    FUSIONRT_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    const CostHardware hw{
        .sm_count = props.multiProcessorCount,
        .hbm_bw_gbps = 2.0 * mem_clock_khz * (props.memoryBusWidth / 8.0) / 1e6,
        .int8_tc_tflops = 2048.0 * props.multiProcessorCount * clock_khz / 1e9,
        .l2_bytes = static_cast<size_t>(props.l2CacheSize),
        .sm_clock_ghz = clock_khz / 1e6,
        .shared_bytes_per_sm = static_cast<int>(props.sharedMemPerMultiprocessor),
        .shared_bytes_per_block = static_cast<int>(props.sharedMemPerBlockOptin),
    };
    const double direct_ms = time_ms(stream, [&] {
        dit_linear_quantized(x.ptr, DType::BF16, w.ptr, DType::Q4_KA, nullptr, DType::F32, y.ptr, DType::BF16,
                             M, N, K, workspace.ptr, bytes, stream, hw);
    });
    const double raw_ms = time_ms(stream, [&] {
        mmq_q4_ka_q8_1(q8.ptr, w.ptr, acc.data<float>(), M, N, K, stream, hw);
    });
    const double ops = 2.0 * M * N * K;
    const size_t direct_bytes = (static_cast<size_t>(M) * K + static_cast<size_t>(M) * N) * sizeof(uint16_t)
        + fixture.packed.size();
    const size_t raw_bytes = static_cast<size_t>(M) * (K / 32) * sizeof(BlockQ8_1)
        + static_cast<size_t>(M) * N * sizeof(float) + fixture.packed.size();
    std::cout << std::fixed << std::setprecision(3)
              << "  [BENCH] BF16 -> Q4_KA -> BF16 M=N=K=4096, " << direct_ms << " ms, "
              << ops / direct_ms / 1e9 << " effective TFLOP/s, "
              << direct_bytes / direct_ms / 1e6 << " effective GB/s, workspace=" << bytes << " bytes\n"
              << "  [BENCH] prequantized Q8_1 -> Q4_KA -> F32 MMQ only, " << raw_ms << " ms, "
              << ops / raw_ms / 1e9 << " effective TFLOP/s, "
              << raw_bytes / raw_ms / 1e6 << " effective GB/s\n";
}
#endif

}  // namespace

void run_dit_ops_tests() {
    const Stream stream;
    test_dit_helpers(stream.value);
    test_dit_accuracy(stream.value);
    test_dit_tiles(stream.value);
}

#ifdef FUSIONRT_DIT_TEST_MAIN
int main() {
    try {
        int device = 0;
        cudaDeviceProp props{};
        FUSIONRT_CUDA_CHECK(cudaGetDevice(&device));
        FUSIONRT_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        std::cout << props.name << " (sm_" << props.major << props.minor << ")\n";
        run_dit_ops_tests();
        const Stream stream;
        benchmark_dit(stream.value, props);
        std::cout << "ALL DIT OPS TESTS PASSED!\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
#endif

#endif
