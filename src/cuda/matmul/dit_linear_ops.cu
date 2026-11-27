#include "fusionrt/dit_ops.h"

#include "fusionrt/cuda_kernels.h"
#include "fusionrt/weight_codec.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <stdexcept>
#include <type_traits>

namespace fusionrt::cuda_ops {
namespace {

constexpr int kThreads = 256;
constexpr int kVector = 8;
constexpr int kQ8Block = 32;
constexpr int kLanesPerQ8 = kQ8Block / kVector;
constexpr int kMaxTileRows = 1024;
constexpr size_t kTileBudget = 48 * 1024 * 1024;

__host__ __device__ bool aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

void require_buffer(const void* ptr, size_t alignment) {
    if (!ptr || !aligned(ptr, alignment))
        throw std::invalid_argument("Missing or misaligned DiT buffer");
}

bool is_float_dtype(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16;
}

template<class Fn>
void visit_float_dtype(DType dtype, Fn&& fn) {
    switch (dtype) {
        case DType::F32:  return fn.template operator()<DType::F32>();
        case DType::F16:  return fn.template operator()<DType::F16>();
        case DType::BF16: return fn.template operator()<DType::BF16>();
        default: throw std::invalid_argument("Unsupported DiT floating-point dtype");
    }
}

unsigned launch_blocks(size_t items, size_t items_per_cta) {
    return static_cast<unsigned>(std::min((items - 1) / items_per_cta + 1, size_t{65535}));
}

template<DType Format>
using Element = std::conditional_t<Format == DType::F32, float, uint16_t>;

template<DType Format>
__device__ __forceinline__ float to_float(Element<Format> value) {
    if constexpr (Format == DType::F32) return value;
    if constexpr (Format == DType::F16) return __half2float(__ushort_as_half(value));
    if constexpr (Format == DType::BF16) return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

template<DType Format>
__device__ __forceinline__ Element<Format> from_float(float value) {
    if constexpr (Format == DType::F32) return value;
    if constexpr (Format == DType::F16) return __half_as_ushort(__float2half_rn(value));
    if constexpr (Format == DType::BF16) return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

template<DType Format>
__device__ __forceinline__ void load_vector(const Element<Format>* src, float (&v)[kVector]) {
    if (aligned(src, 16)) {
        if constexpr (Format == DType::F32) {
            const auto* p = reinterpret_cast<const float4*>(src);
            const float4 a = p[0], b = p[1];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
            v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
            const uint4 a = *reinterpret_cast<const uint4*>(src);
            const unsigned words[] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int j = 0; j < kVector / 2; ++j) {
                v[2 * j] = to_float<Format>(static_cast<uint16_t>(words[j]));
                v[2 * j + 1] = to_float<Format>(static_cast<uint16_t>(words[j] >> 16));
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < kVector; ++j) v[j] = to_float<Format>(src[j]);
    }
}

template<DType Format>
__device__ __forceinline__ void store_vector(Element<Format>* dst, const float (&v)[kVector]) {
    if (aligned(dst, 16)) {
        if constexpr (Format == DType::F32) {
            auto* p = reinterpret_cast<float4*>(dst);
            p[0] = make_float4(v[0], v[1], v[2], v[3]);
            p[1] = make_float4(v[4], v[5], v[6], v[7]);
        } else {
            unsigned words[kVector / 2];
#pragma unroll
            for (int j = 0; j < kVector / 2; ++j) {
                const unsigned lo = from_float<Format>(v[2 * j]);
                const unsigned hi = from_float<Format>(v[2 * j + 1]);
                words[j] = lo | (hi << 16);
            }
            *reinterpret_cast<uint4*>(dst) = make_uint4(words[0], words[1], words[2], words[3]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < kVector; ++j) dst[j] = from_float<Format>(v[j]);
    }
}

template<DType Format>
__global__ void k_dit_quantize(const Element<Format>* x, BlockQ8_1* out, size_t blocks) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % kLanesPerQ8;
    const unsigned mask = 0xfu << ((threadIdx.x & 31) - lane);
    const unsigned stride = gridDim.x * blockDim.x / kLanesPerQ8;
    for (size_t block = tid / kLanesPerQ8; block < blocks; block += stride) {
        float v[kVector];
        load_vector<Format>(x + block * kQ8Block + lane * kVector, v);
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < kVector; ++j) {
            v[j] = isfinite(v[j]) ? v[j] : 0.0f;
            amax = fmaxf(amax, fabsf(v[j]));
        }
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1, kLanesPerQ8));
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2, kLanesPerQ8));
        const float d = amax * (1.0f / 127.0f);
        // Avoid overflow in the reciprocal for tiny finite activations.
        const bool fast = amax >= 127.0f * FLT_MIN && amax < 0x1p126f;
        const float inverse = fast ? __fdividef(127.0f, amax) : 0.0f;
        unsigned words[kVector / 4] = {};
        int sum = 0;
#pragma unroll
        for (int j = 0; j < kVector; ++j) {
            const float scaled = fast ? v[j] * inverse : (d > 0.0f ? (v[j] / amax) * 127.0f : 0.0f);
            const int q = __float2int_rn(scaled);
            words[j / 4] |= (static_cast<unsigned>(q) & 255u) << (8 * (j % 4));
            sum += q;
        }
        auto& dst = out[block];
        reinterpret_cast<uint2*>(dst.qs)[lane] = make_uint2(words[0], words[1]);
        const int lo = __shfl_sync(mask, sum, 0, kLanesPerQ8) + __shfl_sync(mask, sum, 1, kLanesPerQ8);
        const int hi = __shfl_sync(mask, sum, 2, kLanesPerQ8) + __shfl_sync(mask, sum, 3, kLanesPerQ8);
        if (lane == 0) {
            *reinterpret_cast<float4*>(&dst.d) = make_float4(d, d * lo, d * hi, d * (lo + hi));
        }
    }
}

template<DType Src>
__global__ void k_dit_cast(const Element<Src>* src, float* dst, size_t count) {
    const size_t start = (blockIdx.x * blockDim.x + threadIdx.x) * kVector;
    const size_t stride = gridDim.x * blockDim.x * kVector;
    for (size_t i = start; i < count; i += stride) {
        if (count - i >= kVector) {
            float v[kVector];
            load_vector<Src>(src + i, v);
            store_vector<DType::F32>(dst + i, v);
        } else {
            for (size_t j = i; j < count; ++j) dst[j] = to_float<Src>(src[j]);
        }
    }
}

template<DType Dst, DType Bias, bool HasBias>
__global__ void k_dit_epilogue(const float* src, const Element<Bias>* bias, Element<Dst>* dst,
                              size_t count, int N) {
    const size_t start = (blockIdx.x * blockDim.x + threadIdx.x) * kVector;
    const size_t stride = gridDim.x * blockDim.x * kVector;
    for (size_t i = start; i < count; i += stride) {
        int col = 0;
        if constexpr (HasBias) col = static_cast<int>(i % N);
        if (count - i >= kVector && (!HasBias || N - col >= kVector)) {
            float v[kVector];
            load_vector<DType::F32>(src + i, v);
            if constexpr (HasBias) {
                float b[kVector];
                load_vector<Bias>(bias + col, b);
#pragma unroll
                for (int j = 0; j < kVector; ++j) v[j] += b[j];
            }
            store_vector<Dst>(dst + i, v);
        } else {
            const int end = static_cast<int>(min(count - i, static_cast<size_t>(kVector)));
            for (int j = 0; j < end; ++j) {
                float value = src[i + j];
                if constexpr (HasBias) {
                    value += to_float<Bias>(bias[col]);
                    if (++col == N) col = 0;
                }
                dst[i + j] = from_float<Dst>(value);
            }
        }
    }
}

struct Int8Linear {
    using Mmq = void (*)(const void*, const void*, float*, int, int, int, cudaStream_t, const CostHardware&);
    using Gemv = void (*)(const void*, const void*, float*, int, int, cudaStream_t, GemvGeometry);
    Mmq mmq = nullptr;
    Gemv gemv = nullptr;
};

Int8Linear int8_linear(DType dtype) {
    switch (dtype) {
        case DType::Q4_A:  return {mmq_q4_a_q8_1, gemv_q4_a_q8_1};
        case DType::Q4_KA: return {mmq_q4_ka_q8_1, gemv_q4_ka_q8_1};
        case DType::Q8_A:  return {mmq_q8_a_q8_1, gemv_q8_a_q8_1};
        case DType::Q6_A:  return {mmq_q6_a_q8_1, gemv_q6_a_q8_1};
        case DType::Q4_K:  return {mmq_q4_k_q8_1, gemv_q4_k_q8_1};
        case DType::Q5_K:  return {mmq_q5_k_q8_1, gemv_q5_k_q8_1};
        case DType::Q6_K:  return {mmq_q6_k_q8_1, gemv_q6_k_q8_1};
        default: return {};
    }
}

using F32Linear = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

F32Linear f32_linear(DType dtype) {
    switch (dtype) {
        case DType::Q4_0: return fused_gemv_q4_0;
        case DType::Q4_1: return fused_gemv_q4_1;
        case DType::Q8_0: return fused_gemv_q8_0;
        case DType::Q2_K:
            return [](const float* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream) {
                linear_q2_k_f32(x, w, y, M, N, K, stream);
            };
        default: return nullptr;
    }
}

struct WorkspaceLayout {
    int rows = 0;
    size_t x_bytes = 0;
    size_t y_bytes = 0;
    size_t verify_bytes = 0;

    size_t bytes() const { return x_bytes + y_bytes + verify_bytes; }
};

WorkspaceLayout workspace_layout(int M, int N, int K, DType w_dtype, DType x_dtype, DType y_dtype) {
    if (M < 0 || N < 0) throw std::invalid_argument("Negative DiT linear dimensions");
    if (M == 0 || N == 0) return {};
    if (!dit_supports_weight_dtype(w_dtype) || !is_float_dtype(x_dtype) || !is_float_dtype(y_dtype))
        throw std::invalid_argument("Unsupported DiT linear dtype");
    if (K <= 0 || K % static_cast<int>(dtype_block_size(w_dtype)) != 0)
        throw std::invalid_argument("DiT K must be a positive multiple of the weight block size");

    const bool codec_int8 = has_weight_codec_int8(w_dtype);
    const bool int8 = int8_linear(w_dtype).mmq || codec_int8;
    const size_t x_row = int8 ? static_cast<size_t>(K / kQ8Block) * sizeof(BlockQ8_1)
                              : x_dtype == DType::F32 ? 0 : static_cast<size_t>(K) * sizeof(float);
    const size_t y_row = y_dtype == DType::F32 ? 0 : static_cast<size_t>(N) * sizeof(float);
    const size_t verify_reserve = codec_int8 ? codec_verify_scratch_bytes(std::min(M, 8), K) + 15 : 0;
    int rows = kMaxTileRows;
    if (x_row + y_row) {
        rows = static_cast<int>(std::clamp((kTileBudget - std::min(kTileBudget, verify_reserve)) /
                                          (x_row + y_row), size_t{1}, size_t{kMaxTileRows}));
        // Leave full Tensor Core tiles in the steady state for wide projections.
        if (rows >= 128) rows -= rows % 128;
    }
    rows = std::min(rows, M);
    const size_t verify = codec_int8 ? codec_verify_scratch_bytes(std::min(rows, 8), K) : 0;
    size_t y_bytes = rows * y_row;
    // The small-batch codec kernel reads verification scratch in int4 vectors.
    if (verify) y_bytes = (y_bytes + 15) & ~size_t{15};
    return {rows, rows * x_row, y_bytes, verify};
}

}  // namespace

bool dit_supports_weight_dtype(DType w_dtype) {
    return is_weight_codec_dtype(w_dtype) || int8_linear(w_dtype).mmq || f32_linear(w_dtype);
}

size_t dit_linear_workspace_bytes(int M, int N, int K, DType w_dtype, DType x_dtype, DType y_dtype) {
    return workspace_layout(M, N, K, w_dtype, x_dtype, y_dtype).bytes();
}

void dit_quantize_x_q8_1(const void* x, DType x_dtype, void* q8_out, int M, int K, cudaStream_t stream) {
    if (M == 0) return;
    if (M < 0 || K <= 0 || K % kQ8Block != 0)
        throw std::invalid_argument("Invalid DiT Q8_1 shape");
    require_buffer(q8_out, alignof(BlockQ8_1));
    const size_t blocks = static_cast<size_t>(M) * (K / kQ8Block);
    const unsigned grid = launch_blocks(blocks, kThreads / kLanesPerQ8);
    visit_float_dtype(x_dtype, [&]<DType X>() {
        require_buffer(x, alignof(Element<X>));
        k_dit_quantize<X><<<grid, kThreads, 0, stream>>>(
            static_cast<const Element<X>*>(x), static_cast<BlockQ8_1*>(q8_out), blocks);
    });
    FUSIONRT_CUDA_CHECK(cudaGetLastError());
}

void dit_f32_to_out_bias(const float* src_f32, const void* bias, DType bias_dtype,
                         void* dst, DType dst_dtype, int M, int N, cudaStream_t stream) {
    if (M < 0 || N < 0) throw std::invalid_argument("Negative DiT epilogue dimensions");
    if (M == 0 || N == 0) return;
    require_buffer(src_f32, alignof(float));
    const size_t count = static_cast<size_t>(M) * N;
    const unsigned grid = launch_blocks(count, kThreads * kVector);
    visit_float_dtype(dst_dtype, [&]<DType Dst>() {
        require_buffer(dst, alignof(Element<Dst>));
        auto launch = [&]<DType Bias, bool HasBias>() {
            k_dit_epilogue<Dst, Bias, HasBias><<<grid, kThreads, 0, stream>>>(
                src_f32, static_cast<const Element<Bias>*>(bias), static_cast<Element<Dst>*>(dst), count, N);
        };
        if (bias) {
            visit_float_dtype(bias_dtype, [&]<DType Bias>() {
                require_buffer(bias, alignof(Element<Bias>));
                launch.template operator()<Bias, true>();
            });
        } else {
            launch.template operator()<DType::F32, false>();
        }
    });
    FUSIONRT_CUDA_CHECK(cudaGetLastError());
}

void dit_cast_to_f32(const void* src, DType src_dtype, float* dst, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    require_buffer(dst, alignof(float));
    if (src_dtype == DType::F32 && src == dst) return;
    const unsigned grid = launch_blocks(count, kThreads * kVector);
    visit_float_dtype(src_dtype, [&]<DType Src>() {
        require_buffer(src, alignof(Element<Src>));
        k_dit_cast<Src><<<grid, kThreads, 0, stream>>>(static_cast<const Element<Src>*>(src), dst, count);
    });
    FUSIONRT_CUDA_CHECK(cudaGetLastError());
}

void dit_linear_quantized(const void* x, DType x_dtype, const void* w, DType w_dtype,
                          const void* bias, DType bias_dtype, void* y, DType y_dtype,
                          int M, int N, int K, void* workspace, size_t workspace_bytes,
                          cudaStream_t stream, const CostHardware& hw) {
    const auto layout = workspace_layout(M, N, K, w_dtype, x_dtype, y_dtype);
    if (!layout.rows) return;
    if (workspace_bytes < layout.bytes())
        throw std::invalid_argument("Insufficient DiT linear workspace");
    if (layout.bytes()) require_buffer(workspace, 16);
    require_buffer(x, dtype_type_size(x_dtype));
    require_buffer(w, 16);
    require_buffer(y, dtype_type_size(y_dtype));
    if (bias) {
        if (!is_float_dtype(bias_dtype)) throw std::invalid_argument("Unsupported DiT bias dtype");
        require_buffer(bias, dtype_type_size(bias_dtype));
    }

    const auto native = int8_linear(w_dtype);
    const auto fallback = f32_linear(w_dtype);
    const bool codec_int8 = has_weight_codec_int8(w_dtype);
    auto* scratch = static_cast<uint8_t*>(workspace);
    auto* y_f32 = layout.y_bytes ? reinterpret_cast<float*>(scratch + layout.x_bytes) : nullptr;
    auto* verify = layout.verify_bytes ?
        reinterpret_cast<int*>(scratch + layout.x_bytes + layout.y_bytes) : nullptr;

    for (int row = 0; row < M;) {
        const int rows = std::min(M - row, layout.rows);
        const auto* x_tile = static_cast<const uint8_t*>(x) +
            static_cast<size_t>(row) * K * dtype_type_size(x_dtype);
        auto* y_tile = static_cast<uint8_t*>(y) + static_cast<size_t>(row) * N * dtype_type_size(y_dtype);
        auto* acc = y_f32 ? y_f32 : reinterpret_cast<float*>(y_tile);
        if (native.mmq || codec_int8) {
            dit_quantize_x_q8_1(x_tile, x_dtype, scratch, rows, K, stream);
            if (native.mmq) {
                if (rows == 1) native.gemv(scratch, w, acc, N, K, stream, {});
                else native.mmq(scratch, w, acc, rows, N, K, stream, hw);
            } else if (rows >= 16) {
                codec_mmq_q8_1(scratch, w, w_dtype, acc, rows, N, K, stream);
            } else {
                codec_gemv_q8_1(scratch, w, w_dtype, acc, rows, N, K, stream, verify);
            }
        } else {
            const float* input = reinterpret_cast<const float*>(x_tile);
            if (layout.x_bytes) {
                auto* cast = reinterpret_cast<float*>(scratch);
                dit_cast_to_f32(x_tile, x_dtype, cast, static_cast<size_t>(rows) * K, stream);
                input = cast;
            }
            if (fallback) fallback(input, w, acc, rows, N, K, stream);
            else codec_gemv_f32(input, w, w_dtype, acc, rows, N, K, stream);
        }
        FUSIONRT_CUDA_CHECK(cudaGetLastError());
        if (y_f32 || bias)
            dit_f32_to_out_bias(acc, bias, bias_dtype, y_tile, y_dtype, rows, N, stream);
        row += rows;
    }
}

}  // namespace fusionrt::cuda_ops
