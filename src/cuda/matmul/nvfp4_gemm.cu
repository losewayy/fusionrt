#include "fusionrt/nvfp4_ops.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace fusionrt::cuda_ops {
namespace {

constexpr int kScaleBlock = 16;
constexpr int kTileRows = 128;
constexpr int kTileCols = 4;
constexpr int kThreads = 256;
constexpr int kQuantLanes = 4;
constexpr int kReduceBlocks = 512;
constexpr int kOutputTileRows = 1024;
constexpr size_t kWorkspaceAlignment = 256;
constexpr size_t kLtWorkspaceBytes = 16 * 1024 * 1024;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_lt(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cublasLtGetStatusName(status));
}

__host__ __device__ bool aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

void require_buffer(const void* ptr, size_t alignment, const char* name) {
    if (!ptr || !aligned(ptr, alignment))
        throw std::invalid_argument(std::string("Missing or misaligned NVFP4 ") + name);
}

void require_scale(float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale))
        throw std::invalid_argument("NVFP4 global scale must be positive and finite");
}

size_t float_bytes(DType dtype) {
    switch (dtype) {
        case DType::F32: return sizeof(float);
        case DType::F16:
        case DType::BF16: return sizeof(uint16_t);
        default: throw std::invalid_argument("Unsupported NVFP4 floating-point dtype");
    }
}

template<class Fn>
void visit_float_dtype(DType dtype, Fn&& fn) {
    switch (dtype) {
        case DType::F32: return fn.template operator()<DType::F32>();
        case DType::F16: return fn.template operator()<DType::F16>();
        case DType::BF16: return fn.template operator()<DType::BF16>();
        default: throw std::invalid_argument("Unsupported NVFP4 floating-point dtype");
    }
}

template<DType Type>
using Element = std::conditional_t<Type == DType::F32, float, uint16_t>;

template<DType Type>
__device__ float to_float(Element<Type> value) {
    if constexpr (Type == DType::F32) return value;
    if constexpr (Type == DType::F16) return __half2float(__ushort_as_half(value));
    if constexpr (Type == DType::BF16) return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

template<DType Type>
__device__ Element<Type> from_float(float value) {
    if constexpr (Type == DType::F32) return value;
    if constexpr (Type == DType::F16) return __half_as_ushort(__float2half_rn(value));
    if constexpr (Type == DType::BF16) return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

template<DType Type>
__device__ void load_four(const Element<Type>* src, float (&v)[4]) {
    if (aligned(src, 4 * sizeof(*src))) {
        if constexpr (Type == DType::F32) {
            const float4 data = *reinterpret_cast<const float4*>(src);
            v[0] = data.x;
            v[1] = data.y;
            v[2] = data.z;
            v[3] = data.w;
        } else {
            const uint2 data = *reinterpret_cast<const uint2*>(src);
            v[0] = to_float<Type>(static_cast<uint16_t>(data.x));
            v[1] = to_float<Type>(static_cast<uint16_t>(data.x >> 16));
            v[2] = to_float<Type>(static_cast<uint16_t>(data.y));
            v[3] = to_float<Type>(static_cast<uint16_t>(data.y >> 16));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = to_float<Type>(src[i]);
    }
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

unsigned launch_blocks(size_t items, size_t items_per_cta, unsigned limit = 65535) {
    return static_cast<unsigned>(std::min((items - 1) / items_per_cta + 1, size_t{limit}));
}

struct ScaleLayout {
    size_t padded_rows;
    int padded_cols;

    size_t bytes() const { return padded_rows * padded_cols; }
};

ScaleLayout scale_layout(int rows, int cols) {
    if (rows < 0 || cols < 0 || cols % kScaleBlock)
        throw std::invalid_argument("NVFP4 requires nonnegative dimensions and complete 16-element blocks");
    return {round_up(rows, kTileRows), static_cast<int>(round_up(cols / kScaleBlock, kTileCols))};
}

__device__ size_t scale_offset(size_t row, int block, int padded_cols) {
    const size_t tile = row / kTileRows * (padded_cols / kTileCols) + block / kTileCols;
    return tile * (kTileRows * kTileCols) + row % 32 * 16
           + row % kTileRows / 32 * kTileCols + block % kTileCols;
}

__device__ float decode_scale(uint8_t value) {
    __nv_fp8_e4m3 scale;
    scale.__x = value;
    return static_cast<float>(scale);
}

__device__ float finite_or_zero(float value) {
    return isfinite(value) ? value : 0.0f;
}

__device__ uint8_t pack_pair(float first, float second) {
    return __nv_fp4x2_e2m1(make_float2(second, first)).__x;
}

__global__ void swizzle_scales(const uint8_t* in, uint8_t* out, int rows, int cols,
                               ScaleLayout layout) {
    const size_t stride = gridDim.x * blockDim.x;
    const size_t count = layout.padded_rows * layout.padded_cols;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const size_t row = i / layout.padded_cols;
        const int col = static_cast<int>(i % layout.padded_cols);
        out[scale_offset(row, col, layout.padded_cols)] =
            row < static_cast<size_t>(rows) && col < cols ? in[row * cols + col] : 0;
    }
}

template<DType Type>
__global__ void quantize(const Element<Type>* x, float global_scale, const float* dynamic_scale,
                         uint8_t* qdata, uint8_t* scales, int rows, int cols, int packed_stride,
                         ScaleLayout layout) {
    const float tensor_scale = dynamic_scale ? *dynamic_scale : global_scale;
    const size_t stride = gridDim.x * blockDim.x / kQuantLanes;
    const size_t count = layout.padded_rows * layout.padded_cols;
    const int lane = threadIdx.x % kQuantLanes;
    const unsigned mask = 0xfu << ((threadIdx.x & 31) - lane);
    const int num_blocks = cols / kScaleBlock;
    for (size_t i = (blockIdx.x * blockDim.x + threadIdx.x) / kQuantLanes; i < count; i += stride) {
        const size_t row = i / layout.padded_cols;
        const int block = static_cast<int>(i % layout.padded_cols);
        const size_t packed_offset = row * packed_stride + block * (kScaleBlock / 2) + lane * 2;
        uint8_t scale = 0;
        if (row < static_cast<size_t>(rows) && block < num_blocks) {
            float values[4];
            load_four<Type>(x + row * cols + block * kScaleBlock + lane * 4, values);
            float amax = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                values[j] = finite_or_zero(values[j]);
                amax = fmaxf(amax, fabsf(values[j]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1, kQuantLanes));
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2, kQuantLanes));
            scale = __nv_fp8_e4m3(fminf((amax / kE2M1Max) / tensor_scale, kE4M3Max)).__x;
            const float total_scale = decode_scale(scale) * tensor_scale;
            if (total_scale > 0.0f) {
#pragma unroll
                for (int j = 0; j < 4; ++j) values[j] /= total_scale;
                qdata[packed_offset] = pack_pair(values[0], values[1]);
                qdata[packed_offset + 1] = pack_pair(values[2], values[3]);
            } else {
                qdata[packed_offset] = 0;
                qdata[packed_offset + 1] = 0;
            }
        } else if (row < static_cast<size_t>(rows) && block < packed_stride / (kScaleBlock / 2)) {
            qdata[packed_offset] = 0;
            qdata[packed_offset + 1] = 0;
        }
        if (lane == 0) scales[scale_offset(row, block, layout.padded_cols)] = scale;
    }
}

void launch_quantize(const void* x, DType dtype, float global, const float* dynamic_scale,
                     uint8_t* qdata, uint8_t* scales, int rows, int cols, int packed_stride,
                     ScaleLayout layout, cudaStream_t stream) {
    const unsigned blocks = launch_blocks(layout.bytes(), kThreads / kQuantLanes);
    visit_float_dtype(dtype, [&]<DType Type> {
        quantize<Type><<<blocks, kThreads, 0, stream>>>(
            static_cast<const Element<Type>*>(x), global, dynamic_scale, qdata, scales,
            rows, cols, packed_stride, layout);
    });
    check_cuda(cudaGetLastError(), "NVFP4 quantize launch");
}

template<DType Type>
__global__ void dequantize(const uint8_t* qdata, const uint8_t* scales, float global,
                           Element<Type>* out, int rows, int cols, int padded_cols) {
    const int packed_cols = cols / 2;
    const size_t count = static_cast<size_t>(rows) * packed_cols;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const size_t row = i / packed_cols;
        const int block = static_cast<int>(i % packed_cols) / (kScaleBlock / 2);
        const float scale = decode_scale(scales[scale_offset(row, block, padded_cols)]);
        __nv_fp4x2_e2m1 pair;
        pair.__x = qdata[i];
        const float2 values = static_cast<float2>(pair);
        out[2 * i] = from_float<Type>((values.y * scale) * global);
        out[2 * i + 1] = from_float<Type>((values.x * scale) * global);
    }
}

__device__ float warp_max(float amax) {
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
    return amax;
}

__device__ float block_max(float amax) {
    __shared__ float warps[kThreads / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    amax = warp_max(amax);
    if (lane == 0) warps[warp] = amax;
    __syncthreads();
    if (warp == 0) amax = warp_max(lane < kThreads / 32 ? warps[lane] : 0.0f);
    return amax;
}

template<DType Type>
__global__ void reduce_amax(const Element<Type>* x, size_t count, float* partial) {
    const size_t stride = gridDim.x * blockDim.x;
    float amax = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count / 4; i += stride) {
        float v[4];
        load_four<Type>(x + i * 4, v);
#pragma unroll
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(finite_or_zero(v[j])));
    }
    amax = block_max(amax);
    if (threadIdx.x == 0) partial[blockIdx.x] = amax;
}

struct Scalars {
    float global_scale;
    float alpha;
    float beta;
};

__global__ void finish_amax(const float* partial, unsigned count, float weight_scale, Scalars* out) {
    float amax = 0.0f;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) amax = fmaxf(amax, partial[i]);
    amax = block_max(amax);
    if (threadIdx.x == 0) {
        const float scale = amax / (kE2M1Max * kE4M3Max);
        const float global = scale > 0.0f ? scale : 1.0f;
        *out = {global, global * weight_scale, 0.0f};
    }
}

// Each chunk contains one complete block of 16 FP4 weights.
__global__ void pad_weights(const uint2* weights, uint2* padded, int rows, int cols,
                            int padded_rows, int padded_cols) {
    const size_t count = static_cast<size_t>(padded_rows) * padded_cols;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const size_t row = i / padded_cols;
        const int col = static_cast<int>(i % padded_cols);
        padded[i] = row < static_cast<size_t>(rows) && col < cols ? weights[row * cols + col] : uint2{};
    }
}

template<DType Type>
__global__ void store_output(const float* src, const void* bias, DType bias_dtype,
                            Element<Type>* dst, size_t count, int cols, int ld) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const int col = static_cast<int>(i % cols);
        float value = src[(i / cols) * ld + col];
        if (bias) {
            switch (bias_dtype) {
                case DType::F32: value += static_cast<const float*>(bias)[col]; break;
                case DType::F16: value += to_float<DType::F16>(static_cast<const uint16_t*>(bias)[col]); break;
                case DType::BF16: value += to_float<DType::BF16>(static_cast<const uint16_t*>(bias)[col]); break;
                default: break;
            }
        }
        dst[i] = from_float<Type>(value);
    }
}

struct Workspace {
    size_t bytes = 0;
    size_t qx, x_scales, padded_w, partial, scalars, output, lt;
    int padded_n = 0, padded_k = 0;

    size_t reserve(size_t size) {
        constexpr size_t max = std::numeric_limits<size_t>::max();
        if (size > max - bytes - (kWorkspaceAlignment - 1))
            throw std::overflow_error("NVFP4 workspace size overflow");
        const size_t offset = bytes;
        bytes += round_up(size, kWorkspaceAlignment);
        return offset;
    }
};

Workspace workspace_layout(int m, int n, int k) {
    if (m < 0 || n < 0 || k < 0 || (m && n && k == 0))
        throw std::invalid_argument("Invalid NVFP4 linear dimensions");
    if (!m || !n) return {};
    const auto x_layout = scale_layout(m, k);
    const size_t padded_n = round_up(n, 8);
    const size_t padded_k = round_up(k, 32);
    if (std::max(padded_n, padded_k) > std::numeric_limits<int>::max())
        throw std::overflow_error("NVFP4 padded dimension overflow");

    Workspace ws{};
    ws.padded_n = static_cast<int>(padded_n);
    ws.padded_k = static_cast<int>(padded_k);
    ws.qx = ws.reserve(m * padded_k / 2);
    ws.x_scales = ws.reserve(x_layout.bytes());
    ws.padded_w = ws.reserve(padded_n != n || padded_k != k ? padded_n * padded_k / 2 : 0);
    ws.partial = ws.reserve(kReduceBlocks * sizeof(float));
    ws.scalars = ws.reserve(sizeof(Scalars));
    ws.output = ws.reserve(std::min(m, kOutputTileRows) * padded_n * sizeof(float));
    ws.lt = ws.reserve(kLtWorkspaceBytes);
    return ws;
}

class LtHandle {
public:
    LtHandle() { check_lt(cublasLtCreate(&handle), "cublasLtCreate"); }
    ~LtHandle() { cublasLtDestroy(handle); }
    LtHandle(const LtHandle&) = delete;
    LtHandle& operator=(const LtHandle&) = delete;

    cublasLtHandle_t handle{};
};

cublasLtHandle_t lt_handle_for(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, LtHandle> handles;
    const std::lock_guard lock(mutex);
    return handles.try_emplace(device).first->second.handle;
}

template<class T>
void set_attribute(cublasLtMatmulDesc_t op, cublasLtMatmulDescAttributes_t attr, const T& value) {
    check_lt(cublasLtMatmulDescSetAttribute(op, attr, &value, sizeof(value)),
             "cublasLtMatmulDescSetAttribute (NVFP4)");
}

void fp4_matmul(cublasLtHandle_t handle, const uint8_t* w, const uint8_t* w_scales,
                const uint8_t* x, const uint8_t* x_scales, const Scalars* scalars, float host_alpha,
                void* y, cudaDataType_t type, int m, int n, int k, void* workspace, cudaStream_t stream) {
    cublasLtMatmulDescOpaque_t op;
    cublasLtMatrixLayoutOpaque_t a, b, d;
    check_lt(cublasLtMatmulDescInit(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescInit (NVFP4)");
    set_attribute(&op, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
    set_attribute(&op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
    set_attribute(&op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
    set_attribute(&op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, w_scales);
    set_attribute(&op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, x_scales);
    if (scalars) set_attribute(&op, CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_DEVICE);

    // Col-major [N, M] aliases row-major [M, N]. Swap operands for the native TN path.
    check_lt(cublasLtMatrixLayoutInit(&a, CUDA_R_4F_E2M1, k, n, k), "cublasLtMatrixLayoutInit (NVFP4 weights)");
    check_lt(cublasLtMatrixLayoutInit(&b, CUDA_R_4F_E2M1, k, m, k), "cublasLtMatrixLayoutInit (NVFP4 activations)");
    check_lt(cublasLtMatrixLayoutInit(&d, type, n, m, n), "cublasLtMatrixLayoutInit (NVFP4 output)");
    const float zero = 0.0f;
    const float* alpha = scalars ? &scalars->alpha : &host_alpha;
    const float* beta = scalars ? &scalars->beta : &zero;
    check_lt(cublasLtMatmul(handle, &op, alpha, w, &a, x, &b, beta, y, &d, y, &d, nullptr,
                           workspace, kLtWorkspaceBytes, stream),
             "cublasLtMatmul (NVFP4, sm_120+)");
}

}  // namespace

bool nvfp4_device_supported(int device) {
    int count = 0, major = 0, runtime = 0, driver = 0;
    return device >= 0 && cudaGetDeviceCount(&count) == cudaSuccess && device < count
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess
        && major >= 12 && cudaRuntimeGetVersion(&runtime) == cudaSuccess && runtime >= 12080
        && cudaDriverGetVersion(&driver) == cudaSuccess && driver >= 12080
        && cublasLtGetVersion() >= 120804;
}

size_t nvfp4_swizzled_scale_bytes(int rows, int cols) {
    return scale_layout(rows, cols).bytes();
}

void nvfp4_quantize_2d(const void* x, DType x_dtype, float global_scale, uint8_t* qdata_out,
                      uint8_t* block_scales_swizzled_out, int M, int K, cudaStream_t stream) {
    const auto layout = scale_layout(M, K);
    if (!layout.bytes()) return;
    require_scale(global_scale);
    require_buffer(x, float_bytes(x_dtype), "activation");
    require_buffer(qdata_out, 1, "packed output");
    require_buffer(block_scales_swizzled_out, 1, "scale output");
    launch_quantize(x, x_dtype, global_scale, nullptr, qdata_out, block_scales_swizzled_out,
                    M, K, K / 2, layout, stream);
}

void nvfp4_swizzle_scales(const uint8_t* linear_scales, uint8_t* swizzled_scales,
                         int rows, int cols, cudaStream_t stream) {
    const auto layout = scale_layout(rows, cols);
    if (!layout.bytes()) return;
    require_buffer(linear_scales, 1, "scale input");
    require_buffer(swizzled_scales, 1, "scale output");
    swizzle_scales<<<launch_blocks(layout.bytes(), kThreads), kThreads, 0, stream>>>(
        linear_scales, swizzled_scales, rows, cols / kScaleBlock, layout);
    check_cuda(cudaGetLastError(), "NVFP4 scale swizzle launch");
}

void nvfp4_dequantize_2d(const uint8_t* qdata, const uint8_t* block_scales_swizzled,
                        float global_scale, void* out, DType out_dtype, int rows, int cols,
                        cudaStream_t stream) {
    const auto layout = scale_layout(rows, cols);
    if (!layout.bytes()) return;
    require_scale(global_scale);
    require_buffer(qdata, 1, "packed input");
    require_buffer(block_scales_swizzled, 1, "scale input");
    require_buffer(out, float_bytes(out_dtype), "dequantized output");
    const unsigned blocks = launch_blocks(static_cast<size_t>(rows) * cols / 2, kThreads);
    visit_float_dtype(out_dtype, [&]<DType Type> {
        dequantize<Type><<<blocks, kThreads, 0, stream>>>(
            qdata, block_scales_swizzled, global_scale, static_cast<Element<Type>*>(out),
            rows, cols, layout.padded_cols);
    });
    check_cuda(cudaGetLastError(), "NVFP4 dequantize launch");
}

size_t nvfp4_linear_workspace_bytes(int M, int N, int K) {
    return workspace_layout(M, N, K).bytes;
}

void nvfp4_linear(const void* x, DType x_dtype, const uint8_t* w_qdata,
                  const uint8_t* w_scales_swizzled, float w_global_scale, float x_global_scale,
                  const void* bias, DType bias_dtype, void* y, DType y_dtype,
                  int M, int N, int K, void* workspace, size_t workspace_bytes,
                  cudaStream_t stream) {
    const auto ws = workspace_layout(M, N, K);
    if (!ws.bytes) return;
    if (x_dtype != DType::F16 && x_dtype != DType::BF16)
        throw std::invalid_argument("NVFP4 linear requires F16 or BF16 input");
    require_scale(w_global_scale);
    if (!std::isfinite(x_global_scale)) throw std::invalid_argument("Invalid NVFP4 activation scale");
    require_buffer(x, float_bytes(x_dtype), "activation");
    require_buffer(w_qdata, 16, "packed weights");
    require_buffer(w_scales_swizzled, 16, "weight scales");
    const size_t output_bytes = float_bytes(y_dtype);
    require_buffer(y, output_bytes, "output");
    if (bias) require_buffer(bias, float_bytes(bias_dtype), "bias");
    require_buffer(workspace, kWorkspaceAlignment, "workspace");
    if (workspace_bytes < ws.bytes) throw std::invalid_argument("Insufficient NVFP4 workspace");

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    if (!nvfp4_device_supported(device))
        throw std::runtime_error("NVFP4 linear requires an sm_120+ GPU and FP4-capable cuBLASLt");
    const auto handle = lt_handle_for(device);

    auto* buf = static_cast<uint8_t*>(workspace);
    Scalars* scalars = nullptr;
    if (x_global_scale <= 0.0f) {
        scalars = reinterpret_cast<Scalars*>(buf + ws.scalars);
        auto* partial = reinterpret_cast<float*>(buf + ws.partial);
        const size_t count = static_cast<size_t>(M) * K;
        const unsigned blocks = launch_blocks(count / 4, kThreads, kReduceBlocks);
        visit_float_dtype(x_dtype, [&]<DType Type> {
            reduce_amax<Type><<<blocks, kThreads, 0, stream>>>(
                static_cast<const Element<Type>*>(x), count, partial);
        });
        check_cuda(cudaGetLastError(), "NVFP4 amax reduction launch");
        finish_amax<<<1, kThreads, 0, stream>>>(partial, blocks, w_global_scale, scalars);
        check_cuda(cudaGetLastError(), "NVFP4 amax finalize launch");
    }

    auto* qx = buf + ws.qx;
    auto* x_scales = buf + ws.x_scales;
    const auto x_layout = scale_layout(M, K);
    launch_quantize(x, x_dtype, x_global_scale, scalars ? &scalars->global_scale : nullptr,
                    qx, x_scales, M, K, ws.padded_k / 2, x_layout, stream);

    if (ws.padded_n != N || ws.padded_k != K) {
        const int padded_blocks = ws.padded_k / kScaleBlock;
        const unsigned blocks = launch_blocks(static_cast<size_t>(ws.padded_n) * padded_blocks, kThreads);
        pad_weights<<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const uint2*>(w_qdata), reinterpret_cast<uint2*>(buf + ws.padded_w),
            N, K / kScaleBlock, ws.padded_n, padded_blocks);
        check_cuda(cudaGetLastError(), "NVFP4 weight padding launch");
        w_qdata = buf + ws.padded_w;
    }

    const bool f32_output = y_dtype == DType::F32;
    const bool stage = N != ws.padded_n || !aligned(y, 16) || (bias && !f32_output);
    const auto type = stage || f32_output ? CUDA_R_32F : (y_dtype == DType::F16 ? CUDA_R_16F : CUDA_R_16BF);
    // Slice only at complete scale tiles so each GEMM sees a valid swizzled scale base.
    static_assert(kOutputTileRows % kTileRows == 0);
    for (size_t row = 0; row < static_cast<size_t>(M);) {
        const int remaining = M - static_cast<int>(row);
        const int rows = stage ? std::min(remaining, kOutputTileRows) : remaining;
        void* out = static_cast<uint8_t*>(y) + row * N * output_bytes;
        void* dst = stage ? buf + ws.output : out;
        fp4_matmul(handle, w_qdata, w_scales_swizzled, qx + row * (ws.padded_k / 2),
                   x_scales + row * x_layout.padded_cols, scalars, w_global_scale * x_global_scale,
                   dst, type, rows, ws.padded_n, ws.padded_k, buf + ws.lt, stream);
        if (stage || bias) {
            const size_t count = static_cast<size_t>(rows) * N;
            visit_float_dtype(y_dtype, [&]<DType Type> {
                store_output<Type><<<launch_blocks(count, kThreads), kThreads, 0, stream>>>(
                    static_cast<const float*>(dst), bias, bias_dtype, static_cast<Element<Type>*>(out),
                    count, N, ws.padded_n);
            });
            check_cuda(cudaGetLastError(), "NVFP4 output epilogue launch");
        }
        row += rows;
    }
}

}  // namespace fusionrt::cuda_ops
