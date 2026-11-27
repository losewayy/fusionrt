#pragma once

#include "fusionrt/types.h"

#include <cuda_runtime_api.h>

namespace fusionrt::cuda_ops {

bool nvfp4_device_supported(int device = 0);

// cols is the number of unpacked elements and must be divisible by 16.
// One UE4M3 byte per 16 elements, padded to 128 rows and four scale columns
// and stored in cuBLASLt VEC16_UE4M3 tile order.
size_t nvfp4_swizzled_scale_bytes(int rows, int cols);

// All pointers refer to contiguous device buffers on the current device.
// Keep buffers alive and disjoint through completion on stream.
// E2M1 pairs are high-nibble first, as in comfy-kitchen's default layout.
// global_scale is a positive finite dequantization multiplier. Quantization
// accepts F32, F16 and BF16 and treats nonfinite input elements as zero.
void nvfp4_quantize_2d(const void* x, DType x_dtype, float global_scale, uint8_t* qdata_out,
                      uint8_t* block_scales_swizzled_out, int M, int K, cudaStream_t stream);

// Input is [rows, cols / 16]. Output includes zeroed padding.
void nvfp4_swizzle_scales(const uint8_t* linear_scales, uint8_t* swizzled_scales,
                         int rows, int cols, cudaStream_t stream);

// qdata is [rows, cols / 2]. Output is [rows, cols] in F32, F16 or BF16.
void nvfp4_dequantize_2d(const uint8_t* qdata, const uint8_t* block_scales_swizzled,
                        float global_scale, void* out, DType out_dtype, int rows, int cols,
                        cudaStream_t stream);

// Includes packed activations, cuBLASLt scratch and up to one FP32 output tile.
size_t nvfp4_linear_workspace_bytes(int M, int N, int K);

// Computes y[M, N] = x[M, K] * transpose(w[N, K]) + bias[N].
// x is F16 or BF16. Bias (optional) and y support F32, F16 and BF16.
// Weight data and scales use the same layout as nvfp4_quantize_2d.
// Weight buffers must be 16-byte aligned and workspace 256-byte aligned.
// w_global_scale and positive x_global_scale are finite multipliers.
// A finite x_global_scale <= 0 selects amax-based scaling on stream.
// Empty output dimensions are a no-op. Nonempty calls require K % 16 == 0.
void nvfp4_linear(const void* x, DType x_dtype, const uint8_t* w_qdata,
                  const uint8_t* w_scales_swizzled, float w_global_scale, float x_global_scale,
                  const void* bias, DType bias_dtype, void* y, DType y_dtype,
                  int M, int N, int K, void* workspace, size_t workspace_bytes,
                  cudaStream_t stream);

}  // namespace fusionrt::cuda_ops
