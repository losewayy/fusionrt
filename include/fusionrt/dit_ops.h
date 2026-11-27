#pragma once

#include "fusionrt/cost_model.h"

#include <cuda_runtime_api.h>

namespace fusionrt::cuda_ops {

bool dit_supports_weight_dtype(DType w_dtype);

// Activations, bias and output support F32, F16 and BF16. The workspace
// holds at most one row tile's intermediates and is independent of large M.
size_t dit_linear_workspace_bytes(int M, int N, int K, DType w_dtype, DType x_dtype, DType y_dtype);

// Writes M * (K / 32) blocks to a 16-byte aligned BlockQ8_1 buffer.
// NaNs and infinities in x are treated as zero.
void dit_quantize_x_q8_1(const void* x, DType x_dtype, void* q8_out, int M, int K,
                        cudaStream_t stream);

// Adds the optional bias across rows before rounding to dst_dtype.
// src_f32 and dst may alias for F32 output.
void dit_f32_to_out_bias(const float* src_f32, const void* bias, DType bias_dtype,
                         void* dst, DType dst_dtype, int M, int N, cudaStream_t stream);

void dit_cast_to_f32(const void* src, DType src_dtype, float* dst, size_t count, cudaStream_t stream);

// Enqueues y[M, N] = x[M, K] * transpose(w[N, K]) + bias[N] on stream.
// All buffers are contiguous device storage, valid through stream completion.
// Supply 16-byte aligned weights and workspace, with disjoint output and
// workspace. Bias is optional, and workspace may be null if its size is zero.
// M or N equal to zero is a no-op. Nonempty calls require whole weight blocks.
void dit_linear_quantized(const void* x, DType x_dtype, const void* w, DType w_dtype,
                          const void* bias, DType bias_dtype, void* y, DType y_dtype,
                          int M, int N, int K, void* workspace, size_t workspace_bytes,
                          cudaStream_t stream, const CostHardware& hw = {});

}  // namespace fusionrt::cuda_ops
