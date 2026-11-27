#pragma once

#include "fusionrt/model_types.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include "fusionrt/cuda_kernels.h"

namespace fusionrt {

using DecodeGemv = decltype(&cuda_ops::gemv_q4_a_q8_1);
using PrefillMmq = decltype(&cuda_ops::mmq_q4_a_q8_1);
DecodeGemv decode_gemv(DType dtype);
PrefillMmq prefill_mmq(DType dtype);
bool use_codec_mmq(const CudaContext& ctx, int rows, bool allow_prefill);
void dispatch_linear(CudaContext& ctx, const float* x, const Tensor& w, float* y, int M, int N, int K,
                     bool use_fused, bool verify = false);
void dispatch_rmsnorm_linear(CudaContext& ctx, const float* x, const Tensor& norm_w, const Tensor& w,
                             float* y, int M, int N, int K, float eps, bool use_fused, bool verify = false);
void dispatch_add_rmsnorm_linear(CudaContext& ctx, const float* a, const float* b,
                                 const Tensor& norm_w, const Tensor& w,
                                 float* res_out, float* y, int M, int N, int K, float eps, bool use_fused, bool verify = false);

inline Tensor cuda_scratch(CudaContext& ctx, std::initializer_list<int64_t> shape, DType dtype = DType::F32) {
    Tensor t = Tensor::make_contiguous(shape, dtype, Backend::CUDA);
    t.data = ctx.scratch_arena().alloc(t.nbytes());
    return t;
}

namespace model_io {
void vision_linear_cuda(CudaContext& ctx, const Tensor& input, const Tensor& weight, Tensor& output, int rows);
}  // namespace model_io

}  // namespace fusionrt
#endif
