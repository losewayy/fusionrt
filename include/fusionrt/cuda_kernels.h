#pragma once

#include "fusionrt/cuda_backend.h"
#include "fusionrt/tensor.h"

namespace fusionrt::cuda_ops {

// -----------------------------------------------------------------------------
// Unfused Baseline Kernels (for differential & ablation benchmarking)
// -----------------------------------------------------------------------------
void rmsnorm_f32(const float* x, const float* weight, float* y,
                 int rows, int cols, float eps, cudaStream_t stream);

void rmsnorm_f16(const uint16_t* x, const uint16_t* weight, uint16_t* y,
                 int rows, int cols, float eps, cudaStream_t stream);

void linear_f16(const uint16_t* x, const uint16_t* w, uint16_t* y,
                int M, int N, int K, cudaStream_t stream);

void linear_f32(const float* x, const float* w, float* y,
                int M, int N, int K, cudaStream_t stream);

// Mixed-precision linear: F32 activations, F16/BF16 weights -> F32 outputs (supports M >= 1)
void linear_w_f16_f32(const float* x, const uint16_t* w, float* y,
                      int M, int N, int K, cudaStream_t stream);

void linear_w_bf16_f32(const float* x, const uint16_t* w, float* y,
                       int M, int N, int K, cudaStream_t stream);

void linear_q8_0_unfused(const float* x, const void* w_q8, float* y,
                         int M, int N, int K, void* scratch_f32_w, cudaStream_t stream);

void linear_q4_0_unfused(const float* x, const void* w_q4, float* y,
                         int M, int N, int K, void* scratch_f32_w, cudaStream_t stream);

void rope_f32(float* x, int num_heads, int head_dim, int rot_dim, int pos,
              float theta, bool neox, cudaStream_t stream);

void kv_append_f32(const float* k_in, const float* v_in,
                   float* k_cache, float* v_cache,
                   int num_kv_heads, int head_dim, int pos, int max_seq,
                   cudaStream_t stream);

void swiglu_f32(const float* gate, const float* up, float* out, int n, cudaStream_t stream);
void swiglu_interleaved_f32(const float* gate_up, float* out, int rows, int inter_dim, cudaStream_t stream);
// Q8_1 buffers must be 16-byte aligned, with one block per 32 columns.
void quantize_x_q8_1(const float* x, const float* res_b, const float* norm_w, float* res_out,
                     void* q8_out, int M, int K, float eps, cudaStream_t stream);
void swiglu_interleaved_q8_1(const float* gate_up, void* q8_out, int M, int inter_dim, cudaStream_t stream);
void gemv_q4_a_q8_1(const void* x_q8, const void* w_q4_a, float* y, int N, int K, cudaStream_t stream,
                   GemvGeometry geometry = {});
void gemv_q8_a_q8_1(const void* x_q8, const void* w_q8_a, float* y, int N, int K, cudaStream_t stream,
                   GemvGeometry geometry = {});
void gemv_q6_k_q8_1(const void* x_q8, const void* w_q6_k, float* y, int N, int K, cudaStream_t stream,
                   GemvGeometry geometry = {});
void gemv_q4_k_q8_1(const void* x_q8, const void* w_q4_k, float* y, int N, int K, cudaStream_t stream,
                   GemvGeometry geometry = {});
void gemv_q4_ka_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                     GemvGeometry geometry = {});
void gemv_q6_a_q8_1(const void* x, const void* w, float* y, int N, int K, cudaStream_t stream,
                     GemvGeometry geometry = {});
void gemv_q5_k_q8_1(const void* x_q8, const void* w_q5_k, float* y, int N, int K, cudaStream_t stream,
                   GemvGeometry geometry = {});

bool project_q4_a_and_q8_a(const void* x, const void* q4, const void* q8, float* y4, float* y8,
                          int M, int N4, int N8, int K, cudaStream_t stream, GemvGeometry geometry);

void mmq_q4_a_q8_1(const void* x_q8, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void mmq_q8_a_q8_1(const void* x_q8, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void mmq_q6_k_q8_1(const void* x_q8, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void mmq_q4_k_q8_1(const void* x_q8, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void mmq_q4_ka_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void mmq_q6_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void mmq_q5_k_q8_1(const void* x_q8, const void* w, float* y, int M, int N, int K, cudaStream_t stream);
void add_residual_f32(const float* a, const float* b, float* out, int n, cudaStream_t stream);
void add_bias_f32(float* out, const float* bias, int rows, int cols, int stride, cudaStream_t stream);
void add_rmsnorm_f32(const float* a, const float* b, const float* norm_w,
                     float* res_out, float* norm_out, int rows, int cols, float eps, cudaStream_t stream);
void fused_add_rmsnorm_linear_q4_a(const float* a, const float* b, const float* norm_w,
                                   const void* weight_q4_a, float* res_out, float* y,
                                   int N, int K, float eps, cudaStream_t stream);
void fused_add_rmsnorm_linear_q8_a(const float* a, const float* b, const float* norm_w,
                                   const void* weight_q8_a, float* res_out, float* y,
                                   int N, int K, float eps, cudaStream_t stream);
void fused_add_rmsnorm_linear_q6_k(const float* a, const float* b, const float* norm_w,
                                   const void* weight_q6_k, float* res_out, float* y,
                                   int N, int K, float eps, cudaStream_t stream);
void split_q_gate_f32(const float* q_full, float* q_out, float* gate_out, int M, int num_q_heads, int head_dim, cudaStream_t stream, int src_row_stride = 0);
void apply_sigmoid_gate_f32(float* attn_out, const float* gate, int n, cudaStream_t stream);

inline constexpr int kArgmaxParts = 128;
inline constexpr size_t kArgmaxScratchBytes = kArgmaxParts * (sizeof(float) + sizeof(int32_t));

// When pos is supplied, out_idx[0] is the next input token and the optional
// history occupies out_idx[1..history_size], indexed by the current position.
void argmax_f32(const float* x, int n, int32_t* out_idx, cudaStream_t stream,
                 void* scratch = nullptr, int* pos = nullptr, int* kv_len = nullptr, int history_size = 0);

// GPU embedding lookup supporting F32, F16, BF16, Q8_0, Q4_0 tables -> F32 [M, hidden]
void embed_lookup_f32(const void* emb_table, DType table_dtype,
                      const int32_t* token_ids_dev, float* out,
                      int M, int hidden, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Fused Hardware-Focused Kernels (sm_120)
// -----------------------------------------------------------------------------

// Fusion 1: Fused Pre-Norm (RMSNorm) + Linear GEMV/GEMM (supports arbitrary M >= 1, F32/F16/BF16/Q8_0/Q4_0)
void fused_rmsnorm_linear_f32(const float* x, const float* norm_w, const float* weight,
                              float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_f16(const float* x, const float* norm_w, const uint16_t* weight_f16,
                              float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_bf16(const float* x, const float* norm_w, const uint16_t* weight_bf16,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q8_0(const float* x, const float* norm_w, const void* weight_q8,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q4_0(const float* x, const float* norm_w, const void* weight_q4,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q4_1(const float* x, const float* norm_w, const void* weight_q4_1,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q4_k(const float* x, const float* norm_w, const void* weight_q4_k,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q6_k(const float* x, const float* norm_w, const void* weight_q6_k,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q4_a(const float* x, const float* norm_w, const void* weight_q4_a,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

void fused_rmsnorm_linear_q8_a(const float* x, const float* norm_w, const void* weight_q8_a,
                               float* y, int M, int N, int K, float eps, cudaStream_t stream);

// GatedDeltaNet decode scratch (when supplied): 2*nk*dk + 2*nv*(dv+1) floats.
void fused_gated_deltanet_step_f32(
    const float* qkv_proj,
    const float* z,
    const float* a,
    const float* b,
    const float* conv_w,
    const float* conv_bias,
    const float* A_log,
    const float* dt_bias,
    const float* norm_w,
    float* conv_state,
    float* ssm_state,
    float* out,
    int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, cudaStream_t stream, float* scratch = nullptr, bool tiled_value_heads = false);

// Allocate this many floats to enable chunkwise prefill.
size_t gated_deltanet_prefill_scratch_floats(int M, int nk, int nv, int dk, int dv);
void fused_gated_deltanet_prefill_f32(
    const float* qkv_proj, int qkv_stride,
    const float* z, int z_stride,
    const float* a, int a_stride,
    const float* b, int b_stride,
    const float* conv_w,
    const float* conv_bias,
    const float* A_log,
    const float* dt_bias,
    const float* norm_w,
    float* conv_state,
    float* ssm_state,
    float* out,
    int M, int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, cudaStream_t stream, float* scratch = nullptr, size_t scratch_floats = 0,
    bool tiled_value_heads = false);

void unfused_gated_deltanet_step_f32(
    const float* qkv_proj, const float* z, const float* a, const float* b,
    const float* conv_w, const float* conv_bias, const float* A_log, const float* dt_bias,
    const float* norm_w, float* conv_state, float* ssm_state, float* out,
    float* scratch_qkv, float* scratch_y,
    int num_k_heads, int num_v_heads, int dk, int dv,
    float eps, float q_scale, cudaStream_t stream, bool tiled_value_heads = false);

// Fusion 3: Fused Q/K RMSNorm (optional) + RoPE (full or partial) + KV Cache Append
void fused_qk_norm_rope_kv_append_f32(
    float* q,
    float* k,
    const float* v,
    const float* q_norm_w,
    const float* k_norm_w,
    void* k_cache,
    void* v_cache,
    int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    const int* pos_dev, int pos_host, int max_seq,
    float theta, float eps, bool neox, cudaStream_t stream, const float* rotary_factors = nullptr,
    DType kv_dtype = DType::F32);

// Batched Q/K Norm + RoPE + KV Cache Append for M > 1 prompt prefill
void batched_qk_norm_rope_kv_append_f32(
    float* q,                // [M, num_q_heads, head_dim]
    float* k,                // [M, num_kv_heads, head_dim]
    const float* v,          // [M, num_kv_heads, head_dim]
    const float* q_norm_w,   // [head_dim] (nullable)
    const float* k_norm_w,   // [head_dim] (nullable)
    void* k_cache,           // [max_seq, num_kv_heads, head_dim]
    void* v_cache,           // [max_seq, num_kv_heads, head_dim]
    int M, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim,
    int start_pos, int max_seq, float theta, float eps, bool neox, cudaStream_t stream,
    int q_row_stride = 0, int k_row_stride = 0, int v_row_stride = 0,
    const float* rotary_factors = nullptr, DType kv_dtype = DType::F32);

// Fusion 4: Fused W4/Q8 Dequantize-GEMV/GEMM (supports arbitrary M >= 1)
void fused_gemv_q8_0(const float* x, const void* w_q8, float* y,
                     int M, int N, int K, cudaStream_t stream);

void fused_gemv_q4_0(const float* x, const void* w_q4, float* y,
                     int M, int N, int K, cudaStream_t stream);

void linear_repacked_k_f32(const float* x, const void* w, DType dtype, float* y,
                           int M, int N, int K, cudaStream_t stream);
void fused_gemv_q4_1(const float* x, const void* w_q4_1, float* y,
                     int M, int N, int K, cudaStream_t stream);

void fused_gemv_q4_k(const float* x, const void* w_q4_k, float* y,
                     int M, int N, int K, cudaStream_t stream);

void fused_gemv_q5_k(const float* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream);

void fused_gemv_q6_k(const float* x, const void* w_q6_k, float* y,
                     int M, int N, int K, cudaStream_t stream);

void fused_gemv_q4_a(const float* x, const void* w_q4_a, float* y,
                     int M, int N, int K, cudaStream_t stream);

void fused_gemv_q8_a(const float* x, const void* w_q8_a, float* y,
                     int M, int N, int K, cudaStream_t stream);

// Prefill cuBLAS FP16 Tensor Core helpers (M >= 16)
void dequantize_to_f16(const void* w, DType dt, uint16_t* out_f16, int N, int K, cudaStream_t stream);
void rmsnorm_or_copy_to_f16(const float* x, const float* norm_w, uint16_t* out_f16,
                            int M, int K, float eps, cudaStream_t stream);
void cublas_linear_f16_f32(cublasHandle_t handle, const uint16_t* x_f16, const uint16_t* w_f16,
                           float* y, int M, int N, int K);

// GQA Flash / Split-K Decode (supports up to 64K seq_len).
// A Q8_1 output uses at least two splits and a head_dim divisible by 32.
void gqa_splitk_decode_f32(
    const float* q,
    const void* k_cache,
    const void* v_cache,
    float* out,
    float* partial_out,
    float* partial_ml,
    int num_q_heads, int num_kv_heads, int head_dim,
    const int* kv_len_dev, int kv_len_host, int max_seq,
    float scale, int max_splits, cudaStream_t stream, bool fixed_splits = false,
    DType kv_dtype = DType::F32, BlockQ8_1* q8_out = nullptr, const float* gate = nullptr);

// Batched Causal GQA Prefill Attention (M queries at start_pos .. start_pos + M - 1)
void gqa_prefill_causal_f32(
    const float* q,          // [M, num_q_heads, head_dim]
    const void* k_cache,      // [max_seq, num_kv_heads, head_dim]
    const void* v_cache,      // [max_seq, num_kv_heads, head_dim]
    float* out,              // [M, num_q_heads, head_dim]
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq, float scale, cudaStream_t stream,
    int q_row_stride = 0, DType kv_dtype = DType::F32);

// Tensor Core prefill for a long FP16 KV cache. Q and the output remain FP32.
size_t gqa_f16_prefill_workspace(int rows, int q_heads, int head_dim, int kv_len);
void gqa_f16_prefill(cublasHandle_t handle, const float* q, int q_stride,
                     const void* keys, const void* values, float* out, int rows,
                     int q_heads, int kv_heads, int head_dim, int start_pos, float scale,
                     void* workspace, cudaStream_t stream);

// Sparse MoE helper kernels
void moe_topk_router_f32(const float* logits, int* topk_idx, float* topk_weight,
                         int num_experts, int top_k, bool norm_topk, cudaStream_t stream);

void moe_indexed_swiglu_gemv_f32(
    const float* x,
    const int* topk_idx,
    const float* gate_up_w,
    const float* down_w,
    const float* topk_weight,
    float* out,
    float* scratch_inter,
    int hidden, int inter, int top_k, cudaStream_t stream);

void update_pos_and_kvlen_f32(int* pos_dev, int* kv_len_dev, int new_pos, int new_kv_len, cudaStream_t stream);

}  // namespace fusionrt::cuda_ops
