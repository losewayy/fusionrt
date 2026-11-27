#pragma once

#include "fusionrt/models.h"

namespace fusionrt::test {
void test_wave1_inputs();
#ifdef FUSIONRT_ENABLE_CUDA
void test_wave1_cuda(CudaContext&);
void test_wave1_http(CudaContext&);
#endif

void test_quant_and_io();
void test_gdn_loader_shapes();
void test_frontier_gguf();
void test_q2_k_format();
void test_universal_weight_codecs();
void test_hadamard_cpu();
void test_moe_autofit();
void test_sampling_and_benchmark_gates();
void test_agent_protocols();
void test_agent_reasoning();
void test_device_profile_and_cost_model();
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_fused_rmsnorm_linear(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_failed_graph_recapture(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_q2_k(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_weight_codecs(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_codec_int8(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_frontier(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_hadamard(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_codec_model(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_step1_fast_paths(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_gated_deltanet_fused_vs_unfused_and_fp64(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_qk_rope_and_gqa_64k(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_graph_and_moe_offload(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_wave1_decode(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_repacked_quants(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_wave2_prefill(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_sliding_window(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_gemma_epilogues(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_gemma4_gguf();
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
void test_vulkan_all_devices();
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
void test_vulkan_batched_vs_unbatched();
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
void test_vulkan_continued_turn();
#endif
#if defined(FUSIONRT_ENABLE_CUDA) && defined(FUSIONRT_ENABLE_VULKAN)
void test_phase2_loaders_prefill_and_vulkan_e2e(CudaContext& cuda_ctx);
#endif
void test_wave3();
void test_stage7a_json();
void test_stage7b_images_and_protocols();
void test_review_input_validation();
void test_review_gguf_validation();
void test_stage7c_mla();
#ifdef FUSIONRT_ENABLE_CUDA
void test_review_cuda_kernels(CudaContext& ctx);
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_stage7a_speculative();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_review_speculation();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_stage7b_cuda_vision();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_gemma4v_vision();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_chunked_prefill_and_k_quant_gdn();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_session_slots();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_tiered_cuda_kv();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_native_quantized_moe();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cpu_moe_q8_dots();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cpu_moe_multirow();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_long_context_kv_admission();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_dense_auto_kv_fallback();
#endif
#ifdef FUSIONRT_ENABLE_CUDA
void test_cpu_quantized_moe_speed();
#endif

}  // namespace fusionrt::test
