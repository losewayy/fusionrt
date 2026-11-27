#include "test_suites.h"

#include <iostream>
#include <stdexcept>

using namespace fusionrt;
using namespace fusionrt::test;

int main() {
    std::cout << std::unitbuf;
    try {
        test_quant_and_io();
        test_gdn_loader_shapes();
        test_frontier_gguf();
        test_q2_k_format();
        test_universal_weight_codecs();
        test_hadamard_cpu();
        test_moe_autofit();
        test_sampling_and_benchmark_gates();
        test_agent_protocols();
        test_agent_reasoning();
        test_device_profile_and_cost_model();
#ifdef FUSIONRT_ENABLE_CUDA
        CudaContext cuda_ctx(0, 256, 512);
        if (cuda_ctx.device_profile().sm_count != cuda_ctx.info().multi_processor_count ||
            cuda_ctx.device_profile().vram_total_mb !=
                static_cast<double>(cuda_ctx.info().total_vram_bytes) / (1024.0 * 1024.0) ||
            cuda_ctx.cost_hardware().hbm_bw_gbps != cuda_ctx.info().peak_bandwidth_gbps) {
            throw std::runtime_error("CUDA device profile runtime defaults");
        }
        test_cuda_fused_rmsnorm_linear(cuda_ctx);
        test_cuda_failed_graph_recapture(cuda_ctx);
        test_cuda_q2_k(cuda_ctx);
        test_cuda_weight_codecs(cuda_ctx);
        test_cuda_codec_int8(cuda_ctx);
        test_cuda_frontier(cuda_ctx);
        test_cuda_hadamard(cuda_ctx);
        test_cuda_codec_model(cuda_ctx);
        test_cuda_step1_fast_paths(cuda_ctx);
        test_cuda_gated_deltanet_fused_vs_unfused_and_fp64(cuda_ctx);
        test_cuda_qk_rope_and_gqa_64k(cuda_ctx);
        test_cuda_graph_and_moe_offload(cuda_ctx);
        test_cuda_wave1_decode(cuda_ctx);
        test_repacked_quants(cuda_ctx);
        test_cuda_wave2_prefill(cuda_ctx);
        test_cuda_sliding_window(cuda_ctx);
        test_cuda_gemma_epilogues(cuda_ctx);
        test_cuda_gemma4_gguf();
#endif
#ifdef FUSIONRT_ENABLE_VULKAN
        test_vulkan_all_devices();
        test_vulkan_batched_vs_unbatched();
        test_vulkan_continued_turn();
#endif
#if defined(FUSIONRT_ENABLE_CUDA) && defined(FUSIONRT_ENABLE_VULKAN)
        test_phase2_loaders_prefill_and_vulkan_e2e(cuda_ctx);
#endif
        test_wave3();
        test_stage7a_json();
        test_stage7b_images_and_protocols();
        test_review_input_validation();
        test_review_gguf_validation();
        test_stage7c_mla();
#ifdef FUSIONRT_ENABLE_CUDA
        test_review_cuda_kernels(cuda_ctx);
        test_stage7a_speculative();
        test_review_speculation();
        test_stage7b_cuda_vision();
        test_gemma4v_vision();
        test_chunked_prefill_and_k_quant_gdn();
        test_cuda_session_slots();
        test_tiered_cuda_kv();
        test_cuda_native_quantized_moe();
        test_cpu_moe_q8_dots();
        test_cpu_moe_multirow();
        test_long_context_kv_admission();
        test_dense_auto_kv_fallback();
        test_cpu_quantized_moe_speed();
#endif
        std::cout << "\nALL TESTS PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
