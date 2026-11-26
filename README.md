# fusionrt

Experimental C++20 / CUDA / Vulkan LLM inference engine.

This is a personal research codebase for exploring custom quantized GPU kernels, paged long-context caching, and physics-guided heterogeneous execution. It works alongside [inferpeak](https://github.com/losewayy/inferpeak) and [kernprobe](https://github.com/losewayy/kernprobe).

## Current state

- **Model & quant support**: Loads GGUF and Safetensors weights for Dense GQA (Llama 3, Qwen2/2.5/3, SmolLM2), Hybrid GatedDeltaNet (Qwen3.5), Gemma 4 text layers (SWA, shared KV, GeGLU, post-norm, softcapping), and quantized Sparse MoE (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `F16`, `BF16`). Unsupported architectures fail explicitly on load.
- **CUDA engine**: Organized into three orthogonal kernel modules (`matmul/`, `sequence/`, `epilogue/`), featuring repacked 144B 4-bit tiles, INT8 MMQ prefill, O(1)-scratch online FlashAttention-2 prefill, demand-grown paged `F16`/`Q8_0` KV cache with `--slots N` copy-on-write prefix sharing, GDN turn checkpoints, pinned host RAM KV spill (`--kv-vram-mb`, `--host-kv-mb`), AVX2 multi-threaded CPU cold-expert MoE offload (`--moe-gpu-experts`), CUDA Graph decode capture, and a hardware cost model (`cost_model.h`) driven by `inferpeak` profiles.
- **Server**: Local HTTP `serve` supporting `/v1/completions`, OpenAI `/v1/chat/completions`, OpenAI `/v1/responses`, and Anthropic `/v1/messages` tool-calling APIs.

## Backend scope & hardware matrix

- **CUDA (`sm_80`–`sm_120`)**: Primary backend. Local builds default to `-DCMAKE_CUDA_ARCHITECTURES=native` so compilation targets the host GPU cleanly and quickly. Multi-architecture builds use `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"`. Runtime accuracy and performance are verified on RTX 5070 Ti (`sm_120`); `sm_80/86/89/90` are compile-checked in CI and have not yet been runtime-benchmarked on physical hardware.
- **Vulkan**: Retained as an experimental cross-vendor / iGPU baseline. It does not track CUDA's long-context FA2, quantized MMQ, multi-slot KV sharing, or heterogeneous MoE paths.

## Build

Requires CMake 3.24+ and a C++20 compiler (CUDA and Vulkan are optional; default CUDA arch is `native`, override via `-DCMAKE_CUDA_ARCHITECTURES=...`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fusionrt_tests
```

Portable multi-architecture CUDA build (Ampere through Blackwell):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"
cmake --build build -j
```

```bash
./build/fusionrt inspect --model /path/to/model.gguf
./build/fusionrt serve --model /path/to/model.gguf --kv-type q8_0 --slots 2 --max-seq 32768
```

## License

MIT ([LICENSE](LICENSE))
