# fusionrt

Experimental C++20 / CUDA / Vulkan LLM inference engine.

This is a personal research codebase for exploring custom quantized GPU kernels, paged long-context caching, and physics-guided heterogeneous execution. It works alongside [inferpeak](https://github.com/losewayy/inferpeak) and [kernprobe](https://github.com/losewayy/kernprobe).

## Current state

- **Model and quant support** includes dense GQA (Llama 3, Qwen2/2.5/3, SmolLM2), hybrid GatedDeltaNet (Qwen3.5), Gemma 4 text, sparse MoE, and DeepSeek MLA. The engine reads GGUF tensors through memory mapping and also loads selected Safetensors models. Quantized MoE supports `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `F16`, and `BF16`. A Qwen3-30B-A3B Q4_K_M run on a 12 GB RTX 5070 Ti Laptop used 67 GPU and 61 CPU experts per layer and decoded at 8.71 tok/s in a 32-token run.
- **Gemma 4 E4B** uses 512-dimensional heads in global layers, 256-dimensional heads in sliding-window layers, shared KV, and per-layer embeddings. The first 16 generated token IDs matched between CUDA Graph and eager mode in an E4B Q4_K_M run.
- **DeepSeek MLA** stores one FP32 latent vector and its rotary key per token per layer. Absorbed attention uses that cache directly, with CUDA Graph and session-slot support. Synthetic fixtures covering direct Q and Q-LoRA used 172,032 B against 458,752 B for expanded FP32 K/V at the same geometry.
- **CUDA engine**: Organized into three orthogonal kernel modules (`matmul/`, `sequence/`, `epilogue/`), featuring repacked 144B 4-bit tiles, INT8 MMQ prefill, O(1)-scratch online FlashAttention-2 prefill, demand-grown paged `F16`/`Q8_0` KV cache with `--slots N` copy-on-write prefix sharing, GDN turn checkpoints, pinned host RAM KV spill (`--kv-vram-mb`, `--host-kv-mb`), AVX2 multi-threaded CPU cold-expert MoE offload (`--moe-gpu-experts`), CUDA Graph decode capture, and a hardware cost model (`cost_model.h`) driven by `inferpeak` profiles.
- **Speculative decoding** supports prompt lookup (`--spec-ngram`) or a vocabulary-compatible GGUF draft model (`--draft-model`, `--draft-tokens`). On one Qwen3-8B Q4_K_M greedy prompt, both matched all 64 target tokens. Decode throughput was 89.26 tok/s with standard decoding, 142.91 tok/s with a Qwen3-0.6B draft, and 176.30 tok/s with prompt lookup.
- **JSON output** uses a decoding grammar. The CLI offers `--json-grammar`. OpenAI Chat Completions and Responses support `json_object` and `json_schema` with a single root type. Schema field constraints, enabled tools, and custom stop sequences are rejected for JSON output.
- **Multimodal input** uses CUDA `--mmproj` and `--image` or base64 image blocks in HTTP requests. Supported raster formats are non-interlaced 8-bit PNG, uncompressed 24/32-bit BMP, PPM, and PGM. The vision path has synthetic end-to-end coverage. Native Qwen-VL text M-RoPE and Qwen3-VL DeepStack remain unsupported.
- **Local HTTP** supports `/v1/completions`, OpenAI `/v1/chat/completions` and `/v1/responses`, and Anthropic `/v1/messages`. The request body limit is 16 MiB, including base64 data. Model thinking appears in `reasoning_content`, `reasoning` items, or `thinking` blocks, respectively, on the three chat endpoints.

## Backend scope & hardware matrix

- **CUDA (`sm_80`–`sm_120`)** is the primary backend. Local builds default to `-DCMAKE_CUDA_ARCHITECTURES=native`. Multi-architecture builds use `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"`. The runtime measurements above are from an RTX 5070 Ti Laptop (`sm_120`).
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
