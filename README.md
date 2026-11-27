# fusionrt

Experimental C++20 / CUDA / Vulkan LLM inference engine.

This is a personal research codebase for exploring custom quantized GPU kernels, paged long-context caching, and physics-guided heterogeneous execution. It works alongside [inferpeak](https://github.com/losewayy/inferpeak) and [kernprobe](https://github.com/losewayy/kernprobe).

## Current state

- **Model loading** uses memory-mapped GGUF and supports selected Safetensors models. Architectures include dense GQA (Llama 3, Qwen2/2.5/3, SmolLM2), Qwen3.5 GatedDeltaNet, Gemma 4 and sparse MoE. Qwen3-30B-A3B Q4_K_M has run on real weights with GPU-resident and CPU cold experts. Qwen3.5 MoE and Gemma 4 MoE have synthetic GGUF coverage.
- **Quantization** includes the tuned `Q4_0`, `Q8_0`, `Q2_K`, `Q4_K`, `Q5_K` and `Q6_K` paths. `WeightCodec` adds common CPU and CUDA paths for `Q3_K`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `TQ1_0`, `TQ2_0`, `Q2_0`, `PQ2_0` and `PTQ1_0`. These 13 codecs have synthetic parity coverage. The integer-basis codecs also use packed multi-token verification and on-chip INT8 MMQ. Blockwise FWHT handles rotated inputs at the layer boundary, including verified Bonsai PTQ1_0 weights.
- **Gemma 4 E4B** uses 512-dimensional heads in global layers, 256-dimensional heads in sliding-window layers, shared KV, and per-layer embeddings. The first 16 generated token IDs matched between CUDA Graph and eager mode in an E4B Q4_K_M run.
- **DeepSeek MLA** stores one FP32 latent vector and its rotary key per token per layer. Absorbed attention uses that cache directly, with CUDA Graph and session-slot support. Synthetic fixtures covering direct Q and Q-LoRA used 172,032 B against 458,752 B for expanded FP32 K/V at the same geometry.
- **CUDA engine**: Organized into three orthogonal kernel modules (`matmul/`, `sequence/`, `epilogue/`), featuring repacked 144B 4-bit tiles, INT8 MMQ prefill, O(1)-scratch online FlashAttention-2 prefill, demand-grown paged `F16`/`Q8_0` KV cache with `--slots N` copy-on-write prefix sharing, GDN turn checkpoints, pinned host RAM KV spill (`--kv-vram-mb`, `--host-kv-mb`), AVX2 multi-threaded CPU cold-expert MoE offload (`--moe-gpu-experts`), CUDA Graph decode capture, and a hardware cost model (`cost_model.h`) driven by `inferpeak` profiles.
- **Speculative decoding** supports prompt lookup (`--spec-ngram`), a vocabulary-compatible GGUF draft model (`--draft-model`), or an integrated NextN head (`--spec-mtp`). Native MTP supports up to seven draft tokens and CUDA Graph verification. Greedy parity has been checked on Bonsai 2.5 27B PTQ1_0. On one Qwen3-8B Q4_K_M greedy prompt, both matched all 64 target tokens. Decode throughput was 89.26 tok/s with standard decoding, 142.91 tok/s with a Qwen3-0.6B draft, and 176.30 tok/s with prompt lookup.
- **JSON output** uses a decoding grammar. The CLI offers `--json-grammar`. OpenAI Chat Completions and Responses support `json_object` and `json_schema` with a single root type. Schema field constraints, enabled tools, and custom stop sequences are rejected for JSON output.
- **Multimodal input** uses CUDA `--mmproj` and `--image` or base64 image blocks in HTTP requests. Supported raster formats are non-interlaced 8-bit PNG, uncompressed 24/32-bit BMP, PPM, and PGM. Gemma 4 E4B with its matching `gemma4v` projector has been checked on real weights. Other vision paths have synthetic coverage. Native Qwen-VL text M-RoPE and Qwen3-VL DeepStack remain unsupported.
- **Local HTTP** supports `/v1/completions`, OpenAI `/v1/chat/completions` and `/v1/responses`, and Anthropic `/v1/messages`. The request body limit is 16 MiB, including base64 data. Health and model discovery stay responsive during generation. Inference accepts up to eight pending requests and returns HTTP 503 when the queue is full. Health reports the configured speculative mode, and completed HTTP responses include speculative activity and acceptance counts in `fusionrt_diagnostics`. Model thinking appears in `reasoning_content`, `reasoning` items, or `thinking` blocks, respectively, on the three chat endpoints.

## Module map

- `src/io/` parses model files, tokenizer data and images. `src/core/` implements tensors, graph IR, sampling, cost modeling and CPU expert execution.
- `src/cuda/` contains matrix operations, sequence kernels and epilogues. Epilogues separate elementwise operations, Hadamard transforms, and quantized codec and MoE operations.
- `src/models/` owns weight storage and placement, sessions and KV, and Graph and IR execution in separate CUDA components. Generation coordinates them through the model interface. The directory also contains model parsing, vision encoding and the Vulkan runtime.
- `src/server/` adapts HTTP protocols. `src/cli/` and `src/bench/` provide local commands and benchmarks.

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
