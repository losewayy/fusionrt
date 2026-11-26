# fusionrt

Experimental C++20 / CUDA / Vulkan LLM inference engine.

This is an early personal research codebase for exploring simple graph-level operator fusion and custom quantized GPU kernels. It works alongside [inferpeak](https://github.com/losewayy/inferpeak) and [kernprobe](https://github.com/losewayy/kernprobe).

## Current state

- Loads GGUF and Safetensors weights; currently tested on a handful of Llama 3 and Qwen2/2.5/3/3.5 models (Dense GQA and GatedDeltaNet) in common quantization formats (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `F16`, `BF16`).
- Small graph IR for fusing adjacent ops, CUDA Graph decode capture, optional FP16 KV cache (`--kv-type f16`), and basic `inspect`, `chat`, `bench-model`, and local HTTP `serve` commands.
- Mostly tested on a single Windows CUDA (`sm_120`) setup. Unsupported architectures and operators fail explicitly on load.

## Build

Requires CMake 3.24+ and a C++20 compiler (CUDA and Vulkan are optional; default CUDA arch is `sm_120`, override via `-DCMAKE_CUDA_ARCHITECTURES=...`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fusionrt_tests
```

```bash
./build/fusionrt inspect --model /path/to/model.gguf
./build/fusionrt chat --model /path/to/model.gguf --kv-type f16
```

## License

MIT ([LICENSE](LICENSE))
