#include "cpu_moe.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#if defined(FUSIONRT_CPU_MOE_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace fusionrt {
namespace {

class ExpertWorkers {
public:
    ExpertWorkers() {
        const unsigned count = std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
        for (unsigned i = 1; i < count; ++i) {
            workers_.emplace_back([this, i, count](std::stop_token stop) {
                unsigned seen = 0;
                std::unique_lock lock(mutex_);
                while (ready_.wait(lock, stop, [&] { return generation_ != seen; })) {
                    seen = generation_;
                    const auto* job = job_;
                    const int rows = rows_;
                    lock.unlock();
                    (*job)(static_cast<int>(i * rows / count), static_cast<int>((i + 1) * rows / count));
                    lock.lock();
                    if (--remaining_ == 0) done_.notify_one();
                }
            });
        }
    }

    void run(int rows, const std::function<void(int, int)>& job) {
        std::lock_guard dispatch(dispatch_);
        {
            std::lock_guard lock(mutex_);
            rows_ = rows;
            job_ = &job;
            remaining_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        job(0, rows / static_cast<int>(workers_.size() + 1));
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }

private:
    std::mutex dispatch_, mutex_;
    std::condition_variable_any ready_;
    std::condition_variable done_;
    const std::function<void(int, int)>* job_ = nullptr;
    size_t remaining_ = 0;
    int rows_ = 0;
    unsigned generation_ = 0;
    std::vector<std::jthread> workers_;
};

template <DType Type>
float scalar_dense_dot(const uint8_t* row, const float* input, int cols) {
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float weight;
        if constexpr (Type == DType::F32) weight = reinterpret_cast<const float*>(row)[i];
        else if constexpr (Type == DType::F16) weight = fp16_to_fp32(reinterpret_cast<const uint16_t*>(row)[i]);
        else weight = bf16_to_fp32(reinterpret_cast<const uint16_t*>(row)[i]);
        sum += input[i] * weight;
    }
    return sum;
}

template <typename Block, int Size, void (*Dequantize)(const Block*, float*, int64_t)>
float scalar_quantized_dot(const uint8_t* row, const float* input, int cols) {
    const auto* blocks = reinterpret_cast<const Block*>(row);
    float sum = 0.0f;
    for (int i = 0; i < cols; i += Size) {
        float values[Size];
        Dequantize(blocks + i / Size, values, Size);
        for (int j = 0; j < Size; ++j) sum += input[i + j] * values[j];
    }
    return sum;
}

#ifdef FUSIONRT_CPU_MOE_AVX2
bool has_avx2_fma() {
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuid(regs, 1);
    constexpr int mask = (1 << 12) | (1 << 27) | (1 << 28);  // FMA, OSXSAVE, AVX
    if ((regs[2] & mask) != mask || (_xgetbv(0) & 6) != 6) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif

CpuMoeDot row_dot(DType type, CpuMoeMode mode) {
#ifdef FUSIONRT_CPU_MOE_AVX2
    static const bool avx2_fma = has_avx2_fma();
    if (mode == CpuMoeMode::Parallel && avx2_fma) {
        if (const auto dot = cpu_moe_avx2_dot(type)) return dot;
    }
#else
    (void)mode;
#endif
    switch (type) {
        case DType::F32: return scalar_dense_dot<DType::F32>;
        case DType::F16: return scalar_dense_dot<DType::F16>;
        case DType::BF16: return scalar_dense_dot<DType::BF16>;
        case DType::Q4_0: return scalar_quantized_dot<BlockQ4_0, 32, dequantize_row_q4_0>;
        case DType::Q4_1: return scalar_quantized_dot<BlockQ4_1, 32, dequantize_row_q4_1>;
        case DType::Q8_0: return scalar_quantized_dot<BlockQ8_0, 32, dequantize_row_q8_0>;
        case DType::Q4_K: return scalar_quantized_dot<BlockQ4_K, 256, dequantize_row_q4_k>;
        case DType::Q5_K: return scalar_quantized_dot<BlockQ5_K, 256, dequantize_row_q5_k>;
        case DType::Q6_K: return scalar_quantized_dot<BlockQ6_K, 256, dequantize_row_q6_k>;
        default: throw std::invalid_argument("Unsupported CPU expert weight dtype");
    }
}

}  // namespace

void cpu_moe_accumulate(const HostTensor& gate, const HostTensor& up, const HostTensor& down,
                        int expert, const float* input, float weight, float* output, CpuMoeMode mode) {
    const int hidden = static_cast<int>(down.desc.shape[1]);
    const int inter = static_cast<int>(down.desc.shape[2]);
    const bool packed = up.bytes().empty();
    const size_t gate_stride = row_nbytes(gate.desc.dtype, hidden);
    const size_t up_stride = packed ? gate_stride : row_nbytes(up.desc.dtype, hidden);
    const size_t down_stride = row_nbytes(down.desc.dtype, inter);
    const size_t first_row = static_cast<size_t>(expert) * inter;
    const auto* gate_data = gate.bytes().data() + first_row * (packed ? 2 : 1) * gate_stride;
    const auto* up_data = packed ? gate_data + inter * gate_stride : up.bytes().data() + first_row * up_stride;
    const auto* down_data = down.bytes().data() + static_cast<size_t>(expert) * hidden * down_stride;
    const auto gate_dot = row_dot(gate.desc.dtype, mode);
    const auto up_dot = packed ? gate_dot : row_dot(up.desc.dtype, mode);
    const auto down_dot = row_dot(down.desc.dtype, mode);

    std::vector<float> act(inter);
    auto gate_up = [&](int begin, int end) {
        for (int row = begin; row < end; ++row) {
            const float g = gate_dot(gate_data + row * gate_stride, input, hidden);
            const float u = up_dot(up_data + row * up_stride, input, hidden);
            act[row] = (g / (1.0f + std::exp(-g))) * u;
        }
    };
    auto down_accumulate = [&](int begin, int end) {
        for (int row = begin; row < end; ++row) {
            output[row] += weight * down_dot(down_data + row * down_stride, act.data(), inter);
        }
    };
    if (mode == CpuMoeMode::Parallel && static_cast<int64_t>(hidden) * inter >= 64 * 1024) {
        static ExpertWorkers workers;
        workers.run(inter, gate_up);
        workers.run(hidden, down_accumulate);
    } else {
        gate_up(0, inter);
        down_accumulate(0, hidden);
    }
}

}  // namespace fusionrt
