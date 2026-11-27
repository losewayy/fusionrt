#pragma once

#include "fusionrt/tensor.h"
#include <memory>
#include <span>

namespace fusionrt {

enum class CpuMoeMode { Parallel, Scalar, ParallelF32 };

struct CpuQ8K {
    float scale;
    int8_t values[256];
    int16_t sums[16];
};

struct CpuQ8Small {
    float scale;
    int8_t values[32];
    int16_t sum;
};

class CpuWorkers {
public:
    explicit CpuWorkers(int threads = 0, bool spread_affinity = false);
    ~CpuWorkers();
    int count() const;

    template<class Job>
    void run(int64_t rows, const Job& job) {
        run_impl(rows, &job, [](const void* self, int64_t begin, int64_t end) {
            (*static_cast<const Job*>(self))(begin, end);
        });
    }

private:
    using Work = void (*)(const void*, int64_t, int64_t);
    void run_impl(int64_t rows, const void* job, Work work);
    struct State;
    std::unique_ptr<State> state_;
};

struct CpuMoeWorkspace {
    struct Job {
        int row;
        const uint8_t* gate;
        const uint8_t* up;
        const uint8_t* down;
    };
    std::vector<Job> jobs;
    std::vector<int> offsets, grouped, tiles;
    std::vector<float> activation, down_dots;
    std::vector<CpuQ8K> input_q8, activation_q8;
    std::vector<CpuQ8Small> input_q8_small, activation_q8_small;
};

// Routes are token-major and ordered by the router within each token.
int cpu_moe_accumulate_layer(CpuWorkers* workers, CpuMoeWorkspace& scratch,
    const HostTensor& gate, const HostTensor& up, const HostTensor& down,
    const int32_t* indices, const float* weights, int first_cold, int host_first, int rows, int top_k,
    const float* input, float* output, CpuMoeMode mode = CpuMoeMode::Parallel);

// The gate tensor also holds up when the separate up tensor is empty.
void cpu_moe_accumulate(const HostTensor& gate, const HostTensor& up, const HostTensor& down,
                        int expert, const float* input, float weight, float* output,
                        CpuMoeMode mode = CpuMoeMode::ParallelF32);

using CpuMoeDot = float (*)(const uint8_t*, const float*, int);
using CpuMoeQ8Dot = float (*)(const uint8_t*, const CpuQ8K*, int);
using CpuMoeQ8Rows = void (*)(const uint8_t*, const CpuQ8K* const*, int, int, float*);
using CpuMoeQ8SmallDot = float (*)(const uint8_t*, const CpuQ8Small*, int);
CpuMoeQ8Dot cpu_moe_q8_dot(DType dtype);
CpuMoeQ8Dot cpu_moe_avx2_integer_dot(DType dtype);
CpuMoeQ8Rows cpu_moe_q8_rows(DType dtype);
CpuMoeQ8Dot cpu_moe_vnni_dot(DType dtype, bool wide = false);
CpuMoeQ8SmallDot cpu_moe_vnni_small_dot(DType dtype);
void cpu_moe_quantize_q8(const float* input, CpuQ8K* output, int cols);
void cpu_moe_quantize_q8_small(const float* input, CpuQ8Small* output, int cols);
#ifdef FUSIONRT_CPU_MOE_AVX2
CpuMoeDot cpu_moe_avx2_dot(DType dtype);
CpuMoeQ8Dot cpu_moe_avx2_q8_dot(DType dtype);
CpuMoeQ8Rows cpu_moe_avx2_q8_rows(DType dtype);
void cpu_moe_avx2_quantize_q8(const float* input, CpuQ8K* output, int cols);
#endif
#ifdef FUSIONRT_CPU_MOE_AVX512
CpuMoeQ8Dot cpu_moe_avx512_q8_dot(DType dtype, bool wide);
CpuMoeQ8Rows cpu_moe_avx512_q8_rows(DType dtype);
CpuMoeQ8SmallDot cpu_moe_avx512_small_dot(DType dtype);
#endif

}  // namespace fusionrt
