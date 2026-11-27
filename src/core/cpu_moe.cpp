#include "cpu_moe.h"
#include "fusionrt/weight_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <mutex>
#include <thread>
#include <type_traits>

#if defined(FUSIONRT_CPU_MOE_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace fusionrt {
namespace {

struct PhysicalCore {
    unsigned group = 0;
    unsigned number = 0;
};

std::vector<PhysicalCore> visible_physical_cores() {
    std::map<std::string, std::vector<PhysicalCore>> domains;
#ifdef _WIN32
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<uint8_t> records(bytes);
    if (bytes && GetLogicalProcessorInformationEx(RelationProcessorCore,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(records.data()), &bytes)) {
        USHORT group_count = GetActiveProcessorGroupCount();
        std::vector<USHORT> groups(group_count);
        if (GetProcessGroupAffinity(GetCurrentProcess(), &group_count, groups.data())) {
            DWORD_PTR visible = ~DWORD_PTR{0}, system = 0;
            if (group_count == 1) GetProcessAffinityMask(GetCurrentProcess(), &visible, &system);
            DWORD cache_bytes = 0;
            GetLogicalProcessorInformationEx(RelationCache, nullptr, &cache_bytes);
            std::vector<uint8_t> caches(cache_bytes);
            if (cache_bytes && !GetLogicalProcessorInformationEx(RelationCache,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(caches.data()), &cache_bytes))
                cache_bytes = 0;
            for (DWORD offset = 0; offset < bytes;) {
                const auto& entry = *reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(records.data() + offset);
                for (WORD i = 0; i < entry.Processor.GroupCount; ++i) {
                    const auto& mask = entry.Processor.GroupMask[i];
                    const auto cpus = mask.Mask & (group_count == 1 ? visible : ~DWORD_PTR{0});
                    if (!cpus || std::find(groups.begin(), groups.begin() + group_count, mask.Group) == groups.begin() + group_count)
                        continue;
                    const auto number = static_cast<unsigned>(std::countr_zero(cpus));
                    unsigned level = 0;
                    DWORD cache_id = 0;
                    for (DWORD at = 0; at < cache_bytes;) {
                        const auto& cache = *reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(caches.data() + at);
                        const auto& prop = cache.Cache;
                        if (prop.Level > level && (prop.Type == CacheUnified || prop.Type == CacheData) &&
                            prop.GroupMask.Group == mask.Group && (prop.GroupMask.Mask & (DWORD_PTR{1} << number))) {
                            level = prop.Level;
                            cache_id = at;
                        }
                        at += cache.Size;
                    }
                    domains[std::to_string(mask.Group) + ":" + std::to_string(cache_id)].push_back({mask.Group, number});
                    break;
                }
                offset += entry.Size;
            }
        }
    }
#elif defined(__linux__)
    cpu_set_t visible;
    CPU_ZERO(&visible);
    if (sched_getaffinity(0, sizeof(visible), &visible) == 0) {
        std::set<std::string> siblings;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &visible)) continue;
            const auto id = std::to_string(cpu);
            const std::string path = "/sys/devices/system/cpu/cpu" + id;
            std::ifstream topology(path + "/topology/thread_siblings_list");
            std::string core;
            if (!(topology >> core)) core = "cpu" + id;
            if (!siblings.insert(std::move(core)).second) continue;
            int level = 0;
            std::string domain;
            std::error_code error;
            for (std::filesystem::directory_iterator it(path + "/cache", error), end; !error && it != end; it.increment(error)) {
                std::ifstream depth(it->path() / "level"), type(it->path() / "type"), sharing(it->path() / "shared_cpu_list");
                int candidate;
                std::string kind, shared;
                if (depth >> candidate && candidate > level && type >> kind && (kind == "Unified" || kind == "Data") &&
                    sharing >> shared) {
                    level = candidate;
                    domain = std::move(shared);
                }
            }
            domains[domain].push_back({0, static_cast<unsigned>(cpu)});
        }
    }
#endif
    std::vector<PhysicalCore> cores;
    for (size_t i = 0;; ++i) {
        const size_t before = cores.size();
        for (const auto& entry : domains) {
            if (i < entry.second.size()) cores.push_back(entry.second[i]);
        }
        if (cores.size() == before) return cores;
    }
}

void pin_worker(PhysicalCore core) {
#ifdef _WIN32
    GROUP_AFFINITY affinity{};
    affinity.Group = static_cast<WORD>(core.group);
    affinity.Mask = DWORD_PTR{1} << core.number;
    SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
#elif defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(core.number, &affinity);
    pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
#else
    (void)core;
#endif
}

}  // namespace

struct CpuWorkers::State {
public:
    State(int count, std::span<const PhysicalCore> cores) {
        for (int i = 1; i < count; ++i) {
            workers_.emplace_back([this, i, count, pin = !cores.empty(),
                                   core = cores.empty() ? PhysicalCore{} : cores[i % cores.size()]](std::stop_token stop) {
                if (pin) pin_worker(core);
                unsigned seen = 0;
                std::unique_lock lock(mutex_);
                while (ready_.wait(lock, stop, [&] { return generation_ != seen; })) {
                    seen = generation_;
                    const auto* job = job_;
                    const auto work = work_;
                    const auto rows = rows_;
                    lock.unlock();
                    const auto begin = i * rows / count, end = (i + 1) * rows / count;
                    if (begin < end) work(job, begin, end);
                    lock.lock();
                    if (--remaining_ == 0) done_.notify_one();
                }
            });
        }
    }

    void run(int64_t rows, const void* job, Work work) {
        if (workers_.empty()) { work(job, 0, rows); return; }
        std::lock_guard dispatch(dispatch_);
        {
            std::lock_guard lock(mutex_);
            rows_ = rows;
            job_ = job;
            work_ = work;
            remaining_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        const auto first_end = rows / static_cast<int64_t>(workers_.size() + 1);
        if (first_end > 0) work(job, 0, first_end);
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }

    int count() const { return static_cast<int>(workers_.size()) + 1; }

private:
    std::mutex dispatch_, mutex_;
    std::condition_variable_any ready_;
    std::condition_variable done_;
    const void* job_ = nullptr;
    Work work_ = nullptr;
    size_t remaining_ = 0;
    int64_t rows_ = 0;
    unsigned generation_ = 0;
    std::vector<std::jthread> workers_;
};

CpuWorkers::CpuWorkers(int threads, bool spread_affinity) {
    if (threads < 0) throw std::invalid_argument("Expected --threads >= 0");
    const auto cores = threads == 0 || spread_affinity ? visible_physical_cores() : std::vector<PhysicalCore>{};
    const int visible = cores.empty() ? static_cast<int>(std::thread::hardware_concurrency()) : static_cast<int>(cores.size());
    state_ = std::make_unique<State>(threads > 0 ? threads : std::max(1, visible - 1),
                                    spread_affinity ? std::span<const PhysicalCore>(cores) : std::span<const PhysicalCore>());
}

CpuWorkers::~CpuWorkers() = default;
int CpuWorkers::count() const { return state_->count(); }
void CpuWorkers::run_impl(int64_t rows, const void* job, Work work) { state_->run(rows, job, work); }

namespace {

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

bool use_avx2() {
    static const bool supported = has_avx2_fma();
    return supported;
}
#endif

#ifdef FUSIONRT_CPU_MOE_AVX512
bool has_avx512_vnni() {
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuid(regs, 1);
    if (!(regs[2] & (1 << 27)) || (_xgetbv(0) & 0xe6) != 0xe6) return false;
    __cpuidex(regs, 7, 0);
    constexpr unsigned mask = (1u << 16) | (1u << 30) | (1u << 31);  // F, BW, VL
    return (static_cast<unsigned>(regs[1]) & mask) == mask && (regs[2] & (1 << 11));
#else
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
#endif
}

bool use_avx512_vnni() {
    static const bool supported = has_avx512_vnni();
    return supported;
}
#endif

template<class Codec>
float codec_dot(const uint8_t* row, const float* x, int cols) {
    float sum = 0;
    for (int i = 0; i < cols; i += Codec::kBlockSize, row += Codec::kBlockBytes)
        sum += Codec::dot_block(row, x + i);
    return sum;
}

template<class Codec>
void codec_dot_rows(const uint8_t* row, const float* const* x, int cols, int count, float* out) {
    std::fill_n(out, count, 0.0f);
    for (int i = 0; i < cols; i += Codec::kBlockSize, row += Codec::kBlockBytes) {
        float values[Codec::kBlockSize];
        Codec::dequant_block(row, values);
        for (int m = 0; m < count; ++m) {
            float sum = 0.0f;
            for (int j = 0; j < Codec::kBlockSize; ++j) sum += values[j] * x[m][i + j];
            out[m] += sum;
        }
    }
}

using CodecRows = decltype(&codec_dot_rows<WeightCodec<DType::Q3_K>>);
CodecRows codec_rows(DType type, int rows, CpuMoeMode mode) {
    if (rows <= 1 || mode == CpuMoeMode::Scalar || !is_weight_codec_dtype(type)) return nullptr;
    return visit_weight_codec(type, []<class Codec>() -> CodecRows { return codec_dot_rows<Codec>; });
}

CpuMoeDot row_dot(DType type, CpuMoeMode mode) {
    if (is_weight_codec_dtype(type))
        return visit_weight_codec(type, []<class Codec>() -> CpuMoeDot { return codec_dot<Codec>; });
#ifdef FUSIONRT_CPU_MOE_AVX2
    if (mode != CpuMoeMode::Scalar && use_avx2()) {
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
        case DType::Q2_K: return scalar_quantized_dot<BlockQ2_K, 256, dequantize_row_q2_k>;
        case DType::Q4_K: return scalar_quantized_dot<BlockQ4_K, 256, dequantize_row_q4_k>;
        case DType::Q5_K: return scalar_quantized_dot<BlockQ5_K, 256, dequantize_row_q5_k>;
        case DType::Q6_K: return scalar_quantized_dot<BlockQ6_K, 256, dequantize_row_q6_k>;
        default: throw std::invalid_argument("Unsupported CPU expert weight dtype");
    }
}

template<class Block, class Kernel, size_t N>
Kernel benchmark_kernels(const std::array<Kernel, N>& kernels) {
    const auto first = std::find_if(kernels.begin(), kernels.end(), [](Kernel kernel) { return kernel != nullptr; });
    if (first == kernels.end()) return nullptr;
    if (std::count_if(kernels.begin(), kernels.end(), [](Kernel kernel) { return kernel != nullptr; }) == 1)
        return *first;

    constexpr int blocks = 4, cols = blocks * 256;
    alignas(64) std::array<Block, blocks> weights;
    uint32_t seed = 1;
    auto next = [&] {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<uint8_t>(seed >> 24);
    };
    for (auto& block : weights) {
        auto* bytes = reinterpret_cast<uint8_t*>(&block);
        std::generate_n(bytes, sizeof(Block), next);
        block.d = 0x2400;  // Use finite scales with nontrivial packed values.
        if constexpr (!std::is_same_v<Block, BlockQ6_K>) block.dmin = 0x1800;
    }
    alignas(64) std::array<std::array<CpuQ8K, blocks>, 4> activations;
    std::array<const CpuQ8K*, 4> inputs{};
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::array<float, cols> values;
        std::generate(values.begin(), values.end(), [&] { return (static_cast<int>(next()) - 128) / 128.0f; });
        cpu_moe_quantize_q8(values.data(), activations[i].data(), cols);
        inputs[i] = activations[i].data();
    }

    const auto* row = reinterpret_cast<const uint8_t*>(weights.data());
    auto measure = [&](Kernel kernel) {
        const auto start = std::chrono::steady_clock::now();
        float sum = 0.0f;
        for (int i = 0; i < 96; ++i) {
            if constexpr (std::is_same_v<Kernel, CpuMoeQ8Dot>) {
                sum += kernel(row, inputs[i % 4], cols);
            } else {
                float out[4];
                kernel(row, inputs.data(), cols, 2 + i % 3, out);
                sum += out[0];
            }
        }
        volatile float observed = sum;
        (void)observed;
        return std::chrono::steady_clock::now() - start;
    };

    // Warm each candidate and rotate the measurement order to reduce clock and cache bias.
    constexpr size_t rounds = 7;
    using Duration = std::chrono::steady_clock::duration;
    std::array<std::array<Duration, rounds>, N> samples{};
    for (auto kernel : kernels) if (kernel) measure(kernel);
    for (size_t round = 0; round < rounds; ++round) {
        for (size_t i = 0; i < N; ++i) {
            const size_t index = (i + round) % N;
            if (kernels[index]) samples[index][round] = measure(kernels[index]);
        }
    }
    auto best_time = Duration::max();
    Kernel best = *first;
    for (size_t i = 0; i < N; ++i) {
        if (!kernels[i]) continue;
        auto& times = samples[i];
        std::sort(times.begin(), times.end());
        if (times[rounds / 2] < best_time) {
            best_time = times[rounds / 2];
            best = kernels[i];
        }
    }
    return best;
}

template<class Block>
CpuMoeQ8Dot tuned_dot(DType dtype) {
    static const auto selected = benchmark_kernels<Block>(
        std::array{cpu_moe_avx2_integer_dot(dtype), cpu_moe_vnni_dot(dtype), cpu_moe_vnni_dot(dtype, true)});
    return selected;
}

template<class Block>
CpuMoeQ8Rows tuned_rows(DType dtype) {
    static const auto selected = [dtype] {
        CpuMoeQ8Rows avx2 = nullptr, vnni = nullptr;
#ifdef FUSIONRT_CPU_MOE_AVX2
        if (use_avx2()) avx2 = cpu_moe_avx2_q8_rows(dtype);
#endif
#ifdef FUSIONRT_CPU_MOE_AVX512
        if (use_avx512_vnni()) vnni = cpu_moe_avx512_q8_rows(dtype);
#endif
        (void)dtype;
        return benchmark_kernels<Block>(std::array{avx2, vnni});
    }();
    return selected;
}

}  // namespace

CpuMoeQ8Dot cpu_moe_q8_dot(DType dtype) {
    switch (dtype) {
        case DType::Q2_K: return tuned_dot<BlockQ2_K>(dtype);
        case DType::Q4_K: return tuned_dot<BlockQ4_K>(dtype);
        case DType::Q6_K: return tuned_dot<BlockQ6_K>(dtype);
        default: return nullptr;
    }
}

CpuMoeQ8Dot cpu_moe_avx2_integer_dot(DType dtype) {
#ifdef FUSIONRT_CPU_MOE_AVX2
    if (use_avx2()) return cpu_moe_avx2_q8_dot(dtype);
#else
    (void)dtype;
#endif
    return nullptr;
}

CpuMoeQ8Rows cpu_moe_q8_rows(DType dtype) {
    switch (dtype) {
        case DType::Q2_K: return tuned_rows<BlockQ2_K>(dtype);
        case DType::Q4_K: return tuned_rows<BlockQ4_K>(dtype);
        case DType::Q6_K: return tuned_rows<BlockQ6_K>(dtype);
        default: return nullptr;
    }
}

CpuMoeQ8Dot cpu_moe_vnni_dot(DType dtype, bool wide) {
#ifdef FUSIONRT_CPU_MOE_AVX512
    if (use_avx512_vnni()) return cpu_moe_avx512_q8_dot(dtype, wide);
#else
    (void)dtype;
    (void)wide;
#endif
    return nullptr;
}

CpuMoeQ8SmallDot cpu_moe_vnni_small_dot(DType dtype) {
#ifdef FUSIONRT_CPU_MOE_AVX512
    if (use_avx512_vnni()) return cpu_moe_avx512_small_dot(dtype);
#else
    (void)dtype;
#endif
    return nullptr;
}

static float finite_activation(float x) { return std::isfinite(x) ? x : 0.0f; }

void cpu_moe_quantize_q8_small(const float* input, CpuQ8Small* output, int cols) {
    for (int i = 0; i < cols; i += 32) {
        auto& block = output[i / 32];
        float max = 0.0f;
        for (int j = 0; j < 32; ++j) max = std::max(max, std::abs(finite_activation(input[i + j])));
        block.scale = max / 127.0f;
        const float inv = max > 0.0f ? 127.0f / max : 0.0f;
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            block.values[j] = static_cast<int8_t>(std::nearbyint(finite_activation(input[i + j]) * inv));
            sum += block.values[j];
        }
        block.sum = static_cast<int16_t>(sum);
    }
}

void cpu_moe_quantize_q8(const float* input, CpuQ8K* output, int cols) {
#ifdef FUSIONRT_CPU_MOE_AVX2
    if (use_avx2()) {
        cpu_moe_avx2_quantize_q8(input, output, cols);
        return;
    }
#endif
    for (int i = 0; i < cols; i += 256) {
        auto& block = output[i / 256];
        float max = 0.0f;
        for (int j = 0; j < 256; ++j) max = std::max(max, std::abs(finite_activation(input[i + j])));
        block.scale = max / 127.f;
        const float inv = max > 0.f ? 127.f / max : 0.f;
        for (int sub = 0; sub < 16; ++sub) {
            int sum = 0;
            for (int j = sub * 16; j < (sub + 1) * 16; ++j) {
                block.values[j] = static_cast<int8_t>(std::nearbyint(finite_activation(input[i + j]) * inv));
                sum += block.values[j];
            }
            block.sums[sub] = static_cast<int16_t>(sum);
        }
    }
}

template<bool ExpertOps>
static int accumulate_layer(CpuWorkers* workers, CpuMoeWorkspace& scratch,
    const HostTensor& gate, const HostTensor& up, const HostTensor& down,
    const int32_t* indices, const float* weights, int first_cold, int host_first, int rows, int top_k,
    const float* input, float* output, CpuMoeMode mode, bool geglu, const float* gate_scale,
    const float* up_scale) {
    auto activate = [&](float g, float u, int route) {
        if constexpr (ExpertOps) {
            const int expert = indices[route];
            if (gate_scale) g *= gate_scale[expert];
            if (up_scale) u *= up_scale[expert];
            if (geglu) return (0.5f * g) * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
        }
        return (g / (1.0f + std::exp(-g))) * u;
    };
    auto& jobs = scratch.jobs;
    jobs.clear();
    jobs.reserve(static_cast<size_t>(rows) * top_k);
    scratch.offsets.resize(static_cast<size_t>(rows) + 1);
    const int hidden = static_cast<int>(down.desc.shape[1]);
    const int inter = static_cast<int>(down.desc.shape[2]);
    const bool packed = up.bytes().empty();
    const size_t gate_stride = row_nbytes(gate.desc.dtype, hidden);
    const size_t up_stride = packed ? gate_stride : row_nbytes(up.desc.dtype, hidden);
    const size_t down_stride = row_nbytes(down.desc.dtype, inter);
    for (int row = 0; row < rows; ++row) {
        scratch.offsets[row] = static_cast<int>(jobs.size());
        for (int k = 0; k < top_k; ++k) {
            const int route = row * top_k + k;
            const int id = indices[route];
            if (id < 0 || id < first_cold || id < host_first ||
                static_cast<int64_t>(id) - host_first >= down.desc.shape[0]) continue;
            const size_t expert = static_cast<size_t>(id - host_first);
            const size_t first = expert * inter;
            const auto* g = gate.bytes().data() + first * (packed ? 2 : 1) * gate_stride;
            const auto* u = packed ? g + inter * gate_stride : up.bytes().data() + first * up_stride;
            const auto* d = down.bytes().data() + expert * hidden * down_stride;
            jobs.push_back({route, g, u, d});
        }
    }
    const int cold = static_cast<int>(jobs.size());
    scratch.offsets[rows] = cold;
    if (cold == 0) return 0;

    if (rows > 1) {
        auto& grouped = scratch.grouped;
        grouped.resize(cold);
        std::iota(grouped.begin(), grouped.end(), 0);
        std::sort(grouped.begin(), grouped.end(), [&](int a, int b) {
            return indices[jobs[a].row] < indices[jobs[b].row];
        });
    }
    auto& act = scratch.activation;
    act.reserve(static_cast<size_t>(rows) * top_k * inter);
    act.resize(static_cast<size_t>(cold) * inter);
    const auto gate_dot = row_dot(gate.desc.dtype, mode);
    const auto up_dot = packed ? gate_dot : row_dot(up.desc.dtype, mode);
    const auto down_dot = row_dot(down.desc.dtype, mode);
    const auto gate_q8 = mode == CpuMoeMode::Parallel ? cpu_moe_q8_dot(gate.desc.dtype) : nullptr;
    const auto up_q8 = packed ? gate_q8 : mode == CpuMoeMode::Parallel ? cpu_moe_q8_dot(up.desc.dtype) : nullptr;
    const auto down_q8 = mode == CpuMoeMode::Parallel ? cpu_moe_q8_dot(down.desc.dtype) : nullptr;
    const auto gate_small = mode == CpuMoeMode::Parallel ? cpu_moe_vnni_small_dot(gate.desc.dtype) : nullptr;
    const auto up_small = packed ? gate_small : mode == CpuMoeMode::Parallel ? cpu_moe_vnni_small_dot(up.desc.dtype) : nullptr;
    const auto down_small = mode == CpuMoeMode::Parallel ? cpu_moe_vnni_small_dot(down.desc.dtype) : nullptr;
    const int hidden_blocks = hidden / 256, inter_blocks = inter / 256;
    const int hidden_small_blocks = hidden / 32, inter_small_blocks = inter / 32;
    const auto gate_rows_q8 = rows > 1 && gate_q8 ? cpu_moe_q8_rows(gate.desc.dtype) : nullptr;
    const auto up_rows_q8 = rows > 1 && up_q8 ? (packed ? gate_rows_q8 : cpu_moe_q8_rows(up.desc.dtype)) : nullptr;
    const auto down_rows_q8 = rows > 1 && down_q8 ? cpu_moe_q8_rows(down.desc.dtype) : nullptr;
    if (gate_q8 || up_q8) {
        scratch.input_q8.resize(static_cast<size_t>(rows) * hidden_blocks);
        for (int row = 0; row < rows; ++row) {
            if (scratch.offsets[row] == scratch.offsets[row + 1]) continue;
            cpu_moe_quantize_q8(input + static_cast<size_t>(row) * hidden,
                scratch.input_q8.data() + static_cast<size_t>(row) * hidden_blocks, hidden);
        }
    }
    if (gate_small || up_small) {
        scratch.input_q8_small.resize(static_cast<size_t>(rows) * hidden_small_blocks);
        for (int row = 0; row < rows; ++row) {
            if (scratch.offsets[row] == scratch.offsets[row + 1]) continue;
            cpu_moe_quantize_q8_small(input + static_cast<size_t>(row) * hidden,
                scratch.input_q8_small.data() + static_cast<size_t>(row) * hidden_small_blocks, hidden);
        }
    }
    if (down_q8) scratch.activation_q8.resize(static_cast<size_t>(cold) * inter_blocks);
    if (down_small) scratch.activation_q8_small.resize(static_cast<size_t>(cold) * inter_small_blocks);
    const auto gate_codec = codec_rows(gate.desc.dtype, rows, mode);
    const auto up_codec = packed ? gate_codec : codec_rows(up.desc.dtype, rows, mode);
    const auto down_codec = codec_rows(down.desc.dtype, rows, mode);
    const bool batched_gate = gate_rows_q8 || up_rows_q8 || gate_codec || up_codec;
    const bool batched_down = down_rows_q8 || down_codec;
    if (batched_gate || batched_down) {
        auto& tiles = scratch.tiles;
        tiles.clear();
        for (int i = 0; i < cold;) {
            const int expert = indices[jobs[scratch.grouped[i]].row];
            tiles.push_back(i);
            const int limit = std::min(i + 4, cold);
            while (++i < limit && indices[jobs[scratch.grouped[i]].row] == expert) {}
        }
        tiles.push_back(cold);
    }
    auto gate_up_tile = [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const auto tile = static_cast<size_t>(idx / inter);
            const int start = scratch.tiles[tile], count = scratch.tiles[tile + 1] - start;
            const auto row = static_cast<size_t>(idx % inter);
            const auto* ids = scratch.grouped.data() + start;
            const auto& job = jobs[ids[0]];
            std::array<size_t, 4> tokens{};
            std::array<const float*, 4> xs{};
            std::array<const CpuQ8K*, 4> qx{};
            for (int m = 0; m < count; ++m) {
                tokens[m] = static_cast<size_t>(jobs[ids[m]].row / top_k);
                xs[m] = input + tokens[m] * hidden;
                if (gate_q8 || up_q8) qx[m] = scratch.input_q8.data() + tokens[m] * hidden_blocks;
            }
            float g[4], u[4];
            auto dot = [&](auto batch, auto codec, auto quantized, auto small, auto scalar, const uint8_t* w, float* out) {
                if (codec && count > 1) codec(w, xs.data(), hidden, count, out);
                else if (batch && count > 1) batch(w, qx.data(), hidden, count, out);
                else for (int m = 0; m < count; ++m) {
                    const auto token = tokens[m];
                    out[m] = quantized ? quantized(w, qx[m], hidden) :
                        small ? small(w, scratch.input_q8_small.data() + token * hidden_small_blocks, hidden) :
                                scalar(w, input + token * hidden, hidden);
                }
            };
            dot(gate_rows_q8, gate_codec, gate_q8, gate_small, gate_dot, job.gate + row * gate_stride, g);
            dot(up_rows_q8, up_codec, up_q8, up_small, up_dot, job.up + row * up_stride, u);
            for (int m = 0; m < count; ++m)
                act[static_cast<size_t>(ids[m]) * inter + row] = activate(g[m], u[m], jobs[ids[m]].row);
        }
    };
    auto gate_up = [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int tile = static_cast<int>(idx / inter);
            const int job_id = rows > 1 ? scratch.grouped[tile] : tile;
            const auto& job = jobs[job_id];
            const auto token = static_cast<size_t>(job.row / top_k);
            const auto* x = input + token * hidden;
            const auto* qx = (gate_q8 || up_q8) ? scratch.input_q8.data() + token * hidden_blocks : nullptr;
            const auto* sx = (gate_small || up_small) ? scratch.input_q8_small.data() + token * hidden_small_blocks : nullptr;
            const auto row = static_cast<size_t>(idx % inter);
            const auto* g_row = job.gate + row * gate_stride;
            const auto* u_row = job.up + row * up_stride;
            const float g = gate_q8 ? gate_q8(g_row, qx, hidden) :
                gate_small ? gate_small(g_row, sx, hidden) : gate_dot(g_row, x, hidden);
            const float u = up_q8 ? up_q8(u_row, qx, hidden) :
                up_small ? up_small(u_row, sx, hidden) : up_dot(u_row, x, hidden);
            act[static_cast<size_t>(job_id) * inter + row] = activate(g, u, job.row);
        }
    };
    auto down_accumulate = [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const auto token = static_cast<size_t>(idx / hidden);
            const auto row = static_cast<size_t>(idx % hidden);
            float acc = 0.0f;
            for (int j = scratch.offsets[token]; j < scratch.offsets[token + 1]; ++j) {
                const auto& job = jobs[j];
                const auto* d_row = job.down + row * down_stride;
                const float dot = down_q8 ? down_q8(d_row, scratch.activation_q8.data() + static_cast<size_t>(j) * inter_blocks, inter)
                    : down_small ? down_small(d_row, scratch.activation_q8_small.data() + static_cast<size_t>(j) * inter_small_blocks, inter)
                                          : down_dot(d_row, act.data() + static_cast<size_t>(j) * inter, inter);
                acc += weights[job.row] * dot;
            }
            output[idx] = acc;
        }
    };
    const int64_t down_rows = static_cast<int64_t>(rows) * hidden;
    if (batched_gate) {
        const auto units = static_cast<int64_t>(scratch.tiles.size() - 1) * inter;
        if (workers) workers->run(units, gate_up_tile);
        else gate_up_tile(0, units);
    } else {
        const int64_t units = static_cast<int64_t>(cold) * inter;
        if (workers) workers->run(units, gate_up);
        else gate_up(0, units);
    }
    // Quantizing after the gate pass lets workers split the expensive dot products by individual rows.
    if (down_q8) {
        for (size_t job = 0; job < jobs.size(); ++job)
            cpu_moe_quantize_q8(act.data() + job * inter, scratch.activation_q8.data() + job * inter_blocks, inter);
    }
    if (down_small) {
        for (size_t job = 0; job < jobs.size(); ++job)
            cpu_moe_quantize_q8_small(act.data() + job * inter, scratch.activation_q8_small.data() + job * inter_small_blocks, inter);
    }
    if (batched_down) {
        const int lanes = std::min(hidden, workers ? workers->count() : 1);
        scratch.down_dots.resize(static_cast<size_t>(lanes) * cold);
        auto down_tile = [&](int64_t begin, int64_t end) {
            for (auto lane = begin; lane < end; ++lane) {
                auto* dots = scratch.down_dots.data() + lane * cold;
                const auto row_begin = lane * hidden / lanes, row_end = (lane + 1) * hidden / lanes;
                for (auto row = row_begin; row < row_end; ++row) {
                    for (size_t tile = 1; tile < scratch.tiles.size(); ++tile) {
                        const int start = scratch.tiles[tile - 1], count = scratch.tiles[tile] - start;
                        const auto* ids = scratch.grouped.data() + start;
                        const auto* w = jobs[ids[0]].down + row * down_stride;
                        float dst[4];
                        if (down_codec) {
                            const float* xs[4];
                            for (int m = 0; m < count; ++m)
                                xs[m] = act.data() + static_cast<size_t>(ids[m]) * inter;
                            if (count == 1) dst[0] = down_dot(w, xs[0], inter);
                            else down_codec(w, xs, inter, count, dst);
                        } else {
                            const CpuQ8K* qx[4];
                            for (int m = 0; m < count; ++m)
                                qx[m] = scratch.activation_q8.data() + static_cast<size_t>(ids[m]) * inter_blocks;
                            if (count == 1) dst[0] = down_q8(w, qx[0], inter);
                            else down_rows_q8(w, qx, inter, count, dst);
                        }
                        for (int m = 0; m < count; ++m) dots[ids[m]] = dst[m];
                    }
                    for (int token = 0; token < rows; ++token) {
                        float sum = 0.0f;
                        for (int j = scratch.offsets[token]; j < scratch.offsets[token + 1]; ++j)
                            sum += weights[jobs[j].row] * dots[j];
                        output[static_cast<size_t>(token) * hidden + row] = sum;
                    }
                }
            }
        };
        if (workers) workers->run(lanes, down_tile);
        else down_tile(0, lanes);
    } else if (workers) workers->run(down_rows, down_accumulate);
    else down_accumulate(0, down_rows);
    return cold;
}

int cpu_moe_accumulate_layer(CpuWorkers* workers, CpuMoeWorkspace& scratch,
    const HostTensor& gate, const HostTensor& up, const HostTensor& down,
    const int32_t* indices, const float* weights, int first_cold, int host_first, int rows, int top_k,
    const float* input, float* output, CpuMoeMode mode, bool geglu, const float* gate_scale, const float* up_scale) {
    if (geglu || gate_scale || up_scale)
        return accumulate_layer<true>(workers, scratch, gate, up, down, indices, weights, first_cold, host_first,
                                      rows, top_k, input, output, mode, geglu, gate_scale, up_scale);
    return accumulate_layer<false>(workers, scratch, gate, up, down, indices, weights, first_cold, host_first,
                                   rows, top_k, input, output, mode, false, nullptr, nullptr);
}

void cpu_moe_accumulate(const HostTensor& gate, const HostTensor& up, const HostTensor& down,
                        int expert, const float* input, float weight, float* output, CpuMoeMode mode) {
    if (expert < 0 || expert >= down.desc.shape[0]) return;
    thread_local CpuMoeWorkspace workspace;
    thread_local std::vector<float> contribution;
    const int hidden = static_cast<int>(down.desc.shape[1]);
    contribution.resize(hidden);
    CpuWorkers* workers = nullptr;
    if (mode != CpuMoeMode::Scalar) {
        thread_local CpuWorkers pool;
        workers = &pool;
    }
    if (!cpu_moe_accumulate_layer(workers, workspace, gate, up, down, &expert, &weight, 0, 0, 1, 1,
                                  input, contribution.data(), mode)) return;
    for (int i = 0; i < hidden; ++i) output[i] += contribution[i];
}

}  // namespace fusionrt
