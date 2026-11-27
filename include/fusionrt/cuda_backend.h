#pragma once

#include "fusionrt/cost_model.h"
#include "fusionrt/device_profile.h"
#include "fusionrt/tensor.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace fusionrt {

class CpuWorkers;
class MoeRouteTrace;
struct CpuMoeWorkspace;

struct CudaDeviceInfo {
    int device_id = 0;
    std::string name;
    int sm_major = 0;
    int sm_minor = 0;
    size_t total_vram_bytes = 0;
    size_t l2_cache_bytes = 0;
    int multi_processor_count = 0;
    int shared_mem_per_block = 0;
    int shared_mem_per_block_optin = 0;
    int shared_mem_per_sm = 0;
    int warp_size = 32;
    int max_threads_per_block = 1024;
    double mem_clock_khz = 0.0;
    double sm_clock_khz = 0.0;
    int memory_bus_width_bits = 0;
    double peak_bandwidth_gbps = 0.0;
    std::string driver_version;
};

class CudaMoeTiming {
public:
    struct Layer {
        size_t hot = 0, cold = 0;
        double input_wait_ms = 0, cpu_ms = 0, gpu_gap_ms = 0;
        cudaEvent_t gpu_done = nullptr, cpu_done = nullptr;
        bool pending = false;
    };
    explicit CudaMoeTiming(int layers) : layers_(layers) {}
    ~CudaMoeTiming();
    CudaMoeTiming(const CudaMoeTiming&) = delete;
    CudaMoeTiming& operator=(const CudaMoeTiming&) = delete;
    Layer& layer(int index) { return layers_[index]; }
    void begin_gap(Layer& layer, cudaStream_t stream);
    void end_gap(Layer& layer, cudaStream_t stream);
    void collect();
    std::span<const Layer> layers() const { return layers_; }

private:
    std::vector<Layer> layers_;
};

// Bump-pointer GPU arena with 256-byte alignment and save/restore scratch watermark.
// Safe for CUDA Graph capture (no cudaMalloc/cudaFree during capture).
class CudaArena {
public:
    CudaArena() = default;
    ~CudaArena();
    CudaArena(const CudaArena&) = delete;
    CudaArena& operator=(const CudaArena&) = delete;

    void init(size_t capacity_bytes);
    void release();

    void* alloc(size_t bytes, size_t alignment = 256);
    Tensor alloc_tensor(const std::vector<int64_t>& shape, DType dt, const std::string& name = "");

    size_t mark() const { return offset_; }
    void reset(size_t mark = 0) { offset_ = mark; }
    size_t used() const { return offset_; }
    size_t capacity() const { return capacity_; }
    void track_peak(bool enabled) { track_peak_ = enabled; peak_ = offset_; }
    size_t peak() const { return peak_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    bool track_peak_ = false;
    size_t peak_ = 0;
};

class CudaGraphExec {
public:
    CudaGraphExec() = default;
    ~CudaGraphExec();
    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;

    void capture(cudaStream_t stream, const std::function<void()>& fn);
    void launch(cudaStream_t stream);
    void reset();
    bool valid() const { return exec_ != nullptr; }

private:
    friend struct CudaReviewTestAccess;
    cudaError_t (*instantiate_)(cudaGraphExec_t*, cudaGraph_t) = [](cudaGraphExec_t* exec, cudaGraph_t graph) {
        return cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0);
    };
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

struct GemvGeometry {
    int warps = 0;
    int rows_per_warp = 0;
};

class CudaContext {
public:
    explicit CudaContext(int device_id = 0, size_t weight_arena_mb = 256, size_t scratch_arena_mb = 256);
    ~CudaContext();
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    const CudaDeviceInfo& info() const { return info_; }
    cudaStream_t stream() const { return stream_; }
    cublasHandle_t cublas() const { return cublas_; }
    int prefill_batch() const { return prefill_batch_; }
    void set_prefill_batch(int rows) { prefill_batch_ = rows; }
    const DeviceProfile& device_profile() const { return device_profile_; }
    const CostHardware& cost_hardware() const { return cost_hardware_; }
    void set_device_profile(const DeviceProfile& profile);
    int mmq_crossover_m() const { return device_profile_.mmq_crossover_m; }
    void set_mmq_crossover_m(int rows) { device_profile_.mmq_crossover_m = rows; }
    GemvGeometry preferred_gemv_geometry() const {
        return {device_profile_.preferred_gemv_warps, device_profile_.preferred_gemv_rows_per_warp};
    }
    void set_preferred_gemv_geometry(GemvGeometry geometry) {
        device_profile_.preferred_gemv_warps = geometry.warps;
        device_profile_.preferred_gemv_rows_per_warp = geometry.rows_per_warp;
    }
    uint64_t empirical_l2_bytes() const { return device_profile_.empirical_l2_bytes; }
    void set_empirical_l2_bytes(uint64_t bytes) {
        device_profile_.empirical_l2_bytes = bytes;
        set_device_profile(device_profile_);
    }
    DType kv_dtype() const { return kv_request_.value_or(DType::F16); }
    std::optional<DType> requested_kv_dtype() const { return kv_request_; }
    void set_kv_dtype(std::optional<DType> dtype);

    CudaArena& weight_arena() { return weight_arena_; }
    CudaArena& scratch_arena() { return scratch_arena_; }

    Tensor upload(const HostTensor& ht, bool persistent = true);
    void upload_into(const HostTensor& ht, Tensor& dst);
    HostTensor download(const Tensor& t);
    void synchronize();
    float* moe_transfer_buffer(size_t bytes);
    cudaEvent_t moe_input_ready() const { return moe_input_ready_; }
    CudaMoeTiming* moe_timing = nullptr;
    MoeRouteTrace* moe_route_trace = nullptr;
    void set_cpu_threads(int threads, bool spread_affinity = false);
    void set_cpu_moe_q8(bool enabled) { cpu_moe_q8_ = enabled; }
    bool cpu_moe_q8() const { return cpu_moe_q8_; }
    CpuWorkers& cpu_workers();
    CpuMoeWorkspace& cpu_moe_workspace();

    // L2 cache flush buffer (> 2x device L2) for anti-noise benchmarking
    void flush_l2();

    // Driver API module/device query helpers
    static std::vector<CudaDeviceInfo> enumerate_devices();

private:
    CudaDeviceInfo info_;
    DeviceProfile device_profile_;
    CostHardware cost_hardware_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    CudaArena weight_arena_;
    CudaArena scratch_arena_;
    void* l2_flush_buf_ = nullptr;
    size_t l2_flush_bytes_ = 0;
    int prefill_batch_ = 512;
    std::optional<DType> kv_request_ = DType::F32;
    std::unique_ptr<float, decltype(&cudaFreeHost)> moe_transfer_{nullptr, cudaFreeHost};
    size_t moe_transfer_bytes_ = 0;
    cudaEvent_t moe_input_ready_ = nullptr;
    int cpu_threads_ = 0;
    bool cpu_spread_affinity_ = false;
    bool cpu_moe_q8_ = true;
    std::unique_ptr<CpuWorkers> cpu_workers_;
    std::unique_ptr<CpuMoeWorkspace> cpu_moe_workspace_;
};

inline constexpr int kKvPageTokens = 128;

class KvVramExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct CudaKvView {
    uint8_t** pages = nullptr;
    size_t plane_bytes = 0;   // K or V in one page, including the Q8 scale plane.
    int elements = 0;         // Elements in one plane before quantization.
    int max_pages = 0;
    DType dtype = DType::F32;
    bool latent = false;

    size_t page_bytes() const { return (latent ? 1 : 2) * plane_bytes; }
};

CudaKvView make_cuda_kv_view(CudaArena& arena, DType dtype, int max_tokens, int kv_heads, int head_dim,
                             bool latent = false);

class CudaKvPool {
public:
    class Table {
    public:
        Table() = default;
        Table(const Table&) = delete;
        Table& operator=(const Table&) = delete;

    private:
        friend class CudaKvPool;
        std::vector<int> pages_;
        int published_ = 0;
        int spilled_ = 0;
    };

    void ensure(CudaContext& ctx, std::span<const CudaKvView> layers, Table& table, int first_write, int tokens,
                std::span<Table* const> paused = {}, const std::function<void(size_t, bool)>& reclaim = {});
    void release_unused_tail();
    void fork(Table& dst, const Table& src, int tokens);
    void truncate(Table& table, int tokens);
    void release();
    void set_kv_vram_budget_bytes(size_t bytes);
    void set_kv_host_budget_bytes(size_t bytes);
    size_t vram_budget_bytes() const { return vram_budget_bytes_; }
    size_t host_budget_bytes() const { return host_budget_bytes_; }
    size_t bytes() const { return pages_ * page_bytes_; }
    int pages() const { return pages_; }
    size_t reserved_bytes() const { return physical_.size() * page_bytes_; }
    size_t host_bytes() const { return host_pages_ * page_bytes_; }
    int host_pages() const { return host_pages_; }

private:
    friend struct CudaReviewTestAccess;
    cudaError_t (*allocate_)(void**, size_t) = [](void** data, size_t bytes) { return cudaMalloc(data, bytes); };
    struct Page {
        uint8_t* data;
        size_t refs = 0;
        int next_free = -1;
    };
    using PinnedPage = std::unique_ptr<uint8_t, decltype(&cudaFreeHost)>;
    struct HostPage {
        PinnedPage data{nullptr, cudaFreeHost};
        size_t refs = 0;
        int next_free = -1;
    };
    bool grow(int count);
    int acquire();
    void drop(int id);
    using SlotPage = std::pair<Table*, int>;
    std::vector<SlotPage> select_spill_pages(std::span<Table* const> paused, int count) const;
    void spill_pages_to_host(CudaContext& ctx, std::span<const SlotPage> victims);
    void exchange_spilled_pages(CudaContext& ctx, Table& table, int requested, std::span<const SlotPage> victims);
    void restore_pages_from_host(CudaContext& ctx, Table& table, int requested);

    struct Slab {
        std::unique_ptr<uint8_t, decltype(&cudaFree)> data;
        int pages;
    };
    std::vector<Slab> slabs_;
    std::vector<Page> physical_;
    size_t page_bytes_ = 0;
    int pages_ = 0;
    int free_ = -1;
    size_t vram_budget_bytes_ = 0;
    size_t host_budget_bytes_ = 0;
    std::vector<HostPage> host_;
    int host_pages_ = 0;
    int host_free_ = -1;
};

void check_cuda(cudaError_t err, const char* expr, const char* file, int line);
void check_cu(CUresult err, const char* expr, const char* file, int line);

#define FUSIONRT_CUDA_CHECK(expr) ::fusionrt::check_cuda((expr), #expr, __FILE__, __LINE__)
#define FUSIONRT_CU_CHECK(expr) ::fusionrt::check_cu((expr), #expr, __FILE__, __LINE__)

}  // namespace fusionrt
