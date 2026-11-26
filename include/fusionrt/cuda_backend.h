#pragma once

#include "fusionrt/tensor.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fusionrt {

struct CudaDeviceInfo {
    int device_id = 0;
    std::string name;
    int sm_major = 0;
    int sm_minor = 0;
    size_t total_vram_bytes = 0;
    size_t l2_cache_bytes = 0;
    int multi_processor_count = 0;
    int shared_mem_per_block = 0;
    int shared_mem_per_sm = 0;
    int warp_size = 32;
    int max_threads_per_block = 1024;
    double mem_clock_khz = 0.0;
    double sm_clock_khz = 0.0;
    int memory_bus_width_bits = 0;
    double peak_bandwidth_gbps = 0.0;
    std::string driver_version;
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
    int mmq_crossover_m() const { return mmq_crossover_m_; }
    void set_mmq_crossover_m(int rows) { mmq_crossover_m_ = rows; }
    GemvGeometry preferred_gemv_geometry() const { return preferred_gemv_geometry_; }
    void set_preferred_gemv_geometry(GemvGeometry geometry) { preferred_gemv_geometry_ = geometry; }
    uint64_t empirical_l2_bytes() const { return empirical_l2_bytes_; }
    void set_empirical_l2_bytes(uint64_t bytes) { empirical_l2_bytes_ = bytes; }
    DType kv_dtype() const { return kv_dtype_; }
    void set_kv_dtype(DType dtype);

    CudaArena& weight_arena() { return weight_arena_; }
    CudaArena& scratch_arena() { return scratch_arena_; }

    Tensor upload(const HostTensor& ht, bool persistent = true);
    void upload_into(const HostTensor& ht, Tensor& dst);
    HostTensor download(const Tensor& t);
    void synchronize();

    // L2 cache flush buffer (> 2x device L2) for anti-noise benchmarking
    void flush_l2();

    // Driver API module/device query helpers
    static std::vector<CudaDeviceInfo> enumerate_devices();

private:
    CudaDeviceInfo info_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    CudaArena weight_arena_;
    CudaArena scratch_arena_;
    void* l2_flush_buf_ = nullptr;
    size_t l2_flush_bytes_ = 0;
    int mmq_crossover_m_ = 4;
    int prefill_batch_ = 512;
    GemvGeometry preferred_gemv_geometry_;
    uint64_t empirical_l2_bytes_ = 0;
    DType kv_dtype_ = DType::F32;
};

inline constexpr int kKvPageTokens = 128;

struct CudaKvView {
    uint8_t** pages = nullptr;
    size_t plane_bytes = 0;   // K or V in one page, including the Q8 scale plane.
    int elements = 0;         // Elements in one plane before quantization.
    int max_pages = 0;
    DType dtype = DType::F32;
};

CudaKvView make_cuda_kv_view(CudaArena& arena, DType dtype, int max_tokens, int kv_heads, int head_dim);

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
    };

    void ensure(CudaContext& ctx, std::span<const CudaKvView> layers, Table& table, int first_write, int tokens);
    void fork(Table& dst, const Table& src, int tokens);
    void truncate(Table& table, int tokens);
    void release();
    size_t bytes() const { return pages_ * page_bytes_; }
    int pages() const { return pages_; }
    size_t reserved_bytes() const { return physical_.size() * page_bytes_; }

private:
    struct Page {
        uint8_t* data;
        size_t refs = 0;
        int next_free = -1;
    };
    void grow(int count);
    int acquire();
    void drop(int id);

    std::vector<std::unique_ptr<uint8_t, decltype(&cudaFree)>> slabs_;
    std::vector<Page> physical_;
    size_t page_bytes_ = 0;
    int pages_ = 0;
    int free_ = -1;
};

void check_cuda(cudaError_t err, const char* expr, const char* file, int line);
void check_cu(CUresult err, const char* expr, const char* file, int line);

#define FUSIONRT_CUDA_CHECK(expr) ::fusionrt::check_cuda((expr), #expr, __FILE__, __LINE__)
#define FUSIONRT_CU_CHECK(expr) ::fusionrt::check_cu((expr), #expr, __FILE__, __LINE__)

}  // namespace fusionrt
