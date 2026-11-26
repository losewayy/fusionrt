#include "fusionrt/cuda_backend.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace fusionrt {

static void check_cublas(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuBLAS error at " << file << ":" << line << " (" << expr << "): status="
            << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}
#define FUSIONRT_CUBLAS_CHECK(x) check_cublas((x), #x, __FILE__, __LINE__)

void check_cuda(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << " (" << expr << "): "
            << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

void check_cu(CUresult err, const char* expr, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        std::ostringstream oss;
        oss << "CUDA Driver error at " << file << ":" << line << " (" << expr << "): "
            << (msg ? msg : "unknown");
        throw std::runtime_error(oss.str());
    }
}

void CudaContext::set_kv_dtype(DType dtype) {
    if (dtype != DType::F16 && dtype != DType::F32 && dtype != DType::Q8_0) {
        throw std::invalid_argument("CUDA KV cache supports F16, F32 and Q8_0");
    }
    kv_dtype_ = dtype;
}

CudaKvView make_cuda_kv_view(CudaArena& arena, DType dtype, int max_tokens, int kv_heads, int head_dim) {
    if (dtype == DType::Q8_0 && head_dim % 32 != 0) {
        throw std::invalid_argument("Q8_0 KV requires a head dimension divisible by 32");
    }
    const int pages = 1 + (max_tokens - 1) / kKvPageTokens;
    const int elements = kKvPageTokens * kv_heads * head_dim;
    const size_t plane = size_t(elements) / dtype_block_size(dtype) * dtype_type_size(dtype);
    auto* table = static_cast<uint8_t**>(arena.alloc(pages * sizeof(uint8_t*)));
    return {table, plane, elements, pages, dtype};
}

static int kv_pages_for(int tokens) { return tokens > 0 ? 1 + (tokens - 1) / kKvPageTokens : 0; }

void CudaKvPool::grow(int count) {
    if (slabs_.size() == slabs_.capacity()) {
        slabs_.reserve(2 * slabs_.size() + 1);
    }
    const int old_size = static_cast<int>(physical_.size());
    if (physical_.size() + count > physical_.capacity()) {
        physical_.reserve(std::max(2 * physical_.size(), physical_.size() + count));
    }
    uint8_t* data = nullptr;
    FUSIONRT_CUDA_CHECK(cudaMalloc(&data, count * page_bytes_));
    slabs_.emplace_back(data, cudaFree);
    for (int i = 0; i < count; ++i) {
        physical_.push_back({data + i * page_bytes_, 0, i + 1 < count ? old_size + i + 1 : free_});
    }
    free_ = old_size;
}

int CudaKvPool::acquire() {
    const int id = free_;
    auto& page = physical_[id];
    free_ = page.next_free;
    page.refs = 1;
    ++pages_;
    return id;
}

void CudaKvPool::drop(int id) {
    auto& page = physical_[id];
    if (--page.refs == 0) {
        page.next_free = free_;
        free_ = id;
        --pages_;
    }
}

void CudaKvPool::ensure(CudaContext& ctx, std::span<const CudaKvView> layers, Table& table,
                        int first_write, int tokens) {
    if (tokens <= 0 || layers.empty()) return;
    const int requested = kv_pages_for(tokens);
    const int existing = static_cast<int>(table.pages_.size());
    const int first = first_write < tokens ? first_write / kKvPageTokens : requested;
    int needed = std::max(0, requested - existing);
    for (int p = first; p < std::min(requested, existing); ++p) {
        needed += physical_[table.pages_[p]].refs > 1;
    }
    if (requested > layers.front().max_pages) throw std::invalid_argument("KV position exceeds context");
    if (page_bytes_ == 0) {
        for (const auto& layer : layers) page_bytes_ += 2 * layer.plane_bytes;
    }
    const int missing = needed - (static_cast<int>(physical_.size()) - pages_);
    if (missing > 0) grow(missing);
    if (static_cast<size_t>(requested) > table.pages_.capacity()) {
        table.pages_.reserve(std::max(static_cast<size_t>(requested), 2 * table.pages_.size()));
    }
    for (int p = std::min(first, existing); p < requested; ++p) {
        const int old = p < existing ? table.pages_[p] : -1;
        if (old >= 0 && physical_[old].refs == 1) continue;
        const int id = acquire();
        auto* data = physical_[id].data;
        if (old >= 0) {
            table.pages_[p] = id;
            table.published_ = std::min(table.published_, p);
            drop(old);
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(data, physical_[old].data, page_bytes_,
                                                cudaMemcpyDeviceToDevice, ctx.stream()));
        } else {
            table.pages_.push_back(id);
            FUSIONRT_CUDA_CHECK(cudaMemsetAsync(data, 0, page_bytes_, ctx.stream()));
        }
    }
    const int added = requested - table.published_;
    if (added <= 0) return;
    std::vector<uint8_t*> entries(layers.size() * added);
    size_t offset = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto* entry = entries.data() + i * added;
        for (int p = 0; p < added; ++p) entry[p] = physical_[table.pages_[table.published_ + p]].data + offset;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(layers[i].pages + table.published_, entry, added * sizeof(*entry),
                                            cudaMemcpyHostToDevice, ctx.stream()));
        offset += 2 * layers[i].plane_bytes;
    }
    ctx.synchronize();
    table.published_ = requested;
}

void CudaKvPool::fork(Table& dst, const Table& src, int tokens) {
    const int kept = std::min(kv_pages_for(tokens), static_cast<int>(src.pages_.size()));
    if (&dst == &src) {
        truncate(dst, tokens);
        return;
    }
    std::vector<int> prefix(src.pages_.begin(), src.pages_.begin() + kept);
    for (int id : prefix) ++physical_[id].refs;
    truncate(dst, 0);
    dst.pages_ = std::move(prefix);
}

void CudaKvPool::truncate(Table& table, int tokens) {
    const int kept = std::min(kv_pages_for(tokens), static_cast<int>(table.pages_.size()));
    while (table.pages_.size() > static_cast<size_t>(kept)) {
        drop(table.pages_.back());
        table.pages_.pop_back();
    }
    table.published_ = std::min(table.published_, kept);
}

void CudaKvPool::release() {
    slabs_.clear();
    physical_.clear();
    pages_ = 0;
    page_bytes_ = 0;
    free_ = -1;
}

CudaArena::~CudaArena() {
    release();
}

void CudaArena::init(size_t capacity_bytes) {
    release();
    if (capacity_bytes == 0) return;
    FUSIONRT_CUDA_CHECK(cudaMalloc(&base_, capacity_bytes));
    FUSIONRT_CUDA_CHECK(cudaMemset(base_, 0, capacity_bytes));
    // Uploads use a non-blocking stream and do not wait for the default-stream memset.
    FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    capacity_ = capacity_bytes;
    offset_ = 0;
}

void CudaArena::release() {
    if (base_) {
        cudaFree(base_);
        base_ = nullptr;
    }
    capacity_ = 0;
    offset_ = 0;
}

void* CudaArena::alloc(size_t bytes, size_t alignment) {
    size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes > capacity_) {
        std::ostringstream oss;
        oss << "CudaArena OOM: requested " << bytes << " bytes at offset " << aligned
            << " exceeds capacity " << capacity_;
        throw std::runtime_error(oss.str());
    }
    void* ptr = static_cast<uint8_t*>(base_) + aligned;
    offset_ = aligned + bytes;
    if (track_peak_) peak_ = std::max(peak_, offset_);
    return ptr;
}

Tensor CudaArena::alloc_tensor(const std::vector<int64_t>& shape, DType dt, const std::string& name) {
    Tensor t = Tensor::from_shape(shape, dt, Backend::CUDA);
    t.name = name;
    t.data = alloc(t.nbytes());
    return t;
}

CudaGraphExec::~CudaGraphExec() {
    reset();
}

void CudaGraphExec::reset() {
    if (exec_) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

void CudaGraphExec::capture(cudaStream_t stream, const std::function<void()>& fn) {
    reset();
    FUSIONRT_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        fn();
    } catch (...) {
        cudaGraph_t aborted = nullptr;
        cudaStreamEndCapture(stream, &aborted);
        if (aborted) cudaGraphDestroy(aborted);
        throw;
    }
    FUSIONRT_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    FUSIONRT_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
}

void CudaGraphExec::launch(cudaStream_t stream) {
    if (!exec_) throw std::runtime_error("CudaGraphExec::launch called before capture");
    FUSIONRT_CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

std::vector<CudaDeviceInfo> CudaContext::enumerate_devices() {
    std::vector<CudaDeviceInfo> out;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return out;
    }
    int driver_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    std::ostringstream drv_ss;
    drv_ss << (driver_ver / 1000) << "." << ((driver_ver % 1000) / 10);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        CudaDeviceInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.sm_major = prop.major;
        info.sm_minor = prop.minor;
        info.total_vram_bytes = prop.totalGlobalMem;
        info.l2_cache_bytes = static_cast<size_t>(prop.l2CacheSize);
        info.multi_processor_count = prop.multiProcessorCount;
        info.shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
        info.shared_mem_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
        info.warp_size = prop.warpSize;
        info.max_threads_per_block = prop.maxThreadsPerBlock;
        info.memory_bus_width_bits = prop.memoryBusWidth;

        int mem_clock_khz = 0;
        int sm_clock_khz = 0;
        cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, i);
        cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, i);
        info.mem_clock_khz = static_cast<double>(mem_clock_khz);
        info.sm_clock_khz = static_cast<double>(sm_clock_khz);
        // DDR/GDDR effective rate = 2 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9
        info.peak_bandwidth_gbps = 2.0 * info.mem_clock_khz * 1e3 * (info.memory_bus_width_bits / 8.0) / 1e9;
        info.driver_version = drv_ss.str();
        out.push_back(std::move(info));
    }
    return out;
}

CudaContext::CudaContext(int device_id, size_t weight_arena_mb, size_t scratch_arena_mb) {
    FUSIONRT_CU_CHECK(cuInit(0));
    auto devs = enumerate_devices();
    if (device_id < 0 || device_id >= static_cast<int>(devs.size())) {
        throw std::runtime_error("Invalid CUDA device_id");
    }
    info_ = devs[device_id];
    FUSIONRT_CUDA_CHECK(cudaSetDevice(device_id));
    FUSIONRT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    FUSIONRT_CUBLAS_CHECK(cublasCreate(&cublas_));
    FUSIONRT_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    FUSIONRT_CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));
    weight_arena_.init(weight_arena_mb * 1024ULL * 1024ULL);
    scratch_arena_.init(scratch_arena_mb * 1024ULL * 1024ULL);

    // One-time cuBLAS heuristic warmup using scratch arena so first prefill pays 0 init cost
    if (scratch_arena_.capacity() >= 4096) {
        size_t mark = scratch_arena_.mark();
        void* a = scratch_arena_.alloc(16 * 16 * sizeof(uint16_t));
        void* b = scratch_arena_.alloc(16 * 16 * sizeof(uint16_t));
        void* c = scratch_arena_.alloc(16 * 16 * sizeof(float));
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(a, 0, 512, stream_));
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(b, 0, 512, stream_));
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 16, 16, 16, &alpha,
                     a, CUDA_R_16F, 16, b, CUDA_R_16F, 16, &beta, c, CUDA_R_32F, 16,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        scratch_arena_.reset(mark);
    }

    // Allocate L2 flush buffer >= 2x device L2 (or at least 64 MiB)
    l2_flush_bytes_ = std::max<size_t>(info_.l2_cache_bytes * 2, 64ULL * 1024ULL * 1024ULL);
    FUSIONRT_CUDA_CHECK(cudaMalloc(&l2_flush_buf_, l2_flush_bytes_));
    FUSIONRT_CUDA_CHECK(cudaMemsetAsync(l2_flush_buf_, 0xA5, l2_flush_bytes_, stream_));
    FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

CudaContext::~CudaContext() {
    if (l2_flush_buf_) cudaFree(l2_flush_buf_);
    scratch_arena_.release();
    weight_arena_.release();
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
}

Tensor CudaContext::upload(const HostTensor& ht, bool persistent) {
    CudaArena& arena = persistent ? weight_arena_ : scratch_arena_;
    Tensor dst = arena.alloc_tensor(
        std::vector<int64_t>(ht.desc.shape.begin(), ht.desc.shape.begin() + ht.desc.ndim),
        ht.desc.dtype, ht.desc.name);
    if (ht.desc.nbytes() > 0) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst.ptr(), ht.storage.data(), ht.desc.nbytes(),
                                            cudaMemcpyHostToDevice, stream_));
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    return dst;
}

void CudaContext::upload_into(const HostTensor& ht, Tensor& dst) {
    if (ht.desc.nbytes() != dst.nbytes()) {
        throw std::runtime_error("upload_into size mismatch");
    }
    if (ht.desc.nbytes() > 0) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst.ptr(), ht.storage.data(), ht.desc.nbytes(),
                                            cudaMemcpyHostToDevice, stream_));
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

HostTensor CudaContext::download(const Tensor& t) {
    std::vector<int64_t> dims(t.shape.begin(), t.shape.begin() + t.ndim);
    HostTensor ht = HostTensor::create(dims, t.dtype);
    ht.desc.name = t.name;
    if (t.nbytes() > 0) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ht.storage.data(), t.ptr(), t.nbytes(),
                                            cudaMemcpyDeviceToHost, stream_));
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    return ht;
}

void CudaContext::synchronize() {
    FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaContext::flush_l2() {
    FUSIONRT_CUDA_CHECK(cudaMemsetAsync(l2_flush_buf_, 0x5A, l2_flush_bytes_, stream_));
}

}  // namespace fusionrt
