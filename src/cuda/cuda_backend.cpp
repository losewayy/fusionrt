#include "fusionrt/cuda_backend.h"
#include "fusionrt/weight_codec.h"
#include "../core/cpu_moe.h"
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

void CudaContext::set_kv_dtype(std::optional<DType> dtype) {
    if (dtype && dtype != DType::F16 && dtype != DType::F32 && dtype != DType::Q8_0) {
        throw std::invalid_argument("CUDA KV cache supports F16, F32 and Q8_0");
    }
    kv_request_ = dtype;
}

CudaKvView make_cuda_kv_view(CudaArena& arena, DType dtype, int max_tokens, int kv_heads, int head_dim,
                             bool latent) {
    if (dtype == DType::Q8_0 && head_dim % 32 != 0) {
        throw std::invalid_argument("Q8_0 KV requires a head dimension divisible by 32");
    }
    const int pages = 1 + (max_tokens - 1) / kKvPageTokens;
    const int elements = kKvPageTokens * kv_heads * head_dim;
    const size_t plane = size_t(elements) / dtype_block_size(dtype) * dtype_type_size(dtype);
    auto* table = static_cast<uint8_t**>(arena.alloc(pages * sizeof(uint8_t*)));
    return {table, plane, elements, pages, dtype, latent};
}

static int kv_pages_for(int tokens) { return tokens > 0 ? 1 + (tokens - 1) / kKvPageTokens : 0; }

void CudaKvPool::set_kv_vram_budget_bytes(size_t bytes) {
    if (bytes != 0 && bytes < reserved_bytes()) throw std::invalid_argument("KV VRAM budget is below existing allocation");
    vram_budget_bytes_ = bytes;
}

void CudaKvPool::set_kv_host_budget_bytes(size_t bytes) {
    if (bytes < host_bytes()) throw std::invalid_argument("Host KV budget is below existing allocation");
    host_budget_bytes_ = bytes;
}

bool CudaKvPool::grow(int count) {
    if (slabs_.size() == slabs_.capacity()) {
        slabs_.reserve(2 * slabs_.size() + 1);
    }
    const int old_size = static_cast<int>(physical_.size());
    if (physical_.size() + count > physical_.capacity()) {
        physical_.reserve(std::max(2 * physical_.size(), physical_.size() + count));
    }
    void* storage = nullptr;
    const auto status = allocate_(&storage, count * page_bytes_);
    if (status == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        return false;
    }
    FUSIONRT_CUDA_CHECK(status);
    auto* data = static_cast<uint8_t*>(storage);
    slabs_.push_back({{data, cudaFree}, count});
    for (int i = 0; i < count; ++i) {
        physical_.push_back({data + i * page_bytes_, 0, i + 1 < count ? old_size + i + 1 : free_});
    }
    free_ = old_size;
    return true;
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
    if (id < 0) {
        const int index = -id - 1;
        auto& page = host_[index];
        if (--page.refs == 0) {
            page.data.reset();
            page.next_free = host_free_;
            host_free_ = index;
            --host_pages_;
        }
        return;
    }
    auto& page = physical_[id];
    if (--page.refs == 0) {
        page.next_free = free_;
        free_ = id;
        --pages_;
    }
}

std::vector<CudaKvPool::SlotPage> CudaKvPool::select_spill_pages(std::span<Table* const> paused, int count) const {
    std::vector<SlotPage> victims;
    for (auto* table : paused) {
        for (int p = static_cast<int>(table->pages_.size()) - 1; p >= 0; --p) {
            const int id = table->pages_[p];
            if (id >= 0 && physical_[id].refs == 1) victims.emplace_back(table, p);
            if (victims.size() == static_cast<size_t>(count)) return victims;
        }
    }
    throw KvVramExhausted("KV VRAM exhausted with too few private pages in paused slots");
}

void CudaKvPool::spill_pages_to_host(CudaContext& ctx, std::span<const SlotPage> victims) {
    if (victims.empty()) return;
    host_.reserve(host_.size() + victims.size());
    std::vector<PinnedPage> batch;
    batch.reserve(victims.size());
    for (size_t i = 0; i < victims.size(); ++i) {
        uint8_t* data = nullptr;
        const auto status = cudaMallocHost(&data, page_bytes_);
        if (status == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            throw KvVramExhausted("Unable to allocate pinned host KV memory");
        }
        FUSIONRT_CUDA_CHECK(status);
        batch.emplace_back(data, cudaFreeHost);
    }
    try {
        for (size_t i = 0; i < victims.size(); ++i) {
            const auto [table, index] = victims[i];
            const auto& page = physical_[table->pages_[index]];
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(batch[i].get(), page.data, page_bytes_,
                                                cudaMemcpyDeviceToHost, ctx.stream()));
        }
        ctx.synchronize();
    } catch (...) {
        (void)cudaStreamSynchronize(ctx.stream());
        throw;
    }
    for (size_t i = 0; i < victims.size(); ++i) {
        const auto [table, index] = victims[i];
        int id = host_free_;
        if (id < 0) {
            id = static_cast<int>(host_.size());
            host_.emplace_back();
        } else {
            host_free_ = host_[id].next_free;
        }
        host_[id] = {std::move(batch[i]), 1, -1};
        ++host_pages_;
        drop(table->pages_[index]);
        table->pages_[index] = -id - 1;
        table->published_ = std::min(table->published_, index);
        ++table->spilled_;
    }
}

void CudaKvPool::exchange_spilled_pages(CudaContext& ctx, Table& table, int requested,
                                       std::span<const SlotPage> victims) {
    if (victims.empty()) return;
    std::vector<int> incoming;
    const int kept = std::min(requested, static_cast<int>(table.pages_.size()));
    for (int p = 0; p < kept && incoming.size() < victims.size(); ++p) {
        const int id = table.pages_[p];
        if (id < 0 && host_[-id - 1].refs == 1) incoming.push_back(p);
    }
    if (incoming.size() != victims.size()) {
        throw KvVramExhausted("KV VRAM exhausted and host KV budget cannot hold the spill");
    }
    // With both tiers full, reuse the incoming pinned pages for the outgoing pages.
    auto& arena = ctx.scratch_arena();
    const size_t mark = arena.mark();
    const size_t aligned = (mark + 255) & ~size_t{255};
    const size_t chunk = std::min(page_bytes_, arena.capacity() - std::min(arena.capacity(), aligned));
    if (chunk == 0) throw KvVramExhausted("No CUDA scratch available for a full-tier KV exchange");
    auto* scratch = arena.alloc(chunk);
    try {
        const auto stream = ctx.stream();
        for (size_t i = 0; i < victims.size(); ++i) {
            const auto [paused, index] = victims[i];
            auto* gpu = physical_[paused->pages_[index]].data;
            auto* host = host_[-table.pages_[incoming[i]] - 1].data.get();
            for (size_t offset = 0; offset < page_bytes_; offset += chunk) {
                const size_t bytes = std::min(chunk, page_bytes_ - offset);
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(scratch, gpu + offset, bytes, cudaMemcpyDeviceToDevice, stream));
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(gpu + offset, host + offset, bytes, cudaMemcpyHostToDevice, stream));
                FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(host + offset, scratch, bytes, cudaMemcpyDeviceToHost, stream));
            }
        }
        ctx.synchronize();
    } catch (...) {
        (void)cudaStreamSynchronize(ctx.stream());
        arena.reset(mark);
        throw;
    }
    arena.reset(mark);
    for (size_t i = 0; i < victims.size(); ++i) {
        const auto [paused, index] = victims[i];
        std::swap(table.pages_[incoming[i]], paused->pages_[index]);
        paused->published_ = std::min(paused->published_, index);
        ++paused->spilled_;
    }
    table.spilled_ -= static_cast<int>(incoming.size());
    table.published_ = std::min(table.published_, incoming.front());
}

void CudaKvPool::restore_pages_from_host(CudaContext& ctx, Table& table, int requested) {
    std::vector<std::pair<int, int>> batch;
    const int kept = std::min(requested, static_cast<int>(table.pages_.size()));
    for (int p = 0; p < kept; ++p) {
        if (table.pages_[p] < 0) batch.emplace_back(p, -1);
    }
    if (batch.empty()) return;
    try {
        for (auto& [index, id] : batch) {
            id = acquire();
            FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(physical_[id].data, host_[-table.pages_[index] - 1].data.get(),
                                                page_bytes_, cudaMemcpyHostToDevice, ctx.stream()));
        }
        ctx.synchronize();
    } catch (...) {
        (void)cudaStreamSynchronize(ctx.stream());
        for (const auto& entry : batch) {
            if (entry.second >= 0) drop(entry.second);
        }
        throw;
    }
    for (const auto& [index, id] : batch) {
        drop(table.pages_[index]);
        table.pages_[index] = id;
    }
    table.spilled_ -= static_cast<int>(batch.size());
    table.published_ = std::min(table.published_, batch.front().first);
}

void CudaKvPool::ensure(CudaContext& ctx, std::span<const CudaKvView> layers, Table& table,
                        int first_write, int tokens, std::span<Table* const> paused,
                        const std::function<void(size_t, bool)>& reclaim) {
    if (tokens <= 0 || layers.empty()) return;
    const int requested = kv_pages_for(tokens);
    const int existing = static_cast<int>(table.pages_.size());
    const int first = first_write < tokens ? first_write / kKvPageTokens : requested;
    int needed = std::max(0, requested - existing);
    for (int p = table.spilled_ > 0 ? 0 : first; p < std::min(requested, existing); ++p) {
        const int id = table.pages_[p];
        needed += id < 0 || (p >= first && physical_[id].refs > 1);
    }
    if (requested > layers.front().max_pages) throw std::invalid_argument("KV position exceeds context");
    if (page_bytes_ == 0) {
        for (const auto& layer : layers) page_bytes_ += layer.page_bytes();
    }
    if (vram_budget_bytes_ && static_cast<size_t>(requested) > vram_budget_bytes_ / page_bytes_)
        throw std::invalid_argument("Active KV needs " + std::to_string(requested * page_bytes_) +
            " bytes, exceeding the " + std::to_string(vram_budget_bytes_) + "-byte KV budget");
    const int missing = needed - (static_cast<int>(physical_.size()) - pages_);
    if (missing > 0) {
        const int growth = vram_budget_bytes_ == 0 ? missing : static_cast<int>(
            std::min(static_cast<size_t>(missing), (vram_budget_bytes_ - reserved_bytes()) / page_bytes_));
        int to_spill = missing - growth;
        const auto target = reserved_bytes() + growth * page_bytes_;
        if (growth > 0 && reclaim) reclaim(target, false);
        if (growth > 0 && !grow(growth)) {
            int remaining = growth;
            if (reclaim) {
                reclaim(target, true);
                if (grow(growth)) remaining = 0;
            }
            // A single large slab can fail even when smaller allocations still fit.
            for (int chunk = growth / 2; chunk > 0 && remaining > 0; chunk /= 2) {
                while (chunk <= remaining && grow(chunk)) remaining -= chunk;
            }
            to_spill += remaining;
        }
        if (to_spill > 0) {
            const auto victims = select_spill_pages(paused, to_spill);
            const std::span<const SlotPage> batch(victims);
            const size_t host_free = (host_budget_bytes_ - host_bytes()) / page_bytes_;
            const size_t spill = std::min(host_free, batch.size());
            exchange_spilled_pages(ctx, table, requested, batch.subspan(spill));
            spill_pages_to_host(ctx, batch.first(spill));
        }
    }
    if (static_cast<size_t>(requested) > table.pages_.capacity()) {
        table.pages_.reserve(std::max(static_cast<size_t>(requested), 2 * table.pages_.size()));
    }
    if (table.spilled_ > 0) restore_pages_from_host(ctx, table, requested);
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
        if (!layers[i].pages) continue;  // Shared KV layers have no page table.
        auto* entry = entries.data() + i * added;
        for (int p = 0; p < added; ++p) entry[p] = physical_[table.pages_[table.published_ + p]].data + offset;
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(layers[i].pages + table.published_, entry, added * sizeof(*entry),
                                            cudaMemcpyHostToDevice, ctx.stream()));
        offset += layers[i].page_bytes();
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
    int spilled = 0;
    for (int id : prefix) {
        if (id < 0) {
            ++host_[-id - 1].refs;
            ++spilled;
        } else {
            ++physical_[id].refs;
        }
    }
    truncate(dst, 0);
    dst.pages_ = std::move(prefix);
    dst.spilled_ = spilled;
}

void CudaKvPool::truncate(Table& table, int tokens) {
    const int kept = std::min(kv_pages_for(tokens), static_cast<int>(table.pages_.size()));
    while (table.pages_.size() > static_cast<size_t>(kept)) {
        table.spilled_ -= table.pages_.back() < 0;
        drop(table.pages_.back());
        table.pages_.pop_back();
    }
    table.published_ = std::min(table.published_, kept);
}

void CudaKvPool::release_unused_tail() {
    const auto prior = physical_.size();
    while (!slabs_.empty()) {
        const int n = slabs_.back().pages;
        if (std::any_of(physical_.end() - n, physical_.end(), [](const auto& p) { return p.refs != 0; }))
            break;
        physical_.resize(physical_.size() - n);
        slabs_.pop_back();
    }
    if (prior == physical_.size()) return;
    free_ = -1;
    for (int i = static_cast<int>(physical_.size()) - 1; i >= 0; --i) {
        if (physical_[i].refs != 0) continue;
        physical_[i].next_free = free_;
        free_ = i;
    }
}

void CudaKvPool::release() {
    slabs_.clear();
    physical_.clear();
    pages_ = 0;
    page_bytes_ = 0;
    free_ = -1;
    host_.clear();
    host_pages_ = 0;
    host_free_ = -1;
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
    CudaGraphExec next;
    FUSIONRT_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
        fn();
    } catch (...) {
        cudaGraph_t aborted = nullptr;
        cudaStreamEndCapture(stream, &aborted);
        if (aborted) cudaGraphDestroy(aborted);
        throw;
    }
    FUSIONRT_CUDA_CHECK(cudaStreamEndCapture(stream, &next.graph_));
    FUSIONRT_CUDA_CHECK(instantiate_(&next.exec_, next.graph_));
    std::swap(graph_, next.graph_);
    std::swap(exec_, next.exec_);
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
        info.shared_mem_per_block_optin = static_cast<int>(prop.sharedMemPerBlockOptin);
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
    set_device_profile({});
    FUSIONRT_CUDA_CHECK(cudaSetDevice(device_id));
    cudaStream_t stream_handle = nullptr;
    FUSIONRT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_handle, cudaStreamNonBlocking));
    stream_.reset(stream_handle);
    cublasHandle_t cublas_handle = nullptr;
    FUSIONRT_CUBLAS_CHECK(cublasCreate(&cublas_handle));
    cublas_.reset(cublas_handle);
    FUSIONRT_CUBLAS_CHECK(cublasSetStream(cublas(), stream()));
    FUSIONRT_CUBLAS_CHECK(cublasSetMathMode(cublas(), CUBLAS_DEFAULT_MATH));
    weight_arena_.init(weight_arena_mb * 1024ULL * 1024ULL);
    scratch_arena_.init(scratch_arena_mb * 1024ULL * 1024ULL);

    // One-time cuBLAS heuristic warmup using scratch arena so first prefill pays 0 init cost
    if (scratch_arena_.capacity() >= 4096) {
        size_t mark = scratch_arena_.mark();
        void* a = scratch_arena_.alloc(16 * 16 * sizeof(uint16_t));
        void* b = scratch_arena_.alloc(16 * 16 * sizeof(uint16_t));
        void* c = scratch_arena_.alloc(16 * 16 * sizeof(float));
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(a, 0, 512, stream()));
        FUSIONRT_CUDA_CHECK(cudaMemsetAsync(b, 0, 512, stream()));
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas(), CUBLAS_OP_T, CUBLAS_OP_N, 16, 16, 16, &alpha,
                     a, CUDA_R_16F, 16, b, CUDA_R_16F, 16, &beta, c, CUDA_R_32F, 16,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        scratch_arena_.reset(mark);
    }

    // Allocate L2 flush buffer >= 2x device L2 (or at least 64 MiB)
    l2_flush_bytes_ = std::max<size_t>(info_.l2_cache_bytes * 2, 64ULL * 1024ULL * 1024ULL);
    FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream()));
}

void CudaContext::set_device_profile(const DeviceProfile& profile) {
    device_profile_ = profile;
    auto& p = device_profile_;
    if (p.sm_count == 0) p.sm_count = info_.multi_processor_count;
    if (p.vram_total_mb == 0) p.vram_total_mb = static_cast<double>(info_.total_vram_bytes) / (1024 * 1024);
    if (p.reported_l2_bytes == 0) p.reported_l2_bytes = info_.l2_cache_bytes;

    // Dense FP16 FMA throughput per SM cycle. INT8 issues twice as many MACs.
    const int cc = 10 * info_.sm_major + info_.sm_minor;
    int fp16_fmas = 512;
    if (cc == 80) fp16_fmas = 1024;
    if (cc == 90 || cc == 100 || cc == 103) fp16_fmas = 2048;
    const double peak_fp16 = 2.0 * fp16_fmas * p.sm_count * info_.sm_clock_khz / 1e9;
    cost_hardware_ = {
        .sm_count = p.sm_count,
        .hbm_bw_gbps = p.measured_hbm_bw_gbps > 0 ? p.measured_hbm_bw_gbps : info_.peak_bandwidth_gbps,
        .int8_tc_tflops = p.measured_int8_tc_tflops > 0 ? p.measured_int8_tc_tflops : 2 * peak_fp16,
        .fp16_tc_tflops = p.measured_fp16_tc_tflops > 0 ? p.measured_fp16_tc_tflops : peak_fp16,
        .l2_bytes = static_cast<size_t>(p.empirical_l2_bytes > 0 ? p.empirical_l2_bytes : p.reported_l2_bytes),
        .cpu_q4_gemv_gbps = p.cpu_q4_gemv_gbps,
        .pcie_h2d_gbps = p.pcie_h2d_gbps,
        .pcie_latency_us = p.pcie_latency_us,
        .l2_bw_gbps = p.measured_l2_bw_gbps,
        .sm_clock_ghz = info_.sm_clock_khz / 1e6,
        .shared_bytes_per_sm = info_.shared_mem_per_sm,
        .shared_bytes_per_block = std::max(info_.shared_mem_per_block_optin, info_.shared_mem_per_block),
    };
}

CudaContext::~CudaContext() {
    if (moe_input_ready_) {
        (void)cudaStreamSynchronize(stream());
        cudaEventDestroy(moe_input_ready_);
    }
    if (l2_flush_buf_) cudaFree(l2_flush_buf_);
    scratch_arena_.release();
    weight_arena_.release();
}

float* CudaContext::moe_transfer_buffer(size_t bytes) {
    if (bytes > moe_transfer_bytes_) {
        if (moe_transfer_) synchronize();
        if (!moe_input_ready_) {
            FUSIONRT_CUDA_CHECK(cudaEventCreateWithFlags(&moe_input_ready_, cudaEventDisableTiming));
        }
        float* data = nullptr;
        FUSIONRT_CUDA_CHECK(cudaMallocHost(&data, bytes));
        moe_transfer_.reset(data);
        moe_transfer_bytes_ = bytes;
    }
    return moe_transfer_.get();
}

Tensor CudaContext::upload(const HostTensor& ht, bool persistent) {
    CudaArena& arena = persistent ? weight_arena_ : scratch_arena_;
    Tensor dst = arena.alloc_tensor(
        std::vector<int64_t>(ht.desc.shape.begin(), ht.desc.shape.begin() + ht.desc.ndim),
        ht.desc.dtype, ht.desc.name);
    upload_into(ht, dst);
    return dst;
}

void CudaContext::upload_into(const HostTensor& ht, Tensor& dst) {
    if (ht.desc.nbytes() != dst.nbytes()) {
        throw std::runtime_error("upload_into size mismatch");
    }
    dst.codec_tile_scale = 1.f;
    const auto type = ht.desc.dtype;
    if (is_weight_codec_dtype(type) && !has_weight_codec_int8(type)) {
        const size_t stride = dtype_type_size(type);
        const size_t scale_offset = type == DType::Q3_K ? 108 : 0;
        constexpr float coefficient_bound = 2048.f;
        const auto packed = ht.bytes();
        for (size_t block = 0; block < packed.size(); block += stride) {
            uint16_t bits;
            std::memcpy(&bits, packed.data() + block + scale_offset, sizeof(bits));
            if ((bits & 0x7fffu) > fp32_to_fp16(65504.f / coefficient_bound)) {
                dst.codec_tile_scale = coefficient_bound;
                break;
            }
        }
    }
    if (ht.desc.nbytes() > 0) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst.ptr(), ht.bytes().data(), ht.desc.nbytes(),
                                            cudaMemcpyHostToDevice, stream()));
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream()));
    }
}

HostTensor CudaContext::download(const Tensor& t) {
    std::vector<int64_t> dims(t.shape.begin(), t.shape.begin() + t.ndim);
    HostTensor ht = HostTensor::create(dims, t.dtype);
    ht.desc.name = t.name;
    if (t.nbytes() > 0) {
        FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(ht.storage.data(), t.ptr(), t.nbytes(),
                                            cudaMemcpyDeviceToHost, stream()));
        FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream()));
    }
    return ht;
}

void CudaContext::set_cpu_threads(int threads, bool spread_affinity) {
    if (threads < 0) throw std::invalid_argument("Expected --threads >= 0");
    if (cpu_threads_ != threads || cpu_spread_affinity_ != spread_affinity) {
        cpu_workers_.reset();
        cpu_threads_ = threads;
        cpu_spread_affinity_ = spread_affinity;
    }
}

CpuWorkers& CudaContext::cpu_workers() {
    if (!cpu_workers_) cpu_workers_ = std::make_unique<CpuWorkers>(cpu_threads_, cpu_spread_affinity_);
    return *cpu_workers_;
}

CpuMoeWorkspace& CudaContext::cpu_moe_workspace() {
    if (!cpu_moe_workspace_) cpu_moe_workspace_ = std::make_unique<CpuMoeWorkspace>();
    return *cpu_moe_workspace_;
}

void CudaContext::synchronize() {
    FUSIONRT_CUDA_CHECK(cudaStreamSynchronize(stream()));
}

void CudaContext::flush_l2() {
    if (!l2_flush_buf_) FUSIONRT_CUDA_CHECK(cudaMalloc(&l2_flush_buf_, l2_flush_bytes_));
    FUSIONRT_CUDA_CHECK(cudaMemsetAsync(l2_flush_buf_, 0x5A, l2_flush_bytes_, stream()));
}

CudaMoeTiming::~CudaMoeTiming() {
    for (const auto& layer : layers_) {
        if (layer.gpu_done) cudaEventDestroy(layer.gpu_done);
        if (layer.cpu_done) cudaEventDestroy(layer.cpu_done);
    }
}

void CudaMoeTiming::begin_gap(Layer& layer, cudaStream_t stream) {
    if (!layer.gpu_done) {
        FUSIONRT_CUDA_CHECK(cudaEventCreate(&layer.gpu_done));
        FUSIONRT_CUDA_CHECK(cudaEventCreate(&layer.cpu_done));
    }
    FUSIONRT_CUDA_CHECK(cudaEventRecord(layer.gpu_done, stream));
}

void CudaMoeTiming::end_gap(Layer& layer, cudaStream_t stream) {
    FUSIONRT_CUDA_CHECK(cudaEventRecord(layer.cpu_done, stream));
    layer.pending = true;
}

void CudaMoeTiming::collect() {
    for (auto& layer : layers_) {
        if (!layer.pending) continue;
        float elapsed = 0;
        FUSIONRT_CUDA_CHECK(cudaEventElapsedTime(&elapsed, layer.gpu_done, layer.cpu_done));
        layer.gpu_gap_ms += elapsed;
        layer.pending = false;
    }
}

}  // namespace fusionrt
