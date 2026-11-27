#pragma once

#include "fusionrt/device_profile.h"
#include "fusionrt/tensor.h"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusionrt {

struct OpTrace;
class VkContext;

struct VkCoopMatConfig {
    bool supported = false;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    bool fp16_in_fp32_acc = false;
    bool fp16_in_fp16_acc = false;
};

struct VkDeviceInfo {
    uint32_t device_index = 0;
    std::string name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    std::string device_type;  // discrete, integrated, cpu, other
    uint32_t api_version = 0;
    uint32_t subgroup_size = 32;
    uint32_t min_subgroup_size = 32;
    uint32_t max_subgroup_size = 32;
    uint32_t shared_mem_bytes = 0;
    uint64_t device_local_vram_bytes = 0;
    uint64_t min_storage_offset_align = 256;
    float timestamp_period_ns = 1.0f;
    bool has_fp16_int8 = false;
    bool has_16bit_storage = false;
    bool has_8bit_storage = false;
    bool has_coop_matrix = false;
    VkCoopMatConfig coop_mat;
};

struct VkBufferAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocation_size = 0;
    uint32_t heap_index = 0;
    void* mapped = nullptr;
};

class VkArena {
public:
    using Preflight = std::function<void(uint32_t, VkDeviceSize)>;
    VkArena() = default;
    ~VkArena();
    VkArena(const VkArena&) = delete;
    VkArena& operator=(const VkArena&) = delete;

    void init(VkDevice device, VkPhysicalDevice phys, VkDeviceSize capacity_bytes,
              VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props, VkDeviceSize align,
              const Preflight& preflight = {});
    void release(VkDevice device);

    VkDeviceSize alloc(VkDeviceSize bytes);
    Tensor alloc_tensor(const std::vector<int64_t>& shape, DType dt, const std::string& name = "");

    VkDeviceSize mark() const { return offset_; }
    void reset(VkDeviceSize mark = 0) { offset_ = mark; }
    VkBuffer buffer() const { return alloc_.buffer; }
    VkDeviceSize capacity() const { return alloc_.size; }
    VkDeviceSize used() const { return offset_; }
    void* mapped() const { return alloc_.mapped; }

private:
    friend class VkContext;
    VkContext* owner_ = nullptr;
    VkBufferAllocation alloc_;
    VkDeviceSize offset_ = 0;
    VkDeviceSize align_ = 256;
};

struct VkPipelineEntry {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t num_bindings = 0;
    uint32_t read_only_bindings = 0;
};

struct VkBatchDispatch {
    std::string shader_name;
    uint32_t gx = 1, gy = 1, gz = 1;
    std::vector<Tensor> buffers;
    const void* push_data = nullptr;
    uint32_t push_size = 0;
    std::vector<VkBufferCopy> copies;  // Relative to buffers[0] and buffers[1].
};

class VkContext {
public:
    explicit VkContext(uint32_t device_index = 0, size_t weight_arena_mb = 256, size_t scratch_arena_mb = 256);
    ~VkContext();
    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    const VkDeviceInfo& info() const { return info_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }

    bool supports_coopmat() const { return info_.coop_mat.supported && info_.has_fp16_int8; }
    uint32_t subgroup_size() const { return info_.subgroup_size; }
    void set_device_profile(const DeviceProfile& profile) { profile_ = profile; }
    const DeviceProfile& device_profile() const { return profile_; }
    void set_kv_dtype(std::optional<DType> dtype);
    std::optional<DType> kv_dtype() const { return kv_dtype_; }
    void set_prefill_batch(int rows);
    int prefill_batch() const { return prefill_batch_; }
    void set_op_trace(OpTrace* trace) { op_trace_ = trace; }
    OpTrace* op_trace() const { return op_trace_; }

    VkArena& weight_arena() { return weight_arena_; }
    void reserve_model_memory(VkDeviceSize weight_bytes, VkDeviceSize context_bytes);
    bool try_reserve_scratch_memory(VkDeviceSize bytes);
    VkArena& scratch_arena() { return scratch_arena_; }

    Tensor upload(const HostTensor& ht, bool persistent = true);
    HostTensor download(const Tensor& t);
    // Clear nonempty, 4-byte-aligned tensor ranges in a single synchronous submission.
    void zero_buffers(std::span<const Tensor> tensors);
    void synchronize();

    // Load SPIR-V shader pipeline on demand (with descriptor pool caching)
    const VkPipelineEntry& get_pipeline(const std::string& shader_name, uint32_t num_bindings, uint32_t push_bytes);

    // Execute a single compute dispatch synchronously (returns GPU timestamp ms if enabled)
    double dispatch_single(const VkBatchDispatch& d, bool measure_timestamp = false);

    // Execute a sequence of compute dispatches and copies in one synchronous submission.
    // Returns total GPU execution time in milliseconds via VK_QUERY_TYPE_TIMESTAMP.
    double dispatch_batched(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp = true);

    void set_replay_enabled(bool enabled);
    bool replay_enabled() const { return replay_enabled_; }
    void invalidate_replay();
    void reset_arenas();
    size_t last_batch_descriptor_allocs() const { return last_batch_descriptor_allocs_; }
    // Includes the final barrier for later submissions on the same queue.
    size_t last_batch_barriers() const { return last_batch_barriers_; }
    bool last_batch_replay_hit() const { return last_batch_replay_hit_; }

    // Unbatched baseline: submits one command buffer + queue submit per dispatch
    double dispatch_unbatched(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp = true);

    void flush_l2();

    static std::vector<VkDeviceInfo> enumerate_devices();
    static std::string find_shader_path(const std::string& shader_name);

private:
    struct BatchCache;
    double dispatch_cached(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp);
    void record_batch(const std::vector<VkBatchDispatch>& dispatches, bool timed);
    VkDeviceSize available_device_bytes(uint32_t memory_type) const;
    void create_instance_and_device(uint32_t device_index);
    VkCommandBuffer begin_one_time_commands();
    void submit_and_wait(VkCommandBuffer cmd);
    void submit_recorded(VkCommandBuffer cmd);
    void copy_buffer(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize bytes);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    uint32_t descriptor_capacity_ = 0, descriptor_set_capacity_ = 0;
    uint32_t timestamp_bits_ = 0;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::unique_ptr<BatchCache> batch_;
    bool replay_enabled_ = true;
    bool last_batch_replay_hit_ = false;
    size_t last_batch_descriptor_allocs_ = 0;
    size_t last_batch_barriers_ = 0;

    VkDeviceInfo info_;
    DeviceProfile profile_;
    std::optional<DType> kv_dtype_;
    int prefill_batch_ = 512;
    OpTrace* op_trace_ = nullptr;
    VkArena weight_arena_;
    VkArena scratch_arena_;
    VkArena staging_arena_;
    VkDeviceSize l2_flush_bytes_ = 0;
    VkBufferAllocation l2_flush_buf_;

    std::unordered_map<std::string, VkPipelineEntry> pipelines_;
    std::string pipeline_key_;
};

void check_vk(VkResult res, const char* expr, const char* file, int line);
#define FUSIONRT_VK_CHECK(expr) ::fusionrt::check_vk((expr), #expr, __FILE__, __LINE__)

}  // namespace fusionrt
