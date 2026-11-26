#pragma once

#include "fusionrt/tensor.h"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusionrt {

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
    void* mapped = nullptr;
};

class VkArena {
public:
    VkArena() = default;
    ~VkArena();
    VkArena(const VkArena&) = delete;
    VkArena& operator=(const VkArena&) = delete;

    void init(VkDevice device, VkPhysicalDevice phys, VkDeviceSize capacity_bytes,
              VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props, VkDeviceSize align);
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
};

struct VkBatchDispatch {
    std::string shader_name;
    uint32_t gx = 1, gy = 1, gz = 1;
    std::vector<Tensor> buffers;
    const void* push_data = nullptr;
    uint32_t push_size = 0;
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

    VkArena& weight_arena() { return weight_arena_; }
    VkArena& scratch_arena() { return scratch_arena_; }

    Tensor upload(const HostTensor& ht, bool persistent = true);
    HostTensor download(const Tensor& t);
    void synchronize();

    // Load SPIR-V shader pipeline on demand (with descriptor pool caching)
    const VkPipelineEntry& get_pipeline(const std::string& shader_name, uint32_t num_bindings, uint32_t push_bytes);

    // Execute a single compute dispatch synchronously (returns GPU timestamp ms if enabled)
    double dispatch_single(const VkBatchDispatch& d, bool measure_timestamp = false);

    // Execute an entire sequence of compute dispatches in ONE VkCommandBuffer with barriers
    // between dispatches to eliminate per-op command-buffer submission overhead.
    // Returns total GPU execution time in milliseconds via VK_QUERY_TYPE_TIMESTAMP.
    double dispatch_batched(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp = true);

    // Unbatched baseline: submits one command buffer + queue submit per dispatch
    double dispatch_unbatched(const std::vector<VkBatchDispatch>& dispatches, bool measure_timestamp = true);

    void flush_l2();

    static std::vector<VkDeviceInfo> enumerate_devices();
    static std::string find_shader_path(const std::string& shader_name);

private:
    void create_instance_and_device(uint32_t device_index);
    void copy_buffer(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize bytes);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    VkDeviceInfo info_;
    VkArena weight_arena_;
    VkArena scratch_arena_;
    VkArena staging_arena_;
    VkDeviceSize l2_flush_bytes_ = 0;
    VkBufferAllocation l2_flush_buf_;

    std::unordered_map<std::string, VkPipelineEntry> pipelines_;
};

void check_vk(VkResult res, const char* expr, const char* file, int line);
#define FUSIONRT_VK_CHECK(expr) ::fusionrt::check_vk((expr), #expr, __FILE__, __LINE__)

}  // namespace fusionrt
