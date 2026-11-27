#include "fusionrt/vulkan_backend.h"
#include "fusionrt/cost_model.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fusionrt {

void check_vk(VkResult res, const char* expr, const char* file, int line) {
    if (res != VK_SUCCESS) {
        std::ostringstream oss;
        oss << "Vulkan error (" << static_cast<int>(res) << ") at "
            << file << ":" << line << " (" << expr << ")";
        throw std::runtime_error(oss.str());
    }
}

namespace {

uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& mem_props, uint32_t type_filter,
                          VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

VkBufferAllocation create_vk_buffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size,
                                    VkBufferUsageFlags usage, VkMemoryPropertyFlags props, bool map_mem,
                                    const VkArena::Preflight& preflight = {}) {
    VkBufferAllocation out{};
    out.size = size;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = std::max<VkDeviceSize>(size, 256);
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    FUSIONRT_VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &out.buffer));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, out.buffer, &req);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    try {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(phys, &memory);
        mai.memoryTypeIndex = find_memory_type(memory, req.memoryTypeBits, props);
        out.heap_index = memory.memoryTypes[mai.memoryTypeIndex].heapIndex;
        out.allocation_size = req.size;
        if (preflight) preflight(mai.memoryTypeIndex, req.size);
        FUSIONRT_VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &out.memory));
        FUSIONRT_VK_CHECK(vkBindBufferMemory(device, out.buffer, out.memory, 0));
        if (map_mem) FUSIONRT_VK_CHECK(vkMapMemory(device, out.memory, 0, size, 0, &out.mapped));
    } catch (...) {
        vkDestroyBuffer(device, out.buffer, nullptr);
        if (out.memory) vkFreeMemory(device, out.memory, nullptr);
        throw;
    }
    return out;
}

std::vector<uint32_t> read_spv_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open SPIR-V shader: " + path);
    }
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V shader size: " + path);
    }
    std::vector<uint32_t> words(static_cast<size_t>(size / 4));
    if (!ifs.read(reinterpret_cast<char*>(words.data()), size)) {
        throw std::runtime_error("Failed to read SPIR-V shader: " + path);
    }
    return words;
}

std::string get_exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH]{};
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::filesystem::path(buf).parent_path().string();
    }
#endif
    return ".";
}

bool has_device_ext(VkPhysicalDevice phys, const char* ext_name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, exts.data());
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, ext_name) == 0) return true;
    }
    return false;
}

VkDeviceInfo query_vk_device_info(VkInstance instance, VkPhysicalDevice phys, uint32_t idx) {
    VkDeviceInfo info;
    info.device_index = idx;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    VkPhysicalDeviceSubgroupProperties sg_props{};
    sg_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    props2.pNext = &sg_props;
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    sg_props.pNext = &driver;
    vkGetPhysicalDeviceProperties2(phys, &props2);

    const auto& p = props2.properties;
    info.name = p.deviceName;
    info.vendor_id = p.vendorID;
    info.device_id = p.deviceID;
    info.api_version = p.apiVersion;
    info.subgroup_size = sg_props.subgroupSize;
    info.shared_mem_bytes = p.limits.maxComputeSharedMemorySize;
    info.min_storage_offset_align = std::max<uint64_t>(256, p.limits.minStorageBufferOffsetAlignment);
    info.timestamp_period_ns = p.limits.timestampPeriod;

    switch (p.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: info.device_type = "discrete"; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: info.device_type = "integrated"; break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: info.device_type = "cpu"; break;
        default: info.device_type = "other"; break;
    }

    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
        if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.device_local_vram_bytes += mem_props.memoryHeaps[h].size;
        }
    }

    info.has_fp16_int8 = has_device_ext(phys, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    info.has_16bit_storage = has_device_ext(phys, VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    info.has_8bit_storage = has_device_ext(phys, VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    info.has_coop_matrix = has_device_ext(phys, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    // This Radeon 610M driver advertises CoopMat but crashes during pipeline creation.
    const bool broken_coop = driver.driverID == VK_DRIVER_ID_AMD_PROPRIETARY && p.deviceID == 0x164e;
    if (broken_coop) info.has_coop_matrix = false;
    if (info.has_coop_matrix) {
        auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (fn) {
            uint32_t cm_count = 0;
            fn(phys, &cm_count, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> cm_props(cm_count);
            for (auto& c : cm_props) c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            fn(phys, &cm_count, cm_props.data());
            for (const auto& c : cm_props) {
                if (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
                    c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    c.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    c.scope == VK_SCOPE_SUBGROUP_KHR) {
                    if (c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && c.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                        info.coop_mat.fp16_in_fp32_acc = true;
                        info.coop_mat.supported = true;
                        info.coop_mat.m = 16;
                        info.coop_mat.n = 16;
                        info.coop_mat.k = 16;
                    }
                    if (c.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                        info.coop_mat.fp16_in_fp16_acc = true;
                    }
                }
            }
        }
    }
    return info;
}

}  // namespace

VkArena::~VkArena() = default;

void VkArena::init(VkDevice device, VkPhysicalDevice phys, VkDeviceSize capacity_bytes,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_props, VkDeviceSize align,
                   const Preflight& preflight) {
    release(device);
    align_ = std::max<VkDeviceSize>(align, 256);
    bool map_mem = (mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    alloc_ = create_vk_buffer(device, phys, capacity_bytes, usage, mem_props, map_mem, preflight);
    offset_ = 0;
}

void VkArena::release(VkDevice device) {
    if (owner_ && alloc_.buffer) owner_->invalidate_replay();
    if (alloc_.mapped && device != VK_NULL_HANDLE) {
        vkUnmapMemory(device, alloc_.memory);
        alloc_.mapped = nullptr;
    }
    if (alloc_.buffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, alloc_.buffer, nullptr);
        alloc_.buffer = VK_NULL_HANDLE;
    }
    if (alloc_.memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
        vkFreeMemory(device, alloc_.memory, nullptr);
        alloc_.memory = VK_NULL_HANDLE;
    }
    alloc_ = {};
    offset_ = 0;
}

VkDeviceSize VkArena::alloc(VkDeviceSize bytes) {
    VkDeviceSize aligned = (offset_ + align_ - 1) & ~(align_ - 1);
    if (aligned + bytes > alloc_.size) {
        std::ostringstream oss;
        oss << "VkArena OOM: requested " << bytes << " at " << aligned << " exceeds " << alloc_.size;
        throw std::runtime_error(oss.str());
    }
    offset_ = aligned + bytes;
    return aligned;
}

Tensor VkArena::alloc_tensor(const std::vector<int64_t>& shape, DType dt, const std::string& name) {
    Tensor t = Tensor::from_shape(shape, dt, Backend::Vulkan);
    t.name = name;
    t.vk_buffer = reinterpret_cast<uint64_t>(alloc_.buffer);
    t.vk_offset = alloc(std::max<VkDeviceSize>((t.nbytes() + 3) & ~VkDeviceSize{3}, 16));
    return t;
}

std::string VkContext::find_shader_path(const std::string& shader_name) {
    namespace fs = std::filesystem;
    std::string fname = shader_name;
    if (fname.size() < 4 || fname.substr(fname.size() - 4) != ".spv") {
        fname += ".spv";
    }
    std::vector<fs::path> candidates;
    if (const char* env_dir = std::getenv("FUSIONRT_SHADER_DIR")) {
        candidates.push_back(fs::path(env_dir) / fname);
    }
    fs::path exe_dir = get_exe_dir();
    candidates.push_back(exe_dir / "shaders" / fname);
    candidates.push_back(exe_dir / ".." / "shaders" / fname);
    candidates.push_back(fs::current_path() / "shaders" / fname);
    candidates.push_back(fs::current_path() / "build" / "shaders" / fname);

    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
    throw std::runtime_error("Cannot locate compiled SPIR-V shader: " + fname);
}

static VkResult create_instance(VkInstance* instance) {
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "fusionrt",
                                VK_MAKE_VERSION(0, 1, 0), nullptr, 0, VK_API_VERSION_1_3};
    const VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    return vkCreateInstance(&info, nullptr, instance);
}

std::vector<VkDeviceInfo> VkContext::enumerate_devices() {
    std::vector<VkDeviceInfo> result;
    VkInstance inst = VK_NULL_HANDLE;
    if (create_instance(&inst) != VK_SUCCESS) return result;

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> phys(count);
    if (count > 0) {
        vkEnumeratePhysicalDevices(inst, &count, phys.data());
    }
    for (uint32_t i = 0; i < count; ++i) {
        result.push_back(query_vk_device_info(inst, phys[i], i));
    }
    vkDestroyInstance(inst, nullptr);
    return result;
}

void VkContext::create_instance_and_device(uint32_t device_index) {
    FUSIONRT_VK_CHECK(create_instance(&instance_));

    uint32_t count = 0;
    FUSIONRT_VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    if (device_index >= count) {
        throw std::runtime_error("Invalid Vulkan device_index");
    }
    std::vector<VkPhysicalDevice> devs(count);
    FUSIONRT_VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devs.data()));
    phys_ = devs[device_index];
    info_ = query_vk_device_info(instance_, phys_, device_index);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qf_count, qfs.data());
    bool found_qf = false;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            found_qf = true;
            if (!(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;  // prefer async compute if available
        }
    }
    if (!found_qf) throw std::runtime_error("No Vulkan compute queue family found");
    timestamp_bits_ = qfs[queue_family_].timestampValidBits;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char*> exts;
    if (info_.has_fp16_int8) exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    if (info_.has_16bit_storage) exts.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    if (info_.has_8bit_storage) exts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    if (info_.has_coop_matrix) exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkPhysicalDeviceVulkan13Features feat13{};
    feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    feat13.subgroupSizeControl = VK_TRUE;
    feat13.computeFullSubgroups = VK_TRUE;

    VkPhysicalDeviceVulkan12Features feat12{};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feat12.pNext = &feat13;
    feat12.shaderFloat16 = info_.has_fp16_int8 ? VK_TRUE : VK_FALSE;
    feat12.shaderInt8 = info_.has_fp16_int8 ? VK_TRUE : VK_FALSE;
    feat12.storageBuffer8BitAccess = info_.has_8bit_storage ? VK_TRUE : VK_FALSE;
    feat12.uniformAndStorageBuffer8BitAccess = info_.has_8bit_storage ? VK_TRUE : VK_FALSE;
    feat12.vulkanMemoryModel = VK_TRUE;
    feat12.vulkanMemoryModelDeviceScope = VK_TRUE;

    VkPhysicalDevice16BitStorageFeatures feat16{};
    feat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    feat16.storageBuffer16BitAccess = info_.has_16bit_storage ? VK_TRUE : VK_FALSE;
    feat13.pNext = &feat16;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR feat_cm{};
    feat_cm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    feat_cm.cooperativeMatrix = info_.has_coop_matrix ? VK_TRUE : VK_FALSE;
    if (info_.has_coop_matrix) {
        feat16.pNext = &feat_cm;
    }

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &feat12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    FUSIONRT_VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &device_));

    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

struct VkContext::BatchCache {
    struct Pool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        uint32_t sets = 0, bindings = 0;
    };
    struct Access {
        VkBuffer buffer;
        VkDeviceSize begin, end;
        bool writes;
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    std::vector<Pool> pools;
    size_t pool_index = 0;
    std::unordered_map<std::string, VkDescriptorSet> sets;
    std::vector<Access> previous, current;
    std::vector<VkBufferCopy> copies;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<std::pair<const VkPipelineEntry*, VkDescriptorSet>> bindings;
    std::string recorded, key, recorded_bindings, binding_key, descriptor_key;
    std::vector<uint8_t> barriers;
    size_t recorded_barriers = 0;
    bool has_external_buffers = false;
};

VkContext::VkContext(uint32_t device_index, size_t weight_arena_mb, size_t scratch_arena_mb) {
    create_instance_and_device(device_index);

    const VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                       VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queue_family_};
    FUSIONRT_VK_CHECK(vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_));

    const VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
                                     VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    FUSIONRT_VK_CHECK(vkCreateQueryPool(device_, &qpci, nullptr, &query_pool_));

    const VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    FUSIONRT_VK_CHECK(vkCreateFence(device_, &fci, nullptr, &fence_));

    VkBufferUsageFlags buf_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    weight_arena_.owner_ = scratch_arena_.owner_ = this;
    if (weight_arena_mb)
        weight_arena_.init(device_, phys_, weight_arena_mb * 1024ULL * 1024ULL,
                           buf_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, info_.min_storage_offset_align);
    if (scratch_arena_mb)
        scratch_arena_.init(device_, phys_, scratch_arena_mb * 1024ULL * 1024ULL,
                            buf_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, info_.min_storage_offset_align);
    staging_arena_.init(device_, phys_, 8ULL * 1024ULL * 1024ULL,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        info_.min_storage_offset_align);
}

void VkContext::set_kv_dtype(std::optional<DType> dtype) {
    if (dtype && *dtype != DType::F32 && *dtype != DType::F16 && *dtype != DType::Q8_0)
        throw std::invalid_argument("Vulkan --kv-dtype accepts auto, f32, f16 or q8_0");
    kv_dtype_ = dtype;
}

void VkContext::set_prefill_batch(int rows) {
    if (rows < 1 || rows > 512) throw std::invalid_argument("--ubatch must be between 1 and 512");
    prefill_batch_ = rows;
}

VkDeviceSize VkContext::available_device_bytes(uint32_t memory_type) const {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    if (has_device_ext(phys_, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) memory.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(phys_, &memory);
    const auto heap = memory.memoryProperties.memoryTypes[memory_type].heapIndex;
    const auto reserved = [heap](const VkBufferAllocation& a) {
        return a.heap_index == heap ? a.allocation_size : VkDeviceSize{0};
    };
    const auto known_usage = reserved(weight_arena_.alloc_) + reserved(scratch_arena_.alloc_) +
                             reserved(staging_arena_.alloc_) + reserved(l2_flush_buf_);
    auto limit = memory.memoryProperties.memoryHeaps[heap].size;
    if (profile_.vram_total_mb > 0) {
        const double profile_bytes = profile_.vram_total_mb * (1024 * 1024);
        if (profile_bytes < static_cast<double>(limit)) limit = static_cast<VkDeviceSize>(profile_bytes);
    }
    if (budget.heapBudget[heap]) limit = std::min(limit, budget.heapBudget[heap]);
    return wddm_model_available(limit, std::max(known_usage, budget.heapUsage[heap]), 0);
}

void VkContext::reserve_model_memory(VkDeviceSize weight_bytes, VkDeviceSize context_bytes) {
    const auto bytes = weight_bytes + context_bytes;
    const auto alignment = info_.min_storage_offset_align;
    const auto used = (weight_arena_.used() + alignment - 1) & ~(alignment - 1);
    const auto unused = weight_arena_.capacity() - std::min(used, weight_arena_.capacity());
    if (bytes <= unused) return;
    const auto failure = [&](VkDeviceSize available) {
        return std::runtime_error("Vulkan model needs " + std::to_string(weight_bytes) +
            " weight/workspace bytes and " + std::to_string(context_bytes) +
            " KV/state bytes (" + std::to_string(bytes) + " total), but only " +
            std::to_string(available) + " device bytes are available");
    };
    if (used) throw failure(unused);
    VkDeviceSize available = 0;
    try {
        constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        weight_arena_.init(device_, phys_, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, alignment,
                           [&](uint32_t type, VkDeviceSize required) {
            available = available_device_bytes(type);
            if (required > available) throw failure(available);
        });
    } catch (const std::runtime_error& e) {
        if (available >= bytes) throw std::runtime_error(std::string(failure(available).what()) + ". " + e.what());
        throw failure(available);
    }
}

bool VkContext::try_reserve_scratch_memory(VkDeviceSize bytes) {
    const auto alignment = info_.min_storage_offset_align;
    const auto used = (scratch_arena_.used() + alignment - 1) & ~(alignment - 1);
    if (bytes <= scratch_arena_.capacity() - std::min(used, scratch_arena_.capacity())) return true;
    if (used) return false;
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    try {
        scratch_arena_.init(device_, phys_, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, alignment,
                            [&](uint32_t type, VkDeviceSize required) {
            if (required > available_device_bytes(type)) throw std::runtime_error("Vulkan scratch exceeds the device budget");
        });
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

VkContext::~VkContext() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        if (batch_) {
            for (const auto& pool : batch_->pools) vkDestroyDescriptorPool(device_, pool.handle, nullptr);
            batch_.reset();
        }
        for (auto& [name, entry] : pipelines_) {
            if (entry.pipeline) vkDestroyPipeline(device_, entry.pipeline, nullptr);
            if (entry.pipe_layout) vkDestroyPipelineLayout(device_, entry.pipe_layout, nullptr);
            if (entry.desc_layout) vkDestroyDescriptorSetLayout(device_, entry.desc_layout, nullptr);
            if (entry.shader) vkDestroyShaderModule(device_, entry.shader, nullptr);
        }
        if (l2_flush_buf_.buffer) vkDestroyBuffer(device_, l2_flush_buf_.buffer, nullptr);
        if (l2_flush_buf_.memory) vkFreeMemory(device_, l2_flush_buf_.memory, nullptr);
        staging_arena_.release(device_);
        scratch_arena_.release(device_);
        weight_arena_.release(device_);
        if (fence_) vkDestroyFence(device_, fence_, nullptr);
        if (query_pool_) vkDestroyQueryPool(device_, query_pool_, nullptr);
        if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

static VkCommandBuffer allocate_primary(VkDevice device, VkCommandPool pool) {
    const VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                           pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    FUSIONRT_VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));
    return cmd;
}

VkCommandBuffer VkContext::begin_one_time_commands() {
    VkCommandBuffer cmd = allocate_primary(device_, cmd_pool_);
    const VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    FUSIONRT_VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    return cmd;
}

void VkContext::submit_and_wait(VkCommandBuffer cmd) {
    FUSIONRT_VK_CHECK(vkEndCommandBuffer(cmd));
    submit_recorded(cmd);
}

void VkContext::submit_recorded(VkCommandBuffer cmd) {
    const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
    FUSIONRT_VK_CHECK(vkResetFences(device_, 1, &fence_));
    FUSIONRT_VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
    FUSIONRT_VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
}

static constexpr auto compute_and_transfer = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
static constexpr auto device_writes = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
static constexpr auto device_accesses = device_writes | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
static void memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                           VkAccessFlags src_access, VkAccessFlags dst_access) {
    const VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, src_access, dst_access};
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void VkContext::copy_buffer(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize bytes) {
    VkCommandBuffer cmd = begin_one_time_commands();
    memory_barrier(cmd, compute_and_transfer, VK_PIPELINE_STAGE_TRANSFER_BIT, device_writes,
                   VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    const VkBufferCopy region{src_off, dst_off, bytes};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, compute_and_transfer, VK_ACCESS_TRANSFER_WRITE_BIT, device_accesses);
    submit_and_wait(cmd);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
}

Tensor VkContext::upload(const HostTensor& ht, bool persistent) {
    VkArena& arena = persistent ? weight_arena_ : scratch_arena_;
    std::vector<int64_t> dims(ht.desc.shape.begin(), ht.desc.shape.begin() + ht.desc.ndim);
    Tensor dst = arena.alloc_tensor(dims, ht.desc.dtype, ht.desc.name);
    size_t nbytes = ht.desc.nbytes();
    VkDeviceSize chunk_cap = std::max<VkDeviceSize>(4096, staging_arena_.capacity());
    for (size_t off = 0; off < nbytes; off += static_cast<size_t>(chunk_cap)) {
        size_t chunk = std::min<size_t>(nbytes - off, static_cast<size_t>(chunk_cap));
        staging_arena_.reset();
        const auto padded = (chunk + 3) & ~size_t{3};
        const auto st_off = staging_arena_.alloc(padded);
        auto* stage = static_cast<uint8_t*>(staging_arena_.mapped()) + st_off;
        std::memcpy(stage, ht.bytes().data() + off, chunk);
        std::memset(stage + chunk, 0, padded - chunk);
        copy_buffer(staging_arena_.buffer(), st_off,
                    reinterpret_cast<VkBuffer>(dst.vk_buffer), dst.vk_offset + off, padded);
    }
    return dst;
}

HostTensor VkContext::download(const Tensor& t) {
    std::vector<int64_t> dims(t.shape.begin(), t.shape.begin() + t.ndim);
    HostTensor ht = HostTensor::create(dims, t.dtype);
    ht.desc.name = t.name;
    size_t nbytes = t.nbytes();
    VkDeviceSize chunk_cap = std::max<VkDeviceSize>(4096, staging_arena_.capacity());
    for (size_t off = 0; off < nbytes; off += static_cast<size_t>(chunk_cap)) {
        size_t chunk = std::min<size_t>(nbytes - off, static_cast<size_t>(chunk_cap));
        staging_arena_.reset();
        VkDeviceSize st_off = staging_arena_.alloc(chunk);
        copy_buffer(reinterpret_cast<VkBuffer>(t.vk_buffer), t.vk_offset + off,
                    staging_arena_.buffer(), st_off, chunk);
        std::memcpy(ht.storage.data() + off, static_cast<uint8_t*>(staging_arena_.mapped()) + st_off, chunk);
    }
    return ht;
}

void VkContext::zero_buffers(std::span<const Tensor> tensors) {
    if (tensors.empty()) return;

    VkCommandBuffer cmd = begin_one_time_commands();
    memory_barrier(cmd, compute_and_transfer, VK_PIPELINE_STAGE_TRANSFER_BIT, device_writes, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (const auto& tensor : tensors) {
        const auto bytes = (tensor.nbytes() + 3) & ~VkDeviceSize{3};
        if (bytes) vkCmdFillBuffer(cmd, reinterpret_cast<VkBuffer>(tensor.vk_buffer), tensor.vk_offset, bytes, 0);
    }
    memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, compute_and_transfer, VK_ACCESS_TRANSFER_WRITE_BIT, device_accesses);
    submit_and_wait(cmd);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
}

void VkContext::synchronize() {
    FUSIONRT_VK_CHECK(vkQueueWaitIdle(queue_));
}

void VkContext::set_replay_enabled(bool enabled) {
    if (replay_enabled_ != enabled && batch_) batch_->recorded.clear();
    replay_enabled_ = enabled;
}

void VkContext::invalidate_replay() {
    if (!batch_) return;
    batch_->recorded.clear();
    batch_->recorded_bindings.clear();
    batch_->sets.clear();
    batch_->pool_index = 0;
    batch_->has_external_buffers = false;
    for (auto& pool : batch_->pools) {
        FUSIONRT_VK_CHECK(vkResetDescriptorPool(device_, pool.handle, 0));
        pool.sets = pool.bindings = 0;
    }
}

void VkContext::reset_arenas() {
    invalidate_replay();
    weight_arena_.reset();
    scratch_arena_.reset();
}

static void append_key(std::string& key, const auto& value) {
    key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void VkContext::record_batch(const std::vector<VkBatchDispatch>& dispatches, bool timed) {
    auto& c = *batch_;
    c.recorded.clear();
    const bool plan_changed = c.binding_key != c.recorded_bindings;
    if (plan_changed) {
        c.recorded_bindings.clear();
        c.barriers.resize(dispatches.size());
    }
    if (!c.cmd) c.cmd = allocate_primary(device_, cmd_pool_);
    else FUSIONRT_VK_CHECK(vkResetCommandBuffer(c.cmd, 0));
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
    FUSIONRT_VK_CHECK(vkBeginCommandBuffer(c.cmd, &begin));
    if (timed) {
        vkCmdResetQueryPool(c.cmd, query_pool_, 0, 2);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
    }
    c.previous.clear();
    c.recorded_barriers = 0;
    const auto barrier = [&] {
        memory_barrier(c.cmd, compute_and_transfer, compute_and_transfer, device_writes, device_accesses);
        ++c.recorded_barriers;
        c.previous.clear();
    };
    for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& d = dispatches[i];
        if (plan_changed) {
            const auto access = [&](size_t binding, VkDeviceSize offset, VkDeviceSize size, bool writes) {
                const auto& t = d.buffers[binding];
                const auto start = t.vk_offset + offset;
                c.current.push_back({reinterpret_cast<VkBuffer>(t.vk_buffer), start, start + size, writes});
            };
            c.current.clear();
            if (d.copies.empty()) {
                const auto mask = c.bindings[i].first->read_only_bindings;
                for (size_t j = 0; j < d.buffers.size(); ++j)
                    access(j, 0, std::max<VkDeviceSize>((d.buffers[j].nbytes() + 3) & ~VkDeviceSize{3}, 16),
                           j >= 32 || (mask & (1u << j)) == 0);
            } else {
                for (const auto& r : d.copies) {
                    access(0, r.srcOffset, r.size, false);
                    access(1, r.dstOffset, r.size, true);
                }
            }
            // External buffers can alias one another through imported memory.
            c.barriers[i] = (!c.previous.empty() && c.has_external_buffers) ||
                std::any_of(c.current.begin(), c.current.end(), [&](const auto& a) {
                    return std::any_of(c.previous.begin(), c.previous.end(), [&](const auto& b) {
                        return (a.writes || b.writes) && a.buffer == b.buffer && a.begin < b.end && b.begin < a.end;
                    });
                });
        }
        if (c.barriers[i]) barrier();
        if (!d.copies.empty()) {
            c.copies = d.copies;
            for (auto& r : c.copies) {
                r.srcOffset += d.buffers[0].vk_offset;
                r.dstOffset += d.buffers[1].vk_offset;
            }
            vkCmdCopyBuffer(c.cmd, reinterpret_cast<VkBuffer>(d.buffers[0].vk_buffer),
                            reinterpret_cast<VkBuffer>(d.buffers[1].vk_buffer),
                            static_cast<uint32_t>(c.copies.size()), c.copies.data());
        } else {
            const auto& [pipe, set] = c.bindings[i];
            vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
            vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipe_layout, 0, 1, &set, 0, nullptr);
            if (d.push_size)
                vkCmdPushConstants(c.cmd, pipe->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, d.push_size, d.push_data);
            vkCmdDispatch(c.cmd, d.gx, d.gy, d.gz);
        }
        if (plan_changed) c.previous.insert(c.previous.end(), c.current.begin(), c.current.end());
    }
    barrier();  // Make writes visible to later submissions on this queue.
    if (timed) vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
    FUSIONRT_VK_CHECK(vkEndCommandBuffer(c.cmd));
    c.recorded_bindings.swap(c.binding_key);
    if (replay_enabled_) c.recorded.swap(c.key);
}

double VkContext::dispatch_cached(const std::vector<VkBatchDispatch>& dispatches, bool timed) {
    last_batch_descriptor_allocs_ = last_batch_barriers_ = 0;
    last_batch_replay_hit_ = false;
    if (dispatches.empty()) return 0.0;
    if (!batch_) batch_ = std::make_unique<BatchCache>();
    auto& c = *batch_;
    // Keep ordinary scratch rewinds reusable, but cap growth across changing shapes.
    if (c.has_external_buffers || c.sets.size() >= 16384) invalidate_replay();
    timed = timed && timestamp_bits_ > 0;
    c.key.clear();
    c.binding_key.clear();
    append_key(c.key, timed);
    c.bindings.resize(dispatches.size());
    for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& d = dispatches[i];
        for (const auto& b : d.buffers) {
            const auto handle = reinterpret_cast<VkBuffer>(b.vk_buffer);
            c.has_external_buffers |= handle != weight_arena_.buffer() && handle != scratch_arena_.buffer();
        }
        append_key(c.binding_key, d.copies.size());
        if (!d.copies.empty()) {
            for (const auto& b : std::span(d.buffers).first(2)) {
                append_key(c.binding_key, b.vk_buffer);
                append_key(c.binding_key, b.vk_offset);
            }
            for (const auto& r : d.copies) {
                append_key(c.binding_key, r.srcOffset);
                append_key(c.binding_key, r.dstOffset);
                append_key(c.binding_key, r.size);
            }
            continue;
        }

        const auto count = static_cast<uint32_t>(d.buffers.size());
        const auto& pipe = get_pipeline(d.shader_name, count, d.push_size);
        c.descriptor_key.clear();
        append_key(c.descriptor_key, pipe.pipeline);
        c.buffers.clear();
        for (const auto& b : d.buffers) {
            const auto range = std::max<VkDeviceSize>((b.nbytes() + 3) & ~VkDeviceSize{3}, 16);
            c.buffers.push_back({reinterpret_cast<VkBuffer>(b.vk_buffer), b.vk_offset, range});
            append_key(c.descriptor_key, b.vk_buffer);
            append_key(c.descriptor_key, b.vk_offset);
            append_key(c.descriptor_key, range);
        }
        auto it = c.sets.find(c.descriptor_key);
        if (it == c.sets.end()) {
            constexpr uint32_t pool_sets = 1024, pool_bindings = 8192;
            while (c.pool_index < c.pools.size() &&
                   (c.pools[c.pool_index].sets == pool_sets || c.pools[c.pool_index].bindings + count > pool_bindings))
                ++c.pool_index;
            if (c.pool_index == c.pools.size()) {
                const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pool_bindings};
                const VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                       nullptr, 0, pool_sets, 1, &size};
                c.pools.reserve(c.pools.size() + 1);
                BatchCache::Pool pool;
                FUSIONRT_VK_CHECK(vkCreateDescriptorPool(device_, &info, nullptr, &pool.handle));
                c.pools.push_back(pool);
            }
            auto& pool = c.pools[c.pool_index];
            const VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                  nullptr, pool.handle, 1, &pipe.desc_layout};
            VkDescriptorSet set;
            FUSIONRT_VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &set));
            ++pool.sets;
            pool.bindings += count;
            ++last_batch_descriptor_allocs_;
            c.writes.clear();
            for (uint32_t j = 0; j < count; ++j)
                c.writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, j, 0, 1,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &c.buffers[j], nullptr});
            vkUpdateDescriptorSets(device_, count, c.writes.data(), 0, nullptr);
            it = c.sets.emplace(c.descriptor_key, set).first;
        }
        c.bindings[i] = {&pipe, it->second};
        append_key(c.binding_key, it->second);
        append_key(c.key, d.gx);
        append_key(c.key, d.gy);
        append_key(c.key, d.gz);
        append_key(c.key, d.push_size);
        if (d.push_size) c.key.append(static_cast<const char*>(d.push_data), d.push_size);
    }

    last_batch_replay_hit_ = replay_enabled_ && c.key == c.recorded && c.binding_key == c.recorded_bindings;
    if (!last_batch_replay_hit_) record_batch(dispatches, timed);
    last_batch_barriers_ = c.recorded_barriers;
    submit_recorded(c.cmd);
    if (!timed) return 0.0;
    uint64_t ts[2]{};
    FUSIONRT_VK_CHECK(vkGetQueryPoolResults(device_, query_pool_, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return static_cast<double>((ts[1] - ts[0]) & (UINT64_MAX >> (64 - timestamp_bits_))) *
           static_cast<double>(info_.timestamp_period_ns) * 1e-6;
}

// glslc emits GLSL readonly SSBOs as single-member structs with NonWritable.
// Leave every other binding writable, including unfamiliar decorations/types.
static uint32_t readonly_bindings(std::span<const uint32_t> spv) {
    enum { TypeStruct = 30, TypePointer = 32, Variable = 59, Decorate = 71, MemberDecorate = 72,
           GroupDecorate = 74, GroupMemberDecorate = 75 };
    enum { NonWritable = 24, Binding = 33, DescriptorSet = 34 };
    struct Id {
        uint32_t type = 0, binding = 32, set = UINT32_MAX;
        bool single_member = false, readonly = false, readonly_member = false;
    };
    if (spv.size() < 5 || spv[3] == 0 || spv[3] > 0x3fffff) return 0;
    std::vector<Id> ids(spv[3]);
    for (size_t i = 5; i < spv.size();) {
        const auto* p = spv.data() + i;
        const uint32_t count = p[0] >> 16;
        if (!count || count > spv.size() - i) return 0;
        if (count < 2 || p[1] >= ids.size()) {
            i += count;
            continue;
        }
        switch (p[0] & 0xffffu) {
            case GroupDecorate: case GroupMemberDecorate: return 0;
            case TypeStruct: ids[p[1]].single_member = count == 3; break;
            case TypePointer: if (count == 4 && p[3] < ids.size()) ids[p[1]].type = p[3]; break;
            case Variable: if (count >= 4 && p[2] < ids.size()) ids[p[2]].type = p[1]; break;
            case Decorate:
                if (count == 4 && p[2] == Binding) ids[p[1]].binding = p[3];
                if (count == 4 && p[2] == DescriptorSet) ids[p[1]].set = p[3];
                if (count == 3 && p[2] == NonWritable) ids[p[1]].readonly = true;
                break;
            case MemberDecorate:
                if (count == 4 && p[2] == 0 && p[3] == NonWritable) ids[p[1]].readonly_member = true;
                break;
        }
        i += count;
    }
    uint32_t readonly = 0, writable = 0;
    for (const auto& id : ids) {
        if (id.binding >= 32 || id.set != 0) continue;
        const auto& block = ids[ids[id.type].type];
        (id.readonly || (block.single_member && block.readonly_member) ? readonly : writable) |= 1u << id.binding;
    }
    return readonly & ~writable;
}

const VkPipelineEntry& VkContext::get_pipeline(const std::string& shader_name, uint32_t num_bindings, uint32_t push_bytes) {
    auto& key = pipeline_key_;
    key = shader_name;
    append_key(key, num_bindings);
    append_key(key, push_bytes);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;

    VkPipelineEntry entry;
    entry.num_bindings = num_bindings;
    auto spv = read_spv_file(find_shader_path(shader_name));
    entry.read_only_bindings = readonly_bindings(spv);

    const VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                        spv.size() * sizeof(uint32_t), spv.data()};
    FUSIONRT_VK_CHECK(vkCreateShaderModule(device_, &smci, nullptr, &entry.shader));

    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i) {
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }

    const VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                nullptr, 0, num_bindings, bindings.data()};
    FUSIONRT_VK_CHECK(vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &entry.desc_layout));

    const VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    const VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
                                          &entry.desc_layout, uint32_t(push_bytes > 0), push_bytes ? &pcr : nullptr};
    FUSIONRT_VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &entry.pipe_layout));

    const uint32_t sg_size = 32;
    const VkSpecializationMapEntry spec_entry{0, 0, sizeof(sg_size)};
    const VkSpecializationInfo spec_info{1, &spec_entry, sizeof(sg_size), &sg_size};
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req_sg{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr, sg_size};

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
         entry.shader, "main", nullptr}, entry.pipe_layout, VK_NULL_HANDLE, -1};
    if (shader_name == "gemm_coopmat.comp") {
        cpci.stage.pSpecializationInfo = &spec_info;
        cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        cpci.stage.pNext = &req_sg;
    }
    FUSIONRT_VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &entry.pipeline));

    auto [inserted, ok] = pipelines_.emplace(key, entry);
    return inserted->second;
}

void VkContext::flush_l2() {
    if (!l2_flush_buf_.buffer) {
        l2_flush_bytes_ = 64ULL * 1024ULL * 1024ULL;
        l2_flush_buf_ = create_vk_buffer(device_, phys_, l2_flush_bytes_,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    }
    VkCommandBuffer cmd = begin_one_time_commands();
    vkCmdFillBuffer(cmd, l2_flush_buf_.buffer, 0, l2_flush_bytes_, 0x5A5A5A5Au);
    submit_and_wait(cmd);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
}

}  // namespace fusionrt
