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

std::vector<VkDeviceInfo> VkContext::enumerate_devices() {
    std::vector<VkDeviceInfo> result;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "fusionrt";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        return result;
    }

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
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "fusionrt";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    FUSIONRT_VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));

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

VkContext::VkContext(uint32_t device_index, size_t weight_arena_mb, size_t scratch_arena_mb) {
    create_instance_and_device(device_index);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queue_family_;
    FUSIONRT_VK_CHECK(vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_));

    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 64;
    FUSIONRT_VK_CHECK(vkCreateQueryPool(device_, &qpci, nullptr, &query_pool_));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    FUSIONRT_VK_CHECK(vkCreateFence(device_, &fci, nullptr, &fence_));

    VkBufferUsageFlags buf_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;

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
        if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

VkCommandBuffer VkContext::begin_one_time_commands() {
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    FUSIONRT_VK_CHECK(vkAllocateCommandBuffers(device_, &cbai, &cmd));

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    FUSIONRT_VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    return cmd;
}

void VkContext::submit_and_wait(VkCommandBuffer cmd) {
    FUSIONRT_VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    FUSIONRT_VK_CHECK(vkResetFences(device_, 1, &fence_));
    FUSIONRT_VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
    FUSIONRT_VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
}

void VkContext::copy_buffer(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize bytes) {
    VkCommandBuffer cmd = begin_one_time_commands();

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    constexpr auto compute_and_transfer = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd, compute_and_transfer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    VkBufferCopy region{};
    region.srcOffset = src_off;
    region.dstOffset = dst_off;
    region.size = bytes;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

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
    constexpr auto compute_and_transfer = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, compute_and_transfer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    for (const auto& tensor : tensors) {
        const auto bytes = (tensor.nbytes() + 3) & ~VkDeviceSize{3};
        if (bytes) vkCmdFillBuffer(cmd, reinterpret_cast<VkBuffer>(tensor.vk_buffer), tensor.vk_offset, bytes, 0);
    }
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, compute_and_transfer,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    submit_and_wait(cmd);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
}

void VkContext::synchronize() {
    FUSIONRT_VK_CHECK(vkQueueWaitIdle(queue_));
}

const VkPipelineEntry& VkContext::get_pipeline(const std::string& shader_name, uint32_t num_bindings, uint32_t push_bytes) {
    std::ostringstream key_ss;
    key_ss << shader_name << ":" << num_bindings << ":" << push_bytes;
    std::string key = key_ss.str();
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;

    VkPipelineEntry entry;
    entry.num_bindings = num_bindings;
    auto spv = read_spv_file(find_shader_path(shader_name));

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    FUSIONRT_VK_CHECK(vkCreateShaderModule(device_, &smci, nullptr, &entry.shader));

    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = num_bindings;
    dslci.pBindings = bindings.data();
    FUSIONRT_VK_CHECK(vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &entry.desc_layout));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_bytes;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &entry.desc_layout;
    plci.pushConstantRangeCount = (push_bytes > 0) ? 1 : 0;
    plci.pPushConstantRanges = (push_bytes > 0) ? &pcr : nullptr;
    FUSIONRT_VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &entry.pipe_layout));

    uint32_t sg_size = 32;
    VkSpecializationMapEntry spec_entry{};
    spec_entry.constantID = 0;
    spec_entry.offset = 0;
    spec_entry.size = sizeof(uint32_t);
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 1;
    spec_info.pMapEntries = &spec_entry;
    spec_info.dataSize = sizeof(uint32_t);
    spec_info.pData = &sg_size;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req_sg{};
    req_sg.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    req_sg.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = entry.shader;
    cpci.stage.pName = "main";
    if (shader_name == "gemm_coopmat.comp") {
        cpci.stage.pSpecializationInfo = &spec_info;
        cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        cpci.stage.pNext = &req_sg;
    }
    cpci.layout = entry.pipe_layout;
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
