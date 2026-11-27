#pragma once

#include <algorithm>
#include <vulkan/vulkan.h>

namespace fusionrt::vulkan_detail {

inline VkDeviceSize available_heap_bytes(const VkPhysicalDeviceMemoryProperties& memory,
                                         const VkPhysicalDeviceMemoryBudgetPropertiesEXT* budget,
                                         uint32_t memory_type, VkDeviceSize known_usage) {
    const auto heap = memory.memoryTypes[memory_type].heapIndex;
    if (budget && budget->heapBudget[heap])
        return budget->heapBudget[heap] - std::min(budget->heapBudget[heap], budget->heapUsage[heap]);
    const auto size = memory.memoryHeaps[heap].size;
    return size - std::min(size, known_usage);
}

}  // namespace fusionrt::vulkan_detail
