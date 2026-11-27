#pragma once

#include "fusionrt/model_types.h"

namespace fusionrt {

struct VulkanDenseKv {
    Tensor key, value;
};

struct VulkanGdnState {
    Tensor conv, ssm;
};

class VulkanSessionManager {
public:
    void allocate_dense(VkContext&, const DenseGQAConfig&, DType dtype = DType::F32);
    void allocate_gdn(VkContext&, const GatedDeltaNetConfig&, int conv_dim);
    void reset_kv_and_state_vulkan(VkContext&) const;

    const VulkanDenseKv& dense(size_t index) const { return dense_kv_[index]; }
    const VulkanGdnState& gdn(size_t index) const { return gdn_states_[index]; }

private:
    std::vector<VulkanDenseKv> dense_kv_;
    std::vector<VulkanGdnState> gdn_states_;
};

}  // namespace fusionrt
