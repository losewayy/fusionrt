#pragma once

#include "fusionrt/model_types.h"

namespace fusionrt {

struct VulkanDenseKv {
    Tensor key, value, latent;
    bool owns = true;
};

struct VulkanGdnState {
    Tensor conv, ssm, conv_snapshot, ssm_snapshot;
};

class VulkanSessionManager {
public:
    void allocate_dense(VkContext&, const DenseGQAConfig&, DType dtype = DType::F32, int source = -1);
    void allocate_gdn(VkContext&, const GatedDeltaNetConfig&, int conv_dim);
    void reset_kv_and_state_vulkan(VkContext&) const;
    void snapshot_gdn_state(VkContext&, size_t first, size_t end) const;
    void restore_gdn_state(VkContext&, size_t first, size_t end) const;
    size_t gdn_count() const { return gdn_states_.size(); }

    const VulkanDenseKv& dense(size_t index) const { return dense_kv_[index]; }
    const VulkanGdnState& gdn(size_t index) const { return gdn_states_[index]; }

private:
    void copy_gdn_state(VkContext&, size_t first, size_t end, bool restore) const;
    std::vector<VulkanDenseKv> dense_kv_;
    std::vector<VulkanGdnState> gdn_states_;
};

}  // namespace fusionrt
