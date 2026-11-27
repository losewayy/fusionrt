#include "fusionrt/repack.h"
#include "fusionrt/tensor.h"

#include <limits>

namespace fusionrt::repack {
namespace {

constexpr int values_per_block = 256;

bool valid_shape(int out_features, int in_features) {
    return out_features > 0 && in_features > 0 && in_features % values_per_block == 0;
}

DType block_destination(DType src_dtype) {
    switch (src_dtype) {
        case DType::Q4_0: return DType::Q4_A;
        case DType::Q8_0: return DType::Q8_A;
        case DType::Q4_K: return DType::Q4_KA;
        case DType::Q6_K: return DType::Q6_A;
        default: return src_dtype;
    }
}

}  // namespace

bool can_repack_for_mmq(DType src_dtype, int out_features, int in_features) {
    return repacked_dtype_for(src_dtype, out_features, in_features, true) != src_dtype;
}

DType repacked_dtype_for(DType src_dtype, int out_features, int in_features, bool allow_q6_a) {
    if (!valid_shape(out_features, in_features)) return src_dtype;
    if ((src_dtype == DType::Q4_0 || src_dtype == DType::Q8_0) && out_features < 32) return src_dtype;
    if (src_dtype == DType::Q6_K && !allow_q6_a) return src_dtype;
    return block_destination(src_dtype);
}

size_t repacked_nbytes(DType dst_dtype, int out_features, int in_features) {
    if (!valid_shape(out_features, in_features)) return 0;
    switch (dst_dtype) {
        case DType::Q4_A:
        case DType::Q8_A:
        case DType::Q4_KA:
        case DType::Q6_A: {
            const auto blocks = static_cast<uint64_t>(out_features) * (in_features / values_per_block);
            const auto bytes = blocks * dtype_type_size(dst_dtype);
            return bytes <= std::numeric_limits<size_t>::max() ? static_cast<size_t>(bytes) : 0;
        }
        default: return 0;
    }
}

bool repack_matrix_blocks(const void* src, DType src_dtype, void* dst, DType dst_dtype,
                          int out_features, int in_features) {
    if (!src || !dst || src == dst || src_dtype == dst_dtype || block_destination(src_dtype) != dst_dtype ||
        repacked_nbytes(dst_dtype, out_features, in_features) == 0) return false;

    if ((dst_dtype == DType::Q4_KA || dst_dtype == DType::Q6_A) &&
        reinterpret_cast<uintptr_t>(dst) % 16 != 0) return false;

    const int64_t count = static_cast<int64_t>(out_features) * in_features;
    switch (dst_dtype) {
        case DType::Q4_A:
            repack_q4_0_to_q4_a(static_cast<const BlockQ4_0*>(src), static_cast<BlockQ4_A*>(dst), count);
            break;
        case DType::Q8_A:
            repack_q8_0_to_q8_a(static_cast<const BlockQ8_0*>(src), static_cast<BlockQ8_A*>(dst), count);
            break;
        case DType::Q4_KA:
            repack_q4_k_to_q4_ka(static_cast<const BlockQ4_K*>(src), static_cast<BlockQ4_KA*>(dst), count);
            break;
        case DType::Q6_A: {
            repack_q6_k_to_q6_a(static_cast<const BlockQ6_K*>(src), static_cast<BlockQ6_A*>(dst), count);
            auto* bytes = static_cast<uint8_t*>(dst);
            constexpr size_t padding = offsetof(BlockQ6_A, d) + sizeof(uint16_t);
            for (int64_t b = 0; b < count / values_per_block; ++b) {
                std::memset(bytes + b * sizeof(BlockQ6_A) + padding, 0, sizeof(BlockQ6_A) - padding);
            }
            break;
        }
        default: return false;
    }
    return true;
}

}  // namespace fusionrt::repack
