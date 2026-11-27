#pragma once

#include "fusionrt/types.h"

namespace fusionrt::repack {

// MMQ selection requires 256-aligned input features and at least 32 output
// features for Q4_0 and Q8_0. Q6_K is eligible when the caller opts in.
bool can_repack_for_mmq(DType src_dtype, int out_features, int in_features);

// Returns src_dtype when the shape or format is unsuitable or Q6_A is disabled.
DType repacked_dtype_for(DType src_dtype, int out_features, int in_features, bool allow_q6_a = false);

// Returns zero for an unsupported destination format or shape.
size_t repacked_nbytes(DType dst_dtype, int out_features, int in_features);

// Converts contiguous row-major blocks without allocating. The host buffers
// must have sufficient capacity and must not overlap. Q4_KA and Q6_A
// destinations must be 16-byte aligned. Q6_A padding is written as zero.
// An explicit conversion accepts any positive row count.
bool repack_matrix_blocks(const void* src, DType src_dtype, void* dst, DType dst_dtype,
                          int out_features, int in_features);

}  // namespace fusionrt::repack
