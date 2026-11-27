#pragma once

#include "fusionrt/types.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fusionrt {

struct Tensor {
    int ndim = 0;
    std::array<int64_t, 4> shape{1, 1, 1, 1};
    std::array<int64_t, 4> stride{0, 0, 0, 0};  // in elements (for block-quantized, innermost is element stride 1)
    DType dtype = DType::F32;
    Backend backend = Backend::CPU;
    void* data = nullptr;        // CPU host ptr or CUDA device ptr
    uint64_t vk_buffer = 0;      // VkBuffer cast as uint64_t
    uint64_t vk_offset = 0;      // byte offset into VkBuffer
    size_t byte_offset = 0;      // byte offset on CPU/CUDA base pointer
    std::string name;
    int hadamard_block = 0;
    const int8_t* hadamard_signs = nullptr;  // Device vector on uploaded rotated weights.
    int hadamard_head_dim = 0, hadamard_key_heads = 0;

    Tensor() = default;

    static Tensor make_contiguous(std::initializer_list<int64_t> dims, DType dt, Backend b, void* ptr = nullptr) {
        Tensor t;
        t.ndim = static_cast<int>(dims.size());
        if (t.ndim < 1 || t.ndim > 4) {
            throw std::runtime_error("Tensor rank must be 1..4");
        }
        int i = 0;
        for (int64_t d : dims) t.shape[i++] = d;
        t.dtype = dt;
        t.backend = b;
        t.data = ptr;
        t.recompute_contiguous_strides();
        return t;
    }

    static Tensor from_shape(const std::vector<int64_t>& dims, DType dt, Backend b) {
        Tensor t;
        t.ndim = static_cast<int>(dims.size());
        if (t.ndim < 1 || t.ndim > 4) {
            throw std::runtime_error("Tensor rank must be 1..4");
        }
        for (int i = 0; i < t.ndim; ++i) t.shape[i] = dims[i];
        t.dtype = dt;
        t.backend = b;
        t.recompute_contiguous_strides();
        return t;
    }

    void recompute_contiguous_strides() {
        int64_t acc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            stride[i] = acc;
            acc *= shape[i];
        }
    }

    int64_t numel() const {
        if (ndim == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    size_t nbytes() const {
        if (ndim == 0) return 0;
        if (dtype_is_quantized(dtype)) {
            int64_t cols = shape[ndim - 1];
            int64_t rows = numel() / cols;
            return static_cast<size_t>(rows) * row_nbytes(dtype, cols);
        }
        return static_cast<size_t>(numel()) * dtype_type_size(dtype);
    }

    bool is_contiguous() const {
        int64_t expected = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            if (shape[i] != 1 && stride[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    template <typename T = void>
    T* ptr() const {
        return reinterpret_cast<T*>(static_cast<uint8_t*>(data) + byte_offset);
    }

    Tensor view(std::initializer_list<int64_t> new_dims) const {
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot view non-contiguous tensor without copy");
        }
        Tensor v = *this;
        v.ndim = static_cast<int>(new_dims.size());
        int64_t total = 1;
        int i = 0;
        for (int64_t d : new_dims) {
            v.shape[i++] = d;
            total *= d;
        }
        if (total != numel()) {
            throw std::runtime_error("Tensor view element count mismatch");
        }
        v.recompute_contiguous_strides();
        return v;
    }

    Tensor slice_dim0(int64_t start, int64_t count) const {
        if (start < 0 || start + count > shape[0]) {
            throw std::runtime_error("slice_dim0 out of range");
        }
        if (dtype_is_quantized(dtype)) {
            if (ndim < 2) throw std::runtime_error("Cannot slice 1D quantized tensor along dim0");
        }
        Tensor s = *this;
        s.shape[0] = count;
        size_t elem_off = static_cast<size_t>(start) * static_cast<size_t>(stride[0]);
        size_t bytes_off = 0;
        if (dtype_is_quantized(dtype)) {
            size_t bs = dtype_block_size(dtype);
            size_t ts = dtype_type_size(dtype);
            bytes_off = (elem_off / bs) * ts;
        } else {
            bytes_off = elem_off * dtype_type_size(dtype);
        }
        s.byte_offset += bytes_off;
        s.vk_offset += bytes_off;
        return s;
    }
};

// Host tensor with owned storage or a read-only file-backed view.
struct HostTensor {
    Tensor desc;
    std::vector<uint8_t> storage;
    std::shared_ptr<const std::vector<int8_t>> hadamard_signs;

    HostTensor() = default;
    HostTensor(std::initializer_list<int64_t> dims, DType dt) {
        desc = Tensor::make_contiguous(dims, dt, Backend::CPU);
        storage.resize(desc.nbytes(), 0);
        desc.data = storage.data();
    }
    static HostTensor create(const std::vector<int64_t>& dims, DType dt) {
        HostTensor ht;
        ht.desc = Tensor::from_shape(dims, dt, Backend::CPU);
        ht.storage.resize(ht.desc.nbytes(), 0);
        ht.desc.data = ht.storage.data();
        return ht;
    }
    static HostTensor mapped(Tensor desc, std::shared_ptr<const uint8_t> view) {
        HostTensor ht;
        ht.desc = std::move(desc);
        ht.mapped_ = std::move(view);
        return ht;
    }
    std::span<const uint8_t> bytes() const {
        return mapped_ ? std::span<const uint8_t>(mapped_.get(), desc.nbytes()) :
                         std::span<const uint8_t>(storage);
    }
    HostTensor slice_dim0(int64_t start, int64_t count) const {
        auto slice = desc.slice_dim0(start, count);
        const auto part = bytes().subspan(slice.byte_offset, slice.nbytes());
        slice.byte_offset = 0;
        HostTensor out;
        if (mapped_) {
            out = mapped(std::move(slice), {mapped_, part.data()});
        } else {
            out.desc = std::move(slice);
            out.storage.assign(part.begin(), part.end());
            out.desc.data = out.storage.data();
        }
        out.hadamard_signs = hadamard_signs;
        return out;
    }
    template <typename T>
    T* data() {
        if (mapped_) {
            const auto view = bytes();
            storage.assign(view.begin(), view.end());
            mapped_.reset();
            desc.data = storage.data();
        }
        return reinterpret_cast<T*>(storage.data());
    }
    template <typename T>
    const T* data() const { return reinterpret_cast<const T*>(bytes().data()); }

private:
    std::shared_ptr<const uint8_t> mapped_;
};

// Host quantization / conversion utilities for tests & weight loading
void quantize_row_q8_0(const float* src, BlockQ8_0* dst, int64_t cols);
void dequantize_row_q8_0(const BlockQ8_0* src, float* dst, int64_t cols);
void quantize_row_q4_0(const float* src, BlockQ4_0* dst, int64_t cols);
void dequantize_row_q4_0(const BlockQ4_0* src, float* dst, int64_t cols);
void dequantize_row_q4_1(const BlockQ4_1* src, float* dst, int64_t cols);
void quantize_row_q2_k(const float* src, BlockQ2_K* dst, int64_t cols);
void dequantize_row_q2_k(const BlockQ2_K* src, float* dst, int64_t cols);
void dequantize_row_q4_k(const BlockQ4_K* src, float* dst, int64_t cols);
void dequantize_row_q5_k(const BlockQ5_K* src, float* dst, int64_t cols);
void dequantize_row_q6_k(const BlockQ6_K* src, float* dst, int64_t cols);
void repack_q4_k_to_q4_ka(const BlockQ4_K* src, BlockQ4_KA* dst, int64_t cols);
void repack_q6_k_to_q6_a(const BlockQ6_K* src, BlockQ6_A* dst, int64_t cols);
void dequantize_row_q4_ka(const BlockQ4_KA* src, float* dst, int64_t cols);
void dequantize_row_q6_a(const BlockQ6_A* src, float* dst, int64_t cols);

void repack_q4_0_to_q4_a(const BlockQ4_0* src, BlockQ4_A* dst, int64_t cols);
void repack_q8_0_to_q8_a(const BlockQ8_0* src, BlockQ8_A* dst, int64_t cols);
void dequantize_row_q4_a(const BlockQ4_A* src, float* dst, int64_t cols);
void dequantize_row_q8_a(const BlockQ8_A* src, float* dst, int64_t cols);

void convert_f32_to_f16(const float* src, uint16_t* dst, int64_t n);
void convert_f16_to_f32(const uint16_t* src, float* dst, int64_t n);
void convert_f32_to_bf16(const float* src, uint16_t* dst, int64_t n);
void convert_bf16_to_f32(const uint16_t* src, float* dst, int64_t n);

}  // namespace fusionrt
