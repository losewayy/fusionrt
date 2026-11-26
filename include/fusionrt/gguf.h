#pragma once

#include "fusionrt/tensor.h"
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace fusionrt {

using GgufMetaValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, float, bool,
    std::string, uint64_t, int64_t, double,
    std::vector<std::string>,
    std::vector<int32_t>,
    std::vector<uint32_t>,
    std::vector<float>
>;

struct GgufTensorInfo {
    std::string name;
    std::vector<int64_t> shape;  // row-major (outer -> inner), converted from GGUF ne[0..n-1]
    DType dtype = DType::F32;
    uint64_t offset = 0;         // relative to tensor_data_start
    size_t nbytes = 0;
};

class GgufFile {
public:
    bool load(const std::string& path, std::string* err = nullptr);

    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }
    const std::map<std::string, GgufMetaValue>& metadata() const { return metadata_; }
    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }

    const GgufTensorInfo* find_tensor(const std::string& name) const;
    HostTensor read_tensor(const std::string& name) const;

    std::string get_string(const std::string& key, const std::string& def = "") const;
    int64_t get_int(const std::string& key, int64_t def = 0) const;
    double get_float(const std::string& key, double def = 0.0) const;
    bool get_bool(const std::string& key, bool def = false) const;
    std::vector<std::string> get_string_array(const std::string& key) const;
    std::vector<float> get_float_array(const std::string& key) const;
    std::vector<int32_t> get_int_array(const std::string& key) const;

    // Minimal GGUF v3 writer for creating synthetic/test weight files
    static bool write_v3(const std::string& path,
                         const std::map<std::string, GgufMetaValue>& meta,
                         const std::vector<std::pair<std::string, HostTensor>>& tensors,
                         std::string* err = nullptr);

private:
    std::string path_;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    std::map<std::string, GgufMetaValue> metadata_;
    std::vector<GgufTensorInfo> tensors_;
    std::map<std::string, size_t> tensor_index_;
};

}  // namespace fusionrt
