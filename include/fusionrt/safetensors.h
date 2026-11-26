#pragma once

#include "fusionrt/tensor.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fusionrt {

struct SafetensorsEntry {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

class SafetensorsFile {
public:
    bool load(const std::string& path, std::string* err = nullptr);

    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    const std::vector<SafetensorsEntry>& tensors() const { return tensors_; }
    const SafetensorsEntry* find_tensor(const std::string& name) const;
    HostTensor read_tensor(const std::string& name) const;

    static bool write(const std::string& path,
                      const std::map<std::string, std::string>& meta,
                      const std::vector<std::pair<std::string, HostTensor>>& tensors,
                      std::string* err = nullptr);

private:
    std::string path_;
    uint64_t data_offset_ = 0;
    std::map<std::string, std::string> metadata_;
    std::vector<SafetensorsEntry> tensors_;
    std::map<std::string, size_t> tensor_index_;
};

}  // namespace fusionrt
