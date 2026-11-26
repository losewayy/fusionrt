#include "fusionrt/gguf.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fusionrt {

namespace {

enum class GgufValType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    I32 = 26,
    BF16 = 30,
};

DType ggml_to_dtype(uint32_t t) {
    switch (t) {
        case 0: return DType::F32;
        case 1: return DType::F16;
        case 2: return DType::Q4_0;
        case 3: return DType::Q4_1;
        case 8: return DType::Q8_0;
        case 12: return DType::Q4_K;
        case 13: return DType::Q5_K;
        case 14: return DType::Q6_K;
        case 26: return DType::I32;
        case 30: return DType::BF16;
        default: throw std::runtime_error("Unsupported GGML tensor type: " + std::to_string(t));
    }
}

uint32_t dtype_to_ggml(DType dt) {
    switch (dt) {
        case DType::F32: return 0;
        case DType::F16: return 1;
        case DType::Q4_0: return 2;
        case DType::Q4_1: return 3;
        case DType::Q8_0: return 8;
        case DType::Q4_K: return 12;
        case DType::Q5_K: return 13;
        case DType::Q6_K: return 14;
        case DType::I32: return 26;
        case DType::BF16: return 30;
        default: throw std::runtime_error("Unsupported DType for GGUF export");
    }
}

template <typename T>
T read_le(std::ifstream& ifs) {
    T val{};
    if (!ifs.read(reinterpret_cast<char*>(&val), sizeof(T))) {
        throw std::runtime_error("Unexpected EOF reading GGUF");
    }
    return val;
}

template <typename T>
void write_le(std::ofstream& ofs, const T& val) {
    ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

std::string read_str(std::ifstream& ifs) {
    uint64_t len = read_le<uint64_t>(ifs);
    std::string s(static_cast<size_t>(len), '\0');
    if (len > 0 && !ifs.read(s.data(), static_cast<std::streamsize>(len))) {
        throw std::runtime_error("Unexpected EOF reading GGUF string");
    }
    return s;
}

void write_str(std::ofstream& ofs, const std::string& s) {
    uint64_t len = s.size();
    write_le(ofs, len);
    if (len > 0) ofs.write(s.data(), static_cast<std::streamsize>(len));
}

void skip_gguf_val(std::ifstream& ifs, uint32_t vtype);

GgufMetaValue read_gguf_val(std::ifstream& ifs, uint32_t vtype) {
    switch (static_cast<GgufValType>(vtype)) {
        case GgufValType::UINT8: return read_le<uint8_t>(ifs);
        case GgufValType::INT8: return read_le<int8_t>(ifs);
        case GgufValType::UINT16: return read_le<uint16_t>(ifs);
        case GgufValType::INT16: return read_le<int16_t>(ifs);
        case GgufValType::UINT32: return read_le<uint32_t>(ifs);
        case GgufValType::INT32: return read_le<int32_t>(ifs);
        case GgufValType::FLOAT32: return read_le<float>(ifs);
        case GgufValType::BOOL: return static_cast<bool>(read_le<uint8_t>(ifs) != 0);
        case GgufValType::STRING: return read_str(ifs);
        case GgufValType::UINT64: return read_le<uint64_t>(ifs);
        case GgufValType::INT64: return read_le<int64_t>(ifs);
        case GgufValType::FLOAT64: return read_le<double>(ifs);
        case GgufValType::ARRAY: {
            uint32_t elem_type = read_le<uint32_t>(ifs);
            uint64_t n = read_le<uint64_t>(ifs);
            if (elem_type == static_cast<uint32_t>(GgufValType::STRING)) {
                std::vector<std::string> arr;
                arr.reserve(static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) arr.push_back(read_str(ifs));
                return arr;
            } else if (elem_type == static_cast<uint32_t>(GgufValType::BOOL)) {
                std::vector<bool> arr(static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) arr[i] = read_le<uint8_t>(ifs) != 0;
                return arr;
            } else if (elem_type == static_cast<uint32_t>(GgufValType::INT32)) {
                std::vector<int32_t> arr(static_cast<size_t>(n));
                if (n > 0) ifs.read(reinterpret_cast<char*>(arr.data()), static_cast<std::streamsize>(n * 4));
                return arr;
            } else if (elem_type == static_cast<uint32_t>(GgufValType::UINT32)) {
                std::vector<uint32_t> arr(static_cast<size_t>(n));
                if (n > 0) ifs.read(reinterpret_cast<char*>(arr.data()), static_cast<std::streamsize>(n * 4));
                return arr;
            } else if (elem_type == static_cast<uint32_t>(GgufValType::FLOAT32)) {
                std::vector<float> arr(static_cast<size_t>(n));
                if (n > 0) ifs.read(reinterpret_cast<char*>(arr.data()), static_cast<std::streamsize>(n * 4));
                return arr;
            } else {
                for (uint64_t i = 0; i < n; ++i) skip_gguf_val(ifs, elem_type);
                return uint64_t(n);
            }
        }
    }
    throw std::runtime_error("Unsupported GGUF value type: " + std::to_string(vtype));
}

void skip_gguf_val(std::ifstream& ifs, uint32_t vtype) {
    (void)read_gguf_val(ifs, vtype);
}

void write_gguf_val(std::ofstream& ofs, const GgufMetaValue& mv) {
    if (std::holds_alternative<uint8_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::UINT8));
        write_le(ofs, std::get<uint8_t>(mv));
    } else if (std::holds_alternative<int8_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::INT8));
        write_le(ofs, std::get<int8_t>(mv));
    } else if (std::holds_alternative<uint16_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::UINT16));
        write_le(ofs, std::get<uint16_t>(mv));
    } else if (std::holds_alternative<int16_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::INT16));
        write_le(ofs, std::get<int16_t>(mv));
    } else if (std::holds_alternative<uint32_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::UINT32));
        write_le(ofs, std::get<uint32_t>(mv));
    } else if (std::holds_alternative<int32_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::INT32));
        write_le(ofs, std::get<int32_t>(mv));
    } else if (std::holds_alternative<float>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::FLOAT32));
        write_le(ofs, std::get<float>(mv));
    } else if (std::holds_alternative<bool>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::BOOL));
        write_le<uint8_t>(ofs, std::get<bool>(mv) ? 1 : 0);
    } else if (std::holds_alternative<std::string>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::STRING));
        write_str(ofs, std::get<std::string>(mv));
    } else if (std::holds_alternative<uint64_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::UINT64));
        write_le(ofs, std::get<uint64_t>(mv));
    } else if (std::holds_alternative<int64_t>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::INT64));
        write_le(ofs, std::get<int64_t>(mv));
    } else if (std::holds_alternative<double>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::FLOAT64));
        write_le(ofs, std::get<double>(mv));
    } else if (std::holds_alternative<std::vector<std::string>>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::ARRAY));
        write_le(ofs, static_cast<uint32_t>(GgufValType::STRING));
        const auto& arr = std::get<std::vector<std::string>>(mv);
        write_le<uint64_t>(ofs, arr.size());
        for (const auto& s : arr) write_str(ofs, s);
    } else if (std::holds_alternative<std::vector<bool>>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::ARRAY));
        write_le(ofs, static_cast<uint32_t>(GgufValType::BOOL));
        const auto& arr = std::get<std::vector<bool>>(mv);
        write_le<uint64_t>(ofs, arr.size());
        for (bool value : arr) write_le<uint8_t>(ofs, value);
    } else if (std::holds_alternative<std::vector<int32_t>>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::ARRAY));
        write_le(ofs, static_cast<uint32_t>(GgufValType::INT32));
        const auto& arr = std::get<std::vector<int32_t>>(mv);
        write_le<uint64_t>(ofs, arr.size());
        if (!arr.empty()) ofs.write(reinterpret_cast<const char*>(arr.data()), static_cast<std::streamsize>(arr.size() * 4));
    } else if (std::holds_alternative<std::vector<uint32_t>>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::ARRAY));
        write_le(ofs, static_cast<uint32_t>(GgufValType::UINT32));
        const auto& arr = std::get<std::vector<uint32_t>>(mv);
        write_le<uint64_t>(ofs, arr.size());
        if (!arr.empty()) ofs.write(reinterpret_cast<const char*>(arr.data()), static_cast<std::streamsize>(arr.size() * 4));
    } else if (std::holds_alternative<std::vector<float>>(mv)) {
        write_le(ofs, static_cast<uint32_t>(GgufValType::ARRAY));
        write_le(ofs, static_cast<uint32_t>(GgufValType::FLOAT32));
        const auto& arr = std::get<std::vector<float>>(mv);
        write_le<uint64_t>(ofs, arr.size());
        if (!arr.empty()) ofs.write(reinterpret_cast<const char*>(arr.data()), static_cast<std::streamsize>(arr.size() * 4));
    }
}

}  // namespace

bool GgufFile::load(const std::string& path, std::string* err) {
    try {
        path_ = path;
        metadata_.clear();
        tensors_.clear();
        tensor_index_.clear();

        std::ifstream ifs(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
        if (!ifs) {
            if (err) *err = "Cannot open GGUF file: " + path;
            return false;
        }
        char magic[4]{};
        ifs.read(magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0) {
            if (err) *err = "Invalid GGUF magic";
            return false;
        }
        version_ = read_le<uint32_t>(ifs);
        if (version_ < 2 || version_ > 3) {
            if (err) *err = "Unsupported GGUF version: " + std::to_string(version_);
            return false;
        }
        uint64_t n_tensors = read_le<uint64_t>(ifs);
        uint64_t n_kv = read_le<uint64_t>(ifs);

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_str(ifs);
            uint32_t vtype = read_le<uint32_t>(ifs);
            metadata_[key] = read_gguf_val(ifs, vtype);
        }

        alignment_ = static_cast<uint64_t>(get_int("general.alignment", 32));
        if (alignment_ == 0) alignment_ = 32;

        tensors_.reserve(static_cast<size_t>(n_tensors));
        for (uint64_t i = 0; i < n_tensors; ++i) {
            GgufTensorInfo info;
            info.name = read_str(ifs);
            uint32_t n_dims = read_le<uint32_t>(ifs);
            std::vector<int64_t> ne(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                ne[d] = static_cast<int64_t>(read_le<uint64_t>(ifs));
            }
            // GGUF stores innermost dim first (ne[0]); convert to row-major outer -> inner
            info.shape.assign(ne.rbegin(), ne.rend());
            uint32_t ttype = read_le<uint32_t>(ifs);
            info.dtype = ggml_to_dtype(ttype);
            info.offset = read_le<uint64_t>(ifs);
            Tensor tmp = Tensor::from_shape(info.shape, info.dtype, Backend::CPU);
            info.nbytes = tmp.nbytes();
            tensor_index_[info.name] = tensors_.size();
            tensors_.push_back(std::move(info));
        }

        uint64_t cur = static_cast<uint64_t>(ifs.tellg());
        data_offset_ = ((cur + alignment_ - 1) / alignment_) * alignment_;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

const GgufTensorInfo* GgufFile::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

HostTensor GgufFile::read_tensor(const std::string& name) const {
    const GgufTensorInfo* info = find_tensor(name);
    if (!info) throw std::runtime_error("GGUF tensor not found: " + name);
    HostTensor ht = HostTensor::create(info->shape, info->dtype);
    ht.desc.name = name;
    std::ifstream ifs(std::filesystem::path(std::u8string(path_.begin(), path_.end())), std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot reopen GGUF file: " + path_);
    ifs.seekg(static_cast<std::streamoff>(data_offset_ + info->offset), std::ios::beg);
    if (info->nbytes > 0 && !ifs.read(reinterpret_cast<char*>(ht.storage.data()),
                                     static_cast<std::streamsize>(info->nbytes))) {
        throw std::runtime_error("Failed reading GGUF tensor bytes: " + name);
    }
    return ht;
}

std::string GgufFile::get_string(const std::string& key, const std::string& def) const {
    auto it = metadata_.find(key);
    if (it != metadata_.end() && std::holds_alternative<std::string>(it->second)) {
        return std::get<std::string>(it->second);
    }
    return def;
}

int64_t GgufFile::get_int(const std::string& key, int64_t def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& v = it->second;
    if (std::holds_alternative<int64_t>(v)) return std::get<int64_t>(v);
    if (std::holds_alternative<uint64_t>(v)) return static_cast<int64_t>(std::get<uint64_t>(v));
    if (std::holds_alternative<int32_t>(v)) return std::get<int32_t>(v);
    if (std::holds_alternative<uint32_t>(v)) return static_cast<int64_t>(std::get<uint32_t>(v));
    if (std::holds_alternative<int16_t>(v)) return std::get<int16_t>(v);
    if (std::holds_alternative<uint16_t>(v)) return std::get<uint16_t>(v);
    if (std::holds_alternative<int8_t>(v)) return std::get<int8_t>(v);
    if (std::holds_alternative<uint8_t>(v)) return std::get<uint8_t>(v);
    return def;
}

double GgufFile::get_float(const std::string& key, double def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& v = it->second;
    if (std::holds_alternative<double>(v)) return std::get<double>(v);
    if (std::holds_alternative<float>(v)) return static_cast<double>(std::get<float>(v));
    return def;
}

bool GgufFile::get_bool(const std::string& key, bool def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& v = it->second;
    if (std::holds_alternative<bool>(v)) return std::get<bool>(v);
    if (std::holds_alternative<uint8_t>(v)) return std::get<uint8_t>(v) != 0;
    if (std::holds_alternative<int32_t>(v)) return std::get<int32_t>(v) != 0;
    return def;
}

std::vector<std::string> GgufFile::get_string_array(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it != metadata_.end() && std::holds_alternative<std::vector<std::string>>(it->second)) {
        return std::get<std::vector<std::string>>(it->second);
    }
    return {};
}

std::vector<float> GgufFile::get_float_array(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it != metadata_.end() && std::holds_alternative<std::vector<float>>(it->second)) {
        return std::get<std::vector<float>>(it->second);
    }
    return {};
}

std::vector<int32_t> GgufFile::get_int_array(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it != metadata_.end() && std::holds_alternative<std::vector<int32_t>>(it->second)) {
        return std::get<std::vector<int32_t>>(it->second);
    }
    return {};
}

bool GgufFile::write_v3(const std::string& path,
                        const std::map<std::string, GgufMetaValue>& meta,
                        const std::vector<std::pair<std::string, HostTensor>>& tensors,
                        std::string* err) {
    try {
        std::ofstream ofs(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
        if (!ofs) {
            if (err) *err = "Cannot create GGUF file: " + path;
            return false;
        }
        constexpr uint64_t align = 32;
        ofs.write("GGUF", 4);
        write_le<uint32_t>(ofs, 3);
        write_le<uint64_t>(ofs, tensors.size());
        write_le<uint64_t>(ofs, meta.size());

        for (const auto& [k, v] : meta) {
            write_str(ofs, k);
            write_gguf_val(ofs, v);
        }

        uint64_t offset = 0;
        std::vector<uint64_t> offsets;
        offsets.reserve(tensors.size());
        for (const auto& [name, ht] : tensors) {
            write_str(ofs, name);
            write_le<uint32_t>(ofs, static_cast<uint32_t>(ht.desc.ndim));
            for (int d = ht.desc.ndim - 1; d >= 0; --d) {
                write_le<uint64_t>(ofs, static_cast<uint64_t>(ht.desc.shape[d]));
            }
            write_le<uint32_t>(ofs, dtype_to_ggml(ht.desc.dtype));
            write_le<uint64_t>(ofs, offset);
            offsets.push_back(offset);
            uint64_t sz = ht.desc.nbytes();
            offset += ((sz + align - 1) / align) * align;
        }

        uint64_t cur = static_cast<uint64_t>(ofs.tellp());
        uint64_t pad_hdr = (((cur + align - 1) / align) * align) - cur;
        for (uint64_t i = 0; i < pad_hdr; ++i) ofs.put('\0');

        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& ht = tensors[i].second;
            if (!ht.storage.empty()) {
                ofs.write(reinterpret_cast<const char*>(ht.storage.data()),
                          static_cast<std::streamsize>(ht.storage.size()));
            }
            uint64_t sz = ht.storage.size();
            uint64_t pad = (((sz + align - 1) / align) * align) - sz;
            for (uint64_t p = 0; p < pad; ++p) ofs.put('\0');
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

}  // namespace fusionrt
