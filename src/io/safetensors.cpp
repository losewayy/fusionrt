#include "fusionrt/safetensors.h"
#include "fusionrt/json.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fusionrt {

namespace {

DType str_to_st_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I32") return DType::I32;
    if (s == "U8") return DType::U8;
    throw std::runtime_error("Unsupported Safetensors dtype: " + s);
}

std::string st_dtype_to_str(DType dt) {
    switch (dt) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I32: return "I32";
        case DType::U8: return "U8";
        default: throw std::runtime_error("Unsupported DType for Safetensors export");
    }
}

}  // namespace

bool SafetensorsFile::load(const std::string& path, std::string* err) {
    try {
        SafetensorsFile loaded;
        loaded.path_ = path;

        std::ifstream ifs(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
        if (!ifs) {
            if (err) *err = "Cannot open safetensors file: " + path;
            return false;
        }
        uint64_t hdr_len = 0;
        if (!ifs.read(reinterpret_cast<char*>(&hdr_len), 8)) {
            if (err) *err = "Failed reading safetensors header length";
            return false;
        }
        if (hdr_len == 0 || hdr_len > 64ULL * 1024ULL * 1024ULL) {
            if (err) *err = "Invalid safetensors header length";
            return false;
        }
        std::string hdr_str(static_cast<size_t>(hdr_len), '\0');
        if (!ifs.read(hdr_str.data(), static_cast<std::streamsize>(hdr_len))) {
            if (err) *err = "Failed reading safetensors JSON header";
            return false;
        }
        loaded.data_offset_ = 8 + hdr_len;
        ifs.seekg(0, std::ios::end);
        const auto file_size = ifs.tellg();
        if (file_size < 0 || static_cast<uint64_t>(file_size) < loaded.data_offset_)
            throw std::runtime_error("Invalid safetensors file size");
        const uint64_t payload_size = static_cast<uint64_t>(file_size) - loaded.data_offset_;
        const auto integer = [](const JsonValue& val) {
            if (!val.is_int() || val.as_int() < 0)
                throw std::runtime_error("Expected non-negative safetensors integer");
            return static_cast<uint64_t>(val.as_int());
        };

        JsonValue root = JsonValue::parse(hdr_str);
        if (!root.is_object()) {
            if (err) *err = "Safetensors header must be JSON object";
            return false;
        }
        for (const auto& [key, val] : root.as_object()) {
            if (key == "__metadata__") {
                if (val.is_object()) {
                    for (const auto& [mk, mv] : val.as_object()) {
                        if (mv.is_string()) loaded.metadata_[mk] = mv.as_string();
                    }
                }
                continue;
            }
            const auto& obj = val.as_object();
            SafetensorsEntry e;
            e.name = key;
            e.dtype = str_to_st_dtype(obj.at("dtype").as_string());
            const auto& dims = obj.at("shape").as_array();
            if (dims.empty() || dims.size() > Tensor{}.shape.size())
                throw std::runtime_error("Invalid safetensors rank for " + key);
            uint64_t elements = 1;
            bool empty = false;
            for (const auto& dim : dims) {
                const auto d = integer(dim);
                if (d && elements > static_cast<uint64_t>(INT64_MAX) / d)
                    throw std::runtime_error("Safetensors element count overflow for " + key);
                elements *= std::max<uint64_t>(1, d);
                empty |= d == 0;
                e.shape.push_back(static_cast<int64_t>(d));
            }
            const auto& offs = obj.at("data_offsets").as_array();
            if (offs.size() != 2) throw std::runtime_error("Invalid safetensors offsets for " + key);
            e.begin = integer(offs[0]);
            e.end = integer(offs[1]);
            const auto type_bytes = dtype_type_size(e.dtype);
            if (e.begin > e.end || e.end > payload_size ||
                elements > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / type_bytes ||
                (empty ? 0 : elements * type_bytes) != e.end - e.begin)
                throw std::runtime_error("Invalid safetensors byte range for " + key);
            loaded.tensors_.push_back(std::move(e));
        }
        std::vector<const SafetensorsEntry*> ranges;
        for (size_t i = 0; i < loaded.tensors_.size(); ++i) {
            const auto& entry = loaded.tensors_[i];
            ranges.push_back(&entry);
            loaded.tensor_index_[entry.name] = i;
        }
        std::sort(ranges.begin(), ranges.end(), [](const auto* a, const auto* b) {
            return a->begin != b->begin ? a->begin < b->begin : a->end < b->end;
        });
        uint64_t previous_end = 0;
        for (const auto* e : ranges) {
            if (e->begin == e->end) continue;
            if (e->begin < previous_end)
                throw std::runtime_error("Overlapping safetensors byte range for " + e->name);
            previous_end = e->end;
        }
        *this = std::move(loaded);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

const SafetensorsEntry* SafetensorsFile::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

HostTensor SafetensorsFile::read_tensor(const std::string& name) const {
    const SafetensorsEntry* e = find_tensor(name);
    if (!e) throw std::runtime_error("Safetensors tensor not found: " + name);
    HostTensor ht = HostTensor::create(e->shape, e->dtype);
    ht.desc.name = name;
    uint64_t len = e->end - e->begin;
    if (len != ht.storage.size()) {
        throw std::runtime_error("Safetensors byte range mismatch for " + name);
    }
    std::ifstream ifs(std::filesystem::path(std::u8string(path_.begin(), path_.end())), std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot reopen safetensors: " + path_);
    ifs.seekg(static_cast<std::streamoff>(data_offset_ + e->begin), std::ios::beg);
    if (len > 0 && !ifs.read(reinterpret_cast<char*>(ht.storage.data()), static_cast<std::streamsize>(len))) {
        throw std::runtime_error("Failed reading safetensors tensor bytes: " + name);
    }
    return ht;
}

bool SafetensorsFile::write(const std::string& path,
                            const std::map<std::string, std::string>& meta,
                            const std::vector<std::pair<std::string, HostTensor>>& tensors,
                            std::string* err) {
    try {
        JsonObject root;
        if (!meta.empty()) {
            JsonObject mobj;
            for (const auto& [k, v] : meta) mobj[k] = JsonValue(v);
            root["__metadata__"] = JsonValue(std::move(mobj));
        }
        uint64_t offset = 0;
        for (const auto& [name, ht] : tensors) {
            JsonObject tobj;
            tobj["dtype"] = JsonValue(st_dtype_to_str(ht.desc.dtype));
            JsonArray sh;
            for (int d = 0; d < ht.desc.ndim; ++d) sh.emplace_back(ht.desc.shape[d]);
            tobj["shape"] = JsonValue(std::move(sh));
            uint64_t sz = ht.storage.size();
            JsonArray offs;
            offs.emplace_back(static_cast<int64_t>(offset));
            offs.emplace_back(static_cast<int64_t>(offset + sz));
            tobj["data_offsets"] = JsonValue(std::move(offs));
            offset += sz;
            root[name] = JsonValue(std::move(tobj));
        }
        std::string hdr = JsonValue(std::move(root)).dump(0);
        while ((hdr.size() % 8) != 0) hdr.push_back(' ');

        std::ofstream ofs(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
        if (!ofs) {
            if (err) *err = "Cannot create safetensors file: " + path;
            return false;
        }
        uint64_t hdr_len = hdr.size();
        ofs.write(reinterpret_cast<const char*>(&hdr_len), 8);
        ofs.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        for (const auto& [name, ht] : tensors) {
            if (!ht.storage.empty()) {
                ofs.write(reinterpret_cast<const char*>(ht.storage.data()),
                          static_cast<std::streamsize>(ht.storage.size()));
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

}  // namespace fusionrt
