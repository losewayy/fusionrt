#include "model_loader.h"
#include "fusionrt/weight_codec.h"
#include "fusionrt/hadamard.h"
#include <unordered_set>
#include <optional>
#include "../core/gdn_shape.h"
#include "fusionrt/gguf.h"
#include "fusionrt/json.h"
#include "fusionrt/safetensors.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace fusionrt::model_io {

static void validate_gdn_layer(const LoadedLayerRaw& layer, int64_t taps, const GatedDeltaNetConfig& cfg,
                               const std::string& prefix) {
    validate_gdn_shape(taps, cfg.head_k_dim, cfg.head_v_dim, cfg.num_k_heads, cfg.num_v_heads);
    const int64_t hidden = cfg.hidden_size, heads = cfg.num_v_heads;
    const int64_t value_width = heads * cfg.head_v_dim;
    const int64_t channels = 2LL * cfg.num_k_heads * cfg.head_k_dim + value_width;
    auto check = [&](const char* name, const HostTensor& tensor, std::initializer_list<int64_t> expected) {
        const auto& d = tensor.desc;
        if (static_cast<size_t>(d.ndim) == expected.size() &&
            std::equal(expected.begin(), expected.end(), d.shape.begin())) return;
        std::ostringstream msg;
        msg << "Invalid GDN tensor shape: " << prefix << name << " expected [";
        const char* separator = "";
        for (int64_t dim : expected) {
            msg << separator << dim;
            separator = ", ";
        }
        msg << "], got [";
        for (int i = 0; i < d.ndim; ++i) msg << (i ? ", " : "") << d.shape[i];
        msg << ']';
        throw std::invalid_argument(msg.str());
    };
    if (layer.conv_w.desc.ndim == 3) check("conv_w", layer.conv_w, {channels, 1, taps});
    else check("conv_w", layer.conv_w, {channels, taps});
    check("conv_bias", layer.conv_bias, {channels});
    check("A_log", layer.A_log, {heads});
    check("dt_bias", layer.dt_bias, {heads});
    check("gdn_norm_w", layer.gdn_norm_w, {cfg.head_v_dim});
    check("qkv_proj", layer.qkv_proj, {channels, hidden});
    check("z_proj", layer.z_proj, {value_width, hidden});
    check("a_proj", layer.a_proj, {heads, hidden});
    check("b_proj", layer.b_proj, {heads, hidden});
    check("o_proj", layer.o_proj, {hidden, value_width});
    check("attn_norm", layer.attn_norm, {hidden});
    check("mlp_norm", layer.mlp_norm, {hidden});
    if (!layer.router_w.bytes().empty()) return;
    if (layer.gate_up.bytes().empty()) {
        check("gate", layer.gate, {cfg.intermediate_size, hidden});
        check("up", layer.up, {cfg.intermediate_size, hidden});
    } else {
        check("gate_up", layer.gate_up, {2LL * cfg.intermediate_size, hidden});
    }
    check("down", layer.down, {hidden, cfg.intermediate_size});
}

HostTensor make_rand_f32(const std::vector<int64_t>& shape, std::mt19937_64& rng, float scale) {
    HostTensor ht = HostTensor::create(shape, DType::F32);
    std::normal_distribution<float> dist(0.0f, scale);
    float* p = ht.data<float>();
    for (int64_t i = 0; i < ht.desc.numel(); ++i) p[i] = dist(rng);
    return ht;
}

HostTensor make_ones_f32(const std::vector<int64_t>& shape) {
    HostTensor ht = HostTensor::create(shape, DType::F32);
    float* p = ht.data<float>();
    for (int64_t i = 0; i < ht.desc.numel(); ++i) p[i] = 1.0f;
    return ht;
}

HostTensor make_zeros_f32(const std::vector<int64_t>& shape) {
    return HostTensor::create(shape, DType::F32);
}

HostTensor quantize_host_matrix(const HostTensor& f32_ht, DType dt) {
    if (dt == DType::F32) return f32_ht;
    std::vector<int64_t> dims(f32_ht.desc.shape.begin(), f32_ht.desc.shape.begin() + f32_ht.desc.ndim);
    HostTensor out = HostTensor::create(dims, dt);
    int64_t cols = dims.back();
    int64_t rows = f32_ht.desc.numel() / cols;
    const float* src = f32_ht.data<float>();
    if (dt == DType::F16) {
        convert_f32_to_f16(src, reinterpret_cast<uint16_t*>(out.storage.data()), f32_ht.desc.numel());
        return out;
    }
    if (dt == DType::BF16) {
        convert_f32_to_bf16(src, reinterpret_cast<uint16_t*>(out.storage.data()), f32_ht.desc.numel());
        return out;
    }
    if (dt == DType::Q8_0) {
        auto* dst = reinterpret_cast<BlockQ8_0*>(out.storage.data());
        int64_t nb = cols / 32;
        for (int64_t r = 0; r < rows; ++r) quantize_row_q8_0(src + r * cols, dst + r * nb, cols);
        return out;
    }
    if (dt == DType::Q4_0) {
        auto* dst = reinterpret_cast<BlockQ4_0*>(out.storage.data());
        int64_t nb = cols / 32;
        for (int64_t r = 0; r < rows; ++r) quantize_row_q4_0(src + r * cols, dst + r * nb, cols);
        return out;
    }
    if (dt == DType::Q2_K) {
        auto* dst = out.data<BlockQ2_K>();
        const int64_t blocks = cols / 256;
        for (int64_t r = 0; r < rows; ++r) quantize_row_q2_k(src + r * cols, dst + r * blocks, cols);
        return out;
    }
    if (dt == DType::Q4_A) {
        std::vector<BlockQ4_0> tmp(static_cast<size_t>(cols / 32));
        auto* dst = reinterpret_cast<BlockQ4_A*>(out.storage.data());
        int64_t nsa = cols / 256;
        for (int64_t r = 0; r < rows; ++r) {
            quantize_row_q4_0(src + r * cols, tmp.data(), cols);
            repack_q4_0_to_q4_a(tmp.data(), dst + r * nsa, cols);
        }
        return out;
    }
    if (dt == DType::Q8_A) {
        std::vector<BlockQ8_0> tmp(static_cast<size_t>(cols / 32));
        auto* dst = reinterpret_cast<BlockQ8_A*>(out.storage.data());
        int64_t nsa = cols / 256;
        for (int64_t r = 0; r < rows; ++r) {
            quantize_row_q8_0(src + r * cols, tmp.data(), cols);
            repack_q8_0_to_q8_a(tmp.data(), dst + r * nsa, cols);
        }
        return out;
    }
    throw std::runtime_error("Unsupported weight dtype for model quantization");
}

HostTensor concat_rows_same_dtype(const std::vector<HostTensor>& parts) {
    if (parts.empty()) throw std::runtime_error("concat_rows_same_dtype: empty parts");
    DType dt = parts[0].desc.dtype;
    int64_t cols = parts[0].desc.shape[1];
    int64_t total_rows = 0;
    size_t total_bytes = 0;
    for (const auto& p : parts) {
        if (p.desc.dtype != dt || p.desc.ndim != 2 || p.desc.shape[1] != cols || !same_weight_basis(p, parts[0])) {
            throw std::runtime_error("concat_rows_same_dtype: incompatible tensor shapes/dtypes");
        }
        total_rows += p.desc.shape[0];
        total_bytes += p.bytes().size();
    }
    HostTensor out = HostTensor::create({total_rows, cols}, dt);
    out.desc.hadamard_block = parts[0].desc.hadamard_block;
    out.hadamard_signs = parts[0].hadamard_signs;
    size_t off = 0;
    for (const auto& p : parts) {
        std::memcpy(out.storage.data() + off, p.bytes().data(), p.bytes().size());
        off += p.bytes().size();
    }
    return out;
}

HostTensor to_f32_host(const HostTensor& ht) {
    if (ht.desc.hadamard_block)
        throw std::runtime_error("Folded tensor requires a Hadamard-aware operation: " + ht.desc.name);
    if (ht.desc.dtype == DType::F32) return ht;
    std::vector<int64_t> dims(ht.desc.shape.begin(), ht.desc.shape.begin() + ht.desc.ndim);
    HostTensor out = HostTensor::create(dims, DType::F32);
    int64_t n = ht.desc.numel();
    if (ht.desc.dtype == DType::F16) {
        convert_f16_to_f32(reinterpret_cast<const uint16_t*>(ht.bytes().data()), out.data<float>(), n);
    } else if (ht.desc.dtype == DType::BF16) {
        convert_bf16_to_f32(reinterpret_cast<const uint16_t*>(ht.bytes().data()), out.data<float>(), n);
    } else if (is_weight_codec_dtype(ht.desc.dtype)) {
        visit_weight_codec(ht.desc.dtype, [&]<class Codec>() {
            for (int64_t block = 0; block < n / Codec::kBlockSize; ++block)
                Codec::dequant_block(ht.bytes().data() + block * Codec::kBlockBytes,
                                     out.data<float>() + block * Codec::kBlockSize);
        });
    } else {
        int64_t cols = dims.back();
        int64_t rows = n / cols;
        int64_t nb = cols / 32;
        if (ht.desc.dtype == DType::Q8_0) {
            const auto* src = reinterpret_cast<const BlockQ8_0*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q8_0(src + r * nb, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_0) {
            const auto* src = reinterpret_cast<const BlockQ4_0*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_0(src + r * nb, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_1) {
            const auto* src = reinterpret_cast<const BlockQ4_1*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_1(src + r * nb, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q2_K) {
            const auto* src = ht.data<BlockQ2_K>();
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q2_k(src + r * (cols / 256), out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_K) {
            int64_t nbk = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ4_K*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_k(src + r * nbk, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q5_K) {
            int64_t nbk = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ5_K*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q5_k(src + r * nbk, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q6_K) {
            int64_t nbk = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ6_K*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q6_k(src + r * nbk, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_KA) {
            const auto* src = ht.data<BlockQ4_KA>();
            for (int64_t r = 0; r < rows; ++r)
                dequantize_row_q4_ka(src + r * (cols / 256), out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q6_A) {
            const auto* src = ht.data<BlockQ6_A>();
            for (int64_t r = 0; r < rows; ++r)
                dequantize_row_q6_a(src + r * (cols / 256), out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_A) {
            int64_t nsa = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ4_A*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_a(src + r * nsa, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q8_A) {
            int64_t nsa = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ8_A*>(ht.bytes().data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q8_a(src + r * nsa, out.data<float>() + r * cols, cols);
        } else {
            throw std::runtime_error("Unsupported dtype conversion to F32");
        }
    }
    return out;
}


namespace {

struct GgufArchitecture {
    std::string_view name;
    bool rope_neox;
    bool recurrent = false;
};

struct GgufHadamard {
    int block = 0;
    bool gdn_v_grouped = false;
    bool explicit_signs = false;
    std::map<int64_t, std::shared_ptr<const std::vector<int8_t>>> signs;
    std::unordered_set<std::string> pending;

    explicit GgufHadamard(const GgufFile& gf) {
        constexpr std::string_view prefix = "prism.hadamard.";
        const auto& meta = gf.metadata();
        const auto first = meta.lower_bound(std::string(prefix));
        if (first == meta.end() || !first->first.starts_with(prefix)) return;
        const auto version = gf.get_int("prism.hadamard.version");
        const auto tied = gf.get_bool("prism.hadamard.tied_output");
        const auto sign_mode = gf.get_string("prism.hadamard.sign_mode");
        explicit_signs = sign_mode == "explicit";
        if ((version != 1 && version != 2) || (version == 2) != tied ||
            gf.get_string("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" ||
            gf.get_string("prism.hadamard.axis") != "input-last-dimension" ||
            (!explicit_signs && sign_mode != "identity"))
            throw std::runtime_error("Unsupported prism.hadamard version, transform, axis or sign mode");
        const auto group = gf.get_int("prism.hadamard.block_size");
        if (group < 64 || group > 1024) throw std::runtime_error("Invalid prism.hadamard.block_size");
        block = static_cast<int>(group);
        validate_hadamard_shape(1, block, block);
        gdn_v_grouped = gf.get_bool("prism.hadamard.gdn_v_grouped");
        auto uints = [&](const std::string& key) {
            std::vector<uint32_t> result;
            if (const auto it = meta.find(key); it != meta.end()) {
                if (const auto* values = std::get_if<std::vector<uint32_t>>(&it->second)) return *values;
                if (const auto* values = std::get_if<std::vector<int32_t>>(&it->second))
                    result.assign(values->begin(), values->end());
            }
            return result;
        };
        if (explicit_signs) parse_signs(uints("prism.hadamard.sign_widths"), uints("prism.hadamard.sign_values"));

        const auto add = [&](const std::string& name) {
            const auto* t = gf.find_tensor(name);
            if (!t || t->shape.size() != 2 || t->shape.back() % block != 0 ||
                (explicit_signs && !signs.contains(t->shape.back())) ||
                !(t->dtype == DType::Q2_0 || t->dtype == DType::PQ2_0 || t->dtype == DType::PTQ1_0) ||
                !pending.insert(name).second)
                throw std::runtime_error("Unsupported or invalid Hadamard-folded tensor: " + name);
        };
        const auto names = gf.get_string_array("prism.hadamard.weight_names");
        const auto inverse = gf.get_string_array("prism.hadamard.inverse_weight_names");
        if (names.empty()) throw std::runtime_error("Missing prism.hadamard.weight_names");
        for (const auto& name : names) {
            if (name == "token_embd.weight")
                throw std::runtime_error("Use prism.hadamard.inverse_weight_names for embeddings");
            add(name);
        }
        for (const auto& name : inverse) {
            if (name != "token_embd.weight") throw std::runtime_error("Unsupported inverse Hadamard lookup: " + name);
            add(name);
        }
        const bool has_output = gf.find_tensor("output.weight") != nullptr;
        if ((tied && (has_output || inverse.empty())) || (!tied && !has_output && !inverse.empty()))
            throw std::runtime_error("Invalid prism.hadamard tied output metadata");
    }

    void parse_signs(const std::vector<uint32_t>& widths, const std::vector<uint32_t>& values) {
        if (widths.empty()) throw std::runtime_error("Missing prism.hadamard.sign_widths");
        size_t offset = 0;
        for (const auto width : widths) {
            if (width == 0 || width % block != 0 || width > values.size() - offset || signs.contains(width))
                throw std::runtime_error("Invalid prism.hadamard sign geometry");
            auto vector = std::make_shared<std::vector<int8_t>>();
            vector->reserve(width);
            for (size_t i = 0; i < width; ++i) {
                const auto value = values[offset++];
                if (value != 1 && value != UINT32_MAX)
                    throw std::runtime_error("prism.hadamard sign must be -1 or 1");
                vector->push_back(value == 1 ? 1 : -1);
            }
            signs.emplace(width, std::move(vector));
        }
        if (offset != values.size()) throw std::runtime_error("prism.hadamard.sign_values length mismatch");
    }

    void attach(HostTensor& tensor, const std::string& name) {
        if (block && pending.erase(name)) {
            tensor.desc.hadamard_block = block;
            if (explicit_signs) tensor.hadamard_signs = signs.at(tensor.desc.shape[tensor.desc.ndim - 1]);
        }
    }
};

constexpr GgufArchitecture supported_architectures[] = {
    {"llama", false}, {"qwen2", true}, {"qwen3", true}, {"qwen35", true, true}, {"qwen35moe", true, true},
    {"qwen3moe", true}, {"deepseek2", false}, {"mistral", true}, {"phi3", true},
    {"gemma", true}, {"gemma4", true}, {"dense", true}, {"hybrid", true, true}, {"moe", true},
};

}  // namespace

bool parse_gguf_model_raw(const std::string& path, int max_seq_override, LoadedModelRaw& out, std::string* err) {
    GgufFile gf;
    if (!gf.load(path, err, false)) return false;

    const std::string arch = gf.get_string("general.architecture", "llama");
    const auto spec = std::find_if(std::begin(supported_architectures), std::end(supported_architectures),
        [&](const GgufArchitecture& entry) { return entry.name == arch; });
    if (spec == std::end(supported_architectures)) {
        if (err) *err = "Unsupported GGUF architecture: " + arch;
        return false;
    }
    out.cfg.model_architecture = arch;
    const bool gemma4 = arch == "gemma4";
    const bool qwen35moe = arch == "qwen35moe";
    const bool deepseek2 = arch == "deepseek2";
    const std::string prefix = arch + ".";
    auto get_arch_int = [&](const std::string& suffix, int64_t def) {
        return gf.get_int(prefix + suffix, def);
    };
    auto get_arch_float = [&](const std::string& suffix, double def) {
        return gf.get_float(prefix + suffix, def);
    };

    if (gemma4) {
        const bool multimodal = std::any_of(gf.tensors().begin(), gf.tensors().end(), [](const GgufTensorInfo& t) {
            return t.name.starts_with("v.") || t.name.starts_with("a.") || t.name.starts_with("mm.") ||
                t.name.starts_with("vision.") || t.name.starts_with("vision_") || t.name.starts_with("audio.");
        });
        if (multimodal) {
            if (err) *err = "Gemma 4 vision and audio tensors are unsupported";
            return false;
        }
    }

    if (!gf.find_tensor("token_embd.weight")) {
        if (err) *err = "GGUF missing required tensor: token_embd.weight";
        return false;
    }
    if (!gf.find_tensor("output_norm.weight")) {
        if (err) *err = "GGUF missing required tensor: output_norm.weight";
        return false;
    }

    std::optional<GgufHadamard> hadamard;
    try {
        hadamard.emplace(gf);
        if (hadamard->block && (gemma4 || deepseek2 || get_arch_int("expert_count", 0) > 0))
            throw std::runtime_error("Hadamard-folded weights require dense or GDN CUDA projections");
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
    out.hadamard_folded = hadamard->block != 0;
    auto req = [&](const std::string& name) -> HostTensor {
        if (!gf.find_tensor(name)) throw std::runtime_error("GGUF missing required tensor: " + name);
        auto t = gf.view_tensor(name);
        hadamard->attach(t, name);
        return t;
    };
    out.tok_embeddings = req("token_embd.weight");
    out.final_norm_w = to_f32_host(gf.view_tensor("output_norm.weight"));
    if (gf.find_tensor("output.weight")) out.lm_head_w = req("output.weight");

    int64_t emb_vocab = out.tok_embeddings.desc.shape[0];
    int64_t emb_hidden = out.tok_embeddings.desc.shape[1];
    int num_layers = static_cast<int>(get_arch_int("block_count", 0));
    if (num_layers <= 0) {
        if (err) *err = "GGUF missing valid " + arch + ".block_count metadata";
        return false;
    }

    int H = static_cast<int>(get_arch_int("embedding_length", emb_hidden));
    int nq = static_cast<int>(get_arch_int("attention.head_count", 8));
    int nkv = static_cast<int>(get_arch_int("attention.head_count_kv", nq));
    int hd = static_cast<int>(get_arch_int("attention.key_length", H / std::max(1, nq)));
    auto& mla = out.cfg.dense.mla;
    if (deepseek2) {
        hd = static_cast<int>(get_arch_int("attention.key_length_mla", hd));
        nkv = 1;
        mla.query_rank = static_cast<int>(get_arch_int("attention.q_lora_rank", 0));
        mla.latent_rank = static_cast<int>(get_arch_int("attention.kv_lora_rank", 0));
        mla.rope_dims = static_cast<int>(get_arch_int("rope.dimension_count", 0));
        mla.value_dim = static_cast<int>(get_arch_int("attention.value_length_mla", get_arch_int("attention.value_length", 0)));
        mla.leading_dense_layers = static_cast<int>(get_arch_int("leading_dense_block_count",
            get_arch_int("expert_count", 0) > 0 ? 0 : num_layers));
        if (mla.query_rank < 0 || mla.latent_rank <= 0 || mla.rope_dims < 2 ||
            hd <= mla.rope_dims || mla.value_dim <= 0 || mla.leading_dense_layers < 0 ||
            mla.leading_dense_layers > num_layers ||
            2 * int64_t{mla.latent_rank} + mla.rope_dims > (48 * 1024 - 128) / 4 ||
            int64_t{nq} * (int64_t{hd} + mla.value_dim + mla.latent_width()) > std::numeric_limits<int>::max()) {
            if (err) *err = "Invalid DeepSeek MLA dimensions";
            return false;
        }
    }
    int inter = static_cast<int>(get_arch_int("feed_forward_length", H * 3));
    int ctx_len = static_cast<int>(get_arch_int("context_length", 4096));
    if (max_seq_override > 0) ctx_len = max_seq_override;
    else ctx_len = std::min(ctx_len, 4096);

    float eps = static_cast<float>(get_arch_float("attention.layer_norm_rms_epsilon", 1e-6));
    float rope_theta = static_cast<float>(get_arch_float("rope.freq_base", 10000.0));
    int rot_dim = static_cast<int>(get_arch_int("rope.dimension_count", hd));
    float partial_rot = (hd > 0) ? (static_cast<float>(rot_dim) / static_cast<float>(hd)) : 1.0f;

    if (!gemma4 && get_arch_int("attention.sliding_window", 0) > 0) {
        if (err) *err = "GGUF architecture " + arch + " requires sliding-window attention";
        return false;
    }
    if (!deepseek2 && get_arch_int("attention.value_length", hd) != hd) {
        if (err) *err = "Distinct GGUF attention key and value lengths are unsupported";
        return false;
    }
    const auto scaling = gf.get_string(prefix + "rope.scaling.type", "none");
    const bool mla_scaled_rope = deepseek2 && (scaling == "linear" || scaling == "yarn");
    if (scaling != "none" && !mla_scaled_rope && !gf.find_tensor("rope_freqs.weight")) {
        if (err) *err = "Unsupported GGUF RoPE scaling: " + scaling;
        return false;
    }
    if (out.tok_embeddings.desc.ndim != 2 || H != emb_hidden || nq <= 0 || nkv <= 0 ||
        nq % nkv != 0 || hd <= 0 || inter <= 0 || ctx_len <= 0 || rot_dim < 2 ||
        rot_dim > hd || (rot_dim % 2) != 0 || !std::isfinite(rope_theta) || rope_theta <= 0 ||
        (deepseek2 && (!(eps > 0) || !std::isfinite(eps)))) {
        if (err) *err = "Invalid GGUF attention or embedding dimensions for " + arch;
        return false;
    }

    out.cfg.vocab_size = static_cast<int>(emb_vocab);
    const auto mtp_layers = (arch == "qwen35" || qwen35moe) ? get_arch_int("nextn_predict_layers", 0) : 0;
    if (mtp_layers < 0 || mtp_layers > 1 || mtp_layers >= num_layers) {
        if (err) *err = "Qwen GGUF supports a single optional NextN block after the trunk";
        return false;
    }
    out.cfg.num_layers = num_layers - static_cast<int>(mtp_layers);
    if (gemma4) {
        out.cfg.embed_scale = std::sqrt(static_cast<float>(H));
        out.cfg.final_logit_softcapping = static_cast<float>(get_arch_float("final_logit_softcapping", 0.0));
    }
    out.cfg.dense.hidden_size = H;
    out.cfg.dense.num_attention_heads = nq;
    out.cfg.dense.num_key_value_heads = nkv;
    out.cfg.dense.head_dim = hd;
    out.cfg.dense.intermediate_size = inter;
    out.cfg.dense.max_seq_len = ctx_len;
    out.cfg.dense.rms_norm_eps = eps;
    out.cfg.dense.rope_theta = rope_theta;
    out.cfg.dense.partial_rotary_factor = partial_rot;
    out.cfg.dense.rope_neox = spec->rope_neox;
    out.cfg.dense.qk_norm = false;
    out.cfg.dense.weight_dtype = out.lm_head_w.bytes().empty() ? out.tok_embeddings.desc.dtype : out.lm_head_w.desc.dtype;

    out.cfg.gdn.hidden_size = H;
    out.cfg.gdn.intermediate_size = inter;
    out.cfg.gdn.rms_norm_eps = eps;
    out.cfg.gdn.tiled_value_heads = arch == "qwen35" || qwen35moe;
    out.cfg.gdn.num_k_heads = static_cast<int>(get_arch_int("ssm.group_count", nkv));
    out.cfg.gdn.num_v_heads = static_cast<int>(get_arch_int("ssm.time_step_rank", nq));
    out.cfg.gdn.head_k_dim = static_cast<int>(get_arch_int("ssm.state_size", hd));
    int ssm_inner = static_cast<int>(get_arch_int("ssm.inner_size", nq * hd));
    out.cfg.gdn.head_v_dim = (out.cfg.gdn.num_v_heads > 0 && ssm_inner >= out.cfg.gdn.num_v_heads && (ssm_inner % out.cfg.gdn.num_v_heads == 0))
        ? (ssm_inner / out.cfg.gdn.num_v_heads)
        : ssm_inner;

    out.cfg.moe.hidden_size = H;
    out.cfg.moe.intermediate_size = static_cast<int>(get_arch_int("expert_feed_forward_length", inter));
    out.cfg.moe.num_experts = static_cast<int>(get_arch_int("expert_count", 0));
    out.cfg.moe.top_k = static_cast<int>(get_arch_int("expert_used_count", 2));
    out.cfg.moe.norm_topk_prob = gf.get_bool(prefix + "expert_weights_norm",
        gf.get_bool(prefix + "norm_topk_prob", !deepseek2));
    const int mla_experts = out.cfg.moe.num_experts;
    if (qwen35moe) {
        const auto scale = static_cast<float>(get_arch_float("expert_weights_scale", 1.0));
        if (!std::isfinite(scale) || scale < 0) {
            if (err) *err = "Invalid Qwen expert_weights_scale";
            return false;
        }
        out.cfg.moe.norm_topk_prob = true;
        out.cfg.moe.expert_weights_scale = scale == 0 ? 1.0f : scale;
    } else if (gemma4) {
        out.cfg.moe.norm_topk_prob = true;
    }
    const int shared_experts = deepseek2 ? static_cast<int>(get_arch_int("expert_shared_count", 0)) : 0;
    if (deepseek2) {
        const auto gating = get_arch_int("expert_gating_func", 1);
        const float scale = static_cast<float>(get_arch_float("expert_weights_scale", 1.0));
        if (gating < 0 || gating > 2 || !std::isfinite(scale) || scale < 0 || shared_experts < 0) {
            if (err) *err = "Invalid DeepSeek MoE routing metadata";
            return false;
        }
        out.cfg.moe.sigmoid_router = gating == 2;
        out.cfg.moe.expert_weights_scale = scale == 0 ? 1.0f : scale;
        const auto groups = std::max<int64_t>(1, get_arch_int("expert_group_count", 1));
        const auto groups_used = get_arch_int("expert_group_used_count", groups);
        if (groups > 128 || groups_used < 1 || groups_used > groups ||
            (groups > 1 && (mla_experts % groups != 0 || mla_experts / groups < 2 ||
                            out.cfg.moe.top_k > groups_used * (mla_experts / groups)))) {
            if (err) *err = "Invalid DeepSeek MoE expert groups";
            return false;
        }
        out.cfg.moe.expert_groups = static_cast<int>(groups);
        out.cfg.moe.expert_groups_used = static_cast<int>(groups_used);
    }
    out.cfg.moe.gpu_resident_experts = out.cfg.moe.num_experts;
    out.cfg.moe.rms_norm_eps = eps;

    auto matrix = [](const HostTensor& tensor, int64_t rows, int64_t cols) {
        return tensor.desc.ndim == 2 && tensor.desc.shape[0] == rows && tensor.desc.shape[1] == cols;
    };
    if ((!out.lm_head_w.bytes().empty() && !matrix(out.lm_head_w, emb_vocab, H)) ||
        out.final_norm_w.desc.ndim != 1 || out.final_norm_w.desc.shape[0] != H) {
        if (err) *err = "Invalid GGUF output weight or norm shape";
        return false;
    }

    bool saw_gdn = false, saw_moe = false;
    out.layers.resize(static_cast<size_t>(num_layers));

    auto layer_int = [&](const std::string& suffix, int index, int fallback) -> int {
        auto meta = gf.metadata().find(prefix + suffix);
        if (meta == gf.metadata().end()) return fallback;
        auto at = [&](const auto& values) {
            if (values.size() != static_cast<size_t>(num_layers)) {
                throw std::runtime_error("Invalid GGUF per-layer array: " + prefix + suffix);
            }
            return static_cast<int>(values[index]);
        };
        if (const auto* values = std::get_if<std::vector<int32_t>>(&meta->second)) return at(*values);
        if (const auto* values = std::get_if<std::vector<uint32_t>>(&meta->second)) return at(*values);
        if (const auto* values = std::get_if<std::vector<bool>>(&meta->second)) return at(*values);
        if (const auto* value = std::get_if<bool>(&meta->second)) return *value;
        return static_cast<int>(get_arch_int(suffix, fallback));
    };

    try {
        const int64_t ple = gemma4 ? get_arch_int("embedding_length_per_layer_input", 0) : 0;
        if (ple < 0 || ple > std::numeric_limits<int>::max() / num_layers)
            throw std::runtime_error("Invalid Gemma 4 per-layer input width");
        if (ple > 0) {
            auto& weights = out.gemma_inputs;
            weights.token_embeddings = req("per_layer_token_embd.weight");
            weights.projection = req("per_layer_model_proj.weight");
            weights.norm = to_f32_host(req("per_layer_proj_norm.weight"));
            const int64_t width = ple * num_layers;
            if (!matrix(weights.token_embeddings, emb_vocab, width) || !matrix(weights.projection, width, H) ||
                weights.norm.desc.ndim != 1 || weights.norm.desc.shape[0] != ple)
                throw std::runtime_error("Invalid Gemma 4 per-layer input weights");
        }
        const int window = gemma4 ? static_cast<int>(get_arch_int("attention.sliding_window", 0)) : 0;
        const int shared_kv = gemma4 ? static_cast<int>(get_arch_int("attention.shared_kv_layers", 0)) : 0;
        const int swa_hd = gemma4 ? static_cast<int>(get_arch_int("attention.key_length_swa", hd)) : hd;
        const int swa_rot = gemma4 ? static_cast<int>(get_arch_int("rope.dimension_count_swa", swa_hd)) : rot_dim;
        const float swa_theta = gemma4 ? static_cast<float>(get_arch_float("rope.freq_base_swa", rope_theta)) : rope_theta;
        if (gemma4 && (window < 0 || shared_kv < 0 || shared_kv >= num_layers || swa_hd < 2 ||
                       get_arch_int("attention.value_length_swa", swa_hd) != swa_hd ||
                       swa_rot < 2 || swa_rot > swa_hd || swa_rot % 2 != 0 ||
                       !(swa_theta > 0) || !std::isfinite(swa_theta) ||
                       !(eps > 0) || !std::isfinite(eps) ||
                       out.cfg.final_logit_softcapping < 0 || !std::isfinite(out.cfg.final_logit_softcapping))) {
            throw std::runtime_error("Invalid Gemma 4 attention or softcapping metadata");
        }
        if (gemma4 && window > 0 && !gf.metadata().contains(prefix + "attention.sliding_window_pattern")) {
            throw std::runtime_error("Gemma 4 GGUF missing attention.sliding_window_pattern");
        }
        int last_kv_source[2] = {-1, -1};
        if (gf.find_tensor("rope_freqs.weight")) {
            out.rope_factors = to_f32_host(req("rope_freqs.weight"));
            if (out.rope_factors.desc.ndim != 1 || out.rope_factors.desc.shape[0] != rot_dim / 2 ||
                std::any_of(out.rope_factors.data<float>(), out.rope_factors.data<float>() + rot_dim / 2,
                            [&](float factor) { return !(factor > 0) || (!gemma4 && !std::isfinite(factor)); })) {
                throw std::runtime_error("Invalid GGUF RoPE frequency factors");
            }
        }
        if (deepseek2) {
            const bool yarn = scaling == "yarn";
            const double factor = mla_scaled_rope ? get_arch_float("rope.scaling.factor", 1.0) : 1.0;
            const double attn_factor = get_arch_float("rope.scaling.attn_factor", 1.0);
            const double log_mul = get_arch_float("rope.scaling.yarn_log_multiplier", 0.0);
            const double ext = yarn ? get_arch_float("rope.scaling.yarn_ext_factor", 1.0) : 0.0;
            const double original = static_cast<double>(get_arch_int("rope.scaling.original_context_length",
                get_arch_int("context_length", 4096)));
            const double beta_fast = get_arch_float("rope.scaling.yarn_beta_fast", 32.0);
            const double beta_slow = get_arch_float("rope.scaling.yarn_beta_slow", 1.0);
            if (!(factor > 0) || !std::isfinite(factor) || !(attn_factor > 0) ||
                !std::isfinite(attn_factor) || !std::isfinite(log_mul) || !std::isfinite(ext) ||
                (yarn && (!(original > 0) || !(rope_theta > 1) ||
                          !(beta_slow > 0) || !(beta_fast > beta_slow) || !std::isfinite(beta_fast)))) {
                throw std::runtime_error("Invalid DeepSeek MLA RoPE scaling");
            }
            // DeepSeek GGUF stores 0.1 * mscale_all_dim. YaRN scales scores and rotary components separately.
            const double log_factor = std::log(factor);
            const double amplitude = attn_factor * (ext != 0 && log_mul == 0 ? 1 + 0.1 * log_factor : 1);
            mla.rope_amplitude = static_cast<float>(amplitude);
            const double score_scale = scaling == "linear" && factor > 1
                ? 1 + 0.1 * log_factor : amplitude * (1 + log_mul * log_factor);
            mla.attention_scale = static_cast<float>(score_scale * score_scale / std::sqrt(hd));
            if (!(mla.attention_scale > 0) || !std::isfinite(mla.attention_scale) ||
                !(mla.rope_amplitude > 0) || !std::isfinite(mla.rope_amplitude)) {
                throw std::runtime_error("Invalid DeepSeek MLA attention scale");
            }
            auto frequencies = HostTensor::create({mla.rope_dims / 2}, DType::F32);
            auto correction = [&](double beta) {
                return mla.rope_dims * std::log(original / (2 * std::numbers::pi * beta)) /
                    (2 * std::log(rope_theta));
            };
            const double low = yarn ? std::max(0.0, std::floor(correction(beta_fast))) : 0.0;
            const double high = yarn ? std::min(mla.rope_dims - 1.0, std::ceil(correction(beta_slow))) : 0.0;
            const float* factors = out.rope_factors.bytes().empty() ? nullptr : out.rope_factors.data<float>();
            for (int i = 0; i < mla.rope_dims / 2; ++i) {
                const double extrapolate = ext * (1 - std::clamp((i - low) / std::max(high - low, 0.001), 0.0, 1.0));
                const double freq = std::pow(rope_theta, -2.0 * i / mla.rope_dims);
                frequencies.data<float>()[i] = static_cast<float>(
                    freq * (extrapolate + (1 - extrapolate) / factor) / (factors ? factors[i] : 1.0f));
            }
            out.rope_factors = std::move(frequencies);
        }
        for (int i = 0; i < num_layers; ++i) {
            const bool mtp = i >= out.cfg.num_layers;
            std::string p = "blk." + std::to_string(i) + ".";
            LoadedLayerRaw& L = out.layers[static_cast<size_t>(i)];
            int layer_nq = nq, layer_nkv = nkv, layer_hd = hd, layer_inter = inter;
            if (gemma4) {
                const int swa = layer_int("attention.sliding_window_pattern", i, 0);
                if (swa < 0 || swa > 1 || (swa && window == 0)) {
                    throw std::runtime_error("Invalid Gemma 4 sliding-window pattern: " + p);
                }
                layer_nq = layer_int("attention.head_count", i, nq);
                layer_nkv = layer_int("attention.head_count_kv", i, nkv);
                layer_hd = swa ? swa_hd : hd;
                layer_inter = layer_int("feed_forward_length", i, inter);
                L.rotary_dims = swa ? swa_rot : rot_dim;
                L.rope_theta = swa ? swa_theta : rope_theta;
                L.sliding_window = swa ? window : 0;
                if (i >= num_layers - shared_kv) {
                    L.kv_source_layer = last_kv_source[swa];
                    if (L.kv_source_layer < 0) throw std::runtime_error("Missing Gemma 4 shared KV source: " + p);
                    const auto& source = out.layers[L.kv_source_layer];
                    if (layer_nkv != 0 && layer_nkv != source.kv_heads) {
                        throw std::runtime_error("Incompatible Gemma 4 shared KV head count: " + p);
                    }
                    layer_nkv = source.kv_heads;
                } else {
                    last_kv_source[swa] = i;
                }
                if (layer_nq <= 0 || layer_nkv <= 0 || layer_nq % layer_nkv != 0 ||
                    layer_hd > 1024 || layer_inter <= 0) {
                    throw std::runtime_error("Invalid Gemma 4 layer dimensions: " + p);
                }
                L.query_heads = layer_nq;
                L.kv_heads = layer_nkv;
                L.head_dim = layer_hd;
                if (i == 0) out.cfg.dense.num_key_value_heads = layer_nkv;
            }

            const bool has_attn = deepseek2 || gf.find_tensor(p + "attn_q.weight") || gf.find_tensor(p + "attn_qkv.weight");
            const bool has_gdn = gf.find_tensor(p + "ssm_conv1d.weight") || gf.find_tensor(p + "gdn_conv.weight");
            if (!deepseek2 && gf.find_tensor(p + "attn_kv_a_mqa.weight")) {
                throw std::runtime_error("GGUF architecture " + arch + " requires multi-head latent attention");
            }
            if (!gemma4 && arch.starts_with("gemma") && (gf.find_tensor(p + "post_ffw_norm.weight") ||
                                               gf.find_tensor(p + "post_attention_norm.weight"))) {
                throw std::runtime_error("GGUF architecture " + arch + " requires Gemma attention and FFN blocks");
            }
            const bool has_moe = gf.find_tensor(p + "ffn_gate_inp.weight") != nullptr;
            if (deepseek2 && has_moe != (i >= mla.leading_dense_layers)) {
                throw std::runtime_error("DeepSeek MLA layer disagrees with leading_dense_block_count: " + p);
            }
            if (mtp && (!has_attn || has_gdn))
                throw std::runtime_error("Qwen NextN requires a full-attention block: " + p);
            if (has_gdn && !spec->recurrent) throw std::runtime_error("Unexpected GGUF recurrent attention: " + arch);
            if (qwen35moe && !has_moe) throw std::runtime_error("Missing Qwen MoE router: " + p);
            if (!has_attn && !has_gdn && !has_moe) throw std::runtime_error("GGUF layer has no supported attention: " + p);
            if (has_moe) {
                saw_moe = true;
                const auto f32_vector = [&](const std::string& name, int width) {
                    auto t = to_f32_host(req(p + name));
                    if (t.desc.ndim != 1 || t.desc.shape[0] != width ||
                        std::any_of(t.data<float>(), t.data<float>() + width, [](float x) { return !std::isfinite(x); }))
                        throw std::runtime_error("Invalid GGUF vector: " + p + name);
                    return t;
                };
                const auto ff_norm = p + "post_attention_norm.weight";
                L.mlp_norm = f32_vector(qwen35moe && gf.find_tensor(ff_norm) ? "post_attention_norm.weight" : "ffn_norm.weight", H);
                if (gemma4) {
                    L.routed_norm = f32_vector("pre_ffw_norm_2.weight", H);
                    L.dense_post_norm = f32_vector("post_ffw_norm_1.weight", H);
                    L.routed_post_norm = f32_vector("post_ffw_norm_2.weight", H);
                    L.router_scale = f32_vector("ffn_gate_inp.scale", H);
                    for (int h = 0; h < H; ++h) L.router_scale.data<float>()[h] /= std::sqrt(static_cast<float>(H));
                }
                L.router_w = to_f32_host(req(p + "ffn_gate_inp.weight"));
                if (L.router_w.desc.shape[0] > std::numeric_limits<int>::max())
                    throw std::runtime_error("GGUF expert count exceeds supported indexing: " + p);
                const int E = static_cast<int>(L.router_w.desc.shape[0]);
                if (L.mlp_norm.desc.ndim != 1 || L.mlp_norm.desc.shape[0] != H ||
                    L.router_w.desc.ndim != 2 || L.router_w.desc.shape[1] != H || E < out.cfg.moe.top_k ||
                    out.cfg.moe.top_k <= 0 || ((gemma4 || qwen35moe) && E != mla_experts)) {
                    throw std::runtime_error("Unsupported GGUF expert router shape: " + p);
                }
                if (deepseek2) {
                    if (E != mla_experts) {
                        throw std::runtime_error("Invalid DeepSeek MoE router or FFN norm: " + p);
                    }
                    if (gf.find_tensor(p + "exp_probs_b.bias")) {
                        L.router_bias = to_f32_host(req(p + "exp_probs_b.bias"));
                        if (L.router_bias.desc.ndim != 1 || L.router_bias.desc.shape[0] != E ||
                            std::any_of(L.router_bias.data<float>(), L.router_bias.data<float>() + E,
                                        [](float bias) { return !std::isfinite(bias); })) {
                            throw std::runtime_error("Invalid DeepSeek MoE router bias: " + p);
                        }
                    }
                }
                out.cfg.moe.num_experts = E;
                out.cfg.moe.gpu_resident_experts = E;
                if (gf.find_tensor(p + "ffn_gate_up_exps.weight")) {
                    L.experts_gate_up = req(p + "ffn_gate_up_exps.weight");
                    if (L.experts_gate_up.desc.ndim != 3 || (L.experts_gate_up.desc.shape[1] % 2) != 0) {
                        throw std::runtime_error("Invalid GGUF packed expert projections: " + p);
                    }
                    out.cfg.moe.intermediate_size = static_cast<int>(L.experts_gate_up.desc.shape[1] / 2);
                } else if (gf.find_tensor(p + "ffn_gate_exps.weight") && gf.find_tensor(p + "ffn_up_exps.weight")) {
                    L.experts_gate_up = req(p + "ffn_gate_exps.weight");
                    L.experts_up = req(p + "ffn_up_exps.weight");
                    const auto& gate = L.experts_gate_up.desc;
                    const auto& up = L.experts_up.desc;
                    if (gate.ndim != 3 || up.ndim != 3 || gate.shape != up.shape) {
                        throw std::runtime_error("Invalid GGUF expert projections: " + p);
                    }
                    out.cfg.moe.intermediate_size = static_cast<int>(gate.shape[1]);
                } else {
                    std::vector<HostTensor> gu_list, dn_list;
                    int64_t expert_width = get_arch_int("expert_feed_forward_length", 0);
                    for (int e = 0; e < E; ++e) {
                        const std::string es = std::to_string(e);
                        const auto gate = req(p + "ffn_gate." + es + ".weight");
                        const auto up = req(p + "ffn_up." + es + ".weight");
                        const auto down = req(p + "ffn_down." + es + ".weight");
                        if (e == 0 && expert_width == 0 && gate.desc.ndim == 2) expert_width = gate.desc.shape[0];
                        if (expert_width <= 0 || expert_width > std::numeric_limits<int>::max() / 2 ||
                            !matrix(gate, expert_width, H) || !matrix(up, expert_width, H) ||
                            !matrix(down, H, expert_width)) {
                            throw std::runtime_error("Invalid GGUF individual expert FFN shapes: " + p + es);
                        }
                        gu_list.push_back(to_f32_host(gate));
                        gu_list.push_back(to_f32_host(up));
                        dn_list.push_back(to_f32_host(down));
                    }
                    L.experts_gate_up = concat_rows_same_dtype(gu_list);
                    L.experts_gate_up.desc = Tensor::from_shape({E, 2 * expert_width, H}, DType::F32, Backend::CPU);
                    L.experts_down = concat_rows_same_dtype(dn_list);
                    L.experts_down.desc = Tensor::from_shape({E, H, expert_width}, DType::F32, Backend::CPU);
                    out.cfg.moe.intermediate_size = static_cast<int>(expert_width);
                }
                if (L.experts_down.bytes().empty()) L.experts_down = req(p + "ffn_down_exps.weight");
                const int I = out.cfg.moe.intermediate_size;
                if (L.experts_gate_up.desc.ndim != 3 || L.experts_gate_up.desc.shape[0] != E ||
                    L.experts_gate_up.desc.shape[1] != (L.experts_up.bytes().empty() ? 2 * I : I) ||
                    L.experts_gate_up.desc.shape[2] != H ||
                    L.experts_down.desc.ndim != 3 || L.experts_down.desc.shape[0] != E ||
                    L.experts_down.desc.shape[1] != H || L.experts_down.desc.shape[2] != I) {
                    throw std::runtime_error("Invalid GGUF expert FFN shapes: " + p);
                }
                if (I <= 0 || E > std::numeric_limits<int>::max() / std::max<int64_t>(2LL * I, H))
                    throw std::runtime_error("GGUF expert data exceeds supported indexing: " + p);
                if (gf.find_tensor(p + "ffn_gate_exps.scale"))
                    L.expert_gate_scale = f32_vector("ffn_gate_exps.scale", E);
                if (gf.find_tensor(p + "ffn_up_exps.scale"))
                    L.expert_up_scale = f32_vector("ffn_up_exps.scale", E);
                if (gf.find_tensor(p + "ffn_down_exps.scale"))
                    L.expert_down_scale = f32_vector("ffn_down_exps.scale", E);
                if (L.experts_up.bytes().empty()) L.expert_gate_scale = L.expert_up_scale;
                for (const auto* weight : {&L.experts_gate_up, &L.experts_up, &L.experts_down}) {
                    if (weight->desc.dtype == DType::I32) {
                        throw std::runtime_error("Unsupported GGUF expert weight dtype: " + p);
                    }
                }
                if (gf.find_tensor(p + "ffn_gate_shexp.weight") && gf.find_tensor(p + "ffn_up_shexp.weight") && gf.find_tensor(p + "ffn_down_shexp.weight")) {
                    L.has_shared = true;
                    out.cfg.moe.has_shared_expert = true;
                    auto gate = req(p + "ffn_gate_shexp.weight");
                    auto up = req(p + "ffn_up_shexp.weight");
                    L.shared_down = req(p + "ffn_down_shexp.weight");
                    if (deepseek2 || qwen35moe) {
                        const int64_t width = deepseek2 ? int64_t{I} * shared_experts :
                            get_arch_int("expert_shared_feed_forward_length", layer_inter);
                        if (width <= 0 || width > std::numeric_limits<int>::max() / 2 ||
                            !matrix(gate, width, H) || !matrix(up, width, H) || !matrix(L.shared_down, H, width)) {
                            throw std::runtime_error("Invalid MoE shared expert shapes: " + p);
                        }
                        if (gate.desc.dtype != up.desc.dtype) {
                            gate = quantize_host_matrix(to_f32_host(gate), DType::F16);
                            up = quantize_host_matrix(to_f32_host(up), DType::F16);
                        }
                    }
                    L.shared_gate_up = concat_rows_same_dtype({gate, up});
                    out.cfg.moe.shared_intermediate_size = static_cast<int>(L.shared_down.desc.shape[1]);
                } else {
                    out.cfg.moe.has_shared_expert = false;
                }
                if (deepseek2 && shared_experts > 0 && !L.has_shared) {
                    throw std::runtime_error("Missing DeepSeek MoE shared expert weights: " + p);
                }
                if (qwen35moe || gf.find_tensor(p + "ffn_gate_inp_shexp.weight")) {
                    if (!L.has_shared) throw std::runtime_error("Incomplete GGUF gated shared expert: " + p);
                    L.shared_gate = f32_vector("ffn_gate_inp_shexp.weight", H);
                    L.shared_gate.desc = L.shared_gate.desc.view({1, H});
                }
                if (!has_attn && !has_gdn) {
                    if (gemma4 || qwen35moe) throw std::runtime_error("Missing attention in GGUF MoE block: " + p);
                    L.kind = 2;
                    continue;
                }
            }

            if (has_gdn) {
                L.kind = 1;
                saw_gdn = true;
                L.attn_norm = to_f32_host(req(p + "attn_norm.weight"));
                L.qkv_proj = gf.find_tensor(p + "attn_qkv.weight") ? req(p + "attn_qkv.weight") : req(p + "gdn_qkv.weight");
                L.z_proj = gf.find_tensor(p + "attn_gate.weight") ? req(p + "attn_gate.weight") :
                    (gf.find_tensor(p + "ssm_z.weight") ? req(p + "ssm_z.weight") : req(p + "gdn_z.weight"));
                L.a_proj = gf.find_tensor(p + "ssm_alpha.weight") ? req(p + "ssm_alpha.weight") : req(p + "gdn_a.weight");
                L.b_proj = gf.find_tensor(p + "ssm_beta.weight") ? req(p + "ssm_beta.weight") : req(p + "gdn_b.weight");
                L.conv_w = to_f32_host(gf.find_tensor(p + "ssm_conv1d.weight") ? gf.view_tensor(p + "ssm_conv1d.weight") :
                    req(p + "gdn_conv.weight"));
                if (gf.find_tensor(p + "ssm_conv1d.bias")) {
                    L.conv_bias = to_f32_host(gf.view_tensor(p + "ssm_conv1d.bias"));
                } else if (gf.find_tensor(p + "gdn_conv.bias")) {
                    L.conv_bias = to_f32_host(gf.view_tensor(p + "gdn_conv.bias"));
                } else {
                    L.conv_bias = make_zeros_f32({L.conv_w.desc.shape[0]});
                }
                if (gf.find_tensor(p + "ssm_a")) {
                    L.A_log = to_f32_host(gf.view_tensor(p + "ssm_a"));
                    float* ap = L.A_log.data<float>();
                    for (int64_t k = 0; k < L.A_log.desc.numel(); ++k) {
                        if (ap[k] < 0.0f) ap[k] = std::log(std::max(-ap[k], 1e-30f));
                    }
                } else {
                    L.A_log = to_f32_host(req(p + "gdn_a_log"));
                }
                L.dt_bias = to_f32_host(gf.find_tensor(p + "ssm_dt.bias") ? gf.view_tensor(p + "ssm_dt.bias") : req(p + "gdn_dt_bias"));
                L.gdn_norm_w = to_f32_host(gf.find_tensor(p + "ssm_norm.weight") ? gf.view_tensor(p + "ssm_norm.weight") : req(p + "gdn_norm.weight"));
                L.o_proj = gf.find_tensor(p + "ssm_out.weight") ? req(p + "ssm_out.weight") : req(p + "attn_output.weight");
            } else if (deepseek2) {
                const int nope = hd - mla.rope_dims;
                auto norm = [&](const char* name, int width) {
                    auto t = to_f32_host(req(p + name));
                    if (t.desc.ndim != 1 || t.desc.shape[0] != width) {
                        throw std::runtime_error("Invalid DeepSeek MLA norm shape: " + p + name);
                    }
                    return t;
                };
                L.attn_norm = norm("attn_norm.weight", H);
                if (mla.query_rank > 0) {
                    L.mla_q_a = req(p + "attn_q_a.weight");
                    L.mla_q_a_norm = norm("attn_q_a_norm.weight", mla.query_rank);
                    if (!matrix(L.mla_q_a, mla.query_rank, H)) {
                        throw std::runtime_error("Invalid DeepSeek MLA Q-A shape: " + p);
                    }
                }
                L.q_proj = req(p + (mla.query_rank > 0 ? "attn_q_b.weight" : "attn_q.weight"));
                L.mla_kv_a = req(p + "attn_kv_a_mqa.weight");
                L.mla_kv_a_norm = norm("attn_kv_a_norm.weight", mla.latent_rank);
                L.o_proj = req(p + "attn_output.weight");
                if (!matrix(L.q_proj, nq * hd, mla.query_rank > 0 ? mla.query_rank : H) ||
                    !matrix(L.mla_kv_a, mla.latent_width(), H) || !matrix(L.o_proj, H, nq * mla.value_dim)) {
                    throw std::runtime_error("Invalid DeepSeek MLA projection shapes: " + p);
                }
                if (gf.find_tensor(p + "attn_k_b.weight") || gf.find_tensor(p + "attn_v_b.weight")) {
                    L.mla_k_b = req(p + "attn_k_b.weight");
                    L.mla_v_b = req(p + "attn_v_b.weight");
                    const auto& k = L.mla_k_b.desc;
                    const auto& v = L.mla_v_b.desc;
                    if (k.ndim != 3 || k.shape[0] != nq || k.shape[1] != mla.latent_rank || k.shape[2] != nope ||
                        v.ndim != 3 || v.shape[0] != nq || v.shape[1] != mla.value_dim || v.shape[2] != mla.latent_rank) {
                        throw std::runtime_error("Invalid DeepSeek MLA split KV-B shapes: " + p);
                    }
                } else {
                    L.mla_kv_b = req(p + "attn_kv_b.weight");
                    if (!matrix(L.mla_kv_b, nq * (nope + mla.value_dim), mla.latent_rank)) {
                        throw std::runtime_error("Invalid DeepSeek MLA combined KV-B shape: " + p);
                    }
                }
            } else {
                L.kind = 0;
                L.attn_norm = to_f32_host(req(p + "attn_norm.weight"));
                auto norm = [&](const std::string& name, int width) {
                    auto t = to_f32_host(req(p + name));
                    if (t.desc.ndim != 1 || t.desc.shape[0] != width) {
                        throw std::runtime_error("Invalid Gemma 4 normalization or scale shape: " + p + name);
                    }
                    return t;
                };
                if (gemma4) {
                    if (L.attn_norm.desc.ndim != 1 || L.attn_norm.desc.shape[0] != H) {
                        throw std::runtime_error("Invalid Gemma 4 attention norm shape: " + p);
                    }
                    L.attn_post_norm = norm(gf.find_tensor(p + "post_attention_norm.weight")
                        ? "post_attention_norm.weight" : "attn_post_norm.weight", H);
                    L.ffn_post_norm = norm(gf.find_tensor(p + "post_ffw_norm.weight")
                        ? "post_ffw_norm.weight" : "ffn_post_norm.weight", H);
                    if (gf.find_tensor(p + "layer_output_scale.weight")) {
                        L.output_scale = norm("layer_output_scale.weight", 1).data<float>()[0];
                        if (!std::isfinite(L.output_scale)) throw std::runtime_error("Invalid Gemma 4 output scale: " + p);
                    }
                    if (ple > 0) {
                        L.ple_gate = req(p + "inp_gate.weight");
                        L.ple_projection = req(p + "proj.weight");
                        L.ple_post_norm = norm("post_norm.weight", H);
                        if (!matrix(L.ple_gate, ple, H) || !matrix(L.ple_projection, H, ple))
                            throw std::runtime_error("Invalid Gemma 4 per-layer projection shapes: " + p);
                    }
                }
                const bool owns_kv = L.kv_source_layer < 0;
                const int q_dim = layer_nq * layer_hd;
                const int kv_dim = layer_nkv * layer_hd;
                if (gf.find_tensor(p + "attn_qkv.weight")) {
                    L.qkv_proj = req(p + "attn_qkv.weight");
                    // Split q, k, v host slices for unfused/Vulkan paths
                    if (L.qkv_proj.desc.ndim != 2 || L.qkv_proj.desc.shape[1] != H ||
                        L.qkv_proj.desc.shape[0] != q_dim + 2 * kv_dim) {
                        throw std::runtime_error("Invalid GGUF packed QKV shape: " + p + "attn_qkv.weight");
                    }
                    L.q_proj = L.qkv_proj.slice_dim0(0, q_dim);
                    if (owns_kv) {
                        L.k_proj = L.qkv_proj.slice_dim0(q_dim, kv_dim);
                        L.v_proj = L.qkv_proj.slice_dim0(q_dim + kv_dim, kv_dim);
                    }
                } else {
                    L.q_proj = req(p + "attn_q.weight");
                    if (owns_kv) {
                        L.k_proj = req(p + "attn_k.weight");
                        if (!gemma4 || gf.find_tensor(p + "attn_v.weight")) L.v_proj = req(p + "attn_v.weight");
                    }
                    L.gated_q_attn = !gemma4 && L.q_proj.desc.shape[0] == static_cast<int64_t>(2 * q_dim);
                    if (L.gated_q_attn) out.cfg.dense.gated_q_attn = true;
                    if (!matrix(L.q_proj, L.gated_q_attn ? 2 * q_dim : q_dim, H) ||
                        (owns_kv && !matrix(L.k_proj, kv_dim, H)) ||
                        (!L.v_proj.bytes().empty() && !matrix(L.v_proj, kv_dim, H))) {
                        throw std::runtime_error("Invalid GGUF QKV projection shapes: " + p);
                    }
                }
                auto bias = [&](const char* name, int64_t width) {
                    auto key = p + name;
                    if (!gf.find_tensor(key)) return HostTensor{};
                    auto tensor = req(key);
                    if (tensor.desc.ndim != 1 || tensor.desc.shape[0] != width) {
                        throw std::runtime_error("Invalid GGUF bias shape: " + key);
                    }
                    return to_f32_host(tensor);
                };
                L.q_bias = bias("attn_q.bias", L.q_proj.desc.shape[0]);
                if (owns_kv) L.k_bias = bias("attn_k.bias", L.k_proj.desc.shape[0]);
                if (!L.v_proj.bytes().empty()) L.v_bias = bias("attn_v.bias", L.v_proj.desc.shape[0]);
                const bool has_q_norm = gf.find_tensor(p + "attn_q_norm.weight") != nullptr;
                const bool has_k_norm = gf.find_tensor(p + "attn_k_norm.weight") != nullptr;
                if (gemma4) {
                    L.q_norm = norm("attn_q_norm.weight", layer_hd);
                    if (owns_kv) L.k_norm = norm("attn_k_norm.weight", layer_hd);
                    out.cfg.dense.qk_norm = true;
                } else if (has_q_norm != has_k_norm) throw std::runtime_error("Incomplete GGUF Q/K normalization: " + p);
                if (!gemma4 && has_q_norm) {
                    out.cfg.dense.qk_norm = true;
                    L.q_norm = to_f32_host(gf.view_tensor(p + "attn_q_norm.weight"));
                    L.k_norm = to_f32_host(gf.view_tensor(p + "attn_k_norm.weight"));
                    if (L.q_norm.desc.ndim != 1 || L.k_norm.desc.ndim != 1 ||
                        L.q_norm.desc.shape[0] != layer_hd || L.k_norm.desc.shape[0] != layer_hd) {
                        throw std::runtime_error("Invalid GGUF Q/K normalization shape: " + p);
                    }
                }
                L.o_proj = req(p + "attn_output.weight");
                if (!matrix(L.o_proj, H, q_dim)) throw std::runtime_error("Invalid GGUF attention output shape: " + p);
            }
            if (has_moe && !gemma4) {
                if (L.kind == 1) validate_gdn_layer(L, get_arch_int("ssm.conv_kernel", 4), out.cfg.gdn, p);
                continue;
            }
            L.mlp_norm = to_f32_host(!gemma4 && gf.find_tensor(p + "post_attention_norm.weight") ? gf.view_tensor(p + "post_attention_norm.weight") : req(p + "ffn_norm.weight"));
            if ((gemma4 || deepseek2) && (L.mlp_norm.desc.ndim != 1 || L.mlp_norm.desc.shape[0] != H)) {
                throw std::runtime_error("Invalid GGUF FFN norm shape: " + p);
            }
            if (gf.find_tensor(p + "ffn_gate_up.weight")) {
                L.gate_up = req(p + "ffn_gate_up.weight");
            } else {
                auto up = req(p + "ffn_up.weight");
                if (!gf.find_tensor(p + "ffn_gate.weight") && up.desc.shape[0] == 2 * layer_inter) {
                    L.gate_up = std::move(up);  // Phi-3 stores gate and up in one tensor.
                } else {
                    auto gate = req(p + "ffn_gate.weight");
                    if (gate.desc.dtype == up.desc.dtype && same_weight_basis(gate, up)) {
                        L.gate_up = concat_rows_same_dtype({gate, up});
                    } else {
                        L.gate = std::move(gate);
                        L.up = std::move(up);
                    }
                }
            }
            L.down = req(p + "ffn_down.weight");
            const bool valid_gate_up = L.gate_up.bytes().empty()
                ? matrix(L.gate, layer_inter, H) && matrix(L.up, layer_inter, H)
                : matrix(L.gate_up, 2 * layer_inter, H);
            if (!valid_gate_up || !matrix(L.down, H, layer_inter)) {
                throw std::runtime_error("Invalid GGUF FFN projection shapes: " + p);
            }
            if (L.kind == 1) {
                validate_gdn_layer(L, get_arch_int("ssm.conv_kernel", 4), out.cfg.gdn, p);
                if (hadamard->gdn_v_grouped && out.cfg.gdn.tiled_value_heads && L.o_proj.desc.hadamard_block) {
                    L.o_proj.desc.hadamard_head_dim = out.cfg.gdn.head_v_dim;
                    L.o_proj.desc.hadamard_key_heads = out.cfg.gdn.num_k_heads;
                }
            }
        }
        if (mtp_layers) {
            const std::string nextn = "blk." + std::to_string(out.cfg.num_layers) + ".nextn.";
            auto weights = [&](const std::string& name, int64_t rows, int64_t cols, bool optional) {
                const auto key = nextn + name + ".weight";
                if (optional && !gf.find_tensor(key)) return HostTensor{};
                auto t = req(key);
                if (!matrix(t, rows, cols)) throw std::runtime_error("Invalid Qwen NextN matrix: " + key);
                return t;
            };
            auto norm = [&](const std::string& name, bool optional) {
                const auto key = nextn + name + ".weight";
                if (optional && !gf.find_tensor(key)) return HostTensor{};
                auto t = to_f32_host(req(key));
                if (t.desc.ndim != 1 || t.desc.shape[0] != H)
                    throw std::runtime_error("Invalid Qwen NextN norm: " + key);
                return t;
            };
            out.mtp_projection = weights("eh_proj", H, 2LL * H, false);
            out.mtp_embedding = weights("embed_tokens", emb_vocab, H, true);
            out.mtp_head = weights("shared_head_head", emb_vocab, H, true);
            out.mtp_enorm = norm("enorm", false);
            out.mtp_hnorm = norm("hnorm", false);
            out.mtp_head_norm = norm("shared_head_norm", true);
            const auto scale = [&](std::string_view name) {
                const auto key = nextn + std::string(name) + ".scale";
                if (!gf.find_tensor(key)) return 1.f;
                const auto t = to_f32_host(req(key));
                if (t.desc.ndim != 1 || t.desc.shape[0] != 1 || !std::isfinite(t.data<float>()[0]))
                    throw std::runtime_error("Invalid Qwen NextN scale: " + key);
                return t.data<float>()[0];
            };
            out.mtp_projection_scale = scale("eh_proj");
            if (!out.mtp_head.bytes().empty()) out.mtp_head_scale = scale("shared_head_head");
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }

    if (!hadamard->pending.empty()) {
        if (err) *err = "Hadamard-folded tensor has no supported projection: " + *hadamard->pending.begin();
        return false;
    }
    if (saw_gdn) out.cfg.arch = "hybrid";
    else if (saw_moe) out.cfg.arch = "moe";
    else out.cfg.arch = "dense";
    return true;
}

bool parse_safetensors_model_raw(const std::string& path, int max_seq_override, LoadedModelRaw& out, std::string* err) try {
    SafetensorsFile sf;
    if (!sf.load(path, err)) return false;

    std::filesystem::path p(std::u8string(path.begin(), path.end()));
    std::filesystem::path cfg_path = p.parent_path() / "config.json";
    if (!std::filesystem::exists(cfg_path)) {
        if (err) *err = "Safetensors directory missing required config.json next to " + path;
        return false;
    }

    JsonObject cobj;
    try {
        std::ifstream ifs(cfg_path);
        std::ostringstream ss;
        ss << ifs.rdbuf();
        JsonValue cv = JsonValue::parse(ss.str());
        if (!cv.is_object()) throw std::runtime_error("config.json must be a JSON object");
        cobj = cv.as_object();
    } catch (const std::exception& e) {
        if (err) *err = std::string("Failed parsing config.json: ") + e.what();
        return false;
    }

    auto get_int = [&](const std::string& k, int def) {
        const auto it = cobj.find(k);
        if (it == cobj.end()) return def;
        if (!it->second.is_int() || it->second.as_int() < 0 || it->second.as_int() > INT_MAX)
            throw std::runtime_error("Invalid config.json dimension: " + k);
        return static_cast<int>(it->second.as_int());
    };
    auto get_flt = [&](const std::string& k, float def) {
        return cobj.count(k) ? static_cast<float>(cobj.at(k).as_double(def)) : def;
    };
    auto get_bool = [&](const std::string& k, bool def) {
        if (cobj.count(k) && cobj.at(k).is_bool()) return std::get<bool>(cobj.at(k).val);
        return def;
    };

    std::string model_type = cobj.count("model_type") ? cobj.at("model_type").as_string() : "llama";
    int H = get_int("hidden_size", 0);
    int num_layers = get_int("num_hidden_layers", 0);
    int nq = get_int("num_attention_heads", 0);
    int nkv = get_int("num_key_value_heads", nq);
    int hd = get_int("head_dim", (nq > 0) ? (H / nq) : 0);
    int inter = get_int("intermediate_size", 0);
    int vocab = get_int("vocab_size", 0);
    if (H <= 0 || num_layers <= 0 || static_cast<size_t>(num_layers) > sf.tensors().size() ||
        nq <= 0 || nkv <= 0 || nq % nkv != 0 || hd <= 0 || nq > INT_MAX / hd ||
        inter <= 0 || inter > INT_MAX / 2 || vocab <= 0) {
        if (err) *err = "config.json missing required model dimensions (hidden_size/num_hidden_layers/num_attention_heads/intermediate_size/vocab_size)";
        return false;
    }

    const auto req = [&](const std::string& name, std::initializer_list<int64_t> dimensions = {}) {
        const auto* entry = sf.find_tensor(name);
        if (!entry) throw std::runtime_error("Safetensors missing required tensor: " + name);
        if (dimensions.size() && !std::equal(dimensions.begin(), dimensions.end(),
                                            entry->shape.begin(), entry->shape.end()))
            throw std::runtime_error("Safetensors shape disagrees with config.json: " + name);
        if (entry->dtype != DType::F32 && entry->dtype != DType::F16 && entry->dtype != DType::BF16)
            throw std::runtime_error("Unsupported Safetensors model weight dtype: " + name);
        return sf.read_tensor(name);
    };
    out.tok_embeddings = req("model.embed_tokens.weight", {vocab, H});
    out.final_norm_w = to_f32_host(req("model.norm.weight", {H}));
    if (sf.find_tensor("lm_head.weight")) {
        out.lm_head_w = req("lm_head.weight", {vocab, H});
    } else if (get_bool("tie_word_embeddings", true)) {
        out.lm_head_w = out.tok_embeddings;
    } else {
        if (err) *err = "Safetensors missing lm_head.weight (and tie_word_embeddings=false)";
        return false;
    }

    int ctx_len = get_int("max_position_embeddings", 4096);
    if (ctx_len <= 0) throw std::runtime_error("Invalid config.json context length");
    if (max_seq_override > 0) ctx_len = max_seq_override;
    else ctx_len = std::min(ctx_len, 4096);

    out.cfg.vocab_size = vocab;
    out.cfg.num_layers = num_layers;
    out.cfg.dense.hidden_size = H;
    out.cfg.dense.num_attention_heads = nq;
    out.cfg.dense.num_key_value_heads = nkv;
    out.cfg.dense.head_dim = hd;
    out.cfg.dense.intermediate_size = inter;
    out.cfg.dense.max_seq_len = ctx_len;
    out.cfg.dense.rms_norm_eps = get_flt("rms_norm_eps", 1e-6f);
    out.cfg.dense.rope_theta = get_flt("rope_theta", 10000.0f);
    out.cfg.dense.partial_rotary_factor = get_flt("partial_rotary_factor", 1.0f);
    out.cfg.dense.rope_neox = (model_type != "llama_interleaved");
    out.cfg.dense.qk_norm = false;
    out.cfg.dense.weight_dtype = out.lm_head_w.desc.dtype;

    out.cfg.gdn.hidden_size = H;
    out.cfg.gdn.intermediate_size = inter;
    out.cfg.gdn.rms_norm_eps = out.cfg.dense.rms_norm_eps;
    out.cfg.gdn.num_k_heads = get_int("linear_num_key_heads", nkv);
    out.cfg.gdn.num_v_heads = get_int("linear_num_value_heads", nq);
    out.cfg.gdn.head_k_dim = get_int("linear_key_head_dim", hd);
    out.cfg.gdn.head_v_dim = get_int("linear_value_head_dim", hd);

    out.cfg.moe.hidden_size = H;
    out.cfg.moe.intermediate_size = get_int("moe_intermediate_size", inter);
    out.cfg.moe.num_experts = get_int("num_experts", get_int("num_local_experts", 0));
    out.cfg.moe.top_k = get_int("num_experts_per_tok", 2);
    out.cfg.moe.gpu_resident_experts = out.cfg.moe.num_experts;
    out.cfg.moe.rms_norm_eps = out.cfg.dense.rms_norm_eps;

    bool saw_gdn = false, saw_moe = false;
    out.layers.resize(static_cast<size_t>(num_layers));
    for (int i = 0; i < num_layers; ++i) {
        std::string pfx = "model.layers." + std::to_string(i) + ".";
        LoadedLayerRaw& L = out.layers[static_cast<size_t>(i)];
        if (sf.find_tensor(pfx + "linear_attn.conv1d.weight")) {
            L.kind = 1;
            saw_gdn = true;
            L.attn_norm = to_f32_host(req(pfx + "input_layernorm.weight"));
            L.qkv_proj = req(pfx + "linear_attn.in_proj_qkv.weight");
            L.z_proj = req(pfx + "linear_attn.in_proj_z.weight");
            L.a_proj = req(pfx + "linear_attn.in_proj_a.weight");
            L.b_proj = req(pfx + "linear_attn.in_proj_b.weight");
            L.conv_w = to_f32_host(req(pfx + "linear_attn.conv1d.weight"));
            L.conv_bias = sf.find_tensor(pfx + "linear_attn.conv1d.bias")
                ? to_f32_host(sf.read_tensor(pfx + "linear_attn.conv1d.bias"))
                : make_zeros_f32({L.conv_w.desc.shape[0]});
            L.A_log = to_f32_host(req(pfx + "linear_attn.A_log"));
            L.dt_bias = to_f32_host(req(pfx + "linear_attn.dt_bias"));
            L.gdn_norm_w = to_f32_host(req(pfx + "linear_attn.norm.weight"));
            L.o_proj = req(pfx + "linear_attn.out_proj.weight");
            L.mlp_norm = to_f32_host(req(pfx + "post_attention_layernorm.weight"));
            L.gate_up = concat_rows_same_dtype({req(pfx + "mlp.gate_proj.weight", {inter, H}), req(pfx + "mlp.up_proj.weight", {inter, H})});
            L.down = req(pfx + "mlp.down_proj.weight", {H, inter});
            validate_gdn_layer(L, get_int("linear_conv_kernel_dim", 4), out.cfg.gdn, pfx);
        } else if (sf.find_tensor(pfx + "mlp.gate.weight")) {
            L.kind = 2;
            saw_moe = true;
            L.attn_norm = to_f32_host(req(pfx + "input_layernorm.weight", {H}));
            const auto& router = *sf.find_tensor(pfx + "mlp.gate.weight");
            const auto experts = router.shape[0];
            if (experts <= 0 || experts > INT_MAX ||
                experts > static_cast<int64_t>(sf.tensors().size()) / 3 ||
                (out.cfg.moe.num_experts && experts != out.cfg.moe.num_experts) ||
                out.cfg.moe.top_k <= 0 || out.cfg.moe.top_k > experts)
                throw std::runtime_error("Invalid Safetensors MoE router geometry: " + pfx);
            const int E = static_cast<int>(experts);
            L.router_w = to_f32_host(req(pfx + "mlp.gate.weight", {E, H}));
            const auto* first_gate = sf.find_tensor(pfx + "mlp.experts.0.gate_proj.weight");
            if (!first_gate || first_gate->shape[0] <= 0 || first_gate->shape[0] > INT_MAX / 2)
                throw std::runtime_error("Invalid Safetensors expert geometry: " + pfx);
            const int64_t inter_e = get_int("moe_intermediate_size", static_cast<int>(first_gate->shape[0]));
            out.cfg.moe.num_experts = E;
            out.cfg.moe.gpu_resident_experts = E;
            std::vector<HostTensor> gu_list, dn_list;
            for (int e = 0; e < E; ++e) {
                std::string ep = pfx + "mlp.experts." + std::to_string(e) + ".";
                gu_list.push_back(to_f32_host(req(ep + "gate_proj.weight", {inter_e, H})));
                gu_list.push_back(to_f32_host(req(ep + "up_proj.weight", {inter_e, H})));
                dn_list.push_back(to_f32_host(req(ep + "down_proj.weight", {H, inter_e})));
            }
            out.cfg.moe.intermediate_size = static_cast<int>(inter_e);
            L.experts_gate_up = concat_rows_same_dtype(gu_list);
            L.experts_gate_up.desc = Tensor::from_shape({E, 2 * inter_e, H}, DType::F32, Backend::CPU);
            L.experts_down = concat_rows_same_dtype(dn_list);
            L.experts_down.desc = Tensor::from_shape({E, H, inter_e}, DType::F32, Backend::CPU);
            if (const auto* shared_gate = sf.find_tensor(pfx + "mlp.shared_expert.gate_proj.weight")) {
                const auto width = shared_gate->shape[0];
                if (width <= 0 || width > INT_MAX / 2 ||
                    width != get_int("shared_expert_intermediate_size", static_cast<int>(width)))
                    throw std::runtime_error("Invalid Safetensors shared expert geometry: " + pfx);
                L.has_shared = true;
                out.cfg.moe.has_shared_expert = true;
                L.shared_gate_up = concat_rows_same_dtype({
                    req(pfx + "mlp.shared_expert.gate_proj.weight", {width, H}),
                    req(pfx + "mlp.shared_expert.up_proj.weight", {width, H})});
                L.shared_down = req(pfx + "mlp.shared_expert.down_proj.weight", {H, width});
                out.cfg.moe.shared_intermediate_size = static_cast<int>(L.shared_down.desc.shape[1]);
            } else {
                out.cfg.moe.has_shared_expert = false;
            }
        } else {
            L.kind = 0;
            L.attn_norm = to_f32_host(req(pfx + "input_layernorm.weight", {H}));
            L.q_proj = req(pfx + "self_attn.q_proj.weight", {nq * hd, H});
            L.k_proj = req(pfx + "self_attn.k_proj.weight", {nkv * hd, H});
            L.v_proj = req(pfx + "self_attn.v_proj.weight", {nkv * hd, H});
            const bool has_q_norm = sf.find_tensor(pfx + "self_attn.q_norm.weight") != nullptr;
            const bool has_k_norm = sf.find_tensor(pfx + "self_attn.k_norm.weight") != nullptr;
            if (has_q_norm != has_k_norm)
                throw std::runtime_error("Incomplete Safetensors Q/K normalization: " + pfx);
            if (has_q_norm) {
                out.cfg.dense.qk_norm = true;
                L.q_norm = to_f32_host(req(pfx + "self_attn.q_norm.weight", {hd}));
                L.k_norm = to_f32_host(req(pfx + "self_attn.k_norm.weight", {hd}));
            }
            L.o_proj = req(pfx + "self_attn.o_proj.weight", {H, nq * hd});
            L.mlp_norm = to_f32_host(req(pfx + "post_attention_layernorm.weight", {H}));
            L.gate_up = concat_rows_same_dtype({req(pfx + "mlp.gate_proj.weight", {inter, H}), req(pfx + "mlp.up_proj.weight", {inter, H})});
            L.down = req(pfx + "mlp.down_proj.weight", {H, inter});
        }
    }
    if (saw_gdn) out.cfg.arch = "hybrid";
    else if (saw_moe) out.cfg.arch = "moe";
    else out.cfg.arch = "dense";
    return true;
} catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
}

}  // namespace fusionrt::model_io
