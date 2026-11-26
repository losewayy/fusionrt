#include "model_loader.h"
#include "fusionrt/gguf.h"
#include "fusionrt/json.h"
#include "fusionrt/safetensors.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace fusionrt::model_io {

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
        if (p.desc.dtype != dt || p.desc.ndim != 2 || p.desc.shape[1] != cols) {
            throw std::runtime_error("concat_rows_same_dtype: incompatible tensor shapes/dtypes");
        }
        total_rows += p.desc.shape[0];
        total_bytes += p.bytes().size();
    }
    HostTensor out = HostTensor::create({total_rows, cols}, dt);
    size_t off = 0;
    for (const auto& p : parts) {
        std::memcpy(out.storage.data() + off, p.bytes().data(), p.bytes().size());
        off += p.bytes().size();
    }
    return out;
}

HostTensor to_f32_host(const HostTensor& ht) {
    if (ht.desc.dtype == DType::F32) return ht;
    std::vector<int64_t> dims(ht.desc.shape.begin(), ht.desc.shape.begin() + ht.desc.ndim);
    HostTensor out = HostTensor::create(dims, DType::F32);
    int64_t n = ht.desc.numel();
    if (ht.desc.dtype == DType::F16) {
        convert_f16_to_f32(reinterpret_cast<const uint16_t*>(ht.bytes().data()), out.data<float>(), n);
    } else if (ht.desc.dtype == DType::BF16) {
        convert_bf16_to_f32(reinterpret_cast<const uint16_t*>(ht.bytes().data()), out.data<float>(), n);
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
    bool experts = false;
};

constexpr GgufArchitecture supported_architectures[] = {
    {"llama", false}, {"qwen2", true}, {"qwen3", true}, {"qwen35", true, true},
    {"qwen3moe", true, false, true}, {"deepseek2", false, false, true}, {"mistral", true}, {"phi3", true},
    {"gemma", true}, {"gemma4", true}, {"dense", true}, {"hybrid", true, true}, {"moe", true, false, true},
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
    const bool deepseek2 = arch == "deepseek2";
    const std::string prefix = arch + ".";
    auto get_arch_int = [&](const std::string& suffix, int64_t def) {
        return gf.get_int(prefix + suffix, def);
    };
    auto get_arch_float = [&](const std::string& suffix, double def) {
        return gf.get_float(prefix + suffix, def);
    };

    if (gemma4) {
        if (get_arch_int("expert_count", 0) > 0) {
            if (err) *err = "Gemma 4 expert layers are unsupported";
            return false;
        }
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

    out.tok_embeddings = gf.view_tensor("token_embd.weight");
    out.final_norm_w = to_f32_host(gf.view_tensor("output_norm.weight"));
    if (gf.find_tensor("output.weight")) {
        out.lm_head_w = gf.view_tensor("output.weight");
    }

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
    out.cfg.num_layers = num_layers;
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
    out.cfg.gdn.tiled_value_heads = arch == "qwen35";
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

    auto req = [&](const std::string& name) -> HostTensor {
        if (!gf.find_tensor(name)) {
            throw std::runtime_error("GGUF missing required tensor: " + name);
        }
        return gf.view_tensor(name);
    };

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
            const double score_scale = amplitude * (1 + log_mul * log_factor);
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
            if (has_gdn && !spec->recurrent) throw std::runtime_error("Unexpected GGUF recurrent attention: " + arch);
            if (has_moe && !spec->experts) throw std::runtime_error("Unexpected GGUF expert layer: " + arch);
            if (!has_attn && !has_gdn && !has_moe) throw std::runtime_error("GGUF layer has no supported attention: " + p);
            if (has_moe) {
                if (has_gdn) throw std::runtime_error("Combined recurrent attention and MoE is unsupported");
                if (gf.find_tensor(p + "ffn_gate_inp_shexp.weight")) {
                    throw std::runtime_error("GGUF shared expert requires a separate routing gate: " + p);
                }
                saw_moe = true;
                L.mlp_norm = to_f32_host(req(p + "ffn_norm.weight"));
                L.router_w = to_f32_host(req(p + "ffn_gate_inp.weight"));
                int E = static_cast<int>(L.router_w.desc.shape[0]);
                if (L.router_w.desc.ndim != 2 || L.router_w.desc.shape[1] != H || E < out.cfg.moe.top_k ||
                    out.cfg.moe.top_k <= 0 || E > 256) {
                    throw std::runtime_error("Unsupported GGUF expert router shape: " + p);
                }
                if (deepseek2) {
                    if (E != mla_experts || L.mlp_norm.desc.ndim != 1 || L.mlp_norm.desc.shape[0] != H) {
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
                    if (deepseek2) {
                        const int64_t width = int64_t{I} * shared_experts;
                        if (width <= 0 || width > std::numeric_limits<int>::max() / 2 ||
                            !matrix(gate, width, H) || !matrix(up, width, H) || !matrix(L.shared_down, H, width)) {
                            throw std::runtime_error("Invalid DeepSeek MoE shared expert shapes: " + p);
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
                if (!has_attn) {
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
                L.conv_w = to_f32_host(gf.find_tensor(p + "ssm_conv1d.weight") ? gf.view_tensor(p + "ssm_conv1d.weight") : req(p + "gdn_conv.weight"));
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
                    L.qkv_proj = gf.view_tensor(p + "attn_qkv.weight");
                    // Split q, k, v host slices for unfused/Vulkan paths
                    if (L.qkv_proj.desc.ndim != 2 || L.qkv_proj.desc.shape[1] != H ||
                        L.qkv_proj.desc.shape[0] != q_dim + 2 * kv_dim) {
                        throw std::runtime_error("Invalid GGUF packed QKV shape: " + p + "attn_qkv.weight");
                    }
                    size_t row_b = row_nbytes(L.qkv_proj.desc.dtype, H);
                    L.q_proj = HostTensor::create({q_dim, H}, L.qkv_proj.desc.dtype);
                    std::memcpy(L.q_proj.storage.data(), L.qkv_proj.bytes().data(), static_cast<size_t>(q_dim) * row_b);
                    if (owns_kv) {
                        L.k_proj = HostTensor::create({kv_dim, H}, L.qkv_proj.desc.dtype);
                        L.v_proj = HostTensor::create({kv_dim, H}, L.qkv_proj.desc.dtype);
                        std::memcpy(L.k_proj.storage.data(), L.qkv_proj.bytes().data() + static_cast<size_t>(q_dim) * row_b, static_cast<size_t>(kv_dim) * row_b);
                        std::memcpy(L.v_proj.storage.data(), L.qkv_proj.bytes().data() + static_cast<size_t>(q_dim + kv_dim) * row_b, static_cast<size_t>(kv_dim) * row_b);
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
            if (has_moe) continue;
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
                    if (gate.desc.dtype == up.desc.dtype) {
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
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }

    if (saw_gdn) out.cfg.arch = "hybrid";
    else if (saw_moe) out.cfg.arch = "moe";
    else out.cfg.arch = "dense";
    return true;
}

bool parse_safetensors_model_raw(const std::string& path, int max_seq_override, LoadedModelRaw& out, std::string* err) {
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
        return cobj.count(k) ? static_cast<int>(cobj.at(k).as_int(def)) : def;
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
    if (H <= 0 || num_layers <= 0 || nq <= 0 || hd <= 0 || inter <= 0 || vocab <= 0) {
        if (err) *err = "config.json missing required model dimensions (hidden_size/num_hidden_layers/num_attention_heads/intermediate_size/vocab_size)";
        return false;
    }

    if (!sf.find_tensor("model.embed_tokens.weight") || !sf.find_tensor("model.norm.weight")) {
        if (err) *err = "Safetensors missing model.embed_tokens.weight or model.norm.weight";
        return false;
    }
    out.tok_embeddings = sf.read_tensor("model.embed_tokens.weight");
    out.final_norm_w = to_f32_host(sf.read_tensor("model.norm.weight"));
    if (sf.find_tensor("lm_head.weight")) {
        out.lm_head_w = sf.read_tensor("lm_head.weight");
    } else if (get_bool("tie_word_embeddings", true)) {
        out.lm_head_w = out.tok_embeddings;
    } else {
        if (err) *err = "Safetensors missing lm_head.weight (and tie_word_embeddings=false)";
        return false;
    }

    int ctx_len = get_int("max_position_embeddings", 4096);
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

    auto req = [&](const std::string& name) -> HostTensor {
        if (!sf.find_tensor(name)) {
            throw std::runtime_error("Safetensors missing required tensor: " + name);
        }
        return sf.read_tensor(name);
    };

    bool saw_gdn = false, saw_moe = false;
    out.layers.resize(static_cast<size_t>(num_layers));
    try {
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
                L.gate_up = concat_rows_same_dtype({req(pfx + "mlp.gate_proj.weight"), req(pfx + "mlp.up_proj.weight")});
                L.down = req(pfx + "mlp.down_proj.weight");
            } else if (sf.find_tensor(pfx + "mlp.gate.weight")) {
                L.kind = 2;
                saw_moe = true;
                L.attn_norm = to_f32_host(req(pfx + "input_layernorm.weight"));
                L.router_w = to_f32_host(req(pfx + "mlp.gate.weight"));
                int E = static_cast<int>(L.router_w.desc.shape[0]);
                out.cfg.moe.num_experts = E;
                out.cfg.moe.gpu_resident_experts = E;
                std::vector<HostTensor> gu_list, dn_list;
                for (int e = 0; e < E; ++e) {
                    std::string ep = pfx + "mlp.experts." + std::to_string(e) + ".";
                    gu_list.push_back(to_f32_host(req(ep + "gate_proj.weight")));
                    gu_list.push_back(to_f32_host(req(ep + "up_proj.weight")));
                    dn_list.push_back(to_f32_host(req(ep + "down_proj.weight")));
                }
                int64_t inter_e = gu_list[0].desc.shape[0];
                out.cfg.moe.intermediate_size = static_cast<int>(inter_e);
                L.experts_gate_up = concat_rows_same_dtype(gu_list);
                L.experts_gate_up.desc = Tensor::from_shape({E, 2 * inter_e, H}, DType::F32, Backend::CPU);
                L.experts_down = concat_rows_same_dtype(dn_list);
                L.experts_down.desc = Tensor::from_shape({E, H, inter_e}, DType::F32, Backend::CPU);
                if (sf.find_tensor(pfx + "mlp.shared_expert.gate_proj.weight")) {
                    L.has_shared = true;
                    out.cfg.moe.has_shared_expert = true;
                    L.shared_gate_up = concat_rows_same_dtype({
                        req(pfx + "mlp.shared_expert.gate_proj.weight"),
                        req(pfx + "mlp.shared_expert.up_proj.weight")});
                    L.shared_down = req(pfx + "mlp.shared_expert.down_proj.weight");
                    out.cfg.moe.shared_intermediate_size = static_cast<int>(L.shared_down.desc.shape[1]);
                } else {
                    out.cfg.moe.has_shared_expert = false;
                }
            } else {
                L.kind = 0;
                L.attn_norm = to_f32_host(req(pfx + "input_layernorm.weight"));
                L.q_proj = req(pfx + "self_attn.q_proj.weight");
                L.k_proj = req(pfx + "self_attn.k_proj.weight");
                L.v_proj = req(pfx + "self_attn.v_proj.weight");
                if (sf.find_tensor(pfx + "self_attn.q_norm.weight") && sf.find_tensor(pfx + "self_attn.k_norm.weight")) {
                    out.cfg.dense.qk_norm = true;
                    L.q_norm = to_f32_host(sf.read_tensor(pfx + "self_attn.q_norm.weight"));
                    L.k_norm = to_f32_host(sf.read_tensor(pfx + "self_attn.k_norm.weight"));
                }
                L.o_proj = req(pfx + "self_attn.o_proj.weight");
                L.mlp_norm = to_f32_host(req(pfx + "post_attention_layernorm.weight"));
                L.gate_up = concat_rows_same_dtype({req(pfx + "mlp.gate_proj.weight"), req(pfx + "mlp.up_proj.weight")});
                L.down = req(pfx + "mlp.down_proj.weight");
            }
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }

    if (saw_gdn) out.cfg.arch = "hybrid";
    else if (saw_moe) out.cfg.arch = "moe";
    else out.cfg.arch = "dense";
    return true;
}

}  // namespace fusionrt::model_io
