#include "model_loader.h"
#include "fusionrt/gguf.h"
#include "fusionrt/json.h"
#include "fusionrt/safetensors.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
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
        total_bytes += p.storage.size();
    }
    HostTensor out = HostTensor::create({total_rows, cols}, dt);
    size_t off = 0;
    for (const auto& p : parts) {
        std::memcpy(out.storage.data() + off, p.storage.data(), p.storage.size());
        off += p.storage.size();
    }
    return out;
}

HostTensor to_f32_host(const HostTensor& ht) {
    if (ht.desc.dtype == DType::F32) return ht;
    std::vector<int64_t> dims(ht.desc.shape.begin(), ht.desc.shape.begin() + ht.desc.ndim);
    HostTensor out = HostTensor::create(dims, DType::F32);
    int64_t n = ht.desc.numel();
    if (ht.desc.dtype == DType::F16) {
        convert_f16_to_f32(reinterpret_cast<const uint16_t*>(ht.storage.data()), out.data<float>(), n);
    } else if (ht.desc.dtype == DType::BF16) {
        convert_bf16_to_f32(reinterpret_cast<const uint16_t*>(ht.storage.data()), out.data<float>(), n);
    } else {
        int64_t cols = dims.back();
        int64_t rows = n / cols;
        int64_t nb = cols / 32;
        if (ht.desc.dtype == DType::Q8_0) {
            const auto* src = reinterpret_cast<const BlockQ8_0*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q8_0(src + r * nb, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_0) {
            const auto* src = reinterpret_cast<const BlockQ4_0*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_0(src + r * nb, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_1) {
            const auto* src = reinterpret_cast<const BlockQ4_1*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_1(src + r * nb, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_K) {
            int64_t nbk = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ4_K*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_k(src + r * nbk, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q5_K) {
            int64_t nbk = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ5_K*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q5_k(src + r * nbk, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q6_K) {
            int64_t nbk = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ6_K*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q6_k(src + r * nbk, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q4_A) {
            int64_t nsa = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ4_A*>(ht.storage.data());
            for (int64_t r = 0; r < rows; ++r) dequantize_row_q4_a(src + r * nsa, out.data<float>() + r * cols, cols);
        } else if (ht.desc.dtype == DType::Q8_A) {
            int64_t nsa = cols / 256;
            const auto* src = reinterpret_cast<const BlockQ8_A*>(ht.storage.data());
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
    {"qwen3moe", true, false, true}, {"mistral", true}, {"phi3", true},
    {"gemma", true}, {"dense", true}, {"hybrid", true, true}, {"moe", true, false, true},
};

}  // namespace

bool parse_gguf_model_raw(const std::string& path, int max_seq_override, LoadedModelRaw& out, std::string* err) {
    GgufFile gf;
    if (!gf.load(path, err)) return false;

    const std::string arch = gf.get_string("general.architecture", "llama");
    const auto spec = std::find_if(std::begin(supported_architectures), std::end(supported_architectures),
        [&](const GgufArchitecture& entry) { return entry.name == arch; });
    if (spec == std::end(supported_architectures)) {
        if (err) *err = "Unsupported GGUF architecture: " + arch;
        return false;
    }
    out.cfg.model_architecture = arch;
    const std::string prefix = arch + ".";
    auto get_arch_int = [&](const std::string& suffix, int64_t def) {
        return gf.get_int(prefix + suffix, def);
    };
    auto get_arch_float = [&](const std::string& suffix, double def) {
        return gf.get_float(prefix + suffix, def);
    };

    if (!gf.find_tensor("token_embd.weight")) {
        if (err) *err = "GGUF missing required tensor: token_embd.weight";
        return false;
    }
    if (!gf.find_tensor("output_norm.weight")) {
        if (err) *err = "GGUF missing required tensor: output_norm.weight";
        return false;
    }

    out.tok_embeddings = gf.read_tensor("token_embd.weight");
    out.final_norm_w = to_f32_host(gf.read_tensor("output_norm.weight"));
    if (gf.find_tensor("output.weight")) {
        out.lm_head_w = gf.read_tensor("output.weight");
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
    int inter = static_cast<int>(get_arch_int("feed_forward_length", H * 3));
    int ctx_len = static_cast<int>(get_arch_int("context_length", 4096));
    if (max_seq_override > 0) ctx_len = max_seq_override;
    else ctx_len = std::min(ctx_len, 4096);

    float eps = static_cast<float>(get_arch_float("attention.layer_norm_rms_epsilon", 1e-6));
    float rope_theta = static_cast<float>(get_arch_float("rope.freq_base", 10000.0));
    int rot_dim = static_cast<int>(get_arch_int("rope.dimension_count", hd));
    float partial_rot = (hd > 0) ? (static_cast<float>(rot_dim) / static_cast<float>(hd)) : 1.0f;

    if (get_arch_int("attention.sliding_window", 0) > 0) {
        if (err) *err = "GGUF architecture " + arch + " requires sliding-window attention";
        return false;
    }
    if (get_arch_int("attention.value_length", hd) != hd) {
        if (err) *err = "Distinct GGUF attention key and value lengths are unsupported";
        return false;
    }
    const auto scaling = gf.get_string(prefix + "rope.scaling.type", "none");
    if (scaling != "none" && !gf.find_tensor("rope_freqs.weight")) {
        if (err) *err = "Unsupported GGUF RoPE scaling: " + scaling;
        return false;
    }
    if (out.tok_embeddings.desc.ndim != 2 || H != emb_hidden || nq <= 0 || nkv <= 0 ||
        nq % nkv != 0 || hd <= 0 || inter <= 0 || ctx_len <= 0 || rot_dim < 2 ||
        rot_dim > hd || (rot_dim % 2) != 0 || !std::isfinite(rope_theta) || rope_theta <= 0) {
        if (err) *err = "Invalid GGUF attention or embedding dimensions for " + arch;
        return false;
    }

    out.cfg.vocab_size = static_cast<int>(emb_vocab);
    out.cfg.num_layers = num_layers;
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
    out.cfg.dense.weight_dtype = out.lm_head_w.storage.empty() ? out.tok_embeddings.desc.dtype : out.lm_head_w.desc.dtype;

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
    out.cfg.moe.gpu_resident_experts = out.cfg.moe.num_experts;
    out.cfg.moe.rms_norm_eps = eps;

    auto matrix = [](const HostTensor& tensor, int64_t rows, int64_t cols) {
        return tensor.desc.ndim == 2 && tensor.desc.shape[0] == rows && tensor.desc.shape[1] == cols;
    };
    if ((!out.lm_head_w.storage.empty() && !matrix(out.lm_head_w, emb_vocab, H)) ||
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
        return gf.read_tensor(name);
    };

    try {
        if (gf.find_tensor("rope_freqs.weight")) {
            out.rope_factors = to_f32_host(req("rope_freqs.weight"));
            if (out.rope_factors.desc.ndim != 1 || out.rope_factors.desc.shape[0] != rot_dim / 2 ||
                std::any_of(out.rope_factors.data<float>(), out.rope_factors.data<float>() + rot_dim / 2,
                            [](float factor) { return !(factor > 0) || !std::isfinite(factor); })) {
                throw std::runtime_error("Invalid GGUF RoPE frequency factors");
            }
        }
        for (int i = 0; i < num_layers; ++i) {
            std::string p = "blk." + std::to_string(i) + ".";
            LoadedLayerRaw& L = out.layers[static_cast<size_t>(i)];

            const bool has_attn = gf.find_tensor(p + "attn_q.weight") || gf.find_tensor(p + "attn_qkv.weight");
            const bool has_gdn = gf.find_tensor(p + "ssm_conv1d.weight") || gf.find_tensor(p + "gdn_conv.weight");
            if (gf.find_tensor(p + "attn_kv_a_mqa.weight")) {
                throw std::runtime_error("GGUF architecture " + arch + " requires multi-head latent attention");
            }
            if (arch.starts_with("gemma") && (gf.find_tensor(p + "post_ffw_norm.weight") ||
                                               gf.find_tensor(p + "post_attention_norm.weight"))) {
                throw std::runtime_error("GGUF architecture " + arch + " requires Gemma attention and FFN blocks");
            }
            const bool has_moe = gf.find_tensor(p + "ffn_gate_inp.weight") != nullptr;
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
                out.cfg.moe.num_experts = E;
                out.cfg.moe.gpu_resident_experts = E;
                if (gf.find_tensor(p + "ffn_gate_up_exps.weight")) {
                    L.experts_gate_up = to_f32_host(req(p + "ffn_gate_up_exps.weight"));
                    if (L.experts_gate_up.desc.ndim != 3 || (L.experts_gate_up.desc.shape[1] % 2) != 0) {
                        throw std::runtime_error("Invalid GGUF packed expert projections: " + p);
                    }
                    out.cfg.moe.intermediate_size = static_cast<int>(L.experts_gate_up.desc.shape[1] / 2);
                    L.experts_down = to_f32_host(req(p + "ffn_down_exps.weight"));
                } else if (gf.find_tensor(p + "ffn_gate_exps.weight") && gf.find_tensor(p + "ffn_up_exps.weight")) {
                    HostTensor ge = to_f32_host(gf.read_tensor(p + "ffn_gate_exps.weight"));
                    HostTensor ue = to_f32_host(gf.read_tensor(p + "ffn_up_exps.weight"));
                    int64_t inter_e = ge.desc.shape[1];
                    if (ge.desc.ndim != 3 || ue.desc.ndim != 3 || ge.desc.shape != ue.desc.shape ||
                        ge.desc.shape[0] != E || ge.desc.shape[2] != H) {
                        throw std::runtime_error("Invalid GGUF expert projections: " + p);
                    }
                    out.cfg.moe.intermediate_size = static_cast<int>(inter_e);
                    L.experts_gate_up = HostTensor::create({E, 2 * inter_e, H}, DType::F32);
                    for (int e = 0; e < E; ++e) {
                        std::memcpy(L.experts_gate_up.data<float>() + static_cast<size_t>(e) * (2 * inter_e) * H,
                                    ge.data<float>() + static_cast<size_t>(e) * inter_e * H,
                                    static_cast<size_t>(inter_e * H) * sizeof(float));
                        std::memcpy(L.experts_gate_up.data<float>() + (static_cast<size_t>(e) * 2 + 1) * inter_e * H,
                                    ue.data<float>() + static_cast<size_t>(e) * inter_e * H,
                                    static_cast<size_t>(inter_e * H) * sizeof(float));
                    }
                    L.experts_down = to_f32_host(req(p + "ffn_down_exps.weight"));
                } else {
                    std::vector<HostTensor> gu_list, dn_list;
                    for (int e = 0; e < E; ++e) {
                        std::string es = std::to_string(e);
                        gu_list.push_back(to_f32_host(req(p + "ffn_gate." + es + ".weight")));
                        gu_list.push_back(to_f32_host(req(p + "ffn_up." + es + ".weight")));
                        dn_list.push_back(to_f32_host(req(p + "ffn_down." + es + ".weight")));
                    }
                    L.experts_gate_up = concat_rows_same_dtype(gu_list);
                    L.experts_gate_up.desc = Tensor::from_shape({E, 2 * gu_list[0].desc.shape[0], H}, DType::F32, Backend::CPU);
                    L.experts_down = concat_rows_same_dtype(dn_list);
                    L.experts_down.desc = Tensor::from_shape({E, H, dn_list[0].desc.shape[1]}, DType::F32, Backend::CPU);
                    out.cfg.moe.intermediate_size = static_cast<int>(gu_list[0].desc.shape[0]);
                }
                const int I = out.cfg.moe.intermediate_size;
                if (L.experts_gate_up.desc.ndim != 3 || L.experts_gate_up.desc.shape[0] != E ||
                    L.experts_gate_up.desc.shape[1] != 2 * I || L.experts_gate_up.desc.shape[2] != H ||
                    L.experts_down.desc.ndim != 3 || L.experts_down.desc.shape[0] != E ||
                    L.experts_down.desc.shape[1] != H || L.experts_down.desc.shape[2] != I) {
                    throw std::runtime_error("Invalid GGUF expert FFN shapes: " + p);
                }
                if (gf.find_tensor(p + "ffn_gate_shexp.weight") && gf.find_tensor(p + "ffn_up_shexp.weight") && gf.find_tensor(p + "ffn_down_shexp.weight")) {
                    L.has_shared = true;
                    out.cfg.moe.has_shared_expert = true;
                    L.shared_gate_up = concat_rows_same_dtype({
                        gf.read_tensor(p + "ffn_gate_shexp.weight"),
                        gf.read_tensor(p + "ffn_up_shexp.weight")});
                    L.shared_down = gf.read_tensor(p + "ffn_down_shexp.weight");
                    out.cfg.moe.shared_intermediate_size = static_cast<int>(L.shared_down.desc.shape[1]);
                } else {
                    out.cfg.moe.has_shared_expert = false;
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
                L.conv_w = to_f32_host(gf.find_tensor(p + "ssm_conv1d.weight") ? gf.read_tensor(p + "ssm_conv1d.weight") : req(p + "gdn_conv.weight"));
                if (gf.find_tensor(p + "ssm_conv1d.bias")) {
                    L.conv_bias = to_f32_host(gf.read_tensor(p + "ssm_conv1d.bias"));
                } else if (gf.find_tensor(p + "gdn_conv.bias")) {
                    L.conv_bias = to_f32_host(gf.read_tensor(p + "gdn_conv.bias"));
                } else {
                    L.conv_bias = make_zeros_f32({L.conv_w.desc.shape[0]});
                }
                if (gf.find_tensor(p + "ssm_a")) {
                    L.A_log = to_f32_host(gf.read_tensor(p + "ssm_a"));
                    float* ap = L.A_log.data<float>();
                    for (int64_t k = 0; k < L.A_log.desc.numel(); ++k) {
                        if (ap[k] < 0.0f) ap[k] = std::log(std::max(-ap[k], 1e-30f));
                    }
                } else {
                    L.A_log = to_f32_host(req(p + "gdn_a_log"));
                }
                L.dt_bias = to_f32_host(gf.find_tensor(p + "ssm_dt.bias") ? gf.read_tensor(p + "ssm_dt.bias") : req(p + "gdn_dt_bias"));
                L.gdn_norm_w = to_f32_host(gf.find_tensor(p + "ssm_norm.weight") ? gf.read_tensor(p + "ssm_norm.weight") : req(p + "gdn_norm.weight"));
                L.o_proj = gf.find_tensor(p + "ssm_out.weight") ? req(p + "ssm_out.weight") : req(p + "attn_output.weight");
            } else {
                L.kind = 0;
                L.attn_norm = to_f32_host(req(p + "attn_norm.weight"));
                if (gf.find_tensor(p + "attn_qkv.weight")) {
                    L.qkv_proj = gf.read_tensor(p + "attn_qkv.weight");
                    // Split q, k, v host slices for unfused/Vulkan paths
                    int q_dim = nq * hd;
                    int kv_dim = nkv * hd;
                    if (L.qkv_proj.desc.ndim != 2 || L.qkv_proj.desc.shape[1] != H ||
                        L.qkv_proj.desc.shape[0] != q_dim + 2 * kv_dim) {
                        throw std::runtime_error("Invalid GGUF packed QKV shape: " + p + "attn_qkv.weight");
                    }
                    size_t row_b = row_nbytes(L.qkv_proj.desc.dtype, H);
                    L.q_proj = HostTensor::create({q_dim, H}, L.qkv_proj.desc.dtype);
                    L.k_proj = HostTensor::create({kv_dim, H}, L.qkv_proj.desc.dtype);
                    L.v_proj = HostTensor::create({kv_dim, H}, L.qkv_proj.desc.dtype);
                    std::memcpy(L.q_proj.storage.data(), L.qkv_proj.storage.data(), static_cast<size_t>(q_dim) * row_b);
                    std::memcpy(L.k_proj.storage.data(), L.qkv_proj.storage.data() + static_cast<size_t>(q_dim) * row_b, static_cast<size_t>(kv_dim) * row_b);
                    std::memcpy(L.v_proj.storage.data(), L.qkv_proj.storage.data() + static_cast<size_t>(q_dim + kv_dim) * row_b, static_cast<size_t>(kv_dim) * row_b);
                } else {
                    L.q_proj = req(p + "attn_q.weight");
                    L.k_proj = req(p + "attn_k.weight");
                    L.v_proj = req(p + "attn_v.weight");
                    L.gated_q_attn = (L.q_proj.desc.shape[0] == static_cast<int64_t>(2 * nq * hd));
                    if (L.gated_q_attn) out.cfg.dense.gated_q_attn = true;
                    const int q_dim = (L.gated_q_attn ? 2 : 1) * nq * hd;
                    const int kv_dim = nkv * hd;
                    if (!matrix(L.q_proj, q_dim, H) || !matrix(L.k_proj, kv_dim, H) || !matrix(L.v_proj, kv_dim, H)) {
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
                L.k_bias = bias("attn_k.bias", L.k_proj.desc.shape[0]);
                L.v_bias = bias("attn_v.bias", L.v_proj.desc.shape[0]);
                const bool has_q_norm = gf.find_tensor(p + "attn_q_norm.weight") != nullptr;
                const bool has_k_norm = gf.find_tensor(p + "attn_k_norm.weight") != nullptr;
                if (has_q_norm != has_k_norm) throw std::runtime_error("Incomplete GGUF Q/K normalization: " + p);
                if (has_q_norm) {
                    out.cfg.dense.qk_norm = true;
                    L.q_norm = to_f32_host(gf.read_tensor(p + "attn_q_norm.weight"));
                    L.k_norm = to_f32_host(gf.read_tensor(p + "attn_k_norm.weight"));
                }
                L.o_proj = req(p + "attn_output.weight");
                if (!matrix(L.o_proj, H, nq * hd)) throw std::runtime_error("Invalid GGUF attention output shape: " + p);
            }
            if (has_moe) continue;
            L.mlp_norm = to_f32_host(gf.find_tensor(p + "post_attention_norm.weight") ? gf.read_tensor(p + "post_attention_norm.weight") : req(p + "ffn_norm.weight"));
            if (gf.find_tensor(p + "ffn_gate_up.weight")) {
                L.gate_up = req(p + "ffn_gate_up.weight");
            } else {
                auto up = req(p + "ffn_up.weight");
                if (!gf.find_tensor(p + "ffn_gate.weight") && up.desc.shape[0] == 2 * inter) {
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
            const bool valid_gate_up = L.gate_up.storage.empty()
                ? matrix(L.gate, inter, H) && matrix(L.up, inter, H)
                : matrix(L.gate_up, 2 * inter, H);
            if (!valid_gate_up || !matrix(L.down, H, inter)) {
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
