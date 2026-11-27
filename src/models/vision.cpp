#include "fusionrt/vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef FUSIONRT_ENABLE_CUDA
#include "model_loader.h"
#include "fusionrt/cuda_kernels.h"
#endif

namespace fusionrt {
namespace {

const GgufTensorInfo* projection(const GgufFile& g) {
    for (const auto* name : {"mm.2.weight", "mm.input_projection.weight", "mm.0.weight", "mm.proj.weight"}) {
        if (const auto* t = g.find_tensor(name)) return t;
    }
    return nullptr;
}

bool qwen_projector(std::string_view type) { return type.starts_with("qwen"); }
bool gemma4_projector(std::string_view type) { return type == "gemma4" || type == "gemma4v"; }
bool gemma3_projection(const VisionInfo& g, const GgufTensorInfo& t) {
    return (g.projector == "gemma3" || g.projector == "gemma") &&
           t.name == "mm.input_projection.weight" && t.shape[0] == g.hidden_size;
}

}  // namespace

VisionInfo VisionInfo::from_gguf(const GgufFile& g) {
    VisionInfo v;
    v.architecture = g.get_string("general.architecture");
    v.projector = g.get_string("clip.vision.projector_type",
        g.get_string("clip.projector_type", v.architecture == "clip" ? "mlp" : v.architecture));
    if (v.architecture != "clip" && !qwen_projector(v.architecture) && !v.architecture.starts_with("gemma"))
        throw std::invalid_argument("Unsupported vision GGUF architecture: " + v.architecture);
    const auto* patch = g.find_tensor("v.patch_embd.weight");
    const auto* out = projection(g);
    if (!patch || patch->shape.size() < 2 || !out || out->shape.size() != 2)
        throw std::invalid_argument("Vision GGUF requires a patch embedding and a projector");
    const std::string prefix = "clip.vision.";
    auto get = [&](const char* key, int64_t def, int lo, int hi) {
        const auto val = g.get_int(prefix + key, g.get_int(v.architecture + "." + key, def));
        if (val < lo || val > hi) throw std::invalid_argument(std::string("Invalid vision geometry: ") + key);
        return static_cast<int>(val);
    };
    v.patch_size = get("patch_size", patch->shape.size() == 4 ? patch->shape[3] : 0, 1, 64);
    v.image_size = get("image_size", 448, v.patch_size, 8192);
    v.hidden_size = get("embedding_length", patch->shape[0], 1, 8192);
    v.heads = get("attention.head_count", 1, 1, 128);
    v.layers = get("block_count", 0, 0, 128);
    const auto default_merge = qwen_projector(v.projector) ? 2 : gemma4_projector(v.projector) ? 3 :
                               v.projector == "gemma3" || v.projector == "gemma" ? 4 : 1;
    v.spatial_merge = get("spatial_merge_size", g.get_int(prefix + "projector.scale_factor", default_merge), 1, 8);
    const auto output = out->shape[gemma3_projection(v, *out) ? 1 : 0];
    if (output <= 0 || output > 32768 || v.hidden_size % v.heads || v.hidden_size / v.heads > 1024)
        throw std::invalid_argument("Invalid vision head or projector dimensions");
    v.output_size = static_cast<int>(output);
    return v;
}

#ifdef FUSIONRT_ENABLE_CUDA
namespace {

struct Affine {
    Tensor weight, bias;
    std::array<float, 2> input_range{-INFINITY, INFINITY}, output_range{-INFINITY, INFINITY};
    bool clamp_input = false, clamp_output = false;
};

struct VisionBlock {
    Affine attn_norm, qkv, q, k, v, o, q_norm, k_norm, attn_post;
    Affine ffn_norm, up, gate, down, ffn_post;
};

struct ArenaMark {
    CudaArena& arena;
    size_t mark = arena.mark();
    ~ArenaMark() { arena.reset(mark); }
};

float lerp(float a, float b, float t) { return a + (b - a) * t; }

template<class Sample>
float bilinear(float x, float y, int w, int h, Sample sample) {
    x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    return lerp(lerp(sample(x0, y0), sample(x1, y0), x - x0),
                lerp(sample(x0, y1), sample(x1, y1), x - x0), y - y0);
}

void copy_rows(CudaContext& ctx, float* dst, const float* src, size_t size) {
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(dst, src, size * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
}

}  // namespace

struct VisionEncoder::Weights {
    VisionInfo info;
    CudaArena arena;
    Affine patch, temporal, pre_norm, post_norm, projector, projector_down, projector_norm;
    Tensor cls, v_norm, std_bias, std_scale, soft_norm;
    HostTensor position;
    std::vector<VisionBlock> layers;
    std::array<float, 3> mean{}, stddev{1,1,1};
    float eps = 1e-6f, theta = 0;
    int window = 0, window_pattern = 0;
    bool qwen = false, gemma = false, gemma4 = false, rms = false, gelu = true, quick = false;
};

VisionEncoder::VisionEncoder(CudaContext& ctx, const std::string& path, int text_hidden_size)
    : weights_(std::make_unique<Weights>()) {
    GgufFile file;
    std::string error;
    if (!file.load(path, &error, false)) throw std::invalid_argument("Cannot load mmproj: " + error);
    auto& w = *weights_;
    w.info = VisionInfo::from_gguf(file);
    const auto& g = w.info;
    w.qwen = qwen_projector(g.projector);
    w.gemma4 = gemma4_projector(g.projector);
    w.gemma = w.gemma4 || g.projector == "gemma" || g.projector == "gemma3";
    const bool qwen25 = g.projector.starts_with("qwen25") || g.projector.starts_with("qwen2.5");
    if (!w.qwen && !w.gemma && g.projector != "mlp" && g.projector != "clip")
        throw std::invalid_argument("Unsupported mmproj projector: " + g.projector);
    if (text_hidden_size != g.output_size) throw std::invalid_argument("mmproj output width does not match the text model");
    const int H = g.hidden_size, hd = H / g.heads, P = g.patch_size;
    w.eps = static_cast<float>(file.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6));
    w.theta = static_cast<float>(file.get_float("clip.vision.rope.freq_base", w.qwen ? 10000 : w.gemma4 ? 100 : 0));
    w.rms = qwen25 || w.gemma4 || file.get_bool("clip.vision.use_rms_norm", false);
    w.gelu = !file.get_bool("clip.use_silu", qwen25);
    w.quick = !w.qwen && !w.gemma && file.find_tensor("v.class_embd") && !file.get_bool("clip.use_gelu", false);
    w.window_pattern = static_cast<int>(file.get_int("clip.vision.n_wa_pattern", 0));
    w.window = static_cast<int>(file.get_int("clip.vision.window_size", 112)) / (P * g.spatial_merge) * g.spatial_merge;
    if (!(w.eps > 0) || !std::isfinite(w.eps) || w.theta < 0 || !std::isfinite(w.theta) ||
        (w.theta && hd % 4) || w.window_pattern < 0 || (w.window_pattern && w.window < 1)) {
        throw std::invalid_argument("Invalid vision normalization, RoPE or attention window");
    }
    auto channels = [&](const char* key, auto& target) {
        const auto values = file.get_float_array(key);
        if (values.empty()) return;
        if (values.size() != target.size() || std::any_of(values.begin(), values.end(), [](float x) { return !std::isfinite(x); }))
            throw std::invalid_argument("Invalid vision color normalization");
        std::copy(values.begin(), values.end(), target.begin());
    };
    channels("clip.vision.image_mean", w.mean);
    channels("clip.vision.image_std", w.stddev);
    if (std::any_of(w.stddev.begin(), w.stddev.end(), [](float x) { return !(x > 0); }))
        throw std::invalid_argument("Invalid vision image standard deviation");

    size_t bytes = 256 + (w.gemma4 ? (static_cast<size_t>(hd) * sizeof(float) + 255) & ~size_t{255} : 0);
    for (const auto& t : file.tensors()) {
        if ((!t.name.starts_with("v.") && !t.name.starts_with("mm.")) || t.name == "v.position_embd.weight") continue;
        if (t.name.find("deepstack") != std::string::npos)
            throw std::invalid_argument("Unsupported mmproj extension tensor: " + t.name);
        const size_t n = t.shape.size() == 1 ? static_cast<size_t>(t.shape[0]) * sizeof(float) : t.nbytes;
        if (n > std::numeric_limits<size_t>::max() - bytes - 255) throw std::invalid_argument("Vision weights are too large");
        bytes += (n + 255) & ~size_t{255};
    }
    w.arena.init(bytes);
    auto load = [&](const std::string& name, int64_t rows, int64_t cols, bool required) {
        if (!file.find_tensor(name)) {
            if (required) throw std::invalid_argument("Missing mmproj tensor: " + name);
            return Tensor{};
        }
        auto src = file.view_tensor(name);
        if ((cols == 0 && (src.desc.ndim != 1 || src.desc.shape[0] != rows)) ||
            (cols != 0 && (src.desc.ndim < 2 || src.desc.shape[0] != rows || src.desc.numel() / rows != cols ||
                           (src.desc.ndim != 2 && !name.starts_with("v.patch_embd.weight")))) ||
            src.desc.dtype == DType::I32) {
            throw std::invalid_argument("Invalid mmproj tensor shape or dtype: " + name);
        }
        if (cols == 0) src = model_io::to_f32_host(src);
        auto t = w.arena.alloc_tensor(cols ? std::vector<int64_t>{rows, cols} : std::vector<int64_t>{rows},
                                      src.desc.dtype, name);
        ctx.upload_into(src, t);
        return t;
    };
    auto affine = [&](const std::string& base, int64_t rows, int64_t cols, bool required = true) {
        Affine result;
        result.weight = load(base + ".weight", rows, cols, required);
        if (result.weight.data) result.bias = load(base + ".bias", rows, 0, false);
        if (result.weight.data && cols && w.gemma4) {
            auto range = [&](const std::string& tag, std::array<float, 2>& bounds) {
                bool have = false;
                for (int i = 0; i < 2; ++i) {
                    const auto key = base + tag + (i == 0 ? "_min" : "_max");
                    if (!file.find_tensor(key)) continue;
                    const auto value = model_io::to_f32_host(file.view_tensor(key));
                    if (value.desc.numel() != 1 || std::isnan(value.data<float>()[0]))
                        throw std::invalid_argument("Invalid mmproj clamp bound: " + key);
                    bounds[i] = value.data<float>()[0];
                    have = true;
                }
                if (bounds[0] > bounds[1]) throw std::invalid_argument("Invalid mmproj clamp bounds: " + base);
                return have;
            };
            result.clamp_input = range(".input", result.input_range);
            result.clamp_output = range(".output", result.output_range);
        }
        return result;
    };
    w.patch = affine("v.patch_embd", H, 3 * P * P);
    w.temporal.weight = load("v.patch_embd.weight.1", H, 3 * P * P, false);
    w.cls = load("v.class_embd", H, 0, false);
    if (w.cls.data && w.theta) throw std::invalid_argument("Vision class token with 2D RoPE is unsupported");
    w.pre_norm = affine("v.pre_ln", H, 0, false);
    w.post_norm = affine("v.post_ln", H, 0, false);
    if (file.find_tensor("v.position_embd.weight")) {
        w.position = model_io::to_f32_host(file.view_tensor("v.position_embd.weight"));
        const auto& d = w.position.desc;
        if (w.gemma4 ? d.ndim != 3 || d.shape[0] != 2 || d.shape[2] != H : d.ndim != 2 || d.shape[1] != H)
            throw std::invalid_argument("Invalid vision position embedding shape");
    }
    const auto features = file.get_int_array("clip.vision.feature_layer");
    if (features.size() > 1) throw std::invalid_argument("Multiple CLIP feature layers are unsupported");
    int layers = g.layers;
    if (!features.empty()) {
        layers = features.front() < 0 ? g.layers + features.front() + 1 : features.front();
        if (layers < 0 || layers > g.layers) throw std::invalid_argument("Invalid CLIP vision feature layer");
    } else if (w.cls.data && layers > 0) --layers;
    for (int i = 0; i < layers; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        VisionBlock layer;
        layer.attn_norm = affine(b + "ln1", H, 0);
        layer.qkv = affine(b + "attn_qkv", 3 * H, H, false);
        if (!layer.qkv.weight.data) {
            layer.q = affine(b + "attn_q", H, H);
            layer.k = affine(b + "attn_k", H, H);
            layer.v = affine(b + "attn_v", H, H);
        }
        layer.o = affine(b + "attn_out", H, H);
        layer.q_norm = affine(b + "attn_q_norm", hd, 0, false);
        layer.k_norm = affine(b + "attn_k_norm", hd, 0, false);
        layer.attn_post = affine(b + "attn_post_norm", H, 0, false);
        layer.ffn_norm = affine(b + "ln2", H, 0);
        const auto* ff = file.find_tensor(b + "ffn_up.weight");
        if (!ff || ff->shape.size() != 2 || ff->shape[0] > 32768) throw std::invalid_argument("Invalid mmproj FFN width");
        const auto I = ff->shape[0];
        layer.up = affine(b + "ffn_up", I, H);
        layer.gate = affine(b + "ffn_gate", I, H, false);
        layer.down = affine(b + "ffn_down", H, I);
        layer.ffn_post = affine(b + "ffn_post_norm", H, 0, false);
        w.layers.push_back(std::move(layer));
    }
    if (w.gemma4) {
        w.v_norm = w.arena.alloc_tensor({hd}, DType::F32);
        ctx.upload_into(model_io::make_ones_f32({hd}), w.v_norm);
        w.std_bias = load("v.std_bias", H, 0, false);
        w.std_scale = load("v.std_scale", H, 0, false);
        if (static_cast<bool>(w.std_bias.data) != static_cast<bool>(w.std_scale.data))
            throw std::invalid_argument("Expected both Gemma vision standardization vectors");
    }
    if (w.gemma && !w.gemma4) w.soft_norm = load("mm.soft_emb_norm.weight", H, 0, false);
    w.projector_norm = affine("mm.input_norm", H, 0, false);
    const auto* last = projection(file);
    const auto weight_suffix = std::string_view(".weight");
    const std::string name = last->name.substr(0, last->name.size() - weight_suffix.size());
    const int merged = H * (w.gemma ? 1 : g.spatial_merge * g.spatial_merge);
    if (gemma3_projection(g, *last)) {
        if (dtype_is_quantized(last->dtype) || last->dtype == DType::I32 || last->shape[1] != text_hidden_size)
            throw std::invalid_argument("Expected floating-point Gemma vision projection");
        const auto src = file.view_tensor(last->name);
        const size_t stride = dtype_type_size(last->dtype);
        auto transposed = HostTensor::create({text_hidden_size, H}, last->dtype);
        for (int i = 0; i < H; ++i) for (int j = 0; j < text_hidden_size; ++j) {
            const auto read = (static_cast<size_t>(i) * text_hidden_size + j) * stride;
            const auto write = (static_cast<size_t>(j) * H + i) * stride;
            std::memcpy(transposed.storage.data() + write, src.bytes().data() + read, stride);
        }
        w.projector.weight = w.arena.alloc_tensor({text_hidden_size, H}, last->dtype, last->name);
        ctx.upload_into(transposed, w.projector.weight);
    } else if (name == "mm.2") {
        const auto mid = last->shape[1];
        if (mid <= 0 || mid > 32768) throw std::invalid_argument("Invalid vision MLP width");
        w.projector = affine("mm.0", mid, merged);
        w.projector_down = affine(name, text_hidden_size, mid);
    } else {
        w.projector = affine(name, text_hidden_size, merged);
    }
}

VisionEncoder::~VisionEncoder() = default;
const VisionInfo& VisionEncoder::info() const { return weights_->info; }

VisionEmbedding VisionEncoder::encode(CudaContext& ctx, const Image& image, int max_tokens) const {
    const auto& w = *weights_;
    const auto& g = w.info;
    const int P = g.patch_size, H = g.hidden_size, merge = g.spatial_merge, hd = H / g.heads;
    if (image.width <= 0 || image.height <= 0 || image.width > 8192 || image.height > 8192 ||
        image.rgb.size() != static_cast<size_t>(image.width) * image.height * 3)
        throw std::invalid_argument("Invalid RGB image");
    const int limit = std::max(1, g.image_size / (P * merge));
    int gx = limit * merge, gy = gx;
    if (w.qwen || w.gemma4) {
        const int side = std::max(image.width, image.height);
        gx = std::max(1, (image.width * limit + side / 2) / side) * merge;
        gy = std::max(1, (image.height * limit + side / 2) / side) * merge;
    }
    const int patches = gx * gy, count = patches / (merge * merge);
    if (patches > 4096 || count >= max_tokens) throw std::invalid_argument("Image patches exceed the vision or text context");
    const int cls = w.cls.data ? 1 : 0, rows = patches + cls;
    auto& scratch = ctx.scratch_arena();
    ArenaMark restore{scratch};
    auto alloc = [&](int n, int d) { return scratch.alloc_tensor({n, d}, DType::F32); };
    auto linear = [&](const Tensor& in, const Affine& a, Tensor& out, int n) {
        ArenaMark temporary{scratch};
        const int cols = static_cast<int>(a.weight.shape[0]);
        if (a.clamp_input) {
            const int input = static_cast<int>(a.weight.shape[1]);
            auto clipped = alloc(n, input);
            cuda_ops::clamp_f32(in.ptr<float>(), clipped.ptr<float>(), n * input, a.input_range[0], a.input_range[1], ctx.stream());
            model_io::vision_linear_cuda(ctx, clipped, a.weight, out, n);
        } else model_io::vision_linear_cuda(ctx, in, a.weight, out, n);
        if (a.clamp_output)
            cuda_ops::clamp_f32(out.ptr<float>(), out.ptr<float>(), n * cols, a.output_range[0], a.output_range[1], ctx.stream());
        if (a.bias.data) cuda_ops::add_bias_f32(out.ptr<float>(), a.bias.ptr<float>(), n,
            cols, cols, ctx.stream());
    };
    auto norm = [&](const Tensor& in, const Affine& a, Tensor& out, int n, int d) {
        if (w.rms) {
            cuda_ops::rmsnorm_f32(in.ptr<float>(), a.weight.ptr<float>(), out.ptr<float>(), n, d, w.eps, ctx.stream());
            if (a.bias.data) cuda_ops::add_bias_f32(out.ptr<float>(), a.bias.ptr<float>(), n, d, d, ctx.stream());
        } else cuda_ops::layernorm_f32(in.ptr<float>(), a.weight.ptr<float>(), a.bias.data ? a.bias.ptr<float>() : nullptr,
                                       out.ptr<float>(), n, d, w.eps, ctx.stream());
    };
    Tensor hidden = alloc(rows, H);
    {
        ArenaMark temporary{scratch};
        HostTensor pixels({patches, 3 * P * P}, DType::F32);
        auto* dst = pixels.data<float>();
        for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) {
            for (int c = 0; c < 3; ++c) for (int py = 0; py < P; ++py) for (int px = 0; px < P; ++px) {
                const auto sample = bilinear((x * P + px + 0.5f) * image.width / (gx * P) - 0.5f,
                    (y * P + py + 0.5f) * image.height / (gy * P) - 0.5f, image.width, image.height,
                    [&](int ix, int iy) { return image.rgb[(static_cast<size_t>(iy) * image.width + ix) * 3 + c] / 255.f; });
                *dst++ = w.gemma4 ? sample * 2.0f - 1.0f : (sample - w.mean[c]) / w.stddev[c];
            }
        }
        const auto d_pixels = ctx.upload(pixels, false);
        auto patches_out = hidden.slice_dim0(cls, patches);
        linear(d_pixels, w.patch, patches_out, patches);
        if (w.temporal.weight.data) {
            auto other = alloc(patches, H);
            linear(d_pixels, w.temporal, other, patches);
            cuda_ops::add_residual_f32(other.ptr<float>(), patches_out.ptr<float>(), patches_out.ptr<float>(), patches * H, ctx.stream());
        }
        if (cls) copy_rows(ctx, hidden.ptr<float>(), w.cls.ptr<float>(), H);
    }
    if (!w.position.bytes().empty()) {
        ArenaMark temporary{scratch};
        HostTensor pos({rows, H}, DType::F32);
        auto* out = pos.data<float>();
        const auto* table = w.position.data<float>();
        if (w.gemma4) {
            const int side = static_cast<int>(w.position.desc.shape[1]);
            if (gx > side || gy > side) throw std::invalid_argument("Image exceeds vision position table");
            for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) for (int d = 0; d < H; ++d)
                *out++ = table[x * H + d] + table[(side + y) * H + d];
        } else {
            const int n = static_cast<int>(w.position.desc.shape[0]) - cls;
            const int side = static_cast<int>(std::sqrt(static_cast<float>(std::max(0, n))));
            if (side < 1 || side * side != n) throw std::invalid_argument("Vision position table must be square");
            if (cls) out = std::copy_n(table, H, out);
            for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) for (int d = 0; d < H; ++d) {
                *out++ = bilinear(static_cast<float>(x) * (side - 1) / std::max(1, gx - 1),
                    static_cast<float>(y) * (side - 1) / std::max(1, gy - 1), side, side,
                    [&](int ix, int iy) { return table[(cls + iy * side + ix) * H + d]; });
            }
        }
        auto dev = ctx.upload(pos, false);
        cuda_ops::add_residual_f32(dev.ptr<float>(), hidden.ptr<float>(), hidden.ptr<float>(), rows * H, ctx.stream());
    }
    if (w.pre_norm.weight.data) {
        auto tmp = alloc(rows, H);
        norm(hidden, w.pre_norm, tmp, rows, H);
        hidden = tmp;
    }
    for (size_t i = 0; i < w.layers.size(); ++i) {
        ArenaMark temporary{scratch};
        const auto& l = w.layers[i];
        auto x = alloc(rows, H), q = alloc(rows, H), k = alloc(rows, H), v = alloc(rows, H), a = alloc(rows, H);
        norm(hidden, l.attn_norm, x, rows, H);
        if (l.qkv.weight.data) {
            auto packed = alloc(rows, 3 * H);
            linear(x, l.qkv, packed, rows);
            const std::array<Tensor*, 3> projections{&q, &k, &v};
            for (int index = 0; index < 3; ++index) {
                FUSIONRT_CUDA_CHECK(cudaMemcpy2DAsync(projections[index]->ptr(), H * sizeof(float), packed.ptr<float>() + index * H,
                    3 * H * sizeof(float), H * sizeof(float), rows, cudaMemcpyDeviceToDevice, ctx.stream()));
            }
        } else {
            linear(x, l.q, q, rows); linear(x, l.k, k, rows); linear(x, l.v, v, rows);
        }
        if (l.q_norm.weight.data) { auto t = alloc(rows, H); norm(q, l.q_norm, t, rows * g.heads, hd); q = t; }
        if (l.k_norm.weight.data) { auto t = alloc(rows, H); norm(k, l.k_norm, t, rows * g.heads, hd); k = t; }
        if (w.v_norm.data) {
            auto t = alloc(rows, H);
            cuda_ops::rmsnorm_f32(v.ptr<float>(), w.v_norm.ptr<float>(), t.ptr<float>(), rows * g.heads, hd, w.eps, ctx.stream());
            v = t;
        }
        if (w.theta) {
            cuda_ops::rope_2d_f32(q.ptr<float>(), gy, gx, g.heads, hd, w.theta, ctx.stream(), w.qwen);
            cuda_ops::rope_2d_f32(k.ptr<float>(), gy, gx, g.heads, hd, w.theta, ctx.stream(), w.qwen);
        }
        const float scale = w.gemma4 ? 1.f : 1.f / std::sqrt(static_cast<float>(hd));
        auto attention = [&](const Tensor& tq, const Tensor& tk, const Tensor& tv, Tensor& to, int n) {
            cuda_ops::gqa_prefill_causal_f32(tq.ptr<float>(), tk.ptr<float>(), tv.ptr<float>(), to.ptr<float>(),
                n, g.heads, g.heads, hd, 0, n, scale, ctx.stream(), 0, DType::F32, {}, 0, false);
        };
        if (w.window_pattern && (i + 1) % w.window_pattern != 0) {
            if (cls) throw std::invalid_argument("Windowed vision attention does not support a class token");
            for (int y = 0; y < gy; y += w.window) for (int xx = 0; xx < gx; xx += w.window) {
                ArenaMark window{scratch};
                const int height = std::min(w.window, gy - y), width = std::min(w.window, gx - xx);
                const int n = height * width;
                auto tq = alloc(n, H), tk = alloc(n, H), tv = alloc(n, H), to = alloc(n, H);
                const int first = (y * gx + xx) * H;
                auto move = [&](float* dst, int stride, const float* src, int pitch) {
                    FUSIONRT_CUDA_CHECK(cudaMemcpy2DAsync(dst, stride * sizeof(float), src, pitch * sizeof(float),
                        width * H * sizeof(float), height, cudaMemcpyDeviceToDevice, ctx.stream()));
                };
                move(tq.ptr<float>(), width * H, q.ptr<float>() + first, gx * H);
                move(tk.ptr<float>(), width * H, k.ptr<float>() + first, gx * H);
                move(tv.ptr<float>(), width * H, v.ptr<float>() + first, gx * H);
                attention(tq, tk, tv, to, n);
                move(a.ptr<float>() + first, gx * H, to.ptr<float>(), width * H);
            }
        } else attention(q, k, v, a, rows);
        linear(a, l.o, x, rows);
        if (l.attn_post.weight.data) { norm(x, l.attn_post, a, rows, H); std::swap(x, a); }
        cuda_ops::add_residual_f32(x.ptr<float>(), hidden.ptr<float>(), hidden.ptr<float>(), rows * H, ctx.stream());
        norm(hidden, l.ffn_norm, x, rows, H);
        const int I = static_cast<int>(l.up.weight.shape[0]);
        auto up = alloc(rows, I);
        linear(x, l.up, up, rows);
        if (l.gate.weight.data) {
            auto gate = alloc(rows, I);
            linear(x, l.gate, gate, rows);
            (w.gelu ? cuda_ops::geglu_f32 : cuda_ops::swiglu_f32)(gate.ptr<float>(), up.ptr<float>(), up.ptr<float>(),
                                                                rows * I, ctx.stream());
        } else cuda_ops::vision_activation_f32(up.ptr<float>(), rows * I, w.gelu, w.quick, ctx.stream());
        linear(up, l.down, a, rows);
        if (l.ffn_post.weight.data) { norm(a, l.ffn_post, x, rows, H); std::swap(x, a); }
        cuda_ops::add_residual_f32(a.ptr<float>(), hidden.ptr<float>(), hidden.ptr<float>(), rows * H, ctx.stream());
    }
    Tensor features = hidden.slice_dim0(cls, patches);
    if (w.post_norm.weight.data) { auto t = alloc(patches, H); norm(features, w.post_norm, t, patches, H); features = t; }
    if (w.projector_norm.weight.data) { auto t = alloc(patches, H); norm(features, w.projector_norm, t, patches, H); features = t; }
    if (merge > 1) {
        auto merged = alloc(count, H * (w.gemma ? 1 : merge * merge));
        cuda_ops::vision_merge_f32(features.ptr<float>(), merged.ptr<float>(), gy, gx, H, merge, w.gemma, ctx.stream());
        features = merged;
    }
    if (w.gemma4) {
        cuda_ops::scale_f32(features.ptr<float>(), count * H, std::sqrt(static_cast<float>(H)), ctx.stream());
        if (w.std_bias.data)
            cuda_ops::vision_standardize_f32(features.ptr<float>(), w.std_bias.ptr<float>(), w.std_scale.ptr<float>(),
                                              count, H, ctx.stream());
        auto t = alloc(count, H);
        const auto ones = ctx.upload(model_io::make_ones_f32({H}), false);
        cuda_ops::rmsnorm_f32(features.ptr<float>(), ones.ptr<float>(), t.ptr<float>(), count, H, w.eps, ctx.stream());
        features = t;
    } else if (w.soft_norm.data) {
        auto t = alloc(count, H);
        cuda_ops::rmsnorm_f32(features.ptr<float>(), w.soft_norm.ptr<float>(), t.ptr<float>(), count, H, w.eps, ctx.stream());
        features = t;
    }
    VisionEmbedding output;
    output.storage = std::make_unique<CudaArena>();
    output.storage->init(static_cast<size_t>(count) * g.output_size * sizeof(float));
    output.rows = output.storage->alloc_tensor({count, g.output_size}, DType::F32);
    if (w.projector_down.weight.data) {
        const int mid = static_cast<int>(w.projector.weight.shape[0]);
        auto projected = alloc(count, mid);
        linear(features, w.projector, projected, count);
        cuda_ops::vision_activation_f32(projected.ptr<float>(), count * mid, true, false, ctx.stream());
        linear(projected, w.projector_down, output.rows, count);
    } else linear(features, w.projector, output.rows, count);
    ctx.synchronize();
    return output;
}

VisionPrompt encode_vision_prompt(CudaContext& ctx, const VisionEncoder& encoder, const BpeTokenizer& tokenizer,
                                   std::string_view text, std::span<const Image* const> images, int context) {
    std::vector<size_t> offsets;
    for (size_t pos = text.find(image_placeholder); pos != text.npos;
         pos = text.find(image_placeholder, pos + image_placeholder.size())) {
        offsets.push_back(pos);
    }
    return encode_vision_prompt(ctx, encoder, tokenizer, text, images, offsets, context);
}

VisionPrompt encode_vision_prompt(CudaContext& ctx, const VisionEncoder& encoder, const BpeTokenizer& tokenizer,
                                   std::string_view text, std::span<const Image* const> images,
                                   std::span<const size_t> offsets, int context) {
    if (offsets.size() != images.size()) throw std::invalid_argument("Image placeholder count does not match images");
    VisionPrompt prompt;
    const bool qwen = qwen_projector(encoder.info().projector);
    const bool gemma = encoder.info().projector.starts_with("gemma");
    const int32_t start = tokenizer.token_id(qwen ? "<|vision_start|>" : gemma ? "<start_of_image>" : "");
    const int32_t end = tokenizer.token_id(qwen ? "<|vision_end|>" : gemma ? "<end_of_image>" : "");
    const int32_t pad = std::max(0, tokenizer.token_id(qwen ? "<|image_pad|>" : "<pad>"));
    auto append = [&](std::string_view chunk, bool bos) {
        const auto ids = tokenizer.encode(std::string(chunk), bos);
        prompt.tokens.insert(prompt.tokens.end(), ids.begin(), ids.end());
        if (prompt.tokens.size() >= static_cast<size_t>(context)) throw prompt_context_error(prompt.tokens.size(), context);
    };
    size_t consumed = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        const auto place = offsets[i];
        if (place < consumed || place >= text.size() || !text.substr(place).starts_with(image_placeholder) || !images[i])
            throw std::invalid_argument("Invalid image placeholder position");
        append(text.substr(consumed, place - consumed), i == 0);
        if (start >= 0) prompt.tokens.push_back(start);
        prompt.images.push_back(encoder.encode(ctx, *images[i], context - static_cast<int>(prompt.tokens.size()) - (end >= 0)));
        const auto& rows = prompt.images.back().rows;
        prompt.inputs_embeds.push_back({static_cast<int>(prompt.tokens.size()), rows});
        prompt.tokens.insert(prompt.tokens.end(), static_cast<size_t>(rows.shape[0]), pad);
        if (end >= 0) prompt.tokens.push_back(end);
        consumed = place + image_placeholder.size();
    }
    append(text.substr(consumed), images.empty());
    return prompt;
}
#endif

}  // namespace fusionrt
