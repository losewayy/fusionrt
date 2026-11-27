#include "fusionrt/vision.h"
#include "fusionrt/vulkan_kernels.h"
#include "model_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>

namespace fusionrt {
namespace {
const GgufTensorInfo* projection(const GgufFile& g) {
    for (const auto* name : {"mm.2.weight", "mm.input_projection.weight", "mm.0.weight", "mm.proj.weight"})
        if (const auto* t = g.find_tensor(name)) return t;
    return nullptr;
}

bool gemma4_projector(std::string_view type) { return type == "gemma4" || type == "gemma4v"; }
struct Affine {
    Tensor weight, bias;
    std::array<float, 2> input_range{-INFINITY, INFINITY}, output_range{-INFINITY, INFINITY};
};
struct VisionBlock {
    Affine attn_norm, qkv, q, k, v, o, q_norm, k_norm, attn_post;
    Affine ffn_norm, up, gate, down, ffn_post;
};

template<class Sample>
float bilinear(float x, float y, int w, int h, Sample sample) {
    x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    return std::lerp(std::lerp(sample(x0, y0), sample(x1, y0), x - x0),
                     std::lerp(sample(x0, y1), sample(x1, y1), x - x0), y - y0);
}
constexpr int gemma4_max_image_tokens = 1120;

// Resize on whole merge cells. Qwen3-VL uses nearest-even rounding for the initial grid.
std::pair<int, int> image_grid(const Image& image, const VisionInfo& info, bool gemma4,
                               int64_t min_pixels, int64_t max_pixels) {
    const int merge = info.spatial_merge, cell = info.patch_size * merge;
    if (!gemma4 && !max_pixels) {
        const int limit = std::max(1, info.image_size / cell), side = std::max(image.width, image.height);
        return {std::max(1, (image.width * limit + side / 2) / side) * merge,
                std::max(1, (image.height * limit + side / 2) / side) * merge};
    }
    auto round = [&](int axis) {
        const int lower = axis / cell, twice = 2 * (axis % cell);
        return std::max(1, lower + (twice > cell || (twice == cell && (gemma4 || lower % 2))));
    };
    int nx = round(image.width), ny = round(image.height);
    const int64_t cell_area = int64_t(cell) * cell;
    const int64_t aligned = int64_t(nx) * ny * cell_area;
    if (gemma4) {
        min_pixels = 70 * cell_area;
        max_pixels = gemma4_max_image_tokens * cell_area;
    }
    if (aligned < min_pixels || aligned > max_pixels) {
        const bool shrink = aligned > max_pixels;
        const auto target = shrink ? max_pixels : min_pixels;
        const double w = double(image.width) / cell, h = double(image.height) / cell;
        const double scale = gemma4 ? std::sqrt(double(target / cell_area) / (w * h)) :
                                     std::sqrt(double(target) / (int64_t(image.width) * image.height));
        auto adjust = [&](int side) {
            const double count = gemma4 ? (double(side) / cell) * scale : side * scale / cell;
            return std::max(1, static_cast<int>(shrink ? std::floor(count) : std::ceil(count)));
        };
        nx = adjust(image.width);
        ny = adjust(image.height);
    }
    return {nx * merge, ny * merge};
}

// Pillow bicubic coefficients, with antialiasing when downsampling and RGB8 rounding per pass.
Image resize_cubic_axis(const Image& src, int out, bool horizontal) {
    const int width = horizontal ? out : src.width, height = horizontal ? src.height : out;
    Image dst{width, height, std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
    const int in = horizontal ? src.width : src.height;
    const int lines = horizontal ? src.height : src.width;
    const size_t src_step = horizontal ? 3 : static_cast<size_t>(src.width) * 3;
    const size_t dst_step = horizontal ? 3 : static_cast<size_t>(dst.width) * 3;
    const size_t src_line = horizontal ? static_cast<size_t>(src.width) * 3 : 3;
    const size_t dst_line = horizontal ? static_cast<size_t>(dst.width) * 3 : 3;
    constexpr int precision = 22;
    const double scale = static_cast<double>(in) / out;
    const double support = std::max(1.0, scale);
    std::vector<double> weights(static_cast<size_t>(std::ceil(4 * support)) + 1);
    std::vector<int> fixed(weights.size());
    for (int i = 0; i < out; ++i) {
        const double center = (i + 0.5) * scale;
        const int begin = std::max(0, static_cast<int>(center - 2 * support + 0.5));
        const int end = std::min(in, static_cast<int>(center + 2 * support + 0.5));
        double sum = 0;
        for (int j = begin; j < end; ++j) {
            const double x = std::abs(j + 0.5 - center) / support;
            const double weight = x < 1 ? ((1.5 * x - 2.5) * x * x + 1) :
                                  x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0;
            weights[j - begin] = weight;
            sum += weight;
        }
        for (int j = 0; j < end - begin; ++j)
            fixed[j] = static_cast<int>(std::lround(weights[j] / sum * (1 << precision)));
        for (int line = 0; line < lines; ++line) {
            const auto* pixel = src.rgb.data() + line * src_line + begin * src_step;
            std::array<int, 3> rgb{1 << (precision - 1), 1 << (precision - 1), 1 << (precision - 1)};
            for (int j = 0; j < end - begin; ++j)
                for (int c = 0; c < 3; ++c) rgb[c] += pixel[j * src_step + c] * fixed[j];
            auto* target = dst.rgb.data() + line * dst_line + i * dst_step;
            for (int c = 0; c < 3; ++c) target[c] = static_cast<uint8_t>(std::clamp(rgb[c] >> precision, 0, 255));
        }
    }
    return dst;
}

Image resize_bicubic(const Image& src, int width, int height) {
    if (src.width == width) return resize_cubic_axis(src, height, false);
    auto horizontal = resize_cubic_axis(src, width, true);
    return src.height == height ? std::move(horizontal) : resize_cubic_axis(horizontal, height, false);
}

struct ArenaMark {
    VkArena& arena;
    VkDeviceSize mark = arena.mark();
    ~ArenaMark() { arena.reset(mark); }
};
struct Scratch {
    VkContext& ctx;
    ~Scratch() { ctx.scratch_arena().release(ctx.device()); }
};
struct PushVision {
    int mode, M, C, width, height, heads, merge, flags;
    float eps, theta;
};
static_assert(sizeof(PushVision) == 40);
uint32_t groups(int64_t elements, int lanes = 64) { return static_cast<uint32_t>((elements + lanes - 1) / lanes); }
bool clipped(const std::array<float, 2>& range) { return range[0] != -INFINITY || range[1] != INFINITY; }

struct Batch {
    VkContext& ctx;
    std::vector<VkBatchDispatch> ops;
    std::vector<std::vector<uint8_t>> pushes;
    template<class Push>
    void add(const char* shader, uint32_t gx, uint32_t gy, std::initializer_list<Tensor> bindings, const Push& pc) {
        auto& bytes = pushes.emplace_back(sizeof(Push));
        std::memcpy(bytes.data(), &pc, sizeof(Push));
        ops.push_back({shader, gx, gy, 1, bindings, bytes.data(), sizeof(Push), {}});
    }
    // Copy rows with independent source and destination pitches (in floats).
    void copy(const Tensor& src, const Tensor& dst, int rows, int width,
              int src_pitch, int dst_pitch, int src_start = 0, int dst_start = 0) {
        std::vector<VkBufferCopy> ranges;
        ranges.reserve(rows);
        for (int i = 0; i < rows; ++i)
            ranges.push_back({(src_start + VkDeviceSize(i) * src_pitch) * sizeof(float),
                              (dst_start + VkDeviceSize(i) * dst_pitch) * sizeof(float),
                              VkDeviceSize(width) * sizeof(float)});
        ops.push_back({"copy", 0, 0, 0, {src, dst}, nullptr, 0, std::move(ranges)});
    }
    void flush() {
        if (!ops.empty()) ctx.dispatch_batched(ops, false);
        ops.clear();
        pushes.clear();
    }
    int row_batch(int rows, int64_t width, int max_rows) const {
        if (rows <= max_rows) return rows;
        const auto align = static_cast<int>(ctx.info().min_storage_offset_align);
        const auto step = align / std::gcd<int64_t>(align, width * sizeof(float));
        const auto block = static_cast<int>(max_rows / step * step);
        if (block == 0) throw std::invalid_argument("Vision tensor exceeds Vulkan dispatch limits");
        return block;
    }
    template<class Fn>
    void row_dispatch(int rows, int cols, bool norm, const Tensor& x, const Tensor& y, Fn dispatch) {
        const int block = row_batch(rows, cols, norm ? 65535 : 65535 * 64 / cols);
        for (int first = 0; first < rows; first += block) {
            const int n = std::min(block, rows - first);
            dispatch(first, n, norm ? n : groups(int64_t(n) * cols), x.slice_dim0(first, n), y.slice_dim0(first, n));
        }
    }
    void vision(PushVision pc, const Tensor& x, const Tensor& y, const Tensor& w = {}, const Tensor& b = {}) {
        row_dispatch(pc.M, pc.C, pc.mode == 0, x, y, [&](int first, int n, uint32_t gx, const auto& in, const auto& out) {
            pc.M = n;
            const auto aux = b.vk_buffer ? (pc.mode == 3 ? b.slice_dim0(first, n) : b) : in;
            add("epilogue/vision_ops.spv", gx, 1, {in, w.vk_buffer ? w : in, aux, out}, pc);
        });
    }
    void gemma(vk_ops::PushGemma pc, const Tensor& x, const Tensor& y, const Tensor& b = {}) {
        row_dispatch(pc.M, pc.N, pc.mode == 1, x, y, [&](int first, int n, uint32_t gx, const auto& in, const auto& out) {
            pc.M = n;
            const auto aux = b.vk_buffer ? (pc.mode == 1 ? b : b.slice_dim0(first, n)) : in;
            add("epilogue/gemma4_ops.spv", gx, 1, {in, aux, in, out, in, in, in}, pc);
        });
    }
};
}  // namespace

struct VulkanVisionEncoder::Weights {
    explicit Weights(uint32_t index) : device(index, 0, 0) {}
    VkContext device;
    std::mutex mutex;
    VisionInfo info;
    Affine patch, temporal, pre_norm, post_norm, projector, projector_down, projector_norm;
    HostTensor position, std_bias, std_scale;
    std::vector<VisionBlock> layers;
    std::array<float, 3> mean{}, stddev{1, 1, 1};
    float eps = 1e-6f, theta = 0;
    int window = 0, window_pattern = 0;
    int64_t min_pixels = 0, max_pixels = 0;
    bool gemma4 = false, rms = false, gelu = true;
};
VulkanVisionEncoder::VulkanVisionEncoder(VkContext& ctx, const std::string& path, int text_hidden_size)
    : weights_(std::make_unique<Weights>(ctx.info().device_index)) {
    GgufFile file;
    std::string error;
    if (!file.load(path, &error, false)) throw std::invalid_argument("Cannot load mmproj: " + error);
    auto& w = *weights_;
    w.info = VisionInfo::from_gguf(file);
    const auto& g = w.info;
    if (g.projector == "qwen3vl_merger") {
        const int align = g.patch_size * g.spatial_merge;
        const int64_t unit = static_cast<int64_t>(align) * align;
        w.min_pixels = file.get_int("clip.vision.image_min_pixels", 8 * unit);
        w.max_pixels = std::min(file.get_int("clip.vision.image_max_pixels", 4096 * unit),
                                4096 * unit / (g.spatial_merge * g.spatial_merge));
        if (w.min_pixels < unit || w.max_pixels < w.min_pixels)
            throw std::invalid_argument("Invalid Qwen3-VL image pixel budget");
    }
    w.gemma4 = gemma4_projector(g.projector);
    const bool qwen25 = g.projector.starts_with("qwen25") || g.projector.starts_with("qwen2.5");
    if (!w.gemma4 && !g.projector.starts_with("qwen"))
        throw std::invalid_argument("Unsupported mmproj projector: " + g.projector);
    if (text_hidden_size != g.output_size) throw std::invalid_argument("mmproj output width does not match the text model");
    const int H = g.hidden_size, hd = H / g.heads, P = g.patch_size;
    w.eps = static_cast<float>(file.get_float("clip.vision.attention.layer_norm_epsilon", 1e-6));
    w.theta = static_cast<float>(file.get_float("clip.vision.rope.freq_base", w.gemma4 ? 100 : 10000));
    w.rms = qwen25 || w.gemma4 || file.get_bool("clip.vision.use_rms_norm", false);
    w.gelu = w.gemma4 || !file.get_bool("clip.use_silu", qwen25);
    w.window_pattern = static_cast<int>(file.get_int("clip.vision.n_wa_pattern", 0));
    w.window = static_cast<int>(file.get_int("clip.vision.window_size", 112)) / (P * g.spatial_merge) * g.spatial_merge;
    if (!(w.eps > 0) || !std::isfinite(w.eps) || w.theta < 0 || !std::isfinite(w.theta) ||
        hd > 512 || (w.theta && hd % 4) || w.window_pattern < 0 || (w.window_pattern && w.window < 1))
        throw std::invalid_argument("Invalid vision normalization, RoPE or attention window");
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
    auto& device = w.device;
    device.set_device_profile(ctx.device_profile());
    const auto align = std::max<VkDeviceSize>(256, device.info().min_storage_offset_align);
    VkDeviceSize bytes = 0;
    for (const auto& t : file.tensors()) {
        if ((!t.name.starts_with("v.") && !t.name.starts_with("mm.")) || t.name.starts_with("mm.a.") ||
            t.name == "v.position_embd.weight") continue;
        if (t.name.find("deepstack") != std::string::npos)
            throw std::invalid_argument("Unsupported mmproj extension tensor: " + t.name);
        const auto n = t.shape.size() == 1 ? VkDeviceSize(t.shape[0]) * sizeof(float) : t.nbytes;
        if (n > std::numeric_limits<VkDeviceSize>::max() - bytes - align)
            throw std::invalid_argument("Vision weights are too large");
        bytes += (n + align - 1) & ~(align - 1);
    }
    device.reserve_model_memory(bytes, 0);
    auto load = [&](const std::string& name, int64_t rows, int64_t cols, bool required) {
        if (!file.find_tensor(name)) {
            if (required) throw std::invalid_argument("Missing mmproj tensor: " + name);
            return Tensor{};
        }
        auto src = file.view_tensor(name);
        const auto& d = src.desc;
        const bool valid_rank = d.ndim == (cols ? 2 : 1) || (cols && name.starts_with("v.patch_embd.weight"));
        if (!valid_rank || d.shape[0] != rows || d.numel() != rows * std::max<int64_t>(cols, 1) || d.dtype == DType::I32)
            throw std::invalid_argument("Invalid mmproj tensor shape or dtype: " + name);
        if (cols) src.desc = src.desc.view({rows, cols});
        else src = model_io::to_f32_host(src);
        return device.upload(src);
    };
    auto affine = [&](const std::string& base, int64_t rows, int64_t cols, bool required = true) {
        Affine result;
        result.weight = load(base + ".weight", rows, cols, required);
        if (result.weight.vk_buffer) result.bias = load(base + ".bias", rows, 0, false);
        if (result.weight.vk_buffer && cols && w.gemma4) {
            auto range = [&](const std::string& tag, std::array<float, 2>& bounds) {
                for (int i = 0; i < 2; ++i) {
                    const auto key = base + tag + (i == 0 ? "_min" : "_max");
                    if (!file.find_tensor(key)) continue;
                    const auto value = model_io::to_f32_host(file.view_tensor(key));
                    if (value.desc.numel() != 1 || std::isnan(value.data<float>()[0]))
                        throw std::invalid_argument("Invalid mmproj clamp bound: " + key);
                    bounds[i] = value.data<float>()[0];
                }
                if (bounds[0] > bounds[1]) throw std::invalid_argument("Invalid mmproj clamp bounds: " + base);
            };
            range(".input", result.input_range);
            range(".output", result.output_range);
        }
        return result;
    };
    w.patch = affine("v.patch_embd", H, 3 * P * P);
    w.temporal.weight = load("v.patch_embd.weight.1", H, 3 * P * P, false);
    w.pre_norm = affine("v.pre_ln", H, 0, false);
    w.post_norm = affine("v.post_ln", H, 0, false);
    if (file.find_tensor("v.position_embd.weight")) {
        w.position = model_io::to_f32_host(file.view_tensor("v.position_embd.weight"));
        const auto& d = w.position.desc;
        if (w.gemma4 ? d.ndim != 3 || d.shape[0] != 2 || d.shape[2] != H : d.ndim != 2 || d.shape[1] != H)
            throw std::invalid_argument("Invalid vision position embedding shape");
    }
    const auto features = file.get_int_array("clip.vision.feature_layer");
    int layers = features.empty() ? g.layers : features.front();
    if (layers < 0) layers += g.layers + 1;
    if (features.size() > 1 || layers < 0 || layers > g.layers)
        throw std::invalid_argument("Invalid vision feature layer");
    for (int i = 0; i < layers; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        VisionBlock layer;
        layer.attn_norm = affine(b + "ln1", H, 0);
        layer.qkv = affine(b + "attn_qkv", 3 * H, H, false);
        if (!layer.qkv.weight.vk_buffer) {
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
        if (!ff || ff->shape.size() != 2 || ff->shape[0] <= 0 || ff->shape[0] > 32768) throw std::invalid_argument("Invalid mmproj FFN width");
        const auto I = ff->shape[0];
        layer.up = affine(b + "ffn_up", I, H);
        layer.gate = affine(b + "ffn_gate", I, H, false);
        layer.down = affine(b + "ffn_down", H, I);
        layer.ffn_post = affine(b + "ffn_post_norm", H, 0, false);
        w.layers.push_back(std::move(layer));
    }
    if (w.gemma4 && (file.find_tensor("v.std_bias") || file.find_tensor("v.std_scale"))) {
        w.std_bias = model_io::to_f32_host(file.view_tensor("v.std_bias"));
        w.std_scale = model_io::to_f32_host(file.view_tensor("v.std_scale"));
        for (const auto* t : {&w.std_bias, &w.std_scale})
            if (t->desc.ndim != 1 || t->desc.shape[0] != H)
                throw std::invalid_argument("Invalid Gemma standardization vector");
    }
    w.projector_norm = affine("mm.input_norm", H, 0, false);
    const auto* last = projection(file);
    const auto name = last->name.substr(0, last->name.size() - std::string_view(".weight").size());
    const int merged = H * (w.gemma4 ? 1 : g.spatial_merge * g.spatial_merge);
    const bool mlp = name == "mm.2";
    const auto mid = mlp ? last->shape[1] : text_hidden_size;
    if (mid <= 0 || mid > 32768) throw std::invalid_argument("Invalid vision projector width");
    w.projector = affine(mlp ? "mm.0" : name, mid, merged);
    if (mlp) w.projector_down = affine(name, text_hidden_size, mid);
}

VulkanVisionEncoder::~VulkanVisionEncoder() = default;
const VisionInfo& VulkanVisionEncoder::info() const { return weights_->info; }

HostTensor VulkanVisionEncoder::encode(VkContext& model_ctx, const Image& image, int max_tokens) const {
    auto& w = *weights_;
    auto& ctx = w.device;
    if (model_ctx.info().device_index != ctx.info().device_index)
        throw std::invalid_argument("Vision encoder and text model use different Vulkan devices");
    const std::lock_guard lock(w.mutex);
    const auto& g = w.info;
    const int P = g.patch_size, H = g.hidden_size, merge = g.spatial_merge, hd = H / g.heads;
    if (image.width <= 0 || image.height <= 0 || image.width > 8192 || image.height > 8192 ||
        image.rgb.size() != static_cast<size_t>(image.width) * image.height * 3)
        throw std::invalid_argument("Invalid RGB image");
    const auto [gx, gy] = image_grid(image, g, w.gemma4, w.min_pixels, w.max_pixels);
    const auto patch_count = static_cast<int64_t>(gx) * gy;
    const int patch_limit = w.gemma4 ? gemma4_max_image_tokens * merge * merge : 4096;
    if (patch_count > std::min(65535, patch_limit) || patch_count / (merge * merge) >= max_tokens)
        throw std::invalid_argument("Image patches exceed the vision or text context");
    const int rows = static_cast<int>(patch_count), count = rows / (merge * merge);
    const int pixel_width = 3 * P * P, merged_width = H * (w.gemma4 ? 1 : merge * merge);
    const auto size = [](int m, int n) { return VkDeviceSize(m) * n; };
    const auto patch_size = size(rows, pixel_width) + bool(w.temporal.weight.vk_buffer) * size(rows, H);
    VkDeviceSize workspace = std::max(patch_size + std::max(size(rows, pixel_width), size(rows, H)), size(rows, H));
    const int ff_batch = std::min(rows, 1024);
    for (const auto& layer : w.layers) {
        const auto packed = layer.qkv.weight.vk_buffer ? (clipped(layer.qkv.output_range) ? 6 : 3) * size(rows, H) : 0;
        const auto window = w.window_pattern ? 4 * size(std::min(gx, w.window), std::min(gy, w.window)) * H : 0;
        const auto attention = 5 * size(rows, H) + std::max({packed, window, w.gemma4 ? size(rows, H) : 0});
        const int I = static_cast<int>(layer.up.weight.shape[0]);
        const int buffers = layer.gate.weight.vk_buffer && !w.gelu ? 4 : 3;
        const auto mlp = size(ff_batch, H + buffers * std::max(H, I));
        workspace = std::max({workspace, attention, mlp});
    }
    const int projected_width = static_cast<int>(w.projector.weight.shape[0]);
    const auto projection_size = size(count, merged_width + projected_width + g.output_size + (w.std_bias.bytes().empty() ? 0 : H) +
                                              std::max({merged_width, projected_width, g.output_size}));
    const auto align = std::max<VkDeviceSize>(256, ctx.info().min_storage_offset_align);
    if (!ctx.try_reserve_scratch_memory((size(rows, H) + std::max(workspace, projection_size)) * sizeof(float) + 16 * align))
        throw std::runtime_error("Vulkan vision scratch exceeds the device budget");
    Scratch release{ctx};
    auto& scratch = ctx.scratch_arena();
    auto alloc = [&](int m, int n) { return scratch.alloc_tensor({m, n}, DType::F32); };
    Batch batch{ctx, {}, {}};
    auto bias = [&](const Tensor& x, const Tensor& b, const Tensor& pos = {}) {
        const int flags = (b.vk_buffer ? 1 : 0) | (pos.vk_buffer ? 2 : 0);
        if (flags) batch.vision({.mode = 3, .M = int(x.shape[0]), .C = int(x.shape[1]), .flags = flags}, x, x, b, pos);
    };
    auto combine = [&](const Tensor& a, const Tensor& b, const Tensor& out, float sa, float sb) {
        batch.gemma({.mode = 3, .M = int(out.shape[0]), .N = int(out.shape[1]), .scale_a = sa, .scale_b = sb}, a, out, b);
    };
    // Gemma clamping and optional per-channel standardization have no mode in the vision kernels.
    auto transform = [&](const Tensor& x, auto fn) {
        batch.flush();
        auto host = ctx.download(x);
        auto* ptr = host.data<float>();
        for (int64_t i = 0; i < host.desc.numel(); ++i) ptr[i] = fn(ptr[i], i);
        return ctx.upload(host, false);
    };
    auto clamp = [&](const Tensor& x, const std::array<float, 2>& range) {
        return transform(x, [&](float v, int64_t) { return std::fmin(range[1], std::fmax(range[0], v)); });
    };
    auto linear = [&](const Tensor& in, const Affine& a, Tensor& out) {
        const int m = int(in.shape[0]), n = int(a.weight.shape[0]), k = int(a.weight.shape[1]);
        ArenaMark mark{scratch};
        const auto input = clipped(a.input_range) ? clamp(in, a.input_range) : in;
        vk_ops::append_linear_dispatch(batch.ops, input, a.weight, out, m, n, k, batch.pushes, &ctx);
        scratch.reset(mark.mark);
        if (clipped(a.output_range)) batch.copy(clamp(out, a.output_range), out, 1, m * n, 0, 0);
        bias(out, a.bias);
    };
    auto rms = [&](const Tensor& in, const Tensor& weight, const Tensor& out) {
        batch.gemma({.mode = 1, .M = int(in.shape[0]), .N = int(in.shape[1]),
                     .flags = weight.vk_buffer ? 0 : 2, .eps = w.eps}, in, out, weight);
    };
    auto norm = [&](const Tensor& in, const Affine& a, const Tensor& out) {
        if (w.rms) {
            rms(in, a.weight, out);
            bias(out, a.bias);
        } else batch.vision({.mode = 0, .M = int(in.shape[0]), .C = int(in.shape[1]),
                             .flags = a.bias.vk_buffer ? 1 : 0, .eps = w.eps}, in, out, a.weight, a.bias);
    };
    auto activation = [&](const Tensor& x, int kind) {
        batch.vision({.mode = 1, .M = int(x.shape[0]), .C = int(x.shape[1]), .flags = kind}, x, x);
    };
    Tensor hidden = alloc(rows, H);
    {
        ArenaMark mark{scratch};
        HostTensor pixels({rows, 3 * P * P}, DType::F32);
        auto* dst = pixels.data<float>();
        const bool cubic = w.gemma4 || w.max_pixels > 0;
        Image resized;
        if (cubic && (image.width != gx * P || image.height != gy * P))
            resized = resize_bicubic(image, gx * P, gy * P);
        const auto& source = resized.rgb.empty() ? image : resized;
        for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) {
            for (int c = 0; c < 3; ++c) for (int py = 0; py < P; ++py) for (int px = 0; px < P; ++px) {
                const int ix = x * P + px, iy = y * P + py;
                auto pixel = [&](int sx, int sy) { return source.rgb[(size_t(sy) * source.width + sx) * 3 + c] / 255.f; };
                const auto sample = cubic ? pixel(ix, iy) :
                    bilinear((ix + 0.5f) * image.width / (gx * P) - 0.5f,
                             (iy + 0.5f) * image.height / (gy * P) - 0.5f, image.width, image.height, pixel);
                *dst++ = w.gemma4 ? sample * 2.f - 1.f : (sample - w.mean[c]) / w.stddev[c];
            }
        }
        const auto d_pixels = ctx.upload(pixels, false);
        linear(d_pixels, w.patch, hidden);
        if (w.temporal.weight.vk_buffer) {
            auto other = alloc(rows, H);
            linear(d_pixels, w.temporal, other);
            combine(other, hidden, hidden, 1, 1);
        }
        batch.flush();
    }
    if (!w.position.bytes().empty()) {
        ArenaMark mark{scratch};
        HostTensor pos({rows, H}, DType::F32);
        auto* out = pos.data<float>();
        const auto* table = std::as_const(w.position).data<float>();
        if (w.gemma4) {
            const int side = static_cast<int>(w.position.desc.shape[1]);
            if (gx > side || gy > side) throw std::invalid_argument("Image exceeds vision position table");
            for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) for (int d = 0; d < H; ++d)
                *out++ = table[x * H + d] + table[(side + y) * H + d];
        } else {
            const int n = static_cast<int>(w.position.desc.shape[0]);
            const int side = static_cast<int>(std::sqrt(static_cast<float>(std::max(0, n))));
            if (side < 1 || side * side != n) throw std::invalid_argument("Vision position table must be square");
            for (int y = 0; y < gy; ++y) for (int x = 0; x < gx; ++x) for (int d = 0; d < H; ++d) {
                *out++ = bilinear(static_cast<float>(x) * (side - 1) / std::max(1, gx - 1),
                    static_cast<float>(y) * (side - 1) / std::max(1, gy - 1), side, side,
                    [&](int ix, int iy) { return table[(iy * side + ix) * H + d]; });
            }
        }
        bias(hidden, {}, ctx.upload(pos, false));
    }
    if (w.pre_norm.weight.vk_buffer) norm(hidden, w.pre_norm, hidden);
    for (size_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        {
            ArenaMark mark{scratch};
            auto x = alloc(rows, H), q = alloc(rows, H), k = alloc(rows, H), v = alloc(rows, H), a = alloc(rows, H);
            norm(hidden, l.attn_norm, x);
            if (l.qkv.weight.vk_buffer) {
                ArenaMark packed_scope{scratch};
                auto packed = alloc(rows, 3 * H);
                linear(x, l.qkv, packed);
                batch.copy(packed, q, rows, H, 3 * H, H);
                batch.copy(packed, k, rows, H, 3 * H, H, H);
                batch.copy(packed, v, rows, H, 3 * H, H, 2 * H);
            } else {
                linear(x, l.q, q);
                linear(x, l.k, k);
                linear(x, l.v, v);
            }
            const auto q_heads = q.view({rows * g.heads, hd}), k_heads = k.view({rows * g.heads, hd});
            const auto v_heads = v.view({rows * g.heads, hd});
            if (l.q_norm.weight.vk_buffer) norm(q_heads, l.q_norm, q_heads);
            if (l.k_norm.weight.vk_buffer) norm(k_heads, l.k_norm, k_heads);
            if (w.gemma4) rms(v_heads, {}, v_heads);
            if (w.theta) {
                const PushVision rope{2, rows, H, gx, gy, g.heads, 0, w.gemma4 ? 0 : 1, 0, w.theta};
                for (const auto* t : {&q, &k})
                    batch.add("epilogue/vision_ops.spv", g.heads, rows, {*t, *t, *t, *t}, rope);
            }
            const float scale = w.gemma4 ? 1.f : 1.f / std::sqrt(static_cast<float>(hd));
            auto attention = [&](const Tensor& tq, const Tensor& tk, const Tensor& tv, const Tensor& out, int n) {
                batch.add("sequence/flash_attention_prefill.spv", g.heads, n, {tq, tk, tv, out},
                          vk_ops::PushFaPrefill{n, g.heads, g.heads, hd, 0, n, int(DType::F32), 0, 0, scale});
            };
            if (w.window_pattern && (i + 1) % w.window_pattern) {
                for (int y = 0; y < gy; y += w.window) for (int col = 0; col < gx; col += w.window) {
                    ArenaMark window{scratch};
                    const int height = std::min(w.window, gy - y), width = std::min(w.window, gx - col);
                    const int n = height * width, first = (y * gx + col) * H;
                    auto tq = alloc(n, H), tk = alloc(n, H), tv = alloc(n, H), out = alloc(n, H);
                    batch.copy(q, tq, height, width * H, gx * H, width * H, first);
                    batch.copy(k, tk, height, width * H, gx * H, width * H, first);
                    batch.copy(v, tv, height, width * H, gx * H, width * H, first);
                    attention(tq, tk, tv, out, n);
                    batch.copy(out, a, height, width * H, width * H, gx * H, 0, first);
                }
            } else attention(q, k, v, a, rows);
            linear(a, l.o, x);
            if (l.attn_post.weight.vk_buffer) norm(x, l.attn_post, x);
            combine(x, hidden, hidden, 1, 1);
        }
        const int I = int(l.up.weight.shape[0]);
        const int tile = batch.row_batch(rows, H, w.gelu ? ff_batch : std::min(ff_batch, 65535 * 256 / I));
        for (int first = 0; first < rows; first += tile) {
            ArenaMark mark{scratch};
            const int n = std::min(tile, rows - first);
            const auto h = hidden.slice_dim0(first, n);
            auto x = alloc(n, H), up = alloc(n, I);
            norm(h, l.ffn_norm, x);
            linear(x, l.up, up);
            if (l.gate.weight.vk_buffer) {
                auto gate = alloc(n, I);
                linear(x, l.gate, gate);
                if (w.gelu) batch.gemma({.mode = 0, .M = n, .N = I}, gate, up, up);
                else {
                    auto packed = alloc(n, 2 * I);
                    batch.copy(gate, packed, n, I, I, 2 * I);
                    batch.copy(up, packed, n, I, I, 2 * I, 0, I);
                    batch.add("swiglu.comp", groups(int64_t(n) * I, 256), 1, {packed, up}, vk_ops::PushSwiglu{n, I});
                }
            } else activation(up, w.gelu ? 0 : 3);
            linear(up, l.down, x);
            if (l.ffn_post.weight.vk_buffer) norm(x, l.ffn_post, x);
            combine(x, h, h, 1, 1);
            batch.flush();
        }
    }
    Tensor features = hidden;
    for (const auto* a : {&w.post_norm, &w.projector_norm})
        if (a->weight.vk_buffer) norm(features, *a, features);
    if (merge > 1) {
        auto merged = alloc(count, merged_width);
        const int cols = gx / merge, out_rows = gy / merge;
        const int64_t row_width = int64_t(cols) * merged_width;
        const int stripe = batch.row_batch(out_rows, row_width, static_cast<int>(65535 * 64 / row_width));
        for (int y = 0; y < out_rows; y += stripe) {
            const int n = std::min(stripe, out_rows - y), height = n * merge;
            const auto in = features.slice_dim0(y * merge * gx, height * gx);
            const auto out = merged.slice_dim0(y * cols, n * cols);
            batch.add("epilogue/vision_ops.spv", groups(out.numel()), 1, {in, in, in, out},
                      PushVision{4, height * gx, H, gx, height, g.heads, merge, w.gemma4 ? 1 : 0, 0, 0});
        }
        features = merged;
    }
    if (w.gemma4) {
        combine(features, {}, features, std::sqrt(static_cast<float>(H)), 0);
        if (!w.std_bias.bytes().empty()) {
            const auto& weights = std::as_const(w);
            const auto* offset = weights.std_bias.data<float>(), *scale = weights.std_scale.data<float>();
            features = transform(features, [&](float x, int64_t i) { return (x - offset[i % H]) * scale[i % H]; });
        }
        rms(features, {}, features);
    }
    auto output = alloc(count, g.output_size);
    if (w.projector_down.weight.vk_buffer) {
        auto mid = alloc(count, projected_width);
        linear(features, w.projector, mid);
        activation(mid, 0);
        linear(mid, w.projector_down, output);
    } else linear(features, w.projector, output);
    batch.flush();
    return ctx.download(output);
}

VulkanVisionPrompt encode_vision_prompt_vulkan(VkContext& ctx, const VulkanVisionEncoder& encoder,
    const BpeTokenizer& tokenizer, std::string_view text, std::span<const Image* const> images,
    std::span<const size_t> offsets, int context) {
    if (offsets.size() != images.size()) throw std::invalid_argument("Image placeholder count does not match images");
    if (context <= 0) throw std::invalid_argument("Invalid vision prompt context");
    VulkanVisionPrompt prompt;
    const bool gemma4 = gemma4_projector(encoder.info().projector);
    const int32_t start = tokenizer.token_id(gemma4 ? "<|image>" : "<|vision_start|>");
    const int32_t end = tokenizer.token_id(gemma4 ? "<image|>" : "<|vision_end|>");
    if (gemma4 && !images.empty() && (start < 0 || end < 0))
        throw std::invalid_argument("Gemma 4 tokenizer must provide <|image> and <image|>");
    // Gemma 4 uses the padding token for per-layer input lookups on projected image rows.
    const int32_t pad = std::max(0, tokenizer.token_id(gemma4 ? "<pad>" : "<|image_pad|>"));
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
        auto rows = encoder.encode(ctx, *images[i], context - static_cast<int>(prompt.tokens.size()) - (end >= 0));
        const auto count = static_cast<size_t>(rows.desc.shape[0]);
        prompt.inputs_embeds.push_back({static_cast<int>(prompt.tokens.size()), {}, gemma4, std::move(rows)});
        prompt.tokens.insert(prompt.tokens.end(), count, pad);
        if (end >= 0) prompt.tokens.push_back(end);
        consumed = place + image_placeholder.size();
    }
    append(text.substr(consumed), images.empty());
    return prompt;
}
}  // namespace fusionrt
