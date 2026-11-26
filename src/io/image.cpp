#include "fusionrt/image.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fusionrt {
namespace {

constexpr size_t max_encoded = 16 * 1024 * 1024;
using Bytes = std::span<const uint8_t>;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }

Image image(int64_t width, int64_t height) {
    if (width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 8 * 1024 * 1024)
        invalid("Image dimensions exceed the supported limit");
    return {static_cast<int>(width), static_cast<int>(height), std::vector<uint8_t>(width * height * 3)};
}

uint32_t integer(Bytes b, size_t offset, size_t count, bool little = false) {
    if (offset > b.size() || count > b.size() - offset) invalid("Truncated image header");
    uint32_t n = 0;
    for (size_t i = 0; i < count; ++i) n = (n << 8) | b[offset + (little ? count - i - 1 : i)];
    return n;
}

Image pnm(Bytes b) {
    const bool gray = b[1] == '2' || b[1] == '5';
    const bool ascii = b[1] == '2' || b[1] == '3';
    size_t pos = 2;
    auto number = [&]() {
        while (pos < b.size()) {
            if (std::isspace(b[pos])) ++pos;
            else if (b[pos] == '#') while (pos < b.size() && b[pos] != '\n') ++pos;
            else break;
        }
        if (pos == b.size() || b[pos] < '0' || b[pos] > '9') invalid("Invalid PPM/PGM header or sample");
        int n = 0;
        while (pos < b.size() && b[pos] >= '0' && b[pos] <= '9') {
            if (n > 65535) invalid("PPM/PGM number too large");
            n = n * 10 + b[pos++] - '0';
        }
        return n;
    };
    const int w = number(), h = number(), max_sample = number();
    auto out = image(w, h);
    if (max_sample < 1 || max_sample > 65535) invalid("Invalid PPM/PGM sample range");
    const size_t sample_bytes = max_sample > 255 ? 2 : 1;
    if (!ascii) {
        if (pos >= b.size() || !std::isspace(b[pos])) invalid("Missing PPM/PGM raster delimiter");
        if (b[pos++] == '\r' && pos < b.size() && b[pos] == '\n') ++pos;
        const size_t need = out.rgb.size() / (gray ? 3 : 1) * sample_bytes;
        if (need > b.size() - pos) invalid("Truncated PPM/PGM raster");
    }
    auto sample = [&]() {
        const int v = ascii ? number() : static_cast<int>(integer(b, pos, sample_bytes));
        if (!ascii) pos += sample_bytes;
        if (v > max_sample) invalid("PPM/PGM sample exceeds its range");
        return static_cast<uint8_t>((v * 255 + max_sample / 2) / max_sample);
    };
    for (size_t i = 0; i < out.rgb.size(); i += 3) {
        out.rgb[i] = sample();
        out.rgb[i + 1] = gray ? out.rgb[i] : sample();
        out.rgb[i + 2] = gray ? out.rgb[i] : sample();
    }
    return out;
}

Image bmp(Bytes b) {
    auto le = [&](size_t offset, size_t count) { return integer(b, offset, count, true); };
    const size_t offset = le(10, 4), dib = le(14, 4);
    const int w = std::bit_cast<int32_t>(le(18, 4)), h = std::bit_cast<int32_t>(le(22, 4));
    const auto bpp = le(28, 2);
    if (dib < 40 || offset > b.size() || offset < 14 || dib > offset - 14 ||
        le(26, 2) != 1 || le(30, 4) != 0 || (bpp != 24 && bpp != 32))
        invalid("Expected an uncompressed 24-bit or 32-bit BMP");
    auto out = image(w, h < 0 ? -static_cast<int64_t>(h) : h);
    const size_t pixel = bpp / 8, stride = (w * pixel + 3) & ~size_t{3};
    if (stride * out.height > b.size() - offset) invalid("Truncated BMP raster");
    for (int y = 0; y < out.height; ++y) {
        const auto* src = b.data() + offset + stride * (h < 0 ? y : out.height - 1 - y);
        auto* dst = out.rgb.data() + static_cast<size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            dst[x * 3] = src[x * pixel + 2];
            dst[x * 3 + 1] = src[x * pixel + 1];
            dst[x * 3 + 2] = src[x * pixel];
        }
    }
    return out;
}

class Bits {
    Bytes bytes_;
    size_t pos_ = 0;
public:
    explicit Bits(Bytes bytes) : bytes_(bytes) {}
    unsigned get(unsigned n) {
        if (n > bytes_.size() * 8 - pos_) invalid("Truncated PNG deflate stream");
        unsigned value = 0;
        for (unsigned i = 0; i < n; ++i, ++pos_) value |= ((bytes_[pos_ / 8] >> (pos_ % 8)) & 1u) << i;
        return value;
    }
    void align() { pos_ = (pos_ + 7) & ~size_t{7}; }
    bool at_end() const { return (pos_ + 7) / 8 == bytes_.size(); }
};

struct Huffman {
    std::array<unsigned, 16> count{}, first{}, offset{};
    std::vector<unsigned> symbols;
    explicit Huffman(Bytes lengths) : symbols(lengths.size()) {
        for (auto len : lengths) {
            if (len > 15) invalid("Invalid PNG Huffman length");
            if (len) ++count[len];
        }
        for (unsigned len = 1; len <= 15; ++len) {
            first[len] = (first[len - 1] + count[len - 1]) * 2;
            if (first[len] + count[len] > (1u << len)) invalid("Oversubscribed PNG Huffman tree");
            offset[len] = offset[len - 1] + count[len - 1];
        }
        auto next = offset;
        for (unsigned s = 0; s < lengths.size(); ++s) if (lengths[s]) symbols[next[lengths[s]]++] = s;
    }
    unsigned read(Bits& bits) const {
        unsigned code = 0;
        for (unsigned len = 1; len <= 15; ++len) {
            code = code * 2 + bits.get(1);
            if (code >= first[len] && code - first[len] < count[len]) return symbols[offset[len] + code - first[len]];
        }
        invalid("Invalid PNG Huffman code");
    }
};

std::vector<uint8_t> inflate(Bytes b, size_t expected) {
    if (b.size() < 6 || (b[0] & 15) != 8 || (b[0] >> 4) > 7 || (b[1] & 32) ||
        (b[0] * 256 + b[1]) % 31 != 0) invalid("Invalid PNG zlib header");
    Bits bits(b.subspan(2, b.size() - 6));
    std::vector<uint8_t> out;
    out.reserve(expected);
    auto append = [&](unsigned v) {
        if (out.size() == expected) invalid("PNG deflate stream exceeds image dimensions");
        out.push_back(static_cast<uint8_t>(v));
    };
    bool final;
    do {
        final = bits.get(1) != 0;
        const auto type = bits.get(2);
        if (type == 0) {
            bits.align();
            const auto n = bits.get(16);
            if ((n ^ bits.get(16)) != 65535) invalid("Invalid PNG stored deflate block");
            for (unsigned i = 0; i < n; ++i) append(bits.get(8));
            continue;
        }
        std::vector<uint8_t> ll(288), dd(32, 5);
        if (type == 1) {
            for (unsigned i = 0; i < ll.size(); ++i) ll[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
        } else if (type == 2) {
            const unsigned nl = bits.get(5) + 257, nd = bits.get(5) + 1, nc = bits.get(4) + 4;
            constexpr std::array<unsigned, 19> order{16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
            std::array<uint8_t, 19> lengths{};
            for (unsigned i = 0; i < nc; ++i) lengths[order[i]] = static_cast<uint8_t>(bits.get(3));
            const Huffman code(lengths);
            std::vector<uint8_t> all;
            while (all.size() < nl + nd) {
                const auto s = code.read(bits);
                if (s < 16) all.push_back(static_cast<uint8_t>(s));
                else {
                    if (s == 16 && all.empty()) invalid("Invalid PNG Huffman repetition");
                    const unsigned n = s == 16 ? bits.get(2) + 3 : s == 17 ? bits.get(3) + 3 : bits.get(7) + 11;
                    const uint8_t v = s == 16 ? all.back() : 0;
                    if (n > nl + nd - all.size()) invalid("PNG Huffman lengths overflow");
                    all.insert(all.end(), n, v);
                }
            }
            ll.assign(all.begin(), all.begin() + nl);
            dd.assign(all.begin() + nl, all.end());
        } else invalid("Unsupported PNG deflate block");
        if (!ll[256]) invalid("PNG deflate block has no end code");
        const Huffman literal(ll), distance(dd);
        for (;;) {
            const unsigned s = literal.read(bits);
            if (s < 256) { append(s); continue; }
            if (s == 256) break;
            if (s > 285) invalid("Invalid PNG deflate length");
            const unsigned l = s - 257, extra = l < 8 || l == 28 ? 0 : l / 4 - 1;
            const unsigned len = (l == 28 ? 258 : l < 8 ? l + 3 : ((4 + l % 4) << extra) + 3) + bits.get(extra);
            const unsigned d = distance.read(bits);
            if (d > 29) invalid("Invalid PNG deflate distance");
            const unsigned de = d < 2 ? 0 : d / 2 - 1;
            const unsigned dist = (d < 4 ? d + 1 : ((2 + d % 2) << de) + 1) + bits.get(de);
            if (dist > out.size()) invalid("PNG deflate back-reference precedes the raster");
            for (unsigned i = 0; i < len; ++i) append(out[out.size() - dist]);
        }
    } while (!final);
    if (out.size() != expected || !bits.at_end()) invalid("Incorrect PNG raster length");
    unsigned a = 1, c = 0;
    for (auto v : out) { a = (a + v) % 65521; c = (c + a) % 65521; }
    if ((c << 16 | a) != integer(b, b.size() - 4, 4)) invalid("Incorrect PNG zlib checksum");
    return out;
}

uint32_t crc(Bytes bytes) {
    static constexpr auto table = [] {
        std::array<uint32_t, 256> t{};
        for (unsigned i = 0; i < t.size(); ++i) {
            uint32_t n = i;
            for (int j = 0; j < 8; ++j) n = (n >> 1) ^ ((n & 1) ? 0xedb88320u : 0);
            t[i] = n;
        }
        return t;
    }();
    uint32_t n = 0xffffffffu;
    for (auto b : bytes) n = (n >> 8) ^ table[(n ^ b) & 255];
    return ~n;
}

Image png(Bytes b) {
    Image out;
    std::vector<uint8_t> compressed, palette;
    int channels = 0, color = -1;
    bool finished = false;
    for (size_t p = 8; p < b.size();) {
        if (b.size() - p < 12) invalid("Truncated PNG chunk");
        const size_t n = integer(b, p, 4);
        if (n > b.size() - p - 12) invalid("Truncated PNG chunk data");
        const std::string_view type(reinterpret_cast<const char*>(b.data() + p + 4), 4);
        const auto data = b.subspan(p + 8, n);
        if (crc(b.subspan(p + 4, n + 4)) != integer(b, p + 8 + n, 4)) invalid("Incorrect PNG chunk checksum");
        if (type == "IHDR") {
            if (p != 8 || n != 13 || data[8] != 8 || data[10] || data[11] || data[12])
                invalid("Expected non-interlaced 8-bit PNG");
            color = data[9];
            channels = color == 0 || color == 3 ? 1 : color == 2 ? 3 : color == 4 ? 2 : color == 6 ? 4 : 0;
            if (!channels) invalid("Unsupported PNG color type");
            out = image(integer(data, 0, 4), integer(data, 4, 4));
        } else if (!channels) invalid("PNG is missing IHDR");
        else if (type == "IDAT") compressed.insert(compressed.end(), data.begin(), data.end());
        else if (type == "PLTE") {
            if (n == 0 || n % 3 != 0 || n > 768 || !compressed.empty() || !palette.empty()) invalid("Invalid PNG palette");
            palette.assign(data.begin(), data.end());
        } else if (type == "IEND") {
            if (n != 0 || p + 12 != b.size()) invalid("Invalid PNG end chunk");
            finished = true;
            break;
        } else if (std::isupper(static_cast<unsigned char>(type.front()))) invalid("Unsupported PNG critical chunk");
        p += n + 12;
    }
    if (!finished || compressed.empty() || (color == 3 && palette.empty())) invalid("Incomplete PNG image");
    const size_t stride = static_cast<size_t>(out.width) * channels;
    auto raster = inflate(compressed, (stride + 1) * out.height);
    for (int y = 0; y < out.height; ++y) {
        auto* row = raster.data() + (stride + 1) * y + 1;
        const uint8_t filter = row[-1];
        if (filter > 4) invalid("Invalid PNG scanline filter");
        for (size_t x = 0; x < stride; ++x) {
            const int a = x >= static_cast<size_t>(channels) ? row[x - channels] : 0;
            const int b0 = y ? *(row + x - (stride + 1)) : 0;
            const int c = y && x >= static_cast<size_t>(channels) ? *(row + x - (stride + 1) - channels) : 0;
            const int pa = std::abs(b0 - c), pb = std::abs(a - c), pc = std::abs(a + b0 - 2 * c);
            const int add = filter == 0 ? 0 : filter == 1 ? a : filter == 2 ? b0 :
                            filter == 3 ? (a + b0) / 2 : pa <= pb && pa <= pc ? a : pb <= pc ? b0 : c;
            row[x] = static_cast<uint8_t>(row[x] + add);
        }
        for (int x = 0; x < out.width; ++x) {
            const auto* src = row + x * channels;
            if (color == 3) {
                if (static_cast<size_t>(*src) * 3 >= palette.size()) invalid("PNG palette index out of range");
                src = palette.data() + *src * 3;
            }
            auto* dst = out.rgb.data() + (static_cast<size_t>(y) * out.width + x) * 3;
            if (color == 0 || color == 4) std::fill_n(dst, 3, *src);
            else std::copy_n(src, 3, dst);
        }
    }
    return out;
}

}  // namespace

Image decode_image(Bytes bytes) {
    constexpr std::array<uint8_t, 8> png_signature{137,80,78,71,13,10,26,10};
    if (bytes.size() < 2 || bytes.size() > max_encoded) invalid("Empty or oversized image");
    if (bytes.size() >= png_signature.size() && std::equal(png_signature.begin(), png_signature.end(), bytes.begin()))
        return png(bytes);
    if (bytes[0] == 'B' && bytes[1] == 'M') return bmp(bytes);
    if (bytes[0] == 'P' && (bytes[1] == '2' || bytes[1] == '3' || bytes[1] == '5' || bytes[1] == '6')) return pnm(bytes);
    invalid("Unsupported image format (expected PNG, BMP, PPM or PGM)");
}

Image decode_image_base64(std::string_view payload) {
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (payload.empty() || payload.size() % 4 != 0 || payload.size() > 4 * ((max_encoded + 2) / 3))
        invalid("Invalid or oversized image base64");
    std::vector<uint8_t> decoded;
    decoded.reserve(payload.size() / 4 * 3);
    for (size_t i = 0; i < payload.size(); i += 4) {
        const bool p2 = payload[i + 2] == '=', p3 = payload[i + 3] == '=';
        if ((p2 && !p3) || ((p2 || p3) && i + 4 != payload.size())) invalid("Invalid image base64 padding");
        uint32_t bits = 0;
        for (size_t j = 0; j < 4; ++j) {
            const auto n = (j == 2 && p2) || (j == 3 && p3) ? 0 : alphabet.find(payload[i + j]);
            if (n == std::string_view::npos) invalid("Invalid image base64 character");
            bits = (bits << 6) | static_cast<uint32_t>(n);
        }
        if ((p2 && (bits & 0xffffu)) || (p3 && !p2 && (bits & 0xffu))) invalid("Invalid image base64 trailing bits");
        decoded.push_back(static_cast<uint8_t>(bits >> 16));
        if (!p2) decoded.push_back(static_cast<uint8_t>(bits >> 8));
        if (!p3) decoded.push_back(static_cast<uint8_t>(bits));
    }
    return decode_image(decoded);
}

Image decode_image_data_uri(std::string_view uri) {
    const auto comma = uri.find(',');
    if (!uri.starts_with("data:image/") || comma == std::string_view::npos || !uri.substr(0, comma).ends_with(";base64"))
        invalid("Expected an image data URI with base64 encoding");
    return decode_image_base64(uri.substr(comma + 1));
}

Image load_image(const std::string& path) {
    std::ifstream in(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary | std::ios::ate);
    const auto n = in.tellg();
    if (n <= 0 || static_cast<uint64_t>(n) > max_encoded) invalid("Cannot open image or image file is oversized");
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    if (!in.seekg(0).read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        invalid("Could not read image file");
    return decode_image(bytes);
}

}  // namespace fusionrt
