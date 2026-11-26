#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fusionrt {

inline constexpr std::string_view image_placeholder = "<image>";

struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};

Image decode_image(std::span<const uint8_t> bytes);
Image decode_image_base64(std::string_view payload);
Image decode_image_data_uri(std::string_view uri);
Image load_image(const std::string& path);

}  // namespace fusionrt
