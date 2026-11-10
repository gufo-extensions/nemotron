#ifndef GUFO_MODELS_NEMOTRON_TESTS_VISION_SYNTHETIC_HPP_
#define GUFO_MODELS_NEMOTRON_TESTS_VISION_SYNTHETIC_HPP_

#include <cstdint>
#include <gufo/ext/image.hpp>

namespace gufo::models::nemotron::test {

/// tools/vision_golden.py's synthetic image: gradients plus hash noise.
inline core::Image Synthetic(std::uint32_t width, std::uint32_t height) {
  core::Image image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<std::size_t>(width) * height * 3);
  for (std::uint64_t y = 0; y < height; ++y) {
    for (std::uint64_t x = 0; x < width; ++x) {
      for (std::uint64_t c = 0; c < 3; ++c) {
        const std::uint64_t h =
            ((x * 73856093ULL) ^ (y * 19349663ULL) ^ (c * 83492791ULL)) &
            0xFFFFFFFFULL;
        const std::uint64_t v =
            (x * 255) / width + (y * 128) / height + c * 40 + (h >> 13) % 48;
        image.pixels[(y * width + x) * 3 + c] =
            static_cast<std::uint8_t>(v % 256);
      }
    }
  }
  return image;
}

}  // namespace gufo::models::nemotron::test

#endif  // GUFO_MODELS_NEMOTRON_TESTS_VISION_SYNTHETIC_HPP_
