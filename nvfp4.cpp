#include "nvfp4.hpp"

#include <cmath>

namespace gufo::models::nemotron {
namespace {

// E2M1 values doubled, as in ggml's kvalues_mxfp4; the halving is folded into
// Ue4m3ToFloat.
constexpr std::int8_t kValuesE2m1x2[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                           0, -1, -2, -3, -4, -6, -8, -12};

// Unsigned E4M3 (bias 7) scale times 0.5, mirroring ggml_ue4m3_to_fp32.
float Ue4m3ToFloat(std::uint8_t x) noexcept {
  if (x == 0 || x == 0x7F) {
    return 0.0F;
  }
  const int exponent = (x >> 3U) & 0x0F;
  const int mantissa = x & 0x07;
  const float raw =
      exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                    : std::ldexp(1.0F + (static_cast<float>(mantissa) / 8.0F),
                                 exponent - 7);
  return raw * 0.5F;
}

}  // namespace

void DequantizeNvfp4(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const BlockNvfp4*>(src);
  const std::size_t nb = k / kNvfp4BlockElements;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t s = 0; s < 4; ++s) {
      const float d = Ue4m3ToFloat(blocks[b].d[s]);
      float* y = dst + (b * kNvfp4BlockElements) + (s * 16);
      for (std::size_t j = 0; j < 8; ++j) {
        const std::uint8_t packed = blocks[b].qs[(s * 8) + j];
        y[j] = static_cast<float>(kValuesE2m1x2[packed & 0x0FU]) * d;
        y[j + 8] = static_cast<float>(kValuesE2m1x2[packed >> 4U]) * d;
      }
    }
  }
}

}  // namespace gufo::models::nemotron
