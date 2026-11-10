// The patch grid follows NVIDIA's Nemotron-H Omni image processor
// (Apache-2.0, see licenses/nemotron-omni.txt); the resize ports PyTorch's
// antialiased bicubic interpolation (BSD-3-Clause, see licenses/pytorch.txt).
#include "vision_image.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace gufo::models::nemotron {
namespace {

// PyTorch's CPU kernels run as their AVX2 or AVX-512 builds, in which the
// compiler fuses multiply-adds and vectorizes loops; the explicit fma calls
// and the split sums below reproduce those roundings bit for bit.

/// PyTorch's filter for antialiased bicubic resampling (Keys, a = -0.5),
/// in float like `HelperInterpCubic::aa_filter`.
float AaFilter(float x) {
  constexpr float a = -0.5F;
  x = std::abs(x);
  if (x < 1.0F)
    return std::fma(std::fma(a + 2, x, -(a + 3)) * x, x, 1.0F);
  if (x < 2.0F)
    return std::fma(std::fma(std::fma(a, x, -(5 * a)), x, 8 * a), x, -(4 * a));
  return 0.0F;
}

/// Source span and normalized weights of every output position along one
/// axis, as PyTorch's `_compute_index_ranges_weights` computes them with
/// antialiasing. The expressions keep its mix of float and double
/// arithmetic, which decides the last bits of the weights.
struct Taps {
  std::vector<std::int64_t> first;
  std::vector<std::int64_t> count;
  std::vector<float> weights;  ///< [output][stride]
  std::int64_t stride{0};
};

Taps ComputeTaps(std::int64_t input_size, std::int64_t output_size) {
  constexpr int kInterpSize = 4;
  constexpr double kAlignCornersDelta = 0.0;
  const float scale = static_cast<float>(input_size) / output_size;
  const float support =
      (scale >= 1.0) ? (kInterpSize * 0.5) * scale : kInterpSize * 0.5;
  const float invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  Taps taps;
  taps.stride = static_cast<std::int64_t>(std::ceil(support) * 2 + 1);
  taps.first.resize(static_cast<std::size_t>(output_size));
  taps.count.resize(static_cast<std::size_t>(output_size));
  taps.weights.assign(static_cast<std::size_t>(output_size * taps.stride),
                      0.0F);
  for (std::int64_t i = 0; i < output_size; ++i) {
    const float center = scale * (i + 0.5 - kAlignCornersDelta);
    const std::int64_t xmin = std::max(
        static_cast<std::int64_t>(center - support + 0.5 + kAlignCornersDelta),
        std::int64_t{0});
    std::int64_t xsize =
        std::min(static_cast<std::int64_t>(center + support + 0.5 +
                                           kAlignCornersDelta),
                 input_size) -
        xmin;
    xsize = std::clamp(xsize, std::int64_t{0}, taps.stride);
    float* w = taps.weights.data() + i * taps.stride;
    float total = 0.0F;
    for (std::int64_t j = 0; j < xsize; ++j) {
      const float value =
          AaFilter((j + xmin - center + 0.5 - kAlignCornersDelta) * invscale);
      w[j] = value;
      total += value;
    }
    if (total != 0.0F) {
      for (std::int64_t j = 0; j < xsize; ++j)
        w[j] /= total;
    }
    taps.first[static_cast<std::size_t>(i)] = xmin;
    taps.count[static_cast<std::size_t>(i)] = xsize;
  }
  return taps;
}

/// One output value as PyTorch's `interpolate_aa_single_dim` sums it: the
/// first product, then the remaining taps in order. The compiled loop
/// multiplies four taps at a time and adds the products one by one; the
/// taps left over are fused multiply-adds.
float Interpolate(const float* source, std::size_t step, const float* weights,
                  std::int64_t count) {
  float output = source[0] * weights[0];
  const std::int64_t vectorized = 1 + (count - 1) / 4 * 4;
  std::int64_t j = 1;
  for (; j < vectorized; ++j) {
    const float product =
        source[static_cast<std::size_t>(j) * step] * weights[j];
    output += product;
  }
  for (; j < count; ++j)
    output = std::fma(source[static_cast<std::size_t>(j) * step], weights[j],
                      output);
  return output;
}

}  // namespace

std::optional<ImageGrid> PlanGrid(const VisionConfig& config,
                                  std::uint32_t width, std::uint32_t height,
                                  std::uint32_t max_patches) {
  if (width == 0 || height == 0 || config.patch_size == 0 || config.merge == 0)
    return std::nullopt;
  const double patch = config.patch_size;
  const double budget = max_patches == 0
                            ? config.max_patches
                            : std::min(max_patches, config.max_patches);
  const double minimum = config.min_patches;
  const double merge = config.merge;
  // Python's round() of `x + 0.5` rounds halves to even, as nearbyint does
  // in the default rounding mode.
  double h = std::nearbyint(height / patch + 0.5);
  double w = std::nearbyint(width / patch + 0.5);
  const double factor = std::min(std::sqrt(budget / (h * w)), 1.0);
  h = std::floor(factor * h);
  w = std::floor(factor * w);
  if (budget > minimum && h * w < minimum) {
    if (h * w == 0)
      return std::nullopt;
    const double up = std::sqrt(minimum / (h * w));
    h = std::ceil(up * h);
    w = std::ceil(up * w);
  }
  if (const double rest = std::fmod(h, merge); rest != 0) {
    if ((h + merge - rest) * w <= budget)
      h += merge - rest;
    else
      h = std::max(merge, h - rest);
  }
  if (const double rest = std::fmod(w, merge); rest != 0) {
    if (h * (w + merge - rest) <= budget)
      w += merge - rest;
    else
      w = std::max(merge, w - rest);
  }
  if (h < 1 || w < 1 || h * w > budget)
    return std::nullopt;
  return ImageGrid{static_cast<std::uint32_t>(h),
                   static_cast<std::uint32_t>(w)};
}

std::uint32_t ImageTokens(const VisionConfig& config, const ImageGrid& grid) {
  return grid.Patches() / (config.merge * config.merge);
}

std::vector<float> ResizePlanes(const core::Image& image, std::uint32_t width,
                                std::uint32_t height) {
  const std::size_t in_w = image.width;
  const std::size_t in_h = image.height;
  if (in_w == 0 || in_h == 0 || width == 0 || height == 0 ||
      image.pixels.size() != in_w * in_h * 3)
    throw std::invalid_argument("invalid image for resizing");
  // The uint8 pixels as float planes, the tensor the processor resizes.
  std::vector<float> planes(3 * in_w * in_h);
  for (std::size_t c = 0; c < 3; ++c) {
    float* plane = planes.data() + c * in_w * in_h;
    for (std::size_t i = 0; i < in_w * in_h; ++i)
      plane[i] = static_cast<float>(image.pixels[i * 3 + c]);
  }
  // Width first, then height, each pass skipped when its size is kept.
  std::size_t cur_w = in_w;
  if (width != in_w) {
    const Taps taps = ComputeTaps(static_cast<std::int64_t>(in_w), width);
    std::vector<float> out(3 * in_h * width);
    for (std::size_t r = 0; r < 3 * in_h; ++r) {
      const float* src = planes.data() + r * in_w;
      float* dst = out.data() + r * width;
      for (std::size_t x = 0; x < width; ++x)
        dst[x] =
            Interpolate(src + taps.first[x], 1,
                        taps.weights.data() + x * taps.stride, taps.count[x]);
    }
    planes = std::move(out);
    cur_w = width;
  }
  if (height != in_h) {
    const Taps taps = ComputeTaps(static_cast<std::int64_t>(in_h), height);
    std::vector<float> out(3 * height * cur_w);
    for (std::size_t c = 0; c < 3; ++c) {
      const float* src = planes.data() + c * in_h * cur_w;
      float* dst = out.data() + c * height * cur_w;
      for (std::size_t y = 0; y < height; ++y) {
        const float* weights = taps.weights.data() + y * taps.stride;
        const float* row =
            src + static_cast<std::size_t>(taps.first[y]) * cur_w;
        for (std::size_t x = 0; x < cur_w; ++x)
          dst[y * cur_w + x] =
              Interpolate(row + x, cur_w, weights, taps.count[y]);
      }
    }
    planes = std::move(out);
  }
  return planes;
}

std::vector<float> PreprocessImage(const VisionConfig& config,
                                   const core::Image& image,
                                   const ImageGrid& grid) {
  const std::size_t patch = config.patch_size;
  const std::size_t width = grid.cols * patch;
  const std::size_t height = grid.rows * patch;
  const auto planes = ResizePlanes(image, static_cast<std::uint32_t>(width),
                                   static_cast<std::uint32_t>(height));
  const std::size_t values = config.PatchValues();
  std::vector<float> patches(static_cast<std::size_t>(grid.Patches()) * values);
  for (std::size_t c = 0; c < 3; ++c) {
    const float mean = config.mean[c];
    const float deviation = config.deviation[c];
    const float* plane = planes.data() + c * width * height;
    for (std::size_t y = 0; y < height; ++y) {
      const std::size_t row = y / patch;
      const std::size_t py = y % patch;
      for (std::size_t x = 0; x < width; ++x) {
        const std::size_t p = row * grid.cols + x / patch;
        patches[p * values + (c * patch + py) * patch + x % patch] =
            (plane[y * width + x] / 255.0F - mean) / deviation;
      }
    }
  }
  return patches;
}

}  // namespace gufo::models::nemotron
