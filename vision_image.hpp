#ifndef GUFO_MODELS_NEMOTRON_VISION_IMAGE_HPP_
#define GUFO_MODELS_NEMOTRON_VISION_IMAGE_HPP_

#include <cstdint>
#include <gufo/ext/image.hpp>
#include <optional>
#include <vector>

#include "config.hpp"

namespace gufo::models::nemotron {

/// The grid of whole patches NVIDIA's image processor scales an image to.
struct ImageGrid {
  std::uint32_t rows{0};
  std::uint32_t cols{0};

  [[nodiscard]] std::uint32_t Patches() const noexcept { return rows * cols; }
  bool operator==(const ImageGrid&) const = default;
};

/// The processor's `_compute_target_patches`: the nearest grid of whole
/// patches, scaled into [min_patches, budget] with sides divisible by the
/// merge. The budget is max_patches unless given; a budget of min_patches
/// only shrinks images. Nullopt for an empty image or one so elongated that
/// a side would vanish.
[[nodiscard]] std::optional<ImageGrid> PlanGrid(const VisionConfig& config,
                                                std::uint32_t width,
                                                std::uint32_t height,
                                                std::uint32_t budget = 0);

/// Trunk tokens of one image: one per merge x merge patches.
[[nodiscard]] std::uint32_t ImageTokens(const VisionConfig& config,
                                        const ImageGrid& grid);

/// PyTorch's antialiased bicubic interpolation in float32
/// (`interpolate(mode="bicubic", align_corners=False, antialias=True)`) of
/// RGB8 `image` to `width` x `height`: three planes, row-major.
[[nodiscard]] std::vector<float> ResizePlanes(const core::Image& image,
                                              std::uint32_t width,
                                              std::uint32_t height);

/// The processor's pixels for `grid`: resized as above, normalized as
/// (x / 255 - mean) / deviation and cut into patches, rows x cols of
/// PatchValues floats, row-major over the grid, each patch channel-major
/// and then by pixel row.
[[nodiscard]] std::vector<float> PreprocessImage(const VisionConfig& config,
                                                 const core::Image& image,
                                                 const ImageGrid& grid);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_VISION_IMAGE_HPP_
