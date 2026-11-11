#ifndef GUFO_MODELS_NEMOTRON_VISION_ENCODER_HPP_
#define GUFO_MODELS_NEMOTRON_VISION_ENCODER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "config.hpp"
#include "kernels/rocm/vision.hpp"
#include "vision_image.hpp"

namespace gufo::models::nemotron {

/// The trunk input rows of one image on the device: one float32 row of the
/// trunk's width per merged token, row-major over the halved grid.
class VisionEmbedding {
public:
  ~VisionEmbedding();
  VisionEmbedding(const VisionEmbedding&) = delete;
  VisionEmbedding& operator=(const VisionEmbedding&) = delete;

  [[nodiscard]] const float* data() const noexcept { return data_; }
  [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

private:
  friend class VisionEncoder;
  VisionEmbedding() = default;
  float* data_{nullptr};
  std::uint32_t rows_{0};
  std::uint32_t width_{0};
};

struct VisionEncoderOptions {
  /// Format of the matrix-product operands. FP16 keeps the encoder about ten
  /// times closer to the float32 model than BF16, in which NVIDIA runs it;
  /// every operand stays far inside FP16's range (QUALITY.md).
  rocm::VisionOperand operand{rocm::VisionOperand::kF16};
};

/// NVIDIA's vision projector on the GPU: the C-RADIO ViT over an image's
/// patches and the projector into the trunk's embedding space. The residual
/// stream, norms, softmax and accumulation are float32.
class VisionEncoder {
public:
  /// Receives float32 activations after a stage ("embed", "block.N",
  /// "output"), for parity checks.
  using Observer =
      std::function<void(std::string_view stage, std::span<const float>)>;

  /// Loads and uploads the mmproj GGUF at `path` for `trunk`.
  [[nodiscard]] static std::unique_ptr<VisionEncoder> Load(
      const std::string& path, const Config& trunk,
      const VisionEncoderOptions& options = {},
      std::string* error_msg = nullptr);
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  [[nodiscard]] const VisionConfig& config() const noexcept;
  [[nodiscard]] rocm::VisionOperand operand() const noexcept;
  /// Full-content identity of the projector file (gguf-sha256-v1).
  [[nodiscard]] const std::string& identity() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Device scratch an image of the largest grid needs.
  [[nodiscard]] std::size_t MaxScratchBytes() const noexcept;

  /// Picks the matrix-product algorithms for every image size now instead
  /// of during the first images.
  [[nodiscard]] bool Prepare(std::string* error_msg = nullptr);

  /// Encodes PreprocessImage's patches of `grid`. Calls are serialized.
  /// With `yield`, the encoder calls it between two blocks every 250 ms or
  /// so and once at the end, with its stream idle, so that other sessions
  /// can decode; it must not encode.
  [[nodiscard]] std::shared_ptr<const VisionEmbedding> Encode(
      std::span<const float> patches, const ImageGrid& grid,
      std::string* error_msg = nullptr, const Observer& observer = {},
      const std::function<void()>& yield = {});

private:
  struct Impl;
  explicit VisionEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_VISION_ENCODER_HPP_
