#ifndef GUFO_MODELS_NEMOTRON_VISION_PROMPT_HPP_
#define GUFO_MODELS_NEMOTRON_VISION_PROMPT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "config.hpp"
#include "tokenizer.hpp"
#include "vision_image.hpp"

namespace gufo::models::nemotron {

/// The tokens around an image's embedding rows. The chat template renders
/// `placeholder` (<image>) per image; NVIDIA's processor replaces it with
/// `start` (<img>), one `placeholder` per embedding row and `end` (</img>).
struct ImageFraming {
  TokenId placeholder{0};
  TokenId start{0};
  TokenId end{0};
};

/// The framing tokens of `tokenizer`. Throws std::invalid_argument when one
/// is not a single token.
[[nodiscard]] ImageFraming FindImageFraming(const Tokenizer& tokenizer);

/// One image of a request: its encoded bytes (PNG, JPEG or WebP) and whether
/// the client asked for low detail, which caps it at min_patches patches.
struct VisionInput {
  std::shared_ptr<const std::vector<std::uint8_t>> bytes;
  bool low_detail{false};
};

/// One image of an expanded prompt.
struct PromptImage {
  ImageGrid grid;
  /// Position of its first embedding row, after its <img>.
  std::uint32_t offset{0};
  /// Embedding rows: one per merged token.
  std::uint32_t rows{0};
  /// PreprocessImage's patches; null when the prompt was only sized.
  std::shared_ptr<const std::vector<float>> patches;
};

struct VisionPrompt {
  std::vector<TokenId> tokens;
  std::vector<PromptImage> images;
  /// Identity of every image (empty without images), and per image the
  /// identity of the images before it (empty for the first). Each covers the
  /// preprocessing, the encoder, the image's place, grid and pixels.
  std::vector<std::uint8_t> cache_identity;
  std::vector<std::vector<std::uint8_t>> prefix_identities;
  /// Index of each placeholder among the rendered tokens.
  std::vector<std::size_t> placeholders;

  /// Position in `tokens` of the rendered token at `rendered`.
  [[nodiscard]] std::size_t Expanded(std::size_t rendered) const noexcept;
};

/// Expands each placeholder of the rendered prompt into its image's span,
/// decoding and, with `preprocess`, resizing the images in order; without it
/// the prompt is only sized. `encoder_identity` names the encoder whose
/// embeddings the identities stand for. Throws std::invalid_argument when
/// placeholders and images disagree or an image can't be decoded or sized.
[[nodiscard]] VisionPrompt ExpandImages(const VisionConfig& config,
                                        const ImageFraming& framing,
                                        std::span<const TokenId> rendered,
                                        std::span<const VisionInput> images,
                                        std::string_view encoder_identity,
                                        bool preprocess);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_VISION_PROMPT_HPP_
