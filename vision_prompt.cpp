// The image spans follow NVIDIA's Nemotron-H Omni processor (Apache-2.0, see
// licenses/nemotron-omni.txt).
#include "vision_prompt.hpp"

#include <algorithm>
#include <gufo/ext/image.hpp>
#include <gufo/ext/sha256.hpp>
#include <stdexcept>
#include <string>

namespace gufo::models::nemotron {
namespace {

/// Names what an image's embedding rows are computed from; changing the
/// preprocessing must change it, so cached states of other prompts don't
/// match.
constexpr std::string_view kPreprocessing =
    "nemotron-omni-image-v1;torch-bicubic-aa-f32;clip-normalized";

void AddU32(crypto::Sha256Hasher& hasher, std::uint32_t value) {
  const std::uint8_t bytes[4] = {static_cast<std::uint8_t>(value),
                                 static_cast<std::uint8_t>(value >> 8),
                                 static_cast<std::uint8_t>(value >> 16),
                                 static_cast<std::uint8_t>(value >> 24)};
  hasher.Update(bytes);
}

void AddText(crypto::Sha256Hasher& hasher, std::string_view text) {
  AddU32(hasher, static_cast<std::uint32_t>(text.size()));
  hasher.Update(std::span(reinterpret_cast<const std::uint8_t*>(text.data()),
                          text.size()));
}

}  // namespace

ImageFraming FindImageFraming(const Tokenizer& tokenizer) {
  const auto single = [&](std::string_view text) {
    const auto ids = tokenizer.Encode(text, true);
    if (ids.size() != 1)
      throw std::invalid_argument(
          "Nemotron vocabulary has no single token for " + std::string(text));
    return ids.front();
  };
  return {single("<image>"), single("<img>"), single("</img>")};
}

std::size_t VisionPrompt::Expanded(std::size_t rendered) const noexcept {
  std::size_t position = rendered;
  for (std::size_t i = 0; i < placeholders.size() && placeholders[i] < rendered;
       ++i)
    position += images[i].rows + 1;
  return position;
}

VisionPrompt ExpandImages(const VisionConfig& config,
                          const ImageFraming& framing,
                          std::span<const TokenId> rendered,
                          std::span<const VisionInput> images,
                          std::string_view encoder_identity, bool preprocess) {
  VisionPrompt prompt;
  for (std::size_t i = 0; i < rendered.size(); ++i) {
    if (rendered[i] == framing.placeholder)
      prompt.placeholders.push_back(i);
  }
  if (prompt.placeholders.size() != images.size())
    throw std::invalid_argument(
        "the prompt has " + std::to_string(prompt.placeholders.size()) +
        " image placeholders for " + std::to_string(images.size()) + " images");
  crypto::Sha256Hasher all;
  std::size_t next = 0;
  for (std::size_t k = 0; k < images.size(); ++k) {
    const std::string label = "image " + std::to_string(k + 1);
    if (!images[k].bytes)
      throw std::invalid_argument(label + " has no data");
    core::Image image;
    try {
      image = core::DecodeImage(*images[k].bytes);
    } catch (const std::exception& e) {
      throw std::invalid_argument(label + " cannot be decoded: " + e.what());
    }
    const auto grid = PlanGrid(config, image.width, image.height,
                               images[k].low_detail ? config.min_patches : 0);
    if (!grid)
      throw std::invalid_argument(label + " (" + std::to_string(image.width) +
                                  "x" + std::to_string(image.height) +
                                  ") is too elongated to encode");
    const auto at = static_cast<std::ptrdiff_t>(prompt.placeholders[k]);
    prompt.tokens.insert(prompt.tokens.end(),
                         rendered.begin() + static_cast<std::ptrdiff_t>(next),
                         rendered.begin() + at);
    next = prompt.placeholders[k] + 1;
    PromptImage out;
    out.grid = *grid;
    out.rows = ImageTokens(config, *grid);
    prompt.tokens.push_back(framing.start);
    out.offset = static_cast<std::uint32_t>(prompt.tokens.size());
    prompt.tokens.insert(prompt.tokens.end(), out.rows, framing.placeholder);
    prompt.tokens.push_back(framing.end);
    if (preprocess)
      out.patches = std::make_shared<const std::vector<float>>(
          PreprocessImage(config, image, *grid));
    if (k == 0) {
      prompt.prefix_identities.emplace_back();
    } else {
      const auto digest = all.Digest();
      prompt.prefix_identities.emplace_back(digest.begin(), digest.end());
    }
    crypto::Sha256Hasher one;
    AddText(one, kPreprocessing);
    AddText(one, encoder_identity);
    AddU32(one, out.offset);
    AddU32(one, grid->rows);
    AddU32(one, grid->cols);
    AddU32(one, image.width);
    AddU32(one, image.height);
    one.Update(image.pixels);
    all.Update(one.Finish());
    prompt.images.push_back(std::move(out));
  }
  prompt.tokens.insert(prompt.tokens.end(),
                       rendered.begin() + static_cast<std::ptrdiff_t>(next),
                       rendered.end());
  if (!images.empty()) {
    const auto digest = all.Digest();
    prompt.cache_identity.assign(digest.begin(), digest.end());
  }
  return prompt;
}

}  // namespace gufo::models::nemotron
