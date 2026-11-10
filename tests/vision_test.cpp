// Checks the vision input side against goldens of NVIDIA's image processor
// (tools/vision_golden.py).
//
//   nemotron_vision_test image
//       Patch grids, and the resized and normalized patches of synthetic
//       images, bit for bit.
//   nemotron_vision_test prompt
//       Placeholder expansion, positions and cache identities of prompts
//       with images.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "config.hpp"
#include "tests/vision_synthetic.hpp"
#include "vision_image.hpp"
#include "vision_prompt.hpp"

#ifndef GUFO_NEMOTRON_BOLT_VISION_GOLDEN
#define GUFO_NEMOTRON_BOLT_VISION_GOLDEN \
  "tests/fixtures/vision_image_golden.json"
#endif

namespace nb = gufo::models::nemotron;
namespace json = gufo::json;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL nemotron_vision_test: " << message << '\n';
    std::exit(1);
  }
}

json::Value LoadFixture() {
  std::filesystem::path path = GUFO_NEMOTRON_BOLT_VISION_GOLDEN;
  if (!std::filesystem::exists(path)) {
    if (const char* directory = std::getenv("GUFO_FIXTURE_DIR"))
      path = std::filesystem::path(directory) / path.filename();
  }
  std::ifstream input(path, std::ios::binary);
  Check(static_cast<bool>(input), "cannot open " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return json::parse(buffer.str());
}

/// The released projector's values (VisionConfig::FromGguf reads them from
/// the file; these tests need no weights).
nb::VisionConfig BoltVision() {
  nb::VisionConfig c;
  c.width = 1280;
  c.num_layers = 32;
  c.num_heads = 16;
  c.ff_size = 5120;
  c.patch_size = 16;
  c.merge = 2;
  c.prefix_tokens = 10;
  c.position_side = 128;
  c.projector_hidden = 20480;
  c.projection = 4096;
  c.min_patches = 1024;
  c.max_patches = 13312;
  c.mean = {0.48145466F, 0.4578275F, 0.40821073F};
  c.deviation = {0.26862954F, 0.26130258F, 0.27577711F};
  return c;
}

/// An uncompressed PNG of `image` (stored deflate blocks), enough for the
/// decoder.
std::vector<std::uint8_t> EncodePng(const gufo::core::Image& image) {
  std::uint32_t table[256];
  for (std::uint32_t n = 0; n < 256; ++n) {
    std::uint32_t c = n;
    for (int k = 0; k < 8; ++k)
      c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1) : c >> 1;
    table[n] = c;
  }
  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  const auto u32 = [](std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8)
      out.push_back(static_cast<std::uint8_t>(v >> shift));
  };
  const auto chunk = [&](const char* type,
                         const std::vector<std::uint8_t>& data) {
    u32(png, static_cast<std::uint32_t>(data.size()));
    std::vector<std::uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const std::uint8_t b : body)
      crc = table[(crc ^ b) & 0xFFU] ^ (crc >> 8);
    png.insert(png.end(), body.begin(), body.end());
    u32(png, crc ^ 0xFFFFFFFFU);
  };
  std::vector<std::uint8_t> header;
  u32(header, image.width);
  u32(header, image.height);
  header.insert(header.end(), {8, 2, 0, 0, 0});
  chunk("IHDR", header);
  std::vector<std::uint8_t> raw;
  for (std::uint32_t y = 0; y < image.height; ++y) {
    raw.push_back(0);
    const auto* row = image.pixels.data() + std::size_t{y} * image.width * 3;
    raw.insert(raw.end(), row, row + std::size_t{image.width} * 3);
  }
  std::vector<std::uint8_t> zlib = {0x78, 0x01};
  for (std::size_t at = 0; at < raw.size(); at += 65535) {
    const auto n = static_cast<std::uint16_t>(
        std::min<std::size_t>(65535, raw.size() - at));
    zlib.push_back(at + n == raw.size() ? 1 : 0);
    zlib.insert(
        zlib.end(),
        {static_cast<std::uint8_t>(n), static_cast<std::uint8_t>(n >> 8),
         static_cast<std::uint8_t>(~n), static_cast<std::uint8_t>(~n >> 8)});
    zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(at),
                raw.begin() + static_cast<std::ptrdiff_t>(at + n));
  }
  std::uint32_t a = 1;
  std::uint32_t b = 0;
  for (const std::uint8_t v : raw) {
    a = (a + v) % 65521U;
    b = (b + a) % 65521U;
  }
  u32(zlib, (b << 16) | a);
  chunk("IDAT", zlib);
  chunk("IEND", {});
  return png;
}

void CheckPrompt(const nb::VisionConfig& config) {
  const nb::ImageFraming framing{18, 19, 20};
  const auto png = [](std::uint32_t w, std::uint32_t h) {
    return std::make_shared<const std::vector<std::uint8_t>>(
        EncodePng(nb::test::Synthetic(w, h)));
  };
  const std::vector<nb::TokenId> rendered = {1, 2, 18, 3, 18, 4};
  std::vector<nb::VisionInput> images = {{png(640, 480), false},
                                         {png(64, 64), false}};
  const auto prompt =
      nb::ExpandImages(config, framing, rendered, images, "encoder-a", true);
  // 640x480 is 30x40 patches (300 tokens); 64x64 grows to 32x32 (256).
  Check(prompt.images.size() == 2 && prompt.images[0].rows == 300 &&
            prompt.images[1].rows == 256,
        "image token counts");
  Check(prompt.images[0].offset == 3 && prompt.images[1].offset == 306,
        "image offsets");
  Check(prompt.tokens.size() == 6 + 301 + 257, "expanded length");
  Check(prompt.tokens[2] == 19 && prompt.tokens[303] == 20 &&
            prompt.tokens[304] == 3 && prompt.tokens[305] == 19 &&
            prompt.tokens[562] == 20 && prompt.tokens[563] == 4,
        "framing tokens");
  for (const auto& image : prompt.images) {
    for (std::uint32_t i = 0; i < image.rows; ++i)
      Check(prompt.tokens[image.offset + i] == 18, "image rows hold <image>");
    Check(image.patches &&
              image.patches->size() ==
                  std::size_t{image.grid.Patches()} * config.PatchValues(),
          "preprocessed patches");
  }
  Check(prompt.Expanded(0) == 0 && prompt.Expanded(2) == 2 &&
            prompt.Expanded(3) == 304 && prompt.Expanded(5) == 563 &&
            prompt.Expanded(6) == 564,
        "rendered to expanded positions");
  // Identities: nothing before the first image, the first image before the
  // second, every image for the whole prompt.
  Check(prompt.prefix_identities.size() == 2 &&
            prompt.prefix_identities[0].empty() &&
            prompt.prefix_identities[1].size() == 32 &&
            prompt.cache_identity.size() == 32 &&
            prompt.cache_identity != prompt.prefix_identities[1],
        "prefix identities");
  // A different second image keeps what precedes it.
  auto changed = images;
  auto pixels = nb::test::Synthetic(64, 64);
  pixels.pixels[100] ^= 1;
  changed[1].bytes =
      std::make_shared<const std::vector<std::uint8_t>>(EncodePng(pixels));
  const auto other =
      nb::ExpandImages(config, framing, rendered, changed, "encoder-a", false);
  Check(other.tokens == prompt.tokens && !other.images[1].patches,
        "sizing keeps the tokens without pixels");
  Check(other.prefix_identities[1] == prompt.prefix_identities[1] &&
            other.cache_identity != prompt.cache_identity,
        "a changed image changes only the identities that include it");
  // Another encoder changes every image identity.
  const auto encoder =
      nb::ExpandImages(config, framing, rendered, images, "encoder-b", false);
  Check(encoder.prefix_identities[1] != prompt.prefix_identities[1] &&
            encoder.cache_identity != prompt.cache_identity,
        "the encoder is part of the identity");
  // Low detail caps a large image at min_patches.
  const std::vector<nb::VisionInput> large = {{png(2000, 1400), true}};
  const std::vector<nb::TokenId> one = {18};
  const auto low = nb::ExpandImages(config, framing, one, large, "e", false);
  Check(low.images[0].grid.Patches() <= config.min_patches &&
            low.images[0].grid.Patches() > config.min_patches / 2,
        "low detail grid");
  // Placeholders and images must agree, and images must decode.
  const auto throws = [&](const std::vector<nb::TokenId>& tokens,
                          const std::vector<nb::VisionInput>& inputs) {
    try {
      (void)nb::ExpandImages(config, framing, tokens, inputs, "e", false);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  Check(throws(rendered, {images[0]}), "placeholder count mismatch");
  const std::vector<nb::VisionInput> broken = {
      {std::make_shared<const std::vector<std::uint8_t>>(
           std::vector<std::uint8_t>{1, 2, 3}),
       false}};
  Check(throws(one, broken), "undecodable image");
  std::cout << "prompt: expansion, positions and identities match\n";
}

void CheckGrids(const nb::VisionConfig& config, const json::Value& golden) {
  std::size_t count = 0;
  for (const auto& entry : golden.find("grids")->items()) {
    const auto& v = entry.items();
    const auto width = static_cast<std::uint32_t>(v[0].as_size());
    const auto height = static_cast<std::uint32_t>(v[1].as_size());
    const nb::ImageGrid want{static_cast<std::uint32_t>(v[2].as_size()),
                             static_cast<std::uint32_t>(v[3].as_size())};
    const auto got = nb::PlanGrid(config, width, height);
    const std::string size =
        std::to_string(width) + "x" + std::to_string(height);
    if (want.rows == 0) {
      Check(!got.has_value(), "grid of " + size + " should not exist");
    } else {
      Check(got.has_value(), "no grid for " + size);
      Check(*got == want,
            "grid of " + size + " is " + std::to_string(got->rows) + "x" +
                std::to_string(got->cols) + ", want " +
                std::to_string(want.rows) + "x" + std::to_string(want.cols));
    }
    ++count;
  }
  std::cout << "grids: " << count << " match\n";
}

void CheckImages(const nb::VisionConfig& config, const json::Value& golden) {
  const std::size_t samples = golden.member_size("samples");
  for (const auto& entry : golden.find("images")->items()) {
    const auto width = static_cast<std::uint32_t>(entry.member_size("width"));
    const auto height = static_cast<std::uint32_t>(entry.member_size("height"));
    const std::string size =
        std::to_string(width) + "x" + std::to_string(height);
    const nb::ImageGrid grid{
        static_cast<std::uint32_t>(entry.member_size("rows")),
        static_cast<std::uint32_t>(entry.member_size("cols"))};
    const auto values =
        nb::PreprocessImage(config, nb::test::Synthetic(width, height), grid);
    Check(values.size() ==
              static_cast<std::size_t>(grid.Patches()) * config.PatchValues(),
          "patch count of " + size);
    const auto& bits = entry.find("bits")->items();
    Check(bits.size() == samples, "sample count of " + size);
    std::size_t mismatched = 0;
    float worst = 0.0F;
    for (std::size_t k = 0; k < samples; ++k) {
      const std::size_t index = (k * 2654435761ULL) % values.size();
      const auto want_bits = static_cast<std::uint32_t>(bits[k].as_size());
      float want = 0.0F;
      std::memcpy(&want, &want_bits, sizeof(want));
      std::uint32_t got_bits = 0;
      std::memcpy(&got_bits, &values[index], sizeof(got_bits));
      if (got_bits != want_bits) {
        ++mismatched;
        worst = std::max(worst, std::abs(values[index] - want));
      }
    }
    double sum = 0.0;
    double abs_sum = 0.0;
    for (const float v : values) {
      sum += v;
      abs_sum += std::abs(v);
    }
    const double want_sum = entry.member_double("sum");
    const double want_abs = entry.member_double("abs_sum");
    std::cout << size << " -> " << grid.rows << "x" << grid.cols << ": "
              << mismatched << "/" << samples << " samples differ (max "
              << worst << "), sum " << sum << " vs " << want_sum << '\n';
    Check(mismatched == 0, "preprocessed pixels of " + size + " differ");
    Check(std::abs(sum - want_sum) <= 1e-6 * want_abs &&
              std::abs(abs_sum - want_abs) <= 1e-9 * want_abs,
          "pixel sums of " + size + " differ");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc > 1 ? argv[1] : "image";
  const auto config = BoltVision();
  if (mode == "image") {
    const auto golden = LoadFixture();
    CheckGrids(config, golden);
    CheckImages(config, golden);
  } else if (mode == "prompt") {
    CheckPrompt(config);
  } else {
    Check(false, "unknown mode " + std::string(mode));
  }
  std::cout << "PASS\n";
  return 0;
}
