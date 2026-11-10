// Times the vision input path: decoding and preprocessing on the CPU, then
// the encoder on the GPU, for image files or synthetic images of a size.
//
//   nemotron_vision_probe TRUNK.gguf MMPROJ.gguf [--repeat N] WxH|IMAGE...
//
// Each input is encoded once to choose the matrix-product algorithms and
// allocate, then timed over N runs (default 5); the median is printed with
// the encoder's arithmetic rate.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/image.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "tests/vision_synthetic.hpp"
#include "vision_encoder.hpp"
#include "vision_image.hpp"

namespace nb = gufo::models::nemotron;
using Clock = std::chrono::steady_clock;

namespace {

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Multiply-adds times two of one encode: ViT products and attention, then
/// the projector.
double Flops(const nb::VisionConfig& c, const nb::ImageGrid& grid) {
  const double n = grid.Patches();
  const double t = n + c.prefix_tokens;
  const double w = c.width;
  const double vit = 2.0 * t * (4 * w * w + 2 * w * c.ff_size) +
                     4.0 * t * t * w;  // QK^T and PV
  const double merged = n / (c.merge * c.merge);
  const double projector =
      2.0 * merged *
      (static_cast<double>(c.MergedWidth()) * c.projector_hidden +
       static_cast<double>(c.projector_hidden) * c.projection);
  return 2.0 * n * c.PatchValues() * w + c.num_layers * vit + projector;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: nemotron_vision_probe TRUNK.gguf MMPROJ.gguf "
                 "[--repeat N] WxH|IMAGE...\n";
    return 2;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(
        argv[1], &error, {.allow_unsized_tensors = true});
    if (!reader)
      throw std::runtime_error(error);
    const auto trunk = nb::Config::FromGguf(*reader, &error);
    if (!trunk)
      throw std::runtime_error(error);
    const auto encoder = nb::VisionEncoder::Load(argv[2], *trunk, {}, &error);
    if (!encoder || !encoder->Prepare(&error))
      throw std::runtime_error(error);
    const auto& c = encoder->config();
    int repeat = 5;
    std::cout << std::fixed << std::setprecision(1);
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--repeat" && i + 1 < argc) {
        repeat = std::max(1, std::atoi(argv[++i]));
        continue;
      }
      gufo::core::Image image;
      std::vector<double> decode;
      const auto x = arg.find('x');
      if (x != std::string::npos &&
          arg.find_first_not_of("0123456789x") == std::string::npos) {
        image = nb::test::Synthetic(
            static_cast<std::uint32_t>(std::stoul(arg.substr(0, x))),
            static_cast<std::uint32_t>(std::stoul(arg.substr(x + 1))));
      } else {
        const auto bytes = gufo::core::ReadImageFile(arg);
        for (int r = 0; r < repeat; ++r) {
          const auto start = Clock::now();
          image = gufo::core::DecodeImage(bytes);
          decode.push_back(Since(start));
        }
      }
      const auto grid = nb::PlanGrid(c, image.width, image.height);
      if (!grid)
        throw std::runtime_error(arg + ": no grid");
      std::vector<double> preprocess, encode;
      std::vector<float> patches;
      for (int r = 0; r < repeat; ++r) {
        const auto start = Clock::now();
        patches = nb::PreprocessImage(c, image, *grid);
        preprocess.push_back(Since(start));
      }
      if (!encoder->Encode(patches, *grid, &error))
        throw std::runtime_error(error);
      for (int r = 0; r < repeat; ++r) {
        const auto start = Clock::now();
        if (!encoder->Encode(patches, *grid, &error))
          throw std::runtime_error(error);
        encode.push_back(Since(start));
      }
      const double seconds = Median(encode);
      std::cout << arg << " (" << image.width << "x" << image.height
                << "): grid " << grid->rows << "x" << grid->cols << ", "
                << nb::ImageTokens(c, *grid) << " tokens";
      if (!decode.empty())
        std::cout << ", decode " << Median(decode) * 1e3 << " ms";
      std::cout << ", preprocess " << Median(preprocess) * 1e3 << " ms, encode "
                << seconds * 1e3 << " ms (" << Flops(c, *grid) / seconds * 1e-12
                << " TFLOPS)\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "nemotron_vision_probe: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
