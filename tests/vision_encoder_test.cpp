// The GPU vision encoder on NVIDIA's projector against the reference model
// (tools/vision_reference.py): sampled outputs and per-token norms of the
// synthetic images in the fixture, compared with the float32 reference and
// the bfloat16 one that NVIDIA's deployment matches.
//
//   nemotron_vision_encoder_test [TRUNK.gguf]
//
// The trunk comes from the argument or GUFO_NEMOTRON_BOLT_GGUF and the
// projector from beside it (mmproj-<trunk file name>); without them the test
// skips. With GUFO_NEMOTRON_VISION_REFERENCE naming the --dump directory of
// tools/vision_reference.py it also compares every stage in full.
// GUFO_NEMOTRON_VISION_OPERAND=bf16 encodes with BF16 operands instead of FP16.

#include "vision_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "config.hpp"
#include "tests/vision_synthetic.hpp"
#include "vision_image.hpp"

#ifndef GUFO_NEMOTRON_BOLT_VISION_ENCODER_GOLDEN
#define GUFO_NEMOTRON_BOLT_VISION_ENCODER_GOLDEN \
  "tests/fixtures/vision_encoder_golden.json"
#endif

namespace nb = gufo::models::nemotron;
namespace json = gufo::json;

namespace {

constexpr int kSkip = 77;
int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

json::Value LoadFixture() {
  std::filesystem::path path = GUFO_NEMOTRON_BOLT_VISION_ENCODER_GOLDEN;
  if (!std::filesystem::exists(path)) {
    if (const char* directory = std::getenv("GUFO_FIXTURE_DIR"))
      path = std::filesystem::path(directory) / path.filename();
  }
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return json::parse(buffer.str());
}

/// Root-mean-square of a - b relative to that of b.
double RelativeRms(const std::vector<double>& a, const std::vector<double>& b) {
  double diff = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    diff += (a[i] - b[i]) * (a[i] - b[i]);
    norm += b[i] * b[i];
  }
  return std::sqrt(diff / norm);
}

std::vector<double> Numbers(const json::Value& list) {
  std::vector<double> out;
  for (const auto& v : list.items())
    out.push_back(v.as_double());
  return out;
}

std::vector<float> ReadDump(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  input.seekg(0, std::ios::end);
  std::vector<float> values(static_cast<std::size_t>(input.tellg()) /
                            sizeof(float));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  return values;
}

/// Relative RMS error of `got` against a dump, and the worst cosine
/// similarity of a row of `width`.
void CompareStage(const std::string& label, const std::vector<float>& got,
                  const std::vector<float>& want, std::size_t width) {
  if (want.size() != got.size()) {
    Expect(false, label + ": " + std::to_string(got.size()) + " values, the " +
                      "reference has " + std::to_string(want.size()));
    return;
  }
  double diff = 0.0;
  double norm = 0.0;
  double worst_cos = 1.0;
  for (std::size_t row = 0; row < got.size() / width; ++row) {
    double dot = 0.0;
    double ga = 0.0;
    double wa = 0.0;
    for (std::size_t c = 0; c < width; ++c) {
      const double g = got[row * width + c];
      const double w = want[row * width + c];
      diff += (g - w) * (g - w);
      norm += w * w;
      dot += g * w;
      ga += g * g;
      wa += w * w;
    }
    worst_cos = std::min(worst_cos, dot / std::sqrt(ga * wa));
  }
  std::cout << "  " << std::left << std::setw(34) << label << " rel rms "
            << std::setprecision(4) << std::sqrt(diff / norm) << ", min cos "
            << std::setprecision(6) << worst_cos << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path trunk;
  if (argc > 1)
    trunk = argv[1];
  else if (const char* env = std::getenv("GUFO_NEMOTRON_BOLT_GGUF"))
    trunk = env;
  if (trunk.empty()) {
    std::cerr << "SKIP: set GUFO_NEMOTRON_BOLT_GGUF to a Nemotron 3.5 Bolt "
                 "trunk GGUF\n";
    return kSkip;
  }
  const auto projector =
      trunk.parent_path() / ("mmproj-" + trunk.filename().string());
  if (!std::filesystem::is_regular_file(projector)) {
    std::cerr << "SKIP: no projector beside the trunk: " << projector << '\n';
    return kSkip;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(
        trunk.string(), &error, {.allow_unsized_tensors = true});
    if (!reader)
      throw std::runtime_error(error);
    const auto config = nb::Config::FromGguf(*reader, &error);
    if (!config)
      throw std::runtime_error(error);
    nb::VisionEncoderOptions options;
    const char* operand = std::getenv("GUFO_NEMOTRON_VISION_OPERAND");
    const bool f16 = operand == nullptr || std::string_view(operand) != "bf16";
    if (!f16)
      options.operand = nb::rocm::VisionOperand::kBf16;
    const auto start = std::chrono::steady_clock::now();
    const auto encoder =
        nb::VisionEncoder::Load(projector.string(), *config, options, &error);
    if (!encoder)
      throw std::runtime_error(error);
    const auto loaded = std::chrono::steady_clock::now();
    if (!encoder->Prepare(&error))
      throw std::runtime_error(error);
    const auto prepared = std::chrono::steady_clock::now();
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    std::cout << "operand " << (f16 ? "f16" : "bf16") << ", loaded in "
              << seconds(start, loaded) << " s ("
              << encoder->ResidentBytes() / (1 << 20) << " MiB), prepared in "
              << seconds(loaded, prepared) << " s\n";
    const char* dumps = std::getenv("GUFO_NEMOTRON_VISION_REFERENCE");

    const auto golden = LoadFixture();
    const std::size_t samples = golden.member_size("samples");
    const auto& vc = encoder->config();
    for (const auto& entry : golden.find("images")->items()) {
      const auto width = static_cast<std::uint32_t>(entry.member_size("width"));
      const auto height =
          static_cast<std::uint32_t>(entry.member_size("height"));
      const std::string size =
          std::to_string(width) + "x" + std::to_string(height);
      const auto grid = nb::PlanGrid(vc, width, height);
      Expect(grid && grid->rows == entry.member_size("rows") &&
                 grid->cols == entry.member_size("cols"),
             size + ": grid differs from the reference");
      if (!grid)
        continue;
      const auto patches =
          nb::PreprocessImage(vc, nb::test::Synthetic(width, height), *grid);
      std::map<std::string, std::vector<float>> stages;
      const auto observer = [&](std::string_view stage,
                                std::span<const float> values) {
        if (dumps != nullptr || stage == "output")
          stages[std::string(stage)].assign(values.begin(), values.end());
      };
      // The first encode of a size allocates; time the second.
      (void)encoder->Encode(patches, *grid, &error);
      const auto begin = std::chrono::steady_clock::now();
      const auto embedding = encoder->Encode(patches, *grid, &error);
      const auto end = std::chrono::steady_clock::now();
      Expect(embedding != nullptr, size + ": " + error);
      if (!embedding)
        continue;
      (void)encoder->Encode(patches, *grid, &error, observer);
      const auto& output = stages["output"];
      const std::size_t tokens = entry.member_size("tokens");
      Expect(embedding->rows() == tokens &&
                 output.size() == tokens * vc.projection,
             size + ": token count");
      std::cout << size << ": " << tokens << " tokens in "
                << seconds(begin, end) * 1e3 << " ms\n";
      // Sampled outputs and per-token norms against both references.
      std::vector<double> got(samples);
      for (std::size_t k = 0; k < samples; ++k)
        got[k] = output[(k * 2654435761ULL) % output.size()];
      std::vector<double> norms(tokens);
      for (std::size_t t = 0; t < tokens; ++t) {
        double sum = 0.0;
        for (std::size_t c = 0; c < vc.projection; ++c)
          sum += static_cast<double>(output[t * vc.projection + c]) *
                 output[t * vc.projection + c];
        norms[t] = std::sqrt(sum);
      }
      const auto f32_samples = Numbers(*entry.find("float32")->find("samples"));
      const auto bf16_samples =
          Numbers(*entry.find("bfloat16")->find("samples"));
      const auto f32_norms = Numbers(*entry.find("float32")->find("norms"));
      const auto bf16_norms = Numbers(*entry.find("bfloat16")->find("norms"));
      const double ours_f32 = RelativeRms(got, f32_samples);
      const double ours_bf16 = RelativeRms(got, bf16_samples);
      const double baseline = RelativeRms(bf16_samples, f32_samples);
      const double norm_f32 = RelativeRms(norms, f32_norms);
      const double norm_baseline = RelativeRms(bf16_norms, f32_norms);
      std::cout << std::setprecision(4) << "  samples rel rms: vs float32 "
                << ours_f32 << ", vs bfloat16 " << ours_bf16
                << " (bfloat16 vs float32: " << baseline << ")\n"
                << "  token norms rel rms: vs float32 " << norm_f32
                << " (bfloat16 vs float32: " << norm_baseline << ")\n";
      // The encoder keeps float32 where NVIDIA's bfloat16 run rounds; with
      // FP16 operands it stays far closer to the float32 model (0.1 %
      // against 2.5 %), with BF16 ones at least as close.
      const double limit = f16 ? 0.005 : 1.2 * baseline;
      const double norm_limit = f16 ? 0.005 : 1.2 * norm_baseline + 1e-3;
      Expect(ours_f32 <= limit, size + ": outputs drift from float32");
      Expect(norm_f32 <= norm_limit, size + ": token norms drift from float32");
      if (dumps != nullptr) {
        for (const char* dtype : {"float32", "bfloat16"}) {
          for (const auto& [stage, values] : stages) {
            const auto want =
                ReadDump(std::filesystem::path(dumps) /
                         (size + "-" + dtype + "-" + stage + ".f32"));
            if (want.empty())
              continue;
            CompareStage(std::string(dtype) + " " + stage, values, want,
                         stage == "output" ? vc.projection : vc.width);
          }
        }
        // The bfloat16 reference against the float32 one.
        for (const std::string stage :
             {"embed", "block.0", "block.15", "block.31", "output"}) {
          const auto a = ReadDump(std::filesystem::path(dumps) /
                                  (size + "-bfloat16-" + stage + ".f32"));
          const auto b = ReadDump(std::filesystem::path(dumps) /
                                  (size + "-float32-" + stage + ".f32"));
          if (!a.empty())
            CompareStage("reference bfloat16 vs float32 " + stage, a, b,
                         stage == "output" ? vc.projection : vc.width);
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " checks failed\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
