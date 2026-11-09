// Probes NVIDIA's released files: the trunk is claimed, the drafter and the
// vision projector are not, every tensor core can't size is NVFP4 and fits
// its reserved bytes, and the trunk and drafter weights bind.
//
//   nemotron_probe_test [TRUNK.gguf]
//
// The trunk comes from the argument or GUFO_NEMOTRON_BOLT_GGUF; without it
// the test skips. The drafter and projector are looked up beside the trunk
// by NVIDIA's file names and are checked when present.

#include "probe.hpp"

#include <cstdlib>
#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "nvfp4.hpp"
#include "weights.hpp"

namespace nemotron = gufo::models::nemotron;
using gufo::core::GgufReader;

namespace {

constexpr int kSkip = 77;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::unique_ptr<GgufReader> Open(const std::filesystem::path& path) {
  std::string error;
  auto reader = GgufReader::OpenFile(path.string(), &error,
                                     {.allow_unsized_tensors = true});
  Require(reader != nullptr, "cannot open " + path.string() + ": " + error);
  return reader;
}

// Every unsized tensor is NVFP4, and its blocks fit the bytes the file
// reserves for it. Returns how many there are.
std::size_t CheckUnsized(const GgufReader& reader) {
  std::size_t count = 0;
  for (const auto& tensor : reader.GetTensors()) {
    if (tensor.sized)
      continue;
    const std::string name(tensor.name);
    Require(tensor.type == nemotron::kNvfp4, name + " is unsized, not NVFP4");
    const auto elements = tensor.ElementCount();
    Require(elements % nemotron::kNvfp4BlockElements == 0,
            name + " is not whole NVFP4 blocks");
    const auto bytes =
        elements / nemotron::kNvfp4BlockElements * sizeof(nemotron::BlockNvfp4);
    Require(bytes <= tensor.size_bytes, name + " overruns its reserved bytes");
    ++count;
  }
  return count;
}

nemotron::Config CheckTrunk(const std::filesystem::path& path) {
  const auto reader = Open(path);
  Require(nemotron::ProbeBoltGguf(*reader), "trunk not claimed by metadata");
  Require(nemotron::ProbeBolt(path.string()), "trunk not claimed by path");
  Require(CheckUnsized(*reader) > 0, "trunk has no NVFP4 tensors");
  std::string error;
  auto weights = nemotron::ModelWeights::Bind(*reader, &error);
  Require(weights.has_value(), "trunk weights: " + error);
  return weights->config;
}

void CheckDrafter(const std::filesystem::path& path,
                  const nemotron::Config& target) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    std::cout << "absent, not checked: " << path.filename().string() << '\n';
    return;
  }
  const auto reader = Open(path);
  CheckUnsized(*reader);
  std::string error;
  Require(nemotron::DrafterWeights::Bind(*reader, target, &error).has_value(),
          "drafter weights: " + error);
}

void CheckRejected(const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    std::cout << "absent, not checked: " << path.filename().string() << '\n';
    return;
  }
  const auto reader = Open(path);
  Require(!nemotron::ProbeBoltGguf(*reader),
          path.filename().string() + " claimed by metadata");
  Require(!nemotron::ProbeBolt(path.string()),
          path.filename().string() + " claimed by path");
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
  try {
    const auto config = CheckTrunk(trunk);
    const auto dir = trunk.parent_path();
    const auto stem = trunk.stem().string();
    CheckDrafter(dir / (stem + "-LiLiCorr.gguf"), config);
    CheckRejected(dir / (stem + "-LiLiCorr.gguf"));
    CheckRejected(dir / ("mmproj-" + trunk.filename().string()));
    Require(!nemotron::ProbeBolt(dir.string()), "directory claimed");
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
