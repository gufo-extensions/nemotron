#include "probe.hpp"

#include <filesystem>
#include <string>

#include "config.hpp"

namespace gufo::models::nemotron {

bool ProbeBoltGguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
             "nemotron_h_moe" &&
         Config::FromGguf(reader).has_value();
}

bool ProbeBolt(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
      return false;
    std::string error;
    const auto reader = core::GgufReader::OpenFile(
        path.string(), &error, {.allow_unsized_tensors = true});
    return reader && ProbeBoltGguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::nemotron
