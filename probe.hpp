#ifndef GUFO_MODELS_NEMOTRON_PROBE_HPP_
#define GUFO_MODELS_NEMOTRON_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::nemotron {

/// A Nemotron 3.5 Bolt trunk: `nemotron_h_moe` metadata that Config accepts.
[[nodiscard]] bool ProbeBoltGguf(const core::GgufReader& reader);
/// `model_root` is a Nemotron 3.5 Bolt trunk GGUF.
[[nodiscard]] bool ProbeBolt(std::string_view model_root);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_PROBE_HPP_
