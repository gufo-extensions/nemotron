#ifndef GUFO_MODELS_NEMOTRON_RUNNER_HPP_
#define GUFO_MODELS_NEMOTRON_RUNNER_HPP_

#include <cstdint>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

#include "engine.hpp"

namespace gufo::models::nemotron {

/// NVIDIA's recommendation for every thinking mode: temperature 1.0 and
/// top-p 0.95. The server and the terminal commands share it.
[[nodiscard]] sampling::TextSamplingDefaults SamplingDefaults();

class VisionEncoder;

/// A text runner over a loaded model. `draft_tokens` bounds the drafts of
/// one LiLiCorr round (zero decodes one token per step; drafting needs the
/// model's drafter). Empty fingerprints key continuations to this process;
/// the GGUF identities of trunk and drafter make them durable. With `vision`
/// the runner takes images.
[[nodiscard]] std::shared_ptr<ext::TextModelRunner> CreateNemotronRunner(
    std::shared_ptr<Model> model, std::uint32_t context,
    std::uint32_t draft_tokens, std::string artifact_fingerprint = {},
    std::string draft_fingerprint = {},
    std::shared_ptr<VisionEncoder> vision = nullptr);

/// The provider's factory: loads the trunk, for `--speculative dflash2`
/// NVIDIA's LiLiCorr drafter, and for `--mmproj` its vision projector.
[[nodiscard]] std::shared_ptr<ext::TextModelRunner> CreateNemotronTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_RUNNER_HPP_
