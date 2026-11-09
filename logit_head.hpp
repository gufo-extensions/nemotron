#ifndef GUFO_MODELS_NEMOTRON_LOGIT_HEAD_HPP_
#define GUFO_MODELS_NEMOTRON_LOGIT_HEAD_HPP_

#include <cstddef>
#include <gufo/ext/sampling.hpp>
#include <optional>
#include <span>
#include <vector>

namespace gufo::models::nemotron {

/// What a LogitHead must hold: with `keep` nonzero, the keep best tokens;
/// otherwise at least `minimum` tokens and `top_p` of the softmax mass at
/// `temperature`.
struct LogitHeadFilter {
  float temperature{1.0F};
  std::size_t keep{0};
  float top_p{1.0F};
  std::size_t minimum{1};
};

/// A logit row summarized where it was computed, so that sampling need not
/// pass over the whole vocabulary: its largest finite logit, the sum over
/// every finite logit of exp((logit - maximum) / temperature), and every
/// finite token whose logit reaches some floor, ties included, in any order.
/// No candidates means no head.
struct LogitHead {
  float maximum{0.0F};
  float temperature{0.0F};
  double softmax_sum{0.0};
  std::vector<sampling::LogitCandidate> candidates{};
};

/// The head SampleWithHead can read for `config`: top-k sampling at a
/// positive temperature without penalties, keeping top-k raised to min-keep.
/// Other configurations sample from the full row.
[[nodiscard]] std::optional<LogitHeadFilter> HeadFilterFor(
    const sampling::SamplingConfig& config) noexcept;

/// Draws the token `sampler.Sample(logits)` draws and advances the sampler
/// the same way, reading `head` instead of the full row when the sampler's
/// next draw is plain top-k sampling and the head holds candidates.
[[nodiscard]] sampling::TokenId SampleWithHead(sampling::SamplerState& sampler,
                                               std::span<const float> logits,
                                               const LogitHead& head);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_LOGIT_HEAD_HPP_
