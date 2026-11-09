#include "logit_head.hpp"

#include <algorithm>

namespace gufo::models::nemotron {

std::optional<LogitHeadFilter> HeadFilterFor(
    const sampling::SamplingConfig& config) noexcept {
  // Gufo's sampler takes a compact distribution only for bounded top-k; it
  // filters top-p and min-p inside the kept tokens, so the head needs no
  // more than them. Penalties can lift a token the head left out.
  if (!(config.temperature > 0.0F) || config.top_k <= 0 ||
      config.penalties_enabled())
    return std::nullopt;
  const std::size_t minimum = std::max<std::size_t>(config.min_keep, 1);
  return LogitHeadFilter{
      .temperature = config.temperature,
      .keep = std::max(static_cast<std::size_t>(config.top_k), minimum),
      .top_p = config.top_p,
      .minimum = minimum};
}

sampling::TokenId SampleWithHead(sampling::SamplerState& sampler,
                                 std::span<const float> logits,
                                 const LogitHead& head) {
  // A forced or deferred token, a grammar, greedy decoding and penalties all
  // take the full row, as does a head the device could not fill.
  const auto& config = sampler.config();
  if (head.candidates.empty() || !HeadFilterFor(config) ||
      sampler.NeedsConstraintMask() || sampler.ForcedToken() ||
      sampler.SaveDrawState().pending)
    return sampler.Sample(logits);
  return sampler.DistributionFromCandidates(head.candidates)
      .Sample(sampler.mutable_rng_state());
}

}  // namespace gufo::models::nemotron
