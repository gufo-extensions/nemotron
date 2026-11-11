#ifndef GUFO_MODELS_NEMOTRON_SPECULATIVE_HPP_
#define GUFO_MODELS_NEMOTRON_SPECULATIVE_HPP_

#include <cstdint>
#include <functional>
#include <gufo/ext/sampling.hpp>
#include <string>
#include <vector>

#include "logit_head.hpp"

namespace gufo::models::nemotron {

class Session;

/// The distribution the next token is drawn from: its logits row and, when
/// the session summarized it on the device, its head (empty otherwise).
struct Frontier {
  std::vector<float> logits;
  LogitHead head;
};

/// How the trunk has accepted the drafts the lattice was unsure about, in
/// one request: decayed sums of the kept drafts and of the lattice's
/// probabilities for them. The lattice's probabilities track acceptance on
/// average but not per text (code keeps uncertain drafts about twice as
/// often as prose), so rounds scale them by this ratio.
struct DraftHistory {
  double kept{0.0};
  double predicted{0.0};
};

struct SpeculativeRound {
  /// Emitted tokens; the first was drawn from the incoming row.
  std::vector<std::int32_t> tokens;
  /// With keep_rows, the row each token was drawn from, in order.
  std::vector<float> rows;
  std::uint32_t drafted{0};
  std::uint32_t accepted{0};
  bool stop{false};  ///< The last token ends the response and was not run.
};

/// One draft-and-verify round. `frontier` holds the session's next-token
/// distribution and is replaced by the one the next token will be drawn
/// from; for random sampling the session summarizes the rows on the device,
/// so draws read a few candidates instead of the vocabulary. The first
/// token comes from `sampler`; the drafter proposes up to max_tokens - 1
/// more, of which the round keeps the leading ones likely enough to be
/// accepted (at least one) by the lattice's probabilities scaled with
/// `history`, and the trunk verifies them in one pass; `history` then
/// records the verified drafts. Each draft is kept while
/// the sampler, given the trunk's row before it, draws that same draft, so
/// the emitted tokens follow the trunk's own distribution. On the first
/// mismatch the trunk's draw stays pending in the sampler (DeferSample) and
/// opens the next round. Emitted tokens are run and accepted into `sampler`,
/// except a stop token: it ends the round as the last token, is not run, and
/// `frontier` keeps the row it was drawn from. Without a drafter, or with
/// max_tokens == 1, the round is one ordinary decode step.
[[nodiscard]] bool SpeculativeStep(
    Session& session, sampling::SamplerState& sampler, Frontier& frontier,
    DraftHistory& history, std::uint32_t max_tokens,
    const std::function<bool(std::int32_t)>& is_stop, bool keep_rows,
    SpeculativeRound* out, std::string* error_msg = nullptr);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_SPECULATIVE_HPP_
