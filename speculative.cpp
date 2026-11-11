#include "speculative.hpp"

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#include "drafter.hpp"
#include "engine.hpp"

namespace gufo::models::nemotron {

namespace {

/// A round verifies drafts while the walk's transition probabilities,
/// scaled by the request's DraftHistory and multiplied from the first slot
/// on, stay at or above this; a slot the trunk is less likely to reach costs
/// more verification time than it saves on average. Measured with release
/// builds on code, an essay, an explanation, JSON and sampled prose: +3.5 to
/// +15 % tokens per second, except code (-1.8 %), whose rounds already keep
/// 6 of 7 drafts. Without the history's scaling, 0.2 lost 5 % on code.
constexpr float kMinDraftReach = 0.15F;
/// Drafts at or above this probability are nearly always kept and say
/// nothing about the lattice's calibration.
constexpr float kCertainDraft = 0.99F;
/// Per recorded draft, so the history follows the text within ~30 drafts.
constexpr double kHistoryDecay = 0.97;
/// Pseudo-counts that start the ratio at 1.
constexpr double kHistoryPrior = 2.0;

/// Drafts kept for verification: at least one, then while the cumulative
/// scaled probability stays at or above kMinDraftReach.
std::size_t DraftsWorthVerifying(const std::vector<float>& probabilities,
                                 const DraftHistory& history) {
  const auto scale = static_cast<float>((history.kept + kHistoryPrior) /
                                        (history.predicted + kHistoryPrior));
  float reach = 1.0F;
  std::size_t keep = 0;
  for (const float p : probabilities) {
    reach *= std::min(1.0F, p * scale);
    if (keep > 0 && reach < kMinDraftReach)
      break;
    ++keep;
  }
  return keep;
}

}  // namespace

bool SpeculativeStep(Session& session, sampling::SamplerState& sampler,
                     Frontier& frontier, DraftHistory& history,
                     std::uint32_t max_tokens,
                     const std::function<bool(std::int32_t)>& is_stop,
                     bool keep_rows, SpeculativeRound* out,
                     std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return false;
  };
  *out = {};
  if (max_tokens == 0)
    return fail("a decode round emits at least one token");
  const std::size_t vocab = frontier.logits.size();
  const auto filter = HeadFilterFor(sampler.config());
  const auto first = static_cast<std::int32_t>(
      SampleWithHead(sampler, frontier.logits, frontier.head));
  out->tokens.push_back(first);
  if (keep_rows)
    out->rows.assign(frontier.logits.begin(), frontier.logits.end());
  if (is_stop(first)) {
    out->stop = true;
    return true;
  }
  sampler.Accept(first);
  if (session.Position() >= session.MaxContext())
    return fail("the session context is full");

  std::vector<std::int32_t> drafts;
  std::vector<float> probabilities;
  DraftContext* draft = session.draft();
  const std::uint32_t room = session.MaxContext() - session.Position();
  if (draft != nullptr && max_tokens > 1 && room > 1 &&
      session.Position() > 0) {
    DraftLattice lattice;
    if (!draft->Draft(first, session.Position(), &lattice, {}, error_msg))
      return false;
    drafts = lattice.Walk(std::min(max_tokens - 1, room - 1), &probabilities);
    drafts.resize(DraftsWorthVerifying(probabilities, history));
  }
  if (drafts.empty()) {
    const std::int32_t one[] = {first};
    if (filter)
      return session.Append(one, frontier.logits, *filter, &frontier.head,
                            error_msg);
    frontier.head.candidates.clear();
    return session.Append(one, frontier.logits, error_msg);
  }

  std::vector<std::int32_t> block;
  block.reserve(drafts.size() + 1);
  block.push_back(first);
  block.insert(block.end(), drafts.begin(), drafts.end());
  static thread_local std::vector<float> rows;
  static thread_local std::vector<LogitHead> heads;
  rows.resize(block.size() * vocab);
  heads.resize(block.size());
  for (auto& head : heads)
    head.candidates.clear();
  if (!(filter ? session.Verify(block, rows, *filter, heads, error_msg)
               : session.Verify(block, rows, error_msg)))
    return false;
  std::uint32_t kept = 1;
  for (std::size_t i = 0; i < drafts.size(); ++i) {
    const std::span<const float> row(rows.data() + i * vocab, vocab);
    const auto token =
        static_cast<std::int32_t>(SampleWithHead(sampler, row, heads[i]));
    if (token != drafts[i]) {
      sampler.DeferSample(static_cast<sampling::TokenId>(token));
      break;
    }
    out->tokens.push_back(token);
    if (keep_rows)
      out->rows.insert(out->rows.end(), row.begin(), row.end());
    if (is_stop(token)) {
      // Like a first token that stops, it is reported but never run.
      out->stop = true;
      break;
    }
    sampler.Accept(token);
    ++kept;
  }
  out->drafted = static_cast<std::uint32_t>(drafts.size());
  out->accepted = static_cast<std::uint32_t>(out->tokens.size()) - 1;
  // The verified drafts: every accepted one and the first rejected one; a
  // stop token ends the round without a rejection.
  for (std::size_t i = 0; i < drafts.size() && i <= out->accepted; ++i) {
    if (i == out->accepted && out->stop)
      break;
    if (probabilities[i] >= kCertainDraft)
      continue;
    history.kept =
        kHistoryDecay * history.kept + (i < out->accepted ? 1.0 : 0.0);
    history.predicted = kHistoryDecay * history.predicted + probabilities[i];
  }
  if (!session.Commit(kept, error_msg))
    return false;
  const auto next =
      rows.begin() + static_cast<std::ptrdiff_t>((kept - 1) * vocab);
  std::copy(next, next + static_cast<std::ptrdiff_t>(vocab),
            frontier.logits.begin());
  std::swap(frontier.head, heads[kept - 1]);
  return true;
}

}  // namespace gufo::models::nemotron
