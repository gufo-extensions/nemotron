#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/json.hpp>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "continuation_adapter.hpp"
#include "drafter.hpp"
#include "logit_head.hpp"
#include "speculative.hpp"
#include "vision_encoder.hpp"
#include "vision_prompt.hpp"

namespace gufo::models::nemotron {

sampling::TextSamplingDefaults SamplingDefaults() {
  const sampling::TextSamplingValues values{.temperature = 1.0F,
                                            .top_p = 0.95F};
  return {values, values, values};
}

namespace {
using namespace gufo::ext;

/// One cache slot's session plus the next-token distribution at its frontier.
class State final : public TextRunnerState {
public:
  State(Session& session, cache::Adapter& adapter, cache::Slot& slot,
        std::size_t vocabulary)
      : session(&session), adapter_(&adapter), slot_(&slot) {
    frontier.logits.resize(vocabulary);
  }
  Session* session;
  /// The host copy of the frontier's logits and, when the device summarized
  /// them, their head.
  Frontier frontier;
  /// The host copy matches the session's frontier.
  bool logits_ready{false};
  /// The head filter of the request decoding from this state, for the
  /// device summary of each next row.
  std::optional<LogitHeadFilter> head_filter;
  /// The prompt closed reasoning itself ("<think></think>") and no answer
  /// text has been emitted yet: whitespace the model writes first is the
  /// separator after that block, framing rather than answer.
  bool leading_separator{false};
  /// The current prompt's images, and the embeddings of those that prefill
  /// has reached but not yet passed, by image index.
  std::shared_ptr<const VisionPrompt> vision;
  std::map<std::size_t, std::shared_ptr<const VisionEmbedding>> embeddings;

  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    logits_ready = false;
    frontier.head.candidates.clear();
  }

private:
  cache::Adapter* adapter_;
  cache::Slot* slot_;
};

State& Get(TextRunnerState& state) {
  return dynamic_cast<State&>(state);
}
const State& Get(const TextRunnerState& state) {
  return dynamic_cast<const State&>(state);
}

/// The images of a prompt with images, travelling with its request.
struct VisionContext final : TextPromptContext {
  std::shared_ptr<const VisionPrompt> prompt;
};

bool HasImages(const ChatRequest& request) {
  return std::ranges::any_of(request.messages,
                             [](const auto& m) { return !m.images.empty(); });
}

constexpr std::string_view kStateAbi =
    "nemotron-h/state/v1;kv=f16;mamba=f32;chunk=2048";

class Runner final : public HipTextModelRunner {
public:
  Runner(std::shared_ptr<Model> model, std::uint32_t context,
         std::uint32_t draft_tokens, std::string artifact_fingerprint,
         std::string draft_fingerprint, std::shared_ptr<VisionEncoder> vision)
      : model_(std::move(model)),
        vision_(std::move(vision)),
        context_(context),
        draft_tokens_(
            model_->drafter() == nullptr
                ? 0
                : std::min(draft_tokens, model_->drafter()->config().Slots())) {
    const auto valid = [](const std::string& fingerprint) {
      return fingerprint.size() == 64 &&
             fingerprint.find_first_not_of("0123456789abcdef") ==
                 std::string::npos;
    };
    const bool drafting = model_->drafter() != nullptr;
    std::string identity;
    if (!artifact_fingerprint.empty() &&
        (!drafting || !draft_fingerprint.empty())) {
      if (!valid(artifact_fingerprint) ||
          (drafting && !valid(draft_fingerprint)))
        throw std::invalid_argument("invalid Nemotron artifact fingerprint");
      const std::string scheme(core::kGgufIdentityScheme);
      identity =
          "schema=gufo-text-continuation-v2\nmodel_kind=nemotron-h\n"
          "artifact_id=" +
          scheme + ':' + artifact_fingerprint +
          "\nstate_abi=" + std::string(kStateAbi) +
          "\ntokenizer=nemotron-byte-bpe-v1\nchat_template=nemotron-3.5-v1\n"
          "context_tokens=" +
          std::to_string(context_) + "\ndrafter=" +
          (drafting ? scheme + ':' + draft_fingerprint : std::string("none")) +
          '\n';
    } else {
      // Without file fingerprints the loaded model is the identity; durable
      // caches need the artifact identities above.
      std::ostringstream process;
      process << "schema=gufo-process-text-continuation-v1\nmodel_instance="
              << model_.get() << "\ncontext_tokens=" << context_ << '\n';
      identity = process.str();
    }
    component_identity_.assign(identity.begin(), identity.end());
    const auto close = model_->tokenizer().Encode("</think>", true);
    if (close.size() != 1)
      throw std::invalid_argument("Nemotron vocabulary has no </think> token");
    thought_close_ = close.front();
    if (vision_) {
      framing_ = FindImageFraming(model_->tokenizer());
      // Embeddings depend on the projector and on the operand format.
      vision_identity_ =
          std::string(core::kGgufIdentityScheme) + ':' + vision_->identity() +
          (vision_->operand() == rocm::VisionOperand::kF16 ? ";operand=fp16"
                                                           : ";operand=bf16");
    }
  }

  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    return hip::CreateHipComponentCacheResources(
        std::make_unique<ContinuationAdapter>(model_, context_,
                                              component_identity_),
        ledger,
        {[] { return HostSnapshotCeilingBytes(); }, [](std::string_view) {},
         [](std::string_view) {}});
  }
  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& nemotron = dynamic_cast<ContinuationAdapter&>(adapter);
    return std::make_unique<State>(nemotron.GetSession(slot), adapter, slot,
                                   model_->config().vocab_size);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t) const override {
    // The frontier changed under this state; read its logits on demand.
    auto& s = Get(state);
    s.logits_ready = false;
    s.frontier.head.candidates.clear();
  }
  bool HasDecodeFrontier(const TextRunnerState& state) const override {
    return Get(state).session->Position() > 0;
  }

  TextRunnerDescriptor Descriptor() const override {
    const bool drafting = draft_tokens_ != 0;
    return {
        .model_id = "nemotron-3.5-bolt",
        // States of a drafting runner also carry the drafter's context.
        .state_abi =
            std::string(kStateAbi) + (drafting ? ";draft=lilicorr" : ""),
        .max_context = context_,
        .capabilities =
            {
                .supports_vision = vision_ != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = true,
                .incremental_text_is_exact = true,
                .multi_token_decode = drafting,
                // Concurrent drafting requests decode one token each.
                .batched_multi_token_decode = drafting,
                .batched_multi_token_decode_max_width =
                    drafting ? kMaxBatch : 0,
                .prefix_reuse = true,
                .partial_prefill_retention = true,
            },
        .sampling_defaults = SamplingDefaults(),
        .scheduling =
            TextSchedulingProfile{.pass_tokens = kPrefillChunk,
                                  .max_layer_groups = 1,
                                  .max_yield_decode_width = kMaxBatch},
    };
  }

  TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
      capacity = free_bytes;
    return {
        .resident_weights_bytes =
            model_->ResidentBytes() + (vision_ ? vision_->ResidentBytes() : 0),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(context_),
        .temporary_scratch_bytes =
            vision_ ? std::optional(vision_->MaxScratchBytes()) : std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
        // No prompt-state admission charge: every session allocates its
        // K/V rows for the whole context up front, so a prompt commits no
        // memory of its own.
    };
  }

  std::vector<TextExecutionPlan> SupportedPlans() const override {
    // Concurrent requests decode one token each in a shared pass; drafting
    // stays with a request decoding alone.
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= kMaxBatch; ++width)
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    return plans;
  }

  std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    const auto& tokenizer = model_->tokenizer();
    return std::make_shared<const sampling::ConstraintVocabulary>(
        static_cast<std::uint32_t>(tokenizer.VocabSize()),
        [&tokenizer](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              std::string(tokenizer.TokenBytes(id)), tokenizer.IsStop(id)};
        });
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    // The template's Qwen3-Coder XML calls.
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  // llama.cpp forces "\n</think>" for this template, the newline after
  // "<think>" in the generation prompt counting toward the budget.
  ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>"}};
  }
  std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().VocabSize());
  }

  // Raw prompts map control-token spellings to their special tokens, as the
  // chat template's render does and as llama-server's /completion does.
  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    const auto ids = model_->tokenizer().Encode(text, true);
    return {ids.begin(), ids.end()};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<ContentSpan> spans;
    const auto rendered = Render(request, &spans);
    const auto ids = TokenizeChat(model_->tokenizer(), rendered, spans);
    return std::vector<TextRunnerToken>(ids.begin(), ids.end());
  }
  std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    // Images expand into embedding rows, which no text spells.
    if (HasImages(request))
      return std::nullopt;
    return Render(request, nullptr);
  }
  std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    if (HasImages(request))
      return PrepareImagePrompt(request);
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (!prepared)
      return prepared;
    // The stable cache boundary is where the prompt agrees with both ways a
    // conversation goes on. A client that discards the reply appends its next
    // user turn after the history, before the assistant opening. A follow-up
    // re-renders the reply after it, with earlier reasoning as
    // "<think></think>". In both, the template moves the reasoning effort
    // hint to the new last user turn.
    const auto& tokens = prepared->tokens;
    const auto agree = [&](const std::vector<std::uint32_t>& other) {
      return static_cast<std::size_t>(std::mismatch(tokens.begin(),
                                                    tokens.end(), other.begin(),
                                                    other.end())
                                          .first -
                                      tokens.begin());
    };
    std::vector<ContentSpan> spans;
    const auto history = TokenizeChat(model_->tokenizer(),
                                      Render(request, &spans, false), spans);
    auto continuation = request;
    continuation.messages.emplace_back(ext::ChatRole::kAssistant, "");
    continuation.messages.emplace_back(ext::ChatRole::kUser, "");
    spans.clear();
    const auto continued =
        TokenizeChat(model_->tokenizer(), Render(continuation, &spans), spans);
    const std::size_t common = std::min(agree(history), agree(continued));
    // Zero keeps the complete prompt as the stable checkpoint rather than as
    // an optional retry copy.
    prepared->cache_prefix_tokens = common == tokens.size() ? 0 : common;
    return prepared;
  }
  std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    if (!HasImages(request))
      return TextModelRunner::CountPromptTokens(request);
    // Sized as generation prepares it, without resizing the pixels.
    return ExpandRequest(request, false).tokens.size();
  }
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    std::shared_ptr<const VisionPrompt> prompt;
    if (context) {
      const auto* images = dynamic_cast<const VisionContext*>(context.get());
      if (images == nullptr)
        throw std::invalid_argument("invalid Nemotron prompt context");
      prompt = images->prompt;
    }
    auto& s = Get(state);
    if (prompt != s.vision) {
      s.vision = std::move(prompt);
      s.embeddings.clear();
    }
  }
  ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    // Without thinking the prompt ends in "<think></think>".
    return ResolveChatOptions(request.reasoning).enable_thinking
               ? ext::InitialOutputState::kThought
               : ext::InitialOutputState::kText;
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens)
      text += model_->tokenizer().TokenBytes(token);
    return text;
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& s = Get(state);
    if (s.session->Position() != prefix.size())
      throw std::logic_error("Nemotron reused prefix does not match its state");
    // A prompt served whole from the cache ends here; Prefill overrides it
    // when more of the prompt follows.
    s.leading_separator = !prefix.empty() && prefix.back() == thought_close_;
  }

  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t max_input_tokens) const override {
    auto& s = Get(state);
    if (offset != s.session->Position())
      throw std::logic_error("Nemotron prefill offset does not match state");
    if (offset >= prompt.size())
      throw std::logic_error("Nemotron prefill has no remaining input");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, kPrefillChunk});
    if (consumed == 0)
      throw std::invalid_argument("Nemotron prefill budget must be positive");
    const bool decode_ready = offset + consumed == prompt.size();
    const std::vector<std::int32_t> tokens(
        prompt.begin() + static_cast<std::ptrdiff_t>(offset),
        prompt.begin() + static_cast<std::ptrdiff_t>(offset + consumed));
    s.frontier.head.candidates.clear();
    std::string error;
    // The embeddings of the images this step reaches, encoded on first use.
    // Attention is causal, so a step may end inside an image.
    std::vector<InputRows> rows;
    if (s.vision) {
      for (std::size_t i = 0; i < s.vision->images.size(); ++i) {
        const PromptImage& image = s.vision->images[i];
        const std::size_t end = std::size_t{image.offset} + image.rows;
        if (end <= offset || image.offset >= offset + consumed)
          continue;
        auto& embedding = s.embeddings[i];
        if (!embedding) {
          if (!vision_ || !image.patches)
            throw std::logic_error("Nemotron image prompt without its pixels");
          embedding = vision_->Encode(*image.patches, image.grid, &error);
          if (!embedding)
            throw std::runtime_error("Nemotron " + error);
        }
        rows.push_back({image.offset, image.rows, embedding->data()});
      }
    }
    const bool appended = s.session->Append(
        tokens,
        decode_ready ? std::span<float>(s.frontier.logits) : std::span<float>(),
        rows, &error);
    // Images this step finished are not read again unless a restore returns
    // before them, which encodes them anew.
    std::erase_if(s.embeddings, [&](const auto& entry) {
      const PromptImage& image = s.vision->images[entry.first];
      return std::size_t{image.offset} + image.rows <= offset + consumed;
    });
    if (!appended) {
      s.logits_ready = false;
      throw std::runtime_error("Nemotron prefill failed: " + error);
    }
    s.logits_ready = decode_ready;
    if (decode_ready)
      s.leading_separator = prompt.back() == thought_close_;
    return {.consumed_tokens = consumed, .decode_ready = decode_ready};
  }

  TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& s = Get(state);
    if (s.session->Position() >= context_) {
      TextDecodeSelection full;
      full.stop = true;
      return full;
    }
    if (s.session->Position() == 0)
      throw std::logic_error("Nemotron selection before its frontier");
    ReadFrontier(s);
    const auto& logits = s.frontier.logits;
    s.head_filter = HeadFilterFor(sampler.config());
    const auto token = SampleWithHead(sampler, logits, s.frontier.head);
    const bool stop = s.stop_at_eos() && model_->tokenizer().IsStop(token);
    return {.stop = stop,
            .token = token,
            .piece = stop ? std::string{} : Piece(s, token),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = Get(state);
    if (token >= model_->tokenizer().VocabSize())
      throw std::invalid_argument("Nemotron token ID exceeds the vocabulary");
    const std::int32_t one[] = {static_cast<std::int32_t>(token)};
    std::string error;
    auto& frontier = s.frontier;
    frontier.head.candidates.clear();
    const bool ok =
        s.head_filter ? s.session->Append(one, frontier.logits, *s.head_filter,
                                          &frontier.head, &error)
                      : s.session->Append(one, frontier.logits, &error);
    if (!ok) {
      s.logits_ready = false;
      throw std::runtime_error("Nemotron decode failed: " + error);
    }
    s.logits_ready = true;
  }

  /// Requests decoding together take one token each from a shared pass;
  /// drafting needs a request decoding alone.
  std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || decodes.size() > kMaxBatch)
      return TextModelRunner::DecodeBatch(decodes);
    std::vector<TextDecodeStep> steps(decodes.size());
    std::vector<TextRunnerAdvance> advances;
    std::vector<std::size_t> advanced;
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      try {
        auto selection =
            SelectNext(decodes[i].state.get(), decodes[i].sampler.get());
        if (selection.stop) {
          steps[i].stop = true;
          steps[i].terminal_logprobs = std::move(selection.logprobs);
          continue;
        }
        advances.push_back(
            {.state = decodes[i].state, .token = selection.token});
        advanced.push_back(i);
        steps[i].selections.push_back(std::move(selection));
      } catch (...) {
        steps[i].failure = std::current_exception();
      }
    }
    try {
      if (advances.size() == 1)
        Advance(advances.front().state.get(), advances.front().token);
      else if (advances.size() > 1)
        AdvanceBatch(advances);
    } catch (...) {
      for (const std::size_t i : advanced)
        steps[i].failure = std::current_exception();
    }
    const TextExecutionPlan plan{
        .kind = advances.size() > 1 ? TextExecutionPlanKind::kBatched
                                    : TextExecutionPlanKind::kSerial,
        .physical_width = std::max<std::size_t>(advances.size(), 1)};
    for (auto& step : steps)
      step.execution_plan = plan;
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() == 1) {
      Advance(advances.front().state.get(), advances.front().token);
      return;
    }
    if (advances.size() < 2 || advances.size() > kMaxBatch)
      throw std::invalid_argument(
          "Nemotron batched decode requires two to eight sessions");
    std::array<State*, kMaxBatch> states{};
    std::array<Session*, kMaxBatch> sessions{};
    std::array<std::int32_t, kMaxBatch> tokens{};
    std::array<std::span<float>, kMaxBatch> logits{};
    std::array<const LogitHeadFilter*, kMaxBatch> filters{};
    std::array<LogitHead, kMaxBatch> heads{};
    const std::size_t n = advances.size();
    for (std::size_t i = 0; i < n; ++i) {
      auto& s = Get(advances[i].state.get());
      if (advances[i].token >= model_->tokenizer().VocabSize())
        throw std::invalid_argument("Nemotron token ID exceeds the vocabulary");
      states[i] = &s;
      sessions[i] = s.session;
      tokens[i] = static_cast<std::int32_t>(advances[i].token);
      logits[i] = s.frontier.logits;
      filters[i] = s.head_filter ? &*s.head_filter : nullptr;
    }
    std::string error;
    if (!Session::DecodeBatch(
            std::span(sessions).first(n), std::span(tokens).first(n),
            std::span(logits).first(n), std::span(filters).first(n),
            std::span(heads).first(n), &error)) {
      for (std::size_t i = 0; i < n; ++i)
        states[i]->logits_ready = false;
      throw std::runtime_error("Nemotron batched decode failed: " + error);
    }
    for (std::size_t i = 0; i < n; ++i) {
      states[i]->frontier.head = std::move(heads[i]);
      states[i]->logits_ready = true;
    }
  }

  /// A LiLiCorr round: the drafter proposes, the trunk verifies them in one
  /// pass and the tokens the request's sampler agrees with are kept. Only
  /// the RNG and a pending draw from a rejected draft carry over.
  TextDecodeStep DecodeStep(TextRunnerState& state, std::size_t max_tokens,
                            sampling::SamplerState& sampler) const override {
    if (draft_tokens_ == 0 || max_tokens <= 1)
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    auto& s = Get(state);
    auto& session = *s.session;
    if (session.Position() >= context_) {
      TextDecodeStep full;
      full.stop = true;
      return full;
    }
    if (session.Position() == 0)
      throw std::logic_error("Nemotron selection before its frontier");
    ReadFrontier(s);
    const auto budget = static_cast<std::uint32_t>(
        std::min<std::size_t>(max_tokens, std::size_t{draft_tokens_} + 1));
    const bool stop_at_eos = s.stop_at_eos();
    const auto& tokenizer = model_->tokenizer();
    const auto is_stop = [&](std::int32_t token) {
      return stop_at_eos && tokenizer.IsStop(static_cast<TokenId>(token));
    };
    const auto logprobs = sampler.config().logprobs;
    sampling::SamplerState working = sampler;
    SpeculativeRound round;
    std::string error;
    if (!SpeculativeStep(session, working, s.frontier, budget, is_stop,
                         logprobs.has_value(), &round, &error)) {
      s.logits_ready = false;
      throw std::runtime_error("Nemotron draft round failed: " + error);
    }
    sampler.CopyDrawStateFrom(working);
    s.head_filter = HeadFilterFor(sampler.config());
    const std::size_t vocab = s.frontier.logits.size();
    TextDecodeStep step;
    for (std::size_t i = 0; i < round.tokens.size(); ++i) {
      const auto token = static_cast<TextRunnerToken>(round.tokens[i]);
      std::optional<sampling::TokenLogprobs> token_logprobs;
      if (logprobs)
        token_logprobs = sampling::CollectLogprobs(
            std::span<const float>(round.rows.data() + i * vocab, vocab), token,
            logprobs);
      if (round.stop && i + 1 == round.tokens.size()) {
        step.stop = true;
        step.terminal_logprobs = std::move(token_logprobs);
        break;
      }
      step.selections.push_back({.stop = false,
                                 .token = token,
                                 .piece = Piece(s, token),
                                 .logprobs = std::move(token_logprobs)});
    }
    step.draft_rounds = round.drafted != 0 ? 1 : 0;
    step.draft_tokens = round.drafted;
    step.draft_accepted_tokens = round.accepted;
    return step;
  }

  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return Get(state).session->Position();
  }

private:
  /// The text of `token` in the answer. Gufo's host trims the separator after
  /// a "</think>" the model writes, but it has no state for a block the
  /// prompt closed, where the template continues
  /// "<think></think>\n<tool_call>". Without this, a reply's leading newline
  /// would reach clients as text, and a client sending it back would no longer
  /// match the generated tokens. The tokens themselves stay as the model wrote
  /// them.
  std::string Piece(State& s, TextRunnerToken token) const {
    std::string piece(model_->tokenizer().TokenBytes(token));
    if (!s.leading_separator)
      return piece;
    const auto first = piece.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos)
      return {};
    s.leading_separator = false;
    return piece.substr(first);
  }

  /// After a restore or a failure the host copy of the frontier is stale.
  void ReadFrontier(State& s) const {
    if (s.logits_ready)
      return;
    std::string error;
    if (!s.session->ReadLogits(s.frontier.logits, &error))
      throw std::runtime_error("Nemotron frontier read failed: " + error);
    s.frontier.head.candidates.clear();
    s.logits_ready = true;
  }

  std::string Render(const ChatRequest& request,
                     std::vector<ContentSpan>* spans,
                     bool generation_prompt = true) const {
    std::vector<nemotron::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      if (!message.images.empty() && !vision_)
        throw std::invalid_argument("model does not support image input");
      nemotron::ChatMessage converted;
      for (const auto& image : message.images)
        converted.image_offsets.push_back(image.offset);
      switch (message.role) {
        case ext::ChatRole::kSystem:
        case ext::ChatRole::kDeveloper:
          converted.role = nemotron::ChatRole::kSystem;
          break;
        case ext::ChatRole::kUser:
          converted.role = nemotron::ChatRole::kUser;
          break;
        case ext::ChatRole::kAssistant:
          converted.role = nemotron::ChatRole::kAssistant;
          break;
        case ext::ChatRole::kTool:
          converted.role = nemotron::ChatRole::kTool;
          break;
      }
      converted.content = message.content + message.framing_suffix;
      converted.reasoning = message.thought;
      for (const auto& call : message.tool_calls) {
        nemotron::ChatToolCall out{call.name, {}};
        for (const auto& argument : call.arguments)
          out.arguments.push_back(
              {argument.name, argument.value, argument.is_string});
        converted.tool_calls.push_back(std::move(out));
      }
      messages.push_back(std::move(converted));
    }
    std::vector<nemotron::ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      for (const auto& tool : request.tools)
        tools.push_back({tool.name, tool.description, tool.parameters_json,
                         tool.definition_json});
    }
    auto options = ResolveChatOptions(request.reasoning);
    options.add_generation_prompt = generation_prompt;
    return RenderChat(messages, tools, options, spans);
  }

  /// The rendered prompt with each image placeholder expanded into its span.
  VisionPrompt ExpandRequest(const ChatRequest& request,
                             bool preprocess) const {
    if (!vision_)
      throw std::invalid_argument("model does not support image input");
    std::vector<ContentSpan> spans;
    const auto rendered =
        TokenizeChat(model_->tokenizer(), Render(request, &spans), spans);
    std::vector<VisionInput> images;
    for (const auto& message : request.messages) {
      for (const auto& image : message.images)
        images.push_back(
            {image.bytes, image.detail == ext::ChatMessage::ImageDetail::kLow});
    }
    return ExpandImages(vision_->config(), framing_, rendered, images,
                        vision_identity_, preprocess);
  }

  std::optional<TextPreparedPrompt> PrepareImagePrompt(
      const ChatRequest& request) const {
    auto prompt =
        std::make_shared<const VisionPrompt>(ExpandRequest(request, true));
    // The stable boundary as for text prompts (PreparePrompt), found among
    // the rendered tokens and moved past the images expanded before it.
    const auto render = [&](const ChatRequest& r, bool generation_prompt) {
      std::vector<ContentSpan> spans;
      return TokenizeChat(model_->tokenizer(),
                          Render(r, &spans, generation_prompt), spans);
    };
    const auto rendered = render(request, true);
    const auto agree = [&](const std::vector<std::uint32_t>& other) {
      return static_cast<std::size_t>(std::mismatch(rendered.begin(),
                                                    rendered.end(),
                                                    other.begin(), other.end())
                                          .first -
                                      rendered.begin());
    };
    auto continuation = request;
    continuation.messages.emplace_back(ext::ChatRole::kAssistant, "");
    continuation.messages.emplace_back(ext::ChatRole::kUser, "");
    const std::size_t common = std::min(agree(render(request, false)),
                                        agree(render(continuation, true)));
    auto context = std::make_shared<VisionContext>();
    context->cache_identity = prompt->cache_identity;
    // A prefix ending at or before an image holds only the images before it.
    for (std::size_t i = 0; i < prompt->images.size(); ++i)
      context->cache_prefixes.push_back(
          {prompt->images[i].offset, prompt->prefix_identities[i]});
    context->prompt = prompt;
    TextPreparedPrompt prepared;
    prepared.tokens.assign(prompt->tokens.begin(), prompt->tokens.end());
    prepared.context = std::move(context);
    prepared.cache_prefix_tokens =
        common == rendered.size() ? 0 : prompt->Expanded(common);
    return prepared;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<VisionEncoder> vision_;
  ImageFraming framing_;
  std::string vision_identity_;
  std::uint32_t context_;
  std::uint32_t draft_tokens_;
  /// "</think>", which ends the prompt when it closed reasoning itself.
  TextRunnerToken thought_close_{0};
  cache::Identity component_identity_;
};

std::string Fingerprint(const std::string& path, std::string* error) {
  try {
    const auto reader = core::GgufReader::OpenFile(
        path, error, {.allow_unsized_tensors = true});
    if (!reader)
      return {};
    return core::GgufIdentityHex(*reader);
  } catch (const std::exception& e) {
    if (error != nullptr)
      *error = std::string("Nemotron fingerprinting failed: ") + e.what();
    return {};
  }
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateNemotronRunner(
    std::shared_ptr<Model> model, std::uint32_t context,
    std::uint32_t draft_tokens, std::string artifact_fingerprint,
    std::string draft_fingerprint, std::shared_ptr<VisionEncoder> vision) {
  return std::make_shared<Runner>(
      std::move(model), context, draft_tokens, std::move(artifact_fingerprint),
      std::move(draft_fingerprint), std::move(vision));
}

std::shared_ptr<ext::TextModelRunner> CreateNemotronTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return std::shared_ptr<ext::TextModelRunner>{};
  };
  const auto backend = options.speculative.backend;
  if (backend != ext::TextSpeculativeBackend::kDisabled &&
      backend != ext::TextSpeculativeBackend::kDFlash)
    return fail(
        "Nemotron drafts with NVIDIA's LiLiCorr drafter only: "
        "--speculative dflash2 --dflash-model DRAFTER.gguf");
  // The native context unless the command asks for less.
  ModelOptions model_options;
  model_options.max_context = options.max_context;
  auto model = Model::Load(options.model_path, model_options, error);
  if (!model)
    return nullptr;
  const std::uint32_t context = model->MaxContext();
  if (context < 2)
    return fail("Nemotron needs a context of at least two tokens");
  if (backend == ext::TextSpeculativeBackend::kDFlash &&
      !model->LoadDrafter(options.speculative.draft_model_path, error))
    return nullptr;
  // NVIDIA's vision projector, with its matrix-product algorithms chosen
  // now rather than during the first images.
  std::shared_ptr<VisionEncoder> vision;
  if (!options.vision_model_path.empty()) {
    vision = VisionEncoder::Load(options.vision_model_path, model->config(), {},
                                 error);
    if (!vision || !vision->Prepare(error))
      return nullptr;
  }
  std::string fingerprint;
  std::string draft_fingerprint;
  if (!options.disk_cache.directory.empty()) {
    fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (fingerprint.empty()) {
      fingerprint = options.reader ? core::GgufIdentityHex(*options.reader)
                                   : Fingerprint(options.model_path, error);
      if (fingerprint.empty())
        return nullptr;
    }
    if (model->drafter() != nullptr) {
      draft_fingerprint = options.disk_cache.draft_model_artifact_fingerprint;
      if (draft_fingerprint.empty())
        draft_fingerprint =
            Fingerprint(options.speculative.draft_model_path, error);
      if (draft_fingerprint.empty())
        return nullptr;
    }
  }
  try {
    return CreateNemotronRunner(
        std::move(model), context, options.speculative.max_draft_tokens,
        std::move(fingerprint), std::move(draft_fingerprint),
        std::move(vision));
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

}  // namespace gufo::models::nemotron
