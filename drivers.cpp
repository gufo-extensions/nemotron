// The prompt, chat and bench commands over the engine's sessions.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <gufo/ext/reasoning.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "drafter.hpp"
#include "engine.hpp"
#include "speculative.hpp"

namespace gufo::models::nemotron {
namespace {
using Clock = std::chrono::steady_clock;

/// Terminal sessions hold the prompt plus the requested output.
constexpr std::uint32_t kCliContext = 32768;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

ReasoningEffort ParseEffort(std::string_view value) {
  if (value == "minimal")
    return ReasoningEffort::kMinimal;
  if (value == "low")
    return ReasoningEffort::kLow;
  if (value == "medium")
    return ReasoningEffort::kMedium;
  if (value == "high")
    return ReasoningEffort::kHigh;
  if (value == "xhigh")
    return ReasoningEffort::kXHigh;
  if (value == "max")
    return ReasoningEffort::kMax;
  throw std::invalid_argument("unknown reasoning effort: " +
                              std::string(value));
}

ReasoningOptions Reasoning(const ext::TextPromptOptions& opt) {
  ReasoningOptions reasoning;
  if (const auto thinking = opt.thinking_mode())
    reasoning.enabled = *thinking;
  if (opt.reasoning_effort != "auto")
    reasoning.effort = ParseEffort(opt.reasoning_effort);
  if (opt.preserve_thinking == "on")
    reasoning.preserve_thinking = true;
  else if (opt.preserve_thinking == "off")
    reasoning.preserve_thinking = false;
  return reasoning;
}

/// The trunk and, for `--speculative dflash2`, NVIDIA's LiLiCorr drafter.
std::shared_ptr<Model> Load(const std::string& model_path,
                            std::string_view backend,
                            const std::string& drafter_path,
                            std::uint32_t context) {
  if (!backend.empty() && backend != "dflash2")
    throw std::invalid_argument(
        "Nemotron drafts with NVIDIA's LiLiCorr drafter only: "
        "--speculative dflash2 --dflash-model DRAFTER.gguf");
  if (backend == "dflash2" && drafter_path.empty())
    throw std::invalid_argument("--speculative dflash2 needs --dflash-model");
  std::string error;
  auto model =
      Model::Load(model_path, ModelOptions{.max_context = context}, &error);
  if (model && backend == "dflash2" &&
      !model->LoadDrafter(drafter_path, &error))
    model.reset();
  if (!model)
    throw std::runtime_error("load failed: " + error);
  return model;
}

std::vector<std::int32_t> PromptTokens(const Model& model,
                                       const ext::TextPromptOptions& opt,
                                       std::span<const ChatMessage> history) {
  if (!opt.use_chat_template) {
    const auto ids = model.tokenizer().Encode(opt.prompt_text, true);
    return {ids.begin(), ids.end()};
  }
  std::vector<ContentSpan> spans;
  const std::string text =
      RenderChat(history, {}, ResolveChatOptions(Reasoning(opt)), &spans);
  const auto ids = TokenizeChat(model.tokenizer(), text, spans);
  return {ids.begin(), ids.end()};
}

/// Extends the session with `prompt` when it continues the executed tokens,
/// otherwise restarts it, then samples until a stop token or max_tokens.
/// With a drafter each round emits up to draft_tokens + 1 tokens.
void Generate(const ext::TextPromptOptions& opt, const Model& model,
              Session& session, std::vector<std::int32_t>& executed,
              std::span<const std::int32_t> prompt,
              std::string* reply = nullptr) {
  const auto& tokenizer = model.tokenizer();
  if (prompt.empty())
    throw std::invalid_argument("the prompt produced no tokens");
  if (prompt.size() + opt.max_tokens > session.MaxContext())
    throw std::invalid_argument("prompt and output exceed the " +
                                std::to_string(session.MaxContext()) +
                                "-token terminal context");
  std::string error;
  const bool continues =
      executed.size() < prompt.size() &&
      std::equal(executed.begin(), executed.end(), prompt.begin());
  if (!continues) {
    if (!session.Reset(&error))
      throw std::runtime_error("session reset failed: " + error);
    executed.clear();
  }
  Frontier frontier;
  frontier.logits.resize(model.config().vocab_size);
  const auto prefill_start = Clock::now();
  const auto suffix = prompt.subspan(executed.size());
  if (!session.Append(suffix, frontier.logits, &error))
    throw std::runtime_error("prefill failed: " + error);
  executed.assign(prompt.begin(), prompt.end());
  const double prefill_seconds = Seconds(prefill_start);
  if (opt.verbose)
    std::cout << "[Engine]: Nemotron 3.5 Bolt ROCm (gfx1151)\n"
              << "Prompt tokens: " << prompt.size() << " (" << suffix.size()
              << " prefilled in " << prefill_seconds << " s)\n"
              << "--- Generation Output ---\n";

  const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(opt.sampling, history);
  const auto generation_start = Clock::now();
  const std::uint32_t draft_tokens =
      model.drafter() == nullptr
          ? 0
          : std::min(opt.draft_tokens, model.drafter()->config().Slots());
  std::size_t generated = 0;
  std::size_t rounds = 0, drafted = 0, accepted = 0;
  const auto is_stop = [&tokenizer](std::int32_t token) {
    return tokenizer.IsStop(static_cast<TokenId>(token));
  };
  while (generated < opt.max_tokens) {
    const auto budget = static_cast<std::uint32_t>(std::min<std::size_t>(
        opt.max_tokens - generated, std::size_t{draft_tokens} + 1));
    SpeculativeRound round;
    if (!SpeculativeStep(session, sampler, frontier, budget, is_stop, false,
                         &round, &error))
      throw std::runtime_error("decode failed: " + error);
    rounds += round.drafted != 0;
    drafted += round.drafted;
    accepted += round.accepted;
    for (std::size_t i = 0; i < round.tokens.size(); ++i) {
      const std::int32_t token = round.tokens[i];
      if (round.stop && i + 1 == round.tokens.size())
        break;
      const auto piece = tokenizer.TokenBytes(static_cast<TokenId>(token));
      if (reply != nullptr)
        reply->append(piece);
      std::cout << piece << std::flush;
      executed.push_back(token);
      ++generated;
    }
    if (round.stop)
      break;
  }
  std::cout << '\n';
  if (opt.verbose && rounds > 0)
    std::cout << "Draft rounds: " << rounds << ", drafted " << drafted
              << ", accepted " << accepted << " ("
              << static_cast<double>(accepted) / static_cast<double>(rounds)
              << " per round)\n";
  if (opt.verbose && generated > 0) {
    const double seconds = Seconds(generation_start);
    std::cout << "Generated " << generated << " tokens in " << seconds << " s ("
              << static_cast<double>(generated) / seconds << " tok/s)\n";
  }
}

int RunPrompt(const ext::TextPromptOptions& opt, bool interactive,
              Clock::time_point load_start) {
  try {
    if (opt.force_cpu)
      throw std::invalid_argument("Nemotron 3.5 Bolt runs only on ROCm");
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty())
      throw std::invalid_argument("Nemotron 3.5 Bolt takes no image input");
    if (interactive && !opt.use_chat_template)
      throw std::invalid_argument(
          "interactive chat needs chat framing; use prompt --raw for raw text");
    const auto load =
        load_start == Clock::time_point{} ? Clock::now() : load_start;
    auto model = Load(opt.model_path, opt.speculative_backend,
                      opt.dflash_model_path, kCliContext);
    std::cerr << "[ModelLoad]: " << Seconds(load) << " s\n";
    std::string error;
    auto session = model->CreateSession(kCliContext, &error);
    if (!session)
      throw std::runtime_error("session creation failed: " + error);
    std::vector<ChatMessage> history;
    if (!opt.system_prompt.empty())
      history.push_back({ChatRole::kSystem, opt.system_prompt, {}, {}});
    std::vector<std::int32_t> executed;
    if (!interactive) {
      history.push_back({ChatRole::kUser, opt.prompt_text, {}, {}});
      Generate(opt, *model, *session, executed,
               PromptTokens(*model, opt, history));
      return 0;
    }
    const bool thinking = ResolveChatOptions(Reasoning(opt)).enable_thinking;
    std::cout << "=== Gufo Interactive Chat (Nemotron 3.5 Bolt) ===\n"
              << "Type 'exit' or Ctrl+D to quit.\n\n";
    for (std::string input;;) {
      std::cout << ">>> User: " << std::flush;
      if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
        break;
      if (input.empty())
        continue;
      history.push_back({ChatRole::kUser, input, {}, {}});
      std::cout << "<<< Assistant: ";
      std::string reply;
      Generate(opt, *model, *session, executed,
               PromptTokens(*model, opt, history), &reply);
      ChatMessage response;
      response.role = ChatRole::kAssistant;
      constexpr std::string_view end = "</think>";
      const auto boundary = reply.find(end);
      if (thinking && boundary != std::string::npos) {
        response.reasoning = reply.substr(0, boundary);
        response.content = reply.substr(boundary + end.size());
      } else {
        response.content = std::move(reply);
      }
      history.push_back(std::move(response));
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Nemotron: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace

int RunNemotronPrompt(const ext::TextPromptOptions& opt,
                      const core::GgufReader&,
                      std::chrono::steady_clock::time_point load_start) {
  return RunPrompt(opt, false, load_start);
}

int RunNemotronChat(const ext::TextPromptOptions& opt, const core::GgufReader&,
                    std::chrono::steady_clock::time_point load_start) {
  return RunPrompt(opt, true, load_start);
}

/// Single-session prefill (pp) and decode (tg) rates; with the drafter, tg
/// runs draft-and-verify rounds. Concurrent and HTTP measurements go
/// through `gufo serve`.
int RunNemotronBench(const ext::TextBenchOptions& opt,
                     const std::shared_ptr<const core::GgufReader>&,
                     std::chrono::steady_clock::time_point) {
  try {
    if (opt.concurrency != std::vector<std::size_t>{1} || opt.enable_yarn ||
        opt.validate_prefill_tokens != 0)
      throw std::invalid_argument(
          "Nemotron bench measures one session; use gufo serve for "
          "concurrency");
    const auto max = [](const std::vector<std::size_t>& v) {
      return v.empty() ? std::size_t{0} : *std::ranges::max_element(v);
    };
    // Room for the warm-up as well: a full prefill pass and one round.
    const std::size_t context = std::max(
        max(opt.n_depths) + std::max(max(opt.n_prompts), max(opt.n_gens)) + 1,
        kPrefillChunk + kMaxBatch + 2);
    auto model =
        Load(opt.model_path, opt.speculative_backend, opt.dflash_model_path,
             static_cast<std::uint32_t>(context));
    std::string error;
    auto session =
        model->CreateSession(static_cast<std::uint32_t>(context), &error);
    if (!session)
      throw std::runtime_error("session creation failed: " + error);
    const auto& tokenizer = model->tokenizer();
    const auto seed_ids = tokenizer.Encode(
        "The quick brown fox jumps over the lazy dog. ", false);
    const auto fill = [&](std::size_t n) {
      std::vector<std::int32_t> ids(n);
      for (std::size_t i = 0; i < n; ++i)
        ids[i] = static_cast<std::int32_t>(seed_ids[i % seed_ids.size()]);
      return ids;
    };
    const auto require = [&](bool ok) {
      if (!ok)
        throw std::runtime_error(error);
    };
    const std::uint32_t draft_tokens =
        model->drafter() == nullptr
            ? 0
            : std::min(opt.draft_tokens, model->drafter()->config().Slots());
    const std::string name = model->drafter() == nullptr
                                 ? "Nemotron-3.5-Bolt"
                                 : "Nemotron-3.5-Bolt+LiLiCorr";
    Frontier frontier;
    frontier.logits.resize(model->config().vocab_size);
    // Warm the prefill, decode and verification routes.
    require(session->Append(fill(kPrefillChunk + 1), frontier.logits, &error));
    frontier.head.candidates.clear();
    const auto never = [](std::int32_t) { return false; };
    {
      sampling::SamplerState sampler(opt.sampling);
      SpeculativeRound round;
      require(SpeculativeStep(*session, sampler, frontier, draft_tokens + 1,
                              never, false, &round, &error));
    }
    std::cout << "| model | test | depth | repetition | tokens/s | seconds "
                 "|\n| :-- | :-- | --: | --: | --: | --: |\n";
    for (const std::size_t depth : opt.n_depths) {
      const auto prepare = [&] {
        require(session->Reset(&error));
        if (depth != 0)
          require(session->Append(fill(depth), {}, &error));
      };
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        for (const std::size_t n : opt.n_prompts) {
          if (n == 0)
            continue;
          prepare();
          const auto prompt = fill(n);
          const auto start = Clock::now();
          require(session->Append(prompt, {}, &error));
          const double dt = Seconds(start);
          std::cout << "| " << name << " | pp" << n << " | " << depth << " | "
                    << r << " | " << static_cast<double>(n) / dt << " | " << dt
                    << " |\n";
        }
        for (const std::size_t n : opt.n_gens) {
          if (n == 0)
            continue;
          prepare();
          require(session->Append(fill(1), frontier.logits, &error));
          frontier.head.candidates.clear();
          sampling::SamplerState sampler(opt.sampling);
          std::size_t generated = 0;
          const auto start = Clock::now();
          while (generated < n) {
            SpeculativeRound round;
            const auto budget = static_cast<std::uint32_t>(
                std::min<std::size_t>(n - generated, draft_tokens + 1));
            require(SpeculativeStep(*session, sampler, frontier, budget, never,
                                    false, &round, &error));
            generated += round.tokens.size();
          }
          const double dt = Seconds(start);
          std::cout << "| " << name << " | tg" << n << " | " << depth << " | "
                    << r << " | " << static_cast<double>(n) / dt << " | " << dt
                    << " |\n";
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Nemotron: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::models::nemotron
