// Runs the ROCm runtime over a prompt: dumps float32 logits for every prompt
// position (the layout compare_logits.py reads), optionally continues
// greedily, and reports prefill and decode rates.
//
//   nemotron_runtime_probe --model MODEL.gguf
//       (--tokens 1,2,3 | --text TEXT | --text-file PATH) [--dump logits.bin]
//       [--generate N]
//       [--context N] [--step] [--depth N]
//       [--drafter DRAFT.gguf [--temperature T] [--draft-stats OUT.tsv]]
//       [--verify-sweep R]
// --step appends the prompt one token at a time, so the dumped rows come
// from the decode path instead of one prefill. --depth N prefills the first
// N prompt tokens as a cached prefix, timed on their own, before the timed
// prefill of the rest (pp at depth N).
// With a drafter, generation runs draft-and-verify rounds and reports the
// acceptance; temperature 0 (the default) decodes greedily either way.
// --draft-stats (greedy, with a drafter) runs the rounds itself and writes
// one line per drafted slot: round, slot, whether the trunk kept it, the
// lattice's probability of the walked transition and its margin over the
// runner-up, the drafter head's log-probability of the drafted token and
// whether it is the head's top candidate.
// --verify-sweep R (with --generate) replays the generated tokens in fresh
// sessions and times, for blocks of n = 1 to 8 tokens, R rounds of Verify
// followed by Commit of one token and of all n, R one-token Appends and,
// with a drafter, R drafts: the cost of a round by its length.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gufo/ext/sampling.hpp>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "drafter.hpp"
#include "engine.hpp"
#include "speculative.hpp"

namespace nb = gufo::models::nemotron;

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& text) {
  std::vector<std::int32_t> out;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find(',', start);
    out.push_back(std::atoi(text.substr(start, end - start).c_str()));
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return out;
}

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

/// Greedy draft-and-verify rounds as SpeculativeStep runs them, recording
/// per drafted slot what --draft-stats describes. `row` is the frontier.
bool DraftStats(nb::Model& model, nb::Session& session, std::vector<float>& row,
                std::size_t generate, const std::string& path,
                std::vector<std::int32_t>* generated, std::size_t* rounds,
                std::size_t* drafted, std::size_t* accepted,
                std::string* error) {
  std::ofstream out(path);
  out << "round\tslot\tkept\tlattice_p\tlattice_margin\thead_logp\thead_top"
         "\n";
  const std::size_t vocab = model.config().vocab_size;
  const std::uint32_t slots = model.drafter()->config().Slots();
  std::vector<float> head(static_cast<std::size_t>(slots) * vocab);
  std::vector<float> rows((slots + 1) * vocab);
  const auto argmax = [](const float* r, std::size_t n) {
    return static_cast<std::int32_t>(std::max_element(r, r + n) - r);
  };
  const auto is_stop = [&](std::int32_t t) {
    return model.tokenizer().IsStop(static_cast<nb::TokenId>(t));
  };
  while (generated->size() < generate) {
    const std::int32_t first = argmax(row.data(), vocab);
    generated->push_back(first);
    if (is_stop(first) || generated->size() >= generate)
      break;
    nb::DraftLattice lattice;
    if (!session.draft()->Draft(first, session.Position(), &lattice, head,
                                error))
      return false;
    // The walk, with each transition's softmax probability and margin.
    std::vector<std::int32_t> drafts;
    std::vector<float> p(slots), margin(slots);
    std::uint32_t prev = 0;
    const std::uint32_t K = lattice.top_k;
    for (std::uint32_t s = 0; s < lattice.slots; ++s) {
      const float* sc =
          s == 0 ? lattice.scores.data()
                 : lattice.scores.data() + K + ((s - 1) * K + prev) * K;
      const auto best =
          static_cast<std::uint32_t>(std::max_element(sc, sc + K) - sc);
      float second = -1e30F;
      double sum = 0.0;
      for (std::uint32_t j = 0; j < K; ++j) {
        sum += std::exp(static_cast<double>(sc[j] - sc[best]));
        if (j != best)
          second = std::max(second, sc[j]);
      }
      p[s] = static_cast<float>(1.0 / sum);
      margin[s] = sc[best] - second;
      prev = best;
      drafts.push_back(lattice.candidates[s * K + best]);
    }
    const std::size_t room = generate - generated->size();
    if (drafts.size() > room)
      drafts.resize(room);
    std::vector<std::int32_t> block{first};
    block.insert(block.end(), drafts.begin(), drafts.end());
    const std::span<float> out_rows(rows.data(), block.size() * vocab);
    if (!session.Verify(block, out_rows, error))
      return false;
    std::uint32_t kept = 1;
    bool stop = false;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
      if (argmax(rows.data() + i * vocab, vocab) != drafts[i])
        break;
      generated->push_back(drafts[i]);
      ++kept;
      if (is_stop(drafts[i])) {
        stop = true;
        break;
      }
    }
    const std::uint32_t emitted_drafts = kept - 1;
    for (std::size_t sl = 0; sl < drafts.size(); ++sl) {
      const float* h = head.data() + sl * vocab;
      const float top = *std::max_element(h, h + vocab);
      double lse = 0.0;
      for (std::size_t j = 0; j < vocab; ++j)
        lse += std::exp(static_cast<double>(h[j] - top));
      const double logp =
          static_cast<double>(h[drafts[sl]] - top) - std::log(lse);
      out << *rounds << '\t' << sl << '\t' << (sl < emitted_drafts ? 1 : 0)
          << '\t' << p[sl] << '\t' << margin[sl] << '\t' << logp << '\t'
          << (argmax(h, vocab) == drafts[sl] ? 1 : 0) << '\n';
    }
    ++*rounds;
    *drafted += drafts.size();
    *accepted += emitted_drafts;
    // A kept stop token is reported but never run, as in SpeculativeStep.
    if (!session.Commit(stop ? kept - 1 : kept, error))
      return false;
    if (stop)
      break;
    std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>((kept - 1) * vocab),
                vocab, row.begin());
  }
  return true;
}

/// Milliseconds per call of `step` over `rounds` calls.
template<class Step>
bool TimeRounds(std::size_t rounds, double* ms, const Step& step) {
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < rounds; ++i) {
    if (!step(i))
      return false;
  }
  *ms = 1000.0 * Seconds(start) / static_cast<double>(rounds);
  return true;
}

/// The --verify-sweep table: each row starts from a fresh session over the
/// prompt and walks the generated tokens, so verified blocks are the
/// model's own continuation.
bool Sweep(nb::Model& model, std::uint32_t context,
           const std::vector<std::int32_t>& prompt,
           const std::vector<std::int32_t>& generated, std::size_t rounds,
           std::string* error) {
  const std::size_t vocab = model.config().vocab_size;
  std::vector<float> rows(8 * vocab);
  std::vector<float> last(vocab);
  const auto fresh = [&]() {
    auto session = model.CreateSession(context, error);
    if (session && !session->Append(prompt, last, error))
      session.reset();
    return session;
  };
  const auto at = [&](std::size_t i) { return generated.data() + i; };
  std::printf("verify sweep, %zu rounds per row (ms per round)\n", rounds);
  {
    auto session = fresh();
    if (!session)
      return false;
    double ms = 0.0;
    if (rounds > generated.size() ||
        !TimeRounds(rounds, &ms, [&](std::size_t i) {
          return session->Append(std::span<const std::int32_t>(at(i), 1), last,
                                 error);
        }))
      return false;
    std::printf("append 1: %.2f\n", ms);
    if (nb::DraftContext* draft = session->draft()) {
      nb::DraftLattice lattice;
      const std::uint32_t pos = session->Position();
      if (!TimeRounds(rounds, &ms, [&](std::size_t) {
            return draft->Draft(generated[rounds], pos, &lattice, {}, error);
          }))
        return false;
      std::printf("draft: %.2f\n", ms);
    }
  }
  for (std::size_t n = 1; n <= 8; ++n) {
    double one = 0.0;
    double all = 0.0;
    auto a = fresh();
    auto b = fresh();
    if (!a || !b || rounds + n > generated.size() ||
        rounds * n > generated.size())
      return false;
    const std::span<float> out(rows.data(), n * vocab);
    if (!TimeRounds(rounds, &one,
                    [&](std::size_t i) {
                      return a->Verify(std::span<const std::int32_t>(at(i), n),
                                       out, error) &&
                             a->Commit(1, error);
                    }) ||
        !TimeRounds(rounds, &all, [&](std::size_t i) {
          return b->Verify(std::span<const std::int32_t>(at(i * n), n), out,
                           error) &&
                 b->Commit(static_cast<std::uint32_t>(n), error);
        }))
      return false;
    std::printf("verify %zu: commit 1 %.2f, commit %zu %.2f\n", n, one, n, all);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_arg, text, dump, drafter_path;
  float temperature = 0.0F;
  std::uint32_t generate = 0;
  std::uint32_t context = 8192;
  bool step = false;
  std::size_t depth = 0;
  std::size_t sweep = 0;
  std::string draft_stats;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model")
      model_path = value();
    else if (arg == "--tokens")
      tokens_arg = value();
    else if (arg == "--text")
      text = value();
    else if (arg == "--text-file") {
      std::ifstream in(value(), std::ios::binary);
      text.assign(std::istreambuf_iterator<char>(in), {});
    } else if (arg == "--dump")
      dump = value();
    else if (arg == "--drafter")
      drafter_path = value();
    else if (arg == "--temperature")
      temperature = std::stof(value());
    else if (arg == "--generate")
      generate = static_cast<std::uint32_t>(std::stoul(value()));
    else if (arg == "--context")
      context = static_cast<std::uint32_t>(std::stoul(value()));
    else if (arg == "--step")
      step = true;
    else if (arg == "--draft-stats")
      draft_stats = value();
    else if (arg == "--verify-sweep")
      sweep = static_cast<std::size_t>(std::atoll(value().c_str()));
    else if (arg == "--depth")
      depth = std::stoul(value());
    else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty() || (tokens_arg.empty() && text.empty())) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL.gguf (--tokens 1,2,3 | --text TEXT | "
                 "--text-file PATH) [--dump logits.bin] [--generate N] "
                 "[--context N] [--step] [--depth N]\n",
                 argv[0]);
    return 2;
  }

  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  nb::ModelOptions options;
  options.max_context = context;
  auto model = nb::Model::Load(model_path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf("loaded %.2f GB in %.1f s\n", model->ResidentBytes() / 1e9,
              Seconds(load_start));
  if (!drafter_path.empty() && !model->LoadDrafter(drafter_path, &error)) {
    std::fprintf(stderr, "drafter: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(context, &error);
  if (!session) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  if (!tokens_arg.empty()) {
    tokens = ParseTokens(tokens_arg);
  } else {
    for (auto id : model->tokenizer().Encode(text, true))
      tokens.push_back(static_cast<std::int32_t>(id));
  }
  const std::size_t vocab = model->config().vocab_size;
  if (depth > 0) {
    if (depth >= tokens.size()) {
      std::fprintf(stderr, "--depth must leave prompt tokens to prefill\n");
      return 2;
    }
    std::vector<float> last(vocab);
    const auto depth_start = std::chrono::steady_clock::now();
    if (!session->Append(std::span<const std::int32_t>(tokens.data(), depth),
                         last, &error)) {
      std::fprintf(stderr, "depth prefill: %s\n", error.c_str());
      return 1;
    }
    const double seconds = Seconds(depth_start);
    std::printf("depth prefill %zu tokens in %.3f s (%.1f tokens/s)\n", depth,
                seconds, static_cast<double>(depth) / seconds);
    tokens.erase(tokens.begin(),
                 tokens.begin() + static_cast<std::ptrdiff_t>(depth));
  }
  std::vector<float> logits(dump.empty() ? vocab : tokens.size() * vocab);
  const auto prefill_start = std::chrono::steady_clock::now();
  if (step) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const std::size_t row = dump.empty() ? 0 : i;
      if (!session->Append(std::span<const std::int32_t>(&tokens[i], 1),
                           std::span<float>(logits.data() + row * vocab, vocab),
                           &error)) {
        std::fprintf(stderr, "prefill: %s\n", error.c_str());
        return 1;
      }
    }
  } else if (!session->Append(tokens, logits, &error)) {
    std::fprintf(stderr, "prefill: %s\n", error.c_str());
    return 1;
  }
  const double prefill = Seconds(prefill_start);
  std::printf("prefill %zu tokens in %.3f s (%.1f tokens/s)\n", tokens.size(),
              prefill, static_cast<double>(tokens.size()) / prefill);
  if (!dump.empty()) {
    std::ofstream out(dump, std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
  }

  if (generate > 0) {
    nb::Frontier frontier;
    frontier.logits.assign(logits.end() - static_cast<std::ptrdiff_t>(vocab),
                           logits.end());
    std::vector<float>& row = frontier.logits;
    std::vector<std::int32_t> generated;
    const auto decode_start = std::chrono::steady_clock::now();
    gufo::sampling::SamplingConfig sampling;
    sampling.temperature = temperature;
    sampling.top_p = temperature > 0 ? 0.95F : 1.0F;
    sampling.seed = 1;
    gufo::sampling::SamplerState sampler(sampling);
    std::size_t rounds = 0, drafted = 0, accepted = 0;
    nb::DraftHistory history;
    const auto is_stop = [&](std::int32_t t) {
      return model->tokenizer().IsStop(static_cast<nb::TokenId>(t));
    };
    if (!draft_stats.empty()) {
      if (model->drafter() == nullptr || temperature > 0.0F) {
        std::fprintf(stderr, "--draft-stats needs the drafter, greedy\n");
        return 2;
      }
      if (!DraftStats(*model, *session, row, generate, draft_stats, &generated,
                      &rounds, &drafted, &accepted, &error)) {
        std::fprintf(stderr, "draft stats: %s\n", error.c_str());
        return 1;
      }
    }
    while (draft_stats.empty() && model->drafter() != nullptr &&
           generated.size() < generate) {
      nb::SpeculativeRound round;
      if (!nb::SpeculativeStep(
              *session, sampler, frontier, history,
              static_cast<std::uint32_t>(generate - generated.size()), is_stop,
              false, &round, &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
      generated.insert(generated.end(), round.tokens.begin(),
                       round.tokens.end());
      ++rounds;
      drafted += round.drafted;
      accepted += round.accepted;
      if (round.stop)
        break;
    }
    if (rounds != 0)
      std::printf("rounds %zu, drafted %zu, accepted %zu (%.2f per round)\n",
                  rounds, drafted, accepted,
                  static_cast<double>(accepted) / static_cast<double>(rounds));
    for (std::uint32_t i = 0; model->drafter() == nullptr && i < generate;
         ++i) {
      const auto next = static_cast<std::int32_t>(
          std::max_element(row.begin(), row.end()) - row.begin());
      generated.push_back(next);
      if (model->tokenizer().IsStop(static_cast<nb::TokenId>(next)))
        break;
      const std::int32_t one[] = {next};
      if (!session->Append(one, row, &error)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return 1;
      }
    }
    const double decode = Seconds(decode_start);
    std::vector<nb::TokenId> ids(generated.begin(), generated.end());
    std::printf("generated %zu tokens in %.3f s (%.2f tokens/s)\n",
                generated.size(), decode,
                static_cast<double>(generated.size()) / decode);
    std::printf("ids:");
    for (auto id : generated)
      std::printf(" %d", id);
    std::printf("\ntext: %s\n", model->tokenizer().Decode(ids).c_str());
    if (sweep > 0 &&
        !Sweep(*model, context, tokens, generated, sweep, &error)) {
      std::fprintf(stderr, "sweep: %s\n", error.c_str());
      return 1;
    }
  }
  return 0;
}
