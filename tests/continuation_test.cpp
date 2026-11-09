// Continuation adapter on the released files: a checkpoint captured in odd
// pieces and restored in reverse order into a slot holding another, longer
// state reproduces the source session byte for byte (attention rows, Mamba
// states, drafter rings) and its logits and greedy decoding; incomplete or
// inconsistent restores are refused. Adapted from the LFM3 extension's
// adapter test (gufo-org/lfm3-gufo-extension, MIT). Skips without
// GUFO_NEMOTRON_BOLT_GGUF; NVIDIA's drafter beside the trunk adds its rings.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <gufo/ext/transfer_pool.hpp>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "continuation_adapter.hpp"
#include "engine.hpp"

namespace nb = gufo::models::nemotron;
namespace cache = gufo::cache;

namespace {

constexpr int kSkip = 77;
std::size_t kVocabulary = 0;

void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}

template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid adapter operation was accepted");
}

struct Guard final : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows first,
                       cache::Rows end) override {
    overwrites += first < end;
  }
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
  void AfterReset() noexcept override { ++resets; }
  std::size_t overwrites{}, resets{};
};

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};

void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}

// Odd sizes make every tensor boundary cross a piece boundary.
constexpr std::size_t kPiece = (1 << 18) + 13;
constexpr cache::Rows kRowPiece = 97;

auto Acquire(gufo::hip::TransferPool& streams) {
  auto stream = streams.TryAcquire();
  Require(bool(stream), "stream exhausted");
  return stream;
}

Checkpoint Capture(nb::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams) {
  Checkpoint result{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + kRowPiece);
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *Acquire(streams)));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(kPiece, bytes.size() - offset);
        Wait(adapter.CapturePrivatePiece(
            slot, component.id, offset, std::span(bytes).subspan(offset, count),
            *Acquire(streams)));
        offset += count;
      }
    }
  }
  return result;
}

// Reverse order: metadata arrives after the device regions, and the first
// piece of each tensor arrives last. `omit_last` drops the metadata's last
// byte.
void Load(nb::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool omit_last = false) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); });
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const auto bytes = std::span<const std::byte>(checkpoint.bytes[i]);
    if (component.row_bytes) {
      for (cache::Rows end = checkpoint.positions[i].valid_rows; end;) {
        const auto first = end > kRowPiece ? end - kRowPiece : 0;
        Wait(adapter.CopyRowsIn(
            slot, component.id, first, end,
            bytes.subspan(first * component.row_bytes,
                          (end - first) * component.row_bytes),
            *Acquire(streams)));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > kPiece ? end - kPiece : 0;
        auto count = end - first;
        if (omit_last && component.id.value == 2 && first == 0)
          --count;
        Wait(adapter.LoadPrivatePiece(slot, component.id, first,
                                      bytes.subspan(first, count),
                                      *Acquire(streams)));
        end = first;
      }
    }
  }
}

std::vector<std::int32_t> Prompt(std::size_t tokens, std::uint32_t seed) {
  std::vector<std::int32_t> prompt(tokens);
  std::uint32_t state = seed;
  for (auto& token : prompt) {
    state = state * 1664525U + 1013904223U;
    token = static_cast<std::int32_t>(1000 + (state >> 8) % 60000);
  }
  return prompt;
}

std::vector<float> Logits(nb::Session& session) {
  std::vector<float> logits(kVocabulary);
  std::string error;
  Require(session.ReadLogits(logits, &error), error);
  return logits;
}

/// The session's state as host bytes: attention rows, Mamba states and the
/// drafter context.
nb::SessionSnapshot Snapshot(const nb::Session& session) {
  nb::SessionSnapshot snapshot;
  std::string error;
  Require(session.Save(&snapshot, &error), error);
  return snapshot;
}

bool Same(const nb::SessionSnapshot& a, const nb::SessionSnapshot& b) {
  return a.position == b.position && a.kv == b.kv &&
         a.recurrent == b.recurrent && a.draft == b.draft;
}

// Greedy decoding through the captured single-token graph.
std::vector<std::int32_t> Continue(nb::Session& session, std::size_t count) {
  std::vector<std::int32_t> emitted;
  std::string error;
  while (emitted.size() < count) {
    const auto logits = Logits(session);
    const auto token = static_cast<std::int32_t>(
        std::ranges::max_element(logits) - logits.begin());
    const std::int32_t one[] = {token};
    Require(session.Append(one, {}, &error), error);
    emitted.push_back(token);
  }
  return emitted;
}

void Sync(nb::Session& session, std::span<const std::int32_t> prompt) {
  std::string error;
  Require(session.Append(prompt, {}, &error), error);
}

// Prompt lengths exercise partial and whole row chunks and a prefill longer
// than one pass.
void CheckRoundTrip(nb::ContinuationAdapter& adapter,
                    gufo::hip::TransferPool& streams, std::size_t tokens) {
  Guard source_guard, target_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto target = adapter.CreateSlot(target_guard);
  const auto prompt = Prompt(tokens, 0x9e3779b9U);
  Sync(adapter.GetSession(*source), prompt);
  Require(source_guard.overwrites > 0, "prefill bypassed the write guard");

  // A longer unrelated frontier must be fully replaced by a shorter restore.
  const auto other = Prompt(tokens + 200, 0x2545f491U);
  Sync(adapter.GetSession(*target), other);

  const auto checkpoint = Capture(adapter, *source, streams);
  auto& original = adapter.GetSession(*source);
  const auto expected = Snapshot(original);
  Load(adapter, *target, checkpoint, streams);
  Require(adapter.Validate(*target, checkpoint.positions),
          "complete restore failed validation");
  Reject([&] { (void)adapter.Validate(*target, checkpoint.positions); });
  auto& restored = adapter.GetSession(*target);
  Require(Same(Snapshot(restored), expected),
          "restored state differs from the source session");
  Require(Logits(restored) == Logits(original), "restored logits differ");
  Require(Continue(restored, 24) == Continue(original, 24),
          "restored decode diverged from an uninterrupted run");
  Require(Same(Snapshot(restored), Snapshot(original)),
          "state diverged after decoding from a restore");

  const auto later = Capture(adapter, *source, streams);
  Require(adapter.Invalidate(*target), "invalidate failed");
  Require(target_guard.resets == 1, "invalidate skipped AfterReset");
  Require(adapter.Positions(*target).front().valid_rows == 0,
          "invalidate left a frontier");
  Load(adapter, *target, later, streams);
  Require(adapter.Validate(*target, later.positions), "later restore failed");
  Require(Same(Snapshot(adapter.GetSession(*target)),
               Snapshot(adapter.GetSession(*source))),
          "later restore differs");
  std::printf("round trip at %zu tokens: identical state, logits and decode\n",
              tokens);
}

void CheckRefusals(nb::ContinuationAdapter& adapter,
                   gufo::hip::TransferPool& streams) {
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const auto prompt = Prompt(300, 0x9e3779b9U);
  auto& session = adapter.GetSession(*slot);
  Sync(session, prompt);
  const auto checkpoint = Capture(adapter, *slot, streams);

  // Adapter slots restore only through the adapter.
  const auto snapshot = Snapshot(session);
  Reject([&] { (void)session.Restore(snapshot); });

  auto inconsistent = checkpoint.positions;
  inconsistent.back().valid_rows += 1;
  Reject([&] { adapter.BeginRestore(*slot, inconsistent); });
  Require(adapter.Positions(*slot) == checkpoint.positions,
          "refused restore changed the slot");

  // A missing byte latches failure until the slot is invalidated.
  Load(adapter, *slot, checkpoint, streams, true);
  Require(!adapter.Validate(*slot, checkpoint.positions),
          "incomplete restore passed validation");
  Require(!slot->IsValid(), "failed restore left an executable slot");
  Require(!adapter.Validate(*slot, checkpoint.positions),
          "a false validation must stay false");
  Reject([&] { (void)adapter.GetSession(*slot); });
  Require(adapter.Invalidate(*slot), "failed restore cannot be invalidated");
  Require(slot->IsValid(), "invalidate must leave a usable empty slot");

  // A cold prefill after failure reproduces the original state.
  Sync(adapter.GetSession(*slot), prompt);
  Require(adapter.Positions(*slot) == checkpoint.positions,
          "cold prefill after failure lost its frontier");
  std::printf("refusals: inconsistent, incomplete and snapshot restores\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = argc > 1 ? argv[1] : "";
  if (model_path.empty()) {
    if (const char* env = std::getenv("GUFO_NEMOTRON_BOLT_GGUF"))
      model_path = env;
  }
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    std::fprintf(stderr,
                 "SKIP: set GUFO_NEMOTRON_BOLT_GGUF to the Nemotron 3.5 Bolt "
                 "GGUF\n");
    return kSkip;
  }
  try {
    std::string error;
    auto model = nb::Model::Load(model_path, {.max_context = 8192}, &error);
    Require(model != nullptr, error);
    const std::filesystem::path trunk(model_path);
    const auto drafter =
        trunk.parent_path() / (trunk.stem().string() + "-LiLiCorr.gguf");
    if (std::filesystem::exists(drafter))
      Require(model->LoadDrafter(drafter.string(), &error), error);
    std::printf("drafter rings: %s\n",
                model->drafter() != nullptr ? "checked" : "absent");
    kVocabulary = model->config().vocab_size;
    nb::ContinuationAdapter adapter(model, 8192, {1, 2, 3});
    Require(adapter.GetCapabilities().persistent_encoding,
            "adapter is not persistent");
    gufo::hip::TransferPool streams(2);
    CheckRefusals(adapter, streams);
    for (const std::size_t tokens : {3U, 100U, 600U, 1031U, 2500U})
      CheckRoundTrip(adapter, streams, tokens);
    std::printf("continuation checks passed\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
