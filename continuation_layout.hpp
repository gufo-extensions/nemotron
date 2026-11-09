#ifndef GUFO_MODELS_NEMOTRON_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_NEMOTRON_CONTINUATION_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/adapter.hpp>
#include <optional>
#include <span>
#include <vector>

#include "config.hpp"

namespace gufo::models::nemotron {

/// State inventory shared by the adapter and its checks. IDs refer to this
/// representation, never to an allocation or serving slot.
enum class ContinuationPart {
  kLogits,
  kMetadata,
  kConv,
  kSsm,
  /// Attention rows are final once written.
  kKey,
  kValue,
  /// The drafter's rings and anchor rows. Rings overwrite their slots, so
  /// they can't be shared between checkpoints and are copied whole.
  kDraft,
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};

inline constexpr std::size_t kContinuationMetadataBytes = 16;

class ContinuationLayout {
public:
  /// `draft_bytes` is the drafter context's state when sessions draft.
  ContinuationLayout(const Config& config, std::uint32_t context,
                     std::optional<std::size_t> draft_bytes);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t position) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  /// Exact row bytes at a frontier; a checkpoint adds PrivateBytes() to them.
  [[nodiscard]] std::size_t RowBytes(std::uint32_t position) const;
  /// Rows and private state share the same committed pages.
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_CONTINUATION_LAYOUT_HPP_
