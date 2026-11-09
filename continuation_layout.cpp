#include "continuation_layout.hpp"

#include <stdexcept>

namespace gufo::models::nemotron {

ContinuationLayout::ContinuationLayout(const Config& c, std::uint32_t context,
                                       std::optional<std::size_t> draft_bytes)
    : context_(context) {
  if (context == 0 || c.blocks.size() != c.num_layers || c.vocab_size == 0 ||
      c.AttentionKvDim() == 0 || c.ssm_conv_kernel < 2)
    throw std::invalid_argument("invalid Nemotron continuation geometry");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t layer = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{.id = {id},
                            .layout_version = 1,
                            .kind = row ? cache::ComponentKind::kAppendRows
                                        : cache::ComponentKind::kPrivateState,
                            .row_bytes = row,
                            .rows_per_chunk = row ? cache::Rows{256} : 0,
                            .state_bytes = state},
                           part,
                           layer});
  };
  add(ContinuationPart::kLogits, 0, std::size_t{c.vocab_size} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  const std::size_t kv_row = std::size_t{c.AttentionKvDim()} * 2;  // FP16
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    switch (c.blocks[il]) {
      case BlockType::kMamba:
        add(ContinuationPart::kConv, 0,
            std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels() *
                sizeof(float),
            il);
        add(ContinuationPart::kSsm, 0,
            std::size_t{c.ssm_heads} * c.SsmHeadDim() * c.ssm_state_size *
                sizeof(float),
            il);
        break;
      case BlockType::kAttention:
        add(ContinuationPart::kKey, kv_row, 0, il);
        add(ContinuationPart::kValue, kv_row, 0, il);
        break;
      case BlockType::kMoe:
        break;
    }
  }
  if (draft_bytes)
    add(ContinuationPart::kDraft, 0, *draft_bytes);
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t position) const {
  if (position > context_)
    throw std::invalid_argument("invalid Nemotron continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_)
    positions.push_back({component.descriptor.id, position});
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}

std::size_t ContinuationLayout::RowBytes(std::uint32_t position) const {
  const auto positions = Positions(position);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < components_.size(); ++i)
    bytes += components_[i].descriptor.row_bytes * positions[i].valid_rows;
  return bytes;
}

std::vector<cache::BackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram) const {
  // Contiguous page groups keep one payload owner and charge all slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

}  // namespace gufo::models::nemotron
