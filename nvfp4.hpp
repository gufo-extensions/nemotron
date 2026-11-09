#ifndef GUFO_MODELS_NEMOTRON_NVFP4_HPP_
#define GUFO_MODELS_NEMOTRON_NVFP4_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::nemotron {

/// GGML_TYPE_NVFP4 in llama.cpp. Gufo's GGUF reader has no name for it and
/// leaves these tensors unsized; this extension sizes and decodes them.
inline constexpr core::GgmlType kNvfp4 = static_cast<core::GgmlType>(40);

/// NVFP4 block: 64 E2M1 values as four 16-element sub-blocks, each with an
/// unsigned E4M3 scale. Within a sub-block, low nibbles hold elements 0..7
/// and high nibbles elements 8..15. A per-tensor F32 scale is applied by the
/// caller.
struct BlockNvfp4 {
  std::uint8_t d[4];
  std::uint8_t qs[32];
};
static_assert(sizeof(BlockNvfp4) == 36, "BlockNvfp4 must be 36 bytes");

inline constexpr std::size_t kNvfp4BlockElements = 64;

/// Dequantizes `k` values (a multiple of 64) without the per-tensor scale.
void DequantizeNvfp4(const void* src, float* dst, std::size_t k);

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_NVFP4_HPP_
