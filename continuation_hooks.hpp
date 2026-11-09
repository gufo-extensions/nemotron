#ifndef GUFO_MODELS_NEMOTRON_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_NEMOTRON_CONTINUATION_HOOKS_HPP_

#include <cstdint>

namespace gufo::models::nemotron {

/// Installed only in sessions owned by the continuation adapter. Every
/// callback runs on the host before the session changes state or submits
/// device work, and may throw to refuse it.
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  /// Attention rows [first, end) are about to be written.
  virtual void BeforeWrite(std::uint32_t first, std::uint32_t end) = 0;
  virtual void BeforeReset() = 0;
  /// A host snapshot is about to replace the state.
  virtual void BeforeRestore() = 0;
};

}  // namespace gufo::models::nemotron

#endif  // GUFO_MODELS_NEMOTRON_CONTINUATION_HOOKS_HPP_
