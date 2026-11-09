#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"

// Without HIP the provider's library isn't built; registering its probe
// still lets Gufo name the model it can't run here.
void nemotron_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "nemotron35-bolt",
      .probe = &gufo::models::nemotron::ProbeBolt,
      .probe_gguf = &gufo::models::nemotron::ProbeBoltGguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
      .accepts_unsized_tensors = true,
  });
#else
  (void)registry;
#endif
}
