#include <chrono>
#include <gufo/ext/generated/nemotron_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <optional>

#include "probe.hpp"
#include "runner.hpp"

namespace gufo::models::nemotron {

int RunNemotronPrompt(const ext::TextPromptOptions& opt,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start);
int RunNemotronChat(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);
int RunNemotronBench(const ext::TextBenchOptions& opt,
                     const std::shared_ptr<const core::GgufReader>& reader,
                     std::chrono::steady_clock::time_point load_start);

namespace {
std::optional<sampling::TextSamplingDefaults> ProbeSamplingDefaults(
    const core::GgufReader&) {
  return SamplingDefaults();
}
}  // namespace

}  // namespace gufo::models::nemotron

void nemotron_provider_register(gufo::ext::Registry& registry) {
  namespace nemotron = gufo::models::nemotron;
  registry.AddProvider(
      gufo::ext::generated::extension_nemotron::kProvider_nemotron35_bolt);
  registry.AddTextProvider({
      .id = "nemotron35-bolt",
      .probe = &nemotron::ProbeBolt,
      .probe_gguf = &nemotron::ProbeBoltGguf,
      .create = &nemotron::CreateNemotronTextRunner,
      .bench_driver = &nemotron::RunNemotronBench,
      .prompt_driver = &nemotron::RunNemotronPrompt,
      .chat_driver = &nemotron::RunNemotronChat,
      .sampling_defaults = &nemotron::ProbeSamplingDefaults,
      // The trunk's NVFP4 tensors are a type core cannot size.
      .accepts_unsized_tensors = true,
      // NVIDIA's LiLiCorr drafter is a DFlash drafter; rounds draft a fixed
      // number of tokens.
      .speculative_defaults = {{.backend = "dflash2",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1,
                                .draft_policy = "fixed",
                                .draft_policies = {"fixed"}}},
  });
}
