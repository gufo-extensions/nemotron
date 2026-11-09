# Nemotron 3.5 Bolt quality

There is no capability score yet. This page records numerical checks of the
ROCm runtime against llama.cpp on NVIDIA's QAD 24 GB GGUF, the extension's own
tests, and Gufo's functional API suites. Measured November 9, 2026, with
extension 0.1.0 on Gufo `5f9612c3`.

## ROCm runtime versus llama.cpp

llama.cpp build 12213 (`53eb66943`), F32 KV cache, Vulkan with every layer
offloaded and the CPU backend with no offload. Both sides decode the same
token IDs, produced by llama.cpp's tokenizer with special tokens parsed from
the two prompts in
[tools/llama_parity/prompts](../../../tools/llama_parity/prompts): a one-line
question and a conversation with a system message, a code answer and a
follow-up. "Prefill" runs each prompt in one pass; "decode" appends it one
token at a time through the captured decode graph
(`nemotron_runtime_probe --step`). Rows compare next-token distributions at
every prompt position.

| Prompt | Pair | Top-1 agreement | KL median | KL p90 | KL max |
| --- | --- | ---: | ---: | ---: | ---: |
| Short, 28 tokens | llama.cpp Vulkan vs CPU | 28/28 | 6.8e-4 | 2.3e-3 | 5.3e-3 |
| | Gufo prefill vs Vulkan | 28/28 | 4.6e-4 | 2.2e-3 | 3.2e-3 |
| | Gufo prefill vs CPU | 28/28 | 4.7e-4 | 3.7e-3 | 5.8e-3 |
| | Gufo decode vs Vulkan | 28/28 | 5.6e-4 | 2.2e-3 | 4.9e-3 |
| | Gufo decode vs CPU | 28/28 | 4.5e-4 | 3.0e-3 | 5.5e-3 |
| Long, 141 tokens | llama.cpp Vulkan vs CPU | 139/141 | 3.0e-5 | 2.6e-3 | 1.7e-1 |
| | Gufo prefill vs Vulkan | 140/141 | 2.0e-5 | 2.5e-3 | 3.7e-2 |
| | Gufo prefill vs CPU | 138/141 | 5.0e-5 | 2.1e-3 | 4.5e-2 |
| | Gufo decode vs Vulkan | 138/141 | 3.5e-5 | 2.1e-3 | 6.3e-2 |
| | Gufo decode vs CPU | 140/141 | 2.4e-5 | 2.2e-3 | 1.2e-1 |

Gufo agrees with each llama.cpp backend as closely as the two backends agree
with each other. Every disagreement is at one of three positions of the long
prompt, all near ties: positions 26 and 100 (reference top-1 probability 0.12
and 0.38), which also split llama.cpp's backends, and position 132 (0.13 to
0.14).

## Runtime and continuation

- `nemotron_runtime_test`, on the released files with the LiLiCorr drafter:
  - a prompt appended one token at a time through the decode graph matches
    one bulk prefill (KL 7.4e-5, same top-1);
  - decoding two to eight sessions in one batch gives rows bit-identical to
    decoding them one at a time;
  - greedy draft-and-verify rounds emit the same 48 tokens as plain greedy
    decoding, with 38 of them accepted drafts;
  - device logit heads draw the full rows' tokens for top-k sampling.
- `nemotron_continuation_test` captures cache checkpoints in odd pieces and
  restores them in reverse order into sessions holding other, longer states.
  At 3, 100, 600, 1,031 and 2,500 tokens the restored attention rows, Mamba
  states and drafter rings equal the source byte for byte, and so do the
  logits and the next 24 greedy tokens. Inconsistent and incomplete restores
  are refused.
- The GPU operator tests (`nemotron.kernel_ops`, `gemv_ops`, `drafter_ops`,
  `mmq_ops`) compare every kernel with CPU evaluations of the same formulas.
- The tokenizer and the chat template match Hugging Face's on the golden
  fixtures in `tests/fixtures/`.

## Functional API suites

Gufo's `tests/functional/run.py --suite all` with
`--sampling-preset provider`, four sessions, once autoregressive and once
with `--speculative dflash2` and NVIDIA's LiLiCorr drafter. Both modes pass
the same 40 of 53 suites; the drafting run exercises drafts in `state-edges`.

Passed: discovery, openai-count-tokens, messages-count-tokens,
messages-image-counts, responses, stops, structured, tool-streaming,
tool-edges, tool-agent, tool-agent-loop, tool-history, messages-tools,
tool-untyped, tool-choice-cache, tool-parameter-order, tool-type-unions,
sampling-ranges, logprobs, responses-logprobs, batch, long-context,
state-edges, progress, stream-start, metrics, cache-rotation,
cache-concurrency, cache-shared-prefix, system-injection, and the cancel,
persistence and disk-spacing suites in RAM and on disk.

The other thirteen fail for these reasons, the same in both modes:

| Suite | Reason |
| --- | --- |
| `sampling-defaults`, `cache-edits` | Pass when run alone; they fail only after the earlier suites of the full run have filled the RAM cache, where every checkpoint keeps about 175 MB of private state, mostly Mamba states. Their replays then find no checkpoint to restore. |
| `conversation` | A reply discarded after cancellation re-prefills 22 tokens; the suite allows 16. The template moves the reasoning-effort hint to the last user turn, so the hint and the new turn are recomputed. |
| `cache-growth`, `reasoning-separator` | The template replaces earlier reasoning with `<think></think>` unless the request sets `preserve_thinking`, so a replayed turn with reasoning no longer matches the generated tokens. With `preserve_thinking` the generated turn is reused. |
| `cache-depth` | The suite expects its prompt to exceed 16,384 tokens; Nemotron's tokenizer encodes it in 16,217. |
| `tool-native-schemas` | A schema variant changes the prompt by 5 tokens against a tolerance of 4 calibrated on Qwen's tokenizer. |
| `tools`, `reasoning-budget` | At reasoning effort `high` the model reasons past the suites' output limits (about 450 tokens before a forced tool call against 256), then answers correctly. |
| `structured-limits` | Under presence, frequency and repeat penalties with `tool_choice: auto`, the model declines to call the tool. |
| `tool-mixed`, `tool-schema-edges` | The model writes `None` for a value and a JSON object as a string argument where the suites expect other arguments. |
| `tool-reasoning` | The model quotes a call in a code block instead of making it. |

## Reproduce

```sh
L=/path/to/llama.cpp   # a checkout with its build in build/bin; see tools/llama_parity
MODEL=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
TOKENS=$(llama_logits "$MODEL" --special --text-file tools/llama_parity/prompts/long.txt)
llama_logits "$MODEL" --tokens "$TOKENS" --dump vulkan.bin --f32-kv
llama_logits "$MODEL" --tokens "$TOKENS" --dump cpu.bin --f32-kv --cpu
nemotron_runtime_probe --model "$MODEL" --tokens "$TOKENS" --dump prefill.bin
nemotron_runtime_probe --model "$MODEL" --tokens "$TOKENS" --dump decode.bin --step
python3 tools/llama_parity/compare_logits.py vulkan.bin prefill.bin --vocab 131072

# In a Gufo checkout built with this extension:
python3 tests/functional/run.py --output report --sampling-preset provider \
  --record-baseline --expected-input-modalities text --suite all -- \
  build/gpu-test/gufo serve llm --model "$MODEL" --sessions 4 \
  [--speculative dflash2 --dflash-model NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf]
```

The drafter's own comparison with llama.cpp's DFlash implementation
(`tools/llama_parity/llama_dflash.cpp` and `nemotron_drafter_probe`) was not
repeated for this release.
