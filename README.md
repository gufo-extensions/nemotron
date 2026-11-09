# Nemotron Gufo extension

A community [Gufo](https://github.com/gufo-org/gufo) extension for NVIDIA's
Nemotron-H models on AMD Strix Halo (`gfx1151`).

> **Status:** first release, 0.1.0. Text generation runs through
> `gufo serve`, `prompt`, `chat` and `bench`, with LiLiCorr drafting and
> Gufo's continuation cache. See [quality](docs/models/nemotron-3.5-bolt/QUALITY.md),
> [benchmarks](docs/models/nemotron-3.5-bolt/BENCHMARKS.md) and the known
> limits below.

## Models

| Provider | Model | Files |
| --- | --- | --- |
| `nemotron35-bolt` | Nemotron 3.5 Bolt 74B-A9B | NVIDIA's QAD 24 GB GGUF, optionally its LiLiCorr drafter GGUF |

The weights aren't part of this repository. They're published by NVIDIA under
its own license; read it before you download them.

## Use

```sh
MODEL=/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
DRAFT=/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
gufo prompt --model "$MODEL" --prompt 'What is the capital of France?' --think off
gufo serve llm --model "$MODEL" --speculative dflash2 --dflash-model "$DRAFT"
```

- Thinking is on by default and its text arrives as `reasoning_content`;
  `reasoning_effort: "none"` turns it off, `"minimal"`, `"low"` and
  `"medium"` select the template's efficient mode.
- Tool calls use the template's Qwen3-Coder XML format.
- Sampling defaults to temperature 1.0 and top-p 0.95, as NVIDIA recommends.
  For top-k sampling the GPU summarizes each row, so the host sorts a few
  candidates instead of the vocabulary; other settings sample the full row.
- NVIDIA's LiLiCorr drafter is a DFlash drafter, so it loads with
  `--speculative dflash2 --dflash-model`. Each round drafts up to
  `--draft-tokens` (at most 7, the default) tokens and the trunk verifies them
  in one pass; the drafter's draft length is fixed (`--draft-policy fixed`).
  A verified row equals decoding its token alone, so greedy output with the
  drafter equals greedy output without it.
- Concurrent requests decode together, one token each, in batches of up to
  eight; a request decoding alone drafts.
- Conversations continue from Gufo's RAM and disk cache: attention rows,
  Mamba states and the drafter's rings are restored instead of recomputed.

## Known limits

- Text only: NVIDIA's vision projector is not supported yet.
- 40 of Gufo's 53 functional API suites pass, with and without the drafter;
  [QUALITY.md](docs/models/nemotron-3.5-bolt/QUALITY.md) gives the reason for
  each of the others. Most come from the model or its template: earlier
  reasoning is dropped from the history unless the request sets
  `preserve_thinking`, the reasoning-effort hint moves to the last user turn,
  and at effort `high` the model reasons at length before a forced tool call.
- A prompt pass beside decoding sessions runs whole (up to 2,048 tokens), so
  other sessions wait for it.

## Build

Compile the extension into a Gufo source build:

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/absolute/path/to/nemotron
cmake --build --preset release --parallel 4
build/release/gufo diagnose --section extensions
```

With Nix, `nix build` builds the Gufo revision pinned in `flake.nix` with this
extension.

## Tests

```sh
cmake --preset cpu-test -DGUFO_EXTENSION_DIRS=/absolute/path/to/nemotron
cmake --build --preset cpu-test --target nemotron_config_test \
  nemotron_probe_test nemotron_nvfp4_test nemotron_tokenizer_test \
  nemotron_chat_template_test
GUFO_NEMOTRON_BOLT_GGUF=/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  ctest --preset cpu-test -R '^nemotron\.' --output-on-failure
```

Tests labeled `external-model` read the released files and skip without
`GUFO_NEMOTRON_BOLT_GGUF`; the drafter and projector are found beside the
trunk by NVIDIA's file names. The golden fixtures in `tests/fixtures/` come
from the scripts in `tools/`, which run the model's Hugging Face tokenizer and
chat template. `nemotron_reference_probe` runs the scalar CPU reference over
a token sequence; [tools/llama_parity](tools/llama_parity/README.md) compares
its logits with llama.cpp's.

In a `gpu-test` build, `nemotron_kernel_ops_test`, `nemotron_gemv_ops_test`,
`nemotron_drafter_ops_test` and `nemotron_mmq_ops_test` check the HIP kernels
against CPU evaluations of the same formulas (label `gpu`).
`nemotron_runtime_test` (labels `external-model`, `slow`) loads the released
files and checks sessions: one-token decode against prefill, snapshots,
batched decoding and, with the drafter, draft-and-verify rounds against
greedy decoding. `nemotron_continuation_test` restores cache checkpoints into
other sessions and compares state, logits and greedy decoding byte for byte. `nemotron_runtime_probe` dumps the runtime's logits for the
same comparison with llama.cpp; the prompts are in
[tools/llama_parity/prompts](tools/llama_parity/prompts).

## Supported Gufo revisions

| Extension | Gufo | Extension API |
| --- | --- | --- |
| 0.1.0 (unreleased) | `5f9612c3` | 0.2.16 |

## Release history

- 0.1.0 (unreleased): Nemotron 3.5 Bolt 74B-A9B with the ROCm runtime,
  LiLiCorr drafting, batched decoding and continuation through Gufo's cache.

## License

MIT, see [LICENSE](LICENSE). The quantized matrix kernels are adapted from
llama.cpp, DS4 and Gufo, the continuation adapter from the LFM3 extension,
all MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Maintainer: [@mduka](https://github.com/mduka).
