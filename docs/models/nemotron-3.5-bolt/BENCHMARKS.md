# Nemotron 3.5 Bolt benchmarks

AMD Strix Halo (`gfx1151`, Ryzen AI Max+ 395), 128 GB unified memory, OS
profile `performance`. The power mode was measured, not set: package power
averaged 86 to 87 W under load (peaks 118 W) with the GPU at about 2.35 GHz,
the 85 W mode; sustained rates are higher in the 140 W mode. NVIDIA's QAD
24 GB GGUF with its LiLiCorr drafter; both engines draft up to 7 tokens per
round. HTTP on both sides, greedy (temperature 0, seed 1), thinking off.
Extension 0.1.0 on Gufo `5f9612c3`, `release` build with ROCm 10.1;
llama.cpp build 12213 (`53eb66943`), Vulkan. Measured November 9, 2026, with
one sample per point after a warm-up.
Positive gain favors Gufo.
Quality and measurement details belong in [QUALITY.md](QUALITY.md).

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Nemotron 3.5 Bolt 24GB AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 975.43 | 467.20 | +108.8% | 36.26 | 29.20 | +24.2% |
| 8,192 | 933.73 | 442.53 | +111.0% | 35.72 | 29.13 | +22.6% |
| 32,768 | 816.76 | 385.18 | +112.0% | 34.44 | 28.25 | +21.9% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, LiLiCorr

Shared pp, separate mixed and repetitive tg and gains. pp is the highest
measured rate per engine and depth across both text types. Gufo drafts while
a request decodes alone; for concurrent requests see
[Concurrent requests in 0.2.3](#concurrent-requests-in-023).

<!-- bench:single-lilicorr -->
| Nemotron 3.5 Bolt 24GB LiLiCorr<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 999.28 | 445.32 | +124.4% | 41.41 | 26.68 | +55.2% | 53.48 | 34.73 | +54.0% |
| 8,192 | 922.49 | 424.30 | +117.4% | 42.01 | 30.04 | +39.8% | 51.36 | 33.92 | +51.4% |
| 32,768 | 806.67 | 368.47 | +118.9% | 33.23 | 22.06 | +50.6% | 47.95 | 31.78 | +50.9% |
<!-- /bench -->

![Single user, LiLiCorr](artifacts/charts/single-lilicorr.svg)

## Drafting changes in 0.2.2

Decode rates with LiLiCorr from the command line, in tokens per second:
`gufo prompt -v` for 512 tokens and `gufo bench -p 0 -n 256 -r 2`, release
builds on Gufo `0e2de43e`. "0.2.1" is the released extension; "expert
kernels" adds the BF16 and per-expert IQ2_XXS products for draft
verification; "adaptive" also cuts rounds where the chance of reaching a
draft falls below 15 %. Each pair ran in ABBA order (mean of the runs) under
a clean `machine-lock perf`, November 10 and 11, 2026; profile
`performance`, package power 92 to 94 W on average (peaks 119 W), GPU at
about 2.4 GHz. Greedy outputs are identical across all three builds.

| Workload | 0.2.1 | Expert kernels | Adaptive |
| --- | ---: | ---: | ---: |
| Fibonacci function (code), greedy | 105.3 | 111.4 | 109.6 |
| Lighthouse essay, greedy | 48.5 | 51.7 | 59.3 |
| Lighthouse essay, temperature 1, top-p 0.95 | | 49.9 | 56.1 |
| TCP congestion control explanation, greedy | | 68.2 | 71.2 |
| JSON schema and record, greedy | | 72.4 | 74.8 |
| `gufo bench` tg256, greedy | 86.5 | 91.9 | 97.0 |
| `gufo bench` tg256, temperature 1, top-p 0.95 | 41.4 | 44.4 | 51.4 |

Autoregressive decoding (`gufo bench` tg256) stays at 36.6 tokens per second.
The cost of a round grows by about 2.9 ms per drafted token over a fixed
42 ms (drafting, a one-token pass and the round's overhead), so code, which
keeps 6 of 7 drafts, gains little from shorter rounds, and prose, which
keeps 2, gains most.

## Concurrent requests in 0.2.3

Release builds on Gufo `0e2de43e`, served with the projector and LiLiCorr
(`--sessions 4 --context 32768`), greedy, thinking off, each pair of builds
measured back to back under a clean `machine-lock perf` on November 11,
2026; profile `performance`, package power 85 to 86 W on average (peaks
112 W), GPU at 2.3 to 2.5 GHz.

**A decoder beside a cold prompt**
([tools/bench/decode_stalls.py](../../../tools/bench/decode_stalls.py)): one
request streams an essay; after 40 chunks a second request sends a cold
prompt for one output token. The table gives the second request's time and
the streaming request's longest pause while it ran, medians of three after
a warm-up. 0.2.2 runs each 2,048-token pass and the image encoder whole;
0.2.3 splits each pass into four layer groups (Gufo's limit for 2,048
tokens) and pauses the encoder about every 250 ms and after its last block,
with a decode round for the streaming request each time.

| Cold prompt | Tokens | 0.2.2 prompt (s) | 0.2.3 prompt (s) | 0.2.2 longest pause (s) | 0.2.3 longest pause (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text, 4,500 words | 5,300 | 5.85 | 6.31 | 2.10 | 0.60 |
| Rendered screen, 3840x2160 | 3,355 | 6.91 | 7.65 | 4.50 | 0.62 |

The streaming request sends twice as many chunks while the prompt runs (35
instead of 18 beside the text, 82 instead of 42 beside the image); each
remaining pause is one layer group. With a decoder beside them, the image
and text prompts' greedy answers and the decoder's own answer equal those
of 0.2.2 without one.

**Concurrent requests**
([tools/bench/concurrent_bench.py](../../../tools/bench/concurrent_bench.py)):
the requests of each mix start together and write up to 384 tokens (the
Fibonacci function stops at 231); tokens of all requests per second of wall
time, mean of two runs. 0.2.2 decodes concurrent requests one token each in
shared passes; 0.2.3 lets two or three of them draft one round each in turn
when their measured rounds emit tokens faster.

| Mix | 0.2.2 (tok/s) | 0.2.3 (tok/s) | Gain |
| --- | ---: | ---: | ---: |
| Fibonacci function, ring buffer class (code, code) | 65.5 | 91.2 | +39.4% |
| Code, code, JSON schema and record | 68.6 | 95.5 | +39.2% |
| Fibonacci function, lighthouse essay (code, prose) | 57.4 | 66.8 | +16.3% |
| Lighthouse essay, TCP explanation (prose, prose) | 58.2 | 58.6 | +0.8% |
| Code, code, prose, prose | 83.9 | 83.3 | -0.7% |

All 13 answers are identical across both builds. Prose keeps about 2 of 7
drafts, so its rounds emit fewer tokens per second than a shared pass and
both requests mostly share passes; four decoding together always share
them.

**A request alone** takes the same path as in 0.2.2: Gufo asks for layer
groups only beside decoders, and a lone decoder drafts. Builds in the order
0.2.2, 0.2.3, 0.2.3, 0.2.2, one request at a time; medians of all runs
(package power 77 W on average while busy, peaks 110 W):

| Request | 0.2.2 | 0.2.3 | Change |
| --- | ---: | ---: | ---: |
| Cold text prompt, 5,300 tokens (s) | 5.40 | 5.39 | -0.3% |
| Cold 3840x2160 screen, 3,355 tokens (s) | 6.11 | 6.13 | +0.3% |
| Fibonacci function, 231 tokens (tok/s) | 117.8 | 117.7 | 0.0% |
| Lighthouse essay, 384 tokens (tok/s) | 54.9 | 54.8 | -0.1% |
| TCP explanation, 384 tokens (tok/s) | 67.8 | 67.7 | -0.2% |

The answers are identical, and each build's runs spread by up to 0.8 %
(the text prompts by up to 4 %). Beside a decoder, the prompts above take 0.46 s
and 0.74 s longer: 8 and 13 decode rounds of about 57 ms each for the
streaming request, which emits twice as many chunks meanwhile.

Raw results: [stalls](artifacts/concurrency-stalls.jsonl),
[concurrent requests](artifacts/concurrency-requests.jsonl),
[single requests](artifacts/concurrency-single.jsonl).

## Images

Time to the first token of a prompt with one image: a unique system message
(so no prompt cache serves it), the image and "Describe this image.",
thinking off, one output token. The time covers decoding, preparing and
encoding the image and prefilling the prompt. Median of three requests
after a warm-up, single session, over HTTP
([tools/bench/vision_bench.py](../../../tools/bench/vision_bench.py)).
Extension 0.2.0 with NVIDIA's vision projector GGUF; llama.cpp
serves the same files (`--mmproj`, every layer offloaded). The photo, logo and
rendered screens come from
[tools/bench/vision_images.sh](../../../tools/bench/vision_images.sh). Package
power averaged 84 W (peaks 107 to 111 W), profile `performance`; measured
November 9, 2026, under a machine reservation, both runs clean.

| Image | Gufo prompt tokens | llama.cpp prompt tokens | Gufo (s) | llama.cpp (s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Photo, 512x512 | 306 | 310 | 0.630 | 1.416 | +125% |
| Rendered settings page, 1024x1024 | 1,080 | 1,081 | 1.762 | 4.264 | +142% |
| Rendered screen, 1920x1080 | 2,093 | 2,093 | 3.576 | 8.881 | +148% |
| Rendered screen, 3840x2160 | 3,365 | 3,320 | 6.191 | 16.059 | +159% |

Prompt lengths vary by a few tokens with the system message's random
nonce. At 3840x2160 llama.cpp sizes the image to a different grid (76x43
instead of 77x43 tokens); see [QUALITY.md](QUALITY.md#images-versus-llamacpp). Raw
results: [Gufo](artifacts/images-gufo.json),
[llama.cpp](artifacts/images-reference.json).

The encoder alone (`nemotron_vision_probe`, median of five after a warm-up;
package power 81 W, GPU at 2.17 GHz on average):

| Image | Tokens | Decode | Prepare | Encode | Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Photo, 512x512 | 256 | 1.1 ms | 1.3 ms | 77 ms | 20.6 TFLOPS |
| Rendered settings page, 1024x1024 | 1,024 | 4.6 ms | 7.0 ms | 425 ms | 19.6 TFLOPS |
| Rendered screen, 1920x1080 | 2,040 | 9.1 ms | 49 ms | 1,122 ms | 19.6 TFLOPS |
| Rendered screen, 3840x2160 | 3,311 | 37 ms | 146 ms | 2,377 ms | 19.7 TFLOPS |

Decoding and preparing (resize, normalization) run on the request's thread;
encoding runs on the GPU when prefill reaches the image. At 1920x1080 the
encoder is about a third of the time to the first token; half of the
encoder's time is the ViT's attention over 8,170 tokens (about 20 TFLOPS),
most of the rest its matrix products (about 25 TFLOPS).
