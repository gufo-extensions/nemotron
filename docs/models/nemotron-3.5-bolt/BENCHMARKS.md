# Nemotron 3.5 Bolt benchmarks

AMD Strix Halo (`gfx1151`, Ryzen AI Max+ 395), 128 GB unified memory, OS
power profile `performance`, device profile `balanced` (the front button),
whose sustained limit is 85 W: package power averaged 85 to 86 W under load
(peaks 118 W in the first seconds) with the GPU at 2.3 to 2.5 GHz. The
140 W profile can't hold its power for long because of the heat, so the
benchmarks don't use it. NVIDIA's QAD 24 GB GGUF with its LiLiCorr drafter;
both engines draft up to 7 tokens per round. HTTP on both sides, greedy
(temperature 0, seed 1), thinking off. Gufo runs `release` builds with ROCm
10.1; llama.cpp is build 12213 (`53eb66943`), Vulkan. Each section names
the extension and Gufo versions and the date. Positive gain favors Gufo.
Quality and measurement details belong in [QUALITY.md](QUALITY.md).

## Single user, autoregressive

Extension 0.3.0 on Gufo 0.34.0 (`2845939b`), measured November 13, 2026;
llama.cpp on November 11 with the same harness and workloads. Three samples
per point after a warm-up, mean ± sd. pp2048 / tg128; depth is the cached
prefix in tokens. Against 0.2.3 on Gufo `0e2de43e` (the same extension code,
November 11), prompts run 2 to 3 % faster and decoding 0.5 % faster.

<!-- bench:single-ar -->
| Nemotron 3.5 Bolt 24GB AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 986.01 ± 0.96 | 460.02 ± 3.61 | +114.3% | 36.46 ± 0.02 | 29.33 ± 0.11 | +24.3% |
| 8,192 | 948.29 ± 0.72 | 443.62 ± 2.16 | +113.8% | 35.85 ± 0.05 | 29.09 ± 0.12 | +23.2% |
| 32,768 | 827.85 ± 4.75 | 380.48 ± 3.29 | +117.6% | 34.60 ± 0.01 | 28.15 ± 0.05 | +22.9% |
| 65,536 | 702.75 ± 21.29 | 328.39 ± 1.81 | +114.0% | 32.61 ± 0.03 | 26.99 ± 0.16 | +20.8% |
| 131,072 | 556.65 ± 3.34 | 254.62 ± 1.78 | +118.6% | 29.88 ± 0.00 | 25.30 ± 0.15 | +18.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, LiLiCorr

Shared pp, separate mixed and repetitive tg and gains. pp is the highest
measured rate per engine and depth across both text types. Gufo drafts while
a request decodes alone; for concurrent requests see
[Concurrent requests](#concurrent-requests).

<!-- bench:single-lilicorr -->
| Nemotron 3.5 Bolt 24GB LiLiCorr<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 977.55 ± 24.02 | 441.57 ± 2.72 | +121.4% | 51.14 ± 6.89 | 27.38 ± 0.71 | +86.8% | 63.78 ± 2.11 | 33.70 ± 0.98 | +89.3% |
| 8,192 | 926.30 ± 1.68 | 420.77 ± 0.93 | +120.1% | 48.31 ± 2.73 | 27.23 ± 0.58 | +77.4% | N/A | N/A | N/A |
| 32,768 | 810.74 ± 3.52 | 364.54 ± 1.18 | +122.4% | 43.31 ± 3.46 | 21.65 ± 0.24 | +100.0% | N/A | N/A | N/A |
| 65,536 | 700.96 ± 4.65 | 317.53 ± 0.94 | +120.8% | 47.96 ± 3.17 | 21.39 ± 1.05 | +124.2% | 53.02 ± 0.23 | 25.80 ± 1.03 | +105.5% |
| 131,072 | 551.14 ± 3.18 | 248.28 ± 1.82 | +122.0% | 40.90 ± 5.73 | 20.44 ± 2.53 | +100.1% | 50.04 ± 0.59 | 23.93 ± 0.20 | +109.1% |
<!-- /bench -->

![Single user, LiLiCorr](artifacts/charts/single-lilicorr.svg)

After the 8,192- and 32,768-token prefixes Nemotron answers the repetition
prompt with 8 tokens on both engines, so the repetitive text has no rows
there. Another process used one CPU core for about 6 s during the untimed
setup of the 131,072-token mixed-text prefix; none of the timed requests
overlaps it.

## Concurrent requests

Extension 0.3.0 on Gufo 0.34.0 (November 13) against llama.cpp
(November 11), both with LiLiCorr and four slots: Gufo `serve llm
--sessions 4 --context 32768`, llama.cpp `-np 4 -c 131072` (32,768 tokens
per slot) with `--spec-type draft-dflash --spec-draft-n-max 7`. The requests
of each mix start together and write up to 384 tokens
([tools/bench/concurrent_bench.py](../../../tools/bench/concurrent_bench.py));
tokens of all requests per second of wall time, mean of two runs after a
warm-up.

| Mix | Gufo (tok/s) | llama.cpp (tok/s) | Gain |
| --- | ---: | ---: | ---: |
| Fibonacci function alone (code) | 115.9 | 67.6 | +71.4% |
| Lighthouse essay alone (prose) | 54.7 | 28.3 | +93.3% |
| Fibonacci function, ring buffer class (code, code) | 91.6 | 49.5 | +85.3% |
| Lighthouse essay, TCP explanation (prose, prose) | 58.2 | 27.8 | +109.0% |
| Fibonacci function, lighthouse essay (code, prose) | 66.9 | 33.4 | +100.3% |
| Code, code, JSON schema and record | 95.6 | 55.3 | +72.9% |
| Code, code, prose, prose | 83.2 | 37.2 | +124.0% |

Gufo's answers repeat exactly between runs and equal those of 0.2.3, whose
rates were within 1 %. llama.cpp's Fibonacci answer ends after 232 to 280
tokens depending on the run (Gufo's after 231); the rates count every
generated token. Raw results: [both engines](artifacts/concurrency-reference.jsonl).

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
