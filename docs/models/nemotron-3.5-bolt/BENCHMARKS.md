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
a request decodes alone; concurrent requests decode together in batches
without drafting.

<!-- bench:single-lilicorr -->
| Nemotron 3.5 Bolt 24GB LiLiCorr<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 999.28 | 445.32 | +124.4% | 41.41 | 26.68 | +55.2% | 53.48 | 34.73 | +54.0% |
| 8,192 | 922.49 | 424.30 | +117.4% | 42.01 | 30.04 | +39.8% | 51.36 | 33.92 | +51.4% |
| 32,768 | 806.67 | 368.47 | +118.9% | 33.23 | 22.06 | +50.6% | 47.95 | 31.78 | +50.9% |
<!-- /bench -->

![Single user, LiLiCorr](artifacts/charts/single-lilicorr.svg)

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
