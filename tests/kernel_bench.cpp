// SPDX-License-Identifier: MIT
// Times the trunk's products at the model's shapes for 1 to 8 activation
// rows, the short-chunk range of decoding and draft verification, with
// random weights: the router, the dense NVFP4 and Q8_0 products on the
// vector kernels (MMVQ) and on the tiled ones (MMQ, NVFP4 WMMA), the shared
// expert and the routed experts. Every case cycles through enough weight
// copies to miss the caches, as a pass over 88 blocks does.
//
//   nemotron_kernel_bench [--iters N] [--only router,dense,shared,routed]

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "iq2xxs_grid.hpp"
#include "kernels/rocm/gemm.hpp"
#include "kernels/rocm/gemv.hpp"
#include "kernels/rocm/kernels.hpp"
#include "kernels/rocm/mmq/nb_mmq.h"

namespace {

namespace k = gufo::models::nemotron::rocm;

#define HIP_OK(expr)                                                           \
  do {                                                                         \
    const hipError_t err_ = (expr);                                            \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(err_)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define MMQ_OK(expr)                              \
  do {                                            \
    if ((expr) != 0) {                            \
      std::fprintf(stderr, "%s failed\n", #expr); \
      std::exit(1);                               \
    }                                             \
  } while (0)

constexpr std::size_t kCacheBytes = 384ULL << 20U;  ///< Bytes cycled per case.

struct Rng {
  std::uint64_t state;
  std::uint64_t Next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }
  std::uint32_t Below(std::uint32_t n) {
    return static_cast<std::uint32_t>(Next() % n);
  }
};

Rng rng{42};
int iters = 50;
hipStream_t stream;

void* DeviceBytes(const std::vector<std::uint8_t>& h) {
  void* d = nullptr;
  HIP_OK(hipMalloc(&d, h.size()));
  HIP_OK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice));
  return d;
}

/// Random blocks of a weight format with tame scales: `block` bytes per
/// `per` values, the scale bytes set by `scale`.
std::vector<std::uint8_t> RandomBlocks(std::size_t values, int per, int block,
                                       void (*scale)(std::uint8_t*)) {
  std::vector<std::uint8_t> w(values / per * block);
  for (std::uint8_t& b : w)
    b = static_cast<std::uint8_t>(rng.Next());
  for (std::size_t b = 0; b < w.size(); b += block)
    scale(w.data() + b);
  return w;
}

void Q8_0Scale(std::uint8_t* b) {  // fp16 2^-7
  b[0] = 0x00;
  b[1] = 0x20;
}
void Nvfp4Scale(std::uint8_t* b) {  // UE4M3 below 2^1
  for (int i = 0; i < 4; ++i)
    b[i] = static_cast<std::uint8_t>(0x30 + (b[i] & 0x0F));
}
void Iq2xxsScale(std::uint8_t* b) {  // fp16 group scale near 2^-8
  b[0] = static_cast<std::uint8_t>(rng.Next());
  b[1] = 0x1C;
}

/// Copies of one weight tensor, enough to cycle kCacheBytes.
struct Weights {
  std::vector<void*> copies;
  std::size_t bytes{0};
  int next{0};
  const void* Next() {
    next = (next + 1) % static_cast<int>(copies.size());
    return copies[next];
  }
  ~Weights() {
    for (void* p : copies)
      static_cast<void>(hipFree(p));
  }
};

void Fill(Weights& w, std::size_t values, int per, int block,
          void (*scale)(std::uint8_t*)) {
  const auto host = RandomBlocks(values, per, block, scale);
  w.bytes = host.size();
  const std::size_t n = std::max<std::size_t>(2, kCacheBytes / w.bytes + 1);
  for (std::size_t i = 0; i < n; ++i)
    w.copies.push_back(DeviceBytes(host));
}

void FillF32(Weights& w, std::size_t values) {
  std::vector<std::uint8_t> host(values * sizeof(float));
  auto* f = reinterpret_cast<float*>(host.data());
  for (std::size_t i = 0; i < values; ++i)
    f[i] = static_cast<float>(static_cast<int>(rng.Below(2001)) - 1000) * 1e-5F;
  w.bytes = host.size();
  const std::size_t n = std::max<std::size_t>(2, kCacheBytes / w.bytes + 1);
  for (std::size_t i = 0; i < n; ++i)
    w.copies.push_back(DeviceBytes(host));
}

template<class T>
T* Scratch(std::size_t count) {
  T* d = nullptr;
  HIP_OK(hipMalloc(&d, std::max<std::size_t>(count, 1) * sizeof(T)));
  HIP_OK(hipMemset(d, 0, std::max<std::size_t>(count, 1) * sizeof(T)));
  return d;
}

/// Microseconds per call and the weight bytes it streams per microsecond.
void Time(const std::string& label, std::size_t bytes,
          const std::function<void()>& call) {
  for (int i = 0; i < 5; ++i)
    call();
  hipEvent_t start, stop;
  HIP_OK(hipEventCreate(&start));
  HIP_OK(hipEventCreate(&stop));
  HIP_OK(hipEventRecord(start, stream));
  for (int i = 0; i < iters; ++i)
    call();
  HIP_OK(hipEventRecord(stop, stream));
  HIP_OK(hipEventSynchronize(stop));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, start, stop));
  HIP_OK(hipEventDestroy(start));
  HIP_OK(hipEventDestroy(stop));
  const double us = 1000.0 * ms / iters;
  std::printf("%-52s %9.1f us %7.1f GB/s\n", label.c_str(), us,
              static_cast<double>(bytes) / us / 1000.0);
}

std::string Label(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string Label(const char* fmt, ...) {
  char buf[128];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

constexpr int kH = 4096;

void Router(int experts) {
  Weights w;
  FillF32(w, static_cast<std::size_t>(experts) * kH);
  auto* x = Scratch<float>(8 * kH);
  auto* out = Scratch<float>(8 * experts);
  for (int n : {1, 8})
    Time(Label("router F32 %dx%d, n=%d", experts, kH, n), w.bytes, [&] {
      HIP_OK(k::RouterLogits(static_cast<const float*>(w.Next()), x, out,
                             experts, kH, n, stream));
    });
  HIP_OK(hipFree(x));
  HIP_OK(hipFree(out));
}

/// A dense product: MMVQ on Q8_1 rows for every n, the tiled MMQ path and,
/// for NVFP4, the FP16 WMMA product for the larger n.
void Dense(const char* name, int type, int rows, int cols) {
  Weights w;
  const std::size_t values = static_cast<std::size_t>(rows) * cols;
  if (type == NB_MMQ_NVFP4)
    Fill(w, values, 64, 36, Nvfp4Scale);
  else if (type == NB_MMQ_Q8_0)
    Fill(w, values, 32, 34, Q8_0Scale);
  else
    Fill(w, values, 256, 66, Iq2xxsScale);
  auto* x = Scratch<float>(8 * static_cast<std::size_t>(cols));
  auto* x16 = Scratch<__half>(8 * static_cast<std::size_t>(cols));
  auto* xq = Scratch<std::uint8_t>(8 * cols / 32 * k::kQ8_1BlockBytes);
  auto* out = Scratch<float>(8 * static_cast<std::size_t>(rows));
  for (int n : {1, 2, 4, 8})
    Time(Label("%s mmvq %dx%d, n=%d", name, rows, cols, n), w.bytes, [&] {
      MMQ_OK(nb_mmq_dense_q8_1(type, w.Next(), xq, 1.0F, nullptr, out, rows, n,
                               cols, stream));
    });
  for (int n : {4, 8})
    Time(Label("%s mmq %dx%d, n=%d", name, rows, cols, n), w.bytes, [&] {
      MMQ_OK(nb_mmq_dense_tiled(type, w.Next(), x, out, rows, n, cols, stream));
    });
  if (type == NB_MMQ_NVFP4 && rows % k::kNvfp4GemmRows == 0)
    for (int n : {4, 8})
      Time(Label("%s nvfp4 wmma %dx%d, n=%d", name, rows, cols, n), w.bytes,
           [&] {
             HIP_OK(k::Nvfp4Gemm(w.Next(), x16, out, rows, cols, n, stream));
           });
  HIP_OK(hipFree(x));
  HIP_OK(hipFree(x16));
  HIP_OK(hipFree(xq));
  HIP_OK(hipFree(out));
}

/// The shared expert's two products on the IQ2_XXS vector kernel.
void Shared(const std::uint32_t* grid) {
  constexpr int kFf = 5376;
  Weights up, down;
  Fill(up, static_cast<std::size_t>(kFf) * kH, 256, 66, Iq2xxsScale);
  Fill(down, static_cast<std::size_t>(kFf) * kH, 256, 66, Iq2xxsScale);
  auto* h = Scratch<float>(8 * kH);
  auto* mid = Scratch<float>(8 * kFf);
  auto* x = Scratch<float>(8 * kH);
  for (int n : {1, 2, 4, 8})
    Time(Label("shared expert iq2 gemv up+down, n=%d", n), up.bytes * 2, [&] {
      k::Iq2xxsGemvArgs s{};
      s.grid = grid;
      s.n_tokens = n;
      s.weights = up.Next();
      s.x = h;
      s.out = mid;
      s.rows = kFf;
      s.cols = kH;
      s.relu2 = true;
      HIP_OK(k::Iq2xxsGemv(s, stream));
      s.weights = down.Next();
      s.x = mid;
      s.out = x;
      s.rows = kH;
      s.cols = kFf;
      s.relu2 = false;
      s.accumulate = true;
      HIP_OK(k::Iq2xxsGemv(s, stream));
    });
  auto* hb = Scratch<std::uint16_t>(8 * kH);
  auto* midb = Scratch<std::uint16_t>(8 * kFf);
  for (int n : {1, 2, 3, 4, 6, 8})
    Time(Label("shared expert iq2 bf16 up+down, n=%d", n), up.bytes * 2, [&] {
      HIP_OK(k::ToBf16(h, hb, static_cast<std::size_t>(n) * kH, stream));
      k::Iq2xxsGemvArgs s{};
      s.grid = grid;
      s.n_tokens = n;
      s.weights = up.Next();
      s.x_bf16 = hb;
      s.out_bf16 = midb;
      s.rows = kFf;
      s.cols = kH;
      s.relu2 = true;
      HIP_OK(k::Iq2xxsGemv(s, stream));
      s.weights = down.Next();
      s.x_bf16 = midb;
      s.out_bf16 = nullptr;
      s.out = x;
      s.rows = kH;
      s.cols = kFf;
      s.relu2 = false;
      s.accumulate = true;
      HIP_OK(k::Iq2xxsGemv(s, stream));
    });
  HIP_OK(hipFree(hb));
  HIP_OK(hipFree(midb));
  for (int n : {4, 8})
    Time(Label("shared expert iq2 mmq up+down, n=%d", n), up.bytes * 2, [&] {
      MMQ_OK(nb_mmq_dense_tiled(NB_MMQ_IQ2_XXS, up.Next(), h, mid, kFf, n, kH,
                                stream));
      MMQ_OK(nb_mmq_dense_tiled(NB_MMQ_IQ2_XXS, down.Next(), mid, x, kH, n, kFf,
                                stream));
    });
  HIP_OK(hipFree(h));
  HIP_OK(hipFree(mid));
  HIP_OK(hipFree(x));
}

/// Expert ids for n tokens of `used` distinct experts each, drawing from the
/// previous token's experts with probability `reuse`.
std::vector<std::int32_t> Ids(int n, int used, int experts, double reuse) {
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n) * used);
  for (int t = 0; t < n; ++t) {
    for (int s = 0; s < used; ++s) {
      std::int32_t e;
      const auto begin = ids.begin() + t * used;
      do {
        const bool again =
            t > 0 && static_cast<double>(rng.Below(1000)) < reuse * 1000.0;
        e = again ? ids[(t - 1) * used + rng.Below(used)]
                  : static_cast<std::int32_t>(rng.Below(experts));
      } while (std::find(begin, begin + s, e) != begin + s);
      ids[t * used + s] = e;
    }
  }
  return ids;
}

/// The routed experts' up and down products: the vector kernel, one block
/// per slot, against the prefill product grouped by expert.
void Routed(int experts, int ff, const std::uint32_t* grid) {
  constexpr int kLatent = 1024;
  constexpr int kUsed = 9;
  Weights up, down;
  Fill(up, static_cast<std::size_t>(experts) * ff * kLatent, 256, 66,
       Iq2xxsScale);
  Fill(down, static_cast<std::size_t>(experts) * ff * kLatent, 256, 66,
       Iq2xxsScale);
  const std::size_t expert_bytes = up.bytes / experts * 2;
  auto* latent = Scratch<float>(8 * kLatent);
  auto* mid = Scratch<float>(8 * kUsed * static_cast<std::size_t>(ff));
  auto* out = Scratch<float>(8 * kUsed * kLatent);
  auto* ids = Scratch<std::int32_t>(8 * kUsed);
  auto* latent_b = Scratch<std::uint16_t>(8 * kLatent);
  auto* mid_b =
      Scratch<std::uint16_t>(8 * kUsed * static_cast<std::size_t>(ff));
  k::Iq2xxsRoute route{};
  route.max_tiles = k::Iq2xxsRouteMaxTiles(8 * kUsed, experts);
  route.cursor = Scratch<std::int32_t>(experts);
  route.order = Scratch<std::int32_t>(8 * kUsed);
  route.tiles = Scratch<int4>(route.max_tiles);
  route.num_tiles = Scratch<std::int32_t>(1);
  for (double reuse : {0.0, 0.5}) {
    for (int n : {1, 2, 4, 8}) {
      if (n == 1 && reuse > 0.0)
        continue;
      const auto host = Ids(n, kUsed, experts, reuse);
      HIP_OK(hipMemcpy(ids, host.data(), host.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice));
      auto unique = host;
      std::sort(unique.begin(), unique.end());
      const auto distinct = static_cast<std::size_t>(
          std::unique(unique.begin(), unique.end()) - unique.begin());
      Time(Label("routed %dx%dx%d gemv, n=%d, %zu experts", experts, ff,
                 kLatent, n, distinct),
           expert_bytes * distinct, [&] {
             k::Iq2xxsGemvArgs e{};
             e.grid = grid;
             e.n_tokens = n;
             e.weights = up.Next();
             e.x = latent;
             e.ids = ids;
             e.used = kUsed;
             e.out = mid;
             e.rows = ff;
             e.cols = kLatent;
             e.relu2 = true;
             HIP_OK(k::Iq2xxsGemv(e, stream));
             e.weights = down.Next();
             e.x = mid;
             e.x_per_slot = true;
             e.out = out;
             e.rows = kLatent;
             e.cols = ff;
             e.relu2 = false;
             HIP_OK(k::Iq2xxsGemv(e, stream));
           });
      Time(Label("routed %dx%dx%d tiles, n=%d, %zu experts", experts, ff,
                 kLatent, n, distinct),
           expert_bytes * distinct, [&] {
             HIP_OK(k::ToBf16(latent, latent_b,
                              static_cast<std::size_t>(n) * kLatent, stream));
             HIP_OK(k::Iq2xxsRouteShort(ids, n * kUsed, route, stream));
             k::Iq2xxsGemvArgs e{};
             e.grid = grid;
             e.n_tokens = n;
             e.weights = up.Next();
             e.x_bf16 = latent_b;
             e.ids = ids;
             e.route = &route;
             e.used = kUsed;
             e.out_bf16 = mid_b;
             e.rows = ff;
             e.cols = kLatent;
             e.relu2 = true;
             HIP_OK(k::Iq2xxsGemv(e, stream));
             e.weights = down.Next();
             e.x_bf16 = mid_b;
             e.x_per_slot = true;
             e.out_bf16 = nullptr;
             e.out = out;
             e.rows = kLatent;
             e.cols = ff;
             e.relu2 = false;
             HIP_OK(k::Iq2xxsGemv(e, stream));
           });
      if (n < 4)
        continue;
      Time(
          Label("routed %dx%dx%d gemm, n=%d, %zu experts", experts, ff, kLatent,
                n, distinct),
          expert_bytes * distinct, [&] {
            HIP_OK(k::Iq2xxsRouteSlots(ids, n * kUsed, experts, route, stream));
            k::Iq2xxsGemmArgs e{};
            e.grid = grid;
            e.weights = up.Next();
            e.x = latent;
            e.route = &route;
            e.used = kUsed;
            e.out = mid;
            e.rows = ff;
            e.cols = kLatent;
            e.relu2 = true;
            HIP_OK(k::Iq2xxsRoutedGemm(e, stream));
            e.weights = down.Next();
            e.x = mid;
            e.x_per_slot = true;
            e.out = out;
            e.rows = kLatent;
            e.cols = ff;
            e.relu2 = false;
            HIP_OK(k::Iq2xxsRoutedGemm(e, stream));
          });
    }
  }
  HIP_OK(hipFree(latent));
  HIP_OK(hipFree(mid));
  HIP_OK(hipFree(out));
  HIP_OK(hipFree(ids));
  HIP_OK(hipFree(latent_b));
  HIP_OK(hipFree(mid_b));
  HIP_OK(hipFree(route.cursor));
  HIP_OK(hipFree(route.order));
  HIP_OK(hipFree(route.tiles));
  HIP_OK(hipFree(route.num_tiles));
}

}  // namespace

int main(int argc, char** argv) {
  std::string only;  ///< Comma-separated sections; all when empty.
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
      iters = std::max(1, std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc)
      only = argv[++i];
  }
  if (nb_mmq_init(0) != 0) {
    std::fprintf(stderr, "nb_mmq_init failed\n");
    return 1;
  }
  HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  MMQ_OK(nb_mmq_bind_stream(stream));
  std::vector<std::uint32_t> words(k::kIq2xxsGridWords);
  k::Iq2xxsGridBf16(gufo::models::nemotron::kIq2xxsGrid, words.data());
  auto* grid = Scratch<std::uint32_t>(words.size());
  HIP_OK(hipMemcpy(grid, words.data(), words.size() * sizeof(std::uint32_t),
                   hipMemcpyHostToDevice));

  const auto want = [&](const char* section) {
    return only.empty() || only.find(section) != std::string::npos;
  };
  if (want("router")) {
    Router(496);
    Router(240);
  }
  if (want("dense")) {
    Dense("ssm_in nvfp4", NB_MMQ_NVFP4, 18560, kH);
    Dense("ssm_out nvfp4", NB_MMQ_NVFP4, kH, 8192);
    Dense("attn_q q8_0", NB_MMQ_Q8_0, kH, kH);
    Dense("latent_down q8_0", NB_MMQ_Q8_0, 1024, kH);
    Dense("latent_up q8_0", NB_MMQ_Q8_0, kH, 1024);
    Dense("output q8_0", NB_MMQ_Q8_0, 131072, kH);
  }
  if (want("shared"))
    Shared(grid);
  if (want("routed")) {
    Routed(432, 2560, grid);
    Routed(240, 2048, grid);
  }
  nb_mmq_release_stream(stream);
  HIP_OK(hipStreamDestroy(stream));
  return 0;
}
