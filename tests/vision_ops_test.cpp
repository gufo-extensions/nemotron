// The vision encoder's kernels against CPU evaluations of the same formulas
// on random inputs, in both operand formats.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "kernels/rocm/vision.hpp"

namespace rocm = gufo::models::nemotron::rocm;
using rocm::VisionOperand;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void Hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << "FAIL: " << what << ": " << hipGetErrorString(status) << '\n';
    std::exit(1);
  }
}

template<typename T>
struct Device {
  T* data{nullptr};
  std::size_t count{0};
  explicit Device(std::size_t n) : count(n) {
    Hip(hipMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc");
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    Hip(hipMemcpy(data, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice),
        "upload");
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<T> Read() const {
    std::vector<T> host(count);
    Hip(hipDeviceSynchronize(), "sync");
    Hip(hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost),
        "download");
    return host;
  }
};

std::uint16_t Encode(float x, VisionOperand op) {
  if (op == VisionOperand::kBf16) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
  }
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(x));
}

float Decode(std::uint16_t v, VisionOperand op) {
  if (op == VisionOperand::kBf16) {
    const std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
    float x = 0.0F;
    std::memcpy(&x, &u, sizeof(x));
    return x;
  }
  return static_cast<float>(std::bit_cast<_Float16>(v));
}

/// Unit roundoff of the operand format: the largest relative error of
/// rounding to nearest.
double Epsilon(VisionOperand op) {
  return op == VisionOperand::kBf16 ? 1.0 / 256 : 1.0 / 2048;
}

const char* Name(VisionOperand op) {
  return op == VisionOperand::kBf16 ? "bf16" : "f16";
}

std::mt19937 rng(20261109);

std::vector<float> Random(std::size_t n, float scale = 1.0F) {
  std::normal_distribution<float> normal(0.0F, scale);
  std::vector<float> v(n);
  for (auto& x : v)
    x = normal(rng);
  return v;
}

/// |got - want| <= tolerance * (|want| + floor) everywhere.
void Compare(const std::string& what, const std::vector<float>& got,
             const std::vector<double>& want, double tolerance,
             double floor = 1e-3) {
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i)
    worst = std::max(worst,
                     std::abs(got[i] - want[i]) / (std::abs(want[i]) + floor));
  std::cout << what << ": worst relative error " << worst << '\n';
  Expect(worst <= tolerance, what + " exceeds " + std::to_string(tolerance));
}

void TestAttention(VisionOperand op, int tokens) {
  const int heads = 16;
  const int d = rocm::kVisionHeadDim;
  const int width = heads * d;
  std::vector<std::uint16_t> qkv(static_cast<std::size_t>(tokens) * 3 * width);
  const auto values = Random(qkv.size(), 1.0F);
  for (std::size_t i = 0; i < qkv.size(); ++i)
    qkv[i] = Encode(values[i], op);
  Device<std::uint16_t> in(qkv);
  Device<std::uint16_t> out(static_cast<std::size_t>(tokens) * width);
  const float scale = 1.0F / std::sqrt(static_cast<float>(d));
  Hip(rocm::VisionAttention(in.data, out.data, tokens, heads, scale, op,
                            nullptr),
      "VisionAttention");
  const auto raw = out.Read();
  std::vector<float> got(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i)
    got[i] = Decode(raw[i], op);
  std::vector<double> want(got.size());
  const auto at = [&](int t, int part, int h, int e) {
    return static_cast<double>(Decode(
        qkv[static_cast<std::size_t>(t) * 3 * width + part * width + h * d + e],
        op));
  };
  std::vector<double> scores(tokens);
  for (int h = 0; h < heads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      double m = -INFINITY;
      for (int k = 0; k < tokens; ++k) {
        double s = 0.0;
        for (int e = 0; e < d; ++e)
          s += at(t, 0, h, e) * at(k, 1, h, e);
        scores[k] = s * scale;
        m = std::max(m, scores[k]);
      }
      double sum = 0.0;
      for (int k = 0; k < tokens; ++k)
        sum += scores[k] = std::exp(scores[k] - m);
      for (int e = 0; e < d; ++e) {
        double o = 0.0;
        for (int k = 0; k < tokens; ++k)
          o += scores[k] * at(k, 2, h, e);
        want[static_cast<std::size_t>(t) * width + h * d + e] = o / sum;
      }
    }
  }
  // Rounded probabilities and outputs: a few units of the format, relative
  // to the outputs' scale.
  double scale_ref = 0.0;
  for (const double w : want)
    scale_ref = std::max(scale_ref, std::abs(w));
  Compare(std::string("attention ") + Name(op) + " tokens " +
              std::to_string(tokens),
          got, want, 4 * Epsilon(op), 0.25 * scale_ref);
}

void TestLayerNorm(VisionOperand op) {
  const int rows = 37;
  const int width = 1280;
  const auto x = Random(static_cast<std::size_t>(rows) * width, 3.0F);
  const auto w = Random(width);
  const auto b = Random(width, 0.1F);
  Device<float> dx(x), dw(w), db(b);
  Device<std::uint16_t> out(x.size());
  Hip(rocm::VisionLayerNorm(dx.data, dw.data, db.data, out.data, rows, width,
                            1e-6F, op, nullptr),
      "VisionLayerNorm");
  const auto raw = out.Read();
  std::vector<float> got(raw.size());
  std::vector<double> want(raw.size());
  for (int r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int c = 0; c < width; ++c)
      mean += x[r * width + c];
    mean /= width;
    double var = 0.0;
    for (int c = 0; c < width; ++c)
      var += (x[r * width + c] - mean) * (x[r * width + c] - mean);
    const double rstd = 1.0 / std::sqrt(var / width + 1e-6);
    for (int c = 0; c < width; ++c) {
      want[r * width + c] = (x[r * width + c] - mean) * rstd * w[c] + b[c];
      got[r * width + c] = Decode(raw[r * width + c], op);
    }
  }
  Compare(std::string("layer norm ") + Name(op), got, want, Epsilon(op), 1e-2);
}

/// PyTorch's bilinear source index (half-pixel centers, clamped at zero).
void Source(float scale, int dst, int size, int* i0, int* i1, double* l1) {
  float src = scale * (static_cast<float>(dst) + 0.5F) - 0.5F;
  src = std::max(src, 0.0F);
  *i0 = std::min(static_cast<int>(src), size - 1);
  *i1 = *i0 + (*i0 < size - 1 ? 1 : 0);
  *l1 = std::clamp(static_cast<double>(src) - *i0, 0.0, 1.0);
}

void TestEmbed(int grid_rows, int grid_cols) {
  const int side = 128;
  const int width = 64;
  const int prefix = 10;
  const int tokens = prefix + grid_rows * grid_cols;
  const auto positions = Random(static_cast<std::size_t>(side) * side * width);
  const auto prefix_rows = Random(static_cast<std::size_t>(prefix) * width);
  const auto x = Random(static_cast<std::size_t>(tokens) * width);
  Device<float> dp(positions), dprefix(prefix_rows), dx(x);
  Hip(rocm::VisionEmbed(dx.data, dprefix.data, dp.data, prefix, grid_rows,
                        grid_cols, side, width, nullptr),
      "VisionEmbed");
  const auto got = dx.Read();
  std::vector<double> want(got.size());
  const float scale = static_cast<float>(side) /
                      static_cast<float>(std::max(grid_rows, grid_cols));
  for (int t = 0; t < tokens; ++t) {
    for (int c = 0; c < width; ++c) {
      const std::size_t i = static_cast<std::size_t>(t) * width + c;
      if (t < prefix) {
        want[i] = prefix_rows[i];
        continue;
      }
      const int p = t - prefix;
      int y0, y1, x0, x1;
      double ly1, lx1;
      Source(scale, p / grid_cols, side, &y0, &y1, &ly1);
      Source(scale, p % grid_cols, side, &x0, &x1, &lx1);
      const auto pos = [&](int yy, int xx) {
        return static_cast<double>(
            positions[(static_cast<std::size_t>(yy) * side + xx) * width + c]);
      };
      want[i] = x[i] +
                (1 - ly1) * ((1 - lx1) * pos(y0, x0) + lx1 * pos(y0, x1)) +
                ly1 * ((1 - lx1) * pos(y1, x0) + lx1 * pos(y1, x1));
    }
  }
  Compare(
      "embed " + std::to_string(grid_rows) + "x" + std::to_string(grid_cols),
      got, want, 1e-5, 1.0);
}

void TestMergeNorm(VisionOperand op) {
  const int prefix = 10;
  const int grid_rows = 6;
  const int grid_cols = 10;
  const int width = 1280;
  const int merged_width = 4 * width;
  const auto x = Random(
      static_cast<std::size_t>(prefix + grid_rows * grid_cols) * width, 2.0F);
  const auto w = Random(merged_width);
  Device<float> dx(x), dw(w);
  const int merged = grid_rows * grid_cols / 4;
  Device<std::uint16_t> out(static_cast<std::size_t>(merged) * merged_width);
  Hip(rocm::VisionMergeNorm(dx.data, dw.data, out.data, prefix, grid_rows,
                            grid_cols, width, 1e-5F, op, nullptr),
      "VisionMergeNorm");
  const auto raw = out.Read();
  std::vector<float> got(raw.size());
  std::vector<double> want(raw.size());
  for (int m = 0; m < merged; ++m) {
    const int i = m / (grid_cols / 2);
    const int j = m % (grid_cols / 2);
    std::vector<double> v(merged_width);
    double squares = 0.0;
    for (int k = 0; k < merged_width; ++k) {
      const int q = k / width;
      const int patch = (2 * i + q / 2) * grid_cols + 2 * j + q % 2;
      v[k] = x[static_cast<std::size_t>(prefix + patch) * width + k % width];
      squares += v[k] * v[k];
    }
    const double r = 1.0 / std::sqrt(squares / merged_width + 1e-5);
    for (int k = 0; k < merged_width; ++k) {
      want[static_cast<std::size_t>(m) * merged_width + k] = w[k] * v[k] * r;
      got[static_cast<std::size_t>(m) * merged_width + k] =
          Decode(raw[static_cast<std::size_t>(m) * merged_width + k], op);
    }
  }
  Compare(std::string("merge norm ") + Name(op), got, want, Epsilon(op), 1e-2);
}

void TestElementwise(VisionOperand op) {
  const int rows = 19;
  const int cols = 384;
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const auto y = Random(n, 2.0F);
  const auto bias = Random(cols, 0.5F);
  const auto x = Random(n);
  Device<float> dy(y), dbias(bias), dx(x);
  Device<std::uint16_t> out(n);
  std::vector<float> got(n);
  std::vector<double> want(n);
  const auto decode = [&] {
    const auto raw = out.Read();
    for (std::size_t i = 0; i < n; ++i)
      got[i] = Decode(raw[i], op);
  };
  Hip(rocm::VisionBias(dy.data, dbias.data, out.data, rows, cols, op, nullptr),
      "VisionBias");
  decode();
  for (std::size_t i = 0; i < n; ++i)
    want[i] = static_cast<double>(y[i]) + bias[i % cols];
  Compare(std::string("bias ") + Name(op), got, want, Epsilon(op), 0.0);
  Hip(rocm::VisionBiasGelu(dy.data, dbias.data, out.data, rows, cols, op,
                           nullptr),
      "VisionBiasGelu");
  decode();
  for (std::size_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(y[i]) + bias[i % cols];
    want[i] = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
  }
  Compare(std::string("bias gelu ") + Name(op), got, want, Epsilon(op), 1e-3);
  Hip(rocm::VisionRelu2(dy.data, out.data, n, op, nullptr), "VisionRelu2");
  decode();
  for (std::size_t i = 0; i < n; ++i) {
    const double v = std::max(0.0, static_cast<double>(y[i]));
    want[i] = v * v;
  }
  // FP16 squares below its normal range lose precision.
  Compare(std::string("relu2 ") + Name(op), got, want, Epsilon(op), 1e-4);
  Hip(rocm::VisionResidual(dx.data, dy.data, dbias.data, rows, cols, nullptr),
      "VisionResidual");
  const auto residual = dx.Read();
  for (std::size_t i = 0; i < n; ++i)
    want[i] = static_cast<double>(x[i]) + y[i] + bias[i % cols];
  Compare("residual", residual, want, 1e-6, 1.0);
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cerr << "SKIP: no HIP device\n";
    return 77;
  }
  for (const VisionOperand op : {VisionOperand::kBf16, VisionOperand::kF16}) {
    for (const int tokens : {37, 150, 1034})
      TestAttention(op, tokens);
    TestLayerNorm(op);
    TestMergeNorm(op);
    TestElementwise(op);
  }
  TestEmbed(32, 32);
  TestEmbed(68, 120);
  TestEmbed(14, 188);
  TestEmbed(128, 104);
  if (failures != 0) {
    std::cerr << failures << " checks failed\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
