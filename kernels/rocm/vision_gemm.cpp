// Adapted from the OmniVoice extension's hipBLASLt wrapper, whose plan
// cache follows Gufo's Qwen3-ASR hipBLASLt wrapper (MIT, see
// licenses/gufo.txt); algorithms are timed instead of taken by rank.
#include "kernels/rocm/vision_gemm.hpp"

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace gufo::models::nemotron::rocm {
namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
}

void Check(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + " failed (status " +
                             std::to_string(static_cast<int>(status)) + ")");
}

/// Device memory that only grows.
class Scratch {
public:
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  void* Reserve(std::size_t bytes) {
    if (bytes > bytes_) {
      if (data_ != nullptr)
        (void)hipFree(data_);
      data_ = nullptr;
      bytes_ = 0;
      Check(hipMalloc(&data_, bytes), "hipMalloc");
      bytes_ = bytes;
    }
    return data_;
  }
  [[nodiscard]] void* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

/// y[rows][out] = x[rows][in] * w[out][in]^T as a column-major product.
struct Shape {
  std::size_t rows{0};
  std::size_t in{0};
  std::size_t out{0};
  bool operator<(const Shape& other) const {
    return std::tie(rows, in, out) < std::tie(other.rows, other.in, other.out);
  }
};

struct Plan {
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatmulAlgo_t algorithm{};
  std::size_t workspace_bytes = 0;
  bool usable = false;

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    if (c_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(c_layout);
    if (b_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(b_layout);
    if (a_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    if (operation != nullptr)
      (void)hipblasLtMatmulDescDestroy(operation);
  }
};

struct Choice {
  hipblasLtMatmulAlgo_t algorithm{};
  bool usable = false;
};

// Workspace for algorithms that split the reduction.
constexpr std::size_t kWorkspaceBytes = 32U << 20U;
constexpr int kCandidates = 16;
constexpr int kRepeats = 2;
/// Rows of the shape a class is timed at: classes of more rows share it.
constexpr std::size_t kTuningRows = 2048;

/// The shape a product is timed at: rows rounded up to a power of two.
Shape Representative(const Shape& shape) {
  std::size_t rows = 128;
  while (rows < shape.rows && rows < kTuningRows)
    rows *= 2;
  return {rows, shape.in, shape.out};
}

}  // namespace

struct VisionGemm::State {
  hipDataType type = HIP_R_16BF;
  hipStream_t stream = nullptr;
  hipblasLtHandle_t handle = nullptr;
  hipblasHandle_t fallback = nullptr;
  Scratch workspace;
  Scratch tune_a, tune_b, tune_c;
  std::map<Shape, std::unique_ptr<Plan>> plans;
  std::map<Shape, Choice> tuned;

  ~State() {
    plans.clear();
    if (handle != nullptr)
      (void)hipblasLtDestroy(handle);
    if (fallback != nullptr)
      (void)hipblasDestroy(fallback);
  }

  bool Describe(Plan* plan, const Shape& s) const {
    if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                  HIP_R_32F) != HIPBLAS_STATUS_SUCCESS)
      return false;
    const hipblasOperation_t transpose = HIPBLAS_OP_T;
    return hipblasLtMatmulDescSetAttribute(
               plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
               sizeof(transpose)) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan->a_layout, type, s.in, s.out,
                                       static_cast<std::int64_t>(s.in)) ==
               HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan->b_layout, type, s.in, s.rows,
                                       static_cast<std::int64_t>(s.in)) ==
               HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(
               &plan->c_layout, HIP_R_32F, s.out, s.rows,
               static_cast<std::int64_t>(s.out)) == HIPBLAS_STATUS_SUCCESS;
  }

  /// Times hipBLASLt's candidates for `s` on zeroed operands; cached.
  Choice Tune(const Shape& s) {
    if (const auto found = tuned.find(s); found != tuned.end())
      return found->second;
    Choice choice;
    Plan plan;
    if (!Describe(&plan, s)) {
      tuned.emplace(s, choice);
      return choice;
    }
    const std::size_t a_bytes = s.in * s.out * 2;
    const std::size_t b_bytes = s.in * s.rows * 2;
    void* a = tune_a.Reserve(a_bytes);
    void* b = tune_b.Reserve(b_bytes);
    void* c = tune_c.Reserve(s.out * s.rows * sizeof(float));
    Check(hipMemsetAsync(a, 0, a_bytes, stream), "hipMemsetAsync");
    Check(hipMemsetAsync(b, 0, b_bytes, stream), "hipMemsetAsync");
    hipblasLtMatmulPreference_t preference = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&preference) !=
        HIPBLAS_STATUS_SUCCESS) {
      tuned.emplace(s, choice);
      return choice;
    }
    const std::size_t capacity = workspace.size();
    (void)hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &capacity,
        sizeof(capacity));
    hipblasLtMatmulHeuristicResult_t results[kCandidates];
    int found = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(
        handle, plan.operation, plan.a_layout, plan.b_layout, plan.c_layout,
        plan.c_layout, preference, kCandidates, results, &found);
    (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS || found <= 0) {
      tuned.emplace(s, choice);
      return choice;
    }
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    Check(hipEventCreate(&begin), "hipEventCreate");
    Check(hipEventCreate(&end), "hipEventCreate");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    float best = 0.0F;
    for (int index = 0; index < found; ++index) {
      const auto& candidate = results[index];
      if (candidate.state != HIPBLAS_STATUS_SUCCESS ||
          candidate.workspaceSize > capacity)
        continue;
      const auto run = [&] {
        return hipblasLtMatmul(handle, plan.operation, &alpha, a, plan.a_layout,
                               b, plan.b_layout, &beta, c, plan.c_layout, c,
                               plan.c_layout, &candidate.algo, workspace.get(),
                               candidate.workspaceSize, stream);
      };
      if (run() != HIPBLAS_STATUS_SUCCESS)
        continue;
      Check(hipEventRecord(begin, stream), "hipEventRecord");
      for (int repeat = 0; repeat < kRepeats; ++repeat)
        (void)run();
      Check(hipEventRecord(end, stream), "hipEventRecord");
      Check(hipEventSynchronize(end), "hipEventSynchronize");
      float milliseconds = 0.0F;
      Check(hipEventElapsedTime(&milliseconds, begin, end),
            "hipEventElapsedTime");
      if (!choice.usable || milliseconds < best) {
        best = milliseconds;
        choice.algorithm = candidate.algo;
        choice.usable = true;
      }
    }
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    tuned.emplace(s, choice);
    return choice;
  }

  /// The exact shape's descriptors with its class's algorithm.
  Plan* Resolve(const Shape& shape) {
    if (const auto found = plans.find(shape); found != plans.end())
      return found->second.get();
    auto owned = std::make_unique<Plan>();
    Plan* plan = owned.get();
    plans.emplace(shape, std::move(owned));
    if (handle == nullptr || !Describe(plan, shape))
      return plan;
    Choice choice = Tune(Representative(shape));
    if (!choice.usable)
      return plan;
    const float alpha = 1.0F;
    const float beta = 0.0F;
    std::size_t bytes = 0;
    if (hipblaslt_ext::matmulIsAlgoSupported(
            handle, plan->operation, &alpha, plan->a_layout, plan->b_layout,
            &beta, plan->c_layout, plan->c_layout, choice.algorithm,
            bytes) != HIPBLAS_STATUS_SUCCESS ||
        bytes > workspace.size())
      return plan;
    plan->algorithm = choice.algorithm;
    plan->workspace_bytes = bytes;
    plan->usable = true;
    return plan;
  }
};

VisionGemm::VisionGemm(VisionOperand operand, hipStream_t stream)
    : state_(std::make_unique<State>()) {
  state_->type = operand == VisionOperand::kBf16 ? HIP_R_16BF : HIP_R_16F;
  state_->stream = stream;
  Check(hipblasCreate(&state_->fallback), "hipblasCreate");
  Check(hipblasSetStream(state_->fallback, stream), "hipblasSetStream");
  if (hipblasLtCreate(&state_->handle) != HIPBLAS_STATUS_SUCCESS) {
    state_->handle = nullptr;
    return;
  }
  state_->workspace.Reserve(kWorkspaceBytes);
}

VisionGemm::~VisionGemm() = default;

void VisionGemm::Prepare(std::size_t rows, std::size_t in, std::size_t out) {
  if (rows != 0)
    (void)state_->Resolve(Shape{rows, in, out});
}

void VisionGemm::Linear(const std::uint16_t* x, const std::uint16_t* w,
                        float* y, std::size_t rows, std::size_t in,
                        std::size_t out) {
  if (rows == 0)
    return;
  const Shape shape{rows, in, out};
  const float alpha = 1.0F;
  const float beta = 0.0F;
  if (Plan* plan = state_->Resolve(shape); plan->usable) {
    Check(hipblasLtMatmul(state_->handle, plan->operation, &alpha, w,
                          plan->a_layout, x, plan->b_layout, &beta, y,
                          plan->c_layout, y, plan->c_layout, &plan->algorithm,
                          state_->workspace.get(), plan->workspace_bytes,
                          state_->stream),
          "hipblasLtMatmul");
    return;
  }
  Check(hipblasGemmEx(
            state_->fallback, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(out),
            static_cast<int>(rows), static_cast<int>(in), &alpha, w,
            state_->type, static_cast<int>(in), x, state_->type,
            static_cast<int>(in), &beta, y, HIP_R_32F, static_cast<int>(out),
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
        "hipblasGemmEx");
}

}  // namespace gufo::models::nemotron::rocm
