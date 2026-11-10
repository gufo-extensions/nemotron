#ifndef GUFO_MODELS_NEMOTRON_KERNELS_ROCM_VISION_GEMM_HPP_
#define GUFO_MODELS_NEMOTRON_KERNELS_ROCM_VISION_GEMM_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <memory>

#include "kernels/rocm/vision.hpp"

namespace gufo::models::nemotron::rocm {

/// Products of 16-bit operands (BF16 or FP16) with float32 accumulation and
/// output on hipBLASLt: y[rows][out] = x[rows][in] * w[out][in]^T. On gfx1151
/// hipBLASLt runs them on the matrix cores, where rocBLAS's mixed-precision
/// GEMM does not, and its ranking of algorithms predicts their speed poorly,
/// so its candidates are timed once per shape class (rows rounded up to a
/// power of two). Shapes hipBLASLt can't serve fall back to hipBLAS. Throws
/// std::runtime_error when the device fails.
class VisionGemm {
public:
  VisionGemm(VisionOperand operand, hipStream_t stream);
  ~VisionGemm();
  VisionGemm(const VisionGemm&) = delete;
  VisionGemm& operator=(const VisionGemm&) = delete;

  void Linear(const std::uint16_t* x, const std::uint16_t* w, float* y,
              std::size_t rows, std::size_t in, std::size_t out);
  /// Times the algorithms of Linear's shape now, before it runs.
  void Prepare(std::size_t rows, std::size_t in, std::size_t out);

private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace gufo::models::nemotron::rocm

#endif  // GUFO_MODELS_NEMOTRON_KERNELS_ROCM_VISION_GEMM_HPP_
