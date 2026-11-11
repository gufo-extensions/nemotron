#include "vision_encoder.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kernels/rocm/vision_gemm.hpp"
#include "weights.hpp"

namespace gufo::models::nemotron {
namespace {

using rocm::VisionOperand;

/// Encoding time after which the encoder lets other sessions decode, as long
/// as Gufo's scheduler aims to keep a decoder waiting for one layer group.
constexpr std::chrono::milliseconds kYieldInterval{250};

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(status));
}

/// Device memory that only grows.
template<typename T>
class Scratch {
public:
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  T* Reserve(std::size_t count) {
    if (count > count_) {
      if (data_ != nullptr)
        (void)hipFree(data_);
      data_ = nullptr;
      count_ = 0;
      Check(hipMalloc(&data_, count * sizeof(T)), "vision scratch allocation");
      count_ = count;
    }
    return data_;
  }
  [[nodiscard]] std::size_t Bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

struct DeviceLayer {
  const float* norm1{nullptr};
  const float* norm1_b{nullptr};
  const std::uint16_t* qkv{nullptr};  ///< [3 width][width]: q, k, v
  const float* qkv_b{nullptr};        ///< [3 width]
  const std::uint16_t* out{nullptr};
  const float* out_b{nullptr};
  const float* norm2{nullptr};
  const float* norm2_b{nullptr};
  const std::uint16_t* up{nullptr};
  const float* up_b{nullptr};
  const std::uint16_t* down{nullptr};
  const float* down_b{nullptr};
};

}  // namespace

VisionEmbedding::~VisionEmbedding() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

struct VisionEncoder::Impl {
  VisionConfig config;
  VisionOperand operand{VisionOperand::kBf16};
  std::string identity;
  std::vector<void*> allocations;
  std::size_t resident_bytes{0};
  // Device weights.
  const std::uint16_t* patch{nullptr};  ///< [width][PatchValues] operand
  const float* position{nullptr};
  const float* prefix{nullptr};
  std::vector<DeviceLayer> layers;
  const float* projector_norm{nullptr};
  const std::uint16_t* projector_up{nullptr};
  const std::uint16_t* projector_down{nullptr};

  std::mutex mutex;
  hipStream_t stream{nullptr};
  std::unique_ptr<rocm::VisionGemm> gemm;
  Scratch<float> x;            ///< [tokens][width] residual stream
  Scratch<std::uint16_t> h;    ///< [tokens][width] operand rows
  Scratch<float> y;            ///< [tokens][ff_size] product outputs
  Scratch<std::uint16_t> big;  ///< [tokens][ff_size] operand rows

  ~Impl() {
    gemm.reset();
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    for (void* p : allocations)
      (void)hipFree(p);
  }

  void Upload(const VisionWeights& w, const core::GgufReader& reader);
  std::shared_ptr<const VisionEmbedding> Run(
      std::span<const float> patches, const ImageGrid& grid,
      const Observer& observer, const std::function<void()>& yield);
  void Observe(const Observer& observer, std::string_view stage,
               const float* device, std::size_t count) {
    if (!observer)
      return;
    std::vector<float> host(count);
    Check(hipMemcpyAsync(host.data(), device, count * sizeof(float),
                         hipMemcpyDeviceToHost, stream),
          "vision observer copy");
    Check(hipStreamSynchronize(stream), "vision observer copy");
    observer(stage, host);
  }
};

void VisionEncoder::Impl::Upload(const VisionWeights& w,
                                 const core::GgufReader& reader) {
  const VisionConfig& c = config;
  const std::size_t width = c.width;
  // Every tensor goes up in its file format; the patch projection is F32 and
  // becomes an operand afterwards.
  std::size_t total = w.patch.SizeBytes() + w.position.SizeBytes() +
                      w.prefix.SizeBytes() + w.projector_norm.SizeBytes() +
                      w.projector_up.SizeBytes() + w.projector_down.SizeBytes();
  for (const auto& l : w.layers) {
    for (const TensorRef* t : {&l.norm1, &l.norm1_b, &l.q, &l.q_b, &l.k, &l.k_b,
                               &l.v, &l.v_b, &l.out, &l.out_b, &l.norm2,
                               &l.norm2_b, &l.up, &l.up_b, &l.down, &l.down_b})
      total += t->SizeBytes();
  }
  std::string error;
  auto stager = hip::WeightUpload::Create(reader.GetMappedRegions(), &error);
  if (!stager)
    throw std::runtime_error(error);
  const hip::WeightAllocator allocator(total);
  const auto allocate = [&](std::size_t bytes) {
    void* p = nullptr;
    if (allocator.Allocate(&p, bytes) != hipSuccess)
      throw std::runtime_error("vision weight allocation failed");
    allocations.push_back(p);
    resident_bytes += bytes;
    return p;
  };
  const auto copy = [&](const TensorRef& t, void* device) {
    if (!stager->Copy(t.shard, t.file_offset, t.SizeBytes(), device, &error))
      throw std::runtime_error(error);
  };
  const auto single = [&](const TensorRef& t) {
    void* p = allocate(t.SizeBytes());
    copy(t, p);
    return p;
  };
  void* patch_f32 = nullptr;
  Check(hipMalloc(&patch_f32, w.patch.SizeBytes()), "vision upload");
  copy(w.patch, patch_f32);
  position = static_cast<const float*>(single(w.position));
  prefix = static_cast<const float*>(single(w.prefix));
  layers.resize(w.layers.size());
  for (std::size_t il = 0; il < w.layers.size(); ++il) {
    const VisionLayerWeights& l = w.layers[il];
    DeviceLayer& d = layers[il];
    d.norm1 = static_cast<const float*>(single(l.norm1));
    d.norm1_b = static_cast<const float*>(single(l.norm1_b));
    auto* qkv = static_cast<std::uint8_t*>(
        allocate(l.q.SizeBytes() + l.k.SizeBytes() + l.v.SizeBytes()));
    copy(l.q, qkv);
    copy(l.k, qkv + l.q.SizeBytes());
    copy(l.v, qkv + l.q.SizeBytes() + l.k.SizeBytes());
    d.qkv = reinterpret_cast<const std::uint16_t*>(qkv);
    auto* qkv_b = static_cast<std::uint8_t*>(
        allocate(l.q_b.SizeBytes() + l.k_b.SizeBytes() + l.v_b.SizeBytes()));
    copy(l.q_b, qkv_b);
    copy(l.k_b, qkv_b + l.q_b.SizeBytes());
    copy(l.v_b, qkv_b + l.q_b.SizeBytes() + l.k_b.SizeBytes());
    d.qkv_b = reinterpret_cast<const float*>(qkv_b);
    d.out = static_cast<const std::uint16_t*>(single(l.out));
    d.out_b = static_cast<const float*>(single(l.out_b));
    d.norm2 = static_cast<const float*>(single(l.norm2));
    d.norm2_b = static_cast<const float*>(single(l.norm2_b));
    d.up = static_cast<const std::uint16_t*>(single(l.up));
    d.up_b = static_cast<const float*>(single(l.up_b));
    d.down = static_cast<const std::uint16_t*>(single(l.down));
    d.down_b = static_cast<const float*>(single(l.down_b));
  }
  projector_norm = static_cast<const float*>(single(w.projector_norm));
  projector_up = static_cast<const std::uint16_t*>(single(w.projector_up));
  projector_down = static_cast<const std::uint16_t*>(single(w.projector_down));
  auto* patch_op = static_cast<std::uint16_t*>(
      allocate(width * c.PatchValues() * sizeof(std::uint16_t)));
  if (!stager->Finish(&error)) {
    (void)hipFree(patch_f32);
    throw std::runtime_error(error);
  }
  // The operands in the encoder's format; the BF16 matrices stay as loaded
  // for BF16 operands.
  const auto convert = [&](const std::uint16_t* matrix, std::size_t count) {
    Check(rocm::VisionBf16ToOperand(matrix, const_cast<std::uint16_t*>(matrix),
                                    count, operand, stream),
          "vision weight conversion");
  };
  Check(rocm::VisionToOperand(static_cast<const float*>(patch_f32), patch_op,
                              width * c.PatchValues(), operand, stream),
        "vision weight conversion");
  for (const DeviceLayer& d : layers) {
    convert(d.qkv, 3 * width * width);
    convert(d.out, width * width);
    convert(d.up, width * c.ff_size);
    convert(d.down, width * c.ff_size);
  }
  convert(projector_up,
          static_cast<std::size_t>(c.MergedWidth()) * c.projector_hidden);
  convert(projector_down,
          static_cast<std::size_t>(c.projector_hidden) * c.projection);
  const hipError_t synced = hipStreamSynchronize(stream);
  (void)hipFree(patch_f32);
  Check(synced, "vision weight conversion");
  patch = patch_op;
}

std::shared_ptr<const VisionEmbedding> VisionEncoder::Impl::Run(
    std::span<const float> patches, const ImageGrid& grid,
    const Observer& observer, const std::function<void()>& yield) {
  const VisionConfig& c = config;
  const int width = static_cast<int>(c.width);
  const int ff = static_cast<int>(c.ff_size);
  const std::size_t n = grid.Patches();
  if (grid.rows % c.merge != 0 || grid.cols % c.merge != 0 || n == 0 ||
      n > c.max_patches || patches.size() != n * c.PatchValues())
    throw std::invalid_argument("invalid vision encoder input");
  const int tokens = static_cast<int>(c.prefix_tokens + n);
  const std::size_t merged = n / (c.merge * c.merge);
  const std::size_t big_count = static_cast<std::size_t>(tokens) * ff;
  float* xs = x.Reserve(static_cast<std::size_t>(tokens) * width);
  std::uint16_t* hs = h.Reserve(static_cast<std::size_t>(tokens) * width);
  float* ys = y.Reserve(big_count);
  std::uint16_t* bs = big.Reserve(big_count);

  // Patch projection into the rows after the prefix, then positions.
  Check(hipMemcpyAsync(ys, patches.data(), patches.size_bytes(),
                       hipMemcpyHostToDevice, stream),
        "vision patch upload");
  Check(rocm::VisionToOperand(ys, bs, patches.size(), operand, stream),
        "vision patches");
  gemm->Linear(bs, patch,
               xs + static_cast<std::size_t>(c.prefix_tokens) * width, n,
               c.PatchValues(), width);
  Check(rocm::VisionEmbed(
            xs, prefix, position, static_cast<int>(c.prefix_tokens),
            static_cast<int>(grid.rows), static_cast<int>(grid.cols),
            static_cast<int>(c.position_side), width, stream),
        "vision embedding");
  Observe(observer, "embed", xs, static_cast<std::size_t>(tokens) * width);

  const float scale = 1.0F / std::sqrt(static_cast<float>(c.HeadDim()));
  auto since_yield = std::chrono::steady_clock::now();
  for (std::size_t il = 0; il < layers.size(); ++il) {
    if (yield && il > 0) {
      // Between blocks the stream holds nothing but this image's rows.
      Check(hipStreamSynchronize(stream), "vision encoder");
      if (std::chrono::steady_clock::now() - since_yield >= kYieldInterval) {
        yield();
        since_yield = std::chrono::steady_clock::now();
      }
    }
    const DeviceLayer& l = layers[il];
    Check(rocm::VisionLayerNorm(xs, l.norm1, l.norm1_b, hs, tokens, width,
                                c.eps, operand, stream),
          "vision norm");
    gemm->Linear(hs, l.qkv, ys, tokens, width, 3 * width);
    Check(rocm::VisionBias(ys, l.qkv_b, bs, tokens, 3 * width, operand, stream),
          "vision qkv");
    Check(rocm::VisionAttention(bs, hs, tokens, static_cast<int>(c.num_heads),
                                scale, operand, stream),
          "vision attention");
    gemm->Linear(hs, l.out, ys, tokens, width, width);
    Check(rocm::VisionResidual(xs, ys, l.out_b, tokens, width, stream),
          "vision residual");
    Check(rocm::VisionLayerNorm(xs, l.norm2, l.norm2_b, hs, tokens, width,
                                c.eps, operand, stream),
          "vision norm");
    gemm->Linear(hs, l.up, ys, tokens, width, ff);
    Check(rocm::VisionBiasGelu(ys, l.up_b, bs, tokens, ff, operand, stream),
          "vision mlp");
    gemm->Linear(bs, l.down, ys, tokens, ff, width);
    Check(rocm::VisionResidual(xs, ys, l.down_b, tokens, width, stream),
          "vision residual");
    if (observer)
      Observe(observer, "block." + std::to_string(il), xs,
              static_cast<std::size_t>(tokens) * width);
  }

  // The projector: merged patch features, RMSNorm, ReLU² MLP.
  Check(rocm::VisionMergeNorm(
            xs, projector_norm, hs, static_cast<int>(c.prefix_tokens),
            static_cast<int>(grid.rows), static_cast<int>(grid.cols), width,
            c.projector_eps, operand, stream),
        "vision merge");
  gemm->Linear(hs, projector_up, ys, merged, c.MergedWidth(),
               c.projector_hidden);
  Check(rocm::VisionRelu2(ys, bs, merged * c.projector_hidden, operand, stream),
        "vision projector");
  std::shared_ptr<VisionEmbedding> embedding(new VisionEmbedding());
  Check(hipMalloc(&embedding->data_, merged * c.projection * sizeof(float)),
        "vision embedding allocation");
  embedding->rows_ = static_cast<std::uint32_t>(merged);
  embedding->width_ = c.projection;
  gemm->Linear(bs, projector_down, embedding->data_, merged, c.projector_hidden,
               c.projection);
  Check(hipStreamSynchronize(stream), "vision encoder");
  Observe(observer, "output", embedding->data_, merged * c.projection);
  // The pass that takes these rows follows at once; without a round here the
  // encoder's tail and that pass's first layer group add up.
  if (yield)
    yield();
  return embedding;
}

VisionEncoder::VisionEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

VisionEncoder::~VisionEncoder() = default;

std::unique_ptr<VisionEncoder> VisionEncoder::Load(
    const std::string& path, const Config& trunk,
    const VisionEncoderOptions& options, std::string* error_msg) {
  const auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = "vision projector " + path + ": " + std::move(message);
    return std::unique_ptr<VisionEncoder>{};
  };
  std::string error;
  const auto reader = core::GgufReader::OpenFile(path, &error);
  if (!reader)
    return fail(error);
  auto weights = VisionWeights::Bind(*reader, trunk, &error);
  if (!weights)
    return fail(error);
  auto impl = std::make_unique<Impl>();
  impl->config = weights->config;
  impl->operand = options.operand;
  try {
    impl->identity = core::GgufIdentityHex(*reader);
    Check(hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking),
          "vision stream");
    impl->gemm =
        std::make_unique<rocm::VisionGemm>(impl->operand, impl->stream);
    impl->Upload(*weights, *reader);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  return std::unique_ptr<VisionEncoder>(new VisionEncoder(std::move(impl)));
}

const VisionConfig& VisionEncoder::config() const noexcept {
  return impl_->config;
}

rocm::VisionOperand VisionEncoder::operand() const noexcept {
  return impl_->operand;
}

const std::string& VisionEncoder::identity() const noexcept {
  return impl_->identity;
}

std::size_t VisionEncoder::ResidentBytes() const noexcept {
  return impl_->resident_bytes;
}

std::size_t VisionEncoder::MaxScratchBytes() const noexcept {
  const VisionConfig& c = impl_->config;
  const std::size_t tokens = c.prefix_tokens + c.max_patches;
  const std::size_t merged = c.max_patches / (c.merge * c.merge);
  return tokens * c.width * (sizeof(float) + sizeof(std::uint16_t)) +
         tokens * c.ff_size * (sizeof(float) + sizeof(std::uint16_t)) +
         merged * c.projection * sizeof(float);
}

bool VisionEncoder::Prepare(std::string* error_msg) {
  const std::lock_guard lock(impl_->mutex);
  const VisionConfig& c = impl_->config;
  try {
    // ViT products have at least min_patches rows; every class from there up
    // shares one timing. The projector's rows start at a quarter of that.
    const std::size_t tokens = c.prefix_tokens + c.min_patches;
    auto& gemm = *impl_->gemm;
    for (const std::size_t rows : {std::size_t{c.min_patches}, tokens})
      gemm.Prepare(rows, c.PatchValues(), c.width);
    gemm.Prepare(tokens, c.width, 3 * c.width);
    gemm.Prepare(tokens, c.width, c.width);
    gemm.Prepare(tokens, c.width, c.ff_size);
    gemm.Prepare(tokens, c.ff_size, c.width);
    for (std::size_t rows = c.min_patches / (c.merge * c.merge);
         rows <= 2 * std::size_t{c.max_patches} / (c.merge * c.merge);
         rows *= 2) {
      gemm.Prepare(rows, c.MergedWidth(), c.projector_hidden);
      gemm.Prepare(rows, c.projector_hidden, c.projection);
    }
    Check(hipStreamSynchronize(impl_->stream), "vision preparation");
  } catch (const std::exception& e) {
    if (error_msg != nullptr)
      *error_msg = e.what();
    return false;
  }
  return true;
}

std::shared_ptr<const VisionEmbedding> VisionEncoder::Encode(
    std::span<const float> patches, const ImageGrid& grid,
    std::string* error_msg, const Observer& observer,
    const std::function<void()>& yield) {
  const std::lock_guard lock(impl_->mutex);
  try {
    return impl_->Run(patches, grid, observer, yield);
  } catch (const std::exception& e) {
    if (error_msg != nullptr)
      *error_msg = std::string("vision encoder: ") + e.what();
    (void)hipStreamSynchronize(impl_->stream);
    return nullptr;
  }
}

}  // namespace gufo::models::nemotron
