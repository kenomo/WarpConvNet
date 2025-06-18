// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/numeric_types.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "driver_types.h"
#include "include/gemm_error_codes.h"
#include "include/gemm_mma_tiles.h"

namespace py = pybind11;

// Type mapping from PyTorch scalar types to CUTLASS types
template <torch::ScalarType T>
struct torch_to_cutlass;

template <>
struct torch_to_cutlass<torch::kFloat16> {
  using type = cutlass::half_t;
};
template <>
struct torch_to_cutlass<torch::kFloat32> {
  using type = float;
};
template <>
struct torch_to_cutlass<torch::kFloat64> {
  using type = double;
};

#ifndef DISABLE_BFLOAT16
template <>
struct torch_to_cutlass<torch::kBFloat16> {
  using type = cutlass::bfloat16_t;
};
#endif  // DISABLE_BFLOAT16

// A Gather + D scatter
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename TileTag,
          typename Arch>
int run_cutlass_gemm_ad_gather_scatter(const void *tensor_a,
                                       const void *tensor_b,
                                       const void *tensor_c,
                                       void *tensor_d,
                                       const int *indices_a,
                                       const int *indices_d,
                                       int split_k_slices,
                                       int M_A,  // row of A
                                       int K,    // col of A
                                       int N,    // col of B
                                       int M_C,  // row of C
                                       int gather_ad_size,
                                       float alpha,
                                       float beta);

// AB Gather with A Transpose
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename TileTag,
          typename Arch>
int run_cutlass_gemm_trAB_gather(const void *tensor_a,
                                 const void *tensor_b,
                                 const void *tensor_c,
                                 void *tensor_d,
                                 const int *indices_a,
                                 const int *indices_b,
                                 int split_k_slices,
                                 int M_A,  // row of A (not trA)
                                 int K,    // col of A (not trA)
                                 int K_B,  // row of B (different from K since gathering)
                                 int N,    // col of B
                                 int gather_ab_size,
                                 float alpha,
                                 float beta);

// Forward declaration of specialised SM80 kernel for FP32 input with
// gather/scatter
// int run_f32_to_f16_gemm_gather_scatter_sm80(const float *dA,
//                                             const float *dB,
//                                             const float *dC,
//                                             float *dD,
//                                             const int *gatherA_indices,
//                                             const int *scatterD_indices,
//                                             int split_k_slices,
//                                             int M,
//                                             int N,
//                                             int K,
//                                             int gather_rows,
//                                             int scatter_rows,
//                                             float alpha = 1.F,
//                                             float beta = 0.F,
//                                             cudaStream_t stream = nullptr);

// Helper function to dispatch CUTLASS GEMM with automatic type deduction
template <torch::ScalarType ScalarA,
          torch::ScalarType ScalarB,
          torch::ScalarType ScalarOutput,
          torch::ScalarType ScalarAccumulator>
int dispatch_cutlass_gemm_ad_gather_scatter(const torch::Tensor &tensor_a,
                                            const torch::Tensor &tensor_b,
                                            const torch::Tensor &tensor_c,
                                            torch::Tensor &tensor_d,
                                            const torch::Tensor &indices_a,
                                            const torch::Tensor &indices_d,
                                            int split_k_slices,
                                            int mma_tile,
                                            int M_A,             // row of A
                                            int K,               // col of A
                                            int N,               // col of B
                                            int M_C,             // row of C
                                            int gather_ad_size,  // indices_a and indices_d size
                                            float alpha,
                                            float beta);

template <torch::ScalarType ScalarA,
          torch::ScalarType ScalarB,
          torch::ScalarType ScalarOutput,
          torch::ScalarType ScalarAccumulator>
int dispatch_cutlass_gemm_trAB_gather(const torch::Tensor &tensor_a,
                                      const torch::Tensor &tensor_b,
                                      const torch::Tensor &tensor_c,
                                      torch::Tensor &tensor_d,
                                      const torch::Tensor &indices_a,
                                      const torch::Tensor &indices_b,
                                      int split_k_slices,
                                      int mma_tile,
                                      int M_A,  // row of A (not trA)
                                      int K,    // col of A (not trA)
                                      int K_B,  // row of B (different from K since gathering)
                                      int N,    // col of B
                                      int gather_ab_size,
                                      float alpha,
                                      float beta);

// Helper to transform GEMM status into a Python error if not successful
inline void assert_gemm_status(int status) {
  if (status == static_cast<int>(warpconvnet::gemm::GemmStatus::kSuccess)) {
    return;
  }
  const char *msg =
      warpconvnet::gemm::GemmStatusToString(static_cast<warpconvnet::gemm::GemmStatus>(status));
  std::stringstream ss;
  ss << "CUTLASS GEMM failed: " << msg << " (status " << status << ")";
  TORCH_CHECK(false, ss.str());
}

// ------------------- NEW: Common validation helpers -------------------
namespace {

inline void _check_2d(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.dim() == 2, name, " must be 2D");
}

inline torch::Tensor _ensure_2d(torch::Tensor idx) {
  return idx.dim() == 1 ? idx.unsqueeze(1) : idx;
}

struct AdGatherScatterParams {
  int M_A;
  int K;
  int N;
  int M_C;
  int gather_ad_size;
};

inline AdGatherScatterParams validate_ad_gather_scatter_args(torch::Tensor &tensor_a,
                                                             torch::Tensor &tensor_b,
                                                             torch::Tensor &tensor_c,
                                                             torch::Tensor &tensor_d,
                                                             torch::Tensor &indices_a,
                                                             torch::Tensor &indices_d) {
  _check_2d(tensor_a, "tensor_a");
  _check_2d(tensor_b, "tensor_b");
  _check_2d(tensor_c, "tensor_c");
  _check_2d(tensor_d, "tensor_d");

  indices_a = _ensure_2d(indices_a);
  indices_d = _ensure_2d(indices_d);
  TORCH_CHECK(indices_a.dim() == 2, "indices_a must be 2D");
  TORCH_CHECK(indices_d.dim() == 2, "indices_d must be 2D");
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32, "indices_a must be int32");
  TORCH_CHECK(indices_d.scalar_type() == torch::kInt32, "indices_d must be int32");

  // Get dimensions
  int M_A = tensor_a.size(0);  // row of A
  int K = tensor_a.size(1);    // col of A
  int N = tensor_b.size(1);    // col of B
  int M_C = tensor_c.size(0);  // row of C
  int gather_ad_size = indices_a.size(0);

  // Check dimension compatibility
  TORCH_CHECK(tensor_b.size(0) == K,
              "tensor_b first dimension must match tensor_a second dimension");
  TORCH_CHECK(tensor_c.size(1) == N, "tensor_c dimensions must match output");
  TORCH_CHECK(tensor_d.size(0) == M_C, "tensor_c and tensor_d must have the same number of rows");
  TORCH_CHECK(tensor_d.size(1) == N, "tensor_d dimensions must match output");
  TORCH_CHECK(indices_a.size(1) == 1, "indices_a must have 1 column");
  TORCH_CHECK(indices_d.size(1) == 1, "indices_d must have 1 column");
  TORCH_CHECK(indices_a.size(0) == indices_d.size(0),
              "indices_a and indices_d must have same number of rows");

  return {M_A, K, N, M_C, gather_ad_size};
}

struct TrABGatherParams {
  int M_A;
  int K;
  int K_B;
  int N;
  int gather_ab_size;
};

inline TrABGatherParams validate_trAB_gather_args(torch::Tensor &tensor_a,
                                                  torch::Tensor &tensor_b,
                                                  torch::Tensor &tensor_c,
                                                  torch::Tensor &tensor_d,
                                                  torch::Tensor &indices_a,
                                                  torch::Tensor &indices_b) {
  _check_2d(tensor_a, "tensor_a");
  _check_2d(tensor_b, "tensor_b");
  _check_2d(tensor_c, "tensor_c");
  _check_2d(tensor_d, "tensor_d");

  indices_a = _ensure_2d(indices_a);
  indices_b = _ensure_2d(indices_b);
  TORCH_CHECK(indices_a.dim() == 2, "indices_a must be 2D");
  TORCH_CHECK(indices_b.dim() == 2, "indices_b must be 2D");
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32, "indices_a must be int32");
  TORCH_CHECK(indices_b.scalar_type() == torch::kInt32, "indices_b must be int32");

  int M_A = tensor_a.size(0);
  int K = tensor_a.size(1);
  int K_B = tensor_b.size(0);
  int N = tensor_b.size(1);
  int gather_ab_size = indices_a.size(0);

  TORCH_CHECK(tensor_c.size(0) == K && tensor_c.size(1) == N,
              "tensor_c dimensions must be K x N for transpose operation");
  TORCH_CHECK(tensor_d.size(0) == K && tensor_d.size(1) == N,
              "tensor_d dimensions must be K x N for transpose operation");
  TORCH_CHECK(indices_a.size(1) == 1, "indices_a must have 1 column");
  TORCH_CHECK(indices_b.size(1) == 1, "indices_b must have 1 column");
  TORCH_CHECK(indices_a.size(0) == indices_b.size(0),
              "indices_a and indices_b must have the same size");

  return {M_A, K, K_B, N, gather_ab_size};
}

}  // namespace
// ----------------- END helpers -----------------

int cutlass_gemm_AD_gather_scatter(torch::Tensor tensor_a,
                                   torch::Tensor tensor_b,
                                   torch::Tensor tensor_c,
                                   torch::Tensor tensor_d,
                                   torch::Tensor indices_a,
                                   torch::Tensor indices_d,
                                   torch::ScalarType accumulator_type = torch::kFloat32,
                                   int split_k_slices = 1,
                                   int mma_tile = 0,
                                   float alpha = 1.0F,
                                   float beta = 1.0F) {
  // Validate dimensions and get commonly used sizes
  const auto params =
      validate_ad_gather_scatter_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_d);

  // Check the accumulator type early (common for all valid combos)
  TORCH_CHECK(accumulator_type == torch::kFloat16 || accumulator_type == torch::kFloat32,
              "accumulator_type must be float16 or float32");

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_b = tensor_b.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  int status = 0;
  bool handled = false;

#define DISPATCH_GEMM_HANDLE(SA, SB, SO, ALLOW_F16_ACC)                                     \
  if (!handled && scalar_a == SA && scalar_b == SB && scalar_c == SO && scalar_d == SO) {   \
    handled = true;                                                                         \
    if (accumulator_type == torch::kFloat16 && ALLOW_F16_ACC) {                             \
      status = dispatch_cutlass_gemm_ad_gather_scatter<SA, SB, SO, torch::kFloat16>(        \
          tensor_a,                                                                         \
          tensor_b,                                                                         \
          tensor_c,                                                                         \
          tensor_d,                                                                         \
          indices_a,                                                                        \
          indices_d,                                                                        \
          split_k_slices,                                                                   \
          mma_tile,                                                                         \
          params.M_A,                                                                       \
          params.K,                                                                         \
          params.N,                                                                         \
          params.M_C,                                                                       \
          params.gather_ad_size,                                                            \
          alpha,                                                                            \
          beta);                                                                            \
    } else if (accumulator_type == torch::kFloat32) {                                       \
      status = dispatch_cutlass_gemm_ad_gather_scatter<SA, SB, SO, torch::kFloat32>(        \
          tensor_a,                                                                         \
          tensor_b,                                                                         \
          tensor_c,                                                                         \
          tensor_d,                                                                         \
          indices_a,                                                                        \
          indices_d,                                                                        \
          split_k_slices,                                                                   \
          mma_tile,                                                                         \
          params.M_A,                                                                       \
          params.K,                                                                         \
          params.N,                                                                         \
          params.M_C,                                                                       \
          params.gather_ad_size,                                                            \
          alpha,                                                                            \
          beta);                                                                            \
    } else {                                                                                \
      TORCH_CHECK(false, "Unsupported accumulator type for this input/output combination"); \
    }                                                                                       \
  }

  // Supported combinations
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat16, true);
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat32, true);
#ifndef DISABLE_BFLOAT16
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16, false);
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kFloat32, false);
#endif

#undef DISPATCH_GEMM_HANDLE

  // Special handling for FP32 → convert to half precision path
  if (!handled && scalar_a == torch::kFloat32 && scalar_b == torch::kFloat32 &&
      scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    handled = true;
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    status = dispatch_cutlass_gemm_ad_gather_scatter<torch::kFloat16,
                                                     torch::kFloat16,
                                                     torch::kFloat32,
                                                     torch::kFloat32>(tensor_a,
                                                                      tensor_b,
                                                                      tensor_c,
                                                                      tensor_d,
                                                                      indices_a,
                                                                      indices_d,
                                                                      split_k_slices,
                                                                      mma_tile,
                                                                      params.M_A,
                                                                      params.K,
                                                                      params.N,
                                                                      params.M_C,
                                                                      params.gather_ad_size,
                                                                      alpha,
                                                                      beta);
  }

  if (!handled) {
    std::stringstream ss;
    ss << "Unsupported tensor type combination. " << "A: " << scalar_a << ", B: " << scalar_b
       << ", C: " << scalar_c << ", D: " << scalar_d << ", Acc: " << accumulator_type;
    TORCH_CHECK(false, ss.str());
  }

  return status;
}

int cutlass_gemm_trAB_gather(torch::Tensor tensor_a,
                             torch::Tensor tensor_b,
                             torch::Tensor tensor_c,
                             torch::Tensor tensor_d,
                             torch::Tensor indices_a,
                             torch::Tensor indices_b,
                             torch::ScalarType accumulator_type = torch::kFloat32,
                             int split_k_slices = 1,
                             int mma_tile = 0,
                             float alpha = 1.0F,
                             float beta = 1.0F) {
  // Validate dimensions and get commonly used sizes
  const auto params =
      validate_trAB_gather_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_b);

  // Check the accumulator type early (common for all valid combos)
  TORCH_CHECK(accumulator_type == torch::kFloat16 || accumulator_type == torch::kFloat32,
              "accumulator_type must be float16 or float32");

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_b = tensor_b.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  int status = 0;
  bool handled = false;

#define DISPATCH_GEMM_HANDLE(SA, SB, SO, ALLOW_F16_ACC)                                         \
  if (!handled && scalar_a == SA && scalar_b == SB && scalar_c == SO && scalar_d == SO) {       \
    handled = true;                                                                             \
    if (accumulator_type == torch::kFloat16 && ALLOW_F16_ACC) {                                 \
      status =                                                                                  \
          dispatch_cutlass_gemm_trAB_gather<SA, SB, SO, torch::kFloat16>(tensor_a,              \
                                                                         tensor_b,              \
                                                                         tensor_c,              \
                                                                         tensor_d,              \
                                                                         indices_a,             \
                                                                         indices_b,             \
                                                                         split_k_slices,        \
                                                                         mma_tile,              \
                                                                         params.M_A,            \
                                                                         params.K,              \
                                                                         params.K_B,            \
                                                                         params.N,              \
                                                                         params.gather_ab_size, \
                                                                         alpha,                 \
                                                                         beta);                 \
    } else if (accumulator_type == torch::kFloat32) {                                           \
      status =                                                                                  \
          dispatch_cutlass_gemm_trAB_gather<SA, SB, SO, torch::kFloat32>(tensor_a,              \
                                                                         tensor_b,              \
                                                                         tensor_c,              \
                                                                         tensor_d,              \
                                                                         indices_a,             \
                                                                         indices_b,             \
                                                                         split_k_slices,        \
                                                                         mma_tile,              \
                                                                         params.M_A,            \
                                                                         params.K,              \
                                                                         params.K_B,            \
                                                                         params.N,              \
                                                                         params.gather_ab_size, \
                                                                         alpha,                 \
                                                                         beta);                 \
    } else {                                                                                    \
      TORCH_CHECK(false, "Unsupported accumulator type for this input/output combination");     \
    }                                                                                           \
  }

  // Supported combinations
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat16, true);
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat32, true);
#ifndef DISABLE_BFLOAT16
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16, false);
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kFloat32, false);
#endif

#undef DISPATCH_GEMM_HANDLE

  // Special handling for FP32 → convert to half precision path (inputs FP32, outputs+acc FP32)
  if (!handled && scalar_a == torch::kFloat32 && scalar_b == torch::kFloat32 &&
      scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    handled = true;
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    status = dispatch_cutlass_gemm_trAB_gather<torch::kFloat16,
                                               torch::kFloat16,
                                               torch::kFloat32,
                                               torch::kFloat32>(tensor_a,
                                                                tensor_b,
                                                                tensor_c,
                                                                tensor_d,
                                                                indices_a,
                                                                indices_b,
                                                                split_k_slices,
                                                                mma_tile,
                                                                params.M_A,
                                                                params.K,
                                                                params.K_B,
                                                                params.N,
                                                                params.gather_ab_size,
                                                                alpha,
                                                                beta);
  }

  if (!handled) {
    std::stringstream ss;
    ss << "Unsupported tensor type combination. " << "A: " << scalar_a << ", B: " << scalar_b
       << ", C: " << scalar_c << ", D: " << scalar_d << ", Acc: " << accumulator_type;
    TORCH_CHECK(false, ss.str());
  }

  return status;
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "CUDA kernels exposed through PyBind11";

  // Create submodule 'gemm' for all GEMM-related bindings
  py::module_ gemm = m.def_submodule(
      "gemm", "CUTLASS GEMM with gather/scatter operations supporting multiple precisions");

  // Explicit precision functions under gemm submodule
  gemm.def("cutlass_gemm_AD_gather_scatter",
           &cutlass_gemm_AD_gather_scatter,
           "Run CUTLASS GEMM with gather/scatter (half precision inputs and "
           "accumulator)",
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_d"),
           py::arg("accumulator_type") = torch::kFloat32,
           py::arg("split_k_slices") = 1,
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f);

  // trAB Gather
  gemm.def("cutlass_gemm_trAB_gather",
           &cutlass_gemm_trAB_gather,
           "Run CUTLASS GEMM with AB gather and A transpose",
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_b"),
           py::arg("accumulator_type") = torch::kFloat32,
           py::arg("split_k_slices") = 1,
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f);

  // Expose GemmStatus enum inside gemm submodule
  py::enum_<warpconvnet::gemm::GemmStatus>(gemm, "GemmStatus")
      .value("kSuccess", warpconvnet::gemm::GemmStatus::kSuccess)
      .value("kErrorProblemNotSupported", warpconvnet::gemm::GemmStatus::kErrorProblemNotSupported)
      .value("kErrorKernelInitialization",
             warpconvnet::gemm::GemmStatus::kErrorKernelInitialization)
      .value("kErrorKernelExecution", warpconvnet::gemm::GemmStatus::kErrorKernelExecution)
      .value("kErrorUnsupportedConfig", warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig)
      .value("kErrorInvalidParameters", warpconvnet::gemm::GemmStatus::kErrorInvalidParameters)
      .value("kErrorMixedInputUnsupported",
             warpconvnet::gemm::GemmStatus::kErrorMixedInputUnsupported)
      .export_values();

  // Helper function to convert GemmStatus to human-readable text in Python
  gemm.def(
      "gemm_status_to_string",
      [](warpconvnet::gemm::GemmStatus status) {
        return std::string(warpconvnet::gemm::GemmStatusToString(status));
      },
      py::arg("status"),
      "Return the human-readable string associated with a GemmStatus value");
}

// ------------------ Implementation of dispatch helpers with mma_tile switch ------------------

template <torch::ScalarType ScalarA,
          torch::ScalarType ScalarB,
          torch::ScalarType ScalarOutput,
          torch::ScalarType ScalarAccumulator>
int dispatch_cutlass_gemm_ad_gather_scatter(const torch::Tensor &tensor_a,
                                            const torch::Tensor &tensor_b,
                                            const torch::Tensor &tensor_c,
                                            torch::Tensor &tensor_d,
                                            const torch::Tensor &indices_a,
                                            const torch::Tensor &indices_d,
                                            int split_k_slices,
                                            int mma_tile,
                                            int M_A,
                                            int K,
                                            int N,
                                            int M_C,
                                            int gather_ad_size,
                                            float alpha,
                                            float beta) {
  using ElementA = typename torch_to_cutlass<ScalarA>::type;
  using ElementB = typename torch_to_cutlass<ScalarB>::type;
  using ElementOutput = typename torch_to_cutlass<ScalarOutput>::type;
  using ElementAccumulator = typename torch_to_cutlass<ScalarAccumulator>::type;

  TORCH_CHECK(tensor_a.scalar_type() == ScalarA);
  TORCH_CHECK(tensor_b.scalar_type() == ScalarB);
  TORCH_CHECK(tensor_c.scalar_type() == ScalarOutput);
  TORCH_CHECK(tensor_d.scalar_type() == ScalarOutput);

  // Macro for AD gather scatter tile cases
#define GENERATE_AD_TILE_CASE(TILE_NAME)                                                      \
  case warpconvnet::gemm::MMATile::TILE_NAME:                                                 \
    return run_cutlass_gemm_ad_gather_scatter<ElementA,                                       \
                                              ElementB,                                       \
                                              ElementOutput,                                  \
                                              ElementAccumulator,                             \
                                              warpconvnet::gemm::TILE_NAME,                   \
                                              cutlass::arch::Sm80>(tensor_a.data_ptr(),       \
                                                                   tensor_b.data_ptr(),       \
                                                                   tensor_c.data_ptr(),       \
                                                                   tensor_d.data_ptr(),       \
                                                                   indices_a.data_ptr<int>(), \
                                                                   indices_d.data_ptr<int>(), \
                                                                   split_k_slices,            \
                                                                   M_A,                       \
                                                                   K,                         \
                                                                   N,                         \
                                                                   M_C,                       \
                                                                   gather_ad_size,            \
                                                                   alpha,                     \
                                                                   beta);

  warpconvnet::gemm::MMATile tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    GENERATE_AD_TILE_CASE(Tile128x128x32);
    GENERATE_AD_TILE_CASE(Tile128x64x32);
    GENERATE_AD_TILE_CASE(Tile64x128x32);
    GENERATE_AD_TILE_CASE(Tile64x64x32);
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value");
  }
#undef GENERATE_AD_TILE_CASE
}

template <torch::ScalarType ScalarA,
          torch::ScalarType ScalarB,
          torch::ScalarType ScalarOutput,
          torch::ScalarType ScalarAccumulator>
int dispatch_cutlass_gemm_trAB_gather(const torch::Tensor &tensor_a,
                                      const torch::Tensor &tensor_b,
                                      const torch::Tensor &tensor_c,
                                      torch::Tensor &tensor_d,
                                      const torch::Tensor &indices_a,
                                      const torch::Tensor &indices_b,
                                      int split_k_slices,
                                      int mma_tile,
                                      int M_A,
                                      int K,
                                      int K_B,
                                      int N,
                                      int gather_ab_size,
                                      float alpha,
                                      float beta) {
  using ElementA = typename torch_to_cutlass<ScalarA>::type;
  using ElementB = typename torch_to_cutlass<ScalarB>::type;
  using ElementOutput = typename torch_to_cutlass<ScalarOutput>::type;
  using ElementAccumulator = typename torch_to_cutlass<ScalarAccumulator>::type;

  TORCH_CHECK(tensor_a.scalar_type() == ScalarA);
  TORCH_CHECK(tensor_b.scalar_type() == ScalarB);
  TORCH_CHECK(tensor_c.scalar_type() == ScalarOutput);
  TORCH_CHECK(tensor_d.scalar_type() == ScalarOutput);

  // Macro for TrAB gather tile cases
#define GENERATE_TRAB_TILE_CASE(TILE_NAME)                                              \
  case warpconvnet::gemm::MMATile::TILE_NAME:                                           \
    return run_cutlass_gemm_trAB_gather<ElementA,                                       \
                                        ElementB,                                       \
                                        ElementOutput,                                  \
                                        ElementAccumulator,                             \
                                        warpconvnet::gemm::TILE_NAME,                   \
                                        cutlass::arch::Sm80>(tensor_a.data_ptr(),       \
                                                             tensor_b.data_ptr(),       \
                                                             tensor_c.data_ptr(),       \
                                                             tensor_d.data_ptr(),       \
                                                             indices_a.data_ptr<int>(), \
                                                             indices_b.data_ptr<int>(), \
                                                             split_k_slices,            \
                                                             M_A,                       \
                                                             K,                         \
                                                             K_B,                       \
                                                             N,                         \
                                                             gather_ab_size,            \
                                                             alpha,                     \
                                                             beta);

  warpconvnet::gemm::MMATile tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    GENERATE_TRAB_TILE_CASE(Tile128x128x32);
    GENERATE_TRAB_TILE_CASE(Tile128x64x32);
    GENERATE_TRAB_TILE_CASE(Tile64x128x32);
    GENERATE_TRAB_TILE_CASE(Tile64x64x32);
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value");
  }
#undef GENERATE_TRAB_TILE_CASE
}
