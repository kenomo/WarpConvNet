// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "driver_types.h"

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

#ifdef USE_BFLOAT16
template <>
struct torch_to_cutlass<torch::kBFloat16> {
  using type = cutlass::bfloat16_t;
};
#endif  // USE_BFLOAT16

// Forward declaration of the templated function (defined in .cu file)
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator>
int run_cutlass_gemm_gather_scatter_templated(const void *tensor_a,
                                              const void *tensor_b,
                                              const void *tensor_c,
                                              void *tensor_d,
                                              const int *indices_a,
                                              const int *indices_d,
                                              int split_k_slices,
                                              int M,
                                              int N,
                                              int K,
                                              int indices_size,
                                              int out_size,
                                              float alpha,
                                              float beta);

// Forward declaration of specialised SM80 kernel for FP32 input with
// gather/scatter
int run_f32_to_f16_gemm_gather_scatter_sm80(const float *dA,
                                            const float *dB,
                                            const float *dC,
                                            float *dD,
                                            const int *gatherA_indices,
                                            const int *scatterD_indices,
                                            int split_k_slices,
                                            int M,
                                            int N,
                                            int K,
                                            int gather_rows,
                                            int scatter_rows,
                                            float alpha = 1.F,
                                            float beta = 0.F,
                                            cudaStream_t stream = nullptr);

// Helper function to dispatch CUTLASS GEMM with automatic type deduction
template <torch::ScalarType ScalarA,
          torch::ScalarType ScalarB,
          torch::ScalarType ScalarOutput,
          torch::ScalarType ScalarAccumulator>
int dispatch_cutlass_gemm(const torch::Tensor &tensor_a,
                          const torch::Tensor &tensor_b,
                          const torch::Tensor &tensor_c,
                          torch::Tensor &tensor_d,
                          const torch::Tensor &indices_a,
                          const torch::Tensor &indices_d,
                          int split_k_slices,
                          int M,
                          int N,
                          int K,
                          int indices_size,
                          int out_size,
                          float alpha,
                          float beta) {
  // Deduce CUTLASS types from tensor scalar types
  using ElementA = typename torch_to_cutlass<ScalarA>::type;
  using ElementB = typename torch_to_cutlass<ScalarB>::type;
  using ElementOutput = typename torch_to_cutlass<ScalarOutput>::type;
  using ElementAccumulator = typename torch_to_cutlass<ScalarAccumulator>::type;

  static_assert(std::is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same");

  // Assert that tensor_c and tensor_d are the same type
  TORCH_CHECK(tensor_a.scalar_type() == ScalarA);
  TORCH_CHECK(tensor_b.scalar_type() == ScalarB);
  TORCH_CHECK(tensor_c.scalar_type() == ScalarOutput);
  TORCH_CHECK(tensor_d.scalar_type() == ScalarOutput);
  TORCH_CHECK(tensor_c.scalar_type() == tensor_d.scalar_type());

  return run_cutlass_gemm_gather_scatter_templated<ElementA,
                                                   ElementB,
                                                   ElementOutput,
                                                   ElementAccumulator>(tensor_a.data_ptr(),
                                                                       tensor_b.data_ptr(),
                                                                       tensor_c.data_ptr(),
                                                                       tensor_d.data_ptr(),
                                                                       indices_a.data_ptr<int>(),
                                                                       indices_d.data_ptr<int>(),
                                                                       split_k_slices,
                                                                       M,
                                                                       N,
                                                                       K,
                                                                       indices_size,
                                                                       out_size,
                                                                       alpha,
                                                                       beta);
}

torch::Tensor run_cutlass_gemm(torch::Tensor tensor_a,
                               torch::Tensor tensor_b,
                               torch::Tensor tensor_c,
                               torch::Tensor tensor_d,
                               torch::Tensor indices_a,
                               torch::Tensor indices_d,
                               torch::ScalarType accumulator_type = torch::kFloat32,
                               int split_k_slices = 1,
                               float alpha = 1.0F,
                               float beta = 1.0F) {
  // Check tensor dimensions
  TORCH_CHECK(tensor_a.dim() == 2, "tensor_a must be 2D");
  TORCH_CHECK(tensor_b.dim() == 2, "tensor_b must be 2D");
  TORCH_CHECK(tensor_c.dim() == 2, "tensor_c must be 2D");
  TORCH_CHECK(tensor_d.dim() == 2, "tensor_d must be 2D");
  TORCH_CHECK(indices_a.dim() == 2, "indices_a must be 2D");
  TORCH_CHECK(indices_d.dim() == 2, "indices_d must be 2D");

  // Check index tensor types
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32, "indices_a must be int32");
  TORCH_CHECK(indices_d.scalar_type() == torch::kInt32, "indices_d must be int32");

  // Check the accumulator type

  TORCH_CHECK(accumulator_type == torch::kFloat16 || accumulator_type == torch::kFloat32,
              "accumulator_type must be float16 or float32");

  // Get dimensions
  int M = tensor_a.size(0);
  int K = tensor_a.size(1);
  int N = tensor_b.size(1);
  int out_size = tensor_d.size(0);
  int indices_size = indices_a.size(0);

  // Check dimension compatibility
  TORCH_CHECK(tensor_b.size(0) == K,
              "tensor_b first dimension must match tensor_a second dimension");
  TORCH_CHECK(tensor_c.size(0) == M && tensor_c.size(1) == N,
              "tensor_c dimensions must match output");
  TORCH_CHECK(tensor_d.size(1) == N, "tensor_d dimensions must match output");
  TORCH_CHECK(indices_a.size(1) == 1, "indices_a must have 1 column");
  TORCH_CHECK(indices_d.size(1) == 1, "indices_d must have 1 column");
  TORCH_CHECK(indices_d.size(0) == indices_size,
              "indices_a and indices_d must have same number of rows");

  // Dispatch based on tensor scalar types using template specialization
  auto scalar_a = tensor_a.scalar_type();
  auto scalar_b = tensor_b.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  int status = -1;

  // Dispatch to appropriate template instantiation based on scalar types
  if (scalar_a == torch::kFloat16 && scalar_b == torch::kFloat16 && scalar_c == torch::kFloat16 &&
      scalar_d == torch::kFloat16) {
    // Use float accumulator (ScalarC) while keeping A,B,D in FP16
    if (accumulator_type == torch::kFloat16) {
      status =
          dispatch_cutlass_gemm<torch::kFloat16, torch::kFloat16, torch::kFloat16, torch::kFloat16>(
              tensor_a,
              tensor_b,
              tensor_c,
              tensor_d,
              indices_a,
              indices_d,
              split_k_slices,
              M,
              N,
              K,
              indices_size,
              out_size,
              alpha,
              beta);
    } else if (accumulator_type == torch::kFloat32) {
      status =
          dispatch_cutlass_gemm<torch::kFloat16, torch::kFloat16, torch::kFloat16, torch::kFloat32>(
              tensor_a,
              tensor_b,
              tensor_c,
              tensor_d,
              indices_a,
              indices_d,
              split_k_slices,
              M,
              N,
              K,
              indices_size,
              out_size,
              alpha,
              beta);
    } else {
      std::stringstream ss;
      ss << "Unsupported accumulator type. " << "Supported types: float16, float32. "
         << "Got: " << accumulator_type;
      TORCH_CHECK(false, ss.str());
    }
  } else if (scalar_a == torch::kFloat16 && scalar_b == torch::kFloat16 &&
             scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    // Mixed precision: if16of32af32
    if (accumulator_type == torch::kFloat16) {
      status =
          dispatch_cutlass_gemm<torch::kFloat16, torch::kFloat16, torch::kFloat32, torch::kFloat16>(
              tensor_a,
              tensor_b,
              tensor_c,
              tensor_d,
              indices_a,
              indices_d,
              split_k_slices,
              M,
              N,
              K,
              indices_size,
              out_size,
              alpha,
              beta);
    } else if (accumulator_type == torch::kFloat32) {
      status =
          dispatch_cutlass_gemm<torch::kFloat16, torch::kFloat16, torch::kFloat32, torch::kFloat32>(
              tensor_a,
              tensor_b,
              tensor_c,
              tensor_d,
              indices_a,
              indices_d,
              split_k_slices,
              M,
              N,
              K,
              indices_size,
              out_size,
              alpha,
              beta);
    } else {
      std::stringstream ss;
      ss << "Unsupported accumulator type. " << "Supported types: float16, float32. "
         << "Got: " << accumulator_type;
      TORCH_CHECK(false, ss.str());
    }
  } else if (scalar_a == torch::kFloat32 && scalar_b == torch::kFloat32 &&
             scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    // Convert to half precision and run the kernel
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    status =
        dispatch_cutlass_gemm<torch::kFloat16, torch::kFloat16, torch::kFloat32, torch::kFloat32>(
            tensor_a,
            tensor_b,
            tensor_c,
            tensor_d,
            indices_a,
            indices_d,
            split_k_slices,
            M,
            N,
            K,
            indices_size,  // gather rows
            out_size,      // scatter rows
            alpha,
            beta);
  }
#ifdef USE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_b == torch::kBFloat16 &&
           scalar_c == torch::kBFloat16 && scalar_d == torch::kBFloat16) {
    // Pure BF16 path (ibf16obf16) with FP32 accumulator
    if (accumulator_type == torch::kFloat32) {
      status = dispatch_cutlass_gemm<torch::kBFloat16,
                                     torch::kBFloat16,
                                     torch::kBFloat16,
                                     torch::kFloat32>(tensor_a,
                                                      tensor_b,
                                                      tensor_c,
                                                      tensor_d,
                                                      indices_a,
                                                      indices_d,
                                                      split_k_slices,
                                                      M,
                                                      N,
                                                      K,
                                                      indices_size,
                                                      out_size,
                                                      alpha,
                                                      beta);
    } else {
      std::stringstream ss;
      ss << "Unsupported accumulator type. " << "Supported types with BF16 IO: float32. "
         << "Got: " << accumulator_type;
      TORCH_CHECK(false, ss.str());
    }
  } else if (scalar_a == torch::kBFloat16 && scalar_b == torch::kBFloat16 &&
             scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    // BF16 input, FP32 output (ibf16of32) with FP32 accumulator
    if (accumulator_type == torch::kFloat32) {
      status = dispatch_cutlass_gemm<torch::kBFloat16,
                                     torch::kBFloat16,
                                     torch::kFloat32,
                                     torch::kFloat32>(tensor_a,
                                                      tensor_b,
                                                      tensor_c,
                                                      tensor_d,
                                                      indices_a,
                                                      indices_d,
                                                      split_k_slices,
                                                      M,
                                                      N,
                                                      K,
                                                      indices_size,
                                                      out_size,
                                                      alpha,
                                                      beta);
    } else {
      std::stringstream ss;
      ss << "Unsupported accumulator type. "
         << "Supported types with BF16 input / FP32 output: float32. "
         << "Got: " << accumulator_type;
      TORCH_CHECK(false, ss.str());
    }
  }
#endif  // USE_BFLOAT16

  else {
    std::stringstream ss;
    ss << "Unsupported tensor type combination. " << "A: " << scalar_a << ", B: " << scalar_b
       << ", C: " << scalar_c << ", D: " << scalar_d << ", Acc: " << accumulator_type << "\n"
       << "Supported combinations:\n"
       << "1. Input: float16, Output: float16, Accumulator: float16 "
          "(if16of16af16)\n"
       << "2. Input: float16, Output: float16, Accumulator: float32 "
          "(if16of16af32)\n"
       << "3. Input: float32, Output: float32, Accumulator: float32 "
          "(f32of32af32)\n";
#ifdef USE_BFLOAT16
    ss << "4. Input: bfloat16, Output: bfloat16, Accumulator: float32 " << "(ibf16obf16af32)\n"
       << "5. Input: bfloat16, Output: float32, Accumulator: float32 " << "(ibf16of32af32)\n";
#endif
    TORCH_CHECK(false, ss.str());
  }

  if (status != 0) {
    std::stringstream ss;
    ss << "CUTLASS GEMM failed with status code " << status << ". "
       << "See previous log messages for kernel-level diagnostics.";
    TORCH_CHECK(false, ss.str());
  }

  return tensor_d;
}

PYBIND11_MODULE(_C, m) {
  m.doc() =
      "CUTLASS GEMM with gather/scatter operations supporting multiple "
      "precisions";

  // Explicit precision functions
  m.def("run_cutlass_gemm",
        &run_cutlass_gemm,
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
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 1.0f);
}
