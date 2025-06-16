// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cutlass/util/device_memory.h>

#include "include/gemm_operations.h"

// Main templated function implementation - define in the warpconvnet::gemm namespace
namespace warpconvnet {
namespace gemm {

template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename Config,
          typename ArchTag>
int run_cutlass_gemm_with_operations_templated(const void *tensor_a,
                                               const void *tensor_b,
                                               const void *tensor_c,
                                               void *tensor_d,
                                               const int *indices_a,
                                               const int *indices_b,
                                               const int *indices_d,
                                               int split_k_slices,
                                               int M_A,  // Original A matrix rows
                                               int K,    // A matrix columns
                                               int M_B,  // Original B matrix rows
                                               int N,    // B matrix columns
                                               int gather_a_size,
                                               int scatter_d_size,
                                               float alpha,
                                               float beta) {
  using Traits = GemmOperationTraits<ElementInputA, ElementAccumulator, Config, ArchTag>;
  using ElementComputeEpilogue = ElementAccumulator;

  if constexpr (Traits::UseMixedInput) {
    printf("Error: Mixed input not supported for this precision configuration\n");
    return -6;
  }

  // Determine output vector length based on element type
  constexpr int OutputVectorLength = std::is_same_v<ElementOutput, cutlass::half_t>
                                         ? (128 / cutlass::sizeof_bits<ElementOutput>::value)
                                         : (128 / cutlass::sizeof_bits<ElementOutput>::value);

  // Define epilogue operation
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                  OutputVectorLength,
                                                                  ElementAccumulator,
                                                                  ElementComputeEpilogue>;

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                                    typename Traits::LayoutInputA,
                                                    ElementInputB,
                                                    typename Traits::LayoutInputB,
                                                    ElementOutput,
                                                    typename Traits::LayoutOutput,
                                                    ElementAccumulator,
                                                    typename Traits::MMAOp,
                                                    typename Traits::ArchitectureTag,
                                                    typename Traits::ShapeMMAThreadBlock,
                                                    typename Traits::ShapeMMAWarp,
                                                    typename Traits::ShapeMMAOp,
                                                    EpilogueOp,
                                                    SwizzleThreadBlock,
                                                    NumStages,
                                                    Traits::AlignmentA,
                                                    Traits::AlignmentB,
                                                    cutlass::arch::OpMultiplyAdd,
                                                    cutlass::ComplexTransform::kNone,
                                                    cutlass::ComplexTransform::kNone,
                                                    Traits::SupportsGatherA, /*GatherA*/
                                                    Traits::SupportsGatherB, /*GatherB*/
                                                    Traits::SupportsScatterD /*ScatterD*/
                                                    >;

  // Convert void pointers to appropriate types
  auto a_ptr = reinterpret_cast<const ElementInputA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementInputB *>(tensor_b);
  auto c_ptr = reinterpret_cast<const ElementOutput *>(tensor_c);
  auto d_ptr = reinterpret_cast<ElementOutput *>(tensor_d);

  if constexpr (Traits::IsValidConfiguration()) {
    // Native gather/scatter implementation
    // For transpose operation, the layout determines how the matrix is interpreted
    typename Traits::LayoutInputA layout_a(K);
    typename Traits::LayoutInputB layout_b(N);
    typename Traits::LayoutOutput layout_c(N);
    typename Traits::LayoutOutput layout_d(N);

    ElementComputeEpilogue alpha_cutlass = ElementComputeEpilogue(alpha);
    ElementComputeEpilogue beta_cutlass = ElementComputeEpilogue(beta);

    // A: M x K (or K x M if transposed)
    // B: K x N
    // C: M or Scatter_M x N  (When using scatter, size of rows (Scatter_M) > M)
    // D: M or Scatter_M x N  (When using scatter, size of rows (Scatter_M) > M)

    // Derive problem dimensions from original matrix dimensions
    int problem_m, problem_n, problem_k;

    if constexpr (Config::transpose_a) {
      // For A transpose: A[indices_a, :].T @ B[indices_b, :]
      // A[indices_a, :] is gather_a_size × K, transposed to K × gather_a_size
      // B[indices_b, :] is gather_a_size × N
      // Result: K × N
      problem_m = K;              // rows in result (from transposed A)
      problem_n = N;              // columns in result (from B)
      problem_k = gather_a_size;  // inner dimension
    } else if constexpr (Config::gather_a && Config::gather_b) {
      // Standard AB gather: A[indices_a, :] @ B[indices_b, :]
      // Result: gather_a_size × N
      problem_m = gather_a_size;
      problem_n = N;
      problem_k = K;
    } else if constexpr (Config::gather_a && Config::scatter_d) {
      // AD gather scatter: A[indices_a, :] @ B
      // Result scattered to D[indices_d, :]
      problem_m = gather_a_size;
      problem_n = N;
      problem_k = K;
    } else {
      // Standard GEMM dimensions
      problem_m = M_A;
      problem_n = N;
      problem_k = K;
    }

    cutlass::gemm::GemmCoord problem_size(problem_m, problem_n, problem_k);

    // Determine output size based on configuration
    int output_rows = Config::scatter_d ? scatter_d_size : problem_m;

    // Calculate batch strides using original matrix dimensions
    int64_t batch_stride_A = static_cast<int64_t>(M_A) * K * sizeof(ElementInputA);
    int64_t batch_stride_B = static_cast<int64_t>(M_B) * N * sizeof(ElementInputB);
    int64_t batch_stride_C = static_cast<int64_t>(output_rows) * N * sizeof(ElementOutput);
    int64_t batch_stride_D = static_cast<int64_t>(output_rows) * N * sizeof(ElementOutput);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       problem_size,
                                       split_k_slices,
                                       {alpha_cutlass, beta_cutlass},
                                       a_ptr,
                                       b_ptr,
                                       c_ptr,
                                       d_ptr,
                                       /*batch strides*/ batch_stride_A,
                                       batch_stride_B,
                                       batch_stride_C,
                                       batch_stride_D,
                                       layout_a.stride(),
                                       layout_b.stride(),
                                       layout_c.stride(),
                                       layout_d.stride(),
                                       Config::gather_a ? indices_a : nullptr,
                                       Config::gather_b ? indices_b : nullptr,
                                       Config::scatter_d ? indices_d : nullptr};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      printf("Error: Problem size is not supported for %s\n", Traits::GetConfigName());
      return -1;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      printf("Error: CUTLASS kernel initialization failed for %s\n", Traits::GetConfigName());
      return -2;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      printf("Error: CUTLASS kernel execution failed for %s\n", Traits::GetConfigName());
      return -3;
    }
  } else {
    // Configuration not supported
    printf("Error: %s operations not supported for this precision configuration\n",
           Traits::GetConfigName());
    return -4;
  }

  return 0;
}

}  // namespace gemm
}  // namespace warpconvnet

// Use the namespace for convenience in the rest of the file
using namespace warpconvnet::gemm;

// A Gather + D scatter
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator>
int run_cutlass_gemm_ad_gather_scatter(const void *tensor_a,
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
                                       float beta) {
  // Forward to new templated implementation with ConfigAD (A gather + D scatter)
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigAD>(
      tensor_a,
      tensor_b,
      tensor_c,
      tensor_d,
      indices_a,
      nullptr,    // indices_b (no B gather in AD config)
      indices_d,  // indices_d for D scatter
      split_k_slices,
      M,  // M_A (original A matrix rows)
      K,  // K (A columns)
      M,  // M_B (original B matrix rows = M for AD case)
      N,  // N (B columns)
      indices_size,
      out_size,  // scatter_d_size
      alpha,
      beta);
}

// A Gather + D scatter instantiations
INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, float, float)
INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(cutlass::half_t,
                                              cutlass::half_t,
                                              cutlass::half_t,
                                              float)
INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(cutlass::half_t,
                                              cutlass::half_t,
                                              float,
                                              cutlass::half_t)
INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(cutlass::half_t,
                                              cutlass::half_t,
                                              cutlass::half_t,
                                              cutlass::half_t)

#ifndef DISABLE_BFLOAT16
// Bfloat16 precision instantiations (FP32 accumulator)
INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(cutlass::bfloat16_t,
                                              cutlass::bfloat16_t,
                                              cutlass::bfloat16_t,
                                              float)
INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(cutlass::bfloat16_t,
                                              cutlass::bfloat16_t,
                                              float,
                                              float)
#endif  // DISABLE_BFLOAT16

// AB Gather
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator>
int run_cutlass_gemm_ab_gather(const void *tensor_a,
                               const void *tensor_b,
                               const void *tensor_c,
                               void *tensor_d,
                               const int *indices_a,
                               const int *indices_b,
                               int split_k_slices,
                               int M,
                               int N,
                               int K,
                               int indices_size,
                               float alpha,
                               float beta) {
  // Forward to new templated implementation with ConfigAB (A gather + B gather)
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigAB>(
      tensor_a,
      tensor_b,
      tensor_c,
      tensor_d,
      indices_a,
      indices_b,
      nullptr,  // indices_d (no D scatter in AB config)
      split_k_slices,
      M,             // M_A (original A matrix rows)
      K,             // K (A columns)
      K,             // M_B (original B matrix rows = K for AB gather)
      N,             // N (B columns)
      indices_size,  // gather_a_size
      0,             // scatter_d_size (no scatter)
      alpha,
      beta);
}

// Instantiate AB Gather
INSTANTIATE_AB_GATHER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, float, float)
INSTANTIATE_AB_GATHER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, cutlass::half_t, float)
INSTANTIATE_AB_GATHER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, float, cutlass::half_t)
INSTANTIATE_AB_GATHER_GEMM_OPERATIONS(cutlass::half_t,
                                      cutlass::half_t,
                                      cutlass::half_t,
                                      cutlass::half_t)

#ifndef DISABLE_BFLOAT16
// Bfloat16 precision instantiations (FP32 accumulator)
INSTANTIATE_AB_GATHER_GEMM_OPERATIONS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, float)
INSTANTIATE_AB_GATHER_GEMM_OPERATIONS(cutlass::bfloat16_t,
                                      cutlass::bfloat16_t,
                                      cutlass::bfloat16_t,
                                      float)
#endif  // DISABLE_BFLOAT16

// AB Gather with A Transpose
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator>
int run_cutlass_gemm_trAB_gather(const void *tensor_a,
                                 const void *tensor_b,
                                 const void *tensor_c,
                                 void *tensor_d,
                                 const int *indices_a,
                                 const int *indices_b,
                                 int split_k_slices,
                                 int M_A,
                                 int K,
                                 int M_B,
                                 int N,
                                 int gather_a_size,
                                 float alpha,
                                 float beta) {
  // Forward to new templated implementation with ConfigTrAB (A gather + B gather + A transpose)
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigTrAB,
                                                    DefaultSmArch>(
      tensor_a,
      tensor_b,
      tensor_c,
      tensor_d,
      indices_a,
      indices_b,
      nullptr,  // indices_d (no D scatter in AB config)
      split_k_slices,
      M_A,  // M_A (original A matrix rows)
      K,    // K (A columns)
      M_B,  // M_B (original B matrix rows)
      N,    // N (B columns)
      gather_a_size,
      0,  // scatter_d_size (no scatter)
      alpha,
      beta);
}

// Instantiate trAB Gather
INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, float, float);
INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, cutlass::half_t, float);
INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(cutlass::half_t, cutlass::half_t, float, cutlass::half_t);
INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(cutlass::half_t,
                                        cutlass::half_t,
                                        cutlass::half_t,
                                        cutlass::half_t);

#ifndef DISABLE_BFLOAT16
// Bfloat16 precision instantiations (FP32 accumulator)
INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(cutlass::bfloat16_t, cutlass::bfloat16_t, float, float);
INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(cutlass::bfloat16_t,
                                        cutlass::bfloat16_t,
                                        cutlass::bfloat16_t,
                                        float);
#endif  // DISABLE_BFLOAT16
