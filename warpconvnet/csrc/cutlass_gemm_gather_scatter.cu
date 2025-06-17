// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cutlass/util/device_memory.h>

#include "include/gemm_error_codes.h"
#include "include/gemm_operations.h"

// Main templated function implementation - define in the warpconvnet::gemm namespace
namespace warpconvnet {
namespace gemm {

/*
 * @brief Run a GEMM operation with gather/scatter support.
 *
 * @param tensor_a: Pointer to the A matrix.
 * @param tensor_b: Pointer to the B matrix.
 * @param tensor_c: Pointer to the C matrix.
 * @param tensor_d: Pointer to the D matrix.
 *
 * @param indices_a: Indices for the A matrix.
 * @param indices_b: Indices for the B matrix.
 * @param indices_d: Indices for the D matrix.
 *
 * @param split_k_slices: Number of slices to split the K dimension into.
 * @param M_A: Original A matrix rows.
 * @param K: A matrix columns.
 * @param K_B: Original B matrix rows.
 * @param N: B matrix columns.
 * @param M_C: C matrix rows, equal to D matrix rows. (Regardless of whether C is transposed, M_C is
 * the number of rows of the original C matrix before transposition.)
 * @param gather_a_size: indices_a size, equal to indices_b when indices_b is not nullptr.
 *
 * @param alpha: Alpha value for the GEMM operation.
 * @param beta: Beta value for the GEMM operation.
 *
 * @return Status code indicating the success or failure of the operation.
 *
 * trAB gather:
 * D = \alpha * A[indices_a, :].T @ B[indices_b, :] + \beta * C
 *
 * AD gather scatter:
 * D[indices_d, :] = \alpha * A @ B[indices_b, :] + \beta * C[indices_d, :]
 *
 * Assume that the all inputs are row-major unless otherwise specified.
 * All transposition applied during GEMM to L1 load. So M_A is the number of rows of the original A
 * matrix before transposition. Same for M_C.
 *
 * A: M_A x K
 * B: K_B x N
 * C: M_C x N
 * D: M_C x N
 *
 *
 */
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename Config,
          typename ArchTag>
int run_cutlass_gemm_with_operations_templated(
    const void *tensor_a,
    const void *tensor_b,
    const void *tensor_c,
    void *tensor_d,
    const int *indices_a,
    const int *indices_b,
    const int *indices_d,
    int split_k_slices,
    /* Row, col of A,B,C */ int M_A,  // Original A matrix rows
    int K,    // A matrix columns or B matrix rows when indices_b is not nullptr
    int K_B,  // Original B matrix rows when indices_b is not nullptr
    int N,    // B matrix columns or C matrix columns
    int M_C,  // C matrix rows when indices_d is not nullptr
    /* gather scatter size */ int gather_a_size,  // indices_a size, equal to indices_b when
                                                  // indices_b is not nullptr
    int scatter_d_size,                           // indices_d size
    float alpha,
    float beta) {
  using Traits = GemmOperationTraits<ElementInputA, ElementAccumulator, Config, ArchTag>;
  using ElementComputeEpilogue = ElementAccumulator;

  if constexpr (Traits::UseMixedInput) {
    return static_cast<int>(GemmStatus::kErrorMixedInputUnsupported);
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

    // Derive problem dimensions from original matrix dimensions
    int problem_m, problem_n, problem_k, N_B;

    // Currently only support trAB gather and AD gather scatter.
    if constexpr (Config::transpose_a && Config::gather_a && Config::gather_b) {
      // For A transpose: A[indices_a, :].T @ B[indices_b, :]
      // A[indices_a, :] is gather_a_size × K, transposed to K × gather_a_size
      // B[indices_b, :] is gather_a_size × N
      // Result: K × N
      assert(indices_a != nullptr);
      assert(indices_b != nullptr);
      assert(indices_d == nullptr);
      problem_m = K;  // rows in result (from transposed A)
      // TODO(cchoy): Should it be N instead of gather_a_size for problem_n
      problem_n = N;  // columns in result (from B)
      problem_k = gather_a_size;  // inner dimension
      N_B = gather_a_size;
    } else if constexpr (Config::gather_a && Config::scatter_d) {
      assert(gather_a_size == scatter_d_size);
      assert(indices_a != nullptr);
      assert(indices_b == nullptr);
      assert(indices_d != nullptr);
      // AD gather scatter: D[indices_D, :] = A[indices_A, :] @ B + C[indices_D, :]
      problem_m = gather_a_size;
      problem_n = N;
      problem_k = K;
      N_B = N;
    } else {
      // Standard GEMM without neither gather nor scatter
      assert(indices_a == nullptr);
      assert(indices_b == nullptr);
      assert(indices_d == nullptr);
      assert(K == K_B);
      assert(M_A == M_C);
      problem_m = M_A;
      problem_n = N;
      problem_k = K;
      N_B = N;
    }

    cutlass::gemm::GemmCoord problem_size(problem_m, problem_n, problem_k);

    // Calculate batch strides using original matrix dimensions.
    // Do not use the gather/scatter size.
    int64_t batch_stride_A = static_cast<int64_t>(M_A) * K * sizeof(ElementInputA);
    int64_t batch_stride_B = static_cast<int64_t>(K_B) * N * sizeof(ElementInputB);
    int64_t batch_stride_C = static_cast<int64_t>(M_C) * N * sizeof(ElementOutput);
    int64_t batch_stride_D = static_cast<int64_t>(M_C) * N * sizeof(ElementOutput);

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
      return static_cast<int>(GemmStatus::kErrorProblemNotSupported);
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(GemmStatus::kErrorKernelInitialization);
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(GemmStatus::kErrorKernelExecution);
    }
  } else {
    // Configuration not supported
    return static_cast<int>(GemmStatus::kErrorUnsupportedConfig);
  }

  return static_cast<int>(GemmStatus::kSuccess);
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
                                       int M_A,           // row of A
                                       int K,             // col of A
                                       int N,             // col of B
                                       int M_C,           // row of C
                                       int indices_size,  // indices_a and indices_d size
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
      M_A,           // M_A (original A matrix rows)
      K,             // K (A columns)
      K,             // K_B (B matrix rows)
      N,             // N (B columns)
      M_C,           // M_C (C matrix rows, different from M_A when indices_d is not nullptr)
      indices_size,  // indices_a size, equal to indices_b when indices_b is not nullptr
      indices_size,  // indices_d size
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
                                 int M_A,             // row of A (not trA)
                                 int K,               // col of A (not trA)
                                 int K_B,             // row of B (different from K since gathering)
                                 int N,               // col of B
                                 int gather_ab_size,  // indices_a and indices_b size
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
      K_B,  // M_B (original B matrix rows. Different from K when indices_b is not nullptr)
      N,    // N (B columns)
      K,    // M_C (C matrix rows. Since A is transposed, A columns are the same as C rows)
      gather_ab_size,
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
