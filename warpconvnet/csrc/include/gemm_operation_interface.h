// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include "gemm_precision_traits.h"

namespace warpconvnet {
namespace gemm {

// Main templated function for CUTLASS GEMM with configurable operations
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename Config,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
int run_cutlass_gemm_with_operations_templated(const void *tensor_a,
                                               const void *tensor_b,
                                               const void *tensor_c,
                                               void *tensor_d,
                                               const int *indices_a,
                                               const int *indices_b,
                                               const int *indices_d,
                                               int split_k_slices,
                                               int M,
                                               int N,
                                               int K,
                                               int gather_a_size,
                                               int gather_b_size,
                                               int scatter_d_size,
                                               int scatter_CD_M_size,
                                               float alpha = 1.0f,
                                               float beta = 0.0f,
                                               cudaStream_t stream = 0);

// Convenience wrapper functions for specific operation configurations

// AD Gather-Scatter (current functionality)
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
inline int run_cutlass_gemm_ad_gather_scatter(const void *tensor_a,
                                              const void *tensor_b,
                                              const void *tensor_c,
                                              void *tensor_d,
                                              const int *indices_a,
                                              const int *indices_d,
                                              int split_k_slices,
                                              int M,
                                              int N,
                                              int K,
                                              int gather_a_size,
                                              int scatter_d_size,
                                              int scatter_CD_M_size,
                                              float alpha = 1.0f,
                                              float beta = 0.0f,
                                              cudaStream_t stream = 0) {
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigAD,
                                                    ArchTag,
                                                    LayoutA,
                                                    LayoutB,
                                                    LayoutC>(tensor_a,
                                                             tensor_b,
                                                             tensor_c,
                                                             tensor_d,
                                                             indices_a,
                                                             nullptr,
                                                             indices_d,
                                                             split_k_slices,
                                                             M,
                                                             N,
                                                             K,
                                                             gather_a_size,
                                                             0,
                                                             scatter_d_size,
                                                             scatter_CD_M_size,
                                                             alpha,
                                                             beta,
                                                             stream);
}

// AB Gather (new functionality)
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
inline int run_cutlass_gemm_ab_gather(const void *tensor_a,
                                      const void *tensor_b,
                                      const void *tensor_c,
                                      void *tensor_d,
                                      const int *indices_a,
                                      const int *indices_b,
                                      int split_k_slices,
                                      int M,
                                      int N,
                                      int K,
                                      int gather_a_size,
                                      int gather_b_size,
                                      float alpha = 1.0f,
                                      float beta = 0.0f,
                                      cudaStream_t stream = 0) {
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigAB,
                                                    ArchTag,
                                                    LayoutA,
                                                    LayoutB,
                                                    LayoutC>(tensor_a,
                                                             tensor_b,
                                                             tensor_c,
                                                             tensor_d,
                                                             indices_a,
                                                             indices_b,
                                                             nullptr,
                                                             split_k_slices,
                                                             M,
                                                             N,
                                                             K,
                                                             gather_a_size,
                                                             gather_b_size,
                                                             0,  // scatter_d_size
                                                             0,  // scatter_CD_M_size
                                                             alpha,
                                                             beta,
                                                             stream);
}

// A Gather only
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
inline int run_cutlass_gemm_a_gather(const void *tensor_a,
                                     const void *tensor_b,
                                     const void *tensor_c,
                                     void *tensor_d,
                                     const int *indices_a,
                                     int split_k_slices,
                                     int M,
                                     int N,
                                     int K,
                                     int gather_a_size,
                                     float alpha = 1.0f,
                                     float beta = 0.0f,
                                     cudaStream_t stream = 0) {
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigA,
                                                    ArchTag,
                                                    LayoutA,
                                                    LayoutB,
                                                    LayoutC>(tensor_a,
                                                             tensor_b,
                                                             tensor_c,
                                                             tensor_d,
                                                             indices_a,
                                                             nullptr,
                                                             nullptr,
                                                             split_k_slices,
                                                             M,
                                                             N,
                                                             K,
                                                             gather_a_size,
                                                             0,  // gather_b_size
                                                             0,  // scatter_d_size
                                                             0,  // scatter_CD_M_size
                                                             alpha,
                                                             beta,
                                                             stream);
}

// Forward declaration for specialized SM80 kernel for FP32 input with gather/scatter
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
                                            float alpha = 1.f,
                                            float beta = 0.f,
                                            cudaStream_t stream = 0);

}  // namespace gemm
}  // namespace warpconvnet
