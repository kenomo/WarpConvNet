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
          typename ArchTag = DefaultSmArch>
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
                                               float alpha = 1.0f,
                                               float beta = 0.0f);

// Convenience wrapper functions for specific operation configurations

// Note: Concrete implementations of these functions are in cutlass_gemm_gather_scatter.cu
// These wrapper functions have been moved there to avoid name collisions.

// Forward declaration for specialized SM80 kernel for FP32 input with gather/scatter
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
//                                             float alpha = 1.f,
//                                             float beta = 0.f,
//                                             cudaStream_t stream = 0);

}  // namespace gemm
}  // namespace warpconvnet
