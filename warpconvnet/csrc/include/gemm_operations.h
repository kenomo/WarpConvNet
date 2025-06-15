// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Main header file for CUTLASS GEMM operations with gather/scatter support
// This includes all necessary headers for using the GEMM operations library

#include "gemm_operation_interface.h"
#include "gemm_operation_traits.h"
#include "gemm_operation_types.h"
#include "gemm_precision_traits.h"

namespace warpconvnet {
namespace gemm {

// Version and feature information
constexpr int GEMM_OPERATIONS_VERSION_MAJOR = 1;
constexpr int GEMM_OPERATIONS_VERSION_MINOR = 0;
constexpr int GEMM_OPERATIONS_VERSION_PATCH = 0;

// Feature flags
constexpr bool SUPPORTS_GATHER_A = true;
constexpr bool SUPPORTS_GATHER_B = true;
constexpr bool SUPPORTS_SCATTER_D = true;
constexpr bool SUPPORTS_MIXED_PRECISION = false;

// Utility functions for runtime configuration validation
template <typename ElementInput,
          typename ElementAccumulator,
          typename Config,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
constexpr bool is_supported_combination() {
  using Traits = GemmOperationTraits<ElementInput,
                                     ElementAccumulator,
                                     Config,
                                     ArchTag,
                                     LayoutA,
                                     LayoutB,
                                     LayoutC>;
  return Traits::IsValidConfiguration();
}

}  // namespace gemm
}  // namespace warpconvnet

// Convenience macros for common usage patterns
#define WARPCONVNET_GEMM_AD_GATHER_SCATTER(input_type, output_type, accum_type) \
  warpconvnet::gemm::                                                           \
      run_cutlass_gemm_ad_gather_scatter<input_type, input_type, output_type, accum_type>

#define WARPCONVNET_GEMM_AB_GATHER(input_type, output_type, accum_type) \
  warpconvnet::gemm::run_cutlass_gemm_ab_gather<input_type, input_type, output_type, accum_type>

#define WARPCONVNET_GEMM_A_GATHER(input_type, output_type, accum_type) \
  warpconvnet::gemm::run_cutlass_gemm_a_gather<input_type, input_type, output_type, accum_type>

// Template instantiation macros for reducing code duplication
#define INSTANTIATE_GEMM_OPERATIONS(InputA, InputB, Output, Accumulator, Config)              \
  namespace warpconvnet {                                                                     \
  namespace gemm {                                                                            \
  template int run_cutlass_gemm_with_operations_templated<InputA,                             \
                                                          InputB,                             \
                                                          Output,                             \
                                                          Accumulator,                        \
                                                          Config,                             \
                                                          DefaultSmArch,                      \
                                                          DefaultLayoutInputA,                \
                                                          DefaultLayoutInputB,                \
                                                          DefaultLayoutOutput>(const void *,  \
                                                                               const void *,  \
                                                                               const void *,  \
                                                                               void *,        \
                                                                               const int *,   \
                                                                               const int *,   \
                                                                               const int *,   \
                                                                               int,           \
                                                                               int,           \
                                                                               int,           \
                                                                               int,           \
                                                                               int,           \
                                                                               int,           \
                                                                               int,           \
                                                                               int,           \
                                                                               float,         \
                                                                               float,         \
                                                                               cudaStream_t); \
  }                                                                                           \
  }

#define INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(InputA, InputB, Output, Accumulator)     \
  template int run_cutlass_gemm_gather_scatter_templated<InputA, InputB, Output, Accumulator>( \
      const void *,                                                                            \
      const void *,                                                                            \
      const void *,                                                                            \
      void *,                                                                                  \
      const int *,                                                                             \
      const int *,                                                                             \
      int,                                                                                     \
      int,                                                                                     \
      int,                                                                                     \
      int,                                                                                     \
      int,                                                                                     \
      int,                                                                                     \
      float,                                                                                   \
      float);
