// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Main header file for CUTLASS GEMM operations with gather/scatter support
// This includes all necessary headers for using the GEMM operations library

#include "gemm_operation_interface.h"
#include "gemm_operation_traits.h"
#include "gemm_operation_types.h"
#include "gemm_precision_traits.h"

// Template instantiation macros for reducing code duplication
#define INSTANTIATE_GEMM_OPERATIONS(InputA, InputB, Output, Accumulator, Config)       \
  namespace warpconvnet {                                                              \
  namespace gemm {                                                                     \
  template int run_cutlass_gemm_with_operations_templated<InputA,                      \
                                                          InputB,                      \
                                                          Output,                      \
                                                          Accumulator,                 \
                                                          Config,                      \
                                                          DefaultSmArch>(const void *, \
                                                                         const void *, \
                                                                         const void *, \
                                                                         void *,       \
                                                                         const int *,  \
                                                                         const int *,  \
                                                                         const int *,  \
                                                                         int,          \
                                                                         int,          \
                                                                         int,          \
                                                                         int,          \
                                                                         int,          \
                                                                         int,          \
                                                                         int,          \
                                                                         int,          \
                                                                         float,        \
                                                                         float);       \
  }                                                                                    \
  }

#define INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(InputA, InputB, Output, Accumulator) \
  template int run_cutlass_gemm_ad_gather_scatter<InputA, InputB, Output, Accumulator>(    \
      const void *,                                                                        \
      const void *,                                                                        \
      const void *,                                                                        \
      void *,                                                                              \
      const int *,                                                                         \
      const int *,                                                                         \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      float,                                                                               \
      float);

#define INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(InputA, InputB, Output, Accumulator)           \
  template int run_cutlass_gemm_trAB_gather<InputA, InputB, Output, Accumulator>(const void *, \
                                                                                 const void *, \
                                                                                 const void *, \
                                                                                 void *,       \
                                                                                 const int *,  \
                                                                                 const int *,  \
                                                                                 int,          \
                                                                                 int,          \
                                                                                 int,          \
                                                                                 int,          \
                                                                                 int,          \
                                                                                 int,          \
                                                                                 float,        \
                                                                                 float);
