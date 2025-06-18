// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Main header file for CUTLASS GEMM operations with gather/scatter support
// This includes all necessary headers for using the GEMM operations library

#include "gemm_operation_interface.h"
#include "gemm_operation_traits.h"
#include "gemm_operation_types.h"
#include "gemm_precision_traits.h"

#define INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                 \
    InputA, InputB, Output, Accumulator, Tile, Arch)                                   \
  template int                                                                         \
  run_cutlass_gemm_ad_gather_scatter<InputA, InputB, Output, Accumulator, Tile, Arch>( \
      const void *,                                                                    \
      const void *,                                                                    \
      const void *,                                                                    \
      void *,                                                                          \
      const int *,                                                                     \
      const int *,                                                                     \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      float,                                                                           \
      float);

#define INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(InputA, InputB, Output, Accumulator, Tile, Arch) \
  template int run_cutlass_gemm_trAB_gather<InputA, InputB, Output, Accumulator, Tile, Arch>(    \
      const void *,                                                                              \
      const void *,                                                                              \
      const void *,                                                                              \
      void *,                                                                                    \
      const int *,                                                                               \
      const int *,                                                                               \
      int,                                                                                       \
      int,                                                                                       \
      int,                                                                                       \
      int,                                                                                       \
      int,                                                                                       \
      int,                                                                                       \
      float,                                                                                     \
      float);

// -----------------------------------------------------------------------------
// Helper macros to instantiate standard precision combinations for AD gather
// scatter for a given architecture / tile tag.  Keeps call-sites concise.
// -----------------------------------------------------------------------------
#define _WCN_INSTANTIATE_AD_GS_HALF_VARIANTS(TileTag, ArchTag)                    \
  INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                  \
      cutlass::half_t, cutlass::half_t, float, float, TileTag, ArchTag)           \
  INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                  \
      cutlass::half_t, cutlass::half_t, cutlass::half_t, float, TileTag, ArchTag) \
  INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                  \
      cutlass::half_t, cutlass::half_t, float, cutlass::half_t, TileTag, ArchTag) \
  INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                  \
      cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, TileTag, ArchTag)

#ifndef DISABLE_BFLOAT16
#define _WCN_INSTANTIATE_AD_GS_BF16_VARIANTS(TileTag, ArchTag)                                \
  INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                              \
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, TileTag, ArchTag) \
  INSTANTIATE_AD_GATHER_SCATTER_GEMM_OPERATIONS(                                              \
      cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, TileTag, ArchTag)
#else
#define _WCN_INSTANTIATE_AD_GS_BF16_VARIANTS(TileTag, ArchTag)
#endif

#define INSTANTIATE_AD_GS_FOR_ARCH(TileTag, ArchTag)     \
  _WCN_INSTANTIATE_AD_GS_HALF_VARIANTS(TileTag, ArchTag) \
  _WCN_INSTANTIATE_AD_GS_BF16_VARIANTS(TileTag, ArchTag)

// ---- helpers for TrAB gather (A transpose + A,B gather) ---------------------
#define _WCN_INSTANTIATE_TRAB_HALF_VARIANTS(TileTag, ArchTag)                      \
  INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(                                         \
      cutlass::half_t, cutlass::half_t, float, float, TileTag, ArchTag);           \
  INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(                                         \
      cutlass::half_t, cutlass::half_t, cutlass::half_t, float, TileTag, ArchTag); \
  INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(                                         \
      cutlass::half_t, cutlass::half_t, float, cutlass::half_t, TileTag, ArchTag); \
  INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(                                         \
      cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, TileTag, ArchTag);

#ifndef DISABLE_BFLOAT16
#define _WCN_INSTANTIATE_TRAB_BF16_VARIANTS(TileTag, ArchTag)                                  \
  INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(                                                     \
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, TileTag, ArchTag); \
  INSTANTIATE_TrAB_GATHER_GEMM_OPERATIONS(                                                     \
      cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, TileTag, ArchTag);
#else
#define _WCN_INSTANTIATE_TRAB_BF16_VARIANTS(TileTag, ArchTag)
#endif

#define INSTANTIATE_TRAB_FOR_ARCH(TileTag, ArchTag)     \
  _WCN_INSTANTIATE_TRAB_HALF_VARIANTS(TileTag, ArchTag) \
  _WCN_INSTANTIATE_TRAB_BF16_VARIANTS(TileTag, ArchTag)
