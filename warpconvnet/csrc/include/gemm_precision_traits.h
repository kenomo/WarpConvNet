// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "gemm_operation_traits.h"

namespace warpconvnet {
namespace gemm {

// Specialization
static inline const char *mma_tile_to_string(MMATile t) {
  switch (t) {
    case MMATile::Tile128x128x32:
      return "128x128x32";
    case MMATile::Tile128x64x32:
      return "128x64x32";
    case MMATile::Tile64x128x32:
      return "64x128x32";
    case MMATile::Tile64x64x32:
      return "64x64x32";
    default:
      return "unknown";
  }
}

// Base specializations for SM80 - these are inherited by tile-specific traits
template <>
struct GemmPrecisionTraits<cutlass::half_t, cutlass::half_t, DefaultSmArch> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = false;
  using ArchitectureTag = DefaultSmArch;
};

template <>
struct GemmPrecisionTraits<cutlass::half_t, float, DefaultSmArch> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = false;
  using ArchitectureTag = DefaultSmArch;
};

template <>
struct GemmPrecisionTraits<float, float, DefaultSmArch> {
  using MMAOp = cutlass::arch::OpClassSimt;  // Float uses SIMT
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  static constexpr bool SupportsTensorOp = false;  // Float doesn't use TensorOp
  static constexpr bool UseMixedInput = false;
  using ArchitectureTag = DefaultSmArch;
};

#ifndef DISABLE_BFLOAT16
template <>
struct GemmPrecisionTraits<cutlass::bfloat16_t, float, DefaultSmArch> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = false;
  using ArchitectureTag = DefaultSmArch;
};
#endif

// Helper macro – reuses everything from the DefaultSmArch specialisation but
// overrides the shapes.
#define WCN_DEFINE_TILE_TRAITS(                                                           \
    ElementIn, ElementAcc, TileTag, Arch, TB_M, TB_N, TB_K, W_M, W_N, W_K, M_M, M_N, M_K) \
  template <>                                                                             \
  struct GemmPrecisionTraits<ElementIn, ElementAcc, TileTag, Arch>                        \
      : public GemmPrecisionTraits<ElementIn, ElementAcc, Arch> {                         \
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>;               \
    using ShapeMMAWarp = cutlass::gemm::GemmShape<W_M, W_N, W_K>;                         \
    using ShapeMMAOp = cutlass::gemm::GemmShape<M_M, M_N, M_K>;                           \
  };

// ---- Specialisations for Tile128x128x32 ----
// clang-format off
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, cutlass::half_t, Tile128x128x32, DefaultSmArch, 128, 128, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, float, Tile128x128x32, DefaultSmArch, 128, 128, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(float, float, Tile128x128x32, DefaultSmArch, 128, 128, 32, 32, 32, 32, 16, 8, 16)
#ifndef DISABLE_BFLOAT16
WCN_DEFINE_TILE_TRAITS(cutlass::bfloat16_t, float, Tile128x128x32, DefaultSmArch, 128, 128, 32, 32, 32, 32, 16, 8, 16)
#endif

// ---- Specialisations for Tile128x64x32 ----
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, cutlass::half_t, Tile128x64x32, DefaultSmArch, 128, 64, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, float, Tile128x64x32, DefaultSmArch, 128, 64, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(float, float, Tile128x64x32, DefaultSmArch, 128, 64, 32, 32, 32, 32, 16, 8, 16)
#ifndef DISABLE_BFLOAT16
WCN_DEFINE_TILE_TRAITS(cutlass::bfloat16_t, float, Tile128x64x32, DefaultSmArch, 128, 64, 32, 32, 32, 32, 16, 8, 16)
#endif

// ---- Specialisations for Tile64x128x32 ----
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, cutlass::half_t, Tile64x128x32, DefaultSmArch, 64, 128, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, float, Tile64x128x32, DefaultSmArch, 64, 128, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(float, float, Tile64x128x32, DefaultSmArch, 64, 128, 32, 32, 32, 32, 16, 8, 16)
#ifndef DISABLE_BFLOAT16
WCN_DEFINE_TILE_TRAITS(cutlass::bfloat16_t, float, Tile64x128x32, DefaultSmArch, 64, 128, 32, 32, 32, 32, 16, 8, 16)
#endif

// ---- Specialisations for Tile64x64x32 ----
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, cutlass::half_t, Tile64x64x32, DefaultSmArch, 64, 64, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(cutlass::half_t, float, Tile64x64x32, DefaultSmArch, 64, 64, 32, 32, 32, 32, 16, 8, 16)
WCN_DEFINE_TILE_TRAITS(float, float, Tile64x64x32, DefaultSmArch, 64, 64, 32, 32, 32, 32, 16, 8, 16)
#ifndef DISABLE_BFLOAT16
WCN_DEFINE_TILE_TRAITS(cutlass::bfloat16_t, float, Tile64x64x32, DefaultSmArch, 64, 64, 32, 32, 32, 32, 16, 8, 16)
#endif

#undef WCN_DEFINE_TILE_TRAITS
// clang-format on

}  // namespace gemm
}  // namespace warpconvnet
