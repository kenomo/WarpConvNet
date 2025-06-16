// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "gemm_operation_types.h"

namespace warpconvnet {
namespace gemm {

// Base template traits for different precision combinations
template <typename ElementInput, typename ElementAccumulator, typename ArchTag = DefaultSmArch>
struct GemmPrecisionTraits {
  // Default to SIMT for unsupported combinations
  using MMAOp = cutlass::arch::OpClassSimt;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
  static constexpr bool SupportsTensorOp = false;
  static constexpr bool UseMixedInput = false;

  // Architecture tag
  using ArchitectureTag = ArchTag;
};

// Enhanced traits that combine precision and operation configuration
template <typename ElementInput,
          typename ElementAccumulator,
          typename Config,
          typename ArchTag = DefaultSmArch>
struct GemmOperationTraits : public GemmPrecisionTraits<ElementInput, ElementAccumulator, ArchTag> {
  using Base = GemmPrecisionTraits<ElementInput, ElementAccumulator, ArchTag>;

  // Use layouts from Config (which handles transpose logic)
  using LayoutInputA = typename Config::LayoutInputA;
  using LayoutInputB = typename Config::LayoutInputB;
  using LayoutOutput = typename Config::LayoutOutput;

  // Operation configuration
  static constexpr bool SupportsGatherA = Config::gather_a && Base::SupportsTensorOp;
  static constexpr bool SupportsGatherB = Config::gather_b && Base::SupportsTensorOp;
  static constexpr bool SupportsScatterD = Config::scatter_d && Base::SupportsTensorOp;
  static constexpr bool SupportsTransposeA = Config::transpose_a;
  static constexpr bool SupportsTransposeB = Config::transpose_b;

  // Validation helpers
  static constexpr bool IsValidConfiguration() {
    return Config::has_operations() ? Base::SupportsTensorOp : true;
  }

  // Get operation description for debugging
  static constexpr const char* GetConfigName() {
    if constexpr (Config::gather_a && Config::scatter_d && !Config::gather_b) {
      return "AD_GatherScatter";
    } else if constexpr (Config::gather_a && Config::gather_b && !Config::scatter_d &&
                         !Config::transpose_a) {
      return "AB_Gather";
    } else if constexpr (Config::gather_a && Config::gather_b && !Config::scatter_d &&
                         Config::transpose_a) {
      return "TrAB_Gather";
    } else if constexpr (Config::gather_a && !Config::gather_b && !Config::scatter_d) {
      return "A_Gather";
    } else if constexpr (!Config::gather_a && Config::gather_b && !Config::scatter_d) {
      return "B_Gather";
    } else if constexpr (!Config::gather_a && !Config::gather_b && Config::scatter_d) {
      return "D_Scatter";
    } else if constexpr (!Config::has_operations()) {
      return "Standard_GEMM";
    } else {
      return "Custom_Config";
    }
  }
};

}  // namespace gemm
}  // namespace warpconvnet
