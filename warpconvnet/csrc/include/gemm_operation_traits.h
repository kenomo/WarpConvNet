// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "gemm_operation_types.h"

namespace warpconvnet {
namespace gemm {

// Base template traits for different precision combinations
template <typename ElementInput,
          typename ElementAccumulator,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
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

  // Template parameters as type aliases
  using ArchitectureTag = ArchTag;
  using LayoutInputA = LayoutA;
  using LayoutInputB = LayoutB;
  using LayoutOutput = LayoutC;
};

// Enhanced traits that combine precision and operation configuration
template <typename ElementInput,
          typename ElementAccumulator,
          typename Config,
          typename ArchTag = DefaultSmArch,
          typename LayoutA = DefaultLayoutInputA,
          typename LayoutB = DefaultLayoutInputB,
          typename LayoutC = DefaultLayoutOutput>
struct GemmOperationTraits : public GemmPrecisionTraits<ElementInput,
                                                        ElementAccumulator,
                                                        ArchTag,
                                                        LayoutA,
                                                        LayoutB,
                                                        LayoutC> {
  using Base =
      GemmPrecisionTraits<ElementInput, ElementAccumulator, ArchTag, LayoutA, LayoutB, LayoutC>;

  // Operation configuration
  static constexpr bool SupportsGatherA = Config::gather_a && Base::SupportsTensorOp;
  static constexpr bool SupportsGatherB = Config::gather_b && Base::SupportsTensorOp;
  static constexpr bool SupportsScatterD = Config::scatter_d && Base::SupportsTensorOp;

  // Validation helpers
  static constexpr bool IsValidConfiguration() {
    return Config::has_operations() ? Base::SupportsTensorOp : true;
  }

  // Get operation description for debugging
  static constexpr const char* GetConfigName() {
    if constexpr (Config::gather_a && Config::scatter_d && !Config::gather_b) {
      return "AD_GatherScatter";
    } else if constexpr (Config::gather_a && Config::gather_b && !Config::scatter_d) {
      return "AB_Gather";
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
