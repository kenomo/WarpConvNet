// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace warpconvnet {
namespace gemm {

// Default layout and architecture types
using DefaultLayoutInputA = cutlass::layout::RowMajor;
using DefaultLayoutInputB = cutlass::layout::RowMajor;
using DefaultLayoutOutput = cutlass::layout::RowMajor;

// Default CUDA SM architecture
using DefaultSmArch = cutlass::arch::Sm80;

// Define threadblock swizzling
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines
constexpr int NumStages = 5;

// Operation configuration templates
template <bool GatherA, bool GatherB, bool ScatterD>
struct GemmOperationConfig {
  static constexpr bool gather_a = GatherA;
  static constexpr bool gather_b = GatherB;
  static constexpr bool scatter_d = ScatterD;

  // Helper methods for readability
  static constexpr bool has_gather() { return gather_a || gather_b; }
  static constexpr bool has_scatter() { return scatter_d; }
  static constexpr bool has_operations() { return has_gather() || has_scatter(); }
};

// Predefined configurations
using ConfigAD = GemmOperationConfig<true, false, true>;      // A gather + D scatter (current)
using ConfigAB = GemmOperationConfig<true, true, false>;      // A gather + B gather (new)
using ConfigA = GemmOperationConfig<true, false, false>;      // A gather only
using ConfigB = GemmOperationConfig<false, true, false>;      // B gather only
using ConfigD = GemmOperationConfig<false, false, true>;      // D scatter only
using ConfigNone = GemmOperationConfig<false, false, false>;  // No operations (standard GEMM)

}  // namespace gemm
}  // namespace warpconvnet
