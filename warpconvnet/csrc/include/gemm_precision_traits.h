// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "gemm_operation_traits.h"

namespace warpconvnet {
namespace gemm {

// Specialization for half input, half accumulator (default Sm80 + RowMajor)
template <>
struct GemmPrecisionTraits<cutlass::half_t,
                           cutlass::half_t,
                           DefaultSmArch,
                           DefaultLayoutInputA,
                           DefaultLayoutInputB,
                           DefaultLayoutOutput> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = false;

  // Template parameters as type aliases
  using ArchitectureTag = DefaultSmArch;
  using LayoutInputA = DefaultLayoutInputA;
  using LayoutInputB = DefaultLayoutInputB;
  using LayoutOutput = DefaultLayoutOutput;
};

// Specialization for half input, float accumulator (default Sm80 + RowMajor)
template <>
struct GemmPrecisionTraits<cutlass::half_t,
                           float,
                           DefaultSmArch,
                           DefaultLayoutInputA,
                           DefaultLayoutInputB,
                           DefaultLayoutOutput> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = false;

  // Template parameters as type aliases
  using ArchitectureTag = DefaultSmArch;
  using LayoutInputA = DefaultLayoutInputA;
  using LayoutInputB = DefaultLayoutInputB;
  using LayoutOutput = DefaultLayoutOutput;
};

// Specialization for float input, float accumulator (hardware conversion:
// FP32->FP16 with custom iterator) (default Sm80 + RowMajor)
template <>
struct GemmPrecisionTraits<float,
                           float,
                           DefaultSmArch,
                           DefaultLayoutInputA,
                           DefaultLayoutInputB,
                           DefaultLayoutOutput> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;  // FP16 alignment (after conversion)
  static constexpr int AlignmentB = 8;  // FP16 alignment (after conversion)
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = true;  // Enable software FP32→FP16 conversion

  // Template parameters as type aliases
  using ArchitectureTag = DefaultSmArch;
  using LayoutInputA = DefaultLayoutInputA;
  using LayoutInputB = DefaultLayoutInputB;
  using LayoutOutput = DefaultLayoutOutput;
};

#ifndef DISABLE_BFLOAT16
// Specialization for bfloat16 input, float accumulator (default Sm80 + RowMajor)
template <>
struct GemmPrecisionTraits<cutlass::bfloat16_t,
                           float,
                           DefaultSmArch,
                           DefaultLayoutInputA,
                           DefaultLayoutInputB,
                           DefaultLayoutOutput> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsTensorOp = true;
  static constexpr bool UseMixedInput = false;

  // Template parameters as type aliases
  using ArchitectureTag = DefaultSmArch;
  using LayoutInputA = DefaultLayoutInputA;
  using LayoutInputB = DefaultLayoutInputB;
  using LayoutOutput = DefaultLayoutOutput;
};
#endif  // DISABLE_BFLOAT16

}  // namespace gemm
}  // namespace warpconvnet
