// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

// Define layouts
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// Define CUDA SM architecture
using SmArch = cutlass::arch::Sm80;

// Define threadblock swizzling
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines
constexpr int NumStages = 5;

// Template traits for different precision combinations
template <typename ElementInput, typename ElementAccumulator>
struct GemmTraits {
  // Default to SIMT for unsupported combinations
  using MMAOp = cutlass::arch::OpClassSimt;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 8>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
  static constexpr bool SupportsGatherScatter = false;
  static constexpr bool UseMixedInput = false;
  using ArchTag = SmArch;
};

// Specialization for half input, half accumulator
template <>
struct GemmTraits<cutlass::half_t, cutlass::half_t> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsGatherScatter = true;
  static constexpr bool UseMixedInput = false;
  using ArchTag = SmArch;
};

// Specialization for half input, float accumulator
template <>
struct GemmTraits<cutlass::half_t, float> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsGatherScatter = true;
  static constexpr bool UseMixedInput = false;
  using ArchTag = SmArch;
};

// Specialization for float input, float accumulator (hardware conversion:
// FP32->FP16 with custom iterator)
template <>
struct GemmTraits<float, float> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;  // FP16 alignment (after conversion)
  static constexpr int AlignmentB = 8;  // FP16 alignment (after conversion)
  static constexpr bool SupportsGatherScatter = true;
  static constexpr bool UseMixedInput = true;  // Enable software FP32→FP16 conversion
  using ArchTag = SmArch;                      // Use standard architecture
};

#ifdef USE_BFLOAT16
// Specialization for bfloat16 input, float accumulator
template <>
struct GemmTraits<cutlass::bfloat16_t, float> {
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr bool SupportsGatherScatter = true;
  static constexpr bool UseMixedInput = false;
  using ArchTag = SmArch;
};
#endif  // USE_BFLOAT16

// Forward declaration of specialised SM80 kernel for FP32 input with
// gather/scatter
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

// Template function for CUTLASS GEMM with gather/scatter
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator>
int run_cutlass_gemm_gather_scatter_templated(const void *tensor_a,
                                              const void *tensor_b,
                                              const void *tensor_c,
                                              void *tensor_d,
                                              const int *indices_a,
                                              const int *indices_d,
                                              int split_k_slices,
                                              int M,
                                              int N,
                                              int K,
                                              int indices_size,
                                              int out_size,
                                              float alpha,
                                              float beta) {
  using Traits = GemmTraits<ElementInputA, ElementAccumulator>;
  using ElementComputeEpilogue = ElementAccumulator;

  if constexpr (Traits::UseMixedInput) {
    printf("Error: Mixed input not supported for this precision configuration\n");
    return -6;
  }

  // Determine output vector length based on element type
  constexpr int OutputVectorLength = std::is_same_v<ElementOutput, cutlass::half_t>
                                         ? (128 / cutlass::sizeof_bits<ElementOutput>::value)
                                         : (128 / cutlass::sizeof_bits<ElementOutput>::value);

  // Define epilogue operation
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                  OutputVectorLength,
                                                                  ElementAccumulator,
                                                                  ElementComputeEpilogue>;

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                                    LayoutInputA,
                                                    ElementInputB,
                                                    LayoutInputB,
                                                    ElementOutput,
                                                    LayoutOutput,
                                                    ElementAccumulator,
                                                    typename Traits::MMAOp,
                                                    typename Traits::ArchTag,
                                                    typename Traits::ShapeMMAThreadBlock,
                                                    typename Traits::ShapeMMAWarp,
                                                    typename Traits::ShapeMMAOp,
                                                    EpilogueOp,
                                                    SwizzleThreadBlock,
                                                    NumStages,
                                                    Traits::AlignmentA,
                                                    Traits::AlignmentB,
                                                    cutlass::arch::OpMultiplyAdd,
                                                    cutlass::ComplexTransform::kNone,
                                                    cutlass::ComplexTransform::kNone,
                                                    Traits::SupportsGatherScatter, /*GatherA*/
                                                    false,                         /*GatherB*/
                                                    Traits::SupportsGatherScatter  /*ScatterD*/
                                                    >;

  // Convert void pointers to appropriate types
  auto a_ptr = reinterpret_cast<const ElementInputA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementInputB *>(tensor_b);
  auto c_ptr = reinterpret_cast<const ElementOutput *>(tensor_c);
  auto d_ptr = reinterpret_cast<ElementOutput *>(tensor_d);

  if constexpr (Traits::SupportsGatherScatter) {
    // Native gather/scatter implementation
    LayoutInputA layout_a(K);
    LayoutInputB layout_b(N);
    LayoutOutput layout_c(N);
    LayoutOutput layout_d(N);

    ElementComputeEpilogue alpha_cutlass = ElementComputeEpilogue(alpha);
    ElementComputeEpilogue beta_cutlass = ElementComputeEpilogue(beta);

    cutlass::gemm::GemmCoord problem_size(indices_size, N, K);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       problem_size,
                                       split_k_slices,
                                       {alpha_cutlass, beta_cutlass},
                                       a_ptr,
                                       b_ptr,
                                       c_ptr,
                                       d_ptr,
                                       /*batch strides*/ (int64_t)(M * K * sizeof(ElementInputA)),
                                       (int64_t)(K * N * sizeof(ElementInputB)),
                                       (int64_t)(out_size * N * sizeof(ElementOutput)),
                                       (int64_t)(out_size * N * sizeof(ElementOutput)),
                                       layout_a.stride(),
                                       layout_b.stride(),
                                       layout_c.stride(),
                                       layout_d.stride(),
                                       indices_a,
                                       nullptr,
                                       indices_d};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      printf("Error: Problem size is not supported for native gather/scatter\n");
      return -1;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      printf("Error: CUTLASS kernel initialization failed for native gather/scatter\n");
      return -2;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      printf("Error: CUTLASS kernel execution failed gather/scatter\n");
      return -3;
    }
  } else {
    // Gather/scatter not supported for this configuration
    printf(
        "Error: Gather/scatter operations not supported for this "
        "precision configuration\n");
    return -4;
  }

  return 0;
}

// Explicit template instantiations to ensure they are compiled
template int
run_cutlass_gemm_gather_scatter_templated<cutlass::half_t, cutlass::half_t, float, float>(
    const void *,
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float);

template int
run_cutlass_gemm_gather_scatter_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t, float>(
    const void *,
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float);

template int
run_cutlass_gemm_gather_scatter_templated<cutlass::half_t, cutlass::half_t, float, cutlass::half_t>(
    const void *,
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float);

template int run_cutlass_gemm_gather_scatter_templated<cutlass::half_t,
                                                       cutlass::half_t,
                                                       cutlass::half_t,
                                                       cutlass::half_t>(const void *,
                                                                        const void *,
                                                                        const void *,
                                                                        void *,
                                                                        const int *,
                                                                        const int *,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        float,
                                                                        float);

#ifdef USE_BFLOAT16
// Explicit template instantiations for bfloat16 paths (FP32 accumulator)
template int run_cutlass_gemm_gather_scatter_templated<cutlass::bfloat16_t,
                                                       cutlass::bfloat16_t,
                                                       cutlass::bfloat16_t,
                                                       float>(const void *,
                                                              const void *,
                                                              const void *,
                                                              void *,
                                                              const int *,
                                                              const int *,
                                                              int,
                                                              int,
                                                              int,
                                                              int,
                                                              int,
                                                              int,
                                                              float,
                                                              float);

template int
run_cutlass_gemm_gather_scatter_templated<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float>(
    const void *,
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float);
#endif  // USE_BFLOAT16
