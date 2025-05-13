# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union, Generator

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.coords.search.cache import IntSearchCache, IntSearchCacheKey
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.coords.ops.expand import (
    expand_coords,
)
from warpconvnet.nn.functional.sparse_pool import sparse_reduce
from warpconvnet.utils.ntuple import ntuple

# Add necessary imports for CuPy and kernel loading
import cupy as cp
import math
import os
from torch.utils.dlpack import to_dlpack as torch_to_dlpack, from_dlpack as torch_from_dlpack
from warpconvnet.utils.cuda_utils import load_kernel


_KERNEL_CACHE = {}
_BLOCK_SIZE = 16
_MAX_GRID_Y = 65535

# Dtype mapping for kernel template
_DTYPE_TO_CPP_STR = {
    torch.float32: "float",
    torch.float64: "double",
}

# Itype mapping for kernel template
_ITYPE_TO_CPP_STR = {
    torch.int32: "int",
}

_TORCH_DTYPE_TO_CUPY_DTYPE = {
    torch.float16: cp.float16,
    torch.float32: cp.float32,
    torch.float64: cp.float64,
    torch.int32: cp.int32,
}


def _get_cuda_kernel(
    kernel_name: str, feature_dtype: torch.dtype, itype: torch.dtype, block_size: int = 32
):
    dtype_str = _DTYPE_TO_CPP_STR.get(feature_dtype)
    itype_str = "int"

    if dtype_str is None:
        raise ValueError(f"Unsupported feature_dtype for CUDA kernel: {feature_dtype}")

    block_size_str = str(block_size)
    cache_key = (kernel_name, dtype_str, itype_str, block_size_str)

    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_kernel_file = os.path.join(current_script_dir, "cuda", "sparse_conv.cu")

    if not os.path.exists(cuda_kernel_file):
        raise FileNotFoundError(
            f"CUDA kernel file not found: {cuda_kernel_file}. CWD: {os.getcwd()}"
        )

    templated_kernel_name = f"{kernel_name}_{dtype_str}_{itype_str}_b{block_size_str}"
    loaded_kernel = load_kernel(templated_kernel_name, cuda_kernel_file)
    _KERNEL_CACHE[cache_key] = loaded_kernel
    return loaded_kernel


class STRIDED_CONV_MODE(Enum):
    REDUCE_AND_STRIDE = "reduce_and_stride"  # Apply convolution on the pooled input. This increases the density of the input
    STRIDE_ONLY = "stride_only"


class SPATIALLY_SPARSE_CONV_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    EXPLICIT_GEMM_BATCHED = "explicit_gemm_batched"
    IMPLICIT_GEMM = "implicit_gemm"


@dataclass
class SpatiallySparseConvConfig:
    stride: Union[int, List[int], Tuple[int, ...]] = 1
    padding: Union[int, Tuple[int, ...]] = 0
    dilation: Union[int, Tuple[int, ...]] = 1
    conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM


class SpatiallySparseConvImplicitGEMMFunction(Function):
    """
    Implementation of the spatially sparse convolution using Implicit GEMM
    proposed in
    https://github.com/NVIDIA/MinkowskiEngine/blob/master/src/convolution_kernel.cu
    """

    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        compute_dtype: Optional[torch.dtype] = None,
        fwd_block_size: int = 32,
        bwd_block_size: int = 32,
    ) -> Float[Tensor, "M C_out"]:
        """
        Perform a sparse convolution on the input tensor using the Implicit GEMM CUDA kernel.
        """
        device = in_features.device
        feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
        cupy_feature_dtype = _TORCH_DTYPE_TO_CUPY_DTYPE[feature_dtype]
        # Assuming kernel map indices are or can be converted to int32 for the CUDA kernel
        index_dtype = torch.int32

        in_features = in_features.contiguous().detach()
        weight = weight.contiguous().detach()

        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape

        output_feature_tensor_cp = cp.zeros((num_out_coords, C_out), dtype=cupy_feature_dtype)

        if num_out_coords == 0 or K == 0 or C_in == 0 or C_out == 0 or in_features.shape[0] == 0:
            # Handle empty cases: if no output points, or if any dimension is zero.
            ctx.save_for_backward(in_features, weight, kernel_map)
            ctx.num_out_coords = num_out_coords  # Save for backward pass context
            return torch.from_dlpack(output_feature_tensor_cp)

        matmul_kernel = _get_cuda_kernel("matmul", feature_dtype, index_dtype, fwd_block_size)
        batched_features_cp = cp.from_dlpack(in_features)

        for k_idx, (in_map_k, out_map_k) in enumerate(kernel_map):
            # Check if the offset is 0, 0, 0...
            kernel_map_offset = kernel_map.offsets[k_idx]
            if torch.all(kernel_map_offset == 0) and in_map_k.shape[0] == N_in:
                current_weight_k_cp = cp.from_dlpack(weight[k_idx])
                output_feature_tensor_cp += cp.matmul(batched_features_cp, current_weight_k_cp)
                continue

            in_map_k = in_map_k.to(device=device, dtype=index_dtype).contiguous()
            out_map_k = out_map_k.to(device=device, dtype=index_dtype).contiguous()

            num_active_pairs = in_map_k.shape[0]
            if num_active_pairs == 0:
                continue

            current_weight_k_cp: Float[cp.ndarray, "C_in C_out"] = cp.from_dlpack(weight[k_idx])
            in_map_k_cp = cp.from_dlpack(in_map_k)
            out_map_k_cp = cp.from_dlpack(out_map_k)

            # Kernel args: A, wA, hA, B, wB, hB, C, in_map, out_map
            # A: batched_features_cp, wA: C_in, hA: num_active_pairs
            # B: current_weight_k_cp, wB: C_out, hB: C_in
            # C: output_feature_tensor_cp

            threads_y = fwd_block_size
            threads_x = fwd_block_size

            blocks_y = math.ceil(num_active_pairs / threads_y)
            blocks_x = math.ceil(C_out / threads_x)

            if blocks_y > 0 and blocks_x > 0 and blocks_y < _MAX_GRID_Y:
                matmul_kernel(
                    (blocks_x, blocks_y, 1),
                    (threads_x, threads_y, 1),
                    (
                        batched_features_cp,
                        C_in,
                        num_active_pairs,
                        current_weight_k_cp,
                        C_out,
                        C_in,
                        output_feature_tensor_cp,
                        in_map_k_cp,
                        out_map_k_cp,
                    ),
                )
            elif blocks_y >= _MAX_GRID_Y:
                num_iters = math.ceil(blocks_y / _MAX_GRID_Y)
                for iter_idx in range(num_iters):
                    start_idx = iter_idx * _MAX_GRID_Y * threads_y
                    end_idx = min(start_idx + _MAX_GRID_Y * threads_y, num_active_pairs)
                    curr_active_pairs = end_idx - start_idx
                    if curr_active_pairs == 0:
                        continue

                    matmul_kernel(
                        (blocks_x, math.ceil(curr_active_pairs / threads_y), 1),
                        (threads_x, threads_y, 1),
                        (
                            batched_features_cp,
                            C_in,
                            curr_active_pairs,
                            current_weight_k_cp,
                            C_out,
                            C_in,
                            output_feature_tensor_cp,
                            in_map_k_cp[start_idx:end_idx],
                            out_map_k_cp[start_idx:end_idx],
                        ),
                    )

        # output_feature_tensor is already updated in-place by output_feature_tensor_cp
        # No need to recreate from_dlpack unless output_feature_tensor_cp was a copy,
        # but PyTorch tensors from_dlpack share memory.

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.num_out_coords = num_out_coords  # Save for backward pass context
        ctx.fwd_block_size = fwd_block_size
        ctx.bwd_block_size = bwd_block_size
        return torch.from_dlpack(output_feature_tensor_cp)

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Float[Tensor, "N C_in"],
        Float[Tensor, "K C_in C_out"],
        None,
        None,
        None,
        None,
        None,
    ]:
        """
        Perform the backward pass of the sparse convolution using Implicit GEMM CUDA kernel.
        Returns gradients for: batched_features, batch_offsets, weight, kernel_map, conv_algo
        """
        batched_features, weight = ctx.saved_tensors
        num_out_coords = ctx.num_out_coords  # Retrieve from context

        feature_dtype = batched_features.dtype
        cupy_feature_dtype = _TORCH_DTYPE_TO_CUPY_DTYPE[feature_dtype]
        index_dtype = torch.int32  # Must match forward

        N_in, C_in = batched_features.shape
        K, _, C_out = weight.shape

        # Early exit if no work to do or if forward pass produced empty output
        if (
            num_out_coords == 0
            or K == 0
            or C_in == 0
            or C_out == 0
            or N_in == 0
            or grad_output.shape[0] == 0
        ):
            grad_in_features = (
                torch.zeros_like(batched_features) if N_in > 0 and C_in > 0 else None
            )
            grad_weight = torch.zeros_like(weight) if K > 0 and C_in > 0 and C_out > 0 else None
            return grad_in_features, None, grad_weight, None, None

        kernel_map = ctx.kernel_map
        bwd_block_size = ctx.bwd_block_size

        grad_in_features_cp = cp.zeros((N_in, C_in), dtype=cupy_feature_dtype)
        grad_weight_tensor_cp = cp.zeros((K, C_in, C_out), dtype=cupy_feature_dtype)

        matmul2_kernel = _get_cuda_kernel("matmul2", feature_dtype, index_dtype, bwd_block_size)

        grad_output_cp = cp.from_dlpack(grad_output)
        batched_features_cp = cp.from_dlpack(batched_features)

        for k_idx, (in_map_k, out_map_k) in enumerate(kernel_map):
            kernel_map_offset = kernel_map.offsets[k_idx]
            if torch.all(kernel_map_offset == 0) and in_map_k.shape[0] == N_in:
                current_weight_k_cp: Float[cp.ndarray, "C_in C_out"] = cp.from_dlpack(
                    weight[k_idx]
                )
                grad_weight_tensor_cp[k_idx] += cp.matmul(grad_in_features_cp.T, grad_output_cp)
                grad_in_features_cp += cp.matmul(grad_output_cp, current_weight_k_cp.T)
                continue

            num_active_pairs = in_map_k.shape[0]  # also out_map_k.shape[0]
            if num_active_pairs == 0:
                continue

            current_weight_k_cp = cp.from_dlpack(weight[k_idx])
            in_map_k_cp = cp.from_dlpack(in_map_k)
            out_map_k_cp = cp.from_dlpack(out_map_k)

            grad_weight_k_cp = grad_weight_tensor_cp[k_idx]

            # Kernel args: A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map
            # A (grad_out): grad_output_cp, wA: C_out, hA: num_active_pairs
            # B (weight_k): current_weight_k_cp, wB: C_out, hB: C_in
            # D (in_feat): batched_features_cp, wD: C_in, hD: num_active_pairs (num_active_pairs based on in_map used for D)
            # C (grad_in): grad_in_features_cp
            # E (grad_w_k): grad_weight_k_cp

            threads_y = bwd_block_size
            threads_x = bwd_block_size
            blocks_x = math.ceil(C_in / threads_x)
            blocks_y = math.ceil(num_active_pairs / threads_y)

            if blocks_y > 0 and blocks_x > 0 and blocks_y < _MAX_GRID_Y:
                matmul2_kernel(
                    (blocks_x, blocks_y, 1),
                    (threads_x, threads_y, 1),
                    (
                        grad_output_cp,
                        C_out,
                        num_active_pairs,
                        current_weight_k_cp,
                        C_out,
                        C_in,
                        batched_features_cp,
                        C_in,
                        num_active_pairs,
                        grad_in_features_cp,
                        grad_weight_k_cp,  # Pass the slice for E
                        in_map_k_cp,
                        out_map_k_cp,
                    ),
                )
            elif blocks_y >= _MAX_GRID_Y:
                num_iters = math.ceil(blocks_y / _MAX_GRID_Y)
                for iter_idx in range(num_iters):
                    start_idx = iter_idx * _MAX_GRID_Y * threads_y
                    end_idx = min(start_idx + _MAX_GRID_Y * threads_y, num_active_pairs)
                    curr_active_pairs = end_idx - start_idx
                    if curr_active_pairs == 0:
                        continue

                    matmul2_kernel(
                        (blocks_x, math.ceil(curr_active_pairs / threads_y), 1),
                        (threads_x, threads_y, 1),
                        (
                            grad_output_cp,
                            C_out,
                            curr_active_pairs,
                            current_weight_k_cp,
                            C_out,
                            C_in,
                            batched_features_cp,
                            C_in,
                            curr_active_pairs,
                            grad_in_features_cp,
                            grad_weight_k_cp,
                            in_map_k_cp[start_idx:end_idx],
                            out_map_k_cp[start_idx:end_idx],
                        ),
                    )

        # Gradients are accumulated in-place in grad_in_features and grad_weight_tensor
        # via their CuPy counterparts.

        return (
            torch.from_dlpack(grad_in_features_cp),
            torch.from_dlpack(grad_weight_tensor_cp),
            None,
            None,
            None,
            None,
            None,
        )


def _maybe_cast(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Fast dtype conversion if needed."""
    return tensor if dtype is None else tensor.to(dtype=dtype)


class SpatiallySparseConvExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> Float[Tensor, "M C_out"]:
        device = in_features.device

        # Convert input features and weight once
        comp_in_feats = _maybe_cast(in_features, compute_dtype)
        comp_weight = _maybe_cast(weight, compute_dtype)

        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=comp_in_feats.dtype
        )

        for i, (in_map, out_map) in enumerate(kernel_map):
            in_map = in_map.to(device)
            out_map = out_map.to(device)

            num_active_pairs = in_map.shape[0]
            if num_active_pairs == 0:
                continue

            curr_out_features = torch.matmul(comp_in_feats[in_map], comp_weight[i])
            output_feature_tensor[out_map] += curr_out_features.to(device=device)

        ctx.kernel_map = kernel_map
        ctx.compute_dtype = compute_dtype
        ctx.save_for_backward(in_features, weight)

        return output_feature_tensor.to(dtype=in_features.dtype)

    @staticmethod
    def backward(
        ctx, grad_output: Float[Tensor, "M C_out"]
    ) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        compute_dtype = ctx.compute_dtype
        device = in_features.device  # Get device from input

        # When compute_dtype is None, use input's dtype
        dtype_to_use = compute_dtype if compute_dtype is not None else in_features.dtype

        # Convert all tensors to the same dtype
        comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
        comp_weight = weight.to(device=device, dtype=dtype_to_use)
        comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)

        # Create gradients in same dtype
        grad_in_features = torch.zeros_like(comp_in_feats, device=device)
        grad_weight = torch.zeros_like(comp_weight, device=device)

        for i, (in_map, out_map) in enumerate(kernel_map):
            curr_grad_output = comp_grad_output[out_map]
            curr_in_feats = comp_in_feats[in_map]
            curr_weight = comp_weight[i]

            grad_in_features[in_map] += torch.matmul(curr_grad_output, curr_weight.T)
            grad_weight[i] += torch.matmul(curr_in_feats.T, curr_grad_output)

        return (
            grad_in_features.to(dtype=in_features.dtype),
            grad_weight.to(dtype=weight.dtype),
            None,
            None,
            None,
        )


class SpatiallySparseConvBatchedExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        matmul_batch_size: int,
        compute_dtype: torch.dtype = torch.float32,
    ) -> Float[Tensor, "M C_out"]:
        device = in_features.device
        # Create output feature tensor with a dummy row
        output_feature_tensor = torch.zeros(
            num_out_coords + 1, weight.shape[-1], device=device, dtype=in_features.dtype
        )

        for i_start in range(0, len(kernel_map), matmul_batch_size):
            i_end = min(i_start + matmul_batch_size, len(kernel_map))
            # Get the input and output maps of shape B x N
            in_maps, out_maps = kernel_map.get_batch(i_start, i_end, out_format="tensor")
            # Get the input and output features
            curr_in_features = in_features[in_maps.clip(min=0)].to(dtype=compute_dtype)

            # bmm. B x (N + 1) x C_in @ B x C_in x C_out -> B x (N + 1) x C_out
            curr_out_features_batch = torch.bmm(curr_in_features, weight[i_start:i_end])

            # Add to output feature tensor. all negative indices will map to the dummy row
            output_feature_tensor[out_maps + 1] += curr_out_features_batch

        ctx.compute_dtype = compute_dtype
        ctx.kernel_map = kernel_map
        ctx.matmul_batch_size = matmul_batch_size
        ctx.save_for_backward(in_features, weight)

        return output_feature_tensor[1:].clone().to(dtype=in_features.dtype)

    @staticmethod
    def backward(
        ctx, grad_output: Float[Tensor, "M C_out"]
    ) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        matmul_batch_size = ctx.matmul_batch_size
        compute_dtype = ctx.compute_dtype
        device = in_features.device  # Get device from input
        # Add a dummy row to the beginning of the in_maps tensor
        grad_in_features = torch.zeros(
            in_features.shape[0] + 1,
            in_features.shape[1],
            device=device,
            dtype=compute_dtype,
        )
        grad_weight = torch.zeros_like(weight, dtype=compute_dtype)

        for i_start in range(0, len(kernel_map), matmul_batch_size):
            i_end = min(i_start + matmul_batch_size, len(kernel_map))
            # Get the input and output maps of shape B x N
            in_maps, out_maps = kernel_map.get_batch(i_start, i_end, out_format="tensor")

            # Get the input and output features
            invalid_mask = in_maps < -1
            curr_in_features = in_features[in_maps.clip(min=0)]  # B x N x C_in
            curr_out_features = grad_output[out_maps.clip(min=0)]  # B x N x C_out

            # zero out invalid rows
            curr_in_features[invalid_mask] = 0
            curr_out_features[invalid_mask] = 0

            # matmul. B x N x C_out @ B x C_out x C_in -> B x N x C_in
            grad_in_features[in_maps + 1] += torch.bmm(
                curr_out_features, weight[i_start:i_end].transpose(1, 2)
            )

            # matmul. B x C_in x (N + 1) @ B x (N + 1) x C_out -> B x C_in x C_out
            grad_weight[i_start:i_end] += torch.bmm(
                curr_in_features.transpose(1, 2), curr_out_features
            )

        return (
            grad_in_features[1:].clone().to(dtype=in_features.dtype),
            grad_weight.to(dtype=weight.dtype),
            None,
            None,
            None,
            None,
        )


def spatially_sparse_conv(
    input_sparse_tensor: Geometry,
    weight: Float[Tensor, "K C_out C_in"],
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    kernel_dilation: Union[int, List[int], Tuple[int, ...]] = 1,
    bias: Optional[Float[Tensor, "C_out"]] = None,  # noqa: F821
    kernel_search_batch_size: Optional[int] = None,
    kernel_matmul_batch_size: int = 2,
    generative: bool = False,
    output_spatially_sparse_tensor: Optional[Geometry] = None,
    transposed: bool = False,
    conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
    stride_reduce: str = "max",
    out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "hashmap",
    compute_dtype: torch.dtype = torch.float32,
    implicit_matmul_fwd_block_size: Optional[int] = None,
    implicit_matmul_bwd_block_size: Optional[int] = None,
) -> Geometry:
    """
    Perform spatially sparse convolution on the input tensor using the specified algorithm.
    Spatially sparse and feature sparse is not supported yet.

    If stride is not 1, the kernel map will be generated by stride_mode.

    If generative, the output coordinates will be expanded by (kernel size // 2) all directions.

    For transposed convolution, the output coordinates should be provided along with the
    output coordinate stride.
    """
    num_spatial_dims = input_sparse_tensor.num_spatial_dims
    kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
    kernel_dilation = ntuple(kernel_dilation, ndim=num_spatial_dims)
    stride = ntuple(stride, ndim=num_spatial_dims)

    num_total_kernels = np.prod(kernel_size)
    if np.prod(kernel_size) == 1 and np.prod(stride) == 1:
        out_feature_tensor = input_sparse_tensor.feature_tensor @ weight[0]
        if bias is not None:
            out_feature_tensor += bias
        return input_sparse_tensor.replace(
            batched_features=out_feature_tensor,
        )

    if kernel_search_batch_size is None:
        kernel_search_batch_size = max(num_total_kernels // kernel_size[0], 8)

    in_tensor_stride = input_sparse_tensor.stride
    if in_tensor_stride is None:
        in_tensor_stride = ntuple(1, ndim=num_spatial_dims)

    if transposed and not generative:
        assert (
            output_spatially_sparse_tensor is not None
        ), "Output spatially sparse tensor is required for transposed convolution"

    if not transposed:
        out_tensor_stride = tuple(o * s for o, s in zip(stride, in_tensor_stride))
    else:  # transposed
        if (
            output_spatially_sparse_tensor is not None
            and output_spatially_sparse_tensor.stride is not None
        ):
            out_tensor_stride = output_spatially_sparse_tensor.stride
        else:
            out_tensor_stride = ntuple(1, ndim=num_spatial_dims)
        # At least one of the output stride dimensions should be smaller than the input stride dimensions
        assert any(
            o < i for o, i in zip(out_tensor_stride, in_tensor_stride)
        ), "Output stride is larger than input stride"

    # Generate output coordinates and kernel map
    batch_indexed_out_coords, out_offsets, kernel_map = generate_output_coords_and_kernel_map(
        input_sparse_tensor=input_sparse_tensor,
        kernel_size=kernel_size,
        kernel_dilation=kernel_dilation,
        stride=stride,
        generative=generative,
        transposed=transposed,
        output_spatially_sparse_tensor=output_spatially_sparse_tensor,
        kernel_search_batch_size=kernel_search_batch_size,
        stride_mode=stride_mode,
        out_code_backend=out_code_backend,
    )

    num_out_coords = batch_indexed_out_coords.shape[0]

    if stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and any(s != 1 for s in stride):
        input_sparse_tensor = sparse_reduce(
            input_sparse_tensor,
            kernel_size=stride,  # reduce by stride
            stride=stride,
            reduction=stride_reduce,
        )

    # Call explicit gemm
    if conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM:
        out_feature_tensor = SpatiallySparseConvExplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
        )
    elif conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM_BATCHED:
        raise NotImplementedError("Batched explicit gemm is not implemented yet")
        out_feature_tensor = SpatiallySparseConvBatchedExplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            weight,
            kernel_map,
            num_out_coords,
            kernel_matmul_batch_size,
            compute_dtype,
        )
    elif conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.IMPLICIT_GEMM:
        if implicit_matmul_fwd_block_size is None:
            implicit_matmul_fwd_block_size = 16
        if implicit_matmul_bwd_block_size is None:
            implicit_matmul_bwd_block_size = 16

        out_feature_tensor = SpatiallySparseConvImplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            implicit_matmul_fwd_block_size,
            implicit_matmul_bwd_block_size,
        )
    else:
        raise ValueError(f"Unsupported convolution algorithm: {conv_algo}")

    if bias is not None:
        out_feature_tensor += bias

    out_offsets = out_offsets.cpu().int()
    return input_sparse_tensor.replace(
        batched_coordinates=IntCoords(
            batch_indexed_out_coords[:, 1:],
            offsets=out_offsets,
        ),
        batched_features=out_feature_tensor,
        stride=out_tensor_stride,
    )


def generate_output_coords_and_kernel_map(
    input_sparse_tensor: Geometry,
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    stride: Tuple[int, ...],
    generative: bool,
    transposed: bool,
    output_spatially_sparse_tensor: Optional[Geometry],
    kernel_search_batch_size: int,
    stride_mode: STRIDED_CONV_MODE,
    out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "unique",
):
    batch_indexed_in_coords = input_sparse_tensor.batch_indexed_coordinates
    in_to_out_stride_ratio = stride

    if input_sparse_tensor.coordinates.dtype not in (torch.int32, torch.int64):
        assert (
            input_sparse_tensor.voxel_size is not None
        ), "Voxel size is required for non-integer coordinates"
        # TODO(cchoy): Implement a voxel size aware coordinate mapping

    # Out coords and offsets generation
    if output_spatially_sparse_tensor is not None:
        assert (
            not generative
        ), "Output spatially sparse tensor is not supported with generative convolution"
        batch_indexed_out_coords = output_spatially_sparse_tensor.batch_indexed_coordinates
        out_offsets = output_spatially_sparse_tensor.offsets
    elif generative and all(s == 1 for s in stride):
        assert not transposed, "Transposed and generative convolution is not supported yet"
        batch_indexed_out_coords, out_offsets = expand_coords(
            batch_indexed_in_coords,
            kernel_size=kernel_size,
            kernel_dilation=kernel_dilation,
            kernel_batch=kernel_search_batch_size,
        )
    elif any(s != 1 for s in stride):
        batch_indexed_out_coords, out_offsets = stride_coords(
            batch_indexed_in_coords,
            stride,
            backend=out_code_backend,
        )
        # if generative, we need to expand the coordinates in addition
        if generative and stride_mode == STRIDED_CONV_MODE.STRIDE_ONLY:
            batch_indexed_out_coords, out_offsets = expand_coords(
                batch_indexed_out_coords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
                kernel_batch=kernel_search_batch_size,
            )
        elif generative and stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE:
            batch_indexed_expanded_coords, expanded_offsets = expand_coords(
                batch_indexed_out_coords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
                kernel_batch=kernel_search_batch_size,
            )
            # rename
            batch_indexed_in_coords = batch_indexed_out_coords
            batch_indexed_out_coords = batch_indexed_expanded_coords
            out_offsets = expanded_offsets
    elif all(s == 1 for s in stride):
        batch_indexed_out_coords, out_offsets = (
            batch_indexed_in_coords,
            input_sparse_tensor.offsets,
        )
    else:
        raise ValueError(
            f"Unsupported case. stride_mode: {stride_mode}, generative: {generative}, transposed: {transposed}"
        )

    # if input_sparse_tensor.cache is not None, check the cache first
    kernel_map_cache_key = IntSearchCacheKey(
        kernel_size=kernel_size,
        kernel_dilation=kernel_dilation,
        transposed=transposed,
        generative=generative,
        stride_mode=str(stride_mode),
        in_offsets=input_sparse_tensor.offsets,
        out_offsets=out_offsets,
    )
    if input_sparse_tensor.cache is not None:
        kernel_map = input_sparse_tensor.cache.get(kernel_map_cache_key)
        if kernel_map is not None:
            return batch_indexed_out_coords, out_offsets, kernel_map

    # Kernel map generation
    if transposed and not generative:
        if input_sparse_tensor.cache is not None:
            # Check if the kernel map for non transposed case exists
            kernel_map_cache_key_non_transposed = IntSearchCacheKey(
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
                transposed=False,
                generative=generative,
                stride_mode=str(stride_mode),
                in_offsets=out_offsets,
                out_offsets=input_sparse_tensor.offsets,
            )
            kernel_map_non_transposed = input_sparse_tensor.cache.get(
                kernel_map_cache_key_non_transposed
            )
            if kernel_map_non_transposed is not None:
                # Swap in and out maps for transposed kernel map generation and swap it back
                kernel_map = IntSearchResult(
                    in_maps=kernel_map_non_transposed.out_maps,
                    out_maps=kernel_map_non_transposed.in_maps,
                    offsets=kernel_map_non_transposed.offsets,
                )
                return batch_indexed_out_coords, out_offsets, kernel_map

        # Swap in and out maps for transposed kernel map generation and swap it back
        kernel_map = generate_kernel_map(
            batch_indexed_out_coords,
            batch_indexed_in_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
        kernel_map = IntSearchResult(
            in_maps=kernel_map.out_maps,
            out_maps=kernel_map.in_maps,
            offsets=kernel_map.offsets,
        )
    elif stride_mode == STRIDED_CONV_MODE.STRIDE_ONLY:
        kernel_map = generate_kernel_map(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
    elif stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and not generative:
        # Compute mapping from output to output since it will be reduced
        kernel_map = generate_kernel_map(
            batch_indexed_out_coords,
            batch_indexed_out_coords,
            ntuple(1, ndim=input_sparse_tensor.num_spatial_dims),
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
    elif stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and generative:
        kernel_map = generate_kernel_map(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            ntuple(1, ndim=input_sparse_tensor.num_spatial_dims),
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
    else:
        raise ValueError(
            f"Unsupported case. stride_mode: {stride_mode}, generative: {generative}, transposed: {transposed}"
        )

    # put the kernel map in the cache

    if input_sparse_tensor.cache is None:
        input_sparse_tensor._extra_attributes["_cache"] = IntSearchCache()

    input_sparse_tensor.cache.put(kernel_map_cache_key, kernel_map)
    return batch_indexed_out_coords, out_offsets, kernel_map
