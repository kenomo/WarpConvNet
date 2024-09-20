from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

from warpconvnet.geometry.base_geometry import BatchedFeatures
from warpconvnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
    kernel_map_from_size,
)
from warpconvnet.geometry.spatially_sparse_tensor import (
    BatchedDiscreteCoordinates,
    SpatiallySparseTensor,
)
from warpconvnet.nn.functional.sparse_coords_ops import (
    expand_coords,
    generate_output_coords,
)
from warpconvnet.nn.functional.sparse_pool import sparse_reduce
from warpconvnet.utils.ntuple import ntuple


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
        batched_features: Float[Tensor, "N C_in"],
        batch_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: DiscreteNeighborSearchResult,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    ) -> Float[Tensor, "M C_out"]:
        """
        Perform a sparse convolution on the input tensor using the specified algorithm.
        """
        pass

    @staticmethod
    def backward(
        ctx, grad_output: Float[Tensor, "M C_out"]
    ) -> Tuple[Float[Tensor, "N C_in"], None, Float[Tensor, "K C_out C_in"], None, None]:
        """
        Perform the backward pass of the sparse convolution.
        """
        pass


class SpatiallySparseConvExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: DiscreteNeighborSearchResult,
        num_out_coords: int,
    ) -> Float[Tensor, "M C_out"]:
        device = in_features.device
        # Create output feature tensor
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=in_features.dtype
        )

        for i, (in_map, out_map) in enumerate(kernel_map):
            # Get the input and output maps
            in_map = in_map.to(device)
            out_map = out_map.to(device)

            # Get the input and output features
            curr_in_features = in_features[in_map]

            # matmul. N x C_in @ C_in x C_out -> N x C_out
            # bmm. B x N x C_in @ B x C_in x C_out -> B x N x C_out
            curr_out_features = torch.matmul(curr_in_features, weight[i])

            # Add to output feature tensor
            output_feature_tensor[out_map] += curr_out_features

        ctx.kernel_map = kernel_map
        ctx.save_for_backward(in_features, weight)

        return output_feature_tensor

    @staticmethod
    def backward(
        ctx, grad_output: Float[Tensor, "M C_out"]
    ) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map

        # Compute the gradient of the input features
        grad_in_features = torch.zeros_like(in_features)
        grad_weight = torch.zeros_like(weight)

        for i, (in_map, out_map) in enumerate(kernel_map):
            # Get the input and output features
            curr_in_features = in_features[in_map]
            curr_out_features = grad_output[out_map]

            # matmul. N x C_out @ C_out x C_in -> N x C_in
            grad_in_features[in_map] += torch.matmul(curr_out_features, weight[i].T)

            # matmul. M x C_out @ C_out x C_in -> M x C_in
            grad_weight[i] += torch.matmul(curr_in_features.T, curr_out_features)

        return grad_in_features, grad_weight, None, None


class SpatiallySparseConvBatchedExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: DiscreteNeighborSearchResult,
        num_out_coords: int,
        matmul_batch_size: int,
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
            curr_in_features = in_features[in_maps.clip(min=0)]

            # bmm. B x (N + 1) x C_in @ B x C_in x C_out -> B x (N + 1) x C_out
            curr_out_features_batch = torch.bmm(curr_in_features, weight[i_start:i_end])

            # Add to output feature tensor. all negative indices will map to the dummy row
            output_feature_tensor[out_maps + 1] += curr_out_features_batch

        ctx.kernel_map = kernel_map
        ctx.matmul_batch_size = matmul_batch_size
        ctx.save_for_backward(in_features, weight)

        return output_feature_tensor[1:].clone()

    @staticmethod
    def backward(
        ctx, grad_output: Float[Tensor, "M C_out"]
    ) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        matmul_batch_size = ctx.matmul_batch_size

        # Add a dummy row to the beginning of the in_maps tensor
        grad_in_features = torch.zeros(
            in_features.shape[0] + 1,
            in_features.shape[1],
            device=in_features.device,
            dtype=in_features.dtype,
        )
        grad_weight = torch.zeros_like(weight)

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

        return grad_in_features[1:].clone(), grad_weight, None, None, None


def spatially_sparse_conv(
    input_sparse_tensor: SpatiallySparseTensor,
    weight: Float[Tensor, "K C_out C_in"],
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    kernel_dilation: Union[int, List[int], Tuple[int, ...]] = 1,
    bias: Optional[Float[Tensor, "C_out"]] = None,  # noqa: F821
    kernel_search_batch_size: Optional[int] = None,
    kernel_matmul_batch_size: int = 2,
    generative: bool = False,
    output_spatially_sparse_tensor: Optional[SpatiallySparseTensor] = None,
    transposed: bool = False,
    conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
    stride_reduce: str = "max",
) -> SpatiallySparseTensor:
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
            batched_features=BatchedFeatures(
                out_feature_tensor,
                offsets=input_sparse_tensor.offsets,
            ),
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
    )

    num_out_coords = batch_indexed_out_coords.shape[0]

    if stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and any(s != 1 for s in stride):
        input_sparse_tensor = sparse_reduce(
            input_sparse_tensor,
            kernel_size=stride,  # reduce by stride
            stride=stride,
            reduce=stride_reduce,
        )

    # Call explicit gemm
    if conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM:
        out_feature_tensor = SpatiallySparseConvExplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            weight,
            kernel_map,
            num_out_coords,
        )
    elif conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM_BATCHED:
        out_feature_tensor = SpatiallySparseConvBatchedExplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            weight,
            kernel_map,
            num_out_coords,
            kernel_matmul_batch_size,
        )
    else:
        raise ValueError(f"Unsupported convolution algorithm: {conv_algo}")

    if bias is not None:
        out_feature_tensor += bias

    out_offsets = out_offsets.cpu().int()
    return input_sparse_tensor.replace(
        batched_coordinates=BatchedDiscreteCoordinates(
            batch_indexed_out_coords[:, 1:],
            offsets=out_offsets,
        ),
        batched_features=BatchedFeatures(
            out_feature_tensor,
            offsets=out_offsets,
        ),
        stride=out_tensor_stride,
    )


def generate_output_coords_and_kernel_map(
    input_sparse_tensor: SpatiallySparseTensor,
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    stride: Tuple[int, ...],
    generative: bool,
    transposed: bool,
    output_spatially_sparse_tensor: Optional[SpatiallySparseTensor],
    kernel_search_batch_size: int,
    stride_mode: STRIDED_CONV_MODE,
):
    batch_indexed_in_coords = input_sparse_tensor.batch_indexed_coordinates
    in_to_out_stride_ratio = stride

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
        batch_indexed_out_coords, out_offsets = generate_output_coords(
            batch_indexed_in_coords, stride
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

    # Kernel map generation
    if transposed and not generative:
        # Swap in and out maps for transposed kernel map generation and swap it back
        kernel_map = kernel_map_from_size(
            batch_indexed_out_coords,
            batch_indexed_in_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
        kernel_map = DiscreteNeighborSearchResult(
            in_maps=kernel_map.out_maps,
            out_maps=kernel_map.in_maps,
            offsets=kernel_map.offsets,
        )
    elif stride_mode == STRIDED_CONV_MODE.STRIDE_ONLY:
        kernel_map = kernel_map_from_size(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
    elif stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and not generative:
        # Compute mapping from output to output since it will be reduced
        kernel_map = kernel_map_from_size(
            batch_indexed_out_coords,
            batch_indexed_out_coords,
            ntuple(1, ndim=input_sparse_tensor.num_spatial_dims),
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
    elif stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and generative:
        kernel_map = kernel_map_from_size(
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

    return batch_indexed_out_coords, out_offsets, kernel_map
