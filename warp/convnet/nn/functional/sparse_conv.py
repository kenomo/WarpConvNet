from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

from warp.convnet.geometry.base_geometry import BatchedFeatures
from warp.convnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
    kernel_map_from_size,
)
from warp.convnet.geometry.spatially_sparse_tensor import (
    BatchedDiscreteCoordinates,
    SpatiallySparseTensor,
)
from warp.convnet.nn.functional.sparse_coords_ops import (
    expand_coords,
    generate_output_coords,
)
from warp.convnet.utils.ntuple import ntuple


class SPATIALLY_SPARSE_CONV_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
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
        weight: Float[Tensor, "K C_out C_in"],
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
) -> SpatiallySparseTensor:
    """
    Perform spatially sparse convolution on the input tensor using the specified algorithm.
    Spatially sparse and feature sparse is not supported yet.

    If generative, the output coordinates will be expanded by (kernel size // 2) all directions.

    For transposed convolution, the output coordinates should be provided along with the
    output coordinate stride.
    """
    num_spatial_dims = input_sparse_tensor.num_spatial_dims
    kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
    kernel_dilation = ntuple(kernel_dilation, ndim=num_spatial_dims)
    stride = ntuple(stride, ndim=num_spatial_dims)

    num_total_kernels = np.prod(kernel_size)
    if kernel_search_batch_size is None:
        kernel_search_batch_size = max(num_total_kernels // kernel_size[0], 8)

    if transposed and not generative:
        assert (
            output_spatially_sparse_tensor is not None
        ), "Output spatially sparse tensor is required for transposed convolution"

    # Generate output coordinates
    batch_indexed_in_coords = input_sparse_tensor.batch_indexed_coordinates

    in_to_out_stride_ratio = stride

    if output_spatially_sparse_tensor is not None:
        assert (
            not generative
        ), "Output spatially sparse tensor is not supported with generative convolution"
        batch_indexed_out_coords = output_spatially_sparse_tensor.batch_indexed_coordinates
        out_offsets = output_spatially_sparse_tensor.offsets
    elif generative:
        assert all(
            k % 2 == 1 for k in kernel_size
        ), "Kernel size must be odd for generative convolution"
        # Assert stride is 1
        assert all(s == 1 for s in stride), "Stride must be 1 for generative convolution"
        if not transposed:
            batch_indexed_out_coords, out_offsets = expand_coords(
                batch_indexed_in_coords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
                kernel_batch=kernel_search_batch_size,
            )
        else:
            # Transposed and generative
            raise NotImplementedError("Transposed and generative convolution is not supported yet")

    elif all(s == 1 for s in stride):
        batch_indexed_out_coords, out_offsets = (
            batch_indexed_in_coords,
            input_sparse_tensor.offsets,
        )
    else:
        batch_indexed_out_coords, out_offsets = generate_output_coords(
            batch_indexed_in_coords, stride
        )

    # if transposed, switch input and output for kernel map call
    if transposed:
        kernel_map = kernel_map_from_size(
            batch_indexed_out_coords,
            batch_indexed_in_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
        # switch in_map and out_map
        kernel_map = DiscreteNeighborSearchResult(
            in_maps=kernel_map.out_maps,
            out_maps=kernel_map.in_maps,
            offsets=kernel_map.offsets,
        )
    else:
        # Generate kernel map
        kernel_map = kernel_map_from_size(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_search_batch_size,
        )
    num_out_coords = batch_indexed_out_coords.shape[0]

    # Call explicit gemm
    if conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM:
        out_feature_tensor = SpatiallySparseConvExplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            weight,
            kernel_map,
            num_out_coords,
        )
    else:
        raise ValueError(f"Unsupported convolution algorithm: {conv_algo}")

    tensor_stride = input_sparse_tensor.stride
    # if tensor stride is none, set to 1
    if tensor_stride is None:
        tensor_stride = ntuple(1, ndim=num_spatial_dims)
    out_tensor_stride = tuple(o * s for o, s in zip(in_to_out_stride_ratio, tensor_stride))

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
