from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import warp as wp
from warp.convnet.core.hashmap import VectorHashTable
from warp.convnet.geometry.base_geometry import BatchedFeatures
from warp.convnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
    kernel_map_from_size,
)
from warp.convnet.geometry.spatially_sparse_tensor import (
    BatchedDiscreteCoordinates,
    SpatiallySparseTensor,
)
from warp.convnet.utils.batch_index import (
    batch_indexed_coordinates,
    offsets_from_batch_index,
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
        batched_features: Float[Tensor, "N C_in"],
        batch_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        weight: Float[Tensor, "K C_out C_in"],
        kernel_map: DiscreteNeighborSearchResult,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
    ) -> Float[Tensor, "M C_out"]:
        pass

    @staticmethod
    def backward(
        ctx, grad_output: Float[Tensor, "M C_out"]
    ) -> Tuple[Float[Tensor, "N C_in"], None, Float[Tensor, "K C_out C_in"], None, None]:
        pass


@torch.no_grad()
def expand_coords(
    batch_indexed_coords: Int[Tensor, "N 4"],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    kernel_batch: int = 8,
) -> Tuple[Int[Tensor, "M 4"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Expand the coordinates by the kernel size
    """
    # coords to batched coordinates
    batch_indexed_coords_wp = wp.from_torch(batch_indexed_coords, dtype=wp.vec4i)
    # Create a vector hashtable for the batched coordinates
    hashtable = VectorHashTable.from_keys(batch_indexed_coords_wp)
    # Initialize the unique coordinates with the batched coordinates
    unique_coords = batch_indexed_coords

    num_total_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    # Create grids for i, j, k
    i, j, k = torch.meshgrid(
        torch.arange(kernel_size[0], dtype=torch.int32),
        torch.arange(kernel_size[1], dtype=torch.int32),
        torch.arange(kernel_size[2], dtype=torch.int32),
        indexing="ij",
    )

    # Flatten the grids and select the batch range
    i, j, k = i.flatten(), j.flatten(), k.flatten()

    # Calculate offsets
    offsets = torch.stack(
        [
            torch.zeros_like(i),
            (i - kernel_size[0] // 2) * kernel_dilation[0],
            (j - kernel_size[1] // 2) * kernel_dilation[1],
            (k - kernel_size[2] // 2) * kernel_dilation[2],
        ],
        dim=1,
    ).to(batch_indexed_coords.device)

    for batch_start in range(0, num_total_kernels, kernel_batch):
        batch_end = min(batch_start + kernel_batch, num_total_kernels)
        # Calculate offsets
        curr_offsets = offsets[batch_start:batch_end]

        # Apply offsets in batch
        new_batched_coords = batch_indexed_coords.unsqueeze(0) + curr_offsets.unsqueeze(1)
        new_batched_coords = new_batched_coords.view(-1, 4)
        new_batched_coords_wp = wp.from_torch(new_batched_coords, dtype=wp.vec4i)

        # Query the hashtable for all new coordinates at once
        indices_wp = hashtable.search(new_batched_coords_wp)
        not_in_hashtable = wp.to_torch(indices_wp) < 0

        has_new_coords = not_in_hashtable.any()

        if has_new_coords:
            # Add unique coordinates
            unique_coords = torch.cat([unique_coords, new_batched_coords[not_in_hashtable]], dim=0)
            # Update hashtable with new unique coordinates
            hashtable = VectorHashTable.from_keys(wp.from_torch(unique_coords, dtype=wp.vec4i))

    # sort the coordinates and return the coordinate and offset
    out_coords = torch.sort(unique_coords, dim=0)[0]
    out_batch_index = out_coords[:, 0]
    out_offsets = offsets_from_batch_index(out_batch_index, backend="torch")
    return out_coords, out_offsets


@torch.no_grad()
def generate_output_coords(
    batch_indexed_coords: Int[Tensor, "N 4"],
    stride: Tuple[int, ...],
) -> Tuple[Int[Tensor, "M 4"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Downsample the coordinates by the stride.
    """
    # convert to wp array
    device = batch_indexed_coords.device
    batched_stride = torch.tensor([1, *ntuple(stride, ndim=3)], dtype=torch.int32, device=device)
    # discretize the coordinates by floor division
    discretized_coords = torch.floor(batch_indexed_coords / batched_stride).int()
    # Get unique coordinates
    unique_coords = torch.unique(discretized_coords, dim=0, sorted=True)

    out_batch_index = unique_coords[:, 0]
    out_offsets = offsets_from_batch_index(out_batch_index, backend="torch")

    return unique_coords, out_offsets


def spatially_sparse_conv(
    input_sparse_tensor: SpatiallySparseTensor,
    weight: Float[Tensor, "K C_out C_in"],
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    bias: Optional[Float[Tensor, "C_out"]] = None,  # noqa: F821
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
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
    kernel_size = ntuple(kernel_size, ndim=3)
    stride = ntuple(stride, ndim=3)
    padding = ntuple(padding, ndim=3)
    dilation = ntuple(dilation, ndim=3)

    if transposed:
        raise NotImplementedError("Transposed convolution is not supported yet")

    # Generate output coordinates
    stride_is_one = all(s == 1 for s in stride)
    batch_indexed_in_coords = batch_indexed_coordinates(
        input_sparse_tensor.coordinate_tensor,
        input_sparse_tensor.offsets,
        backend="torch",
        return_type="torch",
    )

    if output_spatially_sparse_tensor is not None:
        assert (
            not generative
        ), "Output spatially sparse tensor is not supported with generative convolution"
        # assert stride is 1
        assert all(s == 1 for s in stride), "Stride must be 1 for output spatially sparse tensor"
        batch_indexed_out_coords = batch_indexed_coordinates(
            output_spatially_sparse_tensor.coordinate_tensor,
            output_spatially_sparse_tensor.offsets,
            backend="torch",
            return_type="torch",
        )
        in_to_out_stride_ratio = tuple(
            o // s
            for o, s in zip(output_spatially_sparse_tensor.stride, input_sparse_tensor.stride)
        )
    else:
        in_to_out_stride_ratio = stride

    if generative:
        assert stride_is_one, "Generative with stride not equal to 1 is not supported yet"
        assert all(
            k % 2 == 1 for k in kernel_size
        ), "Kernel size must be odd for generative convolution"
        batch_indexed_out_coords, out_offsets = expand_coords(batch_indexed_in_coords, kernel_size)
    elif stride_is_one:
        batch_indexed_out_coords, out_offsets = (
            batch_indexed_in_coords,
            input_sparse_tensor.offsets,
        )
    else:
        batch_indexed_out_coords, out_offsets = generate_output_coords(
            batch_indexed_in_coords, stride
        )

    # Generate kernel map
    in_maps, out_maps, kernel_map_offsets = kernel_map_from_size(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        in_to_out_stride_ratio,
        kernel_size,
        stride,
        padding,
        dilation,
    )
    num_out_coords = batch_indexed_out_coords.shape[0]

    # Call explicit gemm
    if conv_algo == SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM:
        out_feature_tensor = SpatiallySparseConvExplicitGEMMFunction.apply(
            input_sparse_tensor.feature_tensor,
            input_sparse_tensor.offsets,
            weight,
            in_maps,
            out_maps,
            kernel_map_offsets,
            num_out_coords,
        )
    else:
        raise ValueError(f"Unsupported convolution algorithm: {conv_algo}")

    tensor_stride = input_sparse_tensor.stride
    # if tensor stride is none, set to 1
    if tensor_stride is None:
        tensor_stride = ntuple(1, ndim=3)
    out_tensor_stride = tuple(o * s for o, s in zip(in_to_out_stride_ratio, tensor_stride))

    return SpatiallySparseTensor(
        batched_coordinates=BatchedDiscreteCoordinates(
            coordinates=batch_indexed_out_coords,
            offsets=input_sparse_tensor.offsets,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        batched_features=BatchedFeatures(
            features=out_feature_tensor,
            offsets=input_sparse_tensor.offsets,
        ),
        stride=out_tensor_stride,
    )
