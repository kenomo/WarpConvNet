from typing import Literal, Tuple

import torch
from jaxtyping import Int
from torch import Tensor
from torch_scatter import segment_csr

from warp.convnet.geometry.base_geometry import BatchedFeatures
from warp.convnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
    kernel_map_from_size,
)
from warp.convnet.geometry.spatially_sparse_tensor import (
    BatchedDiscreteCoordinates,
    SpatiallySparseTensor,
)
from warp.convnet.utils.batch_index import offsets_from_batch_index
from warp.convnet.utils.ntuple import ntuple


@torch.no_grad()
def generate_output_coords(
    batch_indexed_coords: Int[Tensor, "N 4"],
    stride: Tuple[int, ...],
) -> Tuple[Int[Tensor, "M 4"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Downsample the coordinates by the stride.
    """
    assert len(stride) == 3, "Stride must be a tuple of 3 integers"
    assert batch_indexed_coords.shape[1] == 4, "Batch indexed coordinates must have 4 columns"
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


def sparse_downsample_reduce(
    spatially_sparse_tensor: SpatiallySparseTensor,
    stride: int | Tuple[int, ...],
    reduce: Literal["mean", "sum", "max", "min"] = "mean",
    kernel_search_batch_size: int = 8,
) -> SpatiallySparseTensor:
    """
    Downsample the spatially sparse tensor by random indices.
    """
    stride = ntuple(stride, ndim=3)
    batch_indexed_in_coords = spatially_sparse_tensor.batch_indexed_coordinates
    batch_indexed_out_coords, output_offsets = generate_output_coords(
        batch_indexed_in_coords, stride
    )
    # Find mapping from in to out
    kernel_map: DiscreteNeighborSearchResult = kernel_map_from_size(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        in_to_out_stride_ratio=stride,
        kernel_size=stride,
        kernel_dilation=ntuple(1, ndim=3),
        kernel_search_batch_size=kernel_search_batch_size,
        kernel_center_offset=ntuple(0, ndim=3),
    )
    in_maps, unique_out_maps, offsets = kernel_map.to_csr()
    in_features = spatially_sparse_tensor.feature_tensor

    out_features = segment_csr(
        in_features[in_maps],
        indptr=offsets.to(in_features.device),
        reduce=reduce,
    )

    if len(unique_out_maps) != batch_indexed_out_coords.shape[0]:
        raise ValueError(
            "Some output coordinates don't have any input maps.",
        )

        # cchoy: This is a rare case where some output coordinates don't have any input maps.
        # We need to zero out the features for those coordinates.
        new_out_features = torch.zeros(
            batch_indexed_out_coords.shape[0],
            in_features.shape[1],
            device=spatially_sparse_tensor.device,
        )
        new_out_features[unique_out_maps] = out_features
        out_features = new_out_features

    output_offsets = output_offsets.cpu()
    return spatially_sparse_tensor.replace(
        batched_coordinates=BatchedDiscreteCoordinates(
            batch_indexed_out_coords[:, 1:],
            output_offsets,
        ),
        batched_features=BatchedFeatures(out_features, output_offsets),
    )


def sparse_downsample_first(
    spatially_sparse_tensor: SpatiallySparseTensor,
    stride: int | Tuple[int, ...],
    kernel_search_batch_size: int = 8,
) -> SpatiallySparseTensor:
    """
    Downsample the spatially sparse tensor by random indices.
    """
    stride = ntuple(stride, ndim=3)
    batch_indexed_in_coords = spatially_sparse_tensor.batch_indexed_coordinates
    batch_indexed_out_coords, output_offsets = generate_output_coords(
        batch_indexed_in_coords, stride
    )
    # Find mapping from in to out
    kernel_map: DiscreteNeighborSearchResult = kernel_map_from_size(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        in_to_out_stride_ratio=stride,
        kernel_size=stride,
        kernel_dilation=ntuple(1, ndim=3),
        kernel_search_batch_size=kernel_search_batch_size,
        kernel_center_offset=ntuple(0, ndim=3),
    )
    in_maps, unique_out_maps, offsets = kernel_map.to_csr()
    # Get the first features defined by offsets
    first_in_maps = in_maps[offsets[:-1]]
    out_features = spatially_sparse_tensor.feature_tensor[first_in_maps]
    if len(unique_out_maps) != batch_indexed_out_coords.shape[0]:
        raise ValueError(
            "Some output coordinates don't have any input maps.",
        )

    output_offsets = output_offsets.cpu()
    return spatially_sparse_tensor.replace(
        batched_coordinates=BatchedDiscreteCoordinates(
            batch_indexed_out_coords[:, 1:],
            output_offsets,
        ),
        batched_features=BatchedFeatures(out_features, output_offsets),
    )
