import warnings
from typing import Literal, Optional, Tuple, Union

import torch

from warp.convnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
)
from warp.convnet.geometry.spatially_sparse_tensor import (
    BatchedDiscreteCoordinates,
    BatchedFeatures,
    SpatiallySparseTensor,
)
from warp.convnet.nn.functional.sparse_ops import (
    generate_output_coords,
    kernel_map_from_size,
    segment_csr,
)
from warp.convnet.utils.ntuple import ntuple


def sparse_reduce(
    spatially_sparse_tensor: SpatiallySparseTensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
    reduce: Literal["max", "min", "mean", "sum"] = "max",
    kernel_search_batch_size: int = 8,
) -> SpatiallySparseTensor:
    """
    Max pooling for spatially sparse tensors.
    """
    if stride is None:
        stride = kernel_size
    ndim = spatially_sparse_tensor.num_spatial_dims
    stride = ntuple(stride, ndim=ndim)
    kernel_size = ntuple(kernel_size, ndim=ndim)
    batch_indexed_in_coords = spatially_sparse_tensor.batch_indexed_coordinates
    batch_indexed_out_coords, output_offsets = generate_output_coords(
        batch_indexed_in_coords, stride
    )
    # Find mapping from in to out
    kernel_map: DiscreteNeighborSearchResult = kernel_map_from_size(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        in_to_out_stride_ratio=stride,
        kernel_size=kernel_size,
        kernel_dilation=ntuple(1, ndim=ndim),
        kernel_search_batch_size=kernel_search_batch_size,
        kernel_center_offset=ntuple(0, ndim=ndim),
    )
    in_maps, unique_out_maps, offsets = kernel_map.to_csr()
    in_features = spatially_sparse_tensor.feature_tensor

    out_features = segment_csr(
        in_features[in_maps],
        indptr=offsets.to(in_features.device),
        reduce=reduce,
    )

    if len(unique_out_maps) != batch_indexed_out_coords.shape[0]:
        warnings.warn(
            f"Some output coordinates don't have any input maps. {batch_indexed_out_coords.shape[0] - len(unique_out_maps)} output coordinates are missing.",
            stacklevel=2,
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


def sparse_max_pool(
    spatially_sparse_tensor: SpatiallySparseTensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
) -> SpatiallySparseTensor:
    """
    Max pooling for spatially sparse tensors.
    """
    return sparse_reduce(spatially_sparse_tensor, kernel_size, stride, "max")


def sparse_avg_pool(
    spatially_sparse_tensor: SpatiallySparseTensor,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
) -> SpatiallySparseTensor:
    """
    Average pooling for spatially sparse tensors.
    """
    return sparse_reduce(spatially_sparse_tensor, kernel_size, stride, "mean")
