from enum import Enum
from typing import List, Literal, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base_geometry import BatchedSpatialFeatures
from warpconvnet.geometry.ops.neighbor_search_continuous import (
    NeighborSearchResult,
    batched_knn_search,
)
from warpconvnet.geometry.ops.random_sample import random_sample
from warpconvnet.geometry.ops.voxel_ops import (
    voxel_downsample_csr_mapping,
    voxel_downsample_random_indices,
)
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS, row_reduction

__all__ = ["point_pool"]


def _to_return_type(
    input_pc: "PointCollection",  # noqa: F821
    down_features: Float[Tensor, "M C"],
    downsample_voxel_size: float,
    down_indices: Int[Tensor, "M"],  # noqa: F821
    down_offsets: Int[Tensor, "B+1"],  # noqa: F821
    return_type: Literal["point", "sparse"] = "point",
) -> BatchedSpatialFeatures:
    if return_type == "point":
        from warpconvnet.geometry.point_collection import (
            BatchedContinuousCoordinates,
            BatchedFeatures,
            PointCollection,
        )

        return PointCollection(
            batched_coordinates=BatchedContinuousCoordinates(
                batched_tensor=input_pc.coordinate_tensor[down_indices],
                offsets=down_offsets,
            ),
            batched_features=BatchedFeatures(batched_tensor=down_features, offsets=down_offsets),
            voxel_size=downsample_voxel_size,
        )
    else:
        from warpconvnet.geometry.spatially_sparse_tensor import (
            BatchedDiscreteCoordinates,
            BatchedFeatures,
            SpatiallySparseTensor,
        )

        discrete_coords = torch.floor(
            input_pc.coordinate_tensor[down_indices] / downsample_voxel_size
        ).int()
        return SpatiallySparseTensor(
            batched_coordinates=BatchedDiscreteCoordinates(
                batched_tensor=discrete_coords, offsets=down_offsets
            ),
            batched_features=BatchedFeatures(batched_tensor=down_features, offsets=down_offsets),
            voxel_size=downsample_voxel_size,
        )


def point_pool(
    pc: "PointCollection",  # noqa: F821
    reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR],
    downsample_num_points: Optional[int] = None,
    downsample_voxel_size: Optional[float] = None,
    return_type: Literal["point", "sparse"] = "point",
    return_neighbor_search_result: bool = False,
) -> BatchedSpatialFeatures:
    """
    Pool points in a point cloud.
    When downsample_num_points is provided, the point cloud will be downsampled to the number of points.
    When downsample_voxel_size is provided, the point cloud will be downsampled to the voxel size.
    When both are provided, the point cloud will be downsampled to the voxel size.

    Args:
        pc: PointCollection
        reduction: Reduction type
        downsample_num_points: Number of points to downsample to
        downsample_voxel_size: Voxel size to downsample to
        return_type: Return type
        return_neighbor_search_result: Return neighbor search result

    Returns:
        BatchedSpatialFeatures
    """
    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)
    # assert at least one of the two is provided
    assert (
        downsample_num_points is not None or downsample_voxel_size is not None
    ), "Either downsample_num_points or downsample_voxel_size must be provided."

    if downsample_num_points is not None:
        from warpconvnet.geometry.point_collection import PointCollection

        sample_idx, down_offsets = random_sample(
            batch_offsets=pc.offsets,
            num_samples_per_batch=downsample_num_points,
        )
        down_coords = pc.coordinate_tensor[sample_idx]
        # nearest neighbor of all points to down_coords
        down_indices = batched_knn_search(
            ref_positions=down_coords,
            ref_offsets=down_offsets,
            query_positions=pc.coordinate_tensor,
            query_offsets=pc.offsets,
            k=1,
        )
        # argsort and get the number of points per down indices
        sorted_down_indices, to_sort_indices = torch.sort(down_indices.view(-1))
        unique_indices, counts = torch.unique_consecutive(sorted_down_indices, return_counts=True)
        knn_offsets = torch.cat(
            [
                torch.zeros(1, dtype=counts.dtype, device=counts.device),
                torch.cumsum(counts, dim=0),
            ],
            dim=0,
        )
        down_features = row_reduction(
            pc.features[to_sort_indices],
            knn_offsets,
            reduction=reduction,
        )
        if len(unique_indices) != len(sample_idx):
            new_down_features = torch.zeros(
                len(sample_idx), pc.features.shape[1], device=pc.features.device
            )
            new_down_features[unique_indices] = down_features
            down_features = new_down_features
        out_pc = PointCollection(
            batched_coordinates=down_coords,
            batched_features=down_features,
            offsets=down_offsets,
            num_points=downsample_num_points,
        )
        if return_neighbor_search_result:
            return out_pc, NeighborSearchResult(down_indices, down_offsets)
        return out_pc

    if reduction == REDUCTIONS.RANDOM:
        perm, down_offsets = voxel_downsample_random_indices(
            batched_points=pc.coordinate_tensor,
            offsets=pc.offsets,
            voxel_size=downsample_voxel_size,
        )
        down_features = pc.features[perm]
        out_sf = _to_return_type(
            input_pc=pc,
            down_features=down_features,
            downsample_voxel_size=downsample_voxel_size,
            down_indices=perm,
            down_offsets=down_offsets,
            return_type=return_type,
        )
        return out_sf

    # voxel downsample
    perm, down_offsets, vox_inices, vox_offsets = voxel_downsample_csr_mapping(
        batched_points=pc.coordinate_tensor,
        offsets=pc.offsets,
        voxel_size=downsample_voxel_size,
    )

    neighbors = NeighborSearchResult(vox_inices, vox_offsets)
    down_features = row_reduction(
        pc.features,
        neighbors.neighbors_row_splits,
        reduction=reduction,
    )

    out_sf = _to_return_type(
        input_pc=pc,
        down_features=down_features,
        downsample_voxel_size=downsample_voxel_size,
        down_indices=perm,
        down_offsets=down_offsets,
        return_type=return_type,
    )

    if return_neighbor_search_result:
        return out_sf, neighbors
    return out_sf
