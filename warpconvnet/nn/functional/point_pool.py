import warnings
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
from warpconvnet.utils.batch_index import batch_index_from_offset

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
    downsample_max_num_points: Optional[int] = None,
    downsample_voxel_size: Optional[float] = None,
    return_type: Literal["point", "sparse"] = "point",
    return_neighbor_search_result: bool = False,
    return_to_unique: bool = False,
) -> BatchedSpatialFeatures:
    """
    Pool points in a point cloud.
    When downsample_max_num_points is provided, the point cloud will be downsampled to the number of points.
    When downsample_voxel_size is provided, the point cloud will be downsampled to the voxel size.
    When both are provided, the point cloud will be downsampled to the voxel size.

    Args:
        pc: PointCollection
        reduction: Reduction type
        downsample_max_num_points: Number of points to downsample to
        downsample_voxel_size: Voxel size to downsample to
        return_type: Return type
        return_neighbor_search_result: Return neighbor search result
        return_to_unique: Return to unique object
    Returns:
        PointCollection or SpatiallySparseTensor
    """
    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)
    # assert at least one of the two is provided
    assert (
        downsample_max_num_points is not None or downsample_voxel_size is not None
    ), "Either downsample_num_points or downsample_voxel_size must be provided."

    if downsample_max_num_points is not None:
        assert (
            not return_to_unique
        ), "return_to_unique must be False when downsample_max_num_points is provided."
        from warpconvnet.geometry.point_collection import PointCollection

        sample_idx, unique_offsets = random_sample(
            batch_offsets=pc.offsets,
            num_samples_per_batch=downsample_max_num_points,
        )
        unique_coords = pc.coordinate_tensor[sample_idx]
        # nearest neighbor of all points to down_coords
        knn_down_indices = batched_knn_search(
            ref_positions=unique_coords,
            ref_offsets=unique_offsets,
            query_positions=pc.coordinate_tensor,
            query_offsets=pc.offsets,
            k=1,
        ).squeeze()
        # argsort and get the number of points per down indices
        sorted_knn_down_indices, to_sorted_knn_down_indices = torch.sort(knn_down_indices)
        unique_indices, counts = torch.unique_consecutive(
            sorted_knn_down_indices, return_counts=True
        )
        knn_offsets = torch.cat(
            [
                torch.zeros(1, dtype=counts.dtype, device=counts.device),
                torch.cumsum(counts, dim=0),
            ],
            dim=0,
        )
        down_features = row_reduction(
            pc.features[to_sorted_knn_down_indices],
            knn_offsets,
            reduction=reduction,
        )
        if len(unique_indices) != len(sample_idx):
            # select survived indices
            unique_coords = unique_coords[unique_indices]
            unique_batch_indices = batch_index_from_offset(unique_offsets, backend="torch")
            unique_batch_indices = unique_batch_indices.to(unique_indices.device)[unique_indices]
            _, unique_counts = torch.unique_consecutive(unique_batch_indices, return_counts=True)
            unique_offsets = torch.cat(
                [
                    torch.zeros(1, dtype=unique_counts.dtype),
                    torch.cumsum(unique_counts.cpu(), dim=0),
                ],
                dim=0,
            )
            if return_neighbor_search_result:
                # update the indices to the unique indices
                warnings.warn(
                    "Neighbor search result requires remapping the indices to the unique indices. "
                    "This may incur additional overhead.",
                    stacklevel=2,
                )
                mapping = torch.zeros(
                    len(sample_idx), dtype=unique_indices.dtype, device=unique_indices.device
                )
                mapping[unique_indices] = torch.arange(
                    len(unique_indices), device=unique_indices.device
                )
                knn_down_indices = mapping[knn_down_indices]

        out_pc = PointCollection(
            batched_coordinates=unique_coords,
            batched_features=down_features,
            offsets=unique_offsets,
            num_points=downsample_max_num_points,
        )
        if return_neighbor_search_result:
            return out_pc, NeighborSearchResult(knn_down_indices, knn_offsets)
        return out_pc

    if reduction == REDUCTIONS.RANDOM:
        assert not return_to_unique, "return_to_unique must be False when reduction is RANDOM."
        assert (
            not return_neighbor_search_result
        ), "return_neighbor_search_result must be False when reduction is RANDOM."
        to_unique_indices, unique_offsets = voxel_downsample_random_indices(
            batched_points=pc.coordinate_tensor,
            offsets=pc.offsets,
            voxel_size=downsample_voxel_size,
        )
        down_features = pc.features[to_unique_indices]
        out_sf = _to_return_type(
            input_pc=pc,
            down_features=down_features,
            downsample_voxel_size=downsample_voxel_size,
            down_indices=to_unique_indices,
            down_offsets=unique_offsets,
            return_type=return_type,
        )
        return out_sf

    # voxel downsample
    (
        unique_coords,
        unique_offsets,
        to_csr_indices,
        to_csr_offsets,
        to_unique,
    ) = voxel_downsample_csr_mapping(
        batched_points=pc.coordinate_tensor,
        offsets=pc.offsets,
        voxel_size=downsample_voxel_size,
    )

    down_features = row_reduction(
        pc.features[to_csr_indices],
        to_csr_offsets,
        reduction=reduction,
    )

    out_sf = _to_return_type(
        input_pc=pc,
        down_features=down_features,
        downsample_voxel_size=downsample_voxel_size,
        down_indices=to_unique.to_unique_indices,
        down_offsets=unique_offsets,
        return_type=return_type,
    )

    if return_to_unique:
        return out_sf, to_unique
    return out_sf
