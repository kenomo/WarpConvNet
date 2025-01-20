import warnings
from typing import List, Literal, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
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
from warpconvnet.utils.batch_index import batch_index_from_offset, offsets_from_offsets
from warpconvnet.utils.unique import ToUnique

__all__ = ["point_pool"]


def _generate_batched_coords(
    coordinate_tensor: Float[Tensor, "N 3"],
    return_type: Literal["point", "sparse"],
    csr_indices: Int[Tensor, "M"],  # noqa: F821
    csr_offsets: Int[Tensor, "B+1"],  # noqa: F821
    to_unique_indices: Int[Tensor, "M"],  # noqa: F821
    unique_offsets: Int[Tensor, "B+1"],  # noqa: F821
    downsample_voxel_size: float,
    avereage_pooled_coordinates: bool = False,
) -> Union["BatchedContinuousCoordinates", "BatchedDiscreteCoordinates"]:  # noqa: F821
    from warpconvnet.geometry.point_collection import BatchedContinuousCoordinates
    from warpconvnet.geometry.spatially_sparse_tensor import BatchedDiscreteCoordinates

    if return_type == "point" and not avereage_pooled_coordinates:
        return BatchedContinuousCoordinates(
            batched_tensor=coordinate_tensor[to_unique_indices],
            offsets=unique_offsets,
        )
    elif return_type == "point" and avereage_pooled_coordinates:
        avg_coords = row_reduction(
            coordinate_tensor[csr_indices],
            csr_offsets,
            reduction=REDUCTIONS.MEAN,
        )
        return BatchedContinuousCoordinates(
            batched_tensor=avg_coords,
            offsets=unique_offsets,
        )
    else:
        return BatchedDiscreteCoordinates(
            batched_tensor=torch.floor(
                coordinate_tensor[to_unique_indices] / downsample_voxel_size
            ).int(),
            offsets=unique_offsets,
        )


def _pool_by_random_sample(
    pc: "PointCollection",  # noqa: F821
    downsample_voxel_size: Optional[float] = None,
    return_type: Literal["point", "sparse"] = "point",
) -> Geometry:
    from warpconvnet.geometry.point_collection import PointCollection
    from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor

    if return_type == "sparse":
        RETURN_CLS = SpatiallySparseTensor
    else:
        RETURN_CLS = PointCollection

    to_unique_indices, unique_offsets = voxel_downsample_random_indices(
        batched_points=pc.coordinate_tensor,
        offsets=pc.offsets,
        voxel_size=downsample_voxel_size,
    )
    down_features = pc.features[to_unique_indices]
    down_coords = _generate_batched_coords(
        pc.coordinate_tensor,
        return_type,
        None,
        None,
        to_unique_indices,
        unique_offsets,
        downsample_voxel_size,
        avereage_pooled_coordinates=False,
    )
    out_sf = RETURN_CLS(
        batched_coordinates=down_coords,
        batched_features=down_features,
        voxel_size=downsample_voxel_size,
    )
    return out_sf


def _pool_by_max_num_points(
    pc: "PointCollection",  # noqa: F821
    reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR],
    downsample_max_num_points: Optional[int] = None,
    return_type: Literal["point", "sparse"] = "point",
    return_neighbor_search_result: bool = False,
) -> Geometry:
    from warpconvnet.geometry.point_collection import PointCollection
    from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor

    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)
    if return_type == "sparse":
        RETURN_CLS = SpatiallySparseTensor
    else:
        RETURN_CLS = PointCollection

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
    unique_indices, counts = torch.unique_consecutive(sorted_knn_down_indices, return_counts=True)
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
                len(sample_idx),
                dtype=unique_indices.dtype,
                device=unique_indices.device,
            )
            mapping[unique_indices] = torch.arange(
                len(unique_indices), device=unique_indices.device
            )
            knn_down_indices = mapping[knn_down_indices]

    out_pc = RETURN_CLS(
        batched_coordinates=unique_coords,
        batched_features=down_features,
        offsets=unique_offsets,
        num_points=downsample_max_num_points,
    )
    if return_neighbor_search_result:
        return out_pc, NeighborSearchResult(knn_down_indices, knn_offsets)
    return out_pc


def point_pool(
    pc: "PointCollection",  # noqa: F821
    reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR],
    downsample_max_num_points: Optional[int] = None,
    downsample_voxel_size: Optional[float] = None,
    return_type: Literal["point", "sparse"] = "point",
    avereage_pooled_coordinates: bool = False,
    return_neighbor_search_result: bool = False,
    return_to_unique: bool = False,
    unique_method: Literal["torch", "ravel", "morton"] = "torch",
) -> Geometry:
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
    from warpconvnet.geometry.point_collection import PointCollection
    from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor

    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)
    # assert at least one of the two is provided
    assert (
        downsample_max_num_points is not None or downsample_voxel_size is not None
    ), "Either downsample_num_points or downsample_voxel_size must be provided."
    if return_type == "sparse":
        assert (
            not avereage_pooled_coordinates
        ), "averaging pooled coordinates is not supported for sparse return type"
        RETURN_CLS = SpatiallySparseTensor
    else:
        RETURN_CLS = PointCollection

    if downsample_max_num_points is not None:
        assert (
            not return_to_unique
        ), "return_to_unique must be False when downsample_max_num_points is provided."
        return _pool_by_max_num_points(
            pc,
            reduction,
            downsample_max_num_points,
            return_type,
            return_neighbor_search_result,
        )

    if reduction == REDUCTIONS.RANDOM:
        assert not return_to_unique, "return_to_unique must be False when reduction is RANDOM."
        assert (
            not return_neighbor_search_result
        ), "return_neighbor_search_result must be False when reduction is RANDOM."
        return _pool_by_random_sample(
            pc,
            downsample_voxel_size,
            return_type,
        )

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
        unique_method=unique_method,
    )

    down_features = row_reduction(
        pc.feature_tensor[to_csr_indices],
        to_csr_offsets,
        reduction=reduction,
    )

    batched_coords = _generate_batched_coords(
        pc.coordinate_tensor,
        return_type,
        to_csr_indices,
        to_csr_offsets,
        to_unique.to_unique_indices,
        unique_offsets,
        downsample_voxel_size,
        avereage_pooled_coordinates,
    )

    out_sf = RETURN_CLS(
        batched_coordinates=batched_coords,
        batched_features=down_features,
        voxel_size=downsample_voxel_size,
    )

    if return_to_unique:
        return out_sf, to_unique
    return out_sf


def point_pool_by_code(
    pc: "PointCollection",  # noqa: F821
    code: Int[Tensor, "N"],  # noqa: F821
    reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR],
    average_pooled_coordinates: bool = False,
    return_to_unique: bool = False,
) -> "PointCollection":  # noqa: F821
    from warpconvnet.geometry.point_collection import PointCollection

    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)

    # get the unique indices
    to_unique = ToUnique(return_to_unique_indices=return_to_unique, unique_method="torch")
    unique_code = to_unique.to_unique(code)

    # get the coordinates
    if average_pooled_coordinates:
        coords = row_reduction(
            pc.coordinate_tensor[to_unique.to_csr_indices],
            to_unique.to_csr_offsets,
            reduction=REDUCTIONS.MEAN,
        )
    else:
        coords = pc.coordinate_tensor[to_unique.to_unique_indices]

    # get the features
    features = row_reduction(
        pc.feature_tensor[to_unique.to_csr_indices],
        to_unique.to_csr_offsets,
        reduction=reduction,
    )

    # get the offsets
    offsets = offsets_from_offsets(
        pc.offsets,
        to_unique.to_unique_indices,
    )

    out_pc = pc.replace(
        coordinate_tensor=coords,
        feature_tensor=features,
        offsets=offsets,
        code=unique_code,
    )
    if return_to_unique:
        return out_pc, to_unique
    return out_pc
