from enum import Enum
from typing import List, Literal, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base_geometry import BatchedSpatialFeatures
from warpconvnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
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
    downsample_voxel_size: Optional[float] = None,
    return_type: Literal["point", "sparse"] = "point",
    return_neighbor_search_result: bool = False,
) -> BatchedSpatialFeatures:
    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)

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
