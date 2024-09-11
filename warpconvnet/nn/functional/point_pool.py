from enum import Enum
from typing import List, Literal, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.geometry.ops.coord_ops import relative_coords
from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warp.convnet.geometry.ops.voxel_ops import (
    voxel_downsample_csr_mapping,
    voxel_downsample_random_indices,
)
from warp.convnet.nn.encoding import sinusoidal_encode
from warp.convnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS, row_reduction

__all__ = [
    "FeaturePoolingArgs",
    "FEATURE_POOLING_MODE",
    "DEFAULT_FEATURE_POOLING_ARGS",
    "pool_features",
    "point_collection_pool",
]


class FEATURE_POOLING_MODE(Enum):
    REDUCTIONS = "reductions"
    ENCODED_COORDS = "encoded_coords"
    RELATIVE_COORDS = "relative_coords"
    RANDOM_SAMPLE = "random_sample"


class FeaturePoolingArgs:
    pooling_mode: FEATURE_POOLING_MODE
    reductions: List[REDUCTIONS]
    encoded_coords_dim: int
    encoded_coords_data_range: float
    downsample_voxel_size: float
    num_random_samples: int

    def __init__(
        self,
        pooling_mode: FEATURE_POOLING_MODE,
        reductions: Optional[List[REDUCTIONS | REDUCTION_TYPES_STR]] = None,
        encoded_coords_dim: Optional[int] = None,
        encoded_coords_data_range: Optional[float] = None,
        downsample_voxel_size: Optional[float] = None,
        num_random_samples: Optional[int] = None,
    ):
        # Type conversions
        if isinstance(pooling_mode, str):
            pooling_mode = FEATURE_POOLING_MODE(pooling_mode)
        assert isinstance(pooling_mode, FEATURE_POOLING_MODE)
        if reductions is None:
            reductions = [REDUCTIONS.MEAN]
        reductions = [
            REDUCTIONS(reduction) if isinstance(reduction, str) else reduction
            for reduction in reductions
        ]

        self.pooling_mode = pooling_mode
        self.reductions = reductions
        if pooling_mode == FEATURE_POOLING_MODE.ENCODED_COORDS:
            assert encoded_coords_dim is not None and encoded_coords_dim % 2 == 0
            assert encoded_coords_data_range is not None
        self.encoded_coords_dim = encoded_coords_dim
        self.encoded_coords_data_range = encoded_coords_data_range
        self.downsample_voxel_size = downsample_voxel_size
        self.num_random_samples = num_random_samples

    def clone(self, downsample_voxel_size: Optional[float] = None) -> "FeaturePoolingArgs":
        return FeaturePoolingArgs(
            pooling_mode=self.pooling_mode,
            reductions=self.reductions,
            encoded_coords_dim=self.encoded_coords_dim,
            encoded_coords_data_range=self.encoded_coords_data_range,
            downsample_voxel_size=downsample_voxel_size or self.downsample_voxel_size,
            num_random_samples=self.num_random_samples,
        )

    def __repr__(self) -> str:
        if self.pooling_mode == FEATURE_POOLING_MODE.REDUCTIONS:
            out_str = f"{self.pooling_mode.name}({self.downsample_voxel_size})"
        elif self.pooling_mode == FEATURE_POOLING_MODE.ENCODED_COORDS:
            out_str = f"{self.pooling_mode.name}(encoded_coords_dim={self.encoded_coords_dim} encoded_coords_data_range={self.encoded_coords_data_range})"
        elif self.pooling_mode == FEATURE_POOLING_MODE.RANDOM_SAMPLE:
            out_str = f"{self.pooling_mode.name}({self.num_random_samples})"
        return out_str


DEFAULT_FEATURE_POOLING_ARGS = FeaturePoolingArgs(
    pooling_mode=FEATURE_POOLING_MODE.RANDOM_SAMPLE,
)


def pool_features(
    in_feats: Float[Tensor, "N C"],  # noqa: F722,F821
    down_coords: Float[Tensor, "M D"],  # noqa: F722,F821
    neighbors: NeighborSearchResult,
    pooling_args: FeaturePoolingArgs,
    perm: Optional[Int[Tensor, "M"]] = None,  # noqa: F722,F821
) -> Float[Tensor, "M C"]:  # noqa: F722,F821
    """
    Pool features from input coordinates to downsampled coordinates.

    Args:
        in_coords: Input coordinates.
        down_coords: Downsampled coordinates.
        pooling_args: Pooling arguments.
        neighbors: Neighbor search return.

    Returns:
        Pooled features.
    """
    if pooling_args.pooling_mode == FEATURE_POOLING_MODE.REDUCTIONS:
        feats = _pool_reductions(in_feats=in_feats, neighbors=neighbors, pooling_args=pooling_args)
    elif pooling_args.pooling_mode == FEATURE_POOLING_MODE.ENCODED_COORDS:
        feats = _pool_encoded_coords(down_coords, pooling_args)
    elif pooling_args.pooling_mode == FEATURE_POOLING_MODE.RELATIVE_COORDS:
        feats = relative_coords(down_coords, neighbors)
    elif pooling_args.pooling_mode == FEATURE_POOLING_MODE.RANDOM_SAMPLE:
        assert perm is not None
        return in_feats[perm]
    else:
        raise ValueError(f"Invalid pooling mode: {pooling_args.pooling_mode}")
    return feats


def _pool_reductions(
    in_feats: Float[Tensor, "N C"],  # noqa: F722,F821
    neighbors: NeighborSearchResult,
    pooling_args: FeaturePoolingArgs,
) -> Float[Tensor, "M C"]:  # noqa: F722,F821
    """
    Pool features using reductions.

    Args:
        in_coords: Input coordinates.
        down_coords: Downsampled coordinates.
        pooling_args: Pooling arguments.
        neighbors: Neighbor search return.

    Returns:
        Pooled features.
    """
    pooled_features = []
    for reduction in pooling_args.reductions:
        pooled_feature = row_reduction(
            in_feats,
            neighbors.neighbors_row_splits,
            reduction,
        )
        pooled_features.append(pooled_feature)
    return torch.cat(pooled_features, dim=-1)


def _pool_encoded_coords(
    coordinates: Float[Tensor, "M D"],  # noqa: F722,F821
    pooling_args: FeaturePoolingArgs,
) -> Float[Tensor, "M C"]:  # noqa: F722,F821
    """
    Pool features using encoded coordinates.

    Args:
        down_coords: Downsampled coordinates.
        pooling_args: Pooling arguments.

    Returns:
        Pooled features.
    """
    freqs = (
        torch.arange(start=0, end=pooling_args.encoded_coords_dim // 2, dtype=float)
        / pooling_args.encoded_coords_data_range
    )
    encoded_coords = sinusoidal_encode(coordinates, freqs)
    return encoded_coords


def point_collection_pool(
    pc: "PointCollection",  # noqa: F821
    pooling_args: FeaturePoolingArgs,
    return_type: Literal["point_collection", "sparse_tensor"] = "point_collection",
) -> Tuple["PointCollection", NeighborSearchResult]:  # noqa: F821
    from warp.convnet.geometry.spatially_sparse_tensor import (
        BatchedDiscreteCoordinates,
        BatchedFeatures,
        SpatiallySparseTensor,
    )

    voxel_size = pooling_args.downsample_voxel_size

    if pooling_args.pooling_mode == FEATURE_POOLING_MODE.RANDOM_SAMPLE:
        perm, down_offsets = voxel_downsample_random_indices(
            batched_points=pc.coordinate_tensor,
            offsets=pc.offsets,
            voxel_size=voxel_size,
        )
        if return_type == "point_collection":
            return (
                pc.replace(
                    batched_coordinates=pc.batched_coordinates.__class__(
                        batched_tensor=pc.coordinate_tensor[perm], offsets=down_offsets
                    ),
                    batched_features=pc.batched_features.__class__(
                        batched_tensor=pc.feature_tensor[perm], offsets=down_offsets
                    ),
                    voxel_size=voxel_size,
                ),
                None,
            )
        else:
            discrete_coords = torch.floor(pc.coordinate_tensor[perm] / voxel_size).int()
            return (
                SpatiallySparseTensor(
                    batched_coordinates=BatchedDiscreteCoordinates(
                        batched_tensor=discrete_coords, offsets=down_offsets
                    ),
                    batched_features=BatchedFeatures(
                        batched_tensor=pc.feature_tensor[perm], offsets=down_offsets
                    ),
                    voxel_size=voxel_size,
                ),
                None,
            )

    perm, down_offsets, vox_inices, vox_offsets = voxel_downsample_csr_mapping(
        batched_points=pc.coordinate_tensor,
        offsets=pc.offsets,
        voxel_size=voxel_size,
    )

    neighbors = NeighborSearchResult(vox_inices, vox_offsets)
    down_coords = pc.coordinate_tensor[perm]
    down_features = pool_features(
        in_feats=pc.feature_tensor,
        down_coords=down_coords,
        neighbors=neighbors,
        pooling_args=pooling_args,
    )

    if return_type == "point_collection":
        return (
            pc.replace(
                batched_coordinates=pc.batched_coordinates.__class__(
                    batched_tensor=down_coords, offsets=down_offsets
                ),
                batched_features=pc.batched_features.__class__(
                    batched_tensor=down_features, offsets=down_offsets
                ),
                voxel_size=voxel_size,
            ),
            neighbors,
        )
    else:
        discrete_coords = torch.floor(pc.coordinate_tensor[perm] / voxel_size).int()
        return (
            SpatiallySparseTensor(
                batched_coordinates=BatchedDiscreteCoordinates(
                    batched_tensor=discrete_coords, offsets=down_offsets
                ),
                batched_features=BatchedFeatures(
                    batched_tensor=down_features, offsets=down_offsets
                ),
                offsets=down_offsets,
                voxel_size=voxel_size,
            ),
            neighbors,
        )
