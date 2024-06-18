from enum import Enum
from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.geometry.ops.neighbor_search import NeighborSearchReturn
from warp.convnet.nn.encoding import sinusoidal_encode
from warp.convnet.ops.reductions import REDUCTION_TYPES, REDUCTIONS, row_reduction


class FEATURE_POOLING_MODE(Enum):
    REDUCTIONS = "reductions"
    ENCODED_COORDS = "encoded_coords"
    RANDOM_SAMPLE = "random_sample"


class FeaturePoolingArgs:
    pooling_mode: FEATURE_POOLING_MODE
    reductions: List[REDUCTION_TYPES]
    encoded_coords_dim: int
    encoded_coords_data_range: float

    def __init__(
        self,
        pooling_mode: FEATURE_POOLING_MODE,
        reductions: Optional[List[REDUCTION_TYPES]] = None,
        encoded_coords_dim: Optional[int] = None,
        encoded_coords_data_range: Optional[float] = None,
    ):
        assert isinstance(pooling_mode, FEATURE_POOLING_MODE)
        self.pooling_mode = pooling_mode
        if pooling_mode == FEATURE_POOLING_MODE.REDUCTIONS:
            assert [reduction in REDUCTIONS for reduction in reductions]
        self.reductions = reductions
        if pooling_mode == FEATURE_POOLING_MODE.ENCODED_COORDS:
            assert encoded_coords_dim is not None and encoded_coords_dim % 2 == 0
            assert encoded_coords_data_range is not None
        self.encoded_coords_dim = encoded_coords_dim
        self.encoded_coords_data_range = encoded_coords_data_range


def pool_features(
    in_feats: Float[Tensor, "N C"],  # noqa: F722,F821
    down_coords: Float[Tensor, "M D"],  # noqa: F722,F821
    neighbors: NeighborSearchReturn,
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
    elif pooling_args.pooling_mode == FEATURE_POOLING_MODE.RANDOM_SAMPLE:
        assert perm is not None
        return in_feats[perm]
    else:
        raise ValueError(f"Invalid pooling mode: {pooling_args.pooling_mode}")
    return feats


def _pool_reductions(
    in_feats: Float[Tensor, "N C"],  # noqa: F722,F821
    neighbors: NeighborSearchReturn,
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
    down_coords: Float[Tensor, "M D"],  # noqa: F722,F821
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
    encoded_coords = sinusoidal_encode(down_coords, freqs)
    return encoded_coords
