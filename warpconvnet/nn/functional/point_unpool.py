from enum import Enum
from typing import Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult


class FEATURE_UNPOOLING_MODE(Enum):
    REPEAT = "repeat"


def _unpool_features(
    pooled_features: Float[Tensor, "N C"],
    pooling_neighbor_search_result: NeighborSearchResult,
    unpooling_mode: Optional[Union[str, FEATURE_UNPOOLING_MODE]] = FEATURE_UNPOOLING_MODE.REPEAT,
) -> Float[Tensor, "M C"]:
    if isinstance(unpooling_mode, str):
        unpooling_mode = FEATURE_UNPOOLING_MODE(unpooling_mode)

    if unpooling_mode == FEATURE_UNPOOLING_MODE.REPEAT:
        # pooling_neighbor_search_result.neighbors_index is the index of the neighbors in the pooled_features
        # pooling_neighbor_search_result.neighbors_row_splits is the row splits of the neighbors in the pooled_features
        rs = pooling_neighbor_search_result.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        # Repeat the features for each neighbor
        return torch.repeat_interleave(pooled_features, num_reps, dim=0)

    raise NotImplementedError(f"Unpooling mode {unpooling_mode} not implemented")


def point_unpool(
    pooled_pc: "PointCollection",  # noqa: F821
    unpooled_pc: "PointCollection",  # noqa: F821
    concat_unpooled_pc: bool,
    unpooling_mode: Optional[Union[str, FEATURE_UNPOOLING_MODE]] = FEATURE_UNPOOLING_MODE.REPEAT,
    pooling_neighbor_search_result: Optional[NeighborSearchResult] = None,
) -> "PointCollection":  # noqa: F821
    if pooling_neighbor_search_result is None:
        # TODO(cchoy): Nearest neighbor search from unpooled_pc to pooled_pc
        raise NotImplementedError("Neighbor search result is required for unpooling")

    unpooled_features = _unpool_features(
        pooled_features=pooled_pc.batched_features.batched_tensor,
        pooling_neighbor_search_result=pooling_neighbor_search_result,
        unpooling_mode=unpooling_mode,
    )
    if concat_unpooled_pc:
        unpooled_features = torch.cat(
            (unpooled_features, unpooled_pc.batched_features.batched_tensor), dim=-1
        )
    return unpooled_pc.__class__(
        batched_coordinates=unpooled_pc.batched_coordinates,
        batched_features=unpooled_pc.batched_features.__class__(
            batched_tensor=unpooled_features,
            offsets=unpooled_pc.offsets,
        ),
    )
