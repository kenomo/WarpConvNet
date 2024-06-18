from typing import List, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING
from warp.convnet.geometry.point_collection import (
    BatchedCoordinates,
    BatchedFeatures,
    PointCollection,
)


class BatchedCOOCoordinates(BatchedCoordinates):
    def check(self):
        # assert self.batched_tensors.shape[1] == 3, "Coordinates must have 3 dimensions"
        assert self.batched_tensors.dtype in [
            torch.int32,
            torch.int64,
        ], "Coordinates must be integers"


class SpatiallySparseTensor(PointCollection):
    batched_coordinates: BatchedCOOCoordinates
    batched_features: BatchedFeatures
    _ordering: POINT_ORDERING

    def __init__(
        self,
        batched_coordinates: List[Float[Tensor, "N 3"]] | BatchedCOOCoordinates,
        batched_features: List[Float[Tensor, "N C"]] | BatchedFeatures,
        _ordering: Optional[POINT_ORDERING] = POINT_ORDERING.RANDOM,
    ):
        if isinstance(batched_coordinates, list):
            assert isinstance(
                batched_features, list
            ), "If coords is a list, features must be a list too."
            assert len(batched_coordinates) == len(batched_features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(batched_coordinates, batched_features))
            batched_coordinates = BatchedCOOCoordinates(batched_coordinates)
            batched_features = BatchedFeatures(batched_features)

        assert isinstance(batched_features, BatchedFeatures) and isinstance(
            batched_coordinates, BatchedCoordinates
        )
        assert len(batched_coordinates) == len(batched_features)
        assert (batched_coordinates.offsets == batched_features.offsets).all()
        # The rest of the shape checks are assumed to be done in the BatchedObject
        self.batched_coordinates = batched_coordinates
        self.batched_features = batched_features
        self._ordering = _ordering
