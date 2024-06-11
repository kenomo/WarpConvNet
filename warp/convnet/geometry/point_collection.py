from typing import Optional, List
from enum import Enum

import warp as wp

from jaxtyping import Float, Int
import torch
from torch import Tensor


class BatchedObject:
    offsets: List[int]
    batched_tensors: Float[Tensor, "N C"]
    batch_size: int

    def __init__(
        self,
        tensors: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],
        offsets: Optional[List[int]] = None,
    ):
        """
        Initialize a batched object with a list of tensors.

        Args:
            tensors: List of tensors to batch or a single tensor. If you provide
            a concatenated tensor, you must provide the offsets.
            
            offsets: List of offsets for each tensor in the batch. If None, the
            tensors are assumed to be a list.
        """
        if isinstance(tensors, list):
            assert offsets is None, "If tensors is a list, offsets must be None."
            self.batch_size = len(tensors)
            offsets = [0] + [len(c) for c in tensors]
            tensors = torch.cat(tensors, dim=0)

        assert offsets is not None and isinstance(tensors, torch.Tensor)
        self.batch_size = len(offsets) - 1
        self.offsets = offsets
        self.batched_tensors = tensors

        self.check()

    def check(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:
        return self.batched_tensors[self.offsets[idx] : self.offsets[idx + 1]]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, offsets={self.offsets})"


class BatchedCoordinates(BatchedObject):
    def check(self):
        assert self.batched_tensors.shape[-1] == 3, "Coordinates must have 3 dimensions"


class BatchedFeatures(BatchedObject):
    def check(self):
        pass


class PointCollection:
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    batched_coordinates: BatchedCoordinates
    batched_features: BatchedFeatures
    _ordering: POINT_ORDERING = POINT_ORDERING.RANDOM

    def __init__(
        self,
        coords: List[Float[Tensor, "N 3"]] | BatchedCoordinates,
        features: List[Float[Tensor, "N C"]] | BatchedFeatures,
    ):
        """
        Initialize a point collection with coordinates and features.
        """
        if isinstance(coords, list):
            assert isinstance(
                features, list
            ), "If coords is a list, features must be a list too."
            assert len(coords) == len(features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(coords, features))
            coords = BatchedCoordinates(coords)
            features = BatchedFeatures(features)

        assert isinstance(features, BatchedFeatures) and isinstance(
            coords, BatchedCoordinates
        )
        self.batched_coordinates = coords
        self.batched_features = features

    def __len__(self) -> int:
        return self.batched_coordinates.batch_size

    def __getitem__(self, idx: int) -> Float[Tensor, "N D"]:
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return PointCollection(
            coords=coords, features=features, offsets=[0, len(coords)]
        )

    def sort(
        self,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
        voxel_size: Optional[float] = None,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smalles descritization and points in the same voxel will have random order.
        """
        pass

    def neighbors(self, radius: float) -> NeighborSearchReturn:
        """
        Returns CSR format neighbor indices
        """
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
