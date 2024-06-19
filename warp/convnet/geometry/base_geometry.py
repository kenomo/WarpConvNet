from dataclasses import dataclass
from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING


@dataclass
class BatchedObject:
    batched_tensor: Float[Tensor, "N C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"]  # noqa: F722,F821
    batch_size: int

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],  # noqa: F722,F821
        offsets: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize a batched object with a list of tensors.

        Args:
            tensors: List of tensors to batch or a single tensor. If you provide
            a concatenated tensor, you must provide the offsets.

            offsets: List of offsets for each tensor in the batch. If None, the
            tensors are assumed to be a list.
        """
        if isinstance(batched_tensor, list):
            assert offsets is None, "If tensors is a list, offsets must be None."
            offsets = [0] + [len(c) for c in batched_tensor]
            # cumsum the offsets
            offsets = torch.tensor(offsets, requires_grad=False).cumsum(dim=0).int()
            batched_tensor = torch.cat(batched_tensor, dim=0)

        assert offsets is not None and isinstance(batched_tensor, torch.Tensor)
        assert (
            batched_tensor.shape[0] == offsets[-1]
        ), f"Offsets {offsets} does not match tensors {batched_tensor.shape}"
        if batch_size is None:
            batch_size = len(offsets) - 1
        assert (
            len(offsets) == batch_size + 1
        ), f"Offsets {offsets} does not match batch size {batch_size}"
        if isinstance(offsets, list):
            offsets = torch.tensor(offsets, requires_grad=False).int()
        assert isinstance(offsets, torch.Tensor) and offsets.requires_grad is False
        self.batch_size = batch_size
        self.offsets = offsets
        self.batched_tensor = batched_tensor

        self.check()

    def check(self):
        raise NotImplementedError

    def to(self, device: str) -> "BatchedObject":
        return self.__class__(
            batched_tensor=self.batched_tensor.to(device),
            offsets=self.offsets,
        )

    @property
    def device(self):
        return self.batched_tensor.device

    @property
    def shape(self):
        return self.batched_tensor.shape

    @property
    def dtype(self):
        return self.batched_tensor.dtype

    def numel(self):
        return self.batched_tensor.numel()

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:  # noqa: F722,F821
        return self.batched_tensor[self.offsets[idx] : self.offsets[idx + 1]]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, offsets={self.offsets})"


class BatchedCoordinates(BatchedObject):
    def neighbors(
        self,
        query_coords: "BatchedCoordinates",
        search_args: dict,
    ) -> "NeighborSearchResult":
        """
        Find the neighbors of the query_coords in the current coordinates.

        Args:
            query_coords: The coordinates to search for neighbors
            search_args: Arguments for the search
        """
        raise NotImplementedError


class BatchedFeatures(BatchedObject):
    def check(self):
        pass

    @property
    def num_channels(self):
        return self.batched_tensor.shape[-1]


@dataclass
class BatchedSpatialFeatures:
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    batched_coordinates: BatchedCoordinates
    batched_features: BatchedFeatures
    _ordering: POINT_ORDERING

    def __len__(self) -> int:
        return self.batched_coordinates.batch_size

    def __getitem__(self, idx: int) -> "BatchedSpatialFeatures":
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return self.__class__(
            batched_coordinates=coords, batched_features=features, offsets=[0, len(coords)]
        )

    def to(self, device: str) -> "BatchedSpatialFeatures":
        return self.__class__(
            batched_coordinates=self.batched_coordinates.to(device),
            batched_features=self.batched_features.to(device),
        )

    @property
    def coords(self):
        return self.batched_coordinates.batched_tensor

    @property
    def features(self):
        return self.batched_features.batched_tensor

    @property
    def offsets(self):
        return self.batched_coordinates.offsets

    @property
    def device(self):
        return self.batched_coordinates.device

    @property
    def num_channels(self):
        return self.batched_features.num_channels

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_size={len(self)}, num_channels={self.num_channels})"
        )

    def sort(self):
        raise NotImplementedError
