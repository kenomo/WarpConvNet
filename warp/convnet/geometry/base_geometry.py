from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING


def _list_to_batched_tensor(
    tensor_list: List[Float[Tensor, "N C"]]  # noqa: F821
) -> Tuple[Float[Tensor, "N C"], Int[Tensor, "B+1"], int]:  # noqa: F821
    """
    Convert a list of tensors to a batched tensor.

    Args:
        tensor_list: List of tensors to batch

    Returns:
        A tuple of the batched tensor, offsets, and batch size
    """
    offsets = [0] + [len(c) for c in tensor_list]
    # cumsum the offsets
    offsets = torch.tensor(offsets, requires_grad=False).cumsum(dim=0).int()
    batched_tensor = torch.cat(tensor_list, dim=0)
    return batched_tensor, offsets, len(offsets) - 1


@dataclass
class BatchedObject:
    batched_tensor: Float[Tensor, "N C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"]  # noqa: F722,F821

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],  # noqa: F722,F821
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
        if isinstance(batched_tensor, list):
            assert offsets is None, "If batched_tensors is a list, offsets must be None."
            batched_tensor, offsets, _ = _list_to_batched_tensor(batched_tensor)

        if isinstance(offsets, list):
            offsets = torch.LongTensor(offsets, requires_grad=False)

        self.offsets = offsets
        self.batched_tensor = batched_tensor

        self.check()

    @property
    def batch_size(self) -> int:
        return len(self.offsets) - 1

    def check(self):
        # offset check
        assert isinstance(
            self.offsets, (torch.IntTensor, torch.LongTensor)
        ), f"Offsets must be a tensor, got {type(self.offsets)}"
        assert self.offsets.requires_grad is False, "Offsets must not require grad"
        assert (
            len(self.offsets) == self.batch_size + 1
        ), f"Offsets {self.offsets} does not match batch size {self.batch_size}"
        # batched_tensor check
        assert (
            self.batched_tensor.shape[0] == self.offsets[-1]
        ), f"Offsets {self.offsets} does not match tensors {self.batched_tensor.shape}"
        assert isinstance(self.batched_tensor, torch.Tensor)

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

    def half(self):
        return self.__class__(
            batched_tensor=self.batched_tensor.half(),
            offsets=self.offsets,
        )

    def float(self):
        return self.__class__(
            batched_tensor=self.batched_tensor.float(),
            offsets=self.offsets,
        )

    def double(self):
        return self.__class__(
            batched_tensor=self.batched_tensor.double(),
            offsets=self.offsets,
        )

    def numel(self):
        return self.batched_tensor.numel()

    def __len__(self) -> int:
        return len(self.batched_tensor)

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

    def __init__(
        self,
        batched_coordinates: BatchedCoordinates,
        batched_features: BatchedFeatures,
        _ordering: POINT_ORDERING = POINT_ORDERING.RANDOM,
    ):
        assert isinstance(batched_features, BatchedFeatures) and isinstance(
            batched_coordinates, BatchedCoordinates
        )
        assert (batched_coordinates.offsets == batched_features.offsets).all()
        assert len(batched_coordinates) == len(batched_features)
        # The rest of the shape checks are assumed to be done in the BatchedObject
        self.batched_coordinates = batched_coordinates
        self.batched_features = batched_features
        self._ordering = _ordering

    def __len__(self) -> int:
        return self.batched_coordinates.batch_size

    def numel(self):
        return self.batched_features.numel()

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

    def half(self):
        return self.__class__(
            batched_coordinates=self.batched_coordinates.half(),
            batched_features=self.batched_features.half(),
            _ordering=self._ordering,
        )

    def float(self):
        return self.__class__(
            batched_coordinates=self.batched_coordinates.float(),
            batched_features=self.batched_features.float(),
            _ordering=self._ordering,
        )

    def double(self):
        return self.__class__(
            batched_coordinates=self.batched_coordinates.double(),
            batched_features=self.batched_features.double(),
            _ordering=self._ordering,
        )
