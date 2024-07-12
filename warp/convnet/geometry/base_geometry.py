from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING

__all__ = [
    "BatchedObject",
    "BatchedCoordinates",
    "BatchedFeatures",
    "BatchedSpatialFeatures",
]


def _list_to_batched_tensor(
    tensor_list: List[Float[Tensor, "N C"]]  # noqa: F821
) -> Tuple[Float[Tensor, "M C"], Int[Tensor, "B+1"], int]:  # noqa: F821
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

    def equal_shape(self, value: object) -> bool:
        return (self.offsets == value.offsets).all() and self.numel() == value.numel()

    def equal_rigorous(self, value: object) -> bool:
        if not isinstance(value, BatchedObject):
            return False
        return self.equal_shape(value) and (self.batched_tensor == value.batched_tensor).all()

    def __eq__(self, value: object) -> bool:
        """
        Accelerated version that only checks length and offsets
        """
        return self.equal_shape(value)

    def binary_op(self, value: object, op: str) -> "BatchedObject":
        """
        Apply a binary operation to the batched tensor
        """
        if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
            return self.__class__(
                batched_tensor=getattr(self.batched_tensor, op)(value),
                offsets=self.offsets,
            )

        assert self.equal_shape(value)
        return self.__class__(
            batched_tensor=getattr(self.batched_tensor, op)(value.batched_tensor),
            offsets=self.offsets,
        )

    def __add__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__add__")

    def __sub__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__sub__")

    def __mul__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__mul__")

    def __truediv__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__truediv__")

    def __floordiv__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__floordiv__")

    def __mod__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__mod__")

    def __pow__(self, value: object) -> "BatchedObject":
        return self.binary_op(value, "__pow__")

    def __str__(self) -> str:
        """Short representation of the object."""
        return (
            f"{self.__class__.__name__}(offsets={self.offsets}, shape={self.batched_tensor.shape})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the object."""
        return f"{self.__class__.__name__}(offsets={self.offsets}, shape={self.batched_tensor.shape}, device={self.device}, dtype={self.dtype})"


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

    def equal_shape(self, value: object) -> bool:
        return (
            (self.offsets == value.offsets).all()
            and self.numel() == value.numel()
            and self.num_channels == value.num_channels
        )


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
    def coordinate_tensor(self):
        return self.batched_coordinates.batched_tensor

    @property
    def feature_tensor(self):
        return self.batched_features.batched_tensor

    @property
    def coordinate_shape(self):
        return tuple(self.batched_coordinates.shape)

    @property
    def feature_shape(self):
        return tuple(self.batched_features.shape)

    @property
    def offsets(self):
        return self.batched_coordinates.offsets

    @property
    def device(self):
        return self.batched_coordinates.device

    @property
    def num_channels(self):
        return self.batched_features.num_channels

    def sort(self):
        raise NotImplementedError

    def _apply_feature_transform(self, feature_transform_fn):
        """
        Apply a feature transform to the features
        """
        out_features = feature_transform_fn(self.feature_tensor)
        return self.__class__(
            batched_coordinates=self.batched_coordinates,
            batched_features=BatchedFeatures(out_features, self.batched_features.offsets),
            _ordering=self._ordering,
        )

    def half(self):
        return self._apply_feature_transform(lambda x: x.half())

    def float(self):
        return self._apply_feature_transform(lambda x: x.float())

    def double(self):
        return self._apply_feature_transform(lambda x: x.double())

    def binary_op(self, value: object, op: str) -> "BatchedSpatialFeatures":
        if isinstance(value, BatchedSpatialFeatures):
            assert self.equal_shape(value), f"Shapes do not match. {self} != {value}"
            return self._apply_feature_transform(lambda x: getattr(x, op)(value.feature_tensor))
        elif isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
            return self._apply_feature_transform(lambda x: getattr(x, op)(value))
        elif isinstance(value, torch.Tensor):
            assert self.equal_shape(value)
            return self._apply_feature_transform(lambda x: getattr(x, op)(value))
        else:
            raise NotImplementedError

    def __add__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__add__")

    def __sub__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__sub__")

    def __mul__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__mul__")

    def __truediv__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__truediv__")

    def __floordiv__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__floordiv__")

    def __mod__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__mod__")

    def __pow__(self, value: object) -> "BatchedSpatialFeatures":
        return self.binary_op(value, "__pow__")

    def equal_rigorous(self, value: object) -> bool:
        raise NotImplementedError

    def equal_shape(self, value: object) -> bool:
        return self.batched_coordinates.equal_shape(
            value.batched_coordinates
        ) and self.batched_features.equal_shape(value.batched_features)

    def __str__(self) -> str:
        """Short representation of the object."""
        return f"{self.__class__.__name__}(feature_shape={self.feature_shape}, coords_shape={self.coordinate_shape})"

    def __repr__(self) -> str:
        """Detailed representation of the object."""
        return f"{self.__class__.__name__}(offsets={self.offsets}, feature_shape={self.feature_shape}, coords_shape={self.coordinate_shape}, device={self.device}, dtype={self.batched_features.dtype})"

    def numel(self):
        return self.batched_features.numel()
