from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warpconvnet.utils.batch_index import batch_indexed_coordinates
from warpconvnet.utils.list_to_batch import list_to_batched_tensor

__all__ = [
    "BatchedObject",
    "BatchedCoordinates",
    "BatchedFeatures",
    "BatchedSpatialFeatures",
]


@dataclass
class BatchedObject:
    batched_tensor: Float[Tensor, "N C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"]  # noqa: F722,F821

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],  # noqa: F722,F821
        offsets: Optional[List[int]] = None,
        device: Optional[str] = None,
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
            batched_tensor, offsets, _ = list_to_batched_tensor(batched_tensor)

        if isinstance(batched_tensor, torch.Tensor) and offsets is None:
            offsets = [0, batched_tensor.shape[0]]

        if isinstance(offsets, list):
            offsets = torch.LongTensor(offsets)

        self.offsets = offsets.cpu()
        if device is not None:
            batched_tensor = batched_tensor.to(device)
        self.batched_tensor = batched_tensor

        self.check()

    @property
    def batch_size(self) -> int:
        return len(self.offsets) - 1

    def check(self):
        # offset check
        assert isinstance(
            self.offsets, (torch.IntTensor, torch.LongTensor)
        ), f"Offsets must be a cpu IntTensor or cpu LongTensor, got {self.offsets}"
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

    def equal_shape(self, value: "BatchedObject") -> bool:
        return (self.offsets == value.offsets).all() and self.numel() == value.numel()

    def equal_rigorous(self, value: "BatchedObject") -> bool:
        if not isinstance(value, BatchedObject):
            return False
        return self.equal_shape(value) and (self.batched_tensor == value.batched_tensor).all()

    def __eq__(self, value: "BatchedObject") -> bool:
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
    @property
    def num_spatial_dims(self):
        return self.batched_tensor.shape[1]  # tensor does not have batch index

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

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return batch_indexed_coordinates(self.batched_tensor, self.offsets)


class BatchedIndices(BatchedObject):
    batched_tensor: Int[Tensor, "N"]  # noqa: F722,F821
    # offsets already defined in BatchedObject

    def half(self):
        raise ValueError("Cannot convert indices to half")

    def float(self):
        raise ValueError("Cannot convert indices to float")

    def double(self):
        raise ValueError("Cannot convert indices to double")


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
    _extra_attributes: Dict[str, Any] = field(default_factory=dict, init=True)  # Store extra args

    def __init__(
        self,
        batched_coordinates: BatchedCoordinates,
        batched_features: BatchedFeatures,
        **kwargs,  # extra arguments for subclasses
    ):
        assert isinstance(batched_features, BatchedFeatures) and isinstance(
            batched_coordinates, BatchedCoordinates
        )
        assert (batched_coordinates.offsets == batched_features.offsets).all()
        assert len(batched_coordinates) == len(batched_features)
        # The rest of the shape checks are assumed to be done in the BatchedObject
        self.batched_coordinates = batched_coordinates
        self.batched_features = batched_features
        # Extra arguments for subclasses
        # First check _extra_attributes in kwargs. This happens when we use dataclasses.replace
        if "_extra_attributes" in kwargs:
            attr = kwargs.pop("_extra_attributes")
            assert isinstance(attr, dict), f"_extra_attributes must be a dictionary, got {attr}"
            # Update kwargs
            for k, v in attr.items():
                kwargs[k] = v
        self._extra_attributes = kwargs

    def __getitem__(self, idx: int) -> "BatchedSpatialFeatures":
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return self.__class__(
            batched_coordinates=coords,
            batched_features=features,
            offsets=torch.tensor([0, len(coords)]),
            **self._extra_attributes,
        )

    def to(self, device: str) -> "BatchedSpatialFeatures":
        return self.__class__(
            batched_coordinates=self.batched_coordinates.to(device),
            batched_features=self.batched_features.to(device),
            **self._extra_attributes,
        )

    @property
    def num_spatial_dims(self):
        return self.batched_coordinates.num_spatial_dims

    @property
    def coordinate_tensor(self):
        return self.batched_coordinates.batched_tensor

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return batch_indexed_coordinates(self.coordinate_tensor, self.offsets)

    @property
    def coordinates(self):
        return self.coordinate_tensor

    @property
    def feature_tensor(self):
        return self.batched_features.batched_tensor

    @property
    def features(self):
        return self.feature_tensor

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

    @property
    def batch_size(self) -> int:
        return len(self.offsets) - 1

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
            **self._extra_attributes,
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
        out_str = f"{self.__class__.__name__}(offsets={self.offsets.tolist()}, feature_shape={self.feature_shape}, coords_shape={self.coordinate_shape}, device={self.device}, dtype={self.batched_features.dtype}"
        if self._extra_attributes:
            out_dict = {k: v for k, v in self._extra_attributes.items() if v is not None}
            # if out_dict has values, add it to the string
            if out_dict:
                out_str += ", "
                out_str += ", ".join([f"{k}={v}" for k, v in out_dict.items()])
        out_str += ")"
        return out_str

    def __len__(self) -> int:
        return len(self.batched_coordinates)

    def numel(self):
        return self.batched_features.numel()

    @property
    def extra_attributes(self):
        return self._extra_attributes.copy()

    def replace(
        self,
        batched_coordinates: Optional[BatchedCoordinates] = None,
        batched_features: Optional[BatchedFeatures] = None,
        **kwargs,
    ):
        """
        Replace the coordinates or features of the point collection.
        """
        # Combine extra attributes and kwargs
        if "_extra_attributes" in kwargs:  # flatten extra attributes
            _extra_attributes = kwargs.pop("_extra_attributes")
            kwargs = {**_extra_attributes, **kwargs}

        if isinstance(batched_features, torch.Tensor):
            assert batched_features.shape[0] == self.feature_tensor.shape[0], (
                f"Feature length {batched_features.shape[0]} does not match the original feature length "
                f"{self.feature_tensor.shape[0]}"
            )
            batched_features = BatchedFeatures(batched_features, self.offsets)

        kwargs = {**self.extra_attributes, **kwargs}
        return self.__class__(
            batched_coordinates=(
                self.batched_coordinates if batched_coordinates is None else batched_coordinates
            ),
            batched_features=(
                self.batched_features if batched_features is None else batched_features
            ),
            **kwargs,
        )
