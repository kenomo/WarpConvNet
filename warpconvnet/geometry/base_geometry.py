from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warpconvnet.ops.batch_copy import cat_to_pad, pad_to_cat
from warpconvnet.utils.batch_index import batch_indexed_coordinates
from warpconvnet.utils.list_to_batch import list_to_cat_tensor, list_to_pad_tensor

__all__ = [
    "BatchedObject",
    "CatBatchedObject",
    "PadBatchedObject",
    "BatchedCoordinates",
    "SpatialFeatures",
    "PadBatchedFeatures",
    "CatBatchedFeatures",
]


@dataclass
class BatchedObject:
    batched_tensor: Float[Tensor, "N C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"]  # noqa: F722,F821

    def __init__(
        self,
        batched_tensor: (List[Float[Tensor, "N C"]] | Float[Tensor, "N C"]),  # noqa: F722,F821
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
            batched_tensor, offsets, _ = list_to_cat_tensor(batched_tensor)

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
        assert isinstance(idx, int), "Index must be an integer"
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


class CatBatchedObject(BatchedObject):
    def check(self):
        super().check()
        assert self.batched_tensor.ndim == 2, "Batched tensor must be 2D"
        assert (
            self.batched_tensor.shape[0] == self.offsets[-1]
        ), f"Offsets {self.offsets} does not match tensors {self.batched_tensor.shape}"

    def from_pad(self, pad_to_multiple: Optional[int] = None) -> "PadBatchedObject":
        return PadBatchedObject.from_cat(self, pad_to_multiple)

    def to_pad(self, pad_to_multiple: Optional[int] = None) -> "PadBatchedObject":
        batched_tensor = cat_to_pad(self.batched_tensor, self.offsets, pad_to_multiple)
        return PadBatchedObject(batched_tensor, self.offsets, pad_to_multiple)


@dataclass
class PadBatchedObject(BatchedObject):
    batched_tensor: Float[Tensor, "B M C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B+1"]  # noqa: F722,F821
    pad_to_multiple: Optional[int] = None

    def __init__(
        self,
        batched_tensor: (List[Float[Tensor, "N C"]] | Float[Tensor, "B M C"]),  # noqa: F722,F821
        offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F722,F821
        pad_to_multiple: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if isinstance(batched_tensor, list):
            assert offsets is None, "If batched_tensors is a list, offsets must be None"
            batched_tensor, offsets, _ = list_to_pad_tensor(batched_tensor, pad_to_multiple)

        if isinstance(batched_tensor, torch.Tensor) and offsets is None:
            assert (
                pad_to_multiple is not None
            ), "pad_to_multiple must be provided if batched_tensor is a tensor"
            if batched_tensor.ndim == 2:
                batched_tensor = batched_tensor.unsqueeze(0)
            offsets = [0, batched_tensor.shape[1]]

        assert batched_tensor.ndim == 3, "Batched tensor must be 3D"

        self.batched_tensor = batched_tensor
        self.offsets = offsets
        self.pad_to_multiple = pad_to_multiple
        if device is not None:
            self.batched_tensor = self.batched_tensor.to(device)

    def check(self):
        super().check()
        assert self.batched_tensor.ndim == 2, "Batched tensor must be 2D"
        assert (
            self.batched_tensor.shape[0] == self.offsets[-1]
        ), f"Offsets {self.offsets} does not match tensors {self.batched_tensor.shape}"

    @property
    def batch_size(self):
        return self.batched_tensor.shape[0]

    @property
    def max_num_points(self):
        return self.batched_tensor.shape[1]

    @property
    def num_channels(self):
        return self.batched_tensor.shape[2]

    def to(self, device: str) -> "PadBatchedObject":
        return PadBatchedObject(
            batched_tensor=self.batched_tensor.to(device),
            offsets=self.offsets.to(device),
            pad_to_multiple=self.pad_to_multiple,
        )

    def equal_shape(self, value: "PadBatchedObject") -> bool:
        if not isinstance(value, PadBatchedObject):
            return False
        return (
            (self.offsets == value.offsets).all()
            and self.max_num_points == value.max_num_points
            and self.num_channels == value.num_channels
        )

    def equal_rigorous(self, value: "PadBatchedObject") -> bool:
        if not isinstance(value, PadBatchedObject):
            return False
        return self.equal_shape(value) and (self.batched_tensor == value.batched_tensor).all()

    def to_cat(self) -> CatBatchedObject:
        unpadded_features = pad_to_cat(self.batched_tensor, self.offsets)
        return CatBatchedObject(unpadded_features, self.offsets)

    @classmethod
    def from_cat(
        cls, batched_object: CatBatchedObject, pad_to_multiple: Optional[int] = None
    ) -> "PadBatchedObject":
        padded_tensor = cat_to_pad(
            batched_object.batched_tensor,
            batched_object.offsets,
            pad_to_multiple=pad_to_multiple,
        )
        return cls(padded_tensor, batched_object.offsets, pad_to_multiple)

    def clear_padding(self, clear_value: float = 0.0) -> None:
        """
        Clear the padded part of the tensor
        """
        num_points = self.offsets.diff()
        for i in range(self.batch_size):
            self.batched_tensor[i, num_points[i] :, :] = clear_value

    def replace(
        self,
        batched_tensor: Optional[Float[Tensor, "B M C"]] = None,
        offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
        pad_to_multiple: Optional[int] = None,
        **kwargs,
    ):
        batched_tensor = batched_tensor if batched_tensor is not None else self.batched_tensor
        if pad_to_multiple is not None:
            # pad the tensor to the same multiple as the original tensor
            new_num_points = (
                (batched_tensor.shape[1] + pad_to_multiple - 1)
                // pad_to_multiple
                * pad_to_multiple
            )
            if new_num_points > batched_tensor.shape[1]:
                batched_tensor = F.pad(
                    batched_tensor, (0, 0, 0, new_num_points - batched_tensor.shape[1])
                )
        return self.__class__(
            batched_tensor=batched_tensor,
            offsets=(offsets if offsets is not None else self.offsets),
            pad_to_multiple=pad_to_multiple,
            **kwargs,
        )


class CatBatchedFeatures(CatBatchedObject):
    @property
    def num_channels(self):
        return self.batched_tensor.shape[-1]

    def equal_shape(self, value: object) -> bool:
        if not isinstance(value, CatBatchedFeatures):
            return False
        return (
            (self.offsets == value.offsets).all()
            and self.numel() == value.numel()
            and self.num_channels == value.num_channels
        )


class PadBatchedFeatures(PadBatchedObject):
    @property
    def num_channels(self):
        return self.batched_tensor.shape[-1]

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:  # noqa: F722,F821
        return self.batched_tensor[idx]


def to_batched_features(
    features: Union[CatBatchedFeatures, PadBatchedFeatures, Tensor],
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    device: Optional[str] = None,
) -> Union[CatBatchedFeatures, PadBatchedFeatures]:
    if isinstance(features, Tensor):
        if features.ndim == 2:
            return CatBatchedFeatures(features, offsets, device=device)
        elif features.ndim == 3:
            return PadBatchedFeatures(features, offsets, device=device)
        else:
            raise ValueError(f"Invalid features tensor shape {features.shape}")
    else:
        assert isinstance(features, (CatBatchedFeatures, PadBatchedFeatures))
        if device is not None:
            features = features.to(device)
        return features


@dataclass
class SpatialFeatures:
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    batched_coordinates: BatchedCoordinates
    batched_features: Union[CatBatchedFeatures, PadBatchedFeatures]
    _extra_attributes: Dict[str, Any] = field(default_factory=dict, init=True)  # Store extra args

    def __init__(
        self,
        batched_coordinates: BatchedCoordinates,
        batched_features: Union[CatBatchedFeatures, PadBatchedFeatures, Tensor],
        **kwargs,
    ):
        self.batched_coordinates = batched_coordinates
        self.batched_features = to_batched_features(
            batched_features,
            batched_coordinates.offsets,
            device=kwargs.get("device", None),
        )

        assert (batched_coordinates.offsets == batched_features.offsets).all()
        # Extra arguments for subclasses
        # First check _extra_attributes in kwargs. This happens when we use dataclasses.replace
        if "_extra_attributes" in kwargs:
            attr = kwargs.pop("_extra_attributes")
            assert isinstance(attr, dict), f"_extra_attributes must be a dictionary, got {attr}"
            # Update kwargs
            for k, v in attr.items():
                kwargs[k] = v
        self._extra_attributes = kwargs

    def __getitem__(self, idx: int) -> "SpatialFeatures":
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return self.__class__(
            batched_coordinates=coords,
            batched_features=features,
            offsets=torch.tensor([0, len(coords)]),
            **self._extra_attributes,
        )

    def to(self, device: str) -> "SpatialFeatures":
        return self.__class__(
            batched_coordinates=self.batched_coordinates.to(device),
            batched_features=self.batched_features.to(device),
            **self._extra_attributes,
        )

    @property
    def num_spatial_dims(self) -> int:
        return self.batched_coordinates.num_spatial_dims

    @property
    def coordinate_tensor(self) -> Tensor:
        return self.batched_coordinates.batched_tensor

    @property
    def coordinates(self) -> Tensor:
        return self.batched_coordinates.batched_tensor

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return batch_indexed_coordinates(self.coordinate_tensor, self.offsets)

    @property
    def feature_tensor(self) -> Tensor:
        return self.batched_features.batched_tensor

    @property
    def features(self) -> Tensor:
        return self.batched_features.batched_tensor

    @property
    def padded_features(self) -> PadBatchedFeatures:
        """
        Explicitly convert batched features to padded features if necessary
        """
        if isinstance(self.batched_features, PadBatchedFeatures):
            return self.batched_features
        elif isinstance(self.batched_features, CatBatchedFeatures):
            padded_tensor = cat_to_pad(
                self.batched_features.batched_tensor,
                self.offsets,
            )
            return PadBatchedFeatures(padded_tensor, self.offsets)
        else:
            raise ValueError(f"Unsupported features type: {type(self.batched_features)}")

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

    @property
    def shape(self):
        raise ValueError("Specify shape in subclass")

    @property
    def dtype(self):
        return self.batched_features.dtype

    def sort(self):
        raise NotImplementedError

    def _apply_feature_transform(self, feature_transform_fn):
        """
        Apply a feature transform to the features
        """
        out_features = feature_transform_fn(self.feature_tensor)
        return self.replace(batched_features=out_features)

    def half(self):
        return self._apply_feature_transform(lambda x: x.half())

    def float(self):
        return self._apply_feature_transform(lambda x: x.float())

    def double(self):
        return self._apply_feature_transform(lambda x: x.double())

    def binary_op(self, value: object, op: str) -> "SpatialFeatures":
        if isinstance(value, SpatialFeatures):
            assert self.equal_shape(value), f"Shapes do not match. {self} != {value}"
            return self._apply_feature_transform(lambda x: getattr(x, op)(value.feature_tensor))
        elif isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
            return self._apply_feature_transform(lambda x: getattr(x, op)(value))
        elif isinstance(value, torch.Tensor):
            assert self.equal_shape(value)
            return self._apply_feature_transform(lambda x: getattr(x, op)(value))
        else:
            raise NotImplementedError

    def __add__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__add__")

    def __sub__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__sub__")

    def __mul__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__mul__")

    def __truediv__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__truediv__")

    def __floordiv__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__floordiv__")

    def __mod__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__mod__")

    def __pow__(self, value: object) -> "SpatialFeatures":
        return self.binary_op(value, "__pow__")

    def equal_rigorous(self, value: object) -> bool:
        raise NotImplementedError

    def equal_shape(self, value: object) -> bool:
        return self.batched_coordinates.equal_shape(
            value.batched_coordinates
        ) and self.batched_features.equal_shape(value.batched_features)

    def __str__(self) -> str:
        """Short representation of the object."""
        return f"{self.__class__.__name__}(feature_shape={self.feature_tensor.shape}, coords_shape={self.batched_coordinates.shape})"

    def __repr__(self) -> str:
        """Detailed representation of the object."""
        out_str = f"{self.__class__.__name__}(offsets={self.offsets.tolist()}, feature_shape={self.feature_tensor.shape}, coords_shape={self.batched_coordinates.shape}, device={self.device}, dtype={self.feature_tensor.dtype}"
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
        return self.offsets[-1] * self.num_channels

    @property
    def extra_attributes(self):
        return self._extra_attributes.copy()

    @property
    def cache(self):
        return self._extra_attributes.get("_cache")

    def replace(
        self,
        batched_coordinates: Optional[BatchedCoordinates] = None,
        batched_features: Optional[Union[CatBatchedFeatures, PadBatchedFeatures, Tensor]] = None,
        **kwargs,
    ):
        """
        Replace the coordinates or features of the point collection.
        """
        # Combine extra attributes and kwargs
        if "_extra_attributes" in kwargs:  # flatten extra attributes
            _extra_attributes = kwargs.pop("_extra_attributes")
            kwargs = {**_extra_attributes, **kwargs}

        assert "batched_features" not in kwargs, "Use features instead of batched_features"

        new_coords = (
            batched_coordinates if batched_coordinates is not None else self.batched_coordinates
        )
        new_features = batched_features if batched_features is not None else self.batched_features
        if isinstance(new_features, torch.Tensor):
            new_features = to_batched_features(new_features, new_coords.offsets)

        new_kwargs = {**self._extra_attributes, **kwargs}
        return self.__class__(new_coords, new_features, **new_kwargs)
