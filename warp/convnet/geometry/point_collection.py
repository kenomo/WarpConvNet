from dataclasses import dataclass
from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.geometry.ops.neighbor_search import (
    NeighborSearchArgs,
    NeighborSearchReturn,
    neighbor_search,
)
from warp.convnet.geometry.ops.point_pool import (
    FEATURE_POOLING_MODE,
    FeaturePoolingArgs,
    pool_features,
)
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING, sort_point_collection


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
    def check(self):
        assert self.batched_tensor.shape[-1] == 3, "Coordinates must have 3 dimensions"

    def voxel_downsample(self, voxel_size: float):
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        perm, down_offsets, _, _ = voxel_downsample(
            coords=self.batched_tensor,
            voxel_size=voxel_size,
            offsets=self.offsets,
        )
        return self.__class__(tensors=self.batched_tensor[perm], offsets=down_offsets)

    def neighbors(
        self,
        search_args: NeighborSearchArgs,
        query_coords: Optional["BatchedCoordinates"] = None,
    ) -> NeighborSearchReturn:
        """
        Returns CSR format neighbor indices
        """
        if query_coords is None:
            query_coords = self

        assert isinstance(
            query_coords, BatchedCoordinates
        ), "query_coords must be BatchedCoordinates"

        return neighbor_search(
            self.batched_tensor,
            self.offsets,
            query_coords.batched_tensor,
            query_coords.offsets,
            search_args,
        )


class BatchedFeatures(BatchedObject):
    def check(self):
        pass

    @property
    def num_channels(self):
        return self.batched_tensor.shape[-1]


@dataclass
class PointCollection:
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
        batched_coordinates: List[Float[Tensor, "N 3"]] | BatchedCoordinates,  # noqa: F722,F821
        batched_features: List[Float[Tensor, "N C"]] | BatchedFeatures,  # noqa: F722,F821
        _ordering: POINT_ORDERING = POINT_ORDERING.RANDOM,
    ):
        """
        Initialize a point collection with coordinates and features.
        """
        if isinstance(batched_coordinates, list):
            assert isinstance(
                batched_features, list
            ), "If coords is a list, features must be a list too."
            assert len(batched_coordinates) == len(batched_features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(batched_coordinates, batched_features))
            batched_coordinates = BatchedCoordinates(batched_coordinates)
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

    def __len__(self) -> int:
        return self.batched_coordinates.batch_size

    def __getitem__(self, idx: int) -> Float[Tensor, "N D"]:
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return PointCollection(
            batched_coordinates=coords, batched_features=features, offsets=[0, len(coords)]
        )

    def to(self, device: str) -> "PointCollection":
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

    def sort(
        self,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
        grid_size: Optional[int] = 1024,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smalles descritization and points in the same voxel will have random order.
        """
        # Warp uses int32 so only 10 bits per coordinate supported. Thus max 1024.
        assert grid_size <= 1024, f"Grid size must be <= 1024, got {grid_size}"
        assert self.device.type != "cpu", "Sorting is only supported on GPU"
        sorted_coords, sorted_feats = sort_point_collection(
            coords=self.batched_coordinates.batched_tensor,
            features=self.batched_features.batched_tensor,
            ordering=ordering,
            grid_size=grid_size,
            offsets=self.batched_coordinates.offsets,
        )
        return self.__class__(
            coords=BatchedCoordinates(sorted_coords, offsets=self.batched_coordinates.offsets),
            features=BatchedFeatures(sorted_feats, offsets=self.batched_features.offsets),
        )

    def voxel_downsample(
        self,
        voxel_size: float,
        pooling_args: FeaturePoolingArgs,
    ):
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        perm, down_offsets, vox_inices, vox_offsets = voxel_downsample(
            batched_points=self.coords,
            offsets=self.offsets,
            voxel_size=voxel_size,
        )

        if pooling_args.pooling_mode == FEATURE_POOLING_MODE.RANDOM_SAMPLE:
            return self.__class__(
                coords=BatchedCoordinates(batched_tensor=self.coords[perm], offsets=down_offsets),
                features=BatchedFeatures(batched_tensor=self.features[perm], offsets=down_offsets),
            )

        neighbors = NeighborSearchReturn(vox_inices, vox_offsets)
        down_coords = self.coords[perm]
        down_features = pool_features(
            in_feats=self.features,
            down_coords=down_coords,
            neighbors=neighbors,
            pooling_args=pooling_args,
        )

        return self.__class__(
            batched_coordinates=BatchedCoordinates(
                batched_tensor=down_coords, offsets=down_offsets
            ),
            batched_features=BatchedFeatures(batched_tensor=down_features, offsets=down_offsets),
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_size={len(self)}, num_channels={self.num_channels})"
        )
