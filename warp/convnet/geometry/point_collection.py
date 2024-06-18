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


class BatchedObject:
    offsets: Int[Tensor, "B + 1"]  # noqa: F722,F821
    batched_tensors: Float[Tensor, "N C"]  # noqa: F722,F821
    batch_size: int

    def __init__(
        self,
        tensors: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],  # noqa: F722,F821
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
            # cumsum the offsets
            offsets = torch.tensor(offsets, requires_grad=False).cumsum(dim=0).int()
            tensors = torch.cat(tensors, dim=0)

        assert offsets is not None and isinstance(tensors, torch.Tensor)
        assert (
            tensors.shape[0] == offsets[-1]
        ), f"Offsets {offsets} does not match tensors {tensors.shape}"
        self.batch_size = len(offsets) - 1
        if isinstance(offsets, list):
            offsets = torch.tensor(offsets, requires_grad=False).int()
        self.offsets = offsets
        self.batched_tensors = tensors

        self.check()

    def check(self):
        raise NotImplementedError

    def to(self, device: str) -> "BatchedObject":
        return self.__class__(
            tensors=self.batched_tensors.to(device),
            offsets=self.offsets,
        )

    @property
    def device(self):
        return self.batched_tensors.device

    @property
    def shape(self):
        return self.batched_tensors.shape

    @property
    def dtype(self):
        return self.batched_tensors.dtype

    def numel(self):
        return self.batched_tensors.numel()

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:  # noqa: F722,F821
        return self.batched_tensors[self.offsets[idx] : self.offsets[idx + 1]]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, offsets={self.offsets})"


class BatchedCoordinates(BatchedObject):
    def check(self):
        assert self.batched_tensors.shape[-1] == 3, "Coordinates must have 3 dimensions"

    def voxel_downsample(self, voxel_size: float):
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        perm, down_offsets, _, _ = voxel_downsample(
            coords=self.batched_tensors,
            voxel_size=voxel_size,
            offsets=self.offsets,
        )
        return self.__class__(tensors=self.batched_tensors[perm], offsets=down_offsets)

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
            self.batched_tensors,
            self.offsets,
            query_coords.batched_tensors,
            query_coords.offsets,
            search_args,
        )


class BatchedFeatures(BatchedObject):
    def check(self):
        pass

    @property
    def num_channels(self):
        return self.batched_tensors.shape[-1]


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
        coords: List[Float[Tensor, "N 3"]] | BatchedCoordinates,  # noqa: F722,F821
        features: List[Float[Tensor, "N C"]] | BatchedFeatures,  # noqa: F722,F821
    ):
        """
        Initialize a point collection with coordinates and features.
        """
        if isinstance(coords, list):
            assert isinstance(features, list), "If coords is a list, features must be a list too."
            assert len(coords) == len(features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(coords, features))
            coords = BatchedCoordinates(coords)
            features = BatchedFeatures(features)

        assert isinstance(features, BatchedFeatures) and isinstance(coords, BatchedCoordinates)
        assert len(coords) == len(features)
        assert (coords.offsets == features.offsets).all()
        # The rest of the shape checks are assumed to be done in the BatchedObject
        self.batched_coordinates = coords
        self.batched_features = features

    def __len__(self) -> int:
        return self.batched_coordinates.batch_size

    def __getitem__(self, idx: int) -> Float[Tensor, "N D"]:
        coords = self.batched_coordinates[idx]
        features = self.batched_features[idx]
        return PointCollection(coords=coords, features=features, offsets=[0, len(coords)])

    def to(self, device: str) -> "PointCollection":
        return self.__class__(
            coords=self.batched_coordinates.to(device),
            features=self.batched_features.to(device),
        )

    @property
    def coords(self):
        return self.batched_coordinates.batched_tensors

    @property
    def features(self):
        return self.batched_features.batched_tensors

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
            coords=self.batched_coordinates.batched_tensors,
            features=self.batched_features.batched_tensors,
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
                coords=BatchedCoordinates(tensors=self.coords[perm], offsets=down_offsets),
                features=BatchedFeatures(tensors=self.features[perm], offsets=down_offsets),
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
            coords=BatchedCoordinates(tensors=down_coords, offsets=down_offsets),
            features=BatchedFeatures(tensors=down_features, offsets=down_offsets),
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_size={len(self)}, num_channels={self.num_channels})"
        )
