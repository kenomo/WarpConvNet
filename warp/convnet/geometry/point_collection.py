from typing import Optional, List, Tuple
from enum import Enum

import warp as wp

from jaxtyping import Float, Int
import torch
from torch import Tensor

from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING, sort_point_collection
from warp.convnet.geometry.ops.neighbor_search import (
    NeighborSearchReturn,
    SEARCH_MODE,
    batched_knn_search,
    batched_radius_search,
)


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
            # cumsum the offsets
            offsets = torch.tensor(offsets).cumsum(dim=0).tolist()
            tensors = torch.cat(tensors, dim=0)

        assert offsets is not None and isinstance(tensors, torch.Tensor)
        assert (
            tensors.shape[0] == offsets[-1]
        ), f"Offsets {offsets} does not match tensors {tensors.shape}"
        self.batch_size = len(offsets) - 1
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
    
    @property
    def device(self):
        return self.batched_tensors.device
    
    def numel(self):
        return self.batched_tensors.numel()

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

    def to(self, device: str) -> "PointCollection":
        return self.__class__(
            coords=self.batched_coordinates.to(device),
            features=self.batched_features.to(device),
        )

    @property
    def device(self):
        return self.batched_coordinates.device

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
            coords=BatchedCoordinates(
                sorted_coords, offsets=self.batched_coordinates.offsets
            ),
            features=BatchedFeatures(
                sorted_feats, offsets=self.batched_features.offsets
            ),
        )

    def neighbors(
        self,
        mode: SEARCH_MODE = SEARCH_MODE.RADIUS,
        batched_queries: Optional[Float[Tensor, "Q D"]] = None,
        queries_offsets: Optional[Int[Tensor, "B + 1"]] = None,
        radius: Optional[float] = None,
        grid_dim: Optional[int | Tuple[int, int, int]] = 128,
        knn_k: Optional[int] = None,
    ) -> NeighborSearchReturn:
        """
        Returns CSR format neighbor indices
        """
        if batched_queries is None:
            batched_queries = self.batched_coordinates.batched_tensors
            queries_offsets = self.batched_coordinates.offsets

        if mode == SEARCH_MODE.RADIUS:
            assert radius is not None, "Radius must be provided for radius search"
            neighbor_index, neighbor_distance, neighbor_split = batched_radius_search(
                ref_positions=self.batched_coordinates.batched_tensors,
                ref_offsets=self.batched_coordinates.offsets,
                query_positions=batched_queries,
                query_offsets=queries_offsets,
                radius=radius,
                grid_dim=grid_dim,
            )
            return NeighborSearchReturn(
                neighbor_index,
                neighbor_split,
            )

        elif mode == SEARCH_MODE.KNN:
            assert knn_k is not None, "knn_k must be provided for knn search"
            # M x K
            neighbor_index = batched_knn_search(
                ref_positions=self.batched_coordinates.batched_tensors,
                ref_offsets=self.batched_coordinates.offsets,
                query_positions=batched_queries,
                query_offsets=queries_offsets,
                k=knn_k,
            )
            return NeighborSearchReturn(neighbor_index)

    def __str__(self) -> str:
        return self.__class__.__name__
