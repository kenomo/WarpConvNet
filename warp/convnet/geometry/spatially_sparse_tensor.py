from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warp.convnet.core.hashmap import VectorHashTable
from warp.convnet.geometry.base_geometry import (
    BatchedCoordinates,
    BatchedFeatures,
    BatchedSpatialFeatures,
    _list_to_batched_tensor,
)
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample_random_indices
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING, sorting_permutation
from warp.convnet.utils.batch_index import batch_indexed_coordinates


class BatchedDiscreteCoordinates(BatchedCoordinates):
    voxel_size: float
    voxel_origin: Float[Tensor, "3"]
    _hashmap: Optional[VectorHashTable]

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],  # noqa: F722,F821
        offsets: Optional[List[int]] = None,
        voxel_size: Optional[float] = None,
        voxel_origin: Optional[Float[Tensor, "3"]] = None,
    ):
        """

        Args:
            batched_tensor: provides the coordinates of the points
            offsets: provides the offsets for each batch
            voxel_size: provides the size of the voxel for converting the coordinates to points
            voxel_origin: provides the origin of the voxel for converting the coordinates to points
        """
        if isinstance(batched_tensor, list):
            assert offsets is None, "If batched_tensors is a list, offsets must be None."
            batched_tensor, offsets, _ = _list_to_batched_tensor(batched_tensor)

        if isinstance(offsets, list):
            offsets = torch.LongTensor(offsets, requires_grad=False)

        self.offsets = offsets
        self.batched_tensor = batched_tensor
        self.voxel_size = voxel_size
        self.voxel_origin = voxel_origin

        self.check()

    def check(self):
        BatchedCoordinates.check(self)
        assert self.batched_tensor.dtype in [
            torch.int32,
            torch.int64,
        ], "Discrete coordinates must be integers"

    def sort(
        self, ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER
    ) -> "BatchedDiscreteCoordinates":
        perm, rank = sorting_permutation(self.batched_tensor, self.offsets, ordering)  # noqa: F821
        return self.__class__(tensors=self.batched_tensor[perm], offsets=self.offsets)

    def neighbors(
        self,
        search_args: "NeighborSearchArgs",  # noqa: F821
        query_coords: Optional["BatchedDiscreteCoordinates"] = None,
    ) -> "NeighborSearchResult":  # noqa: F821
        """
        Returns CSR format neighbor indices
        """
        if query_coords is None:
            query_coords = self

        assert isinstance(
            query_coords, BatchedDiscreteCoordinates
        ), "query_coords must be BatchedCoordinates"

        raise NotImplementedError

    def unique(self) -> "BatchedDiscreteCoordinates":
        unique_indices, batch_offsets = voxel_downsample_random_indices(
            self.batched_tensor, self.offsets, self.voxel_size, self.voxel_origin
        )
        return self.__class__(self.batched_tensor[unique_indices], batch_offsets)

    @property
    def hashmap(self) -> VectorHashTable:
        if not hasattr(self, "_hashmap") or self._hashmap is None:
            bcoords = batch_indexed_coordinates(self.batched_tensor, self.offsets)
            self._hashmap = VectorHashTable.from_keys(bcoords)
        return self._hashmap


class SpatiallySparseTensor(BatchedSpatialFeatures):
    batched_coordinates: BatchedDiscreteCoordinates
    batched_features: BatchedFeatures

    def __init__(
        self,
        batched_coordinates: List[Float[Tensor, "N 3"]]
        | Float[Tensor, "N 3"]
        | BatchedDiscreteCoordinates,
        batched_features: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"] | BatchedFeatures,
        offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722,F821
        **kwargs,
    ):
        if isinstance(batched_coordinates, list):
            assert isinstance(
                batched_features, list
            ), "If coords is a list, features must be a list too."
            assert len(batched_coordinates) == len(batched_features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(batched_coordinates, batched_features))
            batched_coordinates = BatchedDiscreteCoordinates(batched_coordinates)
            batched_features = BatchedFeatures(batched_features)
        elif isinstance(batched_coordinates, Tensor):
            assert (
                isinstance(batched_features, Tensor) and offsets is not None
            ), "If coordinate is a tensor, features must be a tensor and offsets must be provided."
            batched_coordinates = BatchedDiscreteCoordinates(
                tensors=batched_coordinates, offsets=offsets
            )
            batched_features = BatchedFeatures(batched_features, offsets=offsets)

        BatchedSpatialFeatures.__init__(self, batched_coordinates, batched_features, **kwargs)

    def sort(self, ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER) -> "SpatiallySparseTensor":
        if ordering == self.ordering:
            return self

        perm, rank = sorting_permutation(
            self.coordinate_tensor, self.offsets, ordering
        )  # noqa: F821
        coords = BatchedDiscreteCoordinates(self.coordinate_tensor[perm], self.offsets)
        feats = BatchedFeatures(self.feature_tensor[perm], self.offsets)
        kwargs = self._extra_attributes.copy()
        kwargs["ordering"] = ordering
        return self.__class__(coords, feats, **kwargs)

    def unique(self) -> "SpatiallySparseTensor":
        unique_indices, batch_offsets = voxel_downsample_random_indices(
            self.coordinate_tensor,
            self.offsets,
        )
        coords = BatchedDiscreteCoordinates(self.coordinate_tensor[unique_indices], batch_offsets)
        feats = BatchedFeatures(self.feature_tensor[unique_indices], batch_offsets)
        return self.__class__(coords, feats, **self._extra_attributes)

    @property
    def coordinate_hashmap(self) -> VectorHashTable:
        return self.batched_coordinates.hashmap

    @property
    def voxel_size(self):
        return self._extra_attributes.get("voxel_size", None)

    @property
    def ordering(self):
        return self._extra_attributes.get("ordering", None)
