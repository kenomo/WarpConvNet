from typing import List, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from warp.convnet.geometry.base_geometry import (
    BatchedCoordinates,
    BatchedFeatures,
    BatchedSpatialFeatures,
    _list_to_batched_tensor,
)
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample_random_indices
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING, sorting_permutation


class BatchedDiscreteCoordinates(BatchedCoordinates):
    voxel_size: float
    voxel_origin: Float[Tensor, "3"]

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


class SpatiallySparseTensor(BatchedSpatialFeatures):
    batched_coordinates: BatchedDiscreteCoordinates
    batched_features: BatchedFeatures
    _ordering: POINT_ORDERING

    def __init__(
        self,
        batched_coordinates: List[Float[Tensor, "N 3"]] | BatchedDiscreteCoordinates,
        batched_features: List[Float[Tensor, "N C"]] | BatchedFeatures,
        _ordering: Optional[POINT_ORDERING] = POINT_ORDERING.RANDOM,
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

        BatchedSpatialFeatures.__init__(self, batched_coordinates, batched_features, _ordering)

    def sort(self, ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER) -> "SpatiallySparseTensor":
        if ordering == self._ordering:
            return self

        perm, rank = sorting_permutation(self.coords, self.offsets, ordering)  # noqa: F821
        coords = BatchedDiscreteCoordinates(self.coords[perm], self.offsets)
        feats = BatchedFeatures(self.features[perm], self.offsets)
        return self.__class__(coords, feats, ordering)

    def unique(self) -> "SpatiallySparseTensor":
        unique_indices, batch_offsets = voxel_downsample_random_indices(
            self.coords,
            self.offsets,
        )
        coords = BatchedDiscreteCoordinates(self.coords[unique_indices], batch_offsets)
        feats = BatchedFeatures(self.features[unique_indices], batch_offsets)
        return self.__class__(coords, feats, self._ordering)
