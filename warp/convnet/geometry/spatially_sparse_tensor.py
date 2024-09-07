from typing import List, Optional, Tuple, Union

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
from warp.convnet.utils.batch_index import (
    batch_indexed_coordinates,
    offsets_from_batch_index,
)
from warp.convnet.utils.ntuple import ntuple
from warp.convnet.utils.ravel import ravel_multi_index


class BatchedDiscreteCoordinates(BatchedCoordinates):
    voxel_size: float
    voxel_origin: Float[Tensor, "D"]  # noqa: F821
    tensor_stride: Union[int, Tuple[int, ...]]
    _hashmap: Optional[VectorHashTable]

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N D"]] | Float[Tensor, "N D"],  # noqa: F722,F821
        offsets: Optional[List[int]] = None,
        voxel_size: Optional[float] = None,
        voxel_origin: Optional[Float[Tensor, "D"]] = None,  # noqa: F821
        tensor_stride: Optional[Union[int, Tuple[int, ...]]] = None,
        device: Optional[str] = None,
    ):
        """

        Args:
            batched_tensor: provides the coordinates of the points
            offsets: provides the offsets for each batch
            voxel_size: provides the size of the voxel for converting the coordinates to points
            voxel_origin: provides the origin of the voxel for converting the coordinates to points
            tensor_stride: provides the stride of the tensor for converting the coordinates to points
        """
        if isinstance(batched_tensor, list):
            assert offsets is None, "If batched_tensors is a list, offsets must be None."
            batched_tensor, offsets, _ = _list_to_batched_tensor(batched_tensor)

        if isinstance(offsets, list):
            offsets = torch.LongTensor(offsets, requires_grad=False)

        if device is not None:
            batched_tensor = batched_tensor.to(device)

        self.offsets = offsets
        self.batched_tensor = batched_tensor
        self.voxel_size = voxel_size
        self.voxel_origin = voxel_origin
        # Conver the tensor stride to ntuple
        if tensor_stride is not None:
            self.tensor_stride = ntuple(tensor_stride, ndim=3)
        else:
            self.tensor_stride = None

        self.check()

    def check(self):
        BatchedCoordinates.check(self)
        assert self.batched_tensor.dtype in [
            torch.int32,
            torch.int64,
        ], "Discrete coordinates must be integers"
        if self.tensor_stride is not None:
            assert isinstance(self.tensor_stride, (int, tuple))

    def sort(
        self, ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER
    ) -> "BatchedDiscreteCoordinates":
        perm, rank = sorting_permutation(self.batched_tensor, self.offsets, ordering)  # noqa: F821
        return self.__class__(tensors=self.batched_tensor[perm], offsets=self.offsets)

    def neighbors(
        self,
        search_args: "DiscreteNeighborSearchArgs",  # noqa: F821
        query_coords: Optional["BatchedDiscreteCoordinates"] = None,
    ) -> "DiscreteNeighborSearchResult":  # noqa: F821
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

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return batch_indexed_coordinates(self.batched_tensor, self.offsets)


class SpatiallySparseTensor(BatchedSpatialFeatures):
    batched_coordinates: BatchedDiscreteCoordinates
    batched_features: BatchedFeatures

    def __init__(
        self,
        batched_coordinates: (
            List[Float[Tensor, "N 3"]] | Float[Tensor, "N 3"] | BatchedDiscreteCoordinates
        ),
        batched_features: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"] | BatchedFeatures,
        offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722,F821
        device: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(batched_coordinates, list):
            assert isinstance(
                batched_features, list
            ), "If coords is a list, features must be a list too."
            assert len(batched_coordinates) == len(batched_features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(batched_coordinates, batched_features))
            batched_coordinates = BatchedDiscreteCoordinates(batched_coordinates, device=device)
            batched_features = BatchedFeatures(batched_features, device=device)
        elif isinstance(batched_coordinates, Tensor):
            assert (
                isinstance(batched_features, Tensor) and offsets is not None
            ), "If coordinate is a tensor, features must be a tensor and offsets must be provided."
            batched_coordinates = BatchedDiscreteCoordinates(
                tensors=batched_coordinates, offsets=offsets, device=device
            )
            batched_features = BatchedFeatures(batched_features, offsets=offsets, device=device)

        BatchedSpatialFeatures.__init__(self, batched_coordinates, batched_features, **kwargs)

    @classmethod
    def from_dense(
        cls,
        dense_tensor: Float[Tensor, "B C H W"] | Float[Tensor, "B C H W D"],
        channel_dim: int = 1,
        **kwargs,
    ):
        # Move channel dimension to the end
        dense_tensor = dense_tensor.transpose(channel_dim, -1)
        spatial_shape = dense_tensor.shape[:-1]
        # abs sum all elements in the tensor
        abs_sum = torch.abs(dense_tensor).sum(dim=-1, keepdim=False)
        # Find all non-zero elements. Expected to be sorted.
        non_zero_inds = torch.nonzero(abs_sum)

        # Flatten the spatial dimensions
        flattened_tensor = dense_tensor.flatten(0, -2)

        # Convert multi-dimensional indices to flattened indices
        flattened_indices = ravel_multi_index(non_zero_inds, spatial_shape)

        # Use index_select to get the features
        non_zero_feats = torch.index_select(flattened_tensor, 0, flattened_indices)

        offsets = offsets_from_batch_index(non_zero_inds[:, 0])
        return cls(
            batched_coordinates=BatchedDiscreteCoordinates(non_zero_inds[:, 1:], offsets=offsets),
            batched_features=BatchedFeatures(non_zero_feats, offsets=offsets),
            **kwargs,
        )

    def sort(self, ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER) -> "SpatiallySparseTensor":
        if ordering == self.ordering:
            return self

        perm, rank = sorting_permutation(
            self.coordinate_tensor, self.offsets, ordering
        )  # noqa: F821
        coords = BatchedDiscreteCoordinates(self.coordinate_tensor[perm], self.offsets)
        feats = BatchedFeatures(self.feature_tensor[perm], self.offsets)
        kwargs = self.extra_attributes.copy()
        kwargs["ordering"] = ordering
        return self.__class__(coords, feats, **kwargs)

    def unique(self) -> "SpatiallySparseTensor":
        unique_indices, batch_offsets = voxel_downsample_random_indices(
            self.coordinate_tensor,
            self.offsets,
        )
        coords = BatchedDiscreteCoordinates(self.coordinate_tensor[unique_indices], batch_offsets)
        feats = BatchedFeatures(self.feature_tensor[unique_indices], batch_offsets)
        return self.__class__(coords, feats, **self.extra_attributes)

    @property
    def coordinate_hashmap(self) -> VectorHashTable:
        return self.batched_coordinates.hashmap

    @property
    def voxel_size(self):
        return self.extra_attributes.get("voxel_size", None)

    @property
    def ordering(self):
        return self.extra_attributes.get("ordering", None)

    @property
    def stride(self):
        return self.extra_attributes.get("stride", None)

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return self.batched_coordinates.batch_indexed_coordinates

    @property
    def batch_size(self):
        return self.batched_coordinates.batch_size

    def replace(
        self,
        batched_coordinates: Optional[BatchedDiscreteCoordinates] = None,
        batched_features: Optional[BatchedFeatures] = None,
        **kwargs,
    ) -> "SpatiallySparseTensor":
        # copy the _extra_attributes
        extra_attributes = self.extra_attributes.copy()
        extra_attributes.update(kwargs)
        return self.__class__(
            batched_coordinates or self.batched_coordinates,
            batched_features or self.batched_features,
            **extra_attributes,
        )
