from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.core.hashmap import VectorHashTable
from warpconvnet.geometry.base_geometry import (
    BatchedCoordinates,
    BatchedFeatures,
    BatchedSpatialFeatures,
)
from warpconvnet.geometry.ops.voxel_ops import voxel_downsample_random_indices
from warpconvnet.geometry.ops.warp_sort import POINT_ORDERING, sorting_permutation
from warpconvnet.utils.batch_index import (
    batch_indexed_coordinates,
    offsets_from_batch_index,
)
from warpconvnet.utils.list_to_batch import list_to_batched_tensor
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.utils.ravel import ravel_multi_index


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
            batched_tensor, offsets, _ = list_to_batched_tensor(batched_tensor)

        if isinstance(offsets, list):
            offsets = torch.LongTensor(offsets, requires_grad=False)

        if device is not None:
            batched_tensor = batched_tensor.to(device)

        self.offsets = offsets.cpu()
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
        if channel_dim != -1 or dense_tensor.ndim != channel_dim + 1:
            dense_tensor = dense_tensor.moveaxis(channel_dim, -1)
        spatial_shape = dense_tensor.shape[:-1]
        # abs sum all elements in the tensor
        abs_sum = torch.abs(dense_tensor).sum(dim=-1, keepdim=False)
        # Find all non-zero elements. Expected to be sorted.
        non_zero_inds = torch.nonzero(abs_sum).int()

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

    def to_dense(
        self,
        channel_dim: int = 1,
        spatial_shape: Optional[Tuple[int, ...]] = None,
        min_coords: Optional[Tuple[int, ...]] = None,
        max_coords: Optional[Tuple[int, ...]] = None,
    ) -> Float[Tensor, "B C H W D"] | Float[Tensor, "B C H W"]:
        device = self.batched_coordinates.device

        # Get the batch indexed coordinates and features
        batch_indexed_coords = self.batched_coordinates.batch_indexed_coordinates
        features = self.batched_features.batched_tensor

        # Get the spatial shape.
        # If min_coords and max_coords are provided, assert that spatial_shape matches
        if spatial_shape is None and min_coords is None:
            # Get the min max coordinates
            coords = batch_indexed_coords[:, 1:]
            min_coords = coords.min(dim=0).values
            max_coords = coords.max(dim=0).values
            spatial_shape = max_coords - min_coords + 1
            # Shift the coordinates to the min_coords
            batch_indexed_coords[:, 1:] = batch_indexed_coords[:, 1:] - min_coords.to(device)
        elif min_coords is not None:
            # Assert either max_coords or spatial_shape is provided
            assert max_coords is not None or spatial_shape is not None
            # Convert min_coords to tensor
            min_coords = torch.tensor(min_coords, dtype=torch.int32)
            if max_coords is None:
                # convert spatial_shape to tensor
                spatial_shape = torch.tensor(spatial_shape, dtype=torch.int32)
                max_coords = min_coords + spatial_shape - 1
            else:  # both min_coords and max_coords are provided
                # Convert max_coords to tensor
                max_coords = torch.tensor(max_coords, dtype=torch.int32)
                assert len(min_coords) == len(max_coords) == self.num_spatial_dims
                spatial_shape = max_coords - min_coords + 1
            # Shift the coordinates to the min_coords and clip to the spatial_shape
            # Create a mask to identify coordinates within the spatial range
            mask = torch.ones(batch_indexed_coords.shape[0], dtype=torch.bool, device=device)
            for d in range(1, batch_indexed_coords.shape[1]):
                mask &= (batch_indexed_coords[:, d] >= min_coords[d - 1].item()) & (
                    batch_indexed_coords[:, d] < min_coords[d - 1].item() + spatial_shape[d - 1]
                )
            batch_indexed_coords = batch_indexed_coords[mask]
            features = features[mask]
        elif spatial_shape is not None and len(spatial_shape) == self.coordinate_tensor.shape[1]:
            # prepend a batch dimension
            pass
        else:
            raise ValueError(
                f"Provided spatial shape {spatial_shape} must be same length as the number of spatial dimensions {self.num_spatial_dims}."
            )

        # Create a dense tensor
        dense_tensor = torch.zeros(
            (self.batch_size, *spatial_shape, self.num_channels),
            dtype=self.batched_features.dtype,
            device=self.batched_features.device,
        )

        # Flatten view and scatter add
        flattened_indices = ravel_multi_index(
            batch_indexed_coords, (self.batch_size, *spatial_shape)
        )
        dense_tensor.flatten(0, -2)[flattened_indices] = features
        # Put the channel dimension in the specified position and move the rest of the dimensions contiguous
        dense_tensor = dense_tensor.moveaxis(-1, channel_dim)
        return dense_tensor

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

    @property
    def num_spatial_dims(self):
        return self.batched_coordinates.num_spatial_dims

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
