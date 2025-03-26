# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union
from jaxtyping import Float, Int

import torch
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.coords.real import RealCoords
from warpconvnet.geometry.coords.search.hashmap import VectorHashTable
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, morton_code
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures
from warpconvnet.geometry.coords.ops.voxel import voxel_downsample_random_indices
from warpconvnet.geometry.features.ops.convert import to_batched_features
from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.utils.ravel import ravel_multi_index


class Voxels(Geometry):
    def __init__(
        self,
        batched_coordinates: List[Float[Tensor, "N 3"]] | Float[Tensor, "N 3"] | IntCoords,
        batched_features: (
            List[Float[Tensor, "N C"]]
            | Float[Tensor, "N C"]
            | Float[Tensor, "B M C"]
            | CatFeatures
            | PadFeatures
        ),
        offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722,F821
        device: Optional[str] = None,
        **kwargs,
    ):
        # extract tensor_stride/stride from kwargs
        tensor_stride = kwargs.pop("tensor_stride", None) or kwargs.pop("stride", None)

        if isinstance(batched_coordinates, list):
            assert isinstance(
                batched_features, list
            ), "If coords is a list, features must be a list too."
            assert len(batched_coordinates) == len(batched_features)
            # Assert all elements in coords and features have same length
            assert all(len(c) == len(f) for c, f in zip(batched_coordinates, batched_features))
            batched_coordinates = IntCoords(
                batched_coordinates, device=device, tensor_stride=tensor_stride
            )
        elif isinstance(batched_coordinates, Tensor):
            assert (
                isinstance(batched_features, Tensor) and offsets is not None
            ), "If coordinate is a tensor, features must be a tensor and offsets must be provided."
            batched_coordinates = IntCoords(
                batched_coordinates,
                offsets=offsets,
                device=device,
                tensor_stride=tensor_stride,
            )
        else:
            # Input is a BatchedDiscreteCoordinates
            if tensor_stride is not None:
                batched_coordinates.set_tensor_stride(tensor_stride)

        if isinstance(batched_features, list):
            batched_features = CatFeatures(batched_features, device=device)
        elif isinstance(batched_features, Tensor):
            batched_features = to_batched_features(
                batched_features, batched_coordinates.offsets, device=device
            )

        Geometry.__init__(self, batched_coordinates, batched_features, **kwargs)

    @classmethod
    def from_dense(
        cls,
        dense_tensor: Float[Tensor, "B C H W"] | Float[Tensor, "B C H W D"],
        dense_tensor_channel_dim: int = 1,
        target_spatial_sparse_tensor: Optional["Voxels"] = None,
        dense_max_coords: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        # Move channel dimension to the end
        if dense_tensor_channel_dim != -1 or dense_tensor.ndim != dense_tensor_channel_dim + 1:
            dense_tensor = dense_tensor.moveaxis(dense_tensor_channel_dim, -1)
        spatial_shape = dense_tensor.shape[1:-1]
        batched_spatial_shape = dense_tensor.shape[:-1]

        # Flatten the spatial dimensions
        flattened_tensor = dense_tensor.flatten(0, -2)

        if target_spatial_sparse_tensor is None:
            # abs sum all elements in the tensor
            abs_sum = torch.abs(dense_tensor).sum(dim=-1, keepdim=False)
            # Find all non-zero elements. Expected to be sorted.
            non_zero_inds = torch.nonzero(abs_sum).int()
            # Convert multi-dimensional indices to flattened indices
            flattened_indices = ravel_multi_index(non_zero_inds, batched_spatial_shape)

            # Use index_select to get the features
            non_zero_feats = torch.index_select(flattened_tensor, 0, flattened_indices)

            offsets = offsets_from_batch_index(non_zero_inds[:, 0])
            return cls(
                batched_coordinates=IntCoords(non_zero_inds[:, 1:], offsets=offsets),
                batched_features=CatFeatures(non_zero_feats, offsets=offsets),
                **kwargs,
            )
        else:
            assert target_spatial_sparse_tensor.num_spatial_dims == len(spatial_shape)
            assert target_spatial_sparse_tensor.batch_size == batched_spatial_shape[0]
            # Use the provided spatial sparse tensor's coordinate only
            batch_indexed_coords = target_spatial_sparse_tensor.batch_indexed_coordinates
            # subtract the min_coords
            min_coords = target_spatial_sparse_tensor.coordinate_tensor.min(dim=0).values.view(
                1, -1
            )
            batch_indexed_coords[:, 1:] = batch_indexed_coords[:, 1:] - min_coords
            if dense_max_coords is not None:
                invalid = (batch_indexed_coords[:, 1:] > dense_max_coords).any(dim=1)
                batch_indexed_coords[invalid] = 0
            else:
                sparse_max_coords = batch_indexed_coords[:, 1:].max(dim=0).values
                # This assumes the max_coords are already aligned with the spatial_shape.
                assert torch.all(
                    sparse_max_coords.cpu() < torch.tensor(spatial_shape)
                ), "Max coords must be aligned with the spatial shape."
            # Ravel the coordinates. This assumes the max_coords are already aligned with the spatial_shape.
            flattened_indices = ravel_multi_index(batch_indexed_coords, batched_spatial_shape)
            # use index_select to get the features
            non_zero_feats = torch.index_select(flattened_tensor, 0, flattened_indices)
            if dense_max_coords is not None:
                non_zero_feats[invalid] = 0
            return target_spatial_sparse_tensor.replace(batched_features=non_zero_feats)

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

        if isinstance(spatial_shape, Tensor):
            spatial_shape = spatial_shape.tolist()

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

        if channel_dim != -1:
            # Put the channel dimension in the specified position and move the rest of the dimensions contiguous
            dense_tensor = dense_tensor.moveaxis(-1, channel_dim)
        return dense_tensor

    def to_point(self, voxel_size: Optional[float] = None) -> "Points":  # noqa: F821
        if voxel_size is None:
            assert (
                self.voxel_size is not None
            ), "Voxel size must be provided or the object must have been initialized with a voxel size to convert to point."
            voxel_size = self.voxel_size

        # tensor stride
        if self.tensor_stride is not None:
            tensor_stride = self.tensor_stride
            # multiply voxel_size by tensor_stride
            voxel_size = torch.Tensor([[voxel_size * s for s in tensor_stride]]).to(self.device)

        from warpconvnet.geometry.types.points import Points

        batched_points = RealCoords(self.coordinate_tensor * voxel_size, self.offsets)
        return Points(batched_points, self.batched_features)

    def sort(self, ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER) -> "Voxels":
        if ordering == self.ordering:
            return self

        assert isinstance(
            self.batched_features, CatFeatures
        ), "Features must be a CatBatchedFeatures to sort."

        code, perm = morton_code(self.coordinate_tensor, self.offsets, ordering)  # noqa: F821
        kwargs = self.extra_attributes.copy()
        kwargs["ordering"] = ordering
        kwargs["code"] = code[perm]
        return self.__class__(
            batched_coordinates=IntCoords(self.coordinate_tensor[perm], self.offsets),
            batched_features=CatFeatures(self.feature_tensor[perm], self.offsets),
            **kwargs,
        )

    def unique(self) -> "Voxels":
        unique_indices, batch_offsets = voxel_downsample_random_indices(
            self.coordinate_tensor,
            self.offsets,
        )
        coords = IntCoords(self.coordinate_tensor[unique_indices], batch_offsets)
        feats = CatFeatures(self.feature_tensor[unique_indices], batch_offsets)
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
        return self.tensor_stride

    @property
    def tensor_stride(self):
        return self.batched_coordinates.tensor_stride

    def set_tensor_stride(self, tensor_stride: Union[int, Tuple[int, ...]]):
        self.batched_coordinates.set_tensor_stride(tensor_stride)

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        return self.batched_coordinates.batch_indexed_coordinates
