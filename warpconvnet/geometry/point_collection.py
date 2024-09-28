from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base_geometry import (
    BatchedCoordinates,
    BatchedFeatures,
    BatchedSpatialFeatures,
)
from warpconvnet.geometry.ops.neighbor_search_continuous import (
    ContinuousNeighborSearchArgs,
    NeighborSearchCache,
    NeighborSearchResult,
    neighbor_search,
)
from warpconvnet.geometry.ops.voxel_ops import (
    voxel_downsample_csr_mapping,
    voxel_downsample_random_indices,
)
from warpconvnet.geometry.ops.warp_sort import (
    POINT_ORDERING,
    sort_point_collection,
    sorting_permutation,
)
from warpconvnet.nn.functional.encodings import sinusoidal_encoding
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS, row_reduction

__all__ = ["BatchedContinuousCoordinates", "PointCollection"]


def random_downsample(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    sample_points: int,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B+1"]]:  # noqa: F821
    """
    Randomly downsample the coordinates to the specified number of points
    """
    num_points = offsets.diff()
    batch_size = len(num_points)
    # sample sample_points per batch. BxN
    sampled_indices = torch.floor(
        torch.rand(batch_size, sample_points) * num_points.view(-1, 1)
    ).to(torch.int32)
    # Add offsets
    sampled_indices = sampled_indices + offsets[:-1].view(-1, 1)
    sampled_indices = sampled_indices.view(-1)
    return sampled_indices, offsets


class BatchedContinuousCoordinates(BatchedCoordinates):
    def check(self):
        BatchedCoordinates.check(self)
        assert self.batched_tensor.shape[-1] == 3, "Coordinates must have 3 dimensions"

    def voxel_downsample(self, voxel_size: float):
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        perm, down_offsets = voxel_downsample_random_indices(
            coords=self.batched_tensor,
            voxel_size=voxel_size,
            offsets=self.offsets,
        )
        return self.__class__(tensors=self.batched_tensor[perm], offsets=down_offsets)

    def downsample(self, sample_points: int):
        """
        Downsample the coordinates to the specified number of points
        """
        sampled_indices, sample_offsets = random_downsample(self.offsets, sample_points)
        return self.__class__(
            batched_tensor=self.batched_tensor[sampled_indices], offsets=sample_offsets
        )

    def neighbors(
        self,
        search_args: ContinuousNeighborSearchArgs,
        query_coords: Optional["BatchedCoordinates"] = None,
    ) -> NeighborSearchResult:
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

    def sort(
        self,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
        voxel_size: Optional[float] = None,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smallest descritization and points in the same voxel will have random order.
        """
        # Warp uses int32 so only 10 bits per coordinate supported. Thus max 1024.
        assert self.device.type != "cpu", "Sorting is only supported on GPU"
        sorted_order = sorting_permutation(self.batched_tensor, self.offsets, ordering)
        return self.__class__(
            batched_tensor=self.batched_tensor[sorted_order],
            offsets=self.offsets,
        )


class PointCollection(BatchedSpatialFeatures):
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    def __init__(
        self,
        batched_coordinates: (
            List[Float[Tensor, "N 3"]] | Float[Tensor, "N 3"] | BatchedContinuousCoordinates
        ),  # noqa: F722,F821
        batched_features: (
            List[Float[Tensor, "N C"]] | Float[Tensor, "N C"] | BatchedFeatures
        ),  # noqa: F722,F821
        offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722,F821
        device: Optional[str] = None,
        **kwargs,
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
            batched_coordinates = BatchedContinuousCoordinates(batched_coordinates, device=device)
            batched_features = BatchedFeatures(batched_features, device=device)
        elif isinstance(batched_coordinates, Tensor):
            assert (
                isinstance(batched_features, Tensor) and offsets is not None
            ), "If coordinate is a tensor, features must be a tensor and offsets must be provided."
            batched_coordinates = BatchedContinuousCoordinates(
                batched_coordinates, offsets=offsets, device=device
            )
            batched_features = BatchedFeatures(batched_features, offsets=offsets, device=device)

        BatchedSpatialFeatures.__init__(
            self,
            batched_coordinates,
            batched_features,
            **kwargs,
        )

    def sort(
        self,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
        voxel_size: Optional[float] = None,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smallest discretization and points in the same voxel will have random order.
        """
        # Warp uses int32 so only 10 bits per coordinate supported. Thus max 1024.
        assert self.device.type != "cpu", "Sorting is only supported on GPU"
        sorted_coords, sorted_feats = sort_point_collection(
            coords=self.batched_coordinates.batched_tensor,
            features=self.batched_features.batched_tensor,
            ordering=ordering,
            offsets=self.batched_coordinates.offsets,
        )
        return self.__class__(
            batched_coordinates=BatchedContinuousCoordinates(
                sorted_coords, offsets=self.batched_coordinates.offsets
            ),
            batched_features=BatchedFeatures(sorted_feats, offsets=self.batched_features.offsets),
            **self.extra_attributes,
        )

    def voxel_downsample(
        self,
        voxel_size: float,
        reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR] = REDUCTIONS.RANDOM,
    ) -> "PointCollection":
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        extra_args = self.extra_attributes
        extra_args["voxel_size"] = voxel_size
        if reduction == REDUCTIONS.RANDOM:
            to_unique_indicies, unique_offsets = voxel_downsample_random_indices(
                batched_points=self.coordinate_tensor,
                offsets=self.offsets,
                voxel_size=voxel_size,
            )
            return self.__class__(
                batched_coordinates=BatchedContinuousCoordinates(
                    batched_tensor=self.coordinate_tensor[to_unique_indicies],
                    offsets=unique_offsets,
                ),
                batched_features=BatchedFeatures(
                    batched_tensor=self.feature_tensor[to_unique_indicies], offsets=unique_offsets
                ),
                **extra_args,
            )

        # perm, down_offsets, vox_inices, vox_offsets = voxel_downsample_csr_mapping(
        #     batched_points=self.coordinate_tensor,
        #     offsets=self.offsets,
        #     voxel_size=voxel_size,
        # )
        (
            batch_indexed_down_coords,
            unique_offsets,
            to_csr_indices,
            to_csr_offsets,
            to_unique,
        ) = voxel_downsample_csr_mapping(
            batched_points=self.coordinate_tensor,
            offsets=self.offsets,
            voxel_size=voxel_size,
        )

        neighbors = NeighborSearchResult(to_csr_indices, to_csr_offsets)
        down_features = row_reduction(
            self.feature_tensor,
            neighbors.neighbors_row_splits,
            reduction,
        )

        return self.__class__(
            batched_coordinates=BatchedContinuousCoordinates(
                batched_tensor=self.coordinates[to_unique.to_unique_indices],
                offsets=unique_offsets,
            ),
            batched_features=BatchedFeatures(batched_tensor=down_features, offsets=unique_offsets),
            **extra_args,
        )

    def downsample(self, sample_points: int) -> "PointCollection":
        """
        Downsample the coordinates to the specified number of points
        """
        sampled_indices, sample_offsets = random_downsample(self.offsets, sample_points)
        return self.__class__(
            batched_coordinates=BatchedContinuousCoordinates(
                batched_tensor=self.coordinate_tensor[sampled_indices], offsets=sample_offsets
            ),
            batched_features=BatchedFeatures(
                batched_tensor=self.feature_tensor[sampled_indices], offsets=sample_offsets
            ),
            **self.extra_attributes,
        )

    def neighbors(
        self,
        search_args: ContinuousNeighborSearchArgs,
        query_coords: Optional["BatchedCoordinates"] = None,
    ) -> NeighborSearchResult:
        """
        Returns CSR format neighbor indices
        """
        if query_coords is None:
            query_coords = self.batched_coordinates

        assert isinstance(
            query_coords, BatchedCoordinates
        ), "query_coords must be BatchedCoordinates"

        # cache the neighbor search result
        if self.cache is not None:
            neighbor_search_result = self.cache.get(
                search_args, self.offsets, query_coords.offsets
            )
            if neighbor_search_result is not None:
                return neighbor_search_result

        neighbor_search_result = neighbor_search(
            self.coordinate_tensor,
            self.offsets,
            query_coords.batched_tensor,
            query_coords.offsets,
            search_args,
        )
        if self.cache is None:
            self._extra_attributes["_cache"] = NeighborSearchCache()
        self.cache.put(search_args, self.offsets, query_coords.offsets, neighbor_search_result)
        return neighbor_search_result

    @property
    def voxel_size(self):
        return self._extra_attributes.get("voxel_size", None)

    @property
    def ordering(self):
        return self._extra_attributes.get("ordering", None)

    @classmethod
    def from_list_of_coordinates(
        cls,
        coordinates: List[Float[Tensor, "N 3"]],
        features: Optional[List[Float[Tensor, "N C"]]] = None,
        encoding_channels: Optional[int] = None,
        encoding_range: Optional[Tuple[float, float]] = None,
        encoding_dim: Optional[int] = -1,
    ):
        """
        Create a point collection from a list of coordinates.
        """
        # if the input is a tensor, expand it to a list of tensors
        if isinstance(coordinates, Tensor):
            coordinates = list(coordinates)  # this expands the tensor to a list of tensors

        if features is None:
            assert (
                encoding_range is not None
            ), "Encoding range must be provided if encoding channels are provided"
            features = [
                sinusoidal_encoding(coordinates, encoding_channels, encoding_range, encoding_dim)
                for coordinates in coordinates
            ]

        # Create BatchedContinuousCoordinates
        batched_coordinates = BatchedContinuousCoordinates(coordinates)
        # Create BatchedFeatures
        batched_features = BatchedFeatures(features)

        return cls(batched_coordinates, batched_features)

    def to_sparse(
        self,
        voxel_size: Optional[float] = None,
        reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR] = REDUCTIONS.RANDOM,
    ):
        """
        Convert the point collection to a spatially sparse tensor.
        """
        st = point_pool(
            self, reduction=reduction, downsample_voxel_size=voxel_size, return_type="sparse"
        )
        return st
