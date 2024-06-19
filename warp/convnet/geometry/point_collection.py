from typing import List, Optional

from jaxtyping import Float
from torch import Tensor

from warp.convnet.geometry.base_geometry import (
    BatchedCoordinates,
    BatchedFeatures,
    BatchedSpatialFeatures,
)
from warp.convnet.geometry.ops.neighbor_search import (
    NeighborSearchArgs,
    NeighborSearchResult,
    neighbor_search,
)
from warp.convnet.geometry.ops.point_pool import (
    FEATURE_POOLING_MODE,
    FeaturePoolingArgs,
    pool_features,
)
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample
from warp.convnet.geometry.ops.warp_sort import POINT_ORDERING, sort_point_collection


class BatchedContinuousCoordinates(BatchedCoordinates):
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


class PointCollection(BatchedSpatialFeatures):
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    def __init__(
        self,
        batched_coordinates: List[Float[Tensor, "N 3"]]
        | BatchedContinuousCoordinates,  # noqa: F722,F821
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
            batched_coordinates = BatchedContinuousCoordinates(batched_coordinates)
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
            batched_coordinates=BatchedContinuousCoordinates(
                sorted_coords, offsets=self.batched_coordinates.offsets
            ),
            batched_features=BatchedFeatures(sorted_feats, offsets=self.batched_features.offsets),
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
                coords=BatchedContinuousCoordinates(
                    batched_tensor=self.coords[perm], offsets=down_offsets
                ),
                features=BatchedFeatures(batched_tensor=self.features[perm], offsets=down_offsets),
            )

        neighbors = NeighborSearchResult(vox_inices, vox_offsets)
        down_coords = self.coords[perm]
        down_features = pool_features(
            in_feats=self.features,
            down_coords=down_coords,
            neighbors=neighbors,
            pooling_args=pooling_args,
        )

        return self.__class__(
            batched_coordinates=BatchedContinuousCoordinates(
                batched_tensor=down_coords, offsets=down_offsets
            ),
            batched_features=BatchedFeatures(batched_tensor=down_features, offsets=down_offsets),
        )
