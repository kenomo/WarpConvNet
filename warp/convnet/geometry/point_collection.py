from typing import Optional, List
from enum import Enum

import warp as wp


class POINT_ORDERING(Enum):
    RANDOM = Enum.auto()
    Z_ORDER = Enum.auto()


class NeighborSearchReturn:
    pass


class PointCollection:
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    dimension: int = 3
    batch_size: int
    offsets: wp.array1d
    batched_coordinates: wp.Coords  # TODO support for amp types
    batched_features: wp.vec3d  # TODO support for amp types
    _hash_grids: Optional[List[wp.HashGrid]]
    _sorted: bool

    def sort(
        self,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
        voxel_size: Optional[float] = None,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smalles descritization and points in the same voxel will have random order.
        """
        pass

    def neighbors(self, radius: float) -> NeighborSearchReturn:
        """
        Returns CSR format neighbor indices
        """
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
