from typing import Literal, Optional, Tuple, Union

import torch.nn as nn

from warpconvnet.geometry.base_geometry import BatchedSpatialFeatures
from warpconvnet.geometry.ops.neighbor_search_continuous import NeighborSearchResult
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.functional.point_unpool import FEATURE_UNPOOLING_MODE, point_unpool
from warpconvnet.ops.reductions import REDUCTIONS

__all__ = ["PointPoolBase", "PointMaxPool", "PointAvgPool", "PointSumPool", "PointUnpool"]


class PointPoolBase(BaseSpatialModule):
    def __init__(
        self,
        reduction: Union[str, REDUCTIONS] = REDUCTIONS.MAX,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__()
        if isinstance(reduction, str):
            reduction = REDUCTIONS(reduction)
        self.reduction = reduction
        self.downsample_max_num_points = downsample_max_num_points
        self.downsample_voxel_size = downsample_voxel_size
        self.return_type = return_type
        self.return_neighbor_search_result = return_neighbor_search_result

    def forward(
        self, pc: PointCollection
    ) -> Union[BatchedSpatialFeatures, Tuple[BatchedSpatialFeatures, NeighborSearchResult]]:
        return point_pool(
            pc=pc,
            reduction=self.reduction,
            downsample_max_num_points=self.downsample_max_num_points,
            downsample_voxel_size=self.downsample_voxel_size,
            return_type=self.return_type,
            return_neighbor_search_result=self.return_neighbor_search_result,
        )


class PointMaxPool(PointPoolBase):
    def __init__(
        self,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__(
            reduction=REDUCTIONS.MAX,
            downsample_max_num_points=downsample_max_num_points,
            downsample_voxel_size=downsample_voxel_size,
            return_type=return_type,
            return_neighbor_search_result=return_neighbor_search_result,
        )


class PointAvgPool(PointPoolBase):
    def __init__(
        self,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__(
            reduction=REDUCTIONS.AVG,
            downsample_max_num_points=downsample_max_num_points,
            downsample_voxel_size=downsample_voxel_size,
            return_type=return_type,
            return_neighbor_search_result=return_neighbor_search_result,
        )


class PointSumPool(PointPoolBase):
    def __init__(
        self,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__(
            reduction=REDUCTIONS.SUM,
            downsample_max_num_points=downsample_max_num_points,
            downsample_voxel_size=downsample_voxel_size,
            return_type=return_type,
            return_neighbor_search_result=return_neighbor_search_result,
        )


class PointUnpool(BaseSpatialModule):
    def __init__(
        self,
        unpooling_mode: Union[str, FEATURE_UNPOOLING_MODE] = FEATURE_UNPOOLING_MODE.REPEAT,
        concat_unpooled_pc: bool = False,
    ):
        super().__init__()
        if isinstance(unpooling_mode, str):
            unpooling_mode = FEATURE_UNPOOLING_MODE(unpooling_mode)
        self.unpooling_mode = unpooling_mode
        self.concat_unpooled_pc = concat_unpooled_pc

    def forward(self, pooled_pc: PointCollection, unpooled_pc: PointCollection):
        return point_unpool(
            pooled_pc=pooled_pc,
            unpooled_pc=unpooled_pc,
            unpooling_mode=self.unpooling_mode,
            concat_unpooled_pc=self.concat_unpooled_pc,
        )
