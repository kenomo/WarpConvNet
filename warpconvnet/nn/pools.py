from typing import Literal, Tuple, Union

import torch
import torch.nn as nn

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.functional.global_pool import global_pool
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.functional.point_unpool import point_unpool
from warpconvnet.nn.functional.sparse_pool import sparse_reduce, sparse_unpool
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS


class SparsePool(BaseSpatialModule):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        reduce: Literal["max", "min", "mean", "sum", "random"] = "max",
        out_code_backend: str = "morton",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduce = reduce
        self.out_code_backend = out_code_backend

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, reduce={self.reduce})"

    def forward(self, st: Voxels):
        return sparse_reduce(
            st,
            self.kernel_size,
            self.stride,
            self.reduce,
            out_code_backend=self.out_code_backend,
        )


class SparseMaxPool(SparsePool):
    def __init__(self, kernel_size: int, stride: int, out_code_backend: str = "morton"):
        super().__init__(kernel_size, stride, "max")


class SparseMinPool(SparsePool):
    def __init__(self, kernel_size: int, stride: int, out_code_backend: str = "morton"):
        super().__init__(kernel_size, stride, "min")


class GlobalPool(BaseSpatialModule):
    def __init__(self, reduce: Literal["min", "max", "mean", "sum"] = "max"):
        super().__init__()
        self.reduce = reduce

    def forward(self, x: Geometry):
        return global_pool(x, self.reduce)


class SparseUnpool(BaseSpatialModule):
    def __init__(self, kernel_size: int, stride: int, concat_unpooled_st: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.concat_unpooled_st = concat_unpooled_st

    def forward(self, st: Voxels, unpooled_st: Voxels):
        return sparse_unpool(
            st,
            unpooled_st,
            self.kernel_size,
            self.stride,
            self.concat_unpooled_st,
        )


class PointToSparseWrapper(BaseSpatialModule):
    """
    A module that pools points to a spatially sparse tensor given a voxel size and pass it to the inner module.

    The output of the inner module is then converted back to a point cloud.
    """

    def __init__(
        self,
        inner_module: BaseSpatialModule,
        voxel_size: float,
        reduction: Union[REDUCTIONS, REDUCTION_TYPES_STR] = REDUCTIONS.MEAN,
        unique_method: Literal["morton", "ravel", "torch"] = "torch",
        concat_unpooled_pc: bool = True,
    ):
        super().__init__()
        self.inner_module = inner_module
        self.voxel_size = voxel_size
        self.reduction = reduction
        self.concat_unpooled_pc = concat_unpooled_pc
        self.unique_method = unique_method

    def forward(self, pc: Points):
        st, to_unique = point_pool(
            pc,
            reduction=self.reduction,
            downsample_voxel_size=self.voxel_size,
            return_type="sparse",
            return_to_unique=True,
            unique_method=self.unique_method,
        )
        out_st = self.inner_module(st)
        unpooled_pc = point_unpool(
            out_st.to_point(self.voxel_size),
            pc,
            concat_unpooled_pc=self.concat_unpooled_pc,
            to_unique=to_unique,
        )
        return unpooled_pc
