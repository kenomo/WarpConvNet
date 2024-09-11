from typing import Literal

import torch.nn as nn

from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.functional.sparse_pool import sparse_reduce


class SparsePool(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        reduce: Literal["max", "min", "mean", "sum", "random"] = "max",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduce = reduce

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, reduce={self.reduce})"

    def forward(self, st: SpatiallySparseTensor):
        return sparse_reduce(st, self.kernel_size, self.stride, self.reduce)


class SparseMaxPool(SparsePool):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__(kernel_size, stride, "max")


class SparseMinPool(SparsePool):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__(kernel_size, stride, "min")
