from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.nn.functional.sparse_conv import spatially_sparse_conv
from warp.convnet.utils.ntuple import ntuple


class SpatiallySparseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = True,
        transposed: bool = False,
        generative: bool = False,
        kernel_search_batch_size: int = 8,
        num_spatial_dims: Optional[int] = 3,
    ):
        super(SpatiallySparseConv, self).__init__()
        kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.transposed = transposed
        self.generative = generative
        self.kernel_search_batch_size = kernel_search_batch_size
        self.weight = nn.Parameter(torch.randn(np.prod(kernel_size), in_channels, out_channels))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(
        self,
        input_sparse_tensor: SpatiallySparseTensor,
        output_spatially_sparse_tensor: Optional[SpatiallySparseTensor] = None,
    ):
        return spatially_sparse_conv(
            input_sparse_tensor,
            self.weight,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.bias,
            kernel_search_batch_size=self.kernel_search_batch_size,
            output_spatially_sparse_tensor=output_spatially_sparse_tensor,
            transposed=self.transposed,
            generative=self.generative,
        )


class SparseConv2d(SpatiallySparseConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        transposed=False,
        generative: bool = False,
        kernel_search_batch_size=8,
    ):
        super(SparseConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transposed=transposed,
            generative=generative,
            kernel_search_batch_size=kernel_search_batch_size,
            num_spatial_dims=2,
        )


class SparseConv3d(SpatiallySparseConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        transposed=False,
        generative: bool = False,
        kernel_search_batch_size=8,
    ):
        super(SparseConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transposed=transposed,
            generative=generative,
            kernel_search_batch_size=kernel_search_batch_size,
            num_spatial_dims=3,
        )
