from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.functional.sparse_conv import (
    SPATIALLY_SPARSE_CONV_ALGO_MODE,
    STRIDED_CONV_MODE,
    spatially_sparse_conv,
)
from warpconvnet.utils.ntuple import ntuple


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
        kernel_matmul_batch_size: int = 2,
        num_spatial_dims: Optional[int] = 3,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
    ):
        super(SpatiallySparseConv, self).__init__()
        kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.transposed = transposed
        self.generative = generative
        self.kernel_search_batch_size = kernel_search_batch_size
        self.kernel_matmul_batch_size = kernel_matmul_batch_size
        self.conv_algo = conv_algo
        self.weight = nn.Parameter(torch.randn(np.prod(kernel_size), in_channels, out_channels))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

        self.stride_mode = stride_mode

    def __repr__(self):
        # return class name and parameters that are not default
        out_str = f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != 1:
            out_str += f", stride={self.stride}"
        if self.dilation != 1:
            out_str += f", dilation={self.dilation}"
        if self.transposed:
            out_str += f", transposed={self.transposed}"
        if self.generative:
            out_str += f", generative={self.generative}"
        out_str += ")"
        return out_str

    def forward(
        self,
        input_sparse_tensor: SpatiallySparseTensor,
        output_spatially_sparse_tensor: Optional[SpatiallySparseTensor] = None,
    ):
        return spatially_sparse_conv(
            input_sparse_tensor=input_sparse_tensor,
            weight=self.weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            kernel_dilation=self.dilation,
            bias=self.bias,
            kernel_search_batch_size=self.kernel_search_batch_size,
            kernel_matmul_batch_size=self.kernel_matmul_batch_size,
            output_spatially_sparse_tensor=output_spatially_sparse_tensor,
            transposed=self.transposed,
            generative=self.generative,
            conv_algo=self.conv_algo,
            stride_mode=self.stride_mode,
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
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
        kernel_matmul_batch_size: int = 2,
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
            stride_mode=stride_mode,
            conv_algo=conv_algo,
            kernel_matmul_batch_size=kernel_matmul_batch_size,
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
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        conv_algo: SPATIALLY_SPARSE_CONV_ALGO_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
        kernel_matmul_batch_size: int = 2,
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
            stride_mode=stride_mode,
            conv_algo=conv_algo,
            kernel_matmul_batch_size=kernel_matmul_batch_size,
        )
