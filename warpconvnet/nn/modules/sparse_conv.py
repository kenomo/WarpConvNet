import math
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.init import calculate_gain

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.sparse_conv import (
    SPATIALLY_SPARSE_CONV_ALGO_MODE,
    STRIDED_CONV_MODE,
    spatially_sparse_conv,
)
from warpconvnet.utils.ntuple import ntuple


class SpatiallySparseConv(BaseSpatialModule):
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
        out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "unique",
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
        self.num_spatial_dims = num_spatial_dims
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
        self.out_code_backend = out_code_backend
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(torch.randn(np.prod(kernel_size), in_channels, out_channels))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

        self.reset_parameters()
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

    def _calculate_fan_in_and_fan_out(self):
        receptive_field_size = np.prod(self.kernel_size)
        fan_in = self.in_channels * receptive_field_size
        fan_out = self.out_channels * receptive_field_size
        return fan_in, fan_out

    def _calculate_correct_fan(self, mode: Literal["fan_in", "fan_out"]):
        mode = mode.lower()
        assert mode in ["fan_in", "fan_out"]

        fan_in, fan_out = self._calculate_fan_in_and_fan_out()
        return fan_in if mode == "fan_in" else fan_out

    def _custom_kaiming_uniform_(self, tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
        fan = self._calculate_correct_fan(mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(self.num_spatial_dims) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def reset_parameters(self):
        self._custom_kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
            mode="fan_out" if self.transposed else "fan_in",
        )

        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        input_sparse_tensor: Voxels,
        output_spatially_sparse_tensor: Optional[Voxels] = None,
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
            out_code_backend=self.out_code_backend,
            compute_dtype=self.compute_dtype,
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
        out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "unique",
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
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
            out_code_backend=out_code_backend,
            compute_dtype=compute_dtype,
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
        out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "unique",
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
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
            out_code_backend=out_code_backend,
            compute_dtype=compute_dtype,
        )
