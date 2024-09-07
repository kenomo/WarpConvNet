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
        kernel_search_batch_size: int = 8,
    ):
        super(SpatiallySparseConv, self).__init__()
        kernel_size = ntuple(kernel_size, ndim=3)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.transposed = transposed
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
        )
